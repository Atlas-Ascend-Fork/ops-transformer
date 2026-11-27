# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from enum import IntEnum
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch_npu
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from torch.library import impl

QFA_METADATA_OP_NAME = "quant_flash_attn_metadata"
METADATA_STRIDE = (
    16  # 每核 metadata 字段数, 与 AICPU/kernel 侧 METADATA_STRIDE 同名同值
)


class QuantMode(IntEnum):
    """quant_mode 枚举：对外字符串/int 经由本枚举映射为传给算子侧的 int。"""

    A8C8_QKV_HIF8_PER_TENSOR_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 0
    A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 1
    A8C8_QKV_MXFP8_P_MXFP8_SOFTMAX_FP32 = 2
    A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16 = 3
    A8C8_QKV_MXFP8_P_MXFP8_SOFTMAX_FP16 = 4
    A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16 = 5
    A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32 = 6
    A4C4_QKV_HIF4_P_HIF4_LEVEL1_SOFTMAX_FP16 = 7
    A4C4_QKV_HIF4_P_HIF4_LEVEL2_SOFTMAX_FP16 = 8
    A4C4_QKV_HIF4_P_HIF4_LEVEL3_SOFTMAX_FP16 = 9
    A4C4_QKV_HIF4_P_FP8_E4M3_SOFTMAX_FP16 = 10
    A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR_SOFTMAX_FP16 = 12
    A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR = 12


_INT8_QUANT_MODE = int(
    QuantMode.A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR_SOFTMAX_FP16
)


class MaskMode(IntEnum):
    """mask_mode 枚举：对外字符串/int 经由本枚举映射为传给算子侧的 int。"""

    NO_MASK = 0
    CAUSAL = 3
    SLIDING_WINDOW = 4


def _resolve_quant_mode(quant_mode: Union[str, int, "QuantMode"]) -> int:
    """对外 str/int/IntEnum quant_mode 统一为传给算子侧的 int；校验取值合法性。"""
    if isinstance(quant_mode, str):
        try:
            return int(QuantMode[quant_mode.strip().upper()])
        except KeyError as exc:
            valid = ", ".join(m.name.lower() for m in QuantMode)
            raise ValueError(
                f"quant_mode should be one of [{valid}], but got {quant_mode!r}"
            ) from exc
    return int(QuantMode(quant_mode))


def _resolve_mask_mode(mask_mode: Union[str, int, "MaskMode", None]) -> int:
    """对外 str/int/IntEnum/None mask_mode 统一为传给算子侧的 int；None 视为默认 0。"""
    if mask_mode is None:
        return int(MaskMode.NO_MASK)
    if isinstance(mask_mode, str):
        try:
            return int(MaskMode[mask_mode.strip().upper()])
        except KeyError as exc:
            valid = ", ".join(m.name.lower() for m in MaskMode)
            raise ValueError(
                f"mask_mode should be one of [{valid}], but got {mask_mode!r}"
            ) from exc
    return int(MaskMode(mask_mode))


def _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q):
    if seqused_q is not None:
        return seqused_q.size(0)
    elif cu_seqlens_q is not None and cu_seqlens_q.size(0) > 0:
        return cu_seqlens_q.size(0) - 1
    elif batch_size is not None:
        return batch_size
    return 0


def _calculate_max_schedule_size(batch_size, num_heads_q):
    """dim0 按 sectionNum 最坏值(batch*num_heads_q)动态计算, 开启 section 后
    FA/FD 调度区需求随 section 数线性增长, 固定值无法覆盖; 对齐 4096 便于对拍。
    (2, dim0) 布局中第二行即反向 FAG 区, grad 按 shape.GetDim(1)=dim0 推导其偏移。"""
    align_size = 4096
    head_size = METADATA_STRIDE  # head 区占 1 个 stride
    # batch_size 为 -1/None/0(未知, 由 AICPU 从 varlen tensor 推断)时无法预估 section 数,
    # 按 1 兜底: sn 恒为 1 时需求为 16+(aic+aiv)*16, 4096 预算已覆盖
    batch_size = batch_size if batch_size and batch_size > 0 else 1
    aic_num, aiv_num = _get_core_nums()
    fa_size = aic_num * METADATA_STRIDE * batch_size * num_heads_q
    fd_size = aiv_num * METADATA_STRIDE * batch_size * num_heads_q

    schedule_size = head_size + fa_size + fd_size
    return ((schedule_size + align_size - 1) // align_size) * align_size


def _get_core_nums():
    """惰性获取设备核数(理论上限, 与 AICPU 调度/kernel grid 口径一致);
    仅无 NPU 环境(meta/fallback)返回默认值。真机上查询失败时向上抛出,
    避免静默用默认核数导致 metadata 欠分配越界。"""
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return 36, 72
    props = npu.get_device_properties()
    return props.cube_core_num, props.vector_core_num


# 各 layout 期望的 Q/K/V tensor 维度数 (N2TGD 是 descale 专用 layout, 不在此表)
_LAYOUT_EXPECTED_NDIM = {
    "TND": 3,
    "NTD": 3,
    "BSND": 4,
    "BNSD": 4,
    "PA_NZ": 5,
    "PA_BNBD": 4,
    "PA_BBND": 4,
}
_OUT_LAYOUT_EXPECTED_NDIM = {
    "TND": 3,
    "BSND": 4,
    "BNSD": 4,
}

# layout 一致性校验表, 与 op_host/checkers/quant_checker.cpp 中 QFA_LAYOUT_CONSTRAINT_TABLE 的
# requireLayoutConsistent 字段保持一致。仅记录需要 layout_q/kv/out 三者一致的 quant_mode。
_LAYOUT_REQUIRE_CONSISTENT = {
    # HIF8: layout_q, layout_kv, layout_out 三者必须相同
    int(QuantMode.A8C8_QKV_HIF8_PER_TENSOR_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32),
}

# layout_q 与 layout_out 必须一致的 quant_mode (但 layout_kv 可以不同)
_LAYOUT_REQUIRE_Q_OUT_CONSISTENT = {
    # MxFP8: layout_q=TND, layout_out=TND, 但 layout_kv 可为 PA 场景
    int(QuantMode.A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32),
}


def _check_layout_constraint(quant_mode, layout_q, layout_kv, layout_out):
    """校验 layout 一致性
    - HIF8: layout_q、layout_kv、layout_out 三者必须相同
    - MxFP8: layout_q 与 layout_out 必须相同(layout_kv 可不同, 支持 PA 场景)"""
    if quant_mode in _LAYOUT_REQUIRE_CONSISTENT:
        torch._check(
            layout_q == layout_kv == layout_out,
            lambda: f"When quant_mode is {quant_mode}, layout_q, layout_kv and layout_out must be the same, "
            f"but got layout_q={layout_q!r}, layout_kv={layout_kv!r}, layout_out={layout_out!r}",
        )
    elif quant_mode in _LAYOUT_REQUIRE_Q_OUT_CONSISTENT:
        torch._check(
            layout_q == layout_out,
            lambda: f"When quant_mode is {quant_mode}, layout_q and layout_out must be the same, "
            f"but got layout_q={layout_q!r}, layout_out={layout_out!r}",
        )


def _get_output_shape_sizes(q, v, layout_q, layout_kv, quant_mode=None):
    """提取决定输出 shape 的维度: q 的 B/T、S、N 维度(D 恒为 q 最后一维, 不提取)
    以及 v 的 D 维度; t_size/b_size/s_size 按各自 layout 有效, 不适用时为 None。
    调用前需保证 q/v 维度数已通过 _LAYOUT_EXPECTED_NDIM 校验。"""
    if layout_q == "TND":
        t_size, b_size, s_size, n_size = q.size(0), None, None, q.size(1)
    elif layout_q == "NTD":
        # GQA FP8 (quant_mode=6): q 为 [N, T, D], 输出/lse 与 TND 同形 (T/N 主序不同)
        t_size, b_size, s_size, n_size = q.size(1), None, None, q.size(0)
    elif layout_q == "BSND":
        t_size, b_size, s_size, n_size = None, q.size(0), q.size(1), q.size(2)
    else:
        t_size, b_size, s_size, n_size = None, q.size(0), q.size(2), q.size(1)
    if layout_kv == "PA_NZ":
        # PA_NZ stores D in 32-element NZ blocks and may physically pad D=72 to
        # 96. The output remains logical D, so take it from Q rather than V.
        d_size = (
            q.size(-1)
            if quant_mode
            == QuantMode.A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR
            else v.size(2) * v.size(4)
        )
    elif layout_kv == "TND":
        d_size = v.size(2)
    else:
        d_size = v.size(3)
    return t_size, b_size, s_size, n_size, d_size


def _check_int8(condition, message):
    torch._check(condition, lambda: message)


def _int8_metadata_row_size(batch, heads_kv):
    """INT32 words in the active row; the second row is reserved as on A5."""
    return 32 + 8 * batch + 2 * 128 * batch * heads_kv + 512


def _metadata_row_size(batch, heads_q, heads_kv, quant_mode, layout_kv):
    if quant_mode == _INT8_QUANT_MODE and layout_kv not in (
        "PA_BNBD",
        "PA_BBND",
        "PA_NZ",
    ):
        return _int8_metadata_row_size(batch, heads_kv)
    return _calculate_max_schedule_size(batch, heads_q)


def _check_int8_metadata_contract(
    heads_q,
    heads_kv,
    dim,
    dim_v,
    batch,
    max_q,
    max_kv,
    mask,
    left,
    right,
    layout_q,
    layout_kv,
    layout_out,
    cu_q,
    cu_kv,
    used_q,
    used_kv,
    grad,
):
    check = _check_int8
    check(
        heads_q > 0
        and heads_kv > 0
        and heads_q % heads_kv == 0
        and 1 <= dim <= 256
        and (dim_v is None or dim_v == dim)
        and not grad,
        "INT8 metadata requires divisible heads, matching D in [1,256] and dense prefill",
    )
    check(
        _resolve_mask_mode(mask) in (0, 3)
        and all(x is None or -1 <= x <= 2147483647 for x in (left, right)),
        "INT8 metadata requires mask 0/3 and windows in [-1, INT32_MAX]",
    )
    check(
        all(x in ("BNSD", "BSND", "TND") for x in (layout_q, layout_kv, layout_out))
        and (layout_q == "TND") == (layout_kv == "TND") == (layout_out == "TND"),
        "INT8 metadata requires dense BNSD/BSND or all-TND layouts",
    )
    check(
        all(x is None or -1 <= x <= 2147483647 for x in (max_q, max_kv)),
        "INT8 metadata max lengths must be -1 or nonnegative INT32 values",
    )
    if layout_q == "TND":
        check(
            isinstance(cu_q, torch.Tensor) and isinstance(cu_kv, torch.Tensor),
            "TND metadata requires cu_seqlens_q and cu_seqlens_kv",
        )
        check(
            cu_q.dtype == cu_kv.dtype == torch.int32
            and cu_q.dim() == cu_kv.dim() == 1
            and cu_q.numel() == cu_kv.numel()
            and cu_q.numel() >= 2
            and cu_q.is_contiguous()
            and cu_kv.is_contiguous()
            and cu_q.device == cu_kv.device,
            "TND metadata requires contiguous INT32 cumulative lengths [B+1] on the same device",
        )
        check(batch is None, "TND metadata infers batch from cu_seqlens")
        batch = cu_q.numel() - 1
    else:
        check(
            batch is not None
            and batch > 0
            and cu_q is None
            and cu_kv is None
            and (used_q is not None or (max_q is not None and max_q > 0))
            and (used_kv is not None or (max_kv is not None and max_kv > 0)),
            "BNSD/BSND metadata requires positive batch, max length or seqused on each side, and no cu_seqlens",
        )
    _check_int8_used(used_q, used_kv, batch, cu_q.device if cu_q is not None else None)
    return batch


def _check_int8_used(used_q, used_kv, batch, device=None):
    for name, used in (("seqused_q", used_q), ("seqused_kv", used_kv)):
        if used is not None:
            _check_int8(
                isinstance(used, torch.Tensor)
                and used.dtype == torch.int32
                and used.dim() == 1
                and used.numel() == batch
                and used.is_contiguous()
                and (device is None or used.device == device),
                f"INT8 {name} must be a contiguous INT32 [B] tensor on the input device",
            )
    if used_q is not None and used_kv is not None:
        _check_int8(
            used_q.device == used_kv.device,
            "INT8 seqused tensors must use the same device",
        )


def _check_int8_contract(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    layout_q,
    layout_kv,
    layout_out,
    mask_mode,
    cu_q,
    cu_k,
    metadata,
    win_left,
    win_right,
    used_q,
    used_kv,
    unsupported,
):
    """INT8 scales follow their input's token/head order (without D)."""
    torch._check(q is not None, lambda: "q must not be None")
    torch._check(k is not None, lambda: "k must not be None")
    torch._check(v is not None, lambda: "v must not be None")
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    check = _check_int8
    check(
        all(
            layout in ("BNSD", "BSND", "TND")
            for layout in (layout_q, layout_kv, layout_out)
        ),
        "Full INT8 GQA requires BNSD, BSND or TND",
    )
    packed = layout_q == "TND"
    check(
        (layout_kv == "TND") == packed and (layout_out == "TND") == packed,
        "Full INT8 GQA requires all layouts TND for packed sequences",
    )
    check(
        int(mask_mode) in (0, 3)
        and -1 <= win_left <= 2147483647
        and -1 <= win_right <= 2147483647
        and all(x is None for x in unsupported),
        "Full INT8 GQA requires mask 0/3, valid windows, no cache, explicit mask or sinks",
    )
    rank = 3 if packed else 4
    check(
        q.dim() == k.dim() == v.dim() == rank
        and q.dtype == k.dtype == v.dtype == torch.int8,
        "Full INT8 GQA requires layout-matching INT8 Q/K/V",
    )
    hq = q.size(2 if layout_q == "BSND" else 1)
    hk = k.size(2 if layout_kv == "BSND" else 1)
    check(
        tuple(k.shape) == tuple(v.shape)
        and q.size(-1) == k.size(-1)
        and 1 <= q.size(-1) <= 256
        and hq > 0
        and hk > 0
        and hq % hk == 0
        and all(x > 0 for x in q.shape)
        and all(x > 0 for x in k.shape),
        "Full INT8 GQA requires matching K/V, divisible heads and D in [1,256]",
    )
    if packed:
        check(
            isinstance(cu_q, torch.Tensor) and isinstance(cu_k, torch.Tensor),
            "TND requires cu_seqlens_q and cu_seqlens_kv",
        )
        check(
            cu_q.dtype == cu_k.dtype == torch.int32
            and cu_q.dim() == cu_k.dim() == 1
            and cu_q.numel() == cu_k.numel()
            and cu_q.numel() >= 2
            and cu_q.is_contiguous()
            and cu_k.is_contiguous(),
            "TND requires contiguous INT32 cumulative lengths [B+1]",
        )
        out_shape = tuple(q.shape)
        batch = cu_q.numel() - 1
    else:
        check(
            cu_q is None and cu_k is None and q.size(0) == k.size(0),
            "BNSD/BSND requires matching batch and no cu_seqlens",
        )
        batch, sq = q.size(0), q.size(2 if layout_q == "BNSD" else 1)
        out_shape = (
            (batch, hq, sq, q.size(-1))
            if layout_out == "BNSD"
            else (batch, sq, hq, q.size(-1))
        )
    _check_int8_used(used_q, used_kv, batch, q.device)
    check(
        isinstance(metadata, torch.Tensor)
        and metadata.dtype == torch.int32
        and metadata.dim() == 2
        and metadata.size(0) == 2
        and metadata.is_contiguous()
        and metadata.size(1) > 0,
        "INT8 metadata requires a contiguous nonempty INT32 (2, L) scheduling buffer",
    )
    for name, scale in (
        ("q_descale", q_descale),
        ("k_descale", k_descale),
        ("v_descale", v_descale),
        ("p_scale", p_scale),
    ):
        check(isinstance(scale, torch.Tensor), f"Full INT8 GQA requires {name}")
        check(
            scale.dtype == torch.float32 and scale.is_contiguous(),
            "Full INT8 GQA requires contiguous FP32 scales",
        )
    check(
        tuple(q_descale.shape) == tuple(q.shape[:-1])
        and tuple(k_descale.shape) == tuple(k.shape[:-1])
        and tuple(v_descale.shape) == (hk,)
        and tuple(p_scale.shape) == (1,),
        "Full INT8 GQA requires per-token-head q_descale/k_descale, per-head v_descale and static p_scale [1]",
    )
    for tensor in (
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        cu_q,
        cu_k,
        metadata,
    ):
        if tensor is not None:
            check(
                tensor.device == q.device,
                "Full INT8 GQA requires tensors on the same device",
            )
    return out_shape


class QuantFlashAttnOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantFlashAttnOpBuilder, self).__init__(
            "quant_flash_attn", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/quant_flash_attn.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "quant_flash_attn_metadata(int num_heads_q, int num_heads_kv, int head_dim, int quant_mode, *, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, "
            "Tensor? seqused_kv=None, "
            "int? batch_size=None, int? max_seqlen_q=-1, int? max_seqlen_kv=-1, "
            "int? head_dim_v=None, int? mask_mode=0, int? win_left=-1, int? win_right=-1, "
            'str? layout_q="BSND", str? layout_q_descale="BSND", '
            'str? layout_kv="BSND", str? layout_out="BSND", '
            "bool is_grad_enabled=False) -> Tensor",
            "quant_flash_attn(Tensor q, Tensor k, Tensor v, "
            "Tensor q_descale, Tensor k_descale, Tensor v_descale, int quant_mode, "
            "Tensor? block_table=None, Tensor? p_scale=None, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "Tensor? sinks=None, Tensor? attn_mask=None, Tensor? metadata=None, "
            "Tensor? v_tail=None, Tensor? block_table_tail=None, Tensor? seqused_v_tail=None, "
            "float softmax_scale=1.0, int mask_mode=0, int win_left=-1, int win_right=-1, "
            "int max_seqlen_q=-1, int max_seqlen_kv=-1, "
            'str layout_q="BSND", str layout_q_descale="BSND", str layout_kv="BSND", str layout_out="BSND", '
            "bool return_softmax_lse=False) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::" + QFA_METADATA_OP_NAME)
        def quant_flash_attn_metadata_meta(
            num_heads_q: int,
            num_heads_kv: int,
            head_dim: int,
            quant_mode: Union[QuantMode, int],
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            head_dim_v: Optional[int] = None,
            mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_q_descale: Optional[str] = "BSND",
            layout_kv: Optional[str] = "BSND",
            layout_out: Optional[str] = "BSND",
            is_grad_enabled: Optional[bool] = False,
        ):
            quant_mode = _resolve_quant_mode(quant_mode)
            if quant_mode == _INT8_QUANT_MODE and layout_kv not in (
                "PA_BNBD",
                "PA_BBND",
                "PA_NZ",
            ):
                b_size = _check_int8_metadata_contract(
                    num_heads_q,
                    num_heads_kv,
                    head_dim,
                    head_dim_v,
                    batch_size,
                    max_seqlen_q,
                    max_seqlen_kv,
                    mask_mode,
                    win_left,
                    win_right,
                    layout_q,
                    layout_kv,
                    layout_out,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    seqused_q,
                    seqused_kv,
                    is_grad_enabled,
                )
            else:
                b_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q)
            max_schedule_size = _metadata_row_size(
                b_size, num_heads_q, num_heads_kv, quant_mode, layout_kv
            )
            return torch.empty((2, max_schedule_size), dtype=torch.int32, device="meta")

        @impl(get_as_library(), self.name, "Meta")
        def quant_flash_attn_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            q_descale: torch.Tensor,
            k_descale: torch.Tensor,
            v_descale: torch.Tensor,
            quant_mode: Union[QuantMode, int],
            block_table: Optional[torch.Tensor] = None,
            p_scale: Optional[torch.Tensor] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            sinks: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            v_tail: Optional[torch.Tensor] = None,
            block_table_tail: Optional[torch.Tensor] = None,
            seqused_v_tail: Optional[torch.Tensor] = None,
            softmax_scale: Optional[float] = 1.0,
            mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_q_descale: Optional[str] = "BSND",
            layout_kv: Optional[str] = "BSND",
            layout_out: Optional[str] = "BSND",
            return_softmax_lse: Optional[bool] = False,
        ):
            torch._check(
                int(quant_mode) in (0, 1, 3, 5, 6, 7, 10, 12),
                lambda: "Unsupported quant_mode",
            )
            if int(quant_mode) == _INT8_QUANT_MODE and layout_kv not in (
                "PA_BNBD",
                "PA_BBND",
                "PA_NZ",
            ):
                out_shape = _check_int8_contract(
                    q,
                    k,
                    v,
                    q_descale,
                    k_descale,
                    v_descale,
                    p_scale,
                    layout_q,
                    layout_kv,
                    layout_out,
                    mask_mode,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    metadata,
                    win_left,
                    win_right,
                    seqused_q,
                    seqused_kv,
                    (
                        block_table,
                        sinks,
                        attn_mask,
                        v_tail,
                        block_table_tail,
                        seqused_v_tail,
                    ),
                )
                return (
                    torch.empty(out_shape, dtype=torch.bfloat16, device=q.device),
                    torch.empty(
                        (
                            (q.size(1), q.size(0))
                            if layout_q == "TND"
                            else (
                                q.size(0),
                                q.size(2 if layout_q == "BSND" else 1),
                                q.size(1 if layout_q == "BSND" else 2),
                            )
                        )
                        if return_softmax_lse
                        else (0,),
                        dtype=torch.float32,
                        device=q.device,
                    ),
                )
            # 取 shape 前校验 q/v 非空及维度, 避免 None 或维度不足导致 AttributeError/IndexError
            torch._check(q is not None, lambda: "q must not be None")
            torch._check(v is not None, lambda: "v must not be None")
            q_expected = _LAYOUT_EXPECTED_NDIM.get(layout_q)
            torch._check(
                q_expected is not None,
                lambda: f"Unsupported layout_q: {layout_q!r}, expected one of TND/NTD/BSND/BNSD",
            )
            torch._check(
                q.dim() == q_expected,
                lambda: f"q with layout {layout_q} expects {q_expected} dims, but got {q.dim()} dims",
            )
            kv_expected = _LAYOUT_EXPECTED_NDIM.get(layout_kv)
            torch._check(
                kv_expected is not None,
                lambda: f"Unsupported layout_kv: {layout_kv!r}, expected one of TND/BSND/BNSD/PA_NZ/PA_BNBD/PA_BBND",
            )
            torch._check(
                v.dim() == kv_expected,
                lambda: f"v with layout {layout_kv} expects {kv_expected} dims, but got {v.dim()} dims",
            )
            t_size, b_size, s_size, n_size, d_size = _get_output_shape_sizes(
                q, v, layout_q, layout_kv, quant_mode
            )
            if layout_q in ("TND", "NTD"):
                # NTD (GQA FP8, quant_mode=6) 的 lse 与 TND 同为 (N, T)
                softmax_out_size = (n_size, t_size)
            else:
                softmax_out_size = (b_size, n_size, s_size)

            if layout_out == "TND":
                torch._check(
                    layout_q == "TND"
                    or (
                        layout_q == "NTD"
                        and int(quant_mode)
                        == int(
                            QuantMode.A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32
                        )
                    ),
                    lambda: f"When the layout of output is TND, the layout of query must be TND "
                    f"(or NTD when quant_mode=6), but got {layout_q}",
                )
                attention_out_size = (t_size, n_size, d_size)
            elif layout_out == "BNSD":
                torch._check(
                    layout_q == "BNSD",
                    lambda: f"When the layout of output is BNSD, the layout of query must be BNSD, but got {layout_q}",
                )
                attention_out_size = (b_size, n_size, s_size, d_size)
            else:
                torch._check(
                    layout_q != "TND",
                    lambda: f"When the layout of output is BSND, the layout of query "
                    f"must be BNSD or BSND, but got {layout_q}",
                )
                attention_out_size = (b_size, s_size, n_size, d_size)

            return (
                torch.empty(attention_out_size, dtype=torch.bfloat16, device=q.device),
                torch.empty(softmax_out_size, dtype=torch.float32, device=q.device),
            )


# Instantiate the builder
quant_flash_attn_op_builder = QuantFlashAttnOpBuilder()
quant_flash_attn_op_builder._ensure_initialized()


@impl(get_as_library(), QFA_METADATA_OP_NAME, "PrivateUse1")
def quant_flash_attn_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: Union[QuantMode, int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    head_dim_v: Optional[int] = None,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_q_descale: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
    layout_out: Optional[str] = "BSND",
    is_grad_enabled: Optional[bool] = False,
):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    quant_mode = _resolve_quant_mode(quant_mode)
    torch._check(
        quant_mode in (0, 1, 3, 5, 6, 7, 10, 12),
        lambda: f"The quant_mode of quant_flash_attn_metadata only supports 0 (A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32), "
        f"1 (A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32), "
        f"3 (A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16), "
        f"5 (A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16) "
        f"6 (A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) "
        f"7 (A4C4_QKV_HIF4_P_HIF4_LEVEL1_SOFTMAX_FP16) "
        f"10 (A4C4_QKV_HIF4_P_FP8_E4M3_SOFTMAX_FP16), or "
        f"12 (A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR_SOFTMAX_FP16), but got {quant_mode}",
    )
    if layout_q == "TND":
        torch._check(
            batch_size == None,
            lambda: f"When the layout of query is TND, the attribute batch_size of quant_flash_attn_metadata must be None, but got {batch_size}.",
        )
    torch._check(
        is_grad_enabled is not True or quant_mode == 0,
        lambda: f"When is_grad_enabled is True, quant_mode must be 0, but got quant_mode={quant_mode}",
    )
    if quant_mode == _INT8_QUANT_MODE and layout_kv not in (
        "PA_BNBD",
        "PA_BBND",
        "PA_NZ",
    ):
        batch_size = _check_int8_metadata_contract(
            num_heads_q,
            num_heads_kv,
            head_dim,
            head_dim_v,
            batch_size,
            max_seqlen_q,
            max_seqlen_kv,
            mask_mode,
            win_left,
            win_right,
            layout_q,
            layout_kv,
            layout_out,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqused_q,
            seqused_kv,
            is_grad_enabled,
        )
    else:
        batch_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q)
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    mask_mode = _resolve_mask_mode(mask_mode)
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_q_descale = "BSND" if layout_q_descale is None else layout_q_descale
    layout_kv = "BSND" if layout_kv is None else layout_kv
    layout_out = "BSND" if layout_out is None else layout_out
    head_dim_v = head_dim if head_dim_v is None else head_dim_v

    _check_layout_constraint(quant_mode, layout_q, layout_kv, layout_out)
    device = "npu"
    if quant_mode == _INT8_QUANT_MODE:
        device = next(
            (t.device for t in (cu_seqlens_q, seqused_q, seqused_kv) if t is not None),
            "npu",
        )
    max_schedule_size = _metadata_row_size(
        batch_size, num_heads_q, num_heads_kv, quant_mode, layout_kv
    )
    output = torch.empty((2, max_schedule_size), dtype=torch.int32, device=device)

    op_module = quant_flash_attn_op_builder.load()
    op_module.quant_flash_attn_metadata(
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        num_heads_q,
        num_heads_kv,
        head_dim,
        head_dim_v,
        quant_mode,
        mask_mode,
        win_left,
        win_right,
        layout_q,
        layout_q_descale,
        layout_kv,
        layout_out,
        is_grad_enabled,
        output,
    )
    return output


@torch.library.register_kernel("cann_ops_transformer::" + QFA_METADATA_OP_NAME, None)
def quant_flash_attn_metadata_fallback(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: Union[QuantMode, int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    head_dim_v: Optional[int] = None,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_q_descale: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
    layout_out: Optional[str] = "BSND",
    is_grad_enabled: Optional[bool] = False,
):
    # 处理所有 tensor 都为 None 的情况
    return quant_flash_attn_metadata(
        num_heads_q,
        num_heads_kv,
        head_dim,
        quant_mode,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        batch_size,
        max_seqlen_q,
        max_seqlen_kv,
        head_dim_v,
        mask_mode,
        win_left,
        win_right,
        layout_q,
        layout_q_descale,
        layout_kv,
        layout_out,
        is_grad_enabled,
    )


def _int8_should_pad(
    q,
    k,
    v,
    q_descale,
    k_descale,
    quant_mode,
    layout_q,
    layout_kv,
    layout_out,
    mask_mode,
    win_left,
    win_right,
):
    """Amortize whole-tensor D72 padding using shape and physical-core parallelism."""
    if not (
        quant_mode == _INT8_QUANT_MODE
        and win_left == win_right == -1
        and layout_q == layout_kv == layout_out
        and layout_q in ("BNSD", "BSND")
        and q.shape == k.shape == v.shape
        and q.size(-1) == 72
        and q.device.type == "npu"
        and mask_mode in (0, 3)
        and all(x.is_contiguous() for x in (q, k, v, q_descale, k_descale))
    ):
        return False
    head_axis = 2 if layout_q == "BSND" else 1
    batch, heads, seq = q.size(0), q.size(head_axis), q.size(3 - head_axis)
    cores = torch.npu.get_device_properties(q.device.index).cube_core_num
    query_tiles = (seq + 511) // 512
    query_tasks = batch * heads * query_tiles
    return query_tiles > 2 and (
        query_tasks >= 2 * cores or query_tasks * query_tiles >= 4 * cores
    )


def _quant_flash_attn_aligned_padded(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    softmax_scale,
    metadata,
    mask_mode=0,
    layout="BNSD",
    padded_dim=96,
    win_left=-1,
    win_right=-1,
    return_softmax_lse=False,
):
    """Pad all Q/K/V once, then write the logical-width output directly.

    The caller precomputes one metadata tensor for the full padded shape.
    """
    original_dim = q.size(-1)
    padded = F.pad(torch.stack((q, k, v)), (0, padded_dim - original_dim))
    q_pad, k_pad, v_pad = padded.unbind(0)
    return quant_flash_attn_op_builder.load().quant_flash_attn_padded_out(
        q_pad,
        k_pad,
        v_pad,
        q_descale,
        k_descale,
        v_descale,
        _INT8_QUANT_MODE,
        None,
        p_scale,
        None,
        None,
        None,
        None,
        None,
        None,
        metadata,
        None,
        None,
        None,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        -1,
        -1,
        layout,
        layout,
        layout,
        layout,
        return_softmax_lse,
        original_dim,
    )


@impl(get_as_library(), quant_flash_attn_op_builder.name, "PrivateUse1")
def quant_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    quant_mode: Union[QuantMode, int],
    block_table: Optional[torch.Tensor] = None,
    p_scale: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    v_tail: Optional[torch.Tensor] = None,
    block_table_tail: Optional[torch.Tensor] = None,
    seqused_v_tail: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = 1.0,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_q_descale: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
    layout_out: Optional[str] = "BSND",
    return_softmax_lse: Optional[bool] = False,
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    quant_mode = _resolve_quant_mode(quant_mode)
    torch._check(
        quant_mode in (0, 1, 3, 5, 6, 7, 10, 12),
        lambda: f"The quant_mode of quant_flash_attn only supports 0 (A8C8_QKV_HIF8_P_PER_TENSOR_SOFTMAX_FP32), "
        f"1 (A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32), "
        f"3 (A8C8_QKV_MXFP8_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP16), "
        f"5 (A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16) "
        f"6 (A8C8_QK_FP8_E4M3_PER_TOKEN_HEAD_V_FP8_E4M3_PER_HEAD_P_FP8_E4M3_PER_TENSOR_SOFTMAX_FP32) "
        f"7 (A4C4_QKV_HIF4_P_HIF4_LEVEL1_SOFTMAX_FP16) "
        f"10 (A4C4_QKV_HIF4_P_FP8_E4M3_SOFTMAX_FP16), "
        f"12 (A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR_SOFTMAX_FP16), but got {quant_mode}",
    )
    mask_mode = _resolve_mask_mode(mask_mode)

    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    if quant_mode == _INT8_QUANT_MODE and layout_kv not in (
        "PA_BNBD",
        "PA_BBND",
        "PA_NZ",
    ):
        max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
        max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
        _check_int8_contract(
            q,
            k,
            v,
            q_descale,
            k_descale,
            v_descale,
            p_scale,
            layout_q,
            layout_kv,
            layout_out,
            mask_mode,
            cu_seqlens_q,
            cu_seqlens_kv,
            metadata,
            win_left,
            win_right,
            seqused_q,
            seqused_kv,
            (
                block_table,
                sinks,
                attn_mask,
                v_tail,
                block_table_tail,
                seqused_v_tail,
            ),
        )

    _check_layout_constraint(quant_mode, layout_q, layout_kv, layout_out)

    # 取 shape 前校验 q/v 非空及维度, 避免 None 或维度不足导致 AttributeError/IndexError
    torch._check(q is not None, lambda: "q must not be None")
    torch._check(v is not None, lambda: "v must not be None")
    q_expected = _LAYOUT_EXPECTED_NDIM.get(layout_q)
    torch._check(
        q_expected is not None,
        lambda: f"Unsupported layout_q: {layout_q!r}, expected one of TND/NTD/BSND/BNSD",
    )
    torch._check(
        q.dim() == q_expected,
        lambda: f"q with layout {layout_q} expects {q_expected} dims, but got {q.dim()} dims",
    )
    kv_expected = _LAYOUT_EXPECTED_NDIM.get(layout_kv)
    torch._check(
        kv_expected is not None,
        lambda: f"Unsupported layout_kv: {layout_kv!r}, expected one of TND/BSND/BNSD/PA_NZ/PA_BNBD/PA_BBND",
    )
    torch._check(
        v.dim() == kv_expected,
        lambda: f"v with layout {layout_kv} expects {kv_expected} dims, but got {v.dim()} dims",
    )
    out_expected = _OUT_LAYOUT_EXPECTED_NDIM.get(layout_out)
    torch._check(
        out_expected is not None,
        lambda: f"Unsupported layout_out: {layout_out!r}, expected one of TND/BSND/BNSD",
    )
    # 异常拦截: 仅校验决定输出 shape 的维度(q 除 D 外的 B/T/S/N 维度及 v 的 D 维度),
    # 其余维度(如 q 的 D 维)不影响输出 shape, 不在此拦截
    t_size, b_size, s_size, n_size, d_size = _get_output_shape_sizes(
        q, v, layout_q, layout_kv, quant_mode
    )
    torch._check(
        all(
            size > 0
            for size in (t_size, b_size, s_size, n_size, d_size)
            if size is not None
        ),
        lambda: (
            f"The output-related dims of q (except D) and the D dim of v must not be 0, "
            f"got q.shape={tuple(q.shape)}, v.shape={tuple(v.shape)}"
        ),
    )
    # QFA 不支持 k/v/kscale/vscale 的 stride 中含 0
    for name, tensor in (
        ("k", k),
        ("v", v),
        ("k_descale", k_descale),
        ("v_descale", v_descale),
    ):
        if tensor is not None and 0 in tensor.stride():
            raise ValueError(
                f"quant_flash_attn does not support {name} with stride containing 0, "
                f"got {name}.stride() = {tensor.stride()}"
            )

    if quant_mode == int(QuantMode.A4C4_QKV_MXFP4_P_MXFP4_SOFTMAX_FP16):
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.dtype != torch.float4_e2m1fn_x2:
                raise ValueError(
                    f"In MxFP4 mode (quant_mode=5), {name} must be "
                    f"float4_e2m1fn_x2, but got {tensor.dtype}"
                )

    if quant_mode == int(
        QuantMode.A8C8_QK_INT8_PER_TOKEN_HEAD_V_INT8_PER_HEAD_P_INT8_PER_TENSOR
    ):
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if tensor.dtype != torch.int8:
                raise ValueError(
                    f"In A2/A3 paged INT8 mode (quant_mode=12), {name} must be int8, "
                    f"but got {tensor.dtype}"
                )
    op_module = quant_flash_attn_op_builder.load()
    return op_module.quant_flash_attn(
        q,
        k,
        v,
        q_descale,
        k_descale,
        v_descale,
        quant_mode,
        block_table,
        p_scale,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        v_tail,
        block_table_tail,
        seqused_v_tail,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_q_descale,
        layout_kv,
        layout_out,
        return_softmax_lse,
    )


quant_flash_attn.QuantMode = QuantMode
quant_flash_attn.MaskMode = MaskMode
quant_flash_attn_metadata.QuantMode = QuantMode
quant_flash_attn_metadata.MaskMode = MaskMode
