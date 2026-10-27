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
import torch_npu
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from torch.library import impl

QFMLA_METADATA_OP_NAME = "quant_flash_mla_metadata"
METADATA_STRIDE = 16

# MLA固定维度: head_dim_qk = nope(512) + rope(64) = 576, head_dim_v = 512
MLA_HEAD_DIM_QK = 576
MLA_HEAD_DIM_V = 512


class QuantMode(IntEnum):
    """quant_mode 枚举：对外字符串/int 经由本枚举映射为传给算子侧的 int。"""

    MLA_FP8_E4M3_FULLQUANT = 1
    MLA_HIF8_FULLQUANT = 0


class MaskMode(IntEnum):
    """mask_mode 枚举：对外字符串/int 经由本枚举映射为传给算子侧的 int。"""

    NO_MASK = 0
    CAUSAL = 3


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
    return int(quant_mode)


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
    return int(mask_mode)


def _normalize_optional_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """空 (0,) 张量归一成 None。

    ttk 等 CSV 驱动框架对未填写的 optional 输入会以空 (0,) slot 张量传入;
    host 端 CheckSeqLens 会把 size(0)=0 判为非法 shape ("cuSeqLens shape must
    be (batchSize+1,), but got 0")。归一成 None 才能正确表达"未提供"语义。
    """
    if t is not None and t.numel() == 0:
        return None
    return t


def _get_core_nums():
    """Return the core counts used by the metadata AICPU scheduler."""
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        return 36, 72
    props = npu.get_device_properties()
    return props.cube_core_num, props.vector_core_num


def _calculate_max_schedule_size(batch_size, num_heads_kv):
    """Allocate one row for the worst-case MLA metadata section count."""
    align_size = 4096
    batch_size = batch_size if batch_size and batch_size > 0 else 1
    num_heads_kv = num_heads_kv if num_heads_kv and num_heads_kv > 0 else 1
    aic_num, aiv_num = _get_core_nums()
    schedule_size = METADATA_STRIDE + (
        (aic_num + aiv_num) * METADATA_STRIDE * batch_size * num_heads_kv
    )
    return ((schedule_size + align_size - 1) // align_size) * align_size


# 各 layout 期望的 tensor 维度数
_LAYOUT_Q_EXPECTED_NDIM = {
    "TND": 3,
    "BSND": 4,
    "BNSD": 4,
}
_LAYOUT_KV_EXPECTED_NDIM = {
    "PA_NZ": 5,
    "PA_BNBD": 4,
    "PA_BBND": 4,
}

_SUPPORTED_NUM_HEADS_Q = (1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def _check_int32_1d(name: str, tensor: torch.Tensor) -> None:
    torch._check(tensor is not None, lambda: f"{name} must be provided")
    torch._check(
        tensor.dtype == torch.int32,
        lambda: f"{name} must be int32, but got {tensor.dtype}",
    )
    torch._check(
        tensor.dim() == 1, lambda: f"{name} must be 1D, but got {tensor.dim()}D"
    )
    torch._check(tensor.numel() > 0, lambda: f"{name} must be non-empty")


def _check_metadata_parameters(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    quant_mode,
    cu_seqlens_q,
    seqused_q,
    max_seqlen_q,
    max_seqlen_kv,
    head_dim_qk,
    head_dim_v,
    mask_mode,
    layout_q,
) -> None:
    torch._check(
        quant_mode
        in (int(QuantMode.MLA_FP8_E4M3_FULLQUANT), int(QuantMode.MLA_HIF8_FULLQUANT)),
        lambda: f"quant_mode only supports 1 (MLA_FP8_E4M3_FULLQUANT) or "
        f"0 (MLA_HIF8_FULLQUANT), but got {quant_mode}",
    )
    torch._check(
        layout_q in ("TND", "BSND", "BNSD"),
        lambda: f"layout_q only supports 'TND'/'BSND'/'BNSD', but got {layout_q!r}",
    )
    torch._check(
        num_heads_q in _SUPPORTED_NUM_HEADS_Q,
        lambda: f"num_heads_q must be one of {_SUPPORTED_NUM_HEADS_Q}, but got {num_heads_q}",
    )
    torch._check(
        num_heads_kv == 1,
        lambda: f"num_heads_kv only supports 1, but got {num_heads_kv}",
    )
    torch._check(
        head_dim_qk == MLA_HEAD_DIM_QK,
        lambda: f"head_dim_qk only supports 576, but got {head_dim_qk}",
    )
    torch._check(
        head_dim_v == MLA_HEAD_DIM_V,
        lambda: f"head_dim_v only supports 512, but got {head_dim_v}",
    )
    torch._check(
        mask_mode in (0, 3),
        lambda: f"mask_mode only supports 0 or 3, but got {mask_mode}",
    )
    torch._check(
        max_seqlen_kv == -1 or max_seqlen_kv >= 1,
        lambda: f"max_seqlen_kv must be -1 or greater than 0, but got {max_seqlen_kv}",
    )
    # max_seqlen_q 仅在 TND 下参与调度([1,16]); BSND/BNSD 的 s1 来自 q shape, 不校验上限
    torch._check(
        max_seqlen_q == -1
        or (max_seqlen_q >= 1 and (layout_q != "TND" or max_seqlen_q <= 16)),
        lambda: f"max_seqlen_q must be -1 or greater than 0 (TND caps at 16), but got {max_seqlen_q}",
    )
    _check_int32_1d("cache_seqlens", cache_seqlens)
    batch_size = cache_seqlens.numel()
    if layout_q == "TND":
        _check_int32_1d("cu_seqlens_q", cu_seqlens_q)
        torch._check(
            cu_seqlens_q.numel() == batch_size + 1,
            lambda: f"cu_seqlens_q shape must be (B+1,), but got {tuple(cu_seqlens_q.shape)} for B={batch_size}",
        )
        torch._check(
            seqused_q is not None or max_seqlen_q > 0,
            lambda: "When layout_q is TND, at least one of seqused_q or max_seqlen_q must be provided",
        )
    else:
        torch._check(
            cu_seqlens_q is None,
            lambda: "cu_seqlens_q is only supported when layout_q is TND",
        )
    if seqused_q is not None:
        _check_int32_1d("seqused_q", seqused_q)
        torch._check(
            seqused_q.numel() == batch_size,
            lambda: f"seqused_q shape must be (B,), but got {tuple(seqused_q.shape)} for B={batch_size}",
        )


class QuantFlashMlaOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantFlashMlaOpBuilder, self).__init__(
            "quant_flash_mla", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/quant_flash_mla.cpp"]

    def schema(self) -> list:
        """PyTorch operator signature."""
        return [
            "quant_flash_mla_metadata(Tensor cache_seqlens, int num_heads_q, int num_heads_kv, "
            "int quant_mode, *, "
            "Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, "
            "int? max_seqlen_q=-1, int? max_seqlen_kv=-1, "
            "int? head_dim_qk=576, int? head_dim_v=512, int? mask_mode=0, "
            'str? layout_q="TND") -> Tensor',
            "quant_flash_mla(Tensor q, Tensor k_cache, Tensor q_descale, Tensor k_descale, "
            "Tensor block_table, Tensor cache_seqlens, int quant_mode, *, "
            "Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, "
            "Tensor? attn_mask=None, Tensor? metadata=None, "
            "int head_dim_v=512, float softmax_scale=1.0, int mask_mode=0, "
            "int max_seqlen_q=-1, int max_seqlen_kv=-1, "
            'str layout_q="TND", str layout_kv="PA_BNBD", str layout_out="TND", '
            "bool return_softmax_lse=False) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::" + QFMLA_METADATA_OP_NAME)
        def quant_flash_mla_metadata_meta(
            cache_seqlens: torch.Tensor,
            num_heads_q: int,
            num_heads_kv: int,
            quant_mode: Union[QuantMode, int],
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            head_dim_qk: Optional[int] = MLA_HEAD_DIM_QK,
            head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
            mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
            layout_q: Optional[str] = "TND",
        ):
            batch_size = cache_seqlens.size(0) if cache_seqlens.dim() > 0 else 0
            max_schedule_size = _calculate_max_schedule_size(batch_size, num_heads_kv)
            return torch.empty((2, max_schedule_size), dtype=torch.int32, device="meta")

        @torch.library.register_fake("cann_ops_transformer::quant_flash_mla")
        def quant_flash_mla_meta(
            q: torch.Tensor,
            k_cache: torch.Tensor,
            q_descale: torch.Tensor,
            k_descale: torch.Tensor,
            block_table: torch.Tensor,
            cache_seqlens: torch.Tensor,
            quant_mode: Union[QuantMode, int],
            *,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
            softmax_scale: Optional[float] = 1.0,
            mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "TND",
            layout_kv: Optional[str] = "PA_BNBD",
            layout_out: Optional[str] = "TND",
            return_softmax_lse: Optional[bool] = False,
        ):
            # FakeTensor only needs enough structure to infer output shapes.
            # Contract validation belongs to the C++ host checker.
            torch._check(q is not None, lambda: "q must be provided")
            torch._check(
                q.dim() >= 2,
                lambda: f"q must have at least 2 dimensions, but got {q.dim()}D",
            )
            torch._check(head_dim_v is not None, lambda: "head_dim_v must be provided")
            t_size, n_size = q.size(0), q.size(1)
            softmax_out_size = (n_size, t_size) if return_softmax_lse else (0,)
            attention_out_size = (t_size, n_size, head_dim_v)

            return (
                torch.empty(attention_out_size, dtype=torch.bfloat16, device=q.device),
                torch.empty(softmax_out_size, dtype=torch.float, device=q.device),
            )


# Instantiate the builder
quant_flash_mla_op_builder = QuantFlashMlaOpBuilder()
quant_flash_mla_op_builder._ensure_initialized()
# 预加载 C++ 扩展，避免 aclgraph/torch.compile 时在 compile 区域内首次调用 load() 触发 graph break
quant_flash_mla_op_builder.load(verbose=False)


@impl(get_as_library(), QFMLA_METADATA_OP_NAME, "PrivateUse1")
def quant_flash_mla_metadata(
    cache_seqlens: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    quant_mode: Union[QuantMode, int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    head_dim_qk: Optional[int] = MLA_HEAD_DIM_QK,
    head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    layout_q: Optional[str] = "TND",
):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    if quant_mode not in (0, 1):
        raise ValueError(
            f"The quant_mode of quant_flash_mla_metadata only supports "
            f"1 (MLA_FP8_E4M3_FULLQUANT) or 0 (MLA_HIF8_FULLQUANT), but got {quant_mode}"
        )
    if num_heads_kv != 1:
        raise ValueError(
            f"The num_heads_kv of quant_flash_mla only supports 1 (MLA KV shared), "
            f"but got {num_heads_kv}"
        )
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    quant_mode = _resolve_quant_mode(quant_mode)
    mask_mode = _resolve_mask_mode(mask_mode)
    layout_q = "TND" if layout_q is None else layout_q
    head_dim_qk = MLA_HEAD_DIM_QK if head_dim_qk is None else head_dim_qk
    head_dim_v = MLA_HEAD_DIM_V if head_dim_v is None else head_dim_v
    cu_seqlens_q = _normalize_optional_tensor(cu_seqlens_q)
    seqused_q = _normalize_optional_tensor(seqused_q)

    _check_metadata_parameters(
        cache_seqlens,
        num_heads_q,
        num_heads_kv,
        quant_mode,
        cu_seqlens_q,
        seqused_q,
        max_seqlen_q,
        max_seqlen_kv,
        head_dim_qk,
        head_dim_v,
        mask_mode,
        layout_q,
    )

    max_schedule_size = _calculate_max_schedule_size(
        cache_seqlens.size(0), num_heads_kv
    )
    output = torch.empty((2, max_schedule_size), dtype=torch.int32, device="npu")

    op_module = quant_flash_mla_op_builder.load()
    return op_module.quant_flash_mla_metadata(
        cache_seqlens,
        num_heads_q,
        num_heads_kv,
        quant_mode,
        cu_seqlens_q,
        seqused_q,
        max_seqlen_q,
        max_seqlen_kv,
        head_dim_qk,
        head_dim_v,
        mask_mode,
        layout_q,
        output,
    )


@torch.library.register_kernel("cann_ops_transformer::" + QFMLA_METADATA_OP_NAME, None)
def quant_flash_mla_metadata_fallback(
    cache_seqlens: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    quant_mode: Union[QuantMode, int],
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    head_dim_qk: Optional[int] = MLA_HEAD_DIM_QK,
    head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    layout_q: Optional[str] = "TND",
):
    # 处理所有 tensor 都为 None 的情况
    return quant_flash_mla_metadata(
        cache_seqlens,
        num_heads_q,
        num_heads_kv,
        quant_mode,
        cu_seqlens_q,
        seqused_q,
        max_seqlen_q,
        max_seqlen_kv,
        head_dim_qk,
        head_dim_v,
        mask_mode,
        layout_q,
    )


@impl(get_as_library(), quant_flash_mla_op_builder.name, "PrivateUse1")
def quant_flash_mla(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    quant_mode: Union[QuantMode, int],
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    head_dim_v: Optional[int] = MLA_HEAD_DIM_V,
    softmax_scale: Optional[float] = 1.0,
    mask_mode: Optional[Union[MaskMode, int]] = MaskMode.NO_MASK,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "TND",
    layout_kv: Optional[str] = "PA_BNBD",
    layout_out: Optional[str] = "TND",
    return_softmax_lse: Optional[bool] = False,
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    if quant_mode not in (0, 1):
        raise ValueError(
            f"The quant_mode of quant_flash_mla only supports "
            f"1 (MLA_FP8_E4M3_FULLQUANT) or 0 (MLA_HIF8_FULLQUANT), but got {quant_mode}"
        )
    quant_mode = _resolve_quant_mode(quant_mode)
    mask_mode = _resolve_mask_mode(mask_mode)
    if head_dim_v is None:
        head_dim_v = MLA_HEAD_DIM_V
    cu_seqlens_q = _normalize_optional_tensor(cu_seqlens_q)
    seqused_q = _normalize_optional_tensor(seqused_q)

    # Prevent empty outputs from being short-circuited by the framework before
    # host tiling validation. QFA applies the same guard to output-related dims.
    # Rank errors remain the responsibility of the C++ checker.
    if q.dim() >= 2:
        torch._check(
            all(size > 0 for size in (q.size(0), q.size(1), head_dim_v)),
            lambda: (
                "The output-related dims T/N of q and head_dim_v must be greater than 0, "
                f"but got q.shape={tuple(q.shape)}, head_dim_v={head_dim_v}"
            ),
        )
    q_expected = _LAYOUT_Q_EXPECTED_NDIM.get(layout_q)
    if q_expected is None:
        raise ValueError(
            f"Unsupported layout_q: {layout_q!r}, expected one of TND/BSND/BNSD"
        )
    if q.dim() != q_expected:
        raise ValueError(
            f"q with layout {layout_q} expects {q_expected} dims, but got {q.dim()} dims"
        )
    kv_expected = _LAYOUT_KV_EXPECTED_NDIM.get(layout_kv)
    if kv_expected is None:
        raise ValueError(
            f"Unsupported layout_kv: {layout_kv!r}, expected one of PA_BBND/PA_BNBD/PA_NZ"
        )
    if k_cache.dim() != kv_expected:
        raise ValueError(
            f"k_cache with layout {layout_kv} expects {kv_expected} dims, but got {k_cache.dim()} dims"
        )
    if metadata is None:
        raise ValueError("metadata should be provided by quant_flash_mla_metadata")
    if quant_mode == int(QuantMode.MLA_FP8_E4M3_FULLQUANT):
        if q.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"In FP8_E4M3 mode (quant_mode=1), q must be float8_e4m3fn, but got {q.dtype}"
            )
        if k_cache.dtype != torch.float8_e4m3fn:
            raise ValueError(
                f"In FP8_E4M3 mode (quant_mode=1), k_cache must be float8_e4m3fn, but got {k_cache.dtype}"
            )
    elif quant_mode == int(QuantMode.MLA_HIF8_FULLQUANT):
        # HIF8: torch侧无原生hifloat8 dtype, 以uint8承载hif8编码, csrc侧映射为ACL_HIFLOAT8
        if q.dtype != torch.uint8:
            raise ValueError(
                f"In HIF8 mode (quant_mode=0), q must be uint8 (hif8 encoded), but got {q.dtype}"
            )
        if k_cache.dtype != torch.uint8:
            raise ValueError(
                f"In HIF8 mode (quant_mode=0), k_cache must be uint8 (hif8 encoded), but got {k_cache.dtype}"
            )

    # MLA 不支持 k_cache 的 stride 中含 0（仅 k_cache 支持 PA 非连续）
    for name, tensor in (("k_cache", k_cache),):
        if tensor is not None and 0 in tensor.stride():
            raise ValueError(
                f"quant_flash_mla does not support {name} with stride containing 0, "
                f"got {name}.stride() = {tensor.stride()}"
            )

    op_module = quant_flash_mla_op_builder.load()
    return op_module.quant_flash_mla(
        q,
        k_cache,
        q_descale,
        k_descale,
        block_table,
        cache_seqlens,
        quant_mode,
        cu_seqlens_q,
        seqused_q,
        attn_mask,
        metadata,
        head_dim_v,
        softmax_scale,
        mask_mode,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
        return_softmax_lse,
    )


quant_flash_mla.QuantMode = QuantMode
quant_flash_mla.MaskMode = MaskMode
quant_flash_mla_metadata.QuantMode = QuantMode
quant_flash_mla_metadata.MaskMode = MaskMode

quant_flash_mla = torch.ops.cann_ops_transformer.quant_flash_mla
quant_flash_mla_metadata = torch.ops.cann_ops_transformer.quant_flash_mla_metadata
