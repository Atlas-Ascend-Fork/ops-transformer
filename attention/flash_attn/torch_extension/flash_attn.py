# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

FA_METADATA_OP_NAME = "flash_attn_metadata"
METADATA_STRIDE = 16
FAG_METADATA_SIZE = 412  # FAG (Flash Attn Grad) metadata size, in int32 elements


_LAYOUT_DIMS = {
    "TND": 3,
    "BSND": 4,
    "BNSD": 4,
    "PA_BBND": 4,
    "PA_BNBD": 4,
    "PA_NZ": 5,
}


def _check_layout_dims(tensor, name, layout):
    # Layout validity is checked by tiling; only check rank here to prevent out-of-range shape access.
    expected = _LAYOUT_DIMS.get(layout)
    if expected is None:
        return
    torch._check(
        tensor.dim() == expected,
        lambda: f"{name} with layout {layout} expects {expected} dims, "
        f"but got {tensor.dim()} dims {tuple(tensor.shape)}",
    )


def _validate_layout_dims(q, k, v, layout_q, layout_kv):
    # Validate before indexing shapes or inferring output dimensions.
    _check_layout_dims(q, "q", layout_q)
    _check_layout_dims(k, "k", layout_kv)
    _check_layout_dims(v, "v", layout_kv)


def _validate_kv_stride(k, v):
    # FA 不支持 k/v 的 stride 中含 0
    for name, tensor in (("k", k), ("v", v)):
        if 0 in tensor.stride():
            raise ValueError(
                f"flash_attn does not support {name} with stride containing 0, "
                f"got {name}.stride() = {tensor.stride()}"
            )


def _flash_attn_out_sizes(
    q, k, v, layout_q, layout_kv, layout_out, return_softmax_lse=False
):
    """_flash_attn_forward 的输出 shape 推导(与 C++/aclnn 行为一致)。

    softmax_lse: 关闭时为空 tensor {0}, 开启时 TND 为 (N, T)、其余为 (B, N, S)。
    """
    _validate_layout_dims(q, k, v, layout_q, layout_kv)
    if layout_q == "TND":
        t_size = q.size(0)
        n_size = q.size(1)
    elif layout_q == "BSND":
        b_size = q.size(0)
        s_size = q.size(1)
        n_size = q.size(2)
    else:
        b_size = q.size(0)
        n_size = q.size(1)
        s_size = q.size(2)

    if layout_kv == "PA_NZ":
        d_size = v.size(-3) * v.size(-1)
    else:
        d_size = v.size(-1)

    if layout_out == "TND":
        torch._check(
            layout_q == "TND",
            lambda: (
                f"When the layout of output is TND, the layout of query must be TND, but got {layout_q}"
            ),
        )
        attention_out_size = (t_size, n_size, d_size)
    elif layout_out == "BNSD":
        torch._check(
            layout_q == "BNSD",
            lambda: (
                f"When the layout of output is BNSD, the layout of query must be BNSD, but got {layout_q}"
            ),
        )
        attention_out_size = (b_size, n_size, s_size, d_size)
    else:
        torch._check(
            layout_q != "TND",
            lambda: (
                f"When the layout of output is BSND, the layout of query must be BNSD or BSND, but got {layout_q}"
            ),
        )
        attention_out_size = (b_size, s_size, n_size, d_size)

    if not return_softmax_lse:
        softmax_lse_size = (0,)
    elif layout_q == "TND":
        softmax_lse_size = (n_size, t_size)
    else:
        softmax_lse_size = (b_size, n_size, s_size)
    return attention_out_size, softmax_lse_size


def _get_npu_core_nums():
    try:
        properties = torch.npu.get_device_properties()
        return properties.cube_core_num, properties.vector_core_num
    except Exception:
        return 128, 128


def _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q):
    if seqused_q is not None:
        return seqused_q.size(0)
    elif cu_seqlens_q is not None and cu_seqlens_q.size(0) > 0:
        return cu_seqlens_q.size(0) - 1

    if batch_size is None:
        raise ValueError(
            "When seqused_q and cu_seqlens_q are not provided, batch_size must be provided"
        )
    elif batch_size < 0:
        raise ValueError(
            f"When seqused_q and cu_seqlens_q are not provided, "
            f"batch_size must be greater than 0, but got {batch_size}"
        )
    return batch_size


def _fag_deter_elems(
    batch_size, num_heads_q, num_heads_kv, max_seqlen_q, max_seqlen_kv
):
    """Versioned bounded TND schedule tail (int32 words), independent of rounds."""
    return 518 + 32 * batch_size


def _calculate_metadata_size(
    batch_size,
    num_heads_kv,
    num_heads_q=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
):
    """计算 metadata tensor 的对齐后大小"""
    align_size = 4096
    head_size = 1 * METADATA_STRIDE
    aic_num, aiv_num = _get_npu_core_nums()
    fa_size = aic_num * METADATA_STRIDE * batch_size * num_heads_kv
    fd_size = aiv_num * METADATA_STRIDE * batch_size * num_heads_kv

    metadata_size = head_size + fa_size + fd_size
    metadata_size += FAG_METADATA_SIZE  # FAG (Flash Attn Grad) metadata size
    metadata_size += _fag_deter_elems(
        batch_size, num_heads_q, num_heads_kv, max_seqlen_q, max_seqlen_kv
    )
    return ((metadata_size + align_size - 1) // align_size) * align_size


class FlashAttenOpBuilder(OpBuilder):
    def __init__(self):
        super(FlashAttenOpBuilder, self).__init__("flash_attn", category="attention")

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/flash_attn.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "flash_attn_metadata(int num_heads_q, int num_heads_kv, int head_dim, *, "
            "int? head_dim_v=None, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None,"
            "int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_kv=None, "
            "int? mask_mode=None, int? win_left=None, int? win_right=None, "
            "str? layout_q=None, str? layout_kv=None, str? layout_out=None, "
            "bool is_grad_enabled=False, float dropout_p=0.0) -> Tensor",
            "flash_attn(Tensor q, Tensor k, Tensor v,"
            "Tensor?block_table=None, Tensor?cu_seqlens_q=None,"
            "Tensor?cu_seqlens_kv=None, Tensor?seqused_q=None,"
            "Tensor?seqused_kv=None, Tensor?sinks=None, Tensor?attn_mask=None, Tensor?metadata=None,"
            "float softmax_scale=1.0, int mask_mode=0, int win_left=-1, int win_right=-1,"
            "int max_seqlen_q=-1, int max_seqlen_kv=-1,"
            'str layout_q="BSND", str layout_kv="BSND", str layout_out="BSND",'
            "bool return_softmax_lse=False, float dropout_p=0.0, Tensor? seed=None, Tensor? offset=None) -> (Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @torch.library.register_fake("cann_ops_transformer::flash_attn_metadata")
        def flash_attn_metadata_meta(
            num_heads_q: int,
            num_heads_kv: int,
            head_dim: int,
            head_dim_v: Optional[int] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = -1,
            max_seqlen_q: Optional[int] = None,
            max_seqlen_kv: Optional[int] = None,
            mask_mode: Optional[int] = None,
            win_left: Optional[int] = None,
            win_right: Optional[int] = None,
            layout_q: Optional[str] = None,
            layout_kv: Optional[str] = None,
            layout_out: Optional[str] = None,
            is_grad_enabled: Optional[bool] = False,
            dropout_p: float = 0.0,
        ):
            b_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q)
            metadata_size = _calculate_metadata_size(
                b_size, num_heads_kv, num_heads_q, max_seqlen_q, max_seqlen_kv
            )
            return torch.empty((metadata_size,), dtype=torch.int32, device="meta")

        @torch.library.register_fake("cann_ops_transformer::flash_attn")
        def flash_attn_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            block_table: Optional[torch.Tensor] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            sinks: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            softmax_scale: Optional[float] = 1.0,
            mask_mode: Optional[int] = 0,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_kv: Optional[str] = "BSND",
            layout_out: Optional[str] = "BSND",
            return_softmax_lse: Optional[bool] = False,
            dropout_p: Optional[float] = 0.0,
            seed: Optional[torch.Tensor] = None,
            offset: Optional[torch.Tensor] = None,
        ):
            _validate_layout_dims(q, k, v, layout_q, layout_kv)
            if layout_q == "TND":
                t_size = q.size(0)
                n_size = q.size(1)
                softmax_out_size = (n_size, t_size)
            elif layout_q == "BSND":
                b_size = q.size(0)
                s_size = q.size(1)
                n_size = q.size(2)
                softmax_out_size = (b_size, n_size, s_size)
            else:
                b_size = q.size(0)
                n_size = q.size(1)
                s_size = q.size(2)
                softmax_out_size = (b_size, n_size, s_size)

            if layout_kv == "PA_NZ":
                d_size = v.size(-3) * v.size(-1)
            else:
                d_size = v.size(-1)

            if layout_out == "TND":
                torch._check(
                    layout_q == "TND",
                    lambda: (
                        f"When the layout of output is TND, the layout of query must be TND, but got {layout_q}"
                    ),
                )
                attention_out_size = (t_size, n_size, d_size)
            elif layout_out == "BNSD":
                torch._check(
                    layout_q == "BNSD",
                    lambda: (
                        f"When the layout of output is BNSD, the layout of query must be BNSD, but got {layout_q}"
                    ),
                )
                attention_out_size = (b_size, n_size, s_size, d_size)
            else:
                torch._check(
                    layout_q != "TND",
                    lambda: (
                        f"When the layout of output is BSND, the layout of query must be BNSD or BSND, but got {layout_q}"
                    ),
                )
                attention_out_size = (b_size, s_size, n_size, d_size)

            return (
                torch.empty(attention_out_size, dtype=q.dtype, device=q.device),
                torch.empty(softmax_out_size, dtype=torch.float, device=q.device),
            )


# Instantiate the builder
flash_attn_op_builder = FlashAttenOpBuilder()
flash_attn_op_builder._ensure_initialized()


# ===========================================================================
# AutoGrad 接线(compressor/compressor_grad 范式)
#   - 内部 _flash_attn_forward: autograd 载体, 返回 backward 所需中间量
#     (softmax_lse 训练态由公开 wrapper 强制开启);
#   - 内部 _flash_attn_backward: 调用独立反向算子 flash_attn_grad(懒加载);
#   - 芯片无关(wrapper 不感知 arch); 反向不支持的方向(如 TND 未覆盖场景)
#     由反向算子自身拦截;
#   - softmax_lse 为非可微辅助输出, 对其反向显式报错。
# ===========================================================================
@torch.library.custom_op(
    "cann_ops_transformer::_flash_attn_forward", mutates_args=(), device_types="npu"
)
def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    dropout_p: float = 0.0,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    op_module = flash_attn_op_builder.load()
    return op_module.flash_attn(
        q,
        k,
        v,
        block_table,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
        return_softmax_lse,
        dropout_p,
        seed,
        offset,
    )


@torch.library.register_fake("cann_ops_transformer::_flash_attn_forward")
def _flash_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
    dropout_p: float = 0.0,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    attention_out_size, lse_size = _flash_attn_out_sizes(
        q, k, v, layout_q, layout_kv, layout_out, return_softmax_lse
    )
    return (
        torch.empty(attention_out_size, dtype=q.dtype, device=q.device),
        torch.empty(lse_size, dtype=torch.float, device=q.device),
    )


@torch.library.custom_op(
    "cann_ops_transformer::_flash_attn_backward", mutates_args=(), device_types="npu"
)
def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # 反向为独立算子, 懒加载
    from ..flash_attn_grad import flash_attn_grad

    return flash_attn_grad(
        q,
        k,
        v,
        dout,
        attn_out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
    )


@torch.library.register_fake("cann_ops_transformer::_flash_attn_backward")
def _flash_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    layout_out: str = "BSND",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    n_q = q.size(1) if layout_q in ("TND", "BNSD") else q.size(2)
    dsink = torch.empty((n_q,), dtype=torch.float32, device=q.device)
    return (dq, dk, dv, dsink)


def _flash_attn_autograd_setup_context(ctx, inputs, output):
    ctx.set_materialize_grads(False)
    (
        q,
        k,
        v,
        block_table,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
        _return_softmax_lse,
        _dropout_p,
        _seed,
        _offset,
    ) = inputs
    attn_out, softmax_lse = output
    ctx.save_for_backward(
        q,
        k,
        v,
        attn_out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
    )
    ctx.flash_attn_attrs = (
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
    )


def _flash_attn_autograd_backward(ctx, dout, *grads):
    # grads对应LSE；仅公开attn_out参与求导。
    # 非可微辅助输出: 当前反向算子无 dLSE 入口, 显式报错
    if any(g is not None for g in grads):
        raise NotImplementedError(
            "flash_attn: gradient w.r.t. softmax_lse is not supported; "
            "only attn_out is differentiable (dq/dk/dv/dsink)."
        )
    (
        q,
        k,
        v,
        attn_out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
    ) = ctx.saved_tensors
    (
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
    ) = ctx.flash_attn_attrs
    dq, dk, dv, dsink = _flash_attn_backward(
        dout,
        q,
        k,
        v,
        attn_out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
    )
    d_sinks = dsink if sinks is not None else None
    # 梯度按 _flash_attn_forward 的输入序返回
    return (
        dq,
        dk,
        dv,  # q, k, v
        None,  # block_table
        None,
        None,
        None,
        None,  # cu_seqlens_q/kv, seqused_q/kv
        d_sinks,  # sinks
        None,
        None,  # attn_mask, metadata
        None,
        None,
        None,
        None,  # softmax_scale, mask_mode, win_left, win_right
        None,
        None,  # max_seqlen_q, max_seqlen_kv
        None,
        None,
        None,  # layout_q, layout_kv, layout_out
        None,
        None,
        None,
        None,  # return_softmax_lse, dropout_p, seed, offset
    )


_flash_attn_forward.register_autograd(
    _flash_attn_autograd_backward, setup_context=_flash_attn_autograd_setup_context
)


def _flash_attn_is_grad_enabled(q, k, v, sinks) -> bool:
    """训练态判定: grad 模式开启且存在需要梯度的输入(q/k/v/sinks)。"""
    if not torch.is_grad_enabled():
        return False
    return bool(
        q.requires_grad
        or k.requires_grad
        or v.requires_grad
        or (sinks is not None and sinks.requires_grad)
    )


@impl(get_as_library(), FA_METADATA_OP_NAME, "PrivateUse1")
def flash_attn_metadata(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    head_dim_v: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = -1,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    mask_mode: Optional[int] = 0,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
    layout_out: Optional[str] = "BSND",
    is_grad_enabled: Optional[bool] = False,
    dropout_p: float = 0.0,
):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    b_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q)

    batch_size = -1 if batch_size is None else batch_size
    head_dim_v = -1 if head_dim_v is None else head_dim_v
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    mask_mode = 1 if mask_mode is None else mask_mode
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "BSND" if layout_kv is None else layout_kv
    layout_out = "BSND" if layout_out is None else layout_out

    op_module = flash_attn_op_builder.load()
    metadata_size = _calculate_metadata_size(
        b_size, num_heads_kv, num_heads_q, max_seqlen_q, max_seqlen_kv
    )
    output = torch.empty((metadata_size,), dtype=torch.int32, device="npu")

    return op_module.flash_attn_metadata(
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
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
        is_grad_enabled,
        dropout_p,
        output,
    )


_flash_attn_metadata = flash_attn_metadata


@torch.library.register_kernel("cann_ops_transformer::" + FA_METADATA_OP_NAME, None)
def flash_attn_metadata_fallback(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    head_dim_v: Optional[int] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    mask_mode: Optional[int] = None,
    win_left: Optional[int] = None,
    win_right: Optional[int] = None,
    layout_q: Optional[str] = None,
    layout_kv: Optional[str] = None,
    layout_out: Optional[str] = None,
    is_grad_enabled: Optional[bool] = False,
    dropout_p: float = 0.0,
):
    # 处理所有 tensor 都为 None 的情况
    return _flash_attn_metadata(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_q=seqused_q,
        seqused_kv=seqused_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        is_grad_enabled=is_grad_enabled,
        dropout_p=dropout_p,
    )


@impl(get_as_library(), flash_attn_op_builder.name, "PrivateUse1")
@impl(get_as_library(), flash_attn_op_builder.name, "Autograd")
def flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = 1.0,
    mask_mode: Optional[int] = 0,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
    layout_out: Optional[str] = "BSND",
    return_softmax_lse: Optional[bool] = False,
    dropout_p: Optional[float] = 0.0,
    seed: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' / 'Autograd' 同一实现: 训练态(输入需要梯度)经 _flash_attn_forward 出图,
    反向由 _flash_attn_backward -> flash_attn_grad 计算 dq/dk/dv/dsink。
    """
    _validate_layout_dims(q, k, v, layout_q, layout_kv)
    _validate_kv_stride(k, v)
    grad_enabled = _flash_attn_is_grad_enabled(q, k, v, sinks)
    # 训练态强制 softmax_lse 开启(反向必需); 用户未请求时对外仍返回 {0} 占位
    attn_out, softmax_lse = _flash_attn_forward(
        q,
        k,
        v,
        block_table,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
        bool(return_softmax_lse) or grad_enabled,
        dropout_p,
        seed,
        offset,
    )
    if grad_enabled and not return_softmax_lse:
        softmax_lse = torch.empty((0,), dtype=torch.float32, device=q.device)
    return attn_out, softmax_lse


flash_attn = torch.ops.cann_ops_transformer.flash_attn
flash_attn_metadata = torch.ops.cann_ops_transformer.flash_attn_metadata
