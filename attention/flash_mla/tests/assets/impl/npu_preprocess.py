# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Materialize metadata after H2D, before timing and graph capture."""

import torch


METADATA_SLOT_INDEX = 8


def build_metadata(
    q,
    k_cache,
    *,
    cache_seqlens,
    cu_seqlens_q=None,
    seqused_q=None,
    head_dim_v=512,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
):
    # Keep device initialization out of TTK's parent process and CPU prepare.
    import cann_ops_transformer  # noqa: F401

    if cache_seqlens is None:
        raise ValueError("cache_seqlens is required for MLA metadata")
    if layout_q not in ("TND", "BNSD", "BSND"):
        raise ValueError(f"unsupported layout_q: {layout_q}")
    if layout_kv not in ("PA_BBND", "PA_BNBD", "PA_NZ"):
        raise ValueError(f"unsupported layout_kv: {layout_kv}")
    return torch.ops.cann_ops_transformer.flash_mla_metadata(
        cache_seqlens,
        int(q.shape[1 if layout_q in ("TND", "BNSD") else 2]),
        int(k_cache.shape[2 if layout_kv == "PA_BBND" else 1]),
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim_qk=int(q.shape[-1]),
        head_dim_v=int(head_dim_v),
        mask_mode=int(mask_mode),
        win_left=int(win_left),
        win_right=int(win_right),
        layout_q=layout_q,
    ).to(device=q.device)


def metadata_slot_declared(tensor_dtypes):
    if not isinstance(tensor_dtypes, (tuple, list)) or len(tensor_dtypes) != 9:
        raise ValueError("tensor_dtypes must declare all 9 FlashMla input slots")
    return tensor_dtypes[METADATA_SLOT_INDEX] is not None


def run(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
    **kwargs,
):
    if metadata is not None:
        return None
    if not metadata_slot_declared(kwargs["tensor_dtypes"]):
        return None
    generated = build_metadata(
        q,
        k_cache,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        head_dim_v=head_dim_v,
        mask_mode=mask_mode,
        win_left=win_left,
        win_right=win_right,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )
    return {"metadata": generated}
