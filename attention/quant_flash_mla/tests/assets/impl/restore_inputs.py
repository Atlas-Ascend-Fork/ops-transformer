#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Recover logical reference inputs from the tensors actually passed by TTK.

These are CPU reference materialization checks, not operator input validation.
Validation-only CSV cases must use TTK golden_api=disable, never synthesized
replacement tensors or an invented numerical reference.
"""

import numpy as np
import torch


def zeros(shape, dtype):
    if dtype == torch.float8_e4m3fn:
        return torch.zeros(shape, dtype=torch.uint8).view(dtype)
    return torch.zeros(shape, dtype=dtype)


def as_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    array = np.asarray(value)
    if "float8" in str(array.dtype) and array.dtype.itemsize == 1:
        return torch.from_numpy(array.view(np.uint8)).view(torch.float8_e4m3fn)
    return torch.as_tensor(array)


def vector(value, name, optional=False):
    if value is None:
        if optional:
            return None
        raise ValueError(
            f"[GOLDEN] cannot read missing {name}; validation-only cases need golden_api=disable"
        )
    tensor = as_cpu(value)
    if tensor.numel() == 0 and optional:
        return None  # same empty optional slot convention as the Torch entry
    if tensor.ndim != 1:
        raise ValueError(
            f"[GOLDEN] cannot interpret {name} shape {tuple(tensor.shape)} as a vector"
        )
    values = tensor.tolist()
    if any(int(item) != item for item in values):
        raise ValueError(f"[GOLDEN] cannot interpret non-integral {name}")
    return [int(item) for item in values]


def to_bnsd(tensor, layout, allocations, cu, name):
    if layout == "BNSD":
        if tensor.ndim != 4:
            raise ValueError(
                f"[GOLDEN] cannot unpack {name} as BNSD: {tuple(tensor.shape)}"
            )
        return tensor.contiguous()
    if layout == "BSND":
        if tensor.ndim != 4:
            raise ValueError(
                f"[GOLDEN] cannot unpack {name} as BSND: {tuple(tensor.shape)}"
            )
        return tensor.permute(0, 2, 1, 3).contiguous()
    if layout != "TND" or tensor.ndim != 3:
        raise ValueError(
            f"[GOLDEN] cannot unpack {name} layout={layout}, shape={tuple(tensor.shape)}"
        )
    result = zeros(
        (
            len(allocations),
            tensor.shape[1],
            max(1, max(allocations, default=0)),
            tensor.shape[2],
        ),
        tensor.dtype,
    )
    for batch, length in enumerate(allocations):
        result[batch, :, :length] = tensor[cu[batch] : cu[batch + 1]].permute(1, 0, 2)
    return result.contiguous()


def restore_inputs(
    q,
    k_cache,
    q_descale,
    k_descale,
    block_table,
    cache_seqlens,
    cu_seqlens_q,
    seqused_q,
    layout_q,
    layout_kv,
    layout_out,
    head_dim_v,
    quant_mode,
):
    q = as_cpu(q)
    cache = as_cpu(k_cache)
    cache_lengths = vector(cache_seqlens, "cache_seqlens")
    batch_size = len(cache_lengths)
    if not batch_size or any(length < 0 for length in cache_lengths):
        raise ValueError(
            "[GOLDEN] no numerical reference for empty/negative cache lengths; use golden_api=disable"
        )
    cu = vector(cu_seqlens_q, "cu_seqlens_q", optional=True)
    used = vector(seqused_q, "seqused_q", optional=True)
    if layout_q == "TND":
        if (
            q.ndim != 3
            or cu is None
            or len(cu) != batch_size + 1
            or cu[0] != 0
            or any(a > b for a, b in zip(cu, cu[1:]))
            or cu[-1] > q.shape[0]
        ):
            raise ValueError(
                "[GOLDEN] cannot recover TND offsets from the actual cu/q tensors; use golden_api=disable"
            )
        allocations = [b - a for a, b in zip(cu, cu[1:])]
    elif layout_q in ("BSND", "BNSD") and q.ndim == 4 and q.shape[0] == batch_size:
        allocations = [q.shape[1 if layout_q == "BSND" else 2]] * batch_size
        # cu is not used in dense layouts. Operator checks prohibited arguments.
        cu = None
    else:
        raise ValueError(
            f"[GOLDEN] cannot unpack actual q with layout {layout_q}: {tuple(q.shape)}"
        )
    if used is None:
        used = list(allocations)
    if len(used) != batch_size or any(
        length < 0 or length > allocations[i] for i, length in enumerate(used)
    ):
        raise ValueError(
            "[GOLDEN] actual seqused cannot describe valid Q rows; use golden_api=disable"
        )
    q_bnsd = to_bnsd(q, layout_q, allocations, cu, "q")
    q_scale = as_cpu(q_descale)
    if tuple(q_scale.shape) != tuple(q.shape[:-1]):
        raise ValueError(
            "[GOLDEN] actual q_descale cannot be mapped one-to-one to q rows; use golden_api=disable"
        )
    q_scale = to_bnsd(
        q_scale.unsqueeze(-1), layout_q, allocations, cu, "q_descale"
    ).float()
    if layout_kv == "PA_BNBD" and cache.ndim == 4:
        cache_bn = cache
    elif layout_kv == "PA_BBND" and cache.ndim == 4:
        cache_bn = cache.permute(0, 2, 1, 3).contiguous()
    elif layout_kv == "PA_NZ" and cache.ndim == 5:
        cache_bn = cache.permute(0, 1, 3, 2, 4).contiguous().flatten(3)
    else:
        raise ValueError(
            f"[GOLDEN] cannot unpack actual cache layout={layout_kv}, shape={tuple(cache.shape)}"
        )
    table = as_cpu(block_table)
    block_size = cache_bn.shape[2]
    if table.ndim != 2 or table.shape[0] != batch_size or block_size <= 0:
        raise ValueError(
            "[GOLDEN] actual block_table/cache cannot describe a paged input; use golden_api=disable"
        )
    k_bnsd = zeros(
        (batch_size, cache_bn.shape[1], max(cache_lengths), cache_bn.shape[3]),
        cache_bn.dtype,
    )
    for batch, length in enumerate(cache_lengths):
        blocks = (length + block_size - 1) // block_size
        if blocks > table.shape[1]:
            raise ValueError(
                "[GOLDEN] actual block_table lacks pages for cache_seqlens; use golden_api=disable"
            )
        for logical_block in range(blocks):
            value = table[batch, logical_block].item()
            block_id = int(value)
            if block_id != value or not 0 <= block_id < cache_bn.shape[0]:
                raise ValueError(
                    "[GOLDEN] cannot dereference actual block id; use golden_api=disable"
                )
            start = logical_block * block_size
            count = min(block_size, length - start)
            k_bnsd[batch, :, start : start + count] = cache_bn[block_id, :, :count]
    if q_bnsd.shape[-1] != k_bnsd.shape[-1] or not 0 < head_dim_v <= k_bnsd.shape[-1]:
        raise ValueError(
            "[GOLDEN] actual Q/K/V widths cannot form an attention reference; use golden_api=disable"
        )
    # Preserve HIF8 byte encoding. CPU reference decodes it exactly once.
    if quant_mode == 0:
        q_bnsd = (
            q_bnsd.view(torch.uint8) if q_bnsd.dtype == torch.float8_e4m3fn else q_bnsd
        )
        k_bnsd = (
            k_bnsd.view(torch.uint8) if k_bnsd.dtype == torch.float8_e4m3fn else k_bnsd
        )
        if q_bnsd.dtype != torch.uint8 or k_bnsd.dtype != torch.uint8:
            raise ValueError(
                "[GOLDEN] HIF8 reference requires actual byte-encoded inputs"
            )
    k_scale = as_cpu(k_descale).float()
    if k_scale.numel() != 1:
        raise ValueError(
            "[GOLDEN] actual k_descale cannot describe a shared per-tensor scale; use golden_api=disable"
        )
    k_scale = k_scale.reshape(1)
    cfg = {
        "B": batch_size,
        "N_q": q_bnsd.shape[1],
        "N_kv": k_bnsd.shape[1],
        "D": q_bnsd.shape[-1],
        "D_V": head_dim_v,
        "block_size": block_size,
        "seqused_q_vals": used,
        "cache_seqlens_vals": cache_lengths,
        "cu_seqlens_q_vals": cu,
        "input_layout": layout_q,
        "output_layout": layout_out,
        "kv_cache_layout": {"PA_BNBD": "BnNBsD", "PA_BBND": "BnBsND", "PA_NZ": "NZ"}[
            layout_kv
        ],
    }
    tensors = (
        q_bnsd,
        k_bnsd,
        k_bnsd[..., :head_dim_v].contiguous(),
        q_scale,
        k_scale,
        k_scale,
    )
    return cfg, tensors
