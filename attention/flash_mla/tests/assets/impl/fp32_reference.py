# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FP32 attention reference for three-way comparison."""

import torch

from .golden import FlashMlaGolden


def fp32_attention_single(
    q, k, v, scale, tile_size, mask_fn, sink=None, kv_bounds=None
):
    rows, length = q.shape[0], k.shape[0]
    out = torch.zeros((rows, v.shape[-1]), dtype=torch.float32, device=q.device)
    lse = torch.full((rows,), float("inf"), dtype=torch.float32, device=q.device)
    for qs in range(0, rows, tile_size):
        qe = min(qs + tile_size, rows)
        maximum = torch.full(
            (qe - qs, 1), float("-inf"), dtype=torch.float32, device=q.device
        )
        total = torch.zeros_like(maximum)
        if sink is not None:
            maximum.copy_(sink[qs:qe].reshape(-1, 1))
            total.fill_(1.0)
        acc = torch.zeros((qe - qs, v.shape[-1]), dtype=torch.float32, device=q.device)
        first, stop = (0, length) if kv_bounds is None else kv_bounds(qs, qe)
        first = max(0, first // tile_size * tile_size)
        stop = min(length, max(0, stop))
        for ks in range(first, stop, tile_size):
            ke = min(ks + tile_size, length)
            scores = (q[qs:qe] @ k[ks:ke].T) * scale
            mask = mask_fn(qs, qe, ks, ke)
            if mask is not None:
                scores.masked_fill_(mask, float("-inf"))
            new_max = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
            safe_max = torch.where(torch.isneginf(new_max), 0.0, new_max)
            correction = (maximum - safe_max).exp()
            weights = (scores - safe_max).exp()
            total = total * correction + weights.sum(dim=-1, keepdim=True)
            acc = acc * correction + weights @ v[ks:ke]
            maximum = new_max
        empty = total == 0
        out[qs:qe] = acc / torch.where(empty, 1.0, total)
        row_lse = total.log() + maximum
        lse[qs:qe] = torch.where(empty, float("inf"), row_lse).squeeze(-1)
    return out, lse


class FP32Reference(FlashMlaGolden):
    TILE_SIZE = 256
    TILED_CROSSOVER = 0
    _attention_single = staticmethod(fp32_attention_single)

    def forward_golden(self, q, k_cache, **kwargs):
        if q.device.type != "cpu" or k_cache.device.type != "cpu":
            raise ValueError("FP32 reference requires CPU tensors")
        return super().forward_golden(q.float(), k_cache.float(), **kwargs)
