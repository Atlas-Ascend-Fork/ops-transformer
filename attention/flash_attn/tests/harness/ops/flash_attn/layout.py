#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

from typing import Any, Dict, List

import torch
from einops import rearrange


def page_attn_for_bnsd(bnsd_tensor, b, act_seq_lens_kv, block_table, block_size):
    block_num = int(block_table.max()) + 1
    kv_cache_bnsd_shape = (
        block_num,
        bnsd_tensor.shape[1],
        block_size,
        bnsd_tensor.shape[3],
    )
    page_cache_tensor = torch.zeros(size=kv_cache_bnsd_shape, dtype=bnsd_tensor.dtype)
    actual_b = block_table.shape[0]
    is_tnd = b == 1 and actual_b > 1
    cum_kv_offset = 0
    for i in range(actual_b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > i else act_seq_lens_kv[0]
        b_block_num = (b_seq + block_size - 1) // block_size
        if is_tnd:
            batch_dim = 0
            seq_offset = cum_kv_offset
        else:
            batch_dim = i
            seq_offset = 0
        for j in range(b_block_num):
            start_idx = seq_offset + j * block_size
            end_idx = seq_offset + min((j + 1) * block_size, b_seq)
            actual_size = end_idx - start_idx
            slice_data = bnsd_tensor[batch_dim, :, start_idx:end_idx, :]
            page_cache_tensor[block_table[i][j], :, :actual_size, :] = slice_data
        cum_kv_offset += b_seq
    return page_cache_tensor


def dtype_sizeof(data_type):
    if data_type == "fp16" or data_type == "bf16":
        return 2
    elif data_type == "int8" or data_type == "fp8":
        return 1


def rearrange_by_block_table(
    bnsd_tensor, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype
):
    page_cache_tensor = page_attn_for_bnsd(
        bnsd_tensor, b, act_seq_lens_kv, block_table, block_size
    )
    if kv_storage_mode == "PA_BBND":
        return page_cache_tensor.permute(0, 2, 1, 3)
    elif kv_storage_mode == "PA_BNBD":
        return page_cache_tensor
    elif (
        kv_storage_mode == "PA_NZ"
    ):  # PA_NZ: shape=(NumBlocks, N_kv, BlockSize/16, D, 16)，分页注意力NZ格式
        blk_elem = 32 // dtype_sizeof(kv_dtype)
        page_cache_tensor = page_cache_tensor.reshape(
            page_cache_tensor.shape[0],
            page_cache_tensor.shape[1],
            page_cache_tensor.shape[2],
            page_cache_tensor.shape[3] // blk_elem,
            blk_elem,
        ).permute(0, 1, 3, 2, 4)
        return page_cache_tensor
    else:
        return None


def trans_bnsd_to_layout(tensor, layout_type, **kwargs):
    b = kwargs.get("B")
    block_size = kwargs.get("block_size")
    block_table = kwargs.get("block_table")
    seqused_kv = kwargs.get("seqused_kv")
    dtype = kwargs.get("Dtype", "bf16")

    if tensor is None:
        return None
    else:
        if layout_type == "BSH":
            return rearrange(tensor.clone(), "b n s d -> b s (n d)")
        elif layout_type == "SBH":
            return rearrange(tensor.clone(), "b n s d -> s b (n d)")
        elif layout_type == "BSND":
            return tensor.permute(0, 2, 1, 3).contiguous()
        elif layout_type == "TND":
            return tensor[0].permute(1, 0, 2)
        elif (
            layout_type == "PA_BBND"
            or layout_type == "PA_BNBD"
            or layout_type == "PA_NZ"
        ):
            return rearrange_by_block_table(
                tensor, block_table, block_size, b, seqused_kv, layout_type, dtype
            )
        return tensor.clone()


def pa_to_bnsd(t: torch.Tensor, layout: str, params: dict) -> list[torch.Tensor]:
    """PA 格式 → 按 batch 返回 ``(N2, seqlen, D)`` 列表，只分配实际长度。"""
    b = params.get("actual_b", params["B"])  # TND 时 B=1，用 actual_b
    n2 = params.get("N2", params.get("N1", 1))
    s2 = params["S2"]
    seq_kv = params["seqused_kv"]
    bs = params.get("block_size", 128)
    bt = params.get("block_table")
    if bt is None:
        return [t.clone()]

    if isinstance(bt, torch.Tensor):
        bt = bt.cpu().numpy()

    out = []
    for bi in range(b):
        slen = seq_kv[bi] if bi < len(seq_kv) else seq_kv[0]
        slen = max(0, min(int(slen), int(s2)))
        if layout == "PA_BBND":
            d = t.shape[-1]
            result = torch.zeros(n2, slen, d, dtype=t.dtype, device=t.device)
        elif layout == "PA_BNBD":
            d = t.shape[-1]
            result = torch.zeros(n2, slen, d, dtype=t.dtype, device=t.device)
        elif layout == "PA_NZ":
            d = t.shape[2] * t.shape[4]
            result = torch.zeros(n2, slen, d, dtype=t.dtype, device=t.device)
        else:
            return [t.clone()]

        nblk = (slen + bs - 1) // bs
        for bj in range(nblk):
            blk_id = bt[bi][bj] if bi < len(bt) and bj < len(bt[bi]) else bj
            if blk_id < 0 or blk_id >= t.shape[0]:
                continue
            start = bj * bs
            end = min(start + bs, slen)
            cnt = end - start
            if cnt <= 0:
                continue
            if layout == "PA_BBND":
                result[:, start:end, :] = t[blk_id, :cnt, :, :].permute(1, 0, 2)
            elif layout == "PA_BNBD":
                result[:, start:end, :] = t[blk_id, :, :cnt, :]
            else:
                block_data = t[blk_id, :, :, :cnt, :]
                block_data = block_data.permute(0, 2, 1, 3).reshape(n2, cnt, -1)
                result[:, start:end, :] = block_data
        out.append(result)
    return out


class PagedKVLoader:
    """按 logical KV 区间从 PA page 中取 K/V，不生成整段 dense。"""

    def __init__(self, tensor: torch.Tensor, layout: str, params: Dict[str, Any]):
        self.tensor = tensor
        self.layout = layout
        self.block_size = int(params.get("block_size", 128) or 128)
        self.n2 = int(params.get("N2", params.get("N1", 1)) or 1)
        self.pages = int(tensor.shape[0])
        if layout == "PA_BBND":
            self.d = int(tensor.shape[-1])
        elif layout == "PA_BNBD":
            self.d = int(tensor.shape[-1])
        elif layout == "PA_NZ":
            self.d = int(tensor.shape[2] * tensor.shape[4])
        else:
            raise ValueError(f"unsupported PA layout: {layout}")
        block_table = params.get("block_table")
        if block_table is None:
            raise ValueError("PA requires block_table")
        if torch.is_tensor(block_table):
            block_table = block_table.cpu().tolist()
        self.block_table: List[List[int]] = block_table
        used = params.get("seqused_kv")
        if used is None:
            used = [int(params.get("S2", 0))] * len(self.block_table)
        if (
            isinstance(used, (list, tuple))
            and used
            and isinstance(used[0], (list, tuple))
        ):
            used = used[0]
        self.seqused = [int(value) for value in used]

    def num_sequences(self) -> int:
        return len(self.block_table)

    def _seq_len(self, seq: int) -> int:
        if not self.seqused:
            return 0
        return self.seqused[seq] if seq < len(self.seqused) else self.seqused[0]

    def load(self, seq: int, start: int, count: int) -> torch.Tensor:
        """返回 (N2, count, D) 的逻辑 KV block；越界部分截断。"""
        start = max(0, int(start))
        remain = min(int(count), self._seq_len(seq) - start)
        if remain <= 0:
            return torch.empty(
                (self.n2, 0, self.d),
                dtype=self.tensor.dtype,
                device=self.tensor.device,
            )

        parts = []
        while remain > 0:
            block_idx = start // self.block_size
            offset = start % self.block_size
            rows = min(self.block_size - offset, remain)
            if block_idx >= len(self.block_table[seq]):
                raise ValueError(f"PA block_table[{seq}] out of range")
            page_id = int(self.block_table[seq][block_idx])
            if page_id < 0 or page_id >= self.pages:
                raise ValueError(f"invalid PA page id {page_id} for seq {seq}")
            page = self.tensor[page_id]
            if self.layout == "PA_BBND":
                part = page[offset : offset + rows].permute(1, 0, 2)
            elif self.layout == "PA_BNBD":
                part = page[:, offset : offset + rows, :]
            else:
                part = page[:, :, offset : offset + rows, :]
                part = part.permute(0, 2, 1, 3).reshape(self.n2, rows, -1)
            parts.append(part)
            start += rows
            remain -= rows

        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=1)
