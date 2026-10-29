#!/usr/bin/python
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

import math

import numpy as np
import torch


def _get_tnd_idx(acc_q_len, t_idx):
    b_idx = 0
    while t_idx >= acc_q_len[b_idx + 1]:
        b_idx += 1
    s1_idx = t_idx - acc_q_len[b_idx]
    return b_idx, s1_idx


def _randperm(n, device):
    if n <= 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    return torch.randperm(n, device=device).to(torch.int32)


def _gen_topk_indices(
    acc_seqlen_q,
    cur_seqlen_q,
    cur_seqlen_kv,
    block_size,
    block_count,
    n2,
    device,
    seed=42,
):
    """按 TND 变长规则生成随机的 block 级 topk 索引（无效位置为 -1）。"""
    torch.manual_seed(seed)
    t1 = int(acc_seqlen_q[-1])
    topk_indices = torch.full(
        (t1, n2, block_count), -1, dtype=torch.int32, device=device
    )
    for t in range(t1):
        b_idx, s1_idx = _get_tnd_idx(acc_seqlen_q, t)
        cur_s1 = cur_seqlen_q[b_idx]
        cur_s2 = cur_seqlen_kv[b_idx]
        delta_s = cur_s2 - cur_s1
        threshold = delta_s + s1_idx + 1
        threshold = threshold if threshold >= 0 else 0
        valid_block_count = math.ceil(threshold / block_size)
        for j in range(n2):
            if valid_block_count >= block_count:
                topk_indices[t][j][-1] = valid_block_count - 1
                topk_indices[t][j][0:-1] = _randperm(valid_block_count - 1, device)[
                    : block_count - 1
                ]
            else:
                topk_indices[t][j][valid_block_count - 1] = valid_block_count - 1
                topk_indices[t][j][0 : valid_block_count - 1] = _randperm(
                    valid_block_count - 1, device
                )
    return topk_indices


def _cpu_golden(
    topk_indices, acc_seqlen_q, cur_seqlen_q, cur_seqlen_kv, block_size, block_count
):
    """CPU 侧 golden：将 block 级索引展开为逐 token 索引，并处理尾块与无效位置（-1）。"""
    t1, n2, _ = topk_indices.shape
    expanded = topk_indices.unsqueeze(-1) * block_size + torch.arange(
        block_size, dtype=torch.int32, device=topk_indices.device
    )
    mask = (topk_indices <= -1).unsqueeze(-1)
    fill_value = torch.full_like(expanded, -1)
    golden = torch.where(mask, fill_value, expanded).view(t1, n2, -1)

    for t in range(t1):
        b_idx, s1_idx = _get_tnd_idx(acc_seqlen_q, t)
        cur_s1 = cur_seqlen_q[b_idx]
        cur_s2 = cur_seqlen_kv[b_idx]
        threshold = cur_s2 - cur_s1 + 1 + s1_idx
        valid_block_count = math.ceil(threshold / block_size)
        mask_size = valid_block_count * block_size - threshold
        if valid_block_count > block_count:
            if mask_size != 0:
                golden[t, :, -mask_size:] = -1
        else:
            if mask_size != 0:
                end_pos = valid_block_count * block_size
                golden[t, :, end_pos - mask_size : end_pos] = -1
    return golden


def gen_data(params):
    """根据参数生成算子输入、属性与 CPU golden 结果。"""
    n2, cur_seqlen_q, cur_seqlen_kv, block_size, block_count, causal, case_name = params
    device = torch.device("cpu")

    acc_seqlen_q = np.cumsum([0] + list(cur_seqlen_q)).tolist()
    acc_seqlen_kv = np.cumsum([0] + list(cur_seqlen_kv)).tolist()

    topk_indices = _gen_topk_indices(
        acc_seqlen_q, cur_seqlen_q, cur_seqlen_kv, block_size, block_count, n2, device
    )
    cpu_output = _cpu_golden(
        topk_indices, acc_seqlen_q, cur_seqlen_q, cur_seqlen_kv, block_size, block_count
    )

    input_data = {
        "input": {
            "topk_indices": topk_indices,
            "acc_seqlen_q": torch.tensor(
                acc_seqlen_q, dtype=torch.int64, device=device
            ),
            "acc_seqlen_kv": torch.tensor(
                acc_seqlen_kv, dtype=torch.int64, device=device
            ),
        },
        "attr": {
            "causal": causal,
            "block_size": block_size,
        },
        "cpu_output": cpu_output,
        "case_name": case_name,
    }
    return input_data
