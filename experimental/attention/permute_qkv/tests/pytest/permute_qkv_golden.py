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

import numpy as np
import torch


def build_permutation_matrix(q_num, kv_num, spg_size, reverse=False):
    """构建 QKV 头重排列矩阵（复刻源 test）。"""
    total = q_num + 2 * kv_num
    matrix = np.zeros((total, total), dtype=int)
    q_per = q_num // spg_size
    kv_per = kv_num // spg_size

    for g in range(spg_size):
        for offset in range(q_per):
            src = g * q_per + offset
            tgt = (g * (q_per + 2 * kv_per)) + offset
            if not reverse:
                matrix[tgt, src] = 1
            else:
                matrix[src, tgt] = 1

        k_src_base = q_num + g * kv_per
        k_tgt_base = g * (q_per + 2 * kv_per) + q_per
        for offset in range(kv_per):
            src = k_src_base + offset
            tgt = k_tgt_base + offset
            if not reverse:
                matrix[tgt, src] = 1
            else:
                matrix[src, tgt] = 1

        v_src_base = q_num + kv_num + g * kv_per
        v_tgt_base = g * (q_per + 2 * kv_per) + q_per + kv_per
        for offset in range(kv_per):
            src = v_src_base + offset
            tgt = v_tgt_base + offset
            if not reverse:
                matrix[tgt, src] = 1
            else:
                matrix[src, tgt] = 1
    return matrix


def torch_permute(query, key, value, permute_matrix, spg_size):
    """复刻源 test torch_permute 参考实现。"""
    q_seq_len, b, q_head_num, head_dim = query.shape
    kv_head_num = key.shape[2]
    total_qkv_head_num = q_head_num + kv_head_num * 2

    concated_qkv = torch.cat([query, key, value], dim=2)
    permuted = concated_qkv.permute(2, 0, 1, 3).contiguous()  # n s_c b h
    permuted = permuted.view(total_qkv_head_num, -1)
    permuted_qkv = torch.matmul(permute_matrix, permuted).view(
        spg_size, total_qkv_head_num // spg_size, -1
    )
    return permuted_qkv


def gen_inputs(params):
    """params: (s_c, b, split_size, kv_group, n_k, h, dtype, case_name)"""
    s_c, b, split_size, kv_group, n_k, h, dtype, case_name = params
    torch.manual_seed(0)
    device = torch.device("cpu")

    q_head_num = kv_group * n_k
    kv_head_num = n_k

    query = torch.randn(s_c, b, q_head_num, h, dtype=torch.bfloat16, device=device)
    key = torch.randn(s_c, b, kv_head_num, h, dtype=torch.bfloat16, device=device)
    value = torch.randn(s_c, b, kv_head_num, h, dtype=torch.bfloat16, device=device)

    matrix = torch.tensor(
        build_permutation_matrix(q_head_num, kv_head_num, split_size),
        dtype=torch.bfloat16,
        device=device,
    )
    golden = torch_permute(query, key, value, matrix, split_size)

    return {
        "input": {"query": query, "key": key, "value": value},
        "attr": {"split_size": split_size},
        "cpu_output": golden,
        "case_name": case_name,
    }
