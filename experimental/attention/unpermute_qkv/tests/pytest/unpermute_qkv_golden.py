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


def torch_unpermute(inp, matrix, spg, s_c, b, h, q_head, kv_head):
    """复刻源 test torch_unpermute 参考实现，返回 (q, k, v)。"""
    total = q_head + 2 * kv_head
    buffer = inp.view(spg, s_c, b, total // spg, h)
    # 'c s_c b n h -> (c n) s_c b h'
    head_first = buffer.permute(0, 3, 1, 2, 4).reshape(total, s_c, b, h).contiguous()
    shape = head_first.shape
    flat = head_first.view(total, -1)
    unperm = torch.matmul(matrix, flat).view(shape)
    # 'n s_c b h -> s_c b n h'
    rearranged = unperm.permute(1, 2, 0, 3).contiguous()
    q, k, v = torch.split(rearranged, [q_head, kv_head, kv_head], dim=2)
    return q.contiguous(), k.contiguous(), v.contiguous()


def gen_inputs(params):
    """params: (s_c, b, kv_group, split_size, h, n_k, dtype, case_name)"""
    s_c, b, kv_group, split_size, h, n_k, dtype, case_name = params
    torch.manual_seed(0)
    device = torch.device("cpu")

    q_head = n_k * kv_group
    kv_head = n_k
    total = q_head + 2 * kv_head
    in_shape = [split_size, s_c, b, total // split_size, h]
    inp = torch.randn(*in_shape, dtype=torch.bfloat16, device=device)

    matrix = torch.tensor(
        build_permutation_matrix(q_head, kv_head, split_size, True),
        dtype=inp.dtype,
        device=device,
    )
    gq, gk, gv = torch_unpermute(inp, matrix, split_size, s_c, b, h, q_head, kv_head)

    return {
        "input": {"x": inp},
        "attr": {"n_k": n_k},
        "cpu_output": {"q": gq, "k": gk, "v": gv},
        "case_name": case_name,
    }
