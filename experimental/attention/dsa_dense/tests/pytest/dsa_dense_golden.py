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


def _gen_seq_len(batch, t1, seed):
    rng = np.random.RandomState(seed)
    random_numbers = rng.rand(batch)
    normalized = random_numbers / random_numbers.sum()
    q_len_tmp = [int(t1 * normalized[i]) for i in range(batch)]
    q_len = [1 if q_len_tmp[i] == 0 else q_len_tmp[i] for i in range(batch)]
    kv_len = [q_len[i] + int(rng.randint(100, 2000)) for i in range(batch)]
    return q_len, kv_len


def _compute_golden(
    q,
    k,
    q_index,
    k_index,
    weights,
    mask_list,
    actual_q_len,
    actual_kv_len,
    cu_q_len,
    cu_kv_len,
    batch,
    n,
    ni,
    di,
    scale,
    device,
):
    final_main_max = []
    final_main_sum = []
    final_index_max = []
    final_index_sum = []
    dw_batch_list = []
    dq_batch_list = []
    dk_batch_list = []
    gt_loss = torch.zeros(1, dtype=torch.float, device=device)

    for b in range(batch):
        actual_s1 = actual_q_len[b]
        actual_s2 = actual_kv_len[b]
        start_s1 = cu_q_len[b]
        start_s2 = cu_kv_len[b]

        # main attention softmax
        actual_q = q[start_s1 : start_s1 + actual_s1, :, :].permute(1, 0, 2)  # NSD
        actual_k = k[start_s2 : start_s2 + actual_s2, :, :].permute(1, 2, 0)  # NDS
        p_list = []
        main_max_list = []
        main_sum_list = []
        for head in range(n):
            p = actual_q[head] @ actual_k[head]
            p = p * scale
            p = p.masked_fill(mask_list[b].to(torch.bool), -100000000.0)
            main_max, _ = torch.max(p, dim=1, keepdim=True)
            main_sum = torch.sum(torch.exp(p - main_max), dim=1, keepdim=True)
            main_max_list.append(main_max)
            main_sum_list.append(main_sum)
            p = torch.exp(p - main_max) / main_sum
            p_list.append(p.unsqueeze(0))
        p = torch.cat(p_list, dim=0)
        p = torch.sum(p, dim=0, keepdim=False)
        p = p / n
        main_max = torch.cat(main_max_list, dim=1)
        main_sum = torch.cat(main_sum_list, dim=1)
        final_main_max.append(main_max)
        final_main_sum.append(main_sum)

        # indexer softmax
        actual_q_index = q_index[start_s1 : start_s1 + actual_s1, :, :].permute(
            1, 0, 2
        )  # NI S1 DI
        actual_k_index = k_index[start_s2 : start_s2 + actual_s2, :].permute(
            1, 0
        )  # DI S2
        actual_weights = weights[start_s1 : start_s1 + actual_s1, :]
        y_list = []
        relu_list = []
        for head in range(ni):
            y = actual_q_index[head] @ actual_k_index
            y = torch.relu(y)
            relu_list.append(y)
            y = y * (actual_weights[:, head].view(actual_s1, 1))
            y_list.append(y.unsqueeze(0))
        y = torch.cat(y_list, dim=0)
        y = torch.sum(y, dim=0, keepdim=False)
        y = y.masked_fill(mask_list[b].to(torch.bool), -100000000.0)
        index_max, _ = torch.max(y, dim=1, keepdim=True)
        index_sum = torch.sum(torch.exp(y - index_max), dim=1, keepdim=True)
        final_index_max.append(index_max)
        final_index_sum.append(index_sum)
        y = torch.exp(y - index_max) / index_sum

        # gradients
        di = y - p
        dw_list = []
        dq_list = []
        dk_list = []
        for head in range(ni):
            dw = di * relu_list[head]
            dw = torch.sum(dw, dim=1, keepdim=False)
            dw_list.append(dw.unsqueeze(0))

            ds_tmp = di * (actual_weights[:, head].view(actual_s1, 1))
            ds = torch.where(relu_list[head] > 0, ds_tmp, torch.zeros_like(ds_tmp))

            dq = ds @ actual_k_index.T
            dq = dq.unsqueeze(1)
            dq_list.append(dq)

            dk = ds.T @ actual_q_index[head]
            dk = dk.unsqueeze(0)
            dk_list.append(dk)

        dw = torch.cat(dw_list, dim=0)
        dw_batch_list.append(dw)
        dq = torch.cat(dq_list, dim=1)
        dq_batch_list.append(dq)
        dk = torch.cat(dk_list, dim=0)
        dk_batch_list.append(dk)

        # KL loss
        loss = torch.clip(p, min=1e-8) / torch.clip(y, min=1e-8)
        loss = torch.log(loss)
        loss = p * loss
        gt_loss += torch.sum(loss)

    gt_dw = torch.cat(dw_batch_list, dim=1).t().contiguous()  # T1 NI
    gt_dq = torch.cat(dq_batch_list, dim=0)  # T1 NI DI
    gt_dk = torch.cat(dk_batch_list, dim=1)  # NI T2 DI
    gt_dk = torch.sum(gt_dk, dim=0, keepdim=False)  # T2 DI

    main_max = torch.cat(final_main_max, dim=0).permute(1, 0)  # N T1
    main_sum = torch.cat(final_main_sum, dim=0).permute(1, 0)  # N T1
    index_max = torch.cat(final_index_max, dim=0).view(-1)  # T1
    index_sum = torch.cat(final_index_sum, dim=0).view(-1)  # T1

    return gt_loss, gt_dq, gt_dk, gt_dw, main_max, main_sum, index_max, index_sum


def gen_data(params):
    """根据参数生成算子输入、属性与 CPU golden 结果（TND 变长，DSA dense 反向）。"""
    batch, t1_base, n, case_name = params
    device = torch.device("cpu")
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    actual_q_len, actual_kv_len = _gen_seq_len(batch, t1_base, seed)
    t1 = int(np.sum(actual_q_len))
    t2 = int(np.sum(actual_kv_len))
    ni = n // 2
    d = 192
    di = 128
    scale = 1.0 / math.sqrt(d)

    cu_q_len = [0] + list(np.cumsum(actual_q_len))
    cu_kv_len = [0] + list(np.cumsum(actual_kv_len))

    q = torch.randn(t1, n, d, device=device, dtype=torch.bfloat16).to(torch.float)
    k = torch.randn(t2, n, d, device=device, dtype=torch.bfloat16).to(torch.float)
    q_index = torch.randn(t1, ni, di, device=device, dtype=torch.bfloat16).to(
        torch.float
    )
    k_index = torch.randn(t2, di, device=device, dtype=torch.bfloat16).to(torch.float)
    weights = torch.randn(t1, ni, device=device, dtype=torch.float)

    mask_list = []
    for b in range(batch):
        actual_s1 = actual_q_len[b]
        actual_s2 = actual_kv_len[b]
        atten_mask_u = torch.triu(
            torch.ones([actual_s1, actual_s2], dtype=torch.uint8),
            diagonal=(actual_s2 - actual_s1) + 1,
        ).to(device)
        mask_list.append(atten_mask_u)

    gt_loss, gt_dq, gt_dk, gt_dw, main_max, main_sum, index_max, index_sum = (
        _compute_golden(
            q,
            k,
            q_index,
            k_index,
            weights,
            mask_list,
            actual_q_len,
            actual_kv_len,
            cu_q_len,
            cu_kv_len,
            batch,
            n,
            ni,
            di,
            scale,
            device,
        )
    )

    input_data = {
        "input": {
            "query": q.to(torch.bfloat16),
            "key": k.to(torch.bfloat16),
            "max": main_max.contiguous(),
            "sum": main_sum.contiguous(),
            "q_len": torch.tensor(cu_q_len, dtype=torch.int64, device=device),
            "kv_len": torch.tensor(cu_kv_len, dtype=torch.int64, device=device),
            "query_index": q_index.to(torch.bfloat16),
            "key_index": k_index.to(torch.bfloat16),
            "weights": weights,
            "max_index": index_max.contiguous(),
            "sum_index": index_sum.contiguous(),
        },
        "attr": {
            "sparse_mode": 1,
            "scale": scale,
        },
        "cpu_output": {
            "loss": gt_loss,
            "dq": gt_dq,
            "dk": gt_dk,
            "dw": gt_dw,
        },
        "case_name": case_name,
    }
    return input_data
