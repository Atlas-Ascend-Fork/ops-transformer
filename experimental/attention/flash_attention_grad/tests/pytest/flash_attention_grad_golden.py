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


def _fa_backward_golden(query, key, value, dy, cu_q, cu_kv, scale, causal):
    """变长 TND 注意力反向 golden，返回 (attention_out, lse[N,T], dq, dk, dv)。"""
    d = query.shape[-1]
    dq = torch.zeros_like(query)
    dk = torch.zeros_like(key)
    dv = torch.zeros_like(value)
    attn_parts = []
    lse_parts = []
    for b in range(len(cu_q) - 1):
        s1, e1 = cu_q[b], cu_q[b + 1]
        s2, e2 = cu_kv[b], cu_kv[b + 1]
        qb, kb, vb = query[s1:e1].float(), key[s2:e2].float(), value[s2:e2].float()
        dyb = dy[s1:e1].float()
        l1, l2 = e1 - s1, e2 - s2
        sc = torch.einsum("lnd,mnd->lnm", qb, kb) * scale
        if causal:
            mask = torch.triu(
                torch.ones(l1, l2, device=sc.device), diagonal=(l2 - l1) + 1
            ).bool()
            sc = sc.masked_fill(mask[:, None, :], float("-inf"))
        p = torch.softmax(sc, dim=-1)
        attn_parts.append(torch.einsum("lnm,mnd->lnd", p, vb))
        lse_parts.append(torch.logsumexp(sc, dim=-1))

        dv[s2:e2] = torch.einsum("lnm,lnd->mnd", p, dyb)
        dp = torch.einsum("lnd,mnd->lnm", dyb, vb)
        row = torch.sum(dp * p, dim=-1, keepdim=True)
        ds = p * (dp - row)
        dq[s1:e1] = torch.einsum("lnm,mnd->lnd", ds, kb) * scale
        dk[s2:e2] = torch.einsum("lnm,lnd->mnd", ds, qb) * scale
    attn = torch.cat(attn_parts, dim=0)
    lse = torch.cat(lse_parts, dim=0).transpose(0, 1).contiguous()
    return attn, lse, dq, dk, dv


def gen_inputs(params):
    """生成 CPU 侧输入与 golden。

    params: (seq_lens_q, seq_lens_kv, n, d, causal, dtype, case_name)
    """
    seq_lens_q, seq_lens_kv, n, d, causal, dtype, case_name = params
    torch.manual_seed(0)
    device = torch.device("cpu")

    cu_q = [0] + list(np.cumsum(seq_lens_q))
    cu_kv = [0] + list(np.cumsum(seq_lens_kv))
    t1, t2 = int(cu_q[-1]), int(cu_kv[-1])
    scale = 1.0 / math.sqrt(d)

    query = torch.randn(t1, n, d, dtype=torch.bfloat16, device=device).to(torch.float)
    key = torch.randn(t2, n, d, dtype=torch.bfloat16, device=device).to(torch.float)
    value = torch.randn(t2, n, d, dtype=torch.bfloat16, device=device).to(torch.float)
    dy = torch.randn(t1, n, d, dtype=torch.bfloat16, device=device).to(torch.float)

    attn, lse, dq, dk, dv = _fa_backward_golden(
        query, key, value, dy, cu_q, cu_kv, scale, causal
    )

    return {
        "input": {
            "query": query.to(dtype),
            "key": key.to(dtype),
            "value": value.to(dtype),
            "dy": dy.to(dtype),
            "attention_out": attn.to(dtype),
            "softmax_lse": lse,
            "cu_q": torch.tensor(cu_q, dtype=torch.int64),
            "cu_kv": torch.tensor(cu_kv, dtype=torch.int64),
            "max_q": max(seq_lens_q),
            "max_kv": max(seq_lens_kv),
        },
        "attr": {
            "scale": scale,
            "causal": causal,
            "window_size": [-1, -1],
            "dropout_p": 0.0,
            "cu_q_list": cu_q,
            "cu_kv_list": cu_kv,
        },
        "cpu_output": {
            "dq": dq.to(dtype),
            "dk": dk.to(dtype),
            "dv": dv.to(dtype),
        },
        "case_name": case_name,
    }
