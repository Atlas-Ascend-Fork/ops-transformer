#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

import torch
import torch_npu
import aclnn_runner

INT64_MAX = 9223372036854775807


def call_npu(input_data):
    """直接调用 aclnnFlashAttention，返回 (attention_out, lse_out)。"""
    ti = input_data["input"]
    attr = input_data["attr"]
    q = ti["query"].npu()
    k = ti["key"].npu()
    v = ti["value"].npu()
    cu_q = ti["cu_q"].npu()
    cu_kv = ti["cu_kv"].npu()
    t1, n1, d = q.shape
    dv = v.shape[2]
    dev = q.device
    sparse_mode = 3 if attr["causal"] else 0
    cuq = attr["cu_q_list"]
    cukv = attr["cu_kv_list"]
    if attr["causal"]:
        mask = (
            torch.triu(torch.ones(2048, 2048, dtype=torch.float32), diagonal=1)
            .to(torch.bool)
            .to(q.device)
        )
    else:
        mask = None

    smax = torch.empty(n1, t1, dtype=torch.float32, device=dev)
    ssum = torch.empty(n1, t1, dtype=torch.float32, device=dev)
    lse = torch.empty(n1, t1, dtype=torch.float32, device=dev)
    attn = torch.empty(t1, n1, dv, dtype=q.dtype, device=dev)
    mlog = torch.empty(max(n1, 1024), dtype=torch.float32, device=dev)

    aclnn_runner.call(
        "aclnnFlashAttentionGetWorkspaceSize",
        "aclnnFlashAttention",
        [
            ("tensor", q),
            ("tensor", k),
            ("tensor", v),
            ("tensor", cu_q),
            ("tensor", cu_kv),
            ("tensor", mask),
            ("tensor", None),
            ("tensor", None),
            ("double", attr["scale"]),
            ("int64", INT64_MAX),
            ("int64", INT64_MAX),
            ("int64", n1),
            ("str", "TND"),
            ("int64", 0),
            ("int64", sparse_mode),
            ("intarray", cuq),
            ("intarray", cukv),
            ("tensor", smax),
            ("tensor", ssum),
            ("tensor", lse),
            ("tensor", attn),
            ("tensor", mlog),
        ],
    )
    return attn, lse
