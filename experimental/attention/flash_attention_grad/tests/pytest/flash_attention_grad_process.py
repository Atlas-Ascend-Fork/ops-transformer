#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

import torch
import torch_npu
import aclnn_runner

INT32_MAX = 2147483647


def call_npu(input_data):
    """直接调用 aclnnFlashAttentionGrad，返回 (dq, dk, dv)。"""
    ti = input_data["input"]
    attr = input_data["attr"]
    dy = ti["dy"].npu()
    q = ti["query"].npu()
    k = ti["key"].npu()
    v = ti["value"].npu()
    attn_in = ti["attention_out"].npu()
    lse = ti["softmax_lse"].npu()
    cu_q = ti["cu_q"].npu()
    cu_kv = ti["cu_kv"].npu()
    cuq, cukv = attr["cu_q_list"], attr["cu_kv_list"]
    t1, n1, d = q.shape
    t2, n2 = k.shape[0], k.shape[1]
    dev = q.device
    sparse_mode = 3 if attr["causal"] else 0
    if attr["causal"]:
        mask = (
            torch.triu(torch.ones(2048, 2048, dtype=torch.float32), diagonal=1)
            .to(torch.bool)
            .to(dev)
        )
    else:
        mask = None

    dq = torch.empty(t1, n1, d, dtype=q.dtype, device=dev)
    dk = torch.empty(t2, n2, d, dtype=k.dtype, device=dev)
    dv = torch.empty(t2, n2, v.shape[2], dtype=v.dtype, device=dev)
    dsink = torch.empty(n1, dtype=q.dtype, device=dev)

    aclnn_runner.call(
        "aclnnFlashAttentionGradGetWorkspaceSize",
        "aclnnFlashAttentionGrad",
        [
            ("tensor", dy),
            ("tensor", q),
            ("tensor", k),
            ("tensor", v),
            ("tensor", attn_in),
            ("tensor", None),
            ("tensor", None),
            ("tensor", lse),
            ("tensor", cu_q),
            ("tensor", cu_kv),
            ("tensor", mask),
            ("tensor", None),
            ("tensor", None),
            ("double", attr["scale"]),
            ("int64", INT32_MAX),
            ("int64", INT32_MAX),
            ("int64", n1),
            ("str", "TND"),
            ("int64", sparse_mode),
            ("intarray", cuq),
            ("intarray", cukv),
            ("bool", False),
            ("bool", False),
            ("tensor", dq),
            ("tensor", dk),
            ("tensor", dv),
            ("tensor", dsink),
        ],
    )
    return dq, dk, dv
