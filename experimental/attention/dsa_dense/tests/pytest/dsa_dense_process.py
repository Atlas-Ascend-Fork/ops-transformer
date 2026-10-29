#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

import torch
import torch_npu
import aclnn_runner


def call_npu(input_data):
    """直接调用 aclnnDsaDense，返回 {loss, dq, dk, dw}。"""
    ti = input_data["input"]
    attr = input_data["attr"]
    query = ti["query"].npu()
    key = ti["key"].npu()
    mx = ti["max"].npu()
    sm = ti["sum"].npu()
    qlen = ti["q_len"].npu()
    kvlen = ti["kv_len"].npu()
    qidx = ti["query_index"].npu()
    kidx = ti["key_index"].npu()
    wt = ti["weights"].npu()
    midx = ti["max_index"].npu()
    sidx = ti["sum_index"].npu()

    t1, n, _ = query.shape
    t2 = key.shape[0]
    ni, di = qidx.shape[1], qidx.shape[2]
    dt, dev = query.dtype, query.device
    mask = (
        torch.triu(torch.ones(128, 128, dtype=torch.float32), diagonal=1)
        .to(torch.bool)
        .to(dev)
    )

    loss = torch.empty(1, dtype=torch.float32, device=dev)
    dq = torch.empty(t1, ni, di, dtype=dt, device=dev)
    dk = torch.empty(t2, di, dtype=dt, device=dev)
    dw = torch.empty(ni, t1, dtype=torch.float32, device=dev)

    aclnn_runner.call(
        "aclnnDsaDenseGetWorkspaceSize",
        "aclnnDsaDense",
        [
            ("tensor", query),
            ("tensor", key),
            ("tensor", mx),
            ("tensor", sm),
            ("tensor", qlen),
            ("tensor", kvlen),
            ("tensor", mask),
            ("tensor", qidx),
            ("tensor", kidx),
            ("tensor", wt),
            ("tensor", midx),
            ("tensor", sidx),
            ("int64", int(attr["sparse_mode"])),
            ("double", attr["scale"]),
            ("tensor", loss),
            ("tensor", dq),
            ("tensor", dk),
            ("tensor", dw),
        ],
    )
    return {"loss": loss, "dq": dq, "dk": dk, "dw": dw.t().contiguous()}
