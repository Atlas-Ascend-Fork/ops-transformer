#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

import torch
import torch_npu
import aclnn_runner


def call_npu(input_data):
    """直接调用 aclnnBlockIndexerRearrange。"""
    ti = input_data["input"]
    topk = ti["topk_indices"].npu()
    acc_q = ti["acc_seqlen_q"].npu()
    acc_kv = ti["acc_seqlen_kv"].npu()
    causal = input_data["attr"]["causal"]
    block_size = input_data["attr"]["block_size"]
    t1, n2, kcnt = topk.shape
    out = torch.empty(t1, n2, kcnt * block_size, dtype=torch.int32, device=topk.device)
    aclnn_runner.call(
        "aclnnBlockIndexerRearrangeGetWorkspaceSize",
        "aclnnBlockIndexerRearrange",
        [
            ("tensor", topk),
            ("tensor", acc_q),
            ("tensor", acc_kv),
            ("bool", bool(causal)),
            ("int64", int(block_size)),
            ("tensor", out),
        ],
    )
    return out
