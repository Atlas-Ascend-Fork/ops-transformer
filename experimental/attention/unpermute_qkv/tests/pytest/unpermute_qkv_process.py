#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

import torch
import torch_npu
import aclnn_runner


def call_npu(input_data):
    """直接调用 aclnnUnpermuteQkv。"""
    inp = input_data["input"]["x"].npu()
    n_k = input_data["attr"]["n_k"]
    split, s_c, b, inner, h = inp.shape
    total = split * inner
    q_head = total - 2 * n_k
    q = torch.empty(s_c, b, q_head, h, dtype=inp.dtype, device=inp.device)
    k = torch.empty(s_c, b, n_k, h, dtype=inp.dtype, device=inp.device)
    v = torch.empty(s_c, b, n_k, h, dtype=inp.dtype, device=inp.device)
    aclnn_runner.call(
        "aclnnUnpermuteQkvGetWorkspaceSize",
        "aclnnUnpermuteQkv",
        [("tensor", inp), ("tensor", q), ("tensor", k), ("tensor", v)],
    )
    return q, k, v
