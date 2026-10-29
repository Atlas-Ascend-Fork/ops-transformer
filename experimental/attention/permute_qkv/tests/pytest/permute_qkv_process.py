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

import torch
import torch_npu

import aclnn_runner


def call_npu(input_data):
    """直接调用 aclnnPermuteQkv（不依赖 torch_npu.permute_qkv 绑定）。"""
    ti = input_data["input"]
    q = ti["query"].npu()
    k = ti["key"].npu()
    v = ti["value"].npu()
    split_size = input_data["attr"]["split_size"]

    s_c, b, q_head, h = q.shape
    nk = k.shape[2]
    total = q_head + 2 * nk
    out = torch.empty(total, s_c, b, h, dtype=q.dtype, device=q.device)

    aclnn_runner.call(
        "aclnnPermuteQkvGetWorkspaceSize",
        "aclnnPermuteQkv",
        [
            ("tensor", q),
            ("tensor", k),
            ("tensor", v),
            ("int64", split_size),
            ("tensor", out),
        ],
    )
    return out
