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

# 用例结构：(seq_lens_q, seq_lens_kv, n, d, causal, dtype)
TEST_PARAMS = {
    "tnd_b1_causal_bf16": {
        "seq_lens_q": [256],
        "seq_lens_kv": [256],
        "n": 8,
        "d": 128,
        "causal": True,
        "dtype": torch.bfloat16,
    },
    "tnd_b2_causal_bf16": {
        "seq_lens_q": [128, 192],
        "seq_lens_kv": [256, 320],
        "n": 8,
        "d": 128,
        "causal": True,
        "dtype": torch.bfloat16,
    },
    "tnd_b1_noncausal_bf16": {
        "seq_lens_q": [256],
        "seq_lens_kv": [256],
        "n": 8,
        "d": 128,
        "causal": False,
        "dtype": torch.bfloat16,
    },
    "tnd_b1_causal_fp16": {
        "seq_lens_q": [128],
        "seq_lens_kv": [128],
        "n": 4,
        "d": 64,
        "causal": True,
        "dtype": torch.float16,
    },
}

ENABLED_PARAMS = []


def build_params():
    for key in TEST_PARAMS.keys():
        p = TEST_PARAMS[key]
        ENABLED_PARAMS.append(
            (
                p["seq_lens_q"],
                p["seq_lens_kv"],
                p["n"],
                p["d"],
                p["causal"],
                p["dtype"],
                key,
            )
        )


build_params()
