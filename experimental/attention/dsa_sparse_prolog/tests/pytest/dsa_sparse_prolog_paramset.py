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

# 用例结构：(batch_shape, dtype, demote, fusion, scale_q, scale_kv, weight_layout)
#   - batch_shape: (T, He) 或 (B, S, He)；He 由 golden 默认 cfg 决定
#   - weight_layout: nz_kn / nd_kn / nd_nk
TEST_PARAMS = {
    "tnd_B2S64_bf16_demote": {
        "batch_shape": (2, 64),
        "dtype": torch.bfloat16,
        "demote": True,
        "fusion": False,
        "scale_q": 0.0,
        "scale_kv": 0.0,
        "weight_layout": "nd_kn",
    },
    "tnd_T512_bf16_demote": {
        "batch_shape": (512,),
        "dtype": torch.bfloat16,
        "demote": True,
        "fusion": False,
        "scale_q": 0.0,
        "scale_kv": 0.0,
        "weight_layout": "nd_kn",
    },
    "tnd_B1S128_fp16_demote": {
        "batch_shape": (1, 128),
        "dtype": torch.float16,
        "demote": True,
        "fusion": False,
        "scale_q": 0.0,
        "scale_kv": 0.0,
        "weight_layout": "nd_kn",
    },
    "tnd_B2S64_bf16_scale": {
        "batch_shape": (2, 64),
        "dtype": torch.bfloat16,
        "demote": True,
        "fusion": False,
        "scale_q": 2.0,
        "scale_kv": 3.0,
        "weight_layout": "nd_kn",
    },
    "tnd_B2S64_bf16_nd_kn": {
        "batch_shape": (2, 64),
        "dtype": torch.bfloat16,
        "demote": True,
        "fusion": False,
        "scale_q": 0.0,
        "scale_kv": 0.0,
        "weight_layout": "nd_kn",
    },
    "tnd_B2S64_bf16_nd_nk": {
        "batch_shape": (2, 64),
        "dtype": torch.bfloat16,
        "demote": True,
        "fusion": False,
        "scale_q": 0.0,
        "scale_kv": 0.0,
        "weight_layout": "nd_nk",
    },
}

ENABLED_PARAMS = []


def build_params():
    for key in TEST_PARAMS.keys():
        p = TEST_PARAMS[key]
        ENABLED_PARAMS.append(
            (
                p["batch_shape"],
                p["dtype"],
                p["demote"],
                p["fusion"],
                p["scale_q"],
                p["scale_kv"],
                p["weight_layout"],
                key,
            )
        )


build_params()
