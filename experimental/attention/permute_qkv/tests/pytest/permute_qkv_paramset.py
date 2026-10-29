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

# 用例结构：(s_c, b, split_size, kv_group, n_k, h, dtype)
TEST_PARAMS = {
    "tnd_s512_b1_sp2_g2_nk4": {
        "s_c": 512,
        "b": 1,
        "split_size": 2,
        "kv_group": 2,
        "n_k": 4,
        "h": 128,
        "dtype": torch.bfloat16,
    },
    "tnd_s1024_b3_sp8_g8_nk32": {
        "s_c": 1024,
        "b": 3,
        "split_size": 8,
        "kv_group": 8,
        "n_k": 32,
        "h": 128,
        "dtype": torch.bfloat16,
    },
    "tnd_s512_b2_sp2_g4_nk16": {
        "s_c": 512,
        "b": 2,
        "split_size": 2,
        "kv_group": 4,
        "n_k": 16,
        "h": 128,
        "dtype": torch.bfloat16,
    },
}

ENABLED_PARAMS = []


def build_params():
    for key in TEST_PARAMS.keys():
        p = TEST_PARAMS[key]
        ENABLED_PARAMS.append(
            (
                p["s_c"],
                p["b"],
                p["split_size"],
                p["kv_group"],
                p["n_k"],
                p["h"],
                p["dtype"],
                key,
            )
        )


build_params()
