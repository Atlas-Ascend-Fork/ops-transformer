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

# 定义测试参数组合
# params 结构：(batch, t1_base, n, case_name)
#   - t1_base 为按 batch 归一化前的总 query 长度基准；实际 T1 = sum(q_len)
#   - n 为主注意力 head 数，NI = N // 2，D = 192，DI = 128
TEST_PARAMS = {
    "tnd_b2_t256_n8": {"batch": [2], "t1_base": [256], "n": [8]},
    "tnd_b1_t256_n8": {"batch": [1], "t1_base": [256], "n": [8]},
    "tnd_b2_t512_n16": {"batch": [2], "t1_base": [512], "n": [16]},
}

# 按需选择要启用的测试参数（默认启用所有）
ENABLED_PARAMS = []


def build_params():
    for key in TEST_PARAMS.keys():
        params = TEST_PARAMS[key]
        ENABLED_PARAMS.append(
            (
                params["batch"][0],
                params["t1_base"][0],
                params["n"][0],
                key,
            )
        )


build_params()
