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
# params 结构：
# (n2, cur_seqlen_q, cur_seqlen_kv, block_size, block_count, causal, case_name)
TEST_PARAMS = {
    "tnd_b1_bs4_bc512": {
        "n2": [1],
        "cur_seqlen_q": [[200]],
        "cur_seqlen_kv": [[200]],
        "block_size": [4],
        "block_count": [512],
        "causal": [True],
    },
    "tnd_b2_bs4_bc512": {
        "n2": [1],
        "cur_seqlen_q": [[200, 300]],
        "cur_seqlen_kv": [[400, 900]],
        "block_size": [4],
        "block_count": [512],
        "causal": [True],
    },
    "tnd_b2_bs2_bc1024": {
        "n2": [1],
        "cur_seqlen_q": [[512, 512]],
        "cur_seqlen_kv": [[1024, 2048]],
        "block_size": [2],
        "block_count": [1024],
        "causal": [True],
    },
    "tnd_b2_bs8_bc2048": {
        "n2": [1],
        "cur_seqlen_q": [[1024, 2048]],
        "cur_seqlen_kv": [[2048, 8192]],
        "block_size": [8],
        "block_count": [2048],
        "causal": [True],
    },
}

# 按需选择要启用的测试参数（默认启用所有）
ENABLED_PARAMS = []


def build_params():
    for key in TEST_PARAMS.keys():
        params = TEST_PARAMS[key]
        ENABLED_PARAMS.append(
            (
                params["n2"][0],
                params["cur_seqlen_q"][0],
                params["cur_seqlen_kv"][0],
                params["block_size"][0],
                params["block_count"][0],
                params["causal"][0],
                key,
            )
        )


build_params()
