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

import os

import pytest
import torch_npu

import dsa_dense_golden as golden
import dsa_dense_process as process
import result_compare_method as compare
import utils
from dsa_dense_paramset import ENABLED_PARAMS

result_path = os.getenv("DSA_RESULT_SAVE_PATH", "./result/dsa_result.xlsx")

OUTPUT_NAMES = ["loss", "dq", "dk", "dw"]


@pytest.mark.ci
@pytest.mark.parametrize("params", ENABLED_PARAMS)
def test_dsa_dense(params):
    torch_npu.npu.set_device(0)
    case_name = params[3]

    input_data = golden.gen_data(params)
    npu_output = process.call_npu(input_data)

    results = {}
    for name in OUTPUT_NAMES:
        result, max_diff, mean_diff = compare.check_result(
            input_data["cpu_output"][name], npu_output[name], name
        )
        results[name] = (result, max_diff, mean_diff)

    utils.save_result(results, params, result_path)

    failed = [name for name in OUTPUT_NAMES if results[name][0] != "Pass"]
    assert not failed, f"test case {case_name} failed on {failed}"
