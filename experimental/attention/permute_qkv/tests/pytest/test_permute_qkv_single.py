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

import permute_qkv_golden as golden
import permute_qkv_process as process
import result_compare_method
import utils
from permute_qkv_paramset import ENABLED_PARAMS

result_path = os.getenv("PQ_RESULT_SAVE_PATH", "./result/pq_result.xlsx")


@pytest.mark.ci
@pytest.mark.parametrize("params", ENABLED_PARAMS)
def test_permute_qkv(params):
    torch_npu.npu.set_device(0)
    case_name = params[7]

    input_data = golden.gen_inputs(params)
    out = process.call_npu(input_data)

    result, max_diff, mismatch = result_compare_method.check_result(
        input_data["cpu_output"], out, "output"
    )
    utils.save_result(result, max_diff, params, result_path)

    assert result == "Pass", f"test case {case_name} failed (mismatch={mismatch})"
