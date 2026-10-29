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

import flash_attention_golden as golden
import flash_attention_process as process
import result_compare_method
import utils
from flash_attention_paramset import ENABLED_PARAMS

result_path = os.getenv("FA_RESULT_SAVE_PATH", "./result/fa_result.xlsx")


@pytest.mark.ci
@pytest.mark.parametrize("params", ENABLED_PARAMS)
def test_flash_attention(params):
    torch_npu.npu.set_device(0)
    case_name = params[6]

    input_data = golden.gen_inputs(params)
    attn_out, lse_out = process.call_npu(input_data)

    results = {
        "attention_out": result_compare_method.check_result(
            input_data["cpu_output"]["attention_out"], attn_out, "attention_out"
        ),
        "lse_out": result_compare_method.check_result(
            input_data["cpu_output"]["lse_out"], lse_out, "lse_out"
        ),
    }
    utils.save_result(results, params, result_path)

    failed = [name for name, r in results.items() if r[0] != "Pass"]
    assert not failed, f"test case {case_name} failed on {failed}"
