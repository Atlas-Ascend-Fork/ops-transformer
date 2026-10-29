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

import dsa_sparse_prolog_golden as golden
import dsa_sparse_prolog_process as process
import result_compare_method
import utils
from dsa_sparse_prolog_paramset import ENABLED_PARAMS

result_path = os.getenv(
    "DSA_PROLOG_RESULT_SAVE_PATH", "./result/dsa_prolog_result.xlsx"
)


@pytest.mark.ci
@pytest.mark.parametrize("params", ENABLED_PARAMS)
def test_dsa_sparse_prolog(params):
    torch_npu.npu.set_device(0)
    case_name = params[7]

    input_data = golden.gen_inputs(params)
    op_outs = process.call_npu(input_data)
    base_outs = process.run_torch_baseline(input_data)

    ok, rows = result_compare_method.check_outputs(
        op_outs, base_outs, input_data["fp64_output"], input_data["demote"]
    )
    utils.save_result(ok, rows, case_name, result_path)
    assert ok, f"test case {case_name} failed"
