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

import block_indexer_rearrange_golden
import block_indexer_rearrange_process
import result_compare_method
import utils
from block_indexer_rearrange_paramset import ENABLED_PARAMS

result_path = os.getenv("BIR_RESULT_SAVE_PATH", "./result/bir_result.xlsx")


@pytest.mark.ci
@pytest.mark.parametrize("params", ENABLED_PARAMS)
def test_block_indexer_rearrange(params):
    torch_npu.npu.set_device(0)
    case_name = params[6]

    input_data = block_indexer_rearrange_golden.gen_data(params)
    npu_result = block_indexer_rearrange_process.call_npu(input_data)

    result, fulfill_percent = result_compare_method.check_result(
        input_data["cpu_output"], npu_result
    )
    utils.save_result(result, fulfill_percent, input_data, result_path)
    assert result == "Pass", f"test case {case_name} failed"
