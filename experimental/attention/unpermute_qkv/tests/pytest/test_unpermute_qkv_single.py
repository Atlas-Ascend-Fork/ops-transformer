#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

import os

import pytest
import torch_npu

import unpermute_qkv_golden as golden
import unpermute_qkv_process as process
import result_compare_method
import utils
from unpermute_qkv_paramset import ENABLED_PARAMS

result_path = os.getenv("UQ_RESULT_SAVE_PATH", "./result/uq_result.xlsx")


@pytest.mark.ci
@pytest.mark.parametrize("params", ENABLED_PARAMS)
def test_unpermute_qkv(params):
    torch_npu.npu.set_device(0)
    case_name = params[7]

    input_data = golden.gen_inputs(params)
    q, k, v = process.call_npu(input_data)

    results = {
        "q": result_compare_method.check_result(input_data["cpu_output"]["q"], q, "q"),
        "k": result_compare_method.check_result(input_data["cpu_output"]["k"], k, "k"),
        "v": result_compare_method.check_result(input_data["cpu_output"]["v"], v, "v"),
    }
    utils.save_result(results, params, result_path)

    failed = [name for name, r in results.items() if r[0] != "Pass"]
    assert not failed, f"test case {case_name} failed on {failed}"
