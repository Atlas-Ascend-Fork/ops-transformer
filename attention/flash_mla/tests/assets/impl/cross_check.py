# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TTK three-way comparison with an FP32 baseline."""

import logging

import numpy as np
import torch


# Only the current testcase is retained, bounded across a long CSV run.
_CASES = {}
TOLERANCE = {
    dtype: {
        "standard": "cross_check",
        "level": "L0",
        "mare_ratio": 10.0,
        "mere_ratio": 2.0,
        "rmse_ratio": 2.0,
        "small_value": 2**-8 if dtype == "bfloat16" else 2**-10,
        "small_value_atol": 2**-16,
    }
    for dtype in ("float16", "bfloat16", "float32")
}


def enabled():
    from ttk.switches import get_global_storage

    return get_global_storage().compare_method == "cross_check"


def prepare(arguments, testcase_name):
    """Compute before TTK Promote can change the original input precision."""
    _CASES.clear()
    if not enabled():
        return
    if testcase_name is None:
        raise ValueError("MLA cross_check requires testcase_name")
    from .golden import FlashMlaGolden
    from .fp32_reference import FP32Reference

    arguments = {
        name: value.detach().cpu() if torch.is_tensor(value) else value
        for name, value in arguments.items()
        if name != "metadata"
    }
    arguments["device_mode"] = "cpu"
    golden = FlashMlaGolden().forward_golden(**arguments)
    bench = FP32Reference().forward_golden(**arguments)
    if not arguments["return_softmax_lse"]:
        golden, bench = (golden[0], None), (bench[0], None)
    _CASES[str(testcase_name)] = (golden, bench)
    logging.info("[%s] MLA cross_check prepared FP32 reference", testcase_name)


def get_case(testcase_name):
    try:
        return _CASES[str(testcase_name)]
    except KeyError as exc:
        raise RuntimeError(
            "MLA cross_check requires customize_inputs for this testcase; "
            "cached bench is missing (manual golden replay is not supported)"
        ) from exc


def compare(*outputs, compare_context):
    from ttk.core_modules.common.comparison.cross_check import CrossCheckComparison
    from ttk.core_modules.common.comparison.resolve import resolve_tolerance

    if len(outputs) != 4:
        raise ValueError("MLA cross_check expects two NPU and two golden outputs")
    testcase_name = compare_context.testcase_name
    _, bench_outputs = get_case(testcase_name)
    results = []
    for index, (npu, golden, bench) in enumerate(
        zip(outputs[:2], outputs[2:], bench_outputs)
    ):
        if golden is None:
            if index != 1 or bench is not None:
                raise ValueError(
                    "MLA cross_check output suppression differs from bench"
                )
            results.append({"pass": True, "precision": "SUPPRESSED"})
            continue
        if npu is None or bench is None:
            results.append({"pass": False, "precision": "NO_OUTPUT"})
            continue
        dtype = str(npu.dtype).removeprefix("torch.")
        params = resolve_tolerance(TOLERANCE, None, None, [dtype], "cross_check")[
            0
        ].params
        arrays = [
            np.asarray(
                value.detach().cpu().float().numpy()
                if torch.is_tensor(value)
                else value,
                dtype=np.float32,
            )
            for value in (npu, golden, bench)
        ]
        if arrays[0].shape != arrays[1].shape or arrays[0].shape != arrays[2].shape:
            results.append({"pass": False, "precision": "shape_mismatch"})
            continue
        # FP32 is the baseline; mixed precision is the other reference.
        comparator = CrossCheckComparison(
            arrays[0], arrays[2], index, dtype, params, third_party=arrays[1]
        )
        precision, log, passed, metrics = comparator.compare()
        results.append(
            {
                "pass": passed,
                "precision": precision,
                "metrics": metrics,
                "error_info": None if passed else log,
            }
        )
    return results


def as_float32(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def pre_compare(*outputs):
    if len(outputs) != 4:
        raise ValueError("FlashMla expects two device and two golden outputs")
    for index, (actual, expected) in enumerate(zip(outputs[:2], outputs[2:])):
        if expected is None and index == 1:
            continue
        if actual is None or expected is None:
            raise ValueError(f"FlashMla output {index} is missing")
        actual, expected = as_float32(actual), as_float32(expected)
        if actual.shape != expected.shape:
            raise ValueError(
                f"FlashMla output {index} shape mismatch: {actual.shape} vs {expected.shape}"
            )
        if np.isnan(actual).any() or np.isnan(expected).any():
            raise ValueError(f"FlashMla output {index} contains NaN")
    return outputs
