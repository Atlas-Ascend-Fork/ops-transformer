# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Local TTK three-party cross_check L2 fallback when no remote xpu_server is configured."""

import logging

import numpy as np
import torch

from . import fla_adapter as fla_mod
from . import golden as golden_mod

TOLERANCE_OP = {
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
}

TOLERANCE_ADAPTER = {
    "bfloat16": {"standard": "cross_check", "level": "L2"},
    "float32": {"standard": "cross_check", "level": "L2"},
    "bool": {"standard": "binary_equal"},
}

# Cache only the active testcase so eager + graph legs share one third-party pass
# without growing memory across a full CSV run.
_THIRD_PARTY_CACHE = {}


def _to_cpu_tensor(value):
    if value is None or torch.is_tensor(value):
        return value.detach().cpu() if torch.is_tensor(value) else None
    arr = np.asarray(value)
    if str(arr.dtype) == "bfloat16":
        return torch.from_numpy(arr.view(np.uint16).copy()).view(torch.bfloat16)
    return torch.from_numpy(arr.copy())


def _dtype_name(value):
    if torch.is_tensor(value):
        return str(value.dtype).removeprefix("torch.")
    return str(np.asarray(value).dtype)


def _to_compare_array(value):
    if torch.is_tensor(value):
        cpu = value.detach().cpu()
        if cpu.dtype == torch.bfloat16:
            return cpu.float().numpy()
        return cpu.numpy()
    arr = np.asarray(value)
    if str(arr.dtype) == "bfloat16":
        return arr.astype(np.float32)
    return arr


def _compute_third_party(compare_context, *, adapter=False):
    if compare_context is None:
        raise RuntimeError(
            "FlashKDA local cross_check requires compare_context from TTK"
        )
    cache_key = (
        bool(adapter),
        getattr(compare_context, "api_name", None),
        str(getattr(compare_context, "testcase_name", "")),
    )
    if cache_key in _THIRD_PARTY_CACHE:
        return _THIRD_PARTY_CACHE[cache_key]

    raw_inputs = getattr(compare_context, "input_tensors", None)
    if raw_inputs is None:
        raise RuntimeError(
            "FlashKDA local cross_check requires input_tensors in compare_context"
        )
    tensors = [_to_cpu_tensor(t) for t in raw_inputs]
    attrs = dict(getattr(compare_context, "attributes", None) or {})
    if not fla_mod.enabled():
        attrs["device_mode"] = "cpu"

    if len(tensors) < 11:
        raise ValueError(f"FlashKDA expects 11 tensor inputs, got {len(tensors)}")
    (
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        a_log,
        dt_bias,
        cu_seqlens,
        seqused,
    ) = tensors[:11]
    peer = golden_mod.adapter_third_party if adapter else golden_mod.op_third_party
    third_party_outputs = peer(
        q,
        k,
        v,
        g,
        beta,
        metadata=metadata,
        initial_state=initial_state,
        A_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        **attrs,
    )

    if not isinstance(third_party_outputs, (tuple, list)):
        third_party_outputs = (third_party_outputs,)
    _THIRD_PARTY_CACHE.clear()
    _THIRD_PARTY_CACHE[cache_key] = tuple(third_party_outputs)
    return _THIRD_PARTY_CACHE[cache_key]


def _compare_slot(
    npu_out, golden_out, third_party_out, index, tolerance, testcase_name
):
    from ttk.core_modules.common.comparison.cross_check import CrossCheckComparison
    from ttk.core_modules.common.comparison.resolve import resolve_tolerance

    if golden_out is None:
        if npu_out is None:
            return {"pass": True, "precision": "SUPPRESSED"}
        return {
            "pass": False,
            "precision": "UNEXPECTED_OUTPUT",
            "error_info": f"FlashKDA output {index} expected None",
        }
    if npu_out is None:
        return {
            "pass": False,
            "precision": "NO_OUTPUT",
            "error_info": f"FlashKDA output {index} is missing",
        }

    dtype_str = _dtype_name(npu_out)
    actual_arr = _to_compare_array(npu_out)
    golden_arr = _to_compare_array(golden_out)

    if dtype_str == "bool":
        passed = actual_arr.shape == golden_arr.shape and bool(
            np.array_equal(actual_arr, golden_arr)
        )
        return {
            "pass": passed,
            "precision": "100%" if passed else "0%",
            "metrics": {"standard": "binary_equal", "pass": passed},
            "error_info": None
            if passed
            else f"FlashKDA bool output {index} mismatch: {actual_arr} vs {golden_arr}",
        }

    if third_party_out is None:
        return {
            "pass": False,
            "precision": "GOLDEN_FAILURE",
            "error_info": f"FlashKDA third_party output {index} is missing",
        }

    third_arr = _to_compare_array(third_party_out)
    if actual_arr.shape != golden_arr.shape or actual_arr.shape != third_arr.shape:
        return {
            "pass": False,
            "precision": "shape_mismatch",
            "error_info": (
                f"FlashKDA output {index} shape mismatch: "
                f"npu={actual_arr.shape}, golden={golden_arr.shape}, third_party={third_arr.shape}"
            ),
        }

    params = resolve_tolerance(tolerance, None, None, [dtype_str], "cross_check")[
        0
    ].params
    comparator = CrossCheckComparison(
        actual_arr,
        golden_arr,
        index,
        dtype_str,
        params,
        third_party=third_arr,
    )
    precision, log, passed, metrics = comparator.compare()
    logging.info(
        "[%s] FlashKDA local cross_check output %d (%s): %s metrics=%s",
        testcase_name,
        index,
        dtype_str,
        precision,
        metrics,
    )
    return {
        "pass": passed,
        "precision": precision,
        "metrics": metrics,
        "error_info": None if passed else log,
    }


def _compare_all(outputs, *, compare_context, adapter):
    from .npu_preprocess import reset_mode

    reset_mode()
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        raise ValueError(
            "FlashKDA local cross_check expects equal NPU and golden output counts"
        )
    half = len(outputs) // 2
    npu_outputs = outputs[:half]
    golden_outputs = outputs[half:]
    testcase_name = getattr(compare_context, "testcase_name", "unknown")
    tolerance = TOLERANCE_ADAPTER if adapter else TOLERANCE_OP

    # Bool-only adapter cases (invalid / whitebox / meta) do not need third_party.
    needs_float_third_party = any(
        g is not None and n is not None and _dtype_name(n) != "bool"
        for n, g in zip(npu_outputs, golden_outputs)
    )
    if needs_float_third_party:
        third_party_outputs = _compute_third_party(compare_context, adapter=adapter)
        if len(third_party_outputs) != half:
            raise ValueError(
                f"FlashKDA third_party output count mismatch: "
                f"expected {half}, got {len(third_party_outputs)}"
            )
    else:
        third_party_outputs = (None,) * half

    results = [
        _compare_slot(n, g, tp, idx, tolerance, testcase_name)
        for idx, (n, g, tp) in enumerate(
            zip(npu_outputs, golden_outputs, third_party_outputs)
        )
    ]
    return results[0] if len(results) == 1 else results


def compare_op(*outputs, compare_context=None, **_kwargs):
    """Local three-party cross_check L2 comparator for direct ``flash_kda``."""
    return _compare_all(outputs, compare_context=compare_context, adapter=False)


def compare_adapter(*outputs, compare_context=None, **_kwargs):
    """Local three-party cross_check L2 comparator for ``kda_ttk_ops.flash_kda``."""
    return _compare_all(outputs, compare_context=compare_context, adapter=True)
