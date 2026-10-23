#!/usr/bin/python3
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


"""TTK compare plugin for quant_flash_mla.

Receives [npu_out(, npu_lse), golden_out(, golden_lse)] and checks
precision with bf16 tolerance defaults.
"""

import logging
import os
import sys

import numpy as np
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_ASSETS_DIR, "..")  # assets/
if _GOLDEN_DIR not in sys.path:
    sys.path.insert(0, _GOLDEN_DIR)
import qfmla_golden as golden_mod

try:
    from ttk.utilities.container_utils import get_global_storage
except Exception:
    try:
        from ttk.switches import get_global_storage
    except Exception:
        get_global_storage = None

try:
    from ttk.core_modules.common.comparison.cross_check import CrossCheckComparison
    from ttk.core_modules.common.comparison.resolve import resolve_tolerance

    _CROSS_CHECK_AVAILABLE = True
except Exception:
    try:
        from ttk.core_modules.comparison.cross_check import CrossCheckComparison
        from ttk.core_modules.comparison.resolve import resolve_tolerance

        _CROSS_CHECK_AVAILABLE = True
    except Exception:
        _CROSS_CHECK_AVAILABLE = False

logger = logging.getLogger(__name__)

_EPS = 1e-10


def _resolve_csv_tolerance(idx):
    """从 golden 模块读 csv 注入的 precision_tolerances / absolute_precision。

    (对齐 QFA compare.py 风格) ttk 框架不把 testcase 对象直接传给 custom compare,
    golden 插件在调用前把 csv 的 precision_tolerances / absolute_precision 暂存到
    golden_mod 上。本函数返回 (rtol, atol, diff_thd, pct_thd, max_diff_hd);
    csv 省略时返回 (None,)*5 由 check_result 用默认值 (rtol/atol 按 dtype 分支,
    diff_thd=0.005, pct_thd=0.005, max_diff_hd=10)。

    precision_tolerances 格式: tuple of per-output tuple, 每条目可为 2-tuple 或 5-tuple
      2-tuple (向后兼容): (rtol, atol)                       如 ((0.0078125, 0.0001),)
      5-tuple (新):        (rtol, atol, diff_thd, pct_thd, max_diff_hd)
                          如 ((0.0078125, 0.0001, 0.005, 0.005, 10),)
    """
    pt = getattr(golden_mod, "_csv_precision_tolerances", None)
    ap = getattr(golden_mod, "_csv_absolute_precision", None)
    rtol = None
    atol = None
    diff_thd = None
    pct_thd = None
    max_diff_hd = None
    if pt:
        pair = pt[idx] if idx < len(pt) else pt[-1]
        if pair is not None and len(pair) >= 2:
            rtol = float(pair[0])
            atol = float(pair[1])
            if len(pair) >= 3 and pair[2] is not None:
                diff_thd = float(pair[2])
            if len(pair) >= 4 and pair[3] is not None:
                pct_thd = float(pair[3])
            if len(pair) >= 5 and pair[4] is not None:
                max_diff_hd = float(pair[4])
    if atol is None and ap is not None:
        if isinstance(ap, (tuple, list)):
            val = ap[idx] if idx < len(ap) else ap[-1]
            if val is not None:
                atol = float(val)
        else:
            atol = float(ap)
    # replay 模式: pt 为 None (golden 未注入) → 用 bf16 默认, 与 CSV atten 容差一致
    if rtol is None and atol is None:
        rtol, atol = 0.0078125, 0.0001
    return rtol, atol, diff_thd, pct_thd, max_diff_hd


def _to_torch(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        if x.dtype.name == "bfloat16":
            return torch.from_numpy(x.view(np.uint8)).view(torch.bfloat16)
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x
    return torch.as_tensor(x)


def check_result(
    expect,
    npu_result,
    rtol=None,
    atol=None,
    diff_thd=None,
    pct_thd=None,
    max_diff_hd=None,
):
    _diff_thd_default = 0.005
    _pct_thd_default = 0.005
    _max_diff_hd_default = 10
    _rtol_default = 0.005
    _atol_default = 0.000025

    diff_thd = _diff_thd_default if diff_thd is None else diff_thd
    pct_thd = _pct_thd_default if pct_thd is None else pct_thd
    max_diff_hd = _max_diff_hd_default if max_diff_hd is None else max_diff_hd

    real_data = npu_result.detach().cpu().flatten()
    expect_data = (
        expect.detach().cpu().flatten()
        if isinstance(expect, torch.Tensor)
        else torch.as_tensor(expect).flatten()
    )

    if real_data.numel() == 0:
        return "Pass", 100.0, 0
    if real_data.numel() != expect_data.numel():
        return "Failed", 0.0, 0

    split_count = real_data.numel()
    real_float = real_data.to(torch.float32)
    expect_float = expect_data.to(torch.float32)

    if rtol is None or atol is None:
        if npu_result.dtype == torch.bfloat16:
            _rtol_default = 0.0078125
            _atol_default = 0.0001
        rtol = _rtol_default if rtol is None else rtol
        atol = _atol_default if atol is None else atol

    diff_result = torch.isclose(
        real_float, expect_float, rtol=rtol, atol=atol, equal_nan=True
    )
    err_idx = torch.nonzero(~diff_result, as_tuple=False).flatten()

    diff_abs = torch.abs(expect_float - real_float)
    b1 = torch.maximum(torch.abs(real_float), torch.abs(expect_float))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = torch.maximum(b1, torch.tensor(b2, dtype=torch.float32)) + _EPS
    err_diff = (diff_abs / b)[err_idx] if err_idx.numel() > 0 else torch.tensor([])
    err_diff = torch.where(
        torch.isnan(err_diff) | torch.isinf(err_diff),
        torch.tensor(float("inf"), dtype=torch.float32),
        err_diff,
    )

    fulfill_percent = float(split_count - err_idx.numel()) / float(split_count) * 100.0
    pct_thd = (1 - pct_thd) * 100.0
    result = "Pass" if fulfill_percent >= pct_thd else "Failed"

    max_error = 0.0
    if err_diff.numel() > 0:
        max_error = float(torch.max(err_diff).item())
        if max_error >= max_diff_hd:
            result = "Failed"

    logger.info(
        "Rtol=%.6f  Atol=%.6f  PctThd=%.2f%%  PctRlt=%.6f%%  Result=%s  MaxRE=%s",
        rtol,
        atol,
        pct_thd,
        fulfill_percent,
        result,
        max_error,
    )

    return result, fulfill_percent, max_error


# 三方 cross_check 判据(参考 flash_mla assets/impl/cross_check.py):
# NPU 对 bench(fp64 标杆) 的误差以 golden 对 bench 的误差为基准做比值判定,
# L0 = mare/mere/rmse 10/2/2 倍。仅在 --compare cross_check 时启用。
CROSS_CHECK_TOLERANCE = {
    dtype: {
        "standard": "cross_check",
        "level": "L0",
        "mare_ratio": 10.0,
        "mere_ratio": 2.0,
        "rmse_ratio": 2.0,
        "small_value": small_value,
        "small_value_atol": small_value_atol,
    }
    for dtype, small_value, small_value_atol in (
        ("float16", 2**-10, 2**-16),
        ("bfloat16", 2**-10, 2**-16),
        ("float32", 2**-14, 2**-30),
    )
}


def is_cross_check_available():
    return _CROSS_CHECK_AVAILABLE


def cross_check_enabled():
    """三方开关: 仅 CLI 显式 --compare cross_check 时开启, 默认走两方比较。"""
    if not _CROSS_CHECK_AVAILABLE or get_global_storage is None:
        return False
    try:
        method = getattr(get_global_storage(), "compare_method", None)
    except Exception:  # pylint: disable=broad-except
        return False
    return method is not None and str(method).lower() == "cross_check"


def _as_numpy(value):
    """转 numpy 供 ttk CrossCheckComparison 使用(bf16 先升 fp32 以兼容 torch)."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32).numpy()
    return np.asarray(value)


def _dtype_key(value):
    if isinstance(value, torch.Tensor):
        return str(value.dtype).removeprefix("torch.")
    return str(np.asarray(value).dtype)


def _is_empty_output(value):
    if value is None:
        return True
    if hasattr(value, "numel"):
        return int(value.numel()) == 0
    return getattr(value, "size", 0) == 0


def _resolve_cross_check_params(spec_tolerance, dtype_str):
    standards = resolve_tolerance(
        spec_tolerance, None, None, [dtype_str], "cross_check"
    )
    return standards[0].params


def _run_cross_check(outputs, bench_outputs, spec_tolerance, third_party_outputs=None):
    """三方比对: NPU 对 bench 的误差, 以 golden 对 bench 的误差为判据基准。"""
    if len(outputs) < 2 or len(outputs) % 2 != 0:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": "cross_check expects NPU outputs followed by golden outputs",
        }
    half = len(outputs) // 2
    results = []
    for idx, (npu_out, golden_out) in enumerate(zip(outputs[:half], outputs[half:])):
        bench = (
            bench_outputs[idx]
            if bench_outputs is not None and idx < len(bench_outputs)
            else None
        )
        label = "atten" if idx == 0 else f"out{idx}"
        if _is_empty_output(golden_out):
            # golden 未提供该输出(如未开启 lse) → 不参与判定
            results.append({"pass": True, "precision": "N/A"})
            continue
        if _is_empty_output(npu_out):
            results.append(
                {
                    "pass": False,
                    "precision": "NO_OUTPUT",
                    "error_info": f"{label}: NPU output is missing",
                }
            )
            continue
        if _is_empty_output(bench):
            results.append(
                {
                    "pass": False,
                    "precision": "NO_BENCH",
                    "error_info": f"{label}: cross_check bench is missing",
                }
            )
            continue

        dtype_str = _dtype_key(golden_out)
        params = _resolve_cross_check_params(spec_tolerance, dtype_str)
        # third_party 优先用分段量化参考: 与 target 同为逐段 P 量化, 噪声可比;
        # 未提供时回退 golden slot 输出。
        third_party = golden_out
        if (
            third_party_outputs is not None
            and idx < len(third_party_outputs)
            and third_party_outputs[idx] is not None
            and not _is_empty_output(third_party_outputs[idx])
        ):
            third_party = third_party_outputs[idx]
        logger.info(
            "[CROSS_CHECK] %s: dtype=%s third_party=%s params=%s",
            label,
            dtype_str,
            "sectioned_ref" if third_party is not golden_out else "golden",
            params,
        )
        comparison = CrossCheckComparison(
            _as_numpy(npu_out),
            _as_numpy(bench),
            idx,
            dtype_str,
            params,
            third_party=_as_numpy(third_party),
        )
        precision_str, log, is_pass, metrics = comparison.compare()
        metrics = dict(metrics or {})
        metrics.setdefault("label", label)
        results.append(
            {
                "pass": is_pass,
                "precision": precision_str,
                "metrics": metrics,
                "error_info": None if is_pass else f"{label}: {log}",
            }
        )
    return results[0] if len(results) == 1 else results


def compare(*outputs, **kwargs):
    """Compare NPU outputs vs golden outputs.

    TTK passes [npu_out(, npu_lse), golden_out(, golden_lse)].
    bench_outputs 存在时走 ttk 三方 cross_check; 否则按 CSV 容差做双方比较。
    """
    bench_outputs = kwargs.get("bench_outputs")
    spec_tolerance = kwargs.get("spec_tolerance")
    third_party_outputs = kwargs.get("third_party_outputs")
    if (
        bench_outputs is not None
        and spec_tolerance is not None
        and is_cross_check_available()
    ):
        return _run_cross_check(
            outputs, bench_outputs, spec_tolerance, third_party_outputs
        )
    non_none = [o for o in outputs if o is not None]
    if len(non_none) < 2:
        return {
            "pass": False,
            "precision": "invalid",
            "error_info": f"not enough valid outputs: {len(non_none)}",
        }

    valid = []
    for o in non_none:
        if hasattr(o, "size") and o.size == 0:
            continue
        if hasattr(o, "numel") and o.numel() == 0:
            continue
        valid.append(o)

    if len(valid) < 2:
        return {
            "pass": True,
            "precision": "100.0%",
            "error_info": None,
            "metrics": {"result": "Pass", "PctRlt": "100.0%", "MaxRE": 0.0},
        }

    half = len(valid) // 2
    npu_outputs = valid[:half]
    golden_outputs = valid[half:]

    results = []
    for idx, (npu_out, golden_out) in enumerate(zip(npu_outputs, golden_outputs)):
        label = "atten" if idx == 0 else f"out{idx}"
        npu_torch = _to_torch(npu_out)
        golden_torch = _to_torch(golden_out)

        rtol, atol, diff_thd, pct_thd, max_diff_hd = _resolve_csv_tolerance(idx)
        logger.info(
            "[COMPARE] %s: csv rtol=%s atol=%s diff_thd=%s pct_thd=%s max_diff_hd=%s",
            label,
            rtol,
            atol,
            diff_thd,
            pct_thd,
            max_diff_hd,
        )
        result, pct, max_err = check_result(
            golden_torch,
            npu_torch,
            rtol=rtol,
            atol=atol,
            diff_thd=diff_thd,
            pct_thd=pct_thd,
            max_diff_hd=max_diff_hd,
        )
        results.append(
            {
                "pass": result == "Pass",
                "precision": pct,
                "error_info": None if result == "Pass" else f"{label}: MaxRE={max_err}",
                "metrics": {
                    "label": label,
                    "result": result,
                    "PctRlt": pct,
                    "MaxRE": max_err,
                },
            }
        )

    if len(results) == 1:
        return results[0]
    return results
