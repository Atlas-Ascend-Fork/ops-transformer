# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""精度比对：dtype 容差（2 方）与 cross-check 比值（3 方），全部按块归约。"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch

TOLERANCE: Dict[str, Tuple[float, float]] = {
    "fp16": (2.5e-5, 0.005),
    "bf16": (1e-4, 0.0078125),
    "fp32": (1e-6, 1e-4),
    "float16": (2.5e-5, 0.005),
    "bfloat16": (1e-4, 0.0078125),
    "float32": (1e-6, 1e-4),
}
FAIL_RATIO_LIMIT = 0.005
CROSS_CHECK_THRESHOLDS = {"mare_ratio": 2.0, "mere_ratio": 1.2, "rmse_ratio": 1.2}
CHUNK_ELEMS = 4 * 1024 * 1024


def _pair_chunks(expect: torch.Tensor, result: torch.Tensor, idx):
    """取同一块；两侧设备不一致时把设备侧块搬到 CPU（只搬块，不搬整张量）。"""
    ec = expect[idx]
    rc = result[idx]
    if ec.device != rc.device:
        rc = rc.cpu()
        if ec.device.type != "cpu":
            ec = ec.cpu()
    return ec, rc


def _iter_index_chunks(shape, chunk_elems: int = CHUNK_ELEMS):
    """按行主序产出切片元组（每块元素数 ≤ chunk_elems，避免整块 float() 副本）。"""
    numel = 1
    for dim in shape:
        numel *= dim
    if numel <= chunk_elems:
        yield tuple(slice(None) for _ in shape)
        return

    def _rec(dims, prefix, size):
        if size <= chunk_elems or dims >= len(shape) - 1:
            yield tuple(prefix) + tuple(slice(None) for _ in range(len(shape) - dims))
            return
        per = size // shape[dims]
        step = max(1, chunk_elems // per)
        for start in range(0, shape[dims], step):
            end = min(start + step, shape[dims])
            yield from _rec(
                dims + 1, prefix + (slice(start, end),), per * (end - start)
            )

    yield from _rec(0, (), numel)


def tolerance_for(dtype: str) -> Tuple[float, float]:
    return TOLERANCE.get(str(dtype), TOLERANCE["fp32"])


def _dtype_of(tensor: torch.Tensor) -> str:
    return {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.float32: "fp32",
    }.get(tensor.dtype, "fp32")


def compare_two_way(
    expect: torch.Tensor,
    result: torch.Tensor,
    dtype: Optional[str] = None,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    fail_ratio_limit: float = FAIL_RATIO_LIMIT,
    max_samples: int = 10,
) -> Dict[str, Any]:
    """逐元素 |a-b| <= atol + rtol|b|，超差比例 ≤ 阈值即通过。"""
    if expect.shape != result.shape:
        return {
            "passed": False,
            "error": f"shape mismatch: {tuple(expect.shape)} vs {tuple(result.shape)}",
            "fail_cnt": -1,
            "total": int(expect.numel()),
            "fail_ratio": float("nan"),
        }
    dtype = dtype or _dtype_of(expect)
    default_atol, default_rtol = tolerance_for(dtype)
    atol = default_atol if atol is None else atol
    rtol = default_rtol if rtol is None else rtol

    total = max(int(expect.numel()), 1)
    acc_device = (
        expect.device if expect.device == result.device else torch.device("cpu")
    )
    fail_cnt = torch.zeros((), dtype=torch.int64, device=acc_device)
    max_abs = torch.zeros((), dtype=torch.float32, device=acc_device)
    max_rel = torch.zeros((), dtype=torch.float32, device=acc_device)
    sum_abs = torch.zeros((), dtype=torch.float64, device=acc_device)
    sum_rel = torch.zeros((), dtype=torch.float64, device=acc_device)

    for idx in _iter_index_chunks(expect.shape):
        ec, rc = _pair_chunks(expect, result, idx)
        ec = ec.float()
        rc = rc.float()
        diff = (ec - rc).abs()
        rel = diff / (ec.abs() + 1e-6)
        limit = atol + rtol * rc.abs()
        fail = diff > limit
        fail_cnt += fail.sum(dtype=torch.int64)
        if diff.numel():
            max_abs = torch.maximum(max_abs, diff.max())
            max_rel = torch.maximum(max_rel, rel.max())
            sum_abs += diff.double().sum()
            sum_rel += rel.double().sum()

    count = int(fail_cnt.item())
    samples: List[Tuple[int, float, float, float, float]] = []
    if count and max_samples > 0:
        base = 0
        for idx in _iter_index_chunks(expect.shape):
            ec, rc = _pair_chunks(expect, result, idx)
            ec = ec.float()
            rc = rc.float()
            diff = (ec - rc).abs()
            rel = diff / (ec.abs() + 1e-6)
            fail = diff > (atol + rtol * rc.abs())
            if bool(fail.any().item()):
                local = fail.reshape(-1).nonzero(as_tuple=False).squeeze(1)
                for j in local[: max_samples - len(samples)].tolist():
                    samples.append(
                        (
                            base + int(j),
                            float(ec.reshape(-1)[j]),
                            float(rc.reshape(-1)[j]),
                            float(diff.reshape(-1)[j]),
                            float(rel.reshape(-1)[j]),
                        )
                    )
                    if len(samples) >= max_samples:
                        break
                if len(samples) >= max_samples:
                    break
            base += ec.numel()

    fail_ratio = count / total
    return {
        "passed": fail_ratio <= fail_ratio_limit,
        "dtype": dtype,
        "atol": atol,
        "rtol": rtol,
        "fail_ratio_limit": fail_ratio_limit,
        "fail_cnt": count,
        "total": int(expect.numel()),
        "fail_ratio": fail_ratio,
        "max_abs": float(max_abs.item()),
        "mean_abs": float(sum_abs.item()) / total,
        "max_rel": float(max_rel.item()),
        "mean_rel": float(sum_rel.item()) / total,
        "samples": samples,
    }


def compare_three_way(
    truth: torch.Tensor,
    peer: torch.Tensor,
    actual: torch.Tensor,
    thresholds: Optional[Dict[str, float]] = None,
    max_samples: int = 10,
) -> Dict[str, Any]:
    """cross-check：actual/truth 与 peer/truth 的误差比值（"不比同误差预算的 peer 差"）。

    mare = mean|actual-truth| / mean|peer-truth|
    mere = mean(|actual-truth|/(|truth|+eps)) / mean(|peer-truth|/(|truth|+eps))
    rmse = sqrt(mean((actual-truth)^2)) / sqrt(mean((peer-truth)^2))
    """
    if not (truth.shape == peer.shape == actual.shape):
        return {
            "passed": False,
            "error": "shape mismatch",
            "mare_ratio": float("nan"),
            "mere_ratio": float("nan"),
            "rmse_ratio": float("nan"),
        }
    limits = dict(CROSS_CHECK_THRESHOLDS)
    limits.update(thresholds or {})

    a_sum = p_sum = a_rel = p_rel = a_sq = p_sq = 0.0
    total = max(int(truth.numel()), 1)
    samples: List[Tuple[int, float, float, float, float]] = []
    base = 0
    worst = (0.0, 0.0, None)
    for idx in _iter_index_chunks(truth.shape):
        tc, ac = _pair_chunks(truth, actual, idx)
        pc = peer[idx]
        if pc.device != tc.device:
            pc = pc.cpu()
        tc = tc.double()
        pc = pc.double()
        ac = ac.double()
        d_actual = (ac - tc).abs()
        d_peer = (pc - tc).abs()
        denom = tc.abs() + 1e-6
        a_sum += float(d_actual.sum().item())
        p_sum += float(d_peer.sum().item())
        a_rel += float((d_actual / denom).sum().item())
        p_rel += float((d_peer / denom).sum().item())
        a_sq += float((d_actual**2).sum().item())
        p_sq += float((d_peer**2).sum().item())
        if d_actual.numel():
            value, flat = d_actual.reshape(-1).max(dim=0)
            if float(value.item()) > worst[0]:
                j = int(flat.item())
                worst = (
                    float(value.item()),
                    float(d_peer.reshape(-1)[j].item()),
                    (base + j, float(tc.reshape(-1)[j]), float(ac.reshape(-1)[j])),
                )
        base += tc.numel()

    eps = 1e-12
    mare = (a_sum / total) / max(p_sum / total, eps)
    mere = (a_rel / total) / max(p_rel / total, eps)
    rmse = math.sqrt(a_sq / total) / max(math.sqrt(p_sq / total), eps)
    passed = (
        mare <= limits["mare_ratio"]
        and mere <= limits["mere_ratio"]
        and rmse <= limits["rmse_ratio"]
    )
    return {
        "passed": passed,
        "mare_ratio": mare,
        "mere_ratio": mere,
        "rmse_ratio": rmse,
        "limits": limits,
        "mean_abs": a_sum / total,
        "peer_mean_abs": p_sum / total,
        "worst": {
            "abs_err": worst[0],
            "peer_abs_err": worst[1],
            "index": None if worst[2] is None else worst[2][0],
        },
        "samples": samples,
    }
