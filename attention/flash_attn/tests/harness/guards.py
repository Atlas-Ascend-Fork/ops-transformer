# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Case legality and memory guards shared by pytest and offline tools."""

from __future__ import annotations

from typing import Optional

import torch

DEFAULT_MEM_GUARD_GB = 32.0
HBM_GUARD_RATIO = 0.9


def resolve_mem_guard(value: Optional[float]) -> float:
    return DEFAULT_MEM_GUARD_GB if value is None else float(value)


def resolve_hbm_guard(device_id: int) -> Optional[float]:
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return None
    try:
        _, total = torch.npu.mem_get_info(device_id)
    except Exception:  # noqa: BLE001
        try:
            props = torch.npu.get_device_properties(device_id)
            total = int(getattr(props, "total_memory", 0))
        except Exception:  # noqa: BLE001
            return None
    if not total:
        return None
    return float(total) * HBM_GUARD_RATIO / 1024**3


def memory_guard_reason(
    case,
    op,
    *,
    spec,
    compare_mode: str,
    stages,
    mem_guard_gb: Optional[float],
    hbm_guard_gb: Optional[float],
    safety_factor: float = 1.0,
    persist: bool = False,
) -> Optional[str]:
    if not (mem_guard_gb or hbm_guard_gb):
        return None
    estimate = op.estimate_memory(
        case.params,
        golden_device=spec.device,
        compare_mode=compare_mode,
        persist=persist,
        peer_like_truth=op.peer_like_truth(spec),
        with_golden="golden" in stages or "compare" in stages,
        golden_spec=spec,
    )
    host_gib = estimate["host_peak"] * safety_factor / 1024**3
    device_gib = estimate["device_peak"] * safety_factor / 1024**3
    if mem_guard_gb and host_gib > mem_guard_gb:
        return f"estimated host {host_gib:.1f} GiB > guard {mem_guard_gb:.1f} GiB"
    if hbm_guard_gb and device_gib > hbm_guard_gb:
        return f"estimated device {device_gib:.1f} GiB > guard {hbm_guard_gb:.1f} GiB"
    return None


def case_guard_reason(
    case,
    op,
    *,
    spec,
    compare_mode: str,
    stages,
    mem_guard_gb: Optional[float],
    hbm_guard_gb: Optional[float],
    safety_factor: float = 1.0,
    persist: bool = False,
) -> Optional[str]:
    reason = op.legalize(case.params)
    if reason:
        return f"illegal: {reason}"
    return memory_guard_reason(
        case,
        op,
        spec=spec,
        compare_mode=compare_mode,
        stages=stages,
        mem_guard_gb=mem_guard_gb,
        hbm_guard_gb=hbm_guard_gb,
        safety_factor=safety_factor,
        persist=persist,
    )
