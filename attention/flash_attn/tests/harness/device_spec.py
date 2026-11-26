# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Device-model peaks used by performance metrics.

Peak HBM bandwidth is not available from torch_npu/npu-smi; keep it as a
model-keyed table instead of guessing from clock/capacity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DeviceSpec:
    model: str
    peak_gbs: float
    standard_freq_mhz: float


DEVICE_SPECS = {
    "Ascend950PR_9579": DeviceSpec(
        model="Ascend950PR_9579",
        peak_gbs=1600.0,
        standard_freq_mhz=1650.0,
    ),
}


def spec_for_model(model: Optional[str]) -> Optional[DeviceSpec]:
    if not model:
        return None
    return DEVICE_SPECS.get(str(model))


def current_device_model(device_id: int = 0) -> Optional[str]:
    try:
        import torch

        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            return None
        return str(torch.npu.get_device_properties(device_id).name)
    except Exception:  # noqa: BLE001
        return None


def get_device_spec(device_id: int = 0) -> Optional[DeviceSpec]:
    return spec_for_model(current_device_model(device_id))


def resolve_peak_gbs(device_id: int, value: Optional[float]) -> float:
    if value is not None:
        return float(value)
    spec = get_device_spec(device_id)
    return float(spec.peak_gbs) if spec else 0.0
