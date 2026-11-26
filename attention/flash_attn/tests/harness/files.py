# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""case 数据目录与输入指纹（阶段间通信）。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(name: str) -> str:
    flat = _SAFE.sub("_", name).strip("._") or "case"
    if flat == name and len(flat) <= 120:
        return flat
    return f"{flat[:96]}-{hashlib.sha256(name.encode()).hexdigest()[:12]}"


def inputs_digest(inputs: Dict[str, Any], seed: int) -> str:
    """输入指纹：seed + 每个张量的 shape/dtype；大张量不全量哈希。"""
    items = []
    for key in sorted(inputs):
        value = inputs[key]
        if hasattr(value, "shape"):
            items.append(f"{key}:{tuple(value.shape)}:{value.dtype}")
        else:
            items.append(f"{key}:{type(value).__name__}")
    raw = f"seed={seed}|" + "|".join(items)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class CaseFiles:
    def __init__(self, root: Path, case_id: str):
        self.dir = Path(root) / safe_name(case_id)

    def path(self, name: str) -> Path:
        return self.dir / name

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def save_json(self, name: str, payload: Dict[str, Any]) -> Path:
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        return target

    def load_json(self, name: str) -> Optional[Dict[str, Any]]:
        target = self.path(name)
        if not target.exists():
            return None
        with target.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def save_tensors(self, name: str, payload: Dict[str, Any]) -> Path:
        """保存张量字典；设备张量先转 CPU（逐张量），避免整份 host 副本同时驻留。"""
        import torch

        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        cpu_payload = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                cpu_payload[key] = value.detach().cpu()
            else:
                cpu_payload[key] = value
        torch.save(cpu_payload, target)
        return target

    def load_tensors(self, name: str) -> Optional[Dict[str, Any]]:
        import torch

        target = self.path(name)
        parts = sorted(self.dir.glob(f"{Path(name).stem}.part_*.pt"))
        if target.exists():
            return torch.load(target, map_location="cpu", weights_only=False)
        if parts:
            merged = {}
            for part in parts:
                merged.update(torch.load(part, map_location="cpu", weights_only=False))
            return merged
        return None
