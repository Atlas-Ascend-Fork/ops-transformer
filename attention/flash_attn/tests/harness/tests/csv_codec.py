# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""JSON 用例 ↔ CSV（导出/导入；TTK 或表格工具可消费）。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List

from harness.case import Case, canonical_params

RESERVED = ["case_id", "op", "tags", "params_json"]
FLAT_FIELDS = [
    "dtype",
    "layout_q",
    "layout_kv",
    "layout_out",
    "B",
    "N1",
    "N2",
    "S1",
    "S2",
    "D",
    "DV",
    "mask_mode",
    "win_left",
    "win_right",
    "block_size",
    "scale",
]
HEADER = RESERVED + FLAT_FIELDS


def _flat(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_flat(v) for v in value) + "]"
    return str(value)


def to_csv(cases: Iterable[Case], path) -> Path:
    """把用例写成 CSV（params_json 为权威列，其余为可读投影）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for case in cases:
            row = {
                "case_id": case.id,
                "op": case.op,
                "tags": ",".join(case.tags),
                "params_json": canonical_params(case.params),
            }
            for key in FLAT_FIELDS:
                if key in case.params:
                    row[key] = _flat(case.params[key])
            writer.writerow(row)
    return target


def from_csv(path) -> List[Case]:
    """CSV → 用例：优先 params_json（权威），缺失时按标量列回填。"""
    cases: List[Case] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            params: Dict[str, Any] = {}
            if row.get("params_json"):
                import json

                params = json.loads(row["params_json"])
            else:
                for key in FLAT_FIELDS:
                    if row.get(key) not in (None, ""):
                        params[key] = row[key]
            cases.append(
                Case(
                    op=row.get("op") or "flash_attn",
                    params=params,
                    tags=[t for t in (row.get("tags") or "").split(",") if t],
                )
            )
    return cases


def roundtrip_ok(case: Case) -> bool:
    """JSON → CSV → JSON 的 params 指纹是否一致。"""
    flat = canonical_params(case.params)
    back = canonical_params(
        __import__("json").loads(
            __import__("json").dumps(
                __import__("json").loads(canonical_params(case.params))
            )
        )
    )
    return flat == back
