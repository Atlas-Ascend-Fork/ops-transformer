# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""统一结果记录：每个用例一条（功能/性能同构）。

- `results.csv`：唯一结果文件，列 = 核心列 ∪ 各用例 flat 字段（list/dict 以 JSON 字符串存）。

核心保持算子无关：算子特有的形状/派生列由 `op.result_columns(params)` 提供。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

CASE_COLUMNS: Sequence[str] = ("case_id", "op")
EXEC_COLUMNS: Sequence[str] = ("leg", "run_id", "runs", "run_index")
CASE_META_COLUMNS: Sequence[str] = ("category", "tags")
SEED_COLUMNS: Sequence[str] = ("seed",)
RESULT_COLUMNS: Sequence[str] = ("status", "detail")
GOLDEN_COLUMNS: Sequence[str] = (
    "compare_mode",
    "golden_spec",
    "golden_fingerprint",
)
PROFILE_COLUMNS: Sequence[str] = (
    "Block Num",
    "Mix Block Num",
    "Input Shapes",
    "Input Data Types",
    "Input Formats",
    "Output Shapes",
    "Output Data Types",
    "Output Formats",
    "aicore_time(us)",
    "aic_total_cycles",
    "aic_mac_time(us)",
    "aic_mac_ratio",
    "aic_scalar_time(us)",
    "aic_scalar_ratio",
    "aic_mte1_time(us)",
    "aic_mte1_ratio",
    "aic_mte2_time(us)",
    "aic_mte2_ratio",
    "aic_mte3_time(us)",
    "aic_mte3_ratio",
    "aic_fixpipe_time(us)",
    "aic_fixpipe_ratio",
    "aic_icache_miss_rate",
    "aiv_time(us)",
    "aiv_total_cycles",
    "aiv_vec_time(us)",
    "aiv_vec_ratio",
    "aiv_scalar_time(us)",
    "aiv_scalar_ratio",
    "aiv_mte2_time(us)",
    "aiv_mte2_ratio",
    "aiv_mte3_time(us)",
    "aiv_mte3_ratio",
    "aiv_icache_miss_rate",
    "cube_utilization(%)",
)
PERF_COLUMNS: Sequence[str] = (
    "flops",
    "bytes",
    "device_model",
    "standard_freq_mhz",
    "standard_peak_gbs",
    "standard_peak_tflops",
    "aic_core_num",
    "aiv_core_num",
    "theory_us_std",
    "theory_bound_std",
    "theory_us",
    "theory_bound",
    "duration_us",
    "mfu_pct",
    "mfu_std_pct",
    "mbu_pct",
    "aic_freq_mhz",
    "aiv_freq_mhz",
    "tiling_key",
) + PROFILE_COLUMNS
CORE_COLUMNS: Sequence[str] = (
    tuple(CASE_COLUMNS)
    + tuple(EXEC_COLUMNS)
    + tuple(CASE_META_COLUMNS)
    + tuple(RESULT_COLUMNS)
    + tuple(SEED_COLUMNS)
    + tuple(GOLDEN_COLUMNS)
    + tuple(PERF_COLUMNS)
)
METRIC_SUFFIXES = ("_passed", "_fail_ratio", "_max_abs")


def jsonable(value: Any) -> Any:
    """把张量/list/dict 等转成可 JSON 化的形式。"""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def _cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":"))
    return value


def columns(records: Sequence[Dict[str, Any]]) -> List[str]:
    extras: List[str] = []
    seen = set(CORE_COLUMNS)
    for record in records:
        for key in record:
            if key not in seen:
                extras.append(key)
                seen.add(key)
    metrics = [key for key in extras if key.endswith(METRIC_SUFFIXES)]
    case_extra = [key for key in extras if key not in metrics]
    return (
        list(CASE_COLUMNS)
        + list(EXEC_COLUMNS)
        + list(CASE_META_COLUMNS)
        + list(RESULT_COLUMNS)
        + metrics
        + list(SEED_COLUMNS)
        + list(GOLDEN_COLUMNS)
        + case_extra
        + list(PERF_COLUMNS)
    )


def write_csv(records: Sequence[Dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = columns(records)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({key: _cell(value) for key, value in record.items()})
    return path
