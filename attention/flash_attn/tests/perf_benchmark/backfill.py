#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Backfill msprof duration/tiling/frequency into perf results.csv."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from harness import perf as perflib  # noqa: E402
from harness import results as resultslib  # noqa: E402
from harness.device_spec import current_device_model, get_device_spec  # noqa: E402
from harness.ops.flash_attn import PACKAGE  # noqa: E402

TIME_KEYS = ("task_time(us)", "Task Duration(us)", "duration_us")
START_KEYS = ("task_start(us)", "Task Start Time(us)", "start_us")
KERNEL_KEYS = ("kernel_name", "Kernel Name", "kernelName")


def _first(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _task_files(prof_dir: Path) -> List[Path]:
    return sorted(prof_dir.glob("**/task_time_*.csv"))


def _flash_kernels(prof_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in _task_files(prof_dir):
        for row in _read_csv(path):
            name = str(_first(row, KERNEL_KEYS) or "")
            if not name.startswith("FlashAttn_"):
                continue
            rows.append(row)
    return rows


def _op_summary_rows(prof_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(prof_dir.glob("**/op_summary_*.csv")):
        for row in _read_csv(path):
            if str(_first(row, ("OP Type", "op_type")) or "") == "FlashAttn":
                rows.append(row)
    return rows


def _kernel_key(row: Dict[str, Any]) -> tuple:
    return (
        str(_first(row, ("Device_id", "device_id")) or ""),
        str(_first(row, ("stream_id", "Stream ID")) or ""),
        str(_first(row, ("task_id", "Task ID")) or ""),
    )


def _effective_freq(cycles: Any, time_us: Any, cores: Any) -> Optional[float]:
    cycles_v = _num(cycles)
    time_v = _num(time_us)
    cores_v = _num(cores)
    if not cycles_v or not time_v or not cores_v:
        return None
    return cycles_v / (time_v * cores_v)


def _num(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_map(rows: Iterable[Dict[str, Any]]) -> Dict[tuple, Dict[str, Any]]:
    mapped = {}
    for row in rows:
        key = (row.get("case_id"), str(row.get("run_index")))
        mapped.setdefault(key, row)
    return mapped


def backfill(
    results_path: Path,
    manifest_path: Path,
    prof_dir: Path,
) -> int:
    results_path = Path(results_path)
    manifest = _load_manifest(Path(manifest_path))
    result_rows = _read_csv(results_path)
    by_key = _row_map(result_rows)
    manifest_rows = [
        row for row in manifest.get("rows", []) if row.get("case_id") is not None
    ]
    kernels = _flash_kernels(Path(prof_dir))
    details_by_key = {_kernel_key(row): row for row in _op_summary_rows(Path(prof_dir))}
    warmup = int(manifest.get("perf_warmup") or 0)
    case_order = manifest.get("case_order") or []
    if warmup and case_order:
        counts = {}
        for row in manifest_rows:
            case_id = row.get("case_id")
            counts[case_id] = counts.get(case_id, 0) + 1
        selected = []
        cursor = 0
        for case_id in case_order:
            cursor += warmup
            take = counts.get(case_id, 0)
            selected.extend(kernels[cursor : cursor + take])
            cursor += take
        kernels = selected
    count = min(len(manifest_rows), len(kernels))
    if len(kernels) < len(manifest_rows):
        print(
            f"[backfill] kernel count short: manifest={len(manifest_rows)} "
            f"profiling={len(kernels)}; filling {count}",
            file=sys.stderr,
        )

    peak_tflops = float(manifest.get("peak_tflops") or 0.0)
    peak_gbs = float(manifest.get("peak_gbs") or 0.0)
    device_id = int(manifest.get("device_id") or 0)
    cores = perflib.device_aicore_num(device_id)
    aiv_cores = perflib.device_aivcore_num(device_id)
    spec = get_device_spec(device_id)
    device_model = spec.model if spec else (current_device_model(device_id) or "")
    standard_freq = manifest.get("standard_freq_mhz")
    if standard_freq is None and spec is not None:
        standard_freq = spec.standard_freq_mhz
    standard_gbs = manifest.get("standard_peak_gbs")
    if standard_gbs is None and spec is not None:
        standard_gbs = spec.peak_gbs
    filled = 0
    for row, kernel in zip(manifest_rows[:count], kernels[:count]):
        target = by_key.get((row.get("case_id"), str(row.get("run_index"))))
        if target is None:
            continue
        details = details_by_key.get(_kernel_key(kernel)) or {}
        for column in resultslib.PROFILE_COLUMNS:
            if column in details:
                target[column] = details[column]
        aic_freq = _effective_freq(
            details.get("aic_total_cycles"),
            details.get("aicore_time(us)"),
            details.get("Block Num"),
        )
        aiv_freq = _effective_freq(
            details.get("aiv_total_cycles"),
            details.get("aiv_time(us)"),
            details.get("Mix Block Num"),
        )
        target["aic_freq_mhz"] = f"{aic_freq:.6g}" if aic_freq is not None else ""
        target["aiv_freq_mhz"] = f"{aiv_freq:.6g}" if aiv_freq is not None else ""
        duration = _num(_first(kernel, TIME_KEYS))
        if duration is not None:
            target["duration_us"] = f"{duration:.6g}"
            flops = _num(target.get("flops"))
            dtype = str(target.get("dtype") or "fp16")
            row_peak = 0.0
            if aic_freq is not None:
                row_peak = peak_tflops or perflib.peak_tflops(cores, dtype, aic_freq)
                row_peak = row_peak or 0.0
            theory_us, theory_bound = perflib.theory_time(
                flops,
                _num(target.get("bytes")),
                row_peak,
                peak_gbs,
            )
            target["theory_us"] = f"{theory_us:.6g}" if theory_us is not None else ""
            target["theory_bound"] = theory_bound
            mfu = perflib.mfu_pct(flops, duration, row_peak) if row_peak else None
            std_peak = (
                perflib.peak_tflops(cores, dtype, float(standard_freq))
                if standard_freq and cores
                else None
            )
            mfu_std = perflib.mfu_pct(flops, duration, std_peak) if std_peak else None
            theory_std, bound_std = perflib.theory_time(
                flops,
                _num(target.get("bytes")),
                std_peak or 0.0,
                float(standard_gbs) if standard_gbs else 0.0,
            )
            mbu = (
                perflib.mbu_pct(_num(target.get("bytes")), duration, peak_gbs)
                if peak_gbs
                else None
            )
            target["device_model"] = device_model
            target["standard_freq_mhz"] = (
                f"{float(standard_freq):.6g}" if standard_freq else ""
            )
            target["standard_peak_gbs"] = (
                f"{float(standard_gbs):.6g}" if standard_gbs else ""
            )
            target["standard_peak_tflops"] = (
                f"{std_peak:.6g}" if std_peak is not None else ""
            )
            target["aic_core_num"] = "" if cores is None else str(cores)
            target["aiv_core_num"] = "" if aiv_cores is None else str(aiv_cores)
            target["theory_us_std"] = (
                f"{theory_std:.6g}" if theory_std is not None else ""
            )
            target["theory_bound_std"] = bound_std
            target["mfu_pct"] = f"{mfu:.6g}" if mfu is not None else ""
            target["mbu_pct"] = f"{mbu:.6g}" if mbu is not None else ""
            target["mfu_std_pct"] = f"{mfu_std:.6g}" if mfu_std is not None else ""
        name = str(_first(kernel, KERNEL_KEYS) or "")
        key = PACKAGE.tiling_key(name)
        if key:
            target["tiling_key"] = key
        filled += 1

    with results_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(result_rows[0].keys()) if result_rows else []
        )
        writer.writeheader()
        writer.writerows(result_rows)
    print(f"[backfill] filled {filled} rows -> {results_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prof-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    return backfill(args.results, args.manifest, args.prof_dir)


if __name__ == "__main__":
    raise SystemExit(main())
