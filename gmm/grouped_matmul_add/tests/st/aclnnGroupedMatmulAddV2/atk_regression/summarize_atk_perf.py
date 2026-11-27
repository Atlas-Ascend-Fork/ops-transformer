# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import csv
from collections import Counter
import glob
import json
import math
import re
import statistics
import sys
from pathlib import Path

import openpyxl

root = Path(sys.argv[1]).resolve()
assets = Path(__file__).resolve().parent
labels = {
    "base-legacy": "legacy6",
    "final-legacy": "legacy6",
    "fixed8-low": "expanded8",
    "final-low": "expanded8",
    "final-nontranspose-fp32": "nontranspose_fp32_4",
}


def read_run(round_no, label):
    stem = labels[label]
    outdir = root / f"formal-r{round_no}-{label}"
    xlsx = glob.glob(str(outdir / "**" / "*.xlsx"), recursive=True)
    if len(xlsx) != 1:
        raise RuntimeError((str(outdir), xlsx))
    workbook = openpyxl.load_workbook(xlsx[0], data_only=True)
    if "failed cases" not in workbook.sheetnames:
        raise RuntimeError(("missing failed cases sheet", label, round_no))
    failed_rows = [
        r
        for r in list(workbook["failed cases"].values)[1:]
        if any(v is not None and v != "" for v in r)
    ]
    if failed_rows:
        raise RuntimeError(
            ("ATK failed cases sheet not empty", label, round_no, failed_rows)
        )
    sheet = workbook["statistic"]
    rows = list(sheet.values)
    cols = {key: idx for idx, key in enumerate(rows[0])}
    failures = [
        (r[cols["编号"]], r[cols["运行结果"]], r[cols["失败原因"]])
        for r in rows[1:]
        if r[cols["运行结果"]] != "SUCCESS" or r[cols["失败原因"]] not in (None, "")
    ]
    if failures:
        raise RuntimeError(("ATK case failed", label, round_no, failures))
    times = {
        int(r[cols["编号"]]): float(r[cols["pyaclnn_0_Device性能（us）"]])
        for r in rows[1:]
    }
    deviations = {
        int(r[cols["编号"]]): r[cols["pyaclnn_0_性能标准差"]] for r in rows[1:]
    }
    cases = json.loads((assets / f"{stem}.json").read_text())
    if set(times) != set(range(len(cases))):
        raise RuntimeError((label, round_no, times.keys(), len(cases)))
    source_csv = {}
    for path in (root / f"retained-r{round_no}-{label}").rglob("kernel_details.csv"):
        parts = path.parts
        indices = [i for i, part in enumerate(parts) if part == stem]
        if not indices:
            continue
        idx = int(parts[indices[-1] + 1])
        records = list(csv.DictReader(path.open(newline="")))
        duration = [
            float(r["Duration(us)"]) for r in records if r["Type"] == "GroupedMatmulAdd"
        ]
        if (
            len(duration) != 20
            or not math.isclose(
                statistics.mean(duration), times[idx], rel_tol=1e-7, abs_tol=1e-5
            )
            or deviations[idx] is None
            or not math.isclose(
                statistics.pstdev(duration),
                float(deviations[idx]),
                rel_tol=1e-6,
                abs_tol=1e-5,
            )
        ):
            raise RuntimeError(
                (
                    "invalid profiler events",
                    label,
                    round_no,
                    idx,
                    len(duration),
                    statistics.mean(duration) if duration else None,
                    times[idx],
                )
            )
        source_csv[idx] = (str(path), len(duration))
    if set(source_csv) != set(times):
        raise RuntimeError(
            ("missing profiler csv", label, round_no, source_csv.keys(), times.keys())
        )
    log = (root / f"formal-r{round_no}-{label}.log").read_text(errors="replace")
    values = Counter(
        re.findall(
            r"FIXED_INPUT (x|weight|y) shape=(\([^\n]*?\)) dtype=([^ ]+) sha256=([0-9a-f]{64})",
            log,
        )
    )
    groups = Counter(
        re.findall(
            r"FIXED_INPUT groupList values=(\[[^\n]*?\]) sha256=([0-9a-f]{64})", log
        )
    )
    if not values or not groups or "Total Task:" not in log:
        raise RuntimeError(
            ("missing provenance", label, round_no, len(values), len(groups))
        )
    dtype_names = {
        "fp16": "torch.float16",
        "bf16": "torch.bfloat16",
        "fp32": "torch.float32",
    }
    observed_types = Counter()
    for (name, shape, dtype, _digest), count in values.items():
        observed_types[name, shape, dtype] += count
    expected_types = Counter()
    for case in cases:
        for field in case["inputs"]:
            if field["name"] in ("x", "weight", "y"):
                expected_types[
                    field["name"],
                    str(tuple(field["shape"])),
                    dtype_names[field["dtype"]],
                ] += 2
    if observed_types != expected_types:
        raise RuntimeError(
            (
                "input dtype/shape mismatch with case JSON",
                label,
                round_no,
                observed_types,
                expected_types,
            )
        )
    return {
        "times": times,
        "deviations": deviations,
        "xlsx": xlsx[0],
        "csv": source_csv,
        "values": values,
        "groups": groups,
        "warnings": log.count("Incorrect schedule"),
        "cases": cases,
    }


result = []
all_runs = {}
for round_no in (1, 2, 3):
    for label in labels:
        all_runs[round_no, label] = read_run(round_no, label)
    for pair, left_label, right_label in [
        ("legacy", "base-legacy", "final-legacy"),
        ("low16", "fixed8-low", "final-low"),
    ]:
        left, right = all_runs[round_no, left_label], all_runs[round_no, right_label]
        if left["values"] != right["values"] or left["groups"] != right["groups"]:
            raise RuntimeError(("input hash mismatch", round_no, pair))
        for idx, case in enumerate(left["cases"]):
            pars = {p["name"]: p for p in case["inputs"]}
            xshape = pars["x"]["shape"]
            transpose = pars["transposeX"]["range_values"]
            base_us, final_us = left["times"][idx], right["times"][idx]
            result.append(
                {
                    "pair": pair,
                    "round": round_no,
                    "case_id": idx,
                    "M": xshape[1 if transpose else 0],
                    "N": pars["weight"]["shape"][1],
                    "x_dtype": pars["x"]["dtype"],
                    "y_dtype": pars["y"]["dtype"],
                    "transposeX": transpose,
                    "baseline_us": base_us,
                    "final_us": final_us,
                    "change_pct": (final_us / base_us - 1) * 100,
                    "baseline_stddev_us": left["deviations"][idx],
                    "final_stddev_us": right["deviations"][idx],
                    "baseline_events": left["csv"][idx][1],
                    "final_events": right["csv"][idx][1],
                    "baseline_xlsx": left["xlsx"],
                    "final_xlsx": right["xlsx"],
                    "baseline_kernel_csv": left["csv"][idx][0],
                    "final_kernel_csv": right["csv"][idx][0],
                    "baseline_record_warnings": left["warnings"],
                    "final_record_warnings": right["warnings"],
                }
            )
    last = all_runs[round_no, "final-nontranspose-fp32"]
    for idx, case in enumerate(last["cases"]):
        pars = {p["name"]: p for p in case["inputs"]}
        result.append(
            {
                "pair": "nontranspose_fp32_absolute",
                "round": round_no,
                "case_id": idx,
                "M": pars["x"]["shape"][0],
                "N": pars["weight"]["shape"][1],
                "x_dtype": pars["x"]["dtype"],
                "y_dtype": pars["y"]["dtype"],
                "transposeX": False,
                "baseline_us": "",
                "final_us": last["times"][idx],
                "change_pct": "",
                "baseline_stddev_us": "",
                "final_stddev_us": last["deviations"][idx],
                "baseline_events": "",
                "final_events": last["csv"][idx][1],
                "baseline_xlsx": "",
                "final_xlsx": last["xlsx"],
                "baseline_kernel_csv": "",
                "final_kernel_csv": last["csv"][idx][0],
                "baseline_record_warnings": "",
                "final_record_warnings": last["warnings"],
            }
        )

path = root / "atk-perf-paired-results.csv"
with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(result[0]))
    writer.writeheader()
    writer.writerows(result)
print("REPORT", path, "ROWS", len(result), "PROVENANCE_OK")

summary = []
for pair in ("legacy", "low16", "nontranspose_fp32_absolute"):
    ids = sorted({r["case_id"] for r in result if r["pair"] == pair})
    for idx in ids:
        rows = [r for r in result if r["pair"] == pair and r["case_id"] == idx]
        assert len(rows) == 3
        changes = (
            [r["change_pct"] for r in rows]
            if pair != "nontranspose_fp32_absolute"
            else []
        )
        summary.append(
            {
                "pair": pair,
                "case_id": idx,
                "M": rows[0]["M"],
                "N": rows[0]["N"],
                "x_dtype": rows[0]["x_dtype"],
                "y_dtype": rows[0]["y_dtype"],
                "transposeX": rows[0]["transposeX"],
                "baseline_median_us": statistics.median(r["baseline_us"] for r in rows)
                if changes
                else "",
                "final_median_us": statistics.median(r["final_us"] for r in rows),
                "paired_change_median_pct": statistics.median(changes)
                if changes
                else "",
                "paired_change_min_pct": min(changes) if changes else "",
                "paired_change_max_pct": max(changes) if changes else "",
                "observation": (
                    "all_slower"
                    if all(v > 0 for v in changes)
                    else "all_faster"
                    if all(v < 0 for v in changes)
                    else "mixed"
                )
                if changes
                else "absolute_only",
            }
        )
path = root / "atk-perf-summary.csv"
with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
print("SUMMARY", path, "ROWS", len(summary))
