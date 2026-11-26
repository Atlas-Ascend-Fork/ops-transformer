#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Execute a perf batch directly on harness.runner.Runner (no pytest)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from harness import loader, results as resultslib  # noqa: E402
from harness.device_spec import get_device_spec, resolve_peak_gbs  # noqa: E402
from harness.golden_spec import parse_spec  # noqa: E402
from harness.ops.flash_attn import PACKAGE  # noqa: E402
from harness.runner import Runner, RunnerConfig, SessionFatal  # noqa: E402
from cli_common import (  # noqa: E402
    add_case_filter_args,
    add_common_args,
    add_golden_args,
)


def _select(cases, case_id, testset, skip_cases):
    patterns = []
    if skip_cases:
        if str(skip_cases).endswith(".txt"):
            patterns = loader.load_skip_file(skip_cases)
        else:
            patterns = [x.strip() for x in str(skip_cases).split(",") if x.strip()]
    case_ids = [x.strip() for x in str(case_id or "").split(",") if x.strip()]
    testsets = [x.strip() for x in str(testset or "").split(",") if x.strip()]
    return loader.select(cases, case_ids=case_ids, testsets=testsets, skip=patterns)


def _write_manifest(path, cfg, records, verify=False):
    rows = []
    for index, record in enumerate(records, 1):
        if record.get("status") == "skip":
            continue
        rows.append(
            {
                "seq": len(rows) + 1,
                "case_id": record.get("case_id"),
                "run_index": record.get("run_index"),
                "status": record.get("status"),
            }
        )
    case_order = []
    for record in records:
        if record.get("status") == "skip":
            continue
        case_id = record.get("case_id")
        if case_id not in case_order:
            case_order.append(case_id)
    spec = get_device_spec(cfg.device_id)
    payload = {
        "schema": "perf-run/v1",
        "out_dir": str(cfg.out_dir),
        "device_model": spec.model if spec else "",
        "standard_freq_mhz": spec.standard_freq_mhz if spec else None,
        "standard_peak_gbs": spec.peak_gbs if spec else None,
        "golden_spec": cfg.golden_spec.describe(),
        "compare_mode": cfg.compare_mode,
        "perf_runs": cfg.perf_runs,
        "perf_warmup": cfg.perf_warmup,
        "device_id": cfg.device_id,
        "verify": bool(verify),
        "case_order": case_order,
        "peak_tflops": cfg.peak_tflops,
        "peak_gbs": cfg.peak_bw_gbs,
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", action="append", required=True)
    add_common_args(parser, default_out_dir=None)
    parser.add_argument("--perf-runs", type=int, default=5)
    add_golden_args(parser)
    parser.add_argument("--verify", action="store_true")
    add_case_filter_args(parser)
    args = parser.parse_args(argv)
    if not args.out_dir:
        parser.error("--out-dir is required")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = _select(
        loader.load_from_json(args.cases),
        args.case_id,
        args.testset,
        args.skip_cases,
    )
    if not cases:
        print("[perf-worker] no cases matched", file=sys.stderr)
        return 1

    peak_gbs = resolve_peak_gbs(args.device_id, None)
    golden_spec = parse_spec(args.golden)
    cfg = RunnerConfig(
        stages=("gen", "run"),
        golden_spec=golden_spec,
        compare_mode=args.compare_mode,
        device_id=args.device_id,
        out_dir=str(out),
        mem_guard_gb=args.mem_guard_gb,
        graph_mode=args.graph,
        perf=True,
        perf_runs=args.perf_runs,
        peak_tflops=0.0,
        peak_bw_gbs=peak_gbs,
    )
    runner = Runner(cfg, PACKAGE)
    print(f"[perf-worker] {len(cases)} cases -> {out}")
    fatal = False
    try:
        for index, case in enumerate(cases, 1):
            print(f"[perf-worker] {index}/{len(cases)} {case.id}")
            runner.run(case)
    except SessionFatal as exc:
        fatal = True
        print(f"[perf-worker] fatal: {exc}", file=sys.stderr)

    verify = {}
    for record in runner.records:
        for prefix in PACKAGE.metric_columns():
            for key in ("passed", "fail_ratio", "max_abs"):
                record.setdefault(f"{prefix}_{key}", "")
    if args.verify and not fatal:
        verify_cfg = RunnerConfig(
            stages=("gen", "golden", "run", "compare"),
            golden_spec=golden_spec,
            compare_mode=args.compare_mode,
            device_id=args.device_id,
            out_dir=str(out),
            graph_mode=args.graph,
            perf=False,
        )
        verify_runner = Runner(verify_cfg, PACKAGE)
        verify_abort = ""
        for index, case in enumerate(cases, 1):
            print(f"[perf-worker] verify {index}/{len(cases)} {case.id}")
            try:
                result = verify_runner.run(case)
            except SessionFatal as exc:
                fatal = True
                verify_abort = f"verify aborted: {exc}"
                print(f"[perf-worker] {verify_abort}", file=sys.stderr)
                verify[case.id] = {
                    "passed": False,
                    "skipped": False,
                    "error": str(exc),
                    "reason": "",
                    "metrics": None,
                }
                break
            verify[case.id] = {
                "passed": result.passed,
                "skipped": result.skipped,
                "error": result.error,
                "reason": result.reason,
                "metrics": result.metrics,
            }
        (out / "verify.json").write_text(
            json.dumps(verify, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for record in runner.records:
            info = verify.get(record.get("case_id"))
            record["compare_mode"] = args.compare_mode
            record["golden_spec"] = golden_spec.describe()
            record["golden_fingerprint"] = golden_spec.fingerprint()
            if info is None:
                record["status"] = "error"
                record["detail"] = verify_abort or "verify missing"
                continue
            metrics = info.get("metrics") or {}
            for prefix, values in metrics.items():
                if not isinstance(values, dict):
                    continue
                for key in ("passed", "fail_ratio", "max_abs"):
                    if key in values:
                        record[f"{prefix}_{key}"] = values[key]
            detail = info.get("error") or info.get("reason")
            if detail:
                record["detail"] = detail
            if info.get("skipped"):
                record["status"] = "skip"
            elif info.get("error"):
                record["status"] = "error"
            else:
                record["status"] = "pass" if info.get("passed") else "fail"
    else:
        for record in runner.records:
            record["compare_mode"] = ""
            record["golden_spec"] = ""
            record["golden_fingerprint"] = ""

    resultslib.write_csv(runner.records, out / "results.csv")
    _write_manifest(out / "perf_manifest.json", cfg, runner.records, args.verify)
    print(f"[perf-worker] rows={len(runner.records)} -> {out / 'results.csv'}")
    return 2 if fatal else 0


if __name__ == "__main__":
    raise SystemExit(main())
