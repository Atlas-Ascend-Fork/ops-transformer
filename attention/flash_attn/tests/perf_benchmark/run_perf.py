#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run perf in config (generate + plot) or cases (execute only) mode."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
PERF_ROOT = Path(__file__).resolve().parent
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from perf_benchmark.backfill import backfill  # noqa: E402
from perf_benchmark.gen_bench_cases import write_cases  # noqa: E402
from cli_common import (  # noqa: E402
    add_case_filter_args,
    add_common_args,
    add_golden_args,
    default_out_dir,
)


def _worker_args(args) -> list:
    cmd = [
        sys.executable,
        str(PERF_ROOT / "perf_worker.py"),
        *[f"--cases={path}" for path in args.cases],
        f"--out-dir={args.out_dir}",
        f"--perf-runs={args.perf_runs}",
        f"--device-id={args.device_id}",
        f"--golden={args.golden}",
    ]
    if args.mem_guard_gb is not None:
        cmd.append(f"--mem-guard-gb={args.mem_guard_gb}")
    if args.graph:
        cmd.append("--graph")
    for key in ("case_id", "testset", "skip_cases"):
        value = getattr(args, key, None)
        if value:
            cmd.append(f"--{key.replace('_', '-')}={value}")
    return cmd


def _plot_results(out):
    try:
        from perf_benchmark.plot_bench import plot
    except Exception as exc:
        print(f"[perf] skip plot: {exc}", file=sys.stderr)
        return
    try:
        count = plot(
            out / "results.csv",
            out / "cases" / "bench_manifest.json",
            out / "figs",
        )
    except Exception as exc:
        print(f"[perf] plot failed: {exc}", file=sys.stderr)
        return
    print(f"[perf] plotted {count} points -> {out / 'figs'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        type=Path,
        help="bench YAML config (required unless --cases is used)",
    )
    source.add_argument(
        "--cases",
        action="append",
        help="perf case JSON/dir (repeatable; skips generation/plotting)",
    )
    add_common_args(parser, default_out_dir=None)
    parser.add_argument("--perf-runs", type=int, default=5)
    add_golden_args(parser)
    parser.add_argument("--verify", action="store_true")
    add_case_filter_args(parser)
    args = parser.parse_args(argv)

    out = Path(args.out_dir or default_out_dir("perf"))
    out.mkdir(parents=True, exist_ok=True)
    prof_dir = out / "profiling"
    prof_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = str(out)

    bench_mode = not args.cases
    if bench_mode:
        cases_dir = out / "cases"
        cases, _ = write_cases(args.config, cases_dir)
        if not cases:
            parser.error(f"no cases generated from {args.config}")
        args.cases = [str(cases_dir / "bench_cases.json")]
        print(f"[perf] generated {len(cases)} cases from {args.config} -> {cases_dir}")
    else:
        args.cases = [str(Path(path)) for path in args.cases]
        print(f"[perf] cases: {', '.join(args.cases)}")

    worker = _worker_args(args)
    if args.verify:
        worker.append("--verify")
        worker.append(f"--compare-mode={args.compare_mode}")
    cmd = [
        "msprof",
        f"--output={prof_dir}",
        "--aic-mode=task-based",
        *worker,
    ]
    print(f"[perf] {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(TESTS_ROOT), env=dict(os.environ))
    results = out / "results.csv"
    manifest = out / "perf_manifest.json"
    if results.exists() and manifest.exists():
        backfill(results, manifest, prof_dir)
        if bench_mode:
            _plot_results(out)
    else:
        print(
            f"[perf] results/manifest missing: {results}, {manifest}", file=sys.stderr
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
