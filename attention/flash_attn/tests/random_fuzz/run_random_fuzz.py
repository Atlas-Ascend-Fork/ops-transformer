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
"""Execute generated flash_attn cases through the existing runner."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from cli_common import (  # noqa: E402
    add_common_args,
    add_golden_args,
    default_out_dir,
)
from harness import guards  # noqa: E402
from harness import results as resultslib  # noqa: E402
from harness.golden_spec import parse_spec  # noqa: E402
from harness.ops.flash_attn import PACKAGE  # noqa: E402
from harness.runner import Runner, RunnerConfig, SessionFatal  # noqa: E402
from random_fuzz.gen_random_cases import generate_many  # noqa: E402

GEN_CASES_PER_BATCH = 1000
FATAL_STREAK_LIMIT = 20


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    add_common_args(parser, default_out_dir=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--max-cases", type=int, default=0, help="总用例数上限，0=不限制；按批生成"
    )
    parser.add_argument(
        "--max-seconds", type=float, default=0.0, help="总墙钟预算（含生成），0=不限制"
    )
    parser.add_argument("--max-failures", type=int, default=0)
    add_golden_args(parser)
    args = parser.parse_args(argv)
    if not args.max_cases and not args.max_seconds:
        parser.error("at least one of --max-cases / --max-seconds must be > 0")

    out = Path(args.out_dir or default_out_dir("random"))
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    golden_spec = parse_spec(args.golden)
    mem_guard_gb = guards.resolve_mem_guard(args.mem_guard_gb)

    stages = ("gen", "golden", "run", "compare")
    cfg = RunnerConfig(
        stages=stages,
        golden_spec=golden_spec,
        compare_mode=args.compare_mode,
        device_id=args.device_id,
        mem_guard_gb=mem_guard_gb,
        graph_mode=args.graph,
        out_dir=str(out),
    )
    runner = Runner(cfg, PACKAGE)
    print(f"[fuzz] out={out}")

    failures = []

    def write_outputs():
        path = resultslib.write_csv(runner.records, out / "results.csv")
        failure_path = out / "failures.json"
        if failures:
            failure_path.write_text(
                json.dumps(
                    {"count": len(failures), "cases": failures},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return path, failure_path if failures else None

    started = time.monotonic()
    executed = 0
    generated = 0
    current = None
    stop = False
    fatal_streak = 0
    fatal_abort = False
    try:
        while not stop:
            if args.max_seconds and time.monotonic() - started >= args.max_seconds:
                print(f"[fuzz] time budget reached after {executed} cases")
                break
            if args.max_cases:
                remaining = args.max_cases - executed
                if remaining <= 0:
                    break
                want = min(GEN_CASES_PER_BATCH, remaining)
            else:
                want = GEN_CASES_PER_BATCH
            print(f"[fuzz] generating {want} cases in one batch")
            cases, rejects, attempts = generate_many(
                rng, want, golden_spec=golden_spec, mem_guard_gb=mem_guard_gb
            )
            for reason, count in rejects.most_common():
                print(f"[reject] {count:4d} {reason}")
            print(
                f"[fuzz] generated {len(cases)}/{want} cases after {attempts} attempts"
            )
            generated += len(cases)
            if not cases:
                print("[fuzz] no valid cases generated")
                break

            total = args.max_cases if args.max_cases else len(cases)
            for case in cases:
                if args.max_seconds and time.monotonic() - started >= args.max_seconds:
                    print(f"[fuzz] time budget reached after {executed} cases")
                    stop = True
                    break
                if args.max_cases and executed >= args.max_cases:
                    stop = True
                    break
                current = case
                executed += 1
                try:
                    result = runner.run(case)
                except SessionFatal as exc:
                    fatal_streak += 1
                    failures.append(
                        {
                            "op": case.op,
                            "case_id": case.id,
                            "tags": list(case.tags),
                            "status": "fatal",
                            "detail": str(exc),
                            "params": resultslib.jsonable(case.params),
                        }
                    )
                    print(
                        f"[{executed}/{total}] fatal  "
                        f"{case.id.split('/')[-1][:64]} | {exc}"
                    )
                    if fatal_streak >= FATAL_STREAK_LIMIT:
                        print(
                            f"[fuzz] stop after {fatal_streak} consecutive fatal cases"
                        )
                        stop = True
                        fatal_abort = True
                        break
                    continue
                fatal_streak = 0
                status = (
                    "skip"
                    if result.skipped
                    else (
                        "error"
                        if result.error
                        else ("pass" if result.passed else "fail")
                    )
                )
                detail = result.error or result.reason or ""
                print(
                    f"[{executed}/{total}] {status:6s} "
                    f"{case.id.split('/')[-1][:64]}"
                    + (f" | {detail}" if detail else "")
                )
                if status in ("fail", "error"):
                    failures.append(
                        {
                            "op": case.op,
                            "case_id": case.id,
                            "tags": list(case.tags),
                            "status": status,
                            "detail": result.error or result.reason,
                            "params": resultslib.jsonable(case.params),
                            "metrics": resultslib.jsonable(result.metrics),
                        }
                    )
                    if args.max_failures and len(failures) >= args.max_failures:
                        print(f"[fuzz] stop after {len(failures)} failures")
                        stop = True
                        break
            path, _ = write_outputs()
            print(f"[fuzz] checkpoint: {len(runner.records)} rows -> {path}")
    except SessionFatal as exc:
        print(f"[FATAL] {exc}")
        if current is not None:
            failures.append(
                {
                    "op": current.op,
                    "case_id": current.id,
                    "tags": list(current.tags),
                    "status": "fatal",
                    "detail": str(exc),
                    "params": resultslib.jsonable(current.params),
                }
            )
        path, failure_path = write_outputs()
        print(f"[fuzz] results: {path}")
        if failure_path:
            print(f"[fuzz] failures: {failure_path}")
        return 2
    path, failure_path = write_outputs()
    print(f"[fuzz] generated={generated} cases={executed} failures={len(failures)}")
    print(f"[fuzz] results: {path}")
    if failure_path:
        print(f"[fuzz] failures: {failure_path}")
    if fatal_abort:
        return 2
    return 0 if generated else 1


if __name__ == "__main__":
    raise SystemExit(main())
