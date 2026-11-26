#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unified dispatcher for functional / perf / random test frontends."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
MODES = ("functional", "perf", "random")


def _usage() -> str:
    return (
        "usage: run.py {functional|perf|random} [args...]\n\n"
        "  functional  pytest correctness/accuracy frontend\n"
        "  perf        msprof performance frontend\n"
        "  random      random fuzz frontend\n\n"
        "All arguments after the mode are passed through unchanged.\n"
    )


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage(), end="")
        return 0
    mode, args = argv[0], argv[1:]
    if mode not in MODES:
        print(f"run.py: unknown mode {mode!r}\n", file=sys.stderr)
        print(_usage(), end="", file=sys.stderr)
        return 2

    if mode == "functional":
        cmd = [sys.executable, "-m", "pytest", "pytests/test_flash_attn.py", *args]
    elif mode == "perf":
        cmd = [sys.executable, "-m", "perf_benchmark.run_perf", *args]
    else:
        cmd = [sys.executable, "-m", "random_fuzz.run_random_fuzz", *args]

    return subprocess.call(cmd, cwd=str(TESTS_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
