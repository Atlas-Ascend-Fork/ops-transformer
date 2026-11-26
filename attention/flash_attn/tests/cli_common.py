# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared option values and argparse registration; pytest reuses the constants."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

DEFAULT_GOLDEN_SPEC = "device=npu,compute_dtype=fp32,p_dtype=out_dtype"
COMPARE_MODES = ("two_way", "three_way")
OUTPUT_ROOT = Path(__file__).resolve().parent / "out"


def default_out_dir(mode: str) -> str:
    return str(OUTPUT_ROOT / f"{mode}-{time.strftime('%Y%m%d-%H%M%S')}")


def add_common_args(
    parser: argparse.ArgumentParser, default_out_dir: Optional[str] = None
) -> None:
    parser.add_argument("--out-dir", default=default_out_dir)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--mem-guard-gb", type=float, default=None)
    parser.add_argument("--graph", action="store_true", help="图模式执行")


def add_golden_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--golden", default=DEFAULT_GOLDEN_SPEC)
    parser.add_argument(
        "--compare-mode", choices=list(COMPARE_MODES), default=COMPARE_MODES[0]
    )


def add_case_filter_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--testset", default=None)
    parser.add_argument("--skip-cases", default=None)
