# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CSV suite manifest reader and case selector for FlashKDA TTK assets."""

import argparse
import ast
import csv
import json
from pathlib import Path

try:
    from .arch_config import ARCHITECTURES, architecture
except ImportError:
    from arch_config import ARCHITECTURES, architecture

CASES_DIR = Path(__file__).resolve().parents[1] / "cases"
SUITE_FILES = {
    "redline": ("flash_kda_redline.csv",),
    "stc": ("flash_kda_stc.csv",),
    "generalized": ("flash_kda_generalized.csv",),
    "profiling": ("flash_kda_perf_profiling.csv",),
    "invalid": ("flash_kda_invalid.csv",),
    "whitebox": ("flash_kda_whitebox_invalid.csv",),
}
SUITE_FILES["public"] = (
    "flash_kda_redline.csv",
    "flash_kda_stc.csv",
    "flash_kda_generalized.csv",
    "flash_kda_invalid.csv",
)
SUITE_FILES["all"] = (
    "flash_kda_redline.csv",
    "flash_kda_stc.csv",
    "flash_kda_generalized.csv",
    "flash_kda_perf_profiling.csv",
    "flash_kda_invalid.csv",
    "flash_kda_whitebox_invalid.csv",
)


def attributes(row):
    attrs = ast.literal_eval(row["attributes"])
    attrs["testcase_name"] = row["testcase_name"]
    return attrs


def recipe(row):
    return json.loads(attributes(row).get("test_config", "{}"))


def read_cases(suite="public", names=None, *, dut=None, arch=None):
    del dut
    arch = arch or architecture()
    if suite not in SUITE_FILES:
        raise ValueError(f"unknown suite: {suite}")
    filenames = SUITE_FILES["all"] if names else SUITE_FILES[suite]
    rows = []
    for filename in filenames:
        path = CASES_DIR / filename
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    if len({r["testcase_name"] for r in rows}) != len(rows):
        raise ValueError("duplicate testcase_name across suite CSV files")
    selected = []
    for row in rows:
        if (row.get("is_enabled") or "true").lower() != "true":
            continue
        test = recipe(row)
        if names and row["testcase_name"] not in names:
            continue
        if arch not in test.get("architectures", ARCHITECTURES):
            raise ValueError(
                f"{row['testcase_name']} is not applicable to {arch}: {test.get('reason')}"
            )
        selected.append(row)
    if names:
        missing = set(names) - {r["testcase_name"] for r in selected}
        if missing:
            raise ValueError(f"unknown/disabled cases: {sorted(missing)}")
    if not selected:
        raise ValueError(f"no cases selected: {suite}")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="public",
        choices=tuple(SUITE_FILES),
    )
    parser.add_argument("--cases", help="comma-separated case IDs")
    parser.add_argument(
        "--ids", action="store_true", help="print one comma-separated list for TTK -t"
    )
    args = parser.parse_args()
    rows = read_cases(args.suite, args.cases.split(",") if args.cases else None)
    print(("," if args.ids else "\n").join(row["testcase_name"] for row in rows))
