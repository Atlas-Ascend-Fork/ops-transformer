#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Merge all QuantSparseFlashMla msprof runs into one Excel result file."""

import argparse
import logging
import os
import sys
from pathlib import Path


SMLA_PYTEST_PATH = Path(__file__).resolve().parents[3] / "sparse_flash_mla/tests/pytest"
if str(SMLA_PYTEST_PATH) not in sys.path:
    sys.path.insert(0, str(SMLA_PYTEST_PATH))

from perf_data_common import (  # noqa: E402
    QSMLA_PERF_CONFIG as PERF_CONFIG,
    process_all_profiler_data,
)


LOGGER = logging.getLogger(__name__)


def _get_option(source, name, default):
    return getattr(source, name, default)


def process_profiler_data(
    test_result_path,
    profile_root=".",
    output_path=None,
    is_compare=False,
    perf_golden_path="perf_golden.xlsx",
    perf_threshold=10,
    delete_perf_files=False,
    testcase_paths=None,
    merge_into=None,
):
    """Process all QSMLA profiles, including the legacy Namespace call form."""
    if not isinstance(test_result_path, (str, os.PathLike)):
        options = test_result_path
        test_result_path = _get_option(options, "test_result_path", "result.xlsx")
        profile_root = _get_option(options, "profile_root", profile_root)
        output_path = _get_option(options, "output_path", output_path)
        merge_into = _get_option(options, "merge_into", merge_into)
        is_compare = _get_option(options, "is_compare", is_compare)
        perf_golden_path = _get_option(options, "perf_golden_path", perf_golden_path)
        perf_threshold = _get_option(options, "perf_threshold", perf_threshold)
        delete_perf_files = _get_option(
            options,
            "delete_perf_files",
            _get_option(options, "delete_perf_file", delete_perf_files),
        )
        testcase_paths = _get_option(options, "testcase_paths", testcase_paths)

    if testcase_paths is None:
        testcase_paths = (os.environ.get("QSMLA_PT_DIR", "qsmla_testcase"),)

    return process_all_profiler_data(
        config=PERF_CONFIG,
        test_result_path=test_result_path,
        profile_root=profile_root,
        output_path=output_path,
        merge_into=merge_into,
        is_compare=is_compare,
        perf_golden_path=perf_golden_path,
        perf_threshold=perf_threshold,
        delete_perf_files=delete_perf_files,
        testcase_paths=testcase_paths,
    )


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalised = str(value).strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Merge all QuantSparseFlashMla msprof data into one Excel table"
    )
    parser.add_argument("--test_result_path", default="result.xlsx")
    parser.add_argument("--profile_root", default=".")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--output_path", help="write a separate performance export")
    output.add_argument(
        "--merge_into", help="fill matching cases in this accuracy workbook"
    )
    parser.add_argument(
        "--is_compare",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="compare durations with the performance baseline",
    )
    parser.add_argument("--perf_golden_path", default="perf_golden.xlsx")
    parser.add_argument("--perf_threshold", type=float, default=10)
    parser.add_argument(
        "--delete_perf_file",
        "--delete_perf_files",
        dest="delete_perf_files",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="delete PROF* folders after a successful merge",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    try:
        process_profiler_data(args)
    except (FileNotFoundError, ValueError, OSError) as error:
        LOGGER.error("Failed to process profiler data: %s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
