# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""flash_attn 精度/性能用例（一个 case 一个 test item）。"""

from __future__ import annotations

import logging

import pytest

LOGGER = logging.getLogger("fw")


def test_case(case, runner, runner_config):
    from harness.runner import SessionFatal

    try:
        result = runner.run(case)
    except SessionFatal as exc:
        pytest.exit(
            f"device/queue unrecoverable (fail-fast): {exc}",
            returncode=1,
        )
    summary = (
        f"[{'PASS' if result.passed else 'FAIL'}] {case.id} "
        f"metrics={result.metrics} perf_rows={len(result.perf_rows)}"
    )
    LOGGER.warning(summary)
    if result.skipped:
        pytest.skip(result.reason)
    if result.error:
        pytest.fail(f"{case.id}: {result.error}")
    if "compare" in runner_config.stages:
        if runner_config.perf:
            assert all(row.get("passed") for row in result.perf_rows), summary
        else:
            assert result.passed, summary
