# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""pytest 前端：用例参数化、标记、阶段化执行与汇总。

阅读顺序与修改入口见 ../harness/DEVELOPMENT.md。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from cli_common import (  # noqa: E402
    COMPARE_MODES,
    DEFAULT_GOLDEN_SPEC,
    default_out_dir,
)
from harness import loader  # noqa: E402
from harness import results as resultslib  # noqa: E402
from harness.golden_spec import parse_spec  # noqa: E402
from harness.runner import Runner, RunnerConfig, parse_stages  # noqa: E402
from harness.ops.flash_attn import PACKAGE  # noqa: E402

LOGGER = logging.getLogger("fw")

_MARKERS = list(PACKAGE.markers)


def pytest_addoption(parser):
    group = parser.getgroup("flash_attn-fw")
    group.addoption(
        "--cases",
        action="append",
        default=None,
        help="case JSON 文件/目录（可重复）",
    )
    group.addoption(
        "--case-id", default=None, help="用例名过滤：逗号分隔，支持前缀/glob"
    )
    group.addoption("--testset", default=None, help="按 tag/来源过滤：逗号分隔")
    group.addoption("--skip-cases", default=None, help="跳过文件路径")
    group.addoption(
        "--stages", default="all", help="阶段：gen,golden,run,compare 或 all"
    )
    group.addoption(
        "--golden",
        default=DEFAULT_GOLDEN_SPEC,
        help="golden spec（默认 npu + P 落盘到 V dtype），如 device=cpu,p_dtype=fp32 取高精度真值",
    )
    group.addoption(
        "--compare-mode", choices=list(COMPARE_MODES), default=COMPARE_MODES[0]
    )
    group.addoption("--out-dir", default=default_out_dir("functional"))
    group.addoption(
        "--save-data", action="store_true", help="保存 inputs/golden/npu_out"
    )
    group.addoption("--graph", action="store_true", help="图模式执行")
    group.addoption("--device-id", type=int, default=0)
    group.addoption(
        "--mem-guard-gb",
        type=float,
        default=None,
        help="单 case 预估 host 峰值超过该值时跳过；默认取 guards.DEFAULT_MEM_GUARD_GB，0 关闭",
    )


def _split(value):
    return [item.strip() for item in value.split(",") if item.strip()] if value else []


def _collect_cases(config):
    sources = config.getoption("--cases") or []
    if not sources:
        raise pytest.UsageError("no cases: pass --cases")
    cases = loader.load_from_json(sources)
    if not cases:
        raise pytest.UsageError("no cases loaded from --cases")

    skip_opt = config.getoption("--skip-cases")
    patterns = _split(skip_opt)
    if skip_opt and str(skip_opt).endswith(".txt"):
        patterns = loader.load_skip_file(skip_opt)
    selected = loader.select(
        cases,
        case_ids=_split(config.getoption("--case-id")),
        testsets=_split(config.getoption("--testset")),
        skip=patterns,
    )
    if not selected:
        raise pytest.UsageError("no cases matched the given filters")
    return selected


def pytest_generate_tests(metafunc):
    if "case" in metafunc.fixturenames:
        cases = _collect_cases(metafunc.config)
        if not cases:
            raise pytest.UsageError("no cases matched the given filters")
        metafunc.parametrize("case", cases, ids=[case.id for case in cases])


def pytest_collection_modifyitems(config, items):
    for item in items:
        callspec = getattr(item, "callspec", None)
        case_obj = (callspec.params or {}).get("case") if callspec is not None else None
        if case_obj is None or not hasattr(case_obj, "tags"):
            continue
        for tag in case_obj.tags:
            if tag in _MARKERS:
                item.add_marker(getattr(pytest.mark, tag))


@pytest.fixture(scope="session")
def runner_config(request) -> RunnerConfig:
    opts = request.config
    return RunnerConfig(
        stages=parse_stages(opts.getoption("--stages")),
        golden_spec=parse_spec(opts.getoption("--golden")),
        compare_mode=opts.getoption("--compare-mode"),
        graph_mode=opts.getoption("--graph"),
        device_id=opts.getoption("--device-id"),
        out_dir=opts.getoption("--out-dir"),
        save_data=opts.getoption("--save-data"),
        mem_guard_gb=opts.getoption("--mem-guard-gb"),
    )


@pytest.fixture(scope="session")
def runner(runner_config):
    instance = Runner(runner_config, PACKAGE)
    yield instance
    if instance.records:
        root = Path(runner_config.out_dir or default_out_dir("functional"))
        path = resultslib.write_csv(instance.records, root / "results.csv")
        LOGGER.warning("[results] %d cases -> %s", len(instance.records), path)
