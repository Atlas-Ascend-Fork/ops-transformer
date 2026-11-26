# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import csv
import json
from types import SimpleNamespace
from pathlib import Path

from perf_benchmark import perf_worker


class _FakeRunner:
    instances = []

    def __init__(self, config, package):
        self.config = config
        self.package = package
        self.records = []
        _FakeRunner.instances.append(self)

    def run(self, case):
        if self.config.perf:
            self.records.append(
                {
                    "case_id": case.id,
                    "status": "perf",
                    "run_index": 1,
                    "runs": 1,
                    "detail": "",
                }
            )
            return SimpleNamespace()
        return SimpleNamespace(
            passed=False,
            skipped=False,
            error="",
            reason="compare mismatch",
            metrics={"attn": {"passed": False, "fail_ratio": 0.5, "max_abs": 0.1}},
        )


def test_verify_uses_shared_golden_and_compare_fields(tmp_path: Path, monkeypatch):
    cases = tmp_path / "cases.json"
    cases.write_text(
        '{"cases":[{"op":"flash_attn","params":{"B":1,"N1":4,"D":64}}]}',
        encoding="utf-8",
    )
    out = tmp_path / "run"
    _FakeRunner.instances = []
    monkeypatch.setattr(perf_worker, "Runner", _FakeRunner)

    rc = perf_worker.main(
        [
            "--cases",
            str(cases),
            "--out-dir",
            str(out),
            "--perf-runs",
            "1",
            "--verify",
            "--golden",
            "device=cpu,compute_dtype=fp64,p_dtype=fp32",
            "--compare-mode",
            "three_way",
        ]
    )

    assert rc == 0
    with (out / "results.csv").open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["status"] == "fail"
    assert row["detail"] == "compare mismatch"
    assert row["compare_mode"] == "three_way"
    assert row["golden_spec"].startswith("cpu/fp64/p=fp32")
    assert row["attn_passed"] == "False"
    assert row["attn_fail_ratio"] == "0.5"

    verify = json.loads((out / "verify.json").read_text(encoding="utf-8"))
    assert verify[row["case_id"]]["reason"] == "compare mismatch"
