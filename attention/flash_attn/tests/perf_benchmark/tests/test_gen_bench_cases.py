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
import argparse
import re
from pathlib import Path

import pytest
import yaml

from cli_common import (
    COMPARE_MODES,
    DEFAULT_GOLDEN_SPEC,
    OUTPUT_ROOT,
    add_golden_args,
    default_out_dir,
)
from perf_benchmark import run_perf
from perf_benchmark.gen_bench_cases import DEFAULT_CONFIG, _load_config, generate
from perf_benchmark.plot_bench import _shape_lookup


def test_default_config_generates_expected_sweep():
    config = _load_config(DEFAULT_CONFIG)
    cases, manifest = generate(config)

    assert len(cases) == 1020
    assert manifest["counts"] == {
        "s2_curve": 972,
        "batch_linearity": 28,
        "s2_balance": 20,
    }
    assert len({case.id for case in cases}) == len(cases)
    assert manifest["common"]["D"] == 128
    assert manifest["cases"][cases[0].id]["experiment"] in manifest["counts"]


def _write_config(tmp_path: Path, config):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _default_config():
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_config_rejects_unknown_common_key(tmp_path: Path):
    raw = _default_config()
    raw["common"]["unknown"] = 1
    with pytest.raises(ValueError, match="unknown keys"):
        _load_config(_write_config(tmp_path, raw))


def test_config_rejects_missing_common_key(tmp_path: Path):
    raw = _default_config()
    raw["common"].pop("D")
    with pytest.raises(ValueError, match="missing keys"):
        _load_config(_write_config(tmp_path, raw))


def test_disabled_experiments_generate_no_cases(tmp_path: Path):
    raw = _default_config()
    for section in raw["experiments"].values():
        section["enabled"] = False
    config = _load_config(_write_config(tmp_path, raw))

    cases, manifest = generate(config)

    assert cases == []
    assert manifest["counts"] == {}


def test_shape_lookup_reads_config_list():
    manifest = {
        "experiments": {
            "batch_linearity": {
                "shapes": [
                    {"phase": "prefill", "S1": 4096, "S2": 6144},
                    {"phase": "decode", "S1": 1, "S2": 24576},
                ]
            }
        }
    }

    assert _shape_lookup(manifest, "batch_linearity") == {
        "prefill": [4096, 6144],
        "decode": [1, 24576],
    }


def test_missing_experiment_section_is_disabled(tmp_path: Path):
    raw = _default_config()
    raw["experiments"].pop("s2_balance")
    config = _load_config(_write_config(tmp_path, raw))

    cases, manifest = generate(config)

    assert "s2_balance" not in manifest["counts"]
    assert all("s2_balance" not in case.tags for case in cases)


def test_run_perf_generates_cases_from_config(tmp_path: Path, monkeypatch):
    out = tmp_path / "run"
    calls = []

    def fake_call(cmd, cwd=None, env=None):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(run_perf.subprocess, "call", fake_call)
    rc = run_perf.main(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--out-dir",
            str(out),
            "--perf-runs",
            "1",
        ]
    )

    assert rc == 0
    assert (out / "cases" / "bench_cases.json").exists()
    assert (out / "cases" / "bench_manifest.json").exists()
    assert f"--cases={out / 'cases' / 'bench_cases.json'}" in calls[0]


def test_run_perf_plots_after_backfill(tmp_path: Path, monkeypatch):
    out = tmp_path / "run"
    plotted = []

    def fake_call(cmd, cwd=None, env=None):
        (out / "results.csv").write_text("", encoding="utf-8")
        (out / "perf_manifest.json").write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(run_perf.subprocess, "call", fake_call)
    monkeypatch.setattr(run_perf, "backfill", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_perf, "_plot_results", plotted.append)
    rc = run_perf.main(
        ["--config", str(DEFAULT_CONFIG), "--out-dir", str(out), "--perf-runs", "1"]
    )

    assert rc == 0
    assert plotted == [out]


def test_run_perf_cases_mode_skips_generation_and_plot(tmp_path: Path, monkeypatch):
    out = tmp_path / "run"
    cases = tmp_path / "cases.json"
    cases.write_text("[]", encoding="utf-8")
    calls = []
    plotted = []

    def fake_call(cmd, cwd=None, env=None):
        calls.append(cmd)
        (out / "results.csv").write_text("", encoding="utf-8")
        (out / "perf_manifest.json").write_text("{}", encoding="utf-8")
        return 0

    monkeypatch.setattr(run_perf.subprocess, "call", fake_call)
    monkeypatch.setattr(run_perf, "backfill", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_perf, "_plot_results", plotted.append)
    rc = run_perf.main(
        [
            "--cases",
            str(cases),
            "--out-dir",
            str(out),
            "--perf-runs",
            "1",
        ]
    )

    assert rc == 0
    assert not (out / "cases").exists()
    assert plotted == []
    assert f"--cases={cases}" in calls[0]


def test_run_perf_cases_accepts_multiple_paths(tmp_path: Path, monkeypatch):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("[]", encoding="utf-8")
    second.write_text("[]", encoding="utf-8")
    calls = []

    def fake_call(cmd, cwd=None, env=None):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(run_perf.subprocess, "call", fake_call)
    rc = run_perf.main(
        [
            "--cases",
            str(first),
            "--cases",
            str(second),
            "--out-dir",
            str(tmp_path / "run"),
            "--perf-runs",
            "1",
        ]
    )

    assert rc == 0
    assert f"--cases={first}" in calls[0]
    assert f"--cases={second}" in calls[0]


def test_run_perf_requires_case_source(tmp_path: Path):
    with pytest.raises(SystemExit):
        run_perf.main(["--out-dir", str(tmp_path / "run")])


def test_run_perf_rejects_config_and_cases_together(tmp_path: Path):
    with pytest.raises(SystemExit):
        run_perf.main(
            [
                "--config",
                str(DEFAULT_CONFIG),
                "--cases",
                str(tmp_path / "cases.json"),
                "--out-dir",
                str(tmp_path / "run"),
            ]
        )


def test_run_perf_verify_passes_shared_golden_args(tmp_path: Path, monkeypatch):
    out = tmp_path / "run"
    cases = tmp_path / "cases.json"
    cases.write_text("[]", encoding="utf-8")
    calls = []

    def fake_call(cmd, cwd=None, env=None):
        calls.append(cmd)
        return 0

    monkeypatch.setattr(run_perf.subprocess, "call", fake_call)
    rc = run_perf.main(
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
    assert "--verify" in calls[0]
    assert "--golden=device=cpu,compute_dtype=fp64,p_dtype=fp32" in calls[0]
    assert "--compare-mode=three_way" in calls[0]


def test_shared_golden_defaults():
    parser = argparse.ArgumentParser()
    add_golden_args(parser)
    args = parser.parse_args([])

    assert args.golden == DEFAULT_GOLDEN_SPEC
    assert args.compare_mode == COMPARE_MODES[0]


def test_default_out_dir_uses_shared_rule():
    for mode in ("functional", "perf", "random"):
        path = Path(default_out_dir(mode))
        assert path.parent == OUTPUT_ROOT
        assert re.fullmatch(rf"{mode}-\d{{8}}-\d{{6}}", path.name)
