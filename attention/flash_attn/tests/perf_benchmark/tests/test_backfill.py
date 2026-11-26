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
from pathlib import Path

from perf_benchmark.backfill import backfill


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_backfill_reads_current_msprof_columns(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "perf_benchmark.backfill.perflib.device_aicore_num",
        lambda device_id=0: 28,
    )
    monkeypatch.setattr(
        "perf_benchmark.backfill.get_device_spec", lambda device_id=0: None
    )
    monkeypatch.setattr(
        "perf_benchmark.backfill.current_device_model",
        lambda device_id=0: "test-device",
    )
    prof = tmp_path / "profiling"
    output = prof / "PROF_x" / "mindstudio_profiler_output"
    output.mkdir(parents=True)
    _write_csv(
        output / "task_time_1.csv",
        [
            {
                "Device_id": "0",
                "kernel_name": "FlashAttn_abc_67174400",
                "stream_id": "1",
                "task_id": "2",
                "task_time(us)": "12.5",
                "task_start(us)": "100.0",
            }
        ],
        [
            "Device_id",
            "kernel_name",
            "stream_id",
            "task_id",
            "task_time(us)",
            "task_start(us)",
        ],
    )
    _write_csv(
        output / "op_summary_1.csv",
        [
            {
                "Device_id": "0",
                "Stream ID": "1",
                "Task ID": "2",
                "OP Type": "FlashAttn",
                "Block Num": "28",
                "Mix Block Num": "56",
                "aic_total_cycles": "483549",
                "aicore_time(us)": "12.58",
                "aiv_total_cycles": "1000000",
                "aiv_time(us)": "12.0",
                "cube_utilization(%)": "80.5",
            }
        ],
        [
            "Device_id",
            "Stream ID",
            "Task ID",
            "OP Type",
            "Block Num",
            "Mix Block Num",
            "aic_total_cycles",
            "aicore_time(us)",
            "aiv_total_cycles",
            "aiv_time(us)",
            "cube_utilization(%)",
        ],
    )
    results = tmp_path / "results.csv"
    fields = [
        "case_id",
        "run_index",
        "status",
        "device_model",
        "standard_freq_mhz",
        "standard_peak_gbs",
        "standard_peak_tflops",
        "aic_core_num",
        "aiv_core_num",
        "flops",
        "bytes",
        "theory_us",
        "theory_bound",
        "theory_us_std",
        "theory_bound_std",
        "duration_us",
        "tiling_key",
        "mfu_pct",
        "mbu_pct",
        "mfu_std_pct",
    ]
    _write_csv(
        results,
        [
            {
                "case_id": "flash_attn/case",
                "run_index": "1",
                "status": "perf",
                "flops": "1000",
                "bytes": "2000",
            }
        ],
        fields,
    )
    manifest = tmp_path / "perf_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [{"seq": 1, "case_id": "flash_attn/case", "run_index": 1}],
                "peak_tflops": 0.0,
                "peak_gbs": 100.0,
                "standard_freq_mhz": 1650.0,
                "standard_peak_gbs": 100.0,
            }
        ),
        encoding="utf-8",
    )

    assert backfill(results, manifest, prof) == 0
    row = next(csv.DictReader(results.open(encoding="utf-8")))
    assert float(row["duration_us"]) == 12.5
    assert row["tiling_key"] == "67174400"
    assert row["device_model"] == "test-device"
    assert float(row["standard_freq_mhz"]) == 1650.0
    assert float(row["standard_peak_gbs"]) == 100.0
    assert float(row["standard_peak_tflops"]) > 0
    assert row["aic_core_num"] == "28"
    assert row["aiv_core_num"] == "56"
    assert float(row["theory_us"]) > 0
    assert row["theory_bound"] in {"cube", "bandwidth"}
    assert float(row["theory_us_std"]) > 0
    assert row["theory_bound_std"] in {"cube", "bandwidth"}
    assert float(row["mfu_pct"]) > 0
    assert float(row["mbu_pct"]) > 0
    assert float(row["mfu_std_pct"]) > 0
    assert row["Block Num"] == "28"
    assert row["cube_utilization(%)"] == "80.5"
    assert float(row["aic_freq_mhz"]) > 0
    assert float(row["aiv_freq_mhz"]) > 0


def test_backfill_without_aic_freq_leaves_mfu_empty(tmp_path: Path):
    prof = tmp_path / "profiling"
    output = prof / "PROF_x" / "mindstudio_profiler_output"
    output.mkdir(parents=True)
    _write_csv(
        output / "task_time_1.csv",
        [
            {
                "Device_id": "0",
                "kernel_name": "FlashAttn_abc_67174400",
                "stream_id": "1",
                "task_id": "2",
                "task_time(us)": "12.5",
                "task_start(us)": "100.0",
            }
        ],
        [
            "Device_id",
            "kernel_name",
            "stream_id",
            "task_id",
            "task_time(us)",
            "task_start(us)",
        ],
    )
    results = tmp_path / "results.csv"
    _write_csv(
        results,
        [
            {
                "case_id": "flash_attn/case",
                "run_index": "1",
                "status": "perf",
                "flops": "1000",
                "bytes": "2000",
            }
        ],
        [
            "case_id",
            "run_index",
            "status",
            "flops",
            "bytes",
            "theory_us",
            "duration_us",
            "mfu_pct",
            "mbu_pct",
        ],
    )
    manifest = tmp_path / "perf_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "rows": [{"seq": 1, "case_id": "flash_attn/case", "run_index": 1}],
                "peak_tflops": 0.0,
                "peak_gbs": 100.0,
                "standard_freq_mhz": 1650.0,
                "standard_peak_gbs": 100.0,
            }
        ),
        encoding="utf-8",
    )

    assert backfill(results, manifest, prof) == 0
    row = next(csv.DictReader(results.open(encoding="utf-8")))
    assert row["theory_us"] == ""
    assert row["theory_bound"] == ""
    assert row["mfu_pct"] == ""
    assert float(row["mbu_pct"]) > 0
