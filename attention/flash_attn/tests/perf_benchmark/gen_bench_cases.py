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
"""Generate configurable dense flash_attn performance sweep cases."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import yaml

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from harness.case import Case, dump_cases_file  # noqa: E402
from harness.ops.flash_attn import PACKAGE  # noqa: E402

DEFAULT_CONFIG = Path(__file__).with_name("bench_config.yaml")
COMMON_KEYS = {
    "dtype",
    "layouts",
    "D",
    "DV",
    "N1",
    "N2",
    "mask_mode",
    "win_left",
    "win_right",
    "scale",
    "return_softmax_lse",
    "phase_threshold",
}
EXPERIMENT_KEYS = {
    "s2_curve": {"enabled", "batch", "s1", "s2_aligned", "s2_tail_sizes"},
    "batch_linearity": {"enabled", "batch_values", "shapes"},
    "s2_balance": {"enabled", "batch", "S2", "fill", "distributions", "shapes"},
}


def _check_keys(data, allowed, where):
    unknown = set(data) - set(allowed)
    if unknown:
        raise ValueError(f"{where}: unknown keys: {sorted(unknown)}")


def _require(data, keys, where):
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{where}: missing keys: {missing}")


def _load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: config must be a mapping")
    _check_keys(raw, {"version", "common", "experiments"}, "config")
    if raw.get("version") != 1:
        raise ValueError(f"{path}: unsupported version {raw.get('version')!r}")
    common = raw.get("common")
    experiments = raw.get("experiments")
    if not isinstance(common, dict) or not isinstance(experiments, dict):
        raise ValueError(f"{path}: common/experiments must be mappings")
    _check_keys(common, COMMON_KEYS, "common")
    _require(common, COMMON_KEYS - {"phase_threshold"}, "common")
    if not common["layouts"]:
        raise ValueError("common.layouts must not be empty")
    if int(common["N1"]) % int(common["N2"]):
        raise ValueError("common.N1 must be a multiple of common.N2")
    _check_keys(experiments, EXPERIMENT_KEYS, "experiments")
    for name, allowed in EXPERIMENT_KEYS.items():
        section = experiments.get(name)
        if section is None:
            experiments[name] = {"enabled": False}
            continue
        if not isinstance(section, dict):
            raise ValueError(f"experiments.{name} must be a mapping")
        _check_keys(section, allowed, f"experiments.{name}")
        section.setdefault("enabled", True)
    return raw


def _phase(s1, threshold):
    return "decode" if s1 <= threshold else "prefill"


def _case(layout, b, s1, s2, experiment, common):
    scale = common["scale"]
    if scale == "inv_sqrt_d":
        scale = 1.0 / math.sqrt(int(common["D"]))
    params = {
        "B": b,
        "N1": common["N1"],
        "N2": common["N2"],
        "S1": s1,
        "S2": s2,
        "D": common["D"],
        "DV": common["DV"],
        "Dtype": common["dtype"],
        "layout_q": layout,
        "layout_kv": layout,
        "layout_out": layout,
        "mask_mode": common["mask_mode"],
        "win_left": common["win_left"],
        "win_right": common["win_right"],
        "scale": float(scale),
        "return_softmax_lse": bool(common["return_softmax_lse"]),
    }
    return Case(op="flash_attn", params=params, tags=["perf_rdv", "bench", experiment])


def _distribute(total, weights, cap):
    weights = [max(weight, 1e-9) for weight in weights]
    raw = [weight * total / sum(weights) for weight in weights]
    values = [min(cap, int(value)) for value in raw]
    remaining = total - sum(values)
    order = sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)
    while remaining > 0:
        moved = False
        for index in order:
            if values[index] < cap:
                values[index] += 1
                remaining -= 1
                moved = True
                if remaining == 0:
                    break
        if not moved:
            break
    if remaining:
        raise ValueError(f"cannot distribute {total} into {len(weights)} x <= {cap}")
    return values


def _balance_values(distribution, count, total, cap):
    if distribution == "uniform":
        weights = [1.0] * count
    elif distribution == "ramp":
        weights = [float(i + 1) for i in range(count)]
    elif distribution == "skew":
        weights = [1.0 / (i + 1) ** 1.5 for i in range(count)]
    elif distribution == "outlier":
        weights = [8.0] + [1.0] * (count - 1)
    elif distribution == "alternating":
        weights = [2.0 if i % 2 == 0 else 1.0 for i in range(count)]
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    return _distribute(total, weights, cap)


def _add(cases, meta, case, info):
    if case.id in meta:
        raise ValueError(f"duplicate bench case: {case.id}")
    reason = PACKAGE.legalize(case.params)
    if reason:
        raise ValueError(f"illegal generated case {case.id}: {reason}")
    cases.append(case)
    meta[case.id] = info


def _shape_map(rows, key_s2, where):
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{where}: shape must be a mapping")
        _check_keys(row, {"phase", "S1", "S2"}, where)
        _require(row, {"phase", "S1"}, where)
        phase = str(row["phase"])
        if phase in result:
            raise ValueError(f"{where}: duplicate phase {phase}")
        result[phase] = [int(row["S1"]), int(row.get("S2", key_s2))]
    return result


def generate(config):
    common = config["common"]
    experiments = config["experiments"]
    threshold = int(common.get("phase_threshold", 16))
    layouts = list(common["layouts"])
    cases = []
    meta = {}
    counts = Counter()

    s2 = experiments["s2_curve"]
    if s2.get("enabled"):
        s2_curve = sorted(
            {*map(int, s2["s2_aligned"])}
            | {
                int(base) + int(tail)
                for base in s2["s2_aligned"]
                for tail in s2["s2_tail_sizes"]
            }
        )
        for layout in layouts:
            for s1 in s2["s1"]:
                for value in s2_curve:
                    case = _case(
                        layout, int(s2["batch"]), int(s1), value, "s2_curve", common
                    )
                    _add(
                        cases,
                        meta,
                        case,
                        {
                            "experiment": "s2_curve",
                            "layout": layout,
                            "phase": _phase(int(s1), threshold),
                            "B": int(s2["batch"]),
                            "S1": int(s1),
                            "S2": value,
                        },
                    )
                    counts["s2_curve"] += 1

    batch = experiments["batch_linearity"]
    if batch.get("enabled"):
        shapes = _shape_map(batch["shapes"], 0, "experiments.batch_linearity.shapes")
        for layout in layouts:
            for phase, (s1, s2) in shapes.items():
                for b in batch["batch_values"]:
                    case = _case(layout, int(b), s1, s2, "batch_linearity", common)
                    _add(
                        cases,
                        meta,
                        case,
                        {
                            "experiment": "batch_linearity",
                            "layout": layout,
                            "phase": phase,
                            "B": int(b),
                            "S1": s1,
                            "S2": s2,
                        },
                    )
                    counts["batch_linearity"] += 1

    balance = experiments["s2_balance"]
    if balance.get("enabled"):
        shapes = _shape_map(
            balance["shapes"], int(balance["S2"]), "experiments.s2_balance.shapes"
        )
        total = int(int(balance["batch"]) * int(balance["S2"]) * float(balance["fill"]))
        for layout in layouts:
            for phase, (s1, s2) in shapes.items():
                for distribution in balance["distributions"]:
                    values = _balance_values(
                        distribution, int(balance["batch"]), total, int(balance["S2"])
                    )
                    case = _case(
                        layout,
                        int(balance["batch"]),
                        s1,
                        int(balance["S2"]),
                        "s2_balance",
                        common,
                    )
                    case.params["seqused_kv"] = values
                    _add(
                        cases,
                        meta,
                        case,
                        {
                            "experiment": "s2_balance",
                            "layout": layout,
                            "phase": phase,
                            "distribution": distribution,
                            "B": int(balance["batch"]),
                            "S1": s1,
                            "S2": int(balance["S2"]),
                            "total_kv": total,
                        },
                    )
                    counts["s2_balance"] += 1

    manifest = {
        "version": 1,
        "common": dict(common),
        "experiments": dict(experiments),
        "counts": dict(counts),
        "cases": meta,
    }
    return cases, manifest


def write_cases(config_path, out_dir):
    config = _load_config(config_path)
    cases, manifest = generate(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_cases_file(cases, out_dir / "bench_cases.json")
    (out_dir / "bench_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return cases, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("perf_benchmark/generated")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        config = _load_config(args.config)
        cases, manifest = generate(config)
        print(f"common: {json.dumps(config['common'], ensure_ascii=False)}")
        for name, count in manifest["counts"].items():
            print(f"{name}: {count}")
        print(f"total: {len(cases)}")
        return 0

    cases, _ = write_cases(args.config, args.out_dir)
    print(f"generated {len(cases)} cases -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
