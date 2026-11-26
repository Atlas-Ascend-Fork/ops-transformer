# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""阶段引擎：gen → golden → run → compare；性能是 run 的模式（--perf）。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from . import guards
from . import perf as perflib
from .files import CaseFiles, inputs_digest
from .case import Case
from .golden_spec import GoldenSpec

LOGGER = logging.getLogger("fw")

DEFAULT_MEM_GUARD_GB = guards.DEFAULT_MEM_GUARD_GB
HBM_GUARD_RATIO = guards.HBM_GUARD_RATIO

ALL_STAGES = ("gen", "golden", "run", "compare")


UNSUPPORTED_ERROR_PATTERNS = (
    "Failed to execute tiling function",
    "tiling function",
    "is not supported",
    "not supported",
    "Unsupported",
)


class CaseUnsupported(RuntimeError):
    """算子明确拒绝用例（tiling 失败/不支持），按 SKIP 处理而不是 FAIL。"""


class SessionFatal(RuntimeError):
    """设备/队列已不可恢复（如 torch_npu CAN_EXIT），整轮立即终止。"""


FATAL_ERROR_PATTERNS = (
    "CAN_EXIT",
    "The process exits for this inner error",
    "ERR00100",
)


def is_unsupported_error(exc: BaseException) -> bool:
    text = str(exc)
    return any(pattern in text for pattern in UNSUPPORTED_ERROR_PATTERNS)


def is_fatal_error(exc: BaseException) -> bool:
    text = str(exc)
    return any(pattern in text for pattern in FATAL_ERROR_PATTERNS)


def parse_stages(text: str) -> tuple:
    if not text or text == "all":
        return ALL_STAGES
    picked = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = set(picked) - set(ALL_STAGES)
    if unknown:
        raise ValueError(f"unknown stages: {sorted(unknown)}")
    return picked


@dataclass
class RunnerConfig:
    device_id: int = 0
    graph_mode: bool = False
    stages: tuple = ALL_STAGES
    golden_spec: GoldenSpec = field(default_factory=GoldenSpec)
    compare_mode: str = "two_way"  # two_way（默认）| three_way
    out_dir: Optional[str] = None
    save_data: bool = False
    mem_guard_gb: Optional[float] = None
    perf: bool = False
    perf_runs: int = 5
    perf_warmup: int = 0
    peak_tflops: float = 0.0
    peak_bw_gbs: float = 0.0
    prof_dir: Optional[str] = None  # msprof 输出目录（频率来源）


@dataclass
class RunResult:
    case_id: str
    category: str = ""
    passed: bool = True
    skipped: bool = False
    fatal: bool = False
    reason: str = ""
    metrics: Optional[Dict[str, Dict[str, Any]]] = None
    perf_rows: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""


class Runner:
    def __init__(self, config: RunnerConfig, op):
        self.config = config
        self.op = op
        self.spec = config.golden_spec
        self.root = Path(config.out_dir or ".out")
        self.save_data = config.save_data
        self.root.mkdir(parents=True, exist_ok=True)
        self.flusher = perflib.L2Flusher(
            torch.device(f"npu:{config.device_id}")
            if hasattr(torch, "npu") and torch.npu.is_available()
            else torch.device("cpu"),
            enabled=config.perf,
        )
        self.records: List[Dict[str, Any]] = []
        self.started_at = (
            __import__("datetime").datetime.now().isoformat(timespec="seconds")
        )
        self.run_id = __import__("datetime").datetime.now().strftime("%Y%m%d-%H%M%S")
        # 落盘策略：分阶段执行必须落盘；单趟执行只在失败时留最小档
        if config.perf and "run" not in config.stages:
            raise ValueError("--perf requires stages to include run")
        self.on_device = config.golden_spec.device == "npu"
        self.mem_guard_gb = guards.resolve_mem_guard(config.mem_guard_gb)
        self.hbm_guard_gb = guards.resolve_hbm_guard(config.device_id)

    # ---------------------------------------------------------------- helpers
    def _golden(
        self, inputs, case: Case, spec: GoldenSpec, digest: str, art: CaseFiles
    ):
        # inputs/golden 同目录；只有分阶段执行时持久化，peer 参考不落盘
        store = spec.provider != "peer"
        if store and art.exists("golden.pt"):
            meta = art.load_json("golden.json") or {}
            if (
                meta.get("spec_fp") == spec.fingerprint()
                and meta.get("inputs_digest") == digest
            ):
                return art.load_tensors("golden.pt")
        started = time.time()
        payload = self.op.golden(
            inputs, case.params, spec, keep_on_device=self.on_device
        )
        if store and self.save_data:
            art.save_tensors("golden.pt", payload)
            art.save_json(
                "golden.json",
                {
                    "case_id": case.id,
                    "spec": spec.describe(),
                    "spec_fp": spec.fingerprint(),
                    "inputs_digest": digest,
                    "seconds": round(time.time() - started, 3),
                },
            )
        return payload

    def _execute(self, inputs, params, mode: str, keep_on_device: bool = False):
        try:
            return self.op.execute(
                inputs,
                params,
                mode,
                device_id=self.config.device_id,
                keep_on_device=keep_on_device,
            )
        except RuntimeError as exc:
            if is_unsupported_error(exc):
                raise CaseUnsupported(str(exc)) from exc
            raise

    def _move(self, inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        moved = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor) and value.device != device:
                value = value.to(device)
            moved[key] = value
        return moved

    # ------------------------------------------------------------------- main
    def run(self, case: Case) -> RunResult:
        result = RunResult(case_id=case.id)
        result.category = "perf" if self.config.perf else self.op.category(case)
        try:
            return self._run(case, result)
        finally:
            try:
                self.records.extend(self._make_records(case, result))
            except Exception as exc:  # noqa: BLE001 - 记录失败不影响用例结论
                LOGGER.warning("[results] record failed for %s: %s", case.id, exc)

    def _make_records(self, case: Case, result: RunResult):
        if result.perf_rows:
            total = len(result.perf_rows)
            records = []
            for index, perf in enumerate(result.perf_rows, 1):
                status = (
                    "pass"
                    if perf.get("passed")
                    else "fail"
                    if "metrics" in perf
                    else "perf"
                )
                records.append(
                    self._make_record(case, result, perf, index, total, status)
                )
            return records
        return [self._make_record(case, result, None, 1, 1)]

    def _make_record(
        self,
        case: Case,
        result: RunResult,
        perf: Optional[Dict[str, Any]],
        run_index: int,
        runs: int,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        status = status or (
            "fatal"
            if result.fatal
            else "skip"
            if result.skipped
            else "error"
            if result.error
            else "pass"
            if result.passed
            else "fail"
        )
        flat: Dict[str, Any] = {
            "run_id": self.run_id,
            "case_id": case.id,
            "op": self.op.name,
            "category": result.category,
            "status": status,
            "detail": result.error or result.reason,
            "tags": list(case.tags),
            "seed": case.seed,
            "compare_mode": self.config.compare_mode,
            "golden_spec": self.spec.describe(),
            "golden_fingerprint": self.spec.fingerprint(),
            "run_index": run_index,
            "runs": runs,
            "leg": "graph" if self.config.graph_mode else "eager",
        }
        metric_names = getattr(self.op, "metric_columns", None)
        if callable(metric_names):
            for prefix in metric_names():
                for key in ("passed", "fail_ratio", "max_abs"):
                    flat.setdefault(f"{prefix}_{key}", "")
        metrics = perf.get("metrics") if perf else result.metrics
        for prefix, payload in (metrics or {}).items():
            if payload:
                for key in ("passed", "fail_ratio", "max_abs"):
                    flat[f"{prefix}_{key}"] = payload.get(key)
        if perf:
            flat.update(
                {
                    "duration_us": perf.get("duration_us"),
                    "tiling_key": perf.get("tiling_key"),
                    "flops": perf.get("flops"),
                    "bytes": perf.get("bytes"),
                    "theory_us": perf.get("theory_us"),
                    "theory_bound": perf.get("theory_bound"),
                    "mfu_pct": perf.get("mfu_pct"),
                    "mbu_pct": perf.get("mbu_pct"),
                    "mfu_std_pct": perf.get("mfu_std_pct"),
                }
            )
        try:
            flat.update(self.op.result_columns(case.params))
        except Exception as exc:  # noqa: BLE001 - 算子列失败不丢结果
            LOGGER.warning("[results] op columns failed for %s: %s", case.id, exc)
        return flat

    def _check_guards(self, case: Case, stages, result: RunResult) -> bool:
        reason = guards.case_guard_reason(
            case,
            self.op,
            spec=self.spec,
            compare_mode=self.config.compare_mode,
            stages=stages,
            mem_guard_gb=self.mem_guard_gb,
            hbm_guard_gb=self.hbm_guard_gb,
            persist=self.save_data,
        )
        if reason:
            result.skipped = True
            result.reason = reason
            return True
        return False

    def _stage_gen(self, case: Case, stages, device, art: CaseFiles):
        if "gen" in stages:
            gen_device = device if self.on_device else torch.device("cpu")
            inputs = self.op.build_inputs(case.params, gen_device, case.seed)
            digest = inputs_digest(inputs, case.seed)
            if self.save_data:
                art.save_tensors("inputs.pt", inputs)
                art.save_json(
                    "inputs_meta.json",
                    {
                        "seed": case.seed,
                        "digest": digest,
                        "case_sig": case.signature_digest(),
                    },
                )
            return inputs, digest
        inputs = art.load_tensors("inputs.pt")
        meta = art.load_json("inputs_meta.json") or {}
        if inputs is None:
            raise FileNotFoundError(
                f"no saved inputs for {case.id}; run --stages with gen first"
            )
        digest = meta.get("digest", inputs_digest(inputs, case.seed))
        saved_sig = meta.get("case_sig")
        if saved_sig and saved_sig != case.signature_digest():
            raise RuntimeError(
                f"saved data signature mismatch for {case.id}: "
                f"saved data was produced by a different case/norm version; "
                f"rerun with --stages=gen"
            )
        return inputs, digest

    def _stage_golden(self, case: Case, stages, inputs, digest, device, art):
        if "golden" not in stages:
            return None
        # NPU 真值需要设备张量（分阶段时输入从已保存文件载入的是 CPU）
        golden_inputs = self._move(inputs, device) if self.on_device else inputs
        return self._golden(golden_inputs, case, self.spec, digest, art)

    def _stage_run(self, case: Case, stages, inputs, device, art, result, golden=None):
        if "run" not in stages:
            return None
        if self.config.perf:
            # 每次性能执行都是独立的 run，并可在该次执行后立即 compare
            result.perf_rows = self._perf(case, device, art, inputs, golden)
            return None
        dev_inputs = self._move(inputs, device)
        npu_out = self._execute(
            dev_inputs,
            case.params,
            "graph" if self.config.graph_mode else "eager",
            keep_on_device=self.on_device,
        )
        if self.save_data:
            art.save_tensors("npu_out.pt", npu_out)
        return npu_out

    def _compare_op(self, case, golden, npu_out, inputs, device, art):
        if golden is None:
            raise RuntimeError("compare stage requires golden")

        def peer(peer_inputs, peer_params, peer_spec):
            if self.on_device:
                peer_inputs = self._move(peer_inputs, device)
            digest = (art.load_json("inputs_meta.json") or {}).get("digest", "")
            return self._golden(
                peer_inputs,
                Case(self.op.name, peer_params),
                peer_spec,
                digest,
                art,
            )

        return self.op.compare(
            golden,
            npu_out,
            case.params,
            self.spec,
            self.config.compare_mode,
            inputs=inputs,
            peer=peer,
        )

    def _stage_compare(
        self, case, stages, result: RunResult, golden, npu_out, inputs, device, art
    ):
        if "compare" not in stages:
            return
        result.metrics, result.passed = self._compare_op(
            case, golden, npu_out, inputs, device, art
        )

    def _run(self, case: Case, result: RunResult) -> RunResult:
        stages = tuple(self.config.stages)
        device = torch.device(f"npu:{self.config.device_id}")
        if self._check_guards(case, stages, result):
            return result

        art = CaseFiles(self.root, case.id)
        inputs = golden = npu_out = None
        try:
            inputs, digest = self._stage_gen(case, stages, device, art)
            golden = self._stage_golden(case, stages, inputs, digest, device, art)
            if "compare" in stages and golden is None:
                golden = art.load_tensors("golden.pt")
            npu_out = self._stage_run(case, stages, inputs, device, art, result, golden)
            if not self.config.perf:
                self._stage_compare(
                    case, stages, result, golden, npu_out, inputs, device, art
                )
        except CaseUnsupported as exc:
            result.skipped = True
            result.reason = f"op rejected case: {exc}"
        except Exception as exc:  # noqa: BLE001 - 单 case 失败不中断整批
            result.passed = False
            result.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("case failed: %s", case.id)
            if is_fatal_error(exc):
                result.fatal = True
                LOGGER.error(
                    "[FATAL] device/queue unrecoverable at %s: %s", case.id, exc
                )
                raise SessionFatal(str(exc)) from exc
        return result

    def _perf(
        self,
        case: Case,
        device: torch.device,
        art: CaseFiles,
        inputs=None,
        golden=None,
    ):
        if inputs is None:
            inputs = art.load_tensors("inputs.pt")
        dev_inputs = self._move(inputs, device)
        mode = "graph" if self.config.graph_mode else "eager"
        profiler = perflib.Profiler(self.config.prof_dir, extractor=self.op.tiling_key)
        workload = self.op.theoretical_workload(case.params) or {}
        rows = []
        warmup = max(self.config.perf_warmup, 0)
        measured = max(self.config.perf_runs, 1)
        for index in range(warmup + measured):
            profiler.start()
            self.flusher.flush()
            npu_out = self._execute(
                dev_inputs, case.params, mode, keep_on_device=self.on_device
            )
            try:
                sample = profiler.collect()
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("[perf] profiling read failed: %s", exc)
                sample = None
            if index < warmup:
                continue
            duration = sample.get("duration_us") if sample else None
            tiling_key = sample.get("tiling_key") if sample else None
            theory_us, theory_bound = perflib.theory_time(
                workload.get("flops"),
                workload.get("bytes"),
                self.config.peak_tflops,
                self.config.peak_bw_gbs,
            )
            row = {
                "case_id": case.id,
                "op": self.op.name,
                **self.op.perf_context(case.params),
                "run_index": index - warmup + 1,
                "runs": measured,
                "leg": mode,
                "duration_us": duration,
                "flops": workload.get("flops"),
                "bytes": workload.get("bytes"),
                "theory_us": theory_us,
                "theory_bound": theory_bound,
                "mfu_pct": perflib.mfu_pct(
                    workload.get("flops"), duration, self.config.peak_tflops
                ),
                "mbu_pct": perflib.mbu_pct(
                    workload.get("bytes"), duration, self.config.peak_bw_gbs
                ),
                "tiling_key": tiling_key,
            }
            if "compare" in self.config.stages:
                row["metrics"], row["passed"] = self._compare_op(
                    case, golden, npu_out, inputs, device, art
                )
            rows.append(row)
        return rows
