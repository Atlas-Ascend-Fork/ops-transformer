# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""性能采集：L2 清理、冷启动计时、频率/MFU/MBU 导出。"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

LOGGER = logging.getLogger("fw")
MB = 1024 * 1024
L2_FLUSH_MB = 256  # 默认驱逐缓冲下限；实际取 max(该值, 2×设备 L2)


class L2Flusher:
    """冷启动：用大于 L2 的读+写流量驱逐整块缓存。

    - 容量 = max(L2_FLUSH_MB, 2×设备 L2)；拿不到 L2 时只用 L2_FLUSH_MB；
    - 每次 flush 对整块 buffer 做 in-place add（读一遍 + 写一遍），避免 memset
      被当成流式写绕过缓存；
    """

    def __init__(self, device, size_mb: int = L2_FLUSH_MB, enabled: bool = True):
        self.enabled = bool(enabled and size_mb > 0)
        self._device = device
        self._buf = None
        self.bytes = 0
        if self.enabled:
            self.bytes = max(int(size_mb) * MB, 2 * self._l2_bytes(device))

    def _ensure(self) -> None:
        if self._buf is None:
            import torch as _torch

            self._buf = _torch.empty(
                self.bytes // 4, dtype=_torch.int32, device=self._device
            )

    @staticmethod
    def _l2_bytes(device) -> int:
        import torch as _torch

        if not (hasattr(_torch, "npu") and _torch.npu.is_available()):
            return 0
        try:
            props = _torch.npu.get_device_properties(getattr(device, "index", 0) or 0)
        except Exception:  # noqa: BLE001 - 拿不到 L2 就退回配置值
            return 0
        for name in ("l2_cache_size", "l2CacheSize", "l2_size", "L2_cache_size"):
            value = getattr(props, name, None)
            if isinstance(value, int) and value > 0:
                return int(value)
        return 0

    def flush(self) -> None:
        if not self.enabled:
            return
        import torch as _torch

        self._ensure()
        self._buf.add_(1)
        if hasattr(_torch, "npu") and _torch.npu.is_available():
            _torch.npu.synchronize()


class Profiler:
    def __init__(self, prof_dir: Optional[str], extractor=None):
        self.prof_dir = Path(prof_dir) if prof_dir else None
        self.extractor = extractor or (lambda _name: None)
        self.samples = self._read(prof_dir)
        self._seen = {}

    def _task_files(self):
        if not self.prof_dir:
            return []
        return sorted(self.prof_dir.glob("**/task_time_*.csv"))

    @staticmethod
    def _read_csv(path):
        try:
            with open(path, encoding="utf-8-sig") as handle:
                return list(csv.DictReader(handle))
        except OSError:
            return []

    def start(self) -> None:
        self._seen = {path: len(self._read_csv(path)) for path in self._task_files()}

    def collect(self):
        if not self.prof_dir:
            return None
        found = None
        for path in self._task_files():
            rows = self._read_csv(path)
            seen = self._seen.get(path, 0)
            self._seen[path] = len(rows)
            for row in rows[seen:]:
                key = self.extractor(row.get("kernel_name", ""))
                if key is None:
                    continue
                try:
                    duration = float(row.get("Task Duration(us)"))
                except (TypeError, ValueError):
                    duration = None
                found = {
                    "duration_us": duration,
                    "tiling_key": key,
                    "start_us": row.get("Task Start Time(us)"),
                }
        return found

    @staticmethod
    def _read(prof_dir: Optional[str]) -> List[tuple]:
        if not prof_dir:
            return []
        import json

        root = Path(prof_dir)
        candidates = list(root.glob("**/msprof_*.json")) + list(
            root.glob("**/mindstudio_profiler_output/msprof_*.json")
        )
        samples: List[tuple] = []
        for path in {p.resolve(): p for p in candidates}.values():
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            except (OSError, json.JSONDecodeError):
                continue
            records = data if isinstance(data, list) else data.get("traceEvents", [])
            for record in records:
                if not isinstance(record, dict) or record.get("ph") != "C":
                    continue
                if "AI Core Freq" not in str(record.get("name", "")):
                    continue
                args = record.get("args") or {}
                try:
                    ts = float(record.get("ts"))
                    value = float(args.get("MHz"))
                except (TypeError, ValueError):
                    continue
                if value > 0:
                    samples.append((ts, value))
        samples.sort()
        return samples

    @property
    def median(self) -> Optional[float]:
        if not self.samples:
            return None
        import statistics

        return statistics.median(value for _, value in self.samples)

    def in_window(self, start_us: float, stop_us: float) -> Optional[float]:
        if not self.samples or stop_us <= start_us:
            return None
        import bisect
        import statistics

        times = [ts for ts, _ in self.samples]
        lo = bisect.bisect_left(times, start_us)
        hi = bisect.bisect_right(times, stop_us)
        if lo >= hi:
            return None
        return statistics.median(self.samples[i][1] for i in range(lo, hi))

    def nearest(self, ts_us: float) -> Optional[float]:
        if not self.samples:
            return None
        import bisect

        times = [ts for ts, _ in self.samples]
        pos = bisect.bisect_left(times, ts_us)
        best = None
        for idx in (pos - 1, pos):
            if 0 <= idx < len(self.samples):
                dist = abs(self.samples[idx][0] - ts_us)
                if best is None or dist < best[0]:
                    best = (dist, self.samples[idx][1])
        return None if best is None else best[1]


def mfu_pct(
    flops: Optional[float], duration_us: float, peak_tflops: float
) -> Optional[float]:
    if not flops or not duration_us or not peak_tflops:
        return None
    return flops / (duration_us * 1e-6) / (peak_tflops * 1e12) * 100.0


def mbu_pct(
    nbytes: Optional[float], duration_us: float, peak_gbs: float
) -> Optional[float]:
    if not nbytes or not duration_us or not peak_gbs:
        return None
    return nbytes / (duration_us * 1e-6) / (peak_gbs * 1e9) * 100.0


def peak_tflops(aicore_num: int, dtype: str, freq_mhz: float):
    """CUBE 峰值：aicore_num*16*(32/sizeof(T))*16 元素/cycle，×2 FLOPs。"""
    size = {"fp8": 1, "int8": 1, "fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8}.get(
        str(dtype), 0
    )
    if not size or aicore_num <= 0 or freq_mhz <= 0:
        return None
    elements_per_cycle = aicore_num * 16 * (32 // size) * 16
    return 2.0 * elements_per_cycle * freq_mhz * 1e6 / 1e12


def device_aicore_num(device_id: int = 0):
    """从设备属性取 CUBE 核数；拿不到返回 None。"""
    try:
        import torch as _torch

        if not (hasattr(_torch, "npu") and _torch.npu.is_available()):
            return None
        props = _torch.npu.get_device_properties(device_id)
        value = getattr(props, "cube_core_num", None)
        return int(value) if value else None
    except Exception:  # noqa: BLE001
        return None


def theory_time(
    flops: Optional[float],
    nbytes: Optional[float],
    peak_tflops: float,
    peak_gbs: float,
) -> Tuple[Optional[float], str]:
    """返回 (理论耗时 us, 瓶颈类型)；两个峰值都可用时才给出结果。"""
    compute_us = flops / (peak_tflops * 1e12) * 1e6 if flops and peak_tflops else None
    memory_us = nbytes / (peak_gbs * 1e9) * 1e6 if nbytes and peak_gbs else None
    if compute_us is None or memory_us is None:
        return None, ""
    if compute_us >= memory_us:
        return compute_us, "cube"
    return memory_us, "bandwidth"


def device_aivcore_num(device_id: int = 0):
    """从设备属性取 Vector 核数；拿不到返回 None。"""
    try:
        import torch as _torch

        if not (hasattr(_torch, "npu") and _torch.npu.is_available()):
            return None
        props = _torch.npu.get_device_properties(device_id)
        value = getattr(props, "vector_core_num", None)
        return int(value) if value else None
    except Exception:  # noqa: BLE001
        return None
