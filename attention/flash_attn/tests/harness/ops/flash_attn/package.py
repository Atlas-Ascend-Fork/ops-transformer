# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""flash_attn 的算子适配包（唯一需要按算子实现的地方）。"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from harness.golden_spec import GoldenSpec

from . import golden_truth

NAME = "flash_attn"
_DTYPE_SIZE = {"fp16": 2, "bf16": 2, "fp32": 4, "int8": 1, "fp8": 1}


def _one(params: Dict[str, Any], key: str, default=None):
    value = params.get(key, default)
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


class FlashAttnPackage:
    markers = ("ci", "func_rdv", "perf_rdv", "debug", "graph", "legacy", "npu")

    name = NAME

    # ---------------------------------------------------------------- legalize
    def legalize(self, params: Dict[str, Any]) -> Optional[str]:
        def num(key, default=None):
            value = _one(params, key, None)
            if value is None:
                return default
            return int(value)

        def dim(key, fallback, default=1):
            value = _one(params, key, None)
            if value is None:
                value = _one(params, fallback, None)
            if value is None:
                return default
            return int(value)

        b, n1, n2 = num("B", 1), num("N1", 1), num("N2", 1)
        # TND 的 S1/S2 常为空，形状由 cu_seqlens/max_seqlen 决定
        s1, s2, d = dim("S1", "max_seqlen_q"), dim("S2", "max_seqlen_kv"), num("D", 128)
        dv = num("DV", d)
        layout_q = str(_one(params, "layout_q", "BNSD"))
        layout_kv = str(_one(params, "layout_kv", layout_q))
        layout_out = str(_one(params, "layout_out", layout_q))
        mask = num("mask_mode", 0)
        scale = _one(params, "scale", None)
        if scale is not None:
            try:
                if float(scale) <= 0:
                    return f"scale must be > 0, got {scale}"
            except (TypeError, ValueError):
                return f"invalid scale={scale!r}"
        if min(b, n1, n2, s1, s2, d) <= 0:
            return "B/N1/N2/S1/S2/D must be positive"
        if n1 % n2:
            return "N1 must be a multiple of N2"
        if d not in (64, 72, 128, 192, 256):
            return f"unsupported D={d}"
        if layout_q not in ("BNSD", "BSND", "TND"):
            return f"unsupported layout_q={layout_q}"
        if layout_out not in ("BNSD", "BSND", "TND"):
            return f"unsupported layout_out={layout_out}"
        pa_ok = ("BNSD", "BSND", "PA_BBND", "PA_BNBD", "PA_NZ")
        if layout_kv == "TND":
            if layout_q != "TND":
                return "layout_kv=TND requires layout_q=TND"
        elif layout_kv.startswith("PA_"):
            if layout_kv not in pa_ok:
                return f"unsupported layout_kv={layout_kv}"
            if layout_kv == "PA_NZ" and d % 16:
                return "PA_NZ requires D % 16 == 0"
        elif layout_kv not in ("BNSD", "BSND"):
            return f"unsupported layout_kv={layout_kv}"
        if mask not in (0, 3, 4):
            return f"unsupported mask_mode={mask}"
        if dv and dv > 256:
            return f"unsupported DV={dv}"
        return None

    # ------------------------------------------------------------ memory model
    def tensor_footprint(self, params: Dict[str, Any]) -> Dict[str, int]:
        from . import memory

        return memory.tensor_footprint(params)

    def golden_footprint(
        self,
        params: Dict[str, Any],
        *,
        spec: GoldenSpec,
        chunk_bytes: Optional[int] = None,
        keep_on_device: bool = True,
    ) -> Dict[str, int]:
        from . import golden_truth, memory

        return golden_truth.golden_peak(
            memory.tensor_footprint(params),
            params,
            device=spec.device,
            chunk_bytes=chunk_bytes or golden_truth.golden_chunk_bytes(spec.device),
            p_dtype=spec.p_dtype,
            p_block=spec.p_block,
            softmax_mode=spec.softmax_mode,
            softmax_block_size=spec.softmax_block_size,
            kv_loader=spec.kv_loader,
            keep_on_device=keep_on_device,
        )

    def estimate_memory(self, params: Dict[str, Any], **kwargs):
        """兼容入口：按阶段组合张量占用与 golden 峰值。"""
        from . import memory

        return memory.estimate(params, **kwargs)

    def build_inputs(self, params: Dict[str, Any], device: torch.device, seed: int):
        from harness.ops.flash_attn.inputs import flash_attn_inputs
        from .params import materialize

        params = materialize(params)
        layout = str(_one(params, "layout_q", "BNSD"))
        layout_kv = str(_one(params, "layout_kv", layout))
        inputs = flash_attn_inputs.generate(
            params, device=device, layout=layout, layout_kv=layout_kv, seed=seed
        )
        if _one(params, "sinks", False):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed) % (2**31))
            inputs["sinks"] = (
                torch.randn(
                    int(_one(params, "N1", 1)), dtype=torch.float32, generator=gen
                )
                * 0.5
            )
        if str(layout_kv).startswith("PA_"):
            table = params.get("block_table")
            if table is not None and not isinstance(table, torch.Tensor):
                table = torch.tensor(table, dtype=torch.int32)
            inputs["block_table"] = table
        return inputs

    # ------------------------------------------------------------------ golden
    def golden(
        self, inputs, params, spec: GoldenSpec, keep_on_device: bool = False
    ) -> Dict[str, Any]:
        from .params import materialize

        params = materialize(params)
        if spec.provider == "peer":
            return golden_truth.compute(
                inputs,
                params,
                compute_dtype="fp32",
                device={"npu": "npu", "gpu": "gpu"}.get(spec.device, "cpu"),
                p_dtype="out_dtype",
                p_block=spec.p_block or 0,
                softmax_mode=spec.softmax_mode,
                softmax_block_size=spec.softmax_block_size,
                kv_loader=spec.kv_loader,
            )
        if spec.device == "gpu":
            return golden_truth.compute(
                inputs,
                params,
                compute_dtype=spec.compute_dtype,
                device="gpu",
                p_dtype=spec.p_dtype,
                p_block=spec.p_block or 0,
                softmax_mode=spec.softmax_mode,
                softmax_block_size=spec.softmax_block_size,
                kv_loader=spec.kv_loader,
            )
        if spec.device == "npu":
            from harness.ops.flash_attn.golden_npu import NPUGoldenBackend

            backend = NPUGoldenBackend(
                0,
                p_dtype=spec.p_dtype,
                p_block=spec.p_block or 0,
                softmax_mode=spec.softmax_mode,
                softmax_block_size=spec.softmax_block_size,
                kv_loader=spec.kv_loader,
            )
            return backend.compute(inputs, params, keep_on_device=keep_on_device)
        return golden_truth.compute(
            inputs,
            params,
            compute_dtype=spec.compute_dtype,
            p_dtype=spec.p_dtype,
            p_block=spec.p_block or 0,
            softmax_mode=spec.softmax_mode,
            softmax_block_size=spec.softmax_block_size,
            kv_loader=spec.kv_loader,
        )

    # ---------------------------------------------------------------- execute
    def execute(
        self,
        inputs,
        params,
        mode: str = "eager",
        device_id: int = 0,
        keep_on_device: bool = False,
    ):
        from harness.ops.flash_attn.executor import get_backend
        from .params import materialize

        params = materialize(params)

        backend = get_backend(device_id, graph_mode=(mode == "graph"))
        result = backend.compute(inputs, params)
        if keep_on_device:
            return result
        return {
            key: (value.detach().cpu() if isinstance(value, torch.Tensor) else value)
            for key, value in result.items()
        }

    # ------------------------------------------------------------ runner hooks
    def normalize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        from .params import normalize

        return normalize(params)

    def case_name(self, params: Dict[str, Any]) -> str:
        def one(key, default=None):
            value = params.get(key, default)
            return value[0] if isinstance(value, (list, tuple)) and value else value

        s1 = int(one("S1", 1) or 1)
        phase = "prefill" if s1 >= 17 else "decode"
        parts = [
            phase,
            str(one("layout_q", "BNSD")),
            str(one("dtype", one("Dtype", "fp16"))),
            f"d{int(one('D', 0))}",
        ]
        dv = int(one("DV", one("D", 0)) or 0)
        if dv and dv != int(one("D", 0) or 0):
            parts.append(f"dv{dv}")
        parts.extend(
            [
                f"qs{s1}",
                f"kv{int(one('S2', s1) or s1)}",
                f"m{int(one('mask_mode', 0) or 0)}",
            ]
        )
        layout_kv = str(one("layout_kv", ""))
        if layout_kv.startswith("PA_"):
            parts.append(f"pa{int(one('block_size', 0) or 0)}")
        import re

        return re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(parts)).strip("_") or "case"

    def category(self, case) -> str:
        return "perf" if "perf_rdline" in case.tags else "func"

    def metric_columns(self):
        return ("attn", "lse")

    def peer_like_truth(self, spec: GoldenSpec) -> bool:
        return spec.provider == "peer" or spec.p_dtype not in ("fp32", None)

    def tiling_key(self, kernel_name: str):
        import re

        if (
            not kernel_name
            or "FlashAttn" not in kernel_name
            or "Metadata" in kernel_name
        ):
            return None
        match = re.search(r"FlashAttn_[0-9a-fA-F]+_(\d+)", kernel_name)
        return match.group(1) if match else None

    def perf_context(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dtype": _one(params, "dtype"),
            "layout_q": _one(params, "layout_q"),
            "layout_kv": _one(params, "layout_kv"),
            "B": _one(params, "B"),
            "N1": _one(params, "N1"),
            "N2": _one(params, "N2"),
            "S1": _one(params, "S1"),
            "S2": _one(params, "S2"),
            "D": _one(params, "D"),
            "DV": _one(params, "DV", _one(params, "D")),
            "mask": _one(params, "mask_mode"),
        }

    def compare(
        self,
        golden,
        actual,
        params: Dict[str, Any],
        spec: GoldenSpec,
        compare_mode: str,
        inputs=None,
        peer=None,
    ):
        from harness import compare as cmp
        from .layout import trans_bnsd_to_layout

        expect = golden["out"]
        actual_out = actual["out"]
        layout_out = params.get("layout_out", params.get("layout_q", "BNSD"))
        if expect.shape != actual_out.shape:
            expect = trans_bnsd_to_layout(expect, layout_out, **params)

        peer_like = spec.provider == "peer" or spec.p_dtype not in ("fp32", None)
        if compare_mode == "three_way" and not peer_like:
            peer_spec = GoldenSpec(
                device=spec.device,
                compute_dtype=spec.compute_dtype,
                p_dtype="out_dtype",
                p_block=spec.p_block,
                softmax_mode=spec.softmax_mode,
                softmax_block_size=spec.softmax_block_size,
                kv_loader=spec.kv_loader,
                provider="peer",
            )
            if inputs is None or peer is None:
                raise RuntimeError(
                    "three_way compare requires inputs and peer provider"
                )
            peer_out = peer(inputs, params, peer_spec)["out"]
            if peer_out.shape != actual_out.shape:
                peer_out = trans_bnsd_to_layout(peer_out, layout_out, **params)
            attn = cmp.compare_three_way(truth=expect, peer=peer_out, actual=actual_out)
            attn["mode"] = "three_way"
        else:
            attn = cmp.compare_two_way(expect, actual_out, params.get("dtype"))
            attn["mode"] = (
                "two_way(downgraded)"
                if compare_mode == "three_way" and peer_like
                else "two_way"
            )

        metrics = {"attn": attn}
        passed = True if attn.get("skipped") else bool(attn.get("passed", False))
        expect_lse, got_lse = golden.get("lse"), actual.get("lse")
        if expect_lse is not None and got_lse is not None:
            placeholder = (
                min(expect_lse.numel(), got_lse.numel()) <= 1
                and expect_lse.shape != got_lse.shape
            )
            if not placeholder:
                metrics["lse"] = cmp.compare_two_way(expect_lse, got_lse, dtype="fp32")
                passed = passed and bool(metrics["lse"].get("passed", False))
        return metrics, passed

    def result_columns(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """扁平结果列（results.csv 用）：形状、派生 token 数、工作量等。"""
        from . import memory as memlib

        def as_int(value, default=None):
            return default if value is None else int(value)

        def raw_list(key):
            value = params.get(key)
            if isinstance(value, (list, tuple)):
                if len(value) == 1 and isinstance(value[0], (list, tuple)):
                    value = value[0]
                return [int(v) for v in value]
            return None if value is None else [int(value)]

        def max_len(which: str) -> int:
            explicit = params.get(f"max_seqlen_{which}")
            if explicit is not None:
                return int(explicit)
            used = raw_list(f"seqused_{which}")
            if used:
                return max(used)
            cu = raw_list("cu_seqlens_q" if which == "q" else "cu_seqlens_kv")
            if cu:
                return max(cu[i + 1] - cu[i] for i in range(len(cu) - 1))
            return as_int(_one(params, "S1" if which == "q" else "S2", None), 1) or 1

        d = as_int(_one(params, "D", 128), 128)
        dv = as_int(_one(params, "DV", None), d) or d
        b = as_int(_one(params, "B", 1), 1)
        n1 = as_int(_one(params, "N1", 1), 1)
        n2 = as_int(_one(params, "N2", None), n1)
        lq = str(_one(params, "layout_q", "BNSD"))
        lkv = str(_one(params, "layout_kv", lq))
        lo = str(_one(params, "layout_out", lq))
        s1 = as_int(_one(params, "S1", None), None)
        s2 = as_int(_one(params, "S2", None), None)
        max_q, max_kv = max_len("q"), max_len("kv")
        q_pos, kv_pos = memlib.positions(params)
        workload = self.theoretical_workload(params) or {}
        return {
            "phase": "decode" if (s1 if s1 is not None else max_q) <= 16 else "prefill",
            "dtype": _one(params, "dtype", None),
            "layout_q": lq,
            "layout_kv": lkv,
            "layout_out": lo,
            "B": b,
            "N1": n1,
            "N2": n2,
            "S1": s1,
            "S2": s2,
            "D": d,
            "DV": dv,
            "mask_mode": as_int(_one(params, "mask_mode", 0), 0),
            "win_left": as_int(_one(params, "win_left", None), None),
            "win_right": as_int(_one(params, "win_right", None), None),
            "scale": _one(params, "scale", None),
            "return_softmax_lse": bool(_one(params, "return_softmax_lse", False)),
            "sinks": _one(params, "sinks", None) is not None,
            "block_size": as_int(_one(params, "block_size", None), None),
            "max_seqlen_q": max_q,
            "max_seqlen_kv": max_kv,
            "seqused_q": raw_list("seqused_q"),
            "seqused_kv": raw_list("seqused_kv"),
            "cu_seqlens_q": raw_list("cu_seqlens_q"),
            "cu_seqlens_kv": raw_list("cu_seqlens_kv"),
            "q_pos": q_pos,
            "kv_pos": kv_pos,
            "flops": workload.get("flops"),
            "bytes": workload.get("bytes"),
        }

    # ------------------------------------------------------------------ extras
    def _shape(self, params: Dict[str, Any]) -> Dict[str, Any]:
        def as_int(value, default):
            return default if value is None else int(value)

        d = as_int(_one(params, "D", 128), 128)
        dv = as_int(_one(params, "DV", None), d) or d
        b = as_int(_one(params, "B", 1), 1)
        n1 = as_int(_one(params, "N1", 1), 1)
        n2 = as_int(_one(params, "N2", None), n1)
        s1 = as_int(
            _one(params, "S1", None), as_int(_one(params, "max_seqlen_q", 1), 1)
        )
        s2 = as_int(
            _one(params, "S2", None), as_int(_one(params, "max_seqlen_kv", s1), s1)
        )
        lq = str(_one(params, "layout_q", "BNSD"))
        lkv = str(_one(params, "layout_kv", lq))

        def lens(which: str):
            raw = params.get(f"seqused_{which}")
            if isinstance(raw, (list, tuple)):
                if len(raw) == 1 and isinstance(raw[0], (list, tuple)):
                    raw = raw[0]
                used = [int(v) for v in raw]
            elif raw is not None:
                used = [int(raw)] * b
            else:
                used = None
            base = s1 if which == "q" else s2
            layout = lq if which == "q" else lkv
            if layout == "TND":
                cu = params.get("cu_seqlens_q" if which == "q" else "cu_seqlens_kv")
                if cu:
                    seg = [int(cu[i + 1]) - int(cu[i]) for i in range(len(cu) - 1)]
                    if used:
                        return [min(int(u), int(v)) for u, v in zip(used, seg)]
                    return seg
            if used:
                return [int(u) for u in used]
            return [base] * b

        return {
            "b": b,
            "n1": n1,
            "n2": n2,
            "d": d,
            "dv": dv,
            "q_lens": lens("q"),
            "kv_lens": lens("kv"),
        }

    def visible_pairs(self, params: Dict[str, Any]) -> int:
        """有效 QK 对（按 mask/窗口/seqused/TND 分段），用于 FLOPs 与 MFU。"""
        shape = self._shape(params)
        mask = int(_one(params, "mask_mode", 0) or 0)
        win_left = _one(params, "win_left", -1)
        win_right = _one(params, "win_right", -1)
        win_left = -1 if win_left is None else int(win_left)
        win_right = -1 if win_right is None else int(win_right)
        total = 0
        for q_len, kv_len in zip(shape["q_lens"], shape["kv_lens"]):
            if q_len <= 0 or kv_len <= 0:
                continue
            if mask == 3:
                delta = kv_len - q_len
                start = max(0, -delta)
                rows = q_len - start
                if rows > 0:
                    first = start + delta + 1
                    total += rows * (first + kv_len) // 2
            elif mask == 4:
                import torch

                idx = torch.arange(q_len, dtype=torch.int64)
                q_pos = idx + (kv_len - q_len)
                left = win_left if win_left >= 0 else 10**18
                right = win_right if win_right >= 0 else 10**18
                lo = torch.clamp(q_pos - left, min=0)
                hi = torch.clamp(q_pos + right, max=kv_len - 1)
                total += int(torch.clamp(hi - lo + 1, min=0).sum().item())
            else:
                total += q_len * kv_len
        return int(total)

    def theoretical_workload(
        self, params: Dict[str, Any]
    ) -> Optional[Dict[str, float]]:
        from . import memory as memlib

        shape = self._shape(params)
        pairs = self.visible_pairs(params)
        flops = 2.0 * shape["n1"] * pairs * (shape["d"] + shape["dv"])
        q_pos, kv_pos = memlib.positions(params)
        size = _DTYPE_SIZE.get(str(_one(params, "dtype", "fp16")), 2)
        nbytes = (
            q_pos * shape["n1"] * shape["d"]
            + 2 * kv_pos * shape["n2"] * shape["d"]
            + q_pos * shape["n1"] * shape["dv"]
        ) * size
        return {"flops": flops, "bytes": float(nbytes)}
