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
"""Generate random, legal flash_attn cases for fuzzing."""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from harness import guards  # noqa: E402
from harness.case import Case, dump_cases_file  # noqa: E402
from harness.golden_spec import parse_spec  # noqa: E402
from harness.ops.flash_attn import PACKAGE  # noqa: E402


@dataclass(frozen=True)
class Choice:
    """枚举：每次从 values 中均匀取一个。"""

    values: tuple


class IntervalMode(str, Enum):
    CLOSED = "closed"
    RIGHT_OPEN = "right_open"
    LEFT_OPEN = "left_open"
    OPEN = "open"


@dataclass(frozen=True)
class IntRange:
    """整数范围。"""

    low: int
    high: int
    step: int = 1
    mode: IntervalMode = IntervalMode.CLOSED


@dataclass(frozen=True)
class FloatRange:
    """浮点范围。"""

    low: float
    high: float
    mode: IntervalMode = IntervalMode.CLOSED


@dataclass(frozen=True)
class ScaleStrategy:
    """scale 策略：概率 + 取值（None 表示走算子默认）。"""

    name: str
    probability: float
    value: object


@dataclass(frozen=True)
class WindowStrategy:
    """window 策略：概率 + 类型。"""

    name: str
    probability: float


@dataclass(frozen=True)
class BoolSpec:
    """Bool：按 p_true 取 True/False。"""

    p_true: float = 0.5


DTYPES = Choice(("fp16", "bf16"))
SUPPORTED_DV = Choice(((64, 64), (72, 72), (128, 128), (192, 128), (256, 256)))
MASK_MODES = Choice((0, 3, 4))
BATCH_RANGE = IntRange(1, 512)
N2_RANGE = IntRange(1, 16)
G_RANGE = IntRange(1, 32)
BLOCK_SIZE_RANGE = IntRange(16, 1024, 16)

# layout
Q_OUT_LAYOUTS = Choice(
    (
        ("BNSD", "BNSD"),
        ("BNSD", "BSND"),
        ("BSND", "BSND"),
        ("TND", "TND"),
    )
)
PA_LAYOUTS = Choice(("PA_BBND", "PA_BNBD", "PA_NZ"))
SAME_AS_Q = "same"
KV_LAYOUTS = Choice((SAME_AS_Q,) + PA_LAYOUTS.values)

# 特性
RETURN_SOFTMAX_LSE = BoolSpec()
SINKS = BoolSpec()
EMIT_MAX_SEQLEN_Q = BoolSpec()
EMIT_MAX_SEQLEN_KV = BoolSpec()
EMIT_HEAD_DIM_V = BoolSpec()

# seqlen 策略
SEQ_SCENES = Choice(
    (
        "S1_varlen",  # TND Q/KV，纯 cu_seqlens，无 seqused
        "S2_trunc",  # Q=BNSD/BSND（非 TND），seqused_q/kv 截断
        "S3_varlen_trunc",  # TND 变长 + seqused 截断
        "S4_exact",  # TND，seqused 等于各段物理长度
        "S5_empty",  # 含 0 长度段或 seqused=0
        "S6_pa_decode",  # Q=BNSD/BSND，PA KV decode，S1=1..16
        "S7_tnd_pa_varlen",  # TND Q 变长 + PA KV
    )
)
Q_SEQ_RANGE = IntRange(0, 1024 * 1024)
KV_SEQ_RANGE = IntRange(0, 1024 * 1024)
DECODE_S1_RANGE = IntRange(1, 16)
MAX_TOTAL_TOKENS = 1024 * 1024
LARGE_SEQ_THRESHOLD = 128 * 1024
LARGE_CASE_RATE = 0.01
SEQ_BOUNDARY_RATE = 0.2
NORMAL_LOG_MIN = 384
PAD_MAX = 256
SEQ_BOUNDARIES = (
    1,
    2,
    7,
    8,
    9,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    135,
    136,
    137,
    143,
    144,
    145,
    159,
    160,
    161,
    191,
    192,
    193,
    223,
    224,
    225,
    255,
    256,
    257,
    383,
    384,
    385,
    447,
    448,
    449,
    479,
    480,
    481,
    495,
    496,
    497,
    503,
    504,
    505,
    511,
    512,
    513,
    767,
    768,
    769,
    1023,
    1024,
    1025,
    2047,
    2048,
    2049,
    4095,
    4096,
    4097,
    8191,
    8192,
    8193,
    16383,
    16384,
    16385,
    32767,
    32768,
    32769,
    65535,
    65536,
    65537,
    131071,
)

# window 策略（仅 mask=4）
WINDOW_CANDIDATES = Choice((-1, 0, 1, 16, 64, 128, 512, 1024))
WINDOW_STRATEGIES = (
    WindowStrategy("both_inf", 0.02),
    WindowStrategy("left_inf", 0.04),
    WindowStrategy("right_inf", 0.04),
    WindowStrategy("boundary", 0.10),
    WindowStrategy("uniform_range", 0.60),
    WindowStrategy("overflow_left", 0.09),
    WindowStrategy("overflow_right", 0.09),
    WindowStrategy("overflow_both", 0.02),
)

# scale 策略（概率之和为 1）
FP32_MAX = 3.4028234663852886e38
FP32_TINY = 1.1754943508222875e-38
SCALE_STRATEGIES = (
    ScaleStrategy("default", 0.1, None),
    ScaleStrategy("inv_sqrt_d", 0.4, "inv_sqrt_d"),
    ScaleStrategy("range_0_1", 0.4, FloatRange(0.0, 1.0, IntervalMode.LEFT_OPEN)),
    ScaleStrategy("range_1_fp32max", 0.1, FloatRange(1.0, FP32_MAX, IntervalMode.OPEN)),
)

DEFAULT_ATTEMPTS_PER_CASE = 100
GEN_SAFETY_FACTOR = 1.2
GUARD_SPEC = parse_spec("device=npu,compute_dtype=fp32,p_dtype=out_dtype")
GUARD_STAGES = ("gen", "golden", "run", "compare")
GUARD_HBM_GB = guards.resolve_hbm_guard(0)
GEN_CASES = 1000


def _one(params, key, default=None):
    value = params.get(key, default)
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _rand_range(rng, spec, high=None):
    low, top = spec.low, spec.high
    if high is not None:
        top = min(top, high)
    top = max(low, top)
    if spec.mode is IntervalMode.CLOSED:
        start, stop = low, top + 1
    elif spec.mode is IntervalMode.RIGHT_OPEN:
        start, stop = low, top
    elif spec.mode is IntervalMode.LEFT_OPEN:
        start, stop = low + spec.step, top + 1
    elif spec.mode is IntervalMode.OPEN:
        start, stop = low + spec.step, top
    else:
        raise ValueError(f"unsupported interval mode: {spec.mode!r}")
    if stop <= start:
        raise ValueError(f"empty range: {spec!r} capped at {high!r}")
    return rng.randrange(start, stop, spec.step)


def _rand_float(rng, spec):
    low, top = spec.low, spec.high
    if spec.mode in (IntervalMode.LEFT_OPEN, IntervalMode.OPEN):
        low = math.nextafter(low, math.inf)
    if spec.mode in (IntervalMode.RIGHT_OPEN, IntervalMode.OPEN):
        top = math.nextafter(top, -math.inf)
    if spec.mode not in (
        IntervalMode.CLOSED,
        IntervalMode.RIGHT_OPEN,
        IntervalMode.LEFT_OPEN,
        IntervalMode.OPEN,
    ):
        raise ValueError(f"unsupported interval mode: {spec.mode!r}")
    if top < low:
        raise ValueError(f"empty float range: {spec!r}")
    return rng.uniform(low, top)


def _sample(rng, spec):
    if isinstance(spec, Choice):
        return rng.choice(spec.values)
    if isinstance(spec, IntRange):
        return _rand_range(rng, spec)
    if isinstance(spec, FloatRange):
        return _rand_float(rng, spec)
    if isinstance(spec, BoolSpec):
        return rng.random() < spec.p_true
    raise TypeError(f"unsupported config type: {type(spec)!r}")


def _sample_scale(rng, d):
    strategy = rng.choices(
        SCALE_STRATEGIES, weights=[item.probability for item in SCALE_STRATEGIES]
    )[0]
    if strategy.value is None:
        return None
    if strategy.value == "inv_sqrt_d":
        return 1.0 / math.sqrt(d)
    if isinstance(strategy.value, FloatRange):
        return max(_sample(rng, strategy.value), FP32_TINY)
    return float(strategy.value)


def _pick_dv(rng):
    return _sample(rng, SUPPORTED_DV)


def _dense_q_out_layouts():
    return tuple(pair for pair in Q_OUT_LAYOUTS.values if pair[0] != "TND")


def _dense_layouts():
    return tuple(pair[0] for pair in _dense_q_out_layouts())


def _pick_dv_for(d):
    if (d, d) in SUPPORTED_DV.values:
        return d, d
    for combo in SUPPORTED_DV.values:
        if combo[0] == d:
            return combo
    return SUPPORTED_DV.values[0]


def _rand_lengths(rng, count: int, spec, max_len: int):
    return [_rand_range(rng, spec, max_len) for _ in range(count)]


def _seq_cap(max_len, batch):
    return max(1, min(max_len, MAX_TOTAL_TOKENS // max(1, batch)))


def _sample_boundary(rng, top):
    values = [value for value in SEQ_BOUNDARIES if value <= top]
    return rng.choice(values) if values else top


def _sample_seq_len(rng, max_len, large=False):
    top = max(1, int(max_len))
    if large and top >= LARGE_SEQ_THRESHOLD:
        return int(2 ** rng.uniform(math.log2(LARGE_SEQ_THRESHOLD), math.log2(top)))
    top = min(top, LARGE_SEQ_THRESHOLD - 1 - PAD_MAX)
    if rng.random() < SEQ_BOUNDARY_RATE:
        return _sample_boundary(rng, top)
    low = min(NORMAL_LOG_MIN, top)
    return max(1, int(2 ** rng.uniform(math.log2(low), math.log2(top))))


def _fit_large(lengths, index, value, limit, low):
    if value < LARGE_SEQ_THRESHOLD:
        return False
    lengths[index] = value
    if sum(lengths) <= limit:
        return True
    for i in range(len(lengths)):
        if i == index:
            continue
        excess = sum(lengths) - limit
        if excess <= 0:
            break
        take = min(lengths[i] - low, excess)
        lengths[i] -= take
    excess = sum(lengths) - limit
    if excess > 0:
        value = value - excess
        if value < LARGE_SEQ_THRESHOLD:
            return False
        lengths[index] = value
    return True


def _sample_lengths(rng, count, spec, max_len, allow_zero=False, total_limit=None):
    low = 0 if allow_zero else max(1, spec.low)
    top = max(low, min(spec.high, max_len, LARGE_SEQ_THRESHOLD - 1 - PAD_MAX))
    if allow_zero:
        lengths = [
            0 if rng.random() < 0.5 else _sample_seq_len(rng, top) for _ in range(count)
        ]
        if not any(lengths):
            lengths[rng.randrange(count)] = _sample_seq_len(rng, top)
    else:
        lengths = [_sample_seq_len(rng, top) for _ in range(count)]
    if total_limit is not None and sum(lengths) > total_limit:
        scale = total_limit / max(1, sum(lengths))
        lengths = [max(low, int(value * scale)) for value in lengths]
        while sum(lengths) > total_limit:
            index = max(range(count), key=lambda i: lengths[i])
            if lengths[index] <= low:
                break
            lengths[index] -= 1
    return lengths


def _physical_s(rng, lengths, cap):
    pad = rng.randint(0, PAD_MAX)
    return max(1, min(cap, max(lengths, default=0) + pad))


def _apply_large_dense(rng, lengths, max_len, enabled, axis, target):
    if not enabled or axis not in (target, "both") or not lengths:
        return
    if max_len < LARGE_SEQ_THRESHOLD:
        return
    lengths[rng.randrange(len(lengths))] = _sample_seq_len(rng, max_len, large=True)


def _kv_choices(layout_q, d, allowed=None):
    values = allowed if allowed is not None else KV_LAYOUTS.values
    result = []
    for value in values:
        layout = layout_q if value == SAME_AS_Q else value
        if layout == "PA_NZ" and d % 16:
            continue
        result.append(layout)
    return result


def _apply_large_lengths(rng, lengths, max_len, enabled, axis, target, low):
    if not enabled or axis not in (target, "both") or not lengths:
        return
    max_focus = MAX_TOTAL_TOKENS - low * (len(lengths) - 1)
    if max_focus < LARGE_SEQ_THRESHOLD:
        return
    before = list(lengths)
    index = rng.randrange(len(lengths))
    value = _sample_seq_len(rng, min(max_len, max_focus), large=True)
    if not _fit_large(lengths, index, value, MAX_TOTAL_TOKENS, low):
        lengths[:] = before


def _large_value(rng, value, max_len, enabled, axis, target):
    if enabled and axis in (target, "both"):
        return _sample_seq_len(rng, max_len, large=True)
    return value


def _base_params(rng, q_out, layout_kv, batch, d, dv):
    layout_q, layout_out = q_out
    n2 = _rand_range(rng, N2_RANGE)
    n1 = n2 * _rand_range(rng, G_RANGE)
    mask = _sample(rng, MASK_MODES)
    params = {
        "dtype": _sample(rng, DTYPES),
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
        "B": batch,
        "N1": n1,
        "N2": n2,
        "D": d,
        "DV": dv,
        "mask_mode": mask,
        "return_softmax_lse": _sample(rng, RETURN_SOFTMAX_LSE),
        "sinks": _sample(rng, SINKS),
        "scale": None,
    }
    if d != dv or _sample(rng, EMIT_HEAD_DIM_V):
        params["head_dim_v"] = dv
    if layout_kv in PA_LAYOUTS.values:
        params["block_size"] = _rand_range(rng, BLOCK_SIZE_RANGE)
    return params


def _ensure_truncated(rng, used, physical):
    if any(value < limit for value, limit in zip(used, physical)):
        return used
    candidates = [i for i, limit in enumerate(physical) if limit > 1]
    if candidates:
        index = rng.choice(candidates)
        used[index] = rng.randint(1, physical[index] - 1)
    return used


def _ensure_nonzero(rng, values, fallback):
    if values and not any(values):
        values[rng.randrange(len(values))] = fallback
    return values


def _weighted_choice(rng, specs):
    return rng.choices(specs, weights=[item.probability for item in specs])[0]


def _window_values(limit):
    fixed = [value for value in WINDOW_CANDIDATES.values if 0 <= value <= limit]
    boundary = [max(1, limit // 2), max(1, limit - 1), max(1, limit)]
    return fixed, boundary


def _sample_window(rng, s1, s2):
    strategy = _weighted_choice(rng, WINDOW_STRATEGIES)
    left_fixed, left_boundary = _window_values(s2)
    right_fixed, right_boundary = _window_values(s1)
    if strategy.name == "uniform_range":
        return rng.randint(-1, s2), rng.randint(-1, s1)
    if strategy.name == "overflow_left":
        return (
            rng.randint(s2 + 1, s2 + max(1, s2)),
            rng.randint(-1, s1),
        )
    if strategy.name == "overflow_right":
        return (
            rng.randint(-1, s2),
            rng.randint(s1 + 1, s1 + max(1, s1)),
        )
    if strategy.name == "overflow_both":
        return (
            rng.randint(s2 + 1, s2 + max(1, s2)),
            rng.randint(s1 + 1, s1 + max(1, s1)),
        )
    if strategy.name == "both_inf":
        return -1, -1
    if strategy.name == "left_inf":
        return -1, rng.choice(right_fixed)
    if strategy.name == "right_inf":
        return rng.choice(left_fixed), -1
    if strategy.name == "boundary":
        return rng.choice(left_fixed + left_boundary), rng.choice(
            right_fixed + right_boundary
        )
    raise ValueError(f"unsupported window strategy: {strategy.name!r}")


def _apply_window(params, rng):
    if params["mask_mode"] != 4:
        return
    params["win_left"], params["win_right"] = _sample_window(
        rng, params["S1"], params["S2"]
    )


def _actual_max(params, which):
    used = params.get(f"seqused_{which}")
    if isinstance(used, list) and used:
        return max(used)
    cu = params.get(f"cu_seqlens_{which}")
    if isinstance(cu, list) and len(cu) > 1:
        return max(cu[i + 1] - cu[i] for i in range(len(cu) - 1))
    return int(params.get("S1" if which == "q" else "S2", 1))


def _apply_max_seqlen(params, rng):
    if _sample(rng, EMIT_MAX_SEQLEN_Q):
        params["max_seqlen_q"] = _actual_max(params, "q")
    if _sample(rng, EMIT_MAX_SEQLEN_KV):
        params["max_seqlen_kv"] = _actual_max(params, "kv")


def _scene_params(rng, scene, max_seq, max_batch):
    segments = _rand_range(rng, BATCH_RANGE, max_batch)
    d, dv = _pick_dv(rng)
    large = rng.random() < LARGE_CASE_RATE
    large_axis = rng.choice(("q", "kv", "both"))
    cap = _seq_cap(max_seq, segments)

    if scene in ("S1_varlen", "S3_varlen_trunc", "S4_exact", "S7_tnd_pa_varlen"):
        q_out = ("TND", "TND")
        layout_q = q_out[0]
        if scene == "S1_varlen":
            layout_kv = _kv_choices(layout_q, d, (SAME_AS_Q,))[0]
        elif scene == "S7_tnd_pa_varlen":
            layout_kv = rng.choice(_kv_choices(layout_q, d, PA_LAYOUTS.values))
        else:
            layout_kv = rng.choice(_kv_choices(layout_q, d))
        params = _base_params(rng, q_out, layout_kv, 1, d, dv)
        q = _sample_lengths(
            rng, segments, Q_SEQ_RANGE, max_seq, total_limit=MAX_TOTAL_TOKENS
        )
        kv = _sample_lengths(
            rng, segments, KV_SEQ_RANGE, max_seq, total_limit=MAX_TOTAL_TOKENS
        )
        _apply_large_lengths(rng, q, max_seq, large, large_axis, "q", 1)
        _apply_large_lengths(rng, kv, max_seq, large, large_axis, "kv", 1)
        if scene == "S3_varlen_trunc" and max_seq >= 2:
            if max(q) <= 1:
                q[0] = 2
            if layout_kv == "TND" and max(kv) <= 1:
                kv[0] = 2
        cu_q = [0]
        for value in q:
            cu_q.append(cu_q[-1] + value)
        params.update(
            {
                "S1": max(q),
                "S2": max(kv),
                "cu_seqlens_q": cu_q,
            }
        )
        if layout_kv == "TND":
            cu_kv = [0]
            for value in kv:
                cu_kv.append(cu_kv[-1] + value)
            params["cu_seqlens_kv"] = cu_kv
        else:
            params["seqused_kv"] = kv
        if scene == "S3_varlen_trunc":
            params["seqused_q"] = _ensure_truncated(
                rng, [rng.randint(1, value) for value in q], q
            )
            if layout_kv == "TND":
                params["seqused_kv"] = _ensure_truncated(
                    rng, [rng.randint(1, value) for value in kv], kv
                )
        elif scene == "S4_exact":
            params["seqused_q"] = list(q)
            params["seqused_kv"] = list(kv)
        elif scene == "S7_tnd_pa_varlen":
            params["seqused_kv"] = [rng.randint(1, value) for value in kv]
        _apply_window(params, rng)
        _apply_max_seqlen(params, rng)
        return params

    if scene == "S6_pa_decode":
        q_out = rng.choice(_dense_q_out_layouts())
        layout_q = q_out[0]
        layout_kv = rng.choice(_kv_choices(layout_q, d, PA_LAYOUTS.values))
        params = _base_params(rng, q_out, layout_kv, segments, d, dv)
        s1 = _rand_range(rng, DECODE_S1_RANGE, cap)
        kv = _sample_lengths(rng, segments, KV_SEQ_RANGE, cap)
        _apply_large_dense(rng, kv, cap, large, large_axis, "kv")
        params.update(
            {
                "S1": s1,
                "S2": _physical_s(rng, kv, cap),
                "seqused_kv": kv,
            }
        )
        _apply_window(params, rng)
        _apply_max_seqlen(params, rng)
        return params

    if scene == "S5_empty":
        q_out = rng.choice(Q_OUT_LAYOUTS.values)
        layout_q = q_out[0]
        layout_kv = rng.choice(_kv_choices(layout_q, d))
        batch = 1 if layout_q == "TND" else segments
        params = _base_params(rng, q_out, layout_kv, batch, d, dv)
        if layout_q == "TND":
            q = _sample_lengths(
                rng,
                segments,
                Q_SEQ_RANGE,
                max_seq,
                allow_zero=True,
                total_limit=MAX_TOTAL_TOKENS,
            )
            kv = _sample_lengths(
                rng,
                segments,
                KV_SEQ_RANGE,
                max_seq,
                allow_zero=True,
                total_limit=MAX_TOTAL_TOKENS,
            )
            _apply_large_lengths(rng, q, max_seq, large, large_axis, "q", 0)
            _apply_large_lengths(rng, kv, max_seq, large, large_axis, "kv", 0)
            cu_q = [0]
            for value in q:
                cu_q.append(cu_q[-1] + value)
            params.update(
                {
                    "S1": max(q),
                    "S2": max(kv),
                    "cu_seqlens_q": cu_q,
                }
            )
            if layout_kv == "TND":
                cu_kv = [0]
                for value in kv:
                    cu_kv.append(cu_kv[-1] + value)
                params["cu_seqlens_kv"] = cu_kv
            else:
                params["seqused_kv"] = _ensure_nonzero(
                    rng,
                    [0 if rng.random() < 0.5 else value for value in kv],
                    max(kv),
                )
        else:
            q = _sample_lengths(rng, segments, Q_SEQ_RANGE, cap, allow_zero=True)
            kv = _sample_lengths(rng, segments, KV_SEQ_RANGE, cap, allow_zero=True)
            _apply_large_dense(rng, q, cap, large, large_axis, "q")
            _apply_large_dense(rng, kv, cap, large, large_axis, "kv")
            params.update(
                {
                    "S1": _physical_s(rng, q, cap),
                    "S2": _physical_s(rng, kv, cap),
                    "seqused_q": q,
                    "seqused_kv": _ensure_nonzero(rng, kv, max(1, max(kv))),
                }
            )
        _apply_window(params, rng)
        _apply_max_seqlen(params, rng)
        return params

    # S2_trunc: dense Q，dense/PA KV，seqused 截断
    q_out = rng.choice(_dense_q_out_layouts())
    layout_q = q_out[0]
    layout_kv = rng.choice(_kv_choices(layout_q, d))
    params = _base_params(rng, q_out, layout_kv, segments, d, dv)
    q = _sample_lengths(rng, segments, Q_SEQ_RANGE, cap)
    kv = _sample_lengths(rng, segments, KV_SEQ_RANGE, cap)
    _apply_large_dense(rng, q, cap, large, large_axis, "q")
    _apply_large_dense(rng, kv, cap, large, large_axis, "kv")
    params.update(
        {
            "S1": _physical_s(rng, q, cap),
            "S2": _physical_s(rng, kv, cap),
            "seqused_q": q,
            "seqused_kv": kv,
        }
    )
    _apply_window(params, rng)
    _apply_max_seqlen(params, rng)
    return params


def _validate(case):
    reason = PACKAGE.legalize(case.params)
    if reason:
        return reason
    params = case.params
    d = int(_one(params, "D", SUPPORTED_DV.values[0][0]))
    dv = int(_one(params, "DV", d))
    if (d, dv) not in SUPPORTED_DV.values:
        return f"unsupported D/DV combo=({d}, {dv})"
    layout_q = str(_one(params, "layout_q", _dense_layouts()[0]))
    layout_kv = str(_one(params, "layout_kv", layout_q))
    layout_out = str(_one(params, "layout_out", layout_q))
    if (layout_q, layout_out) not in Q_OUT_LAYOUTS.values:
        return f"unsupported q_out_layout=({layout_q}, {layout_out})"
    if layout_kv != layout_q and layout_kv not in PA_LAYOUTS.values:
        return f"unsupported layout_kv={layout_kv}"
    if layout_kv == "PA_NZ" and d % 16:
        return "PA_NZ requires D % 16 == 0"
    if layout_q == "TND":
        cu_q = params.get("cu_seqlens_q")
        if not isinstance(cu_q, list) or len(cu_q) < 2 or cu_q[0] != 0:
            return "invalid cu_seqlens_q for TND"
        if any(cu_q[i + 1] < cu_q[i] for i in range(len(cu_q) - 1)) or cu_q[-1] <= 0:
            return "invalid cu_seqlens_q for TND"
    elif "cu_seqlens_q" in params:
        return "cu_seqlens_q requires layout_q=TND"
    if layout_kv == "TND":
        cu_kv = params.get("cu_seqlens_kv")
        if (
            not isinstance(cu_kv, list)
            or len(cu_kv) != len(params["cu_seqlens_q"])
            or cu_kv[0] != 0
        ):
            return "invalid cu_seqlens_kv for TND"
        if (
            any(cu_kv[i + 1] < cu_kv[i] for i in range(len(cu_kv) - 1))
            or cu_kv[-1] <= 0
        ):
            return "invalid cu_seqlens_kv for TND"
    elif "cu_seqlens_kv" in params:
        return "cu_seqlens_kv requires layout_kv=TND"
    b = int(_one(params, "B", 1))
    s1 = int(_one(params, "S1", 1))
    s2 = int(_one(params, "S2", s1))
    seqused_kv = params.get("seqused_kv")

    def diffs(cu):
        return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]

    q_lens = diffs(params["cu_seqlens_q"]) if layout_q == "TND" else [s1] * b
    if layout_kv == "TND":
        kv_lens = diffs(params["cu_seqlens_kv"])
    elif layout_q == "TND":
        kv_lens = [s2] * (len(params["cu_seqlens_q"]) - 1)
    else:
        kv_lens = [s2] * b

    for key, physical in (("seqused_q", q_lens), ("seqused_kv", kv_lens)):
        used = params.get(key)
        if used is None:
            continue
        if not isinstance(used, list) or len(used) != len(physical):
            return f"invalid {key} length"
        if any(value < 0 or value > limit for value, limit in zip(used, physical)):
            return f"invalid {key} value"

    q_tokens = params["cu_seqlens_q"][-1] if layout_q == "TND" else b * s1
    if layout_kv == "TND":
        kv_tokens = params["cu_seqlens_kv"][-1]
    elif layout_q == "TND" and isinstance(seqused_kv, list):
        kv_tokens = sum(seqused_kv)
    else:
        kv_tokens = b * s2
    if q_tokens > MAX_TOTAL_TOKENS or kv_tokens > MAX_TOTAL_TOKENS:
        return (
            f"total tokens exceed limit: q={q_tokens}, kv={kv_tokens}, "
            f"limit={MAX_TOTAL_TOKENS}"
        )
    return None


def _adjust(params, reason):
    fixed = dict(params)
    if reason.startswith("N1 must be a multiple of N2"):
        fixed["N1"] = int(_one(fixed, "N2", N2_RANGE.low))
    elif reason.startswith("unsupported D/DV combo="):
        d = int(_one(fixed, "D", SUPPORTED_DV.values[0][0]))
        fixed["D"], fixed["DV"] = _pick_dv_for(d)
        fixed.pop("head_dim_v", None)
    elif reason.startswith("unsupported D="):
        fixed["D"], fixed["DV"] = SUPPORTED_DV.values[0]
    elif reason.startswith("unsupported DV="):
        d = int(_one(fixed, "D", SUPPORTED_DV.values[0][0]))
        fixed["DV"] = _pick_dv_for(d)[1]
        fixed.pop("head_dim_v", None)
    elif reason.startswith("unsupported layout_q="):
        fallback = _dense_q_out_layouts()[0]
        fixed.update(
            layout_q=fallback[0], layout_kv=fallback[0], layout_out=fallback[1]
        )
        for key in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "seqused_q",
            "seqused_kv",
            "max_seqlen_q",
            "max_seqlen_kv",
        ):
            fixed.pop(key, None)
    elif reason.startswith("unsupported q_out_layout="):
        fallback = _dense_q_out_layouts()[0]
        fixed.update(
            layout_q=fallback[0], layout_kv=fallback[0], layout_out=fallback[1]
        )
        for key in (
            "cu_seqlens_q",
            "cu_seqlens_kv",
            "seqused_q",
            "seqused_kv",
            "max_seqlen_q",
            "max_seqlen_kv",
        ):
            fixed.pop(key, None)
    elif reason.startswith("layout_kv=TND requires layout_q=TND"):
        fixed["layout_kv"] = fixed.get("layout_q", _dense_layouts()[0])
        fixed.pop("cu_seqlens_kv", None)
        fixed.pop("seqused_kv", None)
    elif reason.startswith("unsupported layout_kv="):
        fixed["layout_kv"] = fixed.get("layout_q", _dense_layouts()[0])
        if fixed["layout_kv"] == "TND" and "cu_seqlens_kv" not in fixed:
            return None
    elif reason.startswith("unsupported layout_out="):
        fixed["layout_out"] = fixed.get("layout_q", _dense_layouts()[0])
    elif reason.startswith("PA_NZ requires D % 16 == 0"):
        fixed["D"], fixed["DV"] = next(
            (combo for combo in SUPPORTED_DV.values if combo[0] % 16 == 0),
            SUPPORTED_DV.values[0],
        )
        fixed.pop("head_dim_v", None)
    elif reason.startswith("unsupported mask_mode="):
        fixed["mask_mode"] = MASK_MODES.values[0]
        fixed.pop("win_left", None)
        fixed.pop("win_right", None)
    elif reason.startswith("scale must be > 0") or reason.startswith("invalid scale"):
        fixed.pop("scale", None)
    elif reason.endswith("must be positive"):
        for key in ("B", "N1", "N2", "S1", "S2", "D"):
            fixed[key] = max(1, int(fixed.get(key, 1) or 1))
        fixed["DV"] = _pick_dv_for(fixed["D"])[1]
        fixed.pop("head_dim_v", None)
    else:
        return None
    d = int(_one(fixed, "D", SUPPORTED_DV.values[0][0]))
    dv = int(_one(fixed, "DV", d))
    if d != dv:
        fixed["head_dim_v"] = dv
    else:
        fixed.pop("head_dim_v", None)
    return fixed


def generate_with_reason(
    rng,
    max_seq=max(Q_SEQ_RANGE.high, KV_SEQ_RANGE.high),
    max_batch=BATCH_RANGE.high,
    golden_spec=GUARD_SPEC,
    mem_guard_gb=guards.DEFAULT_MEM_GUARD_GB,
):
    max_seq = max(1, int(max_seq))
    max_batch = max(1, int(max_batch))
    params = _scene_params(rng, _sample(rng, SEQ_SCENES), max_seq, max_batch)
    params["seed"] = rng.randint(0, 2**31 - 1)
    params["scale"] = _sample_scale(rng, params["D"])
    for _ in range(2):
        try:
            case = Case(op="flash_attn", params=params, tags=["fuzz"])
        except Exception as exc:  # noqa: BLE001
            return None, f"normalize failed: {exc}"
        reason = _validate(case)
        if reason is None:
            reason = guards.memory_guard_reason(
                case,
                PACKAGE,
                spec=golden_spec,
                compare_mode="two_way",
                stages=GUARD_STAGES,
                mem_guard_gb=mem_guard_gb,
                hbm_guard_gb=GUARD_HBM_GB,
                safety_factor=GEN_SAFETY_FACTOR,
                persist=False,
            )
            if reason is None:
                return case, None
            return None, reason
        fixed = _adjust(case.params, reason)
        if fixed is None:
            return None, reason
        params = fixed
    return None, reason


def generate(
    rng,
    max_seq=max(Q_SEQ_RANGE.high, KV_SEQ_RANGE.high),
    max_batch=BATCH_RANGE.high,
    golden_spec=GUARD_SPEC,
    mem_guard_gb=guards.DEFAULT_MEM_GUARD_GB,
):
    return generate_with_reason(rng, max_seq, max_batch, golden_spec, mem_guard_gb)[0]


def generate_many(
    rng,
    num,
    max_seq=max(Q_SEQ_RANGE.high, KV_SEQ_RANGE.high),
    max_batch=BATCH_RANGE.high,
    golden_spec=GUARD_SPEC,
    mem_guard_gb=guards.DEFAULT_MEM_GUARD_GB,
):
    cases = []
    seen = set()
    rejects = Counter()
    max_attempts = max(num * DEFAULT_ATTEMPTS_PER_CASE, num + 100)
    attempts = 0
    while len(cases) < num and attempts < max_attempts:
        attempts += 1
        case, reason = generate_with_reason(
            rng, max_seq, max_batch, golden_spec, mem_guard_gb
        )
        if case is None:
            rejects[reason or "unknown"] += 1
            continue
        if case.id in seen:
            rejects["duplicate case_id"] += 1
            continue
        seen.add(case.id)
        cases.append(case)
    return cases, rejects, attempts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out", type=Path, default=Path("random_fuzz/cases/random.json")
    )
    parser.add_argument(
        "--golden",
        default="device=npu,compute_dtype=fp32,p_dtype=out_dtype",
    )
    parser.add_argument(
        "--mem-guard-gb", type=float, default=guards.DEFAULT_MEM_GUARD_GB
    )
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)
    cases, rejects, attempts = generate_many(
        rng,
        GEN_CASES,
        golden_spec=parse_spec(args.golden),
        mem_guard_gb=args.mem_guard_gb,
    )
    for reason, count in rejects.most_common():
        print(f"[reject] {count:4d} {reason}")
    if len(cases) < GEN_CASES:
        print(
            f"warning: generated only {len(cases)}/{GEN_CASES} valid cases "
            f"after {attempts} attempts"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump_cases_file(cases, args.out)
    print(f"generated {len(cases)} cases -> {args.out}")
    return 0 if cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
