# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""flash_attn 张量占用与执行阶段峰值组合。

张量字节数由本模块按算子布局语义计算；golden 中间峰值由 golden 实现提供
（`golden_truth.golden_peak`）。框架只组合阶段峰值并套用 guard。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

MB = 1024 * 1024
MASK_BYTES = 2048 * 2048  # harness 构造的 int8 占位 mask（mask_mode 3/4）
COMPARE_CHUNK = 4 * MB  # compare 每次处理的元素块（4M float32 临时量峰值 ~5 份）


def _one(params: Dict[str, Any], key: str, default=None):
    value = params.get(key, default)
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _as_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def dtype_bytes(dtype: str) -> int:
    return {
        "fp16": 2,
        "bf16": 2,
        "fp32": 4,
        "fp64": 8,
        "int8": 1,
        "fp8": 1,
    }.get(str(dtype), 2)


def positions(params: Dict[str, Any]) -> Tuple[int, int]:
    """(q_positions, kv_positions)：TND 用 cu 末尾，PA 用块数×block_size。"""
    b = _as_int(_one(params, "B", 1), 1)
    s1 = _as_int(_one(params, "S1", 1), 1)
    s2 = _as_int(_one(params, "S2", s1), s1)
    layout_q = str(_one(params, "layout_q", "BNSD"))
    layout_kv = str(_one(params, "layout_kv", layout_q))
    cu_q = params.get("cu_seqlens_q")
    cu_kv = params.get("cu_seqlens_kv")
    if layout_q == "TND" and cu_q:
        q_pos = _as_int(cu_q[-1], b * s1)
    else:
        q_pos = b * s1
    if layout_kv == "TND" and cu_kv:
        kv_pos = _as_int(cu_kv[-1], b * s2)
    elif layout_kv.startswith("PA_"):
        block = _as_int(_one(params, "block_size", 128), 128) or 128
        seqused = params.get("seqused_kv")
        if seqused:
            total = sum(_as_int(x) for x in seqused)
        else:
            total = b * s2
        kv_pos = ((total + block - 1) // block) * block
    else:
        kv_pos = b * s2
    return q_pos, kv_pos


def _sequence_values(params: Dict[str, Any], key: str, cu_key: str):
    used = params.get(key)
    if used is not None:
        if (
            isinstance(used, (list, tuple))
            and used
            and isinstance(used[0], (list, tuple))
        ):
            used = used[0]
        if isinstance(used, (list, tuple)):
            return [_as_int(value) for value in used]
    cu = params.get(cu_key)
    if cu is not None:
        if isinstance(cu, (list, tuple)) and cu and isinstance(cu[0], (list, tuple)):
            cu = cu[0]
        if isinstance(cu, (list, tuple)) and len(cu) > 1:
            return [_as_int(cu[i + 1]) - _as_int(cu[i]) for i in range(len(cu) - 1)]
    return []


def _max_sequence(params: Dict[str, Any], key: str, cu_key: str, fallback: int) -> int:
    values = _sequence_values(params, key, cu_key)
    return max(values) if values else fallback


def _max_cu(params: Dict[str, Any], key: str, fallback: int) -> int:
    cu = params.get(key)
    if isinstance(cu, (list, tuple)) and cu and isinstance(cu[0], (list, tuple)):
        cu = cu[0]
    if isinstance(cu, (list, tuple)) and len(cu) > 1:
        return max(_as_int(cu[i + 1]) - _as_int(cu[i]) for i in range(len(cu) - 1))
    return fallback


def _tensors(params: Dict[str, Any]) -> Dict[str, int]:
    b = _as_int(_one(params, "B", 1), 1)
    n1 = _as_int(_one(params, "N1", 1), 1)
    n2 = _as_int(_one(params, "N2", n1), n1)
    d = _as_int(_one(params, "D", 128), 128)
    dv = _as_int(_one(params, "DV", d) or d, d)
    size = dtype_bytes(_one(params, "dtype", _one(params, "Dtype", "fp16")))
    mask_mode = _as_int(_one(params, "mask_mode", 0))
    q_pos, kv_pos = positions(params)
    lse = bool(_one(params, "return_softmax_lse", False))
    out_bytes = q_pos * n1 * dv * size
    return {
        "q": q_pos * n1 * d * size,
        "kv": 2 * kv_pos * n2 * d * size,
        "aux": (b + 1) * 8 + _as_int(_one(params, "N1", 0)) * 4,
        "mask": MASK_BYTES if mask_mode in (3, 4) else 0,
        "out": out_bytes,
        "lse": q_pos * n1 * 4 if lse else 0,
        "golden": out_bytes + (q_pos * n1 * 4 if lse else 0),
    }


def tensor_footprint(params: Dict[str, Any]) -> Dict[str, int]:
    """输入/输出张量字节数，以及 golden 单段输出的额外量。"""
    t = _tensors(params)
    layout_q = str(_one(params, "layout_q", "BNSD"))
    layout_kv = str(_one(params, "layout_kv", layout_q))
    s1 = _as_int(_one(params, "S1", 1), 1)
    if layout_q == "TND":
        q_seg = _max_cu(
            params, "cu_seqlens_q", _max_sequence(params, "seqused_q", "", s1)
        )
    else:
        q_seg = s1
    kv_seg = min(
        _max_sequence(
            params, "seqused_kv", "cu_seqlens_kv", _as_int(_one(params, "S2", 1), 1)
        ),
        _as_int(_one(params, "S2", 1), 1),
    )
    n1 = _as_int(_one(params, "N1", 1), 1)
    n2 = _as_int(_one(params, "N2", n1), n1)
    d = _as_int(_one(params, "D", 128), 128)
    dv = _as_int(_one(params, "DV", d), d)
    size = dtype_bytes(_one(params, "dtype", _one(params, "Dtype", "fp16")))
    out_seg = q_seg * n1 * dv * size
    out_extra = int(out_seg * 4 / size)
    pa_dense = t["kv"] if layout_kv.startswith("PA_") else 0
    return {
        **t,
        "inputs": t["q"] + t["kv"] + t["aux"] + t["mask"],
        "q_seg": q_seg,
        "kv_seg": kv_seg,
        "layout_kv": layout_kv,
        "n2": n2,
        "d": d,
        "dv": dv,
        "size": size,
        "out_seg": out_seg,
        "out_extra": out_extra,
        "pa_dense": pa_dense,
    }


def estimate(
    params: Dict[str, Any],
    *,
    golden_device: str = "npu",
    compare_mode: str = "two_way",
    persist: bool = False,
    peer_like_truth: bool = False,
    with_golden: bool = True,
    golden_spec=None,
) -> Dict[str, Any]:
    """按阶段组合峰值；golden 中间量由 golden 实现提供。"""
    from . import golden_truth

    t = tensor_footprint(params)
    on_device_golden = golden_device == "npu"
    output = t["out"] + t["lse"]
    inputs = t["inputs"]
    device = 0
    host = COMPARE_CHUNK

    if on_device_golden:
        device = max(device, inputs, inputs + output)
    else:
        host = max(host, inputs, inputs + output)

    chunk = golden_truth.golden_chunk_bytes(golden_device)
    p_dtype = getattr(golden_spec, "p_dtype", None) if golden_spec else None
    p_block = getattr(golden_spec, "p_block", 0) if golden_spec else 0
    softmax_mode = (
        getattr(golden_spec, "softmax_mode", "auto") if golden_spec else "auto"
    )
    softmax_block_size = (
        getattr(golden_spec, "softmax_block_size", 0) if golden_spec else 0
    )
    kv_loader = getattr(golden_spec, "kv_loader", "dense") if golden_spec else "dense"
    scratch = golden_truth.golden_peak(
        t,
        params,
        device=golden_device,
        chunk_bytes=chunk,
        p_dtype=p_dtype,
        p_block=p_block,
        softmax_mode=softmax_mode,
        softmax_block_size=softmax_block_size,
        kv_loader=kv_loader,
        keep_on_device=on_device_golden,
        include_golden=with_golden,
    )
    device = max(device, scratch["device_peak"])
    host = max(host, scratch["host_peak"])

    if with_golden:
        compare = inputs + output + t["golden"]
        if compare_mode == "three_way" and not peer_like_truth:
            compare += t["golden"]
        if on_device_golden:
            device = max(device, compare)
        else:
            host = max(host, compare)

    if persist and on_device_golden:
        host += max(t["out"] + t["lse"], t["golden"])

    return {"host_peak": int(host), "device_peak": int(device)}
