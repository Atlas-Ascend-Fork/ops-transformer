# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Golden 规格：设备 / 计算精度 / P 落盘精度 / 分块，及指纹。"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GoldenSpec:
    device: str = "cpu"  # cpu | npu | gpu
    compute_dtype: str = "fp32"  # fp32 | fp64
    p_dtype: str = (
        "out_dtype"  # fp32 | out_dtype；默认与 kernel 同口径（P 落盘到 V 的 dtype）
    )
    p_block: int = 0  # P 落盘 / 在线归一化的块大小；0=auto（按 op tiling 规则）
    softmax_mode: str = "auto"  # auto | full | online
    softmax_block_size: int = 0  # online 模式的 KV 块大小；0=auto
    kv_loader: str = "dense"  # dense | paged
    provider: str = "default"  # default(高精度真值) | peer(kernel 同误差预算)

    def fingerprint(self) -> str:
        raw = "|".join(f"{k}={v}" for k, v in sorted(asdict(self).items()))
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def describe(self) -> str:
        return (
            f"{self.device}/{self.compute_dtype}/p={self.p_dtype}"
            f"/blk={self.p_block}/{self.softmax_mode}"
            f"/sblk={self.softmax_block_size}/kv={self.kv_loader}/{self.provider}"
        )


def parse_spec(text: str) -> GoldenSpec:
    """解析 `device=cpu,compute_dtype=fp64,p_dtype=out_dtype` 形式。"""
    fields = {}
    if text:
        for item in text.split(","):
            item = item.strip()
            if not item:
                continue
            key, _, value = item.partition("=")
            if not value:
                raise ValueError(f"golden spec item must be key=value: {item!r}")
            fields[key.strip()] = value.strip()
    unknown = set(fields) - set(GoldenSpec.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown golden spec keys: {sorted(unknown)}")
    if "device" in fields and fields["device"] not in ("cpu", "npu", "gpu"):
        raise ValueError(
            f"device must be 'cpu', 'npu' or 'gpu', got {fields['device']!r}"
        )
    if "p_block" in fields:
        fields["p_block"] = int(fields["p_block"])
    if "softmax_block_size" in fields:
        fields["softmax_block_size"] = int(fields["softmax_block_size"])
    if "softmax_mode" in fields and fields["softmax_mode"] not in (
        "auto",
        "full",
        "online",
    ):
        raise ValueError(
            f"softmax_mode must be 'auto', 'full' or 'online', "
            f"got {fields['softmax_mode']!r}"
        )
    if "kv_loader" in fields and fields["kv_loader"] not in ("dense", "paged"):
        raise ValueError(
            f"kv_loader must be 'dense' or 'paged', got {fields['kv_loader']!r}"
        )
    if fields.get("softmax_block_size", 0) < 0:
        raise ValueError("softmax_block_size must be >= 0")
    if "p_dtype" in fields and fields["p_dtype"] not in ("fp32", "out_dtype"):
        raise ValueError(
            f"p_dtype must be 'fp32' or 'out_dtype' (non-quantized operator), "
            f"got {fields['p_dtype']!r}"
        )
    return GoldenSpec(**fields)
