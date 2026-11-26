# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""参数归一化：补默认值、dtype 别名、TND/PA 派生（自 core/case_loader 迁入）。"""

import torch


def normalize_params(raw: dict, derive_block_table: bool = True) -> dict:
    """将 test_cases 原始字段补全为计算所需参数。"""

    c = dict(raw)
    layout_q = c.get("layout_q", "BNSD")
    c["input_layout"] = layout_q
    c.setdefault("layout_kv", layout_q)
    c.setdefault("layout_out", layout_q)
    c.setdefault("N2", c["N1"])
    c.setdefault("S2", c.get("S1"))
    c.setdefault("DV", c.get("D"))
    c.setdefault("DRope", 0)
    c.setdefault("win_left", 2147483647)
    c.setdefault("win_right", 2147483647)
    c.setdefault("mask_mode", 0)

    for key in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv"):
        if c.get(key) == [None] or c.get(key) is None:
            c.pop(key, None)

    if layout_q == "TND":
        c.setdefault(
            "seqused_q",
            [
                c["cu_seqlens_q"][i + 1] - c["cu_seqlens_q"][i]
                for i in range(len(c["cu_seqlens_q"]) - 1)
            ],
        )
        if c.get("layout_kv", layout_q) not in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            c.setdefault("cu_seqlens_kv", list(c["cu_seqlens_q"]))
            c.setdefault(
                "seqused_kv",
                [
                    c["cu_seqlens_kv"][i + 1] - c["cu_seqlens_kv"][i]
                    for i in range(len(c["cu_seqlens_kv"]) - 1)
                ],
            )
        c["B"] = 1

    layout_kv_val = c.get("layout_kv", layout_q)
    if layout_kv_val in ("PA_BBND", "PA_BNBD", "PA_NZ"):
        c.setdefault("block_size", 128)
        if "seqused_kv" not in c:
            c["seqused_kv"] = [c.get("S2", c.get("S1"))] * c.get("B", 1)

        if layout_kv_val == "PA_NZ":
            dtype_str = c.get("Dtype", "fp16")
            nz_blk_elem = 16 if dtype_str in ("fp16", "bf16") else 32
            c["nz_blk_elem"] = nz_blk_elem
            c["D_nz_sub"] = c["D"] // nz_blk_elem
            c["DV_nz_sub"] = c.get("DV", c["D"]) // nz_blk_elem

        if not derive_block_table:
            return c

        # 用例提供了 block_table 直接用，不重新生成（幂等：二次归一化也走这里）
        if c.get("block_table") is not None and isinstance(c.get("block_table"), list):
            bt_raw = c["block_table"]
            flat = [
                value
                for row in bt_raw
                for value in (row if isinstance(row, (list, tuple)) else [row])
            ]
            nums = [int(v) for v in flat if isinstance(v, (int, float)) and int(v) >= 0]
            c["block_table"] = torch.tensor(bt_raw, dtype=torch.int32)
            c["actual_b"] = int(c["block_table"].shape[0])
            c["num_blocks"] = (max(nums) + 1) if nums else 0
            return c

        seqused_kv = c["seqused_kv"]
        block_size = c["block_size"]
        bt_shape = c.get("block_table_shape", [])

        if bt_shape:
            b_val = bt_shape[0]
            max_blk = bt_shape[1]
        else:
            b_val = c.get("B", 1)
            if layout_q == "TND":
                b_val = len(c["cu_seqlens_q"]) - 1
            max_blk = (max(seqused_kv) + block_size - 1) // block_size

        block_table = torch.full((b_val, max_blk), -1, dtype=torch.int32)
        block_idx = 0
        for i in range(b_val):
            b_seq = seqused_kv[i] if i < len(seqused_kv) else seqused_kv[0]
            b_blk = (b_seq + block_size - 1) // block_size
            for j in range(b_blk):
                block_table[i][j] = block_idx
                block_idx += 1
        c["block_table"] = block_table
        c["actual_b"] = b_val
        c["num_blocks"] = block_idx  # PA 时 K/V 需要的总 page 数
    return c


def normalize(params):
    """幂等归一化（补默认值 / dtype 别名 / TND 派生）；不保留派生的 block_table。"""
    normalized = normalize_params(dict(params), derive_block_table=False)
    # 派生量不入用例参数：block_table 体积大且可由 seqused/B/block_size 重建
    for key in ("block_table", "actual_b", "num_blocks"):
        normalized.pop(key, None)
    dtype = normalized.get("dtype", normalized.get("Dtype", "fp16"))
    normalized["dtype"] = dtype
    normalized["Dtype"] = dtype
    normalized.pop("input_layout", None)  # 与 layout_q 重复
    for key, value in list(normalized.items()):
        if torch.is_tensor(value):
            normalized[key] = value.tolist()

    int_keys = (
        "B",
        "N1",
        "N2",
        "S1",
        "S2",
        "D",
        "DV",
        "DRope",
        "mask_mode",
        "win_left",
        "win_right",
        "block_size",
    )
    for key in int_keys:
        value = normalized.get(key)
        if isinstance(value, str):
            try:
                normalized[key] = int(float(value))
            except ValueError:
                pass
    return normalized


def materialize(params):
    """按需补齐执行期派生量（PA 的 block_table / num_blocks / actual_b）。"""
    p = dict(params)
    if str(p.get("layout_kv", "")).startswith("PA_") and p.get("block_table") is None:
        p = normalize_params(p)
        for key, value in list(p.items()):
            if torch.is_tensor(value):
                p[key] = value.tolist()
    return p
