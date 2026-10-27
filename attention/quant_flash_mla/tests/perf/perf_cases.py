#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""quant_flash_mla 性能用例集: 运行时从 pytest 参数集加载。

用例直接复用 ../pytest/qfmla_paramset_func_rdv.CASES, 避免手工转换后与
功能用例脱节; SKIP_CASES 中的用例同样跳过 (与 pytest 行为一致)。

返回字段:
  B, N_q, N_kv, seqused_q, cache_seqlens, enable_pa, kv_cache_layout,
  sparse_mode (即 mask_mode), enable_lse, num_blocks,
  quant_mode, layout, graph_path
"""

import sys
from pathlib import Path

_PYTEST_DIR = Path(__file__).resolve().parent.parent / "pytest"
if str(_PYTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTEST_DIR))

from qfmla_paramset_func_rdv import CASES as _CASES  # noqa: E402
from qfmla_paramset_func_rdv import SKIP_CASES as _SKIP_CASES  # noqa: E402

FIELDS = (
    "B",
    "N_q",
    "N_kv",
    "seqused_q",
    "cache_seqlens",
    "enable_pa",
    "kv_cache_layout",
    "sparse_mode",
    "enable_lse",
    "num_blocks",
    "quant_mode",
    "layout",
    "graph_path",
)


def load_cases():
    """返回 {case_name: {字段: 值}}。"""
    cases = {}
    for case in _CASES:
        name = case["name"]
        if name in _SKIP_CASES:
            continue
        cases[name] = {
            "B": case["B"],
            "N_q": case["N_q"],
            "N_kv": case["N_kv"],
            "seqused_q": list(case["seqused_q"]),
            "cache_seqlens": list(case["cache_seqlens"]),
            "enable_pa": bool(case.get("enable_pa", True)),
            "kv_cache_layout": case.get("kv_cache_layout", "BnNBsD"),
            "sparse_mode": int(case.get("mask_mode", 0)),
            "enable_lse": bool(case.get("enable_lse", False)),
            "num_blocks": int(case.get("num_blocks", 0)),
            "quant_mode": int(case.get("quant_mode", 1)),
            "layout": case.get("layout", "TND"),
            "graph_path": int(case.get("graph_path", 0)),
        }
    return cases
