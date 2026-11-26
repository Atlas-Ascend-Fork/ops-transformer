# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""flash_attn 的 P 精度契约参数解析（arch35 路由口径）。

参考实现的 P 分块口径必须与 kernel 的 ``s2BaseSize`` 一致：
kernel 在 s2BaseSize 个 KV 列内做一次 online softmax，P 只在该块内按 V 的 dtype 落盘，
分母/lse 用未落盘的 fp32 P。s2BaseSize 由模板决定（见
``op_host/arch35/flash_attn_kernel_config_arch35.h``）：

===============  ============================================================  ===========
模板              条件                                                          s2BaseSize
===============  ============================================================  ===========
DN_Unmerged      vHeadDim<=128, mask∈{0,3}, maxSeqQ>=256                        256
DN               vHeadDim<256, mask==0, 未触发 M 调整                             128
ND               其余（含 vHeadDim>=256 / maxSeq 未知）                          128（触发 M 调整时 256）
===============  ============================================================  ===========
"""

from __future__ import annotations

from typing import Any, Dict

TEMPLATE_ND = "ND"
TEMPLATE_DN = "DN"
TEMPLATE_DN_UNMERGED = "DN_Unmerged"

_UNBOUNDED_WIN = 2147483647


def _one(params: Dict[str, Any], key: str, default=None):
    value = params.get(key, default)
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _int(params: Dict[str, Any], key: str, default: int = 0) -> int:
    value = _one(params, key, None)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _max_len(params: Dict[str, Any], which: str) -> int:
    """与 harness 传给算子的 max_seqlen 同口径：显式值 > seqused > cu 差分 > S。"""
    explicit = params.get(f"max_seqlen_{which}")
    if explicit is not None:
        return int(explicit)
    used = params.get(f"seqused_{which}")
    if used:
        return max(int(v) for v in used)
    cu = params.get("cu_seqlens_q" if which == "q" else "cu_seqlens_kv")
    if cu:
        return max(int(cu[i + 1]) - int(cu[i]) for i in range(len(cu) - 1))
    return _int(params, "S1" if which == "q" else "S2", -1)


def is_adjust_m_base(params: Dict[str, Any]) -> bool:
    """对应 kernel 的 ``IsAdjustMBaseSizeForTemplateND``。"""
    v_dim = _int(params, "DV", _int(params, "D", 128))
    n1 = _int(params, "N1", 1)
    n2 = _int(params, "N2", n1)
    group = max(n1 // max(n2, 1), 1)
    max_q = _max_len(params, "q")
    max_kv = _max_len(params, "kv")
    mask_mode = _int(params, "mask_mode", 0)
    win_left = _one(params, "win_left", -1)
    win_left = -1 if win_left is None else int(win_left)
    if max_q == -1 or max_kv == -1:
        return False
    if v_dim >= 256:
        return group * max_q < 64
    if max_q <= 64 and max_kv > 128:
        if mask_mode in (0, 3):
            return True
        if mask_mode == 4 and (win_left == -1 or win_left > 128):
            return True
    return False


def resolve_template(params: Dict[str, Any]) -> str:
    """对应 kernel 的 ``CalcTemplateId``。"""
    v_dim = _int(params, "DV", _int(params, "D", 128))
    max_q = _max_len(params, "q")
    max_kv = _max_len(params, "kv")
    mask_mode = _int(params, "mask_mode", 0)
    if v_dim >= 256 or max_q == -1 or max_kv == -1:
        return TEMPLATE_ND
    if v_dim <= 128 and mask_mode in (0, 3) and max_q >= 256:
        return TEMPLATE_DN_UNMERGED
    if mask_mode == 0 and not is_adjust_m_base(params):
        return TEMPLATE_DN
    return TEMPLATE_ND


def resolve_p_block(params: Dict[str, Any]) -> int:
    """参考实现 P 落盘 / online 归一化的块大小（= kernel s2BaseSize）。"""
    template = resolve_template(params)
    if template == TEMPLATE_DN_UNMERGED:
        return 256
    if template == TEMPLATE_DN:
        return 128
    return 256 if is_adjust_m_base(params) else 128


__all__ = [
    "TEMPLATE_ND",
    "TEMPLATE_DN",
    "TEMPLATE_DN_UNMERGED",
    "is_adjust_m_base",
    "resolve_template",
    "resolve_p_block",
]
