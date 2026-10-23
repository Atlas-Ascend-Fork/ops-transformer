#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""apply_rotary_pos_emb 的 TTK 多路径 TestSpec golden。

对 query/key 最后一维前 reald = cos.shape[-1] 个元素施加旋转位置编码，
reald < D 时（partial rope）尾部 [reald, D) 原样保留（算子为原地更新语义，
kernel 模式下 TTK 依据输出与输入同名自动 inplace，尾部即输入透传）：

    q_embed[..., :reald] = query[..., :reald] * cos + rotate(query[..., :reald]) * sin
    k_embed[..., :reald] = key[..., :reald]   * cos + rotate(key[..., :reald])   * sin

rotary_mode 决定 rotate 的重排方式（作用在前缀内部）：
    half        cat(-后半段, 前半段)，沿最后一维对半 chunk
    interleave  相邻两元素配对：偶数位取 -x_odd、奇数位取 x_even
    quarter     四等分后 cat(-Q2, Q1, -Q4, Q3)

cos/sin 的 N 轴恒为 1、B 轴可为 1 或等于 q/k 的 B 轴（TND 无 B 轴），与
BSND/SBND/BNSD/TND 四种 layout 的广播语义由 numpy 广播规则统一覆盖，golden
不区分 layout。所有路径统一升 FP32 计算，结果转回输入 dtype。
"""

__spec__ = {
    "apply_rotary_pos_emb": "ApplyRotaryPosEmbKernelSpec",
    "aclnnApplyRotaryPosEmb": "AclnnApplyRotaryPosEmbSpec",
    "aclnnApplyRotaryPosEmbV2": "AclnnApplyRotaryPosEmbV2Spec",
    "cann_ops_transformer.apply_rotary_pos_emb": "ApplyRotaryPosEmbE2ESpec",
    "torch.ops.cann_ops_transformer.apply_rotary_pos_emb": "ApplyRotaryPosEmbE2ESpec",
}

import numpy as np

try:
    from ml_dtypes import bfloat16 as _bf16
except ImportError:  # pragma: no cover
    _bf16 = None


def _to_np(t):
    """Normalize numpy / torch (CPU or NPU) input to a numpy ndarray (or None)."""
    if t is None:
        return None
    if isinstance(t, np.ndarray):
        return t
    import torch

    if isinstance(t, torch.Tensor):
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        # torch CPU half/bfloat16 have no stable numpy() view; lift to fp32.
        if t.dtype in (torch.float16, torch.bfloat16):
            return t.float().numpy()
        return t.numpy()
    return np.asarray(t)


def _numpy_dtype(dtype):
    """dtype 名 → numpy dtype（bf16 经 ml_dtypes）。"""
    name = str(dtype)
    if "bfloat16" in name:
        return _bf16 if _bf16 is not None else np.float32
    return np.dtype(name).type


def _rotate_prefix(prefix, mode):
    """对已切片的前缀张量按 rotary_mode 重排。prefix 最后一维为 reald。"""
    if mode == "half":
        mid = prefix.shape[-1] // 2
        return np.concatenate((-prefix[..., mid:], prefix[..., :mid]), axis=-1)
    if mode == "interleave":
        even = prefix[..., 0::2]
        odd = prefix[..., 1::2]
        out = np.empty(prefix.shape, dtype=prefix.dtype)
        out[..., 0::2] = -odd
        out[..., 1::2] = even
        return out
    if mode == "quarter":
        q = np.split(prefix, 4, axis=-1)
        return np.concatenate((-q[1], q[0], -q[3], q[2]), axis=-1)
    raise ValueError(
        f"golden 仅支持 half/interleave/quarter, 收到 rotary_mode={mode!r}"
    )


def _ref_apply_rotary_pos_emb(query, key, cos, sin, rotary_mode, target):
    """float32 CPU reference shared by all paths.

    query/key: 任意 layout (BSND/SBND/BNSD/TND) 的张量，仅作用于前 reald 列；
    cos/sin:   与 query/key 按 numpy 广播规则对齐（N 轴为 1，B 轴可广播）。
    """
    q = np.asarray(query)
    k = np.asarray(key)
    c = np.asarray(cos)
    s = np.asarray(sin)

    if c.shape != s.shape:
        raise ValueError(f"cos shape {c.shape} 与 sin shape {s.shape} 应相同")
    reald = c.shape[-1]
    if q.shape[-1] < reald or k.shape[-1] < reald:
        raise ValueError("cos/sin 最后一维不能大于 query/key 最后一维")

    def _f32(a):
        return a.astype(np.float32) if a.dtype != np.float32 else a

    q32, k32, c32, s32 = _f32(q), _f32(k), _f32(c), _f32(s)

    def _apply(x):
        y = x.copy()
        prefix = x[..., :reald]
        y[..., :reald] = c32 * prefix + _rotate_prefix(prefix, rotary_mode) * s32
        return y

    q_out = _apply(q32)
    k_out = _apply(k32)

    if target == "bfloat16":
        if _bf16 is None:
            raise RuntimeError("ml_dtypes 不可用, 无法产出 bfloat16 golden")
        return [q_out.astype(_bf16), k_out.astype(_bf16)]
    if target == "float16":
        return [q_out.astype(np.float16), k_out.astype(np.float16)]
    return [q_out, k_out]


def _acl_output_target(dtype):
    dt = str(dtype)
    if "bfloat16" in dt:
        return "bfloat16"
    if "float16" in dt:
        return "float16"
    return "float32"


# ---- Kernel / GEIR（numpy.ndarray）-------------------------------------------------
# GEIR 复用 Kernel 的注册（op_name=apply_rotary_pos_emb）。
class ApplyRotaryPosEmbKernelSpec:
    """apply_rotary_pos_emb 的 Kernel / GEIR 流程 golden（输入为 numpy.ndarray）。

    def.cpp inputs : query, key, cos, sin
    def.cpp outputs: query, key（与输入同名，TTK 自动 inplace）
    def.cpp attrs  : layout(int, 默认 1), rotary_mode(str, 默认 "half")
    """

    def golden(query, key, cos, sin, layout=1, rotary_mode="half", **kwargs):
        if int(layout) not in (1, 2, 3, 4):
            raise ValueError(f"非法 layout={layout!r} (1=BSND, 2=SBND, 3=BNSD, 4=TND)")
        target = _acl_output_target(query.dtype)
        return _ref_apply_rotary_pos_emb(query, key, cos, sin, rotary_mode, target)


# ---- ACLNN : aclnnApplyRotaryPosEmb(queryRef, keyRef, cos, sin, layout) —— 原地更新 ----
class AclnnApplyRotaryPosEmbSpec:
    """aclnnApplyRotaryPosEmb 的 ACLNN 流程 golden（输入为 torch.Tensor，rotary_mode 恒为 half）。"""

    def golden(queryRef, keyRef, cos, sin, layout=1, **kwargs):
        if int(layout) not in (1, 4):
            raise ValueError(
                f"aclnnApplyRotaryPosEmb 仅支持 layout 1/4, 收到 {layout!r}"
            )
        target = _acl_output_target(queryRef.dtype)
        return _ref_apply_rotary_pos_emb(
            _to_np(queryRef), _to_np(keyRef), _to_np(cos), _to_np(sin), "half", target
        )


# ---- ACLNN V2 : aclnnApplyRotaryPosEmbV2(queryRef, keyRef, cos, sin, layout, rotaryMode) ----
class AclnnApplyRotaryPosEmbV2Spec:
    """aclnnApplyRotaryPosEmbV2 的 ACLNN 流程 golden（输入为 torch.Tensor）。"""

    def golden(queryRef, keyRef, cos, sin, layout=1, rotaryMode="half", **kwargs):
        if int(layout) not in (1, 4):
            raise ValueError(
                f"aclnnApplyRotaryPosEmbV2 仅支持 layout 1/4, 收到 {layout!r}"
            )
        target = _acl_output_target(queryRef.dtype)
        return _ref_apply_rotary_pos_emb(
            _to_np(queryRef),
            _to_np(keyRef),
            _to_np(cos),
            _to_np(sin),
            rotaryMode,
            target,
        )


# ---- E2E : cann_ops_transformer.apply_rotary_pos_emb(query, key, cos, sin, layout: str, rotary_mode: str) ----
class ApplyRotaryPosEmbE2ESpec:
    """cann_ops_transformer.apply_rotary_pos_emb 的 E2E 流程 golden（输入为 torch.Tensor）。

    torch extension 内部 clone 后原地调用 aclnnV2，输出为独立张量；
    partial rope 尾部同样为输入透传，与本 golden 语义一致。
    """

    def golden(query, key, cos, sin, layout="BSND", rotary_mode="half", **kwargs):
        if str(layout) not in ("BSND", "BSH", "SBND", "BNSD", "TND"):
            raise ValueError(f"非法 layout={layout!r}")
        target = _acl_output_target(query.dtype)
        return _ref_apply_rotary_pos_emb(
            _to_np(query), _to_np(key), _to_np(cos), _to_np(sin), rotary_mode, target
        )
