#!/usr/bin/python3
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


"""TTK golden plugin for quant_flash_mla.

Regenerates BNSD data via generate_data() (same seeds → same data as inputs),
calls the CPU golden reference, converts output to target layout, and returns
(attention_out, lse_out) for comparison.
"""

import importlib.util
import logging
import math
import os
import sys

import numpy as np
import torch

_ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_DIR = os.path.join(_ASSETS_DIR, "..")  # assets/
if _GOLDEN_DIR not in sys.path:
    sys.path.insert(0, _GOLDEN_DIR)
# golden 模块依赖 pytest/common/ 下的 result_compare_method
_COMMON_DIR = os.path.join(_ASSETS_DIR, "..", "..", "pytest", "common")
if _COMMON_DIR not in sys.path:
    sys.path.insert(0, _COMMON_DIR)

_GOLDEN_MODULE_NAME = "qfmla_assets_golden_reference"
if _GOLDEN_MODULE_NAME in sys.modules:
    golden_mod = sys.modules[_GOLDEN_MODULE_NAME]
else:
    _GOLDEN_PATH = os.path.join(_GOLDEN_DIR, "qfmla_golden.py")
    _GOLDEN_SPEC = importlib.util.spec_from_file_location(
        _GOLDEN_MODULE_NAME, _GOLDEN_PATH
    )
    golden_mod = importlib.util.module_from_spec(_GOLDEN_SPEC)
    sys.modules[_GOLDEN_MODULE_NAME] = golden_mod
    _GOLDEN_SPEC.loader.exec_module(golden_mod)

logger = logging.getLogger(__name__)

_KV_LAYOUT_REVERSE = {
    "PA_BNBD": "BnNBsD",
    "PA_BBND": "BnBsND",
    "PA_NZ": "NZ",
}


def _set_golden_globals(
    quant_mode,
    mask_mode,
    layout_q,
    layout_kv,
    layout_out,
    max_seqlen_q,
    max_seqlen_kv,
    head_dim_v,
    return_softmax_lse,
    softmax_scale,
    q=None,
    k_cache=None,
    cache_seqlens=None,
    **kwargs,
):
    """Set golden module globals from op params and sidecar kwargs.

    generate_data 依赖全局量 (B/N_q/N_kv/D/D_V/SEQUSED_Q/CACHE_SEQLENS/
    种子/数据范围) 决定数据 shape 与 RNG 消费流, 必须与 inputs.py 侧完全对齐,
    否则 "同种子同数据" 失效, golden 再生数据与 NPU 输入数据错位。因此这里的
    数据合成全局量推导逻辑与 inputs.generate_qfmla_inputs 逐条对应; 而 CPU
    golden 计算仍使用原始 seqused_q_values/cache_seqlens_values。
    """
    # case 属性一律不硬编码兜底 (对齐 QFA tests/assets/impl/golden.py 形态):
    # 属性缺失即缺失, 能由 slot 形状推导的就地推导, 不能推导的显式报错, 避免
    # 静默补值掩盖测试脚本漏配参数。
    B = kwargs.get("B")
    N_q = kwargs.get("N_q")
    N_kv = kwargs.get("N_kv")
    D = kwargs.get("D")
    D_V = kwargs.get("D_V")
    if D_V is None:
        D_V = head_dim_v
    block_size = kwargs.get("block_size")
    if block_size is None:
        raise ValueError(
            "[GOLDEN] attribute block_size is not provided by the case and cannot "
            "be derived from slot shapes"
        )
    seqused_q_vals = kwargs.get("seqused_q_values")
    if seqused_q_vals is not None:
        seqused_q_vals = list(seqused_q_vals)
    cache_seqlens_vals = kwargs.get("cache_seqlens_values")
    if cache_seqlens_vals is not None:
        cache_seqlens_vals = list(cache_seqlens_vals)
    cu_seqlens_q_vals = kwargs.get("cu_seqlens_q_values")
    if cu_seqlens_q_vals is not None:
        cu_seqlens_q_vals = list(cu_seqlens_q_vals)
    kv_cache_layout = kwargs.get("kv_cache_layout") or _KV_LAYOUT_REVERSE.get(
        layout_kv, "BnNBsD"
    )
    input_layout = kwargs.get("input_layout", layout_q)
    output_layout = kwargs.get("output_layout", layout_out)

    # ---- B: 与 inputs.py 一致, 优先从 cache_seqlens slot 首维推导 ----
    cache_shape = (
        tuple(int(dim) for dim in cache_seqlens.shape)
        if (cache_seqlens is not None and hasattr(cache_seqlens, "shape"))
        else ()
    )
    if len(cache_shape) == 1 and cache_shape[0] > 0:
        B = cache_shape[0]
    elif cache_seqlens_vals:
        B = max(1, len(cache_seqlens_vals))
    else:
        B = max(1, int(B)) if B is not None else 1

    # ---- 非法 N_q/N_kv/D 从 slot 形状推导 (与 inputs.py 一致) ----
    def _positive_or_derive(name, value, slot, axis):
        try:
            ivalue = int(value)
        except (TypeError, ValueError):
            ivalue = 0
        if ivalue > 0:
            return ivalue
        shape = (
            tuple(int(dim) for dim in slot.shape)
            if (slot is not None and hasattr(slot, "shape"))
            else ()
        )
        idx = axis if axis >= 0 else len(shape) + axis
        if not 0 <= idx < len(shape) or shape[idx] <= 0:
            raise ValueError(
                f"[GOLDEN] attribute {name}={value!r} invalid and cannot be "
                f"derived from slot shape {shape}"
            )
        logger.warning(
            "[GOLDEN] attribute %s=%r invalid - derive %s=%d from slot shape %s",
            name,
            value,
            name,
            shape[idx],
            shape,
        )
        return shape[idx]

    q_shape = (
        tuple(int(dim) for dim in q.shape)
        if (q is not None and hasattr(q, "shape"))
        else ()
    )
    _q_head_axis = 2 if str(layout_q).upper() == "BSND" else 1
    N_q = _positive_or_derive("N_q", N_q, q, _q_head_axis)
    _k_head_axis = 2 if str(layout_kv).upper() == "PA_BBND" else 1
    N_kv = _positive_or_derive("N_kv", N_kv, k_cache, _k_head_axis)

    # D 是算子侧 QK 总 head dim (qkHeadDim, 如 576), 只有一个 D, 直接作为数据
    # 合成宽度 (不再拆 nope/rope); 以 q slot 末维 (算子实际接收的 QK head_dim)
    # 为准, 无槽位信息时用 case 属性, 不做任何兜底补值。
    if D is None:
        raise ValueError("[GOLDEN] attribute D is not provided by the case")
    _q_last_dim = q_shape[-1] if q_shape else 0
    D_total = _q_last_dim if _q_last_dim > 0 else int(D)

    # ---- synthesis seqused: 与 inputs.py 一致, cu 差值优先, 否则按 slot 均分 ----
    synthesis_seqused_q_vals = None
    if (
        cu_seqlens_q_vals is not None
        and len(cu_seqlens_q_vals) == int(B) + 1
        and cu_seqlens_q_vals
        and cu_seqlens_q_vals[0] == 0
        and all(
            cu_seqlens_q_vals[i + 1] >= cu_seqlens_q_vals[i]
            for i in range(len(cu_seqlens_q_vals) - 1)
        )
    ):
        candidate = [
            cu_seqlens_q_vals[i + 1] - cu_seqlens_q_vals[i] for i in range(int(B))
        ]
        if candidate and max(candidate) > 0:
            synthesis_seqused_q_vals = candidate
    if synthesis_seqused_q_vals is None:
        batch_for_synthesis = max(int(B), 1)
        total_q = (
            q_shape[0] if len(q_shape) == 3 and q_shape[0] > 0 else batch_for_synthesis
        )
        base, extra = divmod(total_q, batch_for_synthesis)
        synthesis_seqused_q_vals = [
            base + (1 if i < extra else 0) for i in range(batch_for_synthesis)
        ]
        if max(synthesis_seqused_q_vals) <= 0:
            synthesis_seqused_q_vals = [1] * batch_for_synthesis

    synthesis_cache_seqlens_vals = cache_seqlens_vals
    if (
        not synthesis_cache_seqlens_vals
        or len(synthesis_cache_seqlens_vals) != int(B)
        or any(int(v) <= 0 for v in synthesis_cache_seqlens_vals)
    ):
        synthesis_cache_seqlens_vals = [1] * max(int(B), 1)

    # case 未提供 per-batch 长度 (seqused_q_values/cache_seqlens_values 为 None)
    # 时, CPU golden 与布局转换改用与 inputs.py 同源的 synthesis 推导值, 不再
    # 硬编码 [1]/[1024]; 提供了原始值的用例原样透传。
    if seqused_q_vals is None:
        seqused_q_vals = list(synthesis_seqused_q_vals)
    if cache_seqlens_vals is None:
        cache_seqlens_vals = list(synthesis_cache_seqlens_vals)

    golden_mod.B = B
    golden_mod.N_q = N_q
    golden_mod.N_kv = N_kv
    golden_mod.D = D_total
    golden_mod.D_V = D_V
    golden_mod.QFMLA_QK_HEAD_DIM = D_total
    golden_mod.QFMLA_V_HEAD_DIM = D_V
    golden_mod.QFMLA_QUANT_MODE = quant_mode
    # 数据合成用 synthesis 值 (与 inputs.py 一致); CPU golden 计算用原始值
    golden_mod.SEQUSED_Q = list(synthesis_seqused_q_vals)
    golden_mod.CACHE_SEQLENS = list(synthesis_cache_seqlens_vals)
    golden_mod.ENABLE_PA = True
    golden_mod.ENABLE_LSE = return_softmax_lse
    golden_mod.GOLDEN_MODE = True
    golden_mod.BLOCK_SIZE = block_size
    golden_mod.MASK_MODE = mask_mode
    golden_mod.INPUT_LAYOUT = input_layout
    golden_mod.OUTPUT_LAYOUT = output_layout
    golden_mod.KV_CACHE_LAYOUT = kv_cache_layout
    golden_mod.P_SCALE = float(kwargs.get("p_scale", 1.0))
    golden_mod.GRAPH_PATH = kwargs.get("graph_path", 0)
    golden_mod.NUM_BLOCKS = kwargs.get("num_blocks", 0)
    golden_mod.SCALE_VALUE = softmax_scale
    golden_mod.MAX_SEQLEN_Q = max_seqlen_q
    golden_mod.MAX_SEQLEN_KV = max_seqlen_kv

    # csv precision_tolerances / absolute_precision 暂存到 golden_mod 供 compare
    # 读取 (对齐 QFA golden.py 风格: ttk 框架不把 testcase 对象直接传给 custom
    # compare, golden 插件在调用前把 csv 容差暂存到 golden 模块上)。
    golden_mod._csv_precision_tolerances = kwargs.get("precision_tolerances")
    golden_mod._csv_absolute_precision = kwargs.get("absolute_precision")

    if kwargs.get("q_data_range"):
        golden_mod.Q_DATA_RANGE = tuple(kwargs["q_data_range"])
    if kwargs.get("k_data_range"):
        golden_mod.K_DATA_RANGE = tuple(kwargs["k_data_range"])
    if kwargs.get("v_data_range"):
        golden_mod.V_DATA_RANGE = tuple(kwargs["v_data_range"])

    for s in ("SEED_Q", "SEED_K", "SEED_V", "SEED_BLOCK_TABLE"):
        key = s.lower()
        if key in kwargs:
            setattr(golden_mod, s, kwargs[key])

    return {
        "B": B,
        "N_q": N_q,
        "N_kv": N_kv,
        "D": D_total,
        "D_V": D_V,
        "block_size": block_size,
        "seqused_q_vals": seqused_q_vals,
        "cache_seqlens_vals": cache_seqlens_vals,
        "kv_cache_layout": kv_cache_layout,
        "input_layout": input_layout,
        "output_layout": output_layout,
    }


def cpu_qfmla_mxfp8(
    q,
    k_cache,
    q_descale,
    k_descale,
    block_table,
    cache_seqlens,
    quant_mode,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_v=512,
    layout_q="TND",
    layout_kv="PA_BNBD",
    layout_out="TND",
    return_softmax_lse=False,
    **kwargs,
):
    """CPU golden reference for quant_flash_mla.

    Regenerates BNSD data with same seeds as inputs, computes CPU golden,
    converts output to target layout, returns (attention_out, lse_out).
    """
    testcase_name = kwargs.get("testcase_name", "unknown")
    logger.info(
        "[%s] golden: quant_mode=%s layout_q=%s layout_out=%s",
        testcase_name,
        quant_mode,
        layout_q,
        layout_out,
    )

    # ---- 1. Set golden globals ----
    # 数据合成全局量 (B/N_q/N_kv/D/SEQUSED_Q/CACHE_SEQLENS) 的推导在
    # _set_golden_globals 内完成, 与 inputs.py 逐条对齐 (D_total 语义、
    # cu 差值 synthesis seqused 等), 保证 "同种子同数据"。
    cfg = _set_golden_globals(
        quant_mode=quant_mode,
        mask_mode=mask_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_out=layout_out,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim_v=head_dim_v,
        return_softmax_lse=return_softmax_lse,
        softmax_scale=softmax_scale,
        q=q,
        k_cache=k_cache,
        cache_seqlens=cache_seqlens,
        **kwargs,
    )

    # ---- 2. Regenerate BNSD data (same seeds → same data) ----
    (q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale_t, qr_bf16, kr_bf16) = (
        golden_mod.generate_data(device="cpu")
    )

    logger.info(
        "[%s] golden: regenerated q=%s k=%s deq_q=%s",
        testcase_name,
        tuple(q_fp8.shape),
        tuple(k_fp8.shape),
        tuple(deq_q.shape),
    )

    # ---- 3. Call CPU golden ----
    result_bnsd, lse_bnsd = golden_mod.cpu_fp8_fullquant_mla_golden(
        q_fp8,
        k_fp8,
        v_fp8,
        deq_q,
        deq_k,
        deq_v,
        p_scale_t,
        list(cfg["seqused_q_vals"]),
        list(cfg["cache_seqlens_vals"]),
        qr_bf16=qr_bf16,
        kr_bf16=kr_bf16,
    )

    logger.info(
        "[%s] golden: result=%s lse=%s",
        testcase_name,
        tuple(result_bnsd.shape) if result_bnsd is not None else None,
        tuple(lse_bnsd.shape) if lse_bnsd is not None else None,
    )

    # ---- 4. Convert output to target layout ----
    output_layout = cfg["output_layout"]

    def _convert_tnd_full_rows(bnsd, seqused, cu_values, to_ntd=False):
        """TND/NTD输出按算子语义转换: T=全量T(q行数), 每批按cu偏移定位,
        有效行=seqused[b], 无效行attn补0/lse补inf (NPU实测: 无效行attn=0, lse=inf)。
        bnsd: attn为(B,N,S,Dv) / lse为(B,N,S,1)"""
        is_lse = bnsd.dim() == 4 and bnsd.shape[-1] == 1
        T = int(q.shape[0])
        fill_val = float("inf") if is_lse else 0.0
        if to_ntd:
            out = torch.full(
                (bnsd.shape[1], T, bnsd.shape[3]), fill_val, dtype=bnsd.dtype
            )
        else:
            out = torch.full(
                (T, bnsd.shape[1], bnsd.shape[3]), fill_val, dtype=bnsd.dtype
            )
        for b in range(bnsd.shape[0]):
            act_s = int(seqused[b])
            if act_s <= 0:
                continue
            start = int(cu_values[b])
            valid = bnsd[b, :, :act_s, :]  # (N, act_s, D)
            if to_ntd:
                out[:, start : start + act_s, :] = valid
            else:
                out[start : start + act_s, :, :] = valid.permute(1, 0, 2)
        return out.contiguous()

    cu_values = kwargs.get("cu_seqlens_q_values")
    if output_layout in ("TND", "NTD") and layout_q == "TND" and cu_values:
        # 算子TND输出语义: 全量T(cu偏移定位), 部分seqused时无效行补0/inf
        result_final = _convert_tnd_full_rows(
            result_bnsd,
            cfg["seqused_q_vals"],
            cu_values,
            to_ntd=(output_layout == "NTD"),
        )
    else:
        result_final = golden_mod.convert_q_bnsd_to_layout(
            result_bnsd, cfg["seqused_q_vals"], output_layout
        )

    # ---- 5. Convert lse if needed ----
    lse_final = None
    if return_softmax_lse and lse_bnsd is not None:
        if layout_q == "TND" and cu_values:
            # TND输入: lse输出(N,T)全量T, 无效行补inf (与算子一致)
            lse_final = _convert_tnd_full_rows(
                lse_bnsd, cfg["seqused_q_vals"], cu_values, to_ntd=True
            )
            lse_final = lse_final.squeeze(-1)
        else:
            lse_layout = "BNSD" if output_layout in ("BSND", "BNSD") else output_layout
            lse_final = golden_mod.convert_q_bnsd_to_layout(
                lse_bnsd, cfg["seqused_q_vals"], lse_layout
            )
            if output_layout == "TND":
                # (T, N, 1) → (N, T)
                lse_final = lse_final.permute(1, 0, 2).squeeze(-1).contiguous()
            else:
                # squeeze last dim: (B, N, S, 1) → (B, N, S)
                lse_final = lse_final.squeeze(-1).contiguous()

    # Convert result to bfloat16 to match op output dtype
    result_final = result_final.to(torch.bfloat16)

    logger.info(
        "[%s] golden: final result=%s lse=%s",
        testcase_name,
        tuple(result_final.shape),
        tuple(lse_final.shape) if lse_final is not None else None,
    )

    return result_final, lse_final
