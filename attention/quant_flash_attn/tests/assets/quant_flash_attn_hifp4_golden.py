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
"""HiFP4 (quant_mode=10) Flash Attention Golden.

从旧 quant_flash_attn_golden.py 中提取的 HiFP4 专用代码:
  - CPU golden: cpu_hifp4_golden (三阶量化 FP4 E1M2 + 32bit scale)
  - 辅助: _pack_hifloat4_to_uint8
  - FP4 编解码: HiFP4_To_BF16_reverse_order, _hifp4_cvt_* 等

依赖:
  - VALU 库 (valuPythonWork.libVALU): NPU 浮点单元精确仿真, cpu_hifp4_golden 需要
  - ml_dtypes (bfloat16): numpy bfloat16 支持

metadata 生成与主算子调用由框架 (impl/npu_preprocess.py + TTK 直调) 处理,
本模块只提供 CPU golden 参考实现与 FP4 编解码辅助。

全局变量由调用方 (impl/golden.py _apply_hifp4_golden_globals) 在调用前注入。
"""

import ctypes
import logging
import math
import os as _os
import struct

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ==============================================================================
# HiFP4 依赖: VALU (NPU 浮点单元精确仿真) + ml_dtypes (numpy bfloat16)
# ==============================================================================
_HIFP4_VALU_DIR = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)), "..", "..", "pytest_121hif4_new"
)
_HIFP4_VALU_PKG = _os.path.join(_HIFP4_VALU_DIR, "valuPythonWork")

try:
    import importlib.util as _ilu

    _valu_init = _ilu.spec_from_file_location(
        "valuPythonWork._libvalu", _os.path.join(_HIFP4_VALU_PKG, "__init__.py")
    )
    _valu_mod = _ilu.module_from_spec(_valu_init)
    _valu_init.loader.exec_module(_valu_mod)
    from valuPythonWork import libVALU as _hifp4_valu

    _HIFP4_OP_INFO = _hifp4_valu.OpInfo_t()
    _HIFP4_STATUS = _hifp4_valu.OpStatus_t()
    _HIFP4_OP_INFO_CAST = _hifp4_valu.OpInfo_t()
    _HIFP4_OP_INFO_CAST.roundMode = _hifp4_valu.ROUND_MODE.ROUND_Z_BY_TRUNCATED
    _HIFP4_OP_INFO_CAST.sat_info.ApplySat()
except Exception:
    if _HIFP4_VALU_PKG not in _os.sys.path:
        _os.sys.path.insert(0, _HIFP4_VALU_DIR)
    try:
        from valuPythonWork import libVALU as _hifp4_valu

        _HIFP4_OP_INFO = _hifp4_valu.OpInfo_t()
        _HIFP4_STATUS = _hifp4_valu.OpStatus_t()
        _HIFP4_OP_INFO_CAST = _hifp4_valu.OpInfo_t()
        _HIFP4_OP_INFO_CAST.roundMode = _hifp4_valu.ROUND_MODE.ROUND_Z_BY_TRUNCATED
        _HIFP4_OP_INFO_CAST.sat_info.ApplySat()
    except Exception:
        _hifp4_valu = None
        _HIFP4_OP_INFO = None
        _HIFP4_STATUS = None
        _HIFP4_OP_INFO_CAST = None

try:
    from ml_dtypes import bfloat16 as np_bfloat16
except ImportError:
    np_bfloat16 = np.float32

# ==============================================================================
# 配置区 — 全部由调用方 (impl/golden.py) 注入
# ==============================================================================
GRAPH_PATH = int(_os.environ.get("GRAPH_PATH", "0"))

B = None
N_q = None
N_kv = None
D = None

CU_SEQLENS_Q = None
CU_SEQLENS_KV = None
SEQUSED_Q = None
SEQUSED_KV = None
MAX_SEQLEN_Q = None
MAX_SEQLEN_KV = None

ENABLE_PA = False
KV_CACHE_LAYOUT = "BnNBsD"
BLOCK_SIZE = 0
SPARSE_MODE = 0
Q_SCALE_LAYOUT = "BNSD"
ENABLE_LSE = False
INPUT_LAYOUT = "BNSD"
IS_CONTIGUOUS = True
DEVICE_ID = 0
SOFTMAX_SCALE = None
HIFP4_GROUP_SIZE = 64
ENABLE_ROPE = False
D_rope = 64

LAYOUT_Q = "BNSD"
LAYOUT_Q_DESCALE = "BNSD"
LAYOUT_KV = "BNSD"
LAYOUT_OUT = "BNSD"

# csv precision_tolerances / absolute_precision 由 impl/golden.py 暂存
_csv_precision_tolerances = None
_csv_absolute_precision = None


# ==============================================================================
# E6M2 / FP4 E1M2 编解码
# ==============================================================================


def E6M2_bits_to_float(code: int) -> float:
    if code == 0xFF:
        return float("nan")
    exp_storage = int(code >> 2)
    m_val = code & 0x3
    exp = exp_storage - 48
    mantissa = 1.0 + m_val / 4.0
    return mantissa * (2.0**exp)


def _hifp4_split_32bit(combined: int):
    bits16_val = combined >> 16 & 0xFFFF
    bits8_val = combined >> 8 & 0xFF
    e6m2_code = combined & 0xFF
    bits8_list = [(bits8_val >> (7 - i)) & 1 for i in range(8)]
    bits16_list = [(bits16_val >> (15 - i)) & 1 for i in range(16)]
    e6m2_float = E6M2_bits_to_float(e6m2_code)
    return e6m2_float, bits8_list, bits16_list


def HiFP4_To_BF16_reverse_order(in_tensor_hifp4, GHifp4Scale_, G=64):
    GHifp4Scale = GHifp4Scale_.view(np.int32)
    a = np.array(in_tensor_hifp4)
    Mi = a.shape[0]
    Ni = a.shape[1]
    Mcnt = np.ceil(Mi / G).astype(int)
    GE6M2 = np.zeros((Mcnt, Ni))
    GE1_8 = np.zeros((Mcnt * 8, Ni))
    GE1_16 = np.zeros((Mcnt * 16, Ni))

    for i in range(Mcnt):
        for j in range(Ni):
            e6m2, bits8, bits16 = _hifp4_split_32bit(GHifp4Scale[i][j])
            GE6M2[i, j] = e6m2
            GE1_8[i * 8 : i * 8 + 8, j] = bits8[::-1]
            GE1_16[i * 16 : i * 16 + 16, j] = bits16[::-1]
    x_pf16 = np.zeros((Mi, Ni))
    for i in range(Mcnt):
        for j in range(Ni):
            x = a[i * G : min((i + 1) * G, Mi), j]
            E6M2 = GE6M2[i, j]
            E1_8 = GE1_8[i * 8 : (i + 1) * 8, j]
            DE64_8 = np.repeat(E1_8, 8)
            E1_16 = GE1_16[i * 16 : (i + 1) * 16, j]
            DE64_16 = np.repeat(E1_16, 4)
            de64 = DE64_8 + DE64_16
            grp = E6M2 * (2.0 ** de64[: len(x)]) * x
            x_pf16[i * G : min((i + 1) * G, Mi), j] = grp
    x_pf16 = x_pf16.astype(np_bfloat16)

    return x_pf16


def _hifp4_float_to_hex(f):
    return hex(struct.unpack("<I", struct.pack("<f", f))[0])


def _hifp4_IsRoundOne(sign, man, truncLen):
    roundingTruncLen = 64
    if truncLen >= roundingTruncLen:
        mask0 = 0
    else:
        mask0 = 0x1 << truncLen
    if truncLen > roundingTruncLen:
        mask1 = 0
    else:
        mask1 = 0x1 << (truncLen - 1)
    mask2 = mask1 - 1
    lastBit = (man & mask0) > 0
    truncHighBit = (man & mask1) > 0
    truncLeft = (man & mask2) > 0
    return truncHighBit and (truncLeft or lastBit)


def _hifp4_cvt_bfloat16_to_fp4_e1m2(x):
    sRet = 0
    if x < 0.0:
        sRet = 1
    x_abs = math.fabs(x)
    x = eval(_hifp4_float_to_hex(x_abs))
    x = x >> 16
    ef = x >> 7 & 0xFF
    mf = x & 0x7F
    mLenDelta = 7 - 2
    maxExp = 1
    expBias = 1
    eRet = 0
    mRet = 0
    eNorm = 0
    if ef == 0 and mf != 0:
        eNorm = ef - 127 + 1
    else:
        eNorm = ef - 127
    if (eNorm > (maxExp - expBias)) or (
        (eNorm == (maxExp - expBias)) and ((mf >> mLenDelta) == 0b11)
    ):
        return (sRet << 3) | 0b111
    elif eNorm <= -(expBias):
        eRet = 0
        mf = mf | 0x80
        mLenDelta -= eNorm + expBias - 1
        needRound = _hifp4_IsRoundOne(sRet, mf, mLenDelta)
        mRet = mf >> mLenDelta
        if needRound:
            mRet += 1
    else:
        eRet = eNorm + expBias
        needRound = _hifp4_IsRoundOne(sRet, mf, mLenDelta)
        mRet = mf >> mLenDelta
        if needRound:
            mRet += 1
        if ((mRet & 0b100) != 0) and needRound:
            eRet += 1
            mRet = 0
    if eRet >= 1:
        eRet = 1
    elif eRet == 0 and mRet == 0b100:
        eRet += 1
        mRet = 0
    return ((sRet) << 3) | ((eRet) << 2) | ((mRet) & 3)


def _hifp4_cvt_fp4_e1m2_to_bfloat16(x):
    Fp4e1m2ToBf16 = {
        "0": 0x0,
        "1": 0x3E80,
        "2": 0x3F00,
        "3": 0x3F40,
        "4": 0x3F80,
        "5": 0x3FA0,
        "6": 0x3FC0,
        "7": 0x3FE0,
        "8": 0x8000,
        "9": 0xBE80,
        "10": 0xBF00,
        "11": 0xBF40,
        "12": 0xBF80,
        "13": 0xBFA0,
        "14": 0xBFC0,
        "15": 0xBFE0,
    }
    x = int(x)
    first_fp4val = x & 0x0F
    second_fp4val = (x >> 4) & 0x0F
    return Fp4e1m2ToBf16[str(first_fp4val)], Fp4e1m2ToBf16[str(second_fp4val)]


def _hifp4_bfloat16_tensor_to_fp4_e1m2(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    new_in_tensor = in_tensor.reshape(multi_shape)
    out_tensor = np.zeros(multi_shape)
    for i in range(multi_shape):
        out_tensor[i] = _hifp4_cvt_bfloat16_to_fp4_e1m2(new_in_tensor[i])
    out_tensor = out_tensor.astype(np.uint8)
    fp4_shape = list(shape_tensor)
    fp4_shape[-1] = fp4_shape[-1] // 2
    fp4_tensor = np.zeros(multi_shape // 2).astype(np.uint8)
    for i in range(multi_shape // 2):
        fp4_tensor[i] = (out_tensor[i * 2 + 1] << 4) | out_tensor[i * 2]
    fp4_tensor = fp4_tensor.reshape(fp4_shape)
    return fp4_tensor


def _hifp4_fp4_e1m2_tensor_to_bfloat16(in_tensor):
    shape_tensor = in_tensor.shape
    multi_shape = np.prod(shape_tensor)
    in_tensor = in_tensor.reshape(multi_shape)
    bfloat16_shape = list(shape_tensor)
    bfloat16_shape[-1] = bfloat16_shape[-1] * 2
    bfloat16_tensor = np.zeros(multi_shape * 2).astype(np.uint16)
    fp32_tensor = np.zeros(multi_shape * 2).astype(np.float32)
    for i in range(multi_shape):
        bfloat16_tensor[i * 2], bfloat16_tensor[i * 2 + 1] = (
            _hifp4_cvt_fp4_e1m2_to_bfloat16(in_tensor[i])
        )
        fp32_tensor[i * 2] = struct.unpack(
            "!f", struct.pack("!I", bfloat16_tensor[i * 2].item() << 16)
        )[0]
        fp32_tensor[i * 2 + 1] = struct.unpack(
            "!f", struct.pack("!I", bfloat16_tensor[i * 2 + 1].item() << 16)
        )[0]
    fp32_tensor = fp32_tensor.reshape(bfloat16_shape)
    return fp32_tensor


def hifp4_scale_generate(input_scale, scale_range=(187, 198)):
    pertoken_scale = np.random.randint(
        scale_range[0], scale_range[1], size=input_scale.shape, dtype=np.uint32
    )
    lower_bits_pertoken_scale = pertoken_scale & 0xFF
    pertoken_scale[:] = lower_bits_pertoken_scale
    return pertoken_scale


def _hifp4_hex4_to_fp16(hex_str, big_endian=True):
    raw = bytes.fromhex(hex_str)
    fmt = ">e" if big_endian else "<e"
    return struct.unpack(fmt, raw)[0]


def _hifp4_make_accum_seq(seq_lens):
    result = []
    acc = 0
    for s in seq_lens:
        acc += s
        result.append(acc)
    return result


def _hifp4_trans_bnsd_to_bsh(tensor, shape):
    B = shape[0]
    N = shape[1]
    S = shape[2]
    D = shape[3]
    H = N * D
    return tensor.transpose(0, 2, 1, 3).reshape(B, S, H)


def _hifp4_bsh_to_bnsd(tensor, bsh_shape, headnums, inputLayout="BSH"):
    if inputLayout == "BSH":
        B = bsh_shape[0]
        S = bsh_shape[1]
        H = bsh_shape[2]
        N = headnums
        D = H // N
        return tensor.reshape(B, S, N, int(D / 2)).transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "BSND":
        B = bsh_shape[0]
        S = bsh_shape[1]
        N = bsh_shape[2]
        D = bsh_shape[3]
        return tensor.transpose(0, 2, 1, 3), [B, N, S, D]
    elif inputLayout == "TND" or inputLayout == "NTD_TND":
        actSeqLength = [bsh_shape[0]]
        B = len(actSeqLength)
        S = max(actSeqLength)
        N = bsh_shape[1]
        D = bsh_shape[2]
        if inputLayout == "NTD_TND":
            N = bsh_shape[0]
            tensor = tensor.transpose(1, 0, 2)
        new_tensor = np.zeros((B, N, S, D), dtype=tensor.dtype)
        t_start = 0
        for b_index in range(B):
            act_s = actSeqLength[b_index]
            t_end = t_start + act_s
            if act_s == 0:
                continue
            for n_index in range(N):
                new_tensor[b_index, n_index, 0:act_s, :] = tensor[
                    t_start:t_end, n_index, :
                ]
            t_start += act_s
        return new_tensor, [B, N, S, D]
    else:
        return tensor, bsh_shape


def broadcast_kv_np(num_heads, num_kv_heads, kv_tensor):
    factor = num_heads // num_kv_heads
    B, _, S, D_half = kv_tensor.shape
    result = np.zeros((B, num_heads, S, D_half), dtype=kv_tensor.dtype)
    for i in range(num_heads):
        result[:, i : i + 1, :, :] = kv_tensor[:, i // factor : i // factor + 1, :, :]
    return result


def broadcast_kv_scale_4d(num_heads, num_kv_heads, scale_4d):
    factor = num_heads // num_kv_heads
    B, N_kv, *rest = scale_4d.shape
    result = np.zeros((B, num_heads, *rest), dtype=scale_4d.dtype)
    for i in range(num_heads):
        result[:, i : i + 1, ...] = scale_4d[:, i // factor : i // factor + 1, ...]
    return result


# ==============================================================================
# CPU Golden: _np_pfaattention_act_fp4_new + cpu_hifp4_golden
# ==============================================================================


def _np_pfaattention_act_fp4_new(
    q_tensor,
    k_tensor,
    v_tensor,
    pse_tensor,
    mask_tensor,
    scalar,
    act_seq,
    act_kv_seq,
    preTokens,
    nextTokens,
    pfa_param,
    dequantScale_query_cur,
    k_antiquant_scale_cur,
    v_antiquant_scale_cur,
    dequant_scale1=None,
    dequant_scale2=None,
    quant_scale1=None,
    quant_scale2=None,
    quant_offset2=None,
    p_scale=1.0,
    out_dtype="<class 'numpy.int8'>",
):
    G = HIFP4_GROUP_SIZE
    FP8_DT = torch.float8_e4m3fn
    p_scale_val = (
        p_scale.item() if isinstance(p_scale, torch.Tensor) else float(p_scale)
    )
    ln_p_scale = (
        np.float16(math.log(p_scale_val)) if p_scale_val != 1.0 else np.float16(0.0)
    )
    EPSILON = 1e-20

    Q_BLOCK_SIZE = 128
    K_BLOCK_SIZE = 128
    V_BLOCK_SIZE = 1024
    NUM_C1V1 = 8
    C2_S2_STEP = 256

    qs_begin = 0
    qs_end = q_tensor.shape[2]
    kvs_begin = 0
    kvs_end = k_tensor.shape[2]

    q_tensor = _hifp4_fp4_e1m2_tensor_to_bfloat16(q_tensor)
    k_tensor = _hifp4_fp4_e1m2_tensor_to_bfloat16(k_tensor)
    v_tensor = _hifp4_fp4_e1m2_tensor_to_bfloat16(v_tensor)
    v_antiquant_scale_cur = v_antiquant_scale_cur.transpose(0, 1, 3, 2)

    S = kvs_end - kvs_begin
    qs = q_tensor.shape[2]
    qd = q_tensor.shape[3]
    vd = v_tensor.shape[3]

    TILES_Q = (qs + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE
    TILES_KV = (S + K_BLOCK_SIZE - 1) // K_BLOCK_SIZE
    TILES_V = (S + V_BLOCK_SIZE - 1) // V_BLOCK_SIZE

    q_tensor[0, 0, :, :] = (
        HiFP4_To_BF16_reverse_order(
            q_tensor[0, 0, :, :].copy().T, dequantScale_query_cur[0, 0, :, :].copy().T
        )
    ).T
    out = np.zeros((qs, qd), dtype=np.float32)
    lse_out = np.zeros((qs,), dtype=np.float32)

    for i in range(TILES_Q):
        Sq_start = i * Q_BLOCK_SIZE
        Sq_end = min(Sq_start + Q_BLOCK_SIZE, qs)
        Qi = q_tensor[0, 0, Sq_start:Sq_end, :].astype(np.float32)
        o_i = np.zeros((Sq_end - Sq_start, qd), dtype=np.float32)
        sum_i = np.zeros((Sq_end - Sq_start, 1), dtype=np.float32)
        m_i = np.full((Sq_end - Sq_start, 1), -np.inf, dtype=np.float16)

        for v_idx in range(TILES_V):
            v_start = v_idx * V_BLOCK_SIZE
            v_end = min(v_start + V_BLOCK_SIZE, S)

            vi = v_tensor[:, :, v_start:v_end, :]
            v_dequant_groups = math.ceil((v_end - v_start) / G)
            v_scale_start = v_start // G
            v_scale_end = v_scale_start + v_dequant_groups
            v_antiquant_i = v_antiquant_scale_cur[:, :, v_scale_start:v_scale_end, :]
            vi_dequant = HiFP4_To_BF16_reverse_order(
                vi[0, 0].copy(), v_antiquant_i[0, 0].copy()
            )
            vi_dequant = vi_dequant.astype(np.float32)

            P_drops = []
            m_blocks = []
            k_block_sizes = []
            mi = m_i.copy()

            for k in range(NUM_C1V1):
                k_idx = v_idx * NUM_C1V1 + k
                if k_idx >= TILES_KV:
                    break

                k_start = k_idx * K_BLOCK_SIZE
                k_end = min(k_start + K_BLOCK_SIZE, S)
                ki = k_tensor[:, :, k_start:k_end, :].copy()
                k_block_sizes.append(k_end - k_start)

                k_antiquant_i = k_antiquant_scale_cur[:, :, k_start:k_end, :].copy()
                ki[0, 0] = (
                    HiFP4_To_BF16_reverse_order(
                        ki[0, 0].copy().T, k_antiquant_i[0, 0].copy().T
                    )
                ).T
                S_ik = np.matmul(Qi, ki[0, 0].astype(np.float32).T)
                S_ik = S_ik * 0.08837890625
                S_ik_fp16 = S_ik.astype(np.float16)
                m_block_k = np.max(S_ik_fp16, axis=-1, keepdims=True)
                m_block_k = np.ceil(m_block_k * np.float16(1.4423828125)) * np.float16(
                    0.693359375
                )
                m_block_k = np.maximum(mi, m_block_k)

                m_block_k_p = m_block_k - ln_p_scale

                S_ik_fp16Bin = S_ik_fp16.view(np.uint16)
                m_block_k_fp16Bin = m_block_k_p.astype(np.float16).view(np.uint16)
                s1_size = Sq_end - Sq_start
                s2_size = k_end - k_start
                P_ik_raw = np.zeros((s1_size, s2_size), dtype=np.float16)
                for ii in range(s1_size):
                    for jj in range(s2_size):
                        expdif = _hifp4_valu.base_expDifV0_fp16_to_fp16(
                            _hifp4_valu.fp16_t(S_ik_fp16Bin[ii][jj]),
                            _hifp4_valu.fp16_t(m_block_k_fp16Bin[ii][0]),
                            _HIFP4_OP_INFO,
                            _HIFP4_STATUS,
                        )
                        P_ik_raw[ii][jj] = _hifp4_hex4_to_fp16(f"{expdif.bin:04x}")

                P_ik_drop = (
                    torch.from_numpy(np.ascontiguousarray(P_ik_raw))
                    .to(FP8_DT)
                    .to(torch.float32)
                    .numpy()
                )
                P_drops.append(P_ik_drop)
                m_blocks.append(m_block_k.astype(np.float32))
                mi = m_block_k

            m_final = m_blocks[-1]

            if len(P_drops) == NUM_C1V1:
                P_ij_Vj = np.zeros((Sq_end - Sq_start, vd), dtype=np.float32)
                v_offset = 0
                P_big_list = []
                for k in range(len(P_drops)):
                    v_part_size = k_block_sizes[k]
                    V_part = vi_dequant[v_offset : v_offset + v_part_size, :]
                    v_offset += v_part_size
                    m_blocksBin = m_blocks[k].astype(np.float16).view(np.uint16)
                    m_finalBin = m_final.astype(np.float16).view(np.uint16)
                    rescale_k = np.zeros((Sq_end - Sq_start, 1), dtype=np.float16)
                    for ii in range(m_blocksBin.shape[0]):
                        m_expdif = _hifp4_valu.base_expDifV0_fp16_to_fp16(
                            _hifp4_valu.fp16_t(int(m_blocksBin[ii])),
                            _hifp4_valu.fp16_t(int(m_finalBin[ii])),
                            _HIFP4_OP_INFO,
                            _HIFP4_STATUS,
                        )
                        bf16_r = _hifp4_valu.convertFloat2Float_fp16_to_bf16(
                            m_expdif, _HIFP4_OP_INFO, _HIFP4_STATUS
                        )
                        e8m0_r = _hifp4_valu.convertFloat2Float_bf16_to_e8m0(
                            bf16_r, _HIFP4_OP_INFO_CAST, _HIFP4_STATUS
                        )
                        bf16_back = _hifp4_valu.convertFloat2Float_e8m0_to_bf16(
                            e8m0_r, _HIFP4_OP_INFO, _HIFP4_STATUS
                        )
                        rescale_k[ii] = ctypes.c_float.from_buffer(
                            ctypes.c_uint32(bf16_back.bin << 16)
                        ).value
                    rescale_k = rescale_k.astype(np.float32)
                    P_big_list.append(P_drops[k] * rescale_k)
                    P_ij_Vj += np.matmul(P_drops[k] * rescale_k, V_part)
                P_big = np.concatenate(P_big_list, axis=-1)
                s1_size = Sq_end - Sq_start
                m_iBin = m_i.astype(np.float16).view(np.uint16)
                m_finalBin = m_final.astype(np.float16).view(np.uint16)
                rescale_old = np.zeros((s1_size, 1), dtype=np.float16)
                for ii in range(s1_size):
                    expdif = _hifp4_valu.base_expDifV0_fp16_to_fp16(
                        _hifp4_valu.fp16_t(m_iBin[ii][0]),
                        _hifp4_valu.fp16_t(m_finalBin[ii][0]),
                        _HIFP4_OP_INFO,
                        _HIFP4_STATUS,
                    )
                    rescale_old[ii][0] = _hifp4_hex4_to_fp16(f"{expdif.bin:04x}")
                rescale_old = rescale_old.astype(np.float32)
                sum_i = rescale_old * sum_i
                sum_i += np.sum(P_big, axis=-1, keepdims=True)
                o_i = rescale_old * o_i + P_ij_Vj
                m_i = m_final.copy()
            else:
                P_ij_Vj = np.zeros((Sq_end - Sq_start, vd), dtype=np.float32)
                v_offset = 0
                P_big_list = []
                for k in range(len(P_drops)):
                    v_part_size = k_block_sizes[k]
                    V_part = vi_dequant[v_offset : v_offset + v_part_size, :]
                    v_offset += v_part_size
                    m_blocksBin = m_blocks[k].astype(np.float16).view(np.uint16)
                    m_finalBin = m_final.astype(np.float16).view(np.uint16)
                    rescale_k = np.zeros((Sq_end - Sq_start, 1), dtype=np.float16)
                    for ii in range(m_blocksBin.shape[0]):
                        m_expdif = _hifp4_valu.base_expDifV0_fp16_to_fp16(
                            _hifp4_valu.fp16_t(int(m_blocksBin[ii])),
                            _hifp4_valu.fp16_t(int(m_finalBin[ii])),
                            _HIFP4_OP_INFO,
                            _HIFP4_STATUS,
                        )
                        bf16_r = _hifp4_valu.convertFloat2Float_fp16_to_bf16(
                            m_expdif, _HIFP4_OP_INFO, _HIFP4_STATUS
                        )
                        e8m0_r = _hifp4_valu.convertFloat2Float_bf16_to_e8m0(
                            bf16_r, _HIFP4_OP_INFO_CAST, _HIFP4_STATUS
                        )
                        bf16_back = _hifp4_valu.convertFloat2Float_e8m0_to_bf16(
                            e8m0_r, _HIFP4_OP_INFO, _HIFP4_STATUS
                        )
                        rescale_k[ii] = ctypes.c_float.from_buffer(
                            ctypes.c_uint32(bf16_back.bin << 16)
                        ).value
                    rescale_k = rescale_k.astype(np.float32)
                    P_big_list.append(P_drops[k] * rescale_k)
                    P_ij_Vj += np.matmul(P_drops[k] * rescale_k, V_part)
                P_big = np.concatenate(P_big_list, axis=-1)
                s1_size = Sq_end - Sq_start
                m_iBin = m_i.astype(np.float16).view(np.uint16)
                m_finalBin = m_final.astype(np.float16).view(np.uint16)
                rescale_old = np.zeros((s1_size, 1), dtype=np.float16)
                for ii in range(s1_size):
                    expdif = _hifp4_valu.base_expDifV0_fp16_to_fp16(
                        _hifp4_valu.fp16_t(m_iBin[ii][0]),
                        _hifp4_valu.fp16_t(m_finalBin[ii][0]),
                        _HIFP4_OP_INFO,
                        _HIFP4_STATUS,
                    )
                    rescale_old[ii][0] = _hifp4_hex4_to_fp16(f"{expdif.bin:04x}")
                rescale_old = rescale_old.astype(np.float32)
                sum_i = rescale_old * sum_i
                sum_i += np.sum(P_big, axis=-1, keepdims=True)
                o_i = rescale_old * o_i + P_ij_Vj
                m_i = m_final.copy()

        out[Sq_start:Sq_end, :] = o_i / (sum_i)
        current_sum = sum_i.flatten()
        current_m = m_i.flatten()
        safe_sum = np.maximum(current_sum, 1e-20)
        block_lse = current_m + np.log(current_sum)
        lse_out[Sq_start:Sq_end] = block_lse

    out = out.astype(np_bfloat16)
    return out, lse_out


def cpu_hifp4_golden(
    q_fp4,
    k_fp4,
    v_fp4,
    dequant_scale_q,
    k_antiquant_scale,
    v_antiquant_scale,
    actual_seq_q,
    actual_seq_kv,
    p_scale=1.0,
):
    pfa_param = {"sparse": SPARSE_MODE, "lseflag": False}

    if INPUT_LAYOUT == "BSH":
        q_shape_bsh = [q_fp4.shape[0], q_fp4.shape[1], N_q * D]
        k_shape_bsh = [k_fp4.shape[0], k_fp4.shape[1], N_kv * D]
        v_shape_bsh = [v_fp4.shape[0], v_fp4.shape[1], N_kv * D]
        q_fp4, _ = _hifp4_bsh_to_bnsd(q_fp4, q_shape_bsh, N_q, INPUT_LAYOUT)
        k_fp4, _ = _hifp4_bsh_to_bnsd(k_fp4, k_shape_bsh, N_kv, INPUT_LAYOUT)
        v_fp4, _ = _hifp4_bsh_to_bnsd(v_fp4, v_shape_bsh, N_kv, INPUT_LAYOUT)

    B_, N_q_, Sq, _ = q_fp4.shape

    if N_q_ != N_kv:
        k_fp4 = broadcast_kv_np(N_q_, N_kv, k_fp4)
        v_fp4 = broadcast_kv_np(N_q_, N_kv, v_fp4)
        k_antiquant_scale = broadcast_kv_scale_4d(N_q_, N_kv, k_antiquant_scale)
        v_antiquant_scale = broadcast_kv_scale_4d(N_q_, N_kv, v_antiquant_scale)

    out = np.zeros((B_, N_q_, Sq, D), dtype=np_bfloat16)
    o_lse = np.zeros((B_, N_q_, Sq), dtype=np.float32)
    for b in range(B_):
        for n in range(N_q_):
            q_cur = q_fp4[b : b + 1, n : n + 1, :, :]
            k_cur = k_fp4[b : b + 1, n : n + 1, :, :]
            v_cur = v_fp4[b : b + 1, n : n + 1, :, :]
            dq_q = dequant_scale_q[b : b + 1, n : n + 1, :, :]
            dq_k = k_antiquant_scale[b : b + 1, n : n + 1, :, :]
            dq_v = v_antiquant_scale[b : b + 1, n : n + 1, :, :]

            o, lse = _np_pfaattention_act_fp4_new(
                q_cur,
                k_cur,
                v_cur,
                None,
                None,
                SOFTMAX_SCALE,
                actual_seq_q,
                actual_seq_kv,
                0,
                0,
                pfa_param,
                dq_q,
                dq_k,
                dq_v,
                None,
                None,
                None,
                None,
                None,
                p_scale=p_scale,
                out_dtype=str(np.int8),
            )
            out[b, n, :, :] = o[:, :]
            o_lse[b, n, :] = lse[:]

    if INPUT_LAYOUT == "BSH":
        out = _hifp4_trans_bnsd_to_bsh(out, [B_, N_q_, Sq, D])

    out_uint16 = np.ascontiguousarray(out).view(np.uint16)
    return torch.from_numpy(out_uint16).view(torch.bfloat16), o_lse


def _pack_hifloat4_to_uint8(arr):
    """hifloat4 (unpacked, 1 byte per fp4, 4-bit code in low nibble) -> packed uint8 (2 fp4 per byte).

    输入若为 hifloat4/float4_e1m2 (en_dtypes, itemsize=1, 1 byte per element):
      shape (..., D), 每字节低 4 bit 为一个 fp4 code
    输出为 packed uint8:
      shape (..., D//2), 每字节高 4 bit = 奇数 index code, 低 4 bit = 偶数 index code
    输入若已是 uint8 (packed) 则原样返回 (向后兼容)。
    """
    arr = np.ascontiguousarray(arr)
    dtype_str = str(arr.dtype)
    if "float4" not in dtype_str and "hifloat4" not in dtype_str:
        return arr
    src_u8 = arr.view(np.uint8)
    packed = ((src_u8[..., 1::2] & 0x0F) << 4) | (src_u8[..., 0::2] & 0x0F)
    return packed.astype(np.uint8)
