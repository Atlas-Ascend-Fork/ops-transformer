#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Golden reference for KvCompressEpilog (KV Cache 压缩写入 epilogue).

对 bf16 激活 x（任意 N-D，末维 d；末 64 列为 rope 段、前 d-64 列为 nope 段）做
量化压缩，按 slot_mapping 散写（scatter）到 uint8 cache（4D
[blockNum, blockSize, 1, headDim]，num_slots = blockNum×blockSize）各行的前
kvCacheCol 字节；slot == -1 的 token 跳过；行内 [kvCacheCol, headDim) 与未被
引用的行保持 cache 原值。输出为 cache 的副本（输入不可变）。

quant_mode（int，缺省 1，非 0/1/2 抛 NotImplementedError）决定编码格式：
  0  nope 逐 64 元素组动态量化为 FP8 E4M3，每组 scale 以 bf16（2B 小端）存储；
     rope 段以 bf16 原始位（128B）原样写入。
  1  同 0，但每组 scale 以 E8M0 指数（1B = f32 偏置指数，bias 127）存储。
  2  rope 段乘 x_scale 后量化为 hifloat8（64B，NaN→0、clip ±32768 含端点）；
     nope 按 quant_group_size 元素一组量化为 FLOAT4_E2M1（2 nibble 打包 1 字节，
     偶数下标低半字节），每组 scale 以 bf16（2B 小端）存储。

mode0/1 量化链（逐 64 元素组，组大小硬编码、quant_group_size 被忽略）：
  amax = max|grp|（f32）；s = max(amax, 1e-4) × f32(1/448)；
  round_scale 为真值判断（非 0 一律取整）：scale = 2^⌈log2(s)⌉（f32 位运算：
  偏置指数 + 尾数非零进位），否则 scale = s 原值；mode1 的 E8M0 字节一律取
  （可能已取整的）scale 的偏置指数（rs=0 时为 ⌊log2(s)⌋+127，与数据所用 scale
  不自洽——忠实复现 kernel 语义）；q = grp/scale（f32 除法）→ clip ±448 →
  e4m3fn（RNE）→ 字节 0xFF 改写 0x7F；mode0 scale 为 NaN 时写 0x7FFF 特例
  （仅 rs=0+NaN 可达）。

mode2 量化链（逐 quant_group_size 元素组）：
  scale = bf16(amax × f32(1/6))；amax==0 时 inv = bf16(0) 否则 bf16(6/amax)；
  xq = bf16(grp × inv)（f32 积精确、单次舍入）→ f32 clip ±6 → fp4（RNE）；
  amax=+Inf → nibble 整组覆盖 0、scale 保持 bf16(+Inf)=0x7F80；
  amax=NaN → 仅 scale 覆盖 0x7FFF、nibble 保持 fp4(NaN) 转换结果。

kvCacheCol（headDim 须 ≥ kvCacheCol，否则 ValueError）：
  mode0: (d-64) + 128 + ((d-64)//64)×2，向上对齐 32（填充字节写 0）
  mode1: (d-64) + 128 + ((d-64)//64)×1，不对齐（特例）
  mode2: 64 + (d-64)//2 + ((d-64)//quant_group_size)×2，向上对齐 32（写 0）

精度：x 为 bf16（torch bf16 走 uint16 位视图保位；非 bf16 先 astype(f32) 再
astype(bf16) 数值转换）；mode0/1 全程 f32；mode2 scale/inv 各自 f32 计算后单次
舍入到 bf16。输出 dtype/shape 与输入 cache 相同（uint8）。面向字节级对比
（tolerance binary_equal）。

加载（于 ops-test-kit 仓库目录下执行，插件路径按仓库相对位置调整）：

    python3 -m ttk kernel -i ttk_kernel_kv_compress_epilog_st.csv \
        --plugin tests/assets/golden.py
"""

__spec__ = {
    "kv_compress_epilog": "KvCompressEpilogTestSpec",
}

import numpy as np

# 第三方 dtype 句柄懒加载（ml_dtypes 必需；en_dtypes 仅 mode2 需要），保证仅导入
# 本文件运行其它算子 golden 时不强制依赖。
BF16 = None
F8E4M3 = None
FP4E2M1 = None
HIF8 = None
HAS_EN = False
_DTYPES_READY = False

ROPE_SIZE = 64  # rope 段元素数（x 末 64 列）
ROPE_BYTES = 128  # mode0/1 rope 段字节数（64 × bf16）
GROUP = 64  # mode0/1 量化组大小（硬编码，与 quant_group_size 无关）
FP8_E4M3_MAX = np.float32(448.0)  # e4m3fn 最大有限值
COEFF = np.float32(1.0) / np.float32(448.0)  # 与 kernel 的 1.0f/448.0f 一致
EPS = np.float32(1e-4)  # amax 下限
HIF8_MAX = np.float32(32768.0)  # hifloat8 最大幅值
FP4_E2M1_MAX = np.float32(6.0)  # FLOAT4_E2M1 最大可表示幅值
INV6 = np.float32(1.0) / np.float32(6.0)  # 与 kernel 的 1.0f/6.0f 一致


def _ensure_dtypes():
    """首次调用 golden 时导入 ml_dtypes（必需）与 en_dtypes（mode2 的 fp4/hif8）。"""
    global BF16, F8E4M3, FP4E2M1, HIF8, HAS_EN, _DTYPES_READY
    if _DTYPES_READY:
        return
    try:
        import ml_dtypes
    except ImportError as e:
        raise RuntimeError(
            "ml_dtypes is needed for the kv_compress_epilog golden. "
            "Please install with `pip3 install ml-dtypes`"
        ) from e
    BF16 = ml_dtypes.bfloat16
    F8E4M3 = ml_dtypes.float8_e4m3fn
    try:
        import en_dtypes

        FP4E2M1 = en_dtypes.float4_e2m1
        HIF8 = en_dtypes.hifloat8
        HAS_EN = True
    except ImportError:
        HAS_EN = False
    _DTYPES_READY = True


def _to_numpy(t):
    """ndarray/torch 张量统一为连续 numpy；torch bf16 走 uint16 位视图保位转换。"""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and isinstance(t, torch.Tensor):
        tc = t.detach().cpu().contiguous()
        if tc.dtype == torch.bfloat16:
            return tc.view(torch.uint16).numpy().view(BF16)
        return tc.numpy()
    return np.ascontiguousarray(t)


def _f32_bits(s):
    return np.array([s], dtype=np.float32).view(np.uint32)[0]


def _round_scale_pow2(s):
    """复现 kernel roundScale 分支的位运算：scale = 2^⌈log2(s)⌉（s>0 正常数）。

    bits = f32(s).view(u32)；尾数非零 → 偏置指数进位；scale = expo<<23。
    返回 (scale, expo)。s 为 NaN 时返回 (-0.0, 256)（expo<<23 溢出至符号位），
    与 kernel 位运算结果一致。
    """
    bits = _f32_bits(s)
    mant = int(bits & np.uint32(0x7FFFFF))
    expo = int(bits >> np.uint32(23))
    if mant != 0:
        expo += 1
    scale = np.array([expo << 23], dtype=np.uint32).view(np.float32)[0]
    return scale, expo


def _kv_cache_col(d, quant_mode, quant_group_size):
    """每行写出的字节数（与 host tiling DoOpTiling 一致）。"""
    quant_col = d - ROPE_SIZE
    if quant_mode == 0:
        concat = quant_col + ROPE_BYTES + (quant_col // GROUP) * 2
        return (concat + 31) // 32 * 32
    if quant_mode == 1:
        return quant_col + ROPE_BYTES + (quant_col // GROUP) * 1
    # mode2: 组大小来自 quant_group_size
    concat = ROPE_SIZE + quant_col // 2 + (quant_col // quant_group_size) * 2
    return (concat + 31) // 32 * 32


def _encode_row(x_bf16_row, x_f32_row, quant_mode, round_scale):
    """mode0/1 行编码：[rope bf16 位 128B][nope fp8][scale][pad 0]。"""
    d = x_f32_row.shape[0]
    quant_col = d - ROPE_SIZE
    scale_col = quant_col // GROUP
    scale_bytes = 2 if quant_mode == 0 else 1
    concat = quant_col + ROPE_BYTES + scale_col * scale_bytes
    if quant_mode == 1:
        kv_col = concat  # mode1 不做 32B 对齐（特例）
    else:
        kv_col = (concat + 31) // 32 * 32
    row = np.zeros(kv_col, dtype=np.uint8)

    # rope 段：bf16 原始位拷贝（NaN/Inf/-0 位型原样保留，小端）
    row[:ROPE_BYTES] = (
        np.ascontiguousarray(x_bf16_row[quant_col:]).view(np.uint16).view(np.uint8)
    )

    for g in range(scale_col):
        grp = x_f32_row[g * GROUP : (g + 1) * GROUP]
        amax = np.max(np.abs(grp))
        s = np.maximum(amax, EPS) * COEFF
        if round_scale:  # 真值判断：非 0 一律走向上取整到 2 的幂
            scale, expo = _round_scale_pow2(s)
        else:
            scale = s
            expo = int(_f32_bits(s) >> np.uint32(23))
        q = grp / scale
        q = np.clip(q, -FP8_E4M3_MAX, FP8_E4M3_MAX)
        data = q.astype(F8E4M3).view(np.uint8).copy()
        data[data == 0xFF] = 0x7F  # e4m3fn -NaN 统一改写 +NaN
        row[ROPE_BYTES + g * GROUP : ROPE_BYTES + (g + 1) * GROUP] = data

        off = ROPE_BYTES + quant_col + g * scale_bytes
        if quant_mode == 1:
            row[off] = np.uint8(expo)  # E8M0：scale 的 f32 偏置指数（bias 127）
        elif np.isnan(scale):
            row[off : off + 2] = (0xFF, 0x7F)  # NaN scale 特例 0x7FFF（仅 rs=0 可达）
        else:
            row[off : off + 2] = (
                np.array([scale], dtype=np.float32)
                .astype(BF16)
                .view(np.uint16)
                .view(np.uint8)
            )
    return row


def _encode_row_mode2(x_bf16_row, x_f32_row, group_size, x_scale):
    """mode2 行编码：[rope hifloat8 64B][nope fp4 打包][scale bf16][pad 0]。"""
    if not HAS_EN:
        raise RuntimeError(
            "kv_compress_epilog mode2 golden needs en_dtypes (float4_e2m1/hifloat8)"
        )
    d = x_f32_row.shape[0]
    quant_col = d - ROPE_SIZE
    n_group = quant_col // group_size
    concat = ROPE_SIZE + quant_col // 2 + n_group * 2
    kv_col = (concat + 31) // 32 * 32
    row = np.zeros(kv_col, dtype=np.uint8)

    # rope 段：y = f32(x_rope) × f32(x_scale)；NaN→0；clip ±32768（含端点）→ hif8
    y = x_f32_row[quant_col:] * np.float32(x_scale)
    y = np.where(np.isnan(y), np.float32(0.0), y)
    y = np.clip(y, -HIF8_MAX, HIF8_MAX)
    row[:ROPE_SIZE] = y.astype(HIF8).view(np.uint8)

    fp4_off = ROPE_SIZE
    scale_off = ROPE_SIZE + quant_col // 2
    for g in range(n_group):
        grp_f32 = x_f32_row[g * group_size : (g + 1) * group_size]
        amax = np.max(np.abs(grp_f32))
        scale_bf16 = np.array([amax * INV6], dtype=np.float32).astype(BF16)
        if amax == 0:
            inv = np.zeros(1, dtype=np.float32).astype(BF16)
        else:
            inv = np.array([np.float32(6.0) / amax], dtype=np.float32).astype(BF16)
        # xq = bf16 单次舍入乘积（bf16×bf16 的 f32 积精确 → 一次舍入 = 正确舍入积）
        xq = (grp_f32 * inv.astype(np.float32)).astype(BF16)
        xq = np.clip(xq.astype(np.float32), -FP4_E2M1_MAX, FP4_E2M1_MAX)
        nib = xq.astype(FP4E2M1).view(np.uint8).reshape(-1) & np.uint8(0x0F)
        if np.isinf(amax) and amax > 0:
            nib[:] = 0  # amax=+Inf：nibble 整组覆盖为 FP4E2M1(0)，scale 保持 bf16(+Inf)
        packed = nib[0::2] | (nib[1::2] << np.uint8(4))  # 偶数下标低半字节
        row[fp4_off + g * (group_size // 2) : fp4_off + (g + 1) * (group_size // 2)] = (
            packed
        )
        off = scale_off + g * 2
        if np.isnan(amax):
            row[off : off + 2] = (0xFF, 0x7F)  # amax=NaN：仅 scale 覆盖 0x7FFF
        else:
            row[off : off + 2] = scale_bf16.view(np.uint16).view(np.uint8)
    return row


def _kv_compress_epilog_core(
    cache,
    x,
    slot_mapping,
    quant_group_size=64,
    quant_mode=1,
    round_scale=1,
    x_scale=1.0,
):
    quant_mode = int(quant_mode)
    if quant_mode not in (0, 1, 2):
        raise NotImplementedError(
            "kv_compress_epilog golden: quant_mode should be 0/1/2, got %d" % quant_mode
        )
    out = _to_numpy(cache).copy()
    head_dim = out.shape[-1]
    out2d = out.reshape(-1, head_dim)

    x_bf16 = _to_numpy(x)
    if x_bf16.dtype != BF16:
        x_bf16 = x_bf16.astype(np.float32).astype(BF16)
    x_bf16 = x_bf16.reshape(-1, x_bf16.shape[-1])  # [bs, d]
    x_f32 = x_bf16.astype(np.float32)  # 精确
    bs, d = x_f32.shape
    sm = _to_numpy(slot_mapping).reshape(-1).astype(np.int64)

    kv_cache_col = _kv_cache_col(d, quant_mode, int(quant_group_size))
    if kv_cache_col > head_dim:
        raise ValueError(
            "kv_compress_epilog golden: cache headDim(%d) must be >= kvCacheCol(%d)"
            % (head_dim, kv_cache_col)
        )

    for i in range(bs):
        slot = int(sm[i])
        if slot == -1:  # 仅 -1 跳过该 token
            continue
        if quant_mode == 2:
            enc = _encode_row_mode2(
                x_bf16[i], x_f32[i], int(quant_group_size), float(x_scale)
            )
        else:
            enc = _encode_row(x_bf16[i], x_f32[i], quant_mode, round_scale)
        out2d[slot, :kv_cache_col] = enc
    return np.ascontiguousarray(out)


class KvCompressEpilogTestSpec:
    """KvCompressEpilog 测试规范（kernel/geir 流程 numpy 入参，aclnn 亦可复用）。

    golden 参数：位置输入 cache, x, slot_mapping（顺序同 def.cpp），关键字属性
    quant_group_size/quant_mode/round_scale/x_scale（缺省同算子声明 64/1/True/1.0）。
    """

    @staticmethod
    def golden(
        cache,
        x,
        slot_mapping,
        quant_group_size=64,
        quant_mode=1,
        round_scale=True,
        x_scale=1.0,
        **kwargs,
    ):
        # kernel/geir 传 snake_case 属性；aclnn 跟随 C 接口命名
        quant_group_size = kwargs.pop("quantGroupSize", quant_group_size)
        quant_mode = kwargs.pop("quantMode", quant_mode)
        round_scale = kwargs.pop("roundScale", round_scale)
        x_scale = kwargs.pop("xScale", x_scale)
        _ensure_dtypes()
        out = _kv_compress_epilog_core(
            cache,
            x,
            slot_mapping,
            quant_group_size=quant_group_size,
            quant_mode=quant_mode,
            round_scale=round_scale,
            x_scale=x_scale,
        )
        return [np.ascontiguousarray(out)]

    @staticmethod
    def customize_inputs(cache, x, slot_mapping, **kwargs):
        """slot_mapping 取值约束强（-1 或 [0, num_slots)、不应重复），随机数据无法
        满足：改为确定性生成——均匀散布的唯一 slot 覆盖各 cache block，并按 bs
        规模混入 -1 跳过 token（bs>=2 最后一个、bs>=10 另加中间一个）。"""
        bs = int(np.prod(tuple(x.shape[:-1])))
        num_slots = int(
            np.prod(tuple(cache.shape[:3]))
        )  # [blockNum, blockSize, 1, headDim]
        if bs > num_slots:
            raise ValueError(
                "kv_compress_epilog: bs (%d) must not exceed numSlots (%d)"
                % (bs, num_slots)
            )
        if hasattr(slot_mapping, "copy_"):
            # aclnn 路径：TTK 传 torch 张量并原地消费
            import torch

            slots = torch.arange(bs, dtype=torch.int64, device=slot_mapping.device)
            slots = (slots * num_slots // bs).to(slot_mapping.dtype)
            if bs >= 2:
                slots[-1] = -1
            if bs >= 10:
                slots[bs // 2] = -1
            slot_mapping.copy_(slots.reshape(slot_mapping.shape))
            return (cache, x, slot_mapping)

        slots = np.arange(bs, dtype=np.int64) * num_slots // bs
        if bs >= 2:
            slots[-1] = -1
        if bs >= 10:
            slots[bs // 2] = -1
        slot_mapping[...] = slots.reshape(slot_mapping.shape).astype(slot_mapping.dtype)
        return (cache, x, slot_mapping)

    tolerance = {
        "uint8": {"standard": "binary_equal"},
    }
