/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file compressor_grad_common_arch22.h
 * \brief CompressorGrad arch22 kernel 公共助手（cast/scatter 与 gemm 共用）
 */

#ifndef COMPRESSOR_GRAD_COMMON_ARCH22_H
#define COMPRESSOR_GRAD_COMMON_ARCH22_H

#include "kernel_operator.h"
// MIX kernel 的生成包装层会调用 matmul::clearWorkspace（kfc），与 compressor arch22 同样引入
#include "lib/matmul_intf.h"

using namespace AscendC;

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (b == 0U) ? 0U : (a + b - 1U) / b;
}

__aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
{
    return a < b ? a : b;
}

__aicore__ inline uint64_t MinU64(uint64_t a, uint64_t b)
{
    return a < b ? a : b;
}

// Set-then-wait on an intra-core pipe event: blocks until all prior
// reads/writes of the source pipe are complete (TPipe-managed event id).
template <AscendC::HardEvent evt>
__aicore__ inline void SetWaitFlag()
{
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<evt>(eventId);
    AscendC::WaitFlag<evt>(eventId);
}

// =========================================================================
// bf16-io 全域支持：全局 2 的幂缩放
//   bf16 是合法输入 dtype，其数值域（±3.4e38）远大于 fp16（±65504）。dn /
//   x / wkv / wgate 工作区沿用 fp16 单平面（11-bit，与既有 fp16 落盘精度一致、
//   GEMM 单平面单 mmad），但当输入/中间量可能超出 fp16 上限时，按全局指数
//   2^-a（dn）/ 2^-b（x/w）缩放后落盘；GEMM 的 A/B 同时带 2^-a、2^-b，乘积
//   2^-(a+b) 于输出端乘 2^(a+b) 还原。2 的幂缩放无舍入误差；普通数据
//   a = b = 0 时全部路径与原实现逐位一致（缩放 Muls 直接跳过）。
//   指数来自 Phase(-1) 对六个输入张量的最大绝对值探测（host 侧不可知数据域）：
//     b = ceil(log2(max(|x|,|wkv|,|wgate|))) - 15
//     a = ceil(log2(max|sm|)) + ceil(log2(max|dC|)) + ceil(log2(max|kv|))
//         + ceil(log2(1 + N*max|sm|)) - 15     // |kv - w| <= max|kv|*(1 + N*max|sm|)
//   均钳到 [0, 120]（超范围输入本就不可由 fp16 承载，钳位只保有限性）。
// =========================================================================
constexpr int32_t CG_SCALE_TARGET_EXP = 15; // 缩放后最大值落在 2^15 以内（fp16 上限 2^16）
constexpr int32_t CG_SCALE_MAX_EXP = 120;   // 缩放指数上限（2^-120 仍为规格化 fp32）
constexpr uint32_t CG_SCALE_SLOTS = 6;      // 每核探测槽：x/wkv/wgate/dC/sm/kv
constexpr uint32_t CG_SCALE_STRIDE = 8;     // 每核槽位元素数（32B 对齐，供 DataCopyPad）
constexpr uint32_t CG_PROBE_VEC = 8192;     // 探测分块元素数（≥32B 对齐，队列深度 2）

__aicore__ inline uint32_t CgFloatBits(float x)
{
    union {
        float f;
        uint32_t u;
    } v;
    v.f = x;
    return v.u;
}

// 满足 2^k >= |x| 的最小整数 k（x <= 0 / NaN -> -1000；Inf -> 1000）
__aicore__ inline int32_t CgCeilExp2(float x)
{
    if (!(x > 0.0f)) {
        return -1000;
    }
    const uint32_t bits = CgFloatBits(x);
    const int32_t e = static_cast<int32_t>((bits >> 23) & 0xFFU);
    if (e == 0xFF) {
        return 1000;
    }
    if (e == 0) {
        return -125; // 次正规
    }
    return e - 127 + (((bits & 0x7FFFFFU) != 0U) ? 1 : 0);
}

// 2^k 的 fp32 常量（k 钳到 [-126, 127]）
__aicore__ inline float CgExp2F(int32_t k)
{
    k = (k < -126) ? -126 : ((k > 127) ? 127 : k);
    union {
        float f;
        uint32_t u;
    } v;
    v.u = static_cast<uint32_t>(127 + k) << 23;
    return v.f;
}

__aicore__ inline int32_t CgClampExp(int64_t e)
{
    if (e < 0) {
        return 0;
    }
    return (e > CG_SCALE_MAX_EXP) ? CG_SCALE_MAX_EXP : static_cast<int32_t>(e);
}

// f32 -> ioT cast 的 round mode。arch22(910_93) vconv 对 float->bfloat16_t 仅支持
// RINT/FLOOR/CEIL/ROUND/TRUNC（不支持 CAST_NONE）；RINT 为就近偶数舍入，与
// fp16 路径 CAST_NONE 的语义一致。
template <typename T>
__aicore__ inline constexpr RoundMode CastF32ToIoRoundMode()
{
    return std::is_same_v<T, bfloat16_t> ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
}

// 从探测 partials（每核 CG_SCALE_SLOTS 槽）归并全局最大值并推导缩放指数。
// 所有核读同一数组、同一归并顺序，结果一致；buf 需 >= cores*CG_SCALE_SLOTS 个 f32。
__aicore__ inline void CgProbeScales(const GlobalTensor<float>& part, uint32_t cores, const LocalTensor<float>& buf,
                                     uint32_t nRows, float& scaleDn, float& scaleXw, int32_t& expA, int32_t& expB)
{
    const uint32_t n = cores * CG_SCALE_STRIDE;
    DataCopyExtParams cp{1, n * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
    DataCopyPad(buf, part, cp, pp);
    SetWaitFlag<HardEvent::MTE2_S>();
    float mx = 0.0f;
    float mw = 0.0f;
    float mg = 0.0f;
    float md = 0.0f;
    float ms = 0.0f;
    float mk = 0.0f;
    for (uint32_t i = 0; i < cores; i++) {
        const uint32_t b = i * CG_SCALE_STRIDE;
        const float vx = buf.GetValue(b);
        const float vw = buf.GetValue(b + 1);
        const float vg = buf.GetValue(b + 2);
        const float vd = buf.GetValue(b + 3);
        const float vs = buf.GetValue(b + 4);
        const float vk = buf.GetValue(b + 5);
        mx = (vx > mx) ? vx : mx;
        mw = (vw > mw) ? vw : mw;
        mg = (vg > mg) ? vg : mg;
        md = (vd > md) ? vd : md;
        ms = (vs > ms) ? vs : ms;
        mk = (vk > mk) ? vk : mk;
    }
    const float maxXW = (mx > mw) ? ((mx > mg) ? mx : mg) : ((mw > mg) ? mw : mg);
    expB = CgClampExp(static_cast<int64_t>(CgCeilExp2(maxXW)) - CG_SCALE_TARGET_EXP);
    const int64_t expW = static_cast<int64_t>(CgCeilExp2(ms)) + CgCeilExp2(md) + CgCeilExp2(mk) +
                         CgCeilExp2(1.0f + static_cast<float>(static_cast<int32_t>(nRows)) * ms) - CG_SCALE_TARGET_EXP;
    expA = CgClampExp(expW);
    scaleDn = CgExp2F(-expA);
    scaleXw = CgExp2F(-expB);
}

// 归并 per-core 的 x/w 最大绝对值槽（每核第 0 槽），推导 b 指数。
// 所有 32B 读一次性下发（各轮落到 buf 的不同 32B 区），一次 MTE2_S 后标量归并；
// buf 需 >= cores 个 f32。
__aicore__ inline int32_t CgReduceXwMax(const GlobalTensor<float>& part, uint32_t cores, const LocalTensor<float>& buf)
{
    const uint32_t rounds = (cores + CG_SCALE_STRIDE - 1U) / CG_SCALE_STRIDE;
    for (uint32_t r = 0; r < rounds; r++) {
        const uint32_t i = r * CG_SCALE_STRIDE;
        const uint32_t n = MinU32(cores - i, CG_SCALE_STRIDE);
        DataCopyExtParams cp{1, n * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
        DataCopyPad(buf[r * CG_SCALE_STRIDE], part[i * CG_SCALE_STRIDE], cp, pp);
    }
    SetWaitFlag<HardEvent::MTE2_S>();
    float m = 0.0f;
    for (uint32_t k = 0; k < cores; k++) {
        const float v = buf.GetValue(k);
        m = (v > m) ? v : m;
    }
    return CgClampExp(static_cast<int64_t>(CgCeilExp2(m)) - CG_SCALE_TARGET_EXP);
}

// 发布槽元素偏移：位于 per-core partials 之后，8 元素（32B）对齐
__aicore__ inline uint32_t CgScalePubOff(uint32_t cores)
{
    return ((cores * CG_SCALE_STRIDE + 7U) & ~7U);
}

// cast 相位 core0：把最终倍率写到发布槽（UB->GM 数据通路；标量 GM 写在跨核
// 读侧不可见，必须走 MTE3/MTE2）
// [+0]=scaleDn, [+1]=scaleXw, [+2]=scaleOut(2^expTotal), [+3]=expTotal
__aicore__ inline void CgPublishScales(GlobalTensor<float>& part, uint32_t cores, const LocalTensor<float>& buf,
                                       float scaleDn, float scaleXw, int32_t expTotal)
{
    Duplicate(buf, 0.0f, static_cast<int32_t>(CG_SCALE_STRIDE));
    buf.SetValue(0, scaleDn);
    buf.SetValue(1, scaleXw);
    buf.SetValue(2, CgExp2F(expTotal));
    buf.SetValue(3, static_cast<float>(static_cast<int32_t>(expTotal)));
    SetWaitFlag<HardEvent::V_MTE3>();
    DataCopyExtParams cp{1, CG_SCALE_STRIDE * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPad(part[CgScalePubOff(cores)], buf, cp);
    SetWaitFlag<HardEvent::MTE3_V>();
}

// 读取发布槽（一次 32B MTE2）；buf 需 >= CG_SCALE_STRIDE 个 f32
__aicore__ inline void CgReadPubScales(const GlobalTensor<float>& part, uint32_t cores, const LocalTensor<float>& buf,
                                       float& scaleDn, float& scaleXw, float& scaleOut, int32_t& expTotal)
{
    SetWaitFlag<HardEvent::V_MTE2>();
    DataCopyExtParams cp{1, CG_SCALE_STRIDE * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> pp{false, 0, 0, 0.0f};
    DataCopyPad(buf, part[CgScalePubOff(cores)], cp, pp);
    SetWaitFlag<HardEvent::MTE2_S>();
    scaleDn = buf.GetValue(0);
    scaleXw = buf.GetValue(1);
    scaleOut = buf.GetValue(2);
    expTotal = static_cast<int32_t>(buf.GetValue(3));
}

#endif // COMPRESSOR_GRAD_COMMON_ARCH22_H
