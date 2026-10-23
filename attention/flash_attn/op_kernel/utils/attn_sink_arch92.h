/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file attn_sink_arch92.h
 * \brief arch92 sink 特性级 common: CopyIn(GM->UB 值区布局) + 注入链寄存器直写。
 * sinks 值类型为 float32(与 q dtype 解耦); 块侧 softmax 统计恒 float(using T = float)。
 *
 * 数学目标(Scheme B, prologue 注入): maxUb[r] = inUb[idx(r)], r < m(<= 128)。
 *
 * arch92 采用 UB 基址 Gather(vgather2) + 段式双寄存器一次性直写(区别于单窗口 64 lane
 * reg Gather + A/B 双窗口宽模式):
 *   - m ∈ (0,128] 单次调用覆盖: lo 段 = [0,64), hi 段 = [64,128)(仅 MBASE==128 实例化);
 *   - 全段 CreateMask<ALL> 编译期常量 mask, VF 内无运行时标量/分支:
 *     softmax 槽长 = softmaxBufElementCount = mBaseSize(ArchInfo::CV_RATIO == 1),
 *     恰等于 lo+hi 双段合计宽度, 全槽直写不越槽;
 *   - [m, mBaseSize) 尾部行属本块未用容量: 所有消费侧(update/LSE/copyout)按
 *     actVecMSize 界定, 尾部行永不参与输出 —— 基线 ResetSoftmaxBuffer 本就全槽
 *     Duplicate, update 亦以 preg_all 全 lane 处理, 尾部 lane 内容无语义依赖;
 *   - 未用 lane 的 Gather 索引恒落值区槽内(GS1 最坏 (actS1Size-1+127)/actS1Size <= 128,
 *     rot k 恒 < gSize, gap <= headOff+127 < 槽容量 8320), 槽内读残留无害;
 *   - 索引链 Div/Muls/Sub/Adds 默认 ZEROING(未激活 lane 置零);
 *   - 无 A/B 双窗口、无 chunk 循环。
 *
 * idx(r) 依布局(值区布局由 SinkCopyInGS1/SinkCopyInS1G 决定, 直写函数只读不写):
 *   GS1(BNSD):       idx = (s0 + r) / actS1Size           —— SinkGs1DirectVF;
 *   S1G gSize==1:    恒同值, BRC 单值广播                  —— SinkBrcVF;
 *   S1G case1:       idx = r(值区连续)                     —— SinkContigDirectVF;
 *   S1G case2.1:     前 headLen 行 idx = headOff + r,
 *                    其余行 idx = r - headLen(gap 双段)    —— SinkGapDirectVF;
 *   S1G case2.2/3:   idx = (gStart + r) mod gSize(回绕)    —— SinkRotDirectVF。
 *
 * 值区安全边界(借用 ubVec1Res 槽, 33280B): GS1 gCount <= 128, contig gCount = m <= 128,
 * rot 值区 = gSize <= 128, gap span = headOff + headLen <= 136, 均远小于槽容量。
 */
#ifndef ATTN_SINK_ARCH92_H_
#define ATTN_SINK_ARCH92_H_

#include "kernel_operator.h"

using namespace AscendC;

namespace AttentionCommon {

template <typename INPUT_T>
__aicore__ inline void SinkCopyInGS1(LocalTensor<INPUT_T>& sinkTmpUb, GlobalTensor<INPUT_T>& sinkGm, uint32_t gs1Start,
                                     uint32_t actVecMSize, uint32_t actS1Size, uint32_t n2Idx, uint32_t gSize)
{
    uint32_t gStart = gs1Start / actS1Size;
    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End / actS1Size;

    uint32_t gCount = gEnd - gStart + 1;
    DataCopyExtParams sinkCopyParams;
    sinkCopyParams.blockCount = 1;
    sinkCopyParams.blockLen = gCount * sizeof(INPUT_T);
    sinkCopyParams.srcStride = 0;
    sinkCopyParams.dstStride = 0;
    DataCopyPadExtParams<INPUT_T> sinkPadParams;
    sinkPadParams.isPad = true;
    sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
    DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);
}

template <typename INPUT_T>
__aicore__ inline void SinkCopyInS1G(LocalTensor<INPUT_T>& sinkTmpUb, GlobalTensor<INPUT_T>& sinkGm, uint32_t gs1Start,
                                     uint32_t actVecMSize, uint32_t actS1Size, uint32_t n2Idx, uint32_t gSize)
{
    uint32_t gStart = gs1Start % gSize;
    uint32_t s1Start = gs1Start / gSize;

    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End % gSize;
    uint32_t s1End = gs1End / gSize;

    if (s1Start == s1End) {
        // 情况1: 只有一个头块, 此时[gStart, gEnd]之间的行就是实际处理的M轴
        uint32_t gCount = gEnd - gStart + 1;
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;
        sinkCopyParams.blockLen = gCount * sizeof(INPUT_T);
        sinkCopyParams.srcStride = 0;
        sinkCopyParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> sinkPadParams;
        sinkPadParams.isPad = true;
        sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
        DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);
    } else if (s1Start + 1 == s1End) {
        if (gStart > gEnd) {
            // 情况2.1: 头块 + 尾块, 此时头块的[gStart, gSize)和尾块的[0, gEnd]为实际处理的M轴.
            // 并且gStart > gEnd, 此时它们不重叠.
            // 由于头尾块的长度都可能不对齐, 并且gSize有可能很大, 而UB要求32B对齐,
            // 约定先拷贝尾块的GM[0, gEnd]->UB[0, Align(gEnd + 1, 8) - 1],
            // 再拷贝头块的GM[gStart, gSize)->UB[Align(gEnd + 1, 8), Align(gEnd + 1, 8) + Align(gSize - gStart,
            // 8))
            DataCopyExtParams sinkCopyParams;
            sinkCopyParams.blockCount = 1;
            sinkCopyParams.blockLen = (gEnd + 1) * sizeof(INPUT_T);
            sinkCopyParams.srcStride = 0;
            sinkCopyParams.dstStride = 0;
            DataCopyPadExtParams<INPUT_T> sinkPadParams;
            sinkPadParams.isPad = true;
            sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
            DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);

            uint32_t ubOffset = (gEnd + 1U + 7U) & ~7U; // Align(gEnd + 1, 8), 32B
            sinkCopyParams.blockLen = (gSize - gStart) * sizeof(INPUT_T);
            DataCopyPad(sinkTmpUb[ubOffset], sinkGm[n2Idx * gSize + gStart], sinkCopyParams, sinkPadParams);
        } else {
            // 情况2.2: 头块 + 尾块, 此时头块的[gStart, gSize)和尾块的[0, gEnd]为实际处理的M轴.
            // 并且gStart <= gEnd, 此时它们重叠, 合并后的覆盖范围为[0, gSize)
            DataCopyExtParams sinkCopyParams;
            sinkCopyParams.blockCount = 1;
            sinkCopyParams.blockLen = gSize * sizeof(INPUT_T);
            sinkCopyParams.srcStride = 0;
            sinkCopyParams.dstStride = 0;
            DataCopyPadExtParams<INPUT_T> sinkPadParams;
            sinkPadParams.isPad = true;
            sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
            DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);
        }
    } else {
        // 情况3: 头块 + 中间块 + 尾块, 合并后的覆盖范围为[0, gSize)
        DataCopyExtParams sinkCopyParams;
        sinkCopyParams.blockCount = 1;
        sinkCopyParams.blockLen = gSize * sizeof(INPUT_T);
        sinkCopyParams.srcStride = 0;
        sinkCopyParams.dstStride = 0;
        DataCopyPadExtParams<INPUT_T> sinkPadParams;
        sinkPadParams.isPad = true;
        sinkPadParams.paddingValue = static_cast<INPUT_T>(0);
        DataCopyPad(sinkTmpUb, sinkGm[n2Idx * gSize], sinkCopyParams, sinkPadParams);
    }
}

/*!
 * \brief gSize==1 单值广播直写: maxUb[r] = inUb[0], r < m。
 * 前置: CopyIn 已将唯一 sink 值(float)放入 inUb[0]。BRC_B32 广播到 64 lane,
 * 双段复用同寄存器。全段编译期 mask(CreateMask<ALL>)直写整槽: 广播值对任意
 * lane 语义无害(尾部行不被消费), VF 内无运行时标量。
 */
template <typename T, typename INPUT_T, uint32_t MBASE>
__simd_vf__ inline void SinkBrcVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb)
{
    MicroAPI::RegTensor<T> valReg;
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_BRC_B32>(valReg, inUb);
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(maxUb, valReg, pregAll);
    if constexpr (MBASE == 128U) {
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(maxUb + 64, valReg, pregAll);
    }
}

/*!
 * \brief 连续值段直写: maxUb[r] = inUb[r], r < m。
 * 前置: inUb[0..m) 为本块行值连续段(float, 基址 32B 对齐)。LoadAlign 全宽读,
 * [m, mBaseSize) 尾部读槽内残留, 全段编译期 mask 直写整槽(尾部行不被消费)。
 */
template <typename T, typename INPUT_T, uint32_t MBASE>
__simd_vf__ inline void SinkContigDirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb)
{
    MicroAPI::RegTensor<T> fRegLo;
    MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(fRegLo, inUb);
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(maxUb, fRegLo, pregAll);
    if constexpr (MBASE == 128U) {
        MicroAPI::RegTensor<T> fRegHi;
        MicroAPI::LoadAlign<T, MicroAPI::LoadDist::DIST_NORM>(fRegHi, inUb + 64); // 64 float = 256B 对齐
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(maxUb + 64, fRegHi, pregAll);
    }
}

/*!
 * \brief GS1(BNSD) 直写: maxUb[r] = inUb[(s0 + r) / actS1Size], r < m。
 * 前置: inUb[0..gCount) 为本块 g 值连续区(float, GS1 CopyIn 布局, gCount <= 128)。
 * 索引链 int32: Arange(s0 + segStart) -> Div(actS1Size) -> Gather(inUb)。
 * 全段编译期 mask: 未用 lane 索引最坏 (actS1Size-1+127)/actS1Size <= 128 恒落值区槽内。
 */
template <typename T, typename INPUT_T, uint32_t MBASE>
__simd_vf__ inline void SinkGs1DirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t s0, uint32_t actS1Size)
{
    MicroAPI::RegTensor<T> oReg;
    MicroAPI::RegTensor<int32_t> aReg;
    MicroAPI::RegTensor<int32_t> qReg;
    MicroAPI::RegTensor<int32_t> aszReg;
    MicroAPI::Duplicate<int32_t>(aszReg, static_cast<int32_t>(actS1Size));
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    // lo 段: 行 r ∈ [0,64), idx = (s0 + r) / actS1Size
    MicroAPI::Arange<int32_t>(aReg, static_cast<int32_t>(s0));
    MicroAPI::Div<int32_t>(qReg, aReg, aszReg, pregAll);
    MicroAPI::RegTensor<uint32_t>& qLoU32 = (MicroAPI::RegTensor<uint32_t>&)qReg;
    MicroAPI::Gather(oReg, inUb, qLoU32, pregAll);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregAll);
    if constexpr (MBASE == 128U) {
        // hi 段: 行 r ∈ [64,128), idx = (s0 + r) / actS1Size
        MicroAPI::Arange<int32_t>(aReg, static_cast<int32_t>(s0 + 64U));
        MicroAPI::Div<int32_t>(qReg, aReg, aszReg, pregAll);
        MicroAPI::RegTensor<uint32_t>& qHiU32 = (MicroAPI::RegTensor<uint32_t>&)qReg;
        MicroAPI::Gather(oReg, inUb, qHiU32, pregAll);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(maxUb + 64, oReg, pregAll);
    }
}

/*!
 * \brief 旋转(S1G case2.2/3) 直写: maxUb[r] = inUb[(gStart + r) mod gSize], r < m。
 * 前置: inUb[0..gSize) 为本 n2 完整值区(float, case2.2/3 CopyIn 布局, gSize <= 128)。
 * case2.2(gStart <= gEnd)序列不回绕, 索引恒等重排, 同样正确。
 * 取模无 Mod 指令: k = a - (a / gSize) * gSize, k 恒 ∈ [0, gSize) 全 lane 安全
 * (含未用 lane), 全段编译期 mask 直写整槽。
 */
template <typename T, typename INPUT_T, uint32_t MBASE>
__simd_vf__ inline void SinkRotDirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t gStart, uint32_t gSize)
{
    MicroAPI::RegTensor<T> oReg;
    MicroAPI::RegTensor<int32_t> aReg;
    MicroAPI::RegTensor<int32_t> qReg;
    MicroAPI::RegTensor<int32_t> kReg;
    MicroAPI::RegTensor<int32_t> gszReg;
    MicroAPI::Duplicate<int32_t>(gszReg, static_cast<int32_t>(gSize));
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    // lo 段: k = (gStart + r) mod gSize, r ∈ [0,64)
    MicroAPI::Arange<int32_t>(aReg, static_cast<int32_t>(gStart));
    MicroAPI::Div<int32_t>(qReg, aReg, gszReg, pregAll);
    MicroAPI::Muls<int32_t>(kReg, qReg, static_cast<int32_t>(gSize), pregAll);
    MicroAPI::Sub<int32_t>(kReg, aReg, kReg, pregAll);
    MicroAPI::RegTensor<uint32_t>& kLoU32 = (MicroAPI::RegTensor<uint32_t>&)kReg;
    MicroAPI::Gather(oReg, inUb, kLoU32, pregAll);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregAll);
    if constexpr (MBASE == 128U) {
        // hi 段: k = (gStart + r) mod gSize, r ∈ [64,128)
        MicroAPI::Arange<int32_t>(aReg, static_cast<int32_t>(gStart + 64U));
        MicroAPI::Div<int32_t>(qReg, aReg, gszReg, pregAll);
        MicroAPI::Muls<int32_t>(kReg, qReg, static_cast<int32_t>(gSize), pregAll);
        MicroAPI::Sub<int32_t>(kReg, aReg, kReg, pregAll);
        MicroAPI::RegTensor<uint32_t>& kHiU32 = (MicroAPI::RegTensor<uint32_t>&)kReg;
        MicroAPI::Gather(oReg, inUb, kHiU32, pregAll);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(maxUb + 64, oReg, pregAll);
    }
}

/*!
 * \brief gap 双段(S1G case2.1) 直写。
 * 前置: 尾段值在 inUb[0..gEnd], 头段值在 inUb[headOff..headOff+headLen)
 * (headOff 8 元素对齐 = 32B, 由 CopyIn gap 布局约定; m = headLen + gEnd + 1,
 * 最坏 span = headOff + headLen <= 136)。
 * 行序列: r < headLen 取 inUb[headOff + r], 其余行取 inUb[r - headLen]。
 * 头段 Gather 基址 = inUb + headOff(32B 对齐), 索引 = r 本身;
 * 尾段索引 = r - headLen(Adds ZEROING: 头段 lane 置 0, 恒落槽内);
 * head/tail 分界为语义运行时量, Compares 产 mask 保留;
 * 段界/Store 用编译期 mask(CreateMask<ALL>)直写整槽。
 */
template <typename T, typename INPUT_T, uint32_t MBASE>
__simd_vf__ inline void SinkGapDirectVF(__ubuf__ T* maxUb, __ubuf__ INPUT_T* inUb, uint32_t headOff, uint32_t headLen)
{
    MicroAPI::RegTensor<T> gHead;
    MicroAPI::RegTensor<T> gTail;
    MicroAPI::RegTensor<T> oReg;
    MicroAPI::RegTensor<int32_t> rReg;
    MicroAPI::RegTensor<int32_t> idxTail;
    MicroAPI::RegTensor<uint32_t>& rU32 = (MicroAPI::RegTensor<uint32_t>&)rReg;
    MicroAPI::RegTensor<uint32_t>& idxTailU32 = (MicroAPI::RegTensor<uint32_t>&)idxTail;
    MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    // lo 段: 行 r ∈ [0,64)
    MicroAPI::Arange<int32_t>(rReg, 0);
    MicroAPI::MaskReg pregHeadLo;
    MicroAPI::Compares<int32_t, CMPMODE::LT>(pregHeadLo, rReg, static_cast<int32_t>(headLen), pregAll);
    MicroAPI::MaskReg pregTailLo;
    MicroAPI::Compares<int32_t, CMPMODE::GE>(pregTailLo, rReg, static_cast<int32_t>(headLen), pregAll);
    MicroAPI::Adds<int32_t>(idxTail, rReg, -static_cast<int32_t>(headLen), pregTailLo);
    MicroAPI::Gather(gHead, inUb + headOff, rU32, pregHeadLo);
    MicroAPI::Gather(gTail, inUb, idxTailU32, pregTailLo);
    MicroAPI::Select<T>(oReg, gHead, gTail, pregHeadLo);
    MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(maxUb, oReg, pregAll);
    if constexpr (MBASE == 128U) {
        // hi 段: 行 r ∈ [64,128)
        MicroAPI::Arange<int32_t>(rReg, 64);
        MicroAPI::MaskReg pregHeadHi;
        MicroAPI::Compares<int32_t, CMPMODE::LT>(pregHeadHi, rReg, static_cast<int32_t>(headLen), pregAll);
        MicroAPI::MaskReg pregTailHi;
        MicroAPI::Compares<int32_t, CMPMODE::GE>(pregTailHi, rReg, static_cast<int32_t>(headLen), pregAll);
        MicroAPI::Adds<int32_t>(idxTail, rReg, -static_cast<int32_t>(headLen), pregTailHi);
        MicroAPI::Gather(gHead, inUb + headOff, rU32, pregHeadHi);
        MicroAPI::Gather(gTail, inUb, idxTailU32, pregTailHi);
        MicroAPI::Select<T>(oReg, gHead, gTail, pregHeadHi);
        MicroAPI::StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B32>(maxUb + 64, oReg, pregAll);
    }
}

/*!
 * \brief sink 注入 dispatcher(V 锁内调用), 全场景直写后即返回。
 * \param maxUb       [out] softmaxMax 槽位(本块 m 行, 唯一 REG->UB 落点)
 * \param sinkTmpUb   [in]  float 值区(CopyIn 布局: GS1 紧凑 gCount 值 / S1G 4 情况)
 * \param gs1Start    [in]  本 block 的 GS1 轴起始(相位换算基准)
 * \param actVecMSize [in]  有效行数 m(<= MBASE)
 * \param actS1Size   [in]  GS1 布局的 s1 维块大小
 * \param gSize       [in]  GQA 组大小
 * \tparam  MBASE     [in]  M 基数(编译期, 64/128), 决定 VF 单/双寄存器展开
 *
 * gSize==1 前置: GS1 编址 gs1 = g*actS1Size + s1 且块切分不越 g 轴, gSize==1 时
 * 块内 head 恒 0; S1G (gStart+r) mod 1 == 0 —— 两种 layout 均退化为 inUb[0] 单值,
 * 与 layout 无关, 直接 BRC 复制, 逻辑最简。
 */
template <typename T, typename INPUT_T, bool IS_GS1_LAYOUT, uint32_t MBASE>
__aicore__ inline void SinkExpandMaxVf(const LocalTensor<T>& maxUb, const LocalTensor<INPUT_T>& sinkTmpUb,
                                       uint32_t gs1Start, uint32_t actVecMSize, uint32_t actS1Size, uint32_t gSize)
{
    if (actVecMSize == 0) {
        return;
    }
    __ubuf__ T* maxUbPtr = (__ubuf__ T*)maxUb.GetPhyAddr();
    __ubuf__ INPUT_T* inUb = (__ubuf__ INPUT_T*)sinkTmpUb.GetPhyAddr();

    if (gSize == 1U) {
        // 单值场景与 layout 无关, 最简路径
        SinkBrcVF<T, INPUT_T, MBASE>(maxUbPtr, inUb);
        return;
    }
    if constexpr (IS_GS1_LAYOUT) {
        uint32_t s0 = gs1Start % actS1Size;
        SinkGs1DirectVF<T, INPUT_T, MBASE>(maxUbPtr, inUb, s0, actS1Size);
        return;
    }
    uint32_t gStart = gs1Start % gSize;
    uint32_t s1Start = gs1Start / gSize;
    uint32_t gs1End = gs1Start + actVecMSize - 1;
    uint32_t gEnd = gs1End % gSize;
    uint32_t s1End = gs1End / gSize;
    if (s1Start == s1End) {
        // case1: 单 s1 行, 值区连续
        SinkContigDirectVF<T, INPUT_T, MBASE>(maxUbPtr, inUb);
        return;
    }
    if (s1Start + 1 == s1End && gStart > gEnd) {
        // case2.1: 头段 [gStart, gSize) 与尾段 [0, gEnd] 不重叠, CopyIn 按 gap 布局分段存放
        uint32_t headOff = (gEnd + 1U + 7U) & ~7U;
        uint32_t headLen = gSize - gStart;
        SinkGapDirectVF<T, INPUT_T, MBASE>(maxUbPtr, inUb, headOff, headLen);
        return;
    }
    // case2.2/3: 值区回绕(m 跨多个 g 周期)
    SinkRotDirectVF<T, INPUT_T, MBASE>(maxUbPtr, inUb, gStart, gSize);
}
} // namespace AttentionCommon

#endif // ATTN_SINK_ARCH92_H_
