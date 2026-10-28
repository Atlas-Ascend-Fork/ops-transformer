/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef VF_SOFTMAX_MXFP8_VTAIL_ARCH92_H_
#define VF_SOFTMAX_MXFP8_VTAIL_ARCH92_H_

#include "../../arch35/vf/vf_basic_block_utils.h"

namespace FaVectorApi {
using namespace AscendC;
using namespace MicroAPI;

// 与 Impl256 不同，该 VF 对每个查询只读取一个含 64 个元素的分数向量。
// DN 尾 C1 的输出行跨度为 64 列；ND C1 保留原有 256 列行跨度。
// BF16 输出布局为 [4, M+1, 16]，每个 NZ 分块预留一行填充，以避免 UB 存储体冲突。
template <uint32_t M, uint32_t ScoreStride, bool IsUpdate, bool HasMask>
__simd_vf__ inline void SoftmaxMxfp8VTailVF(__ubuf__ bfloat16_t* dst, __ubuf__ float* sum, __ubuf__ float* max,
                                            __ubuf__ float* scores, __ubuf__ float* rescale, __ubuf__ uint8_t* mask,
                                            uint16_t rows, uint32_t tailLen, float scale, float minValue,
                                            float minStateValue, float pScale)
{
    MaskReg all = CreateMask<float, MaskPattern::ALL>();
    uint32_t validCount = tailLen;
    MaskReg valid = UpdateMask<float>(validCount);
    // 输出固定覆盖 64 列；[tailLen,64) 由后面的有效列掩码显式清零，tailLen 才是真实尾长。
    uint32_t outputCount = 64;
    MaskReg outMask = UpdateMask<bfloat16_t>(outputCount);
    RegTensor<float> score, localMax, previousMax, previousSum, alpha, weight, newSum;
    RegTensor<float> minimum, stateMinimum, zero, lnP, p;
    RegTensor<uint32_t> laneZero;
    // 归约结果从编号 0 的元素取出并广播；这里的 0 不是 batch、头或缓冲槽编号。
    Duplicate(laneZero, 0U);
    RegTensor<bfloat16_t> castWeight, packedWeight, unused;
    Duplicate(minimum, minValue);
    Duplicate(stateMinimum, minStateValue);
    Duplicate(zero, 0.0f);
    Duplicate(p, pScale);
    Ln(lnP, p, all);
    UnalignRegForStore maxStore, sumStore, expStore;
    __ubuf__ float* maxOut = max;
    __ubuf__ float* sumOut = sum;
    __ubuf__ float* expOut = rescale;
    for (uint16_t row = 0; row < rows; ++row) {
        LoadAlign(score, scores + row * ScoreStride);
        Muls(score, score, scale, all);
        Select(score, score, minimum, valid);
        MaskReg masked;
        if constexpr (HasMask) {
            // 掩码每列占 1 字节；转为 uint32_t 指针后，行跨度除以其 4 字节大小来换算元素步长。
            __ubuf__ uint32_t* maskRow = (__ubuf__ uint32_t*)(mask + row * ScoreStride);
            LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::MaskDist::DIST_DS>(
                masked, maskRow, ScoreStride / sizeof(uint32_t));
            Select(score, minimum, score, masked);
        }
        Reduce<Reg::ReduceType::MAX, float, float, Reg::MaskMergeMode::ZEROING>(localMax, score, all);
        Gather<float, uint32_t>(localMax, localMax, laneZero);
        // 保存与 MXFP8 主任务一致的取整后最大值，并遵循减去 ln(pScale) 的约定。
        Muls(localMax, localMax, INV_LN2, all);
        Truncate<float, RoundMode::CAST_CEIL>(localMax, localMax, all);
        Muls(localMax, localMax, LN2, all);
        Sub(localMax, localMax, lnP, all);
        // 负数分数哨兵值乘以 INV_LN2 时可能溢出。
        // 保持与 RowInvalid 和 LSE（正无穷）判断一致的精确状态哨兵值。
        MaskReg invalidMax;
        Compare<float, CMPMODE::LE>(invalidMax, localMax, minimum, all);
        Select(localMax, stateMinimum, localMax, invalidMax);
        if constexpr (IsUpdate) {
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(previousMax, max + row);
            Max(localMax, localMax, previousMax, all);
            ExpSub(alpha, previousMax, localMax, all);
            LoadAlign<float, Reg::LoadDist::DIST_BRC_B32>(previousSum, sum + row);
        } else {
            // 首个 S2 任务没有历史状态需要缩放，更新因子取乘法单位值 1。
            Duplicate(alpha, 1.0f);
        }
        ExpSub(weight, score, localMax, all);
        // 显式清除无效填充位置，不能依赖 exp(minValue - max) 自然变为 0。
        Select(weight, weight, zero, valid);
        if constexpr (HasMask) {
            Select(weight, zero, weight, masked);
        }
        Reduce<Reg::ReduceType::SUM, float, float, Reg::MaskMergeMode::ZEROING>(newSum, weight, all);
        if constexpr (IsUpdate) {
            Mul(previousSum, previousSum, alpha, all);
            Add(newSum, newSum, previousSum, all);
        }
        // 三项状态每行各占一个 float；末尾 1 使各自输出指针每行前进一个元素。
        StoreUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(maxOut, localMax, maxStore, 1);
        StoreUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(sumOut, newSum, sumStore, 1);
        StoreUnAlign<float, Reg::PostLiteral::POST_MODE_UPDATE>(expOut, alpha, expStore, 1);
        // 转为 BF16 后收紧排列，前 64 个 BF16 对应该查询行的四个 16 列片。
        Cast<bfloat16_t, float, castTraitZero>(castWeight, weight, all);
        DeInterleave(packedWeight, unused, castWeight, castWeight);
        // 16 个 BF16 为 32 字节；row*16 定位第一列片中的本行。
        // dataBlockStride=M+1 让四个 32 字节块分别落到四个列片，片内额外预留一行。
        // 末尾 1U 是 repeatStride；下一行仍由 row*16 重新定位，不依赖本次指针后更新。
        __ubuf__ bfloat16_t* rowOut = dst + row * 16U;
        StoreAlign<bfloat16_t, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            rowOut, packedWeight, M + 1U, 1U, outMask);
    }
    // 提交非对齐存储的收尾状态；末尾 0 表示不再附加输出指针位移。
    StoreUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(maxOut, maxStore, 0);
    StoreUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(sumOut, sumStore, 0);
    StoreUnAlignPost<float, Reg::PostLiteral::POST_MODE_UPDATE>(expOut, expStore, 0);
}

template <uint32_t M, uint32_t ScoreStride, bool IsUpdate, bool HasMask>
__aicore__ inline void SoftmaxMxfp8VTail(const LocalTensor<bfloat16_t>& dst, const LocalTensor<float>& sum,
                                         const LocalTensor<float>& max, const LocalTensor<float>& scores,
                                         const LocalTensor<float>& rescale, const LocalTensor<uint8_t>& mask,
                                         uint16_t rows, uint32_t tailLen, float scale, float minValue,
                                         float minStateValue, float pScale)
{
    // 64 是本 VF 一次处理的分数列数；ScoreStride 是源行跨度，DN 为 64，ND 保留 256，不能与尾长混用。
    static_assert(ScoreStride >= 64U && ScoreStride % 64U == 0U, "tail score pitch must contain 64 columns");
    SoftmaxMxfp8VTailVF<M, ScoreStride, IsUpdate, HasMask>(
        (__ubuf__ bfloat16_t*)dst.GetPhyAddr(), (__ubuf__ float*)sum.GetPhyAddr(), (__ubuf__ float*)max.GetPhyAddr(),
        (__ubuf__ float*)scores.GetPhyAddr(), (__ubuf__ float*)rescale.GetPhyAddr(),
        (__ubuf__ uint8_t*)mask.GetPhyAddr(), rows, tailLen, scale, minValue, minStateValue, pScale);
}
} // namespace FaVectorApi
#endif
