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
 * \file mega_moe_layered_vf.h
 * \brief Layered 通信 wave 元数据计算的 VF 实现。
 */

#ifndef MEGA_MOE_LAYERED_VF_H
#define MEGA_MOE_LAYERED_VF_H

#include "kernel_operator.h"
#include "../common/mega_moe_constants.h"

namespace MegaMoeImpl {

using namespace AscendC;

// 计算 layered 规划所需的专家计数元数据：
//   result[0] = Σ ceil(expertCount[i] / L1_TILE_M_256)
//   result[8] = 最后一个 expertCount[i] != 0 的下标；全部为空时为 -1
//
// counts 为连续 UB 上的 per-expert token count，长度为 expertCount；
// result 为至少 64B 的 UB 输出区，result[0] 与 result[8] 分别占一个 32B 对齐槽，
// 调用方读取 result[0] 和 result[8]。调用方保证 expertCount > 0，且只在 AIV 上执行。
__simd_vf__ inline void ComputeLayeredWaveMetaVF(__ubuf__ int32_t* counts, __ubuf__ int32_t* result,
                                                 uint32_t expertCount)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = AscendC::GetVecLen() / sizeof(int32_t);
    static_assert(L1_TILE_M_256 == 256U, "layered GMM1 tile must be 256 rows");
    constexpr int32_t GROUP_ROUND = static_cast<int32_t>(L1_TILE_M_256 - 1U);
    constexpr int16_t GROUP_SHIFT = 8;

    AscendC::Reg::RegTensor<int32_t> countReg;
    AscendC::Reg::RegTensor<int32_t> groupReg;
    AscendC::Reg::RegTensor<int32_t> groupSumReg;
    AscendC::Reg::RegTensor<int32_t> laneIndexReg;
    AscendC::Reg::RegTensor<int32_t> indexReg;
    AscendC::Reg::RegTensor<int32_t> lastIndexReg;

    AscendC::Reg::MaskReg fullMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg firstLaneMask = AscendC::Reg::CreateMask<int32_t, AscendC::Reg::MaskPattern::VL1>();

    AscendC::Reg::Arange(laneIndexReg, static_cast<int32_t>(0));
    AscendC::Reg::Duplicate(groupSumReg, static_cast<int32_t>(0), fullMask);
    AscendC::Reg::Duplicate(lastIndexReg, static_cast<int32_t>(-1), fullMask);

    for (uint32_t offset = 0U; offset < expertCount; offset += ELEMENTS_PER_VECTOR) {
        uint32_t validCount = expertCount - offset;
        validCount = validCount < ELEMENTS_PER_VECTOR ? validCount : ELEMENTS_PER_VECTOR;
        uint32_t maskCount = validCount;
        AscendC::Reg::MaskReg activeMask = AscendC::Reg::UpdateMask<int32_t>(maskCount);

        AscendC::Reg::LoadAlign(countReg, counts + offset);

        // 每个专家的 M-group 数：ceil(count / 256) = (count + 255) >> 8。
        AscendC::Reg::Adds(groupReg, countReg, GROUP_ROUND, activeMask);
        AscendC::Reg::ShiftRights(groupReg, groupReg, GROUP_SHIFT, activeMask);
        AscendC::Reg::Add<int32_t, AscendC::Reg::MaskMergeMode::MERGING>(groupSumReg, groupSumReg, groupReg,
                                                                         activeMask);

        // 非零 count 的 lane 保存当前专家下标，最终对下标取 MAX 得到最后一个非空专家。
        AscendC::Reg::Adds(indexReg, laneIndexReg, static_cast<int32_t>(offset), fullMask);
        AscendC::Reg::MaskReg nonZeroMask;
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(nonZeroMask, countReg, static_cast<int32_t>(0),
                                                              activeMask);
        AscendC::Reg::Select(lastIndexReg, indexReg, lastIndexReg, nonZeroMask);
    }

    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(groupSumReg, groupSumReg, fullMask);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX>(lastIndexReg, lastIndexReg, fullMask);

    AscendC::Reg::StoreAlign(result, groupSumReg, firstLaneMask);
    AscendC::Reg::StoreAlign(result + 8, lastIndexReg, firstLaneMask);
}

// 每个 token 输出一行 64 个 uint32 位图，对应当前 Server 段；未使用位置写零。
template <bool CombineServerDedup>
__simd_vf__ inline void BuildDispatchRelayCandidatesByServerVF(__ubuf__ int32_t* topkIds, __ubuf__ uint32_t* bitmaps,
                                                               uint32_t tokenCount, uint32_t topK, int32_t expertCount,
                                                               int32_t expertsPerServer, int32_t localServer,
                                                               uint32_t serverStart, uint32_t serverCount)
{
    constexpr uint32_t ROW_ELEMENTS = AscendC::GetVecLen() / sizeof(int32_t);
    using namespace AscendC::Reg;
    RegTensor<int32_t> expertReg, laneReg;
    RegTensor<uint32_t> oneReg, zeroReg, bitsReg, groupReg, bitmapReg, outputReg;
    MaskReg fullMask = CreateMask<uint32_t, MaskPattern::ALL>();
    MaskReg activeMask, lowerMask, groupMask, laneMask;
    uint32_t activeCount = topK;
    activeMask = UpdateMask<uint32_t>(activeCount);
    Arange(laneReg, 0);
    Duplicate(oneReg, 1U, fullMask);
    Duplicate(zeroReg, 0U, fullMask);
    ShiftLeft(bitsReg, oneReg, laneReg, activeMask);
    for (uint32_t token = 0; token < tokenCount; ++token) {
        LoadAlign(expertReg, topkIds + token * ROW_ELEMENTS);
        Duplicate(outputReg, 0U, fullMask);
        for (uint32_t index = 0; index < serverCount; ++index) {
            const uint32_t server = serverStart + index;
            if constexpr (!CombineServerDedup) {
                if (server == static_cast<uint32_t>(localServer)) {
                    continue;
                }
            }
            int32_t begin = static_cast<int32_t>(static_cast<uint64_t>(server) * expertsPerServer);
            uint64_t end = static_cast<uint64_t>(server + 1U) * expertsPerServer;
            int32_t limit = end < static_cast<uint64_t>(expertCount) ? static_cast<int32_t>(end) : expertCount;
            Compares<int32_t, CMPMODE::GE>(lowerMask, expertReg, begin, activeMask);
            Compares<int32_t, CMPMODE::LT>(groupMask, expertReg, limit, lowerMask);
            Select(groupReg, bitsReg, zeroReg, groupMask);
            Reduce<ReduceType::SUM>(bitmapReg, groupReg, fullMask);
            Duplicate(bitmapReg, bitmapReg, fullMask);
            Compares<int32_t, CMPMODE::EQ>(laneMask, laneReg, static_cast<int32_t>(index), fullMask);
            Select(outputReg, bitmapReg, outputReg, laneMask);
        }
        StoreAlign(bitmaps + token * ROW_ELEMENTS, outputReg, fullMask);
    }
}

/* 整批匹配贡献位图，仅输出一个 pending：0 表示全部到齐，1 表示仍有缺失。 */
__simd_vf__ inline void CheckServerContributionsVF(__ubuf__ uint32_t* arrivedBitmaps,
                                                   __ubuf__ uint32_t* expectedBitmaps, __ubuf__ uint32_t* pending,
                                                   uint32_t tokenCount)
{
    constexpr uint32_t ELEMENTS_PER_VECTOR = AscendC::GetVecLen() / sizeof(uint32_t);
    using namespace AscendC::Reg;
    RegTensor<uint32_t> arrivedReg, expectedReg, matchedReg, pendingReg, oneReg, resultReg;
    MaskReg fullMask = CreateMask<uint32_t, MaskPattern::ALL>();
    MaskReg firstLaneMask = CreateMask<uint32_t, MaskPattern::VL1>();
    MaskReg activeMask, mismatchMask;
    Duplicate(pendingReg, 0U, fullMask);
    Duplicate(oneReg, 1U, fullMask);
    for (uint32_t offset = 0U; offset < tokenCount; offset += ELEMENTS_PER_VECTOR) {
        uint32_t validCount = tokenCount - offset;
        activeMask = UpdateMask<uint32_t>(validCount);
        LoadAlign(arrivedReg, arrivedBitmaps + offset);
        LoadAlign(expectedReg, expectedBitmaps + offset);
        And(matchedReg, arrivedReg, expectedReg, activeMask);
        Compare<uint32_t, AscendC::CMPMODE::NE>(mismatchMask, matchedReg, expectedReg, activeMask);
        Select(pendingReg, oneReg, pendingReg, mismatchMask);
    }
    Reduce<ReduceType::MAX>(resultReg, pendingReg, fullMask);
    StoreAlign(pending, resultReg, firstLaneMask);
}

} // namespace MegaMoeImpl

#endif
