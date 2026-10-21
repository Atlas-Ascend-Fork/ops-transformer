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

} // namespace MegaMoeImpl

#endif
