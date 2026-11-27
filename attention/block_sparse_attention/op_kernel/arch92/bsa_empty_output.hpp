/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_ARCH92_EMPTY_OUTPUT_HPP
#define BSA_ARCH92_EMPTY_OUTPUT_HPP

#include <limits>
#include "../attn_infra/arch/bsa_arch.hpp"
#include "kernel_operator.h"

namespace BsaEmptyOutput {
// UB 顶部留一行最长的 FP16/BF16 输出和一个 32 B 的 FP32 向量块。
constexpr uint32_t ZERO_ROW_BYTES = 128 * sizeof(uint16_t);
constexpr uint32_t NEG_INF_BYTES = 32;
// The BSA scratch layout occupies the first 256 KiB of the A6 UB.
constexpr uint32_t BSA_UB_LAYOUT_BYTES = 256U * 1024U;
static_assert(BSA_UB_LAYOUT_BYTES <= NpuArch::Arch::Ascend960DT::UB_SIZE);
constexpr uint32_t NEG_INF_OFFSET = BSA_UB_LAYOUT_BYTES - NEG_INF_BYTES;
constexpr uint32_t ZERO_ROW_OFFSET = NEG_INF_OFFSET - ZERO_ROW_BYTES;
static_assert(ZERO_ROW_OFFSET % 32 == 0 && NEG_INF_OFFSET % 32 == 0);

template <bool EmitLse, class UbBuffer, class ElementO>
__aicore__ inline void Write(const UbBuffer& ub, AscendC::GlobalTensor<ElementO>& o, AscendC::GlobalTensor<float>& lse,
                             uint64_t oOffset, uint64_t oRowStride, uint64_t lseOffset, uint64_t lseRowStride,
                             uint32_t rowCount, uint32_t headDim)
{
    static_assert(sizeof(ElementO) == sizeof(uint16_t), "expected FP16 or BF16 attention output");
    if (rowCount == 0) {
        return;
    }
    constexpr uint32_t syncEvent = 7;
    auto zeros = ub.template GetBufferByByte<ElementO>(ZERO_ROW_OFFSET);
    AscendC::Duplicate(zeros, static_cast<ElementO>(0), headDim);
    AscendC::LocalTensor<float> negInf;
    if constexpr (EmitLse) {
        negInf = ub.template GetBufferByByte<float>(NEG_INF_OFFSET);
        AscendC::Duplicate(negInf, -std::numeric_limits<float>::infinity(), NEG_INF_BYTES / sizeof(float));
    }
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(syncEvent);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(syncEvent);
    const AscendC::DataCopyExtParams oCopy(1, headDim * sizeof(ElementO), 0, 0, 0);
    for (uint32_t row = 0; row < rowCount; ++row) {
        AscendC::DataCopyPad(o[oOffset + row * oRowStride], zeros, oCopy);
        if constexpr (EmitLse) {
            // 精确写一个 FP32，避免影响 BSND/TND 中相邻头的有效 LSE。
            AscendC::DataCopyPad(lse[lseOffset + row * lseRowStride], negInf,
                                 AscendC::DataCopyExtParams(1, sizeof(float), 0, 0, 0));
        }
    }
    // 连续空任务复用同一块 UB，须等本轮 DMA 读完后才能重新填充。
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(syncEvent);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(syncEvent);
}
} // namespace BsaEmptyOutput

#endif // BSA_ARCH92_EMPTY_OUTPUT_HPP
