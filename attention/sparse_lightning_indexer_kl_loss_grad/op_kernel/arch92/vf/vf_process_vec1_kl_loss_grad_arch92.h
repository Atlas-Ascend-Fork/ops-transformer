/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file vf_process_vec1_kl_loss_grad_arch92.h
 * \brief
 */
#ifndef VF_PROCESS_VEC1_KL_LOSS_GRAD_ARCH92_H
#define VF_PROCESS_VEC1_KL_LOSS_GRAD_ARCH92_H

#include "kernel_tensor.h"

namespace AscendC {
constexpr uint16_t REDUCE_SIZE = 1;
constexpr uint32_t floatRepSize = 64;
using namespace Reg;
constexpr static AscendC::Reg::CastTrait castTraitB162B32Odd = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};
constexpr static AscendC::Reg::CastTrait castTraitB162B32Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

template <typename T, typename INPUT_T>
__simd_vf__ inline void ProcessVec1BasicVF(__ubuf__ T* dstUb, __ubuf__ T* srcUb, __ubuf__ INPUT_T* weightUb,
                                           const uint32_t m, const uint32_t nBaseSize, const uint32_t validSize)
{
    MaskReg full = CreateMask<float, MaskPattern::ALL>();
    RegTensor<float> zero;
    Duplicate(zero, 0.0f);
    for (uint16_t col = 0; col < nBaseSize; col += floatRepSize) {
        uint32_t remaining = validSize > col ? validSize - col : 0;
        MaskReg valid = UpdateMask<float>(remaining);
        RegTensor<float> sum;
        Duplicate(sum, 0.0f);
        for (uint16_t row = 0; row < m; ++row) {
            RegTensor<float> weight;
            RegTensor<float> value;
            RegTensor<float> masked;
            RegTensor<float> product;
            RegTensor<INPUT_T> weight16;
            if constexpr (IsSameType<INPUT_T, float>::value) {
                LoadAlign<INPUT_T, Reg::LoadDist::DIST_BRC_B32>(weight, weightUb + row);
            } else {
                LoadAlign<INPUT_T, Reg::LoadDist::DIST_BRC_B16>(weight16, weightUb + row);
                Cast<float, INPUT_T, castTraitB162B32Even>(weight, weight16, full);
            }
            LoadAlign(value, srcUb + row * nBaseSize + col);
            Select(masked, value, zero, valid);
            Max(masked, masked, zero, full);
            Mul(product, masked, weight, full);
            Add(sum, sum, product, full);
        }
        __ubuf__ T* dst = dstUb + col;
        StoreAlign(dst, sum, full);
    }
}

template <typename T, typename INPUT_T>
__aicore__ inline void ProcessVec1Vf(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
                                     const LocalTensor<INPUT_T>& weightTensor, const uint32_t m,
                                     const uint32_t nBaseSize, const uint32_t validSize)
{
    __ubuf__ T* dstUb = (__ubuf__ T*)dstTensor.GetPhyAddr();
    __ubuf__ T* srcUb = (__ubuf__ T*)srcTensor.GetPhyAddr();
    __ubuf__ INPUT_T* weightUb = (__ubuf__ INPUT_T*)weightTensor.GetPhyAddr();

    ProcessVec1BasicVF<T, INPUT_T>(dstUb, srcUb, weightUb, m, nBaseSize, validSize);
}
} // namespace AscendC

#endif // VF_PROCESS_VEC1_KL_LOSS_GRAD_ARCH92_H
