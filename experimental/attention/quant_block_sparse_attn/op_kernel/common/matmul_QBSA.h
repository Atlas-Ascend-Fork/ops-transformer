/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_QBSA.h
 * \brief QBSA-specific MXFP8 helpers.
 */
#ifndef MATMUL_QBSA_H
#define MATMUL_QBSA_H

#include "matmul.h"

namespace fa_base_matmul {

__aicore__ inline MMParam MxMakeMMParam(uint32_t singleM, uint32_t singleN, uint32_t singleK, bool isLeftTranspose,
                                        bool isRightTranspose, bool cmatrixInitVal = true, bool isOutKFisrt = true,
                                        uint32_t unitFlag = 0, uint32_t realM = 0)
{
    // Small constructors for MX call sites to make matmul dimensions and transpose flags easier to read.
    // In the MX path, these flags describe the L1 views consumed by LoadDataToL0*Mx;
    // they do not launch an additional explicit transpose kernel.
    MMParam param;
    param.singleM = singleM;
    param.singleN = singleN;
    param.singleK = singleK;
    param.isLeftTranspose = isLeftTranspose;
    param.isRightTranspose = isRightTranspose;
    param.cmatrixInitVal = cmatrixInitVal;
    param.isOutKFisrt = isOutKFisrt;
    param.unitFlag = unitFlag;
    param.realM = realM;
    return param;
}

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MxMatmulFull(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                    L0AType& aL0BuffsDb, L0BType& bL0BuffsDb, const LocalTensor<C>& cL0Tensor,
                                    MMParam& param,
                                    const LocalTensor<AScaleType>& aScaleL1Tensor = LocalTensor<AScaleType>(),
                                    const LocalTensor<BScaleType>& bScaleL1Tensor = LocalTensor<BScaleType>())
{
    // MX wrapper around the existing MatmulFull template. The key differences from ordinary FP8 matmul are:
    // A/B scale tensors are passed through, and L0ADType/L0BDType can be set to mx_fp8_e4m3_t,
    // selecting hardware dequantization during the L1-to-L0 LoadData transfer.
    // Use the physical L1 width, not param.singleK (which can be a shorter tail).
    MatmulFull<A, B, C, baseM, baseN, baseK, AL, BL, L0AType, L0BType, AScaleType, BScaleType, L0ADType, L0BDType>(
        aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param, aScaleL1Tensor, bScaleL1Tensor);
}

template <bool LOAD_B, bool RELEASE_B, typename A, typename B, typename C, uint32_t baseM, uint32_t baseN,
          uint32_t baseK, ABLayout AL, ABLayout BL, typename L0AType, typename L0BType,
          typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t, typename L0ADType = A,
          typename L0BDType = B>
__aicore__ inline void MxMatmulFullReuseB(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                          L0AType& aL0BuffsDb, L0BType& bL0BuffsDb, const LocalTensor<C>& cL0Tensor,
                                          MMParam& param,
                                          const LocalTensor<AScaleType>& aScaleL1Tensor = LocalTensor<AScaleType>(),
                                          const LocalTensor<BScaleType>& bScaleL1Tensor = LocalTensor<BScaleType>())
{
    static_assert(LOAD_B || RELEASE_B, "a reused MX L0B tile must be loaded or released");

    Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.Wait<HardEvent::M_MTE1>();
    LocalTensor<L0ADType> l0aTensor = l0aBuffer.GetTensor<L0ADType>();
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
    if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0AMx<A, L0ADType, baseK>(l0aTensor, aL1Tensor, aScaleL1Tensor, param, 0, param.singleK,
                                            param.singleM);
    } else
#endif
    {
        LoadDataToL0A(l0aTensor, aL1Tensor, param, 0, param.singleK, param.singleM);
    }
    l0aBuffer.Set<HardEvent::MTE1_M>();

    Buffer<BufferType::L0B> l0bBuffer;
    if constexpr (LOAD_B) {
        l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.Wait<HardEvent::M_MTE1>();
        LocalTensor<L0BDType> l0bTensor = l0bBuffer.GetTensor<L0BDType>();
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
        if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0BMx<B, L0BDType>(l0bTensor, bL1Tensor, bScaleL1Tensor, param, 0, param.singleK, param.singleN);
        } else
#endif
        {
            LoadDataToL0B(l0bTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
        }
        l0bBuffer.Set<HardEvent::MTE1_M>();
        l0bBuffer.Wait<HardEvent::MTE1_M>();
    } else {
        // The preceding paired call consumed Get() and deliberately retained that exact L0B tile.
        l0bBuffer = bL0BuffsDb.GetPre();
    }

    l0aBuffer.Wait<HardEvent::MTE1_M>();
    LocalTensor<L0BDType> l0bTensor = l0bBuffer.GetTensor<L0BDType>();
    MmadParams params;
    params.m = param.realM != 0 ? param.realM : param.singleM;
    params.n = param.singleN;
    params.k = param.singleK;
    params.cmatrixInitVal = param.isOutKFisrt;
    params.cmatrixSource = false;
    params.unitFlag = param.unitFlag;
    constexpr uint32_t singleRowM = 1U;
    // For M=1, use the minimum 16-row MMAD fractal; otherwise keep the actual row count and let the underlying matmul
    // handle the tail.
    if (params.m == singleRowM) {
        params.m = ONE_FRACTAL_H_ELEMENT;
    }
    Mmad(cL0Tensor, l0aTensor, l0bTensor, params);

    l0aBuffer.Set<HardEvent::M_MTE1>();
    if constexpr (RELEASE_B) {
        l0bBuffer.Set<HardEvent::M_MTE1>();
    }
}

} // namespace fa_base_matmul

#endif // MATMUL_QBSA_H
