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
 * \file CopyInL1.h
 * \brief
 */
#ifndef COPYINL1_H
#define COPYINL1_H

#include "../common/qbsa_copy_in_l1.h"

template <typename INPUT_T>
__aicore__ inline void CopyToL1Nd2Nz(const LocalTensor<INPUT_T>& l1Tensor, const GlobalTensor<INPUT_T>& gmTensor,
                                     uint32_t nValue, uint32_t dValue, uint32_t srcDValue)
{
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = 1;             // Number of ND matrices.
    gm2L1Nd2NzParams.nValue = nValue;       // Actual row count of one ND matrix, in elements.
    gm2L1Nd2NzParams.dValue = dValue;       // Actual column count of one ND matrix, in elements.
    gm2L1Nd2NzParams.srcNdMatrixStride = 0; // Offset between the starts of adjacent ND matrices, in elements.
    gm2L1Nd2NzParams.srcDValue =
        srcDValue; // Offset between the starts of adjacent rows within one ND matrix, in elements.
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || \
    (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
    if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                  IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
        gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 31) >> 5 << 5;
    } else {
        gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4;
    }
#else
    gm2L1Nd2NzParams.dstNzC0Stride =
        (nValue + 15) >> 4 << 4; // Offset between the starts of adjacent NZ blocks, in blocks.
#endif
    gm2L1Nd2NzParams.dstNzNStride =
        1; // After conversion to NZ, offset between the starts of adjacent ND rows in the NZ matrix, in blocks.
    gm2L1Nd2NzParams.dstNzMatrixStride = 0; // Offset between the starts of two NZ matrices, in elements.
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}

template <typename INPUT_T>
__aicore__ inline void CopyToL1Nd2NzGS1Merge(const LocalTensor<INPUT_T>& l1Tensor,
                                             const GlobalTensor<INPUT_T>& gmTensor, uint32_t ndNum, uint32_t nValue,
                                             uint32_t dValue, uint32_t srcNdMatrixStride, uint32_t srcDValue,
                                             uint32_t dstNzC0Stride) // Copy with merged BSNGD axes.
{
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = ndNum;   // Number of ND matrices.
    gm2L1Nd2NzParams.nValue = nValue; // Actual row count of one ND matrix, in elements.
    gm2L1Nd2NzParams.dValue = dValue; // Actual column count of one ND matrix, in elements.
    gm2L1Nd2NzParams.srcNdMatrixStride =
        srcNdMatrixStride; // Offset between the starts of adjacent ND matrices, in elements.
    gm2L1Nd2NzParams.srcDValue =
        srcDValue; // Offset between the starts of adjacent rows within one ND matrix, in elements.
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || \
    (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
    if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                  IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
        gm2L1Nd2NzParams.dstNzC0Stride =
            (dstNzC0Stride + 31) >>
            5 << 5; // Offset between the starts of adjacent NZ blocks, in blocks, aligned to 32.
    } else {
        gm2L1Nd2NzParams.dstNzC0Stride =
            (dstNzC0Stride + 15) >>
            4 << 4; // Offset between the starts of adjacent NZ blocks, in blocks, aligned to 16.
    }
#else
    gm2L1Nd2NzParams.dstNzC0Stride =
        (dstNzC0Stride + 15) >> 4 << 4; // Offset between the starts of adjacent NZ blocks, in blocks, aligned to 16.
#endif
    gm2L1Nd2NzParams.dstNzNStride =
        1; // After conversion to NZ, offset between the starts of adjacent ND rows in the NZ matrix, in blocks.
    gm2L1Nd2NzParams.dstNzMatrixStride =
        nValue * 32 / sizeof(INPUT_T); // Offset between the starts of two NZ matrices, in elements.
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}
#endif
