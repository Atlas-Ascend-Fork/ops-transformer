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
    gm2L1Nd2NzParams.ndNum = 1;             // ND矩阵的个数
    gm2L1Nd2NzParams.nValue = nValue;       // 单个ND矩阵的实际行数，单位为元素个数
    gm2L1Nd2NzParams.dValue = dValue;       // 单个ND矩阵的实际列数，单位为元素个数
    gm2L1Nd2NzParams.srcNdMatrixStride = 0; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
    gm2L1Nd2NzParams.srcDValue = srcDValue; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || \
    (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
    if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                  IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
        gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 31) >> 5 << 5;
    } else {
        gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4;
    }
#else
    gm2L1Nd2NzParams.dstNzC0Stride = (nValue + 15) >> 4 << 4; // NZ矩阵相邻Block起始地址之间的偏移， 单位为Block个数
#endif
    gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
    gm2L1Nd2NzParams.dstNzMatrixStride = 0; // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}

template <typename INPUT_T>
__aicore__ inline void CopyToL1Nd2NzGS1Merge(const LocalTensor<INPUT_T>& l1Tensor,
                                             const GlobalTensor<INPUT_T>& gmTensor, uint32_t ndNum, uint32_t nValue,
                                             uint32_t dValue, uint32_t srcNdMatrixStride, uint32_t srcDValue,
                                             uint32_t dstNzC0Stride) // BSNGD 合轴拷贝
{
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = ndNum;                         // ND矩阵的个数
    gm2L1Nd2NzParams.nValue = nValue;                       // 单个ND矩阵的实际行数，单位为元素个数
    gm2L1Nd2NzParams.dValue = dValue;                       // 单个ND矩阵的实际列数，单位为元素个数
    gm2L1Nd2NzParams.srcNdMatrixStride = srcNdMatrixStride; // 相邻ND矩阵起始地址之间的偏移， 单位为元素个数
    gm2L1Nd2NzParams.srcDValue = srcDValue; // 同一个ND矩阵中相邻行起始地址之间的偏移， 单位为元素个数
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || \
    (defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202)))
    if constexpr (IsSameType<INPUT_T, fp8_e5m2_t>::value || IsSameType<INPUT_T, fp8_e4m3fn_t>::value ||
                  IsSameType<INPUT_T, hifloat8_t>::value || IsSameType<INPUT_T, int8_t>::value) {
        gm2L1Nd2NzParams.dstNzC0Stride =
            (dstNzC0Stride + 31) >> 5 << 5; // NZ矩阵相邻Block起始地址之间的偏移，单位为Block个数，32对齐
    } else {
        gm2L1Nd2NzParams.dstNzC0Stride =
            (dstNzC0Stride + 15) >> 4 << 4; // NZ矩阵相邻Block起始地址之间的偏移，单位为Block个数，16对齐
    }
#else
    gm2L1Nd2NzParams.dstNzC0Stride =
        (dstNzC0Stride + 15) >> 4 << 4; // NZ矩阵相邻Block起始地址之间的偏移，单位为Block个数，16对齐
#endif
    gm2L1Nd2NzParams.dstNzNStride = 1; // 转换为NZ矩阵后，ND之间相邻两行在NZ矩阵中起始地址之间的偏移， 单位为Block个数
    gm2L1Nd2NzParams.dstNzMatrixStride =
        nValue * 32 / sizeof(INPUT_T); // 两个NZ矩阵，起始地址之间的偏移， 单位为元素数量
    DataCopy(l1Tensor, gmTensor, gm2L1Nd2NzParams);
}
#endif
