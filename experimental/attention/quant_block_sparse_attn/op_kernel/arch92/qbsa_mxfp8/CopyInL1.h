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
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102)
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
__aicore__ inline void MxCopyScaleToL1Nd2Nz(const LocalTensor<INPUT_T>& l1Tensor, const GlobalTensor<INPUT_T>& gmTensor,
                                            uint32_t nValue, uint32_t dValue, uint32_t srcDValue,
                                            uint32_t dstNzC0Stride)
{
    // MXFP8 scales use fp8_e8m0, but the scale transfer path operates at BF16 granularity.
    // Value scale GM shape is [physicalBlock,N,blockSize/64,DV,2]; the last two e8m0 elements
    // are reinterpreted as one BF16 lane. This compresses D, so do not halve the token-group row count within a block.
    Nd2NzParams gm2L1Nd2NzParams;
    gm2L1Nd2NzParams.ndNum = 1;
    gm2L1Nd2NzParams.nValue = nValue;
    gm2L1Nd2NzParams.dValue = dValue >> 1;
    gm2L1Nd2NzParams.srcNdMatrixStride = 0;
    gm2L1Nd2NzParams.srcDValue = srcDValue >> 1;
    gm2L1Nd2NzParams.dstNzC0Stride = dstNzC0Stride;
    gm2L1Nd2NzParams.dstNzNStride = 1;
    gm2L1Nd2NzParams.dstNzMatrixStride = gm2L1Nd2NzParams.nValue;

    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t*)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, gm2L1Nd2NzParams);
}

template <typename INPUT_T>
__aicore__ inline void MxCopyScaleToL1Dn2Nz(const LocalTensor<INPUT_T>& l1Tensor, const GlobalTensor<INPUT_T>& gmTensor,
                                            uint32_t nValue, uint32_t dValue, uint32_t srcDValue)
{
    // The DN2NZ variant handles scales for transposed BMM1 operands.
    // QScale GM shape is [T,N,D/64,2]; KScale GM shape is [blockNum,N,blockSize,D/64,2].
    // After reinterpretation as BF16,
    // two e8m0 elements in the last dimension become one copy element. This ensures that when Q/K are consumed
    // by MX LoadData with transpose semantics, the scale layout matches the data layout.
    Dn2NzParams gm2L1Dn2NzParams;
    gm2L1Dn2NzParams.dnNum = 1;
    gm2L1Dn2NzParams.nValue = nValue >> 1;
    gm2L1Dn2NzParams.dValue = dValue;
    gm2L1Dn2NzParams.srcDnMatrixStride = 0;
    gm2L1Dn2NzParams.srcDValue = srcDValue >> 1;
    gm2L1Dn2NzParams.dstNzC0Stride = nValue >> 1;
    gm2L1Dn2NzParams.dstNzNStride = 1;
    gm2L1Dn2NzParams.dstNzMatrixStride = gm2L1Dn2NzParams.nValue;

    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t*)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, gm2L1Dn2NzParams);
}

template <typename L1Type>
__aicore__ inline void MxGmFlatScaleCopyInToL1PAForDN(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                                      GlobalTensor<int32_t>& blockTableGm, const PAShape& shape,
                                                      const Position& startPos)
{
    // PA scale helper for K scales per token group:
    //   Logical GM shape: [physicalBlock,N,blockSize,D/64,2].
    //   PA lookup: logical token offset -> block_table -> physical block ID + token offset within the block.
    //   Use DN2NZ for the L1 copy because the corresponding data operand is loaded transposed.
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = static_cast<uint64_t>(startPos.bIdx) * shape.maxblockNumPerBatch;
    uint64_t curS2Idx = startPos.s2Offset;
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;
        uint32_t remainRowCnt = static_cast<uint32_t>(curS2Idx % shape.blockSize);
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;
        }

        uint64_t offset = idInBlockTable * shape.pageStride +
                          static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.rowStride +
                          static_cast<uint64_t>(remainRowCnt) * shape.rowStride + startPos.dIdx;
        MxCopyScaleToL1Dn2Nz(l1Tensor[copyFinishRowCnt * shape.actHeadDim], gmTensor[offset], shape.actHeadDim,
                             copyRowCnt, shape.rowStride);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

template <typename L1Type>
__aicore__ inline void MxGmFlatScaleCopyInToL1PAForND(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                                      GlobalTensor<int32_t>& blockTableGm, const PAShape& shape,
                                                      const Position& startPos)
{
    // PA scale helper for value scales per channel group:
    // Logical GM shape: [physicalBlock,N,blockSize/64,DV,2].
    // Each copied row covers 64 value tokens within a block.
    // Use ND2NZ because V is consumed as the non-transposed right operand in BMM2.
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = static_cast<uint64_t>(startPos.bIdx) * shape.maxblockNumPerBatch;
    uint64_t curS2Idx = startPos.s2Offset;
    constexpr uint32_t blockElementCnt = 32U / sizeof(L1Type);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;
        uint32_t remainRowCnt = static_cast<uint32_t>(curS2Idx % shape.blockSize);
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;
        }

        uint64_t offset = idInBlockTable * shape.pageStride +
                          static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.rowStride +
                          static_cast<uint64_t>(remainRowCnt) * shape.rowStride + startPos.dIdx;
        MxCopyScaleToL1Nd2Nz(l1Tensor[copyFinishRowCnt * blockElementCnt], gmTensor[offset], copyRowCnt,
                             shape.actHeadDim, shape.rowStride, shape.copyRowNumAlign);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
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
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102)
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
