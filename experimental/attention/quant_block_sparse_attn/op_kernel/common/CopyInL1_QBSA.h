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
 * \file CopyInL1_QBSA.h
 * \brief QBSA-specific MXFP8 helpers.
 */
#ifndef COPYINL1_QBSA_H
#define COPYINL1_QBSA_H

#include "CopyInL1.h"

template <typename L1Type>
__aicore__ inline void MxGmCopyInToL1PABnbd(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                            GlobalTensor<int32_t>& blockTableGm, const PAShape& shape,
                                            const Position& startPos)
{
    // QBSA validates a positive PA page stride and contiguous BNBD rows on the host.
    // Preserve the physical page stride for views; no layout or stride fallback is needed here.
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset = static_cast<uint64_t>(startPos.bIdx) * shape.maxblockNumPerBatch;
    uint64_t curS2Idx = startPos.s2Offset;
    constexpr uint32_t blockElementCnt = 32U / sizeof(L1Type);
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize;
        uint64_t remainRowCnt = curS2Idx % shape.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset);
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt;
        }

        uint64_t offset = idInBlockTable * shape.pageStride +
                          static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.headDim +
                          remainRowCnt * shape.headDim + startPos.dIdx;
        LocalTensor<L1Type> dstTensor = l1Tensor[copyFinishRowCnt * blockElementCnt];
        GlobalTensor<L1Type> srcTensor = gmTensor[offset];
        DataCopyGmNDToL1(dstTensor, srcTensor, copyRowCnt, shape.copyRowNumAlign, shape.actHeadDim, shape.headDim);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
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

        // Flat BNBD scale rows use headDim as their source row stride.
        uint64_t offset = idInBlockTable * shape.pageStride +
                          static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.headDim +
                          static_cast<uint64_t>(remainRowCnt) * shape.headDim + startPos.dIdx;
        MxCopyScaleToL1Dn2Nz(l1Tensor[copyFinishRowCnt * shape.actHeadDim], gmTensor[offset], shape.actHeadDim,
                             copyRowCnt, shape.headDim);
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

        // Flat BNBD scale rows use headDim as their source row stride.
        uint64_t offset = idInBlockTable * shape.pageStride +
                          static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.headDim +
                          static_cast<uint64_t>(remainRowCnt) * shape.headDim + startPos.dIdx;
        MxCopyScaleToL1Nd2Nz(l1Tensor[copyFinishRowCnt * blockElementCnt], gmTensor[offset], copyRowCnt,
                             shape.actHeadDim, shape.headDim, shape.copyRowNumAlign);
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

#endif // COPYINL1_QBSA_H
