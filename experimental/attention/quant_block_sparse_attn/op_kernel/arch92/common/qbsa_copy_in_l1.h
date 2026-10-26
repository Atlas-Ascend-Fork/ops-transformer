/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_ARCH92_QBSA_COPY_IN_L1_H
#define QBSA_ARCH92_QBSA_COPY_IN_L1_H

enum class KVLAYOUT {
    BNBD, // [blockNums, headNum, blockSize, headDim]
    BBH,  // [blockNums, blockSize, headNum * headDim]
    NZ    // [blockNums, headNum, d1, blockSize, d0], d1 = headDim / d0, d0 = 32 (block byte) / sizeof(KV_T)
};

struct CopyParam {
    uint32_t width;
    uint32_t height;
    uint32_t orgWidth;
};

struct PAShape {
    uint32_t blockNum;
    uint32_t blockSize;
    uint32_t headNum;             // Usually the KV head count.
    uint32_t headDim;             // For MLA, the RoPE dimension is 64 and the non-RoPE dimension is 512.
    uint32_t maxblockNumPerBatch; // Maximum number of entries per block table row.
    uint32_t actHeadDim;          // Actual copied column count, corresponding to d when tiling N in s*d.
    uint32_t copyRowNum;
    uint32_t copyRowNumAlign;
    uint32_t pageStride;
    uint32_t rowStride;
};

struct Position {
    uint32_t bIdx;
    uint32_t n2Idx;
    uint64_t s2Offset;
    uint32_t dIdx; // Splitting the N axis corresponds to splitting the D axis.
};

// Scenario: transfer key/value from GM to L1.
// GM stores data in ND format.
// L1 stores data in NZ format.
// GM rows, columns, and column strides depend on the input layout and may be D or N*D.
template <typename L1Type>
__aicore__ inline void DataCopyGmNDToL1(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor, uint32_t rowAct,
                                        uint32_t rowAlign,
                                        uint32_t col,       // D
                                        uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct; // Row count.

    nd2nzPara.dValue = col;
    nd2nzPara.srcDValue = colStride;
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmTensor, nd2nzPara);
}

template <typename L1Type>
__aicore__ inline void DataCopyGmScaleNDToL1(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                             uint32_t rowAct, uint32_t rowAlign,
                                             uint32_t col,       // D
                                             uint32_t colStride) // D or N*D
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = rowAct;

    nd2nzPara.dValue = col;
    nd2nzPara.srcDValue = colStride;
    nd2nzPara.dstNzC0Stride = rowAlign;
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = nd2nzPara.nValue;

    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t*)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, nd2nzPara);
}

template <typename L1Type>
__aicore__ inline void DataCopyGmScaleDNToL1(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                             uint32_t rowAct, uint32_t rowAlign, uint32_t col, uint32_t colStride)
{
    Dn2NzParams dn2nzPara;
    dn2nzPara.dnNum = 1;
    dn2nzPara.nValue = col >> 1;
    dn2nzPara.dValue = rowAct;
    dn2nzPara.srcDValue = colStride >> 1;
    dn2nzPara.dstNzC0Stride = dn2nzPara.nValue;
    dn2nzPara.dstNzNStride = 1;
    dn2nzPara.srcDnMatrixStride = 0;
    dn2nzPara.dstNzMatrixStride = dn2nzPara.nValue;

    LocalTensor<bfloat16_t> l1TensorCast = l1Tensor.template ReinterpretCast<bfloat16_t>();
    GlobalTensor<bfloat16_t> gmTensorCast;
    gmTensorCast.SetGlobalBuffer(((__gm__ bfloat16_t*)(gmTensor.GetPhyAddr())));
    DataCopy(l1TensorCast, gmTensorCast, dn2nzPara);
}

template <typename L1Type>
__aicore__ inline void DataCopyGmNZToL1(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor, uint32_t rowAct,
                                        uint32_t dstRowStride, uint32_t srcRowStride, uint32_t col)
{
    uint32_t blockElementCnt = 32U / sizeof(L1Type);
    if constexpr (IsSameType<L1Type, int4b_t>::value) {
        blockElementCnt = 64U;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = col / blockElementCnt;
    intriParams.blockLen = rowAct;
    intriParams.dstStride = dstRowStride;
    intriParams.srcStride = srcRowStride;
    DataCopy(l1Tensor, gmTensor, intriParams);
}

template <typename L1Type>
__aicore__ inline void GmCopyInToL1PA(LocalTensor<L1Type>& l1Tensor, GlobalTensor<L1Type>& gmTensor,
                                      GlobalTensor<int32_t>& blockTableGm, KVLAYOUT kvLayout, const PAShape& shape,
                                      const Position& startPos)
{
    uint32_t copyFinishRowCnt = 0;
    uint64_t blockTableBaseOffset =
        static_cast<uint64_t>(startPos.bIdx) * shape.maxblockNumPerBatch; // Base offset into the block table.
    uint64_t curS2Idx = startPos.s2Offset;
    uint32_t blockElementCnt = 32U / sizeof(L1Type); // Number of elements per block.
    while (copyFinishRowCnt < shape.copyRowNum) {
        uint64_t blockIdOffset = curS2Idx / shape.blockSize; // Get the block table index.
        uint64_t remainRowCnt = curS2Idx % shape.blockSize;  // Get the number of rows exceeding the current block.
        uint64_t idInBlockTable =
            blockTableGm.GetValue(blockTableBaseOffset + blockIdOffset); // Block number read from the block table.
        // Calculate the number of rows that can be copied.
        uint32_t copyRowCnt = shape.blockSize - remainRowCnt; // Process only one block at a time.
        if (copyFinishRowCnt + copyRowCnt > shape.copyRowNum) {
            copyRowCnt = shape.copyRowNum - copyFinishRowCnt; // The current block has not been fully copied.
        }
        uint64_t offset = idInBlockTable * shape.blockSize * shape.headNum * shape.headDim; // PA offset.
        if (kvLayout == KVLAYOUT::NZ) {
            offset += static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.headDim +
                      remainRowCnt * blockElementCnt + static_cast<uint64_t>(startPos.dIdx) * shape.blockSize;

            LocalTensor<L1Type> tmpNopeDstTensor = l1Tensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<L1Type> tmpNopeSrcTensor = gmTensor[offset];
            DataCopyGmNZToL1(tmpNopeDstTensor, tmpNopeSrcTensor, copyRowCnt, (shape.copyRowNumAlign - copyRowCnt),
                             (shape.blockSize - copyRowCnt), shape.actHeadDim);
        } else {
            uint64_t dStride = shape.headDim;
            if (kvLayout == KVLAYOUT::BBH) {
                offset += static_cast<uint64_t>(startPos.n2Idx) * shape.headDim +
                          remainRowCnt * shape.headDim * shape.headNum + startPos.dIdx;
                dStride = shape.headDim * shape.headNum;
            } else {
                offset = idInBlockTable * shape.pageStride +
                         static_cast<uint64_t>(startPos.n2Idx) * shape.blockSize * shape.rowStride +
                         remainRowCnt * shape.rowStride + startPos.dIdx;
                dStride = shape.rowStride;
            }

            uint32_t dValue = shape.actHeadDim;
            uint32_t srcDValue = dStride;
            LocalTensor<L1Type> tmpNopeDstTensor = l1Tensor[copyFinishRowCnt * blockElementCnt];
            GlobalTensor<L1Type> tmpNopeSrcTensor = gmTensor[offset];
            DataCopyGmNDToL1(tmpNopeDstTensor, tmpNopeSrcTensor, copyRowCnt, shape.copyRowNumAlign, dValue, srcDValue);
        }
        copyFinishRowCnt += copyRowCnt;
        curS2Idx += copyRowCnt;
    }
}

#endif // QBSA_ARCH92_QBSA_COPY_IN_L1_H
