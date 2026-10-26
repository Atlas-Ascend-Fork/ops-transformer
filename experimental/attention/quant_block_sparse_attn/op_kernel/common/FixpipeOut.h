/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file FixpipeOut.h
 * \brief
 */
#ifndef FIXPIPEOUT_H
#define FIXPIPEOUT_H

constexpr FixpipeConfig PFA_CFG_ROW_MAJOR_UB = {
    CO2Layout::ROW_MAJOR,
    true}; // ROW_MAJOR: enable NZ2ND and output ND data; true specifies that the destination address is in UB.
constexpr FixpipeConfig PFA_CFG_ROW_MAJOR_GM = {
    CO2Layout::ROW_MAJOR,
    false}; // ROW_MAJOR: enable NZ2ND and output ND data; true specifies that the destination address is in UB.
constexpr FixpipeConfig FA_CFG_NZ_UB = {
    CO2Layout::NZ, true}; // Disable NZ2ND and output NZ data; true specifies that the destination address is in UB.

struct fixpipeOutParams {
    uint32_t fixpOutMSize;
    uint32_t fixpOutNSize;
};

template <typename mmOutputType, typename computeType, typename l0cType>
__aicore__ inline void FixpipeMmCopyOutToUB(LocalTensor<mmOutputType>& mmResUb, LocalTensor<l0cType>& L0CTensor,
                                            const fixpipeOutParams& fixpOutParam)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> L0C2UbFixpParams; // L0C->UB
    L0C2UbFixpParams.nSize =
        (fixpOutParam.fixpOutNSize + 7) >>
        3 << 3; // N dimension of the BMM1 result in L0C, equal to mmadParams.n; align to 8 elements (32 bytes).
    L0C2UbFixpParams.mSize =
        (fixpOutParam.fixpOutMSize + 1) >> 1 << 1; // If fewer than 16 rows are valid, output only those rows; the M
                                                   // dimension of the BMM1 result in L0C must be even.
    L0C2UbFixpParams.srcStride =
        ((L0C2UbFixpParams.mSize + 15) >> 4)
        << 4; // Stride between consecutive matmul result segments in L0C (distance between adjacent data block starts),
              // in units of 16 *sizeof(T). // Start address offset between adjacent Z blocks in the source NZ matrix.
    L0C2UbFixpParams.dstStride =
        (L0C2UbFixpParams.nSize + 15) >>
        4 << 4; // Stride between rows in mmResUb, in elements. //
                // 128: determined by comparing dump files; for ND (S1*S2), use the mask to discard invalid data.
    L0C2UbFixpParams.dualDstCtl =
        1; // Dual-destination mode splits along M and writes M/2*N to each UB; M must be a multiple of 2.
    L0C2UbFixpParams.params.ndNum = 1;
    L0C2UbFixpParams.params.srcNdStride = 0;
    L0C2UbFixpParams.params.dstNdStride = 0;
    Fixpipe<mmOutputType, computeType, PFA_CFG_ROW_MAJOR_UB>(
        mmResUb, L0CTensor,
        L0C2UbFixpParams); // Transfer the matrix multiplication result from L0C to UB.
}

template <typename mmOutputType, typename computeType, typename l0cType>
__aicore__ inline void FixpipeMmCopyOutToGm(GlobalTensor<mmOutputType>& mmResGm, LocalTensor<l0cType>& L0CTensor,
                                            const fixpipeOutParams& fixpOutParam)
{
    FixpipeParamsC310<CO2Layout::ROW_MAJOR> L0C2GmFixpParams; // L0C->Gm
    L0C2GmFixpParams.nSize =
        (fixpOutParam.fixpOutNSize + 7) >>
        3 << 3; // N dimension of the BMM1 result in L0C, equal to mmadParams.n; align to 8 elements (32 bytes),
                // calculate by size bucket, and select valid values using the mask in vector1.
    L0C2GmFixpParams.mSize =
        (fixpOutParam.fixpOutMSize + 1) >> 1 << 1; // If fewer than 16 rows are valid, output only those rows; the M
                                                   // dimension of the BMM1 result in L0C equals mmadParams.m.
    L0C2GmFixpParams.srcStride =
        ((L0C2GmFixpParams.mSize + 15) >> 4)
        << 4; // Stride between consecutive BMM1 result segments in L0C (distance between adjacent data block starts).
    L0C2GmFixpParams.dstStride = (L0C2GmFixpParams.nSize + 15) >> 4 << 4; // Stride between rows in mmResGm.
    L0C2GmFixpParams.dualDstCtl = 1;
    L0C2GmFixpParams.params.ndNum = 1;
    L0C2GmFixpParams.params.srcNdStride = 0;
    L0C2GmFixpParams.params.dstNdStride = 0;
    Fixpipe<mmOutputType, computeType, PFA_CFG_ROW_MAJOR_GM>(
        mmResGm, L0CTensor,
        L0C2GmFixpParams); // Transfer the matmul result from L0C to GM.
}
#endif
