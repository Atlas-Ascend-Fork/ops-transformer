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
 * \file vf_cast_transdata.h
 */
#ifndef MY_CAST_TRANSDATA_DECONFLICT_INTERFACE_H
#define MY_CAST_TRANSDATA_DECONFLICT_INTERFACE_H

#include "kernel_tensor.h"
#include "../flash_attention_score_grad_common.h"

namespace AscendC {
#ifndef __CCE_KT_TEST__
using namespace Reg;
template <bool IS_HIFP8 = false>
constexpr AscendC::Reg::CastTrait castTraitFp322Fp8Zero = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    IS_HIFP8 ? AscendC::RoundMode::CAST_ROUND : AscendC::RoundMode::CAST_RINT,
};
template <bool IS_HIFP8 = false>
constexpr AscendC::Reg::CastTrait castTraitFp322Fp8Two = {
    AscendC::Reg::RegLayout::TWO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    IS_HIFP8 ? AscendC::RoundMode::CAST_ROUND : AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::Reg::CastTrait castTraitFp322Fp16Odd = {
    AscendC::Reg::RegLayout::ONE,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
constexpr AscendC::Reg::CastTrait castTraitFp322Fp16Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};
/* **************************************************************************************************
 * cast + ND_2_NZ                                             *
 * ************************************************************************************************* */
/*
 * @ingroup BroadcastSub
 * @brief compute :res = ND_2_NZ(cast(fp32_x))
 * @param [out] dstTensor output LocalTensor
 * @param [in] srcTensor input src LocalTensor
 */

template <typename T1, typename T, uint32_t srcN>
__aicore__ inline void CastTransdataDeconflict(const LocalTensor<T1>& dstTensor, const LocalTensor<T>& srcTensor,
                                               const LocalTensor<uint8_t>& selrIndexesTensor, uint32_t srcM)
{
    const uint32_t blockSize = 32;
    const uint32_t blockN = blockSize / sizeof(T1); // 16
    const uint32_t fullExeSize = srcN;
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint32_t blockStride = (srcM * blockN) * sizeof(T1) / blockSize + 1;
    uint32_t repeatStride = 1;

    if constexpr (IsSameType<T1, half>::value) {
        if constexpr (srcN <= 128) {
            // 128*128 及 64*128基本块走原分支
            __VEC_SCOPE__
            {
                RegTensor<T> vregSrcEven;
                RegTensor<T> vregSrcOdd;
                RegTensor<half> vregCastEven;
                RegTensor<half> vregCastOdd;
                RegTensor<half> vregCastRes;
                MaskReg pregFullExe = CreateMask<T1, MaskPattern::ALL>();

                // [m,n] -> [n1,m1,16,16] -> [n1,m1*16,16] -> [n1,m1*16+1,16]

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                        vregSrcEven, vregSrcOdd, ((__ubuf__ T*&)srcLocalInt), fullExeSize);
                    Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
                    Cast<T1, T, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
                    // 0101: b16 0001: b32 1111: b8
                    Or((RegTensor<uint16_t>&)vregCastRes, (RegTensor<uint16_t>&)vregCastEven,
                       (RegTensor<uint16_t>&)vregCastOdd, pregFullExe);
                    // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
                    DataCopy<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ T1*&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregFullExe);
                }
            }
        } else if constexpr (srcN <= 256) {
            // FP8 64*256基本块走此分支
            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + 128 * sizeof(T);
            uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + 33 * 128 * sizeof(T1);
            __VEC_SCOPE__
            {
                RegTensor<T> vregSrcEven;
                RegTensor<T> vregSrcOdd;
                RegTensor<half> vregCastEven;
                RegTensor<half> vregCastOdd;
                RegTensor<half> vregCastRes;
                MaskReg pregFullExe = CreateMask<T1, MaskPattern::ALL>();

                // [m,n] -> [n1,m1,16,16] -> [n1,m1*16,16] -> [n1,m1*16+1,16]

                // 处理左边的32*128
                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                        vregSrcEven, vregSrcOdd, ((__ubuf__ T*&)srcLocalInt), fullExeSize);
                    Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
                    Cast<T1, T, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
                    // 0101: b16 0001: b32 1111: b8
                    Or((RegTensor<uint16_t>&)vregCastRes, (RegTensor<uint16_t>&)vregCastEven,
                       (RegTensor<uint16_t>&)vregCastOdd, pregFullExe);
                    // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
                    DataCopy<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ T1*&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregFullExe);
                }
                // 处理右边的32*128
                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                        vregSrcEven, vregSrcOdd, ((__ubuf__ T*&)srcLocalIntTail), fullExeSize);
                    Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
                    Cast<T1, T, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
                    // 0101: b16 0001: b32 1111: b8
                    Or((RegTensor<uint16_t>&)vregCastRes, (RegTensor<uint16_t>&)vregCastEven,
                       (RegTensor<uint16_t>&)vregCastOdd, pregFullExe);
                    // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
                    DataCopy<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ T1*&)dstLocalIntTail), vregCastRes, blockStride, repeatStride, pregFullExe);
                }
            }
        }
    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
        if constexpr (srcN <= 128) {
            __VEC_SCOPE__
            {
                RegTensor<T> vregSrcEven;
                RegTensor<T> vregSrcOdd;
                RegTensor<bfloat16_t> vregCastEven;
                RegTensor<bfloat16_t> vregCastOdd;
                RegTensor<bfloat16_t> vregCastRes;
                MaskReg pregFullExe = CreateMask<T1, MaskPattern::ALL>();

                // [m,n] -> [n1,m1,16,16] -> [n1,m1*16,16] -> [n1,m1*16+1,16]

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                        vregSrcEven, vregSrcOdd, ((__ubuf__ T*&)srcLocalInt), fullExeSize);
                    Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
                    Cast<T1, T, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
                    // 0101: b16 0001: b32 1111: b8
                    Or((RegTensor<uint16_t>&)vregCastRes, (RegTensor<uint16_t>&)vregCastEven,
                       (RegTensor<uint16_t>&)vregCastOdd, pregFullExe);
                    // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
                    DataCopy<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ T1*&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregFullExe);
                }
            }
        } else if constexpr (srcN <= 256) {
            // FP8 64*256基本块走此分支
            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + 128 * sizeof(T);
            uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + 33 * 128 * sizeof(T1);
            __VEC_SCOPE__
            {
                RegTensor<T> vregSrcEven;
                RegTensor<T> vregSrcOdd;
                RegTensor<bfloat16_t> vregCastEven;
                RegTensor<bfloat16_t> vregCastOdd;
                RegTensor<bfloat16_t> vregCastRes;
                MaskReg pregFullExe = CreateMask<T1, MaskPattern::ALL>();

                // [m,n] -> [n1,m1,16,16] -> [n1,m1*16,16] -> [n1,m1*16+1,16]

                // 处理左边的32*128
                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                        vregSrcEven, vregSrcOdd, ((__ubuf__ T*&)srcLocalInt), fullExeSize);
                    Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
                    Cast<T1, T, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
                    // 0101: b16 0001: b32 1111: b8
                    Or((RegTensor<uint16_t>&)vregCastRes, (RegTensor<uint16_t>&)vregCastEven,
                       (RegTensor<uint16_t>&)vregCastOdd, pregFullExe);
                    // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
                    DataCopy<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ T1*&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregFullExe);
                }
                // 处理右边的32*128
                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                        vregSrcEven, vregSrcOdd, ((__ubuf__ T*&)srcLocalIntTail), fullExeSize);
                    Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
                    Cast<T1, T, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
                    // 0101: b16 0001: b32 1111: b8
                    Or((RegTensor<uint16_t>&)vregCastRes, (RegTensor<uint16_t>&)vregCastEven,
                       (RegTensor<uint16_t>&)vregCastOdd, pregFullExe);
                    // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
                    DataCopy<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ T1*&)dstLocalIntTail), vregCastRes, blockStride, repeatStride, pregFullExe);
                }
            }
        }
    } else if constexpr (IsSameType<T1, T>::value && srcN == 128) {
        uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + (srcM + 1) * (srcN / 2) * sizeof(T1);
        __VEC_SCOPE__
        {
            RegTensor<T> vregSrc;
            RegTensor<T> vregSrcTail;
            MaskReg pregFullExe = CreateMask<T, MaskPattern::ALL>();
            // [m,n] -> [n1,m1,16,16] -> [n1,m1*16,16] -> [n1,m1*16+1,16]
            for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                DataCopy<T1, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), 64);
                DataCopy<T1, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalInt), 64);
                // high 16bits represents stride with each 8 blocks（256B) low 16bits represent repeat stride
                DataCopy<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T*&)dstLocalInt), vregSrc, blockStride, repeatStride, pregFullExe);
                DataCopy<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                    ((__ubuf__ T*&)dstLocalIntTail), vregSrcTail, blockStride, repeatStride, pregFullExe);
            }
        }
    } else if constexpr (IsSameType<T1, fp8_e5m2_t>::value || IsSameType<T1, fp8_e4m3fn_t>::value ||
                         IsSameType<T1, hifloat8_t>::value) {
        uint64_t selrIndexesInt = selrIndexesTensor.GetPhyAddr();
        if constexpr (srcN <= 128) {
            __VEC_SCOPE__
            {
                RegTensor<T> vregSrcEven;
                RegTensor<T> vregSrcOdd;
                // fp8_e5m2_t
                RegTensor<T1> vregCastEven;
                RegTensor<T1> vregCastOdd;
                RegTensor<T1> vregCastResTmp;
                RegTensor<T1> vregCastRes;
                RegTensor<T1> vregIndexes;
                MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
                MaskReg preg_all_b8 = CreateMask<T1, MaskPattern::ALL>();

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                        vregSrcEven, vregSrcOdd, ((__ubuf__ T*&)srcLocalInt), fullExeSize);
                    Cast<T1, T, castTraitFp322Fp8Zero<IsSameType<T1, hifloat8_t>::value>>(vregCastEven, vregSrcEven,
                                                                                          preg_all);
                    Cast<T1, T, castTraitFp322Fp8Two<IsSameType<T1, hifloat8_t>::value>>(vregCastOdd, vregSrcOdd,
                                                                                         preg_all);
                    Or((RegTensor<uint8_t>&)vregCastResTmp, (RegTensor<uint8_t>&)vregCastEven,
                       (RegTensor<uint8_t>&)vregCastOdd, preg_all_b8);
                    DataCopy<uint8_t, Reg::LoadDist::DIST_NORM>((RegTensor<uint8_t>&)vregIndexes,
                                                                ((__ubuf__ uint8_t*&)selrIndexesInt));
                    Gather<uint8_t>((RegTensor<uint8_t>&)vregCastRes, (RegTensor<uint8_t>&)vregCastResTmp,
                                    (RegTensor<uint8_t>&)vregIndexes);
                    DataCopy<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ T1*&)dstLocalInt), vregCastRes, blockStride, repeatStride, preg_all_b8);
                }
            }
        } else if constexpr (srcN <= 256) {
            uint64_t srcLocalIntTail = srcTensor.GetPhyAddr() + 128 * sizeof(T);
            uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + 33 * 128 * sizeof(T1);
            uint64_t selrIndexesIntTail = selrIndexesTensor.GetPhyAddr() + 128;
            __VEC_SCOPE__
            {
                RegTensor<T> vregSrcEven;
                RegTensor<T> vregSrcOdd;
                // fp8_e5m2_t
                RegTensor<T1> vregCastEven;
                RegTensor<T1> vregCastOdd;
                RegTensor<T1> vregCastResTmp;
                RegTensor<T1> vregCastRes;
                RegTensor<T1> vregIndexes;
                MaskReg preg_all = CreateMask<T, MaskPattern::ALL>();
                MaskReg preg_all_b8 = CreateMask<T1, MaskPattern::ALL>();

                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                        vregSrcEven, vregSrcOdd, ((__ubuf__ T*&)srcLocalInt), fullExeSize);
                    Cast<T1, T, castTraitFp322Fp8Zero<IsSameType<T1, hifloat8_t>::value>>(vregCastEven, vregSrcEven,
                                                                                          preg_all);
                    Cast<T1, T, castTraitFp322Fp8Two<IsSameType<T1, hifloat8_t>::value>>(vregCastOdd, vregSrcOdd,
                                                                                         preg_all);
                    Or((RegTensor<uint8_t>&)vregCastResTmp, (RegTensor<uint8_t>&)vregCastEven,
                       (RegTensor<uint8_t>&)vregCastOdd, preg_all_b8);
                    DataCopy<uint8_t, Reg::LoadDist::DIST_NORM>((RegTensor<uint8_t>&)vregIndexes,
                                                                ((__ubuf__ uint8_t*&)selrIndexesInt));
                    Gather<uint8_t>((RegTensor<uint8_t>&)vregCastRes, (RegTensor<uint8_t>&)vregCastResTmp,
                                    (RegTensor<uint8_t>&)vregIndexes);
                    DataCopy<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ T1*&)dstLocalInt), vregCastRes, blockStride, repeatStride, preg_all_b8);
                }
                for (uint16_t m = 0; m < static_cast<uint16_t>(srcM); m++) {
                    LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
                        vregSrcEven, vregSrcOdd, ((__ubuf__ T*&)srcLocalIntTail), fullExeSize);
                    Cast<T1, T, castTraitFp322Fp8Zero<IsSameType<T1, hifloat8_t>::value>>(vregCastEven, vregSrcEven,
                                                                                          preg_all);
                    Cast<T1, T, castTraitFp322Fp8Two<IsSameType<T1, hifloat8_t>::value>>(vregCastOdd, vregSrcOdd,
                                                                                         preg_all);
                    Or((RegTensor<uint8_t>&)vregCastResTmp, (RegTensor<uint8_t>&)vregCastEven,
                       (RegTensor<uint8_t>&)vregCastOdd, preg_all_b8);
                    DataCopy<uint8_t, Reg::LoadDist::DIST_NORM>((RegTensor<uint8_t>&)vregIndexes,
                                                                ((__ubuf__ uint8_t*&)selrIndexesIntTail));
                    Gather<uint8_t>((RegTensor<uint8_t>&)vregCastRes, (RegTensor<uint8_t>&)vregCastResTmp,
                                    (RegTensor<uint8_t>&)vregIndexes);
                    DataCopy<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
                        ((__ubuf__ T1*&)dstLocalIntTail), vregCastRes, blockStride, repeatStride, preg_all_b8);
                }
            }
        }
    }
}

/* **************************************************************************************************
 * dropout + cast + ND_2_NZ                                                                         *
 * ************************************************************************************************* */
/*
 * @brief Apply a one-bit dropout mask in FP32 registers, then cast and transform ND to NZ.
 *        srcTensor is read-only. Only the first validM rows are masked; all srcM rows are transformed.
 */
template <typename T1, typename T>
__simd_vf__ inline void CastDropoutTransdataDeconflictVFB16(const uint32_t fullExeSize, uint64_t srcLocalInt,
                                                            uint64_t dstLocalInt, uint64_t maskLocalInt,
                                                            uint32_t maskRowStride, uint32_t blockStride,
                                                            uint32_t repeatStride, uint32_t srcM, uint32_t validM,
                                                            float divValue)
{
    RegTensor<T> vregSrcEven;
    RegTensor<T> vregSrcOdd;
    RegTensor<T> vregSelectEven;
    RegTensor<T> vregSelectOdd;
    RegTensor<T> vregZero;
    RegTensor<T1> vregCastEven;
    RegTensor<T1> vregCastOdd;
    RegTensor<T1> vregCastRes;
    MaskReg pregFullExe = CreateMask<T1, MaskPattern::ALL>();
    MaskReg pregAll = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregFalse = CreateMask<int8_t, MaskPattern::ALLF>();
    MaskReg pregPacked;
    MaskReg pregFirstHalf;
    MaskReg pregSecondHalf;
    MaskReg pregEven;
    MaskReg pregOdd;
    Duplicate(vregZero, 0);

    // DIST_DINTLV_B32 deinterleaves FP32 lanes, so deinterleave the packed mask in the same way.
    for (uint16_t m = 0; m < static_cast<uint16_t>(validM); ++m) {
        LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
            vregSrcEven, vregSrcOdd, ((__ubuf__ T*&)srcLocalInt), fullExeSize);
        LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::MaskDist::DIST_US>(
            pregPacked, ((__ubuf__ uint32_t*&)maskLocalInt), maskRowStride);
        MaskInterleave<half>(pregFirstHalf, pregSecondHalf, pregPacked, pregFalse);
        MaskDeInterleave<T>(pregEven, pregOdd, pregFirstHalf, pregSecondHalf);
        Select(vregSelectEven, vregSrcEven, vregZero, pregEven);
        Select(vregSelectOdd, vregSrcOdd, vregZero, pregOdd);
        Muls(vregSrcEven, vregSelectEven, divValue, pregAll);
        Muls(vregSrcOdd, vregSelectOdd, divValue, pregAll);
        Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
        Cast<T1, T, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
        Or((RegTensor<uint16_t>&)vregCastRes, (RegTensor<uint16_t>&)vregCastEven, (RegTensor<uint16_t>&)vregCastOdd,
           pregFullExe);
        StoreAlign<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T1*&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregFullExe);
    }

    // Preserve the old tail-row behavior: transform the source value without applying dropout.
    for (uint16_t m = static_cast<uint16_t>(validM); m < static_cast<uint16_t>(srcM); ++m) {
        LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B32>(
            vregSrcEven, vregSrcOdd, ((__ubuf__ T*&)srcLocalInt), fullExeSize);
        Cast<T1, T, castTraitFp322Fp16Even>(vregCastEven, vregSrcEven, pregFullExe);
        Cast<T1, T, castTraitFp322Fp16Odd>(vregCastOdd, vregSrcOdd, pregFullExe);
        Or((RegTensor<uint16_t>&)vregCastRes, (RegTensor<uint16_t>&)vregCastEven, (RegTensor<uint16_t>&)vregCastOdd,
           pregFullExe);
        StoreAlign<T1, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T1*&)dstLocalInt), vregCastRes, blockStride, repeatStride, pregFullExe);
    }
}

template <typename T1, typename T>
__simd_vf__ inline void CastDropoutTransdataDeconflictVFT(uint64_t srcLocalInt, uint64_t dstLocalInt,
                                                          uint64_t dstLocalIntTail, uint64_t maskLocalInt,
                                                          uint32_t maskRowStride, uint32_t blockStride,
                                                          uint32_t repeatStride, uint32_t srcM, uint32_t validM,
                                                          float divValue)
{
    constexpr uint32_t fp32VectorElements = VECTOR_REG_WIDTH / sizeof(float);
    RegTensor<T> vregSrc;
    RegTensor<T> vregSrcTail;
    RegTensor<T> vregSelect;
    RegTensor<T> vregSelectTail;
    RegTensor<T> vregZero;
    MaskReg pregAll = CreateMask<T, MaskPattern::ALL>();
    MaskReg pregFalse = CreateMask<int8_t, MaskPattern::ALLF>();
    MaskReg pregPacked;
    MaskReg pregFirstHalf;
    MaskReg pregSecondHalf;
    Duplicate(vregZero, 0);

    for (uint16_t m = 0; m < static_cast<uint16_t>(validM); ++m) {
        LoadAlign<T1, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), fp32VectorElements);
        LoadAlign<T1, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalInt),
                                                          fp32VectorElements);
        LoadAlign<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::MaskDist::DIST_US>(
            pregPacked, ((__ubuf__ uint32_t*&)maskLocalInt), maskRowStride);
        MaskInterleave<half>(pregFirstHalf, pregSecondHalf, pregPacked, pregFalse);
        Select(vregSelect, vregSrc, vregZero, pregFirstHalf);
        Select(vregSelectTail, vregSrcTail, vregZero, pregSecondHalf);
        Muls(vregSrc, vregSelect, divValue, pregAll);
        Muls(vregSrcTail, vregSelectTail, divValue, pregAll);
        StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T*&)dstLocalInt), vregSrc, blockStride, repeatStride, pregAll);
        StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T*&)dstLocalIntTail), vregSrcTail, blockStride, repeatStride, pregAll);
    }

    for (uint16_t m = static_cast<uint16_t>(validM); m < static_cast<uint16_t>(srcM); ++m) {
        LoadAlign<T1, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrc, ((__ubuf__ T1*&)srcLocalInt), fp32VectorElements);
        LoadAlign<T1, Reg::PostLiteral::POST_MODE_UPDATE>(vregSrcTail, ((__ubuf__ T1*&)srcLocalInt),
                                                          fp32VectorElements);
        StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T*&)dstLocalInt), vregSrc, blockStride, repeatStride, pregAll);
        StoreAlign<T, Reg::DataCopyMode::DATA_BLOCK_COPY, Reg::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ T*&)dstLocalIntTail), vregSrcTail, blockStride, repeatStride, pregAll);
    }
}

template <typename T1, typename T, uint32_t srcN>
__aicore__ inline void CastDropoutTransdataDeconflict(const LocalTensor<T1>& dstTensor, const LocalTensor<T>& srcTensor,
                                                      const LocalTensor<uint8_t>& dropMaskTensor, uint32_t srcM,
                                                      uint32_t validM, float divValue)
{
    // Caller provides a 128-column FP32 source and an FP16, BF16, or FP32 destination.
    constexpr uint32_t fp32VectorElements = VECTOR_REG_WIDTH / sizeof(float);
    // Each NZ column reserves one data block to avoid UB bank conflicts.
    constexpr uint32_t nzPaddingBlocks = 1;
    constexpr uint32_t nzRowStrideBlocks = 1;

    constexpr uint32_t blockSize = ONE_BLK_SIZE;
    constexpr uint32_t blockN = blockSize / sizeof(T1);
    constexpr uint32_t fullExeSize = srcN;
    // One dropout mask bit is stored for each source element.
    constexpr uint32_t maskRowStride = srcN / commondef::NUM_EIGHT;
    const uint32_t dropoutRows = validM < srcM ? validM : srcM;
    uint64_t srcLocalInt = srcTensor.GetPhyAddr();
    uint64_t dstLocalInt = dstTensor.GetPhyAddr();
    uint64_t maskLocalInt = dropMaskTensor.GetPhyAddr();
    uint32_t blockStride = (srcM * blockN) * sizeof(T1) / blockSize + nzPaddingBlocks;
    constexpr uint32_t repeatStride = nzRowStrideBlocks;

    if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
        CastDropoutTransdataDeconflictVFB16<T1, T>(fullExeSize, srcLocalInt, dstLocalInt, maskLocalInt, maskRowStride,
                                                   blockStride, repeatStride, srcM, dropoutRows, divValue);
    } else {
        uint64_t dstLocalIntTail = dstTensor.GetPhyAddr() + (srcM + nzPaddingBlocks) * fp32VectorElements * sizeof(T1);
        CastDropoutTransdataDeconflictVFT<T1, T>(srcLocalInt, dstLocalInt, dstLocalIntTail, maskLocalInt, maskRowStride,
                                                 blockStride, repeatStride, srcM, dropoutRows, divValue);
    }
}

#else
template <typename T1, typename T, uint32_t srcN>
__aicore__ inline void CastTransdataDeconflict(const LocalTensor<T1>& dstTensor, const LocalTensor<T>& srcTensor,
                                               const LocalTensor<uint8_t>& selrIndexesTensor, uint32_t srcM)
{}

template <typename T1, typename T, uint32_t srcN>
__aicore__ inline void CastDropoutTransdataDeconflict(const LocalTensor<T1>& dstTensor, const LocalTensor<T>& srcTensor,
                                                      const LocalTensor<uint8_t>& dropMaskTensor, uint32_t srcM,
                                                      uint32_t validM, float divValue)
{}
#endif
} // namespace AscendC

#endif // MY_CAST_TRANSDATA_DECONFLICT_INTERFACE_H
