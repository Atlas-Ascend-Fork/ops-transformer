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
 * \file matmul.h
 * \brief
 */
#ifndef MATMUL_H
#define MATMUL_H
#include "buffers_policy.h"
using namespace AscendC;

namespace fa_base_matmul {

constexpr uint32_t UNITFLAG_DISABLE = 0;
constexpr uint32_t UNITFLAG_ENABLE = 2;
constexpr uint32_t UNITFLAG_EN_OUTER_LAST = 3;
static constexpr uint32_t FP16_ONE_FRACTAL_ELEMENT =
    16; // One fractal block is 512 bytes, containing 16*16 FP16 elements.
static constexpr uint32_t INT4_ONE_FRACTAL_ELEMENT =
    64;                                               // One fractal block is 512 bytes, containing 16*64 FP16 elements.
static constexpr uint32_t ONE_FRACTAL_H_ELEMENT = 16; // One fractal block is 512 bytes, with 16 elements along height.
static constexpr uint32_t ONE_FRACTAL_W_BYTE = 32;    // One fractal block is 512 bytes, with 32 bytes along width.
static constexpr uint32_t LOAD3D_L1W_SIZE = 16;
static constexpr uint32_t MMAD_MN_SIZE_10 = 10;
static constexpr uint8_t LOAD3D_STRIDE_W = 1;
static constexpr uint8_t LOAD3D_STRIDE_H = 1;
static constexpr uint8_t LOAD3D_FILTER_W = 1;
static constexpr uint8_t LOAD3D_FILTER_H = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_W = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_H = 1;
static constexpr uint32_t K_STEP_ALIGN_BASE = 2;
static constexpr uint32_t M_STEP_ALIGN_BASE = 2;
static constexpr uint32_t MX_FP4_PTG_PCG_SCALE_PARAM =
    32; // For MXFP4 pseudo quantization during transfer, 32 elements share one scale.
static constexpr uint32_t HI_FP4_PTG_PCG_SCALE_PARAM =
    64; // For HIFP4 pseudo quantization during transfer, 64 elements share one scale.

struct MMParam {
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    bool isLeftTranspose;
    bool isRightTranspose;
    bool cmatrixInitVal = true;
    bool isOutKFisrt = true; // Defaults to true; when splitting K in L1, true indicates the first K iteration.
    uint32_t unitFlag =
        0; // 0: disabled; do not configure unitFlag.
           // 2: enabled; the split-K interface (MatmulK) sets mmadParams.unitFlag to 0b10.
           // 3: enabled; on the last K iteration, the split-K interface (MatmulK) sets mmadParams.unitFlag to
           // 0b11. External callers should set this parameter to 3 on the last iteration of the outer K loop.
    uint32_t realM = 0; // BMM2 uses s1realsize as M; leaving this unset does not affect existing code behavior.
};

__aicore__ inline MMParam MakeMMParam(uint32_t singleM, uint32_t singleN, uint32_t singleK, bool isLeftTranspose,
                                      bool isRightTranspose, bool cmatrixInitVal = true, bool isOutKFisrt = true,
                                      uint32_t unitFlag = 0, uint32_t realM = 0)
{
    return {.singleM = singleM,
            .singleN = singleN,
            .singleK = singleK,
            .isLeftTranspose = isLeftTranspose,
            .isRightTranspose = isRightTranspose,
            .cmatrixInitVal = cmatrixInitVal,
            .isOutKFisrt = isOutKFisrt,
            .unitFlag = unitFlag,
            .realM = realM};
}

enum class ABLayout {
    MK = 0,
    KM = 1,
    KN = 2,
    NK = 3,
};

template <typename T>
__aicore__ inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
template <typename T>
__aicore__ inline uint32_t GetBlockNum(uint32_t size)
{
    if constexpr (IsSameType<T, float>::value) {
        return ((size + 7) >> 3 << 3) >> 3;
    } else if constexpr ((IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                          IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value)) {
        return ((size + 31) >> 5 << 5) >> 5;
    } else {
        return ((size + 15) >> 4 << 4) >> 4;
    }
}
// L1-to-L0A transfer with split-K, split-M, or full loading.
template <typename T>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor, const MMParam& mmParam,
                                     uint64_t L1Aoffset, uint32_t kSplitSize, uint32_t mSplitSize)
{
    LoadData2DParamsV2 loadData2DParamsA; // Parameter structure for the base LoadData API.
    loadData2DParamsA.mStartPosition = 0; // For an M*K matrix, source start along M, in units of 16 elements.
    loadData2DParamsA.kStartPosition = 0; // For an M*K matrix, source start along K, in units of 32 bytes.
    loadData2DParamsA.ifTranspose = mmParam.isLeftTranspose; // Whether to enable transposition of each fractal matrix.
    if (loadData2DParamsA.ifTranspose) {
        loadData2DParamsA.mStep =
            ((kSplitSize + 15) >> 4 << 4) >>
            4; // For an M*K matrix, transfer length along source M (S1 rounded up to a 512-byte fractal block; 16*16
               // FP16 elements require alignment to 16), in units of 16 elements; valid range: mStep in [0,255].
        if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                      IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
            loadData2DParamsA.mStep = (loadData2DParamsA.mStep + 1) >> 1 << 1;
        }
        loadData2DParamsA.kStep =
            GetBlockNum<T>(mSplitSize); // For an M*K matrix, transfer length along source K (qkD FP16 elements), in
                                        // units of 32 bytes; nStep is in [0,255].
    } else {
        loadData2DParamsA.mStep =
            ((mSplitSize + 15) >> 4 << 4) >>
            4; // For an M*K matrix, transfer length along source M (S1 rounded up to a 512-byte fractal block; 16*16
               // FP16 elements require alignment to 16), in units of 16 elements; valid range: mStep in [0,255].
        loadData2DParamsA.kStep =
            GetBlockNum<T>(kSplitSize); // For an M*K matrix, transfer length along source K (qkD FP16 elements), in
                                        // units of 32 bytes; nStep is in [0,255].
    }
    if constexpr (IsSameType<T, float>::value) {
        if (loadData2DParamsA.ifTranspose) {
            loadData2DParamsA.kStep = CeilAlign(loadData2DParamsA.kStep, K_STEP_ALIGN_BASE);
        }
    }
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        // Used with UB-to-L1 transfers: 256 * 32 / 256.
        // Transfer alignment of 64.
        loadData2DParamsA.srcStride =
            loadData2DParamsA.ifTranspose ?
                ((kSplitSize + 63) >> 6 << 6) >> 4 :
                ((mSplitSize + 31) >> 5 << 5) >> 4; // For an M*K matrix, stride between consecutive source fractal
                                                    // starts along K, in units of 512 bytes.
    } else {
        loadData2DParamsA.srcStride =
            loadData2DParamsA.ifTranspose ? ((mmParam.singleK + 15) >> 4 << 4) >> 4 : loadData2DParamsA.mStep;
    }
    if (mmParam.realM != 0) {
        loadData2DParamsA.mStep = ((mmParam.realM + 15) >> 4 << 4) >> 4;
    }
    loadData2DParamsA.dstStride = loadData2DParamsA.ifTranspose ? (mSplitSize + 15) >> 4 : loadData2DParamsA.mStep;
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        if (loadData2DParamsA.ifTranspose && (loadData2DParamsA.dstStride & 1)) {
            uint32_t l0bLoop = (loadData2DParamsA.mStep + 1) >> 1;
            loadData2DParamsA.mStep = M_STEP_ALIGN_BASE;
            uint64_t dstOffset = 0;
            uint64_t dstAddrStride = (mSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsA.mStartPosition;
            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsA.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                LoadData(aL0Tensor[dstOffset], aL1Tensor[L1Aoffset], loadData2DParamsA);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
        }
    } else {
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
    }
}

// L1-to-L0A transfer with split-K, split-M, or full loading.
template <typename T, typename U = T, uint32_t baseK>
__aicore__ inline void LoadDataToL0AMx(LocalTensor<U>& aL0Tensor, const LocalTensor<T>& aL1Tensor,
                                       const LocalTensor<fp8_e8m0_t>& aScaleL1Tensor, const MMParam& mmParam,
                                       uint64_t L1Aoffset, uint32_t kSplitSize, uint32_t mSplitSize)
{
    LoadData2DParamsV2 loadData2DParamsA; // Parameter structure for the base LoadData API.
    loadData2DParamsA.mStartPosition = 0; // For an M*K matrix, source start along M, in units of 16 elements.
    loadData2DParamsA.kStartPosition = 0; // For an M*K matrix, source start along K, in units of 32 bytes.
    loadData2DParamsA.ifTranspose = mmParam.isLeftTranspose; // Whether to enable transposition of each fractal matrix.
    if (loadData2DParamsA.ifTranspose) {
        loadData2DParamsA.mStep =
            ((kSplitSize + 15) >> 4 << 4) >>
            4; // For an M*K matrix, transfer length along source M (S1 rounded up to a 512-byte fractal block; 16*16
               // FP16 elements require alignment to 16), in units of 16 elements; valid range: mStep in [0,255].
        if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                      IsSameType<T, hifloat8_t>::value) {
            loadData2DParamsA.mStep = (loadData2DParamsA.mStep + 1) >> 1 << 1;
        }
        loadData2DParamsA.kStep =
            GetBlockNum<T>(mSplitSize); // For an M*K matrix, transfer length along source K (qkD FP16 elements), in
                                        // units of 32 bytes; nStep is in [0,255].
    } else {
        loadData2DParamsA.mStep =
            ((mSplitSize + 15) >> 4 << 4) >>
            4; // For an M*K matrix, transfer length along source M (S1 rounded up to a 512-byte fractal block; 16*16
               // FP16 elements require alignment to 16), in units of 16 elements; valid range: mStep in [0,255].
        loadData2DParamsA.kStep =
            GetBlockNum<T>(kSplitSize); // For an M*K matrix, transfer length along source K (qkD FP16 elements), in
                                        // units of 32 bytes; nStep is in [0,255].
    }
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value) {
        // Used with UB-to-L1 transfers: 256 * 32 / 256.
        // Transfer alignment of 64.
        loadData2DParamsA.srcStride =
            loadData2DParamsA.ifTranspose ?
                ((baseK + 63) >> 6 << 6) >> 4 :
                ((mSplitSize + 31) >> 5 << 5) >> 4; // For an M*K matrix, stride between consecutive source fractal
                                                    // starts along K, in units of 512 bytes.
    } else {
        loadData2DParamsA.srcStride =
            loadData2DParamsA.ifTranspose ? ((mmParam.singleK + 15) >> 4 << 4) >> 4 : loadData2DParamsA.mStep;
    }
    LoadData2DMxParams loadData2DMxParamsA;
    loadData2DMxParamsA.xStartPosition = 0;
    loadData2DMxParamsA.yStartPosition = 0;
    loadData2DMxParamsA.xStep = ((mSplitSize + 15) >> 4 << 4) >> 4;
    loadData2DMxParamsA.yStep = (kSplitSize + 63) >> 5 >> 1;
    loadData2DMxParamsA.srcStride = loadData2DMxParamsA.yStep;
    loadData2DMxParamsA.dstStride = loadData2DMxParamsA.yStep;
    if (mmParam.realM != 0) {
        loadData2DParamsA.mStep = ((mmParam.realM + 15) >> 4 << 4) >> 4;
        loadData2DMxParamsA.xStep = ((mmParam.realM + 15) >> 4 << 4) >> 4;
    }
    loadData2DParamsA.dstStride = loadData2DParamsA.ifTranspose ? (mSplitSize + 15) >> 4 : loadData2DParamsA.mStep;
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value) {
        if (loadData2DParamsA.ifTranspose && (loadData2DParamsA.dstStride & 1)) {
            uint32_t l0bLoop = (loadData2DParamsA.mStep + 1) >> 1;
            loadData2DParamsA.mStep = M_STEP_ALIGN_BASE;
            loadData2DMxParamsA.xStep = loadData2DParamsA.mStep;
            uint64_t dstOffset = 0;
            uint64_t dstAddrStride = (mSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsA.mStartPosition;
            uint16_t oriMScaleStep = loadData2DMxParamsA.xStartPosition;
            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsA.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                loadData2DMxParamsA.xStartPosition = oriMScaleStep + M_STEP_ALIGN_BASE * idx;
                LoadData(aL0Tensor[dstOffset], aL1Tensor[L1Aoffset], aScaleL1Tensor[L1Aoffset >> 5], loadData2DParamsA,
                         loadData2DMxParamsA);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(aL0Tensor, aL1Tensor[L1Aoffset], aScaleL1Tensor[L1Aoffset >> 5], loadData2DParamsA,
                     loadData2DMxParamsA);
        }
    } else {
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParamsA);
    }
}

// L1-to-L0B transfer with split-K, split-M, or full loading.
template <typename T>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor, const MMParam& mmParam,
                                     uint64_t L1Boffset, uint32_t kSplitSize, uint32_t nSplitSize, int nLoops = 1)
{
    LoadData2DParamsV2 loadData2DParamsB; // Parameter structure for the base LoadData API.
    loadData2DParamsB.mStartPosition = 0; // For an M*K matrix, source start along M, in units of 16 elements.
    loadData2DParamsB.kStartPosition = 0; // For an M*K matrix, source start along K, in units of 32 bytes.
    loadData2DParamsB.ifTranspose =
        !mmParam.isRightTranspose; // Whether to enable transposition of each fractal matrix.
    if (loadData2DParamsB.ifTranspose) {
        loadData2DParamsB.mStep =
            ((kSplitSize + 15) >> 4 << 4) >>
            4; // For an M*K matrix, transfer length along source M (S1 rounded up to a 512-byte fractal block; 16*16
               // FP16 elements require alignment to 16), in units of 16 elements; valid range: mStep in [0,255].
        if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                      IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
            loadData2DParamsB.mStep = (loadData2DParamsB.mStep + 1) >> 1 << 1;
        }
        loadData2DParamsB.kStep =
            GetBlockNum<T>(nSplitSize); // For an M*K matrix, transfer length along source K (qkD FP16 elements), in
                                        // units of 32 bytes; nStep is in [0,255].
    } else {
        loadData2DParamsB.mStep =
            ((nSplitSize + 15) >> 4 << 4) >>
            4; // For an M*K matrix, transfer length along source M (S1 rounded up to a 512-byte fractal block; 16*16
               // FP16 elements require alignment to 16), in units of 16 elements; valid range: mStep in [0,255].
        loadData2DParamsB.kStep =
            GetBlockNum<T>(kSplitSize); // For an M*K matrix, transfer length along source K (qkD FP16 elements), in
                                        // units of 32 bytes; nStep is in [0,255].
    }
    if constexpr (IsSameType<T, float>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DParamsB.kStep = CeilAlign(loadData2DParamsB.kStep, K_STEP_ALIGN_BASE);
        }
    }
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DParamsB.srcStride = ((kSplitSize + 31) >> 5 << 5) >> 4;
        } else {
            loadData2DParamsB.srcStride = ((nSplitSize + 31) >> 5 << 5) >> 4;
        }
    } else {
        loadData2DParamsB.srcStride =
            loadData2DParamsB.ifTranspose ?
                (((mmParam.singleK + 15) >> 4 << 4) >> 4) :
                (((mmParam.singleN + 15) >> 4 << 4) >> 4); // For an M*K matrix, stride between consecutive source
                                                           // fractal starts along K, in units of 512 bytes.
    }
    loadData2DParamsB.dstStride =
        loadData2DParamsB.ifTranspose ?
            (nSplitSize + 15) >> 4 :
            loadData2DParamsB.mStep; // For an M*K matrix, stride between consecutive destination fractal starts along
                                     // K, in units of 512 bytes.
    if constexpr (IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value ||
                  IsSameType<T, hifloat8_t>::value || IsSameType<T, int8_t>::value) {
        if (loadData2DParamsB.ifTranspose && (loadData2DParamsB.dstStride & 1)) {
            uint32_t l0bLoop = (loadData2DParamsB.mStep + 1) >> 1;
            loadData2DParamsB.mStep = M_STEP_ALIGN_BASE;
            uint64_t dstOffset = 0;
            uint64_t dstAddrStride = (nSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsB.mStartPosition;
            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsB.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                LoadData(bL0Tensor[dstOffset], bL1Tensor[L1Boffset], loadData2DParamsB);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
        }
    } else {
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParamsB);
    }
}

template <typename T, typename U = T, typename Scale_T>
__aicore__ inline void LoadDataToL0BMx(LocalTensor<U>& bL0Tensor, const LocalTensor<T>& bL1Tensor,
                                       const LocalTensor<Scale_T>& bScaleL1Tensor, const MMParam& mmParam,
                                       uint64_t L1Boffset, uint32_t kSplitSize, uint32_t nSplitSize, int nLoops = 1)
{
    LoadData2DParamsV2 loadData2DParamsB; // Parameter structure for the base LoadData API.
    loadData2DParamsB.mStartPosition = 0; // For an M*K matrix, source start along M, in units of 16 elements.
    loadData2DParamsB.kStartPosition = 0; // For an M*K matrix, source start along K, in units of 32 bytes.
    loadData2DParamsB.ifTranspose =
        !mmParam.isRightTranspose; // Whether to enable transposition of each fractal matrix.
    if (loadData2DParamsB.ifTranspose) {
        if constexpr (IsSameType<T, fp4x2_e2m1_t>::value || IsSameType<T, hifloat4x2_t>::value) {
            loadData2DParamsB.mStep = ((mmParam.singleK + 63) >> 6 << 6) / 16;
            loadData2DParamsB.kStep = kSplitSize / 64;
            loadData2DParamsB.srcStride = ((mmParam.singleK + 15) >> 4 << 4) / 16;
            loadData2DParamsB.dstStride = kSplitSize / 16;
        } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
            loadData2DParamsB.mStep = ((kSplitSize + 15) >> 4 << 4) >>
                                      4; // For an M*K matrix, transfer length along source M (S1 rounded up to a
                                         // 512-byte fractal block; 16*16 FP16 elements require alignment to 16), in
                                         // units of 16 elements; valid range: mStep in [0,255].
            loadData2DParamsB.kStep =
                GetBlockNum<T>(nSplitSize); // For an M*K matrix, transfer length along source K (qkD FP16 elements), in
                                            // units of 32 bytes; nStep is in [0,255].
            loadData2DParamsB.srcStride = ((kSplitSize + 31) >> 5 << 5) >> 4;
            loadData2DParamsB.dstStride = (nSplitSize + 15) >> 4;
        }
    } else {
        if constexpr (IsSameType<T, fp4x2_e2m1_t>::value || IsSameType<T, hifloat4x2_t>::value) {
            loadData2DParamsB.mStep = ((mmParam.singleN + 15) >> 4 << 4) / 16;
            loadData2DParamsB.kStep = kSplitSize / 64;
            loadData2DParamsB.srcStride = loadData2DParamsB.mStep;
            loadData2DParamsB.dstStride = loadData2DParamsB.mStep;
        } else if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
            loadData2DParamsB.mStep = ((nSplitSize + 15) >> 4 << 4) >>
                                      4; // For an M*K matrix, transfer length along source M (S1 rounded up to a
                                         // 512-byte fractal block; 16*16 FP16 elements require alignment to 16), in
                                         // units of 16 elements; valid range: mStep in [0,255].
            loadData2DParamsB.kStep =
                GetBlockNum<T>(kSplitSize); // For an M*K matrix, transfer length along source K (qkD FP16 elements), in
                                            // units of 32 bytes; nStep is in [0,255].
            loadData2DParamsB.srcStride = ((nSplitSize + 31) >> 5 << 5) >> 4;
            loadData2DParamsB.dstStride = loadData2DParamsB.mStep;
        }
    }

    LoadData2DMxParams loadData2DMxParamsB;
    if constexpr (IsSameType<T, fp8_e4m3fn_t>::value) {
        loadData2DMxParamsB.xStartPosition = 0;
        loadData2DMxParamsB.yStartPosition = 0;
        loadData2DMxParamsB.xStep = ((nSplitSize + 15) >> 4 << 4) >> 4;
        loadData2DMxParamsB.yStep = (kSplitSize + 63) >> 5 >> 1;
        loadData2DMxParamsB.srcStride = loadData2DMxParamsB.yStep;
        loadData2DMxParamsB.dstStride = loadData2DMxParamsB.yStep;
        if (loadData2DParamsB.ifTranspose && (loadData2DParamsB.dstStride & 1)) {
            uint32_t l0bLoop = (loadData2DParamsB.mStep + 1) >> 1;
            loadData2DParamsB.mStep = M_STEP_ALIGN_BASE;
            loadData2DMxParamsB.xStep = loadData2DParamsB.mStep;
            uint64_t dstOffset = 0;
            uint64_t dstAddrStride = (nSplitSize + 15) / 16 * 16 * 32;
            uint16_t oriMStep = loadData2DParamsB.mStartPosition;
            uint16_t oriMScaleStep = loadData2DMxParamsB.xStartPosition;
            for (uint32_t idx = 0; idx < l0bLoop; ++idx) {
                loadData2DParamsB.mStartPosition = oriMStep + M_STEP_ALIGN_BASE * idx;
                loadData2DMxParamsB.xStartPosition = oriMScaleStep + M_STEP_ALIGN_BASE * idx;
                LoadData(bL0Tensor[dstOffset], bL1Tensor[L1Boffset], bScaleL1Tensor[L1Boffset >> 5], loadData2DParamsB,
                         loadData2DMxParamsB);
                dstOffset += dstAddrStride;
            }
        } else {
            LoadData(bL0Tensor, bL1Tensor[L1Boffset], bScaleL1Tensor[L1Boffset >> 5], loadData2DParamsB,
                     loadData2DMxParamsB);
        }
    } else if constexpr (IsSameType<T, fp4x2_e2m1_t>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DMxParamsB.xStartPosition =
                0; // Matrix start along X (M), in fractal units; each unit represents a 32-byte fractal block.
            loadData2DMxParamsB.yStartPosition =
                mmParam.singleN / MX_FP4_PTG_PCG_SCALE_PARAM / 2; // Source start along Y (K), in units of 32 bytes.
            loadData2DMxParamsB.xStep = (kSplitSize + 15) / 16;   // X/M length in 32-byte fractals; xStep: [0,255].
            loadData2DMxParamsB.yStep =
                (mmParam.singleK + 63) / MX_FP4_PTG_PCG_SCALE_PARAM /
                2; // Source transfer length along Y (K), in units of 32 bytes; valid range: yStep in [0,255].
            loadData2DMxParamsB.srcStride =
                mmParam.singleM / MX_FP4_PTG_PCG_SCALE_PARAM /
                2; // Stride between consecutive source fractal starts along X, in units of 32 bytes.
            loadData2DMxParamsB.dstStride =
                loadData2DMxParamsB
                    .yStep; // Stride between consecutive destination fractal starts along X, in units of 32 bytes.
            LoadData(bL0Tensor, bL1Tensor, bScaleL1Tensor, loadData2DParamsB, loadData2DMxParamsB);
        } else {
            loadData2DMxParamsB.xStartPosition =
                mmParam.singleK /
                16; // Matrix start along X (M), in fractal units; each unit represents a 32-byte fractal block.
            loadData2DMxParamsB.yStartPosition = 0;
            loadData2DMxParamsB.xStep = (mmParam.singleN + 15) / 16; // X/M length in 32-byte fractals; xStep: [0,255].
            loadData2DMxParamsB.yStep =
                (kSplitSize + 63) / MX_FP4_PTG_PCG_SCALE_PARAM /
                2; // Source transfer length along Y (K), in units of 32 bytes; valid range: yStep in [0,255].
            loadData2DMxParamsB.srcStride = loadData2DMxParamsB.yStep;
            loadData2DMxParamsB.dstStride =
                loadData2DMxParamsB
                    .yStep; // Stride between consecutive destination fractal starts along X, in units of 32 bytes.
            LoadData(bL0Tensor, bL1Tensor, bScaleL1Tensor, loadData2DParamsB, loadData2DMxParamsB);
        }
    } else if constexpr (IsSameType<T, hifloat4x2_t>::value) {
        if (loadData2DParamsB.ifTranspose) {
            loadData2DMxParamsB.xStartPosition =
                0; // Matrix start along X (M), in fractal units; each unit represents a 32-byte fractal block.
            loadData2DMxParamsB.yStartPosition =
                mmParam.singleN * 2 / HI_FP4_PTG_PCG_SCALE_PARAM; // Source start along Y (K), in units of 32 bytes.
            loadData2DMxParamsB.xStep = (kSplitSize + 15) / 16;   // X/M length in 32-byte fractals; xStep: [0,255].
            loadData2DMxParamsB.yStep =
                (mmParam.singleK + 31) * 2 /
                HI_FP4_PTG_PCG_SCALE_PARAM; // One FP32 value is represented as two BF16 values, doubling the minimum
                                            // fractal count. Source transfer length along Y (K), in units of 32 bytes;
                                            // yStep is in [0, 255].
            loadData2DMxParamsB.srcStride =
                mmParam.singleM * 2 / HI_FP4_PTG_PCG_SCALE_PARAM; // Stride between consecutive source fractal starts
                                                                  // along X, in units of 32 bytes.
            loadData2DMxParamsB.dstStride =
                loadData2DMxParamsB
                    .yStep; // Stride between consecutive destination fractal starts along X, in units of 32 bytes.
            LoadData(bL0Tensor, bL1Tensor, bScaleL1Tensor, loadData2DParamsB, loadData2DMxParamsB);
        } else {
            loadData2DMxParamsB.xStartPosition =
                mmParam.singleK /
                16; // Matrix start along X (M), in fractal units; each unit represents a 32-byte fractal block.
            loadData2DMxParamsB.yStartPosition = 0;                  // Source start along Y (K), in units of 32 bytes.
            loadData2DMxParamsB.xStep = (mmParam.singleN + 15) / 16; // X/M length in 32-byte fractals; xStep: [0,255].
            loadData2DMxParamsB.yStep =
                (kSplitSize + 31) * 2 /
                HI_FP4_PTG_PCG_SCALE_PARAM; // One FP32 value is represented as two BF16 values, doubling the minimum
                                            // fractal count. Source transfer length along Y (K), in units of 32 bytes;
                                            // yStep is in [0, 255].
            loadData2DMxParamsB.srcStride =
                loadData2DMxParamsB
                    .yStep; // Stride between consecutive source fractal starts along X, in units of 32 bytes.
            loadData2DMxParamsB.dstStride =
                loadData2DMxParamsB
                    .yStep; // Stride between consecutive destination fractal starts along X, in units of 32 bytes.
            LoadData(bL0Tensor, bL1Tensor, bScaleL1Tensor, loadData2DParamsB, loadData2DMxParamsB);
        }
    }
}

// Full loading.
// When splitting K externally in L1, pass the cmatrixInitVal flag.
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulFullMX(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                    L0AType& aL0BuffsDb, L0BType& bL0BuffsDb, const LocalTensor<C>& cL0Tensor,
                                    struct MMParam& param,
                                    const LocalTensor<AScaleType>& aScaleL1Tensor = LocalTensor<AScaleType>(),
                                    const LocalTensor<BScaleType>& bScaleL1Tensor = LocalTensor<AScaleType>())
{
    auto l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.template Wait<HardEvent::M_MTE1>();
    LocalTensor<L0ADType> L0ATensor = l0aBuffer.template GetTensor<L0ADType>();
    if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0AMx<A, L0ADType, baseK>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, 0, param.singleK,
                                            param.singleM);
    } else {
        LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM); // s2*d,d,s2
    }

    auto l0bBuffer = bL0BuffsDb.Get();
    LocalTensor<L0BDType> L0BTensor = l0bBuffer.template GetTensor<L0BDType>();
    if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, 0, param.singleK, param.singleN);
    } else {
        LoadDataToL0B(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    }
    l0aBuffer.template Set<HardEvent::MTE1_M>();
    l0aBuffer.template Wait<HardEvent::MTE1_M>();

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    if (param.realM != 0) {
        mmadParams.m = param.realM;
    }
    if (mmadParams.m == 1) {
        mmadParams.m = 16;
    }
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = param.isOutKFisrt;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = param.unitFlag;

    Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

    l0aBuffer.template Set<HardEvent::M_MTE1>();
}
#else
static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
template <typename T, ABLayout AL>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor, const MMParam& mmParam,
                                     uint64_t L1Aoffset, uint32_t kSplitSize, uint32_t mSplitSize)
{
    if constexpr (AL == ABLayout::MK) {
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H = mSplitSize / LOAD3D_L1W_SIZE; // Source operand height.
        loadData3DParams.l1W = LOAD3D_L1W_SIZE;              // Source operand width.
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255; // Tail data does not affect sliding-window results.

        loadData3DParams.mExtension = mSplitSize; // Transfer length along the destination height.
        loadData3DParams.kExtension = kSplitSize; // Transfer length along the destination width.
        loadData3DParams.mStartPt = 0;            // Convolution kernel start along the destination width.
        loadData3DParams.kStartPt = 0;            // Convolution kernel start along the destination height.
        loadData3DParams.strideW = 1;             // Convolution kernel sliding stride along the source width.
        loadData3DParams.strideH = 1;             // Convolution kernel sliding stride along the source height.
        loadData3DParams.filterW = 1;             // Convolution kernel width.
        loadData3DParams.filterSizeW =
            false; // Whether to add 256 elements to the convolution kernel width specified by filterW.
        loadData3DParams.filterH = 1; // Convolution kernel height.
        loadData3DParams.filterSizeH =
            false; // Whether to add 256 elements to the convolution kernel height specified by filterH.
        loadData3DParams.dilationFilterW = 1; // Convolution kernel width dilation factor.
        loadData3DParams.dilationFilterH = 1; // Convolution kernel height dilation factor.
        loadData3DParams.enTranspose = 0;     // Whether to transpose the entire destination matrix.
        loadData3DParams.fMatrixCtrl = 0;
        loadData3DParams.channelSize = kSplitSize; // Source operand channel count; with dilation 1, destination width
                                                   // is filterW*filterH*channelSize.
        LoadData<T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[L1Aoffset], loadData3DParams);
    } else if constexpr (AL == ABLayout::KM) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0; // Fractal matrix ID; transfer starts at fractal block 0 of the source operand.
        loadData2DParams.repeatTimes =
            (kSplitSize / ONE_FRACTAL_H_ELEMENT) *
            (mmParam.singleM /
             (ONE_FRACTAL_W_BYTE / sizeof(T))); // Iteration count; each iteration processes 512 bytes.
        loadData2DParams.srcStride =
            1; // Stride between consecutive source fractal starts across iterations, in units of 512 bytes.
        loadData2DParams.dstGap = 0; // Gap between the end of one destination fractal and the start of the next across
                                     // iterations, in units of 512 bytes.
        loadData2DParams.ifTranspose = true;
        LoadData(aL0Tensor, aL1Tensor[L1Aoffset], loadData2DParams);
    }
}

// L1-to-L0B transfer with split-K, split-N, or full loading.
template <typename T, ABLayout BL>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor, const MMParam& mmParam,
                                     uint64_t L1Boffset, uint32_t kSplitSize, uint32_t nSplitSize)
{
    if constexpr (BL == ABLayout::KN) {
        LoadData3DParamsV2<T> loadData3DParams;
        loadData3DParams.l1H = kSplitSize / LOAD3D_L1W_SIZE; // Source operand height.
        loadData3DParams.l1W = LOAD3D_L1W_SIZE; // Source operand width is 16; destination height is l1H*L1W.
        loadData3DParams.padList[0] = 0;
        loadData3DParams.padList[1] = 0;
        loadData3DParams.padList[2] = 0;
        loadData3DParams.padList[3] = 255; // Tail data does not affect sliding-window results.

        loadData3DParams.mExtension = kSplitSize; // Transfer length along the destination height.
        loadData3DParams.kExtension = nSplitSize; // Transfer length along the destination width.
        loadData3DParams.mStartPt = 0;            // Convolution kernel start along the destination width.
        loadData3DParams.kStartPt = 0;            // Convolution kernel start along the destination height.
        loadData3DParams.strideW = LOAD3D_STRIDE_W;
        loadData3DParams.strideH = LOAD3D_STRIDE_H;
        loadData3DParams.filterW = LOAD3D_FILTER_W;
        loadData3DParams.filterSizeW =
            false; // Whether to add 256 elements to the convolution kernel width specified by filterW.
        loadData3DParams.filterH = LOAD3D_FILTER_H;
        loadData3DParams.filterSizeH =
            false; // Whether to add 256 elements to the convolution kernel height specified by filterH.
        loadData3DParams.dilationFilterW = LOAD3D_DILA_FILTER_W; // Convolution kernel width dilation factor.
        loadData3DParams.dilationFilterH = LOAD3D_DILA_FILTER_H; // Convolution kernel height dilation factor.
        loadData3DParams.enTranspose = 1;                        // Whether to enable transposition.
        loadData3DParams.fMatrixCtrl =
            0; // Select FMATRIX_LEFT or FMATRIX_RIGHT: 0 for FMATRIX_LEFT, 1 for FMATRIX_RIGHT. 1
        loadData3DParams.channelSize = nSplitSize; // Source operand channel count; with dilation 1, destination width
                                                   // is filterW*filterH*channelSize.
        LoadData<T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[L1Boffset], loadData3DParams);
    } else if constexpr (BL == ABLayout::NK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes =
            (nSplitSize + (ONE_FRACTAL_H_ELEMENT - 1)) / ONE_FRACTAL_H_ELEMENT *
            (kSplitSize / (ONE_FRACTAL_W_BYTE / sizeof(T))); // Iteration count; each iteration processes 512 bytes.
        loadData2DParams.srcStride = 1;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;
        LoadData(bL0Tensor, bL1Tensor[L1Boffset], loadData2DParams);
    }
}

#endif
// Full loading.
// When splitting K externally in L1, pass the cmatrixInitVal flag.
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulFull(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor, L0AType& aL0BuffsDb,
                                  L0BType& bL0BuffsDb, const LocalTensor<C>& cL0Tensor, struct MMParam& param,
                                  const LocalTensor<AScaleType>& aScaleL1Tensor = LocalTensor<AScaleType>(),
                                  const LocalTensor<BScaleType>& bScaleL1Tensor = LocalTensor<AScaleType>())
{
    auto l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.template Wait<HardEvent::M_MTE1>();
    LocalTensor<L0ADType> L0ATensor = l0aBuffer.template GetTensor<L0ADType>();
#if ((__CCE_AICORE__ == 310) || (defined __DAV_310R6__) || (__NPU_ARCH__ == 5102))
    if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0AMx<A, L0ADType, baseK>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, 0, param.singleK,
                                            param.singleM); // d,s2
    } else
#endif
    {
        LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM); // s2*d,d,s2
    }
    l0aBuffer.template Set<HardEvent::MTE1_M>();

    auto l0bBuffer = bL0BuffsDb.Get();
    l0bBuffer.template Wait<HardEvent::M_MTE1>();
    LocalTensor<L0BDType> L0BTensor = l0bBuffer.template GetTensor<L0BDType>();
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, 0, param.singleK, param.singleN);
    } else
#endif
    {
        LoadDataToL0B(L0BTensor, bL1Tensor, param, 0, param.singleK, param.singleN);
    }
    l0bBuffer.template Set<HardEvent::MTE1_M>();

    l0aBuffer.template Wait<HardEvent::MTE1_M>();
    l0bBuffer.template Wait<HardEvent::MTE1_M>();

    MmadParams mmadParams;
    mmadParams.m = param.singleM;
    if (param.realM != 0) {
        mmadParams.m = param.realM;
    }
    mmadParams.n = param.singleN;
    mmadParams.k = param.singleK;
    mmadParams.cmatrixInitVal = param.isOutKFisrt;
    mmadParams.cmatrixSource = false;
    mmadParams.unitFlag = param.unitFlag;
    if (mmadParams.m == 1) {
        mmadParams.m = 16;
    }

    Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

    l0aBuffer.template Set<HardEvent::M_MTE1>();
    l0bBuffer.template Set<HardEvent::M_MTE1>();
}

// Split K.
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulK(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor, L0AType& aL0BuffsDb,
                               L0BType& bL0BuffsDb, const LocalTensor<C>& cL0Tensor, const MMParam& param,
                               const LocalTensor<AScaleType>& aScaleL1Tensor = LocalTensor<AScaleType>(),
                               const LocalTensor<BScaleType>& bScaleL1Tensor = LocalTensor<AScaleType>())
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t tailSize = param.singleK % baseK;
    uint32_t tailK = tailSize ? tailSize : baseK;
    uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 : ((param.singleM + 15) >> 4 << 4) * baseK;
    uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 4;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value ||
                  IsSameType<A, hifloat8_t>::value || IsSameType<A, int8_t>::value) {
        L1Aoffset = ((param.singleM + 31) >> 5 << 5) * baseK;
        L1Boffset = ((param.singleN + 31) >> 5 << 5) * baseK;
    }
    if constexpr (IsSameType<A, float>::value) {
        L1Aoffset = param.isLeftTranspose ? baseK << 3 : ((param.singleM + 15) >> 4 << 4) * baseK;
        L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 3;
    }
#endif

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
        auto l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.template Wait<HardEvent::M_MTE1>(); // MTE1 waits for matmul: the previous matmul must finish before
                                                      // new data can be transferred to L0A.
        LocalTensor<L0ADType> L0ATensor = l0aBuffer.template GetTensor<L0ADType>();
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
        if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0AMx<A, L0ADType, baseK>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, k * L1Aoffset, tileK,
                                                param.singleM); // s2,
        } else
#endif
        {
            LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK,
                          param.singleM); // s2*d,d,s2
        }

        auto l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.template Wait<HardEvent::M_MTE1>(); // MTE1 waits for matmul: the previous matmul must finish before
                                                      // new data can be transferred to L0B.
        LocalTensor<L0BDType> L0BTensor = l0bBuffer.template GetTensor<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
        if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, k * L1Boffset, tileK,
                                         param.singleN, loopNum); // tileK.D
        } else
#endif
        {
            LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
        }
        l0bBuffer
            .template Set<HardEvent::MTE1_M>(); // After MTE1 finishes transferring data, signal that matmul can start.
        // l0aBuffer and l0bBuffer share MTE1_M to reduce synchronization instructions and improve performance for
        // D=512.
        l0bBuffer
            .template Wait<HardEvent::MTE1_M>(); // Matmul waits for MTE1: start only after the L0B transfer completes.

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) { // GEMV is enabled for m=1 or by default. The documentation does not describe disabling
                                 // it, so use matrix operations as a workaround.
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                      UNITFLAG_EN_OUTER_LAST :
                                      UNITFLAG_ENABLE;
        }
        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);

        l0aBuffer.template Set<HardEvent::M_MTE1>(); // After matmul completes, notify MTE1 that new data can be
                                                     // transferred to L0A.
        l0bBuffer.template Set<HardEvent::M_MTE1>(); // After matmul completes, notify MTE1 that new data can be
                                                     // transferred to L0B.
    }
}

// Split-K implementation for INT8 with bias.
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType>
__aicore__ inline void MatmulKbias(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                   L0AType& aL0BuffsDb, L0BType& bL0BuffsDb, const LocalTensor<int32_t>& cL0Tensor,
                                   const LocalTensor<int32_t>& biasTensor, const MMParam& param)
{
    uint32_t kLoops =
        (param.singleK + baseK - 1) / baseK; // Tail handling: if dSize is not divisible by 256, add one iteration. The
                                             // tail is always considered relative to one basic block.
    uint32_t tailSize =
        param.singleK %
        baseK; // Split dSize into chunks of 256; if it is not divisible, name the last chunk's dSize tailSize.
    uint32_t tailK = tailSize ? tailSize : baseK; // K of the tail block split along the K axis.
    uint64_t L1Aoffset = param.isLeftTranspose ? baseK << 4 :
                                                 ((param.singleM + 15) >> 4 << 4) *
                                                     baseK; // Align the input s1realsize to a multiple of 16.
    uint64_t L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK :
                                                  baseK << 4; // Align the input s2realsize to a multiple of 16.
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value ||
                  IsSameType<A, hifloat8_t>::value || IsSameType<A, int8_t>::value) {
        L1Aoffset = ((param.singleM + 31) >> 5 << 5) * baseK; // Align the input s1realsize to a multiple of 32.
        L1Boffset = ((param.singleN + 31) >> 5 << 5) * baseK; // Align the input s2realsize to a multiple of 32.
    }
    if constexpr (IsSameType<A, float>::value) {
        L1Aoffset = param.isLeftTranspose ? baseK << 3 : ((param.singleM + 15) >> 4 << 4) * baseK;
        L1Boffset = param.isRightTranspose ? ((param.singleN + 15) >> 4 << 4) * baseK : baseK << 3;
    }
#endif

    for (uint32_t k = 0; k < kLoops; k++) {
        uint32_t tileK = (k == (kLoops - 1)) ? tailK : baseK;
        auto l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.template Wait<HardEvent::M_MTE1>(); // MTE1 waits for matmul: the previous matmul must finish before
                                                      // new data can be transferred to L0A.
        LocalTensor<A> L0ATensor = l0aBuffer.template GetTensor<A>();
        LoadDataToL0A(L0ATensor, aL1Tensor, param, k * L1Aoffset, tileK, param.singleM);
        l0aBuffer
            .template Set<HardEvent::MTE1_M>(); // After MTE1 finishes transferring data, signal that matmul can start.

        auto l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.template Wait<HardEvent::M_MTE1>(); // MTE1 waits for matmul: the previous matmul must finish before
                                                      // new data can be transferred to L0B.
        LocalTensor<B> L0BTensor = l0bBuffer.template GetTensor<B>();
        uint64_t loopNum = param.isRightTranspose ? 1 : kLoops;
        LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, tileK, param.singleN, loopNum);
        l0bBuffer
            .template Set<HardEvent::MTE1_M>(); // After MTE1 finishes transferring data, signal that matmul can start.

        l0aBuffer
            .template Wait<HardEvent::MTE1_M>(); // Matmul waits for MTE1: start only after the L0A transfer completes.
        l0bBuffer
            .template Wait<HardEvent::MTE1_M>(); // Matmul waits for MTE1: start only after the L0B transfer completes.

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = param.singleN;
        mmadParams.k = tileK;
        if (mmadParams.m == 1) { // GEMV is enabled for m=1 or by default. The documentation does not describe disabling
                                 // it, so use matrix operations as a workaround.
            mmadParams.m = 16;
        }
        mmadParams.cmatrixInitVal = false;
        mmadParams.cmatrixSource = (k == 0);
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                      UNITFLAG_EN_OUTER_LAST :
                                      UNITFLAG_ENABLE;
        }

        if (k == 0) {
            Mmad(cL0Tensor, L0ATensor, L0BTensor, biasTensor, mmadParams);
        } else {
            Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);
        }

        l0aBuffer.template Set<HardEvent::M_MTE1>(); // After matmul completes, notify MTE1 that new data can be
                                                     // transferred to L0A.
        l0bBuffer.template Set<HardEvent::M_MTE1>(); // After matmul completes, notify MTE1 that new data can be
                                                     // transferred to L0B.
    }
}

// Split N.
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType, typename AScaleType = fp8_e8m0_t, typename BScaleType = fp8_e8m0_t,
          typename L0ADType = A, typename L0BDType = B>
__aicore__ inline void MatmulN(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor, L0AType& aL0BuffsDb,
                               L0BType& bL0BuffsDb, const LocalTensor<C>& cL0Tensor, const MMParam& param,
                               const LocalTensor<AScaleType>& aScaleL1Tensor = LocalTensor<AScaleType>(),
                               const LocalTensor<BScaleType>& bScaleL1Tensor = LocalTensor<AScaleType>())
{
    uint32_t nLoops = (param.singleN + baseN - 1) / baseN; // Tail block handling.
    uint32_t tailSize = param.singleN % baseN;
    uint32_t tailN = tailSize ? tailSize : baseN;
    uint64_t L1Boffset = param.isRightTranspose ? (baseN << 4) : ((param.singleK + 15) >> 4 << 4) * baseN;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<A, fp8_e5m2_t>::value || IsSameType<A, fp8_e4m3fn_t>::value ||
                  IsSameType<A, hifloat8_t>::value || IsSameType<A, int8_t>::value) {
        L1Boffset = ((param.singleK + 31) >> 5 << 5) * baseN;
    }
#endif
    uint64_t L0Coffset = ((param.singleM + 15) >> 4 << 4) * baseN;
    if (param.realM != 0) {
        L0Coffset = ((param.realM + 15) >> 4 << 4) * baseN;
    }

    auto l0aBuffer = aL0BuffsDb.Get();
    l0aBuffer.template Wait<HardEvent::M_MTE1>(); // MTE1 waits for matmul: the previous matmul must finish before new
                                                  // data can be transferred to L0A.
    LocalTensor<L0ADType> L0ATensor = l0aBuffer.template GetTensor<L0ADType>();
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
    if constexpr (IsSameType<L0ADType, mx_fp8_e4m3_t>::value) {
        LoadDataToL0AMx<A, L0ADType, baseK>(L0ATensor, aL1Tensor, aScaleL1Tensor, param, 0, param.singleK,
                                            param.singleM); // d,s2
    } else
#endif
    {
        LoadDataToL0A(L0ATensor, aL1Tensor, param, 0, param.singleK, param.singleM); // s2*d,d,s2
    }
    for (uint32_t n = 0; n < nLoops; n++) {
        uint32_t tileN = (n == (nLoops - 1)) ? tailN : baseN;

        auto l0bBuffer = bL0BuffsDb.Get();
        l0bBuffer.template Wait<HardEvent::M_MTE1>(); // MTE1 waits for matmul: the previous matmul must finish before
                                                      // new data can be transferred to L0B.
        LocalTensor<L0BDType> L0BTensor = l0bBuffer.template GetTensor<L0BDType>();
        uint64_t loopNum = param.isRightTranspose ? nLoops : 1;
#if (__CCE_AICORE__ == 310) || (defined __DAV_310R6__)
        if constexpr (IsSameType<L0BDType, mx_fp8_e4m3_t>::value) {
            LoadDataToL0BMx<B, L0BDType>(L0BTensor, bL1Tensor, bScaleL1Tensor, param, n * L1Boffset, param.singleK,
                                         tileN, loopNum); // tileK.D
        } else
#endif
        {
            LoadDataToL0B(L0BTensor, bL1Tensor, param, n * L1Boffset, param.singleK, tileN, loopNum);
        }
        l0bBuffer
            .template Set<HardEvent::MTE1_M>(); // After MTE1 finishes transferring data, signal that matmul can start.
        // l0aBuffer and l0bBuffer share MTE1_M to reduce synchronization instructions and improve performance for
        // D=512.
        l0bBuffer
            .template Wait<HardEvent::MTE1_M>(); // Matmul waits for MTE1: start only after the L0B transfer completes.

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        if (param.realM != 0) {
            mmadParams.m = param.realM;
        }
        mmadParams.n = tileN;
        mmadParams.k = param.singleK;
        if (mmadParams.m == 1) {
            mmadParams.m = FP16_ONE_FRACTAL_ELEMENT;
        }
        mmadParams.cmatrixInitVal = param.isOutKFisrt;
        mmadParams.cmatrixSource = false;
        mmadParams.unitFlag = param.unitFlag;
        Mmad(cL0Tensor[n * L0Coffset], L0ATensor, L0BTensor, mmadParams);

        l0bBuffer.template Set<HardEvent::M_MTE1>(); // After matmul completes, notify MTE1 that new data can be
                                                     // transferred to L0B.
    }
    l0aBuffer.template Set<HardEvent::M_MTE1>(); // After matmul completes, notify MTE1 that new data can be transferred
                                                 // to L0A.
}

// Split M.
template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulKM(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                BuffersPolicyDB<BufferType::L0A>& aL0BuffsDb,
                                BuffersPolicyDB<BufferType::L0B>& bL0BuffsDb, const LocalTensor<C>& cL0Tensor,
                                struct MMParam& param)
{
    uint32_t mLoops = (param.singleM + baseM - 1) / baseM; // Tail block handling.
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK; // Tail block handling.
    uint32_t mSplitSize = (mLoops == 1) ? param.singleM : baseM;
    uint32_t mplitTailSize = (param.singleM % baseM) ? (param.singleM % baseM) : mSplitSize;
    uint32_t kSplitSize = (kLoops == 1) ? param.singleK : baseK;
    uint32_t kSplitTailSize = (param.singleK % baseK) ? (param.singleK % baseK) : kSplitSize;
    uint64_t L1Boffset = kSplitSize * param.singleN;
    uint64_t L0Coffset = mSplitSize * param.singleN;

    for (uint32_t k = 0; k < kLoops; k++) {
        kSplitSize = (k == (kLoops - 1)) ? kSplitTailSize : kSplitSize;
        for (uint32_t m = 0; m < mLoops; m++) {
            mSplitSize = (m == (mLoops - 1)) ? mplitTailSize : mSplitSize;
            auto l0aBuffer = aL0BuffsDb.Get();
            l0aBuffer.template Wait<HardEvent::M_MTE1>(); // Acquire.
            LocalTensor<A> L0ATensor = l0aBuffer.template GetTensor<A>();
            LoadDataToL0A(L0ATensor, aL1Tensor, param, k * param.singleM * kSplitSize + m * kSplitSize * mSplitSize,
                          kSplitSize, mSplitSize);
            l0aBuffer.template Set<HardEvent::MTE1_M>();  // Notify.
            l0aBuffer.template Wait<HardEvent::MTE1_M>(); // Wait for L0A.

            auto l0bBuffer = bL0BuffsDb.Get();
            l0bBuffer.template Wait<HardEvent::M_MTE1>(); // Acquire.
            LocalTensor<B> L0BTensor = l0bBuffer.template GetTensor<B>();
            LoadDataToL0B(L0BTensor, bL1Tensor, param, k * L1Boffset, kSplitSize, param.singleN);
            l0bBuffer.template Set<HardEvent::MTE1_M>();  // Notify.
            l0bBuffer.template Wait<HardEvent::MTE1_M>(); // Wait for L0B.

            MmadParams mmadParams;
            mmadParams.m = mSplitSize;
            mmadParams.n = param.singleN;
            mmadParams.k = kSplitSize;
            mmadParams.cmatrixInitVal =
                param.isOutKFisrt && (k == 0); // Whether the initial C matrix value is zero; defaults to true.
            mmadParams.cmatrixSource = false;  //  From CO1.
            Mmad(cL0Tensor[m * L0Coffset], L0ATensor, L0BTensor, mmadParams);

            l0aBuffer.template Set<HardEvent::M_MTE1>(); // Release L0A.
            l0bBuffer.template Set<HardEvent::M_MTE1>(); // Release L0B.
        }
    }
}

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL,
          typename L0AType, typename L0BType>
__aicore__ inline void MatmulBase(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor, L0AType& aL0BuffsDb,
                                  L0BType& bL0BuffsDb, const LocalTensor<C>& cL0Tensor, struct MMParam& param)
{
    if ((param.singleK + baseK - 1) / baseK > 1) {
        MatmulK<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    } else if ((param.singleN + baseN - 1) / baseN > 1) {
        MatmulN<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor, param);
    } else {
        MatmulFull<A, B, C, baseM, baseN, baseK, AL, BL>(aL1Tensor, bL1Tensor, aL0BuffsDb, bL0BuffsDb, cL0Tensor,
                                                         param);
    }
}

template <typename A, typename B, typename C, uint32_t baseM, uint32_t baseN, uint32_t baseK, ABLayout AL, ABLayout BL>
__aicore__ inline void MatmulKPP(const LocalTensor<A>& aL1Tensor, const LocalTensor<B>& bL1Tensor,
                                 BuffersPolicyDB<BufferType::L0A>& aL0BuffsDb,
                                 BuffersPolicyDB<BufferType::L0B>& bL0BuffsDb, const LocalTensor<C>& cL0Tensor,
                                 const MMParam& param)
{
    uint32_t kLoops = (param.singleK + baseK - 1) / baseK;
    uint32_t kSplitSize = (kLoops == 1) ? param.singleK : baseK;
    uint32_t kSplitSizeAlign = AlignUp(kSplitSize, FP16_ONE_FRACTAL_ELEMENT);
    uint64_t L1Aoffset = AlignUp(param.singleM, FP16_ONE_FRACTAL_ELEMENT) * kSplitSize;
    uint64_t L1Boffset = AlignUp(param.singleN, FP16_ONE_FRACTAL_ELEMENT) * kSplitSize;
    for (uint32_t k = 0; k < kLoops; k++) {
        if (k == kLoops - 1) {
            kSplitSize = (param.singleK % baseK) ? (param.singleK % baseK) : kSplitSize;
            kSplitSizeAlign = AlignUp(kSplitSize, FP16_ONE_FRACTAL_ELEMENT);
        }
        Buffer<BufferType::L0A> l0aBuffer = aL0BuffsDb.Get();
        l0aBuffer.Wait<HardEvent::M_MTE1>(); // MTE1 waits for matmul: the previous matmul must finish before new data
                                             // can be transferred to L0A.
        LocalTensor<A> L0ATensor = l0aBuffer.GetTensor<A>();
        LoadDataToL0A<A, AL>(L0ATensor, aL1Tensor, param, k * L1Aoffset, kSplitSizeAlign, param.singleM);
        Buffer<BufferType::L0B> l0bBuffer = bL0BuffsDb.Get();
        LocalTensor<B> L0BTensor = l0bBuffer.GetTensor<B>();
        LoadDataToL0B<B, BL>(L0BTensor, bL1Tensor, param, k * L1Boffset, kSplitSizeAlign, param.singleN);

        l0aBuffer.Set<HardEvent::MTE1_M>();  // After MTE1 finishes transferring data, signal that matmul can start.
        l0aBuffer.Wait<HardEvent::MTE1_M>(); // Matmul waits for MTE1: start only after the L0A transfer completes.

        MmadParams mmadParams;
        mmadParams.m = param.singleM;
        mmadParams.n = param.singleN;
        mmadParams.k = kSplitSize;
        if (mmadParams.m ==
            1) { // For m=1, GEMV is enabled by default and cannot be disabled; use matrix operations as a workaround.
            mmadParams.m = FP16_ONE_FRACTAL_ELEMENT;
        }
        mmadParams.cmatrixInitVal = (param.isOutKFisrt == true) && (k == 0);
        mmadParams.cmatrixSource = false;
        if (param.unitFlag != 0) {
            mmadParams.unitFlag = (param.unitFlag == UNITFLAG_EN_OUTER_LAST) && (k == kLoops - 1) ?
                                      UNITFLAG_EN_OUTER_LAST :
                                      UNITFLAG_ENABLE;
        }

        Mmad(cL0Tensor, L0ATensor, L0BTensor, mmadParams);
#if (__CCE_AICORE__ != 310) && (!(defined __DAV_310R6__))
        if ((mmadParams.m / FP16_ONE_FRACTAL_ELEMENT) * (mmadParams.n / FP16_ONE_FRACTAL_ELEMENT) < MMAD_MN_SIZE_10) {
            AscendC::PipeBarrier<PIPE_M>();
        }
#endif
        l0aBuffer
            .Set<HardEvent::M_MTE1>(); // After matmul completes, notify MTE1 that new data can be transferred to L0A.
    }
}
template <typename T, ABLayout AL>
__aicore__ inline void LoadDataToL0A(LocalTensor<T>& aL0Tensor, const LocalTensor<T>& aL1Tensor, uint32_t rowSize,
                                     uint32_t kSplitSize, uint32_t mSplitSize)
{
    uint32_t blockElementCnt = ONE_FRACTAL_W_BYTE / sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        blockElementCnt = INT4_ONE_FRACTAL_ELEMENT;
    }
    if constexpr (AL == ABLayout::MK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0; // Fractal matrix ID; transfer starts at fractal block 0 of the source operand.
        loadData2DParams.srcStride =
            1; // Stride between consecutive source fractal starts across iterations, in units of 512 bytes.
        loadData2DParams.dstGap =
            kSplitSize / blockElementCnt - 1; // Gap between the end of one destination fractal and the start of the
                                              // next across iterations, in units of 512 bytes.
        loadData2DParams.repeatTimes =
            mSplitSize / ONE_FRACTAL_H_ELEMENT; // Iteration count; each iteration processes 512 bytes.
        loadData2DParams.ifTranspose = false;
        uint32_t loopTimes = kSplitSize / blockElementCnt;
        uint64_t l1Offset = rowSize * blockElementCnt;
        uint64_t l0Offset = ONE_FRACTAL_H_ELEMENT * blockElementCnt;
        for (uint32_t loop = 0; loop < loopTimes; loop++) {
            LoadData(aL0Tensor[loop * l0Offset], aL1Tensor[loop * l1Offset], loadData2DParams);
        }
    } else if constexpr (AL == ABLayout::KM) {
        LoadData2dTransposeParams loadData2dTransposeParams;
        loadData2dTransposeParams.startIndex = 0;
        loadData2dTransposeParams.srcStride = 1;
        loadData2dTransposeParams.dstFracGap = (kSplitSize + blockElementCnt - 1) / blockElementCnt;
        loadData2dTransposeParams.dstGap = mSplitSize / ONE_FRACTAL_H_ELEMENT - 1;
        if (rowSize == kSplitSize) {
            loadData2dTransposeParams.repeatTimes = (kSplitSize + blockElementCnt - 1) / blockElementCnt;
            uint32_t loopTimes = mSplitSize / blockElementCnt;
            uint64_t l1Offset = rowSize * blockElementCnt;
            uint64_t l0Offset = kSplitSize * blockElementCnt;
            for (uint32_t loop = 0; loop < loopTimes; loop++) {
                LoadDataWithTranspose(aL0Tensor[loop * l0Offset], aL1Tensor[loop * l1Offset],
                                      loadData2dTransposeParams);
            }
        } else {
            loadData2dTransposeParams.repeatTimes =
                ((kSplitSize + blockElementCnt - 1) / blockElementCnt) * (mSplitSize / blockElementCnt);
            LoadDataWithTranspose(aL0Tensor, aL1Tensor, loadData2dTransposeParams);
        }
    }
}

template <typename T, ABLayout BL>
__aicore__ inline void LoadDataToL0B(LocalTensor<T>& bL0Tensor, const LocalTensor<T>& bL1Tensor, uint32_t rowSize,
                                     uint32_t kSplitSize, uint32_t nSplitSize)
{
    uint32_t blockElementCnt = ONE_FRACTAL_W_BYTE / sizeof(T);
    if constexpr (IsSameType<T, int4b_t>::value) {
        blockElementCnt = INT4_ONE_FRACTAL_ELEMENT;
    }
    if constexpr (BL == ABLayout::KN) {
        LoadData2dTransposeParams loadData2dTransposeParams;

        loadData2dTransposeParams.startIndex = 0;
        loadData2dTransposeParams.srcStride = 1;
        loadData2dTransposeParams.dstFracGap = 0;
        loadData2dTransposeParams.dstGap = nSplitSize / ONE_FRACTAL_H_ELEMENT - 1;
        loadData2dTransposeParams.repeatTimes = (kSplitSize + blockElementCnt - 1) / blockElementCnt;

        uint32_t loopTimes = nSplitSize / blockElementCnt;
        uint64_t l1Offset = rowSize * blockElementCnt;
        uint64_t l0Offset = blockElementCnt * blockElementCnt;
        for (uint32_t loop = 0; loop < loopTimes; loop++) {
            LoadDataWithTranspose(bL0Tensor[loop * l0Offset], bL1Tensor[loop * l1Offset], loadData2dTransposeParams);
        }
    } else if constexpr (BL == ABLayout::NK) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0; // Fractal matrix ID; transfer starts at fractal block 0 of the source operand.
        loadData2DParams.srcStride =
            1; // Stride between consecutive source fractal starts across iterations, in units of 512 bytes.
        loadData2DParams.dstGap = 0; // Gap between the end of one destination fractal and the start of the next across
                                     // iterations, in units of 512 bytes.
        loadData2DParams.ifTranspose = false;
        if (rowSize == kSplitSize) {
            loadData2DParams.repeatTimes =
                ((nSplitSize + ONE_FRACTAL_H_ELEMENT - 1) / ONE_FRACTAL_H_ELEMENT) *
                (kSplitSize / blockElementCnt); // Iteration count; each iteration processes 512 bytes.
            LoadData(bL0Tensor, bL1Tensor, loadData2DParams);
        } else {
            loadData2DParams.repeatTimes =
                (nSplitSize + ONE_FRACTAL_H_ELEMENT - 1) /
                ONE_FRACTAL_H_ELEMENT; // Iteration count; each iteration processes 512 bytes.
            uint32_t loopTimes = kSplitSize / blockElementCnt;
            uint64_t l1Offset = nSplitSize * blockElementCnt;
            uint64_t l0Offset = rowSize * blockElementCnt;
            for (uint32_t loop = 0; loop < loopTimes; loop++) {
                LoadData(bL0Tensor[loop * l0Offset], bL1Tensor[loop * l1Offset], loadData2DParams);
            }
        }
    }
}
} // namespace fa_base_matmul
#endif
