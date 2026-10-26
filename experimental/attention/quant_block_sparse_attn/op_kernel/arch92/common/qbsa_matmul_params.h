/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_ARCH92_QBSA_MATMUL_PARAMS_H
#define QBSA_ARCH92_QBSA_MATMUL_PARAMS_H

namespace fa_base_matmul {
constexpr uint32_t UNITFLAG_ENABLE = 2;
constexpr uint32_t UNITFLAG_EN_OUTER_LAST = 3;
static constexpr uint32_t FP16_ONE_FRACTAL_ELEMENT =
    16; // One fractal block is 512 bytes, containing 16*16 FP16 elements.
static constexpr uint32_t INT4_ONE_FRACTAL_ELEMENT =
    64;                                               // One fractal block is 512 bytes, containing 16*64 FP16 elements.
static constexpr uint32_t ONE_FRACTAL_H_ELEMENT = 16; // One fractal block is 512 bytes, with 16 elements along height.
static constexpr uint32_t ONE_FRACTAL_W_BYTE = 32;    // One fractal block is 512 bytes, with 32 bytes along width.
static constexpr uint32_t LOAD3D_L1W_SIZE = 16;
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
    uint32_t singleM{};
    uint32_t singleN{};
    uint32_t singleK{};
    bool isLeftTranspose{};
    bool isRightTranspose{};
    bool cmatrixInitVal = true;
    bool isOutKFisrt = true; // Defaults to true; when splitting K in L1, true indicates the first K iteration.
    uint32_t unitFlag =
        0; // 0: disabled; do not configure unitFlag.
           // 2: enabled; the split-K interface (MatmulK) sets mmadParams.unitFlag to 0b10.
           // 3: enabled; on the last K iteration, the split-K interface (MatmulK) sets mmadParams.unitFlag to
           // 0b11. External callers should set this parameter to 3 on the last iteration of the outer K loop.
    uint32_t realM = 0; // BMM2 uses s1realsize as M; leaving this unset does not affect existing code behavior.
};

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
} // namespace fa_base_matmul

#endif // QBSA_ARCH92_QBSA_MATMUL_PARAMS_H
