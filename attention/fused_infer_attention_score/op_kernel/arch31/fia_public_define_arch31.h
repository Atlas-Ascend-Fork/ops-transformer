/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file fia_public_define_arch31.h
 * \brief arch31 (KirinX90 / Kirin9030) 公共定义：FIAtype / ConstInfo / RunInfo / FIA_LAYOUT 等
 */
#ifndef FIA_PUBLIC_DEFINE_ARCH31_H
#define FIA_PUBLIC_DEFINE_ARCH31_H

#include "kernel_operator.h"

#if __has_include("../../../common/op_kernel/const_def.h")
#include "../../../common/op_kernel/const_def.h"
#elif __has_include("../../common/const_def.h")
#include "../../common/const_def.h"
#else
#error "const_def.h not found in either tree layout"
#endif

using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::WaitFlag;

namespace AttentionCommonArch31 {
constexpr uint32_t FIA_MAX_AIC_CORE_NUM_ARCH31 = 1;
constexpr uint32_t ARCH31_VECTOR_REG_WIDTH = 256;
constexpr uint32_t ARCH31_PER_VEG_NUM_FP32 = ARCH31_VECTOR_REG_WIDTH / sizeof(float);
constexpr uint32_t ARCH31_PER_VEG_NUM_FP16 = ARCH31_VECTOR_REG_WIDTH / sizeof(half);
constexpr uint32_t ARCH31_CUBE_M = 16;
constexpr uint32_t ARCH31_CUBE_N = 16;
constexpr uint32_t ARCH31_CUBE_K = 16;

// UB 切分基础参数
constexpr uint32_t ARCH31_M_BASE_DEFAULT = 32;
constexpr uint32_t ARCH31_M_BASE_D512 = 16;
constexpr uint32_t ARCH31_N_BASE_UB = 128;
constexpr uint32_t ARCH31_D_INNER_DEFAULT = 256;

// 单核单 kernel EVENT_ID 分配
constexpr uint8_t ARCH31_EVENT_ID_MM1 = 0;   // Cube 写GM mm1Res -> Vector 读 GM mm1Res
constexpr uint8_t ARCH31_EVENT_ID_P = 1;     // Vector 写 GM softmaxP_fp16 -> Cube 读 GM softmaxP_fp16
constexpr uint8_t ARCH31_EVENT_ID_MM2 = 2;   // Cube 写GM mm2Res -> Vector 读 GM mm2Res
constexpr uint8_t ARCH31_EVENT_ID_MASK = 3;  // attenMask GM -> UB (MTE2_V)
constexpr uint8_t ARCH31_EVENT_ID_L0AB = 3;  // MTE1 装载 L0A/B <-> Mmad 消费
constexpr uint8_t ARCH31_EVENT_ID_L1_QP = 4; // MTE2<->MTE1：qpBufL1（Q/P） GM->L1->L0
constexpr uint8_t ARCH31_EVENT_ID_L1_KV = 5; // MTE2<->MTE1：kvBufL1（K/V） GM->L1->L0
constexpr uint8_t ARCH31_EVENT_ID_L0C = 4;   // M<->FIX

constexpr uint8_t ARCH31_EVENT_ID_OUT = 2;
constexpr uint8_t ARCH31_EVENT_ID_MTE3_DRAIN = 0;
constexpr uint8_t ARCH31_EVENT_ID_KV_TAIL = 4;

constexpr uint8_t ARCH31_EVENT_ID_MM1READY_P0 = 0;
constexpr uint8_t ARCH31_EVENT_ID_MM1READY_P1 = 1;
constexpr uint8_t ARCH31_EVENT_ID_MM2READY_P0 = 2;
constexpr uint8_t ARCH31_EVENT_ID_MM2READY_P1 = 3;

constexpr uint8_t ARCH31_EVENT_ID_MM1FREE_P0 = 0;
constexpr uint8_t ARCH31_EVENT_ID_MM1FREE_P1 = 1;
constexpr uint8_t ARCH31_EVENT_ID_MM2FREE_P0 = 2;
constexpr uint8_t ARCH31_EVENT_ID_MM2FREE_P1 = 3;

constexpr uint8_t ARCH31_EVENT_ID_PL1RDY = 5;
constexpr uint8_t ARCH31_EVENT_ID_PL1FREE_P0 = 0;
constexpr uint8_t ARCH31_EVENT_ID_PL1FREE_P1 = 1;

__aicore__ inline uint32_t Arch31NDealPad(uint32_t nDeal)
{
    return (nDeal + 15U) & ~15U;
}

constexpr uint8_t ARCH31_EVENT_ID_FIX_DRAIN = 0;

enum class FIA_LAYOUT_ARCH31 : uint32_t {
    BSH = 0,
    BSND = 4,
    BNSD = 1,
    NZ = 2,
    TND = 3,
};

constexpr uint32_t ARCH31_TND_MAX_BATCH = 128;

constexpr uint32_t ARCH31_HP_PATH_NONE = 0;
constexpr uint32_t ARCH31_HP_PATH_PREFILL = 1;
constexpr uint32_t ARCH31_HP_PATH_DECODE = 2;

template <typename Q_T, typename KV_T, typename OUT_T, FIA_LAYOUT_ARCH31 LAYOUT_T = FIA_LAYOUT_ARCH31::BNSD,
          FIA_LAYOUT_ARCH31 KV_LAYOUT_T = FIA_LAYOUT_ARCH31::BNSD, uint32_t HEAD_DIM_T = 0U>
struct FIATypeArch31 {
    using queryType = Q_T;
    using kvType = KV_T;
    using outputType = OUT_T;
    static constexpr FIA_LAYOUT_ARCH31 layout = LAYOUT_T;
    static constexpr FIA_LAYOUT_ARCH31 kvLayout = KV_LAYOUT_T;
    static constexpr uint32_t headDim = HEAD_DIM_T;
    static constexpr bool isHp = (HEAD_DIM_T != 0U);
};

// arch31 运行时信息（精简版，去除 arch22 多核字段）
struct RunInfoArch31 {
    bool isValid = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;
    bool isGlobalLastTile = false;

    uint32_t bIdx = 0;
    uint32_t hIdx = 0;
    uint32_t kvHeadIdx = 0;
    uint32_t s1Tile = 0;
    uint32_t s2Tile = 0;
    uint32_t mDeal = 0;
    uint32_t nDeal = 0;
    uint32_t dInner = 0;

    uint64_t qGmOffset = 0;
    uint64_t kGmOffset = 0;
    uint64_t vGmOffset = 0;
    uint64_t pGmOffset = 0;
    uint64_t mm1ResGmOffset = 0;
    uint64_t mm2ResGmOffset = 0;
    uint64_t outGmOffset = 0;
    uint64_t attenMaskGmOffset = 0;

    uint32_t maskRowWidth = 0;
    uint32_t s2Stride = 0;
};

// arch31 常量信息（精简版，去除 arch22 多核字段）
struct ConstInfoArch31 {
    float scaleValue = 0;
    uint64_t batchSize = 0;
    uint64_t gSize = 0;
    uint64_t qHeadNum = 0;
    uint64_t kvHeadNum = 0;
    uint64_t headDim = 0;
    uint64_t qSeqSize = 0;
    uint64_t kvSeqSize = 0; // kv最大S长度

    uint32_t mBaseSize = ARCH31_M_BASE_DEFAULT;
    uint32_t nBaseSize = ARCH31_N_BASE_UB;
    uint32_t dInner = ARCH31_D_INNER_DEFAULT;
    uint32_t nBaseUb = ARCH31_N_BASE_UB;

    uint32_t sparseMode = 0;
    int64_t preTokens = 0;  // sparseMode=4
    int64_t nextTokens = 0; // sparseMode=4
    bool attenMaskFlag = false;
    bool attenMaskTensorFlag = false;
    uint32_t attenMaskBatchStride = 0;
    bool softmaxLseFlag = false;

    uint64_t mm1ResGmSize = 0;
    uint64_t softmaxPFp16GmSize = 0;
    uint64_t mm2ResGmSize = 0;

    FIA_LAYOUT_ARCH31 outputLayout = FIA_LAYOUT_ARCH31::BNSD;

    // 切 D 分档字段
    bool needSplitD = false;
    uint32_t dTile = 0;
    uint32_t kTileQK = 0;
    uint32_t nTileV = 0;
    uint32_t mBaseDeal = 0;
    uint32_t nDealMax = 0;
    uint32_t kTileV = 0;

    int32_t sparseOffset = 0;
    int32_t bandUpperDelta = 0;
    int32_t bandLowerDelta = 0;

    uint32_t maskMDeal = 0;
};

} // namespace AttentionCommonArch31

#endif // FIA_PUBLIC_DEFINE_ARCH31_H
