/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_MXFP8_RUNTIME_H_
#define QBSA_MXFP8_RUNTIME_H_

#include <cstdint>
#include "../../common/util_regbase_QBSA.h"
#include "qbsa_mxfp8_layout.h"

namespace regbasemx {
constexpr uint32_t QBSA_MX_SCALE_LAST_DIM = 2U;
constexpr uint32_t QBSA_MX_S2_BASE_SIZE = 512U;
constexpr uint32_t QBSA_MX_MIN_KV_BLOCK_SIZE = 64U;
constexpr uint32_t QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK = QBSA_MX_S2_BASE_SIZE / QBSA_MX_MIN_KV_BLOCK_SIZE;

// Runtime state for one (B,N1,S1 block,512-token logical S2 tile).
struct MxRunInfo {
    // Pipeline state.
    uint32_t loop = 0U;
    uint32_t mLoop = 0U;
    bool isValid = false;
    bool isFirstS2Loop = false;
    bool isLastS2Loop = false;
    // Bits 0/1 indicate whether 256-token subLoops 0/1 require an actual causal mask.
    uint8_t attenMaskSubLoopBits = 0U;

    // Batch/head/query coordinates.
    uint32_t bIdx = 0U;
    // KV head index.
    uint32_t n2Idx = 0U;
    uint32_t s1Idx = 0U;
    uint32_t realN2Idx = 0U;

    // Actual sequence lengths and the M-axis split between two AIV cores.
    uint64_t actS1Size = 1U;
    uint64_t actS2Size = 1U;
    uint32_t actMSize = 0U;
    uint32_t actVecMSize = 0U;
    uint32_t vecMbaseIdx = 0U;

    // Valid length of the concatenated logical S2 tile.
    uint32_t actSingleLoopS2Size = 0U;

    // Sparse block indices and concatenation information; up to 8 blocks per task.
    uint32_t sparseBlockCount = 0U;
    // Valid block IDs in ascending order, with negative indices at the end.
    int64_t sparseBlockIdx[QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK] = {};
    // Original KV token start for PA/mask addressing.
    uint64_t sparseBlockTokenOffset[QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK] = {};
    // Concatenated tile start shared by K/V/scale/mask.
    uint32_t sparseBlockTileOffset[QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK] = {};
    uint32_t sparseBlockRealSize[QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK] = {};
    // True for a causal boundary crossing or a fully invisible block, requiring a compressed mask transfer.
    bool sparseBlockPartialMask[QBSA_MX_MAX_SPARSE_BLOCK_PER_TASK] = {};

    // TND GM offsets for Q/QScale.
    uint64_t queryTokenBase = 0U;
    uint64_t queryOffset = 0U;
    uint64_t queryScaleOffset = 0U;
};

// Base shapes and TND strides; scale/data offsets are in elements by default.
struct MxBaseConstInfo {
    uint32_t dSize = 0U;
    uint32_t dSizeV = 0U;
    uint32_t qSeqSize = 0U;
    uint64_t queryRowStride = 0U;
    uint64_t queryScaleRowStride = 0U;
    uint32_t gSize = 0U;
    uint32_t n2Size = 0U;
    uint32_t realN2Size = 0U;
    uint32_t coreNum = 0U;
    uint32_t aicIdx = 0U;
    uint8_t subBlockIdx = 0U;
    uint32_t n2GD = 0U;
    uint32_t qScaleN1D = 0U;
    // Stride of one KV head in PA BNBD scales.
    uint32_t kScaleN2D = 0U;
    uint32_t valueScaleN2D = 0U;
    uint32_t attentionOutStride = 0U;
    uint32_t softmaxLseStride = 0U;
    float scaleValue = 0.0F;
};

// PA BNBD addressing parameters.
struct MxPageAttentionConstInfo {
    uint32_t maxBlockNumPerBatch = 0U;
    uint32_t paBlockStride = 0U;
    uint32_t qSparseBlockSize = 0U;
    uint32_t kvSparseBlockSize = 0U;
    uint32_t paBlockSize = 0U;
};

// B_N_Qb_Kb sparse parameters.
struct MxSparseConstInfo {
    uint32_t maxQb = 0U;
    uint32_t maxKb = 0U;
};

// QScale uses [T,N,D/64,2]; K/V scales use the PA BNBD layout.
struct MxScaleConstInfo {
    uint32_t scaleLastDim = QBSA_MX_SCALE_LAST_DIM;
    uint32_t queryScaleDSize = 0U;
    uint32_t keyScaleDSize = 0U;
    uint32_t valueScaleDSize = 0U;
};

struct MxConstInfo : MxBaseConstInfo, MxPageAttentionConstInfo, MxSparseConstInfo, MxScaleConstInfo {};
} // namespace regbasemx

#endif // QBSA_MXFP8_RUNTIME_H_
