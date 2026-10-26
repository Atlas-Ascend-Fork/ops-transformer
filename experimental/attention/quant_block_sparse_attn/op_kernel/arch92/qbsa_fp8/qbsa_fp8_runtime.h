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
 * \file util_regbase.h
 * \brief
 */

#ifndef FLASH_ATTENTION_UTIL_REGBASE_H
#define FLASH_ATTENTION_UTIL_REGBASE_H

#include "util.h"
#include "../../arch35/common/quant_block_sparse_attn_common.h"

using AscendC::QuePosition;
using AscendC::TQue;

#include "../common/quant_block_sparse_attn_common.h"

namespace regbaseutil {
#define COMMON_RUN_INFO \
    int64_t s2StartIdx;  /* S2 start, which may be nonzero for sparse inputs. */ \
    int64_t s2LoopCount; /* Current S2 loop index. */ \
    int64_t s2LoopLimit; \
    int64_t qBOffset; \
    int64_t qBScalarOffset; \
    int64_t s1oIdx = 0; /* S1-axis index. */ \
    int64_t boIdx = 0;  /* B-axis index. */ \
    int64_t n2oIdx = 0; /* N2-axis index. */ \
    int64_t goIdx = 0;  /* G-axis index. */ \
    int32_t s1RealSize; \
    int32_t s1RealSizeAlign32; \
    int32_t halfS1RealSize;      /* Vector-side S1 basic block size; 64 when Cube uses 128. */ \
    int32_t firstHalfS1RealSize; /* When S1 is odd, V0 processes one fewer row than V1. */ \
    int32_t s2RealSize;          /* Actual S2 basic block length. */ \
    int64_t s2AlignedSize;       /* S2 length aligned to 16. */ \
    int32_t s2SparseBlkRealSize[4]; \
    int32_t s2SparseBlkRealAlignedSize[4]; \
    int64_t sparseBase = 0;          /* Current (b,n2,g,qb) base address in sparseIndices. */ \
    int64_t sparseS2TokenOffset = 0; /* KV token offset mapped from sparseIndices. */ \
    int32_t actSparseLen = 0;        /* Current actual selected block count in sparseSeqLen. */ \
    int32_t vec2S1BaseSize;          /* S1 size after the Vec2 split, e.g., split 64 into 32 and 32. */ \
    int32_t vec2S1RealSize;          /* S1 tail size after the Vec2 split, e.g., split 63 into 32 and 31. */ \
    int64_t vecCoreOffset;           /* Vector core's S1 start offset relative to Cube. */ \
    int64_t queryOffset;             /* mm1 Query offset */ \
    int64_t keyOffset;               /* mm1 Key offset */ \
    int64_t valueOffset;             /* mm2 Value offset */ \
\
    int64_t taskId; \
    int64_t multiCoreInnerIdx = 0; \
\
    int64_t attentionOutOffset; \
    int64_t actualS1Size;       /* S1 of the current batch. */ \
    int64_t actualS2Size;       /* S2 of the current batch. */ \
    int64_t preTokensPerBatch;  /* Vec2 upper-left pretoken. */ \
    int64_t nextTokensPerBatch; /* Vec2 upper-left nexttoken. */ \
    uint8_t taskIdMod2; \
    uint8_t taskIdMod3; \
    uint8_t multiCoreIdxMod2 = 0; \
    uint8_t multiCoreIdxMod3 = 0; \
    int64_t sOuterOffset;

struct RunInfo {
    COMMON_RUN_INFO;
    /* Calculate the sparseIndices offset. */
    int64_t sparseBlkIdx[4];      /* Four sparse blocks within a V121 S2=512 basic block. */
    int64_t phyBlkNumIdx[4];      // Index of blockNum after blockTable mapping.
    bool sparseBlkPartialMask[4]; // True for partial computation; false for full computation.
    bool hasRound1;               // Valid sparse blocks exist in slots 2/3 after sorting.
    // LSE output offset.
    int64_t softmaxLseOffset;
};

} // namespace regbaseutil

#endif // FLASH_ATTENTION_UTIL_REGBASE_H
