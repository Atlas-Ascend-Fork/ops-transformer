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
    int64_t s2StartIdx;  /* s2起始, sparse场景可能非0 */ \
    int64_t s2LoopCount; /* s2循环当前index */ \
    int64_t s2LoopLimit; \
    int64_t qBOffset; \
    int64_t qBScalarOffset; \
    int64_t s1oIdx = 0; /* s1轴index */ \
    int64_t boIdx = 0;  /* b轴index */ \
    int64_t n2oIdx = 0; /* n2轴index */ \
    int64_t goIdx = 0;  /* g轴index */ \
    int32_t s1RealSize; \
    int32_t s1RealSizeAlign32; \
    int32_t halfS1RealSize;      /* vec侧s1基本块大小, Cube=128时为64 */ \
    int32_t firstHalfS1RealSize; /* s1非2倍数时v0比v1少算一行 */ \
    int32_t s2RealSize;          /* s2基本块真实长度 */ \
    int64_t s2AlignedSize;       /* s2对齐到16后的长度 */ \
    int32_t s2SparseBlkRealSize[4]; \
    int32_t s2SparseBlkRealAlignedSize[4]; \
    int64_t sparseBase = 0;          /* sparseIndices当前(b,n2,g,qb)基址 */ \
    int64_t sparseS2TokenOffset = 0; /* sparseIndices映射后KV token offset */ \
    int32_t actSparseLen = 0;        /* sparseSeqLen当前实际选块数 */ \
    int32_t vec2S1BaseSize;          /* vec2切分后S1大小, 如64切成两份32 */ \
    int32_t vec2S1RealSize;          /* vec2切分后S1尾块, 如63切成32和31 */ \
    int64_t vecCoreOffset;           /* vec核相对cube起始s1偏移 */ \
    int64_t queryOffset;             /* mm1 Query offset */ \
    int64_t keyOffset;               /* mm1 Key offset */ \
    int64_t valueOffset;             /* mm2 Value offset */ \
\
    int64_t taskId; \
    int64_t multiCoreInnerIdx = 0; \
\
    int64_t attentionOutOffset; \
    int64_t actualS1Size;       /* 当前batch的s1 */ \
    int64_t actualS2Size;       /* 当前batch的s2 */ \
    int64_t preTokensPerBatch;  /* vec2左上顶点pretoken */ \
    int64_t nextTokensPerBatch; /* vec2左上顶点nexttoken */ \
    uint8_t taskIdMod2; \
    uint8_t taskIdMod3; \
    uint8_t multiCoreIdxMod2 = 0; \
    uint8_t multiCoreIdxMod3 = 0; \
    int64_t sOuterOffset;

struct RunInfo {
    COMMON_RUN_INFO;
    /* sparseIndices 偏移量计算 */
    int64_t sparseBlkIdx[4];      /* V121 S2=512基本块内的四个sparse块 */
    int64_t phyBlkNumIdx[4];      // blockTable映射后的blockNum的Idx
    bool sparseBlkPartialMask[4]; // true表示部分计算  false表示全量计算
    bool hasRound1;               // 排序后slot 2/3中存在有效sparse块
    // lse 输出offset
    int64_t softmaxLseOffset;
};

} // namespace regbaseutil

#endif // FLASH_ATTENTION_UTIL_REGBASE_H
