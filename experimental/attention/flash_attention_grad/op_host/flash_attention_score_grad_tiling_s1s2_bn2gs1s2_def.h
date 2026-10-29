/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

/////////////////////////////////////////////////////////////////////////
// S1S2_BNGS1S2
/////////////////////////////////////////////////////////////////////////
BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradS1S2BNGS1S2BaseParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, b);
TILING_DATA_FIELD_DEF(uint32_t, n2);
TILING_DATA_FIELD_DEF(uint32_t, g);
TILING_DATA_FIELD_DEF(int64_t, s1);
TILING_DATA_FIELD_DEF(int64_t, s2);
TILING_DATA_FIELD_DEF(uint32_t, h);
TILING_DATA_FIELD_DEF(uint32_t, d);
TILING_DATA_FIELD_DEF(uint32_t, dV);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, layout);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskOptional);
TILING_DATA_FIELD_DEF(uint32_t, prmMaskOptional);
TILING_DATA_FIELD_DEF(uint32_t, sinkOptional);
TILING_DATA_FIELD_DEF(uint32_t, sinkRequiresGrad);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskDtype);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskShapeType);
TILING_DATA_FIELD_DEF(int64_t, s1Token); // pre_tokens
TILING_DATA_FIELD_DEF(int64_t, s2Token); // next_tokens
TILING_DATA_FIELD_DEF(uint32_t, sparseMode);
TILING_DATA_FIELD_DEF(uint32_t, isSparse);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskCompressMode);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
TILING_DATA_FIELD_DEF(uint32_t, softmaxLseOptional);
TILING_DATA_FIELD_DEF(int64_t, Tq);
TILING_DATA_FIELD_DEF(int64_t, Tkv);
TILING_DATA_FIELD_DEF(uint32_t, s1TileNum);
TILING_DATA_FIELD_DEF(uint32_t, s2TileNum);
TILING_DATA_FIELD_DEF(uint32_t, gs1TileNum);
TILING_DATA_FIELD_DEF(uint64_t, dqkvWorkspaceOffset);
TILING_DATA_FIELD_DEF(uint64_t, dqkvWorkspaceSize);
TILING_DATA_FIELD_DEF(uint32_t, saveToWorkspace);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradS1S2BNGS1S2BaseParamsOp, FlashAttentionScoreGradS1S2BNGS1S2BaseParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, s1Outer);
TILING_DATA_FIELD_DEF(uint32_t, s1CvRatio);
TILING_DATA_FIELD_DEF(uint32_t, s1Inner);
TILING_DATA_FIELD_DEF(uint32_t, s1CvInner);
TILING_DATA_FIELD_DEF(uint32_t, s1Tail);
TILING_DATA_FIELD_DEF(uint32_t, s1CvTail);
TILING_DATA_FIELD_DEF(uint32_t, s2Outer);
TILING_DATA_FIELD_DEF(uint32_t, s2CvRatio);
TILING_DATA_FIELD_DEF(uint32_t, s2Inner);
TILING_DATA_FIELD_DEF(uint32_t, s2Tail);
TILING_DATA_FIELD_DEF(uint32_t, baseMN);
TILING_DATA_FIELD_DEF(uint32_t, sfmgdOuter);
TILING_DATA_FIELD_DEF(uint32_t, sfmgdFactor);
TILING_DATA_FIELD_DEF(uint32_t, sfmgdTail);
TILING_DATA_FIELD_DEF(uint32_t, blockOuter);
TILING_DATA_FIELD_DEF(uint32_t, tmpBufferSize);
TILING_DATA_FIELD_DEF(int64_t, totalS1TileNums);
END_TILING_DATA_DEF;
// 固定写法不能换行，会失败
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParamsOp,
                           FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParams)

BEGIN_TILING_DATA_DEF(BlockNumListParams)
TILING_DATA_FIELD_DEF_ARR(int64_t, 50, blockStarts);
TILING_DATA_FIELD_DEF_ARR(int64_t, 50, blockEnds);
TILING_DATA_FIELD_DEF_ARR(int64_t, 50, splitS1BlockStarts);
TILING_DATA_FIELD_DEF_ARR(int64_t, 50, splitS1BlockEnds);
TILING_DATA_FIELD_DEF_ARR(int64_t, 50, splitS2BlockStarts);
TILING_DATA_FIELD_DEF_ARR(int64_t, 50, splitS2BlockEnds);
TILING_DATA_FIELD_DEF_ARR(int64_t, 50, gs1Offset);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BlockNumListParamsOp, BlockNumListParams)

// dq/dv/dk pre init zero
BEGIN_TILING_DATA_DEF(PreParams)
TILING_DATA_FIELD_DEF(int64_t, qPreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, qPreBlockTotal);
TILING_DATA_FIELD_DEF(int64_t, qPreBlockTail);
TILING_DATA_FIELD_DEF(int64_t, kPreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, kPreBlockTotal);
TILING_DATA_FIELD_DEF(int64_t, kPreBlockTail);
TILING_DATA_FIELD_DEF(int64_t, vPreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, vPreBlockTotal);
TILING_DATA_FIELD_DEF(int64_t, vPreBlockTail);
TILING_DATA_FIELD_DEF(int64_t, sinkPreBlockFactor);
TILING_DATA_FIELD_DEF(int64_t, sinkPreBlockTotal);
TILING_DATA_FIELD_DEF(int64_t, sinkPreBlockTail);
TILING_DATA_FIELD_DEF(int64_t, dsinkSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PreParamsOp, PreParams)

// dq/dv/dk cast fp322T1
BEGIN_TILING_DATA_DEF(PostParams)
TILING_DATA_FIELD_DEF(uint32_t, postUbBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, UbBaseSize);
TILING_DATA_FIELD_DEF(uint32_t, qPostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, qPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, qPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, qPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, kPostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, kPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, kPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, kPostTailNum);
TILING_DATA_FIELD_DEF(uint32_t, vPostBlockFactor);
TILING_DATA_FIELD_DEF(uint64_t, vPostBlockTotal);
TILING_DATA_FIELD_DEF(uint32_t, vPostBaseNum);
TILING_DATA_FIELD_DEF(uint32_t, vPostTailNum);
TILING_DATA_FIELD_DEF(uint64_t, qSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, kSizeAlign);
TILING_DATA_FIELD_DEF(uint64_t, vSizeAlign);
TILING_DATA_FIELD_DEF(uint32_t, dsinkBaseSize);

END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(PostParamsOp, PostParams)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradS1S2BNGS1S2BaseParams, s1s2BNGS1S2BaseParams);
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradS1S2BNGS1S2SplitCoreParams, s1s2BNGS1S2SplitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm3TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
TILING_DATA_FIELD_DEF_STRUCT(BlockNumListParams, s1s2BNGS1S2BlockNumList);
TILING_DATA_FIELD_DEF_STRUCT(PreParams, preTilingData);
TILING_DATA_FIELD_DEF_STRUCT(PostParams, postTilingData);
END_TILING_DATA_DEF;
// 固定写法不能换行，会失败
REGISTER_TILING_DATA_CLASS(FlashAttentionGrad, FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2)
} // namespace optiling
