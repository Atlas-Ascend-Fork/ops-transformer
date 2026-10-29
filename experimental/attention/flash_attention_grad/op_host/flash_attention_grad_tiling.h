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
#include <register/op_impl_registry.h>
#include "base/data_copy_transpose_tiling.h"
#include "base/tiling_templates_registry.h"
#include "base/data_copy_transpose_tiling_def.h"

namespace optiling {

const std::vector<gert::Shape> OLD_TEMPLATE_SUPPORTED_SHAPE = {
    // 5d shape [N,S1,S2,D,B]
};

ge::graphStatus TilingFlashAttentionGrad(gert::TilingContext* context);

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum);
TILING_DATA_FIELD_DEF(uint32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, remainCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, S);
TILING_DATA_FIELD_DEF(uint32_t, Sq);
TILING_DATA_FIELD_DEF(uint32_t, Skv);
TILING_DATA_FIELD_DEF(uint32_t, H);
TILING_DATA_FIELD_DEF(uint32_t, D);            // H/N
TILING_DATA_FIELD_DEF(uint32_t, batch);        // B*N
TILING_DATA_FIELD_DEF(uint32_t, typeBlockNum); // 32/typeSize
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, preTokens);
TILING_DATA_FIELD_DEF(uint32_t, nextTokens);
TILING_DATA_FIELD_DEF(uint32_t, preTokensBlocks);
TILING_DATA_FIELD_DEF(uint32_t, nextTokensBlocks);
TILING_DATA_FIELD_DEF(uint32_t, splitBatchFactor); // batch轴分核因子
TILING_DATA_FIELD_DEF(uint32_t, splitSeqFactor);   // seq分核因子
TILING_DATA_FIELD_DEF(uint32_t, allMatInputNum);   // S*h
TILING_DATA_FIELD_DEF(uint32_t, allMatResNum);     // S*S
TILING_DATA_FIELD_DEF(uint32_t, allReduceNum);     // S*16  or S*8
TILING_DATA_FIELD_DEF(uint32_t, layoutSBH);
TILING_DATA_FIELD_DEF(uint32_t, apiClcQueueSize);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskType);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskShapeType);
TILING_DATA_FIELD_DEF(uint32_t, isHighPerformanceMode);
TILING_DATA_FIELD_DEF(uint32_t, preSeqCoreNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttentionScoreGradBaseParamsOp, FlashAttentionScoreGradBaseParams)

BEGIN_TILING_DATA_DEF(SplitCoreParams)
TILING_DATA_FIELD_DEF(uint32_t, splitedBatchRange);         // batch/splitBatchFactor
TILING_DATA_FIELD_DEF(uint32_t, splitedCoreSeqNum);         // S/splitSeqFactor
TILING_DATA_FIELD_DEF(uint32_t, splitedCoreBatchNum);       // splitedBatchRange * S * h
TILING_DATA_FIELD_DEF(uint32_t, splitAllMatInputNum);       // = splitedCoreSeqNum * h;
TILING_DATA_FIELD_DEF(uint32_t, splitedCoreBatchMaskNum);   // splitedBatchRange * S * S
TILING_DATA_FIELD_DEF(uint32_t, splitAllMatResNum);         // = splitedCoreSeqNum * S;
TILING_DATA_FIELD_DEF(uint32_t, splitedCoreBatchReduceNum); // splitedBatchRange * S * 16 or 8
TILING_DATA_FIELD_DEF(uint32_t, splitAllReduceNum);         // = splitedCoreSeqNum * S;
TILING_DATA_FIELD_DEF(uint32_t, flashKLoopRange);
TILING_DATA_FIELD_DEF(uint32_t, s1OutRange);
TILING_DATA_FIELD_DEF(uint32_t, sOut);
TILING_DATA_FIELD_DEF(uint32_t, sFla);
TILING_DATA_FIELD_DEF(uint32_t, sOutTail);
TILING_DATA_FIELD_DEF(uint32_t, sFlaTail);
TILING_DATA_FIELD_DEF(uint32_t, sOutAlign);
TILING_DATA_FIELD_DEF(uint32_t, sFlaAlign);
TILING_DATA_FIELD_DEF(uint32_t, sOutTailAlign);
TILING_DATA_FIELD_DEF(uint32_t, sFlaTailAlign);
TILING_DATA_FIELD_DEF(uint32_t, sOutCubeAlign);
TILING_DATA_FIELD_DEF(uint32_t, sOutTailCubeAlign);

// splitCoreParams + sOutParams
TILING_DATA_FIELD_DEF(uint32_t, sInnerRange);
TILING_DATA_FIELD_DEF(uint32_t, sInner);
TILING_DATA_FIELD_DEF(uint32_t, innerReduceNum);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResNum);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResTailNum);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResNumAlign);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResTailNumAlign);
TILING_DATA_FIELD_DEF(uint32_t, innerMatInputNum);
TILING_DATA_FIELD_DEF(uint32_t, innerMaskNum);

TILING_DATA_FIELD_DEF(uint32_t, outReduceNum);
TILING_DATA_FIELD_DEF(uint32_t, outMatResNum);
TILING_DATA_FIELD_DEF(uint32_t, outMatResTailNum);
TILING_DATA_FIELD_DEF(uint32_t, outMatInputNum);
TILING_DATA_FIELD_DEF(uint32_t, outMatInputNumCubeAlign);
TILING_DATA_FIELD_DEF(uint32_t, outMaskNum);
TILING_DATA_FIELD_DEF(uint32_t, sFlaMatInputNum);
TILING_DATA_FIELD_DEF(uint32_t, sFlaMatInputTailNum);

TILING_DATA_FIELD_DEF(uint32_t, innerReduceSize);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResSize);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResTailSize);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResSizeAlign);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResTailSizeAlign);
TILING_DATA_FIELD_DEF(uint32_t, innerMatInputSize);
TILING_DATA_FIELD_DEF(uint32_t, innerMaskSize);

TILING_DATA_FIELD_DEF(uint32_t, outReduceSize);
TILING_DATA_FIELD_DEF(uint32_t, outMatResSize);
TILING_DATA_FIELD_DEF(uint32_t, outMatResTailSize);
TILING_DATA_FIELD_DEF(uint32_t, outMatInputSize);
TILING_DATA_FIELD_DEF(uint32_t, outMatInputSizeCubeAlign);
TILING_DATA_FIELD_DEF(uint32_t, outMaskSize);
TILING_DATA_FIELD_DEF(uint32_t, sFlaMatInputSize);
TILING_DATA_FIELD_DEF(uint32_t, sFlaMatInputTailSize);

TILING_DATA_FIELD_DEF(uint32_t, sFlaBlock);
TILING_DATA_FIELD_DEF(uint32_t, sFlaTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, sSubSFlaStride);
TILING_DATA_FIELD_DEF(uint32_t, sSubSFlaTailStride);
TILING_DATA_FIELD_DEF(uint32_t, subMask);
TILING_DATA_FIELD_DEF(uint32_t, sFlaBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, sFlaBlockTailNum);
TILING_DATA_FIELD_DEF(uint32_t, subRange);

TILING_DATA_FIELD_DEF(uint32_t, innerMatInputBlock);
TILING_DATA_FIELD_DEF(uint32_t, innerMaskBlock);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResBlock);
TILING_DATA_FIELD_DEF(uint32_t, innerMatResTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, dropStride);

// splitCoreParams + sOutTailParams
TILING_DATA_FIELD_DEF(uint32_t, tailSInnerRange);
TILING_DATA_FIELD_DEF(uint32_t, tailSInner);

TILING_DATA_FIELD_DEF(uint32_t, tailInnerReduceNum);
TILING_DATA_FIELD_DEF(uint32_t, tailInnerMatResNum);
TILING_DATA_FIELD_DEF(uint32_t, tailInnerMatResTailNum);
TILING_DATA_FIELD_DEF(uint32_t, tailInnerMatInputNum);
TILING_DATA_FIELD_DEF(uint32_t, tailInnerMaskNum);

TILING_DATA_FIELD_DEF(uint32_t, tailOutReduceNum);
TILING_DATA_FIELD_DEF(uint32_t, tailOutMatResNum);
TILING_DATA_FIELD_DEF(uint32_t, tailOutMatResTailNum);
TILING_DATA_FIELD_DEF(uint32_t, tailOutMatInputNum);
TILING_DATA_FIELD_DEF(uint32_t, tailOutMaskNum);
TILING_DATA_FIELD_DEF(uint32_t, tailSFlaMatInputNum);
TILING_DATA_FIELD_DEF(uint32_t, tailSFlaMatInputTailNum);

TILING_DATA_FIELD_DEF(uint32_t, tailInnerReduceSize);
TILING_DATA_FIELD_DEF(uint32_t, tailInnerMatResSize);
TILING_DATA_FIELD_DEF(uint32_t, tailInnerMatResTailSize);
TILING_DATA_FIELD_DEF(uint32_t, tailInnerMatInputSize);
TILING_DATA_FIELD_DEF(uint32_t, tailInnerMaskSize);

TILING_DATA_FIELD_DEF(uint32_t, tailOutReduceSize);
TILING_DATA_FIELD_DEF(uint32_t, tailOutMatResSize);
TILING_DATA_FIELD_DEF(uint32_t, tailOutMatResTailSize);
TILING_DATA_FIELD_DEF(uint32_t, tailOutMatInputSize);
TILING_DATA_FIELD_DEF(uint32_t, tailOutMaskSize);
TILING_DATA_FIELD_DEF(uint32_t, tailSFlaMatInputSize);
TILING_DATA_FIELD_DEF(uint32_t, tailSFlaMatInputTailSize);

TILING_DATA_FIELD_DEF(uint32_t, tailSFlaBlock);
TILING_DATA_FIELD_DEF(uint32_t, tailSFlaTailBlock);
TILING_DATA_FIELD_DEF(uint32_t, tailSSubSFlaStride);
TILING_DATA_FIELD_DEF(uint32_t, tailSSubSFlaTailStride);
TILING_DATA_FIELD_DEF(uint32_t, tailSubMask);
TILING_DATA_FIELD_DEF(uint32_t, tailSFlaBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, tailSFlaBlockTailNum);
TILING_DATA_FIELD_DEF(uint32_t, tailSubRange);

TILING_DATA_FIELD_DEF(uint32_t, tailDropStride);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(SplitCoreParamsOp, SplitCoreParams)

BEGIN_TILING_DATA_DEF(CastTiling)
TILING_DATA_FIELD_DEF(uint32_t, baseOutNum);
TILING_DATA_FIELD_DEF(uint32_t, castUsedUbSize);
TILING_DATA_FIELD_DEF(uint32_t, allNumQuery);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreNumQuery);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreNumQuery);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreLoopTimesQuery);
TILING_DATA_FIELD_DEF(uint32_t, lastLoopNumQuery);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreLoopTimesQuery);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreLastLoopNumQuery);
TILING_DATA_FIELD_DEF(uint32_t, allNumKv);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreNumKv);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreNumKv);
TILING_DATA_FIELD_DEF(uint32_t, singleCoreLoopTimesKv);
TILING_DATA_FIELD_DEF(uint32_t, lastLoopNumKv);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreLoopTimesKv);
TILING_DATA_FIELD_DEF(uint32_t, tailCoreLastLoopNumKv);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CastTilingOp, CastTiling)

BEGIN_TILING_DATA_DEF(EmptyTensorTilingData)
TILING_DATA_FIELD_DEF(uint32_t, formerDqNum);
TILING_DATA_FIELD_DEF(uint32_t, formerDkNum);
TILING_DATA_FIELD_DEF(uint32_t, formerDpseNum);
TILING_DATA_FIELD_DEF(uint32_t, res);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreDqNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreDqNum);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreDkNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreDkNum);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreDpseNum);
TILING_DATA_FIELD_DEF(uint64_t, tailCoreDpseNum);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(EmptyTensorTilingDataOp, EmptyTensorTilingData)

BEGIN_TILING_DATA_DEF(FlashAttentionScoreGradTilingData)
TILING_DATA_FIELD_DEF_STRUCT(FlashAttentionScoreGradBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(SplitCoreParams, splitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(SplitCoreParams, tailSplitCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm3TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm4TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingDataTailCore);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxTilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxGradTilingData);
TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, copyTransposeTiling);
TILING_DATA_FIELD_DEF_STRUCT(CastTiling, castTiling);
TILING_DATA_FIELD_DEF_STRUCT(EmptyTensorTilingData, emptyTensorTilingData);
END_TILING_DATA_DEF;
// REGISTER_TILING_DATA_CLASS(FlashAttentionGrad, FlashAttentionScoreGradTilingData)
} // namespace optiling
