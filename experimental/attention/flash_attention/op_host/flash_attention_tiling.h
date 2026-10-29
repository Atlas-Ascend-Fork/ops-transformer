/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_tiling.h
 * \brief
 */

#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "base/tiling_base.h"
#include "base/tiling_templates_registry.h"
#include "base/tiling_type.h"
#include "base/data_copy_transpose_tiling.h"

namespace optiling {

ge::graphStatus FlashAttentionTilingFunc(gert::TilingContext* context);

BEGIN_TILING_DATA_DEF(InputParams)
TILING_DATA_FIELD_DEF(int64_t, bSize);
TILING_DATA_FIELD_DEF(int64_t, n2Size);
TILING_DATA_FIELD_DEF(int64_t, gSize);
TILING_DATA_FIELD_DEF(int64_t, s1Size);
TILING_DATA_FIELD_DEF(int64_t, s2Size);
TILING_DATA_FIELD_DEF(int64_t, accumS1);
TILING_DATA_FIELD_DEF(int64_t, accumS2);
TILING_DATA_FIELD_DEF(int64_t, alignedS2);
TILING_DATA_FIELD_DEF(int64_t, dSize);
TILING_DATA_FIELD_DEF(int64_t, vDsize);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(int64_t, preTokens);
TILING_DATA_FIELD_DEF(int64_t, nextTokens);

// 1: BSH/BSND, 2: SBH, 3: BNSD
TILING_DATA_FIELD_DEF(uint8_t, layoutType);
// 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskShapeType);
// 0: fp16, 1: bool(uint8)
TILING_DATA_FIELD_DEF(uint8_t, attenMaskDataType);
// ALL: 0, NONE: 1, ANY: 2, CAUSAL: 3, BAND: 4 };
TILING_DATA_FIELD_DEF(uint8_t, attenMaskCompressMode);
// 0: high precise, 1: high performance, 2: invalid line high precise
TILING_DATA_FIELD_DEF(uint8_t, sparseType);
TILING_DATA_FIELD_DEF(uint8_t, prmMaskFlag);
TILING_DATA_FIELD_DEF(uint8_t, outputMaxLogits);
TILING_DATA_FIELD_DEF(uint8_t, sinkFlag);
TILING_DATA_FIELD_DEF(uint8_t, sinkRequiresGradFlag);
// aa3 for attenOut format: 0:disable aa3 or BNSD, 1:BSH/BSND, 2:SBH
TILING_DATA_FIELD_DEF(uint8_t, attenOutLayoutType);
TILING_DATA_FIELD_DEF(uint32_t, attenMaskS2Size);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(InputParamsOp, InputParams)

BEGIN_TILING_DATA_DEF(MultiCoreParams)
TILING_DATA_FIELD_DEF(int32_t, coreNum);
TILING_DATA_FIELD_DEF(int32_t, aivNum);
// BN2GS1.o
TILING_DATA_FIELD_DEF(int64_t, totalSize);
// BN2GS1.o / core_num
TILING_DATA_FIELD_DEF(int64_t, splitFactorSize);
TILING_DATA_FIELD_DEF(int64_t, splitFactorTailSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, 48, sparseStartIdx);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MultiCoreParamsOp, MultiCoreParams)

BEGIN_TILING_DATA_DEF(CoreParams)
TILING_DATA_FIELD_DEF(int32_t, s1BaseSize);
TILING_DATA_FIELD_DEF(int32_t, s1BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, s1OuterSize);
TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseSize);
TILING_DATA_FIELD_DEF(int32_t, s1Vec2BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, s1Vec2OuterSize);
TILING_DATA_FIELD_DEF(int32_t, s2BaseSize);
TILING_DATA_FIELD_DEF(int32_t, s2BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, s2OuterSize);
TILING_DATA_FIELD_DEF(int32_t, dBaseSize);
TILING_DATA_FIELD_DEF(int32_t, dBaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, dOuterSize);
TILING_DATA_FIELD_DEF(int32_t, bBaseSize);
TILING_DATA_FIELD_DEF(int32_t, bBaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, bOuterSize);
TILING_DATA_FIELD_DEF(int32_t, n2BaseSize);
TILING_DATA_FIELD_DEF(int32_t, n2BaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, n2OuterSize);
TILING_DATA_FIELD_DEF(int32_t, gBaseSize);
TILING_DATA_FIELD_DEF(int32_t, gBaseTailSize);
TILING_DATA_FIELD_DEF(int64_t, gOuterSize);
TILING_DATA_FIELD_DEF(int32_t, nRatio);
TILING_DATA_FIELD_DEF(int32_t, rsvd);
TILING_DATA_FIELD_DEF(int64_t, s1SparseValidSize);
TILING_DATA_FIELD_DEF(int64_t, s2SparseValidSize);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(CoreParamsOp, CoreParams)

BEGIN_TILING_DATA_DEF(TensorSizeParams)
TILING_DATA_FIELD_DEF(int32_t, bmm1ResUbSize);
TILING_DATA_FIELD_DEF(int32_t, attenMaskUbSize);
TILING_DATA_FIELD_DEF(int32_t, castUbSize);
TILING_DATA_FIELD_DEF(int32_t, softmaxMaxUbSize);
TILING_DATA_FIELD_DEF(int32_t, softmaxSumUbSize);
TILING_DATA_FIELD_DEF(int32_t, softmaxExpUbSize);
TILING_DATA_FIELD_DEF(int32_t, apiTmpBufferBytes);
TILING_DATA_FIELD_DEF(int32_t, bmm2ResUbSize);
TILING_DATA_FIELD_DEF(int32_t, inputQueBytes);
TILING_DATA_FIELD_DEF(int32_t, outputQueBytes);
// API buffer use remain space of ub
TILING_DATA_FIELD_DEF(int32_t, tmpBufBytes);
TILING_DATA_FIELD_DEF(int32_t, softmaxMaxOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, softmaxSumOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, maxSumApiOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, maxSumApiSize);
TILING_DATA_FIELD_DEF(int32_t, customSoftmaxApiSize);
TILING_DATA_FIELD_DEF(int32_t, dropoutApiSize);
TILING_DATA_FIELD_DEF(int32_t, attenMaskApiSize);
TILING_DATA_FIELD_DEF(int32_t, attenMaskApiOffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2Size);
TILING_DATA_FIELD_DEF(int32_t, bmm1ProcessTInStage2OffsetBytes);
// workspace
TILING_DATA_FIELD_DEF(int32_t, wkspSection1OffsetBytes);
TILING_DATA_FIELD_DEF(int32_t, wkspSection2OffsetBytes);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TensorSizeParamsOp, TensorSizeParams)

BEGIN_TILING_DATA_DEF(FlashAttentionGeneralTilingData)
TILING_DATA_FIELD_DEF_STRUCT(InputParams, inputParams);
TILING_DATA_FIELD_DEF_STRUCT(MultiCoreParams, multiCoreParams);
TILING_DATA_FIELD_DEF_STRUCT(CoreParams, coreParams);
TILING_DATA_FIELD_DEF_STRUCT(TensorSizeParams, tensorSizeParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(SoftMaxTiling, softmaxFlashTilingData);
TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingData);
TILING_DATA_FIELD_DEF_STRUCT(CopyTransposeTiling, transposeTilingDataTailCore);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(FlashAttention, FlashAttentionGeneralTilingData)

} // namespace optiling
