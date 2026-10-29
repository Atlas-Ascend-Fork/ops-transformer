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
 * \file merge_chunk2_tiling.cpp
 * \brief MergeChunk2 tiling
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "merge_chunk2_tiling.h"

namespace {
constexpr uint32_t MAX_TOKEN_BYTES = 176 * 1024;
constexpr uint32_t MAX_EXPERT_BYTES = 4 * 1024;
constexpr uint32_t HIDDEN_ALIGNMENT = 16;
constexpr uint32_t TILING_KEY_BF16 = 0;
constexpr uint32_t TILING_KEY_FP16 = 1;
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    MergeChunk2TilingData tiling;

    const gert::Shape& in1Chunk1Shape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape& in1Chunk2Shape = context->GetInputShape(1)->GetStorageShape();
    const gert::Shape& in2Chunk1Shape = context->GetInputShape(2)->GetStorageShape();
    const gert::Shape& in2Chunk2Shape = context->GetInputShape(3)->GetStorageShape();
    const gert::Shape& groupListChunk1Shape = context->GetInputShape(4)->GetStorageShape();
    const gert::Shape& groupListChunk2Shape = context->GetInputShape(5)->GetStorageShape();

    // check dim
    if (in1Chunk1Shape.GetDimNum() != 2) {
        OP_LOGE(context, "Dim of input1_chunk1 not right");
        return ge::GRAPH_FAILED;
    }
    if (in1Chunk2Shape.GetDimNum() != 2) {
        OP_LOGE(context, "Dim of input1_chunk2 not right");
        return ge::GRAPH_FAILED;
    }
    if (in2Chunk1Shape.GetDimNum() != 2) {
        OP_LOGE(context, "Dim of input2_chunk1 not right");
        return ge::GRAPH_FAILED;
    }
    if (in2Chunk2Shape.GetDimNum() != 2) {
        OP_LOGE(context, "Dim of input2_chunk2 not right");
        return ge::GRAPH_FAILED;
    }
    if (groupListChunk1Shape.GetDimNum() != 1) {
        OP_LOGE(context, "Dim of group_list_chunk1 not right");
        return ge::GRAPH_FAILED;
    }
    if (groupListChunk2Shape.GetDimNum() != 1) {
        OP_LOGE(context, "Dim of group_list_chunk2 not right");
        return ge::GRAPH_FAILED;
    }

    // check shape
    int t1c1 = in1Chunk1Shape.GetDim(0);
    int h1c1 = in1Chunk1Shape.GetDim(1);
    int t1c2 = in1Chunk2Shape.GetDim(0);
    int h1c2 = in1Chunk2Shape.GetDim(1);
    int t2c1 = in2Chunk1Shape.GetDim(0);
    int h2c1 = in2Chunk1Shape.GetDim(1);
    int t2c2 = in2Chunk2Shape.GetDim(0);
    int h2c2 = in2Chunk2Shape.GetDim(1);
    int ec1 = groupListChunk1Shape.GetDim(0);
    int ec2 = groupListChunk2Shape.GetDim(0);
    if (t1c1 != t2c1) {
        OP_LOGE(context, "t1c1 != t2c1. %d, %d", t1c1, t2c1);
        return ge::GRAPH_FAILED;
    }
    if (t1c2 != t2c2) {
        OP_LOGE(context, "t1c2 != t2c2. %d, %d", t1c2, t2c2);
        return ge::GRAPH_FAILED;
    }
    if (h1c1 != h1c2) {
        OP_LOGE(context, "h1c1 != h1c2. %d, %d", h1c1, h1c2);
        return ge::GRAPH_FAILED;
    }
    if (h2c1 != h2c2) {
        OP_LOGE(context, "h2c1 != h2c2. %d, %d", h2c1, h2c2);
        return ge::GRAPH_FAILED;
    }
    if (ec1 != ec2) {
        OP_LOGE(context, "ec1 != ec2. %d, %d", ec1, ec2);
        return ge::GRAPH_FAILED;
    }

    // check range & alignment
    int maxExpert = MAX_EXPERT_BYTES / sizeof(int32_t);
    int maxHidden = MAX_TOKEN_BYTES / 2;
    if (ec1 > maxExpert) {
        OP_LOGE(context, "num_expert exceeds limit. %d, %d", ec1, maxExpert);
        return ge::GRAPH_FAILED;
    }
    if (h1c1 <= 0 || h2c1 <= 0) {
        OP_LOGE(context, "hidden must be positive. %d, %d", h1c1, h2c1);
        return ge::GRAPH_FAILED;
    }
    if (h1c1 % HIDDEN_ALIGNMENT != 0) {
        OP_LOGE(context, "hidden1 not align to 32B. %d", h1c1);
        return ge::GRAPH_FAILED;
    }
    if (h2c1 % HIDDEN_ALIGNMENT != 0) {
        OP_LOGE(context, "hidden2 not align to 32B. %d", h2c1);
        return ge::GRAPH_FAILED;
    }
    if (h1c1 > maxHidden) {
        OP_LOGE(context, "hidden1 exceeds limit. %d, %d", h1c1, maxHidden);
        return ge::GRAPH_FAILED;
    }
    if (h2c1 > maxHidden) {
        OP_LOGE(context, "hidden2 exceeds limit. %d, %d", h2c1, maxHidden);
        return ge::GRAPH_FAILED;
    }

    auto inType = context->GetInputDesc(0)->GetDataType();
    if (inType == ge::DT_BF16) {
        context->SetTilingKey(TILING_KEY_BF16);
    } else if (inType == ge::DT_FLOAT16) {
        context->SetTilingKey(TILING_KEY_FP16);
    } else {
        OP_LOGE(context, "Dtype of input 0 is not in bf16/fp16.");
        return ge::GRAPH_FAILED;
    }

    tiling.set_numExpert(ec1);
    tiling.set_t1(t1c1);
    tiling.set_t2(t1c2);
    tiling.set_h1(h1c1);
    tiling.set_h2(h2c1);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();
    tiling.set_usedCoreNum(coreNum);

    // pure vector op, will launch the same size of vector cores as SetBlockDim does.
    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForMergeChunk2(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<MergeChunk2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MergeChunk2).Tiling(TilingFunc).TilingParse<MergeChunk2CompileInfo>(TilingPrepareForMergeChunk2);

} // namespace optiling
