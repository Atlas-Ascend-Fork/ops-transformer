/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t INPUT_QUERY = 0;
constexpr size_t INPUT_BLOCK_TABLE = 6;
constexpr size_t ATTR_MAX_KV_LENGTH = 1;
constexpr size_t OUTPUT_SCORE = 0;
constexpr int64_t CACHE_BLOCK_SIZE = 128;
} // namespace

static ge::graphStatus InferShapeTqScoreCompute(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TqScoreCompute", "infer shape context is null"), return ge::GRAPH_FAILED);
    const auto* queryShape = context->GetInputShape(INPUT_QUERY);
    const auto* blockTableShape = context->GetInputShape(INPUT_BLOCK_TABLE);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockTableShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_IF(queryShape->GetDimNum() != 3 || blockTableShape->GetDimNum() != 2,
                OP_LOGE(context, "query/block_table ranks must be 3/2"), return ge::GRAPH_FAILED);
    const auto* maxKvLength = attrs->GetAttrPointer<int64_t>(ATTR_MAX_KV_LENGTH);
    OP_CHECK_NULL_WITH_CONTEXT(context, maxKvLength);
    auto* outputShape = context->GetOutputShape(OUTPUT_SCORE);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    *outputShape = *queryShape;
    outputShape->SetDim(2, *maxKvLength < 0 ? blockTableShape->GetDim(1) * CACHE_BLOCK_SIZE : *maxKvLength);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTqScoreCompute(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TqScoreCompute", "infer dtype context is null"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUTPUT_SCORE, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TqScoreCompute).InferShape(InferShapeTqScoreCompute).InferDataType(InferDataTypeTqScoreCompute);
} // namespace ops
