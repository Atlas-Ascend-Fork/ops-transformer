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
constexpr size_t OUTPUT_PACKED = 0;
constexpr int64_t TOKEN_ALIGNMENT = 128;
constexpr int64_t PACKED_QUERY_WIDTH = 512;
} // namespace

static ge::graphStatus InferShapeTqQueryPack(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TqQueryPack", "infer shape context is null"), return ge::GRAPH_FAILED);
    const auto* queryShape = context->GetInputShape(INPUT_QUERY);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_IF(queryShape->GetDimNum() != 3, OP_LOGE(context, "query rank must be 3"), return ge::GRAPH_FAILED);
    auto* outputShape = context->GetOutputShape(OUTPUT_PACKED);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);
    outputShape->SetDimNum(3);
    outputShape->SetDim(0, queryShape->GetDim(1));
    outputShape->SetDim(1, (queryShape->GetDim(0) + TOKEN_ALIGNMENT - 1) / TOKEN_ALIGNMENT * TOKEN_ALIGNMENT);
    outputShape->SetDim(2, PACKED_QUERY_WIDTH);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeTqQueryPack(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TqQueryPack", "infer dtype context is null"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(OUTPUT_PACKED, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TqQueryPack).InferShape(InferShapeTqQueryPack).InferDataType(InferDataTypeTqQueryPack);
} // namespace ops
