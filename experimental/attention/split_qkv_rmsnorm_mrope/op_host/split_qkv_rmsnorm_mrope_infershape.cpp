/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr int64_t Q_HEADS = 16;
constexpr int64_t HEAD_DIM = 256;
} // namespace

static ge::graphStatus InferShapeSplitQkvRmsnormMrope(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* qkvShape = context->GetInputShape(0);
    const auto* attrs = context->GetAttrs();
    if (qkvShape == nullptr || attrs == nullptr || qkvShape->GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }
    const auto* qHeads = attrs->GetAttrPointer<int64_t>(0);
    const auto* kvHeads = attrs->GetAttrPointer<int64_t>(1);
    const auto* hasGate = attrs->GetAttrPointer<bool>(4);
    if (qHeads == nullptr || kvHeads == nullptr || hasGate == nullptr || *qHeads != Q_HEADS ||
        (*kvHeads != 2 && *kvHeads != 4)) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tokens = qkvShape->GetDim(0);
    const int64_t qSize = *qHeads * HEAD_DIM;
    const int64_t kvSize = *kvHeads * HEAD_DIM;
    const int64_t expectedWidth = (*hasGate ? 2 : 1) * qSize + 2 * kvSize;
    if (tokens <= 0 || qkvShape->GetDim(1) != expectedWidth) {
        return ge::GRAPH_FAILED;
    }
    for (size_t index = 0; index < 4; ++index) {
        if (context->GetOutputShape(index) == nullptr) {
            return ge::GRAPH_FAILED;
        }
    }
    context->GetOutputShape(0)->SetDimNum(2);
    context->GetOutputShape(0)->SetDim(0, tokens);
    context->GetOutputShape(0)->SetDim(1, qSize);
    context->GetOutputShape(1)->SetDimNum(2);
    context->GetOutputShape(1)->SetDim(0, tokens);
    context->GetOutputShape(1)->SetDim(1, kvSize);
    context->GetOutputShape(2)->SetDimNum(2);
    context->GetOutputShape(2)->SetDim(0, tokens);
    context->GetOutputShape(2)->SetDim(1, kvSize);
    context->GetOutputShape(3)->SetDimNum(2);
    context->GetOutputShape(3)->SetDim(0, tokens);
    context->GetOutputShape(3)->SetDim(1, *hasGate ? qSize : 0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeSplitQkvRmsnormMrope(gert::InferDataTypeContext* context)
{
    const auto dataType = context->GetInputDataType(0);
    for (size_t index = 0; index < 4; ++index) {
        context->SetOutputDataType(index, dataType);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SplitQkvRmsnormMrope)
    .InferShape(InferShapeSplitQkvRmsnormMrope)
    .InferDataType(InferDataTypeSplitQkvRmsnormMrope);
} // namespace ops
