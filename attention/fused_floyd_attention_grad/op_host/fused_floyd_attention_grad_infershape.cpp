/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_floyd_attention_grad_infershape.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;

namespace ops {

static const uint64_t DIM_NUM_2 = 2;

ge::graphStatus InferShape4FusedFloydAttentionGrad(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"FusedFloydAttentionGrad", "InferShape context is unavailable"})),
                return ge::GRAPH_FAILED);
    OP_LOGI(context, "Enter FusedFloydAttentionGrad runtime infershape impl.");
    const gert::Shape* queryShape = context->GetInputShape(0);
    OP_CHECK_IF(queryShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "query",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* key1Shape = context->GetInputShape(1);
    OP_CHECK_IF(key1Shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "key1",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* value1Shape = context->GetInputShape(2);
    OP_CHECK_IF(value1Shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "value1",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* key2Shape = context->GetInputShape(3);
    OP_CHECK_IF(key2Shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "key2",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* value2Shape = context->GetInputShape(4);
    OP_CHECK_IF(value2Shape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "value2",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);

    gert::Shape* dqShape = context->GetOutputShape(0);
    OP_CHECK_IF(dqShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "dqShape is unavailable during InferShape"})),
                return ge::GRAPH_FAILED);
    *dqShape = *queryShape;

    gert::Shape* dk1Shape = context->GetOutputShape(1);
    OP_CHECK_IF(dk1Shape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "dk1Shape is unavailable during InferShape"})),
                return ge::GRAPH_FAILED);
    *dk1Shape = *key1Shape;

    gert::Shape* dv1Shape = context->GetOutputShape(2);
    OP_CHECK_IF(dv1Shape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "dv1Shape is unavailable during InferShape"})),
                return ge::GRAPH_FAILED);
    *dv1Shape = *value1Shape;

    gert::Shape* dk2Shape = context->GetOutputShape(3);
    OP_CHECK_IF(dk2Shape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "dk2Shape is unavailable during InferShape"})),
                return ge::GRAPH_FAILED);
    *dk2Shape = *key2Shape;

    gert::Shape* dv2Shape = context->GetOutputShape(4);
    OP_CHECK_IF(dv2Shape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "dv2Shape is unavailable during InferShape"})),
                return ge::GRAPH_FAILED);
    *dv2Shape = *value2Shape;

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4FusedFloydAttentionGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"FusedFloydAttentionGrad", "InferDataType context is unavailable"})),
                return ge::GRAPH_FAILED);
    OP_LOGI(context, "Enter FusedFloydAttentionGrad infer data type impl.");

    auto dtype = context->GetInputDataType(0);
    // dq, outidx:0
    context->SetOutputDataType(0, dtype);
    // dk1, outidx:1
    context->SetOutputDataType(1, dtype);
    // dv1, outidx:2
    context->SetOutputDataType(2, dtype);
    // dk2, outidx:3
    context->SetOutputDataType(3, dtype);
    // dv2, outidx:4
    context->SetOutputDataType(4, dtype);
    return GRAPH_SUCCESS;
}

IMPL_OP(FusedFloydAttentionGrad)
    .InferShape(InferShape4FusedFloydAttentionGrad)
    .InferDataType(InferDataType4FusedFloydAttentionGrad);

} // namespace ops
