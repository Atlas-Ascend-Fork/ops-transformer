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
 * \file dsa_dense_infershape.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX_POS = 7;
constexpr uint32_t KEY_INDEX_POS = 8;

constexpr uint32_t LOSS_POS = 0;
constexpr uint32_t DQ_POS = 1;
constexpr uint32_t DK_POS = 2;
constexpr uint32_t DW_POS = 3;

static ge::graphStatus InferShapeDsaDense(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("DsaDense", "InferShapeContext is nullptr!"), return ge::GRAPH_FAILED);

    const gert::Shape* qIndexShape = context->GetInputShape(QUERY_INDEX_POS); // [T1 NI DI]
    OP_CHECK_NULL_WITH_CONTEXT(context, qIndexShape);
    const gert::Shape* kIndexShape = context->GetInputShape(KEY_INDEX_POS); // [T2 DI]
    OP_CHECK_NULL_WITH_CONTEXT(context, kIndexShape);

    gert::Shape* lossShape = context->GetOutputShape(LOSS_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, lossShape);
    gert::Shape* dqShape = context->GetOutputShape(DQ_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, dqShape);
    gert::Shape* dkShape = context->GetOutputShape(DK_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, dkShape);
    gert::Shape* dwShape = context->GetOutputShape(DW_POS);
    OP_CHECK_NULL_WITH_CONTEXT(context, dwShape);

    OP_CHECK_IF(qIndexShape->GetDimNum() != 3, OP_LOGE(context->GetNodeName(), "queryIndex dims must be 3!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kIndexShape->GetDimNum() != 2, OP_LOGE(context->GetNodeName(), "keyIndex dims must be 2!"),
                return ge::GRAPH_FAILED);

    // loss: [1]
    lossShape->SetDimNum(1);
    lossShape->SetDim(0, 1);

    // dq: [T1 NI DI]
    dqShape->SetDimNum(3);
    dqShape->SetDim(0, qIndexShape->GetDim(0));
    dqShape->SetDim(1, qIndexShape->GetDim(1));
    dqShape->SetDim(2, qIndexShape->GetDim(2));

    // dk: [T2 DI]
    dkShape->SetDimNum(2);
    dkShape->SetDim(0, kIndexShape->GetDim(0));
    dkShape->SetDim(1, kIndexShape->GetDim(1));

    // dw: [NI T1]
    dwShape->SetDimNum(2);
    dwShape->SetDim(0, qIndexShape->GetDim(1));
    dwShape->SetDim(1, qIndexShape->GetDim(0));

    OP_LOGD(context->GetNodeName(), "DsaDense InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDsaDense(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("DsaDense", "InferDataTypeContext is nullptr!"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(LOSS_POS, ge::DT_FLOAT);
    context->SetOutputDataType(DQ_POS, context->GetInputDataType(QUERY_INDEX_POS));
    context->SetOutputDataType(DK_POS, context->GetInputDataType(KEY_INDEX_POS));
    context->SetOutputDataType(DW_POS, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaDense).InferShape(InferShapeDsaDense).InferDataType(InferDataTypeDsaDense);
} // namespace ops
