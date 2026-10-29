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
 * \file flash_attention_infershape.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include "err/ops_err.h"

namespace ops {
static ge::graphStatus InferShapeFlashAttention(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(0);
    const gert::Shape* vShape = context->GetInputShape(2);
    if (qShape->GetDimNum() != 3) { // only support TND
        OP_LOGE(context->GetNodeName(), "Dim of Q not right.");
        return ge::GRAPH_FAILED;
    }
    if (vShape->GetDimNum() != 3) { // only support TND
        OP_LOGE(context->GetNodeName(), "Dim of V not right.");
        return ge::GRAPH_FAILED;
    }
    int t1 = qShape->GetDim(0);
    int n1 = qShape->GetDim(1);
    int dv = vShape->GetDim(2);

    gert::Shape* softmaxMaxShape = context->GetOutputShape(0);
    gert::Shape* softmaxSumShape = context->GetOutputShape(1);
    gert::Shape* lseOutShape = context->GetOutputShape(2);
    gert::Shape* attentionOutShape = context->GetOutputShape(3);

    *softmaxMaxShape = gert::Shape({0});  // useless
    *softmaxSumShape = gert::Shape({0});  // useless
    *lseOutShape = gert::Shape({n1, t1}); // TODO: B N S
    *attentionOutShape = gert::Shape({t1, n1, dv});
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeFlashAttention(gert::InferDataTypeContext* context)
{
    auto qType = context->GetInputDataType(0);
    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_FLOAT);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, qType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FlashAttention).InferShape(InferShapeFlashAttention).InferDataType(InferDataTypeFlashAttention);
} // namespace ops
