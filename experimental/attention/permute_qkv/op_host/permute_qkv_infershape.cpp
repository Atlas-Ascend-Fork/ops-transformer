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
 * \file permute_qkv_infershape.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include "err/ops_err.h"

namespace ops {
constexpr uint32_t SPLIT_SIZE_ATTR_INDEX = 0;

static ge::graphStatus InferShapePermuteQkv(gert::InferShapeContext* context)
{
    const gert::Shape* qShape = context->GetInputShape(0); // [s_c, b, q_head_num, h]
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    const gert::Shape* kShape = context->GetInputShape(1); // [s_c, b, n_k, h]
    OP_CHECK_NULL_WITH_CONTEXT(context, kShape);
    gert::Shape* outShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* splitSizePtr = attrs->GetInt(SPLIT_SIZE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, splitSizePtr);

    const int64_t sC = qShape->GetDim(0);
    const int64_t b = qShape->GetDim(1);
    const int64_t qHeadNum = qShape->GetDim(2);
    const int64_t h = qShape->GetDim(3);
    const int64_t nK = kShape->GetDim(2);
    const int64_t totalHead = qHeadNum + 2 * nK;
    const int64_t splitSize = *splitSizePtr;

    OP_CHECK_IF(splitSize <= 0, OP_LOGE(context->GetNodeName(), "split_size must be positive."),
                return ge::GRAPH_FAILED);

    // output: [totalHead, s_c, b, h]
    outShape->SetDimNum(4);
    outShape->SetDim(0, totalHead);
    outShape->SetDim(1, sC);
    outShape->SetDim(2, b);
    outShape->SetDim(3, h);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypePermuteQkv(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("PermuteQkv", "context is nullptr!"), return ge::GRAPH_FAILED);
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(PermuteQkv).InferShape(InferShapePermuteQkv).InferDataType(InferDataTypePermuteQkv);
} // namespace ops
