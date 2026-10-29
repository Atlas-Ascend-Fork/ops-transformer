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
 * \file unpermute_qkv_infershape.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include "err/ops_err.h"

namespace ops {
static ge::graphStatus InferShapeUnpermuteQkv(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("UnpermuteQkv", "context is nullptr!"), return ge::GRAPH_FAILED);
    // Output shapes are provided by the caller (torch adapter derives them from n_k).
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeUnpermuteQkv(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("UnpermuteQkv", "context is nullptr!"), return ge::GRAPH_FAILED);
    auto inType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inType);
    context->SetOutputDataType(1, inType);
    context->SetOutputDataType(2, inType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(UnpermuteQkv).InferShape(InferShapeUnpermuteQkv).InferDataType(InferDataTypeUnpermuteQkv);
} // namespace ops
