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
 * \file block_indexer_rearrange_infershape.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t TOPK_INPUT_INDEX = 0;
constexpr uint32_t TOPK_OUTPUT_INDEX = 0;
constexpr size_t BLOCK_SIZE_ATTR_INDEX = 1;

static ge::graphStatus InferShapeBlockIndexerRearrange(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("BlockIndexerRearrange", "InferShapeContext is nullptr!"),
                return ge::GRAPH_FAILED);

    const gert::Shape* topkShape = context->GetInputShape(TOPK_INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, topkShape);
    gert::Shape* outShape = context->GetOutputShape(TOPK_OUTPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, outShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* blockSizePtr = attrs->GetInt(BLOCK_SIZE_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockSizePtr);

    // topk_output: [T1, N2, blockCount * blockSize]
    outShape->SetDimNum(topkShape->GetDimNum());
    outShape->SetDim(0, topkShape->GetDim(0));
    outShape->SetDim(1, topkShape->GetDim(1));
    outShape->SetDim(2, topkShape->GetDim(2) * (*blockSizePtr));

    OP_LOGD(context->GetNodeName(), "BlockIndexerRearrange InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBlockIndexerRearrange(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("BlockIndexerRearrange", "InferDataTypeContext is nullptr!"),
                return ge::GRAPH_FAILED);
    context->SetOutputDataType(TOPK_OUTPUT_INDEX, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlockIndexerRearrange)
    .InferShape(InferShapeBlockIndexerRearrange)
    .InferDataType(InferDataTypeBlockIndexerRearrange);
} // namespace ops
