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
#include "merge_chunk_shape.h"

namespace ops {
namespace {
ge::graphStatus InferShapeMergeChunk(gert::InferShapeContext* context)
{
    int64_t paths = 0;
    int64_t chunks = 0;
    if (!MergeChunkSupport::ReadListDimensions(context, paths, chunks) ||
        context->GetComputeNodeOutputNum() != static_cast<size_t>(paths + 1)) {
        return ge::GRAPH_FAILED;
    }
    MergeChunkSupport::InferredLayout layout;
    const auto* error = MergeChunkSupport::InferLayout(
        [context](size_t i) { return context->GetDynamicInputShape(MergeChunkSupport::DATA_INDEX, i); },
        [context](size_t i) { return context->GetDynamicInputShape(MergeChunkSupport::COUNTS_INDEX, i); }, layout);
    if (error != nullptr) {
        OP_LOGE(context, "%s", error);
        return ge::GRAPH_FAILED;
    }
    for (int64_t path = 0; path < paths; ++path) {
        auto* shape = context->GetOutputShape(path);
        OP_CHECK_NULL_WITH_CONTEXT(context, shape);
        *shape = gert::Shape({layout.totalRows, layout.hidden[path]});
    }
    auto* counts = context->GetOutputShape(paths);
    OP_CHECK_NULL_WITH_CONTEXT(context, counts);
    *counts = gert::Shape({layout.experts});
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeMergeChunk(gert::InferDataTypeContext* context)
{
    int64_t paths = 0;
    int64_t chunks = 0;
    if (!MergeChunkSupport::ReadListDimensions(context, paths, chunks) ||
        context->GetComputeNodeOutputNum() != static_cast<size_t>(paths + 1)) {
        return ge::GRAPH_FAILED;
    }
    const auto type = context->GetDynamicInputDataType(MergeChunkSupport::DATA_INDEX, 0);
    if (type != ge::DT_FLOAT16 && type != ge::DT_BF16) {
        return ge::GRAPH_FAILED;
    }
    for (int64_t path = 0; path < paths; ++path) {
        context->SetOutputDataType(path, type);
    }
    context->SetOutputDataType(paths, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_INFERSHAPE(MergeChunk).InferShape(InferShapeMergeChunk).InferDataType(InferDataTypeMergeChunk);
} // namespace ops
