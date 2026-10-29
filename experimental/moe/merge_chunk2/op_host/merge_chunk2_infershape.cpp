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
 * \file merge_chunk2_infershape.cpp
 * \brief MergeChunk2 的 shape / dtype 推导
 *
 * out1 = [t1 + t2, h1]
 * out2 = [t1 + t2, h2]
 * group_list = [numExpert]
 */
#include "register/op_impl_registry.h"

using namespace ge;

namespace ops {
namespace {
ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* in1Chunk1Shape = context->GetInputShape(0);
    const gert::Shape* in1Chunk2Shape = context->GetInputShape(1);
    const gert::Shape* in2Chunk1Shape = context->GetInputShape(2);
    const gert::Shape* in2Chunk2Shape = context->GetInputShape(3);
    const gert::Shape* groupListChunk1Shape = context->GetInputShape(4);

    gert::Shape* out1Shape = context->GetOutputShape(0);
    gert::Shape* out2Shape = context->GetOutputShape(1);
    gert::Shape* groupListShape = context->GetOutputShape(2);

    const int64_t t1c1 = in1Chunk1Shape->GetDim(0);
    const int64_t t1c2 = in1Chunk2Shape->GetDim(0);
    const int64_t t2c1 = in2Chunk1Shape->GetDim(0);
    const int64_t t2c2 = in2Chunk2Shape->GetDim(0);
    const int64_t out1Token = (t1c1 >= 0 && t1c2 >= 0) ? t1c1 + t1c2 : -1;
    const int64_t out2Token = (t2c1 >= 0 && t2c2 >= 0) ? t2c1 + t2c2 : -1;

    *out1Shape = gert::Shape({out1Token, in1Chunk1Shape->GetDim(1)});
    *out2Shape = gert::Shape({out2Token, in2Chunk1Shape->GetDim(1)});
    *groupListShape = gert::Shape({groupListChunk1Shape->GetDim(0)});
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(2));
    context->SetOutputDataType(2, context->GetInputDataType(4));
    return GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(MergeChunk2).InferShape(InferShape).InferDataType(InferDataType);
} // namespace ops
