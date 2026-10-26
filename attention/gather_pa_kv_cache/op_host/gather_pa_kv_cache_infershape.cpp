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
 * \file gather_pa_kv_cache.cc
 * \brief
 */

#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
namespace {
static constexpr size_t KEY_IN_INDEX = 4;
static constexpr size_t VALUE_IN_INDEX = 5;
static constexpr size_t KEY_OUT_INDEX = 0;
static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr size_t DIM_4 = 4;
static constexpr size_t DIM_5 = 5;
static constexpr size_t DIM_6 = 6;
static constexpr size_t IS_SEQ_LENS_CUMSUM_INDEX = 1;
static constexpr size_t CACHE_MODE_INDEX = 0;

static ge::graphStatus CheckCommonPagedCacheLoad(gert::InferShapeContext* context)
{
    auto blockTables_shape = context->GetInputShape(DIM_2);
    OP_CHECK_IF(blockTables_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "block_tables"),
                return ge::GRAPH_FAILED);
    auto seqLens_shape = context->GetInputShape(DIM_3);
    OP_CHECK_IF(seqLens_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "seq_lens"),
                return ge::GRAPH_FAILED);
    if (blockTables_shape->GetDimNum() != DIM_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "block_tables",
                                     std::to_string(blockTables_shape->GetDimNum()), "2");
        return ge::GRAPH_FAILED;
    }
    if (seqLens_shape->GetDimNum() != DIM_1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "seq_lens", std::to_string(seqLens_shape->GetDimNum()),
                                     "1");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    auto is_seq_lens_cumsum = attrs->GetAttrPointer<bool>(IS_SEQ_LENS_CUMSUM_INDEX);
    if (is_seq_lens_cumsum == nullptr || *is_seq_lens_cumsum) {
        if ((blockTables_shape->GetDim(DIM_0) + 1) != seqLens_shape->GetDim(DIM_0)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "seq_lens, block_tables",
                Ops::Base::ToString(*seqLens_shape) + ", " + Ops::Base::ToString(*blockTables_shape),
                "seqLens's dim_0 " + std::to_string(seqLens_shape->GetDim(DIM_0)) +
                    " must equal blockTables's dim_0+1 " + std::to_string(blockTables_shape->GetDim(DIM_0) + 1));
            return ge::GRAPH_FAILED;
        }
    } else {
        if (blockTables_shape->GetDim(DIM_0) != seqLens_shape->GetDim(DIM_0)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "seq_lens, block_tables",
                Ops::Base::ToString(*seqLens_shape) + ", " + Ops::Base::ToString(*blockTables_shape),
                "seqLens's dim_0 " + std::to_string(seqLens_shape->GetDim(DIM_0)) + " must equal blockTables's dim_0 " +
                    std::to_string(blockTables_shape->GetDim(DIM_0)));
            return ge::GRAPH_FAILED;
        }
    };
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GatherPaKvCacheNz(gert::InferShapeContext* context)
{
    auto valueCacheShape = context->GetInputShape(DIM_1);
    OP_CHECK_IF(valueCacheShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "value_cache"),
                return ge::GRAPH_FAILED);
    auto keyCacheShape = context->GetInputShape(DIM_0);
    OP_CHECK_IF(keyCacheShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "key_cache"),
                return ge::GRAPH_FAILED);
    if (keyCacheShape->GetDimNum() != DIM_4 || valueCacheShape->GetDimNum() != DIM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "key_cache, value_cache",
            std::to_string(keyCacheShape->GetDimNum()) + ", " + std::to_string(valueCacheShape->GetDimNum()),
            "KCache&VCache's dim num should be 4");
        return ge::GRAPH_FAILED;
    }
    auto valueOutshape = context->GetOutputShape(DIM_1);
    OP_CHECK_IF(valueOutshape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "value output shape"),
                return ge::GRAPH_FAILED);
    auto keyOutshape = context->GetOutputShape(DIM_0);
    OP_CHECK_IF(keyOutshape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "key output shape"),
                return ge::GRAPH_FAILED);

    if (keyOutshape->GetDimNum() != DIM_2 || valueOutshape->GetDimNum() != DIM_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "key, value",
            std::to_string(keyOutshape->GetDimNum()) + ", " + std::to_string(valueOutshape->GetDimNum()),
            "K&V's dim num should be 2");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GatherPaKvCacheNd(gert::InferShapeContext* context)
{
    auto keyCacheShape = context->GetInputShape(DIM_0);
    OP_CHECK_IF(keyCacheShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "key_cache"),
                return ge::GRAPH_FAILED);
    auto valueCacheShape = context->GetInputShape(DIM_1);
    OP_CHECK_IF(valueCacheShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "value_cache"),
                return ge::GRAPH_FAILED);
    auto keyOutshape = context->GetOutputShape(DIM_0);
    OP_CHECK_IF(keyOutshape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "key output shape"),
                return ge::GRAPH_FAILED);
    auto valueOutshape = context->GetOutputShape(DIM_1);
    OP_CHECK_IF(valueOutshape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "value output shape"),
                return ge::GRAPH_FAILED);

    if (keyCacheShape->GetDimNum() != DIM_4 || valueCacheShape->GetDimNum() != DIM_4) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "key_cache, value_cache",
            std::to_string(keyCacheShape->GetDimNum()) + ", " + std::to_string(valueCacheShape->GetDimNum()),
            "KCache&VCache's dim num should be 4");
        return ge::GRAPH_FAILED;
    }
    if (keyOutshape->GetDimNum() != DIM_3 || valueOutshape->GetDimNum() != DIM_3) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "key, value",
            std::to_string(keyOutshape->GetDimNum()) + ", " + std::to_string(valueOutshape->GetDimNum()),
            "K&V's dim num should be 3");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GatherPaKvCache(gert::InferShapeContext* context)
{
    OP_LOGD(context, "Begin to do InferShape4GatherPaKvCache");
    auto tensorKeyIn_shape = context->GetInputShape(DIM_4);
    OP_CHECK_IF(tensorKeyIn_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "key"),
                return ge::GRAPH_FAILED);
    auto tensorValueIn_shape = context->GetInputShape(DIM_5);
    OP_CHECK_IF(tensorValueIn_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "value"),
                return ge::GRAPH_FAILED);
    auto output0_shape = context->GetOutputShape(DIM_0);
    OP_CHECK_IF(output0_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "key output shape"),
                return ge::GRAPH_FAILED);
    auto output1_shape = context->GetOutputShape(DIM_1);
    OP_CHECK_IF(output1_shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "value output shape"),
                return ge::GRAPH_FAILED);
    *output0_shape = *tensorKeyIn_shape;
    *output1_shape = *tensorValueIn_shape;

    for (uint32_t i = 0; i < context->GetComputeNodeInputNum(); i++) {
        const gert::Shape* shape = context->GetInputShape(i);
        if (Ops::Base::IsUnknownRank(*shape)) {
            Ops::Base::SetUnknownRank(*output0_shape);
            Ops::Base::SetUnknownRank(*output1_shape);
            OP_LOGD(context,
                    "input[%u].shape is UnknownRank, set output key.shape and "
                    "value.shape to -2.",
                    i);
            return ge::GRAPH_SUCCESS;
        }
    }
    for (uint32_t i = 0; i < context->GetComputeNodeInputNum(); i++) {
        auto shape = context->GetInputShape(i);
        if (Ops::Base::IsUnknownShape(*shape)) {
            OP_LOGD(context,
                    "input[%u].shape is UnknownShape, set output key.shape and value.shape "
                    "same as input key.shape and value.shape respectively.",
                    i);
            return ge::GRAPH_SUCCESS;
        }
    }

    OP_CHECK_IF(CheckCommonPagedCacheLoad(context) == ge::GRAPH_FAILED,
                OP_LOGE_WITHOUT_REPORT(context, "Failed to check common launch param."), return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    const char* mode = attrs->GetAttrPointer<char>(CACHE_MODE_INDEX);
    if (mode == nullptr || strcmp(mode, "Norm") == 0) {
        return InferShape4GatherPaKvCacheNd(context);
    } else if (strcmp(mode, "PA_NZ") == 0) {
        return InferShape4GatherPaKvCacheNz(context);
    }
    OP_LOGD(context, "The Input Format Only Support ND and NZ!.");
    return ge::GRAPH_FAILED;
}

static graphStatus InferDataType4GatherPaKvCache(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(DIM_0, context->GetInputDataType(DIM_0));
    context->SetOutputDataType(DIM_1, context->GetInputDataType(DIM_1));
    return ge::GRAPH_SUCCESS;
}
} // namespace

IMPL_OP_INFERSHAPE(GatherPaKvCache).InferShape(InferShape4GatherPaKvCache).InferDataType(InferDataType4GatherPaKvCache);
} // namespace ops
