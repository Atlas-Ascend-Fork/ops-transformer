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
 * \file fused_floyd_attention_grad_tiling_common.cpp
 * \brief
 */

#include "fused_floyd_attention_grad_tiling_common.h"
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {
namespace FFAG {
static std::string ShapeToString(const gert::StorageShape* shape)
{
    if (shape == nullptr) {
        return "null";
    }
    const gert::Shape& shapeValue = shape->GetStorageShape();
    std::string shapeStr = "[";
    for (uint32_t i = 0; i < shapeValue.GetDimNum(); i++) {
        shapeStr += std::to_string(shapeValue.GetDim(i));
        if (i + 1 != shapeValue.GetDimNum()) {
            shapeStr += ",";
        }
    }
    return shapeStr + "]";
}

bool CheckSameShape(const gert::StorageShape* aShape, const gert::StorageShape* bShape)
{
    OP_CHECK_IF((aShape == nullptr) || (bShape == nullptr),
                OP_LOGW("fused_floyd_attention_grad_tiling_common", "aShape or bShape is nullptr."), return false);
    uint32_t dimNumA = aShape->GetStorageShape().GetDimNum();
    uint32_t dimNumB = bShape->GetStorageShape().GetDimNum();
    if (dimNumA != dimNumB || dimNumA != SUPPORT_DIM_NUM) {
        return false;
    }

    for (uint32_t i = 0; i < dimNumA; i++) {
        auto dimA = aShape->GetStorageShape().GetDim(i);
        auto dimB = bShape->GetStorageShape().GetDim(i);
        if (dimA != dimB) {
            return false;
        }
    }
    return true;
}

ge::graphStatus CheckSupportShape(gert::TilingContext* context)
{
    const gert::StorageShape* queryShape = context->GetInputShape(QUERY);
    const gert::StorageShape* key1Shape = context->GetInputShape(KEY_1);
    const gert::StorageShape* value1Shape = context->GetInputShape(VALUE_1);
    const gert::StorageShape* key2Shape = context->GetInputShape(KEY_2);
    const gert::StorageShape* value2Shape = context->GetInputShape(VALUE_2);
    const gert::StorageShape* dyShape = context->GetInputShape(DY);
    const gert::StorageShape* attentionInShape = context->GetOptionalInputShape(ATTENTION_IN);

    OP_CHECK_IF(!CheckSameShape(key1Shape, value1Shape),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "key1 and value1",
                    (ShapeToString(key1Shape) + " and " + ShapeToString(value1Shape)).c_str(),
                    "key1 and value1 must have the same shape"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckSameShape(key2Shape, value2Shape),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "key2 and value2",
                    (ShapeToString(key2Shape) + " and " + ShapeToString(value2Shape)).c_str(),
                    "key2 and value2 must have the same shape"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        !CheckSameShape(queryShape, dyShape),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "query and dy",
                                               (ShapeToString(queryShape) + " and " + ShapeToString(dyShape)).c_str(),
                                               "query and dy must have the same shape"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(!CheckSameShape(queryShape, attentionInShape),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "query and attentionIn",
                    (ShapeToString(queryShape) + " and " + ShapeToString(attentionInShape)).c_str(),
                    "query and attentionIn must have the same shape"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionMaskShape(gert::TilingContext* context, int64_t b, int64_t s1, int64_t s3)
{
    auto attentionMaskShape = context->GetOptionalInputShape(ATTEN_MASK);
    if (attentionMaskShape == nullptr) {
        OP_LOGE(context, "FusedFloydAttentionGrad does not support a null atten_mask.");
        return ge::GRAPH_FAILED;
    }
    auto attentionMaskShapeDim = attentionMaskShape->GetStorageShape().GetDimNum();
    if (attentionMaskShapeDim != SUPPORT_DIM_NUM) { // attentionMask only support 5 dimensions
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "atten_mask",
                                                 std::to_string(attentionMaskShapeDim).c_str(),
                                                 "atten_mask must be a 5D tensor");
        return ge::GRAPH_FAILED;
    }
    auto dim0 = attentionMaskShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = attentionMaskShape->GetStorageShape().GetDim(1); // 1:1
    auto dim2 = attentionMaskShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = attentionMaskShape->GetStorageShape().GetDim(3); // 3:1
    auto dim4 = attentionMaskShape->GetStorageShape().GetDim(4); // 4:s3

    OP_CHECK_IF((dim0 != b || dim1 != 1 || dim2 != s1 || dim3 != 1 || dim4 != s3),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "atten_mask", ShapeToString(attentionMaskShape).c_str(),
                    "atten_mask shape must be (b, 1, s1, 1, s3) derived from the query and key1 shapes"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxMaxAndSumShape(gert::TilingContext* context, int64_t b, int64_t n, int64_t s1, int64_t s2,
                                           uint8_t inputIdx)
{
    auto softmaxMaxAndSumShape = context->GetOptionalInputShape(inputIdx);
    if (softmaxMaxAndSumShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxMaxAndSumShapeDim = softmaxMaxAndSumShape->GetStorageShape().GetDimNum();
    if (softmaxMaxAndSumShapeDim != SUPPORT_DIM_NUM) { // softmaxMax/softmaxSum only support 5 dimensions
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "softmax_max/softmax_sum",
                                                 std::to_string(softmaxMaxAndSumShapeDim).c_str(),
                                                 "softmax_max and softmax_sum must be 5D tensors");
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxMaxAndSumShape->GetStorageShape().GetDim(0); // 0:b
    auto dim1 = softmaxMaxAndSumShape->GetStorageShape().GetDim(1); // 1:n
    auto dim2 = softmaxMaxAndSumShape->GetStorageShape().GetDim(2); // 2:s1
    auto dim3 = softmaxMaxAndSumShape->GetStorageShape().GetDim(3); // 3:s2
    auto dim4 = softmaxMaxAndSumShape->GetStorageShape().GetDim(4); // 4:8

    // softmaxMaxAndSum pad to 8
    OP_CHECK_IF((dim0 != b || dim1 != n || dim2 != s1 || dim3 != s2 || dim4 != 8),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "softmax_max/softmax_sum", ShapeToString(softmaxMaxAndSumShape).c_str(),
                    "softmax_max and softmax_sum shape must be (b, n, s1, s2, 8), the last dim is padded to 8"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckInputShapeValid(gert::TilingContext* context, int64_t b, int64_t n, int64_t s1, int64_t s2,
                                     int64_t s3, int64_t d)
{
    OP_CHECK_IF((b == 0 || n == 0 || s1 == 0 || s2 == 0 || s3 == 0 || d == 0),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "query and key1", "0",
                                                         "all shape dimensions must be greater than 0"),
                return ge::GRAPH_FAILED);

    if (s1 % ALIGNED_NUM_16 != 0 || s2 % ALIGNED_NUM_128 != 0) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "query", std::to_string(s1) + ", " + std::to_string(s2),
            "query dim 2 (G) must be a multiple of 16 and dim 3 (S1) must be a multiple of 128");
        return ge::GRAPH_FAILED;
    }

    if (s3 % ALIGNED_NUM_128 != 0) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "key1", std::to_string(s3),
                                                 "key1 dim 3 (S2) must be a multiple of 128");
        return ge::GRAPH_FAILED;
    }

    if (d != 32 && d != 64 && d != 128) { // 32, 64, 128: supported head dims
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "query", std::to_string(d),
                                                 "query head dimension must be one of 32, 64 and 128");
        return ge::GRAPH_FAILED;
    }

    auto ret = CheckSupportShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionMaskShape(context, b, s1, s3);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckSoftmaxMaxAndSumShape(context, b, n, s1, s2, SOFTMAX_SUM);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckSoftmaxMaxAndSumShape(context, b, n, s1, s2, SOFTMAX_MAX);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}
} // namespace FFAG
} // namespace optiling
