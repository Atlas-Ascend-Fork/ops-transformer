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
 * \file nsa_selected_attention_grad_tiling_common.cpp
 * \brief
 */

#include "nsa_selected_attention_grad_tiling_common.h"
#include "log/log.h"
#include "err/ops_err.h"
namespace optiling {
namespace nsa {

ge::graphStatus CheckTndSoftmaxMaxShape(gert::TilingContext* context, int64_t t1, int64_t n1)
{
    auto softmaxMaxShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_MAX));
    if (softmaxMaxShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxMaxShapeDim = softmaxMaxShape->GetStorageShape().GetDimNum();
    if (softmaxMaxShapeDim != 3) { // TND softmaxMax only support 3 dimensions
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "softmax_max", std::to_string(softmaxMaxShapeDim),
            "softmax_max must be 3-dimensional in TND layout (T1, N1, 8); check actual_seq_qlen and head_num");
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxMaxShape->GetStorageShape().GetDim(0); // 0:t1
    auto dim1 = softmaxMaxShape->GetStorageShape().GetDim(1); // 1:n1
    auto dim2 = softmaxMaxShape->GetStorageShape().GetDim(2); // 2:8

    // softmaxMax pad to 8
    OP_CHECK_IF((dim0 != t1 || dim1 != n1 || dim2 != 8),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "softmax_max",
                    "(" + std::to_string(dim0) + "," + std::to_string(dim1) + "," + std::to_string(dim2) +
                        "), should be (" + std::to_string(t1) + "," + std::to_string(n1) + ",8)",
                    "softmax_max shape must be (T1, N1, 8); check actual_seq_qlen and head_num"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndSoftmaxSumShape(gert::TilingContext* context, int64_t t1, int64_t n1)
{
    auto softmaxSumShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_SUM));
    if (softmaxSumShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxSumShapeDim = softmaxSumShape->GetStorageShape().GetDimNum();
    if (softmaxSumShapeDim != 3) { // TND softmaxSum only support 3 dimensions
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "softmax_sum", std::to_string(softmaxSumShapeDim),
            "softmax_sum must be 3-dimensional in TND layout (T1, N1, 8); check actual_seq_qlen and head_num");
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxSumShape->GetStorageShape().GetDim(0); // 0:t1
    auto dim1 = softmaxSumShape->GetStorageShape().GetDim(1); // 1:n1
    auto dim2 = softmaxSumShape->GetStorageShape().GetDim(2); // 2:8

    // softmaxSum pad to 8
    OP_CHECK_IF((dim0 != t1 || dim1 != n1 || dim2 != 8),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    context->GetNodeName(), "softmax_sum",
                    "(" + std::to_string(dim0) + "," + std::to_string(dim1) + "," + std::to_string(dim2) +
                        "), should be (" + std::to_string(t1) + "," + std::to_string(n1) + ",8)",
                    "softmax_sum shape must be (T1, N1, 8); check actual_seq_qlen and head_num"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInShape(gert::TilingContext* context)
{
    auto attentionInShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::ATTENTION_OUT));
    if (attentionInShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    auto valueShape = context->GetInputShape(static_cast<size_t>(InputIndex::VALUE));
    auto attentionInShapeDim = attentionInShape->GetStorageShape().GetDimNum();
    auto queryShapeDim = queryShape->GetStorageShape().GetDimNum();
    auto valueShapeDim = valueShape->GetStorageShape().GetDimNum();
    if (attentionInShapeDim != queryShapeDim) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "attention_in and query",
            std::to_string(attentionInShapeDim) + ", should be " + std::to_string(queryShapeDim),
            "attention_in dim count must equal query dim count");
        return ge::GRAPH_FAILED;
    }
    if (attentionInShapeDim != valueShapeDim) {
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            context->GetNodeName(), "attention_in and value",
            std::to_string(attentionInShapeDim) + ", should be " + std::to_string(valueShapeDim),
            "attention_in dim count must equal value dim count");
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < queryShapeDim - 1; i++) {
        if (attentionInShape->GetStorageShape().GetDim(i) != queryShape->GetStorageShape().GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "attention_in and query",
                                                   "dim " + std::to_string(i) + " of attention_in is " +
                                                       std::to_string(attentionInShape->GetStorageShape().GetDim(i)) +
                                                       ", should be " +
                                                       std::to_string(queryShape->GetStorageShape().GetDim(i)),
                                                   "attention_in shape must match query shape except the last dim");
            return ge::GRAPH_FAILED;
        }
    }
    // check head_dim, attention D == value D
    if (attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1) !=
        valueShape->GetStorageShape().GetDim(queryShapeDim - 1)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "attention_in and value",
            "last dim of attention_in is " +
                std::to_string(attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1)) + ", should be " +
                std::to_string(valueShape->GetStorageShape().GetDim(queryShapeDim - 1)),
            "the last dim (head_dim) of attention_in must equal that of value");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxDtype(const gert::TilingContext* context)
{
    auto softmaxMax = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::SOFTMAX_MAX));
    auto softmaxSum = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::SOFTMAX_SUM));
    OP_CHECK_IF(softmaxMax == nullptr || softmaxSum == nullptr,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context->GetNodeName(), "softmax_max and softmax_sum", "nullptr",
                    "softmax_max and softmax_sum are required inputs in the current mode; provide both tensors"),
                return ge::GRAPH_FAILED);

    auto softmaxMaxType = static_cast<uint32_t>(softmaxMax->GetDataType());
    auto softmaxSumType = static_cast<uint32_t>(softmaxSum->GetDataType());

    bool softmaxTypeCheck = (softmaxMaxType == softmaxSumType) && (softmaxMaxType == ge::DT_FLOAT);
    OP_CHECK_IF(softmaxTypeCheck != true,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "softmax_max and softmax_sum",
                    std::to_string(softmaxMaxType) + " and " + std::to_string(softmaxSumType), "both must be DT_FLOAT"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInDtype(const gert::TilingContext* context)
{
    auto query = context->GetInputDesc(static_cast<size_t>(InputIndex::QUERY));
    OP_CHECK_IF(
        query == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "query", "input descriptor is required"),
        return ge::GRAPH_FAILED);
    auto attentionIn = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::ATTENTION_OUT));
    OP_CHECK_IF(
        attentionIn == nullptr,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "attention_in", "nullptr",
                                              "attention_in is a required input; provide a valid attention_out tensor"),
        return ge::GRAPH_FAILED);

    auto queryType = static_cast<uint32_t>(query->GetDataType());
    auto attentionInType = static_cast<uint32_t>(attentionIn->GetDataType());

    OP_CHECK_IF(
        queryType != attentionInType,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(context->GetNodeName(), "attention_in", std::to_string(attentionInType),
                                               "attention_in dtype must be the same as query dtype"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndShapeValid(gert::TilingContext* context, int64_t t1, int64_t n1, int64_t d, int64_t d2)
{
    if (context == nullptr) {
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"NsaSelectedAttentionGrad", "context is unavailable"}));
        return ge::GRAPH_FAILED;
    }

    auto isShapeInValid = (t1 == 0 || n1 == 0 || d == 0 || d2 == 0);
    OP_CHECK_IF(isShapeInValid,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "query and value",
                                                          "tnd(" + std::to_string(t1) + "," + std::to_string(n1) + "," +
                                                              std::to_string(d) + ") or tnd(" + std::to_string(t1) +
                                                              "," + std::to_string(n1) + "," + std::to_string(d2) + ")",
                                                          "all shape dims must be positive"),
                return ge::GRAPH_FAILED);

    auto ret = CheckTndSoftmaxMaxShape(context, t1, n1);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckTndSoftmaxSumShape(context, t1, n1);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionInShape(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckDtypeValid(gert::TilingContext* context)
{
    if (context == nullptr) {
        REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{"NsaSelectedAttentionGrad", "context is unavailable"}));
        return ge::GRAPH_FAILED;
    }

    auto ret = CheckSoftmaxDtype(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckAttentionInDtype(context);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    return ge::GRAPH_SUCCESS;
}

bool IsSameShape(const gert::StorageShape* aShape, const gert::StorageShape* bShape)
{
    OP_CHECK_IF((aShape == nullptr) || (bShape == nullptr), OP_LOGW("aShape or bShape is nullptr."), return false);
    uint32_t dimSizeA = aShape->GetStorageShape().GetDimNum();
    uint32_t dimSizeB = bShape->GetStorageShape().GetDimNum();
    if (dimSizeA != dimSizeB) {
        return false;
    }

    for (uint32_t i = 0; i < dimSizeA; i++) {
        auto dimA = aShape->GetStorageShape().GetDim(i);
        auto dimB = bShape->GetStorageShape().GetDim(i);
        if (dimA != dimB) {
            return false;
        }
    }
    return true;
}

} // namespace nsa
} // namespace optiling
