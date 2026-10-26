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
 * \file sparse_flash_attention_grad_tiling_common.cpp
 * \brief
 */

#include "log/log.h"
#include "sparse_flash_attention_grad_tiling_common.h"

namespace optiling {
namespace sfag {

ge::graphStatus CheckTndSoftmaxMaxShape(gert::TilingContext* context, int64_t t1, int64_t n2, int64_t g)
{
    auto softmaxMaxShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_MAX));
    if (softmaxMaxShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxMaxShapeDim = softmaxMaxShape->GetStorageShape().GetDimNum();
    if (softmaxMaxShapeDim != 3) { // TND softmaxMax only support 3 dimensions
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "softmax_max",
                                               (std::string() + std::to_string(softmaxMaxShapeDim)).c_str(),
                                               (std::string() + "The shape of softmaxMax is invalid, got " +
                                                std::to_string(softmaxMaxShapeDim) + " dimensions")
                                                   .c_str());
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxMaxShape->GetStorageShape().GetDim(0); // 0:n2
    auto dim1 = softmaxMaxShape->GetStorageShape().GetDim(1); // 1:t1
    auto dim2 = softmaxMaxShape->GetStorageShape().GetDim(2); // 2:g

    // softmaxMax shape: [N2, T1, G]
    OP_CHECK_IF((dim0 != n2 || dim1 != t1 || dim2 != g),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    Ops::Base::GetOpInfo(context).c_str(), "softmax_max",
                    (std::string() + std::to_string(dim0) + ", " + std::to_string(dim1) + ", " + std::to_string(dim2) +
                     ", " + std::to_string(n2) + ", " + std::to_string(t1) + ", " + std::to_string(g))
                        .c_str(),
                    (std::string() + "The shape of softmaxMax is invalid, got (" + std::to_string(dim0) + "," +
                     std::to_string(dim1) + "," + std::to_string(dim2) + "), should be (" + std::to_string(n2) + "," +
                     std::to_string(t1) + "," + std::to_string(g) + ")")
                        .c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndSoftmaxSumShape(gert::TilingContext* context, int64_t t1, int64_t n2, int64_t g)
{
    auto softmaxSumShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SOFTMAX_SUM));
    if (softmaxSumShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto softmaxSumShapeDim = softmaxSumShape->GetStorageShape().GetDimNum();
    if (softmaxSumShapeDim != 3) { // TND softmaxSum only support 3 dimensions
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "softmax_sum",
                                               (std::string() + std::to_string(softmaxSumShapeDim)).c_str(),
                                               (std::string() + "The shape of softmaxSum is invalid, got " +
                                                std::to_string(softmaxSumShapeDim) + " dimensions")
                                                   .c_str());
        return ge::GRAPH_FAILED;
    }
    auto dim0 = softmaxSumShape->GetStorageShape().GetDim(0); // 0:n2
    auto dim1 = softmaxSumShape->GetStorageShape().GetDim(1); // 1:t1
    auto dim2 = softmaxSumShape->GetStorageShape().GetDim(2); // 2:g

    // softmaxMax shape: [N2, T1, G]
    OP_CHECK_IF((dim0 != n2 || dim1 != t1 || dim2 != g),
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    Ops::Base::GetOpInfo(context).c_str(), "softmax_sum",
                    (std::string() + std::to_string(dim0) + ", " + std::to_string(dim1) + ", " + std::to_string(dim2) +
                     ", " + std::to_string(n2) + ", " + std::to_string(t1) + ", " + std::to_string(g))
                        .c_str(),
                    (std::string() + "The shape of softmaxSum is invalid, got (" + std::to_string(dim0) + "," +
                     std::to_string(dim1) + "," + std::to_string(dim2) + "), should be (" + std::to_string(n2) + "," +
                     std::to_string(t1) + "," + std::to_string(g) + ")")
                        .c_str()),
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
    auto valueShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::VALUE));
    if (valueShape == nullptr) {
        valueShape = context->GetInputShape(static_cast<size_t>(InputIndex::KEY));
    }
    auto attentionInShapeDim = attentionInShape->GetStorageShape().GetDimNum();
    auto queryShapeDim = queryShape->GetStorageShape().GetDimNum();
    auto valueShapeDim = valueShape->GetStorageShape().GetDimNum();
    if (attentionInShapeDim != queryShapeDim) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            Ops::Base::GetOpInfo(context).c_str(), "query, out",
            (std::string() + std::to_string(attentionInShapeDim) + ", " + std::to_string(queryShapeDim)).c_str(),
            (std::string() + "The dimnum of attentionIn " + std::to_string(attentionInShapeDim) +
             " should be equal to query " + std::to_string(queryShapeDim))
                .c_str());
        return ge::GRAPH_FAILED;
    }
    if (attentionInShapeDim != valueShapeDim) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            Ops::Base::GetOpInfo(context).c_str(), "out",
            (std::string() + std::to_string(attentionInShapeDim) + ", " + std::to_string(valueShapeDim)).c_str(),
            (std::string() + "The dimnum of attentionIn " + std::to_string(attentionInShapeDim) +
             " should be equal to value " + std::to_string(valueShapeDim))
                .c_str());
        return ge::GRAPH_FAILED;
    }

    for (size_t i = 0; i < queryShapeDim - 1; i++) {
        if (attentionInShape->GetStorageShape().GetDim(i) != queryShape->GetStorageShape().GetDim(i)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                Ops::Base::GetOpInfo(context).c_str(), "out",
                (std::string() + std::to_string(i) + ", " +
                 std::to_string(attentionInShape->GetStorageShape().GetDim(i)) + ", " +
                 std::to_string(queryShape->GetStorageShape().GetDim(i)))
                    .c_str(),
                (std::string() + "The dim " + std::to_string(i) + " of attentionIn shape is invalid, got " +
                 std::to_string(attentionInShape->GetStorageShape().GetDim(i)) + ", should be " +
                 std::to_string(queryShape->GetStorageShape().GetDim(i)))
                    .c_str());
            return ge::GRAPH_FAILED;
        }
    }
    // check head_dim, attention D == value D
    if (attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1) !=
        valueShape->GetStorageShape().GetDim(queryShapeDim - 1)) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            Ops::Base::GetOpInfo(context).c_str(), "out",
            (std::string() + std::to_string(attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1)) + ", " +
             std::to_string(valueShape->GetStorageShape().GetDim(queryShapeDim - 1)))
                .c_str(),
            (std::string() + "The head_dim of attentionIn shape is invalid, got " +
             std::to_string(attentionInShape->GetStorageShape().GetDim(queryShapeDim - 1)) + ", should be " +
             std::to_string(valueShape->GetStorageShape().GetDim(queryShapeDim - 1)))
                .c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSoftmaxDtype(gert::TilingContext* context)
{
    auto softmaxMax = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::SOFTMAX_MAX));
    auto softmaxSum = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::SOFTMAX_SUM));
    OP_CHECK_IF(softmaxMax == nullptr || softmaxSum == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "softmax_max, softmax_sum",
                                                         "softmax_max or softmax_sum is nullptr."),
                return ge::GRAPH_FAILED);

    auto softmaxMaxType = static_cast<uint32_t>(softmaxMax->GetDataType());
    auto softmaxSumType = static_cast<uint32_t>(softmaxSum->GetDataType());

    bool softmaxTypeCheck = (softmaxMaxType == softmaxSumType) && (softmaxMaxType == ge::DT_FLOAT);
    OP_CHECK_IF(
        softmaxTypeCheck != true,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "softmax_max, softmax_sum",
                                                 "softmaxMaxType should be DT_FLOAT and same with softmaxSumType"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckAttentionInDtype(gert::TilingContext* context)
{
    auto query = context->GetInputDesc(static_cast<size_t>(InputIndex::QUERY));
    auto attentionIn = context->GetOptionalInputDesc(static_cast<size_t>(InputIndex::ATTENTION_OUT));
    OP_CHECK_IF(query == nullptr || attentionIn == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "query, out",
                                                         "query or attentionIn is nullptr."),
                return ge::GRAPH_FAILED);

    auto queryType = static_cast<uint32_t>(query->GetDataType());
    auto attentionInType = static_cast<uint32_t>(attentionIn->GetDataType());

    OP_CHECK_IF(queryType != attentionInType,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "query, out",
                                                         "invalid attentionIn dtype should be same with query's dtype"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckTndShapeValid(gert::TilingContext* context, int64_t t1, int64_t n1, int64_t d, int64_t d2,
                                   int64_t n2)
{
    if (context == nullptr) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(), "context is nullptr"}));
        return ge::GRAPH_FAILED;
    }

    auto isShapeInValid = (t1 == 0 || n1 == 0 || d == 0 || d2 == 0 || n2 == 0);
    OP_CHECK_IF(isShapeInValid,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    Ops::Base::GetOpInfo(context).c_str(), "query, key, sparse_indices, d_out, out",
                    (std::string() + std::to_string(t1) + ", " + std::to_string(n1) + ", " + std::to_string(d) + ", " +
                     std::to_string(t1) + ", " + std::to_string(n1) + ", " + std::to_string(d2))
                        .c_str(),
                    (std::string() + "input shape error, got 0 in tnd(" + std::to_string(t1) + "," +
                     std::to_string(n1) + "," + std::to_string(d) + ") or tnd(" + std::to_string(t1) + "," +
                     std::to_string(n1) + "," + std::to_string(d2) + ")")
                        .c_str()),
                return ge::GRAPH_FAILED);

    auto g = n1 / n2;
    auto ret = CheckTndSoftmaxMaxShape(context, t1, n2, g);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    ret = CheckTndSoftmaxSumShape(context, t1, n2, g);
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
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(), "context is nullptr"}));
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

} // namespace sfag
} // namespace optiling
