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
 * \file lightning_indexer_infershape.cpp
 * \brief
 */
#include "base/err_msg.h"
#include "log/log.h"
#include <vector>
#include <string>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ACTUAL_SEQ_K_INDEX = 4;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 0;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 2;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 6;

static ge::graphStatus InferShapeLightningIndexer(gert::InferShapeContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"LightningIndexer", "InferShape/InferDataType context is unavailable; check the "
                                                          "operator registration and context construction."})),
        return ge::GRAPH_FAILED);
    const gert::Shape* queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_IF(queryShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    (context->GetNodeName() == nullptr ? "LightningIndexer" : context->GetNodeName()), "query"),
                return ge::GRAPH_FAILED);
    const gert::Shape* keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_IF(keyShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    (context->GetNodeName() == nullptr ? "LightningIndexer" : context->GetNodeName()), "key"),
                return ge::GRAPH_FAILED);

    gert::Shape* sparseIndicesShape = context->GetOutputShape(0);
    OP_CHECK_IF(
        sparseIndicesShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{(context->GetNodeName() == nullptr ? "LightningIndexer" : context->GetNodeName()),
                                      "InferShape framework object sparseIndicesShape is unavailable; check the "
                                      "operator registration and context construction."})),
        return ge::GRAPH_FAILED);
    gert::Shape* sparseValuesShape = context->GetOutputShape(1);
    OP_CHECK_IF(
        sparseValuesShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{(context->GetNodeName() == nullptr ? "LightningIndexer" : context->GetNodeName()),
                                      "InferShape framework object sparseValuesShape is unavailable; check the "
                                      "operator registration and context construction."})),
        return ge::GRAPH_FAILED);

    auto liAttrs = context->GetAttrs();
    OP_CHECK_IF(
        liAttrs == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{(context->GetNodeName() == nullptr ? "LightningIndexer" : context->GetNodeName()),
                                      "InferShape framework object liAttrs is unavailable; check the operator "
                                      "registration and context construction."})),
        return ge::GRAPH_FAILED);
    const char* inputLayoutQueryPtr = liAttrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_IF(
        inputLayoutQueryPtr == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{(context->GetNodeName() == nullptr ? "LightningIndexer" : context->GetNodeName()),
                                      "InferShape: registered optional attribute layout_query is unavailable; check "
                                      "context construction and default-attribute materialization."})),
        return ge::GRAPH_FAILED);
    const char* inputLayoutKeyPtr = liAttrs->GetAttrPointer<char>(ATTR_KEY_LAYOUT_INDEX);
    OP_CHECK_IF(
        inputLayoutKeyPtr == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{(context->GetNodeName() == nullptr ? "LightningIndexer" : context->GetNodeName()),
                                      "InferShape: registered optional attribute layout_key is unavailable; check "
                                      "context construction and default-attribute materialization."})),
        return ge::GRAPH_FAILED);
    const int64_t* seleced_count = liAttrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    OP_CHECK_IF(
        seleced_count == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{(context->GetNodeName() == nullptr ? "LightningIndexer" : context->GetNodeName()),
                                      "InferShape: registered optional attribute sparse_count is unavailable; check "
                                      "context construction and default-attribute materialization."})),
        return ge::GRAPH_FAILED);
    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    OP_CHECK_IF(
        inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND",
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("LightningIndexer", "layout_query", inputLayoutQueryPtrStr.c_str(),
                                              "The layout_query should be TND or BSND"),
        return ge::GRAPH_FAILED);

    sparseIndicesShape->SetDimNum(queryShape->GetDimNum());
    if (inputLayoutQueryPtrStr == "BSND") {
        OP_CHECK_IF(
            queryShape->GetDimNum() != 4,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                (context->GetNodeName() == nullptr ? "LightningIndexer" : context->GetNodeName()), "query",
                std::to_string(queryShape->GetDimNum()).c_str(), "When layout_query is BSND, query must have rank 4."),
            return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
        sparseIndicesShape->SetDim(1, queryShape->GetDim(1)); // 1:Dim S
        sparseIndicesShape->SetDim(2, keyShape->GetDim(2));   // 2:Dim N
        sparseIndicesShape->SetDim(3, *seleced_count);        // 3:Dim K
    } else {
        OP_CHECK_IF(
            queryShape->GetDimNum() != 3,
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                (context->GetNodeName() == nullptr ? "LightningIndexer" : context->GetNodeName()), "query",
                std::to_string(queryShape->GetDimNum()).c_str(), "When layout_query is TND, query must have rank 3."),
            return ge::GRAPH_FAILED);
        sparseIndicesShape->SetDim(0, queryShape->GetDim(0));            // 0:Dim T
        int32_t nDimIndex = (inputLayoutKeyPtrStr == "PA_BSND") ? 2 : 1; // 2:Key Dim N
        sparseIndicesShape->SetDim(1, keyShape->GetDim(nDimIndex));      // 1:Dim N
        sparseIndicesShape->SetDim(2, *seleced_count);                   // 2:Dim K
    }

    const bool* return_value = liAttrs->GetAttrPointer<bool>(ATTR_RETURN_VALUE_INDEX);
    OP_CHECK_IF(
        return_value == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{(context->GetNodeName() == nullptr ? "LightningIndexer" : context->GetNodeName()),
                                      "InferShape: registered optional attribute return_values is unavailable; check "
                                      "context construction and default-attribute materialization."})),
        return ge::GRAPH_FAILED);
    bool liReturnValueFlag = (return_value != nullptr) ? *return_value : false;
    if (liReturnValueFlag) {
        *sparseValuesShape = *sparseIndicesShape;
    } else {
        sparseValuesShape->SetDimNum(1);
        sparseValuesShape->SetDim(0, 0);
    }
    OP_LOGI(context->GetNodeName(), "LightningIndexer InferShape end.");

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexer(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(
        context == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"LightningIndexer", "InferShape/InferDataType context is unavailable; check the "
                                                          "operator registration and context construction."})),
        return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "Enter LightningIndexer InferDataType impl.");
    // default set q's dtype as fia's output type
    ge::DataType outputType = ge::DT_INT32;
    // attention_out, outidx:0
    context->SetOutputDataType(0, outputType);
    context->SetOutputDataType(1, context->GetInputDataType(QUERY_INDEX));
    OP_LOGI(context->GetNodeName(), "LightningIndexer InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexer)
    .InferShape(InferShapeLightningIndexer)
    .InferDataType(InferDataTypeLightningIndexer);
} // namespace ops
