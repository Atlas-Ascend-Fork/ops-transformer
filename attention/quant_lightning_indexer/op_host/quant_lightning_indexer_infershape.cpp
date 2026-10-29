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
 * \file quant_lightning_indexer_proto.cpp
 * \brief
 */
#include "base/err_msg.h"
#include <vector>
#include <string>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

#include "err/ops_err.h"
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KV_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_SPARSE_COUNT_INDEX = 4;
constexpr uint32_t DIM_NUM_3 = 3;
constexpr uint32_t DIM_NUM_4 = 4;

static ge::graphStatus InferShapeQuantLightningIndexer(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"QuantLightningIndexer",
                                      "InferShape/InferDataType context is unavailable; check the "
                                      "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_IF(queryShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    (context->GetNodeName() == nullptr ? "QuantLightningIndexer" : context->GetNodeName()), "query"),
                return ge::GRAPH_FAILED);
    const gert::Shape* keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_IF(keyShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    (context->GetNodeName() == nullptr ? "QuantLightningIndexer" : context->GetNodeName()), "key"),
                return ge::GRAPH_FAILED);
    gert::Shape* outShape = context->GetOutputShape(0);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "QuantLightningIndexer" : context->GetNodeName()),
                        "InferShape framework object attrs is unavailable; check the operator registration and context "
                        "construction."})),
                return ge::GRAPH_FAILED);
    const char* inputLayoutQueryPtr = attrs->GetAttrPointer<char>(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_IF(inputLayoutQueryPtr == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "QuantLightningIndexer" : context->GetNodeName()),
                        "InferShape: registered optional attribute layout_query is unavailable; check context "
                        "construction and default-attribute materialization."})),
                return ge::GRAPH_FAILED);
    const char* inputLayoutKeyPtr = attrs->GetAttrPointer<char>(ATTR_KV_LAYOUT_INDEX);
    OP_CHECK_IF(inputLayoutKeyPtr == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "QuantLightningIndexer" : context->GetNodeName()),
                        "InferShape: registered optional attribute layout_key is unavailable; check context "
                        "construction and default-attribute materialization."})),
                return ge::GRAPH_FAILED);
    const int64_t* qliSparseCount = attrs->GetInt(ATTR_SPARSE_COUNT_INDEX);
    OP_CHECK_IF(qliSparseCount == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "QuantLightningIndexer" : context->GetNodeName()),
                        "InferShape: registered optional attribute sparse_count is unavailable; check context "
                        "construction and default-attribute materialization."})),
                return ge::GRAPH_FAILED);

    std::string inputLayoutQueryPtrStr = std::string(inputLayoutQueryPtr);
    std::string inputLayoutKeyPtrStr = std::string(inputLayoutKeyPtr);
    if (inputLayoutQueryPtrStr != "TND" && inputLayoutQueryPtrStr != "BSND") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("QuantLightningIndexer", "layout_query", inputLayoutQueryPtrStr.c_str(),
                                              "Layout_query should be TND or BSND");
        return GRAPH_FAILED;
    }

    int64_t keyHeadNum = (inputLayoutKeyPtrStr == "TND") ? keyShape->GetDim(1) : keyShape->GetDim(2);
    if (inputLayoutQueryPtrStr == "BSND") {
        outShape->SetDimNum(DIM_NUM_4);
        outShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim B
        outShape->SetDim(1, queryShape->GetDim(1)); // 1:Dim S
        outShape->SetDim(2, keyHeadNum);            // 2:Dim N
        outShape->SetDim(3, *qliSparseCount);       // 3:Dim K
    } else {
        outShape->SetDimNum(DIM_NUM_3);
        outShape->SetDim(0, queryShape->GetDim(0)); // 0:Dim T
        outShape->SetDim(1, keyHeadNum);            // 1:output shape's N Dim, 2: key shape's N Dim
        outShape->SetDim(2, *qliSparseCount);       // 2:Dim K
    }

    OP_LOGD(context->GetNodeName(), "QuantLightningIndexer InferShape end.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeQuantLightningIndexer(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"QuantLightningIndexer",
                                      "InferShape/InferDataType context is unavailable; check the "
                                      "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context->GetNodeName(), "Enter QuantLightningIndexer InferDataType impl.");
    // default index data type is int32
    ge::DataType outputType = ge::DT_INT32;
    context->SetOutputDataType(0, outputType);
    OP_LOGD(context->GetNodeName(), "QuantLightningIndexer InferDataType end.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(QuantLightningIndexer)
    .InferShape(InferShapeQuantLightningIndexer)
    .InferDataType(InferDataTypeQuantLightningIndexer);
} // namespace ops
