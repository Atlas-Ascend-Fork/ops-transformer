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
 * \file lightning_indexer_grad_infershape.cpp
 * \brief
 */
#include "base/err_msg.h"
#include <vector>
#include <string>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"

using namespace ge;

namespace ops {
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t DY_INDEX = 2;
constexpr uint32_t SPARSE_INDICES_INDEX = 3;
constexpr uint32_t WEIGHTS_INDEX = 4;

constexpr uint32_t ATTR_HEADNUM_INDEX = 0;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 1;
constexpr uint32_t ATTR_SPARSE_MODE_INDEX = 2;
constexpr uint32_t ATTR_PRETOKENS_INDEX = 3;
constexpr uint32_t ATTR_NEXTTOKENS_INDEX = 4;
constexpr uint32_t ATTR_DETERMINSTIC_INDEX = 5;

static ge::graphStatus InferShapeLightningIndexerGrad(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Enter LightningIndexerGrad runtime infershape impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* queryShape = context->GetInputShape(QUERY_INDEX);
    OP_CHECK_IF(queryShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    (context->GetNodeName() == nullptr ? "LightningIndexerGrad" : context->GetNodeName()), "query"),
                return ge::GRAPH_FAILED);
    const gert::Shape* keyShape = context->GetInputShape(KEY_INDEX);
    OP_CHECK_IF(keyShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    (context->GetNodeName() == nullptr ? "LightningIndexerGrad" : context->GetNodeName()), "key"),
                return ge::GRAPH_FAILED);
    const gert::Shape* weightShape = context->GetInputShape(WEIGHTS_INDEX);
    OP_CHECK_IF(weightShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    (context->GetNodeName() == nullptr ? "LightningIndexerGrad" : context->GetNodeName()), "weights"),
                return ge::GRAPH_FAILED);

    gert::Shape* dqShape = context->GetOutputShape(0);
    gert::Shape* dkShape = context->GetOutputShape(1);
    gert::Shape* dweightShape = context->GetOutputShape(2);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "LightningIndexerGrad" : context->GetNodeName()),
                        "InferShape framework object attrs is unavailable; check the operator registration and context "
                        "construction."})),
                return ge::GRAPH_FAILED);
    const char* inputLayouPtr = attrs->GetStr(ATTR_QUERY_LAYOUT_INDEX);
    OP_CHECK_IF(inputLayouPtr == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "LightningIndexerGrad" : context->GetNodeName()),
                        "InferShape: registered optional attribute layout is unavailable; check context construction "
                        "and default-attribute materialization."})),
                return ge::GRAPH_FAILED);
    const int64_t* sparse_mode = attrs->GetInt(ATTR_SPARSE_MODE_INDEX);
    OP_CHECK_IF(sparse_mode == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{
                        (context->GetNodeName() == nullptr ? "LightningIndexerGrad" : context->GetNodeName()),
                        "InferShape: registered optional attribute sparseMode is unavailable; check context "
                        "construction and default-attribute materialization."})),
                return ge::GRAPH_FAILED);
    std::string inputLayoutPtrStr = std::string(inputLayouPtr);

    if ((std::string(inputLayoutPtrStr) != "TND") && (std::string(inputLayoutPtrStr) != "BSND")) {
        OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
            (context->GetNodeName() == nullptr ? "LightningIndexerGrad" : context->GetNodeName()), "layout",
            inputLayoutPtrStr.c_str(), "Supported layouts are TND and BSND.");
        return ge::GRAPH_FAILED;
    }

    if (inputLayoutPtrStr == "BSND") {
        if (queryShape->GetDimNum() != 4) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                (context->GetNodeName() == nullptr ? "LightningIndexerGrad" : context->GetNodeName()), "query",
                std::to_string(queryShape->GetDimNum()).c_str(), "When layout is BSND, query must have rank 4.");
            return ge::GRAPH_FAILED;
        }
    } else {
        if (queryShape->GetDimNum() != 3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                (context->GetNodeName() == nullptr ? "LightningIndexerGrad" : context->GetNodeName()), "query",
                std::to_string(queryShape->GetDimNum()).c_str(), "When layout is TND, query must have rank 3.");
            return ge::GRAPH_FAILED;
        }
    }

    *dqShape = *queryShape;
    *dkShape = *keyShape;
    *dweightShape = *weightShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeLightningIndexerGrad(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    ge::DataType defaultType = context->GetInputDataType(0);

    // dq, outidx:0
    context->SetOutputDataType(0, defaultType);
    // dk, outidx:1
    context->SetOutputDataType(1, defaultType);
    // dweights, outidx:2
    context->SetOutputDataType(2, defaultType);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(LightningIndexerGrad)
    .InferShape(InferShapeLightningIndexerGrad)
    .InferDataType(InferDataTypeLightningIndexerGrad);
} // namespace ops
