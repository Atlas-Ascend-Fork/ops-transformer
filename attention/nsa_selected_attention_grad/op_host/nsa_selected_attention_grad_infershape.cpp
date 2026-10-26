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
 * \file nsa_selected_attention_grad_proto.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;

namespace ops {
namespace nsa {

enum class InputIndex : uint32_t {
    QUERY = 0,
    KEY,
    VALUE,
    ATTENTION_OUT,
    ATTENTION_OUT_GRAD,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    TOPK_INDICES,
    ACTUAL_SEQ_Q_LEN,
    ACTUAL_SEQ_KV_LEN,
    ATTEN_MASK
};

enum class OutputIndex : uint32_t {
    DQ = 0,
    DK,
    DV
};

enum class AttrIndex : uint32_t {
    SCALE_VALUE = 0,
    SELECTED_BLOCK_COUNT,
    SELECTED_BLOCK_SIZE,
    HEAD_NUM, // N1
    INPUT_LAYOUT
};

ge::graphStatus InferShape4NsaSelectedAttentionGrad(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"NsaSelectedAttentionGrad", "InferShape context is unavailable"})),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferShape4NsaSelectedAttentionGrad.");

    const gert::Shape* queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    OP_CHECK_IF(queryShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "query",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* keyShape = context->GetInputShape(static_cast<size_t>(InputIndex::KEY));
    OP_CHECK_IF(keyShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "key",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* valueShape = context->GetInputShape(static_cast<size_t>(InputIndex::VALUE));

    OP_CHECK_IF(valueShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "value",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "attrs is unavailable during InferShape"})),
                return ge::GRAPH_FAILED);
    auto scaleValue = attrs->GetInt(static_cast<size_t>(AttrIndex::SCALE_VALUE));
    OP_CHECK_IF(scaleValue == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "scaleValue",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);
    auto selectedBlockCount = attrs->GetInt(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_COUNT));
    OP_CHECK_IF(selectedBlockCount == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "selectedBlockCount",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);
    auto selectedBlockSize = attrs->GetInt(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_SIZE));
    OP_CHECK_IF(selectedBlockSize == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "selectedBlockSize",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);
    auto headNum = attrs->GetInt(static_cast<size_t>(AttrIndex::HEAD_NUM));
    OP_CHECK_IF(headNum == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "headNum",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);
    const char* inputLayout = attrs->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::INPUT_LAYOUT));
    OP_CHECK_IF(inputLayout == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "inputLayout",
                                                         "required parameter metadata is missing"),
                return ge::GRAPH_FAILED);

    std::string inputLayoutStr = std::string(inputLayout);
    for (auto& c : inputLayoutStr) {
        c = toupper(c);
    }
    if (inputLayoutStr != "TND") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "input_layout", inputLayoutStr.c_str(),
                                              "input_layout must be TND (case-insensitive); set input_layout to TND");
        return GRAPH_FAILED;
    }

    gert::Shape* dqShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DQ));
    OP_CHECK_IF(dqShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "dqShape is unavailable during InferShape"})),
                return ge::GRAPH_FAILED);
    gert::Shape* dkShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DK));
    OP_CHECK_IF(dkShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "dkShape is unavailable during InferShape"})),
                return ge::GRAPH_FAILED);
    gert::Shape* dvShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DV));
    OP_CHECK_IF(dvShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{context->GetNodeName(), "dvShape is unavailable during InferShape"})),
                return ge::GRAPH_FAILED);
    *dqShape = *queryShape;
    *dkShape = *keyShape;
    *dvShape = *valueShape;

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4NsaSelectedAttentionGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"NsaSelectedAttentionGrad", "InferDataType context is unavailable"})),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferDataType4NsaSelectedAttentionGrad.");

    auto dtype = context->GetInputDataType(static_cast<size_t>(InputIndex::QUERY));
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DQ), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DK), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DV), dtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NsaSelectedAttentionGrad)
    .InferShape(InferShape4NsaSelectedAttentionGrad)
    .InferDataType(InferDataType4NsaSelectedAttentionGrad);
} // namespace nsa
} // namespace ops
