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
 * \file dense_flash_attention_infershape.cpp
 * \brief
 */

#include "base/err_msg.h"
#include "log/log.h"
#include <vector>
#include <graph/utils/type_utils.h>
#include "register/op_impl_registry.h"
#include "err/ops_err.h"

using namespace ge;

namespace ops {
enum InputIdx {
    queryEnum = 0,
    keyEnum,
    queryIndexEnum,
    keyIndexEnum,
    weightEnum,
    maxEnum,
    sumEnum,
    maxIndexEnum,
    sumIndexEnum,
    queryRopeEnum,
    keyRopeEnum,
    actualSeqLengthsQueryEnum,
    actualSeqLengthsKvEnum
};

enum OutputIdx {
    dQueryIndexEnum = 0,
    dKeyIndexEnum,
    dWeightEnum,
    lossEnum
};

ge::graphStatus InferShapeDenseLightningIndexerGradKLLoss(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"DenseLightningIndexerGradKLLoss",
                                      "InferShape/InferDataType context is unavailable; check the "
                                      "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeDenseLightningIndexerGradKLLoss");

    // get input shapes
    const gert::Shape* queryShape = context->GetInputShape(queryEnum);
    OP_CHECK_IF(
        queryShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            (context->GetNodeName() == nullptr ? "DenseLightningIndexerGradKLLoss" : context->GetNodeName()), "query"),
        return ge::GRAPH_FAILED);
    const gert::Shape* keyShape = context->GetInputShape(keyEnum);
    OP_CHECK_IF(
        keyShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            (context->GetNodeName() == nullptr ? "DenseLightningIndexerGradKLLoss" : context->GetNodeName()), "key"),
        return ge::GRAPH_FAILED);
    const gert::Shape* weightShape = context->GetInputShape(weightEnum);
    OP_CHECK_IF(
        weightShape == nullptr,
        OP_LOGE_WITH_INVALID_INPUT(
            (context->GetNodeName() == nullptr ? "DenseLightningIndexerGradKLLoss" : context->GetNodeName()), "weight"),
        return ge::GRAPH_FAILED);
    const gert::Shape* queryIndexShape = context->GetInputShape(queryIndexEnum);
    OP_CHECK_IF(queryIndexShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    (context->GetNodeName() == nullptr ? "DenseLightningIndexerGradKLLoss" : context->GetNodeName()),
                    "query_index"),
                return ge::GRAPH_FAILED);
    const gert::Shape* keyIndexShape = context->GetInputShape(keyIndexEnum);
    OP_CHECK_IF(keyIndexShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(
                    (context->GetNodeName() == nullptr ? "DenseLightningIndexerGradKLLoss" : context->GetNodeName()),
                    "key_index"),
                return ge::GRAPH_FAILED);

    // get output shapes
    gert::Shape* dQIndexShape = context->GetOutputShape(dQueryIndexEnum);
    OP_CHECK_IF(dQIndexShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{(context->GetNodeName() == nullptr ? "DenseLightningIndexerGradKLLoss" :
                                                                                   context->GetNodeName()),
                                              "InferShape framework object dQIndexShape is unavailable; check the "
                                              "operator registration and context construction."})),
                return ge::GRAPH_FAILED);
    gert::Shape* dKIndexShape = context->GetOutputShape(dKeyIndexEnum);
    OP_CHECK_IF(dKIndexShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{(context->GetNodeName() == nullptr ? "DenseLightningIndexerGradKLLoss" :
                                                                                   context->GetNodeName()),
                                              "InferShape framework object dKIndexShape is unavailable; check the "
                                              "operator registration and context construction."})),
                return ge::GRAPH_FAILED);
    gert::Shape* dWeightShape = context->GetOutputShape(dWeightEnum);
    OP_CHECK_IF(dWeightShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{(context->GetNodeName() == nullptr ? "DenseLightningIndexerGradKLLoss" :
                                                                                   context->GetNodeName()),
                                              "InferShape framework object dWeightShape is unavailable; check the "
                                              "operator registration and context construction."})),
                return ge::GRAPH_FAILED);
    gert::Shape* lossShape = context->GetOutputShape(lossEnum);
    OP_CHECK_IF(lossShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{(context->GetNodeName() == nullptr ? "DenseLightningIndexerGradKLLoss" :
                                                                                   context->GetNodeName()),
                                              "InferShape framework object lossShape is unavailable; check the "
                                              "operator registration and context construction."})),
                return ge::GRAPH_FAILED);

    // 填充输出shape大小
    *dWeightShape = *weightShape;
    *dQIndexShape = *queryIndexShape;
    *dKIndexShape = *keyIndexShape;
    lossShape->SetDimNum(1); // 设置维度为1
    lossShape->SetDim(0, 1); // 设置第0维数值为1

    OP_LOGD(context->GetNodeName(), "End to do InferShapeDenseLightningIndexerGradKLLoss");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeDenseLightningIndexerGradKLLoss(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1014", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"DenseLightningIndexerGradKLLoss",
                                      "InferShape/InferDataType context is unavailable; check the "
                                      "operator registration and context construction."}));
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeDenseLightningIndexerGradKLLoss");
    const auto inputDataType = context->GetInputDataType(queryEnum);
    const auto inputWeightDataType = context->GetInputDataType(weightEnum);
    context->SetOutputDataType(dQueryIndexEnum, inputDataType);
    context->SetOutputDataType(dKeyIndexEnum, inputDataType);
    context->SetOutputDataType(dWeightEnum, inputWeightDataType);
    context->SetOutputDataType(lossEnum, DT_FLOAT);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeDenseLightningIndexerGradKLLoss");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(DenseLightningIndexerGradKLLoss)
    .InferShape(InferShapeDenseLightningIndexerGradKLLoss)
    .InferDataType(InferDataTypeDenseLightningIndexerGradKLLoss);
} // namespace ops
