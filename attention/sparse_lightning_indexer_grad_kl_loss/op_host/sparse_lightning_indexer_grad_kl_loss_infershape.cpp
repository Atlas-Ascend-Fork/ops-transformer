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
 * \file sparse_lightning_indexer_grad_kl_loss_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
static const uint64_t INDEX_LAYOUT = 1;
static const uint64_t DIM_NUM_0 = 0;
static const uint64_t SIZE_1 = 1;

enum InputIdx {
    queryEnum = 0,
    keyEnum,
    queryIndexEnum,
    keyIndexEnum,
    weightEnum,
    sparseIndexEnum,
    maxEnum,
    sumEnum,
    queryRopeEnum,
    keyRopeEnum,
    actualSeqLengthsQueryEnum,
    actualSeqLengthsKvEnum,
    sinkEnum
};

enum OutputIdx {
    dQueryIndexEnum = 0,
    dKeyIndexEnum,
    dWeightEnum,
    lossEnum
};

ge::graphStatus InferShapeSparseLightningIndexerGradKLLoss(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Start enter ShapeSparseLightningIndexerGradKLLoss runtime infershape impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* queryShape = context->GetInputShape(queryEnum);
    OP_CHECK_IF(queryShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "query",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* keyShape = context->GetInputShape(keyEnum);
    OP_CHECK_IF(keyShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "key",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* weightShape = context->GetInputShape(weightEnum);
    OP_CHECK_IF(weightShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "weight",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* queryIndexShape = context->GetInputShape(queryIndexEnum);
    OP_CHECK_IF(queryIndexShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "query_index",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* keyIndexShape = context->GetInputShape(keyIndexEnum);
    OP_CHECK_IF(keyIndexShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "key_index",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "attributes",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);

    const char* inputLayout = attrs->GetAttrPointer<char>(INDEX_LAYOUT);
    OP_CHECK_IF(inputLayout == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "input_layout",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);

    std::string inputLayoutSlig = std::string(inputLayout);
    for (auto& c : inputLayoutSlig) {
        c = toupper(c);
    }
    if (inputLayoutSlig != "BSND" && inputLayoutSlig != "TND") {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            Ops::Base::GetOpInfo(context).c_str(), "attributes",
            (std::string() + std::string(inputLayoutSlig.c_str())).c_str(),
            (std::string() + "The SparseLightningIndexerGradKLLoss inputLayout should be BSND/TND, but got " +
             std::string(inputLayoutSlig.c_str()) + ".")
                .c_str());
        return GRAPH_FAILED;
    }

    gert::Shape* lossShape = context->GetOutputShape(lossEnum);
    OP_CHECK_IF(lossShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "loss is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    gert::Shape* dWeightShape = context->GetOutputShape(dWeightEnum);
    OP_CHECK_IF(dWeightShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "d_weight is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    gert::Shape* dQIndexShape = context->GetOutputShape(dQueryIndexEnum);
    OP_CHECK_IF(
        dQIndexShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            "d_query_index is null; InferShape cannot continue"})),
        return ge::GRAPH_FAILED);
    gert::Shape* dKIndexShape = context->GetOutputShape(dKeyIndexEnum);
    OP_CHECK_IF(
        dKIndexShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            "d_key_index is null; InferShape cannot continue"})),
        return ge::GRAPH_FAILED);

    lossShape->SetDimNum(SIZE_1);         // 设置维度为1
    lossShape->SetDim(DIM_NUM_0, SIZE_1); // 设置第0维数值为1
    *dWeightShape = *weightShape;
    *dQIndexShape = *queryIndexShape;
    *dKIndexShape = *keyIndexShape;

    if (lossShape->GetShapeSize() != 1) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(),
                                               "query, key, query_index, key_index, weight",
                                               (std::string() + std::to_string(lossShape->GetShapeSize())).c_str(),
                                               (std::string() + "The Shape Len of Loss should be 1, but got " +
                                                std::to_string(lossShape->GetShapeSize()) + ".")
                                                   .c_str());
        return GRAPH_FAILED;
    } else if (lossShape->GetDim(DIM_NUM_0) != 1) {
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(),
                                               "query, key, query_index, key_index, weight",
                                               (std::string() + std::to_string(lossShape->GetDim(DIM_NUM_0))).c_str(),
                                               (std::string() + "The Shape data of Loss should be 1, but got " +
                                                std::to_string(lossShape->GetDim(DIM_NUM_0)) + ".")
                                                   .c_str());
        return GRAPH_FAILED;
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseLightningIndexerGradKLLoss(gert::InferDataTypeContext* context)
{
    OP_LOGI(context, "Start enter ShapeSparseLightningIndexerGradKLLoss runtime infershape impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto inputDataType = context->GetInputDataType(queryEnum);
    const auto inputWeightDataType = context->GetInputDataType(weightEnum);
    context->SetOutputDataType(dQueryIndexEnum, inputDataType);
    context->SetOutputDataType(dKeyIndexEnum, inputDataType);
    context->SetOutputDataType(dWeightEnum, inputWeightDataType);
    context->SetOutputDataType(lossEnum, DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseLightningIndexerGradKLLoss)
    .InferShape(InferShapeSparseLightningIndexerGradKLLoss)
    .InferDataType(InferDataTypeSparseLightningIndexerGradKLLoss);
} // namespace ops
