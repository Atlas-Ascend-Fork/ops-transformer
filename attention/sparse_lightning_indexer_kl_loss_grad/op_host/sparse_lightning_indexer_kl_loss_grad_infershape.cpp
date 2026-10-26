/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_lightning_indexer_kl_loss_grad_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
enum InputIdx {
    qEnum = 0,
    kEnum,
    wEnum,
    sparseIndexEnum,
    attnSoftmaxL1NormEnum
};

enum OutputIdx {
    dqEnum = 0,
    dkEnum,
    dwEnum,
    softmaxOutEnum
};

ge::graphStatus InferShapeSparseLightningIndexerKLLossGrad(gert::InferShapeContext* context)
{
    OP_LOGI(context, "Start enter SparseLightningIndexerKLLossGrad infershape impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape* qShape = context->GetInputShape(qEnum);
    OP_CHECK_IF(qShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "q",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* kShape = context->GetInputShape(kEnum);
    OP_CHECK_IF(kShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "k",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* wShape = context->GetInputShape(wEnum);
    OP_CHECK_IF(wShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "w",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    const gert::Shape* softmaxShape = context->GetInputShape(attnSoftmaxL1NormEnum);
    OP_CHECK_IF(softmaxShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "attn_softmax_l1_norm",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);

    gert::Shape* dqShape = context->GetOutputShape(dqEnum);
    OP_CHECK_IF(dqShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "dq is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    gert::Shape* dkShape = context->GetOutputShape(dkEnum);
    OP_CHECK_IF(dkShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "dk is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    gert::Shape* dwShape = context->GetOutputShape(dwEnum);
    OP_CHECK_IF(dwShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "dw is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    gert::Shape* softmaxOutShape = context->GetOutputShape(softmaxOutEnum);
    OP_CHECK_IF(
        softmaxOutShape == nullptr,
        REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                  (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                            "softmax_out is null; InferShape cannot continue"})),
        return ge::GRAPH_FAILED);

    *dqShape = *qShape;
    *dkShape = *kShape;
    *dwShape = *wShape;
    *softmaxOutShape = *softmaxShape;
    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeSparseLightningIndexerKLLossGrad(gert::InferDataTypeContext* context)
{
    OP_LOGI(context, "Start enter SparseLightningIndexerKLLossGrad infer dtype impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    context->SetOutputDataType(dqEnum, context->GetInputDataType(qEnum));
    context->SetOutputDataType(dkEnum, context->GetInputDataType(kEnum));
    context->SetOutputDataType(dwEnum, ge::DT_FLOAT);
    context->SetOutputDataType(softmaxOutEnum, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseLightningIndexerKLLossGrad)
    .InferShape(InferShapeSparseLightningIndexerKLLossGrad)
    .InferDataType(InferDataTypeSparseLightningIndexerKLLossGrad);
} // namespace ops
