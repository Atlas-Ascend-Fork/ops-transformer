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
 * \file sparse_flash_attention_grad_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "err/ops_err.h"

using namespace ge;

namespace ops {
namespace sfag {

enum class InputIndex : uint32_t {
    QUERY = 0,
    KEY,
    TOPK_INDICES,
    ATTENTION_OUT_GRAD,
    ATTENTION_OUT,
    SOFTMAX_MAX,
    SOFTMAX_SUM,
    SINKS, // oss-sink 输入 [N1] FP32，OPTIONAL
    VALUE,
    ACTUAL_SEQ_Q_LEN,
    ACTUAL_SEQ_KV_LEN,
    Q_ROPE,
    K_ROPE
};

enum class OutputIndex : uint32_t {
    DQ = 0,
    DK,
    DV,
    DQ_ROPE,
    DK_ROPE,
    D_SINKS // sink 梯度 [N1] FP32，OPTIONAL
};

enum class AttrIndex : uint32_t {
    SCALE_VALUE = 0,
    SELECTED_BLOCK_SIZE,
    INPUT_LAYOUT
};

ge::graphStatus InferShape4SparseFlashAttentionGrad(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"SparseFlashAttentionGrad", "InferShapeContext is nullptr"})),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferShape4SparseFlashAttentionGrad.");

    const gert::Shape* queryShape = context->GetInputShape(static_cast<size_t>(InputIndex::QUERY));
    const gert::Shape* keyShape = context->GetInputShape(static_cast<size_t>(InputIndex::KEY));
    const gert::Shape* valueShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::VALUE));
    const gert::Shape* sinksShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::SINKS));
    const gert::Shape* queryRopeShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::Q_ROPE));
    const gert::Shape* keyRopeShape = context->GetOptionalInputShape(static_cast<size_t>(InputIndex::K_ROPE));
    OP_CHECK_IF(queryShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "query",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(keyShape == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "key",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "attributes",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    auto scaleValue = attrs->GetInt(static_cast<size_t>(AttrIndex::SCALE_VALUE));
    OP_CHECK_IF(scaleValue == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "scale_value",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    auto selectedBlockSize = attrs->GetInt(static_cast<size_t>(AttrIndex::SELECTED_BLOCK_SIZE));
    OP_CHECK_IF(selectedBlockSize == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "selected_block_size",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    const char* inputLayout = attrs->GetAttrPointer<char>(static_cast<size_t>(AttrIndex::INPUT_LAYOUT));
    OP_CHECK_IF(inputLayout == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context).c_str(), "input_layout",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);

    std::string inputLayoutSfag = std::string(inputLayout);
    for (auto& c : inputLayoutSfag) {
        c = toupper(c);
    }
    if (inputLayoutSfag != "BSND" && inputLayoutSfag != "TND") {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            Ops::Base::GetOpInfo(context).c_str(), "attributes",
            (std::string() + std::string(inputLayoutSfag.c_str())).c_str(),
            (std::string() + "The SparseFlashAttentionGrad inputLayout should be BSND/TND, but got " +
             std::string(inputLayoutSfag.c_str()) + ".")
                .c_str());
        return GRAPH_FAILED;
    }

    gert::Shape* dqShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DQ));
    gert::Shape* dkShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DK));
    gert::Shape* dvShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DV));
    OP_CHECK_IF(dqShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "dq is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(dkShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "dk is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    *dqShape = *queryShape;
    *dkShape = *keyShape;
    if (valueShape != nullptr && dvShape != nullptr) {
        *dvShape = *valueShape;
    }

    if (queryRopeShape != nullptr) {
        gert::Shape* dqRopeShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DQ_ROPE));
        OP_CHECK_IF(
            dqRopeShape == nullptr,
            REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                      (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                "d_query_rope is null; InferShape cannot continue"})),
            return ge::GRAPH_FAILED);
        *dqRopeShape = *queryRopeShape;
    }
    if (keyRopeShape != nullptr) {
        gert::Shape* dkRopeShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::DK_ROPE));
        OP_CHECK_IF(
            dkRopeShape == nullptr,
            REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                      (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                "d_key_rope is null; InferShape cannot continue"})),
            return ge::GRAPH_FAILED);
        *dkRopeShape = *keyRopeShape;
    }

    // d_sinks shape 随 sinks：sinks 存在时 = sinks shape ([N1])，否则强制返回空 (shape [0])
    gert::Shape* dSinksShape = context->GetOutputShape(static_cast<size_t>(OutputIndex::D_SINKS));
    OP_CHECK_IF(dSinksShape == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context).c_str(),
                                                                    "d_sinks is null; InferShape cannot continue"})),
                return ge::GRAPH_FAILED);
    if (sinksShape != nullptr && sinksShape->GetDimNum() > 0) {
        *dSinksShape = *sinksShape;
    } else {
        dSinksShape->SetDimNum(1);
        dSinksShape->SetDim(0, 0);
    }

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4SparseFlashAttentionGrad(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1014", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{"SparseFlashAttentionGrad", "InferDataTypeContext is nullptr"})),
                return ge::GRAPH_FAILED);
    OP_LOGD(context, "Enter InferDataType4SparseFlashAttentionGrad.");

    auto dtype = context->GetInputDataType(static_cast<size_t>(InputIndex::QUERY));
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DQ), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DK), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DV), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DQ_ROPE), dtype);
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::DK_ROPE), dtype);
    // d_sinks 全程 FP32（与 sinks 及数学链一致，不随 query dtype）
    context->SetOutputDataType(static_cast<size_t>(OutputIndex::D_SINKS), ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SparseFlashAttentionGrad)
    .InferShape(InferShape4SparseFlashAttentionGrad)
    .InferDataType(InferDataType4SparseFlashAttentionGrad);
} // namespace sfag
} // namespace ops
