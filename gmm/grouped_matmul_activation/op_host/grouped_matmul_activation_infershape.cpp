/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include <string>
#include "../../grouped_matmul_swiglu_quant_v2/op_host/grouped_matmul_swiglu_quant_v2_host_utils.h"
namespace ops {
namespace {
constexpr size_t MATRIX_DIM_NUM = 2;
constexpr size_t WEIGHT_SCALE_N_DIM = 2;
constexpr int64_t FEATURE_ALIGNMENT = 128L;
constexpr int64_t SWIGLU_SPLIT_FACTOR = 2L;
} // namespace
static ge::graphStatus InferShape4GroupedMatmulActivation(gert::InferShapeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    const auto* xShape = context->GetInputShape(0);
    const auto* scaleShape = context->GetDynamicInputShape(GroupedMatmulSwigluQuantParamsV2::SCALE_INDEX, 0);
    const auto* attrs = context->GetAttrs();
    auto* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, scaleShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    OP_CHECK_IF(xShape->GetDimNum() != MATRIX_DIM_NUM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "x",
                                                         std::to_string(xShape->GetDimNum()), "must be 2D"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(scaleShape->GetDimNum() != GroupedMatmulSwigluQuantParamsV2::MX_WEIGHT_SCALE_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "weight_scale",
                                                         std::to_string(scaleShape->GetDimNum()), "must be 4D"),
                return ge::GRAPH_FAILED);
    const auto* trans = attrs->GetBool(GroupedMatmulActivationParams::ATTR_INDEX_TRANSPOSE_WEIGHT);
    const int64_t n = scaleShape->GetDim(trans != nullptr && *trans ? 1 : WEIGHT_SCALE_N_DIM);
    OP_CHECK_IF(n != -1 && (n <= 0 || n % FEATURE_ALIGNMENT != 0),
                OP_LOGE(context->GetNodeName(), "N must be positive and divisible by 128"), return ge::GRAPH_FAILED);
    yShape->SetDimNum(MATRIX_DIM_NUM);
    yShape->SetDim(0, xShape->GetDim(0));
    yShape->SetDim(1, n == -1 ? -1 : n / SWIGLU_SPLIT_FACTOR);
    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType4GroupedMatmulActivation(gert::InferDataTypeContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    return context->SetOutputDataType(0, ge::DT_BF16);
}
IMPL_OP_INFERSHAPE(GroupedMatmulActivation)
    .InferShape(InferShape4GroupedMatmulActivation)
    .InferDataType(InferDataType4GroupedMatmulActivation);
} // namespace ops
