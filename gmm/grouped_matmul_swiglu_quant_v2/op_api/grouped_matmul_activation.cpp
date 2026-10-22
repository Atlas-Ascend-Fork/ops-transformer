/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "grouped_matmul_activation.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "log/log.h"

namespace l0op {
OP_TYPE_REGISTER(GroupedMatmulActivation);
const aclTensor* GroupedMatmulActivation(const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList,
                                         const aclTensorList* weight, const aclTensorList* weightScale,
                                         const aclTensorList* bias, int64_t dequantMode, int64_t dequantDtype,
                                         const char* actType, int64_t groupListType, const aclIntArray* tuningConfig,
                                         int64_t swigluMode, float clampLimit, float gluAlpha, float gluBias,
                                         bool transposeWeight, aclOpExecutor* executor)
{
    L0_DFX(GroupedMatmulActivation, x, xScale, groupList, weight, weightScale, bias, dequantMode, dequantDtype, actType,
           groupListType, tuningConfig, swigluMode, clampLimit, gluAlpha, gluBias, transposeWeight);
    if (executor == nullptr || x == nullptr || weightScale == nullptr || weightScale->Size() != 1 ||
        (*weightScale)[0] == nullptr || (*weightScale)[0]->GetViewShape().GetDimNum() != 4) {
        return nullptr;
    }
    const auto& scaleShape = (*weightScale)[0]->GetViewShape();
    int64_t n = scaleShape.GetDim(transposeWeight ? 1 : 2);
    auto out =
        executor->AllocTensor(op::Shape({x->GetViewShape().GetDim(0), n / 2}), op::DataType::DT_BF16, ge::FORMAT_ND);
    if (out == nullptr) {
        return nullptr;
    }
    const aclTensorList* emptyBias = bias;
    if (emptyBias == nullptr || emptyBias->Size() == 0) {
        // Dynamic inputs need a typed descriptor for binary selection. A zero-element
        // tensor represents the empty TensorList, as in GroupedMatmulActivationQuant.
        aclTensor* emptyTensor = executor->AllocTensor({0}, op::DataType::DT_BF16);
        if (emptyTensor == nullptr) {
            return nullptr;
        }
        emptyBias = executor->AllocTensorList(&emptyTensor, 1);
        if (emptyBias == nullptr) {
            return nullptr;
        }
    }
    auto ret = INFER_SHAPE(GroupedMatmulActivation, OP_INPUT(x, xScale, groupList, weight, weightScale, emptyBias),
                           OP_OUTPUT(out),
                           OP_ATTR(dequantMode, dequantDtype, actType, groupListType, tuningConfig, swigluMode,
                                   clampLimit, gluAlpha, gluBias, transposeWeight));
    if (ret != ACLNN_SUCCESS) {
        return nullptr;
    }
    ret = ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmulActivation,
                                      OP_INPUT(x, xScale, groupList, weight, weightScale, emptyBias), OP_OUTPUT(out),
                                      OP_ATTR(dequantMode, dequantDtype, actType, groupListType, tuningConfig,
                                              swigluMode, clampLimit, gluAlpha, gluBias, transposeWeight));
    return ret == ACLNN_SUCCESS ? out : nullptr;
}
} // namespace l0op
