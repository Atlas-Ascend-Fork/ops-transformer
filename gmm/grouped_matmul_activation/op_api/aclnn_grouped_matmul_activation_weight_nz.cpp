/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_grouped_matmul_activation_weight_nz.h"

#include <cstring>

#include "../../grouped_matmul_swiglu_quant_v2/op_api/grouped_matmul_activation_prepare.h"
#include "../../grouped_matmul_swiglu_quant_v2/op_api/grouped_matmul_activation.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "log/log.h"
#include "opdev/format_utils.h"
#include "opdev/op_log.h"

using namespace op;

namespace {
constexpr char API_NAME[] = "aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize";
constexpr int64_t MX_DEQUANT_MODE = 2L;
constexpr int64_t SPLIT_SWIGLU_MODE = 2L;
constexpr int64_t FLOAT_DEQUANT_DTYPE = 0L;
constexpr char SWIGLU_ACT_TYPE[] = "swiglu";
constexpr size_t SINGLE_TENSOR_SIZE = 1UL;
constexpr size_t FIRST_TENSOR_INDEX = 0UL;
constexpr size_t MATRIX_DIM_NUM = 2UL;
constexpr size_t BATCHED_MATRIX_DIM_NUM = 3UL;

#define GMMAWN_CHECK(cond, code, logExpr) \
    do { \
        if (!(cond)) { \
            logExpr; \
            return code; \
        } \
    } while (0)

bool IsA4W4(const aclTensor* x, const aclTensorList* weight)
{
    return x != nullptr && weight != nullptr && weight->Size() == SINGLE_TENSOR_SIZE &&
           (*weight)[FIRST_TENSOR_INDEX] != nullptr && x->GetDataType() == DataType::DT_FLOAT4_E2M1 &&
           (*weight)[FIRST_TENSOR_INDEX]->GetDataType() == DataType::DT_FLOAT4_E2M1;
}

bool IsNzTensor(const aclTensor* tensor)
{
    return tensor != nullptr && ge::GetPrimaryFormat(tensor->GetStorageFormat()) == Format::FORMAT_FRACTAL_NZ;
}

aclnnStatus CheckParams(const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList,
                        const aclTensorList* weight, const aclTensorList* weightScale, const aclTensorList* bias,
                        int64_t dequantMode, int64_t dequantDtype, const char* actType, const aclTensor* y)
{
    GMMAWN_CHECK(x != nullptr && xScale != nullptr && groupList != nullptr && weight != nullptr &&
                     weightScale != nullptr && y != nullptr,
                 ACLNN_ERR_PARAM_NULLPTR,
                 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "required input", "does not support nullptr"));
    GMMAWN_CHECK(IsA4W4(x, weight), ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "x, weight", "unsupported",
                                                       "only FLOAT4_E2M1/FLOAT4_E2M1 is supported"));
    GMMAWN_CHECK(weightScale->Size() == SINGLE_TENSOR_SIZE && (*weightScale)[FIRST_TENSOR_INDEX] != nullptr,
                 ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_TENSORNUM(API_NAME, "weightScale", weightScale->Size(), "1"));
    GMMAWN_CHECK(xScale->GetDataType() == DataType::DT_FLOAT8_E8M0 &&
                     (*weightScale)[FIRST_TENSOR_INDEX]->GetDataType() == DataType::DT_FLOAT8_E8M0,
                 ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "xScale, weightScale", "unsupported",
                                                       "both must be FLOAT8_E8M0"));
    GMMAWN_CHECK(groupList->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "groupList", "unsupported", "must be INT64"));
    GMMAWN_CHECK(IsNzTensor((*weight)[FIRST_TENSOR_INDEX]), ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(API_NAME, "weight", "unsupported", "must be FRACTAL_NZ"));
    const auto& xShape = x->GetViewShape();
    const auto& weightShape = (*weight)[FIRST_TENSOR_INDEX]->GetViewShape();
    GMMAWN_CHECK(xShape.GetDimNum() == MATRIX_DIM_NUM && weightShape.GetDimNum() == BATCHED_MATRIX_DIM_NUM,
                 ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(API_NAME, "x, weight", "unsupported",
                                                          "x must be 2D and weight must be 3D"));
    GMMAWN_CHECK(y->GetDataType() == DataType::DT_BF16, ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(API_NAME, "y", "unsupported", "only BF16 is supported"));
    GMMAWN_CHECK(bias == nullptr || bias->Size() == 0, ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "bias",
                                                          "current A4W4 path only accepts an empty TensorList"));
    GMMAWN_CHECK(
        dequantMode == MX_DEQUANT_MODE && dequantDtype == FLOAT_DEQUANT_DTYPE, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "dequantMode/dequantDtype", "unsupported",
                                              "A4W4 only supports MX dequant mode 2 and float32 accumulation"));
    GMMAWN_CHECK(
        actType != nullptr && std::strcmp(actType, SWIGLU_ACT_TYPE) == 0, ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(API_NAME, "actType", "unsupported", "only actType=swiglu is supported"));
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
    const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList, const aclTensorList* weight,
    const aclTensorList* weightScale, const aclTensorList* bias, int64_t dequantMode, int64_t dequantDtype,
    const char* actType, int64_t groupListType, const aclIntArray* tuningConfig, int64_t swigluMode, double clampLimit,
    double gluAlpha, double gluBias, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    GMMAWN_CHECK(
        workspaceSize != nullptr && executor != nullptr, ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(API_NAME, "workspaceSize/executor", "does not support nullptr"));
    aclnnStatus status =
        CheckParams(x, xScale, groupList, weight, weightScale, bias, dequantMode, dequantDtype, actType, y);
    if (status != ACLNN_SUCCESS) {
        return status;
    }

    GMMAWN_CHECK(swigluMode == SPLIT_SWIGLU_MODE, ACLNN_ERR_PARAM_INVALID,
                 OP_LOGE(ACLNN_ERR_PARAM_INVALID, "only swigluMode=2 is supported"));
    L2_DFX_PHASE_1(aclnnGroupedMatmulActivationWeightNz,
                   DFX_IN(x, xScale, groupList, weight, weightScale, bias, dequantMode, dequantDtype, actType,
                          groupListType, tuningConfig, swigluMode, clampLimit, gluAlpha, gluBias),
                   DFX_OUT(y));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    gmm_activation::PreparedInputs inputs;
    status = gmm_activation::PrepareInputs(x, xScale, groupList, weight, weightScale, groupListType, tuningConfig,
                                           swigluMode, clampLimit, gluAlpha, gluBias, y, uniqueExecutor.get(), inputs);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    const aclTensor* result = l0op::GroupedMatmulActivation(
        inputs.x, inputs.xScale, inputs.groupList, inputs.weight, inputs.weightScale, bias, dequantMode, dequantDtype,
        actType, groupListType, tuningConfig, swigluMode, static_cast<float>(clampLimit), static_cast<float>(gluAlpha),
        static_cast<float>(gluBias), inputs.transposeWeight, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(result, y, uniqueExecutor.get()) != nullptr, ACLNN_ERR_INNER_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulActivationWeightNz(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                 aclrtStream stream)
{
    GMMAWN_CHECK(executor != nullptr && (workspaceSize == 0 || workspace != nullptr), ACLNN_ERR_PARAM_NULLPTR,
                 OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "executor/workspace must not be nullptr"));
    L2_DFX_PHASE_2(aclnnGroupedMatmulActivationWeightNz);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
} // extern "C"
