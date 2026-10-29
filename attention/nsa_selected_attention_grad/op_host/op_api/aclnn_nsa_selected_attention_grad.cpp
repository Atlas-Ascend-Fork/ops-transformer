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
 * \file aclnn_nsa_selected_attention_grad.cpp
 * \brief
 */

#include "aclnn_nsa_selected_attention_grad.h"
#include "nsa_selected_attention_grad.h"
#include <cctype>
#include <string>
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"

using namespace op;

namespace {
constexpr size_t DIM_NUM_2 = 2;
constexpr size_t DIM_NUM_3 = 3;
constexpr int64_t SOFTMAX_TAIL_DIM = 8;
constexpr int64_t HEAD_DIM_128 = 128;
constexpr int64_t HEAD_DIM_192 = 192;
constexpr int64_t SELECTED_BLOCK_SIZE_MIN = 16;
constexpr int64_t SELECTED_BLOCK_SIZE_MAX = 128;
constexpr int64_t SELECTED_BLOCK_SIZE_ALIGN = 16;
constexpr int64_t SELECTED_BLOCK_COUNT_MIN = 1;
constexpr int64_t SELECTED_BLOCK_COUNT_MAX = 128;

std::string GetUpperInputLayout(const char* inputLayout)
{
    std::string inputLayoutStr = std::string(inputLayout);
    for (auto& c : inputLayoutStr) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return inputLayoutStr;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

struct NsaSelectedAttentionGradParams {
    const aclTensor* query = nullptr;
    const aclTensor* key = nullptr;
    const aclTensor* value = nullptr;
    const aclTensor* attentionOut = nullptr;
    const aclTensor* attentionOutGrad = nullptr;
    const aclTensor* softmaxMax = nullptr;
    const aclTensor* softmaxSum = nullptr;
    const aclTensor* topkIndices = nullptr;
    const aclIntArray* actualSeqQLenOptional = nullptr;
    const aclIntArray* actualSeqKvLenOptional = nullptr;
    const aclTensor* attenMaskOptional = nullptr;
    double scaleValue;
    int64_t selectedBlockSize;
    int64_t selectedBlockCount;
    int64_t headNum;
    char* inputLayout;
    int64_t sparseMode;
    const aclTensor* dqOut = nullptr;
    const aclTensor* dkOut = nullptr;
    const aclTensor* dvOut = nullptr;
};

static aclnnStatus CheckDim3(const aclTensor* tensor, const char* name)
{
    const size_t dimNum = tensor->GetViewShape().GetDimNum();
    CHECK_COND(dimNum == DIM_NUM_3, ACLNN_ERR_PARAM_INVALID, "%s must be 3-dimensional in TND layout, but got [%zu].",
               name, dimNum);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShapeAndAttr(const NsaSelectedAttentionGradParams& params)
{
    CHECK_COND(params.inputLayout != nullptr, ACLNN_ERR_PARAM_NULLPTR, "inputLayout must not be nullptr.");
    const std::string inputLayoutStr = GetUpperInputLayout(params.inputLayout);
    CHECK_COND(inputLayoutStr == "TND", ACLNN_ERR_PARAM_INVALID, "inputLayout only supports TND, but got [%s].",
               inputLayoutStr.c_str());
    CHECK_COND(params.sparseMode == 0 || params.sparseMode == 2, ACLNN_ERR_PARAM_INVALID,
               "sparseMode only supports 0 or 2, but got [%ld].", params.sparseMode);
    CHECK_COND(
        params.selectedBlockSize >= SELECTED_BLOCK_SIZE_MIN && params.selectedBlockSize <= SELECTED_BLOCK_SIZE_MAX &&
            params.selectedBlockSize % SELECTED_BLOCK_SIZE_ALIGN == 0,
        ACLNN_ERR_PARAM_INVALID, "selectedBlockSize must be in range [16, 128] and a multiple of 16, but got [%ld].",
        params.selectedBlockSize);
    CHECK_COND(
        params.selectedBlockCount >= SELECTED_BLOCK_COUNT_MIN && params.selectedBlockCount <= SELECTED_BLOCK_COUNT_MAX,
        ACLNN_ERR_PARAM_INVALID, "selectedBlockCount must be in range [1, 128], but got [%ld].",
        params.selectedBlockCount);

    // TND layout: query/dq are [T1, N1, D], key/value/dk are [T2, N2, D], topkIndices is [T1, N2, selectedBlockCount]
    CHECK_RET_CODE(CheckDim3(params.query, "query"), "query dim num check failed.");
    CHECK_RET_CODE(CheckDim3(params.key, "key"), "key dim num check failed.");
    CHECK_RET_CODE(CheckDim3(params.value, "value"), "value dim num check failed.");
    CHECK_RET_CODE(CheckDim3(params.attentionOut, "attentionOut"), "attentionOut dim num check failed.");
    CHECK_RET_CODE(CheckDim3(params.attentionOutGrad, "attentionOutGrad"), "attentionOutGrad dim num check failed.");
    CHECK_RET_CODE(CheckDim3(params.softmaxMax, "softmaxMax"), "softmaxMax dim num check failed.");
    CHECK_RET_CODE(CheckDim3(params.softmaxSum, "softmaxSum"), "softmaxSum dim num check failed.");
    CHECK_RET_CODE(CheckDim3(params.topkIndices, "topkIndices"), "topkIndices dim num check failed.");
    if (params.attenMaskOptional != nullptr) {
        const size_t maskDimNum = params.attenMaskOptional->GetViewShape().GetDimNum();
        CHECK_COND(maskDimNum == DIM_NUM_2, ACLNN_ERR_PARAM_INVALID,
                   "attenMaskOptional must be 2-dimensional, but got [%zu].", maskDimNum);
    }

    const auto& queryShape = params.query->GetViewShape();
    const auto& keyShape = params.key->GetViewShape();
    const auto& valueShape = params.value->GetViewShape();
    const auto& attentionOutShape = params.attentionOut->GetViewShape();
    const auto& attentionOutGradShape = params.attentionOutGrad->GetViewShape();
    const auto& softmaxMaxShape = params.softmaxMax->GetViewShape();
    const auto& softmaxSumShape = params.softmaxSum->GetViewShape();
    const auto& topkIndicesShape = params.topkIndices->GetViewShape();

    const int64_t t1 = queryShape.GetDim(0);
    const int64_t n1 = queryShape.GetDim(1);
    const int64_t headDimQ = queryShape.GetDim(DIM_NUM_2);
    const int64_t n2 = keyShape.GetDim(1);
    const int64_t headDimK = keyShape.GetDim(DIM_NUM_2);
    const int64_t headDimV = valueShape.GetDim(DIM_NUM_2);

    CHECK_COND(params.headNum == n1, ACLNN_ERR_PARAM_INVALID,
               "headNum[%ld] must be equal to the head num of query[%ld].", params.headNum, n1);
    CHECK_COND(n2 > 0 && n1 % n2 == 0, ACLNN_ERR_PARAM_INVALID,
               "The head num of query[%ld] must be a positive multiple of the head num of key[%ld].", n1, n2);
    CHECK_COND(headDimQ == HEAD_DIM_128 || headDimQ == HEAD_DIM_192, ACLNN_ERR_PARAM_INVALID,
               "The head dim of query must be 128 or 192, but got [%ld].", headDimQ);
    CHECK_COND(headDimK == headDimQ, ACLNN_ERR_PARAM_INVALID, "The head dim of query[%ld] and key[%ld] must be equal.",
               headDimQ, headDimK);
    CHECK_COND(headDimV == HEAD_DIM_128 || headDimV == HEAD_DIM_192, ACLNN_ERR_PARAM_INVALID,
               "The head dim of value must be 128 or 192, but got [%ld].", headDimV);
    CHECK_COND(headDimV <= headDimQ, ACLNN_ERR_PARAM_INVALID,
               "The head dim of value[%ld] must not be greater than the head dim of query[%ld].", headDimV, headDimQ);
    CHECK_COND(keyShape.GetDim(0) == valueShape.GetDim(0), ACLNN_ERR_PARAM_INVALID,
               "The T dim of key[%ld] and value[%ld] must be equal.", keyShape.GetDim(0), valueShape.GetDim(0));

    CHECK_COND(softmaxMaxShape.GetDim(0) == t1 && softmaxMaxShape.GetDim(1) == n1 &&
                   softmaxMaxShape.GetDim(DIM_NUM_2) == SOFTMAX_TAIL_DIM,
               ACLNN_ERR_PARAM_INVALID, "softmaxMax shape must be [T1, N1, 8], but got [%s].",
               op::ToString(softmaxMaxShape).GetString());
    CHECK_COND(softmaxSumShape.GetDim(0) == t1 && softmaxSumShape.GetDim(1) == n1 &&
                   softmaxSumShape.GetDim(DIM_NUM_2) == SOFTMAX_TAIL_DIM,
               ACLNN_ERR_PARAM_INVALID, "softmaxSum shape must be [T1, N1, 8], but got [%s].",
               op::ToString(softmaxSumShape).GetString());
    CHECK_COND(attentionOutShape.GetDim(0) == t1 && attentionOutShape.GetDim(1) == n1 &&
                   attentionOutShape.GetDim(DIM_NUM_2) == headDimV,
               ACLNN_ERR_PARAM_INVALID, "attentionOut shape must be [T1, N1, D2], but got [%s].",
               op::ToString(attentionOutShape).GetString());
    CHECK_COND(attentionOutGradShape == attentionOutShape, ACLNN_ERR_PARAM_INVALID,
               "attentionOutGrad shape[%s] must be equal to attentionOut shape[%s].",
               op::ToString(attentionOutGradShape).GetString(), op::ToString(attentionOutShape).GetString());
    CHECK_COND(topkIndicesShape.GetDim(0) == t1 && topkIndicesShape.GetDim(1) == n2, ACLNN_ERR_PARAM_INVALID,
               "The first two dims of topkIndices must be [T1, N2] = [%ld, %ld], but got [%s].", t1, n2,
               op::ToString(topkIndicesShape).GetString());
    CHECK_COND(topkIndicesShape.GetDim(DIM_NUM_2) == params.selectedBlockCount, ACLNN_ERR_PARAM_INVALID,
               "The last dim of topkIndices[%ld] must be equal to selectedBlockCount[%ld].",
               topkIndicesShape.GetDim(DIM_NUM_2), params.selectedBlockCount);

    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(const NsaSelectedAttentionGradParams& params)
{
    CHECK_COND(params.query != nullptr, ACLNN_ERR_PARAM_NULLPTR, "query must not be nullptr.");
    CHECK_COND(params.key != nullptr, ACLNN_ERR_PARAM_NULLPTR, "key must not be nullptr.");
    CHECK_COND(params.value != nullptr, ACLNN_ERR_PARAM_NULLPTR, "value must not be nullptr.");
    CHECK_COND(params.attentionOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "attentionOut must not be nullptr.");
    CHECK_COND(params.attentionOutGrad != nullptr, ACLNN_ERR_PARAM_NULLPTR, "attentionOutGrad must not be nullptr.");
    CHECK_COND(params.softmaxMax != nullptr, ACLNN_ERR_PARAM_NULLPTR, "softmaxMax must not be nullptr.");
    CHECK_COND(params.softmaxSum != nullptr, ACLNN_ERR_PARAM_NULLPTR, "softmaxSum must not be nullptr.");
    CHECK_COND(params.topkIndices != nullptr, ACLNN_ERR_PARAM_NULLPTR, "topkIndices must not be nullptr.");
    CHECK_COND(params.dqOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dqOut must not be nullptr.");
    CHECK_COND(params.dkOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dkOut must not be nullptr.");
    CHECK_COND(params.dvOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "dvOut must not be nullptr.");

    auto query_dtype = params.query->GetDataType();
    CHECK_COND(query_dtype == ACL_FLOAT16 || query_dtype == ACL_BF16, ACLNN_ERR_PARAM_INVALID,
               "The dtype of query is not support.");
    CHECK_COND(query_dtype == params.key->GetDataType(), ACLNN_ERR_PARAM_INVALID, "key dtype error.");
    CHECK_COND(query_dtype == params.value->GetDataType(), ACLNN_ERR_PARAM_INVALID, "value dtype error.");
    CHECK_COND(query_dtype == params.attentionOut->GetDataType(), ACLNN_ERR_PARAM_INVALID, "attentionOut dtype error.");
    CHECK_COND(query_dtype == params.attentionOutGrad->GetDataType(), ACLNN_ERR_PARAM_INVALID,
               "attentionOutGrad dtype error.");
    CHECK_COND(query_dtype == params.dqOut->GetDataType(), ACLNN_ERR_PARAM_INVALID, "dqOut dtype error.");
    CHECK_COND(query_dtype == params.dkOut->GetDataType(), ACLNN_ERR_PARAM_INVALID, "dkOut dtype error.");
    CHECK_COND(query_dtype == params.dvOut->GetDataType(), ACLNN_ERR_PARAM_INVALID, "dvOut dtype error.");

    auto softmaxMax_dtype = params.softmaxMax->GetDataType();
    auto softmaxSum_dtype = params.softmaxSum->GetDataType();
    auto topkIndices_dtype = params.topkIndices->GetDataType();
    CHECK_COND(softmaxMax_dtype == ACL_FLOAT, ACLNN_ERR_PARAM_INVALID, "softmaxMax dtype error.");
    CHECK_COND(softmaxSum_dtype == ACL_FLOAT, ACLNN_ERR_PARAM_INVALID, "softmaxSum dtype error.");
    CHECK_COND(topkIndices_dtype == ACL_INT32, ACLNN_ERR_PARAM_INVALID, "topkIndices dtype error.");
    if (params.attenMaskOptional != nullptr) {
        auto attenMask_dtype = params.attenMaskOptional->GetDataType();
        CHECK_COND(attenMask_dtype == ACL_BOOL || attenMask_dtype == ACL_UINT8, ACLNN_ERR_PARAM_INVALID,
                   "The dtype of attenMaskOptional must be BOOL or UINT8, but got [%s].",
                   op::ToString(DataType(attenMask_dtype)).GetString());
    }

    return CheckShapeAndAttr(params);
}

static bool CheckViewShape(const aclTensor* input1, const aclTensor* input2)
{
    int64_t* viewDims1 = nullptr;
    uint64_t viewDimsNum1 = 0;
    auto ret = aclGetViewShape(input1, &viewDims1, &viewDimsNum1);

    int64_t* viewDims2 = nullptr;
    uint64_t viewDimsNum2 = 0;
    ret = aclGetViewShape(input2, &viewDims2, &viewDimsNum2);

    if (viewDimsNum1 != viewDimsNum2) {
        return false;
    }

    for (uint64_t i = 0; i < viewDimsNum1; i++) {
        if (viewDims1[i] != viewDims2[i]) {
            return false;
        }
    }

    return true;
}
static void CheckFormatValid(const NsaSelectedAttentionGradParams& params)
{
    // 检查输入张量的格式是否为 ND 格式
    if (params.query != nullptr) {
        op::Format format = params.query->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of query gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.key != nullptr) {
        op::Format format = params.key->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of key gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.value != nullptr) {
        op::Format format = params.value->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of value gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.attentionOut != nullptr) {
        op::Format format = params.attentionOut->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of attentionOut gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.softmaxMax != nullptr) {
        op::Format format = params.softmaxMax->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of softmaxMax gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.softmaxSum != nullptr) {
        op::Format format = params.softmaxSum->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of softmaxSum gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.topkIndices != nullptr) {
        op::Format format = params.topkIndices->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of topkIndices gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.softmaxMax != nullptr) {
        op::Format format = params.softmaxMax->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of softmaxMax gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.softmaxSum != nullptr) {
        op::Format format = params.softmaxSum->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of softmaxSum gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }

    if (params.topkIndices != nullptr) {
        op::Format format = params.topkIndices->GetStorageFormat();
        if (format != ge::FORMAT_ND) {
            OP_LOGW("Format of topkIndices gets [%s], this format may lead to precision failure.",
                    op::ToString(format).GetString());
        }
    }
}

aclnnStatus ContiguousAndNsaSelectedAttentionGrad(const NsaSelectedAttentionGradParams& params, aclOpExecutor* executor,
                                                  const aclTensor* dqOut, const aclTensor* dkOut,
                                                  const aclTensor* dvOut)
{
    auto queryContiguous = l0op::Contiguous(params.query, executor);
    CHECK_RET(queryContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto keyContiguous = l0op::Contiguous(params.key, executor);
    CHECK_RET(keyContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto valueContiguous = l0op::Contiguous(params.value, executor);
    CHECK_RET(valueContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto attentionOutContiguous = l0op::Contiguous(params.attentionOut, executor);
    CHECK_RET(attentionOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto attentionOutGradContiguous = l0op::Contiguous(params.attentionOutGrad, executor);
    CHECK_RET(attentionOutGradContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto softmaxMaxContiguous = l0op::Contiguous(params.softmaxMax, executor);
    CHECK_RET(softmaxMaxContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto softmaxSumContiguous = l0op::Contiguous(params.softmaxSum, executor);
    CHECK_RET(softmaxSumContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto topkIndicesContiguous = l0op::Contiguous(params.topkIndices, executor);
    CHECK_RET(topkIndicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    const aclTensor* attenMaskOptionalContiguous = nullptr;
    if (params.attenMaskOptional != nullptr) {
        attenMaskOptionalContiguous = l0op::Contiguous(params.attenMaskOptional, executor);
        CHECK_RET(attenMaskOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    string inputLayoutStr = GetUpperInputLayout(params.inputLayout);

    // call l0 interface
    auto result = l0op::NsaSelectedAttentionGrad(
        queryContiguous, keyContiguous, valueContiguous, attentionOutContiguous, attentionOutGradContiguous,
        softmaxMaxContiguous, softmaxSumContiguous, topkIndicesContiguous, params.actualSeqQLenOptional,
        params.actualSeqKvLenOptional, attenMaskOptionalContiguous, params.scaleValue, params.selectedBlockCount,
        params.selectedBlockSize, params.headNum, inputLayoutStr.c_str(), params.sparseMode, executor);
    // convert output tensor to contiguous tensor
    CHECK_RET(result[0] != nullptr && result[1] != nullptr && result[2] != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_COND(CheckViewShape(result[0], dqOut) != false, ACLNN_ERR_PARAM_INVALID, "The dq Tensor is Invaild");
    CHECK_COND(CheckViewShape(result[1], dkOut) != false, ACLNN_ERR_PARAM_INVALID, "The dk Tensor is Invaild");
    CHECK_COND(CheckViewShape(result[2], dvOut) != false, ACLNN_ERR_PARAM_INVALID, "The dv Tensor is Invaild");

    auto viewCopyResult = l0op::ViewCopy(result[0], dqOut, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    viewCopyResult = l0op::ViewCopy(result[1], dkOut, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    viewCopyResult = l0op::ViewCopy(result[2], dvOut, executor);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaSelectedAttentionGradGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key, const aclTensor* value, const aclTensor* attentionOut,
    const aclTensor* attentionOutGrad, const aclTensor* softmaxMax, const aclTensor* softmaxSum,
    const aclTensor* topkIndices, const aclIntArray* actualSeqQLenOptional, const aclIntArray* actualSeqKvLenOptional,
    const aclTensor* attenMaskOptional, double scaleValue, int64_t selectedBlockSize, int64_t selectedBlockCount,
    int64_t headNum, char* inputLayout, int64_t sparseMode, const aclTensor* dqOut, const aclTensor* dkOut,
    const aclTensor* dvOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnNsaSelectedAttentionGrad,
                   DFX_IN(query, key, value, attentionOut, attentionOutGrad, softmaxMax, softmaxSum, topkIndices,
                          actualSeqQLenOptional, actualSeqKvLenOptional, attenMaskOptional, scaleValue,
                          selectedBlockSize, selectedBlockCount, headNum, inputLayout, sparseMode),
                   DFX_OUT(dqOut, dkOut, dvOut));
    NsaSelectedAttentionGradParams params{query,
                                          key,
                                          value,
                                          attentionOut,
                                          attentionOutGrad,
                                          softmaxMax,
                                          softmaxSum,
                                          topkIndices,
                                          actualSeqQLenOptional,
                                          actualSeqKvLenOptional,
                                          attenMaskOptional,
                                          scaleValue,
                                          selectedBlockSize,
                                          selectedBlockCount,
                                          headNum,
                                          inputLayout,
                                          sparseMode,
                                          dqOut,
                                          dkOut,
                                          dvOut};
    // check params
    aclnnStatus ret = CheckParams(params);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(ret, "aclnnNsaSelectedAttentionGradGetWorkspaceSize checkParams failed. ret=%d.", ret);
        return ret;
    }
    CheckFormatValid(params);
    // create OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    if (query->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // call l0 interface
    ret = ContiguousAndNsaSelectedAttentionGrad(params, uniqueExecutor.get(), dqOut, dkOut, dvOut);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(ret, "ContiguousAndNsaSelectedAttentionGrad failed. ret=%d.", ret);
        return ret;
    }
    // get workspace size
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnNsaSelectedAttentionGrad(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnNsaSelectedAttentionGrad);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
