/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base/err_msg.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/platform.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_incre_flash_attention_v4.h"
#include "aclnnInner_incre_flash_attention.h" // 该文件是自动生成，在build/autogen/inner路径下

#include <cstring>

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

static const int64_t INNER_PRECISE_HIGH_PRECISION = 0;
static const int64_t INNER_PRECISE_HIGH_PERFORMANCE = 1;

static const std::initializer_list<op::DataType> ATTEN_MASK_DTYPE_SUPPORT_LIST = {DataType::DT_BOOL, DataType::DT_INT8,
                                                                                  DataType::DT_UINT8};

static const std::initializer_list<op::DataType> PSE_SHIFT_DTYPE_SUPPORT_LIST = {DataType::DT_FLOAT16,
                                                                                 DataType::DT_BF16};

static const std::initializer_list<op::DataType> ATTENTION_OUT_DTYPE_SUPPORT_LIST = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT8};

static aclnnStatus CheckIncreFlashAttentionV4Params(const aclTensor* query, const aclTensor* pseShift,
                                                    const aclTensor* attenMask, const aclTensor* attentionOut,
                                                    const char* inputLayout, int64_t numHeads, int64_t numKeyValueHeads,
                                                    int64_t innerPrecise)
{
    if (pseShift != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(pseShift, PSE_SHIFT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
        if (query != nullptr) {
            OP_CHECK_DTYPE_NOT_SAME(pseShift, query, return ACLNN_ERR_PARAM_INVALID);
        }
    }
    if (attenMask != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(attenMask, ATTEN_MASK_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    }
    if (attentionOut != nullptr) {
        OP_CHECK_DTYPE_NOT_SUPPORT(attentionOut, ATTENTION_OUT_DTYPE_SUPPORT_LIST, return ACLNN_ERR_PARAM_INVALID);
    }

    if (inputLayout != nullptr && strcmp(inputLayout, "BSH") != 0 && strcmp(inputLayout, "BNSD") != 0 &&
        strcmp(inputLayout, "BSND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "The inputLayout[%s] is invalid, only \"BSH\", \"BNSD\", \"BSND\" are supported.", inputLayout);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (innerPrecise != INNER_PRECISE_HIGH_PRECISION && innerPrecise != INNER_PRECISE_HIGH_PERFORMANCE) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The innerPrecise[%ld] is invalid, only 0, 1 are supported.", innerPrecise);
        return ACLNN_ERR_PARAM_INVALID;
    }

    if (numHeads <= 0 || numKeyValueHeads < 0 || (numKeyValueHeads > 0 && numHeads % numKeyValueHeads != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The numHeads[%ld] and numKeyValueHeads[%ld] are invalid.", numHeads,
                numKeyValueHeads);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnIncreFlashAttentionV4GetWorkspaceSize(
    const aclTensor* query, const aclTensorList* key, const aclTensorList* value, const aclTensor* pseShift,
    const aclTensor* attenMask, const aclIntArray* actualSeqLengths, const aclTensor* dequantScale1,
    const aclTensor* quantScale1, const aclTensor* dequantScale2, const aclTensor* quantScale2,
    const aclTensor* quantOffset2, const aclTensor* antiquantScale, const aclTensor* antiquantOffset,
    const aclTensor* blocktable, const aclTensor* kvPaddingSize, int64_t numHeads, double scaleValue, char* inputLayout,
    int64_t numKeyValueHeads, int64_t blockSize, int64_t innerPrecise, const aclTensor* attentionOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1009", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"aclnnIncreFlashAttentionV4",
                                      "Interface aclnnIncreFlashAttention versions V1 to V4 are no longer supported "
                                      "on Ascend950."}));
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_5102) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1009", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"aclnnIncreFlashAttentionV4",
                                      "Interface aclnnIncreFlashAttention versions V1 to V4 are no longer supported "
                                      "on Ascend5102."}));
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    aclnnStatus checkRet = CheckIncreFlashAttentionV4Params(query, pseShift, attenMask, attentionOut, inputLayout,
                                                            numHeads, numKeyValueHeads, innerPrecise);
    if (checkRet != ACLNN_SUCCESS) {
        return checkRet;
    }
    aclnnStatus ret = aclnnInnerIncreFlashAttentionGetWorkspaceSize(
        query, key, value, pseShift, attenMask, actualSeqLengths, dequantScale1, quantScale1, dequantScale2,
        quantScale2, quantOffset2, antiquantScale, antiquantOffset, blocktable, kvPaddingSize, numHeads, scaleValue,
        inputLayout, numKeyValueHeads, blockSize, innerPrecise, attentionOut, workspaceSize, executor);

    return ret;
}

aclnnStatus aclnnIncreFlashAttentionV4(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       const aclrtStream stream)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1009", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"aclnnIncreFlashAttentionV4",
                                      "Interface aclnnIncreFlashAttention versions V1 to V4 are no longer supported "
                                      "on Ascend950."}));
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_5102) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1009", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{"aclnnIncreFlashAttentionV4",
                                      "Interface aclnnIncreFlashAttention versions V1 to V4 are no longer supported "
                                      "on Ascend5102."}));
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    aclnnStatus ret = aclnnInnerIncreFlashAttention(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
