/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_turbo_quant_attention_score.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace {
OP_TYPE_REGISTER(TqQueryPack);
OP_TYPE_REGISTER(TqScoreCompute);
} // namespace

extern "C" aclnnStatus aclnnTurboQuantAttentionScoreGetWorkspaceSize(
    const aclTensor* queryRotated, const aclTensor* queryQjl, const aclTensor* keyCacheIdx,
    const aclTensor* keyCacheQjl, const aclTensor* keyCacheNorm, const aclTensor* keyCacheGamma,
    const aclTensor* blockTable, const aclTensor* seqLens, int64_t mseBits, const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executorResult)
{
    if (queryRotated == nullptr || queryQjl == nullptr || keyCacheIdx == nullptr || keyCacheQjl == nullptr ||
        keyCacheNorm == nullptr || keyCacheGamma == nullptr || blockTable == nullptr || seqLens == nullptr ||
        out == nullptr || workspaceSize == nullptr || executorResult == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (uniqueExecutor.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    auto* executor = uniqueExecutor.get();
    if (queryRotated->GetViewShape().GetDimNum() != 3 || blockTable->GetViewShape().GetDimNum() != 2 || mseBits < 2 ||
        mseBits > 4) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto tokens = queryRotated->GetViewShape().GetDim(0);
    const auto queryHeads = queryRotated->GetViewShape().GetDim(1);
    const auto batch = blockTable->GetViewShape().GetDim(0);
    if (batch <= 0 || tokens < 0 || queryHeads <= 0 || queryRotated->GetViewShape().GetDim(2) != 128 ||
        queryQjl->GetViewShape() != queryRotated->GetViewShape() || queryRotated->GetDataType() != DataType::DT_BF16 ||
        queryQjl->GetDataType() != DataType::DT_BF16) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (keyCacheIdx->GetViewShape().GetDimNum() != 4 || keyCacheQjl->GetViewShape().GetDimNum() != 4 ||
        keyCacheNorm->GetViewShape().GetDimNum() != 3 ||
        keyCacheGamma->GetViewShape() != keyCacheNorm->GetViewShape() || seqLens->GetViewShape().GetDimNum() != 1) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto keyHeads = keyCacheIdx->GetViewShape().GetDim(2);
    const auto blocks = keyCacheIdx->GetViewShape().GetDim(0);
    const auto pages = blockTable->GetViewShape().GetDim(1);
    if (keyHeads <= 0 || queryHeads % keyHeads != 0 || pages < 0 || blocks < 0 ||
        keyCacheIdx->GetViewShape().GetDim(1) != 128 || keyCacheIdx->GetViewShape().GetDim(3) != 16 * mseBits ||
        keyCacheQjl->GetViewShape() != Shape({blocks, 128, keyHeads, 16}) ||
        keyCacheNorm->GetViewShape() != Shape({blocks, 128, keyHeads}) || seqLens->GetViewShape().GetDim(0) != batch) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (keyCacheIdx->GetDataType() != DataType::DT_UINT8 || keyCacheQjl->GetDataType() != DataType::DT_UINT8 ||
        blockTable->GetDataType() != DataType::DT_INT32 || seqLens->GetDataType() != DataType::DT_INT32 ||
        out->GetDataType() != DataType::DT_FLOAT) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((keyCacheNorm->GetDataType() != DataType::DT_BF16 && keyCacheNorm->GetDataType() != DataType::DT_FLOAT16) ||
        keyCacheGamma->GetDataType() != keyCacheNorm->GetDataType()) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (out->GetViewShape().GetDimNum() != 3) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    const auto maxKvLen = out->GetViewShape().GetDim(2);
    if (maxKvLen < 0 || maxKvLen > pages * 128 || (batch != 1 && tokens != batch) ||
        out->GetViewShape() != Shape({tokens, queryHeads, maxKvLen})) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (tokens == 0 || maxKvLen == 0) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executorResult);
        return ACLNN_SUCCESS;
    }
    const aclTensor* packedQuery = queryRotated;
    if (batch == 1 && tokens > 1) {
        auto* temporary = executor->AllocTensor(Shape({queryHeads, (tokens + 127) / 128 * 128, 512}), DataType::DT_BF16,
                                                Format::FORMAT_ND);
        if (temporary == nullptr) {
            return ACLNN_ERR_INNER_NULLPTR;
        }
        const auto packResult = ADD_TO_LAUNCHER_LIST_AICORE(TqQueryPack, OP_INPUT(queryRotated, queryQjl),
                                                            OP_OUTPUT(temporary), OP_ATTR(mseBits));
        if (packResult != ACLNN_SUCCESS) {
            return packResult;
        }
        packedQuery = temporary;
    }
    const auto scoreResult =
        ADD_TO_LAUNCHER_LIST_AICORE(TqScoreCompute,
                                    OP_INPUT(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm,
                                             keyCacheGamma, blockTable, seqLens, packedQuery),
                                    OP_OUTPUT(out), OP_ATTR(mseBits, maxKvLen));
    if (scoreResult != ACLNN_SUCCESS) {
        return scoreResult;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executorResult);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnTurboQuantAttentionScore(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                     aclrtStream stream)
{
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
