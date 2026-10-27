/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_TURBO_QUANT_ATTENTION_SCORE_H_
#define ACLNN_TURBO_QUANT_ATTENTION_SCORE_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnTurboQuantAttentionScoreGetWorkspaceSize
 * parameters :
 * queryRotated : required
 * queryQjl : required
 * keyCacheIdx : required
 * keyCacheQjl : required
 * keyCacheNorm : required
 * keyCacheGamma : required
 * blockTable : required
 * seqLens : required
 * mseBits : required
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnTurboQuantAttentionScoreGetWorkspaceSize(
    const aclTensor* queryRotated, const aclTensor* queryQjl, const aclTensor* keyCacheIdx,
    const aclTensor* keyCacheQjl, const aclTensor* keyCacheNorm, const aclTensor* keyCacheGamma,
    const aclTensor* blockTable, const aclTensor* seqLens, int64_t mseBits, const aclTensor* out,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/* function: aclnnTurboQuantAttentionScore
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnTurboQuantAttentionScore(void* workspace,
                                                                                 uint64_t workspaceSize,
                                                                                 aclOpExecutor* executor,
                                                                                 aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
