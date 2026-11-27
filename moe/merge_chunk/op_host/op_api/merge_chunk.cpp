/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "merge_chunk.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(MergeChunk);

aclnnStatus LaunchMergeChunk(const aclTensorList* input, const aclTensorList* counts, int64_t paths, int64_t chunks,
                             const aclTensorList* output, const aclTensor* outCounts, aclOpExecutor* executor)
{
    return ADD_TO_LAUNCHER_LIST_AICORE(MergeChunk, OP_INPUT(input, counts), OP_OUTPUT(output, outCounts),
                                       OP_ATTR(paths, chunks));
}
} // namespace l0op
