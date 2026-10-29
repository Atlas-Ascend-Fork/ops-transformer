/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <vector>
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {

constexpr int HIGH_PRECISION = 0;
constexpr int HIGH_PERFORMANCE = 1;

const std::vector<gert::Shape> OLD_TEMPLATE_SUPPORTED_SHAPE = {
    // 5d shape [N,S1,S2,D,B]
};

// 检查是否是8月份的算子 CASE
//   True: 使用老的算子逻辑
//   False: 使用新的泛化算子逻辑
bool IsFlashAttentionScoreTilingGradOldCases(gert::TilingContext* context)
{
    const gert::StorageShape* queryShape = context->GetInputShape(0);
    const gert::StorageShape* keyShape = context->GetInputShape(1);
    const char* inputLayout = context->GetAttrs()->GetAttrPointer<char>(5);
    int32_t n = *context->GetAttrs()->GetAttrPointer<int32_t>(4); // 4 is N index

    // 1. get input shape and attr info
    int64_t b;
    int64_t s1;
    int64_t h1;
    int64_t s2;
    int64_t h2;

    if (strcmp(inputLayout, "BSH") == 0) {
        b = queryShape->GetStorageShape().GetDim(0);
        s1 = queryShape->GetStorageShape().GetDim(1);
        h1 = queryShape->GetStorageShape().GetDim(2);
        s2 = keyShape->GetStorageShape().GetDim(1);
        h2 = keyShape->GetStorageShape().GetDim(2);
    } else if (strcmp(inputLayout, "SBH") == 0) {
        b = queryShape->GetStorageShape().GetDim(1);
        s1 = queryShape->GetStorageShape().GetDim(0);
        h1 = queryShape->GetStorageShape().GetDim(2);
        s2 = keyShape->GetStorageShape().GetDim(0);
        h2 = keyShape->GetStorageShape().GetDim(2);
    } else {
        return false;
    }

    // 2. check input shape
    if (h1 != h2) {
        OP_LOGD(context, "Q and K's shape H is unequal.");
        return false;
    }
    if (n == 0) {
        OP_LOGD(context, "Unsupported shape, please check op_tiling/runtime/flash_attention_score_grad/white_list.h .");
        return false;
    }

    gert::Shape input_shape_4d = {n, s1, s2, h1 / n};
    gert::Shape input_shape_5d = {n, s1, s2, h1 / n, b};
    bool unsupported_4d = std::find(OLD_TEMPLATE_SUPPORTED_SHAPE.begin(), OLD_TEMPLATE_SUPPORTED_SHAPE.end(),
                                    input_shape_4d) == OLD_TEMPLATE_SUPPORTED_SHAPE.end();
    bool unsupported_5d = std::find(OLD_TEMPLATE_SUPPORTED_SHAPE.begin(), OLD_TEMPLATE_SUPPORTED_SHAPE.end(),
                                    input_shape_5d) == OLD_TEMPLATE_SUPPORTED_SHAPE.end();
    if (unsupported_4d && unsupported_5d) {
        OP_LOGD(context, "Unsupported shape, please check white_list.");
        OP_LOGD(context, "current n, s1, s2, d, b is [%d], [%ld], [%ld], [%ld], [%ld]", n, s1, s2, h1 / n, b);
        return false;
    }
    return true;
}

} // namespace optiling
