/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attention_tiling.h"
#include <queue>
#include <cmath>
#include <cfloat>
#include <register/op_impl_registry.h>
#include "base/data_copy_transpose_tiling.h"
#include "base/tiling_templates_registry.h"

namespace optiling {

constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t FRACTAL_NUM = 16;
constexpr uint32_t MAX_S_INNER = 512;
constexpr uint32_t NUM2 = 2;
constexpr uint32_t SIZE_2 = 2;
constexpr uint32_t SIZE_4 = 4;
constexpr uint32_t BASE_SIZE_128 = 128;
constexpr uint32_t BASE_SIZE_64 = 64;
constexpr uint32_t SOFTMAX_BASE_M = 8;
constexpr uint32_t SOFTMAX_BASE_N = 64;

constexpr int HIGH_PRECISION = 0;
constexpr int HIGH_PERFORMANCE = 1;

constexpr uint32_t KEY_NON_FLASH_F16 = 100;
constexpr uint32_t KEY_NON_FLASH_F32 = 101;
constexpr uint32_t REMAIN_SIZE = 100 * 1024 * 1024;
const int64_t DIMS_3 = 3;
const int64_t DIMS_4 = 4;
constexpr size_t PSE_INPUT_INDEX = 3;
constexpr size_t PSE_DIM_2 = 2;
constexpr uint32_t PSE_SHAPE_TYPE_B_N_S_S = 0;
constexpr uint32_t PSE_SHAPE_TYPE_B_N_1_S = 1;
constexpr uint32_t PSE_SHAPE_TYPE_NONE = 3;
constexpr uint32_t ATTE_SHAPE_TYPE_NONE = 3;

constexpr size_t ATTENTION_MASK_INPUT_INDEX = 6;
constexpr size_t QUERY_INPUT_INDEX = 0;
constexpr size_t KEY_INPUT_INDEX = 1;
constexpr size_t VALUE_INPUT_INDEX = 2;
constexpr size_t SOFTMAXMAX_OUPUT_INDEX = 0;
constexpr size_t SOFTMAXSUM_OUPUT_INDEX = 1;
constexpr size_t ATTENTIONOUT_OUPUT_INDEX = 3;
constexpr size_t INPUTLAYOUT_ATTRS_INDEX = 5;
constexpr size_t MIN_COPY_UINT_SIZE = 32;

ge::graphStatus CheckParams(const gert::TilingContext* context)
{
    if (context->GetInputShape(QUERY_INPUT_INDEX) != nullptr && context->GetInputShape(KEY_INPUT_INDEX) != nullptr &&
        context->GetInputShape(VALUE_INPUT_INDEX) != nullptr && context->GetAttrs() != nullptr) {
        auto& queryShape = context->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
        auto& keyShape = context->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
        auto& valueShape = context->GetInputShape(VALUE_INPUT_INDEX)->GetStorageShape();
        const char* inputLayout = context->GetAttrs()->GetAttrPointer<char>(INPUTLAYOUT_ATTRS_INDEX);
        if (strlen(inputLayout) == 3) { // 3: BSH or SBH
            if (inputLayout[0] == 'B') {
                // layout is BSH
                OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                            TILING_INNER_ERR_REPORT(context->GetNodeName(), "query or key shape is invalid"),
                            return ge::GRAPH_FAILED);
            } else {
                if (inputLayout[0] == 'T') { // TND  N1 != N2
                    return ge::SUCCESS;
                }
                // layout is SBH
                OP_CHECK_IF((queryShape.GetDim(1) != keyShape.GetDim(1)),
                            TILING_INNER_ERR_REPORT(context->GetNodeName(), "query or key shape is invalid"),
                            return ge::GRAPH_FAILED);
            }
        } else if (strlen(inputLayout) == 4) { // 4: layout is BNSD or BSND
            OP_CHECK_IF((keyShape != valueShape),
                        TILING_INNER_ERR_REPORT(context->GetNodeName(), "key or value shape is invalid"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((queryShape.GetDim(0) != keyShape.GetDim(0)),
                        TILING_INNER_ERR_REPORT(context->GetNodeName(), "query or key shape is invalid"),
                        return ge::GRAPH_FAILED);
            OP_CHECK_IF((queryShape.GetDim(3) != keyShape.GetDim(3)),
                        TILING_INNER_ERR_REPORT(context->GetNodeName(), "query or key shape is invalid"),
                        return ge::GRAPH_FAILED);
        } else {
            OP_LOGW(context->GetNodeName(), "invalid input_layout[%s].", inputLayout);
            return ge::GRAPH_FAILED;
        }
        return ge::SUCCESS;
    }
    OP_LOGE(context->GetNodeName(), "fail to get shape or attr from context");
    return ge::GRAPH_FAILED;
}

ge::graphStatus FlashAttentionTilingFunc(gert::TilingContext* context)
{
    // if (CheckParams(context) != ge::GRAPH_SUCCESS) {
    //     return ge::GRAPH_FAILED;
    // }

    auto resultCode = TilingRegistry::GetInstance().DoTilingImpl(context);
    return resultCode;
}

IMPL_OP_OPTILING(FlashAttention).Tiling(FlashAttentionTilingFunc);

} // namespace optiling
