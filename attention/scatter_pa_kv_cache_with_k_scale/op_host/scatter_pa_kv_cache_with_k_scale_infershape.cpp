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
#include "version/metadef_version.h"

using namespace ge;

namespace ops {

constexpr size_t INPUT_KEY_CACHE_IDX = 2;
constexpr size_t INPUT_VALUE_CACHE_IDX = 3;
constexpr size_t INPUT_KEY_SCALE_CACHE_IDX = 6;
constexpr size_t OUTPUT_KEY_CACHE_IDX = 0;
constexpr size_t OUTPUT_VALUE_CACHE_IDX = 1;
constexpr size_t OUTPUT_KEY_SCALE_CACHE_IDX = 2;

static ge::graphStatus InferShapeScatterPaKvCacheWithKScale(gert::InferShapeContext* context)
{
    // inferShape key_cache
    OP_LOGD(context, "Begin to do InferShapeScatterPaKvCacheWithKScale");
    auto inputKeyCacheShape = context->GetInputShape(INPUT_KEY_CACHE_IDX);
    OP_CHECK_IF(inputKeyCacheShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "key_cache"),
                return ge::GRAPH_FAILED);
    auto outputKeyCacheShape = context->GetOutputShape(OUTPUT_KEY_CACHE_IDX);
    OP_CHECK_IF(outputKeyCacheShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "key_cache output shape"), return ge::GRAPH_FAILED);
    *outputKeyCacheShape = *inputKeyCacheShape;

    // inferShape value_cache
    auto inputValueCacheShape = context->GetInputShape(INPUT_VALUE_CACHE_IDX);
    OP_CHECK_IF(inputValueCacheShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "value_cache"),
                return ge::GRAPH_FAILED);
    auto outputValueCacheShape = context->GetOutputShape(OUTPUT_VALUE_CACHE_IDX);
    OP_CHECK_IF(outputValueCacheShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "value_cache output shape"),
                return ge::GRAPH_FAILED);
    *outputValueCacheShape = *inputValueCacheShape;

    // inferShape key_scale_cache
    const gert::Shape* kscShape = context->GetInputShape(INPUT_KEY_SCALE_CACHE_IDX);
    OP_CHECK_IF(kscShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "key_scale_cache"),
                return ge::GRAPH_FAILED);
    gert::Shape* outKscShape = context->GetOutputShape(OUTPUT_KEY_SCALE_CACHE_IDX);
    OP_CHECK_IF(outKscShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "key_scale_cache output shape"),
                return ge::GRAPH_FAILED);
    *outKscShape = *kscShape;
    OP_LOGD(context, "End to do InferShapeScatterPaKvCacheWithKScale");

    return GRAPH_SUCCESS;
}

ge::graphStatus InferDataType4ScatterPaKvCacheWithKScale(gert::InferDataTypeContext* context)
{
    OP_LOGD(context, "Begin to do InferDataType4ScatterPaKvCacheWithKScale");
    context->SetOutputDataType(OUTPUT_KEY_CACHE_IDX, context->GetInputDataType(INPUT_KEY_CACHE_IDX));
    context->SetOutputDataType(OUTPUT_VALUE_CACHE_IDX, context->GetInputDataType(INPUT_VALUE_CACHE_IDX));
    context->SetOutputDataType(OUTPUT_KEY_SCALE_CACHE_IDX, context->GetInputDataType(INPUT_KEY_SCALE_CACHE_IDX));
    OP_LOGD(context, "End to do InferDataType4ScatterPaKvCacheWithKScale");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ScatterPaKvCacheWithKScale)
    .InferShape(InferShapeScatterPaKvCacheWithKScale)
    .InferDataType(InferDataType4ScatterPaKvCacheWithKScale)
#if defined(METADEF_VERSION_NUM) && METADEF_VERSION_NUM >= 90200000
    .SetSupportPcieThrough()
#endif
    ;
} // namespace ops
