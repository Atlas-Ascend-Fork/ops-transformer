/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <vector>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr uint64_t TILING_KEY_DEFAULT = 0;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t MIN_MSE_BITS = 2;
constexpr int64_t MAX_MSE_BITS = 4;
} // namespace

BEGIN_TILING_DATA_DEF(TqQueryPackTilingData)
TILING_DATA_FIELD_DEF_ARR(uint32_t, 16400, payload);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TqQueryPack, TqQueryPackTilingData)

static ge::graphStatus PackTiling(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TqQueryPack", "tiling context is null"), return ge::GRAPH_FAILED);
    const auto* queryShape = context->GetInputShape(0);
    const auto* queryQjlShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, queryQjlShape);
    OP_CHECK_IF(queryShape->GetStorageShape() != queryQjlShape->GetStorageShape(),
                OP_LOGE(context, "query_qjl shape must equal query shape"), return ge::GRAPH_FAILED);
    const auto shape = queryShape->GetStorageShape();
    OP_CHECK_IF(shape.GetDimNum() != 3 || shape.GetDim(2) != HEAD_DIM,
                OP_LOGE(context, "query shape must be [tokens, heads, 128]"), return ge::GRAPH_FAILED);
    const auto* attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* bits = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_IF(bits == nullptr || *bits < MIN_MSE_BITS || *bits > MAX_MSE_BITS,
                OP_LOGE(context, "mse_bits must be in [2, 4]"), return ge::GRAPH_FAILED);

    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t tokens = shape.GetDim(0);
    const uint32_t heads = shape.GetDim(1);
    const uint32_t cores =
        std::max(1U, std::min(static_cast<uint32_t>(platform.GetCoreNumAiv()), heads * ((tokens + 31) / 32)));
    std::vector<uint32_t> data(4 + 32 * 512);
    data[0] = tokens;
    data[1] = heads;
    data[2] = cores;
    data[3] = *bits;
    for (int i = 0; i < 32 * 512; ++i) {
        const int coordinate = (i % 256) / 2;
        const int dimension = i % 128;
        data[4 + i] = *bits == 3 ?
                          ((i / 512) * 256 + (i % 512) / 256 * 128 + (coordinate % 16) * 8 + coordinate / 16) * 2 :
                          ((i / 512) * 256 + (i % 256) / 128 * 128 + (dimension % 16) * 8 + dimension / 16) * 2;
    }
    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData->GetData());
    OP_CHECK_IF(rawTilingData->GetCapacity() < data.size() * sizeof(uint32_t),
                OP_LOGE(context, "raw tiling buffer is too small"), return ge::GRAPH_FAILED);
    std::memcpy(rawTilingData->GetData(), data.data(), data.size() * sizeof(uint32_t));
    rawTilingData->SetDataSize(data.size() * sizeof(uint32_t));
    context->SetBlockDim(cores);
    context->SetTilingKey(TILING_KEY_DEFAULT);
    auto* workspaceSizes = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaceSizes);
    workspaceSizes[0] = 0;
    return ge::GRAPH_SUCCESS;
}

struct TqQueryPackCompileInfo {};

static ge::graphStatus TilingParse(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TqQueryPack).Tiling(PackTiling).TilingParse<TqQueryPackCompileInfo>(TilingParse);
} // namespace optiling
