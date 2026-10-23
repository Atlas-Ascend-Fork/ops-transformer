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
#include <cmath>
#include <cstdint>
#include <limits>

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "split_qkv_rmsnorm_mrope_tiling.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
namespace {
constexpr int64_t Q_HEADS = 16;
constexpr int64_t HEAD_DIM = 256;
constexpr int64_t ROPE_AXES = 3;
constexpr int64_t ROPE_DIM = 64;
constexpr size_t QKV_INDEX = 0;
constexpr size_t Q_WEIGHT_INDEX = 1;
constexpr size_t K_WEIGHT_INDEX = 2;
constexpr size_t COS_SIN_INDEX = 3;
constexpr size_t Q_BIAS_INDEX = 4;
constexpr size_t K_BIAS_INDEX = 5;

bool IsVector256(const gert::StorageShape* storageShape)
{
    return storageShape != nullptr && storageShape->GetStorageShape().GetDimNum() == 1 &&
           storageShape->GetStorageShape().GetDim(0) == HEAD_DIM;
}
} // namespace

static ge::graphStatus SplitQkvRmsnormMropeTilingFunc(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto* qkvShape = context->GetInputShape(QKV_INDEX);
    if (qkvShape == nullptr || context->GetAttrs() == nullptr) {
        OP_LOGE(context, "qkv shape or attributes are null");
        return ge::GRAPH_FAILED;
    }
    const auto& shape = qkvShape->GetStorageShape();
    if (shape.GetDimNum() != 2 || shape.GetDim(0) <= 0 ||
        static_cast<uint64_t>(shape.GetDim(0)) > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(context, "qkv must be rank 2 with a positive token count in the uint32 range");
        return ge::GRAPH_FAILED;
    }
    const auto* attrs = context->GetAttrs();
    const auto* qHeadsAttr = attrs->GetAttrPointer<int64_t>(0);
    const auto* kvHeadsAttr = attrs->GetAttrPointer<int64_t>(1);
    const auto* epsAttr = attrs->GetAttrPointer<float>(2);
    const auto* interleavedAttr = attrs->GetAttrPointer<bool>(3);
    const auto* hasGateAttr = attrs->GetAttrPointer<bool>(4);
    if (qHeadsAttr == nullptr || kvHeadsAttr == nullptr || epsAttr == nullptr || interleavedAttr == nullptr ||
        hasGateAttr == nullptr) {
        OP_LOGE(context, "required attributes are null");
        return ge::GRAPH_FAILED;
    }

    if (*qHeadsAttr != Q_HEADS || (*kvHeadsAttr != 2 && *kvHeadsAttr != 4)) {
        OP_LOGE(context, "num_q_heads must be 16 and num_kv_heads must be 2 or 4");
        return ge::GRAPH_FAILED;
    }
    if (!std::isfinite(*epsAttr) || *epsAttr <= 0.0F) {
        OP_LOGE(context, "eps must be finite and greater than zero");
        return ge::GRAPH_FAILED;
    }

    const int64_t expectedWidth = (*hasGateAttr ? 2 : 1) * Q_HEADS * HEAD_DIM + 2 * *kvHeadsAttr * HEAD_DIM;
    if (shape.GetDim(1) != expectedWidth || !IsVector256(context->GetInputShape(Q_WEIGHT_INDEX)) ||
        !IsVector256(context->GetInputShape(K_WEIGHT_INDEX))) {
        OP_LOGE(context, "qkv, q_weight, or k_weight shape is invalid");
        return ge::GRAPH_FAILED;
    }
    const auto* cosSinShape = context->GetInputShape(COS_SIN_INDEX);
    if (cosSinShape == nullptr || cosSinShape->GetStorageShape().GetDimNum() != 3 ||
        cosSinShape->GetStorageShape().GetDim(0) != ROPE_AXES ||
        cosSinShape->GetStorageShape().GetDim(1) != shape.GetDim(0) ||
        cosSinShape->GetStorageShape().GetDim(2) != ROPE_DIM) {
        OP_LOGE(context, "cos_sin must have shape [3, tokens, 64]");
        return ge::GRAPH_FAILED;
    }

    const bool hasQBias = context->GetOptionalInputDesc(Q_BIAS_INDEX) != nullptr;
    const bool hasKBias = context->GetOptionalInputDesc(K_BIAS_INDEX) != nullptr;
    if ((hasQBias && !IsVector256(context->GetOptionalInputShape(Q_BIAS_INDEX))) ||
        (hasKBias && !IsVector256(context->GetOptionalInputShape(K_BIAS_INDEX)))) {
        OP_LOGE(context, "q_bias and k_bias must have shape [256] when present");
        return ge::GRAPH_FAILED;
    }

    const uint32_t tokens = static_cast<uint32_t>(shape.GetDim(0));
    const uint32_t qHeads = static_cast<uint32_t>(*qHeadsAttr);
    const uint32_t kvHeads = static_cast<uint32_t>(*kvHeadsAttr);
    uint32_t maxCores = 1;
    if (context->GetPlatformInfo() != nullptr) {
        platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
        maxCores = platform.GetCoreNumAiv();
    }
    const bool shortPath = tokens == 1 && qHeads == 16 && !hasQBias && !hasKBias && *hasGateAttr && *interleavedAttr &&
                           (kvHeads == 2 || kvHeads == 4);
    const uint32_t tasks = shortPath ? (qHeads + kvHeads) / 2 : tokens;
    const uint32_t blocks = std::max(1U, std::min(tasks, maxCores));

    SplitQkvRmsnormMropeTilingData tiling;
    tiling.set_tokens(tokens);
    tiling.set_qHeads(qHeads);
    tiling.set_kvHeads(kvHeads);
    tiling.set_blocks(blocks);
    tiling.set_eps(*epsAttr);
    tiling.set_hasQBias(hasQBias);
    tiling.set_hasKBias(hasKBias);
    tiling.set_hasGate(*hasGateAttr);
    tiling.set_interleaved(*interleavedAttr);
    context->SetBlockDim(blocks);
    context->SetTilingKey(0);
    auto* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = 0;
    auto* rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr ||
        rawTilingData->GetCapacity() < tiling.GetDataSize()) {
        OP_LOGE(context, "raw tiling buffer is null or too small");
        return ge::GRAPH_FAILED;
    }
    tiling.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

struct SplitQkvRmsnormMropeCompileInfo {};

static ge::graphStatus TilingParseForSplitQkvRmsnormMrope(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SplitQkvRmsnormMrope)
    .Tiling(SplitQkvRmsnormMropeTilingFunc)
    .TilingParse<SplitQkvRmsnormMropeCompileInfo>(TilingParseForSplitQkvRmsnormMrope);
} // namespace optiling
