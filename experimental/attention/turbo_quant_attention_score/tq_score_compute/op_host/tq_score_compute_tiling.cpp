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
#include <limits>
#include <vector>

#include "../op_kernel/tq_score_compute_config.h"
#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "tq_score_compute_tiling.h"

namespace optiling {
namespace {
constexpr uint64_t TILING_KEY_DEFAULT = 0;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t CACHE_BLOCK_SIZE = 128;
constexpr int64_t PACKED_UINT8S_PER_DIM = 16;
constexpr int64_t MIN_MSE_BITS = 2;
constexpr int64_t MAX_MSE_BITS = 4;
} // namespace

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("TqScoreCompute", "tiling context is null"), return ge::GRAPH_FAILED);
    auto shape = [&](int index) -> const gert::Shape& { return context->GetInputShape(index)->GetStorageShape(); };
    for (int index = 0; index < 8; ++index) {
        if (context->GetInputShape(index) == nullptr || context->GetInputDesc(index) == nullptr) {
            OP_LOGE(context, "input shape or desc %d is null", index);
            return ge::GRAPH_FAILED;
        }
    }
    const auto& query = shape(0);
    const auto& queryQjl = shape(1);
    const auto& keyIdx = shape(2);
    const auto& keyQjl = shape(3);
    const auto& keyNorm = shape(4);
    const auto& keyGamma = shape(5);
    const auto& blockTable = shape(6);
    const auto& sequenceLengths = shape(7);
    if (query.GetDimNum() != 3 || queryQjl != query || query.GetDim(2) != HEAD_DIM || query.GetDim(1) <= 0 ||
        keyIdx.GetDimNum() != 4 || keyIdx.GetDim(1) != CACHE_BLOCK_SIZE || keyIdx.GetDim(2) <= 0 ||
        blockTable.GetDimNum() != 2 || sequenceLengths.GetDimNum() != 1 ||
        sequenceLengths.GetDim(0) != blockTable.GetDim(0) || blockTable.GetDim(0) <= 0) {
        OP_LOGE(context, "query, cache, block_table, or seq_lens shape is invalid");
        return ge::GRAPH_FAILED;
    }
    if (keyIdx.GetDim(0) < 0 || static_cast<uint64_t>(keyIdx.GetDim(0)) > std::numeric_limits<uint32_t>::max()) {
        OP_LOGE(context, "physical cache block count exceeds the tiling data range");
        return ge::GRAPH_FAILED;
    }
    const auto* attrs = context->GetAttrs();
    if (attrs == nullptr) {
        OP_LOGE(context, "attributes are null");
        return ge::GRAPH_FAILED;
    }
    const auto* bitsPointer = attrs->GetAttrPointer<int64_t>(0);
    const auto* lengthPointer = attrs->GetAttrPointer<int64_t>(1);
    if (bitsPointer == nullptr || lengthPointer == nullptr || *bitsPointer < MIN_MSE_BITS ||
        *bitsPointer > MAX_MSE_BITS) {
        OP_LOGE(context, "mse_bits or max_kv_len is invalid");
        return ge::GRAPH_FAILED;
    }
    const int64_t bits = *bitsPointer;
    const int64_t length = *lengthPointer < 0 ? blockTable.GetDim(1) * CACHE_BLOCK_SIZE : *lengthPointer;
    if (length < 0 || length > blockTable.GetDim(1) * CACHE_BLOCK_SIZE || query.GetDim(1) % keyIdx.GetDim(2) != 0 ||
        keyIdx.GetDim(3) != PACKED_UINT8S_PER_DIM * bits) {
        OP_LOGE(context, "max_kv_len, GQA ratio, or key_cache_idx width is invalid");
        return ge::GRAPH_FAILED;
    }
    if (keyQjl.GetDimNum() != 4 || keyQjl.GetDim(0) != keyIdx.GetDim(0) || keyQjl.GetDim(1) != CACHE_BLOCK_SIZE ||
        keyQjl.GetDim(2) != keyIdx.GetDim(2) || keyQjl.GetDim(3) != PACKED_UINT8S_PER_DIM || keyNorm.GetDimNum() != 3 ||
        keyGamma != keyNorm || keyNorm.GetDim(0) != keyIdx.GetDim(0) || keyNorm.GetDim(1) != CACHE_BLOCK_SIZE ||
        keyNorm.GetDim(2) != keyIdx.GetDim(2)) {
        OP_LOGE(context, "key cache qjl/norm/gamma shape is invalid");
        return ge::GRAPH_FAILED;
    }
    const auto metadataType = context->GetInputDesc(4)->GetDataType();
    if ((metadataType != ge::DT_BF16 && metadataType != ge::DT_FLOAT16) ||
        context->GetInputDesc(5)->GetDataType() != metadataType) {
        OP_LOGE(context, "key cache norm/gamma dtype must match and be BF16 or FP16");
        return ge::GRAPH_FAILED;
    }

    const uint32_t tokens = query.GetDim(0);
    const uint32_t batch = blockTable.GetDim(0);
    const uint32_t queryHeads = query.GetDim(1);
    const uint32_t keyHeads = keyIdx.GetDim(2);
    const uint32_t pages = blockTable.GetDim(1);
    const bool prefill = batch == 1 && tokens > 1;
    if (!prefill && tokens != batch) {
        OP_LOGE(context, "decode tokens must equal batch; prefill only supports batch 1");
        return ge::GRAPH_FAILED;
    }
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    const uint32_t cores = std::max(
        1U,
        std::min(static_cast<uint32_t>(platform.GetCoreNumAic()),
                 static_cast<uint32_t>(batch * keyHeads * ((queryHeads / keyHeads + 3) / 4) * ((length + 127) / 128))));
    matmul_tiling::MatmulApiTiling matmul(platform);
    matmul.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16,
                    false);
    matmul.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BF16,
                    true);
    matmul.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    matmul.SetShape(prefill ? tqv2::TILE_M : 4, 128, 512);
    matmul.SetOrgShape(prefill ? tqv2::TILE_M : 4, 128, 512);
    matmul.SetBufferSpace(-1, -1, -1);
    TCubeTiling cube;
    if (matmul.GetTiling(cube) != 0) {
        OP_LOGE(context, "failed to generate matmul tiling");
        return ge::GRAPH_FAILED;
    }
    const uint64_t aBytes = prefill ? 0 : static_cast<uint64_t>(cores) * 2 * 4 * 512 * 2;
    const uint64_t bBytes = static_cast<uint64_t>(cores) * 2 * 128 * 512 * 2;
    uint32_t header[16] = {tokens,
                           batch,
                           queryHeads,
                           keyHeads,
                           pages,
                           static_cast<uint32_t>(length),
                           cores,
                           static_cast<uint32_t>(bits),
                           static_cast<uint32_t>(64 + cube.GetDataSize()),
                           static_cast<uint32_t>(prefill),
                           static_cast<uint32_t>(metadataType == ge::DT_FLOAT16),
                           static_cast<uint32_t>(aBytes),
                           static_cast<uint32_t>(aBytes >> 32),
                           static_cast<uint32_t>(keyIdx.GetDim(0)),
                           0,
                           0};
    std::vector<uint32_t> offsets(tqv2::OFFSET_COUNT, 0);
    const uint32_t stride = ((16 * bits + 31) / 32) * 32;
    for (int byte = 0; byte < 4; ++byte) {
        for (int word = 0; word < tqv2::WORDS; ++word) {
            offsets[byte * tqv2::WORDS + word] = ((word / 16) * stride + (word % 16) * bits + byte) * 4;
        }
    }
    for (int word = 0; word < tqv2::WORDS; ++word) {
        offsets[4 * tqv2::WORDS + word] = ((word / 16) * 32 + word % 16) * 4;
    }
    for (int index = 0; index < tqv2::QUERY_OFFSETS; ++index) {
        const int dimension = index % 128;
        offsets[5 * tqv2::WORDS + index] =
            ((index / 512) * 512 + (index % 256) / 128 * 128 + (dimension % 16) * 8 + dimension / 16) * 2;
        if (bits == 3) {
            const int coordinate = (index % 256) / 2;
            offsets[5 * tqv2::WORDS + index] =
                ((index / 512) * 512 + (index % 512) / 256 * 128 + (coordinate % 16) * 8 + coordinate / 16) * 2;
        }
    }
    // For at most 16 KV heads, Chunk() loads an interleaved [key, head] block.
    // For more heads, its 2-byte DataCopyPad blocks occupy 32-byte UB rows, so
    // adjacent keys remain 16 metadata elements apart after the Cast to float.
    const uint32_t metadataStride = tqv2::GetMetadataStride(keyHeads);
    for (int index = 0; index < tqv2::W; ++index) {
        offsets[5 * tqv2::WORDS + tqv2::QUERY_OFFSETS + index] = index * metadataStride * sizeof(float);
    }
    auto* rawTilingData = context->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context, rawTilingData->GetData());
    const uint32_t tilingBytes = header[8] + offsets.size() * sizeof(uint32_t);
    if (rawTilingData->GetCapacity() < tilingBytes) {
        OP_LOGE(context, "raw tiling buffer is too small");
        return ge::GRAPH_FAILED;
    }
    std::memcpy(rawTilingData->GetData(), header, sizeof(header));
    cube.SaveToBuffer(static_cast<uint8_t*>(rawTilingData->GetData()) + sizeof(header), cube.GetDataSize());
    std::memcpy(static_cast<uint8_t*>(rawTilingData->GetData()) + header[8], offsets.data(),
                offsets.size() * sizeof(uint32_t));
    rawTilingData->SetDataSize(tilingBytes);
    context->SetBlockDim(cores);
    context->SetTilingKey(TILING_KEY_DEFAULT);
    auto* workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes == nullptr) {
        OP_LOGE(context, "workspace sizes are null");
        return ge::GRAPH_FAILED;
    }
    workspaceSizes[0] = platform.GetLibApiWorkSpaceSize() + aBytes + bBytes;
    return ge::GRAPH_SUCCESS;
}

struct TqScoreComputeCompileInfo {};

static ge::graphStatus TilingParse(gert::TilingParseContext*)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(TqScoreCompute).Tiling(TilingFunc).TilingParse<TqScoreComputeCompileInfo>(TilingParse);
} // namespace optiling
