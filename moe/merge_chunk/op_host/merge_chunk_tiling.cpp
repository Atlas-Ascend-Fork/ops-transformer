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

#include "log/log.h"
#include "register/op_impl_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "merge_chunk_shape.h"
#include "merge_chunk_tiling.h"

namespace optiling {
namespace {
struct MergeChunkCompileInfo {
    uint32_t cores = 0;
    uint64_t ubBytes = 0;
};

ge::graphStatus ParseMergeChunk(gert::TilingParseContext* context)
{
    auto* info = context->GetCompiledInfo<MergeChunkCompileInfo>();
    auto* platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, info);
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    platform_ascendc::PlatformAscendC platform(platformInfo);
    info->cores = platform.GetCoreNumAiv();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, info->ubBytes);
    return ge::GRAPH_SUCCESS;
}

const gert::Shape* GetLogicalShape(const gert::TilingContext* context, size_t index, size_t item)
{
    const auto* shape = context->GetDynamicInputShape(index, item);
    return shape == nullptr ? nullptr : &shape->GetOriginShape();
}

ge::graphStatus ReadTilingLayout(gert::TilingContext* context, MergeChunkSupport::Layout& layout)
{
    int64_t paths = 0;
    int64_t chunks = 0;
    if (!MergeChunkSupport::ReadListDimensions(context, paths, chunks)) {
        return ge::GRAPH_FAILED;
    }
    const auto* error = MergeChunkSupport::ReadLayout(
        paths, chunks, [context](size_t i) { return GetLogicalShape(context, MergeChunkSupport::DATA_INDEX, i); },
        [context](size_t i) { return GetLogicalShape(context, MergeChunkSupport::COUNTS_INDEX, i); }, layout);
    if (error != nullptr) {
        OP_LOGE(context, "%s", error);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetSplit(gert::TilingContext* context, MergeChunkSupport::Layout& layout, MergeChunkTilingData& data)
{
    const auto* info = context->GetCompileInfo<MergeChunkCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, info);
    const auto longestChunk = *std::max_element(layout.chunkRows.begin(), layout.chunkRows.end());
    const uint32_t cores = std::min(info->cores, longestChunk);
    if (cores == 0 || info->ubBytes < MergeChunkSupport::TOTAL_BUFFER_BYTES) {
        OP_LOGE(context, "MergeChunk requires an AIV core and at least %u bytes of UB.",
                MergeChunkSupport::TOTAL_BUFFER_BYTES);
        return ge::GRAPH_FAILED;
    }
    std::array<uint32_t, MergeChunkSupport::SUPPORTED_CHUNKS> rowsPerCore{}, extraCores{};
    std::array<uint32_t, MergeChunkSupport::SUPPORTED_PATHS> rowsPerTile{};
    for (uint32_t chunk = 0; chunk < layout.chunks; ++chunk) {
        rowsPerCore[chunk] = layout.chunkRows[chunk] / cores;
        extraCores[chunk] = layout.chunkRows[chunk] % cores;
    }
    for (uint32_t path = 0; path < layout.paths; ++path) {
        rowsPerTile[path] =
            MergeChunkSupport::TOKEN_BUFFER_BYTES / (layout.hidden[path] * MergeChunkSupport::ELEMENT_BYTES);
    }
    data.set_paths(layout.paths);
    data.set_chunks(layout.chunks);
    data.set_experts(layout.experts);
    data.set_rowsPerCore(rowsPerCore.data());
    data.set_extraCores(extraCores.data());
    data.set_hidden(layout.hidden.data());
    data.set_rowsPerTile(rowsPerTile.data());
    context->SetBlockDim(cores);
    context->SetTilingKey(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingMergeChunk(gert::TilingContext* context)
{
    MergeChunkSupport::Layout layout;
    MergeChunkTilingData data;
    if (ReadTilingLayout(context, layout) != ge::GRAPH_SUCCESS ||
        SetSplit(context, layout, data) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto* raw = context->GetRawTilingData();
    auto* workspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, raw);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspace);
    if (raw->GetData() == nullptr || raw->GetCapacity() < data.GetDataSize()) {
        return ge::GRAPH_FAILED;
    }
    data.SaveToBuffer(raw->GetData(), raw->GetCapacity());
    raw->SetDataSize(data.GetDataSize());
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
} // namespace
IMPL_OP_OPTILING(MergeChunk).Tiling(TilingMergeChunk).TilingParse<MergeChunkCompileInfo>(ParseMergeChunk);
} // namespace optiling
