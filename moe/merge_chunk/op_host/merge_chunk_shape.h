/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MERGE_CHUNK_SHAPE_H_
#define MERGE_CHUNK_SHAPE_H_

#include <array>
#include <cstddef>
#include <limits>

#include "../op_kernel/merge_chunk_limits.h"

namespace MergeChunkSupport {
constexpr size_t DATA_INDEX = 0;
constexpr size_t COUNTS_INDEX = 1;
constexpr size_t PATHS_ATTR_INDEX = 0;
constexpr size_t CHUNKS_ATTR_INDEX = 1;

struct Layout {
    uint32_t paths = 0;
    uint32_t chunks = 0;
    uint32_t experts = 0;
    uint32_t totalRows = 0;
    std::array<uint32_t, SUPPORTED_CHUNKS> chunkRows{};
    std::array<uint32_t, SUPPORTED_PATHS> hidden{};
};

inline bool ValidListDimensions(int64_t paths, int64_t chunks)
{
    return paths == SUPPORTED_PATHS && chunks == SUPPORTED_CHUNKS;
}

template <typename DataShape, typename CountShape>
const char* ReadLayout(int64_t paths, int64_t chunks, DataShape dataShape, CountShape countShape, Layout& layout)
{
    if (!ValidListDimensions(paths, chunks)) {
        return "num_paths and num_chunks must both equal 2";
    }
    layout.paths = static_cast<uint32_t>(paths);
    layout.chunks = static_cast<uint32_t>(chunks);
    int64_t totalRows = 0;
    for (uint32_t chunk = 0; chunk < layout.chunks; ++chunk) {
        const auto* count = countShape(chunk);
        const auto* data = dataShape(chunk);
        if (count == nullptr || count->GetDimNum() != 1 || data == nullptr || data->GetDimNum() != TOKEN_RANK) {
            return "Each count tensor must have rank 1 and each data tensor must have rank 2";
        }
        const auto experts = count->GetDim(0);
        if (experts <= 0 || experts > MAX_EXPERTS || (chunk != 0 && experts != layout.experts)) {
            return "Expert counts must be positive, bounded and consistent across chunks";
        }
        layout.experts = static_cast<uint32_t>(experts);
        const auto rows = data->GetDim(0);
        if (rows <= 0 || rows > std::numeric_limits<int32_t>::max() - totalRows) {
            return "Chunk row counts must be positive and their sum must fit in int32";
        }
        layout.chunkRows[chunk] = static_cast<uint32_t>(rows);
        totalRows += rows;
    }
    layout.totalRows = static_cast<uint32_t>(totalRows);
    for (uint32_t path = 0; path < layout.paths; ++path) {
        const uint32_t start = path * layout.chunks;
        const auto* first = dataShape(start);
        if (first == nullptr || first->GetDimNum() != TOKEN_RANK) {
            return "Missing rank-2 tensor for a data path";
        }
        const auto hidden = first->GetDim(1);
        if (hidden <= 0 || hidden % HIDDEN_ALIGNMENT != 0 || hidden > TOKEN_BUFFER_BYTES / ELEMENT_BYTES ||
            hidden > std::numeric_limits<int32_t>::max() / totalRows) {
            return "Feature width must be 16-aligned, fit in UB and keep element offsets within int32";
        }
        layout.hidden[path] = static_cast<uint32_t>(hidden);
        for (uint32_t chunk = 0; chunk < layout.chunks; ++chunk) {
            const auto* entry = dataShape(start + chunk);
            if (entry == nullptr || entry->GetDimNum() != TOKEN_RANK || entry->GetDim(0) != layout.chunkRows[chunk] ||
                entry->GetDim(1) != hidden) {
                return "Token rows must agree within a chunk and feature widths must agree within a path";
            }
        }
    }
    if (dataShape(layout.paths * layout.chunks) != nullptr || countShape(layout.chunks) != nullptr) {
        return "Input tensor list lengths do not match num_paths and num_chunks";
    }
    return nullptr;
}

// Inference can refine an unknown dimension from its matching tensor. Runtime
// validation in ReadLayout requires every individual input dimension to be concrete.
constexpr int64_t UNKNOWN_DIM = -1;
struct InferredLayout {
    int64_t totalRows = UNKNOWN_DIM;
    int64_t experts = UNKNOWN_DIM;
    std::array<int64_t, SUPPORTED_PATHS> hidden{};
};

inline bool MergeDimension(int64_t value, int64_t& merged)
{
    if (value == UNKNOWN_DIM) {
        return true;
    }
    if (value <= 0 || (merged != UNKNOWN_DIM && merged != value)) {
        return false;
    }
    merged = value;
    return true;
}

template <typename DataShape, typename CountShape>
const char* InferLayout(DataShape dataShape, CountShape countShape, InferredLayout& result)
{
    constexpr int64_t maxElements = std::numeric_limits<int32_t>::max();
    std::array<int64_t, SUPPORTED_CHUNKS> rows;
    rows.fill(UNKNOWN_DIM);
    result.hidden.fill(UNKNOWN_DIM);
    result.experts = UNKNOWN_DIM;
    for (size_t chunk = 0; chunk < SUPPORTED_CHUNKS; ++chunk) {
        const auto* counts = countShape(chunk);
        if (counts == nullptr || counts->GetDimNum() != 1 || !MergeDimension(counts->GetDim(0), result.experts) ||
            result.experts > MAX_EXPERTS) {
            return "Invalid expert count shape";
        }
        for (size_t path = 0; path < SUPPORTED_PATHS; ++path) {
            const auto* data = dataShape(path * SUPPORTED_CHUNKS + chunk);
            if (data == nullptr || data->GetDimNum() != TOKEN_RANK || !MergeDimension(data->GetDim(0), rows[chunk]) ||
                rows[chunk] > maxElements || !MergeDimension(data->GetDim(1), result.hidden[path])) {
                return "Incompatible token tensor shapes";
            }
        }
    }
    if (dataShape(SUPPORTED_PATHS * SUPPORTED_CHUNKS) != nullptr || countShape(SUPPORTED_CHUNKS) != nullptr) {
        return "Unexpected input tensor list length";
    }
    result.totalRows = rows[0] == UNKNOWN_DIM || rows[1] == UNKNOWN_DIM ? UNKNOWN_DIM : rows[0] + rows[1];
    if (result.totalRows > maxElements) {
        return "Combined row count exceeds int32";
    }
    for (auto width : result.hidden) {
        if (width != UNKNOWN_DIM && (width % HIDDEN_ALIGNMENT != 0 || width > TOKEN_BUFFER_BYTES / ELEMENT_BYTES ||
                                     (result.totalRows != UNKNOWN_DIM && width > maxElements / result.totalRows))) {
            return "Invalid feature width or combined tensor size";
        }
    }
    return nullptr;
}

template <typename Context>
bool ReadListDimensions(Context* context, int64_t& paths, int64_t& chunks)
{
    if (context == nullptr || context->GetAttrs() == nullptr) {
        return false;
    }
    const auto* attrs = context->GetAttrs();
    const auto* pathAttr = attrs->GetInt(PATHS_ATTR_INDEX);
    const auto* chunkAttr = attrs->GetInt(CHUNKS_ATTR_INDEX);
    if (pathAttr == nullptr || chunkAttr == nullptr) {
        return false;
    }
    paths = *pathAttr;
    chunks = *chunkAttr;
    return ValidListDimensions(paths, chunks);
}
} // namespace MergeChunkSupport
#endif
