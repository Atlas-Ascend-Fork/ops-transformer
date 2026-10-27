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

#include <cstdint>

// Host/device agreement for the 910B decode tile and constant-offset layout.
namespace tqv2 {
constexpr int TILE_M = 128;
constexpr int PACK_ROWS = 32;
constexpr int W = 64; // Keys unpacked in one AIV microtile.
constexpr int WORDS = W * 16;
constexpr int E = W * 128;
constexpr int RAW_BYTES = W * 96;
constexpr int QUERY_OFFSETS = 4 * 512;
constexpr int OFFSET_COUNT = 5 * WORDS + QUERY_OFFSETS + W;
constexpr uint32_t MAX_INTERLEAVED_METADATA_HEADS = 16;
constexpr uint32_t PADDED_METADATA_STRIDE = 16;

constexpr uint32_t GetMetadataStride(uint32_t keyHeads)
{
    return keyHeads <= MAX_INTERLEAVED_METADATA_HEADS ? keyHeads : PADDED_METADATA_STRIDE;
}
} // namespace tqv2
