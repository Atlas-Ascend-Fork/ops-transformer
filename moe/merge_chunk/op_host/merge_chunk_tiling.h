/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MERGE_CHUNK_HOST_TILING_H_
#define MERGE_CHUNK_HOST_TILING_H_

#include "register/tilingdata_base.h"
#include "../op_kernel/merge_chunk_limits.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MergeChunkTilingData)
TILING_DATA_FIELD_DEF(uint32_t, paths);
TILING_DATA_FIELD_DEF(uint32_t, chunks);
TILING_DATA_FIELD_DEF(uint32_t, experts);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MergeChunkSupport::SUPPORTED_CHUNKS, rowsPerCore);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MergeChunkSupport::SUPPORTED_CHUNKS, extraCores);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MergeChunkSupport::SUPPORTED_PATHS, hidden);
TILING_DATA_FIELD_DEF_ARR(uint32_t, MergeChunkSupport::SUPPORTED_PATHS, rowsPerTile);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MergeChunk, MergeChunkTilingData)
} // namespace optiling

#endif
