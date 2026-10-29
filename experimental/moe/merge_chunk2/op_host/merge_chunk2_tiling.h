/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file merge_chunk2_tiling.h
 * \brief MergeChunk2 TilingData 定义
 */
#ifndef OPS_OP_HOST_MERGE_CHUNK2_TILING_H_
#define OPS_OP_HOST_MERGE_CHUNK2_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(MergeChunk2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, numExpert);
TILING_DATA_FIELD_DEF(uint32_t, t1);
TILING_DATA_FIELD_DEF(uint32_t, t2);
TILING_DATA_FIELD_DEF(uint32_t, h1);
TILING_DATA_FIELD_DEF(uint32_t, h2);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MergeChunk2, MergeChunk2TilingData)

struct MergeChunk2CompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling
#endif // OPS_OP_HOST_MERGE_CHUNK2_TILING_H_
