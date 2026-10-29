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
 * \file block_indexer_rearrange_tiling.h
 * \brief
 */

#ifndef BLOCK_INDEXER_REARRANGE_TILING_H
#define BLOCK_INDEXER_REARRANGE_TILING_H

#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include <tiling/platform/platform_ascendc.h>
#include "register/op_def_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {

ge::graphStatus BlockIndexerRearrangeTilingFunc(gert::TilingContext* context);

BEGIN_TILING_DATA_DEF(BlockIndexerRearrangeBaseParams)
TILING_DATA_FIELD_DEF(int64_t, totalSeqlen);
TILING_DATA_FIELD_DEF(int64_t, headNum);
TILING_DATA_FIELD_DEF(int64_t, blockCount);
TILING_DATA_FIELD_DEF(int64_t, blockSize);
TILING_DATA_FIELD_DEF(int64_t, causal);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BlockIndexerRearrangeBaseParamsOp, BlockIndexerRearrangeBaseParams)

BEGIN_TILING_DATA_DEF(BlockIndexerRearrangeCoreParams)
TILING_DATA_FIELD_DEF(int32_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int32_t, formerCoreNum);
TILING_DATA_FIELD_DEF(int32_t, formerCoreProcessNums);
TILING_DATA_FIELD_DEF(int32_t, remainCoreProcessNums);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(BlockIndexerRearrangeCoreParamsOp, BlockIndexerRearrangeCoreParams)

BEGIN_TILING_DATA_DEF(BlockIndexerRearrangeTilingData)
TILING_DATA_FIELD_DEF_STRUCT(BlockIndexerRearrangeBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(BlockIndexerRearrangeCoreParams, coreParams);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BlockIndexerRearrange, BlockIndexerRearrangeTilingData)

} // namespace optiling
#endif // BLOCK_INDEXER_REARRANGE_TILING_H
