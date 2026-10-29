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
 * \file dsa_dense_tiling.h
 * \brief
 */

#ifndef _DSA_DENSE_TILING_H_
#define _DSA_DENSE_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(DsaDenseBaseParams)
// base shape
TILING_DATA_FIELD_DEF(uint32_t, B);
TILING_DATA_FIELD_DEF(uint32_t, N);
TILING_DATA_FIELD_DEF(uint32_t, NI);
TILING_DATA_FIELD_DEF(uint32_t, D);
TILING_DATA_FIELD_DEF(uint32_t, DI);
TILING_DATA_FIELD_DEF(uint64_t, T1);
TILING_DATA_FIELD_DEF(uint64_t, T2);
TILING_DATA_FIELD_DEF(uint64_t, maxS1);
TILING_DATA_FIELD_DEF(uint64_t, maxS2);
TILING_DATA_FIELD_DEF(float, scale); // 1 / sqrt(D)
TILING_DATA_FIELD_DEF(uint32_t, cubeS1Tile);
TILING_DATA_FIELD_DEF(uint32_t, cubeS2Tile);
TILING_DATA_FIELD_DEF(uint32_t, vecS1Tile);
TILING_DATA_FIELD_DEF(uint32_t, vecS2Tile);
// workspace offset
TILING_DATA_FIELD_DEF(uint64_t, qkSpaceBytes);
TILING_DATA_FIELD_DEF(uint64_t, pSpaceBytes);
TILING_DATA_FIELD_DEF(uint64_t, ySpaceBytes);
TILING_DATA_FIELD_DEF(uint64_t, reluSpaceBytes);
TILING_DATA_FIELD_DEF(uint64_t, dwSpaceBytes);
TILING_DATA_FIELD_DEF(uint64_t, dkSpaceBytes);
TILING_DATA_FIELD_DEF(uint64_t, dqSpaceBytes);
// AIV num
TILING_DATA_FIELD_DEF(uint32_t, aivNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DsaDenseBaseParamsOp, DsaDenseBaseParams)

BEGIN_TILING_DATA_DEF(DsaDenseTilingData)
TILING_DATA_FIELD_DEF_STRUCT(DsaDenseBaseParams, baseParams);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, mm3TilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DsaDense, DsaDenseTilingData)
} // namespace optiling

#endif
