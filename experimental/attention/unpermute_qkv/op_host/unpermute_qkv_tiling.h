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
 * \file unpermute_qkv_tiling.h
 * \brief
 */

#ifndef UNPERMUTE_QKV_TILING_H
#define UNPERMUTE_QKV_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UnpermuteQkvTilingData)
TILING_DATA_FIELD_DEF(uint32_t, splitSize);
TILING_DATA_FIELD_DEF(uint32_t, sC);
TILING_DATA_FIELD_DEF(uint32_t, b);
TILING_DATA_FIELD_DEF(uint32_t, kvGroup);
TILING_DATA_FIELD_DEF(uint32_t, splitNK);
TILING_DATA_FIELD_DEF(uint32_t, nK);
TILING_DATA_FIELD_DEF(uint32_t, h);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UnpermuteQkv, UnpermuteQkvTilingData)
} // namespace optiling

#endif
