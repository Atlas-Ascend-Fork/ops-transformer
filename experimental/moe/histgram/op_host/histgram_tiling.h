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
 * \file histgram_tiling.h
 * \brief Histgram TilingData 定义
 *
 * 输入为一维 token 序列, 统计每个 expert 出现的次数(numExpert 由属性给出)。
 */
#ifndef OPS_OP_HOST_HISTGRAM_TILING_H_
#define OPS_OP_HOST_HISTGRAM_TILING_H_

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(HistgramTilingData)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum); // 使用的向量核数
TILING_DATA_FIELD_DEF(uint32_t, numExpert);   // expert 个数
TILING_DATA_FIELD_DEF(uint32_t, numToken);    // token 个数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Histgram, HistgramTilingData)

struct HistgramCompileInfo {
    int64_t coreNum = 0;
    uint64_t ubSize = 0;
};

} // namespace optiling
#endif // OPS_OP_HOST_HISTGRAM_TILING_H_
