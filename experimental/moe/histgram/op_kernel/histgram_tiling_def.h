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
 * \file histgram_tiling_def.h
 * \brief Histgram kernel 侧 TilingData 结构定义
 *
 * 字段顺序/类型必须与 op_host/histgram_tiling.h 中 BEGIN_TILING_DATA_DEF 定义保持一致。
 */
#ifndef OPS_OP_KERNEL_HISTGRAM_TILING_DEF_H_
#define OPS_OP_KERNEL_HISTGRAM_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

namespace optiling {
struct HistgramTilingData {
    uint32_t usedCoreNum;
    uint32_t numExpert;
    uint32_t numToken;
};
} // namespace optiling
#endif // OPS_OP_KERNEL_HISTGRAM_TILING_DEF_H_
