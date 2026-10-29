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
 * \file merge_chunk2_tiling_def.h
 * \brief MergeChunk2 kernel 侧 TilingData 结构定义
 *
 * 字段顺序/类型必须与 op_host/merge_chunk2_tiling.h 中
 * BEGIN_TILING_DATA_DEF 定义保持一致。
 */
#ifndef OPS_OP_KERNEL_MERGE_CHUNK2_TILING_DEF_H_
#define OPS_OP_KERNEL_MERGE_CHUNK2_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

struct MergeChunk2KernelTilingData {
    uint32_t usedCoreNum;
    uint32_t numExpert;
    uint32_t t1;
    uint32_t t2;
    uint32_t h1;
    uint32_t h2;
};

#endif // OPS_OP_KERNEL_MERGE_CHUNK2_TILING_DEF_H_
