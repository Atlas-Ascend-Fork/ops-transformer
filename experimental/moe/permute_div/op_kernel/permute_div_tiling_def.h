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
 * \file permute_div_tiling_def.h
 * \brief PermuteDiv kernel 侧 TilingData 结构定义
 */
#ifndef OPS_OP_KERNEL_PERMUTE_DIV_TILING_DEF_H_
#define OPS_OP_KERNEL_PERMUTE_DIV_TILING_DEF_H_

#include "kernel_tiling/kernel_tiling.h"

struct PermuteDivKernelTilingData {
    uint32_t divisor;
    uint32_t coreNum;
    uint32_t indiceNum;
};

#endif // OPS_OP_KERNEL_PERMUTE_DIV_TILING_DEF_H_
