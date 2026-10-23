/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SPLIT_QKV_RMSNORM_MROPE_TILING_DATA_H_
#define SPLIT_QKV_RMSNORM_MROPE_TILING_DATA_H_

#include <cstdint>

namespace optiling {
struct SplitQkvRmsnormMropeTilingData {
    uint32_t tokens;
    uint32_t qHeads;
    uint32_t kvHeads;
    uint32_t blocks;
    float eps;
    uint32_t hasQBias;
    uint32_t hasKBias;
    uint32_t hasGate;
    uint32_t interleaved;
};
} // namespace optiling

#endif // SPLIT_QKV_RMSNORM_MROPE_TILING_DATA_H_
