/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_MXFP8_CONTRACT_H_
#define QBSA_MXFP8_CONTRACT_H_

#include <cstdint>

namespace optiling::qbsa_mxfp8 {
constexpr uint32_t QBSA_MXFP8_S2_BASE_SIZE = 512U;
constexpr uint32_t QBSA_MXFP8_MAX_PA_BLOCK_SIZE = 1024U;
constexpr uint32_t QBSA_MXFP8_SPARSE_BLOCK_SIZE_128 = 128U;
constexpr uint32_t QBSA_MXFP8_SPARSE_BLOCK_SIZE_64 = 64U;
constexpr uint32_t QBSA_MXFP8_SCALE_GROUP_SIZE = 64U;
constexpr uint32_t QBSA_MXFP8_SCALE_LAST_DIM = 2U;
constexpr uint32_t QBSA_MXFP8_PER_TOKEN_GROUP_MODE = 6U;
constexpr uint32_t QBSA_MXFP8_PER_CHANNEL_GROUP_MODE = 8U;

inline bool IsSupportedHeadDim(uint32_t dSize)
{
    return dSize == 64U || dSize == 128U || dSize == 256U;
}
} // namespace optiling::qbsa_mxfp8

#endif // QBSA_MXFP8_CONTRACT_H_
