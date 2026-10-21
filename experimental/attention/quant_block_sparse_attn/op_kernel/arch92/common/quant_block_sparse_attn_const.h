/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_ARCH92_QUANT_BLOCK_SPARSE_ATTN_CONST_H
#define QBSA_ARCH92_QUANT_BLOCK_SPARSE_ATTN_CONST_H

constexpr static int64_t SPARSE_MODE_INT_DEFAULT = 2147483647;

constexpr uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;
constexpr uint32_t NEGATIVE_MIN_VALUE_FP16 = 0xFBFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP32 = 0x7F7FFFFF;
constexpr uint32_t POSITIVE_MAX_VALUE_FP16 = 0x7BFF;
constexpr int64_t FP8_QUANT_BLOCK_SIZE = 128;

constexpr int64_t attenMaskBN2GS1S2 = 0;
constexpr int64_t attenMaskBS1S2 = 1;

constexpr uint16_t SHIFT_NUM_6 = 6;
constexpr uint16_t ADD_NUM_63 = 63;

constexpr uint64_t SYNC_MODE = 4;
constexpr uint64_t KB_TO_BYTES = 1024;
constexpr uint64_t L0C_SIZE = 256;
constexpr uint64_t FLOAT_BYTES = 4;

#endif // QBSA_ARCH92_QUANT_BLOCK_SPARSE_ATTN_CONST_H
