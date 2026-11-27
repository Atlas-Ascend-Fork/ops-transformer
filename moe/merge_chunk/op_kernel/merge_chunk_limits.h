/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MERGE_CHUNK_LIMITS_H_
#define MERGE_CHUNK_LIMITS_H_

#include <stdint.h>

namespace MergeChunkSupport {
constexpr uint32_t SUPPORTED_PATHS = 2;
constexpr uint32_t SUPPORTED_CHUNKS = 2;
constexpr uint32_t TOKEN_RANK = 2;
constexpr uint32_t MAX_EXPERTS = 1024;
constexpr uint32_t ELEMENT_BYTES = 2;
constexpr uint32_t HIDDEN_ALIGNMENT = 16;
constexpr uint32_t COUNT_BUFFER_BYTES = MAX_EXPERTS * sizeof(int32_t);
constexpr uint32_t TOKEN_BUFFER_BYTES = 176 * 1024;
constexpr uint32_t TOTAL_BUFFER_BYTES = TOKEN_BUFFER_BYTES + 2 * COUNT_BUFFER_BYTES;
} // namespace MergeChunkSupport

#endif
