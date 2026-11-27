/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_METADATA_INT8_H_
#define QUANT_FLASH_ATTN_METADATA_INT8_H_
#include <cstdint>
#ifdef QFA_WORKLOAD_INLINE
#define QFA_METADATA_INLINE QFA_WORKLOAD_INLINE
#else
#define QFA_METADATA_INLINE inline
#endif
namespace BaseApi::Int8GqaTnd {
// The host selects the static pipeline; AICPU partitions actual batch/KV-head working sets
// into L2 sections and stores each section's per-core task ranges. No Split-K reduction is needed.
constexpr uint32_t QFA_METADATA_HEADER = 32U;
constexpr uint32_t QFA_METADATA_MAGIC = 0x51464133U; // ASCII QFA3
constexpr uint32_t QFA_METADATA_VERSION = 7U;
constexpr uint32_t QFA_METADATA_MAX_CORES = 128U;
constexpr uint32_t QFA_METADATA_Q_START = 0U;
constexpr uint32_t QFA_METADATA_K_START = 1U;
constexpr uint32_t QFA_METADATA_Q_LENGTH = 2U;
constexpr uint32_t QFA_METADATA_K_LENGTH = 3U;
constexpr uint32_t QFA_METADATA_SEQUENCE_WORDS = 8U;
constexpr uint32_t QFA_METADATA_TILES = 4U;
constexpr uint32_t QFA_METADATA_TASKS = 5U;
constexpr uint32_t QFA_METADATA_PHYSICAL_Q = 6U;
constexpr uint32_t QFA_METADATA_PHYSICAL_K = 7U;
constexpr uint32_t QFA_METADATA_SEQUENCE_FLAGS = 25U;
constexpr uint32_t QFA_METADATA_CORE_RANGE_WORDS = 2U;
constexpr uint32_t QFA_METADATA_MAP_CAPACITY = 512U;
constexpr uint32_t QFA_METADATA_TASK_MODE = 16U;
constexpr uint32_t QFA_METADATA_TASK_STEP = 17U;
constexpr uint32_t QFA_METADATA_USED_CORES = 18U;
constexpr uint32_t QFA_METADATA_BLOCK_M = 19U;
constexpr uint32_t QFA_METADATA_BLOCK_N = 20U;
constexpr uint32_t QFA_METADATA_CONFIG = 21U;
constexpr uint32_t QFA_METADATA_CORE_CAPACITY = 22U;
constexpr uint32_t QFA_METADATA_SECTION_NUM = 23U;
constexpr uint32_t QFA_METADATA_L2_BYTES = 24U;
constexpr uint32_t QFA_METADATA_NORMAL = 0U;
constexpr uint32_t QFA_METADATA_PAIRED = 1U;
constexpr uint32_t QFA_METADATA_MAPPED = 2U;
QFA_METADATA_INLINE uint64_t QfaMetadataSize(uint32_t batch, uint32_t cores, uint32_t headsKv)
{
    return QFA_METADATA_HEADER + static_cast<uint64_t>(QFA_METADATA_SEQUENCE_WORDS) * batch +
           static_cast<uint64_t>(QFA_METADATA_CORE_RANGE_WORDS) * cores * batch * headsKv + QFA_METADATA_MAP_CAPACITY;
}
QFA_METADATA_INLINE uint32_t QfaMetadataSequenceOffset(uint32_t batchIndex)
{
    return QFA_METADATA_HEADER + QFA_METADATA_SEQUENCE_WORDS * batchIndex;
}
QFA_METADATA_INLINE uint32_t QfaMetadataRangeOffset(uint32_t batch)
{
    return QfaMetadataSequenceOffset(batch);
}
QFA_METADATA_INLINE uint32_t QfaMetadataMapOffset(uint32_t batch, uint32_t cores)
{
    return QfaMetadataRangeOffset(batch) + QFA_METADATA_CORE_RANGE_WORDS * cores;
}
// Section zero retains the original range/map offsets, including the single-section fast path.
QFA_METADATA_INLINE uint64_t QfaMetadataSectionOffset(uint32_t batch, uint32_t cores, uint32_t section)
{
    return section == 0U ? QfaMetadataRangeOffset(batch) :
                           static_cast<uint64_t>(QfaMetadataMapOffset(batch, cores)) + QFA_METADATA_MAP_CAPACITY +
                               static_cast<uint64_t>(section - 1U) * QFA_METADATA_CORE_RANGE_WORDS * cores;
}
} // namespace BaseApi::Int8GqaTnd
#undef QFA_METADATA_INLINE
#endif
