/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_METADATA_PAGED_H_
#define QUANT_FLASH_ATTN_METADATA_PAGED_H_

#include <cstdint>
#include "quant_flash_attn_tiling_data_int8.h"

namespace optiling {

constexpr uint32_t QFA_PAGED_METADATA_MAGIC = 0x51464136U; // ASCII QFA6
constexpr uint32_t QFA_PAGED_METADATA_VERSION = 6U;
constexpr uint32_t QFA_PAGED_METADATA_HEADER_WORDS = 32U;
constexpr uint32_t QFA_PAGED_METADATA_WORDS = 4096U;
constexpr uint32_t QFA_PAGED_FAST_PAGE_SIZE = 128U;

enum QfaPagedPlanKind : uint32_t {
    QFA_PAGED_PLAN_GENERAL = 0U,
    // Slot 1 is reserved to preserve the V6 metadata layout.
    QFA_PAGED_PLAN_PAGE_128_ALLOW_PAIRED = 2U,
    QFA_PAGED_PLAN_COUNT = 3U
};

enum QfaPageQCheck : uint32_t {
    QFA_PAGE_ANY_Q,
    QFA_PAGE_Q_GT1_ONLY
};

struct QfaPagedMetaHeaderV6 {
    uint32_t magic;
    uint32_t version;
    uint32_t quantMode;
    uint32_t maskMode;
    uint32_t kvLayout;
    uint32_t batchSize;
    uint32_t qHeadNum;
    uint32_t kvHeadNum;
    uint32_t headDim;
    // Zero denotes an omitted caller bound.
    uint32_t maxSeqQ;
    uint32_t maxSeqKv;
    uint32_t planWords;
    // Equivalent planner inputs may share a slot. PLAN_COUNT denotes an unavailable plan.
    uint32_t planIndex[QFA_PAGED_PLAN_COUNT];
    uint32_t actualMaxQ;
    uint32_t actualMaxKv;
    uint32_t reserved[15];
};

struct QfaPagedMetaPlan {
    uint32_t coreCap;
    uint32_t partialCap;
    QfaHostPlanData plan;
};

struct QfaPagedMetaDataV6 {
    QfaPagedMetaHeaderV6 header;
    QfaPagedMetaPlan plans[QFA_PAGED_PLAN_COUNT];
};

static_assert(sizeof(QfaPagedMetaHeaderV6) == QFA_PAGED_METADATA_HEADER_WORDS * sizeof(uint32_t),
              "Unexpected paged metadata header ABI");
static_assert(sizeof(QfaPagedMetaDataV6) <= QFA_PAGED_METADATA_WORDS * sizeof(uint32_t),
              "Paged metadata exceeds its fixed capacity");

} // namespace optiling

#endif // QUANT_FLASH_ATTN_METADATA_PAGED_H_
