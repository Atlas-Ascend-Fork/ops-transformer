/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_SCHEDULER_INT8_H_
#define QUANT_FLASH_ATTN_SCHEDULER_INT8_H_
#include "kernel_operator.h"
#include "quant_flash_attn_tiling_data_int8.h"
#include "quant_flash_attn_workload_int8.h"

namespace BaseApi::Int8GqaTnd {
// AICPU has already resolved actual lengths and task order. The kernel only decodes its device metadata.
class SequenceTaskScheduler {
    using Sequence = QfaSequenceDescriptor;

public:
    __aicore__ inline void Init(const QuantFlashAttnTilingDataInt8& t, GM_ADDR metadata)
    {
        t_ = t;
        metadata_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadata));
        const uint32_t mode = metadata_.GetValue(QFA_METADATA_TASK_MODE);
        paired_ = mode == QFA_METADATA_PAIRED;
        mapped_ = mode == QFA_METADATA_MAPPED;
        if (mapped_)
            tasks_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadata) +
                                   QfaMetadataMapOffset(t_.batchSize, metadata_.GetValue(QFA_METADATA_CORE_CAPACITY)));
        // Cache up to 32 immutable descriptors; larger batches use the same GM reader.
        descriptorsCached_ = false;
        if (t_.batchSize <= DESCRIPTOR_CAPACITY) {
            for (uint32_t b = 0U; b < t_.batchSize; ++b)
                descriptors_[b] = ReadSequence(b);
            descriptorsCached_ = true;
        }
        seq_ = ReadSequence(0U);
    }

    __aicore__ inline bool Decode(uint64_t task, uint32_t& batch, uint32_t& head, uint32_t& mStart, uint32_t& sq,
                                  uint32_t& sk, uint32_t& qStart, uint32_t& kStart)
    {
        if (mapped_)
            task = tasks_.GetValue(task);
        while (task >= batchBase_ + seq_.tasks) {
            batchBase_ += seq_.tasks;
            if (++batch_ >= t_.batchSize)
                return false;
            seq_ = ReadSequence(batch_);
        }
        const uint32_t local = task - batchBase_;
        batch = batch_;
        uint32_t tile;
        if (paired_) {
            const uint32_t half = seq_.tiles / QFA_MIRRORED_TASK_PAIR_SIZE;
            head = (local / QFA_MIRRORED_TASK_PAIR_SIZE) / half;
            const uint32_t pair = (local / QFA_MIRRORED_TASK_PAIR_SIZE) % half;
            tile = (local & 1U) ? seq_.tiles - 1U - pair : pair;
        } else {
            head = local / seq_.tiles;
            tile = local % seq_.tiles;
        }
        if (t_.sequsedFlags != 0U && t_.layoutQ == QFA_LAYOUT_BNSD) {
            const uint32_t perHead = (static_cast<uint64_t>(seq_.sq) + t_.blockM - 1U) / t_.blockM;
            mStart = (tile / perHead) * seq_.sq + (tile % perHead) * t_.blockM;
        } else {
            mStart = tile * t_.blockM;
        }
        sq = seq_.sq;
        sk = seq_.sk;
        qStart = seq_.qStart;
        kStart = seq_.kStart;
        return true;
    }

    __aicore__ inline Sequence ReadSequence(uint32_t b)
    {
        if (descriptorsCached_)
            return descriptors_[b];
        const uint32_t offset = QfaMetadataSequenceOffset(b);
        return {
            metadata_.GetValue(offset + QFA_METADATA_Q_START),    metadata_.GetValue(offset + QFA_METADATA_K_START),
            metadata_.GetValue(offset + QFA_METADATA_Q_LENGTH),   metadata_.GetValue(offset + QFA_METADATA_K_LENGTH),
            metadata_.GetValue(offset + QFA_METADATA_TILES),      metadata_.GetValue(offset + QFA_METADATA_TASKS),
            metadata_.GetValue(offset + QFA_METADATA_PHYSICAL_Q), metadata_.GetValue(offset + QFA_METADATA_PHYSICAL_K)};
    }

private:
    static constexpr uint32_t DESCRIPTOR_CAPACITY = 32U;
    Sequence descriptors_[DESCRIPTOR_CAPACITY];
    bool descriptorsCached_ = false;
    // Scalar aggregate copies use 64-bit accesses; keep the private copy aligned without changing the tiling ABI.
    alignas(uint64_t) QuantFlashAttnTilingDataInt8 t_{};
    AscendC::GlobalTensor<uint32_t> metadata_;
    AscendC::GlobalTensor<uint32_t> tasks_;
    Sequence seq_{};
    uint64_t batchBase_ = 0U;
    uint32_t batch_ = 0U;
    bool paired_ = false, mapped_ = false;
};

} // namespace BaseApi::Int8GqaTnd

#endif
