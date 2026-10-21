/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_ARCH92_QBSA_BUFFER_TYPES_H
#define QBSA_ARCH92_QBSA_BUFFER_TYPES_H

namespace fa_base_matmul {
enum class BufferType {
    L1 = 0,
    L0A = 1,
    L0B = 2,
    L0C = 3,
    UB = 4,
    GM = 5,
    C2 = 6,
};

enum class SyncType {
    NO_SYNC,
    INNER_CORE_SYNC,
    CROSS_CORE_SYNC_FORWARD,
    CROSS_CORE_SYNC_BOTH,
    CROSS_CORE_SYNC_BACKWARD,
};

constexpr uint32_t INVALID_CROSS_CORE_EVENT_ID = 16;
static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;

template <BufferType Type>
struct BufferInfo {
    // Cons 消费者，Prod 生产者
    __aicore__ const static constexpr HardEvent ConsWaitProdStatus()
    {
        if constexpr (Type == BufferType::L1) {
            return HardEvent::MTE2_MTE1;
        } else if constexpr (Type == BufferType::L0A) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::L0B) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::L0C) {
            return HardEvent::M_FIX;
        } else if constexpr (Type == BufferType::C2) {
            return HardEvent::MTE1_M;
        }
    }

    __aicore__ const static constexpr HardEvent ProdWaitConsStatus()
    {
        if constexpr (Type == BufferType::L1) {
            return HardEvent::MTE1_MTE2;
        } else if constexpr (Type == BufferType::L0A) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == BufferType::L0B) {
            return HardEvent::M_MTE1;
        } else if constexpr (Type == BufferType::L0C) {
            return HardEvent::FIX_M;
        } else if constexpr (Type == BufferType::C2) {
            return HardEvent::M_MTE1;
        }
    }

    __aicore__ const static constexpr TPosition GetTPosition()
    {
        if constexpr (Type == BufferType::L1) {
            return TPosition::A1;
        } else if constexpr (Type == BufferType::L0A) {
            return TPosition::A2;
        } else if constexpr (Type == BufferType::L0B) {
            return TPosition::B2;
        } else if constexpr (Type == BufferType::L0C) {
            return TPosition::CO1;
        } else if constexpr (Type == BufferType::UB) {
            return TPosition::VECIN;
        } else if constexpr (Type == BufferType::GM) {
            return TPosition::GM;
        } else if constexpr (Type == BufferType::C2) {
            return TPosition::C2;
        }
    }

    static constexpr HardEvent EventP2C =
        ConsWaitProdStatus(); // 生产者到消费者方向的HardEvent：消费者等生产者提供/生产者通知消费者已生成
    static constexpr HardEvent EventC2P =
        ProdWaitConsStatus(); // 消费者到生产者方向的HardEvent：生产者等消费者消耗/消费者通知生产者已消耗’
    static constexpr TPosition Position = GetTPosition();
};

// buffer绑定生产者、消费者关系
// L1 buffer的生产者为MTE2或者MTE3，消费者为MTE1
// L0A buffer的生产者为MTE1，消费者为M
// L0B buffer的生产者为MTE1，消费者为M
// L0C buffer的生产者为M，消费者为FIX
} // namespace fa_base_matmul

#endif // QBSA_ARCH92_QBSA_BUFFER_TYPES_H
