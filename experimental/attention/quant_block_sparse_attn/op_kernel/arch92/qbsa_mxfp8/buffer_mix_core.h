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
 * \file buffer_mix_core.h
 * \brief Synchronization management.
 */
#ifndef BUFFER_MIX_CORE_H
#define BUFFER_MIX_CORE_H
#include <type_traits>
#include "lib/matmul_intf.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
using namespace AscendC;
namespace fa_base_matmul {
__BLOCK_LOCAL__ __inline__ uint32_t idCounterNum;
#define MAKE_ID ((++idCounterNum) % 11)

// For cross-core synchronization, AIC flag IDs 0-10 map to AIV0 IDs 0-10 and AIV1 IDs 16-26.
#define AIV0_AIV1_OFFSET 16

enum class BufferType {
    L1 = 0,
    L0A = 1,
    L0B = 2,
    L0C = 3,
    UB = 4,
    GM = 5,
};

enum class SyncType {
    NO_SYNC,
    INNER_CORE_SYNC,
    CROSS_CORE_SYNC_FORWARD,
    CROSS_CORE_SYNC_BOTH,
};

constexpr uint32_t INVALID_CROSS_CORE_EVENT_ID = 16;
static constexpr uint64_t CROSS_CORE_SYNC_MODE = 4;

template <BufferType Type>
struct BufferInfo {
    // Cons: consumer; Prod: producer.
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
        }
    }

    static constexpr HardEvent EventP2C = ConsWaitProdStatus(); // Producer-to-consumer HardEvent: the consumer waits
                                                                // for data, or the producer signals that data is ready.
    static constexpr HardEvent EventC2P =
        ProdWaitConsStatus(); // Consumer-to-producer HardEvent: the producer waits for consumption, or the consumer
                              // signals that data has been consumed.
    static constexpr TPosition Position = GetTPosition();
};

// Bind a buffer to its producer and consumer.
// L1 buffer: MTE2 or MTE3 produces data; MTE1 consumes it.
// L0A buffer: MTE1 produces data; M consumes it.
// L0B buffer: MTE1 produces data; M consumes it.
// L0C buffer: M produces data; FIX consumes it.
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC, uint32_t cvRatio = 2U>
class Buffer {
    using TensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<uint8_t>, LocalTensor<uint8_t>>;

    template <typename T>
    using TargetTensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<T>, LocalTensor<T>>;

public:
    static_assert(cvRatio == 1U || cvRatio == 2U, "Only C:V=1:1 and C:V=1:2 are supported");
    __aicore__ inline Buffer() {}
    __aicore__ inline Buffer(TensorType tensor, uint32_t size)
    {
        tensor_ = tensor;
        size_ = size;
        if constexpr (syncType == SyncType::CROSS_CORE_SYNC_FORWARD) {
            id0_ = MAKE_ID;
            id1_ = INVALID_CROSS_CORE_EVENT_ID;
        } else if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
            id0_ = MAKE_ID;
            id1_ = MAKE_ID;
        } else {
            id0_ = INVALID_CROSS_CORE_EVENT_ID;
            id1_ = INVALID_CROSS_CORE_EVENT_ID;
        }
    }

    __aicore__ inline void Init()
    {
        if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
            p2cEventId_ =
                GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>(); // Ensure this is called only once.
            c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
            SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
        }
    }

    __aicore__ inline void UnInit()
    {
        if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
            WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
            GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventP2C>(
                p2cEventId_); // Ensure this is called only once.
            GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
        }
    }

    template <HardEvent EventType>
    __aicore__ inline void Wait()
    {
        if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
            if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                WaitFlag<BufferInfo<bufferType>::EventP2C>(
                    p2cEventId_); // The consumer waits for the producer to finish producing data.
            } else {
                WaitFlag<BufferInfo<bufferType>::EventC2P>(
                    c2pEventId_); // The producer waits for the consumer to finish consuming data.
            }
        }
    }

    template <HardEvent EventType>
    __aicore__ inline void Set()
    {
        if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
            if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                SetFlag<BufferInfo<bufferType>::EventP2C>(
                    p2cEventId_); // The producer signals that data production is complete.
            } else {
                SetFlag<BufferInfo<bufferType>::EventC2P>(
                    c2pEventId_); // The consumer signals that data consumption is complete.
            }
        }
    }

    __aicore__ inline void SetEventID()
    {
        p2cEventId_ =
            GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>(); // Ensure this is called only once.
        c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
    }

    template <HardEvent EventType>
    __aicore__ inline TEventID GetEventID()
    {
        if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
            return p2cEventId_; // The producer signals that data production is complete.
        } else {
            return c2pEventId_; // The consumer signals that data consumption is complete.
        }
    }

    template <typename T>
    __aicore__ inline TargetTensorType<T> GetTensor()
    {
        return tensor_.template ReinterpretCast<T>();
    }

    template <typename T>
    __aicore__ inline TargetTensorType<T> GetTensor(uint64_t startindex)
    {
        TargetTensorType<T> tmpTensor = tensor_.template ReinterpretCast<T>();
        return tmpTensor[startindex];
    }

private:
    TensorType tensor_;
    uint32_t size_;
    TEventID p2cEventId_;
    TEventID c2pEventId_;
    uint32_t
        id0_; // Forward synchronization: the producer notifies the consumer, or the consumer waits for the producer.
    uint32_t
        id1_; // Reverse synchronization: the consumer notifies the producer, or the producer waits for the consumer.
};
} // namespace fa_base_matmul
#endif
