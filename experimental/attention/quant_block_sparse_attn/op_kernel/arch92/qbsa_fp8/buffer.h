/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file buffer.h
 * \brief同步管理
 */
#ifndef BUFFER_H
#define BUFFER_H
#include <type_traits>
#include "lib/matmul_intf.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "common.h"
using namespace AscendC;
#include "../common/qbsa_buffer_types.h"

namespace fa_base_matmul {
__BLOCK_LOCAL__ __inline__ uint32_t idCounterNum;
#define MAKE_ID ((++idCounterNum) % 11)

// 核间同步中，AIC(flagId 0-10)对应AIV0(flagId 0-10)，对应AIV1(flagId 16-26)
#define AIV0_AIV1_OFFSET 11

template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class Buffer {
    using TensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<uint8_t>, LocalTensor<uint8_t>>;

    template <typename T>
    using TargetTensorType = std::conditional_t<bufferType == BufferType::GM, GlobalTensor<T>, LocalTensor<T>>;

public:
    __aicore__ inline Buffer() {}
    __aicore__ inline Buffer(TensorType tensor, uint32_t size)
    {
        tensor_ = tensor;
        size_ = size;
        if constexpr (syncType == SyncType::CROSS_CORE_SYNC_FORWARD) {
            id0_ = MAKE_ID;
            id1_ = INVALID_CROSS_CORE_EVENT_ID;
        } else if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {
            id0_ = INVALID_CROSS_CORE_EVENT_ID;
            id1_ = MAKE_ID;
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
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                p2cEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>(); // 确保只能被调用一次
                c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
                SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
            }
        }
    }

    __aicore__ inline void UnInit()
    {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventP2C>(p2cEventId_); // 确保只能被调用一次
                GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
            }
        }
    }

    template <HardEvent EventType>
    __aicore__ inline void Wait()
    {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                    WaitFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_); // 消费者等待生产者完成生产
                } else {
                    WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_); // 生产者等待消费者完成消费
                }
            }
        }
    }

    template <HardEvent EventType>
    __aicore__ inline void Set()
    {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                    SetFlag<BufferInfo<bufferType>::EventP2C>(p2cEventId_); // 生产者通知消费者已完成生产
                } else {
                    SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_); // 消费者通知生产者已完成消费
                }
            }
        }
    }

    __aicore__ inline void SetEventID()
    {
        if ASCEND_IS_AIC {
            p2cEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>(); // 确保只能被调用一次
            c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
        }
    }

    template <HardEvent EventType>
    __aicore__ inline TEventID GetEventID()
    {
        if ASCEND_IS_AIC {
            if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                return p2cEventId_; // 生产者通知消费者已完成生产
            } else {
                return c2pEventId_; // 消费者通知生产者已完成消费
            }
        }
    }

    template <bool isReuse = false>
    __aicore__ inline void WaitCrossCore()
    {
        if constexpr (bufferType == BufferType::GM && syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id1_);
                if constexpr (CV_RATIO == 2) {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id1_ + AIV0_AIV1_OFFSET);
                }
            } else {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id0_);
            }
        } else if constexpr (bufferType == BufferType::UB || bufferType == BufferType::GM) {
            // AIC属于生产者，AIV属于消费者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_);
                if constexpr (CV_RATIO == 2) {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_ + AIV0_AIV1_OFFSET);
                }
            } else {
                if constexpr (isReuse) {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id0_);
                } else {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id0_);
                }
            }
        } else if constexpr (bufferType == BufferType::L1) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_);
                if constexpr (CV_RATIO == 2) {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_ + AIV0_AIV1_OFFSET);
                }
            } else {
                if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
                }
            }
        }
    }

    template <bool isReuse = false>
    __aicore__ inline void SetCrossCore()
    {
        if constexpr (bufferType == BufferType::GM && syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_);
                if constexpr (CV_RATIO == 2) {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_ + AIV0_AIV1_OFFSET);
                }
            } else {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
            }
        } else if constexpr (bufferType == BufferType::UB || bufferType == BufferType::GM) {
            // AIC属于生产者，AIV属于消费者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_);
                if constexpr (CV_RATIO == 2) {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_ + AIV0_AIV1_OFFSET);
                }
            } else {
                if constexpr (isReuse) {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
                } else {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id1_);
                }
            }
        } else if constexpr (bufferType == BufferType::L1) {
            // AIC属于消费者，AIV属于生产者，且一个AIC对应两个AIV
            if ASCEND_IS_AIC {
                if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id1_);
                    if constexpr (CV_RATIO == 2) {
                        CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id1_ + AIV0_AIV1_OFFSET);
                    }
                }
            } else {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id0_);
            }
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
    uint32_t id0_; // 用作正向同步：生产者通知消费者，或者消费者等待生产者；
    uint32_t id1_; // 用作反向同步：消费者通知生产者，或者生产者等待消费者；
};
} // namespace fa_base_matmul
#endif
