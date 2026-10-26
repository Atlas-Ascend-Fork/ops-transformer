/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file buffer.h
 * \brief Synchronization management.
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
using namespace AscendC;
namespace fa_base_matmul {
__BLOCK_LOCAL__ __inline__ uint32_t idCounterNum;
#define MAKE_ID ((++idCounterNum) % 11)

__aicore__ inline void ResetIdCounter()
{
    idCounterNum = 0;
}

// For cross-core synchronization, AIC flag IDs 0-10 map to AIV0 IDs 0-10 and AIV1 IDs 16-26.
#define AIV0_AIV1_OFFSET 16

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

enum class SyncMode {
    SET_WAIT_FLAG,
    LOCK_UNLOCK,
};

enum class IdSource {
    INTERNAL, // Allocate the ID internally.
    EXTERNAL, // Pass the ID externally (specified by the user).
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
        } else if constexpr (Type == BufferType::C2) {
            return HardEvent::MTE1_M;
        } else if constexpr (Type == BufferType::GM) {
            return HardEvent::MTE2_S;
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
        } else if constexpr (Type == BufferType::GM) {
            return HardEvent::S_MTE2;
        }
    }

    __aicore__ const static constexpr pipe_t GetProdPipe()
    {
        if constexpr (Type == BufferType::L1) {
            return PIPE_MTE2;
        } else if constexpr (Type == BufferType::L0A) {
            return PIPE_MTE1;
        } else if constexpr (Type == BufferType::L0B) {
            return PIPE_MTE1;
        } else if constexpr (Type == BufferType::L0C) {
            return PIPE_M;
        }
    }

    __aicore__ const static constexpr pipe_t GetConsPipe()
    {
        if constexpr (Type == BufferType::L1) {
            return PIPE_MTE1;
        } else if constexpr (Type == BufferType::L0A) {
            return PIPE_M;
        } else if constexpr (Type == BufferType::L0B) {
            return PIPE_M;
        } else if constexpr (Type == BufferType::L0C) {
            return PIPE_FIX;
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

    static constexpr HardEvent EventP2C = ConsWaitProdStatus(); // Producer-to-consumer HardEvent: the consumer waits
                                                                // for data, or the producer signals that data is ready.
    static constexpr HardEvent EventC2P =
        ProdWaitConsStatus(); // Consumer-to-producer HardEvent: the producer waits for consumption, or the consumer
                              // signals that data has been consumed.
    static constexpr pipe_t ProdPipe = GetProdPipe(); // Producer PIPE used by Mutex.
    static constexpr pipe_t ConsPipe = GetConsPipe(); // Consumer PIPE used by Mutex.
    static constexpr TPosition Position = GetTPosition();
};

// Flag-based buffers keep an empty base instead of carrying an unused mutex ID.
template <SyncMode syncMode>
struct BufferMutexState {};

#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
template <>
struct BufferMutexState<SyncMode::LOCK_UNLOCK> {
protected:
    MutexID mutexId_;
};
#endif

// Bind a buffer to its producer and consumer.
// L1 buffer: MTE2 or MTE3 produces data; MTE1 consumes it.
// L0A buffer: MTE1 produces data; M consumes it.
// L0B buffer: MTE1 produces data; M consumes it.
// L0C buffer: M produces data; FIX consumes it.
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC,
          SyncMode syncMode = SyncMode::SET_WAIT_FLAG>
class Buffer : private BufferMutexState<syncMode> {
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

    template <IdSource idSource = IdSource::INTERNAL>
    __aicore__ inline void Init()
    {
        static_assert(idSource == IdSource::INTERNAL, "idSource should IdSource::INTERNAL.");
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC && idSource == IdSource::INTERNAL) {
                if constexpr (syncMode == SyncMode::SET_WAIT_FLAG) {
                    p2cEventId_ =
                        GetTPipePtr()
                            ->AllocEventID<BufferInfo<bufferType>::EventP2C>(); // Ensure this is called only once.
                    c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
                    SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                } else if constexpr (syncMode == SyncMode::LOCK_UNLOCK) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                    this->mutexId_ = AllocMutexID();
#endif
                }
            }
        }
    }

    template <IdSource idSource = IdSource::EXTERNAL>
    __aicore__ inline void Init(uint32_t id)
    {
        static_assert(idSource == IdSource::EXTERNAL, "idSource should IdSource::EXTERNAL.");
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC && idSource == IdSource::EXTERNAL) {
                if constexpr (syncMode == SyncMode::SET_WAIT_FLAG) {
                    // For static tensors, the user specifies EventId.
                    p2cEventId_ = id;
                    c2pEventId_ = id;
                    SetFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                } else if constexpr (syncMode == SyncMode::LOCK_UNLOCK) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                    this->mutexId_ = id;
#endif
                }
            }
        }
    }

    template <IdSource idSource = IdSource::INTERNAL>
    __aicore__ inline void UnInit()
    {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (syncMode == SyncMode::SET_WAIT_FLAG) {
                    WaitFlag<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                    if constexpr (idSource == IdSource::INTERNAL) {
                        GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventP2C>(
                            p2cEventId_); // Ensure this is called only once.
                        GetTPipePtr()->ReleaseEventID<BufferInfo<bufferType>::EventC2P>(c2pEventId_);
                    }
                } else if constexpr (syncMode == SyncMode::LOCK_UNLOCK) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                    if constexpr (idSource == IdSource::INTERNAL) {
                        ReleaseMutexID(this->mutexId_);
                    }
#endif
                }
            }
        }
    }

    template <pipe_t pipe>
    __aicore__ inline void Lock()
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC && syncMode == SyncMode::LOCK_UNLOCK) {
                Mutex::Lock<pipe>(this->mutexId_);
            }
        }
#endif
    }

    template <pipe_t pipe>
    __aicore__ inline void Unlock()
    {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC && syncMode == SyncMode::LOCK_UNLOCK) {
                Mutex::Unlock<pipe>(this->mutexId_);
            }
        }
#endif
    }

    template <HardEvent EventType>
    __aicore__ inline void Wait()
    {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (syncMode == SyncMode::SET_WAIT_FLAG) {
                    if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                        WaitFlag<BufferInfo<bufferType>::EventP2C>(
                            p2cEventId_); // The consumer waits for the producer to finish producing data.
                    } else {
                        WaitFlag<BufferInfo<bufferType>::EventC2P>(
                            c2pEventId_); // The producer waits for the consumer to finish consuming data.
                    }
                } else if constexpr (syncMode == SyncMode::LOCK_UNLOCK) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                    if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                        Mutex::Lock<BufferInfo<bufferType>::ConsPipe>(this->mutexId_); // Lock the consumer.
                    } else {
                        Mutex::Lock<BufferInfo<bufferType>::ProdPipe>(this->mutexId_); // Lock the producer.
                    }
#endif
                }
            }
        }
    }

    template <HardEvent EventType>
    __aicore__ inline void Set()
    {
        if ASCEND_IS_AIC {
            if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
                if constexpr (syncMode == SyncMode::SET_WAIT_FLAG) {
                    if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                        SetFlag<BufferInfo<bufferType>::EventP2C>(
                            p2cEventId_); // The producer signals that data production is complete.
                    } else {
                        SetFlag<BufferInfo<bufferType>::EventC2P>(
                            c2pEventId_); // The consumer signals that data consumption is complete.
                    }
                } else if constexpr (syncMode == SyncMode::LOCK_UNLOCK) {
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
                    if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                        Mutex::Unlock<BufferInfo<bufferType>::ProdPipe>(this->mutexId_); // Unlock the producer.
                    } else {
                        Mutex::Unlock<BufferInfo<bufferType>::ConsPipe>(this->mutexId_); // Unlock the consumer.
                    }
#endif
                }
            }
        }
    }

    template <IdSource idSource = IdSource::INTERNAL>
    __aicore__ inline void SetEventID()
    {
        static_assert(idSource == IdSource::INTERNAL, "idSource should IdSource::INTERNAL.");
        if ASCEND_IS_AIC {
            if constexpr (idSource == IdSource::INTERNAL && syncMode == SyncMode::SET_WAIT_FLAG) {
                p2cEventId_ =
                    GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventP2C>(); // Ensure this is called only once.
                c2pEventId_ = GetTPipePtr()->AllocEventID<BufferInfo<bufferType>::EventC2P>();
            }
        }
    }

    template <HardEvent EventType>
    __aicore__ inline TEventID GetEventID()
    {
        if ASCEND_IS_AIC {
            if constexpr (EventType == BufferInfo<bufferType>::EventP2C) {
                return p2cEventId_; // The producer signals that data production is complete.
            } else {
                return c2pEventId_; // The consumer signals that data consumption is complete.
            }
        }
    }

    __aicore__ inline void SetCrossCoreID(uint32_t id0, uint32_t id1)
    {
        id0_ = id0;
        id1_ = id1;
    }

    template <bool isReuse = false>
    __aicore__ inline void WaitCrossCore()
    {
        if constexpr (bufferType == BufferType::GM && syncType == SyncType::CROSS_CORE_SYNC_BACKWARD) {
            // AIC is the consumer and AIV is the producer; one AIC corresponds to two AIV cores.
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id1_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id1_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE2>(id0_);
            }
        } else if constexpr (bufferType == BufferType::UB || bufferType == BufferType::GM) {
            // AIC is the producer and AIV is the consumer; one AIC corresponds to two AIV cores.
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id1_ + AIV0_AIV1_OFFSET);
            } else {
                if constexpr (isReuse) {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id0_);
                } else {
                    CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id0_);
                }
            }
        } else if constexpr (bufferType == BufferType::L1) {
            // AIC is the consumer and AIV is the producer; one AIC corresponds to two AIV cores.
            if ASCEND_IS_AIC {
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_);
                CrossCoreWaitFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id0_ + AIV0_AIV1_OFFSET);
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
            // AIC is the consumer and AIV is the producer; one AIC corresponds to two AIV cores.
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_);
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
            }
        } else if constexpr (bufferType == BufferType::UB || bufferType == BufferType::GM) {
            // AIC is the producer and AIV is the consumer; one AIC corresponds to two AIV cores.
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_);
                CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_FIX>(id0_ + AIV0_AIV1_OFFSET);
            } else {
                if constexpr (isReuse) {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE3>(id1_);
                } else {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_V>(id1_);
                }
            }
        } else if constexpr (bufferType == BufferType::L1) {
            // AIC is the consumer and AIV is the producer; one AIC corresponds to two AIV cores.
            if ASCEND_IS_AIC {
                if constexpr (syncType == SyncType::CROSS_CORE_SYNC_BOTH) {
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id1_);
                    CrossCoreSetFlag<CROSS_CORE_SYNC_MODE, PIPE_MTE1>(id1_ + AIV0_AIV1_OFFSET);
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
    uint32_t
        id0_; // Forward synchronization: the producer notifies the consumer, or the consumer waits for the producer.
    uint32_t
        id1_; // Reverse synchronization: the consumer notifies the producer, or the producer waits for the consumer.
};
} // namespace fa_base_matmul
#endif
