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
 * \file buffers_policy.h
 * \brief Manage buffer memory and synchronization together.
 */
#ifndef BUFFERS_POLICY_H
#define BUFFERS_POLICY_H

#include "attn_buffer_manager.h"
#define NUM_2 2
#define NUM_3 3
#define NUM_4 4
// Q reuse and KV reuse.
// Allocate one buffer.
namespace fa_base_matmul {
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC,
          SyncMode syncMode = SyncMode::SET_WAIT_FLAG, IdSource idSource = IdSource::INTERNAL>
class BuffersPolicySingleBuffer {
public:
    __aicore__ inline void Init(BufferManager<bufferType>& bufferManager, uint32_t size, uint32_t id = 0U)
    {
        buffer_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);
        if constexpr (idSource == IdSource::INTERNAL) {
            buffer_.template Init<idSource>();
        } else if constexpr (idSource == IdSource::EXTERNAL) {
            buffer_.template Init<idSource>(id);
        }
    }

    __aicore__ inline void Uninit(BufferManager<bufferType>& bufferManager)
    {
        buffer_.template UnInit<idSource>();
        bufferManager.template FreeBuffer<syncType, syncMode>(buffer_);
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& Get()
    {
        return buffer_;
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& GetPre()
    {
        return Get();
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& GetReused()
    {
        return Get();
    }

private:
    Buffer<bufferType, syncType, syncMode> buffer_;
};

// Allocate two buffers for ping-pong rotation.
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC,
          SyncMode syncMode = SyncMode::SET_WAIT_FLAG, IdSource idSource = IdSource::INTERNAL>
class BuffersPolicyDB {
public:
    __aicore__ inline void Init(BufferManager<bufferType>& bufferManager, uint32_t size, uint32_t pingId = 0U,
                                uint32_t pongId = 0U)
    {
        ping_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);
        pong_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);

        if constexpr (idSource == IdSource::INTERNAL) {
            ping_.template Init<idSource>();
            pong_.template Init<idSource>();
        } else if constexpr (idSource == IdSource::EXTERNAL) {
            ping_.template Init<idSource>(pingId);
            pong_.template Init<idSource>(pongId);
        }
    }

    __aicore__ inline void Uninit(BufferManager<bufferType>& bufferManager)
    {
        ping_.template UnInit<idSource>();
        pong_.template UnInit<idSource>();

        bufferManager.template FreeBuffer<syncType, syncMode>(ping_);
        bufferManager.template FreeBuffer<syncType, syncMode>(pong_);
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& Get()
    {
        if (flag1_) { // 1
            flag1_ = 0;
            return ping_;
        } else { // 0
            flag1_ = 1;
            return pong_;
        }
    }

    // Use with Get: call Get first, then GetPre for Q reuse.
    __aicore__ inline Buffer<bufferType, syncType, syncMode>& GetPre()
    {
        if (flag1_) { // 0->1
            return pong_;
        } else { // 1->0
            return ping_;
        }
    }

    // Use with Get/GetPre: call Get first, GetPre second, and GetReused on the third reuse for KV.
    __aicore__ inline Buffer<bufferType, syncType, syncMode>& GetReused()
    {
        if (flag2_ == 0) {
            flag2_ = 1;
            return pong_;
        } else {
            flag2_ = 0;
            return ping_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& GetReused(bool isNextS2IdxNoChange)
    {
        if (isNextS2IdxNoChange) {
            if (flag2_ == 0) {
                return pong_;
            } else {
                return ping_;
            }
        } else {
            return GetReused();
        }
    }

private:
    Buffer<bufferType, syncType, syncMode> ping_;
    Buffer<bufferType, syncType, syncMode> pong_;
    uint32_t flag1_ = 0;
    uint32_t flag2_ = 0;
};

// Allocate three rotating buffers.
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC,
          SyncMode syncMode = SyncMode::SET_WAIT_FLAG, IdSource idSource = IdSource::INTERNAL>
class BuffersPolicy3buff {
public:
    __aicore__ inline void Init(BufferManager<bufferType>& bufferManager, uint32_t size, uint32_t aId = 0U,
                                uint32_t bId = 0U, uint32_t cId = 0U)
    {
        a_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);
        b_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);
        c_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);

        if constexpr (idSource == IdSource::INTERNAL) {
            a_.template Init<idSource>();
            b_.template Init<idSource>();
            c_.template Init<idSource>();
        } else if constexpr (idSource == IdSource::EXTERNAL) {
            a_.template Init<idSource>(aId);
            b_.template Init<idSource>(bId);
            c_.template Init<idSource>(cId);
        }
    }

    __aicore__ inline void Uninit(BufferManager<bufferType>& bufferManager)
    {
        a_.template UnInit<idSource>();
        b_.template UnInit<idSource>();
        c_.template UnInit<idSource>();

        bufferManager.template FreeBuffer<syncType, syncMode>(a_);
        bufferManager.template FreeBuffer<syncType, syncMode>(b_);
        bufferManager.template FreeBuffer<syncType, syncMode>(c_);
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& Get()
    {
        if (flag1_ == 0) {
            flag1_ = 1;
            return a_;
        } else if (flag1_ == 1) {
            flag1_ = NUM_2;
            return b_;
        } else {
            flag1_ = 0;
            return c_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& GetVec()
    { // mixcore architecture
        if (flag1_vec1_ == 0) {
            flag1_vec1_ = 1;
            return a_;
        } else if (flag1_vec1_ == 1) {
            flag1_vec1_ = NUM_2;
            return b_;
        } else {
            flag1_vec1_ = 0;
            return c_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& GetCube()
    { // mixcore architecture
        if (flag1_bmm2_ == 0) {
            flag1_bmm2_ = 1;
            return a_;
        } else if (flag1_bmm2_ == 1) {
            flag1_bmm2_ = NUM_2;
            return b_;
        } else {
            flag1_bmm2_ = 0;
            return c_;
        }
    }

    // Q reuse.
    __aicore__ inline Buffer<bufferType, syncType, syncMode>& GetPre()
    {
        if (flag1_ == 0) {
            return c_;
        } else if (flag1_ == 1) {
            return a_;
        } else {
            return b_;
        }
    }

    // KV reuse.
    __aicore__ inline Buffer<bufferType, syncType, syncMode>& GetReused()
    {
        if (flag2_ == 0) {
            flag2_ = 1;
            return a_;
        } else if (flag2_ == 1) {
            flag2_ = NUM_2;
            return b_;
        } else {
            flag2_ = 0;
            return c_;
        }
    }

private:
    Buffer<bufferType, syncType, syncMode> a_;
    Buffer<bufferType, syncType, syncMode> b_;
    Buffer<bufferType, syncType, syncMode> c_;
    uint32_t flag1_ = 0;
    uint32_t flag1_vec1_ = 0;
    uint32_t flag1_bmm2_ = 0;
    uint32_t flag2_ = 0;
};

// Allocate four buffers with KV reuse.
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC,
          SyncMode syncMode = SyncMode::SET_WAIT_FLAG, IdSource idSource = IdSource::INTERNAL>
class BuffersPolicy4buff {
public:
    __aicore__ inline void Init(BufferManager<bufferType>& bufferManager, uint32_t size, uint32_t aId = 0U,
                                uint32_t bId = 0U, uint32_t cId = 0U, uint32_t dId = 0U)
    {
        a_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);
        b_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);
        c_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);
        d_ = bufferManager.template AllocBuffer<syncType, syncMode>(size);

        if constexpr (idSource == IdSource::INTERNAL) {
            a_.template Init<idSource>();
            b_.template Init<idSource>();
            c_.template Init<idSource>();
            d_.template Init<idSource>();
        } else if constexpr (idSource == IdSource::EXTERNAL) {
            a_.template Init<idSource>(aId);
            b_.template Init<idSource>(bId);
            c_.template Init<idSource>(cId);
            d_.template Init<idSource>(dId);
        }
    }

    __aicore__ inline void Uninit(BufferManager<bufferType>& bufferManager)
    {
        a_.template UnInit<idSource>();
        b_.template UnInit<idSource>();
        c_.template UnInit<idSource>();
        d_.template UnInit<idSource>();

        bufferManager.template FreeBuffer<syncType, syncMode>(a_);
        bufferManager.template FreeBuffer<syncType, syncMode>(b_);
        bufferManager.template FreeBuffer<syncType, syncMode>(c_);
        bufferManager.template FreeBuffer<syncType, syncMode>(d_);
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& Get(uint32_t id)
    {
        uint32_t flag = id % 4;
        if (flag == 0) {
            return a_;
        } else if (flag == 1) {
            return b_;
        } else if (flag == 2) { // 2:c_
            return c_;
        } else {
            return d_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& Get()
    {
        auto& buffer = Get(head_);
        head_++;
        return buffer;
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& GetReused()
    {
        auto& buffer = Get(used_);
        used_ = (used_ - tail_ + 1) % (head_ - tail_) + tail_;
        return buffer;
    }

    __aicore__ inline Buffer<bufferType, syncType, syncMode>& GetFree()
    {
        if (tail_ == used_) {
            used_++;
        }
        auto& buffer = Get(tail_);
        tail_++;
        return buffer;
    }

private:
    Buffer<bufferType, syncType, syncMode> a_;
    Buffer<bufferType, syncType, syncMode> b_;
    Buffer<bufferType, syncType, syncMode> c_;
    Buffer<bufferType, syncType, syncMode> d_;
    uint32_t tail_ = 0; // Tail of the buffer queue currently in use.
    uint32_t head_ = 0; // Head of the buffer queue currently in use, plus 1.
    uint32_t used_ = 0; // Currently used buffer within the half-open interval from head to tail.
};

template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class Matrix2x2BufferPolicy { // 4buffer
    // Two-dimensional buffer management: row-major addresses and column-major usage.
    // MracBuffer:memory address with row first, alloc/use/free with column first
public:
    __aicore__ inline void Init(BufferManager<bufferType>& bufferManager, uint32_t size)
    {
        bufferM0k0_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM0k1_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM1k0_ = bufferManager.template AllocBuffer<syncType>(size);
        bufferM1k1_ = bufferManager.template AllocBuffer<syncType>(size);

        bufferM0k0_.template Init<IdSource::INTERNAL>();
        bufferM0k1_.template Init<IdSource::INTERNAL>();
        bufferM1k0_.template Init<IdSource::INTERNAL>();
        bufferM1k1_.template Init<IdSource::INTERNAL>();
    }

    __aicore__ inline void Uninit(BufferManager<bufferType>& bufferManager)
    {
        bufferM0k0_.template UnInit<IdSource::INTERNAL>();
        bufferM0k1_.template UnInit<IdSource::INTERNAL>();
        bufferM1k0_.template UnInit<IdSource::INTERNAL>();
        bufferM1k1_.template UnInit<IdSource::INTERNAL>();

        bufferManager.FreeBuffer(bufferM0k0_);
        bufferManager.FreeBuffer(bufferM0k1_);
        bufferManager.FreeBuffer(bufferM1k0_);
        bufferManager.FreeBuffer(bufferM1k1_);
    }

    __aicore__ inline void SetMExtent(int32_t mExtent)
    {
        aIdx_ = -1;
        amIdx_ = (amIdx_ + mSize_ - 1) % mSize_; // Toggle 0->1 and 1->0.
        akIdx_ = 0;

        uIdx_ = -1;
        umIdx_ = (umIdx_ + mSize_ - 1) % mSize_;
        ukIdx_ = 0;

        fIdx_ = -1;
        fmIdx_ = (fmIdx_ + mSize_ - 1) % mSize_;
        fkIdx_ = 0;

        mExtent_ = mExtent;
    }

    __aicore__ inline Buffer<bufferType, syncType>& AllocNext()
    {
        aIdx_++;
        return GetBuffer(aIdx_, amIdx_, akIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType>& ReuseNext()
    {
        uIdx_++;
        return GetBuffer(uIdx_, umIdx_, ukIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType>& FreeNext()
    {
        fIdx_++;
        return GetBuffer(fIdx_, fmIdx_, fkIdx_);
    }

    __aicore__ inline Buffer<bufferType, syncType>& PeekNextK()
    {                                            // Used during Alloc to select the next buffer along k.
        return PeekBuffer(amIdx_, (1 - akIdx_)); // Toggle k.
    }

private:
    __aicore__ inline Buffer<bufferType, syncType>& GetBuffer(int32_t xIdx, int32_t& mIdx, int32_t& kIdx)
    {
        // xIdx is the input index for the current alloc/use/free operation; output mIdx/kIdx identify the next buffer.
        mIdx = (mIdx + mExtent_ - 1) % mExtent_;
        kIdx = (xIdx / mExtent_) % kSize_;
        if (mIdx == 0 && kIdx == 0) {
            return bufferM0k0_;
        } else if (mIdx == 0 && kIdx == 1) {
            return bufferM0k1_;
        } else if (mIdx == 1 && kIdx == 0) {
            return bufferM1k0_;
        } else { // This branch condition is mIdx == 1 && kIdx == 1.
            return bufferM1k1_;
        }
    }

    __aicore__ inline Buffer<bufferType, syncType>& PeekBuffer(int32_t mIdx, int32_t kIdx)
    {
        // Access the buffer without advancing indices.
        if (mIdx == 0 && kIdx == 0) {
            return bufferM0k0_;
        } else if (mIdx == 0 && kIdx == 1) {
            return bufferM0k1_;
        } else if ((mIdx == 1) && (kIdx == 0)) {
            return bufferM1k0_;
        } else { // mIdx == 1 && kIdx == 1
            return bufferM1k1_;
        }
    }

    Buffer<bufferType, syncType> bufferM0k0_;
    Buffer<bufferType, syncType> bufferM0k1_;
    Buffer<bufferType, syncType> bufferM1k0_;
    Buffer<bufferType, syncType> bufferM1k1_;
    int32_t mSize_ = 2; // Total buffer count along m.
    int32_t kSize_ = 2; // Total buffer count along k.

    // Alloc
    int32_t aIdx_ = -1; // Current buffer allocation count.
    int32_t amIdx_ = 0; // m index of the buffer currently being allocated.
    int32_t akIdx_ = 0; // k index of the buffer currently being allocated.

    // Reuse
    int32_t uIdx_ = -1;
    int32_t umIdx_ = 0;
    int32_t ukIdx_ = 0;

    // Free
    int32_t fIdx_ = -1;
    int32_t fmIdx_ = 0;
    int32_t fkIdx_ = 0;

    int32_t mExtent_ = 0; // Actual m usage, either 1 or 2.
};
} // namespace fa_base_matmul
#endif
