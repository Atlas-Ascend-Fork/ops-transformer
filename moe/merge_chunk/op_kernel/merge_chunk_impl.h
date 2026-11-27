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
 * \file merge_chunk_impl.h
 * \brief MergeChunk two-path, two-chunk kernel
 */
#ifndef MERGE_CHUNK_IMPL_H_
#define MERGE_CHUNK_IMPL_H_

#include "kernel_operator.h"
#include "merge_chunk_limits.h"

using namespace AscendC;

namespace MergeChunkSupport {

template <typename T, typename Tiling>
class KernelMergeChunk {
public:
    __aicore__ KernelMergeChunk(const Tiling& tiling, TPipe& pipe)
        : tiling_(tiling),
          pipe_(pipe)
    {}

    __aicore__ void Process(GM_ADDR in1c1, GM_ADDR in1c2, GM_ADDR in2c1, GM_ADDR in2c2, GM_ADDR glc1, GM_ADDR glc2,
                            GM_ADDR out1, GM_ADDR out2, GM_ADDR gl)
    {
        if (g_coreType == AIC) {
            return;
        }

        Init(in1c1, in1c2, in2c1, in2c2, glc1, glc2, out1, out2, gl);
        LoadGroupList();
        if (!ValidCounts(glc1_, 0) || !ValidCounts(glc2_, 1)) {
            return;
        }
        ProcessToken<true>(in1c1Gm_, 0, out1Gm_);
        ProcessToken<false>(in1c2Gm_, 0, out1Gm_);
        ProcessToken<true>(in2c1Gm_, 1, out2Gm_);
        ProcessToken<false>(in2c2Gm_, 1, out2Gm_);
        ProcessGroupList();
    }

    __aicore__ void LoadGroupList()
    {
        const int maxExpert = COUNT_BUFFER_BYTES / sizeof(int32_t);
        glc1_ = ub_.GetWithOffset<int32_t>(maxExpert, TOKEN_BUFFER_BYTES);
        glc2_ = ub_.GetWithOffset<int32_t>(maxExpert, TOKEN_BUFFER_BYTES + COUNT_BUFFER_BYTES);
        DataCopyPad(glc1_, glc1Gm_,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(tiling_.experts * sizeof(int32_t)),
                     /*srcStride*/ 0, /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(glc2_, glc2Gm_,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(tiling_.experts * sizeof(int32_t)),
                     /*srcStride*/ 0, /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    }

    __aicore__ bool ValidCounts(LocalTensor<int32_t>& counts, uint32_t chunk)
    {
        uint32_t remaining = tiling_.rowsPerCore[chunk] * GetBlockNum() + tiling_.extraCores[chunk];
        for (uint32_t expert = 0; expert < tiling_.experts; ++expert) {
            const int32_t count = counts.GetValue(expert);
            if (count < 0 || static_cast<uint32_t>(count) > remaining) {
                return false;
            }
            remaining -= static_cast<uint32_t>(count);
        }
        return remaining == 0;
    }

    __aicore__ void ProcessGroupList()
    {
        if (blockIdx_ == 0) {
            pipe_barrier(PIPE_ALL);
            Add(glc1_, glc1_, glc2_, tiling_.experts);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            DataCopyPad(glGm_, glc1_,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(tiling_.experts * sizeof(int32_t)),
                         /*srcStride*/ 0, /*dstStride*/ 0, 0});
        }
    }

    template <bool isFirstChunk>
    __aicore__ void ProcessToken(const GlobalTensor<T>& input, uint32_t path, GlobalTensor<T>& out)
    {
        const uint32_t chunk = isFirstChunk ? 0 : 1;
        const uint32_t extra = tiling_.extraCores[chunk];
        const uint32_t base = tiling_.rowsPerCore[chunk];
        curToken_ = base + (blockIdx_ < extra ? 1 : 0);
        curTokenOffset_ = blockIdx_ * base + (blockIdx_ < extra ? blockIdx_ : extra);
        if (curToken_ == 0) {
            return;
        }

        const int hidden = tiling_.hidden[path];
        const int maxToken = tiling_.rowsPerTile[path];
        input_ = ub_.GetWithOffset<T>(maxToken * hidden, 0);

        int loop = (curToken_ + maxToken - 1) / maxToken;
        int tailToken = curToken_ - (loop - 1) * maxToken;

        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        for (int i = 0; i < loop; ++i) {
            int curChunkToken = (i == loop - 1) ? tailToken : maxToken;
            int curChunkOffset = i * maxToken;
            ProcessChunk<isFirstChunk>(input, curChunkToken, curChunkOffset, hidden, out);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    template <bool isFirstChunk>
    __aicore__ void ProcessChunk(const GlobalTensor<T>& input, int curChunkToken, int curChunkOffset, int hidden,
                                 GlobalTensor<T>& out)
    {
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

        int chunkStart = curTokenOffset_ + curChunkOffset;
        int chunkEnd = chunkStart + curChunkToken;
        DataCopyPad(input_, input[chunkStart * hidden],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(curChunkToken * hidden * sizeof(T)),
                     /*srcStride*/ 0, /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);

        int sumTokenC1 = 0;
        int sumTokenC2 = 0;
        int sumToken = 0;
        for (uint32_t curExpert = 0; curExpert < tiling_.experts; ++curExpert) {
            int tokenCurExpertC1 = glc1_.GetValue(curExpert);
            int tokenCurExpertC2 = glc2_.GetValue(curExpert);

            int curExpertStartPos;
            int curExpertEndPos;
            if (isFirstChunk) {
                curExpertStartPos = sumTokenC1;
                curExpertEndPos = curExpertStartPos + tokenCurExpertC1;
            } else {
                curExpertStartPos = sumTokenC2;
                curExpertEndPos = curExpertStartPos + tokenCurExpertC2;
            }

            if (curExpertEndPos <= chunkStart) { // 当前专家未到达本chunk
                // update intermediate status
                sumTokenC1 += tokenCurExpertC1;
                sumTokenC2 += tokenCurExpertC2;
                sumToken += (tokenCurExpertC1 + tokenCurExpertC2);
                continue;
            } else if (curExpertStartPos >= chunkEnd) { // 当前专家已经路过了本chunk
                break;
            }

            if (curExpertEndPos == curExpertStartPos) { // 当前专家token为0
                // update intermediate status
                sumTokenC1 += tokenCurExpertC1;
                sumTokenC2 += tokenCurExpertC2;
                sumToken += (tokenCurExpertC1 + tokenCurExpertC2);
                continue;
            }

            int validStart = curExpertStartPos >= chunkStart ? curExpertStartPos : chunkStart;
            int validEnd = curExpertEndPos <= chunkEnd ? curExpertEndPos : chunkEnd;
            int validTokenNum = validEnd - validStart;
            int startOffsetInExpert = chunkStart <= curExpertStartPos ? 0 : chunkStart - curExpertStartPos;
            int startOffsetInChunk = chunkStart <= curExpertStartPos ? curExpertStartPos - chunkStart : 0;

            int outOffset;
            if (isFirstChunk) {
                outOffset = sumToken + startOffsetInExpert;
            } else {
                outOffset = sumToken + tokenCurExpertC1 + startOffsetInExpert;
            }
            DataCopyPad(out[outOffset * hidden], input_[startOffsetInChunk * hidden],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(validTokenNum * hidden * sizeof(T)),
                         /*srcStride*/ 0, /*dstStride*/ 0, 0});

            // update intermediate status
            sumTokenC1 += tokenCurExpertC1;
            sumTokenC2 += tokenCurExpertC2;
            sumToken += (tokenCurExpertC1 + tokenCurExpertC2);
        }
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    }

    __aicore__ void Init(GM_ADDR in1c1, GM_ADDR in1c2, GM_ADDR in2c1, GM_ADDR in2c2, GM_ADDR glc1, GM_ADDR glc2,
                         GM_ADDR out1, GM_ADDR out2, GM_ADDR gl)
    {
        // get block id
        blockIdx_ = GetBlockIdx();

        // gm init
        in1c1Gm_.SetGlobalBuffer((__gm__ T*)in1c1);
        in1c2Gm_.SetGlobalBuffer((__gm__ T*)in1c2);
        out1Gm_.SetGlobalBuffer((__gm__ T*)out1);

        in2c1Gm_.SetGlobalBuffer((__gm__ T*)in2c1);
        in2c2Gm_.SetGlobalBuffer((__gm__ T*)in2c2);
        out2Gm_.SetGlobalBuffer((__gm__ T*)out2);

        glc1Gm_.SetGlobalBuffer((__gm__ int32_t*)glc1);
        glc2Gm_.SetGlobalBuffer((__gm__ int32_t*)glc2);
        glGm_.SetGlobalBuffer((__gm__ int32_t*)gl);

        // alloc ub buf: token buffer + two group-list buffers
        pipe_.InitBuffer(ub_, TOTAL_BUFFER_BYTES);
    }

private:
    const Tiling& tiling_;

    // gm tensors
    GlobalTensor<T> in1c1Gm_;
    GlobalTensor<T> in1c2Gm_;
    GlobalTensor<T> out1Gm_;

    GlobalTensor<T> in2c1Gm_;
    GlobalTensor<T> in2c2Gm_;
    GlobalTensor<T> out2Gm_;

    GlobalTensor<int32_t> glc1Gm_;
    GlobalTensor<int32_t> glc2Gm_;
    GlobalTensor<int32_t> glGm_;

    // ub tensors
    TPipe& pipe_;
    TBuf<> ub_;
    LocalTensor<T> input_;
    LocalTensor<int32_t> glc1_;
    LocalTensor<int32_t> glc2_;

    // core info
    uint32_t blockIdx_ = 0;
    int curToken_;
    int curTokenOffset_;

    // sync event
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_S>());
    event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_MTE3>());
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>());
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
};

} // namespace MergeChunkSupport

#endif // MERGE_CHUNK_IMPL_H_
