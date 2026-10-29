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
 * \file histgram_impl.hpp
 * \brief Histgram kernel 实现
 *
 * 纯向量直方图: 统计 [0, numExpert) 每个 expert 在 token 序列中出现的次数。
 * 按 expert 分块、token 分块, 各核分别统计后写入 workspace, 再汇总写回输出。
 */
#ifndef HISTGRAM_IMPL_HPP
#define HISTGRAM_IMPL_HPP

#include "kernel_operator.h"
#include "histgram_tiling_def.h"

using namespace AscendC;

namespace histgram_ns {

class Histogram {
public:
    __aicore__ Histogram(const optiling::HistgramTilingData& tiling)
        : tiling_(tiling)
    {}

    __aicore__ void Process(GM_ADDR input, GM_ADDR out, GM_ADDR workspace, TPipe& pipe)
    {
        // only vector can go through
        if (g_coreType == AIC) {
            return;
        }
        // get block id
        blockIdx_ = GetBlockIdx();

        // gm init
        inputGm_.SetGlobalBuffer((__gm__ int32_t*)input);
        outputGm_.SetGlobalBuffer((__gm__ int32_t*)out);
        workspaceGm_.SetGlobalBuffer((__gm__ int32_t*)workspace);

        if (tiling_.numToken == 0) {
            Zero();
            return;
        }

        int expertChunkNum = (tiling_.numExpert + EXPERT_CHUNK_SIZE - 1) / EXPERT_CHUNK_SIZE;
        int tailNumExpert = tiling_.numExpert - (expertChunkNum - 1) * EXPERT_CHUNK_SIZE;

        const uint32_t tokenChunkSize = tiling_.usedCoreNum * TOKEN_CHUNK_SIZE;
        int tokenChunkNum = (tiling_.numToken + tokenChunkSize - 1) / tokenChunkSize;
        int tailNumToken = tiling_.numToken - (tokenChunkNum - 1) * tokenChunkSize;

        for (int eid = 0; eid < expertChunkNum; ++eid) {
            int numExpert = (eid == expertChunkNum - 1 ? tailNumExpert : EXPERT_CHUNK_SIZE);
            int expertStartId = eid * EXPERT_CHUNK_SIZE;

            for (int tid = 0; tid < tokenChunkNum; ++tid) {
                int numToken = (tid == tokenChunkNum - 1 ? tailNumToken : tokenChunkSize);
                int tokenStartPos = tid * tokenChunkSize; // cur chunk
                ProcessChunk(pipe, expertStartId, numExpert, tokenStartPos, numToken);
                SyncAll();
                pipe.Reset();
            }
        }
    }

    __aicore__ void Zero()
    {
        // split core
        const int expertPerCore = (tiling_.numExpert + tiling_.usedCoreNum - 1) / tiling_.usedCoreNum;
        const int expertPerCoreTail = expertPerCore - 1;
        const int tailCoreNum = expertPerCore * tiling_.usedCoreNum - tiling_.numExpert;
        const int formerCoreNum = tiling_.usedCoreNum - tailCoreNum;
        int curExpert = expertPerCore;
        if (blockIdx_ >= formerCoreNum) {
            curExpert = expertPerCoreTail;
        }
        int curExpertOffset = blockIdx_ * expertPerCore;
        if (blockIdx_ > formerCoreNum) {
            curExpertOffset = formerCoreNum * expertPerCore + (blockIdx_ - formerCoreNum) * expertPerCoreTail;
        }

        do {
            if (curExpert <= 0) {
                break;
            }
            InitOutput(outputGm_[curExpertOffset], curExpert, static_cast<int32_t>(0));
        } while (0);
        SyncAll();
    }

    __aicore__ void ProcessChunk(TPipe& pipe, int expertStartId, int numExpert, int tokenStartPos, int numToken)
    {
        // split core
        const int tokenPerCore = (numToken + tiling_.usedCoreNum - 1) / tiling_.usedCoreNum;
        const int tokenPerCoreTail = tokenPerCore - 1;
        const int tailCoreNum = tokenPerCore * tiling_.usedCoreNum - numToken;
        const int formerCoreNum = tiling_.usedCoreNum - tailCoreNum;
        int curToken = tokenPerCore;
        if (blockIdx_ >= formerCoreNum) {
            curToken = tokenPerCoreTail;
        }
        int curTokenOffset = blockIdx_ * tokenPerCore;
        if (blockIdx_ > formerCoreNum) {
            curTokenOffset = formerCoreNum * tokenPerCore + (blockIdx_ - formerCoreNum) * tokenPerCoreTail;
        }
        int validCoreNum = (tokenPerCoreTail == 0 ? formerCoreNum : tiling_.usedCoreNum);
        // local tensor init
        int summaryExpert = validCoreNum * numExpert;
        int curTokenAlignTo256B =
            (curToken + REPEAT_IDX_NUM - 1) / REPEAT_IDX_NUM * REPEAT_IDX_NUM; // align to 256B(one repeat)
        int compareResultCount = curTokenAlignTo256B / 8;
        compareResultCount = (compareResultCount + 32 - 1) / 32 * 32;
        int curTokenAlign = (curToken + BLOCK_IDX_NUM - 1) / BLOCK_IDX_NUM * BLOCK_IDX_NUM;
        int numExpertAlign = (numExpert + BLOCK_IDX_NUM - 1) / BLOCK_IDX_NUM * BLOCK_IDX_NUM;
        int summaryExpertAlign = (summaryExpert + BLOCK_IDX_NUM - 1) / BLOCK_IDX_NUM * BLOCK_IDX_NUM;
        do {
            pipe.InitBuffer(summaryBuf_, summaryExpertAlign * sizeof(int32_t));
            pipe.InitBuffer(accumBuf_, 8 * sizeof(int32_t));
            summaryUB_ = summaryBuf_.Get<int32_t>(summaryExpertAlign);
            accumUB_ = accumBuf_.Get<int32_t>(8);

            if (curToken <= 0) {
                break;
            }

            pipe.InitBuffer(inputBuf_, curTokenAlignTo256B * sizeof(int32_t));
            pipe.InitBuffer(inputBufFp32_, curTokenAlignTo256B * sizeof(float));
            pipe.InitBuffer(compareRhsBuf_, curTokenAlignTo256B * sizeof(float));
            pipe.InitBuffer(compareDstBuf_, compareResultCount * sizeof(uint8_t));
            pipe.InitBuffer(selectLhsBuf_, curTokenAlignTo256B * sizeof(float));
            pipe.InitBuffer(reduceSumWorkBuf_, REDUCE_SUM_WORK_NUM * sizeof(float));
            pipe.InitBuffer(reduceSumDstBuf_, 8 * sizeof(float));
            pipe.InitBuffer(histBuf_, numExpertAlign * sizeof(int32_t));

            inputUB_ = inputBuf_.Get<int32_t>(curTokenAlignTo256B);
            inputUBFp32_ = inputBufFp32_.Get<float>(curTokenAlignTo256B);
            compareRhsUB_ = compareRhsBuf_.Get<float>(curTokenAlignTo256B);
            compareDstUB_ = compareDstBuf_.Get<uint8_t>(compareResultCount);
            selectLhsUB_ = selectLhsBuf_.Get<float>(curTokenAlignTo256B);
            reduceSumWorkUB_ = reduceSumWorkBuf_.Get<float>(REDUCE_SUM_WORK_NUM);
            reduceSumDstUB_ = reduceSumDstBuf_.Get<float>(8);
            histUB_ = histBuf_.Get<int32_t>(numExpertAlign);

            // init input to -1
            Duplicate(inputUB_, -1, curTokenAlignTo256B);
            SetFlag<HardEvent::V_MTE2>(eventIdVToMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMTE2);

            CopyInData(curToken, curTokenAlign, curTokenOffset + tokenStartPos);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            Cast(inputUBFp32_, inputUB_, RoundMode::CAST_NONE, curTokenAlignTo256B);
            PipeBarrier<PIPE_V>();

            Duplicate(selectLhsUB_, static_cast<float>(1), curTokenAlignTo256B);
            PipeBarrier<PIPE_V>();

            int expertCount = -1;
            for (int expertId = expertStartId; expertId < expertStartId + numExpert; ++expertId) {
                Duplicate(compareRhsUB_, static_cast<float>(expertId), curTokenAlignTo256B);
                PipeBarrier<PIPE_V>();

                Compare(compareDstUB_, inputUBFp32_, compareRhsUB_, CMPMODE::EQ, curTokenAlignTo256B);
                PipeBarrier<PIPE_V>();

                Select(compareRhsUB_, compareDstUB_, selectLhsUB_, static_cast<float>(0),
                       SELMODE::VSEL_TENSOR_SCALAR_MODE, curTokenAlignTo256B);
                PipeBarrier<PIPE_V>();

                ReduceSum(reduceSumDstUB_, compareRhsUB_, reduceSumWorkUB_, curTokenAlignTo256B);

                SetFlag<HardEvent::V_S>(eventIdVToS);
                WaitFlag<HardEvent::V_S>(eventIdVToS);
                expertCount = static_cast<int>(reduceSumDstUB_.GetValue(0));
                histUB_.SetValue(expertId - expertStartId, expertCount);
            }

            SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);

            CopyOutToWorkspace(numExpert);
        } while (0);

        SyncAll<true>();

        /////////////////// summary ////////////////////////////

        CopyInWorkspace(summaryExpert);
        SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

        auto loopTimes = Ceil(numExpert, tiling_.usedCoreNum);

        SetFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
        for (int i = 0; i < loopTimes; ++i) {
            WaitFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);

            auto expertId = blockIdx_ + i * tiling_.usedCoreNum;

            if (expertId < numExpert) {
                auto accum = 0;

                for (int j = 0; j < validCoreNum; ++j) {
                    auto cur_count = summaryUB_.GetValue(expertId + j * numExpert);
                    accum += cur_count;
                }

                accumUB_.SetValue(0, accum);
                SetFlag<HardEvent::S_MTE3>(eventIdSToMTE3);
                WaitFlag<HardEvent::S_MTE3>(eventIdSToMTE3);

                if (tokenStartPos == 0) {
                    CopyOut(expertId + expertStartId);
                } else {
                    SetAtomicAdd<int32_t>();
                    CopyOut(expertId + expertStartId);
                    SetAtomicNone();
                }
            }
            SetFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
        }
        WaitFlag<HardEvent::MTE3_S>(eventIdMTE3ToS);
    }

private:
    __aicore__ void CopyInData(int curToken, int curTokenAlign, int curTokenOffset)
    {
        DataCopyExtParams copyParams;
        // 实际的行数
        copyParams.blockCount = 1;
        // 这里的单位是字节，即Byte
        copyParams.blockLen = curToken * sizeof(int32_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPadExtParams<int32_t> padParams;
        padParams.isPad = true;
        padParams.leftPadding = 0;
        // 右侧padding -1
        padParams.paddingValue = -1;
        padParams.rightPadding = curTokenAlign - curToken;
        DataCopyPad(inputUB_, inputGm_[curTokenOffset], copyParams, padParams);
    }

    __aicore__ void CopyOutToWorkspace(int numExpert)
    {
        DataCopyExtParams copyParams;
        // 实际的行数
        copyParams.blockCount = 1;
        // 这里的单位是字节，即Byte
        copyParams.blockLen = numExpert * sizeof(int32_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(workspaceGm_[blockIdx_ * numExpert], histUB_, copyParams);
    }

    __aicore__ void CopyInWorkspace(int summaryExpert)
    {
        DataCopyExtParams copyParams;
        // 实际的行数
        copyParams.blockCount = 1;
        // 这里的单位是字节，即Byte
        copyParams.blockLen = summaryExpert * sizeof(int32_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        DataCopyPadExtParams<int32_t> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.paddingValue = 0;
        padParams.rightPadding = 0;
        DataCopyPad(summaryUB_, workspaceGm_, copyParams, padParams);
    }

    __aicore__ void CopyOut(uint32_t expertId)
    {
        DataCopyExtParams copyParams;
        // 实际的行数
        copyParams.blockCount = 1;
        // 这里的单位是字节，即Byte
        copyParams.blockLen = 1 * sizeof(int32_t);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPad(outputGm_[expertId], accumUB_, copyParams);
    }

private:
    const optiling::HistgramTilingData& tiling_;

    GlobalTensor<int32_t> inputGm_;
    GlobalTensor<int32_t> outputGm_;
    GlobalTensor<int32_t> workspaceGm_;

    TBuf<> inputBuf_;
    TBuf<> inputBufFp32_;
    TBuf<> compareRhsBuf_;
    TBuf<> compareDstBuf_;
    TBuf<> selectLhsBuf_;
    TBuf<> reduceSumWorkBuf_;
    TBuf<> reduceSumDstBuf_;
    TBuf<> histBuf_;
    LocalTensor<int32_t> inputUB_;
    LocalTensor<float> inputUBFp32_;
    LocalTensor<float> compareRhsUB_;
    LocalTensor<uint8_t> compareDstUB_;
    LocalTensor<float> selectLhsUB_;
    LocalTensor<float> reduceSumWorkUB_;
    LocalTensor<float> reduceSumDstUB_;
    LocalTensor<int32_t> histUB_;
    TBuf<> accumBuf_;
    LocalTensor<int32_t> accumUB_;
    TBuf<> summaryBuf_;
    LocalTensor<int32_t> summaryUB_;

    int32_t blockIdx_ = 0;

    // some consts
    const uint32_t BLOCK_BYTES = 32;
    const uint32_t REPEAT_BYTES = 256;
    const uint32_t IDX_BYTES = sizeof(int32_t);
    const uint32_t BLOCK_IDX_NUM = BLOCK_BYTES / IDX_BYTES;
    const uint32_t REPEAT_IDX_NUM = REPEAT_BYTES / IDX_BYTES;
    const uint32_t REDUCE_SUM_WORK_NUM = 1024; // hard code here: 1024 repeat times is large enough.
    const uint32_t EXPERT_CHUNK_SIZE = 64;
    const uint32_t TOKEN_CHUNK_BYTES = 30 * 1024;
    const uint32_t TOKEN_CHUNK_SIZE = TOKEN_CHUNK_BYTES / sizeof(int32_t) / REPEAT_BYTES * REPEAT_BYTES; // 7680

    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_S>());
    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_S>());
    event_t eventIdVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE2>());
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    event_t eventIdSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::S_MTE3>());
    event_t eventIdMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_S>());
};

} // namespace histgram_ns

#endif // HISTGRAM_IMPL_HPP
