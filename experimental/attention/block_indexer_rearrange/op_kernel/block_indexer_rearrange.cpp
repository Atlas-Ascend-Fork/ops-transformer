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
 * \file block_indexer_rearrange.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))

constexpr static int32_t REPEAT_MAX_BYTES = 256;
constexpr static int32_t BLOCK_BYTES = 32;
constexpr static int32_t REPEAT_MAX_BLOCKS = 255;

template <typename INPUT_T>
class BlockIndexerRearrange {
public:
    __aicore__ inline BlockIndexerRearrange(){};

    __aicore__ inline void Init(__gm__ uint8_t* topk_indices, __gm__ uint8_t* acc_seqlen_q,
                                __gm__ uint8_t* acc_seqlen_kv, __gm__ uint8_t* topk_output, __gm__ uint8_t* workspace,
                                const BlockIndexerRearrangeTilingData* __restrict tiling, TPipe* tPipe);

    __aicore__ inline void Process();

protected:
    int32_t blockIdx;
    TPipe* pipe;
    const BlockIndexerRearrangeTilingData* __restrict tilingData;

    // host base info
    int32_t batchSize;
    int32_t Tq;
    int32_t Tk;
    int32_t headNum;
    int32_t blockCount;
    int32_t blockSize;
    bool causal;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;

    // host tiling info
    int32_t usedCoreNum;
    int32_t formerCoreNum;
    int32_t formerCoreProcessNums;
    int32_t remainCoreProcessNums;

    // device block info
    int32_t actualProcessNums;
    int32_t t1Offset{0};
    int32_t t2Offset{0};
    int32_t curS1{0};
    int32_t curS2{0};

    // device indexe
    int64_t bIndex{0};
    int64_t n2Index{0};
    int64_t s1Index{0};

    // input & output
    GlobalTensor<INPUT_T> topkInputGm;
    GlobalTensor<INPUT_T> topkOutputGm;
    GM_ADDR seqQlenAddr;
    GM_ADDR seqKlenAddr;

    // tbuf
    TBuf<> topkInputBuf;
    TBuf<> brcbBuf;
    TBuf<> helpBuf;

private:
    __aicore__ inline void GetTndSeqLen(const GM_ADDR actual_seq_qlen_addr, const GM_ADDR actual_seq_kvlen_addr,
                                        const int64_t t1Idx, int64_t& bIdx);
    __aicore__ inline void VecCompute();
};

template <typename INPUT_T>
__aicore__ inline void BlockIndexerRearrange<INPUT_T>::Init(__gm__ uint8_t* topk_indices, __gm__ uint8_t* acc_seqlen_q,
                                                            __gm__ uint8_t* acc_seqlen_kv, __gm__ uint8_t* topk_output,
                                                            __gm__ uint8_t* workspace,
                                                            const BlockIndexerRearrangeTilingData* __restrict tiling,
                                                            TPipe* tPipe)
{
    blockIdx = GetBlockIdx();
    pipe = tPipe;
    tilingData = tiling;

    // base info
    Tq = tilingData->baseParams.totalSeqlen;
    headNum = tilingData->baseParams.headNum;
    blockCount = tilingData->baseParams.blockCount;
    blockSize = tilingData->baseParams.blockSize;
    causal = tilingData->baseParams.causal;

    // tiling info
    usedCoreNum = tilingData->coreParams.usedCoreNum;
    formerCoreNum = tilingData->coreParams.formerCoreNum;
    formerCoreProcessNums = tilingData->coreParams.formerCoreProcessNums;
    remainCoreProcessNums = tilingData->coreParams.remainCoreProcessNums;

    topkInputGm.SetGlobalBuffer((__gm__ INPUT_T*)topk_indices);
    seqQlenAddr = acc_seqlen_q;
    seqKlenAddr = acc_seqlen_kv;
    topkOutputGm.SetGlobalBuffer((__gm__ INPUT_T*)topk_output);

    // init buffer
    pipe->InitBuffer(topkInputBuf, 32 * 1024); // 32k
    pipe->InitBuffer(brcbBuf, 128 * 1024);
    pipe->InitBuffer(helpBuf, 1024);
}

template <typename INPUT_T>
__aicore__ inline void BlockIndexerRearrange<INPUT_T>::Process()
{
    if (blockIdx < formerCoreNum) {
        actualProcessNums = formerCoreProcessNums;
    } else {
        actualProcessNums = remainCoreProcessNums;
    }

    for (int32_t i = 0; i < actualProcessNums; i++) {
        int32_t t1Index = blockIdx + usedCoreNum * i;
        GetTndSeqLen(seqQlenAddr, seqKlenAddr, t1Index, bIndex);
        for (n2Index = 0; n2Index < headNum; n2Index++) {
            VecCompute();
        }
    }
}

template <typename INPUT_T>
__aicore__ inline void BlockIndexerRearrange<INPUT_T>::GetTndSeqLen(const GM_ADDR actual_seq_qlen_addr,
                                                                    const GM_ADDR actual_seq_kvlen_addr,
                                                                    const int64_t t1Idx, int64_t& bIdx)
{
    int64_t curT1 = ((__gm__ int64_t*)actual_seq_qlen_addr)[bIndex];
    while (t1Idx >= curT1) {
        curT1 = ((__gm__ int64_t*)actual_seq_qlen_addr)[++bIndex];
    }

    if (unlikely(bIndex == 0)) {
        t1Offset = 0;
        t2Offset = 0;
        curS1 = ((__gm__ int64_t*)actual_seq_qlen_addr)[bIndex];
        curS2 = ((__gm__ int64_t*)actual_seq_kvlen_addr)[bIndex];
    } else {
        t1Offset = ((__gm__ int64_t*)actual_seq_qlen_addr)[bIndex - 1];
        t2Offset = ((__gm__ int64_t*)actual_seq_kvlen_addr)[bIndex - 1];
        curS1 = ((__gm__ int64_t*)actual_seq_qlen_addr)[bIndex] - ((__gm__ int64_t*)actual_seq_qlen_addr)[bIndex - 1];
        curS2 = ((__gm__ int64_t*)actual_seq_kvlen_addr)[bIndex] - ((__gm__ int64_t*)actual_seq_kvlen_addr)[bIndex - 1];
    }
    s1Index = t1Idx - t1Offset;
}

template <typename INPUT_T>
__aicore__ inline void BlockIndexerRearrange<INPUT_T>::VecCompute()
{
    event_t Mte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_MTE2>());
    event_t vWaitMte2 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t Mte3WaitS = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::S_MTE3>());
    event_t Mte3WaitV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());

    uint64_t s1Offset = (t1Offset + s1Index) * headNum * blockCount;
    uint64_t nOffset = n2Index * blockCount;
    uint64_t gmOffset = s1Offset + nOffset;
    int32_t blockEleNums = BLOCK_BYTES / sizeof(INPUT_T);

    LocalTensor<INPUT_T> topkInputTensor = topkInputBuf.template Get<INPUT_T>();
    LocalTensor<INPUT_T> brcbTensor = brcbBuf.template Get<INPUT_T>();
    LocalTensor<INPUT_T> helpTensor = helpBuf.template Get<INPUT_T>();
    DataCopyExtParams dataCopyParams;
    DataCopyPadExtParams<INPUT_T> padParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = blockCount * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    SET_FLAG(MTE3, MTE2, Mte2WaitMte3);
    WAIT_FLAG(MTE3, MTE2, Mte2WaitMte3);

    DataCopyPad(topkInputTensor, topkInputGm[gmOffset], dataCopyParams, padParams);

    SET_FLAG(MTE2, V, vWaitMte2);
    WAIT_FLAG(MTE2, V, vWaitMte2);

    // brcb parms
    int32_t repeatTims = (blockCount + 8 - 1) / 8;   // 每次固定处理8个
    int32_t maxRepeatOffset = REPEAT_MAX_BLOCKS * 8; // src 一次最大offset，对应dst offset 需要 * blockEleNums
    int32_t loop = repeatTims / REPEAT_MAX_BLOCKS;
    int32_t remain = repeatTims % REPEAT_MAX_BLOCKS;
    for (int32_t i = 0; i < loop; i++) {
        Brcb(brcbTensor[i * maxRepeatOffset * blockEleNums], topkInputTensor[i * maxRepeatOffset], REPEAT_MAX_BLOCKS,
             {1, 8});
    }
    if (likely(remain)) {
        Brcb(brcbTensor[loop * maxRepeatOffset * blockEleNums], topkInputTensor[loop * maxRepeatOffset], remain,
             {1, 8});
    }

    PipeBarrier<PIPE_V>();
    // creat offset vector [0, 1, 2, ...]
    CreateVecIndex(helpTensor, (INPUT_T)0, blockEleNums);
    PipeBarrier<PIPE_V>();

    int32_t validS2Len = s1Index + curS2 - curS1 + 1;
    int32_t validTopkSize = (validS2Len + blockSize - 1) / blockSize;
    validTopkSize = validTopkSize > blockCount ? blockCount : validTopkSize;

    int32_t totalEle = validTopkSize * blockEleNums;
    int32_t maxProcessOnce = REPEAT_MAX_BYTES / sizeof(INPUT_T);
    loop = totalEle / maxProcessOnce;
    remain = totalEle % maxProcessOnce;
    Muls(brcbTensor, brcbTensor, blockSize, totalEle);
    PipeBarrier<PIPE_V>();

    if (unlikely(loop > REPEAT_MAX_BLOCKS)) {
        int32_t max_repeat_loops = loop / REPEAT_MAX_BLOCKS;
        int32_t max_repeat_remains = loop % REPEAT_MAX_BLOCKS;

        for (int32_t idx = 0; idx < max_repeat_loops; idx++) {
            Add(brcbTensor[idx * REPEAT_MAX_BLOCKS * maxProcessOnce],
                brcbTensor[idx * REPEAT_MAX_BLOCKS * maxProcessOnce], helpTensor, maxProcessOnce, REPEAT_MAX_BLOCKS,
                {1, 1, 0, /*block stride*/
                 8, 8, 0 /*dst rep stride*/});
        }
        if (likely(max_repeat_remains)) {
            Add(brcbTensor[max_repeat_loops * REPEAT_MAX_BLOCKS * maxProcessOnce],
                brcbTensor[max_repeat_loops * REPEAT_MAX_BLOCKS * maxProcessOnce], helpTensor, maxProcessOnce,
                REPEAT_MAX_BLOCKS,
                {1, 1, 0, /*block stride*/
                 8, 8, 0 /*dst rep stride*/});
        }
    } else {
        Add(brcbTensor, brcbTensor, helpTensor, maxProcessOnce, loop,
            {1, 1, 0, /*block stride*/
             8, 8, 0 /*dst rep stride*/});
    }
    if (likely(remain)) {
        Add(brcbTensor[loop * maxProcessOnce], brcbTensor[loop * maxProcessOnce], helpTensor, remain, 1,
            {1, 1, 0, /*block stride*/
             8, 8, 0 /*dst rep stride*/});
    }
    PipeBarrier<PIPE_V>();

    int32_t validTail = validS2Len % blockSize;
    int32_t maskLen = blockEleNums - validTail;
    if (validTail > 0) {
        for (int i = 0; i < maskLen; i++) {
            brcbTensor.SetValue((validTopkSize - 1) * blockEleNums + validTail + i, (INPUT_T)-1);
        }
        SET_FLAG(S, MTE3, Mte3WaitS);
        WAIT_FLAG(S, MTE3, Mte3WaitS);
    }

    SET_FLAG(V, MTE3, Mte3WaitV);
    WAIT_FLAG(V, MTE3, Mte3WaitV);

    dataCopyParams.blockCount = blockCount;
    dataCopyParams.blockLen = blockSize * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    int64_t outOffset = (t1Offset + s1Index) * headNum * blockCount * blockSize + n2Index * blockCount * blockSize;
    DataCopyPad(topkOutputGm[outOffset], brcbTensor, dataCopyParams);
}

extern "C" __global__ __aicore__ void block_indexer_rearrange(__gm__ uint8_t* topk_indices,
                                                              __gm__ uint8_t* acc_seqlen_q,
                                                              __gm__ uint8_t* acc_seqlen_kv,
                                                              __gm__ uint8_t* topk_output, __gm__ uint8_t* workspace,
                                                              __gm__ uint8_t* tiling)
{
    TPipe tPipe;
    set_mask_norm();
    SetSysWorkspace(workspace);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);

    GET_TILING_DATA_WITH_STRUCT(BlockIndexerRearrangeTilingData, tiling_data_in, tiling);
    const BlockIndexerRearrangeTilingData* __restrict tilingData = &tiling_data_in;

    if (TILING_KEY_IS(1)) {
        BlockIndexerRearrange<int32_t> op;
        op.Init(topk_indices, acc_seqlen_q, acc_seqlen_kv, topk_output, user, tilingData, &tPipe);
        op.Process();
    }

    return;
}
