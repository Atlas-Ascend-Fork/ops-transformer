/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _FLASH_ATTENTION_GRAD_US1S2_BNGS1S2_POST_KERNEL_H_
#define _FLASH_ATTENTION_GRAD_US1S2_BNGS1S2_POST_KERNEL_H_
#include "kernel_operator.h"

template <typename T1, typename T2>
class FlashAttentionGradUs1s2Bn2gs1s2Post {
public:
    __aicore__ inline FlashAttentionGradUs1s2Bn2gs1s2Post(){};
    __aicore__ inline void Init(__gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dsink,
                                __gm__ uint8_t* workspace,
                                const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData,
                                TPipe* pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessDsink();

    TPipe* pipe;
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TQue<QuePosition::VECIN, 1> tmpQueue;
    TBuf<> vecClc1;
    TBuf<> vecClc2;

    GlobalTensor<T1> dqGm, dkGm, dvGm, dsinkGm;
    // input
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dsinkWorkSpaceGm;

    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict tilingData;
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;

    uint32_t usedCoreNum;
    uint32_t cBlockIdx;
    // query
    uint32_t ubBaseSize;
    uint32_t qPostBlockFactor;
    uint64_t qPostBlockTotal;
    uint32_t qPostBaseNum;
    uint32_t qPostTailNum;
    uint64_t qSizeAlign;
    uint32_t kPostBlockFactor;
    uint64_t kPostBlockTotal;
    uint32_t kPostBaseNum;
    uint32_t kPostTailNum;
    uint64_t kSizeAlign;
    uint32_t vPostBlockFactor;
    uint64_t vPostBlockTotal;
    uint32_t vPostBaseNum;
    uint32_t vPostTailNum;
    uint64_t vSizeAlign;
};

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionGradUs1s2Bn2gs1s2Post<T1, T2>::Init(
    __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dsink, __gm__ uint8_t* workspace,
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData, TPipe* pipe_in)
{
    cBlockIdx = GetBlockIdx();

    tilingData = ordTilingData;
    pipe = pipe_in;

    dqGm.SetGlobalBuffer((__gm__ T1*)dq);
    dkGm.SetGlobalBuffer((__gm__ T1*)dk);
    dvGm.SetGlobalBuffer((__gm__ T1*)dv);
    dsinkGm.SetGlobalBuffer((__gm__ T1*)dsink);

    // tiling_data
    usedCoreNum = tilingData->s1s2BNGS1S2BaseParams.coreNum;
    ubBaseSize = tilingData->postTilingData.postUbBaseSize;
    qPostBlockFactor = tilingData->postTilingData.qPostBlockFactor;
    qPostBlockTotal = tilingData->postTilingData.qPostBlockTotal;
    qPostBaseNum = tilingData->postTilingData.qPostBaseNum;
    qPostTailNum = tilingData->postTilingData.qPostTailNum;
    qSizeAlign = tilingData->postTilingData.qSizeAlign;
    kPostBlockFactor = tilingData->postTilingData.kPostBlockFactor;
    kPostBlockTotal = tilingData->postTilingData.kPostBlockTotal;
    kPostBaseNum = tilingData->postTilingData.kPostBaseNum;
    kPostTailNum = tilingData->postTilingData.kPostTailNum;
    kSizeAlign = tilingData->postTilingData.kSizeAlign;
    vPostBlockFactor = tilingData->postTilingData.vPostBlockFactor;
    vPostBlockTotal = tilingData->postTilingData.vPostBlockTotal;
    vPostBaseNum = tilingData->postTilingData.vPostBaseNum;
    vPostTailNum = tilingData->postTilingData.vPostTailNum;
    vSizeAlign = tilingData->postTilingData.vSizeAlign;

    uint64_t qPostBlockTotal = tilingData->postTilingData.qPostBlockTotal;
    uint64_t kPostBlockTotal = tilingData->postTilingData.kPostBlockTotal;
    uint64_t vPostBlockTotal = tilingData->postTilingData.vPostBlockTotal;

    uint64_t workspaceOffsets = SYNC_GLOBAL_WORKSPACE_SIZE;
    dqWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + qPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + kPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));
    workspaceOffsets =
        (workspaceOffsets + vPostBlockTotal * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    dsinkWorkSpaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffsets / sizeof(T2));

    pipe->InitBuffer(inQueue, 1, ubBaseSize * 2);
    pipe->InitBuffer(outQueue, 1, ubBaseSize);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionGradUs1s2Bn2gs1s2Post<T1, T2>::Process()
{
    bool isLastCore = (cBlockIdx == (usedCoreNum - 1));
    // init q
    uint64_t qBegin = cBlockIdx * qPostBlockFactor * qPostBaseNum;
    uint64_t qEnd = (cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum;

    if (((cBlockIdx + 1) * qPostBlockFactor * qPostBaseNum) > qPostBlockTotal) {
        qEnd = qPostBlockTotal;
    }
    for (uint64_t i = qBegin; i < qEnd; i = i + qPostBaseNum) {
        LocalTensor<float> vecIn = inQueue.AllocTensor<float>();
        LocalTensor<T1> vecOut = outQueue.AllocTensor<T1>();
        uint32_t dataSize = i + qPostBaseNum < qPostBlockTotal ? qPostBaseNum : qPostTailNum;
        DataCopy(vecIn, dqWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.DeQue<float>();

        Muls(vecIn, vecIn, (float)tilingData->s1s2BNGS1S2BaseParams.scaleValue, dataSize);
        PipeBarrier<PIPE_V>();
        Cast(vecOut, vecIn, RoundMode::CAST_ROUND, dataSize);
        outQueue.EnQue(vecOut);
        outQueue.DeQue<T1>();
        DataCopy(dqGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
    }
    pipe_barrier(PIPE_ALL);
    // init k
    uint64_t kBegin = cBlockIdx * kPostBlockFactor * kPostBaseNum;
    uint64_t kEnd = (cBlockIdx + 1) * kPostBlockFactor * kPostBaseNum;
    if (((cBlockIdx + 1) * kPostBlockFactor * kPostBaseNum) > kPostBlockTotal) {
        kEnd = kPostBlockTotal;
    }

    for (uint64_t i = kBegin; i < kEnd; i = i + kPostBaseNum) {
        LocalTensor<float> vecIn = inQueue.AllocTensor<float>();
        LocalTensor<T1> vecOut = outQueue.AllocTensor<T1>();
        uint32_t dataSize = i + kPostBaseNum < kPostBlockTotal ? kPostBaseNum : kPostTailNum;
        DataCopy(vecIn, dkWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.DeQue<float>();
        Muls(vecIn, vecIn, (float)tilingData->s1s2BNGS1S2BaseParams.scaleValue, dataSize);
        PipeBarrier<PIPE_V>();
        Cast(vecOut, vecIn, RoundMode::CAST_ROUND, dataSize);
        outQueue.EnQue(vecOut);
        outQueue.DeQue<T1>();
        DataCopy(dkGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
    }
    pipe_barrier(PIPE_ALL);

    // init v
    uint64_t vBegin = cBlockIdx * vPostBlockFactor * vPostBaseNum;
    uint64_t vEnd = (cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum;
    if (((cBlockIdx + 1) * vPostBlockFactor * vPostBaseNum) > vPostBlockTotal) {
        vEnd = vPostBlockTotal;
    }
    for (uint64_t i = vBegin; i < vEnd; i = i + vPostBaseNum) {
        LocalTensor<float> vecIn = inQueue.AllocTensor<float>();
        LocalTensor<T1> vecOut = outQueue.AllocTensor<T1>();
        uint32_t dataSize = i + vPostBaseNum < vPostBlockTotal ? vPostBaseNum : vPostTailNum;
        DataCopy(vecIn, dvWorkSpaceGm[i], (dataSize + 7) / 8 * 8); // dataSize(fp32) align 32B
        inQueue.EnQue(vecIn);
        inQueue.DeQue<float>();
        Cast(vecOut, vecIn, RoundMode::CAST_ROUND, dataSize);
        outQueue.EnQue(vecOut);
        outQueue.DeQue<T1>();
        DataCopy(dvGm[i], vecOut, (dataSize + 15) / 16 * 16); // dataSize(fp16) align 32B
        inQueue.FreeTensor(vecIn);
        outQueue.FreeTensor(vecOut);
    }
    pipe_barrier(PIPE_ALL);
    ProcessDsink();
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionGradUs1s2Bn2gs1s2Post<T1, T2>::ProcessDsink()
{
    if (!tilingData->s1s2BNGS1S2BaseParams.sinkRequiresGrad) {
        return;
    }
    SyncAll();
    pipe->Reset();
    uint32_t dsinkBaseSize = tilingData->postTilingData.dsinkBaseSize;
    pipe->InitBuffer(inQueue, 1, dsinkBaseSize * 2);
    pipe->InitBuffer(tmpQueue, 1, dsinkBaseSize * 2);
    pipe->InitBuffer(outQueue, 1, dsinkBaseSize);

    uint64_t n1 = tilingData->s1s2BNGS1S2BaseParams.n2 * tilingData->s1s2BNGS1S2BaseParams.g;
    uint64_t totalS1TileNums = tilingData->s1s2BNGS1S2SplitCoreParams.totalS1TileNums;
    for (int64_t coreId = cBlockIdx; coreId < n1; coreId += usedCoreNum) {
        LocalTensor<float> vecIn = inQueue.AllocTensor<float>();
        LocalTensor<float> tmpBuf = tmpQueue.AllocTensor<float>();
        LocalTensor<T1> vecOut = outQueue.AllocTensor<T1>();
        DataCopyPad(vecIn, dsinkWorkSpaceGm[coreId * totalS1TileNums],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(totalS1TileNums * sizeof(float)),
                     /*srcStride*/ 0, /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});

        inQueue.EnQue(vecIn);
        inQueue.DeQue<float>();
        ReduceSum<float>(vecIn, vecIn, tmpBuf, totalS1TileNums);
        PipeBarrier<PIPE_V>();
        Cast(vecOut, vecIn, RoundMode::CAST_ROUND, totalS1TileNums);
        outQueue.EnQue(vecOut);
        outQueue.DeQue<T1>();
        DataCopyPad(dsinkGm[coreId], vecOut,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(T1)),
                     /*srcStride*/ 0, /*dstStride*/ 0, 0});
        inQueue.FreeTensor(vecIn);
        tmpQueue.FreeTensor(tmpBuf);
        outQueue.FreeTensor(vecOut);
    }
}

#endif // _FLASH_ATTENTION_GRAD_US1S2_BNGS1S2_POST_KERNEL_H_
