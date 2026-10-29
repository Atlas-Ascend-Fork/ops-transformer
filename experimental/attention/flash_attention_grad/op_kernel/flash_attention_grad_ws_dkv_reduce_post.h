/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _FLASH_ATTENTION_GRAD_WS_DKV_REDUCE_POST_KERNEL_H_
#define _FLASH_ATTENTION_GRAD_WS_DKV_REDUCE_POST_KERNEL_H_
#include "kernel_operator.h"

template <typename T1, typename T2>
class FlashAttentionGradDkvReducePost {
public:
    __aicore__ inline FlashAttentionGradDkvReducePost(){};
    __aicore__ inline void Init(__gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dsink,
                                __gm__ uint8_t* workspace,
                                const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData,
                                TPipe* pipe_in);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessDsink();

    TPipe* pipe;
    TBuf<> src0Queue;
    TBuf<> src1Queue;
    TBuf<> outQueue;

    GlobalTensor<T1> dkGm;
    GlobalTensor<T1> dvGm;
    GlobalTensor<T1> dsinkGm;
    GlobalTensor<float> dkWorkSpaceGm;
    GlobalTensor<float> dvWorkSpaceGm;
    GlobalTensor<float> dsinkWorkSpaceGm;

    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict tilingData;
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;

    uint32_t usedCoreNum;
    uint32_t cBlockIdx;

    // base info
    uint32_t d;
    uint32_t dV;
    uint32_t b;
    int64_t Tq;
    int64_t Tkv;
    uint32_t n2;
    uint32_t g;
    uint32_t s2TileNum;
    uint32_t gs1TileNum;

    // fp32 src0 = fp32 src0 + fp32 src1
    // fp32 src0 = fp32 src0 * scale
    // bf16 src0 = cast(fp32 src0)
    uint32_t ubBaseSize = 30 * 1024; // 30kb

    // Tkv N2 gs1TileNum D
    uint32_t numS2PerCore;
    uint32_t cuS2;

    // some constants
    const int BLOCK_FLOAT_ELE_NUM = 8;
    const int BLOCK_HALF_ELE_NUM = 16;

    // some events
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>());
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE2>());
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
};

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionGradDkvReducePost<T1, T2>::Init(
    __gm__ uint8_t* dk, __gm__ uint8_t* dv, __gm__ uint8_t* dsink, __gm__ uint8_t* workspace,
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData, TPipe* pipe_in)
{
    cBlockIdx = GetBlockIdx();

    tilingData = ordTilingData;
    pipe = pipe_in;

    dkGm.SetGlobalBuffer((__gm__ T1*)dk);
    dvGm.SetGlobalBuffer((__gm__ T1*)dv);
    dsinkGm.SetGlobalBuffer((__gm__ T1*)dsink);

    // 下面更新一些基本信息
    usedCoreNum = (tilingData->s1s2BNGS1S2SplitCoreParams.blockOuter + 1) / 2 * 2;

    // shape info
    b = tilingData->s1s2BNGS1S2BaseParams.b;
    n2 = tilingData->s1s2BNGS1S2BaseParams.n2;
    g = tilingData->s1s2BNGS1S2BaseParams.g;
    d = tilingData->s1s2BNGS1S2BaseParams.d;
    dV = tilingData->s1s2BNGS1S2BaseParams.dV;

    // 更新token总数
    Tq = tilingData->s1s2BNGS1S2BaseParams.Tq;
    Tkv = tilingData->s1s2BNGS1S2BaseParams.Tkv;
    s2TileNum = tilingData->s1s2BNGS1S2BaseParams.s2TileNum;
    gs1TileNum = tilingData->s1s2BNGS1S2BaseParams.gs1TileNum;

    // 算一下，针对dqkv，多核每个核处理的token数量
    // @lg Tkv N2 G合轴: 每个core负责处理这么多d
    auto numS2PerCoreTiling = (Tkv * n2 * 1 + usedCoreNum - 1) / usedCoreNum;
    auto tailNumS2PerCoreTiling = numS2PerCoreTiling - 1;
    auto tailCoreNum = numS2PerCoreTiling * usedCoreNum - Tkv * n2 * 1;
    auto formerCoreNum = usedCoreNum - tailCoreNum;
    numS2PerCore = cBlockIdx >= formerCoreNum ? tailNumS2PerCoreTiling : numS2PerCoreTiling;
    cuS2 = cBlockIdx * numS2PerCoreTiling;
    if (cBlockIdx > formerCoreNum) {
        cuS2 = formerCoreNum * numS2PerCoreTiling + (cBlockIdx - formerCoreNum) * tailNumS2PerCoreTiling;
    }

    // 下面是dqkv worksapce起始位置
    uint64_t workspaceOffsets = tilingData->s1s2BNGS1S2BaseParams.dqkvWorkspaceOffset;

    uint64_t dqWorkspaceSize =
        (static_cast<uint64_t>(Tq) * (n2 * g) * d * s2TileNum * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE *
        ADDR_ALIGN_SIZE;
    uint64_t dkWorkspaceSize =
        (static_cast<uint64_t>(Tkv) * (n2 * 1) * d * gs1TileNum * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE *
        ADDR_ALIGN_SIZE;
    uint64_t dvWorkspaceSize =
        (static_cast<uint64_t>(Tkv) * (n2 * 1) * dV * gs1TileNum * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE *
        ADDR_ALIGN_SIZE;
    workspaceOffsets = workspaceOffsets + dqWorkspaceSize;
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ T2*)workspace + workspaceOffsets / sizeof(T2));

    workspaceOffsets = workspaceOffsets + dkWorkspaceSize;
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ T2*)workspace + workspaceOffsets / sizeof(T2));

    workspaceOffsets = workspaceOffsets + dvWorkspaceSize;
    dsinkWorkSpaceGm.SetGlobalBuffer((__gm__ T2*)workspace + workspaceOffsets / sizeof(T2));

    pipe->InitBuffer(src0Queue, ubBaseSize * 2);
    pipe->InitBuffer(src1Queue, ubBaseSize * 2);
    pipe->InitBuffer(outQueue, ubBaseSize);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionGradDkvReducePost<T1, T2>::Process()
{
    // 如果core没有分配到tokens，提前return
    if (numS2PerCore <= 0) {
        return;
    }

    // dk
    // Tkv N2 1 gs1TileNum D -> Tkv N2 1 D
    auto dAligned = (d + BLOCK_HALF_ELE_NUM - 1) / BLOCK_HALF_ELE_NUM * BLOCK_HALF_ELE_NUM;
    auto qkHeadDimBytes = dAligned * sizeof(float);
    auto qkInbaseNumRows = 2 * ubBaseSize / qkHeadDimBytes;

    LocalTensor<float> src0 = src0Queue.Get<float>(qkInbaseNumRows * dAligned);
    LocalTensor<float> src1 = src1Queue.Get<float>(qkInbaseNumRows * dAligned);
    LocalTensor<T1> out = outQueue.Get<T1>(qkInbaseNumRows * dAligned);

    // ub能容纳的行数和实际该core分配的行数做个判断
    auto rowsPerLoop = qkInbaseNumRows <= numS2PerCore ? qkInbaseNumRows : numS2PerCore;
    auto loopCount = (numS2PerCore + rowsPerLoop - 1) / rowsPerLoop;
    auto tailRowsPerLoop = numS2PerCore - (loopCount - 1) * rowsPerLoop;

    // 按照16 对齐搬运float数据，如果超了8，将dst stride置为1
    auto paddingRight = dAligned - d;
    auto dstStride = 0;
    if (paddingRight > BLOCK_FLOAT_ELE_NUM) {
        paddingRight -= BLOCK_FLOAT_ELE_NUM;
        dstStride = 1;
    }

    for (int loopIdx = 0; loopIdx < loopCount; ++loopIdx) {
        if (loopIdx > 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }

        auto realRowsPerLoop = (loopIdx == loopCount - 1 ? tailRowsPerLoop : rowsPerLoop);

        auto cuRowsInLoop = loopIdx * rowsPerLoop;
        // 搬入第一块
        auto src0Offset = (cuS2 + cuRowsInLoop) * (gs1TileNum * d);
        DataCopyPad(src0, dkWorkSpaceGm[src0Offset],
                    {static_cast<uint16_t>(realRowsPerLoop), static_cast<uint32_t>(d * sizeof(float)),
                     /*srcStride*/ static_cast<uint32_t>((gs1TileNum * d - d) * sizeof(float)),
                     /*dstStride*/ static_cast<uint32_t>(dstStride), 0},
                    {true, 0, static_cast<uint8_t>(paddingRight), 0});

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        // if necessary, load rhs、do lhs = lhs + rhs
        for (int gS1Idx = 1; gS1Idx < gs1TileNum; ++gS1Idx) {
            if (gS1Idx > 1) {
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            }

            auto src1Offset = src0Offset + gS1Idx * d;
            DataCopyPad(src1, dkWorkSpaceGm[src1Offset],
                        {static_cast<uint16_t>(realRowsPerLoop), static_cast<uint32_t>(d * sizeof(float)),
                         /*srcStride*/ static_cast<uint32_t>((gs1TileNum * d - d) * sizeof(float)),
                         /*dstStride*/ static_cast<uint32_t>(dstStride), 0},
                        {true, 0, static_cast<uint8_t>(paddingRight), 0});

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            PipeBarrier<PIPE_V>();
            Add(src0, src0, src1, realRowsPerLoop * dAligned);

            if (gS1Idx < gs1TileNum - 1) {
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            }
        }

        // 乘以scale
        PipeBarrier<PIPE_V>();
        Muls(src0, src0, (float)tilingData->s1s2BNGS1S2BaseParams.scaleValue, realRowsPerLoop * dAligned);

        PipeBarrier<PIPE_V>();
        Cast(out, src0, RoundMode::CAST_ROUND, realRowsPerLoop * dAligned);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        // 搬出
        auto outOffset = (cuS2 + cuRowsInLoop) * d;

        DataCopyPad(dkGm[outOffset], out,
                    {static_cast<uint16_t>(realRowsPerLoop), static_cast<uint32_t>(d * sizeof(T1)),
                     /*srcStride*/ static_cast<uint32_t>((dAligned - d) / BLOCK_HALF_ELE_NUM), /*dstStride*/ 0, 0});

        if (loopIdx < loopCount - 1) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }

    // @lg 这里需要加一下同步，否则dV的第一次MTE2会刷掉src0，会影响dK的最后一次vector(cast)->MTE3
    pipe_barrier(PIPE_ALL);

    // dV
    // Tkv N2 1 gs1TileNum DV -> Tkv N2 1 DV
    auto dVAligned = (dV + BLOCK_HALF_ELE_NUM - 1) / BLOCK_HALF_ELE_NUM * BLOCK_HALF_ELE_NUM;
    auto vHeadDimBytes = dVAligned * sizeof(float);
    auto vbaseNumRows = 2 * ubBaseSize / vHeadDimBytes;

    // 按照dv size重新申请tensor
    src0 = src0Queue.Get<float>(vbaseNumRows * dVAligned);
    src1 = src1Queue.Get<float>(vbaseNumRows * dVAligned);
    out = outQueue.Get<T1>(vbaseNumRows * dVAligned);

    // 按照dv size更新循环参数
    rowsPerLoop = vbaseNumRows <= numS2PerCore ? vbaseNumRows : numS2PerCore;
    loopCount = (numS2PerCore + rowsPerLoop - 1) / rowsPerLoop;
    tailRowsPerLoop = numS2PerCore - (loopCount - 1) * rowsPerLoop;

    paddingRight = dVAligned - dV;
    if (paddingRight > BLOCK_FLOAT_ELE_NUM) {
        paddingRight -= BLOCK_FLOAT_ELE_NUM;
        dstStride = 1;
    }
    for (int loopIdx = 0; loopIdx < loopCount; ++loopIdx) {
        if (loopIdx > 0) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }

        auto realRowsPerLoop = (loopIdx == loopCount - 1 ? tailRowsPerLoop : rowsPerLoop);

        auto cuRowsInLoop = loopIdx * rowsPerLoop;
        // 搬入第一块
        auto src0Offset = (cuS2 + cuRowsInLoop) * (gs1TileNum * dV);
        DataCopyPad(src0, dvWorkSpaceGm[src0Offset],
                    {static_cast<uint16_t>(realRowsPerLoop), static_cast<uint32_t>(dV * sizeof(float)),
                     /*srcStride*/ static_cast<uint32_t>((gs1TileNum * dV - dV) * sizeof(float)),
                     /*dstStride*/ static_cast<uint32_t>(dstStride), 0},
                    {true, 0, static_cast<uint8_t>(paddingRight), 0});

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        // if necessary, load rhs、do lhs = lhs + rhs
        for (int gS1Idx = 1; gS1Idx < gs1TileNum; ++gS1Idx) {
            if (gS1Idx > 1) {
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            }

            auto src1Offset = src0Offset + gS1Idx * dV;
            DataCopyPad(src1, dvWorkSpaceGm[src1Offset],
                        {static_cast<uint16_t>(realRowsPerLoop), static_cast<uint32_t>(dV * sizeof(float)),
                         /*srcStride*/ static_cast<uint32_t>((gs1TileNum * dV - dV) * sizeof(float)),
                         /*dstStride*/ static_cast<uint32_t>(dstStride), 0},
                        {true, 0, static_cast<uint8_t>(paddingRight), 0});

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            PipeBarrier<PIPE_V>();
            Add(src0, src0, src1, realRowsPerLoop * dVAligned);

            if (gS1Idx < gs1TileNum - 1) {
                SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            }
        }

        PipeBarrier<PIPE_V>();
        Cast(out, src0, RoundMode::CAST_ROUND, realRowsPerLoop * dVAligned);

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        // 搬出
        auto outOffset = (cuS2 + cuRowsInLoop) * dV;
        DataCopyPad(dvGm[outOffset], out,
                    {static_cast<uint16_t>(realRowsPerLoop), static_cast<uint32_t>(dV * sizeof(T1)),
                     /*srcStride*/ static_cast<uint32_t>((dVAligned - dV) / BLOCK_HALF_ELE_NUM), /*dstStride*/ 0, 0});

        if (loopIdx < loopCount - 1) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }
    pipe_barrier(PIPE_ALL);
    ProcessDsink();
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionGradDkvReducePost<T1, T2>::ProcessDsink()
{
    if (!tilingData->s1s2BNGS1S2BaseParams.sinkRequiresGrad) {
        return;
    }
    SyncAll();
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TQue<QuePosition::VECIN, 1> tmpQueue;
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

#endif // _FLASH_ATTENTION_GRAD_WS_DKV_REDUCE_POST_KERNEL_H_
