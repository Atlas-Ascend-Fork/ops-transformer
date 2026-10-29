/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _FLASH_ATTENTION_GRAD_WS_REDUCE_POST_Q_KERNEL_H_
#define _FLASH_ATTENTION_GRAD_WS_REDUCE_POST_Q_KERNEL_H_
#include "kernel_operator.h"

template <typename T1, typename T2>
class FlashAttentionGradDqReducePost {
public:
    __aicore__ inline FlashAttentionGradDqReducePost(){};
    __aicore__ inline void Init(__gm__ uint8_t* dq, __gm__ uint8_t* workspace,
                                const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData,
                                TPipe* pipe_in);
    __aicore__ inline void Process();

    TPipe* pipe;
    TBuf<> src0Queue;
    TBuf<> src1Queue;
    TBuf<> outQueue;

    GlobalTensor<T1> dqGm;
    // input
    GlobalTensor<float> dqWorkSpaceGm;

    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict tilingData;
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;

    uint32_t usedCoreNum;
    uint32_t cBlockIdx;

    // base info
    int64_t maxS2;
    uint32_t d;
    uint32_t b;
    int64_t Tq;
    int64_t Tkv;
    uint32_t n2;
    uint32_t g;
    uint32_t s2TileNum;
    uint32_t validS2TileNum;

    // fp32 src0 = fp32 src0 + fp32 src1
    // fp32 src0 = fp32 src0 * scale
    // bf16 src0 = cast(fp32 src0)
    uint32_t ubBaseSize = 30 * 1024; // 30kb

    // Tq N2 G S2_SPLIT D
    uint32_t numS1PerCore;
    uint32_t cuS1;

    // some constants
    const int BLOCK_HALF_ELE_NUM = 16;
    const int BLOCK_FLOAT_ELE_NUM = 8;

    // some events
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>());
    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE2>());
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
};

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionGradDqReducePost<T1, T2>::Init(
    __gm__ uint8_t* dq, __gm__ uint8_t* workspace,
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData, TPipe* pipe_in)
{
    cBlockIdx = GetBlockIdx();

    tilingData = ordTilingData;
    pipe = pipe_in;

    dqGm.SetGlobalBuffer((__gm__ T1*)dq);

    // 下面更新一些基本信息
    usedCoreNum = (tilingData->s1s2BNGS1S2SplitCoreParams.blockOuter + 1) / 2 * 2;
    // shape info
    b = tilingData->s1s2BNGS1S2BaseParams.b;
    n2 = tilingData->s1s2BNGS1S2BaseParams.n2;
    g = tilingData->s1s2BNGS1S2BaseParams.g;
    maxS2 = tilingData->s1s2BNGS1S2BaseParams.s2;
    d = tilingData->s1s2BNGS1S2BaseParams.d;

    auto s2CvRatio = tilingData->s1s2BNGS1S2SplitCoreParams.s2CvRatio;
    auto s2Inner = tilingData->s1s2BNGS1S2SplitCoreParams.s2Inner;
    auto s2CvInner = s2CvRatio * s2Inner;

    // 更新token总数
    Tq = tilingData->s1s2BNGS1S2BaseParams.Tq;
    Tkv = tilingData->s1s2BNGS1S2BaseParams.Tkv;
    s2TileNum = tilingData->s1s2BNGS1S2BaseParams.s2TileNum;
    // @lg tiling侧已经用max s2去计算了s2TileNum，也就是validS2TileNum
    validS2TileNum = s2TileNum;

    // 算一下，针对dqkv，多核每个核处理的token数量
    // @lg Tq N2 G合轴: 每个core负责处理这么多d
    auto numS1PerCoreTiling = (Tq * n2 * g + usedCoreNum - 1) / usedCoreNum;
    auto tailNumS1PerCoreTiling = numS1PerCoreTiling - 1;
    auto tailCoreNum = numS1PerCoreTiling * usedCoreNum - Tq * n2 * g;
    auto formerCoreNum = usedCoreNum - tailCoreNum;
    numS1PerCore = cBlockIdx >= formerCoreNum ? tailNumS1PerCoreTiling : numS1PerCoreTiling;
    cuS1 = cBlockIdx * numS1PerCoreTiling;
    if (cBlockIdx > formerCoreNum) {
        cuS1 = formerCoreNum * numS1PerCoreTiling + (cBlockIdx - formerCoreNum) * tailNumS1PerCoreTiling;
    }
    // 下面是dqkv worksapce起始位置
    auto workspaceOffsets = tilingData->s1s2BNGS1S2BaseParams.dqkvWorkspaceOffset;
    dqWorkSpaceGm.SetGlobalBuffer((__gm__ T2*)workspace + workspaceOffsets / sizeof(T2));

    pipe->InitBuffer(src0Queue, ubBaseSize * 2);
    pipe->InitBuffer(src1Queue, ubBaseSize * 2);
    pipe->InitBuffer(outQueue, ubBaseSize);
}

template <typename T1, typename T2>
__aicore__ inline void FlashAttentionGradDqReducePost<T1, T2>::Process()
{
    // 如果core没有分配到tokens，提前return
    if (numS1PerCore <= 0) {
        return;
    }
    // dq
    // Tq N2 G S2_SPLIT D
    auto dAligned = (d + BLOCK_HALF_ELE_NUM - 1) / BLOCK_HALF_ELE_NUM * BLOCK_HALF_ELE_NUM;
    auto qkHeadDimBytes = dAligned * sizeof(float);
    auto qkInbaseNumRows = 2 * ubBaseSize / qkHeadDimBytes;

    LocalTensor<float> src0 = src0Queue.Get<float>(qkInbaseNumRows * dAligned);
    LocalTensor<float> src1 = src1Queue.Get<float>(qkInbaseNumRows * dAligned);
    LocalTensor<T1> out = outQueue.Get<T1>(qkInbaseNumRows * dAligned);

    // ub能容纳的行数和实际该core分配的行数做个判断
    auto rowsPerLoop = qkInbaseNumRows <= numS1PerCore ? qkInbaseNumRows : numS1PerCore;
    auto loopCount = (numS1PerCore + rowsPerLoop - 1) / rowsPerLoop;
    auto tailRowsPerLoop = numS1PerCore - (loopCount - 1) * rowsPerLoop;

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
        auto src0Offset = (cuS1 + cuRowsInLoop) * (s2TileNum * d);

        DataCopyPad(src0, dqWorkSpaceGm[src0Offset],
                    {static_cast<uint16_t>(realRowsPerLoop), static_cast<uint32_t>(d * sizeof(float)),
                     /*srcStride*/ static_cast<uint32_t>((s2TileNum * d - d) * sizeof(float)),
                     /*dstStride*/ static_cast<uint32_t>(dstStride), 0},
                    {true, 0, static_cast<uint8_t>(paddingRight), 0});
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        // if necessary, load rhs、do lhs = lhs + rhs
        for (int s2Idx = 1; s2Idx < validS2TileNum; ++s2Idx) {
            if (s2Idx > 1) {
                WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
            }

            auto src1Offset = src0Offset + s2Idx * d;
            DataCopyPad(src1, dqWorkSpaceGm[src1Offset],
                        {static_cast<uint16_t>(realRowsPerLoop), static_cast<uint32_t>(d * sizeof(float)),
                         /*srcStride*/ static_cast<uint32_t>((s2TileNum * d - d) * sizeof(float)),
                         /*dstStride*/ static_cast<uint32_t>(dstStride), 0},
                        {true, 0, static_cast<uint8_t>(paddingRight), 0});

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            PipeBarrier<PIPE_V>();
            Add(src0, src0, src1, realRowsPerLoop * dAligned);

            if (s2Idx < validS2TileNum - 1) {
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
        auto outOffset = (cuS1 + cuRowsInLoop) * d;
        DataCopyPad(dqGm[outOffset], out,
                    {static_cast<uint16_t>(realRowsPerLoop), static_cast<uint32_t>(d * sizeof(T1)),
                     /*srcStride*/ static_cast<uint32_t>((dAligned - d) / BLOCK_HALF_ELE_NUM), /*dstStride*/ 0, 0});

        if (loopIdx < loopCount - 1) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }
}

#endif // _FLASH_ATTENTION_GRAD_WS_REDUCE_POST_DQ_KERNEL_H_
