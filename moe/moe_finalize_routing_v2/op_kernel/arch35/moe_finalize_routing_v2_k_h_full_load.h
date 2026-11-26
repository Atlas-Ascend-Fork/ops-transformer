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
 * \file moe_finalize_routing_v2_k_h_full_load.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_K_H_FULL_LOAD_H_
#define MOE_FINALIZE_ROUTING_V2_K_H_FULL_LOAD_H_

#include "moe_finalize_routing_v2_base.h"
namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;

constexpr int64_t BATCH_COPY_SIZE = 16;
constexpr int64_t BATCH_COPY_EXPERT_NUM = 4;

template <typename T, typename S, int32_t dropPadMode>
class MoeFinalizeRoutingV2KHFullLoad {
public:
    __aicore__ inline MoeFinalizeRoutingV2KHFullLoad() {}

    // expandedX/bias k*h全载，需要fork次搬入
    __aicore__ inline void Init(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR x, GM_ADDR constExpertAlpha1,
                                GM_ADDR constExpertAlpha2, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                const MoeFinalizeRoutingV2RegbaseTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        k = tilingData->k;
        h = tilingData->h;
        hAligned = tilingData->hAligned;
        khAligned = k * hAligned;
        hasX1 = x1 != nullptr;
        hasX2 = x2 != nullptr;
        hasExpertIdx = (expertIdx != nullptr);
        hasBiasAndExpertIdx = (bias != nullptr) && (expertIdx != nullptr);
        hasScales = scales != nullptr;
        hasX = (x != nullptr);
        hasConstExpert = (constExpertAlpha1 != nullptr) && (constExpertAlpha2 != nullptr) && (v != nullptr);
        k1 = k == 1;
        k4 = k <= BATCH_COPY_EXPERT_NUM;

        SetGlobalBuffers(x1, x2, bias, scales, expertIdx, y, expandedX, x, constExpertAlpha1, constExpertAlpha2, v,
                         expandedRowIdx);
        InitQueBuffers();
    }

    __aicore__ inline void Process()
    {
        int64_t tailRowFactor = (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                       tilingData->tailRowFactorOfFormerBlock;
        int64_t rowOuterLoop =
            (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx += 1) {
            int64_t rowInnerLoop = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            ProcessInputWithX(rowOuterIdx, rowInnerLoop);
            ProcessWithId(rowOuterIdx, rowInnerLoop);

            yLocal = yQue.AllocTensor<float>();
            ProcessX1AndX2(yLocal, x1Local, x2Local, rowInnerLoop * h, hasX1, hasX2);
            if (k1) {
                ProcessYWithInputK1(rowOuterIdx, rowInnerLoop);
            } else if (k4 && tilingData->rowFactor >= BATCH_COPY_SIZE) {
                ProcessYWithInputK4(rowOuterIdx, rowInnerLoop);
            } else {
                ProcessYWithInput(rowOuterIdx, rowInnerLoop);
            }

            yQue.EnQue(yLocal);

            if (hasX1) {
                x1Que.FreeTensor(x1Local);
            }
            if (hasX2) {
                x2Que.FreeTensor(x2Local);
            }
            if (hasScales) {
                scalesQue.FreeTensor(scalesLocal);
            }

            if (k1) {
                expandedRowIdxQue.FreeTensor(expandedRowIdxLocal);
                if (hasBiasAndExpertIdx) {
                    expertIdxQue.FreeTensor(expertIdxLocal);
                }
            }
            yLocal = yQue.DeQue<float>();
            int64_t yGmOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * h + rowOuterIdx * tilingData->rowFactor * h;
            CopyOut(yLocal.template ReinterpretCast<T>(), yGm[yGmOffset], 1, rowInnerLoop * h);
            yQue.FreeTensor(yLocal);
        }
    }

private:
    __aicore__ inline void SetGlobalBuffers(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR scales, GM_ADDR expertIdx,
                                            GM_ADDR y, GM_ADDR expandedX, GM_ADDR x, GM_ADDR constExpertAlpha1,
                                            GM_ADDR constExpertAlpha2, GM_ADDR v, GM_ADDR expandedRowIdx)
    {
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        biasGm.SetGlobalBuffer((__gm__ T*)bias);
        scalesGm.SetGlobalBuffer((__gm__ S*)scales);
        expertIdxGm.SetGlobalBuffer((__gm__ int32_t*)expertIdx);
        yGm.SetGlobalBuffer((__gm__ T*)y);
        expandedXGm.SetGlobalBuffer((__gm__ T*)expandedX);
        xGm.SetGlobalBuffer((__gm__ T*)x);
        constExpertAlpha1Gm.SetGlobalBuffer((__gm__ T*)constExpertAlpha1);
        constExpertAlpha2Gm.SetGlobalBuffer((__gm__ T*)constExpertAlpha2);
        vGm.SetGlobalBuffer((__gm__ T*)v);
        expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx);
    }

    __aicore__ inline void InitQueBuffers()
    {
        int32_t rowFactorHAlignedT = RoundUp<T>(tilingData->rowFactor * h);
        int32_t rowFactorHAlignedFloat = RoundUp<float>(tilingData->rowFactor * h);
        int32_t rowFactorKAlignedT = RoundUp<T>(tilingData->rowFactor * k);
        int32_t rowFactorKAlignedTInt32 = RoundUp<int32_t>(tilingData->rowFactor * k);
        if (k1) {
            pipe->InitBuffer(expandedXQue, DOUBLE_BUFFER, tilingData->rowFactor * hAligned * sizeof(T));
            pipe->InitBuffer(expandedRowIdxQue, DOUBLE_BUFFER, rowFactorKAlignedTInt32 * sizeof(int32_t));
        } else if (k4 && tilingData->rowFactor >= BATCH_COPY_SIZE) {
            pipe->InitBuffer(expandedXQue, DOUBLE_BUFFER, BATCH_COPY_SIZE * k * hAligned * sizeof(T));
        } else {
            pipe->InitBuffer(expandedXQue, DOUBLE_BUFFER, k * hAligned * sizeof(T));
        }
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, rowFactorHAlignedFloat * sizeof(float));
        if (hasBiasAndExpertIdx) {
            if (k1) {
                pipe->InitBuffer(biasQue, DOUBLE_BUFFER, tilingData->rowFactor * hAligned * sizeof(T));
                pipe->InitBuffer(expertIdxQue, DOUBLE_BUFFER, rowFactorKAlignedTInt32 * sizeof(int32_t));
            } else if (k4 && tilingData->rowFactor >= BATCH_COPY_SIZE) {
                pipe->InitBuffer(biasQue, DOUBLE_BUFFER, BATCH_COPY_SIZE * k * hAligned * sizeof(T));
            } else {
                pipe->InitBuffer(biasQue, DOUBLE_BUFFER, k * hAligned * sizeof(T));
            }
        }
        if (hasX1) {
            pipe->InitBuffer(x1Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasX2) {
            pipe->InitBuffer(x2Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasScales) {
            pipe->InitBuffer(scalesQue, DOUBLE_BUFFER, rowFactorKAlignedT * sizeof(S));
        }
        if (hasX) {
            pipe->InitBuffer(xBuf, h * sizeof(T));
        }
        if (hasConstExpert) {
            pipe->InitBuffer(constExpertAlpha1Buf, h * sizeof(T));
            pipe->InitBuffer(constExpertAlpha2Buf, h * sizeof(T));
            pipe->InitBuffer(vBuf, h * sizeof(T));
        }
    }

    __aicore__ inline void ProcessWithId(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        // k == 1
        if (k1) {
            expandedRowIdxLocal = expandedRowIdxQue.AllocTensor<int32_t>();
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor;
            CopyIn(expandedRowIdxGm[expandedRowIdxOffset], expandedRowIdxLocal, 1, rowInnerLoop);
            expandedRowIdxQue.EnQue(expandedRowIdxLocal);
            expandedRowIdxLocal = expandedRowIdxQue.DeQue<int32_t>();

            if (hasBiasAndExpertIdx) {
                expertIdxLocal = expertIdxQue.AllocTensor<int32_t>();
                expertIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor;
                CopyIn(expertIdxGm[expertIdxOffset], expertIdxLocal, 1, rowInnerLoop);
                expertIdxQue.EnQue(expertIdxLocal);
                expertIdxLocal = expertIdxQue.DeQue<int32_t>();
            }
        }
    }

    __aicore__ inline void ProcessInputWithX(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        if (hasX1) {
            x1Local = x1Que.AllocTensor<T>();
            int64_t x1GmOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * h + rowOuterIdx * tilingData->rowFactor * h;
            CopyIn(x1Gm[x1GmOffset], x1Local, 1, rowInnerLoop * h);
            x1Que.EnQue(x1Local);
            x1Local = x1Que.DeQue<T>();
        }
        if (hasScales) {
            scalesLocal = scalesQue.AllocTensor<S>();
            int64_t scaleGmOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * k + rowOuterIdx * tilingData->rowFactor * k;
            CopyIn(scalesGm[scaleGmOffset], scalesLocal, 1, rowInnerLoop * k);
            scalesQue.EnQue(scalesLocal);
            scalesLocal = scalesQue.DeQue<S>();
        }
        if (hasX2) {
            x2Local = x2Que.AllocTensor<T>();
            int64_t x2GmOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * h + rowOuterIdx * tilingData->rowFactor * h;
            CopyIn(x2Gm[x2GmOffset], x2Local, 1, rowInnerLoop * h);
            x2Que.EnQue(x2Local);
            x2Local = x2Que.DeQue<T>();
        }
    }

    __aicore__ inline void LoadBatchAsync(int64_t rowOuterIdx, int64_t batchStart, int64_t batchSize)
    {
        expandedXLocal = expandedXQue.AllocTensor<T>();
        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.AllocTensor<T>();
        }

        int64_t offset = 0;
        for (int64_t i = 0; i < batchSize; i++) {
            if (hasX) {
                xLocal = xBuf.Get<T>();
            }
            if (hasConstExpert) {
                constExpertAlpha1Local = constExpertAlpha1Buf.Get<T>();
                constExpertAlpha2Local = constExpertAlpha2Buf.Get<T>();
                vLocal = vBuf.Get<T>();
            }

            int64_t rowInnerIdx = batchStart + i;
            ProcessExpandedXBiasAndScale(rowOuterIdx, rowInnerIdx, offset);
            validKArr[i] = validK;
            offset += tilingData->k * hAligned;
        }

        if (hasBiasAndExpertIdx) {
            biasQue.EnQue(biasLocal);
        }

        expandedXQue.EnQue(expandedXLocal);
    }

    __aicore__ inline void ComputeBatch(int64_t rowOuterIdx, int64_t batchStart, int64_t batchSize)
    {
        int64_t offset = 0;
        expandedXLocal = expandedXQue.DeQue<T>();

        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.DeQue<T>();
        }

        for (int64_t i = 0; i < batchSize; i++) {
            int64_t rowInnerIdx = batchStart + i;

            LocalTensor<S> innerScaleLocal;
            if (hasScales) {
                innerScaleLocal = scalesLocal[rowInnerIdx * tilingData->k];
            }

            ProcessExpandXBiasScaleOptimized<T, S>(yLocal[rowInnerIdx * tilingData->h], expandedXLocal[offset],
                                                   biasLocal[offset], innerScaleLocal, validKArr[i], tilingData->h,
                                                   hasBiasAndExpertIdx, hasScales);

            offset += tilingData->k * hAligned;
        }

        if (hasBiasAndExpertIdx) {
            biasQue.FreeTensor(biasLocal);
        }
        expandedXQue.FreeTensor(expandedXLocal);
    }

    __aicore__ inline void ProcessYWithInputK4(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        constexpr int64_t BATCH_SIZE = BATCH_COPY_SIZE;
        int64_t numBatches = (rowInnerLoop + BATCH_SIZE - 1) / BATCH_SIZE;

        if (numBatches > 0) {
            int64_t batchStart = 0;
            int64_t batchSize = min(BATCH_SIZE, rowInnerLoop - batchStart);
            LoadBatchAsync(rowOuterIdx, batchStart, batchSize);
        }

        for (int64_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            int64_t batchStart = batchIdx * BATCH_SIZE;
            int64_t batchSize = min(BATCH_SIZE, rowInnerLoop - batchStart);

            ComputeBatch(rowOuterIdx, batchStart, batchSize);

            if (batchIdx + 1 < numBatches) {
                int64_t nextBatchStart = (batchIdx + 1) * BATCH_SIZE;
                int64_t nextBatchSize = min(BATCH_SIZE, rowInnerLoop - nextBatchStart);
                LoadBatchAsync(rowOuterIdx, nextBatchStart, nextBatchSize);
            }
        }

        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, rowInnerLoop * tilingData->h);
        }
    }

    __aicore__ inline void AllocTensorsForProcessY()
    {
        expandedXLocal = expandedXQue.AllocTensor<T>();
        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.AllocTensor<T>();
        }
        if (hasX) {
            xLocal = xBuf.Get<T>();
        }
        if (hasConstExpert) {
            constExpertAlpha1Local = constExpertAlpha1Buf.Get<T>();
            constExpertAlpha2Local = constExpertAlpha2Buf.Get<T>();
            vLocal = vBuf.Get<T>();
        }
    }

    __aicore__ inline void EnQueDeQueTensors()
    {
        expandedXQue.EnQue(expandedXLocal);
        expandedXLocal = expandedXQue.DeQue<T>();
        if (hasBiasAndExpertIdx) {
            biasQue.EnQue(biasLocal);
            biasLocal = biasQue.DeQue<T>();
        }
    }

    __aicore__ inline void FreeTensorsForProcessY()
    {
        expandedXQue.FreeTensor(expandedXLocal);
        if (hasBiasAndExpertIdx) {
            biasQue.FreeTensor(biasLocal);
        }
    }

    __aicore__ inline void ProcessYWithInput(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        for (int64_t rowInnerIdx = 0; rowInnerIdx < rowInnerLoop; rowInnerIdx += 1) {
            AllocTensorsForProcessY();
            ProcessExpandedXBiasAndScale(rowOuterIdx, rowInnerIdx, 0);
            EnQueDeQueTensors();
            LocalTensor<S> innerScaleLocal;
            if (hasScales) {
                event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(eventId);
                WaitFlag<HardEvent::S_V>(eventId);
                innerScaleLocal = scalesLocal[rowInnerIdx * tilingData->k];
            }
            ProcessExpandXBiasScaleOptimized<T, S>(yLocal[rowInnerIdx * tilingData->h], expandedXLocal, biasLocal,
                                                   innerScaleLocal, validK, tilingData->h, hasBiasAndExpertIdx,
                                                   hasScales);
            FreeTensorsForProcessY();
        }
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, rowInnerLoop * tilingData->h);
        }
    }

    __aicore__ inline bool IsValidExpandedRowIdx(int64_t expandedRowIdxValue)
    {
        if (expandedRowIdxValue == INVALID_IDX) {
            return false;
        }
        if constexpr (dropPadMode != DROP_PAD_COLUMN && dropPadMode != DROP_PAD_ROW) {
            if (expandedRowIdxValue >= tilingData->activeNum) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline int64_t GetExpertIdxValue(int64_t rowInnerIdx, int64_t rowOuterIdx, int64_t kIdx)
    {
        if (!hasExpertIdx) {
            return -1;
        }
        SetOffsetOfExpertIdx(rowInnerIdx, rowOuterIdx, kIdx);
        return expertIdxGm.GetValue(expertIdxOffset);
    }

    __aicore__ inline bool LoadExpandedXForK(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t expertIdx,
                                             int64_t expandedRowIdxGmValue, int64_t offset)
    {
        if (hasX && hasExpertIdx) {
            if (expertIdx >= tilingData->zeroExpertStart && expertIdx < tilingData->zeroExpertEnd) {
                return false;
            }
            if (expertIdx >= tilingData->copyExpertStart && expertIdx < tilingData->copyExpertEnd) {
                // x = x[i]
                int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * h +
                                    rowOuterIdx * tilingData->rowFactor * h + rowInnerIdx * h;
                CopyIn(xGm[xGmOffset], expandedXLocal[offset + validK * tilingData->hAligned], 1, tilingData->h);
            } else if (hasConstExpert && expertIdx >= tilingData->constantExpertStart &&
                       expertIdx < tilingData->constantExpertEnd) {
                // x = alpha1 * x[i] +  alpha2 * v
                int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * h +
                                    rowOuterIdx * tilingData->rowFactor * h + rowInnerIdx * h;
                CopyIn(xGm[xGmOffset], xLocal, 1, tilingData->h);
                // 不需要有偏移，用完就下一个循环覆盖掉就行
                int64_t constExpertGmOffset = (expertIdx - tilingData->constantExpertStart) * h;
                CopyIn(constExpertAlpha1Gm[constExpertGmOffset], constExpertAlpha1Local, 1, tilingData->h);
                CopyIn(constExpertAlpha2Gm[constExpertGmOffset], constExpertAlpha2Local, 1, tilingData->h);
                CopyIn(vGm[constExpertGmOffset], vLocal, 1, tilingData->h);
                event_t mte2v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(mte2v);
                WaitFlag<HardEvent::MTE2_V>(mte2v);
                vLocal = vLocal * constExpertAlpha2Local;
                xLocal = xLocal * constExpertAlpha1Local;
                xLocal = xLocal + vLocal;
                AscendC::Copy(expandedXLocal[offset + validK * tilingData->hAligned], xLocal, tilingData->h);
                event_t v2mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(v2mte);
                WaitFlag<HardEvent::V_MTE2>(v2mte);
            } else {
                CopyIn(expandedXGm[expandedRowIdxGmValue * tilingData->h],
                       expandedXLocal[offset + validK * tilingData->hAligned], 1, tilingData->h);
            }
        } else {
            CopyIn(expandedXGm[expandedRowIdxGmValue * tilingData->h],
                   expandedXLocal[offset + validK * tilingData->hAligned], 1, tilingData->h);
        }
        return true;
    }

    __aicore__ inline void ProcessExpandedXBiasAndScale(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t offset)
    {
        validK = 0;
        for (int64_t kIdx = 0; kIdx < tilingData->k; kIdx += 1) {
            SetExpandedRowIdxOffset(rowOuterIdx, rowInnerIdx, kIdx);
            int64_t expandedRowIdxGmValue = expandedRowIdxGm.GetValue(expandedRowIdxOffset);
            if (!IsValidExpandedRowIdx(expandedRowIdxGmValue)) {
                continue;
            }
            int64_t expertIdx = GetExpertIdxValue(rowInnerIdx, rowOuterIdx, kIdx);
            if (!LoadExpandedXForK(rowOuterIdx, rowInnerIdx, expertIdx, expandedRowIdxGmValue, offset)) {
                continue;
            }

            if (hasBiasAndExpertIdx) {
                int64_t biasGmOffset = expertIdxGm.GetValue(expertIdxOffset) * tilingData->h;
                CopyIn(biasGm[biasGmOffset], biasLocal[offset + validK * tilingData->hAligned], 1, tilingData->h);
            }
            // 由于会将k轴融到VF中，所以drop_pad场景需要将有效的Scale按序设置
            if (hasScales) {
                S scaleValue = scalesLocal[rowInnerIdx * tilingData->k].GetValue(kIdx);
                scalesLocal[rowInnerIdx * tilingData->k].SetValue(validK, scaleValue);
            }
            validK += 1;
        }
    }

    __aicore__ inline void ProcessYWithInputK1(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        expandedXLocal = expandedXQue.AllocTensor<T>();
        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.AllocTensor<T>();
        }
        if (hasX) {
            xLocal = xBuf.Get<T>();
        }
        if (hasConstExpert) {
            constExpertAlpha1Local = constExpertAlpha1Buf.Get<T>();
            constExpertAlpha2Local = constExpertAlpha2Buf.Get<T>();
            vLocal = vBuf.Get<T>();
        }
        event_t sWaitMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(sWaitMte2);
        WaitFlag<HardEvent::MTE2_S>(sWaitMte2);
        int64_t hOffset = 0;
        int64_t kOffset = 0;
        int64_t khAlignedOffset = 0;
        for (int64_t rowInnerIdx = 0; rowInnerIdx < rowInnerLoop; rowInnerIdx += 1) {
            ProcessExpandedXBiasAndScaleK1(rowOuterIdx, rowInnerIdx, kOffset, khAlignedOffset);
            expandedXQue.EnQue(expandedXLocal);
            expandedXLocal = expandedXQue.DeQue<T>();
            if (hasBiasAndExpertIdx) {
                biasQue.EnQue(biasLocal);
                biasLocal = biasQue.DeQue<T>();
            }
            ProcessExpandXBiasScaleOptimized<T, S>(yLocal[hOffset], expandedXLocal[khAlignedOffset],
                                                   biasLocal[khAlignedOffset], scalesLocal[kOffset], 1, h,
                                                   hasBiasAndExpertIdx, hasScales);
            hOffset += h;
            kOffset += 1;
            khAlignedOffset += khAligned;
        }

        expandedXQue.FreeTensor(expandedXLocal);
        if (hasBiasAndExpertIdx) {
            biasQue.FreeTensor(biasLocal);
        }
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, rowInnerLoop * h);
        }
    }

    __aicore__ inline void ProcessExpandedXBiasAndScaleK1(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kOffset,
                                                          int64_t khAlignedOffset)
    {
        int64_t expandedRowIdxValue = expandedRowIdxLocal.GetValue(rowInnerIdx);
        if (expandedRowIdxValue == INVALID_IDX) {
            return;
        }
        if constexpr (dropPadMode != DROP_PAD_COLUMN && dropPadMode != DROP_PAD_ROW) {
            if (expandedRowIdxValue >= tilingData->activeNum) {
                return;
            }
        }
        int64_t expertIdx = -1;
        if (hasExpertIdx) {
            expertIdx = expertIdxLocal.GetValue(rowInnerIdx);
        }
        if (hasX && hasExpertIdx) {
            if (expertIdx >= tilingData->zeroExpertStart && expertIdx < tilingData->zeroExpertEnd) {
                // 直接不计算
                return;
            }
            if (expertIdx >= tilingData->copyExpertStart && expertIdx < tilingData->copyExpertEnd) {
                // x = x[i]
                int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                    rowOuterIdx * tilingData->rowFactor * tilingData->h + rowInnerIdx * tilingData->h;
                CopyIn(xGm[xGmOffset], expandedXLocal[khAlignedOffset], 1, tilingData->h);
            } else if (expertIdx >= tilingData->constantExpertStart && expertIdx < tilingData->constantExpertEnd) {
                // x = alpha1 * x[i] +  alpha2 * v
                // 不需要有偏移，用完就下一个循环覆盖掉就行
                int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                    rowOuterIdx * tilingData->rowFactor * tilingData->h + rowInnerIdx * tilingData->h;
                CopyIn(xGm[xGmOffset], xLocal, 1, tilingData->h);
                int64_t constExpertGmOffset = (expertIdx - tilingData->constantExpertStart) * tilingData->h;
                CopyIn(constExpertAlpha1Gm[constExpertGmOffset], constExpertAlpha1Local, 1, tilingData->h);
                CopyIn(constExpertAlpha2Gm[constExpertGmOffset], constExpertAlpha2Local, 1, tilingData->h);
                CopyIn(vGm[constExpertGmOffset], vLocal, 1, tilingData->h);
                event_t mte2v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(mte2v);
                WaitFlag<HardEvent::MTE2_V>(mte2v);
                vLocal = vLocal * constExpertAlpha2Local;
                xLocal = xLocal * constExpertAlpha1Local;
                xLocal = xLocal + vLocal;
                AscendC::Copy(expandedXLocal[khAlignedOffset], xLocal, tilingData->h);
                event_t v2mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(v2mte);
                WaitFlag<HardEvent::V_MTE2>(v2mte);
            } else {
                CopyIn(expandedXGm[expandedRowIdxValue * h], expandedXLocal[khAlignedOffset], 1, h);
            }
        } else {
            CopyIn(expandedXGm[expandedRowIdxValue * h], expandedXLocal[khAlignedOffset], 1, h);
        }
        if (hasBiasAndExpertIdx) {
            int64_t biasGmOffset = expertIdxLocal.GetValue(rowInnerIdx) * h;
            CopyIn(biasGm[biasGmOffset], biasLocal[khAlignedOffset], 1, h);
        }

        if (hasScales) {
            S scaleValue = scalesLocal[kOffset].GetValue(0);
            scalesLocal[kOffset].SetValue(0, scaleValue);
        }
    }

    __aicore__ inline void SetExpandedRowIdxOffset(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor +
                                   rowInnerIdx + kIdx * tilingData->row;
        } else {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                                   rowOuterIdx * tilingData->rowFactor * tilingData->k + rowInnerIdx * tilingData->k +
                                   kIdx;
        }
    }

    __aicore__ inline void SetOffsetOfExpertIdx(int64_t rowInnerIdx, int64_t rowOuterIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROP_PAD_COLUMN || dropPadMode == DROPLESS_COLUMN) {
            expertIdxOffset = rowOuterIdx * tilingData->rowFactor * k +
                              GetBlockIdx() * tilingData->rowOfFormerBlock * k + rowInnerIdx * k + kIdx;
        } else {
            expertIdxOffset = expandedRowIdxOffset;
        }
    }

    TPipe* pipe;
    const MoeFinalizeRoutingV2RegbaseTilingData* tilingData;

    LocalTensor<T> expandedXLocal;
    LocalTensor<int32_t> expandedRowIdxLocal;
    LocalTensor<T> x1Local;
    LocalTensor<T> x2Local;
    LocalTensor<T> biasLocal;
    LocalTensor<int32_t> expertIdxLocal;

    LocalTensor<T> xLocal;
    LocalTensor<T> constExpertAlpha1Local;
    LocalTensor<T> constExpertAlpha2Local;
    LocalTensor<T> vLocal;

    LocalTensor<S> scalesLocal;
    LocalTensor<float> yLocal;

    bool hasX1{false};
    bool hasX2{false};
    bool hasBiasAndExpertIdx{false};
    bool hasExpertIdx{false};
    bool hasScales{false};
    bool k1{false};
    bool k4{false};
    bool hasX{false};
    bool hasConstExpert{false};

    int16_t validK{0};
    int16_t validKArr[BATCH_COPY_SIZE]{0}; // k <= 4使用
    int64_t expandedRowIdxOffset{0};
    int64_t expertIdxOffset{0};
    int64_t k{0};
    int64_t h{0};
    int64_t hAligned{0};
    int64_t khAligned{0};

    GlobalTensor<T> expandedXGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> biasGm;
    GlobalTensor<S> scalesGm;
    GlobalTensor<int32_t> expertIdxGm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> constExpertAlpha1Gm;
    GlobalTensor<T> constExpertAlpha2Gm;
    GlobalTensor<T> vGm;
    GlobalTensor<T> yGm;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedXQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedRowIdxQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> biasQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expertIdxQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x2Que;
    TBuf<QuePosition::VECCALC> xBuf;
    TBuf<QuePosition::VECCALC> constExpertAlpha1Buf;
    TBuf<QuePosition::VECCALC> constExpertAlpha2Buf;
    TBuf<QuePosition::VECCALC> vBuf;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> scalesQue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQue;
};

#ifdef MOE_FINALIZE_ROUTING_HAS_MXFP4
// MXFP4 specialization: BF16 output and FP32 accumulation.
template <int32_t dropPadMode>
class MoeFinalizeRoutingV2KHFullLoad<fp4x2_e2m1_t, float8_e8m0_t, dropPadMode>
    : public MoeFinalizeRoutingV2Mx4Base<dropPadMode> {
    using TYPE = bfloat16_t;

public:
    __aicore__ inline void Init(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR routingWeights, GM_ADDR expertIdx, GM_ADDR y,
                                GM_ADDR /*workspace*/, const MoeFinalizeRoutingV2RegbaseTilingData* data, TPipe* pipe)
    {
        this->InitMx4Inputs(expandedX, expandedRowIdx, x1, x2, bias, scales, routingWeights, expertIdx, y, data, pipe);
        const int64_t rows = this->d_->kFactor;
        this->pipe_->InitBuffer(this->packedQue_, MX4_QUEUE_DEPTH, rows * this->packedStride_);
        this->pipe_->InitBuffer(this->scaleQue_, MX4_QUEUE_DEPTH, rows * this->scaleStride_);
        if (this->hasBias_) {
            this->pipe_->InitBuffer(this->biasQue_, MX4_QUEUE_DEPTH, rows * this->auxStride_ * sizeof(TYPE));
        }
    }

    __aicore__ inline void Process()
    {
        const int64_t start = GetBlockIdx() * this->d_->rowOfFormerBlock;
        const int64_t count =
            GetBlockIdx() == GetBlockNum() - 1 ? this->d_->rowOfTailBlock : this->d_->rowOfFormerBlock;
        for (int64_t row = start; row < start + count; ++row) {
            this->ComputeMx4BatchTile(row, 0, static_cast<uint32_t>(this->d_->h));
        }
    }
};
#endif

#ifdef MOE_FINALIZE_ROUTING_HAS_MXFP8
// MXFP8 specializations: BF16 output and FP32 accumulation.
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

template <typename T, int32_t DROP_PAD_MODE>
class MoeFinalizeRoutingV2Mx8KHFullLoadImpl : public MoeFinalizeRoutingV2Mx8RouteBase<DROP_PAD_MODE> {
public:
    __aicore__ MoeFinalizeRoutingV2Mx8KHFullLoadImpl() {}

    __aicore__ void Init(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                         GM_ADDR expandedXScale, GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR x, GM_ADDR alpha1,
                         GM_ADDR alpha2, GM_ADDR v, GM_ADDR y, GM_ADDR /*workspace*/,
                         const MoeFinalizeRoutingV2RegbaseTilingData* data, TPipe* pipe)
    {
        pipe_ = pipe;
        data_ = data;
        coreIdx_ = GetBlockIdx();
        coreNum_ = GetBlockNum();
        isTailCore_ = (coreIdx_ == coreNum_ - 1);
        this->InitMx8Optional(data, pipe, expandedRowIdx, expertIdx, bias, scales, static_cast<uint32_t>(data_->k));
        hasX1_ = x1 != nullptr;
        hasX2_ = x2 != nullptr;
        hasBiasAndExpertIdx_ = (bias != nullptr) && (expertIdx != nullptr);
        expandedXGm_.SetGlobalBuffer((__gm__ uint8_t*)expandedX);
        x1Gm_.SetGlobalBuffer((__gm__ bfloat16_t*)x1);
        x2Gm_.SetGlobalBuffer((__gm__ bfloat16_t*)x2);
        biasGm_.SetGlobalBuffer((__gm__ bfloat16_t*)bias);
        scalesGm_.SetGlobalBuffer((__gm__ uint8_t*)expandedXScale);
        yGm_.SetGlobalBuffer((__gm__ bfloat16_t*)y);

        scalesRowElemNum_ = Mx8ScaleStride(data_->h);
        scalesRowLenAligned_ = Mx8ScaleRowAligned(data_->h);
        scaleNumAligned_ =
            static_cast<uint32_t>(CeilAlign(scalesRowElemNum_, static_cast<int64_t>(MX8_UB_F32_PER_BLOCK)));
        auto scalesLen = static_cast<int64_t>(data_->k) * scalesRowLenAligned_;
        // 每路由槽位（expandedX/bias）按 hAligned 步进保证 32B 对齐；x1/x2/y 批量缓冲按真实 H。
        auto expandedXLen = static_cast<uint32_t>(data_->k * data_->hAligned);
        auto biasLen = static_cast<uint32_t>(data_->k * data_->hAligned * sizeof(bfloat16_t));
        auto rowsBf16Len = static_cast<uint32_t>(data_->rowFactor * data_->h * sizeof(bfloat16_t));
        auto yLen = static_cast<uint32_t>(data_->rowFactor * data_->h * sizeof(float));
        auto scratchLen =
            static_cast<uint32_t>(CeilAlign(Mx8ScratchBytes(static_cast<uint32_t>(data_->hAligned), scaleNumAligned_),
                                            static_cast<uint32_t>(UBBLOCKSIZE)));

        pipe_->InitBuffer(expandedXQue_, DOUBLE_BUFFER, expandedXLen);
        pipe_->InitBuffer(scalesQue_, DOUBLE_BUFFER, static_cast<uint32_t>(scalesLen));
        if (hasX1_) {
            pipe_->InitBuffer(x1Que_, DOUBLE_BUFFER, rowsBf16Len);
        }
        if (hasX2_) {
            pipe_->InitBuffer(x2Que_, DOUBLE_BUFFER, rowsBf16Len);
        }
        if (hasBiasAndExpertIdx_) {
            pipe_->InitBuffer(biasQue_, DOUBLE_BUFFER, biasLen);
        }
        pipe_->InitBuffer(yQue_, DOUBLE_BUFFER, yLen);
        pipe_->InitBuffer(scratchBuf_, scratchLen);
        InitScaleIndex();
    }

    __aicore__ void Process()
    {
        int64_t rowOuterLoop = isTailCore_ ? data_->rowLoopOfTailBlock : data_->rowLoopOfFormerBlock;
        int64_t tailRowFactor = isTailCore_ ? data_->tailRowFactorOfTailBlock : data_->tailRowFactorOfFormerBlock;

        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; ++rowOuterIdx) {
            int64_t rowInnerLoop = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : data_->rowFactor;
            int64_t elemOffset = (coreIdx_ * data_->rowOfFormerBlock + rowOuterIdx * data_->rowFactor) * data_->h;
            int64_t elemLen = rowInnerLoop * data_->h;

            if (hasX1_) {
                LocalTensor<bfloat16_t> x1Local = x1Que_.AllocTensor<bfloat16_t>();
                CopyIn<bfloat16_t>(x1Gm_[elemOffset], x1Local, 1, elemLen);
                x1Que_.EnQue(x1Local);
            }
            if (hasX2_) {
                LocalTensor<bfloat16_t> x2Local = x2Que_.AllocTensor<bfloat16_t>();
                CopyIn<bfloat16_t>(x2Gm_[elemOffset], x2Local, 1, elemLen);
                x2Que_.EnQue(x2Local);
            }

            LocalTensor<float> yLocal = yQue_.AllocTensor<float>();
            ProcessX1AndX2Input(yLocal, static_cast<uint32_t>(elemLen));
            PipeBarrier<PIPE_V>();
            InnerLoopProcessInput(yLocal, rowOuterIdx, rowInnerLoop);
            Cast(yLocal.template ReinterpretCast<bfloat16_t>(), yLocal, RoundMode::CAST_RINT, elemLen);
            yQue_.EnQue(yLocal);

            yLocal = yQue_.DeQue<float>();
            CopyOut<bfloat16_t>(yLocal.template ReinterpretCast<bfloat16_t>(), yGm_[elemOffset], 1, elemLen);
            yQue_.FreeTensor(yLocal);
        }
    }

private:
    __aicore__ void InitScaleIndex()
    {
        LocalTensor<float> scratch = scratchBuf_.Get<float>();
        LocalTensor<float> values, biasValues, scaleExpanded, scaleF32;
        LocalTensor<uint32_t> scaleIndex;
        Mx8GetScratchTensors(scratch, static_cast<uint32_t>(data_->hAligned), scaleNumAligned_, values, biasValues,
                             scaleExpanded, scaleF32, scaleIndex);
        Mx8BuildScaleIndex(scaleIndex, static_cast<uint32_t>(data_->h));
    }

    __aicore__ void ProcessX1AndX2Input(const LocalTensor<float>& yLocal, uint32_t elemNum)
    {
        LocalTensor<bfloat16_t> x1Local, x2Local;
        if (hasX1_) {
            x1Local = x1Que_.DeQue<bfloat16_t>();
        }
        if (hasX2_) {
            x2Local = x2Que_.DeQue<bfloat16_t>();
        }
        ProcessX1AndX2<bfloat16_t>(yLocal, x1Local, x2Local, elemNum, hasX1_, hasX2_);
        if (hasX1_) {
            x1Que_.FreeTensor(x1Local);
        }
        if (hasX2_) {
            x2Que_.FreeTensor(x2Local);
        }
    }

    __aicore__ void InnerLoopProcessInput(const LocalTensor<float>& yLocal, int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        for (int64_t rowInnerIdx = 0; rowInnerIdx < rowInnerLoop; ++rowInnerIdx) {
            uint32_t validK = CopyInExpandedXScaleBias(rowOuterIdx, rowInnerIdx, yLocal[rowInnerIdx * data_->h]);
            ComputeDequantize(yLocal[rowInnerIdx * data_->h], validK);
        }
    }

    __aicore__ uint32_t CopyInExpandedXScaleBias(int64_t rowOuterIdx, int64_t rowInnerIdx,
                                                 const LocalTensor<float>& yRow)
    {
        LocalTensor<uint8_t> expandedXLocal = expandedXQue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> scalesLocal = scalesQue_.AllocTensor<uint8_t>();
        LocalTensor<bfloat16_t> biasLocal;
        if (hasBiasAndExpertIdx_) {
            biasLocal = biasQue_.AllocTensor<bfloat16_t>();
        }
        uint32_t validK = 0;
        const int64_t rowIndex = coreIdx_ * data_->rowOfFormerBlock + rowOuterIdx * data_->rowFactor + rowInnerIdx;
        for (int64_t kIdx = 0; kIdx < data_->k; ++kIdx) {
            int64_t gatherRowIdx, expert;
            if (!this->Route(rowIndex, kIdx, gatherRowIdx, expert, validK)) {
                continue;
            }
            // GM 按真实 H 搬入，UB 槽位按 hAligned 步进（对齐基址，槽内尾部为 padding）
            uint32_t expandedXRowBytes = static_cast<uint32_t>(data_->h);
            uint32_t expandedXSlotBytes = static_cast<uint32_t>(data_->hAligned);
            DataCopyPad<uint8_t>(
                expandedXLocal[validK * expandedXSlotBytes], expandedXGm_[gatherRowIdx * expandedXRowBytes],
                DataCopyExtParams{1, expandedXRowBytes, 0, 0, 0}, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
            DataCopyPad<uint8_t>(scalesLocal[validK * scalesRowLenAligned_],
                                 scalesGm_[gatherRowIdx * scalesRowElemNum_],
                                 DataCopyExtParams{1, static_cast<uint32_t>(scalesRowElemNum_), 0, 0, 0},
                                 DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
            if (hasBiasAndExpertIdx_) {
                int64_t biasGmOffset = expert * data_->h;
                CopyIn<bfloat16_t>(biasGm_[biasGmOffset], biasLocal[validK * data_->hAligned], 1, data_->h);
            }
            validK++;
        }
        expandedXQue_.EnQue(expandedXLocal);
        scalesQue_.EnQue(scalesLocal);
        if (hasBiasAndExpertIdx_) {
            biasQue_.EnQue(biasLocal);
        }
        return validK;
    }

    __aicore__ void ComputeDequantize(const LocalTensor<float>& yLocal, uint32_t validK)
    {
        LocalTensor<uint8_t> expandedXLocal = expandedXQue_.DeQue<uint8_t>();
        LocalTensor<uint8_t> scalesLocal = scalesQue_.DeQue<uint8_t>();
        Mx8WaitScaleCopyReady();
        LocalTensor<bfloat16_t> biasLocal;
        if (hasBiasAndExpertIdx_) {
            biasLocal = biasQue_.DeQue<bfloat16_t>();
        }
        LocalTensor<float> scratch = scratchBuf_.Get<float>();
        LocalTensor<float> values, biasValues, scaleExpanded, scaleF32;
        LocalTensor<uint32_t> scaleIndex;
        Mx8GetScratchTensors(scratch, static_cast<uint32_t>(data_->hAligned), scaleNumAligned_, values, biasValues,
                             scaleExpanded, scaleF32, scaleIndex);

        for (uint32_t r = 0; r < validK; ++r) {
            LocalTensor<uint8_t> xRow = expandedXLocal[r * data_->hAligned];
            LocalTensor<uint8_t> sRow = scalesLocal[r * scalesRowLenAligned_];
            LocalTensor<bfloat16_t> bRow;
            if (hasBiasAndExpertIdx_) {
                bRow = biasLocal[r * data_->hAligned];
            }
            Mx8DequantizeRow<T>(yLocal, xRow, sRow, bRow, values, biasValues, scaleExpanded, scaleF32, scaleIndex,
                                static_cast<uint32_t>(data_->h), hasBiasAndExpertIdx_, this->SlotWeight(r));
        }

        expandedXQue_.FreeTensor(expandedXLocal);
        scalesQue_.FreeTensor(scalesLocal);
        if (hasBiasAndExpertIdx_) {
            biasQue_.FreeTensor(biasLocal);
        }
    }

    TPipe* pipe_{nullptr};
    const MoeFinalizeRoutingV2RegbaseTilingData* data_{nullptr};
    int64_t coreIdx_{0};
    int64_t coreNum_{0};
    bool isTailCore_{false};

    GlobalTensor<uint8_t> expandedXGm_;
    GlobalTensor<bfloat16_t> x1Gm_;
    GlobalTensor<bfloat16_t> x2Gm_;
    GlobalTensor<bfloat16_t> biasGm_;
    GlobalTensor<uint8_t> scalesGm_;
    GlobalTensor<bfloat16_t> yGm_;

    TQue<QuePosition::VECIN, 1> expandedXQue_;
    TQue<QuePosition::VECIN, 1> biasQue_;
    TQue<QuePosition::VECIN, 1> x1Que_;
    TQue<QuePosition::VECIN, 1> x2Que_;
    TQue<QuePosition::VECIN, 1> scalesQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;
    TBuf<TPosition::VECCALC> scratchBuf_;

    bool hasX1_{false};
    bool hasX2_{false};
    bool hasBiasAndExpertIdx_{false};

    int64_t scalesRowLenAligned_{0};
    int64_t scalesRowElemNum_{0};
    uint32_t scaleNumAligned_{0};
};

template <int32_t DROP_PAD_MODE>
class MoeFinalizeRoutingV2KHFullLoad<fp8_e4m3fn_t, float8_e8m0_t, DROP_PAD_MODE>
    : public MoeFinalizeRoutingV2Mx8KHFullLoadImpl<fp8_e4m3fn_t, DROP_PAD_MODE> {};

template <int32_t DROP_PAD_MODE>
class MoeFinalizeRoutingV2KHFullLoad<fp8_e5m2_t, float8_e8m0_t, DROP_PAD_MODE>
    : public MoeFinalizeRoutingV2Mx8KHFullLoadImpl<fp8_e5m2_t, DROP_PAD_MODE> {};
#endif

} // namespace MoeFinalizeRoutingV2Regbase

#endif
