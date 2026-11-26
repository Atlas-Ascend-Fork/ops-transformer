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
 * \file moe_finalize_routing_v2_h_full_load.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_H_FULL_LOAD_H_
#define MOE_FINALIZE_ROUTING_V2_H_FULL_LOAD_H_

#include "moe_finalize_routing_v2_base.h"

namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;
template <typename T, typename S, int32_t dropPadMode>
class MoeFinalizeRoutingV2HFullLoad {
public:
    __aicore__ inline MoeFinalizeRoutingV2HFullLoad() {}

    __aicore__ inline void Init(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR x, GM_ADDR constExpertAlpha1,
                                GM_ADDR constExpertAlpha2, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                const MoeFinalizeRoutingV2RegbaseTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        hasX1 = x1 != nullptr;
        hasX2 = x2 != nullptr;
        hasBiasAndExpertIdx = (bias != nullptr) && (expertIdx != nullptr);
        hasExpertIdx = (expertIdx != nullptr);
        hasScales = scales != nullptr;
        hasX = x != nullptr;
        hasConstExpert = (constExpertAlpha1 != nullptr) && (constExpertAlpha2 != nullptr) && (v != nullptr);

        expandedXGm.SetGlobalBuffer((__gm__ T*)expandedX);
        expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx);
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);

        biasGm.SetGlobalBuffer((__gm__ T*)bias);
        scalesGm.SetGlobalBuffer((__gm__ S*)scales);
        expertIdxGm.SetGlobalBuffer((__gm__ int32_t*)expertIdx);
        xGm.SetGlobalBuffer((__gm__ T*)x);
        constExpertAlpha1Gm.SetGlobalBuffer((__gm__ T*)constExpertAlpha1);
        constExpertAlpha2Gm.SetGlobalBuffer((__gm__ T*)constExpertAlpha2);
        vGm.SetGlobalBuffer((__gm__ T*)v);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        int32_t rowFactorHAlignedT = RoundUp<T>(tilingData->rowFactor * tilingData->h);
        int32_t rowFactorHAlignedFloat = RoundUp<float>(tilingData->rowFactor * tilingData->h);
        int32_t constExpertRangeFactorHAlignedT = RoundUp<T>(tilingData->constExpertRangeFactor * tilingData->h);
        int32_t kFactorAlignedT = RoundUp<T>(tilingData->kFactor);
        pipe->InitBuffer(expandedXQue, DOUBLE_BUFFER, tilingData->kFactor * tilingData->hAligned * sizeof(T));
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, rowFactorHAlignedFloat * sizeof(float));
        if (hasBiasAndExpertIdx) {
            pipe->InitBuffer(biasQue, DOUBLE_BUFFER, tilingData->kFactor * tilingData->hAligned * sizeof(T));
        }
        if (hasX1) {
            pipe->InitBuffer(x1Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasX2) {
            pipe->InitBuffer(x2Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasScales) {
            pipe->InitBuffer(scalesQue, DOUBLE_BUFFER, kFactorAlignedT * sizeof(S));
        }
        if (hasX) {
            // X的大小是row_num, h
            pipe->InitBuffer(xQue, DOUBLE_BUFFER, tilingData->hAligned * sizeof(T));
        }
        if (hasConstExpert) {
            pipe->InitBuffer(constExpertAlpha1Que, DOUBLE_BUFFER, tilingData->hAligned * sizeof(T));
            pipe->InitBuffer(constExpertAlpha2Que, DOUBLE_BUFFER, tilingData->hAligned * sizeof(T));
            pipe->InitBuffer(vQue, DOUBLE_BUFFER, tilingData->hAligned * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        int64_t rowOuterLoop =
            (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                       tilingData->tailRowFactorOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx += 1) {
            int64_t rowInnerLoop = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            // 行循环处理rowInnerLoop个BS
            if (hasX1) {
                x1Local = x1Que.AllocTensor<T>();
                int64_t x1GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                     rowOuterIdx * tilingData->rowFactor * tilingData->h;
                CopyIn(x1Gm[x1GmOffset], x1Local, 1, rowInnerLoop * tilingData->h);
                x1Que.EnQue(x1Local);
                x1Local = x1Que.DeQue<T>();
            }
            if (hasX2) {
                x2Local = x2Que.AllocTensor<T>();
                int64_t x2GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                     rowOuterIdx * tilingData->rowFactor * tilingData->h;
                CopyIn(x2Gm[x2GmOffset], x2Local, 1, rowInnerLoop * tilingData->h);
                x2Que.EnQue(x2Local);
                x2Local = x2Que.DeQue<T>();
            }

            yLocal = yQue.AllocTensor<float>();
            ProcessX1AndX2(yLocal, x1Local, x2Local, rowInnerLoop * tilingData->h, hasX1, hasX2);
            ProcessYWithInput(rowOuterIdx, rowInnerLoop);
            yQue.EnQue(yLocal);
            if (hasX1) {
                x1Que.FreeTensor(x1Local);
            }
            if (hasX2) {
                x2Que.FreeTensor(x2Local);
            }
            yLocal = yQue.DeQue<float>();
            int64_t yGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                rowOuterIdx * tilingData->rowFactor * tilingData->h;
            CopyOut(yLocal.template ReinterpretCast<T>(), yGm[yGmOffset], 1, rowInnerLoop * tilingData->h);
            yQue.FreeTensor(yLocal);
        }
    }

private:
    // rowInnerLoop : rowFactor，就是每次循环处理的BS数
    __aicore__ inline void ProcessYWithInput(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        for (int64_t rowInnerIdx = 0; rowInnerIdx < rowInnerLoop; rowInnerIdx += 1) {
            for (int64_t kOuterIdx = 0; kOuterIdx < tilingData->kLoop; kOuterIdx += 1) {
                int64_t curKFactor = kOuterIdx == tilingData->kLoop - 1 ? tilingData->tailKFactor : tilingData->kFactor;
                ProcessKLoopBody(rowOuterIdx, rowInnerIdx, kOuterIdx, curKFactor);
            }
        }
        CastYOutput(rowInnerLoop);
    }

    __aicore__ inline void ProcessKLoopBody(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kOuterIdx,
                                            int64_t curKFactor)
    {
        if (hasScales) {
            scalesLocal = scalesQue.AllocTensor<S>();
            int64_t scalesGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                                     rowOuterIdx * tilingData->rowFactor * tilingData->k + rowInnerIdx * tilingData->k +
                                     kOuterIdx * tilingData->kFactor;
            CopyIn(scalesGm[scalesGmOffset], scalesLocal, 1, curKFactor);
            scalesQue.EnQue(scalesLocal);
            scalesLocal = scalesQue.DeQue<S>();
        }
        expandedXLocal = expandedXQue.AllocTensor<T>();
        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.AllocTensor<T>();
        }
        if (hasX) {
            xLocal = xQue.AllocTensor<T>();
        }
        if (hasConstExpert) {
            constExpertAlpha1Local = constExpertAlpha1Que.AllocTensor<T>();
            constExpertAlpha2Local = constExpertAlpha2Que.AllocTensor<T>();
            vLocal = vQue.AllocTensor<T>();
        }
        CopyExpanedXAndBiasScales(curKFactor, rowOuterIdx, rowInnerIdx, kOuterIdx);
        EnQueDeQueTensors();
        if (hasScales) {
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventId);
            WaitFlag<HardEvent::S_V>(eventId);
        }
        ProcessExpandXBiasScaleOptimized<T>(yLocal[rowInnerIdx * tilingData->h], expandedXLocal, biasLocal, scalesLocal,
                                            validK, tilingData->h, hasBiasAndExpertIdx, hasScales);
        FreeKTensors();
    }

    __aicore__ inline void EnQueDeQueTensors()
    {
        expandedXQue.EnQue(expandedXLocal);
        expandedXLocal = expandedXQue.DeQue<T>();
        if (hasBiasAndExpertIdx) {
            biasQue.EnQue(biasLocal);
            biasLocal = biasQue.DeQue<T>();
        }
        if (hasX) {
            xQue.EnQue(xLocal);
            xLocal = xQue.DeQue<T>();
        }
        if (hasConstExpert) {
            constExpertAlpha1Que.EnQue(constExpertAlpha1Local);
            constExpertAlpha1Local = constExpertAlpha1Que.DeQue<T>();
            constExpertAlpha2Que.EnQue(constExpertAlpha2Local);
            constExpertAlpha2Local = constExpertAlpha2Que.DeQue<T>();
            vQue.EnQue(vLocal);
            vLocal = vQue.DeQue<T>();
        }
    }

    __aicore__ inline void FreeKTensors()
    {
        expandedXQue.FreeTensor(expandedXLocal);
        if (hasBiasAndExpertIdx) {
            biasQue.FreeTensor(biasLocal);
        }
        if (hasX) {
            xQue.FreeTensor(xLocal);
        }
        if (hasConstExpert) {
            constExpertAlpha1Que.FreeTensor(constExpertAlpha1Local);
            constExpertAlpha2Que.FreeTensor(constExpertAlpha2Local);
            vQue.FreeTensor(vLocal);
        }
        if (hasScales) {
            scalesQue.FreeTensor(scalesLocal);
        }
    }

    __aicore__ inline void CastYOutput(int64_t rowInnerLoop)
    {
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, rowInnerLoop * tilingData->h);
        }
    }

    __aicore__ inline void CopyExpanedXAndBiasScales(int64_t curKFactor, int64_t rowOuterIdx, int64_t rowInnerIdx,
                                                     int64_t kOuterIdx)
    {
        validK = 0;
        for (int64_t kInnerIdx = 0; kInnerIdx < curKFactor; kInnerIdx += 1) {
            CopyInnerLoopBody(rowOuterIdx, rowInnerIdx, kOuterIdx, kInnerIdx);
        }
    }

    __aicore__ inline void CopyInnerLoopBody(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kOuterIdx,
                                             int64_t kInnerIdx)
    {
        SetExpandedRowIdxOffset(rowOuterIdx, rowInnerIdx, kOuterIdx, kInnerIdx);
        int64_t gmValueOfExpandedRowIdx = expandedRowIdxGm.GetValue(expandedRowIdxOffset);
        if (gmValueOfExpandedRowIdx == INVALID_IDX) {
            return;
        }
        if constexpr (dropPadMode != DROP_PAD_ROW && dropPadMode != DROP_PAD_COLUMN) {
            if (tilingData->activeNum <= gmValueOfExpandedRowIdx) {
                return;
            }
        }
        int64_t expertIdx = -1;
        if (hasExpertIdx) {
            SetOffsetForExpertIdx(kOuterIdx, rowOuterIdx, rowInnerIdx, kInnerIdx);
            expertIdx = expertIdxGm.GetValue(expertIdxOffset);
        }

        if (hasX && hasExpertIdx) {
            if (expertIdx >= tilingData->zeroExpertStart && expertIdx < tilingData->zeroExpertEnd) {
                return;
            } else if (expertIdx >= tilingData->copyExpertStart && expertIdx < tilingData->copyExpertEnd) {
                CopyCopyExpertToExpandedX(rowOuterIdx, rowInnerIdx);
            } else if (hasConstExpert && expertIdx >= tilingData->constantExpertStart &&
                       expertIdx < tilingData->constantExpertEnd) {
                CopyConstantExpertToExpandedX(rowOuterIdx, rowInnerIdx, expertIdx);
            } else {
                CopyIn(expandedXGm[gmValueOfExpandedRowIdx * tilingData->h],
                       expandedXLocal[validK * tilingData->hAligned], 1, tilingData->h);
            }
        } else {
            CopyIn(expandedXGm[gmValueOfExpandedRowIdx * tilingData->h], expandedXLocal[validK * tilingData->hAligned],
                   1, tilingData->h);
        }
        if (hasBiasAndExpertIdx) {
            int64_t biasGmOffset = expertIdx * tilingData->h;
            CopyIn(biasGm[biasGmOffset], biasLocal[validK * tilingData->hAligned], 1, tilingData->h);
        }
        if (hasScales) {
            S scaleValue = scalesLocal.GetValue(kInnerIdx);
            scalesLocal.SetValue(validK, scaleValue);
        }
        validK += 1;
    }

    __aicore__ inline void CopyCopyExpertToExpandedX(int64_t rowOuterIdx, int64_t rowInnerIdx)
    {
        int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                            rowOuterIdx * tilingData->rowFactor * tilingData->h + rowInnerIdx * tilingData->h;
        CopyIn(xGm[xGmOffset], expandedXLocal[validK * tilingData->hAligned], 1, tilingData->h);
    }

    __aicore__ inline void CopyConstantExpertToExpandedX(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t expertIdx)
    {
        int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                            rowOuterIdx * tilingData->rowFactor * tilingData->h + rowInnerIdx * tilingData->h;
        CopyIn(xGm[xGmOffset], xLocal, 1, tilingData->h);
        int64_t constExpertGmOffset = (expertIdx - tilingData->constantExpertStart) * tilingData->h;
        CopyIn(constExpertAlpha1Gm[constExpertGmOffset], constExpertAlpha1Local, 1, tilingData->h);
        CopyIn(constExpertAlpha2Gm[constExpertGmOffset], constExpertAlpha2Local, 1, tilingData->h);
        CopyIn(vGm[constExpertGmOffset], vLocal, 1, tilingData->h);
        PipeBarrier<PIPE_ALL>();
        vLocal = vLocal * constExpertAlpha2Local;
        xLocal = xLocal * constExpertAlpha1Local;
        xLocal = xLocal + vLocal;
        AscendC::Copy(expandedXLocal[validK * tilingData->hAligned], xLocal, tilingData->h);
        event_t v2mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(v2mte);
        WaitFlag<HardEvent::V_MTE2>(v2mte);
    }

    __aicore__ inline void SetExpandedRowIdxOffset(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kOuterIdx,
                                                   int64_t kInnerIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor +
                                   rowInnerIdx + (kOuterIdx * tilingData->kFactor + kInnerIdx) * tilingData->row;
        } else {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                                   rowOuterIdx * tilingData->rowFactor * tilingData->k + rowInnerIdx * tilingData->k +
                                   kOuterIdx * tilingData->kFactor + kInnerIdx;
        }
    }

    __aicore__ inline void SetOffsetForExpertIdx(int64_t kOuterIdx, int64_t rowOuterIdx, int64_t rowInnerIdx,
                                                 int64_t kInnerIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expertIdxOffset = rowInnerIdx * tilingData->k + kOuterIdx * tilingData->kFactor + kInnerIdx +
                              GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                              rowOuterIdx * tilingData->rowFactor * tilingData->k;
            return;
        }
        expertIdxOffset = expandedRowIdxOffset;
        return;
    }

private:
    TPipe* pipe;
    const MoeFinalizeRoutingV2RegbaseTilingData* tilingData;

    int64_t expandedRowIdxOffset{0};
    int64_t expertIdxOffset{0};
    int64_t scaleOffset{0};
    int16_t validK{0};

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

    S scale;
    bool hasX1{false};
    bool hasX2{false};
    bool hasBiasAndExpertIdx{false};
    bool hasExpertIdx{false};
    bool hasScales{false};
    bool hasX{false};
    bool hasConstExpert{false};

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedXQue;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> constExpertAlpha1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> constExpertAlpha2Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> vQue;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x2Que;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> scalesQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> biasQue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQue;

    LocalTensor<T> expandedXLocal;
    LocalTensor<T> biasLocal;

    LocalTensor<T> xLocal;
    LocalTensor<T> constExpertAlpha1Local;
    LocalTensor<T> constExpertAlpha2Local;
    LocalTensor<T> vLocal;

    LocalTensor<S> scalesLocal;

    LocalTensor<T> x1Local;
    LocalTensor<T> x2Local;

    LocalTensor<float> yLocal;
};

#ifdef MOE_FINALIZE_ROUTING_HAS_MXFP4
// MXFP4 specialization: BF16 output and FP32 accumulation.
template <int32_t dropPadMode>
class MoeFinalizeRoutingV2HFullLoad<fp4x2_e2m1_t, float8_e8m0_t, dropPadMode>
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
class MoeFinalizeRoutingV2Mx8HFullLoadImpl : public MoeFinalizeRoutingV2Mx8RouteBase<DROP_PAD_MODE> {
public:
    __aicore__ MoeFinalizeRoutingV2Mx8HFullLoadImpl() {}

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
        this->InitMx8Optional(data, pipe, expandedRowIdx, expertIdx, bias, scales,
                              static_cast<uint32_t>(data_->kFactor));
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
        auto scalesLen = static_cast<int64_t>(data_->kFactor) * scalesRowLenAligned_;
        // 每路由槽位（expandedX/bias）按 hAligned 步进保证 32B 对齐；x1/x2/y 批量缓冲按真实 H。
        auto expandedXLen = static_cast<uint32_t>(data_->kFactor * data_->hAligned);
        auto biasLen = static_cast<uint32_t>(data_->kFactor * data_->hAligned * sizeof(bfloat16_t));
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
            for (int64_t kOuterIdx = 0; kOuterIdx < data_->kLoop; kOuterIdx += 1) {
                int64_t curKFactor = kOuterIdx == data_->kLoop - 1 ? data_->tailKFactor : data_->kFactor;
                uint32_t validK = CopyInExpandedXScaleBias(rowOuterIdx, rowInnerIdx, curKFactor, kOuterIdx,
                                                           yLocal[rowInnerIdx * data_->h]);
                ComputeDequantize(yLocal[rowInnerIdx * data_->h], validK);
            }
        }
    }

    __aicore__ uint32_t CopyInExpandedXScaleBias(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t curKFactor,
                                                 int64_t kOuterIdx, const LocalTensor<float>& yRow)
    {
        LocalTensor<uint8_t> expandedXLocal = expandedXQue_.AllocTensor<uint8_t>();
        LocalTensor<uint8_t> scalesLocal = scalesQue_.AllocTensor<uint8_t>();
        LocalTensor<bfloat16_t> biasLocal;
        if (hasBiasAndExpertIdx_) {
            biasLocal = biasQue_.AllocTensor<bfloat16_t>();
        }
        uint32_t validK = 0;
        const int64_t rowIndex = coreIdx_ * data_->rowOfFormerBlock + rowOuterIdx * data_->rowFactor + rowInnerIdx;
        for (int64_t kIdx = 0; kIdx < curKFactor; ++kIdx) {
            // K 块起点必须使用累计偏移 kOuterIdx * kFactor；curKFactor 在最后一个
            // 不满块时会变成 tailKFactor，不能用于计算块起点。
            const int64_t kOffset = kOuterIdx * data_->kFactor + kIdx;
            int64_t gatherRowIdx, expert;
            if (!this->Route(rowIndex, kOffset, gatherRowIdx, expert, validK)) {
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
class MoeFinalizeRoutingV2HFullLoad<fp8_e4m3fn_t, float8_e8m0_t, DROP_PAD_MODE>
    : public MoeFinalizeRoutingV2Mx8HFullLoadImpl<fp8_e4m3fn_t, DROP_PAD_MODE> {};

template <int32_t DROP_PAD_MODE>
class MoeFinalizeRoutingV2HFullLoad<fp8_e5m2_t, float8_e8m0_t, DROP_PAD_MODE>
    : public MoeFinalizeRoutingV2Mx8HFullLoadImpl<fp8_e5m2_t, DROP_PAD_MODE> {};
#endif

} // namespace MoeFinalizeRoutingV2Regbase

#endif
