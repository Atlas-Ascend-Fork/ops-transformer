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
 * \file moe_finalize_routing_v2_row_k_h_full_load.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_ROW_K_H_FULL_LOAD_H_
#define MOE_FINALIZE_ROUTING_V2_ROW_K_H_FULL_LOAD_H_

#include "moe_finalize_routing_v2_base.h"
namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;
template <typename T, typename S, int32_t dropPadMode>
class MoeFinalizeRoutingV2RowKHFullLoad {
public:
    __aicore__ inline MoeFinalizeRoutingV2RowKHFullLoad() {}

    // expandX/bias 全部载入ub，避免多次小段搬运, 此时对应的Que不开db
    __aicore__ inline void Init(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR x, GM_ADDR constExpertAlpha1,
                                GM_ADDR constExpertAlpha2, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                const MoeFinalizeRoutingV2RegbaseTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        hasX1 = (x1 != nullptr);
        hasX2 = (x2 != nullptr);
        hasBiasAndExpertIdx = (bias != nullptr) && (expertIdx != nullptr);
        hasExpertIdx = expertIdx != nullptr;
        hasScales = scales != nullptr;
        hasX = (x != nullptr);
        hasConstExpert = (constExpertAlpha1 != nullptr) && (constExpertAlpha2 != nullptr) && (v != nullptr);

        SetGlobalBuffers(expandedX, expandedRowIdx, x1, x2, bias, scales, expertIdx, x, constExpertAlpha1,
                         constExpertAlpha2, v, y);
        InitQueBuffers();
    }

    __aicore__ inline void Process()
    {
        LoadExpandedX();
        LoadBiasAndAllocOptionalTensors();

        int64_t rowOuterLoop =
            (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        int64_t tailRowFactor = (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->tailRowFactorOfTailBlock :
                                                                       tilingData->tailRowFactorOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx += 1) {
            int64_t rowInnerLoop = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : tilingData->rowFactor;
            ProcessLoopBody(rowOuterIdx, rowInnerLoop);
        }

        FreePersistentTensors();
    }

private:
    __aicore__ inline void LoadExpandedX()
    {
        expandedXLocal = expandedXQue.AllocTensor<T>();
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROPLESS_ROW) {
            CopyIn(expandedXGm, expandedXLocal, 1, tilingData->row * tilingData->k * tilingData->h);
        } else {
            CopyIn(expandedXGm, expandedXLocal, 1, tilingData->e * tilingData->c * tilingData->h);
        }
        expandedXQue.EnQue(expandedXLocal);
        expandedXLocal = expandedXQue.DeQue<T>();
    }

    __aicore__ inline void LoadBiasAndAllocOptionalTensors()
    {
        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.AllocTensor<T>();
            CopyIn(biasGm, biasLocal, 1, tilingData->e * tilingData->h);
            biasQue.EnQue(biasLocal);
            biasLocal = biasQue.DeQue<T>();
        }
        if (hasX) {
            xLocal = xQue.AllocTensor<T>();
        }
        if (hasConstExpert) {
            constExpertAlpha1Local = constExpertAlpha1Que.AllocTensor<T>();
            constExpertAlpha2Local = constExpertAlpha2Que.AllocTensor<T>();
            vLocal = vQue.AllocTensor<T>();
        }
    }

    __aicore__ inline void ProcessLoopBody(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        ProcessInputWithX(rowOuterIdx, rowInnerLoop);

        yLocal = yQue.AllocTensor<float>();
        ProcessX1AndX2(yLocal, x1Local, x2Local, rowInnerLoop * tilingData->h, hasX1, hasX2);
        ProcessYWithInput(rowOuterIdx, rowInnerLoop);
        yQue.EnQue(yLocal);

        if (hasX1) {
            x1Que.FreeTensor(x1Local);
        }
        if (hasScales) {
            scalesQue.FreeTensor(scalesLocal);
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

    __aicore__ inline void FreePersistentTensors()
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
    }
    __aicore__ inline void ProcessInputWithX(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        if (hasX2) {
            x2Local = x2Que.AllocTensor<T>();
            int64_t x2GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                 rowOuterIdx * tilingData->rowFactor * tilingData->h;
            CopyIn(x2Gm[x2GmOffset], x2Local, 1, rowInnerLoop * tilingData->h);
            x2Que.EnQue(x2Local);
            x2Local = x2Que.DeQue<T>();
        }
        if (hasScales) {
            scalesLocal = scalesQue.AllocTensor<S>();
            int64_t scaleGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                                    rowOuterIdx * tilingData->rowFactor * tilingData->k;
            CopyIn(scalesGm[scaleGmOffset], scalesLocal, 1, rowInnerLoop * tilingData->k);
            scalesQue.EnQue(scalesLocal);
            scalesLocal = scalesQue.DeQue<S>();
        }
        if (hasX1) {
            x1Local = x1Que.AllocTensor<T>();
            int64_t x1GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                 rowOuterIdx * tilingData->rowFactor * tilingData->h;
            CopyIn(x1Gm[x1GmOffset], x1Local, 1, rowInnerLoop * tilingData->h);
            x1Que.EnQue(x1Local);
            x1Local = x1Que.DeQue<T>();
        }
    }

    __aicore__ inline void ProcessYWithInput(int64_t rowOuterIdx, int64_t rowInnerLoop)
    {
        for (int64_t rowInnerIdx = 0; rowInnerIdx < rowInnerLoop; rowInnerIdx += 1) {
            for (int64_t kIdx = 0; kIdx < tilingData->k; kIdx += 1) {
                ProcessInnerLoopBody(rowOuterIdx, rowInnerIdx, kIdx);
            }
        }
        CastYOutput(rowInnerLoop);
    }

    __aicore__ inline void ProcessInnerLoopBody(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kIdx)
    {
        SetOffsetForExpandedRowIdx(rowOuterIdx, kIdx, rowInnerIdx);
        int64_t expandedRowIdxGmValue = expandedRowIdxGm.GetValue(expandedRowIdxOffset);
        if (expandedRowIdxGmValue == INVALID_IDX) {
            return;
        }
        if constexpr (dropPadMode != DROP_PAD_COLUMN && dropPadMode != DROP_PAD_ROW) {
            if (expandedRowIdxGmValue >= tilingData->activeNum) {
                return;
            }
        }
        int64_t expertIdx = -1;
        if (hasExpertIdx) {
            SetExpertIdxOffset(rowOuterIdx, rowInnerIdx, kIdx);
            expertIdx = expertIdxGm.GetValue(expertIdxOffset);
        }

        if (hasBiasAndExpertIdx) {
            biasGmOffset = expertIdx * tilingData->h;
        }
        if (hasScales) {
            int64_t scaleOffset = rowInnerIdx * tilingData->k + kIdx;
            scale = scalesLocal.GetValue(scaleOffset);
        }
        LocalTensor<T> innerBiasLocal = hasBiasAndExpertIdx ? biasLocal[biasGmOffset] : expandedXLocal;
        if (hasX && hasExpertIdx) {
            if (expertIdx >= tilingData->zeroExpertStart && expertIdx < tilingData->zeroExpertEnd) {
                return;
            }
            if (expertIdx >= tilingData->copyExpertStart && expertIdx < tilingData->copyExpertEnd) {
                ProcessCopyExpert(rowOuterIdx, rowInnerIdx, innerBiasLocal, scale);
            } else if (hasConstExpert && expertIdx >= tilingData->constantExpertStart &&
                       expertIdx < tilingData->constantExpertEnd) {
                ProcessConstantExpert(rowOuterIdx, rowInnerIdx, expertIdx, innerBiasLocal, scale);
            } else {
                ProcessExpandXBiasScale<T, S>(yLocal[rowInnerIdx * tilingData->h],
                                              expandedXLocal[expandedRowIdxGmValue * tilingData->h], innerBiasLocal,
                                              scale, tilingData->h, hasBiasAndExpertIdx, hasScales);
            }
        } else {
            ProcessExpandXBiasScale<T, S>(yLocal[rowInnerIdx * tilingData->h],
                                          expandedXLocal[expandedRowIdxGmValue * tilingData->h], innerBiasLocal, scale,
                                          tilingData->h, hasBiasAndExpertIdx, hasScales);
        }
    }

    __aicore__ inline void ProcessCopyExpert(int64_t rowOuterIdx, int64_t rowInnerIdx,
                                             const LocalTensor<T>& innerBiasLocal, S scale)
    {
        int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                            rowOuterIdx * tilingData->rowFactor * tilingData->h + rowInnerIdx * tilingData->h;
        CopyIn(xGm[xGmOffset], xLocal, 1, tilingData->h);
        xQue.EnQue(xLocal);
        xLocal = xQue.DeQue<T>();
        ProcessExpandXBiasScale<T, S>(yLocal[rowInnerIdx * tilingData->h], xLocal, innerBiasLocal, scale, tilingData->h,
                                      hasBiasAndExpertIdx, hasScales);
    }

    __aicore__ inline void ProcessConstantExpert(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t expertIdx,
                                                 const LocalTensor<T>& innerBiasLocal, S scale)
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
        constExpertAlpha1Que.EnQue(constExpertAlpha1Local);
        constExpertAlpha1Local = constExpertAlpha1Que.DeQue<T>();
        constExpertAlpha2Que.EnQue(constExpertAlpha2Local);
        constExpertAlpha2Local = constExpertAlpha2Que.DeQue<T>();
        vQue.EnQue(vLocal);
        vLocal = vQue.DeQue<T>();
        xQue.EnQue(xLocal);
        xLocal = xQue.DeQue<T>();
        ProcessExpandXBiasScale<T, S>(yLocal[rowInnerIdx * tilingData->h], xLocal, innerBiasLocal, scale, tilingData->h,
                                      hasBiasAndExpertIdx, hasScales);
        event_t v2mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(v2mte);
        WaitFlag<HardEvent::V_MTE2>(v2mte);
    }

    __aicore__ inline void CastYOutput(int64_t rowInnerLoop)
    {
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, rowInnerLoop * tilingData->h);
        }
    }

    __aicore__ inline void SetOffsetForExpandedRowIdx(int64_t rowOuterIdx, int64_t kIdx, int64_t rowInnerIdx)
    {
        if constexpr (dropPadMode != DROPLESS_COLUMN && dropPadMode != DROP_PAD_COLUMN) {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                                   rowOuterIdx * tilingData->rowFactor * tilingData->k + rowInnerIdx * tilingData->k +
                                   kIdx;
            return;
        }
        expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx * tilingData->rowFactor +
                               rowInnerIdx + kIdx * tilingData->row;
        return;
    }

    __aicore__ inline void SetExpertIdxOffset(int64_t rowOuterIdx, int64_t rowInnerIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expertIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k +
                              rowOuterIdx * tilingData->rowFactor * tilingData->k + rowInnerIdx * tilingData->k + kIdx;
        } else {
            expertIdxOffset = expandedRowIdxOffset;
        }
    }

    __aicore__ inline void SetGlobalBuffers(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2,
                                            GM_ADDR bias, GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR x,
                                            GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR v, GM_ADDR y)
    {
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
    }

    __aicore__ inline void InitQueBuffers()
    {
        int32_t eHAligned = RoundUp<T>(tilingData->e * tilingData->h);
        int32_t rowFactorHAlignedFloat = RoundUp<float>(tilingData->rowFactor * tilingData->h);
        int32_t rowFactorKAlignedT = RoundUp<T>(tilingData->rowFactor * tilingData->k);
        int32_t rowFactorHAlignedT = RoundUp<T>(tilingData->rowFactor * tilingData->h);
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROPLESS_ROW) {
            int32_t rowsKHAligned = RoundUp<T>(tilingData->row * tilingData->k * tilingData->h);
            pipe->InitBuffer(expandedXQue, 1, rowsKHAligned * sizeof(T));
        } else {
            int32_t eCHAligned = RoundUp<T>(tilingData->e * tilingData->c * tilingData->h);
            pipe->InitBuffer(expandedXQue, 1, eCHAligned * sizeof(T));
        }
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, rowFactorHAlignedFloat * sizeof(float));
        if (hasX1) {
            pipe->InitBuffer(x1Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasBiasAndExpertIdx) {
            pipe->InitBuffer(biasQue, 1, eHAligned * sizeof(T));
        }
        if (hasX2) {
            pipe->InitBuffer(x2Que, DOUBLE_BUFFER, rowFactorHAlignedT * sizeof(T));
        }
        if (hasScales) {
            pipe->InitBuffer(scalesQue, DOUBLE_BUFFER, rowFactorKAlignedT * sizeof(S));
        }
        if (hasX) {
            pipe->InitBuffer(xQue, DOUBLE_BUFFER, tilingData->h * sizeof(T));
        }
        if (hasConstExpert) {
            pipe->InitBuffer(constExpertAlpha1Que, DOUBLE_BUFFER, tilingData->h * sizeof(T));
            pipe->InitBuffer(constExpertAlpha2Que, DOUBLE_BUFFER, tilingData->h * sizeof(T));
            pipe->InitBuffer(vQue, DOUBLE_BUFFER, tilingData->h * sizeof(T));
        }
    }

    TPipe* pipe;
    const MoeFinalizeRoutingV2RegbaseTilingData* tilingData;

    S scale;
    bool hasX1{false};
    bool hasX2{false};
    bool hasBiasAndExpertIdx{false};
    bool hasExpertIdx{false};
    bool hasScales{false};
    bool hasX{false};
    bool hasConstExpert{false};
    int64_t biasGmOffset{0};

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

    TQue<QuePosition::VECIN, 1> expandedXQue;
    TQue<QuePosition::VECIN, 1> biasQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x2Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> scalesQue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> xQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> constExpertAlpha1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> constExpertAlpha2Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> vQue;

    LocalTensor<T> expandedXLocal;
    LocalTensor<T> x1Local;
    LocalTensor<T> x2Local;
    LocalTensor<T> biasLocal;
    LocalTensor<S> scalesLocal;
    LocalTensor<float> yLocal;
    LocalTensor<T> xLocal;
    LocalTensor<T> constExpertAlpha1Local;
    LocalTensor<T> constExpertAlpha2Local;
    LocalTensor<T> vLocal;

    int64_t expandedRowIdxOffset{0};
    int64_t expertIdxOffset{0};
};

#ifdef MOE_FINALIZE_ROUTING_HAS_MXFP4
// MXFP4 specialization: BF16 output and FP32 accumulation.
template <int32_t dropPadMode>
class MoeFinalizeRoutingV2RowKHFullLoad<fp4x2_e2m1_t, float8_e8m0_t, dropPadMode>
    : public MoeFinalizeRoutingV2Mx4Base<dropPadMode> {
    using TYPE = bfloat16_t;

public:
    __aicore__ inline void Init(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR routingWeights, GM_ADDR expertIdx, GM_ADDR y,
                                GM_ADDR /*workspace*/, const MoeFinalizeRoutingV2RegbaseTilingData* data, TPipe* pipe)
    {
        this->InitMx4Inputs(expandedX, expandedRowIdx, x1, x2, bias, scales, routingWeights, expertIdx, y, data, pipe);
        this->pipe_->InitBuffer(this->packedQue_, MX4_QUEUE_DEPTH,
                                this->Align(this->d_->activeNum * this->d_->h / NUM_FP4_IN_BYTE));
        this->pipe_->InitBuffer(this->scaleQue_, MX4_QUEUE_DEPTH,
                                this->Align(this->d_->activeNum * this->d_->mx4ScaleRowBytes));
        if (this->hasBias_) {
            this->pipe_->InitBuffer(this->biasQue_, MX4_QUEUE_DEPTH,
                                    this->Align(this->d_->mx4BiasRows * this->d_->h * sizeof(TYPE)));
        }
    }

    __aicore__ inline void Process()
    {
        this->packedLocal_ = this->packedQue_.template AllocTensor<uint8_t>();
        this->scaleLocal_ = this->scaleQue_.template AllocTensor<uint8_t>();
        this->CopyBytes(this->packedLocal_, this->x_, this->d_->activeNum * this->d_->h / NUM_FP4_IN_BYTE);
        this->CopyBytes(this->scaleLocal_, this->scales_, this->d_->activeNum * this->d_->mx4ScaleRowBytes);
        this->packedQue_.EnQue(this->packedLocal_);
        this->scaleQue_.EnQue(this->scaleLocal_);
        this->packedLocal_ = this->packedQue_.template DeQue<uint8_t>();
        this->scaleLocal_ = this->scaleQue_.template DeQue<uint8_t>();
        if (this->hasBias_) {
            this->biasLocal_ = this->biasQue_.template AllocTensor<TYPE>();
            CopyIn(this->bias_, this->biasLocal_, 1, this->d_->mx4BiasRows * this->d_->h);
            this->biasQue_.EnQue(this->biasLocal_);
            this->biasLocal_ = this->biasQue_.template DeQue<TYPE>();
        }
        const int64_t start = GetBlockIdx() * this->d_->rowOfFormerBlock;
        const int64_t count =
            GetBlockIdx() == GetBlockNum() - 1 ? this->d_->rowOfTailBlock : this->d_->rowOfFormerBlock;
        for (int64_t row = start; row < start + count; ++row) {
            ComputeMx4Tile(row, 0, static_cast<uint32_t>(this->d_->h));
        }
        this->packedQue_.FreeTensor(this->packedLocal_);
        this->scaleQue_.FreeTensor(this->scaleLocal_);
        if (this->hasBias_) {
            this->biasQue_.FreeTensor(this->biasLocal_);
        }
    }

private:
    __aicore__ inline void ComputeMx4Tile(int64_t row, int64_t h, uint32_t length)
    {
        this->BeginMx4Tile(row, h, length);
        auto acc = this->accBuf_.template Get<float>();
        auto values = this->valueBuf_.template Get<float>();
        for (int64_t k0 = 0; k0 < this->d_->k; k0 += this->d_->kFactor) {
            int64_t r, expert;
            if (this->ResolveRoute(row, k0, r, expert)) {
                this->Accumulate(acc, values,
                                 this->packedLocal_[r * this->d_->h / NUM_FP4_IN_BYTE + h / NUM_FP4_IN_BYTE],
                                 this->scaleLocal_[r * this->d_->mx4ScaleRowBytes + h / MX4_SCALE_GROUP_SIZE],
                                 this->hasBias_ ? this->biasLocal_[expert * this->d_->h + h] : LocalTensor<TYPE>(),
                                 length, this->RoutingWeight(row, k0));
            }
        }
        this->StoreMx4Tile(row, h, length);
    }
};
#endif

#ifdef MOE_FINALIZE_ROUTING_HAS_MXFP8
// MXFP8 specializations: BF16 output and FP32 accumulation.
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

template <typename T, int32_t DROP_PAD_MODE>
class MoeFinalizeRoutingV2Mx8RowKHFullLoadImpl : public MoeFinalizeRoutingV2Mx8RouteBase<DROP_PAD_MODE> {
public:
    __aicore__ MoeFinalizeRoutingV2Mx8RowKHFullLoadImpl() {}

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
        this->InitMx8Optional(data, pipe, expandedRowIdx, expertIdx, bias, scales, 1);
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
        // 搬运按实际物理行数 activeNum 计算：dropless 2D 时 dim0 可小于 NUM_ROWS*K，
        // 按 NUM_ROWS*K 搬运会读过输入末尾。
        expandedXRowBytes_ = static_cast<uint32_t>(data_->activeNum) * static_cast<uint32_t>(data_->h);
        // scale 与数据同物理行（与 moe_init_routing_v3 的 MXFP8 scale 输出布局一致），整块常驻 UB
        scalesWholeBytes_ = static_cast<int64_t>(data_->activeNum) * scalesRowLenAligned_;
        auto biasLen = static_cast<uint32_t>(data_->e * data_->h * sizeof(bfloat16_t));
        auto rowsBf16Len = static_cast<uint32_t>(data_->rowFactor * data_->h * sizeof(bfloat16_t));
        auto yLen = static_cast<uint32_t>(data_->rowFactor * data_->h * sizeof(float));
        auto scratchLen =
            static_cast<uint32_t>(CeilAlign(Mx8ScratchBytes(static_cast<uint32_t>(data_->hAligned), scaleNumAligned_),
                                            static_cast<uint32_t>(UBBLOCKSIZE)));

        pipe_->InitBuffer(expandedXQue_, 1, expandedXRowBytes_);
        pipe_->InitBuffer(scalesQue_, 1, static_cast<uint32_t>(scalesWholeBytes_));
        if (hasX1_) {
            pipe_->InitBuffer(x1Que_, DOUBLE_BUFFER, rowsBf16Len);
        }
        if (hasX2_) {
            pipe_->InitBuffer(x2Que_, DOUBLE_BUFFER, rowsBf16Len);
        }
        if (hasBiasAndExpertIdx_) {
            pipe_->InitBuffer(biasQue_, 1, biasLen);
        }
        pipe_->InitBuffer(yQue_, DOUBLE_BUFFER, yLen);
        pipe_->InitBuffer(scratchBuf_, scratchLen);
        InitScaleIndex();
    }

    __aicore__ void Process()
    {
        LocalTensor<uint8_t> expandedXLocal = expandedXQue_.AllocTensor<uint8_t>();
        DataCopyPad<uint8_t>(expandedXLocal, expandedXGm_, DataCopyExtParams{1, expandedXRowBytes_, 0, 0, 0},
                             DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        expandedXQue_.EnQue(expandedXLocal);
        expandedXLocal = expandedXQue_.DeQue<uint8_t>();

        // scale 整块拷入一次，随数据按物理行寻址（详见 InnerLoopProcessInput）
        LocalTensor<uint8_t> scalesLocal = scalesQue_.AllocTensor<uint8_t>();
        DataCopyPad<uint8_t>(scalesLocal, scalesGm_,
                             DataCopyExtParams{1, static_cast<uint32_t>(scalesWholeBytes_), 0, 0, 0},
                             DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        scalesQue_.EnQue(scalesLocal);
        scalesLocal = scalesQue_.DeQue<uint8_t>();
        Mx8WaitScaleCopyReady();

        LocalTensor<bfloat16_t> biasLocal;
        if (hasBiasAndExpertIdx_) {
            biasLocal = biasQue_.AllocTensor<bfloat16_t>();
            CopyIn<bfloat16_t>(biasGm_, biasLocal, 1, data_->e * data_->h);
            biasQue_.EnQue(biasLocal);
            biasLocal = biasQue_.DeQue<bfloat16_t>();
        }

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
            InnerLoopProcessInput(yLocal, expandedXLocal, scalesLocal, biasLocal, rowOuterIdx, rowInnerLoop);
            Cast(yLocal.template ReinterpretCast<bfloat16_t>(), yLocal, RoundMode::CAST_RINT, elemLen);
            yQue_.EnQue(yLocal);

            yLocal = yQue_.DeQue<float>();
            CopyOut<bfloat16_t>(yLocal.template ReinterpretCast<bfloat16_t>(), yGm_[elemOffset], 1, elemLen);
            yQue_.FreeTensor(yLocal);
        }

        scalesQue_.FreeTensor(scalesLocal);
        expandedXQue_.FreeTensor(expandedXLocal);
        if (hasBiasAndExpertIdx_) {
            biasQue_.FreeTensor(biasLocal);
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

    __aicore__ void InnerLoopProcessInput(const LocalTensor<float>& yLocal, const LocalTensor<uint8_t>& expandedXLocal,
                                          const LocalTensor<uint8_t>& scalesLocal,
                                          const LocalTensor<bfloat16_t>& biasLocal, int64_t rowOuterIdx,
                                          int64_t rowInnerLoop)
    {
        LocalTensor<float> scratch = scratchBuf_.Get<float>();
        LocalTensor<float> values, biasValues, scaleExpanded, scaleF32;
        LocalTensor<uint32_t> scaleIndex;
        Mx8GetScratchTensors(scratch, static_cast<uint32_t>(data_->hAligned), scaleNumAligned_, values, biasValues,
                             scaleExpanded, scaleF32, scaleIndex);
        for (int64_t rowInnerIdx = 0; rowInnerIdx < rowInnerLoop; ++rowInnerIdx) {
            const int64_t rowIndex = coreIdx_ * data_->rowOfFormerBlock + rowOuterIdx * data_->rowFactor + rowInnerIdx;
            for (int64_t kIdx = 0; kIdx < data_->k; ++kIdx) {
                int64_t gatherRowIdx, expert;
                if (!this->Route(rowIndex, kIdx, gatherRowIdx, expert)) {
                    continue;
                }
                LocalTensor<uint8_t> xRow = expandedXLocal[gatherRowIdx * data_->h];
                // scale 与数据同物理行：直接用 gatherRowIdx 寻址，
                // 与 moe_init_routing_v3 的 MXFP8 scale 输出布局一致
                LocalTensor<uint8_t> sRow = scalesLocal[gatherRowIdx * scalesRowElemNum_];
                LocalTensor<bfloat16_t> bRow;
                if (hasBiasAndExpertIdx_) {
                    bRow = biasLocal[expert * data_->h];
                }
                Mx8DequantizeRow<T>(yLocal[rowInnerIdx * data_->h], xRow, sRow, bRow, values, biasValues, scaleExpanded,
                                    scaleF32, scaleIndex, static_cast<uint32_t>(data_->h), hasBiasAndExpertIdx_,
                                    this->SlotWeight(0));
            }
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
    uint32_t expandedXRowBytes_{0};
    int64_t scalesWholeBytes_{0};
};

template <int32_t DROP_PAD_MODE>
class MoeFinalizeRoutingV2RowKHFullLoad<fp8_e4m3fn_t, float8_e8m0_t, DROP_PAD_MODE>
    : public MoeFinalizeRoutingV2Mx8RowKHFullLoadImpl<fp8_e4m3fn_t, DROP_PAD_MODE> {};

template <int32_t DROP_PAD_MODE>
class MoeFinalizeRoutingV2RowKHFullLoad<fp8_e5m2_t, float8_e8m0_t, DROP_PAD_MODE>
    : public MoeFinalizeRoutingV2Mx8RowKHFullLoadImpl<fp8_e5m2_t, DROP_PAD_MODE> {};
#endif

} // namespace MoeFinalizeRoutingV2Regbase

#endif
