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
 * \file moe_finalize_routing_v2_h_partial_load.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_H_PARTIAL_LOAD_H_
#define MOE_FINALIZE_ROUTING_V2_H_PARTIAL_LOAD_H_

#include "moe_finalize_routing_v2_base.h"
namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;
template <typename T, typename S, int32_t dropPadMode>
class MoeFinalizeRoutingV2HPartialLoad {
public:
    __aicore__ inline MoeFinalizeRoutingV2HPartialLoad() {}

    __aicore__ inline void Init(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                                GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR x, GM_ADDR constExpertAlpha1,
                                GM_ADDR constExpertAlpha2, GM_ADDR v, GM_ADDR y, GM_ADDR workspace,
                                const MoeFinalizeRoutingV2RegbaseTilingData* tilingDataPtr, TPipe* pipePtr)
    {
        hasBiasAndExpertIdx = (bias != nullptr) && (expertIdx != nullptr);
        hasScales = scales != nullptr;
        hasExpertIdx = expertIdx != nullptr;
        pipe = pipePtr;
        tilingData = tilingDataPtr;
        hasX1 = x1 != nullptr;
        hasX2 = x2 != nullptr;

        hasX = x != nullptr;
        hasConstExpert = (constExpertAlpha1 != nullptr) && (constExpertAlpha2 != nullptr) && (v != nullptr);

        expandedXGm.SetGlobalBuffer((__gm__ T*)expandedX);
        expandedRowIdxGm.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx);
        biasGm.SetGlobalBuffer((__gm__ T*)bias);
        scalesGm.SetGlobalBuffer((__gm__ S*)scales);
        x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        expertIdxGm.SetGlobalBuffer((__gm__ int32_t*)expertIdx);
        xGm.SetGlobalBuffer((__gm__ T*)x);
        constExpertAlpha1Gm.SetGlobalBuffer((__gm__ T*)constExpertAlpha1);
        constExpertAlpha2Gm.SetGlobalBuffer((__gm__ T*)constExpertAlpha2);
        vGm.SetGlobalBuffer((__gm__ T*)v);
        yGm.SetGlobalBuffer((__gm__ T*)y);

        int32_t hFactorAlignedT = RoundUp<T>(tilingData->hFactor);
        int32_t hFactorAlignedFloat = RoundUp<float>(tilingData->hFactor);
        pipe->InitBuffer(expandedXQue, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, hFactorAlignedFloat * sizeof(float));
        if (hasBiasAndExpertIdx) {
            pipe->InitBuffer(biasQue, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }
        if (hasX1) {
            pipe->InitBuffer(x1Que, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }
        if (hasX2) {
            pipe->InitBuffer(x2Que, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }

        if (hasConstExpert) {
            pipe->InitBuffer(constExpertAlpha1Que, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
            pipe->InitBuffer(constExpertAlpha2Que, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
            pipe->InitBuffer(vQue, DOUBLE_BUFFER, hFactorAlignedT * sizeof(T));
        }
    }

    __aicore__ inline void Process()
    {
        int64_t rowOuterLoop =
            (GetBlockIdx() == GetBlockNum() - 1) ? tilingData->rowLoopOfTailBlock : tilingData->rowLoopOfFormerBlock;
        // rowOuterLoop: 循环多少BS
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; rowOuterIdx += 1) {
            // hIdx: 循环多少H, 列
            for (int64_t hIdx = 0; hIdx < tilingData->hLoop; hIdx += 1) {
                int64_t hFactor = (hIdx == tilingData->hLoop - 1) ? tilingData->tailHFactor : tilingData->hFactor;
                if (hasX1) {
                    x1Local = x1Que.AllocTensor<T>();
                    int64_t x1GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                         rowOuterIdx * tilingData->h + hIdx * tilingData->hFactor;
                    CopyIn(x1Gm[x1GmOffset], x1Local, 1, hFactor);
                    x1Que.EnQue(x1Local);
                    x1Local = x1Que.DeQue<T>();
                }
                if (hasX2) {
                    x2Local = x2Que.AllocTensor<T>();
                    int64_t x2GmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                         rowOuterIdx * tilingData->h + hIdx * tilingData->hFactor;
                    CopyIn(x2Gm[x2GmOffset], x2Local, 1, hFactor);
                    x2Que.EnQue(x2Local);
                    x2Local = x2Que.DeQue<T>();
                }

                yLocal = yQue.AllocTensor<float>();
                ProcessX1AndX2(yLocal, x1Local, x2Local, hFactor, hasX1, hasX2);
                ProcessYWithInput(rowOuterIdx, hIdx, hFactor);
                yQue.EnQue(yLocal);

                if (hasX1) {
                    x1Que.FreeTensor(x1Local);
                }
                if (hasX2) {
                    x2Que.FreeTensor(x2Local);
                }
                yLocal = yQue.DeQue<float>();
                int64_t yGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h +
                                    rowOuterIdx * tilingData->h + hIdx * tilingData->hFactor;
                CopyOut(yLocal.template ReinterpretCast<T>(), yGm[yGmOffset], 1, hFactor);
                yQue.FreeTensor(yLocal);
            }
        }
    }

private:
    __aicore__ inline void ProcessYWithInput(int64_t rowOuterIdx, int64_t hIdx, int64_t hFactor)
    {
        for (int64_t kIdx = 0; kIdx < tilingData->k; kIdx += 1) {
            ProcessInnerLoopBody(rowOuterIdx, hIdx, hFactor, kIdx);
        }
        CastYOutput(hFactor);
    }

    __aicore__ inline void ProcessInnerLoopBody(int64_t rowOuterIdx, int64_t hIdx, int64_t hFactor, int64_t kIdx)
    {
        SetExpandedRowIdxOffset(rowOuterIdx, kIdx);
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
            SetExpertIdxOffset(rowOuterIdx, kIdx);
            expertIdx = expertIdxGm.GetValue(expertIdxOffset);
        }
        if (hasX && hasExpertIdx) {
            if (!LoadExpandedXByExpertType(rowOuterIdx, hIdx, hFactor, expertIdx, expandedRowIdxGmValue)) {
                return;
            }
        } else {
            expandedXLocal = expandedXQue.AllocTensor<T>();
            CopyIn(expandedXGm[expandedRowIdxGmValue * tilingData->h + hIdx * tilingData->hFactor], expandedXLocal, 1,
                   hFactor);
        }
        expandedXQue.EnQue(expandedXLocal);
        expandedXLocal = expandedXQue.DeQue<T>();
        if (hasBiasAndExpertIdx) {
            int64_t biasGmOffset = expertIdxGm.GetValue(expertIdxOffset) * tilingData->h + hIdx * tilingData->hFactor;
            biasLocal = biasQue.AllocTensor<T>();
            CopyIn(biasGm[biasGmOffset], biasLocal, 1, hFactor);
            biasQue.EnQue(biasLocal);
        }
        if (hasScales) {
            SetScaleOffset(rowOuterIdx, kIdx);
            scale = scalesGm.GetValue(scaleOffset);
        }
        if (hasBiasAndExpertIdx) {
            biasLocal = biasQue.DeQue<T>();
        }
        ProcessExpandXBiasScale<T, S>(yLocal, expandedXLocal, biasLocal, scale, hFactor, hasBiasAndExpertIdx,
                                      hasScales);
        expandedXQue.FreeTensor(expandedXLocal);
        if (hasBiasAndExpertIdx) {
            biasQue.FreeTensor(biasLocal);
        }
    }

    __aicore__ inline bool LoadExpandedXByExpertType(int64_t rowOuterIdx, int64_t hIdx, int64_t hFactor,
                                                     int64_t expertIdx, int64_t expandedRowIdxGmValue)
    {
        if (expertIdx >= tilingData->zeroExpertStart && expertIdx < tilingData->zeroExpertEnd) {
            return false;
        }
        expandedXLocal = expandedXQue.AllocTensor<T>();
        if (expertIdx >= tilingData->copyExpertStart && expertIdx < tilingData->copyExpertEnd) {
            CopyCopyExpert(rowOuterIdx, hIdx, hFactor);
        } else if (hasConstExpert && expertIdx >= tilingData->constantExpertStart &&
                   expertIdx < tilingData->constantExpertEnd) {
            CopyConstantExpert(rowOuterIdx, hIdx, hFactor, expertIdx);
        } else {
            CopyIn(expandedXGm[expandedRowIdxGmValue * tilingData->h + hIdx * tilingData->hFactor], expandedXLocal, 1,
                   hFactor);
        }
        return true;
    }

    __aicore__ inline void CopyCopyExpert(int64_t rowOuterIdx, int64_t hIdx, int64_t hFactor)
    {
        int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h + rowOuterIdx * tilingData->h +
                            hIdx * tilingData->hFactor;
        CopyIn(xGm[xGmOffset], expandedXLocal, 1, hFactor);
    }

    __aicore__ inline void CopyConstantExpert(int64_t rowOuterIdx, int64_t hIdx, int64_t hFactor, int64_t expertIdx)
    {
        constExpertAlpha1Local = constExpertAlpha1Que.AllocTensor<T>();
        constExpertAlpha2Local = constExpertAlpha2Que.AllocTensor<T>();
        vLocal = vQue.AllocTensor<T>();

        int64_t xGmOffset = GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->h + rowOuterIdx * tilingData->h +
                            hIdx * tilingData->hFactor;
        CopyIn(xGm[xGmOffset], expandedXLocal, 1, hFactor);
        int64_t constExpertGmOffset =
            (expertIdx - tilingData->constantExpertStart) * tilingData->h + hIdx * tilingData->hFactor;
        CopyIn(constExpertAlpha1Gm[constExpertGmOffset], constExpertAlpha1Local, 1, hFactor);
        CopyIn(constExpertAlpha2Gm[constExpertGmOffset], constExpertAlpha2Local, 1, hFactor);
        CopyIn(vGm[constExpertGmOffset], vLocal, 1, hFactor);
        PipeBarrier<PIPE_ALL>();
        vLocal = vLocal * constExpertAlpha2Local;
        expandedXLocal = expandedXLocal * constExpertAlpha1Local;
        expandedXLocal = expandedXLocal + vLocal;

        constExpertAlpha1Que.EnQue(constExpertAlpha1Local);
        constExpertAlpha1Local = constExpertAlpha1Que.DeQue<T>();
        constExpertAlpha2Que.EnQue(constExpertAlpha2Local);
        constExpertAlpha2Local = constExpertAlpha2Que.DeQue<T>();
        vQue.EnQue(vLocal);
        vLocal = vQue.DeQue<T>();
        constExpertAlpha1Que.FreeTensor(constExpertAlpha1Local);
        constExpertAlpha2Que.FreeTensor(constExpertAlpha2Local);
        vQue.FreeTensor(vLocal);
    }

    __aicore__ inline void CastYOutput(int64_t hFactor)
    {
        if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
            Cast(yLocal.template ReinterpretCast<T>(), yLocal, RoundMode::CAST_RINT, hFactor);
        }
    }

    __aicore__ inline void SetExpandedRowIdxOffset(int64_t rowOuterIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expandedRowIdxOffset = GetBlockIdx() * tilingData->rowOfFormerBlock + rowOuterIdx + kIdx * tilingData->row;
        } else {
            expandedRowIdxOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k + rowOuterIdx * tilingData->k + kIdx;
        }
    }

    __aicore__ inline void SetExpertIdxOffset(int64_t rowOuterIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            expertIdxOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k + rowOuterIdx * tilingData->k + kIdx;
        } else {
            expertIdxOffset = expandedRowIdxOffset;
        }
    }

    __aicore__ inline void SetScaleOffset(int64_t rowOuterIdx, int64_t kIdx)
    {
        if constexpr (dropPadMode == DROPLESS_COLUMN || dropPadMode == DROP_PAD_COLUMN) {
            scaleOffset =
                GetBlockIdx() * tilingData->rowOfFormerBlock * tilingData->k + rowOuterIdx * tilingData->k + kIdx;
        } else {
            scaleOffset = expandedRowIdxOffset;
        }
    }

private:
    TPipe* pipe;
    const MoeFinalizeRoutingV2RegbaseTilingData* tilingData;
    GlobalTensor<T> expandedXGm;
    GlobalTensor<int32_t> expandedRowIdxGm;
    GlobalTensor<int32_t> expertIdxGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> xGm;
    GlobalTensor<T> constExpertAlpha1Gm;
    GlobalTensor<T> constExpertAlpha2Gm;
    GlobalTensor<T> vGm;

    int64_t expandedRowIdxOffset{0};
    int64_t expertIdxOffset{0};
    int64_t scaleOffset{0};

    GlobalTensor<T> biasGm;
    GlobalTensor<S> scalesGm;

    LocalTensor<T> expandedXLocal;
    LocalTensor<T> x1Local;
    LocalTensor<T> x2Local;
    LocalTensor<T> biasLocal;
    LocalTensor<T> constExpertAlpha1Local;
    LocalTensor<T> constExpertAlpha2Local;
    LocalTensor<T> vLocal;
    LocalTensor<float> yLocal;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> expandedXQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> biasQue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> x2Que;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> constExpertAlpha1Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> constExpertAlpha2Que;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> vQue;

    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> yQue;

    S scale;
    bool hasX1{false};
    bool hasX2{false};
    bool hasBiasAndExpertIdx{false};
    bool hasExpertIdx{false};
    bool hasScales{false};
    bool hasX{false};
    bool hasConstExpert{false};
};

#ifdef MOE_FINALIZE_ROUTING_HAS_MXFP4
// MXFP4 specialization: BF16 output and FP32 accumulation.
template <int32_t dropPadMode>
class MoeFinalizeRoutingV2HPartialLoad<fp4x2_e2m1_t, float8_e8m0_t, dropPadMode>
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
            for (int64_t h = 0; h < this->d_->h; h += this->d_->hFactor) {
                const uint32_t length = static_cast<uint32_t>(this->Min(this->d_->h - h, this->d_->hFactor));
                this->ComputeMx4BatchTile(row, h, length);
            }
        }
    }
};
#endif

#ifdef MOE_FINALIZE_ROUTING_HAS_MXFP8
// MXFP8 specializations: BF16 output and FP32 accumulation.
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

template <typename T, int32_t DROP_PAD_MODE>
class MoeFinalizeRoutingV2Mx8HPartialLoadImpl : public MoeFinalizeRoutingV2Mx8RouteBase<DROP_PAD_MODE> {
public:
    __aicore__ MoeFinalizeRoutingV2Mx8HPartialLoadImpl() {}

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
        auto hFactor = static_cast<uint32_t>(data_->hFactor);
        scaleNumAligned_ = CeilAlign(hFactor / MX8_SCALE_BLOCK_SIZE, MX8_UB_F32_PER_BLOCK);
        auto scaleTileLen = static_cast<uint32_t>(
            CeilAlign(static_cast<int64_t>(hFactor / MX8_SCALE_BLOCK_SIZE), static_cast<int64_t>(UBBLOCKSIZE)));
        auto rowsBf16Len = hFactor * sizeof(bfloat16_t);
        auto yLen = hFactor * sizeof(float);
        auto scratchLen = static_cast<uint32_t>(
            CeilAlign(Mx8ScratchBytes(hFactor, scaleNumAligned_), static_cast<uint32_t>(UBBLOCKSIZE)));

        pipe_->InitBuffer(expandedXQue_, 1, hFactor);
        pipe_->InitBuffer(scalesQue_, 1, scaleTileLen);
        if (hasX1_) {
            pipe_->InitBuffer(x1Que_, 1, rowsBf16Len);
        }
        if (hasX2_) {
            pipe_->InitBuffer(x2Que_, 1, rowsBf16Len);
        }
        if (hasBiasAndExpertIdx_) {
            pipe_->InitBuffer(biasQue_, 1, rowsBf16Len);
        }
        pipe_->InitBuffer(yQue_, 1, yLen);
        pipe_->InitBuffer(scratchBuf_, scratchLen);
        InitScaleIndex();
    }

    __aicore__ void Process()
    {
        int64_t rowOuterLoop = isTailCore_ ? data_->rowLoopOfTailBlock : data_->rowLoopOfFormerBlock;
        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; ++rowOuterIdx) {
            for (int64_t hIdx = 0; hIdx < data_->hLoop; hIdx += 1) {
                int64_t curHFactor = (hIdx == data_->hLoop - 1) ? data_->tailHFactor : data_->hFactor;
                int64_t elemOffset =
                    (coreIdx_ * data_->rowOfFormerBlock) * data_->h + rowOuterIdx * data_->h + hIdx * data_->hFactor;
                int64_t elemLen = curHFactor;

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
                ProcessInput(yLocal, rowOuterIdx, hIdx, curHFactor);
                Cast(yLocal.template ReinterpretCast<bfloat16_t>(), yLocal, RoundMode::CAST_RINT, elemLen);
                yQue_.EnQue(yLocal);

                yLocal = yQue_.DeQue<float>();
                CopyOut<bfloat16_t>(yLocal.template ReinterpretCast<bfloat16_t>(), yGm_[elemOffset], 1, elemLen);
                yQue_.FreeTensor(yLocal);
            }
        }
    }

private:
    __aicore__ void InitScaleIndex()
    {
        LocalTensor<float> scratch = scratchBuf_.Get<float>();
        LocalTensor<float> values, biasValues, scaleExpanded, scaleF32;
        LocalTensor<uint32_t> scaleIndex;
        Mx8GetScratchTensors(scratch, static_cast<uint32_t>(data_->hFactor), scaleNumAligned_, values, biasValues,
                             scaleExpanded, scaleF32, scaleIndex);
        Mx8BuildScaleIndex(scaleIndex, static_cast<uint32_t>(data_->hFactor));
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

    __aicore__ void ProcessInput(const LocalTensor<float>& yLocal, int64_t rowOuterIdx, int64_t hIdx,
                                 int64_t curHFactor)
    {
        // 尾块不足 32 个元素也占一个 scale，按向上取整拷入
        uint32_t numScales =
            static_cast<uint32_t>(Ops::Base::CeilDiv(curHFactor, static_cast<int64_t>(MX8_SCALE_BLOCK_SIZE)));
        const int64_t rowIndex = coreIdx_ * data_->rowOfFormerBlock + rowOuterIdx;
        for (int64_t kIdx = 0; kIdx < data_->k; ++kIdx) {
            int64_t gatherRowIdx, expert;
            if (!this->Route(rowIndex, kIdx, gatherRowIdx, expert)) {
                continue;
            }

            LocalTensor<uint8_t> expandedXLocal = expandedXQue_.AllocTensor<uint8_t>();
            DataCopyPad<uint8_t>(expandedXLocal, expandedXGm_[gatherRowIdx * data_->h + hIdx * data_->hFactor],
                                 DataCopyExtParams{1, static_cast<uint32_t>(curHFactor), 0, 0, 0},
                                 DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
            expandedXQue_.EnQue(expandedXLocal);

            LocalTensor<uint8_t> scalesLocal = scalesQue_.AllocTensor<uint8_t>();
            // scale 与数据同物理行（与 moe_init_routing_v3 的 MXFP8 scale 输出布局一致）
            int64_t scaleOffset = gatherRowIdx * scalesRowElemNum_ + (hIdx * data_->hFactor) / MX8_SCALE_BLOCK_SIZE;
            DataCopyPad<uint8_t>(scalesLocal, scalesGm_[scaleOffset], DataCopyExtParams{1, numScales, 0, 0, 0},
                                 DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
            scalesQue_.EnQue(scalesLocal);

            if (hasBiasAndExpertIdx_) {
                LocalTensor<bfloat16_t> biasLocal = biasQue_.AllocTensor<bfloat16_t>();
                int64_t biasGmOffset = expert * data_->h + hIdx * data_->hFactor;
                CopyIn<bfloat16_t>(biasGm_[biasGmOffset], biasLocal, 1, curHFactor);
                biasQue_.EnQue(biasLocal);
            }
            ComputeDequantize(yLocal, static_cast<uint32_t>(curHFactor));
        }
    }

    __aicore__ void ComputeDequantize(const LocalTensor<float>& yLocal, uint32_t elemNum)
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
        Mx8GetScratchTensors(scratch, static_cast<uint32_t>(data_->hFactor), scaleNumAligned_, values, biasValues,
                             scaleExpanded, scaleF32, scaleIndex);

        Mx8DequantizeRow<T>(yLocal, expandedXLocal, scalesLocal, biasLocal, values, biasValues, scaleExpanded, scaleF32,
                            scaleIndex, elemNum, hasBiasAndExpertIdx_, this->SlotWeight(0));

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

    int64_t scalesRowElemNum_{0};
    uint32_t scaleNumAligned_{0};
};

template <int32_t DROP_PAD_MODE>
class MoeFinalizeRoutingV2HPartialLoad<fp8_e4m3fn_t, float8_e8m0_t, DROP_PAD_MODE>
    : public MoeFinalizeRoutingV2Mx8HPartialLoadImpl<fp8_e4m3fn_t, DROP_PAD_MODE> {};

template <int32_t DROP_PAD_MODE>
class MoeFinalizeRoutingV2HPartialLoad<fp8_e5m2_t, float8_e8m0_t, DROP_PAD_MODE>
    : public MoeFinalizeRoutingV2Mx8HPartialLoadImpl<fp8_e5m2_t, DROP_PAD_MODE> {};
#endif

} // namespace MoeFinalizeRoutingV2Regbase

#endif
