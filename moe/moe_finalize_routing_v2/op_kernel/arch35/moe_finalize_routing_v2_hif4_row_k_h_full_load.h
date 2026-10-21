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
 * \file moe_finalize_routing_v2_hif4_row_k_h_full_load.h
 * \brief
 */

#ifndef __MOE_FINALIZE_ROUTING_V2_HIF4_ROW_K_H_FULL_LOAD_H__
#define __MOE_FINALIZE_ROUTING_V2_HIF4_ROW_K_H_FULL_LOAD_H__
#include "moe_finalize_routing_v2_base.h"

namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;
using Ops::Base::CeilAlign;
using Ops::Base::CeilDiv;

template <int32_t DROP_PAD_MODE>
class MoeFinalizeRoutingV2Hif4RowKHFullLoad {
public:
    __aicore__ MoeFinalizeRoutingV2Hif4RowKHFullLoad(TPipe* pipe, const MoeFinalizeRoutingV2RegbaseTilingData* data)
        : pipe_(pipe),
          data_(data)
    {
        coreIdx_ = GetBlockIdx();
        coreNum_ = GetBlockNum();
        isTailCore_ = (coreIdx_ == coreNum_ - 1);
    }

    __aicore__ void Init(GM_ADDR expandedX, GM_ADDR expandedRowIdx, GM_ADDR x1, GM_ADDR x2, GM_ADDR bias,
                         GM_ADDR scales, GM_ADDR expertIdx, GM_ADDR y, GM_ADDR workspace)
    {
        hasX1_ = x1 != nullptr;
        hasX2_ = x2 != nullptr;
        hasBiasAndExpertIdx_ = (bias != nullptr) && (expertIdx != nullptr);
        // hasScales_ = scales != nullptr;
        expandedXGm_.SetGlobalBuffer((__gm__ uint8_t*)expandedX);
        expandedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdx);
        x1Gm_.SetGlobalBuffer((__gm__ bfloat16_t*)x1);
        x2Gm_.SetGlobalBuffer((__gm__ bfloat16_t*)x2);
        biasGm_.SetGlobalBuffer((__gm__ bfloat16_t*)bias);
        scalesGm_.SetGlobalBuffer((__gm__ uint32_t*)scales);
        expertIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expertIdx);
        yGm_.SetGlobalBuffer((__gm__ bfloat16_t*)y);

        // 计算各TQue单元宽度，前提条件h是64的倍数，因此h、h/2天然对齐32B，只有scalesRow不满足天然对齐
        // InnerLoop每次搬运
        scalesRowElemNum = static_cast<uint32_t>(data_->h / HIF4_BLOCK_SIZE);
        scalesRowLenAligned = CeilAlign(scalesRowElemNum * SIZEOF_B32, UBBLOCKSIZE);
        scalesRowElemNumAligned = scalesRowLenAligned / SIZEOF_B32;
        auto scalesLen = static_cast<uint32_t>(data_->k) * scalesRowLenAligned;
        if constexpr (DROP_PAD_MODE == DROPLESS_COLUMN || DROP_PAD_MODE == DROPLESS_ROW) {
            expandedXRowBytes = static_cast<uint32_t>(data_->row * data_->k * data_->h / NUM_FP4_IN_BYTE);
        } else {
            expandedXRowBytes = static_cast<uint32_t>(data_->e * data_->c * data_->h / NUM_FP4_IN_BYTE);
        }
        auto biasLen = static_cast<uint32_t>(data_->e * data_->h * sizeof(bfloat16_t));
        // OuterLoop每次搬运
        auto rowsLen = static_cast<uint32_t>(data_->rowFactor * data_->h * sizeof(bfloat16_t));

        // 初始化Que
        pipe_->InitBuffer(expandedXQue_, 1, expandedXRowBytes);
        pipe_->InitBuffer(scalesQue_, DOUBLE_BUFFER, data_->rowFactor * scalesLen);
        if (hasX1_) {
            pipe_->InitBuffer(x1Que_, DOUBLE_BUFFER, rowsLen);
        }
        if (hasX2_) {
            pipe_->InitBuffer(x2Que_, DOUBLE_BUFFER, rowsLen);
        }
        if (hasBiasAndExpertIdx_) {
            pipe_->InitBuffer(biasQue_, 1, biasLen);
        }
        pipe_->InitBuffer(yQue_, DOUBLE_BUFFER, rowsLen);
    }

    __aicore__ void Process()
    {
        // expandedX全拷贝
        expandedXLocal = expandedXQue_.AllocTensor<uint8_t>();
        DataCopyPad<uint8_t>(expandedXLocal, expandedXGm_, DataCopyExtParams{1, expandedXRowBytes, 0, 0, 0},
                             DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        expandedXQue_.EnQue(expandedXLocal);
        expandedXLocal = expandedXQue_.DeQue<uint8_t>();

        if (hasBiasAndExpertIdx_) {
            biasLocal = biasQue_.AllocTensor<bfloat16_t>();
            CopyIn<bfloat16_t>(biasGm_, biasLocal, 1, data_->e * data_->h);
            biasQue_.EnQue(biasLocal);
            biasLocal = biasQue_.DeQue<bfloat16_t>();
        }

        // ub大循环取tiling算的rowLoop
        int64_t rowOuterLoop = isTailCore_ ? data_->rowLoopOfTailBlock : data_->rowLoopOfFormerBlock;
        int64_t tailRowFactor = isTailCore_ ? data_->tailRowFactorOfTailBlock : data_->tailRowFactorOfFormerBlock;

        for (int64_t rowOuterIdx = 0; rowOuterIdx < rowOuterLoop; ++rowOuterIdx) {
            // 内部小循环取tiling算的ubFactor，即一次ub大循环能处理几行tokens（这里是以输出的y的行为基准）
            int64_t rowInnerLoop = (rowOuterIdx == rowOuterLoop - 1) ? tailRowFactor : data_->rowFactor;
            int64_t elemOffset = (coreIdx_ * data_->rowOfFormerBlock + rowOuterIdx * data_->rowFactor) *
                                 data_->h;             // 大循环处理的行的起始位置
            int64_t elemLen = rowInnerLoop * data_->h; // 一次大循环处理的元素个数（行数*每行长度h）
            // ub大循环负责拷入x1、x2
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

            // 拷入scales
            // 原算子逻辑里scales和expert_idx都是按行顺序拷入，而expanded_x却是通过每行expanded_row_idx存的索引值Gather拷入
            int64_t scaleOffset =
                (coreIdx_ * data_->rowOfFormerBlock + rowOuterIdx * data_->rowFactor) * data_->k * scalesRowElemNum;
            scalesLocal = scalesQue_.AllocTensor<uint32_t>();
            CopyIn<uint32_t>(scalesGm_[scaleOffset], scalesLocal, 1, rowInnerLoop * scalesRowElemNum * data_->k);
            scalesQue_.EnQue(scalesLocal);
            scalesLocal = scalesQue_.DeQue<uint32_t>();

            LocalTensor<bfloat16_t> yLocal = yQue_.AllocTensor<bfloat16_t>();
            // ub大循环负责构造y=0或y=x1或y=x1+x2
            ProcessX1AndX2BF16(yLocal, static_cast<uint32_t>(elemLen));
            // 内循环用rowInnerLoop*k行expandedX、scales、bias来计算rowInnerLoop行的y
            InnerLoopProcessInput(yLocal, rowOuterIdx, rowInnerLoop);
            yQue_.EnQue(yLocal);
            scalesQue_.FreeTensor(scalesLocal);

            // 拷出y
            yLocal = yQue_.DeQue<bfloat16_t>();
            CopyOut<bfloat16_t>(yLocal, yGm_[elemOffset], 1, elemLen);
            yQue_.FreeTensor(yLocal);
        }

        expandedXQue_.FreeTensor(expandedXLocal);
        if (hasBiasAndExpertIdx_) {
            biasQue_.FreeTensor(biasLocal);
        }
    }

private:
    __aicore__ void ProcessX1AndX2BF16(const LocalTensor<bfloat16_t>& yLocal, uint32_t elemNum)
    {
        LocalTensor<bfloat16_t> x1Local, x2Local;
        __local_mem__ bfloat16_t* x1LocalAddr;
        __local_mem__ bfloat16_t* x2LocalAddr;
        __local_mem__ bfloat16_t* yLocalAddr = (__local_mem__ bfloat16_t*)yLocal.GetPhyAddr();

        uint32_t vfLoopNum = CeilDiv(elemNum, VL_B16);
        if (hasX1_ && hasX2_) {
            x1Local = x1Que_.DeQue<bfloat16_t>();
            x2Local = x2Que_.DeQue<bfloat16_t>();
            x1LocalAddr = (__local_mem__ bfloat16_t*)x1Local.GetPhyAddr();
            x2LocalAddr = (__local_mem__ bfloat16_t*)x2Local.GetPhyAddr();
            VF_CALL<VFComputeYWithX1AndX2BF16<true>>(yLocalAddr, x1LocalAddr, x2LocalAddr, elemNum, vfLoopNum);
            x1Que_.FreeTensor(x1Local);
            x2Que_.FreeTensor(x2Local);
        } else if (hasX1_) {
            x1Local = x1Que_.DeQue<bfloat16_t>();
            x1LocalAddr = (__local_mem__ bfloat16_t*)x1Local.GetPhyAddr();
            VF_CALL<VFComputeYWithX1AndX2BF16<false>>(yLocalAddr, x1LocalAddr, x2LocalAddr, elemNum, vfLoopNum);
            x1Que_.FreeTensor(x1Local);
        } else { // x1和x2都没有
            AscendC::Duplicate(yLocal, static_cast<bfloat16_t>(0), elemNum);
        }
    }

    __aicore__ void InnerLoopProcessInput(const LocalTensor<bfloat16_t>& yLocal, int64_t rowOuterIdx,
                                          int64_t rowInnerLoop)
    {
        // 小循环负责每次搬入k行expandedX/scales/bias
        for (int64_t rowInnerIdx = 0; rowInnerIdx < rowInnerLoop; ++rowInnerIdx) {
            // 拷入expandedX、bias和scales，算出k行的有效行数validK
            ProcessExpandedXScaleBias(yLocal, rowOuterIdx, rowInnerIdx);
        }
    }

    __aicore__ uint32_t ProcessExpandedXScaleBias(const LocalTensor<bfloat16_t>& yLocal, int64_t rowOuterIdx,
                                                  int64_t rowInnerIdx)
    {
        uint32_t validK = 0;
        for (int64_t kIdx = 0; kIdx < data_->k; ++kIdx) {
            // seqRowIndex表示按顺序在整个expandedX内处于第几行
            int64_t seqRowIndex =
                (coreIdx_ * data_->rowOfFormerBlock + rowOuterIdx * data_->rowFactor + rowInnerIdx) * data_->k + kIdx;
            // 计算当前k循环要从expandedRowIdx的哪个位置获取token的索引
            int64_t ridx;
            if constexpr (DROP_PAD_MODE == DROPLESS_COLUMN || DROP_PAD_MODE == DROP_PAD_COLUMN) {
                ridx = coreIdx_ * data_->rowOfFormerBlock + rowOuterIdx * data_->rowFactor + rowInnerIdx +
                       kIdx * data_->row; // 按列索引
            } else {
                ridx = seqRowIndex; // 按行索引
            }
            // gatherRowIdx为当前需要拷入的token在expandedX里的索引
            int64_t gatherRowIdx = expandedRowIdxGm_.GetValue(ridx);
            if constexpr (DROP_PAD_MODE == DROP_PAD_COLUMN || DROP_PAD_MODE == DROP_PAD_ROW) {
                if (gatherRowIdx == INVALID_IDX) {
                    continue;
                }
            } else {
                if (gatherRowIdx >= data_->activeNum) {
                    continue;
                }
            }

            LocalTensor<bfloat16_t> innerBiasLocal;
            if (hasBiasAndExpertIdx_) {
                int64_t biasGmOffset = expertIdxGm_.GetValue(seqRowIndex) * data_->h;
                innerBiasLocal = biasLocal[biasGmOffset];
            }

            ComputeDequantize(yLocal[rowInnerIdx * data_->h], expandedXLocal[gatherRowIdx * data_->h / NUM_FP4_IN_BYTE],
                              scalesLocal[rowInnerIdx * scalesRowElemNum * data_->k + kIdx * scalesRowElemNum],
                              innerBiasLocal, hasBiasAndExpertIdx_);

            validK++;
        }
        return validK;
    }

    __aicore__ void ComputeDequantize(LocalTensor<bfloat16_t> yLocal, LocalTensor<uint8_t> expandedXLocal,
                                      LocalTensor<uint32_t> scalesLocal, LocalTensor<bfloat16_t> biasLocal,
                                      bool hasBias)
    {
        // 计算
        __local_mem__ bfloat16_t* yLocalAddr = (__local_mem__ bfloat16_t*)yLocal.GetPhyAddr();
        __local_mem__ fp4x2_e1m2_t* expandedXLocalAddr = (__local_mem__ fp4x2_e1m2_t*)expandedXLocal.GetPhyAddr();
        __local_mem__ uint32_t* scalesLocalAddr = (__local_mem__ uint32_t*)scalesLocal.GetPhyAddr();
        __local_mem__ bfloat16_t* biasLocalAddr = (__local_mem__ bfloat16_t*)biasLocal.GetPhyAddr();

        uint32_t rowElemNum = static_cast<uint32_t>(data_->h); // row总是对齐访问的
        uint32_t rowLoopNum = rowElemNum / VL_B16;
        uint32_t tailElemNum = rowElemNum - rowLoopNum * VL_B16;
        uint32_t tailLoopNum = CeilDiv(tailElemNum, VL_B16);
        if (hasBias) {
            VF_CALL<VFComputeDequantizeWithBias<true>>(yLocalAddr, expandedXLocalAddr, scalesLocalAddr, biasLocalAddr,
                                                       1, rowElemNum, scalesRowElemNumAligned, rowLoopNum, tailElemNum,
                                                       tailLoopNum);
        } else {
            VF_CALL<VFComputeDequantizeWithBias<false>>(yLocalAddr, expandedXLocalAddr, scalesLocalAddr, biasLocalAddr,
                                                        1, rowElemNum, scalesRowElemNumAligned, rowLoopNum, tailElemNum,
                                                        tailLoopNum);
        }
    }

    TPipe* pipe_{nullptr};
    const MoeFinalizeRoutingV2RegbaseTilingData* data_{nullptr};
    int64_t coreIdx_{0};
    int64_t coreNum_{0};
    bool isTailCore_{false};

    GlobalTensor<uint8_t> expandedXGm_; // 用uint8_t包装fp4x2_e1m2_t
    GlobalTensor<int32_t> expandedRowIdxGm_;
    GlobalTensor<bfloat16_t> x1Gm_;
    GlobalTensor<bfloat16_t> x2Gm_;
    GlobalTensor<bfloat16_t> biasGm_;
    GlobalTensor<uint32_t> scalesGm_;
    GlobalTensor<int32_t> expertIdxGm_;
    GlobalTensor<bfloat16_t> yGm_;

    TQue<QuePosition::VECIN, 1> expandedXQue_;
    TQue<QuePosition::VECIN, 1> biasQue_;
    TQue<QuePosition::VECIN, 1> x1Que_;
    TQue<QuePosition::VECIN, 1> x2Que_;
    TQue<QuePosition::VECIN, 1> scalesQue_;
    TQue<QuePosition::VECOUT, 1> yQue_;

    bool hasX1_{false};
    bool hasX2_{false};
    bool hasBiasAndExpertIdx_{false};

    LocalTensor<uint8_t> expandedXLocal;
    LocalTensor<uint32_t> scalesLocal;
    LocalTensor<bfloat16_t> biasLocal;

    uint32_t scalesRowLenAligned{0};
    uint32_t scalesRowElemNum{0};
    uint32_t scalesRowElemNumAligned{0};
    uint32_t expandedXRowBytes{0};
};
} // namespace MoeFinalizeRoutingV2Regbase

#endif
