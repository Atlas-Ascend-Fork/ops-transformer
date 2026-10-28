/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PARTIAL_ROTARY_MUL_QUANT_NO_ROPE_HIF8_H
#define PARTIAL_ROTARY_MUL_QUANT_NO_ROPE_HIF8_H

#include "partial_rotary_mul_quant_hif8_common.h"

namespace PartialRotaryMulQuantHiF8 {
template <typename T, AscendC::RoundMode roundMode>
class PartialRotaryMulQuantNoRope {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scale, GM_ADDR y,
                                const PartialRotaryMulQuantNoRopeTilingData* tilingData, AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
        tileElements_ = tilingData->tileElements;
        blockStart_ = static_cast<uint64_t>(AscendC::GetBlockIdx()) * tilingData->blockElements;
        blockElements_ = ops::Min(tilingData->blockElements, tilingData->totalElements - blockStart_);
        scale_ = LoadHiFloat8Scale(scale, tilingData->hasScale, tilingData->scale, pipe_, scaleBuffer_);
        xGm_.SetGlobalBuffer((__gm__ T*)x + blockStart_);
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + blockStart_);
        pipe_->InitBuffer(inputQueue_, DOUBLE_BUFFER, tileElements_ * sizeof(T));
        pipe_->InitBuffer(outputQueue_, DOUBLE_BUFFER, tileElements_ * sizeof(uint8_t));
    }

    __aicore__ inline void Process()
    {
        uint64_t loops = ops::CeilDiv(blockElements_, static_cast<uint64_t>(tileElements_));
        CopyIn(0, TileCount(0));
        for (uint64_t i = 0; i < loops; ++i) {
            // Prefetch into the other input buffer before computing this tile.
            // Queue events also allow its output DMA to overlap the next tile.
            if (i + 1 < loops) {
                CopyIn((i + 1) * tileElements_, TileCount(i + 1));
            }
            uint32_t count = TileCount(i);
            Compute(count);
            CopyOut(i * tileElements_, count);
        }
    }

private:
    __aicore__ inline uint32_t TileCount(uint64_t tile) const
    {
        return static_cast<uint32_t>(
            ops::Min(static_cast<uint64_t>(tileElements_), blockElements_ - tile * tileElements_));
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<T> input = inputQueue_.template AllocTensor<T>();
        AscendC::DataCopyExtParams params = {1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padding = {false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(input, xGm_[offset], params, padding);
        inputQueue_.EnQue(input);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        AscendC::LocalTensor<T> input = inputQueue_.template DeQue<T>();
        AscendC::LocalTensor<uint8_t> output = outputQueue_.template AllocTensor<uint8_t>();
        __local_mem__ T* in = (__local_mem__ T*)input.GetPhyAddr();
        __local_mem__ hifloat8_t* out = (__local_mem__ hifloat8_t*)output.GetPhyAddr();
        uint16_t repeats = static_cast<uint16_t>(ops::CeilDiv(count, VL_FLOAT32_SIZE));
        float scale = scale_;
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> value;
            AscendC::MicroAPI::RegTensor<hifloat8_t> quantized;
            AscendC::MicroAPI::MaskReg mask;
            uint32_t remaining = count;
            for (uint16_t i = 0; i < repeats; ++i) {
                mask = AscendC::MicroAPI::UpdateMask<float>(remaining);
                ops::LoadOneTensorForDtypeT<T>(in, value, mask, i * VL_FLOAT32_SIZE);
                AscendC::MicroAPI::Muls(value, value, scale, mask);
                AscendC::MicroAPI::Cast<hifloat8_t, float, Hif8CastTrait<roundMode>>(quantized, value, mask);
                AscendC::MicroAPI::DataCopy<hifloat8_t, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    out + i * VL_FLOAT32_SIZE, quantized, mask);
            }
        }
        inputQueue_.FreeTensor(input);
        outputQueue_.EnQue(output);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        AscendC::LocalTensor<uint8_t> output = outputQueue_.template DeQue<uint8_t>();
        AscendC::DataCopyExtParams params = {1, count, 0, 0, 0};
        AscendC::DataCopyPad(yGm_[offset], output, params);
        outputQueue_.FreeTensor(output);
    }

    AscendC::TPipe* pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, DOUBLE_BUFFER> inputQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, DOUBLE_BUFFER> outputQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBuffer_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<uint8_t> yGm_;
    uint64_t blockStart_ = 0;
    uint64_t blockElements_ = 0;
    uint32_t tileElements_ = 0;
    float scale_ = 1.0f;
};
} // namespace PartialRotaryMulQuantHiF8
#endif
