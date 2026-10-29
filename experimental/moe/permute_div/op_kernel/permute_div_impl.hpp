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
 * \file permute_div_impl.hpp
 * \brief PermuteDiv kernel 实现
 */
#ifndef PERMUTE_DIV_IMPL_HPP_
#define PERMUTE_DIV_IMPL_HPP_

#include "kernel_operator.h"
#include "permute_div_tiling_def.h"

using namespace AscendC;

namespace ops_permute_div {

class PermuteDiv {
public:
    __aicore__ inline PermuteDiv(const PermuteDivKernelTilingData& tiling, TPipe* pipeIn)
        : tiling_(tiling),
          pipe_(pipeIn)
    {}

    template <typename T1, typename T2>
    __aicore__ inline T2 CeilDiv(const T1 a, const T2 b)
    {
        return static_cast<T2>((a + static_cast<T1>(b) - 1) / static_cast<T1>(b));
    }

    __aicore__ inline void Process(GM_ADDR in, GM_ADDR out)
    {
        blockIdx_ = GetBlockIdx();

        // gm tensor
        inputGm_.SetGlobalBuffer((__gm__ int32_t*)in);
        outputGm_.SetGlobalBuffer((__gm__ int32_t*)out);

        // 分核：indice分多核
        const uint32_t avgNum = CeilDiv(tiling_.indiceNum, tiling_.coreNum);
        const uint32_t tailNum = avgNum - 1;
        const uint32_t tailCoreNum = avgNum * tiling_.coreNum - tiling_.indiceNum;
        const uint32_t formerCoreNum = tiling_.coreNum - tailCoreNum;

        uint32_t curNum = avgNum;
        if (this->blockIdx_ >= formerCoreNum) {
            curNum = tailNum;
        }
        uint32_t curOffset = this->blockIdx_ * avgNum;
        if (this->blockIdx_ > formerCoreNum) {
            curOffset = formerCoreNum * avgNum + (this->blockIdx_ - formerCoreNum) * tailNum;
        }

        if (curNum <= 0) {
            return;
        }

        // local tensor
        pipe_->InitBuffer(indiceBuf_, MAX_UB);
        LocalTensor<int32_t> indice = indiceBuf_.Get<int32_t>(curNum);
        LocalTensor<float> indiceFloat = indice.template ReinterpretCast<float>();

        // load数据
        DataCopyPad(indice, inputGm_[curOffset],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(curNum * sizeof(int32_t)),
                     /*srcStride*/ 0, /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        // 计算
        const float multiplier = 1.0f / tiling_.divisor;
        const float eps = 0.4f / tiling_.divisor;
        Cast(indiceFloat, indice, RoundMode::CAST_NONE, curNum); // int32 → fp32
        PipeBarrier<PIPE_V>();
        Muls(indiceFloat, indiceFloat, multiplier, curNum); // x * (1/topk)
        PipeBarrier<PIPE_V>();
        Adds(indiceFloat, indiceFloat, eps, curNum); // 修正舍入误差
        PipeBarrier<PIPE_V>();
        Cast(indice, indiceFloat, RoundMode::CAST_FLOOR, curNum); // floor → int32
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        // 搬出
        DataCopyPad(outputGm_[curOffset], indice,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(curNum * sizeof(int32_t)), /*srcStride*/ 0,
                     /*dstStride*/ 0, 0});
    }

private:
    TPipe* pipe_;
    const PermuteDivKernelTilingData& tiling_;

    GlobalTensor<int32_t> inputGm_;
    GlobalTensor<int32_t> outputGm_;

    TBuf<> indiceBuf_;
    const uint32_t MAX_UB = 184 * 1024;
    int32_t blockIdx_;

    // sync events
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
};
} // namespace ops_permute_div

#endif // PERMUTE_DIV_IMPL_HPP_
