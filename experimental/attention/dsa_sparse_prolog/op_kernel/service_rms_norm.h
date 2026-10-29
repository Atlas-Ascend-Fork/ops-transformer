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
 * \file service_rms_norm.h
 * \brief
 */

#ifndef SERVICE_RMS_NORM_H
#define SERVICE_RMS_NORM_H

#include "dsa_sparse_prolog_comm.h"

namespace DsaSparseProlog {

constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr int32_t INDEX_TWO = 2;
constexpr int32_t INDEX_FOUR = 4;
constexpr int32_t INDEX_EIGHT = 8;
constexpr int32_t INDEX_SIXTEEN = 16;
constexpr int32_t HALf_INTERVAL = 2;
__aicore__ inline int32_t findPowerTwo(int32_t n1)
{
    // find max power of 2 no more than n (32 bit)
    n1 |= n1 >> 1; // Set the first digit of n's binary to 1
    n1 |= n1 >> INDEX_TWO;
    n1 |= n1 >> INDEX_FOUR;
    n1 |= n1 >> INDEX_EIGHT;
    n1 |= n1 >> INDEX_SIXTEEN;
    return (n1 + 1) >> 1;
}
__aicore__ inline float ReduceSumHalfInterval(const LocalTensor<float>& src_local4, int32_t count)
{
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local4, src_local4, src_local4[bodyCount], tailCount);
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(src_local4, src_local4, src_local4[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }

        AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    WholeReduceSum<float, false>(src_local4, src_local4, ELEM_PER_REP_FP32, 1, 0, 1, 0);
#else
    vcadd((__ubuf__ float*)src_local4.GetPhyAddr(), (__ubuf__ float*)src_local4.GetPhyAddr(), 1, 0, 1, 0, false);
#endif
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(event_v_s);
    WaitFlag<HardEvent::V_S>(event_v_s);
    return src_local4.GetValue(0);
}

// 只支持对单个batch的处理 row为1
template <typename T1, typename T2, typename C, typename O>
__aicore__ inline float RmsNorm(const GlobalTensor<T1>& inputGm, const GlobalTensor<T2>& gammaGm, float reciprocal,
                                float epsilon, float scale, int64_t row, int64_t col, LocalTensor<uint8_t>& shareTmpUb,
                                LocalTensor<O>& outputLocal, bool isTypeDemotion = false)
{
    int64_t cnt = row * col;
    uint32_t gammaLocalOffset = (2 * col + 8) * 2;
    LocalTensor<C> xFp32Local = shareTmpUb.ReinterpretCast<C>();

    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1); // wait for vector operations to finish
    // load input [1, col]
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(col * sizeof(T1)), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    if constexpr (std::is_same<T1, float>::value) {
        DataCopyPad(xFp32Local, inputGm, copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    } else {
        LocalTensor<T1> inputLocal = shareTmpUb.ReinterpretCast<T1>()[gammaLocalOffset + col];
        DataCopyPad(inputLocal, inputGm, copyParams, padParams);

        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // Cast input to fp32 [1, col]
        Cast(xFp32Local, inputLocal, RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
    }

    // load gamma [col]
    LocalTensor<T2> gammaLocal = shareTmpUb.ReinterpretCast<T2>()[gammaLocalOffset];
    DataCopy(gammaLocal, gammaGm, cnt);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

    // Calc: xSquare = xFp32 * xFp32, xSquare shape is [1, col]
    if constexpr (std::is_same<T1, float>::value) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    LocalTensor<C> xSquareLocal = xFp32Local[col];
    PipeBarrier<PIPE_V>();
    Mul(xSquareLocal, xFp32Local, xFp32Local, cnt);
    PipeBarrier<PIPE_V>();

    // ops-nn rms_norm
    float reduceOut = ReduceSumHalfInterval(xSquareLocal, col);
    float rstdValue = 1 / sqrt(reduceOut * reciprocal + epsilon);
    PipeBarrier<PIPE_V>();
    Muls(xFp32Local, xFp32Local, rstdValue, cnt);
    PipeBarrier<PIPE_V>();

    if (isTypeDemotion) {
        LocalTensor<T1> yLocal = shareTmpUb.ReinterpretCast<T1>()[gammaLocalOffset + 2 * col];
        Cast(yLocal, xFp32Local, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
        Cast(xFp32Local, yLocal, RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
    }

    // Cast gammaLocal to xSquareLocal (bf16 -> fp32) [col]
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    PipeBarrier<PIPE_V>();
    Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, col);
    PipeBarrier<PIPE_V>();

    // Calc: xFp32Local = xFp32Local * xSquareLocal [1, col] * [col]
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same<C, O>::value) {
        Mul(outputLocal, xFp32Local, xSquareLocal, col);
        PipeBarrier<PIPE_V>();

        if (scale > 1e-6f || scale < -1e-6f) {
            Muls(outputLocal, outputLocal, scale, cnt);
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_V>();
    } else {
        Mul(xFp32Local, xFp32Local, xSquareLocal, col);
        PipeBarrier<PIPE_V>();

        if (scale > 1e-6f || scale < -1e-6f) {
            if (isTypeDemotion) {
                LocalTensor<T1> yLocal = shareTmpUb.ReinterpretCast<T1>()[gammaLocalOffset + 2 * col];
                Cast(yLocal, xFp32Local, RoundMode::CAST_RINT, cnt);
                PipeBarrier<PIPE_V>();
                Cast(xFp32Local, yLocal, RoundMode::CAST_NONE, cnt);
                PipeBarrier<PIPE_V>();
            }
            Muls(xFp32Local, xFp32Local, scale, cnt);
            PipeBarrier<PIPE_V>();
        }
        // Cast xFp32 to outputLocal
        Cast(outputLocal, xFp32Local, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
    }
    return rstdValue;
}

// 只支持对单个batch的处理 row为1
template <typename T1, typename T2, typename C, typename O>
__aicore__ inline float RmsNormCq(const GlobalTensor<T1>& inputGm, const GlobalTensor<T2>& gammaGm, float reciprocal,
                                  float epsilon, float scale, int64_t row, int64_t col,
                                  LocalTensor<uint8_t>& shareTmpUb, LocalTensor<O>& outputLocal,
                                  bool isTypeDemotion = false)
{
    int64_t cnt = row * col;
    uint32_t gammaLocalOffset = (2 * col + 8) * 2;
    LocalTensor<C> xFp32Local = shareTmpUb.ReinterpretCast<C>();

    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1); // wait for vector operations to finish
    // load input [1, col]
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(col * sizeof(T1)), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    if constexpr (std::is_same<T1, float>::value) {
        DataCopyPad(xFp32Local, inputGm, copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    } else {
        LocalTensor<T1> inputLocal = shareTmpUb.ReinterpretCast<T1>()[gammaLocalOffset + col];
        DataCopyPad(inputLocal, inputGm, copyParams, padParams);

        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // Cast input to fp32 [1, col]
        Cast(xFp32Local, inputLocal, RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
    }

    // load gamma [col]
    LocalTensor<T2> gammaLocal = shareTmpUb.ReinterpretCast<T2>()[gammaLocalOffset];
    DataCopy(gammaLocal, gammaGm, cnt);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);

    // Calc: xSquare = xFp32 * xFp32, xSquare shape is [1, col]
    if constexpr (std::is_same<T1, float>::value) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    LocalTensor<C> xSquareLocal = xFp32Local[col];
    PipeBarrier<PIPE_V>();
    Mul(xSquareLocal, xFp32Local, xFp32Local, cnt); // x^2
    PipeBarrier<PIPE_V>();

    // ops-nn rms_norm
    float reduceOut = ReduceSumHalfInterval(xSquareLocal, col);
    float rstdValue = 1 / sqrt(reduceOut * reciprocal + epsilon);
    PipeBarrier<PIPE_V>();
    Muls(xFp32Local, xFp32Local, rstdValue, cnt);
    PipeBarrier<PIPE_V>();

    if (isTypeDemotion) {
        LocalTensor<T1> yLocal = shareTmpUb.ReinterpretCast<T1>()[gammaLocalOffset + 2 * col];
        Cast(yLocal, xFp32Local, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
        Cast(xFp32Local, yLocal, RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
    }

    // Cast gammaLocal to xSquareLocal (bf16 -> fp32) [col]
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
    PipeBarrier<PIPE_V>();
    Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, cnt);
    PipeBarrier<PIPE_V>();

    // Calc: xFp32Local = xFp32Local * xSquareLocal [1, col] * [col]
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same<C, O>::value) {
        Mul(outputLocal, xFp32Local, xSquareLocal, cnt);
        // if(scale > 1e-6f || scale < -1e-6f){
        //     Muls(outputLocal, outputLocal, scale, cnt);
        //     PipeBarrier<PIPE_V>();
        // }
        PipeBarrier<PIPE_V>();
    } else {
        Mul(xFp32Local, xFp32Local, xSquareLocal, cnt);
        PipeBarrier<PIPE_V>();
        // if(scale > 1e-6f || scale < -1e-6f){
        //     Muls(xFp32Local, xFp32Local, scale, cnt);
        //     PipeBarrier<PIPE_V>();
        // }
        // Cast xFp32 to outputLocal
        Cast(outputLocal, xFp32Local, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
    }
    return rstdValue;
}

template <typename T1, typename C, typename O>
__aicore__ inline void Scale(const GlobalTensor<T1>& inputGm, float scale, int64_t row, int64_t col,
                             LocalTensor<uint8_t>& shareTmpUb, LocalTensor<O>& outputLocal)
{
    int64_t cnt = row * col;
    LocalTensor<C> xFp32Local = shareTmpUb.ReinterpretCast<C>();

    SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID1); // wait for vector operations to finish
    // load input [1, col]
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(col * sizeof(T1)), 0, 0, 0};
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    if constexpr (std::is_same<T1, float>::value) {
        DataCopyPad(xFp32Local, inputGm, copyParams, padParams);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    } else {
        LocalTensor<T1> inputLocal = shareTmpUb.ReinterpretCast<T1>()[col * sizeof(C) / sizeof(T1)];
        DataCopyPad(inputLocal, inputGm, copyParams, padParams);

        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
        // Cast input to fp32 [1, col]
        Cast(xFp32Local, inputLocal, RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
    }

    if constexpr (std::is_same<T1, float>::value) {
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same<C, O>::value) {
        if (scale > 1e-6f || scale < -1e-6f) {
            Muls(outputLocal, xFp32Local, scale, cnt);
            PipeBarrier<PIPE_V>();
        } else {
            Muls(outputLocal, xFp32Local, 1.0f, cnt);
            PipeBarrier<PIPE_V>();
        }
    } else {
        if (scale > 1e-6f || scale < -1e-6f) {
            Muls(xFp32Local, xFp32Local, scale, cnt);
            PipeBarrier<PIPE_V>();
        }
        // Cast xFp32 to outputLocal
        Cast(outputLocal, xFp32Local, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
    }
}

} // namespace DsaSparseProlog

#endif
