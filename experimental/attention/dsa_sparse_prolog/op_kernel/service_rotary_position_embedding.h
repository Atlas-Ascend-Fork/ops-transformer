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
 * \file service_rotary_position_embedding.h
 * \brief
 */

#ifndef SERVICE_ROTARY_POSITION_EMBEDDING_H
#define SERVICE_ROTARY_POSITION_EMBEDDING_H

#include "dsa_sparse_prolog_comm.h"

namespace DsaSparseProlog {

// 只支持对单个batch的处理 row为1
template <typename T, typename C, typename O>
__aicore__ inline void RotaryPosEmb(const GlobalTensor<T>& inputGm, const LocalTensor<C>& cosLocal,
                                    const LocalTensor<C>& sinLocal, int64_t stride, int64_t row, int64_t col,
                                    LocalTensor<uint8_t>& shareTmpUb, LocalTensor<O>& outputLocal,
                                    GlobalTensor<C> channelDeqScaleGm = GlobalTensor<C>(), float deQuantScale = -1)
{
    int64_t cnt = row * col;
    // printf("row = %ld, col = %ld, cnt = %ld\n", row, col, cnt);

    // load kLocal [row, col]
    LocalTensor<T> kLocal = shareTmpUb.ReinterpretCast<T>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(T)),
                                 static_cast<uint32_t>((stride - col) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(kLocal, inputGm, copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    int64_t baseOffset;
    if constexpr (std::is_same<T, int32_t>::value) {
        baseOffset = cnt;
    } else {
        baseOffset = cnt / 2;
    }
    // cast input to fp32
    LocalTensor<C> kFp32Local = shareTmpUb.ReinterpretCast<C>()[baseOffset];
    Cast(kFp32Local, kLocal, RoundMode::CAST_NONE, cnt);
    PipeBarrier<PIPE_V>();

    LocalTensor<C> kFp32ReArrLocal = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt];
    LocalTensor<C> kFp32OutputLocal = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt * 2];
    uint64_t rsvdCnt = 0;
    if constexpr (std::is_same<T, int32_t>::value) { // 反量化
        DataCopyExtParams copyParams1{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(C)),
                                      static_cast<uint32_t>((stride - col) * sizeof(C)), 0, 0};
        DataCopyPadExtParams<C> padParams1{false, 0, 0, 0};
        DataCopyPad(kFp32ReArrLocal, channelDeqScaleGm, copyParams1, padParams1); // 复用内存
        SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

        Muls(kFp32Local, kFp32Local, deQuantScale, cnt);
        PipeBarrier<PIPE_V>();
        Mul(kFp32Local, kFp32Local, kFp32ReArrLocal, cnt);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<C> kFp32OutputLocalSinTmp = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt * 3];

    // 数据重排，准备偶数和奇数分量
    // kFp32ReArrLocal[:cnt/2] 保存偶数分量，kFp32ReArrLocal[cnt/2:] 保存奇数分量
    GatherMask(kFp32ReArrLocal, kFp32Local, 1, true, col * row, {1, 1, 0, 0}, rsvdCnt);
    GatherMask(kFp32ReArrLocal[cnt >> 1], kFp32Local, 2, true, col * row, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    uint8_t blockNumPerRow = col / (32 / sizeof(C));
    uint8_t blockNumPerRowHalf = blockNumPerRow >> 1;

    // 计算偶数分量 * cos，奇数分量 * cos
    Mul<C, true>(kFp32OutputLocal, kFp32ReArrLocal, cosLocal, col >> 1, row,
                 {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    Mul<C, true>(kFp32OutputLocal[col >> 1], kFp32ReArrLocal[cnt >> 1], cosLocal[col >> 1], col >> 1, row,
                 {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    // 计算奇数分量 * sin，偶数分量 * sin
    Mul<C, true>(kFp32OutputLocalSinTmp, kFp32ReArrLocal[cnt >> 1], sinLocal, col >> 1, row,
                 {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    Mul<C, true>(kFp32OutputLocalSinTmp[col >> 1], kFp32ReArrLocal, sinLocal[col >> 1], col >> 1, row,
                 {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    PipeBarrier<PIPE_V>();
    Add(kFp32OutputLocal, kFp32OutputLocal, kFp32OutputLocalSinTmp, cnt);
    PipeBarrier<PIPE_V>();

    if constexpr (std::is_same<O, C>::value) {
        DataCopy(outputLocal, kFp32OutputLocal, cnt);
        PipeBarrier<PIPE_V>();
    } else {
        Cast(outputLocal, kFp32OutputLocal, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
    }
}

// 只支持对单个batch的处理 row为1
template <typename T, typename C, typename O>
__aicore__ inline void RotaryPosEmb_noQuant(const GlobalTensor<T>& inputGm, const LocalTensor<C>& cosLocal,
                                            const LocalTensor<C>& sinLocal, int64_t stride, int64_t row, int64_t col,
                                            LocalTensor<uint8_t>& shareTmpUb, LocalTensor<O>& outputLocal,
                                            GlobalTensor<C> channelDeqScaleGm = GlobalTensor<C>(),
                                            bool isTypeDemotion = false, float deQuantScale = -1)
{
    int64_t cnt = row * col;
    // printf("row = %ld, col = %ld, cnt = %ld\n", row, col, cnt);
    PipeBarrier<PIPE_V>();
    // load kLocal [row, col]
    LocalTensor<T> kLocal = shareTmpUb.ReinterpretCast<T>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(T)),
                                 static_cast<uint32_t>((stride - col) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(kLocal, inputGm, copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    int64_t baseOffset;
    baseOffset = cnt / 2;
    // cast input to fp32
    LocalTensor<C> kFp32Local = shareTmpUb.ReinterpretCast<C>()[baseOffset];
    if constexpr (std::is_same<T, C>::value) {
        DataCopy(kFp32Local, kLocal, cnt);
    } else {
        Cast(kFp32Local, kLocal, RoundMode::CAST_NONE, cnt);
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<C> kFp32ReArrLocal = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt];
    LocalTensor<C> kFp32OutputLocal = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt * 2];
    uint64_t rsvdCnt = 0;
    LocalTensor<C> kFp32OutputLocalSinTmp = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt * 3];

    // 数据重排，准备偶数和奇数分量
    // kFp32ReArrLocal[:cnt/2] 保存偶数分量，kFp32ReArrLocal[cnt/2:] 保存奇数分量
    GatherMask(kFp32ReArrLocal, kFp32Local, 1, true, col * row, {1, 1, 0, 0}, rsvdCnt);
    GatherMask(kFp32ReArrLocal[cnt >> 1], kFp32Local, 2, true, col * row, {1, 1, 0, 0}, rsvdCnt);
    PipeBarrier<PIPE_V>();
    uint8_t blockNumPerRow = col / (32 / sizeof(C));
    uint8_t blockNumPerRowHalf = blockNumPerRow >> 1;

    // 计算偶数分量 * cos，奇数分量 * cos
    Mul<C, true>(kFp32OutputLocal, kFp32ReArrLocal, cosLocal, col >> 1, row,
                 {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    Mul<C, true>(kFp32OutputLocal[col >> 1], kFp32ReArrLocal[cnt >> 1], cosLocal[col >> 1], col >> 1, row,
                 {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    // 计算奇数分量 * sin，偶数分量 * sin
    Mul<C, true>(kFp32OutputLocalSinTmp, kFp32ReArrLocal[cnt >> 1], sinLocal, col >> 1, row,
                 {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    Mul<C, true>(kFp32OutputLocalSinTmp[col >> 1], kFp32ReArrLocal, sinLocal[col >> 1], col >> 1, row,
                 {1, 1, 1, blockNumPerRow, blockNumPerRowHalf, 0});
    PipeBarrier<PIPE_V>();
    if (isTypeDemotion) {
        LocalTensor<O> kF16OutputLocal =
            shareTmpUb.ReinterpretCast<O>()[(baseOffset + cnt * 4) * sizeof(C) / sizeof(O)];
        Cast(kF16OutputLocal, kFp32OutputLocal, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
        Cast(kF16OutputLocal[cnt], kFp32OutputLocalSinTmp, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();

        Cast(kFp32OutputLocal, kF16OutputLocal, RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
        Cast(kFp32OutputLocalSinTmp, kF16OutputLocal[cnt], RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
    }
    Add(kFp32OutputLocal, kFp32OutputLocal, kFp32OutputLocalSinTmp, cnt);
    PipeBarrier<PIPE_V>();
    if constexpr (std::is_same<O, C>::value) {
        DataCopy(outputLocal, kFp32OutputLocal, cnt);
        PipeBarrier<PIPE_V>();
    } else {
        Cast(outputLocal, kFp32OutputLocal, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
    }
}

// RotateHalfBf16 (torch_npu.npu_rotary_mul所对应的) 等价实现
template <typename T, typename C, typename O>
__aicore__ inline void RotaryPosEmb_noQuant_rope_fusion(const GlobalTensor<T>& inputGm, const LocalTensor<C>& cosLocal,
                                                        const LocalTensor<C>& sinLocal, int64_t stride, int64_t row,
                                                        int64_t col, LocalTensor<uint8_t>& shareTmpUb,
                                                        LocalTensor<O>& outputLocal)
{
    int64_t cnt = row * col;
    int64_t halfCol = col >> 1;
    PipeBarrier<PIPE_V>();

    // 1. CopyInX
    LocalTensor<T> kLocal = shareTmpUb.ReinterpretCast<T>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(T)),
                                 static_cast<uint32_t>((stride - col) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(kLocal, inputGm, copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

    int64_t baseOffset = cnt / 2;

    // 2. Cast
    LocalTensor<C> kFp32Local = shareTmpUb.ReinterpretCast<C>()[baseOffset];
    if constexpr (std::is_same<T, C>::value) {
        DataCopy(kFp32Local, kLocal, cnt);
    } else {
        Cast(kFp32Local, kLocal, RoundMode::CAST_NONE, cnt);
    }
    PipeBarrier<PIPE_V>();

    // 3. GatherMask, 交错→半分割 (逐行处理)
    LocalTensor<C> xFp32 = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt];
    uint64_t rsvdCnt = 0;
    for (int64_t r = 0; r < row; r++) {
        int64_t srcRowOffset = r * col;
        int64_t dstRowOffset = r * col;
        GatherMask(xFp32[dstRowOffset], kFp32Local[srcRowOffset], 1, true, col, {1, 1, 0, 0}, rsvdCnt);
        GatherMask(xFp32[dstRowOffset + halfCol], kFp32Local[srcRowOffset], 2, true, col, {1, 1, 0, 0}, rsvdCnt);
    }
    PipeBarrier<PIPE_V>();

    // 4. XNewCopy, 前后半交换
    LocalTensor<C> xNewFp32 = shareTmpUb.ReinterpretCast<C>()[baseOffset + cnt * 2];
    uint16_t halfColBlocks = static_cast<uint16_t>(halfCol * sizeof(C) / 32);
    // 每行总跨距 = blockLen + stride = halfColBlocks + halfColBlocks = col 的 block 数
    DataCopyParams xNewCopyParams{static_cast<uint16_t>(row), halfColBlocks, halfColBlocks, halfColBlocks};
    DataCopy(xNewFp32, xFp32[halfCol], xNewCopyParams);
    DataCopy(xNewFp32[halfCol], xFp32, xNewCopyParams);
    PipeBarrier<PIPE_V>();

    // 5. ComputeInner
    uint8_t colBlkNum = static_cast<uint8_t>(col * sizeof(C) / 32);
    Mul<C>(xFp32, xFp32, cosLocal, col, static_cast<int32_t>(row), {1, 1, 1, colBlkNum, colBlkNum, 0});
    Mul<C>(xNewFp32, xNewFp32, sinLocal, col, static_cast<int32_t>(row), {1, 1, 1, colBlkNum, colBlkNum, 0});
    Add<C>(xNewFp32, xNewFp32, xFp32, col, static_cast<int32_t>(row), {1, 1, 1, colBlkNum, colBlkNum, colBlkNum});
    PipeBarrier<PIPE_V>();

    // 6. Cast: fp32 → 输出类型
    if constexpr (std::is_same<O, C>::value) {
        DataCopy(outputLocal, xNewFp32, cnt);
        PipeBarrier<PIPE_V>();
    } else {
        Cast(outputLocal, xNewFp32, RoundMode::CAST_RINT, cnt);
        PipeBarrier<PIPE_V>();
    }
}

// 后量化处理
template <typename T, typename C, typename O>
__aicore__ inline void RopePostQuantPerChannel(LocalTensor<T>& inputLocal, GlobalTensor<C>& quantScaleGm,
                                               int64_t stride, int64_t row, int64_t col,
                                               LocalTensor<uint8_t>& shareTmpUb, LocalTensor<O>& outputLocal)
{
    int64_t cnt = row * col;

    // 加载量化缩放因子到本地
    LocalTensor<C> quantScaleLocal = shareTmpUb.ReinterpretCast<C>();
    DataCopyExtParams copyParams{static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(C)),
                                 static_cast<uint32_t>((stride - col) * sizeof(C)), 0, 0};
    DataCopyPadExtParams<C> padParams{false, 0, 0, 0};
    DataCopyPad(quantScaleLocal, quantScaleGm, copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID1);

    // 输入转fp32（如果不是float）
    LocalTensor<C> inFp32;
    if constexpr (std::is_same<T, float>::value) {
        inFp32 = inputLocal;
    } else {
        inFp32 = shareTmpUb.ReinterpretCast<C>()[cnt];
        Cast(inFp32, inputLocal, RoundMode::CAST_NONE, cnt);
        PipeBarrier<PIPE_V>();
    }
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);

    // 反量化（乘以缩放因子）
    Mul(inFp32, inFp32, quantScaleLocal, cnt);
    PipeBarrier<PIPE_V>();

    // 量化到int32
    LocalTensor<int32_t> tmpInt32 = shareTmpUb.ReinterpretCast<int32_t>();
    LocalTensor<half> tmpHalf = shareTmpUb.ReinterpretCast<half>()[cnt];
    PipeBarrier<PIPE_V>();
    Cast(tmpInt32, inFp32, RoundMode::CAST_RINT, cnt);
    PipeBarrier<PIPE_V>();
    SetDeqScale(static_cast<half>(1.0));
    PipeBarrier<PIPE_V>();
    Cast(tmpHalf, tmpInt32, RoundMode::CAST_ROUND, cnt);
    PipeBarrier<PIPE_V>();
    Cast(outputLocal, tmpHalf, RoundMode::CAST_TRUNC, cnt);
    PipeBarrier<PIPE_V>();
}

} // namespace DsaSparseProlog

#endif
