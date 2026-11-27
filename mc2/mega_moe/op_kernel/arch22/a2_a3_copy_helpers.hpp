/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_A2_A3_COPY_HELPERS_HPP
#define MEGA_MOE_A2_A3_COPY_HELPERS_HPP
#include "kernel_operator.h"
#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "utils/const_args.hpp"

namespace Catlass::Gemm::Kernel::Detail {
using namespace AscendC;

template <class ElementD2>
struct SharedInputQuantBuffers {
    AscendC::LocalTensor<ElementD2> inputLocal;
    AscendC::LocalTensor<float> inputFp32;
    AscendC::LocalTensor<float> tempFp32;
    AscendC::LocalTensor<int8_t> quantLocal;
    AscendC::LocalTensor<float> reduceMaxLocal;
    AscendC::LocalTensor<int4b_t> highI4;
    AscendC::LocalTensor<int4b_t> lowI4;
    AscendC::LocalTensor<half> highHalf;
    AscendC::LocalTensor<half> lowHalf;
    AscendC::LocalTensor<half> lowHalf2;
    AscendC::LocalTensor<int16_t> lowMask;
};

template <bool IsW4A8, class ElementD2, class ArchTag>
CATLASS_DEVICE void InitSharedInputQuantBuffers(Arch::Resource<ArchTag>& resource, uint32_t k,
                                                SharedInputQuantBuffers<ElementD2>& buffers)
{
    uint32_t ubOffset = 0U;
    buffers.inputLocal = resource.ubBuf.template GetBufferByByte<ElementD2>(ubOffset);
    ubOffset += AlignUp(k * sizeof(ElementD2), 32U);
    buffers.inputFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    ubOffset += AlignUp(k * sizeof(float), 32U);
    buffers.tempFp32 = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    ubOffset += AlignUp(k * sizeof(float), 32U);
    buffers.quantLocal = resource.ubBuf.template GetBufferByByte<int8_t>(ubOffset);
    ubOffset += AlignUp(k * sizeof(int8_t), 32U);
    buffers.reduceMaxLocal = resource.ubBuf.template GetBufferByByte<float>(ubOffset);
    ubOffset += 32U;

    if constexpr (IsW4A8) {
        buffers.highI4 = resource.ubBuf.template GetBufferByByte<int4b_t>(ubOffset);
        ubOffset += AlignUp(k / 2U, 32U);
        buffers.lowI4 = resource.ubBuf.template GetBufferByByte<int4b_t>(ubOffset);
        ubOffset += AlignUp(k / 2U, 32U);
        buffers.highHalf = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
        ubOffset += AlignUp(k * sizeof(half), 32U);
        buffers.lowHalf = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
        ubOffset += AlignUp(k * sizeof(half), 32U);
        buffers.lowHalf2 = resource.ubBuf.template GetBufferByByte<half>(ubOffset);
        ubOffset += AlignUp(k * sizeof(half), 32U);
        buffers.lowMask = resource.ubBuf.template GetBufferByByte<int16_t>(ubOffset);
        AscendC::Duplicate(buffers.lowMask, static_cast<int16_t>(0x0F0F), 128U);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <class ElementD2>
CATLASS_DEVICE float QuantizeSharedInputRow(SharedInputQuantBuffers<ElementD2>& buffers, uint32_t k)
{
    AscendC::Cast(buffers.inputFp32, buffers.inputLocal, AscendC::RoundMode::CAST_NONE, k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Abs(buffers.tempFp32, buffers.inputFp32, k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::ReduceMax(buffers.reduceMaxLocal, buffers.tempFp32, buffers.tempFp32, k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID6);

    const float absMax = buffers.reduceMaxLocal.GetValue(0U);
    const float quantScale = absMax > 0.0F ? 127.0F / absMax : 127.0F;
    const float dequantScale = 1.0F / quantScale;
    AscendC::SetFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::S_V>(EVENT_ID6);
    AscendC::Muls(buffers.tempFp32, buffers.inputFp32, quantScale, k);
    AscendC::PipeBarrier<PIPE_V>();
    // Match routed quantization: round in FP32 before narrowing.
    auto quantInt32 = buffers.tempFp32.template ReinterpretCast<int32_t>();
    auto quantHalf = buffers.tempFp32.template ReinterpretCast<half>();
    AscendC::Cast(quantInt32, buffers.tempFp32, AscendC::RoundMode::CAST_RINT, k);
    AscendC::SetDeqScale(static_cast<half>(1.0F));
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(quantHalf, quantInt32, AscendC::RoundMode::CAST_ROUND, k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(buffers.quantLocal, quantHalf, AscendC::RoundMode::CAST_TRUNC, k);
    AscendC::PipeBarrier<PIPE_V>();
    return dequantScale;
}

template <class ElementD2>
CATLASS_DEVICE void PackSharedInputInt4(SharedInputQuantBuffers<ElementD2>& buffers, uint32_t k)
{
    AscendC::Cast(buffers.highHalf, buffers.quantLocal, AscendC::RoundMode::CAST_NONE, k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Muls(buffers.highHalf, buffers.highHalf, static_cast<half>(0.0625F), k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(buffers.highI4, buffers.highHalf, AscendC::RoundMode::CAST_FLOOR, k);

    const uint32_t repeat = (k / 2U) / 128U;
    const uint32_t tail = (k % 256U) / 2U;
    if (repeat > 0U) {
        AscendC::And(buffers.lowHalf.template ReinterpretCast<int16_t>(),
                     buffers.quantLocal.template ReinterpretCast<int16_t>(), buffers.lowMask, 128U, repeat,
                     {1U, 1U, 1U, 8U, 8U, 0U});
    }
    if (tail > 0U) {
        AscendC::And(buffers.lowHalf[repeat * 128U].template ReinterpretCast<int16_t>(),
                     buffers.quantLocal[repeat * 256U].template ReinterpretCast<int16_t>(), buffers.lowMask, tail, 1U,
                     {1U, 1U, 1U, 8U, 8U, 0U});
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(buffers.lowHalf2, buffers.lowHalf.template ReinterpretCast<int8_t>(), AscendC::RoundMode::CAST_NONE,
                  k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(buffers.highHalf, buffers.lowHalf2, static_cast<half>(-8.0F), k);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(buffers.lowI4, buffers.highHalf, AscendC::RoundMode::CAST_NONE, k);
}

template <bool IsW4A8, class ElementD2>
CATLASS_DEVICE void WriteSharedInputRow(SharedInputQuantBuffers<ElementD2>& buffers,
                                        AscendC::GlobalTensor<int8_t>& output,
                                        AscendC::GlobalTensor<float>& outputScale, uint32_t row, uint32_t k,
                                        float dequantScale)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID6);
    if constexpr (IsW4A8) {
        AscendC::DataCopy(output[static_cast<uint64_t>(row) * k], buffers.highI4.template ReinterpretCast<int8_t>(),
                          k / 2U);
        AscendC::DataCopy(output[static_cast<uint64_t>(row) * k + k / 2U],
                          buffers.lowI4.template ReinterpretCast<int8_t>(), k / 2U);
    } else {
        AscendC::DataCopyPad(output[static_cast<uint64_t>(row) * k], buffers.quantLocal,
                             {1U, static_cast<uint32_t>(k * sizeof(int8_t)), 0U, 0U, 0U});
    }
    buffers.reduceMaxLocal.SetValue(0U, dequantScale);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID7);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID7);
    AscendC::DataCopyPad(outputScale[row], buffers.reduceMaxLocal,
                         {1U, static_cast<uint32_t>(sizeof(float)), 0U, 0U, 0U});
}

template <bool IsW4A8, class ElementD2, class ArchTag>
CATLASS_DEVICE void PrepareSharedQuantInput(Arch::Resource<ArchTag>& resource, GM_ADDR inputAddr, GM_ADDR outputAddr,
                                            GM_ADDR scaleAddr, uint32_t rowStart, uint32_t rowCount, uint32_t k)
{
    SharedInputQuantBuffers<ElementD2> buffers;
    InitSharedInputQuantBuffers<IsW4A8>(resource, k, buffers);
    AscendC::GlobalTensor<ElementD2> input;
    AscendC::GlobalTensor<int8_t> output;
    AscendC::GlobalTensor<float> outputScale;
    input.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD2*>(inputAddr));
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(outputAddr));
    outputScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scaleAddr));

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    for (uint32_t localRow = 0U; localRow < rowCount; ++localRow) {
        const uint32_t row = rowStart + localRow;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
        AscendC::DataCopyPad(buffers.inputLocal, input[static_cast<uint64_t>(row) * k],
                             {1U, static_cast<uint32_t>(k * sizeof(ElementD2)), 0U, 0U, 0U}, {});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID6);
        const float dequantScale = QuantizeSharedInputRow(buffers, k);
        if constexpr (IsW4A8) {
            PackSharedInputInt4(buffers, k);
        }
        WriteSharedInputRow<IsW4A8>(buffers, output, outputScale, row, k, dequantScale);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
}

CATLASS_DEVICE uint32_t DispatchWorkerCount(bool fastPath, uint32_t coreNum, uint32_t tasks, uint32_t maxWorkers)
{
    return fastPath ? 1U : max(1U, min(maxWorkers, coreNum / tasks));
}

template <class Params>
CATLASS_DEVICE bool HasDispatchBatchRows(const Params& params, uint32_t first, uint32_t groups,
                                         const LocalTensor<int32_t>& sendCounts, const LocalTensor<int32_t>& recvCounts,
                                         uint32_t recvStride)
{
    // All AIVs must agree before skipping the batch barrier.
    for (uint32_t peer = 0; peer < params.EP; ++peer) {
        for (uint32_t group = first; group < first + groups; ++group) {
            if (sendCounts.GetValue(peer * params.expertPerRank + group) != 0 ||
                recvCounts.GetValue(peer * recvStride + group) != 0) {
                return true;
            }
        }
    }
    return false;
}

template <bool BatchedDispatch>
CATLASS_DEVICE bool CanSkipCombineExpertSync(uint32_t ep, uint32_t coreNum, uint32_t peerWorkers)
{
    return BatchedDispatch && ep <= SERVER_RANK_SIZE_A2 && ep * peerWorkers <= coreNum;
}

// Contiguous slices preserve row order for payload and per-token scales.
CATLASS_DEVICE uint32_t PartitionDispatchRows(uint32_t rows, uint32_t workers, uint32_t worker, uint32_t& first)
{
    const uint32_t capacity = (rows + workers - 1) / workers;
    first = worker * capacity;
    return first < rows ? min(capacity, rows - first) : 0;
}

template <uint32_t BatchRows, typename T, typename CopyIn, typename CopyOut>
CATLASS_DEVICE void CopyDispatchRows(AscendC::GlobalTensor<T>& dst, AscendC::GlobalTensor<T>& src,
                                     AscendC::LocalTensor<T>& buffer, int32_t rows, uint32_t width, CopyIn& copyIn,
                                     CopyOut& copyOut)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int32_t row = 0; row < rows; row += BatchRows) {
        uint32_t count = 1;
        if constexpr (BatchRows > 1) {
            count = min(static_cast<uint32_t>(rows - row), BatchRows);
        }
        const auto offset = row * width;
        const layout::RowMajor tile{count, width};
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        copyIn(buffer, src[offset], tile, tile);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID0);
        copyOut(dst[offset], buffer, tile, tile);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <class ArchTag, typename T>
CATLASS_DEVICE void CopyGMToGMShared(Arch::Resource<ArchTag>& resource, AscendC::GlobalTensor<T> dst,
                                     AscendC::GlobalTensor<T> src, int32_t elemNum, int32_t ubMoveNum)
{
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);

    using TType = Gemm::GemmType<T, layout::RowMajor>;
    using CopyGmToUb = Epilogue::Tile::CopyGm2Ub<ArchTag, TType>;
    using CopyUbToGm = Epilogue::Tile::CopyUb2Gm<ArchTag, TType>;
    CopyGmToUb copyGmToUb;
    CopyUbToGm copyUbToGm;
    constexpr int32_t BufferNum = 2;
    int tmpBufferSize = 32 * 1024 / sizeof(T); // 32 KB
    AscendC::LocalTensor<T> tmpBuffer1 = resource.ubBuf.template GetBufferByByte<T>(0);
    tmpBuffer1.SetSize(tmpBufferSize);
    int tmpBufferOffset = 96 * 1024; // half of UB
    AscendC::LocalTensor<T> tmpBuffer2 = resource.ubBuf.template GetBufferByByte<T>(tmpBufferOffset);
    tmpBuffer2.SetSize(tmpBufferSize);

    int pingpongId = 0;
    auto processCount = CeilDiv(elemNum, ubMoveNum);
    for (uint32_t processIndex = 0; processIndex < processCount; ++processIndex) {
        uint32_t curProcessNum =
            (processIndex == processCount - 1) ? elemNum - ubMoveNum * (processCount - 1) : ubMoveNum;
        AscendC::TEventID EVENT_ID = pingpongId == 0 ? EVENT_ID0 : EVENT_ID1;
        AscendC::LocalTensor<T> buf = pingpongId == 0 ? tmpBuffer1 : tmpBuffer2;
        auto processOffset = processIndex * ubMoveNum;

        auto inputOffset = processOffset;
        auto outputOffset = processOffset;
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        copyGmToUb(buf, src[inputOffset], layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID);
        copyUbToGm(dst[outputOffset], buf, layout::RowMajor{1, curProcessNum}, layout::RowMajor{1, curProcessNum});

        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID);
        pingpongId = (pingpongId + 1) % BufferNum;
    }

    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
}

template <class ArchTag>
CATLASS_DEVICE void LoadDispatchMsdBatch(AscendC::LocalTensor<int8_t>& input, AscendC::LocalTensor<int8_t>& scaleSlab,
                                         AscendC::GlobalTensor<int8_t>& src, uint32_t first, uint32_t count,
                                         uint32_t hiddenSize, uint32_t inputStride, AscendC::TEventID event)
{
    using Type = Gemm::GemmType<int8_t, layout::RowMajor>;
    Epilogue::Tile::CopyGm2Ub<ArchTag, Type> copyIn;
    WaitFlag<HardEvent::V_MTE2>(event);
    WaitFlag<HardEvent::MTE3_MTE2>(event);
    copyIn(input, src[static_cast<uint64_t>(first) * inputStride], layout::RowMajor{count, hiddenSize},
           layout::RowMajor{count, hiddenSize, inputStride});
    copyIn(scaleSlab, src[static_cast<uint64_t>(first) * inputStride + hiddenSize], layout::RowMajor{count, 32},
           layout::RowMajor{count, 32, inputStride});
    SetFlag<HardEvent::MTE2_V>(event);
}

struct BatchedMsdBuffers {
    AscendC::LocalTensor<int8_t> input[2];
    AscendC::LocalTensor<int8_t> scaleSlab[2];
    AscendC::LocalTensor<float> compactScale[2];
    AscendC::LocalTensor<int4b_t> high[2];
    AscendC::LocalTensor<int4b_t> low[2];
    AscendC::LocalTensor<half> highHalf;
    AscendC::LocalTensor<half> lowHalf;
    AscendC::LocalTensor<half> lowHalf2;
    AscendC::LocalTensor<int16_t> mask;
};

template <class ArchTag>
CATLASS_DEVICE void InitBatchedMsdBuffers(Arch::Resource<ArchTag>& resource, uint32_t hiddenSize, uint32_t batchRows,
                                          BatchedMsdBuffers& buffers)
{
    const uint32_t capacity = batchRows * hiddenSize;
    size_t offset = 0;
    for (uint32_t stage = 0; stage < 2; ++stage) {
        buffers.input[stage] = resource.ubBuf.template GetBufferByByte<int8_t>(offset);
        offset += capacity;
        buffers.scaleSlab[stage] = resource.ubBuf.template GetBufferByByte<int8_t>(offset);
        offset += batchRows * 32;
        buffers.compactScale[stage] = resource.ubBuf.template GetBufferByByte<float>(offset);
        offset += 32;
        buffers.high[stage] = resource.ubBuf.template GetBufferByByte<int4b_t>(offset);
        offset += capacity / 2;
        buffers.low[stage] = resource.ubBuf.template GetBufferByByte<int4b_t>(offset);
        offset += capacity / 2;
        SetFlag<HardEvent::V_MTE2>(stage == 0 ? EVENT_ID6 : EVENT_ID7);
        SetFlag<HardEvent::MTE3_MTE2>(stage == 0 ? EVENT_ID6 : EVENT_ID7);
        SetFlag<HardEvent::MTE3_V>(stage == 0 ? EVENT_ID6 : EVENT_ID7);
    }
    buffers.highHalf = resource.ubBuf.template GetBufferByByte<half>(offset);
    offset += capacity * sizeof(half);
    buffers.lowHalf = resource.ubBuf.template GetBufferByByte<half>(offset);
    offset += capacity * sizeof(half);
    buffers.lowHalf2 = resource.ubBuf.template GetBufferByByte<half>(offset);
    offset += capacity * sizeof(half);
    buffers.mask = resource.ubBuf.template GetBufferByByte<int16_t>(offset);
    Duplicate(buffers.mask, static_cast<int16_t>(0x0F0F), 128);
    PipeBarrier<PIPE_V>();
}

CATLASS_DEVICE
void WriteBatchedMsdRows(BatchedMsdBuffers& buffers, uint32_t stage, uint32_t first, uint32_t count,
                         uint32_t hiddenSize, uint32_t outputStride, uint64_t gmOffsetA,
                         AscendC::GlobalTensor<int8_t>& gmOutput)
{
    const uint32_t elements = count * hiddenSize;
    const auto event = stage == 0 ? EVENT_ID6 : EVENT_ID7;
    Cast(buffers.highHalf, buffers.input[stage], RoundMode::CAST_NONE, elements);
    PipeBarrier<PIPE_V>();
    Muls(buffers.highHalf, buffers.highHalf, static_cast<half>(0.0625f), elements);
    PipeBarrier<PIPE_V>();
    Cast(buffers.high[stage], buffers.highHalf, RoundMode::CAST_FLOOR, elements);
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    DataCopyExtParams outputCopy{static_cast<uint16_t>(count), static_cast<uint32_t>(hiddenSize / 2), 0,
                                 static_cast<uint32_t>(outputStride - hiddenSize / 2), 0};
    const uint64_t out = gmOffsetA + static_cast<uint64_t>(first) * outputStride;
    DataCopyPad(gmOutput[out], buffers.high[stage].template ReinterpretCast<int8_t>(), outputCopy);
    And(buffers.lowHalf.template ReinterpretCast<int16_t>(), buffers.input[stage].template ReinterpretCast<int16_t>(),
        buffers.mask, 128, elements / 256, {1, 1, 1, 8, 8, 0});
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(event);
    Cast(buffers.lowHalf2, buffers.lowHalf.template ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, elements);
    PipeBarrier<PIPE_V>();
    Adds(buffers.highHalf, buffers.lowHalf2, static_cast<half>(-8), elements);
    PipeBarrier<PIPE_V>();
    Cast(buffers.low[stage], buffers.highHalf, RoundMode::CAST_NONE, elements);
    SetFlag<HardEvent::V_MTE3>(event);
    WaitFlag<HardEvent::V_MTE3>(event);
    DataCopyPad(gmOutput[out + outputStride / 2], buffers.low[stage].template ReinterpretCast<int8_t>(), outputCopy);
    SetFlag<HardEvent::MTE3_V>(event);
    SetFlag<HardEvent::MTE3_MTE2>(event);
}

template <class ArchTag>
CATLASS_DEVICE void FetchAndPreprocessInt8ToInt4BatchedShared(Arch::Resource<ArchTag>& resource,
                                                              AscendC::GlobalTensor<int8_t> gmOutput,
                                                              uint64_t gmOffsetA, AscendC::GlobalTensor<float> dstScale,
                                                              AscendC::GlobalTensor<int8_t> src, int32_t rows,
                                                              int32_t hiddenSize, uint32_t outputStride,
                                                              uint32_t inputPadding)
{
    if (rows <= 0)
        return;
    constexpr uint32_t BATCH_ROWS = 4;
    const uint32_t inputStride = hiddenSize + inputPadding;
    BatchedMsdBuffers buffers;
    InitBatchedMsdBuffers(resource, hiddenSize, BATCH_ROWS, buffers);
    LoadDispatchMsdBatch<ArchTag>(buffers.input[0], buffers.scaleSlab[0], src, 0,
                                  min(BATCH_ROWS, static_cast<uint32_t>(rows)), hiddenSize, inputStride, EVENT_ID6);
    uint32_t stage = 0;
    for (uint32_t first = 0; first < static_cast<uint32_t>(rows); first += BATCH_ROWS) {
        const uint32_t count = min(BATCH_ROWS, static_cast<uint32_t>(rows) - first);
        const auto event = stage == 0 ? EVENT_ID6 : EVENT_ID7;
        if (first + count < static_cast<uint32_t>(rows)) {
            const uint32_t nextStage = 1 - stage;
            LoadDispatchMsdBatch<ArchTag>(buffers.input[nextStage], buffers.scaleSlab[nextStage], src, first + count,
                                          min(BATCH_ROWS, static_cast<uint32_t>(rows) - first - count), hiddenSize,
                                          inputStride, nextStage == 0 ? EVENT_ID6 : EVENT_ID7);
        }
        WaitFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE3_V>(event);
        SetFlag<HardEvent::MTE2_S>(event);
        WaitFlag<HardEvent::MTE2_S>(event);
        auto scaleValues = buffers.scaleSlab[stage].template ReinterpretCast<float>();
        for (uint32_t row = 0; row < count; ++row) {
            buffers.compactScale[stage].SetValue(row, scaleValues.GetValue(row * 8));
        }
        SetFlag<HardEvent::S_MTE3>(event);
        WaitFlag<HardEvent::S_MTE3>(event);
        DataCopyPad(dstScale[first], buffers.compactScale[stage],
                    {1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0});
        WriteBatchedMsdRows(buffers, stage, first, count, hiddenSize, outputStride, gmOffsetA, gmOutput);
        stage = 1 - stage;
    }
    for (uint32_t stage = 0; stage < 2; ++stage) {
        const auto event = stage == 0 ? EVENT_ID6 : EVENT_ID7;
        WaitFlag<HardEvent::V_MTE2>(event);
        WaitFlag<HardEvent::MTE3_MTE2>(event);
        WaitFlag<HardEvent::MTE3_V>(event);
    }
}

constexpr int32_t INT4_MASK_ELEMENT_NUM = 128; // Number of int16 elements holding the 0x0F0F mask.

struct Int4PreprocessBuffers {
    AscendC::LocalTensor<int8_t> xTensor0;
    AscendC::LocalTensor<int8_t> tmpBuffer0;
    AscendC::LocalTensor<int8_t> xTensor1;
    AscendC::LocalTensor<int8_t> tmpBuffer1;
    AscendC::LocalTensor<int4b_t> xHighI4Tensor;
    AscendC::LocalTensor<int4b_t> xLowI4Tensor;
    AscendC::LocalTensor<half> xHighHalfTensor;
    AscendC::LocalTensor<half> xLowHalfTensor;
    AscendC::LocalTensor<half> xLowHalfTensor2;
    AscendC::LocalTensor<int16_t> xLowI16Tensor;
};

template <class ArchTag>
CATLASS_DEVICE Int4PreprocessBuffers InitInt4PreprocessBuffers(Arch::Resource<ArchTag>& resource, int32_t hiddenSize,
                                                               uint32_t inputPadding)
{
    Int4PreprocessBuffers buffers;
    uint32_t tmpBufferOffset = 0;

    // double buffer 定义： tmpBuffer 包含int8_t token + float scale, xTensor 只包含int8_t token
    buffers.xTensor0 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    buffers.tmpBuffer0 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    buffers.xTensor0.SetSize(hiddenSize);
    buffers.tmpBuffer0.SetSize(hiddenSize + inputPadding);
    tmpBufferOffset += (hiddenSize + inputPadding) * sizeof(int8_t);

    buffers.xTensor1 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    buffers.tmpBuffer1 = resource.ubBuf.template GetBufferByByte<int8_t>(tmpBufferOffset);
    buffers.xTensor1.SetSize(hiddenSize);
    buffers.tmpBuffer1.SetSize(hiddenSize + inputPadding);
    tmpBufferOffset += (hiddenSize + inputPadding) * sizeof(int8_t);
    // double buffer 定义结束

    buffers.xHighI4Tensor = resource.ubBuf.template GetBufferByByte<int4b_t>(tmpBufferOffset);
    buffers.xHighI4Tensor.SetSize(hiddenSize);
    tmpBufferOffset += hiddenSize / INT4_NUM_PER_BYTE;

    buffers.xLowI4Tensor = resource.ubBuf.template GetBufferByByte<int4b_t>(tmpBufferOffset);
    buffers.xLowI4Tensor.SetSize(hiddenSize);
    tmpBufferOffset += hiddenSize / INT4_NUM_PER_BYTE;

    buffers.xHighHalfTensor = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
    buffers.xHighHalfTensor.SetSize(hiddenSize * sizeof(half));
    tmpBufferOffset += hiddenSize * sizeof(half);

    buffers.xLowHalfTensor = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
    buffers.xLowHalfTensor.SetSize(hiddenSize * sizeof(half));
    tmpBufferOffset += hiddenSize * sizeof(half);

    buffers.xLowHalfTensor2 = resource.ubBuf.template GetBufferByByte<half>(tmpBufferOffset);
    buffers.xLowHalfTensor2.SetSize(hiddenSize * sizeof(half));
    tmpBufferOffset += hiddenSize * sizeof(half);

    buffers.xLowI16Tensor = resource.ubBuf.template GetBufferByByte<int16_t>(tmpBufferOffset);
    buffers.xLowI16Tensor.SetSize(INT4_MASK_ELEMENT_NUM * sizeof(int16_t));

    return buffers;
}

CATLASS_DEVICE
void WriteInt4Nibbles(AscendC::GlobalTensor<int8_t>& gmA1I4_I8, uint64_t absStartAddr, uint32_t outputStride,
                      int32_t hiddenSize, AscendC::LocalTensor<int8_t>& xTensor, Int4PreprocessBuffers& buffers,
                      AscendC::TEventID eventId, half oneSixteenth, size_t maskElementNum, size_t lenVk,
                      size_t lastLenVk)
{
    // 高四位处理开始
    Cast(buffers.xHighHalfTensor, xTensor, AscendC::RoundMode::CAST_NONE, hiddenSize);
    PipeBarrier<PIPE_V>();
    Muls(buffers.xHighHalfTensor, buffers.xHighHalfTensor, oneSixteenth, hiddenSize);
    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
    Cast(buffers.xHighI4Tensor, buffers.xHighHalfTensor, AscendC::RoundMode::CAST_FLOOR, hiddenSize);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID6);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID6);
    DataCopy(gmA1I4_I8[absStartAddr], buffers.xHighI4Tensor.ReinterpretCast<int8_t>(), hiddenSize / INT4_NUM_PER_BYTE);
    // 高四位处理结束

    // 低四位处理开始
    SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
    And(buffers.xLowHalfTensor.ReinterpretCast<int16_t>(), xTensor.ReinterpretCast<int16_t>(), buffers.xLowI16Tensor,
        maskElementNum, lenVk, {1, 1, 1, 8, 8, 0});
    if (lastLenVk > 0) {
        And(buffers.xLowHalfTensor[lenVk * maskElementNum].ReinterpretCast<int16_t>(),
            xTensor[lenVk * maskElementNum * TWO].ReinterpretCast<int16_t>(), buffers.xLowI16Tensor, lastLenVk, 1,
            {1, 1, 1, 8, 8, 0});
    }
    PipeBarrier<PIPE_V>();
    SetFlag<HardEvent::V_MTE2>(eventId);
    Cast(buffers.xLowHalfTensor2.ReinterpretCast<half>(), buffers.xLowHalfTensor.ReinterpretCast<int8_t>(),
         AscendC::RoundMode::CAST_NONE, hiddenSize);
    PipeBarrier<PIPE_V>();
    const half MINUS_EIGHT = static_cast<half>(-8);
    Adds(buffers.xHighHalfTensor, buffers.xLowHalfTensor2, MINUS_EIGHT, hiddenSize);
    PipeBarrier<PIPE_V>();
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    Cast(buffers.xLowI4Tensor, buffers.xHighHalfTensor.ReinterpretCast<half>(), AscendC::RoundMode::CAST_NONE,
         hiddenSize);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID7);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID7);
    DataCopy(gmA1I4_I8[absStartAddr + outputStride / INT4_NUM_PER_BYTE], buffers.xLowI4Tensor.ReinterpretCast<int8_t>(),
             hiddenSize / INT4_NUM_PER_BYTE);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
    // 低四位处理结束
}

CATLASS_DEVICE
void PreprocessInt4Rows(AscendC::GlobalTensor<int8_t>& gmA1I4_I8, uint64_t gmOffsetA,
                        AscendC::GlobalTensor<float>& dstScale, AscendC::GlobalTensor<int8_t>& src, int32_t rows,
                        int32_t hiddenSize, uint32_t outputStride, uint32_t copyInNum, Int4PreprocessBuffers& buffers,
                        half oneSixteenth, size_t maskElementNum, size_t lenVk, size_t lastLenVk)
{
    constexpr int32_t BufferNum = 2;
    int pingpongId = 0;
    for (uint32_t processIndex = 0; processIndex < rows; ++processIndex) {
        const uint64_t relStartAddr = processIndex * outputStride;
        uint64_t absStartAddr = gmOffsetA + relStartAddr;
        AscendC::TEventID eventId = pingpongId == 0 ? EVENT_ID6 : EVENT_ID7;
        AscendC::LocalTensor<int8_t> buf = pingpongId == 0 ? buffers.tmpBuffer0 : buffers.tmpBuffer1;
        AscendC::LocalTensor<int8_t> xTensor = pingpongId == 0 ? buffers.xTensor0 : buffers.xTensor1;
        AscendC::LocalTensor<float> bufScale = buf[hiddenSize].template ReinterpretCast<float>();
        auto inputOffset = processIndex * copyInNum;

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(eventId);
        AscendC::DataCopy(buf, src[inputOffset], copyInNum);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventId);
        // 远端 scale 拷贝到GM
        AscendC::DataCopyPad(dstScale[processIndex], bufScale, {1, 4, 0, 0, 0});
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventId);
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        WriteInt4Nibbles(gmA1I4_I8, absStartAddr, outputStride, hiddenSize, xTensor, buffers, eventId, oneSixteenth,
                         maskElementNum, lenVk, lastLenVk);
        pingpongId = (pingpongId + 1) % BufferNum;
    }
}

template <class ArchTag>
CATLASS_DEVICE void FetchAndPreprocessInt8ToInt4Shared(Arch::Resource<ArchTag>& resource,
                                                       AscendC::GlobalTensor<int8_t> gmA1I4_I8, uint64_t gmOffsetA,
                                                       AscendC::GlobalTensor<float> dstScale,
                                                       AscendC::GlobalTensor<int8_t> src, int32_t rows,
                                                       int32_t hiddenSize, uint32_t outputStride,
                                                       uint32_t inputPadding = ALIGN_512)
{
    auto buffers = InitInt4PreprocessBuffers(resource, hiddenSize, inputPadding);

    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);

    Duplicate(buffers.xLowI16Tensor, static_cast<int16_t>(0x0F0F), INT4_MASK_ELEMENT_NUM);
    PipeBarrier<PIPE_V>();
    const size_t lenVk = (hiddenSize / INT4_NUM_PER_BYTE) / INT4_MASK_ELEMENT_NUM;
    const size_t lastLenVk = (hiddenSize % 256) / 2;
    const half oneSixteenth = static_cast<half>(0.0625f);

    SetFlag<HardEvent::S_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::S_MTE2>(EVENT_ID6);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID6);
    SetFlag<HardEvent::V_MTE2>(EVENT_ID7);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
    SetFlag<HardEvent::MTE3_V>(EVENT_ID7);

    uint32_t copyInNum = hiddenSize + inputPadding;

    PreprocessInt4Rows(gmA1I4_I8, gmOffsetA, dstScale, src, rows, hiddenSize, outputStride, copyInNum, buffers,
                       oneSixteenth, INT4_MASK_ELEMENT_NUM, lenVk, lastLenVk);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID7);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID6);
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID7);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
}

} // namespace Catlass::Gemm::Kernel::Detail

#endif // MEGA_MOE_A2_A3_COPY_HELPERS_HPP
