/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OPS_TRANSFORMER_MHC_MEGA_MHC_PROJECTION_H
#define OPS_TRANSFORMER_MHC_MEGA_MHC_PROJECTION_H

#include "compute/mega_mhc_pre.h"
#include "compute/projection_cube.h"
#include "transfer.h"

namespace MegaMhc::Schedule {
using namespace AscendC;

constexpr uint32_t kProjectionStageSlots = 2U;

/** Load a packed row block asynchronously; the event owns the destination until VF consumes it. */
template <typename T>
__aicore__ inline void PrefetchProjectionRows(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling,
                                              uint64_t firstRow, uint32_t rows, uint64_t kStart, uint32_t kCount,
                                              const LocalTensor<T>& input, event_t event)
{
    WaitFlag<HardEvent::V_MTE2>(event);
    DataCopyExtParams copyParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(kCount * sizeof(T)),
                                 static_cast<uint32_t>((tiling.k - kCount) * sizeof(T)), 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(input, tensors.x2Out[firstRow * tiling.k + kStart], copyParams, padParams);
    SetFlag<HardEvent::MTE2_V>(event);
}

/** Transfer a host-sized row block: packed x2 GM -> UB -> FP32 stage GM.
 * One strided load and one store replace per-row DMA handshakes; statistics share the conversion VF.
 */
template <typename T, typename StageTransfer, bool ASYNC_STAGE = false>
__aicore__ inline void StageProjectionRows(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling,
                                           uint64_t firstRow, uint32_t rows, uint64_t kStart, uint32_t kCount,
                                           uint64_t stageOffset, const LocalTensor<T>& input0,
                                           const LocalTensor<T>& input1, const LocalTensor<float>& stage,
                                           const LocalTensor<float>& squares, uint32_t tile, uint32_t tileCount,
                                           uint32_t kIndex)
{
    if (rows == 0U) {
        return;
    }
    const auto input = (tile & 1U) == 0U ? input0 : input1;
    const auto next = (tile & 1U) == 0U ? input1 : input0;
    const event_t event = (tile & 1U) == 0U ? EVENT_ID2 : EVENT_ID3;
    const event_t nextEvent = (tile & 1U) == 0U ? EVENT_ID3 : EVENT_ID2;
    if (tile == 0U) {
        PrefetchProjectionRows(tensors, tiling, firstRow, rows, kStart, kCount, input, event);
    }
    WaitFlag<HardEvent::MTE2_V>(event);
    if (tile + 1U < tileCount) {
        PrefetchProjectionRows(tensors, tiling, firstRow, rows, kStart + tiling.l1K,
                               Kernel::PlannedL1TileKCount(tiling, kIndex, tile + 1U), next, nextEvent);
    }
    const event_t stageEvent =
        tiling.stageUbSlots == kProjectionStageSlots && (tile & 1U) == 0U ? EVENT_ID6 : EVENT_ID7;
    if constexpr (ASYNC_STAGE) {
        WaitFlag<HardEvent::MTE3_V>(stageEvent);
    }
    Compute::PrepareProjectionTile<T, StageTransfer::kPackedNz>(
        reinterpret_cast<__ubuf__ T*>(input.GetPhyAddr()), rows, kCount,
        StageTransfer::kPackedNz ? tiling.stageRowStride : tiling.l1K,
        reinterpret_cast<__ubuf__ float*>(stage.GetPhyAddr()), reinterpret_cast<__ubuf__ float*>(squares.GetPhyAddr()));
    SetFlag<HardEvent::V_MTE2>(event);
    if constexpr (ASYNC_STAGE) {
        StageTransfer::StoreAivRowAsync(tensors.stage, stageOffset, stage, rows * tiling.l1K, tiling, stageEvent);
    } else {
        StageTransfer::StoreAivRow(tensors.stage, stageOffset, stage, rows * tiling.l1K, tiling);
    }
}

/** Store every ordered K partial in its own FP32 workspace plane. */
struct SeparatePartials {
    /** Offset(group,row)=group*T*stride+row*stride. */
    __aicore__ inline static uint64_t Offset(const MegaMhcTilingData& tiling, uint32_t group, uint64_t row)
    {
        return static_cast<uint64_t>(group) * tiling.totalRows * tiling.nzN + row * tiling.nzN;
    }
    __aicore__ inline static void BeforeWrite(uint32_t group, bool last) {}
    __aicore__ inline static void AfterWrite(bool last) {}
    __aicore__ inline static void End() {}
};

/** Accumulate ordered FP32 K partials into one BS-owned workspace plane. */
struct CompactPartials {
    /** Offset(row)=row*stride; one full-K Cube owns the row. */
    __aicore__ inline static uint64_t Offset(const MegaMhcTilingData& tiling, uint32_t group, uint64_t row)
    {
        return row * tiling.nzN;
    }
    /** Overwrite group zero, then add each subsequent complete partial in FP32. */
    __aicore__ inline static void BeforeWrite(uint32_t group, bool last)
    {
        if (last) {
            if (group == 0U) {
                DisableDmaAtomic();
            } else {
                SetAtomicAdd<float>();
            }
        }
    }
    /** Complete FIX before the next group's MMAD to preserve partial addition order. */
    __aicore__ inline static void AfterWrite(bool last)
    {
        if (last) {
            SetFlag<HardEvent::FIX_M>(EVENT_ID6);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID6);
        }
    }
    /** Restore non-atomic output mode after all Cube partial stores. */
    __aicore__ inline static void End()
    {
        DisableDmaAtomic();
    }
};

/** Empty wave policy for projection of already materialized x2 rows. */
struct NoWavePipeline {};

/**
 * Run the shared M/K loop: partial[M,Q]=sum_K(FP32(x2[M,K])*phi[Q,K]).
 * StageTransfer owns A handoff; PartialStore owns ordered K-partial writes.
 * Pipeline advances adjacent Post/Pre/RMS1 waves at the schedule progress points.
 */
template <typename T, typename StageTransfer, bool USE_HF32, bool WAVE_PRODUCER = false,
          typename Pipeline = NoWavePipeline, bool ASYNC_STAGE = true, typename PartialStore = SeparatePartials>
__aicore__ inline void RunProjectionScheduleImpl(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling,
                                                 TPipe* pipe)
{
    Cube::ProjectionCube<USE_HF32> cube;
    if ASCEND_IS_AIC {
        cube.Init(pipe, &tiling);
        StageTransfer::SeedPairCredits();
        const uint32_t aicIndex = GetBlockIdx();
        const uint32_t mIndex = aicIndex / tiling.kGrid;
        const uint32_t kIndex = aicIndex % tiling.kGrid;
        const uint32_t l1TileCount = Kernel::PlannedL1TileCount(tiling, kIndex);

        for (uint32_t wave = 0; wave < tiling.mWaveCount; ++wave) {
            const uint32_t mRows = Kernel::GetWaveRows(tiling, mIndex, wave);
            const uint64_t mStart =
                static_cast<uint64_t>(wave) * tiling.rowsPerWave + static_cast<uint64_t>(mIndex) * tiling.mTile;
            const uint64_t kStart = Kernel::PlannedKStart(tiling, kIndex);
            for (uint32_t tile = 0; tile < l1TileCount; ++tile) {
                uint32_t kCount = 0;
                uint32_t l0FullLoops = 0;
                uint32_t l0Tail = 0;
                uint32_t tailLoadKStep = 0;

                Kernel::GetPlannedL1Tile(tiling, kIndex, tile, kCount, l0FullLoops, l0Tail, tailLoadKStep);
                const uint64_t kTileStart = kStart + static_cast<uint64_t>(tile) * tiling.l1K;
                const uint32_t activeK = kCount;
                const uint32_t l1Slot = (wave * l1TileCount + tile) & 1U;
                const uint64_t stageSlot =
                    (static_cast<uint64_t>(aicIndex) * kProjectionSlots + l1Slot) * tiling.mTile * tiling.l1K;

                if (mRows != 0U) {
                    cube.LoadB(tensors.phi, static_cast<uint32_t>(kTileStart), static_cast<uint32_t>(activeK), l1Slot);
                }
                StageTransfer::WaitReady(0U);
                StageTransfer::WaitReady(1U);
                if (mRows != 0U) {
                    StageTransfer::LoadA1(cube, tensors.stage, stageSlot, mRows, tiling.nzM,
                                          static_cast<uint32_t>(activeK), l1Slot);
                }
                if constexpr (StageTransfer::kReturnAfterLoad) {
                    StageTransfer::ReturnCredit(0U);
                    StageTransfer::ReturnCredit(1U);
                }
                if (mRows != 0U) {
                    const uint32_t groupIndex = static_cast<uint32_t>(kTileStart / tiling.partialGroupK);
                    const uint64_t groupEnd = static_cast<uint64_t>(groupIndex + 1U) * tiling.partialGroupK;
                    const bool firstInPartial = kTileStart == static_cast<uint64_t>(groupIndex) * tiling.partialGroupK;
                    const bool lastInPartial = kTileStart + activeK >= groupEnd || kTileStart + activeK == tiling.k;
                    const uint64_t partialOffset = PartialStore::Offset(tiling, groupIndex, mStart);
                    PartialStore::BeforeWrite(groupIndex, lastInPartial);
                    cube.ComputeL1Tile(mRows, tiling.nzM, l0FullLoops, l0Tail, tailLoadKStep, firstInPartial,
                                       lastInPartial, l1Slot, tensors.partialMm, partialOffset);
                    PartialStore::AfterWrite(lastInPartial);
                }
                if constexpr (!StageTransfer::kReturnAfterLoad) {
                    StageTransfer::ReturnCredit(0U);
                    StageTransfer::ReturnCredit(1U);
                }
            }
            if constexpr (WAVE_PRODUCER) {
                Pipeline::PublishProjection(tiling, wave);
            }
        }
        cube.End();
        PartialStore::End();
    }

    if ASCEND_IS_AIV {
        TBuf<TPosition::VECCALC> rmsRowsBuffer;
        TBuf<TPosition::VECCALC> stageTileBuffer;
        TBuf<TPosition::VECCALC> prefetchBuffer;
        Kernel::InitUbBuffer(pipe, rmsRowsBuffer, tiling.projectionRows * sizeof(float));
        Kernel::InitUbBuffer(pipe, stageTileBuffer,
                             tiling.stageUbSlots * tiling.stageRowStride * tiling.l1K * sizeof(float));
        Kernel::InitUbBuffer(pipe, prefetchBuffer, tiling.projectionRows * tiling.l1K * sizeof(T));
        LocalTensor<T> prefetchInput = prefetchBuffer.Get<T>();
        if constexpr (ASYNC_STAGE) {
            SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
            if (tiling.stageUbSlots == kProjectionStageSlots) {
                SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
            }
        }
        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
        LocalTensor<float> rmsRows = rmsRowsBuffer.Get<float>();
        LocalTensor<float> stageTile = stageTileBuffer.Get<float>();
        __ubuf__ float* rmsRowsPtr = reinterpret_cast<__ubuf__ float*>(rmsRows.GetPhyAddr());

        const uint32_t aivBlockIndex = GetBlockIdx();
        const uint32_t aicIndex = aivBlockIndex / Kernel::kAivPerAic;
        const uint32_t subblock = GetSubBlockIdx();
        const uint32_t mIndex = aicIndex / tiling.kGrid;
        const uint32_t kIndex = aicIndex % tiling.kGrid;
        const uint32_t firstRows = tiling.projectionRows;
        const uint32_t l1TileCount = Kernel::PlannedL1TileCount(tiling, kIndex);
        const uint64_t kStart = Kernel::PlannedKStart(tiling, kIndex);

        TBuf<TPosition::VECCALC> x2TileBuffer;
        const uint32_t tileElements = tiling.projectionRows * tiling.l1K;
        Kernel::InitUbBuffer(pipe, x2TileBuffer, tileElements * sizeof(T));
        LocalTensor<T> x2Tile = x2TileBuffer.Get<T>();
        Pipeline pipeline;
        if constexpr (WAVE_PRODUCER) {
            pipeline.Init(tensors, tiling, pipe);
        }
        for (uint32_t wave = 0; wave < tiling.mWaveCount; ++wave) {
            if constexpr (WAVE_PRODUCER) {
                pipeline.BeginWave(wave);
            }
            const uint32_t mRows = Kernel::GetWaveRows(tiling, mIndex, wave);
            const uint64_t mStart =
                static_cast<uint64_t>(wave) * tiling.rowsPerWave + static_cast<uint64_t>(mIndex) * tiling.mTile;
            const uint32_t rowStart = subblock == 0U ? 0U : firstRows;
            const uint32_t rowCount =
                subblock == 0U ? (mRows < firstRows ? mRows : firstRows) : (mRows > firstRows ? mRows - firstRows : 0U);
            if (rowCount != 0U) {
                Kernel::ZeroFp32Values(rmsRowsPtr, rowCount);
            }

            for (uint32_t tile = 0; tile < l1TileCount; ++tile) {
                const uint32_t kCount = Kernel::PlannedL1TileKCount(tiling, kIndex, tile);
                const uint64_t kTileStart = kStart + static_cast<uint64_t>(tile) * tiling.l1K;
                const uint32_t activeK = kCount;
                const uint32_t slot = (wave * l1TileCount + tile) & 1U;
                const uint64_t stageSlot =
                    (static_cast<uint64_t>(aicIndex) * kProjectionSlots + slot) * tiling.mTile * tiling.l1K +
                    static_cast<uint64_t>(rowStart) * tiling.l1K;

                StageTransfer::AcquireSlot();
                StageProjectionRows<T, StageTransfer, ASYNC_STAGE>(
                    tensors, tiling, mStart + rowStart, rowCount, kTileStart, activeK, stageSlot, x2Tile, prefetchInput,
                    stageTile[tiling.stageUbSlots == kProjectionStageSlots ?
                                  (tile & 1U) * tiling.stageRowStride * tiling.l1K :
                                  0U],
                    rmsRows, tile, l1TileCount, kIndex);
                StageTransfer::PublishReady();
                if constexpr (WAVE_PRODUCER) {
                    pipeline.AfterTile(tensors, wave, tile);
                }
            }

            if (rowCount != 0U) {
                Kernel::CopyUbToGlobal<float>(tensors.partialRms,
                                              static_cast<uint64_t>(kIndex) * tiling.totalRows + mStart + rowStart,
                                              rmsRows, rowCount);
            }
            if constexpr (WAVE_PRODUCER) {
                pipeline.EndWave(tensors, wave);
            }
        }
        if constexpr (WAVE_PRODUCER) {
            pipeline.End(tensors);
        }

        WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
        StageTransfer::DrainCredits();
        if constexpr (ASYNC_STAGE) {
            WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
            if (tiling.stageUbSlots == kProjectionStageSlots) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
            }
        }
    }
}

/** Dispatch the host-selected transport once; each projection VF remains a compile-time layout specialization. */
template <typename T, typename StageTransfer, bool USE_HF32, bool WAVE_PRODUCER = false,
          typename Pipeline = NoWavePipeline, bool ASYNC_STAGE = true, typename PartialStore = SeparatePartials>
__aicore__ inline void RunProjectionSchedule(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe)
{
    if constexpr (StageTransfer::kPackedNz) {
        if (tiling.projectionTransport == static_cast<uint32_t>(ProjectionTransport::GmStage)) {
            RunProjectionScheduleImpl<T, Transfer::GmStage, USE_HF32, WAVE_PRODUCER, Pipeline, ASYNC_STAGE,
                                      PartialStore>(tensors, tiling, pipe);
            return;
        }
    }
    RunProjectionScheduleImpl<T, StageTransfer, USE_HF32, WAVE_PRODUCER, Pipeline, ASYNC_STAGE, PartialStore>(
        tensors, tiling, pipe);
}

constexpr uint16_t kProjectionInputEventBase = 2;

/** Reuse B[K,Q] across host-selected M waves; partial[M,Q]=sum_K(float(x2)*phi).
 * Each M stage owns an L0C slot. Return B's credit after its final nonempty M consumer.
 */
template <typename T, typename StageTransfer, bool USE_HF32, bool RESIDENT_B>
__aicore__ inline void RunWeightReuseCube(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe)
{
    const uint32_t aicIndex = GetBlockIdx();
    const uint32_t mIndex = aicIndex / tiling.kGrid;
    const uint32_t kIndex = aicIndex % tiling.kGrid;
    const uint32_t tileCount = Kernel::PlannedL1TileCount(tiling, kIndex);
    const uint64_t kStart = Kernel::PlannedKStart(tiling, kIndex);
    uint32_t sequence = 0U;
    Cube::ProjectionCube<USE_HF32> cube;
    cube.Init(pipe, &tiling);
    StageTransfer::SeedPairCredits();
    for (uint32_t waveBase = 0U; waveBase < tiling.mWaveCount; waveBase += tiling.projectionMStages) {
        const uint32_t stages = tiling.mWaveCount - waveBase < tiling.projectionMStages ? tiling.mWaveCount - waveBase :
                                                                                          tiling.projectionMStages;
        const uint32_t lastRows = Kernel::GetWaveRows(tiling, mIndex, waveBase + stages - 1U);
        for (uint32_t tile = 0U; tile < tileCount; ++tile) {
            uint32_t kCount = 0U;
            uint32_t l0FullLoops = 0U;
            uint32_t l0Tail = 0U;
            uint32_t tailLoadKStep = 0U;

            Kernel::GetPlannedL1Tile(tiling, kIndex, tile, kCount, l0FullLoops, l0Tail, tailLoadKStep);
            const uint64_t kOffset = kStart + static_cast<uint64_t>(tile) * tiling.l1K;
            const uint32_t bSlot = tile & 1U;
            if (Kernel::GetWaveRows(tiling, mIndex, waveBase) != 0U) {
                cube.LoadB(tensors.phi, static_cast<uint32_t>(kOffset), kCount, bSlot);
            }
            for (uint32_t stage = 0U; stage < stages; ++stage, ++sequence) {
                const uint32_t wave = waveBase + stage;
                const uint32_t rows = Kernel::GetWaveRows(tiling, mIndex, wave);
                const uint64_t mStart =
                    static_cast<uint64_t>(wave) * tiling.rowsPerWave + static_cast<uint64_t>(mIndex) * tiling.mTile;
                const uint32_t slot = sequence & 1U;
                const uint64_t stageOffset =
                    (static_cast<uint64_t>(aicIndex) * kProjectionSlots + slot) * tiling.mTile * tiling.l1K;
                StageTransfer::WaitReady(0U);
                StageTransfer::WaitReady(1U);
                if (rows != 0U) {
                    StageTransfer::LoadA1(cube, tensors.stage, stageOffset, rows, tiling.nzM, kCount, slot);
                }
                if constexpr (StageTransfer::kReturnAfterLoad) {
                    StageTransfer::ReturnCredit(0U);
                    StageTransfer::ReturnCredit(1U);
                }
                if (rows != 0U) {
                    const uint32_t group = static_cast<uint32_t>(kOffset / tiling.partialGroupK);
                    const bool first = kOffset == static_cast<uint64_t>(group) * tiling.partialGroupK;
                    const bool last = kOffset + kCount >= static_cast<uint64_t>(group + 1U) * tiling.partialGroupK ||
                                      kOffset + kCount == tiling.k;
                    const uint64_t partialOffset =
                        static_cast<uint64_t>(group) * tiling.totalRows * tiling.nzN + mStart * tiling.nzN;
                    cube.template ComputeL1Tile<true, RESIDENT_B>(
                        rows, tiling.nzM, l0FullLoops, l0Tail, tailLoadKStep, first, last, slot, tensors.partialMm,
                        partialOffset, bSlot, stage, stage + 1U == stages - static_cast<uint32_t>(lastRows == 0U));
                }
                if constexpr (!StageTransfer::kReturnAfterLoad) {
                    StageTransfer::ReturnCredit(0U);
                    StageTransfer::ReturnCredit(1U);
                }
            }
        }
    }
    cube.End();
}

/** Prefetch x2 M/K tiles, cast and pack A, and accumulate squares[M]=sum_K(float(x2)^2).
 * Input slots hide MTE2 behind V; output slots hide UB-to-L1 transfer behind the next V tile.
 */
template <typename T, typename StageTransfer>
__aicore__ inline void RunWeightReuseVector(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe)
{
    const uint32_t aicIndex = GetBlockIdx() / Kernel::kAivPerAic;
    const uint32_t mIndex = aicIndex / tiling.kGrid;
    const uint32_t kIndex = aicIndex % tiling.kGrid;
    const uint32_t tileCount = Kernel::PlannedL1TileCount(tiling, kIndex);
    const uint64_t kStart = Kernel::PlannedKStart(tiling, kIndex);
    uint32_t sequence = 0U;
    TBuf<TPosition::VECCALC> statisticsBuffer;
    TBuf<TPosition::VECCALC> stageBuffer;
    TBuf<TPosition::VECCALC> inputBuffer;
    // Each stage begins at a register boundary, including smaller M l0Tail geometry.
    Kernel::InitUbBuffer(pipe, statisticsBuffer,
                         tiling.projectionMStages * tiling.projectionStatsStride * sizeof(float));
    Kernel::InitUbBuffer(pipe, stageBuffer, tiling.stageUbSlots * tiling.stageRowStride * tiling.l1K * sizeof(float));
    Kernel::InitUbBuffer(pipe, inputBuffer,
                         tiling.projectionInputSlots * tiling.projectionRows * tiling.l1K * sizeof(T));
    auto statistics = statisticsBuffer.Get<float>();
    auto output = stageBuffer.Get<float>();
    auto input = inputBuffer.Get<T>();
    for (uint32_t inSlot = 0U; inSlot < tiling.projectionInputSlots; ++inSlot) {
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(kProjectionInputEventBase + inSlot));
    }
    SetFlag<HardEvent::MTE3_V>(EVENT_ID7);
    if (tiling.stageUbSlots == kProjectionStageSlots) {
        SetFlag<HardEvent::MTE3_V>(EVENT_ID6);
    }
    const uint32_t rowStart = GetSubBlockIdx() == 0U ? 0U : tiling.projectionRows;
    for (uint32_t waveBase = 0U; waveBase < tiling.mWaveCount; waveBase += tiling.projectionMStages) {
        const uint32_t stages = tiling.mWaveCount - waveBase < tiling.projectionMStages ? tiling.mWaveCount - waveBase :
                                                                                          tiling.projectionMStages;
        Kernel::ZeroFp32Values(reinterpret_cast<__ubuf__ float*>(statistics.GetPhyAddr()),
                               tiling.projectionMStages * tiling.projectionStatsStride);
        for (uint32_t prime = 0U; prime + 1U < tiling.projectionInputSlots && prime < tileCount * stages; ++prime) {
            const uint32_t primeStage = prime % stages;
            const uint32_t primeTile = prime / stages;
            const uint32_t waveRows = Kernel::GetWaveRows(tiling, mIndex, waveBase + primeStage);
            const uint32_t rows =
                waveRows > rowStart ?
                    (waveRows - rowStart < tiling.projectionRows ? waveRows - rowStart : tiling.projectionRows) :
                    0U;
            if (rows != 0U) {
                const uint64_t row = static_cast<uint64_t>(waveBase + primeStage) * tiling.rowsPerWave +
                                     static_cast<uint64_t>(mIndex) * tiling.mTile + rowStart;
                const uint32_t inSlot = (sequence + prime) % tiling.projectionInputSlots;
                PrefetchProjectionRows(tensors, tiling, row, rows,
                                       kStart + static_cast<uint64_t>(primeTile) * tiling.l1K,
                                       Kernel::PlannedL1TileKCount(tiling, kIndex, primeTile),
                                       input[inSlot * tiling.projectionRows * tiling.l1K],
                                       static_cast<event_t>(kProjectionInputEventBase + inSlot));
            }
        }
        for (uint32_t tile = 0U; tile < tileCount; ++tile) {
            const uint32_t kCount = Kernel::PlannedL1TileKCount(tiling, kIndex, tile);
            const uint64_t kOffset = kStart + static_cast<uint64_t>(tile) * tiling.l1K;
            for (uint32_t stage = 0U; stage < stages; ++stage, ++sequence) {
                const uint32_t wave = waveBase + stage;
                const uint32_t waveRows = Kernel::GetWaveRows(tiling, mIndex, wave);
                const uint32_t rows =
                    waveRows > rowStart ?
                        (waveRows - rowStart < tiling.projectionRows ? waveRows - rowStart : tiling.projectionRows) :
                        0U;
                const uint64_t row = static_cast<uint64_t>(wave) * tiling.rowsPerWave +
                                     static_cast<uint64_t>(mIndex) * tiling.mTile + rowStart;
                const uint32_t slot = sequence & 1U;
                const uint32_t inputSlot = sequence % tiling.projectionInputSlots;
                const event_t inEvent = static_cast<event_t>(kProjectionInputEventBase + inputSlot);
                const event_t outEvent =
                    tiling.stageUbSlots == kProjectionStageSlots && slot == 0U ? EVENT_ID6 : EVENT_ID7;
                const auto source = input[inputSlot * tiling.projectionRows * tiling.l1K];
                const auto destination =
                    output[tiling.stageUbSlots == kProjectionStageSlots ? slot * tiling.stageRowStride * tiling.l1K :
                                                                          0U];
                if (rows != 0U) {
                    WaitFlag<HardEvent::MTE2_V>(inEvent);
                }
                const uint32_t nextIndex = tile * stages + stage + tiling.projectionInputSlots - 1U;
                if (nextIndex < tileCount * stages) {
                    const uint32_t nextStage = nextIndex % stages;
                    const uint32_t nextTile = nextIndex / stages;
                    const uint32_t nextWaveRows = Kernel::GetWaveRows(tiling, mIndex, waveBase + nextStage);
                    const uint32_t nextRows =
                        nextWaveRows > rowStart ?
                            (nextWaveRows - rowStart < tiling.projectionRows ? nextWaveRows - rowStart :
                                                                               tiling.projectionRows) :
                            0U;
                    if (nextRows != 0U) {
                        const uint64_t nextRow = static_cast<uint64_t>(waveBase + nextStage) * tiling.rowsPerWave +
                                                 static_cast<uint64_t>(mIndex) * tiling.mTile + rowStart;
                        const uint32_t nextSlot =
                            (sequence + tiling.projectionInputSlots - 1U) % tiling.projectionInputSlots;
                        PrefetchProjectionRows(tensors, tiling, nextRow, nextRows,
                                               kStart + static_cast<uint64_t>(nextTile) * tiling.l1K,
                                               Kernel::PlannedL1TileKCount(tiling, kIndex, nextTile),
                                               input[nextSlot * tiling.projectionRows * tiling.l1K],
                                               static_cast<event_t>(kProjectionInputEventBase + nextSlot));
                    }
                }
                StageTransfer::AcquireSlot();
                if (rows != 0U) {
                    WaitFlag<HardEvent::MTE3_V>(outEvent);
                    Compute::PrepareProjectionTile<T, StageTransfer::kPackedNz>(
                        reinterpret_cast<__ubuf__ T*>(source.GetPhyAddr()), rows, kCount,
                        StageTransfer::kPackedNz ? tiling.stageRowStride : tiling.l1K,
                        reinterpret_cast<__ubuf__ float*>(destination.GetPhyAddr()),
                        reinterpret_cast<__ubuf__ float*>(statistics.GetPhyAddr()) +
                            stage * tiling.projectionStatsStride);
                    SetFlag<HardEvent::V_MTE2>(inEvent);
                    const uint64_t stageOffset =
                        (static_cast<uint64_t>(aicIndex) * kProjectionSlots + slot) * tiling.mTile * tiling.l1K +
                        static_cast<uint64_t>(rowStart) * tiling.l1K;
                    StageTransfer::StoreAivRowAsync(tensors.stage, stageOffset, destination, rows * tiling.l1K, tiling,
                                                    outEvent);
                }
                StageTransfer::PublishReady();
            }
        }
        for (uint32_t stage = 0U; stage < stages; ++stage) {
            const uint32_t wave = waveBase + stage;
            const uint32_t waveRows = Kernel::GetWaveRows(tiling, mIndex, wave);
            const uint32_t rows =
                waveRows > rowStart ?
                    (waveRows - rowStart < tiling.projectionRows ? waveRows - rowStart : tiling.projectionRows) :
                    0U;
            if (rows != 0U) {
                const uint64_t row = static_cast<uint64_t>(wave) * tiling.rowsPerWave +
                                     static_cast<uint64_t>(mIndex) * tiling.mTile + rowStart;
                Kernel::CopyUbToGlobal(tensors.partialRms, static_cast<uint64_t>(kIndex) * tiling.totalRows + row,
                                       statistics[stage * tiling.projectionStatsStride], rows);
            }
        }
    }
    for (uint32_t inSlot = 0U; inSlot < tiling.projectionInputSlots; ++inSlot) {
        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(kProjectionInputEventBase + inSlot));
    }
    StageTransfer::DrainCredits();
    WaitFlag<HardEvent::MTE3_V>(EVENT_ID7);
    if (tiling.stageUbSlots == kProjectionStageSlots) {
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID6);
    }
}

/** Pair K-major Cube projection with Vector preparation, retaining each M stage's K order.
 * Formula: partial[M,Q]=sum_K(float(x2[M,K])*phi[Q,K]), squares[M]=sum_K(float(x2[M,K])^2).
 */
template <typename T, typename StageTransfer, bool USE_HF32, bool RESIDENT_B = false>
__aicore__ inline void RunWeightReuseProjection(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling,
                                                TPipe* pipe)
{
    if ASCEND_IS_AIC {
        RunWeightReuseCube<T, StageTransfer, USE_HF32, RESIDENT_B>(tensors, tiling, pipe);
    }
    if ASCEND_IS_AIV {
        RunWeightReuseVector<T, StageTransfer>(tensors, tiling, pipe);
    }
}

/** Compute partialMm[m,q]=sum_k(float(x2[m,k])*phi[q,k]) and partialRms=sum_k(x2*x2) over each K shard.
 * The wave variant also produces the next Post batch; both share this phase's pipe.
 */
template <typename T, typename StageTransfer, bool USE_HF32, bool WAVE_PRODUCER = false,
          typename Pipeline = NoWavePipeline>
__aicore__ inline void RunProjectionPhase(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling)
{
    TPipe pipe;
    if constexpr (WAVE_PRODUCER) {
        RunProjectionSchedule<T, StageTransfer, USE_HF32, true, Pipeline>(tensors, tiling, &pipe);
    } else {
        if (tiling.projectionMStages > 1U) {
            if constexpr (StageTransfer::kPackedNz) {
                if (tiling.projectionTransport == static_cast<uint32_t>(ProjectionTransport::GmStage)) {
                    if (tiling.projectionBResident != 0U) {
                        RunWeightReuseProjection<T, Transfer::GmStage, USE_HF32, true>(tensors, tiling, &pipe);
                    } else {
                        RunWeightReuseProjection<T, Transfer::GmStage, USE_HF32>(tensors, tiling, &pipe);
                    }
                    return;
                }
            }
            if (tiling.projectionBResident != 0U) {
                RunWeightReuseProjection<T, StageTransfer, USE_HF32, true>(tensors, tiling, &pipe);
            } else {
                RunWeightReuseProjection<T, StageTransfer, USE_HF32>(tensors, tiling, &pipe);
            }
        } else {
            RunProjectionSchedule<T, StageTransfer, USE_HF32>(tensors, tiling, &pipe);
        }
    }
}

/** Project BS-owned rows with ordered compact FP32 partials and the shared A preparation loop. */
template <typename T, typename StageTransfer, bool USE_HF32>
__aicore__ inline void RunBsCompactProjection(Kernel::Tensors<T>& tensors, const MegaMhcTilingData& tiling, TPipe* pipe)
{
    RunProjectionSchedule<T, StageTransfer, USE_HF32, false, NoWavePipeline, true, CompactPartials>(tensors, tiling,
                                                                                                    pipe);
}

} // namespace MegaMhc::Schedule

#endif // OPS_TRANSFORMER_MHC_MEGA_MHC_PROJECTION_H
