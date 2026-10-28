/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mega_mhc_cube_tiling.h"
#include "mega_mhc_tiling_utils.h"

#include <algorithm>
#include <array>

namespace optiling::mega_mhc {
namespace {

constexpr uint64_t kMAlign = 16;
constexpr uint64_t kPairSize = 2U;
constexpr uint64_t kReadWritePassCount = 2U;
constexpr uint64_t kTilingCostComponentCount = 7U;
constexpr uint64_t kMaximumBsMTile = 256U;
constexpr uint64_t kMinimumWaveActiveCores = 4U;
constexpr uint64_t kMmadLoadKUnitElements = 8U;
constexpr uint64_t kCubeNTile = 32;
constexpr uint64_t kKAlign = 256;
constexpr uint64_t kL1KAlign = 128;
constexpr uint64_t kL0KAlign = 32;
constexpr uint64_t kNzNAlign = 16;
constexpr uint64_t kFp32Bytes = sizeof(float);
constexpr uint64_t kSequentialPartialK = 1024;
constexpr uint64_t kSequentialPartialThreshold = 2048;
constexpr uint64_t kWaveProjectionElements = 8192;
constexpr uint64_t kWaveHandoffElements = kPairSize * kWaveProjectionElements;

/** Return ceil(value/divisor), with zero for an invalid divisor. */
uint64_t CeilDiv(uint64_t value, uint64_t divisor)
{
    return divisor == 0 ? 0 : value / divisor + static_cast<uint64_t>(value % divisor != 0);
}

/** Round a host-selected element count up to the required hardware alignment. */
uint64_t AlignUp(uint64_t value, uint64_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

/** Round a host-selected element count down without crossing its capacity. */
uint64_t AlignDown(uint64_t value, uint64_t alignment)
{
    return value / alignment * alignment;
}

/** Check both ping-pong operands against their dedicated per-core L1 budgets. */
bool FitsL1(uint64_t m, uint64_t q, uint64_t l1K, const CubeTilingParams& request)
{
    // l1ABytes/l1BBytes are each operand's complete two-slot budget.
    return kDoubleBufferSlots * m * l1K * kFp32Bytes <= request.l1ABytes &&
           kDoubleBufferSlots * AlignUp(q, kNzNAlign) * l1K * kFp32Bytes <= request.l1BBytes;
}

/** One legal M/K geometry; no execution-template submode is attached to the choice. */
struct CubeTilingCandidate {
    uint64_t mTile = 0U;
    uint64_t mGrid = 0U;
    uint64_t kGrid = 0U;
    uint64_t rawSplitK = 0U;
    uint64_t l1K = 0U;
    uint64_t waves = 0U;
    uint64_t traffic = 0U;
    uint32_t overlapRows = 0U;
};

/** Select the largest 128-aligned L1 K block within a partial; ordered partials cannot cross tile boundaries. */
uint64_t SelectL1K(uint64_t capacity, uint64_t partialK, bool ordered)
{
    uint64_t l1K = AlignDown(std::min(capacity, partialK), kL1KAlign);
    while (ordered && l1K != 0U && partialK % l1K != 0U) {
        l1K -= kL1KAlign;
    }
    return l1K;
}

/** Enumerate capacity-limited M tiles and M/K grids, including one-M-lane geometries.
 * Score MMAD calls and busiest-core padded M*K work before transfer costs.
 * An incumbent bounds wave count and B/partial traffic while refining an existing overlapped epilogue.
 * The initial search retains splitK + waves*baseK for waves, or partialK for staged execution.
 * B traffic accounts for the legal number of simultaneous L0C accumulators. Partial traffic includes
 * both projection stores and epilogue reads. Wave handoffs use one global element-equivalent estimate.
 */
bool SearchMkTiling(const CubeTilingParams& request, uint64_t active, const CubeTilingCandidate* incumbent,
                    CubeTilingCandidate& choice)
{
    const uint64_t nzN = AlignUp(request.q, kNzNAlign);
    const uint64_t transferBudget = kPairSize * kWaveProjectionElements;
    const uint64_t mCapacity = AlignDown(
        std::min({AlignUp(request.rows, kMAlign), request.l1ABytes / (kDoubleBufferSlots * kL1KAlign * kFp32Bytes),
                  request.l0ABytes / (kDoubleBufferSlots * kL0KAlign * kFp32Bytes),
                  request.l0CBytes / (kDoubleBufferSlots * kCubeNTile * kFp32Bytes),
                  request.wavePipeline ? transferBudget / kL1KAlign : request.rows + kMAlign}),
        kMAlign);
    const uint64_t shardAlign = request.rows >= kPairSize * active ? kL1KAlign : kKAlign;
    std::array<uint64_t, kTilingCostComponentCount> bestCost{};
    bool hasTiling = false;
    for (uint64_t mTile = kMAlign; mTile <= mCapacity; mTile *= kPairSize) {
        const uint64_t mBlocks = CeilDiv(request.rows, mTile);
        for (uint64_t mGrid = 1U; mGrid <= std::min(active, mBlocks); ++mGrid) {
            const uint64_t waves = CeilDiv(mBlocks, mGrid);
            if (incumbent != nullptr && waves > incumbent->waves) {
                continue;
            }
            if (request.wavePipeline &&
                (waves < kPairSize || (request.preferCompleteWaves && request.rows % (kPairSize * active) == 0U &&
                                       request.rows % (mTile * mGrid) != 0U))) {
                continue;
            }
            for (uint64_t grid = 1U; grid <= active / mGrid; ++grid) {
                const uint64_t rawSplitK = AlignUp(CeilDiv(request.k, grid), shardAlign);
                const uint64_t partialK = rawSplitK >= kSequentialPartialThreshold ? kSequentialPartialK : rawSplitK;
                const uint64_t units = CeilDiv(request.k, partialK);
                const uint64_t kGrid = std::min(grid, units);
                if (request.wavePipeline && mTile % (kPairSize * kGrid) != 0U) {
                    continue;
                }
                const uint64_t ownedUnits = CeilDiv(units, kGrid);
                const uint64_t splitK = ownedUnits * partialK;
                const uint64_t largeCount = units % kGrid == 0U ? kGrid : units % kGrid;
                const uint64_t shortK = units / kGrid * partialK;
                const uint64_t largeBeforeLast = std::min(kGrid - 1U, largeCount);
                const uint64_t lastK = request.k - largeBeforeLast * splitK - (kGrid - 1U - largeBeforeLast) * shortK;
                uint64_t l1Capacity =
                    std::min({kSequentialPartialK, request.l1ABytes / (kDoubleBufferSlots * mTile * kFp32Bytes),
                              request.l1BBytes / (kDoubleBufferSlots * nzN * kFp32Bytes)});
                if (request.wavePipeline) {
                    // At least two transfers are needed for the producer/consumer tick handshake.
                    l1Capacity = std::min({l1Capacity, transferBudget / mTile, lastK / kPairSize});
                }
                const uint64_t l1K = SelectL1K(l1Capacity, partialK, true);
                const uint64_t baseK =
                    AlignDown(std::min({l1K, request.l0ABytes / (kDoubleBufferSlots * mTile * kFp32Bytes),
                                        request.l0BBytes / (kDoubleBufferSlots * kCubeNTile * kFp32Bytes)}),
                              kL0KAlign);
                if (l1K == 0U || baseK == 0U || !FitsL1(mTile, request.q, l1K, request)) {
                    continue;
                }
                const uint64_t stages =
                    request.wavePipeline ?
                        1U :
                        std::max<uint64_t>(1U, std::min({uint64_t(request.maxReuseMStages), waves,
                                                         request.l0CBytes / (mTile * kCubeNTile * kFp32Bytes)}));
                const uint64_t work = mTile * waves * splitK;
                const uint64_t traffic = mGrid * CeilDiv(waves, stages) * request.k * nzN * kFp32Bytes +
                                         kReadWritePassCount * request.rows * units * (nzN + 1U) * kFp32Bytes;
                if (incumbent != nullptr && traffic > incumbent->traffic) {
                    continue;
                }
                const uint64_t callsPerPartial = partialK / l1K * CeilDiv(l1K, baseK) + CeilDiv(partialK % l1K, baseK);
                uint32_t overlapRows = 0U;
                const uint64_t producerRows = CeilDiv(mTile, kPairSize * kGrid);
                if (request.wavePipeline && request.d != 0U && request.ubBytes != 0U) {
                    const uint64_t producerBytes = CalcProducerUbSize(request.d, false, true);
                    const auto ubTiling = CalcProjectionUbTiling(l1K, CeilDiv(mTile, kPairSize), request.ubBytes,
                                                                 producerBytes, request.packedNz);
                    const uint64_t concurrentBytes = producerBytes + ubTiling.bytes;
                    if (concurrentBytes < request.ubBytes) {
                        overlapRows = FitReductionRows(producerRows, request.d, units, nzN, kGrid,
                                                       request.ubBytes - concurrentBytes);
                    }
                }
                if (incumbent != nullptr && overlapRows == 0U) {
                    continue;
                }
                const uint64_t phaseCost =
                    !request.wavePipeline ?
                        partialK :
                        (incumbent != nullptr ? producerRows * request.k + waves * kWaveHandoffElements :
                                                splitK + waves * baseK);
                const std::array<uint64_t, kTilingCostComponentCount> cost = {waves * ownedUnits * callsPerPartial,
                                                                              work,
                                                                              phaseCost,
                                                                              traffic,
                                                                              active - mGrid * kGrid,
                                                                              waves,
                                                                              mCapacity - mTile};
                if (!hasTiling || cost < bestCost) {
                    choice = {mTile, mGrid, kGrid, rawSplitK, l1K, waves, traffic, overlapRows};
                    bestCost = cost;
                    hasTiling = true;
                }
            }
        }
    }
    return hasTiling;
}

/** Refine only an already-overlapped wave tiling, without increasing its compute, handoffs or traffic. */
bool SelectMkTiling(const CubeTilingParams& request, uint64_t active, CubeTilingCandidate& choice)
{
    CubeTilingCandidate incumbent;
    if (!SearchMkTiling(request, active, nullptr, incumbent)) {
        return false;
    }
    choice = incumbent;
    if (request.wavePipeline && incumbent.overlapRows != 0U) {
        CubeTilingCandidate refined;
        if (SearchMkTiling(request, active, &incumbent, refined)) {
            choice = refined;
        }
    }
    return true;
}

/** Validate the required dimensions and per-core capacities before any tiling division. */
bool IsValidCubeRequest(const CubeTilingParams& request)
{
    return (request.schedule == ScheduleMode::MK || request.schedule == ScheduleMode::BS) && request.rows != 0U &&
           request.k != 0U && request.q != 0U && request.aicCount != 0U && request.aivCount >= kPairSize &&
           request.l1ABytes != 0U && request.l1BBytes != 0U && request.l0ABytes != 0U && request.l0BBytes != 0U &&
           request.l0CBytes >= kFp32Bytes * kCubeNTile * kCubeNTile;
}

} // namespace

/** Calculate MK ownership from one generic grid search; BS retains pair-local complete-K ownership.
 * A shard below 2048 stays intact; larger shards use ordered 1024-element partials.
 * All L1/L0 loops, tails and physical strides are serialized by the host.
 */
bool CalcCubeTiling(const CubeTilingParams& request, CubeTilingInfo& tiling)
{
    if (!IsValidCubeRequest(request)) {
        return false;
    }

    const uint64_t active = std::min(request.aicCount, request.aivCount / kPairSize);
    if (active == 0 ||
        (request.wavePipeline && (active < kMinimumWaveActiveCores || request.schedule == ScheduleMode::BS))) {
        return false;
    }

    uint64_t selectedMTile = 0U;
    uint64_t selectedMGrid = 0U;
    uint64_t selectedKGrid = 1U;
    uint64_t rawSplitK = 0U;
    uint64_t l1K = 0U;
    if (request.schedule == ScheduleMode::MK) {
        CubeTilingCandidate choice;
        if (!SelectMkTiling(request, active, choice)) {
            return false;
        }
        selectedMTile = choice.mTile;
        selectedMGrid = choice.mGrid;
        selectedKGrid = choice.kGrid;
        rawSplitK = choice.rawSplitK;
        l1K = choice.l1K;
    } else {
        // BS divides tokens between paired AIVs, with a two-row ownership grain.
        const uint64_t mCapacity = AlignDown(
            std::min<uint64_t>({kMaximumBsMTile, request.l0CBytes / (kDoubleBufferSlots * kCubeNTile * kFp32Bytes),
                                request.l1ABytes / (kDoubleBufferSlots * kL1KAlign * kFp32Bytes)}),
            kMAlign);
        if (mCapacity == 0U) {
            return false;
        }
        selectedMTile = std::min(mCapacity, AlignUp(CeilDiv(request.rows, active), kPairSize));
        selectedMGrid = std::min(active, CeilDiv(request.rows, selectedMTile));
        rawSplitK = AlignUp(request.k, kKAlign);
        const uint64_t partialK = rawSplitK >= kSequentialPartialThreshold ? kSequentialPartialK : rawSplitK;
        const uint64_t l1Capacity =
            std::min<uint64_t>({kSequentialPartialK,
                                request.l1ABytes / (kDoubleBufferSlots * AlignUp(selectedMTile, kMAlign) * kFp32Bytes),
                                request.l1BBytes / (kDoubleBufferSlots * AlignUp(request.q, kNzNAlign) * kFp32Bytes)});
        l1K = SelectL1K(l1Capacity, partialK, rawSplitK >= kSequentialPartialThreshold);
    }
    const bool useSequentialPartials = request.schedule == ScheduleMode::MK || rawSplitK >= kSequentialPartialThreshold;
    const uint64_t partialGroupK = rawSplitK >= kSequentialPartialThreshold ? kSequentialPartialK : rawSplitK;
    const uint64_t ownershipUnits = CeilDiv(request.k, partialGroupK);
    const uint64_t splitK = useSequentialPartials ? CeilDiv(ownershipUnits, selectedKGrid) * partialGroupK : rawSplitK;
    const uint64_t remainder = useSequentialPartials ? ownershipUnits % selectedKGrid : 0U;
    const uint64_t largeKCoreCount = remainder == 0U ? selectedKGrid : remainder;
    const uint64_t shortSplitK = useSequentialPartials ? ownershipUnits / selectedKGrid * partialGroupK : splitK;
    const uint64_t selectedActive = selectedMGrid * selectedKGrid;
    const uint64_t rowsPerWave = selectedMGrid * selectedMTile;
    const uint64_t mWaveCount = CeilDiv(request.rows, rowsPerWave);
    if (l1K == 0U || !FitsL1(AlignUp(selectedMTile, kMAlign), request.q, l1K, request)) {
        return false;
    }

    const uint64_t baseM =
        AlignDown(std::min<uint64_t>(AlignUp(selectedMTile, kMAlign),
                                     request.l0CBytes / (kDoubleBufferSlots * kCubeNTile * kFp32Bytes)),
                  kMAlign);
    constexpr uint64_t baseN = kCubeNTile;
    // Both L0C slots hold a complete M tile. Reject a smaller
    // hardware footprint instead of silently introducing an unplanned M loop.
    if (baseM < AlignUp(selectedMTile, kMAlign) || kDoubleBufferSlots * baseM * baseN * kFp32Bytes > request.l0CBytes) {
        return false;
    }
    const uint64_t l0AK = request.l0ABytes / (kDoubleBufferSlots * baseM * kFp32Bytes);
    const uint64_t l0BK = request.l0BBytes / (kDoubleBufferSlots * baseN * kFp32Bytes);
    const uint64_t baseK = AlignDown(std::min({l0AK, l0BK, l1K}), kL0KAlign);
    if (baseK == 0) {
        return false;
    }
    const uint64_t nzM = AlignUp(selectedMTile, kMAlign);
    const uint64_t nzN = AlignUp(request.q, kNzNAlign);
    const uint64_t aL1SlotElements = nzM * l1K;
    const uint64_t bL1SlotElements = nzN * l1K;
    const uint64_t l1BOffsetElements = kDoubleBufferSlots * aL1SlotElements;
    const uint64_t l0ASlotElements = baseM * baseK;
    const uint64_t l0BSlotElements = baseN * baseK;
    const uint64_t l0CSlotElements = baseM * baseN;

    const uint64_t lastWaveRows = request.rows - (mWaveCount - 1) * rowsPerWave;
    const uint64_t lastWaveMGrid = CeilDiv(lastWaveRows, selectedMTile);
    const uint64_t lastMRows = lastWaveRows - (lastWaveMGrid - 1) * selectedMTile;
    const uint64_t largeBeforeLast = std::min(selectedKGrid - 1U, largeKCoreCount);
    const uint64_t lastKElements =
        request.k - largeBeforeLast * splitK - (selectedKGrid - 1U - largeBeforeLast) * shortSplitK;
    const uint64_t l1FullLoops = splitK / l1K;
    const uint64_t l1Tail = splitK % l1K;
    const uint64_t lastCoreL1FullLoops = lastKElements / l1K;
    const uint64_t lastCoreL1Tail = lastKElements % l1K;
    const uint64_t l0FullLoops = l1K / baseK;
    const uint64_t l0Tail = l1K % baseK;
    const uint64_t tailL1K = l1Tail == 0 ? l1K : l1Tail;
    const uint64_t tailL1L0FullLoops = tailL1K / baseK;
    const uint64_t tailL1L0Tail = tailL1K % baseK;
    const uint64_t lastCoreTailL1K = lastCoreL1Tail == 0 ? l1K : lastCoreL1Tail;
    const uint64_t lastCoreTailL1L0FullLoops = lastCoreTailL1K / baseK;
    const uint64_t lastCoreTailL1L0Tail = lastCoreTailL1K % baseK;
    const uint64_t mL0FullLoops = selectedMTile / baseM;
    const uint64_t mL0TailRows = selectedMTile % baseM;
    const uint64_t lastMCoreL0FullLoops = lastMRows / baseM;
    const uint64_t lastMCoreL0TailRows = lastMRows % baseM;
    const uint64_t fullMmadKStep = baseK;
    const uint64_t l0TailMmadKStep = l0Tail;
    const uint64_t tailL1TailMmadKStep = tailL1L0Tail;
    const uint64_t lastCoreTailMmadKStep = lastCoreTailL1L0Tail;
    const uint64_t l0TailMmadFinalKStep = l0Tail;
    const uint64_t tailL1TailMmadFinalKStep = tailL1L0Tail;
    const uint64_t lastCoreTailMmadFinalKStep = lastCoreTailL1L0Tail;

    tiling.activeAic = static_cast<uint32_t>(selectedActive);
    tiling.activeAiv = static_cast<uint32_t>(kPairSize * selectedActive);
    tiling.mGrid = static_cast<uint32_t>(selectedMGrid);
    tiling.kGrid = static_cast<uint32_t>(selectedKGrid);
    tiling.mWaveCount = static_cast<uint32_t>(mWaveCount);
    tiling.rowsPerWave = static_cast<uint32_t>(rowsPerWave);
    tiling.lastWaveMGrid = static_cast<uint32_t>(lastWaveMGrid);
    tiling.lastWaveRows = static_cast<uint32_t>(lastWaveRows);
    tiling.mTile = static_cast<uint32_t>(selectedMTile);
    tiling.splitK = static_cast<uint32_t>(splitK);
    tiling.largeKCoreCount = static_cast<uint32_t>(largeKCoreCount);
    tiling.shortSplitK = static_cast<uint32_t>(shortSplitK);
    tiling.shortKCoreL1FullLoops = static_cast<uint32_t>(shortSplitK / l1K);
    tiling.l1K = static_cast<uint32_t>(l1K);
    tiling.l1KFullLoops = static_cast<uint32_t>(l1FullLoops);
    tiling.l1KTail = static_cast<uint32_t>(l1Tail);
    tiling.lastKCoreL1FullLoops = static_cast<uint32_t>(lastCoreL1FullLoops);
    tiling.lastKCoreL1Tail = static_cast<uint32_t>(lastCoreL1Tail);
    tiling.lastMCoreRows = static_cast<uint32_t>(lastMRows);
    tiling.lastKCoreElements = static_cast<uint32_t>(lastKElements);
    tiling.mL0FullLoops = static_cast<uint32_t>(mL0FullLoops);
    tiling.mL0TailRows = static_cast<uint32_t>(mL0TailRows);
    tiling.lastMCoreL0FullLoops = static_cast<uint32_t>(lastMCoreL0FullLoops);
    tiling.lastMCoreL0TailRows = static_cast<uint32_t>(lastMCoreL0TailRows);
    tiling.loadMStride = static_cast<uint32_t>(nzM / kMAlign);
    tiling.loadNStride = static_cast<uint32_t>(nzN / kMAlign);
    tiling.fullL0LoadKStep = static_cast<uint32_t>(baseK / kMmadLoadKUnitElements);
    tiling.l0TailLoadKStep = static_cast<uint32_t>(l0Tail / kMmadLoadKUnitElements);
    tiling.tailL1TailLoadKStep = static_cast<uint32_t>(tailL1L0Tail / kMmadLoadKUnitElements);
    tiling.lastCoreTailLoadKStep = static_cast<uint32_t>(lastCoreTailL1L0Tail / kMmadLoadKUnitElements);
    tiling.baseM = static_cast<uint32_t>(baseM);
    tiling.baseN = static_cast<uint32_t>(baseN);
    tiling.baseK = static_cast<uint32_t>(baseK);
    tiling.nzM = static_cast<uint32_t>(nzM);
    tiling.nzN = static_cast<uint32_t>(nzN);
    tiling.l1ASlotElements = static_cast<uint32_t>(aL1SlotElements);
    tiling.l1BOffsetElements = static_cast<uint32_t>(l1BOffsetElements);
    tiling.l1BSlotElements = static_cast<uint32_t>(bL1SlotElements);
    tiling.l0ASlotElements = static_cast<uint32_t>(l0ASlotElements);
    tiling.l0BSlotElements = static_cast<uint32_t>(l0BSlotElements);
    tiling.l0CSlotElements = static_cast<uint32_t>(l0CSlotElements);
    // Fixpipe stores the padded Cube N tile. Keep Q=24's valid coefficients in
    // the first columns and reserve the full 32-column destination stride.
    tiling.fixpipeRowStride = static_cast<uint32_t>(nzN);
    tiling.fullMmadKStep = static_cast<uint32_t>(fullMmadKStep);
    tiling.fullMmadKLoopCount = 1U;
    tiling.l0TailMmadKStep = static_cast<uint32_t>(l0TailMmadKStep);
    tiling.l0TailMmadKLoopCount = static_cast<uint32_t>(l0Tail != 0);
    tiling.l0TailMmadFinalKStep = static_cast<uint32_t>(l0Tail == 0 ? 0 : l0TailMmadFinalKStep);
    tiling.tailL1TailMmadKStep = static_cast<uint32_t>(tailL1TailMmadKStep);
    tiling.tailL1TailMmadKLoopCount = static_cast<uint32_t>(tailL1L0Tail != 0);
    tiling.tailL1TailMmadFinalKStep = static_cast<uint32_t>(tailL1L0Tail == 0 ? 0 : tailL1TailMmadFinalKStep);
    tiling.lastCoreTailMmadKStep = static_cast<uint32_t>(lastCoreTailMmadKStep);
    tiling.lastCoreTailMmadKLoopCount = static_cast<uint32_t>(lastCoreTailL1L0Tail != 0);
    tiling.lastCoreTailMmadFinalKStep =
        static_cast<uint32_t>(lastCoreTailL1L0Tail == 0 ? 0 : lastCoreTailMmadFinalKStep);
    tiling.l0KLoops = static_cast<uint32_t>(l0FullLoops);
    tiling.l0KTail = static_cast<uint32_t>(l0Tail);
    tiling.tailL1L0FullLoops = static_cast<uint32_t>(tailL1L0FullLoops);
    tiling.tailL1L0Tail = static_cast<uint32_t>(tailL1L0Tail);
    tiling.lastCoreTailL1L0FullLoops = static_cast<uint32_t>(lastCoreTailL1L0FullLoops);
    tiling.lastCoreTailL1L0Tail = static_cast<uint32_t>(lastCoreTailL1L0Tail);
    tiling.partialGroupK = static_cast<uint32_t>(partialGroupK);
    tiling.partialGroupCount = static_cast<uint32_t>(CeilDiv(request.k, tiling.partialGroupK));
    return true;
}

} // namespace optiling::mega_mhc
