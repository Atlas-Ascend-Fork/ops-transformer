/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_BLOCK_EPILOGUE_MASK2IDX_VF_ARCH35_HPP
#define BSA_BLOCK_EPILOGUE_MASK2IDX_VF_ARCH35_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/epilogue/bsa_epilogue_dispatch_policy.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/bsa_matrix_coord.hpp"

#include "kernel_operator.h"

namespace NpuArch::Epilogue::Block::MaskVF {
template <uint32_t IndexStride>
__simd_vf__ inline void CompactMaskTileVF(__ubuf__ uint8_t* mask, __ubuf__ int32_t* ids, __ubuf__ int32_t* counts,
                                          uint16_t rows, uint32_t maskStride, uint32_t firstColumn, uint32_t columns,
                                          uint16_t repeats)
{
    using namespace AscendC::Reg;
    constexpr uint32_t lanes = 64; // One unpacked mask byte per int32 index lane.
    MaskReg all = CreateMask<int32_t, MaskPattern::ALL>();
    RegTensor<int32_t> ones, zeros;
    Duplicate(ones, 1);
    Duplicate(zeros, 0);
    for (uint16_t row = 0; row < rows; ++row) {
        auto* output = ids + row * IndexStride;
        auto* input = mask + row * maskStride;
        uint32_t remaining = columns;
        RegTensor<int32_t> laneCounts, selected, indices, packed, rowCount;
        RegTensor<uint8_t> bytes;
        MaskReg keep;
        UnalignRegForStore cache;
        Duplicate(laneCounts, 0);
        ClearSpr<AscendC::SpecialPurposeReg::AR>();
        for (uint16_t i = 0; i < repeats; ++i) {
            const uint32_t offset = static_cast<uint32_t>(i) * lanes;
            LoadAlign<uint8_t, LoadDist::DIST_UNPACK4_B8>(bytes, input + offset);
            MaskReg valid = UpdateMask<uint32_t>(remaining);
            Compares<uint8_t, AscendC::CMPMODE::NE>(keep, bytes, static_cast<uint8_t>(0), valid);
            Arange(indices, static_cast<int32_t>(firstColumn + offset));
            Squeeze<int32_t, GatherMaskMode::STORE_REG>(packed, indices, keep);
            StoreUnAlign<int32_t, PostLiteral::POST_MODE_UPDATE>(output, packed, cache);
            Select(selected, ones, zeros, keep);
            Add(laneCounts, laneCounts, selected, all);
        }
        StoreUnAlignPost(output, cache);
        ReduceSum(rowCount, laneCounts, all);
        StoreAlign<int32_t, StoreDist::DIST_FIRST_ELEMENT_B32>(counts + row, rowCount, all);
    }
}
template <uint32_t IndexStride>
__aicore__ inline void CompactMaskTileCallVF(const AscendC::LocalTensor<uint8_t>& maskUb,
                                             const AscendC::LocalTensor<int32_t>& idsUb,
                                             const AscendC::LocalTensor<int32_t>& countsUb, uint16_t rows,
                                             uint32_t maskStride, uint32_t firstColumn, uint32_t columns)
{
    auto* mask = reinterpret_cast<__ubuf__ uint8_t*>(maskUb.GetPhyAddr());
    auto* ids = reinterpret_cast<__ubuf__ int32_t*>(idsUb.GetPhyAddr());
    auto* counts = reinterpret_cast<__ubuf__ int32_t*>(countsUb.GetPhyAddr());
    const uint16_t repeats = (columns + 63) / 64;
    CompactMaskTileVF<IndexStride>(mask, ids, counts, rows, maskStride, firstColumn, columns, repeats);
}

template <uint32_t Stride, uint32_t Capacity>
__simd_vf__ inline void AppendRowsVF(__ubuf__ int32_t* ids, __ubuf__ int32_t* counts, __ubuf__ int32_t* cache,
                                     __ubuf__ int32_t* totals, uint16_t rows, bool first)
{
    using namespace AscendC::Reg;
    MaskReg all = CreateMask<int32_t, MaskPattern::ALL>();
    for (uint16_t k = 0; k < rows; ++k) {
        RegTensor<int32_t> prior, count, sum, pos, lane, value;
        LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(count, counts + k);
        if (first) {
            Duplicate(prior, 0);
        } else {
            LoadAlign<int32_t, LoadDist::DIST_BRC_B32>(prior, totals + k);
        }
        Add(sum, prior, count, all);
        StoreAlign<int32_t, StoreDist::DIST_FIRST_ELEMENT_B32>(totals + k, sum, all);
        for (uint16_t group = 0; group < Capacity / 64; ++group) {
            Arange(lane, static_cast<int32_t>(group * 64));
            Add(pos, prior, lane, all);
            MaskReg selected, fits;
            Compare<int32_t, AscendC::CMPMODE::LT>(selected, lane, count, all);
            Compares<int32_t, AscendC::CMPMODE::LT>(fits, pos, static_cast<int32_t>(Capacity), selected);
            LoadAlign(value, ids + k * Stride + group * 64);
            Scatter(cache + k * Capacity, value, reinterpret_cast<RegTensor<uint32_t>&>(pos), fits);
        }
    }
}
} // namespace NpuArch::Epilogue::Block::MaskVF

namespace NpuArch::Epilogue::Block {

template <class ElementSparseMask_, class ElementSparseIdx_, class ElementSparseCount_>
class BlockEpilogue<EpilogueBsaMask2IdxVF, ElementSparseMask_, ElementSparseIdx_, ElementSparseCount_> {
public:
    using DispatchPolicy = EpilogueBsaMask2IdxVF;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementSparseMask = ElementSparseMask_;
    using ElementSparseIdx = ElementSparseIdx_;
    using ElementSparseCount = ElementSparseCount_;

    static constexpr uint32_t IO_STAGES = DispatchPolicy::IO_STAGES;
    static constexpr uint32_t PRE_ROW_TILE = 8;
    static constexpr uint32_t PRE_COL_TILE = 1024;
    static constexpr uint32_t PRE_ELEM_NUM_PER_LOOP = PRE_ROW_TILE * PRE_COL_TILE;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource)
    {
        constexpr uint32_t MASK_PAT_IN_UINT8 = 0;
        // Keep the UB offsets used by the validated mask VF.
        constexpr uint32_t ROW_CACHE = IO_STAGES * PRE_ELEM_NUM_PER_LOOP;
        constexpr uint32_t ROW_TOTALS = 3 * PRE_ELEM_NUM_PER_LOOP;
        constexpr uint32_t TILE_COUNTS = 9 * PRE_ELEM_NUM_PER_LOOP;
        constexpr uint32_t RSVD_SPARSE_IDX = 13 * PRE_ELEM_NUM_PER_LOOP;
        constexpr uint32_t RSVD_SPARSE_COUNT = 21 * PRE_ELEM_NUM_PER_LOOP;

        static_assert(ROW_CACHE + PRE_ROW_TILE * CACHE_IDS * sizeof(int32_t) <= ROW_TOTALS);
        static_assert(ROW_TOTALS + PRE_ROW_TILE * sizeof(int32_t) <= TILE_COUNTS);
        for (uint32_t i = 0; i < IO_STAGES; i++) {
            maskPatUb8[i] = resource.ubBuf.template GetBufferByByte<ElementSparseMask>(
                MASK_PAT_IN_UINT8 + i * PRE_ELEM_NUM_PER_LOOP * sizeof(ElementSparseMask));
            sparseIdxUb[i] = resource.ubBuf.template GetBufferByByte<ElementSparseIdx>(
                RSVD_SPARSE_IDX + i * PRE_ELEM_NUM_PER_LOOP * sizeof(ElementSparseIdx));
            sparseCountUb[i] = resource.ubBuf.template GetBufferByByte<ElementSparseCount>(
                RSVD_SPARSE_COUNT + i * PRE_ROW_TILE * sizeof(ElementSparseCount));
        }
        rowCache = resource.ubBuf.template GetBufferByByte<int32_t>(ROW_CACHE);
        rowTotals = resource.ubBuf.template GetBufferByByte<int32_t>(ROW_TOTALS);
        tileCountsUb = resource.ubBuf.template GetBufferByByte<int32_t>(TILE_COUNTS);
    }

    __aicore__ inline void operator()(AscendC::GlobalTensor<ElementSparseMask> gSparseMask,
                                      AscendC::GlobalTensor<ElementSparseIdx> gSparseIdx,
                                      AscendC::GlobalTensor<ElementSparseCount> gSparseCount,
                                      uint32_t totalRowNumBlockMask, uint32_t yBlockNumAligned,
                                      uint32_t avgRowPerSubCore, uint32_t preActiveSubCoreNum)
    {
        uint32_t subCoreIdx = AscendC::GetBlockIdx();
        uint32_t curSubCoreRowOffset = subCoreIdx * avgRowPerSubCore;
        uint32_t actDealtRow =
            (subCoreIdx == preActiveSubCoreNum - 1) ? (totalRowNumBlockMask - curSubCoreRowOffset) : avgRowPerSubCore;

        if (subCoreIdx < preActiveSubCoreNum) {
            uint32_t rowLoop = CeilDiv(actDealtRow, PRE_ROW_TILE);
            uint32_t colLoop = CeilDiv(yBlockNumAligned, PRE_COL_TILE);
            uint32_t IdxPingPongFlag = 0;
            uint32_t CountPingPongFlag = 0;
            for (uint32_t i = 0; i < rowLoop; i++) {
                uint32_t curLoopRowOffset = i * PRE_ROW_TILE;
                uint32_t globalRowOffset = curSubCoreRowOffset + curLoopRowOffset;
                uint32_t actDealtRowCurLoop = (i == rowLoop - 1) ? (actDealtRow - curLoopRowOffset) : PRE_ROW_TILE;

                if (colLoop > 1 && TryWriteCachedRows(gSparseMask, gSparseIdx, gSparseCount, globalRowOffset,
                                                      actDealtRowCurLoop, yBlockNumAligned, colLoop, IdxPingPongFlag)) {
                    continue;
                }
                uint64_t rsvdCountPerRow[PRE_ROW_TILE] = {0};
                for (uint32_t j = 0; j < colLoop; j++) {
                    CompactColumnTile(gSparseMask, globalRowOffset, actDealtRowCurLoop, yBlockNumAligned, j, colLoop,
                                      IdxPingPongFlag);
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(IdxPingPongFlag);
                    // The selected-index slices occupy disjoint rows in UB.
                    // Publish the whole column tile before issuing its GM writes.
                    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(0);
                    for (uint32_t k = 0; k < actDealtRowCurLoop; k++) {
                        uint64_t count = tileCountsUb.GetValue(IdxPingPongFlag * PRE_ROW_TILE + k);
                        if (count == 0) {
                            continue;
                        }
                        uint64_t row = static_cast<uint64_t>(globalRowOffset) + k;
                        uint64_t indexOffset = row * yBlockNumAligned + rsvdCountPerRow[k];
                        AscendC::DataCopyPad(gSparseIdx[indexOffset], sparseIdxUb[IdxPingPongFlag][k * PRE_COL_TILE],
                                             AscendC::DataCopyExtParams(1, count * sizeof(ElementSparseIdx), 0, 0, 0));
                        rsvdCountPerRow[k] += count;
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(IdxPingPongFlag);
                    // MTE3_V guards the index buffer until this ping-pong slot is reused.
                    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(0);
                    IdxPingPongFlag = 1 - IdxPingPongFlag;
                }
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(CountPingPongFlag);
                for (uint32_t k = 0; k < actDealtRowCurLoop; k++) {
                    sparseCountUb[CountPingPongFlag].SetValue(k, static_cast<int32_t>(rsvdCountPerRow[k]));
                }
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(CountPingPongFlag + 2);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(CountPingPongFlag + 2);
                AscendC::DataCopyPad(
                    gSparseCount[globalRowOffset], sparseCountUb[CountPingPongFlag],
                    AscendC::DataCopyExtParams(1, actDealtRowCurLoop * sizeof(ElementSparseCount), 0, 0, 0));
                AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(CountPingPongFlag);

                CountPingPongFlag = 1 - CountPingPongFlag;
            }
        }
    }

private:
    __aicore__ inline void CompactColumnTile(AscendC::GlobalTensor<ElementSparseMask> gSparseMask,
                                             uint32_t globalRowOffset, uint32_t actDealtRowCurLoop,
                                             uint32_t yBlockNumAligned, uint32_t j, uint32_t colLoop,
                                             uint32_t IdxPingPongFlag)
    {
        uint32_t curLoopColOffset = j * PRE_COL_TILE;
        uint32_t actDealtColCurLoop = (j == colLoop - 1) ? (yBlockNumAligned - curLoopColOffset) : PRE_COL_TILE;
        uint32_t actDealtColCurLoop32 = CeilDiv(actDealtColCurLoop, 32) * 32;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(IdxPingPongFlag);
        uint64_t maskOffset = static_cast<uint64_t>(globalRowOffset) * yBlockNumAligned + curLoopColOffset;
        AscendC::DataCopyPad(
            maskPatUb8[IdxPingPongFlag], gSparseMask[maskOffset],
            AscendC::DataCopyExtParams(actDealtRowCurLoop, actDealtColCurLoop * sizeof(ElementSparseMask),
                                       (yBlockNumAligned - actDealtColCurLoop) * sizeof(ElementSparseMask), 0, 0),
            AscendC::DataCopyPadExtParams<ElementSparseMask>(true, 0, (actDealtColCurLoop32 - actDealtColCurLoop), 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(IdxPingPongFlag);
        MaskVF::CompactMaskTileCallVF<PRE_COL_TILE>(maskPatUb8[IdxPingPongFlag], sparseIdxUb[IdxPingPongFlag],
                                                    tileCountsUb[IdxPingPongFlag * PRE_ROW_TILE], actDealtRowCurLoop,
                                                    actDealtColCurLoop32, curLoopColOffset, actDealtColCurLoop);
    }

    // Returns false before any GM write if a row group exceeds the UB cache.
    __aicore__ inline bool TryWriteCachedRows(AscendC::GlobalTensor<ElementSparseMask> gSparseMask,
                                              AscendC::GlobalTensor<ElementSparseIdx> gSparseIdx,
                                              AscendC::GlobalTensor<ElementSparseCount> gSparseCount,
                                              uint32_t globalRowOffset, uint32_t actDealtRowCurLoop,
                                              uint32_t yBlockNumAligned, uint32_t colLoop, uint32_t& IdxPingPongFlag)
    {
        for (uint32_t j = 0; j < colLoop; j++) {
            CompactColumnTile(gSparseMask, globalRowOffset, actDealtRowCurLoop, yBlockNumAligned, j, colLoop,
                              IdxPingPongFlag);
            AscendC::PipeBarrier<PIPE_V>();
            MaskVF::AppendRowsVF<PRE_COL_TILE, CACHE_IDS>(
                reinterpret_cast<__ubuf__ int32_t*>(sparseIdxUb[IdxPingPongFlag].GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(tileCountsUb[IdxPingPongFlag * PRE_ROW_TILE].GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(rowCache.GetPhyAddr()),
                reinterpret_cast<__ubuf__ int32_t*>(rowTotals.GetPhyAddr()), actDealtRowCurLoop, j == 0);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(IdxPingPongFlag);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(IdxPingPongFlag);
            IdxPingPongFlag = 1 - IdxPingPongFlag;
        }
        AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        uint32_t cachedCounts[PRE_ROW_TILE];
        bool fits = true;
        for (uint32_t k = 0; k < actDealtRowCurLoop; ++k) {
            cachedCounts[k] = static_cast<uint32_t>(rowTotals.GetValue(k));
            fits = fits && cachedCounts[k] <= CACHE_IDS;
        }
        if (fits) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
            for (uint32_t k = 0; k < actDealtRowCurLoop; ++k) {
                if (cachedCounts[k] > 0) {
                    uint64_t offset = (static_cast<uint64_t>(globalRowOffset) + k) * yBlockNumAligned;
                    AscendC::DataCopyPad(
                        gSparseIdx[offset], rowCache[k * CACHE_IDS],
                        AscendC::DataCopyExtParams(1, cachedCounts[k] * sizeof(ElementSparseIdx), 0, 0, 0));
                }
            }
            AscendC::DataCopyPad(
                gSparseCount[globalRowOffset], rowTotals,
                AscendC::DataCopyExtParams(1, actDealtRowCurLoop * sizeof(ElementSparseCount), 0, 0, 0));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            return true;
        }
        return false;
    }

    AscendC::LocalTensor<uint8_t> maskPatUb8[IO_STAGES];
    AscendC::LocalTensor<int32_t> sparseCountUb[IO_STAGES];
    static constexpr uint32_t CACHE_IDS = 128;
    static_assert(CACHE_IDS % 64 == 0 && CACHE_IDS <= PRE_COL_TILE);
    AscendC::LocalTensor<int32_t> rowCache;
    AscendC::LocalTensor<int32_t> rowTotals;
    AscendC::LocalTensor<int32_t> tileCountsUb;
    AscendC::LocalTensor<int32_t> sparseIdxUb[IO_STAGES];
};

} // namespace NpuArch::Epilogue::Block
#endif // BSA_BLOCK_EPILOGUE_MASK2IDX_VF_ARCH35_HPP
