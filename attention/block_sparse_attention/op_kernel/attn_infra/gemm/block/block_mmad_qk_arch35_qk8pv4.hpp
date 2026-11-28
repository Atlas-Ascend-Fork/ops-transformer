/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GEMM_BLOCK_QK_ARCH35_QK8PV4_HPP
#define GEMM_BLOCK_QK_ARCH35_QK8PV4_HPP

#include "../../../attn_infra/bsa_base_defs.hpp"
#include "../../../attn_infra/arch/bsa_resource.hpp"
#include "../../../attn_infra/arch/bsa_cross_core_sync.hpp"
#include "../../../attn_infra/bsa_coord.hpp"
#include "../../../attn_infra/gemm/bsa_gemm_dispatch_policy.hpp"
#include "../../../attn_infra/gemm/bsa_helper.hpp"
#include "../../../attn_infra/bsa_gemm_coord.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_gemm_tile_copy.hpp"
#include "../../../attn_infra/gemm/tile_common/bsa_tile_mmad.hpp"
#include "block_mmad_arch35_utils.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0_mx_fp8_a5.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_l1_to_l0_mx_a5.hpp"
#include "../../../attn_infra/gemm/tile_common/copy_gm_to_l1_mx_scale_dn2nz_a5.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "../../../tla/tensor_bsa.hpp"

namespace NpuArch::Gemm::Block {

template <class L1TileShape_, class L0TileShape_, class ElementA_, class ElementB_, class ElementC_, class ElementBias_,
          class TileCopy_, class TileMmad_>
struct BlockMmadQKQkMxfp8PvMxfp4 {
public:
    using DispatchPolicy = MmadAtlasA5BsaQKMxfp8<true>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using TileCopy = TileCopy_;
    using ElementA = ElementA_;
    using ElementB = ElementB_;
    using ElementC = ElementC_;

    using TileMmad = TileMmad_;

    using CopyL1ToL0A = typename TileCopy::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy::CopyL1ToL0B;

    using LayoutTagL1A = typename TileCopy::LayoutTagL1A;
    using LayoutTagL1B = typename TileCopy::LayoutTagL1B;
    using LayoutTagL0A = typename TileCopy::LayoutTagL0A;
    using LayoutTagL0B = typename TileCopy::LayoutTagL0B;
    using ElementAccumulator = typename TileCopy::ElementAccumulator;

    static constexpr uint32_t L0_TILE_M = tla::get<0>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_N = tla::get<1>(L0TileShape_{});
    static constexpr uint32_t L0_TILE_K = tla::get<2>(L0TileShape_{});

    // Q(M) 方向统一 pad 到 128，满足 Fixpipe nSize 32B 倍数约束
    static constexpr uint32_t Q_M_PAD = 128;

    static constexpr AscendC::FixpipeConfig CFG_ROW_MAJOR_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

    // QK initializes and drains the shared L1 K/V events once.
    static constexpr uint32_t Q_EVENT0 = QK_MXFP8_PV_MXFP4::Q_EVENT0;
    static constexpr uint32_t Q_EVENT1 = Q_EVENT0 + 1;
    static constexpr uint32_t QK_L0AB_EVENT0 = 2;
    static constexpr uint32_t QK_L0AB_EVENT1 = 3;
    static constexpr uint32_t QK_L0C_EVENT0 = 2;
    static constexpr uint32_t QK_L0C_EVENT1 = 3;

    __aicore__ inline BlockMmadQKQkMxfp8PvMxfp4(Arch::Resource<ArchTag>& resource, uint32_t& kvBufId,
                                                uint64_t softmaxScale)
        : kBufId(kvBufId)
    {
        softmaxScale_ = softmaxScale;
        for (uint32_t i = 0; i < QK_MXFP8_PV_MXFP4::L1_Q_BUF_CNT; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(QK_MXFP8_PV_MXFP4::L1_Q_BUF_OFFSET +
                                                                             QK_MXFP8_PV_MXFP4::L1_Q_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < QK_MXFP8_PV_MXFP4::L1_Q_BUF_CNT; i++) {
            l1BScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(
                QK_MXFP8_PV_MXFP4::L1_Q_DESCALE_BUF_OFFSET + QK_MXFP8_PV_MXFP4::L1_Q_DESCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < QK_MXFP8_PV_MXFP4::L1_KV_BUF_CNT; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(QK_MXFP8_PV_MXFP4::L1_KV_BUF_OFFSET +
                                                                             QK_MXFP8_PV_MXFP4::L1_KV_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < QK_MXFP8_PV_MXFP4::L1_KV_BUF_CNT; i++) {
            l1AScaleTensor[i] = resource.l1Buf.template GetBufferByByte<uint8_t>(
                QK_MXFP8_PV_MXFP4::L1_KV_DESCALE_BUF_OFFSET + QK_MXFP8_PV_MXFP4::L1_KV_DESCALE_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < QK_MXFP8_PV_MXFP4::L0A_QK_BUF_CNT; i++) {
            l0ATensor[i] = resource.l0ABuf.template GetBufferByByte<ElementB>(QK_MXFP8_PV_MXFP4::L0A_QK_BUF_OFFSET +
                                                                              QK_MXFP8_PV_MXFP4::L0A_QK_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < QK_MXFP8_PV_MXFP4::L0B_QK_BUF_CNT; i++) {
            l0BTensor[i] = resource.l0BBuf.template GetBufferByByte<ElementA>(QK_MXFP8_PV_MXFP4::L0B_QK_BUF_OFFSET +
                                                                              QK_MXFP8_PV_MXFP4::L0B_QK_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < QK_MXFP8_PV_MXFP4::L0C_QK_BUF_CNT; i++) {
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<float>(QK_MXFP8_PV_MXFP4::L0C_QK_BUF_OFFSET +
                                                                           QK_MXFP8_PV_MXFP4::L0C_QK_BUF_SIZE * i);
        }

        AllocEventID();
    }

    __aicore__ inline void AllocEventID()
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        for (uint32_t slot = 0; slot < QK_MXFP8_PV_MXFP4::L1_KV_BUF_CNT; ++slot) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(QK_MXFP8_PV_MXFP4::KV_EVENT0 + slot);
        }
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    __aicore__ inline void FreeEventID()
    {
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(Q_EVENT1);
        for (uint32_t slot = 0; slot < QK_MXFP8_PV_MXFP4::L1_KV_BUF_CNT; ++slot) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(QK_MXFP8_PV_MXFP4::KV_EVENT0 + slot);
        }
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT1);
    }

    __aicore__ inline ~BlockMmadQKQkMxfp8PvMxfp4() {}

    __aicore__ inline uint32_t CeilDivision(uint32_t numerator, uint32_t denominator)
    {
        return (numerator + denominator - 1) / denominator;
    }

    __aicore__ inline uint32_t MinU32(uint32_t a, uint32_t b)
    {
        return (a < b) ? a : b;
    }

    // Q 稠密 GM → L1（RowMajor → zN）
    template <class TensorQ, class TensorL1Q>
    __aicore__ inline void CopyQGmToL1(uint32_t qsActBaseTile, uint32_t embed, TensorQ gQTensorTla,
                                       TensorL1Q l1QTensorTla)
    {
        auto l1QTileTla = GetTile(l1QTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, embed));
        // BSND: GetTile 确保 dValue = embed 而非 qHeadMul * embed
        auto gQTileTla = GetTile(gQTensorTla, tla::MakeCoord(0, 0), tla::MakeShape(qsActBaseTile, embed));

        using CopyGmToL1Q = Tile::TileCopyTla<ArchTag, decltype(gQTileTla), decltype(l1QTensorTla)>;
        CopyGmToL1Q copyGmToL1Q;
        copyGmToL1Q(l1QTileTla, gQTileTla);
    }

    // Q-scale 稠密 GM → L1（Host 沿 D Dn2Nz）
    __aicore__ inline void CopyQScaleGmToL1(uint32_t qsActBaseTile, uint32_t scaleK,
                                            AscendC::GlobalTensor<uint8_t>& gQDequantScale, uint32_t qBufId,
                                            uint32_t qHeadMul)
    {
        copyGmToL1MxScaleDn2Nz(l1BScaleTensor[qBufId], gQDequantScale, qsActBaseTile, scaleK, 0, 0, qHeadMul);
    }

    template <class TensorK, class TensorL1K>
    __aicore__ inline void CopyFullKSegment(TensorK gKTensorTla, TensorL1K l1KTensorTla,
                                            AscendC::GlobalTensor<uint8_t> gKScale, uint32_t srcStart,
                                            uint32_t dstStart, uint32_t rows, uint32_t embed, uint32_t scaleK,
                                            uint32_t kBufId, uint32_t kvHeadMul, uint32_t parentRows)
    {
        using Copy = Tile::TileCopyTla<ArchTag, decltype(gKTensorTla), decltype(l1KTensorTla)>;
        Copy copy;
        auto src = GetTile(gKTensorTla, tla::MakeCoord(srcStart, 0), tla::MakeShape(rows, embed));
        auto dst = GetTile(l1KTensorTla, tla::MakeCoord(dstStart, 0), tla::MakeShape(rows, embed));
        copy(dst, src);
        const uint32_t half = scaleK / 2;
        copyGmToL1MxScaleDn2Nz(l1AScaleTensor[kBufId], gKScale, rows, scaleK, srcStart * kvHeadMul * half,
                               dstStart * half, kvHeadMul);
    }

    // K 数据 + K-scale 融合稀疏 gather：按 gSparseBlockIdx 逐段搬 GM → L1。
    // 两者的分段边界必然一致，故只遍历一遍 index，每段发 K(Nd2Nz) / K-scale(Dn2Nz) 两条 copy。
    // ori 连续递增即代表 GM 上物理相邻，向前偷看合并成单条 copy。
    // 段推进天然落在 y-block 边界上（仅首段 innerStart 可能非 0），故循环内无除法/取模。
    // firstYBlockIdx / firstOriYBlockIdx 由调用方提前预取，避免首个 index 的阻塞访存落在此处。
    template <class TensorK, class TensorL1K, class TileInfoT>
    __aicore__ inline void SparseKFusedBaseTileL1FullLoad(
        TensorK gKTensorTla, TensorL1K l1KTensorTla, AscendC::GlobalTensor<uint8_t> gKScale,
        AscendC::GlobalTensor<int32_t> gSparseBlockIdx, uint32_t gatheredKvSTileIdx, int64_t kvSeqlen,
        uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t yBlockNumAval, uint32_t yBlockNumRsvd,
        uint32_t kvsActBaseTile, uint32_t embed, uint32_t kBufId, uint32_t kvHeadMul, uint32_t firstYBlockIdx,
        uint32_t firstOriYBlockIdx, TileInfoT& tileInfo)
    {
        const bool savePlan = blockShapeY == 128 && kvSBaseTile == 256;
        tileInfo.kvGatherCount = 0;
        using CopyGmToL1K = Tile::TileCopyTla<ArchTag, decltype(gKTensorTla), decltype(l1KTensorTla)>;
        CopyGmToL1K copyGmToL1K;

        uint32_t scaleK = CeilDivision(embed, QK_MXFP8_PV_MXFP4::FP4_C0_ELEMS) * 2;
        uint32_t scaleRowHalf = scaleK / 2;

        const uint32_t kvSeqlenU = static_cast<uint32_t>(kvSeqlen);
        uint32_t gatheredYBlockIdx = firstYBlockIdx;
        uint32_t yBlockInnerStart = gatheredKvSTileIdx * kvSBaseTile - firstYBlockIdx * blockShapeY;
        uint32_t oriYBlockIdx = firstOriYBlockIdx;
        uint32_t dealtLenAccum = 0;

        // Full 128+128 tiles need only one additional index read and no gather scan.
        // Tail/other block shapes retain the original generic path.
        if (savePlan && kvsActBaseTile == 256 && firstYBlockIdx + 1 < yBlockNumRsvd &&
            firstOriYBlockIdx < (kvSeqlenU >> 7)) {
            const uint32_t second = static_cast<uint32_t>(gSparseBlockIdx.GetValue(firstYBlockIdx + 1));
            if (second < (kvSeqlenU >> 7)) {
                const uint32_t src0 = firstOriYBlockIdx << 7;
                tileInfo.kvGatherStarts[0] = src0;
                if (second == firstOriYBlockIdx + 1) {
                    tileInfo.kvGatherCount = 1;
                    tileInfo.kvGatherLengths[0] = 256;
                    CopyFullKSegment(gKTensorTla, l1KTensorTla, gKScale, src0, 0, 256, embed, scaleK, kBufId, kvHeadMul,
                                     kvSBaseTile);
                } else {
                    const uint32_t src1 = second << 7;
                    tileInfo.kvGatherCount = 2;
                    tileInfo.kvGatherStarts[1] = src1;
                    tileInfo.kvGatherLengths[0] = 128;
                    tileInfo.kvGatherLengths[1] = 128;
                    CopyFullKSegment(gKTensorTla, l1KTensorTla, gKScale, src0, 0, 128, embed, scaleK, kBufId, kvHeadMul,
                                     kvSBaseTile);
                    CopyFullKSegment(gKTensorTla, l1KTensorTla, gKScale, src1, 128, 128, embed, scaleK, kBufId,
                                     kvHeadMul, kvSBaseTile);
                }
                return;
            }
        }

        while (dealtLenAccum < kvsActBaseTile && gatheredYBlockIdx < yBlockNumRsvd && oriYBlockIdx < yBlockNumAval) {
            uint32_t oriStartOffset = oriYBlockIdx * blockShapeY + yBlockInnerStart;
            if (oriStartOffset >= kvSeqlenU) {
                break;
            }

            uint32_t curYBlockSize =
                (oriYBlockIdx == yBlockNumAval - 1) ? (kvSeqlenU - oriYBlockIdx * blockShapeY) : blockShapeY;
            if (curYBlockSize <= yBlockInnerStart) {
                break;
            }
            uint32_t availLen = curYBlockSize - yBlockInnerStart;
            // 尾部不满块之后 gather 流即终止（与原实现的 curDealtLen==0 退出等价）
            bool runEndsShort = (curYBlockSize < blockShapeY);

            // 向前偷看：ori 连续则 GM 地址连续，拼进同一条 copy
            uint32_t peekYBlockIdx = gatheredYBlockIdx + 1;
            uint32_t peekOriIdx = oriYBlockIdx + 1;
            while (!runEndsShort && dealtLenAccum + availLen < kvsActBaseTile && peekYBlockIdx < yBlockNumRsvd &&
                   peekOriIdx < yBlockNumAval &&
                   static_cast<uint32_t>(gSparseBlockIdx.GetValue(peekYBlockIdx)) == peekOriIdx) {
                uint32_t peekBlockSize =
                    (peekOriIdx == yBlockNumAval - 1) ? (kvSeqlenU - peekOriIdx * blockShapeY) : blockShapeY;
                availLen += peekBlockSize;
                runEndsShort = (peekBlockSize < blockShapeY);
                ++peekYBlockIdx;
                ++peekOriIdx;
            }

            uint32_t curDealtLen = MinU32(availLen, kvsActBaseTile - dealtLenAccum);
            if (savePlan) {
                const uint32_t segment = tileInfo.kvGatherCount++;
                tileInfo.kvGatherStarts[segment] = oriStartOffset;
                tileInfo.kvGatherLengths[segment] = curDealtLen;
            }

            auto gKTile = GetTile(gKTensorTla, tla::MakeCoord(oriStartOffset, 0), tla::MakeShape(curDealtLen, embed));
            auto l1KTile = GetTile(l1KTensorTla, tla::MakeCoord(dealtLenAccum, 0), tla::MakeShape(curDealtLen, embed));
            copyGmToL1K(l1KTile, gKTile);

            uint32_t srcB16Off = oriStartOffset * kvHeadMul * scaleRowHalf;
            uint32_t dstB16Off = dealtLenAccum * scaleRowHalf;
            copyGmToL1MxScaleDn2Nz(l1AScaleTensor[kBufId], gKScale, curDealtLen, scaleK, srcB16Off, dstB16Off,
                                   kvHeadMul);

            dealtLenAccum += curDealtLen;
            if (runEndsShort) {
                break;
            }
            // 未退出循环时必有 curDealtLen == availLen，偷看过的块已全部消费
            gatheredYBlockIdx = peekYBlockIdx;
            yBlockInnerStart = 0;
            if (dealtLenAccum < kvsActBaseTile) {
                oriYBlockIdx = gSparseBlockIdx.GetValue(gatheredYBlockIdx);
            }
        }
    }

    // Q → L0B（非转置，带 mx scale）
    __aicore__ inline void LoadQToL0B(uint32_t qsActBaseTileAlign16, uint32_t embed, uint32_t qsActBaseTileAlign16L0,
                                      uint32_t scaleMAlign16, uint32_t scaleK, uint32_t qBufId, uint32_t qkL0abBufId)
    {
        copyL1ToL0BMxQk(l0BTensor[qBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                        l1BTensor[qBufId].template ReinterpretCast<fp8_e4m3fn_t>(),
                        l1BScaleTensor[qBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), qsActBaseTileAlign16,
                        embed, qsActBaseTileAlign16L0, scaleMAlign16, 0);
    }

    // K → L0A（非转置，带 mx scale，n 方向子切分）
    __aicore__ inline void LoadKToL0A(uint32_t nSubRowStart, uint32_t nCur, uint32_t embed,
                                      uint32_t kvsActBaseTileAlign16, uint32_t scaleK, uint32_t kBufId,
                                      uint32_t qkL0abBufId)
    {
        copyL1ToL0AMxQk(l0ATensor[qkL0abBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                        l1ATensor[kBufId].template ReinterpretCast<fp8_e4m3fn_t>(),
                        l1AScaleTensor[kBufId].template ReinterpretCast<AscendC::fp8_e8m0_t>(), nSubRowStart, nCur,
                        embed, kvsActBaseTileAlign16, 0, nSubRowStart);
    }

    // C = K · Qᵀ，每 nSub 独立计算，initC 恒 true
    __aicore__ inline void MatmulQK(uint32_t n, uint32_t qsActBaseTileAlign16, uint32_t embed, uint32_t qkL0abBufId,
                                    uint32_t qkL0cBufId)
    {
        auto l0ALayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0A>(n, embed);
        auto l0ATensorTla = tla::MakeTensor(l0ATensor[qkL0abBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                                            l0ALayout, Arch::PositionL0A{});
        auto l0BLayout = tla::MakeLayout<AscendC::mx_fp8_e4m3_t, LayoutTagL0B>(embed, qsActBaseTileAlign16);
        auto l0BTensorTla = tla::MakeTensor(l0BTensor[qBufId].template ReinterpretCast<AscendC::mx_fp8_e4m3_t>(),
                                            l0BLayout, Arch::PositionL0B{});
        auto l0CLayout = tla::MakeLayoutL0C(n, qsActBaseTileAlign16);
        auto l0CTensorTla = tla::MakeTensor(l0CTensor[qkL0cBufId], l0CLayout, Arch::PositionL0C{});
        tileMmad(l0CTensorTla, l0ATensorTla, l0BTensorTla, n, qsActBaseTileAlign16, embed, true);
    }

    // L0C → UB fixpipe（NO_SPLIT 模式，subBlockId 选目标 AIV）
    template <class TensorUB>
    __aicore__ inline void FixpipeMm1(uint32_t qsActBaseTileAlign16, uint32_t n, TensorUB& ubSTensorTla,
                                      bool subBlockId, uint32_t qkL0cBufId)
    {
        auto l0CTensorTla =
            tla::MakeTensor(l0CTensor[qkL0cBufId], tla::MakeLayoutL0C(n, qsActBaseTileAlign16), Arch::PositionL0C{});

        using CopyL0CToUB =
            Tile::CopyL0CToUBTla<ArchTag, decltype(l0CTensorTla), TensorUB, Tile::CopyL0CToUBMode::NO_SPLIT,
                                 Tile::ScaleGranularity::PER_TENSOR, false>;
        CopyL0CToUB copyL0CToUB;
        copyL0CToUB(ubSTensorTla, l0CTensorTla, softmaxScale_, subBlockId);
    }

    // 发一次 nSub 的 fixpipe（含 M_FIX / FIX_M 的成对同步）。
    //   拆出来是为了能延后一拍发射：调用点把它插在 mmad 之后、下一轮的 WaitFlag<MTE1_M> 之前，
    //   用它的发射时间掩盖跨流水 flag 的往返延迟。
    template <class TensorUB>
    __aicore__ inline void EmitFixpipeMm1(uint32_t l0cBufId, uint32_t nSubRowStart, uint32_t nCur,
                                          uint32_t qsActBaseTileAlign16, TensorUB& ubSTensorTla, uint32_t subBlockIdx)
    {
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(QK_L0C_EVENT0 + l0cBufId);
        // nCur 偶数对齐用于 Fixpipe mSize
        uint32_t nCurEven = (nCur + 1) >> 1 << 1;
        auto ubSSubTla =
            GetTile(ubSTensorTla, tla::MakeCoord(nSubRowStart, 0), tla::MakeShape(nCurEven, qsActBaseTileAlign16));
        FixpipeMm1(qsActBaseTileAlign16, nCur, ubSSubTla, subBlockIdx != 0, l0cBufId);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(QK_L0C_EVENT0 + l0cBufId);
    }

    template <bool FullKvTile, class TensorUB>
    __aicore__ inline void ComputeQKTile(uint32_t kvsActBaseTile, uint32_t kRowsAlign16, uint32_t qRowsAlign16,
                                         uint32_t embed, uint32_t scaleK, TensorUB& ubSTensorTla, uint32_t subBlockIdx)
    {
        // n 方向按 L0_TILE_N 子切分。
        // [软流水] fixpipe 延后一拍发射：原实现里 SetFlag/WaitFlag 成对紧挨着（M 等 MTE1、
        //   FIX 等 M），两次跨流水往返中间没有任何工作可掩盖，实测每个 nSub 白等约 92 cycle。
        //   改成：mmad(k) 发完只 Set 不 Wait，把 fixpipe(k-1) 插到 LoadK(k) 与 mmad(k) 之间，
        //   于是 MTE1_M 的往返被 fixpipe 的发射盖住、M_FIX 的往返被 LoadK 盖住。
        //   L0C 双缓冲天然够用：c_k 的 FIX_M 由 k+1 轮释放，k+2 轮才复用。
        //   注意不能把 pending 的 fixpipe 带出本次调用——kernel 紧接着会 CrossCoreSetFlag
        //   通知向量核 S 已就绪，所以循环结束必须 flush。
        constexpr uint32_t fullTileRows = QK_MXFP8_PV_MXFP4::S2_BASE_TILE_SIZE;
        const uint32_t nLoopNum = FullKvTile ? (fullTileRows / L0_TILE_N) : CeilDivision(kvsActBaseTile, L0_TILE_N);
        bool fixPending = false;
        uint32_t pendL0cBufId = 0;
        uint32_t pendNSubRowStart = 0;
        uint32_t pendNCur = 0;

        for (uint32_t nSub = 0; nSub < nLoopNum; ++nSub) {
            uint32_t l0ABEventId = QK_L0AB_EVENT0 + qkL0abBufId;
            uint32_t l0CEventId = QK_L0C_EVENT0 + qkL0cBufId;
            uint32_t nSubRowStart = nSub * L0_TILE_N;
            const uint32_t nCur = FullKvTile ? L0_TILE_N : MinU32(L0_TILE_N, kvsActBaseTile - nSubRowStart);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CEventId);

            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
            // Q 已在 task 首 tile 常驻两个 L0B 槽，此处只需装 K
            LoadKToL0A(nSubRowStart, nCur, embed, kRowsAlign16, scaleK, kBufId, qkL0abBufId);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);

            // 上一轮的 fixpipe 插在这里，掩盖刚发出的 MTE1_M 往返
            if (fixPending) {
                EmitFixpipeMm1(pendL0cBufId, pendNSubRowStart, pendNCur, qRowsAlign16, ubSTensorTla, subBlockIdx);
                fixPending = false;
            }

            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABEventId);
            MatmulQK(nCur, qRowsAlign16, embed, qkL0abBufId, qkL0cBufId);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABEventId);
            qkL0abBufId = (qkL0abBufId + 1) % QK_MXFP8_PV_MXFP4::L0A_QK_BUF_CNT;

            // 只 Set 不 Wait，本轮的 fixpipe 推迟到下一轮（或循环外的 flush）
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CEventId);
            pendL0cBufId = qkL0cBufId;
            pendNSubRowStart = nSubRowStart;
            pendNCur = nCur;
            fixPending = true;
            qkL0cBufId = (qkL0cBufId + 1) % QK_MXFP8_PV_MXFP4::L0C_QK_BUF_CNT;
        }
        if (fixPending) {
            EmitFixpipeMm1(pendL0cBufId, pendNSubRowStart, pendNCur, qRowsAlign16, ubSTensorTla, subBlockIdx);
        }
    }

    template <class TensorQ, class TensorK, class TensorUB, class TaskInfoT, class TileInfoT>
    __aicore__ inline void operator()(TensorQ& gQTensorTla, TensorK& gKTensorTla,
                                      AscendC::GlobalTensor<uint8_t> gQDequantScale,
                                      AscendC::GlobalTensor<uint8_t> gKDequantScale,
                                      AscendC::GlobalTensor<int32_t> gSparseBlockIdx, TensorUB& ubSTensorTla,
                                      TaskInfoT& curTaskInfo, TileInfoT& curTileInfo, uint32_t embed,
                                      uint32_t kvSBaseTile, uint32_t blockShapeY, uint32_t subBlockIdx,
                                      uint32_t kvHeadMul, uint32_t qHeadMul)
    {
        uint32_t qsActBaseTile = curTaskInfo.qsActBaseTile;
        int64_t kvSeqlen = curTaskInfo.kvSeqlen;
        uint32_t yBlockNumAval = curTaskInfo.yBlockNumAval;
        uint32_t yBlockNumRsvd = curTaskInfo.yBlockNumRsvd;
        uint32_t qsActBaseTileAlign16 = curTaskInfo.qsActBaseTileAlign16;
        uint32_t qsActBaseTileAlign128 = curTaskInfo.qsActBaseTileAlign128;

        uint32_t kvsActBaseTile = curTileInfo.kvsActBaseTile;
        uint32_t gatheredKvSTileIdx = curTileInfo.pvGatheredKvSTileIdx;
        bool isFirstKvsTile = curTileInfo.isFirstKvsTile;
        bool isLastKvsTile = curTileInfo.isLastKvsTile;
        uint32_t kvsActBaseTileAlign16 = curTileInfo.kvsActBaseTileAlign16;

        uint32_t scaleK = CeilDivision(embed, QK_MXFP8_PV_MXFP4::FP4_C0_ELEMS) * 2;
        uint32_t l1KEventId = QK_MXFP8_PV_MXFP4::KV_EVENT0 + kBufId;
        uint32_t l1QEventId = Q_EVENT0 + qBufId;

        // 完整保留行的 gather 序号等于原始序号，不必再读 sparse index。
        // 稀疏行仍提前读首个 index，把阻塞访存藏进后面的 Q 搬运。
        const bool denseRow = curTaskInfo.isDenseRow != 0 && blockShapeY == 128 && kvSBaseTile == 256;
        uint32_t firstYBlockIdx = 0;
        uint32_t firstOriYBlockIdx = 0;
        if (!denseRow) {
            firstYBlockIdx = (gatheredKvSTileIdx * kvSBaseTile) / blockShapeY;
            firstOriYBlockIdx = static_cast<uint32_t>(gSparseBlockIdx.GetValue(firstYBlockIdx));
        }

        // Q/QScale 仅在 task 首 KV tile 搬 GM → L1，后续 tile 复用
        if (isFirstKvsTile) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId);
            auto l1QLayout = tla::MakeLayout<ElementA, layout::zN>(qsActBaseTileAlign16, embed);
            auto l1QTensorTla = tla::MakeTensor(l1BTensor[qBufId], l1QLayout, Arch::PositionL1{});
            CopyQGmToL1(qsActBaseTile, embed, gQTensorTla, l1QTensorTla);
            CopyQScaleGmToL1(qsActBaseTile, scaleK, gQDequantScale, qBufId, qHeadMul);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1QEventId);

            // Load the selected task Q into its L0B slot. A single input
            // gate serializes reuse of K; Q stays resident for this task.
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
            LoadQToL0B(qsActBaseTileAlign16, embed, qsActBaseTileAlign16, qsActBaseTileAlign16, scaleK, qBufId,
                       qkL0abBufId);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(QK_L0AB_EVENT1);
        }

        // 每 tile 稀疏 gather K / K-scale 到 L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
        auto l1KLayout = tla::MakeLayout<ElementA, layout::zN>(kvsActBaseTile, embed);
        auto l1KTensorTla = tla::MakeTensor(l1ATensor[kBufId], l1KLayout, Arch::PositionL1{});
        if (denseRow) {
            const uint32_t srcStart = gatheredKvSTileIdx * kvSBaseTile;
            CopyFullKSegment(gKTensorTla, l1KTensorTla, gKDequantScale, srcStart, 0, kvsActBaseTile, embed, scaleK,
                             kBufId, kvHeadMul, kvsActBaseTile);
        } else {
            SparseKFusedBaseTileL1FullLoad(gKTensorTla, l1KTensorTla, gKDequantScale, gSparseBlockIdx,
                                           gatheredKvSTileIdx, kvSeqlen, kvSBaseTile, blockShapeY, yBlockNumAval,
                                           yBlockNumRsvd, kvsActBaseTile, embed, kBufId, kvHeadMul, firstYBlockIdx,
                                           firstOriYBlockIdx, curTileInfo);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KEventId);

        if (kvsActBaseTile == QK_MXFP8_PV_MXFP4::S2_BASE_TILE_SIZE) {
            ComputeQKTile<true>(QK_MXFP8_PV_MXFP4::S2_BASE_TILE_SIZE, QK_MXFP8_PV_MXFP4::S2_BASE_TILE_SIZE,
                                qsActBaseTileAlign16, embed, scaleK, ubSTensorTla, subBlockIdx);
        } else {
            ComputeQKTile<false>(kvsActBaseTile, kvsActBaseTileAlign16, qsActBaseTileAlign16, embed, scaleK,
                                 ubSTensorTla, subBlockIdx);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KEventId);
        kBufId = (kBufId + 1) % QK_MXFP8_PV_MXFP4::L1_KV_BUF_CNT;

        if (unlikely(isLastKvsTile)) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1QEventId);
            qBufId = (qBufId + 1) % QK_MXFP8_PV_MXFP4::L1_Q_BUF_CNT;
        }
    }

protected:
    AscendC::LocalTensor<ElementB> l1ATensor[QK_MXFP8_PV_MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<ElementA> l1BTensor[QK_MXFP8_PV_MXFP4::L1_Q_BUF_CNT];
    AscendC::LocalTensor<ElementB> l0ATensor[QK_MXFP8_PV_MXFP4::L0A_QK_BUF_CNT];
    AscendC::LocalTensor<ElementA> l0BTensor[QK_MXFP8_PV_MXFP4::L0B_QK_BUF_CNT];
    AscendC::LocalTensor<float> l0CTensor[QK_MXFP8_PV_MXFP4::L0C_QK_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1AScaleTensor[QK_MXFP8_PV_MXFP4::L1_KV_BUF_CNT];
    AscendC::LocalTensor<uint8_t> l1BScaleTensor[QK_MXFP8_PV_MXFP4::L1_Q_BUF_CNT];

    Tile::CopyL1ToL0AMxFp8QKA5 copyL1ToL0AMxQk;
    Tile::CopyL1ToL0BMxFp8QKA5 copyL1ToL0BMxQk;
    TileMmad tileMmad;
    Tile::CopyGmToL1MxScaleDn2NzA5 copyGmToL1MxScaleDn2Nz;

    uint32_t qBufId = 0;
    uint32_t& kBufId;
    uint32_t qkL0abBufId = 0;
    uint32_t qkL0cBufId = 0;
    uint64_t softmaxScale_;
};

} // namespace NpuArch::Gemm::Block

#endif // GEMM_BLOCK_QK_ARCH35_QK8PV4_HPP
