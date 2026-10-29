/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file fia_kernel_nonquant_arch31.h
 * \brief arch31 (KirinX90 / Kirin9030) kernel 主类：FiaKernelNonQuantArch31。
 *  -- 单核单 kernel + AIC/AIV 混合执行
 * 核心循环
 *  for b in [0, B]:
 *    for hIdx in [0, H]:
 *      kvHeadIdx = hIdx / group  // GQA 映射
 *      for s1Tile in [0, S1, M_BASE):
 *        mDeal = min(M_BASE, S1 - s1Tile)
 *        vecService.InitAccumulators(mDeal, D_inner)
 *        for s2Tile in [0, S2, N_BASE):
 *          nDeal = min(N_BASE, S2 - s2Tile)
 *          cubeService.ComputeMm1(...)     // QK^T -> GM mm1ResGm
 *          SetFlag<FIX_S>(FIX_DRAIN)
 *          WaitFlag(FIX_S)(FIX_DRAIN)
 *          SetFlag<MTE3_V>(EVENT_ID0)
 *          WaitFlag(MTE3_V)(EVENT_ID0)
 *          // attenMask 搬运
 *          vecService.ComputeVec1OnlineSoftmax(...)
 *          SetFlag<V_MTE2>(EVENT_ID1)
 *          WaitFlag(V_MTE2)(EVENT_ID1)
 *          cubeService.ComputeMm2(...)     // attn V -> GM mm2ResGm
 *          SetFlag<FIX_S>(FIX_DRAIN)
 *          WaitFlag(FIX_S)(FIX_DRAIN)
 *          SetFlag<MTE3_V>(EVENT_ID2)
 *          WaitFlag(MTE3_V)(EVENT_ID2)
 *          vecService.AccumulateO(...)     // 在线 O 累积
 *        vecService.FinalizeAndCopyOut(...) // 最终归一化 + Cast + 输出
 */
#ifndef FIA_KERNEL_NONQUANT_ARCH31_H
#define FIA_KERNEL_NONQUANT_ARCH31_H

#include "fia_public_define_arch31.h"
#include "fia_block_cube_nonquant_arch31.h"
#include "fia_block_vec_nonquant_arch31.h"
#include "fused_infer_attention_score_tilingkey_arch31.h"

using namespace AttentionCommonArch31;
using AttentionCommon::Min;
using AttentionCommon::Max;
using AttentionCommon::CeilDiv;

namespace AttentionCommonArch31 {

template <typename FIAT, typename CubeBlockType = FiaBlockCubeNonQuantArch31<FIAT>,
          typename VecBlockType = FiaBlockVecNonQuantArch31<FIAT>, uint32_t HP_PATH = ARCH31_HP_PATH_NONE>
class FiaKernelNonQuantArch31 {
public:
    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;

    __aicore__ inline FiaKernelNonQuantArch31() {}

    __aicore__ inline void ParseTilingData(const FusedInferAttentionScoreTilingData* __restrict tilingData)
    {
        // baseParams
        constInfo.scaleValue = tilingData->baseParams.scaleValue;
        constInfo.batchSize = tilingData->baseParams.bSize;
        constInfo.gSize = tilingData->baseParams.gSize;
        constInfo.kvHeadNum = tilingData->baseParams.n2Size;
        constInfo.qHeadNum = constInfo.gSize * constInfo.kvHeadNum;
        constInfo.qSeqSize = tilingData->baseParams.s1Size;
        constInfo.kvSeqSize = tilingData->baseParams.s2Size;
        constInfo.headDim = tilingData->baseParams.headDim;
        constInfo.softmaxLseFlag = (tilingData->baseParams.softmaxLseFlag != 0);
        constInfo.outputLayout = static_cast<FIA_LAYOUT_ARCH31>(tilingData->baseParams.outputLayout);

        // maskParams
        uint32_t maskFlagField = tilingData->maskParams.attenMaskFlag;
        constInfo.attenMaskFlag = ((maskFlagField & 0x1u) != 0);
        constInfo.attenMaskTensorFlag = (((maskFlagField >> 1) & 0x1u) != 0);
        constInfo.attenMaskBatchStride = tilingData->maskParams.attenMaskBatchStride;
        constInfo.sparseMode = tilingData->maskParams.sparseMode;
        constInfo.preTokens = tilingData->maskParams.preToken;
        constInfo.nextTokens = tilingData->maskParams.nextToken;

        constInfo.sparseOffset = static_cast<int32_t>(constInfo.kvSeqSize) - static_cast<int32_t>(constInfo.qSeqSize);
        if (constInfo.sparseMode == 4) {
            int64_t minNext64 = (constInfo.nextTokens < 0) ? constInfo.nextTokens : 0;
            constInfo.bandUpperDelta = constInfo.sparseOffset + static_cast<int32_t>(minNext64);
            constInfo.bandLowerDelta = constInfo.sparseOffset - static_cast<int32_t>(constInfo.preTokens);
        }

        // innerSplitParams
        constInfo.mBaseSize = tilingData->innerSplitParams.mBaseSize;
        constInfo.nBaseSize = tilingData->innerSplitParams.s2BaseSize;
        constInfo.nBaseUb = constInfo.nBaseSize;
        constInfo.dTile = tilingData->innerSplitParams.dTile;
        constInfo.kTileQK = tilingData->innerSplitParams.kTileQK;
        constInfo.nTileV = tilingData->innerSplitParams.nTileV;
        constInfo.mBaseDeal = tilingData->innerSplitParams.mBaseDeal;
        constInfo.needSplitD = (tilingData->innerSplitParams.needSplitD != 0);
        constInfo.dInner = static_cast<uint32_t>(constInfo.headDim);
        constInfo.nDealMax = constInfo.nBaseSize;
        constInfo.kTileV = tilingData->innerSplitParams.kTileV;

        // workspaceParams
        constInfo.mm1ResGmSize = tilingData->workspaceParams.mm1ResSize;
        constInfo.mm2ResGmSize = tilingData->workspaceParams.mm2ResSize;
        constInfo.softmaxPFp16GmSize = constInfo.mm1ResGmSize;

        constInfo.maskMDeal = static_cast<uint32_t>(constInfo.qSeqSize);
    }

    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* attenMask, __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
                                __gm__ uint8_t* tiling, __gm__ int64_t* actualSeqLenQGm,
                                __gm__ int64_t* actualSeqLenKvGm, TPipe* pipe)
    {
        GET_TILING_DATA_WITH_STRUCT(FusedInferAttentionScoreTilingData, tiling_data_in_arch31, tiling);
        const FusedInferAttentionScoreTilingData* __restrict tiling_data_arch31 = &tiling_data_in_arch31;

        ParseTilingData(tiling_data_arch31);

        queryGm.SetGlobalBuffer((__gm__ Q_T*)query);
        ListTensorDesc keyListTensorDesc((__gm__ void*)(key));
        __gm__ uint8_t* keyData = (__gm__ uint8_t*)keyListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
        keyGm.SetGlobalBuffer((__gm__ KV_T*)keyData);
        ListTensorDesc valueListTensorDesc((__gm__ void*)(value));
        __gm__ uint8_t* valueData = (__gm__ uint8_t*)valueListTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
        valueGm.SetGlobalBuffer((__gm__ KV_T*)valueData);
        attentionOutGm.SetGlobalBuffer((__gm__ OUT_T*)attentionOut);
        if (attenMask != nullptr) {
            attenMaskGm.SetGlobalBuffer((__gm__ uint8_t*)attenMask);
        }

        __gm__ uint8_t* user = GetUserWorkspace(workspace);
        uint64_t offset = 0;
        mm1ResGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(user + offset));
        offset += constInfo.mm1ResGmSize * sizeof(half);
        softmaxPFp16Gm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(user + offset));
        offset += constInfo.softmaxPFp16GmSize * sizeof(half);
        mm2ResGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(user + offset));
        offset += constInfo.mm2ResGmSize * sizeof(half);

        // 初始化 Cube / Vec service
        cubeService.Init(queryGm, keyGm, valueGm, mm1ResGm, softmaxPFp16Gm, mm2ResGm, constInfo, pipe);
        vecService.Init(mm1ResGm, softmaxPFp16Gm, mm2ResGm, attentionOutGm, attenMaskGm, attenMask, constInfo, pipe);

        if constexpr (FIAT::layout == FIA_LAYOUT_ARCH31::TND) {
            actualSeqLenQGm_ = actualSeqLenQGm;
            actualSeqLenKvGm_ = actualSeqLenKvGm;
        }
    }

    __aicore__ inline void Process()
    {
        if constexpr (HP_PATH == ARCH31_HP_PATH_PREFILL) {
            ProcessHpOnchip();
        } else if constexpr (HP_PATH == ARCH31_HP_PATH_DECODE) {
            ProcessDecodeHpHeadPack();
        } else {
            ProcessLegacy();
        }
    }

private:
    __aicore__ inline uint32_t HeadDim() const
    {
        return FIAT::headDim != 0U ? FIAT::headDim : static_cast<uint32_t>(constInfo.headDim);
    }
    __aicore__ inline void ProcessDecodeHpHeadPack()
    {
        constexpr uint32_t S1_HP = 1U;
        constexpr uint32_t gPack = 4U;
        uint32_t B = static_cast<uint32_t>(constInfo.batchSize);
        uint32_t H = static_cast<uint32_t>(constInfo.qHeadNum);
        uint32_t KvH = static_cast<uint32_t>(constInfo.kvHeadNum);
        uint32_t S2 = static_cast<uint32_t>(constInfo.kvSeqSize);
        uint32_t D = HeadDim();
        uint32_t N_BASE = constInfo.nBaseSize;
        uint32_t dInner = constInfo.dInner;

        SetFlag<HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
        SetFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        SetFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        SetFlag<HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
        SetFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM1FREE_P0);
        SetFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM2FREE_P0);

        for (uint32_t bIdx = 0; bIdx < B; bIdx++) {
            for (uint32_t kvHeadIdx = 0; kvHeadIdx < KvH; kvHeadIdx++) {
                uint32_t h0 = kvHeadIdx * gPack;
                vecService.InitAccumulators(gPack, dInner);
                for (uint32_t s2Tile = 0; s2Tile < S2; s2Tile += N_BASE) {
                    uint32_t nDeal = Min(N_BASE, S2 - s2Tile);
                    RunInfoArch31 info;
                    info.bIdx = bIdx;
                    info.hIdx = h0;
                    info.kvHeadIdx = kvHeadIdx;
                    info.s1Tile = 0;
                    info.mDeal = gPack;
                    info.nDeal = nDeal;
                    info.dInner = dInner;
                    info.s2Stride = (S2 + 15U) & ~15U;
                    info.isFirstS2Loop = (s2Tile == 0);
                    info.isLastS2Loop = (s2Tile + N_BASE >= S2);
                    info.isGlobalLastTile = (bIdx + 1 == B) && (kvHeadIdx + 1 == KvH) && (s2Tile + N_BASE >= S2);

                    info.qGmOffset = (static_cast<uint64_t>(bIdx) * H + h0) * S1_HP * D;
                    info.mm2ResGmOffset = (static_cast<uint64_t>(bIdx) * H + h0) * S1_HP * D;
                    info.outGmOffset = (static_cast<uint64_t>(bIdx) * H + h0) * S1_HP * D;
                    info.kGmOffset = (static_cast<uint64_t>(bIdx * KvH + kvHeadIdx) * S2 + s2Tile) * D;
                    info.vGmOffset = info.kGmOffset;
                    info.pGmOffset = (static_cast<uint64_t>(bIdx * H + h0) * S1_HP) * info.s2Stride + s2Tile;
                    info.mm1ResGmOffset = info.pGmOffset;
                    if (constInfo.attenMaskTensorFlag) {
                        info.attenMaskGmOffset = static_cast<uint64_t>(bIdx) * constInfo.attenMaskBatchStride + s2Tile;
                    }

                    WaitFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM1FREE_P0);
                    cubeService.ComputeMm1HpCoarse(info, 0U);
                    SetFlag<HardEvent::FIX_S>(ARCH31_EVENT_ID_FIX_DRAIN);
                    WaitFlag<HardEvent::FIX_S>(ARCH31_EVENT_ID_FIX_DRAIN);

                    vecService.CopyAttenMaskGmToUb(info);
                    SetFlag<HardEvent::FIX_MTE1>(ARCH31_EVENT_ID_MM1);
                    WaitFlag<HardEvent::FIX_MTE1>(ARCH31_EVENT_ID_MM1);
                    vecService.CopyMm1ResL1ToUb(info, cubeService.GetQkResL1Tensor(0U));
                    SetFlag<HardEvent::MTE1_V>(ARCH31_EVENT_ID_MM1);
                    WaitFlag<HardEvent::MTE1_V>(ARCH31_EVENT_ID_MM1);
                    SetFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM1FREE_P0);
                    vecService.ComputeVec1OnlineSoftmax(info);
                    SetFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_P);
                    WaitFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_P);
                    vecService.CopySoftmaxPFp16UbToGm(info);
                    SetFlag<HardEvent::MTE3_MTE2>(ARCH31_EVENT_ID_MTE3_DRAIN);
                    WaitFlag<HardEvent::MTE3_MTE2>(ARCH31_EVENT_ID_MTE3_DRAIN);
                    SetFlag<HardEvent::V_MTE2>(ARCH31_EVENT_ID_P);
                    WaitFlag<HardEvent::V_MTE2>(ARCH31_EVENT_ID_P);
                    WaitFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM2FREE_P0);
                    cubeService.ComputeMm2HpCoarse(info, 0U);
                    SetFlag<HardEvent::FIX_S>(ARCH31_EVENT_ID_FIX_DRAIN);
                    WaitFlag<HardEvent::FIX_S>(ARCH31_EVENT_ID_FIX_DRAIN);
                    SetFlag<HardEvent::FIX_MTE1>(ARCH31_EVENT_ID_MM2);
                    WaitFlag<HardEvent::FIX_MTE1>(ARCH31_EVENT_ID_MM2);

                    vecService.CopyMm2ResL1ToUb(info, cubeService.GetPvResL1Tensor(0U));
                    SetFlag<HardEvent::MTE1_V>(ARCH31_EVENT_ID_MM2);
                    WaitFlag<HardEvent::MTE1_V>(ARCH31_EVENT_ID_MM2);
                    SetFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM2FREE_P0);
                    vecService.AccumulateO(info);
                    SetFlag<HardEvent::V_MTE1>(ARCH31_EVENT_ID_MM2);
                    WaitFlag<HardEvent::V_MTE1>(ARCH31_EVENT_ID_MM2);
                }
                RunInfoArch31 info;
                info.bIdx = bIdx;
                info.hIdx = h0;
                info.kvHeadIdx = kvHeadIdx;
                info.s1Tile = 0;
                info.mDeal = gPack;
                info.dInner = dInner;
                info.s2Stride = (S2 + 15U) & ~15U;
                info.outGmOffset = (static_cast<uint64_t>(bIdx) * H + h0) * S1_HP * D;
                vecService.FinalizeAndCopyOut(info);
                SetFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_OUT);
                WaitFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_OUT);
                vecService.CopyOutFp16UbToGm(info);
                SetFlag<HardEvent::MTE3_MTE2>(ARCH31_EVENT_ID_MTE3_DRAIN);
                WaitFlag<HardEvent::MTE3_MTE2>(ARCH31_EVENT_ID_MTE3_DRAIN);
            }
        }
        WaitFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM1FREE_P0);
        WaitFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM2FREE_P0);
        WaitFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        WaitFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        WaitFlag<HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
        WaitFlag<HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
    }

    __aicore__ inline void ProcessHpOnchip()
    {
        uint32_t B = static_cast<uint32_t>(constInfo.batchSize);
        uint32_t H = static_cast<uint32_t>(constInfo.qHeadNum);
        uint32_t KvH = static_cast<uint32_t>(constInfo.kvHeadNum);
        uint32_t group = H / KvH;
        uint32_t S1 = static_cast<uint32_t>(constInfo.qSeqSize);
        uint32_t S2 = static_cast<uint32_t>(constInfo.kvSeqSize);
        uint32_t D = HeadDim();
        uint32_t M_BASE = constInfo.mBaseSize;
        uint32_t N_BASE = constInfo.nBaseSize;
        uint32_t dInner = constInfo.dInner;
        uint64_t mm1Slot = constInfo.mm1ResGmSize >> 1;
        uint64_t mm2Slot = constInfo.mm2ResGmSize >> 1;

        SetFlag<HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
        SetFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        SetFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        SetFlag<HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);

        for (uint32_t bIdx = 0; bIdx < B; bIdx++) {
            for (uint32_t hIdx = 0; hIdx < H; hIdx++) {
                uint32_t kvHeadIdx = hIdx / group;
                for (uint32_t s1Tile = 0; s1Tile < S1; s1Tile += M_BASE) {
                    uint32_t mDeal = Min(M_BASE, S1 - s1Tile);
                    vecService.InitAccumulators(mDeal, dInner);

                    RunInfoArch31 info;
                    info.bIdx = bIdx;
                    info.hIdx = hIdx;
                    info.kvHeadIdx = kvHeadIdx;
                    info.s1Tile = s1Tile;
                    info.mDeal = mDeal;
                    info.dInner = dInner;
                    info.s2Stride = (S2 + 15U) & ~15U;
                    info.qGmOffset = (static_cast<uint64_t>(bIdx) * H + hIdx) * S1 * D + s1Tile * D;
                    info.mm2ResGmOffset = (static_cast<uint64_t>(bIdx) * H + hIdx) * S1 * D + s1Tile * D;
                    info.outGmOffset = (static_cast<uint64_t>(bIdx) * H + hIdx) * S1 * D + s1Tile * D;

                    const uint32_t s2Tiles = (S2 + N_BASE - 1U) / N_BASE;
                    const uint64_t rowBase =
                        (static_cast<uint64_t>(bIdx * H + hIdx) * S1) * info.s2Stride + s1Tile * info.s2Stride;
                    for (uint32_t tile = 0; tile < s2Tiles; tile++) {
                        uint32_t pp = tile & 1U;
                        uint32_t s2Tile = tile * N_BASE;
                        uint32_t nDeal = Min(N_BASE, S2 - s2Tile);
                        bool isLast = (tile + 1U == s2Tiles);

                        RunInfoArch31 cv = info;
                        cv.s2Tile = s2Tile;
                        cv.nDeal = nDeal;
                        cv.isFirstS2Loop = (tile == 0);
                        cv.isLastS2Loop = isLast;
                        cv.isGlobalLastTile = (bIdx + 1 == B) && (hIdx + 1 == H) && (s1Tile + M_BASE >= S1) && isLast;
                        cv.kGmOffset = (static_cast<uint64_t>(bIdx * KvH + kvHeadIdx) * S2 + s2Tile) * D;
                        cv.vGmOffset = cv.kGmOffset;
                        cv.mm1ResGmOffset = rowBase + s2Tile + static_cast<uint64_t>(pp) * mm1Slot;
                        if (constInfo.attenMaskFlag || constInfo.attenMaskTensorFlag) {
                            cv.attenMaskGmOffset = static_cast<uint64_t>(bIdx) * constInfo.attenMaskBatchStride +
                                                   static_cast<uint64_t>(s1Tile) * S2 + s2Tile;
                        }

                        if (tile >= 2U) {
                            WaitFlag<HardEvent::MTE1_FIX>(pp ? ARCH31_EVENT_ID_MM1FREE_P1 : ARCH31_EVENT_ID_MM1FREE_P0);
                        }
                        cubeService.ComputeMm1HpCoarse(cv, pp);
                        SetFlag<HardEvent::FIX_MTE1>(pp ? ARCH31_EVENT_ID_MM1READY_P1 : ARCH31_EVENT_ID_MM1READY_P0);
                        if (tile == 0U && !(bIdx == 0 && hIdx == 0 && s1Tile == 0)) {
                            WaitFlag<HardEvent::MTE3_MTE1>(ARCH31_EVENT_ID_MTE3_DRAIN);
                        }
                        vecService.CopyAttenMaskGmToUb(cv);
                        WaitFlag<HardEvent::FIX_MTE1>(pp ? ARCH31_EVENT_ID_MM1READY_P1 : ARCH31_EVENT_ID_MM1READY_P0);
                        if (tile >= 1U) {
                            WaitFlag<HardEvent::V_MTE1>(ARCH31_EVENT_ID_MM1);
                        }
                        vecService.CopyMm1ResL1ToUb(cv, cubeService.GetQkResL1Tensor(pp));
                        SetFlag<HardEvent::MTE1_V>(ARCH31_EVENT_ID_MM1);
                        WaitFlag<HardEvent::MTE1_V>(ARCH31_EVENT_ID_MM1);
                        SetFlag<HardEvent::MTE1_FIX>(pp ? ARCH31_EVENT_ID_MM1FREE_P1 : ARCH31_EVENT_ID_MM1FREE_P0);
                        vecService.ComputeVec1OnlineSoftmax(cv);
                        SetFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_P);
                        WaitFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_P);
                        if (tile >= 2U) {
                            WaitFlag<HardEvent::MTE1_MTE3>(pp ? ARCH31_EVENT_ID_PL1FREE_P1 :
                                                                ARCH31_EVENT_ID_PL1FREE_P0);
                        }
                        vecService.CopyPUbToL1Nd2Nz(cv, cubeService.GetPL1Tensor(pp));
                        SetFlag<HardEvent::MTE3_MTE1>(ARCH31_EVENT_ID_PL1RDY);

                        if (tile >= 2U) {
                            WaitFlag<HardEvent::MTE1_FIX>(pp ? ARCH31_EVENT_ID_MM2FREE_P1 : ARCH31_EVENT_ID_MM2FREE_P0);
                        }
                        cv.mm2ResGmOffset = (static_cast<uint64_t>(bIdx) * H + hIdx) * S1 * D + s1Tile * D +
                                            static_cast<uint64_t>(pp) * mm2Slot;
                        WaitFlag<HardEvent::MTE3_MTE1>(ARCH31_EVENT_ID_PL1RDY);
                        cubeService.ComputeMm2OnchipHpCoarse(cv, pp);
                        SetFlag<HardEvent::FIX_MTE1>(pp ? ARCH31_EVENT_ID_MM2READY_P1 : ARCH31_EVENT_ID_MM2READY_P0);

                        WaitFlag<HardEvent::FIX_MTE1>(pp ? ARCH31_EVENT_ID_MM2READY_P1 : ARCH31_EVENT_ID_MM2READY_P0);
                        vecService.CopyMm2ResL1ToUb(cv, cubeService.GetPvResL1Tensor(pp));
                        SetFlag<HardEvent::MTE1_V>(ARCH31_EVENT_ID_MM2);
                        WaitFlag<HardEvent::MTE1_V>(ARCH31_EVENT_ID_MM2);
                        SetFlag<HardEvent::MTE1_FIX>(pp ? ARCH31_EVENT_ID_MM2FREE_P1 : ARCH31_EVENT_ID_MM2FREE_P0);
                        vecService.AccumulateO(cv);
                        SetFlag<HardEvent::V_MTE1>(ARCH31_EVENT_ID_MM1);
                    }

                    vecService.FinalizeAndCopyOut(info);
                    SetFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_OUT);
                    WaitFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_OUT);
                    vecService.CopyOutFp16UbToGm(info);
                    SetFlag<HardEvent::MTE3_MTE2>(ARCH31_EVENT_ID_MTE3_DRAIN);
                    WaitFlag<HardEvent::MTE3_MTE2>(ARCH31_EVENT_ID_MTE3_DRAIN);
                    SetFlag<HardEvent::MTE3_MTE1>(ARCH31_EVENT_ID_MTE3_DRAIN);
                    if (s2Tiles > 0U) {
                        WaitFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM1FREE_P0);
                        WaitFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM2FREE_P0);
                        WaitFlag<HardEvent::MTE1_MTE3>(ARCH31_EVENT_ID_PL1FREE_P0);
                        WaitFlag<HardEvent::V_MTE1>(ARCH31_EVENT_ID_MM1);
                    }
                    if (s2Tiles > 1U) {
                        WaitFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM1FREE_P1);
                        WaitFlag<HardEvent::MTE1_FIX>(ARCH31_EVENT_ID_MM2FREE_P1);
                        WaitFlag<HardEvent::MTE1_MTE3>(ARCH31_EVENT_ID_PL1FREE_P1);
                    }
                }
            }
        }
        WaitFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        WaitFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        WaitFlag<HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
        WaitFlag<HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
        WaitFlag<HardEvent::MTE3_MTE1>(ARCH31_EVENT_ID_MTE3_DRAIN);
    }

public:
    __aicore__ inline void ProcessLegacy()
    {
        uint32_t B = static_cast<uint32_t>(constInfo.batchSize);
        uint32_t H = static_cast<uint32_t>(constInfo.qHeadNum);
        uint32_t KvH = static_cast<uint32_t>(constInfo.kvHeadNum);
        uint32_t group = H / KvH;
        uint32_t S1 = static_cast<uint32_t>(constInfo.qSeqSize);
        uint32_t S2 = static_cast<uint32_t>(constInfo.kvSeqSize);
        uint32_t D = static_cast<uint32_t>(constInfo.headDim);
        uint32_t M_BASE = constInfo.mBaseSize;
        uint32_t N_BASE = constInfo.nBaseSize;
        uint32_t dInner = constInfo.dInner;

        constexpr bool isTndLayout = (FIAT::layout == FIA_LAYOUT_ARCH31::TND);
        uint32_t seqQEnd[ARCH31_TND_MAX_BATCH + 1];
        uint32_t seqKVEnd[ARCH31_TND_MAX_BATCH + 1];
        uint32_t prefixSq[ARCH31_TND_MAX_BATCH + 1];
        uint32_t prefixMM1[ARCH31_TND_MAX_BATCH + 1];
        if constexpr (isTndLayout) {
            if (actualSeqLenQGm_ == nullptr || actualSeqLenKvGm_ == nullptr) {
                return;
            }
            seqQEnd[0] = 0;
            seqKVEnd[0] = 0;
            prefixSq[0] = 0;
            prefixMM1[0] = 0;
            for (uint32_t i = 0; i < B; i++) {
                uint32_t s1i = static_cast<uint32_t>(actualSeqLenQGm_[i]) - seqQEnd[i];
                uint32_t s2i = static_cast<uint32_t>(actualSeqLenKvGm_[i]) - seqKVEnd[i];
                seqQEnd[i + 1] = static_cast<uint32_t>(actualSeqLenQGm_[i]);
                seqKVEnd[i + 1] = static_cast<uint32_t>(actualSeqLenKvGm_[i]);
                prefixSq[i + 1] = prefixSq[i] + s1i * s1i;
                prefixMM1[i + 1] = prefixMM1[i] + s1i * ((s2i + 15U) & ~15U);
            }
        }

        SetFlag<HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
        SetFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        SetFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        SetFlag<HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);

        for (uint32_t bIdx = 0; bIdx < B; bIdx++) {
            if constexpr (isTndLayout) {
                S1 = seqQEnd[bIdx + 1] - seqQEnd[bIdx];
                S2 = seqKVEnd[bIdx + 1] - seqKVEnd[bIdx];
            }
            for (uint32_t hIdx = 0; hIdx < H; hIdx++) {
                uint32_t kvHeadIdx = hIdx / group;
                for (uint32_t s1Tile = 0; s1Tile < S1; s1Tile += M_BASE) {
                    uint32_t mDeal = Min(M_BASE, S1 - s1Tile);
                    vecService.InitAccumulators(mDeal, dInner);

                    RunInfoArch31 info;
                    info.bIdx = bIdx;
                    info.hIdx = hIdx;
                    info.kvHeadIdx = kvHeadIdx;
                    info.s1Tile = s1Tile;
                    info.mDeal = mDeal;
                    info.dInner = dInner;
                    info.s2Stride = (S2 + 15U) & ~15U;
                    if constexpr (isTndLayout) {
                        info.maskRowWidth = S1;
                    }
                    constexpr bool isBnsdLayout = (FIAT::layout == FIA_LAYOUT_ARCH31::BNSD);
                    if constexpr (isTndLayout) {
                        info.qGmOffset = (static_cast<uint64_t>(seqQEnd[bIdx]) + s1Tile) * H * D + hIdx * D;
                        info.mm2ResGmOffset = static_cast<uint64_t>(seqQEnd[bIdx]) * H * D +
                                              static_cast<uint64_t>(hIdx) * S1 * D + s1Tile * D;
                        info.outGmOffset = (static_cast<uint64_t>(seqQEnd[bIdx]) + s1Tile) * H * D + hIdx * D;
                    } else if (isBnsdLayout) {
                        // s1Tile 维度的 GM 偏移在整个 S2 循环内不变
                        info.qGmOffset = (bIdx * H + hIdx) * S1 * D + s1Tile * D;
                        info.mm2ResGmOffset = (bIdx * H + hIdx) * S1 * D + s1Tile * D;
                        info.outGmOffset = (bIdx * H + hIdx) * S1 * D + s1Tile * D;
                    } else {
                        info.qGmOffset = (static_cast<uint64_t>(bIdx) * S1 + s1Tile) * H * D + hIdx * D;
                        info.mm2ResGmOffset = (bIdx * H + hIdx) * S1 * D + s1Tile * D;
                        info.outGmOffset = (static_cast<uint64_t>(bIdx) * S1 + s1Tile) * H * D + hIdx * D;
                    }

                    for (uint32_t s2Tile = 0; s2Tile < S2; s2Tile += N_BASE) {
                        uint32_t nDeal = Min(N_BASE, S2 - s2Tile);
                        info.s2Tile = s2Tile;
                        info.nDeal = nDeal;
                        info.isFirstS2Loop = (s2Tile == 0);
                        info.isLastS2Loop = (s2Tile + N_BASE >= S2);
                        info.isGlobalLastTile =
                            (bIdx + 1 == B) && (hIdx + 1 == H) && (s1Tile + M_BASE >= S1) && info.isLastS2Loop;

                        if (isBnsdLayout) {
                            info.kGmOffset = ((bIdx * KvH + kvHeadIdx) * S2 + s2Tile) * D;
                            info.vGmOffset = ((bIdx * KvH + kvHeadIdx) * S2 + s2Tile) * D;
                        } else if constexpr (isTndLayout) {
                            info.kGmOffset = (static_cast<uint64_t>(seqKVEnd[bIdx]) + s2Tile) * KvH * D + kvHeadIdx * D;
                            info.vGmOffset = (static_cast<uint64_t>(seqKVEnd[bIdx]) + s2Tile) * KvH * D + kvHeadIdx * D;
                        } else {
                            info.kGmOffset = (static_cast<uint64_t>(bIdx) * S2 + s2Tile) * KvH * D + kvHeadIdx * D;
                            info.vGmOffset = (static_cast<uint64_t>(bIdx) * S2 + s2Tile) * KvH * D + kvHeadIdx * D;
                        }
                        if constexpr (isTndLayout) {
                            info.pGmOffset = static_cast<uint64_t>(H) * prefixMM1[bIdx] +
                                             static_cast<uint64_t>(hIdx) * S1 * info.s2Stride + s1Tile * info.s2Stride +
                                             s2Tile;
                            info.mm1ResGmOffset = static_cast<uint64_t>(H) * prefixMM1[bIdx] +
                                                  static_cast<uint64_t>(hIdx) * S1 * info.s2Stride +
                                                  s1Tile * info.s2Stride + s2Tile;
                        } else {
                            info.pGmOffset = (bIdx * H + hIdx) * S1 * info.s2Stride + s1Tile * info.s2Stride + s2Tile;
                            info.mm1ResGmOffset =
                                (bIdx * H + hIdx) * S1 * info.s2Stride + s1Tile * info.s2Stride + s2Tile;
                        }
                        if (constInfo.attenMaskFlag) {
                            info.attenMaskGmOffset = bIdx * S1 * S2 + s1Tile * S2 + s2Tile;
                        } else if (constInfo.attenMaskTensorFlag) {
                            if (constInfo.sparseMode == 9) {
                                if constexpr (isTndLayout) {
                                    info.attenMaskGmOffset = static_cast<uint64_t>(prefixSq[bIdx]) +
                                                             static_cast<uint64_t>(s1Tile) * S1 + s2Tile;
                                } else {
                                    info.attenMaskGmOffset =
                                        static_cast<uint64_t>(bIdx) * constInfo.attenMaskBatchStride +
                                        static_cast<uint64_t>(s1Tile) * S1 + s2Tile;
                                }
                            } else {
                                info.attenMaskGmOffset = static_cast<uint64_t>(bIdx) * constInfo.attenMaskBatchStride +
                                                         static_cast<uint64_t>(s1Tile) * S2 + s2Tile;
                            }
                        }

                        // Cube Stage 1: QK^T -> GM mm1ResGM
                        cubeService.ComputeMm1(info);
                        SetFlag<HardEvent::FIX_S>(ARCH31_EVENT_ID_FIX_DRAIN);
                        WaitFlag<HardEvent::FIX_S>(ARCH31_EVENT_ID_FIX_DRAIN);
                        // Cube 写 GM 完 -> 通知 Vector 可读 mm1Res
                        SetFlag<HardEvent::MTE3_V>(ARCH31_EVENT_ID_MM1);

                        vecService.CopyAttenMaskGmToUb(info);

                        // Vector Stage 1: online softmax
                        // Vector 读 GM 前，等 Cube 写完 mm1ResGm
                        WaitFlag<HardEvent::MTE3_V>(ARCH31_EVENT_ID_MM1);
                        vecService.CopyMm1ResGmToUb(info);
                        SetFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_MM1);
                        WaitFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_MM1);
                        // 在线 softmax + scale + mask + P Cast
                        vecService.ComputeVec1OnlineSoftmax(info);
                        SetFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_P);
                        WaitFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_P);
                        vecService.CopySoftmaxPFp16UbToGm(info);
                        SetFlag<HardEvent::MTE3_MTE2>(ARCH31_EVENT_ID_MTE3_DRAIN);
                        WaitFlag<HardEvent::MTE3_MTE2>(ARCH31_EVENT_ID_MTE3_DRAIN);
                        // Vector 写 GM 完 通知Cube 可读
                        SetFlag<HardEvent::V_MTE2>(ARCH31_EVENT_ID_P);
                        // Cube Stage 2: attn V
                        WaitFlag<HardEvent::V_MTE2>(ARCH31_EVENT_ID_P);
                        cubeService.ComputeMm2(info);
                        SetFlag<HardEvent::FIX_S>(ARCH31_EVENT_ID_FIX_DRAIN);
                        WaitFlag<HardEvent::FIX_S>(ARCH31_EVENT_ID_FIX_DRAIN);
                        SetFlag<HardEvent::MTE3_V>(ARCH31_EVENT_ID_MM2);
                        WaitFlag<HardEvent::MTE3_V>(ARCH31_EVENT_ID_MM2);

                        // Vector Stage 2: 在线 O 累积
                        vecService.CopyMm2ResGmToUb(info);
                        SetFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_MM2);
                        WaitFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_MM2);
                        vecService.AccumulateO(info);
                        SetFlag<HardEvent::V_MTE2>(ARCH31_EVENT_ID_MM2);
                        WaitFlag<HardEvent::V_MTE2>(ARCH31_EVENT_ID_MM2);
                    }
                    // S2 循环结束：最终归一化 + Cast
                    vecService.FinalizeAndCopyOut(info);
                    SetFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_OUT);
                    WaitFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_OUT);
                    vecService.CopyOutFp16UbToGm(info);
                    SetFlag<HardEvent::MTE3_MTE2>(ARCH31_EVENT_ID_MTE3_DRAIN);
                    WaitFlag<HardEvent::MTE3_MTE2>(ARCH31_EVENT_ID_MTE3_DRAIN);
                }
            }
        }
        WaitFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        WaitFlag<HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        WaitFlag<HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
        WaitFlag<HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
    }

protected:
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<uint8_t> attenMaskGm;

    GlobalTensor<half> mm1ResGm;
    GlobalTensor<half> softmaxPFp16Gm;
    GlobalTensor<half> mm2ResGm;

    CubeBlockType cubeService;
    VecBlockType vecService;

    __gm__ int64_t* actualSeqLenQGm_ = nullptr;
    __gm__ int64_t* actualSeqLenKvGm_ = nullptr;

    ConstInfoArch31 constInfo{};
};

} // namespace AttentionCommonArch31

#endif // FIA_KERNEL_NONQUANT_ARCH31_H
