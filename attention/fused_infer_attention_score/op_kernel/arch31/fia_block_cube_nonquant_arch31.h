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
 * \file fia_block_cube_nonquant_arch31.h
 * \brief arch31 (KirinX90 / Kirin9030) Cube Stage1/2 实现：Mmad + FixPipe + L1/L0 ping-pong（单核精简版）
 */
#ifndef FIA_BLOCK_CUBE_NONQUANT_ARCH31_H
#define FIA_BLOCK_CUBE_NONQUANT_ARCH31_H

#include "fia_public_define_arch31.h"

using namespace AttentionCommonArch31;
using AttentionCommon::Min;
using AttentionCommon::Max;
using AttentionCommon::CeilDiv;

namespace AttentionCommonArch31 {

template <typename FIAT>
class FiaBlockCubeNonQuantArch31 {
public:
    using T = half;
    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;
    using MM_OUT_T = half;

    __aicore__ inline FiaBlockCubeNonQuantArch31() {}

    __aicore__ inline uint32_t HeadDim() const
    {
        return FIAT::headDim != 0U ? FIAT::headDim : static_cast<uint32_t>(constInfo.headDim);
    }

    __aicore__ inline void Init(GlobalTensor<Q_T> queryGmIn, GlobalTensor<KV_T> keyGmIn, GlobalTensor<KV_T> valueGmIn,
                                GlobalTensor<MM_OUT_T> mm1ResGmIn, GlobalTensor<half> softmaxPFp16GmIn,
                                GlobalTensor<MM_OUT_T> mm2ResGmIn, ConstInfoArch31 constInfoIn, TPipe* pipe)
    {
        queryGm = queryGmIn;
        keyGm = keyGmIn;
        valueGm = valueGmIn;
        mm1ResGm = mm1ResGmIn;
        softmaxPFp16Gm = softmaxPFp16GmIn;
        mm2ResGm = mm2ResGmIn;
        constInfo = constInfoIn;
        InitBuffers(pipe);
    }

    __aicore__ inline void InitBuffers(TPipe* pipe)
    {
        uint32_t mBaseDeal = constInfo.mBaseDeal;
        uint32_t kTileQK = constInfo.kTileQK;
        uint32_t nTileV = constInfo.nTileV;
        uint32_t nDealMax = constInfo.nDealMax;
        uint32_t kTileV = constInfo.kTileV;
        uint32_t D = HeadDim();
        if (mBaseDeal == 0) {
            mBaseDeal = constInfo.mBaseSize;
        }
        if (kTileQK == 0) {
            kTileQK = Max(D / 2, ARCH31_CUBE_K);
        }
        if (nTileV == 0) {
            nTileV = D;
        }
        if (nDealMax == 0) {
            nDealMax = constInfo.nBaseSize;
        }
        if (kTileV == 0) {
            kTileV = nDealMax;
        }

        uint64_t mFragRows = static_cast<uint64_t>(CeilDiv(mBaseDeal, 16U)) * 16U;
        uint64_t dFragCols = static_cast<uint64_t>(CeilDiv(D, 16U)) * 16U;
        uint64_t pFragCols = static_cast<uint64_t>(CeilDiv(nDealMax, 16U)) * 16U;
        uint64_t qpFragCols = Max(dFragCols, pFragCols);
        pipe->InitBuffer(qpBufL1, 2 * mFragRows * qpFragCols * sizeof(Q_T));
        uint64_t nFragRows = static_cast<uint64_t>(CeilDiv(nDealMax, 16U)) * 16U;
        pipe->InitBuffer(kvBufL1, 2 * nFragRows * dFragCols * sizeof(KV_T));

        constexpr bool isHp = FIAT::isHp;
        if (isHp) {
            uint64_t l0aHp = static_cast<uint64_t>(mBaseDeal) * Max(D, nDealMax) * sizeof(Q_T);
            uint64_t l0bHp = static_cast<uint64_t>(Max(nDealMax * D, nDealMax * nTileV)) * sizeof(KV_T);
            uint64_t l0cHp = static_cast<uint64_t>(mBaseDeal) * Max(nDealMax, nTileV) * sizeof(float) * 2ULL;
            pipe->InitBuffer(tmpBufL0A, l0aHp);
            pipe->InitBuffer(tmpBufL0B, l0bHp);
            pipe->InitBuffer(tmpBufL0C, l0cHp);
        } else {
            uint64_t l0aSingle = static_cast<uint64_t>(ARCH31_CUBE_M) * Max(kTileQK, kTileV) * sizeof(Q_T);
            uint64_t l0bSingle = static_cast<uint64_t>(Max(nDealMax * kTileQK, kTileV * nTileV)) * sizeof(KV_T);
            uint64_t l0cSingle = static_cast<uint64_t>(ARCH31_CUBE_M) * Max(nDealMax, nTileV) * sizeof(half);
            pipe->InitBuffer(tmpBufL0A, 2 * l0aSingle);
            pipe->InitBuffer(tmpBufL0B, 2 * l0bSingle);
            pipe->InitBuffer(tmpBufL0C, 2 * l0cSingle);
        }
        pipe->InitBuffer(kvTailPadUb, 16U * dFragCols * sizeof(KV_T));
        uint64_t pFrag1M = CeilDiv(mBaseDeal, 16U);
        uint64_t pFrag1N = CeilDiv(nDealMax, 16U);
        pipe->InitBuffer(pOnchipBufL1, 2 * pFrag1M * pFrag1N * ARCH31_FRAC_HALF_ELEMS * sizeof(half));
        if (isHp) {
            pipe->InitBuffer(qkResL1, 2 * mFragRows * pFragCols * sizeof(half));
            pipe->InitBuffer(pvResL1, 2 * mFragRows * dFragCols * sizeof(half));
        }
    }

    __aicore__ inline LocalTensor<half> GetQkResL1Tensor(uint32_t slot)
    {
        LocalTensor<half> ts = qkResL1.Get<half>();
        uint64_t mFragRows =
            static_cast<uint64_t>(CeilDiv(constInfo.mBaseDeal != 0 ? constInfo.mBaseDeal : constInfo.mBaseSize, 16U)) *
            16U;
        uint32_t nDealMax = constInfo.nDealMax != 0 ? constInfo.nDealMax : constInfo.nBaseSize;
        uint64_t qkFragCols = static_cast<uint64_t>(CeilDiv(nDealMax, 16U) * 16U);
        return ts[static_cast<uint64_t>(slot) * mFragRows * qkFragCols];
    }

    __aicore__ inline LocalTensor<half> GetPvResL1Tensor(uint32_t slot)
    {
        LocalTensor<half> ts = pvResL1.Get<half>();
        uint32_t D = HeadDim();
        uint64_t pvFragCols = static_cast<uint64_t>(CeilDiv(D, 16U) * 16U);
        uint64_t mFragRows =
            static_cast<uint64_t>(CeilDiv(constInfo.mBaseDeal != 0 ? constInfo.mBaseDeal : constInfo.mBaseSize, 16U)) *
            16U;
        return ts[static_cast<uint64_t>(slot) * mFragRows * pvFragCols];
    }

    // Cube Stage 1: QK^T
    __aicore__ inline void ComputeMm1(const RunInfoArch31& info)
    {
        uint64_t qGmOffset = info.qGmOffset;
        uint64_t kGmOffset = info.kGmOffset;
        uint32_t mDeal = info.mDeal;
        uint32_t nDeal = info.nDeal;
        uint32_t D = static_cast<uint32_t>(constInfo.headDim);
        uint32_t kTileQK = constInfo.kTileQK;
        if (kTileQK == 0) {
            kTileQK = Max(D / 2, ARCH31_CUBE_K);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        CopyQGmToL1(qGmOffset, mDeal, D);
        CopyKeyToL1(kGmOffset, nDeal, D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_QP);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_QP);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);

        uint32_t qN1Total = CeilDiv(mDeal, 16U);
        uint32_t kN1Total = CeilDiv(nDeal, 16U);
        LocalTensor<half> cL0Tensor = tmpBufL0C.Get<half>();
        LocalTensor<Q_T> aL0Tensor = tmpBufL0A.Get<Q_T>();
        LocalTensor<KV_T> bL0Tensor = tmpBufL0B.Get<KV_T>();
        LocalTensor<Q_T> aL1Tensor = qpBufL1.Get<Q_T>();
        LocalTensor<KV_T> bL1Tensor = kvBufL1.Get<KV_T>();
        for (uint32_t mTile = 0; mTile < mDeal; mTile += ARCH31_CUBE_M) {
            uint32_t mActSize = Min(ARCH31_CUBE_M, mDeal - mTile);
            // MMAD 分形要求 m=16，补齐只作用于矩阵乘；Fixpipe的mSize必须用
            // 有效行数：若用于补齐，会把L0C尾部15行零按dstStride写出
            const uint32_t mFixRows = mActSize;
            if (mActSize < ARCH31_CUBE_M) {
                mActSize = ARCH31_CUBE_M;
            }

            uint32_t nDealEff = Arch31NDealPad(nDeal);
            for (uint32_t nTile = 0; nTile < nDealEff; nTile += ARCH31_CUBE_N) {
                uint32_t nActSize = Min(ARCH31_CUBE_N, nDealEff - nTile);
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
                for (uint32_t kTile = 0; kTile < D; kTile += kTileQK) {
                    uint32_t kActSize = Min(kTileQK, D - kTile);
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
                    LoadL1ToL0A(aL0Tensor, aL1Tensor, mTile, kTile, mActSize, kActSize, qN1Total);
                    LoadL1ToL0BK(bL0Tensor, bL1Tensor, nTile, kTile, nActSize, kActSize, kN1Total);
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);

                    MmadParams mmadParams;
                    mmadParams.m = mActSize;
                    mmadParams.n = nActSize;
                    mmadParams.k = kActSize;
                    mmadParams.cmatrixSource = false;
                    mmadParams.cmatrixInitVal = (kTile == 0) ? 1 : 0;
                    Arch31Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                    AscendC::PipeBarrier<PIPE_M>();
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
                }
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);

                FixpipeParamsV220 fixParams;
                fixParams.mSize = mFixRows;
                fixParams.nSize = nActSize;
                fixParams.srcStride = CeilDiv(mActSize, ARCH31_CUBE_M) * ARCH31_CUBE_M;
                fixParams.ndNum = 1;
                uint64_t s2Stride = static_cast<uint64_t>(info.s2Stride);
                fixParams.dstStride = static_cast<uint32_t>(s2Stride);
                uint64_t mm1ResGmOffset = info.mm1ResGmOffset + mTile * s2Stride + nTile;
                Fixpipe(mm1ResGm[mm1ResGmOffset], cL0Tensor, fixParams);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
    }

    // Cube Stage 2： attn V
    __aicore__ inline void ComputeMm2(const RunInfoArch31& info)
    {
        uint64_t pGmOffset = info.pGmOffset;
        uint64_t vGmOffset = info.vGmOffset;
        uint32_t mDeal = info.mDeal;
        uint32_t nDeal = info.nDeal;
        uint32_t D = static_cast<uint32_t>(constInfo.headDim);
        uint32_t nTileV = constInfo.nTileV;
        uint32_t kTileV = constInfo.kTileV;
        if (nTileV == 0) {
            nTileV = D;
        }
        uint32_t nDealMax = constInfo.nDealMax;
        if (nDealMax == 0) {
            nDealMax = constInfo.nBaseSize;
        }
        if (kTileV == 0) {
            kTileV = nDealMax;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        uint64_t pRowStride = static_cast<uint64_t>(info.s2Stride);
        uint32_t nDealEff = Arch31NDealPad(nDeal);
        CopyPGmToL1(pGmOffset, mDeal, nDealEff, pRowStride);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_QP);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_QP);

        uint32_t pN1Total = CeilDiv(mDeal, 16U);
        uint32_t vN1Total = CeilDiv(nDeal, 16U);
        LocalTensor<half> cL0Tensor = tmpBufL0C.Get<half>();
        LocalTensor<half> aL0Tensor = tmpBufL0A.Get<half>();
        LocalTensor<KV_T> bL0Tensor = tmpBufL0B.Get<KV_T>();
        LocalTensor<half> aL1Tensor = qpBufL1.Get<half>();
        LocalTensor<KV_T> bL1Tensor = kvBufL1.Get<KV_T>();
        for (uint32_t nTileIdx = 0; nTileIdx < D; nTileIdx += nTileV) {
            uint32_t nAct = Min(nTileV, D - nTileIdx);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
            CopyValueToL1(vGmOffset + nTileIdx, nDeal, nAct);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);

            for (uint32_t mTile = 0; mTile < mDeal; mTile += ARCH31_CUBE_M) {
                uint32_t mActSize = Min(ARCH31_CUBE_M, mDeal - mTile);
                const uint32_t mFixRows = mActSize;
                if (mActSize < ARCH31_CUBE_M) {
                    mActSize = ARCH31_CUBE_M;
                }
                for (uint32_t nTile = 0; nTile < nAct; nTile += ARCH31_CUBE_N) {
                    uint32_t nActSize = Min(ARCH31_CUBE_N, nAct - nTile);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
                    for (uint32_t kTile = 0; kTile < nDealEff; kTile += kTileV) {
                        uint32_t kActSize = Min(kTileV, nDealEff - kTile);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
                        LoadL1ToL0A(aL0Tensor, aL1Tensor, mTile, kTile, mActSize, kActSize, pN1Total);
                        LoadL1ToL0BV(bL0Tensor, bL1Tensor, kTile, nTile, kActSize, nActSize, vN1Total);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);

                        MmadParams mmadParams;
                        mmadParams.m = mActSize;
                        mmadParams.n = nActSize;
                        mmadParams.k = kActSize;
                        mmadParams.cmatrixSource = false;
                        mmadParams.cmatrixInitVal = (kTile == 0) ? 1 : 0;
                        uint32_t bHalfOffset = (nTile % 16U) / 8U * (ARCH31_FRAC_HALF_ELEMS / 2U);
                        Arch31Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams, bHalfOffset);
                        AscendC::PipeBarrier<PIPE_M>();
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);

                    FixpipeParamsV220 fixParams;
                    fixParams.mSize = mFixRows;
                    fixParams.nSize = nActSize;
                    fixParams.srcStride = CeilDiv(mActSize, ARCH31_CUBE_M) * ARCH31_CUBE_M;
                    fixParams.ndNum = 1;
                    fixParams.dstStride = D;
                    uint64_t mm2ResGmOffset = info.mm2ResGmOffset + mTile * D + nTileIdx + nTile;
                    Fixpipe(mm2ResGm[mm2ResGmOffset], cL0Tensor, fixParams);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
    }

    __aicore__ inline LocalTensor<half> GetPL1Tensor(uint32_t slot)
    {
        LocalTensor<half> ts = pOnchipBufL1.Get<half>();
        uint32_t mFrag1 = CeilDiv(constInfo.mBaseSize, 16U);
        uint32_t nFrag1 = CeilDiv(constInfo.nDealMax, 16U);
        return ts[slot * static_cast<uint64_t>(mFrag1) * nFrag1 * ARCH31_FRAC_HALF_ELEMS];
    }

    __aicore__ inline void ComputeMm2OnchipPAOnly(const RunInfoArch31& info, uint32_t slot)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t nDeal = info.nDeal;
        uint64_t vGmOffset = info.vGmOffset;
        uint32_t D = static_cast<uint32_t>(constInfo.headDim);
        uint32_t nTileV = constInfo.nTileV;
        uint32_t kTileV = constInfo.kTileV;
        if (nTileV == 0) {
            nTileV = D;
        }
        uint32_t nDealMax = constInfo.nDealMax;
        if (nDealMax == 0) {
            nDealMax = constInfo.nBaseSize;
        }
        if (kTileV == 0) {
            kTileV = nDealMax;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        uint64_t pRowStride = static_cast<uint64_t>(info.s2Stride);
        uint32_t nDealEff = Arch31NDealPad(nDeal);
        (void)pRowStride;

        uint32_t pN1Total = CeilDiv(mDeal, 16U);
        uint32_t vN1Total = CeilDiv(nDeal, 16U);
        LocalTensor<half> cL0Tensor = tmpBufL0C.Get<half>();
        LocalTensor<half> aL0Tensor = tmpBufL0A.Get<half>();
        LocalTensor<KV_T> bL0Tensor = tmpBufL0B.Get<KV_T>();
        LocalTensor<half> aL1Tensor = GetPL1Tensor(slot);
        LocalTensor<KV_T> bL1Tensor = kvBufL1.Get<KV_T>();
        for (uint32_t nTileIdx = 0; nTileIdx < D; nTileIdx += nTileV) {
            uint32_t nAct = Min(nTileV, D - nTileIdx);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
            CopyValueToL1(vGmOffset + nTileIdx, nDeal, nAct);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);

            for (uint32_t mTile = 0; mTile < mDeal; mTile += ARCH31_CUBE_M) {
                uint32_t mActSize = Min(ARCH31_CUBE_M, mDeal - mTile);
                const uint32_t mFixRows = mActSize;
                if (mActSize < ARCH31_CUBE_M) {
                    mActSize = ARCH31_CUBE_M;
                }
                for (uint32_t nTile = 0; nTile < nAct; nTile += ARCH31_CUBE_N) {
                    uint32_t nActSize = Min(ARCH31_CUBE_N, nAct - nTile);
                    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
                    for (uint32_t kTile = 0; kTile < nDealEff; kTile += kTileV) {
                        uint32_t kActSize = Min(kTileV, nDealEff - kTile);
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
                        LoadL1ToL0A(aL0Tensor, aL1Tensor, mTile, kTile, mActSize, kActSize, pN1Total);
                        LoadL1ToL0BV(bL0Tensor, bL1Tensor, kTile, nTile, kActSize, nActSize, vN1Total);
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);

                        MmadParams mmadParams;
                        mmadParams.m = mActSize;
                        mmadParams.n = nActSize;
                        mmadParams.k = kActSize;
                        mmadParams.cmatrixSource = false;
                        mmadParams.cmatrixInitVal = (kTile == 0) ? 1 : 0;
                        uint32_t bHalfOffset = (nTile % 16U) / 8U * (ARCH31_FRAC_HALF_ELEMS / 2U);
                        Arch31Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams, bHalfOffset);
                        AscendC::PipeBarrier<PIPE_M>();
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);
                    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);

                    FixpipeParamsV220 fixParams;
                    fixParams.mSize = mFixRows;
                    fixParams.nSize = nActSize;
                    fixParams.srcStride = CeilDiv(mActSize, ARCH31_CUBE_M) * ARCH31_CUBE_M;
                    fixParams.ndNum = 1;
                    fixParams.dstStride = D;
                    uint64_t mm2ResGmOffset = info.mm2ResGmOffset + mTile * D + nTileIdx + nTile;
                    Fixpipe(mm2ResGm[mm2ResGmOffset], cL0Tensor, fixParams);
                    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(slot ? ARCH31_EVENT_ID_PL1FREE_P1 : ARCH31_EVENT_ID_PL1FREE_P0);
    }

    __aicore__ inline void LoadL1ToL0BKHpCoarse(LocalTensor<KV_T>& bL0, LocalTensor<KV_T>& bL1, uint32_t nTile,
                                                uint32_t kStart, uint32_t nActSize, uint32_t kActSize,
                                                uint32_t n1TotalSrc)
    {
        uint32_t nStep = CeilDiv(nActSize, 16U);
        uint32_t kStep = CeilDiv(kActSize, 16U);
        uint64_t srcFracBase = static_cast<uint64_t>(kStart / 16U) * n1TotalSrc + nTile / 16U;
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003))
        for (uint32_t k1 = 0; k1 < kStep; ++k1) {
            LoadData2DParams loadData2DParams;
            loadData2DParams.startIndex = 0;
            loadData2DParams.repeatTimes = static_cast<uint8_t>(nStep);
            loadData2DParams.srcStride = 1;
            loadData2DParams.dstGap = 0;
            loadData2DParams.ifTranspose = false;
            uint64_t srcOff = static_cast<uint64_t>(kStart / 16U + k1) * n1TotalSrc + nTile / 16U;
            LoadData(bL0[k1 * nStep * ARCH31_FRAC_HALF_ELEMS], bL1[srcOff * ARCH31_FRAC_HALF_ELEMS], loadData2DParams);
        }
#elif (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113))
        LoadData2DParamsV2 loadData2DParams;
        loadData2DParams.mStartPosition = 0;
        loadData2DParams.kStartPosition = 0;
        loadData2DParams.mStep = static_cast<uint16_t>(kStep);
        loadData2DParams.kStep = static_cast<uint16_t>(nStep);
        loadData2DParams.srcStride = static_cast<int32_t>(n1TotalSrc);
        loadData2DParams.dstStride = static_cast<uint16_t>(kStep);
        loadData2DParams.ifTranspose = false;
        LoadData(bL0, bL1[srcFracBase * ARCH31_FRAC_HALF_ELEMS], loadData2DParams);
#endif
    }

    __aicore__ inline void LoadL1ToL0BVHpCoarse(LocalTensor<KV_T>& bL0, LocalTensor<KV_T>& bL1, uint32_t kStart,
                                                uint32_t nTile, uint32_t kActSize, uint32_t nActSize,
                                                uint32_t n1TotalSrc)
    {
        uint32_t mStep = CeilDiv(kActSize, 16U);
        uint32_t nStep = CeilDiv(nActSize, 16U);
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003))
        for (uint32_t n1 = 0; n1 < nStep; ++n1) {
            LoadData2dTransposeParams loadData2DParams;
            loadData2DParams.startIndex = 0;
            loadData2DParams.repeatTimes = static_cast<uint8_t>(mStep);
            loadData2DParams.srcStride = 1;
            loadData2DParams.dstGap = static_cast<uint16_t>(nStep - 1U);
            loadData2DParams.dstFracGap = 0;
            loadData2DParams.addrMode = 0;
            uint64_t srcOff = (static_cast<uint64_t>(nTile / 16U + n1) * n1TotalSrc) + (kStart / 16U);
            LoadDataWithTranspose(bL0[n1 * ARCH31_FRAC_HALF_ELEMS], bL1[srcOff * ARCH31_FRAC_HALF_ELEMS],
                                  loadData2DParams);
        }
#elif (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113))
        LoadData2DParamsV2 loadData2DParams;
        loadData2DParams.mStartPosition = 0;
        loadData2DParams.kStartPosition = 0;
        loadData2DParams.mStep = static_cast<uint16_t>(mStep);
        loadData2DParams.kStep = static_cast<uint16_t>(nStep);
        loadData2DParams.srcStride = static_cast<int32_t>(n1TotalSrc);
        loadData2DParams.dstStride = static_cast<uint16_t>(mStep);
        loadData2DParams.ifTranspose = true;
        uint64_t srcFracBase = static_cast<uint64_t>(nTile / 16U) * n1TotalSrc + kStart / 16U;
        LoadData(bL0, bL1[srcFracBase * ARCH31_FRAC_HALF_ELEMS], loadData2DParams);
#else
        (void)bL0;
        (void)bL1;
        (void)kStart;
        (void)nTile;
        (void)kActSize;
        (void)nActSize;
        (void)n1TotalSrc;
#endif
    }

    __aicore__ inline void ComputeMm1HpCoarse(const RunInfoArch31& info, uint32_t slot)
    {
        uint64_t qGmOffset = info.qGmOffset;
        uint64_t kGmOffset = info.kGmOffset;
        uint32_t mDeal = info.mDeal;
        uint32_t nDeal = info.nDeal;
        uint32_t D = HeadDim();

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        CopyQGmToL1(qGmOffset, mDeal, D);
        CopyKeyToL1(kGmOffset, nDeal, D);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_QP);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_QP);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);

        uint32_t qN1Total = CeilDiv(mDeal, 16U);
        uint32_t kN1Total = CeilDiv(nDeal, 16U);
        uint32_t nDealEff = Arch31NDealPad(nDeal);
        uint32_t mPad = Arch31NDealPad(mDeal);
        uint32_t nDealMax = constInfo.nDealMax != 0 ? constInfo.nDealMax : constInfo.nBaseSize;
        uint64_t qkFragCols = static_cast<uint64_t>(CeilDiv(nDealMax, 16U) * 16U);

        LocalTensor<half> cL0Tensor = tmpBufL0C.Get<half>();
        LocalTensor<Q_T> aL0Tensor = tmpBufL0A.Get<Q_T>();
        LocalTensor<KV_T> bL0Tensor = tmpBufL0B.Get<KV_T>();
        LocalTensor<Q_T> aL1Tensor = qpBufL1.Get<Q_T>();
        LocalTensor<KV_T> bL1Tensor = kvBufL1.Get<KV_T>();
        LocalTensor<half> qkSlotTensor = GetQkResL1Tensor(slot);

        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
        LoadL1ToL0A(aL0Tensor, aL1Tensor, 0U, 0U, mPad, D, qN1Total);
        LoadL1ToL0BKHpCoarse(bL0Tensor, bL1Tensor, 0U, 0U, nDealEff, D, kN1Total);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);

        MmadParams mmadParams;
        mmadParams.m = mPad;
        mmadParams.n = nDealEff;
        mmadParams.k = D;
        mmadParams.cmatrixSource = false;
        mmadParams.cmatrixInitVal = 1;
        Arch31Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
        AscendC::PipeBarrier<PIPE_M>();
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);

        FixpipeParamsV220 fixParams;
        fixParams.mSize = mDeal;
        fixParams.nSize = nDealEff;
        fixParams.srcStride = mPad;
        fixParams.ndNum = 1;
        fixParams.dstStride = static_cast<uint32_t>(qkFragCols);
        Fixpipe(qkSlotTensor, cL0Tensor, fixParams);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
    }

    __aicore__ inline void ComputeMm2OnchipHpCoarse(const RunInfoArch31& info, uint32_t slot)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t nDeal = info.nDeal;
        uint64_t vGmOffset = info.vGmOffset;
        uint32_t D = HeadDim();
        uint32_t nTileV = constInfo.nTileV;
        if (nTileV == 0) {
            nTileV = D;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        uint32_t nDealEff = Arch31NDealPad(nDeal);
        uint32_t mPad = Arch31NDealPad(mDeal);
        uint32_t pN1Total = CeilDiv(mDeal, 16U);
        uint32_t vN1Total = CeilDiv(nDeal, 16U);
        uint64_t pvFragCols = static_cast<uint64_t>(CeilDiv(D, 16U) * 16U);

        LocalTensor<half> cL0Tensor = tmpBufL0C.Get<half>();
        LocalTensor<half> aL0Tensor = tmpBufL0A.Get<half>();
        LocalTensor<KV_T> bL0Tensor = tmpBufL0B.Get<KV_T>();
        LocalTensor<half> aL1Tensor = GetPL1Tensor(slot);
        LocalTensor<KV_T> bL1Tensor = kvBufL1.Get<KV_T>();
        LocalTensor<half> pvSlotTensor = GetPvResL1Tensor(slot);

        for (uint32_t nTileIdx = 0; nTileIdx < D; nTileIdx += nTileV) {
            uint32_t nAct = Min(nTileV, D - nTileIdx);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
            CopyValueToL1(vGmOffset + nTileIdx, nDeal, nAct);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
            LoadL1ToL0A(aL0Tensor, aL1Tensor, 0U, 0U, mPad, nDealEff, pN1Total);
            LoadL1ToL0BVHpCoarse(bL0Tensor, bL1Tensor, 0U, 0U, nDealEff, nAct, vN1Total);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);

            MmadParams mmadParams;
            mmadParams.m = mPad;
            mmadParams.n = nAct;
            mmadParams.k = nDealEff;
            mmadParams.cmatrixSource = false;
            mmadParams.cmatrixInitVal = 1;
            Arch31Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
            AscendC::PipeBarrier<PIPE_M>();
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);

            FixpipeParamsV220 fixParams;
            fixParams.mSize = mDeal;
            fixParams.nSize = nAct;
            fixParams.srcStride = mPad;
            fixParams.ndNum = 1;
            fixParams.dstStride = static_cast<uint32_t>(pvFragCols);
            Fixpipe(pvSlotTensor[nTileIdx], cL0Tensor, fixParams);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE3>(slot ? ARCH31_EVENT_ID_PL1FREE_P1 : ARCH31_EVENT_ID_PL1FREE_P0);
    }

    __aicore__ inline void ComputeMm2HpCoarse(const RunInfoArch31& info, uint32_t slot)
    {
        uint64_t pGmOffset = info.pGmOffset;
        uint64_t vGmOffset = info.vGmOffset;
        uint32_t mDeal = info.mDeal;
        uint32_t nDeal = info.nDeal;
        uint32_t D = HeadDim();
        uint32_t nTileV = constInfo.nTileV;
        if (nTileV == 0) {
            nTileV = D;
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
        uint64_t pRowStride = static_cast<uint64_t>(info.s2Stride);
        uint32_t nDealEff = Arch31NDealPad(nDeal);
        uint32_t mPad = Arch31NDealPad(mDeal);
        CopyPGmToL1(pGmOffset, mDeal, nDealEff, pRowStride);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_QP);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_QP);

        uint32_t pN1Total = CeilDiv(mDeal, 16U);
        uint32_t vN1Total = CeilDiv(nDeal, 16U);
        uint64_t pvFragCols = static_cast<uint64_t>(CeilDiv(D, 16U) * 16U);

        LocalTensor<half> cL0Tensor = tmpBufL0C.Get<half>();
        LocalTensor<half> aL0Tensor = tmpBufL0A.Get<half>();
        LocalTensor<KV_T> bL0Tensor = tmpBufL0B.Get<KV_T>();
        LocalTensor<half> aL1Tensor = qpBufL1.Get<half>();
        LocalTensor<KV_T> bL1Tensor = kvBufL1.Get<KV_T>();
        LocalTensor<half> pvSlotTensor = GetPvResL1Tensor(slot);

        for (uint32_t nTileIdx = 0; nTileIdx < D; nTileIdx += nTileV) {
            uint32_t nAct = Min(nTileV, D - nTileIdx);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
            CopyValueToL1(vGmOffset + nTileIdx, nDeal, nAct);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(ARCH31_EVENT_ID_L1_KV);

            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
            LoadL1ToL0A(aL0Tensor, aL1Tensor, 0U, 0U, mPad, nDealEff, pN1Total);
            LoadL1ToL0BVHpCoarse(bL0Tensor, bL1Tensor, 0U, 0U, nDealEff, nAct, vN1Total);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(ARCH31_EVENT_ID_L0AB);

            MmadParams mmadParams;
            mmadParams.m = mPad;
            mmadParams.n = nAct;
            mmadParams.k = nDealEff;
            mmadParams.cmatrixSource = false;
            mmadParams.cmatrixInitVal = 1;
            Arch31Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
            AscendC::PipeBarrier<PIPE_M>();
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(ARCH31_EVENT_ID_L0AB);
            AscendC::SetFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);
            AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(ARCH31_EVENT_ID_L0C);

            FixpipeParamsV220 fixParams;
            fixParams.mSize = mDeal;
            fixParams.nSize = nAct;
            fixParams.srcStride = mPad;
            fixParams.ndNum = 1;
            fixParams.dstStride = static_cast<uint32_t>(pvFragCols);
            Fixpipe(pvSlotTensor[nTileIdx], cL0Tensor, fixParams);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(ARCH31_EVENT_ID_L0C);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_KV);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(ARCH31_EVENT_ID_L1_QP);
    }

protected:
    __aicore__ inline void Arch31Mmad(const LocalTensor<half>& dst, const LocalTensor<half>& fm,
                                      const LocalTensor<half>& filter, const MmadParams& mmadParams,
                                      uint32_t bHalfOffset = 0)
    {
        mad(reinterpret_cast<__cc__ half*>(dst.GetPhyAddr()), reinterpret_cast<__ca__ half*>(fm.GetPhyAddr()),
            reinterpret_cast<__cb__ half*>(filter.GetPhyAddr()) + bHalfOffset, 0, mmadParams.m, mmadParams.k,
            mmadParams.n, mmadParams.unitFlag, false, mmadParams.cmatrixSource, mmadParams.cmatrixInitVal);
    }

    __aicore__ inline void CopyNd2NzHelper(LocalTensor<half>& l1Tensor, GlobalTensor<half>& srcGm, uint64_t gmOffset,
                                           uint32_t rows, uint32_t cols, uint64_t gmRowStride)
    {
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = rows;
        nd2nzParams.dValue = cols;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = gmRowStride;
        nd2nzParams.dstNzC0Stride = CeilDiv(rows, 16U) * 16U;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        DataCopy(l1Tensor, srcGm[gmOffset], nd2nzParams);
    }

    __aicore__ inline void CopyQGmToL1(uint64_t qGmOffset, uint32_t mDeal, uint32_t dInner)
    {
        constexpr bool isBnsdLayout = (FIAT::layout == FIA_LAYOUT_ARCH31::BNSD);
        uint64_t qRowStride = isBnsdLayout ? static_cast<uint64_t>(dInner) : constInfo.qHeadNum * constInfo.headDim;
        LocalTensor<Q_T> qpL1 = qpBufL1.Get<Q_T>();
        CopyNd2NzHelper(qpL1, queryGm, qGmOffset, mDeal, dInner, qRowStride);
    }

    __aicore__ inline void CopyTailFracPadToL1(GlobalTensor<KV_T>& srcGm, uint64_t gmOffset, uint32_t nDeal,
                                               uint32_t cols, uint64_t gmRowStride, LocalTensor<KV_T>& l1Tensor)
    {
        uint32_t nDealPad = Arch31NDealPad(nDeal);
        uint32_t nDealWhole = nDeal & ~15U;
        uint32_t nDealRem = nDeal - nDealWhole;
        uint32_t n1Total = CeilDiv(nDeal, 16U);
        LocalTensor<KV_T> padUb = kvTailPadUb.Get<KV_T>();
        if (nDealWhole > 0) {
            Nd2NzParams nd2nzParams;
            nd2nzParams.ndNum = 1;
            nd2nzParams.nValue = nDealWhole;
            nd2nzParams.dValue = cols;
            nd2nzParams.srcNdMatrixStride = 0;
            nd2nzParams.srcDValue = gmRowStride;
            nd2nzParams.dstNzC0Stride = n1Total * 16U;
            nd2nzParams.dstNzNStride = 1;
            nd2nzParams.dstNzMatrixStride = 0;
            DataCopy(l1Tensor, srcGm[gmOffset], nd2nzParams);
        }
        DataCopyExtParams extParams;
        extParams.blockCount = static_cast<uint16_t>(nDealRem);
        extParams.blockLen = cols * sizeof(KV_T);
        extParams.srcStride = static_cast<uint32_t>((gmRowStride - cols) * sizeof(KV_T));
        extParams.dstStride = 0;
        DataCopyPadExtParams<KV_T> padExtParams(false, 0, 0, 0);
        DataCopyPad(padUb, srcGm[gmOffset + nDealWhole * gmRowStride], extParams, padExtParams);
        SetFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_KV_TAIL);
        WaitFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_KV_TAIL);

        __VEC_SCOPE__
        {
            for (uint16_t r = static_cast<uint16_t>(nDealRem); r < static_cast<uint16_t>(nDealPad); r++) {
                Duplicate<KV_T>(padUb[r * cols], 0, cols);
            }
        }
        SetFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_KV_TAIL);
        WaitFlag<HardEvent::V_MTE3>(ARCH31_EVENT_ID_KV_TAIL);
        uint32_t k1Total = cols / 16U;
        DataCopyParams cpyParams;
        cpyParams.blockCount = 16U;
        cpyParams.blockLen = 1U;
        cpyParams.srcStride = (cols - 16U) * sizeof(KV_T) / ONE_BLOCK_SIZE;
        cpyParams.dstStride = 0;
        for (uint32_t g = 0; g < k1Total; g++) {
            uint64_t l1Dst = (static_cast<uint64_t>(g) * n1Total + (n1Total - 1)) * ARCH31_FRAC_HALF_ELEMS;
            DataCopy(l1Tensor[l1Dst], padUb[g * 16U], cpyParams);
        }
        SetFlag<HardEvent::MTE3_MTE1>(ARCH31_EVENT_ID_KV_TAIL);
        WaitFlag<HardEvent::MTE3_MTE1>(ARCH31_EVENT_ID_KV_TAIL);
    }

    __aicore__ inline void CopyKeyToL1(uint64_t kGmOffset, uint32_t nDeal, uint32_t dInner)
    {
        constexpr bool isBnsdLayout = (FIAT::layout == FIA_LAYOUT_ARCH31::BNSD);
        uint64_t kRowStride = isBnsdLayout ? static_cast<uint64_t>(dInner) : constInfo.kvHeadNum * constInfo.headDim;
        LocalTensor<KV_T> kvL1 = kvBufL1.Get<KV_T>();
        if (nDeal % 16U != 0) {
            CopyTailFracPadToL1(keyGm, kGmOffset, nDeal, dInner, kRowStride, kvL1);
            return;
        }
        CopyNd2NzHelper(kvL1, keyGm, kGmOffset, nDeal, dInner, kRowStride);
    }

    __aicore__ inline void CopyValueToL1(uint64_t vGmOffset, uint32_t nDeal, uint32_t dInner)
    {
        constexpr bool isBnsdLayout = (FIAT::layout == FIA_LAYOUT_ARCH31::BNSD);
        uint64_t vRowStride =
            isBnsdLayout ? static_cast<uint64_t>(constInfo.headDim) : constInfo.kvHeadNum * constInfo.headDim;
        LocalTensor<KV_T> kvL1 = kvBufL1.Get<KV_T>();
        if (nDeal % 16U != 0) {
            CopyTailFracPadToL1(valueGm, vGmOffset, nDeal, dInner, vRowStride, kvL1);
            return;
        }
        CopyNd2NzHelper(kvL1, valueGm, vGmOffset, nDeal, dInner, vRowStride);
    }

    __aicore__ inline void CopyPGmToL1(uint64_t pGmOffset, uint32_t mDeal, uint32_t nDeal, uint64_t pRowStride)
    {
        LocalTensor<half> qpL1 = qpBufL1.Get<half>();
        CopyNd2NzHelper(qpL1, softmaxPFp16Gm, pGmOffset, mDeal, nDeal, pRowStride);
    }

    static constexpr uint32_t ARCH31_FRAC_HALF_ELEMS = 256; // 16*16

    __aicore__ inline void LoadL1ToL0A(LocalTensor<Q_T>& aL0, LocalTensor<Q_T>& aL1, uint32_t mTile, uint32_t kStart,
                                       uint32_t mActSize, uint32_t kActSize, uint32_t n1TotalSrc)
    {
        uint32_t mStep = CeilDiv(mActSize, 16U);
        uint32_t kStep = CeilDiv(kActSize, 16U);
        uint64_t srcFracBase = static_cast<uint64_t>(kStart / 16U) * n1TotalSrc + mTile / 16U;
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003))
        for (uint32_t m1 = 0; m1 < mStep; ++m1) {
            LoadData2DParams loadData2DParams;
            loadData2DParams.startIndex = 0;
            loadData2DParams.repeatTimes = static_cast<uint8_t>(kStep);
            loadData2DParams.srcStride = static_cast<uint16_t>(n1TotalSrc);
            loadData2DParams.dstGap = 0;
            loadData2DParams.ifTranspose = false;
            LoadData(aL0[m1 * kStep * ARCH31_FRAC_HALF_ELEMS], aL1[(srcFracBase + m1) * ARCH31_FRAC_HALF_ELEMS],
                     loadData2DParams);
        }
#elif (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113))
        LoadData2DParamsV2 loadData2DParams;
        loadData2DParams.mStartPosition = 0;
        loadData2DParams.kStartPosition = 0;
        loadData2DParams.mStep = static_cast<uint16_t>(mStep);
        loadData2DParams.kStep = static_cast<uint16_t>(kStep);
        loadData2DParams.srcStride = static_cast<int32_t>(n1TotalSrc);
        loadData2DParams.dstStride = static_cast<uint16_t>(mStep);
        loadData2DParams.ifTranspose = false;
        LoadData(aL0, aL1[srcFracBase * ARCH31_FRAC_HALF_ELEMS], loadData2DParams);
#endif
    }

    __aicore__ inline void LoadL1ToL0BK(LocalTensor<KV_T>& bL0, LocalTensor<KV_T>& bL1, uint32_t nTile, uint32_t kStart,
                                        uint32_t nActSize, uint32_t kActSize, uint32_t n1TotalSrc)
    {
        uint32_t mStep = CeilDiv(nActSize, 16U);
        uint32_t kStep = CeilDiv(kActSize, 16U);
        uint32_t nHalf = (nTile % 16U) / 8U;
        uint64_t srcFracBase = static_cast<uint64_t>(kStart / 16U) * n1TotalSrc + nTile / 16U;
        uint64_t srcHalfOffset = static_cast<uint64_t>(nHalf) * (ARCH31_FRAC_HALF_ELEMS / 2U);
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003))
        for (uint32_t m1 = 0; m1 < mStep; ++m1) {
            LoadData2DParams loadData2DParams;
            loadData2DParams.startIndex = 0;
            loadData2DParams.repeatTimes = static_cast<uint8_t>(kStep);
            loadData2DParams.srcStride = static_cast<uint16_t>(n1TotalSrc);
            loadData2DParams.dstGap = 0;
            loadData2DParams.ifTranspose = false;
            LoadData(bL0[m1 * kStep * ARCH31_FRAC_HALF_ELEMS],
                     bL1[(srcFracBase + m1) * ARCH31_FRAC_HALF_ELEMS + srcHalfOffset], loadData2DParams);
        }
#elif (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113))
        LoadData2DParamsV2 loadData2DParams;
        loadData2DParams.mStartPosition = 0;
        loadData2DParams.kStartPosition = 0;
        loadData2DParams.mStep = static_cast<uint16_t>(mStep);
        loadData2DParams.kStep = static_cast<uint16_t>(kStep);
        loadData2DParams.srcStride = static_cast<int32_t>(n1TotalSrc);
        loadData2DParams.dstStride = static_cast<uint16_t>(mStep);
        loadData2DParams.ifTranspose = false;
        LoadData(bL0, bL1[srcFracBase * ARCH31_FRAC_HALF_ELEMS], loadData2DParams);
#endif
    }

    __aicore__ inline void LoadL1ToL0BV(LocalTensor<KV_T>& bL0, LocalTensor<KV_T>& bL1, uint32_t kStart, uint32_t nTile,
                                        uint32_t kActSize, uint32_t nActSize, uint32_t n1TotalSrc)
    {
        uint32_t mStep = CeilDiv(kActSize, 16U);
        uint32_t kStep = CeilDiv(nActSize, 16U);
        uint32_t dstStrideFrags = CeilDiv(nActSize, 16U);
        uint64_t srcFracBase = static_cast<uint64_t>(nTile / 16U) * n1TotalSrc + kStart / 16U;
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003))
        for (uint32_t k1 = 0; k1 < kStep; ++k1) {
            LoadData2dTransposeParams loadData2DParams;
            loadData2DParams.startIndex = 0;
            loadData2DParams.repeatTimes = static_cast<uint8_t>(mStep);
            loadData2DParams.srcStride = 1;
            loadData2DParams.dstGap = 0;
            loadData2DParams.dstFracGap = 0;
            loadData2DParams.addrMode = 0;
            LoadDataWithTranspose(bL0[k1 * dstStrideFrags * ARCH31_FRAC_HALF_ELEMS],
                                  bL1[(srcFracBase + k1 * n1TotalSrc) * ARCH31_FRAC_HALF_ELEMS], loadData2DParams);
        }
#elif (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3113))
        LoadData2DParamsV2 loadData2DParams;
        loadData2DParams.mStartPosition = 0;
        loadData2DParams.kStartPosition = 0;
        loadData2DParams.mStep = static_cast<uint16_t>(mStep);
        loadData2DParams.kStep = static_cast<uint16_t>(kStep);
        loadData2DParams.srcStride = static_cast<int32_t>(n1TotalSrc);
        loadData2DParams.dstStride = static_cast<uint16_t>(dstStrideFrags);
        loadData2DParams.ifTranspose = true;
        LoadData(bL0, bL1[srcFracBase * ARCH31_FRAC_HALF_ELEMS], loadData2DParams);
#endif
    }

protected:
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;   // Cube QK^T 输出
    GlobalTensor<half> softmaxPFp16Gm; // softmaxP FP16
    GlobalTensor<MM_OUT_T> mm2ResGm;   // Cube attn V 输出

    TBuf<TPosition::A1> qpBufL1;
    TBuf<TPosition::A1> kvBufL1;
    TBuf<TPosition::A1> pOnchipBufL1;
    TBuf<TPosition::A1> qkResL1;
    TBuf<TPosition::A1> pvResL1;
    TBuf<TPosition::A2> tmpBufL0A;
    TBuf<TPosition::B2> tmpBufL0B;
    TBuf<TPosition::CO1> tmpBufL0C;
    TBuf<TPosition::VECCALC> kvTailPadUb;

    ConstInfoArch31 constInfo{};
};

} // namespace AttentionCommonArch31

#endif // FIA_BLOCK_CUBE_NONQUANT_ARCH31_H
