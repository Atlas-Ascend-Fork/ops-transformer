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
 * \file fia_block_vec_nonquant_arch31.h
 * \brief arch31 (KirinX90 / Kirin9030) Vector Stage 实现，使用MicroAPIO / RegBase。
 */
#ifndef FIA_BLOCK_VEC_NONQUANT_ARCH31_H
#define FIA_BLOCK_VEC_NONQUANT_ARCH31_H

#include "fia_public_define_arch31.h"

#if defined(__has_include) && __has_include("micro_api/kernel_micro_intf.h")
#include "micro_api/kernel_micro_intf.h"
#else
namespace MicroAPI = AscendC::Reg;
#endif

using namespace AttentionCommonArch31;
using AttentionCommon::Min;
using AttentionCommon::Max;
using AttentionCommon::CeilDiv;

namespace AttentionCommonArch31 {
#if defined(__has_include) && __has_include("micro_api/kernel_micro_intf.h")
__aicore__ inline void FiaMaskSlideB8Arch31(MicroAPI::MaskReg& dstMask, MicroAPI::MaskReg& srcMask0,
                                            MicroAPI::MaskReg& srcMask1, int16_t slideAmount)
{
    MicroAPI::MaskSlide(dstMask, srcMask0, srcMask1, slideAmount);
}
#else
__aicore__ inline void FiaMaskSlideB8Arch31(MicroAPI::MaskReg& dstMask, MicroAPI::MaskReg& srcMask0,
                                            MicroAPI::MaskReg& srcMask1, int16_t slideAmount)
{
    pslide_b8(dstMask, srcMask0, srcMask1, slideAmount);
}
#endif

struct ReduceMaxOp {
    template <typename T>
    static __aicore__ inline void Init(MicroAPI::RegTensor<T>& reg)
    {
        MicroAPI::Duplicate<T>(reg, static_cast<T>(-3.0e+38f));
    }
    template <typename T, MicroAPI::MaskMergeMode mode>
    static __aicore__ inline void Apply(MicroAPI::RegTensor<T>& dst, MicroAPI::RegTensor<T>& src0,
                                        MicroAPI::RegTensor<T>& src1, MicroAPI::MaskReg& preg)
    {
        MicroAPI::Max<T, mode>(dst, src0, src1, preg);
    }
    template <typename T, MicroAPI::MaskMergeMode mode>
    static __aicore__ inline void Reduce(MicroAPI::RegTensor<T>& dst, MicroAPI::RegTensor<T>& src,
                                         MicroAPI::MaskReg& preg)
    {
        MicroAPI::ReduceMax<T, mode>(dst, src, preg);
    }
};

struct ReduceSumOp {
    template <typename T>
    static __aicore__ inline void Init(MicroAPI::RegTensor<T>& reg)
    {
        MicroAPI::Duplicate<T>(reg, static_cast<T>(0.0f));
    }
    template <typename T, MicroAPI::MaskMergeMode mode>
    static __aicore__ inline void Apply(MicroAPI::RegTensor<T>& dst, MicroAPI::RegTensor<T>& src0,
                                        MicroAPI::RegTensor<T>& src1, MicroAPI::MaskReg& preg)
    {
        MicroAPI::Add<T, mode>(dst, src0, src1, preg);
    }
    template <typename T, MicroAPI::MaskMergeMode mode>
    static __aicore__ inline void Reduce(MicroAPI::RegTensor<T>& dst, MicroAPI::RegTensor<T>& src,
                                         MicroAPI::MaskReg& preg)
    {
        MicroAPI::ReduceSum<T, T, mode>(dst, src, preg);
    }
};

template <typename FIAT>
class FiaBlockVecNonQuantArch31 {
public:
    using T = float; // 中间累积用 FP32 保精度
    using Q_T = typename FIAT::queryType;
    using KV_T = typename FIAT::kvType;
    using OUT_T = typename FIAT::outputType;

    static constexpr uint32_t perVegNum = ARCH31_PER_VEG_NUM_FP32;
    static constexpr uint32_t ONE_BLOCK_SIZE = 32U;

    __aicore__ inline uint32_t HeadDim() const
    {
        return FIAT::headDim != 0U ? FIAT::headDim : static_cast<uint32_t>(constInfo.headDim);
    }

    template <typename U, TPosition Pos>
    static __aicore__ inline __ubuf__ U* GetUbAddr(TBuf<Pos>& buf)
    {
        LocalTensor<U> ts = buf.template Get<U>();
        return reinterpret_cast<__ubuf__ U*>(ts.GetPhyAddr());
    }

    static constexpr float FLOAT_INF_NEG = -3e+38f;
    static constexpr float FLOAT_INF_POS = 3e+38f;

    static __aicore__ inline float Uint2Float(uint32_t v)
    {
        return static_cast<float>(static_cast<int32_t>(v));
    }
    static __aicore__ inline float Int64ToFloat(int64_t v)
    {
        return static_cast<float>(static_cast<int32_t>(v));
    }

    static constexpr MicroAPI::CastTrait castF32F16 = {
        MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::SAT,
        MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT,
    };
    static constexpr MicroAPI::CastTrait castF16F32 = {
        MicroAPI::RegLayout::ZERO,
        MicroAPI::SatMode::UNKNOWN,
        MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::UNKNOWN,
    };

    __aicore__ inline FiaBlockVecNonQuantArch31() {}

    __aicore__ inline void Init(GlobalTensor<half> mm1ResGmIn, GlobalTensor<half> softmaxPFp16GmIn,
                                GlobalTensor<half> mm2ResGmIn, GlobalTensor<half> attentionOutGmIn,
                                GlobalTensor<uint8_t> attenMaskGmIn, __gm__ uint8_t* attenMaskPtrIn,
                                ConstInfoArch31 constInfoIn, TPipe* pipe)
    {
        mm1ResGm = mm1ResGmIn;
        softmaxPFp16Gm = softmaxPFp16GmIn;
        mm2ResGm = mm2ResGmIn;
        attentionOutGm = attentionOutGmIn;
        attenMaskGm = attenMaskGmIn;
        attenMaskPtr = attenMaskPtrIn;
        constInfo = constInfoIn;

        uint32_t scratchRows = constInfo.mBaseSize;
        uint32_t scratchCols = Max(constInfo.nBaseUb, constInfo.dInner);
        pipe->InitBuffer(s2ScratchUb, scratchRows * scratchCols * sizeof(T));
        pipe->InitBuffer(pOutFp16Ub, scratchRows * scratchCols * sizeof(half));
        pipe->InitBuffer(rowMaxAccUb, constInfo.mBaseSize * sizeof(T));
        pipe->InitBuffer(rowMaxPrevUb, constInfo.mBaseSize * sizeof(T));
        pipe->InitBuffer(rowSumAccUb, constInfo.mBaseSize * sizeof(T));
        pipe->InitBuffer(O_accUb, constInfo.mBaseSize * constInfo.dInner * sizeof(T));
        if (constInfo.attenMaskFlag || constInfo.attenMaskTensorFlag) {
            pipe->InitBuffer(attenMaskUb, constInfo.mBaseSize * constInfo.nBaseUb * sizeof(uint8_t));
        }
    }

    // 在线 softmax 累积初始化
    __aicore__ inline void InitAccumulators(uint32_t mDeal, uint32_t dInner)
    {
        __ubuf__ T* rowMaxAddr = GetUbAddr<T>(rowMaxAccUb);
        __ubuf__ T* rowMaxPrevAddr = GetUbAddr<T>(rowMaxPrevUb);
        __ubuf__ T* rowSumAddr = GetUbAddr<T>(rowSumAccUb);
        __ubuf__ T* oAccAddr = GetUbAddr<T>(O_accUb);
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> negInfReg, zeroReg;
            MicroAPI::Duplicate<T>(negInfReg, FLOAT_INF_NEG);
            MicroAPI::Duplicate<T>(zeroReg, static_cast<T>(0.0f));
            uint32_t totalMax = mDeal;
            uint32_t totalO = mDeal * dInner;
            uint16_t rowMaxLoopNum = static_cast<uint16_t>(CeilDiv(totalMax, perVegNum));
            uint32_t remainingMax = totalMax;
            for (uint16_t i = 0; i < rowMaxLoopNum; i++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<T>(remainingMax);
                MicroAPI::DataCopy(rowMaxAddr + i * perVegNum, negInfReg, pregLoop);
                MicroAPI::DataCopy(rowMaxPrevAddr + i * perVegNum, negInfReg, pregLoop);
                MicroAPI::DataCopy(rowSumAddr + i * perVegNum, zeroReg, pregLoop);
            }
            uint16_t oAccLoopNum = static_cast<uint16_t>(CeilDiv(totalO, perVegNum));
            uint32_t remainingO = totalO;
            for (uint16_t i = 0; i < oAccLoopNum; i++) {
                MicroAPI::MaskReg pregLoop = MicroAPI::UpdateMask<T>(remainingO);
                MicroAPI::DataCopy(oAccAddr + i * perVegNum, zeroReg, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyMm1ResGmToUb(const RunInfoArch31& info)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t nDeal = info.nDeal;
        uint64_t s2Stride = static_cast<uint64_t>(info.s2Stride);
        uint32_t nDealEff = Arch31NDealPad(nDeal);
        LocalTensor<half> mm1Ub = pOutFp16Ub.Get<half>();
        DataCopyParams mm1CopyParams;
        mm1CopyParams.blockCount = mDeal;
        mm1CopyParams.blockLen = nDealEff * sizeof(half) / ONE_BLOCK_SIZE;
        mm1CopyParams.srcStride = (s2Stride * sizeof(half) - nDealEff * sizeof(half)) / ONE_BLOCK_SIZE;
        mm1CopyParams.dstStride = 0;
        DataCopy(mm1Ub, mm1ResGm[info.mm1ResGmOffset], mm1CopyParams);
    }

    __aicore__ inline void CopyMm2ResGmToUb(const RunInfoArch31& info)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t dInner = info.dInner;
        LocalTensor<half> mm2Ub = pOutFp16Ub.Get<half>();
        DataCopyParams mm2CopyParams;
        mm2CopyParams.blockCount = mDeal;
        mm2CopyParams.blockLen = dInner * sizeof(half) / ONE_BLOCK_SIZE;
        mm2CopyParams.srcStride = (constInfo.headDim * sizeof(half) - dInner * sizeof(half)) / ONE_BLOCK_SIZE;
        mm2CopyParams.dstStride = 0;
        DataCopy(mm2Ub, mm2ResGm[info.mm2ResGmOffset], mm2CopyParams);
    }

    __aicore__ inline void CopyMm1ResL1ToUb(const RunInfoArch31& info, LocalTensor<half> srcL1)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t nDeal = info.nDeal;
        uint32_t nDealEff = Arch31NDealPad(nDeal);
        uint32_t nDealMax = constInfo.nDealMax != 0 ? constInfo.nDealMax : constInfo.nBaseSize;
        uint64_t qkFragCols = static_cast<uint64_t>(CeilDiv(nDealMax, 16U) * 16U);
        LocalTensor<half> mm1Ub = pOutFp16Ub.Get<half>();
        DataCopyParams mm1CopyParams;
        mm1CopyParams.blockCount = static_cast<uint16_t>(mDeal);
        mm1CopyParams.blockLen = static_cast<uint16_t>(nDealEff * sizeof(half) / ONE_BLOCK_SIZE);
        mm1CopyParams.srcStride =
            static_cast<uint16_t>((qkFragCols * sizeof(half) - nDealEff * sizeof(half)) / ONE_BLOCK_SIZE);
        mm1CopyParams.dstStride = 0;
        DataCopy(mm1Ub, srcL1, mm1CopyParams);
    }

    __aicore__ inline void CopyMm2ResL1ToUb(const RunInfoArch31& info, LocalTensor<half> srcL1)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t dInner = info.dInner;
        uint32_t D = HeadDim();
        uint64_t pvFragCols = static_cast<uint64_t>(CeilDiv(D, 16U) * 16U);
        LocalTensor<half> mm2Ub = pOutFp16Ub.Get<half>();
        DataCopyParams mm2CopyParams;
        mm2CopyParams.blockCount = static_cast<uint16_t>(mDeal);
        mm2CopyParams.blockLen = static_cast<uint16_t>(dInner * sizeof(half) / ONE_BLOCK_SIZE);
        mm2CopyParams.srcStride =
            static_cast<uint16_t>((pvFragCols * sizeof(half) - dInner * sizeof(half)) / ONE_BLOCK_SIZE);
        mm2CopyParams.dstStride = 0;
        DataCopy(mm2Ub, srcL1, mm2CopyParams);
    }

    __aicore__ inline void CopySoftmaxPFp16UbToGm(const RunInfoArch31& info)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t nDeal = info.nDeal;
        uint64_t s2Stride = static_cast<uint64_t>(info.s2Stride);
        uint32_t nDealEff = Arch31NDealPad(nDeal);
        LocalTensor<half> pFp16Ub = pOutFp16Ub.Get<half>();
        DataCopyParams pCopyParams;
        pCopyParams.blockCount = mDeal;
        pCopyParams.blockLen = nDealEff * sizeof(half) / ONE_BLOCK_SIZE;
        pCopyParams.srcStride = 0;
        pCopyParams.dstStride = (s2Stride * sizeof(half) - nDealEff * sizeof(half)) / ONE_BLOCK_SIZE;
        DataCopy(softmaxPFp16Gm[info.pGmOffset], pFp16Ub, pCopyParams);
    }

    __aicore__ inline void CopyOutFp16UbToGm(const RunInfoArch31& info)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t dInner = info.dInner;
        constexpr bool isBnsdLayout = (FIAT::layout == FIA_LAYOUT_ARCH31::BNSD);
        uint64_t outRowStride = isBnsdLayout ? constInfo.headDim : constInfo.qHeadNum * constInfo.headDim;
        LocalTensor<half> outUb = pOutFp16Ub.Get<half>();
        DataCopyParams outCopyParams;
        outCopyParams.blockCount = mDeal;
        outCopyParams.blockLen = dInner * sizeof(half) / ONE_BLOCK_SIZE;
        outCopyParams.srcStride = 0;
        outCopyParams.dstStride =
            static_cast<uint16_t>((outRowStride * sizeof(half) - dInner * sizeof(half)) / ONE_BLOCK_SIZE);
        DataCopy(attentionOutGm[info.outGmOffset], outUb, outCopyParams);
    }

    __aicore__ inline void CopyPUbToL1Nd2Nz(const RunInfoArch31& info, LocalTensor<half> dstL1)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t nDeal = info.nDeal;
        uint32_t nDealEff = Arch31NDealPad(nDeal);
        LocalTensor<half> srcUb = pOutFp16Ub.Get<half>();
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = static_cast<uint16_t>(mDeal);
        nd2nzParams.dValue = static_cast<uint16_t>(nDealEff);
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = static_cast<uint16_t>(nDealEff);
        nd2nzParams.dstNzC0Stride = static_cast<uint16_t>(CeilDiv(mDeal, 16U) * 16U);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        DataCopy(dstL1, srcUb, nd2nzParams);
    }

    __aicore__ inline void BroadcastMaskRows(uint32_t mRows, uint32_t mDeal, uint32_t nDeal)
    {
        if (mDeal <= mRows) {
            return;
        }
        __ubuf__ uint8_t* maskBase = GetUbAddr<uint8_t>(attenMaskUb);
        __VEC_SCOPE__
        {
            MicroAPI::MaskReg preg = MicroAPI::UpdateMask<uint8_t>(nDeal);
            MicroAPI::RegTensor<uint8_t> rowReg;
            MicroAPI::DataCopy(rowReg, maskBase);
            for (uint16_t r = static_cast<uint16_t>(mRows); r < static_cast<uint16_t>(mDeal); ++r) {
                MicroAPI::DataCopy(maskBase + r * constInfo.nBaseUb, rowReg, preg);
            }
        }
    }

    __aicore__ inline void CopyAttenMaskGmToUb(const RunInfoArch31& info)
    {
        if (!constInfo.attenMaskTensorFlag || attenMaskPtr == nullptr) {
            return;
        }
        bool isMaskFirstTile = (info.bIdx == 0 && info.hIdx == 0 && info.s1Tile == 0 && info.isFirstS2Loop);
        if (!isMaskFirstTile) {
            WaitFlag<HardEvent::V_MTE2>(ARCH31_EVENT_ID_MASK);
        }
        uint32_t mDeal = info.mDeal;
        uint32_t mRows = mDeal;
        if (constInfo.maskMDeal != 0 && mDeal > constInfo.maskMDeal) {
            mRows = constInfo.maskMDeal;
        }
        uint32_t nDeal = info.nDeal;
        LocalTensor<uint8_t> maskUb = attenMaskUb.Get<uint8_t>();
        if (constInfo.sparseMode == 9) {
            uint32_t mWidth = (FIAT::layout == FIA_LAYOUT_ARCH31::TND) ? info.maskRowWidth :
                                                                         static_cast<uint32_t>(constInfo.qSeqSize);
            bool hasMaskCols = (info.s2Tile < mWidth);
            uint32_t maskNDeal = hasMaskCols ? Min(nDeal, mWidth - info.s2Tile) : 0U;
            uint32_t maskNDealAligned = (maskNDeal + ONE_BLOCK_SIZE - 1) / ONE_BLOCK_SIZE * ONE_BLOCK_SIZE;
            for (uint32_t r = 0; r < mDeal; ++r) {
                Duplicate<uint8_t>(maskUb[r * constInfo.nBaseUb], 0, static_cast<int32_t>(nDeal));
            }
            if (hasMaskCols) {
                SetFlag<HardEvent::V_MTE2>(ARCH31_EVENT_ID_MASK);
                WaitFlag<HardEvent::V_MTE2>(ARCH31_EVENT_ID_MASK);
                DataCopyExtParams extParams;
                extParams.blockCount = static_cast<uint16_t>(mDeal);
                extParams.blockLen = maskNDeal;
                extParams.srcStride = mWidth - maskNDeal;
                extParams.dstStride = (constInfo.nBaseUb - maskNDealAligned) / ONE_BLOCK_SIZE;
                DataCopyPadExtParams<uint8_t> padExtParams(true, 0, 0, 0);
                DataCopyPad(maskUb, attenMaskGm[info.attenMaskGmOffset], extParams, padExtParams);
            }
            SetFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_MASK);

            if (hasMaskCols && maskNDealAligned > maskNDeal) {
                WaitFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_MASK);
                uint32_t tailStart = maskNDeal & ~(ONE_BLOCK_SIZE - 1);
                uint32_t tailData = maskNDeal - tailStart;
                __ubuf__ uint8_t* maskBase = GetUbAddr<uint8_t>(attenMaskUb);
                for (uint32_t r = 0; r < mDeal; ++r) {
                    __ubuf__ uint8_t* rowAddr = maskBase + r * constInfo.nBaseUb + tailStart;
                    uint32_t realLanes = tailData;
                    uint32_t fullLanes = ONE_BLOCK_SIZE;
                    __VEC_SCOPE__
                    {
                        MicroAPI::MaskReg predKeep = MicroAPI::UpdateMask<uint8_t>(realLanes);
                        MicroAPI::MaskReg predFull = MicroAPI::UpdateMask<uint8_t>(fullLanes);
                        MicroAPI::RegTensor<uint8_t> valReg, zeroReg;
                        MicroAPI::Duplicate(zeroReg, 0);
                        MicroAPI::DataCopy(valReg, rowAddr);
                        MicroAPI::Select<uint8_t>(valReg, valReg, zeroReg, predKeep);
                        MicroAPI::DataCopy(rowAddr, valReg, predFull);
                    }
                }
                SetFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_MASK);
            }
            return;
        }
        if (nDeal % ONE_BLOCK_SIZE != 0) {
            uint32_t maskNDealAligned = (nDeal + ONE_BLOCK_SIZE - 1) / ONE_BLOCK_SIZE * ONE_BLOCK_SIZE;
            DataCopyExtParams extParams;
            extParams.blockCount = static_cast<uint16_t>(mRows);
            extParams.blockLen = nDeal;
            extParams.srcStride = static_cast<uint32_t>(constInfo.kvSeqSize) - nDeal;
            extParams.dstStride = (constInfo.nBaseUb - maskNDealAligned) / ONE_BLOCK_SIZE;
            DataCopyPadExtParams<uint8_t> padExtParams(true, 0, 0, 0);
            DataCopyPad(maskUb, attenMaskGm[info.attenMaskGmOffset], extParams, padExtParams);
            SetFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_MASK);
            return;
        }
        DataCopyParams maskCopyParams;
        maskCopyParams.blockCount = mRows;
        maskCopyParams.blockLen = nDeal / ONE_BLOCK_SIZE;
        maskCopyParams.srcStride = (static_cast<uint32_t>(constInfo.kvSeqSize) - nDeal) / ONE_BLOCK_SIZE;
        maskCopyParams.dstStride = (constInfo.nBaseUb - nDeal) / ONE_BLOCK_SIZE;
        DataCopy(maskUb, attenMaskGm[info.attenMaskGmOffset], maskCopyParams);
        SetFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_MASK);
    }

    __aicore__ inline void ComputeVec1OnlineSoftmax(const RunInfoArch31& info)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t nDeal = info.nDeal;
        uint32_t nDealEff = Arch31NDealPad(nDeal);
        uint32_t total = mDeal * nDealEff;

        __ubuf__ T* mm1Addr = GetUbAddr<T>(s2ScratchUb);
        __ubuf__ half* mm1HalfAddr = GetUbAddr<half>(pOutFp16Ub);
        __ubuf__ T* expAddr = mm1Addr;
        __ubuf__ T* rowMaxAddr = GetUbAddr<T>(rowMaxAccUb);
        __ubuf__ T* rowMaxPrevAddr = GetUbAddr<T>(rowMaxPrevUb);
        __ubuf__ T* rowSumAddr = GetUbAddr<T>(rowSumAccUb);
        __ubuf__ half* pFp16Addr = GetUbAddr<half>(pOutFp16Ub);
        __ubuf__ uint8_t* maskAddr = nullptr;
        if (constInfo.attenMaskFlag || constInfo.attenMaskTensorFlag) {
            maskAddr = GetUbAddr<uint8_t>(attenMaskUb);
        }
        if (constInfo.attenMaskTensorFlag && attenMaskPtr != nullptr) {
            WaitFlag<HardEvent::MTE2_V>(ARCH31_EVENT_ID_MASK);
            if (constInfo.maskMDeal != 0 && info.mDeal > constInfo.maskMDeal) {
                BroadcastMaskRows(constInfo.maskMDeal, info.mDeal, nDeal);
            }
        }
        __VEC_SCOPE__
        {
            uint32_t remaining = mDeal;
            uint16_t loopNum = static_cast<uint16_t>(CeilDiv(mDeal, perVegNum));
            for (uint16_t i = 0; i < loopNum; i++) {
                MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(remaining);
                MicroAPI::RegTensor<T> prevReg;
                MicroAPI::DataCopy(prevReg, rowMaxAddr + i * perVegNum);
                MicroAPI::DataCopy(rowMaxPrevAddr + i * perVegNum, prevReg, preg);
            }
        }

        __VEC_SCOPE__
        {
            uint32_t remaining = total;
            uint16_t loopNum = static_cast<uint16_t>(CeilDiv(total, perVegNum));
            for (uint16_t i = 0; i < loopNum; i++) {
                MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(remaining);
                MicroAPI::RegTensor<T> srcReg, dstReg;
                MicroAPI::RegTensor<half> srcHalfReg;
                MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcHalfReg, mm1HalfAddr + i * perVegNum);
                MicroAPI::Cast<T, half, castF16F32>(srcReg, srcHalfReg, preg);
                MicroAPI::Muls<T, T>(dstReg, srcReg, constInfo.scaleValue, preg);
                if (constInfo.attenMaskFlag || constInfo.attenMaskTensorFlag) {
                    ApplyMask(dstReg, dstReg, maskAddr, info, i, preg, nDealEff);
                }
                MicroAPI::DataCopy(mm1Addr + i * perVegNum, dstReg, preg);
            }
        }
        if (constInfo.attenMaskTensorFlag && attenMaskPtr != nullptr) {
            SetFlag<HardEvent::V_MTE2>(ARCH31_EVENT_ID_MASK);
            if (info.isGlobalLastTile) {
                WaitFlag<HardEvent::V_MTE2>(ARCH31_EVENT_ID_MASK);
            }
        }

        if (nDealEff != nDeal) {
            uint32_t winBaseCol = nDeal & ~7U;
            uint32_t realInWin = nDeal - winBaseCol;
            __VEC_SCOPE__
            {
                MicroAPI::RegTensor<T> negInfReg, winReg;
                MicroAPI::Duplicate<T>(negInfReg, FLOAT_INF_NEG);
                for (uint16_t r = 0; r < static_cast<uint16_t>(mDeal); r++) {
                    uint32_t winLanes = 8U;
                    MicroAPI::MaskReg pregWin = MicroAPI::UpdateMask<T>(winLanes);
                    if (realInWin > 0U) {
                        MicroAPI::DataCopy(winReg, mm1Addr + r * nDealEff + winBaseCol);
                        uint32_t realLanes = realInWin;
                        MicroAPI::MaskReg pregReal = MicroAPI::UpdateMask<T>(realLanes);
                        MicroAPI::Select<T>(winReg, winReg, negInfReg, pregReal);
                        MicroAPI::DataCopy(mm1Addr + r * nDealEff + winBaseCol, winReg, pregWin);
                    } else {
                        MicroAPI::DataCopy(mm1Addr + r * nDealEff + winBaseCol, negInfReg, pregWin);
                    }
                    for (uint16_t w = static_cast<uint16_t>(winBaseCol + 8U); w < static_cast<uint16_t>(nDealEff);
                         w += 8U) {
                        MicroAPI::DataCopy(mm1Addr + r * nDealEff + w, negInfReg, pregWin);
                    }
                }
            }
        }

        RowReduceAccumulate<ReduceMaxOp>(rowMaxAddr, mm1Addr, mDeal, nDealEff);
        __VEC_SCOPE__
        {
            uint16_t mRows = static_cast<uint16_t>(mDeal);
            uint16_t perRowLoopNum = static_cast<uint16_t>(CeilDiv(nDealEff, perVegNum));
            for (uint16_t rowIdx = 0; rowIdx < mRows; rowIdx++) {
                uint32_t remaining = nDealEff;
                MicroAPI::RegTensor<T> srcReg, maxReg, expReg;
                BroadcastRowScalar(maxReg, rowMaxAddr, rowIdx, mDeal, nDealEff);
                for (uint16_t j = 0; j < perRowLoopNum; j++) {
                    MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(remaining);
                    MicroAPI::DataCopy(srcReg, mm1Addr + rowIdx * nDealEff + j * perVegNum);
                    MicroAPI::FusedExpSub<T>(expReg, srcReg, maxReg, preg);
                    MicroAPI::DataCopy(expAddr + rowIdx * nDealEff + j * perVegNum, expReg, preg);
                }
            }
        }

        if (!info.isFirstS2Loop) {
            __VEC_SCOPE__
            {
                uint32_t remaining = mDeal;
                uint16_t loopNum = static_cast<uint16_t>(CeilDiv(mDeal, perVegNum));
                for (uint16_t i = 0; i < loopNum; i++) {
                    MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(remaining);
                    MicroAPI::RegTensor<T> prevReg, curReg, diffReg, scaleReg, sumReg;
                    MicroAPI::DataCopy(prevReg, rowMaxPrevAddr + i * perVegNum);
                    MicroAPI::DataCopy(curReg, rowMaxAddr + i * perVegNum);
                    MicroAPI::Sub<T>(diffReg, prevReg, curReg, preg);
                    MicroAPI::Exp<T>(scaleReg, diffReg, preg);
                    MicroAPI::DataCopy(sumReg, rowSumAddr + i * perVegNum);
                    MicroAPI::Mul<T>(sumReg, sumReg, scaleReg, preg);
                    MicroAPI::DataCopy(rowSumAddr + i * perVegNum, sumReg, preg);
                }
            }
        }

        RowReduceAccumulate<ReduceSumOp>(rowSumAddr, expAddr, mDeal, nDealEff);

        __VEC_SCOPE__
        {
            uint32_t remaining = total;
            uint16_t loopNum = static_cast<uint16_t>(CeilDiv(total, perVegNum));
            for (uint16_t i = 0; i < loopNum; i++) {
                MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(remaining);
                MicroAPI::RegTensor<T> pReg;
                MicroAPI::RegTensor<half> pFp16Reg;
                MicroAPI::DataCopy(pReg, expAddr + i * perVegNum);
                MicroAPI::Cast<half, T, castF32F16>(pFp16Reg, pReg, preg);
                MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(pFp16Addr + i * perVegNum, pFp16Reg, preg);
            }
        }
    }

    __aicore__ inline void AccumulateO(const RunInfoArch31& info)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t dInner = info.dInner;
        uint32_t total = mDeal * dInner;

        __ubuf__ T* oAccAddr = GetUbAddr<T>(O_accUb);
        __ubuf__ half* mm2HalfAddr = GetUbAddr<half>(pOutFp16Ub);
        __ubuf__ T* rowMaxPrevAddr = GetUbAddr<T>(rowMaxPrevUb);
        __ubuf__ T* rowMaxAddr = GetUbAddr<T>(rowMaxAccUb);

        __VEC_SCOPE__
        {
            uint32_t remaining = total;
            uint32_t perRowLoop = CeilDiv(dInner, perVegNum);
            uint16_t loopNum = static_cast<uint16_t>(CeilDiv(total, perVegNum));
            for (uint16_t i = 0; i < loopNum; i++) {
                MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(remaining);
                MicroAPI::RegTensor<T> oReg, mm2Reg, prevReg, curReg, diffReg, scaleReg;
                MicroAPI::RegTensor<half> mm2HalfReg;
                MicroAPI::DataCopy(oReg, oAccAddr + i * perVegNum);
                MicroAPI::DataCopy<half, MicroAPI::LoadDist::DIST_UNPACK_B16>(mm2HalfReg, mm2HalfAddr + i * perVegNum);
                MicroAPI::Cast<T, half, castF16F32>(mm2Reg, mm2HalfReg, preg);
                if (!info.isFirstS2Loop) {
                    uint32_t rowIdx = i / perRowLoop;
                    BroadcastRowScalar(prevReg, rowMaxPrevAddr, rowIdx, mDeal, dInner);
                    BroadcastRowScalar(curReg, rowMaxAddr, rowIdx, mDeal, dInner);
                    MicroAPI::Sub<T>(diffReg, prevReg, curReg, preg);
                    MicroAPI::Exp<T>(scaleReg, diffReg, preg);
                    MicroAPI::Mul<T>(oReg, oReg, scaleReg, preg);
                }
                MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING>(oReg, oReg, mm2Reg, preg);
                MicroAPI::DataCopy(oAccAddr + i * perVegNum, oReg, preg);
            }
        }
    }

    __aicore__ inline void FinalizeAndCopyOut(const RunInfoArch31& info)
    {
        uint32_t mDeal = info.mDeal;
        uint32_t dInner = info.dInner;
        uint32_t total = mDeal * dInner;

        __ubuf__ T* oAccAddr = GetUbAddr<T>(O_accUb);
        __ubuf__ T* rowSumAddr = GetUbAddr<T>(rowSumAccUb);
        __ubuf__ T* rowMaxAddr = GetUbAddr<T>(rowMaxAccUb);
        __ubuf__ half* outAddr = GetUbAddr<half>(pOutFp16Ub);

        __VEC_SCOPE__
        {
            uint32_t remaining = total;
            uint32_t perRowLoop = CeilDiv(dInner, perVegNum);
            uint16_t loopNum = static_cast<uint16_t>(CeilDiv(total, perVegNum));
            for (uint16_t i = 0; i < loopNum; i++) {
                MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(remaining);
                MicroAPI::RegTensor<T> oReg, sumReg, curReg, divReg;
                MicroAPI::RegTensor<half> outReg;
                MicroAPI::DataCopy(oReg, oAccAddr + i * perVegNum);
                uint32_t rowIdx = i / perRowLoop;
                BroadcastRowScalar(sumReg, rowSumAddr, rowIdx, mDeal, dInner);
                BroadcastRowScalar(curReg, rowMaxAddr, rowIdx, mDeal, dInner);
                MicroAPI::MaskReg zeroMask;
                MicroAPI::CompareScalar<T, CMPMODE::EQ>(zeroMask, curReg, static_cast<T>(FLOAT_INF_NEG), preg);
                MicroAPI::RegTensor<T> zeroReg;
                MicroAPI::Duplicate<T>(zeroReg, static_cast<T>(0.0f));
                MicroAPI::Div<T>(divReg, oReg, sumReg, preg);
                MicroAPI::Select<T>(divReg, zeroReg, divReg, zeroMask);
                MicroAPI::Cast<half, T, castF32F16>(outReg, divReg, preg);
                MicroAPI::DataCopy<half, MicroAPI::StoreDist::DIST_PACK_B32>(outAddr + i * perVegNum, outReg, preg);
            }
        }
    }

protected:
    __aicore__ inline void BroadcastRowScalar(MicroAPI::RegTensor<T>& maxReg, __ubuf__ T* rowMaxAddr, uint32_t rowIdx,
                                              uint32_t mDeal, uint32_t nDeal)
    {
        (void)mDeal;
        (void)nDeal;
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(maxReg, rowMaxAddr + rowIdx);
    }

    template <typename ReduceOp>
    __aicore__ inline void RowReduceAccumulate(__ubuf__ T* accAddr, __ubuf__ T* srcAddr, uint32_t mDeal, uint32_t nDeal)
    {
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> tmpReg, accReg, scalarReg;
            MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<T>();
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();
            uint16_t perRowLoopNum = static_cast<uint16_t>(CeilDiv(nDeal, perVegNum));
            uint16_t mRows = static_cast<uint16_t>(mDeal);
            for (uint16_t rowIdx = 0; rowIdx < mRows; rowIdx++) {
                ReduceOp::template Init<T>(tmpReg);
                uint32_t remaining = nDeal;
                for (uint16_t j = 0; j < perRowLoopNum; j++) {
                    MicroAPI::MaskReg preg = MicroAPI::UpdateMask<T>(remaining);
                    MicroAPI::RegTensor<T> srcReg;
                    MicroAPI::DataCopy(srcReg, srcAddr + rowIdx * nDeal + j * perVegNum);
                    ReduceOp::template Apply<T, MicroAPI::MaskMergeMode::MERGING>(tmpReg, tmpReg, srcReg, preg);
                }
                ReduceOp::template Reduce<T, MicroAPI::MaskMergeMode::ZEROING>(scalarReg, tmpReg, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(accReg, accAddr + rowIdx);
                ReduceOp::template Apply<T, MicroAPI::MaskMergeMode::MERGING>(accReg, accReg, scalarReg, pregAll);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(accAddr + rowIdx, accReg, pregOne);
            }
        }
    }

    __aicore__ inline void ApplyMask(MicroAPI::RegTensor<T>& dstReg, MicroAPI::RegTensor<T>& srcReg,
                                     __ubuf__ uint8_t* maskAddr, const RunInfoArch31& info, uint16_t i,
                                     MicroAPI::MaskReg& dataPreg, uint32_t nDealDomain)
    {
        MicroAPI::MaskReg maskPreg = BuildMaskPreg(constInfo.sparseMode, maskAddr, info, i, nDealDomain);
        MicroAPI::RegTensor<T> negInfReg;
        MicroAPI::Duplicate<T>(negInfReg, FLOAT_INF_NEG);
        MicroAPI::Select<T>(dstReg, negInfReg, srcReg, maskPreg);
        (void)dataPreg;
    }

    __aicore__ inline MicroAPI::MaskReg BuildMaskPreg(uint32_t sparseMode, __ubuf__ uint8_t* maskAddr,
                                                      const RunInfoArch31& info, uint16_t i, uint32_t nDealDomain)
    {
        MicroAPI::MaskReg maskPreg;
        if (constInfo.attenMaskTensorFlag) {
            uint32_t nDeal = nDealDomain;
            if (sparseMode == 9 && nDeal < perVegNum) {
                uint32_t winStart = static_cast<uint32_t>(i) * perVegNum;
                uint16_t rFirst = static_cast<uint16_t>(winStart / nDeal);
                uint16_t rLast = static_cast<uint16_t>(
                    Min((winStart + perVegNum - 1) / nDeal, static_cast<uint32_t>(info.mDeal - 1)));
                MicroAPI::MaskReg accPreg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALLF>();
                MicroAPI::MaskReg zeroPreg = MicroAPI::CreateMask<uint8_t, MicroAPI::MaskPattern::ALLF>();
                MicroAPI::MaskReg pregAllB8 = MicroAPI::CreateMask<uint8_t>();
                for (uint16_t r = rFirst; r <= rLast; r++) {
                    int32_t off =
                        static_cast<int32_t>(static_cast<uint32_t>(r) * nDeal) - static_cast<int32_t>(winStart);
                    __ubuf__ uint8_t* rowAddr = maskAddr + static_cast<uint32_t>(r) * constInfo.nBaseUb;
                    __ubuf__ uint8_t* fragB = rowAddr - off;
                    uint32_t relB = static_cast<uint32_t>(fragB - maskAddr);
                    uint32_t relA = relB & ~static_cast<uint32_t>(31);
                    __ubuf__ uint8_t* loadA = maskAddr + relA;
                    int16_t slide = static_cast<int16_t>(relB - relA);
                    MicroAPI::RegTensor<uint8_t> u8MaskReg;
                    MicroAPI::DataCopy(u8MaskReg, loadA);
                    MicroAPI::MaskReg pregX;
                    MicroAPI::CompareScalar<uint8_t, CMPMODE::NE>(pregX, u8MaskReg, 0, pregAllB8);
                    MicroAPI::MaskReg slidPreg;
                    FiaMaskSlideB8Arch31(slidPreg, pregX, zeroPreg, slide);
                    MicroAPI::MaskOr(accPreg, accPreg, slidPreg, pregAllB8);
                }
                MicroAPI::MaskReg unpackPreg;
                MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(unpackPreg, accPreg);
                MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(maskPreg, unpackPreg);
                return maskPreg;
            }

            uint32_t perRowLoop = CeilDiv(nDeal, perVegNum);
            uint32_t s1Local = i / perRowLoop;
            uint32_t chunkInRow = i % perRowLoop;
            __ubuf__ uint8_t* chunkAddr = maskAddr + s1Local * constInfo.nBaseUb + chunkInRow * perVegNum;
            MicroAPI::RegTensor<uint8_t> u8MaskReg;
            MicroAPI::DataCopy(u8MaskReg, chunkAddr);
            MicroAPI::MaskReg pregAllB8 = MicroAPI::CreateMask<uint8_t>();
            MicroAPI::CompareScalar<uint8_t, CMPMODE::NE>(maskPreg, u8MaskReg, 0, pregAllB8);
            MicroAPI::MaskReg unpackPreg;
            MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(unpackPreg, maskPreg);
            MicroAPI::MaskUnPack<MicroAPI::HighLowPart::LOWEST>(maskPreg, unpackPreg);
            return maskPreg;
        }
        (void)maskAddr;
        if (sparseMode == 0) {
            maskPreg = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
            return maskPreg;
        }

        uint32_t nDeal = nDealDomain;
        uint32_t perRowLoop = CeilDiv(nDeal, perVegNum);
        uint32_t s1Local = i / perRowLoop;
        uint32_t chunkInRow = i % perRowLoop;
        uint32_t s1Idx = info.s1Tile + s1Local;
        uint32_t s2Start = info.s2Tile + chunkInRow * perVegNum;

        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<int32_t>();
        if (sparseMode == 2) {
            MicroAPI::RegTensor<int32_t> s1IdxReg, s2IdxReg;
            MicroAPI::Duplicate<int32_t>(s1IdxReg, static_cast<int32_t>(s1Idx));
            MicroAPI::Arange<int32_t>(s2IdxReg, static_cast<int32_t>(s2Start));
            MicroAPI::Compare<int32_t, CMPMODE::GT>(maskPreg, s2IdxReg, s1IdxReg, pregAll);
            return maskPreg;
        } else if (sparseMode == 3) {
            MicroAPI::RegTensor<int32_t> s1OffReg, s2IdxReg;
            MicroAPI::Duplicate<int32_t>(s1OffReg, static_cast<int32_t>(s1Idx) + constInfo.sparseOffset);
            MicroAPI::Arange<int32_t>(s2IdxReg, static_cast<int32_t>(s2Start));
            MicroAPI::Compare<int32_t, CMPMODE::GT>(maskPreg, s2IdxReg, s1OffReg, pregAll);
            return maskPreg;
        } else if (sparseMode == 4) {
            MicroAPI::RegTensor<int32_t> s1IdxReg, s2IdxReg, upperReg, lowerReg;
            MicroAPI::Duplicate<int32_t>(s1IdxReg, static_cast<int32_t>(s1Idx));
            MicroAPI::Arange<int32_t>(s2IdxReg, static_cast<int32_t>(s2Start));
            MicroAPI::Adds<int32_t, int32_t>(upperReg, s1IdxReg, constInfo.bandUpperDelta, pregAll);
            MicroAPI::Adds<int32_t, int32_t>(lowerReg, s1IdxReg, constInfo.bandLowerDelta, pregAll);
            MicroAPI::MaskReg upperMask, lowerMask;
            MicroAPI::Compare<int32_t, CMPMODE::GT>(upperMask, s2IdxReg, upperReg, pregAll);
            MicroAPI::Compare<int32_t, CMPMODE::LT>(lowerMask, s2IdxReg, lowerReg, pregAll);
            MicroAPI::MaskOr(maskPreg, upperMask, lowerMask, pregAll);
            return maskPreg;
        }
        maskPreg = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        return maskPreg;
    }

protected:
    GlobalTensor<half> mm1ResGm;
    GlobalTensor<half> softmaxPFp16Gm;
    GlobalTensor<half> mm2ResGm;
    GlobalTensor<half> attentionOutGm;
    GlobalTensor<uint8_t> attenMaskGm;
    __gm__ uint8_t* attenMaskPtr = nullptr;

    TBuf<TPosition::VECIN> s2ScratchUb;
    TBuf<TPosition::VECOUT> pOutFp16Ub;
    TBuf<TPosition::VECCALC> rowMaxAccUb;
    TBuf<TPosition::VECCALC> rowMaxPrevUb;
    TBuf<TPosition::VECCALC> rowSumAccUb;
    TBuf<TPosition::VECCALC> O_accUb;
    TBuf<TPosition::VECIN> attenMaskUb;

    ConstInfoArch31 constInfo{};
};

} // namespace AttentionCommonArch31

#endif // FIA_BLOCK_VEC_NONQUANT_ARCH31_H
