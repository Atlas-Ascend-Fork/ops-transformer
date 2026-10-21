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
 * \file moe_v3_gather_hif4_quant.h
 * \brief
 */
#ifndef MOE_V3_GATHER_HIF4_QUANT_H_REGBASE
#define MOE_V3_GATHER_HIF4_QUANT_H_REGBASE

#include "moe_v3_common.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MoeInitRoutingV3 {
using namespace AscendC;
using AscendC::Reg::LoadDist;
using AscendC::Reg::StoreDist;
using AscendC::Reg::RegLayout;
using AscendC::Reg::SatMode;
using AscendC::Reg::MaskMergeMode;
using AscendC::Reg::MaskPattern;
using AscendC::Reg::PostLiteral;
using AscendC::Reg::MaskDist;

// hif4量化一个块的元素数量为64个，即1个scale对应64个x的元素
constexpr int64_t HIFP4_BLOCK_SIZE = 64;
constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM = 8; // 一次reg循环计算8个VL
constexpr int64_t HIFP4_MAXPER8_REG_ONELOOP = 16;
constexpr int64_t HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM = 256; // scale 输出的计算，一次reg循环计算256个group的scale
constexpr int64_t HIFP4_OUTPUT_REG_ONELOOP = 2;
constexpr uint32_t HIFP4_CAST_COPYOUT_PACK4_NUM = 4;
constexpr uint32_t HIFP4_BRC_BLOCK_NUM = 8;
constexpr int64_t HIFP4_MAXPER4_ALIGN = 128;
constexpr int64_t HIFP4_MAXPER8_ALIGN = 128;
constexpr int64_t HIFP4_SCALETMP_ALIGN = 256;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t DIGIT_FOUR = 4;
constexpr int64_t DIGIT_EIGHT = 8;

template <typename T, typename U>
class MoeGatherOutHif4Quant {
public:
    __aicore__ inline MoeGatherOutHif4Quant(){};
    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr, GM_ADDR expandedRowIdxAddr,
                                GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr, GM_ADDR sortedExpertIdxAddr,
                                const MoeInitRoutingV3Arch35TilingData* tilingData, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitKernelTiling(GM_ADDR sortedExpertIdxAddr,
                                            const MoeInitRoutingV3Arch35TilingData* tilingData);
    __aicore__ inline void CopyInExpandedExpertIdx(int64_t progress);
    __aicore__ inline void CopyExpandedXAndHif4Quant(int64_t progress);
    __aicore__ inline void CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols);
    __aicore__ inline void Compute(int64_t regLoop, int64_t maxPer8RegLoop, int64_t scaleRegLoop, int64_t outRegLoop);
    __aicore__ inline void ComputeMaxPer4(uint16_t regLoop, __ubuf__ float* xAddr1, __ubuf__ uint16_t* maxPer4BufAddr);
    __aicore__ inline void ComputeMaxPer8(uint16_t maxPer8RegLoop, __ubuf__ uint16_t* maxPer4BufReadAddr,
                                          __ubuf__ uint16_t* e8BufAddr, __ubuf__ uint16_t* maxPer8BufAddr);
    __aicore__ inline void ComputeTmpScale(uint16_t regLoop, __ubuf__ uint16_t* maxPer4BufRead2Addr,
                                           __ubuf__ uint16_t* maxPer8BufReadAddr, __ubuf__ uint16_t* e8BufRead1Addr,
                                           __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
                                           __ubuf__ uint16_t* tmpScaleAddr);
    __aicore__ inline void ComputeScale(uint16_t scaleRegLoop, __ubuf__ uint8_t* e8BufRead2Addr,
                                        __ubuf__ uint8_t* e1p8BufReadAddr, __ubuf__ uint8_t* e1p16BufReadAddr,
                                        __ubuf__ uint16_t* scaleAddr);
    __aicore__ inline void ComputeOut(uint16_t outRegLoop, __ubuf__ T* xAddr2, __ubuf__ uint16_t* tmpScaleReadAddr,
                                      __ubuf__ int32_t* yAddr);
    __aicore__ inline void CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols, int64_t loopScaleCols);

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, 1> xInQueue_;
    TQue<QuePosition::VECIN, 1> sortedRowIdxInQueue_;
    TQue<QuePosition::VECOUT, 1> xQuantOutQueue_;
    TQue<QuePosition::VECOUT, 1> mxScaleOutQueue_;
    TBuf<TPosition::VECCALC> bufE8_;
    TBuf<TPosition::VECCALC> buf1E8_;
    TBuf<TPosition::VECCALC> buf1E16_;
    TBuf<TPosition::VECCALC> bufTmpScale_;
    TBuf<TPosition::VECCALC> bufMaxPer4_;
    TBuf<TPosition::VECCALC> bufMaxPer8_;

    GlobalTensor<T> xInGm_;
    GlobalTensor<uint8_t> expandedXOutGm_;
    GlobalTensor<float> expandedScaleOutGm_;
    GlobalTensor<int32_t> sortedRowIdxGm_;
    GlobalTensor<int32_t> expertTotalCountGm_;

    const MoeV3Arch35GatherOutComputeTilingData* gatherOutTilingData_;

    int64_t needCoreNum_;
    int64_t blockIdx_;

    int64_t scaleCols_; // 一个token的scale有多少个元素（列）
    int64_t n_;
    int64_t k_;
    int64_t cols_;
    int64_t perCoreRow_;
    int64_t currentLoopRows_;
    int64_t coreRows_;
    int64_t perLoopRows_;
    int64_t lastLoopRows_;
    int64_t rowLoops_;
    int64_t perLoopCols_;
    int64_t lastLoopCols_;
    int64_t colLoops_;
    int64_t perLoopScaleCols_;
    int64_t lastLoopScaleCols_;
    int64_t indicesOffset_;
    int64_t rowIdxType_ = 0;
    int64_t groupNum_; // 64个x为一组，构成基本量化单元
    int64_t gmCopyInRegStep_ = 0;
    int64_t gmCopyOutRegStep_ = 0;
    int64_t gmCopyOutScaleRegStep_ = 0;
    int64_t gmCopyInRegStep2_ = 0;
    int64_t gmCopyTmpE1_ = 0;
    int64_t gmCopyTmpE8_ = 0;
    int64_t gmCopyTmpE16_ = 0;
    int64_t gmCopyTmpBufE8Step_ = 0;
    int64_t gmCopyTmpBufOtherStep_ = 0;
    int64_t bufE8Buffer_ = 0;
    int64_t buf1E8Buffer_ = 0;
    int64_t buf1E16Buffer_ = 0;
    int64_t maxPer4Buffer_ = 0;
    int64_t maxPer8Buffer_ = 0;
    int64_t tmpScaleBuffer_ = 0;

    // 一个RegTensor的长度Bytes，通常是256
    const uint32_t vRegSize_ = Ops::Base::GetVRegSize();
    // 一个RegTensor能塞下多少T(B32)类型的元素
    const int32_t vlForB32_ = vRegSize_ / sizeof(float);
    constexpr static Reg::CastTrait castTrait_ = {RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING,
                                                  RoundMode::CAST_RINT};
    constexpr static Reg::CastTrait castTraitReciprocalCast_ = {RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING,
                                                                RoundMode::CAST_RINT};
};

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::Init(GM_ADDR xAddr, GM_ADDR unused_ScaleAddr,
                                                         GM_ADDR sortedExpertIdxAddr, GM_ADDR expandedRowIdxAddr,
                                                         GM_ADDR expandedXAddr, GM_ADDR expandedScaleAddr,
                                                         const MoeInitRoutingV3Arch35TilingData* tilingData,
                                                         TPipe* tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    InitKernelTiling(sortedExpertIdxAddr, tilingData);

    xInGm_.SetGlobalBuffer((__gm__ T*)xAddr);
    expandedXOutGm_.SetGlobalBuffer((__gm__ uint8_t*)expandedXAddr);
    if (rowIdxType_ == SCATTER) {
        sortedRowIdxGm_.SetGlobalBuffer((__gm__ int32_t*)expandedRowIdxAddr + blockIdx_ * perCoreRow_,
                                        Align(perCoreRow_, sizeof(int32_t)));
    } else {
        sortedRowIdxGm_.SetGlobalBuffer(
            (__gm__ int32_t*)sortedExpertIdxAddr + Align(n_ * k_, sizeof(int32_t)) + blockIdx_ * perCoreRow_,
            Align(perCoreRow_, sizeof(int32_t)));
    }
    expandedScaleOutGm_.SetGlobalBuffer((__gm__ float*)expandedScaleAddr);

    pipe_->InitBuffer(sortedRowIdxInQueue_, 1, AlignBytes(perLoopRows_, sizeof(int32_t)));
    pipe_->InitBuffer(xInQueue_, 1, groupNum_ * HIFP4_BLOCK_SIZE * sizeof(T));
    pipe_->InitBuffer(xQuantOutQueue_, 1, groupNum_ * HIFP4_BLOCK_SIZE * sizeof(uint8_t) / 2); // hif4为4bit
    pipe_->InitBuffer(mxScaleOutQueue_, 1,
                      Ops::Base::CeilDiv<int64_t>(groupNum_, HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM) *
                          HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM * sizeof(float));

    bufE8Buffer_ = Ops::Base::CeilAlign(groupNum_, BLOCK_BYTES);
    buf1E8Buffer_ = Ops::Base::CeilAlign(groupNum_, BLOCK_BYTES);
    buf1E16Buffer_ = Ops::Base::CeilAlign(groupNum_, BLOCK_BYTES);
    maxPer4Buffer_ = Ops::Base::CeilAlign(groupNum_, static_cast<int64_t>(HIFP4_MAXPER4_ALIGN));
    maxPer8Buffer_ = Ops::Base::CeilAlign(groupNum_, static_cast<int64_t>(HIFP4_MAXPER8_ALIGN));
    tmpScaleBuffer_ = Ops::Base::CeilAlign(groupNum_, static_cast<int64_t>(HIFP4_SCALETMP_ALIGN));

    pipe_->InitBuffer(bufE8_, bufE8Buffer_ * sizeof(int16_t));
    pipe_->InitBuffer(buf1E8_, buf1E8Buffer_ * sizeof(int32_t));
    pipe_->InitBuffer(buf1E16_, buf1E16Buffer_ * sizeof(int16_t));
    pipe_->InitBuffer(bufMaxPer4_, maxPer4Buffer_ * 16 * sizeof(int16_t));
    pipe_->InitBuffer(bufMaxPer8_, maxPer8Buffer_ * 8 * sizeof(int16_t));
    pipe_->InitBuffer(bufTmpScale_, tmpScaleBuffer_ * 32 * sizeof(int16_t));

    gmCopyInRegStep_ = vRegSize_ / sizeof(float) * DIGIT_TWO; // 按照类型float，双搬入  //128
    gmCopyOutRegStep_ = vRegSize_ / sizeof(int32_t) /
                        HIFP4_CAST_COPYOUT_PACK4_NUM;   // 类型 uint32  DQ_HIFP4_CAST_COPYOUT_PACK4_NUM是4   //16
    gmCopyOutScaleRegStep_ = vRegSize_ / sizeof(float); // 类型 fp32//64
    gmCopyInRegStep2_ = vRegSize_ / sizeof(T);          // 按照类型float//128
    gmCopyTmpE1_ = vRegSize_ * DIGIT_FOUR / HIFP4_BLOCK_SIZE / DIGIT_TWO;
    gmCopyTmpE8_ = vRegSize_ * DIGIT_FOUR / DIGIT_EIGHT;
    gmCopyTmpE16_ = vRegSize_ * DIGIT_FOUR / DIGIT_FOUR;
    gmCopyTmpBufE8Step_ = vRegSize_ * DIGIT_TWO;
    gmCopyTmpBufOtherStep_ = vRegSize_;
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::InitKernelTiling(GM_ADDR sortedExpertIdxAddr,
                                                                     const MoeInitRoutingV3Arch35TilingData* tilingData)
{
    gatherOutTilingData_ = &(tilingData->gatherOutComputeParamsOp);
    cols_ = tilingData->cols;
    scaleCols_ = Ops::Base::CeilDiv<int64_t>(cols_, HIFP4_BLOCK_SIZE);
    n_ = tilingData->n;
    k_ = tilingData->k;
    rowIdxType_ = tilingData->rowIdxType;

    // core split
    int64_t actualExpertNum_ = tilingData->actualExpertNum;
    expertTotalCountGm_.SetGlobalBuffer((__gm__ int32_t*)sortedExpertIdxAddr + Align(n_ * k_, sizeof(int32_t)) * 2 +
                                            Align(actualExpertNum_, sizeof(int32_t)),
                                        actualExpertNum_);
    // 由于有效专家数可能较少，故实际上expertTotalCount_会小于n*k
    int64_t expertTotalCount_ = expertTotalCountGm_.GetValue(0);
    perCoreRow_ = Ceil(expertTotalCount_, tilingData->coreNum);
    needCoreNum_ = Ceil(expertTotalCount_, perCoreRow_);
    int64_t lastCoreIndicesElements = expertTotalCount_ - (needCoreNum_ - 1) * perCoreRow_;

    // inner core split
    int64_t originPerLoopElements;
    if (blockIdx_ == needCoreNum_ - 1) {
        coreRows_ = lastCoreIndicesElements;
        originPerLoopElements = gatherOutTilingData_->lastCorePerLoopIndicesElements;
    } else {
        coreRows_ = perCoreRow_;
        originPerLoopElements = gatherOutTilingData_->perCorePerLoopIndicesElements;
    }
    // 计算实际每轮loop搬入的rowIdx的行数
    perLoopRows_ = Min(coreRows_, originPerLoopElements);
    rowLoops_ = Ceil(coreRows_, perLoopRows_);
    lastLoopRows_ = coreRows_ - (rowLoops_ - 1) * perLoopRows_;

    // cols split
    // 计算x在h维度是否需要切分，如一行满载，则perLoopCols = cols
    perLoopCols_ = gatherOutTilingData_->perLoopCols;
    lastLoopCols_ = gatherOutTilingData_->lastLoopCols;
    colLoops_ = gatherOutTilingData_->colsLoops;
    perLoopScaleCols_ = perLoopCols_ / HIFP4_BLOCK_SIZE; // perLoopCols_在tiling侧计算，已经对齐到64的整数倍了
    lastLoopScaleCols_ = scaleCols_ - (colLoops_ - 1) * perLoopScaleCols_;

    groupNum_ = perLoopCols_ / HIFP4_BLOCK_SIZE;
}

// __ubuf__ 类型表示用于矢量计算的Local Memory（Unified Buffer），是LocalTensor实际存储的物理位置
template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::Compute(int64_t regLoop, int64_t maxPer8RegLoop,
                                                            int64_t scaleRegLoop, int64_t outRegLoop)
{
    LocalTensor<T> xLocal = xInQueue_.template DeQue<T>();
    LocalTensor<uint8_t> xQuantLocal = xQuantOutQueue_.template AllocTensor<uint8_t>();
    LocalTensor<float> scale = mxScaleOutQueue_.template AllocTensor<float>();

    LocalTensor<uint16_t> e8buf = bufE8_.Get<uint16_t>();
    LocalTensor<uint8_t> e1p8buf = buf1E8_.Get<uint8_t>();
    LocalTensor<uint8_t> e1p16buf = buf1E16_.Get<uint8_t>();
    LocalTensor<uint16_t> tmpscale = bufTmpScale_.Get<uint16_t>();
    LocalTensor<uint16_t> maxper4 = bufMaxPer4_.Get<uint16_t>();
    LocalTensor<uint16_t> maxper8 = bufMaxPer8_.Get<uint16_t>();

    auto xAddr1 = (__ubuf__ float*)xLocal.GetPhyAddr();
    auto xAddr2 = (__ubuf__ T*)xLocal.GetPhyAddr();
    auto yAddr = (__ubuf__ int32_t*)xQuantLocal.GetPhyAddr();
    auto scaleAddr = (__ubuf__ uint16_t*)scale.GetPhyAddr();

    auto e8BufAddr = (__ubuf__ uint16_t*)e8buf.GetPhyAddr();
    auto e8BufAddr2 = (__ubuf__ uint8_t*)e8buf.GetPhyAddr();
    auto e1p8BufAddr = (__ubuf__ uint32_t*)e1p8buf.GetPhyAddr();
    auto e1p16BufAddr = (__ubuf__ uint8_t*)e1p16buf.GetPhyAddr();
    auto tmpScaleAddr = (__ubuf__ uint16_t*)tmpscale.GetPhyAddr();
    auto maxPer4BufAddr = (__ubuf__ uint16_t*)maxper4.GetPhyAddr();
    auto maxPer8BufAddr = (__ubuf__ uint16_t*)maxper8.GetPhyAddr();

    auto e8BufRead1Addr = e8BufAddr;
    auto e8BufRead2Addr = e8BufAddr2;
    auto e1p8BufReadAddr = (__ubuf__ uint8_t*)e1p8buf.GetPhyAddr();
    auto e1p16BufReadAddr = e1p16BufAddr;
    auto tmpScaleReadAddr = tmpScaleAddr;
    auto maxPer4BufReadAddr = maxPer4BufAddr;
    auto maxPer4BufRead2Addr = maxPer4BufAddr;
    auto maxPer8BufReadAddr = maxPer8BufAddr;

    ComputeMaxPer4(static_cast<uint16_t>(regLoop), xAddr1, maxPer4BufAddr);
    ComputeMaxPer8(static_cast<uint16_t>(maxPer8RegLoop), maxPer4BufReadAddr, e8BufAddr, maxPer8BufAddr);
    ComputeTmpScale(static_cast<uint16_t>(regLoop), maxPer4BufRead2Addr, maxPer8BufReadAddr, e8BufRead1Addr,
                    e1p8BufAddr, e1p16BufAddr, tmpScaleAddr);
    ComputeOut(static_cast<uint16_t>(outRegLoop), xAddr2, tmpScaleReadAddr, yAddr);
    ComputeScale(static_cast<uint16_t>(scaleRegLoop), e8BufRead2Addr, e1p8BufReadAddr, e1p16BufReadAddr, scaleAddr);

    xQuantOutQueue_.template EnQue(xQuantLocal);
    mxScaleOutQueue_.template EnQue(scale);
    xInQueue_.template FreeTensor(xLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::ComputeMaxPer4(uint16_t regLoop, __ubuf__ float* xAddr1,
                                                                   __ubuf__ uint16_t* maxPer4BufAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregIn1;
        Reg::RegTensor<T> vregIn2;
        Reg::RegTensor<T> vregIn3;
        Reg::RegTensor<T> vregIn4;
        Reg::RegTensor<T> vregMax1;
        Reg::RegTensor<T> vregMax2;
        Reg::RegTensor<T> vregOut;
        Reg::RegTensor<bfloat16_t> vregOutbf16;
        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::Duplicate(expMaskBF16, 0x7fff);
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B32>(
                (Reg::RegTensor<float>&)vregIn1, (Reg::RegTensor<float>&)vregIn2, xAddr1, gmCopyInRegStep_);
            Reg::And((Reg::RegTensor<uint16_t>&)vregIn1, (Reg::RegTensor<uint16_t>&)vregIn1, expMaskBF16, maskAll);
            Reg::And((Reg::RegTensor<uint16_t>&)vregIn2, (Reg::RegTensor<uint16_t>&)vregIn2, expMaskBF16, maskAll);
            Reg::LoadAlign<float, Reg::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B32>(
                (Reg::RegTensor<float>&)vregIn3, (Reg::RegTensor<float>&)vregIn4, xAddr1, gmCopyInRegStep_);
            Reg::And((Reg::RegTensor<uint16_t>&)vregIn3, (Reg::RegTensor<uint16_t>&)vregIn3, expMaskBF16, maskAll);
            Reg::And((Reg::RegTensor<uint16_t>&)vregIn4, (Reg::RegTensor<uint16_t>&)vregIn4, expMaskBF16, maskAll);
            Reg::Max(vregMax1, vregIn1, vregIn2, maskAll);
            Reg::Max(vregMax2, vregIn3, vregIn4, maskAll);
            Reg::DeInterleave(vregMax1, vregMax2, vregMax1, vregMax2);
            Reg::Max(vregOut, vregMax1, vregMax2, maskAll);
            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxPer4BufAddr, (Reg::RegTensor<uint16_t>&)vregOut, gmCopyInRegStep2_, maskAll);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::ComputeMaxPer8(uint16_t maxPer8RegLoop,
                                                                   __ubuf__ uint16_t* maxPer4BufReadAddr,
                                                                   __ubuf__ uint16_t* e8BufAddr,
                                                                   __ubuf__ uint16_t* maxPer8BufAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> vreg0Maxper4;
        Reg::RegTensor<bfloat16_t> vreg1Maxper4;
        Reg::RegTensor<bfloat16_t> vreg0Maxper8;
        Reg::RegTensor<bfloat16_t> vreg0Maxper64;
        Reg::RegTensor<bfloat16_t> vreg1Maxper64;
        Reg::RegTensor<uint16_t> vreg0E6m2;
        Reg::RegTensor<uint16_t> vreg1E6m2;
        Reg::RegTensor<uint16_t> vregNumscale;
        Reg::Duplicate(vregNumscale, 0x3e12);
        Reg::UnalignRegForStore ureg1;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        static constexpr Reg::CastTrait castTraitE6M2 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                         Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        for (uint16_t loopIdx = 0; loopIdx < maxPer8RegLoop; loopIdx++) {
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DINTLV_B16>(
                (Reg::RegTensor<uint16_t>&)vreg0Maxper4, (Reg::RegTensor<uint16_t>&)vreg1Maxper4, maxPer4BufReadAddr,
                gmCopyInRegStep2_ * DIGIT_TWO);
            Reg::Max(vreg0Maxper8, vreg0Maxper4, vreg1Maxper4, maskAll);
            Reg::Interleave(vreg0Maxper4, vreg1Maxper4, vreg0Maxper4, vreg1Maxper4);

            Reg::ReduceDataBlock<Reg::ReduceType::MAX, uint16_t>((Reg::RegTensor<uint16_t>&)vreg0Maxper64,
                                                                 (Reg::RegTensor<uint16_t>&)vreg0Maxper4, maskAll);
            Reg::ReduceDataBlock<Reg::ReduceType::MAX, uint16_t>((Reg::RegTensor<uint16_t>&)vreg1Maxper64,
                                                                 (Reg::RegTensor<uint16_t>&)vreg1Maxper4, maskAll);

            Reg::Mul(vreg0Maxper64, vreg0Maxper64, (Reg::RegTensor<bfloat16_t>&)vregNumscale, maskAll);
            Reg::Mul(vreg1Maxper64, vreg1Maxper64, (Reg::RegTensor<bfloat16_t>&)vregNumscale, maskAll);

            Reg::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((Reg::RegTensor<fp8_e6m2_t>&)vreg0E6m2, vreg0Maxper64,
                                                             maskAll); // 只有前8位才是有效值
            Reg::Cast<fp8_e6m2_t, bfloat16_t, castTraitE6M2>((Reg::RegTensor<fp8_e6m2_t>&)vreg1E6m2, vreg1Maxper64,
                                                             maskAll); // 只有前8位才是有效值

            Reg::StoreAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                maxPer8BufAddr, (Reg::RegTensor<uint16_t>&)vreg0Maxper8, gmCopyInRegStep2_, maskAll);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(e8BufAddr, vreg0E6m2, ureg1, 8);
            Reg::StoreUnAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE>(e8BufAddr, vreg1E6m2, ureg1, 8);
        }
        Reg::StoreUnAlignPost(e8BufAddr, ureg1, 0);
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::ComputeTmpScale(
    uint16_t regLoop, __ubuf__ uint16_t* maxPer4BufRead2Addr, __ubuf__ uint16_t* maxPer8BufReadAddr,
    __ubuf__ uint16_t* e8BufRead1Addr, __ubuf__ uint32_t* e1p8BufAddr, __ubuf__ uint8_t* e1p16BufAddr,
    __ubuf__ uint16_t* tmpScaleAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<bfloat16_t> vreg0Maxper4;
        Reg::RegTensor<bfloat16_t> vreg0Maxper8;
        Reg::RegTensor<bfloat16_t> vreg0Maxper64;
        Reg::RegTensor<bfloat16_t> vb16_Ea_rec_bf16;
        Reg::RegTensor<bfloat16_t> vreg0E6m2;
        Reg::RegTensor<bfloat16_t> vregTmpScale;

        float number1 = 1.0;
        Reg::RegTensor<bfloat16_t> vregNum1;
        Reg::Duplicate(vregNum1, number1);
        float numberhalf = 0.5;
        Reg::RegTensor<bfloat16_t> vregNumhalf;
        Reg::Duplicate(vregNumhalf, static_cast<bfloat16_t>(numberhalf));

        Reg::MaskReg vb1E8Comp;
        Reg::MaskReg vb1E8CompTmp;
        Reg::MaskReg vb1E16Comp;
        Reg::MaskReg vb1E16CompTmp;
        Reg::UnalignRegForStore ureg1;

        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < regLoop; loopIdx++) {
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_E2B_B16>(
                (Reg::RegTensor<uint16_t>&)vreg0Maxper64, e8BufRead1Addr, 8);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_US_B16>(
                (Reg::RegTensor<uint16_t>&)vreg0Maxper8, maxPer8BufReadAddr, 64);
            Reg::LoadAlign<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(
                (Reg::RegTensor<uint16_t>&)vreg0Maxper4, maxPer4BufRead2Addr, 128);

            Reg::ReciprocalCast<bfloat16_t, fp8_e6m2_t, castTraitReciprocalCast_>(
                (Reg::RegTensor<bfloat16_t>&)vb16_Ea_rec_bf16, (Reg::RegTensor<fp8_e6m2_t>&)vreg0Maxper64, maskAll);

            Reg::Mul(vreg0Maxper8, vreg0Maxper8, vb16_Ea_rec_bf16, maskAll);
            Reg::Compares<bfloat16_t, CMPMODE::GE>(vb1E8Comp, vreg0Maxper8, 4, maskAll);
            Reg::Select(vreg0Maxper8, vregNumhalf, vregNum1, vb1E8Comp);
            Reg::Mul(vreg0Maxper4, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);
            Reg::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            Reg::Compares<bfloat16_t, CMPMODE::GE>(vb1E16Comp, vreg0Maxper4, 2, maskAll);
            Reg::Select(vreg0Maxper4, vregNumhalf, vregNum1, vb1E16Comp);
            Reg::Mul(vreg0Maxper4, vreg0Maxper4, vreg0Maxper8, maskAll);
            Reg::Mul(vregTmpScale, vreg0Maxper4, vb16_Ea_rec_bf16, maskAll);

            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(tmpScaleAddr + loopIdx * gmCopyTmpBufOtherStep_,
                                                                      (Reg::RegTensor<uint16_t>&)vregTmpScale,
                                                                      (Reg::RegTensor<uint16_t>&)vregTmpScale, maskAll);

            Reg::StoreUnAlign<uint32_t>(e1p8BufAddr, vb1E8Comp, ureg1);

            Reg::StoreAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, MaskDist::DIST_PACK>(e1p16BufAddr, vb1E16Comp, 16);
        }
        Reg::StoreUnAlignPost<uint32_t, Reg::PostLiteral::POST_MODE_UPDATE>(e1p8BufAddr, ureg1, 0);
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::ComputeScale(uint16_t scaleRegLoop,
                                                                 __ubuf__ uint8_t* e8BufRead2Addr,
                                                                 __ubuf__ uint8_t* e1p8BufReadAddr,
                                                                 __ubuf__ uint8_t* e1p16BufReadAddr,
                                                                 __ubuf__ uint16_t* scaleAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint8_t> vregE8;
        Reg::RegTensor<uint8_t> vreg1E8;
        Reg::RegTensor<uint8_t> vreg1E16First;
        Reg::RegTensor<uint8_t> vreg1E16Second;

        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < scaleRegLoop; loopIdx++) {
            Reg::LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_DS_B8>(vregE8, e8BufRead2Addr,
                                                                                         gmCopyTmpBufE8Step_);
            Reg::LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(vreg1E8, e1p8BufReadAddr,
                                                                                        gmCopyTmpBufOtherStep_);
            Reg::LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(vreg1E16First, e1p16BufReadAddr,
                                                                                        gmCopyTmpBufOtherStep_);
            Reg::LoadAlign<uint8_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(
                vreg1E16Second, e1p16BufReadAddr, gmCopyTmpBufOtherStep_);

            Reg::Interleave<uint8_t>(vregE8, vreg1E8, vregE8, vreg1E8);
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(
                scaleAddr + loopIdx * gmCopyTmpBufE8Step_, (Reg::RegTensor<uint16_t>&)vregE8,
                (Reg::RegTensor<uint16_t>&)vreg1E16First, maskAll);
            Reg::StoreAlign<uint16_t, Reg::StoreDist::DIST_INTLV_B16>(
                scaleAddr + loopIdx * gmCopyTmpBufE8Step_ + gmCopyTmpBufOtherStep_, (Reg::RegTensor<uint16_t>&)vreg1E8,
                (Reg::RegTensor<uint16_t>&)vreg1E16Second, maskAll);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::ComputeOut(uint16_t outRegLoop, __ubuf__ T* xAddr2,
                                                               __ubuf__ uint16_t* tmpScaleReadAddr,
                                                               __ubuf__ int32_t* yAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> vregOut;
        Reg::RegTensor<uint16_t> vregScaleTmp;
        Reg::RegTensor<fp4x2_e1m2_t> vbOut;
        Reg::RegTensor<bfloat16_t> vregOutbf16;
        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t>();
        for (uint16_t loopIdx = 0; loopIdx < outRegLoop; loopIdx++) {
            Reg::LoadAlign<T, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_NORM>(vregOutbf16, xAddr2,
                                                                                  gmCopyInRegStep2_);
            Reg::LoadAlign<uint16_t, PostLiteral::POST_MODE_UPDATE, LoadDist::DIST_US_B16>(
                vregScaleTmp, tmpScaleReadAddr, gmCopyOutScaleRegStep_);
            Reg::Mul((Reg::RegTensor<bfloat16_t>&)vregOutbf16, (Reg::RegTensor<bfloat16_t>&)vregOutbf16,
                     (Reg::RegTensor<bfloat16_t>&)vregScaleTmp, maskAll);
            Reg::Cast<fp4x2_e1m2_t, bfloat16_t, castTrait_>(vbOut, vregOutbf16, maskAll);
            Reg::StoreAlign<int32_t, Reg::PostLiteral::POST_MODE_UPDATE, StoreDist::DIST_PACK4_B32>(
                yAddr, (Reg::RegTensor<int32_t>&)vbOut, gmCopyOutRegStep_, maskAll);
        }
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::Process()
{
    if (blockIdx_ < needCoreNum_) {
        currentLoopRows_ = perLoopRows_;
        for (int64_t loop = 0; loop < rowLoops_ - 1; loop++) {
            CopyInExpandedExpertIdx(loop);
            CopyExpandedXAndHif4Quant(loop);
        }

        currentLoopRows_ = lastLoopRows_;
        CopyInExpandedExpertIdx(rowLoops_ - 1);
        CopyExpandedXAndHif4Quant(rowLoops_ - 1);
    }
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::CopyInExpandedExpertIdx(int64_t progress)
{
    indicesOffset_ = progress * perLoopRows_;
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.AllocTensor<int32_t>();
    DataCopyExtParams dataCopyParams{1, static_cast<uint32_t>(currentLoopRows_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(indicesLocal, sortedRowIdxGm_[indicesOffset_], dataCopyParams, dataCopyPadParams);
    sortedRowIdxInQueue_.EnQue<int32_t>(indicesLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::CopyExpandedXAndHif4Quant(int64_t progress)
{
    LocalTensor<int32_t> indicesLocal = sortedRowIdxInQueue_.DeQue<int32_t>();
    SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
    for (int64_t index = 0; index < currentLoopRows_; index++) {
        int64_t srcIdx = static_cast<int64_t>(indicesLocal.GetValue(index));
        int64_t dstIdx = perCoreRow_ * blockIdx_ + perLoopRows_ * progress + index;
        for (int64_t j = 0; j < colLoops_; j++) {
            // 每行切分成cols，按cols读入-计算量化-拷出
            int64_t loopCols = (j == colLoops_ - 1) ? lastLoopCols_ : perLoopCols_;
            int64_t loopScaleCols = (j == colLoops_ - 1) ? lastLoopScaleCols_ : perLoopScaleCols_;
            ;
            CopyIn(srcIdx / k_, j, loopCols);
            Compute(Ops::Base::CeilDiv<int64_t>(groupNum_, HIFP4_OUTPUT_REG_ONELOOP_GROUP_NUM),
                    Ops::Base::CeilDiv<int64_t>(groupNum_, HIFP4_MAXPER8_REG_ONELOOP),
                    Ops::Base::CeilDiv<int64_t>(groupNum_, HIFP4_OUTPUT_SCALE_REG_ONELOOP_GROUP_NUM),
                    Ops::Base::CeilDiv<int64_t>(groupNum_, HIFP4_OUTPUT_REG_ONELOOP));
            CopyOut(dstIdx, j, loopCols, loopScaleCols);
        }
    }
    sortedRowIdxInQueue_.FreeTensor(indicesLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::CopyIn(int64_t srcIdx, int64_t colIdx, int64_t loopCols)
{
    LocalTensor<T> inLocal = xInQueue_.AllocTensor<T>();
    DataCopyExtParams copyInParam = {1, static_cast<uint32_t>(loopCols * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
    DataCopyPad(inLocal, xInGm_[srcIdx * cols_ + colIdx * perLoopCols_], copyInParam, padParams);
    xInQueue_.EnQue(inLocal);
}

template <typename T, typename U>
__aicore__ inline void MoeGatherOutHif4Quant<T, U>::CopyOut(int64_t dstIdx, int64_t colIdx, int64_t loopCols,
                                                            int64_t loopScaleCols)
{
    LocalTensor<float> mxScaleLocal = mxScaleOutQueue_.DeQue<float>();
    LocalTensor<uint8_t> outLocal = xQuantOutQueue_.DeQue<uint8_t>();

    // 按照uint8 搬出 多少个unit8（一个unin8包含两个hif4）
    DataCopyExtParams copyOutParams = {1, static_cast<uint32_t>(loopCols * sizeof(uint8_t) / 2), 0, 0, 0};
    DataCopyPad<uint8_t>(expandedXOutGm_[(dstIdx * cols_ + colIdx * perLoopCols_) / 2], outLocal, copyOutParams);

    DataCopyExtParams copyScaleParams = {1, static_cast<uint32_t>(loopScaleCols * sizeof(float)), 0, 0, 0};
    DataCopyPad<float>(expandedScaleOutGm_[dstIdx * scaleCols_ + colIdx * loopScaleCols], mxScaleLocal,
                       copyScaleParams);

    xQuantOutQueue_.FreeTensor(outLocal);
    mxScaleOutQueue_.FreeTensor(mxScaleLocal);
}

} // namespace MoeInitRoutingV3
#endif // MOE_V3_GATHER_HIF4_QUANT_H_REGBASE
