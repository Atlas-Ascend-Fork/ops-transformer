/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH92_REG_HIGH_PREC
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_ARCH92_REG_HIGH_PREC

#include "../../bsa_base_defs.hpp"
#include "../../arch/bsa_resource.hpp"
#include "../bsa_epilogue_dispatch_policy.hpp"
#include "../tile_common/bsa_epilogue_tile_copy.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "../../bsa_gemm_coord.hpp"
#include "../../bsa_matrix_coord.hpp"
#include "../../../tla/tensor_bsa.hpp"

namespace NpuArch::Epilogue::Block {

enum class DRegSplitStages {
    ONE,
    TWO
};

template <class ElementO_, class ElementOTmp_, class ElementS_, class ElementKV_, class TileCopy_,
          class OTmpSrcPos_, // the src TPosition of pv res, viable configurations: GM/L0C
          LseMode LSE_MODE_, LseFormat LSE_FORMAT_, bool transposedMm1_, bool zNOnlineSoftmax_>
class BlockEpilogue<Arch92::EpilogueBsaRescaleO<LSE_MODE_, LSE_FORMAT_, transposedMm1_, zNOnlineSoftmax_>, ElementO_,
                    ElementOTmp_, ElementS_, ElementKV_, TileCopy_, OTmpSrcPos_> {
public:
    using DispatchPolicy = Arch92::EpilogueBsaRescaleO<LSE_MODE_, LSE_FORMAT_, transposedMm1_, zNOnlineSoftmax_>;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementO = ElementO_;
    using ElementOTmp = ElementOTmp_;
    using ElementLse = float;
    using SMDtype = ElementS_;
    using ElementKV = ElementKV_;
    using TileCopy = TileCopy_;
    using OTmpSrcPos = OTmpSrcPos_;

    using CopyUbToGmO = typename TileCopy::CopyUbToGmO;

    static constexpr bool FULL_QUANT_HIF8 = AscendC::IsSameType<ElementKV, hifloat8_t>::value;
    static constexpr uint32_t UB_OTMP_BUF_STAGES = FULL_QUANT_HIF8 ? 1 : 2;
    static constexpr uint32_t UB_DM_BUF_MAX_STAGES = 3;
    static constexpr uint32_t UB_UINT8_BLOCK_SIZE = 32768;
    static constexpr uint32_t DM_UB_GLOBAL_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_ROW_MAX_ELEM_NUM = 64;
    static constexpr uint32_t RESCALE_COL_MAX_ELEM_NUM = 128;
    static constexpr uint32_t RESCALE_VREG_SIZE = 256 / sizeof(ElementOTmp);
    static constexpr float MAX_VALUE_RECIPROCAL_FP8 = 1.0f / 448.0f;
    static constexpr bool FULL_QUANT_FP8 = AscendC::IsSameType<ElementKV, fp8_e4m3fn_t>::value;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, UBufTileHelper& uBufTileHelper,
                                    float pScaleValue_ = 1.0f, float vScaleValue_ = 1.0f)
    {
        pScaleValue = pScaleValue_;
        vScaleValue = vScaleValue_;
        for (uint32_t i = 0; i < UB_OTMP_BUF_STAGES; i++) {
            loUbTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                uBufTileHelper.loStartOffset +
                uBufTileHelper.qBaseTilePerSubCore * uBufTileHelper.embedPerSubCore * sizeof(ElementOTmp) * i);
        }
        for (uint32_t i = 0; i < UB_DM_BUF_MAX_STAGES; i++) {
            dmUbTensor32[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                uBufTileHelper.dmStartOffset + uBufTileHelper.qBaseTilePerSubCore * sizeof(float) * i);
        }
        goUbTensor32 = resource.ubBuf.template GetBufferByByte<ElementOTmp>(uBufTileHelper.goStartOffset);
        goUbTensor16 = resource.ubBuf.template GetBufferByByte<ElementO>(uBufTileHelper.goStartOffset);
        glUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.glStartOffset);
        gmUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.gmStartOffset);
        lseUbTensor32 = resource.ubBuf.template GetBufferByByte<float>(uBufTileHelper.lseStartOffset);
    }

    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorLseGm, class TensorLseUb>
    __aicore__ inline void CopyUbToGmLse(TensorLseGm const& gLseTensorTlaTile, TensorLseUb const& ubLseTensorTla)
    {
        AscendC::DataCopyExtParams repeatParams;
        if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
                      (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
            repeatParams.blockCount = tla::get<0>(ubLseTensorTla.shape());
            repeatParams.blockLen = sizeof(float);
            repeatParams.srcStride = 0;
            repeatParams.dstStride = (tla::get<0>(gLseTensorTlaTile.stride()) - 1) * sizeof(float);
        } else if constexpr (DispatchPolicy::LSE_FORMAT == LseFormat::BNS1) {
            repeatParams.blockCount = 1;
            repeatParams.blockLen = tla::get<0>(ubLseTensorTla.shape()) * sizeof(float);
            repeatParams.srcStride = 0;
            repeatParams.dstStride = 0;
        }
        auto dstOffset = gLseTensorTlaTile.layout()(gLseTensorTlaTile.coord());
        auto srcOffset = ubLseTensorTla.layout()(ubLseTensorTla.coord());
        AscendC::DataCopyPad(gLseTensorTlaTile.data()[dstOffset], ubLseTensorTla.data()[srcOffset], repeatParams);
    }

    template <class TensorO, class TensorLse>
    __aicore__ inline void ComputeTile(TensorO& gOTensorTlaTile, TensorLse& gLseTensorTlaTile, uint32_t curTileMod,
                                       uint32_t ubOTmpBufId, bool isFirstKvSTile, bool isLastKvSTile,
                                       uint32_t colStride, Arch::CrossCoreFlag mm2ToReFlag)
    {
        uint32_t rowNum = tla::get<0>(gOTensorTlaTile.shape());
        uint32_t colNum = tla::get<1>(gOTensorTlaTile.shape());
        uint32_t vlElemNum = AscendC::GetVecLen() / sizeof(ElementOTmp);
        uint32_t colFullLoop = CeilDiv(colNum, vlElemNum) - 1;
        uint32_t colTail = (colNum - 1) % vlElemNum + 1;
        uint32_t dStages = CeilDiv(colNum, RESCALE_VREG_SIZE);

        __ubuf__ ElementOTmp* goUb = (__ubuf__ ElementOTmp*)goUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp* loUb = (__ubuf__ ElementOTmp*)loUbTensor[ubOTmpBufId].GetPhyAddr();
        __ubuf__ ElementOTmp* glUb = (__ubuf__ ElementOTmp*)glUbTensor32.GetPhyAddr();
        __ubuf__ ElementOTmp* dmUb = (__ubuf__ ElementOTmp*)dmUbTensor32[curTileMod].GetPhyAddr();
        __ubuf__ float* gmUb = (__ubuf__ float*)gmUbTensor32.GetPhyAddr();
        __ubuf__ float* lseUb = (__ubuf__ float*)lseUbTensor32.GetPhyAddr();

        Arch::CrossCoreWaitFlag<4, PIPE_V>(mm2ToReFlag);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        if (isFirstKvSTile) {
            if (!isLastKvSTile) {
                uint32_t totalCopyElems = rowNum * colStride;
                AscendC::DataCopy(goUbTensor32, loUbTensor[ubOTmpBufId], totalCopyElems);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                if (dStages == 1) {
                    DivFuncLastAndFirst<DRegSplitStages::ONE>(goUb, loUb, glUb, rowNum, colStride, colTail, vlElemNum);
                } else if (dStages == 2) {
                    DivFuncLastAndFirst<DRegSplitStages::TWO>(goUb, loUb, glUb, rowNum, colStride, colTail, vlElemNum);
                }
            }
        } else if (!isLastKvSTile) {
            if (dStages == 1) {
                RescaleFunc<DRegSplitStages::ONE>(goUb, loUb, dmUb, rowNum, colStride, colTail, vlElemNum);
            } else if (dStages == 2) {
                RescaleFunc<DRegSplitStages::TWO>(goUb, loUb, dmUb, rowNum, colStride, colTail, vlElemNum);
            }
        } else {
            if (dStages == 1) {
                RescaleFuncLastNotFirst<DRegSplitStages::ONE>(goUb, loUb, dmUb, glUb, rowNum, colStride, colTail,
                                                              vlElemNum);
            } else if (dStages == 2) {
                RescaleFuncLastNotFirst<DRegSplitStages::TWO>(goUb, loUb, dmUb, glUb, rowNum, colStride, colTail,
                                                              vlElemNum);
            }
        }
        // release lo buf
        Arch::CrossCoreSetFlag<4, PIPE_V>(mm2ToReFlag);
        if (isLastKvSTile) {
            if constexpr (DispatchPolicy::LSE_MODE == LseMode::OUT_ONLY) {
                uint32_t colNumLseUb = 0;
                uint32_t colStrideLseUb = 0;
                if constexpr ((DispatchPolicy::LSE_FORMAT == LseFormat::TN1) ||
                              (DispatchPolicy::LSE_FORMAT == LseFormat::BSN1)) {
                    LogSumExpFuncQSAxisIncontinuous(gmUb, glUb, lseUb, rowNum);
                    colNumLseUb = 8;
                    colStrideLseUb = 8;
                } else if constexpr (DispatchPolicy::LSE_FORMAT == LseFormat::BNS1) {
                    LogSumExpFuncQSAxisContinuous(gmUb, glUb, lseUb, rowNum);
                    colNumLseUb = 1;
                    colStrideLseUb = 1;
                }
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
                auto ubLseLayoutTla = tla::MakeLayout(tla::MakeShape(rowNum, colNumLseUb),
                                                      tla::MakeStride(colStrideLseUb, tla::Int<1>{}));
                auto ubLseTensorTla = tla::MakeTensor(lseUbTensor32, ubLseLayoutTla, Arch::PositionUB{});
                CopyUbToGmLse(gLseTensorTlaTile, ubLseTensorTla);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (std::is_same<ElementO, bfloat16_t>::value) {
                AscendC::Cast(goUbTensor16, goUbTensor32, AscendC::RoundMode::CAST_RINT, rowNum * colStride);
            } else {
                AscendC::Cast(goUbTensor16, goUbTensor32, AscendC::RoundMode::CAST_NONE, rowNum * colStride);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            auto ubOLayoutTla =
                tla::MakeLayout(tla::MakeShape(rowNum, colNum), tla::MakeStride(colStride, tla::Int<1>{}));
            auto ubOTensorTla = tla::MakeTensor(goUbTensor16, ubOLayoutTla, Arch::PositionUB{});
            copyUbToGmO(gOTensorTlaTile, ubOTensorTla);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
    }

    template <DRegSplitStages dRegSplitStages>
    __simd_vf__ inline void RescaleFunc(__ubuf__ ElementOTmp* goUb, __ubuf__ ElementOTmp* loUb,
                                        __ubuf__ ElementOTmp* dmUb, uint32_t row, uint32_t colStride, uint32_t colTail,
                                        uint32_t vlElemNum)
    {}

    template <>
    __simd_vf__ inline void RescaleFunc<DRegSplitStages::ONE>(__ubuf__ ElementOTmp* goUb, __ubuf__ ElementOTmp* loUb,
                                                              __ubuf__ ElementOTmp* dmUb, uint32_t row,
                                                              uint32_t colStride, uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::Reg;
        RegTensor<float> dmVreg;
        RegTensor<float> goPreVreg;
        RegTensor<float> loVreg;
        RegTensor<float> mulVreg;
        RegTensor<float> goCurVreg;
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg, goUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg, loUb + i * colStride);
            Mul(mulVreg, goPreVreg, dmVreg, pregTail);
            Add(goCurVreg, mulVreg, loVreg, pregTail);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, goCurVreg, pregTail);
        }
    }

    template <>
    __simd_vf__ inline void RescaleFunc<DRegSplitStages::TWO>(__ubuf__ ElementOTmp* goUb, __ubuf__ ElementOTmp* loUb,
                                                              __ubuf__ ElementOTmp* dmUb, uint32_t row,
                                                              uint32_t colStride, uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::Reg;
        RegTensor<float> dmVreg;
        RegTensor<float> goPreVreg0;
        RegTensor<float> goPreVreg1;
        RegTensor<float> loVreg0;
        RegTensor<float> loVreg1;
        RegTensor<float> mulVreg0;
        RegTensor<float> mulVreg1;
        RegTensor<float> goCurVreg0;
        RegTensor<float> goCurVreg1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg0, goUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg1, goUb + i * colStride + vlElemNum);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg0, loUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg1, loUb + i * colStride + vlElemNum);
            Mul(mulVreg0, goPreVreg0, dmVreg, pregFull);
            Mul(mulVreg1, goPreVreg1, dmVreg, pregTail);
            Add(goCurVreg0, mulVreg0, loVreg0, pregFull);
            Add(goCurVreg1, mulVreg1, loVreg1, pregTail);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, goCurVreg0, pregFull);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride + vlElemNum, goCurVreg1, pregTail);
        }
    }

    template <DRegSplitStages dRegSplitStages>
    __simd_vf__ inline void RescaleFuncLastNotFirst(__ubuf__ ElementOTmp* goUb, __ubuf__ ElementOTmp* loUb,
                                                    __ubuf__ ElementOTmp* dmUb, __ubuf__ ElementOTmp* glUb,
                                                    uint32_t row, uint32_t colStride, uint32_t colTail,
                                                    uint32_t vlElemNum)
    {}

    template <>
    __simd_vf__ inline void RescaleFuncLastNotFirst<DRegSplitStages::ONE>(
        __ubuf__ ElementOTmp* goUb, __ubuf__ ElementOTmp* loUb, __ubuf__ ElementOTmp* dmUb, __ubuf__ ElementOTmp* glUb,
        uint32_t row, uint32_t colStride, uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::Reg;
        RegTensor<float> dmVreg;
        RegTensor<float> goPreVreg;
        RegTensor<float> loVreg;
        RegTensor<float> mulVreg;
        RegTensor<float> goCurVreg;
        RegTensor<float> glVreg;
        RegTensor<float> divVreg;
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg, goUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg, loUb + i * colStride);
            Mul(mulVreg, goPreVreg, dmVreg, pregTail);
            Add(goCurVreg, mulVreg, loVreg, pregTail);
            Div(divVreg, goCurVreg, glVreg, pregTail);
            if constexpr (FULL_QUANT_FP8 && !zNOnlineSoftmax_) {
                Muls(divVreg, divVreg, MAX_VALUE_RECIPROCAL_FP8, pregTail);
            }
            if constexpr (FULL_QUANT_HIF8) {
                Muls(divVreg, divVreg, vScaleValue, pregTail);
            }
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, divVreg, pregTail);
        }
    }

    template <>
    __simd_vf__ inline void RescaleFuncLastNotFirst<DRegSplitStages::TWO>(
        __ubuf__ ElementOTmp* goUb, __ubuf__ ElementOTmp* loUb, __ubuf__ ElementOTmp* dmUb, __ubuf__ ElementOTmp* glUb,
        uint32_t row, uint32_t colStride, uint32_t colTail, uint32_t vlElemNum)
    {
        using namespace AscendC::Reg;
        RegTensor<float> dmVreg;
        RegTensor<float> goPreVreg0;
        RegTensor<float> goPreVreg1;
        RegTensor<float> loVreg0;
        RegTensor<float> loVreg1;
        RegTensor<float> mulVreg0;
        RegTensor<float> mulVreg1;
        RegTensor<float> goCurVreg0;
        RegTensor<float> goCurVreg1;
        RegTensor<float> glVreg;
        RegTensor<float> divVreg0;
        RegTensor<float> divVreg1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(dmVreg, dmUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg0, goUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goPreVreg1, goUb + i * colStride + vlElemNum);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg0, loUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(loVreg1, loUb + i * colStride + vlElemNum);
            Mul(mulVreg0, goPreVreg0, dmVreg, pregFull);
            Mul(mulVreg1, goPreVreg1, dmVreg, pregTail);
            Add(goCurVreg0, mulVreg0, loVreg0, pregFull);
            Add(goCurVreg1, mulVreg1, loVreg1, pregTail);
            Div(divVreg0, goCurVreg0, glVreg, pregFull);
            Div(divVreg1, goCurVreg1, glVreg, pregTail);
            if constexpr (FULL_QUANT_FP8 && !zNOnlineSoftmax_) {
                Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregFull);
                Muls(divVreg1, divVreg1, MAX_VALUE_RECIPROCAL_FP8, pregTail);
            }
            if constexpr (FULL_QUANT_HIF8) {
                Muls(divVreg0, divVreg0, vScaleValue, pregFull);
                Muls(divVreg1, divVreg1, vScaleValue, pregTail);
            }
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, divVreg0, pregFull);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride + vlElemNum, divVreg1, pregTail);
        }
    }

    template <DRegSplitStages dRegSplitStages>
    __simd_vf__ inline void DivFuncLastAndFirst(__ubuf__ ElementOTmp* goUb, __ubuf__ ElementOTmp* loUb,
                                                __ubuf__ ElementOTmp* glUb, uint32_t row, uint32_t colStride,
                                                uint32_t colTail, uint32_t vlElemNum)
    {}

    template <>
    __simd_vf__ inline void DivFuncLastAndFirst<DRegSplitStages::ONE>(__ubuf__ ElementOTmp* goUb,
                                                                      __ubuf__ ElementOTmp* loUb,
                                                                      __ubuf__ ElementOTmp* glUb, uint32_t row,
                                                                      uint32_t colStride, uint32_t colTail,
                                                                      uint32_t vlElemNum)
    {
        using namespace AscendC::Reg;
        RegTensor<float> goCurVreg;
        RegTensor<float> glVreg;
        RegTensor<float> divVreg;
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goCurVreg, loUb + i * colStride);
            Div(divVreg, goCurVreg, glVreg, pregTail);
            if constexpr (FULL_QUANT_FP8 && !zNOnlineSoftmax_) {
                Muls(divVreg, divVreg, MAX_VALUE_RECIPROCAL_FP8, pregTail);
            }
            if constexpr (FULL_QUANT_HIF8) {
                Muls(divVreg, divVreg, vScaleValue, pregTail);
            }
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, divVreg, pregTail);
        }
    }

    template <>
    __simd_vf__ inline void DivFuncLastAndFirst<DRegSplitStages::TWO>(__ubuf__ ElementOTmp* goUb,
                                                                      __ubuf__ ElementOTmp* loUb,
                                                                      __ubuf__ ElementOTmp* glUb, uint32_t row,
                                                                      uint32_t colStride, uint32_t colTail,
                                                                      uint32_t vlElemNum)
    {
        using namespace AscendC::Reg;
        RegTensor<float> goCurVreg0;
        RegTensor<float> goCurVreg1;
        RegTensor<float> glVreg;
        RegTensor<float> divVreg0;
        RegTensor<float> divVreg1;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(colTail);
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goCurVreg0, loUb + i * colStride);
            LoadAlign<ElementOTmp, LoadDist::DIST_NORM>(goCurVreg1, loUb + i * colStride + vlElemNum);
            Div(divVreg0, goCurVreg0, glVreg, pregFull);
            Div(divVreg1, goCurVreg1, glVreg, pregTail);
            if constexpr (FULL_QUANT_FP8 && !zNOnlineSoftmax_) {
                Muls(divVreg0, divVreg0, MAX_VALUE_RECIPROCAL_FP8, pregFull);
                Muls(divVreg1, divVreg1, MAX_VALUE_RECIPROCAL_FP8, pregTail);
            }
            if constexpr (FULL_QUANT_HIF8) {
                Muls(divVreg0, divVreg0, vScaleValue, pregFull);
                Muls(divVreg1, divVreg1, vScaleValue, pregTail);
            }
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride, divVreg0, pregFull);
            StoreAlign<ElementOTmp, StoreDist::DIST_NORM_B32>(goUb + i * colStride + vlElemNum, divVreg1, pregTail);
        }
    }

    // When qS can be copied from UB to GM continuously,
    // qS would be stored from reg to UB continuously.
    __simd_vf__ inline void LogSumExpFuncQSAxisContinuous(__ubuf__ float* gmUb, __ubuf__ float* glUb,
                                                          __ubuf__ float* lseUb, uint32_t row)
    {
        // This vf works only when the rowNum in each AIV does not exceed 64
        using namespace AscendC::Reg;
        RegTensor<float> gmVreg;
        RegTensor<float> glVreg;
        RegTensor<float> logGlVreg;
        RegTensor<float> lseVreg;
        RegTensor<float> pScaleVreg;
        RegTensor<float> lnPScaleVreg;
        MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(row);
        static constexpr LnSpecificMode mode = {MaskMergeMode::ZEROING, AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE};

        LoadAlign<float, LoadDist::DIST_NORM>(glVreg, glUb);
        LoadAlign<float, LoadDist::DIST_NORM>(gmVreg, gmUb);
        Ln<float, &mode>(logGlVreg, glVreg, pregTail);
        Add(lseVreg, logGlVreg, gmVreg, pregTail);
        if constexpr (FULL_QUANT_HIF8) {
            Duplicate(pScaleVreg, pScaleValue);
            Ln<float, &mode>(lnPScaleVreg, pScaleVreg, pregTail);
            Sub(lseVreg, lseVreg, lnPScaleVreg, pregTail);
        }
        StoreAlign<float, StoreDist::DIST_NORM_B32>(lseUb, lseVreg, pregTail);
    }

    // When qS cannot be copied from UB to GM continuously,
    // lse would be broadcasted and then stored from reg to UB continuously.
    __simd_vf__ inline void LogSumExpFuncQSAxisIncontinuous(__ubuf__ float* gmUb, __ubuf__ float* glUb,
                                                            __ubuf__ float* lseUb, uint32_t row)
    {
        using namespace AscendC::Reg;
        RegTensor<float> gmRowwiseVreg;
        RegTensor<float> glRowwiseVreg;
        RegTensor<float> logGlRowwiseVreg;
        RegTensor<float> lseRowwiseVreg;
        RegTensor<float> pScaleVreg;
        RegTensor<float> lnPScaleVreg;
        MaskReg pregBlock = CreateMask<float, MaskPattern::VL8>();
        static constexpr LnSpecificMode mode = {MaskMergeMode::ZEROING, AscendC::LnAlgo::PRECISION_1ULP_FTZ_FALSE};
        static constexpr uint32_t lseRowStride = 32 / sizeof(float);

        if constexpr (FULL_QUANT_HIF8) {
            Duplicate(pScaleVreg, pScaleValue);
            Ln<float, &mode>(lnPScaleVreg, pScaleVreg, pregBlock);
        }

        // gl/gm 只写入有效 Q 行，逐行处理以免奇数尾块读取下一行未初始化的值。
        // TN1/BSN1 在 UB 中每行占一个 32B 块，供 CopyUbToGmLse 跨步写回。
        for (uint16_t i = 0; i < row; i++) {
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(glRowwiseVreg, glUb + i);
            LoadAlign<ElementOTmp, LoadDist::DIST_BRC_B32>(gmRowwiseVreg, gmUb + i);
            Ln<float, &mode>(logGlRowwiseVreg, glRowwiseVreg, pregBlock);
            Add(lseRowwiseVreg, logGlRowwiseVreg, gmRowwiseVreg, pregBlock);
            if constexpr (FULL_QUANT_HIF8) {
                Sub(lseRowwiseVreg, lseRowwiseVreg, lnPScaleVreg, pregBlock);
            }
            StoreAlign<float, StoreDist::DIST_NORM_B32>(lseUb + i * lseRowStride, lseRowwiseVreg, pregBlock);
        }
    }

    template <class TensorO, class TensorLse>
    __aicore__ inline void operator()(TensorO& gOTensor, TensorLse& gLseTensor, GemmCoord actualOriShape,
                                      uint32_t curTileMod, uint32_t gatheredKvSTileIdx, bool isFirstKvSTile,
                                      bool isLastKvSTile, Arch::CrossCoreFlag mm2ToReFlag)
    {
        const uint32_t rowNum = actualOriShape[0];
        const uint32_t colNum = actualOriShape[1];
        const uint32_t colStride = RoundUp(colNum, 8);
        auto gOTensorTlaTile = GetTile(gOTensor, tla::MakeCoord(0, 0), tla::MakeShape(rowNum, colNum));
        auto gLseTensorTlaTile = GetTile(gLseTensor, tla::MakeCoord(0, 0), tla::MakeShape(rowNum, 1));
        uint32_t ubOTmpBufId = gatheredKvSTileIdx % UB_OTMP_BUF_STAGES;

        if (rowNum > 0) {
            ComputeTile(gOTensorTlaTile, gLseTensorTlaTile, curTileMod, ubOTmpBufId, isFirstKvSTile, isLastKvSTile,
                        colStride, mm2ToReFlag);
        } else {
            Arch::CrossCoreWaitFlag<4, PIPE_V>(mm2ToReFlag);
            Arch::CrossCoreSetFlag<4, PIPE_V>(mm2ToReFlag);
        }
    }

private:
    float pScaleValue{1.0f};
    // Per-tensor V dequantization scale. Only used for the full-quant HIF8 path, where the MM2
    // Fixpipe no longer applies it; applied on the normalized output below.
    float vScaleValue{1.0f};
    AscendC::LocalTensor<ElementOTmp> loUbTensor[UB_OTMP_BUF_STAGES];
    AscendC::LocalTensor<float> gmUbTensor32;
    AscendC::LocalTensor<float> dmUbTensor32[UB_DM_BUF_MAX_STAGES];
    AscendC::LocalTensor<float> glUbTensor32;
    AscendC::LocalTensor<ElementO> goUbTensor16;
    AscendC::LocalTensor<ElementOTmp> goUbTensor32;
    AscendC::LocalTensor<float> lseUbTensor32;

    CopyUbToGmO copyUbToGmO;
};
} // namespace NpuArch::Epilogue::Block
#endif
