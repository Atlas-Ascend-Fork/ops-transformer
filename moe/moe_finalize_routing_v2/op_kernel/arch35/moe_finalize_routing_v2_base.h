/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_finalize_routing_v2_partial_load.h
 * \brief
 */

#ifndef MOE_FINALIZE_ROUTING_V2_BASE_H_
#define MOE_FINALIZE_ROUTING_V2_BASE_H_
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace MoeFinalizeRoutingV2Regbase {
using namespace AscendC;
using namespace AscendC::Reg;
using AscendC::Reg::MaskReg;
using AscendC::Reg::RegTensor;
using AscendC::Reg::UnalignRegForLoad;
using AscendC::Reg::UnalignRegForStore;

constexpr int32_t DROPLESS_COLUMN = 0; // 按列读取
constexpr int32_t DROP_PAD_COLUMN = 1; // 按列读取
constexpr int32_t DROPLESS_ROW = 2;    // 按行读取
constexpr int32_t DROP_PAD_ROW = 3;    // 按行读取
constexpr int32_t INVALID_IDX = -1;
constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int32_t VL_FP32 = Ops::Base::GetVRegSize() / sizeof(float);

constexpr uint32_t UBBLOCKSIZE = 32u;
constexpr uint32_t VREGSIZE = 256u;
constexpr uint32_t VL_B16 = VREGSIZE / sizeof(bfloat16_t); // 128
constexpr uint32_t HIF4_BLOCK_SIZE = 64u;
constexpr uint32_t NUM_FP4_IN_BYTE = 2u;
constexpr uint32_t X_ADDR_STRIDE = VL_B16 / NUM_FP4_IN_BYTE;      // 64
constexpr uint32_t SCALES_ADDR_STRIDE = VL_B16 / HIF4_BLOCK_SIZE; // 2
constexpr uint32_t SIZEOF_B32 = static_cast<uint32_t>(sizeof(uint32_t));

constexpr AscendC::Reg::CastTrait castTraitB162B32Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

constexpr AscendC::Reg::CastTrait castTraitB322B16Even = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::NO_SAT,
    AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT,
};

constexpr CastTrait castTraitB4ToB16 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
constexpr CastTrait castTraitE6M2ToBF16 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
constexpr CastTrait castTraitB8ToB16 = {
    RegLayout::ZERO,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};
constexpr CastTrait castTraitI16ToF16 = {
    RegLayout::UNKNOWN,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};
constexpr CastTrait castTraitF16ToBF16 = {
    RegLayout::UNKNOWN,
    SatMode::UNKNOWN,
    MaskMergeMode::ZEROING,
    RoundMode::CAST_RINT,
};

template <typename T>
__aicore__ inline int32_t RoundUp(int32_t num)
{
    int32_t elemNum = Ops::Base::GetUbBlockSize() / sizeof(T);
    return Ops::Base::CeilAlign(num, elemNum);
}

template <typename T>
__aicore__ inline void LoadInputData(RegTensor<float>& dst, __ubuf__ T* src, MaskReg pregLoop, uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void LoadInputDataUnalign(RegTensor<float>& dst, __ubuf__ T*& src, UnalignRegForLoad& uSrc,
                                            MaskReg pregLoop, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadUnAlign(dst, uSrc, src, postUpdateStride);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        RegTensor<T> tmpUnPack;
        LoadUnAlign(tmp, uSrc, src, postUpdateStride);
        UnPack((RegTensor<uint32_t>&)tmpUnPack, (RegTensor<uint16_t>&)tmp);
        Cast<float, T, castTraitB162B32Even>(dst, tmpUnPack, pregLoop);
    }
}

template <typename T>
__aicore__ inline void LoadInputDataWithBrc(RegTensor<float>& dst, __ubuf__ T* src, MaskReg pregLoop,
                                            uint32_t srcOffset)
{
    if constexpr (IsSameType<T, float>::value) {
        LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(dst, src + srcOffset);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        LoadAlign<T, AscendC::Reg::LoadDist::DIST_BRC_B16>(tmp, src + srcOffset);
        Cast<float, T, castTraitB162B32Even>(dst, tmp, pregLoop);
    }
}

template <typename T>
__aicore__ inline void StoreOuputDataUnalign(RegTensor<float>& src, __ubuf__ T*& dst, UnalignRegForStore& uDst,
                                             MaskReg pregLoop, uint32_t postUpdateStride)
{
    if constexpr (IsSameType<T, float>::value) {
        StoreUnAlign(dst, src, uDst, postUpdateStride);
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        RegTensor<T> tmp;
        RegTensor<T> tmpPack;
        Cast<T, float, castTraitB322B16Even>(tmp, src, pregLoop);
        Pack((RegTensor<uint16_t>&)tmpPack, (RegTensor<uint32_t>&)tmp);
        StoreUnAlign(dst, tmpPack, uDst, postUpdateStride);
    }
}

template <typename T, bool hasX2>
__aicore__ inline void VFProcessX1AndX2(const LocalTensor<float>& yLocal, const LocalTensor<T>& x1Local,
                                        const LocalTensor<T>& x2Local, uint16_t processLen)
{
    __ubuf__ float* yLocalAddr = (__ubuf__ float*)yLocal.GetPhyAddr();
    __ubuf__ T* x1LocalAddr = (__ubuf__ T*)x1Local.GetPhyAddr();
    __ubuf__ T* x2LocalAddr = (__ubuf__ T*)x2Local.GetPhyAddr();
    uint16_t loopCount = Ops::Base::CeilDiv<uint16_t>(processLen, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> x1;
        RegTensor<float> x2;
        RegTensor<float> y;
        MaskReg pregLoop;
        uint32_t sreg = processLen;
        for (uint16_t i = 0; i < loopCount; i += 1) {
            pregLoop = UpdateMask<float>(sreg);
            LoadInputData<T>(x1, x1LocalAddr, pregLoop, i * VL_FP32);
            if constexpr (hasX2) {
                LoadInputData<T>(x2, x2LocalAddr, pregLoop, i * VL_FP32);
                Add(y, x1, x2, pregLoop);
                StoreAlign(yLocalAddr + i * VL_FP32, y, pregLoop);
            } else {
                StoreAlign(yLocalAddr + i * VL_FP32, x1, pregLoop);
            }
        }
    }
}

template <typename T>
__aicore__ inline void ProcessX1AndX2(const LocalTensor<float>& yLocal, const LocalTensor<T>& x1Local,
                                      const LocalTensor<T>& x2Local, uint16_t processLen, bool hasX1, bool hasX2)
{
    if (hasX1 && hasX2) {
        VFProcessX1AndX2<T, true>(yLocal, x1Local, x2Local, processLen);
    } else if (hasX1 && !hasX2) {
        VFProcessX1AndX2<T, false>(yLocal, x1Local, x2Local, processLen);
    } else {
        AscendC::Duplicate(yLocal, static_cast<float>(0.0f), processLen);
    }
}

template <typename T, typename S, bool hasBias, bool hasScale>
__aicore__ inline void VFProcessExpandXBiasScale(const LocalTensor<float> yLocal, const LocalTensor<T>& expandedXLocal,
                                                 const LocalTensor<T>& biasLocal, S scale, uint16_t processLen)
{
    __ubuf__ float* yLocalAddr = (__ubuf__ float*)yLocal.GetPhyAddr();
    __ubuf__ float* srcLocalAddr = yLocalAddr;
    __ubuf__ T* expandedXLocalAddr = (__ubuf__ T*)expandedXLocal.GetPhyAddr();
    __ubuf__ T* biasLocalAddr = hasBias ? (__ubuf__ T*)biasLocal.GetPhyAddr() : nullptr;

    uint16_t loopCount = processLen / VL_FP32;
    uint16_t tailNum = processLen - loopCount * VL_FP32;
    uint16_t tailLoop = Ops::Base::CeilDiv<uint16_t>(tailNum, VL_FP32);
    __VEC_SCOPE__
    {
        RegTensor<float> expandedX;
        RegTensor<float> bias;
        RegTensor<float> y;
        RegTensor<float> src;
        RegTensor<float> scaleReg;
        RegTensor<S> tmp;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<S, AscendC::Reg::MaskPattern::ALL>();
        UnalignRegForLoad uSrc;
        UnalignRegForLoad uExpandedX;
        UnalignRegForLoad uBias;
        UnalignRegForStore uDst;
        LoadUnAlignPre<float>(uSrc, srcLocalAddr);
        LoadUnAlignPre<T>(uExpandedX, expandedXLocalAddr);
        if constexpr (hasBias) {
            LoadUnAlignPre<T>(uBias, biasLocalAddr);
        }
        if constexpr (hasScale) {
            if constexpr (IsSameType<S, float>::value) {
                Duplicate(scaleReg, scale, pregMain);
            } else if constexpr (IsSameType<S, half>::value || IsSameType<S, bfloat16_t>::value) {
                Duplicate(tmp, scale, pregMain);
                Cast<float, S, castTraitB162B32Even>(scaleReg, tmp, pregMain);
            }
        }
        for (uint16_t i = 0; i < loopCount; i++) {
            LoadInputDataUnalign<T>(expandedX, expandedXLocalAddr, uExpandedX, pregMain, VL_FP32);
            LoadInputDataUnalign<float>(src, srcLocalAddr, uSrc, pregMain, VL_FP32);
            if constexpr (hasBias) {
                LoadInputDataUnalign<T>(bias, biasLocalAddr, uBias, pregMain, VL_FP32);
                Add(expandedX, expandedX, bias, pregMain);
            }
            if constexpr (hasScale) {
                Mul(expandedX, expandedX, scaleReg, pregMain);
            }
            Add(y, src, expandedX, pregMain);
            StoreOuputDataUnalign<float>(y, yLocalAddr, uDst, pregMain, VL_FP32);
        }

        uint32_t sreg = tailNum;
        for (uint16_t i = 0; i < tailLoop; i++) {
            pregLoop = UpdateMask<float>(sreg);
            LoadInputDataUnalign<T>(expandedX, expandedXLocalAddr, uExpandedX, pregLoop, tailNum);
            LoadInputDataUnalign<float>(src, srcLocalAddr, uSrc, pregLoop, tailNum);
            if constexpr (hasBias) {
                LoadInputDataUnalign<T>(bias, biasLocalAddr, uBias, pregLoop, tailNum);
                Add(expandedX, expandedX, bias, pregLoop);
            }
            if constexpr (hasScale) {
                Mul(expandedX, expandedX, scaleReg, pregLoop);
            }
            Add(y, src, expandedX, pregLoop);
            StoreOuputDataUnalign<float>(y, yLocalAddr, uDst, pregLoop, tailNum);
        }
        StoreUnAlignPost(yLocalAddr, uDst, 0);
    }
}

template <typename T, typename S>
__aicore__ inline void ProcessExpandXBiasScale(const LocalTensor<float> yLocal, const LocalTensor<T>& expandedXLocal,
                                               const LocalTensor<T>& biasLocal, S scale, uint16_t processLen,
                                               bool hasBias, bool hasScale)
{
    if (hasBias && hasScale) {
        VFProcessExpandXBiasScale<T, S, true, true>(yLocal, expandedXLocal, biasLocal, scale, processLen);
    } else if (hasBias && !hasScale) {
        VFProcessExpandXBiasScale<T, S, true, false>(yLocal, expandedXLocal, biasLocal, scale, processLen);
    } else if (!hasBias && hasScale) {
        VFProcessExpandXBiasScale<T, S, false, true>(yLocal, expandedXLocal, biasLocal, scale, processLen);
    } else {
        VFProcessExpandXBiasScale<T, S, false, false>(yLocal, expandedXLocal, biasLocal, scale, processLen);
    }
}

template <typename T, typename S, bool hasBias, bool hasScale>
__aicore__ inline void VFProcessExpandXBiasScaleOptimized(const LocalTensor<float>& yLocal,
                                                          const LocalTensor<T>& expandedXLocal,
                                                          const LocalTensor<T>& biasLocal,
                                                          const LocalTensor<S>& scalesLocal, uint16_t validK,
                                                          uint16_t processLen)
{
    __ubuf__ float* yOriginAddr = (__ubuf__ float*)yLocal.GetPhyAddr();
    __ubuf__ float* yLocalAddr = yOriginAddr;
    __ubuf__ float* srcLocalAddr = yOriginAddr;
    __ubuf__ T* expandedXLocalAddr = (__ubuf__ T*)expandedXLocal.GetPhyAddr();
    __ubuf__ T* biasLocalAddr = hasBias ? (__ubuf__ T*)biasLocal.GetPhyAddr() : nullptr;
    __ubuf__ S* scalesLocalAddr = hasScale ? (__ubuf__ S*)scalesLocal.GetPhyAddr() : nullptr;

    uint16_t loopCount = processLen / VL_FP32;
    uint16_t tailNum = processLen - loopCount * VL_FP32;
    uint16_t tailLoop = Ops::Base::CeilDiv<uint16_t>(tailNum, VL_FP32);
    uint16_t processLenAlign = RoundUp<T>(processLen);
    __VEC_SCOPE__
    {
        RegTensor<float> expandedX;
        RegTensor<float> bias;
        RegTensor<float> y;
        RegTensor<float> src;
        RegTensor<float> scaleReg;
        RegTensor<T> tmp;
        MaskReg pregLoop;
        MaskReg pregMain = CreateMask<T, AscendC::Reg::MaskPattern::ALL>();
        UnalignRegForLoad uSrc;
        UnalignRegForLoad uExpandedX;
        UnalignRegForLoad uBias;
        UnalignRegForStore uDst;

        for (uint16_t k = 0; k < validK; k++) {
            srcLocalAddr = yOriginAddr;
            yLocalAddr = yOriginAddr;
            LoadUnAlignPre<float>(uSrc, srcLocalAddr);
            if constexpr (hasScale) {
                LoadInputDataWithBrc<S>(scaleReg, scalesLocalAddr, pregMain, k);
            }
            for (uint16_t i = 0; i < loopCount; i++) {
                LoadInputDataUnalign<float>(src, srcLocalAddr, uSrc, pregMain, VL_FP32);
                LoadInputData<T>(expandedX, expandedXLocalAddr, pregMain, k * processLenAlign + i * VL_FP32);
                if constexpr (hasBias) {
                    LoadInputData<T>(bias, biasLocalAddr, pregMain, k * processLenAlign + i * VL_FP32);
                    Add(expandedX, expandedX, bias, pregMain);
                }
                if constexpr (hasScale) {
                    Mul(expandedX, expandedX, scaleReg, pregMain);
                }
                Add(y, src, expandedX, pregMain);
                StoreOuputDataUnalign<float>(y, yLocalAddr, uDst, pregMain, VL_FP32);
            }
            uint32_t sreg = tailNum;
            for (uint16_t i = 0; i < tailLoop; i++) {
                pregLoop = UpdateMask<float>(sreg);
                LoadInputDataUnalign<float>(src, srcLocalAddr, uSrc, pregLoop, tailNum);
                LoadInputData<T>(expandedX, expandedXLocalAddr, pregLoop, k * processLenAlign + loopCount * VL_FP32);
                if constexpr (hasBias) {
                    LoadInputData<T>(bias, biasLocalAddr, pregMain, k * processLenAlign + loopCount * VL_FP32);
                    Add(expandedX, expandedX, bias, pregLoop);
                }
                if constexpr (hasScale) {
                    Mul(expandedX, expandedX, scaleReg, pregLoop);
                }
                Add(y, src, expandedX, pregLoop);
                StoreOuputDataUnalign<float>(y, yLocalAddr, uDst, pregLoop, tailNum);
            }
            StoreUnAlignPost(yLocalAddr, uDst, 0);
            LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
        }
    }
}

template <typename T, typename S>
__aicore__ inline void ProcessExpandXBiasScaleOptimized(const LocalTensor<float>& yLocal,
                                                        const LocalTensor<T>& expandedXLocal,
                                                        const LocalTensor<T>& biasLocal,
                                                        const LocalTensor<S>& scalesLocal, uint16_t validK,
                                                        uint16_t processLen, bool hasBias, bool hasScale)
{
    if (hasBias && hasScale) {
        VFProcessExpandXBiasScaleOptimized<T, S, true, true>(yLocal, expandedXLocal, biasLocal, scalesLocal, validK,
                                                             processLen);
    } else if (hasBias && !hasScale) {
        VFProcessExpandXBiasScaleOptimized<T, S, true, false>(yLocal, expandedXLocal, biasLocal, scalesLocal, validK,
                                                              processLen);
    } else if (!hasBias && hasScale) {
        VFProcessExpandXBiasScaleOptimized<T, S, false, true>(yLocal, expandedXLocal, biasLocal, scalesLocal, validK,
                                                              processLen);
    } else {
        VFProcessExpandXBiasScaleOptimized<T, S, false, false>(yLocal, expandedXLocal, biasLocal, scalesLocal, validK,
                                                               processLen);
    }
}

template <typename T>
__aicore__ inline void CopyIn(const GlobalTensor<T>& inputGm, const LocalTensor<T>& inputTensor, const uint16_t nBurst,
                              const uint32_t copyLen)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(inputTensor, inputGm, dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T>
__aicore__ inline void CopyOut(const LocalTensor<T>& outputTensor, const GlobalTensor<T>& outputGm,
                               const uint16_t nBurst, const uint32_t copyLen)
{
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = nBurst;
    dataCopyParams.blockLen = copyLen * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(outputGm, outputTensor, dataCopyParams);
}

// 以下 hif4（hifloat4）反量化相关实现依赖 fp8_e6m2_t/fp4x2_e1m2_t，仅 960（9201/9202）工具链支持，
// 且本文件被所有 kernel（含非 hif4 的 950 浮点 kernel）包含，故这里按架构守卫只编译 hif4 相关实现。
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202))

// 计算一行的y+=x1+x2
template <bool HAS_X2>
__simd_vf__ void VFComputeYWithX1AndX2BF16(__ubuf__ bfloat16_t* yAddr, __ubuf__ bfloat16_t* x1Addr,
                                           __ubuf__ bfloat16_t* x2Addr, uint32_t elemNum, uint32_t loopNum)
{
    using namespace AscendC::Reg;

    RegTensor<bfloat16_t> x1, x2;
    MaskReg loopMask;
    uint32_t sregVal = elemNum;
    for (uint32_t i = 0; i < loopNum; ++i) {
        loopMask = UpdateMask<bfloat16_t>(sregVal);
        // 拷入x1
        LoadAlign(x1, x1Addr + i * VL_B16);
        if constexpr (HAS_X2) {
            // 拷入x2
            LoadAlign(x2, x2Addr + i * VL_B16);
            // 加到x1上
            Add(x1, x1, x2, loopMask);
        }
        StoreAlign(yAddr + i * VL_B16, x1, loopMask);
    }
}

/**
 * 把4字节src的低2字节，复制拷贝到2个2字节的dst。
 * 要求src的高2字节已清0。
 */
__simd_callee__ void RegCastRepU32ToU16(RegTensor<uint16_t>& dstReg, RegTensor<uint32_t>& srcReg)
{
    using namespace AscendC::Reg;
    static constexpr int16_t LOW_B16_BITS = static_cast<int16_t>(16);

    MaskReg b16AllMask = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg b32AllMask = CreateMask<uint32_t, MaskPattern::ALL>();

    RegTensor<uint32_t> srcHighPosReg;
    ShiftLefts(srcHighPosReg, srcReg, LOW_B16_BITS, b32AllMask); // 把B32的低2字节移动到高2字节，低2字节为0
    Add(dstReg, (RegTensor<uint16_t>&)srcReg, (RegTensor<uint16_t>&)srcHighPosReg, b16AllMask);
}

/*
 * 从hif4的scale中提取出Ea、Eb、Ec
 * 输入uint32_t类型的inReg，提取前1个或2个scales（目标类型BF16的每一个yReg对应2个scales）
 * 输出与yReg元素对齐的eaReg、ebReg、ecReg
 */
template <bool COMPUTE_SINGLE_SCALE = false>
__simd_callee__ void RegExtractHif4ScalesForB16(RegTensor<uint32_t>& inReg, RegTensor<bfloat16_t>& eaRegOut,
                                                RegTensor<uint16_t>& ebRegOut, RegTensor<uint16_t>& ecRegOut,
                                                RegTensor<int16_t>& rep4IndexB16, RegTensor<int16_t>& rep8IndexB16,
                                                RegTensor<uint32_t>& u32LowestB8Reg,
                                                RegTensor<uint32_t>& u32LowestB16Reg,
                                                RegTensor<uint16_t>& u16LowestB1Reg, MaskReg& b16HalfTailMask)
{
    using namespace AscendC::Reg;

    constexpr int16_t EB_SHIFT_NUM = static_cast<int16_t>(8);
    constexpr int16_t EC_SHIFT_NUM = static_cast<int16_t>(16);

    MaskReg b16AllMask = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg b32AllMask = CreateMask<uint32_t, MaskPattern::ALL>();

    // 从inReg提取scaleReg
    RegTensor<uint32_t> scaleReg;
    if constexpr (COMPUTE_SINGLE_SCALE) {
        MaskReg b32HalfMask = CreateMask<uint32_t, MaskPattern::H>();
        Duplicate(scaleReg, inReg, b32HalfMask);
    } else {
        RegTensor<int32_t> scaleIndexB32;
        Duplicate(scaleIndexB32, static_cast<int32_t>(1), b16HalfTailMask); // 0...0,1...1
        Gather(scaleReg, inReg, (RegTensor<uint32_t>&)scaleIndexB32);       // inReg[0],...,inReg[1],...
    }

    // 提取Ea=scaleB32[0~7]
    RegTensor<uint32_t> eaU32Reg;
    RegTensor<uint16_t> eaU16Reg;
    And(eaU32Reg, scaleReg, u32LowestB8Reg, b32AllMask); // Ea=Scale&0xFF
    RegCastRepU32ToU16(eaU16Reg, eaU32Reg);
    Cast<bfloat16_t, fp8_e6m2_t, castTraitE6M2ToBF16>(eaRegOut, (RegTensor<fp8_e6m2_t>&)eaU16Reg,
                                                      b16AllMask); // 取eaU32的低8位直接Cast成bf16

    // 提取Eb=scaleB32[8~15]，其中每8个元素用一个相同的值
    RegTensor<uint32_t> ebU32Reg;
    ShiftRights(ebU32Reg, scaleReg, EB_SHIFT_NUM, b32AllMask); // Eb=Scale>>8
    And(ebU32Reg, ebU32Reg, u32LowestB8Reg, b32AllMask);       // Eb=(Scale>>8)&0xFF
    RegCastRepU32ToU16(ebRegOut, ebU32Reg);
    ShiftRight(ebRegOut, ebRegOut, rep8IndexB16, b16AllMask); // Eb>>i, i=0->7,0->7
    And(ebRegOut, ebRegOut, u16LowestB1Reg, b16AllMask);      // Ebi = (Eb>>i)&0b1

    // 提取Ec=scaleB32[16~31]，其中每4个元素用一个相同的值
    RegTensor<uint32_t> ecU32Reg;
    ShiftRights(ecU32Reg, scaleReg, EC_SHIFT_NUM, b32AllMask); // Ec=Scale>>16
    And(ecU32Reg, ecU32Reg, u32LowestB16Reg, b32AllMask);      // Ec=(Scale>>16)&&0xFFFF
    RegCastRepU32ToU16(ecRegOut, ecU32Reg);
    ShiftRight(ecRegOut, ecRegOut, rep4IndexB16, b16AllMask); // Ec>>i, i=0->15,0->15
    And(ecRegOut, ecRegOut, u16LowestB1Reg, b16AllMask);      // Eci = (Ec>>i)&0b1
}

template <bool HAS_BIAS>
__simd_callee__ void RegDequantizeWithBias(__ubuf__ fp4x2_e1m2_t* xAddr, __ubuf__ bfloat16_t* biasAddr, MaskReg& mask,
                                           RegTensor<bfloat16_t>& eaReg, RegTensor<uint16_t>& ebReg,
                                           RegTensor<uint16_t>& ecReg, RegTensor<bfloat16_t>& xBF16RegOut)
{
    using namespace AscendC::Reg;

    MaskReg b16AllMask = CreateMask<uint16_t, MaskPattern::ALL>();

    RegTensor<fp4x2_e1m2_t> xInReg;
    RegTensor<half> powF16Reg;
    RegTensor<bfloat16_t> biasInReg, powBF16Reg;
    RegTensor<uint16_t> ebcReg;
    RegTensor<int16_t> powReg;

    // 拷入x，每个fp4x2放在4字节的低位，因为1个fp4x2后面要cast成2个bf16，4字节
    LoadAlign<fp4x2_e1m2_t, LoadDist::DIST_UNPACK4_B8>(xInReg, xAddr);
    Cast<bfloat16_t, fp4x2_e1m2_t, castTraitB4ToB16>(xBF16RegOut, xInReg, b16AllMask);

    // 计算xBF16RegOut=Cast<bf16, e1m2>(x)*pow(2,Eb+Ec)*(1/Ea_rec)
    Add(ebcReg, ebReg, ecReg, mask); // ebcReg=Eb+Ec
    Duplicate(powReg, static_cast<int16_t>(1));
    ShiftLeft(powReg, powReg, (RegTensor<int16_t>&)ebcReg, mask); // powReg=2^(Eb+Ec)
    // uint16居然没法直接转bf16，需要转成half中转一下（此处由于ebcReg不会超过65504因此可以中转）
    Cast<half, int16_t, castTraitI16ToF16>(powF16Reg, powReg, mask);
    Cast<bfloat16_t, half, castTraitF16ToBF16>(powBF16Reg, powF16Reg, mask);
    Mul(xBF16RegOut, xBF16RegOut, powBF16Reg, mask); // xBF16RegOut=Cast<bf16,fp4>(xInReg)*2^(Eb+Ec)
    // eaReg=1/Ea_rec
    Mul(xBF16RegOut, xBF16RegOut, eaReg, mask); // xBF16RegOut=Cast<bf16,fp4>(xInReg)*2^(Eb+Ec)*(1/Ea_rec)

    // 如果有bias就累加bias到xBF16RegOut
    if constexpr (HAS_BIAS) {
        LoadAlign(biasInReg, biasAddr);
        Add(xBF16RegOut, xBF16RegOut, biasInReg, mask);
    }
}

template <typename T>
__simd_callee__ void RegAccumlateDstAddrWithSrc(__ubuf__ T* dstAddr, RegTensor<T>& srcReg, MaskReg& mask)
{
    using namespace AscendC::Reg;
    RegTensor<T> dstReg;
    LoadAlign(dstReg, dstAddr);
    Add(dstReg, dstReg, srcReg, mask);
    StoreAlign(dstAddr, dstReg, mask);
}

// 计算k行的x反量化+bias累加到y
template <bool HAS_BIAS>
__simd_vf__ void VFComputeDequantizeWithBias(__ubuf__ bfloat16_t* yAddr, __ubuf__ fp4x2_e1m2_t* xAddr,
                                             __ubuf__ uint32_t* scalesAddr, __ubuf__ bfloat16_t* biasAddr,
                                             uint32_t validK, uint32_t rowElemNum, uint32_t scalesRowElemNumAligned,
                                             uint32_t rowLoopNum, uint32_t tailElemNum, uint32_t tailLoopNum)
{
    using namespace AscendC::Reg;

    MaskReg b32AllMask = CreateMask<uint32_t, MaskPattern::ALL>();
    MaskReg b32HalfMask = CreateMask<uint32_t, MaskPattern::H>();
    MaskReg b16AllMask = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg b16HalfMask = CreateMask<uint16_t, MaskPattern::H>();
    MaskReg b16HalfTailMask;
    Xor(b16HalfTailMask, b16AllMask, b16HalfMask, b16AllMask); // 0000...1111...

    RegTensor<int16_t> gIdxB16Reg, tmpI16Reg;
    Duplicate(tmpI16Reg, static_cast<int16_t>(VL_B16), b16HalfMask);             // 128,128,...,0,0,...,0
    Arange(gIdxB16Reg, static_cast<int16_t>(-static_cast<int16_t>(VL_B16) / 2)); //-64,-63,...,0,1,...,63
    Add(gIdxB16Reg, gIdxB16Reg, tmpI16Reg, b16AllMask);                          // 64,65,...,127,0,1,...,63
    RegTensor<uint32_t> u32OneReg;
    Duplicate(u32OneReg, static_cast<uint32_t>(1));
    UnalignRegForLoad uScales;
    RegTensor<uint32_t> scalesInReg, scaleReg0, scaleReg1;
    RegTensor<bfloat16_t> eaReg, eaReg0, eaReg1;
    RegTensor<uint16_t> ebReg, ebReg0, ebReg1, ecReg, ecReg0, ecReg1;
    RegTensor<bfloat16_t> xBF16Reg, biasInReg;
    // 构造Eb和Ec用的索引
    RegTensor<int16_t> rep1IndexB16, rep2IndexB16, rep4IndexB16, rep8IndexB16, tmpIndexB16;
    Arange<int16_t>(rep1IndexB16, 0);                                  // 0,1,2,...,127
    Interleave(rep2IndexB16, tmpIndexB16, rep1IndexB16, rep1IndexB16); // 0,0,1,1,...,63,63
    Interleave(rep4IndexB16, tmpIndexB16, rep2IndexB16, rep2IndexB16); // 0...0,15...15,16...16,...,31...31
    Interleave(rep8IndexB16, tmpIndexB16, rep4IndexB16, rep4IndexB16); // 0...0,...,7...7,8...8,...,15...15
    Duplicate(tmpIndexB16, static_cast<int16_t>(16), b16HalfTailMask); // 0...0,...,0...0,16...16,...,16...16
    Sub(rep4IndexB16, rep4IndexB16, tmpIndexB16, b16AllMask);          // 0...0,...,15...15,0...0,15...15
    Duplicate(tmpIndexB16, static_cast<int16_t>(8), b16HalfTailMask);  // 0...0,...,0...0,8...8,...,8...8
    Sub(rep8IndexB16, rep8IndexB16, tmpIndexB16, b16AllMask);          // 0...0,...,7...7,0...0,7...7
    // 用于提取32位宽元素的低8位、低16位
    RegTensor<uint32_t> u32LowestB8Reg, u32LowestB16Reg;
    Duplicate(u32LowestB8Reg, static_cast<uint32_t>(0xFF));
    Duplicate(u32LowestB16Reg, static_cast<uint32_t>(0xFFFF));
    // 用于提取16位宽元素的低1位
    RegTensor<uint16_t> u16LowestB1Reg;
    Duplicate(u16LowestB1Reg, static_cast<uint16_t>(0b1));

    // 按expandedX的validK行遍历，算完写入y的一行
    for (uint32_t k = 0; k < validK; k++) {
        // scales要用非对齐拷入，因此需要用POST_MODE_UPDATE模式，需要调用LoadUnAlignPre，且需要单独处理尾数据tailElemNum
        __ubuf__ uint32_t* rowScalesAddr = scalesAddr + k * scalesRowElemNumAligned;
        LoadUnAlignPre<uint32_t>(uScales, rowScalesAddr);
        // 处理整块vf数据
        for (uint32_t i = 0; i < rowLoopNum; ++i) {
            // 1. 提取计算scale
            // 拷入scales，每轮Loop只使用scales[0]和scales[1]
            LoadUnAlign<uint32_t, PostLiteral::POST_MODE_UPDATE>(scalesInReg, uScales, rowScalesAddr,
                                                                 SCALES_ADDR_STRIDE);
            // 从scalesInReg[0:2]提取Ea、Eb、Ec
            RegExtractHif4ScalesForB16<false>(scalesInReg, eaReg, ebReg, ecReg, rep4IndexB16, rep8IndexB16,
                                              u32LowestB8Reg, u32LowestB16Reg, u16LowestB1Reg, b16HalfTailMask);
            // 2. 计算y=Cast<bf16, e1m2>(x)*pow(2,Eb+Ec)/EaB16+bias
            // 计算xBF16Reg，xAddr对于每轮k循环偏移rowElemNum/2个fp4x2，即rowElemNum个fp4
            RegDequantizeWithBias<HAS_BIAS>(xAddr + k * rowElemNum / NUM_FP4_IN_BYTE + X_ADDR_STRIDE * i,
                                            biasAddr + k * rowElemNum + VL_B16 * i, b16AllMask, eaReg, ebReg, ecReg,
                                            xBF16Reg);
            // 累加xBF16Reg到yAddr，因为涉及读取累加再写入，mask要用all，否则每轮尾部都被清0（好像清零也没关系）
            // 这里如果h不是64的整数倍的话，要适配用非对齐搬入搬出（目前是要求h为64整数倍）
            RegAccumlateDstAddrWithSrc(yAddr + VL_B16 * i, xBF16Reg, b16AllMask);
        }
        // 处理尾数据, for循环的作用是if
        MaskReg tailElemMask;
        uint32_t remainTailElemNum = tailElemNum;
        for (uint32_t i = 0; i < tailLoopNum; i++) {
            tailElemMask = UpdateMask<bfloat16_t>(remainTailElemNum);
            LoadUnAlign<uint32_t, PostLiteral::POST_MODE_UPDATE>(scalesInReg, uScales, rowScalesAddr,
                                                                 SCALES_ADDR_STRIDE);
            // 从scalesInReg[0]提取Ea、Eb、Ec
            RegExtractHif4ScalesForB16<true>(scalesInReg, eaReg, ebReg, ecReg, rep4IndexB16, rep8IndexB16,
                                             u32LowestB8Reg, u32LowestB16Reg, u16LowestB1Reg, b16HalfTailMask);
            // xAddr对于每轮k循环偏移rowElemNum/2个fp4x2，即rowElemNum个fp4
            RegDequantizeWithBias<HAS_BIAS>(xAddr + k * rowElemNum / NUM_FP4_IN_BYTE + X_ADDR_STRIDE * rowLoopNum,
                                            biasAddr + k * rowElemNum + VL_B16 * rowLoopNum, tailElemMask, eaReg, ebReg,
                                            ecReg, xBF16Reg);
            RegAccumlateDstAddrWithSrc(yAddr + VL_B16 * rowLoopNum, xBF16Reg, tailElemMask);
        }
        // 下一轮读写需要等到上一轮的yAddr写完
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
}
#endif

} // namespace MoeFinalizeRoutingV2Regbase

#endif
