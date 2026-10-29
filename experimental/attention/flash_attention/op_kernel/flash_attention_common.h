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
 * \file flash_attention_score_common.h
 * \brief
 */
#ifndef FLASH_ATTENTION_SCORE_COMMON_H
#define FLASH_ATTENTION_SCORE_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stdarg.h"

using namespace AscendC;
using namespace matmul;

constexpr MatmulConfig CFG_EXCEED = GetNormalConfig(true);
constexpr static uint32_t GM_DOUBLE_BUFFER = 2;
constexpr static uint64_t BLOCK_BYTE = 32;
constexpr static int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
constexpr static int32_t SOFTMAX_K_ALIGNED_SIZE = 64;
constexpr static uint64_t DATACOPYPAD_PADDING_VALUE_ZERO = 0;
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP32 = 0xFF7FFFFF;
constexpr static uint32_t NEGATIVE_MIN_VAULE_FP16 = 0xFBFF;
constexpr static uint32_t POSITIVE_MAX_VALUE_FP32 = 0x7F7FFFFF;
constexpr static uint32_t POSITIVE_MAX_VALUE_FP16 = 0x7BFF;
constexpr static uint16_t SOFTMAX_CHECK_RES_DEFAULT_VALUE = 0xFFFF;

enum class SparseModeEnum {
    ALL = 0,
    NONE = 1,
    ANY = 2,
    CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    BANDX1 = 6
};

enum class ImplModeEnum {
    AA_HIGH_PRECISION = 0,
    AA_HIGH_PERFORMANCE = 1,
    AA_INVALID_LINE_HIGH_PRECISION = 2
};

enum class AttenMaskCompressMode {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE = 1,
    RIGHT_DOWN_CAUSAL_MODE = 2,
    BAND_MODE = 3
};

// 每个 s2 基本块的 mask 计算档位: 用于跳过 band/causal 内部块(mask 全通)的 mask 搬入与计算
//   NO_NEED_COMPUTE_MODE      : 整块落在有效区内, mask 全通, 无需搬入/计算
//   PRE_ONLY_MODE             : 仅左窗(pre)边切到该块, 只搬 pre mask 做一次
//   CAUSAL_OR_NEXT_ONLY_MODE  : 仅对角/右(next)边切到该块, 只搬主 mask 做一次
//   PRE_AND_NEXT_MODE         : pre 与 next 两侧都切到该块, 需搬两次
enum class AttenMaskComputeMode {
    NO_NEED_COMPUTE_MODE = 0,
    PRE_ONLY_MODE = 1,
    CAUSAL_OR_NEXT_ONLY_MODE = 2,
    PRE_AND_NEXT_MODE = 3
};

// All realsize do not means real input size, instead of meaning
// current loop process size
struct SplitExtraInfo {
    int64_t s2StartIdx;
    int64_t s2EndIdx;
    int64_t s2LoopCount;
    int64_t s1oIdx;
    int64_t s1TileIdx;
    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t taskId;
    int64_t s1RealSize;
    int64_t s2RealSize;
    int64_t s2AlignedSize;
    int64_t vec1S1BaseSize;
    int64_t vec1S1RealSize;
    int64_t vec2S1BaseSize;
    int64_t vec2S1RealSize;
    int64_t realSplitN;
    int64_t s2LoopLimit;
    int64_t multiCoreInnerIdx;
    int64_t qCoreOffset;
    int64_t s1SeqSize;
    int64_t s2SeqSize;
    int64_t s1SizeAcc;
    int64_t s2SizeAcc;
    int64_t attenB1SSOffset;
};

namespace math {
template <typename T>
__aicore__ inline T Ceil(T a, T b)
{
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline T Align(T a, T b)
{
#ifdef __CCE_KT_TEST__
    assert(b != 0 && "b cannot be 0");
#endif
    return (a + b - 1) / b * b;
}
} // namespace math

template <typename T>
__aicore__ inline void GetExtremeValue(T& negativeScalar, T& positiveScalar)
{
    if constexpr (IsSameType<T, float>::value) {
        uint32_t tmp1 = NEGATIVE_MIN_VAULE_FP32;
        uint32_t tmp2 = POSITIVE_MAX_VALUE_FP32;
        negativeScalar = *((float*)&tmp1);
        positiveScalar = *((float*)&tmp2);
    } else {
        uint16_t tmp1 = NEGATIVE_MIN_VAULE_FP16;
        uint16_t tmp2 = POSITIVE_MAX_VALUE_FP16;
        negativeScalar = *((half*)&tmp1);
        positiveScalar = *((half*)&tmp2);
    }
}

template <typename T>
__aicore__ inline void DataCopy2D(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal, const uint32_t d0,
                                  const uint32_t d1, const uint32_t orgD1, uint64_t paddingValue = 0)
{
    if (d1 % (BLOCK_BYTE / sizeof(T)) == 0 && orgD1 % (BLOCK_BYTE / sizeof(T)) == 0) {
        auto d1Blocks = math::Ceil(d1 * sizeof(T), BLOCK_BYTE);
        auto orgD1Blocks = math::Ceil(orgD1 * sizeof(T), BLOCK_BYTE);
        DataCopyParams copyParams(d0, d1Blocks, orgD1Blocks - d1Blocks, 0);
        DataCopy(dstLocal, srcGlobal, copyParams);
    } else {
        auto d1Bytes = d1 * sizeof(T);
        auto d1Aligned = math::Align(static_cast<int64_t>(d1), static_cast<int64_t>(BLOCK_BYTE / sizeof(T)));
#if __CCE_KT_TEST__
        assert(d0 <= 65535 && "count must in [0, 4095]");
        assert(d1Bytes <= 65535 && "len must in [0, 65535]");
#endif
        DataCopyParams copyParams(static_cast<uint16_t>(d0), static_cast<uint16_t>(d1Bytes),
                                  orgD1 * sizeof(T) - d1Bytes, 0);
        DataCopyPadParams padParams(true, 0, static_cast<uint8_t>(d1Aligned - d1), paddingValue);
        DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
    }
}

__aicore__ inline bool IsBasicBlockInSoftMax(int32_t srcM, int32_t srcK)
{
    return srcM % SOFTMAX_M_ALIGNED_SIZE == 0 && srcK % SOFTMAX_K_ALIGNED_SIZE == 0;
}

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename INPUT_T, typename T = INPUT_T, bool isBasicBlock = false>
class FlashAttentionCommon {
public:
    __aicore__ inline FlashAttentionCommon();
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* attenMask, __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum,
                                __gm__ uint8_t* softmaxOut, __gm__ uint8_t* attentionOut, __gm__ uint8_t* workspace,
                                const FlashAttentionGeneralTilingData* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void Process();

    // define matmul
    using a1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b1Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T, true>;
    using bias1Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c1Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::Matmul<a1Type, b1Type, c1Type, bias1Type, CFG_EXCEED> bmm1;

    using c1NzType = MatmulType<TPosition::GM, CubeFormat::NZ, T>;
    matmul::Matmul<a1Type, b1Type, c1NzType, bias1Type, CFG_EXCEED> bmm1Nz;
    // define batchmatmul
    using a2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using b2Type = MatmulType<TPosition::GM, CubeFormat::ND, INPUT_T>;
    using bias2Type = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using c2Type = MatmulType<TPosition::GM, CubeFormat::ND, T>;
    matmul::Matmul<a2Type, b2Type, c2Type, bias2Type, CFG_EXCEED> bmm2;

protected:
    template <typename T1, typename T2>
    __aicore__ inline T1 CeilDiv(T1 a, T2 b)
    {
        if (b == 0) {
            return 0;
        }
        return (a + b - 1) / b;
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 Max(T1 a, T2 b)
    {
        return (a > b) ? (a) : (b);
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 Min(T1 a, T2 b)
    {
        return (a > b) ? (b) : (a);
    };

    int32_t blockIdx;
    const FlashAttentionGeneralTilingData* __restrict tilingData;

    // tiling index
    int64_t boIdx;
    int64_t n2oIdx;
    int64_t goIdx;
    int64_t s1oIdx;
    int64_t s1TileIdx;
    // sparse index
    int64_t s2StartIdx;
    int64_t s2EndIdx;
    int64_t s2LoopStart;

    // 轴计算
    int64_t gD;
    int64_t n2G;
    int64_t n2D;
    int64_t n2GD;
    int64_t n2outD;
    int64_t n2GoutD;

    // 轴 & base size 计算
    int64_t s1BaseN2GD;
    int64_t s2BaseN2D;
    int64_t s2BaseNratioSize;
    int64_t s2BaseNratioN2D;

    // MMA 计算
    int64_t mm1Ka;
    int64_t mm1Kb;
    int64_t mm2Kb;

    // global inputs & outputs
    GlobalTensor<INPUT_T> queryGm;
    GlobalTensor<INPUT_T> keyGm;
    GlobalTensor<T> workspaceGm;
    GlobalTensor<INPUT_T> valueGm;
    GlobalTensor<INPUT_T> attentionOutGm;
    GlobalTensor<float> maxLogitsGm;
    GlobalTensor<float> softmaxMaxGm;
    GlobalTensor<float> softmaxSumGm;
    GlobalTensor<float> lseOutGm;
    GlobalTensor<float> maxLogitsWorkspaceGm;
    GlobalTensor<uint8_t> dropMaskGm;
    GlobalTensor<INPUT_T> attenMaskGm;
    GlobalTensor<uint8_t> attenMaskGmInt;
    GlobalTensor<uint8_t> prmMaskGm;
    GlobalTensor<INPUT_T> sinkGm;

    // 资源分配
    TBuf<> maskTBufPing;
    TBuf<> maskTBufPong;
    TBuf<> tmpTBuf;
    TBuf<> stage1PingBuf;
    TBuf<> stage1PongBuf;
    TBuf<> stage2TBuf;
    TBuf<> softmaxSumPingBuf;
    TBuf<> softmaxSumPongBuf;
    TBuf<> softmaxExpPingBuf;
    TBuf<> softmaxExpPongBuf;
    TBuf<> softmaxMaxBuf;
    TBuf<> maxLogitsBuf;
    TBuf<> sinkBuf;
    TBuf<> postBuf;
    TBuf<> commonTBuf; // common的复用空间

    TPipe* pipe;
    LocalTensor<T> softmaxExpUb;
    GlobalTensor<T> mm1ResPing;
    GlobalTensor<T> mm1ResPong;
    GlobalTensor<T> mm2Res[2];
    GlobalTensor<T> vec2ResPing;
    GlobalTensor<T> vec2ResPong;
    GlobalTensor<INPUT_T> stage1ResPing;
    GlobalTensor<INPUT_T> stage1ResPong;

    // 1 bolck = 32 bytes
    // 1 byte  = 8bits
    // fp32BaseSize = 32bytes / sizeof(fp32) = 8
    // fp32BlockSize = 256bytes / sizeof(fp32) = 64
    constexpr static int32_t blockBytes = 32;
    constexpr static int32_t byteBitRatio = 8;
    constexpr static int32_t fp32BlockSize = 64;
    constexpr static int32_t fp32BaseSize = 8;
    constexpr static int32_t repeatMaxBytes = 256;
    constexpr static int32_t repeatMaxTimes = 255;
    constexpr static int32_t postBytes = 180 * 1024;

    // 0级接口的block间隔范围需要满足32B对齐
    constexpr static int32_t blockSize = blockBytes / sizeof(T);

    bool hasAttenMask;
    bool hasPrmMask;
    bool hasSink;
    bool isInputTAttenMask;
    bool needMaxLogits;
    int32_t repeatMaxSize;

    T negativeFloatScalar;
    T positiveFloatScalar;

    // base tiling params
    uint32_t s1BaseSize;
    uint32_t s2BaseSize;
    uint32_t dSize;
    uint32_t vDsize;
    uint32_t vDsizeAlign;

    // attenmask
    int64_t attenMaskBN2GS1S2 = 0;
    int64_t attenMaskBS1S2 = 1;
    int64_t attenMaskS1S2 = 2;
    int64_t attenMaskFlattenedS1S2 = 3;

protected:
    __aicore__ inline void InitInput(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                     __gm__ uint8_t* attenMask, __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum,
                                     __gm__ uint8_t* softmaxOut, __gm__ uint8_t* attentionOut,
                                     __gm__ uint8_t* workspace,
                                     const FlashAttentionGeneralTilingData* __restrict tiling, TPipe* tPipe);

    __aicore__ inline void InitBuffer();

    __aicore__ inline int32_t Align(int32_t shape);

    __aicore__ inline void BoolCopyIn(LocalTensor<uint8_t>& dstTensor, GlobalTensor<uint8_t>& srcTensor, int64_t offset,
                                      int32_t s1Size, int32_t s2Size, int64_t totalS2Size, SplitExtraInfo& extraInfo);

    __aicore__ inline int64_t ComputeOffsetForCausal(const int64_t& delta, const uint32_t& s1BaseSize,
                                                     const uint32_t& s2BaseSize, const uint32_t& attenMaskS2Size);

    __aicore__ inline void ComputeAttenMask(SplitExtraInfo& extraInfo, LocalTensor<T>& bmm1ResUb,
                                            const uint8_t maskType);
};

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline FlashAttentionCommon<INPUT_T, T, isBasicBlock>::FlashAttentionCommon()
{}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int32_t FlashAttentionCommon<INPUT_T, T, isBasicBlock>::Align(int32_t size)
{
    int32_t alignFactor = 16;
    int32_t alignedSize = this->CeilDiv(size, alignFactor) * alignFactor;
    return alignedSize;
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionCommon<INPUT_T, T, isBasicBlock>::BoolCopyIn(LocalTensor<uint8_t>& dstTensor,
                                                                                  GlobalTensor<uint8_t>& srcTensor,
                                                                                  int64_t offset, int32_t s1Size,
                                                                                  int32_t s2Size, int64_t totalS2Size,
                                                                                  SplitExtraInfo& extraInfo)
{
    int32_t alignedS2Size = this->CeilDiv(s2Size, this->blockBytes) * this->blockBytes;
    uint32_t shapeArray[] = {static_cast<uint32_t>(s1Size), static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.dstStride = 0;
    if (totalS2Size % this->blockBytes == 0) {
        dataCopyParams.blockLen = alignedS2Size / this->blockBytes;
        dataCopyParams.srcStride = (totalS2Size - alignedS2Size) / this->blockBytes;
        DataCopy(dstTensor, srcTensor[offset], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size;
        dataCopyParams.srcStride = totalS2Size - s2Size;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = alignedS2Size - s2Size;
        dataCopyPadParams.paddingValue = 0;
        DataCopyPad(dstTensor, srcTensor[offset], dataCopyParams, dataCopyPadParams);
    }
}

// align mask to 2048*2048
template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline int64_t FlashAttentionCommon<INPUT_T, T, isBasicBlock>::ComputeOffsetForCausal(
    const int64_t& delta, const uint32_t& s1BaseSize, const uint32_t& s2BaseSize, const uint32_t& attenMaskS2Size)
{
    if (delta <= 0) {
        return this->Min(-1 * delta, s1BaseSize);
    } else {
        return this->Min(delta, s2BaseSize) * attenMaskS2Size;
    }
}

template <typename INPUT_T, typename T, bool isBasicBlock>
__aicore__ inline void FlashAttentionCommon<INPUT_T, T, isBasicBlock>::ComputeAttenMask(SplitExtraInfo& extraInfo,
                                                                                        LocalTensor<T>& bmm1ResUb,
                                                                                        const uint8_t maskType)
{
    if (!this->hasAttenMask) {
        return;
    }

    LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();
    int32_t alignedS2Size = this->CeilDiv(extraInfo.s2RealSize, this->blockBytes) * this->blockBytes;
    uint32_t shapeArray[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize), static_cast<uint32_t>(alignedS2Size)};
    attenMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    attenMaskUb.SetSize(extraInfo.vec1S1RealSize * alignedS2Size);
    bmm1ResUb.SetSize(extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);
    LocalTensor<uint8_t> apiTmpBuffer = commonTBuf.template Get<uint8_t>();
    // 待实现batch
    SelectWithBytesMaskShapeInfo shapeInfo;
    shapeInfo.firstAxis = extraInfo.vec1S1RealSize;
    shapeInfo.srcLastAxis = extraInfo.s2AlignedSize;
    shapeInfo.maskLastAxis = alignedS2Size;

    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

    if (maskType == 0) {
        SelectWithBytesMask(bmm1ResUb, bmm1ResUb, this->negativeFloatScalar, attenMaskUb, apiTmpBuffer, shapeInfo);
    } else {
        SelectWithBytesMask(bmm1ResUb, this->negativeFloatScalar, bmm1ResUb, attenMaskUb, apiTmpBuffer, shapeInfo);
    }
    return;
}

#endif // FLASH_ATTENTION_SCORE_COMMON_H
