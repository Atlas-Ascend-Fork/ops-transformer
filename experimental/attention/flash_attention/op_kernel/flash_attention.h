/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

/*!
 * \file fusion_attention_var_len_score.h
 * \brief
 */

#ifndef FLASH_ATTENTION_VAR_LEN_H
#define FLASH_ATTENTION_VAR_LEN_H

#include "flash_attention_common.h"

using namespace AscendC;

// INPUT_T - means data type for input
// T       - means data type when calc
template <typename INPUT_T, typename T>
class FlashAttention : public FlashAttentionCommon<INPUT_T, T> {
public:
    __aicore__ inline FlashAttention(){};

    __aicore__ inline void UnpackInit(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                      __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKv,
                                      __gm__ uint8_t* attenMask, __gm__ uint8_t* prmMask, __gm__ uint8_t* sink,
                                      __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum, __gm__ uint8_t* lseOut,
                                      __gm__ uint8_t* attentionOut, __gm__ uint8_t* maxLogits,
                                      __gm__ uint8_t* workspace,
                                      const FlashAttentionGeneralTilingData* __restrict tiling, TPipe* tPipe);

    __aicore__ inline void Process();

protected:
    // init and calculate index、offset
    __aicore__ inline void InitInput(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                     __gm__ uint8_t* attenMask, __gm__ uint8_t* prmMask, __gm__ uint8_t* sink,
                                     __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum, __gm__ uint8_t* lseOut,
                                     __gm__ uint8_t* attentionOut, __gm__ uint8_t* maxLogits, __gm__ uint8_t* workspace,
                                     const FlashAttentionGeneralTilingData* __restrict tiling, TPipe* tPipe);

    __aicore__ inline void InitBuffer();

    __aicore__ inline void GetS1LoopRange(int64_t& multiCoreInnerOffset, int64_t& multiCoreInnerLimit);

    __aicore__ inline void GetS2LoopRange();

    __aicore__ inline void ComputeConstexpr();

    __aicore__ inline void ComputeAxisIdx(int64_t multiCoreInnerIdx);

    __aicore__ inline void CalS1OuterSize(int64_t offset);

    __aicore__ inline void GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t& actualSeqQlen, int64_t& actualSeqKvlen);

    __aicore__ inline void ComputeBmm1Tail(SplitExtraInfo& extraInfo);

    __aicore__ inline int64_t ComputeOffsetForNoCompress(SplitExtraInfo& extraInfo, int64_t loopIdx);

    __aicore__ inline void SetExtraInfo(SplitExtraInfo& extraInfo, int64_t taskId, int64_t s2LoopCount,
                                        int64_t s2LoopLimit, int64_t multiCoreInnerIdx);

    // bmm set
    __aicore__ inline void IterateBmm1(SplitExtraInfo& extraInfo);

    __aicore__ inline void WaitBmm1Result();

    __aicore__ inline void GetBmm1Result(SplitExtraInfo& extraInfo, LocalTensor<T>& bmm1ResUb, int64_t loopIdx);

    __aicore__ inline void IterateBmm2(SplitExtraInfo& extraInfo);

    __aicore__ inline void WaitBmm2Result();

    // vector copyin&out&compute

    __aicore__ inline void SoftMaxCompute(SplitExtraInfo& extraInfo, LocalTensor<T>& srcTensor, int64_t loopIdx);

    __aicore__ inline void Bmm2DataCopyOut(SplitExtraInfo& extraInfo, int64_t vec2s1oIdx);

    __aicore__ inline void SoftmaxSumDataCopyOut(SplitExtraInfo& extraInfo, int64_t vec2s1oIdx);

    __aicore__ inline void SoftmaxMaxDataCopyOut(SplitExtraInfo& extraInfo);

    __aicore__ inline void LseComputeAndCopy(SplitExtraInfo& extraInfo);

    __aicore__ inline void CopyInAttenMask(SplitExtraInfo& extraInfo, int64_t loopIdx, int64_t maskOffset);

    __aicore__ inline int64_t ComputeAttenMaskOffset(SplitExtraInfo& extraInfo, int64_t loopIdx);

    __aicore__ inline void GetAttenMaskComputeMode(int64_t delta, int64_t deltaPre, SplitExtraInfo& extraInfo);

    __aicore__ inline void CopyInPrmMask(SplitExtraInfo& extraInfo);

    __aicore__ inline void ComputePrmMask(SplitExtraInfo& extraInfo, LocalTensor<T>& bmm1ResUb);

    __aicore__ inline void CopyInSink();

    __aicore__ inline void ComputeAndCopyMaxLogits(const SplitExtraInfo& extraInfo);

    __aicore__ inline void Bit2Int8CopyIn(LocalTensor<uint8_t>& dstTensor, GlobalTensor<uint8_t>& srcTensor,
                                          int64_t offset, int32_t s1Size, int32_t s2Size, int32_t batchSize,
                                          SplitExtraInfo& extraInfo);

    __aicore__ inline void Bmm2ResultDiv(SplitExtraInfo& extraInfo, int64_t vec2s1oIdx);

    __aicore__ inline void Bmm2ResultMul(SplitExtraInfo& extraInfo, LocalTensor<T>& bmm2ResUb, int64_t vec2s1oIdx);

    __aicore__ inline void ProcessVec1(SplitExtraInfo& extraInfo);

    __aicore__ inline void ProcessVec2(SplitExtraInfo& extraInfo);

    __aicore__ inline void PostProcess();

    __aicore__ inline void DataCopyIn(SplitExtraInfo& extraInfo, LocalTensor<INPUT_T>& dstTensor,
                                      GlobalTensor<INPUT_T>& srcTensor, int64_t offset, int32_t s1Size, int32_t s2Size,
                                      int32_t actualS2Len);
    GM_ADDR actualSeqQlenAddr;
    GM_ADDR actualSeqKvlenAddr;
    GlobalTensor<uint64_t> actualSeqGm;
    GlobalTensor<uint64_t> actualSeqKvGm;
    // Unpack 用参数
    uint64_t maxSeqS1Len = 0;
    uint64_t s1OuterSizeAcc;
    uint64_t s1SizeAcc;
    uint64_t s2SizeAcc;
    uint64_t attenB1SSOffset;
    int64_t attenMaskOffsetPre;
    int64_t attenMaskOffset;
    // 当前 s2 基本块的 mask 计算档位, 默认 CAUSAL_OR_NEXT_ONLY_MODE(单次主 mask, 兼容 NO_COMPRESS)
    AttenMaskComputeMode attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
    int64_t attenMaskTT = 99;
    int64_t s1SizeSum = 0;
    int64_t s2SizeSum = 0;
    int64_t accS1TileNum;
    uint64_t splitS2sizeLimit = 1024;
};

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::UnpackInit(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* actualSeqLengthsKv, __gm__ uint8_t* attenMask, __gm__ uint8_t* prmMask, __gm__ uint8_t* sink,
    __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum, __gm__ uint8_t* lseOut, __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* maxLogits, __gm__ uint8_t* workspace, const FlashAttentionGeneralTilingData* __restrict tiling,
    TPipe* tPipe)
{
    this->InitInput(query, key, value, attenMask, prmMask, sink, softmaxMax, softmaxSum, lseOut, attentionOut,
                    maxLogits, workspace, tiling, tPipe); // gm设置
    actualSeqGm.SetGlobalBuffer((__gm__ uint64_t*)actualSeqLengths);
    actualSeqKvGm.SetGlobalBuffer((__gm__ uint64_t*)actualSeqLengthsKv);
    this->ComputeConstexpr();
    this->InitBuffer();

    LocalTensor<T> apiTmpBuffer = this->commonTBuf.template Get<T>();
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510))
    DropOutBitModeInit(apiTmpBuffer);
#endif

    actualSeqQlenAddr = actualSeqLengths;
    actualSeqKvlenAddr = actualSeqLengthsKv;
    return;
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::InitInput(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* attenMask,
    __gm__ uint8_t* prmMask, __gm__ uint8_t* sink, __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum,
    __gm__ uint8_t* lseOut, __gm__ uint8_t* attentionOut, __gm__ uint8_t* maxLogits, __gm__ uint8_t* workspace,
    const FlashAttentionGeneralTilingData* __restrict tiling, TPipe* tPipe)
{
    this->blockIdx = GetBlockIdx();
    this->pipe = tPipe;
    this->repeatMaxSize = this->repeatMaxBytes / sizeof(T);

    // init tiling data
    this->tilingData = tiling;
    this->s1BaseSize = this->tilingData->coreParams.s1BaseSize;
    this->s2BaseSize = this->tilingData->coreParams.s2BaseSize;
    this->dSize = this->tilingData->inputParams.dSize;
    this->vDsize = this->tilingData->inputParams.vDsize;
    this->vDsizeAlign = this->CeilDiv(this->vDsize, 16) * 16;

    this->s1SizeSum = this->tilingData->inputParams.accumS1;
    this->s2SizeSum = this->tilingData->inputParams.accumS2;
    this->needMaxLogits = (this->tilingData->inputParams.outputMaxLogits == 1);

    // init global buffer
    this->queryGm.SetGlobalBuffer((__gm__ INPUT_T*)query);
    this->keyGm.SetGlobalBuffer((__gm__ INPUT_T*)key);
    this->valueGm.SetGlobalBuffer((__gm__ INPUT_T*)value);

    this->attenMaskGmInt.SetGlobalBuffer((__gm__ uint8_t*)attenMask);
    this->prmMaskGm.SetGlobalBuffer((__gm__ uint8_t*)prmMask);
    this->sinkGm.SetGlobalBuffer((__gm__ INPUT_T*)sink);
    this->softmaxMaxGm.SetGlobalBuffer((__gm__ float*)softmaxMax);
    this->softmaxSumGm.SetGlobalBuffer((__gm__ float*)softmaxSum);
    this->lseOutGm.SetGlobalBuffer((__gm__ float*)lseOut);
    this->attentionOutGm.SetGlobalBuffer((__gm__ INPUT_T*)attentionOut);
    this->maxLogitsGm.SetGlobalBuffer((__gm__ float*)maxLogits);

    // 补齐到512， 统一按T处理
    int64_t mm1ResultSize = this->s1BaseSize * this->s2BaseSize; // 128 * 128
    int64_t mmNRatioOffset =
        this->CeilDiv(mm1ResultSize * this->tilingData->coreParams.nRatio, 128) * 128 * sizeof(T); // 128 * 1024
    int64_t mm2ResultSize = this->s1BaseSize * this->vDsizeAlign;                                  // 128 * 64(128)
    int64_t mm2Offset = this->CeilDiv(mm2ResultSize, 128) * 128 * 4;
    int64_t bmm1AndVec1Ratio = GM_DOUBLE_BUFFER;
    int64_t vector1OffsetPing = 0;
    int64_t vector1OffsetPong = mmNRatioOffset;

    int64_t totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * GM_DOUBLE_BUFFER;
    if (this->vDsizeAlign > 64) {
        totalOffset = mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2 * GM_DOUBLE_BUFFER;
    }

    // bmm1Result， 占用mmNRatioOffset空间
    this->mm1ResPing.SetGlobalBuffer((__gm__ T*)(workspace + this->blockIdx * totalOffset));
    this->mm1ResPong.SetGlobalBuffer((__gm__ T*)(workspace + this->blockIdx * totalOffset + mmNRatioOffset));

    // stage1Result， 占用mmNRatioOffset空间
    this->stage1ResPing.SetGlobalBuffer(
        (__gm__ INPUT_T*)(workspace + this->blockIdx * totalOffset + vector1OffsetPing));
    this->stage1ResPong.SetGlobalBuffer(
        (__gm__ INPUT_T*)(workspace + this->blockIdx * totalOffset + vector1OffsetPong));

    // bmm2Result， 占用mmOffset空间
    this->mm2Res[0].SetGlobalBuffer(
        (__gm__ T*)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio));
    this->mm2Res[1].SetGlobalBuffer(
        (__gm__ T*)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset));

    // vec2阶段。占用mmOffset空间， 仅在D轴大于64的情况下出现
    if (this->vDsizeAlign > 64) {
        this->vec2ResPing.SetGlobalBuffer(
            (__gm__ T*)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 2));
        this->vec2ResPong.SetGlobalBuffer(
            (__gm__ T*)(workspace + this->blockIdx * totalOffset + mmNRatioOffset * bmm1AndVec1Ratio + mm2Offset * 3));
    }
    this->maxLogitsWorkspaceGm.SetGlobalBuffer(
        (__gm__ T*)(workspace + this->tilingData->multiCoreParams.aivNum * totalOffset));

    this->hasAttenMask = this->tilingData->tensorSizeParams.attenMaskUbSize > 0;
    this->hasPrmMask = this->tilingData->inputParams.prmMaskFlag;
    this->hasSink = this->tilingData->inputParams.sinkFlag;
    this->isInputTAttenMask = this->tilingData->inputParams.attenMaskDataType == 0;
    if constexpr (IsSameType<T, half>::value) {
        this->negativeIntScalar = NEGATIVE_MIN_VAULE_FP16;
    }
    GetExtremeValue(this->negativeFloatScalar, this->positiveFloatScalar);
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::ComputeConstexpr()
{
    this->gD = this->tilingData->inputParams.gSize * this->dSize;
    this->n2D = this->tilingData->inputParams.n2Size * this->dSize;
    this->n2outD = this->tilingData->inputParams.n2Size * this->vDsize;
    this->n2G = this->tilingData->inputParams.n2Size * this->tilingData->inputParams.gSize;
    this->n2GD = this->tilingData->inputParams.n2Size * this->gD;
    this->n2GoutD = this->n2outD * this->tilingData->inputParams.gSize;

    // 计算切分轴的乘积
    this->s2BaseN2D = this->s2BaseSize * this->n2D;
    this->s2BaseNratioSize = this->s2BaseSize * this->tilingData->coreParams.nRatio;

    // layoutmm1Ka(bs)ND
    this->s1BaseN2GD = this->s1BaseSize * this->n2GD;
    this->s2BaseNratioN2D = this->s2BaseN2D * this->tilingData->coreParams.nRatio;

    // layout(bs)ND
    this->mm1Ka = this->n2GD;
    this->mm1Kb = this->n2D;
    this->mm2Kb = this->n2outD;

    this->accS1TileNum = this->tilingData->multiCoreParams.totalSize / this->n2G;
}

// used for varlen db init with double buffer setting
template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::InitBuffer()
{
    uint64_t stage1Size = 8 * 1024;
    uint64_t stage1PongSize = 8 * 1024;
    uint64_t stage2Size = 64 * 128;

    // 可选输入的buffer空间，保持和stage1处理的size一致
    this->pipe->InitBuffer(this->maskTBufPing, stage1Size);              // 可以给attenmask 8k(同dropmask共用)
    this->pipe->InitBuffer(this->maskTBufPong, stage1Size);              // 可以给prmMask 8k(独占)
    this->pipe->InitBuffer(this->tmpTBuf, stage1Size * sizeof(INPUT_T)); // pse 16k

    this->pipe->InitBuffer(this->stage1PingBuf, stage2Size * sizeof(T)); // t.a 32k
    this->pipe->InitBuffer(this->stage2TBuf, stage2Size * sizeof(T));    // t.c 32k
    this->pipe->InitBuffer(this->commonTBuf, stage2Size * sizeof(T));    // t.b 32k

    this->pipe->InitBuffer(this->softmaxSumPingBuf, this->s1BaseSize * this->blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxSumPongBuf, this->s1BaseSize * this->blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxMaxBuf, this->s1BaseSize * this->blockBytes);     // 4k
    this->pipe->InitBuffer(this->softmaxExpPingBuf, this->s1BaseSize * this->blockBytes); // 4k
    this->pipe->InitBuffer(this->softmaxExpPongBuf, this->s1BaseSize * this->blockBytes); // 4k

    this->pipe->InitBuffer(this->stage1PongBuf, stage1Size * sizeof(T)); // i.a 32k
    if (this->needMaxLogits) {
        this->pipe->InitBuffer(this->maxLogitsBuf, this->fp32BaseSize * sizeof(T)); // 32Byte
    }
    if (this->hasSink) {
        uint64_t hasSinkSize = 512 * 3;
        this->pipe->InitBuffer(this->sinkBuf, hasSinkSize * sizeof(INPUT_T)); // 3k
    }
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::GetSeqQlenKvlenByBoidx(int64_t boIdx, int64_t& actualSeqQlen,
                                                                          int64_t& actualSeqKvlen)
{
    // begin with 0
    actualSeqQlen = ((__gm__ int64_t*)actualSeqQlenAddr)[boIdx + 1] - ((__gm__ int64_t*)actualSeqQlenAddr)[boIdx];
    actualSeqKvlen = ((__gm__ int64_t*)actualSeqKvlenAddr)[boIdx + 1] - ((__gm__ int64_t*)actualSeqKvlenAddr)[boIdx];

    return;
}

// 初始化s1方向上的累加值
template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::CalS1OuterSize(int64_t offset)
{
    int64_t actualS1Outersize = 0;
    // 用于取actualS1Len下标
    this->boIdx = 0;
    this->s1OuterSizeAcc = 0;
    this->attenB1SSOffset = 0;
    this->s1SizeAcc = 0;
    this->s2SizeAcc = 0;
    this->s1TileIdx = 0;

    int64_t actualS1Len;
    int64_t actualS2Len;
    for (uint64_t i = 0; i < this->tilingData->inputParams.bSize; i++) {
        GetSeqQlenKvlenByBoidx(i, actualS1Len, actualS2Len);
        actualS1Outersize += (this->CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
        if (offset >= actualS1Outersize) {
            this->s1OuterSizeAcc = actualS1Outersize;
            this->s1SizeAcc += actualS1Len;
            this->s2SizeAcc += actualS2Len;
            this->s1TileIdx += this->CeilDiv(actualS1Len, this->s1BaseSize);
            this->attenB1SSOffset += actualS1Len * actualS2Len;
            this->boIdx++;
        } else {
            break;
        }
    }
    return;
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::ComputeAxisIdx(int64_t multiCoreInnerIdx)
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len,
                           actualS2Len); // boIdx代表当前正在处理的这句话；s1SizeAcc等代表这句话之前的offset；
    int64_t actualS1Outersize = this->s1OuterSizeAcc + (this->CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
    while (multiCoreInnerIdx >= actualS1Outersize) {
        this->s1OuterSizeAcc = actualS1Outersize;
        this->s1SizeAcc += actualS1Len;
        this->s2SizeAcc += actualS2Len;
        this->s1TileIdx += this->CeilDiv(actualS1Len, this->s1BaseSize);
        this->attenB1SSOffset += actualS1Len * actualS2Len;
        this->boIdx++;
        GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
        actualS1Outersize = this->s1OuterSizeAcc + (this->CeilDiv(actualS1Len, this->s1BaseSize) * this->n2G);
    }
    // 计算轴的idx
    int64_t tmpS1Outersize = this->CeilDiv(actualS1Len, this->s1BaseSize);
    actualS1Outersize = multiCoreInnerIdx - this->s1OuterSizeAcc;
    this->n2oIdx = actualS1Outersize / tmpS1Outersize / this->tilingData->inputParams.gSize;
    this->goIdx = actualS1Outersize / tmpS1Outersize % this->tilingData->inputParams.gSize;
    this->s1oIdx = actualS1Outersize % tmpS1Outersize;
}

// sparse functions
template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::GetS1LoopRange(int64_t& multiCoreInnerOffset,
                                                                  int64_t& multiCoreInnerLimit)
{
    // sparse场景下负载均衡后每个核获取的结果
    multiCoreInnerOffset = this->tilingData->multiCoreParams.sparseStartIdx[this->blockIdx];
    if (likely((this->tilingData->multiCoreParams.coreNum - 1) > this->blockIdx)) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.sparseStartIdx[this->blockIdx + 1];
    } else {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::GetS2LoopRange()
{
    int64_t actualS1Len;
    int64_t actualS2Len;
    GetSeqQlenKvlenByBoidx(this->boIdx, actualS1Len, actualS2Len);
    int64_t offset = actualS2Len - actualS1Len;
    if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::CAUSAL)) { // 下三角
        this->s2StartIdx = 0;
        this->s2EndIdx = this->Min((this->s1oIdx + 1) * this->s1BaseSize + offset, actualS2Len);
    } else if (this->tilingData->inputParams.sparseType == static_cast<uint8_t>(SparseModeEnum::BAND) or
               this->tilingData->inputParams.sparseType ==
                   static_cast<uint8_t>(SparseModeEnum::BANDX1)) { // 对角线往外扩散场景, s1和s2可能不同
        this->s2StartIdx = this->Max(this->s1oIdx * this->tilingData->coreParams.s1BaseSize + offset -
                                         this->tilingData->coreParams.s1SparseValidSize,
                                     0);
        this->s2EndIdx =
            this->Min((this->s1oIdx + 1) * this->s1BaseSize + offset + this->tilingData->coreParams.s2SparseValidSize,
                      actualS2Len);
        // s1baseSize行都无效时，需要将startIdx设置为0，,endIdx设置为S2realSize
        if (this->s2EndIdx - this->s2StartIdx <= 0) {
            this->s2StartIdx = 0;
            this->s2EndIdx = actualS2Len;
        }
    } else { // 其它场景全计算
        this->s2StartIdx = 0;
        this->s2EndIdx = actualS2Len;
    }
    return;
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::DataCopyIn(SplitExtraInfo& extraInfo,
                                                              LocalTensor<INPUT_T>& dstTensor,
                                                              GlobalTensor<INPUT_T>& srcTensor, int64_t offset,
                                                              int32_t s1Size, int32_t s2Size, int32_t actualS2Len)
{
    int32_t dtypeSize = sizeof(INPUT_T);
    int32_t alignedS2Size = this->Align(s2Size);
    uint32_t shapeArray[] = {static_cast<uint32_t>(s1Size), static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = s1Size;
    dataCopyParams.blockLen = this->CeilDiv(s2Size * dtypeSize, this->blockBytes); // 单位32B
    dataCopyParams.dstStride = 0;                                                  // gap
    if (actualS2Len * dtypeSize % this->blockBytes == 0) {
        dataCopyParams.srcStride =
            (actualS2Len * dtypeSize - dataCopyParams.blockLen * this->blockBytes) / this->blockBytes; // srcGap
        DataCopy(dstTensor, srcTensor[offset], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size * dtypeSize; // 单位Byte
        dataCopyParams.srcStride = (actualS2Len * dtypeSize - dataCopyParams.blockLen);
        dataCopyParams.dstStride = 0;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.rightPadding = alignedS2Size - s2Size;
        DataCopyPad(dstTensor, srcTensor[offset], dataCopyParams, dataCopyPadParams);
    }
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::ProcessVec1(SplitExtraInfo& extraInfo)
{
    LocalTensor<T> stage1PingTensor = this->stage1PingBuf.template Get<T>(); // t.a 32k
    LocalTensor<T> stage1PongTensor = this->stage1PongBuf.template Get<T>(); // i.a 32k
    // AllocEventID 从内存池重新分配一个ID，set wait可以分开用，需要手动release
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte2A = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    event_t eventIdVToMte2B = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    // FetchEventID 从内存池取，每次取的都是固定的ID，set和wait需要连着用
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    // realSplitN = s1RealSize / vec1S1BaseSize, s1RealSize 表示每个core 上处理的基本块大小

    this->CopyInPrmMask(extraInfo);
    PipeBarrier<PIPE_V>();
    for (uint32_t loopIdx = 0; loopIdx < extraInfo.realSplitN; loopIdx++) {
        // 这里按比例切分s1，vec上有s1方向一层循环，计算实际处理s1的size
        extraInfo.vec1S1RealSize = extraInfo.vec1S1BaseSize;
        if (loopIdx == extraInfo.realSplitN - 1) {
            extraInfo.vec1S1RealSize = extraInfo.s1RealSize - loopIdx * extraInfo.vec1S1BaseSize;
        }

        // 第一次copy 到 stage1PongTensor，无需同步等待，mm1 结果直接copy 到 stage1PongTensor
        // eventIdVToMte2B 为 MTE2 等 vector 信号，MTE2对应下面 GetBmm1Result 的copy in
        // vector 对应下面muls或者attenmask 计算的，根据eventID 确认
        if (loopIdx > 0) {
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
        }
        this->GetBmm1Result(extraInfo, stage1PongTensor, loopIdx);

        this->attenMaskOffset = this->ComputeAttenMaskOffset(extraInfo, loopIdx);

        // 这里MTE2 不需要等是因为前面 eventIdVToMte2B 的MTE2等V 保障
        // 在指令队列里 GetBmm1Result 要等V 执行完，而 CopyInAttenMask 在 GetBmm1Result 后面
        // 依据 mask 计算档位决定是否搬入及搬入哪份 mask:
        //   NO_NEED  -> 全通, 跳过搬入
        //   PRE_ONLY -> 只需 pre mask, 用 attenMaskOffsetPre
        //   其余      -> 主(causal/next) mask, 用 attenMaskOffset
        if (this->attenMaskComputeMode != AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
            int64_t firstMaskOffset = (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_ONLY_MODE) ?
                                          this->attenMaskOffsetPre :
                                          this->attenMaskOffset;
            this->CopyInAttenMask(extraInfo, loopIdx, firstMaskOffset);
        }

        // V等MTE2，对应下面Mul计算
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        Muls(stage1PingTensor, stage1PongTensor, static_cast<T>(this->tilingData->inputParams.scaleValue),
             extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);

        // prmMask compute
        PipeBarrier<PIPE_V>();
        this->ComputePrmMask(extraInfo, stage1PingTensor);

        PipeBarrier<PIPE_V>();
        // 第一次 mask 计算: PRE_ONLY 用 pre mask(maskType=1), 其余用主 mask(maskType=0)
        // NO_NEED 全通, 跳过计算
        if (this->attenMaskComputeMode != AttenMaskComputeMode::NO_NEED_COMPUTE_MODE) {
            uint8_t firstMaskType = (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_ONLY_MODE) ? 1 : 0;
            this->ComputeAttenMask(extraInfo, stage1PingTensor, firstMaskType);
        }

        // pre 与 next 两侧都切到该块时, 追加一次 pre mask 的搬入与计算(maskType=1)
        if (this->attenMaskComputeMode == AttenMaskComputeMode::PRE_AND_NEXT_MODE) {
            event_t eventIdVToMte2Tmp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2Tmp);
            WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2Tmp);
            this->CopyInAttenMask(extraInfo, loopIdx, this->attenMaskOffsetPre);
            event_t eventIdMte2ToVTmp = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVTmp);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVTmp);
            this->ComputeAttenMask(extraInfo, stage1PingTensor, 1);
        }

        // 设置vector MTE2 同步信号，与前面 GetBmm1Result 处 copyin 对应
        // 这里会将上面 MTE2、vector同步兜底
        if (loopIdx < extraInfo.realSplitN - 1) {
            SetFlag<HardEvent::V_MTE2>(eventIdVToMte2B);
        }

        this->SoftMaxCompute(extraInfo, stage1PingTensor, loopIdx);

        // 下面cast 第一次不需要等搬出，第二次cast计算和搬出用的需要确定 stage1CastTensor 上一轮搬完，因此需要同步判断
        if (loopIdx > 0) {
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
        PipeBarrier<PIPE_V>();

        if constexpr (!IsSameType<T, INPUT_T>::value) {
            LocalTensor<INPUT_T> stage1CastTensor = this->tmpTBuf.template Get<INPUT_T>();
            Cast(stage1CastTensor, stage1PingTensor, RoundMode::CAST_ROUND,
                 extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            // 这里workspace 的pingpong 是cube 和vector间的double buffer
            // stage1 = mm1 + vec1，stage2 = mm2 + vec2
            // stage1里的vec1 和stage2的mm2 可以并行，stage1里的mm1和 stage2的vec2可以并行
            // 实际 vec1和vec2 串行，mm1和mm2串行

            if (extraInfo.taskId % 2 == 0) {
                DataCopy(this->stage1ResPing[loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2AlignedSize],
                         stage1CastTensor, extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);
            } else {
                DataCopy(this->stage1ResPong[loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2AlignedSize],
                         stage1CastTensor, extraInfo.vec1S1RealSize * extraInfo.s2AlignedSize);
            }
        }
        // 对应上面cast 的同步
        if (loopIdx < extraInfo.realSplitN - 1) {
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventIdMte2ToV);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2A);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventIdVToMte2B);

    return;
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::ProcessVec2(SplitExtraInfo& extraInfo)
{
    // 获取缓存bmm2的计算结果
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<T> stage2BufTensor = this->commonTBuf.template Get<T>();
    // vec2S1BaseSize 按照8k 切分基本块，vec2处理数据块为 vec2S1BaseSize * D
    int64_t vec2LoopLimit =
        this->CeilDiv(extraInfo.s1RealSize, extraInfo.vec2S1BaseSize); // s1RealSize=128, vec2S1BaseSize=(64, 128)?

    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    for (int64_t vec2s1oIdx = 0; vec2s1oIdx < vec2LoopLimit; vec2s1oIdx++) {
        // 这里cube 和vector计算有个配比，在s1方向上做切分，varlen场景目前默认8
        // s1RealSize 为当前loop s1 实际大小，vec2S1RealSize 为当前loop vector每轮处理大小
        extraInfo.vec2S1RealSize = extraInfo.vec2S1BaseSize;
        if (vec2s1oIdx == vec2LoopLimit - 1) {
            extraInfo.vec2S1RealSize = extraInfo.s1RealSize - vec2s1oIdx * extraInfo.vec2S1BaseSize;
        }
        int64_t mm2ResCalcSize = extraInfo.vec2S1RealSize * this->vDsizeAlign;
        // cube offset 使用实际D
        int64_t mm2ResOffset = vec2s1oIdx * extraInfo.vec2S1BaseSize * this->vDsize;
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

        // mm2 计算结果跟vector pingpong，stage2BufTensor 当前轮次矩阵乘结果
        // 注意这里db 是taskId 对应s2 方向loop，因为mm2 在s2方向上有切分
        if (likely(this->vDsizeAlign == this->vDsize)) {
            DataCopy(stage2BufTensor, this->mm2Res[extraInfo.taskId % 2][mm2ResOffset], mm2ResCalcSize);
        } else {
            DataCopyParams dataCopyParams;
            DataCopyPadParams dataCopyPadParams;
            dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
            dataCopyParams.dstStride = 0;
            dataCopyParams.srcStride = 0;
            dataCopyParams.blockLen = this->vDsize * sizeof(T);
            dataCopyPadParams.rightPadding = this->vDsizeAlign - this->vDsize;
            dataCopyPadParams.paddingValue = 0;
            if (dataCopyPadParams.rightPadding > this->blockSize) {
                // 8对齐场景，内部vector需要16对齐，我们在data copy的时候需要手动补0
                dataCopyPadParams.rightPadding -= this->blockSize;
                dataCopyParams.dstStride = 1;
                uint32_t dAlign8 = (this->vDsize + 7) / 8 * 8;
                Duplicate<T>(stage2BufTensor[dAlign8], 0, this->blockSize, extraInfo.vec2S1RealSize, 0,
                             this->vDsizeAlign * sizeof(T) / this->blockBytes);
            }
            DataCopyPad(stage2BufTensor, this->mm2Res[extraInfo.taskId % 2][mm2ResOffset], dataCopyParams,
                        dataCopyPadParams);
            // 重新计算，对应 vec 实际offset，这里补了pad
            mm2ResOffset = vec2s1oIdx * extraInfo.vec2S1BaseSize * this->vDsizeAlign;
        }

        // 如果s1 方向loop num > 1, 每次算完一整行s2 才换到下一行。
        // 此时在 s2 loop里还需要上一轮vector 计算结果做矩阵乘reduce 轴上的累加和，对应下面Add 操作
        if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            if (extraInfo.multiCoreInnerIdx % 2 == 0) {
                DataCopy(bmm2ResUb, this->vec2ResPing[mm2ResOffset], mm2ResCalcSize);
            } else {
                DataCopy(bmm2ResUb, this->vec2ResPong[mm2ResOffset], mm2ResCalcSize);
            }
        }

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        // s2 方向第1次计算，不需要累加，直接取 mm2 计算结果，或者一轮计算完直接搬出
        if (unlikely(extraInfo.s2LoopCount == 0)) {
            DataCopy(bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
        } else {
            // 基于flashsoftmax 结果更新当前块结果
            this->Bmm2ResultMul(extraInfo, bmm2ResUb, vec2s1oIdx);
            PipeBarrier<PIPE_V>();
            Add(bmm2ResUb, bmm2ResUb, stage2BufTensor, mm2ResCalcSize);
        }

        // 算完一整行reduceK 即可搬出
        if (extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
            Bmm2ResultDiv(extraInfo, vec2s1oIdx);
            Bmm2DataCopyOut(extraInfo, vec2s1oIdx);
            // if (this->tilingData->inputParams.sinkFlag == 1)
            // {
            //     SoftmaxSumDataCopyOut(extraInfo, vec2s1oIdx);
            // }
            SoftmaxSumDataCopyOut(extraInfo, vec2s1oIdx);
            event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        } else if (vec2LoopLimit > 1) {
            SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            // db vector计算结果，累加和做准备
            if (extraInfo.multiCoreInnerIdx % 2 == 0) {
                DataCopy(this->vec2ResPing[mm2ResOffset], bmm2ResUb, mm2ResCalcSize);
            } else {
                DataCopy(this->vec2ResPong[mm2ResOffset], bmm2ResUb, mm2ResCalcSize);
            }
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }
    return;
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::PostProcess()
{
    if (!this->needMaxLogits) {
        return;
    }

    SyncAll();

    this->pipe->Reset();
    this->pipe->InitBuffer(this->postBuf, this->postBytes); // 180KB
    this->pipe->InitBuffer(this->maxLogitsBuf, 4 * 1024);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    int elementNum = this->postBytes / 2 / sizeof(float);
    LocalTensor<float> postTensor = this->postBuf.template GetWithOffset<float>(elementNum, 0);
    LocalTensor<float> workBuf = this->postBuf.template GetWithOffset<float>(elementNum, this->postBytes / 2);
    LocalTensor<float> maxTensor = this->maxLogitsBuf.template Get<float>();

    for (int core = this->blockIdx; core < this->n2G; core += this->tilingData->multiCoreParams.coreNum) {
        DataCopyPad(postTensor, this->maxLogitsWorkspaceGm[core * this->accS1TileNum],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(this->accS1TileNum * sizeof(float)),
                     /*srcStride*/ 0, /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});

        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

        ReduceMax<float>(maxTensor, postTensor, workBuf, this->accS1TileNum, /*calIndex*/ false);
        PipeBarrier<PIPE_V>();

        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

        DataCopyPad(
            this->maxLogitsGm[core], maxTensor,
            {static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(float)), /*srcStride*/ 0, /*dstStride*/ 0, 0});
        pipe_barrier(PIPE_ALL);
    }
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::Process()
{
    // 新增s1loop每次执行完跳的距离
    int64_t LocalOffset = 1;
    // 确定核内切分起点
    int64_t multiCoreInnerOffset = this->blockIdx * this->tilingData->multiCoreParams.splitFactorSize;
    int64_t multiCoreInnerLimit = multiCoreInnerOffset + this->tilingData->multiCoreParams.splitFactorSize;
    if (this->tilingData->multiCoreParams.totalSize < multiCoreInnerLimit) {
        multiCoreInnerLimit = this->tilingData->multiCoreParams.totalSize;
    }
    // 计算sparse场景下s1的循环范围
    this->GetS1LoopRange(multiCoreInnerOffset, multiCoreInnerLimit);
    // 初始化AxisIdx
    this->CalS1OuterSize(multiCoreInnerOffset);
    SplitExtraInfo extraInfo[3];
    int64_t taskId = 0;
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    CopyInSink();
    for (int64_t multiCoreInnerIdx = multiCoreInnerOffset; multiCoreInnerIdx < multiCoreInnerLimit;
         ++multiCoreInnerIdx) {
        this->ComputeAxisIdx(multiCoreInnerIdx);
        // s2轴循环计数, 支持sparse和非sparse场景
        this->GetS2LoopRange();
        int64_t s2LoopLimit = this->CeilDiv(this->s2EndIdx - this->s2StartIdx, this->s2BaseNratioSize) - 1;
        for (int64_t s2LoopCount = 0; s2LoopCount <= s2LoopLimit; s2LoopCount++) {
            if (taskId > 0) {
                // 对应extraInfo[(i+2)%3]
                this->WaitBmm1Result();
            }
            this->SetExtraInfo(extraInfo[taskId % 3], taskId, s2LoopCount, s2LoopLimit, multiCoreInnerIdx);

            this->IterateBmm1(extraInfo[taskId % 3]); // base MM 大小[128, 1024]

            if (taskId > 0) {
                ProcessVec1(extraInfo[(taskId + 2) % 3]);
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }

            if (taskId > 1) {
                // 对应extraInfo[(i+1)%3]
                this->WaitBmm2Result();
            }

            if (taskId > 0) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
            }

            if (taskId > 1) {
                this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
            }
            taskId++;
        }
    }
    if (taskId >= 1) {
        // 对应extraInfo[(i+2)%3]
        this->WaitBmm1Result();
        this->ProcessVec1(extraInfo[(taskId + 2) % 3]);
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        if (taskId > 1) {
            // 对应extraInfo[(i+1)%3]
            this->WaitBmm2Result();
        }
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        this->IterateBmm2(extraInfo[(taskId + 2) % 3]);
        if (taskId > 1) {
            this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
        }
    }
    taskId++;
    if (taskId >= 2) {
        // 对应extraInfo[(i+1)%3]
        this->WaitBmm2Result();
        pipe_barrier(PIPE_ALL);
        this->ProcessVec2(extraInfo[(taskId + 1) % 3]);
    }
    PostProcess();
};

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::SetExtraInfo(SplitExtraInfo& extraInfo, int64_t taskId,
                                                                int64_t s2LoopCount, int64_t s2LoopLimit,
                                                                int64_t multiCoreInnerIdx)
{
    extraInfo.s2StartIdx = this->s2StartIdx;
    extraInfo.s2EndIdx = this->s2EndIdx;
    extraInfo.s2LoopCount = s2LoopCount;
    extraInfo.s1oIdx = this->s1oIdx;
    extraInfo.s1TileIdx = this->s1TileIdx;
    extraInfo.boIdx = this->boIdx;
    extraInfo.n2oIdx = this->n2oIdx;
    extraInfo.goIdx = this->goIdx;
    extraInfo.taskId = taskId;
    extraInfo.s2LoopLimit = s2LoopLimit;
    extraInfo.multiCoreInnerIdx = multiCoreInnerIdx;
    extraInfo.s1SizeAcc = s1SizeAcc;
    extraInfo.s2SizeAcc = s2SizeAcc;
    extraInfo.attenB1SSOffset = attenB1SSOffset;

    GetSeqQlenKvlenByBoidx(extraInfo.boIdx, extraInfo.s1SeqSize, extraInfo.s2SeqSize);
    this->ComputeBmm1Tail(extraInfo);
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::ComputeBmm1Tail(SplitExtraInfo& extraInfo)
{
    extraInfo.s1RealSize = this->s1BaseSize; // 128
    if (extraInfo.s1SeqSize < (extraInfo.s1oIdx + 1) * this->s1BaseSize) {
        extraInfo.s1RealSize = extraInfo.s1SeqSize - extraInfo.s1oIdx * this->s1BaseSize;
    }
    extraInfo.s2RealSize = this->s2BaseNratioSize;
    extraInfo.s2AlignedSize = extraInfo.s2RealSize;
    if (extraInfo.s2StartIdx + (extraInfo.s2LoopCount + 1) * extraInfo.s2RealSize > extraInfo.s2EndIdx) {
        extraInfo.s2RealSize = extraInfo.s2EndIdx - extraInfo.s2LoopCount * extraInfo.s2RealSize - extraInfo.s2StartIdx;
        extraInfo.s2AlignedSize = this->Align(extraInfo.s2RealSize);
    }
    // 基于s2 实际长度调整，确保一次loop 处理基本块为 128 * 1024
    extraInfo.vec1S1BaseSize = this->Min(1024 / extraInfo.s2AlignedSize * 8, extraInfo.s1RealSize);
    extraInfo.realSplitN = this->CeilDiv(extraInfo.s1RealSize, extraInfo.vec1S1BaseSize);

    if (this->vDsizeAlign > 64) {
        extraInfo.vec2S1BaseSize = 64 * 128 / this->vDsizeAlign; // D=64: vec2S1BaseSize=128, D=128: vec2S1BaseSize=64
    } else {
        extraInfo.vec2S1BaseSize = extraInfo.s1RealSize; // vec2S1BaseSize=128
    }

    return;
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::IterateBmm1(SplitExtraInfo& extraInfo)
{
    // 设置m、n、ka、kb、kc
    // 这里 kc = s2RealSize，表示搬出数据
    this->bmm1.SetOrgShape(extraInfo.s1RealSize, this->mm1Kb, this->mm1Ka, this->mm1Kb, extraInfo.s2RealSize);

    // 计算gm上的offset
    int64_t bOffset = 0;
    int64_t s1Offset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;

    // // 对于varlen场景，只支持 layout(bs)ND
    bOffset = extraInfo.s1SizeAcc * this->n2GD;
    s1Offset = extraInfo.s1oIdx * this->s1BaseN2GD;
    n2Offset = extraInfo.n2oIdx * this->gD;
    gOffset = extraInfo.goIdx * this->dSize;

    // query 在每个core 上的offset
    int64_t qCoreOffset = bOffset + n2Offset + gOffset + s1Offset;
    extraInfo.qCoreOffset = qCoreOffset;
    this->bmm1.SetTensorA(this->queryGm[extraInfo.qCoreOffset]);

    // 计算gm上的offset
    int64_t s2Offset = 0;
    // batch 对应在外轴便宜
    // s2 切分成两层，s2StartIdx 对应完整s2。s2LoopCount 对应基本块s2
    // n2Offset 对应head方向偏移
    bOffset = extraInfo.s2SizeAcc * this->n2D;
    s2Offset = extraInfo.s2StartIdx * this->n2D + extraInfo.s2LoopCount * this->s2BaseNratioN2D;
    n2Offset = extraInfo.n2oIdx * this->dSize;

    int64_t kCoreOffset = bOffset + n2Offset + s2Offset;

    this->bmm1.SetTensorB(this->keyGm[kCoreOffset], true);
    this->bmm1.SetTail(extraInfo.s1RealSize, extraInfo.s2RealSize);

    // cube 和 vector间并行，这里对应的是 s2 切分上的db。
    // mm1起两份，这样vec1(0)计算时，mm1(1)可以启动了
    if (extraInfo.taskId % 2 == 0) {
        this->bmm1.template IterateAll<false>(this->mm1ResPing, false, false, true);
    } else {
        this->bmm1.template IterateAll<false>(this->mm1ResPong, false, false, true);
    }
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::WaitBmm1Result()
{
    this->bmm1.WaitIterateAll();
    this->bmm1.End();
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::GetBmm1Result(SplitExtraInfo& extraInfo, LocalTensor<T>& bmm1ResUb,
                                                                 int64_t loopIdx)
{
    if (likely(extraInfo.s2AlignedSize == extraInfo.s2RealSize)) {
        if (extraInfo.taskId % 2 == 0) {
            DataCopy2D(bmm1ResUb, this->mm1ResPing[loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2RealSize],
                       extraInfo.vec1S1RealSize, extraInfo.s2RealSize, extraInfo.s2RealSize);
        } else {
            DataCopy2D(bmm1ResUb, this->mm1ResPong[loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2RealSize],
                       extraInfo.vec1S1RealSize, extraInfo.s2RealSize, extraInfo.s2RealSize);
        }
    } else {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = extraInfo.vec1S1RealSize;
        dataCopyParams.blockLen = extraInfo.s2RealSize * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = extraInfo.s2AlignedSize - extraInfo.s2RealSize;
        if (dataCopyPadParams.rightPadding > this->blockSize) {
            dataCopyPadParams.rightPadding -= this->blockSize;
            dataCopyParams.dstStride = 1;
            int32_t s2BlockAlignedSize = this->CeilDiv(extraInfo.s2RealSize, this->blockSize) * this->blockSize;
            Duplicate<T>(bmm1ResUb[s2BlockAlignedSize], 0, this->blockSize, extraInfo.vec1S1RealSize, 0,
                         extraInfo.s2AlignedSize * sizeof(T) / this->blockBytes);
        }
        dataCopyPadParams.paddingValue = 0;
        if (extraInfo.taskId % 2 == 0) {
            DataCopyPad(bmm1ResUb, this->mm1ResPing[loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2RealSize],
                        dataCopyParams, dataCopyPadParams);
        } else {
            DataCopyPad(bmm1ResUb, this->mm1ResPong[loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2RealSize],
                        dataCopyParams, dataCopyPadParams);
        }
    }
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                 static_cast<uint32_t>(extraInfo.s2AlignedSize)};
    bmm1ResUb.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, DataFormat::ND));
}

template <typename INPUT_T, typename T>
__aicore__ inline int64_t FlashAttention<INPUT_T, T>::ComputeOffsetForNoCompress(SplitExtraInfo& extraInfo,
                                                                                 int64_t loopIdx)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t gOffset = 0;

    int64_t s1Offset = extraInfo.s1oIdx * this->s1BaseSize * extraInfo.s2SeqSize +
                       loopIdx * extraInfo.vec1S1BaseSize * extraInfo.s2SeqSize;
    int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNratioSize;
    if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskBN2GS1S2 ||
        this->tilingData->inputParams.attenMaskShapeType == this->attenMaskBS1S2 ||
        this->tilingData->inputParams.attenMaskShapeType == this->attenMaskFlattenedS1S2) {
        bOffset = extraInfo.attenB1SSOffset;
    } else if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskS1S2) {
        s1Offset = extraInfo.s1oIdx * this->s1BaseSize * this->tilingData->inputParams.s2Size +
                   loopIdx * extraInfo.vec1S1BaseSize * this->tilingData->inputParams.s2Size;
    } else if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskTT) {
        s1Offset = extraInfo.s1SizeAcc + extraInfo.s1oIdx * this->s1BaseSize + loopIdx * extraInfo.vec1S1BaseSize;
        s1Offset = s1Offset * this->s2SizeSum;
        s2Offset = s2Offset + extraInfo.s2SizeAcc;
    }
    return bOffset + n2Offset + gOffset + s1Offset + s2Offset;
}

template <typename INPUT_T, typename T>
__aicore__ inline int64_t FlashAttention<INPUT_T, T>::ComputeAttenMaskOffset(SplitExtraInfo& extraInfo, int64_t loopIdx)
{
    // no compress mode
    if (this->tilingData->inputParams.attenMaskCompressMode ==
        static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
        // no-compress: 用户传入完整 bool mask, 始终按主 mask(maskType=0)搬入/计算
        this->attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
        return this->ComputeOffsetForNoCompress(extraInfo, loopIdx);
    }
    // compress mode
    int64_t delta = 0;
    int64_t deltaPre = 0;
    int64_t deltaN = extraInfo.s1SeqSize - extraInfo.s2SeqSize;
    int64_t s1Offset = extraInfo.s1oIdx * this->s1BaseSize + loopIdx * extraInfo.vec1S1BaseSize;
    int64_t s2Offset = extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNratioSize;
    if (this->tilingData->inputParams.attenMaskCompressMode ==
        static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE)) {
        // 左上对齐: 对角线过 (0,0), 不平移
        delta = s1Offset - s2Offset;
    } else if (this->tilingData->inputParams.attenMaskCompressMode ==
               static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
        // 右下对齐: 用 deltaN 把对角线平移到右下角
        delta = s1Offset - s2Offset - deltaN;
    } else if (this->tilingData->inputParams.attenMaskCompressMode ==
               static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
        // 右下对齐 band: 把 pre/next 窗口换算成右下 causal 等价的有效边界
        // rightDownPreTokens/rightDownNextTokens 为右下对齐后的 pre/next token 数,
        //   Max(..., 0) 对窗口宽于该维度的情况做截断(该侧不再切块)
        int64_t preTokens = this->tilingData->coreParams.s1SparseValidSize;
        int64_t nextTokens = this->tilingData->coreParams.s2SparseValidSize;
        int64_t rightDownPreTokens =
            extraInfo.s1SeqSize - this->Max(extraInfo.s2SeqSize - preTokens, static_cast<int64_t>(0));
        int64_t rightDownNextTokens =
            extraInfo.s2SeqSize - this->Max(extraInfo.s1SeqSize - nextTokens, static_cast<int64_t>(0));
        deltaPre = s1Offset - s2Offset - rightDownPreTokens - 1;
        this->attenMaskOffsetPre = this->ComputeOffsetForCausal(
            deltaPre, extraInfo.vec1S1BaseSize, this->s2BaseNratioSize, this->tilingData->inputParams.attenMaskS2Size);
        delta = s1Offset - s2Offset + rightDownNextTokens;
    } else {
        return 0;
    }

    // 依据 delta(causal/next 边)与 deltaPre(band pre 边)确定该基本块的 mask 计算档位,
    // 用于跳过 mask 全通块的搬入/计算
    this->GetAttenMaskComputeMode(delta, deltaPre, extraInfo);
    return this->ComputeOffsetForCausal(delta, extraInfo.vec1S1BaseSize, this->s2BaseNratioSize,
                                        this->tilingData->inputParams.attenMaskS2Size);
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::GetAttenMaskComputeMode(int64_t delta, int64_t deltaPre,
                                                                           SplitExtraInfo& extraInfo)
{
    // causalOrNextFactor >= 0: causal/next 边已越过当前基本块右下角, 该块 next 侧全通
    int64_t causalOrNextFactor = delta - extraInfo.s2AlignedSize;
    if (this->tilingData->inputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE) ||
        this->tilingData->inputParams.attenMaskCompressMode ==
            static_cast<uint8_t>(AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE)) {
        if (causalOrNextFactor >= 0) {
            // 整块落在 causal 有效区内, mask 全通, 无需搬入/计算
            this->attenMaskComputeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
        } else {
            this->attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
        }
        return;
    }
    if (this->tilingData->inputParams.attenMaskCompressMode == static_cast<uint8_t>(AttenMaskCompressMode::BAND_MODE)) {
        // preFactor <= 0: pre 边未切到当前基本块, 该块 pre 侧全通
        int64_t preFactor = deltaPre + 1 + extraInfo.vec1S1BaseSize;
        if (causalOrNextFactor >= 0 && preFactor <= 0) {
            this->attenMaskComputeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
        } else if (causalOrNextFactor < 0 && preFactor <= 0) {
            this->attenMaskComputeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
        } else if (causalOrNextFactor >= 0 && preFactor > 0) {
            this->attenMaskComputeMode = AttenMaskComputeMode::PRE_ONLY_MODE;
        } else {
            this->attenMaskComputeMode = AttenMaskComputeMode::PRE_AND_NEXT_MODE;
        }
    }
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::CopyInAttenMask(SplitExtraInfo& extraInfo, int64_t loopIdx,
                                                                   int64_t maskOffset)
{
    if (!this->hasAttenMask) {
        return;
    }
    LocalTensor<uint8_t> attenMaskUb = this->maskTBufPing.template Get<uint8_t>();

    int32_t s2StrideSize = extraInfo.s1SeqSize;
    if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskS1S2) {
        s2StrideSize = this->tilingData->inputParams.s2Size;
    } else if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskTT) {
        s2StrideSize = this->s2SizeSum;
    } else if (this->tilingData->inputParams.attenMaskShapeType == this->attenMaskFlattenedS1S2) {
        s2StrideSize = extraInfo.s2SeqSize;
    }

    // band compress mode
    if (this->tilingData->inputParams.attenMaskCompressMode !=
        static_cast<uint8_t>(AttenMaskCompressMode::NO_COMPRESS_MODE)) {
        s2StrideSize = this->tilingData->inputParams.attenMaskS2Size;
    }
    this->BoolCopyIn(attenMaskUb, this->attenMaskGmInt, maskOffset, extraInfo.vec1S1RealSize, extraInfo.s2RealSize,
                     s2StrideSize, extraInfo);
    return;
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::CopyInPrmMask(SplitExtraInfo& extraInfo)
{
    if (!this->hasPrmMask) {
        return;
    }
    LocalTensor<uint8_t> prmMaskUb = this->maskTBufPong.template Get<uint8_t>();
    int32_t alignedS2Size = this->CeilDiv(extraInfo.s2RealSize, this->blockBytes) * this->blockBytes;
    int64_t s2Offset = extraInfo.s2SizeAcc + extraInfo.s2StartIdx + extraInfo.s2LoopCount * this->s2BaseNratioSize;
    int64_t validRemainS2 = this->Min(this->s2SizeSum - s2Offset, extraInfo.s2RealSize);

    event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = validRemainS2 * sizeof(uint8_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = true;
    dataCopyPadParams.rightPadding = alignedS2Size - validRemainS2;
    dataCopyPadParams.paddingValue = 0;
    SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    DataCopyPad(prmMaskUb, this->prmMaskGm[s2Offset], dataCopyParams, dataCopyPadParams);
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    int64_t loop = alignedS2Size / this->repeatMaxBytes;
    int64_t remain = alignedS2Size % this->repeatMaxBytes;
    uint8_t repeaTimes = extraInfo.vec1S1BaseSize; // extraInfo.vec1S1RealSize = extraInfo.vec1S1BaseSize;
    int64_t dstRepStride = alignedS2Size / this->blockBytes;
    int64_t repeatMaxSize16 = this->repeatMaxBytes / sizeof(half);
    uint64_t mask = repeatMaxSize16;

    if (repeaTimes > 1) {
        auto prmMaskUbTmp = prmMaskUb.ReinterpretCast<int16_t>();
        // CopyRepeatParams repeatParams;
        // repeatParams.srcStride = 1;
        // repeatParams.dstStride = 1;
        // repeatParams.srcRepeatSize = 0;
        // repeatParams.dstRepeatSize = dstRepStride;
        UnaryRepeatParams repeatParams;
        repeatParams.srcBlkStride = 1;
        repeatParams.dstBlkStride = 1;
        repeatParams.srcRepStride = 0;
        repeatParams.dstRepStride = dstRepStride;

        for (int i = 0; i < loop; i++) {
            // Copy(prmMaskUbTmp[i * repeatMaxSize16], prmMaskUbTmp[i * repeatMaxSize16], mask,
            //          repeaTimes, repeatParams);
            Adds(prmMaskUbTmp[i * repeatMaxSize16], prmMaskUbTmp[i * repeatMaxSize16], (int16_t)0, mask, repeaTimes,
                 repeatParams);
        }
        if (likely(remain)) {
            mask = this->CeilDiv(remain, sizeof(half));
            // Copy(prmMaskUbTmp[loop * repeatMaxSize16], prmMaskUbTmp[loop * repeatMaxSize16], mask,
            //          repeaTimes, repeatParams);
            Adds(prmMaskUbTmp[loop * repeatMaxSize16], prmMaskUbTmp[loop * repeatMaxSize16], (int16_t)0, mask,
                 repeaTimes, repeatParams);
        }
    }
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::ComputePrmMask(SplitExtraInfo& extraInfo, LocalTensor<T>& bmm1ResUb)
{
    if (!this->hasPrmMask) {
        return;
    }
    LocalTensor<uint8_t> prmMaskUb = this->maskTBufPong.template Get<uint8_t>();
    int32_t alignedS2Size = this->CeilDiv(extraInfo.s2RealSize, this->blockBytes) * this->blockBytes;
    // uint32_t shapeArray[] = {static_cast<uint32_t>(extraInfo.vec1S1BaseSize), static_cast<uint32_t>(alignedS2Size)};
    // prmMaskUb.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    // prmMaskUb.SetSize(extraInfo.vec1S1BaseSize * alignedS2Size);
    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();

    SelectWithBytesMaskShapeInfo shapeInfo;
    shapeInfo.firstAxis = extraInfo.vec1S1RealSize;
    shapeInfo.srcLastAxis = extraInfo.s2AlignedSize;
    shapeInfo.maskLastAxis = alignedS2Size;

    SelectWithBytesMask<T, uint8_t, false>(bmm1ResUb, this->negativeFloatScalar, bmm1ResUb, prmMaskUb, apiTmpBuffer,
                                           shapeInfo);

    return;
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::CopyInSink()
{
    if (!this->hasSink) {
        return;
    }

    LocalTensor<INPUT_T> sinkUb = this->sinkBuf.template GetWithOffset<INPUT_T>(512, 0);
    LocalTensor<T> sinkUbFloat = this->sinkBuf.template GetWithOffset<T>(512, 512 * sizeof(INPUT_T));
    int32_t alignSize = this->blockBytes / sizeof(INPUT_T);
    int32_t alignedN1Size = this->CeilDiv(this->n2G, alignSize) * alignSize;

    DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = alignedN1Size * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = true;
    dataCopyPadParams.rightPadding = alignedN1Size - this->n2G;
    dataCopyPadParams.paddingValue = 0;
    DataCopyPad(sinkUb, this->sinkGm, copyParams, dataCopyPadParams);
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    Cast(sinkUbFloat, sinkUb, RoundMode::CAST_NONE, alignedN1Size);
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::SoftMaxCompute(SplitExtraInfo& extraInfo, LocalTensor<T>& srcTensor,
                                                                  int64_t loopIdx)
{
    uint32_t bmm1ResUbShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                 static_cast<uint32_t>(extraInfo.s2AlignedSize)};
    uint32_t bmm1ResUbOrgShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                                    static_cast<uint32_t>(extraInfo.s2RealSize)};
    srcTensor.SetShapeInfo(ShapeInfo(2, bmm1ResUbShape, 2, bmm1ResUbOrgShape, DataFormat::ND));

    uint32_t maxSumShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                              static_cast<uint32_t>(this->fp32BaseSize)};
    LocalTensor<T> sumUb;
    if (extraInfo.multiCoreInnerIdx % 2 == 0) {
        sumUb = this->softmaxSumPingBuf.template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->fp32BaseSize];
    } else {
        sumUb = this->softmaxSumPongBuf.template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->fp32BaseSize];
    }
    LocalTensor<T> maxUb =
        this->softmaxMaxBuf.template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->fp32BaseSize];

    sumUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));
    maxUb.SetShapeInfo(ShapeInfo(2, maxSumShape, DataFormat::ND));

    uint32_t expShape[] = {static_cast<uint32_t>(extraInfo.vec1S1RealSize),
                           static_cast<uint32_t>(this->blockBytes / sizeof(T))};
    LocalTensor<T> expUb;
    if (extraInfo.taskId % 2 == 0) {
        expUb = this->softmaxExpPingBuf
                    .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->blockBytes / sizeof(T)];
    } else {
        expUb = this->softmaxExpPongBuf
                    .template Get<T>()[loopIdx * extraInfo.vec1S1BaseSize * this->blockBytes / sizeof(T)];
    }
    expUb.SetShapeInfo(ShapeInfo(2, expShape, DataFormat::ND));

    LocalTensor<uint8_t> apiTmpBuffer = this->commonTBuf.template Get<uint8_t>();
    PipeBarrier<PIPE_V>();
    if (unlikely(extraInfo.s2LoopCount == 0)) {
        if (IsBasicBlockInSoftMax(extraInfo.vec1S1RealSize, extraInfo.s2RealSize)) {
            SoftmaxFlashV2<T, false, true, true>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer,
                                                 this->tilingData->softmaxFlashTilingData);
        } else {
            SoftmaxFlashV2<T, false, true, false>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer,
                                                  this->tilingData->softmaxFlashTilingData);
        }
    } else {
        if (IsBasicBlockInSoftMax(extraInfo.vec1S1RealSize, extraInfo.s2RealSize)) {
            SoftmaxFlashV2<T, true, true, true>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer,
                                                this->tilingData->softmaxFlashTilingData);
        } else {
            SoftmaxFlashV2<T, true, true, false>(srcTensor, sumUb, maxUb, srcTensor, expUb, sumUb, maxUb, apiTmpBuffer,
                                                 this->tilingData->softmaxFlashTilingData);
        }
    }
    if (loopIdx == extraInfo.realSplitN - 1 && extraInfo.s2LoopCount == extraInfo.s2LoopLimit) {
        ComputeAndCopyMaxLogits(extraInfo);
        if (this->tilingData->inputParams.sinkFlag == 1) {
            // rebase 到start 位置
            maxUb = this->softmaxMaxBuf.template Get<float>();
            if (extraInfo.multiCoreInnerIdx % 2 == 0) {
                sumUb = this->softmaxSumPingBuf.template Get<float>();
            } else {
                sumUb = this->softmaxSumPongBuf.template Get<float>();
            }
            // softmaxsum add bias
            LocalTensor<T> sinkUbFloat = this->sinkBuf.template GetWithOffset<T>(512, 512 * sizeof(INPUT_T));
            int32_t calcSize = sumUb.GetSize(); // 1k
            int32_t nIdx = extraInfo.n2oIdx * this->tilingData->inputParams.gSize + extraInfo.goIdx;
            LocalTensor<T> sinkTensor = this->commonTBuf.template GetWithOffset<T>(calcSize, 0);
            LocalTensor<T> expNegMaxTensor =
                this->commonTBuf.template GetWithOffset<T>(calcSize, calcSize * sizeof(float));

            PipeBarrier<PIPE_V>();
            float sinkValue = sinkUbFloat.GetValue(nIdx);
            Duplicate(sinkTensor, sinkValue, calcSize);
            PipeBarrier<PIPE_V>();
            Duplicate(expNegMaxTensor, float(0), calcSize);
            PipeBarrier<PIPE_V>();
            Sub(expNegMaxTensor, expNegMaxTensor, maxUb, calcSize);
            PipeBarrier<PIPE_V>();
            Exp(expNegMaxTensor, expNegMaxTensor, calcSize);
            PipeBarrier<PIPE_V>();
            Mul(sinkTensor, sinkTensor, expNegMaxTensor, calcSize);
            PipeBarrier<PIPE_V>();
            Add(sumUb, sumUb, sinkTensor, calcSize);
            PipeBarrier<PIPE_V>();
        }
        LseComputeAndCopy(extraInfo);
        SoftmaxMaxDataCopyOut(extraInfo);
    }
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::IterateBmm2(SplitExtraInfo& extraInfo)
{
    int64_t bOffset = 0;
    int64_t n2Offset = 0;
    int64_t s2Offset = 0;

    // (bs)ND, this D is value D size, may not equal query
    bOffset = extraInfo.s2SizeAcc * this->n2outD;
    s2Offset = extraInfo.s2StartIdx * this->tilingData->inputParams.n2Size * this->vDsize +
               extraInfo.s2LoopCount * this->s2BaseNratioSize * this->n2outD;
    n2Offset = extraInfo.n2oIdx * this->vDsize;

    int64_t vCoreOffset = bOffset + n2Offset + s2Offset;
    this->bmm2.SetOrgShape(extraInfo.s1RealSize, this->mm2Kb, extraInfo.s2AlignedSize, this->mm2Kb, this->vDsize);
    if (extraInfo.taskId % 2 == 0) {
        this->bmm2.SetTensorA(this->stage1ResPing);
    } else {
        this->bmm2.SetTensorA(this->stage1ResPong);
    }
    this->bmm2.SetTensorB(this->valueGm[vCoreOffset]);
    this->bmm2.SetTail(extraInfo.s1RealSize, this->vDsize, extraInfo.s2RealSize); // 128 * 64

    this->bmm2.template IterateAll<false>(this->mm2Res[extraInfo.taskId % 2], false, false, true);
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::WaitBmm2Result()
{
    this->bmm2.WaitIterateAll();
    this->bmm2.End();
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::ComputeAndCopyMaxLogits(const SplitExtraInfo& extraInfo)
{
    if (!this->needMaxLogits) {
        return;
    }
    LocalTensor<T> maxUb = this->softmaxMaxBuf.template Get<T>();
    LocalTensor<T> dst = this->maxLogitsBuf.template Get<T>();
    LocalTensor<T> workBuf = this->commonTBuf.template Get<T>();
    // src: extraInfo.vec1S1RealSize * this->fp32BaseSize
    PipeBarrier<PIPE_V>();
    ReduceMax<T>(dst, maxUb, workBuf, extraInfo.s1RealSize * this->fp32BaseSize, /*calIndex*/ false);
    PipeBarrier<PIPE_V>();

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = 1 * sizeof(float);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPad(this->maxLogitsWorkspaceGm[(extraInfo.n2oIdx * this->tilingData->inputParams.gSize + extraInfo.goIdx) *
                                               this->accS1TileNum +
                                           extraInfo.s1TileIdx + extraInfo.s1oIdx],
                dst, copyParams);
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::LseComputeAndCopy(SplitExtraInfo& extraInfo)
{
    LocalTensor<float> maxUb;
    maxUb = this->softmaxMaxBuf.template Get<float>();

    LocalTensor<float> sumUb;
    if (extraInfo.multiCoreInnerIdx % 2 == 0) {
        sumUb = this->softmaxSumPingBuf.template Get<float>();
    } else {
        sumUb = this->softmaxSumPongBuf.template Get<float>();
    }
    int32_t calcSize = sumUb.GetSize();
    LocalTensor<float> commonBufTensor = this->commonTBuf.template Get<float>();
    // shape [extraInfo.s1RealSize, 8], 256byte one repeate
    int32_t repeat_num = (extraInfo.s1RealSize * this->fp32BaseSize) / this->fp32BlockSize;
    int32_t remain_num = (extraInfo.s1RealSize * this->fp32BaseSize) - repeat_num * this->fp32BlockSize;
    // 用optionalInputQueue的queue
    PipeBarrier<PIPE_V>();
    Log(commonBufTensor, sumUb, calcSize);
    PipeBarrier<PIPE_V>();
    Add(commonBufTensor, maxUb, commonBufTensor, calcSize);
    PipeBarrier<PIPE_V>();
    BlockReduceMax<float>(commonBufTensor, commonBufTensor, repeat_num, this->fp32BlockSize, 1, 1, 8);
    if (likely(remain_num)) {
        BlockReduceMax<float>(commonBufTensor[repeat_num * this->fp32BlockSize / 8],
                              commonBufTensor[repeat_num * this->fp32BlockSize], 1, remain_num, 1, 1, 8);
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = extraInfo.s1RealSize * sizeof(T);
    // dataCopyParams.blockCount = extraInfo.s1RealSize;        // copy s1RealSize blocks each loop
    // dataCopyParams.blockLen   = 1 * sizeof(T);               // each block copy 4 byte
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    int64_t boffset;
    int64_t noffset;
    int64_t soffset;
    // NT layout, default setting
    boffset = extraInfo.s1SizeAcc;
    noffset = (extraInfo.n2oIdx * this->tilingData->inputParams.gSize + extraInfo.goIdx) * s1SizeSum;
    soffset = extraInfo.s1oIdx * this->s1BaseSize;
    // BNmaxS layout
    int64_t dst_offset = boffset + noffset + soffset;
    DataCopyPad(this->lseOutGm[dst_offset], commonBufTensor, dataCopyParams);
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::Bmm2ResultDiv(SplitExtraInfo& extraInfo, int64_t vec2s1oIdx)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 1;
    repeatParams.src0RepStride = this->vDsizeAlign / this->blockSize;
    repeatParams.src1BlkStride = 0;
    repeatParams.src1RepStride = 1;
    repeatParams.dstRepStride = this->vDsizeAlign / this->blockSize;
    int32_t loop = this->vDsizeAlign / this->repeatMaxSize;
    int32_t remain = this->vDsizeAlign % this->repeatMaxSize;

    LocalTensor<float> sumUb;
    int32_t calcSize = sumUb.GetSize();
    if (extraInfo.multiCoreInnerIdx % 2 == 0) {
        sumUb = this->softmaxSumPingBuf.template Get<float>();
    } else {
        sumUb = this->softmaxSumPongBuf.template Get<float>();
    }

    // 用optionalInputQueue的queue
    PipeBarrier<PIPE_V>();
    if constexpr (IsSameType<T, half>::value) {
        LocalTensor<float> commonBufTensor = this->commonTBuf.template Get<float>();
        Copy(commonBufTensor, sumUb, 64, calcSize / 64, {2, 1, 16, 8});
        Copy(commonBufTensor[8], sumUb, 64, calcSize / 64, {2, 1, 16, 8});
        LocalTensor<T> sumCastTensor =
            this->softmaxExpPingBuf.template Get<T>(this->tilingData->tensorSizeParams.softmaxExpUbSize);
        PipeBarrier<PIPE_V>();
        Cast(sumCastTensor, commonBufTensor, RoundMode::CAST_ROUND, 2 * sumUb.GetSize());
        PipeBarrier<PIPE_V>();
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResUb[i * this->repeatMaxSize], bmm2ResUb[i * this->repeatMaxSize], sumCastTensor,
                this->repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * this->repeatMaxSize], bmm2ResUb[loop * this->repeatMaxSize], sumCastTensor, remain,
                extraInfo.vec2S1RealSize, repeatParams);
        }
    } else {
        for (int i = 0; i < loop; i++) {
            Div(bmm2ResUb[i * this->repeatMaxSize], bmm2ResUb[i * this->repeatMaxSize],
                sumUb[vec2s1oIdx * extraInfo.vec2S1BaseSize * 8], this->repeatMaxSize, extraInfo.vec2S1RealSize,
                repeatParams);
        }
        if (likely(remain)) {
            Div(bmm2ResUb[loop * this->repeatMaxSize], bmm2ResUb[loop * this->repeatMaxSize],
                sumUb[vec2s1oIdx * extraInfo.vec2S1BaseSize * 8], remain, extraInfo.vec2S1RealSize, repeatParams);
        }
    }
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::Bmm2ResultMul(SplitExtraInfo& extraInfo, LocalTensor<T>& bmm2ResUb,
                                                                 int64_t vec2s1oIdx)
{
    PipeBarrier<PIPE_V>();
    LocalTensor<T> expUb;
    if (extraInfo.taskId % 2 == 0) {
        expUb = this->softmaxExpPingBuf.template Get<T>();
    } else {
        expUb = this->softmaxExpPongBuf.template Get<T>();
    }

    BinaryRepeatParams repeatParams;
    repeatParams.src0BlkStride = 0;
    repeatParams.src0RepStride = 1;
    repeatParams.src1RepStride = this->vDsizeAlign / this->blockSize;
    repeatParams.dstRepStride = this->vDsizeAlign / this->blockSize;

    // s1长度可能会超过255限制，修改成双重循环
    // 根据一次最多计算的byte数量，对bmm2Res分组mul
    int32_t loop = this->vDsizeAlign / this->repeatMaxSize;
    int32_t remain = this->vDsizeAlign % this->repeatMaxSize;
    for (int i = 0; i < loop; i++) {
        Mul(bmm2ResUb[i * this->repeatMaxSize], expUb[vec2s1oIdx * extraInfo.vec2S1BaseSize * 8],
            bmm2ResUb[i * this->repeatMaxSize], this->repeatMaxSize, extraInfo.vec2S1RealSize, repeatParams);
    }
    if (likely(remain)) {
        Mul(bmm2ResUb[loop * this->repeatMaxSize], expUb[vec2s1oIdx * extraInfo.vec2S1BaseSize * 8],
            bmm2ResUb[loop * this->repeatMaxSize], remain, extraInfo.vec2S1RealSize, repeatParams);
    }
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::Bmm2DataCopyOut(SplitExtraInfo& extraInfo, int64_t vec2s1oIdx)
{
    LocalTensor<T> bmm2ResUb = this->stage2TBuf.template Get<T>();
    LocalTensor<INPUT_T> attenOut = this->stage2TBuf.template Get<INPUT_T>();
    int32_t calcSize = bmm2ResUb.GetSize();
    PipeBarrier<PIPE_V>();

    if constexpr (!IsSameType<INPUT_T, T>::value) {
        Cast(attenOut, bmm2ResUb, RoundMode::CAST_ROUND, calcSize);
    }
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockLen = this->vDsize * sizeof(INPUT_T);
    dataCopyParams.srcStride = 0; // 32 byte
    int64_t dstStride = 0;
    // only support layoutTND
    // 每次搬运对应一个actual seq，因此stride对应 (ngSize - 1) * D
    dstStride = (this->n2G - 1) * this->vDsize * sizeof(INPUT_T);

    int64_t coreBoffset = extraInfo.s1SizeAcc * this->n2GoutD;
    int64_t coreS1offset = extraInfo.s1oIdx * this->s1BaseSize * this->n2GoutD;
    int64_t coreN2offset = extraInfo.n2oIdx * this->tilingData->inputParams.gSize * this->vDsize;
    int64_t coreGoffset = extraInfo.goIdx * this->vDsize;
    int64_t attenCoreOffset = coreBoffset + coreS1offset + coreN2offset + coreGoffset;

    // dataCopyParams.dstStride类型定义uint16_t，65535是其最大值
    if (likely(dstStride <= 65535)) {
        dataCopyParams.blockCount = extraInfo.vec2S1RealSize;
        dataCopyParams.dstStride = static_cast<uint16_t>(dstStride);
        DataCopyPad(this->attentionOutGm[attenCoreOffset + vec2s1oIdx * extraInfo.vec2S1BaseSize * this->n2GoutD],
                    attenOut, dataCopyParams);
    } else {
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;
        int64_t datacopyOffset = this->vDsize;
        for (uint32_t i = 0; i < extraInfo.vec2S1RealSize; i++) {
            DataCopyPad(this->attentionOutGm[attenCoreOffset + vec2s1oIdx * extraInfo.vec2S1BaseSize * this->n2GoutD +
                                             i * datacopyOffset],
                        attenOut[i * this->n2G * this->vDsize], dataCopyParams);
        }
    }
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::SoftmaxMaxDataCopyOut(SplitExtraInfo& extraInfo)
{
    LocalTensor<float> maxTensor = this->softmaxMaxBuf.template Get<float>();

    int32_t repeat_num = (extraInfo.s1RealSize * this->fp32BaseSize) / this->fp32BlockSize;
    int32_t remain_num = (extraInfo.s1RealSize * this->fp32BaseSize) - repeat_num * this->fp32BlockSize;
    PipeBarrier<PIPE_V>();
    BlockReduceMax<float>(maxTensor, maxTensor, repeat_num, this->fp32BlockSize, 1, 1, 8);
    if (likely(remain_num)) {
        BlockReduceMax<float>(maxTensor[repeat_num * this->fp32BlockSize / 8],
                              maxTensor[repeat_num * this->fp32BlockSize], 1, remain_num, 1, 1, 8);
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = extraInfo.s1RealSize * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    int64_t boffset;
    int64_t noffset;
    int64_t soffset;
    boffset = extraInfo.s1SizeAcc;
    noffset = (extraInfo.n2oIdx * this->tilingData->inputParams.gSize + extraInfo.goIdx) * s1SizeSum;
    soffset = extraInfo.s1oIdx * this->s1BaseSize;
    int64_t dst_offset = boffset + noffset + soffset;

    DataCopyPad(this->softmaxMaxGm[dst_offset], maxTensor, dataCopyParams);
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::SoftmaxSumDataCopyOut(SplitExtraInfo& extraInfo, int64_t vec2s1oIdx)
{
    LocalTensor<float> sumTensor;

    int64_t softmaxSumOffset = vec2s1oIdx * extraInfo.vec2S1BaseSize * this->fp32BaseSize;
    if (extraInfo.multiCoreInnerIdx % 2 == 0) {
        sumTensor = this->softmaxSumPingBuf.template Get<float>()[softmaxSumOffset];
    } else {
        sumTensor = this->softmaxSumPongBuf.template Get<float>()[softmaxSumOffset];
    }
    int32_t repeat_num = (extraInfo.vec2S1RealSize * this->fp32BaseSize) / this->fp32BlockSize;
    int32_t remain_num = (extraInfo.vec2S1RealSize * this->fp32BaseSize) - repeat_num * this->fp32BlockSize;
    PipeBarrier<PIPE_V>();
    BlockReduceMax<float>(sumTensor, sumTensor, repeat_num, this->fp32BlockSize, 1, 1, 8);
    if (likely(remain_num)) {
        BlockReduceMax<float>(sumTensor[repeat_num * this->fp32BlockSize / 8],
                              sumTensor[repeat_num * this->fp32BlockSize], 1, remain_num, 1, 1, 8);
    }

    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));

    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = extraInfo.vec2S1RealSize * sizeof(T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;

    int64_t boffset;
    int64_t noffset;
    int64_t soffset;
    boffset = extraInfo.s1SizeAcc;
    noffset = (extraInfo.n2oIdx * this->tilingData->inputParams.gSize + extraInfo.goIdx) * s1SizeSum;
    soffset = extraInfo.s1oIdx * this->s1BaseSize + vec2s1oIdx * extraInfo.vec2S1BaseSize;
    int64_t dst_offset = boffset + noffset + soffset;
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(this->softmaxSumGm[dst_offset], sumTensor, dataCopyParams);
}

template <typename INPUT_T, typename T>
__aicore__ inline void FlashAttention<INPUT_T, T>::Bit2Int8CopyIn(LocalTensor<uint8_t>& dstTensor,
                                                                  GlobalTensor<uint8_t>& srcTensor, int64_t offset,
                                                                  int32_t s1Size, int32_t s2Size, int32_t batchSize,
                                                                  SplitExtraInfo& extraInfo)
{
    int32_t alignedS2Size = this->CeilDiv(s2Size / this->byteBitRatio, this->blockBytes) * this->blockBytes;
    uint32_t shapeArray[] = {static_cast<uint32_t>(batchSize * s1Size), static_cast<uint32_t>(alignedS2Size)};
    dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
    dstTensor.SetSize(batchSize * s1Size * alignedS2Size);
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = batchSize * s1Size;
    dataCopyParams.blockLen = this->CeilDiv(s2Size / this->byteBitRatio, this->blockBytes);
    dataCopyParams.dstStride = 0; // gap

    if (extraInfo.s2SeqSize / this->byteBitRatio % this->blockBytes == 0) {
        dataCopyParams.srcStride =
            (extraInfo.s2SeqSize / this->byteBitRatio - dataCopyParams.blockLen * this->blockBytes) /
            this->blockBytes; // srcGap
        DataCopy(dstTensor, srcTensor[offset / this->byteBitRatio], dataCopyParams);
    } else {
        dataCopyParams.blockLen = s2Size / this->byteBitRatio;
        dataCopyParams.srcStride = (extraInfo.s2SeqSize / this->byteBitRatio - dataCopyParams.blockLen);
        DataCopyPadParams dataCopyPadParams;
        dataCopyPadParams.isPad = true;
        dataCopyPadParams.rightPadding = alignedS2Size - s2Size / this->byteBitRatio;
        dataCopyPadParams.paddingValue = DATACOPYPAD_PADDING_VALUE_ZERO;
        DataCopyPad(dstTensor, srcTensor[offset / this->byteBitRatio], dataCopyParams, dataCopyPadParams);
    }
}

#endif // FLASH_ATTENTION_VAR_LEN_H
