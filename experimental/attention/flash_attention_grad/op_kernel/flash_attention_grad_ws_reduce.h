/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef _FLASH_ATTENTION_GRAD_WS_REDUCE_H_
#define _FLASH_ATTENTION_GRAD_WS_REDUCE_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

template <typename T1, typename T2, const bool IS_TND_INPUTS = true,
          const CubeFormat MM_OUT_FORMAT = CubeFormat::ND_ALIGN>
class FlashAttentionGradWsReduce {
public:
    __aicore__ inline FlashAttentionGradWsReduce(){};

    __aicore__ inline void Init(__gm__ uint8_t* dx, __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* forward_res, __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum,
                                __gm__ uint8_t* softmax_lse, __gm__ uint8_t* actual_seq_qlen,
                                __gm__ uint8_t* actual_seq_kvlen, __gm__ uint8_t* atten_mask, __gm__ uint8_t* prm_mask,
                                __gm__ uint8_t* sink, __gm__ uint8_t* workspace,
                                const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData,
                                TPipe* pipe_in);
    __aicore__ inline void CopyInSoftMaxGrad(LocalTensor<T1>& dstTensor, LocalTensor<T1>& dstTensor2,
                                             uint64_t softmaxGradOffset, uint32_t s1Extend, uint32_t dExtend,
                                             uint32_t dExtendAlign);
    __aicore__ inline void CalcSoftMaxGrad(LocalTensor<float>& sfmgClc3, uint64_t aTensorOffset, uint32_t s1Extend);
    __aicore__ inline void CopyInSoftMax(LocalTensor<float>& dstTensor, uint32_t s1Extend, uint32_t softMaxOffset);
    __aicore__ inline void CalcSoftMax(LocalTensor<T2>& dstTensor, LocalTensor<float>& srcTensor, uint32_t s1Extend,
                                       uint32_t s2Extend, uint32_t s2ExtendAlign, const SoftMaxTiling& tiling);
    __aicore__ inline void CopyInAttenMaskBool(LocalTensor<uint8_t>& dstTensor, uint64_t attenMaskOffset,
                                               uint32_t s1Extend, uint32_t s2Extend);
    __aicore__ inline void CalcAttenMaskBool(LocalTensor<T2>& dstTensor, LocalTensor<uint8_t> srcTensor,
                                             uint32_t s1Extend, uint32_t s2Extend, uint8_t maskType = 0);
    __aicore__ inline void CalcAttenMaskOffset(uint64_t& attenMaskOffset, const int64_t delta, uint32_t s1VSize,
                                               uint32_t s2VSize);
    __aicore__ inline void CopyInPrmMaskBool(LocalTensor<uint8_t>& dstTensor, uint64_t prmMaskOffset, uint32_t s1Extend,
                                             uint32_t s2Extend);
    __aicore__ inline void CalcPrmMaskBool(LocalTensor<T2>& dstTensor, LocalTensor<uint8_t> srcTensor,
                                           uint32_t s1Extend, uint32_t s2Extend);
    __aicore__ inline void CalSinkGrad(LocalTensor<T2>& srcTensor, uint32_t s1Extend, uint32_t sfmgInnerOffset,
                                       uint32_t curS1Idx);

    __aicore__ inline void Process();
    __aicore__ inline void UpdateCalToken(int64_t bIdx);
    __aicore__ inline bool IfCubeBlockNeedCompute(uint32_t baseIndex);
    __aicore__ inline void InitIndex(uint32_t index);
    __aicore__ inline void SubGrapA(uint32_t curIdx, uint32_t curS1Idx, uint32_t curS2Idx);
    __aicore__ inline void SubGrapB(uint32_t curIdx, uint32_t curS1Idx, uint32_t curS2Idx);
    __aicore__ inline void Compute(uint32_t preIndex, uint32_t nextIndex);
    __aicore__ inline void SyncALLCores();
    __aicore__ inline void GetSeqQlenKvlenByBidx(int64_t bIdx, int64_t& actualSeqQlen, int64_t& actualSeqKvlen);
    __aicore__ inline void GetExtremeValue();

    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using cType1 = MatmulType<TPosition::GM, MM_OUT_FORMAT, T2>;
    using biasType1 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using aType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1, true>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, T1>;
    using cType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using biasType2 = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    Matmul<aType1, bType1, cType1, biasType1> mm1;
    Matmul<aType2, bType2, cType2, biasType2> mm3;
    Matmul<aType2, bType2, cType2, biasType2> mm4;

protected:
    TPipe* pipe;
    TBuf<> ubBuffer;
    TBuf<> tmpBuffer;
    TBuf<> vecClc3;
    TBuf<> softmaxBuffer; // on s1
    TBuf<> sinkBuffer;

    uint32_t coreNum;
    uint64_t cBlockIdx;
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict TilingData;

    // input
    GlobalTensor<T1> dxGm, queryGm, keyGm, valueGm, forwardResGm, sinkGm;
    GlobalTensor<int64_t> seqS1Gm, seqS2Gm;
    GlobalTensor<uint8_t> attenMaskU8Gm, prmMaskGm;
    GlobalTensor<float> softmaxMaxGm, softmaxSumGm, softmaxLseGm;

    // output
    GlobalTensor<float> dqWorkSpaceGm, dkWorkSpaceGm, dvWorkSpaceGm, dSinkWorkspaceGm;
    GlobalTensor<T1> tmpWorkSpaceGm, mulWorkSpaceGm;

    // workspace
    GlobalTensor<T2> mm1WorkspaceGm;
    GlobalTensor<T2> mm2WorkspaceGm;

    __gm__ uint8_t* actual_seq_qlen_addr;
    __gm__ uint8_t* actual_seq_kvlen_addr;

    // AscendC
    GlobalTensor<int32_t> syncGlobal;

    int32_t layout;
    constexpr static uint32_t BNGSD = 0;
    constexpr static uint32_t SBNGD = 1;
    constexpr static uint32_t BSNGD = 2;
    constexpr static uint32_t TND = 3;
    constexpr static uint32_t AAG3BSH = 4; // qkv BNSD dy BSH
    constexpr static uint32_t AAG3SBH = 5; // qkv BNSD dy SBH

    T2 mulsValue = -10000.0;
    T2 negativeScalar;
    T2 positiveScalar;
    // optional control
    int64_t s1Token;
    int64_t s2Token;
    int64_t actualCalcS1Token;
    int64_t actualCalcS2Token;

    uint32_t sparseMode;
    bool isSparse;
    int64_t totalBaseIdx{0};
    int64_t preTotal{0};
    int64_t bDimTail{0};
    int64_t actualS1Len;
    int64_t actualS2Len;

    // org shape info
    int64_t b;
    int64_t Tq;
    int64_t Tkv;
    int64_t n2;
    int64_t g;
    int64_t s1;
    int64_t s2;
    int64_t d;
    int64_t dV;
    int64_t attenMaksDimS2;

    int64_t baseMN;
    int64_t cubeBaseMN;

    // split info
    uint32_t s1Outer;
    uint32_t s1CvRatio;
    uint32_t s1Inner;
    uint32_t s1CvInner;
    uint32_t s1CvTail;
    uint32_t s1CvExtend;

    uint32_t s2Outer;
    uint32_t s2CvRatio;
    uint32_t s2Inner;
    uint32_t s2CvInner;
    uint32_t s1TileNum;
    uint32_t s2TileNum;
    uint32_t gs1TileNum;
    uint32_t sfmgdOuter;
    uint32_t sfmgdInner;
    uint32_t sfmgdTail;
    uint32_t sfmgdTailAlign;
    bool dropBitMode;

    uint32_t blockOuter;
    int64_t accumlateTileS1 = 0;

    // sparse block info
    int64_t blockStarts[50];
    int64_t blockEnds[50];

    // buferinfo
    uint32_t baseUbSize;
    uint32_t matmalWorkspaceSize;
    uint32_t tmpBufferSize;

    uint32_t baseIdx;
    uint32_t bDimIdx;
    uint32_t n2DimIdx;
    uint32_t n2DimTail;
    uint32_t gDimIdx;
    uint32_t gDimTail;
    uint32_t s1oDimIdx;
    uint32_t s2oCvDimIdx;
    uint32_t s2oCvDimTail;
    int64_t gs1Offset;

    uint32_t preS2CvBegin;
    uint32_t preS2CvEnd;
    uint32_t nextS2CvBegin;
    uint32_t nextS2CvEnd;

    int32_t isStart = 1;
    int32_t isFirstInitIndex = 1;
    uint32_t pingpongIdx = 1;

    // db
    uint64_t s2CvExtend = 0;
    uint64_t s2CvExtendAlign = 0;
    uint32_t s1VecLoop = 0;
    uint32_t s1VecSize = 0;
    uint32_t s1ExtendSubGraph = 0;
    uint32_t s2Extend = 0;
    uint32_t s2ExtendAlign = 0;
    uint32_t s2VecLoop = 0;
    uint32_t s2VecSize = 0;
    uint32_t s2VecSizeAlign = 0;

    // unpack
    uint32_t bDimIdxTmp = 0;
    uint32_t n2DimIdxTmp = 0;
    uint32_t gDimIdxTmp = 0;
    uint32_t s1oDimIdxTmp = 0;
    uint32_t s2oDimIdxTmp = 0;

    // db buffer
    constexpr static uint32_t T2Begin = 0;
    constexpr static uint32_t T1Begin = 33 * 1024;
    constexpr static uint32_t MaskBegin = 50 * 1024;
    constexpr static uint32_t TmpBegin = 58 * 1024; // 暂时空置，loop on s2
    constexpr static uint32_t DbBegin = 66 * 1024;

    // calDtype
    constexpr static uint32_t calDtypeBytes = 4;

    // other const
    constexpr static uint32_t cal_block_num = 32 / sizeof(T2);
    constexpr static uint32_t cal_repeat_num = 256 / sizeof(T2);
    constexpr static uint32_t input_block_num = 32 / sizeof(T1);
    constexpr static uint32_t SYNC_GLOBAL_WORKSPACE_SIZE = 16 * 1024;
    constexpr static uint32_t ADDR_ALIGN_SIZE = 512;
    constexpr static uint32_t INPUT_NUMS = 2;
    constexpr static uint32_t BLOCK_SIZE = 32;
    constexpr static uint64_t C0_SIZE = 16;
    constexpr static uint64_t VEC_REPEAT = 8;
    constexpr static uint32_t MAX_BASIC_BLOCK_SIZE = 1024;
    constexpr static uint32_t PSE_PERFORMANCE_MODE = 0x12;

    constexpr static uint32_t VEC_S2_LEN = 256;
    constexpr static uint32_t repeatMaxBytes = 256;
};

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::Init(
    __gm__ uint8_t* dx, __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* forward_res,
    __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum, __gm__ uint8_t* softmax_lse,
    __gm__ uint8_t* actual_seq_qlen, __gm__ uint8_t* actual_seq_kvlen, __gm__ uint8_t* atten_mask,
    __gm__ uint8_t* prm_mask, __gm__ uint8_t* sink, __gm__ uint8_t* workspace,
    const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict ordTilingData, TPipe* pipe_in)
{
    keyGm.SetGlobalBuffer((__gm__ T1*)key);
    valueGm.SetGlobalBuffer((__gm__ T1*)value);
    dxGm.SetGlobalBuffer((__gm__ T1*)dx);
    queryGm.SetGlobalBuffer((__gm__ T1*)query);
    forwardResGm.SetGlobalBuffer((__gm__ T1*)forward_res);
    sinkGm.SetGlobalBuffer((__gm__ T1*)sink);
    seqS1Gm.SetGlobalBuffer((__gm__ int64_t*)actual_seq_qlen);
    seqS2Gm.SetGlobalBuffer((__gm__ int64_t*)actual_seq_kvlen);

    attenMaskU8Gm.SetGlobalBuffer((__gm__ uint8_t*)atten_mask);
    prmMaskGm.SetGlobalBuffer((__gm__ uint8_t*)prm_mask);

    softmaxMaxGm.SetGlobalBuffer((__gm__ float*)softmax_max);
    softmaxSumGm.SetGlobalBuffer((__gm__ float*)softmax_sum);
    softmaxLseGm.SetGlobalBuffer((__gm__ float*)softmax_lse);

    // init current core tilingInfo
    cBlockIdx = GetBlockIdx();
    TilingData = ordTilingData;
    pipe = pipe_in;
    coreNum = TilingData->s1s2BNGS1S2BaseParams.coreNum;

    // shape info
    b = TilingData->s1s2BNGS1S2BaseParams.b;
    n2 = TilingData->s1s2BNGS1S2BaseParams.n2;
    g = TilingData->s1s2BNGS1S2BaseParams.g;
    s1 = TilingData->s1s2BNGS1S2BaseParams.s1;
    s2 = TilingData->s1s2BNGS1S2BaseParams.s2;
    d = TilingData->s1s2BNGS1S2BaseParams.d;
    dV = TilingData->s1s2BNGS1S2BaseParams.dV;
    attenMaksDimS2 = TilingData->s1s2BNGS1S2BaseParams.attenMaskS2Size;

    s1Token = TilingData->s1s2BNGS1S2BaseParams.s1Token;
    s2Token = TilingData->s1s2BNGS1S2BaseParams.s2Token;
    actualCalcS1Token = s1Token;
    actualCalcS2Token = s2Token;
    sparseMode = TilingData->s1s2BNGS1S2BaseParams.sparseMode;
    isSparse = false;
    if (TilingData->s1s2BNGS1S2BaseParams.isSparse == 1) {
        isSparse = true;
    }

    // split info
    s1Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s1Outer;
    s1CvRatio = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvRatio;
    s1Inner = TilingData->s1s2BNGS1S2SplitCoreParams.s1Inner;
    s1CvInner = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvInner;
    s1CvTail = TilingData->s1s2BNGS1S2SplitCoreParams.s1CvTail;
    s2Outer = TilingData->s1s2BNGS1S2SplitCoreParams.s2Outer;
    s2CvRatio = TilingData->s1s2BNGS1S2SplitCoreParams.s2CvRatio;
    s2Inner = TilingData->s1s2BNGS1S2SplitCoreParams.s2Inner;

    s2CvInner = s2CvRatio * s2Inner;
    // 更新token总数和s1 s2方向tile数量
    Tq = TilingData->s1s2BNGS1S2BaseParams.Tq;
    Tkv = TilingData->s1s2BNGS1S2BaseParams.Tkv;
    s2TileNum = TilingData->s1s2BNGS1S2BaseParams.s2TileNum;
    gs1TileNum = TilingData->s1s2BNGS1S2BaseParams.gs1TileNum;

    sfmgdOuter = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdOuter;
    sfmgdInner = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdFactor;
    sfmgdTail = TilingData->s1s2BNGS1S2SplitCoreParams.sfmgdTail;
    sfmgdTailAlign = (sfmgdTail + 15) / 16 * 16;

    // no sparse blockouter ceil to even
    blockOuter = (TilingData->s1s2BNGS1S2SplitCoreParams.blockOuter + 1) / 2 * 2;
    tmpBufferSize = TilingData->s1s2BNGS1S2SplitCoreParams.tmpBufferSize;
    baseMN = TilingData->s1s2BNGS1S2SplitCoreParams.baseMN;
    cubeBaseMN = s1CvRatio * s2CvRatio * baseMN;

    actual_seq_qlen_addr = actual_seq_qlen;
    actual_seq_kvlen_addr = actual_seq_kvlen;

    for (uint32_t i = 0; i < 50; i++) {
        blockStarts[i] = TilingData->s1s2BNGS1S2BlockNumList.splitS2BlockStarts[i];
        blockEnds[i] = TilingData->s1s2BNGS1S2BlockNumList.splitS2BlockEnds[i];
    }

    uint64_t qPostBlockTotal = TilingData->postTilingData.qPostBlockTotal;
    uint64_t kPostBlockTotal = TilingData->postTilingData.kPostBlockTotal;
    uint64_t vPostBlockTotal = TilingData->postTilingData.vPostBlockTotal;

    // init workspace address
    syncGlobal.SetGlobalBuffer((__gm__ int32_t*)workspace);
    uint64_t workspaceOffsets = SYNC_GLOBAL_WORKSPACE_SIZE;

    // matmal1 and matmal2 workspace size
    matmalWorkspaceSize = cubeBaseMN * sizeof(float);
    mm1WorkspaceGm.SetGlobalBuffer((__gm__ T2*)(workspace + workspaceOffsets + cBlockIdx * matmalWorkspaceSize));
    mm2WorkspaceGm.SetGlobalBuffer(
        (__gm__ T2*)(workspace + workspaceOffsets + coreNum * matmalWorkspaceSize + cBlockIdx * matmalWorkspaceSize));

    // drop workspace offset
    workspaceOffsets = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(float) * INPUT_NUMS + ADDR_ALIGN_SIZE) /
                       ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    tmpWorkSpaceGm.SetGlobalBuffer((__gm__ T1*)workspace + workspaceOffsets / sizeof(T1));

    // mul workspace offset
    workspaceOffsets = (workspaceOffsets + coreNum * cubeBaseMN * sizeof(half) * 2 + ADDR_ALIGN_SIZE) /
                       ADDR_ALIGN_SIZE * ADDR_ALIGN_SIZE;
    mulWorkSpaceGm.SetGlobalBuffer((__gm__ T1*)workspace + workspaceOffsets / sizeof(T1));

    // dq reduce sum workspace offset
    uint64_t dqWorkspaceSize =
        (static_cast<uint64_t>(Tq) * (n2 * g) * d * s2TileNum * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE *
        ADDR_ALIGN_SIZE;
    uint64_t dkWorkspaceSize =
        (static_cast<uint64_t>(Tkv) * (n2 * 1) * d * gs1TileNum * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE *
        ADDR_ALIGN_SIZE;
    uint64_t dvWorkspaceSize =
        (static_cast<uint64_t>(Tkv) * (n2 * 1) * dV * gs1TileNum * sizeof(float) + ADDR_ALIGN_SIZE) / ADDR_ALIGN_SIZE *
        ADDR_ALIGN_SIZE;

    workspaceOffsets = TilingData->s1s2BNGS1S2BaseParams.dqkvWorkspaceOffset;
    dqWorkSpaceGm.SetGlobalBuffer((__gm__ T2*)workspace + workspaceOffsets / sizeof(T2));

    workspaceOffsets = workspaceOffsets + dqWorkspaceSize;
    dkWorkSpaceGm.SetGlobalBuffer((__gm__ T2*)workspace + workspaceOffsets / sizeof(T2));

    workspaceOffsets = workspaceOffsets + dkWorkspaceSize;
    dvWorkSpaceGm.SetGlobalBuffer((__gm__ T2*)workspace + workspaceOffsets / sizeof(T2));

    workspaceOffsets = workspaceOffsets + dvWorkspaceSize;
    dSinkWorkspaceGm.SetGlobalBuffer((__gm__ T2*)workspace + workspaceOffsets / sizeof(T2));

    auto dqkvWorkspaceSize = dqWorkspaceSize + dkWorkspaceSize + dvWorkspaceSize;
    auto dqkvWorkspaceFloatNum = dqkvWorkspaceSize / sizeof(float); // 这里一定是可以整除的
    auto dqkvNumPerCore = (dqkvWorkspaceFloatNum + blockOuter - 1) / blockOuter;
    auto dqkvTailNumPerCore = dqkvNumPerCore - 1;
    auto tailCoreNum = dqkvNumPerCore * blockOuter - dqkvWorkspaceFloatNum;
    auto formerCoreNum = blockOuter - tailCoreNum;
    auto curDqkvNumPerCore = cBlockIdx >= formerCoreNum ? dqkvTailNumPerCore : dqkvNumPerCore;
    auto cuDqkvNum = cBlockIdx * dqkvNumPerCore;
    if (cBlockIdx > formerCoreNum) {
        cuDqkvNum = formerCoreNum * dqkvNumPerCore + (cBlockIdx - formerCoreNum) * dqkvTailNumPerCore;
    }

    // dqkv相关workspace全0初始化
    InitOutput<float>(dqWorkSpaceGm[cuDqkvNum], curDqkvNumPerCore, 0.0);
    if (TilingData->s1s2BNGS1S2BaseParams.sinkRequiresGrad) {
        uint64_t dSinkOffset = cBlockIdx * TilingData->preTilingData.sinkPreBlockFactor;
        uint64_t initdSinkSize = cBlockIdx == TilingData->preTilingData.sinkPreBlockTotal - 1 ?
                                     TilingData->preTilingData.sinkPreBlockTail :
                                     TilingData->preTilingData.sinkPreBlockFactor;
        InitOutput<float>(dSinkWorkspaceGm[dSinkOffset], initdSinkSize, 0);
    }
    SyncAll();

    pipe->InitBuffer(ubBuffer, 132 * 1024);
    pipe->InitBuffer(tmpBuffer, 32 * 1024);
    pipe->InitBuffer(vecClc3, 8 * 1024);
    pipe->InitBuffer(softmaxBuffer, 16 * 1024);
    pipe->InitBuffer(sinkBuffer, 1 * 1024); // s1 * sizeof(float)
    GetExtremeValue();
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::GetSeqQlenKvlenByBidx(
    int64_t bIdx, int64_t& actualSeqQlen, int64_t& actualSeqKvlen)
{
    actualSeqQlen = ((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx + 1] - ((__gm__ int64_t*)actual_seq_qlen_addr)[bIdx];
    actualSeqKvlen =
        ((__gm__ int64_t*)actual_seq_kvlen_addr)[bIdx + 1] - ((__gm__ int64_t*)actual_seq_kvlen_addr)[bIdx];
    return;
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::GetExtremeValue()
{
    if constexpr (IsSameType<T2, float>::value) {
        uint32_t tmp1 = 0xFF7FFFFF;
        uint32_t tmp2 = 0x7F7FFFFF;
        negativeScalar = *((float*)&tmp1);
        positiveScalar = *((float*)&tmp2);
    } else {
        uint16_t tmp1 = 0xFBFF;
        uint16_t tmp2 = 0x7BFF;
        negativeScalar = *((half*)&tmp1);
        positiveScalar = *((half*)&tmp2);
    }

    return;
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::CopyInSoftMaxGrad(
    LocalTensor<T1>& dstTensor, LocalTensor<T1>& dstTensor2, uint64_t softmaxGradFrontOffset, uint32_t s1Extend,
    uint32_t dExtend, uint32_t dExtendAlign)
{
    uint64_t transpse_stride = 0;
    // only support TND now
    transpse_stride = (n2 * g * dV - dExtend) * sizeof(T1);

    if (transpse_stride == 0) {
        DataCopy(dstTensor, forwardResGm[softmaxGradFrontOffset], s1Extend * dExtend);
        DataCopy(dstTensor2, dxGm[softmaxGradFrontOffset], s1Extend * dExtend);
    } else {
        DataCopyPad(dstTensor, forwardResGm[softmaxGradFrontOffset],
                    {static_cast<uint16_t>(s1Extend), static_cast<uint32_t>(dExtend * sizeof(T1)),
                     static_cast<uint32_t>(transpse_stride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
        DataCopyPad(dstTensor2, dxGm[softmaxGradFrontOffset],
                    {static_cast<uint16_t>(s1Extend), static_cast<uint32_t>(dExtend * sizeof(T1)),
                     static_cast<uint32_t>(transpse_stride), 0, 0},
                    {true, 0, static_cast<uint8_t>((dExtendAlign - dExtend)), 0});
    }
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::CalcSoftMaxGrad(
    LocalTensor<float>& sfmgClc3, uint64_t aTensorOffset, uint32_t s1Extend)
{
    LocalTensor<float> sfmgClc1 = ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(T2), 0);
    LocalTensor<float> sfmgClc2 = ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(T2), 32 * 1024);
    Duplicate<float>(sfmgClc3, 0.0, s1Extend * 32 / sizeof(float));

    event_t curEventId = EVENT_ID7;
    for (uint32_t sfmgdIdx = 0; sfmgdIdx < sfmgdOuter; sfmgdIdx++) {
        LocalTensor<T1> vecInBuffer = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), 64 * 1024);
        LocalTensor<T1> vecInBuffer2 = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), 80 * 1024);
        LocalTensor<uint8_t> vecOutBuffer = ubBuffer.GetWithOffset<uint8_t>(32 * 1024 / sizeof(uint8_t), 96 * 1024);
        LocalTensor<T2> softmaxGradTmp = tmpBuffer.GetWithOffset<T2>(8 * 1024 / sizeof(T2), 0);
        uint64_t softmaxGradFrontOffset = aTensorOffset + sfmgdIdx * sfmgdInner;
        uint32_t dExtend = (sfmgdIdx == sfmgdOuter - 1) ? sfmgdTail : sfmgdInner;
        uint32_t dExtendAlign = (sfmgdIdx == sfmgdOuter - 1) ? sfmgdTailAlign : sfmgdInner;
        bool isBasicBlock = (s1Extend % 8 == 0) && (dExtend % 64 == 0);

        if (sfmgdIdx > 0) {
            wait_flag(PIPE_V, PIPE_MTE2, curEventId);
        }
        // vecInBuffer  = sfmgClc1: dout
        // vecInBuffer2 = sfmgClc2: out
        // @lg: 读O和dO,从softmaxGradFrontOffset这个位置读s1Extend这么多行到ub local tensor，ub
        // tensor对齐到dExtendAlign，右侧补0
        // @lg: ub tensor行、列都有冗余，内存上连续存储，相当于每次copy dExtendAlign数据量，拷贝s1Extend次，后续在使用该
        // local tensor时，需要主动SetShapeInfo
        CopyInSoftMaxGrad(vecInBuffer, vecInBuffer2, softmaxGradFrontOffset, s1Extend, dExtend, dExtendAlign);
        set_flag(PIPE_MTE2, PIPE_V, curEventId);
        wait_flag(PIPE_MTE2, PIPE_V, curEventId);
        Cast(sfmgClc1, vecInBuffer, RoundMode::CAST_NONE, s1Extend * dExtendAlign);
        Cast(sfmgClc2, vecInBuffer2, RoundMode::CAST_NONE, s1Extend * dExtendAlign);
        PipeBarrier<PIPE_V>();
        uint32_t shapeArray[2];
        shapeArray[0] = s1Extend;
        shapeArray[1] = dExtendAlign;
        sfmgClc1.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        sfmgClc2.SetShapeInfo(ShapeInfo(2, shapeArray, DataFormat::ND));
        uint32_t shapeArray1[2];
        shapeArray1[0] = s1Extend;
        shapeArray1[1] = 32 / sizeof(float);
        softmaxGradTmp.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));

        if (isBasicBlock) {
            SoftmaxGradFront<float, true>(softmaxGradTmp, sfmgClc1, sfmgClc2, vecOutBuffer,
                                          TilingData->softmaxGradTilingData);
        } else {
            SoftmaxGradFront<float, false>(softmaxGradTmp, sfmgClc1, sfmgClc2, vecOutBuffer,
                                           TilingData->softmaxGradTilingData);
        }
        PipeBarrier<PIPE_V>();

        Add(sfmgClc3, softmaxGradTmp, sfmgClc3, s1Extend * 32 / sizeof(float));
        if (sfmgdIdx < (sfmgdOuter - 1)) {
            set_flag(PIPE_V, PIPE_MTE2, curEventId);
        }
    }
    pipe_barrier(PIPE_ALL);
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::CopyInSoftMax(
    LocalTensor<float>& dstTensor, uint32_t s1Extend, uint32_t softMaxOffset)
{
    if (TilingData->s1s2BNGS1S2BaseParams.softmaxLseOptional) {
        DataCopyPad(dstTensor, softmaxLseGm[softMaxOffset], {static_cast<uint16_t>(s1Extend), sizeof(T2), 0, 0},
                    {false, 0, 0, 0});
    } else {
        DataCopyPad(dstTensor, softmaxMaxGm[softMaxOffset], {static_cast<uint16_t>(s1Extend), sizeof(T2), 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(dstTensor[s1Extend * 32 / sizeof(float)], softmaxSumGm[softMaxOffset],
                    {static_cast<uint16_t>(s1Extend), sizeof(T2), 0, 0}, {false, 0, 0, 0});
    }
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::CalcSoftMax(
    LocalTensor<T2>& dstTensor, LocalTensor<float>& srcTensor, uint32_t s1Extend, uint32_t s2Extend,
    uint32_t s2ExtendAlign, const SoftMaxTiling& tiling)
{
    bool isBasicBlock = (s1Extend % 8 == 0) && (s2Extend % 64 == 0);

    if (TilingData->s1s2BNGS1S2BaseParams.softmaxLseOptional) {
        LocalTensor<T2> vecOutBuffer = tmpBuffer.Get<T2>();
        uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

        for (uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
            uint32_t subMaskCount =
                (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
            Sub(vecOutBuffer[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num], srcTensor, subMaskCount,
                s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8), 1});
            PipeBarrier<PIPE_V>();
            Exp(dstTensor[subIdx * cal_repeat_num], vecOutBuffer[subIdx * cal_repeat_num], subMaskCount, s1Extend,
                {static_cast<uint8_t>(1), static_cast<uint8_t>(1), static_cast<uint8_t>(s2ExtendAlign / 8),
                 static_cast<uint8_t>(s2ExtendAlign / 8)});
            PipeBarrier<PIPE_V>();
        }
    } else {
        if (isBasicBlock) {
            LocalTensor<uint8_t> vecOutBuffer = tmpBuffer.Get<uint8_t>();
            uint32_t shapeArray1[2];
            shapeArray1[0] = s1Extend;
            shapeArray1[1] = s2Extend;
            dstTensor.SetShapeInfo(ShapeInfo(2, shapeArray1, DataFormat::ND));

            SimpleSoftMax<T2, true, true>(dstTensor, srcTensor[s1Extend * 32 / sizeof(float)], srcTensor, dstTensor,
                                          vecOutBuffer, tiling);

        } else {
            LocalTensor<T2> vecOutBuffer = tmpBuffer.Get<T2>();
            uint32_t sub_block_count = (s2Extend + cal_repeat_num - 1) / cal_repeat_num;

            for (uint32_t subIdx = 0; subIdx < sub_block_count; subIdx++) {
                uint32_t subMaskCount =
                    (subIdx == sub_block_count - 1) ? (s2Extend - subIdx * cal_repeat_num) : cal_repeat_num;
                Sub(dstTensor[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num], srcTensor, subMaskCount,
                    s1Extend,
                    {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                     static_cast<uint8_t>(s2ExtendAlign / 8), 1});
                PipeBarrier<PIPE_V>();
                Exp(vecOutBuffer[subIdx * cal_repeat_num], dstTensor[subIdx * cal_repeat_num], subMaskCount, s1Extend,
                    {static_cast<uint8_t>(1), static_cast<uint8_t>(1), static_cast<uint8_t>(s2ExtendAlign / 8),
                     static_cast<uint8_t>(s2ExtendAlign / 8)});
                PipeBarrier<PIPE_V>();
                Div(dstTensor[subIdx * cal_repeat_num], vecOutBuffer[subIdx * cal_repeat_num], srcTensor[s1Extend * 8],
                    subMaskCount, s1Extend,
                    {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
                     static_cast<uint8_t>(s2ExtendAlign / 8), 1});
                PipeBarrier<PIPE_V>();
            }
        }
    }
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::CopyInAttenMaskBool(
    LocalTensor<uint8_t>& dstTensor, uint64_t attenMaskOffset, uint32_t s1Extend, uint32_t s2Extend)
{
    DataCopyPad(dstTensor, attenMaskU8Gm[attenMaskOffset],
                {static_cast<uint16_t>(s1Extend), static_cast<uint16_t>(s2Extend * sizeof(uint8_t)),
                 static_cast<uint16_t>((attenMaksDimS2 - s2Extend) * sizeof(uint8_t)), 0},
                {false, 0, 0, 0});
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::CalcAttenMaskBool(
    LocalTensor<T2>& dstTensor, LocalTensor<uint8_t> srcTensor, uint32_t s1Extend, uint32_t s2Extend, uint8_t maskType)
{
    LocalTensor<uint8_t> tmpUbBuffer = tmpBuffer.Get<uint8_t>();

    T2 scalar;
    if constexpr (IsSameType<T2, float>::value) {
        uint32_t tmp = 0xFF7FFFFF;
        // @lg: scalar即-inf
        scalar = *((float*)&tmp);
    } else {
        uint16_t tmp = 0xFBFF;
        scalar = *((half*)&tmp);
    }

    SelectWithBytesMaskShapeInfo info;
    info.firstAxis = s1Extend;
    info.srcLastAxis = s2Extend;
    info.maskLastAxis = (s2Extend * sizeof(uint8_t) + 31) / 32 * 32 / sizeof(uint8_t);
    dstTensor.SetSize(info.firstAxis * info.srcLastAxis);
    srcTensor.SetSize(info.firstAxis * info.maskLastAxis);
    if (maskType == 0) {
        SelectWithBytesMask(dstTensor, dstTensor, scalar, srcTensor, tmpUbBuffer, info);
    } else {
        SelectWithBytesMask(dstTensor, scalar, dstTensor, srcTensor, tmpUbBuffer, info);
    }
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::CopyInPrmMaskBool(
    LocalTensor<uint8_t>& dstTensor, uint64_t prmMaskOffset, uint32_t s1Extend, uint32_t s2Extend)
{
    int64_t alignS2MaskSize = (s2Extend + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    DataCopyParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = s2Extend * sizeof(uint8_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadParams dataCopyPadParams;
    dataCopyPadParams.isPad = true;
    dataCopyPadParams.rightPadding = alignS2MaskSize - s2Extend;
    dataCopyPadParams.paddingValue = 0;
    DataCopyPad(dstTensor, prmMaskGm[prmMaskOffset], dataCopyParams, dataCopyPadParams);
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::CalcPrmMaskBool(
    LocalTensor<T2>& dstTensor, LocalTensor<uint8_t> srcTensor, uint32_t s1Extend, uint32_t s2Extend)
{
    if (!TilingData->s1s2BNGS1S2BaseParams.prmMaskOptional) {
        return;
    }
    int64_t alignS2MaskSize = (s2Extend + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
    int64_t loop = alignS2MaskSize / repeatMaxBytes; // repeatMaxBytes = 256
    int64_t remain = alignS2MaskSize % repeatMaxBytes;
    uint8_t repeaTimes = s1Extend;
    int64_t dstRepStride = alignS2MaskSize / BLOCK_SIZE;
    int64_t repeatMaxSize16 = repeatMaxBytes / sizeof(half);
    uint64_t mask = repeatMaxSize16;

    if (repeaTimes > 1) {
        auto prmMaskUbTmp = srcTensor.ReinterpretCast<int16_t>();
        UnaryRepeatParams repeatParams;
        repeatParams.srcBlkStride = 1;
        repeatParams.dstBlkStride = 1;
        repeatParams.srcRepStride = 0;
        repeatParams.dstRepStride = dstRepStride;

        for (int i = 0; i < loop; i++) {
            Adds(prmMaskUbTmp[i * repeatMaxSize16], prmMaskUbTmp[i * repeatMaxSize16], (int16_t)0, mask, repeaTimes,
                 repeatParams);
        }
        if (likely(remain)) {
            mask = (remain + sizeof(half) - 1) / sizeof(half);
            Adds(prmMaskUbTmp[loop * repeatMaxSize16], prmMaskUbTmp[loop * repeatMaxSize16], (int16_t)0, mask,
                 repeaTimes, repeatParams);
        }
    }
    PipeBarrier<PIPE_V>();

    SelectWithBytesMaskShapeInfo shapeInfo;
    shapeInfo.firstAxis = s1Extend;
    shapeInfo.srcLastAxis = s2Extend;
    shapeInfo.maskLastAxis = alignS2MaskSize;
    dstTensor.SetSize(shapeInfo.firstAxis * shapeInfo.srcLastAxis);
    srcTensor.SetSize(shapeInfo.firstAxis * shapeInfo.maskLastAxis);

    LocalTensor<uint8_t> tmpUbBuffer = tmpBuffer.Get<uint8_t>();
    SelectWithBytesMask(dstTensor, negativeScalar, dstTensor, srcTensor, tmpUbBuffer, shapeInfo);
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::Process()
{
    gs1Offset = (gs1TileNum == 2) ? cBlockIdx % 2 : TilingData->s1s2BNGS1S2BlockNumList.gs1Offset[cBlockIdx];
    if (isSparse) {
        int64_t preIndex = blockStarts[cBlockIdx];
        if (blockEnds[cBlockIdx] == 0) {
            return;
        }
        InitIndex(preIndex);
        for (int64_t blockInnerIdx = blockStarts[cBlockIdx] + 1; blockInnerIdx < blockEnds[cBlockIdx] + 1;
             blockInnerIdx++) {
            if (isStart == 1) {
                if (!IfCubeBlockNeedCompute(preIndex)) {
                    preIndex = blockInnerIdx;
                    continue;
                }
            }
            preS2CvBegin = nextS2CvBegin;
            preS2CvEnd = nextS2CvEnd;
            if (IfCubeBlockNeedCompute(blockInnerIdx) || (blockInnerIdx >= blockEnds[cBlockIdx])) {
                int64_t nextIndex = blockInnerIdx;
                if (blockInnerIdx >= blockEnds[cBlockIdx]) {
                    nextIndex = 0u;
                }
                Compute(preIndex, nextIndex);
                preIndex = nextIndex;
            }
        }
    } else {
        uint32_t preIndex = cBlockIdx;
        uint32_t total = b * n2 * g * s1Outer * s2Outer;
        if (cBlockIdx >= total) {
            return;
        }

        IfCubeBlockNeedCompute(preIndex);
        preS2CvBegin = nextS2CvBegin;
        preS2CvEnd = nextS2CvEnd;

        int64_t totalTemp = total + blockOuter;
        for (int64_t blockInnerIdx = cBlockIdx + blockOuter; blockInnerIdx < totalTemp; blockInnerIdx += blockOuter) {
            if (IfCubeBlockNeedCompute(blockInnerIdx) || (blockInnerIdx >= total)) {
                int64_t nextIndex = blockInnerIdx;
                if (blockInnerIdx >= total) {
                    nextIndex = 0u;
                }
                Compute(preIndex, nextIndex);
                preIndex = nextIndex;
                preS2CvBegin = nextS2CvBegin;
                preS2CvEnd = nextS2CvEnd;
            }
        }
        Compute(preIndex, 0u);
    }
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::UpdateCalToken(int64_t bIdx)
{
    if (TilingData->s1s2BNGS1S2BaseParams.attenMaskOptional) {
        int64_t actualS1Tmp;
        int64_t actualS2Tmp;
        if (sparseMode == 3 || sparseMode == 4) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Tmp, actualS2Tmp);
            actualCalcS1Token = s1Token + actualS1Tmp - actualS2Tmp;
            actualCalcS2Token = s2Token - actualS1Tmp + actualS2Tmp;
        }
    }
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline bool FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::IfCubeBlockNeedCompute(
    uint32_t baseIdx)
{
    int64_t resbaseIdx = baseIdx < 0 ? 0 : baseIdx;
    if (resbaseIdx < totalBaseIdx) {
        int32_t s1OuterTmp = (actualS1Len + s1CvInner - 1) / s1CvInner;
        int32_t s2OuterTmp = (actualS2Len + s2CvInner - 1) / s2CvInner;
        int32_t s1s2OuterTmp = s1OuterTmp * s2OuterTmp;

        int32_t gDimTail_ = (resbaseIdx - preTotal) % (g * s1s2OuterTmp);
        int32_t s2oCvDimIdx_ = gDimTail_ / (g * s1OuterTmp);
        int32_t s1oDimIdx_ = gDimTail_ % s1OuterTmp;
        int32_t s2IdxLeft = s2oCvDimIdx_ * s2CvInner;
        int32_t s2IdxRight =
            (s2oCvDimIdx_ + 1) * s2CvInner < actualS2Len ? (s2oCvDimIdx_ + 1) * s2CvInner : actualS2Len;
        UpdateCalToken(bDimIdx);
        int32_t s2SparseLeftPre = int64_t(s1CvInner * s1oDimIdx_) - actualCalcS1Token;
        int32_t s2SparseLeft = s2SparseLeftPre < 0 ? 0 : s2SparseLeftPre;
        s2SparseLeft = s2SparseLeft / 64 * 64;
        int32_t s2SparseRight = (s1CvInner * (s1oDimIdx_ + 1) + actualCalcS2Token + 63) / 64 * 64 < 0 ?
                                    0 :
                                    (s1CvInner * (s1oDimIdx_ + 1) + actualCalcS2Token + 63) / 64 * 64;
        s2SparseRight = static_cast<uint32_t>(s2SparseRight) < actualS2Len ? s2SparseRight : actualS2Len;
        if (s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft) {
            nextS2CvBegin = s2IdxLeft < s2SparseLeft ? s2SparseLeft : s2IdxLeft;
            nextS2CvEnd = s2IdxRight > s2SparseRight ? s2SparseRight : s2IdxRight;
            return true;
        } else {
            return false;
        }
    } else {
        int64_t actualS1Len1;
        int64_t actualS2Len1;
        for (int64_t bIdx = bDimIdx + 1; bIdx < b; bIdx++) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len1, actualS2Len1);
            int32_t s1OuterTmp = (actualS1Len1 + s1CvInner - 1) / s1CvInner;
            int32_t s2OuterTmp = (actualS2Len1 + s2CvInner - 1) / s2CvInner;
            int32_t s1s2OuterTmp = s1OuterTmp * s2OuterTmp;
            int64_t preTotal1 = totalBaseIdx;
            int64_t totalBaseIdx1 = totalBaseIdx + static_cast<int64_t>(n2) * g * s1s2OuterTmp;
            if (resbaseIdx < totalBaseIdx1) {
                int32_t gDimTail_ = (resbaseIdx - preTotal1) % (g * s1s2OuterTmp);
                int32_t s2oCvDimIdx_ = gDimTail_ / (g * s1OuterTmp);
                int32_t s1oDimIdx_ = gDimTail_ % s1OuterTmp;
                int32_t s2IdxLeft = s2oCvDimIdx_ * s2CvInner;
                int32_t s2IdxRight =
                    (s2oCvDimIdx_ + 1) * s2CvInner < actualS2Len1 ? (s2oCvDimIdx_ + 1) * s2CvInner : actualS2Len1;
                // 得到当前s1oDimIdx、s2IdxLeft、s2IdxRight
                UpdateCalToken(bIdx);
                int32_t s2SparseLeftPre = int64_t(s1CvInner * s1oDimIdx_) - actualCalcS1Token;
                int32_t s2SparseLeft = s2SparseLeftPre < 0 ? 0 : s2SparseLeftPre;
                s2SparseLeft = s2SparseLeft / 64 * 64;
                int32_t s2SparseRight = (s1CvInner * (s1oDimIdx_ + 1) + actualCalcS2Token + 63) / 64 * 64 < 0 ?
                                            0 :
                                            (s1CvInner * (s1oDimIdx_ + 1) + actualCalcS2Token + 63) / 64 * 64;
                s2SparseRight = static_cast<uint32_t>(s2SparseRight) < actualS2Len1 ? s2SparseRight : actualS2Len1;
                if (s2IdxLeft < s2SparseRight && s2IdxRight > s2SparseLeft) {
                    nextS2CvBegin = s2IdxLeft < s2SparseLeft ? s2SparseLeft : s2IdxLeft;
                    nextS2CvEnd = s2IdxRight > s2SparseRight ? s2SparseRight : s2IdxRight;
                    return true;
                } else {
                    return false;
                }
            }
        }
    }
    return false;
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::CalcAttenMaskOffset(
    uint64_t& attenMaskOffset, const int64_t delta, uint32_t s1VSize, uint32_t s2VSize)
{
    if (delta == 0) {
        attenMaskOffset = 0;
    } else if (delta < 0) {
        if (-delta > s1VSize) {
            attenMaskOffset = s1VSize;
        } else {
            attenMaskOffset = -delta;
        }
    } else {
        if (delta > s2VSize) {
            attenMaskOffset = s2VSize * attenMaksDimS2;
        } else {
            attenMaskOffset = delta * attenMaksDimS2;
        }
    }
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::CalSinkGrad(
    LocalTensor<T2>& srcTensor, uint32_t s1Extend, uint32_t sfmgInnerOffset, uint32_t curS1Idx)
{
    LocalTensor<float> sfmgClc3 = vecClc3.Get<float>()[sfmgInnerOffset * cal_block_num];
    int32_t calcSize = s1Extend * cal_block_num;
    LocalTensor<T2> sinkUbFloat = tmpBuffer.template GetWithOffset<T2>(calcSize, 0);
    LocalTensor<T2> tmpTensor = tmpBuffer.template GetWithOffset<T2>(calcSize, calcSize * sizeof(T2));
    LocalTensor<T2> sinkTensor = sinkBuffer.GetWithOffset<float>(1 * 1024 / sizeof(float), 0);

    int32_t nIdx = n2DimIdxTmp * g + gDimIdxTmp;
    float sinkValue = 0;
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        sinkValue = ToFloat(sinkGm.GetValue(nIdx));
    } else {
        sinkValue = static_cast<T2>(sinkGm.GetValue(nIdx));
    }

    // s1Extend * 32 / sizeof(float)
    Duplicate(sinkUbFloat, sinkValue, calcSize);
    PipeBarrier<PIPE_V>();
    Duplicate(tmpTensor, float(0), calcSize);
    PipeBarrier<PIPE_V>();
    Sub(tmpTensor, tmpTensor, srcTensor, calcSize);
    PipeBarrier<PIPE_V>();
    Exp(tmpTensor, tmpTensor, calcSize);
    PipeBarrier<PIPE_V>();
    Mul(sinkUbFloat, sinkUbFloat, tmpTensor, calcSize);
    PipeBarrier<PIPE_V>();
    Add(sinkUbFloat, srcTensor[s1Extend * 8], sinkUbFloat, calcSize);
    PipeBarrier<PIPE_V>();
    Div(sinkUbFloat, sfmgClc3, sinkUbFloat, calcSize);
    PipeBarrier<PIPE_V>();
    Duplicate(tmpTensor, float(-1), calcSize);
    PipeBarrier<PIPE_V>();
    Mul(sinkUbFloat, sinkUbFloat, tmpTensor, calcSize);
    PipeBarrier<PIPE_V>();

    int32_t repeat_num = (s1Extend * cal_block_num) / cal_repeat_num;
    int32_t remain_num = (s1Extend * cal_block_num) - repeat_num * cal_repeat_num;

    BlockReduceMax<float>(sinkUbFloat, sinkUbFloat, repeat_num, cal_repeat_num, 1, 1, 8);
    if (likely(remain_num)) {
        BlockReduceMax<float>(sinkUbFloat[repeat_num * cal_repeat_num / 8], sinkUbFloat[repeat_num * cal_repeat_num], 1,
                              remain_num, 1, 1, 8);
    }
    PipeBarrier<PIPE_V>();
    // 32k for temp
    LocalTensor<T2> sharedTmpBuffer = ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), 0);
    ReduceSum<float>(sinkUbFloat, sinkUbFloat, sharedTmpBuffer, s1Extend);
    PipeBarrier<PIPE_V>();
    uint32_t alignSize = (s1Extend + cal_block_num - 1) / cal_block_num * cal_block_num;

    Add(sinkTensor, sinkTensor, sinkUbFloat, alignSize);
    if (curS1Idx == s1VecLoop - 1 && s2oCvDimIdx == 0) {
        event_t eventIdVtoMTE3 = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>());
        SetFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVtoMTE3);
        int64_t dSinkOffset =
            nIdx * TilingData->s1s2BNGS1S2SplitCoreParams.totalS1TileNums + accumlateTileS1 + s1oDimIdxTmp;
        DataCopyPad(dSinkWorkspaceGm[dSinkOffset], sinkTensor,
                    {static_cast<uint16_t>(1), static_cast<uint16_t>(1 * sizeof(float)), static_cast<uint16_t>(0),
                     static_cast<uint16_t>(0)});
    }
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::InitIndex(uint32_t index)
{
    int64_t resbaseIdx = index < 0 ? 0 : index;
    if (unlikely(isFirstInitIndex == 1)) {
        for (uint32_t bIdx = 0; bIdx < b; bIdx++) {
            GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
            uint32_t s1OuterTmp = (actualS1Len + s1CvInner - 1) / s1CvInner;
            uint32_t s2OuterTmp = (actualS2Len + s2CvInner - 1) / s2CvInner;
            preTotal = totalBaseIdx;
            totalBaseIdx += static_cast<int64_t>(n2) * g * s1OuterTmp * s2OuterTmp;
            if (resbaseIdx < totalBaseIdx) {
                uint32_t s1CvTailTmp = actualS1Len - (s1OuterTmp - 1) * s1CvInner;
                bDimIdx = bIdx;
                bDimTail = resbaseIdx - preTotal;
                n2DimIdx = bDimTail / (g * s1OuterTmp * s2OuterTmp);
                n2DimTail = bDimTail % (g * s1OuterTmp * s2OuterTmp);
                gDimIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
                s2oCvDimIdx = n2DimTail / (g * s1OuterTmp);
                s2oCvDimTail = n2DimTail % (s1OuterTmp * g);
                gDimIdx = s2oCvDimTail / s1OuterTmp;
                gDimTail = s2oCvDimTail % s1OuterTmp;
                s1oDimIdx = gDimTail % s1OuterTmp;
                s1CvExtend = (s1oDimIdx == s1OuterTmp - 1) ? s1CvTailTmp : s1CvInner;
                break;
            } else {
                accumlateTileS1 += s1OuterTmp;
            }
        }
    } else {
        if (resbaseIdx < totalBaseIdx) {
            uint32_t s1OuterTmp = (actualS1Len + s1CvInner - 1) / s1CvInner;
            uint32_t s2OuterTmp = (actualS2Len + s2CvInner - 1) / s2CvInner;
            uint32_t s1CvTailTmp = actualS1Len - (s1OuterTmp - 1) * s1CvInner;
            bDimTail = resbaseIdx - preTotal;
            n2DimIdx = bDimTail / (g * s1OuterTmp * s2OuterTmp);
            n2DimTail = bDimTail % (g * s1OuterTmp * s2OuterTmp);
            gDimIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
            // gDimTail = n2DimTail % (s1OuterTmp * s2OuterTmp);
            s2oCvDimIdx = n2DimTail / (g * s1OuterTmp);
            s2oCvDimTail = n2DimTail % (s1OuterTmp * g);
            gDimIdx = s2oCvDimTail / s1OuterTmp;
            gDimTail = s2oCvDimTail % s1OuterTmp;
            s1oDimIdx = gDimTail % s1OuterTmp;
            s1CvExtend = (s1oDimIdx == s1OuterTmp - 1) ? s1CvTailTmp : s1CvInner;
        } else {
            // 进到这个分支，先把前一个batch的TileS1加上，再查找下一个batch。下一个batch肯定是要计算的。下面的循环只走一次，便break；
            GetSeqQlenKvlenByBidx(bDimIdx, actualS1Len, actualS2Len);
            uint32_t s1OuterTmp = (actualS1Len + s1CvInner - 1) / s1CvInner;
            accumlateTileS1 += s1OuterTmp;
            bDimIdx += 1;
            for (uint32_t bIdx = bDimIdx; bIdx < b; bIdx++) {
                GetSeqQlenKvlenByBidx(bIdx, actualS1Len, actualS2Len);
                uint32_t s1OuterTmp = (actualS1Len + s1CvInner - 1) / s1CvInner;
                uint32_t s2OuterTmp = (actualS2Len + s2CvInner - 1) / s2CvInner;
                preTotal = totalBaseIdx;
                totalBaseIdx += static_cast<int64_t>(n2) * g * s1OuterTmp * s2OuterTmp;
                if (resbaseIdx < totalBaseIdx) {
                    uint32_t s1CvTailTmp = actualS1Len - (s1OuterTmp - 1) * s1CvInner;
                    bDimTail = resbaseIdx - preTotal;
                    n2DimIdx = bDimTail / (g * s1OuterTmp * s2OuterTmp);
                    n2DimTail = bDimTail % (g * s1OuterTmp * s2OuterTmp);
                    gDimIdx = n2DimTail / (s1OuterTmp * s2OuterTmp);
                    // gDimTail = n2DimTail % (s1OuterTmp * s2OuterTmp);
                    s2oCvDimIdx = n2DimTail / (g * s1OuterTmp);
                    s2oCvDimTail = n2DimTail % (s1OuterTmp * g);
                    gDimIdx = s2oCvDimTail / s1OuterTmp;
                    gDimTail = s2oCvDimTail % s1OuterTmp;
                    s1oDimIdx = gDimTail % s1OuterTmp;
                    s1CvExtend = (s1oDimIdx == s1OuterTmp - 1) ? s1CvTailTmp : s1CvInner;
                    break;
                }
            }
        }
    }
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::SubGrapA(uint32_t curIdx,
                                                                                                  uint32_t curS1Idx,
                                                                                                  uint32_t curS2Idx)
{
    s2Extend = (curS2Idx == s2VecLoop - 1) ? (s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
    s2ExtendAlign = (s2Extend + 15) / 16 * 16;
    uint32_t s2VBegin = preS2CvBegin + curS2Idx * s2VecSize;
    int64_t actualS1Tmp;
    int64_t actualS2Tmp;
    GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Tmp, actualS2Tmp);

    event_t curEventId = ((curIdx % 2) == 0) ? EVENT_ID7 : EVENT_ID6;
    uint32_t ubBufferOffset = ((curIdx % 2) == 0) ? 0 : DbBegin;

    if (curIdx > 1) {
        wait_flag(PIPE_MTE3, PIPE_MTE2, curEventId);
    }
    LocalTensor<uint8_t> attenMaskUbuint8 =
        ubBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + MaskBegin);
    uint64_t attenMaskOffsetPre = 0;
    if (TilingData->s1s2BNGS1S2BaseParams.attenMaskOptional) {
        uint64_t attenMaskOffset = 0;
        uint64_t compressMode = TilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode;
        int64_t causal_delta = (int64_t)(s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) - (int64_t)s2VBegin;
        if (compressMode == 1) { // causal s1==s2
            attenMaksDimS2 = 2048;
            CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1ExtendSubGraph, s2Extend);
        } else if (compressMode == 2) { // causal s1!=s2
            attenMaksDimS2 = 2048;
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Tmp, actualS2Tmp);
            causal_delta = causal_delta - actualS1Tmp + actualS2Tmp;
            CalcAttenMaskOffset(attenMaskOffset, causal_delta, s1ExtendSubGraph, s2Extend);
        } else if (compressMode == 3) { // band
            attenMaksDimS2 = 2048;
            int64_t pre_delta = causal_delta - actualCalcS1Token - 1;
            CalcAttenMaskOffset(attenMaskOffsetPre, pre_delta, s1ExtendSubGraph, s2Extend);
            int64_t next_delta = causal_delta + actualCalcS2Token;
            CalcAttenMaskOffset(attenMaskOffset, next_delta, s1ExtendSubGraph, s2Extend);
        } else if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 0) {
            /*
            // "This code is intended for a Tq*Tkv-sized mask.
            // Due to the enormous GPU memory overhead, we temporarily avoid using it.
            // Instead, we recommend passing masks as a list of tensors to reduce memory consumption."
            int64_t curSumS1Token = ((__gm__ int64_t *)actual_seq_qlen_addr)[bDimIdxTmp];
            int64_t curSumS2Token = ((__gm__ int64_t *)actual_seq_kvlen_addr)[bDimIdxTmp];
            attenMaksDimS2 = (uint32_t)(TilingData->s1s2BNGS1S2BaseParams.Tkv);
            attenMaskOffset += (curSumS1Token + s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) *
            (uint32_t)(TilingData->s1s2BNGS1S2BaseParams.Tkv) + s2VBegin + curSumS2Token;
            */
            attenMaksDimS2 = (uint32_t)s2;
            attenMaskOffset += (s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) * s2 + s2VBegin;
        } else if (TilingData->s1s2BNGS1S2BaseParams.attenMaskShapeType == 1) { // for mask tensor lists
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Tmp, actualS2Tmp);
            attenMaksDimS2 = (uint32_t)actualS2Tmp;
            for (uint32_t bidx = 0; bidx < bDimIdxTmp; bidx++) {
                GetSeqQlenKvlenByBidx(bidx, actualS1Tmp, actualS2Tmp);
                attenMaskOffset += actualS1Tmp * actualS2Tmp;
            }
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Tmp, actualS2Tmp);
            attenMaskOffset += (s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) * (int64_t)actualS2Tmp + s2VBegin;
        } else {
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Tmp, actualS2Tmp);
            attenMaksDimS2 = (uint32_t)actualS2Tmp;
            for (uint32_t bidx = 0; bidx < bDimIdxTmp; bidx++) {
                GetSeqQlenKvlenByBidx(bidx, actualS1Tmp, actualS2Tmp);
                attenMaskOffset += n2 * g * actualS1Tmp * actualS2Tmp;
            }
            GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Tmp, actualS2Tmp);
            attenMaskOffset +=
                ((n2DimIdxTmp * g + gDimIdxTmp) * actualS1Tmp + s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize) *
                    actualS2Tmp +
                s2VBegin;
        }
        // uint8_t
        CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffset, s1ExtendSubGraph, s2Extend);
    }

    // mm2 res is fine
    // @lg mm2保存的是S矩阵
    LocalTensor<float> vecClc2Buffer =
        ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);
    if (s2VecLoop == 1) {
        DataCopy(vecClc2Buffer, mm2WorkspaceGm[curS1Idx * s1VecSize * s2ExtendAlign], s1ExtendSubGraph * s2ExtendAlign);
    } else {
        DataCopyPad(vecClc2Buffer, mm2WorkspaceGm[curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                    {false, 0, 0, 0});
    }
    set_flag(PIPE_MTE2, PIPE_V, curEventId);
    wait_flag(PIPE_MTE2, PIPE_V, curEventId);

    PipeBarrier<PIPE_V>();
    Muls(vecClc2Buffer, vecClc2Buffer, (T2)(TilingData->s1s2BNGS1S2BaseParams.scaleValue),
         s1ExtendSubGraph * s2ExtendAlign);
    ///////////////////////////////////////////////////////////////
    // attenMask
    ///////////////////////////////////////////////////////////////
    // attenMaskOffset     attenMaskShapeType  0--111S1S2        1--B11S1S2         2--BN2GS1S2
    if (TilingData->s1s2BNGS1S2BaseParams.attenMaskOptional) {
        PipeBarrier<PIPE_V>();
        // uint8_t
        CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign);
        if (TilingData->s1s2BNGS1S2BaseParams.attenMaskCompressMode == 3) { // band
            set_flag(PIPE_V, PIPE_MTE2, curEventId);
            wait_flag(PIPE_V, PIPE_MTE2, curEventId);
            CopyInAttenMaskBool(attenMaskUbuint8, attenMaskOffsetPre, s1ExtendSubGraph, s2Extend);
            set_flag(PIPE_MTE2, PIPE_V, curEventId);
            wait_flag(PIPE_MTE2, PIPE_V, curEventId);
            CalcAttenMaskBool(vecClc2Buffer, attenMaskUbuint8, s1ExtendSubGraph, s2ExtendAlign, 1);
        }
    }
    // 复用atten mask buffer
    if (TilingData->s1s2BNGS1S2BaseParams.prmMaskOptional) {
        PipeBarrier<PIPE_V>();
        LocalTensor<uint8_t> prmMaskBuffer =
            ubBuffer.GetWithOffset<uint8_t>(8 * 1024 / sizeof(uint8_t), ubBufferOffset + MaskBegin);
        auto prmOffset = ((__gm__ int64_t*)actual_seq_kvlen_addr)[bDimIdxTmp] + s2VBegin;
        set_flag(PIPE_V, PIPE_MTE2, curEventId);
        wait_flag(PIPE_V, PIPE_MTE2, curEventId);
        CopyInPrmMaskBool(prmMaskBuffer, prmOffset, s1ExtendSubGraph, s2Extend);
        set_flag(PIPE_MTE2, PIPE_V, curEventId);
        wait_flag(PIPE_MTE2, PIPE_V, curEventId);
        CalcPrmMaskBool(vecClc2Buffer, prmMaskBuffer, s1ExtendSubGraph, s2ExtendAlign);
    }
    ///////////////////////////////////////////////////////////////
    // simpleSoftMax
    ///////////////////////////////////////////////////////////////
    PipeBarrier<PIPE_V>();
    uint32_t dBoffset = ((curS1Idx % 2) == 0) ? 0 : 8 * 1024;
    LocalTensor<float> softmaxTensor = softmaxBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), dBoffset);
    CalcSoftMax(vecClc2Buffer, softmaxTensor, s1ExtendSubGraph, s2Extend, s2ExtendAlign, TilingData->softmaxTilingData);
    ///////////////////////////////////////////////////////////////
    // cast fp322bf16
    ///////////////////////////////////////////////////////////////
    LocalTensor<T1> vecOut1Buffer1 = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
    PipeBarrier<PIPE_V>();
    Cast(vecOut1Buffer1, vecClc2Buffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);
    set_flag(PIPE_V, PIPE_MTE3, curEventId);
    wait_flag(PIPE_V, PIPE_MTE3, curEventId);
    DataCopyPad(tmpWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                               curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                vecOut1Buffer1,
                {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                 static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
    set_flag(PIPE_MTE3, PIPE_MTE2, curEventId);
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::SubGrapB(uint32_t curIdx,
                                                                                                  uint32_t curS1Idx,
                                                                                                  uint32_t curS2Idx)
{
    event_t curEventId = ((curIdx % 2) == 0) ? EVENT_ID7 : EVENT_ID6;
    uint32_t ubBufferOffset = ((curIdx % 2) == 0) ? 0 : DbBegin;
    s2Extend = (curS2Idx == s2VecLoop - 1) ? (s2CvExtend - (s2VecLoop - 1) * s2VecSize) : s2VecSize;
    s2ExtendAlign = (s2Extend + 15) / 16 * 16;
    int64_t actualS1Tmp;
    int64_t actualS2Tmp;

    wait_flag(PIPE_MTE3, PIPE_MTE2, curEventId);
    // @lg 把dP拷贝进来
    LocalTensor<T2> vecClc1Buffer = ubBuffer.GetWithOffset<T2>(32 * 1024 / sizeof(T2), ubBufferOffset + T1Begin);
    if (s2VecLoop == 1) {
        DataCopy(vecClc1Buffer, mm1WorkspaceGm[curS1Idx * s1VecSize * s2ExtendAlign], s1ExtendSubGraph * s2ExtendAlign);
    } else {
        DataCopyPad(vecClc1Buffer, mm1WorkspaceGm[curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                    {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(float)),
                     static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(float)), 0},
                    {false, 0, 0, 0});
    }
    set_flag(PIPE_MTE2, PIPE_V, curEventId);
    wait_flag(PIPE_MTE2, PIPE_V, curEventId);

    ///////////////////////////////////////////////////////////////
    // sub to improve
    ///////////////////////////////////////////////////////////////
    uint32_t sub_block_cout = (s2ExtendAlign + cal_repeat_num - 1) / cal_repeat_num;
    uint32_t sfmgStartIndex = s1CvRatio > 1 ? 0 : curS1Idx * s1VecSize * 8;
    LocalTensor<float> sfmgClc3 = vecClc3.Get<float>();
    PipeBarrier<PIPE_V>();

    // @lg
    /*
        __aicore__ BinaryRepeatParams(const uint8_t dstBlkStrideIn, const uint8_t src0BlkStrideIn,
          const uint8_t src1BlkStrideIn, const uint8_t dstRepStrideIn, const uint8_t src0RepStrideIn,
          const uint8_t src1RepStrideIn)
      {
          dstBlkStride = dstBlkStrideIn;
          src0BlkStride = src0BlkStrideIn;
          src1BlkStride = src1BlkStrideIn;
          dstRepStride = dstRepStrideIn;
          src0RepStride = src0RepStrideIn;
          src1RepStride = src1RepStrideIn;
      }
      Repeat times（迭代的次数）
      Block stride（单次迭代内不同block间地址步长）
      Repeat stride（相邻迭代间相同block的地址步长）
      Mask（用于控制参与运算的计算单元）

      下方(s2ExtendAlign / 8)是指，s2ExtendAlign包含多少个32B
    */
    // @lg

    for (uint32_t subIdx = 0; subIdx < sub_block_cout; subIdx++) {
        uint32_t subMaskCout =
            (subIdx == sub_block_cout - 1) ? (s2ExtendAlign - subIdx * cal_repeat_num) : cal_repeat_num;
        Sub(vecClc1Buffer[subIdx * cal_repeat_num], vecClc1Buffer[subIdx * cal_repeat_num], sfmgClc3[sfmgStartIndex],
            subMaskCout, s1ExtendSubGraph,
            {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2ExtendAlign / 8),
             static_cast<uint8_t>(s2ExtendAlign / 8), 1});
    }

    ///////////////////////////////////////////////////////////////
    // mul
    ///////////////////////////////////////////////////////////////
    PipeBarrier<PIPE_V>();
    LocalTensor<float> vecClc2Buffer =
        ubBuffer.GetWithOffset<float>(32 * 1024 / sizeof(float), ubBufferOffset + T2Begin);
    // @lg 这里计算的是dS = P * (dP - D)
    Mul(vecClc1Buffer, vecClc1Buffer, vecClc2Buffer, s1ExtendSubGraph * s2ExtendAlign);

    LocalTensor<T1> vecOutBuffer = ubBuffer.GetWithOffset<T1>(16 * 1024 / sizeof(T1), ubBufferOffset + T1Begin);
    PipeBarrier<PIPE_V>();
    Cast(vecOutBuffer, vecClc1Buffer, RoundMode::CAST_ROUND, s1ExtendSubGraph * s2ExtendAlign);

    set_flag(PIPE_V, PIPE_MTE3, curEventId);
    wait_flag(PIPE_V, PIPE_MTE3, curEventId);

    DataCopyPad(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN +
                               curS1Idx * s1VecSize * s2CvExtendAlign + curS2Idx * s2VecSize],
                vecOutBuffer,
                {static_cast<uint16_t>(s1ExtendSubGraph), static_cast<uint16_t>(s2ExtendAlign * sizeof(T1)), 0,
                 static_cast<uint16_t>((s2CvExtendAlign - s2ExtendAlign) * sizeof(T1))});
    if ((s1VecLoop * s2VecLoop > 2) && (curIdx < (s1VecLoop * s2VecLoop - 2))) {
        set_flag(PIPE_MTE3, PIPE_MTE2, curEventId);
    }
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::Compute(uint32_t preIndex,
                                                                                                 uint32_t nextIndex)
{
    pingpongIdx = 1 - pingpongIdx;
    if (isFirstInitIndex == 1) {
        isFirstInitIndex = 0;
        InitIndex(preIndex);
    }
    bDimIdxTmp = bDimIdx;
    n2DimIdxTmp = n2DimIdx;
    gDimIdxTmp = gDimIdx;
    s1oDimIdxTmp = s1oDimIdx;
    s2oDimIdxTmp = s2oCvDimIdx;
    uint64_t mm1aTensorOffsetCv = 0;      // dOut
    uint64_t mm2aTensorOffsetCv = 0;      // Q
    uint64_t mm2aTensorOffsetCvDqTmp = 0; // Q
    uint64_t mm1bTensorOffsetCv = 0;      // V
    uint64_t mm1bTensorOffsetCvTmp = 0;   // V
    uint64_t mm2bTensorOffsetCv = 0;      // K
    uint64_t mm2bTensorOffsetCvTmp = 0;   // K
    s2CvExtend = preS2CvEnd - preS2CvBegin;
    s2CvExtendAlign = (s2CvExtend + 15) / 16 * 16;
    uint64_t mm1S1StrideSize = 0;
    uint64_t mm1S2StrideSize = 0;
    uint64_t mm2S1StrideSize = 0;
    uint64_t mm2S2StrideSize = 0;
    int64_t actualS1Tmp;
    int64_t actualS2Tmp;

    if (bDimIdxTmp > 0) {
        mm1aTensorOffsetCv = ((__gm__ int64_t*)actual_seq_qlen_addr)[bDimIdxTmp] * n2 * g * dV;
        mm1bTensorOffsetCv = ((__gm__ int64_t*)actual_seq_kvlen_addr)[bDimIdxTmp] * n2 * dV;
        mm1bTensorOffsetCvTmp = ((__gm__ int64_t*)actual_seq_kvlen_addr)[bDimIdxTmp] * n2 * gs1TileNum * dV;
        mm2aTensorOffsetCv = ((__gm__ int64_t*)actual_seq_qlen_addr)[bDimIdxTmp] * n2 * g * d;
        mm2aTensorOffsetCvDqTmp = ((__gm__ int64_t*)actual_seq_qlen_addr)[bDimIdxTmp] * n2 * g * s2TileNum * d;
        mm2bTensorOffsetCv = ((__gm__ int64_t*)actual_seq_kvlen_addr)[bDimIdxTmp] * n2 * d;
        mm2bTensorOffsetCvTmp = ((__gm__ int64_t*)actual_seq_kvlen_addr)[bDimIdxTmp] * n2 * gs1TileNum * d;
    }
    UpdateCalToken(bDimIdxTmp);
    mm1aTensorOffsetCv += ((s1oDimIdxTmp * s1CvInner * n2 + n2DimIdxTmp) * g + gDimIdxTmp) * dV;
    mm2aTensorOffsetCv += ((s1oDimIdxTmp * s1CvInner * n2 + n2DimIdxTmp) * g + gDimIdxTmp) * d;
    // s2oDimIdxTmp 对应s2outerIdx，切了多少分多核存多少份
    mm2aTensorOffsetCvDqTmp +=
        (((s1oDimIdxTmp * s1CvInner * n2 + n2DimIdxTmp) * g + gDimIdxTmp) * s2TileNum + s2oDimIdxTmp) * d;

    mm1bTensorOffsetCv += (preS2CvBegin * n2 + n2DimIdxTmp) * dV;
    mm1bTensorOffsetCvTmp += ((preS2CvBegin * n2 + n2DimIdxTmp) * gs1TileNum + gs1Offset) * dV;
    mm2bTensorOffsetCv += (preS2CvBegin * n2 + n2DimIdxTmp) * d;
    mm2bTensorOffsetCvTmp += ((preS2CvBegin * n2 + n2DimIdxTmp) * gs1TileNum + gs1Offset) * d;
    mm1S1StrideSize = n2 * g * dV;
    mm2S1StrideSize = n2 * g * d;
    mm1S2StrideSize = n2 * dV;
    mm2S2StrideSize = n2 * d;
    GetSeqQlenKvlenByBidx(bDimIdxTmp, actualS1Tmp, actualS2Tmp);

    // @lg: mm1计算dP = dO @ V_T
    if (isStart == 1) {
        // @lg M Ka x N Kb
        mm1.SetOrgShape(actualS1Tmp, s2, mm1S1StrideSize, mm1S2StrideSize, s2CvExtendAlign);
        mm1.SetTail(s1CvExtend, s2CvExtend, dV); // M N K
        mm1.SetTensorA(dxGm[mm1aTensorOffsetCv]);
        mm1.SetTensorB(valueGm[mm1bTensorOffsetCv], true);
        mm1.template IterateAll<false>(mm1WorkspaceGm, false, false, true);

        // @lg: mm1计算Q @ K_T
        mm1.SetOrgShape(actualS1Tmp, s2, mm2S1StrideSize, mm2S2StrideSize, s2CvExtendAlign);
        mm1.SetTail(s1CvExtend, s2CvExtend, d); // M N K
        mm1.SetTensorA(queryGm[mm2aTensorOffsetCv]);
        mm1.SetTensorB(keyGm[mm2bTensorOffsetCv], true);
        mm1.template IterateAll<false>(mm2WorkspaceGm, false, false, true);

        isStart = 0;
    }

    s2VecSize = s2CvExtend > VEC_S2_LEN ? VEC_S2_LEN : s2CvExtend;
    s2VecLoop = (s2CvExtend + s2VecSize - 1) / s2VecSize;

    if (TilingData->s1s2BNGS1S2BaseParams.attenMaskOptional) {
        // attenmask last dim 32B align
        s2VecSizeAlign = (s2VecSize + 31) / 32 * 32;
    } else if (TilingData->s1s2BNGS1S2BaseParams.prmMaskOptional) {
        s2VecSizeAlign = (s2VecSize + 31) / 32 * 32;
    } else {
        s2VecSizeAlign = (s2VecSize + 15) / 16 * 16;
    }

    s1VecSize = baseMN / s2VecSizeAlign;
    s1VecSize = s1VecSize < s1CvExtend ? s1VecSize : s1CvExtend;
    s1VecLoop = (s1CvExtend + s1VecSize - 1) / s1VecSize;
    ///////////////////////////////////////////////////////////////
    // SoftmaxGradFront
    ///////////////////////////////////////////////////////////////
    LocalTensor<float> sfmgClc3 = vecClc3.Get<float>();
    CalcSoftMaxGrad(sfmgClc3, mm1aTensorOffsetCv, s1CvExtend);

    mm1.WaitIterateAll();
    mm1.WaitIterateAll();
    // @lg subgraphA用于计算P、subgraphB用于计算dS = P * (dP - D), vector侧做了 ping pong，具体timeline：
    // ping subgraphA -> subgraphB
    // pong subgraphA -> subgraphB
    // 即，subgraph A、B在同一eventid下具有前后依赖关系，B需要用到A的P矩阵结果
    uint32_t curIdxPing = 0;
    uint32_t curIdxPong = 0;
    uint32_t curS2IdxPing = 0;
    uint32_t curS2IdxPong = 0;
    if (TilingData->s1s2BNGS1S2BaseParams.sinkRequiresGrad) {
        LocalTensor<T2> sinkTensor = sinkBuffer.GetWithOffset<float>(1 * 1024 / sizeof(float), 0);
        Duplicate(sinkTensor, float(0), 1 * 1024 / sizeof(float));
    }
    for (uint32_t curS1Idx = 0; curS1Idx < s1VecLoop; curS1Idx++) {
        s1ExtendSubGraph = (curS1Idx == s1VecLoop - 1) ? (s1CvExtend - (s1VecLoop - 1) * s1VecSize) : s1VecSize;
        uint32_t dBoffset = ((curS1Idx % 2) == 0) ? 0 : 8 * 1024;
        LocalTensor<float> softmaxTensor = softmaxBuffer.GetWithOffset<float>(8 * 1024 / sizeof(float), dBoffset);
        uint32_t softMaxOffset = 0;
        if (bDimIdxTmp > 0) {
            softMaxOffset = ((__gm__ int64_t*)actual_seq_qlen_addr)[bDimIdxTmp];
        } // BATCH OFFSET
        softMaxOffset += (n2DimIdxTmp * g + gDimIdxTmp) * ((__gm__ int64_t*)actual_seq_qlen_addr)[b]; // NG OFFSET
        softMaxOffset += s1oDimIdxTmp * s1CvInner + curS1Idx * s1VecSize;                             // S1 OFFSET

        CopyInSoftMax(softmaxTensor, s1ExtendSubGraph, softMaxOffset);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>());
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        // sink cal after softmax copyin
        if (TilingData->s1s2BNGS1S2BaseParams.sinkRequiresGrad) {
            uint32_t sfmgInnerOffset = curS1Idx * s1VecSize;
            CalSinkGrad(softmaxTensor, s1ExtendSubGraph, sfmgInnerOffset, curS1Idx);
            event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>());
            SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
            WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        }
        for (uint32_t curS2Idx = 0; curS2Idx < s2VecLoop; curS2Idx = curS2Idx + 2) {
            curS2IdxPing = curS2Idx;
            curS2IdxPong = curS2Idx + 1;
            curIdxPing = curS1Idx * s2VecLoop + curS2IdxPing;
            curIdxPong = curS1Idx * s2VecLoop + curS2IdxPong;
            SubGrapA(curIdxPing, curS1Idx, curS2IdxPing);
            if (curS2IdxPong < s2VecLoop) {
                SubGrapA(curIdxPong, curS1Idx, curS2IdxPong);
            }

            SubGrapB(curIdxPing, curS1Idx, curS2IdxPing);
            if (curS2IdxPong < s2VecLoop) {
                SubGrapB(curIdxPong, curS1Idx, curS2IdxPong);
            }
        }
    }

    uint32_t preS1Extend = s1CvExtend;
    if (nextIndex != 0) {
        InitIndex(nextIndex);
        uint64_t nextS2CvExtend = nextS2CvEnd - nextS2CvBegin;
        uint64_t nextS2CvExtendAlign = (nextS2CvExtend + 15) / 16 * 16;
        uint64_t mm1aTensorOffsetCv1 = 0;
        uint64_t mm2aTensorOffsetCv1 = 0;
        uint64_t mm1bTensorOffsetCv1 = 0;
        uint64_t mm2bTensorOffsetCv1 = 0;
        uint32_t bDimIdxTmp = bDimIdx;
        uint32_t n2DimIdxTmp = n2DimIdx;
        uint32_t gDimIdxTmp = gDimIdx;
        uint32_t s1oDimIdxTmp = s1oDimIdx;

        if (bDimIdxTmp > 0) {
            mm1aTensorOffsetCv1 = ((__gm__ int64_t*)actual_seq_qlen_addr)[bDimIdxTmp] * n2 * g * dV;
            mm1bTensorOffsetCv1 = ((__gm__ int64_t*)actual_seq_kvlen_addr)[bDimIdxTmp] * n2 * dV;
            mm2aTensorOffsetCv1 = ((__gm__ int64_t*)actual_seq_qlen_addr)[bDimIdxTmp] * n2 * g * d;
            mm2bTensorOffsetCv1 = ((__gm__ int64_t*)actual_seq_kvlen_addr)[bDimIdxTmp] * n2 * d;
        }
        mm1aTensorOffsetCv1 += ((s1oDimIdxTmp * s1CvInner * n2 + n2DimIdxTmp) * g + gDimIdxTmp) * dV;
        mm2aTensorOffsetCv1 += ((s1oDimIdxTmp * s1CvInner * n2 + n2DimIdxTmp) * g + gDimIdxTmp) * d;
        mm1bTensorOffsetCv1 += (nextS2CvBegin * n2 + n2DimIdxTmp) * dV;
        mm2bTensorOffsetCv1 += (nextS2CvBegin * n2 + n2DimIdxTmp) * d;

        mm1.SetOrgShape(actualS1Tmp, s2, mm1S1StrideSize, mm1S2StrideSize, nextS2CvExtendAlign);
        mm1.SetTail(s1CvExtend, nextS2CvExtend, dV);
        mm1.SetTensorA(dxGm[mm1aTensorOffsetCv1]);
        mm1.SetTensorB(valueGm[mm1bTensorOffsetCv1], true);
        mm1.template IterateAll<false>(mm1WorkspaceGm, false, false, true);
        mm1.SetOrgShape(actualS1Tmp, s2, mm2S1StrideSize, mm2S2StrideSize, nextS2CvExtendAlign);
        mm1.SetTail(s1CvExtend, nextS2CvExtend, d);
        mm1.SetTensorA(queryGm[mm2aTensorOffsetCv1]);
        mm1.SetTensorB(keyGm[mm2bTensorOffsetCv1], true);
        mm1.template IterateAll<false>(mm2WorkspaceGm, false, false, true);
    }

    // /////////////////////////////////////////////////////////////
    // Matmal4 dq
    // /////////////////////////////////////////////////////////////
    mm4.SetOrgShape(actualS1Tmp, n2 * d, s2CvExtendAlign, actualS2Tmp, n2 * g * s2TileNum * d);
    mm4.SetTail(preS1Extend, -1, s2CvExtend); // M N K
    mm4.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN]);
    mm4.SetTensorB(keyGm[mm2bTensorOffsetCv]);
    mm4.template IterateAll<false>(dqWorkSpaceGm[mm2aTensorOffsetCvDqTmp], false);
    mm4.End();

    ///////////////////////////////////////////////////////////////
    // Matmal4 dk
    ///////////////////////////////////////////////////////////////
    // lfet: [N2, G, S2, S1]  right: [N2, G, S1, D]
    mm3.SetOrgShape(s2CvExtendAlign, n2 * g * d, actualS1Tmp, actualS1Tmp, n2 * gs1TileNum * d);
    mm3.SetTail(s2CvExtend, d, preS1Extend);
    mm3.SetTensorA(mulWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN], true);
    mm3.SetTensorB(queryGm[mm2aTensorOffsetCv]);
    mm3.template IterateAll<false>(dkWorkSpaceGm[mm2bTensorOffsetCvTmp], true);

    mm3.End();

    ///////////////////////////////////////////////////////////////
    // Matmal5 dv
    ///////////////////////////////////////////////////////////////
    mm3.SetOrgShape(s2CvExtendAlign, n2 * g * dV, actualS1Tmp, actualS1Tmp, n2 * gs1TileNum * dV);
    mm3.SetTail(s2CvExtend, dV, preS1Extend);
    mm3.SetTensorA(tmpWorkSpaceGm[pingpongIdx * coreNum * cubeBaseMN + cBlockIdx * cubeBaseMN], true);
    mm3.SetTensorB(dxGm[mm1aTensorOffsetCv]);
    if (nextIndex == 0) {
        mm3.template IterateAll<true>(dvWorkSpaceGm[mm1bTensorOffsetCvTmp], true);
    } else {
        mm3.template IterateAll<false>(dvWorkSpaceGm[mm1bTensorOffsetCvTmp], true);
    }
    mm3.End();
}

template <typename T1, typename T2, const bool IS_TND_INPUTS, const CubeFormat MM_OUT_FORMAT>
__aicore__ inline void FlashAttentionGradWsReduce<T1, T2, IS_TND_INPUTS, MM_OUT_FORMAT>::SyncALLCores()
{
    SyncAll();
}

#endif // _FLASH_ATTENTION_GRAD_WS_REDUCE_H_
