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
 * \file dsa_dense.h
 * \brief
 */

#ifndef _DSA_DENSE_H_
#define _DSA_DENSE_H_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace ops_dsa_dense {

template <typename InType, typename CalcType>
class DsaDense {
public:
    __aicore__ inline DsaDense() {}

    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* max, __gm__ uint8_t* sum,
                                __gm__ uint8_t* qLen, __gm__ uint8_t* kvLen, __gm__ uint8_t* mask,
                                __gm__ uint8_t* queryIndex, __gm__ uint8_t* keyIndex, __gm__ uint8_t* weights,
                                __gm__ uint8_t* maxIndex, __gm__ uint8_t* sumIndex, __gm__ uint8_t* loss,
                                __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dw, __gm__ uint8_t* workspace,
                                const DsaDenseTilingData* __restrict tiling, TPipe* tPipe);

    __aicore__ inline void Process();

public:
    // mamtul obj
    using aType1 = MatmulType<TPosition::GM, CubeFormat::ND, InType, false>;
    using bType1 = MatmulType<TPosition::GM, CubeFormat::ND, InType, true>;
    using aType2 = MatmulType<TPosition::GM, CubeFormat::ND, InType, true>;
    using bType2 = MatmulType<TPosition::GM, CubeFormat::ND, InType, false>;
    using aType3 = MatmulType<TPosition::GM, CubeFormat::ND, InType, false>;
    using bType3 = MatmulType<TPosition::GM, CubeFormat::ND, InType, false>;
    using cType = MatmulType<TPosition::GM, CubeFormat::ND, CalcType>;
    using biasType = MatmulType<TPosition::GM, CubeFormat::ND, CalcType>;
    Matmul<aType1, bType1, cType, biasType> mm1;
    Matmul<aType2, bType2, cType, biasType> mm2;
    Matmul<aType3, bType3, cType, biasType> mm3;

private:
    __aicore__ inline uint64_t QLenForBatch(uint32_t batchIdx)
    {
        return ((__gm__ uint64_t*)qLenPtr_)[batchIdx + 1] - ((__gm__ uint64_t*)qLenPtr_)[batchIdx];
    }

    __aicore__ inline uint64_t QLenBeforeBatch(uint32_t batchIdx)
    {
        return ((__gm__ uint64_t*)qLenPtr_)[batchIdx];
    }

    __aicore__ inline uint64_t KVLenForBatch(uint32_t batchIdx)
    {
        return ((__gm__ uint64_t*)kvLenPtr_)[batchIdx + 1] - ((__gm__ uint64_t*)kvLenPtr_)[batchIdx];
    }

    __aicore__ inline uint64_t KVLenBeforeBatch(uint32_t batchIdx)
    {
        return ((__gm__ uint64_t*)kvLenPtr_)[batchIdx];
    }

    __aicore__ inline GlobalTensor<CalcType>& QKSpace(uint32_t head)
    {
        if (head % 2 == 0) {
            return this->qkSpacePing_;
        } else {
            return this->qkSpacePong_;
        }
    }

    __aicore__ inline GlobalTensor<InType>& SGradSpace(uint32_t head)
    {
        if (head % 2 == 0) {
            return this->dsSpacePing_;
        } else {
            return this->dsSpacePong_;
        }
    }

    __aicore__ inline void LoadBalance();

    __aicore__ inline void ZeroGradQKW();

    __aicore__ inline void ZeroGradW();

    __aicore__ inline void PostGradQ();

    __aicore__ inline void PostGradK();

    __aicore__ inline void PostLoss();

    __aicore__ inline void CalcGraphP();

    __aicore__ inline void Softmax(LocalTensor<CalcType>& src, LocalTensor<CalcType>& common,
                                   const LocalTensor<CalcType>& max, const LocalTensor<CalcType>& sum, uint64_t s1Len,
                                   uint64_t s1Offset, uint64_t s2Len, uint64_t s2LenAlign8);

    __aicore__ inline void PreSoftmax(LocalTensor<CalcType>& src, LocalTensor<uint8_t>& maskBuffer,
                                      LocalTensor<uint8_t>& sharedTmpBuffer, uint64_t s1Len, uint64_t s1Offset);

    __aicore__ inline void CalcGraphY();

    __aicore__ inline void CalcGraphGrad();

    __aicore__ inline void LaunchMM23(uint32_t head);

    __aicore__ inline void CalcGraphLoss(uint32_t head, LocalTensor<CalcType>& p, LocalTensor<CalcType>& y,
                                         LocalTensor<CalcType>& loss, uint32_t calcCount);

    // loss多核累加、dk多核累加、dq cast、dw cast
    __aicore__ inline void PostProcess();

    __aicore__ inline void CalcS1TileInfo(const uint32_t s1TileIdx);

    __aicore__ inline void CalcS2TileInfo(const uint32_t s2TileIdx);

    template <typename T1, typename T2>
    __aicore__ inline T2 CeilDiv(const T1 a, const T2 b)
    {
        return static_cast<T2>((a + static_cast<T1>(b) - 1) / static_cast<T1>(b));
    }

private:
    // ascend info
    TPipe* pipe_;
    const DsaDenseTilingData* __restrict tiling_;
    uint32_t blockIdx_;
    uint32_t aivNum_;

    // tile Info
    uint32_t bIdx_ = 0;      // 当前tile属于该句子
    uint32_t cuS1Tiles_ = 0; // 该句子之前句子的s1 Tile总数
    uint64_t cuS1_ = 0;      // 该句子之前句子的s1总长
    uint64_t cuS2_ = 0;      // 该句子之前句子的s2总长
    uint32_t nIdx_;
    uint32_t s1Idx_; // 当前tile在该句子内的s1坐标
    uint32_t s2Idx_; // 当前tile在该句子内的s2坐标
    uint64_t s1_;    // 当前tile所属句子s1长度
    uint64_t s2_;
    uint64_t historyLen_;
    uint64_t causalLen_;
    uint32_t s2TileLimit_;
    uint32_t s1TileReal_; // 当前tile在s1方向真实token数量
    uint32_t s2TileReal_;
    uint32_t s2TileRealAlign8_;
    uint32_t s2TileRealAlign16_;
    uint32_t s2TileRealAlign32_;
    uint32_t tileMode_;    // 0: normal tile; 1: causal tile; 2: skip tile
    uint64_t offsetForT1_; // 当前tile在T1维度offset
    uint64_t offsetForT2_; // 当前tile在T2维度offset

    //// load balance ////
    static constexpr uint32_t MAX_CORE = 50;
    uint32_t tileStarts_[MAX_CORE];

    //// shape params ////
    uint32_t B_;
    uint32_t N_;
    uint32_t NI_;
    uint32_t D_;
    uint32_t DI_;
    uint64_t T1_;
    uint64_t T2_;
    CalcType scale_;
    CalcType l1NormScale_;
    uint32_t cubeS1Tile_;
    uint32_t cubeS2Tile_;
    uint32_t vecS1Tile_;
    uint32_t vecS2Tile_;

    //// ub buffer ////
    TBuf<> ubBuf_;
    LocalTensor<CalcType> lossBuffer_;

    //// global tensors ////
    // inputs
    GlobalTensor<InType> query_;
    GlobalTensor<InType> key_;
    GlobalTensor<CalcType> max_;
    GlobalTensor<CalcType> sum_;
    GlobalTensor<uint64_t> qLen_;
    GlobalTensor<uint64_t> kvLen_;
    GM_ADDR qLenPtr_;
    GM_ADDR kvLenPtr_;
    GlobalTensor<uint8_t> mask_;
    GlobalTensor<InType> queryIndex_;
    GlobalTensor<InType> keyIndex_;
    GlobalTensor<CalcType> weights_;
    GlobalTensor<CalcType> maxIndex_;
    GlobalTensor<CalcType> sumIndex_;
    // outputs
    GlobalTensor<CalcType> loss_;
    GlobalTensor<InType> dq_;
    GlobalTensor<InType> dk_;
    GlobalTensor<CalcType> dw_;
    // workspace
    GlobalTensor<CalcType> qkSpacePing_;
    GlobalTensor<CalcType> qkSpacePong_;
    GlobalTensor<InType> dsSpacePing_;
    GlobalTensor<InType> dsSpacePong_;
    GlobalTensor<CalcType> pBaseSpace_;
    GlobalTensor<CalcType> pSpace_;
    GlobalTensor<CalcType> ySpace_;
    GlobalTensor<CalcType> reluSpace_;
    GlobalTensor<CalcType> dwSpace_;
    GlobalTensor<CalcType> dkBaseSpace_;
    GlobalTensor<CalcType> dkSpace_;
    GlobalTensor<CalcType> dqSpace_;

    //// tools ////
    const uint32_t NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF;
    const float negativeScalar_ = *((float*)&NEGATIVE_MIN_VALUE_FP32);
    const uint32_t db_ = 2;
    const uint32_t maxUbBytes_ = 184 * 1024;
    const uint32_t lossBufferPos_ = 183 * 1024;
    const uint32_t gmAlignment_ = 64;
    const float delta_ = 1e-8; // log(0) = -inf，防止0出现，保证loss数值安全
    // tile mode
    const uint32_t NORMAL_TILE = 0;
    const uint32_t CAUSAL_TILE = 1;

    //// sync flag ////
    event_t eventIdVToMte3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
    event_t eventIdMte3ToV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_V>());
    event_t eventIdMte2ToV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
};

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* max, __gm__ uint8_t* sum, __gm__ uint8_t* qLen,
    __gm__ uint8_t* kvLen, __gm__ uint8_t* mask, __gm__ uint8_t* queryIndex, __gm__ uint8_t* keyIndex,
    __gm__ uint8_t* weights, __gm__ uint8_t* maxIndex, __gm__ uint8_t* sumIndex, __gm__ uint8_t* loss,
    __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dw, __gm__ uint8_t* workspace,
    const DsaDenseTilingData* __restrict tiling, TPipe* tPipe)
{
    // 1. ascend info
    this->blockIdx_ = GetBlockIdx();
    this->pipe_ = tPipe;
    this->tiling_ = tiling;
    this->aivNum_ = tiling_->baseParams.aivNum;

    // 2. common shape
    this->B_ = tiling_->baseParams.B;
    this->N_ = tiling_->baseParams.N;
    this->NI_ = tiling_->baseParams.NI;
    this->D_ = tiling_->baseParams.D;
    this->DI_ = tiling_->baseParams.DI;
    this->T1_ = tiling_->baseParams.T1;
    this->T2_ = tiling_->baseParams.T2;
    this->scale_ = tiling_->baseParams.scale;
    this->l1NormScale_ = 1.0f / this->N_;
    this->cubeS1Tile_ = tiling_->baseParams.cubeS1Tile;
    this->cubeS2Tile_ = tiling_->baseParams.cubeS2Tile;
    this->vecS1Tile_ = tiling_->baseParams.vecS1Tile; // TODO: 暂时未用到，每个graph内自己设置了计算粒度
    this->vecS2Tile_ = tiling_->baseParams.vecS2Tile;

    // 3. global tensors
    this->query_.SetGlobalBuffer((__gm__ InType*)query);
    this->key_.SetGlobalBuffer((__gm__ InType*)key);
    this->max_.SetGlobalBuffer((__gm__ CalcType*)max);
    this->sum_.SetGlobalBuffer((__gm__ CalcType*)sum);
    this->qLen_.SetGlobalBuffer((__gm__ uint64_t*)qLen);
    this->kvLen_.SetGlobalBuffer((__gm__ uint64_t*)kvLen);
    qLenPtr_ = qLen;
    kvLenPtr_ = kvLen;
    this->mask_.SetGlobalBuffer((__gm__ uint8_t*)mask);
    this->queryIndex_.SetGlobalBuffer((__gm__ InType*)queryIndex);
    this->keyIndex_.SetGlobalBuffer((__gm__ InType*)keyIndex);
    this->weights_.SetGlobalBuffer((__gm__ CalcType*)weights);
    this->maxIndex_.SetGlobalBuffer((__gm__ CalcType*)maxIndex);
    this->sumIndex_.SetGlobalBuffer((__gm__ CalcType*)sumIndex);

    this->loss_.SetGlobalBuffer((__gm__ CalcType*)loss);
    this->dq_.SetGlobalBuffer((__gm__ InType*)dq);
    this->dk_.SetGlobalBuffer((__gm__ InType*)dk);
    this->dw_.SetGlobalBuffer((__gm__ CalcType*)dw);

    uint64_t workspaceOffset = 0;
    this->qkSpacePing_.SetGlobalBuffer(
        (__gm__ CalcType*)(workspace + this->blockIdx_ * this->tiling_->baseParams.qkSpaceBytes));
    this->dsSpacePing_.SetGlobalBuffer(
        (__gm__ InType*)(workspace + this->blockIdx_ * this->tiling_->baseParams.qkSpaceBytes));
    this->qkSpacePong_.SetGlobalBuffer((__gm__ CalcType*)(workspace +
                                                          this->blockIdx_ * this->tiling_->baseParams.qkSpaceBytes +
                                                          (this->tiling_->baseParams.qkSpaceBytes / this->db_)));
    this->dsSpacePong_.SetGlobalBuffer((__gm__ InType*)(workspace +
                                                        this->blockIdx_ * this->tiling_->baseParams.qkSpaceBytes +
                                                        (this->tiling_->baseParams.qkSpaceBytes / this->db_)));
    workspaceOffset += this->aivNum_ * this->tiling_->baseParams.qkSpaceBytes;
    // PRINTF("LG TEST aivNum_: %d \n", this->aivNum_);
    this->pBaseSpace_.SetGlobalBuffer((__gm__ CalcType*)(workspace + workspaceOffset));
    this->pSpace_.SetGlobalBuffer(
        (__gm__ CalcType*)(workspace + workspaceOffset + this->blockIdx_ * this->tiling_->baseParams.pSpaceBytes));
    workspaceOffset += this->aivNum_ * this->tiling_->baseParams.pSpaceBytes;

    this->ySpace_.SetGlobalBuffer(
        (__gm__ CalcType*)(workspace + workspaceOffset + this->blockIdx_ * this->tiling_->baseParams.ySpaceBytes));
    workspaceOffset += this->aivNum_ * this->tiling_->baseParams.ySpaceBytes;

    this->reluSpace_.SetGlobalBuffer(
        (__gm__ CalcType*)(workspace + workspaceOffset + this->blockIdx_ * this->tiling_->baseParams.reluSpaceBytes));
    workspaceOffset += this->aivNum_ * this->tiling_->baseParams.reluSpaceBytes;

    this->dwSpace_.SetGlobalBuffer((__gm__ CalcType*)(workspace + workspaceOffset));
    workspaceOffset += this->tiling_->baseParams.dwSpaceBytes;

    this->dkBaseSpace_.SetGlobalBuffer((__gm__ CalcType*)(workspace + workspaceOffset));
    this->dkSpace_.SetGlobalBuffer(
        (__gm__ CalcType*)(workspace + workspaceOffset + this->blockIdx_ * this->tiling_->baseParams.dkSpaceBytes));
    workspaceOffset += this->aivNum_ * this->tiling_->baseParams.dkSpaceBytes;

    this->dqSpace_.SetGlobalBuffer((__gm__ CalcType*)(workspace + workspaceOffset));

    // 4. local tensors
    this->pipe_->InitBuffer(this->ubBuf_, this->maxUbBytes_); // 184kb
    this->lossBuffer_ = this->ubBuf_.template GetWithOffset<CalcType>(8, this->lossBufferPos_);
    Duplicate(this->lossBuffer_, 0.0f, 8);
    PipeBarrier<PIPE_V>();

    // 5. zero grad q k buffer
    this->ZeroGradQKW();

    // 6. load balance
    this->LoadBalance();
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::Process()
{
    uint32_t s1StartIdx = this->tileStarts_[this->blockIdx_];
    uint32_t s1EndIdx = this->tileStarts_[this->blockIdx_ + 1];
    // PRINTF("CORE, s1StartIdx, s1EndIdx: %d %d %d \n", this->blockIdx_, s1StartIdx, s1EndIdx);

    // 没有分到tile的core，不会执行下述loop，不需要这显式return
    // if (s1EndIdx <= s1StartIdx) {
    //     return;
    // }

    // tile(s1) level vecLoop
    for (uint32_t s1TileIdx = s1StartIdx; s1TileIdx < s1EndIdx; ++s1TileIdx) {
        this->CalcS1TileInfo(s1TileIdx);

        // s2 level vecLoop
        for (uint32_t s2TileIdx = 0; s2TileIdx < this->s2TileLimit_; ++s2TileIdx) {
            this->CalcS2TileInfo(s2TileIdx);

            this->CalcGraphP();

            this->CalcGraphY();

            this->CalcGraphGrad();
        }
    }

    // 所有core都会走到这里
    // 确保所有core都完成之前工作
    SyncAll<true>();
    this->PostProcess();
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::PostProcess()
{
    // 确保前序搬运、计算都完成
    PipeBarrier<PIPE_ALL>();

    // 搬出loss
    DataCopyPad(
        this->pBaseSpace_[this->blockIdx_], this->lossBuffer_,
        {static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(CalcType)), /*srcStride*/ 0, /*dstStride*/ 0, 0});
    SyncAll<true>();

    // dq cast & 搬出
    this->PostGradQ();

    // // dk reduce sum & cast & 搬出
    this->PostGradK();

    // loss reduce & 搬出
    this->PostLoss();
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::PostLoss()
{
    if (this->blockIdx_ == 0) {
        const uint32_t bufferElements = this->aivNum_;
        const uint32_t bufferBytes = bufferElements * sizeof(CalcType);
        LocalTensor<CalcType> lossBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, 0);
        LocalTensor<CalcType> lossSumBuffer =
            this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, bufferBytes);
        LocalTensor<CalcType> workBuffer =
            this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, bufferBytes * 2);
        DataCopyPad(lossBuffer, this->pBaseSpace_,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(this->aivNum_ * sizeof(CalcType)), /*srcStride*/ 0,
                     /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});

        SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);

        ReduceSum<CalcType>(lossSumBuffer, lossBuffer, /*sharedTmpBuffer，占位符，不会真正用到*/ workBuffer,
                            this->aivNum_);
        PipeBarrier<PIPE_V>();

        // 搬出
        SetFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
        DataCopyPad(
            this->loss_, lossSumBuffer,
            {static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(CalcType)), /*srcStride*/ 0, /*dstStride*/ 0, 0});

        PipeBarrier<PIPE_ALL>();
    }
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::PostGradK()
{
    // T2分多核
    const uint64_t avgT2 = CeilDiv(this->T2_, this->aivNum_);
    const uint64_t tailT2 = avgT2 - 1;
    const uint64_t tailCoreNum = avgT2 * this->aivNum_ - this->T2_;
    const uint64_t formerCoreNum = this->aivNum_ - tailCoreNum;

    uint64_t curT2 = avgT2;
    if (this->blockIdx_ >= formerCoreNum) {
        curT2 = tailT2;
    }
    uint64_t curT2Offset = this->blockIdx_ * avgT2;
    if (this->blockIdx_ > formerCoreNum) {
        curT2Offset = formerCoreNum * avgT2 + (this->blockIdx_ - formerCoreNum) * tailT2;
    }

    if (curT2 <= 0) {
        return;
    }

    // realAivNum个dk reduce到一起
    uint32_t coreId = 0;
    for (; coreId < this->aivNum_; ++coreId) {
        if (this->tileStarts_[coreId + 1] == 0) { // 未分到tile的core，end=0
            break;
        }
    }
    const uint32_t realAivNum = coreId;
    // PRINTF("CORE, realAivNum curT2, curT2Offset: %d %d %d %d \n", this->blockIdx_, realAivNum, curT2, curT2Offset);

    // ub划分
    const uint64_t dkElements = curT2 * this->DI_;
    const uint64_t bufferElements = this->maxUbBytes_ / sizeof(CalcType) / 2;
    const uint64_t bufferBytes = bufferElements * sizeof(CalcType);
    LocalTensor<CalcType> dkBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, 0);
    LocalTensor<CalcType> dkSumBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, bufferBytes);
    LocalTensor<InType> dkSumBufferRaw = this->ubBuf_.template GetWithOffset<InType>(bufferElements, bufferBytes);

    // dk元素loop
    const uint32_t dkLoopLimit = CeilDiv(dkElements, bufferElements);
    for (uint32_t dkLoop = 0; dkLoop < dkLoopLimit; ++dkLoop) {
        const uint64_t realElements =
            (dkLoop == dkLoopLimit - 1 ? dkElements - dkLoop * bufferElements : bufferElements);
        const uint64_t dkOffset = dkLoop * bufferElements;

        Duplicate(dkSumBuffer, 0.0f, realElements);
        PipeBarrier<PIPE_V>();

        // core方向loop
        for (uint32_t block = 0; block < realAivNum; ++block) {
            // load k from workspace
            uint64_t dkOffsetGlobal = block * (this->tiling_->baseParams.dkSpaceBytes / sizeof(CalcType)) +
                                      curT2Offset * this->DI_ + dkOffset;
            DataCopyPad(dkBuffer, this->dkBaseSpace_[dkOffsetGlobal],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(realElements * sizeof(CalcType)),
                         /*srcStride*/ 0, /*dstStride*/ 0, 0},
                        {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);

            Add(dkSumBuffer, dkSumBuffer, dkBuffer, realElements);
            PipeBarrier<PIPE_ALL>();
        }

        Cast(dkSumBufferRaw, dkSumBuffer, RoundMode::CAST_RINT, realElements);
        PipeBarrier<PIPE_V>();

        // 搬出
        SetFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
        DataCopyPad(this->dk_[curT2Offset * this->DI_ + dkOffset], dkSumBufferRaw,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(realElements * sizeof(InType)), /*srcStride*/ 0,
                     /*dstStride*/ 0, 0});

        PipeBarrier<PIPE_ALL>();
    }
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::PostGradQ()
{
    // T1分多核
    const uint64_t avgT1 = CeilDiv(this->T1_, this->aivNum_);
    const uint64_t tailT1 = avgT1 - 1;
    const uint64_t tailCoreNum = avgT1 * this->aivNum_ - this->T1_;
    const uint64_t formerCoreNum = this->aivNum_ - tailCoreNum;

    uint64_t curT1 = avgT1;
    if (this->blockIdx_ >= formerCoreNum) {
        curT1 = tailT1;
    }
    uint64_t curT1Offset = this->blockIdx_ * avgT1;
    if (this->blockIdx_ > formerCoreNum) {
        curT1Offset = formerCoreNum * avgT1 + (this->blockIdx_ - formerCoreNum) * tailT1;
    }

    if (curT1 <= 0) {
        return;
    }

    const uint64_t dqElements = curT1 * this->NI_ * this->DI_;
    const uint64_t bufferElements = this->maxUbBytes_ / sizeof(CalcType);
    const uint64_t bufferBytes = bufferElements * sizeof(CalcType);
    LocalTensor<CalcType> dqBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, 0);
    LocalTensor<InType> dqBufferRaw = this->ubBuf_.template GetWithOffset<InType>(bufferElements, 0);

    const uint32_t dqLoopLimit = CeilDiv(dqElements, bufferElements);
    for (uint32_t dqLoop = 0; dqLoop < dqLoopLimit; ++dqLoop) {
        const uint64_t realElements =
            (dqLoop == dqLoopLimit - 1 ? dqElements - dqLoop * bufferElements : bufferElements);
        const uint64_t dqOffset = dqLoop * bufferElements;
        // load dq
        DataCopyPad(dqBuffer, this->dqSpace_[curT1Offset * this->NI_ * this->DI_ + dqOffset],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(realElements * sizeof(CalcType)), /*srcStride*/ 0,
                     /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});
        // cast
        SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
        Cast(dqBufferRaw, dqBuffer, RoundMode::CAST_RINT, realElements);
        PipeBarrier<PIPE_V>();
        // 搬出
        SetFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
        DataCopyPad(this->dq_[curT1Offset * this->NI_ * this->DI_ + dqOffset], dqBufferRaw,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(realElements * sizeof(InType)), /*srcStride*/ 0,
                     /*dstStride*/ 0, 0});
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::ZeroGradW()
{
    // dw全部元素分多核
    const uint64_t totalElements = this->T1_ * this->NI_;
    const uint64_t totalGmParts = (totalElements + this->gmAlignment_ - 1) / this->gmAlignment_; // align up

    const uint64_t gmPartsPerCore = (totalGmParts + this->aivNum_ - 1) / this->aivNum_;
    const uint64_t gmPartsPerCoreTail = gmPartsPerCore - 1;
    const uint64_t tailCoreNum = gmPartsPerCore * this->aivNum_ - totalGmParts;
    const uint64_t formerCoreNum = this->aivNum_ - tailCoreNum;

    uint64_t gmPartsCurCore = gmPartsPerCore;
    if (blockIdx_ >= formerCoreNum) {
        gmPartsCurCore = gmPartsPerCoreTail;
    }
    uint64_t gmPartsOffsetCurCore = blockIdx_ * gmPartsPerCore;
    if (blockIdx_ > formerCoreNum) {
        gmPartsOffsetCurCore = formerCoreNum * gmPartsPerCore + (blockIdx_ - formerCoreNum) * gmPartsPerCoreTail;
    }

    int64_t elementsCurCore = gmPartsCurCore * this->gmAlignment_;
    uint64_t elementsOffsetCurCore = gmPartsOffsetCurCore * this->gmAlignment_;
    if (elementsOffsetCurCore + elementsCurCore > totalElements) { // 当前core处理的elements, 不应该超过总elements个数
        elementsCurCore = totalElements - elementsOffsetCurCore;
    }
    if (elementsCurCore > 0) {
        GlobalTensor<CalcType> curDwSpace = this->dw_[elementsOffsetCurCore];
        InitGlobalMemory<CalcType>(curDwSpace, static_cast<uint64_t>(elementsCurCore), 0.0f);
    }
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::LoadBalance()
{
    for (size_t i = 0; i < MAX_CORE; ++i) {
        this->tileStarts_[i] = 0;
    }

    // 计算总tiles
    uint32_t totalCausalTiles = 0;
    for (auto batch = 0; batch < this->B_; ++batch) {
        const uint64_t s1 = this->QLenForBatch(batch);
        const uint64_t s2 = this->KVLenForBatch(batch);
        const uint64_t historyKVLen = s2 - s1 > 0 ? s2 - s1 : 0;
        const uint32_t s1TileCount = CeilDiv(s1, this->cubeS1Tile_);
        for (auto s1TileIdx = 0; s1TileIdx < s1TileCount; ++s1TileIdx) {
            const uint32_t causalTileNums =
                CeilDiv(historyKVLen + static_cast<uint64_t>(s1TileIdx * this->cubeS1Tile_), this->cubeS2Tile_) +
                1 /*diagnoal tile*/;
            totalCausalTiles += causalTileNums;
        }
    }

    // 计算平均每个core分到的tiles数量阈值
    const uint32_t avgTiles = CeilDiv(totalCausalTiles, this->aivNum_);

    // 贪心算法，拿够即停
    uint32_t tilesPerCore = 0;
    uint32_t cuS1Tiles = 0;
    int coreIdx = 0;
    for (auto batch = 0; batch < this->B_; ++batch) {
        const uint64_t s1 = this->QLenForBatch(batch);
        const uint64_t s2 = this->KVLenForBatch(batch);
        const uint64_t historyKVLen = s2 - s1 > 0 ? s2 - s1 : 0;
        const uint32_t s1TileCount = CeilDiv(s1, this->cubeS1Tile_);
        for (auto s1TileIdx = 0; s1TileIdx < s1TileCount; ++s1TileIdx) {
            const uint32_t causalTileNums =
                CeilDiv(historyKVLen + static_cast<uint64_t>(s1TileIdx * this->cubeS1Tile_), this->cubeS2Tile_) +
                1 /*diagnoal tile*/;
            tilesPerCore += causalTileNums;
            if (tilesPerCore >= avgTiles) {
                this->tileStarts_[coreIdx + 1] = cuS1Tiles + s1TileIdx + 1; // [start, end)
                tilesPerCore = 0;
                coreIdx++;
            }
        }
        cuS1Tiles += s1TileCount;
    }

    // 贪心结束后，处理尾块(尾块tiles数量没超过avg，进而被错误丢弃)
    if (this->tileStarts_[coreIdx] < cuS1Tiles) {
        this->tileStarts_[coreIdx + 1] = cuS1Tiles;
    }
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::ZeroGradQKW()
{
    // zero dk
    InitGlobalMemory<CalcType>(this->dkSpace_,
                               static_cast<uint64_t>(this->tiling_->baseParams.dkSpaceBytes / sizeof(CalcType)), 0.0f);
    // PIPE_ALL是必须的：https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/ascendcopapi/atlasascendc_api_07_0891.html
    // 该api本质是执行分配ub tensor -> duplicate ub -> mte3行为
    // 会刷新ub
    PipeBarrier<PIPE_ALL>();

    // zero dq: [T1 NI DI]
    const uint64_t avgT1 = CeilDiv(this->T1_, this->aivNum_);
    const uint64_t tailT1 = avgT1 - 1;
    const uint64_t tailCoreNum = avgT1 * this->aivNum_ - this->T1_;
    const uint64_t formerCoreNum = this->aivNum_ - tailCoreNum;

    uint64_t curT1 = avgT1;
    if (this->blockIdx_ >= formerCoreNum) {
        curT1 = tailT1;
    }
    uint64_t curT1Offset = this->blockIdx_ * avgT1;
    if (this->blockIdx_ > formerCoreNum) {
        curT1Offset = formerCoreNum * avgT1 + (this->blockIdx_ - formerCoreNum) * tailT1;
    }

    if (curT1 > 0) {
        GlobalTensor<CalcType> curDqSpace = this->dqSpace_[curT1Offset * this->NI_ * this->DI_];
        InitGlobalMemory<CalcType>(curDqSpace, static_cast<uint64_t>(curT1 * this->NI_ * this->DI_), 0.0f);
    }
    PipeBarrier<PIPE_ALL>();

    // zero dw: [T1 NI]
    this->ZeroGradW();

    SyncAll<true>();
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::CalcGraphGrad()
{
    // ub划分
    const uint32_t vecS1Tile = 8; // TODO: 暂且按8
    const uint32_t bufferElements = vecS1Tile * this->cubeS2Tile_;
    const uint32_t bufferBytes = bufferElements * sizeof(CalcType);
    const uint32_t weightElements = this->cubeS1Tile_ * 8;
    const uint32_t weightBytes = weightElements * sizeof(CalcType); // 固定4KB
    const uint32_t dwElements = this->cubeS1Tile_;
    const uint32_t dwBytes = dwElements * sizeof(CalcType);
    LocalTensor<CalcType> pBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, 0);
    LocalTensor<CalcType> yBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, bufferBytes);
    LocalTensor<CalcType> diBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, bufferBytes * 2);
    LocalTensor<uint8_t> sharedTmpBuffer = diBuffer.template ReinterpretCast<uint8_t>();
    LocalTensor<CalcType> reluBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, bufferBytes * 3);
    LocalTensor<CalcType> dsBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, bufferBytes * 4);
    LocalTensor<InType> dsBufferRaw = this->ubBuf_.template GetWithOffset<InType>(bufferElements, bufferBytes * 4);
    LocalTensor<CalcType> weightBuffer =
        this->ubBuf_.template GetWithOffset<CalcType>(weightElements, bufferBytes * 5); // 8KB
    LocalTensor<CalcType> dwBuffer =
        this->ubBuf_.template GetWithOffset<CalcType>(dwElements, bufferBytes * 5 + weightBytes); // 0.5KB
    LocalTensor<uint8_t> reluMaskBuffer = this->ubBuf_.template GetWithOffset<uint8_t>(
        bufferElements / 8, bufferBytes * 5 + weightBytes + dwBytes); // 1KB
    // TODO: 重复load Y P
    for (uint32_t head = 0; head < this->NI_; ++head) {
        // load weights: from [T1 NI] layout -> [vecS1Real 8] TODO: 很低效
        DataCopyPad(weightBuffer, this->weights_[this->offsetForT1_ * this->NI_ + head],
                    {static_cast<uint16_t>(this->s1TileReal_),
                     static_cast<uint32_t>(
                         sizeof(CalcType)), // leftPadding、rightPadding都为0：dummy会默认填充待搬运数据块的第一个元素值
                     /*srcStride*/ static_cast<uint32_t>((this->NI_ - 1) * sizeof(CalcType)),
                     /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
        PipeBarrier<PIPE_V>();

        const uint32_t vecLoopLimit = CeilDiv(this->s1TileReal_, vecS1Tile);
        GlobalTensor<InType>& dsSpace = this->SGradSpace(head);
        for (uint32_t vecLoop = 0; vecLoop < vecLoopLimit; ++vecLoop) {
            const uint32_t vecS1Real =
                (vecLoop == vecLoopLimit - 1 ? this->s1TileReal_ - vecLoop * vecS1Tile : vecS1Tile);
            const uint32_t vecS1Offset = vecLoop * vecS1Tile;
            const uint32_t calcCount = vecS1Real * this->s2TileRealAlign8_;
            // load Y & P
            DataCopyPad(
                yBuffer, this->ySpace_[vecS1Offset * this->cubeS2Tile_],
                {static_cast<uint16_t>(vecS1Real),
                 static_cast<uint32_t>(this->s2TileReal_ * sizeof(CalcType)), // 本api会右填充dummy保证32B align
                 /*srcStride*/ static_cast<uint32_t>((this->cubeS2Tile_ - this->s2TileReal_) * sizeof(CalcType)),
                 /*dstStride*/ 0, 0},
                {true, 0, static_cast<uint8_t>(this->s2TileRealAlign8_ - this->s2TileReal_),
                 0.0f}); // 填充非零值，否则影响loss reduce
            DataCopyPad(
                pBuffer, this->pSpace_[vecS1Offset * this->cubeS2Tile_],
                {static_cast<uint16_t>(vecS1Real),
                 static_cast<uint32_t>(this->s2TileReal_ * sizeof(CalcType)), // 本api会右填充dummy保证32B align
                 /*srcStride*/ static_cast<uint32_t>((this->cubeS2Tile_ - this->s2TileReal_) * sizeof(CalcType)),
                 /*dstStride*/ 0, 0},
                {true, 0, static_cast<uint8_t>(this->s2TileRealAlign8_ - this->s2TileReal_),
                 0.0f}); // 填充0，否则影响loss reduce

            SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
            Muls(pBuffer, pBuffer, this->l1NormScale_, calcCount);
            PipeBarrier<PIPE_V>();

            // load relu: [NI cubeS1Tile_ cubeS2Tile_]
            DataCopyPad(
                reluBuffer,
                this->reluSpace_[head * this->cubeS1Tile_ * this->cubeS2Tile_ + vecS1Offset * this->cubeS2Tile_],
                {static_cast<uint16_t>(vecS1Real),
                 static_cast<uint32_t>(this->s2TileReal_ * sizeof(CalcType)), // 本api会右填充dummy保证32B align
                 /*srcStride*/ static_cast<uint32_t>((this->cubeS2Tile_ - this->s2TileReal_) * sizeof(CalcType)),
                 /*dstStride*/ 0, 0},
                {false, 0, 0, 0});
            // 计算dI
            Sub(diBuffer, yBuffer, pBuffer, calcCount);
            PipeBarrier<PIPE_V>();

            // 计算dS': dI * weight
            uint32_t mulLoopLimit = (this->s2TileReal_ + 64 - 1) / 64;
            for (uint32_t mulLoop = 0; mulLoop < mulLoopLimit; mulLoop++) {
                uint32_t maskCount = (mulLoop == mulLoopLimit - 1) ? (this->s2TileReal_ - mulLoop * 64) : 64;
                Mul(dsBuffer[mulLoop * 64], diBuffer[mulLoop * 64], weightBuffer[vecS1Offset * 8], maskCount,
                    /*repeatTime*/ vecS1Real,
                    {static_cast<uint8_t>(1), static_cast<uint8_t>(1), /*src1BlkStrideIn*/ 0,
                     static_cast<uint8_t>(this->s2TileRealAlign8_ / 8),
                     static_cast<uint8_t>(this->s2TileRealAlign8_ / 8),
                     /*src1RepStrideIn*/ static_cast<uint8_t>(1)});
                PipeBarrier<PIPE_V>();
            }

            // 计算dS(relu grad)
            SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);

            // calc relu > 0 position
            CompareScalar(reluMaskBuffer, reluBuffer, static_cast<CalcType>(0), CMPMODE::GT,
                          (calcCount + 63) / 64 * 64); // 需要保证calCount个元素所占空间256字节对齐
            PipeBarrier<PIPE_V>();
            Select(dsBuffer, reluMaskBuffer, dsBuffer, static_cast<CalcType>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE,
                   calcCount); // calcCount不能超过64*255
            PipeBarrier<PIPE_V>();
            Cast(dsBufferRaw, dsBuffer, RoundMode::CAST_RINT, calcCount);
            PipeBarrier<PIPE_V>();

            // 搬出ds
            SetFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
            // flatten搬出: dsSpace layout: [s1TileReal_, s2TileRealAlign8_]
            DataCopyPad(
                dsSpace[vecS1Offset * this->s2TileRealAlign8_], dsBufferRaw,
                {static_cast<uint16_t>(1), static_cast<uint32_t>(vecS1Real * this->s2TileRealAlign8_ * sizeof(InType)),
                 /*srcStride*/ 0, /*dstStride*/ 0, 0});

            // 计算dw
            Mul(reluBuffer, reluBuffer, diBuffer, calcCount); // diBuffer 到此已经没用了
            PipeBarrier<PIPE_V>();
            // 用高阶api Sum对s2维度reduce(不支持源操作数与目的操作数地址重叠)
            AscendC::SumParams params;
            params.outter = vecS1Real;
            params.n = this->s2TileReal_;
            params.inner = this->s2TileRealAlign8_;
            // from hw: "按n累加, padding部分不会纳入计算"
            AscendC::Sum(dwBuffer[vecS1Offset], reluBuffer, sharedTmpBuffer, params); // TODO: 确保workbuf足够
            PipeBarrier<PIPE_V>();

            // 计算loss - 最后一个head进行
            // 此时pBuffer yBuffer diBuffer可用
            CalcGraphLoss(head, pBuffer, yBuffer, diBuffer, calcCount);

            // 下次搬入需要等本次计算完成
            PipeBarrier<PIPE_ALL>();
        } // for s1

        // 确保dS搬出、dw计算都完成
        // PipeBarrier<PIPE_ALL>();

        // 搬出dw: dw按[NI T1] layout存
        SetAtomicAdd<CalcType>();
        DataCopyPad(this->dw_[head * this->T1_ + this->offsetForT1_], dwBuffer,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(this->s1TileReal_ * sizeof(CalcType)),
                     /*srcStride*/ 0, /*dstStride*/ 0, 0});
        SetAtomicNone();

        // 确保上个mm2、mm3完成
        /*
            waitIterateAll:
            - Cube就一个，串行执行iterateAll;
            -
           入参waitIterateAll=true才能和WaitIterateAll接口配合等待，之前若干个iterateAll异步，wait等的是这个入参设置为true的iterateAll
            - from hw: "CUBE侧就一个单元，mm2算完才能mm3"
        */
        if (head > 0) {
            mm3.WaitIterateAll();
        }

        // IterateAll内部会插入s_mte3和pipe_barrier(MTE3), 会等前面mte3和scalar都做完才发消息给cube
        LaunchMM23(head);

        // 确保本head任务都完成再开始下个head
        // TODO: PIPE_ALL不等待cube？
        PipeBarrier<PIPE_ALL>();
    } // for head
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::CalcGraphLoss(uint32_t head, LocalTensor<CalcType>& p,
                                                                 LocalTensor<CalcType>& y, LocalTensor<CalcType>& loss,
                                                                 uint32_t calcCount)
{
    if (head == this->NI_ - 1) {
        // p = p < delta ? delta : p
        AscendC::Maxs(loss, p, this->delta_, calcCount);
        PipeBarrier<PIPE_V>();

        // y = y < delta ? delta : y
        AscendC::Maxs(y, y, this->delta_, calcCount);
        PipeBarrier<PIPE_V>();

        // log(p)
        Ln(loss, loss, calcCount);
        PipeBarrier<PIPE_V>();

        // log(y)
        Ln(y, y, calcCount);
        PipeBarrier<PIPE_V>();

        // log(p) - log(y)
        Sub(loss, loss, y, calcCount);
        PipeBarrier<PIPE_V>();

        // p * (log(p) - log(y))
        Mul(loss, p, loss, calcCount);
        PipeBarrier<PIPE_V>();

        ReduceSum<CalcType>(y, loss, /*sharedTmpBuffer，占位符，不会真正用到*/ p, calcCount);
        PipeBarrier<PIPE_V>();

        Add(this->lossBuffer_, lossBuffer_, y, /*calCount*/ 1);
        PipeBarrier<PIPE_V>();
    }
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::LaunchMM23(uint32_t head)
{
    // dk = ds.T @ q = [s1TileReal_, s2TileRealAlign8_].T @ [T1 NI DI] = [KA M].T @ [KB N]
    uint64_t qOffset = this->offsetForT1_ * this->NI_ * this->DI_ + head * this->DI_;
    uint64_t dkOffset = this->offsetForT2_ * this->DI_; // [T2 DI]
    mm2.SetOrgShape(this->s2TileRealAlign8_, this->NI_ * this->DI_, this->s1TileReal_, this->T1_,
                    this->DI_); // TODO: double check这里，尤其N和KB
    mm2.SetTail(this->s2TileReal_, this->DI_, this->s1TileReal_);
    mm2.SetTensorA(this->SGradSpace(head), true);
    mm2.SetTensorB(this->queryIndex_[qOffset], false);
    bool useSync = head == this->NI_ - 1;
    if (useSync) { // 最后一个head，使用同步模式
        mm2.template IterateAll</*sync*/ true>(this->dkSpace_[dkOffset], /*enAtomic*/ true, /*enSequentialWrite*/ false,
                                               /*waitIterateAll*/ false);
    } else { // 这里iterateall不需要被wait
        mm2.template IterateAll</*sync*/ false>(this->dkSpace_[dkOffset], /*enAtomic*/ true,
                                                /*enSequentialWrite*/ false, /*waitIterateAll*/ false);
    }

    // register mm obj切换不需要mm.End(), IterateAll内部自动使能该功能

    // dq = ds @ k = [s1TileReal_, s2TileRealAlign8_] @ [T2 DI] = [M KA] @ [KB N]
    uint64_t kOffset = this->offsetForT2_ * this->DI_;
    uint64_t dqOffset = this->offsetForT1_ * this->NI_ * this->DI_ + head * this->DI_; // [T1 NI DI]
    mm3.SetOrgShape(this->s1TileReal_, this->DI_, this->s2TileRealAlign8_, this->T2_, this->NI_ * this->DI_);
    mm3.SetTail(this->s1TileReal_, this->DI_, this->s2TileReal_);
    mm3.SetTensorA(this->SGradSpace(head), false);
    mm3.SetTensorB(this->keyIndex_[kOffset], false);
    if (useSync) { // 最后一个head，使用同步模式
        mm3.template IterateAll</*sync*/ true>(this->dqSpace_[dqOffset], /*enAtomic*/ true, /*enSequentialWrite*/ false,
                                               /*waitIterateAll*/ false);
    } else { // 非最后一个head, iterateall需要被wait
        mm3.template IterateAll</*sync*/ false>(this->dqSpace_[dqOffset], /*enAtomic*/ true,
                                                /*enSequentialWrite*/ false, /*waitIterateAll*/ true);
    }
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::CalcGraphY()
{
    // calc s1 loop info
    // 确保head不能超过cubeS1Tile_
    uint32_t vecS1Tile = this->cubeS1Tile_ / this->NI_;
    vecS1Tile = vecS1Tile > 8 ? 8 : vecS1Tile;
    const uint32_t vecLoopLimit = CeilDiv(this->s1TileReal_, vecS1Tile);
    const uint32_t firstVecS1Tile = (vecLoopLimit == 1 ? this->s1TileReal_ : vecS1Tile);

    // ub划分
    const uint32_t bufferElements = vecS1Tile * this->cubeS2Tile_;
    const uint32_t bufferBytes = bufferElements * sizeof(CalcType);
    const uint32_t weightElements = this->cubeS1Tile_ * 8; // fp32类型
    const uint32_t weightBytes = weightElements * sizeof(CalcType);
    const uint32_t maskElements = vecS1Tile * this->cubeS1Tile_;
    const uint32_t maskBytes = maskElements * sizeof(uint8_t);
    const uint32_t softmaxElements = this->cubeS1Tile_ * 8;
    const uint32_t softmaxBytes = softmaxElements * sizeof(CalcType);
    LocalTensor<CalcType> qkBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, 0);
    LocalTensor<CalcType> iBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, bufferBytes);
    LocalTensor<CalcType> apiBuffer =
        this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, bufferBytes * 2); // 32KB
    LocalTensor<uint8_t> sharedTmpBuffer = apiBuffer.template ReinterpretCast<uint8_t>();
    LocalTensor<CalcType> weightBuffer = this->ubBuf_.template GetWithOffset<CalcType>(weightElements, bufferBytes * 3);
    LocalTensor<uint8_t> maskBuffer =
        this->ubBuf_.template GetWithOffset<uint8_t>(maskElements, bufferBytes * 3 + weightBytes); // 1KB
    LocalTensor<CalcType> maxBuffer = this->ubBuf_.template GetWithOffset<CalcType>(
        softmaxElements, bufferBytes * 3 + weightBytes + maskBytes); // 固定4KB
    LocalTensor<CalcType> sumBuffer = this->ubBuf_.template GetWithOffset<CalcType>(
        softmaxElements, bufferBytes * 3 + weightBytes + maskBytes + softmaxBytes); // 固定4KB

    // 提前发射Q_INDEX @ K_INDEX.T: [T1NI DI] @ [T2 DI].T
    uint64_t queryOffset = this->offsetForT1_ * this->NI_ * this->DI_;
    uint64_t keyOffset = this->offsetForT2_ * this->DI_;
    mm1.SetOrgShape(this->s1_, this->s2_, this->DI_, this->DI_, this->cubeS2Tile_);
    mm1.SetTail(firstVecS1Tile * this->NI_, this->s2TileReal_, this->DI_);
    mm1.SetTensorA(this->queryIndex_[queryOffset], false);
    mm1.SetTensorB(this->keyIndex_[keyOffset], true);
    mm1.template IterateAll</*sync*/ false>(this->QKSpace(0), /*enAtomic*/ false, /*enSequentialWrite*/ false,
                                            /*waitIterateAll*/ true);

    // 搬入softmax max + sum: [T] layout
    DataCopyPad(maxBuffer, this->maxIndex_[this->offsetForT1_],
                {static_cast<uint16_t>(this->s1TileReal_),
                 static_cast<uint32_t>(
                     sizeof(CalcType)), // leftPadding、rightPadding都为0：dummy会默认填充待搬运数据块的第一个元素值
                 /*srcStride*/ 0, /*dstStride*/ 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(sumBuffer, this->sumIndex_[this->offsetForT1_],
                {static_cast<uint16_t>(this->s1TileReal_),
                 static_cast<uint32_t>(
                     sizeof(CalcType)), // leftPadding、rightPadding都为0：dummy会默认填充待搬运数据块的第一个元素值
                 /*srcStride*/ 0, /*dstStride*/ 0, 0},
                {false, 0, 0, 0});

    /* for s1 loop:
        时间轴:
            ---- 下个s1 cube计算 ----
            ---- 本s1 vector计算 ----
    */
    for (uint32_t vecLoop = 0; vecLoop < vecLoopLimit; ++vecLoop) {
        mm1.WaitIterateAll();

        const uint32_t vecS1Real = (vecLoop == vecLoopLimit - 1 ? this->s1TileReal_ - vecLoop * vecS1Tile : vecS1Tile);
        const uint32_t vecS1Offset = vecLoop * vecS1Tile;
        const uint32_t calcCount = vecS1Real * this->s2TileRealAlign8_;

        // for (int i = 0; i < vecS1Real; ++i) {
        //     uint32_t array[] = {static_cast<uint32_t>(1), static_cast<uint32_t>(this->s2TileReal_)};
        //     AscendC::ShapeInfo shapeInfo(2, array);
        //     AscendC::DumpTensor(this->QKSpace(vecLoop)[i * this->cubeS2Tile_], 33333, this->s2TileReal_, shapeInfo);
        // }

        //// 下个vecS1Tile cube计算 ////
        if (vecLoop < vecLoopLimit - 1) {
            uint64_t queryOffsetNext = (this->offsetForT1_ + (vecLoop + 1) * vecS1Tile) * this->NI_ * this->DI_;
            // uint64_t keyOffsetNext = (this->offsetForT2_ + (vecLoop + 1) * vecS1Tile) * this->DI_;
            uint64_t keyOffsetNext = keyOffset; // s2没有移动
            mm1.SetOrgShape(this->s1_, this->s2_, this->DI_, this->DI_, this->cubeS2Tile_);
            const uint32_t vecS1RealNext =
                ((vecLoop + 1) == vecLoopLimit - 1 ? this->s1TileReal_ - (vecLoop + 1) * vecS1Tile : vecS1Tile);
            mm1.SetTail(vecS1RealNext * this->NI_, this->s2TileReal_, this->DI_);
            mm1.SetTensorA(this->queryIndex_[queryOffsetNext], false);
            mm1.SetTensorB(this->keyIndex_[keyOffsetNext], true);
            mm1.template IterateAll</*sync*/ false>(this->QKSpace(vecLoop + 1), /*enAtomic*/ false,
                                                    /*enSequentialWrite*/ false, /*waitIterateAll*/ true);
        }

        //// 本vecS1Tile vector计算 ////
        GlobalTensor<CalcType>& qkSpace = this->QKSpace(vecLoop);

        // load weights: from [T1 NI] layout -> [vecS1Real NI 8]
        DataCopyPad(weightBuffer, this->weights_[(this->offsetForT1_ + vecS1Offset) * this->NI_],
                    {static_cast<uint16_t>(vecS1Real * this->NI_),
                     static_cast<uint32_t>(
                         sizeof(CalcType)), // leftPadding、rightPadding都为0：dummy会默认填充待搬运数据块的第一个元素值
                     /*srcStride*/ 0, /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
        PipeBarrier<PIPE_V>();

        for (uint32_t head = 0; head < this->NI_; ++head) {
            // load qk: [vecS1Tile NI cubeS2Tile_] layout
            DataCopyPad(qkBuffer, qkSpace[head * this->cubeS2Tile_],
                        {static_cast<uint16_t>(vecS1Real),
                         static_cast<uint32_t>(this->s2TileReal_ * sizeof(CalcType)), // 本api会右填充dummy保证32B align
                                                                                      /*srcStride*/
                         static_cast<uint32_t>((this->NI_ * this->cubeS2Tile_ - this->s2TileReal_) * sizeof(CalcType)),
                         /*dstStride*/ 0, 0},
                        {false, 0, 0, 0});
            SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);

            // relu
            Relu(qkBuffer, qkBuffer, calcCount);
            PipeBarrier<PIPE_V>();

            // relu结果落盘: [NI cubeS1Tile_ cubeS2Tile_]
            SetFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
            DataCopyPad(
                this->reluSpace_[head * this->cubeS1Tile_ * this->cubeS2Tile_ + vecS1Offset * this->cubeS2Tile_],
                qkBuffer,
                {static_cast<uint16_t>(vecS1Real), static_cast<uint32_t>(this->s2TileReal_ * sizeof(CalcType)),
                 /*srcStride*/ 0,
                 /*dstStride*/ static_cast<uint32_t>((this->cubeS2Tile_ - this->s2TileReal_) * sizeof(CalcType)), 0});
            SetFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV_);
            WaitFlag<HardEvent::MTE3_V>(this->eventIdMte3ToV_);

            // mul weights
            uint32_t mulLoopLimit = (this->s2TileReal_ + 64 - 1) / 64;
            for (uint32_t mulLoop = 0; mulLoop < mulLoopLimit; mulLoop++) {
                uint32_t maskCount = (mulLoop == mulLoopLimit - 1) ? (this->s2TileReal_ - mulLoop * 64) : 64;
                Mul(qkBuffer[mulLoop * 64], qkBuffer[mulLoop * 64], weightBuffer[head * 8], maskCount,
                    /*repeatTime*/ vecS1Real,
                    {static_cast<uint8_t>(1), static_cast<uint8_t>(1), /*src1BlkStrideIn*/ 0,
                     static_cast<uint8_t>(this->s2TileRealAlign8_ / 8),
                     static_cast<uint8_t>(this->s2TileRealAlign8_ / 8),
                     /*src1RepStrideIn*/ static_cast<uint8_t>(this->NI_ * 8 / 8)}); // TODO: NI有限制，uint8_t类型repeat
                                                                                    // stride不能超过255
                PipeBarrier<PIPE_V>();
            }

            // reduce sum over head
            if (head > 0) {
                Add(iBuffer, qkBuffer, iBuffer, calcCount);
            } else {
                Adds(iBuffer, qkBuffer, 0.0f, calcCount);
            }
            // 下次搬入需要等本次计算完成
            PipeBarrier<PIPE_ALL>();
        } // for head

        // apply atten mask
        this->PreSoftmax(iBuffer, maskBuffer, sharedTmpBuffer, vecS1Real, vecS1Offset);

        // Softmax: exp(src - max) / sum
        this->Softmax(iBuffer, apiBuffer, maxBuffer, sumBuffer, vecS1Real, vecS1Offset, this->s2TileReal_,
                      this->s2TileRealAlign8_);

        // reduce结果搬出
        SetFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
        WaitFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
        DataCopyPad(
            this->ySpace_[vecS1Offset * this->cubeS2Tile_], iBuffer,
            {static_cast<uint16_t>(vecS1Real), static_cast<uint32_t>(this->s2TileReal_ * sizeof(CalcType)),
             /*srcStride*/ 0,
             /*dstStride*/ static_cast<uint32_t>((this->cubeS2Tile_ - this->s2TileReal_) * sizeof(CalcType)), 0});
        // 下次搬入需要等本次搬出完成
        PipeBarrier<PIPE_ALL>();
    } // for s1

    // for (int i = 0; i < this->s1TileReal_; ++i) {
    //     uint32_t array[] = {static_cast<uint32_t>(1), static_cast<uint32_t>(this->s2TileRealAlign8_)};
    //     AscendC::ShapeInfo shapeInfo(2, array);
    //     AscendC::DumpTensor(this->ySpace_[i * this->cubeS2Tile_], 22222, this->s2TileRealAlign8_, shapeInfo);
    // }
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::CalcGraphP()
{
    // 提前发射Q @ K.T: [T1 N D] @ [T2 N D].T
    uint64_t queryOffset = this->offsetForT1_ * this->N_ * this->D_;
    uint64_t keyOffset = this->offsetForT2_ * this->N_ * this->D_;
    mm1.SetOrgShape(this->s1_, this->s2_, this->N_ * this->D_, this->N_ * this->D_, this->cubeS2Tile_);
    mm1.SetTail(this->s1TileReal_, this->s2TileReal_, this->D_);
    mm1.SetTensorA(this->query_[queryOffset], false);
    mm1.SetTensorB(this->key_[keyOffset], true);
    mm1.template IterateAll</*sync*/ false>(this->QKSpace(0), /*enAtomic*/ false, /*enSequentialWrite*/ false,
                                            /*waitIterateAll*/ true);

    // ub划分
    const uint32_t vecS1Tile = 16; // 从下方ub使用情况，判断这里可以按16 loop
    const uint32_t bufferElements = vecS1Tile * this->cubeS2Tile_;
    const uint32_t bufferBytes = bufferElements * sizeof(CalcType);
    const uint32_t maskBufferElements = vecS1Tile * this->cubeS1Tile_;
    const uint32_t maskBufferBytes = maskBufferElements * sizeof(uint8_t);
    const uint32_t softmaxElements = this->cubeS1Tile_ * 8;
    const uint32_t softmaxBytes = softmaxElements * sizeof(CalcType);
    LocalTensor<CalcType> qkBuffer = this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, 0); // 32KB
    LocalTensor<CalcType> apiBuffer =
        this->ubBuf_.template GetWithOffset<CalcType>(bufferElements, bufferBytes); // 32KB
    LocalTensor<uint8_t> sharedTmpBuffer = apiBuffer.template ReinterpretCast<uint8_t>();
    LocalTensor<uint8_t> maskBuffer =
        this->ubBuf_.template GetWithOffset<uint8_t>(maskBufferElements, bufferBytes * 2); // 1KB
    LocalTensor<CalcType> maxBuffer =
        this->ubBuf_.template GetWithOffset<CalcType>(softmaxElements, bufferBytes * 2 + maskBufferBytes); // 固定4KB
    LocalTensor<CalcType> sumBuffer = this->ubBuf_.template GetWithOffset<CalcType>(
        softmaxElements, bufferBytes * 2 + maskBufferBytes + softmaxBytes); // 固定4KB

    /* for N loop:
        时间轴:
            ---- 下个head cube计算 ----
            ---- 本head vector计算 ----
    */
    for (uint32_t head = 0; head < this->N_; ++head) {
        mm1.WaitIterateAll();

        //// 下个head cube计算 ////
        if (head < this->N_ - 1) {
            uint64_t queryOffsetNextHead = queryOffset + (head + 1) * this->D_;
            uint64_t keyOffsetNextHead = keyOffset + (head + 1) * this->D_;
            mm1.SetOrgShape(this->s1_, this->s2_, this->N_ * this->D_, this->N_ * this->D_, this->cubeS2Tile_);
            mm1.SetTail(this->s1TileReal_, this->s2TileReal_, this->D_);
            mm1.SetTensorA(this->query_[queryOffsetNextHead], false);
            mm1.SetTensorB(this->key_[keyOffsetNextHead], true);
            mm1.template IterateAll</*sync*/ false>(this->QKSpace(head + 1), /*enAtomic*/ false,
                                                    /*enSequentialWrite*/ false, /*waitIterateAll*/ true);
        }

        //// 本head vector计算 ////
        const uint32_t vecLoopLimit = CeilDiv(this->s1TileReal_, vecS1Tile);
        GlobalTensor<CalcType>& qkSpace = this->QKSpace(head);

        // 搬入softmax max + sum: [N T] layout
        DataCopyPad(maxBuffer, this->max_[head * this->T1_ + this->offsetForT1_],
                    {static_cast<uint16_t>(this->s1TileReal_),
                     static_cast<uint32_t>(
                         sizeof(CalcType)), // leftPadding、rightPadding都为0：dummy会默认填充待搬运数据块的第一个元素值
                     /*srcStride*/ 0, /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(sumBuffer, this->sum_[head * this->T1_ + this->offsetForT1_],
                    {static_cast<uint16_t>(this->s1TileReal_),
                     static_cast<uint32_t>(
                         sizeof(CalcType)), // leftPadding、rightPadding都为0：dummy会默认填充待搬运数据块的第一个元素值
                     /*srcStride*/ 0, /*dstStride*/ 0, 0},
                    {false, 0, 0, 0});

        // TODO: vec流水精调，MTE2可以提前发射之类
        for (uint32_t vecLoop = 0; vecLoop < vecLoopLimit; ++vecLoop) {
            const uint32_t vecS1Real =
                (vecLoop == vecLoopLimit - 1 ? this->s1TileReal_ - vecLoop * vecS1Tile : vecS1Tile);
            const uint32_t vecS1Offset = vecLoop * vecS1Tile;
            const uint32_t calcCount = vecS1Real * this->s2TileRealAlign8_;

            // Mul scale
            DataCopyPad(
                qkBuffer, qkSpace[vecS1Offset * this->cubeS2Tile_],
                {static_cast<uint16_t>(vecS1Real),
                 static_cast<uint32_t>(this->s2TileReal_ * sizeof(CalcType)), // 本api会右填充dummy保证32B align
                 /*srcStride*/ static_cast<uint32_t>((this->cubeS2Tile_ - this->s2TileReal_) * sizeof(CalcType)),
                 /*dstStride*/ 0, 0},
                {false, 0, 0, 0});

            SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);

            Muls(qkBuffer, qkBuffer, this->scale_, calcCount);
            PipeBarrier<PIPE_V>();

            // PreSoftmax
            this->PreSoftmax(qkBuffer, maskBuffer, sharedTmpBuffer, vecS1Real, vecS1Offset);

            // Softmax: exp(src - max) / sum
            this->Softmax(qkBuffer, apiBuffer, maxBuffer, sumBuffer, vecS1Real, vecS1Offset, this->s2TileReal_,
                          this->s2TileRealAlign8_);

            // 搬出到pSpace
            SetFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);
            WaitFlag<HardEvent::V_MTE3>(this->eventIdVToMte3_);

            if (head > 0) {
                SetAtomicAdd<CalcType>();
            }
            DataCopyPad(
                this->pSpace_[vecS1Offset * this->cubeS2Tile_], qkBuffer,
                {static_cast<uint16_t>(vecS1Real), static_cast<uint32_t>(this->s2TileReal_ * sizeof(CalcType)),
                 /*srcStride*/ 0,
                 /*dstStride*/ static_cast<uint32_t>((this->cubeS2Tile_ - this->s2TileReal_) * sizeof(CalcType)), 0});
            if (head > 0) {
                SetAtomicNone();
            }

            // 下次搬入需要等本次搬出完成
            PipeBarrier<PIPE_ALL>();
        } // for vec
    } // for head

    // for (int i = 0; i < this->s1TileReal_; ++i) {
    //     uint32_t array[] = {static_cast<uint32_t>(1), static_cast<uint32_t>(this->s2TileRealAlign8_)};
    //     AscendC::ShapeInfo shapeInfo(2, array);
    //     AscendC::DumpTensor(this->pSpace_[i * this->cubeS2Tile_], 11111, this->s2TileRealAlign8_, shapeInfo);
    // }
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::PreSoftmax(LocalTensor<CalcType>& src,
                                                              LocalTensor<uint8_t>& maskBuffer,
                                                              LocalTensor<uint8_t>& sharedTmpBuffer, uint64_t s1Len,
                                                              uint64_t s1Offset)
{
    if (this->tileMode_ == CAUSAL_TILE) {
        // mask shape = [cubeS1Tile_, cubeS1Tile_]
        DataCopyPad(
            maskBuffer, this->mask_[s1Offset * this->cubeS1Tile_],
            {static_cast<uint16_t>(s1Len), static_cast<uint32_t>(this->s2TileRealAlign32_ * sizeof(uint8_t)),
             /*srcStride*/ static_cast<uint32_t>((this->cubeS1Tile_ - this->s2TileRealAlign32_) * sizeof(uint8_t)),
             /*dstStride*/ 0, 0},
            {false, 0, 0, 0});

        SetFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);
        WaitFlag<HardEvent::MTE2_V>(this->eventIdMte2ToV_);

        SelectWithBytesMaskShapeInfo info;
        info.firstAxis = s1Len;
        info.srcLastAxis = this->s2TileRealAlign8_;
        info.maskLastAxis = this->s2TileRealAlign32_;
        src.SetSize(info.firstAxis * info.srcLastAxis);
        maskBuffer.SetSize(info.firstAxis * info.maskLastAxis);
        /* "src0为srcTensor（tensor类型），src1为srcScalar（scalar类型）:
            若mask的值为0，选择srcTensor相应的值放入dstLocal，否则选择srcScalar的值放入dstLocal"
        */
        // mask正确赋值右上角为1
        SelectWithBytesMask(src, /*src0*/ src, /*src1*/ this->negativeScalar_, /*mask*/ maskBuffer, sharedTmpBuffer,
                            info);
        PipeBarrier<PIPE_V>();
    }
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::Softmax(LocalTensor<CalcType>& src, LocalTensor<CalcType>& common,
                                                           const LocalTensor<CalcType>& max,
                                                           const LocalTensor<CalcType>& sum, uint64_t s1Len,
                                                           uint64_t s1Offset, uint64_t s2Len, uint64_t s2LenAlign8)
{
    uint32_t softmaxLoopLimit = (s2Len + 64 - 1) / 64;
    for (uint32_t softmaxLoop = 0; softmaxLoop < softmaxLoopLimit; softmaxLoop++) {
        uint32_t maskCount = (softmaxLoop == softmaxLoopLimit - 1) ? (s2Len - softmaxLoop * 64) : 64;
        Sub(src[softmaxLoop * 64], src[softmaxLoop * 64], max[s1Offset * 8], maskCount, /*repeatTime*/ s1Len,
            {static_cast<uint8_t>(1), static_cast<uint8_t>(1), /*src1BlkStrideIn*/ 0,
             static_cast<uint8_t>(s2LenAlign8 / 8), static_cast<uint8_t>(s2LenAlign8 / 8), /*src1RepStrideIn*/ 1});
        PipeBarrier<PIPE_V>();
        // TODO: Exp不支持源操作数与目的操作数地址重叠?
        Exp(common[softmaxLoop * 64], src[softmaxLoop * 64], maskCount, s1Len,
            {static_cast<uint16_t>(1), /*srcBlkStrideIn*/ static_cast<uint16_t>(1),
             static_cast<uint8_t>(s2LenAlign8 / 8), /*srcRepStrideIn*/ static_cast<uint8_t>(s2LenAlign8 / 8)});
        PipeBarrier<PIPE_V>();
        Div(src[softmaxLoop * 64], common[softmaxLoop * 64], sum[s1Offset * 8], maskCount, s1Len,
            {static_cast<uint8_t>(1), static_cast<uint8_t>(1), 0, static_cast<uint8_t>(s2LenAlign8 / 8),
             static_cast<uint8_t>(s2LenAlign8 / 8), 1});
        PipeBarrier<PIPE_V>();
    }
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::CalcS1TileInfo(const uint32_t s1TileIdx)
{
    uint32_t bIdx = this->bIdx_;           // 初始为0
    uint32_t cuS1Tiles = this->cuS1Tiles_; // 初始为0
    uint64_t cuS1 = this->cuS1_;           // 初始为0
    uint64_t cuS2 = this->cuS2_;           // 初始为0
    for (; bIdx < this->B_; ++bIdx) {
        this->s1_ = QLenForBatch(bIdx);
        uint32_t s1TileNum = CeilDiv(this->s1_, this->cubeS1Tile_);
        this->s2_ = KVLenForBatch(bIdx);
        if (cuS1Tiles + s1TileNum > s1TileIdx) {
            this->bIdx_ = bIdx;
            this->cuS1Tiles_ = cuS1Tiles;
            this->cuS1_ = cuS1;
            this->cuS2_ = cuS2;
            this->s1Idx_ = s1TileIdx - cuS1Tiles;
            this->s1TileReal_ =
                (this->s1Idx_ == s1TileNum - 1 ? this->s1_ - this->s1Idx_ * this->cubeS1Tile_ : this->cubeS1Tile_);
            // 计算s2信息
            this->historyLen_ = (this->s2_ - this->s1_) > 0 ? (this->s2_ - this->s1_) : 0;
            this->causalLen_ = this->historyLen_ + this->s1Idx_ * this->cubeS1Tile_;
            this->s2TileLimit_ = CeilDiv(this->causalLen_, this->cubeS2Tile_) + 1 /*diagnoal tile*/;
            this->offsetForT1_ = this->cuS1_ + this->s1Idx_ * this->cubeS1Tile_;
            break;
        }
        cuS1Tiles += s1TileNum;
        cuS1 += this->s1_;
        cuS2 += this->s2_;
    }
    // PRINTF("CORE, this->bIdx_, this->s1_, this->s2_, this->cuS1Tiles_, this->cuS1_, this->cuS2_, this->s1Idx_,
    // this->s1TileReal_, this->historyLen_, this->causalLen_, this->s2TileLimit_, this->offsetForT1_: %d %d %d %d %d %d
    // %d %d %d %d %d %d %d\n", this->blockIdx_, this->bIdx_, this->s1_, this->s2_, this->cuS1Tiles_, this->cuS1_,
    // this->cuS2_, this->s1Idx_, this->s1TileReal_, this->historyLen_, this->causalLen_, this->s2TileLimit_,
    // this->offsetForT1_);
}

template <typename InType, typename CalcType>
__aicore__ inline void DsaDense<InType, CalcType>::CalcS2TileInfo(const uint32_t s2TileIdx)
{
    this->s2Idx_ = s2TileIdx;
    if (s2TileIdx < this->s2TileLimit_ - 1) {
        this->tileMode_ = NORMAL_TILE;
        if ((s2TileIdx + 1) * this->cubeS2Tile_ > this->causalLen_) {
            this->s2TileReal_ = this->causalLen_ - s2TileIdx * this->cubeS2Tile_;
        } else {
            this->s2TileReal_ = this->cubeS2Tile_;
        }
    } else { // 对角线上的块
        this->tileMode_ = CAUSAL_TILE;
        this->s2TileReal_ = this->s1TileReal_;
    }
    this->s2TileRealAlign8_ = (this->s2TileReal_ + 7) / 8 * 8;
    this->s2TileRealAlign16_ = (this->s2TileReal_ + 15) / 16 * 16;
    this->s2TileRealAlign32_ = (this->s2TileReal_ + 31) / 32 * 32;
    this->offsetForT2_ =
        this->cuS2_ +
        (this->tileMode_ == CAUSAL_TILE ? this->causalLen_ : static_cast<uint64_t>(this->s2Idx_) * this->cubeS2Tile_);

    // PRINTF("CORE, this->bIdx_, this->s2TileReal_, this->tileMode_: %d %d %d %d \n", this->blockIdx_, this->bIdx_,
    // this->s2TileReal_, this->tileMode_);
}

} // namespace ops_dsa_dense

#endif
