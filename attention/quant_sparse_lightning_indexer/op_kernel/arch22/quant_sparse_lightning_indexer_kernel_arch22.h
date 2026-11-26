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
 * \file quant_sparse_lightning_indexer_kernel_arch22.h
 * \brief
 */

#ifndef QUANT_SPARSE_LIGHTNING_INDEXER_KERNEL_H
#define QUANT_SPARSE_LIGHTNING_INDEXER_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_sparse_lightning_indexer_common_arch22.h"
#include "quant_sparse_lightning_indexer_service_vector_arch22.h"
#include "quant_sparse_lightning_indexer_service_cube_arch22.h"
#include "../quant_sparse_lightning_indexer_metadata.h"

namespace QSLIKernel {
using namespace QSLICommon;
using namespace QSLIServiceVec;
using namespace matmul;
using namespace optiling::detail;
using namespace optiling;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

// 由于S2循环前，RunInfo还没有赋值，使用TempLoopInfo临时存放B、N、S1轴相关的信息
// 同时减少重复计算
struct TempLoopInfo {
    uint32_t bN2Idx = 0;
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t gS1Idx = 0U;
    uint32_t gS1LoopEnd = 0U;  // gS1方向循环的结束Idx
    uint32_t s2LoopEnd = 0U;   // S2方向循环的结束Idx
    uint32_t actS1Size = 1ULL; // 当前Batch循环处理的S1轴的实际大小
    uint32_t actS2Size = 0ULL;
    uint32_t actS2SizeOrig = 0ULL;
    bool curActSeqLenIsZero = false;
    bool needDealActS1LessThanS1 = false; // S1的实际长度小于shape的S1长度时，是否需要清理输出
    bool isNeedLD = false;                // 该基本块是否需要LD
    uint32_t actMBaseSize = 0U;           // m轴(gS1)方向实际大小
    uint32_t mBasicSizeTail = 0U;         // gS1方向循环的尾基本块大小
    uint32_t s2BasicSizeTail = 0U;        // S2方向循环的尾基本块大小
    uint32_t validS2Len = 0U;
    // 候选驱动（512 token 候选 tile）：本 batch 首行全局行号、任务内最大 L、整行 tile 上界
    uint32_t actS1RowBase = 0U;
    uint32_t maxCandLen = 0U;
    uint32_t maxCandTiles = 0U;
    bool candDriven = false;
};

template <typename QSLIT>
class QSLIPreload {
public:
    __aicore__ inline QSLIPreload(){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* weights,
                                __gm__ uint8_t* queryScale, __gm__ uint8_t* keyScale, __gm__ uint8_t* cuSeqlensQ,
                                __gm__ uint8_t* cuSeqlensK, __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedK,
                                __gm__ uint8_t* cmpResidualK, __gm__ uint8_t* blockTable,
                                __gm__ uint8_t* outputIdxOffset, __gm__ uint8_t* metadata,
                                __gm__ uint8_t* candidateBlockIndices, __gm__ uint8_t* candidateBlockLength,
                                __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseValues, __gm__ uint8_t* workspace,
                                const QSLITilingDataConsumer* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void Process();

    // =================================类型定义区=================================
    using Q_T = typename QSLIT::queryType;
    using K_T = typename QSLIT::keyType;
    using OUT_T = typename QSLIT::outputType;
    static constexpr bool PAGE_ATTENTION = QSLIT::pageAttention;
    static constexpr LI_LAYOUT Q_LAYOUT_T = QSLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QSLIT::keyLayout;

    using MM1_OUT_T = float;

    QSLIMatmul<QSLIT> matmulService;
    QSLIVector<QSLIT> vectorService;

    // =================================常量区=================================
    static constexpr uint32_t SYNC_C1_V1_FLAG = 4;
    static constexpr uint32_t SYNC_V1_C1_FLAG = 5;

    // 参照 v1: s1BaseSize 固定, mBaseSize = s1BaseSize * gSize (支持 gSize=32/64)
    static constexpr uint32_t S1_BASE_SIZE = 4;
    static constexpr uint32_t S2_BASE_SIZE = 2048;
    static constexpr uint32_t HEAD_DIM = 128;
    static constexpr uint32_t K_HEAD_NUM = 1;
    static constexpr uint32_t GM_ALIGN_BYTES = 512;
    static constexpr uint32_t LI_QUANT_PRELOAD_TASK_CACHE_SIZE = 2;

    // for workspace double
    static constexpr uint32_t WS_DOUBLE = 2;
    static constexpr uint32_t ELE_NUM_PER_BLOCK = 16;

    static constexpr int64_t LD_PREFETCH_LEN = 2;

protected:
    TPipe* pipe = nullptr;

    // offset
    uint64_t queryCoreOffset = 0ULL;
    uint64_t keyCoreOffset = 0ULL;
    uint64_t keyScaleCoreOffset = 0ULL;
    uint64_t weightsCoreOffset = 0ULL;
    uint64_t indiceOutCoreOffset = 0ULL;
    // 命名沿用拆分前(候选索引当时是输出): 此处是 candidate_block_indices 作为输入的按行偏移
    uint64_t candidateOutCoreOffset = 0ULL;
    uint64_t outputIdxOffsetCoreOffset = 0ULL; // A15
    uint32_t coreZeroEnable = 1U;

    // ================================Global Buffer区=================================
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<K_T> keyGm;
    GlobalTensor<half> weightsGm;
    GlobalTensor<uint32_t> metadataGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> candidateBlockIndicesInGm;
    GlobalTensor<int32_t> candidateBlockLengthGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> outputIdxOffsetGm; // A15: 每行输出索引偏移

    GlobalTensor<uint32_t> actualSeqLengthsGmQ;
    GlobalTensor<uint32_t> actualSeqLengthsGm;
    GlobalTensor<uint32_t> cmpResidualKGm;

    // ================================类成员变量====================================
    // aic、aiv核信息
    uint32_t tmpBlockIdx = 0U;
    uint32_t aiCoreIdx = 0U;

    QSLICommon::ConstInfo constInfo{};
    QSLICommon::LdSplitCoreInfo ldInfo{};
    TempLoopInfo tempLoopInfo{};
    uint32_t candTaskSeq_ = 0U; // 候选任务序号（AIC/AIV 同步递增）：W 槽、分数/aux 区 parity 基准

    // ================================Init functions==================================
    __aicore__ inline void InitTilingData(const QSLITilingDataConsumer* __restrict tilingData);
    __aicore__ inline void InitBuffers();
    __aicore__ inline void InitActualSeqLen(__gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensK,
                                            __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedK,
                                            __gm__ uint8_t* cmpResidualK);
    // ================================Split Core================================
    __aicore__ inline void SplitCore();
    __aicore__ inline uint32_t GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size, uint32_t actS2SizeOrig,
                                                       uint32_t& validS2Len);
    __aicore__ inline uint32_t GetTotalBaseBlockNum();
    // ================================Process functions================================
    __aicore__ inline void ProcessMain();
    __aicore__ inline void ProcessBaseBlock(uint32_t loop, uint64_t s2LoopIdx,
                                            QSLICommon::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE]);
    // 候选驱动任务级派发：AIV 先算 W 再跑 gather/消费流水，AIC 等 W 就绪后从 staging 算 QK
    __aicore__ inline void ProcessCandTask(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void ProcessInvalid();
    __aicore__ inline void ProcessDecode();
    // ================================Params Calc=====================================
    __aicore__ inline void CalcGS1LoopParams(uint32_t bN2Idx);
    __aicore__ inline void GetBN2Idx(uint32_t bN2Idx);
    __aicore__ inline uint32_t GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                               GlobalTensor<uint32_t>& actualSeqLengthsGm, uint32_t defaultSeqLen);
    __aicore__ inline uint32_t GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims, uint32_t cmpResiduaKLenDims,
                                                  bool isAccumSeq, GlobalTensor<uint32_t>& actualSeqLengthsGm,
                                                  GlobalTensor<uint32_t>& cmpResidualKGm, uint32_t defaultSeqLen,
                                                  uint32_t cmpRatio);
    __aicore__ inline void GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t& actS1Size, uint32_t& actS2Size,
                                               uint32_t& actS2SizeOrig);
    __aicore__ inline void CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx);
    __aicore__ inline void CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QSLICommon::RunInfo& runInfo);
    __aicore__ inline void DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start);
};

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::InitTilingData(const QSLITilingDataConsumer* __restrict tilingData)
{
    constInfo.batchSize = tilingData->bSize;
    constInfo.qHeadNum = constInfo.gSize = tilingData->gSize;
    constInfo.kSeqSize = tilingData->s2Size;
    constInfo.qSeqSize = tilingData->s1Size;
    constInfo.attenMaskFlag = (tilingData->sparseMode == 3);
    constInfo.kCacheBlockSize = tilingData->blockSize;
    constInfo.maxBlockNumPerBatch = tilingData->maxBlockNumPerBatch;
    constInfo.keyStride0 = tilingData->keyStride0; // A11: key 0轴非连续
    constInfo.keyDequantScaleStride0 = tilingData->keyDequantScaleStride0;
    constInfo.sparseCount = tilingData->sparseCount;
    constInfo.cmpRatio = tilingData->cmpRatio;
    constInfo.outputLayout = Q_LAYOUT_T; // 输出和输入形状一致
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS1 = true;
    }
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) {
        constInfo.isAccumSeqS2 = true;
    }

    constInfo.kHeadNum = K_HEAD_NUM;
    constInfo.headDim = HEAD_DIM;

    constInfo.s1BaseSize = S1_BASE_SIZE;
    constInfo.s2BaseSize = S2_BASE_SIZE;
    constInfo.mBaseSize = constInfo.s1BaseSize * constInfo.gSize;

    // candidate (two-level topk)
    constInfo.candidateMode = tilingData->candidateMode;
    constInfo.candidateTopkBlocks = tilingData->candidateTopkBlocks;
    constInfo.candidateBlockSize = tilingData->candidateBlockSize;
    // 非空 candidate_block_length：按 512 token 候选 tile 执行（metadata 的 s2 区间同口径）
    constInfo.candidateDriven = (tilingData->candidateDriven != 0U);
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::InitBuffers()
{
    if ASCEND_IS_AIV {
        vectorService.InitBuffers(pipe);
    } else {
        matmulService.InitBuffers(pipe);
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::InitActualSeqLen(__gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensK,
                                                            __gm__ uint8_t* sequsedQ, __gm__ uint8_t* sequsedK,
                                                            __gm__ uint8_t* cmpResidualK)
{
    // Q side: cu_seqlens (TND) or seqused (non-TND)
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
        if (cuSeqlensQ != nullptr) {
            constInfo.actualLenQDims = constInfo.batchSize + 1;
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t*)cuSeqlensQ, constInfo.actualLenQDims);
        } else {
            constInfo.actualLenQDims = 0;
        }
    } else {
        if (sequsedQ != nullptr) {
            constInfo.actualLenQDims = constInfo.batchSize;
            actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint32_t*)sequsedQ, constInfo.actualLenQDims);
        } else {
            constInfo.actualLenQDims = 0;
        }
    }
    // K side: cu_seqlens (TND) or seqused (non-TND)
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) {
        if (cuSeqlensK != nullptr) {
            constInfo.actualLenDims = constInfo.batchSize + 1;
            actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t*)cuSeqlensK, constInfo.actualLenDims);
        } else {
            constInfo.actualLenDims = 0;
        }
    } else {
        if (sequsedK != nullptr) {
            constInfo.actualLenDims = constInfo.batchSize;
            actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint32_t*)sequsedK, constInfo.actualLenDims);
        } else {
            constInfo.actualLenDims = 0;
        }
    }
    // cmpResidualK获取
    if (cmpResidualK != nullptr) {
        constInfo.cmpResiduaKLenDims = constInfo.batchSize;
        cmpResidualKGm.SetGlobalBuffer((__gm__ uint32_t*)cmpResidualK, constInfo.batchSize);
    } else {
        constInfo.cmpResiduaKLenDims = 0;
    }
}

template <typename QSLIT>
__aicore__ inline uint32_t QSLIPreload<QSLIT>::GetActualSeqLen(uint32_t bIdx, uint32_t actualLenDims, bool isAccumSeq,
                                                               GlobalTensor<uint32_t>& actualSeqLengthsGm,
                                                               uint32_t defaultSeqLen)
{
    if (actualLenDims == 0) {
        return defaultSeqLen;
    } else if (isAccumSeq) {
        // TND with cu_seqlens: length[i] = cu_seqlens[i+1] - cu_seqlens[i]
        return actualSeqLengthsGm.GetValue(bIdx + 1) - actualSeqLengthsGm.GetValue(bIdx);
    } else {
        // non-TND with seqused: length[i] = seqused[i]
        return actualSeqLengthsGm.GetValue(bIdx);
    }
}

template <typename QSLIT>
__aicore__ inline uint32_t QSLIPreload<QSLIT>::GetActualSeqLenKey(uint32_t bIdx, uint32_t actualLenDims,
                                                                  uint32_t cmpResiduaKLenDims, bool isAccumSeq,
                                                                  GlobalTensor<uint32_t>& actualSeqLengthsGm,
                                                                  GlobalTensor<uint32_t>& cmpResidualKGm,
                                                                  uint32_t defaultSeqLen, uint32_t cmpRatio)
{
    uint32_t cmpResidualK; // 当前bidx对应的cmpResidualK
    if (cmpResiduaKLenDims == 0) {
        cmpResidualK = 0;
    } else {
        cmpResidualK = cmpResidualKGm.GetValue(bIdx);
    }
    if (actualLenDims == 0) {
        return defaultSeqLen * cmpRatio + cmpResidualK;
    } else if (isAccumSeq) {
        // TND with cu_seqlens: length[i] = cu_seqlens[i+1] - cu_seqlens[i]
        return (actualSeqLengthsGm.GetValue(bIdx + 1) - actualSeqLengthsGm.GetValue(bIdx)) * cmpRatio + cmpResidualK;
    } else {
        return (actualSeqLengthsGm.GetValue(bIdx)) * cmpRatio + cmpResidualK;
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::GetS1S2ActualSeqLen(uint32_t bIdx, uint32_t& actS1Size, uint32_t& actS2Size,
                                                               uint32_t& actS2SizeOrig)
{
    actS1Size = GetActualSeqLen(bIdx, constInfo.actualLenQDims, constInfo.isAccumSeqS1, actualSeqLengthsGmQ,
                                constInfo.qSeqSize);
    actS2SizeOrig = GetActualSeqLenKey(bIdx, constInfo.actualLenDims, constInfo.cmpResiduaKLenDims,
                                       constInfo.isAccumSeqS2, actualSeqLengthsGm, cmpResidualKGm, constInfo.kSeqSize,
                                       constInfo.cmpRatio); // 压缩前的actS2Size
    actS2Size = actS2SizeOrig / constInfo.cmpRatio;         // 真实使用的压缩后S2长度
}

template <typename QSLIT>
__aicore__ inline uint32_t QSLIPreload<QSLIT>::GetS2BaseBlockNumOnMask(uint32_t s1gIdx, uint32_t actS1Size,
                                                                       uint32_t actS2SizeOrig, uint32_t& validS2Len)
{
    if (actS2SizeOrig / constInfo.cmpRatio == 0) {
        validS2Len = 0;
        return 0;
    }
    uint32_t s1Offset = constInfo.s1BaseSize * s1gIdx;
    int32_t validS2LenBase =
        static_cast<int32_t>(actS2SizeOrig) - static_cast<int32_t>(actS1Size); // 压缩前的validS2LenBase
    validS2Len = (static_cast<int32_t>(s1Offset) + validS2LenBase + static_cast<int32_t>(constInfo.s1BaseSize)) /
                 static_cast<int32_t>(constInfo.cmpRatio);
    validS2Len = Min(validS2Len, static_cast<int32_t>(actS2SizeOrig) / constInfo.cmpRatio);
    validS2Len = Max(validS2Len, 1);
    return (validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
}

template <typename QSLIT>
__aicore__ inline uint32_t QSLIPreload<QSLIT>::GetTotalBaseBlockNum()
{
    uint32_t totalBlockNum = 0;
    uint32_t actS1Size, actS2Size, actS2SizeOrig;
    uint32_t s1GBaseNum, s2BaseNum;
    uint32_t validS2Len = 0;
    for (uint32_t bIdx = 0; bIdx < constInfo.batchSize; bIdx++) {
        GetS1S2ActualSeqLen(bIdx, actS1Size, actS2Size, actS2SizeOrig);
        s1GBaseNum = CeilDiv(actS1Size, constInfo.s1BaseSize);
        if (!constInfo.attenMaskFlag) {
            s2BaseNum = CeilDiv(actS2Size, constInfo.s2BaseSize);
            totalBlockNum += s1GBaseNum * s2BaseNum * constInfo.kHeadNum;
            continue;
        }
        for (uint32_t s1gIdx = 0; s1gIdx < s1GBaseNum; s1gIdx++) {
            s2BaseNum = GetS2BaseBlockNumOnMask(s1gIdx, actS1Size, actS2SizeOrig, validS2Len);
            totalBlockNum += s2BaseNum * constInfo.kHeadNum;
        }
    }
    return totalBlockNum;
}

// 多核版本，双闭区间。基本原则：计算每个核最少处理的块数,
// 剩余的部分前面的核每个核多处理一块
template <typename QSLIT>
__aicore__ void inline QSLIPreload<QSLIT>::SplitCore()
{
    constInfo.coreEnable = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_CORE_ENABLE_INDEX, false));
    if (aiCoreIdx != 0) {
        constInfo.bN2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_BN2_START_INDEX, false));
        constInfo.gS1Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_M_START_INDEX, false));
        constInfo.s2Start = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_S2_START_INDEX, false));
    }
    constInfo.bN2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_BN2_END_INDEX, false));
    constInfo.gS1End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_M_END_INDEX, false));
    constInfo.s2End = metadataGm.GetValue(GetAttrAbsIndex(aiCoreIdx, QLI_V2_S2_END_INDEX, false));

    // 如果0核都没有启动，说明所有核都没启动
    coreZeroEnable = metadataGm.GetValue(GetAttrAbsIndex(0, QLI_V2_CORE_ENABLE_INDEX, false));

    // LD 第一个workspace的索引
    uint32_t ldFirstWorkSpaceIndex = GetAttrAbsIndex(aiCoreIdx, QLI_V2_FIRST_QLD_V2_DATA_WORKSPACE_IDX_INDEX, false);
    ldInfo.saveWorkSpaceIdx = metadataGm.GetValue(ldFirstWorkSpaceIndex);
    if ASCEND_IS_AIV {
        uint32_t vecCoreIdx = tmpBlockIdx; // 此时代表vectoe核数
        uint32_t ldCoreEnableIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_CORE_ENABLE_INDEX, true);
        ldInfo.isLdCoreEnable = metadataGm.GetValue(ldCoreEnableIndex);

        if (!ldInfo.isLdCoreEnable) {
            return;
        }

        // LD 参数信息
        uint32_t ldBn2IdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_BN2_IDX_INDEX, true);
        uint32_t ldMIdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_IDX_INDEX, true);
        uint32_t ldWorkspaceIdxIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_WORKSPACE_IDX_INDEX, true);
        uint32_t ldWorkspaceNumINDEX = GetAttrAbsIndex(vecCoreIdx, QLD_V2_WORKSPACE_NUM_INDEX, true);
        uint32_t ldMstartIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_START_INDEX, true);
        uint32_t ldMNumIndex = GetAttrAbsIndex(vecCoreIdx, QLD_V2_M_NUM_INDEX, true);

        ldInfo.bn2Idx = metadataGm.GetValue(ldBn2IdxIndex);
        ldInfo.bIdx = ldInfo.bn2Idx / constInfo.kHeadNum;
        ldInfo.n2Idx = ldInfo.bn2Idx % constInfo.kHeadNum;
        ldInfo.mIdx = metadataGm.GetValue(ldMIdxIndex);
        ldInfo.workspaceIdx = metadataGm.GetValue(ldWorkspaceIdxIndex);
        ldInfo.workspaceNum = metadataGm.GetValue(ldWorkspaceNumINDEX);
        ldInfo.mStart = metadataGm.GetValue(ldMstartIndex);
        ldInfo.mNum = metadataGm.GetValue(ldMNumIndex);
        uint64_t actualSeqQPrefixSum = 0;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (ldInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(ldInfo.bIdx);
        } else { // BSND
            actualSeqQPrefixSum = (ldInfo.bIdx <= 0) ? 0 : ldInfo.bIdx * constInfo.qSeqSize;
        }
        // 搬出Topk的初始偏移地址
        ldInfo.indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount +
            static_cast<uint64_t>(ldInfo.n2Idx) * constInfo.sparseCount +
            static_cast<uint64_t>(ldInfo.mIdx) * constInfo.s1BaseSize * constInfo.kHeadNum * constInfo.sparseCount;
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::DealActSeqLenIsZero(uint32_t bIdx, uint32_t n2Idx, uint32_t s1Start)
{
    if ASCEND_IS_AIV {
        if (constInfo.outputLayout == LI_LAYOUT::TND) {
            uint32_t tSize = actualSeqLengthsGmQ.GetValue(constInfo.batchSize);
            uint32_t tBase = bIdx == 0 ? 0 : actualSeqLengthsGmQ.GetValue(bIdx);
            uint32_t s1Count = tempLoopInfo.actS1Size;

            for (uint32_t s1Idx = s1Start; s1Idx < s1Count; s1Idx++) {
                uint64_t indiceOutOffset =
                    (tBase + s1Idx) * constInfo.kHeadNum * constInfo.sparseCount + // T轴、s1轴偏移
                    n2Idx * constInfo.sparseCount;                                 // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        } else if (constInfo.outputLayout == LI_LAYOUT::BSND) {
            for (uint32_t s1Idx = s1Start; s1Idx < constInfo.qSeqSize; s1Idx++) {
                // B,S1,N2,K
                uint64_t indiceOutOffset =
                    static_cast<uint64_t>(bIdx) * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount +
                    static_cast<uint64_t>(s1Idx) * constInfo.kHeadNum * constInfo.sparseCount + // B轴、S1轴偏移
                    static_cast<uint64_t>(n2Idx) * constInfo.sparseCount;                       // N2轴偏移
                vectorService.CleanInvalidOutput(indiceOutOffset);
            }
        }
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* weights, __gm__ uint8_t* queryScale,
    __gm__ uint8_t* keyScale, __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensK, __gm__ uint8_t* sequsedQ,
    __gm__ uint8_t* sequsedK, __gm__ uint8_t* cmpResidualK, __gm__ uint8_t* blockTable, __gm__ uint8_t* outputIdxOffset,
    __gm__ uint8_t* metadata, __gm__ uint8_t* candidateBlockIndices, __gm__ uint8_t* candidateBlockLength,
    __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseValues, __gm__ uint8_t* workspace,
    const QSLITilingDataConsumer* __restrict tiling, TPipe* tPipe)
{
    if ASCEND_IS_AIV {
        tmpBlockIdx = GetBlockIdx(); // vec:0-47
        aiCoreIdx = tmpBlockIdx / 2;
    } else {
        tmpBlockIdx = GetBlockIdx(); // cube:0-23
        aiCoreIdx = tmpBlockIdx;
    }

    InitTilingData(tiling);
    InitActualSeqLen(cuSeqlensQ, cuSeqlensK, sequsedQ, sequsedK, cmpResidualK);

    // 候选输入需在 AIC/AIV 两侧都可见：AIC 用候选块索引做 K 寻址，
    // 两侧的 CalcS2LoopParams 都要读 candidate_block_length 现算任务 tile 上界。
    if (candidateBlockIndices != nullptr) {
        candidateBlockIndicesInGm.SetGlobalBuffer((__gm__ int32_t*)candidateBlockIndices);
    }
    if (candidateBlockLength != nullptr) {
        candidateBlockLengthGm.SetGlobalBuffer((__gm__ int32_t*)candidateBlockLength);
    }

    if (metadata != nullptr) {
        metadataGm.SetGlobalBuffer((__gm__ uint32_t*)metadata);
        // 计算分核
        SplitCore();
    }

    pipe = tPipe;
    // workspace 内存排布
    // |mm1ResGm(存S)
    uint64_t offset = 0;

    // mm1开DoubleBuffer
    GlobalTensor<MM1_OUT_T> mm1ResGm; // 存放S
    uint64_t singleCoreMm1ResSize = WS_DOUBLE * constInfo.s1BaseSize * constInfo.s2BaseSize * sizeof(MM1_OUT_T);
    mm1ResGm.SetGlobalBuffer((__gm__ MM1_OUT_T*)(workspace + aiCoreIdx * singleCoreMm1ResSize));
    offset += GetBlockNum() * singleCoreMm1ResSize;

    GlobalTensor<half> weightWorkspaceGm; // v1阶段处理w*scale后的结果
    uint64_t weightMemSize = BLOCK_CUBE * constInfo.mBaseSize * WS_DOUBLE * sizeof(half);
    weightWorkspaceGm.SetGlobalBuffer((__gm__ half*)(workspace + offset + aiCoreIdx * weightMemSize));
    offset += GetBlockNum() * weightMemSize;

    // ld流程需要ws大小: [aicnum, 2, s1BaseSize, topkOut_*2]
    // (aic, 8, 2, 2, 2048)
    // (aic, s1_cube, 头尾, idx/value, K)
    GlobalTensor<float> vec1ResGm; // 存放TopK计算中间结果
    vec1ResGm.SetGlobalBuffer((__gm__ float*)(workspace + offset));
    offset += GetBlockNum() * constInfo.s1BaseSize * WS_DOUBLE * WS_DOUBLE * BASE_TOPK * sizeof(float);

    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    // 候选驱动三块工作区（仅非空 candidate_block_length 时占用，与 host tiling 同口径）：
    // staging（AIV 写 K、AIC 读）、分数区（AIC 写、AIV 读）、kscale 辅助区（AIV 写、AIV 读）
    GlobalTensor<int8_t> candStagingKeyGm;
    GlobalTensor<float> candScoreGm;
    GlobalTensor<half> candAuxScaleGm;
    if (constInfo.candidateDriven) {
        uint64_t candStagingKeyBytes =
            static_cast<uint64_t>(CAND_RING_SLOTS) * CAND_TILE_TOKENS * static_cast<uint64_t>(constInfo.headDim);
        candStagingKeyGm.SetGlobalBuffer((__gm__ int8_t*)(workspace + offset + aiCoreIdx * candStagingKeyBytes));
        offset += GetBlockNum() * candStagingKeyBytes;
        // 分数区按任务序号奇偶双缓冲：跨任务延迟消费时，下一任务的写不覆盖尚未读的分数
        uint64_t candScoreBytes =
            WS_DOUBLE * static_cast<uint64_t>(CAND_SCORE_TILES_PER_TASK) * CAND_TILE_TOKENS * sizeof(float);
        candScoreGm.SetGlobalBuffer((__gm__ float*)(workspace + offset + aiCoreIdx * candScoreBytes));
        offset += GetBlockNum() * candScoreBytes;
        // aux 区独立 4 周期（见 common 头 CAND_AUX_PERIOD 说明）
        uint64_t candAuxBytes = static_cast<uint64_t>(CAND_AUX_PERIOD) * CAND_SCORE_TILES_PER_TASK *
                                CAND_AUX_SCALE_HALVES_PER_TILE * sizeof(half);
        candAuxScaleGm.SetGlobalBuffer((__gm__ half*)(workspace + offset + aiCoreIdx * candAuxBytes));
        offset += GetBlockNum() * candAuxBytes;
    }
    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, ldInfo, tiling);
        indiceOutGm.SetGlobalBuffer((__gm__ int32_t*)sparseIndices);
        weightsGm.SetGlobalBuffer((__gm__ half*)weights);
        qScaleGm.SetGlobalBuffer((__gm__ half*)queryScale);
        kScaleGm.SetGlobalBuffer((__gm__ half*)keyScale);
        keyGm.SetGlobalBuffer((__gm__ K_T*)key); // 候选 gather 需要 K 源（与 cube 同一 GM 布局）
        blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        // consumer 算子: candidate_block_indices 为 REQUIRED 输入
        candidateBlockIndicesInGm.SetGlobalBuffer((__gm__ int32_t*)candidateBlockIndices);
        if (candidateBlockLength != nullptr) {
            candidateBlockLengthGm.SetGlobalBuffer((__gm__ int32_t*)candidateBlockLength);
        }
        if (outputIdxOffset != nullptr) { // A15: 使能 output_idx_offset (参照 arch35)
            outputIdxOffsetGm.SetGlobalBuffer((__gm__ int32_t*)outputIdxOffset);
        }
        // A15: 有效标志须与 tensor 一并传给 vector (InitParams 已先值拷贝 constInfo, 此处再改标志不生效)
        bool outputIdxOffsetValid = (outputIdxOffset != nullptr);
        vectorService.InitVecInputTensor(weightsGm, qScaleGm, kScaleGm, indiceOutGm, blockTableGm);
        vectorService.InitVecCandidateTensor(candidateBlockIndicesInGm, candidateBlockLengthGm, outputIdxOffsetGm,
                                             outputIdxOffsetValid);
        vectorService.InitVecWorkspaceTensor(weightWorkspaceGm, mm1ResGm, vec1ResGm);
        vectorService.InitCandGatherTensor(keyGm, candStagingKeyGm, candScoreGm, candAuxScaleGm);
    } else {
        matmulService.InitParams(constInfo);
        queryGm.SetGlobalBuffer((__gm__ Q_T*)query);
        if constexpr (PAGE_ATTENTION) {
            blockTableGm.SetGlobalBuffer((__gm__ int32_t*)blockTable);
        }
        keyGm.SetGlobalBuffer((__gm__ K_T*)key);
        matmulService.InitMm1GlobalTensor(blockTableGm, keyGm, queryGm, mm1ResGm, weightWorkspaceGm);
        // 候选驱动: cube 从 staging 连续读 K、分数写候选分数区
        matmulService.InitCandMm1Tensor(candStagingKeyGm, candScoreGm);
    }
    InitBuffers();
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::GetBN2Idx(uint32_t bN2Idx)
{
    tempLoopInfo.bN2Idx = bN2Idx;
    tempLoopInfo.bIdx = bN2Idx / constInfo.kHeadNum;
    tempLoopInfo.n2Idx = bN2Idx % constInfo.kHeadNum;
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::CalcS2LoopParams(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    tempLoopInfo.gS1Idx = gS1LoopIdx;
    tempLoopInfo.actMBaseSize = constInfo.mBaseSize;
    uint32_t remainedGS1Size = tempLoopInfo.actS1Size * constInfo.gSize - tempLoopInfo.gS1Idx * constInfo.mBaseSize;
    if (remainedGS1Size <= constInfo.mBaseSize && remainedGS1Size > 0) {
        tempLoopInfo.actMBaseSize = tempLoopInfo.mBasicSizeTail;
    }

    bool isEnd = (bN2LoopIdx + 1 == constInfo.bN2End) && (gS1LoopIdx + 1 == tempLoopInfo.gS1LoopEnd);
    uint32_t s2BlockNum;
    uint32_t validS2Len = 0;
    if (constInfo.attenMaskFlag) {
        s2BlockNum = GetS2BaseBlockNumOnMask(gS1LoopIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2SizeOrig,
                                             tempLoopInfo.validS2Len);
    } else {
        s2BlockNum = (tempLoopInfo.actS2Size + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
        tempLoopInfo.validS2Len = tempLoopInfo.actS2Size;
    }
    tempLoopInfo.s2LoopEnd = (isEnd && constInfo.s2End != 0) ? constInfo.s2End : s2BlockNum;
    if (constInfo.s2Start > 0 || tempLoopInfo.s2LoopEnd < s2BlockNum) {
        tempLoopInfo.isNeedLD = true;
    } else {
        tempLoopInfo.isNeedLD = false;
    }
    tempLoopInfo.s2BasicSizeTail = tempLoopInfo.validS2Len % constInfo.s2BaseSize;
    tempLoopInfo.s2BasicSizeTail =
        (tempLoopInfo.s2BasicSizeTail == 0) ? constInfo.s2BaseSize : tempLoopInfo.s2BasicSizeTail;

    // 候选驱动：s2 轴语义改为 512 token 候选 tile 索引（与 QSLI metadata 的
    // CandidateTileTokens=512 同口径），循环上界取 metadata 的 s2End（候选 tile 数）。
    // 全量扫的 s2BaseSize=2048 只服务非候选路径，此处不混用。
    if (constInfo.candidateDriven) {
        const uint32_t taskTokens = tempLoopInfo.actMBaseSize / static_cast<uint32_t>(constInfo.gSize);
        const uint32_t rowBase = tempLoopInfo.actS1RowBase + gS1LoopIdx * constInfo.s1BaseSize;
        uint32_t maxL = 0U;
        for (uint32_t t = 0U; t < taskTokens; t++) {
            const int32_t len = candidateBlockLengthGm.GetValue(rowBase + t);
            const uint32_t l = (len > 0) ? static_cast<uint32_t>(len) : 0U;
            maxL = (l > maxL) ? l : maxL;
        }
        if (maxL > constInfo.candidateTopkBlocks) {
            maxL = constInfo.candidateTopkBlocks;
        }
        tempLoopInfo.maxCandLen = maxL;
        tempLoopInfo.maxCandTiles = (maxL * constInfo.candidateBlockSize + CAND_TILE_TOKENS - 1U) / CAND_TILE_TOKENS;
        tempLoopInfo.candDriven = true;
        // L 全 0 的任务 maxCandTiles 保持 0：任务级派发按"空任务"处理（整行输出 -1 或空 partial），
        // 不产生 ring 流量、也不做 W/分数握手。
        tempLoopInfo.s2LoopEnd = (isEnd && constInfo.s2End != 0) ? constInfo.s2End : tempLoopInfo.maxCandTiles;
        if (tempLoopInfo.s2LoopEnd > tempLoopInfo.maxCandTiles) {
            tempLoopInfo.s2LoopEnd = tempLoopInfo.maxCandTiles;
        }
        tempLoopInfo.isNeedLD = (constInfo.s2Start > 0 || tempLoopInfo.s2LoopEnd < tempLoopInfo.maxCandTiles);
        // 分片任务若本核分到的区间为空（该行候选 tile 少于 batch 上界），保留空区间
        // （tileBegin >= tileEnd）：vector 侧按 FD 约定直接溢写空 partial，AIC 不参与，
        // 不需要再伪造一个空 tile。
        tempLoopInfo.s2BasicSizeTail = CAND_TILE_TOKENS;
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::CalcGS1LoopParams(uint32_t bN2LoopIdx)
{
    GetBN2Idx(bN2LoopIdx);
    GetS1S2ActualSeqLen(tempLoopInfo.bIdx, tempLoopInfo.actS1Size, tempLoopInfo.actS2Size, tempLoopInfo.actS2SizeOrig);
    // 候选长度/候选索引的行基址：TND 用 cu_seqlens_q 前缀，BSND 用 bIdx × qSeqSize
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
        tempLoopInfo.actS1RowBase = (tempLoopInfo.bIdx <= 0) ? 0U : actualSeqLengthsGmQ.GetValue(tempLoopInfo.bIdx);
    } else {
        tempLoopInfo.actS1RowBase = tempLoopInfo.bIdx * constInfo.qSeqSize;
    }
    if ((tempLoopInfo.actS2Size == 0) || (tempLoopInfo.actS1Size == 0)) {
        tempLoopInfo.curActSeqLenIsZero = true;
        return;
    }
    tempLoopInfo.curActSeqLenIsZero = false;
    tempLoopInfo.mBasicSizeTail = (tempLoopInfo.actS1Size * constInfo.gSize) % constInfo.mBaseSize;
    tempLoopInfo.mBasicSizeTail =
        (tempLoopInfo.mBasicSizeTail == 0) ? constInfo.mBaseSize : tempLoopInfo.mBasicSizeTail;

    uint32_t gS1SplitNum = (tempLoopInfo.actS1Size * constInfo.gSize + constInfo.mBaseSize - 1) / constInfo.mBaseSize;
    tempLoopInfo.gS1LoopEnd =
        (bN2LoopIdx + 1 == constInfo.bN2End && constInfo.gS1End != 0) ? constInfo.gS1End : gS1SplitNum;
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        if (tempLoopInfo.gS1LoopEnd == gS1SplitNum && constInfo.qSeqSize > tempLoopInfo.actS1Size) {
            tempLoopInfo.needDealActS1LessThanS1 = true;
        }
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::CalcRunInfo(uint32_t loop, uint32_t s2LoopIdx, QSLICommon::RunInfo& runInfo)
{
    runInfo.loop = loop;
    runInfo.bIdx = tempLoopInfo.bIdx;
    runInfo.n2Idx = tempLoopInfo.n2Idx;
    runInfo.gS1Idx = tempLoopInfo.gS1Idx;
    runInfo.s2Idx = s2LoopIdx;
    runInfo.bN2Idx = tempLoopInfo.bN2Idx;
    runInfo.isValid = s2LoopIdx < tempLoopInfo.s2LoopEnd;
    runInfo.isNeedLD = tempLoopInfo.isNeedLD;
    if (runInfo.isNeedLD && s2LoopIdx + 1 == tempLoopInfo.s2LoopEnd) {
        runInfo.saveWorkSpaceIdx = ldInfo.saveWorkSpaceIdx;
        ldInfo.saveWorkSpaceIdx++;
    }

    if (!runInfo.isValid) {
        return;
    }

    runInfo.actS1Size = tempLoopInfo.actS1Size;
    runInfo.actS2Size = tempLoopInfo.actS2Size;
    runInfo.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    // 计算实际基本块size
    runInfo.actMBaseSize = tempLoopInfo.actMBaseSize;
    runInfo.actualSingleProcessSInnerSize = constInfo.s2BaseSize;
    uint32_t s2SplitNum = (tempLoopInfo.validS2Len + constInfo.s2BaseSize - 1) / constInfo.s2BaseSize;
    if (runInfo.s2Idx == s2SplitNum - 1) {
        runInfo.actualSingleProcessSInnerSize = tempLoopInfo.s2BasicSizeTail;
    }
    runInfo.actualSingleProcessSInnerSizeAlign =
        QSLICommon::Align((uint32_t)runInfo.actualSingleProcessSInnerSize, QSLICommon::ConstInfo::BUFFER_SIZE_BYTE_32B);

    runInfo.isFirstS2InnerLoop = s2LoopIdx == constInfo.s2Start;
    runInfo.isLastS2InnerLoop = (s2LoopIdx + 1 == tempLoopInfo.s2LoopEnd);

    if (runInfo.isFirstS2InnerLoop) {
        uint64_t actualSeqQPrefixSum;
        if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(runInfo.bIdx);
        } else { // BSND
            actualSeqQPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.qSeqSize;
        }
        uint64_t tndBIdxOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim;
        // B,S1,N1(N2,G),D
        queryCoreOffset = tndBIdxOffset + runInfo.gS1Idx * constInfo.mBaseSize * constInfo.headDim;
        // B,S1,N1(N2,G)/T,N1(N2,G)
        weightsCoreOffset = actualSeqQPrefixSum * constInfo.qHeadNum + runInfo.n2Idx * constInfo.gSize;
        // B,S1,N2,k/T,N2,k
        indiceOutCoreOffset =
            actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.sparseCount + runInfo.n2Idx * constInfo.sparseCount;
        // candidate_block_indices(输入) 布局 B,S1,N2,candBlocks / T,N2,candBlocks
        candidateOutCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum * constInfo.candidateTopkBlocks +
                                 runInfo.n2Idx * constInfo.candidateTopkBlocks;
        // A15: output_idx_offset 布局 [行, N2] (每行每 key 头一个 int32), batch 级前缀
        outputIdxOffsetCoreOffset = actualSeqQPrefixSum * constInfo.kHeadNum + runInfo.n2Idx;
    }
    uint64_t actualSeqKPrefixSum;
    if constexpr (K_LAYOUT_T == LI_LAYOUT::TND) { // T N2 D, cu_seqlens_k
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : actualSeqLengthsGm.GetValue(runInfo.bIdx);
    } else {
        actualSeqKPrefixSum = (runInfo.bIdx <= 0) ? 0 : runInfo.bIdx * constInfo.kSeqSize;
    }
    uint64_t tndBIdxOffsetForK = actualSeqKPrefixSum * constInfo.kHeadNum * constInfo.headDim;
    keyCoreOffset = tndBIdxOffsetForK + runInfo.s2Idx * constInfo.s2BaseSize * constInfo.kHeadNum * constInfo.headDim;
    keyScaleCoreOffset = (actualSeqKPrefixSum + runInfo.s2Idx * constInfo.s2BaseSize) * constInfo.kHeadNum;
    runInfo.tensorQueryOffset = queryCoreOffset;
    runInfo.tensorKeyOffset = keyCoreOffset;
    runInfo.tensorKeyScaleOffset = keyScaleCoreOffset;
    runInfo.tensorWeightsOffset = weightsCoreOffset;
    runInfo.indiceOutOffset = indiceOutCoreOffset;
    runInfo.candidateOutOffset = candidateOutCoreOffset;
    runInfo.outputIdxOffsetCoreOffset = outputIdxOffsetCoreOffset;
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::Process()
{
    // 没有计算任务，直接清理输出
    if (coreZeroEnable == 0) {
        ProcessInvalid();
        return;
    }
    ProcessMain();

    ProcessDecode();
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::ProcessInvalid()
{
    if ASCEND_IS_AIV {
        uint32_t aivCoreNum = GetBlockNum() * 2; // 2 means c:v = 1:2
        uint64_t totalOutputSize =
            constInfo.batchSize * constInfo.qSeqSize * constInfo.kHeadNum * constInfo.sparseCount;
        uint64_t singleCoreSize =
            QSLICommon::Align((totalOutputSize + aivCoreNum - 1) / aivCoreNum, GM_ALIGN_BYTES / sizeof(OUT_T));
        uint64_t baseSize = tmpBlockIdx * singleCoreSize;
        if (baseSize < totalOutputSize) {
            uint64_t dealSize =
                (baseSize + singleCoreSize <= totalOutputSize) ? singleCoreSize : totalOutputSize - baseSize;
            GlobalTensor<OUT_T> output = indiceOutGm[baseSize];
            AscendC::InitGlobalMemory(output, dealSize, constInfo.INVALID_IDX);
        }
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::ProcessMain()
{
    // 无任务核直接返回
    if (constInfo.coreEnable == 0) {
        return;
    }

    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
        CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
    } else {
        matmulService.AllocEventID();
        CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
        // 候选 ring 初始令牌：FREE 预置 RING_SLOTS 个（单 id 计数信号量，AIV 每 gather 一个
        // tile 消耗 1 个，AIC 每算完 1 个 tile 回补 1 个）；WFREE 预置 2 个（W 双槽）。
        if (constInfo.candidateDriven) {
            for (uint32_t s = 0U; s < CAND_RING_SLOTS; s++) {
                CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(CAND_FREE_FLAG);
            }
            for (uint32_t s = 0U; s < 2U; s++) {
                CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(CAND_WFREE_FLAG);
            }
        }
    }

    QSLICommon::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE];

    // 适配左闭右开
    if (constInfo.bN2Start == constInfo.bN2End) {
        if (constInfo.gS1Start != constInfo.gS1End || constInfo.s2Start != constInfo.s2End) {
            constInfo.bN2End += 1;
            if (constInfo.s2End != 0) {
                constInfo.gS1End += 1;
            }
        }
    } else if ((constInfo.gS1End != 0) || (constInfo.s2End != 0)) {
        constInfo.bN2End += 1;
        if (constInfo.s2End != 0) {
            constInfo.gS1End += 1;
        }
    }

    uint32_t gloop = 0;
    for (uint32_t bN2LoopIdx = constInfo.bN2Start; bN2LoopIdx < constInfo.bN2End; bN2LoopIdx++) {
        CalcGS1LoopParams(bN2LoopIdx);
        if (tempLoopInfo.curActSeqLenIsZero) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, 0U);

            if ASCEND_IS_AIV {
                if (bN2LoopIdx + 1 == constInfo.bN2End && gloop > 0) {
                    CrossCoreWaitFlag(constInfo.syncC1V1);
                    vectorService.ProcessVec1(runInfo[1 - gloop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE]);
                    CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(
                        constInfo.syncV1C1); // 反向同步 1
                }
            }
            continue;
        }
        for (uint32_t gS1LoopIdx = constInfo.gS1Start; gS1LoopIdx < tempLoopInfo.gS1LoopEnd; gS1LoopIdx++) {
            CalcS2LoopParams(bN2LoopIdx, gS1LoopIdx);
            // 候选驱动：任务级派发（AIV 的 gather 与 AIC 的 QK 用 ring 解耦，能跨越 S2 子循环
            // 重叠），不再走下面按 2048 token 全量扫块的双缓冲 ProcessBaseBlock 循环。
            if (tempLoopInfo.candDriven) {
                ProcessCandTask(bN2LoopIdx, gS1LoopIdx);
                constInfo.s2Start = 0;
                continue;
            }
            bool isEnd = (bN2LoopIdx + 1 == constInfo.bN2End) && (gS1LoopIdx + 1 == tempLoopInfo.gS1LoopEnd);
            uint32_t extraLoop = isEnd ? LI_QUANT_PRELOAD_TASK_CACHE_SIZE - 1 : 0; // 只preload一轮

            for (uint32_t s2LoopIdx = constInfo.s2Start; s2LoopIdx < (tempLoopInfo.s2LoopEnd + extraLoop);
                 s2LoopIdx++) {
                ProcessBaseBlock(gloop, s2LoopIdx, runInfo);
                ++gloop;
            }
            constInfo.s2Start = 0;
        }
        if (tempLoopInfo.needDealActS1LessThanS1) {
            DealActSeqLenIsZero(tempLoopInfo.bIdx, tempLoopInfo.n2Idx, tempLoopInfo.actS1Size);
        }
        constInfo.gS1Start = 0;
    }

    if ASCEND_IS_AIV {
        // 核尾排空：候选任务内最后几个 token 的分数消费延迟到这里，先等 AIC 主循环结束
        // 令牌（其分数写已推进/落地），再读分数写输出。
        if (constInfo.candidateDriven) {
            CrossCoreWaitFlag(CAND_AIC_DONE_FLAG);
            vectorService.DrainPendingCandScore();
        }
        vectorService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncC1V0);
        CrossCoreWaitFlag(constInfo.syncC1V0);
    } else {
        matmulService.FreeEventID();
        CrossCoreWaitFlag(constInfo.syncV1C1);
        CrossCoreWaitFlag(constInfo.syncV1C1);
        if (constInfo.candidateDriven) {
            // AIC_DONE 之前必须先等 fixpipe 队列清空：核尾排空读最后一个 tile 的分数，
            // 若分数写未落地，AIV 会读到旧区域数据。
            AscendC::PipeBarrier<PIPE_FIX>();
            CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(CAND_AIC_DONE_FLAG);
        }
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::ProcessBaseBlock(
    uint32_t loop, uint64_t s2LoopIdx, QSLICommon::RunInfo runInfo[LI_QUANT_PRELOAD_TASK_CACHE_SIZE])
{
    int32_t curTaskId = loop % LI_QUANT_PRELOAD_TASK_CACHE_SIZE;
    QSLICommon::RunInfo& curRunInfo = runInfo[curTaskId];
    QSLICommon::RunInfo& lastRunInfo = runInfo[1 - curTaskId];

    CalcRunInfo(loop, s2LoopIdx, curRunInfo);

    if (curRunInfo.isValid) {
        if ASCEND_IS_AIC {
            if (curRunInfo.isFirstS2InnerLoop) {
                CrossCoreWaitFlag(constInfo.syncV0C1);
            }
            CrossCoreWaitFlag(constInfo.syncV1C1); // 反向同步 1
            matmulService.ComputeMm1(curRunInfo);
            CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V1);
            if (curRunInfo.isLastS2InnerLoop) {
                // 反向同步 0
                CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(constInfo.syncC1V0);
            }
        } else {
            if (curRunInfo.isFirstS2InnerLoop) {
                CrossCoreWaitFlag(constInfo.syncC1V0); // 反向同步 0
                vectorService.ProcessVec0(curRunInfo);
                CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV0C1);
            }
        }
    }

    if (lastRunInfo.isValid) {
        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(constInfo.syncC1V1);
            vectorService.ProcessVec1(lastRunInfo);
            CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C1); // 反向同步 1
        }
        lastRunInfo.isValid = false;
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::ProcessCandTask(uint32_t bN2LoopIdx, uint32_t gS1LoopIdx)
{
    (void)bN2LoopIdx;
    const uint32_t taskTokens = tempLoopInfo.actMBaseSize / static_cast<uint32_t>(constInfo.gSize);
    const uint32_t maxTiles = tempLoopInfo.maxCandTiles;
    const uint32_t tileBegin = static_cast<uint32_t>(constInfo.s2Start);
    const uint32_t tileEnd = tempLoopInfo.s2LoopEnd;
    const bool needSpill = tempLoopInfo.isNeedLD;
    // 空任务（行内 L 全 0 或本核分到的分片区间为空）：不做 W/ring/分数握手，
    // 只由 vector 侧输出整行 -1 或溢写空 partial。
    const bool emptyTask = (maxTiles == 0U) || (tileEnd <= tileBegin);

    QSLICommon::RunInfo candRun{};
    candRun.loop = candTaskSeq_; // W 槽 parity（AIC/AIV 用同一任务序号）
    candRun.candidateTaskSeq = candTaskSeq_;
    candTaskSeq_++;
    candRun.bIdx = tempLoopInfo.bIdx;
    candRun.n2Idx = tempLoopInfo.n2Idx;
    candRun.gS1Idx = tempLoopInfo.gS1Idx;
    candRun.actS1Size = tempLoopInfo.actS1Size;
    candRun.actS2Size = tempLoopInfo.actS2Size;
    candRun.actS2SizeOrig = tempLoopInfo.actS2SizeOrig;
    candRun.actMBaseSize = tempLoopInfo.actMBaseSize;
    candRun.actualSingleProcessSInnerSize = CAND_TILE_TOKENS;
    candRun.actualSingleProcessSInnerSizeAlign = CAND_TILE_TOKENS;
    candRun.candidateRowBase = tempLoopInfo.actS1RowBase + gS1LoopIdx * constInfo.s1BaseSize;
    candRun.candidateTileBegin = tileBegin;
    candRun.candidateTileEnd = tileEnd;
    candRun.candidateMaxTiles = maxTiles;
    candRun.candidateNeedSpill = needSpill;
    candRun.candidateWsBase = ldInfo.saveWorkSpaceIdx;
    candRun.isFirstS2InnerLoop = true;
    candRun.isLastS2InnerLoop = true;
    candRun.isValid = true;
    candRun.isNeedLD = needSpill;

    // Q/权重/输出偏移：与全量扫同一套前缀口径（TND 用 cu_seqlens_q，BSND 用 bIdx×qSeqSize）
    uint64_t actualSeqQPrefixSum;
    if constexpr (Q_LAYOUT_T == LI_LAYOUT::TND) {
        actualSeqQPrefixSum = (candRun.bIdx <= 0) ? 0 : actualSeqLengthsGmQ.GetValue(candRun.bIdx);
    } else {
        actualSeqQPrefixSum = (candRun.bIdx <= 0) ? 0 : candRun.bIdx * constInfo.qSeqSize;
    }
    candRun.tensorQueryOffset = actualSeqQPrefixSum * constInfo.qHeadNum * constInfo.headDim +
                                gS1LoopIdx * constInfo.mBaseSize * constInfo.headDim;
    candRun.tensorWeightsOffset = actualSeqQPrefixSum * constInfo.qHeadNum + candRun.n2Idx * constInfo.gSize;
    // 输出偏移只保留 n2 列前缀：candidateRowBase 是全局行号（已含 batch 前缀），
    // 消费侧按 row×sparseCount 逐行叠加，避免重复计入 batch 前缀。
    candRun.indiceOutOffset = static_cast<uint64_t>(candRun.n2Idx) * constInfo.sparseCount;
    candRun.outputIdxOffsetCoreOffset = static_cast<uint64_t>(candRun.n2Idx);

    if ASCEND_IS_AIV {
        if (!emptyTask) {
            // 先回收 W 槽（AIC 最多领先 2 个任务，不会覆盖正在读的槽），再算 w×qscale 并广播
            CrossCoreWaitFlag(CAND_WFREE_FLAG);
            vectorService.ProcessVec0(candRun);
            CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(CAND_WREADY_FLAG);
        }
        vectorService.RunCandTask(candRun);
    } else {
        if (!emptyTask) {
            CrossCoreWaitFlag(CAND_WREADY_FLAG);
            matmulService.ComputeMm1Candidate(candRun, taskTokens, maxTiles, tileBegin, tileEnd);
            CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(CAND_WFREE_FLAG);
        }
    }
    // FD 分片任务消耗一个 LD 工作区槽位（槽位序列须与 metadata 的分片分配顺序一致）
    if (needSpill) {
        ldInfo.saveWorkSpaceIdx++;
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIPreload<QSLIT>::ProcessDecode()
{
    if ASCEND_IS_AIV {
        vectorService.InitLDBuffers(pipe);
        ICachePreLoad(LD_PREFETCH_LEN);
        SyncAll();
        if (ldInfo.isLdCoreEnable) {
            vectorService.ProcessLD();
        }
    }
}

} // namespace QSLIKernel
#endif // QUANT_SPARSE_LIGHTNING_INDEXER_KERNEL_H
