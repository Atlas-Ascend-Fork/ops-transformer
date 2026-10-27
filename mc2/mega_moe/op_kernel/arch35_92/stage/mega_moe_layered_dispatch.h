/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_DISPATCH_H
#define MEGA_MOE_LAYERED_DISPATCH_H

#include "adv_api/transpose/confusion_transpose.h"

#include "../mega_moe_vf/mega_moe_layered_vf.h"

// Internal member definitions; included by mega_moe_layered.h after the class declaration.
namespace MegaMoeImpl {

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitDispatchRounds(uint32_t fixedBefore,
                                                                                          uint32_t dispatchFixedCost)
{
    uint32_t dispatchAvailUb = 0U;
    if (COMM_WORK_UB_LIMIT > fixedBefore && COMM_WORK_UB_LIMIT - fixedBefore > dispatchFixedCost) {
        dispatchAvailUb = COMM_WORK_UB_LIMIT - fixedBefore - dispatchFixedCost;
    }
    uint32_t maxDispatchRoundSize = dispatchAvailUb * 8U / 65U;
    maxDispatchRoundSize = (maxDispatchRoundSize / 256U) * 256U;
    if (maxDispatchRoundSize == 0U) {
        maxDispatchRoundSize = 256U;
    }
    if (maskRouteCapacity_ <= static_cast<uint64_t>(maxDispatchRoundSize)) {
        dispatchRoundSendTotalNum_ = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<int64_t>(maskRouteCapacity_), static_cast<int64_t>(ALIGN_256)));
    } else {
        uint32_t minDispatchRounds = static_cast<uint32_t>(
            Ops::Base::CeilDiv(static_cast<int64_t>(maskRouteCapacity_), static_cast<int64_t>(maxDispatchRoundSize)));
        uint32_t evenDispatchRoundSize = static_cast<uint32_t>(Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(static_cast<int64_t>(maskRouteCapacity_), static_cast<int64_t>(minDispatchRounds)),
            static_cast<int64_t>(ALIGN_256)));
        dispatchRoundSendTotalNum_ =
            (evenDispatchRoundSize <= maxDispatchRoundSize) ? evenDispatchRoundSize : maxDispatchRoundSize;
    }
    if (dispatchRoundSendTotalNum_ == 0U) {
        dispatchRoundSendTotalNum_ = 256U;
    }
    dispatchTotalRounds_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maskRouteCapacity_), static_cast<int64_t>(dispatchRoundSendTotalNum_)));
    dispatchRoundMaskAlignSize_ = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(dispatchRoundSendTotalNum_) / 8, static_cast<int64_t>(ALIGN_32)));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitDispatchIndexBuffers(
    uint32_t fixedBefore)
{
    uint32_t validTopkIndexTensorAddr = fixedBefore;
    uint32_t validTopkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(dispatchRoundSendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    validTopkIndexTensor_ = reinterpret_cast<__ubuf__ int32_t*>(validTopkIndexTensorAddr);
    uint32_t topkIndexTensorAddr = validTopkIndexTensorAddr + validTopkIndexTensorSize;
    uint32_t topkIndexTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(dispatchRoundSendTotalNum_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    topkIndexTensor_ = reinterpret_cast<__ubuf__ int32_t*>(topkIndexTensorAddr);
    // gatherMaskTensor_ 仅容纳单轮 mask 位。
    uint32_t gatherMaskTensorAddr = topkIndexTensorAddr + topkIndexTensorSize;
    uint32_t gatherMaskTensorSize = dispatchRoundMaskAlignSize_;
    gatherMaskTensor_ = reinterpret_cast<__ubuf__ uint8_t*>(gatherMaskTensorAddr);
    gatherMaskInt32Tensor_ = reinterpret_cast<__ubuf__ uint32_t*>(gatherMaskTensorAddr);
    return gatherMaskTensorAddr + gatherMaskTensorSize;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitDispatchCopyBuffers(
    uint32_t copyTmpBaseAddr, uint32_t relayFlagTensorSize, uint32_t dispatchWorkTensorSize,
    uint32_t expertTokenNumsOutTensorSize)
{
    uint32_t tokenScaleSize = mxQuantTokenScaleAlignBytes_;
    uint32_t copyTmpBufferSize = tokenScaleSize;
    for (int32_t index = 0; index < DISPATCH_COPY_BUFFER_COUNT; ++index) {
        copyTmpTensors_[index] = reinterpret_cast<__ubuf__ ActivationType*>(
            copyTmpBaseAddr + static_cast<uint32_t>(index) * copyTmpBufferSize);
    }
    relayFlagTensor_ = reinterpret_cast<__ubuf__ uint64_t*>(copyTmpBaseAddr);
    // 中继标志快照和本地搬运缓冲的生命期不重叠，因此复用同一段 UB。
    // 一级量化已经在 Wave 前完成，Dispatch 阶段不再保留旧的单 token 量化和稀疏槽位 UB。
    uint32_t dispatchWorkTensorEnd = copyTmpBaseAddr + dispatchWorkTensorSize;
    // 整表建表和 ExpertTokenNumCopyOut 共用每专家一个计数的暂存区。
    // Tensor大小：moeExpertPerRank_ * sizeof(int32_t) 对齐至32字节；
    uint32_t expertTokenNumsOutTensorAddr = dispatchWorkTensorEnd;
    expertTokenNumsOutTensor_ = reinterpret_cast<__ubuf__ int32_t*>(expertTokenNumsOutTensorAddr);
    // 后续 Dispatch 接收从该地址申请 metadata 临时区。
    ubBufferUsedAddr_ = expertTokenNumsOutTensorAddr + expertTokenNumsOutTensorSize;
}

// =================================================================================================
// DispatchBuffInit：Dispatch 接收、计数发布和 ExpertTokenNumCopyOut 使用的 buffer。
// =================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    // Dispatch 阶段 UB 布局：Hcomm 私有区、前缀和、分轮索引/掩码、
    // 中继标志快照/本地搬运缓冲、专家计数输出和元数据区依次排列。
    // 按最大 256 行元数据预留后，通信工作区不得越过 184 KiB 处的 Hcomm WQE 专用区。
    hcommBatchWqeTensor_ = reinterpret_cast<__ubuf__ uint8_t*>(LAYERED_HCOMM_BATCH_UB_OFFSET);
    uint32_t cumsumInfoTensorAddr = ALIGN_512;
    uint32_t cumsumInfoTensorSize = Ops::Base::CeilAlign(
        static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    cumsumInfoTensor_ = reinterpret_cast<__ubuf__ int32_t*>(cumsumInfoTensorAddr);
    // 分轮次参数计算：validTopkIndexTensor_ + topkIndexTensor_ + gatherMaskTensor_ 均为 round-sized
    // 专家计数直接从 GM 读取，不再加载完整 mask slot 到 UB。
    // round 张量: validTopkIndex(R*4) + topkIndex(R*4) + gatherMask(R/8) = 8R + R/8 = 65R/8
    // copyTmpTensors_ 与 relayFlagTensor_ 复用地址，只计两者中的较大值；此外预留专家计数输出和
    // 256 行 metadata。DispatchPrepare 使用的量化缓冲生命周期不重叠，不计入本阶段。
    uint32_t copyTmpTotalSize = static_cast<uint32_t>(DISPATCH_COPY_BUFFER_COUNT) * mxQuantTokenScaleAlignBytes_;
    uint32_t relayFlagTensorSize =
        Ops::Base::CeilAlign(static_cast<uint32_t>(DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY * sizeof(uint64_t)),
                             static_cast<uint32_t>(ALIGN_32));
    uint32_t expertTokenNumsOutTensorSize =
        Ops::Base::CeilAlign(static_cast<int64_t>(moeExpertPerRank_ * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32));
    uint32_t dispatchWorkTensorSize = relayFlagTensorSize > copyTmpTotalSize ? relayFlagTensorSize : copyTmpTotalSize;
    uint32_t dispatchFixedCost = dispatchWorkTensorSize + expertTokenNumsOutTensorSize + L1_TILE_M_256 * ALIGN_32;
    uint32_t fixedBefore = cumsumInfoTensorAddr + cumsumInfoTensorSize;
    InitDispatchRounds(fixedBefore, dispatchFixedCost);
    uint32_t copyTmpBaseAddr = InitDispatchIndexBuffers(fixedBefore);
    InitDispatchCopyBuffers(copyTmpBaseAddr, relayFlagTensorSize, dispatchWorkTensorSize, expertTokenNumsOutTensorSize);
    {
        LocalTensor<int32_t> values(TPosition::VECCALC, reinterpret_cast<uint64_t>(cumsumInfoTensor_),
                                    (cumsumInfoTensorSize / sizeof(int32_t)));
        Duplicate<int32_t>(values, 0, (cumsumInfoTensorSize / sizeof(int32_t)));
    }
    PipeBarrier<PIPE_ALL>();
}

// =================================================================================================
// DispatchPrepareBuffInit：一级 Dispatch 前置量化使用的双缓冲。
// SendMask/reset 和本阶段之间已有全核同步，Wave 内 DispatchBuffInit 会在前置阶段结束后复用同一 UB。
// =================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchPrepareBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }

    uint32_t xInTensorSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    uint32_t xInTensorAddr1 = ALIGN_512;
    uint32_t xInTensorAddr2 = xInTensorAddr1 + xInTensorSize;
    xInTensor1_ = reinterpret_cast<__ubuf__ bfloat16_t*>(xInTensorAddr1);
    xInTensor2_ = reinterpret_cast<__ubuf__ bfloat16_t*>(xInTensorAddr2);

    uint32_t mxTempTensorAddr = xInTensorAddr2 + xInTensorSize;
    mxTempTensor_ = reinterpret_cast<__ubuf__ uint16_t*>(mxTempTensorAddr);

    uint32_t xOutTensorSize = mxQuantTokenScaleAlignBytes_;
    uint32_t xOutTensorAddr1 = mxTempTensorAddr + MX_QUANT_TEMP_UB_BYTES;
    uint32_t xOutTensorAddr2 = xOutTensorAddr1 + xOutTensorSize;
    xOutTensor1_ = reinterpret_cast<__ubuf__ ActivationType*>(xOutTensorAddr1);
    xOutTensor2_ = reinterpret_cast<__ubuf__ ActivationType*>(xOutTensorAddr2);

    uint32_t readyFlagTensorAddr = xOutTensorAddr2 + xOutTensorSize;
    relayReadyFillTensor_ = reinterpret_cast<__ubuf__ uint64_t*>(readyFlagTensorAddr);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitResetBatch()
{
    const auto& workspace = params_.workspace.layout;
    // 工作区统一管理连续排布的 Activation、Dispatch、计数和 Combine 标志。
    uint64_t totalFlagInt32 = (workspace.reset.flags.sizeBytes / SIZE_INT_32);
    int64_t tokenGroupResetSize = static_cast<int64_t>(moeExpertPerRank_) * blockAivNum_ * INT_CACHELINE;
    totalFlagInt32 = (static_cast<int64_t>(totalFlagInt32) > tokenGroupResetSize) ?
                         static_cast<int64_t>(totalFlagInt32) :
                         tokenGroupResetSize;
    uint32_t resetNumPerCore = Ops::Base::CeilDiv(totalFlagInt32, static_cast<uint64_t>(blockAivNum_));
    resetBatchElementCount_ = resetNumPerCore < static_cast<uint32_t>(DISPATCH_RESET_BATCH) ?
                                  static_cast<int32_t>(resetNumPerCore) :
                                  DISPATCH_RESET_BATCH;
    uint32_t resetTensorSize =
        Ops::Base::CeilAlign(static_cast<uint64_t>(resetBatchElementCount_), static_cast<uint64_t>(INT32_PER_256B)) *
        sizeof(int32_t);
    resetBatchElementCount_ = resetTensorSize / sizeof(int32_t);
    return resetTensorSize;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitSendMaskRounds(uint32_t resetTensorSize)
{
    // 分轮次参数计算：UB 预算扣除固定开销后，求解最大 roundSendTotalNum_
    // round 张量: topkIds(R*4) + 2*maskSlot(CeilAlign(R/8,32)+32) + gatherOut(R*4)
    //            = 8R + R/4 + 64  (R 为 256 对齐时 CeilAlign(R/8,32)=R/8)
    // 解: R <= (available - 64) * 4 / 33，再 floor-align 到 256（保证 GM 写 32B 对齐）
    uint32_t fixedUbCosts = ALIGN_512 + resetTensorSize;
    uint32_t availableUb = COMM_WORK_UB_LIMIT - fixedUbCosts;
    uint32_t maxRoundSize = (availableUb - 64U) * 4U / 33U;
    if (maxRoundSize == 0) {
        maxRoundSize = 256U;
    }
    maxRoundSize = (maxRoundSize / 256U) * 256U;
    if (static_cast<uint64_t>(sendTotalNum_) <= static_cast<uint64_t>(maxRoundSize)) {
        roundSendTotalNum_ = static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(ALIGN_256)));
    } else {
        uint32_t minRounds = static_cast<uint32_t>(
            Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(maxRoundSize)));
        uint32_t evenRoundSize = static_cast<uint32_t>(Ops::Base::CeilAlign(
            Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(minRounds)),
            static_cast<int64_t>(ALIGN_256)));
        roundSendTotalNum_ = (evenRoundSize <= maxRoundSize) ? evenRoundSize : maxRoundSize;
    }
    if (roundSendTotalNum_ == 0) {
        roundSendTotalNum_ = 256U;
    }
    roundCompareCount_ = roundSendTotalNum_; // 16K
    roundMaskAlignSize_ = static_cast<uint32_t>(
        Ops::Base::CeilAlign(static_cast<int64_t>(roundCompareCount_) / 8, static_cast<int64_t>(ALIGN_32)));
    roundMaskSlotSize_ = roundMaskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    totalRounds_ = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(sendTotalNum_), static_cast<int64_t>(roundSendTotalNum_)));
}

// ======================================================================================
// SendAndQuantBuffInit：SendMaskCal & ResetFlagList localTensor申请
// --------------------------------------------------------------------------------------
//   大 bs 场景下 sendTotalNum 可能超出 UB 256KB 限制，因此按 roundSendTotalNum_ 分轮次
//   分配 UB 张量（topkIds / sendMask / sendGatherOut），每轮仅加载一部分 topkIds，
//   在 UB 中计算部分 mask 后拼写到 workspace 对应偏移，最终合并为完整 mask slot。
// ======================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendAndQuantBuffInit()
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    // 输入准备与 DispatchBuffInit 生命期不重叠，可复用同一 UB 地址。
    // 减去 Hcomm 和分批清零区后，按 8R + R/4 + 64 字节计算单轮容量，避免清零区随 GM 总量增长。
    LocalTensor<uint8_t> hcommTensor(TPosition::VECCALC, 0, ALIGN_512);
    comm_.Init(hcommTensor, ALIGN_512);
    hcommBatchWqeTensor_ = reinterpret_cast<__ubuf__ uint8_t*>(LAYERED_HCOMM_BATCH_UB_OFFSET);

    uint32_t resetTensorSize = InitResetBatch();
    InitSendMaskRounds(resetTensorSize);
    // Tensor用处：SendMaskCal 每轮搬运本卡 topkIds 的一个子段；
    // Tensor大小：roundCompareCount_ 个 int32（256B 对齐）；
    uint32_t topkIdsTensorAddr = ALIGN_512;
    uint32_t topkIdsTensorSize = roundCompareCount_ * sizeof(int32_t);
    topkIdsTensor_ = reinterpret_cast<__ubuf__ int32_t*>(topkIdsTensorAddr);

    uint32_t resetTensorAddr = topkIdsTensorAddr + topkIdsTensorSize;
    resetTensor_ = reinterpret_cast<__ubuf__ int32_t*>(resetTensorAddr);
    {
        LocalTensor<int32_t> values(TPosition::VECCALC, reinterpret_cast<uint64_t>(resetTensor_),
                                    (resetTensorSize / sizeof(int32_t)));
        Duplicate<int32_t>(values, 0, (resetTensorSize / sizeof(int32_t)));
    }
    // Tensor用处：SendMaskCal 每轮存储部分 mask 位；DOUBLE_BUFFER 双缓冲；
    // Tensor大小：roundMaskSlotSize_（每轮部分 [mask|count] 槽位大小）；
    uint32_t sendMaskAddr = resetTensorAddr + resetTensorSize;
    for (int32_t index = 0; index < DOUBLE_BUFFER; ++index) {
        sendMaskTensor_[index] =
            reinterpret_cast<__ubuf__ uint8_t*>(sendMaskAddr + static_cast<uint32_t>(index) * roundMaskSlotSize_);
    }

    // Tensor用处：SendMaskCal 每轮 GatherMask 计 count 的废弃输出 scratch；
    // Tensor大小：roundCompareCount_ 个 int32（256B 对齐）；
    uint32_t sendGatherOutAddr = sendMaskAddr + static_cast<uint32_t>(DOUBLE_BUFFER) * roundMaskSlotSize_;
    uint32_t sendGatherOutSize = roundCompareCount_ * sizeof(int32_t);
    sendGatherOutTensor_ = reinterpret_cast<__ubuf__ int32_t*>(sendGatherOutAddr);
}

// ===============================================================================================
// ResetFlagList：清理本卡 workspace 中连续排布的 GMM/dispatch flag 和 AIC-AIV1 ready sequence。
// ===============================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetFlagList()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    if constexpr (g_coreType == AIC) {
        return;
    }
    AivJobContext job{aivCoreIdx_, blockAivNum_};
    PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
    ResetWorkspaceRegion<1>(job, workspace.sync.activationReady.get_ptr(buffer),
                            (workspace.reset.flags.sizeBytes / SIZE_INT_32), resetBatchElementCount_, resetTensor_);
    if constexpr (TopkWeightsPrefetch) {
        int32_t statusElementCount = (static_cast<int32_t>(moeExpertPerRank_) *
                                          static_cast<int32_t>(params_.tilingData->compute.maxTilesPerExpert) +
                                      1) *
                                     INT_CACHELINE;
        ResetWorkspaceRegion(GetBalancedWorkRange(static_cast<uint32_t>(statusElementCount), job),
                             workspace.sync.gmm1Tiles.get_ptr(buffer), resetBatchElementCount_, resetTensor_);
    }
}

// ==================================================
// ExpertTokenNumCopyOut：输出本卡各 MoE 专家收到的 token 总数（不包含共享专家）。
// ==================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ExpertTokenNumCopyOut()
{
    // 归属核 Wave 在 Dispatch、GMM 和 Combine 之间复用 AIV1 UB，输出前必须恢复完整前缀和。
    CopyGmToUb(cumsumInfoTensor_, cumsumGm_, static_cast<uint32_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)),
               GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
    PipeNotify<PIPE_MTE2, PIPE_S>(0);
    PipeWait<PIPE_MTE2, PIPE_S>(0);
    int32_t lastRankIdx = static_cast<int32_t>(worldSize_ - 1);
    expertTokenNumsOutTensor_[0] = cumsumInfoTensor_[lastRankIdx];
    for (int32_t expertIdx = 1; expertIdx < moeExpertPerRank_; expertIdx++) {
        int32_t cur = cumsumInfoTensor_[expertIdx * static_cast<int32_t>(worldSize_) + lastRankIdx];
        int32_t prev = cumsumInfoTensor_[(expertIdx - 1) * static_cast<int32_t>(worldSize_) + lastRankIdx];
        expertTokenNumsOutTensor_[expertIdx] = cur - prev;
    }
    PipeFence<PIPE_S, PIPE_MTE3, SYNC_EVENT_ID2>();

    CopyUbToGm(expertTokenNumsOut_, expertTokenNumsOutTensor_,
               static_cast<uint32_t>(moeExpertPerRank_ * sizeof(int32_t)));
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(0);
    PipeWait<PIPE_MTE3, PIPE_MTE2>(0);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildExpertMaskRound(
    int32_t curExpertId, uint32_t roundIdx, uint64_t roundStart, uint32_t roundLen, uint32_t curRoundCompareCount,
    uint64_t& totalSendCnt)
{
    // Phase 1: 加载本轮 topkIds 子段；用非法 expert id 填充对齐尾部。
    {
        LocalTensor<int32_t> values(TPosition::VECCALC, reinterpret_cast<uint64_t>(topkIdsTensor_), roundCompareCount_);
        Duplicate<int32_t>(values, -1, roundCompareCount_);
    }
    PipeFence<PIPE_V, PIPE_MTE2, SYNC_EVENT_ID1>();
    __gm__ int32_t* roundSrcGlobal =
        (__gm__ int32_t*)(params_.expertIdxGmAddr + static_cast<uint64_t>(roundStart) * sizeof(int32_t));

    CopyGmToUb(topkIdsTensor_, roundSrcGlobal, static_cast<uint32_t>(roundLen * sizeof(int32_t)));
    PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID1>();

    // Phase 2: CompareScalar → 本轮部分 mask + GatherMask 累加 count
    __ubuf__ uint8_t* maskBuf = sendMaskTensor_[roundIdx % DOUBLE_BUFFER];
    LocalTensor<uint8_t> mask(TPosition::VECCALC, reinterpret_cast<uint64_t>(maskBuf), roundMaskSlotSize_);
    LocalTensor<int32_t> ids(TPosition::VECCALC, reinterpret_cast<uint64_t>(topkIdsTensor_), roundCompareCount_);
    LocalTensor<int32_t> selected(TPosition::VECCALC, reinterpret_cast<uint64_t>(sendGatherOutTensor_),
                                  roundCompareCount_);
    auto maskPattern = mask.template ReinterpretCast<uint32_t>();
    CompareScalar(mask, ids, curExpertId, AscendC::CMPMODE::EQ, curRoundCompareCount);
    uint64_t roundSendCnt = 0;
    GatherMask(selected, ids, maskPattern, true, static_cast<uint32_t>(roundLen), {1, 1, 0, 0}, roundSendCnt);
    PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID2>();
    totalSendCnt += roundSendCnt;
    PipeFence<PIPE_S, PIPE_MTE3, SYNC_EVENT_ID3>();
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildExpertMask(uint32_t curRankId,
                                                                                           int32_t curExpertId,
                                                                                           uint64_t srcOffset,
                                                                                           uint64_t dstOffset)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    uint64_t totalSendCnt = 0;
    for (uint32_t roundIdx = 0; roundIdx < totalRounds_; ++roundIdx) {
        uint64_t roundStart = static_cast<uint64_t>(roundIdx) * static_cast<uint64_t>(roundSendTotalNum_);
        uint64_t roundLen64 =
            (roundIdx + 1 < totalRounds_) ? static_cast<uint64_t>(roundSendTotalNum_) : sendTotalNum_ - roundStart;
        uint32_t roundLen = static_cast<uint32_t>(roundLen64);
        uint32_t curRoundCompareCount =
            (roundLen == roundSendTotalNum_) ?
                roundCompareCount_ :
                static_cast<uint32_t>(
                    Ops::Base::CeilAlign(static_cast<int64_t>(roundLen) * static_cast<int64_t>(sizeof(int32_t)),
                                         static_cast<int64_t>(ALIGN_256)) /
                    static_cast<int64_t>(sizeof(int32_t)));
        uint32_t curRoundMaskAlignSize =
            (roundLen == roundSendTotalNum_) ?
                roundMaskAlignSize_ :
                static_cast<uint32_t>(Ops::Base::CeilAlign(static_cast<int64_t>(curRoundCompareCount) / 8,
                                                           static_cast<int64_t>(ALIGN_32)));

        BuildExpertMaskRound(curExpertId, roundIdx, roundStart, roundLen, curRoundCompareCount, totalSendCnt);
        __ubuf__ uint8_t* maskBuf = sendMaskTensor_[roundIdx % DOUBLE_BUFFER];
        // Phase 2b: 将本轮部分 mask 写入 workspace 或 local win 对应字节偏移
        // roundByteOffset = roundStart/8，因 roundSendTotalNum_ 为 256 对齐，该偏移 32B 对齐
        uint64_t roundByteOffset = static_cast<uint64_t>(roundStart) / 8U;
        uint32_t writeBytes =
            (static_cast<uint64_t>(curRoundMaskAlignSize) <= static_cast<uint64_t>(maskAlignSize_) - roundByteOffset) ?
                curRoundMaskAlignSize :
                static_cast<uint32_t>(static_cast<uint64_t>(maskAlignSize_) - roundByteOffset);

        if (curRankId == rankId_) {
            __gm__ uint8_t* winDstGlobal =
                (__gm__ uint8_t*)(GetRankWinAddrWithOffset(rankId_, dstOffset + roundByteOffset));
            CopyUbToGm(winDstGlobal, maskBuf, writeBytes);
        } else {
            __gm__ uint8_t* wsDstGlobal =
                (__gm__ uint8_t*)(workspace.urma.mask.get_ptr(buffer) + srcOffset + roundByteOffset);
            CopyUbToGm(wsDstGlobal, maskBuf, writeBytes);
        }
        PipeBarrier<PIPE_ALL>();
    }
    return totalSendCnt;
}

// ======================================================================================================
// SendMaskCal：对本卡 topk 按通信域内所有专家id计算mask位，并发送至目标专家卡
// ------------------------------------------------------------------------------------------------------
//   大 bs 场景下 sendTotalNum 可能超出 UB 限制，因此按 roundSendTotalNum_ 分轮次计算：
//   Phase 1: 对每个全局专家，逐轮加载 topkIds 子段 → CompareScalar → 部分 mask；
//   Phase 2: 每轮部分 mask 拼写到 workspace/win 对应字节偏移，GatherMask 累加本轮 count；
//   Phase 3: 所有轮次完成后，写入 total count，URMA 发送完整 mask slot。
// ======================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendMaskCal()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t totalExperts = static_cast<int32_t>(worldSize_);
    // 按卡处理，只使用前 worldSize 个核。
    for (uint32_t curRankId = aivCoreIdx_; curRankId < static_cast<uint32_t>(totalExperts); curRankId += blockAivNum_) {
        Channel maskChannel{};
        if (curRankId != rankId_) {
            maskChannel = comm_.GetChannel(curRankId);
            // SendMask 与紧随其后的第一次全卡同步由同一物理 AIV 负责，锁跨两个阶段复用，
            // 避免在中间没有换核时提前 Unlock 刷新通道 counter。
            comm_.Lock(maskChannel);
        }
        for (uint32_t expertIdIndex = 0; expertIdIndex < moeExpertPerRank_; ++expertIdIndex) { // 按专家处理
            int32_t curExpertId = static_cast<int32_t>(curRankId * moeExpertPerRank_ + expertIdIndex);
            uint64_t srcOffset = static_cast<uint64_t>(expertIdIndex * static_cast<int32_t>(worldSize_) +
                                                       static_cast<int32_t>(curRankId)) *
                                 static_cast<uint64_t>(maskSlotSize_);
            uint64_t dstOffset =
                peermem.maskRecv.offsetBytes + static_cast<uint64_t>(expertIdIndex * static_cast<int32_t>(worldSize_) +
                                                                     static_cast<int32_t>(rankId_)) *
                                                   static_cast<uint64_t>(maskSlotSize_);
            uint64_t totalSendCnt = BuildExpertMask(curRankId, curExpertId, srcOffset, dstOffset);
            // Phase 3: 写入 total count 并发送完整 mask slot
            if (curRankId == rankId_) {
                __gm__ int32_t* winCountPtr =
                    reinterpret_cast<__gm__ int32_t*>(GetRankWinAddrWithOffset(rankId_, dstOffset + maskAlignSize_));
                GmWrite(winCountPtr, static_cast<int32_t>(totalSendCnt));
                PipeBarrier<PIPE_ALL>();
            } else {
                __gm__ int32_t* wsCountPtr =
                    reinterpret_cast<__gm__ int32_t*>(workspace.urma.mask.get_ptr(buffer) + srcOffset + maskAlignSize_);
                GmWrite(wsCountPtr, static_cast<int32_t>(totalSendCnt));
                PipeBarrier<PIPE_ALL>();
                // E_dsburma2: WriteNbi 的 DMA 引擎从 maskSlot 直读（含尾部 count），旁路写必须先 DSB 排空，
                // 否则远端专家计数表可能收到旧 count，等待目标与实际 token 数错位即挂死/丢数。
                GmDataBarrier();
                GM_ADDR remoteDataAddr = GetRankWinAddrWithOffset(curRankId, dstOffset);
                GM_ADDR localGmAddr = workspace.urma.mask.get_ptr(buffer) + srcOffset;
                comm_.WriteNbi(maskChannel, remoteDataAddr, localGmAddr, maskSlotSize_);
            }
        }
    }
    // SendMask 使用 direct WriteNbi，不创建批量 handle。先让本 AIV 负责的所有远端通道
    // 都拥有在途请求，再统一 Drain；锁继续保留给紧随其后的 CrossRankSync 复用。
    for (uint32_t curRankId = aivCoreIdx_; curRankId < static_cast<uint32_t>(totalExperts); curRankId += blockAivNum_) {
        if (curRankId == rankId_) {
            continue;
        }
        Channel maskChannel = comm_.GetChannel(curRankId);
        comm_.Drain(maskChannel);
    }
}

// ======================================================================
// LoadTopkWeightsToUb：prefetch 或 Server Combine 携带权重，否则仅做 MTE2_V 同步
// ======================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadTopkWeightsToUb(
    __ubuf__ ActivationType* xOutTensor, int32_t currentOffset, int32_t index, TEventID event)
{
    if (RecordHasWeights()) {
        QuantProcessConfig config{mxQuantTokenAlignBytes_, mxQuantScaleAlignBytes_, mxQuantTokenScaleAlignBytes_,
                                  mxQuantScaleNumAlignPerToken_};
        QuantProcessScratch<ActivationType> scratch{};
        scratch.mxTempTensor = mxTempTensor_;
        uint32_t tokenIndex = currentOffset + index;
        GM_ADDR tokenTopkWeightsAddr =
            params_.probsGmAddr + static_cast<uint64_t>(tokenIndex) * topK_ * sizeof(TopkWeightsType);
        PrefetchTopkWeights<TopkWeightsType>(tokenTopkWeightsAddr, topK_, config, scratch, xOutTensor, event);
    } else {
        // Without weight prefetch, this event still waits for the input token copy.
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::DispatchRelayQueueServerOffset(
    uint32_t targetServer) const
{
    return static_cast<uint64_t>(targetServer) * dispatchRelayQueueBytesPerServer_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RelayTokenOffset(uint32_t sourceServer,
                                                                                            uint32_t tokenId) const
{
    return (static_cast<uint64_t>(sourceServer) *
                static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank) +
            tokenId) *
           static_cast<uint64_t>(relayRecordBytes_);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RelayFlagOffset(uint32_t sourceServer,
                                                                                           uint32_t tokenId) const
{
    if (IsCombineServerDedupEnabled()) {
        return sourceServer * CalcServerCombineReadyRowBytes(params_.tilingData->compute.numMaxTokensPerRank) +
               static_cast<uint64_t>(tokenId) * sizeof(uint64_t);
    }
    return (static_cast<uint64_t>(sourceServer) *
                static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank) +
            tokenId) *
           sizeof(uint64_t);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ResetCommunicationState()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    if constexpr (g_coreType == AIC) {
        return;
    }

    // 同一 AIV owner 清零目标 Server 的队列 count 和 Combine 列表行头。
    const int64_t tokenListStride =
        CalcServerCombineTokenListStrideBytes(params_.tilingData->compute.numMaxTokensPerRank);
    for (uint32_t targetServer = aivCoreIdx_; targetServer < serverNum_; targetServer += blockAivNum_) {
        auto countPtr = reinterpret_cast<__gm__ int32_t*>(workspace.urma.relaySendQueue.get_ptr(buffer) +
                                                          DispatchRelayQueueServerOffset(targetServer));
        GmWrite(countPtr, int32_t(0));
        if (IsCombineServerDedupEnabled()) {
            auto listCount = reinterpret_cast<__gm__ int32_t*>(workspace.urma.serverCombine.tokenIds.get_ptr(buffer) +
                                                               targetServer * tokenListStride);
            GmWrite(listCount, int32_t(0));
        }
    }
    const AivJobContext job{aivCoreIdx_, blockAivNum_};
    // CCL ready 区连续清零；普通模式的两个 Combine ready 区大小为 0。
    const auto readyBytes =
        peermem.dispatchFlag.sizeBytes + peermem.sourceReady.sizeBytes + peermem.contributionReady.sizeBytes;
    const auto readyRange = GetBalancedWorkRange(static_cast<int32_t>(readyBytes / sizeof(int32_t)), job);
    ResetWorkspaceRegion(readyRange, peermem.dispatchFlag.get_ptr(commBuffer), resetBatchElementCount_, resetTensor_);
    // Server Combine 的连续 token 状态区。
    if (IsCombineServerDedupEnabled()) {
        const auto completedCountRange =
            GetBalancedWorkRange(static_cast<int32_t>(workspace.reset.serverCombine.sizeBytes / sizeof(int32_t)), job);
        ResetWorkspaceRegion(completedCountRange, workspace.urma.serverCombine.completedCounts.get_ptr(buffer),
                             resetBatchElementCount_, resetTensor_);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::QuantizeTokenInUb(
    __ubuf__ bfloat16_t* input, __ubuf__ ActivationType* output, __ubuf__ uint16_t* scratch)
{
    __ubuf__ bfloat16_t* srcAddr = reinterpret_cast<__ubuf__ bfloat16_t*>(input);
    __ubuf__ uint16_t* maxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(scratch);
    __ubuf__ uint16_t* halfScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>(
        (scratch + (Ops::Base::CeilAlign(mxQuantScaleNumAlignPerToken_, static_cast<uint32_t>(ALIGN_32)))));
    __ubuf__ int8_t* outDataAddr = reinterpret_cast<__ubuf__ int8_t*>(output);
    __ubuf__ uint16_t* mxScaleAddr = reinterpret_cast<__ubuf__ uint16_t*>((output + (mxQuantTokenAlignBytes_)));
    Quant::ComputeMaxExp(srcAddr, maxExpAddr, k_);
    Quant::ComputeScale<QuantOutType>(maxExpAddr, mxScaleAddr, halfScaleAddr, mxQuantScaleNumAlignPerToken_);
    if constexpr (QuantMode == E2M1_QUANT) {
        Quant::ComputeFp4Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, k_);
    } else {
        Quant::ComputeFp8Data<bfloat16_t, QuantOutType, AscendC::RoundMode::CAST_TRUNC, AscendC::RoundMode::CAST_RINT>(
            srcAddr, halfScaleAddr, outDataAddr, k_);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::QuantizeLocalTokensToRelay()
{
    const auto& peermem = params_.communication.layout;
    WorkRange tokenRange = GetBalancedWorkRange(m_, {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_});
    uint32_t tokenNumInCore = tokenRange.count;
    uint32_t tokenStart = tokenRange.start;
    if (tokenNumInCore == 0U) {
        return;
    }

    __gm__ bfloat16_t* inputGm = nullptr;
    __gm__ uint8_t* quantOutputGm = nullptr;
    inputGm = reinterpret_cast<__gm__ bfloat16_t*>(params_.aGmAddr) + static_cast<uint64_t>(tokenStart) * k_;

    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);
    for (uint32_t index = 0U; index < tokenNumInCore; ++index) {
        bool useFirstBuffer = (index & 1U) == 0U;
        TEventID event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        __ubuf__ bfloat16_t* xInTensor = useFirstBuffer ? xInTensor1_ : xInTensor2_;
        __ubuf__ ActivationType* xOutTensor = useFirstBuffer ? xOutTensor1_ : xOutTensor2_;
        uint32_t tokenIdx = tokenStart + index;

        PipeWait<PIPE_MTE3, PIPE_MTE2>(event);
        CopyGmToUb(xInTensor, (inputGm + (static_cast<uint64_t>(index) * k_)),
                   static_cast<uint16_t>(k_ * sizeof(bfloat16_t)), GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
        LoadTopkWeightsToUb(xOutTensor, tokenStart, index, event);

        QuantizeTokenInUb(xInTensor, xOutTensor, mxTempTensor_);

        PipeNotify<PIPE_V, PIPE_MTE3>(event);
        PipeWait<PIPE_V, PIPE_MTE3>(event);
        uint64_t relayOffset = RelayTokenOffset(serverId_, tokenIdx);
        GM_ADDR recordAddr = GetRankWinAddrWithOffset(rankId_, peermem.dispatchReceive.offsetBytes) + relayOffset;
        quantOutputGm = reinterpret_cast<__gm__ uint8_t*>(recordAddr);
        __ubuf__ uint8_t* xOutBytesTensor = reinterpret_cast<__ubuf__ uint8_t*>(xOutTensor);
        CopyUbToGm(quantOutputGm, xOutBytesTensor, mxQuantTokenScaleAlignBytes_);
        PipeNotify<PIPE_MTE3, PIPE_MTE2>(event);
    }
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SetLocalRelayReadyFlags()
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    uint32_t baseTokenNum = m_ / blockAivNum_;
    uint32_t tokenRemainder = m_ % blockAivNum_;
    uint32_t tokenNumInCore = baseTokenNum + static_cast<uint32_t>(aivCoreIdx_ < tokenRemainder);
    uint32_t tokenStart = aivCoreIdx_ * baseTokenNum + ((aivCoreIdx_ < tokenRemainder) ? aivCoreIdx_ : tokenRemainder);
    if (tokenNumInCore == 0U) {
        return;
    }

    LocalTensor<uint64_t> readyFlags(TPosition::VECCALC, reinterpret_cast<uint64_t>(relayReadyFillTensor_),
                                     DISPATCH_READY_FLAG_BATCH_TOKENS);
    Duplicate<uint64_t>(readyFlags, uint64_t(1), DISPATCH_READY_FLAG_BATCH_TOKENS);
    PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
    __gm__ uint64_t* readyFlagGlobal = nullptr;
    readyFlagGlobal = reinterpret_cast<__gm__ uint64_t*>(peermem.dispatchFlag.get_ptr(commBuffer)) +
                      static_cast<uint64_t>(serverId_) * params_.tilingData->compute.numMaxTokensPerRank + tokenStart;
    for (uint32_t offset = 0U; offset < tokenNumInCore; offset += DISPATCH_READY_FLAG_BATCH_TOKENS) {
        uint32_t remaining = tokenNumInCore - offset;
        uint32_t currentTokenCount =
            remaining < DISPATCH_READY_FLAG_BATCH_TOKENS ? remaining : DISPATCH_READY_FLAG_BATCH_TOKENS;
        CopyUbToGm(readyFlagGlobal + offset, relayReadyFillTensor_,
                   currentTokenCount * static_cast<uint32_t>(sizeof(uint64_t)));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::EnqueueDispatchRelayToken(uint32_t targetServer,
                                                                                                 uint32_t tokenIdx)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    GM_ADDR queueBase = workspace.urma.relaySendQueue.get_ptr(buffer) + DispatchRelayQueueServerOffset(targetServer);
    __gm__ int32_t* countPtr = reinterpret_cast<__gm__ int32_t*>(queueBase);
    int32_t slotIdx = AtomicAdd(countPtr, int32_t(1));
    uint64_t metaOffset = ALIGN_32 + static_cast<uint64_t>(slotIdx) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
    __gm__ int32_t* metaPtr = reinterpret_cast<__gm__ int32_t*>(queueBase + metaOffset);
    GmWrite(metaPtr, static_cast<int32_t>(tokenIdx));
}

template <TemplateMegaMoeLayeredTypeClass>
template <bool CombineServerDedup>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildDispatchRelayCandidates(
    const LocalTensor<int32_t>& topkIds, const LocalTensor<uint32_t>& masks, uint32_t tokenCount,
    const WorkRange& serverRange)
{
    CallVf<BuildDispatchRelayCandidatesByServerVF<CombineServerDedup>>(
        reinterpret_cast<__ubuf__ int32_t*>(topkIds.GetPhyAddr()),
        reinterpret_cast<__ubuf__ uint32_t*>(masks.GetPhyAddr()), tokenCount, topK_,
        static_cast<int32_t>(worldSize_ * moeExpertPerRank_),
        static_cast<int32_t>(moeExpertPerRank_ * rankNumPerServer_), static_cast<int32_t>(serverId_), serverRange.start,
        serverRange.count);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::EnqueueDispatchRelayCandidates(
    const LocalTensor<uint32_t>& masks, const WorkRange& tokenRange, const WorkRange& serverRange)
{
    constexpr uint32_t ROW_ELEMENTS = AscendC::GetVecLen() / sizeof(int32_t);
    for (uint32_t token = 0; token < tokenRange.count; ++token) {
        for (uint32_t index = 0; index < serverRange.count; ++index) {
            uint32_t bitmapValue = masks.GetValue(token * ROW_ELEMENTS + index);
            if (bitmapValue == 0U) {
                continue;
            }
            EnqueueDispatchRelayToken(serverRange.start + index, tokenRange.start + token);
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::TransposeDispatchRelayBitmaps(
    const LocalTensor<uint32_t>& masks, const LocalTensor<uint32_t>& transposedMasks, uint32_t tokenCount)
{
    constexpr uint32_t ROW_ELEMENTS = AscendC::GetVecLen() / sizeof(uint32_t);
    constexpr uint32_t BLOCK_ELEMENTS = ALIGN_32 / sizeof(uint32_t);
    const uint32_t alignedCount = Ops::Base::CeilAlign(tokenCount, BLOCK_ELEMENTS);
    // 场景 13：[1, alignedCount, 64] -> [1, 64, alignedCount]，每个 Server 行首按 32B 对齐。
    ConfusionTransposeTiling tiling{};
    tiling.param0 = 1U;
    tiling.param1 = alignedCount;
    tiling.param2 = ROW_ELEMENTS;
    // 950 的 021 路径额外 scratch 为 0；传有效 UB 视图，避免无 scratch 重载申请栈缓冲。
    Transpose<uint32_t>(transposedMasks, masks, transposedMasks.template ReinterpretCast<uint8_t>(),
                        TransposeType::TRANSPOSE_ND2ND_021, tiling);
    return alignedCount;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishDispatchRelayBitmaps(
    const LocalTensor<uint32_t>& masks, const WorkRange& tokenRange, const WorkRange& serverRange, uint32_t rowElements)
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    // GM 保留 8B ready 槽；Compact 只写低 32 位，高位由入口整区清零保证。
    const DataCopyExtParams copyParams{static_cast<uint16_t>(tokenRange.count), sizeof(uint32_t), 0U, sizeof(uint32_t),
                                       0U};
    __gm__ uint32_t* readyGlobal = nullptr;
    for (uint32_t index = 0; index < serverRange.count; ++index) {
        const uint32_t server = serverRange.start + index;
        readyGlobal = reinterpret_cast<__gm__ uint32_t*>(peermem.sourceReady.get_ptr(commBuffer) +
                                                         RelayFlagOffset(server, tokenRange.start));
        CopyUbToGm(readyGlobal, reinterpret_cast<__ubuf__ uint32_t*>((masks[index * rowElements]).GetPhyAddr()),
                   CopyRows{static_cast<uint16_t>(tokenRange.count), sizeof(uint32_t),
                            (sizeof(uint32_t)) + (sizeof(uint32_t)), sizeof(uint32_t)});
        if (server == serverId_) {
            readyGlobal = reinterpret_cast<__gm__ uint32_t*>(peermem.dispatchFlag.get_ptr(commBuffer) +
                                                             RelayFlagOffset(server, tokenRange.start));
            CopyUbToGm(readyGlobal, reinterpret_cast<__ubuf__ uint32_t*>((masks[index * rowElements]).GetPhyAddr()),
                       CopyRows{static_cast<uint16_t>(tokenRange.count), sizeof(uint32_t),
                                (sizeof(uint32_t)) + (sizeof(uint32_t)), sizeof(uint32_t)});
        }
    }
}

/*
 * 当前 UB：[0,512B) 通信 scratch；[512B,184KiB) 建表；[184KiB,248KiB) 保留给 WQE。
 * 普通模式 topkIds/masks 两块，每批 367 token；去重模式增加独立转置区，每批 240 token。
 * 每块每 token 预留 256B；去重三块共 180KiB，转置后按 Server 行批量 MTE 发布。
 * 输入每批只搬一次；每段位图完成标量消费及 MTE 发布后，复用同一输出缓冲。
 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareDispatchRelayQueues()
{
    constexpr uint32_t ROW_ELEMENTS = AscendC::GetVecLen() / sizeof(int32_t);
    constexpr uint32_t ROW_BYTES = ROW_ELEMENTS * sizeof(int32_t);
    constexpr uint32_t BLOCK_ELEMENTS = ALIGN_32 / sizeof(uint32_t);
    constexpr uint32_t NORMAL_BATCH_TOKENS = (COMM_WORK_UB_LIMIT - ALIGN_512) / (2U * ROW_BYTES);
    constexpr uint32_t DEDUP_BATCH_TOKENS =
        (COMM_WORK_UB_LIMIT - ALIGN_512) / (3U * ROW_BYTES) / BLOCK_ELEMENTS * BLOCK_ELEMENTS;
    static_assert(NORMAL_BATCH_TOKENS > 0U && NORMAL_BATCH_TOKENS <= UINT16_MAX);
    static_assert(DEDUP_BATCH_TOKENS > 0U && DEDUP_BATCH_TOKENS <= UINT16_MAX);
    const bool combineServerDedup = IsCombineServerDedupEnabled();
    const uint32_t batchTokens = combineServerDedup ? DEDUP_BATCH_TOKENS : NORMAL_BATCH_TOKENS;
    const uint32_t bufferBytes = batchTokens * ROW_BYTES;
    LocalTensor<int32_t> topkIds(TPosition::VECCALC, ALIGN_512, batchTokens * ROW_ELEMENTS);
    LocalTensor<uint32_t> masks(TPosition::VECCALC, ALIGN_512 + bufferBytes, batchTokens * ROW_ELEMENTS);
    LocalTensor<uint32_t> transposedMasks;
    if (combineServerDedup) {
        transposedMasks =
            LocalTensor<uint32_t>(TPosition::VECCALC, ALIGN_512 + 2U * bufferBytes, batchTokens * ROW_ELEMENTS);
    }
    // 全部物理 AIV 处理互不重叠的 token 区间。
    const auto tokenRange = GetBalancedWorkRange(m_, {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_});
    if (combineServerDedup) {
        BuildDispatchRelayQueues<true>(topkIds, masks, transposedMasks, tokenRange, batchTokens);
    } else {
        BuildDispatchRelayQueues<false>(topkIds, masks, transposedMasks, tokenRange, batchTokens);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
template <bool CombineServerDedup>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::BuildDispatchRelayQueues(
    const LocalTensor<int32_t>& topkIds, const LocalTensor<uint32_t>& masks,
    const LocalTensor<uint32_t>& transposedMasks, const WorkRange& tokenRange, uint32_t batchTokens)
{
    constexpr uint32_t ROW_ELEMENTS = AscendC::GetVecLen() / sizeof(int32_t);
    __gm__ int32_t* topkIdsGlobal = nullptr;
    topkIdsGlobal = reinterpret_cast<__gm__ int32_t*>(params_.expertIdxGmAddr);
    uint32_t rowBlocks = (topK_ * sizeof(int32_t) + ALIGN_32 - 1U) / ALIGN_32;
    // 预置缓冲可用事件，循环内固定消费并补回。
    PipeNotify<PIPE_S, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
    PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
    if constexpr (CombineServerDedup) {
        PipeNotify<PIPE_MTE3, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
    }
    for (uint32_t tokenOffset = 0; tokenOffset < tokenRange.count; tokenOffset += batchTokens) {
        const WorkRange batchRange{tokenRange.start + tokenOffset, (tokenRange.count - tokenOffset < batchTokens) ?
                                                                       tokenRange.count - tokenOffset :
                                                                       batchTokens};
        // 覆盖 topkIds 前，等待输入缓冲可复用。
        PipeWait<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
        // GM 行紧凑，UB 每行补齐到 256B；dstStride 以 32B 块计。
        CopyGmToUb(
            reinterpret_cast<__ubuf__ int32_t*>((topkIds).GetPhyAddr()),
            (topkIdsGlobal + (static_cast<uint64_t>(batchRange.start) * topK_)),
            CopyRows{static_cast<uint16_t>(batchRange.count), static_cast<uint32_t>(topK_ * sizeof(int32_t)),
                     static_cast<uint32_t>(topK_ * sizeof(int32_t)),
                     static_cast<uint32_t>(
                         (Ops::Base::CeilAlign(static_cast<uint32_t>(static_cast<uint32_t>(topK_ * sizeof(int32_t))),
                                               static_cast<uint32_t>(ALIGN_32))) +
                         static_cast<uint32_t>(
                             static_cast<uint32_t>(ROW_ELEMENTS * sizeof(int32_t) / ALIGN_32 - rowBlocks)) *
                             ALIGN_32)});
        PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();
        for (uint32_t serverStart = 0; serverStart < serverNum_; serverStart += ROW_ELEMENTS) {
            const WorkRange serverRange{
                serverStart, (serverNum_ - serverStart < ROW_ELEMENTS) ? serverNum_ - serverStart : ROW_ELEMENTS};
            // 覆盖 masks 前，等待标量读取结束。
            PipeWait<PIPE_S, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
            BuildDispatchRelayCandidates<CombineServerDedup>(topkIds, masks, batchRange.count, serverRange);
            PipeNotify<PIPE_V, PIPE_S>(static_cast<event_t>(SYNC_EVENT_ID2));
            if constexpr (CombineServerDedup) {
                // 当前 VF 可与上段搬出重叠；仅转置覆盖输出前等待上段 MTE3 读完。
                PipeWait<PIPE_MTE3, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
                // 950 的 V 流水依赖由硬件保证，转置直接读取当前 VF 的输出。
                const uint32_t rowElements = TransposeDispatchRelayBitmaps(masks, transposedMasks, batchRange.count);
                PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
                PublishDispatchRelayBitmaps(transposedMasks, batchRange, serverRange, rowElements);
                PipeNotify<PIPE_MTE3, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
            }
            PipeWait<PIPE_V, PIPE_S>(static_cast<event_t>(SYNC_EVENT_ID2));
            EnqueueDispatchRelayCandidates(masks, batchRange, serverRange);
            PipeNotify<PIPE_S, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
        }
        PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
    }
    // 回收最后一轮可用事件；空 token 区间也回收入口预置事件。
    PipeWait<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
    PipeWait<PIPE_S, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
    if constexpr (CombineServerDedup) {
        PipeWait<PIPE_MTE3, PIPE_V>(static_cast<event_t>(SYNC_EVENT_ID2));
        // 上面的 Wait 只阻塞 V；Scalar 返回前仍需确认全部 GM ready 发布完成。
        PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID2>();
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AppendDispatchRelayData(uint32_t relayRank,
                                                                                               BatchHandle& batchHandle,
                                                                                               int32_t tokenIdx)
{
    const auto& peermem = params_.communication.layout;
    uint64_t relayOffset = RelayTokenOffset(serverId_, static_cast<uint32_t>(tokenIdx));
    GM_ADDR srcAddr = GetRankWinAddrWithOffset(rankId_, peermem.dispatchReceive.offsetBytes) + relayOffset;
    GM_ADDR dstAddr = GetRankWinAddrWithOffset(relayRank, peermem.dispatchReceive.offsetBytes) + relayOffset;
    comm_.WriteRelaxedNbi(batchHandle, dstAddr, srcAddr, mxQuantTokenScaleAlignBytes_);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AppendDispatchRelayFlag(
    uint32_t targetServer, uint32_t relayRank, BatchHandle& batchHandle, int32_t tokenIdx, bool firstFlag)
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    uint64_t flagOffset = RelayFlagOffset(serverId_, static_cast<uint32_t>(tokenIdx));
    GM_ADDR localFlagAddr =
        IsCombineServerDedupEnabled() ?
            peermem.sourceReady.get_ptr(commBuffer) + RelayFlagOffset(targetServer, static_cast<uint32_t>(tokenIdx)) :
            GetRankWinAddrWithOffset(rankId_, peermem.dispatchFlag.offsetBytes) + flagOffset;
    GM_ADDR remoteFlagAddr = GetRankWinAddrWithOffset(relayRank, peermem.dispatchFlag.offsetBytes) + flagOffset;
    if (firstFlag) {
        comm_.WriteStrongNbi(batchHandle, remoteFlagAddr, localFlagAddr, sizeof(uint64_t));
    } else {
        comm_.WriteRelaxedNbi(batchHandle, remoteFlagAddr, localFlagAddr, sizeof(uint64_t));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CommitDispatchRelayBatch(
    uint32_t targetServer, uint32_t relayRank, BatchHandle& batchHandle, int32_t batchStart, uint32_t batchTokenCount,
    const LocalTensor<int32_t>& tokenIds)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const uint64_t metaOffset = DispatchRelayQueueServerOffset(targetServer) + ALIGN_32 +
                                static_cast<uint64_t>(batchStart) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
    __gm__ int32_t* queueTokenIds = nullptr;
    queueTokenIds = reinterpret_cast<__gm__ int32_t*>(workspace.urma.relaySendQueue.get_ptr(buffer) + metaOffset);
    // 上批两轮标量读取结束后复用 UB；Compact 将间隔 32B 的 token ID 紧凑搬入。
    PipeFence<PIPE_S, PIPE_MTE2, SYNC_EVENT_ID2>();
    const DataCopyExtParams copyParams{static_cast<uint16_t>(batchTokenCount), sizeof(int32_t),
                                       DISPATCH_RELAY_QUEUE_ENTRY_BYTES - sizeof(int32_t), 0U, 0U};
    CopyGmToUb(reinterpret_cast<__ubuf__ int32_t*>((tokenIds).GetPhyAddr()), queueTokenIds,
               CopyRows{static_cast<uint16_t>(batchTokenCount), sizeof(int32_t),
                        (sizeof(int32_t)) + (DISPATCH_RELAY_QUEUE_ENTRY_BYTES - sizeof(int32_t)), sizeof(int32_t)});
    PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID2>();
    // 先提交本批全部 token 数据，再单独组装并提交 flag。首条 flag 使用 odr=6 建立批次边界；
    // 后续 flag 使用 odr=5。三种 WQE 均关闭 CQE，这一配置不外溢到二级读和 Combine。
    for (uint32_t index = 0U; index < batchTokenCount; ++index) {
        AppendDispatchRelayData(relayRank, batchHandle, tokenIds.GetValue(index));
    }
    comm_.BatchCommit(batchHandle);
    for (uint32_t index = 0U; index < batchTokenCount; ++index) {
        AppendDispatchRelayFlag(targetServer, relayRank, batchHandle, tokenIds.GetValue(index), index == 0U);
    }
    comm_.BatchCommit(batchHandle);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishLocalDispatchRelayCount()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    const uint32_t capacity = static_cast<uint32_t>(params_.tilingData->compute.numMaxTokensPerRank);
    auto countPtr = reinterpret_cast<__gm__ int32_t*>(workspace.urma.relaySendQueue.get_ptr(buffer) +
                                                      DispatchRelayQueueServerOffset(serverId_));
    const uint64_t encodedCount = static_cast<uint64_t>(GmRead(countPtr)) + 1U;
    const uint64_t countOffset = RelayFlagOffset(serverId_, capacity);
    GmWrite(reinterpret_cast<__gm__ uint64_t*>(peermem.sourceReady.get_ptr(commBuffer) + countOffset), encodedCount);
    GmWrite(reinterpret_cast<__gm__ uint64_t*>(peermem.dispatchFlag.get_ptr(commBuffer) + countOffset), encodedCount);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendDispatchRelayQueue(
    uint32_t targetServer, uint32_t relayRank, const LocalTensor<int32_t>& tokenIds)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    // 队列已经由全部 AIV 构建并同步完成，[0, count) 可直接按连续 slot 分批发送。
    __gm__ int32_t* countPtr = reinterpret_cast<__gm__ int32_t*>(workspace.urma.relaySendQueue.get_ptr(buffer) +
                                                                 DispatchRelayQueueServerOffset(targetServer));
    int32_t tokenCount = GmRead(countPtr);
    Channel channel = comm_.GetChannel(relayRank);
    BatchHandle batchHandle =
        comm_.MakeBatchHandle(channel,
                              LocalTensor<uint8_t>(TPosition::VECCALC, reinterpret_cast<uint64_t>(hcommBatchWqeTensor_),
                                                   LAYERED_HCOMM_BATCH_UB_BYTES),
                              LAYERED_HCOMM_BATCH_UB_BYTES, GetRankWinAddrWithOffset(relayRank, 0));
    for (int32_t batchStart = 0; batchStart < tokenCount;
         batchStart += static_cast<int32_t>(DISPATCH_SEND_BATCH_TOKEN_CAPACITY)) {
        int32_t remaining = tokenCount - batchStart;
        uint32_t batchTokenCount =
            static_cast<uint32_t>(remaining < static_cast<int32_t>(DISPATCH_SEND_BATCH_TOKEN_CAPACITY) ?
                                      remaining :
                                      static_cast<int32_t>(DISPATCH_SEND_BATCH_TOKEN_CAPACITY));
        CommitDispatchRelayBatch(targetServer, relayRank, batchHandle, batchStart, batchTokenCount, tokenIds);
    }
    if (IsCombineServerDedupEnabled()) {
        const uint32_t capacity = static_cast<uint32_t>(params_.tilingData->compute.numMaxTokensPerRank);
        GM_ADDR sourceCount = peermem.sourceReady.get_ptr(commBuffer) + RelayFlagOffset(targetServer, capacity);
        GmWrite(reinterpret_cast<__gm__ uint64_t*>(sourceCount), static_cast<uint64_t>(tokenCount) + 1U);
        // E_dsburma3: 该 count 只发布一次且远端以非零为放行判据；旁路写未排空时 DMA 会把 0 发走，
        // 远端 BuildServerCombineLists 等非零即永挂。WriteNbi 读源前必须 DSB。
        GmDataBarrier();
        // 空队列也发布 count+1；默认写生成 CQE，完成等待留给后续读依赖或 Wave 出口统一 Drain。
        GM_ADDR remoteCount = GetRankWinAddrWithOffset(relayRank, peermem.dispatchFlag.offsetBytes) +
                              RelayFlagOffset(serverId_, capacity);
        comm_.WriteNbi(channel, remoteCount, sourceCount, sizeof(uint64_t));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SendDispatchRelayQueues()
{
    // 本 Server 的中继就是本卡；由固定归属 AIV 发布数量，不经过通信通道。
    if (IsCombineServerDedupEnabled()) {
        if (IsChannelOwner(rankId_)) {
            PublishLocalDispatchRelayCount();
        }
    }
    // 各通道串行发送；复用后续接收 metadata 的既有 UB 区，不改变接收绑定。
    LocalTensor<int32_t> tokenIds(TPosition::VECCALC, ubBufferUsedAddr_, DISPATCH_SEND_BATCH_TOKEN_CAPACITY);
    for (uint32_t targetServer = 0; targetServer < serverNum_; ++targetServer) {
        uint32_t relayRank = targetServer * rankNumPerServer_ + rankIdInServer_;
        // 固定 channel owner 发送各远端中继队列。
        if (targetServer == serverId_ || relayRank >= worldSize_ || !IsChannelOwner(relayRank)) {
            continue;
        }
        SendDispatchRelayQueue(targetServer, relayRank, tokenIds);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadTokenFromLocalRelay(uint32_t srcServer,
                                                                                               uint32_t tokenIndex,
                                                                                               int32_t bufferIdx,
                                                                                               uint32_t copyInNum)
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    GM_ADDR readyAddr = peermem.dispatchFlag.get_ptr(commBuffer) + RelayFlagOffset(srcServer, tokenIndex);
    while (GmRead(reinterpret_cast<__gm__ uint64_t*>(readyAddr)) == 0U) {
    }

    uint64_t remoteCopyOffset = RelayTokenOffset(srcServer, tokenIndex);
    GM_ADDR localRecordAddr = GetRankWinAddrWithOffset(rankId_, peermem.dispatchReceive.offsetBytes) + remoteCopyOffset;
    __gm__ ActivationType* relayRecordsGm = reinterpret_cast<__gm__ ActivationType*>(localRecordAddr);
    CopyGmToUb(copyTmpTensors_[bufferIdx], relayRecordsGm, (copyInNum) * sizeof(ActivationType));
    PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID1>();
}

// 一次构建完整专家计数表并备份前缀和，同时统计计算 wave 数。
// 全部写回完成后仅发布本 block 的整表就绪标志。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareExpertTokenCountTable()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    gmmWaveCount_ = 0U;
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (subBlockIdx_ != 1U || moeExpertPerRank_ == 0U) {
        return;
    }

    // count 位于 mask slot 尾部，按 slot stride 压缩搬入连续 UB 表。
    // maskRecv 基址 512B 对齐、maskAlignSize_ 32B 对齐；blockCount <= 2048，满足 Compact Copy。
    auto* countGm = reinterpret_cast<__gm__ int32_t*>(peermem.maskRecv.get_ptr(commBuffer) + maskAlignSize_);
    PipeFence<PIPE_V, PIPE_MTE2, SYNC_EVENT_ID2>();
    CopyGmToUb(cumsumInfoTensor_, countGm,
               CopyRows{static_cast<uint16_t>(worldSize_ * moeExpertPerRank_), sizeof(int32_t), maskSlotSize_,
                        sizeof(int32_t)},
               GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
    PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();

    // 复用 MTE 的 count 表计算：前缀和 + 专家差分 + maxOutputSize clamp。
    // URMA mask count 高 8 位为 0，0x00FFFFFF 的 epoch mask 在当前场景下是无操作。
    ComputeExpertCountTables(cumsumInfoTensor_, expertTokenNumsOutTensor_, moeExpertPerRank_, worldSize_,
                             static_cast<uint32_t>(maxOutputSize_));
    PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID2>();

    const BlockWorkspaceContext countWorkspace{blockIdx_, aicNum_};
    uint64_t countOffset = GetExpertCountWorkspaceOffset(countWorkspace, moeExpertPerRank_, 0U, true);
    PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
    CopyUbToGm(expertRecvCountsGm_ + countOffset, expertTokenNumsOutTensor_,
               moeExpertPerRank_ * static_cast<uint32_t>(sizeof(int32_t)));
    CopyUbToGm(cumsumGm_, cumsumInfoTensor_, worldSize_ * moeExpertPerRank_ * static_cast<uint32_t>(sizeof(int32_t)));
    // 两张表写回完成后才发布就绪标志；预取权重由后续接收和消费位置的同步独立保护。
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID2>();

    __gm__ int32_t* tokenCountReadyFlag =
        reinterpret_cast<__gm__ int32_t*>(workspace.sync.countsReady.get_ptr(buffer)) +
        static_cast<uint64_t>(blockIdx_) * INT_CACHELINE;
    GmWrite(tokenCountReadyFlag, static_cast<int32_t>(1));

    // 缓存仅由本核 AIV1 消费，先发布整表就绪，再用 VF 完成元数据统计，避免延后其他角色读取计数。
    // VF 复用 metadata 区前缀：result[0] 为 totalMGroups，result[8] 为最后非空专家下标。
    auto* waveMetaTensor = reinterpret_cast<__ubuf__ int32_t*>(ubBufferUsedAddr_);
    CallVf<ComputeLayeredWaveMetaVF>(expertTokenNumsOutTensor_, waveMetaTensor, moeExpertPerRank_);
    PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID2>();

    const int32_t totalMGroups = waveMetaTensor[0];
    const int32_t lastIndex = waveMetaTensor[8];
    const uint32_t lastNonEmptyExpertEnd = lastIndex >= 0 ? static_cast<uint32_t>(lastIndex) + 1U : 0U;
    if (totalMGroups <= 0) {
        gmmWaveCount_ = 1U; // 全部专家为空时，规划器仍需一个元数据 wave 完成遍历。
        return;
    }
    const uint64_t totalMGroups64 = static_cast<uint64_t>(totalMGroups);
    gmmWaveCount_ = (totalMGroups64 - 1U) / mGroupsPerWave_ + 1U;
    // 最后一个满 wave 后仍有空专家时，保留规划器产生的元数据尾 wave。
    if (totalMGroups64 % mGroupsPerWave_ == 0U && lastNonEmptyExpertEnd < moeExpertPerRank_) {
        ++gmmWaveCount_;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateLocalRelayWeight(int32_t copyIdx,
                                                                                              int32_t outIdx,
                                                                                              TEventID eventId)
{
    if constexpr (TopkWeightsPrefetch) {
        PipeWait<PIPE_MTE2, PIPE_S>(eventId);
        __ubuf__ ActivationType* weightBuf = copyTmpTensors_[outIdx];
        uint32_t weightOffsetInUb = mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_;
        __ubuf__ int32_t* bufWeightsInt32 = reinterpret_cast<__ubuf__ int32_t*>((weightBuf + (weightOffsetInUb)));
        int32_t topkIndex = (metaInfoTensor_ + (copyIdx * INT32_PER_256B))[TOPK_INDEX];
        int32_t weightBits = bufWeightsInt32[static_cast<uint32_t>(topkIndex)];
        (metaInfoTensor_ + (copyIdx * INT32_PER_256B))[WEIGHT_INDEX] = weightBits;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadLocalRelayToken(
    uint32_t srcServer, uint32_t tokenIndex, int32_t bufferIdx, uint32_t copyInNum, TEventID eventId)
{
    PipeWait<PIPE_MTE3, PIPE_MTE2>(eventId);
    LoadTokenFromLocalRelay(srcServer, tokenIndex, bufferIdx, copyInNum);
    PipeNotify<PIPE_MTE2, PIPE_MTE3>(eventId);
    if constexpr (TopkWeightsPrefetch) {
        PipeNotify<PIPE_MTE2, PIPE_S>(eventId);
    }
}

// ============================================================================
// CopyTokensFromLocalRelay：本卡中转使用 UB 多 buffer 搬运 token 与 scale
// ----------------------------------------------------------------------------
//   prime: 发出前 DISPATCH_COPY_BUFFER_COUNT 个 token 的 MTE2。
//   steady: 每轮执行 MTE3_out[i] + MTE2_in[i + DISPATCH_COPY_BUFFER_COUNT]，循环复用槽位。
//   drain: 收尾不再发新 MTE2，只等待 MTE3 完成。
// ============================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyTokensFromLocalRelay(
    int32_t rowDstOffsetInCore, uint32_t srcServer, int32_t copyNum, int64_t widthA, int64_t widthAScale,
    uint32_t copyInNum)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    constexpr TEventID kBufEvents[DISPATCH_COPY_BUFFER_COUNT] = {EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5};
    __gm__ ActivationType* recvDataGm = nullptr;
    __gm__ QuantScaleOutType* recvScaleGm = nullptr;
    recvDataGm = reinterpret_cast<__gm__ ActivationType*>(workspace.routed.gmm1Input.data.get_ptr(buffer) +
                                                          static_cast<uint64_t>(rowDstOffsetInCore) * widthA *
                                                              sizeof(ActivationType));
    recvScaleGm = reinterpret_cast<__gm__ QuantScaleOutType*>(workspace.routed.gmm1Input.scale.get_ptr(buffer) +
                                                              static_cast<uint64_t>(rowDstOffsetInCore) * widthAScale *
                                                                  sizeof(QuantScaleOutType));

    for (int32_t bufferIdx = 0; bufferIdx < DISPATCH_COPY_BUFFER_COUNT; ++bufferIdx) {
        PipeNotify<PIPE_MTE3, PIPE_MTE2>(kBufEvents[bufferIdx]);
    }

    int32_t primeCount = copyNum < DISPATCH_COPY_BUFFER_COUNT ? copyNum : DISPATCH_COPY_BUFFER_COUNT;
    for (int32_t primeIdx = 0; primeIdx < primeCount; ++primeIdx) {
        int32_t tokenIndex = (metaInfoTensor_ + (primeIdx * INT32_PER_256B))[TOKEN_ID];
        TEventID eventId = kBufEvents[primeIdx];
        LoadLocalRelayToken(srcServer, static_cast<uint32_t>(tokenIndex), primeIdx, copyInNum, eventId);
    }

    for (int32_t copyIdx = 0; copyIdx < copyNum; ++copyIdx) {
        int32_t outIdx = copyIdx % DISPATCH_COPY_BUFFER_COUNT;
        TEventID eventId = kBufEvents[outIdx];
        PipeWait<PIPE_MTE2, PIPE_MTE3>(eventId);

        UpdateLocalRelayWeight(copyIdx, outIdx, eventId);
        __ubuf__ ActivationType* tokenScaleBuf = copyTmpTensors_[outIdx];
        __ubuf__ QuantScaleOutType* scaleBuf =
            reinterpret_cast<__ubuf__ QuantScaleOutType*>((tokenScaleBuf + (mxQuantTokenAlignBytes_)));
        CopyUbToGm((recvDataGm + (copyIdx * widthA)), tokenScaleBuf,
                   static_cast<uint16_t>(widthA * sizeof(ActivationType)));
        CopyUbToGm((recvScaleGm + (copyIdx * widthAScale)), scaleBuf,
                   static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)));
        PipeNotify<PIPE_MTE3, PIPE_MTE2>(eventId);

        int32_t nextIdx = copyIdx + DISPATCH_COPY_BUFFER_COUNT;
        if (nextIdx < copyNum) {
            int32_t tokenIndex = (metaInfoTensor_ + (nextIdx * INT32_PER_256B))[TOKEN_ID];
            // 等待本轮 MTE3 完成后再复用 outIdx 槽。
            LoadLocalRelayToken(srcServer, static_cast<uint32_t>(tokenIndex), outIdx, copyInNum, eventId);
        }
    }

    for (int32_t bufferIdx = 0; bufferIdx < DISPATCH_COPY_BUFFER_COUNT; ++bufferIdx) {
        PipeWait<PIPE_MTE3, PIPE_MTE2>(kBufEvents[bufferIdx]);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PublishDispatchRows(uint32_t expertIdx,
                                                                                           int32_t globalRowStart,
                                                                                           int32_t rowCount)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    if (rowCount <= 0) {
        return;
    }
    int32_t priorExpertRows = expertIdx == 0U ? 0 : cumsumInfoTensor_[expertIdx * worldSize_ - 1U];
    int32_t localRow = globalRowStart - priorExpertRows;
    int32_t remainingRows = rowCount;
    __gm__ int32_t* expertFlagBase = workspace.sync.dispatchReady.template get_ptr<int32_t>(buffer) +
                                     static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_;
    while (remainingRows > 0) {
        int32_t waveIdx = localRow / static_cast<int32_t>(L1_TILE_M_256);
        int32_t rowsInWave = static_cast<int32_t>(L1_TILE_M_256) - localRow % static_cast<int32_t>(L1_TILE_M_256);
        if (rowsInWave > remainingRows) {
            rowsInWave = remainingRows;
        }
        atomicAdd(expertFlagBase + static_cast<int64_t>(waveIdx) * INT_CACHELINE, rowsInWave);
        localRow += rowsInWave;
        remainingRows -= rowsInWave;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint32_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::FindDispatchSegmentEnd(
    uint32_t segmentStart, uint32_t batchTokenCount)
{
    __ubuf__ int32_t* firstRecord = (metaInfoTensor_ + (segmentStart * INT32_PER_256B));
    uint32_t expertIdx = static_cast<uint32_t>(firstRecord[DISPATCH_BATCH_EXPERT_ID]);
    int32_t dstRowStart = firstRecord[DISPATCH_BATCH_DST_ROW];
    uint32_t segmentEnd = segmentStart + 1U;
    while (segmentEnd < batchTokenCount) {
        __ubuf__ int32_t* nextRecord = (metaInfoTensor_ + (segmentEnd * INT32_PER_256B));
        if (static_cast<uint32_t>(nextRecord[DISPATCH_BATCH_EXPERT_ID]) != expertIdx ||
            nextRecord[DISPATCH_BATCH_DST_ROW] != dstRowStart + static_cast<int32_t>(segmentEnd - segmentStart)) {
            break;
        }
        ++segmentEnd;
    }
    return segmentEnd;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReceiveRemoteDispatchBatch(
    uint32_t relayRank, uint32_t batchTokenCount)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    if (batchTokenCount == 0U) {
        return;
    }

    GM_ADDR scratchAddr = workspace.urma.remoteReadySnapshot.get_ptr(buffer) +
                          static_cast<uint64_t>(blockIdx_) * dispatchRelayFlagSnapshotBytesPerBlock_;
    __gm__ uint64_t* scratchFlagGlobalTensor = nullptr;
    scratchFlagGlobalTensor = reinterpret_cast<__gm__ uint64_t*>(scratchAddr);
    DataCopyPadExtParams<uint64_t> flagCopyPadParams{true, 0U, 0U, 0U};
    Channel channel = comm_.GetChannel(relayRank);

    // 批次可以跨专家和源 Server，但不能跨 relay 通道。ready flag 按 token-id 的 256 槽窗口读取，
    // 只处理当前批次实际命中的窗口。token id 属于源 Rank，上界必须用全卡容量而非本卡 m_。
    for (uint32_t srcServer = 0U; srcServer < serverNum_; ++srcServer) {
        for (uint64_t windowStart64 = 0U;
             windowStart64 < static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank);
             windowStart64 += DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY) {
            uint32_t windowStart = static_cast<uint32_t>(windowStart64);
            uint64_t windowEnd64 = windowStart64 + DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY;
            if (windowEnd64 > static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank)) {
                windowEnd64 = static_cast<uint64_t>(params_.tilingData->compute.numMaxTokensPerRank);
            }
            uint32_t windowEnd = static_cast<uint32_t>(windowEnd64);
            bool windowHasToken = false;
            for (uint32_t index = 0U; index < batchTokenCount; ++index) {
                __ubuf__ int32_t* record = (metaInfoTensor_ + (index * INT32_PER_256B));
                if (static_cast<uint32_t>(record[DISPATCH_BATCH_SRC_SERVER]) != srcServer) {
                    continue;
                }
                uint32_t tokenIndex = static_cast<uint32_t>(record[TOKEN_ID]);
                if (tokenIndex >= windowStart && tokenIndex < windowEnd) {
                    windowHasToken = true;
                    break;
                }
            }
            if (!windowHasToken) {
                continue;
            }

            uint64_t flagSnapshotBytes = static_cast<uint64_t>(windowEnd - windowStart) * sizeof(uint64_t);

            GM_ADDR remoteFlagAddr = GetRankWinAddrWithOffset(relayRank, peermem.dispatchFlag.offsetBytes) +
                                     RelayFlagOffset(srcServer, windowStart);
            bool windowTokensReady = false;
            while (!windowTokensReady) {
                comm_.ReadNbi(channel, scratchAddr, remoteFlagAddr, flagSnapshotBytes);
                comm_.Drain(channel);
                PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID1>();
                CopyGmToUb(relayFlagTensor_, reinterpret_cast<__gm__ uint64_t*>(scratchAddr),
                           static_cast<uint32_t>(flagSnapshotBytes), GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
                PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID1>();

                windowTokensReady = true;
                for (uint32_t index = 0U; index < batchTokenCount; ++index) {
                    __ubuf__ int32_t* record = (metaInfoTensor_ + (index * INT32_PER_256B));
                    if (static_cast<uint32_t>(record[DISPATCH_BATCH_SRC_SERVER]) != srcServer) {
                        continue;
                    }
                    uint32_t tokenIndex = static_cast<uint32_t>(record[TOKEN_ID]);
                    if (tokenIndex >= windowStart && tokenIndex < windowEnd) {
                        const uint64_t ready = relayFlagTensor_[tokenIndex - windowStart];
                        if (ready == 0U) {
                            windowTokensReady = false;
                            break;
                        }
                    }
                }
            }
        }
    }

    BatchHandle batchHandle = comm_.MakeBatchHandle(
        channel,
        LocalTensor<uint8_t>(TPosition::VECCALC, reinterpret_cast<uintptr_t>(hcommBatchWqeTensor_),
                             LAYERED_HCOMM_BATCH_UB_BYTES),
        LAYERED_HCOMM_BATCH_UB_BYTES, GetRankWinAddrWithOffset(relayRank, 0));
    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    for (uint32_t index = 0U; index < batchTokenCount; ++index) {
        __ubuf__ int32_t* record = (metaInfoTensor_ + (index * INT32_PER_256B));
        uint32_t srcServer = static_cast<uint32_t>(record[DISPATCH_BATCH_SRC_SERVER]);
        uint32_t tokenIndex = static_cast<uint32_t>(record[TOKEN_ID]);
        uint32_t dstRow = static_cast<uint32_t>(record[DISPATCH_BATCH_DST_ROW]);
        GM_ADDR remoteRecordAddr = GetRankWinAddrWithOffset(relayRank, peermem.dispatchReceive.offsetBytes) +
                                   RelayTokenOffset(srcServer, tokenIndex);
        GM_ADDR tokenDstAddr = workspace.routed.gmm1Input.data.get_ptr(buffer) +
                               static_cast<uint64_t>(dstRow) * widthA * sizeof(ActivationType);
        GM_ADDR scaleDstAddr = workspace.routed.gmm1Input.scale.get_ptr(buffer) +
                               static_cast<uint64_t>(dstRow) * widthAScale * sizeof(QuantScaleOutType);
        comm_.ReadNbi(batchHandle, tokenDstAddr, remoteRecordAddr, widthA * sizeof(ActivationType));
        comm_.ReadNbi(batchHandle, scaleDstAddr, remoteRecordAddr + mxQuantTokenAlignBytes_,
                      widthAScale * sizeof(QuantScaleOutType));
        if constexpr (TopkWeightsPrefetch) {
            GM_ADDR weightDstAddr =
                workspace.routing.tokenWeights.get_ptr(buffer) + static_cast<uint64_t>(dstRow) * weightAlignBytes_;
            comm_.ReadNbi(batchHandle, weightDstAddr,
                          remoteRecordAddr + mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, weightAlignBytes_);
        }
    }
    comm_.BatchCommit(batchHandle);
    comm_.Drain(batchHandle);

    if constexpr (TopkWeightsPrefetch) {
        for (uint32_t index = 0U; index < batchTokenCount; ++index) {
            __ubuf__ int32_t* record = (metaInfoTensor_ + (index * INT32_PER_256B));
            uint32_t dstRow = static_cast<uint32_t>(record[DISPATCH_BATCH_DST_ROW]);
            uint32_t topkIndex = static_cast<uint32_t>(record[TOPK_INDEX]);
            __gm__ int32_t* weightGmI32 = reinterpret_cast<__gm__ int32_t*>(
                workspace.routing.tokenWeights.get_ptr(buffer) + static_cast<uint64_t>(dstRow) * weightAlignBytes_);
            record[WEIGHT_INDEX] = GmRead(weightGmI32 + topkIndex);
        }
    }

    // 每个 expert/rank 段内的 metadata 连续。先写完全部 metadata，再发布专家就绪计数。
    // 元数据及预取权重由标量更新，搬出前需保证这些 UB 写入已完成。
    PipeFence<PIPE_S, PIPE_MTE3, SYNC_EVENT_ID3>();
    uint32_t segmentStart = 0U;
    while (segmentStart < batchTokenCount) {
        __ubuf__ int32_t* firstRecord = (metaInfoTensor_ + (segmentStart * INT32_PER_256B));
        int32_t dstRowStart = firstRecord[DISPATCH_BATCH_DST_ROW];
        uint32_t segmentEnd = FindDispatchSegmentEnd(segmentStart, batchTokenCount);
        CopyUbToGm((metadataGm_ + (static_cast<uint64_t>(dstRowStart) * INT32_PER_256B)),
                   (metaInfoTensor_ + (segmentStart * INT32_PER_256B)),
                   ((segmentEnd - segmentStart) * INT32_PER_256B) * sizeof(int32_t));
        segmentStart = segmentEnd;
    }
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID3>();

    segmentStart = 0U;
    while (segmentStart < batchTokenCount) {
        __ubuf__ int32_t* firstRecord = (metaInfoTensor_ + (segmentStart * INT32_PER_256B));
        uint32_t expertIdx = static_cast<uint32_t>(firstRecord[DISPATCH_BATCH_EXPERT_ID]);
        int32_t dstRowStart = firstRecord[DISPATCH_BATCH_DST_ROW];
        uint32_t segmentEnd = FindDispatchSegmentEnd(segmentStart, batchTokenCount);
        PublishDispatchRows(expertIdx, dstRowStart, static_cast<int32_t>(segmentEnd - segmentStart));
        segmentStart = segmentEnd;
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyLocalDispatchTokens(int32_t dstRow,
                                                                                               int32_t srcRank,
                                                                                               int32_t tokenOffset,
                                                                                               int32_t tokenCount)
{
    if (tokenCount <= 0) {
        return;
    }
    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale = Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                          MXFP_MULTI_BASE_SIZE; // 输出 token-scale 长度,紧密排列
    uint32_t copyInNum = mxQuantTokenScaleAlignBytes_ / sizeof(ActivationType);
    if constexpr (!TopkWeightsPrefetch) {
        // mode4 一级记录仍保留中继归约所需权重，二级只搬 token 和 scale。
        copyInNum =
            Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32)) /
            sizeof(ActivationType);
    }
    uint32_t srcServer = static_cast<uint32_t>(srcRank) / rankNumPerServer_;

    for (int32_t i = 0; i < tokenCount; ++i) {
        int32_t topkIndex = validTopkIndexTensor_[tokenOffset + i];
        int32_t tokenIndex = topkIndex / topK_;
        (metaInfoTensor_ + (i * INT32_PER_256B))[RANK_ID] = srcRank;
        (metaInfoTensor_ + (i * INT32_PER_256B))[TOKEN_ID] = tokenIndex;
        (metaInfoTensor_ + (i * INT32_PER_256B))[TOPK_INDEX] = topkIndex % topK_;
    }

    CopyTokensFromLocalRelay(dstRow, srcServer, tokenCount, widthA, widthAScale, copyInNum);
    PipeFence<PIPE_S, PIPE_MTE3, SYNC_EVENT_ID3>();
    CopyUbToGm((metadataGm_ + (dstRow * INT32_PER_256B)), metaInfoTensor_,
               (tokenCount * INT32_PER_256B) * sizeof(int32_t));
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID3>();
}

// 加载完整 cumsum 表到 UB，并初始化 dispatch metadata 临时区。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareDispatchReceive()
{
    CopyGmToUb(cumsumInfoTensor_, cumsumGm_, static_cast<uint32_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)),
               GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
    PipeFence<PIPE_MTE2, PIPE_S, SYNC_EVENT_ID2>();
    metaInfoTensor_ = reinterpret_cast<__ubuf__ int32_t*>(ubBufferUsedAddr_);
}

// 计算当前专家与通信 wave 的交集，返回全局紧凑行号区间。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CalcDispatchExpertSlice(
    uint32_t expertIdx, uint32_t expertBegin, const ExpertTokenRange& range, int32_t& sliceStart, int32_t& sliceEnd)
{
    const int32_t expertStart = expertIdx == 0U ? 0 : cumsumInfoTensor_[expertIdx * worldSize_ - 1U];
    const int32_t expertEndRow = cumsumInfoTensor_[(expertIdx + 1U) * worldSize_ - 1U];
    sliceStart = expertStart + static_cast<int32_t>(expertIdx == expertBegin ? range.begin.tokenIndexInExpert : 0U);
    sliceEnd = expertIdx == range.end.expertIdx ? expertStart + static_cast<int32_t>(range.end.tokenIndexInExpert) :
                                                  expertEndRow;
    sliceEnd = sliceEnd < static_cast<int32_t>(maxOutputSize_) ? sliceEnd : static_cast<int32_t>(maxOutputSize_);
}

// 搬入本轮 mask 并 GatherMask，返回本轮命中的 topk 行数。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline uint64_t MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadAndGatherDispatchRound(
    __gm__ uint8_t* rankMaskBasePtr, uint32_t roundIdx)
{
    uint64_t roundStart = static_cast<uint64_t>(roundIdx) * static_cast<uint64_t>(dispatchRoundSendTotalNum_);
    uint32_t roundLen = (roundIdx + 1U < dispatchTotalRounds_) ? dispatchRoundSendTotalNum_ :
                                                                 static_cast<uint32_t>(maskRouteCapacity_ - roundStart);
    uint32_t topkIndexTensorElemCount =
        static_cast<uint32_t>(
            Ops::Base::CeilAlign(static_cast<int64_t>(roundLen * sizeof(int32_t)), static_cast<int64_t>(ALIGN_32))) /
        sizeof(int32_t);
    uint32_t curRoundMaskBytes =
        (roundLen == dispatchRoundSendTotalNum_) ?
            dispatchRoundMaskAlignSize_ :
            static_cast<uint32_t>(Ops::Base::CeilAlign(static_cast<int64_t>(Ops::Base::CeilDiv(roundLen, 8U)),
                                                       static_cast<int64_t>(ALIGN_32)));
    uint64_t roundMaskByteOffset = roundStart / 8U;
    auto* roundMaskSrc = rankMaskBasePtr + roundMaskByteOffset;
    PipeFence<PIPE_V, PIPE_MTE2, SYNC_EVENT_ID1>();
    CopyGmToUb(gatherMaskTensor_, roundMaskSrc, curRoundMaskBytes);
    PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID1>();
    LocalTensor<int32_t> indices(TPosition::VECCALC, reinterpret_cast<uint64_t>(topkIndexTensor_),
                                 topkIndexTensorElemCount);
    CreateVecIndex(indices, static_cast<int32_t>(roundStart), topkIndexTensorElemCount);

    uint64_t roundTokenCount = 0U;
    LocalTensor<uint32_t> rankMaskSlice(TPosition::VECCALC, reinterpret_cast<uint64_t>(gatherMaskInt32Tensor_),
                                        curRoundMaskBytes / sizeof(uint32_t));
    LocalTensor<int32_t> selected(TPosition::VECCALC, reinterpret_cast<uint64_t>(validTopkIndexTensor_),
                                  topkIndexTensorElemCount);
    GatherMask(selected, indices, rankMaskSlice, true, roundLen, {1, 1, 0, 0}, roundTokenCount);
    PipeFence<PIPE_V, PIPE_S, SYNC_EVENT_ID4>();
    return roundTokenCount;
}

// 将本轮命中行裁剪到通信 wave 的 [sourceBegin, sourceEnd)。
// mask 容量尾部可能残留上次调用数据，validRoundCount 只取 rankTokenCount 的剩余量。
__aicore__ inline void ClipDispatchRoundToWave(int32_t accumulatedRows, uint64_t roundTokenCount,
                                               int32_t rankTokenCount, int32_t sourceBegin, int32_t sourceEnd,
                                               int32_t& validRoundCount, int32_t& copyBegin, int32_t& roundCopyCount,
                                               int32_t& gatheredOffset)
{
    const int32_t remainingRankRows = rankTokenCount - accumulatedRows;
    validRoundCount = static_cast<int32_t>(roundTokenCount);
    if (validRoundCount > remainingRankRows) {
        validRoundCount = remainingRankRows;
    }
    copyBegin = accumulatedRows > sourceBegin ? accumulatedRows : sourceBegin;
    const int32_t roundEnd = accumulatedRows + validRoundCount;
    const int32_t copyEnd = roundEnd < sourceEnd ? roundEnd : sourceEnd;
    roundCopyCount = copyEnd - copyBegin;
    gatheredOffset = copyBegin - accumulatedRows;
}

// 本地 relay：按 256 行 chunk 搬入，并发布对应 group 的 GMM1 ready。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CopyLocalDispatchRound(
    uint32_t expertIdx, uint32_t srcRank, int32_t rowStart, int32_t copyBegin, int32_t roundCopyCount,
    int32_t gatheredOffset)
{
    for (int32_t chunkStart = 0; chunkStart < roundCopyCount; chunkStart += static_cast<int32_t>(L1_TILE_M_256)) {
        int32_t chunkRows = roundCopyCount - chunkStart;
        if (chunkRows > static_cast<int32_t>(L1_TILE_M_256)) {
            chunkRows = static_cast<int32_t>(L1_TILE_M_256);
        }
        int32_t dstRow = rowStart + copyBegin + chunkStart;
        CopyLocalDispatchTokens(dstRow, static_cast<int32_t>(srcRank), gatheredOffset + chunkStart, chunkRows);
        PublishDispatchRows(expertIdx, dstRow, chunkRows);
    }
}

// 远端 relay：逐 token 填 metadata，攒满批次后立即读取到本地 workspace。
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AppendRemoteDispatchRound(
    uint32_t relayRank, uint32_t expertIdx, uint32_t srcServer, uint32_t srcRank, int32_t rowStart, int32_t copyBegin,
    int32_t roundCopyCount, int32_t gatheredOffset, uint32_t& remoteBatchTokenCount)
{
    for (int32_t tokenOffset = 0; tokenOffset < roundCopyCount; ++tokenOffset) {
        int32_t topkIndex = validTopkIndexTensor_[gatheredOffset + tokenOffset];
        int32_t dstRow = rowStart + copyBegin + tokenOffset;
        auto* record = metaInfoTensor_ + remoteBatchTokenCount * INT32_PER_256B;
        record[RANK_ID] = static_cast<int32_t>(srcRank);
        record[TOKEN_ID] = topkIndex / static_cast<int32_t>(topK_);
        record[TOPK_INDEX] = topkIndex % static_cast<int32_t>(topK_);
        record[WEIGHT_INDEX] = 0;
        record[DISPATCH_BATCH_DST_ROW] = dstRow;
        record[DISPATCH_BATCH_EXPERT_ID] = static_cast<int32_t>(expertIdx);
        record[DISPATCH_BATCH_SRC_SERVER] = static_cast<int32_t>(srcServer);
        ++remoteBatchTokenCount;
        if (remoteBatchTokenCount == DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY) {
            ReceiveRemoteDispatchBatch(relayRank, remoteBatchTokenCount);
            remoteBatchTokenCount = 0U;
        }
    }
}

// =================================================================================================
// ReceiveDispatchTokenRange：接收一个通信 wave 覆盖的 dispatch token 行区间。
//   - 按 relay 通道、专家、源 rank、dispatch 轮裁剪；
//   - 本地 relay 直接搬入，远端 relay 攒 metadata 后走 ReceiveRemoteDispatchBatch；
//   - 按 256 行 group 发布 GMM1 dispatch ready flag。
// =================================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ReceiveDispatchTokenRange(
    const ExpertTokenRange& range)
{
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    const uint32_t expertBegin = range.begin.expertIdx;
    const uint32_t expertEnd = range.end.expertIdx + (range.end.tokenIndexInExpert != 0U ? 1U : 0U);
    if (expertBegin >= expertEnd || expertBegin >= moeExpertPerRank_) {
        return;
    }
    PrepareDispatchReceive();

    for (uint32_t srcRankInServer = 0U; srcRankInServer < rankNumPerServer_; ++srcRankInServer) {
        uint32_t relayRank = serverId_ * rankNumPerServer_ + srcRankInServer;
        if (relayRank >= worldSize_ || !IsChannelOwner(relayRank)) {
            continue;
        }
        bool isRemoteRelay = relayRank != rankId_;
        uint32_t remoteBatchTokenCount = 0U;

        for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
            int32_t sliceStart = 0;
            int32_t sliceEnd = 0;
            CalcDispatchExpertSlice(expertIdx, expertBegin, range, sliceStart, sliceEnd);
            for (uint32_t srcServer = 0U; srcServer < serverNum_; ++srcServer) {
                uint32_t srcRank = srcServer * rankNumPerServer_ + srcRankInServer;
                if (srcRank >= worldSize_) {
                    continue;
                }

                int32_t rowStart =
                    (srcRank == 0U && expertIdx == 0U) ? 0 : cumsumInfoTensor_[expertIdx * worldSize_ + srcRank - 1U];
                if (rowStart >= static_cast<int32_t>(maxOutputSize_)) {
                    continue;
                }
                __gm__ uint8_t* rankMaskBasePtr =
                    reinterpret_cast<__gm__ uint8_t*>(peermem.maskRecv.get_ptr(commBuffer) +
                                                      static_cast<uint64_t>(expertIdx) * worldSize_ * maskSlotSize_ +
                                                      static_cast<uint64_t>(srcRank) * maskSlotSize_);
                __gm__ int32_t* rankCountPtr = reinterpret_cast<__gm__ int32_t*>(rankMaskBasePtr + maskAlignSize_);
                const int32_t rankTokenCount = GmRead(rankCountPtr);
                if (rankTokenCount <= 0) {
                    continue;
                }

                // 将当前 rank 收集到的 token 序号范围裁剪到当前专家的 wave 分段内。
                const int32_t sourceBegin = sliceStart > rowStart ? sliceStart - rowStart : 0;
                const int32_t sourceEnd = sliceEnd < rowStart + rankTokenCount ? sliceEnd - rowStart : rankTokenCount;
                if (sourceBegin >= sourceEnd) {
                    continue;
                }
                int32_t accumulatedRows = 0;
                for (uint32_t roundIdx = 0U; roundIdx < dispatchTotalRounds_ && accumulatedRows < sourceEnd;
                     ++roundIdx) {
                    uint64_t roundTokenCount = LoadAndGatherDispatchRound(rankMaskBasePtr, roundIdx);
                    if (roundTokenCount == 0U) {
                        continue;
                    }

                    int32_t validRoundCount = 0;
                    int32_t copyBegin = 0;
                    int32_t roundCopyCount = 0;
                    int32_t gatheredOffset = 0;
                    ClipDispatchRoundToWave(accumulatedRows, roundTokenCount, rankTokenCount, sourceBegin, sourceEnd,
                                            validRoundCount, copyBegin, roundCopyCount, gatheredOffset);
                    if (roundCopyCount > 0) {
                        if (!isRemoteRelay) {
                            CopyLocalDispatchRound(expertIdx, srcRank, rowStart, copyBegin, roundCopyCount,
                                                   gatheredOffset);
                        } else {
                            AppendRemoteDispatchRound(relayRank, expertIdx, srcServer, srcRank, rowStart, copyBegin,
                                                      roundCopyCount, gatheredOffset, remoteBatchTokenCount);
                        }
                    }
                    accumulatedRows += validRoundCount;
                }
            }
        }

        // 扫描完当前通信 Wave 后，仅通道尾批允许不足 256 token。
        if (isRemoteRelay && remoteBatchTokenCount != 0U) {
            ReceiveRemoteDispatchBatch(relayRank, remoteBatchTokenCount);
        }
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline auto MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitSharedExpertInputBuffers()
    -> QuantProcessScratch<ActivationType>
{
    uint32_t xInAddr = ALIGN_512;
    uint32_t xInSize = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_128)) * sizeof(bfloat16_t);
    __ubuf__ bfloat16_t* xInBuf0 = reinterpret_cast<__ubuf__ bfloat16_t*>(xInAddr);
    __ubuf__ bfloat16_t* xInBuf1 = reinterpret_cast<__ubuf__ bfloat16_t*>(xInAddr + xInSize);
    uint32_t mxTempAddr = xInAddr + xInSize * 2;
    __ubuf__ uint16_t* mxTempBuf = reinterpret_cast<__ubuf__ uint16_t*>(mxTempAddr);
    uint32_t xOutAddr = mxTempAddr + MX_QUANT_TEMP_UB_BYTES;
    uint32_t xOutSize =
        Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    __ubuf__ ActivationType* xOutBuf0 = reinterpret_cast<__ubuf__ ActivationType*>(xOutAddr);
    __ubuf__ ActivationType* xOutBuf1 = reinterpret_cast<__ubuf__ ActivationType*>(xOutAddr + xOutSize);
    QuantProcessScratch<ActivationType> scratch{};
    scratch.xInTensor0 = xInBuf0;
    scratch.xInTensor1 = xInBuf1;
    scratch.xOutTensor0 = xOutBuf0;
    scratch.xOutTensor1 = xOutBuf1;
    scratch.mxTempTensor = mxTempBuf;
    return scratch;
}

// ===============================================================
// SharedExpertCopyInput：从原始 bf16 输入量化后写入共享专家专用缓冲区
//   源: aGmAddr [bs × h] bf16（layered URMA 模式下 quantTokenScale 区域未分配，需直接从原始输入量化）
//   目标: sharedExpertInputDataPtr [bs × h] fp8 连续, sharedExpertInputScalePtr [bs × scaleN] 连续
//   AIV 执行，在 AIC GMM1 开始前调用
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::SharedExpertCopyInput()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    if constexpr (g_coreType == AIC) {
        return;
    }
    int32_t currentNum;
    int32_t currentOffset;
    TilingByCore(m_, currentNum, currentOffset, 1);

    int64_t widthA = k_ / A_ELEMS_PER_BYTE;
    int64_t widthAScale =
        Ops::Base::CeilDiv(static_cast<int64_t>(k_), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;

    auto scratch = InitSharedExpertInputBuffers();
    __gm__ bfloat16_t* inputGm = nullptr;
    __gm__ ActivationType* dataOutputGm = nullptr;
    __gm__ QuantScaleOutType* scaleOutputGm = nullptr;
    dataOutputGm = workspace.shared.gmm1Input.data.template get_ptr<ActivationType>(buffer);
    scaleOutputGm = workspace.shared.gmm1Input.scale.template get_ptr<QuantScaleOutType>(buffer);

    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeNotify<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);
    for (int32_t index = 0; index < currentNum; index++) {
        int32_t tokenIdx = currentOffset + index;
        auto event = (index % DOUBLE_BUFFER == 0) ? EVENT_ID0 : EVENT_ID1;
        auto xInBuf = (index % DOUBLE_BUFFER == 0) ? scratch.xInTensor0 : scratch.xInTensor1;
        auto xOutBuf = (index % DOUBLE_BUFFER == 0) ? scratch.xOutTensor0 : scratch.xOutTensor1;

        inputGm = reinterpret_cast<__gm__ bfloat16_t*>(params_.aGmAddr +
                                                       static_cast<uint64_t>(tokenIdx) * k_ * sizeof(bfloat16_t));
        PipeWait<PIPE_MTE3, PIPE_MTE2>(event);
        CopyGmToUb(xInBuf, inputGm, static_cast<uint16_t>(k_ * sizeof(bfloat16_t)),
                   GmLoadCacheMode::NORMAL_FIRST_VICTIM, true);
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);

        QuantizeTokenInUb(xInBuf, xOutBuf, scratch.mxTempTensor);

        PipeNotify<PIPE_V, PIPE_MTE3>(event);
        PipeWait<PIPE_V, PIPE_MTE3>(event);
        __ubuf__ QuantScaleOutType* bufScale =
            reinterpret_cast<__ubuf__ QuantScaleOutType*>((xOutBuf + (mxQuantTokenAlignBytes_)));
        CopyUbToGm((dataOutputGm + (tokenIdx * widthA)), xOutBuf,
                   static_cast<uint16_t>(widthA * sizeof(ActivationType)));
        CopyUbToGm((scaleOutputGm + (tokenIdx * widthAScale)), bufScale,
                   static_cast<uint16_t>(widthAScale * sizeof(QuantScaleOutType)));
        PipeNotify<PIPE_MTE3, PIPE_MTE2>(event);
    }
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID0);
    PipeWait<PIPE_MTE3, PIPE_MTE2>(EVENT_ID1);
    PipeBarrier<PIPE_ALL>();
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_LAYERED_DISPATCH_H
