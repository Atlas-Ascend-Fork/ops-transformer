/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_COMBINE_H
#define MEGA_MOE_LAYERED_COMBINE_H

// Internal member definitions; included by mega_moe_layered.h after the class declaration.
namespace MegaMoeImpl {

/**
 * 绑定 Combine 消费的 GMM2 输出与组同步计数地址。
 */
__aicore__ inline void UpdateCombineGlobalBuffer(GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state,
                                                 const WorkspaceLayout& workspace, GM_ADDR workspaceBase,
                                                 uint32_t combineSyncSlotCountPerExpert)
{
    UpdateMoeExpertCombineGlobalBuffer(workspace, workspaceBase, gmmAddrInfo, state);
    gmmAddrInfo.gmm2CombineSyncCounter =
        (__gm__ int32_t*)workspace.urma.sync.combineCounters.get_ptr(workspaceBase) +
        static_cast<uint64_t>(state.expertIdx) * combineSyncSlotCountPerExpert * static_cast<uint64_t>(INT_CACHELINE);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitCombineBuffers()
{
    if constexpr (g_coreType == AIV) {
        // Combine 仅在当前 AIV1 完成 Dispatch 张量使用后启动，并从 UB 首地址重建视图。
        // Hcomm 使用前 512 字节；Combine metadata 和计算临时区从其后开始复用 UB。
        uint32_t nAlign32 = Ops::Base::CeilAlign(k_, static_cast<uint32_t>(ALIGN_32));
        uint32_t quantTokenSizeBytes = static_cast<uint32_t>(CalcCombineTokenBytes(k_, sizeof(bfloat16_t), true));
        uint32_t singleTokenBytes = nAlign32 * sizeof(bfloat16_t) + quantTokenSizeBytes;
        combineUbTensorSize_ = (singleTokenBytes * 2) / sizeof(bfloat16_t);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline bool MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GetCombineRankRange(uint32_t expertIdx,
                                                                                           uint32_t originRank,
                                                                                           uint32_t& rowStart,
                                                                                           uint32_t& tokenCount)
{
    // originRank 指 token 原始输入卡，用于查询结果行范围；实际发送目标由 batchState 保存。
    uint64_t expertBegin = static_cast<uint64_t>(expertIdx) * worldSize_;
    uint64_t rankEndIndex = expertBegin + originRank;
    CleanInvalidateGmCache();
    int32_t rankEnd = cumsumGm_[rankEndIndex];

    int32_t expertStart = 0;
    if (expertBegin != 0U) {
        CleanInvalidateGmCache();
        expertStart = *(cumsumGm_ + (expertBegin - 1U));
    }
    int32_t rankStart = expertStart;
    if (originRank != 0U) {
        CleanInvalidateGmCache();
        rankStart = cumsumGm_[rankEndIndex - 1U];
    }
    rowStart = static_cast<uint32_t>(rankStart - expertStart);
    tokenCount = static_cast<uint32_t>(rankEnd - rankStart);
    return tokenCount != 0U;
}

/* 处理当前专家的一段结果行，等待对应 GMM2 分组完成，再构造普通写或去重贡献请求。 */
template <TemplateMegaMoeLayeredTypeClass>
template <bool CombineServerDedup>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CombineTokensInRange(
    const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& gmm2State, uint32_t expertIdx, const WorkRange& rowRange,
    CombineBatchState& batchState)
{
    const uint8_t CombineQuantMode = params_.tilingData->combineQuantMode;
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;

    uint32_t mExpert = Get<M_VALUE>(gmm2State.problemShape);
    uint32_t nTilesPerGroup = Ops::Base::CeilDiv(k_, L1_TILE_N);
    GroupSyncSlotLayout slotLayout = CalcGroupSyncSlotLayout(mExpert, blockNum_);
    __gm__ int32_t* expertCounterBase = reinterpret_cast<__gm__ int32_t*>(gmmAddrInfo.gmm2CombineSyncCounter);

    uint32_t processedCount = 0U;
    while (processedCount < rowRange.count) {
        uint32_t currentRow = rowRange.start + processedCount;
        uint32_t targetGroup = currentRow / L1_TILE_M_256;
        uint32_t firstSyncSlot = 0;
        uint32_t syncSlotCount = 0;
        GetGroupSyncSlotRange(targetGroup, slotLayout, firstSyncSlot, syncSlotCount);
        __gm__ int32_t* counterAddr = GetCombineSyncCounterAddress(expertCounterBase, firstSyncSlot);
        WaitUntilGmFlagAtLeast(counterAddr, static_cast<int32_t>(nTilesPerGroup));

        uint32_t groupEndRow = (targetGroup + 1U) * L1_TILE_M_256;
        groupEndRow = groupEndRow < mExpert ? groupEndRow : mExpert;
        uint32_t batchCount = groupEndRow - currentRow;
        uint32_t remainingTokens = rowRange.count - processedCount;
        batchCount = batchCount < remainingTokens ? batchCount : remainingTokens;

        // 两种模式共用同一 metadata 基址；普通 Group 与去重 Group 都自行搬入 metadata 并绑定 UB。
        GM_ADDR metaInfoAddr =
            workspace.routing.receivedTokenMetadata.get_ptr(buffer) +
            static_cast<uint64_t>(gmm2State.globalTokenStartIndex + currentRow) * META_INFO_SIZE * sizeof(int32_t);
        if constexpr (CombineServerDedup) {
            CombineImpl::CombineServerDedupTokenGroup(currentRow, batchCount, k_, expertIdx, rankId_,
                                                      gmmAddrInfo.gmm2OutGlobal, metaInfoAddr, params_, comm_,
                                                      batchState);
        } else {
            CombineImpl::CombineTokenGroup<bfloat16_t>(CombineQuantMode, currentRow, batchCount, k_, expertIdx, rankId_,
                                                       gmmAddrInfo.gmm2OutGlobal, metaInfoAddr, combineUbTensorSize_,
                                                       params_, comm_, batchState);
        }
        processedCount += batchCount;
    }
}

/*
 * 处理当前专家发往实际 targetRank 的结果，expertRowRange 限定当前通信 Wave 的专家行范围。
 * 普通模式处理该输入卡；Server 去重模式处理各 Server 中与中继同编号的原始输入卡。
 */
template <TemplateMegaMoeLayeredTypeClass>
template <bool CombineServerDedup>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::CombineTokensOfExpert(
    const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& gmm2State, uint32_t expertIdx,
    const WorkRange& expertRowRange, CombineBatchState& batchState)
{
    const uint32_t targetRank = batchState.dstRankId;
    // 处理当前专家中属于一张原始输入卡的结果，共用当前实际目标通道的批次。
    auto combineTokensFromRank = [&](uint32_t originRank) __attribute__((cce_aicore))
    {
        WorkRange rowRange;
        if (!GetCombineRankRange(expertIdx, originRank, rowRange.start, rowRange.count)) {
            return;
        }
        // 一个专家可能跨多个通信 Wave；仅发送来源卡范围与本 Wave 的交集。
        const uint32_t rowEnd = rowRange.start + rowRange.count;
        const uint32_t waveEnd = expertRowRange.start + expertRowRange.count;
        const uint32_t begin = rowRange.start > expertRowRange.start ? rowRange.start : expertRowRange.start;
        const uint32_t end = rowEnd < waveEnd ? rowEnd : waveEnd;
        if (begin >= end) {
            return;
        }
        CombineTokensInRange<CombineServerDedup>(gmmAddrInfo, gmm2State, expertIdx, WorkRange{begin, end - begin},
                                                 batchState);
    };
    if constexpr (!CombineServerDedup) {
        combineTokensFromRank(targetRank);
        return;
    }

    // 本 Server 中继负责各 Server 内与其同编号的原始输入卡。
    const uint32_t rankInServer = targetRank % rankNumPerServer_;
    for (uint32_t originServer = 0; originServer < serverNum_; ++originServer) {
        const uint32_t originRank = originServer * rankNumPerServer_ + rankInServer;
        if (originRank < worldSize_) {
            combineTokensFromRank(originRank);
        }
    }
}

/*
 * 向实际 targetRank 处理一个通信 Wave 的 Combine；保留完整专家状态，并逐专家裁剪行范围。
 * 跨计算 Wave、专家和输入卡积累同一通道的请求，满批提交，通信 Wave 结束时提交尾批。
 */
template <TemplateMegaMoeLayeredTypeClass>
template <bool CombineServerDedup>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineToTargetRank(
    ExpertLoopState waveBeginState, const ExpertTokenRange& range, uint32_t targetRank)
{
    CombineBatchState batchState{};
    batchState.dstRankId = targetRank;
    if (targetRank != rankId_) {
        Channel channel = comm_.GetChannel(targetRank);
        batchState.batchHandle = comm_.MakeBatchHandle(channel, GetRankWinAddrWithOffset(targetRank, 0));
    }
    ExpertLoopState combineState = waveBeginState;
    GMMAddrInfo combineAddrInfo{};
    const uint32_t expertBegin = range.begin.expertIdx;
    const uint32_t expertEnd = range.end.expertIdx + (range.end.tokenIndexInExpert != 0U ? 1U : 0U);
    for (uint32_t expertIdx = expertBegin; expertIdx < expertEnd; ++expertIdx) {
        const uint32_t rowBegin = expertIdx == expertBegin ? range.begin.tokenIndexInExpert : 0U;
        // 首专家从中间行继续时，waveBeginState 已保留该专家的完整形状及基址。
        if (rowBegin == 0U) {
            const BlockWorkspaceContext countWorkspace{blockIdx_, aicNum_};
            const uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(
                params_.workspace.layout.routing.recvCounts.get_ptr(params_.workspace.base), countWorkspace,
                moeExpertPerRank_, expertIdx);
            UpdateExpertLoopState(combineState, expertIdx, expertTokenCount);
            if (expertTokenCount == 0U) {
                continue;
            }
        }
        const uint32_t expertRows = Get<M_VALUE>(combineState.problemShape);
        const uint32_t rowEnd = expertIdx == range.end.expertIdx ? range.end.tokenIndexInExpert : expertRows;
        if (rowBegin >= rowEnd) {
            continue;
        }
        // Combine 按消费范围绑定 GMM2 输出与组同步计数，地址由累计行数和专家序号直接计算。
        UpdateCombineGlobalBuffer(combineAddrInfo, combineState, params_.workspace.layout, params_.workspace.base,
                                  static_cast<uint32_t>(params_.tilingData->compute.combineSyncSlotCountPerExpert));
        CombineTokensOfExpert<CombineServerDedup>(combineAddrInfo, combineState, expertIdx,
                                                  WorkRange{rowBegin, rowEnd - rowBegin}, batchState);
    }
    if (targetRank != rankId_ && batchState.pendingTokenCount != 0U) {
        comm_.BatchCommit(batchState.batchHandle);
        batchState.pendingTokenCount = 0U;
    }
}

/* 按实际目标卡分配通信 Wave 的 Combine；去重模式的目标为本 Server 内中继，普通模式为源卡。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineTokenRange(
    const ExpertLoopState& waveBeginState, const ExpertTokenRange& range)
{
    const uint32_t expertBegin = range.begin.expertIdx;
    const uint32_t expertEnd = range.end.expertIdx + (range.end.tokenIndexInExpert != 0U ? 1U : 0U);
    if (expertBegin >= expertEnd) {
        return;
    }

    // 每个实际目标卡在一个通信 Wave 中共用批次；普通模式回源卡，去重模式发往本 Server 中继。
    const bool combineServerDedup = IsCombineServerDedupEnabled();
    const WorkRange targetRange =
        combineServerDedup ? WorkRange{serverId_ * rankNumPerServer_, rankNumPerServer_} : WorkRange{0U, worldSize_};
    for (uint32_t targetRank = targetRange.start;
         targetRank < targetRange.start + targetRange.count && targetRank < worldSize_; ++targetRank) {
        if (!IsChannelOwner(targetRank)) {
            continue;
        }
        if (combineServerDedup) {
            ProcessCombineToTargetRank<true>(waveBeginState, range, targetRank);
        } else {
            ProcessCombineToTargetRank<false>(waveBeginState, range, targetRank);
        }
    }
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_LAYERED_COMBINE_H
