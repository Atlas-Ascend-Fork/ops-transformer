/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_WAVE_ARCH92_H
#define MEGA_MOE_LAYERED_WAVE_ARCH92_H

#include "mega_moe_layered_arch92.h"

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
namespace MegaMoeImpl {

#define TemplateMegaMoeLayeredWaveTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t Weight1Format, int32_t Weight2Format, typename CommPolicy
#define TemplateMegaMoeLayeredWaveTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, Weight1Format, Weight2Format, CommPolicy

template <typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode,
          int32_t Weight1Format, int32_t Weight2Format, typename CommPolicy = LayeredUrmaComm>
class MegaMoeLayeredWave : public MegaMoeLayered<TemplateMegaMoeLayeredWaveTypeFunc> {
    using Base = MegaMoeLayered<TemplateMegaMoeLayeredWaveTypeFunc>;
    friend Base;

public:
    __aicore__ inline explicit MegaMoeLayeredWave(GM_ADDR context)
        : Base(context)
    {}
    __aicore__ inline void Process();

private:
    using Base::DISPATCH_READY_CACHE_RESERVE_BYTES;
    using Base::GMM1_TILE_M;
    using Base::aicNum_;
    using Base::blockIdx_;
    using Base::commonConfig_;
    using Base::dispatchSourceReadyBitsUb_;
    using Base::gmmWaveCount_;
    using Base::mGroupsPerWave_;
    using Base::moeExpertPerRank_;
    using Base::params_;
    using Base::startBlockIdx_;
    using Base::DispatchBuffInit;
    using Base::GroupMatmulWithActivationQuant;
    using Base::GroupMatmulWithCombine;
    using Base::InitCombineBuffers;
    using Base::LockOwnedRemoteChannels;
    using Base::PrepareExpertTokenCountTable;
    using Base::ProcessCombineTokenRange;
    using Base::ReceiveDispatchTokenRange;
    using Base::StartInterServerDispatch;
    using Base::UpdateGmm1GlobalBuffer;
    using Base::UpdateGmm2GlobalBuffer;
    using Base::worldSize_;

    __aicore__ inline void InitDispatchSourceReadyCache();
    __aicore__ inline ExpertTokenRange PlanTokenWave(const ExpertTokenPosition& begin, uint32_t groupTarget);
    __aicore__ inline void AdjustStartBlockIdxForNextWave(const ExpertTokenRange& wave, uint32_t waveStartBlockIdx,
                                                          uint32_t gmm1EndBlockIdx);
    template <AddrUpdateMode Mode>
    __aicore__ inline void ProcessGmmTokenRange(const ExpertTokenRange& range, ExpertLoopState& state,
                                                GMMAddrInfo& addrInfo);
    __aicore__ inline void ProcessMoeExpertWaveLoop(ExpertLoopState& gmm1State, ExpertLoopState& gmm2State,
                                                    GMMAddrInfo& gmm1AddrInfo, GMMAddrInfo& gmm2AddrInfo,
                                                    ExpertTokenRange currentWave);
    __aicore__ inline void ProcessMoeExpertWave(Gmm1ActivationSync& gmm1ActivationSync);
};

template <TemplateMegaMoeLayeredWaveTypeClass>
__aicore__ inline void MegaMoeLayeredWave<TemplateMegaMoeLayeredWaveTypeFunc>::InitDispatchSourceReadyCache()
{
    if constexpr (g_coreType == AIV) {
        const uint32_t readyCacheBytes =
            Ops::Base::CeilAlign(Ops::Base::CeilDiv(worldSize_, uint32_t(32)) * static_cast<uint32_t>(sizeof(uint32_t)),
                                 static_cast<uint32_t>(ALIGN_32));
        dispatchSourceReadyBitsUb_ =
            reinterpret_cast<__ubuf__ uint32_t*>(ARCH92_LAYERED_UB_BYTES - DISPATCH_READY_CACHE_RESERVE_BYTES);
        // 每次调用初始化一次；计数表准备阶段的 V→S 同步保证首次标量访问前清零完成。
        // 后续仅由当前 AIV 以标量方式读写。
        const uint32_t cacheWordCount = readyCacheBytes / sizeof(uint32_t);
        LocalTensor<uint32_t> readyCache(TPosition::VECCALC, reinterpret_cast<uint64_t>(dispatchSourceReadyBitsUb_),
                                         cacheWordCount);
        Duplicate<uint32_t>(readyCache, 0U, cacheWordCount);
    }
}

// 完整计数表在入口就绪后，按调用方给定的 M-group 预算规划 wave。
template <TemplateMegaMoeLayeredWaveTypeClass>
__aicore__ inline ExpertTokenRange MegaMoeLayeredWave<TemplateMegaMoeLayeredWaveTypeFunc>::PlanTokenWave(
    const ExpertTokenPosition& begin, uint32_t groupTarget)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    ExpertTokenRange range{begin, begin};
    uint32_t groupCount = 0U;
    const BlockWorkspaceContext countWorkspace{blockIdx_, aicNum_};
    while (range.end.expertIdx < moeExpertPerRank_ && groupCount < groupTarget) {
        uint32_t rowCount = GetExpertTokenCountFromWorkspace(workspace.routing.recvCounts.get_ptr(buffer),
                                                             countWorkspace, moeExpertPerRank_, range.end.expertIdx);
        if (rowCount == 0U || range.end.tokenIndexInExpert >= rowCount) {
            ++range.end.expertIdx;
            range.end.tokenIndexInExpert = 0U;
            continue;
        }
        AdvanceExpertTokenPositionInWave<GMM1_TILE_M>(rowCount, groupTarget, groupCount, range.end);
    }
    return range;
}

// 保持原 GMM wave 的固定角色共振：GMM1 结束时若整体 block 游标发生偏移，则下一 wave 从该偏移开始。
template <TemplateMegaMoeLayeredWaveTypeClass>
__aicore__ inline void MegaMoeLayeredWave<TemplateMegaMoeLayeredWaveTypeFunc>::AdjustStartBlockIdxForNextWave(
    const ExpertTokenRange&, uint32_t, uint32_t)
{
    // 与 Arch35 A8W4 一样，激活和通信由同一个 AIV 处理，无需按 AIV0 角色补偿。
}

template <TemplateMegaMoeLayeredWaveTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayeredWave<TemplateMegaMoeLayeredWaveTypeFunc>::ProcessGmmTokenRange(
    const ExpertTokenRange& range, ExpertLoopState& state, GMMAddrInfo& addrInfo)
{
    uint32_t expertEnd = range.end.expertIdx + (range.end.tokenIndexInExpert == 0U ? 0U : 1U);
    for (uint32_t expertIdx = range.begin.expertIdx; expertIdx < expertEnd; ++expertIdx) {
        uint32_t rowBegin = expertIdx == range.begin.expertIdx ? range.begin.tokenIndexInExpert : 0U;
        if (rowBegin == 0U) {
            // 按专家顺序推进累计行偏移：空专家直接跳过，非空专家把 M 换成当前专家行数。
            const BlockWorkspaceContext countWorkspace{blockIdx_, aicNum_};
            const uint32_t expertTokenCount = GetExpertTokenCountFromWorkspace(
                params_.workspace.layout.routing.recvCounts.get_ptr(params_.workspace.base), countWorkspace,
                moeExpertPerRank_, expertIdx);
            UpdateExpertLoopState(state, expertIdx, expertTokenCount);
            if (expertTokenCount == 0U) {
                continue;
            }
        }
        uint32_t expertRows = static_cast<uint32_t>(Get<M_VALUE>(state.problemShape));
        uint32_t rowEnd = expertIdx == range.end.expertIdx ? range.end.tokenIndexInExpert : expertRows;
        if (rowEnd <= rowBegin) {
            continue;
        }
        // 临时计算视图只切 M；地址由 globalTokenStartIndex + 专家内偏移在绑定时计算。
        ExpertLoopState problem = state;
        Get<M_VALUE>(problem.problemShape) = static_cast<int64_t>(rowEnd - rowBegin);
        if constexpr (Mode == AddrUpdateMode::GMM1) {
            UpdateGmm1GlobalBuffer(addrInfo, problem, rowBegin);
            GroupMatmulWithActivationQuant(addrInfo, problem, expertIdx, rowBegin);
        } else {
            UpdateGmm2GlobalBuffer(addrInfo, problem, rowBegin);
            GroupMatmulWithCombine(addrInfo, problem, expertRows, rowBegin);
        }
    }
}

// =================================================================================================
// ProcessMoeExpertWaveLoop：按“计算 wave + 通信 wave”两层结构滚动执行 GMM1/GMM2/Combine。
//   - 计算 wave（GMM wave）：一次 GMM1 + GMM2，消费一个 ExpertTokenRange；
//   - 通信 wave：固定每个 comm wave 承载 gmmWavesPerCommunicationWave = ceil(gmmWaveCount /
//     MAX_COMMUNICATION_WAVES) 个计算 wave；最后不足的部分作为尾块；
//     每个 comm wave 执行一次 Dispatch 接收、gmmWavesPerCommunicationWave 个计算 wave、一次 Combine 提交；
//   - currentWave：下一个待执行的计算 wave；communicationWave：当前通信 wave 已覆盖的范围；
//   - combineWaveBeginState：通信 wave 起点处的 GMM2 游标；
//   - gmmWavesPerCommunicationWave 只在 AIV 上计算，AIC 固定为 1；
//   - AIV 负责通信、激活和 Combine；AIC 只执行计算 wave。
// =================================================================================================
template <TemplateMegaMoeLayeredWaveTypeClass>
__aicore__ inline void MegaMoeLayeredWave<TemplateMegaMoeLayeredWaveTypeFunc>::ProcessMoeExpertWaveLoop(
    ExpertLoopState& gmm1State, ExpertLoopState& gmm2State, GMMAddrInfo& gmm1AddrInfo, GMMAddrInfo& gmm2AddrInfo,
    ExpertTokenRange currentWave)
{
    constexpr uint32_t MAX_COMMUNICATION_WAVES = 2U;
    uint64_t gmmWavesPerCommunicationWave = 1U;
    if constexpr (g_coreType == AIV) {
        // 固定每个 comm wave 承载 gmmWavesPerCommunicationWave = ceil(gmmWaveCount / MAX_COMMUNICATION_WAVES)，
        // 最后不足的部分作为尾块。
        gmmWavesPerCommunicationWave = gmmWaveCount_ == 0U ? 1U : (gmmWaveCount_ - 1U) / MAX_COMMUNICATION_WAVES + 1U;
    }
    // AIV 预取第二段 Dispatch，再执行当前段激活；保持 Arch35 A8W4 的原有顺序。
    // dispatchRevData 按 maxOutputSize 整块分配，提前接收不增加内存占用。
    bool dispatchPrefetched = false;
    while (currentWave.begin.expertIdx < moeExpertPerRank_) {
        // 外层通信 wave：AIV 一次性接收 gmmWavesPerCommunicationWave 个计算 wave 范围；
        // AIC 固定按 1 个计算 wave 处理。
        ExpertTokenRange communicationWave = currentWave;
        ExpertLoopState combineWaveBeginState = gmm2State;
        if constexpr (g_coreType == AIV) {
            // 整个 comm wave 的 M-group 预算 = gmmWavesPerCommunicationWave * mGroupsPerWave_；
            // 该值受 maxOutputSize/256 约束，uint32 足够。
            const uint64_t groupTarget64 = gmmWavesPerCommunicationWave * static_cast<uint64_t>(mGroupsPerWave_);
            if (!dispatchPrefetched) {
                ExpertTokenRange dispatchWave = PlanTokenWave(currentWave.begin, static_cast<uint32_t>(groupTarget64));
                ReceiveDispatchTokenRange(dispatchWave);
                // 预取下一通信 wave 的范围：从本段起点按计算 wave 推进 gmmWavesPerCommunicationWave 次。
                ExpertTokenRange nextWaveBegin = currentWave;
                for (uint64_t i = 0U; i < gmmWavesPerCommunicationWave; ++i) {
                    nextWaveBegin = PlanTokenWave(nextWaveBegin.end, mGroupsPerWave_);
                }
                if (nextWaveBegin.begin.expertIdx < moeExpertPerRank_) {
                    ExpertTokenRange nextDispatch =
                        PlanTokenWave(nextWaveBegin.begin, static_cast<uint32_t>(groupTarget64));
                    ReceiveDispatchTokenRange(nextDispatch);
                    dispatchPrefetched = true;
                }
            }
        }

        // 内层计算 wave：满 comm wave 预算的 gmmWavesPerCommunicationWave 个走正常路径，
        // 最后不足的部分作为尾块处理。
        for (uint64_t gmmWaveIdx = 0U;
             gmmWaveIdx < gmmWavesPerCommunicationWave && currentWave.begin.expertIdx < moeExpertPerRank_;
             ++gmmWaveIdx) {
            // 单个计算 wave：GMM1、GMM2 使用同一 token 区间；GMM1 结束后的 block 游标用于固定角色共振。
            const uint32_t waveStartBlockIdx = startBlockIdx_;
            ProcessGmmTokenRange<AddrUpdateMode::GMM1>(currentWave, gmm1State, gmm1AddrInfo);
            const uint32_t gmm1EndBlockIdx = startBlockIdx_;
            ProcessGmmTokenRange<AddrUpdateMode::GMM2>(currentWave, gmm2State, gmm2AddrInfo);
            communicationWave.end = currentWave.end;
            AdjustStartBlockIdxForNextWave(currentWave, waveStartBlockIdx, gmm1EndBlockIdx);
            // 当前计算 wave 的 GMM 调用结束后，再推进下一计算 wave 的边界。
            currentWave = PlanTokenWave(currentWave.end, mGroupsPerWave_);
        }

        // 通信 wave 尾：以当前通信 wave 范围执行 Combine；WQE 可跨计算 wave 累积，尾批在此提交。
        if constexpr (g_coreType == AIV) {
            ProcessCombineTokenRange(combineWaveBeginState, communicationWave);
        }
    }
}

template <TemplateMegaMoeLayeredWaveTypeClass>
__aicore__ inline void MegaMoeLayeredWave<TemplateMegaMoeLayeredWaveTypeFunc>::ProcessMoeExpertWave(
    Gmm1ActivationSync& gmm1ActivationSync)
{
    const auto& workspace = params_.workspace.layout;
    const auto base = params_.workspace.base;
    DispatchBuffInit();
    InitCombineBuffers();
    InitDispatchSourceReadyCache();
    LockOwnedRemoteChannels();
    auto gmm1State = CreateExpertLoopState(commonConfig_);
    auto gmm2State = CreateExpertLoopState(commonConfig_);
    GMMAddrInfo gmm1Info{};
    GMMAddrInfo gmm2Info{};
    gmm1Info.gmm1ActivationSync = &gmm1ActivationSync;
    if constexpr (g_coreType == AIV) {
        StartInterServerDispatch();
        PrepareExpertTokenCountTable();
    } else if (moeExpertPerRank_ != 0U) {
        WaitForMoeExpertTokenCountReady(workspace.sync.countsReady.get_ptr(base), {blockIdx_, aicNum_}, 0U);
    }
    auto currentWave = PlanTokenWave({}, mGroupsPerWave_);
    ProcessMoeExpertWaveLoop(gmm1State, gmm2State, gmm1Info, gmm2Info, currentWave);
}

template <TemplateMegaMoeLayeredWaveTypeClass>
__aicore__ inline void MegaMoeLayeredWave<TemplateMegaMoeLayeredWaveTypeFunc>::Process()
{
    this->ProcessWave(*this);
}

#undef TemplateMegaMoeLayeredWaveTypeClass
#undef TemplateMegaMoeLayeredWaveTypeFunc
} // namespace MegaMoeImpl
#endif // ENABLE_MEGA_MOE_LAYERED_KERNEL
#endif // MEGA_MOE_LAYERED_WAVE_ARCH92_H
