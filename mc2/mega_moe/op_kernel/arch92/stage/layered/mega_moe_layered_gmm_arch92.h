/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_GMM_ARCH92_H
#define MEGA_MOE_LAYERED_GMM_ARCH92_H

namespace MegaMoeImpl {

// ==================================================================================
// UpdateGmm1GlobalBuffer：绑定 GMM1 的输入、权重、输出、tile 状态与 flag 地址。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateGmm1GlobalBuffer(
    GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state, uint32_t rowOffsetInExpert)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    // 地址 = 专家前累计行数 + 专家内偏移；行相关项随行数线性增长，权重/flag 只依赖专家序号。
    const uint64_t rows = static_cast<uint64_t>(state.globalTokenStartIndex) + rowOffsetInExpert;
    const uint32_t expertIdx = state.expertIdx;
    const uint64_t n = static_cast<uint64_t>(Get<N_VALUE>(state.problemShape));
    const uint64_t k = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
    const uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    gmmAddrInfo.gmm1OutGlobal = workspace.routed.gmm1Output.get_ptr(buffer) + rows * n * sizeof(bfloat16_t);
    gmmAddrInfo.metaInfoGlobal = workspace.routing.receivedTokenMetadata.get_ptr(buffer);
    gmmAddrInfo.aGlobal =
        workspace.routed.gmm1Input.data.get_ptr(buffer) + rows * k / A_ELEMS_PER_BYTE * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal =
        workspace.routed.gmm1Input.scale.get_ptr(buffer) + rows * scaleK * sizeof(QuantScaleOutType);
    gmmAddrInfo.bGlobal =
        GetExpertWeightAddr<Weight1Type>(params_.bGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
                                         static_cast<uint64_t>(expertIdx) * (n * k / B_ELEMS_PER_BYTE));
    gmmAddrInfo.bScaleGlobal =
        GetExpertWeightAddr<QuantScaleOutType>(params_.bScaleGmAddr, params_.tilingData->isPerExpertWeightTensor,
                                               expertIdx, static_cast<uint64_t>(expertIdx) * (n * scaleK));
    gmmAddrInfo.activationToGmm2Flag = (__gm__ int32_t*)workspace.sync.activationReady.get_ptr(buffer) +
                                       static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_ +
                                       static_cast<uint64_t>(rowOffsetInExpert / GMM1_TILE_M) * INT_CACHELINE;
    // wave-grain dispatch-gmm1 flag: per-expert 步长是 dispatchFlagSlotsPerExpert_,而不是 INT_CACHELINE。
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)workspace.sync.dispatchReady.get_ptr(buffer) +
                                     static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_ +
                                     static_cast<uint64_t>(rowOffsetInExpert / GMM1_TILE_M) * INT_CACHELINE;
}

// ==================================================================================
// UpdateGmm2GlobalBuffer：绑定 GMM2 的激活输入、权重、输出与 flag 地址。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateGmm2GlobalBuffer(
    GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state, uint32_t rowOffsetInExpert)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    // 地址 = 专家前累计行数 + 专家内偏移；行相关项随行数线性增长，权重/flag 只依赖专家序号。
    const uint64_t rows = static_cast<uint64_t>(state.globalTokenStartIndex) + rowOffsetInExpert;
    const uint32_t expertIdx = state.expertIdx;
    const uint64_t k = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
    const uint64_t activationCols = static_cast<uint64_t>(Get<N_VALUE>(state.problemShape)) / activationOutputDivisor_;
    const uint64_t scaleN =
        Ops::Base::CeilDiv(activationCols, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
    gmmAddrInfo.gmm2OutGlobal = workspace.routed.gmm2Output.get_ptr(buffer) + rows * k * sizeof(bfloat16_t);
    gmmAddrInfo.metaInfoGlobal =
        workspace.routing.receivedTokenMetadata.get_ptr(buffer) + rows * META_INFO_SIZE * sizeof(int32_t);
    gmmAddrInfo.aGlobal = workspace.routed.gmm2Input.data.get_ptr(buffer) +
                          rows * activationCols / C_ELEMS_PER_BYTE * sizeof(ActivationType);
    gmmAddrInfo.aScaleGlobal =
        workspace.routed.gmm2Input.scale.get_ptr(buffer) + rows * scaleN * sizeof(QuantScaleOutType);
    gmmAddrInfo.bGlobal =
        GetExpertWeightAddr<Weight1Type>(params_.b2GmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
                                         static_cast<uint64_t>(expertIdx) * (k * activationCols / B_ELEMS_PER_BYTE));
    gmmAddrInfo.bScaleGlobal =
        GetExpertWeightAddr<QuantScaleOutType>(params_.b2ScaleGmAddr, params_.tilingData->isPerExpertWeightTensor,
                                               expertIdx, static_cast<uint64_t>(expertIdx) * (k * scaleN));
    gmmAddrInfo.gmm2CombineSyncCounter = (__gm__ int32_t*)workspace.urma.sync.combineCounters.get_ptr(buffer) +
                                         static_cast<uint64_t>(expertIdx) *
                                             params_.tilingData->compute.combineSyncSlotCountPerExpert *
                                             static_cast<uint64_t>(INT_CACHELINE);
    gmmAddrInfo.gmm2CombineLogicalCoreCount = blockNum_;
    gmmAddrInfo.activationToGmm2Flag = (__gm__ int32_t*)workspace.sync.activationReady.get_ptr(buffer) +
                                       static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_ +
                                       static_cast<uint64_t>(rowOffsetInExpert / GMM1_TILE_M) * INT_CACHELINE;
    gmmAddrInfo.dispatchToGmm1Flag = (__gm__ int32_t*)workspace.sync.dispatchReady.get_ptr(buffer) +
                                     static_cast<uint64_t>(expertIdx) * dispatchFlagSlotsPerExpert_ +
                                     static_cast<uint64_t>(rowOffsetInExpert / GMM1_TILE_M) * INT_CACHELINE;
}

// ==================================================================================
// UpdateSharedGlobalBuffer：共享 GMM1 复用本卡量化记录，其余中间结果使用 shared workspace。
// ==================================================================================
template <TemplateMegaMoeLayeredTypeClass>
template <AddrUpdateMode Mode>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UpdateSharedGlobalBuffer(
    GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state)
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const uint64_t rows = static_cast<uint64_t>(state.globalTokenStartIndex);
    const uint32_t expertIdx = state.expertIdx;

    if constexpr (Mode == AddrUpdateMode::GMM1) {
        const uint64_t n = static_cast<uint64_t>(Get<N_VALUE>(state.problemShape));
        const uint64_t k = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
        const uint64_t scaleK = Ops::Base::CeilDiv(k, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        gmmAddrInfo.aGlobal = params_.communication.layout.dispatchReceive.get_ptr(params_.communication.base) +
                              RelayTokenOffset(serverId_, 0U);
        gmmAddrInfo.aScaleGlobal = gmmAddrInfo.aGlobal + mxQuantTokenAlignBytes_;
        gmmAddrInfo.gmm1OutGlobal = workspace.shared.gmm1Output.get_ptr(buffer) + rows * n * sizeof(bfloat16_t);
        gmmAddrInfo.bGlobal =
            GetExpertWeightAddr<Weight1Type>(params_.sharedBGmAddr, params_.tilingData->isPerExpertWeightTensor,
                                             expertIdx, static_cast<uint64_t>(expertIdx) * (n * k / B_ELEMS_PER_BYTE));
        gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleOutType>(
            params_.sharedBScaleGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            static_cast<uint64_t>(expertIdx) * (n * scaleK));
    } else if constexpr (Mode == AddrUpdateMode::GMM2) {
        const uint64_t k = static_cast<uint64_t>(Get<K_VALUE>(state.problemShape));
        const uint64_t activationCols =
            static_cast<uint64_t>(Get<N_VALUE>(state.problemShape)) / activationOutputDivisor_;
        const uint64_t scaleN =
            Ops::Base::CeilDiv(activationCols, static_cast<uint64_t>(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        const uint32_t tokenGroupCount = Ops::Base::CeilDiv(m_, COMBINE_TOKEN_GROUP_SIZE);
        gmmAddrInfo.sharedExpertGmm2TileCounter =
            workspace.urma.sync.sharedGmm2Tiles.template get_ptr<int32_t>(buffer) +
            static_cast<uint64_t>(expertIdx) * tokenGroupCount * INT_CACHELINE;
        gmmAddrInfo.gmm2OutGlobal = workspace.shared.gmm2Output.get_ptr(buffer) + rows * k * sizeof(bfloat16_t);
        gmmAddrInfo.aGlobal = workspace.shared.gmm2Input.data.get_ptr(buffer) +
                              rows * activationCols / C_ELEMS_PER_BYTE * sizeof(ActivationType);
        gmmAddrInfo.aScaleGlobal =
            workspace.shared.gmm2Input.scale.get_ptr(buffer) + rows * scaleN * sizeof(QuantScaleOutType);
        gmmAddrInfo.bGlobal = GetExpertWeightAddr<Weight1Type>(
            params_.sharedB2GmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            static_cast<uint64_t>(expertIdx) * (k * activationCols / B_ELEMS_PER_BYTE));
        gmmAddrInfo.bScaleGlobal = GetExpertWeightAddr<QuantScaleOutType>(
            params_.sharedB2ScaleGmAddr, params_.tilingData->isPerExpertWeightTensor, expertIdx,
            static_cast<uint64_t>(expertIdx) * (k * scaleN));
    }
    gmmAddrInfo.activationToGmm2Flag = nullptr;
    gmmAddrInfo.dispatchToGmm1Flag = nullptr;
}

} // namespace MegaMoeImpl
#endif // MEGA_MOE_LAYERED_GMM_ARCH92_H
