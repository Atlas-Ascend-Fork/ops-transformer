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
 * \file mega_moe_layered_arch92.h
 * \brief Arch92 1C1V 的 URMA 实现，沿用 Arch35 A8W4 的 Wave 和 ACK 策略。
 */

#ifndef MEGA_MOE_LAYERED_ARCH92_H
#define MEGA_MOE_LAYERED_ARCH92_H

#include "../arch35_92/common/mega_moe_sync.h"
#include "kernel_operator.h"

#include "../arch35_92/common/mega_moe_hcomm_version.h"

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator_list_tensor_intf.h"

#include "../../../common/op_kernel/mc2_kernel_utils.h"
#include "../../../common/op_kernel/quantize_functions.h"

#include "../arch35_92/common/mega_moe_types.h"
#include "../arch35_92/common/mega_moe_gm_layout.h"
#include "../arch35_92/common/mega_moe_utils.h"
#include "../arch35_92/common/mega_moe_mxfp8_utils.h"
#include "../arch35_92/common/mega_moe_layered_comm.h"

#include "../arch35_92/blaze/epilogue/mega_moe_block_epilogue_activation_mx_quant.h"

#include "../arch35_92/stage/mega_moe_token_quant.h"
#include "../arch35_92/stage/mega_moe_expert_count.h"
#include "../arch35_92/mega_moe_vf/mega_moe_layered_vf.h"
#include "../arch35_92/stage/layered/mega_moe_layered_combine_send.h"
#include "stage/mega_moe_gmm1_activation_arch92.h"
#include "stage/mega_moe_gmm2_combine_arch92.h"
#include "common/mega_moe_layout_arch92.h"

namespace MegaMoeImpl {

using namespace AscendC;

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
using TupleShape = Shape<int64_t, int64_t, int64_t, int64_t>;

struct DispatchRowRange {
    int32_t beginRow = 0;
    int32_t endRow = 0;
};

struct DispatchRoundSlice {
    int32_t validRowCount = 0;
    int32_t sourceRowBegin = 0;
    int32_t copyRowCount = 0;
    int32_t gatheredOffset = 0;
};

#define TemplateMegaMoeLayeredTypeClass \
    typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode, \
        int32_t Weight1Format, int32_t Weight2Format, typename CommPolicy
#define TemplateMegaMoeLayeredTypeFunc \
    XType, OutputType, TopkWeightsType, Weight1Type, QuantMode, Weight1Format, Weight2Format, CommPolicy

template <TemplateMegaMoeLayeredTypeClass>
class MegaMoeLayeredWave;

template <typename XType, typename OutputType, typename TopkWeightsType, typename Weight1Type, int32_t QuantMode,
          int32_t Weight1Format, int32_t Weight2Format, typename CommPolicy = LayeredUrmaComm>
class MegaMoeLayered {
public:
    using Channel = typename CommPolicy::Channel;
    using BatchHandle = typename CommPolicy::BatchHandle;
    using CombineBatchState = CombineImpl::LayeredCombineBatchState<CommPolicy>;
    using MoeQuantConfig = QuantConfig<Weight1Type, QuantMode>;
    using QuantOutType = typename MoeQuantConfig::QuantOutType;
    using ActivationType = typename MoeQuantConfig::QuantStorageType;
    using QuantScaleOutType = typename MoeQuantConfig::QuantScaleType;
    using ActivationQuantOutType =
        std::conditional_t<MoeQuantConfig::AXW_MODE == AxWMode::A4W4, fp8_e4m3fn_t, QuantOutType>;
    __aicore__ inline explicit MegaMoeLayered(GM_ADDR context)
        : mc2Context_(reinterpret_cast<__gm__ Mc2MoeContext*>(context)),
          comm_(mc2Context_)
    {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2,
                                GM_ADDR xActiveMask, GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales,
                                GM_ADDR sharedWeight1, GM_ADDR sharedWeight2, GM_ADDR sharedWeightScales1,
                                GM_ADDR sharedWeightScales2, GM_ADDR yOut, GM_ADDR expertTokenNumsOut,
                                GM_ADDR workspaceGM, const MegaMoeTilingData* tilingData);

private:
    friend class MegaMoeLayeredWave<TemplateMegaMoeLayeredTypeFunc>;

    template <typename Wave>
    __aicore__ inline void ProcessWave(Wave& wave);

    __aicore__ inline void PrepareWithoutSharedExpert();
    __aicore__ inline void PrepareSharedExpertInput(uint32_t maskWorkerCount);
    __aicore__ inline void DrainAndUnlockMaskChannels(const AivJobContext& job);
    __aicore__ inline void PrepareWithSharedExpert(Gmm1ActivationSync& sharedGmm1ActivationSync);

    __aicore__ inline bool IsCombineServerDedupEnabled() const
    {
        return IsServerCombineDedupOn(params_.tilingData->compute.dedupMode);
    }
    __aicore__ inline bool RecordHasWeights(bool TopkWeightsPrefetch) const
    {
        return TopkWeightsPrefetch || IsCombineServerDedupEnabled();
    }
    __aicore__ inline uint64_t GatherCombineTokenIds(uint32_t originServer, const WorkRange& tokenRange,
                                                     const LocalTensor<int32_t>& tokenBitmaps,
                                                     const LocalTensor<int32_t>& tokenIds,
                                                     const LocalTensor<int32_t>& selectedTokenIds,
                                                     const LocalTensor<int32_t>& selectedTokenBitmaps,
                                                     const LocalTensor<uint8_t>& compareMask);
    __aicore__ inline void BuildServerCombineLists();
    __aicore__ inline bool IsServerContributionsReady(uint32_t originServer,
                                                      const LocalTensor<uint32_t>& tokenIdsTensor,
                                                      const LocalTensor<uint32_t>& expectedTopkBitmapsTensor,
                                                      uint32_t batchTokenCount, uint32_t workUbOffset) const;
    __aicore__ inline uint64_t WaitForServerCombineTokenCounts(const LocalTensor<uint32_t>& tokenCountsTensor,
                                                               uint64_t tokenListStrideBytes) const;
    __aicore__ inline void ProcessCombineRelay();
    __aicore__ inline void ReduceRelayContributions();
    __aicore__ inline void ReduceRelayContributionsInRange(const WorkRange& tokenRange,
                                                           const LocalTensor<uint32_t>& tokenCountsTensor,
                                                           const LocalTensor<uint32_t>& tokenIdsTensor,
                                                           const LocalTensor<uint32_t>& expectedTopkBitmapsTensor,
                                                           uint64_t tokenListStrideBytes, uint32_t workUbOffset);
    __aicore__ inline void ReduceRelayContributionsForBatch(uint32_t originServer, const WorkRange& batchRange,
                                                            __gm__ uint32_t* tokenIdsGlobalTensor,
                                                            __gm__ uint32_t* expectedTopkBitmapsGlobalTensor,
                                                            const LocalTensor<uint32_t>& tokenIdsTensor,
                                                            const LocalTensor<uint32_t>& expectedTopkBitmapsTensor,
                                                            uint32_t workUbOffset);
    template <bool IsTopkWeightsPrefetch>
    __aicore__ inline void ReduceRelayContributionsForToken(uint32_t originServer, uint32_t tokenIdx,
                                                            uint32_t expectedTopkBitmap, uint32_t workUbOffset);
    template <bool IsTopkWeightsPrefetch>
    __aicore__ inline void AccumulateRelayContributions(uint32_t expectedTopkBitmap,
                                                        __gm__ bfloat16_t* contributionGlobalTensor,
                                                        const LocalTensor<bfloat16_t>& contributionTensor,
                                                        const LocalTensor<float>& inputTensor,
                                                        const LocalTensor<float>& sumTensor,
                                                        const LocalTensor<float>& topkWeightsTensor);
    __aicore__ inline void SendServerCombineResults();
    __aicore__ inline void SendRelayResultsToTargetRank(uint32_t targetServer, uint32_t targetRank, uint32_t count,
                                                        uint32_t batchTokens, __gm__ uint32_t* tokenIdsGlobalTensor,
                                                        const LocalTensor<uint32_t>& tokenIdsTensor);
    __aicore__ inline void UnpermuteServerCombine();
    __aicore__ inline void InitTopology(const MegaMoeTilingData* tilingData);
    __aicore__ inline void InitDispatchLayout(const MegaMoeTilingData* tilingData);
    __aicore__ inline void InitWorkspace(GM_ADDR workspaceGM, const MegaMoeTilingData* tilingData);
    __aicore__ inline uint32_t InitResetBatch();
    __aicore__ inline void InitSendMaskRounds(uint32_t resetTensorSize, uint32_t countTensorSize);
    __aicore__ inline void InitDispatchRounds(uint32_t fixedBefore, uint32_t dispatchFixedCost);
    __aicore__ inline uint32_t InitDispatchIndexBuffers(uint32_t fixedBefore);
    __aicore__ inline void InitDispatchReceiveScratch(uint32_t copyBufferUbOffset, uint32_t copyBufferTotalBytes,
                                                      uint32_t expertCountBufferBytes);
    __aicore__ inline GM_ADDR GetSendMaskSlot(uint32_t expertId) const;
    __aicore__ inline void UpdateLocalRelayWeight(int32_t copyIdx, int32_t outIdx, TEventID eventId);
    __aicore__ inline void UnpermuteTokenChunk(const WorkRange& tokenRange,
                                               const MegaMoeUnpermuteBufferConfig& bufferConfig,
                                               uint32_t& inputBufferIdx, uint32_t& readySharedGroup);
    __aicore__ inline void LoadUnpermuteExpertInput(__gm__ bfloat16_t* expandedX, int32_t tokenIdx, int32_t expId,
                                                    TEventID event, __ubuf__ bfloat16_t* dataInBf16,
                                                    __ubuf__ float* dataInFp32);
    __aicore__ inline void AccumulateUnpermuteExperts(__gm__ bfloat16_t* expandedX, int32_t tokenIdx, int32_t localIdx,
                                                      const MegaMoeUnpermuteBufferConfig& bufferConfig,
                                                      uint32_t& inputBufferIdx);
    __aicore__ inline void LoadLocalRelayToken(uint32_t srcServer, uint32_t tokenIndex, int32_t bufferIdx,
                                               uint32_t copyInNum, TEventID eventId);
    __aicore__ inline uint32_t GetChannelOwnerBlock(uint32_t rank) const;
    __aicore__ inline bool IsChannelOwner(uint32_t rank) const;
    __aicore__ inline void PrepareExpertTokenCountTable();
    __aicore__ inline void EnsureDispatchSourceReady(uint32_t relayRank, uint32_t srcServer);
    __aicore__ inline void DispatchBuffInit();
    __aicore__ inline void DispatchPrepareBuffInit();
    __aicore__ inline void SendAndQuantBuffInit(uint32_t maskWorkerCount);
    __aicore__ inline MegaMoeUnpermuteBufferConfig UnpermuteBuffInit();
    __aicore__ inline void ResetState();
    __aicore__ inline void CopyExpertMaskCounts(const WorkRange& expertRange);
    __aicore__ inline void ComputeExpertMasks(const WorkRange& expertRange);
    __aicore__ inline void SendExpertMasks(const AivJobContext& job);
    __aicore__ inline void PrepareDispatchReceive();
    __aicore__ inline DispatchRowRange CalcDispatchExpertSlice(uint32_t expertIdx, uint32_t expertBegin,
                                                               const ExpertTokenRange& range);
    __aicore__ inline uint64_t LoadAndGatherDispatchRound(__gm__ uint8_t* rankMaskBasePtr, uint32_t roundIdx);
    __aicore__ inline void CopyLocalDispatchRound(uint32_t expertIdx, uint32_t srcRank, int32_t rowStart,
                                                  int32_t copyBegin, int32_t roundCopyCount, int32_t gatheredOffset);
    __aicore__ inline void AppendRemoteDispatchRound(uint32_t relayRank, uint32_t expertIdx, uint32_t srcServer,
                                                     uint32_t srcRank, int32_t rowStart, int32_t copyBegin,
                                                     int32_t roundCopyCount, int32_t gatheredOffset,
                                                     uint32_t& remoteBatchTokenCount);
    __aicore__ inline void ReceiveDispatchTokenRange(const ExpertTokenRange& range);
    template <bool IsLocalServer>
    __aicore__ inline void ReceiveDispatchServerPhase(const ExpertTokenRange& range);
    template <bool IsLocalServer>
    __aicore__ inline void ReceiveDispatchExpertFromServers(uint32_t relayRank, uint32_t expertIdx,
                                                            const ExpertTokenRange& range,
                                                            uint32_t& remoteBatchTokenCount);
    template <bool IsLocalServer>
    __aicore__ inline void ReceiveDispatchFromSourceRank(uint32_t relayRank, uint32_t expertIdx, uint32_t srcServer,
                                                         __gm__ uint8_t* expertMaskBase,
                                                         const DispatchRowRange& expertRows,
                                                         uint32_t& remoteBatchTokenCount);
    __aicore__ inline uint32_t FindDispatchSegmentEnd(uint32_t segmentStart, uint32_t batchTokenCount);
    __aicore__ inline void ReceiveRemoteDispatchBatch(uint32_t relayRank, uint32_t batchTokenCount);
    __aicore__ inline void PublishDispatchRows(uint32_t expertIdx, int32_t globalRowStart, int32_t rowCount);
    __aicore__ inline void UpdateGmm1GlobalBuffer(GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state,
                                                  uint32_t rowOffsetInExpert = 0U);
    __aicore__ inline void UpdateGmm2GlobalBuffer(GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state,
                                                  uint32_t rowOffsetInExpert = 0U);
    template <AddrUpdateMode Mode>
    __aicore__ inline void UpdateSharedGlobalBuffer(GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state);
    __aicore__ inline void Unpermute();
    __aicore__ inline void InitCombineBuffers();
    __aicore__ inline void ProcessCombineTokenRange(const ExpertLoopState& waveBeginState,
                                                    const ExpertTokenRange& range);
    __aicore__ inline int32_t AdvanceRankSyncRound();
    template <bool ReuseChannelLock>
    __aicore__ inline void CrossRankSync(const AivJobContext& syncJob);
    template <bool ReuseChannelLock>
    __aicore__ inline void CrossRankSync(const AivJobContext& syncJob, int32_t syncRound);
    __aicore__ inline void ExpertTokenNumCopyOut();
    __aicore__ inline void CopyLocalDispatchTokens(int32_t dstRow, int32_t srcRank, int32_t tokenOffset,
                                                   int32_t tokenCount);
    __aicore__ inline void PublishLocalDispatchRelayCount();
    __aicore__ inline GM_ADDR PrepareDispatchReadySource(uint32_t targetServer);
    __aicore__ inline void SendDispatchToRemoteServer(uint32_t targetServer, uint32_t relayRank);
    __aicore__ inline void StartInterServerDispatch();
    __aicore__ inline uint64_t DispatchBulkReadyFlagOffset(uint32_t sourceServer) const;
    template <bool CombineServerDedup>
    __aicore__ inline void BuildDispatchRelayCandidates(const LocalTensor<int32_t>& topkIds,
                                                        const LocalTensor<uint32_t>& masks, uint32_t tokenCount,
                                                        const WorkRange& serverRange);
    __aicore__ inline void EnqueueDispatchRelayCandidates(const LocalTensor<uint32_t>& masks,
                                                          const WorkRange& tokenRange, const WorkRange& serverRange);
    __aicore__ inline uint32_t TransposeDispatchRelayBitmaps(const LocalTensor<uint32_t>& masks,
                                                             const LocalTensor<uint32_t>& transposedMasks,
                                                             uint32_t tokenCount);
    __aicore__ inline void PublishDispatchRelayBitmaps(const LocalTensor<uint32_t>& masks, const WorkRange& tokenRange,
                                                       const WorkRange& serverRange, uint32_t rowElements);
    __aicore__ inline void PrepareDispatchRelayQueues(const AivJobContext& job);
    template <bool CombineServerDedup>
    __aicore__ inline void BuildDispatchRelayQueues(const LocalTensor<int32_t>& topkIds,
                                                    const LocalTensor<uint32_t>& masks,
                                                    const LocalTensor<uint32_t>& transposedMasks,
                                                    const WorkRange& tokenRange, uint32_t batchTokens);
    __aicore__ inline void EnqueueDispatchRelayToken(uint32_t targetServer, uint32_t tokenIdx);
    __aicore__ inline void LoadTokenFromLocalRelay(uint32_t srcServer, uint32_t tokenIndex, int32_t bufferIdx,
                                                   uint32_t copyInNum);
    __aicore__ inline void CopyTokensFromLocalRelay(int32_t rowDstOffsetInCore, uint32_t srcServer, int32_t copyNum,
                                                    int64_t widthA, int64_t widthAScale, uint32_t copyInNum);
    __aicore__ inline void QuantizeTokenInUb(__ubuf__ bfloat16_t* input, __ubuf__ ActivationType* output,
                                             __ubuf__ uint16_t* scratch);
    __aicore__ inline void QuantizeLocalTokensToRelay();
    __aicore__ inline uint64_t DispatchRelayQueueServerOffset(uint32_t targetServer) const;
    __aicore__ inline uint64_t RelayTokenOffset(uint32_t sourceServer, uint32_t tokenId) const;
    __aicore__ inline uint64_t RelayFlagOffset(uint32_t sourceServer, uint32_t tokenId) const;
    __aicore__ inline void ProcessSharedExpertGmm1Loop(GMMAddrInfo& sharedGmm1AddrInfo,
                                                       ExpertLoopState& sharedGmm1State);
    __aicore__ inline void ProcessSharedExpertGmm1(Gmm1ActivationSync& sharedGmm1ActivationSync);
    __aicore__ inline void ProcessSharedExpertGmm2Loop(GMMAddrInfo& sharedGmm2AddrInfo,
                                                       ExpertLoopState& sharedGmm2State);
    __aicore__ inline void ProcessSharedExpertGmm2();
    __aicore__ inline void FinishCombineAndSync();
    __aicore__ inline void WaitForSharedExpertResult(uint32_t sharedExpertIdx, uint32_t tokenIdx);
    __aicore__ inline void UnpermuteSharedExpert(int32_t tokenIdx, uint32_t& readySharedGroup,
                                                 const MegaMoeUnpermuteBufferConfig& bufferConfig,
                                                 uint32_t& inputBufferIdx);
    __aicore__ inline void LoadTopkWeightsToUb(__ubuf__ ActivationType* xOutTensor, int32_t currentOffset,
                                               int32_t index, TEventID event);
    template <bool IsShared, typename Epilogue, typename... LayoutArgs>
    __aicore__ inline void RunGmm1WithEpilogue(Epilogue& epilogue, const GMMAddrInfo& gmmAddrInfo,
                                               const ExpertLoopState& state, uint32_t expertIdx,
                                               uint32_t rowOffsetInExpert = 0U, const LayoutArgs&... layoutArgs);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithActivationQuant(const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state,
                                                          uint32_t expertIdx, uint32_t rowOffsetInExpert = 0U);
    template <bool IsShared = false>
    __aicore__ inline void GroupMatmulWithCombine(const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state,
                                                  uint32_t expertTokenCount, uint32_t rowOffsetInExpert = 0U);
    __aicore__ inline bool GetCombineRankRange(uint32_t expertIdx, uint32_t dstRank, uint32_t& rowStart,
                                               uint32_t& tokenCount);
    template <bool CombineServerDedup>
    __aicore__ inline void CombineTokensOfExpert(const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& gmm2State,
                                                 uint32_t expertIdx, const WorkRange& expertRowRange,
                                                 CombineBatchState& batchState);
    template <bool CombineServerDedup>
    __aicore__ inline void ProcessCombineToTargetRank(ExpertLoopState waveBeginState, const ExpertTokenRange& range,
                                                      uint32_t targetRank);
    template <bool CombineServerDedup>
    __aicore__ inline void CombineTokensInRange(const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& gmm2State,
                                                uint32_t expertIdx, const WorkRange& rowRange,
                                                CombineBatchState& batchState);
    __aicore__ inline void LockOwnedRemoteChannels();
    __aicore__ inline void DrainAndUnlockOwnedRemoteChannels();

    __gm__ Mc2MoeContext* mc2Context_{nullptr};
    CommPolicy comm_;
    Params params_{};

    __gm__ int32_t* expertTokenNumsOut_ = nullptr;
    __gm__ int32_t* metadataGm_ = nullptr;
    __gm__ int32_t* expertRecvCountsGm_ = nullptr;
    // 完整前缀和在建表时备份到 GM，Dispatch 接收和 CopyOut 在 UB 被计算覆盖后从 GM 恢复。
    __gm__ int32_t* cumsumGm_ = nullptr;

    uint32_t m_ = 0;
    uint32_t k_ = 0;
    uint32_t activationOutputDivisor_ = ACTIVATION_N_HALF;
    // 专家循环状态与地址绑定的公共阶段配置，InitTopology 里一次填好。
    MoeStageCommonConfig commonConfig_{};
    uint32_t aicNum_ = 0;
    uint32_t topK_ = 0;
    uint32_t rankId_ = 0;
    uint32_t worldSize_ = 0;
    uint32_t rankNumPerServer_ = 0;
    uint32_t serverNum_ = 0;
    uint32_t serverId_ = 0;
    uint32_t rankIdInServer_ = 0;
    int64_t hiddenDim_ = 0;
    uint64_t maxOutputSize_ = 0;
    uint32_t startBlockIdx_ = 0;
    uint32_t mGroupsPerWave_ = 1U;
    // 由本核 AIV 在建表时计算，仅用于划分通信 wave。
    uint64_t gmmWaveCount_ = 0U;
    int32_t dispatchFlagSlotsPerExpert_ = 0;
    int32_t maxWavesPerExpert_ = 0;
    // 每个 AIC 配对一个 AIV，由该 AIV 负责通信和激活。
    uint32_t blockNum_ = GetBlockNum();
    uint32_t blockAivNum_ = GetBlockNum();
    uint32_t blockIdx_ = MegaMoeHardwarePolicy::GetAicBlockIdx();
    uint32_t aivCoreIdx_ = GetBlockIdx();
    uint32_t mxQuantScaleNumAlignPerToken_ = 0;
    uint32_t mxQuantTokenAlignBytes_ = 0;
    uint32_t mxQuantScaleAlignBytes_ = 0;
    uint32_t mxQuantTokenScaleAlignBytes_ = 0;
    uint32_t weightAlignBytes_ = 0;
    uint32_t ubBufferUsedAddr_ = 0;
    uint64_t sendTotalNum_ = 0;      // 本 Rank 真实 route 数：bs * topK
    uint64_t maskRouteCapacity_ = 0; // 对称 mask 容量：numMaxTokensPerRank * topK
    uint32_t maskAlignSize_ = 0;
    uint32_t maskSlotSize_ = 0;      // 单个 win 槽位 = maskAlignSize_(mask) + 32B(count)
    uint32_t roundSendTotalNum_ = 0; // 分轮次：每轮处理的 token*topK 数量（256 对齐保证 GM 写 32B 对齐）
    uint32_t roundCompareCount_ = 0;          // 分轮次：每轮 CompareScalar 的元素数
    uint32_t roundMaskAlignSize_ = 0;         // 分轮次：每轮部分 mask 的字节大小（32B 对齐）
    uint32_t roundMaskSlotSize_ = 0;          // 分轮次：每轮部分 [mask|count] 槽位字节大小
    uint32_t totalRounds_ = 0;                // 分轮次：总轮数 = CeilDiv(sendTotalNum, roundSendTotalNum)
    uint32_t dispatchRoundSendTotalNum_ = 0;  // dispatch分轮次：每轮处理的 token*topK 数量（256 对齐）
    uint32_t dispatchTotalRounds_ = 0;        // dispatch分轮次：容量上界总轮数
    uint32_t dispatchRoundMaskAlignSize_ = 0; // dispatch分轮次：每轮mask的字节大小（32B对齐）
    uint32_t relayRecordBytes_ = 0;
    uint64_t dispatchRelayQueueBytesPerServer_ = 0;
    uint64_t dispatchRelayFlagSnapshotBytesPerBlock_ = 0;
    int64_t combineUbTensorSize_ = 0; // combineUbTensor 的大小（元素数）
    uint32_t topKWeightsTempAddr_ = 0;
    uint32_t sharedExpertNum_ = 0;
    uint32_t moeExpertPerRank_ = 0;

    static constexpr uint32_t A_ELEMS_PER_BYTE = MoeQuantConfig::A_ELEMS_PER_BYTE;
    static constexpr uint32_t B_ELEMS_PER_BYTE = MoeQuantConfig::B_ELEMS_PER_BYTE;
    static constexpr uint32_t C_ELEMS_PER_BYTE = PackedElementTraits<ActivationQuantOutType>::ELEMENTS_PER_BYTE;
    // 来源就绪位图每个 rank 占一个 bit；按 512B 预留，使下移后的 WQE 区继续保持对齐。
    static constexpr uint32_t DISPATCH_READY_CACHE_RESERVE_BYTES = 512U;
    static constexpr int32_t DISPATCH_COPY_BUFFER_COUNT = 5;
    static constexpr uint32_t MX_QUANT_TEMP_UB_BYTES = 2U * 1024U;
    static constexpr uint32_t DISPATCH_MAX_TOPK = 32U;
    // 启用 topk 权重在激活量化阶段同步计算时，二级读取每 token 需要 data/scale/weight 三条 WQE。
    // 64 KiB 可容纳 1024 条 WQE，足够一次提交完整的 256-token 批次（最多 768 条 WQE）。
    static_assert(LAYERED_HCOMM_BATCH_WQE_CAPACITY >= L1_TILE_M_256 * 3U,
                  "Layered Hcomm UB must hold 256 three-WQE token records");
    // Wave 通信需要同时保留批量描述符和其他搬运张量，因此动态 UB 预算止于批量区起点。
    static constexpr uint32_t COMM_WORK_UB_LIMIT = ARCH92_LAYERED_WORK_UB_BYTES;
    static constexpr uint32_t SEND_MASK_HCOMM_UB_OFFSET = ARCH92_LAYERED_HCOMM_OFFSET;
    // AIV 私有持久存储；Dispatch/Combine 临时缓冲的上界不得进入该区域。
    __ubuf__ uint32_t* dispatchSourceReadyBitsUb_ = nullptr;
    // 一级按协议长度限制连续分块；二级按最多 256 条 metadata 组织批次。
    static constexpr uint32_t DISPATCH_RELAY_QUEUE_ENTRY_BYTES = ALIGN_32;
    static constexpr uint32_t DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY = L1_TILE_M_256;
    static constexpr uint32_t DISPATCH_BATCH_DST_ROW = 4U;
    static constexpr uint32_t DISPATCH_BATCH_EXPERT_ID = 5U;
    static constexpr uint32_t DISPATCH_BATCH_SRC_SERVER = 6U;
    __ubuf__ int32_t* topkIndexTensor_ = nullptr;
    __ubuf__ uint8_t* gatherMaskTensor_ = nullptr;
    __ubuf__ uint32_t* gatherMaskInt32Tensor_ = nullptr;
    __ubuf__ int32_t* validTopkIndexTensor_ = nullptr;
    __ubuf__ int32_t* cumsumInfoTensor_ = nullptr;
    __ubuf__ ActivationType* copyTmpTensors_[DISPATCH_COPY_BUFFER_COUNT]{}; // 5 路软流水：占用 EVENT_ID1..EVENT_ID5。
    __ubuf__ int32_t* metaInfoTensor_ = nullptr;
    __ubuf__ bfloat16_t* xInTensor1_ = nullptr;
    __ubuf__ bfloat16_t* xInTensor2_ = nullptr;
    __ubuf__ ActivationType* xOutTensor1_ = nullptr;
    __ubuf__ ActivationType* xOutTensor2_ = nullptr;
    __ubuf__ uint16_t* mxTempTensor_ = nullptr;
    __ubuf__ int32_t* resetTensor_ = nullptr;
    int32_t resetBatchElementCount_ = 0;
    __ubuf__ int32_t* topkIdsTensor_ = nullptr;
    __ubuf__ uint8_t* sendMaskTensor_[DOUBLE_BUFFER]{}; // ComputeExpertMasks 位图搬出的 ping-pong 缓冲
    __ubuf__ int32_t* sendMaskCounts_ = nullptr;        // 每个任务一个 32B count 槽
    __ubuf__ int32_t* expertTokenNumsOutTensor_ = nullptr;
    __ubuf__ bfloat16_t* dataResTensor_ = nullptr;
    __ubuf__ float* dataResFp32Tensor_ = nullptr;
    __ubuf__ float* topKWeightsTensor_ = nullptr;
    __ubuf__ float* fp32ScaleTensor_ = nullptr;
    __ubuf__ bfloat16_t* bf16ScaleTensor_ = nullptr;

    // GMM1 固定按 256 行切块；权重前移时后处理可按 128 行细分。
    static constexpr uint32_t GMM1_TILE_M = L1_TILE_M_256;

    // GMM1 的每个 tile 固定为 [gate128, up128]，由同一 epilogue 消费。
    using BlockEpilogue = BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, L1_TILE_N>;
    using SharedBlockEpilogue = BlockEpilogueActivationMxQuant<ActivationQuantOutType, bfloat16_t, L1_TILE_N>;
    BlockEpilogue epilogueOp_;
    SharedBlockEpilogue sharedEpilogueOp_;
};

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitTopology(const MegaMoeTilingData* tilingData)
{
    m_ = static_cast<uint32_t>(tilingData->compute.bs);
    k_ = static_cast<uint32_t>(tilingData->compute.h);
    aicNum_ = static_cast<uint32_t>(tilingData->compute.aicNum);
    topK_ = static_cast<uint32_t>(tilingData->compute.topK);
    sendTotalNum_ = static_cast<uint64_t>(m_) * static_cast<uint64_t>(topK_);
    maskRouteCapacity_ = static_cast<uint64_t>(tilingData->compute.numMaxTokensPerRank) * static_cast<uint64_t>(topK_);
    worldSize_ = static_cast<uint32_t>(tilingData->compute.epWorldSize);
    moeExpertPerRank_ = static_cast<uint32_t>(tilingData->compute.moeExpertPerRank);
    mGroupsPerWave_ =
        tilingData->compute.mGroupsPerWave == 0 ? 1U : static_cast<uint32_t>(tilingData->compute.mGroupsPerWave);
    sharedExpertNum_ = static_cast<uint32_t>(tilingData->compute.sharedExpertNum);
    maxOutputSize_ = static_cast<uint64_t>(tilingData->compute.maxOutputSize);
    // 与 WorkspaceInfo 构造里 flagDispatchToGmm1Ptr 的分配公式保持一致。
    maxWavesPerExpert_ = static_cast<int32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(maxOutputSize_), static_cast<int64_t>(L1_TILE_M_256)));
    dispatchFlagSlotsPerExpert_ = maxWavesPerExpert_ * INT_CACHELINE;
    hiddenDim_ = tilingData->compute.hiddenDim;
    activationOutputDivisor_ = GetActivationOutputDivisor(tilingData->actMode);
    rankId_ = mc2Context_->epRankId;
    rankNumPerServer_ = tilingData->rankNumPerServer;
    if (rankNumPerServer_ == 0U || rankNumPerServer_ > worldSize_) {
        rankNumPerServer_ = worldSize_;
    }
    serverNum_ = Ops::Base::CeilDiv(worldSize_, rankNumPerServer_);
    serverId_ = rankId_ / rankNumPerServer_;
    rankIdInServer_ = rankId_ % rankNumPerServer_;
    for (int i = 0; i < worldSize_; i++) {
        g_winRankAddr_[i] = (GM_ADDR)mc2Context_->epHcclBuffer_[i];
    }
    commonConfig_.rankId = rankId_;
    commonConfig_.worldSize = worldSize_;
    commonConfig_.moeExpertPerRank = moeExpertPerRank_;
    commonConfig_.sharedExpertNum = sharedExpertNum_;
    commonConfig_.tokenNum = m_;
    commonConfig_.topK = topK_;
    commonConfig_.tokenHiddenDim = k_;
    commonConfig_.gmm1OutputDim = static_cast<uint32_t>(hiddenDim_);
    commonConfig_.activationOutputDivisor = activationOutputDivisor_;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitDispatchLayout(
    const MegaMoeTilingData* tilingData)
{
    // mask 是跨 Rank 共享的槽布局，几何必须与 PeermemInfo 一样按全卡容量计算。
    maskAlignSize_ = static_cast<uint32_t>(
        CalcDispatchMaskAlignSize(tilingData->compute.numMaxTokensPerRank, tilingData->compute.topK));
    // 每个 win 槽位再追加 32B 存 count（源卡 ComputeExpertMasks 同时算好）, 须与 PeermemInfo 的 maskSlotSize 一致。
    maskSlotSize_ = maskAlignSize_ + static_cast<uint32_t>(ALIGN_32);
    mxQuantScaleNumAlignPerToken_ = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenAlignBytes_ =
        Ops::Base::CeilAlign(static_cast<uint32_t>(k_ / A_ELEMS_PER_BYTE), static_cast<uint32_t>(ALIGN_256)) *
        sizeof(ActivationType);
    // 有效 scale 数仅用于量化；存储段须对齐，使后续权重的 UB 起点满足 32B 要求。
    mxQuantScaleAlignBytes_ = Ops::Base::CeilAlign(
        mxQuantScaleNumAlignPerToken_ * static_cast<uint32_t>(sizeof(uint8_t)), static_cast<uint32_t>(ALIGN_32));
    mxQuantTokenScaleAlignBytes_ =
        Ops::Base::CeilAlign(mxQuantTokenAlignBytes_ + mxQuantScaleAlignBytes_, static_cast<uint32_t>(ALIGN_32));
    if (RecordHasWeights(params_.tilingData->isTopkWeightsPrefetch)) {
        weightAlignBytes_ =
            Ops::Base::CeilAlign(static_cast<uint32_t>(topK_ * sizeof(float)), static_cast<uint32_t>(ALIGN_32));
        mxQuantTokenScaleAlignBytes_ += weightAlignBytes_;
    }
    relayRecordBytes_ =
        Ops::Base::CeilAlign(static_cast<uint64_t>(mxQuantTokenScaleAlignBytes_), static_cast<uint64_t>(ALIGN_512));
    dispatchRelayQueueBytesPerServer_ =
        static_cast<uint64_t>(ALIGN_32) + static_cast<uint64_t>(m_) * DISPATCH_RELAY_QUEUE_ENTRY_BYTES;
    // 保留 GM 快照原有行宽，来源级 ready 轮询只使用行首一个 uint64_t。
    uint64_t flagSnapshotBytes = static_cast<uint64_t>(DISPATCH_RECEIVE_BATCH_TOKEN_CAPACITY) * sizeof(uint64_t);
    dispatchRelayFlagSnapshotBytesPerBlock_ = Ops::Base::CeilAlign(flagSnapshotBytes, static_cast<uint64_t>(ALIGN_512));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::InitWorkspace(
    GM_ADDR workspaceGM, const MegaMoeTilingData* tilingData)
{
    (void)workspaceGM;
    params_.communication.base = g_winRankAddr_[rankId_];
    params_.workspace.layout =
        LayoutArch92::MakeWorkspaceLayout(tilingData->compute, tilingData->moeGmmMode, tilingData->sharedGmmMode,
                                          /*alignCombineRows=*/false, /*useLayeredUrmaLayout=*/true);
    params_.communication.layout = PeermemInfo(tilingData->comm, params_.workspace.layout.workspaceSize);
    params_.workspace.base = params_.communication.layout.workspace.get_ptr(params_.communication.base);
#if defined(ASCENDC_OOM) && ASCENDC_OOM == 1
    // HCCL 窗口: 逐 rank 登记整窗(ccl_buffer_size), 覆盖通信布局区与 workspace 区
    for (int i = 0; i < worldSize_; i++) {
        OOMRegisterRange(reinterpret_cast<GM_ADDR>(mc2Context_->epHcclBuffer_[i]), tilingData->cclBufferSize);
    }
#endif
    params_.tilingData = tilingData;
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    expertTokenNumsOut_ = (__gm__ int32_t*)params_.expertTokenNumsOutGmAddr;
    expertRecvCountsGm_ = workspace.routing.recvCounts.template get_ptr<int32_t>(buffer);
    metadataGm_ = workspace.routing.receivedTokenMetadata.template get_ptr<int32_t>(buffer);
    // 每个 block 备份完整专家表，每个专家包含 worldSize 个 rank 前缀值。
    uint64_t cumsumStride =
        Ops::Base::CeilAlign(static_cast<int64_t>(worldSize_ * moeExpertPerRank_ * sizeof(int32_t)), ALIGN_32);
    cumsumGm_ = reinterpret_cast<__gm__ int32_t*>(workspace.routing.recvPrefixBackup.get_ptr(buffer) +
                                                  cumsumStride * blockIdx_);
    epilogueOp_.Init({.yGmAddr = workspace.routed.gmm2Input.data.get_ptr(buffer),
                      .yScaleGmAddr = workspace.routed.gmm2Input.scale.get_ptr(buffer),
                      .clampLimit = tilingData->clampLimit,
                      .actMode = tilingData->actMode,
                      .actSubMode = tilingData->actSubMode,
                      .activationAlpha = tilingData->activationAlpha,
                      .activationBeta = tilingData->activationBeta,
                      .topkWeightsPrefetch = tilingData->isTopkWeightsPrefetch});
    InitDispatchLayout(tilingData);
}

// ========================
// Init：初始化 & 偏移计算
// ========================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Init(
    GM_ADDR x, GM_ADDR topkIds, GM_ADDR topkWeights, GM_ADDR weight1, GM_ADDR weight2, GM_ADDR xActiveMask,
    GM_ADDR weightScales1, GM_ADDR weightScales2, GM_ADDR scales, GM_ADDR sharedWeight1, GM_ADDR sharedWeight2,
    GM_ADDR sharedWeightScales1, GM_ADDR sharedWeightScales2, GM_ADDR yOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR workspaceGM, const MegaMoeTilingData* tilingData)
{
    if constexpr (g_coreType == AIV) {
        InitCopyPadding();
    }
    InitTopology(tilingData);
    params_.aGmAddr = x;
    params_.expertIdxGmAddr = topkIds;
    // 保留 TensorList 入口；每个 expert 更新地址时再按 2D-list/3D-stacked 布局解析。
    params_.bGmAddr = weight1;
    params_.b2GmAddr = weight2;
    params_.bScaleGmAddr = weightScales1;
    params_.b2ScaleGmAddr = weightScales2;
    params_.sharedBGmAddr = sharedWeight1;
    params_.sharedB2GmAddr = sharedWeight2;
    params_.sharedBScaleGmAddr = sharedWeightScales1;
    params_.sharedB2ScaleGmAddr = sharedWeightScales2;

    params_.y2GmAddr = yOut;
    params_.expertTokenNumsOutGmAddr = expertTokenNumsOut;
    params_.probsGmAddr = topkWeights;
    InitWorkspace(workspaceGM, tilingData);
}

// ===============================================================
// GroupMatmulWithActivationQuant：共享和 MoE 专家均通过原生 Generic GMM1 写 GM。
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared, typename Epilogue, typename... LayoutArgs>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::RunGmm1WithEpilogue(
    Epilogue& epilogue, const GMMAddrInfo& info, const ExpertLoopState& state, uint32_t expertIdx,
    uint32_t rowOffsetInExpert, const LayoutArgs&... layoutArgs)
{
    const bool prefetch = !IsShared && params_.tilingData->isTopkWeightsPrefetch;
    if constexpr (g_coreType == AIV) {
        const uint64_t rows = state.globalTokenStartIndex + rowOffsetInExpert;
        const uint64_t cols = Get<N_VALUE>(state.problemShape) / activationOutputDivisor_;
        const uint64_t scaleCols = Ops::Base::CeilDiv(cols, uint64_t(MXFP_DIVISOR_SIZE)) * MXFP_MULTI_BASE_SIZE;
        const typename Epilogue::BlockCoord vecBaseOffset{static_cast<int64_t>(rows * cols / C_ELEMS_PER_BYTE),
                                                          static_cast<int64_t>(rows * scaleCols)};
        epilogue.UpdateGlobalAddr(vecBaseOffset);
    }
    RunGmm1Generic<false, QuantOutType, ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType,
                   QuantScaleOutType, Weight1Format != FORMAT_ND, GMM1_TILE_M, IsShared, !IsShared>(
        prefetch, epilogue, params_, state.problemShape, info, startBlockIdx_, {blockIdx_, blockNum_},
        static_cast<uint32_t>(state.globalTokenStartIndex) + rowOffsetInExpert, expertIdx, activationOutputDivisor_,
        nullptr, false, layoutArgs...);
}

template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GroupMatmulWithActivationQuant(
    const GMMAddrInfo& gmmAddrInfo, const ExpertLoopState& state, uint32_t expertIdx, uint32_t rowOffsetInExpert)
{
    if constexpr (IsShared) {
        // 量化记录交织存储，GMM 输入和 scale 的行跨度分别以各自逻辑元素为单位。
        const StridedAConfig inputLayout{static_cast<uint32_t>(relayRecordBytes_) * A_ELEMS_PER_BYTE,
                                         static_cast<uint32_t>(relayRecordBytes_ / sizeof(QuantScaleOutType))};
        RunGmm1WithEpilogue<IsShared>(sharedEpilogueOp_, gmmAddrInfo, state, expertIdx, 0U, inputLayout);
    } else {
        RunGmm1WithEpilogue<IsShared>(epilogueOp_, gmmAddrInfo, state, expertIdx, rowOffsetInExpert);
    }
}

// ===============================================================
// GroupMatmulWithCombine：先按实现路径分发，再按 combine 模式分发。
// IsShared=true 时跳过 activation flag 等待和 Combine 后处理，供共享专家使用。
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
template <bool IsShared>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::GroupMatmulWithCombine(
    const GMMAddrInfo& info, const ExpertLoopState& state, uint32_t expertTokenCount, uint32_t rowOffsetInExpert)
{
    // 在 Generic 路径中，AIC 使用局部 MMAD 上下文，AIV 只推进 tile 游标。
    RunGmm2Generic<false, ActivationQuantOutType, Weight1Type, bfloat16_t, QuantScaleOutType, QuantScaleOutType,
                   Weight2Format != FORMAT_ND, true, GMM1_TILE_M, IsShared, !IsShared, false>(
        params_.tilingData->combineQuantMode, params_.tilingData->isTopkWeightsPrefetch, state.problemShape, info,
        startBlockIdx_, {blockIdx_, blockNum_}, nullptr, false, rowOffsetInExpert, nullptr, activationOutputDivisor_,
        expertTokenCount);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm1Loop(
    GMMAddrInfo& sharedGmm1AddrInfo, ExpertLoopState& sharedGmm1State)
{
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        UpdateExpertLoopState(sharedGmm1State, sharedIdx, m_);
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM1>(sharedGmm1AddrInfo, sharedGmm1State);
        GroupMatmulWithActivationQuant<true>(sharedGmm1AddrInfo, sharedGmm1State, sharedIdx);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm1(
    Gmm1ActivationSync& sharedGmm1ActivationSync)
{
    const auto& w = params_.workspace.layout;
    const auto base = params_.workspace.base;
    sharedEpilogueOp_.Init({.yGmAddr = w.shared.gmm2Input.data.get_ptr(base),
                            .yScaleGmAddr = w.shared.gmm2Input.scale.get_ptr(base),
                            .clampLimit = params_.tilingData->clampLimit,
                            .actMode = params_.tilingData->actMode,
                            .actSubMode = params_.tilingData->actSubMode,
                            .activationAlpha = params_.tilingData->activationAlpha,
                            .activationBeta = params_.tilingData->activationBeta});
    GMMAddrInfo info{};
    info.gmm1ActivationSync = &sharedGmm1ActivationSync;
    auto state = CreateExpertLoopState(commonConfig_);
    ProcessSharedExpertGmm1Loop(info, state);
    startBlockIdx_ = 0U;
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm2Loop(
    GMMAddrInfo& sharedGmm2AddrInfo, ExpertLoopState& sharedGmm2State)
{
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        UpdateExpertLoopState(sharedGmm2State, sharedIdx, m_);
        UpdateSharedGlobalBuffer<AddrUpdateMode::GMM2>(sharedGmm2AddrInfo, sharedGmm2State);
        GroupMatmulWithCombine<true>(sharedGmm2AddrInfo, sharedGmm2State,
                                     static_cast<uint32_t>(Get<M_VALUE>(sharedGmm2State.problemShape)));
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessSharedExpertGmm2()
{
    GMMAddrInfo info{};
    auto state = CreateExpertLoopState(commonConfig_);
    ProcessSharedExpertGmm2Loop(info, state);
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareWithoutSharedExpert()
{
    const AivJobContext maskJob{aivCoreIdx_, blockAivNum_};
    if constexpr (g_coreType == AIV) {
        SendAndQuantBuffInit(maskJob.totalJobs);
        ResetState();
        const auto expertRange = GetBalancedWorkRange(worldSize_ * moeExpertPerRank_, maskJob);
        if (expertRange.count != 0U) {
            ComputeExpertMasks(expertRange);
        }
        // 同一 MTE3 队列上的清零、mask 和 count 写回完成后，再允许 owner 提交通信。
        PipeBarrier<PIPE_ALL>();
        AllVectorBarrier();
        SendExpertMasks(maskJob);
        PipeBarrier<PIPE_ALL>();
    }
    AllCoreBarrier(); // AIC 等待同步标志清零完成

    if constexpr (g_coreType == AIV) {
        // 所有物理 AIV 在跨卡同步前完成各自 token 分片的量化。
        DispatchPrepareBuffInit();
        QuantizeLocalTokensToRelay();
        // Server Combine 仍需位图和有效 token 数量；
        // 普通整批发送路径不再构造或读取 token 队列。
        if (IsCombineServerDedupEnabled()) {
            PrepareDispatchRelayQueues(maskJob);
            GmDataBarrier();
        }
        // 首次跨卡同步前收口 SendMask 的发送完成，使同步通知代表本 AIV 的 mask 已发送。
        for (int rankIndex = aivCoreIdx_; rankIndex < worldSize_; rankIndex += blockAivNum_) {
            if (rankIndex != rankId_) {
                comm_.Drain(comm_.GetChannel(rankIndex));
            }
        }
        // 每个对端的跨卡同步通知由一个 AIV 发送，发送前须确保本卡所有 AIV 均已完成量化，
        // 使本 Server 的二级接收可以直接读取原始记录。
        AllVectorBarrier();
        const AivJobContext syncJob{aivCoreIdx_, blockAivNum_};
        CrossRankSync<true>(syncJob); // 复用本核 SendMask 的锁，同步结束后释放。
        AllVectorBarrier();           // 等本卡全部同步核完成，才进入 MoE wave。
    }
}

/* 全部 AIV 清理本轮状态，并量化各自负责的本卡输入。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareSharedExpertInput(
    uint32_t maskWorkerCount)
{
    SendAndQuantBuffInit(maskWorkerCount);
    ResetState();
    PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID3>(); // 清零搬出读完 UB 后，允许量化搬入覆盖。
    DispatchPrepareBuffInit();
    QuantizeLocalTokensToRelay();
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::PrepareWithSharedExpert(
    Gmm1ActivationSync& sharedGmm1ActivationSync)
{
    const AivJobContext job{aivCoreIdx_, blockAivNum_};
    if constexpr (g_coreType == AIV) {
        PrepareSharedExpertInput(job.totalJobs);
        GmDataBarrier();
        SyncInterArrive<PIPE_MTE3>(TOKEN_DATA_READY_FLAG);
    }
    if constexpr (g_coreType == AIC) {
        SyncIntraWait<PIPE_MTE2>(INPUT_PREPARE_DONE_FLAG);
        PipeNotify<PIPE_MTE2, PIPE_S>(SYNC_EVENT_ID3);
        ProcessSharedExpertGmm1(sharedGmm1ActivationSync);
        PipeWait<PIPE_MTE2, PIPE_S>(SYNC_EVENT_ID3);
    } else {
        SyncInterWait<PIPE_MTE3>(TOKEN_DATA_READY_FLAG);
        SyncIntraArrive<PIPE_MTE3>(INPUT_PREPARE_DONE_FLAG);
        PipeFence<PIPE_MTE3, PIPE_V, SYNC_EVENT_ID3>();
        const auto experts = GetBalancedWorkRange(worldSize_ * moeExpertPerRank_, job);
        if (experts.count != 0U) {
            ComputeExpertMasks(experts);
        }
        AllVectorBarrier();
        SendExpertMasks(job);
        PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID3>();
        if (IsCombineServerDedupEnabled()) {
            PrepareDispatchRelayQueues(job);
            GmDataBarrier();
            PipeFence<PIPE_MTE3, PIPE_MTE2, SYNC_EVENT_ID3>();
        }
        ProcessSharedExpertGmm1(sharedGmm1ActivationSync);
        DrainAndUnlockMaskChannels(job);
        AllVectorBarrier();
        CrossRankSync<false>(job);
        AllVectorBarrier();
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessCombineRelay()
{
    // 本卡所有 AIV 都提交原始贡献后，才开始等待贡献到齐。
    PipeBarrier<PIPE_ALL>();
    AllVectorBarrier();
    BuildServerCombineLists();
    ReduceRelayContributions();
    // 本核归约结束后，按目标 rank 等待对应的完成计数。
    SendServerCombineResults();
}

/* AIV 收尾通信，全部 AIV 会合后分担跨卡同步，确认 Combine 结果可读。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::FinishCombineAndSync()
{
    if constexpr (g_coreType == AIV) {
        PipeBarrier<PIPE_ALL>();
        ExpertTokenNumCopyOut();
        DrainAndUnlockOwnedRemoteChannels();
        AllVectorBarrier();
        CrossRankSync<false>({aivCoreIdx_, blockAivNum_});
        AllVectorBarrier();
    }
}

template <TemplateMegaMoeLayeredTypeClass>
template <typename Wave>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::ProcessWave(Wave& wave)
{
    int64_t oriOverflowMode = GetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(0);
    Gmm1ActivationSync sharedGmm1ActivationSync(0U, GmmEventPair::SHARED_GMM1);
    if (sharedExpertNum_ > 0) {
        PrepareWithSharedExpert(sharedGmm1ActivationSync);
    } else {
        PrepareWithoutSharedExpert();
    }

    // URMA Layered 的 A8W8、A4W4 和 A8W4 共用同一专家 Wave 状态机。
    Gmm1ActivationSync gmm1ActivationSync(0U, GmmEventPair::MOE_GMM1);
    wave.ProcessMoeExpertWave(gmm1ActivationSync);

    if constexpr (g_coreType == AIC) {
        if (sharedExpertNum_ > 0) {
            // 共享激活写回完成后，读取共享 GMM2 输入。
            sharedGmm1ActivationSync.EndSync();
            ProcessSharedExpertGmm2();
        }
    } else {
        if (IsCombineServerDedupEnabled()) {
            ProcessCombineRelay();
        }
        FinishCombineAndSync();
    }
    if constexpr (g_coreType == AIV) {
        // 全部 AIV 均分 token；共享结果的组就绪等待在 Unpermute 内部完成。
        Unpermute();
    }
    // 出口排空剩余 ACK；共享 GMM1 的 AIC 已在共享 GMM2 前完成回收。
    sharedGmm1ActivationSync.EndSync();
    gmm1ActivationSync.EndSync();
    SetCtrlSpr<OVERFLOW_MODE_CTRL, OVERFLOW_MODE_CTRL>(oriOverflowMode);
}

#endif

} // namespace MegaMoeImpl

#if defined(ENABLE_MEGA_MOE_LAYERED_KERNEL)
#include "stage/layered/mega_moe_layered_gmm_arch92.h"
#include "stage/layered/mega_moe_layered_sync_arch92.h"
#include "../arch35_92/stage/layered/mega_moe_layered_dispatch.h"
#include "../arch35_92/stage/layered/mega_moe_layered_combine.h"
#include "../arch35_92/stage/layered/mega_moe_layered_combine_relay.h"
#include "../arch35_92/stage/layered/mega_moe_layered_unpermute.h"
#include "stage/layered/mega_moe_layered_unpermute_arch92.h"
#endif

#undef TemplateMegaMoeLayeredTypeClass
#undef TemplateMegaMoeLayeredTypeFunc
#endif
