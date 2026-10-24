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
 * \file mega_moe_tiling_print.h
 * \brief MegaMoe Host Tiling 诊断打印。
 */

#ifndef MEGA_MOE_TILING_PRINT_H
#define MEGA_MOE_TILING_PRINT_H

#include "../../../op_kernel/arch35_92/mega_moe_tiling.h"
#include "../../../op_kernel/arch35_92/common/mega_moe_peermem.h"
#include "mc2_log.h"

namespace optiling {
namespace {

using namespace MegaMoeImpl;

/*
 * 输出各阶段的 UB 分批及缓冲区配置。
 */
static void PrintMegaMoeBufferConfigs(const MegaMoeTilingData* tilingData, const char* nodeName)
{
    const auto& dispatchConfig = tilingData->dispatchBufferConfig;
    OP_LOGD(nodeName, "dispatch: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, copyBufferBytes=%u",
            dispatchConfig.routeItemsPerBatch, dispatchConfig.routeBatchCount, dispatchConfig.bufferCount,
            dispatchConfig.copyBufferBytes);

    const auto& sendMaskConfigWithExtraExpert = tilingData->sendMaskConfigForCoreWithExtraExpert;
    const auto& sendMaskConfigWithoutExtraExpert = tilingData->sendMaskConfigForCoreWithoutExtraExpert;
    OP_LOGD(nodeName, "sendMask: coreCountWithExtraExpert=%u", tilingData->sendMaskCoreCountWithExtraExpert);
    OP_LOGD(nodeName,
            "sendMaskWithExtraExpert: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, bufferBytes=%u",
            sendMaskConfigWithExtraExpert.routeItemsPerBatch, sendMaskConfigWithExtraExpert.routeBatchCount,
            sendMaskConfigWithExtraExpert.bufferCount, sendMaskConfigWithExtraExpert.bufferBytes);
    OP_LOGD(nodeName,
            "sendMaskWithoutExtraExpert: routeItemsPerBatch=%d, routeBatchCount=%d, bufferCount=%d, bufferBytes=%u",
            sendMaskConfigWithoutExtraExpert.routeItemsPerBatch, sendMaskConfigWithoutExtraExpert.routeBatchCount,
            sendMaskConfigWithoutExtraExpert.bufferCount, sendMaskConfigWithoutExtraExpert.bufferBytes);

    const auto& unpermuteFullChunkConfig = tilingData->unpermuteConfigForFullTokenChunk;
    const auto& unpermuteTailChunkConfig = tilingData->unpermuteConfigForTailTokenChunk;
    OP_LOGD(nodeName, "unpermute: fullTokenChunkCoreCount=%u", tilingData->unpermuteFullTokenChunkCoreCount);
    OP_LOGD(nodeName,
            "unpermuteFullChunk: tokensPerBatch=%d, inputBufferCount=%d, bf16SlotElements=%u, fp32SlotElements=%u, "
            "weightBufferBytes=%u, conversionBufferBytes=%u",
            unpermuteFullChunkConfig.tokensPerBatch, unpermuteFullChunkConfig.inputBufferCount,
            unpermuteFullChunkConfig.bf16SlotElementCount, unpermuteFullChunkConfig.fp32SlotElementCount,
            unpermuteFullChunkConfig.topKWeightsBufferBytes, unpermuteFullChunkConfig.topKWeightsConversionBufferBytes);
    OP_LOGD(nodeName,
            "unpermuteTailChunk: tokensPerBatch=%d, inputBufferCount=%d, bf16SlotElements=%u, fp32SlotElements=%u, "
            "weightBufferBytes=%u, conversionBufferBytes=%u",
            unpermuteTailChunkConfig.tokensPerBatch, unpermuteTailChunkConfig.inputBufferCount,
            unpermuteTailChunkConfig.bf16SlotElementCount, unpermuteTailChunkConfig.fp32SlotElementCount,
            unpermuteTailChunkConfig.topKWeightsBufferBytes, unpermuteTailChunkConfig.topKWeightsConversionBufferBytes);
}

/*
 * 输出问题规模、执行模式以及各阶段自适应缓冲区配置，供 tiling 诊断使用。
 */
void PrintMegaMoeTilingData(const MegaMoeTilingData* tilingData, const char* nodeName)
{
    OP_TILING_CHECK(tilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(nodeName, "tilingData"), return);
    OP_LOGD(nodeName, "========== MegaMoeTilingData ==========");

    // 问题规模、专家拓扑及执行模式。
    OP_LOGD(nodeName,
            "shape: bs=%ld, numMaxTokensPerRank=%ld, h=%ld, hiddenDim=%ld, topK=%ld, maxOutputSize=%ld, "
            "isPerExpertWeightTensor=%d",
            tilingData->compute.bs, tilingData->compute.numMaxTokensPerRank, tilingData->compute.h,
            tilingData->compute.hiddenDim, tilingData->compute.topK, tilingData->compute.maxOutputSize,
            tilingData->isPerExpertWeightTensor);
    OP_LOGD(nodeName,
            "topology: moeExpertPerRank=%ld, sharedExpertNum=%ld, epWorldSize=%ld, aicNum=%ld, blockAivNum=%u, "
            "blockNumPerEP=%u, topoType=%ld, rankNumPerServer=%u",
            tilingData->compute.moeExpertPerRank, tilingData->compute.sharedExpertNum, tilingData->compute.epWorldSize,
            tilingData->compute.aicNum, tilingData->blockAivNum, tilingData->blockNumPerEP,
            tilingData->compute.topoType, tilingData->rankNumPerServer);
    OP_LOGD(nodeName,
            "mode: moeGmmMode=%u, sharedGmmMode=%u, isSharedQuantIndependent=%u, combineQuantMode=%ld, "
            "clampLimit=%f",
            static_cast<uint32_t>(tilingData->moeGmmMode), static_cast<uint32_t>(tilingData->sharedGmmMode),
            static_cast<uint32_t>(tilingData->compute.sharedInputIndependent), tilingData->combineQuantMode,
            tilingData->clampLimit);
    OP_LOGD(nodeName, "combineSync: slotCountPerExpert=%ld", tilingData->compute.combineSyncSlotCountPerExpert);
    OP_LOGD(nodeName, "dedup: dedupMode=%d, topkWeightsPrefetch=%d", tilingData->compute.dedupMode,
            tilingData->topkWeightsPrefetch);
    OP_LOGD(nodeName, "topkWeightsPrefetch is %d", tilingData->topkWeightsPrefetch);
    OP_LOGD(nodeName, "preQuant: tileTokens=%u, stageBytes=%u", tilingData->preQuantTileTokens,
            tilingData->preQuantStageBytes);
    OP_LOGD(nodeName, "mGroupsPerWave is %ld", tilingData->compute.mGroupsPerWave);
    PrintMegaMoeBufferConfigs(tilingData, nodeName);
}

/*
 * 输出 CCL 计算区域的偏移和总大小，供主机与核内布局核对使用。
 */
template <typename WorkspaceLayoutType>
void PrintWorkspaceLayout(const WorkspaceLayoutType* layout, const char* nodeName)
{
    OP_LOGD(nodeName, "dispatchRevDataOffset:         %ld\n", layout->dispatchRevDataOffset);
    OP_LOGD(nodeName, "dispatchRevScaleOffset:        %ld\n", layout->dispatchRevScaleOffset);
    OP_LOGD(nodeName, "activationQuantDataOffset:     %ld\n", layout->activationQuantDataOffset);
    OP_LOGD(nodeName, "activationQuantScaleOffset:    %ld\n", layout->activationQuantScaleOffset);
    OP_LOGD(nodeName, "expertRecvTokenCountOffset:    %ld\n", layout->expertRecvTokenCountOffset);
    OP_LOGD(nodeName, "metaInfoOffset:                %ld\n", layout->metaInfoOffset);
    OP_LOGD(nodeName, "flagActivationToGmm2Offset:    %ld\n", layout->flagActivationToGmm2Offset);
    OP_LOGD(nodeName, "sharedActivationToGmm2Offset:  %ld\n", layout->sharedActivationToGmm2Offset);
    OP_LOGD(nodeName, "flagDispatchToGmm1Offset:      %ld\n", layout->flagDispatchToGmm1Offset);
    OP_LOGD(nodeName, "flagSendCntCalToUpdParamsOffset: %ld\n", layout->flagSendCntCalToUpdParamsOffset);
    OP_LOGD(nodeName, "flagGmmToEpilogueOffset:       %ld\n", layout->flagGmmToEpilogueOffset);
    OP_LOGD(nodeName, "gmm2ReadyOffset:               %ld\n", layout->gmm2ReadyOffset);
    OP_LOGD(nodeName, "gmm2CombineSyncCounterOffset:  %ld\n", layout->gmm2CombineSyncCounterOffset);
    OP_LOGD(nodeName, "gmm2MmadResOffset:             %ld\n", layout->gmm2MmadResOffset);
    OP_LOGD(nodeName, "sharedExpertInputOffset:       %ld\n", layout->sharedExpertInputOffset);
    OP_LOGD(nodeName, "workspaceSize:                 %ld\n", layout->workspaceSize);
}

/*
 * 按 kernel 共用的 peermem 布局公式计算并输出各分区大小。
 */
void PrintPeermemInfo(const MegaMoeTilingData* tilingData, const char* nodeName)
{
    OP_LOGD(nodeName, "========== PeermemInfo ==========");
    int64_t exceptionDumpRegionSize = tilingData->compute.topoType == TOPO_TYPE_MTE ? EXCEPTION_DUMP_REGION_SIZE : 0;
    OP_LOGD(nodeName, "exceptionDumpRegionSize: {%ld}\n", exceptionDumpRegionSize);
    const PeermemInfo info(tilingData->comm);
    const int64_t dataOffset =
        tilingData->compute.topoType == TOPO_TYPE_MTE ? info.expertCountRecv.offset : info.maskRecv.offset;
    OP_LOGD(nodeName, "peermemDataOffset: {%ld}\n", dataOffset);
    OP_LOGD(nodeName, "maskRecvSize: {%ld}\n", info.maskRecv.size);
    OP_LOGD(nodeName, "expertCountRecvSize: {%ld}\n", info.expertCountRecv.size);
    OP_LOGD(nodeName, "dispatchRecordAreaSize: {%ld}\n",
            info.quantTokenScale.size + info.dispatchReceive.size + info.dispatchFlag.size);
    OP_LOGD(nodeName, "combineSendSize: {%ld}\n", info.combineSend.size);
    OP_LOGD(nodeName, "sourceReadySize: {%ld}\n", info.sourceReady.size);
    OP_LOGD(nodeName, "contributionSize: {%ld}\n", info.contribution.size);
    OP_LOGD(nodeName, "contributionReadySize: {%ld}\n", info.contributionReady.size);
    OP_LOGD(nodeName, "combineReceiveSize: {%ld}\n", info.combineReceive.size);
    OP_LOGD(nodeName, "total PeermemInfo Size: {%ld}\n", exceptionDumpRegionSize + info.totalSize);
}

} // namespace
} // namespace optiling

#endif // MEGA_MOE_TILING_PRINT_H
