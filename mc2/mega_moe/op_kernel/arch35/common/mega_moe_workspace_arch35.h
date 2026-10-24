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
 * \file mega_moe_workspace_arch35.h
 * \brief
 */

#ifndef MEGA_MOE_WORKSPACE_ARCH35_H
#define MEGA_MOE_WORKSPACE_ARCH35_H

// peermem 窗口的布局与尺寸（含 HOST_DEVICE / GM_ADDR 宏与基础 include）。
#include "../../arch35_92/common/mega_moe_peermem.h"

namespace MegaMoeImpl {

// 保留原有名称，布局结果直接使用公共计算类型，不再重复适配 tiling。
using WorkspaceLayoutArch35 = CclComputeLayout;

struct WorkspaceInfoArch35 {
    GM_ADDR dispatchRevDataPtr{nullptr};
    GM_ADDR dispatchRevScalePtr{nullptr};
    GM_ADDR dispatchRevWeightsPtr{nullptr};
    GM_ADDR activationQuantDataPtr{nullptr};
    GM_ADDR activationQuantScalePtr{nullptr};
    GM_ADDR expertRecvTokenCountPtr{nullptr};
    GM_ADDR metaInfoPtr{nullptr};
    GM_ADDR flagActivationToGmm2Ptr{nullptr};
    GM_ADDR sharedActivationToGmm2Ptr{nullptr};
    GM_ADDR flagDispatchToGmm1Ptr{nullptr};
    GM_ADDR flagSendCntCalToUpdParamsPtr{nullptr};
    GM_ADDR flagGmmToEpiloguePtr{nullptr};
    GM_ADDR flagAiv1TopkValidIndexSyncPtr{nullptr};
    GM_ADDR gmm2ReadyPtr{nullptr};
    GM_ADDR gmm2CombineSyncCounterPtr{nullptr};
    GM_ADDR cumsumInfoPtr{nullptr};
    GM_ADDR gmm1MmadResPtr{nullptr};
    GM_ADDR gmm2MmadResPtr{nullptr};
    GM_ADDR sharedExpertResultPtr{nullptr};
    GM_ADDR sharedExpertGmm1OutPtr{nullptr};
    GM_ADDR sharedExpertInputPtr{nullptr};
    GM_ADDR sharedExpertInputDataPtr{nullptr};
    GM_ADDR sharedExpertInputScalePtr{nullptr};
    GM_ADDR sharedExpertActivationDataPtr{nullptr};
    GM_ADDR sharedExpertActivationScalePtr{nullptr};
    GM_ADDR gmm1TileStatusPtr{nullptr};
    GM_ADDR sharedExpertGmm2TileCounterPtr{nullptr};

    GM_ADDR dedupPrepassReadyPtr{nullptr};
    GM_ADDR dedupRowDescPtr{nullptr};

    GM_ADDR maskSlotPtr{nullptr};
    GM_ADDR dispatchRelaySendQueuePtr{nullptr};
    GM_ADDR dispatchRemoteReadyFlagSnapshotPtr{nullptr};
    GM_ADDR tokenIdsPtr{nullptr};
    GM_ADDR tokenBitmapsPtr{nullptr};
    GM_ADDR combineRelayCompletedCountPtr{nullptr};
    int64_t serverCombineResetBytes{0};

    int64_t flagResetElementCount{0};
    int64_t gmm1TileStatusElementCount{0};

private:
    // 取得 cclComputePtr 后，将有效偏移绑定为可访问地址。
    HOST_DEVICE static GM_ADDR ResolveWorkspaceAddress(GM_ADDR base, int64_t offset)
    {
        if (base == nullptr || offset < 0) {
            return nullptr;
        }
        return base + offset;
    }

public:
    HOST_DEVICE void Bind(GM_ADDR base, const WorkspaceLayoutArch35& layout)
    {
        dispatchRevDataPtr = ResolveWorkspaceAddress(base, layout.dispatchRevDataOffset);
        dispatchRevScalePtr = ResolveWorkspaceAddress(base, layout.dispatchRevScaleOffset);
        dispatchRevWeightsPtr = ResolveWorkspaceAddress(base, layout.dispatchRevWeightsOffset);
        activationQuantDataPtr = ResolveWorkspaceAddress(base, layout.activationQuantDataOffset);
        activationQuantScalePtr = ResolveWorkspaceAddress(base, layout.activationQuantScaleOffset);
        expertRecvTokenCountPtr = ResolveWorkspaceAddress(base, layout.expertRecvTokenCountOffset);
        metaInfoPtr = ResolveWorkspaceAddress(base, layout.metaInfoOffset);
        flagActivationToGmm2Ptr = ResolveWorkspaceAddress(base, layout.flagActivationToGmm2Offset);
        sharedActivationToGmm2Ptr = ResolveWorkspaceAddress(base, layout.sharedActivationToGmm2Offset);
        flagDispatchToGmm1Ptr = ResolveWorkspaceAddress(base, layout.flagDispatchToGmm1Offset);
        flagSendCntCalToUpdParamsPtr = ResolveWorkspaceAddress(base, layout.flagSendCntCalToUpdParamsOffset);
        flagGmmToEpiloguePtr = ResolveWorkspaceAddress(base, layout.flagGmmToEpilogueOffset);
        flagAiv1TopkValidIndexSyncPtr = ResolveWorkspaceAddress(base, layout.flagAiv1TopkValidIndexSyncOffset);
        gmm2ReadyPtr = ResolveWorkspaceAddress(base, layout.gmm2ReadyOffset);
        gmm2CombineSyncCounterPtr = ResolveWorkspaceAddress(base, layout.gmm2CombineSyncCounterOffset);
        cumsumInfoPtr = ResolveWorkspaceAddress(base, layout.cumsumInfoOffset);
        gmm1MmadResPtr = ResolveWorkspaceAddress(base, layout.gmm1MmadResOffset);
        gmm2MmadResPtr = ResolveWorkspaceAddress(base, layout.gmm2MmadResOffset);
        sharedExpertResultPtr = ResolveWorkspaceAddress(base, layout.sharedExpertResultOffset);
        sharedExpertGmm1OutPtr = ResolveWorkspaceAddress(base, layout.sharedExpertGmm1OutOffset);
        sharedExpertInputPtr = ResolveWorkspaceAddress(base, layout.sharedExpertInputOffset);
        sharedExpertInputDataPtr = ResolveWorkspaceAddress(base, layout.sharedExpertInputDataOffset);
        sharedExpertInputScalePtr = ResolveWorkspaceAddress(base, layout.sharedExpertInputScaleOffset);
        sharedExpertActivationDataPtr = ResolveWorkspaceAddress(base, layout.sharedExpertActivationDataOffset);
        sharedExpertActivationScalePtr = ResolveWorkspaceAddress(base, layout.sharedExpertActivationScaleOffset);
        gmm1TileStatusPtr = ResolveWorkspaceAddress(base, layout.gmm1TileStatusOffset);
        sharedExpertGmm2TileCounterPtr = ResolveWorkspaceAddress(base, layout.sharedExpertGmm2TileCounterOffset);
        dedupPrepassReadyPtr = ResolveWorkspaceAddress(base, layout.dedupPrepassReadyOffset);
        dedupRowDescPtr = ResolveWorkspaceAddress(base, layout.dedupRowDescOffset);
        maskSlotPtr = ResolveWorkspaceAddress(base, layout.maskSlotOffset);
        dispatchRelaySendQueuePtr = ResolveWorkspaceAddress(base, layout.dispatchRelaySendQueueOffset);
        dispatchRemoteReadyFlagSnapshotPtr =
            ResolveWorkspaceAddress(base, layout.dispatchRemoteReadyFlagSnapshotOffset);
        combineRelayCompletedCountPtr = ResolveWorkspaceAddress(base, layout.combineRelayCompletedCountOffset);
        tokenIdsPtr = ResolveWorkspaceAddress(base, layout.tokenIdsOffset);
        tokenBitmapsPtr = ResolveWorkspaceAddress(base, layout.tokenBitmapsOffset);
        serverCombineResetBytes = layout.serverCombineResetBytes;
        flagResetElementCount = layout.flagResetElementCount;
        gmm1TileStatusElementCount = layout.gmm1TileStatusElementCount;
    }
};

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WORKSPACE_ARCH35_H
