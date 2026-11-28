/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "../../../../op_kernel/arch92/common/mega_moe_layout_arch92.h"

namespace MegaMoeUT {
namespace {
using namespace MegaMoeImpl;

CclComputeLayoutParams MakeUrmaParams()
{
    CclComputeLayoutParams params{};
    params.topoType = TOPO_TYPE_URMA;
    params.epWorldSize = 8;
    params.serverNum = 2;
    params.moeExpertPerRank = 8;
    params.numMaxTokensPerRank = 2048;
    params.bs = 2048;
    params.topK = 6;
    params.maxOutputSize = 98304;
    params.h = 7168;
    params.hiddenDim = 4096;
    params.aicNum = 64;
    params.combineSyncSlotCountPerExpert = 64;
    return params;
}

// 测试数据直接传给公共布局计算；Host 和 Torch 的入参校验在此前完成。
CclCommLayoutParams MakeUrmaCommParams(const CclComputeLayoutParams& params, bool fp4)
{
    CclCommLayoutParams comm{};
    comm.topoType = params.topoType;
    comm.epWorldSize = params.epWorldSize;
    comm.serverNum = params.serverNum;
    comm.moeExpertPerRank = params.moeExpertPerRank;
    comm.numMaxTokensPerRank = params.numMaxTokensPerRank;
    comm.topK = params.topK;
    comm.h = params.h;
    comm.yDtypeSize = SIZE_BF_16;
    comm.elemsPerByte = fp4 ? 2U : 1U;
    comm.dispatchStorage = {fp4 ? MemoryLayoutDetail::FP4_BITS : MemoryLayoutDetail::FP8_BITS};
    comm.serverCombineReduce = IsServerCombineDedupOn(params.dedupMode);
    comm.topkWeightsPrefetch = params.topkWeightsPrefetch || comm.serverCombineReduce;
    comm.isQuantCombine = params.combineQuant;
    return comm;
}
} // namespace

TEST(MegaMoeArch92UrmaLayout, KeepsFullGmm1AndFp8ActivationForAllExpertModes)
{
    for (uint8_t mode : {GMM_MODE_A8W8_ND, GMM_MODE_A8W8_NZ, GMM_MODE_A8W4_NZ, GMM_MODE_A4W4_ND, GMM_MODE_A4W4_NZ}) {
        for (int64_t shared : {0, 1, 4}) {
            for (uint32_t divisor : {1U, static_cast<uint32_t>(MemoryLayoutDetail::ACTIVATION_N_HALF)}) {
                auto params = MakeUrmaParams();
                params.sharedExpertNum = shared;
                params.activationOutputDivisor = divisor;
                const auto layout = LayoutArch92::MakeWorkspaceLayout(params, mode, mode, false, true);
                EXPECT_EQ(layout.routed.gmm1Input.data.sizeBytes,
                          params.maxOutputSize * params.h / (IsA4W4GmmMode(mode) ? 2 : 1));
                EXPECT_EQ(layout.routed.gmm1Output.sizeBytes, SIZE_BF_16 * params.maxOutputSize * params.hiddenDim);
                EXPECT_EQ(layout.routed.gmm2Input.data.sizeBytes, params.maxOutputSize * params.hiddenDim / divisor);
                EXPECT_EQ(layout.shared.gmm2Input.data.sizeBytes, shared * params.bs * params.hiddenDim / divisor);
                EXPECT_EQ(layout.shared.gmm1Input.data.sizeBytes, 0);
                const auto groups = MemoryLayoutDetail::CeilDiv(params.bs, MemoryLayoutDetail::L1_TILE_M_256);
                EXPECT_EQ(layout.urma.sync.sharedGmm2Tiles.sizeBytes,
                          shared * groups * MemoryLayoutDetail::INT_CACHELINE * SIZE_INT_32);
            }
        }
    }
}

TEST(MegaMoeArch92UrmaLayout, A8W4LargeBuffersMatchArch35AtTheSameGeometry)
{
    auto params = MakeUrmaParams();
    params.sharedExpertNum = 1;
    params.reserveGmm1Output = true;
    const WorkspaceLayout arch35(params);
    const auto arch92 = LayoutArch92::MakeWorkspaceLayout(params, GMM_MODE_A8W4_NZ, GMM_MODE_A8W4_NZ, false, true);
    EXPECT_EQ(arch92.routed.gmm1Input.data.sizeBytes, arch35.routed.gmm1Input.data.sizeBytes);
    EXPECT_EQ(arch92.routed.gmm1Output.sizeBytes, arch35.routed.gmm1Output.sizeBytes);
    EXPECT_EQ(arch92.routed.gmm2Input.data.sizeBytes, arch35.routed.gmm2Input.data.sizeBytes);
    EXPECT_EQ(arch92.routed.gmm2Output.sizeBytes, arch35.routed.gmm2Output.sizeBytes);
    EXPECT_EQ(arch92.shared.gmm1Output.sizeBytes, arch35.shared.gmm1Output.sizeBytes);
    EXPECT_EQ(arch92.workspaceSize, arch35.workspaceSize);
}

TEST(MegaMoeArch92UrmaLayout, ServerCombineMatchesRelayLayoutForFp8AndFp4)
{
    for (int64_t serverNum : {1, 2}) {
        for (uint8_t mode : {GMM_MODE_A8W8_ND, GMM_MODE_A8W4_NZ, GMM_MODE_A4W4_NZ}) {
            for (bool prefetch : {false, true}) {
                auto params = MakeUrmaParams();
                params.serverNum = serverNum;
                params.dedupMode = DEDUP_MODE_SERVER_COMBINE;
                params.topkWeightsPrefetch = prefetch;
                const auto comm = MakeUrmaCommParams(params, IsA4W4GmmMode(mode));
                const auto layout = LayoutArch92::MakeWorkspaceLayout(params, mode, mode, false, true);
                const auto communication = CalcCclCommLayout(comm);
                const auto relaySlots = serverNum * params.numMaxTokensPerRank;
                EXPECT_GT(layout.urma.serverCombine.tokenIds.sizeBytes, 0);
                EXPECT_GT(layout.urma.serverCombine.tokenBitmaps.sizeBytes, 0);
                const auto flagBytes = MemoryLayoutDetail::CeilAlign(
                    serverNum * MemoryLayoutDetail::INT_CACHELINE * SIZE_INT_32, MemoryLayoutDetail::ALIGN_512);
                EXPECT_EQ(layout.urma.serverCombine.completedCounts.sizeBytes, flagBytes);
                EXPECT_EQ(layout.reset.serverCombine.offsetBytes,
                          layout.urma.serverCombine.completedCounts.offsetBytes);
                EXPECT_EQ(layout.reset.serverCombine.sizeBytes, flagBytes);
                const WorkspaceLayout arch35(params);
                EXPECT_EQ(arch35.urma.serverCombine.completedCounts.sizeBytes, flagBytes);
                EXPECT_EQ(communication.contribution.sizeBytes, relaySlots * params.topK * params.h * SIZE_BF_16);
                EXPECT_EQ(communication.contributionReady.sizeBytes, relaySlots * params.topK * sizeof(uint64_t));
                EXPECT_EQ(communication.combineReceive.sizeBytes, relaySlots * params.h * SIZE_BF_16);
                EXPECT_EQ(communication.combineSend.sizeBytes, 0);
            }
        }
    }
}

TEST(MegaMoeArch92UrmaLayout, Fp8CapacityBoundCoversNativeW4AndFullWidthGelu)
{
    for (int32_t dedupMode : {0, DEDUP_MODE_SERVER_COMBINE}) {
        for (bool prefetch : {false, true}) {
            auto capacity = MakeUrmaParams();
            capacity.hiddenDim = 16384;
            capacity.sharedExpertNum = 4;
            capacity.dedupMode = dedupMode;
            capacity.topkWeightsPrefetch = prefetch;
            const auto capacityComm = MakeUrmaCommParams(capacity, false);
            const auto capacityLayout =
                LayoutArch92::MakeWorkspaceLayout(capacity, GMM_MODE_A8W8_ND, GMM_MODE_A8W8_ND, false, true);
            auto actual = capacity;
            actual.hiddenDim = 8192;
            actual.activationOutputDivisor = 1;
            actual.bs = 257;
            const auto actualComm = MakeUrmaCommParams(actual, true);
            const auto actualLayout =
                LayoutArch92::MakeWorkspaceLayout(actual, GMM_MODE_A4W4_NZ, GMM_MODE_A4W4_NZ, false, true);
            EXPECT_GE(CalcCclComputeOffset(capacityComm), CalcCclComputeOffset(actualComm));
            EXPECT_GE(capacityLayout.workspaceSize, actualLayout.workspaceSize);
        }
    }
}

} // namespace MegaMoeUT
