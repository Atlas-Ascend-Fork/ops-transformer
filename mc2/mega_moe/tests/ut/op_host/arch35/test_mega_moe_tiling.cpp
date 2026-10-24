/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN " AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_kernel/arch35/common/mega_moe_workspace_arch35.h"
#include "../../../../op_kernel/arch35_92/common/mega_moe_utils.h"
#include "../../../../op_kernel/arch35_92/mega_moe_tiling.h"
#include "mc2_tiling_case_executor.h"

namespace MegaMoeUT {

constexpr int64_t ARCH35_CCL_BUFFER_SIZE = 1024LL * 1024LL * 1024LL;

class MegaMoeArch35TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MegaMoeArch35TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MegaMoeArch35TilingTest TearDown" << std::endl;
    }
};

static void FinalizeTestLayoutParams(MegaMoeTilingData& tilingData)
{
    using namespace MegaMoeImpl;
    auto& params = tilingData.compute;
    params.activationOutputDivisor = GetActivationOutputDivisor(tilingData.actMode);
    params.combineQuant = tilingData.combineQuantMode != COMBINE_NO_QUANT;
    params.topkWeightsPrefetch = tilingData.topkWeightsPrefetch == 1;
    params.dispatchFp4 = IsA4W4GmmMode(tilingData.moeGmmMode);
    params.reserveGmm1Output = tilingData.moeGmmMode == GMM_MODE_A8W4_NZ || params.topkWeightsPrefetch;
    params.reserveCumsum = IsW4GmmMode(tilingData.moeGmmMode);
    params.needsGmmEpilogue =
        IsW4GmmMode(tilingData.moeGmmMode) || (params.sharedExpertNum > 0 && IsW4GmmMode(tilingData.sharedGmmMode));
    params.sharedInputElementsPerByte = IsA4W4GmmMode(tilingData.sharedGmmMode) ? 2U : 1U;
    const int64_t rankNumPerServer =
        tilingData.rankNumPerServer == 0 || tilingData.rankNumPerServer > params.epWorldSize ?
            params.epWorldSize :
            tilingData.rankNumPerServer;
    params.serverNum = rankNumPerServer > 0 ? Ops::Base::CeilDiv(params.epWorldSize, rankNumPerServer) : 1;
    params.ring = CalcCclStageRingConfig(params);
    auto& comm = tilingData.comm;
    comm.numMaxTokensPerRank = params.numMaxTokensPerRank;
    comm.topK = params.topK;
    comm.h = params.h;
    comm.moeExpertPerRank = params.moeExpertPerRank;
    comm.epWorldSize = params.epWorldSize;
    comm.yDtypeSize = SIZE_BF_16;
    comm.elemsPerByte = params.dispatchFp4 ? 2U : 1U;
    comm.serverCombineReduce = IsServerCombineDedupOn(params.dedupMode);
    comm.topkWeightsPrefetch =
        params.topkWeightsPrefetch || IsCombineDedupOn(params.dedupMode) || comm.serverCombineReduce;
    comm.isQuantCombine = params.combineQuant;
    comm.topoType = params.topoType;
    comm.serverNum = params.serverNum;
}

TEST_F(MegaMoeArch35TilingTest, MemoryLayoutConstantsMatchKernelContract)
{
    using namespace MegaMoeImpl;
    // tiling 按原始字节传输，保持标准布局和可平凡复制；异常信息区最多保存 2048B tiling。
    EXPECT_TRUE(std::is_standard_layout<MegaMoeTilingData>::value);
    EXPECT_TRUE(std::is_trivially_copyable<MegaMoeTilingData>::value);
    EXPECT_LE(sizeof(MegaMoeTilingData), 2048U);
    EXPECT_EQ(MemoryLayoutDetail::INT_CACHELINE, INT_CACHELINE);
    EXPECT_EQ(MemoryLayoutDetail::MXFP_DIVISOR_SIZE, MXFP_DIVISOR_SIZE);
    EXPECT_EQ(MemoryLayoutDetail::MXFP_SCALE_GROUP_NUM, MXFP_SCALE_GROUP_NUM);
    EXPECT_EQ(MemoryLayoutDetail::MXFP_MULTI_BASE_SIZE, MXFP_MULTI_BASE_SIZE);
    EXPECT_EQ(MemoryLayoutDetail::L1_TILE_M_256, L1_TILE_M_256);
    EXPECT_EQ(MemoryLayoutDetail::ACTIVATION_N_HALF, ACTIVATION_N_HALF);
    EXPECT_EQ(MemoryLayoutDetail::DEDUP_PREPASS_MAX_CHUNKS, DEDUP_PREPASS_MAX_CHUNKS);
    EXPECT_EQ(MemoryLayoutDetail::DEDUP_ROW_DESC_HEADER_INT32, DEDUP_ROW_DESC_HEADER_INT32);
    EXPECT_EQ(MemoryLayoutDetail::MAX_INT16_TOPK_INDEX_COUNT, MAX_INT16_TOPK_INDEX_COUNT);
    EXPECT_EQ(MemoryLayoutDetail::INT32_PER_32B, INT32_PER_256B);
}

TEST_F(MegaMoeArch35TilingTest, DedupModePredicatesMatchKernelContract)
{
    using namespace MegaMoeImpl;
    EXPECT_EQ(DEDUP_MODE_OFF, 0);
    EXPECT_EQ(DEDUP_MODE_COMBINE_ONLY, 1);
    EXPECT_EQ(DEDUP_MODE_DISPATCH_ONLY, 2);
    EXPECT_EQ(DEDUP_MODE_BOTH, 3);
    EXPECT_FALSE(IsDispatchDedupOn(DEDUP_MODE_OFF));
    EXPECT_FALSE(IsCombineDedupOn(DEDUP_MODE_OFF));
    EXPECT_FALSE(IsDispatchDedupOn(DEDUP_MODE_COMBINE_ONLY));
    EXPECT_TRUE(IsCombineDedupOn(DEDUP_MODE_COMBINE_ONLY));
    EXPECT_TRUE(IsDispatchDedupOn(DEDUP_MODE_DISPATCH_ONLY));
    EXPECT_FALSE(IsCombineDedupOn(DEDUP_MODE_DISPATCH_ONLY));
    EXPECT_TRUE(IsDispatchDedupOn(DEDUP_MODE_BOTH));
    EXPECT_TRUE(IsCombineDedupOn(DEDUP_MODE_BOTH));
}

TEST_F(MegaMoeArch35TilingTest, GmMemoryRegionsPreserveTopologyOrder)
{
    using namespace MegaMoeImpl;
    MegaMoeTilingData params{};
    params.compute.numMaxTokensPerRank = 128;
    params.compute.topK = 8;
    params.compute.h = 4096;
    params.compute.moeExpertPerRank = 4;
    params.compute.epWorldSize = 4;
    params.compute.topoType = TOPO_TYPE_MTE;
    params.rankNumPerServer = 2;
    FinalizeTestLayoutParams(params);
    const PeermemInfo mte(params.comm);
    // 固定输入的通信分区参考值，不通过公共布局函数重新生成预期偏移和容量。
    const MemoryRegionLayout mteExpected[] = {{0, 49152},        {49152, 12288}, {1110016, 4096}, {61440, 1048576},
                                              {1114112, 540672}, {-1, 0},        {-1, 0},         {1654784, 8388608}};
    const GmMemoryRegion mteRegions[] = {mte.rankSync,        mte.syncCount,       mte.maskRecv,
                                         mte.expertCountRecv, mte.quantTokenScale, mte.dispatchReceive,
                                         mte.dispatchFlag,    mte.combineSend};
    for (uint32_t index = 0; index < 8U; ++index) {
        EXPECT_EQ(mteRegions[index].offset, mteExpected[index].offset);
        EXPECT_EQ(mteRegions[index].size, mteExpected[index].size);
    }
    EXPECT_EQ(mte.totalSize, 10043392LL);
    EXPECT_EQ(mte.rankSync.offset, 0);
    EXPECT_EQ(mte.syncCount.offset + mte.syncCount.size, PEERMEM_DATA_OFFSET);
    EXPECT_EQ(mte.expertCountRecv.offset, PEERMEM_DATA_OFFSET);
    EXPECT_EQ(mte.expertCountRecv.size, PEERMEM_MTE_COUNT_REGION_SIZE);
    EXPECT_EQ(mte.maskRecv.offset, mte.expertCountRecv.offset + mte.expertCountRecv.size);
    EXPECT_EQ(mte.combineSend.offset, mte.quantTokenScale.offset + mte.quantTokenScale.size);
    EXPECT_EQ(mte.totalSize, mte.combineSend.offset + mte.combineSend.size);
    EXPECT_EQ(mte.dispatchReceive.ptr, nullptr);
    EXPECT_EQ(mte.dispatchFlag.ptr, nullptr);
    EXPECT_EQ(mte.combineSend.ptr, nullptr);
    std::vector<uint8_t> window(static_cast<size_t>(mte.totalSize));
    const PeermemInfo bound(params.comm, window.data());
    EXPECT_EQ(bound.rankSync.ptr, window.data());
    EXPECT_EQ(bound.maskRecv.ptr, window.data() + bound.maskRecv.offset);
    EXPECT_EQ(bound.combineSend.ptr, window.data() + bound.combineSend.offset);
    EXPECT_EQ(bound.cclComputePtr, window.data() + Ops::Base::CeilAlign(bound.totalSize, ALIGN_512));
    EXPECT_EQ(bound.dispatchReceive.ptr, nullptr);
    EXPECT_EQ(bound.dispatchFlag.ptr, nullptr);
    const PeermemInfo copy = bound;
    EXPECT_EQ(copy.maskRecv.ptr, bound.maskRecv.ptr);
    EXPECT_EQ(copy.combineSend.ptr, bound.combineSend.ptr);
    EXPECT_EQ(copy.totalSize, bound.totalSize);
    auto emptyParams = params;
    emptyParams.compute.numMaxTokensPerRank = 0;
    FinalizeTestLayoutParams(emptyParams);
    const PeermemInfo empty(emptyParams.comm, window.data());
    EXPECT_EQ(empty.combineSend.size, 0);
    EXPECT_EQ(empty.combineSend.ptr, window.data() + empty.combineSend.offset);

    params.compute.topoType = TOPO_TYPE_URMA;
    params.compute.epWorldSize = 1024;
    params.rankNumPerServer = 8;
    FinalizeTestLayoutParams(params);
    const PeermemInfo urma(params.comm);
    const MemoryRegionLayout urmaExpected[] = {{0, 65536}, {65536, 12288},     {77824, 655360},    {733184, 16384},
                                               {-1, 0},    {749568, 75497472}, {76247040, 131072}, {76378112, 8388608}};
    const GmMemoryRegion urmaRegions[] = {urma.rankSync,        urma.syncCount,       urma.maskRecv,
                                          urma.expertCountRecv, urma.quantTokenScale, urma.dispatchReceive,
                                          urma.dispatchFlag,    urma.combineSend};
    for (uint32_t index = 0; index < 8U; ++index) {
        EXPECT_EQ(urmaRegions[index].offset, urmaExpected[index].offset);
        EXPECT_EQ(urmaRegions[index].size, urmaExpected[index].size);
    }
    EXPECT_EQ(urma.totalSize, 84766720LL);
    EXPECT_EQ(urma.syncCount.offset, 1024 * PEERMEM_SYNC_SLOT_SIZE);
    EXPECT_EQ(urma.maskRecv.offset, urma.syncCount.offset + urma.syncCount.size);
    EXPECT_EQ(urma.expertCountRecv.offset, urma.maskRecv.offset + urma.maskRecv.size);
    EXPECT_EQ(urma.dispatchReceive.offset, urma.expertCountRecv.offset + urma.expertCountRecv.size);
    EXPECT_EQ(urma.dispatchFlag.offset, urma.dispatchReceive.offset + urma.dispatchReceive.size);
    EXPECT_EQ(urma.combineSend.offset, urma.dispatchFlag.offset + urma.dispatchFlag.size);
    EXPECT_EQ(urma.totalSize, urma.combineSend.offset + urma.combineSend.size);
    EXPECT_EQ(urma.quantTokenScale.ptr, nullptr);
}

TEST_F(MegaMoeArch35TilingTest, GmMemoryRegionsKeepCapacityOptionsIndependent)
{
    using namespace MegaMoeImpl;
    MegaMoeTilingData params{};
    params.compute.numMaxTokensPerRank = 128;
    params.compute.topK = 8;
    params.compute.h = 4096;
    params.compute.moeExpertPerRank = 4;
    params.compute.epWorldSize = 770;
    params.compute.topoType = TOPO_TYPE_URMA;
    params.rankNumPerServer = 770;
    FinalizeTestLayoutParams(params);
    const PeermemInfo a8(params.comm);
    EXPECT_EQ(a8.maskRecv.offset % ALIGN_512, 0);
    EXPECT_GT(a8.maskRecv.offset, a8.syncCount.offset + a8.syncCount.size);
    auto fp4Comm = params.comm;
    fp4Comm.elemsPerByte = 2U;
    const PeermemInfo a4(fp4Comm);
    EXPECT_LT(a4.dispatchReceive.size, a8.dispatchReceive.size);
    EXPECT_LE(a4.totalSize, a8.totalSize); // Host 的 1 byte 容量上界覆盖 A4。
    params.topkWeightsPrefetch = true;
    FinalizeTestLayoutParams(params);
    params.comm.elemsPerByte = 2U;
    const PeermemInfo weights(params.comm);
    EXPECT_GE(weights.dispatchReceive.size, a4.dispatchReceive.size);
    EXPECT_EQ(weights.combineSend.size, a4.combineSend.size);
    params.combineQuantMode = 1;
    FinalizeTestLayoutParams(params);
    params.comm.elemsPerByte = 2U;
    const PeermemInfo quant(params.comm);
    EXPECT_EQ(quant.dispatchReceive.size, weights.dispatchReceive.size);
    EXPECT_LT(quant.combineSend.size, weights.combineSend.size);
    EXPECT_EQ(quant.totalSize, quant.combineSend.offset + quant.combineSend.size);
}

static int64_t CalcCclCommunicationCapacity(const MegaMoeTilingData& tilingData)
{
    int64_t communicationSize = MegaMoeImpl::PeermemInfo(tilingData.comm).totalSize;
    if (tilingData.compute.topoType == MegaMoeImpl::TOPO_TYPE_MTE) {
        communicationSize += MegaMoeImpl::EXCEPTION_DUMP_REGION_SIZE;
    }
    return Ops::Base::CeilAlign(communicationSize, static_cast<int64_t>(MegaMoeImpl::ALIGN_512));
}

static void ExpectCclBufferCapacity(const TilingInfo& tilingInfo, int64_t cclBufferSize = ARCH35_CCL_BUFFER_SIZE)
{
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    MegaMoeImpl::WorkspaceLayoutArch35 workspaceLayout(tilingData.compute);
    const int64_t communicationCapacity = CalcCclCommunicationCapacity(tilingData);

    ASSERT_GT(workspaceLayout.workspaceSize, 0);
    EXPECT_GT(communicationCapacity, 0);
    EXPECT_EQ(communicationCapacity % MegaMoeImpl::ALIGN_512, 0);
    EXPECT_LE(communicationCapacity + workspaceLayout.workspaceSize, cclBufferSize);
    ASSERT_EQ(tilingInfo.workspaceSizes.size(), 1U);
}

static bool RunMteCclBufferCase(int64_t cclBufferSize, TilingInfo& tilingInfo, int64_t maxRecvTokenNum = 0,
                                int64_t numMaxTokensPerRank = 0, int64_t combineCommMode = 0, int64_t topoType = 0,
                                bool includeCombineCommMode = true, int64_t moeExpertPerRank = 4)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 8}, {128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{moeExpertPerRank, 2048, 4096}, {moeExpertPerRank, 2048, 4096}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{moeExpertPerRank, 4096, 1024}, {moeExpertPerRank, 4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{moeExpertPerRank, 2048, 64, 2}, {moeExpertPerRank, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{moeExpertPerRank, 4096, 16, 2}, {moeExpertPerRank, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{moeExpertPerRank}, {moeExpertPerRank}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4 * moeExpertPerRank)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(cclBufferSize)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxRecvTokenNum)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numMaxTokensPerRank)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topoType)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"combine_comm_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(combineCommMode)},
        },
        &compileInfo);
    if (!includeCombineCommMode) {
        // Combine 可选开关位于属性表末尾，模拟旧图省略该属性的情况。
        tilingContextPara.attrs_.pop_back();
    }
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    const bool tilingSucceeded = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    return tilingSucceeded;
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_FP8E4M3FN)
{
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, tilingInfo));
    EXPECT_EQ(tilingInfo.tilingKey, 664612UL);
    ExpectCclBufferCapacity(tilingInfo);
}

TEST_F(MegaMoeArch35TilingTest, MteCommunicationDedupUsesNaturalDispatchBenefitGate)
{
    for (const int64_t moeExpertPerRank : {4, 8}) {
        for (const int64_t combineCommMode : {0, 1}) {
            SCOPED_TRACE(testing::Message() << "experts=" << moeExpertPerRank << ", combine=" << combineCommMode);
            TilingInfo tilingInfo;
            ASSERT_TRUE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, tilingInfo, 257, 128, combineCommMode,
                                            MegaMoeImpl::TOPO_TYPE_MTE, true, moeExpertPerRank));
            ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
            MegaMoeTilingData tilingData{};
            std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
            // 每卡专家数增加后单专家节省量下降，沿用主线收益门自动关闭 Dispatch 去重。
            EXPECT_EQ(tilingData.compute.dedupMode, (moeExpertPerRank == 4 ? 2 : 0) | combineCommMode);
            ExpectCclBufferCapacity(tilingInfo);
        }
    }
}

TEST_F(MegaMoeArch35TilingTest, MteCombineDedupDefaultsOffWithoutChangingDispatchGate)
{
    for (const bool includeCombineCommMode : {false, true}) {
        TilingInfo tilingInfo;
        ASSERT_TRUE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, tilingInfo, 0, 0, 0, MegaMoeImpl::TOPO_TYPE_MTE,
                                        includeCombineCommMode));
        ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
        MegaMoeTilingData tilingData{};
        std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
        EXPECT_EQ(tilingData.compute.dedupMode, 2);
    }
}

TEST_F(MegaMoeArch35TilingTest, CombineDedupAttrRejectsInvalidValues)
{
    for (const int64_t invalidMode : {-1, 2, 3}) {
        TilingInfo tilingInfo;
        EXPECT_FALSE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, tilingInfo, 257, 256, invalidMode));
    }
}

TEST_F(MegaMoeArch35TilingTest, CombineDedupRequestRejectsNonMteTopology)
{
    TilingInfo tilingInfo;
    EXPECT_FALSE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, tilingInfo, 257, 256, 1, MegaMoeImpl::TOPO_TYPE_URMA));
}

TEST_F(MegaMoeArch35TilingTest, DedupPrepassUbOverflowDisablesOnlyAutomaticDispatch)
{
    // topK=8 时 prepass 每 chunk 最多容纳 512 token，超过 128 个 chunk 时仅自动 Dispatch 可降级。
    constexpr int64_t oversizedNumMax = 512LL * MegaMoeImpl::DEDUP_PREPASS_MAX_CHUNKS + 1LL;
    for (const int64_t combineCommMode : {0, 1}) {
        TilingInfo boundaryInfo;
        ASSERT_TRUE(RunMteCclBufferCase(64LL * ARCH35_CCL_BUFFER_SIZE, boundaryInfo, 257, oversizedNumMax - 1,
                                        combineCommMode));
        ASSERT_GE(boundaryInfo.tilingDataSize, sizeof(MegaMoeTilingData));
        MegaMoeTilingData boundaryData{};
        std::memcpy(&boundaryData, boundaryInfo.tilingData.get(), sizeof(boundaryData));
        EXPECT_EQ(boundaryData.compute.dedupMode, 2 | combineCommMode);
        TilingInfo overflowInfo;
        const bool succeeded =
            RunMteCclBufferCase(64LL * ARCH35_CCL_BUFFER_SIZE, overflowInfo, 257, oversizedNumMax, combineCommMode);
        if (combineCommMode == 1) {
            EXPECT_FALSE(succeeded);
        } else {
            ASSERT_TRUE(succeeded);
            ASSERT_GE(overflowInfo.tilingDataSize, sizeof(MegaMoeTilingData));
            MegaMoeTilingData overflowData{};
            std::memcpy(&overflowData, overflowInfo.tilingData.get(), sizeof(overflowData));
            EXPECT_EQ(overflowData.compute.dedupMode, 0);
        }
    }
}

TEST_F(MegaMoeArch35TilingTest, CclBufferExactCapacityAndOneByteShort)
{
    TilingInfo initialInfo;
    ASSERT_TRUE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, initialInfo));
    ASSERT_GE(initialInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, initialInfo.tilingData.get(), sizeof(tilingData));
    MegaMoeImpl::WorkspaceLayoutArch35 workspaceLayout(tilingData.compute);
    const int64_t communicationCapacity = CalcCclCommunicationCapacity(tilingData);
    const int64_t requiredSize = communicationCapacity + workspaceLayout.workspaceSize;

    TilingInfo exactInfo;
    ASSERT_TRUE(RunMteCclBufferCase(requiredSize, exactInfo));
    ExpectCclBufferCapacity(exactInfo, requiredSize);
    TilingInfo shortInfo;
    EXPECT_FALSE(RunMteCclBufferCase(requiredSize - 1, shortInfo));
    TilingInfo communicationOnlyInfo;
    EXPECT_FALSE(RunMteCclBufferCase(communicationCapacity, communicationOnlyInfo));

    TilingInfo largerInfo;
    ASSERT_TRUE(RunMteCclBufferCase(requiredSize + 512, largerInfo));
    ExpectCclBufferCapacity(largerInfo, requiredSize + 512);
}

TEST_F(MegaMoeArch35TilingTest, MteCombineDedupTilingUsesExpertReadyAndFullResults)
{
    TilingInfo probeInfo;
    ASSERT_TRUE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, probeInfo, 257, 8192, 1));
    ASSERT_GE(probeInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData probeData{};
    std::memcpy(&probeData, probeInfo.tilingData.get(), sizeof(probeData));
    const int64_t maxRecvTokenNum = 3LL * probeData.compute.mGroupsPerWave * 256LL + 1LL;
    constexpr int64_t cclBufferSize = 8LL * ARCH35_CCL_BUFFER_SIZE;
    TilingInfo tilingInfo;
    ASSERT_TRUE(RunMteCclBufferCase(cclBufferSize, tilingInfo, maxRecvTokenNum, 8192, 1));
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    ASSERT_TRUE(MegaMoeImpl::IsCombineDedupOn(tilingData.compute.dedupMode));
    ASSERT_TRUE(tilingData.compute.ring.activation);
    EXPECT_FALSE(tilingData.compute.ring.dispatch);
    EXPECT_FALSE(tilingData.compute.ring.gmm2);
    const MegaMoeImpl::WorkspaceLayoutArch35 layout(tilingData.compute);
    EXPECT_EQ(layout.gmm2CombineSyncCounterOffset, MegaMoeImpl::INVALID_WORKSPACE_OFFSET);
    EXPECT_GE(layout.gmm2ReadyOffset, layout.flagActivationToGmm2Offset);
    EXPECT_GE(layout.gmm2MmadResOffset, 0);
    EXPECT_EQ(layout.gmm2MmadResOffset % 512LL, 0);
    EXPECT_EQ(layout.dedupRowDescOffset - layout.gmm2MmadResOffset,
              Ops::Base::CeilAlign(2LL * tilingData.compute.maxOutputSize * tilingData.compute.h, 512LL));
    ExpectCclBufferCapacity(tilingInfo, cclBufferSize);
}

TEST_F(MegaMoeArch35TilingTest, CclDataOffsetFitsFp4CapacityBudget)
{
    for (int64_t topoType : {MegaMoeImpl::TOPO_TYPE_MTE, MegaMoeImpl::TOPO_TYPE_URMA}) {
        for (int32_t prefetch : {0, 1}) {
            for (int32_t combineQuantMode : {0, 3, 4}) {
                for (bool combineDedup : {false, true}) {
                    if (combineDedup && (topoType == MegaMoeImpl::TOPO_TYPE_URMA ||
                                         combineQuantMode != MegaMoeImpl::COMBINE_NO_QUANT)) {
                        continue;
                    }
                    SCOPED_TRACE(testing::Message()
                                 << "topoType=" << topoType << ", prefetch=" << prefetch
                                 << ", combineQuantMode=" << combineQuantMode << ", combineDedup=" << combineDedup);
                    MegaMoeTilingData params{};
                    params.compute.numMaxTokensPerRank = 257;
                    params.compute.topK = 8;
                    params.compute.h = 4096;
                    params.compute.moeExpertPerRank = 2;
                    params.compute.epWorldSize = 8;
                    params.topkWeightsPrefetch = prefetch;
                    params.compute.dedupMode = combineDedup ? 3 : 0;
                    params.combineQuantMode = combineQuantMode;
                    params.compute.topoType = topoType;
                    params.rankNumPerServer = topoType == MegaMoeImpl::TOPO_TYPE_URMA ? 4 : 8;

                    params.moeGmmMode = MegaMoeImpl::GMM_MODE_A4W4_ND;
                    FinalizeTestLayoutParams(params);
                    auto budgetParams = params;
                    budgetParams.comm.elemsPerByte = 1U;
                    const int64_t dumpBytes =
                        topoType == MegaMoeImpl::TOPO_TYPE_MTE ? MegaMoeImpl::EXCEPTION_DUMP_REGION_SIZE : 0;
                    const int64_t communicationCapacity = Ops::Base::CeilAlign(
                        dumpBytes + MegaMoeImpl::PeermemInfo(budgetParams.comm).totalSize, MegaMoeImpl::ALIGN_512);
                    const int64_t communicationSize = MegaMoeImpl::PeermemInfo(params.comm).totalSize;
                    const int64_t cclDataOffset = Ops::Base::CeilAlign(communicationSize, MegaMoeImpl::ALIGN_512);

                    // 实际布局按 FP4 压缩记录推进，业务起点位于通信末尾并保持 512B 对齐。
                    EXPECT_GE(cclDataOffset, communicationSize);
                    EXPECT_LT(cclDataOffset - communicationSize, MegaMoeImpl::ALIGN_512);
                    EXPECT_EQ(cclDataOffset % MegaMoeImpl::ALIGN_512, 0);
                    // MTE 的异常信息区在传入基址之前，保守申请容量仍覆盖实际业务起点。
                    EXPECT_LT(dumpBytes + cclDataOffset, communicationCapacity);
                    EXPECT_LE(communicationSize, MegaMoeImpl::PeermemInfo(budgetParams.comm).totalSize);
                }
            }
        }
    }
}

TEST_F(MegaMoeArch35TilingTest, CclBufferRejectsReceiveCapacityTruncation)
{
    TilingInfo tilingInfo;
    const int64_t uint32Max = std::numeric_limits<uint32_t>::max();
    EXPECT_FALSE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, tilingInfo, uint32Max + 1));
    // 默认接收容量为最大 token 数乘以 EP，再乘以 topK 与本卡专家数的较小值。
    EXPECT_FALSE(RunMteCclBufferCase(ARCH35_CCL_BUFFER_SIZE, tilingInfo, 0, uint32Max));
}

TEST_F(MegaMoeArch35TilingTest, H5120_BS256_URMA_InvalidRankNumPerServer)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 5120}, {256, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{256, 6}, {256, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256, 6}, {256, 6}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8, 3072, 5120}, {8, 3072, 5120}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{8, 5120, 1536}, {8, 5120, 1536}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{8, 3072, 80, 2}, {8, 3072, 80, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{8, 5120, 24, 2}, {8, 5120, 24, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{256, 5120}, {256, 5120}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(256)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    uint64_t expectTilingKey = 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, expectTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, H7168_BS512_MTE_RankNumIgnored)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 7168}, {512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{512, 8}, {512, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4096, 7168}, {2, 4096, 7168}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 7168, 2048}, {2, 7168, 2048}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{{2, 4096, 112, 2}, {2, 4096, 112, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{2, 7168, 32, 2}, {2, 7168, 32, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{512, 7168}, {512, 7168}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    uint64_t expectTilingKey = 664612UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, DifferentNConfig)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT8_E5M2, ge::FORMAT_ND},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(35)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 8995UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

static void RunUrmaTilingCase(int64_t xBs, int64_t topkIdsBs, int64_t topkWeightsBs, int64_t numMaxTokensPerRank,
                              int64_t maxRecvTokenNum, int64_t hiddenDim, const std::string& activation,
                              const std::vector<float>& activationParams, ge::graphStatus expectedStatus,
                              int64_t combineCommMode = 0, int64_t combineQuantMode = 0, int64_t topoType = 1,
                              int64_t h = 4096)
{
    ASSERT_GT(hiddenDim, 0);
    ASSERT_EQ(hiddenDim % (MegaMoeImpl::ACTIVATION_N_HALF * MegaMoeImpl::MXFP_DIVISOR_SIZE), 0);

    const int64_t activationHidden = activation == "gelu" ? hiddenDim : hiddenDim / MegaMoeImpl::ACTIVATION_N_HALF;

    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{xBs, h}, {xBs, h}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{topkIdsBs, 8}, {topkIdsBs, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{topkWeightsBs, 8}, {topkWeightsBs, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, hiddenDim, h}, {8, hiddenDim, h}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},
            {{{8, h, activationHidden}, {8, h, activationHidden}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_FRACTAL_NZ},
            {{{8, hiddenDim, (h + 63) / 64, 2}, {8, hiddenDim, (h + 63) / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{8, h, activationHidden / MegaMoeImpl::MXFP_DIVISOR_SIZE, 2},
              {8, h, activationHidden / MegaMoeImpl::MXFP_DIVISOR_SIZE, 2}},
             ge::DT_FLOAT8_E8M0,
             ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{xBs, h}, {xBs, h}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(
                                    combineCommMode == 2 ? 268435456 : ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxRecvTokenNum)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(36)},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(combineQuantMode)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(numMaxTokensPerRank)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>(activation)},
            {"activation_params", Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>(activationParams)},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(topoType)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"combine_comm_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(combineCommMode)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    const uint64_t expectedTilingKey = expectedStatus == ge::GRAPH_SUCCESS ? 42607652UL : 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);

    if (expectedStatus != ge::GRAPH_SUCCESS || (activation != "situglu" && combineCommMode != 2)) {
        return;
    }

    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    TilingInfo tilingInfo;
    const bool tilingSucceeded = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    ASSERT_TRUE(tilingSucceeded);
    ExpectCclBufferCapacity(tilingInfo);

    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    if (combineCommMode == 2) {
        EXPECT_EQ(tilingData.compute.dedupMode, MegaMoeImpl::DEDUP_MODE_SERVER_COMBINE);
        EXPECT_TRUE(tilingData.comm.serverCombineReduce);
        EXPECT_TRUE(tilingData.comm.topkWeightsPrefetch);
        EXPECT_FALSE(tilingData.compute.ring.dispatch);
        EXPECT_FALSE(tilingData.compute.ring.gmm1);
        EXPECT_FALSE(tilingData.compute.ring.activation);
        EXPECT_FALSE(tilingData.compute.ring.gmm2);
        MegaMoeImpl::WorkspaceLayoutArch35 workspace(tilingData.compute);
        EXPECT_GE(workspace.combineRelayCompletedCountOffset, 0);
        EXPECT_GE(workspace.tokenIdsOffset, 0);
        EXPECT_GE(workspace.tokenBitmapsOffset, 0);
        EXPECT_GT(workspace.serverCombineResetBytes, 0);
    }
    if (activation != "situglu") {
        return;
    }
    ASSERT_TRUE(activationParams.size() == 1U || activationParams.size() == 2U);
    EXPECT_EQ(tilingData.actMode, static_cast<uint8_t>(MegaMoeImpl::MegaMoeActMode::SITU));
    EXPECT_EQ(tilingData.actSubMode,
              static_cast<uint8_t>(activationParams.size() == 2U ? MegaMoeImpl::MegaMoeActSubMode::LINEAR :
                                                                   MegaMoeImpl::MegaMoeActSubMode::DEFAULT));
    EXPECT_FLOAT_EQ(tilingData.activationAlpha, activationParams.size() == 2U ? activationParams[1] : 0.0f);
    EXPECT_FLOAT_EQ(tilingData.activationBeta, activationParams[0]);
}

TEST_F(MegaMoeArch35TilingTest, UbxUsesOrdinaryLayeredTiling)
{
    RunUrmaTilingCase(17, 17, 17, 33, 0, 2048, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS, 0, 0,
                      2);
}

TEST_F(MegaMoeArch35TilingTest, UbxSupportsServerCombineDedup)
{
    RunUrmaTilingCase(17, 17, 17, 33, 0, 2048, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS, 2, 0,
                      2);
}

TEST_F(MegaMoeArch35TilingTest, UbxRejectsQuantizedServerCombineDedup)
{
    RunUrmaTilingCase(17, 17, 17, 33, 0, 2048, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED, 2, 3,
                      2);
}

TEST_F(MegaMoeArch35TilingTest, UbxRejectsCardCombineDedup)
{
    RunUrmaTilingCase(17, 17, 17, 33, 0, 2048, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED, 1, 0,
                      2);
}

TEST_F(MegaMoeArch35TilingTest, UbxSupportsGeluServerCombineDedup)
{
    RunUrmaTilingCase(17, 17, 17, 33, 0, 2048, "gelu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS, 2, 0, 2);
}

static void RunUrmaVariableBsInputShapeCase(int64_t xBs, int64_t topkIdsBs, int64_t topkWeightsBs,
                                            int64_t numMaxTokensPerRank, int64_t maxRecvTokenNum,
                                            ge::graphStatus expectedStatus)
{
    RunUrmaTilingCase(xBs, topkIdsBs, topkWeightsBs, numMaxTokensPerRank, maxRecvTokenNum, 2048, "swiglu",
                      {std::numeric_limits<float>::max()}, expectedStatus);
}

static void RunUrmaVariableBsCase(int64_t bs, int64_t numMaxTokensPerRank, int64_t maxRecvTokenNum,
                                  ge::graphStatus expectedStatus)
{
    RunUrmaVariableBsInputShapeCase(bs, bs, bs, numMaxTokensPerRank, maxRecvTokenNum, expectedStatus);
}

static void RunUrmaActivationCase(int64_t hiddenDim, const std::string& activation,
                                  const std::vector<float>& activationParams, ge::graphStatus expectedStatus)
{
    RunUrmaTilingCase(128, 128, 128, 256, 0, hiddenDim, activation, activationParams, expectedStatus);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_NumMaxTokens256_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 256, 0, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SITU_Default_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "situglu", {1.0f}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SITU_Linear_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "situglu", {1.0f, 2.0f}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N768_SITU_URMA_2Servers)
{
    RunUrmaActivationCase(768, "situglu", {1.0f}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N512_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(512, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N1536_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(1536, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N8192_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(8192, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N256_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(256, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N640_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(640, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N8448_SwiGlu_URMA_2Servers)
{
    RunUrmaActivationCase(8448, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SwiGluStep_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "swiglustep", {}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_N2048_SITU_MissingBeta_URMA_2Servers)
{
    RunUrmaActivationCase(2048, "situglu", {}, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_NumMaxTokensZero_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 0, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_NumMaxTokens64_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 64, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS0_NumMaxTokens256_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(0, 256, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_TopkIdsBS64_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsInputShapeCase(128, 64, 128, 256, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_TopkWeightsBS64_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsInputShapeCase(128, 128, 64, 256, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_RouteCapacityOverflow_FP8E4M3FN_URMA_2Servers)
{
    const int64_t numMaxTokensPerRank = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) / 8 + 1;
    RunUrmaVariableBsCase(128, numMaxTokensPerRank, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_OutputCapacityOverflow_FP8E4M3FN_URMA_2Servers)
{
    const int64_t numMaxTokensPerRank = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) / 16 + 1;
    RunUrmaVariableBsCase(128, numMaxTokensPerRank, 0, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_MaxRecvTokenNumOverflow_FP8E4M3FN_URMA_2Servers)
{
    const int64_t maxRecvTokenNum = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    RunUrmaVariableBsCase(128, 256, maxRecvTokenNum, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, H4096_MaxRecvTokenNumAboveCapacity_FP8E4M3FN_URMA_2Servers)
{
    RunUrmaVariableBsCase(128, 256, 4097, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, BS128_NumMaxTokens256_URMA_WorkspaceUsesCapacityMaskAndAllServers)
{
    MegaMoeTilingData tilingData{};
    tilingData.compute.bs = 128;
    tilingData.compute.h = 4096;
    tilingData.compute.hiddenDim = 2048;
    tilingData.compute.epWorldSize = 2;
    tilingData.compute.maxOutputSize = 4096;
    tilingData.compute.topK = 8;
    tilingData.compute.aicNum = 28;
    tilingData.compute.moeExpertPerRank = 8;
    tilingData.compute.combineSyncSlotCountPerExpert = 1;
    tilingData.compute.topoType = MegaMoeImpl::TOPO_TYPE_URMA;
    tilingData.compute.numMaxTokensPerRank = 256;
    tilingData.rankNumPerServer = 1;

    FinalizeTestLayoutParams(tilingData);
    MegaMoeImpl::WorkspaceLayoutArch35 workspaceLayout(tilingData.compute);
    const int64_t actualMaskWorkspaceBytes =
        workspaceLayout.dispatchRelaySendQueueOffset - workspaceLayout.maskSlotOffset;
    const int64_t capacityMaskSlotBytes =
        MegaMoeImpl::CalcDispatchMaskAlignSize(tilingData.compute.numMaxTokensPerRank, tilingData.compute.topK) +
        MegaMoeImpl::ALIGN_32;
    const int64_t expectedMaskWorkspaceBytes =
        Ops::Base::CeilAlign(static_cast<int64_t>(tilingData.compute.moeExpertPerRank) *
                                 tilingData.compute.epWorldSize * capacityMaskSlotBytes,
                             static_cast<int64_t>(MegaMoeImpl::ALIGN_512));
    const int64_t actualRelayQueueBytes =
        workspaceLayout.dispatchRemoteReadyFlagSnapshotOffset - workspaceLayout.dispatchRelaySendQueueOffset;
    const int64_t serverNum = Ops::Base::CeilDiv<int64_t>(tilingData.compute.epWorldSize, tilingData.rankNumPerServer);
    const int64_t expectedRelayQueueBytes = Ops::Base::CeilAlign(
        serverNum * (MegaMoeImpl::ALIGN_32 + static_cast<int64_t>(tilingData.compute.bs) * MegaMoeImpl::ALIGN_32),
        static_cast<int64_t>(MegaMoeImpl::ALIGN_512));
    const int64_t localBsMaskSlotBytes =
        MegaMoeImpl::CalcDispatchMaskAlignSize(tilingData.compute.bs, tilingData.compute.topK) + MegaMoeImpl::ALIGN_32;
    const int64_t localBsMaskWorkspaceBytes =
        Ops::Base::CeilAlign(static_cast<int64_t>(tilingData.compute.moeExpertPerRank) *
                                 tilingData.compute.epWorldSize * localBsMaskSlotBytes,
                             static_cast<int64_t>(MegaMoeImpl::ALIGN_512));

    EXPECT_EQ(actualMaskWorkspaceBytes, expectedMaskWorkspaceBytes);
    EXPECT_EQ(actualRelayQueueBytes, expectedRelayQueueBytes);
    EXPECT_NE(actualMaskWorkspaceBytes, localBsMaskWorkspaceBytes);
}

static void RunA8W4FormatCase(ge::Format weightOneFormat, ge::Format weightTwoFormat, ge::graphStatus expectedStatus,
                              uint64_t expectedTilingKey, int64_t h = 4096)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, h}, {128, h}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2048, h}, {4, 2048, h}}, ge::DT_FLOAT4_E2M1, weightOneFormat},
            {{{4, h, 1024}, {4, h, 1024}}, ge::DT_FLOAT4_E2M1, weightTwoFormat},
            {{{4, 2048, (h + 63) / 64, 2}, {4, 2048, (h + 63) / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, h, 16, 2}, {4, h, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, h}, {128, h}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
    if (expectedStatus != ge::GRAPH_SUCCESS) {
        return;
    }
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
    TilingInfo tilingInfo;
    const bool tilingSucceeded = ExecuteTiling(tilingContextPara, tilingInfo);
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
    ASSERT_TRUE(tilingSucceeded);
    ASSERT_GE(tilingInfo.tilingDataSize, sizeof(MegaMoeTilingData));
    MegaMoeTilingData tilingData{};
    std::memcpy(&tilingData, tilingInfo.tilingData.get(), sizeof(tilingData));
    EXPECT_EQ(tilingData.compute.h, h);
    ExpectCclBufferCapacity(tilingInfo);
    if (h != 4096) {
        return; // 下方固定 L1 offset 断言仅对应原 H4096 用例。
    }
    const auto& layout = tilingData.a8w4L1Layout;
    EXPECT_EQ(layout.aBufferNum, 3U);
    EXPECT_EQ(layout.bBufferNum, 4U);
    EXPECT_EQ(layout.tileK, 256U);
    EXPECT_EQ(layout.scaleK, 2048U);
    const uint64_t expectedBOffsets[] = {192U * 1024, 320U * 1024, 256U * 1024, 384U * 1024};
    for (uint64_t slot = 0; slot < 4U; ++slot) {
        EXPECT_EQ(layout.bOffsets[slot], expectedBOffsets[slot]);
    }
    EXPECT_EQ(layout.aOffsets[2] + layout.aSlotBytes, layout.bOffsets[0]);
    EXPECT_EQ(layout.scaleAOffsets[0], 448U * 1024);
    EXPECT_EQ(layout.scaleBOffsets[1] + layout.scaleBSlotBytes, 512U * 1024);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A8W4_URMA)
{
    RunA8W4FormatCase(ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ_C0_32, ge::GRAPH_SUCCESS, 46802984UL);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A8W4_URMA_InvalidWeight2Format)
{
    RunA8W4FormatCase(ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_ND, ge::GRAPH_FAILED, 0UL);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_URMA)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2048, 4096}, {4, 2048, 4096}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_ND},
            {{{4, 4096, 1024}, {4, 4096, 1024}}, ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ_C0_32},
            {{{4, 2048, 64, 2}, {4, 2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, 4096, 16, 2}, {4, 4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT4_E2M1))},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    uint64_t expectTilingKey = 62138408UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
}

static void RunA4W4NzFormatCase(ge::Format weightOneFormat, ge::Format weightTwoFormat, ge::graphStatus expectedStatus,
                                uint64_t expectedTilingKey, int64_t h = 4096)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    gert::TilingContextPara tilingContextPara(
        "MegaMoe",
        {
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, h}, {128, h}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{128, 6}, {128, 6}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{4, 2048, h}, {4, 2048, h}}, ge::DT_FLOAT4_E2M1, weightOneFormat},
            {{{4, h, 1024}, {4, h, 1024}}, ge::DT_FLOAT4_E2M1, weightTwoFormat},
            {{{4, 2048, (h + 63) / 64, 2}, {4, 2048, (h + 63) / 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{{4, h, 16, 2}, {4, h, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT8, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{128, h}, {128, h}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
            {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
            {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"dispatch_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT4_E2M1))},
            {"shared_expert_quant_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
            {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
            {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
            {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
            {"activation_params",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
            {"activation_out_dtype",
             Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
            {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
            {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
            {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        },
        &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_NZ_URMA)
{
    RunA4W4NzFormatCase(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_32, ge::GRAPH_SUCCESS, 62138408UL);
}

TEST_F(MegaMoeArch35TilingTest, UrmaH1088PaddedRecords)
{
    using namespace MegaMoeImpl;
    // 有效 scale 是 34B，一级权重起点分别为 1344B（FP8）与 832B（FP4）。
    EXPECT_EQ(CalcQuantTokenScaleBytes(1088, 1, 11, false), 1344);
    EXPECT_EQ(CalcQuantTokenScaleBytes(1088, 1, 11, true), 1408);
    EXPECT_EQ(CalcQuantTokenScaleBytes(1088, 2, 11, false), 832);
    EXPECT_EQ(CalcQuantTokenScaleBytes(1088, 2, 11, true), 896);
    EXPECT_EQ(CalcCombineTokenBytes(1088, 2, true), 1344);
    EXPECT_EQ(CalcCombineTokenBytes(1088, 2, false), 2176);
}

TEST_F(MegaMoeArch35TilingTest, UrmaH64AlignmentQuantModes)
{
    for (int64_t h : {1024, 1088, 1152, 1216, 1984, 2048, 8128, 8192}) {
        SCOPED_TRACE(h);
        RunUrmaTilingCase(128, 128, 128, 256, 0, 2048, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS,
                          0, 0, 1, h);
        RunA8W4FormatCase(ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ_C0_32, ge::GRAPH_SUCCESS, 46802984UL, h);
        RunA4W4NzFormatCase(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_FRACTAL_NZ_C0_32, ge::GRAPH_SUCCESS, 62138408UL, h);
    }
}

TEST_F(MegaMoeArch35TilingTest, UrmaH64AlignmentKeepsRangeAndGranularity)
{
    for (int64_t h : {960, 1056, 8256}) {
        SCOPED_TRACE(h);
        RunUrmaTilingCase(128, 128, 128, 256, 0, 2048, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED,
                          0, 0, 1, h);
    }
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_NZ_URMA_InvalidWeight1Format)
{
    RunA4W4NzFormatCase(ge::FORMAT_FRACTAL_NZ_C0_32, ge::FORMAT_FRACTAL_NZ_C0_32, ge::GRAPH_FAILED, 0UL);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_A4W4_NZ_URMA_InvalidWeight2Format)
{
    RunA4W4NzFormatCase(ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND, ge::GRAPH_FAILED, 0UL);
}

static void RunPerExpertTensorListCase(uint32_t weightTwoTensorCount, ge::graphStatus expectedStatus)
{
    struct MegaMoeCompileInfo {
    } compileInfo;

    using TensorDescription = gert::TilingContextPara::TensorDescription;
    std::vector<TensorDescription> inputs{
        {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{128, 8}, {128, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        {{{128, 8}, {128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    for (uint32_t tensorIdx = 0; tensorIdx < 4U; ++tensorIdx) {
        inputs.push_back({{{2048, 4096}, {2048, 4096}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    }
    for (uint32_t tensorIdx = 0; tensorIdx < weightTwoTensorCount; ++tensorIdx) {
        inputs.push_back({{{4096, 1024}, {4096, 1024}}, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND});
    }
    for (uint32_t tensorIdx = 0; tensorIdx < 4U; ++tensorIdx) {
        inputs.push_back({{{2048, 64, 2}, {2048, 64, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    }
    for (uint32_t tensorIdx = 0; tensorIdx < 4U; ++tensorIdx) {
        inputs.push_back({{{4096, 16, 2}, {4096, 16, 2}}, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND});
    }

    const std::vector<TensorDescription> outputs{
        {{{128, 4096}, {128, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
        {{{4}, {4}}, ge::DT_INT32, ge::FORMAT_ND},
    };
    const std::vector<gert::TilingContextPara::OpAttr> attrs{
        {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(ARCH35_CCL_BUFFER_SIZE)},
        {"max_recv_token_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"dispatch_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
        {"dispatch_quant_out_dtype",
         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))},
        {"shared_expert_quant_out_dtype",
         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
        {"combine_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
        {"num_max_tokens_per_rank", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
        {"activation", Ops::Transformer::AnyValue::CreateFrom<std::string>("swiglu")},
        {"activation_params",
         Ops::Transformer::AnyValue::CreateFrom<std::vector<float>>({std::numeric_limits<float>::max()})},
        {"activation_out_dtype",
         Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
        {"transpose_weight1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"transpose_weight2", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"weight1_interleave", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
        {"topo_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        {"rank_num_per_server", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        {"topk_weights_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
    };
    const std::vector<uint32_t> inputInstanceNum{
        1U, 1U, 1U, 1U, 4U, weightTwoTensorCount, 4U, 4U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U,
    };
    const std::vector<uint32_t> outputInstanceNum{1U, 1U};
    gert::TilingContextPara tilingContextPara("MegaMoe", inputs, outputs, attrs, inputInstanceNum, outputInstanceNum,
                                              &compileInfo);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    const uint64_t expectedTilingKey = expectedStatus == ge::GRAPH_SUCCESS ? 42607652UL : 0UL;
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, expectedStatus, expectedTilingKey);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_FP8E4M3FN_URMA_PerExpertTensorLists)
{
    RunPerExpertTensorListCase(4U, ge::GRAPH_SUCCESS);
}

TEST_F(MegaMoeArch35TilingTest, H4096_BS128_FP8E4M3FN_URMA_PerExpertTensorListCountMismatch)
{
    RunPerExpertTensorListCase(3U, ge::GRAPH_FAILED);
}

TEST_F(MegaMoeArch35TilingTest, UrmaServerCombineReduce)
{
    RunUrmaTilingCase(128, 128, 128, 128, 0, 2048, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_SUCCESS, 2);
}

TEST_F(MegaMoeArch35TilingTest, UrmaRejectsMteCombineDedup)
{
    RunUrmaTilingCase(128, 128, 128, 128, 0, 2048, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED, 1);
}

TEST_F(MegaMoeArch35TilingTest, ServerCombineModeDoesNotEnableMteDedup)
{
    using namespace MegaMoeImpl;
    EXPECT_EQ(DEDUP_MODE_SERVER_COMBINE, 4);
    EXPECT_TRUE(IsServerCombineDedupOn(DEDUP_MODE_SERVER_COMBINE));
    EXPECT_FALSE(IsDispatchDedupOn(DEDUP_MODE_SERVER_COMBINE));
    EXPECT_FALSE(IsCombineDedupOn(DEDUP_MODE_SERVER_COMBINE));
    for (int32_t mode = 0; mode <= 3; ++mode) {
        EXPECT_FALSE(IsServerCombineDedupOn(mode));
    }
    MegaMoeTilingData tiling{};
    tiling.compute.topoType = TOPO_TYPE_URMA;
    tiling.compute.dedupMode = DEDUP_MODE_SERVER_COMBINE;
    tiling.compute.epWorldSize = 2;
    tiling.compute.mGroupsPerWave = 1;
    tiling.compute.maxOutputSize = 1024;
    FinalizeTestLayoutParams(tiling);
    EXPECT_TRUE(tiling.comm.serverCombineReduce);
    EXPECT_TRUE(tiling.comm.topkWeightsPrefetch);
    EXPECT_FALSE(tiling.compute.ring.dispatch);
    EXPECT_FALSE(tiling.compute.ring.gmm1);
    EXPECT_FALSE(tiling.compute.ring.activation);
    EXPECT_FALSE(tiling.compute.ring.gmm2);
}

TEST_F(MegaMoeArch35TilingTest, ServerCombineReduceRejectsMte)
{
    RunUrmaTilingCase(128, 128, 128, 128, 0, 2048, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED, 2,
                      0, 0);
}

TEST_F(MegaMoeArch35TilingTest, ServerCombineReduceRejectsQuantizedCombine)
{
    RunUrmaTilingCase(128, 128, 128, 128, 0, 2048, "swiglu", {std::numeric_limits<float>::max()}, ge::GRAPH_FAILED, 2,
                      4);
}

TEST_F(MegaMoeArch35TilingTest, ServerCombineOmitsLegacyReceiveRegion)
{
    using namespace MegaMoeImpl;
    MegaMoeTilingData params{};
    params.compute.numMaxTokensPerRank = 256;
    params.compute.topK = 8;
    params.compute.h = 4096;
    params.compute.moeExpertPerRank = 8;
    params.compute.epWorldSize = 2;
    params.compute.topoType = TOPO_TYPE_URMA;
    params.rankNumPerServer = 1;
    FinalizeTestLayoutParams(params);
    const auto legacy = PeermemInfo(params.comm);
    EXPECT_EQ(legacy.combineSend.size, 256LL * 8 * 4096 * 2);
    params.compute.dedupMode = DEDUP_MODE_SERVER_COMBINE;
    params.compute.aicNum = 28;
    FinalizeTestLayoutParams(params);
    const auto relay = PeermemInfo(params.comm);
    EXPECT_EQ(relay.combineSend.size, 0);
    EXPECT_GT(relay.sourceReady.size, 0);
    EXPECT_EQ(relay.combineSend.offset, -1);
    EXPECT_EQ(relay.combineSend.ptr, nullptr);
    EXPECT_EQ(relay.totalSize, relay.combineReceive.offset + relay.combineReceive.size);
}

TEST_F(MegaMoeArch35TilingTest, ServerCombineCountAndContributionSlotsStayWithinRegions)
{
    using namespace MegaMoeImpl;
    // 4104 仅检验布局算术，不表示 tiling 已放宽 H 的约束。
    for (uint32_t h : {4096U, 4104U}) {
        for (int64_t tokenCapacity : {1, 31, 32, 255, 256, 257}) {
            for (uint32_t topK : {1U, 32U}) {
                for (uint32_t serverNum : {2U, 4U, 40U}) {
                    MegaMoeTilingData tiling{};
                    tiling.compute.topoType = TOPO_TYPE_URMA;
                    tiling.compute.dedupMode = DEDUP_MODE_SERVER_COMBINE;
                    tiling.compute.numMaxTokensPerRank = tokenCapacity;
                    tiling.compute.epWorldSize = serverNum * 2;
                    tiling.rankNumPerServer = 2;
                    tiling.compute.topK = topK;
                    tiling.compute.h = h;
                    tiling.compute.aicNum = 48;
                    FinalizeTestLayoutParams(tiling);
                    PeermemInfo layout(tiling.comm);
                    const int64_t readyRowBytes = CalcServerCombineReadyRowBytes(tokenCapacity);
                    EXPECT_EQ(readyRowBytes, Ops::Base::CeilAlign((tokenCapacity + 1) * 8, ALIGN_32));
                    EXPECT_LE(tokenCapacity * 8 + 8, readyRowBytes);
                    EXPECT_EQ(layout.dispatchFlag.size, Ops::Base::CeilAlign(serverNum * readyRowBytes, ALIGN_512));
                    EXPECT_EQ(layout.sourceReady.size, layout.dispatchFlag.size);
                    const int64_t tokenBytes = static_cast<int64_t>(h) * SIZE_OF_BF16;
                    const int64_t slots = topK;
                    const int64_t tokens = serverNum * tokenCapacity;
                    EXPECT_EQ(layout.contribution.size, Ops::Base::CeilAlign(tokens * slots * tokenBytes, ALIGN_512));
                    // 专家贡献数据和 ready 均按 K 个槽分配，回传接收区独立按 S 分配。
                    EXPECT_EQ(layout.contributionReady.size, Ops::Base::CeilAlign(tokens * topK * 8, ALIGN_512));
                    EXPECT_LE(((tokens - 1) * slots + 1) * tokenBytes, layout.contribution.size);
                    EXPECT_LE(((tokens - 1) * slots + topK) * tokenBytes, layout.contribution.size);
                    EXPECT_EQ(layout.sourceReady.offset, layout.dispatchFlag.offset + layout.dispatchFlag.size);
                    EXPECT_EQ(layout.contributionReady.offset, layout.sourceReady.offset + layout.sourceReady.size);
                    EXPECT_EQ(layout.contribution.offset,
                              layout.contributionReady.offset + layout.contributionReady.size);
                    EXPECT_EQ(layout.combineReceive.offset, layout.contribution.offset + layout.contribution.size);
                    const int64_t clearBytes =
                        layout.dispatchFlag.size + layout.sourceReady.size + layout.contributionReady.size;
                    EXPECT_EQ(layout.dispatchFlag.offset + clearBytes, layout.contribution.offset);
                    WorkspaceLayoutArch35 workspace(tiling.compute);
                    EXPECT_EQ(workspace.dispatchRevWeightsOffset, INVALID_WORKSPACE_OFFSET);
                    auto prefetchTiling = tiling;
                    prefetchTiling.topkWeightsPrefetch = 1;
                    FinalizeTestLayoutParams(prefetchTiling);
                    WorkspaceLayoutArch35 prefetchWorkspace(prefetchTiling.compute);
                    EXPECT_NE(prefetchWorkspace.dispatchRevWeightsOffset, INVALID_WORKSPACE_OFFSET);
                    EXPECT_EQ(workspace.serverCombineResetBytes,
                              Ops::Base::CeilAlign(serverNum * INT_CACHELINE * static_cast<int64_t>(sizeof(int32_t)),
                                                   ALIGN_512));
                    EXPECT_EQ(workspace.combineRelayCompletedCountOffset + workspace.serverCombineResetBytes,
                              workspace.tokenIdsOffset);
                    const int64_t rowStride = CalcServerCombineTokenListStrideBytes(tokenCapacity);
                    EXPECT_EQ(rowStride % sizeof(int32_t), 0);
                    EXPECT_EQ((tokenCapacity + 1) * static_cast<int64_t>(sizeof(int32_t)), rowStride);
                    const int64_t listsSize = Ops::Base::CeilAlign(serverNum * rowStride, ALIGN_512);
                    EXPECT_LE((serverNum - 1) * rowStride + (tokenCapacity + 1) * 4, listsSize);
                    EXPECT_EQ(workspace.tokenBitmapsOffset, workspace.tokenIdsOffset + listsSize);
                    const int64_t bitmapBytes = Ops::Base::CeilAlign(
                        serverNum * tokenCapacity * static_cast<int64_t>(sizeof(uint32_t)), ALIGN_512);
                    EXPECT_EQ(workspace.workspaceSize, workspace.tokenBitmapsOffset + bitmapBytes);
                    EXPECT_EQ(layout.combineReceive.size, Ops::Base::CeilAlign(tokens * tokenBytes, ALIGN_512));
                    EXPECT_EQ(layout.totalSize, layout.combineReceive.offset + layout.combineReceive.size);
                    EXPECT_FALSE(tiling.compute.ring.dispatch);
                    EXPECT_FALSE(tiling.compute.ring.gmm1);
                    EXPECT_FALSE(tiling.compute.ring.activation);
                    EXPECT_FALSE(tiling.compute.ring.gmm2);
                    // 检查全部通信区域；未分配区域不占空间，已分配区域按协议连续推进。
                    const GmMemoryRegion regions[] = {
                        layout.rankSync,          layout.syncCount,       layout.maskRecv,     layout.expertCountRecv,
                        layout.quantTokenScale,   layout.dispatchReceive, layout.dispatchFlag, layout.sourceReady,
                        layout.contributionReady, layout.combineSend,     layout.contribution, layout.combineReceive};
                    int64_t cursor = 0;
                    for (const auto& region : regions) {
                        if (region.offset < 0) {
                            EXPECT_EQ(region.size, 0);
                        } else {
                            EXPECT_EQ(region.offset, cursor);
                            cursor += region.size;
                        }
                        EXPECT_EQ(region.ptr, nullptr);
                    }
                    EXPECT_EQ(cursor, layout.totalSize);
                    if (h == 4096U && tokenCapacity == 1 && topK == 1U && serverNum == 2U) {
                        std::vector<uint8_t> commStorage(static_cast<size_t>(layout.totalSize + ALIGN_512));
                        const PeermemInfo bound(tiling.comm, commStorage.data());
                        EXPECT_EQ(bound.sourceReady.ptr, commStorage.data() + bound.sourceReady.offset);
                        EXPECT_EQ(bound.contributionReady.ptr, commStorage.data() + bound.contributionReady.offset);
                        EXPECT_EQ(bound.contribution.ptr, commStorage.data() + bound.contribution.offset);
                        EXPECT_EQ(bound.combineReceive.ptr, commStorage.data() + bound.combineReceive.offset);
                        std::vector<uint8_t> computeStorage(static_cast<size_t>(workspace.workspaceSize));
                        WorkspaceInfoArch35 info;
                        info.Bind(computeStorage.data(), workspace);
                        EXPECT_EQ(info.combineRelayCompletedCountPtr,
                                  computeStorage.data() + workspace.combineRelayCompletedCountOffset);
                        EXPECT_EQ(info.tokenIdsPtr, computeStorage.data() + workspace.tokenIdsOffset);
                        EXPECT_EQ(info.tokenBitmapsPtr, computeStorage.data() + workspace.tokenBitmapsOffset);
                        EXPECT_EQ(info.serverCombineResetBytes, workspace.serverCombineResetBytes);
                    }
                }
            }
        }
    }
}

static MegaMoeTilingData MakeMteRingLayoutCase()
{
    MegaMoeTilingData tilingData{};
    tilingData.compute.bs = 128;
    tilingData.compute.h = 4096;
    tilingData.compute.hiddenDim = 2048;
    tilingData.compute.epWorldSize = 4;
    tilingData.compute.numMaxTokensPerRank = 8192;
    tilingData.compute.maxOutputSize = 32768;
    tilingData.compute.topK = 8;
    tilingData.compute.aicNum = 28;
    tilingData.compute.moeExpertPerRank = 4;
    tilingData.compute.topoType = MegaMoeImpl::TOPO_TYPE_MTE;
    tilingData.moeGmmMode = MegaMoeImpl::GMM_MODE_A8W4_NZ;
    tilingData.compute.mGroupsPerWave = 14;
    tilingData.compute.combineSyncSlotCountPerExpert =
        Ops::Base::CeilDiv<uint64_t>(tilingData.compute.maxOutputSize, 256U);
    FinalizeTestLayoutParams(tilingData);
    return tilingData;
}

static void ExpectRoutedBufferRows(MegaMoeTilingData tilingData, int64_t rows)
{
    FinalizeTestLayoutParams(tilingData);
    MegaMoeImpl::WorkspaceLayoutArch35 layout(tilingData.compute);
    const int64_t scaleGroup = 64;
    const int64_t storageAlign = 512;
    const int64_t dispatchElementsPerByte = MegaMoeImpl::IsA4W4GmmMode(tilingData.moeGmmMode) ? 2LL : 1LL;
    const int64_t dispatchRows =
        tilingData.compute.dedupMode == 0 ? rows : static_cast<int64_t>(tilingData.compute.maxOutputSize);
    EXPECT_EQ(tilingData.compute.ring.dispatch,
              tilingData.compute.ring.activation && tilingData.compute.dedupMode == 0);
    const int64_t dispatchBytes = dispatchRows * tilingData.compute.h / dispatchElementsPerByte;
    const int64_t dispatchScaleBytes =
        dispatchRows * Ops::Base::CeilDiv(static_cast<int64_t>(tilingData.compute.h), scaleGroup) * 2LL;
    const int64_t activationBytes = rows * tilingData.compute.hiddenDim / 2LL;
    const int64_t activationScaleBytes =
        rows * Ops::Base::CeilDiv(static_cast<int64_t>(tilingData.compute.hiddenDim / 2U), scaleGroup) * 2LL;
    EXPECT_EQ(layout.dispatchRevScaleOffset - layout.dispatchRevDataOffset,
              Ops::Base::CeilAlign(dispatchBytes, storageAlign));
    const int64_t afterDispatchScale =
        layout.dispatchRevWeightsOffset >= 0 ? layout.dispatchRevWeightsOffset : layout.activationQuantDataOffset;
    EXPECT_EQ(afterDispatchScale - layout.dispatchRevScaleOffset,
              Ops::Base::CeilAlign(dispatchScaleBytes, storageAlign));
    EXPECT_EQ(layout.activationQuantScaleOffset - layout.activationQuantDataOffset,
              Ops::Base::CeilAlign(activationBytes, storageAlign));
    EXPECT_EQ(layout.expertRecvTokenCountOffset - layout.activationQuantScaleOffset,
              Ops::Base::CeilAlign(activationScaleBytes, storageAlign));
    EXPECT_LE(layout.dispatchRevScaleOffset + Ops::Base::CeilAlign(dispatchScaleBytes, storageAlign),
              layout.activationQuantDataOffset);
    const int64_t firstGmmResult = layout.gmm1MmadResOffset >= 0 ? layout.gmm1MmadResOffset : layout.gmm2MmadResOffset;
    EXPECT_LE(layout.activationQuantScaleOffset + Ops::Base::CeilAlign(activationScaleBytes, storageAlign),
              firstGmmResult);
    if (layout.gmm1MmadResOffset >= 0) {
        const int64_t afterGmm1 =
            layout.gmm1MmadResOffset + static_cast<int64_t>(2) * rows * tilingData.compute.hiddenDim;
        EXPECT_EQ(layout.gmm2MmadResOffset, Ops::Base::CeilAlign(afterGmm1, storageAlign));
    }
    const int64_t afterGmm2 = layout.gmm1TileStatusOffset >= 0     ? layout.gmm1TileStatusOffset :
                              layout.dedupRowDescOffset >= 0       ? layout.dedupRowDescOffset :
                              layout.maskSlotOffset >= 0           ? layout.maskSlotOffset :
                              layout.sharedExpertResultOffset >= 0 ? layout.sharedExpertResultOffset :
                                                                     layout.workspaceSize;
    const int64_t gmm2Rows = MegaMoeImpl::IsCombineDedupOn(tilingData.compute.dedupMode) ?
                                 static_cast<int64_t>(tilingData.compute.maxOutputSize) :
                                 rows;
    EXPECT_GE(layout.gmm2MmadResOffset, 0);
    EXPECT_EQ(afterGmm2 - layout.gmm2MmadResOffset,
              Ops::Base::CeilAlign(static_cast<int64_t>(2) * gmm2Rows * tilingData.compute.h, storageAlign));
}

TEST_F(MegaMoeArch35TilingTest, MteRingsUseWaveBudgetAndRetainGlobalMetadata)
{
    auto tilingData = MakeMteRingLayoutCase();
    const int64_t ringRows = 3LL * 14LL * 256LL;
    ASSERT_TRUE(tilingData.compute.ring.activation);
    EXPECT_EQ((tilingData.compute.ring.activation ? static_cast<int64_t>(tilingData.compute.ring.rows) :
                                                    tilingData.compute.maxOutputSize),
              ringRows);
    ExpectRoutedBufferRows(tilingData, ringRows);
    MegaMoeImpl::WorkspaceLayoutArch35 first(tilingData.compute);
    EXPECT_EQ(first.flagActivationToGmm2Offset - first.metaInfoOffset, 32768LL * 32LL);

    tilingData.compute.maxOutputSize *= 2U;
    FinalizeTestLayoutParams(tilingData);
    MegaMoeImpl::WorkspaceLayoutArch35 second(tilingData.compute);
    EXPECT_EQ((tilingData.compute.ring.activation ? static_cast<int64_t>(tilingData.compute.ring.rows) :
                                                    tilingData.compute.maxOutputSize),
              ringRows);
    ExpectRoutedBufferRows(tilingData, ringRows);
    EXPECT_EQ(second.flagActivationToGmm2Offset - second.metaInfoOffset, 65536LL * 32LL);
    EXPECT_GT(second.workspaceSize, first.workspaceSize);
    EXPECT_LT(second.workspaceSize - first.workspaceSize, 32768LL * tilingData.compute.h);

    tilingData.compute.mGroupsPerWave = 7;
    FinalizeTestLayoutParams(tilingData);
    EXPECT_EQ((tilingData.compute.ring.activation ? static_cast<int64_t>(tilingData.compute.ring.rows) :
                                                    tilingData.compute.maxOutputSize),
              ringRows / 2LL);
    ExpectRoutedBufferRows(tilingData, ringRows / 2LL);
}

TEST_F(MegaMoeArch35TilingTest, MteRingsOnlyAllocateActualGmm1GmResults)
{
    auto tilingData = MakeMteRingLayoutCase();
    for (const uint8_t mode : {MegaMoeImpl::GMM_MODE_A8W8_ND, MegaMoeImpl::GMM_MODE_A8W8_NZ,
                               MegaMoeImpl::GMM_MODE_A4W4_ND, MegaMoeImpl::GMM_MODE_A4W4_NZ}) {
        tilingData.moeGmmMode = mode;
        tilingData.topkWeightsPrefetch = 0;
        FinalizeTestLayoutParams(tilingData);
        MegaMoeImpl::WorkspaceLayoutArch35 noPrefetch(tilingData.compute);
        EXPECT_EQ(noPrefetch.gmm1MmadResOffset, MegaMoeImpl::INVALID_WORKSPACE_OFFSET);
        ExpectRoutedBufferRows(tilingData, 3LL * 14LL * 256LL);
        tilingData.topkWeightsPrefetch = 1;
        tilingData.compute.maxTilesPerExpert = 1024;
        FinalizeTestLayoutParams(tilingData);
        MegaMoeImpl::WorkspaceLayoutArch35 prefetch(tilingData.compute);
        EXPECT_GE(prefetch.gmm1MmadResOffset, 0);
        EXPECT_EQ(prefetch.gmm1TileStatusElementCount, (4LL * 1024LL + 1LL) * 16LL);
        ExpectRoutedBufferRows(tilingData, 3LL * 14LL * 256LL);
    }
}

static void ExpectMteDedupLayouts(MegaMoeTilingData tilingData)
{
    FinalizeTestLayoutParams(tilingData);
    const MegaMoeImpl::WorkspaceLayoutArch35 layout(tilingData.compute);
    const bool ringEnabled = tilingData.compute.ring.activation;
    const bool dispatchRingEnabled = ringEnabled && tilingData.compute.dedupMode == 0;
    const bool gmm2RingEnabled = ringEnabled && !MegaMoeImpl::IsCombineDedupOn(tilingData.compute.dedupMode);
    EXPECT_EQ(tilingData.compute.ring.dispatch, dispatchRingEnabled);
    EXPECT_EQ(tilingData.compute.ring.gmm2, gmm2RingEnabled);
    EXPECT_GE(layout.gmm2MmadResOffset, 0);
    if (tilingData.compute.topoType == MegaMoeImpl::TOPO_TYPE_MTE) {
        EXPECT_EQ(layout.gmm2MmadResOffset % 512LL, 0LL);
        if (MegaMoeImpl::IsCombineDedupOn(tilingData.compute.dedupMode)) {
            // 专家粒度 Combine 不限制专家内部的 Dispatch 前进，mode1/3 都必须保留完整输入。
            const int64_t inputElementsPerByte = tilingData.compute.dispatchFp4 ? 2LL : 1LL;
            EXPECT_EQ(layout.dispatchRevScaleOffset - layout.dispatchRevDataOffset,
                      Ops::Base::CeilAlign<int64_t>(
                          tilingData.compute.maxOutputSize * tilingData.compute.h / inputElementsPerByte, 512LL));
            const int64_t afterDispatchScale = layout.dispatchRevWeightsOffset >= 0 ? layout.dispatchRevWeightsOffset :
                                                                                      layout.activationQuantDataOffset;
            const int64_t inputScaleBytes =
                tilingData.compute.maxOutputSize * Ops::Base::CeilDiv<int64_t>(tilingData.compute.h, 64LL) * 2LL;
            EXPECT_EQ(afterDispatchScale - layout.dispatchRevScaleOffset,
                      Ops::Base::CeilAlign<int64_t>(inputScaleBytes, 512LL));

            auto otherDedupMode = tilingData;
            otherDedupMode.compute.dedupMode = tilingData.compute.dedupMode == 1 ? 3 : 1;
            FinalizeTestLayoutParams(otherDedupMode);
            const MegaMoeImpl::WorkspaceLayoutArch35 otherLayout(otherDedupMode.compute);
            EXPECT_EQ(layout.dispatchRevDataOffset, otherLayout.dispatchRevDataOffset);
            EXPECT_EQ(layout.dispatchRevScaleOffset, otherLayout.dispatchRevScaleOffset);
            EXPECT_EQ(layout.dispatchRevWeightsOffset, otherLayout.dispatchRevWeightsOffset);
            EXPECT_EQ(layout.activationQuantDataOffset, otherLayout.activationQuantDataOffset);
            EXPECT_EQ(layout.flagActivationToGmm2Offset, otherLayout.flagActivationToGmm2Offset);
            EXPECT_EQ(layout.gmm1MmadResOffset, otherLayout.gmm1MmadResOffset);
            EXPECT_EQ(layout.gmm2MmadResOffset, otherLayout.gmm2MmadResOffset);
            EXPECT_EQ(layout.dedupRowDescOffset, otherLayout.dedupRowDescOffset);
            EXPECT_EQ(layout.sharedExpertResultOffset, otherLayout.sharedExpertResultOffset);
            EXPECT_EQ(layout.workspaceSize, otherLayout.workspaceSize);
        }
    }
}

TEST_F(MegaMoeArch35TilingTest, MteRingsDedupKeepPersistentStageInputsAndOtherFallbacks)
{
    for (const int32_t dedupMode : {1, 2, 3}) {
        auto tilingData = MakeMteRingLayoutCase();
        tilingData.compute.dedupMode = dedupMode;
        FinalizeTestLayoutParams(tilingData);
        EXPECT_TRUE(tilingData.compute.ring.activation);
        EXPECT_EQ((tilingData.compute.ring.activation ? static_cast<int64_t>(tilingData.compute.ring.rows) :
                                                        tilingData.compute.maxOutputSize),
                  3LL * 14LL * 256LL);
        ExpectRoutedBufferRows(tilingData, 3LL * 14LL * 256LL);
        for (const int32_t prefetch : {0, 1}) {
            tilingData.topkWeightsPrefetch = prefetch;
            ExpectMteDedupLayouts(tilingData);
        }
    }
    for (const int64_t combineQuantMode : {3LL, 4LL}) {
        auto tilingData = MakeMteRingLayoutCase();
        tilingData.combineQuantMode = combineQuantMode;
        FinalizeTestLayoutParams(tilingData);
        EXPECT_FALSE(tilingData.compute.ring.activation);
        ExpectRoutedBufferRows(tilingData, 32768LL);
        tilingData.compute.dedupMode = 3;
        ExpectMteDedupLayouts(tilingData);
    }
    auto urmaData = MakeMteRingLayoutCase();
    urmaData.compute.topoType = MegaMoeImpl::TOPO_TYPE_URMA;
    urmaData.rankNumPerServer = 2;
    FinalizeTestLayoutParams(urmaData);
    EXPECT_FALSE(urmaData.compute.ring.activation);
    EXPECT_EQ((urmaData.compute.ring.activation ? static_cast<int64_t>(urmaData.compute.ring.rows) :
                                                  urmaData.compute.maxOutputSize),
              32768LL);
    MegaMoeImpl::WorkspaceLayoutArch35 urma(urmaData.compute);
    EXPECT_EQ(urma.dispatchRevScaleOffset - urma.dispatchRevDataOffset, 32768LL * 4096LL);
    urmaData.compute.dedupMode = 3;
    ExpectMteDedupLayouts(urmaData);

    auto unsetWave = MakeMteRingLayoutCase();
    unsetWave.compute.mGroupsPerWave = 0;
    FinalizeTestLayoutParams(unsetWave);
    EXPECT_FALSE(unsetWave.compute.ring.activation);
    ExpectRoutedBufferRows(unsetWave, 32768LL);
    unsetWave.compute.dedupMode = 3;
    ExpectMteDedupLayouts(unsetWave);
}

TEST_F(MegaMoeArch35TilingTest, MteCommunicationDedupKeepsFullInputAcrossQuantAndWeightModes)
{
    for (const uint8_t mode : {MegaMoeImpl::GMM_MODE_A8W8_ND, MegaMoeImpl::GMM_MODE_A4W4_ND}) {
        auto tilingData = MakeMteRingLayoutCase();
        tilingData.compute.h = 1056;
        tilingData.moeGmmMode = mode;
        for (const int32_t dedupMode : {0, 1, 2, 3}) {
            tilingData.compute.dedupMode = dedupMode;
            for (const int32_t prefetch : {0, 1}) {
                tilingData.topkWeightsPrefetch = prefetch;
                ExpectRoutedBufferRows(tilingData, 3LL * 14LL * 256LL);
                ExpectMteDedupLayouts(tilingData);
            }
        }
    }
}

TEST_F(MegaMoeArch35TilingTest, MteDedupPersistentAreasRetainClippedLogicalRows)
{
    for (const uint32_t maxOutput : {1U, 255U, 256U, 257U}) {
        auto tilingData = MakeMteRingLayoutCase();
        tilingData.compute.maxOutputSize = maxOutput;
        tilingData.compute.combineSyncSlotCountPerExpert = Ops::Base::CeilDiv<uint64_t>(maxOutput, 256U);
        tilingData.compute.dedupMode = 3;
        tilingData.topkWeightsPrefetch = 0;
        ExpectRoutedBufferRows(tilingData, maxOutput);
        ExpectMteDedupLayouts(tilingData);
        FinalizeTestLayoutParams(tilingData);
        const MegaMoeImpl::WorkspaceLayoutArch35 layout(tilingData.compute);
        EXPECT_FALSE(tilingData.compute.ring.dispatch);
        EXPECT_FALSE(tilingData.compute.ring.gmm2);
        EXPECT_GE(layout.gmm2MmadResOffset, 0);
        EXPECT_EQ(layout.activationQuantDataOffset - layout.dispatchRevWeightsOffset,
                  Ops::Base::CeilAlign(static_cast<int64_t>(maxOutput) * 32LL, 512LL));
        EXPECT_EQ(layout.flagActivationToGmm2Offset - layout.metaInfoOffset,
                  Ops::Base::CeilAlign(static_cast<int64_t>(maxOutput) * 32LL, 512LL));
    }
}

TEST_F(MegaMoeArch35TilingTest, MteCombineDedupKeepsExpertReadyAndOneFullResultArea)
{
    for (const uint32_t maxOutput : {1U, 255U, 256U, 257U, 32768U}) {
        for (const int32_t dedupMode : {1, 3}) {
            auto tilingData = MakeMteRingLayoutCase();
            tilingData.compute.maxOutputSize = maxOutput;
            tilingData.compute.dedupMode = dedupMode;
            tilingData.compute.sharedExpertNum = 1;
            tilingData.compute.combineSyncSlotCountPerExpert = Ops::Base::CeilDiv<uint64_t>(maxOutput, 256U);
            FinalizeTestLayoutParams(tilingData);
            const MegaMoeImpl::WorkspaceLayoutArch35 layout(tilingData.compute);
            const int64_t readyBytes = static_cast<int64_t>(sizeof(int32_t)) * tilingData.compute.moeExpertPerRank *
                                       tilingData.compute.aicNum * MegaMoeImpl::INT_CACHELINE;
            EXPECT_EQ(layout.gmm2CombineSyncCounterOffset, MegaMoeImpl::INVALID_WORKSPACE_OFFSET);
            ASSERT_GE(layout.gmm2ReadyOffset, layout.flagActivationToGmm2Offset);
            EXPECT_EQ(layout.sharedExpertGmm2TileCounterOffset - layout.gmm2ReadyOffset, readyBytes);
            const int64_t resetRegionEnd = layout.flagActivationToGmm2Offset +
                                           static_cast<int64_t>(sizeof(int32_t)) * layout.flagResetElementCount;
            EXPECT_LE(layout.gmm2ReadyOffset + readyBytes, resetRegionEnd);
            EXPECT_LE(resetRegionEnd, layout.gmm1MmadResOffset);
            const bool ringEnabled = tilingData.compute.ring.activation;
            const int64_t rows = ringEnabled ? 3LL * 14LL * 256LL : static_cast<int64_t>(maxOutput);
            EXPECT_GE(layout.gmm2MmadResOffset, 0);
            EXPECT_EQ(layout.dedupRowDescOffset - layout.gmm2MmadResOffset,
                      Ops::Base::CeilAlign(2LL * maxOutput * tilingData.compute.h, 512LL));
            ExpectRoutedBufferRows(tilingData, rows);
            ExpectMteDedupLayouts(tilingData);

            // GMM2 始终使用同一个结果区；容量随去重与循环配置变化，未取得基址时不构造地址。
            MegaMoeImpl::WorkspaceInfoArch35 info;
            info.Bind(nullptr, layout);
            EXPECT_EQ(info.gmm2CombineSyncCounterPtr, nullptr);
            EXPECT_EQ(info.gmm2ReadyPtr, nullptr);
            EXPECT_EQ(info.gmm2MmadResPtr, nullptr);
            EXPECT_EQ(info.flagResetElementCount, layout.flagResetElementCount);
            if (maxOutput == 1U) {
                // 使用小容量实例核对真实基址绑定，避免只覆盖空指针分支。
                std::vector<uint8_t> storage(static_cast<size_t>(layout.workspaceSize));
                info.Bind(storage.data(), layout);
                EXPECT_EQ(info.gmm2MmadResPtr, storage.data() + layout.gmm2MmadResOffset);
                EXPECT_EQ(info.sharedExpertResultPtr, storage.data() + layout.sharedExpertResultOffset);
                EXPECT_LE(layout.gmm2MmadResOffset + 2LL * maxOutput * tilingData.compute.h, layout.dedupRowDescOffset);
            }
        }
    }
}

TEST_F(MegaMoeArch35TilingTest, MteSmallCasesFallBackAtThreeWaveCapacity)
{
    constexpr uint32_t threeWaveRows = 3U * 14U * 256U;
    for (const uint32_t maxOutput : {1U, threeWaveRows - 1U, threeWaveRows, threeWaveRows + 1U}) {
        for (const int32_t dedupMode : {0, 1, 2, 3}) {
            auto tilingData = MakeMteRingLayoutCase();
            tilingData.compute.maxOutputSize = maxOutput;
            tilingData.compute.dedupMode = dedupMode;
            const bool ringEnabled = maxOutput > threeWaveRows;
            const int64_t rows = ringEnabled ? threeWaveRows : maxOutput;
            SCOPED_TRACE(testing::Message() << "maxOutput=" << maxOutput << ", dedup=" << dedupMode);
            FinalizeTestLayoutParams(tilingData);
            EXPECT_EQ(tilingData.compute.ring.activation, ringEnabled);
            EXPECT_EQ(tilingData.compute.ring.gmm1, ringEnabled);
            EXPECT_EQ(tilingData.compute.ring.dispatch, ringEnabled && dedupMode == 0);
            EXPECT_EQ(tilingData.compute.ring.gmm2, ringEnabled && (dedupMode == 0 || dedupMode == 2));
            EXPECT_EQ(tilingData.compute.ring.rows, threeWaveRows);
            EXPECT_EQ((tilingData.compute.ring.activation ? static_cast<int64_t>(tilingData.compute.ring.rows) :
                                                            tilingData.compute.maxOutputSize),
                      rows);
            ExpectRoutedBufferRows(tilingData, rows);
            ExpectMteDedupLayouts(tilingData);
            if (!ringEnabled) {
                FinalizeTestLayoutParams(tilingData);
                const MegaMoeImpl::WorkspaceLayoutArch35 layout(tilingData.compute);
                EXPECT_FALSE(tilingData.compute.ring.dispatch);
                EXPECT_FALSE(tilingData.compute.ring.gmm2);
                EXPECT_GE(layout.gmm2MmadResOffset, 0);
            }
        }
    }
}

TEST_F(MegaMoeArch35TilingTest, MteRingMappingPadsExpertTailsAndWrapsFullGroups)
{
    const uint64_t ringRows = 3ULL * 4ULL * 256ULL;
    uint64_t paddedRow = 0;
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(paddedRow, ringRows), 0U);
    paddedRow += MegaMoeImpl::CalcMtePaddedRowCount(100);
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(paddedRow, ringRows), 256U);
    paddedRow += MegaMoeImpl::CalcMtePaddedRowCount(300);
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(paddedRow, ringRows), 768U);
    paddedRow += MegaMoeImpl::CalcMtePaddedRowCount(50);
    EXPECT_EQ(paddedRow, 1024U);
    EXPECT_EQ(MegaMoeImpl::CalcMtePaddedRowCount(0), 0U);
    EXPECT_EQ(MegaMoeImpl::CalcMtePaddedRowCount(1), 256U);
    EXPECT_EQ(MegaMoeImpl::CalcMtePaddedRowCount(255), 256U);
    EXPECT_EQ(MegaMoeImpl::CalcMtePaddedRowCount(256), 256U);
    EXPECT_EQ(MegaMoeImpl::CalcMtePaddedRowCount(257), 512U);
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(ringRows - 256U, ringRows), ringRows - 256U);
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(ringRows, ringRows), 0U);
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(5U * ringRows + 17U, ringRows), 17U);
    const uint64_t largeGroupRow = (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1ULL) * 256ULL;
    EXPECT_EQ(MegaMoeImpl::CalcMteRingPhysicalRow(largeGroupRow, ringRows), largeGroupRow % ringRows);
}

static MegaMoeTilingData MakeMemoryPlannerCase(uint32_t aicNum, uint8_t gmmMode, int32_t dedupMode,
                                               uint32_t sharedExpertNum, bool independentSharedQuant, int64_t topoType,
                                               int64_t combineQuantMode, bool prefetch)
{
    auto tilingData = MakeMteRingLayoutCase();
    tilingData.compute.bs = 257;
    tilingData.compute.numMaxTokensPerRank = 8193;
    tilingData.compute.epWorldSize = 8;
    tilingData.compute.moeExpertPerRank = 2;
    tilingData.compute.maxOutputSize = 8193U * 8U * 2U;
    tilingData.compute.h = 4096;
    tilingData.compute.hiddenDim = 8192;
    tilingData.compute.aicNum = aicNum;
    tilingData.moeGmmMode = gmmMode;
    tilingData.compute.dedupMode = dedupMode;
    tilingData.compute.sharedExpertNum = sharedExpertNum;
    tilingData.compute.sharedInputIndependent = independentSharedQuant;
    // 独立量化覆盖共享 FP4 与 MoE FP8，以及共享 FP8 与 MoE FP4 两种方向。
    tilingData.sharedGmmMode = gmmMode;
    if (independentSharedQuant) {
        tilingData.sharedGmmMode =
            MegaMoeImpl::IsA4W4GmmMode(gmmMode) ? MegaMoeImpl::GMM_MODE_A8W8_ND : MegaMoeImpl::GMM_MODE_A4W4_ND;
    }
    tilingData.compute.topoType = topoType;
    tilingData.rankNumPerServer =
        topoType == MegaMoeImpl::TOPO_TYPE_URMA ? 4U : static_cast<uint32_t>(tilingData.compute.epWorldSize);
    tilingData.combineQuantMode = combineQuantMode;
    tilingData.topkWeightsPrefetch = prefetch ? 1 : 0;
    const int64_t gmm1Groups = Ops::Base::CeilDiv<int64_t>(4LL * aicNum, tilingData.compute.hiddenDim / 256LL);
    const int64_t gmm2Groups = Ops::Base::CeilDiv<int64_t>(aicNum, tilingData.compute.h / 256LL);
    tilingData.compute.mGroupsPerWave = gmm1Groups > gmm2Groups ? gmm1Groups : gmm2Groups;
    const int64_t maxMGroups = Ops::Base::CeilDiv<int64_t>(tilingData.compute.maxOutputSize, 256LL);
    tilingData.compute.maxTilesPerExpert =
        prefetch ? Ops::Base::CeilAlign<int64_t>(maxMGroups * (tilingData.compute.hiddenDim / 256LL), 16LL) : 0U;
    const int64_t combineGroups =
        Ops::Base::CeilDiv<int64_t>(tilingData.compute.numMaxTokensPerRank * tilingData.compute.epWorldSize, 256LL);
    tilingData.compute.combineSyncSlotCountPerExpert = combineGroups > aicNum ? combineGroups : aicNum;
    FinalizeTestLayoutParams(tilingData);
    return tilingData;
}

static void ExpectPreparedLayoutParams(MegaMoeTilingData tilingData)
{
    FinalizeTestLayoutParams(tilingData);
    const auto& params = tilingData.compute;
    EXPECT_EQ(params.dispatchFp4, MegaMoeImpl::IsA4W4GmmMode(tilingData.moeGmmMode));
    EXPECT_EQ(params.reserveGmm1Output,
              tilingData.moeGmmMode == MegaMoeImpl::GMM_MODE_A8W4_NZ || tilingData.topkWeightsPrefetch == 1);
    EXPECT_EQ(params.reserveCumsum, MegaMoeImpl::IsW4GmmMode(tilingData.moeGmmMode));
    EXPECT_EQ(params.needsGmmEpilogue,
              MegaMoeImpl::IsW4GmmMode(tilingData.moeGmmMode) ||
                  (tilingData.compute.sharedExpertNum > 0 && MegaMoeImpl::IsW4GmmMode(tilingData.sharedGmmMode)));
    EXPECT_EQ(params.combineQuant, tilingData.combineQuantMode != MegaMoeImpl::COMBINE_NO_QUANT);
    EXPECT_EQ(params.topkWeightsPrefetch, tilingData.topkWeightsPrefetch == 1);
    EXPECT_EQ(params.sharedInputElementsPerByte, MegaMoeImpl::IsA4W4GmmMode(tilingData.sharedGmmMode) ? 2U : 1U);
    EXPECT_EQ(params.activationOutputDivisor, MegaMoeImpl::GetActivationOutputDivisor(tilingData.actMode));
    EXPECT_EQ(tilingData.comm.numMaxTokensPerRank, params.numMaxTokensPerRank);
    EXPECT_EQ(tilingData.comm.h, params.h);
    EXPECT_EQ(tilingData.comm.moeExpertPerRank, params.moeExpertPerRank);
    EXPECT_EQ(tilingData.comm.epWorldSize, params.epWorldSize);
    EXPECT_EQ(tilingData.comm.topK, params.topK);
    EXPECT_EQ(tilingData.comm.topoType, params.topoType);
    EXPECT_EQ(tilingData.comm.elemsPerByte, params.dispatchFp4 ? 2U : 1U);
    EXPECT_EQ(tilingData.comm.topkWeightsPrefetch, params.topkWeightsPrefetch ||
                                                       MegaMoeImpl::IsCombineDedupOn(params.dedupMode) ||
                                                       MegaMoeImpl::IsServerCombineDedupOn(params.dedupMode));
    EXPECT_EQ(tilingData.comm.serverCombineReduce, MegaMoeImpl::IsServerCombineDedupOn(params.dedupMode));
    const bool ringEnabled = params.topoType == MegaMoeImpl::TOPO_TYPE_MTE && !params.combineQuant &&
                             params.mGroupsPerWave > 0 && params.maxOutputSize > 3LL * params.mGroupsPerWave * 256LL;
    EXPECT_EQ(params.ring.gmm1, ringEnabled);
    EXPECT_EQ(params.ring.activation, ringEnabled);
    EXPECT_EQ(params.ring.dispatch, ringEnabled && params.dedupMode == 0);
    EXPECT_EQ(params.ring.gmm2, ringEnabled && !MegaMoeImpl::IsCombineDedupOn(params.dedupMode));
    EXPECT_EQ(params.ring.rows, static_cast<uint64_t>(3LL * params.mGroupsPerWave * 256LL));
    if (params.topoType == MegaMoeImpl::TOPO_TYPE_MTE && MegaMoeImpl::IsCombineDedupOn(params.dedupMode)) {
        ExpectMteDedupLayouts(tilingData);
    }
}

static void ExpectSharedMemoryBudgetCoversExact(MegaMoeTilingData tilingData)
{
    FinalizeTestLayoutParams(tilingData);
    // 本测试直接给未知执行特征取保守值，不复刻 Torch 的业务入参转换。
    auto budgetParams = tilingData.compute;
    budgetParams.bs = tilingData.compute.numMaxTokensPerRank;
    budgetParams.aicNum = 36;
    budgetParams.sharedExpertNum = 4;
    budgetParams.hiddenDim = 2LL * MegaMoeImpl::GetActivationOutputWidth(&tilingData);
    budgetParams.activationOutputDivisor = 2;
    budgetParams.mGroupsPerWave =
        MegaMoeImpl::CalcMemoryMGroupsPerWave(budgetParams.hiddenDim / 2LL, budgetParams.h, budgetParams.aicNum);
    budgetParams.maxTilesPerExpert =
        MegaMoeImpl::CalcMemoryMaxTilesPerExpert(budgetParams.maxOutputSize, budgetParams.hiddenDim);
    const int64_t maxTokenGroups =
        Ops::Base::CeilDiv<int64_t>(budgetParams.numMaxTokensPerRank * budgetParams.epWorldSize, 256LL);
    budgetParams.combineSyncSlotCountPerExpert = maxTokenGroups > 36 ? maxTokenGroups : 36;
    if (budgetParams.topoType == MegaMoeImpl::TOPO_TYPE_MTE) {
        budgetParams.dedupMode &= 1;
    }
    budgetParams.dispatchFp4 = false;
    budgetParams.reserveGmm1Output = true;
    budgetParams.reserveCumsum = true;
    budgetParams.needsGmmEpilogue = true;
    budgetParams.sharedInputIndependent = true;
    budgetParams.sharedInputElementsPerByte = 1;
    budgetParams.ring = MegaMoeImpl::CalcCclStageRingConfig(budgetParams);
    auto dispatchOnParams = budgetParams;
    if (budgetParams.topoType == MegaMoeImpl::TOPO_TYPE_MTE) {
        dispatchOnParams.dedupMode |= 2;
    }
    dispatchOnParams.ring = MegaMoeImpl::CalcCclStageRingConfig(dispatchOnParams);
    const MegaMoeImpl::CclComputeLayout dispatchOffBudget(budgetParams);
    const MegaMoeImpl::CclComputeLayout dispatchOnBudget(dispatchOnParams);
    const auto& budget =
        dispatchOnBudget.workspaceSize > dispatchOffBudget.workspaceSize ? dispatchOnBudget : dispatchOffBudget;
    const MegaMoeImpl::WorkspaceLayoutArch35 exact(tilingData.compute);
    EXPECT_GE(budget.workspaceSize, exact.workspaceSize);
    auto commBudgetParams = tilingData.comm;
    commBudgetParams.elemsPerByte = 1;
    EXPECT_GE(MegaMoeImpl::CalcCclCommLayout(commBudgetParams).totalSize,
              MegaMoeImpl::CalcCclCommLayout(tilingData.comm).totalSize);

    if (MegaMoeImpl::IsDispatchDedupOn(tilingData.compute.dedupMode) ||
        MegaMoeImpl::IsCombineDedupOn(tilingData.compute.dedupMode)) {
        EXPECT_EQ(dispatchOnBudget.dispatchRevScaleOffset - dispatchOnBudget.dispatchRevDataOffset,
                  Ops::Base::CeilAlign(1LL * tilingData.compute.maxOutputSize * tilingData.compute.h, 512LL));
    }
    const int64_t threeWaveRows = 3LL * budgetParams.mGroupsPerWave * 256LL;
    const bool budgetRingEnabled = budgetParams.topoType == MegaMoeImpl::TOPO_TYPE_MTE && !budgetParams.combineQuant &&
                                   budgetParams.mGroupsPerWave > 0 && budgetParams.maxOutputSize > threeWaveRows;
    const int64_t budgetResultRows = budgetRingEnabled && !MegaMoeImpl::IsCombineDedupOn(budgetParams.dedupMode) ?
                                         threeWaveRows :
                                         budgetParams.maxOutputSize;
    const int64_t afterGmm2 = budget.gmm1TileStatusOffset >= 0     ? budget.gmm1TileStatusOffset :
                              budget.dedupRowDescOffset >= 0       ? budget.dedupRowDescOffset :
                              budget.maskSlotOffset >= 0           ? budget.maskSlotOffset :
                              budget.sharedExpertResultOffset >= 0 ? budget.sharedExpertResultOffset :
                                                                     budget.workspaceSize;
    EXPECT_GE(budget.gmm2MmadResOffset, 0);
    EXPECT_EQ(afterGmm2 - budget.gmm2MmadResOffset,
              Ops::Base::CeilAlign(2LL * budgetResultRows * tilingData.compute.h, 512LL));
}

TEST_F(MegaMoeArch35TilingTest, PreparedLayoutParamsMatchExecutionFeatures)
{
    for (const uint32_t aicNum : {32U, 36U}) {
        for (const uint8_t gmmMode :
             {MegaMoeImpl::GMM_MODE_A8W8_ND, MegaMoeImpl::GMM_MODE_A8W4_NZ, MegaMoeImpl::GMM_MODE_A4W4_ND}) {
            for (const int32_t dedupMode : {0, 1, 2, 3}) {
                for (const uint32_t sharedExpertNum : {0U, 1U, 4U}) {
                    for (const bool independentSharedQuant : {false, true}) {
                        for (const bool prefetch : {false, true}) {
                            SCOPED_TRACE(testing::Message()
                                         << "aic=" << aicNum << ", mode=" << +gmmMode << ", dedup=" << dedupMode
                                         << ", shared=" << sharedExpertNum << ", independent=" << independentSharedQuant
                                         << ", prefetch=" << prefetch);
                            const auto tilingData =
                                MakeMemoryPlannerCase(aicNum, gmmMode, dedupMode, sharedExpertNum,
                                                      independentSharedQuant, MegaMoeImpl::TOPO_TYPE_MTE, 0, prefetch);
                            ExpectPreparedLayoutParams(tilingData);
                            ExpectSharedMemoryBudgetCoversExact(tilingData);
                        }
                    }
                }
            }
        }
    }
}

TEST_F(MegaMoeArch35TilingTest, SharedMemoryBudgetCoversQuantCombineAndUrma)
{
    for (const int64_t topoType : {MegaMoeImpl::TOPO_TYPE_MTE, MegaMoeImpl::TOPO_TYPE_URMA}) {
        for (const int64_t combineQuantMode : {0LL, 3LL, 4LL}) {
            for (const int32_t dedupMode : {0, 2, 4}) {
                if ((topoType == MegaMoeImpl::TOPO_TYPE_URMA && dedupMode == 2) ||
                    (dedupMode == 4 && (topoType != MegaMoeImpl::TOPO_TYPE_URMA || combineQuantMode != 0))) {
                    continue;
                }
                for (const uint32_t aicNum : {32U, 36U}) {
                    for (const uint8_t gmmMode : {MegaMoeImpl::GMM_MODE_A8W8_ND, MegaMoeImpl::GMM_MODE_A8W4_NZ,
                                                  MegaMoeImpl::GMM_MODE_A4W4_ND}) {
                        for (const uint32_t sharedExpertNum : {0U, 1U, 4U}) {
                            for (const bool prefetch : {false, true}) {
                                SCOPED_TRACE(testing::Message()
                                             << "topo=" << topoType << ", combineQuant=" << combineQuantMode
                                             << ", dedup=" << dedupMode << ", aic=" << aicNum << ", mode=" << +gmmMode
                                             << ", shared=" << sharedExpertNum << ", prefetch=" << prefetch);
                                const auto tilingData =
                                    MakeMemoryPlannerCase(aicNum, gmmMode, dedupMode, sharedExpertNum, true, topoType,
                                                          combineQuantMode, prefetch);
                                ExpectPreparedLayoutParams(tilingData);
                                ExpectSharedMemoryBudgetCoversExact(tilingData);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST_F(MegaMoeArch35TilingTest, SharedMemoryBudgetUsesDefaultCapacityAndClippedPersistentRows)
{
    for (const int32_t dedupMode : {0, 1, 2, 3}) {
        auto tilingData = MakeMemoryPlannerCase(36, MegaMoeImpl::GMM_MODE_A8W8_ND, dedupMode, 1, true,
                                                MegaMoeImpl::TOPO_TYPE_MTE, 0, true);
        const int64_t defaultMaxOutput =
            tilingData.compute.numMaxTokensPerRank * tilingData.compute.epWorldSize *
            (tilingData.compute.topK < tilingData.compute.moeExpertPerRank ? tilingData.compute.topK :
                                                                             tilingData.compute.moeExpertPerRank);
        EXPECT_EQ(defaultMaxOutput, tilingData.compute.maxOutputSize);
        ExpectSharedMemoryBudgetCoversExact(tilingData);
        for (const uint32_t maxOutput : {1U, 255U, 256U, 257U, 3840U, 3841U, 6912U, 6913U}) {
            SCOPED_TRACE(testing::Message() << "dedup=" << dedupMode << ", maxOutput=" << maxOutput);
            tilingData.compute.maxOutputSize = maxOutput;
            tilingData.compute.maxTilesPerExpert = Ops::Base::CeilAlign<int64_t>(
                Ops::Base::CeilDiv(maxOutput, 256U) * (tilingData.compute.hiddenDim / 256LL), 16LL);
            ExpectPreparedLayoutParams(tilingData);
            ExpectSharedMemoryBudgetCoversExact(tilingData);
        }
    }
}

TEST_F(MegaMoeArch35TilingTest, SharedMemoryBudgetCoversGeluAndGatedActivation)
{
    for (const auto actMode : {MegaMoeImpl::MegaMoeActMode::GELU, MegaMoeImpl::MegaMoeActMode::SWIGLU}) {
        for (const uint32_t aicNum : {32U, 36U}) {
            for (const int32_t dedupMode : {0, 1, 2, 3}) {
                for (const uint32_t sharedExpertNum : {0U, 4U}) {
                    for (const bool prefetch : {false, true}) {
                        auto tilingData = MakeMemoryPlannerCase(aicNum, MegaMoeImpl::GMM_MODE_A8W8_ND, dedupMode,
                                                                sharedExpertNum, true, MegaMoeImpl::TOPO_TYPE_MTE,
                                                                MegaMoeImpl::COMBINE_NO_QUANT, prefetch);
                        tilingData.actMode = static_cast<uint8_t>(actMode);
                        tilingData.compute.hiddenDim =
                            4096U * MegaMoeImpl::GetActivationOutputDivisor(tilingData.actMode);
                        tilingData.compute.mGroupsPerWave = static_cast<uint32_t>(MegaMoeImpl::CalcMemoryMGroupsPerWave(
                            tilingData.compute.hiddenDim, tilingData.compute.h, aicNum));
                        tilingData.compute.maxTilesPerExpert =
                            prefetch ? static_cast<uint32_t>(MegaMoeImpl::CalcMemoryMaxTilesPerExpert(
                                           tilingData.compute.maxOutputSize, tilingData.compute.hiddenDim)) :
                                       0U;
                        ExpectPreparedLayoutParams(tilingData);
                        ExpectSharedMemoryBudgetCoversExact(tilingData);
                        tilingData.compute.topoType = MegaMoeImpl::TOPO_TYPE_URMA;
                        tilingData.rankNumPerServer = 4U;
                        tilingData.compute.dedupMode = 0;
                        ExpectPreparedLayoutParams(tilingData);
                        ExpectSharedMemoryBudgetCoversExact(tilingData);
                    }
                }
            }
        }
    }
    EXPECT_EQ(MegaMoeImpl::CalcMemoryMGroupsPerWave(4096, 4096, 36), 9);
}

TEST_F(MegaMoeArch35TilingTest, SharedCclCommPlannerMapsRegionsAndParameters)
{
    for (const auto topoType : {MegaMoeImpl::TOPO_TYPE_MTE, MegaMoeImpl::TOPO_TYPE_URMA}) {
        for (const bool prefetch : {false, true}) {
            for (const int32_t dedupMode : {0, 4}) {
                if (dedupMode == 4 && topoType != MegaMoeImpl::TOPO_TYPE_URMA) {
                    continue;
                }
                auto tilingData = MakeMemoryPlannerCase(36, MegaMoeImpl::GMM_MODE_A8W8_ND, dedupMode, 1, true, topoType,
                                                        MegaMoeImpl::COMBINE_NO_QUANT, prefetch);
                const auto planned = MegaMoeImpl::CalcCclCommLayout(tilingData.comm);
                const MegaMoeImpl::PeermemInfo actual(tilingData.comm);
                const MegaMoeImpl::MemoryRegionLayout plannedRegions[] = {
                    planned.rankSync,          planned.syncCount,       planned.maskRecv,     planned.expertCountRecv,
                    planned.quantTokenScale,   planned.dispatchReceive, planned.dispatchFlag, planned.sourceReady,
                    planned.contributionReady, planned.combineSend,     planned.contribution, planned.combineReceive};
                const MegaMoeImpl::GmMemoryRegion actualRegions[] = {
                    actual.rankSync,          actual.syncCount,       actual.maskRecv,     actual.expertCountRecv,
                    actual.quantTokenScale,   actual.dispatchReceive, actual.dispatchFlag, actual.sourceReady,
                    actual.contributionReady, actual.combineSend,     actual.contribution, actual.combineReceive};
                for (uint32_t index = 0; index < 12U; ++index) {
                    EXPECT_EQ(plannedRegions[index].offset, actualRegions[index].offset);
                    EXPECT_EQ(plannedRegions[index].size, actualRegions[index].size);
                    EXPECT_EQ(actualRegions[index].ptr, nullptr);
                }
                EXPECT_EQ(planned.totalSize, actual.totalSize);
            }
        }
    }
}

} // namespace MegaMoeUT
