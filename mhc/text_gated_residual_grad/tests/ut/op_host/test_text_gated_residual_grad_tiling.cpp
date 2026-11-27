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
#include <cstring>
#include <vector>

#include "../../../op_host/op_tiling/text_gated_residual_grad_tiling.h"
#include "../../../op_kernel/arch35/text_gated_residual_grad_pipeline_arch35.h"
#include "../../../op_kernel/arch35/text_gated_residual_grad_tiling_data_arch35.h"
#include "tiling_case_executor.h"

namespace {
namespace hcgrad = TextGatedResidualGrad;

constexpr int64_t CHANNELS = hcgrad::HC_GRAD_TUNED_CHANNELS;
constexpr int64_t HIDDEN = hcgrad::HC_GRAD_TUNED_HIDDEN_SIZE;
constexpr int64_t PROJECTION = hcgrad::HC_GRAD_TUNED_PROJECTION_SIZE;
constexpr int64_t FEATURES = CHANNELS * HIDDEN;
constexpr int64_t CHUNK_ROWS = hcgrad::HC_GRAD_CHUNK_ROWS;
constexpr int64_t WEIGHT_NZ_THRESHOLD = 3 * CHUNK_ROWS;
constexpr size_t FULL_PLAN = 0;
constexpr size_t TAIL_PLAN = 1;

testing::AssertionResult GetTiling(int64_t tokens, bool a5, ge::DataType dtype,
                                   TextGatedResidualGradTilingDataArch35& data)
{
    using Tensor = gert::TilingContextPara::TensorDescription;
    const std::vector<Tensor> inputs = {
        {{{1, tokens, HIDDEN}, {1, tokens, HIDDEN}}, dtype, ge::FORMAT_ND},
        {{{1, tokens, FEATURES}, {1, tokens, FEATURES}}, dtype, ge::FORMAT_ND},
        {{{PROJECTION, FEATURES}, {PROJECTION, FEATURES}}, dtype, ge::FORMAT_ND},
        {{{FEATURES, PROJECTION}, {FEATURES, PROJECTION}}, dtype, ge::FORMAT_ND},
        {{{1, tokens, CHANNELS}, {1, tokens, CHANNELS}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1, tokens, PROJECTION}, {1, tokens, PROJECTION}}, dtype, ge::FORMAT_ND},
        {{{1, tokens, FEATURES}, {1, tokens, FEATURES}}, dtype, ge::FORMAT_ND},
        {{{1, tokens, CHANNELS}, {1, tokens, CHANNELS}}, dtype, ge::FORMAT_ND},
        {{{CHANNELS, FEATURES}, {CHANNELS, FEATURES}}, dtype, ge::FORMAT_ND},
        {{{FEATURES}, {FEATURES}}, ge::DT_FLOAT, ge::FORMAT_ND},
        {{{1, tokens, CHANNELS}, {1, tokens, CHANNELS}}, dtype, ge::FORMAT_ND},
    };
    const std::vector<Tensor> outputs = {
        {{{1, tokens, FEATURES}, {1, tokens, FEATURES}}, dtype, ge::FORMAT_ND},
        {{{PROJECTION, FEATURES}, {PROJECTION, FEATURES}}, dtype, ge::FORMAT_ND},
        {{{FEATURES, PROJECTION}, {FEATURES, PROJECTION}}, dtype, ge::FORMAT_ND},
        {{{CHANNELS, FEATURES}, {CHANNELS, FEATURES}}, dtype, ge::FORMAT_ND},
        {{{FEATURES}, {FEATURES}}, ge::DT_FLOAT, ge::FORMAT_ND},
    };
    optiling::TextGatedResidualGradCompileInfo compileInfo{};
    const char* soc = a5 ? "Ascend950" : "Ascend910B";
    const uint64_t cubeCores = a5 ? 28U : 20U;
    const uint64_t ubBytes = a5 ? 262144U : 196608U;
    gert::TilingContextPara context("TextGatedResidualGrad", inputs, outputs,
                                    {{"channels", Ops::Transformer::AnyValue::CreateFrom<int64_t>(CHANNELS)}},
                                    &compileInfo, soc, cubeCores, ubBytes, sizeof(data));
    if (a5) {
        context.socInfoString_ = R"({"hardware_info":{"UB_SIZE":262144,"L1_SIZE":524288,)"
                                 R"("L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":262144,)"
                                 R"("CORE_NUM":28,"socVersion":"Ascend950"}})";
    } else {
        context.socInfoString_ = R"({"hardware_info":{"UB_SIZE":196608,"L2_SIZE":33554432,"L1_SIZE":524288,)"
                                 R"("L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":131072,)"
                                 R"("CORE_NUM":20,"cube_core_cnt":20,"vector_core_cnt":40,"socVersion":"Ascend910B"}})";
    }

    TilingInfo info;
    if (!ExecuteTiling(context, info)) {
        return testing::AssertionFailure() << "Tiling failed for " << soc << " with " << tokens << " tokens";
    }
    if (info.tilingDataSize != sizeof(data) || info.workspaceSizes.size() != 1U) {
        return testing::AssertionFailure() << "Unexpected tiling data size " << info.tilingDataSize
                                           << " or workspace count " << info.workspaceSizes.size();
    }
    std::memcpy(&data, info.tilingData.get(), sizeof(data));
    EXPECT_EQ(info.blockNum, cubeCores);
    EXPECT_EQ(data.n, tokens);
    EXPECT_GE(info.workspaceSizes.front(), data.workspaceSize);
    return testing::AssertionSuccess();
}

TEST(TextGatedResidualGradTiling, A2UsesSharedNzScratchBelowThreshold)
{
    TextGatedResidualGradTilingDataArch35 data{};
    ASSERT_TRUE(GetTiling(WEIGHT_NZ_THRESHOLD - hcgrad::HC_GRAD_VECTOR_ALIGN, false, ge::DT_BF16, data));

    EXPECT_EQ(data.scheduleMode, hcgrad::HC_GRAD_FULL_BATCH_SCHEDULE);
    EXPECT_EQ(data.fullBatchRows, CHUNK_ROWS);
    EXPECT_EQ(data.aicNum, 20);
    EXPECT_EQ(data.aivNum, 40);
    EXPECT_GT(data.nnScratchBytes, 0U);
    EXPECT_EQ(data.a2WeightNzOffset, 0U);
    EXPECT_EQ(data.workspaceSize, static_cast<int64_t>(data.nnScratchOffset + data.nnScratchBytes));
}

TEST(TextGatedResidualGradTiling, A2AllocatesSeparateWeightNzScratchAtThreshold)
{
    TextGatedResidualGradTilingDataArch35 data{};
    ASSERT_TRUE(GetTiling(WEIGHT_NZ_THRESHOLD, false, ge::DT_BF16, data));

    EXPECT_EQ(data.scheduleMode, hcgrad::HC_GRAD_FULL_BATCH_SCHEDULE);
    EXPECT_EQ(data.fullBatchRows, CHUNK_ROWS);
    const uint64_t dxScratchBytes =
        static_cast<uint64_t>(CHUNK_ROWS) * hcgrad::HcGradAlign16(PROJECTION + CHANNELS) * sizeof(uint16_t);
    EXPECT_EQ(data.a2WeightNzOffset, data.nnScratchOffset + hcgrad::HcGradAlign(dxScratchBytes));
    ASSERT_GT(data.a2WeightNzOffset, data.nnScratchOffset);
    const auto workspace =
        hcgrad::BuildHcGradA2WeightNzLayout(data.a2WeightNzOffset, CHUNK_ROWS, CHANNELS, PROJECTION, FEATURES);
    EXPECT_EQ(data.nnScratchBytes, workspace.end - data.nnScratchOffset);
    EXPECT_EQ(data.workspaceSize, static_cast<int64_t>(workspace.end));
}

TEST(TextGatedResidualGradTiling, A5WeightGemmUsesFullAndTailChunkRows)
{
    constexpr int64_t TAIL_ROWS = 1024;
    TextGatedResidualGradTilingDataArch35 data{};
    ASSERT_TRUE(GetTiling(3 * CHUNK_ROWS + TAIL_ROWS, true, ge::DT_FLOAT16, data));

    EXPECT_EQ(data.scheduleMode, hcgrad::HC_GRAD_FULL_BATCH_SCHEDULE);
    EXPECT_EQ(data.fullBatchRows, CHUNK_ROWS);
    EXPECT_EQ(data.a2WeightNzOffset, 0U);
    for (const auto stage : {hcgrad::HC_GRAD_NN_STAGE_C1B, hcgrad::HC_GRAD_NN_STAGE_G56}) {
        EXPECT_EQ(data.nnA5[FULL_PLAN][stage].k, CHUNK_ROWS);
        EXPECT_EQ(data.nnA5[TAIL_PLAN][stage].k, TAIL_ROWS);
    }
}

TEST(TextGatedResidualGradTiling, A5SmallShapeUsesTokenSchedule)
{
    constexpr int64_t TOKENS = 64;
    TextGatedResidualGradTilingDataArch35 data{};
    ASSERT_TRUE(GetTiling(TOKENS, true, ge::DT_BF16, data));

    EXPECT_EQ(data.scheduleMode, 1);
    EXPECT_EQ(data.fullBatchRows, 0);
    EXPECT_EQ(data.a2WeightNzOffset, 0U);
    EXPECT_GT(data.scheduleBeatCount, 0);
    EXPECT_EQ(data.nnA5[FULL_PLAN][hcgrad::HC_GRAD_NN_STAGE_C1B].k, TOKENS);
    EXPECT_EQ(data.nnA5[FULL_PLAN][hcgrad::HC_GRAD_NN_STAGE_G56].k, TOKENS);
}
} // namespace
