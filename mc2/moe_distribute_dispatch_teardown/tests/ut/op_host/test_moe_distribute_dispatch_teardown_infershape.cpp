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
 * \file test_moe_distribute_dispatch_teardown_infershape.cpp
 * \brief infershape ut
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MoeDistributeDispatchTeardownInferShapeUT {

static const std::string OP_NAME = "MoeDistributeDispatchTeardown";

class MoeDistributeDispatchTeardownInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeDispatchTeardownInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeDispatchTeardownInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(MoeDistributeDispatchTeardownInferShapeTest, InferShapeMoeExpertPath)
{
    gert::StorageShape xShape = {{8, 7168}, {}};
    gert::StorageShape yShape = {{64, 7168}, {}};
    gert::StorageShape expertIdsShape = {{8, 8}, {}};
    gert::StorageShape commCmdShape = {{1536}, {}};

    gert::InfershapeContextPara para(OP_NAME,
                                     {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {yShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
                                      {commCmdShape, ge::DT_INT32, ge::FORMAT_ND}},
                                     {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                      {{}, ge::DT_INT32, ge::FORMAT_ND},
                                      {{}, ge::DT_INT64, ge::FORMAT_ND}},
                                     {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                      {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                      {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                      {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                      {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

TEST_F(MoeDistributeDispatchTeardownInferShapeTest, InferShapeSharedExpertPath)
{
    gert::StorageShape xShape = {{8, 512}, {}};
    gert::StorageShape yShape = {{40, 512}, {}};
    gert::StorageShape expertIdsShape = {{8, 4}, {}};
    gert::StorageShape commCmdShape = {{768}, {}};

    gert::InfershapeContextPara para(OP_NAME,
                                     {{xShape, ge::DT_BF16, ge::FORMAT_ND},
                                      {yShape, ge::DT_BF16, ge::FORMAT_ND},
                                      {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
                                      {commCmdShape, ge::DT_INT32, ge::FORMAT_ND}},
                                     {{{}, ge::DT_BF16, ge::FORMAT_ND},
                                      {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                      {{}, ge::DT_INT32, ge::FORMAT_ND},
                                      {{}, ge::DT_INT64, ge::FORMAT_ND}},
                                     {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
                                      {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                      {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                      {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                      {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                      {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

TEST_F(MoeDistributeDispatchTeardownInferShapeTest, InferShapeNegLocalExpert)
{
    gert::StorageShape xShape = {{8, 256}, {}};
    gert::StorageShape yShape = {{32, 256}, {}};
    gert::StorageShape expertIdsShape = {{8, 4}, {}};
    gert::StorageShape commCmdShape = {{448}, {}};

    gert::InfershapeContextPara para(OP_NAME,
                                     {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {yShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
                                      {commCmdShape, ge::DT_INT32, ge::FORMAT_ND}},
                                     {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                      {{}, ge::DT_INT32, ge::FORMAT_ND},
                                      {{}, ge::DT_INT64, ge::FORMAT_ND}},
                                     {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                      {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                      {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                      {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                      {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
                                      {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                      {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

TEST_F(MoeDistributeDispatchTeardownInferShapeTest, InferShapeDim1Dynamic)
{
    gert::StorageShape xShape = {{-1}, {}};
    gert::StorageShape yShape = {{-1}, {}};
    gert::StorageShape expertIdsShape = {{-1}, {}};
    gert::StorageShape commCmdShape = {{-1}, {}};

    gert::InfershapeContextPara para(OP_NAME,
                                     {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {yShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
                                      {commCmdShape, ge::DT_INT32, ge::FORMAT_ND}},
                                     {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                      {{}, ge::DT_INT32, ge::FORMAT_ND},
                                      {{}, ge::DT_INT64, ge::FORMAT_ND}},
                                     {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                      {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
                                      {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                      {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

TEST_F(MoeDistributeDispatchTeardownInferShapeTest, InferShapeSharedRankGeWorld)
{
    gert::StorageShape xShape = {{4, 128}, {}};
    gert::StorageShape yShape = {{8, 128}, {}};
    gert::StorageShape expertIdsShape = {{4, 2}, {}};
    gert::StorageShape commCmdShape = {{64}, {}};

    gert::InfershapeContextPara para(OP_NAME,
                                     {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {yShape, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND},
                                      {commCmdShape, ge::DT_INT32, ge::FORMAT_ND}},
                                     {{{}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                      {{}, ge::DT_FLOAT, ge::FORMAT_ND},
                                      {{}, ge::DT_INT32, ge::FORMAT_ND},
                                      {{}, ge::DT_INT64, ge::FORMAT_ND}},
                                     {{"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                      {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(16)},
                                      {"ccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                      {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                      {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"expert_token_nums_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                      {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
                                      {"comm_alg", Ops::Transformer::AnyValue::CreateFrom<std::string>("")}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_SUCCESS);
}

} // namespace MoeDistributeDispatchTeardownInferShapeUT
