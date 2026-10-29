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
 * \file test_moe_distribute_dispatch_setup_infershape.cpp
 * \brief infershape ut
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_shape_context_faker.h"

namespace MoeDistributeDispatchSetupInferShapeUT {

static const std::string OP_NAME = "MoeDistributeDispatchSetup";

class MoeDistributeDispatchSetupInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeDispatchSetupInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeDispatchSetupInferShapeTest TearDown" << std::endl;
    }
};

TEST_F(MoeDistributeDispatchSetupInferShapeTest, InferShapeQuant0Basic)
{
    gert::StorageShape xShape = {{8, 7168}, {}};
    gert::StorageShape expertIdsShape = {{8, 8}, {}};
    gert::StorageShape yOutShape = {{64, 7168}, {}};
    gert::StorageShape expandIdxOutShape = {{64}, {}};
    gert::StorageShape commCmdInfoOutShape = {{1536}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME, {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yOutShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expandIdxOutShape, ge::DT_INT32, ge::FORMAT_ND},
         {commCmdInfoOutShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{64, 7168}, {64}, {1536}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeDistributeDispatchSetupInferShapeTest, InferShapeQuantModeNonZero)
{
    gert::StorageShape xShape = {{4, 100}, {}};
    gert::StorageShape expertIdsShape = {{4, 2}, {}};
    gert::StorageShape yOutShape = {{8, 512}, {}};
    gert::StorageShape expandIdxOutShape = {{8}, {}};
    gert::StorageShape commCmdInfoOutShape = {{640}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME, {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yOutShape, ge::DT_INT8, ge::FORMAT_ND},
         {expandIdxOutShape, ge::DT_INT32, ge::FORMAT_ND},
         {commCmdInfoOutShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{8, 512}, {8}, {640}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeDistributeDispatchSetupInferShapeTest, InferShapeSharedExpertRank)
{
    gert::StorageShape xShape = {{8, 512}, {}};
    gert::StorageShape expertIdsShape = {{8, 4}, {}};
    gert::StorageShape yOutShape = {{40, 512}, {}};
    gert::StorageShape expandIdxOutShape = {{32}, {}};
    gert::StorageShape commCmdInfoOutShape = {{768}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME, {{xShape, ge::DT_BF16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yOutShape, ge::DT_BF16, ge::FORMAT_ND},
         {expandIdxOutShape, ge::DT_INT32, ge::FORMAT_ND},
         {commCmdInfoOutShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{40, 512}, {32}, {768}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeDistributeDispatchSetupInferShapeTest, InferShapeGlobalBsAndNegLocalExpert)
{
    gert::StorageShape xShape = {{8, 256}, {}};
    gert::StorageShape expertIdsShape = {{8, 4}, {}};
    gert::StorageShape yOutShape = {{32, 256}, {}};
    gert::StorageShape expandIdxOutShape = {{32}, {}};
    gert::StorageShape commCmdInfoOutShape = {{448}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME, {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yOutShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expandIdxOutShape, ge::DT_INT32, ge::FORMAT_ND},
         {commCmdInfoOutShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(64)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 4}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 256}, {32}, {448}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeDistributeDispatchSetupInferShapeTest, InferShapeDim1Dynamic)
{
    gert::StorageShape xShape = {{-1}, {}};
    gert::StorageShape expertIdsShape = {{-1}, {}};
    gert::StorageShape yOutShape = {{-1, -1}, {}};
    gert::StorageShape expandIdxOutShape = {{-1}, {}};
    gert::StorageShape commCmdInfoOutShape = {{-1}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        OP_NAME, {{xShape, ge::DT_FLOAT16, ge::FORMAT_ND}, {expertIdsShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{yOutShape, ge::DT_FLOAT16, ge::FORMAT_ND},
         {expandIdxOutShape, ge::DT_INT32, ge::FORMAT_ND},
         {commCmdInfoOutShape, ge::DT_INT32, ge::FORMAT_ND}},
        {{"group_ep", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"ep_rank_id", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"moe_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(32)},
         {"expert_shard_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"shared_expert_rank_num", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"global_bs", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 2}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{1, 0}, {1}, {-16}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

} // namespace MoeDistributeDispatchSetupInferShapeUT
