/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

using namespace std;

namespace InplaceMatmulAllReduceAddRmsNormUT {
class InplaceMatmulAllReduceAddRmsNormArch22TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InplaceMatmulAllReduceAddRmsNormArch22TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InplaceMatmulAllReduceAddRmsNormArch22TilingTest TearDown" << std::endl;
    }
};

TEST_F(InplaceMatmulAllReduceAddRmsNormArch22TilingTest, TestTilingParse)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("InplaceMatmulAllReduceAddRmsNorm");
    ASSERT_NE(opImpl, nullptr);
    auto tilingParseFunc = opImpl->tiling_parse;
    ASSERT_NE(tilingParseFunc, nullptr);
    auto ret = tilingParseFunc(nullptr);
    ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(InplaceMatmulAllReduceAddRmsNormArch22TilingTest, TestTilingBasic)
{
    struct CompileInfo {};
    CompileInfo compileInfo;
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    gert::TilingContextPara tilingContextPara(
        "InplaceMatmulAllReduceAddRmsNorm",
        {
            {{{16, 32}, {16, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 16}, {32, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16, 16}, {1, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16}, {16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{1, 16, 16}, {1, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{1, 16, 16}, {1, 16, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {{"epsilon", Ops::Transformer::AnyValue::CreateFrom<float>(1e-6f)},
         {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("test_group")},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>("sum")},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"antiquant_group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo, "Ascend910B", coreNum, ubSize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
}
} // namespace InplaceMatmulAllReduceAddRmsNormUT
