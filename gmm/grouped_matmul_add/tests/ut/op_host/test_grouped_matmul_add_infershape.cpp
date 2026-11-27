/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class GroupedMatmulAddInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulAddProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulAddProto TearDown" << std::endl;
    }
};

TEST_F(GroupedMatmulAddInfershape, grouped_matmul_add_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmulAdd",
        {
            // input info
            {{{256, 128}, {256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// output info
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {// attr
         {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 512},
    }; // 预期输出shape
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape); // 框架中已提供该接口
}

TEST_F(GroupedMatmulAddInfershape, grouped_matmul_add_infershape_1)
{
    gert::InfershapeContextPara infershapeContextPara(
        "GroupedMatmulAdd",
        {
            // input info
            {{{256, 128}, {256, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
            {{{256, 512}, {256, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {// output info
         {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {// attr
         {"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});
    std::vector<std::vector<int64_t>> expectOutputShape = {
        {256, 512},
    }; // 预期输出shape
    int groupListTypeMode = 2;
    int groupListTypeIndex = 3;
    for (int i = 0; i < groupListTypeMode; i++) {
        infershapeContextPara.attrs_.at(groupListTypeIndex).attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(i);
        ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
    }
}

TEST_F(GroupedMatmulAddInfershape, NonTransposedXRetainsYShape)
{
    for (const auto dtype : {ge::DT_FLOAT16, ge::DT_BF16}) {
        for (const auto yType : {ge::DT_FLOAT, dtype}) {
            gert::InfershapeContextPara context(
                "GroupedMatmulAdd",
                {{{{17, 32}, {17, 32}}, dtype, ge::FORMAT_ND},
                 {{{32, 48}, {32, 48}}, dtype, ge::FORMAT_ND},
                 {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                 {{{2, 17, 48}, {2, 17, 48}}, yType, ge::FORMAT_ND}},
                {{{{}, {}}, yType, ge::FORMAT_ND}},
                {{"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                 {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                 {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}});
            ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 17, 48}});
        }
    }
}

TEST_F(GroupedMatmulAddInfershape, OutputDataTypeMatchesYAndRejectsMismatches)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto impl = registry->GetOpImpl("GroupedMatmulAdd");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);
    struct TestCase {
        ge::DataType x;
        ge::DataType weight;
        ge::DataType y;
        ge::graphStatus expected;
    };
    const TestCase cases[] = {
        {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::GRAPH_SUCCESS},
        {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::GRAPH_SUCCESS},
        {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::GRAPH_SUCCESS},
        {ge::DT_BF16, ge::DT_BF16, ge::DT_BF16, ge::GRAPH_SUCCESS},
        {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT, ge::GRAPH_FAILED},
        {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_BF16, ge::GRAPH_FAILED},
        {ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT16, ge::GRAPH_FAILED},
    };
    for (const auto& test : cases) {
        auto x = test.x;
        auto weight = test.weight;
        auto groupList = ge::DT_INT64;
        auto y = test.y;
        std::vector<void*> inputs{&x, &weight, &groupList, &y};
        auto holder = gert::InferDataTypeContextFaker()
                          .SetOpType("GroupedMatmulAdd")
                          .NodeIoNum(4, 1)
                          .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                          .InputDataTypes(inputs)
                          .Build();
        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(impl->infer_datatype(context), test.expected);
        if (test.expected == ge::GRAPH_SUCCESS) {
            EXPECT_EQ(context->GetOutputDataType(0), test.y);
        }
    }
}
