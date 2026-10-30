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
#include "mc2_infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "infer_datatype_context_faker.h"

class DistributeBarrierExtendInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DistributeBarrierExtendInfershape SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "DistributeBarrierExtendInfershape TearDown" << std::endl;
    }
};

TEST_F(DistributeBarrierExtendInfershape, InferShape0)
{
    gert::StorageShape contextShape = {{1, 2052}, {1, 2052}};
    gert::StorageShape xRefShape = {{32, 7168}, {}};

    gert::InfershapeContextPara infershapeContextPara(
        "DistributeBarrierExtend",
        {{contextShape, ge::DT_INT32, ge::FORMAT_ND}, {xRefShape, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(288)}});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    std::vector<std::vector<int64_t>> expectOutputShape = {{32, 7168}};
    Mc2ExecuteTestCase(infershapeContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(DistributeBarrierExtendInfershape, InferDataType)
{
    auto opImpl =
        gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DistributeBarrierExtend");
    ASSERT_NE(opImpl, nullptr);
    auto data_type_func = opImpl->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_0 = ge::DT_INT32;
    ge::DataType input_1 = ge::DT_FLOAT16;
    ge::DataType output_0 = ge::DT_FLOAT16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(2, 1)
                              .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_0, &input_1})
                              .OutputDataTypes({&output_0})
                              .Build();

    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_FLOAT16);
}

TEST_F(DistributeBarrierExtendInfershape, InferDataTypeBf16)
{
    auto opImpl =
        gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("DistributeBarrierExtend");
    ASSERT_NE(opImpl, nullptr);
    auto data_type_func = opImpl->infer_datatype;
    ASSERT_NE(data_type_func, nullptr);

    ge::DataType input_0 = ge::DT_INT32;
    ge::DataType input_1 = ge::DT_BF16;
    ge::DataType output_0 = ge::DT_BF16;
    auto context_holder = gert::InferDataTypeContextFaker()
                              .NodeIoNum(2, 1)
                              .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                              .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                              .InputDataTypes({&input_0, &input_1})
                              .OutputDataTypes({&output_0})
                              .Build();

    auto context = context_holder.GetContext<gert::InferDataTypeContext>();
    EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
}
