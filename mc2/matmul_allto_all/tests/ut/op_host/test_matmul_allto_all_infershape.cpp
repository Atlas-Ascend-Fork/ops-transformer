/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_matmul_allto_all_infershape.cpp
 * \brief infershape ut
 */

#include <iostream>
#include <gtest/gtest.h>
#include "mc2_infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace MatmulAlltoAllInferShapeUT {

static const std::string OP_NAME = "MatmulAlltoAll";

class MatmulAlltoAllInferShapeTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAlltoAllInferShapeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAlltoAllInferShapeTest TearDown" << std::endl;
    }
};

static gert::InfershapeContextPara MakePara(const gert::StorageShape& x1, const gert::StorageShape& x2,
                                            int64_t worldSize, bool transX1, bool transX2,
                                            const std::vector<int64_t>& axes,
                                            int64_t yDtype = static_cast<int64_t>(ge::DT_FLOAT16))
{
    return gert::InfershapeContextPara(
        OP_NAME, {{x1, ge::DT_FLOAT16, ge::FORMAT_ND}, {x2, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{{}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(worldSize)},
         {"all2all_axes", Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(axes)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(yDtype)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_UNDEFINED))},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(transX1)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(transX2)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("ai_cpu")}});
}

TEST_F(MatmulAlltoAllInferShapeTest, InferShapeSuccess)
{
    // out0 = m * world_size = 8*8=64, out1 = ceil(n/world_size)=ceil(16/8)=2
    auto para = MakePara({{8, 64}, {}}, {{64, 16}, {}}, 8, false, false, {-1, -2});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectShape{{64, 2}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectShape);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferShapeTransX2Success)
{
    auto para = MakePara({{8, 64}, {}}, {{16, 64}, {}}, 8, false, true, {-1, -2});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectShape{{64, 2}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectShape);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferShapeDynamicMSuccess)
{
    auto para = MakePara({{-1, 64}, {}}, {{64, 16}, {}}, 8, false, false, {-1, -2});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    std::vector<std::vector<int64_t>> expectShape{{-1, 16}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectShape);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferShapeBadWorldSize)
{
    auto para = MakePara({{8, 64}, {}}, {{64, 16}, {}}, 3, false, false, {-1, -2});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferShapeKMismatch)
{
    auto para = MakePara({{8, 64}, {}}, {{32, 16}, {}}, 8, false, false, {-1, -2});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferShapeTransX1TrueFailed)
{
    auto para = MakePara({{8, 64}, {}}, {{64, 16}, {}}, 8, true, false, {-1, -2});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferShapeBadAxes)
{
    auto para = MakePara({{8, 64}, {}}, {{64, 16}, {}}, 8, false, false, {0, 1});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferShapeBadAxesSize)
{
    auto para = MakePara({{8, 64}, {}}, {{64, 16}, {}}, 8, false, false, {-1});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferShapeKTooLarge)
{
    auto para = MakePara({{8, 70000}, {}}, {{70000, 16}, {}}, 8, false, false, {-1, -2});
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(para, hcomTopologyMockValues, ge::GRAPH_FAILED);
}

TEST_F(MatmulAlltoAllInferShapeTest, InferDataTypeSuccess)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl(OP_NAME.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);
}

} // namespace MatmulAlltoAllInferShapeUT
