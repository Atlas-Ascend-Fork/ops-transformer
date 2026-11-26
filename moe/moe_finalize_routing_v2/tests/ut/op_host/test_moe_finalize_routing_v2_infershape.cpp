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

namespace {

std::vector<gert::InfershapeContextPara::OpAttr> DropPadModeAttr(int64_t mode)
{
    return {{"drop_pad_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(mode)},
            {"zero_expert_range",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
            {"copy_expert_range",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
            {"constant_expert_range",
             Ops::Transformer::AnyValue::CreateFrom<std::vector<int64_t>>(std::vector<int64_t>{-1, -1})},
            {"k", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}};
}
} // namespace

class MoeFinalizeRoutingV2Proto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2Proto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeFinalizeRoutingV2Proto TearDown" << std::endl;
    }
};

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_infershape_1)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 1, 5}, {6, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(1));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_error_shape_0)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5, 1}, {3, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_error_shape_1)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 1, 5}, {6, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(5));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_error_shape_2)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 1, 5}, {6, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5, 1}, {3, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_error_shape_3)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 1, 5}, {6, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5, 1}, {3, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(1));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_error_shape_4) // drop_pad_mode=1
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeFinalizeRoutingV2",
        {
            {{{6, 1, 6, 1, 5}, {6, 1, 6, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        DropPadModeAttr(1));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_error_shape_5) // drop_pad_mode=0
{
    gert::InfershapeContextPara infershapeContextPara(
        "MoeFinalizeRoutingV2",
        {
            {{{6, 1, 6, 1, 5}, {6, 1, 6, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_error_shape_6)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 1, 5}, {6, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 1}, {6, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_error_shape_7)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 1, 5}, {6, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5, 1}, {3, 5, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_error_shape_8)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 1, 5}, {6, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2, 1}, {3, 2, 1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_error_shape_9)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 1, 5}, {6, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2, 1}, {3, 2, 1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_infershape_mode2)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(2));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_infershape_mode3)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 1, 5}, {6, 1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(3));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_row_idx)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_bias)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_skip1)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_scales)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{-1, 5}, {-1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1, -1}, {-1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_error_mismatch_row)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{4, 5}, {4, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{3, 5}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_FAILED, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_dynamic_shape_0)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(1));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_dynamic_shape_1)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{-1, -1, -1}, {-1, -1, -1}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{
                                                                -1,
                                                                -1,
                                                            },
                                                            {
                                                                -1,
                                                                -1,
                                                            }},
                                                           ge::DT_FLOAT,
                                                           ge::FORMAT_ND},
                                                          {{{
                                                                -1,
                                                                -1,
                                                            },
                                                            {
                                                                -1,
                                                                -1,
                                                            }},
                                                           ge::DT_FLOAT,
                                                           ge::FORMAT_ND},
                                                          {{{
                                                                -1,
                                                                -1,
                                                            },
                                                            {
                                                                -1,
                                                                -1,
                                                            }},
                                                           ge::DT_FLOAT,
                                                           ge::FORMAT_ND},
                                                          {{{
                                                                -1,
                                                                -1,
                                                            },
                                                            {
                                                                -1,
                                                                -1,
                                                            }},
                                                           ge::DT_FLOAT,
                                                           ge::FORMAT_ND},
                                                          {{{
                                                                -1,
                                                                -1,
                                                            },
                                                            {
                                                                -1,
                                                                -1,
                                                            }},
                                                           ge::DT_INT32,
                                                           ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(1));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_expanded_x_dim)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{-1, 5}, {-1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_expanded_row_idx_dim)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1}, {-1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_skip1_partial_dim)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-1, 5}, {-1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_skip2_partial_dim)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1, 5}, {-1, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_expert_idx_partial_dim)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-1, 2}, {-1, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-1, -1}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_bias_rank)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_skip1_rank)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_skip2_rank)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_scales_rank)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(MoeFinalizeRoutingV2Proto, moe_finalize_routing_v2_unknown_expert_idx_rank)
{
    gert::InfershapeContextPara infershapeContextPara("MoeFinalizeRoutingV2",
                                                      {
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6}, {6}}, ge::DT_INT32, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 5}, {3, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{6, 5}, {6, 5}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{3, 2}, {3, 2}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                          {{{-2}, {-2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                      },
                                                      {
                                                          {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                      },
                                                      DropPadModeAttr(0));
    std::vector<std::vector<int64_t>> expectOutputShape = {{-2}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

// infer dataType
TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_0)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;
    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref1, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_SUCCESS);
        ASSERT_NE(context, nullptr);

        EXPECT_EQ(context->GetOutputDataType(0), output_ref);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_err1)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref1, &input_ref1, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_err2)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_err3)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref1, &input_ref1, &input_ref, &input_ref, &input_ref, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_err4)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref1, &input_ref, &input_ref, &input_ref, &input_ref1, &input_ref1})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_err5)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto data_type_func = spaceRegistry->GetOpImpl("MoeFinalizeRoutingV2")->infer_datatype;

    if (data_type_func != nullptr) {
        ge::DataType input_ref = ge::DT_FLOAT;
        ge::DataType input_ref1 = ge::DT_INT32;
        ge::DataType output_ref = ge::DT_FLOAT;
        auto context_holder =
            gert::InferDataTypeContextFaker()
                .NodeIoNum(7, 1)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                .NodeInputTd(0, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                .InputDataTypes({&input_ref, &input_ref1, &input_ref, &input_ref, &input_ref, &input_ref, &input_ref})
                .OutputDataTypes({&output_ref})
                .Build();
        auto context = context_holder.GetContext<gert::InferDataTypeContext>();
        EXPECT_EQ(data_type_func(context), ge::GRAPH_FAILED);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_mxfp4)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto impl = registry->GetOpImpl("MoeFinalizeRoutingV2");
    ASSERT_NE(impl, nullptr);
    const auto infer = impl->infer_datatype;
    ASSERT_NE(infer, nullptr);
    const std::vector<ge::DataType> validTypes = {
        ge::DT_FLOAT4_E2M1, ge::DT_INT32,     ge::DT_BF16,      ge::DT_BF16,      ge::DT_BF16,      ge::DT_FLOAT,
        ge::DT_INT32,       ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_UNDEFINED, ge::DT_FLOAT8_E8M0};
    constexpr size_t validCase = 0;
    const size_t missingScaleCase = validTypes.size();
    // Exercise the valid combination, invalid auxiliary types and a missing scale.
    for (size_t invalidInput = validCase; invalidInput <= missingScaleCase; ++invalidInput) {
        auto types = validTypes;
        if (invalidInput == missingScaleCase) {
            types.back() = ge::DT_UNDEFINED;
        } else if (invalidInput != validCase) {
            types[invalidInput] = types[invalidInput] == ge::DT_UNDEFINED ? ge::DT_BF16 : ge::DT_FLOAT16;
        }
        std::vector<void*> inputs;
        inputs.reserve(types.size());
        for (auto& type : types) {
            inputs.push_back(&type);
        }
        ge::DataType output = ge::DT_UNDEFINED;
        auto holder = gert::InferDataTypeContextFaker()
                          .NodeIoNum(types.size(), 1)
                          .IrInstanceNum(std::vector<uint32_t>(types.size(), 1))
                          .InputDataTypes(inputs)
                          .OutputDataTypes({&output})
                          .Build();
        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_NE(context, nullptr);
        EXPECT_EQ(infer(context), invalidInput == validCase ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED)
            << "invalid input index: " << invalidInput;
        if (invalidInput == validCase) {
            EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
        }
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_ignores_expanded_x_scale_for_float)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto impl = registry->GetOpImpl("MoeFinalizeRoutingV2");
    ASSERT_NE(impl, nullptr);
    ASSERT_NE(impl->infer_datatype, nullptr);

    for (auto scale : {ge::DT_FLOAT8_E8M0, ge::DT_INT32}) {
        std::vector<ge::DataType> types(12, ge::DT_UNDEFINED);
        types[0] = ge::DT_BF16;
        types[1] = ge::DT_INT32;
        types.back() = scale;
        std::vector<void*> inputs;
        for (auto& type : types) {
            inputs.push_back(&type);
        }
        ge::DataType output = ge::DT_UNDEFINED;
        auto holder = gert::InferDataTypeContextFaker()
                          .NodeIoNum(types.size(), 1)
                          .IrInstanceNum(std::vector<uint32_t>(types.size(), 1))
                          .InputDataTypes(inputs)
                          .OutputDataTypes({&output})
                          .Build();
        auto context = holder.GetContext<gert::InferDataTypeContext>();
        ASSERT_EQ(impl->infer_datatype(context), ge::GRAPH_SUCCESS);
        EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    }
}

TEST_F(MoeFinalizeRoutingV2Proto, dtype_infer_mxfp8)
{
    // Named slot indices of the V5 signature (0-based input order), used by the cases below.
    enum Slot : size_t {
        SLOT_X1 = 2,
        SLOT_X2 = 3,
        SLOT_BIAS = 4,
        SLOT_SCALES = 5,
        SLOT_X = 7,
        SLOT_ALPHA1 = 8,
        SLOT_ALPHA2 = 9,
        SLOT_V = 10,
    };

    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    const auto impl = registry->GetOpImpl("MoeFinalizeRoutingV2");
    ASSERT_NE(impl, nullptr);
    const auto infer = impl->infer_datatype;
    ASSERT_NE(infer, nullptr);

    std::vector<ge::DataType> validTypes = {ge::DT_FLOAT8_E4M3FN, ge::DT_INT32,     ge::DT_BF16,
                                            ge::DT_BF16,          ge::DT_BF16,      ge::DT_FLOAT,
                                            ge::DT_INT32,         ge::DT_UNDEFINED, ge::DT_UNDEFINED,
                                            ge::DT_UNDEFINED,     ge::DT_UNDEFINED, ge::DT_FLOAT8_E8M0};

    // The context keeps raw pointers into these variables, so both live in the
    // enclosing scope of every infer() call below.
    std::vector<ge::DataType> types = validTypes;
    ge::DataType output = ge::DT_UNDEFINED;
    auto buildHolder = [&types, &output]() {
        std::vector<void*> inputs;
        inputs.reserve(types.size());
        for (auto& type : types) {
            inputs.push_back(&type);
        }
        return gert::InferDataTypeContextFaker()
            .NodeIoNum(types.size(), 1)
            .IrInstanceNum(std::vector<uint32_t>(types.size(), 1))
            .InputDataTypes(inputs)
            .OutputDataTypes({&output})
            .Build();
    };

    for (auto quantDtype : {ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E5M2}) {
        {
            // The valid combination infers a BF16 output.
            types = validTypes;
            output = ge::DT_UNDEFINED;
            auto holder = buildHolder();
            auto context = holder.GetContext<gert::InferDataTypeContext>();
            EXPECT_EQ(infer(context), ge::GRAPH_SUCCESS);
            EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
        }
        // x1/x2/bias must be BF16 and scales must be FLOAT32 for quantized inputs.
        for (Slot slot : {SLOT_X1, SLOT_X2, SLOT_BIAS, SLOT_SCALES}) {
            types = validTypes;
            types[slot] = ge::DT_FLOAT16;
            output = ge::DT_UNDEFINED;
            auto holder = buildHolder();
            auto context = holder.GetContext<gert::InferDataTypeContext>();
            EXPECT_EQ(infer(context), ge::GRAPH_FAILED) << "aux slot: " << slot;
        }
        // MXFP8 requires empty special expert inputs.
        for (Slot slot : {SLOT_X, SLOT_ALPHA1, SLOT_ALPHA2, SLOT_V}) {
            types = validTypes;
            types[slot] = ge::DT_BF16;
            output = ge::DT_UNDEFINED;
            auto holder = buildHolder();
            auto context = holder.GetContext<gert::InferDataTypeContext>();
            EXPECT_EQ(infer(context), ge::GRAPH_FAILED) << "special expert slot: " << slot;
        }
        {
            // The dequantization scale is required for MXFP8.
            types = validTypes;
            types.back() = ge::DT_UNDEFINED;
            output = ge::DT_UNDEFINED;
            auto holder = buildHolder();
            auto context = holder.GetContext<gert::InferDataTypeContext>();
            EXPECT_EQ(infer(context), ge::GRAPH_FAILED);
        }
        {
            // The scale dtype must be E8M0.
            types = validTypes;
            types.back() = ge::DT_FLOAT;
            output = ge::DT_UNDEFINED;
            auto holder = buildHolder();
            auto context = holder.GetContext<gert::InferDataTypeContext>();
            EXPECT_EQ(infer(context), ge::GRAPH_FAILED);
        }
        {
            // Regression: a non-quantized expanded_x must reject FP8 auxiliary inputs.
            // IsValidType covers only non-quantized paths, so an E4M3FN x1 alongside
            // a BF16 expanded_x fails dtype inference.
            for (Slot slot : {SLOT_X1, SLOT_X2, SLOT_BIAS, SLOT_SCALES}) {
                types = validTypes;
                types[0] = ge::DT_BF16;
                types[slot] = quantDtype;
                output = ge::DT_UNDEFINED;
                auto holder = buildHolder();
                auto context = holder.GetContext<gert::InferDataTypeContext>();
                EXPECT_EQ(infer(context), ge::GRAPH_FAILED) << "non-quantized aux slot: " << slot;
            }
        }
    }
}
