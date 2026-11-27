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
 * \file test_grouped_matmul_add.cpp
 * \brief
 */

#include <iostream>
#include <gtest/gtest.h>
#include "../../../op_host/op_tiling/grouped_matmul_add_tiling.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace ge;

class GroupedMatmulAddTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatmulAddTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatmulAddTiling TearDown" << std::endl;
    }
};

TEST_F(GroupedMatmulAddTiling, test_tiling_bf16)
{
    optiling::GroupedMatmulAddCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd", // op_name
                                              {                   // input info
                                               // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                               {{{1280, 345}, {1280, 345}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{1280, 567}, {1280, 567}}, ge::DT_BF16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                               {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              // output info
                                              {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    int64_t expectTilingKey = 1; // tilngkey
    string expectTilingData =
        "2 128 2 0 1481763717121 5497558139447 1481763718400 5497558139136 1099511627904 34359738432 4294967304 1 0 "
        "1970324836974592 131072 4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GroupedMatmulAddTiling, test_tiling_fp16)
{
    optiling::GroupedMatmulAddCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("GroupedMatmulAdd", // op_name
                                              {                   // input info
                                               // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
                                               {{{1280, 345}, {1280, 345}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{1280, 567}, {1280, 567}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                               {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                               {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                              // output info
                                              {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}}, &compileInfo);
    int64_t expectTilingKey = 0; // tilngkey
    string expectTilingData =
        "2 128 2 0 1481763717121 5497558139447 1481763718400 5497558139136 1099511627904 34359738432 4294967304 1 0 "
        "1970324836974592 131072 4294967297 4294967297 17179869188 0 8589934594 1 0 0 0 0 0 0 0 0 ";
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}

TEST_F(GroupedMatmulAddTiling, test_tiling_bf16_v2)
{
    optiling::GroupedMatmulAddCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulAdd", // op_name
        {                   // input info
         // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {{{1280, 345}, {1280, 345}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{1280, 567}, {1280, 567}}, ge::DT_BF16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        // output info
        {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo);
    int64_t expectTilingKey = 1;                       // tilngkey
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    int groupListTypeMode = 2;
    int groupListTypeIndex = 3;
    string expectTilingData = "";
    for (int i = 0; i < groupListTypeMode; i++) {
        expectTilingData =
            "2 128 2 " + to_string(i) +
            " 1481763717121 5497558139447 1481763718400 5497558139136 "
            "1099511627904 34359738432 4294967304 1 0 1970324836974592 131072 4294967297 4294967297 17179869188 0 "
            "8589934594 1 0 0 0 0 0 0 0 0 ";
        tilingContextPara.attrs_.at(groupListTypeIndex).attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(i);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
    }
}

TEST_F(GroupedMatmulAddTiling, test_tiling_fp16_v2)
{
    optiling::GroupedMatmulAddCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "GroupedMatmulAdd", // op_name
        {                   // input info
         // shape都需要重复一次，比如shape为{16,16}，要填入{{16, 16}, {16, 16}}
         {{{1280, 345}, {1280, 345}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{1280, 567}, {1280, 567}}, ge::DT_FLOAT16, ge::FORMAT_ND},
         {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
         {{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        // output info
        {{{{690, 567}, {690, 567}}, ge::DT_FLOAT, ge::FORMAT_ND}},
        {{"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}},
        &compileInfo);
    int64_t expectTilingKey = 0;                       // tilngkey
    std::vector<size_t> expectWorkspaces = {33554432}; // workspace
    int groupListTypeMode = 2;
    int groupListTypeIndex = 3;
    string expectTilingData = "";
    for (int i = 0; i < groupListTypeMode; i++) {
        expectTilingData =
            "2 128 2 " + to_string(i) +
            " 1481763717121 5497558139447 1481763718400 5497558139136 "
            "1099511627904 34359738432 4294967304 1 0 1970324836974592 131072 4294967297 4294967297 17179869188 0 "
            "8589934594 1 0 0 0 0 0 0 0 0 ";
        tilingContextPara.attrs_.at(groupListTypeIndex).attr_ = Ops::Transformer::AnyValue::CreateFrom<int64_t>(i);
        ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
    }
}

TEST_F(GroupedMatmulAddTiling, ExpandedPathAcceptsLargerYBacking)
{
    optiling::GroupedMatmulAddCompileInfo compileInfo = {};
    gert::TilingContextPara context("GroupedMatmulAdd",
                                    {{{{17, 32}, {17, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{1120}, {1120}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                    {{{{1120}, {1120}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                    {{"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                     {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                     {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                     {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                    &compileInfo);
    TilingInfo result;
    ASSERT_TRUE(ExecuteTiling(context, result));
    EXPECT_EQ(result.tilingKey, 6);
}

TEST_F(GroupedMatmulAddTiling, ExpandedPathRejectsTooSmallYBacking)
{
    optiling::GroupedMatmulAddCompileInfo compileInfo = {};
    gert::TilingContextPara context("GroupedMatmulAdd",
                                    {{{{17, 32}, {17, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{32, 32}, {32, 32}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{1087}, {1087}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                    {{{{1087}, {1087}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
                                    {{"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                     {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                     {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                     {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                    &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(GroupedMatmulAddTiling, ExpandedPathRejectsSchedulingOverflow)
{
    constexpr int64_t M = 2147483647;
    optiling::GroupedMatmulAddCompileInfo compileInfo = {};
    gert::TilingContextPara context("GroupedMatmulAdd",
                                    {{{{M, 16}, {M, 16}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{16, 65520}, {16, 65520}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                     {{{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND},
                                     {{{1, M, 65520}, {1, M, 65520}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                    {{{{1, M, 65520}, {1, M, 65520}}, ge::DT_FLOAT, ge::FORMAT_ND}},
                                    {{"transpose_x", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                     {"transpose_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
                                     {"group_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                     {"group_list_type", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)}},
                                    &compileInfo);
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}
