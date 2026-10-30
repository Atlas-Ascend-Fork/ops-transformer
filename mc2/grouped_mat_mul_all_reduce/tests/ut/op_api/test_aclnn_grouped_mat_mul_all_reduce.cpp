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
 * \file test_aclnn_grouped_mat_mul_all_reduce.cpp
 * \brief
 */

#include <vector>
#include <array>
#include "gtest/gtest.h"

#include "../../../op_api/aclnn_grouped_mat_mul_all_reduce.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
namespace {
class L2GroupedMatMulAllReduceTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "gemm_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "gemm_test TearDown" << endl;
    }
};

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case0)
{
    vector<vector<int64_t>> xDims = {{256, 256}, {1024, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{100}, {256}};
    vector<vector<int64_t>> outDims = {{256, 256}, {1024, 256}};
    vector<int64_t> groupList = {};
    int64_t splitItemOptional = 0;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto x = TensorDesc(xDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
        xGroup.emplace_back(x);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        outGroup.emplace_back(out);
    }
    auto xList = TensorListDesc({xGroup[0], xGroup[1]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case1)
{
    vector<vector<int64_t>> xDims = {{1280, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{100}, {256}};
    vector<vector<int64_t>> outDims = {{256, 256}, {1024, 256}};
    vector<int64_t> groupList = {256, 1280};
    int64_t splitItemOptional = 1;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    auto x = TensorDesc(xDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    xGroup.emplace_back(x);
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);

        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        outGroup.emplace_back(out);
    }
    auto xList = TensorListDesc({xGroup[0]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case2)
{
    vector<vector<int64_t>> xDims = {{256, 256}, {1024, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{100}, {256}};
    vector<vector<int64_t>> outDims = {{1280, 256}};
    vector<int64_t> groupList = {};
    int64_t splitItemOptional = 2;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    auto out = TensorDesc(outDims[0], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    outGroup.emplace_back(out);
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto x = TensorDesc(xDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        xGroup.emplace_back(x);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
    }

    auto xList = TensorListDesc({xGroup[0], xGroup[1]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0]});
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case3)
{
    vector<vector<int64_t>> xDims = {{1280, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{100}, {256}};
    vector<vector<int64_t>> outDims = {{1280, 256}};
    vector<int64_t> groupList = {256, 1280};
    int64_t splitItemOptional = 3;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    auto x = TensorDesc(xDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    xGroup.emplace_back(x);
    auto out = TensorDesc(outDims[0], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    outGroup.emplace_back(out);
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
    }
    auto xList = TensorListDesc({xGroup[0]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0]});
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case4)
{
    vector<vector<int64_t>> xDims = {{256, 256}, {1024, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {100, 256}};
    vector<vector<int64_t>> biasDims = {{256}, {256}};
    vector<vector<int64_t>> outDims = {{256, 256}, {1024, 256}};
    vector<int64_t> groupList = {};
    int64_t splitItemOptional = 0;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    for (size_t i = 0; i < weightDims.size(); i++) {
        auto x = TensorDesc(xDims[i], ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto weight = TensorDesc(weightDims[i], ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_BF16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
        xGroup.emplace_back(x);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        outGroup.emplace_back(out);
    }
    auto xList = TensorListDesc({xGroup[0], xGroup[1]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case5)
{
    vector<vector<int64_t>> xDims = {{256, 256}, {1024, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{100}, {256}};
    vector<vector<int64_t>> outDims = {{256, 256}, {1024, 256}};
    vector<int64_t> groupList = {};
    int64_t splitItemOptional = 5;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    for (size_t i = 0; i < weightDims.size(); i++) {
        auto x = TensorDesc(xDims[i], ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto weight = TensorDesc(weightDims[i], ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_BF16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
        xGroup.emplace_back(x);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        outGroup.emplace_back(out);
    }
    auto xList = TensorListDesc({xGroup[0], xGroup[1]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case7)
{
    vector<vector<int64_t>> xDims = {{256, 256}, {1024, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{0}};
    vector<vector<int64_t>> outDims = {{256, 100}, {1024, 256}};
    vector<int64_t> groupList = {};
    int64_t splitItemOptional = 0;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    for (size_t i = 0; i < weightDims.size(); i++) {
        auto x = TensorDesc(xDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
        xGroup.emplace_back(x);
        weightGroup.emplace_back(weight);
        outGroup.emplace_back(out);
    }
    auto bias = TensorDesc(biasDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    biasGroup.emplace_back(bias);
    auto xList = TensorListDesc({xGroup[0], xGroup[1]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case10)
{
    vector<vector<int64_t>> xDims = {{1280, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{256}, {256}};
    vector<vector<int64_t>> scaleDims = {{256}, {256}};
    vector<vector<int64_t>> outDims = {{1280, 256}};
    vector<int64_t> groupList = {256, 1024};
    int64_t splitItemOptional = 3;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> scaleGroup;
    std::vector<TensorDesc> outGroup;

    auto x = TensorDesc(xDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    xGroup.emplace_back(x);
    auto out = TensorDesc(outDims[0], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);
    outGroup.emplace_back(out);
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto weight = TensorDesc(weightDims[i], ACL_INT8, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto scale = TensorDesc(scaleDims[i], ACL_UINT64, ACL_FORMAT_ND).ValueRange(0.01, 0.04);
        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        scaleGroup.emplace_back(scale);
    }
    auto xList = TensorListDesc({xGroup[0]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto scaleList = TensorListDesc({scaleGroup[0], scaleGroup[1]});
    auto outList = TensorListDesc({outGroup[0]});
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 0, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(getWorkspaceResult, ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Ascend910B2Case11)
{
    vector<vector<int64_t>> xDims = {{1280, 256}};
    vector<vector<int64_t>> weightDims = {{256, 256}, {256, 256}};
    vector<vector<int64_t>> biasDims = {{256}, {256}};
    vector<vector<int64_t>> outDims = {{256, 256}, {1024, 256}};
    vector<int64_t> groupList = {1280, 256};
    int64_t splitItemOptional = 1;

    std::vector<TensorDesc> xGroup;
    std::vector<TensorDesc> weightGroup;
    std::vector<TensorDesc> biasGroup;
    std::vector<TensorDesc> outGroup;

    auto x = TensorDesc(xDims[0], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    xGroup.emplace_back(x);
    for (size_t i = 0; i < weightDims.size(); i++) {
        auto weight = TensorDesc(weightDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto bias = TensorDesc(biasDims[i], ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
        auto out = TensorDesc(outDims[i], ACL_FLOAT16, ACL_FORMAT_ND).Precision(0.00001, 0.00001);

        weightGroup.emplace_back(weight);
        biasGroup.emplace_back(bias);
        outGroup.emplace_back(out);
    }
    auto xList = TensorListDesc({xGroup[0]});
    auto weightList = TensorListDesc({weightGroup[0], weightGroup[1]});
    auto biasList = TensorListDesc({biasGroup[0], biasGroup[1]});
    auto outList = TensorListDesc({outGroup[0], outGroup[1]});
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());

    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, splitItemOptional, "test_group", "sum", 8, 1),
                        OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// splitItem=0 合法：groupList 必须为 nullptr；bias 可空以走 CreateEmptyTensor
TEST_F(L2GroupedMatMulAllReduceTest, ValidSplit0NoBias)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x1 = TensorDesc({512, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto w1 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y1 = TensorDesc({512, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto xList = TensorListDesc({x0, x1});
    auto weightList = TensorListDesc({w0, w1});
    auto outList = TensorListDesc({y0, y1});
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, nullptr, nullptr, 0, "test_group", "sum", 0, 1), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2GroupedMatMulAllReduceTest, ValidSplit0WithBias)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto x1 = TensorDesc({512, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto w1 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto b0 = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto b1 = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y1 = TensorDesc({512, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto xList = TensorListDesc({x0, x1});
    auto weightList = TensorListDesc({w0, w1});
    auto biasList = TensorListDesc({b0, b1});
    auto outList = TensorListDesc({y0, y1});
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, nullptr, 0, "test_group", "sum", 0, 1), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2GroupedMatMulAllReduceTest, ValidSplit1GroupList)
{
    auto x0 = TensorDesc({768, 256}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto w1 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto b0 = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto b1 = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y1 = TensorDesc({512, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> groupList = {256, 768};
    auto xList = TensorListDesc({x0});
    auto weightList = TensorListDesc({w0, w1});
    auto biasList = TensorListDesc({b0, b1});
    auto outList = TensorListDesc({y0, y1});
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(xList, weightList, biasList, array, 1, "test_group", "sum", 0, 1), OUTPUT(outList));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(L2GroupedMatMulAllReduceTest, Split0XGroupSizeMismatch)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w1 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y1 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0, w1}), nullptr, nullptr, 0, "test_group", "sum", 0, 1),
                  OUTPUT(TensorListDesc({y0, y1})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Split0GroupListNotNull)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> groupList = {256};
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, array, 0, "test_group", "sum", 0, 1),
                        OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Split1XGroupSizeNotOne)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto x1 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> groupList = {256};
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0, x1}), TensorListDesc({w0}), nullptr, array, 1, "test_group", "sum", 0, 1),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Split1NullGroupList)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 1, "test_group", "sum", 0, 1),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, InvalidSplitItem)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 9, "test_group", "sum", 0, 1),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, BiasGroupSizeMismatch)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w1 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto b0 = TensorDesc({128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y1 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(TensorListDesc({x0, x0}), TensorListDesc({w0, w1}), TensorListDesc({b0}), nullptr, 0,
                              "test_group", "sum", 0, 1),
                        OUTPUT(TensorListDesc({y0, y1})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, BiasNot1D)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto b0 = TensorDesc({128, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnGroupedMatMulAllReduce,
        INPUT(TensorListDesc({x0}), TensorListDesc({w0}), TensorListDesc({b0}), nullptr, 0, "test_group", "sum", 0, 1),
        OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, XDimOutOfRange)
{
    auto x0 = TensorDesc({2, 2, 2, 2, 2, 2, 256}, ACL_FLOAT16, ACL_FORMAT_ND); // 7D > max 6
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({2, 2, 2, 2, 2, 2, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 0, "test_group", "sum", 0, 1),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, WeightNot2D)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({1, 256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 0, "test_group", "sum", 0, 1),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, YGroupSizeMismatchSplit0)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w1 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnGroupedMatMulAllReduce,
        INPUT(TensorListDesc({x0, x0}), TensorListDesc({w0, w1}), nullptr, nullptr, 0, "test_group", "sum", 0, 1),
        OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, YGroupSizeMismatchSplit2)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y1 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 2, "test_group", "sum", 0, 1),
                  OUTPUT(TensorListDesc({y0, y1})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, DtypeMismatchXWeight)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 0, "test_group", "sum", 0, 1),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, InvalidStreamMode)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 0, "test_group", "sum", 0, 0),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, InvalidReduceOp)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 0, "test_group", "max", 0, 1),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, InvalidCommTurn)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 0, "test_group", "sum", 2, 1),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, ExecuteInner)
{
    char workspace[64] = {};
    aclnnStatus ret = aclnnGroupedMatMulAllReduce(workspace, sizeof(workspace), nullptr, nullptr);
    EXPECT_TRUE(ret == ACLNN_SUCCESS || ret == ACLNN_ERR_INNER || ret == ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, KDimMismatch)
{
    auto x0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 64}, ACL_FLOAT16, ACL_FORMAT_ND); // K 256 != 128
    auto y0 = TensorDesc({256, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 0, "test_group", "sum", 0, 1),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Split0XYDimsMismatch)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128, 1}, ACL_FLOAT16, ACL_FORMAT_ND); // 3D y vs 2D x
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 0, "test_group", "sum", 0, 1),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, Split1XNot2D)
{
    auto x0 = TensorDesc({2, 256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    vector<int64_t> groupList = {256};
    aclIntArray* array = aclCreateIntArray(groupList.data(), groupList.size());
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, array, 1, "test_group", "sum", 0, 1),
                        OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, YDtypeMismatch)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_BF16, ACL_FORMAT_ND);
    auto ut =
        OP_API_UT(aclnnGroupedMatMulAllReduce,
                  INPUT(TensorListDesc({x0}), TensorListDesc({w0}), nullptr, nullptr, 0, "test_group", "sum", 0, 1),
                  OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, BiasDtypeMismatch)
{
    auto x0 = TensorDesc({256, 256}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto w0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto b0 = TensorDesc({128}, ACL_FLOAT, ACL_FORMAT_ND);
    auto y0 = TensorDesc({256, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(
        aclnnGroupedMatMulAllReduce,
        INPUT(TensorListDesc({x0}), TensorListDesc({w0}), TensorListDesc({b0}), nullptr, 0, "test_group", "sum", 0, 1),
        OUTPUT(TensorListDesc({y0})));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(L2GroupedMatMulAllReduceTest, WeightGroupTooLarge)
{
    std::vector<TensorDesc> xs, ws, ys;
    for (int i = 0; i < 65; ++i) {
        xs.emplace_back(TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND));
        ws.emplace_back(TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND));
        ys.emplace_back(TensorDesc({16, 16}, ACL_FLOAT16, ACL_FORMAT_ND));
    }
    auto ut = OP_API_UT(aclnnGroupedMatMulAllReduce,
                        INPUT(TensorListDesc(xs), TensorListDesc(ws), nullptr, nullptr, 0, "test_group", "sum", 0, 1),
                        OUTPUT(TensorListDesc(ys)));
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

} // namespace
