/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <array>
#include <float.h>
#include "gtest/gtest.h"
#include "../../../../op_api/aclnn_grouped_matmul_add.h"
#include "../../../../op_api/aclnn_grouped_matmul_add_v2.h"

#include "opdev/platform.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"

using namespace std;

class l2_grouped_matmul_add_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "l2_grouped_matmul_add_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "l2_grouped_matmul_add_test TearDown" << endl;
    }
};

TEST_F(l2_grouped_matmul_add_test, Ascend910B2_grouped_matmul_add_fp16)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorDesc({K, M}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto groupList = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto yRef = TensorDesc({M, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t groupType = 2;
    bool transposeX = true;
    bool transposeWeight = false;

    auto ut = OP_API_UT(aclnnGroupedMatmulAdd,
                        INPUT(x, weight, groupList, yRef, transposeX, transposeWeight, groupType), OUTPUT());
    uint64_t workspaceSize = 0;
    aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
}

TEST_F(l2_grouped_matmul_add_test, Ascend910B2_grouped_matmul_add_fp16_v2)
{
    size_t M = 345;
    size_t K = 1280;
    size_t N = 567;
    size_t E = 2;
    auto x = TensorDesc({K, M}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto weight = TensorDesc({K, N}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-10, 10);
    auto groupList = TensorDesc({E}, ACL_INT64, ACL_FORMAT_ND);
    auto yRef = TensorDesc({M, N}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(-10, 10);
    int64_t groupType = 2;
    bool transposeX = true;
    bool transposeWeight = false;

    uint64_t workspaceSize = 0;
    int groupListTypeMode = 2;
    for (int i = 0; i < groupListTypeMode; i++) {
        // groupListType = 0 or 1.
        auto ut = OP_API_UT(aclnnGroupedMatmulAddV2,
                            INPUT(x, weight, groupList, yRef, transposeX, transposeWeight, groupType, i), OUTPUT());

        aclnnStatus getWorkspaceResult = ut.TestGetWorkspaceSize(&workspaceSize);
    }
}

TEST_F(l2_grouped_matmul_add_test, Ascend910B2_expanded_path_rejects_invalid_metadata)
{
    constexpr aclnnStatus PARAM_INVALID = 161002;
    struct TestCase {
        const char* name;
        std::vector<int64_t> x;
        std::vector<int64_t> weight;
        std::vector<int64_t> group;
        std::vector<int64_t> y;
        std::vector<int64_t> yStride;
        int64_t yOffset;
        std::vector<int64_t> yStorage;
        aclDataType yType;
        bool transpose;
    };
    const int64_t maxI32 = 2147483647;
    const TestCase cases[] = {
        {"unaligned N", {17, 32}, {32, 33}, {2}, {2, 17, 33}, {}, 0, {}, ACL_FLOAT16, false},
        {"non-contiguous y", {17, 32}, {32, 32}, {2}, {2, 17, 32}, {544, 1, 17}, 0, {}, ACL_FLOAT16, false},
        {"unaligned y offset", {17, 32}, {32, 32}, {2}, {2, 17, 32}, {544, 32, 1}, 1, {1089}, ACL_FLOAT16, false},
        {"x last dimension limit", {17, 65536}, {65536, 16}, {1}, {1, 17, 16}, {}, 0, {}, ACL_FLOAT, false},
        {"weight last dimension limit", {17, 32}, {32, 65536}, {1}, {1, 17, 65536}, {}, 0, {}, ACL_FLOAT, false},
        {"M exceeds int32", {maxI32 + 1, 16}, {16, 16}, {1}, {1, maxI32 + 1, 16}, {}, 0, {}, ACL_FLOAT, false},
        {"groups exceed int32", {1, 16}, {16, 16}, {maxI32 + 1}, {maxI32 + 1, 1, 16}, {}, 0, {}, ACL_FLOAT, false},
        {"output product exceeds int64", {maxI32, 16}, {16, 65520}, {maxI32}, {1}, {}, 0, {}, ACL_FLOAT, false},
    };
    for (const auto& test : cases) {
        SCOPED_TRACE(test.name);
        auto x = TensorDesc(test.x, ACL_FLOAT16, ACL_FORMAT_ND);
        auto weight = TensorDesc(test.weight, ACL_FLOAT16, ACL_FORMAT_ND);
        auto groupList = TensorDesc(test.group, ACL_INT64, ACL_FORMAT_ND);
        auto y = TensorDesc(test.y, test.yType, ACL_FORMAT_ND, test.yStride, test.yOffset, test.yStorage);
        auto ut = OP_API_UT(aclnnGroupedMatmulAddV2,
                            INPUT(x, weight, groupList, y, test.transpose, false, int64_t{2}, int64_t{1}), OUTPUT());
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), PARAM_INVALID);
    }
}

TEST_F(l2_grouped_matmul_add_test, Ascend950_nontransposed_x_fp32_output)
{
    const auto previousSoc = op::GetCurrentPlatformInfo().GetSocVersion();
    op::SetPlatformSocVersion(op::SocVersion::ASCEND950);
    auto x = TensorDesc({3, 8}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto weight = TensorDesc({8, 4}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto groupList = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND).Value(std::vector<int64_t>{3, 8});
    auto y = TensorDesc({2, 3, 4}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t workspaceSize = 0;
    auto ut = OP_API_UT(aclnnGroupedMatmulAdd, INPUT(x, weight, groupList, y, false, false, int64_t{2}), OUTPUT());
    EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), ACLNN_SUCCESS);
    op::SetPlatformSocVersion(previousSoc);
}
