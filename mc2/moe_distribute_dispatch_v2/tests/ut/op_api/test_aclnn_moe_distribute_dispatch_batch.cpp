/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_api/aclnn_moe_distribute_dispatch_v2.h"
#include "../../../op_api/aclnn_moe_distribute_dispatch_v3.h"
#include "../../../op_api/aclnn_moe_distribute_dispatch_v4.h"
#include "../../../op_api/moe_distribute_dispatch_v2_base.h"

#include <vector>
#include "gtest/gtest.h"
#include "op_api_ut_common/tensor_desc.h"

namespace MoeDistributeDispatchBatch {
namespace {
constexpr uint64_t WORKSPACE_SENTINEL = 123;

class L2AclnnMoeDistributeDispatchBatchTest : public testing::TestWithParam<int> {
protected:
    void ExpectRejected(const aclTensor* x, const aclTensor* expertIds, aclnnStatus expectedStatus)
    {
        // Output shapes are supplied, so validation must not depend on InferShape being called.
        auto expandX = TensorDesc({16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
        auto dynamicScales = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
        auto assistInfo = TensorDesc({16 * 128}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
        auto expertTokensNums = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
        auto epRecvCounts = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
        auto tpRecvCounts = TensorDesc({1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
        auto expandScales = TensorDesc({16}, ACL_FLOAT, ACL_FORMAT_ND).ToAclType();
        uint64_t workspaceSize = WORKSPACE_SENTINEL;
        aclOpExecutor* executor = nullptr;
        aclnnStatus status = ACLNN_SUCCESS;
        if (GetParam() == 2) {
            status = aclnnMoeDistributeDispatchV2GetWorkspaceSize(
                x, expertIds, nullptr, nullptr, nullptr, "batch_test_ep", 2, 0, 2, "", 1, 0, 0, 0, 0, 0, 0, 1,
                "fullmesh", expandX.get(), dynamicScales.get(), assistInfo.get(), expertTokensNums.get(),
                epRecvCounts.get(), tpRecvCounts.get(), expandScales.get(), &workspaceSize, &executor);
        } else if (GetParam() == 3) {
            status = aclnnMoeDistributeDispatchV3GetWorkspaceSize(
                x, expertIds, nullptr, nullptr, nullptr, nullptr, "batch_test_ep", 2, 0, 2, "", 1, 0, 0, 0, 0, 0, 0, 1,
                "fullmesh", 0, 0, 0, expandX.get(), dynamicScales.get(), assistInfo.get(), expertTokensNums.get(),
                epRecvCounts.get(), tpRecvCounts.get(), expandScales.get(), &workspaceSize, &executor);
        } else {
            status = aclnnMoeDistributeDispatchV4GetWorkspaceSize(
                x, expertIds, nullptr, nullptr, nullptr, nullptr, nullptr, "batch_test_ep", 2, 0, 2, "", 1, 0, 0, 0, 0,
                0, 0, 1, "fullmesh", 0, 0, 0, expandX.get(), dynamicScales.get(), assistInfo.get(),
                expertTokensNums.get(), epRecvCounts.get(), tpRecvCounts.get(), expandScales.get(), &workspaceSize,
                &executor);
        }
        EXPECT_EQ(status, expectedStatus);
        EXPECT_EQ(workspaceSize, WORKSPACE_SENTINEL);
        EXPECT_EQ(executor, nullptr);
    }
};

TEST_P(L2AclnnMoeDistributeDispatchBatchTest, RejectsFewerExpertIdRows)
{
    auto x = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto expertIds = TensorDesc({7, 1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    ExpectRejected(x.get(), expertIds.get(), ACLNN_ERR_PARAM_INVALID);
}

TEST_P(L2AclnnMoeDistributeDispatchBatchTest, RejectsMoreExpertIdRows)
{
    auto x = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto expertIds = TensorDesc({9, 1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    ExpectRejected(x.get(), expertIds.get(), ACLNN_ERR_PARAM_INVALID);
}

TEST_P(L2AclnnMoeDistributeDispatchBatchTest, RejectsNullXBeforeReadingShape)
{
    auto expertIds = TensorDesc({8, 1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    ExpectRejected(nullptr, expertIds.get(), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_P(L2AclnnMoeDistributeDispatchBatchTest, RejectsNullExpertIdsBeforeReadingShape)
{
    auto x = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    ExpectRejected(x.get(), nullptr, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_P(L2AclnnMoeDistributeDispatchBatchTest, RejectsInvalidXRankBeforeReadingBatch)
{
    auto expertIds = TensorDesc({8, 1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    for (const std::vector<int64_t>& shape : std::vector<std::vector<int64_t>>{{}, {8}, {8, 1, 512}}) {
        SCOPED_TRACE(testing::Message() << "x rank: " << shape.size());
        auto x = TensorDesc(shape, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
        ExpectRejected(x.get(), expertIds.get(), ACLNN_ERR_PARAM_INVALID);
    }
}

TEST_P(L2AclnnMoeDistributeDispatchBatchTest, RejectsInvalidExpertIdsRankBeforeReadingBatch)
{
    auto x = TensorDesc({8, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    for (const std::vector<int64_t>& shape : std::vector<std::vector<int64_t>>{{}, {8}, {8, 1, 1}}) {
        SCOPED_TRACE(testing::Message() << "expertIds rank: " << shape.size());
        auto expertIds = TensorDesc(shape, ACL_INT32, ACL_FORMAT_ND).ToAclType();
        ExpectRejected(x.get(), expertIds.get(), ACLNN_ERR_PARAM_INVALID);
    }
}

INSTANTIATE_TEST_SUITE_P(PublicApiVersions, L2AclnnMoeDistributeDispatchBatchTest, testing::Values(2, 3, 4));

TEST(L2AclnnMoeDistributeDispatchBatchParamsTest, AcceptsMatchingBatchWithDifferentHiddenAndTopKDimensions)
{
    auto expandX = TensorDesc({16, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto assistInfo = TensorDesc({16 * 128}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    auto expertTokensNums = TensorDesc({1}, ACL_INT64, ACL_FORMAT_ND).ToAclType();
    auto epRecvCounts = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
    for (int64_t batch : {1, 8, 16}) {
        SCOPED_TRACE(testing::Message() << "batch: " << batch);
        auto x = TensorDesc({batch, 512}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
        auto expertIds = TensorDesc({batch, 1}, ACL_INT32, ACL_FORMAT_ND).ToAclType();
        EXPECT_EQ(DispatchCheckParams(x.get(), expertIds.get(), nullptr, "batch_test_ep", nullptr, 0, expandX.get(),
                                      nullptr, assistInfo.get(), expertTokensNums.get(), epRecvCounts.get(), nullptr),
                  ACLNN_SUCCESS);
    }
}
} // namespace
} // namespace MoeDistributeDispatchBatch
