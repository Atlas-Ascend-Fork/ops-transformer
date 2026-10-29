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
 * \file test_aclnn_moe_ep_dispatch.cpp
 * \brief MoeEpDispatch aclnn UT
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "aclnn/aclnn_base.h"

extern "C" {
aclnnStatus aclnnMoeEpDispatchGetWorkspaceSize(
    const aclTensor* context, const aclTensor* x, const aclTensor* topkIdx, const aclTensor* topkWeights,
    const aclTensor* scales, const aclTensor* cachedDstSlotIdx, const aclTensor* cachedRouteCount,
    const aclTensor* cachedRouteDstScaleout, const aclTensor* cachedRouteScaleoutSlot, int64_t epWorldSize,
    int64_t epRankId, int64_t numExperts, int64_t numMaxTokensPerRank, int64_t cclBufferSize, int64_t expertAlignment,
    bool doCpuSync, int64_t hostPinnedCounterAddr, int64_t topoType, int64_t rankNumPerServer,
    aclTensor* numRecvTokensPerRank, aclTensor* numRecvTokensPerExpert, aclTensor* dstBufferSlotIdx,
    aclTensor* routeCount, aclTensor* routeDstScaleout, aclTensor* routeScaleoutSlot, uint64_t* workspaceSize,
    aclOpExecutor** executor);
aclnnStatus aclnnMoeEpDispatch(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
}

using namespace op;

class AclnnMoeEpDispatchTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
    }
};

TEST_F(AclnnMoeEpDispatchTest, nullptr_context)
{
    auto x = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto dstBufferSlotIdx = TensorDesc({2, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto routeCount = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto routeDstScaleout = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto routeScaleoutSlot = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(nullptr, x, topkIdx, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 2, 0, 4, 8,
                              12000, 1, true, 0, 0, 1, numRecvTokensPerRank, numRecvTokensPerExpert, dstBufferSlotIdx,
                              routeCount, routeDstScaleout, routeScaleoutSlot),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMoeEpDispatchTest, nullptr_x)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto dstBufferSlotIdx = TensorDesc({2, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto routeCount = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto routeDstScaleout = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto routeScaleoutSlot = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(context, nullptr, topkIdx, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 2, 0, 4,
                              8, 12000, 1, true, 0, 0, 1, numRecvTokensPerRank, numRecvTokensPerExpert,
                              dstBufferSlotIdx, routeCount, routeDstScaleout, routeScaleoutSlot),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMoeEpDispatchTest, nullptr_routeScaleoutSlot)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto dstBufferSlotIdx = TensorDesc({2, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto routeCount = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto routeDstScaleout = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(context, x, topkIdx, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 2, 0, 4, 8,
                              12000, 1, true, 0, 0, 1, numRecvTokensPerRank, numRecvTokensPerExpert, dstBufferSlotIdx,
                              routeCount, routeDstScaleout, nullptr),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnMoeEpDispatchTest, invalid_epWorldSize)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto dstBufferSlotIdx = TensorDesc({2, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto routeCount = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto routeDstScaleout = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto routeScaleoutSlot = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(context, x, topkIdx, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 1, 0, 4, 8,
                              12000, 1, true, 0, 0, 1, numRecvTokensPerRank, numRecvTokensPerExpert, dstBufferSlotIdx,
                              routeCount, routeDstScaleout, routeScaleoutSlot),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMoeEpDispatchTest, invalid_cached_with_cpu_sync)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto cachedDstSlotIdx = TensorDesc({2, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto dstBufferSlotIdx = TensorDesc({2, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto routeCount = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto routeDstScaleout = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto routeScaleoutSlot = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(context, x, topkIdx, nullptr, nullptr, cachedDstSlotIdx, nullptr, nullptr, nullptr, 2, 0,
                              4, 8, 12000, 1, true, 0, 0, 1, numRecvTokensPerRank, numRecvTokensPerExpert,
                              dstBufferSlotIdx, routeCount, routeDstScaleout, routeScaleoutSlot),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMoeEpDispatchTest, invalid_partial_cached_route)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto cachedRouteCount = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto dstBufferSlotIdx = TensorDesc({2, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto routeCount = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto routeDstScaleout = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto routeScaleoutSlot = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(context, x, topkIdx, nullptr, nullptr, nullptr, cachedRouteCount, nullptr, nullptr, 2, 0,
                              4, 8, 12000, 1, true, 0, 0, 1, numRecvTokensPerRank, numRecvTokensPerExpert,
                              dstBufferSlotIdx, routeCount, routeDstScaleout, routeScaleoutSlot),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMoeEpDispatchTest, invalid_hybrid_cached_missing_route)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto cachedDstSlotIdx = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto dstBufferSlotIdx = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto routeCount = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto routeDstScaleout = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto routeScaleoutSlot = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(context, x, topkIdx, nullptr, nullptr, cachedDstSlotIdx, nullptr, nullptr, nullptr, 8, 0,
                              16, 8, 12000, 1, false, 0, 1, 4, numRecvTokensPerRank, numRecvTokensPerExpert,
                              dstBufferSlotIdx, routeCount, routeDstScaleout, routeScaleoutSlot),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMoeEpDispatchTest, invalid_rank_per_server)
{
    auto context = TensorDesc({2052}, ACL_INT32, ACL_FORMAT_ND);
    auto x = TensorDesc({8, 128}, ACL_FLOAT16, ACL_FORMAT_ND);
    auto topkIdx = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerRank = TensorDesc({2}, ACL_INT32, ACL_FORMAT_ND);
    auto numRecvTokensPerExpert = TensorDesc({2}, ACL_INT64, ACL_FORMAT_ND);
    auto dstBufferSlotIdx = TensorDesc({2, 9}, ACL_INT32, ACL_FORMAT_ND);
    auto routeCount = TensorDesc({8}, ACL_INT32, ACL_FORMAT_ND);
    auto routeDstScaleout = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto routeScaleoutSlot = TensorDesc({8, 2}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnMoeEpDispatch,
                        INPUT(context, x, topkIdx, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 2, 0, 4, 8,
                              12000, 1, true, 0, 0, 3, numRecvTokensPerRank, numRecvTokensPerExpert, dstBufferSlotIdx,
                              routeCount, routeDstScaleout, routeScaleoutSlot),
                        OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_INVALID);
}

TEST_F(AclnnMoeEpDispatchTest, execute_entry)
{
    EXPECT_EQ(aclnnMoeEpDispatch(nullptr, 0, nullptr, nullptr), ACLNN_SUCCESS);
}
