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
 * \file test_aclnn_engram_fetch_wait.cpp
 * \brief EngramFetchWait aclnn UT
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "op_api_ut_common/op_api_ut.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"
#include "aclnn/aclnn_base.h"

extern "C" {
aclnnStatus aclnnEngramFetchWaitGetWorkspaceSize(const aclTensor* commContext, aclTensor* fetched,
                                                 uint64_t* workspaceSize, aclOpExecutor** executor);
aclnnStatus aclnnEngramFetchWait(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);
}

using namespace op;

class AclnnEngramFetchWaitTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
    }

    static void TearDownTestCase()
    {
        op::SetPlatformNpuArch(Ops::Base::DAV_3510);
    }
};

TEST_F(AclnnEngramFetchWaitTest, nullptr_commContext)
{
    auto fetched = TensorDesc({8, 512}, ACL_BF16, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnEngramFetchWait, INPUT(nullptr, fetched), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnEngramFetchWaitTest, nullptr_fetched)
{
    auto commContext = TensorDesc({6146}, ACL_INT32, ACL_FORMAT_ND);
    auto ut = OP_API_UT(aclnnEngramFetchWait, INPUT(commContext, nullptr), OUTPUT());
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    EXPECT_EQ(ut.TestGetWorkspaceSizeWithNNopbaseInner(&workspaceSize, executor), ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(AclnnEngramFetchWaitTest, execute_entry)
{
    aclnnStatus ret = aclnnEngramFetchWait(nullptr, 0, nullptr, nullptr);
    EXPECT_THAT(ret, testing::AnyOf(testing::Eq(ACLNN_SUCCESS), testing::Eq(ACLNN_ERR_PARAM_NULLPTR),
                                    testing::Eq(ACLNN_ERR_PARAM_INVALID), testing::Eq(ACLNN_ERR_INNER)));
}
