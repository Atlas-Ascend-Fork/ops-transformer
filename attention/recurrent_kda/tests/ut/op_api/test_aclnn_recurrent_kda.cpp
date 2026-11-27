/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_recurrent_kda.h"
#include "opdev/op_log.h"

namespace {
class RecurrentKdaNullptrTest : public testing::TestWithParam<size_t> {};

TEST_P(RecurrentKdaNullptrTest, RequiredTensorReturnsParamNullptr)
{
    // Every other tensor is a non-null descriptor. No tensor is dereferenced before CheckNotNull returns.
    const std::array<int64_t, 4> shape = {1, 1, 1, 128};
    const std::array<int64_t, 4> strides = {128, 128, 128, 1};
    auto* tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                   shape.size(), nullptr);
    ASSERT_NE(tensor, nullptr);
    std::array<aclTensor*, 8> tensors;
    tensors.fill(tensor);
    tensors[GetParam()] = nullptr;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    const auto ret = aclnnRecurrentKdaGetWorkspaceSize(tensors[0], tensors[1], tensors[2], tensors[3], tensors[4],
                                                       tensors[5], nullptr, nullptr, nullptr, nullptr, nullptr, "BSND",
                                                       1.0, true, false, false, false, false, false, false, -5.0, false,
                                                       tensors[6], tensors[7], &workspaceSize, &executor);
    EXPECT_EQ(ret, ACLNN_ERR_PARAM_NULLPTR);
    EXPECT_EQ(executor, nullptr);
    aclDestroyTensor(tensor);
}

INSTANTIATE_TEST_SUITE_P(RequiredTensors, RecurrentKdaNullptrTest, testing::Range<size_t>(0, 8));
} // namespace
