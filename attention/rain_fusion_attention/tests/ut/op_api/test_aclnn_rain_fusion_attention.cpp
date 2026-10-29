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
 * \file test_aclnn_rain_fusion_attention.cpp
 * \brief aclnnRainFusionAttention CheckEmbeddingSize / blockShapeY UT (issue 6677).
 */

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_rain_fusion_attention.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
void DestroyAclTensor(aclTensor* tensor)
{
    Release(tensor);
}

void DestroyAclIntArray(aclIntArray* arr)
{
    aclDestroyIntArray(arr);
}

using AclTensorPtr = unique_ptr<aclTensor, decltype(&DestroyAclTensor)>;
using AclIntArrayPtr = unique_ptr<aclIntArray, decltype(&DestroyAclIntArray)>;

AclTensorPtr MakeTensor(const vector<int64_t>& shape, aclDataType dtype, aclFormat fmt = ACL_FORMAT_ND)
{
    return AclTensorPtr(TensorDesc(shape, dtype, fmt).ToAclTypeRawPtr(), DestroyAclTensor);
}

aclnnStatus CallGetWorkspace(int64_t headDim, int64_t blockShapeY, aclOpExecutor** executor)
{
    constexpr int64_t t = 4;
    constexpr int64_t kvT = 4;
    constexpr int64_t n = 8;
    constexpr int64_t kvN = 2;
    constexpr int64_t maxKvBlockNum = 4;

    auto query = MakeTensor({t, n, headDim}, ACL_FLOAT16);
    auto key = MakeTensor({kvT, kvN, headDim}, ACL_FLOAT16);
    auto value = MakeTensor({kvT, kvN, headDim}, ACL_FLOAT16);
    auto selectIdx = MakeTensor({t, n, maxKvBlockNum}, ACL_INT32);
    auto selectNumIdx = MakeTensor({t, n}, ACL_INT32);
    auto attentionOut = MakeTensor({t, n, headDim}, ACL_FLOAT16);

    int64_t blockShapeData[2] = {128, blockShapeY};
    AclIntArrayPtr blockShape(aclCreateIntArray(blockShapeData, 2), DestroyAclIntArray);
    int64_t qSeq[2] = {2, 2};
    int64_t kvSeq[2] = {2, 2};
    AclIntArrayPtr actSeqQ(aclCreateIntArray(qSeq, 2), DestroyAclIntArray);
    AclIntArrayPtr actSeqKv(aclCreateIntArray(kvSeq, 2), DestroyAclIntArray);

    char qLayout[] = "TND";
    char kvLayout[] = "TND";
    uint64_t workspaceSize = 0;
    *executor = nullptr;

    return aclnnRainFusionAttentionGetWorkspaceSize(
        query.get(), key.get(), value.get(), selectIdx.get(), selectNumIdx.get(), blockShape.get(),
        /*attenMaskOptional=*/nullptr, actSeqQ.get(), actSeqKv.get(), /*blockTableOptional=*/nullptr, qLayout, kvLayout,
        kvN, /*maskType=*/0, /*scaleValue=*/0.1, /*innerPrecise=*/1, /*blockSize=*/128, attentionOut.get(),
        /*softmaxLseOptional=*/nullptr, &workspaceSize, executor);
}
} // namespace

class RainFusionAttentionOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "RainFusionAttentionOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "RainFusionAttentionOpapiUt TearDown" << endl;
    }
};

// issue 6677：headDim=32 → ACLNN_ERR_PARAM_INVALID
TEST_F(RainFusionAttentionOpapiUt, EmbeddingSize_32_ParamInvalid)
{
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = CallGetWorkspace(/*headDim=*/32, /*blockShapeY=*/128, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// issue 6677：headDim=96 → ACLNN_ERR_PARAM_INVALID
TEST_F(RainFusionAttentionOpapiUt, EmbeddingSize_96_ParamInvalid)
{
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = CallGetWorkspace(/*headDim=*/96, /*blockShapeY=*/128, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// issue 6677：blockShapeY=64 不能被 128 整除 → ACLNN_ERR_PARAM_INVALID
TEST_F(RainFusionAttentionOpapiUt, BlockShapeY_64_ParamInvalid)
{
    aclOpExecutor* executor = nullptr;
    aclnnStatus aclRet = CallGetWorkspace(/*headDim=*/128, /*blockShapeY=*/64, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}
