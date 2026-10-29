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
 * \file test_aclnn_nsa_selected_attention_infer.cpp
 * \brief aclnnNsaSelectedAttentionInfer TopkIndicesShapeCheck / BSH layout UT.
 */

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_nsa_selected_attention_infer.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

namespace {
constexpr int64_t kBatch = 1;
constexpr int64_t kQSeq = 1;
constexpr int64_t kNumHeads = 1;
constexpr int64_t kNumKvHeads = 1;
constexpr int64_t kHeadDim = 192;
constexpr int64_t kHeadDimV = 128;
constexpr int64_t kSelectBlockSize = 128;
constexpr int64_t kSelectBlockCount = 2;
constexpr int64_t kPageBlockSize = 128;
constexpr int64_t kMaxBlockNumPerBatch = 2;
constexpr double kScale = 1.0;

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
} // namespace

class NsaSelectedAttentionInferOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "NsaSelectedAttentionInferOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "NsaSelectedAttentionInferOpapiUt TearDown" << endl;
    }
};

// BSH + 4D topk：QKV 3 维，topkIndices 4 维 [B,S,numKeyValueHeads,selectBlockCount]
TEST_F(NsaSelectedAttentionInferOpapiUt, Bsh_Topk4d_Success)
{
    auto query = MakeTensor({kBatch, kQSeq, kNumHeads * kHeadDim}, ACL_FLOAT16);
    auto key = MakeTensor({kMaxBlockNumPerBatch, kPageBlockSize, kNumKvHeads * kHeadDim}, ACL_FLOAT16);
    auto value = MakeTensor({kMaxBlockNumPerBatch, kPageBlockSize, kNumKvHeads * kHeadDimV}, ACL_FLOAT16);
    auto topkIndices = MakeTensor({kBatch, kQSeq, kNumKvHeads, kSelectBlockCount}, ACL_INT32);
    auto blockTable = MakeTensor({kBatch, kMaxBlockNumPerBatch}, ACL_INT32);
    auto output = MakeTensor({kBatch, kQSeq, kNumHeads * kHeadDimV}, ACL_FLOAT16);

    int64_t actQ[1] = {kQSeq};
    int64_t actKv[1] = {kPageBlockSize * kMaxBlockNumPerBatch};
    AclIntArrayPtr actSeqQ(aclCreateIntArray(actQ, 1), DestroyAclIntArray);
    AclIntArrayPtr actSeqKv(aclCreateIntArray(actKv, 1), DestroyAclIntArray);

    char layout[] = "BSH";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnNsaSelectedAttentionInferGetWorkspaceSize(
        query.get(), key.get(), value.get(), topkIndices.get(), /*attenMaskOptional=*/nullptr, blockTable.get(),
        actSeqQ.get(), actSeqKv.get(), layout, kNumHeads, kNumKvHeads, kSelectBlockSize, kSelectBlockCount,
        kPageBlockSize, kScale, /*sparseMode=*/0, output.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// BSH + 3D topk：TopkIndicesShapeCheck 应返回 ACLNN_ERR_PARAM_INVALID (161002)
TEST_F(NsaSelectedAttentionInferOpapiUt, Bsh_Topk3d_ParamInvalid)
{
    auto query = MakeTensor({kBatch, kQSeq, kNumHeads * kHeadDim}, ACL_FLOAT16);
    auto key = MakeTensor({kMaxBlockNumPerBatch, kPageBlockSize, kNumKvHeads * kHeadDim}, ACL_FLOAT16);
    auto value = MakeTensor({kMaxBlockNumPerBatch, kPageBlockSize, kNumKvHeads * kHeadDimV}, ACL_FLOAT16);
    // 错误：BSH 下误传 3 维 topk（issue 6615）
    auto topkIndices = MakeTensor({kBatch, kQSeq, kSelectBlockCount}, ACL_INT32);
    auto blockTable = MakeTensor({kBatch, kMaxBlockNumPerBatch}, ACL_INT32);
    auto output = MakeTensor({kBatch, kQSeq, kNumHeads * kHeadDimV}, ACL_FLOAT16);

    int64_t actQ[1] = {kQSeq};
    int64_t actKv[1] = {kPageBlockSize * kMaxBlockNumPerBatch};
    AclIntArrayPtr actSeqQ(aclCreateIntArray(actQ, 1), DestroyAclIntArray);
    AclIntArrayPtr actSeqKv(aclCreateIntArray(actKv, 1), DestroyAclIntArray);

    char layout[] = "BSH";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnNsaSelectedAttentionInferGetWorkspaceSize(
        query.get(), key.get(), value.get(), topkIndices.get(), /*attenMaskOptional=*/nullptr, blockTable.get(),
        actSeqQ.get(), actSeqKv.get(), layout, kNumHeads, kNumKvHeads, kSelectBlockSize, kSelectBlockCount,
        kPageBlockSize, kScale, /*sparseMode=*/0, output.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// BSND + 3D topk：同样应被 TopkIndicesShapeCheck 拦截
TEST_F(NsaSelectedAttentionInferOpapiUt, Bsnd_Topk3d_ParamInvalid)
{
    auto query = MakeTensor({kBatch, kQSeq, kNumHeads, kHeadDim}, ACL_FLOAT16);
    auto key = MakeTensor({kMaxBlockNumPerBatch, kPageBlockSize, kNumKvHeads, kHeadDim}, ACL_FLOAT16);
    auto value = MakeTensor({kMaxBlockNumPerBatch, kPageBlockSize, kNumKvHeads, kHeadDimV}, ACL_FLOAT16);
    auto topkIndices = MakeTensor({kBatch, kQSeq, kSelectBlockCount}, ACL_INT32);
    auto blockTable = MakeTensor({kBatch, kMaxBlockNumPerBatch}, ACL_INT32);
    auto output = MakeTensor({kBatch, kQSeq, kNumHeads, kHeadDimV}, ACL_FLOAT16);

    int64_t actQ[1] = {kQSeq};
    int64_t actKv[1] = {kPageBlockSize * kMaxBlockNumPerBatch};
    AclIntArrayPtr actSeqQ(aclCreateIntArray(actQ, 1), DestroyAclIntArray);
    AclIntArrayPtr actSeqKv(aclCreateIntArray(actKv, 1), DestroyAclIntArray);

    char layout[] = "BSND";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnNsaSelectedAttentionInferGetWorkspaceSize(
        query.get(), key.get(), value.get(), topkIndices.get(), /*attenMaskOptional=*/nullptr, blockTable.get(),
        actSeqQ.get(), actSeqKv.get(), layout, kNumHeads, kNumKvHeads, kSelectBlockSize, kSelectBlockCount,
        kPageBlockSize, kScale, /*sparseMode=*/0, output.get(), &workspaceSize, &executor);

    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}
