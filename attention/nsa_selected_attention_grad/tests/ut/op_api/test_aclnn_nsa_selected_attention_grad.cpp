/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_nsa_selected_attention_grad.h"
#include "op_api_ut_common/tensor_desc.h"
#include "opdev/op_errno.h"
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
} // namespace

class NsaSelectedAttentionGradOpapiUt : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
        cout << "NsaSelectedAttentionGradOpapiUt SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "NsaSelectedAttentionGradOpapiUt TearDown" << endl;
    }
};

// A1: positive - typical TND, fp16, no mask. Workspace query should succeed.
TEST_F(NsaSelectedAttentionGradOpapiUt, A1_tnd_fp16_no_mask)
{
    constexpr int64_t b = 1;
    constexpr int64_t s1 = 2;
    constexpr int64_t s2 = 1024;
    constexpr int64_t t1 = b * s1;
    constexpr int64_t t2 = b * s2;
    constexpr int64_t n1 = 4;
    constexpr int64_t n2 = 1;
    constexpr int64_t d = 192;
    constexpr int64_t d2 = 128;
    constexpr int64_t selectedBlockCount = 16;
    constexpr int64_t selectedBlockSize = 64;

    auto query = MakeTensor({t1, n1, d}, ACL_FLOAT16);
    auto key = MakeTensor({t2, n2, d}, ACL_FLOAT16);
    auto value = MakeTensor({t2, n2, d2}, ACL_FLOAT16);
    auto attentionOut = MakeTensor({t1, n1, d2}, ACL_FLOAT16);
    auto attentionOutGrad = MakeTensor({t1, n1, d2}, ACL_FLOAT16);
    auto softmaxMax = MakeTensor({t1, n1, 8}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({t1, n1, 8}, ACL_FLOAT);
    auto topkIndices = MakeTensor({t1, n2, selectedBlockCount}, ACL_INT32);

    auto dqOut = MakeTensor({}, ACL_FLOAT16);
    auto dkOut = MakeTensor({}, ACL_FLOAT16);
    auto dvOut = MakeTensor({}, ACL_FLOAT16);

    int64_t actQ[1] = {t1};
    int64_t actKv[1] = {t2};
    AclIntArrayPtr actSeqQ(aclCreateIntArray(actQ, 1), DestroyAclIntArray);
    AclIntArrayPtr actSeqKv(aclCreateIntArray(actKv, 1), DestroyAclIntArray);

    char inputLayout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnNsaSelectedAttentionGradGetWorkspaceSize(
        query.get(), key.get(), value.get(), attentionOut.get(), attentionOutGrad.get(), softmaxMax.get(),
        softmaxSum.get(), topkIndices.get(), actSeqQ.get(), actSeqKv.get(), /*attenMaskOptional=*/nullptr,
        /*scaleValue=*/0.088388, selectedBlockSize, selectedBlockCount,
        /*headNum=*/n1, inputLayout, /*sparseMode=*/0, dqOut.get(), dkOut.get(), dvOut.get(), &workspaceSize,
        &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

// A2: positive - bf16 + sparse_mode=2 with atten_mask.
TEST_F(NsaSelectedAttentionGradOpapiUt, A2_tnd_bf16_with_mask)
{
    constexpr int64_t b = 1;
    constexpr int64_t t1 = 2;
    constexpr int64_t t2 = 2048;
    constexpr int64_t n1 = 4;
    constexpr int64_t n2 = 1;
    constexpr int64_t d = 192;
    constexpr int64_t d2 = 128;
    constexpr int64_t selectedBlockCount = 16;
    constexpr int64_t selectedBlockSize = 64;

    auto query = MakeTensor({t1, n1, d}, ACL_BF16);
    auto key = MakeTensor({t2, n2, d}, ACL_BF16);
    auto value = MakeTensor({t2, n2, d2}, ACL_BF16);
    auto attentionOut = MakeTensor({t1, n1, d2}, ACL_BF16);
    auto attentionOutGrad = MakeTensor({t1, n1, d2}, ACL_BF16);
    auto softmaxMax = MakeTensor({t1, n1, 8}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({t1, n1, 8}, ACL_FLOAT);
    auto topkIndices = MakeTensor({t1, n2, selectedBlockCount}, ACL_INT32);
    auto attenMask = MakeTensor({selectedBlockSize, selectedBlockSize}, ACL_BOOL);

    auto dqOut = MakeTensor({}, ACL_BF16);
    auto dkOut = MakeTensor({}, ACL_BF16);
    auto dvOut = MakeTensor({}, ACL_BF16);

    int64_t actQ[1] = {t1};
    int64_t actKv[1] = {t2};
    AclIntArrayPtr actSeqQ(aclCreateIntArray(actQ, 1), DestroyAclIntArray);
    AclIntArrayPtr actSeqKv(aclCreateIntArray(actKv, 1), DestroyAclIntArray);

    char inputLayout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnNsaSelectedAttentionGradGetWorkspaceSize(
        query.get(), key.get(), value.get(), attentionOut.get(), attentionOutGrad.get(), softmaxMax.get(),
        softmaxSum.get(), topkIndices.get(), actSeqQ.get(), actSeqKv.get(), attenMask.get(),
        /*scaleValue=*/0.088388, selectedBlockSize, selectedBlockCount,
        /*headNum=*/n1, inputLayout, /*sparseMode=*/2, dqOut.get(), dkOut.get(), dvOut.get(), &workspaceSize,
        &executor);

    EXPECT_EQ(aclRet, ACL_SUCCESS);
    EXPECT_NE(executor, nullptr);
}

// E1: negative - null query should be rejected by CheckParams.
TEST_F(NsaSelectedAttentionGradOpapiUt, E1_null_query)
{
    auto key = MakeTensor({1024, 1, 192}, ACL_FLOAT16);
    auto value = MakeTensor({1024, 1, 128}, ACL_FLOAT16);
    auto attentionOut = MakeTensor({2, 4, 128}, ACL_FLOAT16);
    auto attentionOutGrad = MakeTensor({2, 4, 128}, ACL_FLOAT16);
    auto softmaxMax = MakeTensor({2, 4, 8}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({2, 4, 8}, ACL_FLOAT);
    auto topkIndices = MakeTensor({2, 1, 16}, ACL_INT32);
    auto dqOut = MakeTensor({2, 4, 192}, ACL_FLOAT16);
    auto dkOut = MakeTensor({1024, 1, 192}, ACL_FLOAT16);
    auto dvOut = MakeTensor({1024, 1, 128}, ACL_FLOAT16);

    char inputLayout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnNsaSelectedAttentionGradGetWorkspaceSize(
        /*query=*/nullptr, key.get(), value.get(), attentionOut.get(), attentionOutGrad.get(), softmaxMax.get(),
        softmaxSum.get(), topkIndices.get(),
        /*actualSeqQLenOptional=*/nullptr, /*actualSeqKvLenOptional=*/nullptr,
        /*attenMaskOptional=*/nullptr,
        /*scaleValue=*/0.088388, /*selectedBlockSize=*/64, /*selectedBlockCount=*/16,
        /*headNum=*/4, inputLayout, /*sparseMode=*/0, dqOut.get(), dkOut.get(), dvOut.get(), &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// E2: negative - missing topkIndices.
TEST_F(NsaSelectedAttentionGradOpapiUt, E2_null_topk_indices)
{
    auto query = MakeTensor({2, 4, 192}, ACL_FLOAT16);
    auto key = MakeTensor({1024, 1, 192}, ACL_FLOAT16);
    auto value = MakeTensor({1024, 1, 128}, ACL_FLOAT16);
    auto attentionOut = MakeTensor({2, 4, 128}, ACL_FLOAT16);
    auto attentionOutGrad = MakeTensor({2, 4, 128}, ACL_FLOAT16);
    auto softmaxMax = MakeTensor({2, 4, 8}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({2, 4, 8}, ACL_FLOAT);
    auto dqOut = MakeTensor({2, 4, 192}, ACL_FLOAT16);
    auto dkOut = MakeTensor({1024, 1, 192}, ACL_FLOAT16);
    auto dvOut = MakeTensor({1024, 1, 128}, ACL_FLOAT16);

    char inputLayout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnNsaSelectedAttentionGradGetWorkspaceSize(
        query.get(), key.get(), value.get(), attentionOut.get(), attentionOutGrad.get(), softmaxMax.get(),
        softmaxSum.get(), /*topkIndices=*/nullptr,
        /*actualSeqQLenOptional=*/nullptr, /*actualSeqKvLenOptional=*/nullptr,
        /*attenMaskOptional=*/nullptr,
        /*scaleValue=*/0.088388, /*selectedBlockSize=*/64, /*selectedBlockCount=*/16,
        /*headNum=*/4, inputLayout, /*sparseMode=*/0, dqOut.get(), dkOut.get(), dvOut.get(), &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

// E3: negative - missing dq output.
TEST_F(NsaSelectedAttentionGradOpapiUt, E3_null_dq_output)
{
    auto query = MakeTensor({2, 4, 192}, ACL_FLOAT16);
    auto key = MakeTensor({1024, 1, 192}, ACL_FLOAT16);
    auto value = MakeTensor({1024, 1, 128}, ACL_FLOAT16);
    auto attentionOut = MakeTensor({2, 4, 128}, ACL_FLOAT16);
    auto attentionOutGrad = MakeTensor({2, 4, 128}, ACL_FLOAT16);
    auto softmaxMax = MakeTensor({2, 4, 8}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({2, 4, 8}, ACL_FLOAT);
    auto topkIndices = MakeTensor({2, 1, 16}, ACL_INT32);
    auto dkOut = MakeTensor({1024, 1, 192}, ACL_FLOAT16);
    auto dvOut = MakeTensor({1024, 1, 128}, ACL_FLOAT16);

    char inputLayout[] = "TND";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnNsaSelectedAttentionGradGetWorkspaceSize(
        query.get(), key.get(), value.get(), attentionOut.get(), attentionOutGrad.get(), softmaxMax.get(),
        softmaxSum.get(), topkIndices.get(),
        /*actualSeqQLenOptional=*/nullptr, /*actualSeqKvLenOptional=*/nullptr,
        /*attenMaskOptional=*/nullptr,
        /*scaleValue=*/0.088388, /*selectedBlockSize=*/64, /*selectedBlockCount=*/16,
        /*headNum=*/4, inputLayout, /*sparseMode=*/0,
        /*dqOut=*/nullptr, dkOut.get(), dvOut.get(), &workspaceSize, &executor);

    EXPECT_NE(aclRet, ACL_SUCCESS);
    EXPECT_EQ(executor, nullptr);
}

namespace {
// 基准用例：TND / fp16 / N1=4 / N2=1 / D=192 / D2=128 / selectedBlockSize=64 / selectedBlockCount=16
struct NsaGradIllegalCase {
    int64_t t1 = 2;
    int64_t t2 = 1024;
    int64_t n1 = 4;
    int64_t n2 = 1;
    int64_t d = 192;
    int64_t d2 = 128;
    int64_t selectedBlockSize = 64;
    int64_t selectedBlockCount = 16;
    int64_t topkLastDim = 16;
    int64_t softmaxTailDim = 8;
    int64_t headNum = 4;
    int64_t sparseMode = 0;
    const char* inputLayout = "TND";
    bool withMask = false;
    aclDataType attenMaskDtype = ACL_BOOL;
};

aclnnStatus RunIllegalCase(const NsaGradIllegalCase& c)
{
    auto query = MakeTensor({c.t1, c.n1, c.d}, ACL_FLOAT16);
    auto key = MakeTensor({c.t2, c.n2, c.d}, ACL_FLOAT16);
    auto value = MakeTensor({c.t2, c.n2, c.d2}, ACL_FLOAT16);
    auto attentionOut = MakeTensor({c.t1, c.n1, c.d2}, ACL_FLOAT16);
    auto attentionOutGrad = MakeTensor({c.t1, c.n1, c.d2}, ACL_FLOAT16);
    auto softmaxMax = MakeTensor({c.t1, c.n1, c.softmaxTailDim}, ACL_FLOAT);
    auto softmaxSum = MakeTensor({c.t1, c.n1, c.softmaxTailDim}, ACL_FLOAT);
    auto topkIndices = MakeTensor({c.t1, c.n2, c.topkLastDim}, ACL_INT32);
    AclTensorPtr attenMask(nullptr, DestroyAclTensor);
    if (c.withMask) {
        attenMask = MakeTensor({c.selectedBlockSize, c.selectedBlockSize}, c.attenMaskDtype);
    }
    auto dqOut = MakeTensor({c.t1, c.n1, c.d}, ACL_FLOAT16);
    auto dkOut = MakeTensor({c.t2, c.n2, c.d}, ACL_FLOAT16);
    auto dvOut = MakeTensor({c.t2, c.n2, c.d2}, ACL_FLOAT16);

    int64_t actQ[1] = {c.t1};
    int64_t actKv[1] = {c.t2};
    AclIntArrayPtr actSeqQ(aclCreateIntArray(actQ, 1), DestroyAclIntArray);
    AclIntArrayPtr actSeqKv(aclCreateIntArray(actKv, 1), DestroyAclIntArray);

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    return aclnnNsaSelectedAttentionGradGetWorkspaceSize(
        query.get(), key.get(), value.get(), attentionOut.get(), attentionOutGrad.get(), softmaxMax.get(),
        softmaxSum.get(), topkIndices.get(), actSeqQ.get(), actSeqKv.get(), attenMask.get(), /*scaleValue=*/0.088388,
        c.selectedBlockSize, c.selectedBlockCount, c.headNum, const_cast<char*>(c.inputLayout), c.sparseMode,
        dqOut.get(), dkOut.get(), dvOut.get(), &workspaceSize, &executor);
}
} // namespace

// E4: inputLayout 非 TND 时返回 161002。
TEST_F(NsaSelectedAttentionGradOpapiUt, E4_invalid_input_layout)
{
    NsaGradIllegalCase c;
    c.inputLayout = "BSND";
    EXPECT_EQ(RunIllegalCase(c), ACLNN_ERR_PARAM_INVALID);
}

// E5: sparseMode 不在 {0, 2} 时返回 161002。
TEST_F(NsaSelectedAttentionGradOpapiUt, E5_invalid_sparse_mode)
{
    NsaGradIllegalCase c;
    c.sparseMode = 1;
    EXPECT_EQ(RunIllegalCase(c), ACLNN_ERR_PARAM_INVALID);
}

// E6: selectedBlockSize 非 16 的整数倍时返回 161002。
TEST_F(NsaSelectedAttentionGradOpapiUt, E6_invalid_selected_block_size)
{
    NsaGradIllegalCase c;
    c.selectedBlockSize = 24;
    EXPECT_EQ(RunIllegalCase(c), ACLNN_ERR_PARAM_INVALID);
}

// E7: selectedBlockCount 为 0 时返回 161002。
TEST_F(NsaSelectedAttentionGradOpapiUt, E7_invalid_selected_block_count)
{
    NsaGradIllegalCase c;
    c.selectedBlockCount = 0;
    EXPECT_EQ(RunIllegalCase(c), ACLNN_ERR_PARAM_INVALID);
}

// E8: topkIndices 末维与 selectedBlockCount 不一致时返回 161002。
TEST_F(NsaSelectedAttentionGradOpapiUt, E8_topk_indices_last_dim_mismatch)
{
    NsaGradIllegalCase c;
    c.topkLastDim = 2;
    EXPECT_EQ(RunIllegalCase(c), ACLNN_ERR_PARAM_INVALID);
}

// E9: D 既不是 128 也不是 192 时返回 161002。
TEST_F(NsaSelectedAttentionGradOpapiUt, E9_invalid_head_dim)
{
    NsaGradIllegalCase c;
    c.d = 64;
    c.d2 = 64;
    EXPECT_EQ(RunIllegalCase(c), ACLNN_ERR_PARAM_INVALID);
}

// E10: N1 不是 N2 的整数倍时返回 161002。
TEST_F(NsaSelectedAttentionGradOpapiUt, E10_n1_not_multiple_of_n2)
{
    NsaGradIllegalCase c;
    c.n1 = 5;
    c.n2 = 2;
    c.headNum = 5;
    EXPECT_EQ(RunIllegalCase(c), ACLNN_ERR_PARAM_INVALID);
}

// E11: headNum 与 query 的 N1 不一致时返回 161002。
TEST_F(NsaSelectedAttentionGradOpapiUt, E11_head_num_mismatch)
{
    NsaGradIllegalCase c;
    c.headNum = 8;
    EXPECT_EQ(RunIllegalCase(c), ACLNN_ERR_PARAM_INVALID);
}

// E12: attenMaskOptional dtype 不是 BOOL/UINT8 时返回 161002。
TEST_F(NsaSelectedAttentionGradOpapiUt, E12_invalid_atten_mask_dtype)
{
    NsaGradIllegalCase c;
    c.withMask = true;
    c.attenMaskDtype = ACL_FLOAT;
    EXPECT_EQ(RunIllegalCase(c), ACLNN_ERR_PARAM_INVALID);
}

// E13: softmaxMax 形状不是 [T1, N1, 8] 时返回 161002。
TEST_F(NsaSelectedAttentionGradOpapiUt, E13_invalid_softmax_max_shape)
{
    NsaGradIllegalCase c;
    c.softmaxTailDim = 4;
    EXPECT_EQ(RunIllegalCase(c), ACLNN_ERR_PARAM_INVALID);
}
