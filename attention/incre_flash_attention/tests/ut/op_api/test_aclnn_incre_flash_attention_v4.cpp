/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_api/aclnn_incre_flash_attention_v4.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
using namespace std;
using namespace op;

class incre_flash_attention_v4_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        op::SetPlatformSocVersion(op::SocVersion::ASCEND910B);
    }

    static void TearDownTestCase() {}
};

// dav5102 reject
TEST_F(incre_flash_attention_v4_opapi_ut, incre_flash_attention_v4_aclnn_5102_unsupported)
{
    op::ScopedPlatformNpuArch guard(NpuArch::DAV_5102);
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 8, 1.0, inputLayout, 8, 0, 1, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_RUNTIME_ERROR);
    EXPECT_EQ(executor, nullptr);

    aclRet = aclnnIncreFlashAttentionV4(nullptr, 0, nullptr, nullptr);
    EXPECT_EQ(aclRet, ACLNN_ERR_RUNTIME_ERROR);
}

// pseShift dtype must be FLOAT16/BFLOAT16
TEST_F(incre_flash_attention_v4_opapi_ut, incre_flash_attention_v4_pse_shift_dtype_invalid)
{
    TensorDesc pseShiftDesc({1, 4, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor* pseShift = pseShiftDesc.ToAclTypeRawPtr();
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        nullptr, nullptr, nullptr, pseShift, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 8, 1.0, inputLayout, 8, 0, 1, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);

    Release(pseShift);
}

// pseShift dtype must match query dtype
TEST_F(incre_flash_attention_v4_opapi_ut, incre_flash_attention_v4_pse_shift_query_dtype_mismatch)
{
    TensorDesc queryDesc({1, 8, 1, 64}, ACL_BF16, ACL_FORMAT_ND);
    aclTensor* query = queryDesc.ToAclTypeRawPtr();
    TensorDesc pseShiftDesc({1, 8, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* pseShift = pseShiftDesc.ToAclTypeRawPtr();
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        query, nullptr, nullptr, pseShift, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 8, 1.0, inputLayout, 8, 0, 1, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);

    Release(query);
    Release(pseShift);
}

// attenMask dtype must be BOOL/INT8/UINT8
TEST_F(incre_flash_attention_v4_opapi_ut, incre_flash_attention_v4_atten_mask_dtype_invalid)
{
    TensorDesc attenMaskDesc({1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* attenMask = attenMaskDesc.ToAclTypeRawPtr();
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, attenMask, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 8, 1.0, inputLayout, 8, 0, 1, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);

    Release(attenMask);
}

// attentionOut dtype must be FLOAT16/BFLOAT16/INT8
TEST_F(incre_flash_attention_v4_opapi_ut, incre_flash_attention_v4_attention_out_dtype_invalid)
{
    TensorDesc attentionOutDesc({1, 4, 1, 64}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor* attentionOut = attentionOutDesc.ToAclTypeRawPtr();
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 8, 1.0, inputLayout, 8, 0, 1, attentionOut, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);

    Release(attentionOut);
}

// inputLayout must be BSH/BNSD/BSND
TEST_F(incre_flash_attention_v4_opapi_ut, incre_flash_attention_v4_input_layout_invalid)
{
    char inputLayout[] = "XYZ";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 8, 1.0, inputLayout, 8, 0, 1, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// innerPrecise must be 0/1
TEST_F(incre_flash_attention_v4_opapi_ut, incre_flash_attention_v4_inner_precise_invalid)
{
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 8, 1.0, inputLayout, 8, 0, 5, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// innerPrecise=2 (ATB) is not a valid value for the public aclnn interface
TEST_F(incre_flash_attention_v4_opapi_ut, incre_flash_attention_v4_inner_precise_atb_invalid)
{
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 8, 1.0, inputLayout, 8, 0, 2, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// numHeads must be positive
TEST_F(incre_flash_attention_v4_opapi_ut, incre_flash_attention_v4_num_heads_invalid)
{
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 0, 1.0, inputLayout, 8, 0, 1, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// numHeads must be divisible by numKeyValueHeads
TEST_F(incre_flash_attention_v4_opapi_ut, incre_flash_attention_v4_num_heads_not_divisible)
{
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 8, 1.0, inputLayout, 3, 0, 1, nullptr, &workspaceSize, &executor);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
    EXPECT_EQ(executor, nullptr);
}

// valid params should pass op-api check and delegate to inner (reject only by inner for null query)
TEST_F(incre_flash_attention_v4_opapi_ut, incre_flash_attention_v4_valid_params)
{
    TensorDesc pseShiftDesc({1, 4, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* pseShift = pseShiftDesc.ToAclTypeRawPtr();
    TensorDesc attenMaskDesc({1, 1}, ACL_INT8, ACL_FORMAT_ND);
    aclTensor* attenMask = attenMaskDesc.ToAclTypeRawPtr();
    TensorDesc attentionOutDesc({1, 4, 1, 64}, ACL_FLOAT16, ACL_FORMAT_ND);
    aclTensor* attentionOut = attentionOutDesc.ToAclTypeRawPtr();
    char inputLayout[] = "BNSD";
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus aclRet = aclnnIncreFlashAttentionV4GetWorkspaceSize(
        nullptr, nullptr, nullptr, pseShift, attenMask, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, 8, 1.0, inputLayout, 8, 0, 1, attentionOut, &workspaceSize, &executor);
    EXPECT_NE(aclRet, ACLNN_ERR_PARAM_INVALID);

    Release(pseShift);
    Release(attenMask);
    Release(attentionOut);
}
