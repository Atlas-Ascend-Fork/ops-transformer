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
#include "gtest/gtest.h"

#include "../../../../op_host/op_api/aclnn_norm_rope_concat_grad.h"

#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"

using namespace std;
using namespace op;

class l2_norm_rope_concat_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "norm_rope_concat_grad_test SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "norm_rope_concat_grad_test TearDown" << endl;
    }
};

TEST_F(l2_norm_rope_concat_grad_test, Ascend910B_case_1)
{
    uint64_t batchSize = 2;
    uint64_t querySeq = 4;
    uint64_t keySeq = 8;
    uint64_t valueSeq = keySeq;
    uint64_t encoderQuerySeq = 16;
    uint64_t encoderKeySeq = 32;
    uint64_t encoderValueSeq = encoderKeySeq;
    uint64_t ropeSeq = 8;
    uint64_t headNum = 32;
    uint64_t headDim = 64;

    auto gradQueryOutput =
        TensorDesc({batchSize * headNum * (querySeq + encoderQuerySeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto gradKeyOutput =
        TensorDesc({batchSize * headNum * (keySeq + encoderKeySeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto gradValueOutput =
        TensorDesc({batchSize * headNum * (valueSeq + encoderValueSeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto query = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto key = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto encoderQuery =
        TensorDesc({batchSize * encoderQuerySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto encoderKey =
        TensorDesc({batchSize * encoderKeySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryMean = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryRstd = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyMean = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyRstd = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryMean =
        TensorDesc({batchSize * encoderQuerySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryRstd =
        TensorDesc({batchSize * encoderQuerySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyMean =
        TensorDesc({batchSize * encoderKeySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyRstd =
        TensorDesc({batchSize * encoderKeySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto ropeSin = TensorDesc({ropeSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto ropeCos = TensorDesc({ropeSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradQuery = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKey = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValue = TensorDesc({batchSize * valueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderQuery =
        TensorDesc({batchSize * encoderQuerySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderKey =
        TensorDesc({batchSize * encoderKeySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderValue =
        TensorDesc({batchSize * encoderValueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormQueryBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormKeyBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedQueryBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedKeyBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    int64_t normType = 2;
    int64_t normAddedType = 2;
    int64_t ropeType = 1;
    int64_t concatOrder = 0;

    auto ut = OP_API_UT(
        aclnnNormRopeConcatBackward,
        INPUT(nullptr, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey, normQueryWeight,
              normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd, normAddedQueryWeight,
              normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean, normAddedKeyRstd, ropeSin,
              ropeCos, normType, normAddedType, ropeType, concatOrder),
        OUTPUT(gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey, gradEncoderValue, gradNormQueryWeight,
               gradNormQueryBias, gradNormKeyWeight, gradNormKeyBias, gradNormAddedQueryWeight, gradNormAddedQueryBias,
               gradNormAddedKeyWeight, gradNormAddedKeyBias));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(l2_norm_rope_concat_grad_test, Ascend910B_case_2)
{
    uint64_t batchSize = 2;
    uint64_t querySeq = 4;
    uint64_t keySeq = 8;
    uint64_t valueSeq = keySeq;
    uint64_t encoderQuerySeq = 16;
    uint64_t encoderKeySeq = 32;
    uint64_t encoderValueSeq = encoderKeySeq;
    uint64_t ropeSeq = 8;
    uint64_t headNum = 32;
    uint64_t headDim = 64;

    auto gradQueryOutput =
        TensorDesc({batchSize * headNum * (querySeq + encoderQuerySeq) * headDim}, ACL_INT32, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto gradKeyOutput =
        TensorDesc({batchSize * headNum * (keySeq + encoderKeySeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto gradValueOutput =
        TensorDesc({batchSize * headNum * (valueSeq + encoderValueSeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto query = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto key = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto encoderQuery =
        TensorDesc({batchSize * encoderQuerySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto encoderKey =
        TensorDesc({batchSize * encoderKeySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryMean = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryRstd = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyMean = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyRstd = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryMean =
        TensorDesc({batchSize * encoderQuerySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryRstd =
        TensorDesc({batchSize * encoderQuerySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyMean =
        TensorDesc({batchSize * encoderKeySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyRstd =
        TensorDesc({batchSize * encoderKeySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto ropeSin = TensorDesc({ropeSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto ropeCos = TensorDesc({ropeSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradQuery = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKey = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValue = TensorDesc({batchSize * valueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderQuery =
        TensorDesc({batchSize * encoderQuerySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderKey =
        TensorDesc({batchSize * encoderKeySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderValue =
        TensorDesc({batchSize * encoderValueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormQueryBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormKeyBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedQueryBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedKeyBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    int64_t normType = 2;
    int64_t normAddedType = 2;
    int64_t ropeType = 1;
    int64_t concatOrder = 0;

    auto ut = OP_API_UT(
        aclnnNormRopeConcatBackward,
        INPUT(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey, normQueryWeight,
              normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd, normAddedQueryWeight,
              normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean, normAddedKeyRstd, ropeSin,
              ropeCos, normType, normAddedType, ropeType, concatOrder),
        OUTPUT(gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey, gradEncoderValue, gradNormQueryWeight,
               gradNormQueryBias, gradNormKeyWeight, gradNormKeyBias, gradNormAddedQueryWeight, gradNormAddedQueryBias,
               gradNormAddedKeyWeight, gradNormAddedKeyBias));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue#6539 组合1：输入全部给齐，仅8个可选输出gradNorm*Weight/Bias传nullptr（表示不输出），应正常返回
TEST_F(l2_norm_rope_concat_grad_test, Ascend910B_case_3_optional_output_nullptr)
{
    uint64_t batchSize = 2;
    uint64_t querySeq = 4;
    uint64_t keySeq = 8;
    uint64_t valueSeq = keySeq;
    uint64_t encoderQuerySeq = 16;
    uint64_t encoderKeySeq = 32;
    uint64_t encoderValueSeq = encoderKeySeq;
    uint64_t ropeSeq = 8;
    uint64_t headNum = 32;
    uint64_t headDim = 64;

    auto gradQueryOutput =
        TensorDesc({batchSize * headNum * (querySeq + encoderQuerySeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto gradKeyOutput =
        TensorDesc({batchSize * headNum * (keySeq + encoderKeySeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto gradValueOutput =
        TensorDesc({batchSize * headNum * (valueSeq + encoderValueSeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto query = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto key = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto encoderQuery =
        TensorDesc({batchSize * encoderQuerySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto encoderKey =
        TensorDesc({batchSize * encoderKeySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryMean = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryRstd = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyMean = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyRstd = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryMean =
        TensorDesc({batchSize * encoderQuerySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryRstd =
        TensorDesc({batchSize * encoderQuerySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyMean =
        TensorDesc({batchSize * encoderKeySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyRstd =
        TensorDesc({batchSize * encoderKeySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto ropeSin = TensorDesc({ropeSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto ropeCos = TensorDesc({ropeSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradQuery = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKey = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValue = TensorDesc({batchSize * valueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderQuery =
        TensorDesc({batchSize * encoderQuerySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderKey =
        TensorDesc({batchSize * encoderKeySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderValue =
        TensorDesc({batchSize * encoderValueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    int64_t normType = 2;
    int64_t normAddedType = 2;
    int64_t ropeType = 1;
    int64_t concatOrder = 0;

    auto ut = OP_API_UT(
        aclnnNormRopeConcatBackward,
        INPUT(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey, normQueryWeight,
              normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd, normAddedQueryWeight,
              normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean, normAddedKeyRstd, ropeSin,
              ropeCos, normType, normAddedType, ropeType, concatOrder),
        OUTPUT(gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey, gradEncoderValue,
               /*gradNormQueryWeight*/ nullptr, /*gradNormQueryBias*/ nullptr,
               /*gradNormKeyWeight*/ nullptr, /*gradNormKeyBias*/ nullptr,
               /*gradNormAddedQueryWeight*/ nullptr, /*gradNormAddedQueryBias*/ nullptr,
               /*gradNormAddedKeyWeight*/ nullptr, /*gradNormAddedKeyBias*/ nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// issue#6539 组合2：全部可选输入传nullptr（normType=normAddedType=ropeType=0），最小"只算concat梯度"调用，应正常返回
TEST_F(l2_norm_rope_concat_grad_test, Ascend910B_case_4_all_optional_input_nullptr)
{
    uint64_t batchSize = 2;
    uint64_t querySeq = 4;
    uint64_t keySeq = 8;
    uint64_t valueSeq = keySeq;
    uint64_t headNum = 32;
    uint64_t headDim = 64;

    auto gradQueryOutput =
        TensorDesc({batchSize * headNum * querySeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKeyOutput =
        TensorDesc({batchSize * headNum * keySeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValueOutput =
        TensorDesc({batchSize * headNum * valueSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto query = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto key = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradQuery = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKey = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValue = TensorDesc({batchSize * valueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    int64_t normType = 0;
    int64_t normAddedType = 0;
    int64_t ropeType = 0;
    int64_t concatOrder = 0;

    auto ut = OP_API_UT(
        aclnnNormRopeConcatBackward,
        INPUT(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key,
              /*encoderQuery*/ nullptr, /*encoderKey*/ nullptr,
              /*normQueryWeight*/ nullptr, /*normQueryMean*/ nullptr, /*normQueryRstd*/ nullptr,
              /*normKeyWeight*/ nullptr, /*normKeyMean*/ nullptr, /*normKeyRstd*/ nullptr,
              /*normAddedQueryWeight*/ nullptr, /*normAddedQueryMean*/ nullptr, /*normAddedQueryRstd*/ nullptr,
              /*normAddedKeyWeight*/ nullptr, /*normAddedKeyMean*/ nullptr, /*normAddedKeyRstd*/ nullptr,
              /*ropeSin*/ nullptr, /*ropeCos*/ nullptr, normType, normAddedType, ropeType, concatOrder),
        OUTPUT(gradQuery, gradKey, gradValue,
               /*gradEncoderQuery*/ nullptr, /*gradEncoderKey*/ nullptr, /*gradEncoderValue*/ nullptr,
               /*gradNormQueryWeight*/ nullptr, /*gradNormQueryBias*/ nullptr,
               /*gradNormKeyWeight*/ nullptr, /*gradNormKeyBias*/ nullptr,
               /*gradNormAddedQueryWeight*/ nullptr, /*gradNormAddedQueryBias*/ nullptr,
               /*gradNormAddedKeyWeight*/ nullptr, /*gradNormAddedKeyBias*/ nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// issue#6539 组合3：normType=1（层归一化、无仿射），按资料不传normQueryWeight/normKeyWeight，应正常返回
TEST_F(l2_norm_rope_concat_grad_test, Ascend910B_case_5_norm_without_affine_weight)
{
    uint64_t batchSize = 2;
    uint64_t querySeq = 4;
    uint64_t keySeq = 8;
    uint64_t valueSeq = keySeq;
    uint64_t headNum = 32;
    uint64_t headDim = 64;

    auto gradQueryOutput =
        TensorDesc({batchSize * headNum * querySeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKeyOutput =
        TensorDesc({batchSize * headNum * keySeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValueOutput =
        TensorDesc({batchSize * headNum * valueSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto query = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto key = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryMean = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryRstd = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyMean = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyRstd = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradQuery = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKey = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValue = TensorDesc({batchSize * valueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    int64_t normType = 1;
    int64_t normAddedType = 0;
    int64_t ropeType = 0;
    int64_t concatOrder = 0;

    auto ut = OP_API_UT(
        aclnnNormRopeConcatBackward,
        INPUT(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key,
              /*encoderQuery*/ nullptr, /*encoderKey*/ nullptr,
              /*normQueryWeight*/ nullptr, normQueryMean, normQueryRstd,
              /*normKeyWeight*/ nullptr, normKeyMean, normKeyRstd,
              /*normAddedQueryWeight*/ nullptr, /*normAddedQueryMean*/ nullptr, /*normAddedQueryRstd*/ nullptr,
              /*normAddedKeyWeight*/ nullptr, /*normAddedKeyMean*/ nullptr, /*normAddedKeyRstd*/ nullptr,
              /*ropeSin*/ nullptr, /*ropeCos*/ nullptr, normType, normAddedType, ropeType, concatOrder),
        OUTPUT(gradQuery, gradKey, gradValue,
               /*gradEncoderQuery*/ nullptr, /*gradEncoderKey*/ nullptr, /*gradEncoderValue*/ nullptr,
               /*gradNormQueryWeight*/ nullptr, /*gradNormQueryBias*/ nullptr,
               /*gradNormKeyWeight*/ nullptr, /*gradNormKeyBias*/ nullptr,
               /*gradNormAddedQueryWeight*/ nullptr, /*gradNormAddedQueryBias*/ nullptr,
               /*gradNormAddedKeyWeight*/ nullptr, /*gradNormAddedKeyBias*/ nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// issue#6539 组合4：encoderQuery/encoderKey参与计算但normAddedType=0，不传normAdded*归一化输入，应正常返回
TEST_F(l2_norm_rope_concat_grad_test, Ascend910B_case_6_encoder_without_norm_added)
{
    uint64_t batchSize = 2;
    uint64_t querySeq = 4;
    uint64_t keySeq = 8;
    uint64_t valueSeq = keySeq;
    uint64_t encoderQuerySeq = 16;
    uint64_t encoderKeySeq = 32;
    uint64_t encoderValueSeq = encoderKeySeq;
    uint64_t headNum = 32;
    uint64_t headDim = 64;

    auto gradQueryOutput =
        TensorDesc({batchSize * headNum * (querySeq + encoderQuerySeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto gradKeyOutput =
        TensorDesc({batchSize * headNum * (keySeq + encoderKeySeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto gradValueOutput =
        TensorDesc({batchSize * headNum * (valueSeq + encoderValueSeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto query = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto key = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto encoderQuery =
        TensorDesc({batchSize * encoderQuerySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto encoderKey =
        TensorDesc({batchSize * encoderKeySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradQuery = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKey = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValue = TensorDesc({batchSize * valueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderQuery =
        TensorDesc({batchSize * encoderQuerySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderKey =
        TensorDesc({batchSize * encoderKeySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderValue =
        TensorDesc({batchSize * encoderValueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    int64_t normType = 0;
    int64_t normAddedType = 0;
    int64_t ropeType = 0;
    int64_t concatOrder = 0;

    auto ut = OP_API_UT(
        aclnnNormRopeConcatBackward,
        INPUT(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey,
              /*normQueryWeight*/ nullptr, /*normQueryMean*/ nullptr, /*normQueryRstd*/ nullptr,
              /*normKeyWeight*/ nullptr, /*normKeyMean*/ nullptr, /*normKeyRstd*/ nullptr,
              /*normAddedQueryWeight*/ nullptr, /*normAddedQueryMean*/ nullptr, /*normAddedQueryRstd*/ nullptr,
              /*normAddedKeyWeight*/ nullptr, /*normAddedKeyMean*/ nullptr, /*normAddedKeyRstd*/ nullptr,
              /*ropeSin*/ nullptr, /*ropeCos*/ nullptr, normType, normAddedType, ropeType, concatOrder),
        OUTPUT(gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey, gradEncoderValue,
               /*gradNormQueryWeight*/ nullptr, /*gradNormQueryBias*/ nullptr,
               /*gradNormKeyWeight*/ nullptr, /*gradNormKeyBias*/ nullptr,
               /*gradNormAddedQueryWeight*/ nullptr, /*gradNormAddedQueryBias*/ nullptr,
               /*gradNormAddedKeyWeight*/ nullptr, /*gradNormAddedKeyBias*/ nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_SUCCESS);
}

// issue#6539 组合5：ropeType=1但只传ropeSin、ropeCos传nullptr，第一段接口应返回161002
TEST_F(l2_norm_rope_concat_grad_test, Ascend910B_case_7_rope_sin_cos_not_in_pair)
{
    uint64_t batchSize = 2;
    uint64_t querySeq = 4;
    uint64_t keySeq = 8;
    uint64_t valueSeq = keySeq;
    uint64_t encoderQuerySeq = 16;
    uint64_t encoderKeySeq = 32;
    uint64_t encoderValueSeq = encoderKeySeq;
    uint64_t ropeSeq = 8;
    uint64_t headNum = 32;
    uint64_t headDim = 64;

    auto gradQueryOutput =
        TensorDesc({batchSize * headNum * (querySeq + encoderQuerySeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto gradKeyOutput =
        TensorDesc({batchSize * headNum * (keySeq + encoderKeySeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto gradValueOutput =
        TensorDesc({batchSize * headNum * (valueSeq + encoderValueSeq) * headDim}, ACL_FLOAT, ACL_FORMAT_ND)
            .ValueRange(0, 20);
    auto query = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto key = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto encoderQuery =
        TensorDesc({batchSize * encoderQuerySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto encoderKey =
        TensorDesc({batchSize * encoderKeySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryMean = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryRstd = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyMean = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyRstd = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryMean =
        TensorDesc({batchSize * encoderQuerySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryRstd =
        TensorDesc({batchSize * encoderQuerySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyMean =
        TensorDesc({batchSize * encoderKeySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyRstd =
        TensorDesc({batchSize * encoderKeySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto ropeSin = TensorDesc({ropeSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradQuery = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKey = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValue = TensorDesc({batchSize * valueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderQuery =
        TensorDesc({batchSize * encoderQuerySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderKey =
        TensorDesc({batchSize * encoderKeySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradEncoderValue =
        TensorDesc({batchSize * encoderValueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormQueryBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormKeyBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedQueryBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedKeyBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    int64_t normType = 2;
    int64_t normAddedType = 2;
    int64_t ropeType = 1;
    int64_t concatOrder = 0;

    auto ut = OP_API_UT(
        aclnnNormRopeConcatBackward,
        INPUT(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key, encoderQuery, encoderKey, normQueryWeight,
              normQueryMean, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd, normAddedQueryWeight,
              normAddedQueryMean, normAddedQueryRstd, normAddedKeyWeight, normAddedKeyMean, normAddedKeyRstd, ropeSin,
              /*ropeCos*/ nullptr, normType, normAddedType, ropeType, concatOrder),
        OUTPUT(gradQuery, gradKey, gradValue, gradEncoderQuery, gradEncoderKey, gradEncoderValue, gradNormQueryWeight,
               gradNormQueryBias, gradNormKeyWeight, gradNormKeyBias, gradNormAddedQueryWeight, gradNormAddedQueryBias,
               gradNormAddedKeyWeight, gradNormAddedKeyBias));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// issue#6539 组合6：不传encoderQuery/encoderKey但normAddedType=2，第一段接口应返回161002（原为内部错误561103）
TEST_F(l2_norm_rope_concat_grad_test, Ascend910B_case_8_norm_added_without_encoder)
{
    uint64_t batchSize = 2;
    uint64_t querySeq = 4;
    uint64_t keySeq = 8;
    uint64_t valueSeq = keySeq;
    uint64_t encoderQuerySeq = 16;
    uint64_t encoderKeySeq = 32;
    uint64_t encoderValueSeq = encoderKeySeq;
    uint64_t ropeSeq = 8;
    uint64_t headNum = 32;
    uint64_t headDim = 64;

    auto gradQueryOutput =
        TensorDesc({batchSize * headNum * querySeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKeyOutput =
        TensorDesc({batchSize * headNum * keySeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValueOutput =
        TensorDesc({batchSize * headNum * valueSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto query = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto key = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryMean = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryRstd = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyMean = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyRstd = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryMean =
        TensorDesc({batchSize * encoderQuerySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedQueryRstd =
        TensorDesc({batchSize * encoderQuerySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyMean =
        TensorDesc({batchSize * encoderKeySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normAddedKeyRstd =
        TensorDesc({batchSize * encoderKeySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto ropeSin = TensorDesc({ropeSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto ropeCos = TensorDesc({ropeSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradQuery = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKey = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValue = TensorDesc({batchSize * valueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormQueryBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormKeyBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedQueryBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradNormAddedKeyBias = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    int64_t normType = 2;
    int64_t normAddedType = 2;
    int64_t ropeType = 1;
    int64_t concatOrder = 0;

    auto ut = OP_API_UT(
        aclnnNormRopeConcatBackward,
        INPUT(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key,
              /*encoderQuery*/ nullptr, /*encoderKey*/ nullptr, normQueryWeight, normQueryMean, normQueryRstd,
              normKeyWeight, normKeyMean, normKeyRstd, normAddedQueryWeight, normAddedQueryMean, normAddedQueryRstd,
              normAddedKeyWeight, normAddedKeyMean, normAddedKeyRstd, ropeSin, ropeCos, normType, normAddedType,
              ropeType, concatOrder),
        OUTPUT(gradQuery, gradKey, gradValue,
               /*gradEncoderQuery*/ nullptr, /*gradEncoderKey*/ nullptr, /*gradEncoderValue*/ nullptr,
               gradNormQueryWeight, gradNormQueryBias, gradNormKeyWeight, gradNormKeyBias, gradNormAddedQueryWeight,
               gradNormAddedQueryBias, gradNormAddedKeyWeight, gradNormAddedKeyBias));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}

// normType=2（带仿射LayerNorm）但不传normQueryMean，第一段接口应返回161002
TEST_F(l2_norm_rope_concat_grad_test, Ascend910B_case_9_norm_without_mean)
{
    uint64_t batchSize = 2;
    uint64_t querySeq = 4;
    uint64_t keySeq = 8;
    uint64_t valueSeq = keySeq;
    uint64_t headNum = 32;
    uint64_t headDim = 64;

    auto gradQueryOutput =
        TensorDesc({batchSize * headNum * querySeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKeyOutput =
        TensorDesc({batchSize * headNum * keySeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValueOutput =
        TensorDesc({batchSize * headNum * valueSeq * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto query = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto key = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normQueryRstd = TensorDesc({batchSize * querySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyWeight = TensorDesc({headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyMean = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto normKeyRstd = TensorDesc({batchSize * keySeq * headNum * 1}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradQuery = TensorDesc({batchSize * querySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradKey = TensorDesc({batchSize * keySeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);
    auto gradValue = TensorDesc({batchSize * valueSeq * headNum * headDim}, ACL_FLOAT, ACL_FORMAT_ND).ValueRange(0, 20);

    int64_t normType = 2;
    int64_t normAddedType = 0;
    int64_t ropeType = 0;
    int64_t concatOrder = 0;

    auto ut = OP_API_UT(
        aclnnNormRopeConcatBackward,
        INPUT(gradQueryOutput, gradKeyOutput, gradValueOutput, query, key,
              /*encoderQuery*/ nullptr, /*encoderKey*/ nullptr, normQueryWeight,
              /*normQueryMean*/ nullptr, normQueryRstd, normKeyWeight, normKeyMean, normKeyRstd,
              /*normAddedQueryWeight*/ nullptr, /*normAddedQueryMean*/ nullptr, /*normAddedQueryRstd*/ nullptr,
              /*normAddedKeyWeight*/ nullptr, /*normAddedKeyMean*/ nullptr, /*normAddedKeyRstd*/ nullptr,
              /*ropeSin*/ nullptr, /*ropeCos*/ nullptr, normType, normAddedType, ropeType, concatOrder),
        OUTPUT(gradQuery, gradKey, gradValue,
               /*gradEncoderQuery*/ nullptr, /*gradEncoderKey*/ nullptr, /*gradEncoderValue*/ nullptr,
               /*gradNormQueryWeight*/ nullptr, /*gradNormQueryBias*/ nullptr,
               /*gradNormKeyWeight*/ nullptr, /*gradNormKeyBias*/ nullptr,
               /*gradNormAddedQueryWeight*/ nullptr, /*gradNormAddedQueryBias*/ nullptr,
               /*gradNormAddedKeyWeight*/ nullptr, /*gradNormAddedKeyBias*/ nullptr));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_INVALID);
}
