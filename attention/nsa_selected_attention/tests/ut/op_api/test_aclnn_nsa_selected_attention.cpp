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
 * \file test_aclnn_nsa_selected_attention.cpp
 * \brief NsaSelectedAttention aclnn op_api UT.
 */

#include <vector>
#include <array>
#include <cstring>
#include "gtest/gtest.h"
#include "../../../op_host/op_api/aclnn_nsa_selected_attention.h"
#include "op_api_ut_common/tensor_desc.h"
#include "op_api_ut_common/scalar_desc.h"
#include "op_api_ut_common/array_desc.h"
#include "op_api_ut_common/op_api_ut.h"
#include "opdev/platform.h"
#include "opdev/op_errno.h"

using namespace std;
using namespace op;

namespace {
constexpr int64_t kT = 128;
constexpr int64_t kN1 = 4; // query head_num
constexpr int64_t kN2 = 2; // kv head_num (G = kN1 / kN2 = 2)
constexpr int64_t kQueryD = 128;
constexpr int64_t kValueD = 128;
constexpr int64_t kSelectedBlockSize = 64;
constexpr int64_t kSelectedBlockCount = 16;
} // namespace

class nsa_selected_attention_opapi_ut : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "nsa_selected_attention_opapi_ut SetUp" << endl;
    }

    static void TearDownTestCase()
    {
        cout << "nsa_selected_attention_opapi_ut TearDown" << endl;
    }
};

TEST_F(nsa_selected_attention_opapi_ut, nsa_selected_attention_aclnn_0)
{
    // Inputs: required tensors (TND, fp16)
    auto tensorQ = TensorDesc({kT, kN1, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({kT, kN2, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({kT, kN2, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorTopkIndices =
        TensorDesc({kT, kN2, kSelectedBlockCount}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});

    // CheckNsaParam treats actualSeqQLen/KvLen as required, so build a single-batch prefix-sum.
    auto actualSeqQLen = IntArrayDesc(vector<int64_t>{kT});
    auto actualSeqKvLen = IntArrayDesc(vector<int64_t>{kT});

    // Attributes
    const double scaleValue = 0.088388;
    const int64_t headNum = kN1;
    char layout[] = "TND";
    const int64_t sparseMode = 0;
    const int64_t selectedBlockSize = kSelectedBlockSize;
    const int64_t selectedBlockCount = kSelectedBlockCount;

    // Outputs: shape strictly follows infershape (T,N1,8) / (T,N1,8) / (T,N1,valueD)
    auto tensorSoftmaxMax = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxSum = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorAttentionOut = TensorDesc({kT, kN1, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaSelectedAttention,
                        INPUT(tensorQ, tensorK, tensorV, tensorTopkIndices,
                              nullptr, // attenMaskOptional
                              actualSeqQLen, actualSeqKvLen, scaleValue, headNum, layout, sparseMode, selectedBlockSize,
                              selectedBlockCount),
                        OUTPUT(tensorSoftmaxMax, tensorSoftmaxSum, tensorAttentionOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(nsa_selected_attention_opapi_ut, nsa_selected_attention_aclnn_bf16)
{
    auto tensorQ = TensorDesc({kT, kN1, kQueryD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({kT, kN2, kQueryD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({kT, kN2, kValueD}, ACL_BF16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorTopkIndices =
        TensorDesc({kT, kN2, kSelectedBlockCount}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});

    auto actualSeqQLen = IntArrayDesc(vector<int64_t>{kT});
    auto actualSeqKvLen = IntArrayDesc(vector<int64_t>{kT});

    const double scaleValue = 0.088388;
    const int64_t headNum = kN1;
    char layout[] = "TND";
    const int64_t sparseMode = 0;
    const int64_t selectedBlockSize = kSelectedBlockSize;
    const int64_t selectedBlockCount = kSelectedBlockCount;

    auto tensorSoftmaxMax = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxSum = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorAttentionOut = TensorDesc({kT, kN1, kValueD}, ACL_BF16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaSelectedAttention,
                        INPUT(tensorQ, tensorK, tensorV, tensorTopkIndices, nullptr, actualSeqQLen, actualSeqKvLen,
                              scaleValue, headNum, layout, sparseMode, selectedBlockSize, selectedBlockCount),
                        OUTPUT(tensorSoftmaxMax, tensorSoftmaxSum, tensorAttentionOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

TEST_F(nsa_selected_attention_opapi_ut, nsa_selected_attention_aclnn_varlen)
{
    // Multi-batch via actualSeqQLen / actualSeqKvLen (both prefix-summed).
    // Two equal-length segments of 64 tokens => total T = 128.
    auto tensorQ = TensorDesc({kT, kN1, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({kT, kN2, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({kT, kN2, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorTopkIndices =
        TensorDesc({kT, kN2, kSelectedBlockCount}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});

    auto actualSeqQLen = IntArrayDesc(vector<int64_t>{64, 128});
    auto actualSeqKvLen = IntArrayDesc(vector<int64_t>{64, 128});

    const double scaleValue = 0.088388;
    const int64_t headNum = kN1;
    char layout[] = "TND";
    const int64_t sparseMode = 0;
    const int64_t selectedBlockSize = kSelectedBlockSize;
    const int64_t selectedBlockCount = kSelectedBlockCount;

    auto tensorSoftmaxMax = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxSum = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorAttentionOut = TensorDesc({kT, kN1, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaSelectedAttention,
                        INPUT(tensorQ, tensorK, tensorV, tensorTopkIndices,
                              nullptr, // attenMaskOptional
                              actualSeqQLen, actualSeqKvLen, scaleValue, headNum, layout, sparseMode, selectedBlockSize,
                              selectedBlockCount),
                        OUTPUT(tensorSoftmaxMax, tensorSoftmaxSum, tensorAttentionOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACL_SUCCESS);
}

// Negative case: the current CheckNsaParam() rejects null actualSeqQLen / actualSeqKvLen
TEST_F(nsa_selected_attention_opapi_ut, nsa_selected_attention_aclnn_null_actual_seq_len_returns_error)
{
    auto tensorQ = TensorDesc({kT, kN1, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorK = TensorDesc({kT, kN2, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorV = TensorDesc({kT, kN2, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND).ValueRange(-1, 1);
    auto tensorTopkIndices =
        TensorDesc({kT, kN2, kSelectedBlockCount}, ACL_INT32, ACL_FORMAT_ND).Value(vector<int32_t>{0});

    const double scaleValue = 0.088388;
    const int64_t headNum = kN1;
    char layout[] = "TND";
    const int64_t sparseMode = 0;
    const int64_t selectedBlockSize = kSelectedBlockSize;
    const int64_t selectedBlockCount = kSelectedBlockCount;

    auto tensorSoftmaxMax = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorSoftmaxSum = TensorDesc({kT, kN1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
    auto tensorAttentionOut = TensorDesc({kT, kN1, kValueD}, ACL_FLOAT16, ACL_FORMAT_ND);

    auto ut = OP_API_UT(aclnnNsaSelectedAttention,
                        INPUT(tensorQ, tensorK, tensorV, tensorTopkIndices,
                              nullptr, // attenMaskOptional
                              nullptr, // actualSeqQLenOptional   (nullptr on purpose)
                              nullptr, // actualSeqKvLenOptional  (nullptr on purpose)
                              scaleValue, headNum, layout, sparseMode, selectedBlockSize, selectedBlockCount),
                        OUTPUT(tensorSoftmaxMax, tensorSoftmaxSum, tensorAttentionOut));

    uint64_t workspaceSize = 0;
    aclnnStatus aclRet = ut.TestGetWorkspaceSize(&workspaceSize);
    EXPECT_EQ(aclRet, ACLNN_ERR_PARAM_NULLPTR);
}

TEST_F(nsa_selected_attention_opapi_ut, required_null_parameters_return_param_nullptr)
{
    auto tensor = TensorDesc({kT, kN1, kQueryD}, ACL_FLOAT16, ACL_FORMAT_ND).ToAclType();
    auto seq = IntArrayDesc(vector<int64_t>{kT}).ToAclType();
    for (size_t i = 0; i < 12; ++i) {
        SCOPED_TRACE(i);
        array<const aclTensor*, 7> tensors;
        tensors.fill(tensor.get());
        if (i < tensors.size()) {
            tensors[i] = nullptr;
        }
        char layout[] = "TND";
        uint64_t workspaceSize = 0;
        aclOpExecutor* executor = nullptr;
        EXPECT_EQ(aclnnNsaSelectedAttentionGetWorkspaceSize(
                      tensors[0], tensors[1], tensors[2], tensors[3], nullptr, i == 7 ? nullptr : seq.get(),
                      i == 8 ? nullptr : seq.get(), 0.088388, kN1, i == 9 ? nullptr : layout, 0, kSelectedBlockSize,
                      kSelectedBlockCount, tensors[4], tensors[5], tensors[6], i == 10 ? nullptr : &workspaceSize,
                      i == 11 ? nullptr : &executor),
                  ACLNN_ERR_PARAM_NULLPTR);
    }
}

TEST_F(nsa_selected_attention_opapi_ut, rejects_invalid_parameters_before_empty_output_shortcut)
{
    struct Case {
        const char* name;
        aclDataType dtype;
        aclDataType indexType;
        int64_t mode;
        int64_t blockSize;
        int64_t blockCount;
        bool empty;
        const char* layout;
        aclnnStatus expected;
    };
    const Case cases[] = {
        {"issue base", ACL_FLOAT16, ACL_INT32, 2, 16, 1, false, "TND", ACLNN_SUCCESS},
        {"bf16", ACL_BF16, ACL_INT32, 0, 16, 1, false, "TND", ACLNN_SUCCESS},
        {"issue fp32", ACL_FLOAT, ACL_INT32, 2, 16, 1, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"issue int64 indices", ACL_FLOAT16, ACL_INT64, 2, 16, 1, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"issue sparse mode", ACL_FLOAT16, ACL_INT32, 5, 16, 1, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"issue zero count", ACL_FLOAT16, ACL_INT32, 2, 16, 0, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"issue layout", ACL_FLOAT16, ACL_INT32, 2, 16, 1, false, "BSND", ACLNN_ERR_PARAM_INVALID},
        {"issue empty layout", ACL_FLOAT16, ACL_INT32, 2, 16, 1, true, "BSND", ACLNN_ERR_PARAM_INVALID},
        {"valid empty output", ACL_FLOAT16, ACL_INT32, 2, 16, 1, true, "TND", ACLNN_SUCCESS},
        {"empty fp32", ACL_FLOAT, ACL_INT32, 2, 16, 1, true, "TND", ACLNN_ERR_PARAM_INVALID},
        {"empty int64 indices", ACL_FLOAT16, ACL_INT64, 2, 16, 1, true, "TND", ACLNN_ERR_PARAM_INVALID},
        {"empty sparse mode", ACL_FLOAT16, ACL_INT32, 5, 16, 1, true, "TND", ACLNN_ERR_PARAM_INVALID},
        {"empty zero count", ACL_FLOAT16, ACL_INT32, 2, 16, 0, true, "TND", ACLNN_ERR_PARAM_INVALID},
        {"zero block size", ACL_FLOAT16, ACL_INT32, 2, 0, 1, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"negative block size", ACL_FLOAT16, ACL_INT32, 2, -16, 1, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"unaligned block size", ACL_FLOAT16, ACL_INT32, 2, 17, 1, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"large block size", ACL_FLOAT16, ACL_INT32, 2, 144, 1, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"negative count", ACL_FLOAT16, ACL_INT32, 2, 16, -1, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"large count", ACL_FLOAT16, ACL_INT32, 2, 16, 129, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"selection equals 8192", ACL_FLOAT16, ACL_INT32, 2, 64, 128, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"selection above 8192", ACL_FLOAT16, ACL_INT32, 2, 128, 128, false, "TND", ACLNN_ERR_PARAM_INVALID},
        {"valid max block size", ACL_FLOAT16, ACL_INT32, 2, 128, 1, false, "TND", ACLNN_SUCCESS},
        {"valid max count", ACL_FLOAT16, ACL_INT32, 2, 16, 128, false, "TND", ACLNN_SUCCESS},
    };

    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        const int64_t outTokens = c.empty ? 0 : 64;
        auto q = TensorDesc({64, 1, 192}, c.dtype, ACL_FORMAT_ND);
        auto k = TensorDesc({4096, 1, 192}, c.dtype, ACL_FORMAT_ND);
        auto v = TensorDesc({4096, 1, 128}, c.dtype, ACL_FORMAT_ND);
        auto indices = TensorDesc({64, 1, c.blockCount > 0 ? c.blockCount : 1}, c.indexType, ACL_FORMAT_ND);
        auto lengthsQ = IntArrayDesc(vector<int64_t>{64});
        auto lengthsKv = IntArrayDesc(vector<int64_t>{4096});
        auto max = TensorDesc({outTokens, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
        auto sum = TensorDesc({outTokens, 1, 8}, ACL_FLOAT, ACL_FORMAT_ND);
        auto out = TensorDesc({outTokens, 1, 128}, c.dtype, ACL_FORMAT_ND);
        char layout[5] = {};
        strcpy(layout, c.layout);
        auto ut = OP_API_UT(aclnnNsaSelectedAttention,
                            INPUT(q, k, v, indices, nullptr, lengthsQ, lengthsKv, 0.0721687836, 1, layout, c.mode,
                                  c.blockSize, c.blockCount),
                            OUTPUT(max, sum, out));
        uint64_t workspaceSize = 0;
        EXPECT_EQ(ut.TestGetWorkspaceSize(&workspaceSize), c.expected);
    }
}
