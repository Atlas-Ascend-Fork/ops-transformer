/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace {
gert::InfershapeContextPara MakeInferShapeContext(const std::string& layout, ge::DataType dtype)
{
    gert::InfershapeContextPara context(
        "SparseLightningIndexerKLLossGrad",
        {
            {{{2, 128, 8, 128}, {2, 128, 8, 128}}, dtype, ge::FORMAT_ND},      // q
            {{{2, 256, 1, 128}, {2, 256, 1, 128}}, dtype, ge::FORMAT_ND},      // k
            {{{2, 128, 8}, {2, 128, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},         // w
            {{{2, 128, 1, 64}, {2, 128, 1, 64}}, ge::DT_INT32, ge::FORMAT_ND}, // sparse_indices
            {{{2, 128, 1, 64}, {2, 128, 1, 64}}, ge::DT_FLOAT, ge::FORMAT_ND}, // attn_softmax_l1_norm
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // cu_seqlens_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // cu_seqlens_k
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // seqused_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // seqused_k
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                           // cmp_residual_k
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},                       // metadata
        },
        {
            {{{}, {}}, dtype, ge::FORMAT_ND},        // dq
            {{{}, {}}, dtype, ge::FORMAT_ND},        // dk
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // dw
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_out
        },
        {
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
            {"layout_k", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
        });
    if (layout == "TND") {
        context.inputTensorDesc_[0].shape_ = {{256, 8, 128}, {256, 8, 128}};
        context.inputTensorDesc_[1].shape_ = {{512, 1, 128}, {512, 1, 128}};
        context.inputTensorDesc_[2].shape_ = {{256, 8}, {256, 8}};
        context.inputTensorDesc_[3].shape_ = {{256, 1, 64}, {256, 1, 64}};
        context.inputTensorDesc_[4].shape_ = {{256, 1, 64}, {256, 1, 64}};
        context.inputTensorDesc_[5].shape_ = {{3}, {3}};
        context.inputTensorDesc_[6].shape_ = {{3}, {3}};
    }
    return context;
}

void CheckInferDataType(ge::DataType queryType, ge::DataType keyType)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto opImpl = registry->GetOpImpl("SparseLightningIndexerKLLossGrad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);
    ge::DataType floatType = ge::DT_FLOAT;
    ge::DataType intType = ge::DT_INT32;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("SparseLightningIndexerKLLossGrad")
                             .NodeIoNum(11, 4)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&queryType, &keyType, &floatType, &intType, &floatType, &intType,
                                              &intType, &intType, &intType, &intType, &intType})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), queryType);
    EXPECT_EQ(context->GetOutputDataType(1), keyType);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_FLOAT);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT);
}
} // namespace

class SparseLightningIndexerKLLossGradProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SparseLightningIndexerKLLossGradProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SparseLightningIndexerKLLossGradProto TearDown" << std::endl;
    }
};

TEST_F(SparseLightningIndexerKLLossGradProto, sparse_lightning_indexer_kl_loss_grad_infershape_bsnd_fp16)
{
    auto context = MakeInferShapeContext("BSND", ge::DT_FLOAT16);
    const std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 128, 8, 128}, {2, 256, 1, 128}, {2, 128, 8}, {2, 128, 1, 64}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SparseLightningIndexerKLLossGradProto, sparse_lightning_indexer_kl_loss_grad_infershape_bsnd_bf16)
{
    auto context = MakeInferShapeContext("BSND", ge::DT_BF16);
    const std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 128, 8, 128}, {2, 256, 1, 128}, {2, 128, 8}, {2, 128, 1, 64}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SparseLightningIndexerKLLossGradProto, sparse_lightning_indexer_kl_loss_grad_infershape_tnd_fp16)
{
    auto context = MakeInferShapeContext("TND", ge::DT_FLOAT16);
    const std::vector<std::vector<int64_t>> expectOutputShape = {{256, 8, 128}, {512, 1, 128}, {256, 8}, {256, 1, 64}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SparseLightningIndexerKLLossGradProto, sparse_lightning_indexer_kl_loss_grad_infershape_tnd_bf16)
{
    auto context = MakeInferShapeContext("TND", ge::DT_BF16);
    const std::vector<std::vector<int64_t>> expectOutputShape = {{256, 8, 128}, {512, 1, 128}, {256, 8}, {256, 1, 64}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SparseLightningIndexerKLLossGradProto, sparse_lightning_indexer_kl_loss_grad_infershape_dynamic_dimensions)
{
    auto context = MakeInferShapeContext("TND", ge::DT_BF16);
    context.inputTensorDesc_[0].shape_ = {{-1, 8, 128}, {-1, 8, 128}};
    context.inputTensorDesc_[1].shape_ = {{-1, 1, 128}, {-1, 1, 128}};
    context.inputTensorDesc_[2].shape_ = {{-1, 8}, {-1, 8}};
    context.inputTensorDesc_[4].shape_ = {{-1, 1, 64}, {-1, 1, 64}};
    const std::vector<std::vector<int64_t>> expectOutputShape = {{-1, 8, 128}, {-1, 1, 128}, {-1, 8}, {-1, 1, 64}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(SparseLightningIndexerKLLossGradProto, sparse_lightning_indexer_kl_loss_grad_infershape_unknown_rank)
{
    auto context = MakeInferShapeContext("BSND", ge::DT_FLOAT16);
    for (size_t index : {0U, 1U, 2U, 4U}) {
        context.inputTensorDesc_[index].shape_ = {{-2}, {-2}};
    }
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-2}, {-2}, {-2}, {-2}});
}

TEST_F(SparseLightningIndexerKLLossGradProto, sparse_lightning_indexer_kl_loss_grad_inferdtype_fp16)
{
    CheckInferDataType(ge::DT_FLOAT16, ge::DT_FLOAT16);
}

TEST_F(SparseLightningIndexerKLLossGradProto, sparse_lightning_indexer_kl_loss_grad_inferdtype_bf16)
{
    CheckInferDataType(ge::DT_BF16, ge::DT_BF16);
}

TEST_F(SparseLightningIndexerKLLossGradProto, sparse_lightning_indexer_kl_loss_grad_inferdtype_independent_query_key)
{
    // Inference propagates each input independently; tiling validates compatible dtypes.
    CheckInferDataType(ge::DT_FLOAT16, ge::DT_BF16);
}
