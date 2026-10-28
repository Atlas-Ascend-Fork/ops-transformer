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
gert::InfershapeContextPara MakeInferShapeContext(const std::string& layout = "BSND")
{
    gert::InfershapeContextPara context(
        "QuantFlashAttnGrad",
        {
            {{{2, 512, 4, 128}, {2, 512, 4, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},   // q
            {{{2, 1024, 4, 128}, {2, 1024, 4, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // k
            {{{2, 1024, 4, 128}, {2, 1024, 4, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND}, // v
            {{{2, 512, 4, 128}, {2, 512, 4, 128}}, ge::DT_HIFLOAT8, ge::FORMAT_ND},   // dout
            {{{2, 512, 4, 128}, {2, 512, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},       // attn_out
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // q_descale
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // k_descale
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // v_descale
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // do_descale
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // p_scale
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},                                // ds_scale
            {{{2, 4, 512}, {2, 4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},                // softmax_lse
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                  // cu_seqlens_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                  // cu_seqlens_kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                  // seqused_q
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                                  // seqused_kv
            {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},                                   // sinks
            {{{}, {}}, ge::DT_INT8, ge::FORMAT_ND},                                   // attn_mask
            {{{2, 16}, {2, 16}}, ge::DT_INT32, ge::FORMAT_ND},                        // metadata
        },
        {
            {{{2, 512, 4, 128}, {2, 512, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND},   // dq
            {{{2, 1024, 4, 128}, {2, 1024, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // dk
            {{{2, 1024, 4, 128}, {2, 1024, 4, 128}}, ge::DT_BF16, ge::FORMAT_ND}, // dv
            {{{4}, {4}}, ge::DT_FLOAT, ge::FORMAT_ND},                            // dsink
        },
        {
            {"quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(0.088388f)},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)},
            {"win_left", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"win_right", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(layout)},
        });
    if (layout == "BNSD") {
        context.inputTensorDesc_[0].shape_ = {{2, 4, 512, 128}, {2, 4, 512, 128}};
        context.inputTensorDesc_[1].shape_ = {{2, 4, 1024, 128}, {2, 4, 1024, 128}};
    } else if (layout == "TND") {
        context.inputTensorDesc_[0].shape_ = {{1024, 4, 128}, {1024, 4, 128}};
        context.inputTensorDesc_[1].shape_ = {{2048, 4, 128}, {2048, 4, 128}};
        context.inputTensorDesc_[11].shape_ = {{4, 1024}, {4, 1024}};
        context.inputTensorDesc_[12].shape_ = {{3}, {3}};
        context.inputTensorDesc_[13].shape_ = {{3}, {3}};
        context.inputTensorDesc_[18].shape_ = {{2, 37}, {2, 37}};
    }
    context.inputTensorDesc_[2].shape_ = context.inputTensorDesc_[1].shape_;
    context.inputTensorDesc_[3].shape_ = context.inputTensorDesc_[0].shape_;
    context.inputTensorDesc_[4].shape_ = context.inputTensorDesc_[0].shape_;
    for (size_t index = 0; index < 3; ++index) {
        context.outputTensorDesc_[index].shape_ = context.inputTensorDesc_[index].shape_;
    }
    for (auto& output : context.outputTensorDesc_) {
        output.shape_ = {{}, {}};
    }
    return context;
}
} // namespace

class QuantFlashAttnGradProto : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantFlashAttnGradProto SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantFlashAttnGradProto TearDown" << std::endl;
    }
};

TEST_F(QuantFlashAttnGradProto, quant_flash_attn_grad_infershape_bsnd)
{
    auto context = MakeInferShapeContext("BSND");
    const std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 512, 4, 128}, {2, 1024, 4, 128}, {2, 1024, 4, 128}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantFlashAttnGradProto, quant_flash_attn_grad_infershape_bnsd)
{
    auto context = MakeInferShapeContext("BNSD");
    const std::vector<std::vector<int64_t>> expectOutputShape = {
        {2, 4, 512, 128}, {2, 4, 1024, 128}, {2, 4, 1024, 128}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantFlashAttnGradProto, quant_flash_attn_grad_infershape_tnd)
{
    auto context = MakeInferShapeContext("TND");
    const std::vector<std::vector<int64_t>> expectOutputShape = {{1024, 4, 128}, {2048, 4, 128}, {2048, 4, 128}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(QuantFlashAttnGradProto, quant_flash_attn_grad_infershape_lowercase_layout)
{
    auto context = MakeInferShapeContext("BSND");
    context.attrs_[7].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::string>("bsnd");
    context.attrs_[8].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::string>("BsNd");
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{2, 512, 4, 128}, {2, 1024, 4, 128}, {2, 1024, 4, 128}});
}

TEST_F(QuantFlashAttnGradProto, quant_flash_attn_grad_infershape_invalid_layout)
{
    for (size_t index : {7U, 8U}) {
        for (const std::string& layout : {"", "BSH", "BNSD"}) {
            SCOPED_TRACE(testing::Message() << "attr=" << index << ", layout=" << layout);
            auto context = MakeInferShapeContext();
            context.attrs_[index].attr_ = Ops::Transformer::AnyValue::CreateFrom<std::string>(layout);
            ExecuteTestCase(context, ge::GRAPH_FAILED);
        }
    }
}

TEST_F(QuantFlashAttnGradProto, quant_flash_attn_grad_infershape_missing_layout)
{
    auto context = MakeInferShapeContext();
    context.attrs_.pop_back();
    ExecuteTestCase(context, ge::GRAPH_FAILED);
}

TEST_F(QuantFlashAttnGradProto, quant_flash_attn_grad_infershape_dynamic_dimensions)
{
    auto context = MakeInferShapeContext("TND");
    context.inputTensorDesc_[0].shape_ = {{-1, 4, 128}, {-1, 4, 128}};
    context.inputTensorDesc_[1].shape_ = {{-1, 4, 128}, {-1, 4, 128}};
    context.inputTensorDesc_[2].shape_ = {{-1, 4, 128}, {-1, 4, 128}};
    ExecuteTestCase(context, ge::GRAPH_SUCCESS, {{-1, 4, 128}, {-1, 4, 128}, {-1, 4, 128}});
}

TEST_F(QuantFlashAttnGradProto, quant_flash_attn_grad_inferdtype_hifloat8)
{
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(registry, nullptr);
    auto opImpl = registry->GetOpImpl("QuantFlashAttnGrad");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);
    ge::DataType quantType = ge::DT_HIFLOAT8;
    ge::DataType bf16Type = ge::DT_BF16;
    ge::DataType floatType = ge::DT_FLOAT;
    ge::DataType intType = ge::DT_INT32;
    ge::DataType maskType = ge::DT_INT8;
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("QuantFlashAttnGrad")
                             .NodeIoNum(19, 4)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(1, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(2, ge::FORMAT_ND, ge::FORMAT_ND)
                             .NodeOutputTd(3, ge::FORMAT_ND, ge::FORMAT_ND)
                             .InputDataTypes({&quantType, &quantType, &quantType, &quantType, &bf16Type, &floatType,
                                              &floatType, &floatType, &floatType, &floatType, &floatType, &floatType,
                                              &intType, &intType, &intType, &intType, &bf16Type, &maskType, &intType})
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    ASSERT_EQ(opImpl->infer_datatype(context), ge::GRAPH_SUCCESS);
    EXPECT_EQ(context->GetOutputDataType(0), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(1), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(2), ge::DT_BF16);
    EXPECT_EQ(context->GetOutputDataType(3), ge::DT_FLOAT);
}
