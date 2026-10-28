/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include "quant_flash_mla_param.h"
#include "infer_datatype_context_faker.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace QuantFlashMlaUT {

class QuantFlashMlaInferDTypeTest : public testing::TestWithParam<QuantFlashMlaInferDTypeUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantFlashMla InferDTypeTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantFlashMla InferDTypeTest TearDown" << std::endl;
    }
};

// 临时添加，待框架修复后删除
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantFlashMlaInferDTypeTest);

TEST_P(QuantFlashMlaInferDTypeTest, param)
{
    auto param = GetParam();

    std::vector<void*> inputDataTypes;
    if (param.inputInstance[0] == 1)
        inputDataTypes.emplace_back(&param.q);
    if (param.inputInstance[1] == 1)
        inputDataTypes.emplace_back(&param.k_cache);
    if (param.inputInstance[2] == 1)
        inputDataTypes.emplace_back(&param.q_descale);
    if (param.inputInstance[3] == 1)
        inputDataTypes.emplace_back(&param.k_descale);
    if (param.inputInstance[4] == 1)
        inputDataTypes.emplace_back(&param.block_table);
    if (param.inputInstance[5] == 1)
        inputDataTypes.emplace_back(&param.cache_seqlens);
    if (param.inputInstance[6] == 1)
        inputDataTypes.emplace_back(&param.cu_seqlens_q);
    if (param.inputInstance[7] == 1)
        inputDataTypes.emplace_back(&param.seqused_q);
    if (param.inputInstance[8] == 1)
        inputDataTypes.emplace_back(&param.attn_mask);
    if (param.inputInstance[9] == 1)
        inputDataTypes.emplace_back(&param.metadata);

    std::vector<void*> outputDataTypes;
    if (param.outputInstance[0] == 1)
        outputDataTypes.emplace_back(&param.attn_out);
    if (param.outputInstance[1] == 1)
        outputDataTypes.emplace_back(&param.softmax_lse);

    auto contextHolder =
        gert::InferDataTypeContextFaker()
            .SetOpType("QuantFlashMla")
            .IrInstanceNum(param.inputInstance, param.outputInstance)
            .InputDataTypes(inputDataTypes)
            .OutputDataTypes(outputDataTypes)
            .NodeAttrs(
                {{"quant_compute_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quant_compute_mode)},
                 {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
                 {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
                 {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_q)},
                 {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_kv)},
                 {"head_dim_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.head_dim_v)},
                 {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
                 {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
                 {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_out)},
                 {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(param.return_softmax_lse)}})
            .Build();

    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    auto inferDtypeFunc = spaceRegistry->GetOpImpl("QuantFlashMla")->infer_datatype;
    ASSERT_EQ(inferDtypeFunc(contextHolder.GetContext<gert::InferDataTypeContext>()), param.expectResult);
    if (param.expectResult == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(0), param.attn_out);
        EXPECT_EQ(contextHolder.GetContext<gert::InferDataTypeContext>()->GetOutputDataType(1), param.softmax_lse);
    }
}

INSTANTIATE_TEST_SUITE_P(
    QuantFlashMla, QuantFlashMlaInferDTypeTest,
    testing::ValuesIn(GetCasesFromCsv<QuantFlashMlaInferDTypeUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantFlashMlaInferDTypeUtParam>);

} // namespace QuantFlashMlaUT
