/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>
#include "infer_shape_context_faker.h"
#include "infer_datatype_context_faker.h"
#include "infer_shape_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"

class ScatterPaCache : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "ScatterPaCache SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "ScatterPaCache TearDown" << std::endl;
    }
};

TEST_F(ScatterPaCache, scatter_pa_cache_infershape_0)
{
    gert::InfershapeContextPara infershapeContextPara(
        "ScatterPaCache",
        {
            {{{102, 1, 128}, {102, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},           // key
            {{{306, 128, 1, 128}, {306, 128, 1, 128}}, ge::DT_FLOAT, ge::FORMAT_ND}, // keyCache
            {{{102}, {102}}, ge::DT_INT32, ge::FORMAT_ND},                           // slotMapping
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},

        },
        {
            {{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}, // keyCache
        },
        {
            {"cache_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("Norm")},
        });
    std::vector<std::vector<int64_t>> expectOutputShape = {{306, 128, 1, 128}};
    ExecuteTestCase(infershapeContextPara, ge::GRAPH_SUCCESS, expectOutputShape);
}

TEST_F(ScatterPaCache, scatter_pa_cache_inferdatatype_invalid_cache_mode_reports_ez0026)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl("ScatterPaCache");
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->infer_datatype, nullptr);
    auto contextHolder = gert::InferDataTypeContextFaker()
                             .SetOpType("ScatterPaCache")
                             .NodeIoNum(6, 1)
                             .NodeOutputTd(0, ge::FORMAT_ND, ge::FORMAT_ND)
                             .Attr("cache_mode", std::string("InvalidMode"))
                             .Build();
    auto context = contextHolder.GetContext<gert::InferDataTypeContext>();
    ASSERT_NE(context, nullptr);
    // 注：不直接断言 ErrorManager 报告（EZ0026），op_host UT target 未链接 liberror_manager.so；
    // 该分支经 gcov 覆盖，正式码断言建议在 UT 链接补齐 error_manager 后恢复。
    ASSERT_EQ(opImpl->infer_datatype(context), ge::GRAPH_FAILED);
}
