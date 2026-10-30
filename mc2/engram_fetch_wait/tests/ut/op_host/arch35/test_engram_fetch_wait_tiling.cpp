/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_engram_fetch_wait_tiling.cpp
 * \brief EngramFetchWait host tiling UT
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"

namespace EngramFetchWaitUT {

static const std::string OP_NAME = "EngramFetchWait";

struct EngramFetchWaitTestParam {
    std::string caseName;
    std::initializer_list<int64_t> commContextShape;
    ge::DataType commContextDtype;
    std::initializer_list<int64_t> fetchedShape;
    ge::DataType fetchedDtype;
    std::string socVersion;
    ge::graphStatus status;
    uint64_t expectTilingKey;
};

inline std::ostream& operator<<(std::ostream& os, const EngramFetchWaitTestParam& param)
{
    return os << param.caseName;
}

static EngramFetchWaitTestParam g_testCases[] = {
    {"success_bf16", {6146}, ge::DT_INT32, {8, 512}, ge::DT_BF16, "3510", ge::GRAPH_SUCCESS, 0UL},
    {"success_fp16", {6146}, ge::DT_INT32, {8, 512}, ge::DT_FLOAT16, "3510", ge::GRAPH_SUCCESS, 0UL},
    {"success_fp32", {6146}, ge::DT_INT32, {16, 256}, ge::DT_FLOAT, "3510", ge::GRAPH_SUCCESS, 0UL},
    {"fail_commContext_bf16", {6146}, ge::DT_BF16, {8, 512}, ge::DT_BF16, "3510", ge::GRAPH_FAILED, 0UL},
    {"fail_commContext_fp16", {6146}, ge::DT_FLOAT16, {8, 512}, ge::DT_FLOAT16, "3510", ge::GRAPH_FAILED, 0UL},
};

class EngramFetchWaitArch35TilingTest : public testing::TestWithParam<EngramFetchWaitTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "EngramFetchWaitArch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "EngramFetchWaitArch35TilingTest TearDown." << std::endl;
    }
};

static struct EngramFetchWaitCompileInfo {
} compileInfo;

static gert::TilingContextPara BuildTilingContextPara(const EngramFetchWaitTestParam& param)
{
    std::cout << "[TEST_CASE] " << param.caseName << std::endl;
    gert::StorageShape commContextShape = {param.commContextShape, param.commContextShape};
    gert::StorageShape fetchedShape = {param.fetchedShape, param.fetchedShape};

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {{commContextShape, param.commContextDtype, ge::FORMAT_ND}, {fetchedShape, param.fetchedDtype, ge::FORMAT_ND}});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_(
        {{fetchedShape, param.fetchedDtype, ge::FORMAT_ND}});

    std::vector<gert::TilingContextPara::OpAttr> attrs_;

    return gert::TilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                   param.socVersion);
}

TEST_P(EngramFetchWaitArch35TilingTest, GeneralCases)
{
    auto param = GetParam();
    auto tilingContextPara = BuildTilingContextPara(param);
    ExecuteTestCase(tilingContextPara, param.status, param.expectTilingKey, "", {});
}

INSTANTIATE_TEST_CASE_P(EngramFetchWaitTilingUT, EngramFetchWaitArch35TilingTest, testing::ValuesIn(g_testCases));

} // namespace EngramFetchWaitUT
