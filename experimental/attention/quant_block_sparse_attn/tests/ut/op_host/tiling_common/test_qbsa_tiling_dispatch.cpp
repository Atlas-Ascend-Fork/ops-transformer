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
#include <cstdint>
#include <iostream>

#include "../test_utils/qbsa_tiling_case_executor.h"
#include "../../../../op_kernel/arch92/qbsa_fp8/qbsa_fp8_tiling_data.h"
#include "../../../../op_kernel/arch92/qbsa_fp8/qbsa_fp8_tiling_key.h"

namespace QuantBlockSparseAttnUT {

class QbsaTilingDispatchTest : public testing::TestWithParam<QuantBlockSparseAttnTilingUtParam> {};

TEST_P(QbsaTilingDispatchTest, RejectUnsupportedQuantMode)
{
    const auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;
    TilingInfo tilingInfo;
    EXPECT_FALSE(ExecuteQbsaTilingCase(param, tilingInfo));
}

INSTANTIATE_TEST_SUITE_P(
    QbsaTilingDispatch, QbsaTilingDispatchTest,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnTilingUtParam>);

TEST(QbsaTilingDispatchContractTest, UnsupportedModeIsNotRecognized)
{
    for (uint32_t mode : {0U, 3U, 4U, UINT32_MAX}) {
        EXPECT_FALSE(optiling::IsQbsaQuantModeSupported(mode));
    }
}

TEST(QbsaTilingDispatchContractTest, Fp8DispatchesA5AndA6WithIndependentPayloads)
{
    std::string casePath = __FILE__;
    casePath.replace(casePath.find("tiling_common"), std::string::npos, "arch35/qbsa_fp8/test_qbsa_fp8_tiling.csv");
    const auto cases = GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(casePath);
    ASSERT_FALSE(cases.empty());
    auto param = cases.front();
    PrepareQbsaFp8TilingParam(param);
    TilingInfo a5;
    TilingInfo a6;
    ASSERT_TRUE(ExecuteQbsaTilingCase(param, a5));
    ASSERT_TRUE(ExecuteQbsaTilingCase(param, a6, 4096U, false, "Ascend960dt"));
    EXPECT_NE(a5.tilingKey, a6.tilingKey);
    EXPECT_LE(a5.blockNum, 36U);
    EXPECT_LE(a6.blockNum, 64U);
    EXPECT_EQ(a5.tilingDataSize, sizeof(optiling::QuantBlockSparseAttnFp8TilingData));
    EXPECT_EQ(a6.tilingDataSize, sizeof(optiling::QuantBlockSparseAttnFp8Arch92TilingData));
    const auto data = DecodeQbsaTilingData<optiling::QuantBlockSparseAttnFp8Arch92TilingData>(a6);
    EXPECT_EQ(data.commonParams.baseParams.coreNum, a6.blockNum);
    EXPECT_EQ(data.fp8Params.singleCoreSize,
              (data.commonParams.outputParams.totalOutputSize + a6.blockNum - 1) / a6.blockNum);
    TilingInfo undersized;
    EXPECT_FALSE(ExecuteQbsaTilingCase(
        param, undersized, sizeof(optiling::QuantBlockSparseAttnFp8Arch92TilingData) - 1U, false, "Ascend960dt"));
}

} // namespace QuantBlockSparseAttnUT
