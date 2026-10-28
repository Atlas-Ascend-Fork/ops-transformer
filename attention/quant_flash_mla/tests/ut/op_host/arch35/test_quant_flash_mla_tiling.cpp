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
#include "../quant_flash_mla_param.h"
#include "tiling_case_executor.h"

namespace QuantFlashMlaUT {

class QuantFlashMlaArch35TilingTest : public testing::TestWithParam<QuantFlashMlaTilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "QuantFlashMla Arch35 TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "QuantFlashMla Arch35 TilingTest TearDown" << std::endl;
    }
};

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(QuantFlashMlaArch35TilingTest);

TEST_P(QuantFlashMlaArch35TilingTest, param)
{
    auto param = GetParam();
    optiling::quant_flash_mla::QuantFlashMlaCompileInfo compileInfo = {};

    const std::string A5SocInfo = "{\n"
                                  "  \"hardware_info\": {\n"
                                  "    \"BT_SIZE\": 0,\n"
                                  "    \"load3d_constraints\": \"1\",\n"
                                  "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                  "    \"Intrinsic_data_move_l12ub\": true,\n"
                                  "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                  "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                  "    \"UB_SIZE\": 196608,\n"
                                  "    \"L2_SIZE\": 117440512,\n"
                                  "    \"L1_SIZE\": 524288,\n"
                                  "    \"L0A_SIZE\": 65536,\n"
                                  "    \"L0B_SIZE\": 65536,\n"
                                  "    \"L0C_SIZE\": 65536,\n"
                                  "    \"vector_core_cnt\": 64,\n"
                                  "    \"cube_core_cnt\": 32,\n"
                                  "    \"socVersion\": \"Ascend950\"\n"
                                  "  }\n"
                                  "}";

    gert::TilingContextPara tilingContextPara(
        "QuantFlashMla",
        {param.q, param.k_cache, param.q_descale, param.k_descale, param.block_table, param.cache_seqlens,
         param.cu_seqlens_q, param.seqused_q, param.attn_mask, param.metadata},
        {param.attn_out, param.softmax_lse},
        {
            {"quant_compute_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.quant_compute_mode)},
            {"softmax_scale", Ops::Transformer::AnyValue::CreateFrom<float>(param.softmax_scale)},
            {"mask_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.mask_mode)},
            {"max_seqlen_q", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_q)},
            {"max_seqlen_kv", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.max_seqlen_kv)},
            {"head_dim_v", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.head_dim_v)},
            {"layout_q", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_q)},
            {"layout_kv", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_kv)},
            {"layout_out", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.layout_out)},
            {"return_softmax_lse", Ops::Transformer::AnyValue::CreateFrom<bool>(param.return_softmax_lse)},
        },
        &compileInfo, "Ascend950", 64, 196608, 16384, A5SocInfo);
    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, param.expectTilingDataHash, {}, 0,
                    true);
}

INSTANTIATE_TEST_SUITE_P(
    QuantFlashMla, QuantFlashMlaArch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<QuantFlashMlaTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantFlashMlaTilingUtParam>);

} // namespace QuantFlashMlaUT
