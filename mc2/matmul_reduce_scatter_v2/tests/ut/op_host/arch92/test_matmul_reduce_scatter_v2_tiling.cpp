/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "../matmul_reduce_scatter_v2_host_ut_param.h"
#include "mc2_tiling_case_executor.h"
#ifdef ENABLE_MMRS_DT_HIFLOAT4

namespace matmul_reduce_scatter_v2_ut {

namespace {
constexpr uint64_t KIB = 1024;
constexpr uint64_t ASCEND960_UB_BYTES = 384 * KIB;
constexpr uint64_t ASCEND960_L1_BYTES = 512 * KIB;
constexpr uint64_t ASCEND960_L0AB_BYTES = 64 * KIB;
constexpr uint64_t ASCEND960_L0C_BYTES = 256 * KIB;
constexpr uint64_t ASCEND960_BT_BYTES = KIB;
constexpr uint64_t TEST_L2_BYTES = 32 * KIB * KIB;
} // namespace

class MatmulReduceScatterV2Arch92TilingTest : public testing::TestWithParam<MatmulReduceScatterV2TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulReduceScatterV2 Arch92TilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulReduceScatterV2 Arch92TilingTest TearDown" << std::endl;
    }
};

TEST_P(MatmulReduceScatterV2Arch92TilingTest, param)
{
    auto param = GetParam();
    struct MatmulReduceScatterV2CompileInfo {};
    MatmulReduceScatterV2CompileInfo compileInfo;
    gert::TilingContextPara tilingContextPara(
        "MatmulReduceScatterV2",
        {
            param.x1,
            param.x2,
            param.bias,
            param.x1_scale,
            param.x2_scale,
            param.quant_scale,
        },
        {
            param.y,
            param.amax_out,
        },
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"reduce_op", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.reduce_op)},
         {"is_trans_a", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_a)},
         {"is_trans_b", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_trans_b)},
         {"comm_turn", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.comm_turn)},
         {"rank_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.rank_size)},
         {"block_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.block_size)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.group_size)},
         {"is_amax_out", Ops::Transformer::AnyValue::CreateFrom<bool>(param.is_amax_out)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.y_dtype)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.comm_mode)}},
        param.inputInstance, param.outputInstance, &compileInfo, param.soc, param.coreNum, param.ubsize);
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rank_size}};
    ASSERT_EQ(param.ubsize, ASCEND960_UB_BYTES);
    // 960 的 AIC:AIV 核数比为 1:1。
    tilingContextPara.socInfoString_ =
        R"({"hardware_info":{"BT_SIZE":)" + std::to_string(ASCEND960_BT_BYTES) +
        R"(,"load3d_constraints":"1",)"
        R"("Intrinsic_fix_pipe_l0c2out":false,"Intrinsic_data_move_l12ub":true,)"
        R"("Intrinsic_data_move_l0c2ub":true,"Intrinsic_data_move_out2l1_nd2nz":false,)"
        R"("L2_SIZE":)" +
        std::to_string(TEST_L2_BYTES) + R"(,"UB_SIZE":)" + std::to_string(ASCEND960_UB_BYTES) + R"(,"L1_SIZE":)" +
        std::to_string(ASCEND960_L1_BYTES) + R"(,"L0A_SIZE":)" + std::to_string(ASCEND960_L0AB_BYTES) +
        R"(,"L0B_SIZE":)" + std::to_string(ASCEND960_L0AB_BYTES) + R"(,"L0C_SIZE":)" +
        std::to_string(ASCEND960_L0C_BYTES) + R"(,"CORE_NUM":)" + std::to_string(param.coreNum) +
        R"(,"cube_core_cnt":)" + std::to_string(param.coreNum) + R"(,"vector_core_cnt":)" +
        std::to_string(param.coreNum) + "}}";
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, param.expectResult, param.expectTilingKey,
                       param.expectTilingDataHash, {}, MC2_TILING_DATA_RESERVED_LEN, true);
}

INSTANTIATE_TEST_SUITE_P(
    MatmulReduceScatterV2Ascend960, MatmulReduceScatterV2Arch92TilingTest,
    testing::ValuesIn(GetCasesFromCsv<MatmulReduceScatterV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    GetCaseInfoString<MatmulReduceScatterV2TilingUtParam>);

} // namespace matmul_reduce_scatter_v2_ut
#endif
