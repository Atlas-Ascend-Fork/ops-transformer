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
#include "tiling_case_executor.h"
#include "gmm_csv_ge_parse_utils.h"
#include "../../../../grouped_matmul_swiglu_quant_v2/op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.h"
#include "../../../../grouped_matmul_swiglu_quant_v2/op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tensor_api_tiling_data.h"

TEST(GroupedMatmulActivationTiling, IndependentBf16IrKeepsBothModesOnDirectOutputPath)
{
    using Desc = gert::TilingContextPara::TensorDescription;
    using Value = Ops::Transformer::AnyValue;
    const auto tensor = [](std::vector<int64_t> shape, ge::DataType dtype) {
        return Desc(ops::ut::MakeGertStorageShape(shape, shape), dtype, ge::FORMAT_ND);
    };
    for (int64_t swigluMode : {0L, 2L}) {
        for (bool trans : {false, true}) {
            SCOPED_TRACE(::testing::Message() << "swigluMode=" << swigluMode << ", transposeWeight=" << trans);
            optiling::GMMSwigluV2CompileInfo info{};
            info.aicNum_ = 32;
            info.aivNum_ = 64;
            info.ubSize_ = 262144;
            info.l1Size_ = 524288;
            info.l0CSize_ = 262144;
            info.npuArch_ = static_cast<int32_t>(NpuArch::DAV_3510);
            info.supportL12BtBf16 = true;
            const std::vector<int64_t> w =
                trans ? std::vector<int64_t>{1, 256, 128} : std::vector<int64_t>{1, 128, 256};
            const std::vector<int64_t> storage =
                trans ? std::vector<int64_t>{1, 2, 16, 16, 64} : std::vector<int64_t>{1, 4, 8, 16, 64};
            const std::vector<int64_t> scale =
                trans ? std::vector<int64_t>{1, 256, 2, 2} : std::vector<int64_t>{1, 2, 256, 2};
            gert::TilingContextPara context(
                "GroupedMatmulActivation",
                {tensor({32, 128}, ge::DT_FLOAT4_E2M1), tensor({32, 2, 2}, ge::DT_FLOAT8_E8M0),
                 tensor({1}, ge::DT_INT64),
                 Desc(ops::ut::MakeGertStorageShape(w, storage), ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ),
                 tensor(scale, ge::DT_FLOAT8_E8M0), tensor({0}, ge::DT_BF16)},
                {tensor({32, 128}, ge::DT_BF16)},
                {{"dequant_mode", Value::CreateFrom<int64_t>(2)},
                 {"dequant_dtype", Value::CreateFrom<int64_t>(0)},
                 {"act_type", Value::CreateFrom<std::string>("swiglu")},
                 {"group_list_type", Value::CreateFrom<int64_t>(0)},
                 {"tuning_config", Value::CreateFrom<std::vector<int64_t>>({})},
                 {"swiglu_mode", Value::CreateFrom<int64_t>(swigluMode)},
                 {"clamp_limit", Value::CreateFrom<float>(7.0F)},
                 {"glu_alpha", Value::CreateFrom<float>(1.702F)},
                 {"glu_bias", Value::CreateFrom<float>(1.0F)},
                 {"transpose_weight", Value::CreateFrom<bool>(trans)}},
                {1, 1, 1, 1, 1, 1}, {1}, &info, "Ascend950", 32, 262144);
            context.socInfoString_ =
                R"({"hardware_info":{"CORE_NUM":32,"cube_core_cnt":32,"vector_core_cnt":64,"UB_SIZE":262144,"L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":262144}})";
            TilingInfo result;
            ASSERT_TRUE(ExecuteTiling(context, result));
            EXPECT_EQ(result.tilingKey, trans ? 1 : 0);
            ASSERT_EQ(result.tilingDataSize,
                      sizeof(GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData));
            const auto* tiling =
                reinterpret_cast<const GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData*>(
                    result.tilingData.get());
            EXPECT_EQ(tiling->swigluParams.swigluMode, swigluMode);
        }
    }
}

TEST(GroupedMatmulActivationTiling, LegacyV2Fp4ModeZeroKeepsQuantizedOutput)
{
    using Desc = gert::TilingContextPara::TensorDescription;
    using Value = Ops::Transformer::AnyValue;
    const auto tensor = [](std::vector<int64_t> shape, ge::DataType dtype) {
        return Desc(ops::ut::MakeGertStorageShape(shape, shape), dtype, ge::FORMAT_ND);
    };
    for (bool trans : {false, true}) {
        optiling::GMMSwigluV2CompileInfo info{};
        info.aicNum_ = 32;
        info.aivNum_ = 64;
        info.ubSize_ = 262144;
        info.l1Size_ = 524288;
        info.l0CSize_ = 262144;
        info.npuArch_ = static_cast<int32_t>(NpuArch::DAV_3510);
        info.supportL12BtBf16 = true;
        const std::vector<int64_t> w = trans ? std::vector<int64_t>{1, 256, 128} : std::vector<int64_t>{1, 128, 256};
        const std::vector<int64_t> storage =
            trans ? std::vector<int64_t>{1, 2, 16, 16, 64} : std::vector<int64_t>{1, 4, 8, 16, 64};
        const std::vector<int64_t> scale =
            trans ? std::vector<int64_t>{1, 256, 2, 2} : std::vector<int64_t>{1, 2, 256, 2};
        gert::TilingContextPara context(
            "GroupedMatmulSwigluQuantV2",
            {tensor({32, 128}, ge::DT_FLOAT4_E2M1), tensor({32, 2, 2}, ge::DT_FLOAT8_E8M0), tensor({1}, ge::DT_INT64),
             Desc(ops::ut::MakeGertStorageShape(w, storage), ge::DT_FLOAT4_E2M1, ge::FORMAT_FRACTAL_NZ),
             tensor(scale, ge::DT_FLOAT8_E8M0)},
            {tensor({32, 128}, ge::DT_FLOAT8_E4M3FN), tensor({32, 2, 2}, ge::DT_FLOAT8_E8M0)},
            {{"dequant_mode", Value::CreateFrom<int64_t>(2)},
             {"dequant_dtype", Value::CreateFrom<int64_t>(0)},
             {"quant_mode", Value::CreateFrom<int64_t>(2)},
             {"quant_dtype", Value::CreateFrom<int64_t>(static_cast<int64_t>(ge::DT_FLOAT8_E4M3FN))},
             {"transpose_weight", Value::CreateFrom<bool>(trans)},
             {"group_list_type", Value::CreateFrom<int64_t>(0)},
             {"tuning_config", Value::CreateFrom<std::vector<int64_t>>({})},
             {"swiglu_mode", Value::CreateFrom<int64_t>(0)},
             {"clamp_limit", Value::CreateFrom<float>(7.0F)},
             {"glu_alpha", Value::CreateFrom<float>(1.702F)},
             {"glu_bias", Value::CreateFrom<float>(1.0F)},
             {"round_mode", Value::CreateFrom<std::string>("rint")},
             {"scale_alg", Value::CreateFrom<int64_t>(0)},
             {"dst_type_max", Value::CreateFrom<float>(0.0F)}},
            {1, 1, 1, 1, 1, 0, 0, 0}, {1, 1}, &info, "Ascend950", 32, 262144);
        context.socInfoString_ =
            R"({"hardware_info":{"CORE_NUM":32,"cube_core_cnt":32,"vector_core_cnt":64,"UB_SIZE":262144,"L1_SIZE":524288,"L0A_SIZE":65536,"L0B_SIZE":65536,"L0C_SIZE":262144}})";
        TilingInfo result;
        ASSERT_TRUE(ExecuteTiling(context, result));
        EXPECT_EQ(result.tilingKey, trans ? 1 : 0);
        EXPECT_NE(result.tilingDataSize,
                  sizeof(GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData));
    }
}
