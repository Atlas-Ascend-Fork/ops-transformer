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
#include "../allto_all_matmul_v2_host_ut_param.h"
#include "tiling_case_executor.h"
#include "base/registry/op_impl_space_registry_v2.h"
#include "apace/kernel/fusions/all_to_all_matmul/all_to_all_matmul_tiling_data.h"

namespace AlltoAllMatmulV2UT {

static const std::string OP_NAME = "AlltoAllMatmulV2";

struct AlltoAllMatmulV2CompileInfo {
} compileInfo;

class AlltoAllMatmulV2Arch35TilingTest : public testing::TestWithParam<AlltoAllMatmulV2TilingUtParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllMatmulV2Arch35TilingTest SetUp." << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllMatmulV2Arch35TilingTest TearDown." << std::endl;
    }
};

TEST_P(AlltoAllMatmulV2Arch35TilingTest, param)
{
    auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc_(
        {param.context, param.x1, param.x2, param.bias, param.x1Scale, param.x2Scale});

    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc_({param.y, param.all2allOut});

    std::vector<gert::TilingContextPara::OpAttr> attrs_(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.worldSize)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.hcclBufferSize)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.yDtypeAttr)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantMode)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x2QuantMode)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.x1QuantDtype)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX1)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(param.transposeX2)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.groupSize)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.commMode)},
         {"precision_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.precisionMode)}});

    gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc_, outputTensorDesc_, attrs_, &compileInfo,
                                              param.soc, param.coreNum);
    ExecuteTestCase(tilingContextPara, param.expectResult, param.expectTilingKey, "", {});
}

INSTANTIATE_TEST_SUITE_P(
    AlltoAllMatmulV2TilingUT, AlltoAllMatmulV2Arch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<AlltoAllMatmulV2TilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<AlltoAllMatmulV2TilingUtParam>);

// Cover the rawTilingData capacity check (L651-652 and L656-657 in tiling_base.h)
// by creating TilingContextPara with artificially small/zero tilingDataSize.
TEST(AlltoAllMatmulV2SmallTilingDataTest, CapacityTooSmall)
{
    AlltoAllMatmulV2CompileInfo compileInfo;
    gert::StorageShape ctxShape = GetStorageShape("1 2052");
    gert::StorageShape x1Shape = GetStorageShape("57086 1536");
    gert::StorageShape x2Shape = GetStorageShape("9216 3072");
    gert::StorageShape x1ScaleShape = GetStorageShape("57086 24 2");
    gert::StorageShape x2ScaleShape = GetStorageShape("9216 48 2");
    gert::StorageShape yShape = GetStorageShape("28543 9216");
    gert::StorageShape all2allOutShape = GetStorageShape("28543 3072");

    gert::TilingContextPara::TensorDescription context(ctxShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x1Scale(x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x2Scale(x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription y(yShape, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription all2allOut(all2allOutShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);

    std::vector<gert::TilingContextPara::TensorDescription> inputTensorDesc(
        {context, x1, x2, TD_DEFAULT, x1Scale, x2Scale});
    std::vector<gert::TilingContextPara::TensorDescription> outputTensorDesc({y, all2allOut});

    std::vector<gert::TilingContextPara::OpAttr> attrs(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1073741824)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(27)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4295032864)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv_urma")},
         {"precision_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    // Case 1: tilingDataSize=1 → covers L656-657 (capacity too small)
    {
        gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo,
                                                  "3510", 64);
        tilingContextPara.tilingDataSize_ = 1;
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, UINT64_MAX, "", {});
    }
    // Case 2: tilingDataSize=0 → CreateCap(0) may return null → covers L651-652
    {
        gert::TilingContextPara tilingContextPara(OP_NAME, inputTensorDesc, outputTensorDesc, attrs, &compileInfo,
                                                  "3510", 64);
        tilingContextPara.tilingDataSize_ = 0;
        ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, UINT64_MAX, "", {});
    }
}

// 引擎命中(ws=4)且 cap 真正捆绑(cap<m)的 win 预算用例:
// 断实际写回切分不超可用窗口反推的 cap, 删掉引擎 maxTileM 注入即失败
TEST(AlltoAllMatmulV2WinBudgetTest, EngineCapBinds)
{
    gert::StorageShape ctxShape = GetStorageShape("1 2052");
    gert::StorageShape x1Shape = GetStorageShape("262144 1024");
    gert::StorageShape x2Shape = GetStorageShape("2560 4096");
    gert::StorageShape x1ScaleShape = GetStorageShape("262144 16 2");
    gert::StorageShape x2ScaleShape = GetStorageShape("2560 64 2");
    gert::StorageShape yShape = GetStorageShape("65536 2560");
    gert::StorageShape outShape = GetStorageShape("65536 1024");

    gert::TilingContextPara::TensorDescription context(ctxShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x1Scale(x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x2Scale(x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription y(yShape, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription out(outShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);

    std::vector<gert::TilingContextPara::TensorDescription> ins({context, x1, x2, TD_DEFAULT, x1Scale, x2Scale});
    std::vector<gert::TilingContextPara::TensorDescription> outs({y, out});

    constexpr int64_t kWinAttr = 419430400; // 400MB, 命中 >=400MB 门槛
    std::vector<gert::TilingContextPara::OpAttr> attrs(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kWinAttr)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(27)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4295032864)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv_urma")},
         {"precision_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    gert::TilingContextPara para(OP_NAME, ins, outs, attrs, &compileInfo, "3510", 64);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_NE(info.tilingData, nullptr);
    const auto* td = reinterpret_cast<const AllToAllQuantMatmulTilingData*>(info.tilingData.get());

    // cap 复算(与 tiling 同口径, 扣除 2MB 预留): (usable - ws*m*scaleRow) / (ws*(2*bytesRow+scaleRow))
    constexpr uint64_t m = 65536UL;
    constexpr uint64_t ws = 4UL;
    constexpr uint64_t bytesRow = 1024UL;
    constexpr uint64_t scaleRow = ((1024UL + 63UL) / 64UL) * 2UL;
    constexpr uint64_t reserve = 2UL * 1024UL * 1024UL;
    const uint64_t usable = static_cast<uint64_t>(kWinAttr) - reserve;
    const uint64_t cap = (usable - ws * m * scaleRow) / (ws * (2UL * bytesRow + scaleRow));
    ASSERT_LT(cap, m); // cap 真实捆绑, 否则用例无鉴别力
    const uint64_t maxTile = std::max(td->commTilingData.splitAxisTileSize, td->commTilingData.splitAxisTailSize);
    EXPECT_LE(maxTile, cap);
}

// fit-balance 未命中(标定仅 4P, ws=8 必 miss) + win 预算约束(COF 结果须落在 cap 内):
// 校验实际写回的切分四元组不超 cap, 防止兜底调用丢失导致 COF 超限切分漏过
TEST(AlltoAllMatmulV2WinBudgetTest, CofMissRespectsCap)
{
    gert::StorageShape ctxShape = GetStorageShape("1 2052");
    gert::StorageShape x1Shape = GetStorageShape("131072 2048");
    gert::StorageShape x2Shape = GetStorageShape("2560 16384");
    gert::StorageShape x1ScaleShape = GetStorageShape("131072 32 2");
    gert::StorageShape x2ScaleShape = GetStorageShape("2560 256 2");
    gert::StorageShape yShape = GetStorageShape("16384 2560"); // mPerRank x N
    gert::StorageShape outShape = GetStorageShape("16384 2048");

    gert::TilingContextPara::TensorDescription context(ctxShape, ge::DT_INT32, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x1(x1Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x2(x2Shape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x1Scale(x1ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription x2Scale(x2ScaleShape, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription y(yShape, ge::DT_BF16, ge::FORMAT_ND);
    gert::TilingContextPara::TensorDescription out(outShape, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);

    std::vector<gert::TilingContextPara::TensorDescription> ins({context, x1, x2, TD_DEFAULT, x1Scale, x2Scale});
    std::vector<gert::TilingContextPara::TensorDescription> outs({y, out});

    constexpr int64_t kWinBudget = 419430400; // 400MB 工作预算(a2a 常量口径)
    std::vector<gert::TilingContextPara::OpAttr> attrs(
        {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>("group")},
         {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(8)},
         {"hccl_buffer_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kWinBudget)},
         {"y_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(27)},
         {"x1_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"x2_quant_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(6)},
         {"x1_quant_dtype", Ops::Transformer::AnyValue::CreateFrom<int64_t>(28)},
         {"transpose_x1", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
         {"transpose_x2", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
         {"group_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4295032864)},
         {"comm_mode", Ops::Transformer::AnyValue::CreateFrom<std::string>("aiv_urma")},
         {"precision_mode", Ops::Transformer::AnyValue::CreateFrom<int64_t>(0)}});

    gert::TilingContextPara para(OP_NAME, ins, outs, attrs, &compileInfo, "3510", 64);
    TilingInfo info;
    ASSERT_TRUE(ExecuteTiling(para, info));
    ASSERT_NE(info.tilingData, nullptr);
    ASSERT_GE(info.tilingDataSize, sizeof(AllToAllQuantMatmulTilingData));
    const auto* td = reinterpret_cast<const AllToAllQuantMatmulTilingData*>(info.tilingData.get());
    const uint64_t totalTiles = td->commTilingData.splitAxisTileCnt + td->commTilingData.splitAxisTailCnt;
    ASSERT_GT(totalTiles, 0UL);

    // cap 复算(与 tiling 同口径): (预算 - ws*m*scaleRow) / (ws*(2*bytesRow + scaleRow))
    constexpr uint64_t m = 16384UL;
    constexpr uint64_t ws = 8UL;
    constexpr uint64_t bytesRow = 2048UL;                         // ka, fp8 1B
    constexpr uint64_t scaleRow = ((2048UL + 63UL) / 64UL) * 2UL; // ceil(ka/64)*2
    constexpr uint64_t reserve = 2UL * 1024UL * 1024UL;
    const uint64_t cap =
        (static_cast<uint64_t>(kWinBudget) - reserve - ws * m * scaleRow) / (ws * (2UL * bytesRow + scaleRow));
    const uint64_t maxTile = std::max(td->commTilingData.splitAxisTileSize, td->commTilingData.splitAxisTailSize);
    EXPECT_LE(maxTile, cap);
    // scale 切分恒镜像 data 切分(splitAxis 四元组一致)
    EXPECT_EQ(td->scaleCommTilingData.splitAxisTileCnt, td->commTilingData.splitAxisTileCnt);
    EXPECT_EQ(td->scaleCommTilingData.splitAxisTileSize, td->commTilingData.splitAxisTileSize);
    EXPECT_EQ(td->scaleCommTilingData.splitAxisTailCnt, td->commTilingData.splitAxisTailCnt);
    EXPECT_EQ(td->scaleCommTilingData.splitAxisTailSize, td->commTilingData.splitAxisTailSize);
}

// Cover TilingParseForAlltoAllMatmulV2 (L48/L50/L51 in tiling.cpp) which
// is registered but never called by the tiling UT.
TEST(AlltoAllMatmulV2TilingParseTest, CallTilingParse)
{
    auto spaceRegistry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
    ASSERT_NE(spaceRegistry, nullptr);
    auto opImpl = spaceRegistry->GetOpImpl(OP_NAME.c_str());
    ASSERT_NE(opImpl, nullptr);
    ASSERT_NE(opImpl->tiling_parse, nullptr);
    // TilingParseForAlltoAllMatmulV2 ignores the context parameter ((void)context),
    // so passing nullptr is safe and covers the 3-line function body.
    auto ret = opImpl->tiling_parse(nullptr);
    EXPECT_EQ(ge::GRAPH_SUCCESS, static_cast<ge::graphStatus>(ret));
}

} // namespace AlltoAllMatmulV2UT
