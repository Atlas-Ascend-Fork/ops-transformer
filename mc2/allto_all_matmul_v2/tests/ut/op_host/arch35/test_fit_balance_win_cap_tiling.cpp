/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// fit-balance win 预算约束(ceil 上限)链路单测: 兜底重切分支 + 引擎约束内切分

#include <gtest/gtest.h>
#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"

namespace FitBalanceWinCapUT {

// EnforceTileCapOnSplit: 未超限 / 不限 / 超限重切 / 粒度不足放弃 / 轮数约束四分支
TEST(FitBalanceWinCapTest, SplitFallbackBranches)
{
    // 未超限: 原样返回 0
    uint64_t cnt = 2UL, len = 300UL, tcnt = 1UL, tlen = 100UL;
    EXPECT_EQ(EnforceTileCapOnSplit(cnt, len, tcnt, tlen, 700UL, 1024UL, 256UL), 0UL);
    EXPECT_EQ(cnt, 2UL);
    EXPECT_EQ(len, 300UL);
    EXPECT_EQ(tlen, 100UL);

    // maxTileM=0: 不限, 原样
    cnt = 1UL;
    len = 5000UL;
    tcnt = 0UL;
    tlen = 0UL;
    EXPECT_EQ(EnforceTileCapOnSplit(cnt, len, tcnt, tlen, 5000UL, 0UL, 256UL), 0UL);
    EXPECT_EQ(len, 5000UL);

    // 超限 cap=1000: 512 优先(min(512, alignDown(1000,256)=768)); 2000 = 3x512 + 464
    cnt = 1UL;
    len = 2000UL;
    tcnt = 0UL;
    tlen = 0UL;
    EXPECT_EQ(EnforceTileCapOnSplit(cnt, len, tcnt, tlen, 2000UL, 1000UL, 256UL), 512UL);
    EXPECT_EQ(cnt, 3UL);
    EXPECT_EQ(len, 512UL);
    EXPECT_EQ(tcnt, 1UL);
    EXPECT_EQ(tlen, 464UL);

    // 超限 cap=400: alignDown(400,256)=256, 512 放不下取 256
    cnt = 1UL;
    len = 2000UL;
    tcnt = 0UL;
    tlen = 0UL;
    EXPECT_EQ(EnforceTileCapOnSplit(cnt, len, tcnt, tlen, 2000UL, 400UL, 256UL), 256UL);
    EXPECT_EQ(len, 256UL);

    // cap < baseM: 容不下最小对齐粒度, 放弃修正
    cnt = 1UL;
    len = 2000UL;
    tcnt = 0UL;
    tlen = 0UL;
    EXPECT_EQ(EnforceTileCapOnSplit(cnt, len, tcnt, tlen, 2000UL, 100UL, 256UL), 0UL);
    EXPECT_EQ(len, 2000UL);

    // 边界等值: max(tile,len)==cap 不重切
    cnt = 1UL;
    len = 1024UL;
    tcnt = 0UL;
    tlen = 0UL;
    EXPECT_EQ(EnforceTileCapOnSplit(cnt, len, tcnt, tlen, 5000UL, 1024UL, 256UL), 0UL);
    EXPECT_EQ(len, 1024UL);

    // 轮数约束: m=32257(512 粒度 64 轮>63), cap=2560>512 → 退到 alignDown(cap)=2560(13 轮)
    cnt = 1UL;
    len = 32257UL;
    tcnt = 0UL;
    tlen = 0UL;
    EXPECT_EQ(EnforceTileCapOnSplit(cnt, len, tcnt, tlen, 32257UL, 2560UL, 256UL), 2560UL);
    EXPECT_EQ(cnt, 12UL);
    EXPECT_EQ(len, 2560UL);
    EXPECT_EQ(tlen, 1537UL);

    // 轮数约束: cap 粒度仍超 63 轮 → 放弃
    cnt = 1UL;
    len = 200000UL;
    tcnt = 0UL;
    tlen = 0UL;
    EXPECT_EQ(EnforceTileCapOnSplit(cnt, len, tcnt, tlen, 200000UL, 2560UL, 256UL), 0UL);
    EXPECT_EQ(len, 200000UL);

    // 轮数约束: maxTileM<=512 无法放粗 → 放弃
    cnt = 1UL;
    len = 40000UL;
    tcnt = 0UL;
    tlen = 0UL;
    EXPECT_EQ(EnforceTileCapOnSplit(cnt, len, tcnt, tlen, 40000UL, 512UL, 256UL), 0UL);
    EXPECT_EQ(len, 40000UL);
}

// EnforceTileCap: CommCutResult 载体(含 shortTileAtBack/totalTileCnt 补齐)
TEST(FitBalanceWinCapTest, CutResultFallback)
{
    MatmulInputParams mmInput{};
    mmInput.mValue = 4000UL;
    mmInput.baseM = 256UL;
    CommCutResult cut{};
    cut.numLongTile = 1U;
    cut.longTileLen = 4000UL;
    cut.numShortTile = 0U;
    cut.shortTileLen = 0UL;
    cut.totalTileCnt = 1U;

    // maxTileM=0: 空操作
    EnforceTileCap(cut, mmInput);
    EXPECT_EQ(cut.longTileLen, 4000UL);
    EXPECT_EQ(cut.totalTileCnt, 1U);

    // cap 生效: 重切为 7x512 + 416
    mmInput.maxTileM = 1000UL;
    EnforceTileCap(cut, mmInput);
    EXPECT_EQ(cut.longTileLen, 512UL);
    EXPECT_EQ(cut.numLongTile, 7U);
    EXPECT_EQ(cut.shortTileLen, 416UL);
    EXPECT_EQ(cut.numShortTile, 1U);
    EXPECT_EQ(cut.totalTileCnt, 8U);
    EXPECT_TRUE(cut.shortTileAtBack);
}

// 引擎切分: maxTileM 注入后搜索限幅、NoCut 超限否决、粒度不足退 NoCut 交终检
TEST(FitBalanceWinCapTest, EngineCutRespectsCap)
{
    FitBalanceCalibration calib{};
    calib.dataflow.order = CommComputeOrder::COMM_FIRST;
    calib.dataflow.localMMPosition = LocalMMPosition::FRONT;
    calib.dataflow.perRankM = true;
    calib.commRows[0] = {TilingCommEngine::AIV_URMA, 4U, 24.7, 5.1, 5.29};
    calib.mmFit.baseBlock = {256UL, 256UL, 128UL};

    FitBalanceTilingArgs args = MakeFitBalanceEngineArgs(
        2909UL, 2560UL, 10701UL, 4U, 28U, ge::DataType::DT_FLOAT8_E4M3FN, 1UL, 2UL, TilingCommEngine::AIV_URMA);
    bool halveBaseM = false;

    // 不限制: 记录自然切分
    args.maxTileM = 0UL;
    const CommCutResult unlimited = GetFitBalanceCut(args, calib, &halveBaseM);
    ASSERT_GT(unlimited.totalTileCnt, 0U);
    const uint64_t unlimitedMaxTile = std::max(unlimited.longTileLen, unlimited.shortTileLen);

    // 限制 1536: 切分不得超限
    args.maxTileM = 1536UL;
    const CommCutResult capped = GetFitBalanceCut(args, calib, &halveBaseM);
    ASSERT_GT(capped.totalTileCnt, 0U);
    EXPECT_LE(std::max(capped.longTileLen, capped.shortTileLen), 1536UL);
    EXPECT_LE(std::max(capped.longTileLen, capped.shortTileLen), unlimitedMaxTile);
}

// K 轴族(perRankM=false, a2a 形态): 候选可行性筛选——超限候选不参与择优
TEST(FitBalanceWinCapTest, EngineCutRespectsCapKaxis)
{
    FitBalanceCalibration calib{};
    calib.dataflow.order = CommComputeOrder::COMM_FIRST;
    calib.dataflow.localMMPosition = LocalMMPosition::NONE;
    calib.dataflow.perRankM = false;
    calib.commRows[0] = {TilingCommEngine::AIV_URMA, 4U, 6.18, 8.0, 5.29};
    calib.mmFit.baseBlock = {256UL, 256UL, 128UL};

    FitBalanceTilingArgs args = MakeFitBalanceEngineArgs(
        16384UL, 2560UL, 8192UL, 4U, 28U, ge::DataType::DT_FLOAT8_E4M3FN, 1UL, 2UL, TilingCommEngine::AIV_URMA);
    bool halveBaseM = false;
    args.maxTileM = 0UL;
    const CommCutResult unlimited = GetFitBalanceCut(args, calib, &halveBaseM);
    ASSERT_GT(unlimited.totalTileCnt, 0U);

    args.maxTileM = 2048UL;
    const CommCutResult capped = GetFitBalanceCut(args, calib, &halveBaseM);
    ASSERT_GT(capped.totalTileCnt, 0U);
    EXPECT_LE(std::max(capped.longTileLen, capped.shortTileLen), 2048UL);
    EXPECT_LE(std::max(capped.longTileLen, capped.shortTileLen),
              std::max(unlimited.longTileLen, unlimited.shortTileLen));
}

} // namespace FitBalanceWinCapUT
