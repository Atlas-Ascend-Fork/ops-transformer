/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <numeric>
#include "flash_attention_grad_tiling_s_bn2gs1s2.h"
#include "base/tiling_type.h"
#include "base/tiling_templates_registry.h"

namespace optiling {

constexpr uint64_t WORKSPACE_LIMIT = 2ULL * 1024 * 1024 * 1024; // dq workspace空间限制, 设置2g
constexpr uint64_t SPLIT_ALIGN_SIZE = 8;
constexpr uint32_t MAX_STRIDE_SIZE = 16 * 1024;

bool FlashAttentionGradTilingSBn2gs1s2::IsCapable()
{
    // 确定性实现版本

    bool deterministic = *context_->GetAttrs()->GetAttrPointer<bool>(ATTR_DETERMINISTIC_INDEX);
    if (deterministic) {
        OP_LOGD(opName, "In determinstic, tile s1/s2 seprately to calculate dq/dk/dv");
        return true;
    }
    return false;
}

uint64_t FlashAttentionGradTilingSBn2gs1s2::GetTilingKey() const
{
    // DtypeEnum::BFLOAT16:2, DtypeEnum::FLOAT16_PRECISION:3
    auto dtypeValue = fBaseParams.mode == BF16 ? DtypeEnum::BFLOAT16 : DtypeEnum::FLOAT16_PRECISION;
    uint64_t tilingKey = 9 * 100 + static_cast<uint64_t>(dtypeValue) * 10;
    OP_LOGD(opName, "FAGTiling Us2Bbn2gs1s2 DoTiling success, tiling is %lu.", tilingKey);
    return tilingKey;
}

ge::graphStatus FlashAttentionGradTilingSBn2gs1s2::DoOpTiling()
{
    DoSplit();
    DoSparse();
    DoPreTiling();
    DoPostTiling();
    DetermineMode();
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionGradTilingSBn2gs1s2::DoSplit()
{
    fBaseParams.s1CvRatio = S1CV_RATIO_DEFAULT;
    fBaseParams.s2CvRatio = S2CV_RATIO_DEFAULT;
    // @lg: 这里先沿用dk
    if (fBaseParams.d == 64) { // d size is 64
        fBaseParams.s2CvRatio = CV_RATIO_16;
        if (fBaseParams.s1 >= 256) {     // 256 is s1 size
            if (fBaseParams.s2 <= 128) { // 128 is s2 size
                fBaseParams.s1CvRatio = CV_RATIO_4;
                fBaseParams.s2CvRatio = CV_RATIO_2;
            }
        }
    }

    std::tuple<uint32_t, uint32_t, uint32_t> bestSplitRes = FuzzyForBestSplit();
    // s1 inner size, 每个core上每次处理vector 数据量
    uint32_t s1Inner = std::get<0>(bestSplitRes);
    // s1 inner size, 每个core上每次处理cube 数据量
    uint32_t s1CvInner = s1Inner * fBaseParams.s1CvRatio;
    uint32_t s1Outer = (fBaseParams.s1 + s1CvInner - 1) / s1CvInner;
    uint32_t s1TailTmp = fBaseParams.s1 % s1Inner;
    uint32_t s1CvTailTmp = fBaseParams.s1 % s1CvInner;
    fBaseParams.s1Tail = s1TailTmp == 0 ? s1Inner : s1TailTmp;
    fBaseParams.s1CvTail = s1CvTailTmp == 0 ? s1CvInner : s1CvTailTmp;
    fBaseParams.s1Inner = s1Inner;
    fBaseParams.s1CvInner = s1CvInner;
    fBaseParams.s1Outer = s1Outer;

    uint32_t s2Inner = std::get<1>(bestSplitRes);
    uint32_t cvS2Inner = s2Inner * fBaseParams.s2CvRatio;
    uint32_t s2Outer = (fBaseParams.s2 + cvS2Inner - 1) / cvS2Inner;
    uint32_t s2TailTmp = fBaseParams.s2 % s2Inner;
    uint32_t s2Tail = s2TailTmp == 0 ? s2Inner : s2TailTmp;
    fBaseParams.s2Outer = s2Outer;
    fBaseParams.cvS2Inner = cvS2Inner;
    fBaseParams.s2Inner = s2Inner;
    fBaseParams.s2Tail = s2Tail;
    fBaseParams.baseMN = s1Inner * s2Inner;

    uint32_t sfmgdInner = std::get<2>(bestSplitRes);
    uint32_t sfmgdOuter = (fBaseParams.dV + sfmgdInner - 1) / sfmgdInner;
    uint32_t sfmgdTailTmp = fBaseParams.dV % sfmgdInner;
    uint32_t sfmgdTail = sfmgdTailTmp == 0 ? sfmgdInner : sfmgdTailTmp;
    fBaseParams.sfmgdOuter = sfmgdOuter;
    fBaseParams.sfmgdInner = sfmgdInner;
    fBaseParams.sfmgdTail = sfmgdTail;

    fBaseParams.s1TileNum = (fBaseParams.s1 + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
    fBaseParams.s2TileNum = (fBaseParams.s2 + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
    // 如果batch 和s2 切分已经打满多核，gs1可以不切分
    uint64_t gs1TileNum = 1;
    // hyperparameter seeting threahold = 6, todo: prof 优化后调整阈值计算逻辑
    uint64_t splitBlockNums = 0;
    for (size_t i = 0; i < fBaseParams.b; i++) {
        auto curSeqKv = fBaseParams.actualSeqKvlen[i];
        splitBlockNums += (curSeqKv + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner * fBaseParams.n2;
    }

    if (fBaseParams.b >= 6 || splitBlockNums >= fBaseParams.coreNum) {
        gs1TileNum = 2;
    } else {
        gs1TileNum = fBaseParams.coreNum;
    }

    uint64_t dqWorkspaceSize = (static_cast<uint64_t>(fBaseParams.t1) * (fBaseParams.n2 * fBaseParams.g) *
                                    fBaseParams.d * fBaseParams.s2TileNum * sizeof(float) +
                                GM_ALIGN) /
                               GM_ALIGN * GM_ALIGN;
    uint64_t dkWorkspaceSize =
        (static_cast<uint64_t>(fBaseParams.t2) * fBaseParams.n2 * fBaseParams.d * gs1TileNum * sizeof(float) +
         GM_ALIGN) /
        GM_ALIGN * GM_ALIGN;
    uint64_t dvWorkspaceSize =
        (static_cast<uint64_t>(fBaseParams.t2) * fBaseParams.n2 * fBaseParams.dV * gs1TileNum * sizeof(float) +
         GM_ALIGN) /
        GM_ALIGN * GM_ALIGN;
    fBaseParams.dqkvWorkspaceSize = dqWorkspaceSize + dkWorkspaceSize + dvWorkspaceSize;

    if (fBaseParams.dqkvWorkspaceSize > WORKSPACE_LIMIT) {
        gs1TileNum = 1;
        fBaseParams.saveToWorkspace = 0;
        OP_LOGD(opName, "DoSplit, Cal dq, dk&dv seprately, do not save tensor.");
    } else {
        fBaseParams.saveToWorkspace = 1;
        OP_LOGD(opName, "DoSplit, Split all group & s1 & s2, save all intermediate result.");
    }
    fBaseParams.gs1TileNum = static_cast<uint32_t>(gs1TileNum);
    fBaseParams.dqkvWorkspaceSize = dqWorkspaceSize + dkWorkspaceSize + dvWorkspaceSize;

    for (uint32_t i = 0; i < fBaseParams.b; i++) {
        uint32_t actualS1Len = fBaseParams.actualSeqQlen[i];
        fBaseParams.totalS1TileNums += (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
    }
    OP_LOGD(opName, "DoSplit, totalS1TileNums IS %d.", fBaseParams.totalS1TileNums);
}

void FlashAttentionGradTilingSBn2gs1s2::DoSparse()
{
    if (fBaseParams.layoutType == INPUT_FROAMT_TND) {
        GetSparseSplitS2BlockInfo();
        if (!fBaseParams.saveToWorkspace) {
            GetSparseSplitS1BlockInfo();
        }
    } else {
        OP_LOGE("DoSparse", "Unsupport layout in DoSparse, only support TND layout now.");
    }
}

ge::graphStatus FlashAttentionGradTilingSBn2gs1s2::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    size_t workspaceSize = MUL_CORE_SYNC_BUFFER;

    // matmal1/matmal2 workspace size
    size_t vectorCoreNum = fBaseParams.coreNum;
    workspaceSize = (workspaceSize +
                     vectorCoreNum * fBaseParams.s1CvRatio * fBaseParams.s2CvRatio * fBaseParams.baseMN * FP32_BYTES *
                         MATMAL_INPUT_NUMS +
                     GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;
    // CV ratio workspace size fp16
    // tmp workspace size
    workspaceSize = (workspaceSize +
                     vectorCoreNum * fBaseParams.s1CvRatio * fBaseParams.s2CvRatio * fBaseParams.baseMN * FP16_BYTES *
                         2 // 2 means pingpong
                     + GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;
    // mul workspace size
    workspaceSize = (workspaceSize +
                     vectorCoreNum * fBaseParams.s1CvRatio * fBaseParams.s2CvRatio * fBaseParams.baseMN * FP16_BYTES *
                         2 // 2 means pingpong
                     + GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;

    // dq workspace size
    fBaseParams.dqkvWorkspaceOffset = workspaceSize;
    if (fBaseParams.saveToWorkspace == 1) {
        workspaceSize = workspaceSize + fBaseParams.dqkvWorkspaceSize;
    } else {
        // matmal3 q
        workspaceSize =
            (workspaceSize + static_cast<uint64_t>(fBaseParams.qSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
        // matmal3 k
        workspaceSize =
            (workspaceSize + static_cast<uint64_t>(fBaseParams.kSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
        // matmal3 v
        workspaceSize =
            (workspaceSize + static_cast<uint64_t>(fBaseParams.vSize) * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }
    // dsik workspace
    workspaceSize =
        (workspaceSize + fBaseParams.n1 * fBaseParams.totalS1TileNums * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    workspaceSize = workspaceSize + WORKSPACE_BUFFER;
    workspaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionGradTilingSBn2gs1s2::SaveToTilingData()
{
    // 先调用基类的SaveToTilingData
    FlashAttentionScoreGradTilingUs1s2Bs2::SaveToTilingData();

    // 然后添加派生类特有的数据保存逻辑
    tilingData.s1s2BNGS1S2BlockNumList.set_splitS1BlockStarts(fBaseParams.splitS1BlockStarts);
    tilingData.s1s2BNGS1S2BlockNumList.set_splitS2BlockStarts(fBaseParams.splitS2BlockStarts);
    tilingData.s1s2BNGS1S2BlockNumList.set_splitS1BlockEnds(fBaseParams.splitS1BlockEnds);
    tilingData.s1s2BNGS1S2BlockNumList.set_splitS2BlockEnds(fBaseParams.splitS2BlockEnds);
    tilingData.s1s2BNGS1S2BlockNumList.set_gs1Offset(fBaseParams.gs1Offset);

    // dq save workspace infos
    tilingData.s1s2BNGS1S2BaseParams.set_Tq(fBaseParams.t1);
    tilingData.s1s2BNGS1S2BaseParams.set_Tkv(fBaseParams.t2);
    tilingData.s1s2BNGS1S2BaseParams.set_s2TileNum(fBaseParams.s1TileNum);
    tilingData.s1s2BNGS1S2BaseParams.set_s2TileNum(fBaseParams.s2TileNum);
    tilingData.s1s2BNGS1S2BaseParams.set_gs1TileNum(fBaseParams.gs1TileNum);
    tilingData.s1s2BNGS1S2BaseParams.set_dqkvWorkspaceOffset(fBaseParams.dqkvWorkspaceOffset);
    tilingData.s1s2BNGS1S2BaseParams.set_dqkvWorkspaceSize(fBaseParams.dqkvWorkspaceSize);
    tilingData.s1s2BNGS1S2BaseParams.set_saveToWorkspace(fBaseParams.saveToWorkspace);

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionGradTilingSBn2gs1s2::FillSplitS2BlockInfo(
    std::vector<std::vector<std::vector<uint32_t>>>& calculatedBlockInfo,
    std::vector<std::vector<uint32_t>>& totalBlockInfo)
{
    for (uint32_t i = 0; i < fBaseParams.b; i++) {
        uint32_t actualS1Len = fBaseParams.actualSeqQlen[i];
        uint32_t actualS2Len = fBaseParams.actualSeqKvlen[i];
        int64_t actualCalcS1Token = fBaseParams.s1Token;
        int64_t actualCalcS2Token = fBaseParams.s2Token;
        if (fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL || fBaseParams.sparseMode == BAND) {
            actualCalcS1Token = fBaseParams.s1Token + actualS1Len - actualS2Len;
            actualCalcS2Token = fBaseParams.s2Token - actualS1Len + actualS2Len;
        }

        auto actualS1Outer = (actualS1Len + fBaseParams.s1Inner - 1) / fBaseParams.s1Inner;
        auto actualS2Outer = (actualS2Len + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;

        calculatedBlockInfo[i][0][BEGIN_IDX] = 0; // BEGIN_IDX 行的有效起始位置
        calculatedBlockInfo[i][0][END_IDX] =      // END_IDX 行的有效终止位置
            uint32_t(std::min(int64_t(actualS1Len), std::max(0L, fBaseParams.cvS2Inner + actualCalcS1Token)) +
                     fBaseParams.s1Inner - 1) /
            fBaseParams.s1Inner;
        calculatedBlockInfo[i][0][SUM_S1S2] = calculatedBlockInfo[i][0][END_IDX];
        // unpad 场景下s2Outer是按照最大的s2计算得到的
        // 这里是s1 不做切分，因此在s2 列方向上做loop，判断落在哪一个loop 内，进而确定 s1 的起始和终止位置
        for (uint32_t j = 1; j < fBaseParams.s2Outer; j++) {
            // 这里的 s2Outer 是根据最大seq 计算的，因此要加上check，超过了有效句长要置0
            if (fBaseParams.cvS2Inner * j >= actualS2Len) {
                calculatedBlockInfo[i][j][BEGIN_IDX] = 0;
                calculatedBlockInfo[i][j][END_IDX] = 0;
            } else {
                calculatedBlockInfo[i][j][BEGIN_IDX] =
                    uint32_t(std::min(std::max(int64_t(fBaseParams.cvS2Inner * j) - actualCalcS2Token, 0L),
                                      int64_t(actualS1Len))) /
                    fBaseParams.s1Inner;
                calculatedBlockInfo[i][j][END_IDX] =
                    uint32_t(std::min(int64_t(actualS1Len),
                                      std::max(int64_t(fBaseParams.cvS2Inner * (j + 1)) + actualCalcS1Token, 0L)) +
                             fBaseParams.s1Inner - 1) /
                    fBaseParams.s1Inner;
            }
            calculatedBlockInfo[i][j][SUM_S1S2] = calculatedBlockInfo[i][j - 1][SUM_S1S2] +
                                                  calculatedBlockInfo[i][j][END_IDX] -
                                                  calculatedBlockInfo[i][j][BEGIN_IDX];
            calculatedBlockInfo[i][j][SUM_ALL] = 0; // 初始化清零
        }
        if (i == 0) {
            calculatedBlockInfo[0][0][SUM_ALL] = // SUM_ALL 为从0到当前batch的所有有效块总和
                fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[0][fBaseParams.s2Outer - 1][SUM_S1S2];
            totalBlockInfo[0][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[0][0];
        } else {
            calculatedBlockInfo[i][0][SUM_ALL] =
                fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[i][fBaseParams.s2Outer - 1][SUM_S1S2] +
                calculatedBlockInfo[i - 1][0][SUM_ALL];
            totalBlockInfo[i][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
    }
}

ge::graphStatus FlashAttentionGradTilingSBn2gs1s2::GetSparseSplitS2BlockInfo()
{
    std::vector<std::vector<std::vector<uint32_t>>> calculatedBlockInfo(
        fBaseParams.b,
        std::vector<std::vector<uint32_t>>(fBaseParams.s2Outer, std::vector<uint32_t>(CALCULATED_BLOCK_DIMENSION)));
    std::vector<std::vector<uint32_t>> totalBlockInfo(fBaseParams.b, std::vector<uint32_t>(TOTAL_BLOCK_DIMENSION));
    FillSplitS2BlockInfo(calculatedBlockInfo, totalBlockInfo);
    // block split
    uint32_t fusedOuter = calculatedBlockInfo[fBaseParams.b - 1][0][SUM_ALL];
    uint32_t blockOuter = fBaseParams.coreNum;
    uint32_t blockFactor = (fusedOuter + blockOuter - 1) / blockOuter;

    OP_LOGD(opName, "GetSparseSplitS2BlockInfo, fusedOuter = %u: blockFactor = %u, blockOuter = %u", fusedOuter,
            blockFactor, blockOuter);
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;

    std::vector<uint32_t> blockList;
    std::vector<uint32_t> batchIdxList;
    // 2 means cut half on gs1 save to workspace
    // 1 meas no cut on gs1, do not save
    for (uint32_t bidx = 0; bidx < fBaseParams.b; bidx++) {
        for (uint32_t nidx = 0; nidx < fBaseParams.n2; nidx++) {
            for (uint32_t s2idx = 0; s2idx < fBaseParams.s2Outer; s2idx++) {
                uint32_t need_process_gs1 =
                    (calculatedBlockInfo[bidx][s2idx][END_IDX] - calculatedBlockInfo[bidx][s2idx][BEGIN_IDX]) *
                    fBaseParams.g;
                if (need_process_gs1 > 0) {
                    if (fBaseParams.gs1TileNum == 2 || fBaseParams.gs1TileNum == 1) {
                        uint32_t first_half = need_process_gs1 / fBaseParams.gs1TileNum;
                        uint32_t second_half = need_process_gs1 - first_half;
                        blockList.push_back(first_half);
                        blockList.push_back(second_half);
                        batchIdxList.push_back(bidx);
                        batchIdxList.push_back(bidx);
                    } else {
                        blockList.push_back(need_process_gs1);
                        batchIdxList.push_back(bidx);
                    }
                }
            }
        }
    }

    // 按照 blockFactor 划分后，每个位置对应的一个或多个tile 块
    // blockCluster {{...}, {...}, {...}}
    std::vector<std::vector<uint32_t>> blockCluster;
    std::vector<std::vector<uint32_t>> bIdxCluster;
    auto splitFactor =
        (fBaseParams.gs1TileNum == 2) ? fBaseParams.blockFactor : std::max(fusedOuter / blockList.size(), size_t(1));

    uint32_t current_sum = 0;
    std::vector<uint32_t> current_list;
    std::vector<uint32_t> bidx_list;
    for (size_t idx = 0; idx < blockList.size(); idx++) {
        // 判断当前分块是否需要添加
        if (std::abs(static_cast<int32_t>(current_sum + blockList[idx] - splitFactor)) <
            std::abs(static_cast<int32_t>(current_sum - splitFactor))) {
            current_sum += blockList[idx];
            current_list.push_back(blockList[idx]);
            bidx_list.push_back(batchIdxList[idx]);
        } else {
            if (current_list.size() > 0) {
                blockCluster.push_back(current_list);
                bIdxCluster.push_back(bidx_list);
            }
            current_sum = blockList[idx];
            current_list = {blockList[idx]};
            bidx_list = {batchIdxList[idx]};
        }
    }
    if (current_list.size() > 0) {
        blockCluster.push_back(current_list);
        bIdxCluster.push_back(bidx_list);
    }

    // 不保存中间结果到workspace，计算两次
    // gs1TileNum == 2, SplitS1Naive 不会再切分blockCluster
    if (!fBaseParams.saveToWorkspace) {
        SplitS1Naive(calculatedBlockInfo, totalBlockInfo, blockCluster, bIdxCluster);
    } else {
        // 保存中间结果
        if (fBaseParams.gs1TileNum == 2) {
            SplitS1Naive(calculatedBlockInfo, totalBlockInfo, blockCluster, bIdxCluster);
        } else {
            SplitS1Balanced(calculatedBlockInfo, totalBlockInfo, blockCluster, bIdxCluster);
        }
    }

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionGradTilingSBn2gs1s2::FillSplitS1BlockInfo(
    std::vector<std::vector<std::vector<uint32_t>>>& calculatedBlockInfo,
    std::vector<std::vector<uint32_t>>& totalBlockInfo)
{
    for (uint32_t i = 0; i < fBaseParams.b; i++) {
        uint32_t actualS1Len = fBaseParams.actualSeqQlen[i];
        uint32_t actualS2Len = fBaseParams.actualSeqKvlen[i];
        int64_t actualCalcS1Token = fBaseParams.s1Token;
        int64_t actualCalcS2Token = fBaseParams.s2Token;
        if (fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL || fBaseParams.sparseMode == BAND) {
            actualCalcS1Token = fBaseParams.s1Token + actualS1Len - actualS2Len;
            actualCalcS2Token = fBaseParams.s2Token - actualS1Len + actualS2Len;
        }

        auto actualS1Outer = (actualS1Len + fBaseParams.s1Inner - 1) / fBaseParams.s1Inner;
        auto actualS2Outer = (actualS2Len + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;

        // loop on s2 direction
        calculatedBlockInfo[i][0][BEGIN_IDX] = 0; // BEGIN_IDX 列的有效起始位置
        calculatedBlockInfo[i][0][END_IDX] =      // END_IDX 列的有效终止位置
            uint32_t(std::min(int64_t(actualS2Len), std::max(0L, fBaseParams.s1Inner + actualCalcS2Token)) +
                     fBaseParams.cvS2Inner - 1) /
            fBaseParams.cvS2Inner;
        calculatedBlockInfo[i][0][SUM_S1S2] =
            calculatedBlockInfo[i][0][END_IDX]; // SUM_S1S2 当前batch上， 列的有效块总和

        // unpad 场景下s1Outer是按照最大的s1计算得到的
        for (uint32_t j = 1; j < fBaseParams.s1Outer; j++) {
            if (fBaseParams.s1Inner * j >= actualS1Len) {
                calculatedBlockInfo[i][j][BEGIN_IDX] = 0;
                calculatedBlockInfo[i][j][END_IDX] = 0;
            } else {
                calculatedBlockInfo[i][j][BEGIN_IDX] =
                    uint32_t(std::min(std::max(int64_t(fBaseParams.s1Inner * j) - actualCalcS1Token, 0L),
                                      int64_t(actualS2Len))) /
                    fBaseParams.cvS2Inner;
                calculatedBlockInfo[i][j][END_IDX] =
                    uint32_t(std::min(int64_t(actualS2Len),
                                      std::max(int64_t(fBaseParams.s1Inner * (j + 1)) + actualCalcS2Token, 0L)) +
                             fBaseParams.cvS2Inner - 1) /
                    fBaseParams.cvS2Inner;
            }
            calculatedBlockInfo[i][j][SUM_S1S2] = calculatedBlockInfo[i][j - 1][SUM_S1S2] +
                                                  calculatedBlockInfo[i][j][END_IDX] -
                                                  calculatedBlockInfo[i][j][BEGIN_IDX];
            calculatedBlockInfo[i][j][SUM_ALL] = 0; // 初始化清零
        }
        if (i == 0) {
            calculatedBlockInfo[0][0][SUM_ALL] = // SUM_ALL 为从0到当前batch的所有游侠块总和
                fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[0][fBaseParams.s1Outer - 1][SUM_S1S2];
            totalBlockInfo[0][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[0][0];
        } else {
            calculatedBlockInfo[i][0][SUM_ALL] =
                fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[i][fBaseParams.s1Outer - 1][SUM_S1S2] +
                calculatedBlockInfo[i - 1][0][SUM_ALL];
            totalBlockInfo[i][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
    }
}

void FlashAttentionGradTilingSBn2gs1s2::SplitS1Naive(
    std::vector<std::vector<std::vector<uint32_t>>> calculatedBlockInfo,
    std::vector<std::vector<uint32_t>> totalBlockInfo, std::vector<std::vector<uint32_t>> blockCluster,
    std::vector<std::vector<uint32_t>> bIdxCluster)
{
    // totalBlockInfo[b][0] 表示当前batch 处理的实际 s1o * s2o
    // totalBlockInfo[b][1] 表示当前batch 处理的实际 n * g * s1o * s2o (这里对应原始输入，未经过attmask掩掉)
    // blockEnds[blockOuter - 1] = totalBlockInfo[fBaseParams.b - 1][1];
    // calculatedBlockInfo SUM_ALL 带了N2 G 维度

    uint32_t bIdx = 0;
    uint32_t bTail = 0;
    uint32_t n2Idx = 0;
    uint32_t n2Tail = 0;
    uint32_t gIdx = 0;
    uint32_t gTail = 0;
    int64_t s1oIdx = 0;
    int64_t s1oTail = 0;
    int64_t s2oIdx = 0;
    int64_t s2oTail = 0;
    int64_t s1OuterTmp = 0;

    std::vector<int64_t> blockStarts(CORE_LIST_NUM, 0);
    std::vector<int64_t> blockEnds(CORE_LIST_NUM, totalBlockInfo[fBaseParams.b - 1][1]);

    // 如果结果列表的长度大于所需的子列表数量，合并相邻的子列表
    while (blockCluster.size() > fBaseParams.blockOuter) {
        uint32_t min_diff = std::numeric_limits<uint32_t>::max();
        uint32_t merge_index = -1;
        for (size_t i = 0; i < blockCluster.size() - 1; ++i) {
            uint32_t merged_cluster = std::accumulate(blockCluster[i].begin(), blockCluster[i].end(), 0) +
                                      std::accumulate(blockCluster[i + 1].begin(), blockCluster[i + 1].end(), 0);
            uint32_t diff =
                std::abs(static_cast<int32_t>(merged_cluster) - static_cast<int32_t>(fBaseParams.blockFactor));
            if (diff < min_diff) {
                min_diff = diff;
                merge_index = i;
            }
        }
        std::vector<uint32_t> merged_list = blockCluster[merge_index];
        merged_list.insert(merged_list.end(), blockCluster[merge_index + 1].begin(),
                           blockCluster[merge_index + 1].end());
        blockCluster[merge_index] = merged_list;
        blockCluster.erase(blockCluster.begin() + merge_index + 1);

        merged_list = bIdxCluster[merge_index];
        merged_list.insert(merged_list.end(), bIdxCluster[merge_index + 1].begin(), bIdxCluster[merge_index + 1].end());
        bIdxCluster[merge_index] = merged_list;
        bIdxCluster.erase(bIdxCluster.begin() + merge_index + 1);
    }

    // for (size_t i = 0; i < blockCluster.size(); ++i) {
    //   std::cout << " after merge BlockCluster " << i + 1 << ": ";
    //   for (const auto& elem : blockCluster[i]) {
    //       std::cout << elem << " ";
    //   }
    //   std::cout << std::endl;
    // }

    uint32_t currentIdx = 0;
    for (size_t c_idx = 0; c_idx < blockCluster.size(); c_idx++) {
        // 计算当前 cluster 对应的总tile 数和对应处理的batchId
        currentIdx += std::accumulate(blockCluster[c_idx].begin(), blockCluster[c_idx].end(), 0);
        bIdx = bIdxCluster[c_idx][bIdxCluster[c_idx].size() - 1];
        auto s1os2o = calculatedBlockInfo[bIdx][fBaseParams.s2Outer - 1][SUM_S1S2];
        auto gs1os2o = s1os2o * fBaseParams.g;
        bTail = (bIdx == 0) ? currentIdx : currentIdx - calculatedBlockInfo[bIdx - 1][0][SUM_ALL];
        n2Idx = bTail / gs1os2o;
        n2Tail = bTail % gs1os2o;

        // 确认当前计算块落在第几列上
        s1OuterTmp = (fBaseParams.actualSeqQlen[bIdx] + fBaseParams.s1Inner - 1) / fBaseParams.s1Inner;
        for (uint32_t i = 0; i < fBaseParams.s2Outer; i++) {
            if (calculatedBlockInfo[bIdx][i][SUM_S1S2] * fBaseParams.g >= n2Tail) {
                s2oIdx = i;
                s2oTail = (i == 0) ? n2Tail : n2Tail - calculatedBlockInfo[bIdx][i - 1][SUM_S1S2] * fBaseParams.g;

                uint32_t calculatedS1 =
                    (i == 0) ? calculatedBlockInfo[bIdx][i][SUM_S1S2] :
                               calculatedBlockInfo[bIdx][i][SUM_S1S2] - calculatedBlockInfo[bIdx][i - 1][SUM_S1S2];
                if (calculatedS1 == 0) {
                    // 该 s2 块被滑窗完全裁剪，跳过
                    continue;
                }
                gIdx = s2oTail / calculatedS1;
                if (gIdx != fBaseParams.g)
                    s1oIdx = (s2oTail % calculatedS1) + calculatedBlockInfo[bIdx][i][BEGIN_IDX];
                else
                    s1oIdx = 0;
                break;
            }
        }

        if (bIdx == 0) {
            blockStarts[c_idx + 1] = n2Idx * totalBlockInfo[bIdx][0] * fBaseParams.g +
                                     s2oIdx * s1OuterTmp * fBaseParams.g + gIdx * s1OuterTmp + s1oIdx;
        } else {
            blockStarts[c_idx + 1] = totalBlockInfo[bIdx - 1][1] + n2Idx * totalBlockInfo[bIdx][0] * fBaseParams.g +
                                     s2oIdx * s1OuterTmp * fBaseParams.g + gIdx * s1OuterTmp + s1oIdx;
        }
        blockEnds[c_idx] = blockStarts[c_idx + 1];
    }
    for (uint32_t c = blockCluster.size() + 1; c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.splitS2BlockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.splitS2BlockEnds));
}

void FlashAttentionGradTilingSBn2gs1s2::SplitS1Balanced(
    std::vector<std::vector<std::vector<uint32_t>>> calculatedBlockInfo,
    std::vector<std::vector<uint32_t>> totalBlockInfo, std::vector<std::vector<uint32_t>> blockCluster,
    std::vector<std::vector<uint32_t>> bIdxCluster)
{
    uint32_t bIdx = 0;
    uint32_t bTail = 0;
    uint32_t n2Idx = 0;
    uint32_t n2Tail = 0;
    uint32_t gIdx = 0;
    uint32_t gTail = 0;
    int64_t s1oIdx = 0;
    int64_t s1oTail = 0;
    int64_t s2oIdx = 0;
    int64_t s2oTail = 0;
    int64_t s1OuterTmp = 0;

    std::vector<int64_t> blockStarts(CORE_LIST_NUM, 0);
    std::vector<int64_t> blockEnds(CORE_LIST_NUM, totalBlockInfo[fBaseParams.b - 1][1]);
    std::vector<int64_t> gs1Offset(CORE_LIST_NUM, 0);
    uint32_t totalSize = 0;
    for (size_t bidx = 0; bidx < fBaseParams.b; ++bidx) {
        totalSize += calculatedBlockInfo[bidx][fBaseParams.s2Outer - 1][SUM_S1S2] * fBaseParams.g * fBaseParams.n2;
    }

    std::vector<double> doubleAlloc;
    std::vector<uint32_t> coreSplits(blockCluster.size(), 0);
    std::vector<double> remainders(blockCluster.size(), 0.0);
    for (size_t idx = 0; idx < blockCluster.size(); idx++) {
        uint32_t curBlockSize = std::accumulate(blockCluster[idx].begin(), blockCluster[idx].end(), 0);
        double splitCoreNums = static_cast<double>(fBaseParams.coreNum) * curBlockSize / totalSize;
        // 当前block 太小了，可能分不到1个核，要手动置1 确保分到核
        if (splitCoreNums < 1.0) {
            coreSplits[idx] = 1;
            remainders[idx] = splitCoreNums - 1;
        } else {
            coreSplits[idx] = static_cast<uint32_t>(splitCoreNums);
            remainders[idx] = (splitCoreNums - static_cast<uint32_t>(splitCoreNums));
        }
    }

    int remaining = fBaseParams.coreNum - std::accumulate(coreSplits.begin(), coreSplits.end(), 0);
    // 按余数大小分配（相同余数保持原始顺序）
    std::vector<size_t> sorted_indices(bIdxCluster.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t i, size_t j) {
        return (remainders[i] > remainders[j]) || (remainders[i] == remainders[j] && i < j);
    });

    // 分配剩余数值
    if (remaining < 0) {
        // 分配多了从其他地方减下来，前提是当前核分的核数 > 1，并且倒序减
        remaining = std::abs(remaining);
        for (size_t i = 0; i < remaining; ++i) {
            for (size_t j = sorted_indices.size() - 1; j >= 0; j--) {
                if (coreSplits[sorted_indices[j]] > 1) {
                    coreSplits[sorted_indices[j]] -= 1;
                    break;
                }
            }
        }
    } else {
        for (size_t i = 0; i < remaining; ++i) {
            coreSplits[sorted_indices[i]] += 1;
        }
    }

    uint32_t split = 0;
    uint32_t remain = 0;
    uint32_t accmulatedCoreNum = 0;
    std::vector<uint32_t> blockProcessSize(fBaseParams.coreNum, 0);

    for (size_t idx = 0; idx < blockCluster.size(); idx++) {
        uint32_t curBlockSize = std::accumulate(blockCluster[idx].begin(), blockCluster[idx].end(), 0);
        uint32_t start = (idx == 0) ? 0 : accmulatedCoreNum;
        uint32_t end = (idx == 0) ? coreSplits[idx] : accmulatedCoreNum + coreSplits[idx];
        if (curBlockSize >= coreSplits[idx]) {
            split = curBlockSize / coreSplits[idx];
            remain = curBlockSize % coreSplits[idx];
            for (size_t coreIdx = start; coreIdx < end; coreIdx++) {
                if (coreIdx - start < remain)
                    blockProcessSize[coreIdx] = split + 1;
                else
                    blockProcessSize[coreIdx] = split;
                gs1Offset[coreIdx] = coreIdx - start;
            }
            accmulatedCoreNum += coreSplits[idx];
        } else {
            for (size_t coreIdx = start; coreIdx < start + curBlockSize; coreIdx++) {
                blockProcessSize[coreIdx] = 1;
                gs1Offset[coreIdx] = coreIdx - start;
            }
            accmulatedCoreNum += curBlockSize;
        }
    }

    uint32_t accmulatedCalSize = 0;
    for (size_t blockIdx = 0; blockIdx < blockProcessSize.size(); blockIdx++) {
        accmulatedCalSize += blockProcessSize[blockIdx];
        for (size_t bidx = 0; bidx < fBaseParams.b; bidx++) {
            if (calculatedBlockInfo[bidx][0][SUM_ALL] >= accmulatedCalSize) {
                bIdx = bidx;
                break;
            }
        }
        bTail = (bIdx == 0) ? accmulatedCalSize : accmulatedCalSize - calculatedBlockInfo[bIdx - 1][0][SUM_ALL];
        auto s1os2o = calculatedBlockInfo[bIdx][fBaseParams.s2Outer - 1][SUM_S1S2];
        auto gs1os2o = s1os2o * fBaseParams.g;
        n2Idx = bTail / gs1os2o;
        n2Tail = bTail % gs1os2o;

        s1OuterTmp = (fBaseParams.actualSeqQlen[bIdx] + fBaseParams.s1Inner - 1) / fBaseParams.s1Inner;
        for (uint32_t i = 0; i < fBaseParams.s2Outer; i++) {
            if (calculatedBlockInfo[bIdx][i][SUM_S1S2] * fBaseParams.g >= n2Tail) {
                s2oIdx = i;
                s2oTail = (i == 0) ? n2Tail : n2Tail - calculatedBlockInfo[bIdx][i - 1][SUM_S1S2] * fBaseParams.g;
                uint32_t calculatedS1 =
                    (i == 0) ? calculatedBlockInfo[bIdx][i][SUM_S1S2] :
                               calculatedBlockInfo[bIdx][i][SUM_S1S2] - calculatedBlockInfo[bIdx][i - 1][SUM_S1S2];
                if (calculatedS1 == 0) {
                    // 该 s2 块被滑窗完全裁剪，跳过
                    continue;
                }
                gIdx = s2oTail / calculatedS1;
                if (gIdx != fBaseParams.g)
                    s1oIdx = (s2oTail % calculatedS1) + calculatedBlockInfo[bIdx][i][BEGIN_IDX];
                else
                    s1oIdx = 0;
                break;
            }
        }
        if (bIdx == 0) {
            blockStarts[blockIdx + 1] = n2Idx * totalBlockInfo[bIdx][0] * fBaseParams.g +
                                        s2oIdx * s1OuterTmp * fBaseParams.g + gIdx * s1OuterTmp + s1oIdx;
        } else {
            blockStarts[blockIdx + 1] = totalBlockInfo[bIdx - 1][1] + n2Idx * totalBlockInfo[bIdx][0] * fBaseParams.g +
                                        s2oIdx * s1OuterTmp * fBaseParams.g + gIdx * s1OuterTmp + s1oIdx;
        }
        blockEnds[blockIdx] = blockStarts[blockIdx + 1];
    }

    for (uint32_t c = accmulatedCoreNum + 1; c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.splitS2BlockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.splitS2BlockEnds));
    std::copy(std::begin(gs1Offset), std::end(gs1Offset), std::begin(fBaseParams.gs1Offset));
}

ge::graphStatus FlashAttentionGradTilingSBn2gs1s2::GetSparseSplitS1BlockInfo()
{
    std::vector<std::vector<std::vector<uint32_t>>> calculatedBlockInfo(
        fBaseParams.b,
        std::vector<std::vector<uint32_t>>(fBaseParams.s1Outer, std::vector<uint32_t>(CALCULATED_BLOCK_DIMENSION)));
    std::vector<std::vector<uint32_t>> totalBlockInfo(fBaseParams.b, std::vector<uint32_t>(TOTAL_BLOCK_DIMENSION));
    FillSplitS1BlockInfo(calculatedBlockInfo, totalBlockInfo);
    // block split
    uint32_t fusedOuter = calculatedBlockInfo[fBaseParams.b - 1][0][SUM_ALL];
    uint32_t blockOuter = fBaseParams.coreNum;
    uint32_t blockFactor = (fusedOuter + blockOuter - 1) / blockOuter;

    OP_LOGD(opName, "GetSparseSplitS1BlockInfo, fusedOuter = %u: blockFactor = %u, blockOuter = %u", fusedOuter,
            blockFactor, blockOuter);
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;

    uint32_t bIdx = 0;
    uint32_t bTail = 0;
    uint32_t n2Idx = 0;
    uint32_t n2Tail = 0;
    uint32_t gIdx = 0;
    uint32_t gTail = 0;
    int64_t s1oIdx = 0;
    int64_t s2oIdx = 0;
    int64_t s2oTail = 0;

    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    blockStarts[0] = 0;
    blockEnds[blockOuter - 1] = totalBlockInfo[fBaseParams.b - 1][1];
    uint32_t s2OuterTmp = 0;

    for (uint32_t c = 1; c < blockOuter; c++) {
        uint32_t currentIdx = std::min(c * blockFactor, fusedOuter);

        for (uint32_t b = 0; b < fBaseParams.b; b++) {
            if (calculatedBlockInfo[b][0][SUM_ALL] >= currentIdx) {
                bIdx = b;
                auto s1os2o = calculatedBlockInfo[b][fBaseParams.s1Outer - 1][SUM_S1S2];
                auto gs1os2o = s1os2o * fBaseParams.g;
                bTail = (b == 0) ? currentIdx : currentIdx - calculatedBlockInfo[b - 1][0][SUM_ALL];
                n2Idx = bTail / gs1os2o;
                n2Tail = bTail % gs1os2o;
                gIdx = n2Tail / s1os2o;
                gTail = n2Tail % s1os2o;

                for (uint32_t i = 0; i < fBaseParams.s1Outer; i++) {
                    if (calculatedBlockInfo[b][i][SUM_S1S2] >= gTail) {
                        s1oIdx = i;
                        s2oTail = (i == 0) ? gTail : gTail - calculatedBlockInfo[b][i - 1][SUM_S1S2];
                        s2oIdx = calculatedBlockInfo[b][i][BEGIN_IDX] + s2oTail;
                        break;
                    }
                }
                uint32_t remain_s2o =
                    calculatedBlockInfo[b][s1oIdx][END_IDX] - calculatedBlockInfo[b][s1oIdx][BEGIN_IDX] - s2oTail;
                if (s2oTail > remain_s2o)
                    s1oIdx += 1;
                s2OuterTmp = (fBaseParams.actualSeqKvlen[b] + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
                break;
            }
        }
        if (bIdx == 0) {
            blockStarts[c] = (n2Idx * fBaseParams.g + gIdx) * totalBlockInfo[bIdx][0] + s1oIdx * s2OuterTmp;
        } else {
            blockStarts[c] = totalBlockInfo[bIdx - 1][1] + (n2Idx * fBaseParams.g + gIdx) * totalBlockInfo[bIdx][0] +
                             s1oIdx * s2OuterTmp;
        }
        blockEnds[c - 1] = blockStarts[c];
    }
    for (uint32_t c = blockOuter; c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }

    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.splitS1BlockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.splitS1BlockEnds));

    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("FlashAttentionGrad", FlashAttentionGradTilingSBn2gs1s2, 1);

} // namespace optiling
