/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MOE_DISTRIBUTE_COMBINE_V2_A5_ENTRY_H
#define MOE_DISTRIBUTE_COMBINE_V2_A5_ENTRY_H

// 说明：本头文件同时被 host 侧(moe_distribute_combine_v2_torch.cpp, g++)与 device 侧
// (moe_distribute_combine_v2_entry.cpp, bisheng)包含，仅可出现 host/device 公共声明，
// 不可包含 device 专用代码。包含方需先包含 devkit 头文件以提供 GM_ADDR 定义。
// 注意：参数列表与 mc2 arch35 的 moe_distribute_combine_v2 模板 kernel 一致
// （residualX/gamma 在 apt.cpp 内固定为 nullptr，不作为入口参数）。

struct MoeDistributeCombineV2SkArgs {
    int32_t tilingKey;
    GM_ADDR expandX;
    GM_ADDR expertIds;
    GM_ADDR assistInfoForCombine;
    GM_ADDR epSendCount;
    GM_ADDR scales;
    GM_ADDR tpSendCounts;
    GM_ADDR xActiveMask;
    GM_ADDR activationScale;
    GM_ADDR weightScale;
    GM_ADDR groupList;
    GM_ADDR expandScales;
    GM_ADDR sharedExpertX;
    GM_ADDR elasticInfo;
    GM_ADDR oriX;
    GM_ADDR constExpertAlpha1;
    GM_ADDR constExpertAlpha2;
    GM_ADDR constExpertV;
    GM_ADDR performanceInfo;
    GM_ADDR XOut;
    GM_ADDR workspaceGM;
    GM_ADDR tilingGM;
    GM_ADDR mc2Context;
};

// <<<>>>调用函数声明
void moe_distribute_combine_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream, GM_ADDR expandX,
                                     GM_ADDR expertIds, GM_ADDR assistInfoForCombine, GM_ADDR epSendCount,
                                     GM_ADDR scales, GM_ADDR tpSendCounts, GM_ADDR xActiveMask, GM_ADDR activationScale,
                                     GM_ADDR weightScale, GM_ADDR groupList, GM_ADDR expandScales,
                                     GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX,
                                     GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2, GM_ADDR constExpertV,
                                     GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM,
                                     GM_ADDR mc2Context);

#endif // MOE_DISTRIBUTE_COMBINE_V2_A5_ENTRY_H
