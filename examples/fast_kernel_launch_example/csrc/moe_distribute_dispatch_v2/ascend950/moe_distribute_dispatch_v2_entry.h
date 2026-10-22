/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MOE_DISTRIBUTE_DISPATCH_V2_A5_ENTRY_H
#define MOE_DISTRIBUTE_DISPATCH_V2_A5_ENTRY_H

// 说明：本头文件同时被 host 侧(moe_distribute_dispatch_v2_torch.cpp, g++)与 device 侧
// (moe_distribute_dispatch_v2_entry.cpp, bisheng)包含，仅可出现 host/device 公共声明，
// 不可包含 device 专用代码。包含方需先包含 devkit 头文件以提供 GM_ADDR 定义。

struct MoeDistributeDispatchV2SkArgs {
    int32_t tilingKey;
    GM_ADDR x;
    GM_ADDR expertIds;
    GM_ADDR scales;
    GM_ADDR xActiveMask;
    GM_ADDR expertScales;
    GM_ADDR elasticInfo;
    GM_ADDR performanceInfo;
    GM_ADDR expandXOut;
    GM_ADDR dynamicScalesOut;
    GM_ADDR assistInfoOut;
    GM_ADDR expertTokenNumsOut;
    GM_ADDR epSendCountsOut;
    GM_ADDR tpSendCountsOut;
    GM_ADDR expandScalesOut;
    GM_ADDR workspaceGM;
    GM_ADDR tilingGM;
    GM_ADDR mc2Context;
};

// <<<>>>调用函数声明
void moe_distribute_dispatch_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream, GM_ADDR x, GM_ADDR expertIds,
                                      GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo,
                                      GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                      GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
                                      GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM,
                                      GM_ADDR tilingGM, GM_ADDR mc2Context);

#endif // MOE_DISTRIBUTE_DISPATCH_V2_A5_ENTRY_H
