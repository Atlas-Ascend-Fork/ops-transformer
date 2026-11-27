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
 * \file distribute_barrier_extend_apt.cpp
 * \brief APT (arch35) kernel entry for DistributeBarrierExtend.
 */

#include "distribute_barrier_extend.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void distribute_barrier_extend(GM_ADDR mc2Context, GM_ADDR xRef, GM_ADDR timeOut,
                                                                GM_ADDR elasticInfo, GM_ADDR xRefOut,
                                                                GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DistributeBarrierTilingData);
    GET_TILING_DATA_WITH_STRUCT(DistributeBarrierTilingData, tilingData, tilingGM);
    TPipe pipe;
    if (tilingData.distributeBarrierInfo.isAuto) {
        NsDistributeBarrierExtend::DistributeBarrier op;
        op.Init(mc2Context, timeOut, elasticInfo, &pipe, &tilingData);
        op.Process();
    } else {
        DistributeBarrierImpl::DistributeBarrier<DTYPE_X_REF> op;
        op.Init(mc2Context, timeOut, elasticInfo, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}
