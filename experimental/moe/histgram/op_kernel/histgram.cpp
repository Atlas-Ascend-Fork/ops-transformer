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
 * \file histgram.cpp
 * \brief Histgram kernel 入口
 */

#include "kernel_operator.h"
#include "histgram_tiling_def.h"
#include "histgram_impl.hpp"

extern "C" __global__ __aicore__ void histgram(GM_ADDR input, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(optiling::HistgramTilingData, tilingDataIn, tiling);
    const optiling::HistgramTilingData* __restrict__ tilingData = &tilingDataIn;

    SetSysWorkspace(workspace);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    TPipe pipe;
    histgram_ns::Histogram histogram(*tilingData);
    histogram.Process(input, out, user, pipe);
}
