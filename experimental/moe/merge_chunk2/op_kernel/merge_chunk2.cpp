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
 * \file merge_chunk2.cpp
 * \brief MergeChunk2 kernel 入口
 */

#include "kernel_operator.h"
#include "merge_chunk2_tiling_def.h"
#include "merge_chunk2_impl.hpp"

extern "C" __global__ __aicore__ void merge_chunk2(GM_ADDR in1c1, GM_ADDR in1c2, GM_ADDR in2c1, GM_ADDR in2c2,
                                                   GM_ADDR glc1, GM_ADDR glc2, GM_ADDR out1, GM_ADDR out2, GM_ADDR gl,
                                                   GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA_WITH_STRUCT(MergeChunk2KernelTilingData, tiling_data, tiling);

    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        ops_merge_chunk2::MergeChunk2<bfloat16_t> mergeChunk2(tiling_data, pipe);
        mergeChunk2.Process(in1c1, in1c2, in2c1, in2c2, glc1, glc2, out1, out2, gl);
    } else if (TILING_KEY_IS(1)) {
        ops_merge_chunk2::MergeChunk2<half> mergeChunk2(tiling_data, pipe);
        mergeChunk2.Process(in1c1, in1c2, in2c1, in2c2, glc1, glc2, out1, out2, gl);
    }
}
