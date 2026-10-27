/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "score_core.h"
#include "tq_score_compute_tiling_data.h"
extern "C" __global__ __aicore__ void tq_score_compute(GM_ADDR q, GM_ADDR s, GM_ADDR i, GM_ADDR j, GM_ADDR n, GM_ADDR g,
                                                       GM_ADDR t, GM_ADDR l, GM_ADDR packed, GM_ADDR o,
                                                       GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::TqScoreComputeTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    GlobalTensor<uint32_t> h;
    h.SetGlobalBuffer((__gm__ uint32_t*)tiling);
    uint32_t tokens = h.GetValue(0), batch = h.GetValue(1), qh = h.GetValue(2), kh = h.GetValue(3),
             pages = h.GetValue(4), len = h.GetValue(5), cores = h.GetValue(6), bits = h.GetValue(7);
    if (tokens == 0 || len == 0) {
        return;
    }
    bool prefill = h.GetValue(9), fp16 = h.GetValue(10);
    uint64_t aBytes = uint64_t(h.GetValue(11)) + (uint64_t(h.GetValue(12)) << 32);
    uint32_t physicalBlocks = h.GetValue(13);
    GM_ADDR user = GetUserWorkspace(workspace);
    GM_ADDR aw = prefill ? packed : user;
    GM_ADDR bw = user + aBytes;
    GM_ADDR offsets = tiling + h.GetValue(8);
    GM_ADDR cube = tiling + 64;
#define CALL(B, M) \
    Run<B, 512, M>(q, s, i, j, n, g, t, l, o, aw, bw, offsets, cube, prefill ? tokens : batch, kh, pages, len, cores, \
                   prefill, qh, physicalBlocks)
    if (fp16) {
        if (bits == 2) {
            CALL(2, half);
        } else if (bits == 3) {
            CALL(3, half);
        } else {
            CALL(4, half);
        }
    } else {
        if (bits == 2) {
            CALL(2, bfloat16_t);
        } else if (bits == 3) {
            CALL(3, bfloat16_t);
        } else {
            CALL(4, bfloat16_t);
        }
    }
#undef CALL
}
