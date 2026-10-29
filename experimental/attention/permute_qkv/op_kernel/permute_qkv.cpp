/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "permute_qkv.h"

extern "C" __global__ __aicore__ void permute_qkv(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output, GM_ADDR workspace,
                                                  GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    using namespace PermuteQkv;

    if (TILING_KEY_IS(0)) {
        TPipe pipe;
        using inType = bfloat16_t;
        PermuteQkvKernel<inType> permute_qkv_op(tiling_data, &pipe);
        permute_qkv_op.Process(q, k, v, output);
    } else if (TILING_KEY_IS(1)) {
        TPipe pipe;
        using inType = half;
        PermuteQkvKernel<inType> permute_qkv_op(tiling_data, &pipe);
        permute_qkv_op.Process(q, k, v, output);
    }
}
