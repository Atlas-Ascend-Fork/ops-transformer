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
 * \file dsa_dense.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "dsa_dense.h"

using namespace AscendC;
using namespace matmul;

extern "C" __global__ __aicore__ void dsa_dense(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* max,
                                                __gm__ uint8_t* sum, __gm__ uint8_t* qLen, __gm__ uint8_t* kvLen,
                                                __gm__ uint8_t* mask, __gm__ uint8_t* queryIndex,
                                                __gm__ uint8_t* keyIndex, __gm__ uint8_t* weights,
                                                __gm__ uint8_t* maxIndex, __gm__ uint8_t* sumIndex,
                                                __gm__ uint8_t* loss, __gm__ uint8_t* dq, __gm__ uint8_t* dk,
                                                __gm__ uint8_t* dw, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    // global TPipe obj
    TPipe tPipe;
    AscendCUtils::SetOverflow(1);

    // user worksapce
    SetSysWorkspace(workspace);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);

    // tiling obj
    GET_TILING_DATA(tiling_data, tiling);

    if (TILING_KEY_IS(0)) {
        const TCubeTiling* __restrict mm1Tiling = &(tiling_data.mm1TilingData);
        const TCubeTiling* __restrict mm2Tiling = &(tiling_data.mm2TilingData);
        const TCubeTiling* __restrict mm3Tiling = &(tiling_data.mm3TilingData);

        ops_dsa_dense::DsaDense<bfloat16_t, float> op;
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.mm1, mm1Tiling, op.mm2, mm2Tiling, op.mm3, mm3Tiling);
        op.Init(query, key, max, sum, qLen, kvLen, mask, queryIndex, keyIndex, weights, maxIndex, sumIndex, loss, dq,
                dk, dw, user, &tiling_data, &tPipe);
        op.Process();
    }
}
