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
 * \file flash_attention_score.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "flash_attention.h"

#ifdef __DAV_C220_CUBE__
#define COPY_TILING_DATA(tiling) \
    GET_TILING_DATA_MEMBER(FlashAttentionGeneralTilingData, bmm1TilingData, bmm1TilingDataVar, tiling); \
    GET_TILING_DATA_MEMBER(FlashAttentionGeneralTilingData, bmm2TilingData, bmm2TilingDataVar, tiling); \
    const FlashAttentionGeneralTilingData* __restrict tilingData = nullptr; \
    const TCubeTiling* __restrict bmm1tiling = &bmm1TilingDataVar; \
    const TCubeTiling* __restrict bmm2tiling = &bmm2TilingDataVar;
#else
#define COPY_TILING_DATA(tiling) \
    GET_TILING_DATA_WITH_STRUCT(FlashAttentionGeneralTilingData, tilingDataIn, tiling); \
    const FlashAttentionGeneralTilingData* __restrict tilingData = &tilingDataIn; \
    const TCubeTiling* __restrict bmm1tiling = &(tilingData->bmm1TilingData); \
    const TCubeTiling* __restrict bmm2tiling = &(tilingData->bmm2TilingData);
#endif

#ifdef __DAV_C220_CUBE__
#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...) \
    do { \
        templateClass<__VA_ARGS__> op; \
        COPY_TILING_DATA(tiling); \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling); \
    } while (0)
#else
#define INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(templateClass, ...) \
    do { \
        COPY_TILING_DATA(tiling); \
        __gm__ uint8_t* user = GetUserWorkspace(workspace); \
        templateClass<__VA_ARGS__> op; \
        REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling); \
        op.UnpackInit(query, key, value, actualSeqLengths, actualSeqLengthsKv, attenMask, prmMask, sink, softmaxMax, \
                      softmaxSum, lseOut, attentionOut, maxLogits, user, tilingData, &tPipe); \
        op.Process(); \
    } while (0)
#endif

extern "C" __global__ __aicore__ void flash_attention(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* actualSeqLengths,
    __gm__ uint8_t* actualSeqLengthsKv, __gm__ uint8_t* attenMask, __gm__ uint8_t* prmMask, __gm__ uint8_t* sink,
    __gm__ uint8_t* softmaxMax, __gm__ uint8_t* softmaxSum, __gm__ uint8_t* lseOut, __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* maxLogits, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    TPipe tPipe;
    set_mask_norm();
    SetSysWorkspace(workspace);

    const TCubeTiling* __restrict bmm1tiling;
    const TCubeTiling* __restrict bmm2tiling;

    if (TILING_KEY_IS(10000000000002233943UL)) {
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttention, half, float);
        return;
    } else if (TILING_KEY_IS(10000000000002232943UL)) { // VarLen SplitS1S2HighPerf: BF16
        INVOKE_FA_GENERAL_OP_IMPL_VAR_LEN(FlashAttention, bfloat16_t, float);
        return;
    }
}
