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

#include "flash_attention_grad_bn2gs1s2_us1s2.h"
#include "flash_attention_grad_bn2gs1s2_us1s2_post.h"

#include "flash_attention_grad_bn2gs1_us1s2.h"
#include "flash_attention_grad_bn2gs2_us1s2.h"
#include "flash_attention_grad_ws_reduce.h"
#include "flash_attention_grad_ws_dq_reduce_post.h"
#include "flash_attention_grad_ws_dkv_reduce_post.h"

constexpr MatmulConfig MM_CFG_EXCEED = GetNormalConfig(true);
constexpr MatmulConfig MM_CFG_NORMAL = GetNormalConfig(false);
constexpr CubeFormat MM_NZ_OUT_FORMAT = CubeFormat::NZ;
constexpr CubeFormat MM_ND_OUT_FORMAT = CubeFormat::ND_ALIGN;
constexpr uint64_t INPUT_NONE = 0;
constexpr uint64_t INPUT_EXIST = 1;

constexpr static const uint64_t BSH = 0;
constexpr static const uint64_t SBH = 1;
constexpr static const uint64_t BNSD = 2;
constexpr static const uint64_t BSND = 3;
constexpr static const uint64_t TND = 4;

#define INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(INPUT_TYPE, IS_TND_INPUTS, MM_OUT_FORMAT) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2, tiling_data_in, tiling_data); \
        const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict tilingData = &tiling_data_in; \
        const TCubeTiling* __restrict bmm1tiling = &(tilingData->mm1TilingData); \
        const TCubeTiling* __restrict bmm2tiling = &(tilingData->mm2TilingData); \
        const TCubeTiling* __restrict bmm3tiling = &(tilingData->mm3TilingData); \
        FlashAttentionGradUs1s2Bn2gs1s2<INPUT_TYPE, float, IS_TND_INPUTS, MM_OUT_FORMAT> op; \
        REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm3, bmm2tiling, op.mm4, bmm3tiling); \
        op.Init(dy, query, key, value, attention_in, softmax_max, softmax_sum, softmax_lse, actual_seq_qlen, \
                actual_seq_kvlen, atten_mask, prm_mask, sink, user, tilingData, &pipeIn); \
        op.Process(); \
        op.SyncALLCores(); \
        pipeIn.Destroy(); \
        TPipe pipePost; \
        FlashAttentionGradUs1s2Bn2gs1s2Post<INPUT_TYPE, float> opPost; \
        opPost.Init(dq, dk, dv, dsink, user, tilingData, &pipePost); \
        opPost.Process(); \
    } while (0)

#define INVOKE_FAG_CERTAIN_IMPL(INPUT_TYPE, IS_TND_INPUTS, MM_OUT_FORMAT) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2, tiling_data_in, tiling_data); \
        const FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2* __restrict tilingData = &tiling_data_in; \
        const TCubeTiling* __restrict bmm1tiling = &(tilingData->mm1TilingData); \
        const TCubeTiling* __restrict bmm2tiling = &(tilingData->mm2TilingData); \
        const TCubeTiling* __restrict bmm3tiling = &(tilingData->mm3TilingData); \
        if (tilingData->s1s2BNGS1S2BaseParams.saveToWorkspace) { \
            FlashAttentionGradWsReduce<INPUT_TYPE, float, IS_TND_INPUTS, MM_OUT_FORMAT> op; \
            REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op.mm1, bmm1tiling, op.mm3, bmm2tiling, op.mm4, \
                              bmm3tiling); \
            op.Init(dy, query, key, value, attention_in, softmax_max, softmax_sum, softmax_lse, actual_seq_qlen, \
                    actual_seq_kvlen, atten_mask, prm_mask, sink, user, tilingData, &pipeIn); \
            op.Process(); \
            op.SyncALLCores(); \
            pipeIn.Destroy(); \
            TPipe pipePostKV; \
            FlashAttentionGradDkvReducePost<INPUT_TYPE, float> kvPost; \
            kvPost.Init(dk, dv, dsink, user, tilingData, &pipePostKV); \
            kvPost.Process(); \
            pipePostKV.Destroy(); \
            SyncAll(); \
            TPipe pipePostQ; \
            FlashAttentionGradDqReducePost<INPUT_TYPE, float> qPost; \
            qPost.Init(dq, user, tilingData, &pipePostQ); \
            qPost.Process(); \
        } else { \
            FlashAttentionGradUs1s2Bn2gs1<INPUT_TYPE, float, IS_TND_INPUTS, MM_OUT_FORMAT> op1; \
            FlashAttentionGradUs1s2Bn2gs2<INPUT_TYPE, float, IS_TND_INPUTS, MM_OUT_FORMAT> op2; \
            REGIST_MATMUL_OBJ(&pipeIn, GetSysWorkSpacePtr(), op1.mm1, bmm1tiling, op1.mm4, bmm3tiling, op2.mm1, \
                              bmm1tiling, op2.mm3, bmm2tiling); \
            op1.Init(dy, query, key, value, attention_in, softmax_max, softmax_sum, softmax_lse, actual_seq_qlen, \
                     actual_seq_kvlen, atten_mask, prm_mask, sink, user, tilingData, &pipeIn); \
            op1.Process(); \
            op1.SyncALLCores(); \
            pipeIn.Destroy(); \
            TPipe pipeNewBase; \
            op2.Init(dy, query, key, value, attention_in, softmax_max, softmax_sum, softmax_lse, actual_seq_qlen, \
                     actual_seq_kvlen, atten_mask, prm_mask, sink, user, tilingData, &pipeNewBase); \
            op2.Process(); \
            op2.SyncALLCores(); \
            pipeNewBase.Destroy(); \
\
            TPipe pipePost; \
            FlashAttentionGradUs1s2Bn2gs1s2Post<INPUT_TYPE, float> opPost; \
            opPost.Init(dq, dk, dv, dsink, user, tilingData, &pipePost); \
            opPost.Process(); \
        } \
    } while (0)

// implementation of kernel function
extern "C" __global__ __aicore__ void flash_attention_grad(
    __gm__ uint8_t* dy, __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* attention_in,
    __gm__ uint8_t* softmax_max, __gm__ uint8_t* softmax_sum, __gm__ uint8_t* softmax_lse,
    __gm__ uint8_t* actual_seq_qlen, __gm__ uint8_t* actual_seq_kvlen, __gm__ uint8_t* atten_mask,
    __gm__ uint8_t* prm_mask, __gm__ uint8_t* sink, __gm__ uint8_t* dq, __gm__ uint8_t* dk, __gm__ uint8_t* dv,
    __gm__ uint8_t* dsink, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling_data)
{
    TPipe pipeIn;
    set_mask_norm();
    SetSysWorkspace(workspace);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);

// --------------------------------------------float16 tilingkey------------------------------------------------------
#if (ORIG_DTYPE_QUERY == DT_FLOAT16)
    // -----------------------1.1 start---------------------------------
    if (TILING_KEY_IS(10000000000001003434UL)) {
        // input: tnd, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(half, true, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(930)) { // deterministic cal
        // input: tnd, mm_out:nd
        INVOKE_FAG_CERTAIN_IMPL(half, true, MM_ND_OUT_FORMAT);
        return;
    }
    // -----------------------1.1 end---------------------------------
#endif
// --------------------------------------------------------------------------------------------------------------------

// --------------------------------------------bfloat16 tilingkey------------------------------------------------------
#if (ORIG_DTYPE_QUERY == DT_BF16)
    if (TILING_KEY_IS(10000000000001002434UL)) {
        // unpack:1, mm_out:nd
        INVOKE_FAG_GENERAL_S1S2_BN2GS1S2_IMPL(bfloat16_t, true, MM_ND_OUT_FORMAT);
        return;
    } else if (TILING_KEY_IS(920)) { // deterministic cal
        // unpack:1, mm_out:nd, deterministic impl
        INVOKE_FAG_CERTAIN_IMPL(bfloat16_t, true, MM_ND_OUT_FORMAT);
        return;
    }
    // -----------------------1.1 end---------------------------------
#endif
    // --------------------------------------------------------------------------------------------------------------------
}
