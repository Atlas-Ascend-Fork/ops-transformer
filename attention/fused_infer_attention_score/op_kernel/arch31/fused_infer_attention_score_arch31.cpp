/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file fused_infer_attention_score_arch31.cpp
 * \brief arch31 (KirinX90 / Kirin9030) kernel 入口
 */

#include "fia_kernel_nonquant_arch31.h"

using namespace AttentionCommonArch31;

static __aicore__ void fused_infer_attention_arch31(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* pse_shift,
    __gm__ uint8_t* attenMask, __gm__ uint8_t* actualSeqLengths, __gm__ uint8_t* actualSeqLengthsKV,
    __gm__ uint8_t* deq_scale1, __gm__ uint8_t* quant_scale1, __gm__ uint8_t* deq_scale2, __gm__ uint8_t* quant_scale2,
    __gm__ uint8_t* quant_offset2, __gm__ uint8_t* antiquantScale, __gm__ uint8_t* antiquantOffset,
    __gm__ uint8_t* blocktable, __gm__ uint8_t* queryPaddingSize, __gm__ uint8_t* kvPaddingSize,
    __gm__ uint8_t* keyAntiquantScale, __gm__ uint8_t* keyAntiquantOffset, __gm__ uint8_t* valueAntiquantScale,
    __gm__ uint8_t* valueAntiquantOffset, __gm__ uint8_t* keySharedPrefix, __gm__ uint8_t* valueSharedPrefix,
    __gm__ uint8_t* actualSharedPrefixLen, __gm__ uint8_t* queryRope, __gm__ uint8_t* keyRope,
    __gm__ uint8_t* keyRopeAntiquantScale, __gm__ uint8_t* dequantScaleQuery, __gm__ uint8_t* learnableSink,
    __gm__ uint8_t* qStartIdx, __gm__ uint8_t* kvStartIdx, __gm__ uint8_t* attentionOut, __gm__ uint8_t* softmaxLse,
    __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    InitSocState();

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);

    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_KVBNSD_NOPA_NOMASK_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_KVBNSD_NOPA_LEFTUP_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_KVBNSD_NOPA_CAUSAL_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_KVBNSD_NOPA_BAND_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_KVBNSD_NOPA_EXPLICITMASK_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSH_KVBNSD_NOPA_NOMASK_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSH_KVBNSD_NOPA_CAUSAL_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSND_KVBNSD_NOPA_NOMASK_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSND_KVBNSD_NOPA_CAUSAL_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_KVBNSD_NOPA_TREEMASK_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSH_KVBNSD_NOPA_TREEMASK_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSND_KVBNSD_NOPA_TREEMASK_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_TND_KVBNSD_NOPA_NOMASK_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_TND_KVBNSD_NOPA_TREEMASK_NONFD_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_HP_PREFILL_D256_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_HP_PREFILL_D512_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_HP_DECODE_D256_TILING);
    TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_HP_DECODE_D512_TILING);

    if (TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_HP_PREFILL_D256_TILING)) {
        using FIAT = FIATypeArch31<half, half, half, FIA_LAYOUT_ARCH31::BNSD, FIA_LAYOUT_ARCH31::BNSD, 256U>;
        using CubeBlockType = FiaBlockCubeNonQuantArch31<FIAT>;
        using VecBlockType = FiaBlockVecNonQuantArch31<FIAT>;
        FiaKernelNonQuantArch31<FIAT, CubeBlockType, VecBlockType, ARCH31_HP_PATH_PREFILL> op;
        TPipe tPipe;
        op.Init(query, key, value, attenMask, attentionOut, workspace, tiling, nullptr, nullptr, &tPipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_HP_PREFILL_D512_TILING)) {
        using FIAT = FIATypeArch31<half, half, half, FIA_LAYOUT_ARCH31::BNSD, FIA_LAYOUT_ARCH31::BNSD, 512U>;
        using CubeBlockType = FiaBlockCubeNonQuantArch31<FIAT>;
        using VecBlockType = FiaBlockVecNonQuantArch31<FIAT>;
        FiaKernelNonQuantArch31<FIAT, CubeBlockType, VecBlockType, ARCH31_HP_PATH_PREFILL> op;
        TPipe tPipe;
        op.Init(query, key, value, attenMask, attentionOut, workspace, tiling, nullptr, nullptr, &tPipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_HP_DECODE_D256_TILING)) {
        using FIAT = FIATypeArch31<half, half, half, FIA_LAYOUT_ARCH31::BNSD, FIA_LAYOUT_ARCH31::BNSD, 256U>;
        using CubeBlockType = FiaBlockCubeNonQuantArch31<FIAT>;
        using VecBlockType = FiaBlockVecNonQuantArch31<FIAT>;
        FiaKernelNonQuantArch31<FIAT, CubeBlockType, VecBlockType, ARCH31_HP_PATH_DECODE> op;
        TPipe tPipe;
        op.Init(query, key, value, attenMask, attentionOut, workspace, tiling, nullptr, nullptr, &tPipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BNSD_HP_DECODE_D512_TILING)) {
        using FIAT = FIATypeArch31<half, half, half, FIA_LAYOUT_ARCH31::BNSD, FIA_LAYOUT_ARCH31::BNSD, 512U>;
        using CubeBlockType = FiaBlockCubeNonQuantArch31<FIAT>;
        using VecBlockType = FiaBlockVecNonQuantArch31<FIAT>;
        FiaKernelNonQuantArch31<FIAT, CubeBlockType, VecBlockType, ARCH31_HP_PATH_DECODE> op;
        TPipe tPipe;
        op.Init(query, key, value, attenMask, attentionOut, workspace, tiling, nullptr, nullptr, &tPipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSH_KVBNSD_NOPA_NOMASK_NONFD_TILING) ||
        TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSH_KVBNSD_NOPA_CAUSAL_NONFD_TILING) ||
        TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSH_KVBNSD_NOPA_TREEMASK_NONFD_TILING)) {
        using FIAT = FIATypeArch31<half, half, half, FIA_LAYOUT_ARCH31::BSH, FIA_LAYOUT_ARCH31::BSH>;
        using CubeBlockType = FiaBlockCubeNonQuantArch31<FIAT>;
        using VecBlockType = FiaBlockVecNonQuantArch31<FIAT>;
        FiaKernelNonQuantArch31<FIAT, CubeBlockType, VecBlockType> op;
        TPipe tPipe;
        op.Init(query, key, value, attenMask, attentionOut, workspace, tiling, nullptr, nullptr, &tPipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSND_KVBNSD_NOPA_NOMASK_NONFD_TILING) ||
        TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSND_KVBNSD_NOPA_CAUSAL_NONFD_TILING) ||
        TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_BSND_KVBNSD_NOPA_TREEMASK_NONFD_TILING)) {
        using FIAT = FIATypeArch31<half, half, half, FIA_LAYOUT_ARCH31::BSND, FIA_LAYOUT_ARCH31::BSND>;
        using CubeBlockType = FiaBlockCubeNonQuantArch31<FIAT>;
        using VecBlockType = FiaBlockVecNonQuantArch31<FIAT>;
        FiaKernelNonQuantArch31<FIAT, CubeBlockType, VecBlockType> op;
        TPipe tPipe;
        op.Init(query, key, value, attenMask, attentionOut, workspace, tiling, nullptr, nullptr, &tPipe);
        op.Process();
        return;
    }

    if (TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_TND_KVBNSD_NOPA_NOMASK_NONFD_TILING) ||
        TILING_KEY_IS(ARCH31_QF16_KVF16_OUTF16_TND_KVBNSD_NOPA_TREEMASK_NONFD_TILING)) {
        using FIAT = FIATypeArch31<half, half, half, FIA_LAYOUT_ARCH31::TND, FIA_LAYOUT_ARCH31::TND>;
        using CubeBlockType = FiaBlockCubeNonQuantArch31<FIAT>;
        using VecBlockType = FiaBlockVecNonQuantArch31<FIAT>;
        FiaKernelNonQuantArch31<FIAT, CubeBlockType, VecBlockType> op;
        TPipe tPipe;
        op.Init(query, key, value, attenMask, attentionOut, workspace, tiling,
                reinterpret_cast<__gm__ int64_t*>(actualSeqLengths),
                reinterpret_cast<__gm__ int64_t*>(actualSeqLengthsKV), &tPipe);
        op.Process();
        return;
    }

    using FIAT = FIATypeArch31<half, half, half, FIA_LAYOUT_ARCH31::BNSD, FIA_LAYOUT_ARCH31::BNSD>;
    using CubeBlockType = FiaBlockCubeNonQuantArch31<FIAT>;
    using VecBlockType = FiaBlockVecNonQuantArch31<FIAT>;
    FiaKernelNonQuantArch31<FIAT, CubeBlockType, VecBlockType> op;
    TPipe tPipe;
    op.Init(query, key, value, attenMask, attentionOut, workspace, tiling, nullptr, nullptr, &tPipe);
    op.Process();
}
