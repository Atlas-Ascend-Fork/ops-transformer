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
 * \file quant_flash_attn.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

/* --- arch22: full INT8 GQA with required Q/K/V/P scales --- */
#include "lib/matmul_intf.h"
#include <type_traits>
#include "arch22/quant_flash_attn_template_tiling_key_arch22.h"
#include "arch22/quant_flash_attn_kernel_int8.h"
#include "arch22/quant_flash_attn_entry_arch22.h"

using namespace AscendC;

template <uint8_t inOutLayoutType, uint16_t rawConfig, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd, bool hasVTail>
__global__ __aicore__ void quant_flash_attn(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dequantScaleQuery,
    __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* pScale, __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* sequsedQ,
    __gm__ uint8_t* sequsedKv, __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
    __gm__ uint8_t* vTail, __gm__ uint8_t* blockTableTail, __gm__ uint8_t* sequsedVTail, __gm__ uint8_t* attnOut,
    __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    using namespace BaseApi::Int8GqaTnd;
    REGISTER_TILING_DEFAULT(QuantFlashAttnTilingPayloadInt8);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (KvLayoutType == QFA_INT8_TPL_PAGED_KV_LAYOUT) {
        quant_flash_attn_decode<rawConfig>(query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue,
                                           blockTable, pScale, cuSeqLensQ, cuSeqLensKv, sequsedQ, sequsedKv, sinks,
                                           attnMask, metadata, vTail, blockTableTail, sequsedVTail, attnOut, softmaxLse,
                                           workspace, tiling);
        return;
    } else {
        constexpr uint16_t config = rawConfig & QFA_INT8_CFG_PIPELINE_MASK;
        constexpr bool WINDOWED = (rawConfig & QFA_INT8_CFG_WINDOW_FLAG) != 0U;
        constexpr bool RETURN_LSE = (rawConfig & QFA_INT8_CFG_LSE_FLAG) != 0U;
        constexpr auto path = inOutLayoutType == QFA_INT8_TPL_BNSD ? BaseApi::Int8GqaPath::BNSD :
                              inOutLayoutType == QFA_INT8_TPL_TND  ? BaseApi::Int8GqaPath::TND :
                                                                     BaseApi::Int8GqaPath::BSND;
        constexpr bool packedD64 = config == QFA_INT8_CFG_D64_M512 && path == BaseApi::Int8GqaPath::TND;
        constexpr bool d64Opt = config == QFA_INT8_CFG_D64_M512_TASK_MAP;
        constexpr uint16_t pipeline = config == QFA_INT8_CFG_D128_M512_TASK_MAP ? QFA_INT8_CFG_D128_M512 :
                                      d64Opt                                    ? QFA_INT8_CFG_D64_M512 :
                                                                                  config;
        constexpr bool longTnd = path == BaseApi::Int8GqaPath::TND &&
                                 (pipeline == QFA_INT8_CFG_RESIDENT_Q || pipeline == QFA_INT8_CFG_D128_M512);
        constexpr uint16_t features = d64Opt ? QFA_VEC_D64_TASK_MAP_FEATURES :
                                      packedD64 || config == QFA_INT8_CFG_D96_M512_FIXPIPE ||
                                              config == QFA_INT8_CFG_D96_M256_CAUSAL_MHA ||
                                              (longTnd && pipeline == QFA_INT8_CFG_D128_M512) ?
                                               QFA_VEC_PREFILL_FEATURES :
                                               0U;
        BaseApi::QuantFlashAttnKernelGqaInt8<
            RETURN_LSE, path, pipeline, packedD64 || (d64Opt && path != BaseApi::Int8GqaPath::BNSD), features,
            features != 0U || (longTnd && pipeline == QFA_INT8_CFG_RESIDENT_Q), d64Opt, longTnd,
            longTnd && pipeline == QFA_INT8_CFG_RESIDENT_Q, WINDOWED>::Process(query, key, value, dequantScaleQuery,
                                                                               dequantScaleKey, dequantScaleValue,
                                                                               pScale, metadata, attnOut, softmaxLse,
                                                                               workspace, tiling);
    }
}
