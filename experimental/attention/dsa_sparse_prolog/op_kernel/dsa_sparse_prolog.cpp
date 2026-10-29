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
 * \file dsa_sparse_prolog.cpp
 * \brief
 */

#include "kernel_dsa_sparse_prolog_vec_s1_cub_s2.h"

using namespace DsaSparseProlog;

#define COPY_TILING_DATA_BASE_PARAMS(tiling) \
    GET_TILING_DATA_MEMBER(DsaSparsePrologTilingData, baseParams, tilingDataIn, tiling); \
    const DsaSparsePrologTilingData* __restrict tilingData = nullptr; \
    const DsaSparsePrologBaseParams* __restrict tilingDataBaseParams = &tilingDataIn;

#define INVOKE_DSA_SPARSE_PROLOG_NO_KFC_OP_IMPL(templateClass, ...) \
    do { \
        COPY_TILING_DATA_BASE_PARAMS(tiling); \
        templateClass<MLAPType<__VA_ARGS__>> op(&pipe, tilingData, tilingDataBaseParams); \
        op.Init(tokenX, weightDq, weightUqQr, weightDkvKr, rmsnormGammaCq, rmsnormGammaCkv, ropeSin, ropeCos, \
                qNopeOut, qPeOut, ctKvOut, kPeOut, qrOut, qMOut, qRstdOut, ckvMOut, ckvRstdOut, workspace); \
        op.Process(); \
    } while (0)

extern "C" __global__ __aicore__ void dsa_sparse_prolog(
    __gm__ uint8_t* tokenX, __gm__ uint8_t* weightDq, __gm__ uint8_t* weightUqQr, __gm__ uint8_t* weightDkvKr,
    __gm__ uint8_t* rmsnormGammaCq, __gm__ uint8_t* rmsnormGammaCkv, __gm__ uint8_t* ropeSin, __gm__ uint8_t* ropeCos,
    __gm__ uint8_t* qNopeOut, __gm__ uint8_t* qPeOut, __gm__ uint8_t* ctKvOut, __gm__ uint8_t* kPeOut,
    __gm__ uint8_t* qrOut, __gm__ uint8_t* qMOut, __gm__ uint8_t* qRstdOut, __gm__ uint8_t* ckvMOut,
    __gm__ uint8_t* ckvRstdOut, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    TPipe pipe;
    // 十位代表场景    0-FP16     1-BF16
    if (TILING_KEY_IS(10000000000000000)) {
        // input FP16
        INVOKE_DSA_SPARSE_PROLOG_NO_KFC_OP_IMPL(DsaSparsePrologVecS1CubS2, half);
    } else if (TILING_KEY_IS(10000000000000010)) {
        // input BF16
        INVOKE_DSA_SPARSE_PROLOG_NO_KFC_OP_IMPL(DsaSparsePrologVecS1CubS2, bfloat16_t);
    }
}
