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
 * \file partial_rotary_mul_quant.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "arch35/rotary_position_embedding_reg_ab_hif8.h"
#include "arch35/rotary_position_embedding_reg_a_and_b_hif8.h"
#include "arch35/rotary_position_embedding_reg_aba_and_ba_hif8.h"
#include "arch35/rotary_position_embedding_reg_bab_hif8.h"
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
#include "arch35/rotary_position_embedding_reg_ab_hif4.h"
#include "arch35/rotary_position_embedding_reg_a_and_b_hif4.h"
#endif

using namespace AscendC;

extern "C" __global__ __aicore__ void partial_rotary_mul_quant(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR scale,
                                                               GM_ADDR y, GM_ADDR y_scale, GM_ADDR workspace,
                                                               GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(QuantRopeRegbaseTilingData, tilingData, tiling);
    if (TILING_KEY_IS(30010)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingABAAndBA<DTYPE_X, false, RoundMode::CAST_ROUND> op;
        op.Init(x, cos, sin, scale, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(30011)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingABAAndBA<DTYPE_X, true, RoundMode::CAST_ROUND> op;
        op.Init(x, cos, sin, scale, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(30020)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingBAB<DTYPE_X, RoundMode::CAST_ROUND> op(&pipe, &tilingData);
        op.Init(x, cos, sin, scale, y);
        op.Process();
    } else if (TILING_KEY_IS(30030)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingAB<DTYPE_X, RoundMode::CAST_ROUND> op;
        op.Init(x, cos, sin, scale, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(30040)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingAAndB<DTYPE_X, false, RoundMode::CAST_ROUND> op;
        op.Init(x, cos, sin, scale, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(30041)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingAAndB<DTYPE_X, true, RoundMode::CAST_ROUND> op;
        op.Init(x, cos, sin, scale, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(40010)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingABAAndBA<DTYPE_X, false, RoundMode::CAST_HYBRID> op;
        op.Init(x, cos, sin, scale, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(40011)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingABAAndBA<DTYPE_X, true, RoundMode::CAST_HYBRID> op;
        op.Init(x, cos, sin, scale, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(40020)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingBAB<DTYPE_X, RoundMode::CAST_HYBRID> op(&pipe, &tilingData);
        op.Init(x, cos, sin, scale, y);
        op.Process();
    } else if (TILING_KEY_IS(40030)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingAB<DTYPE_X, RoundMode::CAST_HYBRID> op;
        op.Init(x, cos, sin, scale, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(40040)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingAAndB<DTYPE_X, false, RoundMode::CAST_HYBRID> op;
        op.Init(x, cos, sin, scale, y, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(40041)) {
        PartialRotaryMulQuantHiF8::RotaryPositionEmbeddingAAndB<DTYPE_X, true, RoundMode::CAST_HYBRID> op;
        op.Init(x, cos, sin, scale, y, workspace, &tilingData, &pipe);
        op.Process();
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201 || __NPU_ARCH__ == 9202)
    } else if (TILING_KEY_IS(20030)) {
        PartialRotaryMulQuantHiF4::RotaryPositionEmbeddingAB<DTYPE_X> op;
        op.Init(x, cos, sin, y, y_scale, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20040)) {
        PartialRotaryMulQuantHiF4::RotaryPositionEmbeddingAAndB<DTYPE_X, false> op;
        op.Init(x, cos, sin, y, y_scale, workspace, &tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(20041)) {
        PartialRotaryMulQuantHiF4::RotaryPositionEmbeddingAAndB<DTYPE_X, true> op;
        op.Init(x, cos, sin, y, y_scale, workspace, &tilingData, &pipe);
        op.Process();
#endif
    }
}
