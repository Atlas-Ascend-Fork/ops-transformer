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
 * \file grouped_matmul_activation_proto.h
 * \brief Operator prototype for grouped matmul with direct BF16 SwiGLU output.
 */

#ifndef GROUPED_MATMUL_ACTIVATION_PROTO_H_
#define GROUPED_MATMUL_ACTIVATION_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
/**
 * @brief Fuses MXFP4 E2M1 A4W4 grouped matmul with split SwiGLU activation.
 * Supports Ascend 950PR/Ascend 950DT, WeightNZ weights and direct BF16 output.
 * K and N must be positive multiples of 128. M is the total token count,
 * K is the input feature size, N is the grouped matmul output feature size,
 * and E is the expert count. MX scales are applied before FP32 accumulation.
 * No output MX quantization or output scale is produced.
 *
 * @par Inputs:
 * @li x: A FLOAT4_E2M1 tensor with logical shape (M, K), in ND format.
 * @li x_scale: A FLOAT8_E8M0 tensor with shape (M, K/64, 2), in ND format.
 * Each scale pair represents two consecutive blocks of 32 K elements.
 * @li group_list: An INT64 tensor with shape (E), in ND format. Interpreted
 * as cumulative group end positions or per-group token counts by group_list_type.
 * @li weight: A dynamic FLOAT4_E2M1 tensor list, in FRACTAL_NZ format.
 * Exactly one tensor contains all E experts. Its logical view shape is (E, K, N)
 * when transpose_weight is false and (E, N, K) when transpose_weight is true.
 * @li weight_scale: A dynamic FLOAT8_E8M0 tensor list, in ND format.
 * Exactly one tensor with shape (E, K/64, N, 2) when transpose_weight is false
 * or (E, N, K/64, 2) when transpose_weight is true.
 * @li bias: A dynamic BF16 tensor list reserved for grouped matmul bias.
 * The current A4W4 path requires it to be empty.
 *
 * @par Attributes:
 * @li dequant_mode: An int. Only 2 (MX dequantization) is supported. Default: 2.
 * @li dequant_dtype: An int. Only 0 (FP32 intermediate result) is supported. Default: 0.
 * @li act_type: A string. Only "swiglu" is supported. Default: "swiglu".
 * @li group_list_type: An int. 0 means cumulative group end positions;
 * 1 means per-group token counts. Default: 0.
 * @li tuning_config: A list int. Reserved; must be empty. Default: {}.
 * @li swiglu_mode: An int. The public activation interface only supports 2.
 * Split the grouped matmul result C into front and back halves A and B along N.
 * The mode-2 formulas are: \n
 * u = min(A, clamp_limit). \n
 * v = clip(B, -clamp_limit, clamp_limit) + glu_bias. \n
 * y = u / (1 + exp(-glu_alpha * u)) * v, then cast to BF16. \n
 * Internal mode 0 is retained only for the legacy WeightNzV3 A4W4 BF16 caller. \n
 * y = A / (1 + exp(-A)) * B, without clamp, alpha or bias. \n
 * Default: 2.
 * @li clamp_limit: A float. Must be finite and positive. Default: 7.0.
 * @li glu_alpha: A float. The sigmoid input multiplier; must be finite. Default: 1.702.
 * @li glu_bias: A float. The linear branch offset; must be finite. Default: 1.0.
 * @li transpose_weight: A bool. false means weight uses the (E, K, N) view and
 * weight_scale uses (E, K/64, N, 2); true means weight uses the (E, N, K) view
 * and weight_scale uses (E, N, K/64, 2). ACLNN derives this internal attribute
 * from matching weight and weight_scale views. Default: false.
 *
 * @par Outputs:
 * y: A BF16 tensor with shape (M, N/2), in ND format. It is the only output.
 */
REG_OP(GroupedMatmulActivation)
    .INPUT(x, TensorType({DT_FLOAT4_E2M1}))
    .INPUT(x_scale, TensorType({DT_FLOAT8_E8M0}))
    .INPUT(group_list, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(weight, TensorType({DT_FLOAT4_E2M1}))
    .DYNAMIC_INPUT(weight_scale, TensorType({DT_FLOAT8_E8M0}))
    .DYNAMIC_INPUT(bias, TensorType({DT_BF16}))
    .OUTPUT(y, TensorType({DT_BF16}))
    .ATTR(dequant_mode, Int, 2)
    .ATTR(dequant_dtype, Int, 0)
    .ATTR(act_type, String, "swiglu")
    .ATTR(group_list_type, Int, 0)
    .ATTR(tuning_config, ListInt, {})
    .ATTR(swiglu_mode, Int, 2)
    .ATTR(clamp_limit, Float, 7.0f)
    .ATTR(glu_alpha, Float, 1.702f)
    .ATTR(glu_bias, Float, 1.0f)
    .ATTR(transpose_weight, Bool, false)
    .OP_END_FACTORY_REG(GroupedMatmulActivation)
} // namespace ge

#endif // GROUPED_MATMUL_ACTIVATION_PROTO_H_
