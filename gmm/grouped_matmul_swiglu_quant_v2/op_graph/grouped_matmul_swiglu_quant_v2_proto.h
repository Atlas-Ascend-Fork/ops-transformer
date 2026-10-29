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
 * \file grouped_matmul_swiglu_quant_v2_proto.h
 * \brief Graph IR definition for GroupedMatmulSwigluQuantV2.
 */
#ifndef OPS_TRANSFORMER_GROUPED_MATMUL_SWIGLU_QUANT_V2_PROTO_H_
#define OPS_TRANSFORMER_GROUPED_MATMUL_SWIGLU_QUANT_V2_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
// clang-format off
/**
 * @brief GroupedMatmulSwigluQuantV2 operator interface implementation.
 *
 * @par Inputs:
 * @li x: A tensor. The format supports ND. The type supports int8, int4, float8_e5m2, float8_e4m3fn, float4_e2m1, hifloat8 and float4_e1m2.
 * INT4 is retained for the existing integer A4W4 path; it is distinct from the floating-point FP4 types.
 * float4_e1m2 is supported only in MXFP4 scenarios where weight uses FRACTAL_NZ; x remains ND and its logical K must be even.
 * @li x_scale: A tensor. Scales for x for each token. The format supports ND. The type supports float32, float8_e8m0.
 * @li group_list: A tensor. Indicates the matmul size distribution along separated dimension.
 * The format supports ND. The type supports int64. The first dimension of group_list should not be greater than 1024, meaning it supports up to 1024 groups.
 * @li weight: A tensor List. The format supports ND/NZ. The type supports int8, int4, float8_e5m2, float8_e4m3fn, float4_e2m1, hifloat8 and float4_e1m2.
 * INT4 is retained for the existing integer A8W4 MSD and A4W4 paths. float4_e1m2 requires FRACTAL_NZ weight format.
 * @li weight_scale: A tensor List. Format supports ND. The type supports float32, float8_e8m0, bfloat16, float16 and uint64.
 * uint64 is the packed dequantization-scale representation used by the existing integer INT4-weight A8W4 MSD/A4W4 paths,
 * not a numeric uint64 scale for MX quantization; MX scales use float8_e8m0.
 * The registered dtype sets cover multiple platforms and modes; valid combinations are checked by the operator.
 * @li weight_assist_matrix: A tensor List. The format supports ND. The type supports float32.
 * @li bias: A tensor. The format supports ND. The type supports float32.
 * @li smooth_scale: A tensor. The format supports ND. The type supports float32.
 *
 * @par Attributes:
 * @li dequant_mode: Optional. Indicates the mode of dequant. dtype: Int64. Default: 0. Reserved parameter.
 * @li dequant_dtype: Optional. Indicates the dtype of dequant. dtype: Int64. Default: 0. Reserved parameter.
 * @li quant_mode: Optional. Indicates the mode of quant. dtype: Int64. Default: 0. Reserved parameter.
 * @li quant_dtype: Optional. Indicates the quantized output dtype. dtype: Int64. Default: 0.
 * Supported output types and input/output combinations depend on the platform and quantization mode.
 * @li transpose_weight: A bool. Default: false. When true, the operator interprets weight with transposed K/N axes.
 * A transpose fusion pass may remove an explicit weight transpose and set this attribute to true.
 * MX weight_scale follows the same transpose semantics while its final packing axis is preserved.
 * @li group_list_type: An int. Indicates whether the value in group_list is cumsum or count.
 * 0, value in group_list is cumsum. 1, value in group_list is count. Default: 0.
 * @li tuning_config: A ListInt. The first element in the array indicates the expected number of tokens processed by each group.
 * The operator tiling performs optimal tiling based on the first element in the array. Default: {0}.
 * @li swiglu_mode: Optional. An int. Default: 0, preserving the original SwiGLU computation.
 * Mode 2 splits the matmul result into two halves along the last dimension, with the left half as the activation branch.
 * Mode 2 is supported only on Ascend 950 for MXFP8 with a single weight tensor in FRACTAL_NZ format.
 * In mode 2, x, weight and y must be float8_e4m3fn; x_scale, weight_scale and y_scale must be float8_e8m0.
 * The aclnnGroupedMatmulSwigluQuantWeightNzV3 interface only accepts swiglu_mode 2.
 * @li clamp_limit: Optional. A float. Default: 7.0. Used in mode 2; must be finite, positive and representable by float32.
 * The activation branch is clamped to an upper bound of clamp_limit; the linear branch is clamped to [-clamp_limit, clamp_limit].
 * @li glu_alpha: Optional. A float. Default: 1.702. Scales the sigmoid input in mode 2; must be finite and representable by float32.
 * @li glu_bias: Optional. A float. Default: 1.0. Added to the linear branch in mode 2; must be finite and representable by float32.
 * Mode 2 computes act * sigmoid(glu_alpha * act) * (linear + glu_bias) after clamping.
 * @li round_mode: Optional. A string. Default: "rint". Only "rint" is supported in the mode 2 MXFP8 weight NZ scenario.
 * @li scale_alg: Optional. An int. Default: 0. In the mode 2 MXFP8 weight NZ scenario, supports 0 (OCP) and 1 (cuBLAS).
 * @li dst_type_max: Optional. A float. Default: 0.0. Only 0.0 is supported in the mode 2 MXFP8 weight NZ scenario.
 *
 * @par Output:
 * @li y: A tensor. The type supports int8, float8_e5m2, float8_e4m3fn, float4_e2m1, float4_e1m2, hifloat8. The format supports ND.
 * @li y_scale: A tensor. The type supports float32, float8_e8m0. The format supports ND.

*\n
*\n
* - The following are the supported format, data types and shapes(for Ascend 950 AI Processor):
*\n
| Tensor    | x                         | x_scale             | group_list | weight                                            | weight_scale                                    | y                                                    | y_scale                  |
| :-------: | :-----------------------: | :-----------------: | :--------: | :------------------------------------------------:| :---------------------------------------------: | :--------------------------------------------------: |:-----------------------: |
| Format    | ND                        | ND                  | ND         | ND                                                | ND                                              | ND                                                   | ND                       |
| Dtype     | float8_e4m3fn/float8_e5m2 | float8_e8m0         | int64      | float8_e4m3fn/float8_e5m2                         | float8_e8m0                                     | float8_e4m3fn/float8_e5m2                            | float8_e8m0              |
| Shape     | (M,K)                     | (M,ceil(K/64),2)    | (B,)       | [(B,K,N)]/[(B,N,K)]                               | [(B,ceil(K/64),N,2)]/[(B,N,ceil(K/64),2)]       | (M,N/2)                                              | (M,ceil((N/2)/64),2)     |
| Dtype     | float4_e2m1               | float8_e8m0         | int64      | float4_e2m1                                       | float8_e8m0                                     | float8_e4m3fn/float8_e5m2/float4_e2m1                | float8_e8m0              |
| Shape     | (M,K)                     | (M,ceil(K/64),2)    | (B,)       | [(B,K,N)]/[(B,N,K)]                               | [(B,ceil(K/64),N,2)]/[(B,N,ceil(K/64),2)]       | (M,N/2)                                              | (M,ceil((N/2)/64),2)     |
| Dtype     | int8                      | float32             | int64      | int8                                              | bf16/float32/float16                            | int8                                                 | float32                  |
| Shape     | (M,K)                     | (M,)                | (B,)       | [(B,K,N)]/[(B,N,K)]                               | (B,N,2)                                         | (M,N/2)                                              | (M,)                     |
| Dtype     | float8_e4m3fn/float8_e5m2 | float32             | int64      | float8_e4m3fn/float8_e5m2                         | bf16/float32                                    | float8_e4m3fn/float8_e5m2                            | float32                  |
| Shape     | (M,K)                     | (M,)                | (B,)       | [(B,K,N)]/[(B,N,K)]                               | (B,N,2)                                         | (M,N/2)                                              | (M,)                     |
| Dtype     | hifloat8                  | float32             | int64      | hifloat8                                          | bf16/float32                                    | hifloat8                                             | float32                  |
| Shape     | (M,K)                     | (M,)                | (B,)       | [(B,K,N)]/[(B,N,K)]                               | (B,N,2)                                         | (M,N/2)                                              | (M,)                     |
| Format2   | ND                        | ND                  | ND         | FRACTAL_NZ                                        | ND                                              | ND                                                   | ND                       |
| Dtype     | float8_e4m3fn             | float8_e8m0         | int64      | float8_e4m3fn                                     | float8_e8m0                                     | float8_e4m3fn                                        | float8_e8m0              |
| Shape     | (M,K)                     | (M,ceil(K/64),2)    | (B,)       | [(B,N/32,K/16,16,32)]/[(B,K/32,N/16,16,32)]       | [(B,ceil(K/64),N,2)]/[(B,N,ceil(K/64),2)]       | (M,N/2)                                              | (M,ceil((N/2)/64),2)     |
| Dtype     | float4_e2m1               | float8_e8m0         | int64      | float4_e2m1                                       | float8_e8m0                                     | float4_e2m1                                          | float8_e8m0              |
| Shape     | (M,K)                     | (M,ceil(K/64),2)    | (B,)       | [(B,N/64,K/16,16,64)]/[(B,K/64,N/16,16,64)]       | [(B,ceil(K/64),N,2)]/[(B,N,ceil(K/64),2)]       | (M,N/2)                                              | (M,ceil((N/2)/64),2)     |
| Dtype     | float4_e1m2               | float8_e8m0         | int64      | float4_e2m1                                       | float8_e8m0                                     | float4_e2m1                                          | float8_e8m0              |
| Shape     | (M,K)                     | (M,ceil(K/64),2)    | (B,)       | [(B,N/64,K/16,16,64)]/[(B,K/64,N/16,16,64)]       | [(B,ceil(K/64),N,2)]/[(B,N,ceil(K/64),2)]       | (M,N/2)                                              | (M,ceil((N/2)/64),2)     |
| Dtype     | float4_e2m1               | float8_e8m0         | int64      | float4_e1m2                                       | float8_e8m0                                     | float4_e2m1                                          | float8_e8m0              |
| Shape     | (M,K)                     | (M,ceil(K/64),2)    | (B,)       | [(B,N/64,K/16,16,64)]/[(B,K/64,N/16,16,64)]       | [(B,ceil(K/64),N,2)]/[(B,N,ceil(K/64),2)]       | (M,N/2)                                              | (M,ceil((N/2)/64),2)     |
| Dtype     | float4_e1m2               | float8_e8m0         | int64      | float4_e1m2                                       | float8_e8m0                                     | float4_e2m1                                          | float8_e8m0              |
| Shape     | (M,K)                     | (M,ceil(K/64),2)    | (B,)       | [(B,N/64,K/16,16,64)]/[(B,K/64,N/16,16,64)]       | [(B,ceil(K/64),N,2)]/[(B,N,ceil(K/64),2)]       | (M,N/2)                                              | (M,ceil((N/2)/64),2)     |
*\n
*\n
* @attention Constraints:
 * @li On Ascend 950, N must be 128 aligned for the original SwiGLU computation (swiglu_mode 0).
 * For the mode 2 MXFP8 weight NZ scenario, N must be positive and 64 aligned; weight supports both transpose and non-transpose.
* @li When group_list_type is 0, group_list must be a non-negative monotone non-decreasing array. The last value in group_list must be no greater than the first dimension of tensor x;
* When group_list_type is 1, it must be a non-negative array. Sum of values in group_list must be no greater than the first dimension of tensor x. \n
*/
// clang-format on
REG_OP(GroupedMatmulSwigluQuantV2)
    .INPUT(x, TensorType({DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8,
                          DT_FLOAT4_E1M2}))
    .INPUT(x_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .INPUT(group_list, TensorType({DT_INT64}))
    .DYNAMIC_INPUT(weight, TensorType({DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8,
                                       DT_FLOAT4_E1M2}))
    .DYNAMIC_INPUT(weight_scale, TensorType({DT_FLOAT, DT_BF16, DT_FLOAT16, DT_UINT64, DT_FLOAT8_E8M0}))
    .DYNAMIC_INPUT(weight_assist_matrix, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT}))
    .OPTIONAL_INPUT(smooth_scale, TensorType({DT_FLOAT}))
    .ATTR(dequant_mode, Int, 0)
    .ATTR(dequant_dtype, Int, 0)
    .ATTR(quant_mode, Int, 0)
    .ATTR(quant_dtype, Int, 0)
    .ATTR(transpose_weight, Bool, false)
    .ATTR(group_list_type, Int, 0)
    .ATTR(tuning_config, ListInt, {0})
    .ATTR(swiglu_mode, Int, 0)
    .ATTR(clamp_limit, Float, 7.0F)
    .ATTR(glu_alpha, Float, 1.702F)
    .ATTR(glu_bias, Float, 1.0F)
    .ATTR(round_mode, String, "rint")
    .ATTR(scale_alg, Int, 0)
    .ATTR(dst_type_max, Float, 0.0F)
    .OUTPUT(y, TensorType({DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8, DT_FLOAT4_E1M2}))
    .OUTPUT(y_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OP_END_FACTORY_REG(GroupedMatmulSwigluQuantV2)

} // namespace ge

#endif
