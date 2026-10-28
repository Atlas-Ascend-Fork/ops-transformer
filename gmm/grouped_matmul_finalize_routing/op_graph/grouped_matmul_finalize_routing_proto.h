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
 * \file grouped_matmul_finalize_routing_proto.h
 * \brief Graph IR definition for GroupedMatmulFinalizeRouting.
 */
#ifndef OPS_TRANSFORMER_GROUPED_MATMUL_FINALIZE_ROUTING_PROTO_H_
#define OPS_TRANSFORMER_GROUPED_MATMUL_FINALIZE_ROUTING_PROTO_H_

#include "graph/operator_reg.h"

namespace ge {
// clang-format off
/**
* @brief Combines grouped matrix multiplication with MoE finalize routing.
* The grouped matmul results are weighted by logit and combined according to row_index.
* Weight tensors may use ND or a supported Ascend affinity data layout, depending on the quantization mode.

* @par Inputs:
* @li x: A tensor, which is the input x in the formula, supports the ND data format (refer to Data Format), and the shape supports 2 dimensions with the dimension being (m, k). The data type supports INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8.
* @li w: A weight tensor. The registered types are INT8, INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1 and DT_HIFLOAT8.
* Supported ND shapes are (e, k, n) or (e, n, k), selected by transpose_w. Physical affinity-layout shapes depend on dtype and platform.
* For the legacy affinity layout when transpose_w is false, the shape is (e, n1, k1, k0, n0), where k0 = 16 and n0 = 32, and ceilDiv(k, 16) = k1.
* The aclnnCalculateMatmulWeightSizeV2 interface and the aclnnTransMatmulWeight interface can be used to complete the conversion of the input format from the ND format to the Ascend affinity data layout format.
* @li scale: Quantization scaling factors in ND format. Registered types are FLOAT32, INT64, BFLOAT16 and DT_FLOAT8_E8M0.
* In the legacy INT4-weight path, scale has shape (e, 1, n); ordinary per-channel paths use (e, n) or the supported (e, 1, n) view.
* MX quantization uses FLOAT8_E8M0 with shape (e, ceil(k/64), n, 2) or (e, n, ceil(k/64), 2), consistent with transpose_w.
* The values e and n must agree with w. Valid dtype/shape combinations depend on the quantization mode; not every registered type is valid in every path.
* @li bias: A bias tensor in ND format. Registered types are FLOAT32 and BFLOAT16, subject to the platform and quantization mode.
* The current Ascend950 MX BFLOAT16-output TensorAPI path does not support bias.
* @li offset: A tensor of offset, only supported when w is INT4, the shape is three-dimensional (e, 1, n), the format supports ND. The supported data types is FLOAT32.
* @li pertoken_scale: The dequantization parameters for matrix calculation support the ND data format (refer to Data Format). They correspond to the x matrix, with a dimension of (m). The supported data type is FLOAT32, DT_FLOAT8_E8M0. Non-contiguous tensors are not supported.
* @li group_list: An INT64 tensor in ND format describing the M-axis distribution of matmul groups; its length must not exceed 1024.
* @li shared_input: In the MoE (Mixture of Experts) calculation, the output of the shared experts needs to undergo a combine operation with the output of the MoE experts. The supported data types are bfloat16.
* @li logit: Per-token routing weights applied before combining matmul results. The registered types are FLOAT32 and BFLOAT16.
* FLOAT32 is used by the ordinary paths. BFLOAT16 logit is supported only in the MX BFLOAT16-output path; it is not enabled for all quantization modes.
* @li row_index: The outputs of the MoE (Mixture of Experts) are combined according to the rowIndex, where the values in rowIndex serve as the indices for the scatter add operation during the combination. The supported data types are int64 and int32.
* @par Attributes:
* @li dtype: An int selecting the output type. Default: 0 (FLOAT32). Value 2 selects BFLOAT16 in the supported MX path.
* FLOAT16 output is not supported by this Graph IR; value 1 does not enable FLOAT16 output in the current implementation.
* @li shared_input_weight: The coefficients for combining the shared experts and the MoE experts. The shareInput is multiplied first with these coefficients, and then the result is accumulated with the output of the MoE experts. The supported data type is float32.
* @li shared_input_offset: The offset of the output of the shared experts in the total output. The supported data type is int64.
* @li transpose_x: Whether the left matrix is transposed. Default value: false(not transposed).
* @li transpose_w: Whether the right matrix is transposed. Default value: false(not transposed).
* @li output_bs: The first dimension of the combined output y, whose shape is (output_bs, n).
* @li group_list_type: Group type of GroupedMatmul. Default value: 1. When configured as 0: It is in the cumsum mode, which means it is the prefix sum. When configured as 1: It is in the count mode. The supported data type is int64. \n

* @par Outputs:
* @li y: A single ND tensor containing the combined result, with shape (output_bs, n).
* Registered types are FLOAT32 and BFLOAT16. BFLOAT16 is limited to the supported MX path described below; FLOAT16 is not registered. \n

* @attention Constraints:
* Ordinary quantization combinations with FLOAT32 output (BFLOAT16 output is listed separately in the MX section): \n
*  | x             | w             |  scale           | bias     | offset  | pertokenScale | groupList | sharedInput | logit   | rowIndex    |  out    | \n
*  | ----          | ----          | -------          | -------  | ------- | ------------- | --------- | ----------- | ------- | --------    | ------- | \n
*  | INT8          | INT4          | INT64            | FLOAT32  | FLOAT32 | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64       | FLOAT32 | \n
*  | INT8          | INT4          | INT64            | FLOAT32  | null    | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64       | FLOAT32 | \n
*  | INT8          | INT8          | FLOAT32          | null     | null    | FLOAT32       | INT64     | BFLOAT16    | FLOAT32 | INT64       | FLOAT32 | \n
*  | INT8          | INT8          | FLOAT32          | null     | null    | FLOAT32       | INT64     | null        | null    | INT64       | FLOAT32 | \n
*  | INT8          | INT8          | FLOAT32          | BFLOAT16 | null    | FLOAT32       | INT64     | null        | null    | INT64       | FLOAT32 | \n
*  | INT8          | INT8          | BFLOAT16         | BFLOAT16 | null    | FLOAT32       | INT64     | null        | null    | INT64       | FLOAT32 | \n
*  | INT8          | INT8          | FLOAT32/BFLOAT16 | BFLOAT16 | null    | FLOAT32/null  | INT64     | BFLOAT16    | FLOAT32 | INT64/INT32 | FLOAT32 | \n
*  | FLOAT8_E4M3FN | FLOAT8_E4M3FN | FLOAT32/BFLOAT16 | BFLOAT16 | null    | FLOAT32/null  | INT64     | BFLOAT16    | FLOAT32 | INT64       | FLOAT32 | \n
*  | HIFLOAT8      | HIFLOAT8      | FLOAT32/BFLOAT16 | BFLOAT16 | null    | FLOAT32/null  | INT64     | BFLOAT16    | FLOAT32 | INT64       | FLOAT32 | \n
* Legacy non-MX deployment ranges below are not universal limits for all MX paths: \n
* In the MoE interpretation, m = batch * topk, where topk is the number of selected experts per token; the legacy m range is [1, 16 * 1024 * 8]. \n
* Legacy k values are 256, 512, 1024, 1408 and 2048; legacy n values are 2048, 7168 and 7680. Other values depend on the selected implementation. \n
* The legacy e range is [1, 256]. MX group limits are given below. \n
* For shared_input, the legacy shape is (batch/dp, n), where dp is the data-parallel partition count, batch is in [1, 16 * 1024], and batch/dp is in [1, 2 * 1024]. \n
* The operator receives output_bs directly; the above batch/dp description does not introduce another output_bs attribute or an undefined p parameter. \n
* When group_list_type is 0, group_list must be non-negative and monotone non-decreasing, with its last value no greater than m. \n
* When group_list_type is 1, group_list must be non-negative, with the sum of its values no greater than m. \n

* @attention MX Quantization Constraints:
*  |             x             |            w              |     scale     |      bias    | offset  | pertokenScale | groupList |  sharedInput  | logit   | rowIndex |  out    | \n
*  | ------------------------- | ------------------------- | ------------- | ------------ | ------- | ------------- | --------- | ------------- | ------- | -------- | ------- | \n
*  | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E8M0   | FLOAT32/NULL | NULL    | FLOAT8_E8M0   | INT64     | BFLOAT16/NULL | FLOAT32 | INT64    | FLOAT32 | \n
*  | FLOAT4_E2M1               | FLOAT4_E2M1               | FLOAT8_E8M0   | FLOAT32/NULL | NULL    | FLOAT8_E8M0   | INT64     | BFLOAT16/NULL | FLOAT32 | INT64    | FLOAT32 | \n
* Additional MX BFLOAT16-output combinations on the supported Ascend950 TensorAPI path (weight must use FRACTAL_NZ): \n
*  | x                         | w                         | scale         | bias | offset | pertoken_scale | group_list | shared_input  | logit            | row_index | y        | \n
*  | ------------------------- | ------------------------- | ------------- | ---- | ------ | -------------- | ---------- | ------------- | ---------------- | --------- | -------- | \n
*  | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E4M3FN/FLOAT8_E5M2 | FLOAT8_E8M0   | NULL | NULL   | FLOAT8_E8M0    | INT64      | BFLOAT16/NULL | FLOAT32/BFLOAT16 | INT32     | BFLOAT16 | \n
*  | FLOAT4_E2M1               | FLOAT4_E2M1               | FLOAT8_E8M0   | NULL | NULL   | FLOAT8_E8M0    | INT64      | BFLOAT16/NULL | FLOAT32/BFLOAT16 | INT32     | BFLOAT16 | \n
* These rows describe the dtype relationships, not all shape/layout restrictions. They do not enable BFLOAT16 output for ordinary INT8/HIFLOAT8 paths. \n
* PertokenScale dimension is represented as: (m, ceil(k/64), 2). The data type supports FLOAT8_E8M0, and the shape supports 3 dimensions.
* Scale dimension is represented as: (e, ceil(k/64), n, 2) or (e, n, ceil(k/64), 2). The data type supports FLOAT8_E8M0, and the shape supports 4 dimensions.
* In the MXFP4 scenario, k must be even. When transpose_w is false, n must also be even.
* In MXFP4/MXFP8 scenarios, weight(w) transposition is supported. The transposition property of w must be consistent with that of scale.
* e must be less than or equal to 1024.
* In the MXFP4 scenario, k cannot be equal to 2.
*/
// clang-format on
#ifndef OPS_PROTO_DEF_GROUPEDMATMULFINALIZEROUTING
#define OPS_PROTO_DEF_GROUPEDMATMULFINALIZEROUTING
// Keep the shared guard used by ops_proto_legacy.h. The HIFLOAT8 input
// types follow the existing canndev fusion_ops.h definition.
REG_OP(GroupedMatmulFinalizeRouting)
    .INPUT(x, TensorType({DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8}))
    .INPUT(w, TensorType({DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1, DT_HIFLOAT8}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT, DT_INT64, DT_BF16, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(pertoken_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
    .OPTIONAL_INPUT(group_list, TensorType({DT_INT64}))
    .OPTIONAL_INPUT(shared_input, TensorType({DT_BF16}))
    .OPTIONAL_INPUT(logit, TensorType({DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(row_index, TensorType({DT_INT64, DT_INT32}))
    .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_BF16}))
    .ATTR(dtype, Int, 0)
    .ATTR(shared_input_weight, Float, 1.0)
    .ATTR(shared_input_offset, Int, 0)
    .ATTR(transpose_x, Bool, false)
    .ATTR(transpose_w, Bool, false)
    .ATTR(output_bs, Int, 0)
    .ATTR(group_list_type, Int, 1)
    .ATTR(tuning_config, ListInt, {0})
    .OP_END_FACTORY_REG(GroupedMatmulFinalizeRouting)
#endif // OPS_PROTO_DEF_GROUPEDMATMULFINALIZEROUTING

} // namespace ge

#endif
