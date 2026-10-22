/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file partial_rotary_mul_quant_proto.h
 * \brief
 */
#ifndef OPS_OP_PROTO_INC_PARTIAL_ROTARY_MUL_QUANT_OPS_H_
#define OPS_OP_PROTO_INC_PARTIAL_ROTARY_MUL_QUANT_OPS_H_

#include "graph/operator_reg.h"
#include "common/include/op_host/op_const_def_transformer.h"

namespace ge {
/**
 * @brief Partial rotary position embedding with quantization.
 * @par Inputs:
 * @li x: A tensor. Must be one of the following types: float16, float32, bfloat16.
 * @li cos: A tensor for rotary position embedding cos. Must be one of the following types: float16, float32, bfloat16.
 * @li sin: A tensor for rotary position embedding sin. Must be one of the following types: float16, float32, bfloat16.
 * @li scale: Optional. HiFloat8 multiplier: single-element float32 tensor; absent means 1.0.
 * @par Attributes:
 * @li y_dtype: Optional. 0: HiFloat8 (Ascend950), 1: HiFloat4 (Ascend960DT). Defaults to 0.
 * @li quant_mode: Optional. Quantization mode. Defaults to 0.
 * @li round_mode: Optional. HiFloat8 supports "round" and "hybird" (CAST_HYBRID). Defaults to "round".
 * @li axis: Optional. Axis for quantization. Ignored for HiFloat8. Defaults to -1.
 * @li rotary_mode: Optional. Rotary mode. 0-"half", 1-"interleave", 2-"quarter", 3-"deepseek_interleave". Defaults to
 * 0.
 * @li partial_slice: Optional. Slice range [start, end] for partial rotation. Defaults to [0, 0].
 * @par Outputs:
 * @li y: A tensor of type hifloat8 or hifloat4, with the same shape as x.
 * @li y_scale: An empty float32 tensor with shape [0] for HiFloat8; hifloat4_scale for HiFloat4.
 * @attention Constraints:
 * @li Inputs must be 4D tensors.
 */
REG_OP(PartialRotaryMulQuant)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))
    .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))
    .ATTR(y_dtype, Int, 0)
    .ATTR(quant_mode, Int, 0)
    .ATTR(round_mode, String, "round")
    .ATTR(axis, Int, -1)
    .ATTR(rotary_mode, Int, 0)
    .ATTR(partial_slice, ListInt, {0, 0})
    .OUTPUT(y, TensorType({DT_HIFLOAT8, Ops::Transformer::DT_HIFLOAT4}))
    .OUTPUT(y_scale, TensorType({DT_FLOAT, Ops::Transformer::DT_HIFLOAT4_SCALE}))
    .OP_END_FACTORY_REG(PartialRotaryMulQuant)

} // namespace ge

#endif
