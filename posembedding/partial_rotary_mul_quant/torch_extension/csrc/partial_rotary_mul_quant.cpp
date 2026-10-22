/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor> PartialRotaryMulQuant(const at::Tensor& x, const at::Tensor& cos,
                                                         const at::Tensor& sin, const c10::optional<at::Tensor>& scale,
                                                         int64_t yDtype, int64_t quantMode,
                                                         const std::string& roundMode, int64_t axis, int64_t rotaryMode,
                                                         at::IntArrayRef partialSlice)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "x must be an NPU tensor");
    const c10::OptionalDeviceGuard deviceGuard(x.device());
    TORCH_CHECK(yDtype == 0, "y_dtype only supports 0 (HiFloat8)");
    TORCH_CHECK(rotaryMode >= 0 && rotaryMode <= 3, "rotary_mode must be 0, 1, 2 or 3");
    TORCH_CHECK(roundMode == "round" || roundMode == "hybird", "round_mode must be round or hybird");
    TORCH_CHECK(x.dim() == 4 && cos.dim() == 4 && sin.dim() == 4, "x, cos and sin must be 4D");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kFloat,
                "x must have dtype float16, bfloat16 or float32");
    TORCH_CHECK(cos.scalar_type() == x.scalar_type() && sin.scalar_type() == x.scalar_type(),
                "x, cos and sin must have the same dtype");
    TORCH_CHECK(cos.device() == x.device() && sin.device() == x.device(),
                "x, cos and sin must be on the same NPU device");
    TORCH_CHECK(cos.sizes() == sin.sizes(), "cos and sin must have the same shape");
    TORCH_CHECK(partialSlice.size() == 2, "partial_slice must contain [start, end]");
    int64_t d = x.size(3);
    int64_t start = partialSlice[0];
    int64_t end = partialSlice[1];
    int64_t divisor = rotaryMode == 2 ? 4 : 2;
    TORCH_CHECK(0 <= start && start < end && end == d && d <= 1024,
                "partial_slice must satisfy 0 <= start < end == D <= 1024");
    TORCH_CHECK(d % divisor == 0 && (end - start) % divisor == 0, "D and rotary slice length must be divisible by ",
                divisor);
    TORCH_CHECK(cos.size(3) == end - start, "cos/sin last dimension must equal rotary slice length");
    for (int64_t i = 0; i < 4; ++i) {
        TORCH_CHECK(x.size(i) > 0 && cos.size(i) > 0, "x, cos and sin must have positive dimensions");
        if (i < 3) {
            TORCH_CHECK(cos.size(i) == 1 || cos.size(i) == x.size(i), "cos/sin leading dimensions must broadcast to x");
        }
    }
    if (scale.has_value()) {
        TORCH_CHECK(scale->device() == x.device(), "scale must be on the same NPU device as x");
        TORCH_CHECK(scale->scalar_type() == at::kFloat && scale->numel() == 1,
                    "scale must be a single-element FP32 tensor");
    }
    // PyTorch stores HiFloat8 as bytes; pass its real dtype to ACLNN.
    at::Tensor y = at::empty(x.sizes(), x.options().dtype(at::kByte));
    at::Tensor yScale = at::empty({0}, x.options().dtype(at::kFloat));
    TensorWrapper yWrapper{y, ACL_HIFLOAT8};
    const char* roundModePtr = roundMode.c_str();
    ACLNN_CMD(aclnnPartialRotaryMulQuant, x, cos, sin, scale, yDtype, quantMode, roundModePtr, axis, rotaryMode,
              partialSlice, yWrapper, yScale);
    return std::make_tuple(y, yScale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("partial_rotary_mul_quant", &PartialRotaryMulQuant, "Partial RoPE and HiFloat8 quantization (Ascend950)");
}
} // namespace op_api
