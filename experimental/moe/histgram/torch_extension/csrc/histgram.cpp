/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Torch bridge for Histgram.
#include <acl/acl.h>
#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
at::Tensor Histgram(const at::Tensor& input, int64_t num_expert)
{
    TORCH_CHECK(input.device().type() == at::kPrivateUse1, "input must be on NPU");
    TORCH_CHECK(input.is_contiguous(), "input should be contiguous");
    TORCH_CHECK(input.dim() == 1, "input rank must be 1, i.e. [num_token]");
    TORCH_CHECK(input.scalar_type() == at::kInt, "input dtype must be int32");
    TORCH_CHECK(num_expert > 0, "num_expert must be positive");

    at::Tensor out{nullptr};
    {
        // DeviceGuard 必须在申请输出张量之前生效, 否则非默认卡调用时输出会落到错误设备。
        auto local_device = c10::Device(input.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        out = at::empty({num_expert}, input.options());
    }

    ACLNN_CMD(aclnnHistgram, input, num_expert, out);
    return out;
}
} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("histgram", &op_api::Histgram, "Histgram: histogram of token ids over experts");
}
