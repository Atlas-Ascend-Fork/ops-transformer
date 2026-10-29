/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Torch bridge for PermuteDiv.
#include <acl/acl.h>
#include <torch/extension.h>

#include "aclnn_common.h"

namespace op_api {
at::Tensor PermuteDiv(const at::Tensor& indice, int64_t divisor)
{
    TORCH_CHECK(indice.device().type() == at::kPrivateUse1, "indice must be on NPU");
    TORCH_CHECK(indice.is_contiguous(), "indice should be contiguous");
    TORCH_CHECK(indice.dim() == 1, "indice rank must be 1");
    TORCH_CHECK(indice.scalar_type() == at::kInt, "indice dtype must be int32");
    TORCH_CHECK(divisor > 0, "divisor must be positive");

    at::Tensor out{nullptr};
    {
        // DeviceGuard 必须在申请输出张量之前生效, 否则非默认卡调用时输出会落到错误设备。
        auto local_device = c10::Device(indice.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        out = at::empty(indice.sizes(), indice.options());
    }

    ACLNN_CMD(aclnnPermuteDiv, indice, divisor, out);
    return out;
}
} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("permute_div", &op_api::PermuteDiv, "PermuteDiv: floor(indice / divisor)");
}
