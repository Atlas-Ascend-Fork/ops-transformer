/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Torch bridge for MergeChunk2.
#include <acl/acl.h>
#include <torch/extension.h>
#include <tuple>

#include "aclnn_common.h"

namespace op_api {
std::tuple<at::Tensor, at::Tensor, at::Tensor> MergeChunk2(
    const at::Tensor& input1_chunk1, const at::Tensor& input1_chunk2, const at::Tensor& input2_chunk1,
    const at::Tensor& input2_chunk2, const at::Tensor& group_list_chunk1, const at::Tensor& group_list_chunk2)
{
    TORCH_CHECK(input1_chunk1.device().type() == at::kPrivateUse1, "input1_chunk1 must be on NPU");
    TORCH_CHECK(input1_chunk2.device() == input1_chunk1.device() && input2_chunk1.device() == input1_chunk1.device() &&
                    input2_chunk2.device() == input1_chunk1.device() &&
                    group_list_chunk1.device() == input1_chunk1.device() &&
                    group_list_chunk2.device() == input1_chunk1.device(),
                "all inputs must be on the same device");
    TORCH_CHECK(input1_chunk1.is_contiguous() && input1_chunk2.is_contiguous() && input2_chunk1.is_contiguous() &&
                    input2_chunk2.is_contiguous() && group_list_chunk1.is_contiguous() &&
                    group_list_chunk2.is_contiguous(),
                "all inputs should be contiguous");
    TORCH_CHECK(
        input1_chunk1.dim() == 2 && input1_chunk2.dim() == 2 && input2_chunk1.dim() == 2 && input2_chunk2.dim() == 2,
        "input1_chunk1/input1_chunk2/input2_chunk1/input2_chunk2 rank must be 2");
    TORCH_CHECK(group_list_chunk1.dim() == 1 && group_list_chunk2.dim() == 1,
                "group_list_chunk1/group_list_chunk2 rank must be 1");
    TORCH_CHECK(input1_chunk1.size(0) == input2_chunk1.size(0),
                "input1_chunk1 and input2_chunk1 must have the same token num");
    TORCH_CHECK(input1_chunk2.size(0) == input2_chunk2.size(0),
                "input1_chunk2 and input2_chunk2 must have the same token num");
    TORCH_CHECK(input1_chunk1.size(1) == input1_chunk2.size(1),
                "input1_chunk1 and input1_chunk2 must have the same hidden size");
    TORCH_CHECK(input2_chunk1.size(1) == input2_chunk2.size(1),
                "input2_chunk1 and input2_chunk2 must have the same hidden size");
    TORCH_CHECK(group_list_chunk1.size(0) == group_list_chunk2.size(0),
                "group_list_chunk1 and group_list_chunk2 must have the same num expert");
    TORCH_CHECK(input1_chunk1.scalar_type() == input1_chunk2.scalar_type() &&
                    input1_chunk1.scalar_type() == input2_chunk1.scalar_type() &&
                    input1_chunk1.scalar_type() == input2_chunk2.scalar_type(),
                "all data inputs must have the same dtype");
    TORCH_CHECK(input1_chunk1.scalar_type() == at::kHalf || input1_chunk1.scalar_type() == at::kBFloat16,
                "data inputs dtype must be float16 or bfloat16");
    TORCH_CHECK(group_list_chunk1.scalar_type() == at::kInt && group_list_chunk2.scalar_type() == at::kInt,
                "group_list_chunk1/group_list_chunk2 dtype must be int32");

    at::Tensor out1{nullptr};
    at::Tensor out2{nullptr};
    at::Tensor group_list{nullptr};
    {
        // DeviceGuard 必须在申请输出张量之前生效, 否则非默认卡调用时输出会落到错误设备。
        auto local_device = c10::Device(input1_chunk1.device());
        const c10::OptionalDeviceGuard device_guard(local_device);
        out1 =
            at::empty({input1_chunk1.size(0) + input1_chunk2.size(0), input1_chunk1.size(1)}, input1_chunk1.options());
        out2 =
            at::empty({input2_chunk1.size(0) + input2_chunk2.size(0), input2_chunk1.size(1)}, input2_chunk1.options());
        group_list = at::empty({group_list_chunk1.size(0)}, group_list_chunk1.options());
    }

    ACLNN_CMD(aclnnMergeChunk2, input1_chunk1, input1_chunk2, input2_chunk1, input2_chunk2, group_list_chunk1,
              group_list_chunk2, out1, out2, group_list);
    return {out1, out2, group_list};
}
} // namespace op_api

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("merge_chunk2", &op_api::MergeChunk2, "MergeChunk2: merge two chunks by expert");
}
