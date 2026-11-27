#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import struct

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset


def tsoftmax(x):
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    x_sub = x.sub(x_max)
    y = torch.exp(x_sub)
    x_sum = y.sum(dim=-1, keepdim=True)
    ans = y.div(x_sum)
    return ans, x_max, x_sum


def floyd_attn_forward(Q, K1, K2, V1, V2, atten_mask, scale):
    fp32_bits = 0xFF7FFFFF
    float_value = struct.unpack(">f", struct.pack(">I", fp32_bits))[0]
    dtype = Q.dtype
    if dtype != torch.float32:
        golden_dtype = torch.float32
    else:
        golden_dtype = torch.float64
    s = torch.einsum("bhikc,bhijc->bhikj", Q, K1) + torch.einsum(
        "bhikc,bhjkc->bhikj", Q, K2
    )

    s = s.to(golden_dtype)
    s = s * scale
    if atten_mask != None:
        s = s + atten_mask.bool() * float_value
    p, x_max, x_sum = tsoftmax(s)

    p = p.to(dtype)
    output = torch.einsum("bhikj,bhijc->bhikc", p, V1) + torch.einsum(
        "bhikj,bhjkc->bhikc", p, V2
    )

    output = output.to(dtype)
    x_max = x_max.to(golden_dtype)
    x_sum = x_sum.to(golden_dtype)
    return output, x_max, x_sum


def floyd_attn_backward(
    Q, K1, K2, V1, V2, grad, atten_mask, x_max, x_sum, atten_in, scale
):
    fp32_bits = 0xFF7FFFFF
    float_value = struct.unpack(">f", struct.pack(">I", fp32_bits))[0]
    dtype = Q.dtype
    if dtype != torch.float32:
        golden_dtype = torch.float32
    else:
        golden_dtype = torch.float64
    Q = Q.to(golden_dtype)
    K1 = K1.to(golden_dtype)
    V1 = V1.to(golden_dtype)
    K2 = K2.to(golden_dtype)
    V2 = V2.to(golden_dtype)
    grad = grad.to(golden_dtype)
    atten_in = atten_in.to(golden_dtype)

    s = torch.einsum("bhikc,bhijc->bhikj", Q.to(dtype), K1.to(dtype)).to(
        golden_dtype
    ) + torch.einsum("bhikc,bhjkc->bhikj", Q.to(dtype), K2.to(dtype)).to(golden_dtype)
    dp = torch.einsum("bhikc,bhijc->bhikj", grad.to(dtype), V1.to(dtype)).to(
        golden_dtype
    ) + torch.einsum("bhikc,bhjkc->bhikj", grad.to(dtype), V2.to(dtype)).to(
        golden_dtype
    )
    s = s.to(golden_dtype)
    dp = dp.to(golden_dtype)

    s = s * scale
    if atten_mask != None:
        s = s + atten_mask.bool() * float_value
    p = torch.exp(s - x_max[:, :, :, :, 0:1]) / x_sum[:, :, :, :, 0:1]
    ds = p * (dp - (grad * atten_in).sum(dim=-1, keepdim=True)) * scale

    ds = ds.to(dtype)
    p = p.to(dtype)
    dQ = torch.einsum("bhikj,bhijc->bhikc", ds.to(dtype), K1.to(dtype)).to(
        golden_dtype
    ) + torch.einsum("bhikj,bhjkc->bhikc", ds.to(dtype), K2.to(dtype)).to(golden_dtype)
    dK1 = torch.einsum("bhikj,bhikc->bhijc", ds.to(dtype), Q.to(dtype)).to(golden_dtype)
    dK2 = torch.einsum("bhikj,bhikc->bhjkc", ds.to(dtype), Q.to(dtype)).to(golden_dtype)
    dV1 = torch.einsum("bhikj,bhikc->bhijc", p.to(dtype), grad.to(dtype)).to(
        golden_dtype
    )
    dV2 = torch.einsum("bhikj,bhikc->bhjkc", p.to(dtype), grad.to(dtype)).to(
        golden_dtype
    )

    dQ = dQ.to(dtype)
    dK1 = dK1.to(dtype)
    dK2 = dK2.to(dtype)
    dV1 = dV1.to(dtype)
    dV2 = dV2.to(dtype)
    return dQ, dK1, dV1, dK2, dV2


@register("ascend_function_ffag")
class AscendFunctionFusedFloydAttentionGrad(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(AscendFunctionFusedFloydAttentionGrad, self).__init__(task_result)
        OpsDataset.seed_everything()

    def init_by_input_data(self, input_data: InputDataset):
        Q = input_data.kwargs["query"]
        K1 = input_data.kwargs["key1"]
        V1 = input_data.kwargs["value1"]
        K2 = input_data.kwargs["key2"]
        V2 = input_data.kwargs["value2"]
        atten_mask = input_data.kwargs["atten_mask"]
        scale = input_data.kwargs["scale"]

        device = "cpu"
        origin_q_dtype = Q.dtype
        origin_x_max_dtype = input_data.kwargs["softmax_max"].dtype
        golden_dtype = torch.float64
        Q = Q.to(golden_dtype)
        K1 = K1.to(golden_dtype)
        V1 = V1.to(golden_dtype)
        K2 = K2.to(golden_dtype)
        V2 = V2.to(golden_dtype)

        # 获取正向输出，作为反向输入
        atten_in, x_max, x_sum = floyd_attn_forward(
            Q.to(device),
            K1.to(device),
            K2.to(device),
            V1.to(device),
            V2.to(device),
            atten_mask.to(device),
            scale,
        )
        x_max = x_max.repeat(1, 1, 1, 1, 8)
        x_sum = x_sum.repeat(1, 1, 1, 1, 8)

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu" or self.device == "pyaclnn":
            device = f"npu:{self.device_id}"
        else:
            device = "cpu"

        # 数据类型一定要转换正确
        x_max = x_max.to(origin_x_max_dtype)
        x_sum = x_sum.to(origin_x_max_dtype)
        atten_in = atten_in.to(origin_q_dtype)

        # 重新赋值
        input_data.kwargs["softmax_max"] = x_max
        input_data.kwargs["softmax_sum"] = x_sum
        input_data.kwargs["atten_in"] = atten_in

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        Q = input_data.kwargs["query"]
        K1 = input_data.kwargs["key1"]
        V1 = input_data.kwargs["value1"]
        K2 = input_data.kwargs["key2"]
        V2 = input_data.kwargs["value2"]
        grad = input_data.kwargs["dy"]
        atten_mask = input_data.kwargs["atten_mask"]
        x_max = input_data.kwargs["softmax_max"]
        x_sum = input_data.kwargs["softmax_sum"]
        atten_in = input_data.kwargs["atten_in"]
        scale = input_data.kwargs["scale"]

        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        # 反向标杆
        dQ, dK1, dV1, dK2, dV2 = floyd_attn_backward(
            Q.to(device),
            K1.to(device),
            K2.to(device),
            V1.to(device),
            V2.to(device),
            grad.to(device),
            atten_mask.to(device),
            x_max.to(device),
            x_sum.to(device),
            atten_in.to(device),
            scale,
        )
        dQ = dQ.contiguous()
        dK1 = dK1.contiguous()
        dV1 = dV1.contiguous()
        dK2 = dK2.contiguous()
        dV2 = dV2.contiguous()

        return dQ, dK1, dV1, dK2, dV2
