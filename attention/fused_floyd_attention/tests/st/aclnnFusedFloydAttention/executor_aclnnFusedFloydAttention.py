#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import struct
import torch

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset

SUPPORT_DTYPE = (torch.bfloat16, torch.float16)


def _batched_mm(a, b):
    """Batched matmul via torch.bmm. a: (N,M,K), b: (N,K,P) -> (N,M,P)"""
    if a.device.type != "cpu" and a.dtype in SUPPORT_DTYPE:
        return torch.bmm(a, b, out_dtype=torch.float32)
    return torch.bmm(a, b)


def _mm_qk1(q, k1):
    """einsum('bhikc,bhijc->bhikj', q, k1)"""
    b, h, i, k, c = q.shape
    j = k1.shape[3]
    q2 = q.reshape(b * h * i, k, c)
    k1_t = k1.reshape(b * h * i, j, c).transpose(1, 2)
    return _batched_mm(q2, k1_t).reshape(b, h, i, k, j)


def _mm_qk2(q, k2):
    """einsum('bhikc,bhjkc->bhikj', q, k2)"""
    b, h, i, k, c = q.shape
    j = k2.shape[2]
    q2 = q.permute(0, 1, 3, 2, 4).reshape(b * h * k, i, c)
    k2_t = k2.permute(0, 1, 3, 2, 4).reshape(b * h * k, j, c).transpose(1, 2)
    return _batched_mm(q2, k2_t).reshape(b, h, k, i, j).permute(0, 1, 3, 2, 4)


def _mm_pv1(p, v1):
    """einsum('bhikj,bhijc->bhikc', p, v1)"""
    b, h, i, k, j = p.shape
    c = v1.shape[4]
    p2 = p.reshape(b * h * i, k, j)
    v1_2 = v1.reshape(b * h * i, j, c)
    return _batched_mm(p2, v1_2).reshape(b, h, i, k, c)


def _mm_pv2(p, v2):
    """einsum('bhikj,bhjkc->bhikc', p, v2)"""
    b, h, i, k, j = p.shape
    c = v2.shape[4]
    p2 = p.permute(0, 1, 3, 2, 4).reshape(b * h * k, i, j)
    v2_2 = v2.permute(0, 1, 3, 2, 4).reshape(b * h * k, j, c)
    return _batched_mm(p2, v2_2).reshape(b, h, k, i, c).permute(0, 1, 3, 2, 4)


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
    golden_dtype = torch.float64 if dtype == torch.float32 else torch.float32

    s = _mm_qk1(Q, K1) + _mm_qk2(Q, K2)

    s = s.to(golden_dtype)
    s = s * scale
    if atten_mask is not None:
        s = s.masked_fill(atten_mask.bool(), float_value)
    p, x_max, x_sum = tsoftmax(s)

    p = p.to(dtype)
    output = _mm_pv1(p, V1) + _mm_pv2(p, V2)

    output = output.to(dtype)
    x_max = x_max.to(golden_dtype)
    x_sum = x_sum.to(golden_dtype)
    return x_max, x_sum, output


@register("ascend_function_ffa")
class AscendFunctionFusedFloydAttention(BaseApi):
    """npu_golden / cpu / gpu 标杆：torch.bmm 实现，注册名对齐 yaml api_type。"""

    def __init__(self, task_result: TaskResult):
        super(AscendFunctionFusedFloydAttention, self).__init__(task_result)
        OpsDataset.seed_everything()

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        Q = input_data.kwargs["query"]
        K1 = input_data.kwargs["key1"]
        V1 = input_data.kwargs["value1"]
        K2 = input_data.kwargs["key2"]
        V2 = input_data.kwargs["value2"]
        atten_mask = input_data.kwargs.get("atten_mask")
        scale = input_data.kwargs["scale"]

        # 精度标杆固定走 CPU torch.bmm：
        device = "cpu"
        atten = atten_mask.to(device) if atten_mask is not None else None
        softmax_max, softmax_sum, attention_out = floyd_attn_forward(
            Q.to(device),
            K1.to(device),
            K2.to(device),
            V1.to(device),
            V2.to(device),
            atten,
            scale,
        )
        # NPU 算子 softmax_max/sum 最后一维为 8 的广播格式
        softmax_max = softmax_max.repeat(1, 1, 1, 1, 8)
        softmax_sum = softmax_sum.repeat(1, 1, 1, 1, 8)
        attention_out = attention_out.to(Q.dtype)
        return softmax_max, softmax_sum, attention_out
