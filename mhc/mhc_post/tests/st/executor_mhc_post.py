# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from atk.configs.dataset_config import InputDataset

from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi


@register("execute_aclnn_mhc_post")
class FunctionApi(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):
        # 获取原始张量
        x = input_data.kwargs.get("x")
        h_res = input_data.kwargs.get("h_res")
        h_out = input_data.kwargs.get("h_out")
        h_post = input_data.kwargs.get("h_post")

        # 记录原始信息
        orig_dtype = x.dtype
        device = x.device
        out_shape = list(x.shape)

        # 计算输出
        # Scale h_out by each post weight.
        term1 = h_post.to(device).unsqueeze(-1) * h_out.to(device).unsqueeze(-2)
        # Reduce source heads to obtain each destination head.
        term2 = torch.sum(
            h_res.to(device).unsqueeze(-1) * x.to(device).unsqueeze(-2),
            dim=x.to(device).dim() - 2,
        )

        out = term1 + term2

        # 转回原始数据类型和设备，并 reshape
        out = out.to(dtype=orig_dtype, device=device)
        out = out.reshape(out_shape)
        return out
