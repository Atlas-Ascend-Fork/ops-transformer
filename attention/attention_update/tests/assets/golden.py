#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
attention_update golden
"""

__golden__ = {"kernel": {"attention_update": "attention_update_golden"}}


def attention_update_golden(lse, go, *, update_type=0, **kwargs):
    """
    Golden function for attention_update.
    All the parameters (names and order) follow @attention_update_def.cpp without outputs.
    lse / go are TensorList (ParamType=DYNAMIC) inputs: nested sequences of numpy.ndarray.
    update_type / sp are attributes.

    Args:
        lse: TensorList of lse numpy.ndarray, shape (total_length,), float32
        go: TensorList of localOut numpy.ndarray, shape (total_length, hd), float32/float16/bfloat16
        **kwargs: {input,output}_{dtypes,ori_shapes,formats,ori_formats},
                  full_soc_version, short_soc_version, testcase_name, sp

    Returns:
        (output, lse_m)
    """
    import torch
    from ttk.utilities import numpy_to_torch_tensor
    from ttk.utilities import torch_to_numpy_tensor

    torch.set_printoptions(threshold=torch.inf)

    dtype = go[0].dtype
    dtype_str = str(dtype)

    all_lse = []
    all_localOut = []

    for x_lse in lse:
        lse_t = numpy_to_torch_tensor(x_lse)
        all_lse.append(lse_t)

    for x_out in go:
        localOut = numpy_to_torch_tensor(x_out)
        if dtype_str != "float32":
            localOut = localOut.to(torch.float32)
        all_localOut.append(localOut)

    all_lse = torch.stack(all_lse)
    all_out = torch.stack(all_localOut)

    sp = all_out.shape[0]
    hd = all_out.shape[-1]

    total_length = all_out.shape[1]

    lse_max, _ = torch.max(all_lse, dim=0)
    lse_max_expand = lse_max.unsqueeze(0)
    lse_sub = all_lse - lse_max_expand
    lse_sub_exp = torch.exp(lse_sub)
    lse_sub_exp_sum = torch.sum(lse_sub_exp, dim=0)
    lse_out = lse_max + torch.log(lse_sub_exp_sum)

    lse_out_expand = lse_out.unsqueeze(0)
    lse_out_expand = all_lse - lse_out_expand

    lse_out_expand = lse_out_expand.unsqueeze(2)
    lse_out_expand = lse_out_expand.expand(sp, total_length, hd)
    lse_out_expand = torch.exp(lse_out_expand)

    prod_per_sp = all_out * lse_out_expand
    all_out = torch.sum(prod_per_sp, dim=0)

    if update_type == 0:
        lse_out = torch.ones(0)
    if dtype_str == "float16":
        all_out = all_out.to(torch.float16)
    elif dtype_str == "bfloat16":
        all_out = all_out.to(torch.bfloat16)

    all_out = torch_to_numpy_tensor(all_out)
    lse_out = torch_to_numpy_tensor(lse_out)

    return all_out, lse_out
