#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Golden reference for InplacePartialRotaryMul (partial rotary position embedding).

对 x 最后一维的 [start, end) 区间（partial_slice）施加旋转位置编码，区间外
元素原样保留：

    y[..., j] = x[..., j] * cos[..., j] + rotate(x)[..., j] * sin[..., j]

rotary_mode（int，缺省 0）决定 rotate 的重排方式：
    0  half        cat(-后半段, 前半段)，torch.chunk(x, 2, -1)
    1  interleave  stack(-x_odd, x_even) 后 reshape（当前算子仅支持此模式）
    2  quarter     cat(-Q2, Q1, -Q4, Q3)，torch.chunk(x, 4, -1)
    3  deepseek    y = cat(x_even, x_odd) * cos + cat(-x_odd, x_even) * sin
    其他           raise ValueError("invalid mode: %d" % mode)

partial_slice（ListInt，缺省 [0, 0]）：
    - start == end（含缺省 [0, 0]）为空切片：不做旋转、x 原样返回。此为算子侧
      语义（tiling 对 sliceLength==0 提前返回；接口文档"当 sliceEnd 和
      sliceStart 相同时，不做旋转位置编码，直接返回"）。原 golden 把 [0, 0] 当
      全量旋转哨兵，与算子实现不一致，本实现按算子语义执行；全量旋转应传
      [0, D]。
    - 其余按 Python/torch 切片语义作用于最后一维（负索引、end 越界钳制）；
      cos/sin 不被切片，其最后一维宽度须等于切片宽度（算子约束），并按 torch
      广播规则与 x 对齐（golden 不做任何 reshape 适配）。

精度策略：x 为 float16/bfloat16 时三个输入一律转 float32 计算、最后一次性
cast 回 x 原 dtype（cos/sin 为 fp64 也降到 fp32）；否则各自保持原 dtype
（中间精度由 torch 类型提升决定），结果 cast 回 x 原 dtype。输出 shape 与
dtype 恒等于 x 的输入 shape 与 dtype。golden 不修改任何输入、返回新建的
C 连续数组（inplace 语义由被测算子承担）。

原 golden 中从未被调用的 rotate_deepseek_every_two（公式与 mode 3 实际实现
不符）为死代码，未移植。

加载（kernel/geir 流程，输入序与属性名跟随 inplace_partial_rotary_mul_def.cpp；于
ops-test-kit 仓库目录下执行，插件路径按仓库相对位置调整）：

    python3 -m ttk kernel -i cases.csv --plugin tests/assets/golden.py
"""

__spec__ = {
    "inplace_partial_rotary_mul": "InplacePartialRotaryMulTestSpec",
}

import numpy as np


def rotate_half(x):
    import torch

    x_chunk = torch.chunk(x, 2, -1)
    return torch.cat((x_chunk[1] * (-1), x_chunk[0]), dim=-1)


def rotate_every_two(x):
    import torch

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    stacked = torch.stack((-x_odd, x_even), dim=-1)
    return stacked.reshape(x.shape)


def rotate_quarter(x):
    import torch

    x_chunk = torch.chunk(x, 4, -1)
    return torch.cat(
        (x_chunk[1] * (-1), x_chunk[0], x_chunk[3] * (-1), x_chunk[2]), dim=-1
    )


def apply_rotary(x, cos, sin, mode):
    import torch

    if mode == 0:
        y = cos * x + rotate_half(x) * sin
    elif mode == 1:
        y = cos * x + rotate_every_two(x) * sin
    elif mode == 2:
        y = cos * x + rotate_quarter(x) * sin
    elif mode == 3:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        input_part1 = torch.cat((x_even, x_odd), dim=-1)
        input_part2 = torch.cat((x_odd * (-1), x_even), dim=-1)
        y = input_part1 * cos + input_part2 * sin
    else:
        raise ValueError("invalid mode: %d" % mode)
    return y


def _numpy_to_torch(arr):
    """ndarray 转 torch 张量；ml_dtypes bfloat16 走 int16 视图（from_numpy 不收）。"""
    import torch

    if arr.dtype.name == "bfloat16":
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


class InplacePartialRotaryMulTestSpec:
    """InplacePartialRotaryMul 测试规范（kernel/geir 流程，numpy 入参）。

    golden 参数：位置输入 x, cos, sin（顺序同 def.cpp），关键字属性
    rotary_mode、partial_slice（缺省同算子声明 0 / (0, 0)）。
    """

    def golden(x, cos, sin, rotary_mode=0, partial_slice=(0, 0), **kwargs):
        import torch

        out_dtype = x.dtype
        if x.dtype.name in ("bfloat16", "float16"):
            x_tensor = torch.from_numpy(x.astype(np.float32))
            cos_tensor = torch.from_numpy(cos.astype(np.float32))
            sin_tensor = torch.from_numpy(sin.astype(np.float32))
        else:
            x_tensor = _numpy_to_torch(x.copy())
            cos_tensor = _numpy_to_torch(cos.copy())
            sin_tensor = _numpy_to_torch(sin.copy())

        start = partial_slice[0]
        end = partial_slice[1]
        if start == end:
            # 空切片（含缺省 [0, 0]）：算子侧不做旋转直接返回
            return [x.copy()]

        # 部分旋转：clone 后仅写回 [start, end) 区间，cos/sin 以全表参与广播
        y = x_tensor.clone()
        y[..., start:end] = apply_rotary(
            x_tensor[..., start:end], cos_tensor, sin_tensor, rotary_mode
        )
        return [y.numpy().astype(out_dtype, copy=False)]
