# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch adapter for Histgram."""

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import OpBuilder, get_as_library


class HistgramOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("histgram", category="moe")

    def sources(self):
        return ["csrc/moe/histgram.cpp"]

    def schema(self):
        return "histgram(Tensor input, int num_expert) -> Tensor"

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def histgram_meta(input, num_expert):
            if input.dim() != 1:
                raise ValueError("input rank must be 1, i.e. [num_token]")
            if num_expert <= 0:
                raise ValueError("num_expert must be positive")
            return torch.empty((num_expert,), device="meta", dtype=input.dtype)


_histgram_op_builder = HistgramOpBuilder()
_histgram_op_builder._ensure_initialized()


@impl(get_as_library(), _histgram_op_builder.name, "PrivateUse1")
def _histgram(input: torch.Tensor, num_expert: int) -> torch.Tensor:
    module = _histgram_op_builder.load()
    return module.histgram(input, num_expert)


def histgram(input: torch.Tensor, num_expert: int) -> torch.Tensor:
    """Histgram 直方图计算, 封装 aclnnHistgram。

    统计一维 token 序列中 [0, num_expert) 每个 expert 出现的次数:
        out[e] = count(input == e), e in [0, num_expert)

    Args:
        input (Tensor): 一维 token 序列, shape 为 [num_token], dtype 固定 int32,
            元素取值属于 [0, num_expert)。
        num_expert (int): expert 个数, 需大于 0。

    Returns:
        Tensor: 输出张量, shape 为 [num_expert], dtype 为 int32。
    """
    return torch.ops.cann_ops_transformer.histgram(input, num_expert)
