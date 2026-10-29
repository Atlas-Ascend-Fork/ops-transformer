# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch adapter for PermuteDiv."""

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import OpBuilder, get_as_library


class PermuteDivOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("permute_div", category="moe")

    def sources(self):
        return ["csrc/moe/permute_div.cpp"]

    def schema(self):
        return "permute_div(Tensor indice, int divisor) -> Tensor"

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def permute_div_meta(indice, divisor):
            if indice.dim() != 1:
                raise ValueError("indice rank must be 1")
            if divisor < 1:
                raise ValueError("divisor must be positive")
            return torch.empty(indice.shape, device="meta", dtype=indice.dtype)


_permute_div_op_builder = PermuteDivOpBuilder()
_permute_div_op_builder._ensure_initialized()


@impl(get_as_library(), _permute_div_op_builder.name, "PrivateUse1")
def _permute_div(indice: torch.Tensor, divisor: int) -> torch.Tensor:
    module = _permute_div_op_builder.load()
    return module.permute_div(indice, divisor)


def permute_div(indice: torch.Tensor, divisor: int) -> torch.Tensor:
    """PermuteDiv, 对一维 int32 序列做 floor(indice / divisor)。

    Args:
        indice (Tensor): 一维 int32 序列。
        divisor (int): 除数, 必须大于 0。

    Returns:
        Tensor: 与 indice 同 shape、同 dtype 的输出。
    """
    return torch.ops.cann_ops_transformer.permute_div(indice, divisor)
