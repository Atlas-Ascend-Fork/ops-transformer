# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List, Optional, Tuple

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


def _validate(x, cos, sin, scale, y_dtype, round_mode, rotary_mode, partial_slice):
    if y_dtype != 0:
        raise ValueError("y_dtype only supports 0 (HiFloat8).")
    if rotary_mode not in (0, 1, 2, 3):
        raise ValueError("rotary_mode must be 0, 1, 2 or 3.")
    if round_mode not in ("round", "hybird"):
        raise ValueError("round_mode must be 'round' or 'hybird' (CANN spelling).")
    if x.dim() != 4 or cos.dim() != 4 or sin.dim() != 4:
        raise ValueError("x, cos and sin must be 4D tensors.")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("x must have dtype float16, bfloat16 or float32.")
    if cos.dtype != x.dtype or sin.dtype != x.dtype:
        raise ValueError("x, cos and sin must have the same dtype.")
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have the same shape.")
    for tensor in (x, cos, sin):
        if any(size <= 0 for size in tensor.shape):
            raise ValueError("x, cos and sin must have positive dimensions.")
    if partial_slice is None or len(partial_slice) != 2:
        raise ValueError("partial_slice must contain [start, end].")
    start, end = partial_slice
    d = x.size(-1)
    divisor = 4 if rotary_mode == 2 else 2
    if not (0 <= start < end == d) or d > 1024:
        raise ValueError("partial_slice must satisfy 0 <= start < end == D <= 1024.")
    if d % divisor != 0 or (end - start) % divisor != 0:
        raise ValueError(f"D and rotary slice length must be divisible by {divisor}.")
    if cos.size(-1) != end - start:
        raise ValueError("cos/sin last dimension must equal the rotary slice length.")
    for actual, expected in zip(cos.shape[:3], x.shape[:3]):
        if actual != 1 and actual != expected:
            raise ValueError("cos/sin leading dimensions must broadcast to x.")
    if scale is not None and (scale.dtype != torch.float32 or scale.numel() != 1):
        raise ValueError("scale must be a single-element FP32 tensor.")


class PartialRotaryMulQuantOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("partial_rotary_mul_quant", category="posembedding")

    def sources(self):
        return ["csrc/posembedding/partial_rotary_mul_quant.cpp"]

    def schema(self):
        return (
            "partial_rotary_mul_quant(Tensor x, Tensor cos, Tensor sin, "
            "Tensor? scale=None, int y_dtype=0, int quant_mode=0, "
            'str round_mode="round", int axis=-1, int rotary_mode=0, '
            "int[]? partial_slice=None) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def partial_rotary_mul_quant_meta(
            x,
            cos,
            sin,
            scale=None,
            y_dtype=0,
            quant_mode=0,
            round_mode="round",
            axis=-1,
            rotary_mode=0,
            partial_slice=None,
        ):
            _validate(
                x, cos, sin, scale, y_dtype, round_mode, rotary_mode, partial_slice
            )
            return (
                x.new_empty(x.shape, dtype=torch.uint8),
                x.new_empty((0,), dtype=torch.float32),
            )


partial_rotary_mul_quant_op_builder = PartialRotaryMulQuantOpBuilder()
partial_rotary_mul_quant_op_builder._ensure_initialized()
# Compile before Dynamo tracing or ACL graph capture begins.
partial_rotary_mul_quant_op_builder.load()


@impl(get_as_library(), partial_rotary_mul_quant_op_builder.name, "PrivateUse1")
def _partial_rotary_mul_quant(
    x,
    cos,
    sin,
    scale=None,
    y_dtype=0,
    quant_mode=0,
    round_mode="round",
    axis=-1,
    rotary_mode=0,
    partial_slice=None,
):
    if partial_slice is None:
        raise ValueError("partial_slice must contain [start, end].")
    return partial_rotary_mul_quant_op_builder.load().partial_rotary_mul_quant(
        x,
        cos,
        sin,
        scale,
        y_dtype,
        quant_mode,
        round_mode,
        axis,
        rotary_mode,
        partial_slice,
    )


def partial_rotary_mul_quant(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    y_dtype: int = 0,
    quant_mode: int = 0,
    round_mode: str = "round",
    axis: int = -1,
    rotary_mode: int = 0,
    partial_slice: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Partial RoPE followed by HiFloat8 quantization on Ascend950.

    x/cos/sin use FP16, BF16 or FP32, with four dimensions. cos/sin broadcast
    over x's leading dimensions and cover only [start, end). partial_slice is
    required; end must equal D (<=1024). rotary_mode is 0 (half), 1 (interleave),
    2 (quarter), or 3 (interleave-half). D and slice length must be multiples
    of 2, or 4 for quarter mode. scale optionally contains one FP32 value and
    multiplies the entire result, including the unrotated prefix.

    y_dtype must be 0. round_mode accepts 'round' and 'hybird' (CANN spelling).
    quant_mode and axis are retained for ACLNN compatibility and ignored.
    Returns HiFloat8 bytes as uint8 with x's shape and an empty FP32 y_scale
    of shape (0,). This inference operator does not implement backward.
    """
    return torch.ops.cann_ops_transformer.partial_rotary_mul_quant(
        x,
        cos,
        sin,
        scale,
        y_dtype,
        quant_mode,
        round_mode,
        axis,
        rotary_mode,
        partial_slice,
    )
