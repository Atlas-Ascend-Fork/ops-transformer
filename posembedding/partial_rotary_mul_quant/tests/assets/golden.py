# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TTK CPU Golden for the Ascend950 HiFloat8 implementation.

RoPE, scaling and HiFloat8 rounding use CPU Torch operations in FP32. The
standalone float32_to_hifloat8 function returns uint8 encoding bytes and supports
ROUND (TA) and HYBRID (TA/SSR). NumPy/en_dtypes are used only at the input/output
boundary. Comparison is delegated to TTK's built-in criteria. GEIR and Torch
E2E adapters expose the same reference with their respective signatures.

Quarter/deepseek_interleave follow the local arch35 kernel. The TA/SSR encoding
follows attention/quant_sparse_flash_mla/tests/pytest/generate_hifloat8_data.py
in ops-transformer. SSR derives F14/T14 from the FP32 mantissa, so conversion is
deterministic for a fixed input; it does not draw independent random numbers.

Accepts NumPy arrays (TTK's HiFloat8 fallback) and CPU Torch tensors. No NPU
runtime, custom_ops, or TTK internal registration is needed by this file.
Inputs are assumed valid; the Golden does not validate operator arguments.
"""

import math

import numpy as np
import torch
from en_dtypes import hifloat8

__spec__ = {
    "partial_rotary_mul_quant": "PartialRotaryMulQuantTestSpec",
    "aclnnPartialRotaryMulQuant": "PartialRotaryMulQuantAclnnTestSpec",
    "torch.ops.cann_ops_transformer.partial_rotary_mul_quant": "PartialRotaryMulQuantE2eTestSpec",
}

# Limit FP32 temporary arrays for the large prefill CSV. The final output still
# has the full declared shape; chunking does not change elementwise arithmetic.
_CHUNK_ELEMENTS = 4 * 1024 * 1024


def _float32_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32)
    # Torch cannot import NumPy's extension BF16/HiFloat8 dtypes directly.
    # This conversion is only the input boundary; arithmetic stays in Torch.
    array = np.asarray(value, dtype=np.float32)
    if any(stride < 0 for stride in array.strides):
        array = array.copy()
    return torch.from_numpy(array)


def _hifloat8_fields(exponent):
    """Return retained fraction bits and the positive zero-fraction encoding."""
    magnitude = exponent.abs()
    negative = (exponent < 0).to(torch.int32)
    fraction_bits = torch.where(magnitude < 4, 3, torch.where(magnitude < 8, 2, 1)).to(
        torch.int32
    )
    fraction_bits = torch.where(exponent < -15, 0, fraction_bits)
    base = torch.where(
        magnitude < 8,
        64 + negative * 16 + (magnitude - 4) * 4,
        96 + negative * 16 + (magnitude - 8) * 2,
    )
    base = torch.where(magnitude < 4, 32 + negative * 16 + (magnitude - 2) * 8, base)
    base = torch.where(magnitude == 1, 16 + negative * 8, base)
    base = torch.where(exponent == 0, 8, base)
    base = torch.where(exponent < -15, exponent + 23, base)
    return fraction_bits, base


def float32_to_hifloat8(value, round_mode="round"):
    """Convert numeric input to CPU Torch uint8 HiFloat8 encoding bytes.

    ROUND uses TA (nearest, ties away from zero). HYBRID uses TA for |E| < 4
    and simplified stochastic rounding (SSR) elsewhere, with E the unbiased
    FP32 exponent. SSR compares the discarded mantissa's F14 and T14 fields;
    F14 >= T14 increments the retained fraction, including at exact values.

    Overflow uses Inf rather than saturation, matching the reference's
    over_mode=True: |x| >= 40960 maps to signed Inf. NaN maps to 0x80 and both
    signs of zero/underflow map to 0x00. The input shape is preserved.
    """
    round_mode = "round" if round_mode in (None, "") else round_mode
    x = _float32_tensor(value).contiguous()
    bits = x.view(torch.int32)
    sign = (bits >> 24) & 0x80
    raw_exponent = ((bits >> 23) & 0xFF) - 127
    # Subnormal FP32 values underflow to HiFloat8 zero. Clipping also makes
    # intermediate fields well-defined for zero, Inf and NaN before masking.
    exponent = raw_exponent.clamp(-23, 15)
    fraction = bits & 0x7FFFFF
    fraction_bits, _ = _hifloat8_fields(exponent)
    shift = 23 - fraction_bits
    retained = fraction >> shift

    # TA: inspect the guard bit and propagate a possible exponent carry.
    rounded = retained + ((fraction >> (shift - 1)) & 1)
    rounded = torch.where(exponent == -23, 1, rounded)
    if round_mode == "hybrid":
        discarded = fraction - (retained << shift)
        f14 = discarded >> (shift - 14)
        t14 = discarded & 0x3FFF
        # At E=-23 the implicit leading one participates in underflow rounding.
        f14 = torch.where(exponent == -23, (fraction >> 10) + 8192, f14)
        t14 = torch.where(exponent == -23, fraction & 0x3FFF, t14)
        ssr = retained + (f14 >= t14).to(torch.int32)
        rounded = torch.where(exponent.abs() < 4, rounded, ssr)

    carry = rounded >= (1 << fraction_bits)
    exponent = exponent + carry.to(torch.int32)
    rounded = torch.where(carry, 0, rounded)
    _, base = _hifloat8_fields(exponent)
    encoded = sign + base + rounded
    encoded = torch.where((raw_exponent < -23) | (exponent <= -23), 0, encoded)
    encoded = torch.where(x.abs() >= 40960, sign + 0x6F, encoded)
    encoded = torch.where(torch.isnan(x), 0x80, encoded)
    return encoded.to(torch.uint8)


def _trig_chunk(value, start, stop):
    return _float32_tensor(value if value.shape[0] == 1 else value[start:stop])


def _rope_chunk(x, cosine, sine, start, end, mode):
    result = _float32_tensor(x).clone()
    if start == end:
        return result
    tail = result[..., start:end]
    if mode == 1:
        rotated = torch.empty_like(tail)
        rotated[..., ::2] = -tail[..., 1::2]
        rotated[..., 1::2] = tail[..., ::2]
        rope_result = cosine * tail + rotated * sine
    elif mode in (0, 3):
        half = (end - start) // 2
        if mode == 3:
            # Deinterleave x, but keep cos/sin in contiguous halves. The
            # output is concatenated halves, unlike mode 1's even/odd output.
            x1, x2 = tail[..., ::2], tail[..., 1::2]
        else:
            x1, x2 = tail[..., :half], tail[..., half:]
        rope_result = torch.cat(
            (
                x1 * cosine[..., :half] - x2 * sine[..., :half],
                x2 * cosine[..., half:] + x1 * sine[..., half:],
            ),
            dim=-1,
        )
    elif mode == 2:
        # Two independent half rotations, one per half of the tail.
        quarter = (end - start) // 4
        rotated = torch.empty_like(tail)
        rotated[..., :quarter] = -tail[..., quarter : 2 * quarter]
        rotated[..., quarter : 2 * quarter] = tail[..., :quarter]
        rotated[..., 2 * quarter : 3 * quarter] = -tail[..., 3 * quarter :]
        rotated[..., 3 * quarter :] = tail[..., 2 * quarter : 3 * quarter]
        rope_result = cosine * tail + rotated * sine
    result[..., start:end] = rope_result
    return result


def _fp32_chunks(x, cos, sin, start, end, mode, scale):
    rows_per_chunk = max(1, _CHUNK_ELEMENTS // math.prod(x.shape[1:]))
    for first in range(0, x.shape[0], rows_per_chunk):
        last = min(x.shape[0], first + rows_per_chunk)
        result = _rope_chunk(
            x[first:last],
            _trig_chunk(cos, first, last),
            _trig_chunk(sin, first, last),
            start,
            end,
            mode,
        )
        # Scale the entire D, including the non-RoPE prefix, in FP32.
        result.mul_(scale)
        yield first, last, result


class PartialRotaryMulQuantAclnnTestSpec:
    """Torch Golden with native HiFloat8 output and TTK built-in comparison."""

    # HiFloat8 uses TTK's default requant; the empty FP32 scale is binary equal.
    tolerance = {"float32": {"standard": "binary_equal"}}

    @staticmethod
    def golden(
        x,
        cos,
        sin,
        scaleOptional=None,
        yDtype=0,
        quantMode=0,
        roundModeOptional="round",
        axis=-1,
        rotaryMode=0,
        partialSliceOptional=None,
        yOut=None,
        yScaleOut=None,
        **kwargs,
    ):
        """Parameters follow aclnnPartialRotaryMulQuantGetWorkspaceSize.

        Pure output templates and TTK metadata are unused. quantMode and axis
        are ignored by the HiFloat8 kernel. Inverse RoPE is supplied as -sin;
        the Golden must not negate it a second time.

        All four rotary modes and round/hybrid are supported for HiFloat8.
        Both modes return their actual encoding reference for TTK's built-in
        HiFloat8 comparison.
        """
        shape = tuple(x.shape)
        bounds = [] if partialSliceOptional is None else list(partialSliceOptional)
        start, end = bounds or (0, 0)
        scale = (
            torch.tensor(1.0, dtype=torch.float32)
            if scaleOptional is None
            else _float32_tensor(scaleOptional).reshape(())
        )
        output = torch.empty(shape, dtype=torch.uint8)
        for first, last, result in _fp32_chunks(
            x, cos, sin, start, end, rotaryMode, scale
        ):
            output[first:last] = float32_to_hifloat8(result, roundModeOptional)
        # Torch has no native HiFloat8 dtype. Reinterpret the encoding only;
        # astype(hifloat8) here would numerically convert the uint8 byte values.
        return [
            output.numpy().view(hifloat8),
            torch.empty(0, dtype=torch.float32).numpy(),
        ]


class PartialRotaryMulQuantTestSpec:
    """GEIR/Kernel adapter returning native HiFloat8 outputs."""

    tolerance = PartialRotaryMulQuantAclnnTestSpec.tolerance

    @staticmethod
    def golden(
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
        **kwargs,
    ):
        return PartialRotaryMulQuantAclnnTestSpec.golden(
            x,
            cos,
            sin,
            scale,
            yDtype=y_dtype,
            quantMode=quant_mode,
            roundModeOptional=round_mode,
            axis=axis,
            rotaryMode=rotary_mode,
            partialSliceOptional=partial_slice,
        )


class PartialRotaryMulQuantE2eTestSpec:
    """Adapt the ACLNN reference to Torch's snake_case API and HiFloat8 bytes."""

    # uint8 encoding bytes use TTK's default binary comparison in the E2E path.
    tolerance = PartialRotaryMulQuantAclnnTestSpec.tolerance

    @staticmethod
    def golden(
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
        **kwargs,
    ):
        y, y_scale = PartialRotaryMulQuantAclnnTestSpec.golden(
            x,
            cos,
            sin,
            scale,
            yDtype=y_dtype,
            quantMode=quant_mode,
            roundModeOptional=round_mode,
            axis=axis,
            rotaryMode=rotary_mode,
            partialSliceOptional=partial_slice,
        )
        # Torch exposes the encoding as uint8, not numeric integer values.
        return [y.view(np.uint8), y_scale]
