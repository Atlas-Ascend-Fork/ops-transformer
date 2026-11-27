#!/usr/bin/python
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

"""CPU reference and output validation for TTK cross-checks."""

import re
import numpy as np
import torch

from .golden import cpu_sparse_flash_attention, mixed_reference

TOLERANCE = {
    dtype: {"standard": "cross_check", "level": "L2"}
    for dtype in ("float16", "bfloat16", "float32")
}


def _with_torch_stats(outputs):
    """Match the Torch API's empty FP32 statistics when LSE is disabled."""
    attention, maximum, total = outputs
    if maximum is None and total is None:
        return (
            attention,
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
        )
    return outputs


def torch_golden(*args, testcase_name=None, **kwargs):
    outputs = cpu_sparse_flash_attention(*args, testcase_name=testcase_name, **kwargs)
    return _with_torch_stats(outputs)


class TorchReference:
    uses_torch_stats = True

    def __call__(self, query, key, value, sparse_indices, scale_value, **kwargs):
        if query.device.type != "cpu":
            raise ValueError(
                f"third-party reference requires CPU inputs, got {query.device}"
            )
        outputs = mixed_reference(
            query, key, value, sparse_indices, scale_value, **kwargs
        )
        for index, output in enumerate(outputs):
            if output is not None and torch.isnan(output).any():
                raise ValueError(f"third-party output {index} contains NaN")
        return _with_torch_stats(outputs) if self.uses_torch_stats else outputs


class AclnnTorchReference(TorchReference):
    uses_torch_stats = False

    def __call__(self, query, **kwargs):
        arguments = {}
        for name, value in kwargs.items():
            base = name.removesuffix("Optional")
            normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", base).lower()
            if normalized in arguments:
                raise ValueError(f"duplicate ACLNN reference argument: {normalized}")
            arguments[normalized] = value
        return super().__call__(query, **arguments)


def as_float32(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().float().numpy()
    return np.asarray(value, dtype=np.float32)


def _is_empty_stat(value):
    return (
        isinstance(value, (torch.Tensor, np.ndarray))
        and value.shape == (0,)
        and value.dtype in (torch.float32, np.dtype("float32"))
    )


def pre_compare(*outputs, uses_torch_stats=True):
    count = 3
    if len(outputs) != count * 2:
        raise ValueError(f"expected {count} target and {count} golden outputs")
    for index, (actual, expected) in enumerate(zip(outputs[:count], outputs[count:])):
        if index > 0 and (
            expected is None or (uses_torch_stats and expected.shape == (0,))
        ):
            if uses_torch_stats and (
                not _is_empty_stat(actual)
                or (expected is not None and not _is_empty_stat(expected))
            ):
                raise ValueError(
                    f"disabled output {index} requires an empty FP32 vector"
                )
            continue
        if actual is None or expected is None:
            raise ValueError(f"output {index} is missing")
        actual, expected = as_float32(actual), as_float32(expected)
        if actual.shape != expected.shape:
            raise ValueError(
                f"output {index} shape mismatch: {actual.shape} vs {expected.shape}"
            )
        if np.isnan(actual).any() or np.isnan(expected).any():
            raise ValueError(f"output {index} contains NaN")
    return outputs


def pre_compare_aclnn(*outputs):
    return pre_compare(*outputs, uses_torch_stats=False)
