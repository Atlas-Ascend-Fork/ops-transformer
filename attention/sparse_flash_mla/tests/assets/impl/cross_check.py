# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TTK L2 comparison and CPU ordinary reference for SparseFlashMla."""

import logging
import re

import torch
from smla_ttk_golden import cpu_tensor, mixed_reference

logger = logging.getLogger(__name__)

TOLERANCE = {
    dtype: {"standard": "cross_check", "level": "L2"}
    for dtype in ("float16", "bfloat16", "float32")
}


def aclnn_names(arguments):
    return {
        re.sub(
            r"([a-z0-9])([A-Z])", r"\1_\2", name.removesuffix("Optional")
        ).lower(): value
        for name, value in arguments.items()
    }


class TorchReference:
    def __call__(self, q, **kwargs):
        logger.info("SMLA CPU ordinary third-party reference")
        return mixed_reference(q, **kwargs)


class AclnnTorchReference:
    def __call__(self, q, **kwargs):
        values = aclnn_names(kwargs)
        return TorchReference()(q, **values)


def pre_compare(attention, lse, golden_attention, golden_lse, *, check_empty_lse=True):
    for index, (actual, expected) in enumerate(
        ((attention, golden_attention), (lse, golden_lse))
    ):
        if index == 1 and (
            expected is None or (check_empty_lse and expected.shape == (0,))
        ):
            if check_empty_lse and (
                actual is None
                or actual.shape != (0,)
                or cpu_tensor(actual).dtype != torch.float32
                or (
                    expected is not None and cpu_tensor(expected).dtype != torch.float32
                )
            ):
                raise ValueError("disabled LSE output must be an empty FP32 vector")
            continue
        if actual is None or expected is None or actual.shape != expected.shape:
            raise ValueError(
                f"MLA output {index} has a missing slot or mismatched shape"
            )
        for value in (actual, expected):
            if torch.isnan(cpu_tensor(value)).any():
                raise ValueError(f"MLA output {index} contains NaN")
    return attention, lse, golden_attention, golden_lse
