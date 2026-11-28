# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.api_execute.base_api import BaseApi


def sequence_lengths(boundaries, length_type):
    if length_type == 0:
        return [end - start for start, end in zip([0, *boundaries], boundaries)]
    if length_type == 1:
        return list(boundaries)
    raise ValueError(f"Unsupported actSeqLenType: {length_type}")


def compress_grad(output_grad, input_, weight, lengths, stride):
    """Accumulate both gradients over complete windows within each sequence."""
    block_size = weight.shape[0]
    windows = [max(0, (length - block_size) // stride + 1) for length in lengths]
    if any(length < 0 for length in lengths) or sum(lengths) != input_.shape[0]:
        raise ValueError("Sequence lengths must cover input exactly")
    if output_grad.shape != (sum(windows), *input_.shape[1:]):
        raise ValueError("outputGrad shape must match the number of complete windows")
    if input_.shape[1] != weight.shape[1]:
        raise ValueError("Weight and input head counts must match")

    input_grad = torch.zeros_like(input_, dtype=torch.float32)
    weight_grad = torch.zeros_like(weight, dtype=torch.float32)
    x, dy, w = input_.float(), output_grad.float(), weight.float()
    input_start = output_start = 0
    for length, count in zip(lengths, windows):
        for index in range(count):
            start = input_start + index * stride
            window = slice(start, start + block_size)
            grad = dy[output_start + index]
            input_grad[window] += w.unsqueeze(-1) * grad
            weight_grad += (x[window] * grad).sum(dim=-1)
        input_start += length
        output_start += count
    return input_grad.to(input_.dtype), weight_grad.to(weight.dtype)


@register("nsa_compress_grad_reference")
class NsaCompressGradReference(BaseApi):
    def __call__(self, input_data, with_output=False):
        kw = input_data.kwargs
        return compress_grad(
            kw["outputGrad"].cpu(),
            kw["input"].cpu(),
            kw["weight"].cpu(),
            sequence_lengths(kw["actSeqLenOptional"], kw["actSeqLenType"]),
            kw["compressStride"],
        )


@register("nsa_compress_grad_aclnn")
class NsaCompressGradAclnn(AclnnBaseApi):
    pass
