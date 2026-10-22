# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Reference implementation for GroupedMatmulActivationWeightNz torch tests.

The caller supplies already dequantized logical MXFP4 values. Packing to FRACTAL_NZ
and E8M0/FP4 decoding are input-construction concerns; this reference deliberately
keeps the mathematical contract visible and checks the grouped GEMM and SwiGLU path.
"""

import math

import torch


def grouped_matmul_activation_weight_nz_golden(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_list: torch.Tensor,
    group_list_type: int = 0,
    activation_type: str = "swiglu",
    swiglu_mode: int = 2,
    clamp_limit: float = 7.0,
    glu_alpha: float = 1.702,
    glu_bias: float = 1.0,
) -> torch.Tensor:
    """Computes the BF16-output GMM+SwiGLU reference from logical FP32 tensors."""
    if x.ndim != 2 or weight.ndim != 3:
        raise ValueError("x must be [M, K] and weight must be [E, K, N].")
    if weight.shape[1] != x.shape[1] or weight.shape[2] % 2 != 0:
        raise ValueError("K must match and N must be even.")
    if group_list.numel() != weight.shape[0]:
        raise ValueError("group_list length must equal the expert count.")
    if activation_type != "swiglu":
        raise ValueError("only activation_type='swiglu' is supported.")
    if group_list_type not in (0, 1):
        raise ValueError("group_list_type must be 0 or 1.")
    if swiglu_mode != 2:
        raise ValueError("only swiglu_mode=2 is supported.")
    if (
        not all(math.isfinite(v) for v in (clamp_limit, glu_alpha, glu_bias))
        or clamp_limit <= 0
    ):
        raise ValueError(
            "SwiGLU parameters must be finite and clamp_limit must be positive."
        )

    counts = group_list.to(torch.int64)
    if group_list_type == 0:
        counts = torch.diff(torch.cat((torch.zeros(1, dtype=torch.int64), counts)))
    if torch.any(counts < 0) or int(counts.sum()) != x.shape[0]:
        raise ValueError("group_list does not describe a valid token partition.")

    output = torch.empty((x.shape[0], weight.shape[2] // 2), dtype=torch.float32)
    token_begin = 0
    for expert, count in enumerate(counts.tolist()):
        token_end = token_begin + count
        gmm = x[token_begin:token_end].float().matmul(weight[expert].float())
        act, gate = torch.chunk(gmm, 2, dim=-1)
        act = torch.minimum(act, torch.tensor(clamp_limit, dtype=act.dtype))
        gate = torch.clamp(gate, -clamp_limit, clamp_limit) + glu_bias
        output[token_begin:token_end] = (act / (1 + torch.exp(-glu_alpha * act))) * gate
        token_begin = token_end
    return output.to(torch.bfloat16)
