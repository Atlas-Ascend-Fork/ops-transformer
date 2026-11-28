#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""ATK case generator, CPU benchmark, and A2/A3 aclnn adapter for SFMLA L1Norm.

All case tensors are generated in memory from the scalar parameters in
``atk_aclnnSparseFlashMlaSoftmaxL1Norm.json``. No input, output, or golden tensor is
stored in the repository.
"""

from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass
from typing import Any

import torch

try:
    from atk.tasks.api_execute import register
    from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
    from atk.tasks.api_execute.base_api import BaseApi
    from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr
except ImportError:  # Allow local generator/reference checks without ATK.

    def register(_name):
        return lambda cls: cls

    class BaseApi:  # type: ignore[no-redef]
        pass

    class AclnnBaseApi:  # type: ignore[no-redef]
        pass

    class TensorPtr:  # type: ignore[no-redef]
        pass


FP16 = 0
BF16 = 1
BSND = 0
TND = 1
BASE_SEED = 2026111200


@dataclass(frozen=True)
class CaseSpec:
    case_index: int
    dtype_code: int
    layout_code: int
    batch: int
    sq_max: int
    sk_max: int
    n1: int
    n2: int
    head_dim: int
    kout: int
    mask_mode: int
    cmp_ratio: int
    sparse: bool
    seqused: bool

    @property
    def dtype(self):
        return torch.float16 if self.dtype_code == FP16 else torch.bfloat16

    @property
    def layout(self):
        return "BSND" if self.layout_code == BSND else "TND"


def _as_int(kwargs: dict[str, Any], name: str) -> int:
    value = kwargs[name]
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def spec_from_kwargs(kwargs: dict[str, Any]) -> CaseSpec:
    spec = CaseSpec(
        case_index=_as_int(kwargs, "case_index"),
        dtype_code=_as_int(kwargs, "dtype_code"),
        layout_code=_as_int(kwargs, "layout_code"),
        batch=_as_int(kwargs, "batch"),
        sq_max=_as_int(kwargs, "sq_max"),
        sk_max=_as_int(kwargs, "sk_max"),
        n1=_as_int(kwargs, "n1"),
        n2=_as_int(kwargs, "n2"),
        head_dim=_as_int(kwargs, "head_dim"),
        kout=_as_int(kwargs, "kout"),
        mask_mode=_as_int(kwargs, "mask_mode"),
        cmp_ratio=_as_int(kwargs, "cmp_ratio"),
        sparse=bool(_as_int(kwargs, "sparse")),
        seqused=bool(_as_int(kwargs, "seqused")),
    )
    if spec.dtype_code not in (FP16, BF16):
        raise ValueError(f"dtype_code must be 0 or 1, got {spec.dtype_code}")
    if spec.layout_code not in (BSND, TND):
        raise ValueError(f"layout_code must be 0 or 1, got {spec.layout_code}")
    if spec.n2 != 1 or not 1 <= spec.n1 <= 128:
        raise ValueError(f"unsupported heads: n1={spec.n1}, n2={spec.n2}")
    if spec.head_dim not in (256, 512):
        raise ValueError(f"head_dim must be 256 or 512, got {spec.head_dim}")
    if spec.mask_mode not in (0, 3) or not 1 <= spec.cmp_ratio <= 128:
        raise ValueError("mask_mode/cmp_ratio is outside the supported range")
    return spec


def _causal_length(
    q_index: int, q_length: int, k_length: int, cmp_ratio: int, residual: int
) -> int:
    original_k = k_length * cmp_ratio + (residual if cmp_ratio > 1 else 0)
    return min(k_length, max((original_k - q_length + q_index + 1) // cmp_ratio, 0))


def _case_capacities(spec: CaseSpec):
    if spec.layout_code == BSND:
        q_caps = torch.full((spec.batch,), spec.sq_max, dtype=torch.int32)
        k_caps = torch.full((spec.batch,), spec.sk_max, dtype=torch.int32)
    else:
        q_mod = min(spec.sq_max, 3)
        k_mod = min(spec.sk_max, 4)
        q_caps = torch.tensor(
            [max(1, spec.sq_max - (b % q_mod)) for b in range(spec.batch)],
            dtype=torch.int32,
        )
        k_caps = torch.tensor(
            [max(1, spec.sk_max - ((b * 2) % k_mod)) for b in range(spec.batch)],
            dtype=torch.int32,
        )
    if spec.seqused:
        q_lengths = torch.tensor(
            [
                max(1, int(cap) - ((spec.case_index + b * 3) % min(int(cap), 4)))
                for b, cap in enumerate(q_caps)
            ],
            dtype=torch.int32,
        )
        k_lengths = torch.tensor(
            [
                max(1, int(cap) - ((spec.case_index * 3 + b * 5) % min(int(cap), 5)))
                for b, cap in enumerate(k_caps)
            ],
            dtype=torch.int32,
        )
    else:
        q_lengths = q_caps.clone()
        k_lengths = k_caps.clone()
    cuq = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(q_caps, dim=0, dtype=torch.int32),
        )
    )
    cuk = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(k_caps, dim=0, dtype=torch.int32),
        )
    )
    return q_caps, k_caps, q_lengths, k_lengths, cuq, cuk


def build_case(spec: CaseSpec, benchmark_mode: str | None = None) -> dict[str, Any]:
    """Generate one deterministic case and optionally its high/low CPU output."""
    if benchmark_mode not in (None, "high", "low"):
        raise ValueError("benchmark_mode must be high, low, or None")

    q_caps, k_caps, q_lengths, k_lengths, cuq, cuk = _case_capacities(spec)
    total_q = int(q_caps.sum())
    total_k = int(k_caps.sum())
    q_shape = (
        (spec.batch, spec.sq_max, spec.n1, spec.head_dim)
        if spec.layout_code == BSND
        else (total_q, spec.n1, spec.head_dim)
    )
    k_shape = (
        (spec.batch, spec.sk_max, spec.n2, spec.head_dim)
        if spec.layout_code == BSND
        else (total_k, spec.n2, spec.head_dim)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(BASE_SEED + spec.case_index)
    input_std = (0.05, 0.12, 0.5, 1.0)[spec.case_index % 4]
    q = (torch.randn(q_shape, generator=generator, dtype=torch.float32) * input_std).to(
        spec.dtype
    )
    k = (torch.randn(k_shape, generator=generator, dtype=torch.float32) * input_std).to(
        spec.dtype
    )
    scale = 1.0 / math.sqrt(spec.head_dim)

    if spec.layout_code == BSND:
        lse = torch.full(
            (spec.batch, spec.n2, spec.sq_max, spec.n1), -torch.inf, dtype=torch.float32
        )
        out_shape = (spec.batch, spec.sq_max, spec.n2, spec.kout)
        sparse_indices = (
            torch.zeros(out_shape, dtype=torch.int32) if spec.sparse else None
        )
        topk = (
            torch.zeros((spec.batch, spec.sq_max, spec.n2), dtype=torch.int32)
            if spec.sparse and spec.mask_mode == 0
            else None
        )
    else:
        lse = torch.full((spec.n2, total_q, spec.n1), -torch.inf, dtype=torch.float32)
        out_shape = (total_q, spec.n2, spec.kout)
        sparse_indices = (
            torch.zeros(out_shape, dtype=torch.int32) if spec.sparse else None
        )
        topk = (
            torch.zeros((total_q, spec.n2), dtype=torch.int32)
            if spec.sparse and spec.mask_mode == 0
            else None
        )
    golden = torch.zeros(out_shape, dtype=torch.float32) if benchmark_mode else None
    residual = torch.tensor(
        [
            (spec.case_index + b * 11) % spec.cmp_ratio if spec.cmp_ratio > 1 else 0
            for b in range(spec.batch)
        ],
        dtype=torch.int32,
    )

    for batch_index in range(spec.batch):
        q_valid = int(q_lengths[batch_index])
        k_valid = int(k_lengths[batch_index])
        q_cap = int(q_caps[batch_index])
        q_base = int(cuq[batch_index])
        k_base = int(cuk[batch_index])
        for q_relative in range(q_cap):
            if q_relative >= q_valid:
                continue
            q_index = q_relative if spec.layout_code == BSND else q_base + q_relative
            q_row = (
                q[batch_index, q_relative] if spec.layout_code == BSND else q[q_index]
            )
            valid = k_valid
            if spec.mask_mode == 3:
                q_reference = q_valid if spec.layout_code == BSND else q_cap
                valid = _causal_length(
                    q_relative,
                    q_reference,
                    k_valid,
                    spec.cmp_ratio,
                    int(residual[batch_index]),
                )
            if spec.sparse:
                if spec.mask_mode == 0:
                    valid = min(
                        valid,
                        (spec.case_index + batch_index * 3 + q_relative * 5)
                        % (spec.kout + 1),
                    )
                    topk_slot = (
                        (batch_index, q_relative, 0)
                        if spec.layout_code == BSND
                        else (q_index, 0)
                    )
                    topk[topk_slot] = valid
                else:
                    valid = min(valid, spec.kout)
                row_indices = (
                    torch.arange(spec.kout, dtype=torch.int32) * 7
                    + spec.case_index
                    + q_relative * 3
                ) % k_valid
                index_slot = (
                    (batch_index, q_relative, 0)
                    if spec.layout_code == BSND
                    else (q_index, 0)
                )
                sparse_indices[index_slot] = row_indices
                selected = row_indices[:valid].to(torch.int64)
            else:
                valid = min(valid, spec.kout)
                selected = torch.arange(valid, dtype=torch.int64)
            if valid == 0:
                continue
            key_rows = (
                k[batch_index, selected, 0]
                if spec.layout_code == BSND
                else k[k_base + selected, 0]
            )
            scores = (
                torch.matmul(q_row.float(), key_rows.float().transpose(0, 1)) * scale
            )
            row_lse = torch.logsumexp(scores, dim=-1)
            if spec.layout_code == BSND:
                lse[batch_index, 0, q_relative] = row_lse
                out_slot = (batch_index, q_relative, 0, slice(0, valid))
            else:
                lse[0, q_index] = row_lse
                out_slot = (q_index, 0, slice(0, valid))
            if benchmark_mode == "high":
                reference_scores = (
                    torch.matmul(q_row.double(), key_rows.double().transpose(0, 1))
                    * scale
                )
                probs = torch.exp(reference_scores - row_lse.double()[:, None]).mean(
                    dim=0
                )
                golden[out_slot] = probs.float()
            elif benchmark_mode == "low":
                probs = torch.exp(scores - row_lse[:, None]).mean(dim=0)
                golden[out_slot] = probs.float()

    return {
        "q": q,
        "k": k,
        "softmax_lse": lse,
        "sparse_indices": sparse_indices,
        "cu_seqlens_q": cuq if spec.layout_code == TND else None,
        "cu_seqlens_k": cuk if spec.layout_code == TND else None,
        "seqused_q": q_lengths if spec.seqused else None,
        "seqused_k": k_lengths if spec.seqused else None,
        "cmp_residual_k": residual
        if spec.mask_mode == 3 and spec.cmp_ratio > 1
        else None,
        "topk_length": topk,
        # Arch22 requires a 1-D INT32 metadata tensor but does not consume its fields.
        "metadata": torch.zeros(64, dtype=torch.int32),
        "softmax_scale": scale,
        "max_seqlen_k": spec.kout if spec.layout_code == TND and not spec.sparse else 0,
        "cmp_ratio": spec.cmp_ratio,
        "mask_mode": spec.mask_mode,
        "layout": spec.layout,
        "out_shape": out_shape,
        "golden": golden,
    }


@register("function_sfmla_softmax_l1_norm_atk")
class SfmlaSoftmaxL1NormGolden(BaseApi):
    """CPU high/low benchmark selected by SFMLA_BENCHMARK_MODE."""

    def __call__(self, input_data, with_output: bool = False):
        mode = os.environ.get("SFMLA_BENCHMARK_MODE", "high").lower()
        if mode not in ("high", "low"):
            raise ValueError("SFMLA_BENCHMARK_MODE must be high or low")
        case = build_case(spec_from_kwargs(input_data.kwargs), mode)
        return (case["golden"],)


@register("pyaclnn_aclnn_sfmla_softmax_l1_norm_atk")
class PyAclnnSfmlaSoftmaxL1NormATK(AclnnBaseApi):
    """A2/A3 aclnn adapter for the main SparseFlashMlaSoftmaxL1Norm operator."""

    _TENSOR_INPUTS = (
        "q",
        "k",
        "softmax_lse",
        "sparse_indices",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "seqused_q",
        "seqused_k",
        "cmp_residual_k",
        "topk_length",
        "metadata",
    )

    def get_cpp_func_signature_type(self):
        return (
            "aclnnStatus aclnnSparseFlashMlaSoftmaxL1NormGetWorkspaceSize("
            "const aclTensor *q, const aclTensor *k, const aclTensor *softmaxLse, "
            "const aclTensor *sparseIndicesOptional, const aclTensor *cuSeqlensQOptional, "
            "const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional, "
            "const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional, "
            "const aclTensor *topkLengthOptional, const aclTensor *metadataOptional, "
            "double softmaxScale, int64_t maxSeqlenK, int64_t cmpRatio, int64_t maskMode, "
            "char *layoutQOptional, char *layoutKOptional, const aclTensor *softmaxL1Norm, "
            "uint64_t *workspaceSize, aclOpExecutor **executor)"
        )

    def init_by_input_data(self, input_data):
        case = build_case(spec_from_kwargs(input_data.kwargs))
        args = []
        for name in self._TENSOR_INPUTS:
            tensor = case[name]
            if tensor is None:
                args.append(TensorPtr())
            else:
                args.extend(self.backend.convert_input_data(tensor, name=name))
        args.extend(
            [
                ctypes.c_double(case["softmax_scale"]),
                ctypes.c_int64(case["max_seqlen_k"]),
                ctypes.c_int64(case["cmp_ratio"]),
                ctypes.c_int64(case["mask_mode"]),
                ctypes.c_char_p(case["layout"].encode("ascii")),
                ctypes.c_char_p(case["layout"].encode("ascii")),
            ]
        )
        output = torch.empty(case["out_shape"], dtype=torch.float32)
        output_package = self.torch_tensor_to_acl(output)
        args.append(output_package)
        return args, [output_package]
