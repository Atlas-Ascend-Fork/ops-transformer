# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""FlashKDA output contract checks, valid-region packing, and TTK pre_compare masking.

Numerical precision comparison is delegated to TTK's native ``CrossCheckComparison``
(remote ``xpu_server`` when configured, or local fallback via ``cross_check.py``).
This module contains no custom numerical comparator; it handles only output
return-protocol validation, adapter valid-region packing, and out-of-contract
padding/slot zero-masking for TTK ``TestSpec.pre_compare``.
"""

import torch

from . import inputs as inputs_mod
from .arch_config import architecture

SLOTS = range(3)
FORWARD_SLOTS = range(16)
# 公开三项输出的接口类型，空槽位也使用对应类型。
PUBLIC_OUTPUT_DTYPES = (torch.bfloat16, torch.float32, torch.bfloat16)


def numerical_slots(c, *, forward=False):
    """Inference compares only public results, regardless of the entry point."""
    return FORWARD_SLOTS if forward and not c["inference"] else SLOTS


def check_arity(outputs, *, forward=False):
    slots = FORWARD_SLOTS if forward else SLOTS
    if not isinstance(outputs, (tuple, list)) or len(outputs) != len(slots):
        entry = "flash_kda_fwd" if forward else "flash_kda"
        raise AssertionError(
            f"{entry}: expected fixed {len(slots)} return slots, got {type(outputs)}"
        )


def _empty(dtype=torch.float32):
    return torch.empty(0, dtype=dtype)


def chunk_base(start, b):
    return start // 64 + b


def _valid_chunk_slots(valid_lengths, *, packed, max_chunks=None):
    """Return valid chunk_states slot indices in packed or flattened-dense order."""
    valid_slots = set()
    start = 0
    for bi, n in enumerate(valid_lengths):
        base = chunk_base(start, bi) if packed else bi * max_chunks
        valid_slots.update(range(base, base + (n + 63) // 64))
        start += n
    return valid_slots


def check(outputs, inputs, c, *, meta=False, forward=False, arch=None):
    check_arity(outputs, forward=forward)
    f = inputs_mod.effective(c)
    train = not c["inference"]
    b, nq, nv, s = c["batch_size"], c["nheads_qk"], c["nheads_v"], c["seqlens"]
    packed = c["layout"] == "TND"
    j = (s + 63) // 64 + (b if packed else 0)
    chunk = (j, nv) if packed else (b, j, nv)
    qshape = tuple(inputs[0].shape)
    rshape = (nq, s) if packed else (b, nq, s)
    bshape = (nv, s) if packed else (b, nv, s)
    token_prefix = (nv, s) if packed else (b, nv, s)
    key_cache = token_prefix + (inputs[0].shape[-1],)
    value_cache = token_prefix + (inputs[2].shape[-1],)
    out_shape = (s, nv) if packed else (b, s, nv)
    out_shape += (inputs[2].shape[-1],)
    shape = [
        out_shape,
        (b, nv, 128, 128),
        chunk + (128, 128),
        token_prefix + (64,),
        token_prefix + (64,),
        key_cache,
        value_cache,
        key_cache,
        key_cache,
        value_cache,
        key_cache,
        qshape,
        rshape,
        qshape,
        rshape,
        bshape,
    ]
    enabled = (
        [
            True,
            f["output_final_state"],
            (not train and f["return_intermediate_states"])
            or (train and f["disable_recompute"]),
            train,
            train,
        ]
        + [train and f["disable_recompute"]] * 6
        + [train and f["use_qk_l2norm_in_kernel"]] * 4
        + [train and f["use_beta_sigmoid_in_kernel"]]
    )
    fp32 = {1, 10, 12, 14, 15}
    for i, (value, on) in enumerate(zip(outputs, enabled)):
        if not on:
            if value is not None:
                raise AssertionError(f"{c['id']} slot {i}: expected None")
        else:
            dtype = torch.float32 if i in fp32 else torch.bfloat16
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape[i]
                or value.dtype != dtype
            ):
                raise AssertionError(
                    f"{c['id']} slot {i}: expected Tensor {shape[i]} {dtype}, got {getattr(value, 'shape', None)} {getattr(value, 'dtype', None)}"
                )
    if meta:
        return
    selected_arch = arch or architecture(
        metadata=inputs[5] if len(inputs) > 5 else None
    )
    if selected_arch == "arch92" and outputs[2] is not None:
        states = outputs[2] if packed else outputs[2].flatten(0, 1)
        valid_slots = _valid_chunk_slots(
            c["valid_lengths"], packed=packed, max_chunks=j
        )
        for slot in set(range(states.shape[0])) - valid_slots:
            if not torch.equal(states[slot], torch.zeros_like(states[slot])):
                raise AssertionError(f"chunk_states unused slot {slot} is not zero")
    if not forward:
        return
    if train:
        start = 0
        for bi, n in enumerate(c["valid_lengths"]):
            aq_sequence = (
                outputs[3][:, start : start + n] if packed else outputs[3][bi, :, :n]
            )
            ak_sequence = (
                outputs[4][:, start : start + n] if packed else outputs[4][bi, :, :n]
            )
            for ci in range((n + 63) // 64):
                valid = min(64, n - ci * 64)
                rows = slice(ci * 64, ci * 64 + valid)
                aq = aq_sequence[:, rows, :valid]
                ak = ak_sequence[:, rows, :valid]
                if not torch.equal(aq.triu(1), torch.zeros_like(aq)):
                    raise AssertionError("Aqk valid upper triangle is not zero")
                if not torch.equal(
                    ak.diagonal(dim1=-2, dim2=-1),
                    torch.ones_like(ak.diagonal(dim1=-2, dim2=-1)),
                ):
                    raise AssertionError("Akk valid diagonal is not one")
            start += n


def pack(outputs, c, *, forward=False):
    """Extract valid regions into 1D tensors for adapter finiteness/recovery checks."""
    check_arity(outputs, forward=forward)
    result = []
    for slot in numerical_slots(c, forward=forward):
        value = outputs[slot]
        if value is None:
            dtype = (
                PUBLIC_OUTPUT_DTYPES[slot]
                if slot < len(PUBLIC_OUTPUT_DTYPES)
                else torch.float32
            )
            result.append(_empty(dtype=dtype).to(outputs[0].device))
            continue
        if slot == 1:
            parts = [value.flatten()]
        elif slot == 2:
            parts = []
            start = 0
            for b, n in enumerate(c["valid_lengths"]):
                count = (n + 63) // 64
                states = (
                    value[chunk_base(start, b) : chunk_base(start, b) + count]
                    if c["layout"] == "TND"
                    else value[b, :count]
                )
                parts.append(states.flatten())
                start += n
        elif slot in (3, 4):
            parts = []
            start = 0
            for b, n in enumerate(c["valid_lengths"]):
                sequence = (
                    value[:, start : start + n]
                    if c["layout"] == "TND"
                    else value[b, :, :n]
                )
                for row_start in range(0, n, 64):
                    valid = min(64, n - row_start)
                    parts.append(
                        sequence[:, row_start : row_start + valid, :valid].flatten()
                    )
                start += n
        else:
            # 将头维度在前的标量和缓存转换为切片辅助函数要求的 token 维度在前的布局。
            if slot in (5, 6, 7, 8, 9, 10, 12, 14, 15):
                if c["layout"] == "TND":
                    value = value.transpose(0, 1)
                elif c["layout"] == "BSND":
                    value = value.transpose(1, 2)
            # 定长 out 固定为 BSND，切片辅助函数按输入布局解释各轴。
            if slot == 0 and c["layout"] == "BNSD":
                value = value.transpose(1, 2)
            parts = [x.flatten() for x in inputs_mod.sequences(value, c)]
        result.append(torch.cat(parts))
    return result


def _clone_buffer(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return value.copy()


def _validate_public_output_shapes(out, chunk_states, c):
    """Guard against stale active-case context before zero-masking padding regions."""
    valid_lengths = c["valid_lengths"]
    if len(valid_lengths) != c["batch_size"]:
        raise ValueError(
            f"FlashKDA pre_compare valid_lengths mismatch for {c['id']}: "
            f"len={len(valid_lengths)} vs batch_size={c['batch_size']}"
        )
    packed = c["layout"] == "TND"
    if out is not None:
        expected_out = (
            (c["seqlens"], c["nheads_v"])
            if packed
            else (c["batch_size"], c["seqlens"], c["nheads_v"])
        )
        if (
            out.ndim < len(expected_out)
            or tuple(out.shape[: len(expected_out)]) != expected_out
        ):
            raise ValueError(
                f"FlashKDA pre_compare out shape mismatch for {c['id']}: "
                f"got {tuple(out.shape)}, expected prefix {expected_out} ({c['layout']})"
            )
    if chunk_states is not None:
        expected_chunks = (
            (c["seqlens"] + 63) // 64 + c["batch_size"]
            if packed
            else (c["seqlens"] + 63) // 64
        )
        expected_prefix = (
            (expected_chunks, c["nheads_v"])
            if packed
            else (c["batch_size"], expected_chunks, c["nheads_v"])
        )
        if (
            chunk_states.ndim < len(expected_prefix)
            or tuple(chunk_states.shape[: len(expected_prefix)]) != expected_prefix
        ):
            raise ValueError(
                f"FlashKDA pre_compare chunk_states shape mismatch for {c['id']}: "
                f"got {tuple(chunk_states.shape)}, expected prefix {expected_prefix} ({c['layout']})"
            )


def mask_public_outputs(outputs, c, *, arch=None):
    """Zero out unwritten padding tokens and unused arch35 chunk_states slots.

    Direct ``torch.ops.cann_ops_transformer.flash_kda`` returns full-shape
    tensors whose padding tokens (fixed-length ``valid_lengths[b] < S``) and
    arch35 unused ``chunk_states`` slots contain uninitialized HBM contents.
    Zeroing only those out-of-contract regions preserves tensor shapes so TTK
    native remote and local ``CrossCheckComparison`` stay shape-compatible with
    ``third_party`` while keeping valid regions bit-exact and leaving
    ``CrossCheckComparison`` large/small metrics unaffected.
    """
    if len(outputs) != 3:
        return tuple(outputs)
    out, final_state, chunk_states = outputs
    _validate_public_output_shapes(out, chunk_states, c)

    valid_lengths = c["valid_lengths"]
    packed = c["layout"] == "TND"
    selected_arch = arch or architecture()

    if out is not None and not packed and out.ndim >= 2:
        seq_cap = out.shape[1]
        if any(n < seq_cap for n in valid_lengths):
            out = _clone_buffer(out)
            for bi, n in enumerate(valid_lengths):
                if n < seq_cap:
                    out[bi, n:] = 0

    if chunk_states is not None and selected_arch != "arch92":
        if packed and chunk_states.ndim >= 1:
            valid_slots = _valid_chunk_slots(valid_lengths, packed=True)
            total_slots = chunk_states.shape[0]
            if len(valid_slots) < total_slots:
                chunk_states = _clone_buffer(chunk_states)
                for slot_idx in range(total_slots):
                    if slot_idx not in valid_slots:
                        chunk_states[slot_idx] = 0
        elif not packed and chunk_states.ndim >= 2:
            max_chunks = chunk_states.shape[1]
            if any((n + 63) // 64 < max_chunks for n in valid_lengths):
                chunk_states = _clone_buffer(chunk_states)
                for bi, n in enumerate(valid_lengths):
                    count = (n + 63) // 64
                    if count < max_chunks:
                        chunk_states[bi, count:] = 0

    return (out, final_state, chunk_states)


def pre_compare_public(*outputs):
    """TTK ``pre_compare`` hook: mask out-of-contract padding on public FlashKDA outputs."""
    from .npu_preprocess import get_active_case

    active = get_active_case()
    if active is None or len(outputs) != 6:
        return None
    arch, c = active
    masked_npu = mask_public_outputs(outputs[:3], c, arch=arch)
    masked_golden = mask_public_outputs(outputs[3:], c, arch=arch)
    return masked_npu + masked_golden
