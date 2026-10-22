# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import pytest
import torch
import torch_npu
from cann_ops_transformer.ops import chunk_gated_delta_rule_compute_wy

from test_chunk_gated_delta_rule_compute_wy import golden_compute_wy

CHUNK_SIZE = 64
COSINE_THRESHOLD = 0.999999
REPEAT_COUNT = 5


def minimum_chunk_cosine(actual, expected):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    rows = actual.double().reshape(-1, CHUNK_SIZE * actual.shape[-1])
    reference = expected.double().reshape_as(rows)
    denominator = rows.norm(dim=-1) * reference.norm(dim=-1)
    similarity = (rows * reference).sum(dim=-1) / denominator.clamp_min(1e-30)
    both_zero = (rows == 0).all(dim=-1) & (reference == 0).all(dim=-1)
    return torch.where(both_zero, torch.ones_like(similarity), similarity).min().item()


def check_outputs(actual, expected):
    assert len(actual) == len(expected) == 5
    for index, (result, reference) in enumerate(zip(actual, expected)):
        assert result.shape == reference.shape, index
        assert result.dtype == reference.dtype, index
        assert torch.isfinite(result).all(), index
    for index in (0, 1):
        torch.testing.assert_close(actual[index], expected[index], rtol=0, atol=0)
    torch.testing.assert_close(actual[4], expected[4], rtol=1e-5, atol=1e-6)
    scores = tuple(minimum_chunk_cosine(actual[i], expected[i]) for i in (2, 3))
    assert min(scores) >= COSINE_THRESHOLD, scores
    return scores


def make_inputs(shape, decay=0.1, correlated=False, zero=None, range_guard=False):
    batch, tokens, key_heads, value_heads, key_dim, value_dim = shape
    generator = torch.Generator().manual_seed(42)
    q = (
        torch.randn(batch, tokens, key_heads, key_dim, generator=generator) * 0.05
    ).half()
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, key_heads, key_dim, generator=generator), dim=-1
    ).half()
    if correlated:
        direction = torch.nn.functional.normalize(
            torch.randn(batch, 1, key_heads, key_dim, generator=generator), dim=-1
        )
        k = direction.expand(batch, tokens, key_heads, key_dim).contiguous().half()
    v = (
        torch.randn(batch, tokens, value_heads, value_dim, generator=generator) * 0.1
    ).half()
    g = torch.full((batch, tokens, value_heads), -decay, dtype=torch.float32)
    beta = torch.full((batch, tokens, value_heads), 0.8, dtype=torch.float16)
    if zero == "key":
        k.zero_()
    elif zero == "beta":
        beta.zero_()
    if range_guard:
        # The first U row exceeds the candidate's 60000 guard but fits FP16.
        # U rejection occurs after W was stored, requiring a whole-task retry.
        v.fill_(62000)
        beta.fill_(1)
    return q, k, v, g, beta


def assert_large_row_sum(inputs):
    _, k, _, g, beta = inputs
    batch, tokens, key_heads, dim = k.shape
    value_heads = g.shape[-1]
    key = k.double().transpose(1, 2).repeat_interleave(value_heads // key_heads, dim=1)
    key = key.reshape(batch, value_heads, -1, CHUNK_SIZE, dim)
    prefix = (
        g.double()
        .transpose(1, 2)
        .reshape(batch, value_heads, -1, CHUNK_SIZE)
        .cumsum(-1)
    )
    coefficient = beta.double().transpose(1, 2).reshape_as(prefix).unsqueeze(-1)
    delta = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
    # Mask before exp so unused upper-triangle lanes cannot overflow.
    decay = delta.tril(-1).exp().tril(-1)
    lower = (coefficient * key) @ key.transpose(-1, -2) * decay
    assert (lower.abs().sum(-1).amax(-1) >= 0.75).all()


def run_repeated(inputs, record_property):
    if "310p" not in torch_npu.npu.get_device_name(0).lower():
        pytest.skip("This operator supports Ascend 310P")
    torch_npu.npu.set_device(0)
    expected = golden_compute_wy(*inputs)
    device_inputs = tuple(tensor.npu() for tensor in inputs)
    first = None
    minimum = [1.0, 1.0]
    for _ in range(REPEAT_COUNT + 1):
        output = chunk_gated_delta_rule_compute_wy(
            *device_inputs, chunk_size=CHUNK_SIZE
        )
        torch_npu.npu.synchronize()
        current = tuple(tensor.cpu() for tensor in output)
        scores = check_outputs(current, expected)
        minimum = [min(old, new) for old, new in zip(minimum, scores)]
        if first is None:
            first = current
        else:
            assert all(torch.equal(a, b) for a, b in zip(first, current)), (
                "Nondeterministic output"
            )
    record_property("minimum_w_chunk_cosine", minimum[0])
    record_property("minimum_u_chunk_cosine", minimum[1])


@pytest.mark.ci
@pytest.mark.parametrize(
    "batch,tokens", [(1, 64), (1, 832), (1, 1280), (4, 128), (4, 2048)]
)
@pytest.mark.parametrize("decay", [0.001, 0.1, 0.3])
def test_inverse_split_large_row_sum(batch, tokens, decay, record_property):
    inputs = make_inputs((batch, tokens, 2, 4, 128, 128), decay, correlated=True)
    assert_large_row_sum(inputs)
    run_repeated(inputs, record_property)


@pytest.mark.ci
@pytest.mark.parametrize(
    "key_dim,value_dim",
    [(d, d) for d in (16, 32, 48, 64, 80, 96, 112, 128)]
    + [(16, 128), (128, 16), (64, 128), (128, 64)],
)
def test_inverse_split_supported_dimensions(key_dim, value_dim, record_property):
    run_repeated(make_inputs((2, 128, 1, 2, key_dim, value_dim)), record_property)


@pytest.mark.ci
@pytest.mark.parametrize("zero", ["key", "beta"])
def test_inverse_split_zero_rhs(zero, record_property):
    run_repeated(make_inputs((1, 128, 1, 2, 128, 128), zero=zero), record_property)


@pytest.mark.ci
def test_inverse_split_zero_decay(record_property):
    inputs = make_inputs((1, 128, 1, 2, 128, 128), decay=0.0, correlated=True)
    assert_large_row_sum(inputs)
    run_repeated(inputs, record_property)


@pytest.mark.ci
def test_inverse_split_u_range_guard_retries_entire_task(record_property):
    inputs = make_inputs((1, 128, 1, 2, 128, 128), correlated=True, range_guard=True)
    assert_large_row_sum(inputs)
    assert golden_compute_wy(*inputs)[3].abs().max() > 60000
    run_repeated(inputs, record_property)
