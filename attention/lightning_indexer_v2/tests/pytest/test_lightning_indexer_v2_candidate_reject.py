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

"""LightningIndexerV2 candidate host 侧校验拒绝测试（仅 arch22 / A2）。

逐条验证 candidate 属性开启时 host tiling 的拒绝行为（surface 为 torch RuntimeError）：

  C3' candidate_topk_blocks 开启时 ∈ (0, 2048] 且 64 的倍数（-1 为 off）
  C4' candidate_block_size ∈ [2, 64] 且 2 的幂（恒校验：off 时同样拒绝带病配置）
  V1  candidate 开启 + topk > 2048
  V2  return_value=2（值域越界，∈{0,1}）：candidate 模式已放开 rv（rv=1 合法），
      rv=2 由融合 torch 入口 lightning_indexer 的 candidate 路径 TORCH_CHECK 拒绝

Run:
    python3 -m pytest -rA -s test_lightning_indexer_v2_candidate_reject.py -v -m ci
"""

import pytest

# ---- guarded imports: keep the module collectable on hosts without NPU / built op package
_GOLDEN_IMPORT_ERROR = None
try:
    import torch
    import torch_npu  # noqa: F401
    import lightning_indexer_v2_golden as golden
except Exception as _exc:  # noqa: BLE001
    torch = None
    golden = None
    _GOLDEN_IMPORT_ERROR = _exc


def _device_arch():
    """arch22: Ascend910B / Ascend910_93; arch35: Ascend950; unknown: dev host."""
    if torch is None:
        return "unknown"
    try:
        props = torch.npu.get_device_properties()
        name = (getattr(props, "name", "") or "").lower()
    except Exception:  # noqa: BLE001
        return "unknown"
    if "ascend950" in name or "3510" in name:
        return "arch35"
    if "910b" in name or "910_93" in name or "2201" in name:
        return "arch22"
    return "unknown"


def _skip_if_unavailable():
    if golden is None:
        pytest.skip(
            f"lightning_indexer_v2_golden import failed: {_GOLDEN_IMPORT_ERROR!r}"
        )
    if _device_arch() == "arch35":
        pytest.skip(
            "candidate source is arch22-only (Ascend910B / Ascend910_93); "
            "candidate-on is host-rejected on arch35 (Ascend950)"
        )
    if golden.resolve_candidate_op()[0] is None:
        pytest.skip("candidate torch op not registered yet")


def _reject_case(params, candidate_topk_blocks, candidate_block_size, return_value=0):
    """Wrap golden.liv2_candidate_host_reject with the common skip + assert plumbing."""
    _skip_if_unavailable()
    rejected, message = golden.liv2_candidate_host_reject(
        params, candidate_topk_blocks, candidate_block_size, return_value=return_value
    )
    print(message)
    assert rejected, message
    return message


# BSND 基座参数（27 字段）：B=1 S1=8 S2=512 N1=4 N2=1 D=128 topk=64
def _base_params(topk=64, return_value=0):
    return (
        1,
        8,
        512,
        0,
        0,
        4,
        1,
        128,
        None,
        None,
        torch.float16,
        None,
        None,
        None,
        None,
        None,
        None,
        "BSND",
        "BSND",
        topk,
        0,
        [-2, 2],
        [-2, 2],
        [-2, 2],
        1,
        return_value,
        -1,
    )


@pytest.mark.ci
def test_reject_cand_topk_blocks_not_multiple_of_64():
    """C3'：candidate_topk_blocks=100（非 64 的倍数）必须被拒绝。"""
    _reject_case(_base_params(), 100, 8)


@pytest.mark.ci
def test_reject_cand_topk_blocks_over_2048():
    """C3'：candidate_topk_blocks=2112（> 2048）必须被拒绝。"""
    _reject_case(_base_params(), 2112, 8)


@pytest.mark.ci
def test_reject_cand_topk_blocks_negative_not_off():
    """C3'：candidate_topk_blocks=-2（非 -1 的负值）必须被拒绝。"""
    _reject_case(_base_params(), -2, 8)


@pytest.mark.ci
def test_reject_cand_block_size_not_power_of_two():
    """C4'：candidate_block_size=12（非 2 的幂）必须被拒绝。"""
    _reject_case(_base_params(), 64, 12)


@pytest.mark.ci
def test_reject_cand_block_size_below_min():
    """C4'：candidate_block_size=1（< 2）必须被拒绝。"""
    _reject_case(_base_params(), 64, 1)


@pytest.mark.ci
def test_reject_cand_block_size_above_max():
    """C4'：candidate_block_size=128（> 64）必须被拒绝。"""
    _reject_case(_base_params(), 64, 128)


@pytest.mark.ci
def test_reject_cand_block_size_checked_even_when_off():
    """C4' 恒校验：candidate off（-1）时 candidate_block_size=12 同样必须被拒绝。"""
    _reject_case(_base_params(), -1, 12)


@pytest.mark.ci
def test_reject_v1_topk_over_2048_with_candidate_on():
    """V1：candidate 开启（64）+ topk=3072（> 2048）必须被拒绝。"""
    _reject_case(_base_params(topk=3072), 64, 8)


@pytest.mark.ci
def test_reject_v2_rv2_with_candidate_on():
    """V2：candidate 开启（64）+ return_value=2（值域越界，∈{0,1}）必须被拒绝。

    candidate 模式已放开 return_value（rv=1 合法），但值域越界仍拒绝：报错来自
    融合 torch 入口 lightning_indexer 的 candidate 路径 TORCH_CHECK（"Return_value
    only supports 0 or 1"），经 candidate 参数传入即触发。
    """
    message = _reject_case(_base_params(), 64, 8, return_value=2)
    assert "Return_value only supports 0 or 1" in message, (
        f"rv=2 rejection must surface the TORCH_CHECK message, got: {message}"
    )


@pytest.mark.ci
def test_accept_candidate_rv1_on():
    """candidate 开启 + return_value=1（已放开的合法组合）必须被接受。"""
    _skip_if_unavailable()
    rejected, message = golden.liv2_candidate_host_reject(
        _base_params(return_value=1), 64, 8, return_value=1
    )
    print(message)
    assert not rejected, (
        f"candidate on + return_value=1 is a legal combination, but got: {message}"
    )


@pytest.mark.ci
def test_accept_candidate_min_valid_config():
    """合法下界配置必须被接受：candBlocks=64 + bs=2 + topk=64 + return_value=0。"""
    _skip_if_unavailable()
    rejected, message = golden.liv2_candidate_host_reject(_base_params(), 64, 2)
    assert not rejected, (
        f"legal candidate config (candBlocks=64, bs=2) must be accepted, but got: {message}"
    )
