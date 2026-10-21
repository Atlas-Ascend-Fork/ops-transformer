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

"""SparseLightningIndexer host 侧校验拒绝测试（C2-C8 / R18，仅 arch22 / A2）。

逐条验证 aclnnSparseLightningIndexer host tiling 对非法配置的拒绝行为
（surface 为 torch RuntimeError / ValueError）：

  C2  candidate_block_indices rank/dim 校验（BSND rank4 [B,S1,N2,K] / TND rank3 [T,N2,K]）
  C3  candBlocks ∈ (0, 2048] 且 64 的倍数
  C4  candidate_block_size ∈ [2, 64] 且 2 的幂
  C5  topk ∈ (0, 2048]（恒校验，无 over-2K 回退）
  C6  return_value ∈ {0, 1}：rv=2（其它值）拒绝 / rv=1 接受（本次开放，正反双面）
  C7  candidate_block_length 仅接受 None / numel==0
  C8  cmpRatio ∈ (0, 128] 且 2 的幂
  R18 arch22 不消费入参硬拒绝：seqused_q / output_idx_offset / metadata 非空、
      max_seqlen_q != -1

Run:
    python3 -m pytest -rA -s test_sparse_lightning_indexer_reject.py -v -m ci
"""

import pytest

# ---- guarded imports: keep the module collectable on hosts without NPU / built op package
_GOLDEN_IMPORT_ERROR = None
try:
    import torch
    import torch_npu  # noqa: F401
    import sparse_lightning_indexer_golden as sli_golden
except Exception as _exc:  # noqa: BLE001
    torch = None
    sli_golden = None
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
    if sli_golden is None:
        pytest.skip(
            f"sparse_lightning_indexer_golden import failed: {_GOLDEN_IMPORT_ERROR!r}"
        )
    if _device_arch() == "arch35":
        pytest.skip(
            "sparse_lightning_indexer is arch22-only (Ascend910B / Ascend910_93)"
        )
    if sli_golden.resolve_sparse_op()[0] is None:
        pytest.skip("sparse torch op not registered yet")


# BSND 基座参数（27 字段）：B=1 S1=8 S2=256 N1=4 N2=1 D=128 topk=64
BSND_PARAMS = (
    1,
    8,
    256,
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
    64,
    0,
    [-2, 2],
    [-2, 2],
    [-2, 2],
    1,
    0,
    -1,
)

# TND 基座参数：B=2 T=6 S2=192（cu_seqlens_k=[0,64,192]）
TND_PARAMS = (
    2,
    4,
    128,
    6,
    192,
    4,
    1,
    128,
    None,
    None,
    torch.float16,
    [0, 2, 6],
    [0, 64, 192],
    None,
    None,
    None,
    None,
    "TND",
    "TND",
    64,
    0,
    [-2, 2],
    [-2, 2],
    [-2, 2],
    1,
    0,
    -1,
)


def _reject_case(params, **kwargs):
    """Wrap sli_golden.sparse_host_reject with the common skip + assert-rejected plumbing."""
    _skip_if_unavailable()
    rejected, message = sli_golden.sparse_host_reject(params, **kwargs)
    print(message)
    assert rejected, message


@pytest.mark.ci
def test_reject_c3_candblocks_not_multiple_of_64():
    """C3：candBlocks=100（非 64 的倍数）必须被拒绝。"""
    _reject_case(BSND_PARAMS, cand_blocks_shape=100)


@pytest.mark.ci
def test_reject_c3_candblocks_over_2048():
    """C3：candBlocks=2112（> 2048）必须被拒绝。"""
    _reject_case(BSND_PARAMS, cand_blocks_shape=2112)


@pytest.mark.ci
def test_reject_c3_candblocks_tuple_over_2048():
    """C3：显式 shape 末维 4096（> 2048）必须被拒绝。"""
    _reject_case(BSND_PARAMS, cand_blocks_shape=(1, 8, 1, 4096))


@pytest.mark.ci
def test_reject_c2_bsnd_rank3_candidate():
    """C2：BSND q 但 candidate rank 3（期望 rank 4 [B,S1,N2,K]）必须被拒绝。"""
    _reject_case(BSND_PARAMS, cand_blocks_shape=64, cand_shape_override=(8, 1, 64))


@pytest.mark.ci
def test_reject_c2_tnd_rank4_candidate():
    """C2：TND q 但 candidate rank 4（期望 rank 3 [T,N2,K]）必须被拒绝。"""
    _reject_case(
        TND_PARAMS,
        cand_blocks_shape=64,
        cand_shape_override=(6, 4, 1, 64),
    )


@pytest.mark.ci
def test_reject_c2_bsnd_wrong_batch_dim():
    """C2：BSND candidate 的 B 维（2）与 q 的 B（1）不一致必须被拒绝。"""
    _reject_case(BSND_PARAMS, cand_blocks_shape=64, cand_shape_override=(2, 8, 1, 64))


@pytest.mark.ci
def test_reject_c2_bsnd_wrong_s1_dim():
    """C2：BSND candidate 的 S1 维（9）与 q 的 S1（8）不一致必须被拒绝。"""
    _reject_case(BSND_PARAMS, cand_blocks_shape=64, cand_shape_override=(1, 9, 1, 64))


@pytest.mark.ci
def test_reject_c4_blocksize_not_power_of_two():
    """C4：candidate_block_size=12（非 2 的幂）必须被拒绝。"""
    _reject_case(BSND_PARAMS, cand_blocks_shape=64, block_size=12)


@pytest.mark.ci
def test_reject_c4_blocksize_below_min():
    """C4：candidate_block_size=1（< 2）必须被拒绝。"""
    _reject_case(BSND_PARAMS, cand_blocks_shape=64, block_size=1)


@pytest.mark.ci
def test_reject_c4_blocksize_above_max():
    """C4：candidate_block_size=128（> 64）必须被拒绝。"""
    _reject_case(BSND_PARAMS, cand_blocks_shape=64, block_size=128)


@pytest.mark.ci
def test_reject_c7_block_length_nonempty():
    """C7：candidate_block_length 非空（numel=3）必须被拒绝。"""
    _reject_case(
        BSND_PARAMS,
        cand_blocks_shape=64,
        block_length=torch.zeros(3, dtype=torch.int32),
    )


@pytest.mark.ci
def test_reject_c5_topk_over_2048():
    """C5：topk=2049（> 2048，本算子无 over-2K 回退）必须被拒绝。"""
    _reject_case(BSND_PARAMS, cand_blocks_shape=64, topk_override=2049)


@pytest.mark.ci
def test_reject_c8_cmp_ratio_not_power_of_two():
    """C8：cmp_ratio=3（非 2 的幂）必须被拒绝。"""
    _reject_case(
        BSND_PARAMS,
        cand_blocks_shape=64,
        kwargs_override={"cmp_ratio": 3},
    )


@pytest.mark.ci
def test_reject_r18_max_seqlen_q_not_minus_one():
    """R18：arch22 不消费 max_seqlen_q，非 -1（=4）必须被拒绝。"""
    _reject_case(
        BSND_PARAMS,
        cand_blocks_shape=64,
        kwargs_override={"max_seqlen_q": 4},
    )


@pytest.mark.ci
def test_reject_r18_seqused_q_present():
    """R18：arch22 不消费 seqused_q，非空必须被拒绝。"""
    _reject_case(
        BSND_PARAMS,
        cand_blocks_shape=64,
        kwargs_override={"seqused_q": torch.zeros(1, dtype=torch.int32)},
    )


@pytest.mark.ci
def test_reject_r18_output_idx_offset_present():
    """R18：arch22 不消费 output_idx_offset，非空必须被拒绝。"""
    _reject_case(
        BSND_PARAMS,
        cand_blocks_shape=64,
        kwargs_override={
            "output_idx_offset": torch.zeros(3, dtype=torch.int32),
        },
    )


@pytest.mark.ci
def test_reject_r18_metadata_present():
    """R18：arch22 使用运行时 SplitCore，metadata 非空必须被拒绝。"""
    _reject_case(
        BSND_PARAMS,
        cand_blocks_shape=64,
        kwargs_override={"metadata": torch.zeros(1024, dtype=torch.int32)},
    )


@pytest.mark.ci
def test_accept_c7_block_length_empty_tensor():
    """C7 反向：candidate_block_length 传 numel==0 的空 tensor 应被接受（合法路径）。"""
    _skip_if_unavailable()
    rejected, message = sli_golden.sparse_host_reject(
        BSND_PARAMS,
        cand_blocks_shape=64,
        block_length=torch.zeros(0, dtype=torch.int32),
    )
    # 合法配置不应被 host 拒绝；若因数据/环境问题报错则 failed 而非 silently pass
    assert not rejected, (
        f"empty candidate_block_length (numel==0) must be accepted, but got: {message}"
    )


@pytest.mark.ci
def test_reject_c6_return_value_2():
    """C6：return_value=2（∉{0,1}）必须被拒绝（S10 前置校验保留）。"""
    _reject_case(
        BSND_PARAMS,
        cand_blocks_shape=64,
        kwargs_override={"return_value": 2},
    )


@pytest.mark.ci
def test_accept_c6_return_value_1():
    """C6 反向：return_value=1 必须被 host 接受（R5 反转：values 真实分输出开放）。"""
    _skip_if_unavailable()
    rejected, message = sli_golden.sparse_host_reject(
        BSND_PARAMS,
        cand_blocks_shape=64,
        kwargs_override={"return_value": 1},
    )
    # 合法配置不应被 host 拒绝；若因数据/环境问题报错则 failed 而非 silently pass
    assert not rejected, f"return_value=1 must be accepted, but got: {message}"


@pytest.mark.ci
def test_reject_c3_candblocks_zero():
    """C3：candBlocks=0（(0, 2048] 下界外）必须被拒绝。"""
    _reject_case(BSND_PARAMS, cand_blocks_shape=0)


@pytest.mark.ci
def test_reject_c8_cmp_ratio_over_128():
    """C8：cmp_ratio=256（2 的幂但超 (0, 128] 上界）必须被拒绝。"""
    _reject_case(
        BSND_PARAMS,
        cand_blocks_shape=64,
        kwargs_override={"cmp_ratio": 256},
    )
