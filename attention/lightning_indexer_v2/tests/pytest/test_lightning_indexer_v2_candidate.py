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

"""LightningIndexerV2 candidate（两级 TopK source）单用例测试套（仅 arch22 / A2）。

覆盖 candidate 新功能面（candidate_topk_blocks != -1 开启）：
  * candidate_block_size 全 6 档（2/4/8/16/32/64）；
  * candidate_topk_blocks 边界（64 下界 / 2048 上界 / numBlocks<candBlocks 满覆盖退化）；
  * layout 泛化（BSND×BSND / TND×TND / BSND×PA / TND×PA —— TND 输出 [T,N2,K]）；
  * mask_mode / cmp_ratio / 变长 / dtype 组合；
  * return_value 维度（candidate 模式已放开，值域 ∈{0,1}）：rv=1 时 sparse_values 输出
    位置级 topk 真实分（float32，shape 与 sparse_indices 相同，无效槽 -inf），与现网
    lightning_indexer 入口 rv=1 语义一致；rv=0 恒空 (0,)；
  * off（-1）回归：candidate 关闭时输出 (0,) 且 sparse 与现网一致。

Per case 校验三件套（与 test_lightning_indexer_v2_model.py 相同）：
  1. sparse_indices vs golden（check_result；source 不得改变位置级 topk）；
  2. candidate_block_length 恒空 (0,)；
  3. candidate_block_indices vs golden（check_result_candidate；行级块号集合 + -1 槽数）。
rv=1 用例追加：sparse_values shape/dtype 断言 + 与 golden 位置级真实分按同族口径
（rtol=5e-3 / atol=2.5e-5，equal_nan）isclose 比对。

Run:
    python3 -m pytest -rA -s test_lightning_indexer_v2_candidate.py -v -m ci
    LIV2_CAND_CASE_NAMES=LIV2_CAND_BS_2 python3 -m pytest -s test_lightning_indexer_v2_candidate.py -m ci
"""

import os

import pytest

# ---- guarded imports: keep the module collectable on hosts without NPU / built op package
_GOLDEN_IMPORT_ERROR = None
try:
    import torch
    import torch_npu  # noqa: F401
    import lightning_indexer_v2_golden as golden
    import result_compare_method as rc
except Exception as _exc:  # noqa: BLE001
    torch = None
    golden = None
    rc = None
    _GOLDEN_IMPORT_ERROR = _exc


BASE_FIELDS = [
    "batch_size",
    "q_seq",
    "k_seq",
    "q_t_size",
    "k_t_size",
    "q_head_num",
    "k_head_num",
    "head_dim",
    "block_size",
    "block_num",
    "qk_dtype",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "seqused_q",
    "seqused_k",
    "cmp_residual_k",
    "output_idx_offset",
    "layout_q",
    "layout_k",
    "topk",
    "mask_mode",
    "query_datarange",
    "key_datarange",
    "weights_datarange",
    "cmp_ratio",
    "return_value",
    "max_seqlen_q",
]
CANDIDATE_FIELDS = ["candidate_topk_blocks", "candidate_block_size"]
ALL_FIELDS = BASE_FIELDS + CANDIDATE_FIELDS

# 开启 candidate 的缺省档位
CAND_BLOCKS_DEFAULT = 64
CAND_BLOCK_SIZE_DEFAULT = 8


def _cand_case(
    name,
    candidate_topk_blocks=CAND_BLOCKS_DEFAULT,
    candidate_block_size=CAND_BLOCK_SIZE_DEFAULT,
    batch_size=1,
    q_seq=8,
    k_seq=512,
    q_head_num=4,
    qk_dtype=torch.float16 if torch is not None else None,
    seqused_k=None,
    cmp_residual_k=None,
    layout_q="BSND",
    layout_k="BSND",
    q_t_size=0,
    k_t_size=0,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    block_size=None,
    block_num=None,
    topk=64,
    mask_mode=0,
    cmp_ratio=1,
    return_value=0,
):
    """Build one candidate case dict（单一组合，不做笛卡尔展开）。"""
    params = {
        "batch_size": batch_size,
        "q_seq": q_seq,
        "k_seq": k_seq,
        "q_t_size": q_t_size,
        "k_t_size": k_t_size,
        "q_head_num": q_head_num,
        "k_head_num": 1,
        "head_dim": 128,
        "block_size": block_size,
        "block_num": block_num,
        "qk_dtype": qk_dtype,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "seqused_q": None,
        "seqused_k": seqused_k,
        "cmp_residual_k": cmp_residual_k,
        "output_idx_offset": None,
        "layout_q": layout_q,
        "layout_k": layout_k,
        "topk": topk,
        "mask_mode": mask_mode,
        "query_datarange": [-2.0, 2.0],
        "key_datarange": [-2.0, 2.0],
        "weights_datarange": [-2.0, 2.0],
        "cmp_ratio": cmp_ratio,
        "return_value": return_value,
        "max_seqlen_q": -1,
        "candidate_topk_blocks": candidate_topk_blocks,
        "candidate_block_size": candidate_block_size,
    }
    missing = [f for f in ALL_FIELDS if f not in params]
    if missing:
        raise ValueError(f"case '{name}' missing fields: {missing}")
    return name, params


_CASES = [
    # (a) 基线：BSND×BSND FP16
    _cand_case("LIV2_CAND_BASE"),
    # (b) candidate_block_size 全 6 档（C4 粒度扫描；S2=512 保证 numBlocks 覆盖各档）
    _cand_case("LIV2_CAND_BS_2", candidate_block_size=2),
    _cand_case("LIV2_CAND_BS_4", candidate_block_size=4),
    _cand_case("LIV2_CAND_BS_16", candidate_block_size=16),
    _cand_case("LIV2_CAND_BS_32", candidate_block_size=32),
    _cand_case("LIV2_CAND_BS_64", candidate_block_size=64),
    # (c) candidate_topk_blocks 边界（C3）
    #     下界 64：S2=512/bs=8 → numBlocks=64 == candBlocks，恰好满覆盖（-1 槽为 0）
    _cand_case("LIV2_CAND_BLOCKS_MIN", candidate_topk_blocks=64),
    #     上界 2048：numBlocks=64 < 2048，满覆盖退化（大量 -1 槽）
    _cand_case("LIV2_CAND_BLOCKS_MAX", candidate_topk_blocks=2048),
    #     numBlocks(128) < candBlocks(192...) 非常规档：candBlocks=128，numBlocks=64，全选+尾 -1
    _cand_case("LIV2_CAND_BLOCKS_GT_NUMBLOCKS", candidate_topk_blocks=128),
    # (d) layout 泛化：TND×TND（输出 [T,N2,K]）
    _cand_case(
        "LIV2_CAND_TND_TND",
        layout_q="TND",
        layout_k="TND",
        batch_size=2,
        q_seq=4,
        k_seq=192,
        q_t_size=6,
        k_t_size=192,
        cu_seqlens_q=[0, 2, 6],
        cu_seqlens_k=[0, 64, 192],
    ),
    # (e) layout 泛化：BSND×PA_BBND（block_table 链路）
    _cand_case(
        "LIV2_CAND_PA",
        layout_k="PA_BBND",
        block_size=128,
        block_num=4,
        seqused_k=[512],
    ),
    # (f) layout 泛化：TND×PA_BBND
    #     block_num 按 batch 独占物理块计：2 batch × ceil(512/128)=4 块 = 8 块
    _cand_case(
        "LIV2_CAND_TND_PA",
        layout_q="TND",
        layout_k="PA_BBND",
        batch_size=2,
        q_seq=4,
        q_t_size=6,
        k_seq=512,
        cu_seqlens_q=[0, 2, 6],
        block_size=128,
        block_num=8,
        seqused_k=[512, 512],
    ),
    # (g) mask_mode=3 + cmp_ratio=2（causal 可达前缀 + 压缩残差）
    _cand_case(
        "LIV2_CAND_MASK3_CMP2",
        k_seq=513,
        seqused_k=[513],
        cmp_residual_k=[1],
        mask_mode=3,
        cmp_ratio=2,
    ),
    # (h) mask_mode=0 + cmp_ratio=128（上界）
    _cand_case(
        "LIV2_CAND_CMP128",
        k_seq=520,
        seqused_k=[520],
        cmp_ratio=128,
    ),
    # (i) BF16 dtype
    _cand_case("LIV2_CAND_BF16", qk_dtype=torch.bfloat16),
    # (j) B>1 变长 seqused_k（2049 % bs=8 != 0，S2 非对齐）
    _cand_case(
        "LIV2_CAND_VARLEN",
        batch_size=2,
        k_seq=2049,
        seqused_k=[2049, 1024],
        cmp_residual_k=[1, 1],
        mask_mode=3,
        cmp_ratio=2,
    ),
    # (k) TND 变长 + mask3 + 候选（TND 输出布局上的 causal 候选选择）
    _cand_case(
        "LIV2_CAND_TND_VARLEN_MASK3",
        layout_q="TND",
        layout_k="TND",
        batch_size=2,
        q_seq=4,
        k_seq=192,
        q_t_size=6,
        k_t_size=192,
        cu_seqlens_q=[0, 2, 6],
        cu_seqlens_k=[0, 129, 192],
        cmp_residual_k=[1, 1],
        mask_mode=3,
        cmp_ratio=2,
    ),
    # (l) topk=2048（candidate 开启时 topk 上界）
    _cand_case(
        "LIV2_CAND_TOPK_MAX",
        k_seq=2048,
        topk=2048,
        candidate_topk_blocks=128,
    ),
    # (m) topk=1（candidate 开启时 topk 下界）
    _cand_case("LIV2_CAND_TOPK_MIN", topk=1),
    # (n) S2 尾块：k_seq=513（非 128 对齐），尾块 -inf 补齐
    _cand_case("LIV2_CAND_S2_TAIL", k_seq=513),
    # (o) S1 大行数（多 S1 行的候选并行走 S1Base 分支）
    _cand_case("LIV2_CAND_S1_LARGE", q_seq=64, q_head_num=8),
    # (p) off（-1）回归：candidate 关闭，输出 (0,) 且 sparse 与现网一致
    #     （candidate 与 return_value 的组合面由 (r) 组 rv=1 用例覆盖）
    _cand_case(
        "LIV2_CAND_OFF_REGRESSION",
        candidate_topk_blocks=-1,
        k_seq=512,
        return_value=0,
    ),
    # (r) return_value 维度（candidate 模式已放开，rv∈{0,1}）：
    #     rv=1 时 sparse_values 输出位置级 topk 真实分（shape 与 sparse_indices 相同）
    _cand_case("LIV2_CAND_RV1_BASE", return_value=1),
    #     TND 变体：rv=1 输出 [T,N2,topk] float32
    _cand_case(
        "LIV2_CAND_RV1_TND_TND",
        return_value=1,
        layout_q="TND",
        layout_k="TND",
        batch_size=2,
        q_seq=4,
        k_seq=192,
        q_t_size=6,
        k_t_size=192,
        cu_seqlens_q=[0, 2, 6],
        cu_seqlens_k=[0, 64, 192],
    ),
    #     causal（mask3）+ cmp_ratio=2：rv=1 值含不可达 -inf 槽
    _cand_case(
        "LIV2_CAND_RV1_MASK3_CMP2",
        return_value=1,
        k_seq=513,
        seqused_k=[513],
        cmp_residual_k=[1],
        mask_mode=3,
        cmp_ratio=2,
    ),
]

_CASE_MAP = dict(_CASES)

requested_names = {
    name.strip()
    for name in os.environ.get("LIV2_CAND_CASE_NAMES", "").split(",")
    if name.strip()
}


def _build_cases():
    cases = []
    matched = set()
    for name, params in _CASES:
        if requested_names and name not in requested_names:
            continue
        matched.add(name)
        cases.append(pytest.param(dict(params), id=name))
    unknown = sorted(requested_names - matched)
    if unknown:
        raise ValueError(f"unknown candidate case(s): {unknown}")
    return cases


CASES = _build_cases()


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


def _collect_skip_reason():
    if golden is None:
        return f"lightning_indexer_v2_golden import failed: {_GOLDEN_IMPORT_ERROR!r}"
    return golden.candidate_ops_unavailable_reason()


_SKIP_REASON = _collect_skip_reason()


@pytest.mark.ci
@pytest.mark.parametrize(
    "case", CASES if CASES else [pytest.param(None, id="no_cases")]
)
def test_liv2_candidate(case):
    if _SKIP_REASON:
        pytest.skip(_SKIP_REASON)
    if case is None:
        pytest.skip("no candidate cases collected")

    arch = _device_arch()
    if arch == "arch35":
        pytest.skip(
            "candidate source is arch22-only (Ascend910B / Ascend910_93); "
            "candidate-on is host-rejected on arch35 (Ascend950)"
        )

    test_data = tuple(case[f] for f in BASE_FIELDS)
    candidate_topk_blocks = int(case["candidate_topk_blocks"])
    candidate_block_size = int(case["candidate_block_size"])
    candidate_off = candidate_topk_blocks == golden.CANDIDATE_TOPK_BLOCKS_OFF

    op, op_name = golden.resolve_candidate_op()
    if op is None:
        pytest.skip(
            "candidate torch op not registered (looked for "
            f"{golden._CANDIDATE_OP_NAMES}); kernel/op_api still under development"
        )

    torch_npu.npu.set_device(0)
    return_value = int(case["return_value"])
    print(
        f"[case] B={case['batch_size']} S1={case['q_seq']} S2={case['k_seq']} "
        f"N1={case['q_head_num']} topk={case['topk']} "
        f"candBlocks={candidate_topk_blocks} bs={candidate_block_size} "
        f"mask={case['mask_mode']} cmp={case['cmp_ratio']} rv={return_value} "
        f"layout_q={case['layout_q']} layout_k={case['layout_k']} "
        f"dtype={case['qk_dtype']} arch={arch} op={op_name}"
    )

    out = golden.liv2_output_candidate_single(
        test_data,
        candidate_topk_blocks,
        candidate_block_size,
        return_value=return_value,
    )

    # ---- (a) sparse_indices regression: source must not change the position-level topk
    #     （return_value 已放开：rv 仅影响 sparse_values 输出，indices 恒不变）
    sparse_result, sparse_pct = rc.check_result(
        out["cpu_sparse"],
        out["npu_sparse"],
        out["topk_value"],
        case["output_idx_offset"],
        test_data,
        out["cpu_topk_value"],
        out["npu_sparse_value"],
    )
    print(f"sparse_indices result={sparse_result} fulfill={sparse_pct}")
    assert sparse_result == "Pass", (
        "candidate source changed sparse_indices (must be bit-stable)"
    )

    # ---- (b) candidate_block_length is a reserved empty tensor
    blk_len = out["npu_candidate_block_length"]
    assert blk_len is not None and blk_len.numel() == 0, (
        f"candidate_block_length must be empty (0,), got numel="
        f"{0 if blk_len is None else blk_len.numel()}"
    )

    # ---- (c) candidate_block_indices: row-level block-id set + -1 slot hard check
    if candidate_off:
        # off：candidateBlockIndices 输出 shape (0,)
        cand_out = out["npu_candidate"]
        assert cand_out.numel() == 0, (
            f"candidate off must yield empty (0,) output, got shape {tuple(cand_out.shape)}"
        )
    else:
        cand_result, cand_pct = rc.check_result_candidate(
            out["cpu_candidate"],
            out["npu_candidate"],
            test_data,
            candidate_topk_blocks,
            candidate_block_size,
            expect_blk_score=out["cpu_blk"],
        )
        print(f"candidate_block_indices result={cand_result} fulfill={cand_pct}")
        assert cand_result == "Pass", "candidate_block_indices mismatch vs golden"

    # ---- (d) return_value=1：sparse_values 输出位置级 topk 真实分
    if return_value:
        npu_sparse_value = out["npu_sparse_value"]
        assert npu_sparse_value.dtype == torch.float32, (
            f"sparse_values must be float32, got {npu_sparse_value.dtype}"
        )
        assert tuple(npu_sparse_value.shape) == tuple(out["npu_sparse"].shape), (
            f"sparse_values shape {tuple(npu_sparse_value.shape)} must equal "
            f"sparse_indices shape {tuple(out['npu_sparse'].shape)}"
        )
        rv_result, rv_pct = rc.check_result_return_value(
            out["cpu_topk_value"], npu_sparse_value, test_data
        )
        print(f"sparse_values(rv=1) result={rv_result} fulfill={rv_pct}")
        assert rv_result == "Pass", "sparse_values mismatch vs position-level golden"
    else:
        # rv=0：sparse_values 恒空 (0,)
        rv0_value = out["npu_sparse_value"]
        assert rv0_value is not None and rv0_value.numel() == 0, (
            f"return_value=0 must yield empty (0,) sparse_values, got shape "
            f"{tuple(rv0_value.shape)}"
        )


@pytest.mark.ci
def test_liv2_candidate_rv1_consistency():
    """rv 开关一致性：同输入下 rv=0/rv=1 的 sparse_indices 与 candidate_block_indices 逐位一致。

    candidate 模式放开 return_value 的语义核心：rv 只增发 sparse_values（位置级 topk
    真实分），不得扰动位置级 topk 管线与块级候选选择。本用例以同一份输入数据分别以
    rv=0 / rv=1 调用 candidate 入口，做逐位比对，并复核 rv=1 values 与 golden 的
    位置级真实分按同族口径（rtol=5e-3 / atol=2.5e-5，equal_nan）isclose。
    """
    if _SKIP_REASON:
        pytest.skip(_SKIP_REASON)
    arch = _device_arch()
    if arch == "arch35":
        pytest.skip(
            "candidate source is arch22-only (Ascend910B / Ascend910_93); "
            "candidate-on is host-rejected on arch35 (Ascend950)"
        )

    params = _CASE_MAP["LIV2_CAND_RV1_BASE"]
    assert int(params["return_value"]) == 1
    test_data = tuple(params[f] for f in BASE_FIELDS)
    candidate_topk_blocks = int(params["candidate_topk_blocks"])
    candidate_block_size = int(params["candidate_block_size"])

    data, out_rv0, out_rv1 = golden.liv2_output_candidate_rv_pair(
        test_data, candidate_topk_blocks, candidate_block_size
    )
    idx0, val0, cand0, blen0 = out_rv0
    idx1, val1, cand1, blen1 = out_rv1

    # rv 开关不得扰动 indices / candidate 输出（逐位一致）
    assert torch.equal(idx0, idx1), "rv=1 must not change sparse_indices (bit-exact)"
    assert torch.equal(cand0, cand1), (
        "rv=1 must not change candidate_block_indices (bit-exact)"
    )
    assert blen0.numel() == 0 and blen1.numel() == 0

    # rv=1 values：shape/dtype + 与 golden 位置级真实分 isclose
    assert tuple(val1.shape) == tuple(idx1.shape), (
        f"sparse_values shape {tuple(val1.shape)} must equal sparse_indices shape "
        f"{tuple(idx1.shape)}"
    )
    assert val1.dtype == torch.float32, (
        f"sparse_values must be float32, got {val1.dtype}"
    )
    assert val0.numel() == 0, "rv=0 sparse_values must be empty (0,)"

    rv_result, rv_pct = rc.check_result_return_value(
        data["cpu_topk_value"], val1, test_data
    )
    print(f"rv-pair sparse_values(rv=1) result={rv_result} fulfill={rv_pct}")
    assert rv_result == "Pass", "rv=1 sparse_values mismatch vs position-level golden"
