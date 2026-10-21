#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""CPU golden + NPU driver for SparseLightningIndexer (candidate consumer / candidate_mode==2).

Golden formalization (per batch b, query row i, N2 head n) — 无 leak 两档掩码语义，
唯一真值 = docs/spec.yaml math_semantics.formula（L302-337，对齐 QSLI DSL golden）：
    score(b,i,:)    fp32 index_score (GeneralizedLIV2 V1 DoReduce output), -inf at unreachable positions
    vl(b,i)         reachable prefix length (== kernel cuRealAcSeq; finite count of the score row)
    blockSize       candidate_block_size in [2,64], power of two
    candSet(b,i)    {c : c in candidate_block_indices(b,i,:), c >= 0, c < numBlocks}
                    # 脏数据（重复/越界/-1）免疫：一律按"非候选"处理
    valid(b,i,p)  = (p < vl) ∧ (floor(p/blockSize) ∈ candSet(b,i))
    score'(b,i,p) = score(b,i,p)   若 valid
                  = -inf           其它（可达但不在候选内的位置与不可达位置合并为同一无效档）
    nvalid(b,i)   = |{p : score'(b,i,p) > -inf}|
    sparse_indices(b,i,k) = topK(score')_k   若 k < min(topk, nvalid)
                          = -1               否则（候选内有效数不足 topk 的槽位 -1 补齐；全无效行全 -1，
                                              不输出任何候选外索引）
    sparse_values(b,i,k)  = score(b,i, sparse_indices(b,i,k))  若 return_value=1 且 k 为有效槽
                          = 0                  无效槽（idx=-1 对应位）；return_value=0 时恒空 (0,)
排序仅承诺：有效槽为候选内可达位置按分数降序；-1 槽填充次序与真实分同分次序均不承诺
（比对走多重集合口径 result_compare_method.check_result 三重门，禁止逐槽位强比）。

有限值约定：测试输入 q/k/w 由 datarange 有限区间注入、保证有限值，golden 不处理 NaN 输入；
q/k/w 注入 NaN 不在测试范围（equal_nan 比对口径仅为 spec extreme nan_propagates 的防御性承载）。

The LIV2 golden infrastructure (GeneralizedLIV2 / generate_liv2_test_data) is imported from the
sibling lightning_indexer_v2 op directory (shared golden module, O16).
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

_LIV2_PYTEST_DIR = (
    Path(__file__).resolve().parents[2]
    / ".."
    / "lightning_indexer_v2"
    / "tests"
    / "pytest"
).resolve()
if str(_LIV2_PYTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_LIV2_PYTEST_DIR))

# Pure, dependency-light golden helpers (numpy/torch only) are imported eagerly so the consumer
# golden core can be exercised on a pure-CPU host (self-check) without torch_npu/cann_ops_transformer.
_LIV2_IMPORT_ERROR = None
try:
    import result_compare_method as rc  # noqa: E402
    import sli_candidate_golden as cand_golden  # noqa: E402
except Exception as _exc:  # noqa: BLE001
    rc = None
    cand_golden = None
    _LIV2_IMPORT_ERROR = _exc

# lightning_indexer_v2_golden transitively imports cann_ops_transformer (heavy JIT import).  It is
# only needed by the NPU driver, so it is imported lazily to keep this module cheap to import on CPU.
liv2_golden = None


def _ensure_liv2_golden():
    """Lazily import the LIV2 golden infra (NPU driver only); raise SparseOpUnavailableError on failure."""
    global liv2_golden, _LIV2_IMPORT_ERROR
    if liv2_golden is not None:
        return liv2_golden
    try:
        import lightning_indexer_v2_golden as _liv2  # noqa: E402
    except Exception as _exc:  # noqa: BLE001
        _LIV2_IMPORT_ERROR = _exc
        liv2_golden = None
        raise SparseOpUnavailableError(
            f"LIV2 golden infra import failed: {_exc!r}"
        ) from _exc
    liv2_golden = _liv2
    return liv2_golden


NEG_INF = float("-inf")


class SparseOpUnavailableError(RuntimeError):
    """Raised when the sparse_lightning_indexer torch op is not registered yet."""


# ---------------------------------------------------------------------------
# candidate 输入构造（pytest 侧）：self / dirty / missing / full
# ---------------------------------------------------------------------------


def build_candidate_self(score_bnsd, cand_blocks, block_size):
    """(a) 自洽候选：同 score 过 select_candidate_blocks_ref（真实 source 行为）。"""
    arr = np.asarray(score_bnsd, dtype=np.float32)
    return cand_golden.compute_candidate_golden_bnsd(arr, cand_blocks, block_size)


def build_candidate_full(score_bnsd, cand_blocks, block_size, rng=None):
    """(e) 全候选（覆盖全部块号）：candBlocks ≥ numBlocks 时块号 0..numBlocks-1 全列、余 -1。"""
    arr = np.asarray(score_bnsd, dtype=np.float32)
    finite = np.isfinite(arr) & (arr > NEG_INF)
    act_s2 = (
        finite.sum(axis=-1).reshape(arr.shape[0], -1).max(axis=1)
        if arr.shape[0] > 0
        else np.zeros((0,), np.int64)
    )
    out = np.full(arr.shape[:-1] + (cand_blocks,), -1, dtype=np.int32)
    for b in range(arr.shape[0]):
        s2 = int(act_s2[b])
        nb = (s2 + block_size - 1) // block_size if s2 > 0 else 0
        width = min(cand_blocks, nb)
        if width > 0:
            ids = np.arange(width, dtype=np.int32)
            out[b][..., :width] = np.broadcast_to(ids, out[b].shape[:-1] + (width,))
    return out


def build_candidate_dirty(score_bnsd, cand_blocks, block_size, rng=None, seed=0):
    """(b) 随机子集脏数据：重复块号 + 越界块号 + -1 槽（鲁棒性契约：一律按"非候选"处理）。"""
    if rng is None:
        rng = np.random.default_rng(seed)
    arr = np.asarray(score_bnsd, dtype=np.float32)
    finite = np.isfinite(arr) & (arr > NEG_INF)
    act_s2 = (
        finite.sum(axis=-1).reshape(arr.shape[0], -1).max(axis=1)
        if arr.shape[0] > 0
        else np.zeros((0,), np.int64)
    )
    out = np.full(arr.shape[:-1] + (cand_blocks,), -1, dtype=np.int32)
    for b in range(arr.shape[0]):
        nb = (
            (int(act_s2[b]) + block_size - 1) // block_size if int(act_s2[b]) > 0 else 0
        )
        if nb == 0:
            continue
        shape = out[b].shape
        flat = out[b].reshape(-1, cand_blocks)
        for r in range(flat.shape[0]):
            # 随机取 1..min(candBlocks, 2*nb) 个（含重复），再混入越界与 -1
            take = int(rng.integers(1, min(cand_blocks, 2 * nb) + 1))
            vals = rng.integers(0, nb, size=take).astype(np.int32)  # 可重复
            dirty = rng.random(take)
            vals = np.where(
                dirty < 0.1,
                -1,
                np.where(dirty < 0.2, nb + int(rng.integers(1, 64)), vals),
            ).astype(np.int32)
            flat[r, :take] = vals
        out[b] = flat.reshape(shape)
    return out


def build_candidate_missing(
    score_bnsd, cand_blocks, block_size, keep_ratio=0.5, seed=0
):
    """(c) 故意缺块：仅保留部分块号（覆盖不全），配合 vl 构造 nvalid < topk 的 -1 补坑场景。"""
    rng = np.random.default_rng(seed)
    arr = np.asarray(score_bnsd, dtype=np.float32)
    finite = np.isfinite(arr) & (arr > NEG_INF)
    act_s2 = (
        finite.sum(axis=-1).reshape(arr.shape[0], -1).max(axis=1)
        if arr.shape[0] > 0
        else np.zeros((0,), np.int64)
    )
    out = np.full(arr.shape[:-1] + (cand_blocks,), -1, dtype=np.int32)
    for b in range(arr.shape[0]):
        nb = (
            (int(act_s2[b]) + block_size - 1) // block_size if int(act_s2[b]) > 0 else 0
        )
        if nb == 0:
            continue
        keep = max(1, int(nb * keep_ratio))
        kept = rng.choice(nb, size=keep, replace=False).astype(np.int32)
        width = min(cand_blocks, kept.size)
        shape = out[b].shape
        out[b][..., :width] = np.broadcast_to(kept[:width], shape[:-1] + (width,))
    return out


# ---------------------------------------------------------------------------
# consumer golden 核心：无 leak 两档掩码（候选外可达位与不可达位统一 -inf）→ topk
# ---------------------------------------------------------------------------


def mask_score_bnsd(score_bnsd, cand_bnsd, block_size):
    """无 leak 掩码：在 w-sum + causal mask 之后、sort 之前把无效位统一置 -inf。

    对齐 spec formula（docs/spec.yaml math_semantics.formula，两档语义）：
        score'(b,i,p) = score(b,i,p)  若 p < vl 且 floor(p/blockSize) ∈ candSet(b,i)
                      = -inf          其它（可达但不在候选内的位置 与 p >= vl 不可达位
                                          合并为同一无效档；候选外位置一律不入选）

    有限值约定：score 输入保证有限值（q/k/w datarange 有限区间注入），golden 不处理
    NaN 输入；q/k/w 注入 NaN 不在测试范围。

    Args:
        score_bnsd: [B, N2, S1, S2max] fp32 score（-inf 在不可达位置）。
        cand_bnsd:  [B, N2, S1, K] int32 候选块号（脏数据免疫：越界/-1 视为非候选）。
        block_size: candidate_block_size。

    Returns:
        masked: [B, N2, S1, S2max] fp32，score'（候选外可达位置与不可达位置 → -inf）。
    """
    arr = np.asarray(score_bnsd, dtype=np.float32)
    cand = np.asarray(cand_bnsd, dtype=np.int64)
    out = arr.copy()
    B, N2, S1, S2max = arr.shape
    for b in range(B):
        for n in range(N2):
            for i in range(S1):
                row = arr[b, n, i]
                finite = np.isfinite(row) & (row > NEG_INF)
                vl = int(finite.sum())
                if vl == 0:
                    continue
                cand_row = cand[b, n, i]
                cand_set = set(int(c) for c in cand_row if int(c) >= 0)
                pos = np.arange(vl, dtype=np.int64)
                blk = pos // block_size
                in_cand = (
                    np.isin(blk, np.fromiter(cand_set, dtype=np.int64))
                    if cand_set
                    else np.zeros(vl, bool)
                )
                out[b, n, i, :vl] = np.where(in_cand, row[:vl], NEG_INF).astype(
                    np.float32
                )
                out[b, n, i, vl:] = NEG_INF
    return out


def consumer_topk_golden_bnsd(masked_bnsd, topk):
    """掩码后走原 sort/topk：行级稳定降序（tie 按索引升序），-inf 槽 → -1。

    无 leak 掩码（mask_score_bnsd）后 -inf 位自然落 -1：nvalid 不足 topk 的剩余槽位
    由预填 -1 补齐（全无效行全 -1）；-1 槽填充次序与同分次序均不承诺（多重集合口径）。

    注意：无 leak 掩码后行内有效位（finite）不再保证是连续前缀（候选块号可以不从
    block 0 起，前缀头部出现 -inf 空洞），故排序域为整行，与 spec formula 的全行
    argsort 逐条对应（L329）：-inf 位在降序排序中自然沉底，候选外无效位与 ≥vl
    不可达位同样经 valid 判据落 -1。
    """
    arr = np.asarray(masked_bnsd, dtype=np.float32)
    B, N2, S1, S2max = arr.shape
    out = np.full((B, N2, S1, topk), -1, dtype=np.int32)
    take = min(topk, S2max)
    for b in range(B):
        for n in range(N2):
            for i in range(S1):
                row = arr[b, n, i]
                if not np.any(np.isfinite(row) & (row > NEG_INF)):
                    continue
                # 稳定降序（整行）：值相同保持索引升序（与 kernel (value,idx) 对排序
                # tie-break 一致）；-inf 位沉底，经 valid 判据落 -1
                order = np.argsort(-row, kind="stable")
                sel = order[:take]
                valid = row[sel] > NEG_INF
                out[b, n, i, : sel.size] = np.where(valid, sel, -1).astype(np.int32)
    return out


def compute_consumer_golden(
    score_bnsd, cand_bnsd, topk, block_size, layout_q, cu_seqlens_q=None, return_value=0
):
    """consumer golden 主入口：无 leak 掩码 + topk（rv=1 时增 values 产出），
    并转到算子输出布局（BSND [B,S1,N2,K] / TND [T,N2,K]）。

    Returns:
        (indices, values, masked_score):
            indices: int32 输出布局，无效槽 -1。
            values:  float32 输出布局连续内存（比较器按 [total_rows, topk] reshape）；
                     rv=1 时有效槽为入选位置真实分、idx=-1 对应槽 0
                     （spec formula：where(sel_valid, take_along_axis(score, sel), 0)）；
                     rv=0 时恒空 (0,)（kernel 不写入）。
            masked_score: [B,N2,S1,S2max] fp32 score'（比较器边界仲裁用）。
    """
    masked = mask_score_bnsd(score_bnsd, cand_bnsd, block_size)
    idx_bnsd = consumer_topk_golden_bnsd(masked, topk)
    idx = cand_golden.bnsd_to_output_layout(idx_bnsd, layout_q, cu_seqlens_q, fill=-1)
    if int(return_value) == 1:
        # values：有效槽（idx != -1）取真实分（掩码位上 score' == score），无效槽 0
        idx_i64 = np.asarray(idx_bnsd, dtype=np.int64)
        safe_idx = np.clip(idx_i64, 0, None)  # -1 槽钳到 0，取值后由 where 归零
        gathered = np.take_along_axis(
            np.asarray(score_bnsd, dtype=np.float32), safe_idx, axis=-1
        )
        values_bnsd = np.where(idx_i64 >= 0, gathered, np.float32(0)).astype(np.float32)
        values = cand_golden.bnsd_to_output_layout(
            values_bnsd, layout_q, cu_seqlens_q, fill=0
        )
        values_out = np.ascontiguousarray(values, dtype=np.float32)
    else:
        values_out = np.zeros((0,), dtype=np.float32)
    return (
        torch.from_numpy(np.ascontiguousarray(idx)).to(torch.int32),
        values_out,
        masked,
    )


# ---------------------------------------------------------------------------
# NPU driver
# ---------------------------------------------------------------------------

_SPARSE_OP_NAMES = ("sparse_lightning_indexer",)

# 算子包直连查找清单（按优先级）：
#   1. 裁剪 vendor 包（cann_ops_transformer_custom_li，随 LI/SLI 专项包发布）
#   2. 通用 custom 包（cann_ops_transformer_custom）
#   3. 主干包（cann_ops_transformer，SLI 已随主干发布，与 LIV2 的
#      resolve_candidate_op 主干解析口径一致）
# 每个包先查包根（__init__ re-export 算子符号），再查 .ops 子包；包导入失败或算子缺失
# 一律 (None, None)（调用方转 SparseOpUnavailableError -> pytest skip），测试侧不做 torch.ops 探测。
_SPARSE_OPS_PACKAGES = (
    "cann_ops_transformer_custom_li",
    "cann_ops_transformer_custom",
    "cann_ops_transformer",
)


def resolve_sparse_op():
    """Return (callable, name) for the sparse torch op, or (None, None) if unavailable.

    Directly import each ops package (root, then the ``ops`` submodule) and check the
    ``sparse_lightning_indexer`` symbol; package import failure or missing op yields
    (None, None) (caller converts to SparseOpUnavailableError -> pytest skip).
    """
    import importlib

    for mod_name in _SPARSE_OPS_PACKAGES:
        for candidate in (mod_name, f"{mod_name}.ops"):
            try:
                ops_module = importlib.import_module(candidate)
            except Exception:  # noqa: BLE001 - uninstalled package is a normal skip path
                continue
            for name in _SPARSE_OP_NAMES:
                # the ops package uses a lazy __getattr__ that may raise for an unregistered op
                try:
                    fn = getattr(ops_module, name, None)
                except Exception:  # noqa: BLE001
                    fn = None
                if fn is not None:
                    return fn, name
    return None, None


def _to_npu(tensor):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        return tensor.npu()
    return torch.as_tensor(tensor).npu()


def prepare_sparse_npu_args(params, data, cand_block_size):
    """Build the kwargs for sparse_lightning_indexer from generate_liv2_test_data() output."""
    qk_dtype = data["query"].dtype
    query = data["query"].to(qk_dtype)
    key = data["key"].to(qk_dtype)
    block_fusion = data.get("blockFusion")
    if block_fusion is not None:
        block_size = int(params[8])
        block_num = int(params[9])
        head_dim = int(params[7])
        k_head_num = int(params[6])
        block_fusion = block_fusion.to(qk_dtype).npu()
        key = block_fusion[:, : block_size * k_head_num * head_dim].view(
            block_num, block_size, k_head_num, head_dim
        )
    else:
        key = key.npu()

    meta_in = {
        mk: (mv.npu() if torch.is_tensor(mv) else mv)
        for mk, mv in data["metadata_input"].items()
    }
    # metadata 先传空处理（临时方案，arch22）：当前环境（cann-9.0.0 libopapi）缺
    # aclnnLightningIndexerV2Metadata 符号，而 arch22 kernel 忽略 metadata 输入；
    # 与 LIV2 侧 liv2_build_metadata 的 LI_METADATA_EMPTY 开关保持一致。
    if os.environ.get("LI_METADATA_EMPTY", "1") != "0":
        metadata = None
    else:
        metadata = _ensure_liv2_golden().lightning_indexer_metadata(**meta_in).npu()
    return {
        "q": query.npu(),
        "k": key,
        "w": data["weights"].npu(),
        "cu_seqlens_q": _to_npu(data.get("cu_seqlens_q")),
        "cu_seqlens_k": _to_npu(data.get("cu_seqlens_k")),
        "seqused_q": _to_npu(data.get("seqused_q")),
        "seqused_k": _to_npu(data.get("seqused_k")),
        "cmp_residual_k": _to_npu(data.get("cmp_residual_k_for_npu")),
        "block_table": _to_npu(data.get("block_table")),
        "output_idx_offset": _to_npu(data.get("output_idx_offset")),
        "metadata": metadata,
        "max_seqlen_q": int(data["max_seqlen_q"]),
        "layout_q": data["layout_query"],
        "layout_k": data["layout_key"],
        "mask_mode": int(data["mask_mode"]),
        "cmp_ratio": int(data["cmp_ratio"]),
        # return_value 透传（G3 schema 落地后随 kwargs 下发）：LIV2 generate_liv2_test_data
        # 的 data dict 当前未携带 return_value 键（D2 已核实），防御式回退 params[25]
        # （BASE_FIELDS 的 return_value 位）；缺省 0，不破坏现网 rv=0 调用。
        "return_value": int(data.get("return_value", params[25])),
        "candidate_block_length": None,
        "candidate_block_size": int(cand_block_size),
    }


def sparse_output_single(
    params, cand_spec, cand_blocks, block_size=8, generate_golden=True
):
    """Run the CPU consumer golden and (when available) the NPU sparse op.

    Args:
        params: the 27-field LIV2 base param tuple (BASE_FIELDS).
        cand_spec: "self" | "dirty" | "missing" | "full"（候选来源）。
        cand_blocks: candidate_block_indices 输出宽度 K。
        block_size: candidate_block_size。

    Returns a dict with: cpu_result, cpu_values(rv=1 真实分 / rv=0 恒空), cpu_degraded(score'),
    topk_value(原始 score), data, npu_result, cand_npu_layout（下发的候选张量，算子布局）, op_name。
    Raises SparseOpUnavailableError when the sparse torch op is not registered yet.
    """
    if liv2_golden is None and cand_golden is None:
        raise SparseOpUnavailableError(
            f"LIV2 golden infra import failed: {_LIV2_IMPORT_ERROR!r}"
        )
    # 先解析 sparse op：包不可用/算子缺失时在做任何重活（数据生成/CPU golden）之前直接 skip
    op, op_name = resolve_sparse_op()
    if op is None:
        raise SparseOpUnavailableError(
            "sparse torch op unavailable: ops package import failed or op missing "
            f"(tried {list(_SPARSE_OPS_PACKAGES)}, looked for {list(_SPARSE_OP_NAMES)}); "
            "kernel/op_api still under development"
        )
    data = _ensure_liv2_golden().generate_liv2_test_data(
        params, generate_golden=generate_golden
    )
    score = data["topk_value"]  # BNSD [B,N2,S1,S2max] fp32
    layout_q = data["layout_query"]
    cu_seqlens_q = data.get("cu_seqlens_q")

    if cand_spec == "self":
        cand_bnsd = build_candidate_self(score, cand_blocks, block_size)
    elif cand_spec == "dirty":
        cand_bnsd = build_candidate_dirty(score, cand_blocks, block_size)
    elif cand_spec == "missing":
        cand_bnsd = build_candidate_missing(score, cand_blocks, block_size)
    elif cand_spec == "full":
        cand_bnsd = build_candidate_full(score, cand_blocks, block_size)
    else:
        raise ValueError(f"unknown cand_spec: {cand_spec}")

    return_value = int(params[25])
    cpu_result, cpu_values, cpu_degraded = compute_consumer_golden(
        score,
        cand_bnsd,
        int(data["topk"]),
        block_size,
        layout_q,
        cu_seqlens_q,
        return_value,
    )

    # 候选张量转算子输入布局（BSND [B,S1,N2,K] / TND [T,N2,K]）
    cand_layout = cand_golden.bnsd_to_output_layout(
        cand_bnsd, layout_q, cu_seqlens_q, fill=-1
    )
    cand_npu_layout = torch.from_numpy(np.ascontiguousarray(cand_layout)).to(
        torch.int32
    )

    run_kwargs = prepare_sparse_npu_args(params, data, block_size)
    q = run_kwargs.pop("q")
    k = run_kwargs.pop("k")
    w = run_kwargs.pop("w")
    outputs = op(
        q,
        k,
        w,
        int(data["topk"]),
        cand_npu_layout.npu(),
        **run_kwargs,
    )
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise SparseOpUnavailableError(
            f"sparse op '{op_name}' returned {type(outputs)} (expected a 2-tuple)"
        )
    npu_result, npu_values = outputs
    torch.npu.synchronize()

    return {
        "params": params,
        "data": data,
        "op_name": op_name,
        "cand_spec": cand_spec,
        "cand_blocks": cand_blocks,
        "block_size": block_size,
        "cpu_result": cpu_result,
        "cpu_values": cpu_values,
        "cpu_degraded": cpu_degraded,
        "topk_value": score,  # 原始 score（诊断用）
        "cand_npu_layout": cand_npu_layout,
        "npu_result": npu_result,
        "npu_values": npu_values,
    }


def sparse_host_reject(
    params,
    cand_blocks_shape,
    block_size=8,
    block_length=None,
    topk_override=None,
    cand_shape_override=None,
    kwargs_override=None,
):
    """Assert the host validation rejects an illegal consumer configuration.

    Builds the NPU inputs WITHOUT materializing the CPU golden, then calls the sparse op
    expecting it to raise. Returns (rejected: bool, message: str).

    Args:
        cand_shape_override: explicit full shape for the candidate tensor (bypasses the
            per-layout skeleton), e.g. a rank-3 shape under a BSND query (C2 rank check)
            or a wrong batch dim (C2 dim-match check).
        kwargs_override: dict merged into the run kwargs AFTER preparation, e.g.
            ``{"metadata": <tensor>, "seqused_q": <tensor>, "max_seqlen_q": 4,
            "cmp_ratio": 3, "return_value": 2}`` to exercise the arch22 hard-reject inputs
            (rv=2 → C6 拒绝；rv=1 已开放、应被接受).
    """
    op, op_name = resolve_sparse_op()
    if op is None:
        raise SparseOpUnavailableError(
            "sparse torch op unavailable: ops package import failed or op missing "
            f"(tried {list(_SPARSE_OPS_PACKAGES)}, looked for {list(_SPARSE_OP_NAMES)}); "
            "kernel/op_api still under development"
        )
    data = _ensure_liv2_golden().generate_liv2_test_data(params, generate_golden=False)
    run_kwargs = prepare_sparse_npu_args(params, data, block_size)
    q = run_kwargs.pop("q")
    k = run_kwargs.pop("k")
    w = run_kwargs.pop("w")
    topk = int(data["topk"]) if topk_override is None else int(topk_override)

    # 候选张量：按 layout 构造合法骨架（形状错误由 cand_blocks / cand_shape_override 显式控制）
    layout_q = data["layout_query"]
    if cand_shape_override is not None:
        cand = torch.full(tuple(cand_shape_override), -1, dtype=torch.int32)
        cand_last_dim = cand.shape[-1]
    else:
        cand_last_dim = (
            cand_blocks_shape[-1]
            if hasattr(cand_blocks_shape, "__len__")
            else int(cand_blocks_shape)
        )
        if layout_q == "BSND":
            cand = torch.full(
                (data["query"].shape[0], data["query"].shape[1], 1, cand_last_dim),
                -1,
                dtype=torch.int32,
            )
        else:
            cand = torch.full(
                (data["query"].shape[0], 1, cand_last_dim), -1, dtype=torch.int32
            )
    if block_length is not None:
        run_kwargs["candidate_block_length"] = block_length.npu()
    if kwargs_override:
        for key, value in kwargs_override.items():
            run_kwargs[key] = value.npu() if torch.is_tensor(value) else value
    try:
        op(q, k, w, topk, cand.npu(), **run_kwargs)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001 - host validation surfaces as a torch RuntimeError
        return True, f"{op_name} rejected as expected: {type(exc).__name__}: {exc}"
    return (
        False,
        f"{op_name} did NOT reject candBlocks={cand_last_dim}, block_size={block_size}",
    )
