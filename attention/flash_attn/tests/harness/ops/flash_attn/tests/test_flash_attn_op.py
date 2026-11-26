# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""flash_attn operator-specific unit tests (CPU)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

HARNESS_ROOT = Path(__file__).resolve().parents[4]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from harness.case import canonical_params  # noqa: E402
from harness.ops.flash_attn import PACKAGE  # noqa: E402
from harness.ops.flash_attn import layout as layoutlib  # noqa: E402
from harness.ops.flash_attn import golden_truth  # noqa: E402
from harness.ops.flash_attn import memory as memlib  # noqa: E402
from harness.golden_spec import parse_spec  # noqa: E402
from harness.ops.flash_attn.params import normalize  # noqa: E402
from harness.perf import Profiler  # noqa: E402


def test_normalize_is_idempotent_and_fills_defaults():
    once = normalize(
        {"dtype": "bf16", "B": 1, "N1": 4, "N2": 2, "S1": 256, "S2": 288, "D": 72}
    )
    twice = normalize(once)
    assert canonical_params(once) == canonical_params(twice)
    assert once["Dtype"] == "bf16" and once["dtype"] == "bf16"
    assert once["DV"] == 72
    assert once["layout_kv"] == "BNSD"
    assert (
        once["block_table"]
        if str(once["layout_kv"]).startswith("PA_")
        else "block_table" not in once
    )


def test_p_dtype_default_resolves_to_v_dtype():
    from harness.ops.flash_attn.golden_truth import _resolve_p_dtype

    assert _resolve_p_dtype("out_dtype", torch.bfloat16) == torch.bfloat16
    assert _resolve_p_dtype("out_dtype", torch.float16) == torch.float16
    assert _resolve_p_dtype("out_dtype", torch.float32) is None
    assert _resolve_p_dtype("fp32", torch.bfloat16) is None


def test_golden_truth_handles_empty_segment():
    """空 KV（seqused_kv=0）→ 输出 0、lse=inf；TND 变长正常段落不受影响。"""
    from harness.ops.flash_attn import golden_truth

    params = {
        "layout_q": "BNSD",
        "layout_kv": "BNSD",
        "layout_out": "BNSD",
        "B": 1,
        "N1": 1,
        "N2": 1,
        "S1": 2,
        "S2": 4,
        "D": 4,
        "mask_mode": 0,
        "seqused_kv": [0],
        "dtype": "fp16",
        "return_softmax_lse": True,
    }
    torch.manual_seed(0)
    inputs = {
        "q": torch.randn(1, 1, 2, 4, dtype=torch.float16),
        "k": torch.randn(1, 1, 4, 4, dtype=torch.float16),
        "v": torch.randn(1, 1, 4, 4, dtype=torch.float16),
    }
    result = golden_truth.compute(inputs, params)
    assert torch.equal(result["out"], torch.zeros_like(result["out"]))
    assert torch.isinf(result["lse"]).all()

    # TND 变长（含零长段）：token 数不变，正常段落 lse 有限
    tnd = dict(params)
    tnd.update(
        {
            "layout_q": "TND",
            "layout_kv": "TND",
            "layout_out": "TND",
            "cu_seqlens_q": [0, 2, 2, 5],
            "cu_seqlens_kv": [0, 2, 2, 5],
            "seqused_kv": None,
            "S1": 5,
            "S2": 5,
        }
    )
    tnd.pop("seqused_kv")
    tnd_inputs = {
        "q": torch.randn(5, 1, 4, dtype=torch.float16),
        "k": torch.randn(5, 1, 4, dtype=torch.float16),
        "v": torch.randn(5, 1, 4, dtype=torch.float16),
    }
    out = golden_truth.compute(tnd_inputs, tnd)
    assert tuple(out["out"].shape) == (5, 1, 4)
    assert torch.isfinite(out["lse"]).all()


def test_tiling_key_from_kernel_name():
    name = "FlashAttn_761df6ce7ae0609539d8a454b0416279_369360897"
    assert PACKAGE.tiling_key(name) == "369360897"
    assert PACKAGE.tiling_key(name + "_mix_aic") == "369360897"
    assert PACKAGE.tiling_key("FlashAttnMetadata") is None
    assert PACKAGE.tiling_key("") is None


def test_profiler_without_prof_dir():
    assert Profiler(None).collect() is None
    assert Profiler("/nonexistent/path").collect() is None
    assert PACKAGE.tiling_key(None) is None


def _mem_params(**over):
    p = {
        "B": 2,
        "N1": 8,
        "N2": 2,
        "S1": 256,
        "S2": 512,
        "D": 128,
        "DV": 128,
        "mask_mode": 3,
        "dtype": "fp16",
        "return_softmax_lse": True,
    }
    p.update(over)
    return p


def test_memory_estimate_follows_residency_and_compare_mode():
    p = _mem_params()
    cpu3 = memlib.estimate(
        p, golden_device="cpu", compare_mode="three_way", persist=False
    )
    npu3 = memlib.estimate(
        p, golden_device="npu", compare_mode="three_way", persist=False
    )
    npu2 = memlib.estimate(
        p, golden_device="npu", compare_mode="two_way", persist=False
    )

    # NPU golden：host 只剩比对分块；设备侧含 输入+输出+真值+peer
    assert npu3["host_peak"] == memlib.COMPARE_CHUNK
    assert npu3["device_peak"] >= npu2["device_peak"]
    # CPU golden：host 含输入+真值+peer，设备侧只有 op 输出+输入
    assert cpu3["host_peak"] > npu3["host_peak"]
    assert cpu3["device_peak"] < npu3["device_peak"]


def test_tensor_and_golden_footprints_are_separate():
    p = _mem_params()
    tensor = PACKAGE.tensor_footprint(p)
    npu = PACKAGE.golden_footprint(p, spec=parse_spec("device=npu"))
    cpu = PACKAGE.golden_footprint(p, spec=parse_spec("device=cpu"))
    assert tensor["inputs"] > 0
    assert npu["host_peak"] == 0 and npu["device_peak"] > tensor["inputs"]
    assert cpu["device_peak"] == 0 and cpu["host_peak"] > tensor["inputs"]


def test_memory_estimate_models_single_segment_fp32_peak():
    p = _mem_params()
    one = memlib.estimate(
        {**p, "B": 1, "S1": 256}, golden_device="npu", compare_mode="two_way"
    )
    two = memlib.estimate(
        {**p, "B": 2, "S1": 128, "S2": 256},
        golden_device="npu",
        compare_mode="two_way",
    )
    assert one["device_peak"] > two["device_peak"]


def test_memory_estimate_prefers_paged_pa_blocks():
    p = _mem_params()
    p.update(
        {
            "layout_kv": "PA_BBND",
            "seqused_kv": [512, 512],
            "block_size": 128,
            "actual_b": 2,
        }
    )
    dense = memlib.estimate(
        p,
        golden_device="npu",
        compare_mode="two_way",
        golden_spec=parse_spec(
            "device=npu,p_dtype=out_dtype,softmax_mode=online,kv_loader=dense"
        ),
    )
    paged = memlib.estimate(
        p,
        golden_device="npu",
        compare_mode="two_way",
        golden_spec=parse_spec(
            "device=npu,p_dtype=out_dtype,softmax_mode=online,kv_loader=paged"
        ),
    )
    assert paged["device_peak"] < dense["device_peak"]


def test_golden_peak_only_models_full_k_t_when_needed():
    p = _mem_params()
    footprint = PACKAGE.tensor_footprint(p)
    full = golden_truth.golden_peak(
        footprint, p, device="npu", chunk_bytes=1024**3, p_dtype="fp32"
    )
    blocked = golden_truth.golden_peak(
        footprint,
        p,
        device="npu",
        chunk_bytes=1024**3,
        p_dtype="out_dtype",
        p_block=128,
    )
    assert blocked["device_peak"] < full["device_peak"]


def test_memory_estimate_persist_and_scale():
    p = _mem_params()
    base = memlib.estimate(
        p, golden_device="npu", compare_mode="two_way", persist=False
    )
    kept = memlib.estimate(p, golden_device="npu", compare_mode="two_way", persist=True)
    assert kept["host_peak"] > base["host_peak"]

    bigger = memlib.estimate(
        _mem_params(S1=512), golden_device="npu", compare_mode="two_way", persist=False
    )
    assert bigger["device_peak"] > base["device_peak"]


def test_policy_p_block_matches_arch35_rules():
    """P 分块口径 = kernel s2BaseSize（arch35 模板路由）。"""
    from harness.ops.flash_attn import policy

    def block(**over):
        params = {
            "N1": 8,
            "N2": 8,
            "D": 128,
            "DV": 128,
            "S1": 1024,
            "S2": 1024,
            "mask_mode": 0,
            "win_left": -1,
        }
        params.update(over)
        return policy.resolve_p_block(params)

    assert (
        policy.resolve_template(
            {
                "N1": 8,
                "N2": 8,
                "D": 128,
                "DV": 128,
                "S1": 1024,
                "S2": 1024,
                "mask_mode": 3,
            }
        )
        == policy.TEMPLATE_DN_UNMERGED
    )
    assert block(mask_mode=3) == 256  # DN_Unmerged（mask 0/3 + maxQ>=256）
    assert block() == 256  # mask 0 同样走 DN_Unmerged
    assert block(S1=128) == 128  # maxQ<256 -> DN
    assert block(DV=256) == 128  # ND（vHeadDim>=256，不触发 M 调整）
    assert block(DV=256, S1=2) == 256  # ND + M 调整（group*maxQ<64）
    assert block(S1=1, S2=1024, mask_mode=3) == 256  # decode -> ND + M 调整
    assert block(S1=64, S2=1024, mask_mode=4, win_left=64) == 128  # winLeft<=128 不调整
    assert block(S1=64, S2=1024, mask_mode=4, win_left=-1) == 256


def _manual_peer(q, k, v, scale, group, p_block, mask_mode=0):
    """独立实现 kernel 口径：分母用 fp32 P，P@V 用落盘到 V dtype 的 P。"""
    n1, s1, _ = q.shape
    s2 = k.shape[1]
    qf, kf, vf = q.float(), k.float(), v.float()
    full_mask = None
    if mask_mode == 3:
        q_pos = torch.arange(s1) + (s2 - s1)
        full_mask = torch.arange(s2).unsqueeze(0) > q_pos.unsqueeze(1)
    out = torch.zeros(n1, s1, v.shape[-1])
    lse = torch.full((n1, s1), float("inf"))
    for h in range(n1):
        kk, vv = kf[h // group], vf[h // group]
        m = torch.full((s1, 1), torch.finfo(torch.float32).min)
        s = torch.zeros(s1, 1)
        acc = torch.zeros(s1, v.shape[-1])
        for j in range(0, s2, p_block):
            stop = min(j + p_block, s2)
            scores = (qf[h] @ kk[j:stop].T) * scale
            if full_mask is not None:
                scores = scores.masked_fill(full_mask[:, j:stop], float("-inf"))
            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
            probs = torch.exp(scores - m_new)
            prob_mm = probs.to(v.dtype).float()
            alpha = torch.exp(m - m_new)
            s = s * alpha + probs.sum(dim=-1, keepdim=True)
            acc = acc * alpha + prob_mm @ vv[j:stop]
            m = m_new
        out[h] = acc / s
        lse[h] = (torch.log(s) + m).squeeze(-1)
    return out, lse


def _manual_peer_cast_sum(q, k, v, scale, group, p_block):
    """错误口径（对照）：分母也用落盘后的 P。"""
    n1, s1, _ = q.shape
    s2 = k.shape[1]
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.zeros(n1, s1, v.shape[-1])
    lse = torch.full((n1, s1), float("inf"))
    for h in range(n1):
        kk, vv = kf[h // group], vf[h // group]
        m = torch.full((s1, 1), torch.finfo(torch.float32).min)
        s = torch.zeros(s1, 1)
        acc = torch.zeros(s1, v.shape[-1])
        for j in range(0, s2, p_block):
            stop = min(j + p_block, s2)
            scores = (qf[h] @ kk[j:stop].T) * scale
            m_new = torch.maximum(m, scores.max(dim=-1, keepdim=True).values)
            probs = torch.exp(scores - m_new).to(v.dtype).float()
            alpha = torch.exp(m - m_new)
            s = s * alpha + probs.sum(dim=-1, keepdim=True)
            acc = acc * alpha + probs @ vv[j:stop]
            m = m_new
        out[h] = acc / s
        lse[h] = (torch.log(s) + m).squeeze(-1)
    return out, lse


def test_peer_p_cast_is_mm_only():
    """P 落盘只作用于 P@V；分母/lse 必须用 fp32 P（kernel 的 sum->cast->MM 顺序）。"""
    from harness.ops.flash_attn import golden_truth

    params = {
        "layout_q": "BNSD",
        "layout_kv": "BNSD",
        "layout_out": "BNSD",
        "B": 1,
        "N1": 2,
        "N2": 1,
        "S1": 3,
        "S2": 5,
        "D": 4,
        "DV": 4,
        "mask_mode": 0,
        "dtype": "bf16",
        "return_softmax_lse": True,
    }
    torch.manual_seed(0)
    q = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    k = torch.randn(1, 1, 5, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 1, 5, 4, dtype=torch.bfloat16)
    res = golden_truth.compute(
        {"q": q, "k": k, "v": v}, params, p_dtype="out_dtype", p_block=2
    )
    scale = 1.0 / math.sqrt(4)
    good_out, good_lse = _manual_peer(q[0], k[0], v[0], scale, 2, 2)
    bad_out, bad_lse = _manual_peer_cast_sum(q[0], k[0], v[0], scale, 2, 2)

    good_dlse = (res["lse"] - good_lse).abs().max().item()
    bad_dlse = (res["lse"] - bad_lse).abs().max().item()
    assert good_dlse < 1e-5, good_dlse
    assert bad_dlse > 1e-4, bad_dlse
    torch.testing.assert_close(res["out"].float()[0], good_out, atol=2e-2, rtol=2e-2)

    # 带 mask（causal，S1<S2，delta 右对齐）时逐块 mask 切片同样要正确
    masked_params = dict(params)
    masked_params.update({"mask_mode": 3})
    res_masked = golden_truth.compute(
        {"q": q, "k": k, "v": v}, masked_params, p_dtype="out_dtype", p_block=2
    )
    good_out, good_lse = _manual_peer(q[0], k[0], v[0], scale, 2, 2, mask_mode=3)
    assert (res_masked["lse"] - good_lse).abs().max().item() < 1e-5
    torch.testing.assert_close(
        res_masked["out"].float()[0], good_out, atol=2e-2, rtol=2e-2
    )

    with pytest.raises(ValueError):
        golden_truth.compute({"q": q, "k": k, "v": v}, params, p_dtype="fp16")


def test_tnd_pa_is_supported_and_matches_dense():
    """TND + PA 走同一条向量化路径（无特殊回退），结果与 dense KV 一致。"""
    from harness.ops.flash_attn import golden_truth

    torch.manual_seed(0)
    q = torch.randn(2, 2, 8, dtype=torch.float16)  # (T, N1, D)
    pages_k = torch.randn(4, 4, 1, 8, dtype=torch.float16)
    pages_v = torch.randn(4, 4, 1, 8, dtype=torch.float16)
    params = {
        "layout_q": "TND",
        "layout_kv": "PA_BBND",
        "layout_out": "TND",
        "B": 1,
        "N1": 2,
        "N2": 1,
        "S1": 1,
        "S2": 8,
        "D": 8,
        "DV": 8,
        "mask_mode": 0,
        "dtype": "fp16",
        "cu_seqlens_q": [0, 1, 2],
        "seqused_q": [1, 1],
        "seqused_kv": [8, 8],
        "block_size": 4,
        "block_table": [[0, 1], [2, 3]],
        "num_blocks": 4,
        "actual_b": 2,
        "return_softmax_lse": True,
    }
    pa = golden_truth.compute({"q": q, "k": pages_k, "v": pages_v}, params)
    assert tuple(pa["out"].shape) == (2, 2, 8)
    assert torch.isfinite(pa["out"]).all()

    dense_k = torch.stack(
        [
            torch.cat([pages_k[0], pages_k[1]]).permute(1, 0, 2),
            torch.cat([pages_k[2], pages_k[3]]).permute(1, 0, 2),
        ]
    )
    dense_v = torch.stack(
        [
            torch.cat([pages_v[0], pages_v[1]]).permute(1, 0, 2),
            torch.cat([pages_v[2], pages_v[3]]).permute(1, 0, 2),
        ]
    )
    dense_params = dict(params)
    dense_params.update({"layout_kv": "BNSD"})
    for key in ("block_size", "block_table", "num_blocks", "actual_b"):
        dense_params.pop(key)
    dense = golden_truth.compute({"q": q, "k": dense_k, "v": dense_v}, dense_params)
    torch.testing.assert_close(pa["out"], dense["out"])
    torch.testing.assert_close(pa["lse"], dense["lse"])


def test_phase_threshold_is_shape_based():
    """S1<=16 为 decode，其余 prefill；报告 phase 不受历史 case_id 前缀影响。"""
    from harness.ops.flash_attn import PACKAGE

    base = {
        "N1": 8,
        "N2": 8,
        "D": 128,
        "DV": 128,
        "S2": 1024,
        "layout_q": "BNSD",
        "layout_kv": "BNSD",
    }
    for s1, phase in ((1, "decode"), (16, "decode"), (17, "prefill"), (145, "prefill")):
        params = dict(base, S1=s1)
        assert PACKAGE.case_name(params).startswith(phase)


def test_scale_zero_is_honored():
    """显式 scale=0 必须生效（不能被 `or 默认值` 吞掉）：输出为可见项的 V 均值。"""
    from harness.ops.flash_attn import golden_truth

    params = {
        "layout_q": "BNSD",
        "layout_kv": "BNSD",
        "layout_out": "BNSD",
        "B": 1,
        "N1": 1,
        "N2": 1,
        "S1": 2,
        "S2": 4,
        "D": 4,
        "mask_mode": 0,
        "dtype": "fp16",
        "scale": 0.0,
        "return_softmax_lse": True,
    }
    torch.manual_seed(0)
    q = torch.randn(1, 1, 2, 4, dtype=torch.float16)
    k = torch.randn(1, 1, 4, 4, dtype=torch.float16)
    v = torch.randn(1, 1, 4, 4, dtype=torch.float16)
    res = golden_truth.compute({"q": q, "k": k, "v": v}, params)
    expect = v.mean(dim=2).unsqueeze(2).expand_as(res["out"]).to(torch.float16)
    torch.testing.assert_close(res["out"], expect)
    torch.testing.assert_close(
        res["lse"], torch.full((1, 1, 2), math.log(4.0), dtype=torch.float32)
    )


def test_legalize_rejects_non_positive_scale():
    """显式 scale<=0 属非法输入（算子契约），应返回原因而不是执行。"""
    from harness.ops.flash_attn.package import FlashAttnPackage

    pkg = FlashAttnPackage()
    assert pkg.legalize({"D": 128}) is None
    for bad in (0, 0.0, -0.125, -1.0):
        reason = pkg.legalize({"D": 128, "scale": bad})
        assert reason and "scale" in reason, (bad, reason)


def test_legalize_allows_tnd_without_s1_s2():
    """TND 形状由 cu_seqlens/max_seqlen 决定，S1/S2 为空不应判非法。"""
    from harness.ops.flash_attn.package import FlashAttnPackage

    pkg = FlashAttnPackage()
    params = {
        "layout_q": "TND",
        "layout_kv": "TND",
        "layout_out": "TND",
        "B": 1,
        "N1": 2,
        "N2": 2,
        "S1": None,
        "S2": None,
        "max_seqlen_q": 5852,
        "max_seqlen_kv": 6472,
        "D": 128,
        "mask_mode": 3,
        "cu_seqlens_q": [0, 5852],
        "cu_seqlens_kv": [0, 6472],
    }
    assert pkg.legalize(params) is None


def test_theoretical_workload_tnd_uses_segments():
    """TND/变长必须按每段 s1*s2 求和，不能用 B*S1*S2；字节按 cu 总 token 数。"""
    from harness.ops.flash_attn.package import FlashAttnPackage

    pkg = FlashAttnPackage()
    params = {
        "layout_q": "TND",
        "layout_kv": "TND",
        "layout_out": "TND",
        "B": 1,
        "N1": 4,
        "N2": 4,
        "S1": 8,
        "S2": 8,
        "D": 128,
        "DV": 128,
        "dtype": "bf16",
        "cu_seqlens_q": [0, 4, 12],
        "cu_seqlens_kv": [0, 8, 12],
        "seqused_q": [4, 8],
        "seqused_kv": [8, 4],
    }
    w = pkg.theoretical_workload(params)
    expect_work = 4 * 8 + 8 * 4
    assert w["flops"] == 4.0 * 4 * expect_work * 128
    expect_bytes = (12 * 4 * 128 + 2 * 12 * 4 * 128 + 12 * 4 * 128) * 2
    assert w["bytes"] == float(expect_bytes)


def test_pa_to_bnsd_allocates_actual_lengths():
    pages = torch.arange(4 * 4 * 2 * 8, dtype=torch.float32).reshape(4, 4, 2, 8)
    params = {
        "B": 1,
        "actual_b": 2,
        "N2": 2,
        "S2": 16,
        "seqused_kv": [4, 6],
        "block_size": 4,
        "block_table": [[0, 1], [2, 3]],
    }
    out = layoutlib.pa_to_bnsd(pages, "PA_BBND", params)
    assert [tuple(x.shape) for x in out] == [(2, 4, 8), (2, 6, 8)]
    expect = torch.cat([pages[0], pages[1]]).permute(1, 0, 2)[:, :4]
    torch.testing.assert_close(out[0], expect)


def test_trans_bnsd_to_bsnd_is_contiguous():
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    y = layoutlib.trans_bnsd_to_layout(x, "BSND")
    assert y.shape == (2, 4, 3, 5)
    assert y.is_contiguous()


def test_paged_kv_loader_crosses_pages():
    pages = torch.arange(4 * 4 * 2 * 8, dtype=torch.float32).reshape(4, 4, 2, 8)
    params = {
        "N2": 2,
        "block_size": 4,
        "seqused_kv": [4, 6],
        "block_table": [[0, 1], [2, 3]],
    }
    loader = layoutlib.PagedKVLoader(pages, "PA_BBND", params)
    assert loader.num_sequences() == 2
    expect = torch.cat([pages[2][2:4], pages[3][0:2]]).permute(1, 0, 2)
    torch.testing.assert_close(loader.load(1, 2, 4), expect)
    assert tuple(loader.load(1, 0, 100).shape) == (2, 6, 8)


def test_paged_kv_loader_pa_layouts():
    params = {
        "B": 1,
        "actual_b": 2,
        "N2": 2,
        "S2": 16,
        "block_size": 4,
        "seqused_kv": [4, 6],
        "block_table": [[0, 1], [2, 3]],
    }
    cases = (
        (
            "PA_BBND",
            torch.arange(4 * 4 * 2 * 8, dtype=torch.float32).reshape(4, 4, 2, 8),
        ),
        (
            "PA_BNBD",
            torch.arange(4 * 2 * 4 * 8, dtype=torch.float32).reshape(4, 2, 4, 8),
        ),
        (
            "PA_NZ",
            torch.arange(4 * 2 * 1 * 4 * 8, dtype=torch.float32).reshape(4, 2, 1, 4, 8),
        ),
    )
    for layout, pages in cases:
        loader = layoutlib.PagedKVLoader(pages, layout, params)
        dense = layoutlib.pa_to_bnsd(pages.float(), layout, params)[1]
        torch.testing.assert_close(loader.load(1, 2, 4), dense[:, 2:6])


def test_paged_golden_matches_dense_online():
    from harness.ops.flash_attn import golden_truth

    torch.manual_seed(0)
    q = torch.randn(2, 2, 8, dtype=torch.float16)
    pages_k = torch.randn(4, 4, 1, 8, dtype=torch.float16)
    pages_v = torch.randn(4, 4, 1, 8, dtype=torch.float16)
    params = {
        "layout_q": "TND",
        "layout_kv": "PA_BBND",
        "layout_out": "TND",
        "B": 1,
        "N1": 2,
        "N2": 1,
        "S1": 1,
        "S2": 8,
        "D": 8,
        "DV": 8,
        "mask_mode": 0,
        "dtype": "fp16",
        "cu_seqlens_q": [0, 1, 2],
        "seqused_q": [1, 1],
        "seqused_kv": [8, 8],
        "block_size": 4,
        "block_table": [[0, 1], [2, 3]],
        "actual_b": 2,
    }
    common = dict(
        p_dtype="out_dtype",
        p_block=4,
        softmax_mode="online",
        softmax_block_size=4,
    )
    dense = golden_truth.compute(
        {"q": q, "k": pages_k, "v": pages_v}, params, kv_loader="dense", **common
    )
    paged = golden_truth.compute(
        {"q": q, "k": pages_k, "v": pages_v}, params, kv_loader="paged", **common
    )
    torch.testing.assert_close(paged["out"], dense["out"])
    torch.testing.assert_close(paged["lse"], dense["lse"])

    common["p_dtype"] = None
    dense = golden_truth.compute(
        {"q": q, "k": pages_k, "v": pages_v}, params, kv_loader="dense", **common
    )
    paged = golden_truth.compute(
        {"q": q, "k": pages_k, "v": pages_v}, params, kv_loader="paged", **common
    )
    torch.testing.assert_close(paged["out"], dense["out"])
    torch.testing.assert_close(paged["lse"], dense["lse"])
