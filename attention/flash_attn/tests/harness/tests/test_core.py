# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""框架自身单测（纯 CPU，不上板）：命名/归一化/编解码/缓存/判定。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

HARNESS_ROOT = Path(__file__).resolve().parents[2]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from harness import compare as cmp  # noqa: E402
from harness import loader  # noqa: E402
from harness.tests import csv_codec  # noqa: E402
from harness.case import NORM_VERSION, Case, canonical_params, make_case_id  # noqa: E402
from harness.golden_spec import parse_spec  # noqa: E402
from harness.perf import Profiler  # noqa: E402
from harness.ops.flash_attn import PACKAGE  # noqa: E402

BASE = {
    "dtype": "fp16",
    "layout_q": "BNSD",
    "layout_kv": "BNSD",
    "layout_out": "BNSD",
    "B": 1,
    "N1": 4,
    "N2": 2,
    "S1": 256,
    "S2": 512,
    "D": 128,
    "mask_mode": 3,
}


def _case(**overrides):
    params = dict(BASE)
    params.update(overrides)
    return Case(op="flash_attn", params=params, tags=["ci"])


def test_case_id_is_order_independent_and_versioned():
    left = make_case_id("flash_attn", {"a": 1, "b": [2, 3], "c": 0.1})
    right = make_case_id("flash_attn", {"c": 0.1, "b": [2, 3], "a": 1})
    assert left == right
    assert NORM_VERSION in canonical_params({"x": 1} | {"_norm": NORM_VERSION})
    assert make_case_id("flash_attn", {"a": 2}) != left


def test_json_csv_roundtrip(tmp_path):
    cases = [
        _case(),
        _case(dtype="bf16", D=64, mask_mode=0, layout_q="BSND", layout_kv="BSND"),
    ]
    path = csv_codec.to_csv(cases, tmp_path / "cases.csv")
    back = csv_codec.from_csv(path)
    assert [c.id for c in back] == [c.id for c in cases]
    for left, right in zip(cases, back):
        assert canonical_params(left.params) == canonical_params(right.params)


def _write_case_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_cases_validates_structure(tmp_path):
    path = _write_case_json(
        tmp_path / "cases.json",
        {
            "schema": "case/v1",
            "count": 1,
            "cases": [{"op": "flash_attn", "params": BASE}],
        },
    )

    cases = loader.load_from_json([str(path)])

    assert len(cases) == 1
    assert cases[0].op == "flash_attn"


def test_load_cases_rejects_count_mismatch(tmp_path):
    path = _write_case_json(
        tmp_path / "bad.json",
        {
            "schema": "case/v1",
            "count": 2,
            "cases": [{"op": "flash_attn", "params": BASE}],
        },
    )

    with pytest.raises(ValueError, match="count=2"):
        loader.load_from_json([str(path)])


def test_load_cases_accepts_failure_replay_payload(tmp_path):
    path = _write_case_json(
        tmp_path / "failures.json",
        {
            "count": 1,
            "cases": [
                {
                    "op": "flash_attn",
                    "params": BASE,
                    "tags": ["fuzz"],
                    "status": "fail",
                    "detail": "compare mismatch",
                }
            ],
        },
    )

    cases = loader.load_from_json([str(path)])

    assert len(cases) == 1


def test_load_cases_rejects_bad_case_row(tmp_path):
    path = _write_case_json(
        tmp_path / "bad.json",
        {"schema": "case/v1", "cases": [{"op": "flash_attn"}]},
    )

    with pytest.raises(ValueError, match="params must be an object"):
        loader.load_from_json([str(path)])


def test_loader_selection_and_skip():
    a = _case(S1=256, S2=512)
    b = _case(S1=1, S2=1024)
    b.tags = ["debug"]
    c = _case(S1=256, S2=257)
    c.tags = ["func_rdv"]
    picked = loader.select([a, b, c], case_ids=["kv512"])
    assert picked == [a]
    picked = loader.select([a, b, c], case_ids=["*kv25*"])
    assert picked == [c]
    picked = loader.select([a, b, c], testsets=["func_rdv"])
    assert picked == [c]
    picked = loader.select([a, b, c], skip=["kv512"])
    assert picked == [b, c]


def test_case_signature_changes_with_params():
    assert _case().signature_digest() != _case(S2=513).signature_digest()
    assert _case().signature_digest() == _case().signature_digest()


def test_parse_spec_validation():
    spec = parse_spec(
        "device=npu,compute_dtype=fp64,p_block=128,"
        "softmax_mode=online,softmax_block_size=256,kv_loader=paged"
    )
    assert spec.p_block == 128
    assert spec.p_dtype == "out_dtype"  # 默认与 kernel 同口径
    assert spec.softmax_mode == "online"
    assert spec.softmax_block_size == 256
    assert spec.kv_loader == "paged"
    with pytest.raises(ValueError):
        parse_spec("unknown=1")
    with pytest.raises(ValueError):
        parse_spec("p_dtype=fp16")
    with pytest.raises(ValueError):
        parse_spec("softmax_mode=invalid")
    with pytest.raises(ValueError):
        parse_spec("kv_loader=invalid")
    with pytest.raises(ValueError):
        parse_spec("softmax_block_size=-1")


def test_compare_two_way_pass_and_fail():
    expect = torch.zeros(4, 8, 16, dtype=torch.float16)
    near = expect.clone()
    stats = cmp.compare_two_way(expect, near, "fp16")
    assert stats["passed"] and stats["fail_cnt"] == 0

    bad = expect.clone()
    bad[0, 0, 0] = 1.0
    stats = cmp.compare_two_way(expect, bad, "fp16")
    assert stats["fail_cnt"] == 1 and stats["samples"][0][0] == 0
    assert not stats["passed"] if stats["fail_ratio"] > cmp.FAIL_RATIO_LIMIT else True

    stats = cmp.compare_two_way(expect, torch.zeros(2, 2), "fp16")
    assert not stats["passed"] and "shape mismatch" in stats["error"]


def test_compare_three_way_ratio_semantics():
    torch.manual_seed(0)
    truth = torch.randn(2, 4, 8, 16)
    peer = truth + torch.randn_like(truth) * 1e-3
    same = truth + torch.randn_like(truth) * 1e-3
    assert cmp.compare_three_way(truth, peer, same)["passed"]

    worse = truth + torch.randn_like(truth) * 1e-1
    stats = cmp.compare_three_way(truth, peer, worse)
    assert not stats["passed"] and stats["mare_ratio"] > 2.0


def test_profiler_reads_task_time_duration_and_tiling(tmp_path):
    """模拟 msprof 追加 task_time 行：一次读取 duration + tiling key。"""
    perf_dir = tmp_path / "prof"
    perf_dir.mkdir()
    task_time = perf_dir / "task_time_1.csv"
    header = (
        "Device_id,kernel_name,kernel_type,stream_id,task_id,"
        "Task Start Time(us),Task Duration(us)\n"
    )
    task_time.write_text(
        header
        + "0,FlashAttn_aaa_111111,AI_CORE,1,1,100.0,10.0\n"
        + "0,FlashAttnMetadata,AI_CPU,1,2,100.0,5.0\n",
        encoding="utf-8",
    )
    prof = Profiler(str(perf_dir), extractor=PACKAGE.tiling_key)
    prof.start()
    assert prof.collect() is None

    with open(task_time, "a", encoding="utf-8") as handle:
        handle.write("0,FlashAttn_bbb_222222,AI_CORE,1,3,200.0,20.0\n")
    sample = prof.collect()
    assert sample["tiling_key"] == "222222"
    assert sample["duration_us"] == 20.0
    assert prof.collect() is None


def test_mem_guard_resolution():
    from harness import guards

    assert guards.resolve_mem_guard(None) == 32.0
    assert guards.resolve_mem_guard(0) == 0.0  # 0 = 关闭
    assert guards.resolve_mem_guard(3.5) == 3.5  # 显式覆盖


def test_unsupported_operator_error_detection():
    from harness.runner import is_unsupported_error

    assert is_unsupported_error(
        RuntimeError(
            "call aclnnFlashAttn failed ... Reason: Failed to execute tiling function."
        )
    )
    assert is_unsupported_error(RuntimeError("shape is not supported"))
    assert not is_unsupported_error(RuntimeError("device out of memory"))


def test_results_csv_roundtrip(tmp_path):
    import csv as csvlib

    from harness import results as reslib

    records = [
        {
            "run_id": "r1",
            "case_id": "flash_attn/case__0000",
            "category": "func",
            "status": "pass",
            "phase": "prefill",
            "seqused_q": [1, 2],
            "scale": None,
        }
    ]
    csv_path = reslib.write_csv(records, tmp_path / "results.csv")
    rows = list(csvlib.DictReader(csv_path.open()))
    assert rows[0]["seqused_q"] == "[1,2]"
    assert rows[0]["phase"] == "prefill"


def test_fatal_error_detection():
    """设备/队列消失类错误必须识别为 fatal（整轮终止），缺 tilingKey 本身不算。"""
    from harness.runner import is_fatal_error

    assert is_fatal_error(RuntimeError("Inner error happened with CAN_EXIT status"))
    assert is_fatal_error(RuntimeError("The process exits for this inner error"))
    assert is_fatal_error(RuntimeError("ERR00100 PTA call acl api failed"))
    assert not is_fatal_error(RuntimeError("Cannot find tilingKey[131329]"))
    assert not is_fatal_error(RuntimeError("device out of memory"))
