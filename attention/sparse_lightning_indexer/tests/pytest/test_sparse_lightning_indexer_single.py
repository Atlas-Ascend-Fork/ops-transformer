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

"""SparseLightningIndexer 单用例测试主程序（candidate consumer，仅 arch22 / A2）。

消费 test_sparse_lightning_indexer_paramset 的参数集（ENABLED_PARAMSETS），逐条：
  1. 组装 27 字段 LIV2 基础参数 + consumer 扩展参数（cand_spec/cand_blocks/cand_block_size）；
  2. CPU consumer golden（无 leak 两档掩码 → topk，候选内有效数不足 topk 时 -1 补齐；
     return_value 双档：rv=0 恒空 values / rv=1 真实分 values）；
  3. NPU sparse_lightning_indexer 直调（eager）；
  4. 精度对比（check_result：多重集合门 → compare_topk_valid 官方规则 → -1 槽硬校验；
     rv=1 时另做 values 独立 isclose 断言）。

用例筛选：
    SLI_CASE_NAMES=SLI_FP16_01,SLI_PA_04 python3 -m pytest -s test_sparse_lightning_indexer_single.py -m ci
"""

import importlib
import itertools
import os

import numpy as np
import pytest

# ---- guarded imports: keep the module collectable on hosts without NPU / built op package
_GOLDEN_IMPORT_ERROR = None
try:
    import torch
    import torch_npu  # noqa: F401  (register npu dispatch before the golden imports)
    import sparse_lightning_indexer_golden as sli_golden
except Exception as _exc:  # noqa: BLE001
    torch = None
    sli_golden = None
    _GOLDEN_IMPORT_ERROR = _exc


try:
    paramset_module = importlib.import_module("test_sparse_lightning_indexer_paramset")
    ENABLED_PARAMSETS = getattr(paramset_module, "ENABLED_PARAMSETS", [])
except Exception as _exc:  # noqa: BLE001 - collectable on hosts without torch/NPU
    ENABLED_PARAMSETS = []

requested_names = {
    name.strip()
    for name in os.environ.get("SLI_CASE_NAMES", "").split(",")
    if name.strip()
}
matched_names = set()
test_created = False

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
CANDIDATE_FIELDS = ["cand_spec", "cand_blocks", "cand_block_size"]
ALL_FIELDS = BASE_FIELDS + CANDIDATE_FIELDS


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


for paramset_index, (paramset_name, params) in enumerate(ENABLED_PARAMSETS):
    # 将 params 的所有字段注册为局部变量
    for key, value in params.items():
        locals()[f"param_{key}"] = value
    param_values = [
        locals()["param_batch_size"],
        locals()["param_q_seq"],
        locals()["param_k_seq"],
        locals()["param_q_t_size"],
        locals()["param_k_t_size"],
        locals()["param_q_head_num"],
        locals()["param_k_head_num"],
        locals()["param_head_dim"],
        locals()["param_block_size"],
        locals()["param_block_num"],
        locals()["param_qk_dtype"],
        locals()["param_cu_seqlens_q"],
        locals()["param_cu_seqlens_k"],
        locals()["param_seqused_q"],
        locals()["param_seqused_k"],
        locals()["param_cmp_residual_k"],
        locals()["param_output_idx_offset"],
        locals()["param_layout_q"],
        locals()["param_layout_k"],
        locals()["param_topk"],
        locals()["param_mask_mode"],
        locals()["param_query_datarange"],
        locals()["param_key_datarange"],
        locals()["param_weights_datarange"],
        locals()["param_cmp_ratio"],
        locals()["param_return_value"],
        locals()["param_max_seqlen_q"],
        locals()["param_cand_spec"],
        locals()["param_cand_blocks"],
        locals()["param_cand_block_size"],
    ]

    # 生成所有组合，并转换为字典列表
    combinations = list(itertools.product(*param_values))
    locals()["param_combinations"] = []
    for combo_index, combo in enumerate(combinations, start=1):
        case_name = (
            paramset_name
            if len(combinations) == 1
            else f"{paramset_name}_{combo_index:03d}"
        )
        if requested_names and not (
            paramset_name in requested_names or case_name in requested_names
        ):
            continue
        matched_names.update((paramset_name, case_name))
        param_dict = dict(zip(ALL_FIELDS, combo))
        locals()["param_combinations"].append(pytest.param(param_dict, id=case_name))
    if not locals()["param_combinations"]:
        continue

    @pytest.mark.ci
    @pytest.mark.parametrize("param_combinations", locals()["param_combinations"])
    def test_sparse_lightning_indexer(param_combinations):
        if sli_golden is None:
            pytest.skip(
                f"sparse_lightning_indexer_golden import failed: {_GOLDEN_IMPORT_ERROR!r}"
            )
        arch = _device_arch()
        if arch == "arch35":
            pytest.skip(
                "sparse_lightning_indexer is arch22-only (Ascend910B / Ascend910_93)"
            )

        test_data = tuple(param_combinations[f] for f in BASE_FIELDS)
        cand_spec = param_combinations["cand_spec"]
        cand_blocks = int(param_combinations["cand_blocks"])
        cand_block_size = int(param_combinations["cand_block_size"])

        torch_npu.npu.set_device(0)
        print(
            f"[case] B={param_combinations['batch_size']} "
            f"S1={param_combinations['q_seq']} S2={param_combinations['k_seq']} "
            f"N1={param_combinations['q_head_num']} topk={param_combinations['topk']} "
            f"mask={param_combinations['mask_mode']} cmp={param_combinations['cmp_ratio']} "
            f"layout_q={param_combinations['layout_q']} "
            f"layout_k={param_combinations['layout_k']} "
            f"dtype={param_combinations['qk_dtype']} cand_spec={cand_spec} "
            f"candBlocks={cand_blocks} bs={cand_block_size} "
            f"rv={param_combinations['return_value']} arch={arch}"
        )

        try:
            out = sli_golden.sparse_output_single(
                test_data, cand_spec, cand_blocks, cand_block_size
            )
        except sli_golden.SparseOpUnavailableError as exc:
            pytest.skip(f"sparse torch op not available: {exc}")

        return_value = int(param_combinations["return_value"])

        # 精度对比（compare_topk_valid 官方规则 + -1 槽硬校验；
        # 边界仲裁使用无 leak 掩码 score'（golden 口径）；rv=1 时比较器启用 values 仲裁）
        rc = sli_golden.rc
        result, fulfill_percent = rc.check_result(
            out["cpu_result"],
            out["npu_result"],
            out["cpu_degraded"],
            param_combinations["output_idx_offset"],
            test_data,
            out["cpu_values"] if return_value == 1 else None,
            out["npu_values"],
        )
        print(f"result={result} fulfill={fulfill_percent}")
        assert result == "Pass"

        # ---- values 独立断言块（主口径，spec numerical_tolerance.float32；
        #      check_result 的 values 内嵌仲裁仅在索引集合不一致行触发，不能替代主口径） ----
        npu_values = out["npu_values"].cpu()
        if return_value == 0:
            # rv=0：sparse_values 恒空 (0,)，kernel 不写入
            assert npu_values.numel() == 0, (
                f"rv=0 sparse_values must be empty (0,), got shape "
                f"{tuple(npu_values.shape)}"
            )
        else:
            # rv=1：shape 与 indices 一致、dtype 恒 FLOAT32
            assert tuple(npu_values.shape) == tuple(out["npu_result"].shape), (
                f"rv=1 sparse_values shape {tuple(npu_values.shape)} != indices shape "
                f"{tuple(out['npu_result'].shape)}"
            )
            assert npu_values.dtype == torch.float32, (
                f"rv=1 sparse_values dtype must be float32, got {npu_values.dtype}"
            )
            # 无哨兵污染（REQ §6.4 / spec no_leak_intermediates）：
            # 输出中不允许 |value| >= 1e29 量级值（-1e30 哨兵特征消失的可测性推论）
            assert not torch.any(npu_values.abs() >= 1e29), (
                "rv=1 sparse_values contains sentinel-magnitude (|value| >= 1e29) values"
            )
            # idx=-1 对应槽 value==0（spec：无效槽填 0）
            invalid_mask = out["npu_result"].cpu() == -1
            if invalid_mask.any():
                assert torch.all(npu_values[invalid_mask] == 0), (
                    "rv=1 sparse_values must be 0 at idx=-1 slots"
                )
            # isclose 逐槽主口径（equal_nan；有效槽比真实分、无效槽比 0）。
            # 容差对齐 spec numerical_tolerance 声明的判据源 result_compare_method
            # （judge_value_by_isclose：atol=2.5e-5、rtol=5e-3，LIV2 同族口径）：
            # kernel 打分域 fp32 累加序与 numpy golden 存在固有非结合性尾差
            # （实测 max_abs ~6.5e-5、集中于小 |v| 槽，98304 槽 0.005%~0.012%），
            # rtol=atol=1e-5 在大 shape（regime2，topk=192×S1=512）上按数据
            # 批次抖动（A1-P 穿刺实测 RV1_REGIME2_37 偶发违例），故放齐到同族判据。
            cpu_values_t = torch.from_numpy(np.ascontiguousarray(out["cpu_values"]))
            assert torch.isclose(
                cpu_values_t, npu_values, rtol=5e-3, atol=2.5e-5, equal_nan=True
            ).all(), (
                "rv=1 sparse_values mismatch vs golden (isclose atol=2.5e-5, rtol=5e-3)"
            )

    globals()[f"test_sparse_lightning_indexer_{paramset_index:03d}"] = (
        test_sparse_lightning_indexer
    )
    test_created = True

unknown_names = sorted(requested_names - matched_names)
if unknown_names:
    raise ValueError(
        f"unknown case(s) in test_sparse_lightning_indexer_paramset: {unknown_names}"
    )

if test_created:
    del test_sparse_lightning_indexer
