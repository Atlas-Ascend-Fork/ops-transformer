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

"""Pytest-backed CPU golden adapter for SparseFlashAttention."""

import copy
import importlib.util
import sys
import threading
from importlib import metadata
from pathlib import Path

import torch

try:
    from ttk.utilities.container_utils import get_global_storage
except Exception:
    get_global_storage = None


def _get_compare_method():
    if get_global_storage is not None:
        try:
            return getattr(get_global_storage(), "compare_method", None)
        except Exception:
            return None
    return None


def compare_method_is_cross_check():
    return _get_compare_method() == "cross_check"


class CaseDataStore:
    """Keep the pytest-generated result paired with its TTK testcase."""

    def __init__(self):
        self.cases = {}

    def put(self, testcase_name, data):
        if testcase_name is None:
            raise RuntimeError(
                "SparseFlashAttention requires testcase_name for pytest-backed golden"
            )
        self.cases = {str(testcase_name): data}

    def get(self, testcase_name):
        data = self.cases.get(str(testcase_name)) if testcase_name is not None else None
        if data is None:
            raise RuntimeError(
                "SparseFlashAttention golden requires customize_inputs to run "
                "before golden generation"
            )
        return data


def peek_bench(testcase_name):
    """取当前用例的三方标杆输出(cross_check 用)；未生成或未开启三方时返回 None。"""
    if testcase_name is None:
        return None
    entry = CASE_DATA.cases.get(str(testcase_name))
    if not entry:
        return None
    return entry.get("bench")


def depage_cache(cache, block_table, block_size):
    """PA cache 反查逻辑布局 (B, NB*block_size, N, D)。"""
    B, NB = block_table.shape
    out = torch.zeros(
        (B, NB * block_size, cache.shape[2], cache.shape[3]), dtype=cache.dtype
    )
    for b in range(B):
        for i in range(NB):
            blk = int(block_table[b, i])
            if blk == -1:
                continue
            out[b, i * block_size : (i + 1) * block_size] = cache[blk]
    return out


def flatten_slots(value):
    flat = []
    if isinstance(value, (list, tuple)):
        for item in value:
            flat.extend(flatten_slots(item))
    else:
        flat.append(value)
    return flat


def unpack_context_tensors(compare_context):
    """按 API 签名从 compare_context 张量槽定位输入与 sinks。

    aclnn 列表为 C 签名全长槽(含输出槽): V1 = 9 输入 + 3 输出、V2 = 10 输入 +
    3 输出。旧版 V1 无 sinks, 截断输入位并显式 sinks=None, 避免把 attentionOut
    误当 sinks。e2e 框架接口列表仅含声明的输入槽(未用可选位为 None 占位), 按
    签名序 (query, key, value, sparse_indices, block_table, actual_seq_q,
    actual_seq_kv, query_rope, key_rope, sinks) 解析, 尾部缺省补 None。
    """
    tensors = flatten_slots(compare_context.input_tensors or ())
    api_name = getattr(compare_context, "api_name", "") or ""
    if api_name == "aclnnSparseFlashAttention":
        if len(tensors) < 9:
            raise RuntimeError(
                "SFA lazy bench expects 9 input slots for %s, got %d"
                % (api_name, len(tensors))
            )
        return tensors[:9], None
    if api_name == "aclnnSparseFlashAttentionV2":
        if len(tensors) < 10:
            raise RuntimeError(
                "SFA lazy bench expects 10 input slots for %s, got %d"
                % (api_name, len(tensors))
            )
        return tensors[:9], tensors[9]
    padded = list(tensors[:10])
    padded += [None] * (10 - len(padded))
    return padded[:9], padded[9]


def compute_bench_from_context(compare_context):
    """cross_check 懒计算: 用比对侧 compare_context 携带的 API 输入张量(与 NPU
    完全对齐)重建无 exp 量化三方标杆, 并缓存进 CASE_DATA。生成阶段不再预产
    bench, 超大 case 非 cross_check 场景零额外开销。
    """
    if compare_context is None:
        return None
    attrs = dict(compare_context.attributes or {})
    inputs, sinks = unpack_context_tensors(compare_context)
    query, key, _, sparse_indices, block_table, _, _, query_rope, key_rope = inputs
    params, pytest_golden = SparseFlashAttentionPytestAdapter.convert_params(
        attrs, compare_context.testcase_name
    )
    layout_kv = str(params.get("layout_kv", "BSND")).upper()
    is_pa = layout_kv == "PA_BSND"
    if is_pa:
        block_size = int(attrs.get("pytest_block_size") or 512)
        key_logical = depage_cache(key, block_table, block_size)
        krope_logical = depage_cache(key_rope, block_table, block_size)
    else:
        key_logical, krope_logical = key, key_rope
    input_tensor_dict = {
        "query": query,
        "key": key_logical,
        "value": key_logical,
        "sparse_indices": sparse_indices,
        "block_table": block_table if is_pa else None,
        "query_rope": query_rope,
        "key_rope": krope_logical,
        "sinks": sinks,
        "scale_value": attrs.get("scale_value", params["scalevalue"]),
        "sparse_block_size": params["sparse_blocksize"],
        "layout_query": params["layout_query"],
        "layout_kv": params["layout_kv"],
        "sparse_mode": params["sparsemode"],
        "attention_mode": params["attention_mode"],
        "pre_tokens": (1 << 63) - 1,
        "next_tokens": (1 << 63) - 1,
        "return_softmax_lse": params["return_softmax_lse"],
    }
    si_api = sparse_indices
    orig_si = pytest_golden._generate_sparse_indices
    orig_bt = pytest_golden._generate_block_table_and_cache

    def reuse_si(ito, fa_param, p, out_shape, layoutQuery):
        if fa_param["tnd_flag"]:
            shape = fa_param["sparse_indices_bnsd_shape"]
            si_bnsd = torch.full(tuple(shape), -1, dtype=torch.int32)
            t0 = 0
            for b, act in enumerate(fa_param["actualSeqLengths_q"]):
                si_bnsd[b, :, :act, :] = si_api[t0 : t0 + act].permute(1, 0, 2)
                t0 += act
            fa_param["sparse_indices_bnsd_tensor"] = si_bnsd
        else:
            fa_param["sparse_indices_bnsd_tensor"] = si_api.permute(
                0, 2, 1, 3
            ).contiguous()
        ito["sparse_indices"] = si_api

    def reuse_bt(ito, fa_param, p, act_kv, out_shape):
        fa_param["block_table"] = block_table
        ito["block_table"] = block_table
        pytest_golden.kv_pa_preprocessing(ito, fa_param, p)

    pytest_golden._generate_sparse_indices = reuse_si
    if is_pa:
        pytest_golden._generate_block_table_and_cache = reuse_bt
    try:
        out = pytest_golden.compute_golden(input_tensor_dict, params, return_bench=True)
    finally:
        pytest_golden._generate_sparse_indices = orig_si
        pytest_golden._generate_block_table_and_cache = orig_bt
    if not isinstance(out, (tuple, list)) or len(out) != 4:
        raise RuntimeError("SFA lazy bench: unexpected compute_golden output")
    bench = normalize_pytest_outputs(
        (out[3], out[1], out[2]), query, params["return_softmax_lse"]
    )
    entry = CASE_DATA.cases.get(str(compare_context.testcase_name))
    if entry is not None:
        entry["bench"] = bench
    return bench


CASE_DATA = CaseDataStore()


class SparseFlashAttentionPytestAdapter:
    """Load pytest and restore its source parameter combination from CSV."""

    modules = None
    modules_lock = threading.Lock()

    LOCAL_MODULE_NAMES = (
        "check_valid_param",
        "generate_tensor_data",
        "result_compare_method",
        "sparse_flash_attention_golden",
    )

    PARAMETER_NAMES = (
        "Testcase_Prefix",
        "layout_query",
        "layout_kv",
        "q_type",
        "kv_type",
        "B",
        "T1",
        "T2",
        "S1",
        "S2",
        "N1",
        "N2",
        "D",
        "K",
        "scale_value",
        "sparse_block_size",
        "rope_head_dim",
        "sparse_mode",
        "attention_mode",
        "return_softmax_lse",
        "use_sinks",
        "block_size",
        "block_num",
        "actual_seq_q",
        "actual_seq_kv",
        "range_query",
        "range_key",
        "range_query_rope",
        "range_key_rope",
        "range_sinks",
    )

    REQUIRED_PARAMETER_NAMES = (
        "Testcase_Prefix",
        "layout_query",
        "layout_kv",
        "q_type",
        "kv_type",
        "B",
        "S1",
        "S2",
        "N1",
        "N2",
        "D",
        "K",
        "scale_value",
        "sparse_block_size",
        "rope_head_dim",
        "sparse_mode",
        "attention_mode",
        "actual_seq_q",
        "actual_seq_kv",
    )

    @staticmethod
    def package_version(*distribution_names):
        for name in distribution_names:
            try:
                return metadata.version(name)
            except metadata.PackageNotFoundError:
                continue
        return "not-installed"

    @classmethod
    def load_error(cls, module_path, dependency_path, exc):
        versions = (
            f"python={sys.version.split()[0]}, "
            f"tensorflow={cls.package_version('tensorflow', 'tensorflow-cpu')}, "
            f"numpy={cls.package_version('numpy')}"
        )
        return RuntimeError(
            "Failed to load SparseFlashAttention pytest parameter conversion; "
            f"module={module_path}; dependency={dependency_path}; {versions}; "
            f"original error: {type(exc).__name__}: {exc}"
        )

    @classmethod
    def load_modules(cls):
        if cls.modules is not None:
            return cls.modules
        with cls.modules_lock:
            if cls.modules is not None:
                return cls.modules

            pytest_dir = Path(__file__).resolve().parents[2] / "pytest"
            module_path = pytest_dir / "utils.py"
            dependency_path = pytest_dir / "generate_tensor_data.py"
            module_name = "sfa_ttk_pytest_utils"
            saved_modules = {
                name: sys.modules.pop(name, None) for name in cls.LOCAL_MODULE_NAMES
            }
            sys.path.insert(0, str(pytest_dir))
            try:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"cannot create import spec for {module_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                pytest_golden = module.sparse_flash_attention_golden
            except Exception as exc:
                sys.modules.pop(module_name, None)
                raise cls.load_error(module_path, dependency_path, exc) from exc
            finally:
                sys.path.remove(str(pytest_dir))
                for name in cls.LOCAL_MODULE_NAMES:
                    sys.modules.pop(name, None)
                for name, saved in saved_modules.items():
                    if saved is not None:
                        sys.modules[name] = saved
            cls.modules = (module, pytest_golden)
            return cls.modules

    @staticmethod
    def torch_dtype(value, field):
        if isinstance(value, torch.dtype):
            return value
        normalized = str(value).strip().lower().removeprefix("torch.")
        mapping = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        if normalized not in mapping:
            raise ValueError(
                f"Unsupported SparseFlashAttention pytest {field}: {value!r}"
            )
        return mapping[normalized]

    @classmethod
    def convert_params(cls, attributes, testcase_name):
        if not isinstance(attributes, dict):
            raise ValueError(
                "SparseFlashAttention input attributes must be a dictionary"
            )
        missing = [
            f"pytest_{name}"
            for name in cls.REQUIRED_PARAMETER_NAMES
            if f"pytest_{name}" not in attributes
        ]
        if missing:
            raise ValueError(
                f"SparseFlashAttention CSV is missing explicit pytest fields: {missing}"
            )
        combination = {
            name: copy.deepcopy(attributes[f"pytest_{name}"])
            for name in cls.PARAMETER_NAMES
            if f"pytest_{name}" in attributes
        }
        combination["q_type"] = cls.torch_dtype(combination["q_type"], "q_type")
        combination["kv_type"] = cls.torch_dtype(combination["kv_type"], "kv_type")
        # The pytest random generator accepts list ranges, not tuple ranges.
        for name, value in tuple(combination.items()):
            if name.startswith("range_") and isinstance(value, tuple):
                combination[name] = list(value)
        pytest_utils, pytest_golden = cls.load_modules()
        params = pytest_utils.convert_param_combination_to_cs_format(combination)
        params["case_name"] = testcase_name or params["case_name"]
        return params, pytest_golden


def normalize_pytest_outputs(outputs, query, return_softmax_lse):
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise RuntimeError(
            "SparseFlashAttention pytest compute_cpu returned an invalid output tuple"
        )
    # 对齐算子Cast(CAST_ROUND)语义:仅"有限值"超出目标dtype表示范围时钳到上下界，±inf/NaN 原样保留
    attn_fp32 = outputs[0].detach().cpu().float()
    attn_out = torch.where(
        torch.isfinite(attn_fp32),
        attn_fp32.clamp(-torch.finfo(query.dtype).max, torch.finfo(query.dtype).max),
        attn_fp32,
    ).to(dtype=query.dtype)
    if not bool(return_softmax_lse):
        # None 占位落盘为 golden_i_none marker，TTK 原生按抑制比较处理，
        # 使 e2e prepare 的 bin 目录可被 aclnn replay 直接复用。
        return attn_out, None, None
    softmax_max = outputs[1].detach().cpu()
    softmax_sum = outputs[2].detach().cpu()
    return attn_out, softmax_max, softmax_sum


def cpu_sparse_flash_attention(
    query, key, value, sparse_indices, scale_value, *, testcase_name=None, **kwargs
):
    """Return the result generated by pytest for this exact TTK testcase."""
    del query, key, value, sparse_indices, scale_value, kwargs
    return CASE_DATA.get(testcase_name)["golden"]


def cpu_sparse_flash_attention_aclnn(*args, testcase_name=None, **kwargs):
    """ACLNN 直调版 golden：按 aclnn C 签名顺序接收全部位置参数。"""
    del args, kwargs
    return CASE_DATA.get(testcase_name)["golden"]


__golden__ = {
    "e2e": {
        "torch_npu.npu_sparse_flash_attention": "cpu_sparse_flash_attention",
        "torch.ops.cann_ops_transformer.sparse_flash_attention": (
            "cpu_sparse_flash_attention"
        ),
    },
    "aclnn": {
        "aclnnSparseFlashAttention": "cpu_sparse_flash_attention_aclnn",
        "aclnnSparseFlashAttentionV2": "cpu_sparse_flash_attention_aclnn",
    },
}
