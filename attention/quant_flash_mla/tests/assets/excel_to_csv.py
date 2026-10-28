#!/usr/bin/python3
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
"""Convert an MLA workbook to TTK CSV while retaining explicit fields.

Explicit layouts, shapes, dtypes, zeros, negative bounds and sequence values
are never normalized into valid inputs. Missing core shapes are inferred only
where unambiguous; missing optional sequence tensors stay omitted. Missing
metadata uses a dynamic (-1,-1) slot, materialized on the execution machine.
This converter no longer forces legacy red.xlsx rows to TND.
"""

import argparse
import csv
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _xlsx_minireader import load_sheet, list_sheet_names

_XLSX = os.path.join(_HERE, "red.xlsx")

# 11 列 header
HEADER = [
    "testcase_name",
    "api_name",
    "tensor_view_shapes",
    "tensor_dtypes",
    "tensor_formats",
    "attributes",
    "output_tensor_indexes",
    "golden_api",
    "input_data_ranges",
    "precision_tolerances",
    "absolute_precision",
]

API_NAME = "torch.ops.cann_ops_transformer.quant_flash_mla"

CSV_PROFILES = [
    ("qfmla_excel.csv", API_NAME, ""),
]

# excel dtype → csv dtype
DTYPE_MAP = {
    "FLOAT8_E4M3FN": "float8_e4m3fn",
    "FP8_E4M3": "float8_e4m3fn",
    "FP8_E4M3FN": "float8_e4m3fn",
    "HIFP8": "float8_e4m3fn",  # HIFP8 → FP8_E4M3FN (转换)
    "HIFLOAT8": "float8_e4m3fn",
    "HFLOAT8": "float8_e4m3fn",
    "FLOAT8_E5M2": "float8_e5m2",
    "FP8_E5M2": "float8_e5m2",
    "BF16": "bfloat16",
    "FP16": "float16",
    "FP32": "float32",
    "FLOAT32": "float32",
    "INT32": "int32",
    "INT8": "int8",
    "BOOL": "bool",
}

ABSOLUTE_PRECISION_DEFAULT = 1e-8

# tensor 顺序（10 个），对齐 MLA 算子 schema 位置参数顺序。
# 每项：(shape 列名, dtype 列名, datarange 列名, dtype 缺省值, 是否读 Excel datarange)。
_TENSOR_SPECS = [
    ("q_shape", "q_dtype", "q_datarange", "float8_e4m3fn", True),
    ("k_cache_shape", "k_cache_dtype", "k_cache_datarange", "float8_e4m3fn", True),
    ("q_descale_shape", "q_descale_dtype", "q_descale_datarange", "float32", False),
    ("k_descale_shape", "k_descale_dtype", "k_descale_datarange", "float32", False),
    ("block_table_shape", "block_table_dtype", "block_table_datarange", "int32", False),
    (
        "cache_seqlens_shape",
        "cache_seqlens_dtype",
        "cache_seqlens_datarange",
        "int32",
        False,
    ),
    (
        "cu_seqlens_q_shape",
        "cu_seqlens_q_dtype",
        "cu_seqlens_q_datarange",
        "int32",
        False,
    ),
    ("seqused_q_shape", "seqused_q_dtype", "seqused_q_datarange", "int32", False),
    ("attn_mask_shape", "attn_mask_dtype", "attn_mask_datarange", "int8", False),
    ("metadata_shape", "metadata_dtype", "metadata_datarange", "int32", False),
]

_DRANGE_PLACEHOLDER = (0, 1)

# MLA MLA: V = K 的 nope 前 512 维, K 有 576 维 (nope 512 + rope 64)
# MLA 不使用 V descale (V 与 K 共享 per-tensor scale), MLA 不使用 p_scale descale (固定 1.0)

METADATA_STRIDE = 16

# PA layout_kv → golden kv_cache_layout 名
_KV_LAYOUT_MAP = {
    "PA_BNBD": "BnNBsD",
    "PA_BBND": "BnBsND",
    "PA_NZ": "NZ",
}


# -----------------------------------------------------------------------------------------------------------
# metadata slot shape 推导（镜像 quant_flash_mla_metadata 的输出推导）
# -----------------------------------------------------------------------------------------------------------
_CORE_NUMS = None


# -----------------------------------------------------------------------------------------------------------
# 按表头名查列索引
# -----------------------------------------------------------------------------------------------------------
def _norm_name(name: str) -> str:
    return name.strip().lower().replace("_", "")


_HEADER_ALIASES = {
    # Excel 有 .1 后缀的列存实际值，无后缀列为空
    "cache_seqlens_value": "cache_seqlens_value.1",
    "cu_seqlens_q_value": "cu_seqlens_q_value.1",
}
_HEADER_ALIASES_NORM = {
    _norm_name(k): _norm_name(v) for k, v in _HEADER_ALIASES.items()
}


class _Columns:
    def __init__(self, header_row: dict):
        self._by_name = {}
        for col_idx, text in header_row.items():
            if not text:
                continue
            key = _norm_name(text)
            if key not in self._by_name:
                self._by_name[key] = col_idx

    def get(self, name):
        if name is None:
            return None
        key = _norm_name(name)
        # Try alias first, then fall back to original name
        alias_key = _HEADER_ALIASES_NORM.get(key)
        if alias_key is not None and alias_key in self._by_name:
            return self._by_name[alias_key]
        return self._by_name.get(key)


# -----------------------------------------------------------------------------------------------------------
# 值转换器
# -----------------------------------------------------------------------------------------------------------
def _strip_or_none(s):
    if s is None:
        return None
    s = s.strip()
    return s or None


def _is_int_literal(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        try:
            int(float(s))
            return True
        except ValueError:
            return False


def _str_to_shape(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = s.strip("[]()")
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if not parts:
        return None
    dims = []
    for p in parts:
        if not _is_int_literal(p):
            return None
        dims.append(int(float(p)))
    return tuple(dims)


def _str_to_datarange(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    s = s.strip("[]()")
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != 2:
        raise ValueError(f"unexpected datarange: {s!r}")
    return (float(parts[0]), float(parts[1]))


def _str_to_int(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return int(float(s))


def _str_to_float(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return float(s)


def _str_to_int_list(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    return [int(float(p)) for p in parts]


def _map_dtype(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s not in DTYPE_MAP:
        raise ValueError(f"unknown excel dtype: {s!r}")
    return DTYPE_MAP[s]


def _bool_to_int(s):
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    if s in ("True", "TRUE", "1", "true"):
        return 1
    if s in ("False", "FALSE", "0", "false"):
        return 0
    raise ValueError(f"unexpected bool: {s!r}")


def _map_dtype_or(s, default):
    v = _map_dtype(s)
    return v if v is not None else default


# -----------------------------------------------------------------------------------------------------------
# Explicit sequence fields (no layout migration or sequence repair)
# -----------------------------------------------------------------------------------------------------------
def _ensure_tnd_fields(row, cols):
    """Read explicit sequence values and layouts without repairing them."""
    return (
        _str_to_int_list(row.get(cols.get("seqused_q_value"))),
        _str_to_int_list(row.get(cols.get("cu_seqlens_q_value"))),
        _strip_or_none(row.get(cols.get("layout_q"))),
        _strip_or_none(row.get(cols.get("layout_out"))),
    )


def _calc_k_cache_shape(cache_seqlens, block_size, n_kv, d_dim, layout_kv):
    """计算 PA K cache shape。"""
    import math

    blocks_per_batch = [math.ceil(s / block_size) for s in cache_seqlens]
    total_blocks = sum(blocks_per_batch)
    kv_layout = _KV_LAYOUT_MAP.get(layout_kv, layout_kv)
    if kv_layout == "BnNBsD":
        return (total_blocks, n_kv, block_size, d_dim)
    elif kv_layout == "BnBsND":
        return (total_blocks, block_size, n_kv, d_dim)
    elif kv_layout == "NZ":
        d0 = 32  # fp8 element_size = 1, so 32//1 = 32
        d1 = d_dim // d0
        return (total_blocks, n_kv, d1, block_size, d0)
    else:
        raise ValueError(f"Unknown kv_layout: {kv_layout}")


# -----------------------------------------------------------------------------------------------------------
# 每行映射：excel row → csv record
# -----------------------------------------------------------------------------------------------------------
def _build_tensor_lists(row, cols, sq_values, cu_values):
    """Preserve every explicit shape/dtype; infer only missing core storage."""
    import math

    def value(name, default):
        parsed = _str_to_int(row.get(cols.get(name)))
        return default if parsed is None else parsed

    B, S = value("B", 1), value("Q_S", 1)
    Nq, Nk, D, Dv = (
        value("Q_N", 1),
        value("KV_N", 1),
        value("Q_D", 576),
        value("V_D", 512),
    )
    block_size = value("blocksize", 128)
    layout = _strip_or_none(row.get(cols.get("layout_q"))) or "TND"
    kv_layout = _strip_or_none(row.get(cols.get("layout_kv"))) or "PA_BNBD"
    if layout == "TND":
        T = (
            cu_values[-1]
            if cu_values
            else (sum(sq_values) if sq_values is not None else B * S)
        )
        q_shape = (T, Nq, D)
    elif layout == "BSND":
        q_shape = (B, S, Nq, D)
    elif layout == "BNSD":
        q_shape = (B, Nq, S, D)
    else:
        q_shape = (
            None  # An unknown layout needs an explicit shape, not a replacement layout.
        )
    cache_values = _str_to_int_list(row.get(cols.get("cache_seqlens_value")))
    inferred = {
        "q_shape": q_shape,
        "q_descale_shape": None if q_shape is None else q_shape[:-1],
        "k_descale_shape": (1,),
        "cache_seqlens_shape": (B,),
        "seqused_q_shape": None if sq_values is None else (len(sq_values),),
        "cu_seqlens_q_shape": None if cu_values is None else (len(cu_values),),
        "attn_mask_shape": (2048, 2048) if value("mask_mode", 0) == 3 else None,
        # Dynamic output is resolved on the execution machine; no guessed core counts.
        "metadata_shape": (-1, -1),
    }
    # Capacity cannot be inferred from malformed lengths; explicit shapes remain usable.
    if (
        cache_values is not None
        and block_size > 0
        and kv_layout in _KV_LAYOUT_MAP
        and all(length >= 0 for length in cache_values)
    ):
        inferred["k_cache_shape"] = _calc_k_cache_shape(
            cache_values, block_size, Nk, D, kv_layout
        )
        inferred["block_table_shape"] = (
            B,
            max((math.ceil(x / block_size) for x in cache_values), default=0),
        )
    optional = {"cu_seqlens_q_shape", "seqused_q_shape", "attn_mask_shape"}
    shapes, dtypes, ranges = [], [], []
    for shape_col, dtype_col, range_col, default_dtype, use_range in _TENSOR_SPECS:
        shape = _str_to_shape(row.get(cols.get(shape_col)))
        if shape is None:
            shape = inferred.get(shape_col)
        if shape is None and shape_col not in optional:
            raise ValueError(
                f"missing {shape_col}; cannot infer storage without changing the case"
            )
        shapes.append(shape)
        dtypes.append(_map_dtype_or(row.get(cols.get(dtype_col)), default_dtype))
        drange = _str_to_datarange(row.get(cols.get(range_col))) if use_range else None
        ranges.append(_DRANGE_PLACEHOLDER if drange is None else drange)
    return shapes, dtypes, ranges


def _build_attributes(row, cols, sq_values, cu_values):
    """Emit explicit attributes, retaining zeros/negatives and omitted optionals."""
    attrs = {}
    integer_columns = {
        "B": "B",
        "N_q": "Q_N",
        "N_kv": "KV_N",
        "D": "Q_D",
        "D_V": "V_D",
        "block_size": "blocksize",
        "quant_mode": "quant_mode",
        "mask_mode": "mask_mode",
        "max_seqlen_q": "max_seqlen_q",
        "max_seqlen_kv": "max_seqlen_kv",
        "head_dim_v": "head_dim_v",
    }
    for target, column in integer_columns.items():
        parsed = _str_to_int(row.get(cols.get(column)))
        if parsed is not None:
            attrs[target] = parsed
    for name in ("layout_q", "layout_kv", "layout_out"):
        parsed = _strip_or_none(row.get(cols.get(name)))
        if parsed is not None:
            attrs[name] = parsed
    scale = _str_to_float(row.get(cols.get("softmax_scale")))
    if scale is not None:
        attrs["softmax_scale"] = scale
    enabled = _bool_to_int(row.get(cols.get("return_softmax_lse")))
    if enabled is not None:
        attrs["return_softmax_lse"] = bool(enabled)
    for key, values in (
        ("seqused_q_values", sq_values),
        ("cu_seqlens_q_values", cu_values),
        (
            "cache_seqlens_values",
            _str_to_int_list(row.get(cols.get("cache_seqlens_value"))),
        ),
    ):
        if values is not None:
            attrs[key] = values
    attrs["precision_tolerances"] = _precision_tolerances(row, cols)
    attrs["absolute_precision"] = ABSOLUTE_PRECISION_DEFAULT
    return attrs


def _precision_tolerances(row, cols):
    """默认 ((0.005, 0.000025, 0.005, 0.005, 10),);
    若 attn_out_dtype 含 BF16 → ((0.0078125, 0.0001, 0.005, 0.005, 10),)
    """
    out_dtype = _strip_or_none(row.get(cols.get("attn_out_dtype")))
    if out_dtype and "BF16" in out_dtype.upper():
        return ((0.0078125, 0.0001, 0.005, 0.005, 10),)
    return ((0.005, 0.000025, 0.005, 0.005, 10),)


def _row_to_csv(excel_row, api_name, testcase_suffix, cols):
    """把一行 excel 数据转成一条 csv 行。"""
    raw_name = _strip_or_none(excel_row.get(cols.get("testcase_name")))
    if not raw_name:
        return None
    testcase_name = raw_name.lower() + testcase_suffix

    # Read original sequence fields
    sq_values, cu_values, _, _ = _ensure_tnd_fields(excel_row, cols)

    shapes, dtypes, data_ranges = _build_tensor_lists(
        excel_row, cols, sq_values, cu_values
    )
    attrs = _build_attributes(excel_row, cols, sq_values, cu_values)

    return [
        testcase_name,
        api_name,
        repr(tuple(shapes)),
        repr(tuple(dtypes)),
        "",  # tensor_formats
        repr(attrs),
        "",  # output_tensor_indexes
        "",  # golden_api
        repr(tuple(data_ranges)),
        repr(_precision_tolerances(excel_row, cols)),
        repr(ABSOLUTE_PRECISION_DEFAULT),
    ]


def _parse_sheet_arg(s):
    if s is None:
        return 1
    s = s.strip()
    if s.isdigit():
        return int(s)
    return s


def main():
    parser = argparse.ArgumentParser(
        description="xlsx → ttk 标准 CSV (MLA quant_flash_mla, preserve explicit fields)"
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="xlsx sheet 序号 (1-based int) 或 sheet 名 (str)。默认 1。",
    )
    parser.add_argument(
        "--list-sheets",
        action="store_true",
        help="列出 xlsx 中所有 sheet 名后退出",
    )
    parser.add_argument(
        "--xlsx",
        default=_XLSX,
        help=f"输入 xlsx 路径 (默认 {_XLSX})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出 CSV 路径 (默认同目录 qfmla_excel.csv)",
    )
    args = parser.parse_args()

    if args.list_sheets:
        names = list_sheet_names(args.xlsx)
        print(f"sheets in {args.xlsx}:")
        for i, nm in enumerate(names, 1):
            print(f"  [{i}] {nm}")
        return

    sheet = _parse_sheet_arg(args.sheet)
    xlsx_path = args.xlsx

    print(f"[excel_to_csv] reading sheet={sheet!r} from {xlsx_path}")
    rows = load_sheet(xlsx_path, sheet)
    if not rows:
        raise RuntimeError(f"xlsx sheet {sheet!r} has no rows")
    header_row = rows[0]
    data_rows = rows[1:]
    if len(data_rows) < 1:
        raise RuntimeError(f"expected >=1 data rows in sheet, got {len(data_rows)}")

    cols = _Columns(header_row)

    a_header = _strip_or_none(header_row.get(cols.get("testcase_name")))
    if a_header != "testcase_name":
        raise RuntimeError(f"unexpected header A: {a_header!r}")

    print(f"[excel_to_csv] {len(data_rows)} data rows; explicit fields preserved")

    for fname, api_name, suffix in CSV_PROFILES:
        out_path = args.out or os.path.join(_HERE, fname)
        written = 0
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)
            for row in data_rows:
                csv_row = _row_to_csv(row, api_name, suffix, cols)
                if csv_row is None:
                    continue
                writer.writerow(csv_row)
                written += 1
        print(f"wrote {out_path} ({written} case rows)")

    print(f"\n1 csv file written with {written} case rows.")


if __name__ == "__main__":
    main()
