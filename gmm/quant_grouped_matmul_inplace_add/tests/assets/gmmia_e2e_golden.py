#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""GMMIA Torch test uses real Torch API arguments and decodes byte carriers."""

import sys
from pathlib import Path
from runpy import run_path

_ASSETS = Path(__file__).resolve().parent
_REF = run_path(str(_ASSETS / "gmmia_reference.py"))
_as_array, _inplace = _REF["as_array"], _REF["inplace"]
_prepare = run_path(str(_ASSETS / "inputs.py"))["_prepare_gmmia_inputs"]


class E2eSpec:
    @staticmethod
    def customize_inputs(
        input,
        x1,
        x2,
        x2_scale,
        group_list,
        *,
        x1_scale=None,
        group_list_type=0,
        x1_dtype=None,
        x2_dtype=None,
        ttk_graph_counts=None,
        ttk_group_list=None,
        **kwargs,
    ):
        _prepare(
            x1,
            x2,
            x1_scale,
            x2_scale,
            group_list,
            group_list_type,
            hifloat8=x1_dtype == 290,
        )
        if ttk_graph_counts is not None:
            counts = group_list.new_tensor(ttk_graph_counts)
            group_list.copy_(counts if group_list_type == 1 else counts.cumsum(0))
        if ttk_group_list is not None:
            group_list.copy_(group_list.new_tensor(ttk_group_list))

    @staticmethod
    def golden(
        input,
        x1,
        x2,
        x2_scale,
        group_list,
        *,
        x1_scale=None,
        group_list_type=0,
        x1_dtype=None,
        x2_dtype=None,
        x1_scale_dtype=None,
        x2_scale_dtype=None,
        ttk_graph_counts=None,
        **kwargs,
    ):
        scale1 = _as_array(x1_scale, x1_scale_dtype)
        if ttk_graph_counts is None and scale1 is not None and scale1.ndim == 3:
            scale1 = scale1.swapaxes(0, 1)
        x1_array = _as_array(x1, x1_dtype)
        return _inplace(
            x1_array.T if ttk_graph_counts is None else x1_array,
            _as_array(x2, x2_dtype),
            scale1,
            _as_array(x2_scale, x2_scale_dtype),
            _as_array(group_list),
            _as_array(input),
            group_list_type,
        )


sys.path.insert(0, str(_ASSETS.parents[2] / "common/tests/ttk"))
try:
    from gmm_ttk_graph_models import InplaceGraph
finally:
    sys.path.pop(0)


class PreparedInplaceGraph(InplaceGraph):
    """Bind logical inputs before Dynamo traces the graph-side transposes.

    The eager leg consumes API views, materialized by npu_preprocess below.
    Restoring the logical layout in __init__ keeps that eager-only adaptation
    outside the traced forward and preserves the documented MX scale transpose.
    """

    def __init__(self, x1, x1_scale=None):
        super().__init__()
        self.register_buffer("x1", x1.transpose(0, 1), persistent=False)
        logical_scale = (
            x1_scale.transpose(0, 1)
            if x1_scale is not None and x1_scale.ndim == 3
            else x1_scale
        )
        self.register_buffer("x1_scale", logical_scale, persistent=False)

    def forward(self, input, x2, x2_scale, group_list, **kwargs):
        return super().forward(
            input,
            self.x1,
            x2,
            x2_scale,
            group_list,
            x1_scale=self.x1_scale,
            **kwargs,
        )


class GraphSpec(E2eSpec):
    torch_graph = PreparedInplaceGraph

    @staticmethod
    def npu_preprocess(
        input,
        x1,
        x2,
        x2_scale,
        group_list,
        *,
        x1_scale=None,
        ttk_graph_counts=None,
        **kwargs,
    ):
        """Expose API views without copying or changing low precision values."""
        if ttk_graph_counts is not None:
            if x1.stride(0) > x1.stride(1):
                x1.data = x1.transpose(0, 1).detach()
            if (
                x1_scale is not None
                and x1_scale.ndim == 3
                and x1_scale.stride(0) > x1_scale.stride(1)
            ):
                x1_scale.data = x1_scale.transpose(0, 1).detach()


__spec__ = {
    "torch_npu.npu_add_quant_gmm": "E2eSpec",
    "torch_npu.npu_add_quant_gmm_": "GraphSpec",
}
