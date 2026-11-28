# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TTK E2E golden entry point for grouped_matmul_finalize_routing."""

import sys
from pathlib import Path
from runpy import run_path


_SHARED = Path(__file__).resolve().parents[3] / "common/tests/ttk/fused_spec.py"
_make_spec = run_path(str(_SHARED))["make_spec"]
sys.path.insert(0, str(_SHARED.parent))
try:
    from gmm_ttk_graph_models import RoutingGraph
finally:
    sys.path.pop(0)


class PreparedRoutingGraph(RoutingGraph):
    """Keep packed NZ graph inputs contiguous and trace transposes in forward.

    The eager preprocessor exposes transposed API views. Restore their source
    layout before tracing so TorchAir does not clone a discontiguous NZ input.
    """

    def __init__(self, w, scale=None, w_dtype=None, x_dtype=None):
        super().__init__()
        self._w_dtype = w_dtype
        self._x_dtype = x_dtype
        if w_dtype == self._fp4_e2m1 and x_dtype != self._fp4_e2m1:
            w = w.transpose(-1, -2)
            scale = scale.transpose(-3, -2)
        self.register_buffer("weight", w, persistent=False)
        self.register_buffer("weight_scale", scale, persistent=False)

    def forward(self, x, group_list, **kwargs):
        return super().forward(
            x,
            self.weight,
            group_list,
            scale=self.weight_scale,
            w_dtype=self._w_dtype,
            x_dtype=self._x_dtype,
            **kwargs,
        )


E2ESpec = _make_spec("routing", "e2e")
E2ESpec.torch_graph = PreparedRoutingGraph


__spec__ = {
    "torch_npu.npu_grouped_matmul_finalize_routing": "E2ESpec",
}
