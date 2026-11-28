# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
from pathlib import Path
from runpy import run_path

import numpy as np
import torch

_REFERENCE = run_path(
    str(Path(__file__).resolve().parents[4] / "common/tests/ttk/fused_spec.py")
)


def test_transposed_storage_keeps_values_for_either_axis_order():
    original = torch.arange(24).reshape(2, 3, 4)
    for axes in ((-1, -2), (-2, -1)):
        view = _REFERENCE["_with_transposed_storage"](original, *axes)
        torch.testing.assert_close(view, original)
        assert view.stride(-2) == 1
        assert view.stride(-1) == 3
        assert _REFERENCE["_with_transposed_storage"](view, *axes) is view


def test_fp4_transposed_carrier_unpacks_k_axis():
    packed = torch.full((1, 2, 3), 0x21, dtype=torch.uint8)
    view = _REFERENCE["_with_transposed_storage"](packed, -1, -2)
    actual = _REFERENCE["_decode"](view, "float4_e2m1")
    expected = np.array([[[0.5] * 3, [1.0] * 3, [0.5] * 3, [1.0] * 3]])
    np.testing.assert_array_equal(actual, expected)
