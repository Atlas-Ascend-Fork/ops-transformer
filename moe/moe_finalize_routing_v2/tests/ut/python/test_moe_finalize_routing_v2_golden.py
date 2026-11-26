#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU only. From repo root:

python -B -m unittest discover -s moe/moe_finalize_routing_v2/tests/ut/python -p test_moe_finalize_routing_v2_golden.py -v

Requires numpy and torch. Regression for PR #16931 review (ConanHuang): the
MXFP8 golden must decode native E8M0 scales as raw exponent bytes. A native
float8_e8m0fnu tensor holding exponent 127 (2**0 = 1.0) must not be routed
through a BF16 numeric fallback, whose storage bytes (0x80 0x3F) would be
misread as exponent 0x80. Both the uint8-encoded and the native E8M0
representations of the same scale bytes must produce identical outputs.
"""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch


GOLDEN_PATH = Path(__file__).resolve().parents[2] / "assets/golden.py"


def _load_golden():
    spec = importlib.util.spec_from_file_location("mfr2_golden", GOLDEN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMoeFinalizeRoutingV2GoldenMxfp8(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = _load_golden()

    def _run_mxfp8_case(self, scale_tensor):
        """One route, 32 E4M3FN ones, scale exponent 127 (2**0) everywhere."""
        codes = np.full((1, 32), 0x38, dtype=np.uint8)  # E4M3FN encoding of 1.0
        expanded_x = torch.from_numpy(codes).view(torch.float8_e4m3fn)
        row_idx = np.zeros(1, dtype=np.int32)

        outs = self.golden._golden_shared(
            self.golden._to_numpy(expanded_x),
            row_idx,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.golden._to_numpy(scale_tensor),
            2,
            None,
            None,
            None,
            1,
        )
        self.assertEqual(len(outs), 1)
        return outs[0].astype(np.float32)

    def test_native_e8m0_keeps_raw_bytes(self):
        if getattr(torch, "float8_e8m0fnu", None) is None:
            self.skipTest("torch has no float8_e8m0fnu dtype")
        scale_bytes = np.array([[127, 127]], dtype=np.uint8)
        native = torch.from_numpy(scale_bytes.copy()).view(torch.float8_e8m0fnu)
        converted = self.golden._to_numpy(native)
        self.assertEqual(converted.dtype, np.uint8)
        np.testing.assert_array_equal(converted.view(np.uint8), scale_bytes)

    def test_uint8_and_native_e8m0_scales_agree(self):
        # scale exponent 127 -> factor 2**0 = 1.0, so both representations must
        # yield an all-ones BF16 output. With the former BF16 fallback the
        # native variant decoded 0x80 as exponent 0x80 and returned 2.0.
        scale_bytes = np.array([[127, 127]], dtype=np.uint8)
        out_uint8 = self._run_mxfp8_case(torch.from_numpy(scale_bytes.copy())).reshape(
            1, 32
        )
        if getattr(torch, "float8_e8m0fnu", None) is None:
            self.skipTest("torch has no float8_e8m0fnu dtype")
        native = torch.from_numpy(scale_bytes.copy()).view(torch.float8_e8m0fnu)
        out_native = self._run_mxfp8_case(native).reshape(1, 32)
        np.testing.assert_allclose(
            out_uint8, np.ones((1, 32), np.float32), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            out_native, np.ones((1, 32), np.float32), rtol=0, atol=0
        )
        np.testing.assert_array_equal(out_uint8, out_native)


if __name__ == "__main__":
    unittest.main()
