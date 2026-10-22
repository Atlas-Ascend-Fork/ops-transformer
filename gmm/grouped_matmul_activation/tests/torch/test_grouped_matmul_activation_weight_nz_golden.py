# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import unittest
import torch
from grouped_matmul_activation_weight_nz_golden import (
    grouped_matmul_activation_weight_nz_golden as golden,
)


class GoldenContract(unittest.TestCase):
    def inputs(self):
        return torch.ones(3, 128), torch.ones(2, 128, 128), torch.tensor([1, 3])

    def test_default_split_swiglu_and_empty_expert(self):
        x, w, _ = self.inputs()
        result = golden(x, w, torch.tensor([0, 3]))
        u = torch.tensor(7.0)
        expected = (u / (1 + torch.exp(-1.702 * u))) * 8.0
        self.assertTrue(
            torch.equal(
                result,
                torch.full((3, 64), expected.to(torch.bfloat16), dtype=torch.bfloat16),
            )
        )

    def test_count_and_cumulative_partition_agree(self):
        x, w, groups = self.inputs()
        self.assertTrue(
            torch.equal(
                golden(x, w, groups),
                golden(x, w, torch.tensor([1, 2]), group_list_type=1),
            )
        )

    def test_partial_partition_rejected(self):
        x, w, _ = self.inputs()
        with self.assertRaisesRegex(ValueError, "valid token partition"):
            golden(x, w, torch.tensor([1, 2]))

    def test_mode_zero_rejected(self):
        with self.assertRaisesRegex(ValueError, "only swiglu_mode=2"):
            golden(*self.inputs(), swiglu_mode=0)

    def test_nonfinite_attr_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            golden(*self.inputs(), glu_alpha=float("nan"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
