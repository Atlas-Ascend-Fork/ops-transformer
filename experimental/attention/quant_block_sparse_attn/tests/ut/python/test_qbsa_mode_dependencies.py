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

"""Enforce ownership boundaries by traversing local C++ include dependencies."""

import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]


def include_closure(directory):
    pending = [
        p.resolve() for p in directory.rglob("*") if p.suffix in {".cpp", ".h", ".hpp"}
    ]
    visited = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        for include in re.findall(
            r'^\s*#include\s+"([^"]+)"', path.read_text(encoding="utf-8"), re.M
        ):
            target = (path.parent / include).resolve()
            if target.is_file() and ROOT.resolve() in target.parents:
                pending.append(target)
    return visited


class QbsaModeDependencyTest(unittest.TestCase):
    def assert_no_dependency(self, owner, forbidden):
        forbidden = forbidden.resolve()
        reached = [p for p in include_closure(owner) if forbidden in p.parents]
        self.assertEqual(reached, [], f"{owner} depends on another mode: {reached}")

    def test_host_strategies_do_not_include_each_other(self):
        for mode, other in [("fp8", "mxfp8"), ("mxfp8", "fp8")]:
            with self.subTest(mode=mode):
                base = ROOT / "op_host/arch35"
                self.assert_no_dependency(base / f"qbsa_{mode}", base / f"qbsa_{other}")

    def test_kernel_blocks_do_not_include_each_other(self):
        for mode, other in [("fp8", "mxfp8"), ("mxfp8", "fp8")]:
            with self.subTest(mode=mode):
                base = ROOT / "op_kernel/arch35"
                self.assert_no_dependency(base / f"qbsa_{mode}", base / f"qbsa_{other}")

    def test_arch92_modes_do_not_include_each_other(self):
        for surface in ["op_host", "op_kernel"]:
            for mode, other in [("fp8", "mxfp8"), ("mxfp8", "fp8")]:
                with self.subTest(surface=surface, mode=mode):
                    base = ROOT / surface / "arch92"
                    owner = base / f"qbsa_{mode}"
                    self.assertTrue(owner.is_dir())
                    self.assert_no_dependency(owner, base / f"qbsa_{other}")

    def test_arch92_shared_helpers_are_owned_by_common_and_used_by_both_modes(self):
        base = ROOT / "op_kernel/arch92"
        common = base / "common"
        shared = set(common.glob("*.h"))
        self.assertTrue(shared)
        for mode in ["fp8", "mxfp8"]:
            owner = base / f"qbsa_{mode}"
            with self.subTest(mode=mode):
                self.assertFalse((owner / "common").exists())
                self.assert_no_dependency(common, owner)
                reached = include_closure(owner)
                self.assertTrue(
                    all(path.resolve() in reached for path in shared),
                    f"{owner} does not consume all shared helpers",
                )

    def test_arch92_relative_includes_exist(self):
        for surface in ["op_host", "op_kernel"]:
            for path in (ROOT / surface / "arch92").rglob("*"):
                if path.suffix not in {".h", ".cpp", ".hpp"}:
                    continue
                for include in re.findall(
                    r'^\s*#include\s+"([^"]+)"', path.read_text(encoding="utf-8"), re.M
                ):
                    if include.startswith("."):
                        self.assertTrue(
                            (path.parent / include).is_file(), f"{path}: {include}"
                        )

    def test_common_headers_do_not_include_mode_private_headers(self):
        for common in [ROOT / "op_host/tiling_common", ROOT / "op_kernel/common"]:
            for mode in ["fp8", "mxfp8"]:
                with self.subTest(common=common, mode=mode):
                    self.assert_no_dependency(
                        common, common.parent / "arch35" / f"qbsa_{mode}"
                    )


if __name__ == "__main__":
    unittest.main()
