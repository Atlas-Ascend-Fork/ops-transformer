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

"""Static wire-schema checks; these do not replace compilation with the CANN SDK."""

import ctypes
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[3]
HEADERS = [
    ROOT / "op_kernel/arch35/common/quant_block_sparse_attn_tiling_data.h",
    ROOT / "op_kernel/arch35/qbsa_fp8/qbsa_fp8_tiling_data.h",
    ROOT / "op_kernel/arch35/qbsa_mxfp8/qbsa_mxfp8_tiling_data.h",
    ROOT / "op_kernel/arch92/qbsa_fp8/qbsa_fp8_tiling_data.h",
]


def load_wire_types():
    types = {
        "uint8_t": ctypes.c_uint8,
        "uint32_t": ctypes.c_uint32,
        "uint64_t": ctypes.c_uint64,
        "int64_t": ctypes.c_int64,
        "float": ctypes.c_float,
    }
    source = "\n".join(p.read_text(encoding="utf-8") for p in HEADERS)
    source = re.sub(r"//[^\n]*", "", source)
    core_count = int(re.search(r"QBSA_MAX_CORE_NUM = (\d+)U", source)[1])
    for match in re.finditer(r"struct (\w+)\s*\{([^{}]*)\};", source, re.S):
        name, body = match.groups()
        fields = []
        for declaration in body.split(";"):
            if not declaration.strip():
                continue
            field = re.fullmatch(r"\s*(\w+)\s+(\w+)(?:\[(\w+)\])?\s*", declaration)
            if field is None:
                raise ValueError(f"Unsupported wire declaration: {declaration}")
            type_name, field_name, count = field.groups()
            ctype = types[type_name]
            if count:
                size = {
                    "QBSA_CORE_SPLIT_NUM": core_count + 1,
                    "QBSA_ARCH92_CORE_SPLIT_NUM": 65,
                }.get(count)
                if size is None:
                    size = int(count)
                ctype *= size
            fields.append((field_name, ctype))
        types[name] = type(name, (ctypes.Structure,), {"_pack_": 8, "_fields_": fields})
    return types, source


class QbsaWireLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.types, cls.source = load_wire_types()

    def test_independent_payload_sizes(self):
        for name, size in {
            "QuantBlockSparseAttnCommonParams": 120,
            "QuantBlockSparseAttnFp8TilingData": 304,
            "QuantBlockSparseAttnMxfp8TilingData": 216,
            "QuantBlockSparseAttnFp8Arch92TilingData": 416,
        }.items():
            with self.subTest(name=name):
                self.assertEqual(ctypes.sizeof(self.types[name]), size)

    def test_arch92_can_store_boundary_after_last_of_64_cores(self):
        wire = self.types["QuantBlockSparseAttnFp8Arch92TilingData"]()
        wire.commonParams.baseParams.coreNum = 64
        wire.fp8Params.bnStartIdx[64] = 128
        decoded = type(wire).from_buffer_copy(bytes(wire))
        self.assertEqual(len(decoded.fp8Params.bnStartIdx), 65)
        self.assertEqual(decoded.fp8Params.bnStartIdx[64], 128)
        self.assertEqual(
            len(self.types["QuantBlockSparseAttnFp8Params"]().bnStartIdx), 37
        )

    def test_mx_payload_preserves_existing_field_offsets(self):
        wire = self.types["QuantBlockSparseAttnMxParams"]
        for name, offset in {
            "keyStrides": 0,
            "valueStrides": 16,
            "kScaleStrides": 32,
            "vScaleStrides": 48,
            "qSeqSize": 64,
            "scaleGroupSize": 68,
            "scaleLastDim": 72,
            "queryScaleDSize": 76,
            "keyScaleDSize": 80,
            "valueScaleBlockSize": 84,
            "valueScaleDSize": 88,
            "pScaleDtype": 92,
        }.items():
            with self.subTest(field=name):
                self.assertEqual(getattr(wire, name).offset, offset)

    def test_cpp_size_and_offset_assertions_match_schema(self):
        for name, size in re.findall(r"sizeof\((\w+)\) == (\d+)U", self.source):
            self.assertEqual(ctypes.sizeof(self.types[name]), int(size), name)
        for name, field, offset in re.findall(
            r"offsetof\((\w+), (\w+)\) == (\d+)U", self.source
        ):
            self.assertEqual(
                getattr(self.types[name], field).offset, int(offset), (name, field)
            )


if __name__ == "__main__":
    unittest.main()
