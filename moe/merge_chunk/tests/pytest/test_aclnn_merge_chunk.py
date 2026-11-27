# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
from contextlib import ExitStack

import numpy as np
import pytest

from aclnn_utils import AclnnRuntime, C_VOID_P_P


def make_list(runtime, stack, members):
    lib = runtime.nnopbase
    lib.aclCreateTensorList.argtypes = [C_VOID_P_P, ctypes.c_uint64]
    lib.aclCreateTensorList.restype = ctypes.c_void_p
    lib.aclDestroyTensorList.argtypes = [ctypes.c_void_p]
    lib.aclDestroyTensorList.restype = ctypes.c_int
    pointers = (ctypes.c_void_p * len(members))(*(member.ptr for member in members))
    descriptor = lib.aclCreateTensorList(pointers, len(members))
    assert descriptor
    stack.callback(lib.aclDestroyTensorList, descriptor)
    for member in members:
        member.ptr = None  # The list owns the descriptors, but not their buffers.
    return descriptor


def golden(inputs, counts, widths):
    starts = [np.cumsum([0, *row.tolist()]) for row in counts]
    return [
        np.concatenate(
            [
                inputs[path * len(counts) + c][
                    starts[c][expert] : starts[c][expert + 1]
                ]
                for expert in range(len(counts[0]))
                for c in range(len(counts))
            ],
            axis=0,
        )
        for path in range(len(widths))
    ], np.sum(counts, axis=0, dtype=np.int32)


@pytest.mark.ci
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "widths, counts",
    [
        ([16], [[1, 0, 2]]),
        ([16], [[1], [2]]),
        ([16, 48], [[3]]),
        ([16, 32], [[0, 1, 0, 3], [3, 0, 2, 0]]),
        ([16, 32], [[1, 2, 1], [2, 1, 1]]),
        ([16, 32, 64], [[1, 2, 1], [2, 1, 1]]),
        ([16, 32], [[1, 2, 1], [2, 1, 1], [0, 1, 1]]),
        ([16, 32, 64], [[1, 2, 1], [2, 1, 1], [0, 1, 1]]),
        ([16384, 32], [[157, 0, 164], [0, 8, 2]]),
        ([16] * 32, [[1, 1, 0]]),
        ([16], [[1, 0, 1]] * 32),
        ([16] * 32, [[1, 0, 1]] * 32),
        ([90112, 16], [[1] + [0] * 1022 + [1], [0] * 1023 + [1]]),
    ],
)
def test_merge_chunk(widths, counts, precision):
    runtime = AclnnRuntime.instance()
    get_workspace = runtime.set_opapi(
        "aclnnMergeChunkGetWorkspaceSize",
        [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint64),
            C_VOID_P_P,
        ],
    )
    launch = runtime.set_opapi(
        "aclnnMergeChunk",
        [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
    )
    counts = [np.asarray(row, dtype=np.int32) for row in counts]
    rng = np.random.default_rng(42)

    def sample(shape):
        if precision == "bf16":
            # Compare exact BF16 payload bits using the helper's uint16 backing storage.
            return rng.integers(0x3C00, 0x4100, size=shape, dtype=np.uint16)
        return rng.standard_normal(shape).astype(np.float16)

    host = [sample((int(row.sum()), width)) for width in widths for row in counts]
    expected, merged_counts = golden(host, counts, widths)

    with ExitStack() as stack:

        def allocate(array):
            tensor = runtime.tensor(array)
            stack.callback(tensor.close)
            return tensor

        inputs = [allocate(x) for x in host]
        group_lists = [allocate(x) for x in counts]
        outputs = [allocate(np.zeros_like(x)) for x in expected]
        out_counts = allocate(np.zeros_like(merged_counts))
        acl_input = make_list(runtime, stack, inputs)
        acl_counts = make_list(runtime, stack, group_lists)
        acl_output = make_list(runtime, stack, outputs)
        bad, _, _ = runtime.get_workspace(
            get_workspace,
            acl_input,
            acl_counts,
            len(widths) + 1,
            len(counts),
            acl_output,
            out_counts.ptr,
        )
        assert bad == 161002

        status, workspace, executor = runtime.get_workspace(
            get_workspace,
            acl_input,
            acl_counts,
            len(widths),
            len(counts),
            acl_output,
            out_counts.ptr,
        )
        if (len(widths), len(counts)) != (2, 2):
            assert status == 161002 and workspace == 0 and not executor
            return
        assert status == 0, f"GetWorkspaceSize failed: {status}"
        assert runtime.execute(launch, workspace, executor) == 0
        for actual, want in zip(outputs, expected):
            np.testing.assert_array_equal(
                actual.copy_back(want.dtype, want.shape), want
            )
        np.testing.assert_array_equal(
            out_counts.copy_back(merged_counts.dtype, merged_counts.shape),
            merged_counts,
        )
