# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes

import numpy as np
import pytest

from aclnn_utils import C_VOID_P_P, AclnnRuntime


def _golden(in1c1, in1c2, in2c1, in2c2, glc1, glc2):
    out1_parts = []
    out2_parts = []
    sum1 = 0
    sum2 = 0
    for expert_id in range(glc1.shape[0]):
        n1 = int(glc1[expert_id])
        n2 = int(glc2[expert_id])
        out1_parts.append(in1c1[sum1 : sum1 + n1])
        out1_parts.append(in1c2[sum2 : sum2 + n2])
        out2_parts.append(in2c1[sum1 : sum1 + n1])
        out2_parts.append(in2c2[sum2 : sum2 + n2])
        sum1 += n1
        sum2 += n2
    return (
        np.concatenate(out1_parts, axis=0),
        np.concatenate(out2_parts, axis=0),
        glc1 + glc2,
    )


@pytest.mark.ci
def test_aclnn_merge_chunk2():
    runtime = AclnnRuntime.instance()
    try:
        get_workspace = runtime.set_opapi(
            "aclnnMergeChunk2GetWorkspaceSize",
            [ctypes.c_void_p] * 9 + [ctypes.POINTER(ctypes.c_uint64), C_VOID_P_P],
        )
        execute = runtime.set_opapi(
            "aclnnMergeChunk2",
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
    except RuntimeError as error:
        pytest.skip(str(error))

    t1, t2, h1, h2 = 8, 6, 16, 16
    rng = np.random.default_rng(0)
    in1c1 = rng.standard_normal((t1, h1)).astype(np.float16)
    in1c2 = rng.standard_normal((t2, h1)).astype(np.float16)
    in2c1 = rng.standard_normal((t1, h2)).astype(np.float16)
    in2c2 = rng.standard_normal((t2, h2)).astype(np.float16)
    glc1 = np.array([2, 2, 2, 2], dtype=np.int32)
    glc2 = np.array([1, 2, 1, 2], dtype=np.int32)
    out1_expected, out2_expected, gl_expected = _golden(
        in1c1, in1c2, in2c1, in2c2, glc1, glc2
    )

    tensors = []
    try:
        in1c1_t = runtime.tensor(in1c1)
        in1c2_t = runtime.tensor(in1c2)
        in2c1_t = runtime.tensor(in2c1)
        in2c2_t = runtime.tensor(in2c2)
        glc1_t = runtime.tensor(glc1)
        glc2_t = runtime.tensor(glc2)
        out1 = runtime.tensor(np.zeros((t1 + t2, h1), dtype=np.float16))
        out2 = runtime.tensor(np.zeros((t1 + t2, h2), dtype=np.float16))
        gl = runtime.tensor(np.zeros(glc1.shape[0], dtype=np.int32))
        tensors.extend(
            [in1c1_t, in1c2_t, in2c1_t, in2c2_t, glc1_t, glc2_t, out1, out2, gl]
        )

        ret, workspace_size, executor = runtime.get_workspace(
            get_workspace,
            in1c1_t.ptr,
            in1c2_t.ptr,
            in2c1_t.ptr,
            in2c2_t.ptr,
            glc1_t.ptr,
            glc2_t.ptr,
            out1.ptr,
            out2.ptr,
            gl.ptr,
        )
        assert ret == 0, f"aclnnMergeChunk2GetWorkspaceSize failed: {ret}"
        ret = runtime.execute(execute, workspace_size, executor)
        assert ret == 0, f"aclnnMergeChunk2 failed: {ret}"

        out1_host = out1.copy_back(np.float16, [t1 + t2, h1])
        out2_host = out2.copy_back(np.float16, [t1 + t2, h2])
        gl_host = gl.copy_back(np.int32, [glc1.shape[0]])
        np.testing.assert_array_equal(out1_host, out1_expected)
        np.testing.assert_array_equal(out2_host, out2_expected)
        np.testing.assert_array_equal(gl_host, gl_expected)
    finally:
        for tensor in reversed(tensors):
            tensor.close()
