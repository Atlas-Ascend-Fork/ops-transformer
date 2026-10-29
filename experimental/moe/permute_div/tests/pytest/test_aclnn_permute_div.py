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


@pytest.mark.ci
@pytest.mark.parametrize(
    "num_indice,divisor",
    [
        pytest.param(1024, 12, id="n_1024_d_12"),
        pytest.param(1, 1, id="n_1_d_1"),
        pytest.param(257, 7, id="n_257_d_7"),
    ],
)
def test_aclnn_permute_div(num_indice, divisor):
    runtime = AclnnRuntime.instance()
    try:
        get_workspace = runtime.set_opapi(
            "aclnnPermuteDivGetWorkspaceSize",
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                C_VOID_P_P,
            ],
        )
        execute = runtime.set_opapi(
            "aclnnPermuteDiv",
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
    except RuntimeError as error:
        pytest.skip(str(error))

    indice_host = np.arange(num_indice, dtype=np.int32)
    expected = np.floor_divide(indice_host, divisor).astype(np.int32)

    tensors = []
    try:
        indice = runtime.tensor(indice_host)
        out = runtime.tensor(np.zeros(num_indice, dtype=np.int32))
        tensors.extend([indice, out])

        ret, workspace_size, executor = runtime.get_workspace(
            get_workspace, indice.ptr, divisor, out.ptr
        )
        assert ret == 0, f"aclnnPermuteDivGetWorkspaceSize failed: {ret}"
        ret = runtime.execute(execute, workspace_size, executor)
        assert ret == 0, f"aclnnPermuteDiv failed: {ret}"

        out_host = out.copy_back(np.int32, [num_indice])
        np.testing.assert_array_equal(out_host, expected)
    finally:
        for tensor in reversed(tensors):
            tensor.close()
