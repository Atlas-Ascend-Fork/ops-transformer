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
    "num_token,num_expert",
    [
        pytest.param(100, 8, id="num_token_100_expert_8"),
        pytest.param(1, 1, id="single_token_single_expert"),
        pytest.param(257, 16, id="num_token_257_expert_16"),
    ],
)
def test_aclnn_histgram(num_token, num_expert):
    runtime = AclnnRuntime.instance()
    try:
        get_workspace = runtime.set_opapi(
            "aclnnHistgramGetWorkspaceSize",
            [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                C_VOID_P_P,
            ],
        )
        execute = runtime.set_opapi(
            "aclnnHistgram",
            [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p],
        )
    except RuntimeError as error:
        pytest.skip(str(error))

    rng = np.random.default_rng(0)
    input_host = rng.integers(0, num_expert, size=(num_token,), dtype=np.int32)
    expected = np.bincount(input_host, minlength=num_expert).astype(np.int32)

    tensors = []
    try:
        input_tensor = runtime.tensor(input_host)
        out = runtime.tensor(np.zeros(num_expert, dtype=np.int32))
        tensors.extend([input_tensor, out])

        ret, workspace_size, executor = runtime.get_workspace(
            get_workspace, input_tensor.ptr, num_expert, out.ptr
        )
        assert ret == 0, f"aclnnHistgramGetWorkspaceSize failed: {ret}"
        ret = runtime.execute(execute, workspace_size, executor)
        assert ret == 0, f"aclnnHistgram failed: {ret}"

        out_host = out.copy_back(np.int32, [num_expert])
        np.testing.assert_array_equal(out_host, expected)
    finally:
        for tensor in reversed(tensors):
            tensor.close()
