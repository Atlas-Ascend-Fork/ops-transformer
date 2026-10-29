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

"""直接调用 CANN aclnn C 接口（ctypes），不依赖 torch_npu 的算子 op-plugin 绑定。

依赖：libcust_opapi.so（算子自带 aclnn）、libnnopbase.so（aclCreateTensor/aclCreateIntArray）、
libascendcl（aclrt*）。torch/torch_npu 仅用于申请 NPU 显存和获取 aclrtStream。
"""

import ctypes
from ctypes import (
    c_void_p,
    c_int64,
    c_uint64,
    c_int,
    c_char_p,
    c_double,
    c_bool,
    byref,
    POINTER,
)

import torch
import torch_npu

_OPAPI = ctypes.CDLL("libcust_opapi.so")
_NNOP = ctypes.CDLL("libnnopbase.so")

_DTYPE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.int8: 2,
    torch.int32: 3,
    torch.uint8: 4,
    torch.int16: 6,
    torch.uint16: 7,
    torch.uint32: 8,
    torch.uint64: 10,
    torch.int64: 9,
    torch.float64: 11,
    torch.bool: 12,
    torch.bfloat16: 27,
}
_FORMAT_ND = 2

_NNOP.aclCreateTensor.restype = c_void_p
_NNOP.aclCreateTensor.argtypes = [
    POINTER(c_int64),
    c_uint64,
    c_int,
    POINTER(c_int64),
    c_int64,
    c_int,
    POINTER(c_int64),
    c_uint64,
    c_void_p,
]
_NNOP.aclCreateIntArray.restype = c_void_p
_NNOP.aclCreateIntArray.argtypes = [POINTER(c_int64), c_uint64]


def to_acl(tensor):
    """把 torch NPU tensor 包装成 aclTensor（None -> 空指针，用于可选输入）。"""
    if tensor is None:
        return c_void_p(None)
    shape = list(tensor.shape)
    n = len(shape)
    dims = (c_int64 * n)(*shape)
    strides, s = [], 1
    for d in reversed(shape):
        strides.insert(0, s)
        s *= d
    strd = (c_int64 * n)(*strides)
    return _NNOP.aclCreateTensor(
        dims,
        n,
        _DTYPE[tensor.dtype],
        strd,
        0,
        _FORMAT_ND,
        dims,
        n,
        c_void_p(tensor.data_ptr()),
    )


def int_array(values):
    if values is None:
        return c_void_p(None)
    arr = (c_int64 * len(values))(*values)
    return _NNOP.aclCreateIntArray(arr, len(values))


_CTYPE = {
    "tensor": c_void_p,
    "int64": c_int64,
    "double": c_double,
    "bool": c_bool,
    "str": c_char_p,
    "intarray": c_void_p,
}


def call(get_ws_name, op_name, spec):
    """spec: [(kind, value), ...]，按 aclnn GetWorkspaceSize 的形参顺序（含输出）。

    kind ∈ {tensor, int64, double, bool, str, intarray}
    """
    fn = getattr(_OPAPI, get_ws_name)
    fn.restype = c_int
    argtypes, cargs = [], []
    for kind, value in spec:
        argtypes.append(_CTYPE[kind])
        if kind == "tensor":
            cargs.append(to_acl(value))
        elif kind == "intarray":
            cargs.append(int_array(value))
        elif kind == "str":
            cargs.append(c_char_p(None) if value is None else c_char_p(value.encode()))
        else:
            cargs.append(value)

    ws = c_uint64(0)
    executor = c_void_p()
    fn.argtypes = argtypes + [POINTER(c_uint64), POINTER(c_void_p)]
    st = fn(*(cargs + [byref(ws), byref(executor)]))
    if st != 0:
        raise RuntimeError("%s failed, status=%d" % (get_ws_name, st))

    run = getattr(_OPAPI, op_name)
    run.restype = c_int
    run.argtypes = [c_void_p, c_uint64, c_void_p, c_void_p]
    dev = torch.npu.current_device()
    ws_tensor = torch.empty(
        max(int(ws.value), 1), dtype=torch.uint8, device="npu:%d" % dev
    )
    stream = c_void_p(torch_npu.npu.current_stream().npu_stream)
    st = run(c_void_p(ws_tensor.data_ptr()), ws.value, executor, stream)
    if st != 0:
        raise RuntimeError("%s failed, status=%d" % (op_name, st))
    torch.npu.synchronize()
