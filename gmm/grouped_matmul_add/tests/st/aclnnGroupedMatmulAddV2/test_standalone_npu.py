#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Run GmmAddV2 accuracy checks against an explicitly selected custom vendor.

On an A2/A3 host with CANN, PyTorch and torch_npu installed:

  source /path/to/cann/set_env.sh
  export ASCEND_CUSTOM_OPP_PATH=/path/to/custom/vendors/<vendor>
  export LD_LIBRARY_PATH=/path/to/custom/vendors/<vendor>/op_api/lib:$LD_LIBRARY_PATH
  python test_standalone_npu.py --opapi /path/to/custom/vendors/<vendor>/op_api/lib/libcust_opapi.so

Use --device to select an idle NPU. The selected opapi is loaded by absolute
path; the vendor installation and kernel selection must still be verified in
the CANN runtime logs for the target environment.
"""

import argparse
import ctypes
import itertools
import os
import random
from pathlib import Path

import torch

# Importing torch_npu registers the NPU device.
import torch_npu  # noqa: F401


DTYPE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.int32: 3,
    torch.int64: 9,
    torch.bfloat16: 27,
}
FORMAT_ND = 2
ACLNN_ERR_PARAM_INVALID = 161002
M, N = 273, 272  # Both dimensions cross the A2/A3 base-block boundary; N is 16-aligned.
LENGTHS = (37, 0, 23, 52)  # Nonuniform groups, an empty group, and non-aligned group K.
TOLERANCES = {
    torch.float32: (2.0**-10, 2.0**-16, 1.0e-2),
    torch.float16: (2.0**-9, 2.0**-9, 1.0e-1),
    torch.bfloat16: (2.0**-6, 2.0**-6, 1.0),
}


class DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def assert_symbol_in_library(function, expected_path):
    """Avoid silently resolving an ACLNN entry point from a dependency."""
    libdl = ctypes.CDLL("libdl.so.2")
    libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    libdl.dladdr.restype = ctypes.c_int
    info = DlInfo()
    if (
        libdl.dladdr(ctypes.cast(function, ctypes.c_void_p), ctypes.byref(info)) == 0
        or not info.dli_fname
    ):
        raise RuntimeError("dladdr could not determine the ACLNN symbol origin")
    actual_path = Path(os.fsdecode(info.dli_fname)).resolve(strict=True)
    if actual_path != expected_path:
        raise RuntimeError(
            f"ACLNN symbol resolved from {actual_path}, expected {expected_path}"
        )


def open_api(opapi_path):
    """Resolve the exact vendor opapi library, never the generic system opapi."""
    if not Path(opapi_path).is_absolute():
        raise ValueError("--opapi must be an absolute path")
    path = Path(opapi_path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"--opapi must name a vendor shared library: {path}")
    opp_path = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "")
    roots = [Path(entry).resolve() for entry in opp_path.split(os.pathsep) if entry]
    if not roots:
        raise RuntimeError(
            "ASCEND_CUSTOM_OPP_PATH must identify the custom vendor before Python starts"
        )
    if not any(root == path or root in path.parents for root in roots):
        raise RuntimeError(f"--opapi is not inside ASCEND_CUSTOM_OPP_PATH: {path}")
    print(f"custom opapi: {path}", flush=True)
    print(f"ASCEND_CUSTOM_OPP_PATH={opp_path}", flush=True)
    nnop = ctypes.CDLL("libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
    opapi = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    ptr, i64, u64 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64
    p_i64 = ctypes.POINTER(i64)
    nnop.aclCreateTensor.argtypes = [
        p_i64,
        u64,
        ctypes.c_int,
        p_i64,
        i64,
        ctypes.c_int,
        p_i64,
        u64,
        ptr,
    ]
    nnop.aclCreateTensor.restype = ptr
    nnop.aclDestroyTensor.argtypes = [ptr]
    nnop.aclDestroyTensor.restype = ctypes.c_int
    get_workspace = opapi.aclnnGroupedMatmulAddV2GetWorkspaceSize
    assert_symbol_in_library(get_workspace, path)
    get_workspace.argtypes = [
        ptr,
        ptr,
        ptr,
        ptr,
        ctypes.c_bool,
        ctypes.c_bool,
        i64,
        i64,
        ctypes.POINTER(u64),
        ctypes.POINTER(ptr),
    ]
    get_workspace.restype = ctypes.c_int
    run = opapi.aclnnGroupedMatmulAddV2
    assert_symbol_in_library(run, path)
    run.argtypes = [ptr, u64, ptr, ptr]
    run.restype = ctypes.c_int
    return nnop, get_workspace, run


def make_acl_tensor(nnop, tensor, storage_tensor=None):
    # A view must describe its actual backing allocation, stride and offset.
    # Do not pass tensor.data_ptr() together with storage_offset: that would
    # apply the offset twice to sliced views.
    if storage_tensor is None:
        if not tensor.is_contiguous() or tensor.storage_offset() != 0:
            raise ValueError(
                "a non-contiguous or offset tensor needs its backing storage_tensor"
            )
        storage_tensor = tensor
    if not storage_tensor.is_contiguous() or storage_tensor.storage_offset() != 0:
        raise ValueError(
            "storage_tensor must be the contiguous zero-offset backing allocation"
        )
    offset = tensor.storage_offset()
    if tensor.data_ptr() != storage_tensor.data_ptr() + offset * tensor.element_size():
        raise ValueError("tensor and storage_tensor do not share the expected storage")
    shape = tuple(tensor.shape)
    ndim = len(shape)
    dims = (ctypes.c_int64 * ndim)(*shape)
    strides = (ctypes.c_int64 * ndim)(*tensor.stride())
    storage_shape = tuple(storage_tensor.shape)
    storage_dims = (ctypes.c_int64 * len(storage_shape))(*storage_shape)
    handle = nnop.aclCreateTensor(
        dims,
        ndim,
        DTYPE[tensor.dtype],
        strides,
        offset,
        FORMAT_ND,
        storage_dims,
        len(storage_shape),
        ctypes.c_void_p(storage_tensor.data_ptr()),
    )
    if not handle:
        raise RuntimeError(
            f"aclCreateTensor failed for shape={shape}, dtype={tensor.dtype}"
        )
    return handle


def invoke(
    nnop,
    get_workspace,
    run,
    x,
    w,
    groups,
    y,
    transpose_x,
    list_type,
    storage_tensors=None,
):
    handles = []
    storage_tensors = storage_tensors or (None, None, None, None)
    try:
        for tensor, storage in zip((x, w, groups, y), storage_tensors):
            handles.append(make_acl_tensor(nnop, tensor, storage))
        workspace_size, executor = ctypes.c_uint64(), ctypes.c_void_p()
        status = get_workspace(
            *handles,
            transpose_x,
            False,
            2,
            list_type,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if status != 0:
            raise RuntimeError(f"GetWorkspaceSize failed: status={status}")
        if not executor.value:
            raise RuntimeError("GetWorkspaceSize returned a null executor")
        workspace = torch.empty(
            max(workspace_size.value, 1), dtype=torch.uint8, device=y.device
        )
        stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        status = run(
            ctypes.c_void_p(workspace.data_ptr()),
            workspace_size.value,
            executor,
            stream,
        )
        if status != 0:
            raise RuntimeError(f"aclnnGroupedMatmulAddV2 failed: status={status}")
        torch.npu.synchronize()
    finally:
        for handle in handles:
            nnop.aclDestroyTensor(handle)


def check_group(actual, expected, dtype, group_idx):
    rtol, atol, max_allowed = TOLERANCES[dtype]
    actual = actual.float()
    expected = expected.float()
    if not torch.isfinite(actual).all():
        raise AssertionError(f"group {group_idx}: output contains NaN or Inf")
    err = (actual - expected).abs()
    matched = (err <= atol + rtol * expected.abs()).float().mean().item()
    max_error = err.max().item()
    if matched < 0.99 or max_error > max_allowed:
        raise AssertionError(
            f"group {group_idx}: matched={matched:.6f}, max_error={max_error:.8g}, "
            f"required >=0.99 and max_error<={max_allowed}"
        )
    return matched, max_error


def run_case(
    api,
    device,
    input_dtype,
    output_dtype,
    transpose_x,
    list_type,
    n=N,
    noncontiguous_inputs=False,
    noncontiguous_groups=False,
    m=M,
    lengths=LENGTHS,
):
    torch.manual_seed(20261120)
    total_k = sum(lengths)
    # Quantize inputs first: the golden must use exactly what the device sees.
    a_cpu = (torch.randn(m, total_k, dtype=torch.float32) * 0.2).to(input_dtype)
    w_cpu = (torch.randn(total_k, n, dtype=torch.float32) * 0.2).to(input_dtype)
    y_cpu = (torch.randn(len(lengths), m, n, dtype=torch.float32) * 0.2).to(
        output_dtype
    )
    x_storage = a_cpu.t().contiguous() if transpose_x else a_cpu.contiguous()
    values = list(lengths)
    if list_type == 0:
        values = list(itertools.accumulate(values))
    if noncontiguous_inputs:
        # Transpose the backing allocations so the ACL views have genuinely
        # non-contiguous strides while preserving the intended matrix values.
        x_backing = x_storage.t().contiguous().to(device)
        w_backing = w_cpu.t().contiguous().to(device)
        x, w = x_backing.t(), w_backing.t()
        if x.is_contiguous() or w.is_contiguous():
            raise AssertionError("expected non-contiguous x and weight views")
        storage_tensors = (x_backing, w_backing, None, None)
    else:
        x, w = x_storage.to(device), w_cpu.to(device)
        storage_tensors = None
    if noncontiguous_groups:
        groups_backing = torch.zeros(
            2 * len(values) + 1, dtype=torch.int64, device=device
        )
        groups = groups_backing[1 : 1 + 2 * len(values) : 2]
        groups.copy_(torch.tensor(values, dtype=torch.int64, device=device))
        current_storage = storage_tensors or (None, None, None, None)
        storage_tensors = (
            current_storage[0],
            current_storage[1],
            groups_backing,
            current_storage[3],
        )
    else:
        groups = torch.tensor(values, dtype=torch.int64, device=device)
    y = y_cpu.to(device)
    invoke(*api, x, w, groups, y, transpose_x, list_type, storage_tensors)
    actual = y.cpu()

    start = 0
    stats = []
    for group_idx, group_k in enumerate(lengths):
        end = start + group_k
        if group_k == 0:
            if not torch.equal(actual[group_idx], y_cpu[group_idx]):
                raise AssertionError(
                    f"group {group_idx}: zero-length group changed the original yRef"
                )
        else:
            # FP32 Matmul and FP32 add, followed by one cast to the output dtype.
            expected = (
                a_cpu[:, start:end].float() @ w_cpu[start:end, :].float()
                + y_cpu[group_idx].float()
            ).to(output_dtype)
            stats.append(
                check_group(actual[group_idx], expected, output_dtype, group_idx)
            )
        start = end
    ratio = min(item[0] for item in stats)
    max_error = max(item[1] for item in stats)
    print(
        f"PASS input={input_dtype} output={output_dtype} transposeX={transpose_x} "
        f"groupListType={list_type} M={m} N={n} groups={len(lengths)} "
        f"noncontiguousInputs={noncontiguous_inputs} noncontiguousGroups={noncontiguous_groups} "
        f"min_group_match={ratio:.6f} max_abs={max_error:.8g}",
        flush=True,
    )


def expect_param_error(
    api, label, tensors, transpose_x=True, list_type=1, storage_tensors=None
):
    nnop, get_workspace, _ = api
    handles = []
    storage_tensors = storage_tensors or (None, None, None, None)
    try:
        for tensor, storage in zip(tensors, storage_tensors):
            handles.append(make_acl_tensor(nnop, tensor, storage))
        workspace_size, executor = ctypes.c_uint64(), ctypes.c_void_p()
        status = get_workspace(
            *handles,
            transpose_x,
            False,
            2,
            list_type,
            ctypes.byref(workspace_size),
            ctypes.byref(executor),
        )
        if status != ACLNN_ERR_PARAM_INVALID:
            raise AssertionError(
                f"{label}: expected {ACLNN_ERR_PARAM_INVALID}, got {status}"
            )
        print(f"PASS negative {label}: status={status}", flush=True)
    finally:
        for handle in handles:
            nnop.aclDestroyTensor(handle)


def negative_cases(api, device):
    groups = torch.tensor([13, 19], dtype=torch.int64, device=device)
    x = torch.randn(32, 17, dtype=torch.float16, device=device)
    weight = torch.randn(32, 32, dtype=torch.float16, device=device)

    weight_unaligned = torch.randn(32, 33, dtype=torch.float16, device=device)
    y_unaligned = torch.randn(2, 17, 33, dtype=torch.float16, device=device)
    expect_param_error(
        api, "16-bit output with N=33", (x, weight_unaligned, groups, y_unaligned)
    )

    weight_wrong_dtype = weight.to(torch.bfloat16)
    y_float = torch.randn(2, 17, 32, dtype=torch.float32, device=device)
    expect_param_error(
        api, "different x and weight dtype", (x, weight_wrong_dtype, groups, y_float)
    )
    y_wrong_dtype = y_float.to(torch.bfloat16)
    expect_param_error(
        api, "16-bit yRef dtype differs from x", (x, weight, groups, y_wrong_dtype)
    )
    groups_wrong_dtype = groups.to(torch.int32)
    expect_param_error(
        api, "groupList dtype is int32", (x, weight, groups_wrong_dtype, y_float)
    )

    y_backing = torch.randn(2, 32, 17, dtype=torch.float16, device=device)
    y_view = y_backing.transpose(1, 2)
    if y_view.is_contiguous():
        raise AssertionError(
            "non-contiguous yRef test unexpectedly constructed a contiguous view"
        )
    expect_param_error(
        api,
        "non-contiguous 16-bit yRef",
        (x, weight, groups, y_view),
        storage_tensors=(None, None, None, y_backing),
    )

    y_backing = torch.randn(2 * 17 * 32 + 1, dtype=torch.float16, device=device)
    y_view = y_backing[1:].view(2, 17, 32)
    if not y_view.is_contiguous() or y_view.storage_offset() != 1:
        raise AssertionError(
            "offset yRef test did not construct the intended contiguous offset view"
        )
    expect_param_error(
        api,
        "unaligned 16-bit yRef offset",
        (x, weight, groups, y_view),
        storage_tensors=(None, None, None, y_backing),
    )


def metadata_limit_cases(api, device):
    """Check rejection before launch; tiny backing buffers must never be accessed."""
    nnop, get_workspace, _ = api
    backing = torch.empty(64, dtype=torch.uint8, device=device)

    def descriptor(shape, dtype):
        strides = []
        stride = 1
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        dims = (ctypes.c_int64 * len(shape))(*shape)
        stride_array = (ctypes.c_int64 * len(shape))(*strides)
        # This descriptor deliberately claims the logical storage size; only
        # GetWorkspaceSize is called and every case must be rejected there.
        handle = nnop.aclCreateTensor(
            dims,
            len(shape),
            DTYPE[dtype],
            stride_array,
            0,
            FORMAT_ND,
            dims,
            len(shape),
            ctypes.c_void_p(backing.data_ptr()),
        )
        if not handle:
            raise RuntimeError(f"aclCreateTensor rejected metadata shape {shape}")
        return handle

    max_i32 = 2**31 - 1
    cases = [
        (
            "non-transposed x K=65536",
            ((1, 65536), (65536, 16), (1,), (1, 1, 16)),
            False,
            torch.float32,
        ),
        (
            "transposed x M=65536",
            ((16, 65536), (16, 16), (1,), (1, 65536, 16)),
            True,
            torch.float16,
        ),
        (
            "weight N=65536",
            ((1, 16), (16, 65536), (1,), (1, 1, 65536)),
            False,
            torch.float32,
        ),
        (
            "M exceeds int32",
            ((max_i32 + 1, 16), (16, 16), (1,), (1, max_i32 + 1, 16)),
            False,
            torch.float32,
        ),
        (
            "K exceeds int32",
            ((max_i32 + 1, 16), (max_i32 + 1, 16), (1,), (1, 16, 16)),
            True,
            torch.float16,
        ),
        (
            "group count exceeds int32",
            ((1, 16), (16, 16), (max_i32 + 1,), (max_i32 + 1, 1, 16)),
            False,
            torch.float32,
        ),
        (
            "M*N*groups exceeds int64",
            ((max_i32, 16), (16, 65520), (max_i32,), (1,)),
            False,
            torch.float32,
        ),
    ]
    for label, shapes, transpose_x, output_dtype in cases:
        handles = []
        try:
            for shape, dtype in zip(
                shapes, (torch.float16, torch.float16, torch.int64, output_dtype)
            ):
                handles.append(descriptor(shape, dtype))
            workspace_size, executor = ctypes.c_uint64(), ctypes.c_void_p()
            status = get_workspace(
                *handles,
                transpose_x,
                False,
                2,
                1,
                ctypes.byref(workspace_size),
                ctypes.byref(executor),
            )
            if status != ACLNN_ERR_PARAM_INVALID:
                raise AssertionError(
                    f"{label}: expected {ACLNN_ERR_PARAM_INVALID}, got {status}; do not launch"
                )
            print(f"PASS metadata limit {label}: status={status}", flush=True)
        finally:
            for handle in handles:
                nnop.aclDestroyTensor(handle)
    return len(cases)


def all_zero_groups_case(
    api, device, input_dtype, output_dtype, transpose_x, list_type
):
    # Empty groups may still refer to a nonempty X/weight; every y group must be preserved.
    torch.manual_seed(20261121)
    m, n, total_k, group_count = 17, 32, 64, 3
    shape = (total_k, m) if transpose_x else (m, total_k)
    x = torch.randn(shape, dtype=input_dtype, device=device)
    w = torch.randn(total_k, n, dtype=input_dtype, device=device)
    groups = torch.zeros(group_count, dtype=torch.int64, device=device)
    y = torch.randn(group_count, m, n, dtype=output_dtype, device=device)
    before = y.cpu().clone()
    invoke(*api, x, w, groups, y, transpose_x, list_type)
    if not torch.equal(y.cpu(), before):
        raise AssertionError("all-zero groups changed yRef")
    print(
        f"PASS all-zero groups input={input_dtype} output={output_dtype} "
        f"transposeX={transpose_x} groupListType={list_type}",
        flush=True,
    )


def empty_k_case(api, device, input_dtype, output_dtype, transpose_x):
    m, n = 17, 32
    x = torch.empty((0, m) if transpose_x else (m, 0), dtype=input_dtype, device=device)
    weight = torch.empty((0, n), dtype=input_dtype, device=device)
    groups = torch.zeros(2, dtype=torch.int64, device=device)
    y = torch.randn(2, m, n, dtype=output_dtype, device=device)
    before = y.cpu().clone()
    invoke(*api, x, weight, groups, y, transpose_x, 1)
    if not torch.equal(y.cpu(), before):
        raise AssertionError("empty K changed yRef")
    print(
        f"PASS empty K input={input_dtype} output={output_dtype} transposeX={transpose_x}",
        flush=True,
    )


def empty_output_axis_case(
    api, device, input_dtype, output_dtype, transpose_x, empty_m
):
    m, n, k = (0, 32, 8) if empty_m else (17, 0, 8)
    x = torch.empty((k, m) if transpose_x else (m, k), dtype=input_dtype, device=device)
    weight = torch.empty((k, n), dtype=input_dtype, device=device)
    groups = torch.tensor([3, 8], dtype=torch.int64, device=device)
    y = torch.empty((2, m, n), dtype=output_dtype, device=device)
    invoke(*api, x, weight, groups, y, transpose_x, 0)
    print(
        f"PASS empty {'M' if empty_m else 'N'} input={input_dtype} "
        f"output={output_dtype} transposeX={transpose_x}",
        flush=True,
    )


def empty_group_list_case(api, device, input_dtype, output_dtype, transpose_x):
    m, n, k = 17, 32, 8
    x = torch.randn((k, m) if transpose_x else (m, k), dtype=input_dtype, device=device)
    weight = torch.randn((k, n), dtype=input_dtype, device=device)
    groups = torch.empty((0,), dtype=torch.int64, device=device)
    y = torch.empty((0, m, n), dtype=output_dtype, device=device)
    invoke(*api, x, weight, groups, y, transpose_x, 0)
    print(
        f"PASS empty group list input={input_dtype} output={output_dtype} transposeX={transpose_x}",
        flush=True,
    )


def aligned_output_offset_case(api, device, input_dtype, transpose_x, m=17, n=32):
    torch.manual_seed(20261122)
    total_k = 32
    a_cpu = (torch.randn(m, total_k) * 0.2).to(input_dtype)
    w_cpu = (torch.randn(total_k, n) * 0.2).to(input_dtype)
    x_cpu = a_cpu.t().contiguous() if transpose_x else a_cpu
    x, w = x_cpu.to(device), w_cpu.to(device)
    groups = torch.tensor([13, 19], dtype=torch.int64, device=device)
    output_elements = 2 * m * n
    y_backing = torch.randn(16 + output_elements + 16, dtype=input_dtype, device=device)
    prefix = y_backing[:16].cpu().clone()
    suffix = y_backing[16 + output_elements :].cpu().clone()
    y = y_backing[16 : 16 + output_elements].view(2, m, n)
    old = y.cpu().clone()
    invoke(*api, x, w, groups, y, transpose_x, 1, (None, None, None, y_backing))
    actual = y.cpu()
    if not torch.equal(y_backing[:16].cpu(), prefix) or not torch.equal(
        y_backing[16 + output_elements :].cpu(), suffix
    ):
        raise AssertionError(
            "aligned yRef offset case modified the backing allocation outside the view"
        )
    for group_idx, (start, end) in enumerate(((0, 13), (13, 32))):
        expected = (
            a_cpu[:, start:end].float() @ w_cpu[start:end, :].float()
            + old[group_idx].float()
        ).to(input_dtype)
        check_group(actual[group_idx], expected, input_dtype, group_idx)
    print(
        f"PASS aligned yRef offset input={input_dtype} transposeX={transpose_x} M={m} N={n}",
        flush=True,
    )


def shape_sweep(api, device, large=False):
    """Exercise tile boundaries or near-limit dimensions across supported types."""
    if large:
        shapes = [(1, 16, (16384, 49151)), (17, 65520, (1, 3)), (65535, 16, (1, 3))]
    else:
        shapes = [
            (1, 16, (1,)),
            (15, 48, (15, 0, 1)),
            (16, 112, (16, 17)),
            (17, 128, (1, 31, 0)),
            (127, 144, (17, 65)),
            (128, 240, (33, 1, 17)),
            (129, 256, (127, 0, 2)),
            (257, 272, (129, 7, 31)),
            (33, 512, (23, 0, 101)),
        ]
    cases = 0
    for (
        m,
        n,
        lengths,
    ), input_dtype, same_type, transpose_x, list_type in itertools.product(
        shapes, (torch.float16, torch.bfloat16), (False, True), (False, True), (0, 1)
    ):
        output_dtype = input_dtype if same_type else torch.float32
        run_case(
            api,
            device,
            input_dtype,
            output_dtype,
            transpose_x,
            list_type,
            m=m,
            n=n,
            lengths=lengths,
        )
        cases += 1
    print(
        f"PASS: {cases} {'large dimension' if large else 'shape'} sweep cases",
        flush=True,
    )


def random_sweep(api, device):
    """Run reproducible mixed group layouts and dimensions, including FP32 N tails."""
    rng = random.Random(20261120)
    cases = 0
    for input_dtype, same_type, transpose_x, list_type in itertools.product(
        (torch.float16, torch.bfloat16), (False, True), (False, True), (0, 1)
    ):
        output_dtype = input_dtype if same_type else torch.float32
        for _ in range(8):
            m = rng.randint(1, 300)
            n = 16 * rng.randint(1, 19) if same_type else rng.randint(1, 300)
            lengths = [rng.randint(0, 32) for _ in range(rng.randint(1, 8))]
            if not any(lengths):
                lengths[0] = 1
            run_case(
                api,
                device,
                input_dtype,
                output_dtype,
                transpose_x,
                list_type,
                m=m,
                n=n,
                lengths=tuple(lengths),
            )
            cases += 1
    print(f"PASS: {cases} random sweep cases", flush=True)


def group_sweep(api, device):
    """Cover the documented group-count maximum, including many empty groups."""
    lengths = tuple(idx // 257 + 1 if idx % 257 == 0 else 0 for idx in range(1024))
    cases = 0
    for input_dtype, same_type, transpose_x, list_type in itertools.product(
        (torch.float16, torch.bfloat16), (False, True), (False, True), (0, 1)
    ):
        output_dtype = input_dtype if same_type else torch.float32
        run_case(
            api,
            device,
            input_dtype,
            output_dtype,
            transpose_x,
            list_type,
            m=3,
            n=272,
            lengths=lengths,
        )
        cases += 1
    print(f"PASS: {cases} maximum-group cases", flush=True)


def offset_sweep(api, device):
    """Verify a nonzero output offset on small and multi-tile shapes."""
    cases = 0
    for input_dtype, transpose_x, (m, n) in itertools.product(
        (torch.float16, torch.bfloat16), (False, True), ((17, 32), (273, 272))
    ):
        aligned_output_offset_case(api, device, input_dtype, transpose_x, m=m, n=n)
        cases += 1
    print(f"PASS: {cases} aligned output offset cases", flush=True)


def chunk_boundary_sweep(api, device):
    """Cover both sides of the low-precision row-chunk selection and tile tails."""
    cases = 0
    for m, n, input_dtype, transpose_x in itertools.product(
        (31, 32, 33, 159, 160, 161),
        (32, 272),
        (torch.float16, torch.bfloat16),
        (False, True),
    ):
        run_case(
            api,
            device,
            input_dtype,
            input_dtype,
            transpose_x,
            1,
            m=m,
            n=n,
            lengths=(17, 0, 23),
        )
        cases += 1
    print(f"PASS: {cases} low-precision chunk boundary cases", flush=True)


def rounding_midpoint_cases(api, device):
    """Check exact ties and repeated writeback for both 16-bit output types."""
    m, k, n = 17, 64, 32
    groups = torch.tensor([k], dtype=torch.int64, device=device)
    cases = 0
    for dtype, transpose_x in itertools.product(
        (torch.float16, torch.bfloat16), (False, True)
    ):
        half_ulp = 2.0 ** (-11 if dtype == torch.float16 else -8)
        a = torch.zeros(m, k, dtype=dtype)
        a[:, 0] = 1
        w = torch.zeros(k, n, dtype=dtype)
        w[0] = torch.tensor([half_ulp, -half_ulp], dtype=dtype).repeat(n // 2)

        # The first pair starts at an even significand, the second at an odd one.
        # The accumulation lands exactly halfway between adjacent values in every column.
        initial_row = torch.tensor(
            [1, -1, 1 + 2 * half_ulp, -1 - 2 * half_ulp], dtype=dtype
        )
        expected = initial_row.repeat(m, n // 4).reshape(1, m, n)
        delta = (a.float() @ w.float()).unsqueeze(0)
        x = (a.t().contiguous() if transpose_x else a).to(device)
        weight, y = w.to(device), expected.to(device)
        for step in range(4):
            expected = (expected.float() + delta).to(dtype)
            invoke(*api, x, weight, groups, y, transpose_x, 0)
            actual = y.cpu()
            if not torch.equal(actual, expected):
                raise AssertionError(
                    f"rounding midpoint mismatch: dtype={dtype}, transposeX={transpose_x}, "
                    f"step={step + 1}, actual[:4]={actual[0, 0, :4].tolist()}, "
                    f"expected[:4]={expected[0, 0, :4].tolist()}"
                )
        cases += 1
    return cases


def legacy_v1_suite(opapi_path, api, device):
    """Exercise the original ACLNN entry point, whose group list is cumulative."""
    path = Path(opapi_path).resolve(strict=True)
    opapi = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    ptr, i64, u64 = ctypes.c_void_p, ctypes.c_int64, ctypes.c_uint64
    get_workspace_v1 = opapi.aclnnGroupedMatmulAddGetWorkspaceSize
    run_v1 = opapi.aclnnGroupedMatmulAdd
    assert_symbol_in_library(get_workspace_v1, path)
    assert_symbol_in_library(run_v1, path)
    get_workspace_v1.argtypes = [
        ptr,
        ptr,
        ptr,
        ptr,
        ctypes.c_bool,
        ctypes.c_bool,
        i64,
        ctypes.POINTER(u64),
        ctypes.POINTER(ptr),
    ]
    get_workspace_v1.restype = ctypes.c_int
    run_v1.argtypes = [ptr, u64, ptr, ptr]
    run_v1.restype = ctypes.c_int

    def get_workspace(
        x,
        weight,
        groups,
        y,
        transpose_x,
        transpose_w,
        group_type,
        list_type,
        size,
        executor,
    ):
        if list_type != 0:
            raise ValueError("the legacy API uses cumulative group lists")
        return get_workspace_v1(
            x, weight, groups, y, transpose_x, transpose_w, group_type, size, executor
        )

    legacy_api = (api[0], get_workspace, run_v1)
    cases = 0
    for input_dtype, same_type, transpose_x in itertools.product(
        (torch.float16, torch.bfloat16), (False, True), (False, True)
    ):
        output_dtype = input_dtype if same_type else torch.float32
        run_case(legacy_api, device, input_dtype, output_dtype, transpose_x, 0)
        cases += 1
    print(f"PASS: {cases} legacy ACLNN entry point cases", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--opapi", required=True, help="absolute path to custom vendor libcust_opapi.so"
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device index")
    parser.add_argument(
        "--metadata-limits-only",
        action="store_true",
        help="run only metadata limit rejection cases",
    )
    parser.add_argument(
        "--shape-sweep", action="store_true", help="run the additional shape sweep"
    )
    parser.add_argument(
        "--large-dim-sweep", action="store_true", help="run near-limit dimension cases"
    )
    parser.add_argument(
        "--random-sweep",
        action="store_true",
        help="run deterministic mixed-shape cases",
    )
    parser.add_argument(
        "--group-sweep", action="store_true", help="run maximum-group-count cases"
    )
    parser.add_argument(
        "--offset-sweep", action="store_true", help="run aligned output offset cases"
    )
    parser.add_argument(
        "--chunk-boundary-sweep",
        action="store_true",
        help="run low-precision chunk boundary cases",
    )
    parser.add_argument(
        "--rounding-midpoints-only",
        action="store_true",
        help="run exact midpoint cases",
    )
    parser.add_argument(
        "--legacy-v1",
        action="store_true",
        help="test the original cumulative-list ACLNN API",
    )
    args = parser.parse_args()
    device = f"npu:{args.device}"
    torch.npu.set_device(device)
    api = open_api(args.opapi)
    if args.metadata_limits_only:
        print(
            f"PASS: {metadata_limit_cases(api, device)} metadata limit rejection cases",
            flush=True,
        )
        return
    if args.shape_sweep:
        shape_sweep(api, device)
        return
    if args.large_dim_sweep:
        shape_sweep(api, device, large=True)
        return
    if args.random_sweep:
        random_sweep(api, device)
        return
    if args.group_sweep:
        group_sweep(api, device)
        return
    if args.offset_sweep:
        offset_sweep(api, device)
        return
    if args.chunk_boundary_sweep:
        chunk_boundary_sweep(api, device)
        return
    if args.rounding_midpoints_only:
        print(
            f"PASS: {rounding_midpoint_cases(api, device)} exact midpoint rounding cases",
            flush=True,
        )
        return
    if args.legacy_v1:
        legacy_v1_suite(args.opapi, api, device)
        return
    cases = 0
    for input_dtype, same_type, transpose_x, list_type in itertools.product(
        (torch.float16, torch.bfloat16), (False, True), (False, True), (0, 1)
    ):
        output_dtype = input_dtype if same_type else torch.float32
        run_case(api, device, input_dtype, output_dtype, transpose_x, list_type)
        cases += 1
    # FP32 remains supported for unaligned N and non-contiguous input views.
    for input_dtype, transpose_x in itertools.product(
        (torch.float16, torch.bfloat16), (True, False)
    ):
        run_case(
            api,
            device,
            input_dtype,
            torch.float32,
            transpose_x,
            1,
            n=257,
            noncontiguous_inputs=True,
        )
        cases += 1
    # Low-precision output on a small shape, for both input dtypes and transpose modes.
    for input_dtype, transpose_x in itertools.product(
        (torch.float16, torch.bfloat16), (True, False)
    ):
        run_case(api, device, input_dtype, input_dtype, transpose_x, 1, m=17, n=32)
        cases += 1
    for input_dtype, transpose_x in itertools.product(
        (torch.float16, torch.bfloat16), (True, False)
    ):
        run_case(
            api,
            device,
            input_dtype,
            input_dtype,
            transpose_x,
            1,
            m=17,
            n=32,
            noncontiguous_inputs=True,
        )
        cases += 1
        aligned_output_offset_case(api, device, input_dtype, transpose_x)
        cases += 1
        aligned_output_offset_case(api, device, input_dtype, transpose_x, m=273, n=272)
        cases += 1
    for input_dtype, same_type, transpose_x, list_type in itertools.product(
        (torch.float16, torch.bfloat16), (False, True), (False, True), (0, 1)
    ):
        output_dtype = input_dtype if same_type else torch.float32
        run_case(
            api,
            device,
            input_dtype,
            output_dtype,
            transpose_x,
            list_type,
            m=17,
            n=32,
            noncontiguous_groups=True,
        )
        cases += 1
    for input_dtype, same_type, transpose_x, list_type in itertools.product(
        (torch.float16, torch.bfloat16), (False, True), (False, True), (0, 1)
    ):
        output_dtype = input_dtype if same_type else torch.float32
        all_zero_groups_case(
            api, device, input_dtype, output_dtype, transpose_x, list_type
        )
        cases += 1
    for input_dtype, same_type, transpose_x in itertools.product(
        (torch.float16, torch.bfloat16), (False, True), (False, True)
    ):
        output_dtype = input_dtype if same_type else torch.float32
        empty_k_case(api, device, input_dtype, output_dtype, transpose_x)
        cases += 1
        empty_group_list_case(api, device, input_dtype, output_dtype, transpose_x)
        cases += 1
        for empty_m in (False, True):
            empty_output_axis_case(
                api, device, input_dtype, output_dtype, transpose_x, empty_m
            )
            cases += 1
    # More tasks than AIV logical slots: exercise the group carry and per-core loop.
    many_groups = tuple(0 if idx % 11 == 0 else idx % 4 + 1 for idx in range(37))
    for input_dtype, same_type, transpose_x, list_type in itertools.product(
        (torch.float16, torch.bfloat16), (False, True), (False, True), (0, 1)
    ):
        output_dtype = input_dtype if same_type else torch.float32
        run_case(
            api,
            device,
            input_dtype,
            output_dtype,
            transpose_x,
            list_type,
            m=65,
            n=272,
            lengths=many_groups,
        )
        cases += 1
    cases += rounding_midpoint_cases(api, device)
    negative_cases(api, device)
    limit_count = metadata_limit_cases(api, device)
    print(
        f"PASS: {cases} positive cases, 6 negative cases and {limit_count} metadata limit cases",
        flush=True,
    )


if __name__ == "__main__":
    main()
