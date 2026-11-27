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

"""Validate independent GmmAdd invocations on two NPU streams."""

import argparse
import ctypes
import itertools

import torch
import torch_npu  # noqa: F401

import test_standalone_npu as t


def run_case(api, device, input_dtype, output_dtype, transpose_x):
    nnop, get_workspace, run = api
    m, n, lengths = 129, 272, (37, 0, 23, 52)
    total_k = sum(lengths)
    streams = [torch.npu.Stream(device=device) for _ in range(2)]
    inputs = []
    for seed in (20261120, 20261121):
        torch.manual_seed(seed)
        a_cpu = (torch.randn(m, total_k) * 0.2).to(input_dtype)
        w_cpu = (torch.randn(total_k, n) * 0.2).to(input_dtype)
        y_cpu = (torch.randn(len(lengths), m, n) * 0.2).to(output_dtype)
        x = (a_cpu.t().contiguous() if transpose_x else a_cpu).to(device)
        w = w_cpu.to(device)
        groups = torch.tensor(lengths, dtype=torch.int64, device=device)
        y = y_cpu.to(device)
        inputs.append((x, w, groups, y, a_cpu, w_cpu, y_cpu))
    torch.npu.synchronize()
    pending = []
    try:
        for stream, (x, w, groups, y, _, _, _) in zip(streams, inputs):
            handles = [t.make_acl_tensor(nnop, tensor) for tensor in (x, w, groups, y)]
            pending.append((handles, None))
            size, executor = ctypes.c_uint64(), ctypes.c_void_p()
            status = get_workspace(
                *handles,
                transpose_x,
                False,
                2,
                1,
                ctypes.byref(size),
                ctypes.byref(executor),
            )
            if status != 0 or not executor.value:
                raise RuntimeError(f"GetWorkspaceSize failed: {status}")
            with torch.npu.stream(stream):
                workspace = torch.empty(
                    max(size.value, 1), dtype=torch.uint8, device=device
                )
                pending[-1] = (handles, workspace)
                status = run(
                    ctypes.c_void_p(workspace.data_ptr()),
                    size.value,
                    executor,
                    ctypes.c_void_p(stream.npu_stream),
                )
                if status != 0:
                    raise RuntimeError(f"launch failed: {status}")
        torch.npu.synchronize()
        for _, _, _, y, a_cpu, w_cpu, y_cpu in inputs:
            actual = y.cpu()
            start = 0
            for group_idx, group_k in enumerate(lengths):
                end = start + group_k
                if group_k:
                    expected = (
                        a_cpu[:, start:end].float() @ w_cpu[start:end].float()
                        + y_cpu[group_idx].float()
                    ).to(output_dtype)
                    t.check_group(actual[group_idx], expected, output_dtype, group_idx)
                elif not torch.equal(actual[group_idx], y_cpu[group_idx]):
                    raise AssertionError("empty group changed")
                start = end
    finally:
        torch.npu.synchronize()
        for handles, _ in pending:
            for handle in handles:
                nnop.aclDestroyTensor(handle)
    print(
        f"PASS two streams input={input_dtype} output={output_dtype} transposeX={transpose_x}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opapi", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    device = f"npu:{args.device}"
    torch.npu.set_device(device)
    api = t.open_api(args.opapi)
    for dtype, same_type, transpose_x in itertools.product(
        (torch.float16, torch.bfloat16), (False, True), (False, True)
    ):
        run_case(api, device, dtype, dtype if same_type else torch.float32, transpose_x)
    print("PASS: 8 two-stream cases", flush=True)


if __name__ == "__main__":
    main()
