#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU Golden validation for eager, TorchAir GE and ACL graph on Ascend950.

Requires torch, matching torch_npu/torchair, numpy and en_dtypes.
Import the installed extension via --package (default: cann_ops_transformer).
Example:
  python test_torch_partial_rotary_mul_quant.py --package cann_ops_transformer_rotary_quant \
      --mode all --device 1 --report results.json

ROUND: <=1 HiFloat8 encoding step, with at most 0.1% invalid elements, matching
the existing ACLNN Golden's requant criterion. HYBRID: deterministic TA region
uses the same criterion; stochastic region checks enclosing HiFloat8 values
(including exact-value carry). This verifies the rounding envelope, not the
statistical distribution of stochastic rounding. Every execution is compared
to CPU math, including the second execution after changing inputs and scale.
"""

import argparse
import importlib
import json
import os
from pathlib import Path

import numpy as np
from en_dtypes import hifloat8
import torch
import torch_npu


def cpu_reference(x, cos, sin, scale, rotary_mode, start):
    """FP32 arithmetic, with no NPU/custom operator calls in the Golden."""
    result = x.float().numpy().copy()
    c, s = cos.float().numpy(), sin.float().numpy()
    tail = result[..., start:].copy()
    if rotary_mode == 1:
        rotated = np.empty_like(tail)
        rotated[..., ::2], rotated[..., 1::2] = -tail[..., 1::2], tail[..., ::2]
        result[..., start:] = tail * c + rotated * s
    elif rotary_mode in (0, 3):
        half = tail.shape[-1] // 2
        a, b = (
            (tail[..., :half], tail[..., half:])
            if rotary_mode == 0
            else (tail[..., ::2], tail[..., 1::2])
        )
        result[..., start:] = np.concatenate(
            (
                a * c[..., :half] - b * s[..., :half],
                b * c[..., half:] + a * s[..., half:],
            ),
            axis=-1,
        )
    else:
        q = tail.shape[-1] // 4
        rotated = np.concatenate(
            (
                -tail[..., q : 2 * q],
                tail[..., :q],
                -tail[..., 3 * q :],
                tail[..., 2 * q : 3 * q],
            ),
            axis=-1,
        )
        result[..., start:] = tail * c + rotated * s
    if scale is not None:
        result *= np.float32(scale.item())
    return result


def verify(outputs, reference, round_mode):
    y, y_scale = outputs
    assert tuple(y.shape) == reference.shape and y.dtype == torch.uint8
    assert tuple(y_scale.shape) == (0,) and y_scale.dtype == torch.float32
    actual = y.cpu().numpy()
    expected = reference.astype(hifloat8)
    delta = np.abs(
        actual.view(np.int8).astype(np.int16) - expected.view(np.int8).astype(np.int16)
    )
    valid = delta <= 1
    decoded = actual.view(hifloat8).astype(np.float32)
    if round_mode == "hybrid":
        magnitude = np.abs(reference)
        stochastic = (magnitude > 0) & ((magnitude < 0.125) | (magnitude >= 16))
        values = np.arange(256, dtype=np.uint8).view(hifloat8).astype(np.float32)
        values = np.sort(values[np.isfinite(values)])
        source = reference[stochastic]
        lower = np.searchsorted(values, source, side="right") - 1
        upper = np.searchsorted(values, source, side="left")
        lo = values[np.clip(lower, 0, len(values) - 1)]
        hi = values[np.clip(upper, 0, len(values) - 1)]
        exact = lo == hi
        positive, negative = exact & (source > 0), exact & (source < 0)
        hi[positive] = values[np.clip(upper[positive] + 1, 0, len(values) - 1)]
        lo[negative] = values[np.clip(lower[negative] - 1, 0, len(values) - 1)]
        valid[stochastic] = (decoded[stochastic] == lo) | (decoded[stochastic] == hi)
    valid &= np.isfinite(decoded)
    valid[reference == 0] = decoded[reference == 0] == 0
    invalid = int(valid.size - np.count_nonzero(valid))
    assert invalid / valid.size <= 0.001, (
        f"{invalid}/{valid.size} elements outside HiFloat8 rounding tolerance; "
        f"first invalid coordinates: {np.argwhere(~valid)[:8].tolist()}"
    )
    return {
        "elements": valid.size,
        "invalid": invalid,
        "byte_equal_fraction": float(np.mean(actual == expected.view(np.uint8))),
        "max_abs_error": float(np.max(np.abs(decoded - reference))),
    }


def cases():
    # Include all supported broadcast patterns, unaligned HiF8 slice starts,
    # full RoPE, D=1024, short tail, and strided x/cos/sin.
    patterns = [
        (2, 3, 4),
        (1, 1, 1),
        (2, 1, 1),
        (2, 3, 1),
        (1, 3, 1),
        (1, 1, 4),
        (2, 1, 4),
        (1, 3, 4),
    ]
    for dtype_index, dtype in enumerate((torch.float16, torch.bfloat16, torch.float32)):
        for rotary_mode in range(4):
            for round_index, round_mode in enumerate(("round", "hybrid")):
                index = dtype_index * 8 + rotary_mode * 2 + round_index
                d, start = [
                    (128, 0),
                    (128, 64),
                    (128, 60),
                    (1024, 960),
                    (80, 16),
                    (68, 64),
                ][index % 6]
                yield {
                    "dtype": dtype,
                    "rotary_mode": rotary_mode,
                    "round_mode": round_mode,
                    "shape": (2, 3, 4, d),
                    "start": start,
                    "cos_shape": (*patterns[index % 8], d - start),
                    "strided": index % 3 == 0,
                    "scale": index % 3 != 0,
                }


def make_inputs(case, seed, device, replay):
    generator = torch.Generator().manual_seed(seed)
    # Range includes underflow/TA/stochastic regions after the two scale values.
    inputs = [
        torch.randn(shape, generator=generator).mul_(4).to(case["dtype"])
        for shape in (case["shape"], case["cos_shape"], case["cos_shape"])
    ]
    # Explicit zeros also catch unwritten/incorrectly scaled prefixes.
    inputs[0][..., ::7] = 0
    scale_value = (0.0 if seed % 5 == 0 else -8.0) if replay else 0.03125
    scale = torch.tensor([scale_value], dtype=torch.float32) if case["scale"] else None
    reference = cpu_reference(*inputs, scale, case["rotary_mode"], case["start"])
    npu_inputs = []
    for value in inputs:
        if case["strided"]:
            backing = torch.empty(
                (*value.shape[:-1], value.shape[-1] * 2),
                dtype=value.dtype,
                device=device,
            )
            view = backing[..., ::2]
            view.copy_(value)
            npu_inputs.append(view)
        else:
            npu_inputs.append(value.to(device))
    npu_inputs.append(None if scale is None else scale.to(device))
    return npu_inputs, reference


def validate_rejections(device):
    """Check actual dispatcher and Meta reject HiF4 and malformed parameters."""
    count = 0
    for target in (device, "meta"):
        x = torch.empty((2, 3, 4, 128), dtype=torch.float16, device=target)
        c = torch.empty((1, 1, 1, 64), dtype=x.dtype, device=target)
        bad_options = [
            dict(y_dtype=1),
            dict(rotary_mode=4),
            dict(round_mode="invalid"),
            dict(partial_slice=[0]),
            dict(partial_slice=[64, 120]),
            dict(partial_slice=[63, 128]),
            dict(scale=torch.empty((2,), dtype=torch.float32, device=target)),
        ]
        for options in bad_options:
            kwargs = {"partial_slice": [64, 128], **options}
            try:
                torch.ops.cann_ops_transformer.partial_rotary_mul_quant(
                    x, c, c, **kwargs
                )
            except (ValueError, RuntimeError):
                count += 1
            else:
                raise AssertionError(
                    f"Invalid arguments accepted on {target}: {options}"
                )
    return count


def run(args, report):
    package = importlib.import_module(args.package)
    public_op = package.ops.partial_rotary_mul_quant
    device = f"npu:{args.device}"
    torch.npu.set_device(device)
    name = torch.npu.get_device_name(args.device)
    assert name.startswith("Ascend950"), f"This test requires Ascend950, got {name}"
    report.update(
        {
            "device": name,
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "cann": os.environ.get("ASCEND_HOME_PATH"),
            "package": args.package,
            "ge_dynamic": not args.ge_static,
        }
    )
    report["rejection_checks"] = validate_rejections(device)
    modes = ("eager", "ge", "aclgraph") if args.mode == "all" else (args.mode,)
    for mode in modes:
        for index, case in enumerate(cases()):
            if args.case_limit is not None and index >= args.case_limit:
                break

            def operation(x, cos, sin, scale):
                return public_op(
                    x,
                    cos,
                    sin,
                    scale,
                    round_mode=case["round_mode"],
                    rotary_mode=case["rotary_mode"],
                    partial_slice=[case["start"], case["shape"][-1]],
                )

            if mode == "eager":
                model = operation
            else:
                import torchair
                from torchair.configs.compiler_config import CompilerConfig

                # Separate graphs per case; both calls below reuse this compiled model.
                torch._dynamo.reset()
                config = CompilerConfig()
                config.mode = "max-autotune" if mode == "ge" else "reduce-overhead"
                # Match the reference Torch interface's GE dynamic-shape setup.
                # CANN 9.2 static specialization currently fails to compile the
                # quarter-mode tail-length=4 boundary (dynamic GE works).
                model = torch.compile(
                    operation,
                    backend=torchair.get_npu_backend(compiler_config=config),
                    fullgraph=True,
                    dynamic=(mode == "ge" and not args.ge_static),
                )
            for replay in range(2):
                inputs, reference = make_inputs(
                    case, args.seed + index * 2 + replay, device, replay
                )
                with torch.no_grad():
                    outputs = model(*inputs)
                    torch.npu.synchronize()
                    metrics = verify(outputs, reference, case["round_mode"])
                record = {
                    "mode": mode,
                    "case": index,
                    "replay": replay,
                    **{
                        key: str(value) if isinstance(value, torch.dtype) else value
                        for key, value in case.items()
                    },
                    **metrics,
                    "pass": True,
                }
                report["results"].append(record)
                print(
                    f"PASS {mode} case={index} replay={replay} {case['dtype']} "
                    f"rotary={case['rotary_mode']} {case['round_mode']} "
                    f"invalid={metrics['invalid']}/{metrics['elements']}",
                    flush=True,
                )
            # ACL graphs retain outputs; release before the next case.
            del outputs, model, inputs
    maps = Path("/proc/self/maps")
    if maps.exists():
        report["runtime_libraries"] = sorted(
            {
                line.split()[-1]
                for line in maps.read_text().splitlines()
                if any(
                    name in line
                    for name in ("libascendcl.so", "libopapi", "libcust_opapi")
                )
            }
        )
    print(
        f"PASS: {len(report['results'])} accuracy checks; "
        f"{report['rejection_checks']} argument rejection checks",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="cann_ops_transformer")
    parser.add_argument(
        "--mode", choices=("eager", "ge", "aclgraph", "all"), default="all"
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ge-static",
        action="store_true",
        help="Use static GE specialization instead of dynamic GE",
    )
    parser.add_argument(
        "--case-limit", type=int, help="Run only the first N cases (smoke test)"
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = {"results": [], "pass": False}
    try:
        run(args, report)
        report["pass"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
