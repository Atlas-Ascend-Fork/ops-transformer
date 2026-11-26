# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

try:
    import torch
    import torchair
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge._ge_graph import (
        DataType,
        _ge_dtype_to_ge_proto_dtype,
        auto_convert_to_tensor,
    )
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:
    _HIFLOAT4_GE_DTYPE = getattr(DataType, "DT_HIFLOAT4", None)
    _HIFLOAT4_SCALE_GE_DTYPE = getattr(DataType, "DT_HIFLOAT4_SCALE", None)
    _HIFLOAT4_AVAILABLE = (
        _HIFLOAT4_GE_DTYPE is not None and _HIFLOAT4_SCALE_GE_DTYPE is not None
    )

    @auto_convert_to_tensor([False, False, False, False], [False, False, False, True])
    def PartialRotaryMulQuant(
        x,
        cos,
        sin,
        scale=None,
        *,
        y_dtype=0,
        quant_mode=0,
        round_mode="round",
        axis=-1,
        rotary_mode=0,
        partial_slice=None,
    ):
        return ge_op(
            op_type="PartialRotaryMulQuant",
            inputs={"x": x, "cos": cos, "sin": sin, "scale": scale},
            attrs={
                "y_dtype": attr.Int(y_dtype),
                "quant_mode": attr.Int(quant_mode),
                "round_mode": attr.Str(round_mode),
                "axis": attr.Int(axis),
                "rotary_mode": attr.Int(rotary_mode),
                # An empty list lets GE tiling use the runtime D for full HiFloat4 RoPE.
                "partial_slice": attr.ListInt(
                    partial_slice or ([] if y_dtype == 1 else [0, 0])
                ),
            },
            outputs=["y", "y_scale"],
            ir=IrDef("PartialRotaryMulQuant")
            .input("x", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16")
            .input("cos", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16")
            .input("sin", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16")
            .optional_input("scale", "DT_FLOAT")
            .attr("y_dtype", attr.Int(0))
            .attr("quant_mode", attr.Int(0))
            .attr("round_mode", attr.Str("round"))
            .attr("axis", attr.Int(-1))
            .attr("rotary_mode", attr.Int(0))
            .attr("partial_slice", attr.ListInt([0, 0]))
            .output(
                "y",
                "DT_HIFLOAT8, DT_HIFLOAT4" if _HIFLOAT4_AVAILABLE else "DT_HIFLOAT8",
            )
            .output(
                "y_scale",
                "DT_FLOAT, DT_HIFLOAT4_SCALE" if _HIFLOAT4_AVAILABLE else "DT_FLOAT",
            ),
        )

    def _pack_hifloat4_to_uint8(y):
        # Group two logical values along D before reinterpreting them as one byte.
        packed_shape = ge.Div(
            ge.Shape(y), ge.Const([1, 1, 1, 2], dtype=DataType.DT_INT32)
        )
        pair_shape = ge.ConcatV2(
            [packed_shape, ge.Const([2], dtype=DataType.DT_INT32)], concat_dim=0, N=2
        )
        packed = ge.Bitcast(ge.Reshape(y, pair_shape), type=DataType.DT_UINT8)
        packed = ge.Reshape(packed, packed_shape)
        packed.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_UINT8)
        return packed

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.partial_rotary_mul_quant.default
    )
    def convert_partial_rotary_mul_quant(
        x,
        cos,
        sin,
        scale=None,
        y_dtype=0,
        quant_mode=0,
        round_mode="round",
        axis=-1,
        rotary_mode=0,
        partial_slice=None,
        meta_outputs=None,
    ):
        if y_dtype not in (0, 1):
            raise ValueError("y_dtype must be 0 (HiFloat8) or 1 (HiFloat4).")
        if y_dtype == 1 and not _HIFLOAT4_AVAILABLE:
            raise RuntimeError(
                "HiFloat4 graph mode requires TorchAir with DT_HIFLOAT4 and DT_HIFLOAT4_SCALE support."
            )
        if partial_slice is not None and len(partial_slice) not in (0, 2):
            raise ValueError("partial_slice must be empty or contain [start, end].")
        y, y_scale = PartialRotaryMulQuant(
            x,
            cos,
            sin,
            scale,
            y_dtype=y_dtype,
            quant_mode=quant_mode,
            round_mode=round_mode,
            axis=axis,
            rotary_mode=rotary_mode,
            partial_slice=partial_slice,
        )
        if y_dtype == 1:
            y.desc.dtype = _ge_dtype_to_ge_proto_dtype(_HIFLOAT4_GE_DTYPE)
            y_scale.desc.dtype = _ge_dtype_to_ge_proto_dtype(_HIFLOAT4_SCALE_GE_DTYPE)
            # Both scale types use 32-bit storage with the same shape.
            y_scale = ge.Bitcast(y_scale, type=DataType.DT_FLOAT)
            y_scale.desc.dtype = _ge_dtype_to_ge_proto_dtype(DataType.DT_FLOAT)
            return _pack_hifloat4_to_uint8(y), y_scale

        # Expose the byte storage expected by PyTorch without numeric conversion.
        y.desc.dtype = DataType.DT_HIFLOAT8
        y_scale.desc.dtype = DataType.DT_FLOAT
        return ge.Bitcast(y, type=DataType.DT_UINT8), y_scale
else:

    def convert_partial_rotary_mul_quant(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
