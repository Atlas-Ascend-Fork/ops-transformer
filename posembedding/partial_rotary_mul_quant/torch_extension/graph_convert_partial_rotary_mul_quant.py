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
    from torchair.ge._ge_graph import DataType, auto_convert_to_tensor
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

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
                "partial_slice": attr.ListInt(partial_slice),
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
            .output("y", "DT_HIFLOAT8")
            .output("y_scale", "DT_FLOAT"),
        )

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
        if y_dtype != 0:
            raise ValueError("Ascend950 only supports y_dtype=0 (HiFloat8).")
        if partial_slice is None or len(partial_slice) != 2:
            raise ValueError("partial_slice must contain [start, end].")
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
        # Expose the byte storage expected by PyTorch without numeric conversion.
        y.desc.dtype = DataType.DT_HIFLOAT8
        y_scale.desc.dtype = DataType.DT_FLOAT
        return ge.Bitcast(y, type=DataType.DT_UINT8), y_scale
else:

    def convert_partial_rotary_mul_quant(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
