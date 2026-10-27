# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TorchAir GE conversion for the FfnWorkerBatching ACLNN operator."""

from typing import List, Optional

try:
    import torch
    from torchair.ge import attr
    from torchair.ge._ge_graph import (
        Tensor,
        TensorSpec,
        auto_convert_to_tensor,
        DataType,
    )
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef, is_cann_compat
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

_FIRST_MX_TOKEN_DTYPE = 3
_FP4_TOKEN_DTYPE = 5
_OUTPUT_NAMES = (
    "y",
    "group_list",
    "session_ids",
    "micro_batch_ids",
    "token_ids",
    "expert_offsets",
    "dynamic_scale",
    "actual_token_num",
)


def _pack_fp4_to_uint8(y):
    """Pack a logical-nibble FP4 GE tensor (..., H) into packed uint8 (..., H/2).

    Mirrors the official npu_add_rms_norm_dynamic_mx_quant converter: reshape
    to (..., H/2, 2) nibble pairs, Bitcast to uint8 bytes, reshape to the
    packed shape. The byte layout matches torch's float4_e2m1fn_x2 storage,
    so eager (mx_output_uint8=True) and GE graph outputs compare equal.
    """
    rank = 2  # y is always [Y, H]; op outputs carry no shape at conversion time
    div_x2 = ge.Cast(ge.Const([1] * (rank - 1) + [2]), dst_type=DataType.DT_INT32)
    packed_shape = ge.Div(ge.Shape(y), div_x2)
    pair_shape = ge.ConcatV2(
        [packed_shape, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
        concat_dim=0,
        N=2,
    )
    packed = ge.Bitcast(ge.Reshape(y, pair_shape), type=DataType.DT_UINT8)
    return ge.Reshape(packed, packed_shape)


def _torchair_supports_fp4_x2():
    """Detect TorchAir FP4 x2 output materialization (GE logical nibbles ->
    torch packed pairs). Older bundled builds map DT_FLOAT4_E2M1 to uint8."""
    try:
        from torchair.ge._ge_graph import DataType, ge_type_to_torch_type

        return ge_type_to_torch_type(DataType.DT_FLOAT4_E2M1) is getattr(
            torch, "float4_e2m1fn_x2", None
        )
    except Exception:  # noqa: BLE001
        return False


if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor([False], [False])
    def FfnWorkerBatching(
        schedule_context: Tensor,
        *,
        expert_num: int,
        max_out_shape: List[int],
        token_dtype: int = 0,
        need_schedule: int = 0,
        layer_num: int = 0,
        sync_flag: bool = False,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(FfnWorkerBatching)
        .INPUT(schedule_context, TensorType({DT_INT8}))
        .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_INT8,
                              DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))
        .OUTPUT(group_list, TensorType({DT_INT64}))
        .OUTPUT(session_ids, TensorType({DT_INT32}))
        .OUTPUT(micro_batch_ids, TensorType({DT_INT32}))
        .OUTPUT(token_ids, TensorType({DT_INT32}))
        .OUTPUT(expert_offsets, TensorType({DT_INT32}))
        .OUTPUT(dynamic_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
        .OUTPUT(actual_token_num, TensorType({DT_INT64}))
        .REQUIRED_ATTR(expert_num, Int)
        .REQUIRED_ATTR(max_out_shape, ListInt)
        .ATTR(token_dtype, Int, 0)
        .ATTR(need_schedule, Int, 0)
        .ATTR(layer_num, Int, 0)
        .ATTR(sync_flag, Bool, false)
        """
        from .ffn_worker_batching import _checked_y_rows

        _checked_y_rows(max_out_shape)
        # ge_op normally checks only non-default optional attributes. Even when
        # sync_flag is False, require the UPDATED prototype: an older registration
        # has a shorter attribute array than the current tiling implementation.
        # Silently omitting the default would hide this ABI mismatch until launch.
        error = is_cann_compat(
            "FfnWorkerBatching",
            runtime_optional_inputs=(),
            runtime_optional_attrs=("sync_flag",),
        )
        if error:
            raise RuntimeError(
                "FfnWorkerBatching GE requires the updated CANN/OPP prototype "
                "containing sync_flag, including when sync_flag=False. " + error
            )

        return ge_op(
            op_type="FfnWorkerBatching",
            inputs={"schedule_context": schedule_context},
            attrs={
                "expert_num": attr.Int(expert_num),
                "max_out_shape": attr.ListInt(max_out_shape),
                "token_dtype": attr.Int(token_dtype),
                "need_schedule": attr.Int(need_schedule),
                "layer_num": attr.Int(layer_num),
                "sync_flag": attr.Bool(sync_flag),
            },
            outputs=list(_OUTPUT_NAMES),
            dependencies=[] if dependencies is None else dependencies,
            node_name=node_name,
            ir=IrDef("FfnWorkerBatching")
            .input("schedule_context", "DT_INT8")
            .required_attr("expert_num", attr.Int)
            .required_attr("max_out_shape", attr.ListInt)
            .attr("token_dtype", attr.Int(0))
            .attr("need_schedule", attr.Int(0))
            .attr("layer_num", attr.Int(0))
            .attr("sync_flag", attr.Bool(False))
            .output(
                "y",
                "DT_FLOAT16, DT_BF16, DT_INT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1",
            )
            .output("group_list", "DT_INT64")
            .output("session_ids", "DT_INT32")
            .output("micro_batch_ids", "DT_INT32")
            .output("token_ids", "DT_INT32")
            .output("expert_offsets", "DT_INT32")
            .output("dynamic_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .output("actual_token_num", "DT_INT64"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.ffn_worker_batching.default
    )
    def convert_ffn_worker_batching(
        schedule_context: Tensor,
        expert_num: int,
        max_out_shape: List[int],
        *,
        token_dtype: int = 0,
        need_schedule: int = 0,
        layer_num: int = 0,
        sync_flag: bool = False,
        mx_output_uint8: bool = False,
        meta_outputs: Optional[List[TensorSpec]] = None,
    ):
        """Convert the dispatcher call to GE, preserving attribute/output order.

        Requires the current FfnWorkerBatching OPP and TorchAir support for the
        mutable schema's auto_functionalized_v2 lowering. Referenced buffers must
        remain alive; external producers own their synchronization. GE graph
        transport for FP4 uses the mx_output_uint8 byte contract (packed
        uint8, same bytes as eager); native float4_e2m1fn_x2 outputs require
        TorchAir FP4 materialization support and are rejected here until then.
        """
        # This flag changes the Torch storage representation, not the GE/ACLNN
        # operator contract. The GE graph honors it via a Bitcast epilogue on
        # the op outputs (below), never by forwarding it as a GE attribute.
        if token_dtype == _FP4_TOKEN_DTYPE and not mx_output_uint8:
            if not _torchair_supports_fp4_x2():
                raise NotImplementedError(
                    "FfnWorkerBatching GE requires TorchAir FP4 x2 output "
                    "materialization (ge_type_to_torch_type maps DT_FLOAT4_E2M1 "
                    "to torch.float4_e2m1fn_x2). Upgrade TorchAir, or run GE "
                    "graph legs with mx_output_uint8=True (byte-identical "
                    "uint8 transport). Got token_dtype={token_dtype}, "
                    "mx_output_uint8={mx_output_uint8}.".format(
                        token_dtype=token_dtype, mx_output_uint8=mx_output_uint8
                    )
                )
            # Native x2: the op's logical (Y, H) FP4 output flows to NetOutput
            # unchanged; TorchAir materializes it as float4_e2m1fn_x2 (Y, H/2).
        outputs = FfnWorkerBatching(
            schedule_context,
            expert_num=expert_num,
            max_out_shape=max_out_shape,
            token_dtype=token_dtype,
            need_schedule=need_schedule,
            layer_num=layer_num,
            sync_flag=sync_flag,
        )
        if mx_output_uint8 and token_dtype >= _FIRST_MX_TOKEN_DTYPE:
            outputs = list(outputs)
            # E8M0 scale shares uint8 itemsize: bitcast keeps shape, exposes
            # the raw exponent bytes without any value conversion.
            outputs[6] = ge.Bitcast(outputs[6], type=DataType.DT_UINT8)
            if token_dtype == _FP4_TOKEN_DTYPE:
                outputs[0] = _pack_fp4_to_uint8(outputs[0])
            # auto_functionalized_v2 unpacks tuple returns (op outputs + base
            # views must line up with the FX node meta); a list would nest.
            return tuple(outputs)
        return outputs
else:

    def convert_ffn_worker_batching(*args, **kwargs):
        """Report the optional GE dependency without breaking eager imports."""
        raise RuntimeError(
            "FfnWorkerBatching GE converter requires torchair, but torchair is not available."
        )
