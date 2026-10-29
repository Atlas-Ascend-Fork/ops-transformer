# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch adapter for MergeChunk2."""

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import OpBuilder, get_as_library


class MergeChunk2OpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("merge_chunk2", category="moe")

    def sources(self):
        return ["csrc/moe/merge_chunk2.cpp"]

    def schema(self):
        return (
            "merge_chunk2(Tensor input1_chunk1, Tensor input1_chunk2, "
            "Tensor input2_chunk1, Tensor input2_chunk2, "
            "Tensor group_list_chunk1, Tensor group_list_chunk2) "
            "-> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def merge_chunk2_meta(
            input1_chunk1,
            input1_chunk2,
            input2_chunk1,
            input2_chunk2,
            group_list_chunk1,
            group_list_chunk2,
        ):
            if input1_chunk1.dim() != 2 or input1_chunk2.dim() != 2:
                raise ValueError("input1_chunk1/input1_chunk2 rank must be 2")
            if input2_chunk1.dim() != 2 or input2_chunk2.dim() != 2:
                raise ValueError("input2_chunk1/input2_chunk2 rank must be 2")
            if group_list_chunk1.dim() != 1 or group_list_chunk2.dim() != 1:
                raise ValueError("group_list_chunk1/group_list_chunk2 rank must be 1")
            if input1_chunk1.size(0) != input2_chunk1.size(0):
                raise ValueError(
                    "input1_chunk1 and input2_chunk1 must have the same token num"
                )
            if input1_chunk2.size(0) != input2_chunk2.size(0):
                raise ValueError(
                    "input1_chunk2 and input2_chunk2 must have the same token num"
                )
            if input1_chunk1.size(1) != input1_chunk2.size(1):
                raise ValueError(
                    "input1_chunk1 and input1_chunk2 must have the same hidden size"
                )
            if input2_chunk1.size(1) != input2_chunk2.size(1):
                raise ValueError(
                    "input2_chunk1 and input2_chunk2 must have the same hidden size"
                )
            if group_list_chunk1.size(0) != group_list_chunk2.size(0):
                raise ValueError(
                    "group_list_chunk1 and group_list_chunk2 must have the same num expert"
                )

            return (
                torch.empty(
                    (
                        input1_chunk1.size(0) + input1_chunk2.size(0),
                        input1_chunk1.size(1),
                    ),
                    device="meta",
                    dtype=input1_chunk1.dtype,
                ),
                torch.empty(
                    (
                        input2_chunk1.size(0) + input2_chunk2.size(0),
                        input2_chunk1.size(1),
                    ),
                    device="meta",
                    dtype=input2_chunk1.dtype,
                ),
                torch.empty(
                    (group_list_chunk1.size(0),),
                    device="meta",
                    dtype=group_list_chunk1.dtype,
                ),
            )


_merge_chunk2_op_builder = MergeChunk2OpBuilder()
_merge_chunk2_op_builder._ensure_initialized()


@impl(get_as_library(), _merge_chunk2_op_builder.name, "PrivateUse1")
def _merge_chunk2(
    input1_chunk1: torch.Tensor,
    input1_chunk2: torch.Tensor,
    input2_chunk1: torch.Tensor,
    input2_chunk2: torch.Tensor,
    group_list_chunk1: torch.Tensor,
    group_list_chunk2: torch.Tensor,
):
    module = _merge_chunk2_op_builder.load()
    return module.merge_chunk2(
        input1_chunk1,
        input1_chunk2,
        input2_chunk1,
        input2_chunk2,
        group_list_chunk1,
        group_list_chunk2,
    )


def merge_chunk2(
    input1_chunk1: torch.Tensor,
    input1_chunk2: torch.Tensor,
    input2_chunk1: torch.Tensor,
    input2_chunk2: torch.Tensor,
    group_list_chunk1: torch.Tensor,
    group_list_chunk2: torch.Tensor,
):
    """MergeChunk2, 按 expert 合并两个 chunk 的 token, 封装 aclnnMergeChunk2。

    对每个 expert e:
        out1 依次拼接 input1_chunk1 中属于 e 的 token 和 input1_chunk2 中属于 e 的 token;
        out2 依次拼接 input2_chunk1 中属于 e 的 token 和 input2_chunk2 中属于 e 的 token;
        group_list[e] = group_list_chunk1[e] + group_list_chunk2[e]。

    Args:
        input1_chunk1 (Tensor): [t1, h1], bf16/fp16
        input1_chunk2 (Tensor): [t2, h1], bf16/fp16
        input2_chunk1 (Tensor): [t1, h2], bf16/fp16
        input2_chunk2 (Tensor): [t2, h2], bf16/fp16
        group_list_chunk1 (Tensor): [num_expert], int32
        group_list_chunk2 (Tensor): [num_expert], int32

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            out1: [t1 + t2, h1]
            out2: [t1 + t2, h2]
            group_list: [num_expert], int32
    """
    return torch.ops.cann_ops_transformer.merge_chunk2(
        input1_chunk1,
        input1_chunk2,
        input2_chunk1,
        input2_chunk2,
        group_list_chunk1,
        group_list_chunk2,
    )
