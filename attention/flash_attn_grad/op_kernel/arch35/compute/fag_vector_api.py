# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""VF 微内核。

本文件只放 ``@pl.vector_function`` 的**纯寄存器级**实现，一律「进 UB
地址、出 UB 地址」，不碰 GM、不碰 L1、不读 runInfo；搬运与编排在上层的
:mod:`fag_block_vec`。

VF 可以按 dtype / 列宽分档写多份（如 128 与 192 两档 softmaxGradFront），
调用方只需在一处选档。
"""

import pypto_pro.language as pl

# --- generated imports (fag_fiximports.py) ---
from fag_common import (
    NZP_BLOCK_STRIDE,
    NZP_REPEAT_STRIDE,
    VECTOR_BASEN,
    VF_LANES,
    VF_LANES_F16,
)
# --- end generated imports ---


# `vf` 由 parser 按语法识别(_call_parser.py: attrs[0] == "vf")，无需 import

# ---- V1: softmaxGradFront  sfmg = rowsum(dy * y) ----


@pl.vector_function
def softmax_grad_front_row_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
):
    """ASC 128 column tree; zero tail lanes before one row reduction."""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    aregSfmg = vf.unalign_reg_for_store()
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregA0 = vf.add(vregMul, vregMul2, pregFullExe)
        vregSum = vf.reduce_sum(vregA0, pregFullExe)
        vf.store_unalign(sfmg_ub, vregSum, aregSfmg, 1, post_update=True)
    vf.store_unalign_post(sfmg_ub, aregSfmg, 0, post_update=True)


@pl.vector_function
def softmax_grad_front_row192_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
):
    """ASC 192 column tree; zero tail lanes before one row reduction."""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    aregSfmg = vf.unalign_reg_for_store()
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA0 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 128)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 128)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregA1 = vf.add(vregMul, vregMul2, pregFullExe)
        vregA0 = vf.add(vregA0, vregA1, pregFullExe)
        vregSum = vf.reduce_sum(vregA0, pregFullExe)
        vf.store_unalign(sfmg_ub, vregSum, aregSfmg, 1, post_update=True)
    vf.store_unalign_post(sfmg_ub, aregSfmg, 0, post_update=True)


@pl.vector_function
def softmax_grad_front_row256_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
):
    """ASC 256 column tree; zero tail lanes before one row reduction."""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    aregSfmg = vf.unalign_reg_for_store()
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA0 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 128)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 128)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregA1 = vf.add(vregMul, vregMul2, pregFullExe)
        vregA0 = vf.add(vregA0, vregA1, pregFullExe)
        vregSum = vf.reduce_sum(vregA0, pregFullExe)
        vf.store_unalign(sfmg_ub, vregSum, aregSfmg, 1, post_update=True)
    vf.store_unalign_post(sfmg_ub, aregSfmg, 0, post_update=True)


@pl.vector_function
def softmax_grad_front_row384_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
    dstRowOffset: pl.DT_INT64,
):
    """ASC 384 column tree; zero tail lanes before one row reduction."""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA0 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 128)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 128)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA1 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 256)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 256)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregA2 = vf.add(vregMul, vregMul2, pregFullExe)
        vregA0 = vf.add(vregA0, vregA1, pregFullExe)
        vregA0 = vf.add(vregA0, vregA2, pregFullExe)
        vregSum = vf.reduce_sum(vregA0, pregFullExe)
        vf.store_align(
            sfmg_ub,
            vregSum,
            pregFullExe,
            m + dstRowOffset,
            dist=pl.StoreDist.FIRST_ELEMENT_B32,
        )


@pl.vector_function
def softmax_grad_front_row512_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
    dstRowOffset: pl.DT_INT64,
):
    """ASC 512 column tree; zero tail lanes before one row reduction."""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA0 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 128)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 128)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA1 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 256)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 256)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA2 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 384)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 384)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregA3 = vf.add(vregMul, vregMul2, pregFullExe)
        vregA0 = vf.add(vregA0, vregA3, pregFullExe)
        vregA1 = vf.add(vregA1, vregA2, pregFullExe)
        vregA0 = vf.add(vregA0, vregA1, pregFullExe)
        vregSum = vf.reduce_sum(vregA0, pregFullExe)
        vf.store_align(
            sfmg_ub,
            vregSum,
            pregFullExe,
            m + dstRowOffset,
            dist=pl.StoreDist.FIRST_ELEMENT_B32,
        )


@pl.vector_function
def softmax_grad_front_row640_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
    dstRowOffset: pl.DT_INT64,
):
    """ASC 640 column tree; zero tail lanes before one row reduction."""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA0 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 128)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 128)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA1 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 256)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 256)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA2 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 384)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 384)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA3 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 512)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 512)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregA4 = vf.add(vregMul, vregMul2, pregFullExe)
        vregA1 = vf.add(vregA1, vregA2, pregFullExe)
        vregA0 = vf.add(vregA0, vregA4, pregFullExe)
        vregA0 = vf.add(vregA0, vregA3, pregFullExe)
        vregA0 = vf.add(vregA1, vregA0, pregFullExe)
        vregSum = vf.reduce_sum(vregA0, pregFullExe)
        vf.store_align(
            sfmg_ub,
            vregSum,
            pregFullExe,
            m + dstRowOffset,
            dist=pl.StoreDist.FIRST_ELEMENT_B32,
        )


@pl.vector_function
def softmax_grad_front_row768_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
    dstRowOffset: pl.DT_INT64,
):
    """ASC 768 column tree; zero tail lanes before one row reduction."""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA0 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 128)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 128)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA1 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 256)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 256)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA2 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 384)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 384)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA3 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 512)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 512)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregFullExe)
        vregMul2 = vf.mul(vregDy1, vregY1, pregFullExe)
        vregA4 = vf.add(vregMul, vregMul2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 640)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 640)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregA5 = vf.add(vregMul, vregMul2, pregFullExe)
        vregA3 = vf.add(vregA3, vregA4, pregFullExe)
        vregA0 = vf.add(vregA0, vregA5, pregFullExe)
        vregA1 = vf.add(vregA1, vregA2, pregFullExe)
        vregA0 = vf.add(vregA0, vregA1, pregFullExe)
        vregA0 = vf.add(vregA3, vregA0, pregFullExe)
        vregSum = vf.reduce_sum(vregA0, pregFullExe)
        vf.store_align(
            sfmg_ub,
            vregSum,
            pregFullExe,
            m + dstRowOffset,
            dist=pl.StoreDist.FIRST_ELEMENT_B32,
        )


# ---- V3: dS = (dP - sfmg) * P，随后 cast + ND2NZ ----


@pl.vector_function
def broadcast_sub_mul_cast_vf(
    nzp_ub, mm1_ub, mm2_ub, sfmg_ub, scale: pl.DT_FP32, srcM: pl.DT_INT64
):
    """dS = (dP - broadcast(sfmg)) * P，并 cast 成 16-bit。

    sfmg 按行 BRC_B32 广播，128 列拆成两个 64 lane 寄存器；
    两个 fp32 寄存器 cast 后交错写回一个 16-bit 行。
    此处不乘 scale（scale 在 dQ/dK 后处理施加）。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    for m in pl.range(0, srcM):
        vregSfmg = vf.load_align(sfmg_ub, m, dist=pl.LoadDist.BRC_B32)
        # 用 DINTLV_B32 一次载入 128 个 fp32 并拆成 even/odd，
        # 使后续 Even/Odd cast + Or 的合并顺序正确；
        # sub/mul 为逐元素运算，只要 dP/P 拆分方式一致即可
        vregDpEven, vregDpOdd = vf.load_align(
            mm1_ub, m * VECTOR_BASEN, dist=pl.LoadDist.DINTLV_B32
        )
        vregPEven, vregPOdd = vf.load_align(
            mm2_ub, m * VECTOR_BASEN, dist=pl.LoadDist.DINTLV_B32
        )
        vregSub0 = vf.sub(vregDpEven, vregSfmg, pregFullExe)
        vregDs0 = vf.mul(vregSub0, vregPEven, pregFullExe)
        vregSub1 = vf.sub(vregDpOdd, vregSfmg, pregFullExe)
        vregDs1 = vf.mul(vregSub1, vregPOdd, pregFullExe)
        # 这里不乘 scale：scale 必须在矩阵乘之后、fp32 结果上施加
        # (post 阶段 muls_cast_dqkv)。虽然 scale*(dS@K)==(scale*dS)@K
        # 数学等价，但先乘会让 16-bit 舍入发生在不同量级上，训练中误差会累积。
        # fp32 -> 16-bit：Even/Odd 分别 cast 进 b16 寄存器的两个半区再 Or 合并。
        vregCastEven = vf.astype(
            vregDs0, pregFullExeB16, dtype=input_dtype, layout=pl.CastLayout.ZERO
        )
        vregCastOdd = vf.astype(
            vregDs1, pregFullExeB16, dtype=input_dtype, layout=pl.CastLayout.ONE
        )
        vregCastRes = vf.or_(vregCastEven, vregCastOdd, pregFullExeB16)
        # DataBlock 拷贝模式(vsstb)：一条 store 同时完成 ND->NZ 重排，
        # 省掉独立的 pl.move —— 后者跑在 vector 计算流水上。
        # block_stride=65 带 +1 把相邻分形列错开到不同 bank。
        vf.store_align(
            nzp_ub,
            vregCastRes,
            pregFullExeB16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZP_BLOCK_STRIDE,
            repeat_stride=NZP_REPEAT_STRIDE,
            post_update=True,
        )


# ---- V4: P cast + ND2NZ ----


@pl.vector_function
def cast_p_vf(nzp_ub, mm2_ub, srcM: pl.DT_INT64):
    """P: fp32 -> 16-bit，并按 DataBlock 模式写成 NZ。"""
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    for m in pl.range(0, srcM):
        vregPEven, vregPOdd = vf.load_align(
            mm2_ub, m * VECTOR_BASEN, dist=pl.LoadDist.DINTLV_B32
        )
        vregCastEven = vf.astype(
            vregPEven, pregFullExeB16, dtype=input_dtype, layout=pl.CastLayout.ZERO
        )
        vregCastOdd = vf.astype(
            vregPOdd, pregFullExeB16, dtype=input_dtype, layout=pl.CastLayout.ONE
        )
        vregCastRes = vf.or_(vregCastEven, vregCastOdd, pregFullExeB16)
        # DataBlock 拷贝模式(vsstb)：一条 store 同时完成 ND->NZ 重排，
        # 省掉独立的 pl.move —— 后者跑在 vector 计算流水上。
        # block_stride=65 带 +1 把相邻分形列错开到不同 bank。
        vf.store_align(
            nzp_ub,
            vregCastRes,
            pregFullExeB16,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            block_stride=NZP_BLOCK_STRIDE,
            repeat_stride=NZP_REPEAT_STRIDE,
            post_update=True,
        )


# ---- pre: 清零 ----


@pl.vector_function
def fill_zero_vf(dst_ub, nvecs: pl.DT_INT64):
    """把 UB tile 的前 nvecs 个 vreg(每个 VF_LANES 个 fp32)填 0。

    1D tile 连续排布。调用方传 ceil(nElems / VF_LANES)。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    vregZero = vf.full(0.0, pregFullExe, dtype=pl.DT_FP32)
    for i in pl.range(0, nvecs):
        vf.store_align(dst_ub + i * VF_LANES, vregZero, pregFullExe)


@pl.vector_function
def fill_zero_f16_vf(dst_ub, nvecs: pl.DT_INT64):
    """把 UB tile 的前 nvecs 个 vreg(每个 VF_LANES_F16 个 16-bit)填 0。"""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    vregZero = vf.full(0.0, pregFullExe, dtype=input_dtype)
    for i in pl.range(0, nvecs):
        vf.store_align(dst_ub + i * VF_LANES_F16, vregZero, pregFullExe)


@pl.vector_function
def fill_scaled_zero_bn2s2_vf(dst, scale: pl.DT_FP32, vectors: pl.DT_INT64):
    # Match muls_cast_vf(+0, scale), including negative and nonfinite scales.
    mask32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    mask16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    zero = vf.full(0.0, mask32, dtype=pl.DT_FP32)
    scaled = vf.muls(zero, scale, mask32)
    even = vf.astype(scaled, mask16, dtype=input_dtype, layout=pl.CastLayout.ZERO)
    odd = vf.astype(scaled, mask16, dtype=input_dtype, layout=pl.CastLayout.ONE)
    packed = vf.or_(even, odd, mask16)
    for i in pl.range(0, vectors):
        vf.store_align(dst + i * VF_LANES_F16, packed, mask16)


@pl.vector_function
def dsink_partial_vf(
    sink_part_ub,
    p_ub,
    dp_ub,
    lse_ub,
    sink: pl.DT_FP32,
    srcM: pl.DT_INT64,
    loopTimes: pl.DT_INT64,
    tailCols: pl.DT_INT64,
    rowShift: pl.DT_INT64,
):
    """本 s2 块的 dsink 部分和，写 sink_part_ub 的 lane 0。

    -Σ_m (Σ_n P·dP)·exp(sink-lse[m+rowShift])。列按 VF_LANES 分段，尾段用 tailCols。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregOne = vf.update_mask(1, dtype=pl.DT_FP32)
    pregTail = vf.update_mask(tailCols, dtype=pl.DT_FP32)
    vregSink = vf.full(sink, dtype=pl.DT_FP32)
    vregAcc = vf.full(0.0, dtype=pl.DT_FP32)
    for m in pl.range(0, srcM):
        vregLse = vf.load_align(lse_ub, m + rowShift, dist=pl.LoadDist.BRC_B32)
        vregExp = vf.exp_sub(vregSink, vregLse, pregFullExe)
        # 段内先做 elementwise 累加（不跨 lane），整行只做 1 次 reduce_sum：
        # exp 是行内标量，提到归约之后乘，数学等价、又少一轮跨 lane 归约。
        vregRow = vf.full(0.0, dtype=pl.DT_FP32)
        for n in pl.range(0, loopTimes):
            off = m * VECTOR_BASEN + n * VF_LANES
            vregP = vf.load_align(p_ub, off)
            vregDp = vf.load_align(dp_ub, off)
            vregMul = vf.mul(vregP, vregDp, pregFullExe)
            vregRow = vf.add(vregRow, vregMul, pregFullExe)
        offTail = m * VECTOR_BASEN + loopTimes * VF_LANES
        vregP = vf.load_align(p_ub, offTail)
        vregDp = vf.load_align(dp_ub, offTail)
        vregMul = vf.mul(vregP, vregDp, pregTail)
        vregRow = vf.add(vregRow, vregMul, pregFullExe)
        vregRow = vf.mul(vregRow, vregExp, pregFullExe)
        vregRed = vf.reduce_sum(vregRow, pregFullExe)
        vregAcc = vf.add(vregAcc, vregRed, pregOne)
    vregNeg = vf.muls(vregAcc, -1.0, pregFullExe)
    vf.store_align(sink_part_ub, vregNeg, pregOne)


@pl.vector_function
def dsink_reduce_add_vf(acc_ub, row_ub, nsegs: pl.DT_INT64):
    """acc += row。AIV0 按固定行序归约各 AIV 的 dsink 部分和。"""
    pregFull = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(0, nsegs):
        vregAcc = vf.load_align(acc_ub, i * VF_LANES)
        vregRow = vf.load_align(row_ub, i * VF_LANES)
        vregSum = vf.add(vregAcc, vregRow, pregFull)
        vf.store_align(acc_ub + i * VF_LANES, vregSum, pregFull)


# ---- post / V5 / V6: fp32 -> muls(scale) -> 16-bit ----


@pl.vector_function
def muls_cast_vf(dst_ub, src_ub, scale: pl.DT_FP32, nsegs: pl.DT_INT64):
    """fp32 -> muls(scale) -> 16-bit，按 128 个元素一段扁平遍历整块 tile。

    Muls/Cast 直接吃整块元素，完全不看行宽。Vec tile 是紧凑 RowMajor，行距就等于
    声明宽度 PHYS_W，整块在 UB 里连续，可以当一维数组遍历。

    按行走的老写法在 D≤64 时很吃亏：一次 DINTLV_B32 能覆盖 128 个 fp32，
    但一行只有 64 个，even/odd 掩码各只开一半 lane，VF 利用率 50%。扁平后
    三条掩码恒为全满，既省掉每次调用的 plt_b32/plt_b16 标量配置，vmuls 也
    跑在满 lane 上。

    掩码全满会连 padding 列一起算(D=47 时每行多算 17 列的残值)，但那些列
    被 store 的 valid_shape 挡在 UB 里，出不到 GM。调用方保证
    rows*PHYS_W % 128 == 0，故段边界不会越出 tile。
    """
    pregFull32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFull16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    for m in pl.range(0, nsegs):
        vregEven, vregOdd = vf.load_align(
            src_ub, m * VECTOR_BASEN, dist=pl.LoadDist.DINTLV_B32
        )
        vregEven = vf.muls(vregEven, scale, pregFull32)
        vregOdd = vf.muls(vregOdd, scale, pregFull32)
        vregCastEven = vf.astype(
            vregEven, pregFull16, dtype=input_dtype, layout=pl.CastLayout.ZERO
        )
        vregCastOdd = vf.astype(
            vregOdd, pregFull16, dtype=input_dtype, layout=pl.CastLayout.ONE
        )
        vregRes = vf.or_(vregCastEven, vregCastOdd, pregFull16)
        vf.store_align(dst_ub + m * VECTOR_BASEN, vregRes, pregFull16)


# BN2S2 variants retain their original reduction order and argument contracts.
# Distinct names keep both template families visible to codegen without rebinding.


@pl.vector_function
def softmax_grad_front_single_bn2s2_vf(
    sfmg, y, dy, rows: pl.DT_INT64, shift: pl.DT_INT64, stride: pl.DT_INT64
):
    b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    fp32 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    out = vf.unalign_reg_for_store()
    for row in pl.range(0, rows):
        vy = vf.load_align(y, (row + shift) * stride)
        vdy = vf.load_align(dy, (row + shift) * stride)
        fy = vf.astype(vy, b16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        fdy = vf.astype(vdy, b16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        scalar = vf.mul(fdy, fy, fp32)
        vf.store_unalign(sfmg, scalar, out, 1, post_update=True)
    vf.store_unalign_post(sfmg, out, 0, post_update=True)


@pl.vector_function
def softmax_grad_front_row_bn2s2_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
):
    """sfmg：Dv≤128，整行一次 128-16bit load，even/odd mask 盖真实宽度。"""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    aregSfmg = vf.unalign_reg_for_store()
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregRed = vf.reduce_sum(vregMul, pregEven)
        vregRed2 = vf.reduce_sum(vregMul2, pregOdd)
        vregSum = vf.add(vregRed, vregRed2, pregFullExe)
        vf.store_unalign(sfmg_ub, vregSum, aregSfmg, 1, post_update=True)
    vf.store_unalign_post(sfmg_ub, aregSfmg, 0, post_update=True)


@pl.vector_function
def softmax_grad_front_row192_bn2s2_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
    evenSize2: pl.DT_INT64,
    oddSize2: pl.DT_INT64,
):
    """sfmg：Dv=192 档。先收前 128 列，再 colOff=128 收剩余，两趟 reduce 相加。

    对标老 FAG `MySoftmaxGradFrontCast<..., 384, HEAD_DIM_ALIGN>`：独立档位
    函数，不把 128/192 写进同一个 VF。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    pregEven2 = vf.update_mask(evenSize2, dtype=pl.DT_FP32)
    pregOdd2 = vf.update_mask(oddSize2, dtype=pl.DT_FP32)
    aregSfmg = vf.unalign_reg_for_store()
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregRed = vf.reduce_sum(vregMul, pregEven)
        vregRed2 = vf.reduce_sum(vregMul2, pregOdd)
        vregSum = vf.add(vregRed, vregRed2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 128)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 128)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven2)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd2)
        vregRed = vf.reduce_sum(vregMul, pregEven2)
        vregRed2 = vf.reduce_sum(vregMul2, pregOdd2)
        vregSum2 = vf.add(vregRed, vregRed2, pregFullExe)
        vregSum = vf.add(vregSum, vregSum2, pregFullExe)
        vf.store_unalign(sfmg_ub, vregSum, aregSfmg, 1, post_update=True)
    vf.store_unalign_post(sfmg_ub, aregSfmg, 0, post_update=True)


@pl.vector_function
def softmax_grad_front_row256_bn2s2_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
    evenSize2: pl.DT_INT64,
    oddSize2: pl.DT_INT64,
):
    """sfmg：Dv=256 档。先收前 128 列，再 colOff=128 收剩余，两趟 reduce 相加。

    独立档位函数，不把 128/192/256 写进同一个 VF。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    pregEven2 = vf.update_mask(evenSize2, dtype=pl.DT_FP32)
    pregOdd2 = vf.update_mask(oddSize2, dtype=pl.DT_FP32)
    aregSfmg = vf.unalign_reg_for_store()
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregRed = vf.reduce_sum(vregMul, pregEven)
        vregRed2 = vf.reduce_sum(vregMul2, pregOdd)
        vregSum = vf.add(vregRed, vregRed2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 128)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 128)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven2)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd2)
        vregRed = vf.reduce_sum(vregMul, pregEven2)
        vregRed2 = vf.reduce_sum(vregMul2, pregOdd2)
        vregSum2 = vf.add(vregRed, vregRed2, pregFullExe)
        vregSum = vf.add(vregSum, vregSum2, pregFullExe)
        vf.store_unalign(sfmg_ub, vregSum, aregSfmg, 1, post_update=True)
    vf.store_unalign_post(sfmg_ub, aregSfmg, 0, post_update=True)


@pl.vector_function
def softmax_grad_front_row256_acc_vf(
    sfmg_ub,
    y_ub,
    dy_ub,
    srcM: pl.DT_INT64,
    rowShift: pl.DT_INT64,
    rowStride: pl.DT_INT64,
    evenSize: pl.DT_INT64,
    oddSize: pl.DT_INT64,
    evenSize2: pl.DT_INT64,
    oddSize2: pl.DT_INT64,
):
    """768 档后续窗：与 VF256 相同，但把本窗 reduce 累加进已有 sfmg。"""
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregFullExeB16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pregEven = vf.update_mask(evenSize, dtype=pl.DT_FP32)
    pregOdd = vf.update_mask(oddSize, dtype=pl.DT_FP32)
    pregEven2 = vf.update_mask(evenSize2, dtype=pl.DT_FP32)
    pregOdd2 = vf.update_mask(oddSize2, dtype=pl.DT_FP32)
    for m in pl.range(0, srcM):
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd)
        vregRed = vf.reduce_sum(vregMul, pregEven)
        vregRed2 = vf.reduce_sum(vregMul2, pregOdd)
        vregSum = vf.add(vregRed, vregRed2, pregFullExe)
        vregYHalf = vf.load_align(y_ub, (m + rowShift) * rowStride + 128)
        vregDyHalf = vf.load_align(dy_ub, (m + rowShift) * rowStride + 128)
        vregY = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregY1 = vf.astype(
            vregYHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregDy = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO
        )
        vregDy1 = vf.astype(
            vregDyHalf, pregFullExeB16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE
        )
        vregMul = vf.mul(vregDy, vregY, pregEven2)
        vregMul2 = vf.mul(vregDy1, vregY1, pregOdd2)
        vregRed = vf.reduce_sum(vregMul, pregEven2)
        vregRed2 = vf.reduce_sum(vregMul2, pregOdd2)
        vregSum2 = vf.add(vregRed, vregRed2, pregFullExe)
        vregSum = vf.add(vregSum, vregSum2, pregFullExe)
        vregOld = vf.load_align(sfmg_ub, m, dist=pl.LoadDist.BRC_B32)
        vregSum = vf.add(vregSum, vregOld, pregFullExe)
        # Keep the base fixed: later iterations read prior-window sums at base + m.
        vf.store_align(
            sfmg_ub,
            vregSum,
            pregFullExe,
            m,
            dist=pl.StoreDist.FIRST_ELEMENT_B32,
        )


@pl.vector_function
def dsink_partial_bn2s2_vf(
    sink_part_ub,
    p_ub,
    dp_ub,
    lse_ub,
    sink: pl.DT_FP32,
    srcM: pl.DT_INT64,
    loopTimes: pl.DT_INT64,
    tailCols: pl.DT_INT64,
    rowShift: pl.DT_INT64,
):
    """本 s2 块的 dsink 部分和，写 sink_part_ub 的 lane 0。

    -Σ_m (Σ_n P·dP)·exp(sink-lse[m+rowShift])。列按 VF_LANES 分段，尾段用 tailCols。
    """
    pregFullExe = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    pregOne = vf.update_mask(1, dtype=pl.DT_FP32)
    pregTail = vf.update_mask(tailCols, dtype=pl.DT_FP32)
    vregSink = vf.full(sink, dtype=pl.DT_FP32)
    vregAcc = vf.full(0.0, dtype=pl.DT_FP32)
    for m in pl.range(0, srcM):
        vregLse = vf.load_align(lse_ub, m + rowShift, dist=pl.LoadDist.BRC_B32)
        vregExp = vf.exp_sub(vregSink, vregLse, pregFullExe)
        for n in pl.range(0, loopTimes):
            off = m * VECTOR_BASEN + n * VF_LANES
            vregP = vf.load_align(p_ub, off)
            vregDp = vf.load_align(dp_ub, off)
            vregMul = vf.mul(vregP, vregDp, pregFullExe)
            vregMul = vf.mul(vregMul, vregExp, pregFullExe)
            vregRed = vf.reduce_sum(vregMul, pregFullExe)
            vregAcc = vf.add(vregAcc, vregRed, pregOne)
        offTail = m * VECTOR_BASEN + loopTimes * VF_LANES
        vregP = vf.load_align(p_ub, offTail)
        vregDp = vf.load_align(dp_ub, offTail)
        vregMul = vf.mul(vregP, vregDp, pregTail)
        vregMul = vf.mul(vregMul, vregExp, pregTail)
        vregRed = vf.reduce_sum(vregMul, pregTail)
        vregAcc = vf.add(vregAcc, vregRed, pregOne)
    vregNeg = vf.muls(vregAcc, -1.0, pregFullExe)
    vf.store_align(sink_part_ub, vregNeg, pregOne)


@pl.vector_function
def plan_merge_band_mask_vf(mask_next, mask_pre, rows: pl.DT_INT64):
    # A uint8 register contains 256 lanes, unlike the 64-lane FP32 VFs.
    # Process pairs of 128-byte rows so no load/store crosses either tile.
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT8)
    one = vf.full(1, full, dtype=pl.DT_UINT8)
    for pair in pl.range(0, rows // 2):
        previous = vf.load_align(mask_pre, pair * 256)
        following = vf.load_align(mask_next, pair * 256)
        inverse = vf.xor(previous, one, full)
        merged = vf.or_(inverse, following, full)
        vf.store_align(mask_next + pair * 256, merged, full)
    if rows % 2 != 0:
        tail = vf.update_mask(128, dtype=pl.DT_UINT8)
        previous_tail = vf.load_align(mask_pre, rows // 2 * 256)
        following_tail = vf.load_align(mask_next, rows // 2 * 256)
        inverse_tail = vf.xor(previous_tail, one, tail)
        merged_tail = vf.or_(inverse_tail, following_tail, tail)
        vf.store_align(mask_next + rows // 2 * 256, merged_tail, tail)


@pl.vector_function
def plan_sink_partial_vf(
    dst,
    p,
    dp,
    lse,
    sink: pl.DT_FP32,
    rows: pl.DT_INT64,
    loops: pl.DT_INT64,
    tail: pl.DT_INT64,
    shift: pl.DT_INT64,
):
    # ASC CalculateSinkVF: positive partial; negation belongs to final post.
    # exp(sink-lse) is the explicit LSE interface adaptation of exp(sink-max)/sum.
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.update_mask(1, dtype=pl.DT_FP32)
    tail_mask = vf.update_mask(tail, dtype=pl.DT_FP32)
    sink_reg = vf.full(sink, dtype=pl.DT_FP32)
    acc = vf.full(0.0, dtype=pl.DT_FP32)
    for m in pl.range(0, rows):
        normalizer = vf.load_align(lse, m + shift, dist=pl.LoadDist.BRC_B32)
        weight = vf.exp_sub(sink_reg, normalizer, full)
        for n in pl.range(0, loops):
            pp = vf.load_align(p, m * 128 + n * 64)
            dd = vf.load_align(dp, m * 128 + n * 64)
            product = vf.mul(pp, dd, full)
            weighted = vf.mul(product, weight, full)
            reduced = vf.reduce_sum(weighted, full)
            acc = vf.add(acc, reduced, one)
        pp_tail = vf.load_align(p, m * 128 + loops * 64)
        dd_tail = vf.load_align(dp, m * 128 + loops * 64)
        product_tail = vf.mul(pp_tail, dd_tail, tail_mask)
        weighted_tail = vf.mul(product_tail, weight, tail_mask)
        reduced_tail = vf.reduce_sum(weighted_tail, tail_mask)
        acc = vf.add(acc, reduced_tail, one)
    vf.store_align(dst, acc, one)


@pl.vector_function
def plan_cast_only_vf(dst, src, nsegs: pl.DT_INT64):
    # dV has no scale operation in ASC. Do not insert a floating multiply by 1.
    mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    for i in pl.range(0, nsegs):
        even, odd = vf.load_align(src, i * 128, dist=pl.LoadDist.DINTLV_B32)
        low = vf.astype(even, mask, dtype=input_dtype, layout=pl.CastLayout.ZERO)
        high = vf.astype(odd, mask, dtype=input_dtype, layout=pl.CastLayout.ONE)
        result = vf.or_(low, high, mask)
        vf.store_align(dst + i * 128, result, mask)


@pl.vector_function
def plan_sink_reduce_chunk_vf(total, src, loops: pl.DT_INT64, tail: pl.DT_INT64):
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.update_mask(1, dtype=pl.DT_FP32)
    last = vf.update_mask(tail, dtype=pl.DT_FP32)
    acc = vf.full(0.0, dtype=pl.DT_FP32)
    for i in pl.range(0, loops):
        value = vf.load_align(src, i * 64)
        reduced = vf.reduce_sum(value, full)
        acc = vf.add(acc, reduced, one)
    value_tail = vf.load_align(src, loops * 64)
    reduced_tail = vf.reduce_sum(value_tail, last)
    acc = vf.add(acc, reduced_tail, one)
    previous = vf.load_align(total)
    updated = vf.add(previous, acc, one)
    vf.store_align(total, updated, one)


@pl.vector_function
def plan_sink_negate_vf(total):
    one = vf.update_mask(1, dtype=pl.DT_FP32)
    value = vf.load_align(total)
    negative = vf.muls(value, -1.0, one)
    vf.store_align(total, negative, one)


@pl.vector_function
def softmax_grad_front_row320_single_vf(
    sfmg_ub, y_ub, dy_ub, srcM: pl.DT_INT64, tailEven: pl.DT_INT64, tailOdd: pl.DT_INT64
):
    """D257..320: full-row input, paired products then one reduction.

    Physical view is 81x320; caller loads at most80 rows. The final
    128-element read stays in the original slot. Tail products explicitly
    zero inactive lanes before combining all64 FP32 lanes.
    """
    full = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    full16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=input_dtype)
    pe = vf.update_mask(tailEven, dtype=pl.DT_FP32)
    po = vf.update_mask(tailOdd, dtype=pl.DT_FP32)
    out = vf.unalign_reg_for_store()
    for m in pl.range(0, srcM):
        yh0 = vf.load_align(y_ub, m * 320 + 0)
        dh0 = vf.load_align(dy_ub, m * 320 + 0)
        ye0 = vf.astype(yh0, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        yo0 = vf.astype(yh0, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        de0 = vf.astype(dh0, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        do0 = vf.astype(dh0, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        me0 = vf.mul(de0, ye0, full, mode=pl.MergeMode.ZEROING)
        mo0 = vf.mul(do0, yo0, full, mode=pl.MergeMode.ZEROING)
        pair0 = vf.add(me0, mo0, full)
        yh1 = vf.load_align(y_ub, m * 320 + 128)
        dh1 = vf.load_align(dy_ub, m * 320 + 128)
        ye1 = vf.astype(yh1, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        yo1 = vf.astype(yh1, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        de1 = vf.astype(dh1, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        do1 = vf.astype(dh1, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        me1 = vf.mul(de1, ye1, full, mode=pl.MergeMode.ZEROING)
        mo1 = vf.mul(do1, yo1, full, mode=pl.MergeMode.ZEROING)
        pair1 = vf.add(me1, mo1, full)
        yh2 = vf.load_align(y_ub, m * 320 + 256)
        dh2 = vf.load_align(dy_ub, m * 320 + 256)
        ye2 = vf.astype(yh2, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        yo2 = vf.astype(yh2, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        de2 = vf.astype(dh2, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        do2 = vf.astype(dh2, full16, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        me2 = vf.mul(de2, ye2, pe, mode=pl.MergeMode.ZEROING)
        mo2 = vf.mul(do2, yo2, po, mode=pl.MergeMode.ZEROING)
        pair2 = vf.add(me2, mo2, full)
        sum01 = vf.add(pair0, pair1, full)
        sum012 = vf.add(sum01, pair2, full)
        reduced = vf.reduce_sum(sum012, full)
        vf.store_unalign(sfmg_ub, reduced, out, 1, post_update=True)
    vf.store_unalign_post(sfmg_ub, out, 0, post_update=True)
