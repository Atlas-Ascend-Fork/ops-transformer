# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Cube 侧计算。

五个 matmul 阶段，编号与 CV 同步 flag 一致::

    C1  mm1 : dP = dO @ V^T     → ``iterate_mm_k``（Dy 宽）
    C2  mm2 : S  = Q  @ K^T     → ``iterate_mm_k``（D 宽）
    C3  mm3 : dQ = dS @ K       → ``iterate_mm_dsk``
    C4  mm4 : dK = dS^T @ Q     → ``iterate_mm_n``（D 宽）
    C5  mm5 : dV = P^T  @ dO    → ``iterate_mm_n``（Dv 宽）

每个阶段只做「载入 -> matmul -> 输出」，不含分核逻辑、不含跨核同步 ——
同步全部由 :mod:`fag_kernel` 的主循环在阶段之间插入。

（1）D 轴分块：64/128 一块，192/256 两块，768 是 3 窗 × 128+128。
（2）C4/C5 驻留 L0C 时必须编译期拆 ``acc0`` / ``acc1``，不能在 ``pl.range`` 里
用 ``i==0`` 选 tensor —— PyPTO 会把 ``at`` 绑成同一块，192 的第二段 96 列
写进 acc0，flush 的 acc1 是垃圾。
（3）驻留发生在：GS1S2 的 dV（``d_align<=192``，以及非 swizzle D256 大 S）、
dK（``d_align<=128``，以及非 swizzle D192/Dv128），
BN2 wide的 dK/dV（192）与 dV（256），BN2 narrow的 dK/dV。

其余编译期分流：

（1）``d_align==768``：每窗从 GM 重装，空块 ``next()`` 仍推进
（2）GS1S2：C4(dK) 在 ``d_align>128`` 默认每块 atomic 出 GM；
  非 swizzle D192/Dv128 则驻留 N128+64，按 KV owner 切换时 flush。
  C5(dV) 在 ``d_align>192`` 默认这样；非 swizzle D256 且 S1/S2>64 由
  ``iterate_mm_pdy_gs_dv256`` 接入驻留
（3）BN2 narrow：C3 按 ``is_bn2_multiblk`` 二选一，单块写 UB、多块写 per-core 2D 条带 ws；
  C4 的 M 取这一列的 ``s2RealSize``
（4）BN2 wide：C3(dQ) 三档都写 4D ws；C4(dK) 仅 256/768写（192 驻留 L0C，flush 搬 UB 再 cast）；
  C5(dV) 仅 768 写（192/256 驻留 L0C，fixpipe 直出）。
  ND 写回的 M 按真实长度裁；NZ workspace 独占 S128 tile，MAD 的物理 M 固定128，
  post 只回写真实行。
"""

import pypto_pro.language as pl

from fag_nzout import store_nz_partial

# --- generated imports (fag_fiximports.py) ---
from fag_common import (
    CUBE_BASEM,
    CUBE_BASEN,
    D_CHUNK_MAXW,
    D_CHUNK_OFF0,
    D_CHUNK_OFF1,
    D_CHUNK_W0,
    D_CHUNK_W1,
    D_NCHUNKS,
    D768_PHYS,
    VECTOR_BASEM,
    VECTOR_BASEN,
    chunk_local,
    chunk_off,
    chunk_valid_d,
    chunk_win,
    dv_chunk_valid,
)
from fag_tiling import Tmpl
from memory.fag_buffers import (
    make_ds_l1_no_sync,
    make_ds_t_l1_no_sync,
    make_p_t_l1_no_sync,
    make_l0a_no_sync,
    make_q_l1_no_sync,
    make_do_l1_no_sync,
    make_v_l1_no_sync,
    make_k_l1_no_sync,
    make_k_nd_l1_no_sync,
)
# --- end generated imports ---


# ==================================================================
#  三种片上 matmul：K 轴累加 / N 轴写出 / N 轴驻留
# ==================================================================
def next_cube_l0a(left, slotState=0, resource_bn2s2=False):
    """Allocate exactly once at each original left.next() site.

    Target consumers can therefore use the same physical slot without reading
    private TileGroup cursor or Tile mutex fields. Other paths keep next().
    """
    if (
        not resource_bn2s2
        and template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        slotState.l0a_slot = (slotState.l0a_slot + 1) % 2
        return left[slotState.l0a_slot]
    return left.next()


def _mm_k_chunk(
    lt,
    rt,
    at,
    srcL,
    srcR,
    dOff,
    validD,
    s2Real,
    isFirst,
    mReal,
    use_unitflag=False,
    is_final=True,
    set_acc_shape=True,
):
    """MM2/MM1 的一个 K 分块：D 落在 K 轴，各块累加到同一个 acc。"""
    pl.set_validshape(lt, [mReal, validD])
    pl.set_validshape(rt, [validD, s2Real])
    if set_acc_shape:
        pl.set_validshape(at, [mReal, s2Real])
    pl.move(lt, srcL, [0, dOff])
    pl.move(rt, srcR, [dOff, 0])
    if use_unitflag:
        if isFirst:
            if is_final:
                pl.matmul(at, lt, rt, phase=pl.AccPhase.Final)
            else:
                pl.matmul(at, lt, rt, phase=pl.AccPhase.Partial)
        else:
            if is_final:
                pl.matmul_acc(at, at, lt, rt, phase=pl.AccPhase.Final)
            else:
                pl.matmul_acc(at, at, lt, rt, phase=pl.AccPhase.Partial)
    else:
        if isFirst:
            pl.matmul(at, lt, rt)
        else:
            pl.matmul_acc(at, at, lt, rt)


def _mm_n_chunk(
    left,
    right,
    acc,
    srcL,
    srcR,
    tensor_out,
    outIdx,
    mReal,
    kReal,
    dOff,
    validD,
    l1Off,
    is_first=0,
    nz_index=0,
    nz_cols=0,
    resource_bn2s2=False,
):
    """N 轴一块：next() 必推进；validD==0 不算（768 空块 / Dv 截断）。

    ``is_first==1`` 用 AtomicNone 覆盖（BN2 每个 (b,n2,s2) 的首个 s1 块 / 每个
    s1 列的首个 s2 块），否则 AtomicAdd 累加。GS1S2 恒 AtomicAdd（is_first=0）。
    """
    lt = left.next()
    rt = right.next()
    at = acc.next()
    if validD > 0:
        if not resource_bn2s2 and nz_out == 1:
            # MAD's packed column stride must match the physical NZ row pitch,
            # including S tails. Expanding only the store view is insufficient.
            mReal = CUBE_BASEM
        pl.set_validshape(lt, [mReal, kReal])
        pl.set_validshape(rt, [kReal, validD])
        pl.set_validshape(at, [mReal, validD])
        pl.move(lt, srcL)
        pl.move(rt, srcR, [0, l1Off])
        if resource_bn2s2:
            pl.matmul(at, lt, rt, phase=pl.AccPhase.Final)
            if is_first == 1:
                pl.store(
                    tensor_out,
                    at,
                    [outIdx[0], outIdx[1], outIdx[2], dOff],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicNone,
                    phase=pl.STPhase.Final,
                )
            else:
                pl.store(
                    tensor_out,
                    at,
                    [outIdx[0], outIdx[1], outIdx[2], dOff],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                    phase=pl.STPhase.Final,
                )
        else:
            pl.matmul(at, lt, rt)
            if nz_out == 1:
                store_nz_partial(tensor_out, at, nz_index, nz_cols)
            elif is_first == 1:
                pl.store(
                    tensor_out,
                    at,
                    [outIdx[0], outIdx[1], outIdx[2], dOff],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicNone,
                )
            else:
                pl.store(
                    tensor_out,
                    at,
                    [outIdx[0], outIdx[1], outIdx[2], dOff],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                )


def _mm_resident_chunk(
    left,
    right,
    at,
    srcL,
    srcR,
    mReal,
    kReal,
    dOff,
    validD,
    isAcc,
    use_unitflag=False,
    is_final=True,
    slotState=0,
    resource_bn2s2=False,
):
    """C4/C5 驻留 L0C：D 在 N 轴，跨 s1 累加进同一块。"""
    lt = next_cube_l0a(left, slotState, resource_bn2s2)
    rt = right.next()
    if not resource_bn2s2 and nz_out == 1:
        mReal = CUBE_BASEN
    pl.set_validshape(lt, [mReal, kReal])
    pl.set_validshape(rt, [kReal, validD])
    pl.set_validshape(at, [mReal, validD])
    pl.move(lt, srcL)
    pl.move(rt, srcR, [0, dOff])
    if use_unitflag:
        if isAcc == 1:
            if is_final == 1:
                pl.matmul_acc(at, at, lt, rt, phase=pl.AccPhase.Final)
            else:
                pl.matmul_acc(at, at, lt, rt, phase=pl.AccPhase.Partial)
        else:
            if is_final == 1:
                pl.matmul(at, lt, rt, phase=pl.AccPhase.Final)
            else:
                pl.matmul(at, lt, rt, phase=pl.AccPhase.Partial)
    else:
        if isAcc == 1:
            pl.matmul_acc(at, at, lt, rt)
        else:
            pl.matmul(at, lt, rt)


def _mm_n_chunk_loaded_left(
    lt,
    right,
    acc,
    srcR,
    tensor_out,
    outIdx,
    mReal,
    kReal,
    dOff,
    validD,
    l1Off,
    first_store=False,
):
    """GS dQ/dK: consume one shared L0A tile across all output-column chunks.

    Only right/acc advance here. Their empty-chunk advancement still matches the
    wide-D window schedule, while the caller keeps lt live through the last MMAD.
    """
    rt = right.next()
    at = acc.next()
    if validD > 0:
        pl.set_validshape(rt, [kReal, validD])
        pl.set_validshape(at, [mReal, validD])
        pl.move(rt, srcR, [0, l1Off])
        if template == Tmpl.BN2S2:
            pl.matmul(at, lt, rt, phase=pl.AccPhase.Final)
            if first_store:
                pl.store(
                    tensor_out,
                    at,
                    [outIdx[0], outIdx[1], outIdx[2], dOff],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicNone,
                    phase=pl.STPhase.Final,
                )
            else:
                pl.store(
                    tensor_out,
                    at,
                    [outIdx[0], outIdx[1], outIdx[2], dOff],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                    phase=pl.STPhase.Final,
                )
        else:
            pl.matmul(at, lt, rt)
            pl.store(
                tensor_out,
                at,
                [outIdx[0], outIdx[1], outIdx[2], dOff],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
            )


def _mm_resident_chunk_loaded_left(
    lt,
    right,
    at,
    srcR,
    mReal,
    kReal,
    dOff,
    validD,
    isAcc,
    use_unitflag=False,
    is_final=True,
):
    """Resident N chunk sharing a caller-owned L0A tile with the next chunk."""
    rt = right.next()
    pl.set_validshape(rt, [kReal, validD])
    pl.set_validshape(at, [mReal, validD])
    pl.move(rt, srcR, [0, dOff])
    if use_unitflag:
        # Both N regions follow the same owner boundary. C3 is always final;
        # resident C4 remains partial until its matching flush is scheduled.
        if isAcc == 1:
            if is_final == 1:
                pl.matmul_acc(at, at, lt, rt, phase=pl.AccPhase.Final)
            else:
                pl.matmul_acc(at, at, lt, rt, phase=pl.AccPhase.Partial)
        else:
            if is_final == 1:
                pl.matmul(at, lt, rt, phase=pl.AccPhase.Final)
            else:
                pl.matmul(at, lt, rt, phase=pl.AccPhase.Partial)
    else:
        if isAcc == 1:
            pl.matmul_acc(at, at, lt, rt)
        else:
            pl.matmul(at, lt, rt)


def _mm_n128_64(
    left,
    right,
    at0,
    at1,
    srcL,
    srcR,
    mReal,
    kReal,
    nReal,
    isAcc,
    use_unitflag=False,
    is_final=True,
    slotState=0,
):
    """GS D192/Dv128: two N chunks share one L0A load and double-buffer L0B.

    at0/at1 are compact views at fixed offsets 0/64KiB of one 96KiB buffer.
    Their common mutex also protects the full-width view used by full-M FIX.
    Phased C3 completes both regions each task. Phased resident dK completes
    both only when the caller has decided to flush this owner.
    """
    lt = next_cube_l0a(left, slotState)
    pl.set_validshape(lt, [mReal, kReal])
    pl.move(lt, srcL)
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        # The auto MTE1 producer has released this slot. Only the bare alias
        # enters the two MMADs, so the explicit M lock is not acquired twice.
        left_bare = make_l0a_no_sync()
        lt = left_bare[slotState.l0a_slot]
        pl.set_validshape(lt, [mReal, kReal])
        pl.system.mutex_lock(
            pipe=pl.PipeType.M, mutex_id=slotState.l0a_ids[slotState.l0a_slot]
        )
    _mm_resident_chunk_loaded_left(
        lt,
        right,
        at0,
        srcR,
        mReal,
        kReal,
        0,
        pl.min(128, nReal),
        isAcc,
        use_unitflag,
        is_final,
    )
    nTail = pl.max(pl.min(64, nReal - 128), 0)
    if nTail > 0:
        _mm_resident_chunk_loaded_left(
            lt,
            right,
            at1,
            srcR,
            mReal,
            kReal,
            128,
            nTail,
            isAcc,
            use_unitflag,
            is_final,
        )

    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        pl.system.mutex_unlock(
            pipe=pl.PipeType.M, mutex_id=slotState.l0a_ids[slotState.l0a_slot]
        )


def _store_n128_64(tensor_out, at0, at1, outIdx, mReal, nReal, use_unitflag=False):
    """Full-M chunks are contiguous; tail-M chunks need separate FIX reads."""
    if mReal == CUBE_BASEM:
        full = pl.reinterpret(at0, shape=[CUBE_BASEM, 192])
        # reinterpret deliberately does not inherit valid_shape.
        pl.set_validshape(full, [mReal, nReal])
        if use_unitflag:
            pl.store(
                tensor_out,
                full,
                [outIdx[0], outIdx[1], outIdx[2], 0],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
                phase=pl.STPhase.Final,
            )
        else:
            pl.store(
                tensor_out,
                full,
                [outIdx[0], outIdx[1], outIdx[2], 0],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
            )
    else:
        # MMAD packs each chunk at ceil16(M), irrespective of physical Rows.
        # Keep the fixed 64KiB gap out of FIX by reading the chunks separately.
        pl.set_validshape(at0, [mReal, pl.min(128, nReal)])
        if use_unitflag:
            pl.store(
                tensor_out,
                at0,
                [outIdx[0], outIdx[1], outIdx[2], 0],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
                phase=pl.STPhase.Final,
            )
        else:
            pl.store(
                tensor_out,
                at0,
                [outIdx[0], outIdx[1], outIdx[2], 0],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
            )
        nTail = pl.max(pl.min(64, nReal - 128), 0)
        if nTail > 0:
            pl.set_validshape(at1, [mReal, nTail])
            if use_unitflag:
                pl.store(
                    tensor_out,
                    at1,
                    [outIdx[0], outIdx[1], outIdx[2], 128],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                    phase=pl.STPhase.Final,
                )
            else:
                pl.store(
                    tensor_out,
                    at1,
                    [outIdx[0], outIdx[1], outIdx[2], 128],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicAdd,
                )


# ==================================================================
#  C1 / C2 —— D 在 K 轴，一块 acc，Fixpipe DualModeSplitM
# ==================================================================
def _load_score_operand(tile, tensor, batch, offset, head, column, transpose=False):
    pl.load(
        tile,
        tensor,
        [batch, offset, head, column],
        order=[3, 1] if transpose else [1, 3],
    )


def prefetch_qdo_l1(constInfo, runInfo, tensor_q, tensor_dy, q_t, do_t):
    """预取一个已解码有效任务的 Q/dO；调用者保证该 L1 槽的旧读者已释放。"""
    pl.set_validshape(q_t, [runInfo.s1RealSize, constInfo.dSize])
    pl.set_validshape(do_t, [runInfo.s1RealSize, constInfo.dvSize])
    _load_score_operand(
        q_t, tensor_q, runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q, 0, False
    )
    _load_score_operand(
        do_t, tensor_dy, runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q, 0, False
    )


def _score_k_pair(
    left,
    right,
    at,
    l_t,
    r_t,
    dimSize,
    mReal,
    nReal,
    use_unitflag=False,
    slotState=0,
    resource_bn2s2=False,
):
    # 64/128/192/256：编译期一块或两块。isFirst 不能绑 pl.range 的 i。
    lt = next_cube_l0a(left, slotState, resource_bn2s2)
    rt = right.next()
    w0 = pl.min(
        (D_CHUNK_W0_bn2s2[d_align] if resource_bn2s2 else D_CHUNK_W0[d_align]), dimSize
    )
    _mm_k_chunk(
        lt,
        rt,
        at,
        l_t,
        r_t,
        D_CHUNK_OFF0[d_align],
        w0,
        nReal,
        True,
        mReal,
        use_unitflag,
        (D_NCHUNKS[d_align] == 1) or dimSize <= D_CHUNK_OFF1[d_align],
    )
    if D_NCHUNKS[d_align] > 1:
        lt1 = next_cube_l0a(left, slotState, resource_bn2s2)
        rt1 = right.next()
        w1 = pl.max(
            pl.min(
                D_CHUNK_W1[d_align],
                dimSize - D_CHUNK_OFF1[d_align],
            ),
            0,
        )
        if w1 > 0:
            _mm_k_chunk(
                lt1,
                rt1,
                at,
                l_t,
                r_t,
                D_CHUNK_OFF1[d_align],
                w1,
                nReal,
                False,
                mReal,
                use_unitflag,
            )


def _score_k_windows(
    runInfo,
    tensor_l,
    tensor_r,
    l_t,
    r_t,
    left,
    right,
    at,
    dimSize,
    mReal,
    slotState=0,
    resource_bn2s2=False,
):
    for i in pl.range(0, D_NCHUNKS[d_align]):
        dOff = chunk_off(i, resource_bn2s2)
        localOff = chunk_local(i, resource_bn2s2)
        validD = chunk_valid_d(dimSize, i, resource_bn2s2)
        if i % 2 == 0:
            winOff = chunk_win(i)
            winW = pl.max(pl.min(D768_PHYS, dimSize - winOff), 0)
            pl.set_validshape(l_t, [runInfo.s1RealSize, winW])
            pl.set_validshape(r_t, [winW, runInfo.s2RealSize])
            if winW > 0:
                _load_score_operand(
                    l_t,
                    tensor_l,
                    runInfo.idx0Q,
                    runInfo.queryOffset,
                    runInfo.idx2Q,
                    winOff,
                    False,
                )
                _load_score_operand(
                    r_t,
                    tensor_r,
                    runInfo.idx0KV,
                    runInfo.keyOffset,
                    runInfo.idx2KV,
                    winOff,
                    True,
                )
        lt = next_cube_l0a(left, slotState, resource_bn2s2)
        rt = right.next()
        if validD > 0:
            _mm_k_chunk(
                lt,
                rt,
                at,
                l_t,
                r_t,
                localOff,
                validD,
                runInfo.s2RealSize,
                i == 0,
                mReal,
                resource_bn2s2,
                dOff + validD >= dimSize,
            )


def iterate_mm_k(
    constInfo,
    runInfo,
    tensor_l,
    tensor_r,
    l_t,
    r_l1,
    left,
    right,
    acc,
    mm_ub,
    loadKV,
    dimSize,
    is_value=False,
    loadQDo=1,
    copyNext=0,
    nextInfo=0,
    next_l_t=0,
    slotState=0,
    qSlot=0,
    kSlot=0,
):
    """C1/C2 共用：``[S1, dim] @ [dim, S2]``，dim 是 D 或 Dv。

    GS 的普通/确定性、swizzle/非 swizzle 都用实际 M；PTO 在
    DualModeSplitM 前将有效 M 补偶数，两个 AIV 按 ceil(M/2) 分界，
    第二个 AIV 只消费 floor(M/2) 行。BN2 single D256 dense 非确族也采用
    该映射，并由 schedule、NZ+1 物理 stride 和最终输出共同遵守；其它族保持原值。
    """
    use_unitflag = (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and (
                    (deter == 1 and layout == 1 and (deter_family == 12))
                    or (
                        deter == 0
                        and layout == 0
                        and (mask_mode == 0 or mask_mode == 3)
                        and (sink == 0)
                    )
                )
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    )
    r_t = r_l1.current()
    mReal = CUBE_BASEM
    if template == Tmpl.BN2GS1S2 or (
        template == Tmpl.BN2
        and d_align == 256
        and dv_align == 256
        and is_bn2_multiblk == 0
        and deter == 0
        and mask_mode == 0
        and sink == 0
        and swizzle == 0
        and nz_out == 0
    ):
        mReal = runInfo.s1RealSize
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ) and not is_value:
        r_t = r_l1[kSlot]
    elif template == Tmpl.BN2GS1S2 and d_align <= 256 and not is_value:
        # K advances on KV changes, not on every Q task or next-task Q/dO prefetch.
        # The independent C3 cursor follows the same loadKV sequence. One-ahead
        # paths retain current/previous K; remaining two-round swizzle paths
        # consume their old C3 before loading the current K.
        if loadKV == 1:
            r_t = r_l1.next()
    if d_align < 512:
        pl.set_validshape(l_t, [runInfo.s1RealSize, dimSize])
        pl.set_validshape(r_t, [dimSize, runInfo.s2RealSize])
        if loadQDo == 1:
            _load_score_operand(
                l_t,
                tensor_l,
                runInfo.idx0Q,
                runInfo.queryOffset,
                runInfo.idx2Q,
                0,
                False,
            )
        if (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and (
                (
                    swizzle == 1
                    and deter == 0
                    and layout == 0
                    and (mask_mode == 0 or mask_mode == 3)
                    and sink == 0
                )
                or (
                    deter == 1
                    and layout == 0
                    and mask_mode == 0
                    and sink == 0
                    and (
                        (swizzle == 1 and deter_family == 1)
                        or (swizzle == 0 and deter_family == 21)
                    )
                )
            )
        ):
            # Match ASC: consume current Q/dO after its optional first load,
            # before current K/V and next-task Q/dO are issued on MTE2.
            if is_value:
                left_bare = make_do_l1_no_sync()
                leftMutex = slotState.do_ids[qSlot]
            else:
                left_bare = make_q_l1_no_sync()
                leftMutex = slotState.q_ids[qSlot]
            l_t = left_bare[qSlot]
            pl.set_validshape(l_t, [runInfo.s1RealSize, dimSize])
            pl.system.mutex_lock(pipe=pl.PipeType.MTE1, mutex_id=leftMutex)
        if loadKV == 1:
            _load_score_operand(
                r_t,
                tensor_r,
                runInfo.idx0KV,
                runInfo.keyOffset,
                runInfo.idx2KV,
                0,
                True,
            )
        if (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and (
                (
                    swizzle == 1
                    and deter == 0
                    and layout == 0
                    and (mask_mode == 0 or mask_mode == 3)
                    and sink == 0
                )
                or (
                    swizzle == 0
                    and deter == 1
                    and (layout == 1)
                    and (deter_family == 13)
                )
                or (
                    deter == 1
                    and layout == 0
                    and mask_mode == 0
                    and sink == 0
                    and (
                        (swizzle == 1 and deter_family == 1)
                        or (swizzle == 0 and deter_family == 21)
                    )
                )
            )
        ):
            # Match AscendC's MTE2 issue order: current K then next Q for
            # C2, current V then next dO for C1. The current left operand
            # was loaded at startup or by the preceding valid task.
            if copyNext:
                pl.set_validshape(next_l_t, [nextInfo.s1RealSize, dimSize])
                _load_score_operand(
                    next_l_t,
                    tensor_l,
                    nextInfo.idx0Q,
                    nextInfo.queryOffset,
                    nextInfo.idx2Q,
                    0,
                    False,
                )
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        # K/V consumption starts after next Q/dO prefetch was issued. V has
        # one fixed slot; K uses the task's saved physical writer slot.
        if is_value:
            right_bare = make_v_l1_no_sync()
            r_t = right_bare[0]
            rightMutex = slotState.v_id
        else:
            right_bare = make_k_l1_no_sync()
            r_t = right_bare[kSlot]
            rightMutex = slotState.k_ids[kSlot]
        pl.set_validshape(r_t, [dimSize, runInfo.s2RealSize])
        pl.system.mutex_lock(pipe=pl.PipeType.MTE1, mutex_id=rightMutex)
    at = acc.next()
    if d_align >= 512:
        _score_k_windows(
            runInfo,
            tensor_l,
            tensor_r,
            l_t,
            r_t,
            left,
            right,
            at,
            dimSize,
            mReal,
            slotState,
        )
    elif (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            swizzle == 0
            or (swizzle == 1 and deter == 1 and (layout == 1) and (deter_family == 12))
            or (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                swizzle == 1
                and deter == 0
                and not (
                    layout == 0 and (mask_mode == 0 or mask_mode == 3) and (sink == 0)
                )
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ) and is_value:
        # C1 reduces over Dv128 independently of the D192 K split used by C2.
        lt = next_cube_l0a(left, slotState)
        rt = right.next()
        _mm_k_chunk(
            lt,
            rt,
            at,
            l_t,
            r_t,
            0,
            dimSize,
            runInfo.s2RealSize,
            True,
            mReal,
            use_unitflag,
        )
    elif (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ) and not is_value:
        # C2 matches AscendC's K128+64 split. The existing 128x128 L0A/L0B
        # tiles fit the first chunk; unrelated paths keep their original split below.
        lt = next_cube_l0a(left, slotState)
        rt = right.next()
        w0 = pl.min(128, dimSize)
        _mm_k_chunk(
            lt,
            rt,
            at,
            l_t,
            r_t,
            0,
            w0,
            runInfo.s2RealSize,
            True,
            mReal,
            use_unitflag,
            dimSize <= 128,
        )
        lt1 = next_cube_l0a(left, slotState)
        rt1 = right.next()
        w1 = pl.max(pl.min(64, dimSize - 128), 0)
        if w1 > 0:
            _mm_k_chunk(
                lt1,
                rt1,
                at,
                l_t,
                r_t,
                128,
                w1,
                runInfo.s2RealSize,
                False,
                mReal,
                use_unitflag,
            )
    else:
        _score_k_pair(
            left,
            right,
            at,
            l_t,
            r_t,
            dimSize,
            mReal,
            runInfo.s2RealSize,
            use_unitflag,
            slotState,
        )
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE1, mutex_id=rightMutex)
    if use_unitflag:
        pl.move(
            mm_ub,
            at,
            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
            phase=pl.STPhase.Final,
        )
    else:
        pl.move(mm_ub, at, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)

    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        # Match ASC's outer left-operand lifetime through the FIX issue.
        pl.system.mutex_unlock(pipe=pl.PipeType.MTE1, mutex_id=leftMutex)


def iterate_mm_qk(
    constInfo,
    runInfo,
    tensor_q,
    tensor_k,
    q_t,
    k_l1,
    left,
    right,
    acc,
    mm2_ub,
    loadKV,
    loadQDo=1,
    copyNext=0,
    nextInfo=0,
    next_l_t=0,
    slotState=0,
    qSlot=0,
    kSlot=0,
):
    iterate_mm_k(
        constInfo,
        runInfo,
        tensor_q,
        tensor_k,
        q_t,
        k_l1,
        left,
        right,
        acc,
        mm2_ub,
        loadKV,
        constInfo.dSize,
        False,
        loadQDo,
        copyNext,
        nextInfo,
        next_l_t,
        slotState,
        qSlot,
        kSlot,
    )


def iterate_mm_dyv(
    constInfo,
    runInfo,
    tensor_dy,
    tensor_v,
    do_t,
    v_l1,
    left,
    right,
    acc,
    mm1_ub,
    loadKV,
    loadQDo=1,
    copyNext=0,
    nextInfo=0,
    next_l_t=0,
    slotState=0,
    qSlot=0,
    kSlot=0,
):
    iterate_mm_k(
        constInfo,
        runInfo,
        tensor_dy,
        tensor_v,
        do_t,
        v_l1,
        left,
        right,
        acc,
        mm1_ub,
        loadKV,
        constInfo.dvSize,
        True,
        loadQDo,
        copyNext,
        nextInfo,
        next_l_t,
        slotState,
        qSlot,
        kSlot,
    )


# ==================================================================
#  C3 —— D 在 N 轴，各块 atomic 出不相交的 dQ 列
# ==================================================================
def iterate_mm_dsk(
    constInfo,
    runInfo,
    tensor_k,
    tensor_dq_ws,
    loadKV,
    ds_l1,
    k_l1,
    left,
    right,
    acc,
    mm1_ub,
    cBlockIdx,
    acc_n64=None,
    slotState=0,
    kSlot=0,
):
    """C3：``dQ = dS @ K``。

    GS1S2 各块 atomic 出 4D workspace；BN2 单块后写 UB 或 2D workspace
    （``is_bn2_multiblk``）。``mm1_ub`` / ``cBlockIdx`` 只在 BN2 用到。
    """
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        # The caller waits for both AIV dS-ready signals and releases dS only
        # after C4's L1 read. Keep that CV lifetime, without an extra L1 mutex.
        ds_no_sync = make_ds_l1_no_sync()
        ds_l1 = ds_no_sync.current()
        pl.set_validshape(ds_l1, [runInfo.s1RealSize, runInfo.s2RealSize])
    k_t = k_l1.current()
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        k_t = k_l1[kSlot]
    elif template == Tmpl.BN2GS1S2 and d_align <= 256:
        if loadKV == 1:
            k_t = k_l1.next()
    outIdx = [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q]
    if d_align < 512:
        pl.set_validshape(k_t, [runInfo.s2RealSize, constInfo.dSize])
        if loadKV == 1 and not (template == Tmpl.BN2GS1S2 and d_align <= 256):
            pl.load(
                k_t,
                tensor_k,
                [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, 0],
                order=[1, 3],
            )
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            swizzle == 0
            or (swizzle == 1 and deter == 1 and (layout == 1) and (deter_family == 12))
            or (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                swizzle == 1
                and deter == 0
                and not (
                    layout == 0 and (mask_mode == 0 or mask_mode == 3) and (sink == 0)
                )
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        at0 = acc.next()
        at1 = acc_n64.current()
        use_unitflag = (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and (
                (
                    swizzle == 1
                    and (
                        (deter == 1 and layout == 1 and (deter_family == 12))
                        or (
                            deter == 0
                            and layout == 0
                            and (mask_mode == 0 or mask_mode == 3)
                            and sink == 0
                        )
                    )
                )
                or (
                    deter == 1
                    and layout == 0
                    and mask_mode == 0
                    and sink == 0
                    and (
                        (swizzle == 1 and deter_family == 1)
                        or (swizzle == 0 and deter_family == 21)
                    )
                )
            )
        )
        if (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and (
                (
                    swizzle == 1
                    and deter == 0
                    and layout == 0
                    and (mask_mode == 0 or mask_mode == 3)
                    and sink == 0
                )
                or (
                    deter == 1
                    and layout == 0
                    and mask_mode == 0
                    and sink == 0
                    and (
                        (swizzle == 1 and deter_family == 1)
                        or (swizzle == 0 and deter_family == 21)
                    )
                )
            )
        ):
            k_bare = make_k_nd_l1_no_sync()
            k_t = k_bare[kSlot]
            pl.set_validshape(k_t, [runInfo.s2RealSize, constInfo.dSize])
            pl.system.mutex_lock(pipe=pl.PipeType.MTE1, mutex_id=slotState.k_ids[kSlot])
        _mm_n128_64(
            left,
            right,
            at0,
            at1,
            ds_l1,
            k_t,
            runInfo.s1RealSize,
            runInfo.s2RealSize,
            constInfo.dSize,
            0,
            use_unitflag,
            True,
            slotState,
        )
        if (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and (
                (
                    swizzle == 1
                    and deter == 0
                    and layout == 0
                    and (mask_mode == 0 or mask_mode == 3)
                    and sink == 0
                )
                or (
                    deter == 1
                    and layout == 0
                    and mask_mode == 0
                    and sink == 0
                    and (
                        (swizzle == 1 and deter_family == 1)
                        or (swizzle == 0 and deter_family == 21)
                    )
                )
            )
        ):
            pl.system.mutex_unlock(
                pipe=pl.PipeType.MTE1, mutex_id=slotState.k_ids[kSlot]
            )
        _store_n128_64(
            tensor_dq_ws,
            at0,
            at1,
            outIdx,
            runInfo.s1RealSize,
            constInfo.dSize,
            use_unitflag,
        )
        return
    if template == Tmpl.BN2GS1S2 and d_align > 128:
        # Splitting output D changes K's columns, but dS is identical in each MMAD.
        lt_dq = left.next()
        pl.set_validshape(lt_dq, [runInfo.s1RealSize, runInfo.s2RealSize])
        pl.move(lt_dq, ds_l1)
    for i in pl.range(0, D_NCHUNKS[d_align]):
        dOff = chunk_off(i)
        localOff = chunk_local(i)
        validD = chunk_valid_d(constInfo.dSize, i)
        if d_align >= 512:
            if i % 2 == 0:
                winOff = chunk_win(i)
                winW = pl.max(pl.min(D768_PHYS, constInfo.dSize - winOff), 0)
                pl.set_validshape(k_t, [runInfo.s2RealSize, winW])
                if winW > 0:
                    pl.load(
                        k_t,
                        tensor_k,
                        [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, winOff],
                        order=[1, 3],
                    )
        if template == Tmpl.BN2 and d_align <= 128:
            lt = left.next()
            rt = right.next()
            at = acc.next()
            if validD > 0:
                pl.set_validshape(lt, [CUBE_BASEM, runInfo.s2RealSize])
                pl.set_validshape(rt, [runInfo.s2RealSize, validD])
                pl.set_validshape(at, [CUBE_BASEM, validD])
                pl.move(lt, ds_l1)
                pl.move(rt, k_t, [0, localOff])
                pl.matmul(at, lt, rt)
                if is_bn2_multiblk == 1:
                    rowOff = (
                        cBlockIdx * constInfo.s1Outer + runInfo.s1oIdx
                    ) * CUBE_BASEM
                    if runInfo.dqFirst == 1:
                        pl.store(
                            tensor_dq_ws,
                            at,
                            [rowOff, 0],
                            atomic=pl.AtomicType.AtomicNone,
                        )
                    else:
                        pl.store(
                            tensor_dq_ws,
                            at,
                            [rowOff, 0],
                            atomic=pl.AtomicType.AtomicAdd,
                        )
                else:
                    pl.set_validshape(mm1_ub, [VECTOR_BASEM, VECTOR_BASEN])
                    pl.move(
                        mm1_ub,
                        at,
                        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
                    )
        elif template == Tmpl.BN2:
            # BN2 wide：dQ 每个 D 分块直接写 4D fp32 workspace；dQ[s1] 在核内
            # 按 s2 累加，故本 s1 列的首个 s2 贡献（dqFirst）用 AtomicNone 覆盖，
            # 其余 AtomicAdd，不需要全局清零；M 必须按 s1RealSize 裁
            lt = left.next()
            rt = right.next()
            at = acc.next()
            if validD > 0:
                pl.set_validshape(lt, [runInfo.s1RealSize, runInfo.s2RealSize])
                pl.set_validshape(rt, [runInfo.s2RealSize, validD])
                pl.set_validshape(at, [runInfo.s1RealSize, validD])
                pl.move(lt, ds_l1)
                pl.move(rt, k_t, [0, localOff])
                pl.matmul(at, lt, rt)
                if runInfo.dqFirst == 1:
                    pl.store(
                        tensor_dq_ws,
                        at,
                        [outIdx[0], outIdx[1], outIdx[2], dOff],
                        order=[1, 3],
                        atomic=pl.AtomicType.AtomicNone,
                    )
                else:
                    pl.store(
                        tensor_dq_ws,
                        at,
                        [outIdx[0], outIdx[1], outIdx[2], dOff],
                        order=[1, 3],
                        atomic=pl.AtomicType.AtomicAdd,
                    )
        elif template == Tmpl.BN2GS1S2 and d_align > 128:
            if d_align >= 512 or validD > 0:
                _mm_n_chunk_loaded_left(
                    lt_dq,
                    right,
                    acc,
                    k_t,
                    tensor_dq_ws,
                    outIdx,
                    runInfo.s1RealSize,
                    runInfo.s2RealSize,
                    dOff,
                    validD,
                    localOff,
                )
        elif d_align >= 512 or validD > 0:
            _mm_n_chunk(
                left,
                right,
                acc,
                ds_l1,
                k_t,
                tensor_dq_ws,
                outIdx,
                runInfo.s1RealSize,
                runInfo.s2RealSize,
                dOff,
                validD,
                localOff,
                0,
                runInfo.nzQTile,
                constInfo.dAlign,
            )


# ==================================================================
#  C4 / C5 —— D 在 N 轴：GS1S2 的 C4 ≤128 与非 swizzle D192/Dv128 驻留，
#  C5 ≤192 与非 swizzle D256 大 S 驻留，其余 atomic 出 GM；BN2 192 dK/dV 都驻留，
#  256 档只驻留 dV，dK 256/768 与 dV 768 走 atomic 出 GM
# ==================================================================
def iterate_mm_n(
    constInfo,
    runInfo,
    srcL,
    srcR,
    left,
    right,
    acc0,
    tensor_out,
    tensor_src,
    dimSize,
    srcIdx,
    outIdx,
    mReal,
    is_first=0,
    reuse_left=False,
):
    """C4/C5 写出路径：GS1S2 的 C4(dK) ``d_align>128``、C5(dV) ``d_align>192``；
    BN2 的 C4(dK) 256/768、C5(dV) 768。每块 atomic 出 GM。

    ``reuse_left`` only enables GS dK's AtomicAdd path: dS^T is loaded once
    and kept in L0A across every N chunk, while L0B/L0C keep their own cursors.

    ``tensor_out`` / ``tensor_src`` 必须是真 Tensor。新前端不允许把
    ``None`` 默认值再往下传 —— 驻留路径走 ``iterate_mm_n_l0c``。
    """
    kReal = runInfo.s1RealSize
    if template == Tmpl.BN2GS1S2 and reuse_left:
        # dS^T is independent of the output D chunk, including wide-D windows.
        lt_dk = left.next()
        pl.set_validshape(lt_dk, [mReal, kReal])
        pl.move(lt_dk, srcL)
    for i in pl.range(0, D_NCHUNKS[d_align]):
        dOff = chunk_off(i)
        localOff = chunk_local(i)
        validD = chunk_valid_d(dimSize, i)
        if d_align >= 512:
            if i % 2 == 0:
                winOff = chunk_win(i)
                winW = pl.max(pl.min(D768_PHYS, dimSize - winOff), 0)
                pl.set_validshape(srcR, [runInfo.s1RealSize, winW])
                if winW > 0:
                    pl.load(
                        srcR,
                        tensor_src,
                        [srcIdx[0], srcIdx[1], srcIdx[2], winOff],
                        order=[1, 3],
                    )
        if d_align >= 512 or validD > 0:
            if template == Tmpl.BN2GS1S2 and reuse_left:
                _mm_n_chunk_loaded_left(
                    lt_dk,
                    right,
                    acc0,
                    srcR,
                    tensor_out,
                    outIdx,
                    mReal,
                    kReal,
                    dOff,
                    validD,
                    localOff,
                )
            else:
                _mm_n_chunk(
                    left,
                    right,
                    acc0,
                    srcL,
                    srcR,
                    tensor_out,
                    outIdx,
                    mReal,
                    kReal,
                    dOff,
                    validD,
                    localOff,
                    is_first,
                    runInfo.nzKvTile,
                    constInfo.dAlign,
                )


def _resident_n_pair(
    srcL,
    srcR,
    left,
    right,
    acc0,
    acc1,
    isAcc,
    dimSize,
    mReal,
    kReal,
    resource_bn2s2=False,
    flush=True,
):
    off0 = D_CHUNK_OFF0[d_align]
    w0 = pl.max(
        pl.min(
            (D_CHUNK_W0_bn2s2[d_align] if resource_bn2s2 else D_CHUNK_W0[d_align]),
            dimSize - off0,
        ),
        0,
    )
    if w0 > 0:
        _mm_resident_chunk(
            left,
            right,
            acc0.current(),
            srcL,
            srcR,
            mReal,
            kReal,
            off0,
            w0,
            isAcc,
            resource_bn2s2,
            flush,
            0,
            resource_bn2s2,
        )
    if D_NCHUNKS[d_align] > 1:
        off1 = D_CHUNK_OFF1[d_align]
        w1 = pl.max(
            pl.min(
                D_CHUNK_W1[d_align],
                dimSize - off1,
            ),
            0,
        )
        if w1 > 0:
            _mm_resident_chunk(
                left,
                right,
                acc1.current(),
                srcL,
                srcR,
                mReal,
                kReal,
                off1,
                w1,
                isAcc,
                resource_bn2s2,
                flush,
                0,
                resource_bn2s2,
            )


def iterate_mm_n_l0c(
    constInfo,
    runInfo,
    srcL,
    srcR,
    left,
    right,
    acc0,
    acc1,
    isAcc,
    dimSize,
    mReal,
    slotState=0,
):
    """
    C4/C5 驻留 L0C。不接收 Tensor 参数。

    acc0/acc1 必须是编译期两路。D=64/128 一块；BN2 D192 切 96+96；
    GS D192/Dv128 的 dK 切 128+64；D256 C5(dV) 切 128+128。
    """
    kReal = runInfo.s1RealSize
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            swizzle == 0
            or (swizzle == 1 and deter == 1 and (layout == 1) and (deter_family == 12))
            or (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                swizzle == 1
                and deter == 0
                and not (
                    layout == 0 and (mask_mode == 0 or mask_mode == 3) and (sink == 0)
                )
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        # C5 has its own Dv128 path. Both dK N chunks remain resident together.
        use_unitflag = (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and (
                (
                    swizzle == 1
                    and (
                        (deter == 1 and layout == 1 and (deter_family == 12))
                        or (
                            deter == 0
                            and layout == 0
                            and (mask_mode == 0 or mask_mode == 3)
                            and sink == 0
                        )
                        or (
                            deter == 0
                            and not (
                                layout == 0
                                and (mask_mode == 0 or mask_mode == 3)
                                and (sink == 0)
                            )
                        )
                    )
                )
                or (
                    deter == 1
                    and layout == 0
                    and mask_mode == 0
                    and sink == 0
                    and (
                        (swizzle == 1 and deter_family == 1)
                        or (swizzle == 0 and deter_family == 21)
                    )
                )
            )
        )
        _mm_n128_64(
            left,
            right,
            acc0.current(),
            acc1.current(),
            srcL,
            srcR,
            mReal,
            kReal,
            dimSize,
            isAcc,
            use_unitflag,
            runInfo.flushDkv,
            slotState,
        )
        return
    _resident_n_pair(srcL, srcR, left, right, acc0, acc1, isAcc, dimSize, mReal, kReal)


def iterate_mm_dsq(
    constInfo,
    runInfo,
    ds_t_l1,
    q_t,
    left,
    right,
    acc0,
    acc1=None,
    isAcc=0,
    tensor_out=None,
    tensor_src=None,
    slotState=0,
    qSlot=0,
):
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        ds_t_no_sync = make_ds_t_l1_no_sync()
        ds_t_l1 = ds_t_no_sync.current()
        pl.set_validshape(ds_t_l1, [runInfo.s2RealSize, runInfo.s1RealSize])
    # M 是 s2（dS^T 的行）。两档都必须用 s2RealSize，不能钉 CUBE_BASEN：
    mReal = runInfo.s2RealSize
    if (template == Tmpl.BN2 and d_align == 192) or (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            swizzle == 0
            or (swizzle == 1 and deter == 1 and (layout == 1) and (deter_family == 12))
            or (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                swizzle == 1
                and deter == 0
                and not (
                    layout == 0 and (mask_mode == 0 or mask_mode == 3) and (sink == 0)
                )
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        # dK stays in L0C across tasks for the same owner. GS D192/Dv128
        # uses flush_dk; BN2 uses flush_dk_l0c_to_ub. Flush tokens stay outside.
        if (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and (
                (
                    swizzle == 1
                    and deter == 0
                    and layout == 0
                    and (mask_mode == 0 or mask_mode == 3)
                    and sink == 0
                )
                or (
                    deter == 1
                    and layout == 0
                    and mask_mode == 0
                    and sink == 0
                    and (
                        (swizzle == 1 and deter_family == 1)
                        or (swizzle == 0 and deter_family == 21)
                    )
                )
            )
        ):
            q_bare = make_q_l1_no_sync()
            q_t = q_bare[qSlot]
            pl.set_validshape(q_t, [runInfo.s1RealSize, constInfo.dSize])
            pl.system.mutex_lock(pipe=pl.PipeType.MTE1, mutex_id=slotState.q_ids[qSlot])
        iterate_mm_n_l0c(
            constInfo,
            runInfo,
            ds_t_l1,
            q_t,
            left,
            right,
            acc0,
            acc1,
            isAcc,
            constInfo.dSize,
            mReal,
            slotState,
        )
        if (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and (
                (
                    swizzle == 1
                    and deter == 0
                    and layout == 0
                    and (mask_mode == 0 or mask_mode == 3)
                    and sink == 0
                )
                or (
                    deter == 1
                    and layout == 0
                    and mask_mode == 0
                    and sink == 0
                    and (
                        (swizzle == 1 and deter_family == 1)
                        or (swizzle == 0 and deter_family == 21)
                    )
                )
            )
        ):
            pl.system.mutex_unlock(
                pipe=pl.PipeType.MTE1, mutex_id=slotState.q_ids[qSlot]
            )
        return
    if template == Tmpl.BN2 and d_align > 128:
        # BN2 wide 的 256/768：dK 每块 atomic 出 workspace
        iterate_mm_n(
            constInfo,
            runInfo,
            ds_t_l1,
            q_t,
            left,
            right,
            acc0,
            tensor_out,
            tensor_src,
            constInfo.dSize,
            [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q],
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV],
            mReal,
            1 - isAcc,
        )
        return
    if d_align > 128:
        # GS1S2 192/256/768：dK 每块 atomic 出 workspace
        iterate_mm_n(
            constInfo,
            runInfo,
            ds_t_l1,
            q_t,
            left,
            right,
            acc0,
            tensor_out,
            tensor_src,
            constInfo.dSize,
            [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q],
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV],
            mReal,
            0,
            True,
        )
    else:
        # GS1S2 64/128 与 BN2 narrow：dK 常驻 L0C（单分块）
        iterate_mm_n_l0c(
            constInfo,
            runInfo,
            ds_t_l1,
            q_t,
            left,
            right,
            acc0,
            acc1,
            isAcc,
            constInfo.dSize,
            mReal,
        )


def iterate_mm_pdy(
    constInfo,
    runInfo,
    p_t_l1,
    do_t,
    left,
    right,
    acc0,
    acc1=None,
    isAcc=0,
    tensor_out=None,
    tensor_src=None,
    slotState=0,
):
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        # P-ready/P-free surround this read; the Vector UB source retains its
        # own V/MTE3 mutex through the unchanged writer path.
        p_t_no_sync = make_p_t_l1_no_sync()
        p_t_l1 = p_t_no_sync.current()
        pl.set_validshape(p_t_l1, [runInfo.s2RealSize, runInfo.s1RealSize])
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            swizzle == 0
            or (swizzle == 1 and deter == 1 and (layout == 1) and (deter_family == 12))
            or (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                swizzle == 1
                and deter == 0
                and not (
                    layout == 0 and (mask_mode == 0 or mask_mode == 3) and (sink == 0)
                )
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        # One independent Dv128 output tile; D192 remains split for dQ/dK.
        use_unitflag = (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and (
                (
                    swizzle == 1
                    and (
                        (deter == 1 and layout == 1 and (deter_family == 12))
                        or (
                            deter == 0
                            and layout == 0
                            and (mask_mode == 0 or mask_mode == 3)
                            and sink == 0
                        )
                        or (
                            deter == 0
                            and not (
                                layout == 0
                                and (mask_mode == 0 or mask_mode == 3)
                                and (sink == 0)
                            )
                        )
                    )
                )
                or (
                    deter == 1
                    and layout == 0
                    and mask_mode == 0
                    and sink == 0
                    and (
                        (swizzle == 1 and deter_family == 1)
                        or (swizzle == 0 and deter_family == 21)
                    )
                )
            )
        )
        _mm_resident_chunk(
            left,
            right,
            acc0.current(),
            p_t_l1,
            do_t,
            runInfo.s2RealSize,
            runInfo.s1RealSize,
            0,
            constInfo.dvSize,
            isAcc,
            use_unitflag,
            runInfo.flushDkv,
            slotState,
        )
        return
    if template == Tmpl.BN2 and (d_align == 192 or d_align == 256):
        # dV 无 scale，直接驻留 L0C，flush 时 fixpipe 一次出 f16
        iterate_mm_n_l0c(
            constInfo,
            runInfo,
            p_t_l1,
            do_t,
            left,
            right,
            acc0,
            acc1,
            isAcc,
            constInfo.dvSize,
            runInfo.s2RealSize,
        )
        return
    if template == Tmpl.BN2 and d_align >= 512:
        # BN2 768：dV 与 dK 同口径（同 (b,n2,s2) 首块 AtomicNone）写 4D
        # workspace，M 按 s2RealSize 裁 —— 该 workspace 的 S2 轴没有 128 行 pad。
        iterate_mm_n(
            constInfo,
            runInfo,
            p_t_l1,
            do_t,
            left,
            right,
            acc0,
            tensor_out,
            tensor_src,
            constInfo.dvSize,
            [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q],
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV],
            runInfo.s2RealSize,
            1 - isAcc,
        )
        return
    if d_align > 192:
        # GS1S2 256/768：dV 与 dK 同口径，每块 atomic 出 workspace
        # （pre 已清零，故恒 AtomicAdd）。
        iterate_mm_n(
            constInfo,
            runInfo,
            p_t_l1,
            do_t,
            left,
            right,
            acc0,
            tensor_out,
            tensor_src,
            constInfo.dvSize,
            [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q],
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV],
            runInfo.s2RealSize,
        )
    else:
        # GS1S2 64/128/192：dV 常驻 L0C，s2 切换时 flush_dv 累加回 ws
        iterate_mm_n_l0c(
            constInfo,
            runInfo,
            p_t_l1,
            do_t,
            left,
            right,
            acc0,
            acc1,
            isAcc,
            constInfo.dvSize,
            runInfo.s2RealSize,
        )


def iterate_mm_pdy_gs_dv256(
    constInfo,
    runInfo,
    p_t_l1,
    do_t,
    left,
    right,
    acc,
    dv_acc0,
    dv_acc1,
    tensor_dv_ws,
    tensor_dy,
):
    """GS D256 C5: retain dV across tasks with the same KV destination.

    Call only from the non-swizzle D256 specialization. Match AscendC's
    IsDqkvSplitNEnabled gate using launch-wide S1/S2; a TND batch tail must
    not change the storage policy while a resident accumulator is live.
    """
    if constInfo.s1Size > 64 and constInfo.s2Size > 64:
        iterate_mm_n_l0c(
            constInfo,
            runInfo,
            p_t_l1,
            do_t,
            left,
            right,
            dv_acc0,
            dv_acc1,
            runInfo.dkvAcc,
            constInfo.dvSize,
            runInfo.s2RealSize,
        )
    else:
        iterate_mm_pdy(
            constInfo,
            runInfo,
            p_t_l1,
            do_t,
            left,
            right,
            acc,
            acc,
            0,
            tensor_dv_ws,
            tensor_dy,
        )


def flush_dk(constInfo, runInfo, tensor_dk_ws, dk_acc0, dk_acc1):
    """Write resident GS dK, including the N128+64 D192/Dv128 allocation."""
    i0 = runInfo.idx0KV
    i1 = runInfo.keyOffset
    i2 = runInfo.idx2KV
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            swizzle == 0
            or (swizzle == 1 and deter == 1 and (layout == 1) and (deter_family == 12))
            or (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                swizzle == 1
                and deter == 0
                and not (
                    layout == 0 and (mask_mode == 0 or mask_mode == 3) and (sink == 0)
                )
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        use_unitflag = (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and (
                (
                    swizzle == 1
                    and (
                        (deter == 1 and layout == 1 and (deter_family == 12))
                        or (
                            deter == 0
                            and layout == 0
                            and (mask_mode == 0 or mask_mode == 3)
                            and sink == 0
                        )
                        or (
                            deter == 0
                            and not (
                                layout == 0
                                and (mask_mode == 0 or mask_mode == 3)
                                and (sink == 0)
                            )
                        )
                    )
                )
                or (
                    deter == 1
                    and layout == 0
                    and mask_mode == 0
                    and sink == 0
                    and (
                        (swizzle == 1 and deter_family == 1)
                        or (swizzle == 0 and deter_family == 21)
                    )
                )
            )
        )
        _store_n128_64(
            tensor_dk_ws,
            dk_acc0.current(),
            dk_acc1.current(),
            [i0, i1, i2],
            runInfo.s2RealSize,
            constInfo.dSize,
            use_unitflag,
        )
        return
    if nz_out == 1:
        store_nz_partial(
            tensor_dk_ws, dk_acc0.current(), runInfo.nzKvTile, constInfo.dAlign
        )
    else:
        pl.store(
            tensor_dk_ws,
            dk_acc0.current(),
            [i0, i1, i2, D_CHUNK_OFF0[d_align]],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicAdd,
        )
    if D_NCHUNKS[d_align] > 1:
        pl.store(
            tensor_dk_ws,
            dk_acc1.current(),
            [i0, i1, i2, D_CHUNK_OFF1[d_align]],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicAdd,
        )


def flush_dv(constInfo, runInfo, tensor_dv_ws, dv_acc0, dv_acc1):
    """把 GS1S2 驻留 dV 写回 GM；D256 调用方还需检查大 S 门槛。"""
    i0 = runInfo.idx0KV
    i1 = runInfo.keyOffset
    i2 = runInfo.idx2KV
    if (
        template == Tmpl.BN2GS1S2
        and d_align == 192
        and dv_align == 128
        and (
            swizzle == 0
            or (swizzle == 1 and deter == 1 and (layout == 1) and (deter_family == 12))
            or (
                swizzle == 1
                and deter == 0
                and layout == 0
                and (mask_mode == 0 or mask_mode == 3)
                and sink == 0
            )
            or (
                swizzle == 1
                and deter == 0
                and not (
                    layout == 0 and (mask_mode == 0 or mask_mode == 3) and (sink == 0)
                )
            )
            or (
                deter == 1
                and layout == 0
                and mask_mode == 0
                and sink == 0
                and (
                    (swizzle == 1 and deter_family == 1)
                    or (swizzle == 0 and deter_family == 21)
                )
            )
        )
    ):
        # acc1 aliases acc0 in this tiling key and must not be flushed again.
        use_unitflag = (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and (
                (
                    swizzle == 1
                    and (
                        (deter == 1 and layout == 1 and (deter_family == 12))
                        or (
                            deter == 0
                            and layout == 0
                            and (mask_mode == 0 or mask_mode == 3)
                            and sink == 0
                        )
                        or (
                            deter == 0
                            and not (
                                layout == 0
                                and (mask_mode == 0 or mask_mode == 3)
                                and (sink == 0)
                            )
                        )
                    )
                )
                or (
                    deter == 1
                    and layout == 0
                    and mask_mode == 0
                    and sink == 0
                    and (
                        (swizzle == 1 and deter_family == 1)
                        or (swizzle == 0 and deter_family == 21)
                    )
                )
            )
        )
        if use_unitflag:
            pl.store(
                tensor_dv_ws,
                dv_acc0.current(),
                [i0, i1, i2, 0],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
                phase=pl.STPhase.Final,
            )
        else:
            pl.store(
                tensor_dv_ws,
                dv_acc0.current(),
                [i0, i1, i2, 0],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
            )
        return
    if dv_chunk_valid(constInfo, 0) > 0:
        if nz_out == 1:
            store_nz_partial(
                tensor_dv_ws, dv_acc0.current(), runInfo.nzKvTile, constInfo.dAlign
            )
        else:
            pl.store(
                tensor_dv_ws,
                dv_acc0.current(),
                [i0, i1, i2, D_CHUNK_OFF0[d_align]],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
            )
    if D_NCHUNKS[d_align] > 1:
        if dv_chunk_valid(constInfo, 1) > 0:
            pl.store(
                tensor_dv_ws,
                dv_acc1.current(),
                [i0, i1, i2, D_CHUNK_OFF1[d_align]],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
            )


def flush_dk_l0c_to_ub(constInfo, runInfo, dk_c0, dk_c1, mm_ub0, mm_ub1):
    """
    BN2 192（P2）：dK 需要乘 scale，把两个 D 分块从 L0C 搬进 UB 交给 vector。
    chunk0 复用 mm2 的上一槽（与 narrow 同款，受 SYNC_DETER_FIX 保护），
    chunk1 用 :func:`make_l0c_load_ub` 的专用槽。只在 flushDkv==1 调用。
    """
    w0 = chunk_valid_d(constInfo.dSize, 0)
    if w0 > 0:
        pl.set_validshape(mm_ub0, [VECTOR_BASEM, w0])
        pl.move(mm_ub0, dk_c0.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    if D_NCHUNKS[d_align] > 1 and chunk_valid_d(constInfo.dSize, 1) > 0:
        w1 = chunk_valid_d(constInfo.dSize, 1)
        pl.set_validshape(mm_ub1, [VECTOR_BASEM, w1])
        pl.move(mm_ub1, dk_c1.current(), acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)


def flush_dv_l0c_to_out(constInfo, runInfo, tensor_dv_out, dv_acc0, dv_acc1):
    """BN2 192/256（P2）：dV 不乘 scale，直接 fixpipe fp32→16-bit 落 GM。"""
    i0 = runInfo.idx0KV
    i1 = runInfo.keyOffset
    i2 = runInfo.idx2KV
    if dv_chunk_valid(constInfo, 0) > 0:
        pl.store(
            tensor_dv_out,
            dv_acc0.current(),
            [i0, i1, i2, D_CHUNK_OFF0[d_align]],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicNone,
        )
    if D_NCHUNKS[d_align] > 1 and dv_chunk_valid(constInfo, 1) > 0:
        pl.store(
            tensor_dv_out,
            dv_acc1.current(),
            [i0, i1, i2, D_CHUNK_OFF1[d_align]],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicNone,
        )


def flush_dkv(
    constInfo, runInfo, tensor_dk_ws, tensor_dv_ws, dk_acc0, dk_acc1, dv_acc0, dv_acc1
):
    """普通路径兼容入口；newDeter 在各自 FIX token 内分开 flush。"""
    flush_dk(constInfo, runInfo, tensor_dk_ws, dk_acc0, dk_acc1)
    flush_dv(constInfo, runInfo, tensor_dv_ws, dv_acc0, dv_acc1)


# BN2S2 implementation: separate resource and synchronization semantics.
from memory import fag_mem as mem
from fag_common import (
    CUBE_BASEM_bn2s2,
    CUBE_BASEN_bn2s2,
    D_CHUNK_OFF0_bn2s2,
    D_CHUNK_OFF1_bn2s2,
    D_CHUNK_W0_bn2s2,
    D_CHUNK_W1_bn2s2,
    D_NCHUNKS_bn2s2,
    SYNC_C1_TO_V2_FLAG_bn2s2,
    SYNC_C2_TO_V2_FLAG_bn2s2,
    VECTOR_BASEM_bn2s2,
    VECTOR_BASEN_bn2s2,
)
from schedule.deterministic.tnd.fag_deter_tnd_plan import (
    PlanSplit_bn2s2,
    plan_commit_target_bn2s2,
)

# --- end generated imports ---


# ==================================================================
#  三种片上 matmul：K 轴累加 / N 轴写出 / N 轴驻留
# ==================================================================


def _mm_n_chunk_bn2s2(
    left, right, acc, srcL, srcR, tensor_out, outIdx, mReal, kReal, dOff, validD, l1Off
):
    _mm_n_chunk(
        left,
        right,
        acc,
        srcL,
        srcR,
        tensor_out,
        outIdx,
        mReal,
        kReal,
        dOff,
        validD,
        l1Off,
        0,
        0,
        0,
        True,
    )


def iterate_normal_n_left_reuse_bn2s2(
    srcL,
    srcR,
    left,
    left_read,
    right,
    acc,
    tensor_out,
    outIdx,
    mReal,
    kReal,
    dimSize,
    tensor_src=None,
    srcIdx=None,
    first_store=False,
):
    """Ordinary wide gradients: retain one L0A generation for all N strips.

    Keep each existing L0C acquire/MMAD/atomic store and L0B generation.
    The unmanaged alias prevents the first MMAD from releasing the left
    consumer token before the last strip, matching ASC MatmulNMutex.
    """
    if dimSize > 0:
        lt = left.next()
        pl.set_validshape(lt, [mReal, kReal])
        pl.move(lt, srcL)
        left_slot = left.cursor.cursor % 2
        lr = left_read[left_slot]
        pl.set_validshape(lr, [mReal, kReal])
        left_mutex = mem.MTX_L0A_bn2s2[d_align][left_slot]
        pl.system.mutex_lock(pipe=pl.PipeType.M, mutex_id=left_mutex)
        for i in pl.range(0, D_NCHUNKS_bn2s2[d_align]):
            validD = chunk_valid_d(dimSize, i, True)
            if d_align >= 512:
                if i % 2 == 0:
                    winOff = chunk_win(i)
                    winW = pl.max(pl.min(D768_PHYS, dimSize - winOff), 0)
                    if winW > 0:
                        pl.set_validshape(srcR, [kReal, winW])
                        pl.load(
                            srcR,
                            tensor_src,
                            [srcIdx[0], srcIdx[1], srcIdx[2], winOff],
                            order=[1, 3],
                        )
            # Wide keys advance right/acc even for a padded tail slot.
            if d_align >= 512 or validD > 0:
                _mm_n_chunk_loaded_left(
                    lr,
                    right,
                    acc,
                    srcR,
                    tensor_out,
                    outIdx,
                    mReal,
                    kReal,
                    chunk_off(i, True),
                    validD,
                    chunk_local(i, True),
                    first_store,
                )
        pl.system.mutex_unlock(pipe=pl.PipeType.M, mutex_id=left_mutex)


def iterate_normal_n_left_reuse_l0c_bn2s2(
    srcL, srcR, left, left_read, right, acc0, acc1, isAcc, mReal, kReal, dimSize, flush
):
    """D256 resident dV: retain L0A, leaving both L0C owners and flush unchanged."""
    if dimSize > 0:
        lt = left.next()
        pl.set_validshape(lt, [mReal, kReal])
        pl.move(lt, srcL)
        left_slot = left.cursor.cursor % 2
        lr = left_read[left_slot]
        pl.set_validshape(lr, [mReal, kReal])
        left_mutex = mem.MTX_L0A_bn2s2[d_align][left_slot]
        pl.system.mutex_lock(pipe=pl.PipeType.M, mutex_id=left_mutex)
        off0 = D_CHUNK_OFF0_bn2s2[d_align]
        w0 = pl.max(pl.min(D_CHUNK_W0_bn2s2[d_align], dimSize - off0), 0)
        if w0 > 0:
            _mm_resident_chunk_loaded_left(
                lr,
                right,
                acc0.current(),
                srcR,
                mReal,
                kReal,
                off0,
                w0,
                isAcc,
                True,
                flush,
            )
        if D_NCHUNKS_bn2s2[d_align] > 1:
            off1 = D_CHUNK_OFF1_bn2s2[d_align]
            w1 = pl.max(pl.min(D_CHUNK_W1_bn2s2[d_align], dimSize - off1), 0)
            if w1 > 0:
                _mm_resident_chunk_loaded_left(
                    lr,
                    right,
                    acc1.current(),
                    srcR,
                    mReal,
                    kReal,
                    off1,
                    w1,
                    isAcc,
                    True,
                    flush,
                )
        pl.system.mutex_unlock(pipe=pl.PipeType.M, mutex_id=left_mutex)


# ==================================================================
#  C1 / C2 —— D 在 K 轴，一块 acc，Fixpipe DualModeSplitM
# ==================================================================


def _mm_score_128_bn2s2(left, right, at, src_l, src_r, m, n, dim, single, phased):
    # Only the last K product of a phased score publishes FINAL to FIX.
    lt = left.next()
    rt = right.next()
    first_k = pl.min(128, dim)
    _mm_k_chunk(lt, rt, at, src_l, src_r, 0, first_k, n, True, m, phased, single)
    if not single:
        lt1 = left.next()
        rt1 = right.next()
        _mm_k_chunk(
            lt1, rt1, at, src_l, src_r, 128, dim - 128, n, False, m, phased, True, False
        )


def iterate_mm_k_bn2s2(
    constInfo,
    runInfo,
    tensor_l,
    tensor_r,
    l_t,
    r_l1,
    left,
    right,
    acc,
    mm_ub,
    loadKV,
    dimSize,
    m_rows=CUBE_BASEM_bn2s2,
    preloaded=False,
    right_index=0,
    single_k128=False,
):
    """C1/C2 共用：``[S1, dim] @ [dim, S2]``，dim 是 D 或 Dv。

    默认 M=CUBE_BASEM，保留未迁移调用方的固定 64 分界。plan 确定性族与普通
    BN2S2（含内部 GS 回退）显式传 s1RealSize，使 matmul 和 SplitM 同时使用
    真实 M，并由配套 AIV 按 ceil(M/2) 消费。
    """
    if d_align == 768:
        r_t = r_l1[right_index]
    else:
        r_t = r_l1.current()
    if d_align < 512 and not preloaded:
        pl.set_validshape(l_t, [runInfo.s1RealSize, dimSize])
        pl.set_validshape(r_t, [dimSize, runInfo.s2RealSize])
        _load_score_operand(
            l_t, tensor_l, runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q, 0, False
        )
        if loadKV == 1:
            _load_score_operand(
                r_t,
                tensor_r,
                runInfo.idx0KV,
                runInfo.keyOffset,
                runInfo.idx2KV,
                0,
                True,
            )
    at = acc.next()
    if d_align == 512:
        _score_k_windows(
            runInfo,
            tensor_l,
            tensor_r,
            l_t,
            r_t,
            left,
            right,
            at,
            dimSize,
            m_rows,
            0,
            True,
        )
    elif d_align == 768:
        pl.set_validshape(l_t, [runInfo.s1RealSize, dimSize])
        pl.set_validshape(r_t, [dimSize, runInfo.s2RealSize])
        _load_score_operand(
            l_t, tensor_l, runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q, 0, False
        )
        _load_score_operand(
            r_t, tensor_r, runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, 0, True
        )
        for i in pl.range(0, D_NCHUNKS_bn2s2[d_align]):
            dOff = chunk_off(i, True)
            localOff = dOff
            validD = chunk_valid_d(dimSize, i, True)
            lt = left.next()
            rt = right.next()
            if validD > 0:
                # One L0C hardware lifetime across every K strip, finalized
                # exactly once before the matching score FIX consumer.
                pl.set_validshape(lt, [m_rows, validD])
                pl.set_validshape(rt, [validD, runInfo.s2RealSize])
                pl.set_validshape(at, [m_rows, runInfo.s2RealSize])
                pl.move(lt, l_t, [0, localOff])
                pl.move(rt, r_t, [localOff, 0])
                if i == 0:
                    if dOff + validD >= dimSize:
                        pl.matmul(at, lt, rt, phase=pl.AccPhase.Final)
                    else:
                        pl.matmul(at, lt, rt, phase=pl.AccPhase.Partial)
                else:
                    if dOff + validD >= dimSize:
                        pl.matmul_acc(at, at, lt, rt, phase=pl.AccPhase.Final)
                    else:
                        pl.matmul_acc(at, at, lt, rt, phase=pl.AccPhase.Partial)
    elif (
        template == Tmpl.BN2S2
        and d_align == 192
        and preloaded
        and (dv_align == 128 or (deter == 0 and dv_align == 192))
    ):
        # Both score operands finish one K reduction before the split-M FIX.
        _mm_score_128_bn2s2(
            left,
            right,
            at,
            l_t,
            r_t,
            m_rows,
            runInfo.s2RealSize,
            dimSize,
            single_k128,
            True,
        )
    elif (
        template == Tmpl.BN2S2
        and deter == 0
        and layout == 1
        and d_align == 64
        and dv_align == 64
        and preloaded
    ):
        # Both widths fit in the single score K tile. Keep one final MMAD and
        # the original split-M FIX for each consumer.
        lt = left.next()
        rt = right.next()
        _mm_k_chunk(
            lt,
            rt,
            at,
            l_t,
            r_t,
            0,
            dimSize,
            runInfo.s2RealSize,
            True,
            m_rows,
            True,
            True,
        )
    elif single_k128:
        # D192/Dv128 C1 only: one 128-wide reduction fits each 32 KiB
        # L0A/L0B slot. C2 retains its independent 96+96 reduction.
        lt = left.next()
        rt = right.next()
        _mm_k_chunk(
            lt,
            rt,
            at,
            l_t,
            r_t,
            0,
            dimSize,
            runInfo.s2RealSize,
            True,
            m_rows,
            True,
            True,
        )
    else:
        _score_k_pair(
            left,
            right,
            at,
            l_t,
            r_t,
            dimSize,
            m_rows,
            runInfo.s2RealSize,
            True,
            0,
            True,
        )
    pl.move(
        mm_ub,
        at,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )


# ==================================================================
#  C3 —— D 在 N 轴，各块 atomic 出不相交的 dQ 列
# ==================================================================
def _mm_dq_full96_bn2s2(
    left, right, acc, srcL, srcR, tensor_out, outIdx, mReal, kReal, dim
):
    # Preserve both 96-column products and L0A/L0B double buffering.
    lt = left.next()
    at = acc.next()
    pl.set_validshape(lt, [mReal, kReal])
    pl.move(lt, srcL)
    pl.set_validshape(at, [mReal, dim])
    rt0 = right.next()
    pl.set_validshape(rt0, [kReal, 96])
    pl.move(rt0, srcR, [0, 0])
    first = at[0:mReal, 0:96]
    pl.matmul(first, lt, rt0, phase=pl.AccPhase.Final)
    rt1 = right.next()
    pl.set_validshape(rt1, [kReal, dim - 96])
    pl.move(rt1, srcR, [0, 96])
    # Flatten the storage coordinates only; preserve bytes and parent ownership.
    # 16 * (ceil16(M) * 6) == ceil16(M) * 96 elements.
    carrier = pl.reinterpret(at, shape=[16, 1536])
    pl.set_validshape(carrier, [16, 1536])
    storage_col = ((mReal + 15) // 16) * 96
    tail_storage = carrier[0:16, storage_col : storage_col + 16]
    second = pl.reinterpret(tail_storage, shape=[128, 192])
    pl.set_validshape(second, [mReal, dim - 96])
    pl.matmul(second, lt, rt1, phase=pl.AccPhase.Final)
    pl.store(
        tensor_out,
        at,
        [outIdx[0], outIdx[1], outIdx[2], 0],
        order=[1, 3],
        atomic=pl.AtomicType.AtomicAdd,
        phase=pl.STPhase.Final,
    )


def _mm_dq_full128_bn2s2(
    left, right, acc, srcL, srcR, tensor_out, outIdx, mReal, kReal, dim, left_read
):
    # ASC MatmulNMutex: each disjoint N region is FINAL, then one parent FIX releases both.
    # Do not use Partial for the first N region: this is not a K accumulation.
    lt = left.next()
    at = acc.next()
    pl.set_validshape(lt, [mReal, kReal])
    pl.move(lt, srcL)
    left_slot = left.cursor.cursor % 2
    lr = left_read[left_slot]
    pl.set_validshape(lr, [mReal, kReal])
    left_mutex = mem.MTX_L0A_bn2s2[d_align][left_slot]
    pl.system.mutex_lock(pipe=pl.PipeType.M, mutex_id=left_mutex)
    pl.set_validshape(at, [mReal, dim])
    rt0 = right.next()
    pl.set_validshape(rt0, [kReal, 128])
    pl.move(rt0, srcR, [0, 0])
    first = at[0:mReal, 0:128]
    pl.matmul(first, lr, rt0, phase=pl.AccPhase.Final)
    rt1 = right.next()
    pl.set_validshape(rt1, [kReal, dim - 128])
    pl.move(rt1, srcR, [0, 128])
    # Flatten the storage coordinates only; preserve bytes and parent ownership.
    # 16 * (ceil16(M) * 8) == ceil16(M) * 128 elements.
    carrier = pl.reinterpret(at, shape=[16, 1536])
    pl.set_validshape(carrier, [16, 1536])
    storage_col = ((mReal + 15) // 16) * 128
    tail_storage = carrier[0:16, storage_col : storage_col + 16]
    second = pl.reinterpret(tail_storage, shape=[128, 192])
    pl.set_validshape(second, [mReal, dim - 128])
    pl.matmul(second, lr, rt1, phase=pl.AccPhase.Final)
    pl.system.mutex_unlock(pipe=pl.PipeType.M, mutex_id=left_mutex)
    pl.store(
        tensor_out,
        at,
        [outIdx[0], outIdx[1], outIdx[2], 0],
        order=[1, 3],
        atomic=pl.AtomicType.AtomicAdd,
        phase=pl.STPhase.Final,
    )


def iterate_mm_dsk_bn2s2(
    constInfo,
    runInfo,
    tensor_k,
    tensor_dq_ws,
    loadKV,
    ds_l1,
    k_l1,
    left,
    right,
    acc,
    mm1_ub,
    cBlockIdx,
    reuse_k=False,
    left_read=None,
):
    """C3：``dQ = dS @ K``。

    GS1S2 各块 atomic 出 4D workspace；BN2 单块后写 UB 或 2D workspace
    （``is_bn2_multiblk``）。``mm1_ub`` / ``cBlockIdx`` 只在 BN2 用到。
    """
    k_t = k_l1.current()
    outIdx = [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q]
    if d_align < 512:
        pl.set_validshape(k_t, [runInfo.s2RealSize, constInfo.dSize])
        if not reuse_k:
            if loadKV == 1:
                pl.load(
                    k_t,
                    tensor_k,
                    [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, 0],
                    order=[1, 3],
                )
    if template == Tmpl.BN2S2 and deter == 0 and d_align >= 256:
        iterate_normal_n_left_reuse_bn2s2(
            ds_l1,
            k_t,
            left,
            left_read,
            right,
            acc,
            tensor_dq_ws,
            outIdx,
            runInfo.s1RealSize,
            runInfo.s2RealSize,
            constInfo.dSize,
            tensor_k,
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV],
        )
    elif (
        template == Tmpl.BN2S2
        and deter == 0
        and layout == 1
        and d_align == 64
        and dv_align == 64
    ):
        _mm_n_chunk_bn2s2(
            left,
            right,
            acc,
            ds_l1,
            k_t,
            tensor_dq_ws,
            outIdx,
            runInfo.s1RealSize,
            runInfo.s2RealSize,
            0,
            constInfo.dSize,
            0,
        )
    elif template == Tmpl.BN2S2 and deter == 1 and d_align == 192 and dv_align == 128:
        _mm_dq_full128_bn2s2(
            left,
            right,
            acc,
            ds_l1,
            k_t,
            tensor_dq_ws,
            outIdx,
            runInfo.s1RealSize,
            runInfo.s2RealSize,
            constInfo.dSize,
            left_read,
        )
    elif template == Tmpl.BN2S2 and deter == 1 and d_align == 192 and dv_align <= 128:
        _mm_dq_full96_bn2s2(
            left,
            right,
            acc,
            ds_l1,
            k_t,
            tensor_dq_ws,
            outIdx,
            runInfo.s1RealSize,
            runInfo.s2RealSize,
            constInfo.dSize,
        )
    elif template == Tmpl.BN2S2 and deter == 1 and d_align == 192 and dv_align == 192:
        held_lt = left.next()
        pl.set_validshape(held_lt, [runInfo.s1RealSize, runInfo.s2RealSize])
        pl.move(held_lt, ds_l1)
        _mm_n_chunk_loaded_left(
            held_lt,
            right,
            acc,
            k_t,
            tensor_dq_ws,
            outIdx,
            runInfo.s1RealSize,
            runInfo.s2RealSize,
            0,
            pl.min(96, constInfo.dSize),
            0,
        )
        _mm_n_chunk_loaded_left(
            held_lt,
            right,
            acc,
            k_t,
            tensor_dq_ws,
            outIdx,
            runInfo.s1RealSize,
            runInfo.s2RealSize,
            96,
            pl.max(pl.min(96, constInfo.dSize - 96), 0),
            96,
        )
    else:
        for i in pl.range(0, D_NCHUNKS_bn2s2[d_align]):
            dOff = chunk_off(i, True)
            localOff = chunk_local(i, True)
            validD = chunk_valid_d(constInfo.dSize, i, True)
            if d_align >= 512:
                if i % 2 == 0:
                    winOff = chunk_win(i)
                    winW = pl.max(pl.min(D768_PHYS, constInfo.dSize - winOff), 0)
                    pl.set_validshape(k_t, [runInfo.s2RealSize, winW])
                    if winW > 0:
                        pl.load(
                            k_t,
                            tensor_k,
                            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV, winOff],
                            order=[1, 3],
                        )
            if template == Tmpl.BN2:
                lt = left.next()
                rt = right.next()
                at = acc.next()
                if validD > 0:
                    pl.set_validshape(lt, [CUBE_BASEM_bn2s2, runInfo.s2RealSize])
                    pl.set_validshape(rt, [runInfo.s2RealSize, validD])
                    pl.set_validshape(at, [CUBE_BASEM_bn2s2, validD])
                    pl.move(lt, ds_l1)
                    pl.move(rt, k_t, [0, localOff])
                    pl.matmul(at, lt, rt)
                    if is_bn2_multiblk == 1:
                        rowOff = (
                            cBlockIdx * constInfo.s1Outer + runInfo.s1oIdx
                        ) * CUBE_BASEM_bn2s2
                        if runInfo.dqFirst == 1:
                            pl.store(
                                tensor_dq_ws,
                                at,
                                [rowOff, 0],
                                atomic=pl.AtomicType.AtomicNone,
                            )
                        else:
                            pl.store(
                                tensor_dq_ws,
                                at,
                                [rowOff, 0],
                                atomic=pl.AtomicType.AtomicAdd,
                            )
                    else:
                        pl.set_validshape(
                            mm1_ub, [VECTOR_BASEM_bn2s2, VECTOR_BASEN_bn2s2]
                        )
                        pl.move(
                            mm1_ub,
                            at,
                            acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
                        )
            elif d_align >= 512 or validD > 0:
                _mm_n_chunk_bn2s2(
                    left,
                    right,
                    acc,
                    ds_l1,
                    k_t,
                    tensor_dq_ws,
                    outIdx,
                    runInfo.s1RealSize,
                    runInfo.s2RealSize,
                    dOff,
                    validD,
                    localOff,
                )


# ==================================================================
#  C4 / C5 —— D 在 N 轴：≤192 驻留 L0C，>192 atomic 出 GM
# ==================================================================
def iterate_mm_n_bn2s2(
    constInfo,
    runInfo,
    srcL,
    srcR,
    left,
    right,
    acc0,
    tensor_out,
    tensor_src,
    dimSize,
    srcIdx,
    outIdx,
    mReal,
    left_read=None,
):
    """C4/C5 写出路径（``d_align>192``）：每块 atomic 出 GM。

    ``tensor_out`` / ``tensor_src`` 必须是真 Tensor。新前端不允许把
    ``None`` 默认值再往下传 —— 驻留路径走 ``iterate_mm_n_l0c``。
    """
    kReal = runInfo.s1RealSize
    if template == Tmpl.BN2S2 and deter == 0 and d_align >= 512:
        iterate_normal_n_left_reuse_bn2s2(
            srcL,
            srcR,
            left,
            left_read,
            right,
            acc0,
            tensor_out,
            outIdx,
            mReal,
            kReal,
            dimSize,
            tensor_src,
            srcIdx,
        )
    else:
        for i in pl.range(0, D_NCHUNKS_bn2s2[d_align]):
            dOff = chunk_off(i, True)
            localOff = chunk_local(i, True)
            validD = chunk_valid_d(dimSize, i, True)
            if d_align >= 512:
                if i % 2 == 0:
                    winOff = chunk_win(i)
                    winW = pl.max(pl.min(D768_PHYS, dimSize - winOff), 0)
                    pl.set_validshape(srcR, [runInfo.s1RealSize, winW])
                    if winW > 0:
                        pl.load(
                            srcR,
                            tensor_src,
                            [srcIdx[0], srcIdx[1], srcIdx[2], winOff],
                            order=[1, 3],
                        )
            if d_align >= 512 or validD > 0:
                _mm_n_chunk_bn2s2(
                    left,
                    right,
                    acc0,
                    srcL,
                    srcR,
                    tensor_out,
                    outIdx,
                    mReal,
                    kReal,
                    dOff,
                    validD,
                    localOff,
                )


def iterate_mm_n_l0c_bn2s2(
    constInfo,
    runInfo,
    srcL,
    srcR,
    left,
    right,
    acc0,
    acc1,
    isAcc,
    dimSize,
    mReal,
):
    """C4/C5 驻留 L0C（``d_align<=192``）。不接收 Tensor 参数。

    acc0/acc1 必须是编译期两路。D=64/128 一块；D=192 是 96+96
    （Dv=128 时 C5 第二块 validD=32）。
    """
    kReal = runInfo.s1RealSize
    _resident_n_pair(
        srcL,
        srcR,
        left,
        right,
        acc0,
        acc1,
        isAcc,
        dimSize,
        mReal,
        kReal,
        True,
        runInfo.flushDkv,
    )


def iterate_mm_dsq_bn2s2(
    constInfo,
    runInfo,
    ds_t_l1,
    q_t,
    left,
    right,
    acc0,
    acc1=None,
    isAcc=0,
    tensor_out=None,
    tensor_src=None,
    left_read=None,
):
    mReal = runInfo.s2RealSize
    if d_align > 192:
        iterate_mm_n_bn2s2(
            constInfo,
            runInfo,
            ds_t_l1,
            q_t,
            left,
            right,
            acc0,
            tensor_out,
            tensor_src,
            constInfo.dSize,
            [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q],
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV],
            mReal,
            left_read,
        )
    else:
        iterate_mm_n_l0c_bn2s2(
            constInfo,
            runInfo,
            ds_t_l1,
            q_t,
            left,
            right,
            acc0,
            acc1,
            isAcc,
            constInfo.dSize,
            mReal,
        )


def iterate_mm_pdy_bn2s2(
    constInfo,
    runInfo,
    p_t_l1,
    do_t,
    left,
    right,
    acc0,
    acc1=None,
    isAcc=0,
    tensor_out=None,
    tensor_src=None,
    left_read=None,
):
    if d_align > 192:
        iterate_mm_n_bn2s2(
            constInfo,
            runInfo,
            p_t_l1,
            do_t,
            left,
            right,
            acc0,
            tensor_out,
            tensor_src,
            constInfo.dvSize,
            [runInfo.idx0Q, runInfo.queryOffset, runInfo.idx2Q],
            [runInfo.idx0KV, runInfo.keyOffset, runInfo.idx2KV],
            runInfo.s2RealSize,
            left_read,
        )
    else:
        iterate_mm_n_l0c_bn2s2(
            constInfo,
            runInfo,
            p_t_l1,
            do_t,
            left,
            right,
            acc0,
            acc1,
            isAcc,
            constInfo.dvSize,
            runInfo.s2RealSize,
        )


def store_resident_kv_bn2s2(
    info, target_ws, accumulator, col, owner, final_output, owned_path
):
    if (
        layout == 1
        and deter == 0
        and d_align == 192
        and (mask_mode == 0 or mask_mode == 3)
        and owned_path
    ):
        if owner == 1:
            pl.store(
                final_output,
                accumulator,
                [info.idx0KV, info.keyOffset, info.idx2KV, col],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicNone,
                phase=pl.STPhase.Final,
            )
        else:
            pl.store(
                target_ws,
                accumulator,
                [info.idx0KV, info.keyOffset, info.idx2KV, col],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
                phase=pl.STPhase.Final,
            )
    else:
        pl.store(
            target_ws,
            accumulator,
            [info.idx0KV, info.keyOffset, info.idx2KV, col],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicAdd,
            phase=pl.STPhase.Final,
        )


def flush_owned_dk_bn2s2(
    constInfo, runInfo, tensor_dk_ws, dk_acc0, dk_acc1, owner, direct_out, owned_path
):
    """把 L0C 上累加完的 dK 写回 GM。只用于 d_align≤192。"""
    store_resident_kv_bn2s2(
        runInfo,
        tensor_dk_ws,
        dk_acc0.current(),
        D_CHUNK_OFF0_bn2s2[d_align],
        owner,
        direct_out,
        owned_path,
    )
    if D_NCHUNKS_bn2s2[d_align] > 1:
        store_resident_kv_bn2s2(
            runInfo,
            tensor_dk_ws,
            dk_acc1.current(),
            D_CHUNK_OFF1_bn2s2[d_align],
            owner,
            direct_out,
            owned_path,
        )


def flush_dk_bn2s2(constInfo, runInfo, tensor_dk_ws, dk_acc0, dk_acc1):
    flush_owned_dk_bn2s2(
        constInfo, runInfo, tensor_dk_ws, dk_acc0, dk_acc1, 0, tensor_dk_ws, False
    )


def flush_owned_dv_bn2s2(
    constInfo, runInfo, tensor_dv_ws, dv_acc0, dv_acc1, owner, direct_out, owned_path
):
    """把 L0C 上累加完的 dV 写回 GM。只用于 d_align≤192。"""
    if dv_chunk_valid(constInfo, 0, True) > 0:
        store_resident_kv_bn2s2(
            runInfo,
            tensor_dv_ws,
            dv_acc0.current(),
            D_CHUNK_OFF0_bn2s2[d_align],
            owner,
            direct_out,
            owned_path,
        )
    if D_NCHUNKS_bn2s2[d_align] > 1:
        if dv_chunk_valid(constInfo, 1, True) > 0:
            store_resident_kv_bn2s2(
                runInfo,
                tensor_dv_ws,
                dv_acc1.current(),
                D_CHUNK_OFF1_bn2s2[d_align],
                owner,
                direct_out,
                owned_path,
            )


def flush_dv_bn2s2(constInfo, runInfo, tensor_dv_ws, dv_acc0, dv_acc1):
    flush_owned_dv_bn2s2(
        constInfo, runInfo, tensor_dv_ws, dv_acc0, dv_acc1, 0, tensor_dv_ws, False
    )


def flush_dkv_bn2s2(
    constInfo, runInfo, tensor_dk_ws, tensor_dv_ws, dk_acc0, dk_acc1, dv_acc0, dv_acc1
):
    """普通路径兼容入口；newDeter 在各自 FIX token 内分开 flush。"""
    flush_dk_bn2s2(constInfo, runInfo, tensor_dk_ws, dk_acc0, dk_acc1)
    flush_dv_bn2s2(constInfo, runInfo, tensor_dv_ws, dv_acc0, dv_acc1)


# ==================================================================
#  BN2S2 TND metadata 确定性：共享 C4/C5 与局部写回策略
# ==================================================================
# logicalSplit 只影响写回地址与累加方式，仍共用同一 matmul 调用链。


def plan_store_kv_bn2s2(
    policy, info, ws_ptr, tensor_out, at, mm2_ub, dim, d_off, is_dv, resident
):
    direct = False
    if (d_align == 64 and dv_align == 64) or (d_align == 128 and dv_align == 128):
        direct = policy.logicalSplit == PlanSplit_bn2s2.BN2S2 and info.specialIndex < 0
    if direct:
        if is_dv:
            pl.store(
                tensor_out,
                at,
                [0, info.keyOffset, info.idx2KV, d_off],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicNone,
                phase=pl.STPhase.Final,
            )
        else:
            # Keep the actual matmul M: both its packed L0C stride and SplitM
            # tail boundary depend on it. Expanding M after matmul changes the
            # source stride and can read the wrong fractal on a short column.
            pl.set_validshape(mm2_ub, [64, 128])
            pl.move(
                mm2_ub,
                at,
                acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
                phase=pl.STPhase.Final,
            )
    else:
        offset, stride, atomic = plan_commit_target_bn2s2(policy, info, is_dv, resident)
        target = pl.make_tensor(
            pl.addptr(ws_ptr, offset), [128, dim], [stride, 1], dtype=pl.DT_FP32
        )
        if d_align == 768:
            pl.system.bar_fix()
        if atomic != 0:
            pl.store(
                target,
                at,
                [0, d_off],
                atomic=pl.AtomicType.AtomicAdd,
                phase=pl.STPhase.Final,
            )
        else:
            pl.store(
                target,
                at,
                [0, d_off],
                atomic=pl.AtomicType.AtomicNone,
                phase=pl.STPhase.Final,
            )


def plan_resident_chunk_bn2s2(
    policy,
    info,
    src_l,
    src_r,
    left,
    right,
    at,
    ws_ptr,
    tensor_out,
    mm2_ub,
    dim,
    d_off,
    width,
    is_dv,
    resident,
):
    valid_d = pl.max(pl.min(width, dim - d_off), 0)
    if valid_d > 0:
        lt = left.next()
        rt = right.next()
        pl.set_validshape(lt, [info.s2RealSize, info.s1RealSize])
        pl.set_validshape(rt, [info.s1RealSize, valid_d])
        pl.set_validshape(at, [info.s2RealSize, valid_d])
        pl.move(lt, src_l)
        pl.move(rt, src_r, [0, d_off])
        flush = resident == 0 or info.flushDkv != 0
        _plan_resident_phase_mmad_bn2s2(
            at, lt, rt, resident != 0 and info.dkvAcc != 0, flush
        )
        if flush:
            plan_store_kv_bn2s2(
                policy,
                info,
                ws_ptr,
                tensor_out,
                at,
                mm2_ub,
                dim,
                d_off,
                is_dv,
                resident,
            )


def _plan_resident_phase_mmad_bn2s2(at, lt, rt, accumulate, flush):
    # Each disjoint N strip remains resident until the whole-parent final store.
    # Match ASC MatmulNMutex: every strip uses the same Partial/Final generation.
    if flush != 0:
        if accumulate != 0:
            pl.matmul_acc(at, at, lt, rt, phase=pl.AccPhase.Final)
        else:
            pl.matmul(at, lt, rt, phase=pl.AccPhase.Final)
    else:
        if accumulate != 0:
            pl.matmul_acc(at, at, lt, rt, phase=pl.AccPhase.Partial)
        else:
            pl.matmul(at, lt, rt, phase=pl.AccPhase.Partial)


def plan_resident_full_bn2s2(
    policy,
    info,
    src_l,
    src_r,
    left,
    right,
    at,
    ws_ptr,
    tensor_out,
    mm2_ub,
    dim,
    full_width,
    is_dv,
    resident,
):
    # Preserve one parent accumulator across S1 tasks; only its final consumer
    # writes the full result. Compiler verification of compact subviews is required.
    lt = left.next()
    pl.set_validshape(lt, [info.s2RealSize, info.s1RealSize])
    pl.move(lt, src_l)
    pl.set_validshape(at, [info.s2RealSize, dim])
    rt0 = right.next()
    pl.set_validshape(rt0, [info.s1RealSize, pl.min(128, dim)])
    pl.move(rt0, src_r, [0, 0])
    first = at[0 : info.s2RealSize, 0:128]
    _plan_resident_phase_mmad_bn2s2(
        first,
        lt,
        rt0,
        resident != 0 and info.dkvAcc != 0,
        resident == 0 or info.flushDkv != 0,
    )
    if full_width > 128:
        if dim > 128:
            rt1 = right.next()
            pl.set_validshape(rt1, [info.s1RealSize, dim - 128])
            pl.move(rt1, src_r, [0, 128])
            # MMAD packs columns at ceil16(M), whereas a subview uses its
            # static 128-row pitch. This storage coordinate therefore selects
            # byte offset ceil16(M)*128*sizeof(float), as ASC's second strip.
            tail_storage_col = ((info.s2RealSize + 15) // 16) * 16
            second = at[0 : info.s2RealSize, tail_storage_col : tail_storage_col + 64]
            pl.set_validshape(second, [info.s2RealSize, dim - 128])
            _plan_resident_phase_mmad_bn2s2(
                second,
                lt,
                rt1,
                resident != 0 and info.dkvAcc != 0,
                resident == 0 or info.flushDkv != 0,
            )
    if resident == 0 or info.flushDkv != 0:
        plan_store_kv_bn2s2(
            policy, info, ws_ptr, tensor_out, at, mm2_ub, dim, 0, is_dv, resident
        )


def plan_resident_full_held_bn2s2(
    policy,
    info,
    src_l,
    src_r,
    left,
    right,
    at,
    ws_ptr,
    tensor_out,
    mm2_ub,
    dim,
    full_width,
    is_dv,
    resident,
    left_read,
):
    # Preserve one parent accumulator across S1 tasks; only its final consumer
    # writes the full result. Compiler verification of compact subviews is required.
    lt = left.next()
    pl.set_validshape(lt, [info.s2RealSize, info.s1RealSize])
    pl.move(lt, src_l)
    left_slot = left.cursor.cursor % 2
    lr = left_read[left_slot]
    pl.set_validshape(lr, [info.s2RealSize, info.s1RealSize])
    left_mutex = mem.MTX_L0A_bn2s2[d_align][left_slot]
    pl.system.mutex_lock(pipe=pl.PipeType.M, mutex_id=left_mutex)
    pl.set_validshape(at, [info.s2RealSize, dim])
    rt0 = right.next()
    pl.set_validshape(rt0, [info.s1RealSize, pl.min(128, dim)])
    pl.move(rt0, src_r, [0, 0])
    first = at[0 : info.s2RealSize, 0:128]
    _plan_resident_phase_mmad_bn2s2(
        first,
        lr,
        rt0,
        resident != 0 and info.dkvAcc != 0,
        resident == 0 or info.flushDkv != 0,
    )
    if full_width > 128:
        if dim > 128:
            rt1 = right.next()
            pl.set_validshape(rt1, [info.s1RealSize, dim - 128])
            pl.move(rt1, src_r, [0, 128])
            # MMAD packs columns at ceil16(M), whereas a subview uses its
            # static 128-row pitch. This storage coordinate therefore selects
            # byte offset ceil16(M)*128*sizeof(float), as ASC's second strip.
            tail_storage_col = ((info.s2RealSize + 15) // 16) * 16
            second = at[0 : info.s2RealSize, tail_storage_col : tail_storage_col + 64]
            pl.set_validshape(second, [info.s2RealSize, dim - 128])
            _plan_resident_phase_mmad_bn2s2(
                second,
                lr,
                rt1,
                resident != 0 and info.dkvAcc != 0,
                resident == 0 or info.flushDkv != 0,
            )
    pl.system.mutex_unlock(pipe=pl.PipeType.M, mutex_id=left_mutex)
    if resident == 0 or info.flushDkv != 0:
        plan_store_kv_bn2s2(
            policy, info, ws_ptr, tensor_out, at, mm2_ub, dim, 0, is_dv, resident
        )


def _plan_held_left_strip_bn2s2(
    policy, info, lt, src_r, right, acc, ws_ptr, tensor_out, mm2_ub, dim, is_dv, d_off
):
    rt = right.next()
    at = acc.next()
    valid_d = pl.max(pl.min(96, dim - d_off), 0)
    if valid_d > 0:
        pl.set_validshape(rt, [info.s1RealSize, valid_d])
        pl.set_validshape(at, [info.s2RealSize, valid_d])
        pl.move(rt, src_r, [0, d_off])
        pl.matmul(at, lt, rt, phase=pl.AccPhase.Final)
        plan_store_kv_bn2s2(
            policy, info, ws_ptr, tensor_out, at, mm2_ub, dim, d_off, is_dv, False
        )


def plan_streamed_kv_bn2s2(
    policy,
    info,
    src_l,
    src_r,
    left,
    right,
    acc,
    ws_ptr,
    tensor_out,
    tensor_src,
    mm2_ub,
    dim,
    is_dv,
):
    if template == Tmpl.BN2S2 and d_align == 192 and dv_align == 192:
        held_lt = left.next()
        pl.set_validshape(held_lt, [info.s2RealSize, info.s1RealSize])
        pl.move(held_lt, src_l)
        _plan_held_left_strip_bn2s2(
            policy,
            info,
            held_lt,
            src_r,
            right,
            acc,
            ws_ptr,
            tensor_out,
            mm2_ub,
            dim,
            is_dv,
            0,
        )
        _plan_held_left_strip_bn2s2(
            policy,
            info,
            held_lt,
            src_r,
            right,
            acc,
            ws_ptr,
            tensor_out,
            mm2_ub,
            dim,
            is_dv,
            96,
        )
    else:
        for i in pl.range(0, D_NCHUNKS_bn2s2[d_align]):
            step = D_CHUNK_W0_bn2s2[d_align]
            d_off = i * step
            local_off = d_off
            if d_align == 768:
                local_off = (i % 2) * 128
                if i % 2 == 0:
                    win = (i // 2) * 256
                    win_size = pl.max(pl.min(256, dim - win), 0)
                    if win_size > 0:
                        pl.set_validshape(src_r, [info.s1RealSize, win_size])
                        pl.load(
                            src_r,
                            tensor_src,
                            [0, info.queryOffset, info.idx2Q, win],
                            order=[1, 3],
                        )
            lt = left.next()
            rt = right.next()
            at = acc.next()
            valid_d = pl.max(pl.min(step, dim - d_off), 0)
            if valid_d > 0:
                pl.set_validshape(lt, [info.s2RealSize, info.s1RealSize])
                pl.set_validshape(rt, [info.s1RealSize, valid_d])
                pl.set_validshape(at, [info.s2RealSize, valid_d])
                pl.move(lt, src_l)
                pl.move(rt, src_r, [0, local_off])
                pl.matmul(at, lt, rt, phase=pl.AccPhase.Final)
                plan_store_kv_bn2s2(
                    policy,
                    info,
                    ws_ptr,
                    tensor_out,
                    at,
                    mm2_ub,
                    dim,
                    d_off,
                    is_dv,
                    False,
                )


def iterate_plan_dk_bn2s2(
    ci,
    policy,
    info,
    ds_t,
    q_l1,
    left,
    right,
    acc,
    dk0,
    dk1,
    ws_ptr,
    tensor_dk,
    tensor_q,
    mm2_ub,
    left_read=None,
):
    if d_align == 192 and dv_align == 128:
        plan_resident_full_held_bn2s2(
            policy,
            info,
            ds_t,
            q_l1,
            left,
            right,
            dk0.current(),
            ws_ptr,
            tensor_dk,
            mm2_ub,
            ci.dSize,
            192,
            False,
            True,
            left_read,
        )
    elif d_align == 192 and dv_align <= 128:
        plan_resident_full_bn2s2(
            policy,
            info,
            ds_t,
            q_l1,
            left,
            right,
            dk0.current(),
            ws_ptr,
            tensor_dk,
            mm2_ub,
            ci.dSize,
            192,
            False,
            True,
        )
    elif d_align <= 128:
        plan_resident_chunk_bn2s2(
            policy,
            info,
            ds_t,
            q_l1,
            left,
            right,
            dk0.current(),
            ws_ptr,
            tensor_dk,
            mm2_ub,
            ci.dSize,
            D_CHUNK_OFF0_bn2s2[d_align],
            D_CHUNK_W0_bn2s2[d_align],
            False,
            True,
        )
    else:
        plan_streamed_kv_bn2s2(
            policy,
            info,
            ds_t,
            q_l1,
            left,
            right,
            acc,
            ws_ptr,
            tensor_dk,
            tensor_q,
            mm2_ub,
            ci.dSize,
            False,
        )


def iterate_plan_dv_bn2s2(
    ci,
    policy,
    info,
    p_t,
    dy_l1,
    left,
    right,
    acc,
    dv0,
    dv1,
    ws_ptr,
    tensor_dv,
    tensor_dy,
    mm2_ub,
):
    if d_align == 192:
        plan_resident_full_bn2s2(
            policy,
            info,
            p_t,
            dy_l1,
            left,
            right,
            dv0.current(),
            ws_ptr,
            tensor_dv,
            mm2_ub,
            ci.dvSize,
            dv_align,
            True,
            True,
        )
    elif d_align <= 256:
        resident = pl.astype(1, pl.DT_INT64)
        if d_align == 256:
            resident = pl.astype(ci.s1Size > 64 and ci.s2Size > 64, pl.DT_INT64)
        plan_resident_chunk_bn2s2(
            policy,
            info,
            p_t,
            dy_l1,
            left,
            right,
            dv0.current(),
            ws_ptr,
            tensor_dv,
            mm2_ub,
            ci.dvSize,
            D_CHUNK_OFF0_bn2s2[d_align],
            D_CHUNK_W0_bn2s2[d_align],
            True,
            resident,
        )
        if d_align >= 192:
            plan_resident_chunk_bn2s2(
                policy,
                info,
                p_t,
                dy_l1,
                left,
                right,
                dv1.current(),
                ws_ptr,
                tensor_dv,
                mm2_ub,
                ci.dvSize,
                D_CHUNK_OFF1_bn2s2[d_align],
                D_CHUNK_W1_bn2s2[d_align],
                True,
                resident,
            )
    else:
        plan_streamed_kv_bn2s2(
            policy,
            info,
            p_t,
            dy_l1,
            left,
            right,
            acc,
            ws_ptr,
            tensor_dv,
            tensor_dy,
            mm2_ub,
            ci.dvSize,
            True,
        )


def plan_wide_window_mmad_bn2s2(
    src_l, src_r, left, right, at, m, k, n, left_read, strips=4
):
    lt = left.next()
    pl.set_validshape(lt, [m, k])
    pl.move(lt, src_l)
    # Capture the selected physical DB slot after next(); source cursor stays
    # unchanged through this window. The alias has no automatic M lock.
    left_slot = left.cursor.cursor % 2
    lr = left_read[left_slot]
    pl.set_validshape(lr, [m, k])
    left_mutex = mem.MTX_L0A_bn2s2[d_align][left_slot]
    pl.system.mutex_lock(pipe=pl.PipeType.M, mutex_id=left_mutex)
    pl.set_validshape(at, [m, n])
    for j in pl.range(0, strips):
        width = pl.max(pl.min(128, n - j * 128), 0)
        if width > 0:
            rt = right.next()
            pl.set_validshape(rt, [k, width])
            pl.move(rt, src_r, [0, j * 128])
            storage_col = ((m + 15) // 16) * 16 * j
            piece = at[0:m, storage_col : storage_col + 128]
            pl.set_validshape(piece, [m, width])
            # ASC MatmulNMutex forwards the parent OUTER_LAST flag to every
            # disjoint N strip; K-reduction phase handling remains unchanged.
            pl.matmul(piece, lr, rt, phase=pl.AccPhase.Final)
    pl.system.mutex_unlock(pipe=pl.PipeType.M, mutex_id=left_mutex)


def normal_384_score_bn2s2(
    info, query, dim, writer, lr, rr, right_mutex, left, right, at, result
):
    m = info.s1RealSize
    n = info.s2RealSize
    pl.set_validshape(writer, [m, dim])
    _load_score_operand(writer, query, info.idx0Q, info.queryOffset, info.idx2Q, 0)
    pl.set_validshape(lr, [m, dim])
    pl.set_validshape(rr, [dim, n])
    pl.system.mutex_lock(pipe=pl.PipeType.MTE1, mutex_id=mem.L1_DO_MUTEX_bn2s2[512][1])
    pl.system.mutex_lock(pipe=pl.PipeType.MTE1, mutex_id=right_mutex)
    pl.set_validshape(at, [m, n])
    # The admitted real width is (256,384]. Only the last K strip publishes to FIX.
    _mm_k_chunk(
        left.next(), right.next(), at, lr, rr, 0, 128, n, True, m, True, False, False
    )
    _mm_k_chunk(
        left.next(), right.next(), at, lr, rr, 128, 128, n, False, m, True, False, False
    )
    _mm_k_chunk(
        left.next(),
        right.next(),
        at,
        lr,
        rr,
        256,
        dim - 256,
        n,
        False,
        m,
        True,
        True,
        False,
    )
    pl.system.mutex_unlock(pipe=pl.PipeType.MTE1, mutex_id=right_mutex)
    pl.system.mutex_unlock(
        pipe=pl.PipeType.MTE1, mutex_id=mem.L1_DO_MUTEX_bn2s2[512][1]
    )
    pl.move(
        result,
        at,
        acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM,
        phase=pl.STPhase.Final,
    )


def normal_384_score_pair_bn2s2(
    ci,
    info,
    gm,
    scratch,
    scratch_read,
    k_write,
    k_read,
    v_write,
    v_read,
    k_slot,
    left,
    right,
    at,
    mm1,
    mm2,
    task,
):
    if info.loadKV == 1:
        kt = k_write[k_slot]
        vt = v_write[0]
        pl.set_validshape(kt, [ci.dSize, info.s2RealSize])
        pl.set_validshape(vt, [ci.dvSize, info.s2RealSize])
        _load_score_operand(
            kt, gm.tensor_k, info.idx0KV, info.keyOffset, info.idx2KV, 0, True
        )
        _load_score_operand(
            vt, gm.tensor_v, info.idx0KV, info.keyOffset, info.idx2KV, 0, True
        )
    normal_384_score_bn2s2(
        info,
        gm.tensor_q,
        ci.dSize,
        scratch[0],
        scratch_read[0],
        k_read[k_slot],
        mem.L1_Q_MUTEX_bn2s2[512][k_slot],
        left,
        right,
        at,
        mm2,
    )
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX, event_id=SYNC_C2_TO_V2_FLAG_bn2s2[task % 2]
    )
    normal_384_score_bn2s2(
        info,
        gm.tensor_dy,
        ci.dvSize,
        scratch[0],
        scratch_read[0],
        v_read[0],
        mem.MTX_L1_K_ND_bn2s2[512][0],
        left,
        right,
        at,
        mm1,
    )
    pl.system.set_cross_core(
        pipe=pl.PipeType.FIX, event_id=SYNC_C1_TO_V2_FLAG_bn2s2[task % 2]
    )


def normal_384_gradient_bn2s2(
    src_l,
    rhs,
    rhs_mutex,
    out,
    out_idx,
    m,
    k,
    dim,
    left,
    left_read,
    right,
    at,
    first_store=False,
):
    pl.set_validshape(rhs, [k, dim])
    pl.system.mutex_lock(pipe=pl.PipeType.MTE1, mutex_id=rhs_mutex)
    plan_wide_window_mmad_bn2s2(src_l, rhs, left, right, at, m, k, dim, left_read, 3)
    pl.system.mutex_unlock(pipe=pl.PipeType.MTE1, mutex_id=rhs_mutex)
    if first_store:
        pl.store(
            out,
            at,
            [out_idx[0], out_idx[1], out_idx[2], 0],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicNone,
            phase=pl.STPhase.Final,
        )
    else:
        pl.store(
            out,
            at,
            [out_idx[0], out_idx[1], out_idx[2], 0],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicAdd,
            phase=pl.STPhase.Final,
        )


def normal_384_dq_bn2s2(ci, info, ds, k_read, k_slot, out, left, left_read, right, at):
    normal_384_gradient_bn2s2(
        ds,
        k_read[k_slot],
        mem.L1_Q_MUTEX_bn2s2[512][k_slot],
        out,
        [info.idx0Q, info.queryOffset, info.idx2Q],
        info.s1RealSize,
        info.s2RealSize,
        ci.dSize,
        left,
        left_read,
        right,
        at,
    )


def normal_384_slot_gradient_bn2s2(
    info, src_l, gm, dim, writer, reader, out, left, left_read, right, at
):
    pl.set_validshape(writer, [info.s1RealSize, dim])
    _load_score_operand(writer, gm, info.idx0Q, info.queryOffset, info.idx2Q, 0)
    normal_384_gradient_bn2s2(
        src_l,
        reader,
        mem.L1_DO_MUTEX_bn2s2[512][1],
        out,
        [0, 0, 0],
        info.s2RealSize,
        info.s1RealSize,
        dim,
        left,
        left_read,
        right,
        at,
        info.dkvAcc == 0,
    )


def plan_wide_dq_bn2s2(
    ci,
    info,
    tensor_k,
    tensor_out,
    ds_l1,
    k_l1,
    left,
    right,
    acc,
    common_slot=0,
    rotate=False,
    left_read=None,
):
    if not rotate:
        kt = k_l1.current()
    for window in pl.range(0, 2):
        offset = window * 512
        width = pl.max(pl.min(512, ci.dSize - offset), 0)
        if width > 0:
            if rotate:
                kt = k_l1[(common_slot + window) & 1]
            pl.set_validshape(kt, [info.s2RealSize, width])
            pl.load(
                kt,
                tensor_k,
                [info.idx0KV, info.keyOffset, info.idx2KV, offset],
                order=[1, 3],
            )
            at = acc.next()
            plan_wide_window_mmad_bn2s2(
                ds_l1,
                kt,
                left,
                right,
                at,
                info.s1RealSize,
                info.s2RealSize,
                width,
                left_read,
            )
            pl.store(
                tensor_out,
                at,
                [info.idx0Q, info.queryOffset, info.idx2Q, offset],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicAdd,
                phase=pl.STPhase.Final,
            )


def plan_wide_kv_bn2s2(
    policy,
    info,
    src_l,
    src_r,
    left,
    right,
    acc,
    ws_ptr,
    tensor_out,
    tensor_src,
    mm2_ub,
    dim,
    is_dv,
    common_slot=0,
    rotate=False,
    left_read=None,
):
    for window in pl.range(0, 2):
        offset = window * 512
        width = pl.max(pl.min(512, dim - offset), 0)
        if width > 0:
            if rotate:
                rt = src_r[(common_slot + window) & 1]
            else:
                rt = src_r
            pl.set_validshape(rt, [info.s1RealSize, width])
            pl.load(
                rt, tensor_src, [0, info.queryOffset, info.idx2Q, offset], order=[1, 3]
            )
            at = acc.next()
            plan_wide_window_mmad_bn2s2(
                src_l,
                rt,
                left,
                right,
                at,
                info.s2RealSize,
                info.s1RealSize,
                width,
                left_read,
            )
            plan_store_kv_bn2s2(
                policy, info, ws_ptr, tensor_out, at, mm2_ub, dim, offset, is_dv, False
            )


def iterate_normal_score_qk_bn2s2(
    ci, info, q, k, qt, kt, left, right, acc, result, load
):
    # Main and GS fallback share actual-M score computation. D192/Dv128 callers
    # pair this phased score accumulator with normal_full_dq_bn2s2 below.
    iterate_mm_k_bn2s2(
        ci,
        info,
        q,
        k,
        qt,
        kt,
        left,
        right,
        acc,
        result,
        load,
        ci.dSize,
        info.s1RealSize,
        d_align < 512,
    )


def iterate_normal_score_dyv_bn2s2(
    ci, info, dy, v, dyt, vt, left, right, acc, result, load
):
    iterate_mm_k_bn2s2(
        ci,
        info,
        dy,
        v,
        dyt,
        vt,
        left,
        right,
        acc,
        result,
        load,
        ci.dvSize,
        info.s1RealSize,
        d_align < 512,
        0,
        d_align == 192 and dv_align == 128,
    )


def prefetch_normal_kv_bn2s2(ci, info, k, v, kt, vt):
    # K and its delayed dQ view must advance only on a changed KV owner.
    if info.loadKV == 1:
        pl.set_validshape(kt, [ci.dSize, info.s2RealSize])
        pl.set_validshape(vt, [ci.dvSize, info.s2RealSize])
        _load_score_operand(kt, k, info.idx0KV, info.keyOffset, info.idx2KV, 0, True)
        _load_score_operand(vt, v, info.idx0KV, info.keyOffset, info.idx2KV, 0, True)


def normal_resident_full_held_bn2s2(
    info,
    src_l,
    src_r,
    left,
    right,
    at,
    tensor_out,
    dim,
    full_width,
    left_read,
    owner,
    direct_out,
    owned_path,
):
    # Shared D192/Dv128 lifecycle for ordinary BN2S2 and its GS fallback:
    # dK uses a 192-column parent (128+64); dV uses one 128-column parent.
    # Preserve each parent across S1 tasks under the caller's dkvAcc/flushDkv
    # owner. This helper performs its final FIX; callers must not also run the
    # legacy 96-column flush_dkv_bn2s2 on these accumulators.
    # Compiler verification of compact subviews is required.
    lt = left.next()
    pl.set_validshape(lt, [info.s2RealSize, info.s1RealSize])
    pl.move(lt, src_l)
    left_slot = left.cursor.cursor % 2
    lr = left_read[left_slot]
    pl.set_validshape(lr, [info.s2RealSize, info.s1RealSize])
    left_mutex = mem.MTX_L0A_bn2s2[d_align][left_slot]
    pl.system.mutex_lock(pipe=pl.PipeType.M, mutex_id=left_mutex)
    pl.set_validshape(at, [info.s2RealSize, dim])
    rt0 = right.next()
    pl.set_validshape(rt0, [info.s1RealSize, pl.min(128, dim)])
    pl.move(rt0, src_r, [0, 0])
    first = at[0 : info.s2RealSize, 0:128]
    _plan_resident_phase_mmad_bn2s2(
        first,
        lr,
        rt0,
        info.dkvAcc != 0,
        info.flushDkv != 0,
    )
    if full_width > 128:
        if dim > 128:
            rt1 = right.next()
            pl.set_validshape(rt1, [info.s1RealSize, dim - 128])
            pl.move(rt1, src_r, [0, 128])
            # MMAD packs columns at ceil16(M), whereas a subview uses its
            # static 128-row pitch. This storage coordinate therefore selects
            # byte offset ceil16(M)*128*sizeof(float), as ASC's second strip.
            tail_storage_col = ((info.s2RealSize + 15) // 16) * 16
            second = at[0 : info.s2RealSize, tail_storage_col : tail_storage_col + 64]
            pl.set_validshape(second, [info.s2RealSize, dim - 128])
            _plan_resident_phase_mmad_bn2s2(
                second,
                lr,
                rt1,
                info.dkvAcc != 0,
                info.flushDkv != 0,
            )
    pl.system.mutex_unlock(pipe=pl.PipeType.M, mutex_id=left_mutex)
    if info.flushDkv != 0:
        store_resident_kv_bn2s2(info, tensor_out, at, 0, owner, direct_out, owned_path)


def normal_full_dq_bn2s2(
    ci, info, tensor_out, ds, knormal, left, right, acc, left_read
):
    # Consume the matching preloaded K generation with the same phased parent
    # lifecycle as C1/C2; keep the L0A consumer lock across both N strips.
    kt = knormal.current()
    pl.set_validshape(kt, [info.s2RealSize, ci.dSize])
    _mm_dq_full128_bn2s2(
        left,
        right,
        acc,
        ds,
        kt,
        tensor_out,
        [info.idx0Q, info.queryOffset, info.idx2Q],
        info.s1RealSize,
        info.s2RealSize,
        ci.dSize,
        left_read,
    )
