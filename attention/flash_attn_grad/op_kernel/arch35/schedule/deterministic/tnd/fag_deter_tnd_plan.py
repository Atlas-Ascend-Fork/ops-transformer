# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Versioned AICPU TND plans; no host readback and no per-round task table.

Coordinate formulas mirror AscendC deter.h production mode 2, line, owned BN2,
causal mirror P2, and causal column C4. A missing/invalid protocol takes the
existing closed-form schedule and exposes protocolStatus for trace validation.
"""

# Shared coordinate formulas: only dependency-closed identical bodies.
from schedule.deterministic.fag_deter_formulas import (
    fused_valid_for_mask,
    cal_gqa_dense_uncached,
    _in_range,
    band_pq,
    cal_band,
    cal_dense,
    cal_gqa_dense_no_tune,
    pack_fused,
    select_gqa_dense_round,
)
from schedule.deterministic.fag_deter_legacy import (
    deter_lin_max_round,
    _deter_lin_gidx_host,
    _deter_sparse_coordinates,
)

from schedule.deterministic.tnd.fag_deter_tnd_dense import (
    pl,
    init_tnd_dense_schedule,
    tnd_dense_gidx,
    tnd_deter_lengths,
)
from schedule.deterministic.fag_deter_state import (
    make_tnd_deter_schedule,
    reset_tnd_batch_cursor,
)
from schedule.deterministic.tnd.fag_deter_tnd_sparse import (
    init_tnd_sparse_schedule,
    tnd_sparse_gidx,
    tnd_sparse_band_shape,
)
from schedule.deterministic.tnd.fag_deter_tnd_normalized import (
    apply_tnd_normalized_plan,
    tnd_normalized_gidx,
)

TND_PLAN_MAGIC = 0x46545031
TND_PLAN_VERSION = 3
TND_PLAN_HEADER_WORDS = 16
TND_PLAN_BATCH_OFFSET = 122
TND_PLAN_BATCH_WORDS = 32
TND_PLAN_CUT_OFFSET = 48


def tnd_plan_i64(metadata, offset):
    low = pl.astype(pl.getval(metadata, offset), pl.DT_INT64)
    high = pl.astype(pl.getval(metadata, offset + 1), pl.DT_INT64)
    if low < 0:
        low = low + 4294967296
    return high * 4294967296 + low


def tnd_plan_scalar(metadata, base, field):
    return tnd_plan_i64(metadata, base + 16 + 2 * field)


def tnd_plan_batch(metadata, base, batch, field):
    return tnd_plan_i64(metadata, base + 122 + batch * 32 + field * 2)


def tnd_window_is_dense(ci, cuq, cuk, usedq, usedk):
    """Source SetSparseParams uses actual TND maxima and unshifted windows."""
    dense = pl.astype(1, pl.DT_INT64)
    pre = ci.s1Token - ci.s1Size + ci.s2Size
    nxt = ci.s2Token + ci.s1Size - ci.s2Size
    for batch in pl.range(ci.bSize):
        sq, sk = tnd_deter_lengths(ci, batch, cuq, cuk, usedq, usedk)
        if sq > pre or sk > nxt:
            dense = pl.astype(0, pl.DT_INT64)
    return dense


def tnd_plan_records_valid(
    ci,
    k,
    metadata,
    base,
    strategy,
    rounds,
    singles,
    sparse_type,
    cuq,
    cuk,
    usedq,
    usedk,
):
    """Validate only this compiled family's records once before optimized use."""
    valid = pl.astype(1, pl.DT_INT64)
    area = pl.astype(0, pl.DT_INT64)
    mode2_rounds = pl.astype(0, pl.DT_INT64)
    single_tasks = pl.astype(0, pl.DT_INT64)
    run_end = pl.astype(0, pl.DT_INT64)
    run_begin = pl.astype(0, pl.DT_INT64)
    line_end = singles
    run_m = pl.astype(0, pl.DT_INT64)
    run_n = pl.astype(0, pl.DT_INT64)
    run_p = pl.astype(0, pl.DT_INT64)
    run_q = pl.astype(0, pl.DT_INT64)
    if ci.bSize <= 0 or k <= 0 or k > 36:
        valid = pl.astype(0, pl.DT_INT64)
    for batch in pl.range(ci.bSize):
        sq, sk = tnd_deter_lengths(ci, batch, cuq, cuk, usedq, usedk)
        m = (sq + 127) // 128
        n = (sk + 127) // 128
        if (
            sq < 0
            or sk < 0
            or m > ci.s1Outer
            or n > ci.s2Outer
            or tnd_plan_batch(metadata, base, batch, 0) != sq
            or tnd_plan_batch(metadata, base, batch, 1) != sk
            or tnd_plan_batch(metadata, base, batch, 2) != m
            or tnd_plan_batch(metadata, base, batch, 3) != n
        ):
            valid = pl.astype(0, pl.DT_INT64)
        if deter_family == 12 or (
            deter_family == 13 and mask_mode == 4 and strategy == 1
        ):
            if (
                m <= 0
                or n <= 0
                or m < pl.min(k, n)
                or tnd_plan_batch(metadata, base, batch, 7) != mode2_rounds
            ):
                valid = pl.astype(0, pl.DT_INT64)
            mode2_rounds = mode2_rounds + ((n * ci.n2Size + k - 1) // k) * m
        elif deter_family >= 14:
            if m <= 0 or n <= 0 or tnd_plan_batch(metadata, base, batch, 6) != area:
                valid = pl.astype(0, pl.DT_INT64)
            if deter_family == 14:
                area = area + m * n
            elif deter_family == 15 or deter_family == 17:
                if sq != sk:
                    valid = pl.astype(0, pl.DT_INT64)
                area = area + m * (m + 1) // 2
            else:
                if sq != sk:
                    valid = pl.astype(0, pl.DT_INT64)
                area = area + (m + 1) * m
        else:
            p = pl.astype(0, pl.DT_INT64)
            q = pl.astype(0, pl.DT_INT64)
            shift = pl.astype(0, pl.DT_INT64)
            if mask_mode == 3:
                if sq != sk:
                    valid = pl.astype(0, pl.DT_INT64)
            else:
                m, n, p, q, mo, no, vm, vn, whole, bound = tnd_sparse_band_shape(
                    ci, sq, sk
                )
                shift = mo - no
                if sq <= 0 or sk <= 0:
                    m = pl.astype(0, pl.DT_INT64)
                    n = pl.astype(0, pl.DT_INT64)
                    p = pl.astype(0, pl.DT_INT64)
                    q = pl.astype(0, pl.DT_INT64)
                    shift = pl.astype(0, pl.DT_INT64)
            if (
                tnd_plan_batch(metadata, base, batch, 4) != p
                or tnd_plan_batch(metadata, base, batch, 5) != q
                or tnd_plan_batch(metadata, base, batch, 13) != m
                or tnd_plan_batch(metadata, base, batch, 14) != n
                or tnd_plan_batch(metadata, base, batch, 15) != shift
                or tnd_plan_batch(metadata, base, batch, 11) != single_tasks
            ):
                valid = pl.astype(0, pl.DT_INT64)
            if mask_mode == 3 and m == 1 and n == 1:
                single_tasks = single_tasks + ci.n2Size
            if tnd_plan_batch(metadata, base, batch, 12) != single_tasks:
                valid = pl.astype(0, pl.DT_INT64)
            if batch == run_end:
                run_begin = batch
                run_end = tnd_plan_batch(metadata, base, batch, 9)
                run_m = m
                run_n = n
                run_p = p
                run_q = q
                if run_end <= batch or run_end > ci.bSize:
                    valid = pl.astype(0, pl.DT_INT64)
                copies = pl.max(0, run_end - batch) * ci.n2Size
                run_rounds = pl.astype(0, pl.DT_INT64)
                if m > 0 and n > 0:
                    if mask_mode == 3:
                        if m != 1 or n != 1:
                            pairs = copies // 2
                            run_rounds = (((2 * n - m + 3) * pairs + k - 1) // k) * m
                            run_rounds = (
                                run_rounds + (copies % 2) * ((n + k - 1) // k) * m
                            )
                    elif p + q >= 2:
                        if p >= m:
                            run_rounds = ((n * copies + k - 1) // k) * m
                        elif p + q > m:
                            fold = pl.max(0, n - q - p + 1)
                            run_rounds = (((n - fold) * copies + k - 1) // k) * m
                        else:
                            l1 = q - 1
                            l2 = pl.min(n - q + 1, m + 2 - p - q)
                            l3 = pl.max(0, pl.min(p + n - m - 1, p + q - 2))
                            overlap = pl.astype(0, pl.DT_INT64)
                            if l3 != 0 and p <= l3:
                                overlap = pl.min(l3 - p + 1, l1)
                            width = (l1 + l2 + l3 - overlap) * copies + overlap
                            run_rounds = (p + q - 1) * ((width + k - 1) // k)
                line_end = line_end + run_rounds
            if (
                tnd_plan_batch(metadata, base, batch, 8) != run_begin
                or tnd_plan_batch(metadata, base, batch, 9) != run_end
                or tnd_plan_batch(metadata, base, batch, 10) != line_end
                or m != run_m
                or n != run_n
                or p != run_p
                or q != run_q
            ):
                valid = pl.astype(0, pl.DT_INT64)
    if (
        deter_family == 12 or (deter_family == 13 and mask_mode == 4 and strategy == 1)
    ) and (mode2_rounds != rounds or ci.gSize != 1):
        valid = pl.astype(0, pl.DT_INT64)
    if (
        deter_family == 13
        and strategy == 2
        and (
            line_end != rounds
            or (single_tasks + k - 1) // k != singles
            or run_end != ci.bSize
            or ci.gSize != 1
        )
    ):
        valid = pl.astype(0, pl.DT_INT64)
    if deter_family >= 14:
        scale = ci.n2Size * ci.gSize
        if deter_family == 16 or deter_family == 18:
            scale = (ci.n2Size // 2) * ci.gSize
        total = area * scale
        previous = tnd_plan_i64(metadata, base + 48)
        maximum = pl.astype(0, pl.DT_INT64)
        batch = pl.astype(0, pl.DT_INT64)
        if previous != 0:
            valid = pl.astype(0, pl.DT_INT64)
        for core in pl.range(k):
            cut = tnd_plan_i64(metadata, base + 48 + 2 * (core + 1))
            length = cut - previous
            if cut < previous or cut > total or length > rounds:
                valid = pl.astype(0, pl.DT_INT64)
            maximum = pl.max(maximum, length)
            while (
                batch + 1 < ci.bSize
                and cut >= tnd_plan_batch(metadata, base, batch + 1, 6) * scale
            ):
                batch = batch + 1
            m = tnd_plan_batch(metadata, base, batch, 2)
            n = tnd_plan_batch(metadata, base, batch, 3)
            local = cut - tnd_plan_batch(metadata, base, batch, 6) * scale
            unit = pl.max(1, m * n * ci.gSize)
            if deter_family == 15 or deter_family == 17:
                unit = pl.max(1, (m + 1) * ci.gSize)
            if deter_family == 16 or deter_family == 18:
                unit = pl.max(1, (m + 1) * ci.gSize)
            # The mirror's final odd centre column is half a paired unit.
            endpoint = pl.astype(0, pl.DT_INT64)
            if cut == total:
                endpoint = pl.astype(1, pl.DT_INT64)
            if local % unit != 0 and endpoint == 0:
                valid = pl.astype(0, pl.DT_INT64)
            previous = cut
        if previous != total or maximum != rounds:
            valid = pl.astype(0, pl.DT_INT64)
    return valid


def init_tnd_metadata_schedule(ci, k, metadata, cuq, cuk, usedq, usedk):
    """Validate the fixed header before any candidate/array reads.

    Status: 0 accepted, 1 no tail, 2 malformed header, 3 producer rejected,
    4 family mismatch, 5 reference mode-2 cache/full-core gate rejected,
    6 no optimized candidate, 7 invalid candidate scalars, 8 invalid records,
    9 malformed or inconsistent optional normalized plan; native TND fallback.
    """
    offset = pl.astype(0, pl.DT_INT64)
    status = pl.astype(1, pl.DT_INT64)
    strategy = pl.astype(0, pl.DT_INT64)
    rounds = pl.astype(0, pl.DT_INT64)
    singles = pl.astype(0, pl.DT_INT64)
    pair_group = pl.astype(0, pl.DT_INT64)
    sparse_type = pl.astype(0, pl.DT_INT64)
    dense_window = pl.astype(0, pl.DT_INT64)
    if mask_mode == 4:
        dense_window = tnd_window_is_dense(ci, cuq, cuk, usedq, usedk)
    normal_tail = pl.astype(0, pl.DT_INT64)
    expected_family = deter_family
    if deter_family == 17:
        expected_family = 15
    elif deter_family == 18:
        expected_family = 16
    elif deter_family == 19 or deter_family == 20:
        expected_family = 16
        if ci.n2Size == 1:
            expected_family = 15
    expected = 122 + 32 * ci.bSize
    if ci.metadataWordCount >= 16:
        fag_offset = pl.astype(pl.getval(metadata, 7), pl.DT_INT64)
        # Main FAG now has 416 words (fallback at 412); WJF's plan may follow.
        # The independent, aligned LWY extension is described by head12..15.
        offset = pl.astype(pl.getval(metadata, 12), pl.DT_INT64)
        declared_words = pl.astype(pl.getval(metadata, 13), pl.DT_INT64)
        if (
            fag_offset >= 16
            and offset >= fag_offset + 416
            and offset % 16 == 0
            and declared_words == expected + 396 + 290
            and offset <= ci.metadataWordCount
            and declared_words <= ci.metadataWordCount - offset
            and pl.getval(metadata, 14) == 3
            and pl.getval(metadata, 15) == 0x46545031
        ):
            status = pl.astype(2, pl.DT_INT64)
            version = pl.getval(metadata, offset + 1)
            wire_words = expected + 396 + 290
            if (
                pl.getval(metadata, offset) == 0x46545031
                and version == 3
                and pl.getval(metadata, offset + 2) == wire_words
                and pl.getval(metadata, offset + 3) == 16
                and pl.getval(metadata, offset + 4) == k
                and pl.getval(metadata, offset + 5) == ci.bSize
                and pl.getval(metadata, offset + 9) == 16
                and pl.getval(metadata, offset + 10) == 16
                and pl.getval(metadata, offset + 11) == 37
                and pl.getval(metadata, offset + 12) == expected + 396
                and pl.getval(metadata, offset + 13) == 290
                and pl.getval(metadata, offset + 14) == 1
                and pl.getval(metadata, offset + 15) == 0x46534E31
                and wire_words <= ci.metadataWordCount - offset
            ):
                status = pl.astype(3, pl.DT_INT64)
                if pl.getval(metadata, offset + 8) == 0:
                    status = pl.astype(4, pl.DT_INT64)
                    if pl.getval(metadata, offset + 6) == expected_family:
                        if version == 3:
                            normal_tail = pl.astype(1, pl.DT_INT64)
                        strategy = pl.astype(
                            pl.getval(metadata, offset + 7), pl.DT_INT64
                        )
                        rounds = tnd_plan_scalar(metadata, offset, 0)
                        singles = tnd_plan_scalar(metadata, offset, 1)
                        pair_group = tnd_plan_scalar(metadata, offset, 6)
                        sparse_type = tnd_plan_scalar(metadata, offset, 9)
                        status = pl.astype(7, pl.DT_INT64)
                        legal = pl.astype(0, pl.DT_INT64)
                        if (
                            deter_family == 12
                            and strategy == 1
                            and ci.gSize == 1
                            and sparse_type == 0
                        ):
                            legal = pl.astype(1, pl.DT_INT64)
                        if deter_family == 13 and ci.gSize == 1:
                            if strategy == 2:
                                if (mask_mode == 3 and sparse_type == 1) or (
                                    mask_mode == 4
                                    and sparse_type == 2
                                    and dense_window == 0
                                ):
                                    legal = pl.astype(1, pl.DT_INT64)
                            if (
                                mask_mode == 4
                                and strategy == 1
                                and sparse_type == 0
                                and dense_window == 1
                            ):
                                legal = pl.astype(1, pl.DT_INT64)
                        if (
                            deter_family == 14
                            and strategy == 3
                            and ci.gSize > 1
                            and sparse_type == 0
                        ):
                            if mask_mode == 0 or (mask_mode == 4 and dense_window == 1):
                                legal = pl.astype(1, pl.DT_INT64)
                        if (
                            (deter_family == 15 or deter_family == 17)
                            and strategy == 4
                            and pair_group == 2
                            and ci.n2Size == 1
                            and ci.gSize >= 4
                            and ci.gSize % 4 == 0
                            and sparse_type == 1
                        ):
                            legal = pl.astype(1, pl.DT_INT64)
                        if (
                            (deter_family == 16 or deter_family == 18)
                            and strategy == 5
                            and pair_group == 4
                            and ci.n2Size >= 2
                            and ci.n2Size % 2 == 0
                            and ci.gSize >= 4
                            and ci.gSize % 4 == 0
                            and sparse_type == 1
                        ):
                            legal = pl.astype(1, pl.DT_INT64)
                        if (
                            legal == 1
                            and rounds > 0
                            and rounds < 4294967295
                            and singles >= 0
                            and singles <= rounds
                            and sparse_type >= 0
                            and sparse_type <= 2
                            and tnd_plan_scalar(metadata, offset, 11) == ci.n2Size
                            and tnd_plan_scalar(metadata, offset, 12) == ci.gSize
                            and tnd_plan_scalar(metadata, offset, 10) == mask_mode
                        ):
                            status = pl.astype(0, pl.DT_INT64)
                            if deter_family == 12 or (
                                deter_family == 13 and mask_mode == 4 and strategy == 1
                            ):
                                if (
                                    tnd_plan_scalar(metadata, offset, 7)
                                    <= ci.deterL2Bytes
                                    or tnd_plan_scalar(metadata, offset, 8) != k
                                ):
                                    status = pl.astype(5, pl.DT_INT64)
                        if strategy == 0:
                            status = pl.astype(6, pl.DT_INT64)
                        if status == 0:
                            if (
                                tnd_plan_records_valid(
                                    ci,
                                    k,
                                    metadata,
                                    offset,
                                    strategy,
                                    rounds,
                                    singles,
                                    sparse_type,
                                    cuq,
                                    cuk,
                                    usedq,
                                    usedk,
                                )
                                == 0
                            ):
                                status = pl.astype(8, pl.DT_INT64)
    plan = make_tnd_deter_schedule(rounds, singles, 0, 0, 0, 0, 0)
    if status != 0:
        if mask_mode == 0 or (mask_mode == 4 and dense_window == 1):
            plan = init_tnd_dense_schedule(ci, k, cuq, cuk, usedq, usedk)
        else:
            plan = init_tnd_sparse_schedule(ci, k, cuq, cuk, usedq, usedk)
        strategy = pl.astype(0, pl.DT_INT64)
        if mask_mode == 4:
            sparse_type = pl.astype(2, pl.DT_INT64)
            if dense_window == 1:
                sparse_type = pl.astype(0, pl.DT_INT64)
    plan.metaOffset = offset
    plan.strategy = strategy
    plan.protocolStatus = status
    plan.pairGroup = pair_group
    plan.sparseType = sparse_type
    if normal_tail == 1:
        plan = apply_tnd_normalized_plan(
            ci, k, metadata, offset, cuq, cuk, usedq, usedk, plan
        )
        if plan.protocolStatus == 9:
            if mask_mode == 0 or (mask_mode == 4 and dense_window == 1):
                plan = init_tnd_dense_schedule(ci, k, cuq, cuk, usedq, usedk)
            else:
                plan = init_tnd_sparse_schedule(ci, k, cuq, cuk, usedq, usedk)
            plan.metaOffset = offset
            plan.protocolStatus = pl.astype(9, pl.DT_INT64)
            plan.sparseType = sparse_type
    return plan


def tnd_plan_encode(ci, batch, head, group, x, y):
    """Zero-based logical coordinates to the existing synthetic max-shape ID."""
    return (
        ((batch * ci.n2Size + head) * ci.gSize + group) * ci.s1oS2o + y * ci.s1Outer + x
    )


def tnd_mode2_gidx(ci, c, k, r, plan, metadata):
    out = pl.astype(-1, pl.DT_INT64)
    batch = pl.astype(0, pl.DT_INT64)
    found = pl.astype(0, pl.DT_INT64)
    while batch < ci.bSize and found == 0:
        begin = tnd_plan_batch(metadata, plan.metaOffset, batch, 7)
        m = tnd_plan_batch(metadata, plan.metaOffset, batch, 2)
        n = tnd_plan_batch(metadata, plan.metaOffset, batch, 3)
        if m > 0 and n > 0:
            end = begin + ((n * ci.n2Size * ci.gSize + k - 1) // k) * m
            if r >= begin and r < end:
                delta = r - begin
                ident = (delta // m) * k + c
                if ident < n * ci.n2Size:
                    head = ident // n
                    y = ident % n
                    x = (y + delta) % m
                    out = pl.astype(
                        tnd_plan_encode(ci, batch, head, 0, x, y), pl.DT_INT64
                    )
                found = pl.astype(1, pl.DT_INT64)
        batch = batch + 1
    return out


def load_tnd_owned_batch(ci, plan, metadata):
    """Cache one accepted metadata batch in the owned task coordinate space."""
    m = tnd_plan_batch(metadata, plan.metaOffset, plan.cursorBatch, 2)
    n = tnd_plan_batch(metadata, plan.metaOffset, plan.cursorBatch, 3)
    prefix = tnd_plan_batch(metadata, plan.metaOffset, plan.cursorBatch, 6)
    scale = ci.n2Size * ci.gSize
    area = m * n
    if deter_family == 15 or deter_family == 17:
        area = m * (m + 1) // 2
    if deter_family == 16 or deter_family == 18:
        n = pl.min(m, n)
        area = (2 * m - n + 1) * n
        scale = (ci.n2Size // 2) * ci.gSize
    plan.cursorM = pl.astype(m, pl.DT_INT64)
    plan.cursorN = pl.astype(n, pl.DT_INT64)
    plan.cursorArea = pl.astype(area, pl.DT_INT64)
    plan.cursorBegin = pl.astype(prefix * scale, pl.DT_INT64)
    plan.cursorEnd = pl.astype(plan.cursorBegin + area * scale, pl.DT_INT64)


def seek_tnd_owned_batch(ci, task, plan, metadata):
    """Seek a zero-based task; phase 6 keeps owned and fallback cursors apart.

    Increasing physical rounds only cross each batch once. Reverse seeking
    preserves random-access decoder checks, including empty batch intervals.
    The caller guarantees a positive batch count before entering this helper.
    """
    reset_tnd_batch_cursor(plan, 6)
    if plan.cursorReady == 0:
        load_tnd_owned_batch(ci, plan, metadata)
        plan.cursorReady = pl.astype(1, pl.DT_INT64)
    while task >= plan.cursorEnd and plan.cursorBatch + 1 < ci.bSize:
        plan.cursorBatch = pl.astype(plan.cursorBatch + 1, pl.DT_INT64)
        load_tnd_owned_batch(ci, plan, metadata)
    while task < plan.cursorBegin and plan.cursorBatch > 0:
        plan.cursorBatch = pl.astype(plan.cursorBatch - 1, pl.DT_INT64)
        load_tnd_owned_batch(ci, plan, metadata)


def tnd_owned_gidx(ci, c, k, r, plan, metadata):
    out = pl.astype(-1, pl.DT_INT64)
    if plan.ownedCore != c:
        plan.ownedBegin = tnd_plan_i64(metadata, plan.metaOffset + 48 + 2 * c)
        plan.ownedEnd = tnd_plan_i64(metadata, plan.metaOffset + 48 + 2 * (c + 1))
        plan.ownedCore = pl.astype(c, pl.DT_INT64)
        plan.cursorPhase = pl.astype(-1, pl.DT_INT64)
    begin = plan.ownedBegin
    end = plan.ownedEnd
    if begin >= 0 and end >= begin and r < end - begin and ci.bSize > 0:
        task = begin + r
        seek_tnd_owned_batch(ci, task, plan, metadata)
        batch = plan.cursorBatch
        m = plan.cursorM
        n = plan.cursorN
        local = task - plan.cursorBegin
        if m > 0 and n > 0 and local >= 0 and task < plan.cursorEnd:
            head = pl.astype(0, pl.DT_INT64)
            group = pl.astype(0, pl.DT_INT64)
            x = pl.astype(0, pl.DT_INT64)
            y = pl.astype(0, pl.DT_INT64)
            if deter_family == 14:
                group_tasks = m * n * ci.gSize
                head = local // group_tasks
                rem = local % group_tasks
                y = rem // (ci.gSize * m)
                group = (rem % (ci.gSize * m)) // m
                x = rem % m
            elif deter_family == 15 or deter_family == 17:
                height = m + 1
                column_tasks = height * (ci.gSize // 2)
                column = local // column_tasks
                yy = pl.astype(m - column // 2, pl.DT_INT64)
                if column % 2 == 0:
                    yy = column // 2 + 1
                z = (r + yy) % column_tasks
                a = z // plan.pairGroup
                head_pair = plan.pairGroup * (a // height) + z % plan.pairGroup
                xx = a % height + 1
                if xx >= yy + 1:
                    group = 2 * head_pair
                    x = xx - 2
                    y = yy - 1
                else:
                    group = 2 * head_pair + 1
                    x = m - xx
                    y = m - yy
            else:
                height = 2 * m - n + 1
                column = local // (height * ci.gSize)
                pair = column // n
                yy = column % n + 1
                a = r // 4
                group = 4 * ((a // height) % (ci.gSize // 4)) + r % 4
                xx = (a + c + 1) % height + 1
                if xx >= yy + m - n + 1:
                    head = 2 * pair
                    x = xx - (m - n + 1) - 1
                    y = yy - 1
                else:
                    head = 2 * pair + 1
                    x = m - xx
                    y = n - yy
            out = pl.astype(tnd_plan_encode(ci, batch, head, group, x, y), pl.DT_INT64)
    return out


def tnd_line_dense(k, m, n, copies, c, r):
    valid = pl.astype(0, pl.DT_INT64)
    w = pl.astype(0, pl.DT_INT64)
    x = pl.astype(0, pl.DT_INT64)
    y = pl.astype(0, pl.DT_INT64)
    if m > 0 and n > 0 and copies > 0 and r >= 0:
        ident = (r // m) * k + c
        if ident < n * copies:
            w = ident // n
            y = ident % n
            x = (y + r) % m
            valid = pl.astype(1, pl.DT_INT64)
    return valid, w, x, y


def tnd_line_gidx(ci, c, k, r, plan, metadata):
    out = pl.astype(-1, pl.DT_INT64)
    batch = pl.astype(0, pl.DT_INT64)
    found = pl.astype(0, pl.DT_INT64)
    if r < plan.r0:
        task = r * k + c
        while batch < ci.bSize and found == 0:
            begin = tnd_plan_batch(metadata, plan.metaOffset, batch, 11)
            end = tnd_plan_batch(metadata, plan.metaOffset, batch, 12)
            if task >= begin and task < end:
                out = pl.astype(
                    tnd_plan_encode(ci, batch, task - begin, 0, 0, 0), pl.DT_INT64
                )
                found = pl.astype(1, pl.DT_INT64)
            batch = batch + 1
    else:
        begin = plan.r0
        while batch < ci.bSize and found == 0:
            run_end = tnd_plan_batch(metadata, plan.metaOffset, batch, 9)
            end = tnd_plan_batch(metadata, plan.metaOffset, batch, 10)
            if r >= begin and r < end and run_end > batch and run_end <= ci.bSize:
                m = tnd_plan_batch(metadata, plan.metaOffset, batch, 13)
                n = tnd_plan_batch(metadata, plan.metaOffset, batch, 14)
                p = tnd_plan_batch(metadata, plan.metaOffset, batch, 4)
                q = tnd_plan_batch(metadata, plan.metaOffset, batch, 5)
                copies = (run_end - batch) * ci.n2Size
                local = r - begin
                valid = pl.astype(0, pl.DT_INT64)
                w = pl.astype(0, pl.DT_INT64)
                x = pl.astype(0, pl.DT_INT64)
                y = pl.astype(0, pl.DT_INT64)
                if mask_mode == 3:
                    vm = m
                    vn = 2 * n - m + 3
                    pairs = copies // 2
                    pair_rounds = ((vn * pairs + k - 1) // k) * vm
                    if local < pair_rounds:
                        valid, pair, x, y = tnd_line_dense(k, vm, vn, pairs, c, local)
                        w = 2 * pair
                        if y + 1 >= x + 1 + (n + 1) - m + 1:
                            y = 2 * (n + 1) - m - (y + 1) + 1
                            x = m - x - 1
                            w = 2 * pair + 1
                    else:
                        valid, unused, x, y = tnd_line_dense(
                            k, m, n, 1, c, local - pair_rounds
                        )
                        w = copies - 1
                    if y >= n or y > x + n - m + 1:
                        valid = pl.astype(0, pl.DT_INT64)
                elif p >= m:
                    valid, w, x, y = tnd_line_dense(k, m, n, copies, c, local)
                    if y > x + q - 1:
                        valid = pl.astype(0, pl.DT_INT64)
                elif p + q > m:
                    l1 = m - p
                    l2 = p + q - m
                    l3 = n - l1 - l2
                    fold = pl.max(0, l3 - p + 1)
                    valid, w, x, y = tnd_line_dense(k, m, n - fold, copies, c, local)
                    if fold > 0 and y < fold and x + 1 >= p + y + 1:
                        y = n - fold + y
                    elif y < l1 and x + 1 >= p + y + 1:
                        valid = pl.astype(0, pl.DT_INT64)
                    if y + 1 > l1 + l2 and x + 1 <= y + 1 - l1 - l2:
                        valid = pl.astype(0, pl.DT_INT64)
                else:
                    seg = p + q - 1
                    if seg > 0:
                        layer = local // seg
                        phase = local % seg + 1
                        l3 = pl.max(0, pl.min(p + n - m - 1, p + q - 2))
                        ident = layer * k + c + 1
                        if l3 == 0 or n - m < 1:
                            w = (ident - 1) // n
                            y = (ident - 1) % n
                            x = y + phase - q
                            valid = pl.astype(1, pl.DT_INT64)
                        else:
                            x_abs = ident + phase - q
                            if x_abs >= 1:
                                w = (x_abs - 1) // m
                                x = (x_abs - 1) % m
                                y = x + q - phase
                                valid = pl.astype(1, pl.DT_INT64)
                if (
                    valid == 1
                    and w >= 0
                    and w < copies
                    and x >= 0
                    and x < m
                    and y >= 0
                    and y < n
                ):
                    target_batch = batch + w // ci.n2Size
                    raw_m = tnd_plan_batch(metadata, plan.metaOffset, target_batch, 2)
                    raw_n = tnd_plan_batch(metadata, plan.metaOffset, target_batch, 3)
                    if mask_mode == 3:
                        x = x + pl.max(0, raw_m - raw_n - 1)
                    else:
                        shift = tnd_plan_batch(
                            metadata, plan.metaOffset, target_batch, 15
                        )
                        x = x + pl.max(0, shift)
                        y = y + pl.max(0, -shift)
                    out = pl.astype(
                        tnd_plan_encode(ci, target_batch, w % ci.n2Size, 0, x, y),
                        pl.DT_INT64,
                    )
                found = pl.astype(1, pl.DT_INT64)
            begin = end
            # A corrupt run range must never produce an infinite decoder loop.
            batch = pl.max(batch + 1, run_end)
    return out


def tnd_source_valid_gidx(ci, index, cuq, cuk, usedq, usedk):
    """Apply AscendC IsValidDeterForTnd after decoding a physical round."""
    if mask_mode == 3 or mask_mode == 4:
        if index >= 0:
            batch = index // (ci.n2Size * ci.gSize * ci.s1oS2o)
            x = index % ci.s1Outer
            y = (index % ci.s1oS2o) // ci.s1Outer
            actual_q, actual_k = tnd_deter_lengths(ci, batch, cuq, cuk, usedq, usedk)
            pre = ci.s1Token - ci.s1Size + ci.s2Size + actual_q - actual_k
            nxt = ci.s2Token + ci.s1Size - ci.s2Size - actual_q + actual_k
            left = pl.max(x * 128 - pre, 0) // 64 * 64
            right = (pl.min((x + 1) * 128, ci.s1Size) + nxt + 63) // 64 * 64
            right = pl.min(right, actual_k)
            if y * 128 >= right or pl.min((y + 1) * 128, actual_k) <= left:
                index = pl.astype(-1, pl.DT_INT64)
    return index


def tnd_metadata_gidx(ci, c, k, r, plan, metadata, cuq, cuk, usedq, usedk):
    out = pl.astype(-1, pl.DT_INT64)
    if c >= 0 and c < k and r >= 0 and r < plan.maxRound:
        if plan.normalized == 1:
            out = tnd_normalized_gidx(ci, c, k, r, plan, metadata)
        elif plan.strategy == 0:
            if mask_mode == 0 or (mask_mode == 4 and plan.sparseType == 0):
                out = tnd_dense_gidx(ci, c, k, r, plan, cuq, cuk, usedq, usedk)
            else:
                out = tnd_sparse_gidx(ci, c, k, r, plan, cuq, cuk, usedq, usedk)
        elif deter_family == 12:
            out = tnd_mode2_gidx(ci, c, k, r, plan, metadata)
        elif deter_family == 13:
            # Full finite windows normalize to Dense only after actual lengths
            # are known. Keep precisely mode2 + line in this mask4 family.
            if mask_mode == 4 and plan.strategy == 1:
                out = tnd_mode2_gidx(ci, c, k, r, plan, metadata)
            else:
                out = tnd_line_gidx(ci, c, k, r, plan, metadata)
        else:
            out = tnd_owned_gidx(ci, c, k, r, plan, metadata)
        if plan.normalized == 0:
            out = tnd_source_valid_gidx(ci, out, cuq, cuk, usedq, usedk)
    return out


# BN2S2-specific implementation. Names remain distinct from the public family.
# Resource layout, metadata protocol and task order are preserved.

try:
    import pypto_pro.language as pl_bn2s2
except ImportError:  # host 对拍
    from types import SimpleNamespace as _SimpleNamespace

    class _Pl:
        DT_INT64 = 0

        @staticmethod
        def min(a, b):
            return a if a < b else b

        @staticmethod
        def max(a, b):
            return a if a > b else b

        @staticmethod
        def astype(x, _t):
            return int(x)

        range = staticmethod(range)

        @staticmethod
        def getval(tensor, index):
            return tensor[index]

        @staticmethod
        def struct(_name, **values):
            return _SimpleNamespace(**values)

    pl_bn2s2 = _Pl()

CUBE_BASEM_bn2s2 = 128
CUBE_BASEN_bn2s2 = 128


def cal_gqa_dense_bn2s2(k, m, n, b, core_id, round_id, g, dense_round=0):
    return cal_gqa_dense_uncached(k, m, n, b, core_id, round_id, g, dense_round)


def deter_lin_max_round_bn2s2(constInfo, coreNum):
    return deter_lin_max_round(constInfo, coreNum, True)


def _fused_valid_bn2s2(constInfo, s1o, s2o):
    return fused_valid_for_mask(
        constInfo, s1o, s2o, mask_mode, CUBE_BASEM_bn2s2, CUBE_BASEN_bn2s2
    )


def _deter_lin_gidx_host_bn2s2(constInfo, cBlockIdx, coreNum, roundId):
    return _deter_lin_gidx_host(
        constInfo, cBlockIdx, coreNum, roundId, 0, 0, 0, 0, True
    )


def deter_lin_gidx_bn2s2(constInfo, cBlockIdx, coreNum, roundId):
    """(cBlockIdx, roundId) 0-based → fusedIdx，空槽 -1。

    无 seqused 时保留对应的 AscendC dense/causal/band 闭式。
    seqused 的 batch 几何在核上读取；先枚举完整物理 dense 网格，再由
    skip_empty_deter 按有效长度和 mask 过滤，避免静态稀疏几何提前漏块。
    """
    k = coreNum
    j = cBlockIdx + 1
    r = roundId + 1
    b = constInfo.bSize
    n2 = constInfo.n2Size
    g = pl_bn2s2.max(constInfo.gSize, 1)
    s1o = constInfo.s1Outer
    s2o = constInfo.s2Outer
    bb = b * n2
    moff = 0
    noff = 0
    dense_grid = constInfo.sequsedQOptional + constInfo.sequsedKvOptional
    if mask_mode == 0:
        dense_grid = 1
    if dense_grid > 0:
        w, x, y = cal_dense(k, s1o, s2o, bb, j, r)
        dense_round = select_gqa_dense_round(k, s1o, s2o, bb, g)
        wg, xg, yg = cal_gqa_dense_bn2s2(k, s1o, s2o, bb, j, r, g, dense_round)
    else:
        w, x, y, wg, xg, yg, moff, noff = _deter_sparse_coordinates(
            constInfo, k, j, r, bb, g, s1o, s2o, True
        )
    is_mha = _in_range(g, 1, 1)
    w = w * is_mha + wg * (1 - is_mha)
    x = x * is_mha + xg * (1 - is_mha)
    y = y * is_mha + yg * (1 - is_mha)
    fused = pack_fused(w, x, y, b, n2, g, s1o, s2o, moff, noff)
    s1s = pl_bn2s2.max(s1o, 1)
    s2s = pl_bn2s2.max(s2o, 1)
    s1_idx = fused % s1s
    s2_idx = (fused % (s1s * s2s)) // s1s
    hit = pl_bn2s2.min(pl_bn2s2.max(fused + 1, 0), 1)
    if constInfo.sequsedQOptional + constInfo.sequsedKvOptional == 0:
        hit = hit * _fused_valid_bn2s2(constInfo, s1_idx, s2_idx)
    return pl_bn2s2.astype(fused, pl_bn2s2.DT_INT64) * hit + pl_bn2s2.astype(
        -1, pl_bn2s2.DT_INT64
    ) * (1 - hit)


_deter_lin_gidx_device_bn2s2 = deter_lin_gidx_bn2s2


if type(pl_bn2s2).__name__ == "_Pl":
    deter_lin_gidx_bn2s2 = _deter_lin_gidx_host_bn2s2


# ==================================================================
# BN2S2 TND metadata 确定性：协议、坐标与写回顺序
# ==================================================================

# 字段须与 flash_attn_metadata/op_kernel_aicpu/flash_attn_metadata.h 一致。
# logicalSplit 来自 metadata；这里解码既定分核，不重新选择模板或轮数。
PLAN_MAGIC_bn2s2 = 0x46414744504C414E
PLAN_VERSION_bn2s2 = 5
PLAN_WORDS_bn2s2 = 1365
PLAN_BYTES_bn2s2 = 10920


class PF_bn2s2:
    """Signed int64 word offsets into the internal plan Tensor."""

    magic = 0
    version = 1
    byteSize = 2
    status = 3
    compiledFamily = 4
    initialSplit = 5
    logicalSplit = 6
    fallbackReasons = 7
    batch = 8
    totalQ = 9
    totalKv = 10
    maxQ = 11
    maxKv = 12
    headsQ = 13
    headsKv = 14
    dimQ = 15
    dimV = 16
    alignQ = 17
    alignV = 18
    supportedDtypeMask = 19
    maskMode = 20
    windowLeft = 21
    windowRight = 22
    sink = 23
    aicNum = 24
    aivNum = 25
    l2Bytes = 26
    reservedPlanMode = 27
    effectiveLayout = 28
    sparseType = 29
    deterType = 30
    s1Token = 31
    s2Token = 32
    s1Inner = 33
    s2Inner = 34
    s1CvRatio = 35
    s2CvRatio = 36
    scheduleKind = 37
    isSwizzle = 38
    lineDeter = 39
    coreDivide = 40
    maxRound = 41
    blockOuter = 42
    blockFactor = 43
    prefixStep = 44
    tailZeroCount = 45
    hasZeroSequence = 46
    invalidRow = 47
    invalidCol = 48
    ascWriteFormat = 49
    writeFormat = 50
    workspaceBytes = 51
    dqOffset = 52
    dqBytes = 53
    dkOffset = 54
    dkBytes = 55
    dvOffset = 56
    dvBytes = 57
    scratchStride = 58
    specialOffset = 59
    specialBytes = 60
    specialCount = 61
    enablePreSfmg = 62
    sfmgOffset = 63
    sfmgBytes = 64
    sfmgRows = 65
    sfmgStride = 66
    sinkOffset = 67
    sinkBytes = 68
    sinkS1Outer = 69
    sinkS2Outer = 70
    sinkReduceAxis = 71
    qPreBlockFactor = 72
    qPreBlockTotal = 73
    kPreBlockFactor = 74
    kPreBlockTotal = 75
    vPreBlockFactor = 76
    vPreBlockTotal = 77
    qPostBlockFactor = 78
    qPostBlockTotal = 79
    kPostBlockFactor = 80
    kPostBlockTotal = 81
    vPostBlockFactor = 82
    vPostBlockTotal = 83
    blockStarts = 84
    blockEnds = 120
    startBatch = 156
    startNeedSyncRound = 192
    endNeedSyncRound = 228
    separateDkOffset = 264
    deterPrefix = 300
    deterPrefixAlign = 432
    deterPrefix0 = 564
    deterPrefix1 = 696
    deterPrefix2 = 828
    swizzleAreaPrefix = 960
    swizzleAlignAreaPrefix = 1089
    swizzleRoundPrefix = 1218
    libapiBytes = 1347
    ubBytes = 1348
    maskPresent = 1349
    maskShapeType = 1350
    sValueZeroUnderTnd = 1351
    s1Outer = 1352
    s2Outer = 1353
    sfmgUsedCoreNum = 1354
    sfmgNormalCoreRows = 1355
    sfmgTailCoreRows = 1356
    sfmgLoopRows = 1357
    sfmgDyBufferLen = 1358
    sfmgOutputBufferLen = 1359
    postBaseElements = 1360
    sinkPreBlockFactor = 1361
    sinkPreBlockTotal = 1362
    sinkPostBlockFactor = 1363
    sinkPostBlockTotal = 1364


class PM_bn2s2:
    """Physical metadata extension defined by the common C++ headers."""

    MAGIC = 5062405853342351666
    VERSION = 2
    HEAD_OFFSET_INDEX = 10
    HEAD_MAGIC_INDEX = 11
    HEAD_MAGIC = 0x46414733
    HEADER_WORDS = 16
    ARRAY_BEGIN = 84
    ARRAY_END = 1347
    ARRAY_WORDS = 1263
    SCALAR_WORDS = 102
    BANK_WORDS = 1467
    BANK_COUNT = 2
    SINK_VARIANTS = 2
    BYTES = 23616
    I32_WORDS = 5904


class PlanSplit_bn2s2:
    GS = 0
    BN2 = 1
    BN2S2 = 3


class PlanDType_bn2s2:
    FP16 = 0
    BF16 = 1
    F16_ANY = 2


class PlanStatus_bn2s2:
    INVALID = 0
    READY = 1


class PlanWriteFormat_bn2s2:
    UNKNOWN = -1
    ND = 0
    NZ = 1


class PlanSchedule_bn2s2:
    DENSE = 0
    CAUSAL = 1
    BAND = 2
    LINE = 3
    DENSE_SWIZZLE = 4


class PlanFallback_bn2s2:
    NONE = 0
    SUPPORT_BN2S2 = 1
    SYNC_COST = 2
    UNSATURATED = 4
    CAUSAL = 8
    LINE = 16


def plan_i64_bn2s2(words, field):
    # Scalar field IDs are compile-time constants at every call site.
    index = field
    if field >= PM_bn2s2.ARRAY_END:
        index = field - PM_bn2s2.ARRAY_WORDS
    return pl_bn2s2.astype(pl_bn2s2.getval(words.scalars, index), pl_bn2s2.DT_INT64)


def plan_array_i64_bn2s2(words, field):
    return pl_bn2s2.astype(
        pl_bn2s2.getval(words.schedule, field - PM_bn2s2.ARRAY_BEGIN), pl_bn2s2.DT_INT64
    )


def seq_i64_bn2s2(sequence, index):
    return pl_bn2s2.astype(pl_bn2s2.getval(sequence, index), pl_bn2s2.DT_INT64)


def metadata_plan_views_bn2s2(metadata, metadata_bytes, mask_present, key_sink):
    # Descriptors and complete plan validation belong to host/the producer.
    # Retain only a uniform ABI gate before consuming offsets or collectives.
    # The caller has already checked the full extension's minimum capacity.
    head = pl_bn2s2.make_tensor(metadata, [16], [1], dtype=pl_bn2s2.DT_INT32)
    # head[8..9] is owned by FA initialization; directory v3 uses [10..11].
    offset = seq_i64_bn2s2(head, PM_bn2s2.HEAD_OFFSET_INDEX)
    valid = (
        seq_i64_bn2s2(head, PM_bn2s2.HEAD_MAGIC_INDEX) == PM_bn2s2.HEAD_MAGIC
        and offset >= 16
        and offset <= metadata_bytes // 4 - PM_bn2s2.I32_WORDS
        and offset % 16 == 0
    )
    if not valid:
        offset = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    payload = pl_bn2s2.make_ptr(
        pl_bn2s2.addptr(metadata, offset), dtype=pl_bn2s2.DT_INT64
    )
    header = pl_bn2s2.make_tensor(
        payload, [PM_bn2s2.HEADER_WORDS], [1], dtype=pl_bn2s2.DT_INT64
    )
    valid = (
        valid
        and seq_i64_bn2s2(header, 0) == PM_bn2s2.MAGIC
        and seq_i64_bn2s2(header, 1) == PM_bn2s2.VERSION
        and seq_i64_bn2s2(header, 3) == PlanStatus_bn2s2.READY
        and seq_i64_bn2s2(header, 4) == PLAN_VERSION_bn2s2
    )
    bank = PM_bn2s2.HEADER_WORDS + mask_present * PM_bn2s2.BANK_WORDS
    schedule = pl_bn2s2.make_tensor(
        pl_bn2s2.addptr(payload, bank),
        [PM_bn2s2.ARRAY_WORDS],
        [1],
        dtype=pl_bn2s2.DT_INT64,
    )
    scalar_base = bank + PM_bn2s2.ARRAY_WORDS + key_sink * PM_bn2s2.SCALAR_WORDS
    scalars = pl_bn2s2.make_tensor(
        pl_bn2s2.addptr(payload, scalar_base),
        [PM_bn2s2.SCALAR_WORDS],
        [1],
        dtype=pl_bn2s2.DT_INT64,
    )
    return pl_bn2s2.make_tuple(schedule=schedule, scalars=scalars, valid=valid)


def plan_execution_header_bn2s2(words):
    return pl_bn2s2.struct(
        "FagPlanExecution",
        batch=plan_i64_bn2s2(words, PF_bn2s2.batch),
        heads=plan_i64_bn2s2(words, PF_bn2s2.headsQ),
        cores=plan_i64_bn2s2(words, PF_bn2s2.aicNum),
        maxRound=plan_i64_bn2s2(words, PF_bn2s2.maxRound),
        step=plan_i64_bn2s2(words, PF_bn2s2.prefixStep),
        logicalSplit=plan_i64_bn2s2(words, PF_bn2s2.logicalSplit),
        deterType=plan_i64_bn2s2(words, PF_bn2s2.deterType),
        line=plan_i64_bn2s2(words, PF_bn2s2.lineDeter),
        swizzle=plan_i64_bn2s2(words, PF_bn2s2.isSwizzle),
        maskMode=plan_i64_bn2s2(words, PF_bn2s2.maskMode),
        tokenQ=plan_i64_bn2s2(words, PF_bn2s2.s1Token),
        tokenKv=plan_i64_bn2s2(words, PF_bn2s2.s2Token),
        maxQ=plan_i64_bn2s2(words, PF_bn2s2.maxQ),
        maxKv=plan_i64_bn2s2(words, PF_bn2s2.maxKv),
        dimQ=plan_i64_bn2s2(words, PF_bn2s2.dimQ),
        dimV=plan_i64_bn2s2(words, PF_bn2s2.dimV),
        totalQ=plan_i64_bn2s2(words, PF_bn2s2.totalQ),
        sinkS1Outer=plan_i64_bn2s2(words, PF_bn2s2.sinkS1Outer),
        sinkS2Outer=plan_i64_bn2s2(words, PF_bn2s2.sinkS2Outer),
        ascPairBatch=pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64),
        ascPairM=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        ascPairN=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        fixedBatch=pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64),
        fixedRows=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        fixedCols=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        fixedQBase=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        fixedKvBase=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        fixedSq=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        fixedSk=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
    )


def plan_header_matches_bn2s2(words, core_num, capacity):
    """Check launch-dependent limits once, before any core enters a barrier.

    Host validates descriptors and specialization; metadata validates all plan
    fields before publishing READY. It never incorporates runtime-lib workspace.
    A different stream core limit or insufficient allocation cannot consume it.
    """
    return (
        words.valid
        and plan_i64_bn2s2(words, PF_bn2s2.aicNum) == core_num
        and plan_i64_bn2s2(words, PF_bn2s2.workspaceBytes) <= capacity
    )


def plan_coordinate_bn2s2():
    return pl_bn2s2.struct(
        "FagPlanCoordinate",
        batch=pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64),
        head=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        row=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        col=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        sq=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        sk=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        rows=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        cols=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        m=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        n=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        p=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        q=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        mo=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
        no=pl_bn2s2.astype(0, pl_bn2s2.DT_INT64),
    )


def plan_gcd_bn2s2(a, b):
    while b > 0:
        remainder = a % b
        a = b
        b = remainder
    return a


def plan_ceil_signed_bn2s2(a, b):
    # ASC Ceil is (a+b-1)/b with C++ division toward zero, even for a<0.
    numerator = a + b - 1
    result = numerator // b
    if numerator < 0:
        result = -((-numerator) // b)
    return result


def plan_load_geometry_bn2s2(c, cuq, cuk, batch):
    c.sq = seq_i64_bn2s2(cuq, batch + 1) - seq_i64_bn2s2(cuq, batch)
    c.sk = seq_i64_bn2s2(cuk, batch + 1) - seq_i64_bn2s2(cuk, batch)
    c.rows = (c.sq + 127) // 128
    c.cols = (c.sk + 127) // 128
    c.m = c.rows
    c.n = c.cols
    c.mo = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    c.no = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)


def plan_band_geometry_bn2s2(c, h, dense):
    """UpdateMNPQ<128,128,isDense,true>, without the draft's bound changes."""
    pre = h.tokenQ
    post = h.tokenKv
    if h.maskMode == 3 or h.maskMode == 4:
        pre = pre + c.sq - c.sk
        post = post - c.sq + c.sk
    m = c.rows
    n = c.cols
    p = pl_bn2s2.min(plan_ceil_signed_bn2s2(pre, 128) + 1, m)
    q = pl_bn2s2.min(plan_ceil_signed_bn2s2(post, 128) + 1, n)
    if p < 0:
        c.no = -p
        n = n + p
        q = p + q
        p = pl_bn2s2.astype(1, pl_bn2s2.DT_INT64)
    elif q < 0:
        c.mo = -q
        m = m + q
        p = p + q
        q = pl_bn2s2.astype(1, pl_bn2s2.DT_INT64)
    actual_m = m
    actual_n = pl_bn2s2.min(m - 1 + q, n)
    if p + q <= m:
        l1 = q - 1
        l2 = pl_bn2s2.min(n - q + 1, m + 2 - p - q)
        l3 = pl_bn2s2.max(0, pl_bn2s2.min(p + n - m - 1, p + q - 2))
        if l3 == 0:
            actual_m = p + q + l2 - 2
        actual_n = l1 + l2 + l3
    c.m = actual_m
    c.n = actual_n
    c.p = p
    c.q = q
    vm = actual_m
    vn = actual_n
    if dense:
        if p + q <= actual_m:
            if actual_n >= actual_m:
                vn = p + q - 1
            else:
                vm = p + q - 1
        elif p + q <= actual_n:
            vn = p + q - 1
    return vm, vn


def plan_virtual_shape_bn2s2(c, h, kind, flag):
    m = c.rows
    n = c.cols
    if kind == 3:
        n = pl_bn2s2.min(m, n)
        if flag == 0:
            m = 2 * m - n + 1
        elif flag == 1:
            m = m - (n + 1) // 2 + 1
            n = n // 2
        else:
            m = m - n // 2
            n = (n + 1) // 2
    elif kind == 4:
        m, n = plan_band_geometry_bn2s2(c, h, True)
    return m, n


def plan_dense_phase_bn2s2(
    words, h, cuq, cuk, c, prefix, rounds, heads, core, r, kind, flag
):
    """CalTNDDenseIndex<128,128,kind,true>; core/r are one-based."""
    c.batch = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    if r <= rounds and rounds > 0 and heads > 0:
        index = (core - 1) * rounds + r
        group = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        while (group + 1) * h.step < h.batch and index > plan_array_i64_bn2s2(
            words, prefix + group + 1
        ) * heads:
            group = group + 1
        delta = index - plan_array_i64_bn2s2(words, prefix + group) * heads
        batch = group * h.step
        if batch < h.batch:
            plan_load_geometry_bn2s2(c, cuq, cuk, batch)
            m, n = plan_virtual_shape_bn2s2(c, h, kind, flag)
            if h.step > 1:
                area = m * n * heads
                while delta > area and batch + 1 < h.batch:
                    delta = delta - area
                    batch = batch + 1
                    plan_load_geometry_bn2s2(c, cuq, cuk, batch)
                    m, n = plan_virtual_shape_bn2s2(c, h, kind, flag)
                    area = m * n * heads
            # Empty batches and unused final-core slots have no coordinate. The
            # guard avoids ASC's undefined /0 on a zero-sized final batch.
            if m > 0 and n > 0 and delta > 0 and delta <= m * n * heads:
                head = (delta - 1) // (m * n) + 1
                delta = (delta - 1) % (m * n) + 1
                common = plan_gcd_bn2s2(m, rounds)
                t1 = rounds // common
                t2 = m // common
                y = (delta - 1) // m + 1
                if t1 < n:
                    tail = (n - 1) % t1 + 1
                    if y <= n - tail:
                        adjustment = (y + t1 - 1) // t1
                        delta = delta + adjustment
                        if delta > adjustment * t2 * rounds:
                            delta = delta - t2 * rounds
                c.batch = batch * heads + head
                c.row = (delta - 1) % m + 1
                c.col = (delta - 1) // m + 1


def plan_dense_swizzle_bn2s2(words, h, cuq, cuk, c, core, r):
    """CalTNDDenseSwizzleIndex mode 2; prefix is the final AICPU plan."""
    c.batch = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    # Prefix[b+1] is the exclusive end of batch b. Equal entries skip
    # zero-work batches while batch indexes continue to address the physical cu.
    lo = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    hi = pl_bn2s2.astype(h.batch, pl_bn2s2.DT_INT64)
    while lo < hi:
        mid = (lo + hi) // 2
        if plan_array_i64_bn2s2(words, PF_bn2s2.swizzleRoundPrefix + mid + 1) < r:
            lo = mid + 1
        else:
            hi = mid
    if lo < h.batch:
        plan_load_geometry_bn2s2(c, cuq, cuk, lo)
        delta = r - 1 - plan_array_i64_bn2s2(words, PF_bn2s2.swizzleRoundPrefix + lo)
        if c.rows > 0 and c.cols > 0:
            linear = delta // c.rows * h.cores + core - 1
            if linear < c.cols * h.heads:
                head = linear // c.cols
                col = linear % c.cols
                c.batch = lo * h.heads + head + 1
                c.row = (col + delta) % c.rows + 1
                c.col = col + 1


def plan_round_suffix_bn2s2(h):
    return (h.batch + h.step) // h.step


def plan_prefix_bucket_bn2s2(words, prefix, h, target):
    group = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    last = (h.batch - 1) // h.step
    while group < last:
        mid = (group + last) // 2
        if target > plan_array_i64_bn2s2(words, prefix + mid + 1):
            group = mid + 1
        else:
            last = mid
    return group


def plan_causal_whole_position_bn2s2(c, m, n, a):
    even_n = n // 2 * 2
    length = 2 * m - even_n + 1
    local_rounds = even_n * length // 2
    x = a - local_rounds - 1 + n
    y = n
    if a <= local_rounds:
        y = (a + length - 1) // length
        rem = (a - 1) % length + 1
        x = rem + y - 1
        if x > m:
            x = 2 * m + 1 - x
            y = even_n + 1 - y
    c.row = x
    c.col = y


def plan_causal_phase_bn2s2(words, h, cuq, cuk, c, core, r):
    """CalTNDCausalIndex<128,128>, MHA classic four-phase schedule."""
    suffix = plan_round_suffix_bn2s2(h)
    n10 = h.heads // h.cores
    n11 = h.heads % h.cores // 2
    r01 = plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix0 + suffix)
    r02 = plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix0 + suffix + 1)
    r0 = r01 + r02
    r1 = plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix1 + suffix)
    r2 = plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix2 + suffix)
    c.batch = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    if r <= r01:
        judge = (r * 2 + n10 - 1) // n10
        group = plan_prefix_bucket_bn2s2(words, PF_bn2s2.deterPrefix0, h, judge)
        batch = group * h.step
        a = r - plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix0 + group) * n10 // 2
        if batch < h.batch:
            plan_load_geometry_bn2s2(c, cuq, cuk, batch)
            m = c.rows
            n = pl_bn2s2.min(m, c.cols)
            area = (2 * m - n + 1) * n // 2
            while a > area * n10 and batch + 1 < h.batch:
                a = a - area * n10
                batch = batch + 1
                plan_load_geometry_bn2s2(c, cuq, cuk, batch)
                m = c.rows
                n = pl_bn2s2.min(m, c.cols)
                area = (2 * m - n + 1) * n // 2
            if area > 0 and a <= area * n10:
                plan_causal_whole_position_bn2s2(c, m, n, (a - 1) % area + 1)
                w = (a - 1) // area * h.cores + core
                w = (w - 1) // h.cores * h.cores + (c.col - 1 + w - 1) % h.cores + 1
                c.batch = batch * h.heads + w
    elif r <= r0:
        plan_dense_phase_bn2s2(
            words, h, cuq, cuk, c, PF_bn2s2.deterPrefix0, r02, n11, core, r - r01, 3, 0
        )
        if c.batch > 0:
            m = c.rows
            n = pl_bn2s2.min(m, c.cols)
            batch = (c.batch - 1) // n11
            head = (c.batch - 1) % n11 + 1
            if c.row >= c.col + m - n + 1:
                c.row = c.row - (m - n + 1)
                head = 2 * head - 1
            else:
                c.row = m + 1 - c.row
                c.col = n + 1 - c.col
                head = 2 * head
            c.batch = batch * h.heads + n10 * h.cores + head
    elif r <= r0 + r1:
        plan_dense_phase_bn2s2(
            words, h, cuq, cuk, c, PF_bn2s2.deterPrefix1, r1, 1, core, r - r0, 3, 1
        )
        if c.batch > 0:
            m = c.rows
            n = pl_bn2s2.min(m, c.cols)
            if c.row >= c.col + m - n + 1:
                c.row = c.row - (m - n + 1)
            else:
                c.row = m + 1 - c.row
                c.col = n + 1 - c.col
            c.batch = c.batch * h.heads
    else:
        plan_dense_phase_bn2s2(
            words, h, cuq, cuk, c, PF_bn2s2.deterPrefix2, r2, 1, core, r - r0 - r1, 3, 2
        )
        if c.batch > 0:
            c.row = c.row + pl_bn2s2.min(c.rows, c.cols) // 2
            c.batch = c.batch * h.heads


def plan_abs_ceil_bn2s2(a, b):
    result = (a + b - 1) // b
    if a < 0:
        result = -((-a + b - 1) // b)
    return result


def plan_band_whole_position_bn2s2(c, m, n, p, q, a):
    """CalPosWholeBatch, retaining ASC's order in both band geometries."""
    l1 = q - 1
    l2 = pl_bn2s2.min(n - q + 1, m + 2 - p - q)
    l3 = pl_bn2s2.max(0, pl_bn2s2.min(p + n - m - 1, p + q - 2))
    r1 = (2 * p - 2 + q) * l1 // 2
    r2 = (p + q - 1) * l2
    if p + q > m:
        l1 = m - p
        l2 = p + q - m
        l3 = pl_bn2s2.min(m - 1, n - q)
        r1 = (p + m - 1) * l1 // 2
        r2 = m * l2
    x = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    y = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    if a <= r1:
        l11 = l1 // 2 * 2
        length = 2 * p + l11 - 1
        rounds = l11 * length // 2
        if a <= rounds:
            y = (a + length - 1) // length
            rem = (a - 1) % length + 1
            x = p + y - rem
            if x < 1:
                y = l11 + 1 - y
                x = 1 - x
            elif p + q <= m:
                x = p + y - x
        else:
            x = a - rounds
            y = l1
    elif a <= r1 + r2:
        a2 = a - r1
        if p + q > m:
            y = (a2 + m - 1) // m
            x = (a2 - 1) % m + 1
        else:
            width = p + q - 1
            y = (a2 + width - 1) // width
            x = (a2 - 1) % width + y
        y = y + l1
    else:
        a3 = a - r1 - r2
        l31 = l3 // 2 * 2
        length = 2 * m - l31 - 1
        if p + q <= m:
            length = 2 * (p + q) - l31 - 3
        rounds = l31 * length // 2
        if a3 <= rounds:
            y = (a3 + length - 1) // length
            rem = (a3 - 1) % length + 1
            x = y + rem
            if p + q <= m:
                x = x + 1 + m - (p + q)
            if x > m:
                y = l31 + 1 - y
                if p + q > m:
                    x = x - m + y
                else:
                    x = x - (p + q - 1) + y
        else:
            x = a3 - rounds + l3
            if p + q <= m:
                x = x + m - (p + q - 1)
            y = l3
        y = y + l1 + l2
    c.row = x
    c.col = y


def plan_band_phase_bn2s2(words, h, cuq, cuk, c, core, r):
    """CalTNDBandIndex<128,128>, including whole-head and remainder phases."""
    suffix = plan_round_suffix_bn2s2(h)
    r0 = plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix0 + suffix)
    r1 = plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix1 + suffix)
    n11 = h.heads // h.cores
    n12 = h.heads % h.cores
    c.batch = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    if r <= r0:
        group = plan_prefix_bucket_bn2s2(words, PF_bn2s2.deterPrefix0, h, r)
        a = r - plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix0 + group)
        batch = group * h.step
        found = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        area = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        while batch < h.batch:
            plan_load_geometry_bn2s2(c, cuq, cuk, batch)
            m, n = plan_band_geometry_bn2s2(c, h, False)
            p = c.p
            q = c.q
            area = m * n - (m - p) * (m - p + 1) // 2 - (n - q) * (n - q + 1) // 2
            if area > 0 and a <= area * n11:
                found = pl_bn2s2.astype(1, pl_bn2s2.DT_INT64)
                break
            a = a - area * n11
            batch = batch + 1
        if found == 1:
            plan_band_whole_position_bn2s2(c, c.m, c.n, c.p, c.q, (a - 1) % area + 1)
            c.batch = batch * h.heads + (a - 1) // area * h.cores + core
    elif r <= r0 + r1:
        plan_dense_phase_bn2s2(
            words, h, cuq, cuk, c, PF_bn2s2.deterPrefix1, r1, n12, core, r - r0, 4, 0
        )
        if c.batch > 0:
            batch = (
                (c.batch - 1) // n12 * h.heads + n11 * h.cores + (c.batch - 1) % n12 + 1
            )
            m = c.m
            n = c.n
            p = c.p
            q = c.q
            x = c.row
            y = c.col
            if p + q <= m:
                if n >= m:
                    if not (y - q + 1 <= x and x <= p + y - 1):
                        c.col = (
                            plan_abs_ceil_bn2s2(x - (p + y - 1), p + q - 1)
                            * (p + q - 1)
                            + y
                        )
                else:
                    if not (x - p + 1 <= y and y <= x + q - 1):
                        c.row = (
                            plan_abs_ceil_bn2s2(y - (q + x - 1), p + q - 1)
                            * (p + q - 1)
                            + x
                        )
                c.batch = batch
            elif p + q <= n:
                if x - p + 1 <= y and y <= x + q - 1:
                    c.batch = batch
                elif y < x - p + 1 and y + p + q - 1 <= n:
                    c.batch = batch
                    c.col = y + p + q - 1
                else:
                    c.batch = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
            elif x - p + 1 <= y and y <= x + q - 1:
                c.batch = batch
            else:
                c.batch = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)


def plan_finalize_coordinate_bn2s2(c, h):
    batch = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    head = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    row = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    col = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    if c.batch > 0:
        b = (c.batch - 1) // h.heads
        n = (c.batch - 1) % h.heads
        x = c.row - 1
        y = c.col - 1
        if h.deterType == 4:
            x = x + c.mo
            y = y + c.no
        if b < h.batch and x >= 0 and x < c.rows and y >= 0 and y < c.cols:
            batch = b
            head = n
            row = x
            col = y
    return batch, head, row, col


def plan_special_column_bn2s2(words, key_offset):
    """SpecialS2Index reads the final kernel table, not the host scratch list.

    ASC swizzle clears deterPrefix2 while separateDkOffset can still contain
    the abandoned split-column candidates. Only the former is authoritative.
    """
    special = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    for slot in pl_bn2s2.range(36):
        if plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix2 + slot) == key_offset:
            special = pl_bn2s2.astype(slot, pl_bn2s2.DT_INT64)
            break
    return special


def plan_restore_special_owner_bn2s2(words, h, cuk, core, saved):
    """Recover a final writer whose own tasks were all filtered by sparsity.

    This intentionally fixes an ASC boundary gap: retirement alone cannot
    identify every special-column owner. The final table still owns the slot,
    and all producers keep their original tasks and accumulation order.
    Call only after the existing full-core barrier, before special-column post.
    """
    if (
        h.logicalSplit == PlanSplit_bn2s2.BN2S2
        and h.swizzle == 0
        and plan_i64_bn2s2(words, PF_bn2s2.specialCount) > 0
        and saved.rows == 0
    ):
        key = plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix2 + core)
        # Key zero is valid. Duplicate keys keep the same first-match owner
        # as plan_bind_run; later slots must never write the column again.
        if key >= 0 and plan_special_column_bn2s2(words, key) == core:
            token_head = key // h.dimQ
            token = token_head // h.heads
            if key % h.dimQ != 0 or token >= seq_i64_bn2s2(cuk, h.batch):
                pl_bn2s2.trap()
            else:
                # Upper bound on batch ends skips zero-length batches and
                # costs O(log B) only for an owner with no retired task.
                lo = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
                hi = h.batch
                while lo < hi:
                    mid = (lo + hi) // 2
                    if token >= seq_i64_bn2s2(cuk, mid + 1):
                        lo = mid + 1
                    else:
                        hi = mid
                if lo >= h.batch:
                    pl_bn2s2.trap()
                else:
                    begin = seq_i64_bn2s2(cuk, lo)
                    end = seq_i64_bn2s2(cuk, lo + 1)
                    if token < begin or token >= end or (token - begin) % 128 != 0:
                        pl_bn2s2.trap()
                    else:
                        saved.keyToken = token
                        saved.head = token_head % h.heads
                        saved.rows = pl_bn2s2.min(128, end - token)


def plan_needs_bn_sync_bn2s2(words, physical_loop):
    """DeterSync's inclusive physical-loop intervals, including empty rounds."""
    required = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    if physical_loop != 0:
        for slot in pl_bn2s2.range(36):
            end = plan_array_i64_bn2s2(words, PF_bn2s2.endNeedSyncRound + slot)
            if end == 0:
                break
            # ASC compares uint64_t intervals. Its all-round sentinel UINT64_MAX
            # is encoded as -1 by the signed int64 wire; it is not an empty range.
            if physical_loop >= plan_array_i64_bn2s2(
                words, PF_bn2s2.startNeedSyncRound + slot
            ) and (end == -1 or physical_loop <= end):
                required = pl_bn2s2.astype(1, pl_bn2s2.DT_INT64)
                break
    return required


def plan_line_dense_bn2s2(k, m, n, copies, j, r):
    w = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    x = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    y = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    valid = False
    if j >= 1 and j <= k and r >= 1 and copies > 0 and m > 0 and n > 0:
        delta = r - 1
        linear = delta // m * k + j
        if linear <= n * copies:
            w = (linear - 1) // n + 1
            y = (linear - 1) % n + 1
            x = (y + delta - 1) % m + 1
            valid = True
    return valid, w, x, y


def plan_line_shape_bn2s2(c, h, cuq, cuk, batch, band):
    # LoadTNDOuterMN treats either empty side as an empty rectangle.
    plan_load_geometry_bn2s2(c, cuq, cuk, batch)
    m = c.rows
    n = c.cols
    p = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    q = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    if c.sq <= 0 or c.sk <= 0:
        m = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        n = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        c.rows = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        c.cols = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    elif band:
        vm, vn = plan_band_geometry_bn2s2(c, h, False)
        if vm > 0:
            m = vm
        if vn > 0:
            n = vn
        p = c.p
        q = c.q
    else:
        m = m - pl_bn2s2.max(0, m - n - 1)
    return m, n, p, q


def plan_line_run_bn2s2(h, cuq, cuk, batch, c, band):
    m, n, p, q = plan_line_shape_bn2s2(c, h, cuq, cuk, batch, band)
    first = batch
    last = batch
    neighbor = plan_coordinate_bn2s2()
    while first > 0:
        pm, pn, pp, pq = plan_line_shape_bn2s2(neighbor, h, cuq, cuk, first - 1, band)
        if pm != m or pn != n or pp != p or pq != q:
            break
        first = first - 1
    while last + 1 < h.batch:
        nm, nn, np, nq = plan_line_shape_bn2s2(neighbor, h, cuq, cuk, last + 1, band)
        if nm != m or nn != n or np != p or nq != q:
            break
        last = last + 1
    return first, last, m, n, p, q


def plan_line_bind_bn2s2(c, h, cuq, cuk, first, copy, x, y, band, shift_row):
    batch = first + (copy - 1) // h.heads
    plan_load_geometry_bn2s2(c, cuq, cuk, batch)
    if band:
        vm, vn = plan_band_geometry_bn2s2(c, h, False)
    elif shift_row:
        x = x + pl_bn2s2.max(0, c.rows - c.cols - 1)
    c.batch = first * h.heads + copy
    c.row = x
    c.col = y


def plan_line_causal_bn2s2(words, h, cuq, cuk, c, core, r):
    c.batch = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    if r <= plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix0):
        task = (r - 1) * h.cores + core
        bucket = plan_prefix_bucket_bn2s2(words, PF_bn2s2.deterPrefix1, h, task)
        remaining = task - plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix1 + bucket)
        for batch in pl_bn2s2.range(bucket * h.step, h.batch):
            m, n, p, q = plan_line_shape_bn2s2(c, h, cuq, cuk, batch, False)
            if m == 1 and n == 1:
                if remaining <= h.heads:
                    # ASC explicitly uses LEFT_UP=true for the single-tile phase.
                    plan_line_bind_bn2s2(
                        c, h, cuq, cuk, batch, remaining, 1, 1, False, False
                    )
                    break
                remaining = remaining - h.heads
    else:
        bucket = plan_prefix_bucket_bn2s2(words, PF_bn2s2.deterPrefix0, h, r)
        a = r - plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix0 + bucket)
        batch = bucket * h.step
        while batch < h.batch and a >= 1:
            first, last, m, n, p, q = plan_line_run_bn2s2(h, cuq, cuk, batch, c, False)
            if batch > first or m <= 0 or n <= 0 or (m == 1 and n == 1):
                batch = last + 1
                continue
            vm = m
            vn = 2 * n - m + 3
            total = h.heads * (last - first + 1)
            pairs = total // 2
            pair_rounds = ((vn * pairs + h.cores - 1) // h.cores) * vm
            single_rounds = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
            if total % 2 == 1:
                single_rounds = ((n + h.cores - 1) // h.cores) * m
            if a > pair_rounds + single_rounds:
                a = a - pair_rounds - single_rounds
                batch = last + 1
                continue
            if a <= pair_rounds and pairs > 0:
                valid, pair, x, y = plan_line_dense_bn2s2(
                    h.cores, vm, vn, pairs, core, a
                )
                copy = 2 * pair - 1
                if y >= x + n + 1 - m + 1:
                    y = 2 * (n + 1) - m - y + 2
                    x = m + 1 - x
                    copy = 2 * pair
                if (
                    valid
                    and copy >= 1
                    and copy <= pairs * 2
                    and x >= 1
                    and x <= m
                    and y >= 1
                    and y <= n
                    and y <= x + n - m + 1
                ):
                    plan_line_bind_bn2s2(c, h, cuq, cuk, first, copy, x, y, False, True)
            else:
                valid, w, x, y = plan_line_dense_bn2s2(
                    h.cores, m, n, 1, core, a - pair_rounds
                )
                if (
                    valid
                    and x >= 1
                    and x <= m
                    and y >= 1
                    and y <= n
                    and y <= x + n - m + 1
                ):
                    plan_line_bind_bn2s2(
                        c, h, cuq, cuk, first, total, x, y, False, True
                    )
            break


def plan_line_wide_cols_bn2s2(m, n, p, q):
    new_n = pl_bn2s2.min(m - 1 + q, n)
    l1 = m - p
    l2 = p + q - m
    l3 = new_n - l1 - l2
    return new_n - pl_bn2s2.max(0, l3 - p + 1)


def plan_line_band_rounds_bn2s2(k, copies, m, n, p, q):
    rounds = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
    if p + q >= 2:
        if p >= m:
            rounds = ((n * copies + k - 1) // k) * m
        elif p + q <= m:
            l1 = q - 1
            l2 = pl_bn2s2.min(n - q + 1, m + 2 - p - q)
            l3 = pl_bn2s2.max(0, pl_bn2s2.min(p + n - m - 1, p + q - 2))
            overlap = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
            if l3 != 0 and p <= l3:
                overlap = pl_bn2s2.min(l3 - p + 1, l1)
            rounds = (p + q - 1) * (
                ((l1 + l2 + l3 - overlap) * copies + overlap + k - 1) // k
            )
        else:
            rounds = ((plan_line_wide_cols_bn2s2(m, n, p, q) * copies + k - 1) // k) * m
    return rounds


def plan_line_band_bn2s2(words, h, cuq, cuk, c, core, r):
    c.batch = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    bucket = plan_prefix_bucket_bn2s2(words, PF_bn2s2.deterPrefix0, h, r)
    a = r - plan_array_i64_bn2s2(words, PF_bn2s2.deterPrefix0 + bucket)
    batch = bucket * h.step
    while batch < h.batch and a >= 1:
        first, last, m, n, p, q = plan_line_run_bn2s2(h, cuq, cuk, batch, c, True)
        if batch > first:
            batch = last + 1
            continue
        copies = h.heads * (last - first + 1)
        rounds = plan_line_band_rounds_bn2s2(h.cores, copies, m, n, p, q)
        if a > rounds:
            a = a - rounds
            batch = last + 1
            continue
        valid = False
        w = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        x = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        y = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        if p >= m:
            valid, w, x, y = plan_line_dense_bn2s2(h.cores, m, n, copies, core, a)
            valid = valid and y <= x + q - 1
        elif p + q > m:
            valid, w, x, y = plan_line_dense_bn2s2(
                h.cores, m, plan_line_wide_cols_bn2s2(m, n, p, q), copies, core, a
            )
            l1 = m - p
            l2 = p + q - m
            l3 = n - l1 - l2
            fold = pl_bn2s2.max(0, l3 - p + 1)
            if fold > 0 and y <= fold and x >= p + y:
                y = n - fold + y
            elif y <= l1 and x >= p + y:
                valid = False
            if y > l1 + l2 and x <= y - l1 - l2:
                valid = False
        else:
            l3 = pl_bn2s2.max(0, pl_bn2s2.min(p + n - m - 1, p + q - 2))
            seg = p + q - 1
            a1 = (a + seg - 1) // seg
            a2 = (a - 1) % seg + 1
            linear = (a1 - 1) * h.cores + core
            if l3 == 0 or n - m < 1:
                w = (linear + n - 1) // n
                y = (linear - 1) % n + 1
                x = y + a2 - q
                valid = True
            else:
                absolute_x = linear + a2 - q
                if absolute_x >= 1:
                    w = (absolute_x + m - 1) // m
                    x = (absolute_x - 1) % m + 1
                    y = x + q - a2
                    valid = True
        if valid and w >= 1 and w <= copies and x >= 1 and x <= m and y >= 1 and y <= n:
            plan_line_bind_bn2s2(c, h, cuq, cuk, first, w, x, y, True, False)
        break


def plan_admit_asc_pair_swizzle_bn2s2(h, cuq, cuk):
    # ASC SupportTrans2BS2N2GD / CalDeterMaxLoopNum: one active group,
    # RIGHT_DOWN_CAUSAL, equal heads, even head pairs, m<n, GS commit.
    # Metadata and original cu offsets remain authoritative; inputs are not compacted.
    if (
        h.logicalSplit == PlanSplit_bn2s2.GS
        and h.maskMode == 3
        and h.deterType == 4
        and h.heads > 0
        and h.heads % 2 == 0
        and h.cores > 0
    ):
        count = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        active = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
        m = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        n = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        sq = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        for b in pl_bn2s2.range(h.batch):
            qlen = seq_i64_bn2s2(cuq, b + 1) - seq_i64_bn2s2(cuq, b)
            klen = seq_i64_bn2s2(cuk, b + 1) - seq_i64_bn2s2(cuk, b)
            if qlen > 0 and klen > 0:
                count = count + 1
                active = b
                sq = qlen
                m = (qlen + 127) // 128
                n = (klen + 127) // 128
        if count == 1 and m > 0 and m < n and sq >= h.cores * 128:
            h.ascPairBatch = active
            h.ascPairM = m
            h.ascPairN = n
            nn = 2 * n - m + 3
            h.maxRound = pl_bn2s2.max(
                m * ((nn * (h.heads // 2) + h.cores - 1) // h.cores), nn
            )


def plan_admit_fixed_rectangle_bn2s2(h, cuq, cuk, mask_mode, reuse_dense=False):
    """ASC fixed-layout GS for one live, underfilled rectangle in physical TND."""
    if (
        h.logicalSplit == PlanSplit_bn2s2.GS
        and h.batch > 1
        and h.heads > 0
        and h.line == 0
        and h.swizzle == 0
        and (
            (mask_mode == 0 and h.deterType == 2)
            or (mask_mode == 3 and h.deterType == 4)
        )
    ):
        count = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        active = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
        sq = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        sk = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        active_q = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        active_k = pl_bn2s2.astype(0, pl_bn2s2.DT_INT64)
        q_end = seq_i64_bn2s2(cuq, 0)
        k_end = seq_i64_bn2s2(cuk, 0)
        for batch in pl_bn2s2.range(h.batch):
            q_base = q_end
            k_base = k_end
            q_end = seq_i64_bn2s2(cuq, batch + 1)
            k_end = seq_i64_bn2s2(cuk, batch + 1)
            qlen = q_end - q_base
            klen = k_end - k_base
            if qlen > 0 and klen > 0:
                count = count + 1
                active = batch
                sq = qlen
                sk = klen
                active_q = q_base
                active_k = k_base
                if count > 1:
                    break
        m = (sq + 127) // 128
        n = (sk + 127) // 128
        k = h.heads * m
        if count == 1 and k < h.cores:
            admit = pl_bn2s2.astype(mask_mode == 0, pl_bn2s2.DT_INT64)
            if mask_mode == 3:
                # GenBandInfo with b<cores, p=m and q=n has no whole-head
                # phase or excluded tiles. CalBandIndex calls CalDenseIndex
                # on its full tail. For m>=2,n>m the hybrid BAND candidate
                # has <50% slot utilization and DENSE cannot fill the cores.
                pre = h.tokenQ + sq - sk
                post = h.tokenKv - sq + sk
                p = plan_ceil_signed_bn2s2(pre, 128) + 1
                q = plan_ceil_signed_bn2s2(post, 128) + 1
                admit = pl_bn2s2.astype(
                    m >= 2 and n > m and p >= m and q >= n, pl_bn2s2.DT_INT64
                )
            if admit != 0:
                h.fixedBatch = active
                h.fixedRows = m
                h.fixedCols = n
                h.maxRound = ((n * h.heads + k - 1) // k) * m
                if mask_mode == 0 and reuse_dense:
                    h.fixedQBase = active_q
                    h.fixedKvBase = active_k
                    h.fixedSq = sq
                    h.fixedSk = sk


def plan_fixed_rectangle_bn2s2(h, core, round_id):
    # CalDenseIndex with zero-based inputs. Here k = Min(cores, heads*m).
    # A column's physical owner is preserved in every round of its wave.
    m = pl_bn2s2.max(h.fixedRows, 1)
    k = h.heads * m
    linear = (round_id // m) * k + core
    head = linear % h.heads
    col = linear // h.heads
    row = (col + round_id) % m
    batch = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    if (
        round_id >= 0
        and round_id < h.maxRound
        and core >= 0
        and core < k
        and col < h.fixedCols
    ):
        batch = h.fixedBatch
    return batch, head, row, col


def plan_asc_pair_swizzle_bn2s2(h, core, round_id):
    # Verbatim coordinate algebra from ASC CalDenseSwizzleIndex followed by
    # CalCausalSwizzleIndex, m!=n. 0-based public inputs/outputs.
    m = pl_bn2s2.max(h.ascPairM, 1)
    n = h.ascPairN
    nn = 2 * n - m + 3
    k = pl_bn2s2.min(h.cores, (h.heads // 2) * m)
    col_id = (round_id // m) * k + core
    pair = col_id // nn + 1
    y = col_id % nn + 1
    x = (y - 1 + round_id) % m + 1
    head = 2 * pair - 1
    if y >= x + n + 2 - m:
        y = 2 * (n + 1) - m - y + 2
        x = m + 1 - x
        head = 2 * pair
    batch = pl_bn2s2.astype(-1, pl_bn2s2.DT_INT64)
    if (
        core < k
        and pair <= h.heads // 2
        and head >= 1
        and head <= h.heads
        and x >= 1
        and x <= m
        and y >= 1
        and y <= n
    ):
        batch = h.ascPairBatch
    return batch, head - 1, x - 1, y - 1


def decode_plan_coordinate_bn2s2(words, h, cuq, cuk, core_zero, round_zero, mask_mode):
    if h.fixedBatch >= 0:
        return plan_fixed_rectangle_bn2s2(h, core_zero, round_zero)
    if h.ascPairBatch >= 0:
        return plan_asc_pair_swizzle_bn2s2(h, core_zero, round_zero)
    c = plan_coordinate_bn2s2()
    if round_zero >= 0 and round_zero < h.maxRound:
        # mask_mode is a tiling-key constant: dense binaries do not contain the
        # sparse/line engines, and mask 4 does not contain the causal engine.
        if mask_mode == 0:
            if h.swizzle != 0:
                plan_dense_swizzle_bn2s2(
                    words, h, cuq, cuk, c, core_zero + 1, round_zero + 1
                )
            else:
                plan_dense_phase_bn2s2(
                    words,
                    h,
                    cuq,
                    cuk,
                    c,
                    PF_bn2s2.deterPrefix0,
                    h.maxRound,
                    h.heads,
                    core_zero + 1,
                    round_zero + 1,
                    2,
                    0,
                )
        else:
            if mask_mode == 3 and h.deterType == 3:
                if h.line != 0:
                    plan_line_causal_bn2s2(
                        words, h, cuq, cuk, c, core_zero + 1, round_zero + 1
                    )
                else:
                    plan_causal_phase_bn2s2(
                        words, h, cuq, cuk, c, core_zero + 1, round_zero + 1
                    )
            else:
                if h.line != 0:
                    plan_line_band_bn2s2(
                        words, h, cuq, cuk, c, core_zero + 1, round_zero + 1
                    )
                else:
                    plan_band_phase_bn2s2(
                        words, h, cuq, cuk, c, core_zero + 1, round_zero + 1
                    )
    return plan_finalize_coordinate_bn2s2(c, h)


def plan_commit_header_bn2s2(words, h, core):
    """One ND workspace policy; scalar addresses select the runtime strategy."""
    align = plan_i64_bn2s2(words, PF_bn2s2.alignQ)
    dk_stride = (h.dimQ + 15) // 16 * 16
    dv_stride = (h.dimV + 15) // 16 * 16
    if h.logicalSplit == PlanSplit_bn2s2.GS:
        dk_stride = h.heads * h.dimQ
        dv_stride = h.heads * h.dimV
    return pl_bn2s2.struct(
        "FagPlanCommit",
        logicalSplit=h.logicalSplit,
        core=core,
        align=align,
        dkBase=plan_i64_bn2s2(words, PF_bn2s2.dkOffset) // 4,
        dvBase=plan_i64_bn2s2(words, PF_bn2s2.dvOffset) // 4,
        specialBase=plan_i64_bn2s2(words, PF_bn2s2.specialOffset) // 4,
        scratchStride=plan_i64_bn2s2(words, PF_bn2s2.scratchStride) // 4,
        dkStride=dk_stride,
        dvStride=dv_stride,
    )


def plan_commit_target_bn2s2(policy, info, is_dv, resident):
    """ASC IterateMm{DsQ,PDy}Fixpout ND address and AtomicNone/Add decision.

    Return FP32 element offset into user workspace, ND row stride, and the
    atomic flag. The caller shares its matmul and branches only at the store.
    The special region is interleaved [slot, dk/dv, 128, D_align], not two halves.
    """
    offset = policy.dkBase
    stride = policy.dkStride
    output_offset = info.dkOutOffset
    if is_dv:
        offset = policy.dvBase
        stride = policy.dvStride
        output_offset = info.dvOutOffset
    atomic = pl_bn2s2.astype(1, pl_bn2s2.DT_INT64)
    if policy.logicalSplit == PlanSplit_bn2s2.GS:
        offset = offset + output_offset
    else:
        offset = offset + policy.core * policy.scratchStride
        atomic = pl_bn2s2.astype(not resident and info.dkvAcc != 0, pl_bn2s2.DT_INT64)
        if info.specialIndex >= 0:
            offset = policy.specialBase + info.specialIndex * 128 * policy.align * 2
            if is_dv:
                offset = offset + 128 * policy.align
            if info.taskId != 0:
                atomic = pl_bn2s2.astype(1, pl_bn2s2.DT_INT64)
    return offset, stride, atomic
