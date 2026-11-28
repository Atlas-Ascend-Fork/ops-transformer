# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Kernel-private deterministic schedule state and cached initialization.

Shared by legacy, fixed and TND decoders. Field order and the historical
FagTndDeterSchedule struct name are preserved for code generation stability.
"""

try:
    import pypto_pro.language as pl
except ImportError:  # CPU coordinate parity tests, without a device runtime.
    from types import SimpleNamespace

    class _Pl:
        DT_INT64 = 0
        min = staticmethod(min)
        max = staticmethod(max)
        range = staticmethod(range)

        @staticmethod
        def astype(value, _dtype):
            return int(value)

        @staticmethod
        def getval(tensor, index):
            return tensor[index]

        @staticmethod
        def struct(_name, **values):
            return SimpleNamespace(**values)

    pl = _Pl()


from schedule.deterministic.fag_deter_formulas import (
    _gcd,
    select_gqa_dense_round,
    _gen_gqa_band,
    gqa_dense_parameters,
)


def init_gqa_dense_plan(plan, k, m, n, b, g, dense_round):
    """Cache immutable GQA geometry in the kernel-private schedule."""
    if k > 0 and m > 0 and n > 0 and b > 0 and g > 1:
        selected_round = pl.astype(dense_round, pl.DT_INT64)
        if selected_round <= 0:
            selected_round = select_gqa_dense_round(k, m, n, b, g)
        active_k = pl.min(pl.min(k, b * g * m), b * n)
        nbg = pl.max(b * g, 1)
        common = pl.max(_gcd(nbg, selected_round), 1)
        plan.gqaDenseK = pl.astype(active_k, pl.DT_INT64)
        plan.gqaDenseRound = pl.astype(selected_round, pl.DT_INT64)
        plan.gqaDenseT1 = pl.astype(selected_round // common, pl.DT_INT64)
        plan.gqaDenseT2 = pl.astype(nbg // common, pl.DT_INT64)
    return plan


def init_gqa_band_plan(plan, k, m, n, p, q, b, g):
    """ASC GenGQABandInfo lifetime: once before the fixed-family task loop."""
    k, m, n, p, q, b, n1, b1, b2, l1, l2, l3, rm, rm1, r1, r2, rm2 = _gen_gqa_band(
        k, m, n, p, q, b, g
    )
    plan.gqaBandK = pl.astype(k, pl.DT_INT64)
    plan.gqaBandM = pl.astype(m, pl.DT_INT64)
    plan.gqaBandN = pl.astype(n, pl.DT_INT64)
    plan.gqaBandP = pl.astype(p, pl.DT_INT64)
    plan.gqaBandQ = pl.astype(q, pl.DT_INT64)
    plan.gqaBandB = pl.astype(b, pl.DT_INT64)
    plan.gqaBandG = pl.astype(n1, pl.DT_INT64)
    plan.gqaBandB1 = pl.astype(b1, pl.DT_INT64)
    plan.gqaBandB2 = pl.astype(b2, pl.DT_INT64)
    plan.gqaBandL1 = pl.astype(l1, pl.DT_INT64)
    plan.gqaBandL2 = pl.astype(l2, pl.DT_INT64)
    plan.gqaBandL3 = pl.astype(l3, pl.DT_INT64)
    plan.gqaBandRm = pl.astype(rm, pl.DT_INT64)
    plan.gqaBandRm1 = pl.astype(rm1, pl.DT_INT64)
    plan.gqaBandR1 = pl.astype(r1, pl.DT_INT64)
    plan.gqaBandR2 = pl.astype(r2, pl.DT_INT64)
    # Exactly one tuned dense geometry can be used by this band invocation.
    # The no-tune branch keeps its original decoder and needs no GCD cache.
    dense_m = m
    dense_n = n
    tuned = pl.astype(0, pl.DT_INT64)
    if b2 > 0:
        if p + q > m:
            if p + q <= n:
                dense_n = p + q - 1
            tuned = pl.astype(1, pl.DT_INT64)
        elif n > m:
            if r2 <= r1:
                dense_n = p + q - 1
                tuned = pl.astype(1, pl.DT_INT64)
        else:
            dense_m = p + q - 1
            tuned = pl.astype(1, pl.DT_INT64)
        if tuned == 1:
            dk, dr, dt1, dt2 = gqa_dense_parameters(k, dense_m, dense_n, b2, n1, 0)
            plan.gqaBandDenseK = pl.astype(dk, pl.DT_INT64)
            plan.gqaBandDenseRound = pl.astype(dr, pl.DT_INT64)
            plan.gqaBandDenseT1 = pl.astype(dt1, pl.DT_INT64)
            plan.gqaBandDenseT2 = pl.astype(dt2, pl.DT_INT64)
    return plan


def make_tnd_deter_schedule(max_round, r0, r1, r2, core_divide, is_causal, r_whole):
    """Kernel-private plan and batch cursor; no tiling/workspace ABI fields.

    Legacy phases use (begin, end] in the unpermuted one-based ID space.
    Owned metadata phase 6 uses [begin, end) in zero-based task space.
    A phase change invalidates the cursor because per-batch counts differ.
    """
    return pl.struct(
        "FagTndDeterSchedule",
        metaOffset=pl.astype(0, pl.DT_INT64),
        normalized=pl.astype(0, pl.DT_INT64),
        normOffset=pl.astype(0, pl.DT_INT64),
        normFamily=pl.astype(0, pl.DT_INT64),
        strategy=pl.astype(0, pl.DT_INT64),
        protocolStatus=pl.astype(1, pl.DT_INT64),
        pairGroup=pl.astype(0, pl.DT_INT64),
        sparseType=pl.astype(0, pl.DT_INT64),
        maxRound=pl.astype(max_round, pl.DT_INT64),
        r0=pl.astype(r0, pl.DT_INT64),
        r1=pl.astype(r1, pl.DT_INT64),
        r2=pl.astype(r2, pl.DT_INT64),
        coreDivide=pl.astype(core_divide, pl.DT_INT64),
        lineDeter=pl.astype(0, pl.DT_INT64),
        isCausal=pl.astype(is_causal, pl.DT_INT64),
        rWhole=pl.astype(r_whole, pl.DT_INT64),
        # Immutable decoder parameters; private to one kernel invocation.
        gqaDenseK=pl.astype(0, pl.DT_INT64),
        gqaDenseRound=pl.astype(0, pl.DT_INT64),
        gqaDenseT1=pl.astype(0, pl.DT_INT64),
        gqaDenseT2=pl.astype(0, pl.DT_INT64),
        ownedCore=pl.astype(-1, pl.DT_INT64),
        ownedBegin=pl.astype(0, pl.DT_INT64),
        ownedEnd=pl.astype(0, pl.DT_INT64),
        cursorPhase=pl.astype(-1, pl.DT_INT64),
        cursorBatch=pl.astype(0, pl.DT_INT64),
        cursorBegin=pl.astype(0, pl.DT_INT64),
        cursorEnd=pl.astype(0, pl.DT_INT64),
        cursorReady=pl.astype(0, pl.DT_INT64),
        cursorSq=pl.astype(0, pl.DT_INT64),
        cursorSk=pl.astype(0, pl.DT_INT64),
        cursorM=pl.astype(0, pl.DT_INT64),
        cursorN=pl.astype(0, pl.DT_INT64),
        cursorP=pl.astype(0, pl.DT_INT64),
        cursorQ=pl.astype(0, pl.DT_INT64),
        cursorMo=pl.astype(0, pl.DT_INT64),
        cursorNo=pl.astype(0, pl.DT_INT64),
        cursorVm=pl.astype(0, pl.DT_INT64),
        cursorVn=pl.astype(0, pl.DT_INT64),
        cursorArea=pl.astype(0, pl.DT_INT64),
        # Fixed family24 private scalar cache; no metadata/workspace ABI change.
        gqaBandK=pl.astype(0, pl.DT_INT64),
        gqaBandM=pl.astype(0, pl.DT_INT64),
        gqaBandN=pl.astype(0, pl.DT_INT64),
        gqaBandP=pl.astype(0, pl.DT_INT64),
        gqaBandQ=pl.astype(0, pl.DT_INT64),
        gqaBandB=pl.astype(0, pl.DT_INT64),
        gqaBandG=pl.astype(0, pl.DT_INT64),
        gqaBandB1=pl.astype(0, pl.DT_INT64),
        gqaBandB2=pl.astype(0, pl.DT_INT64),
        gqaBandL1=pl.astype(0, pl.DT_INT64),
        gqaBandL2=pl.astype(0, pl.DT_INT64),
        gqaBandL3=pl.astype(0, pl.DT_INT64),
        gqaBandRm=pl.astype(0, pl.DT_INT64),
        gqaBandRm1=pl.astype(0, pl.DT_INT64),
        gqaBandR1=pl.astype(0, pl.DT_INT64),
        gqaBandR2=pl.astype(0, pl.DT_INT64),
        gqaBandDenseK=pl.astype(0, pl.DT_INT64),
        gqaBandDenseRound=pl.astype(0, pl.DT_INT64),
        gqaBandDenseT1=pl.astype(0, pl.DT_INT64),
        gqaBandDenseT2=pl.astype(0, pl.DT_INT64),
    )


def reset_tnd_batch_cursor(plan, phase):
    """Reset once per phase, not once per physical task or empty round."""
    if plan.cursorPhase != phase:
        plan.cursorPhase = pl.astype(phase, pl.DT_INT64)
        plan.cursorBatch = pl.astype(0, pl.DT_INT64)
        plan.cursorBegin = pl.astype(0, pl.DT_INT64)
        plan.cursorEnd = pl.astype(0, pl.DT_INT64)
        plan.cursorReady = pl.astype(0, pl.DT_INT64)
