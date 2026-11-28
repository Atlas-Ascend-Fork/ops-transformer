# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""分核与 RunInfo。

职责是把「本核该干哪些块」翻译成逐 task 的 :class:`FagRunInfo`：

1. **块级有效性**：``is_block_valid_i`` / ``skip_invalid``；
2. **分核映射**：普通计算保留 metadata/列分核；确定性计算按编译期 family
   选择固定布局 Host 计划、TND metadata 计划或兼容闭式，每轮返回一个物理坐标；
3. **逐 task 展开**：``set_run_info`` 把全局块下标拆成 (b, n2, g, s2o, s1o)
   与各张量的 GM 偏移，并算好 K/V 复用与 dK/dV flush 的开关。

cube 与 vector 两侧调用**同一套**函数、读**同一块** metadata —— 这是两侧
task 数一致的前提，也是 set/wait 能配对的前提。任何只改一侧的分核逻辑都会
直接死锁。
"""

import pypto_pro.language as pl

# --- generated imports (fag_fiximports.py) ---
from fag_colsplit import (
    causal_core_count,
    causal_swizzle_idx,
    col_core_count,
    col_prefix_body,
    col_local_to_global,
    col_packed_to_dense,
)
from fag_common import (
    CUBE_BASEM,
    CUBE_BASEN,
    VECTOR_BASEM,
    META_HEAD_FAG_START_INDEX,
    META_FAG_BLOCK_STARTS_OFFSET,
    META_FAG_BLOCK_ENDS_OFFSET,
    META_FAG_PRE_AIV_NUM,
    META_FAG_DQ_ROWS,
    META_FAG_DKV_ROWS,
    META_FAG_DQ_ROW_STARTS,
    META_FAG_DQ_ROW_ENDS,
    META_FAG_DKV_ROW_STARTS,
    META_FAG_DKV_ROW_ENDS,
    META_FAG_AIV_LIST_NUM,
)
from schedule.deterministic.fag_deter_legacy import (
    deter_lin_gidx,
    deter_lin_max_round,
    deter_lin_round_bounds,
)
from schedule.deterministic.fag_deter_state import (
    init_gqa_dense_plan,
    init_gqa_band_plan,
)
from schedule.deterministic.fag_deter_fixed import (
    deter_fixed_gidx,
)
from schedule.deterministic.tnd.fag_deter_tnd_plan import (
    init_tnd_metadata_schedule,
    tnd_metadata_gidx,
)
from schedule.deterministic.tnd.fag_deter_tnd_dense import (
    init_tnd_dense_schedule,
    tnd_dense_gidx,
)
from schedule.deterministic.fag_deter_state import (
    make_tnd_deter_schedule,
)
from schedule.deterministic.tnd.fag_deter_tnd_sparse import (
    init_tnd_sparse_schedule,
    tnd_sparse_gidx,
)
from fag_tiling import Tmpl
# --- end generated imports ---


# ==================================================================
#  块级有效性与「下一块」
# ==================================================================


def is_block_valid_i(
    constInfo,
    index,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    """块级 IsValid。mask_mode 是 tilingkey，只编一侧公式。

    TND（layout==1）走 _is_block_valid_tnd 变长几何解码。
    比较做成 0/1 算术：此函数会内联进 skip_invalid 的 while。0930 移植。
    """
    if layout == 1:
        return _is_block_valid_tnd(
            constInfo,
            index,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
            batch_cursor,
        )
    valid = 0
    if constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
        physicalIndex = index
        if deter == 0:
            physicalIndex = seqused_to_padded_index(
                constInfo, index, used_q_tensor, used_kv_tensor
            )
        valid = seqused_block_valid(
            constInfo, physicalIndex, used_q_tensor, used_kv_tensor
        )
    else:
        valid = padded_block_valid(constInfo, index)
    return valid


def padded_block_valid(constInfo, index):
    """Original fixed-length mask predicate; callers without seqused stay here."""
    if mask_mode == 0:
        return 1
    gDimTail = index % constInfo.s1oS2o
    s2o = gDimTail // constInfo.s1Outer
    s1o = gDimTail % constInfo.s1Outer
    s2Left = s2o * CUBE_BASEN
    if mask_mode == 3:
        s2Ignored = constInfo.s1Size - CUBE_BASEM * (s1o + 1)
        s2EndLen = pl.min(pl.max(constInfo.s2Size - s2Ignored, 0), constInfo.s2Size)
        return pl.min(pl.max(s2EndLen - s2Left, 0), 1)
    s2Right = pl.min((s2o + 1) * CUBE_BASEN, constInfo.s2Size)
    s2SparseLeft = pl.max(CUBE_BASEM * s1o - constInfo.s1Token, 0)
    s2SparseLeft = (s2SparseLeft // 64) * 64
    rowEnd = pl.min(CUBE_BASEM * (s1o + 1), constInfo.s1Size)
    s2SparseRight = ((rowEnd + constInfo.s2Token + 63) // 64) * 64
    s2SparseRight = pl.min(s2SparseRight, constInfo.s2Size)
    a = pl.min(pl.max(s2SparseRight - s2Left, 0), 1)
    b = pl.min(pl.max(s2Right - s2SparseLeft, 0), 1)
    return a * b


def skip_invalid(
    constInfo,
    idx,
    blockEnd,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    """线性支：while 无效则 idx++，越界 -1。TND 时用变长几何解码。0930 移植。"""
    # Fetch remains outside the pipeline's runtime condition. At end-of-range,
    # decode the preceding tile instead of a possibly empty trailing batch;
    # ``over`` still forces the exhausted fetch to return -1 below.
    checkIdx = idx
    if layout == 1:
        checkIdx = pl.min(idx, pl.max(blockEnd - 1, 0))
    v = is_block_valid_i(
        constInfo,
        checkIdx,
        cu_q_tensor,
        cu_kv_tensor,
        used_q_tensor,
        used_kv_tensor,
        batch_cursor,
    )
    over = pl.min(pl.max(idx + 1 - blockEnd, 0), 1)
    v = pl.min(v + over, 1)
    while v == 0:
        idx = idx + 1
        checkIdx = idx
        if layout == 1:
            checkIdx = pl.min(idx, pl.max(blockEnd - 1, 0))
        cur = is_block_valid_i(
            constInfo,
            checkIdx,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
            batch_cursor,
        )
        over = pl.min(pl.max(idx + 1 - blockEnd, 0), 1)
        v = pl.min(cur + over, 1)
    if idx >= blockEnd:
        return pl.astype(-1, pl.DT_INT64)
    return pl.astype(idx, pl.DT_INT64)


def deter_max_round(constInfo, coreNum):
    """本核循环上界。非 swizzle 闭式估轮；swizzle 取 core0 的列数（lockstep）。"""
    if swizzle == 0:
        return deter_lin_max_round(constInfo, coreNum)
    return block_range_of(constInfo, 0, coreNum, 0, 0)


def init_deter_schedule(
    constInfo,
    coreNum,
    cu_q_tensor,
    cu_kv_tensor,
    used_q_tensor,
    used_kv_tensor,
    metadata_tensor=None,
):
    """Initialize one compiled schedule family on both cube and vector cores.

    TND locality families validate the metadata plan; legacy TND computes its
    original closed form. Fixed families cache Host-selected parameters, while
    legacy GQA selects its immutable dense schedule once per invocation.
    """
    if deter_family >= 12 and deter_family <= 20:
        return init_tnd_metadata_schedule(
            constInfo,
            coreNum,
            metadata_tensor,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
        )
    if layout == 1:
        if mask_mode == 0:
            return init_tnd_dense_schedule(
                constInfo,
                coreNum,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
            )
        return init_tnd_sparse_schedule(
            constInfo,
            coreNum,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
        )
    plan = make_tnd_deter_schedule(0, 0, 0, 0, 0, 0, 0)
    if deter_family == 4:
        plan = init_gqa_dense_plan(
            plan,
            coreNum,
            constInfo.deterM,
            constInfo.deterN,
            constInfo.bSize * constInfo.n2Size,
            constInfo.gSize,
            constInfo.deterDenseRound,
        )
    elif deter_family == 24:
        plan = init_gqa_band_plan(
            plan,
            constInfo.deterBandK,
            constInfo.deterM,
            constInfo.deterN,
            constInfo.deterP,
            constInfo.deterQ,
            constInfo.bSize * constInfo.n2Size,
            constInfo.gSize,
        )
    elif deter_family == 0 and swizzle == 0:
        dense_grid = constInfo.sequsedQOptional + constInfo.sequsedKvOptional
        if mask_mode == 0:
            dense_grid = 1
        if dense_grid > 0 and constInfo.gSize > 1:
            plan = init_gqa_dense_plan(
                plan,
                coreNum,
                constInfo.s1Outer,
                constInfo.s2Outer,
                constInfo.bSize * constInfo.n2Size,
                constInfo.gSize,
                0,
            )
    return plan


def init_deter_sync_policy(plan):
    """Writer ownership changes tokens only for verified TND metadata plans."""
    sync_dq = pl.astype(1, pl.DT_INT64)
    sync_kv = pl.astype(1, pl.DT_INT64)
    dq_group = pl.astype(1, pl.DT_INT64)
    if deter_family == 14:
        if plan.strategy == 3:
            sync_dq = pl.astype(0, pl.DT_INT64)
            sync_kv = pl.astype(0, pl.DT_INT64)
    elif deter_family == 15 or deter_family == 17:
        if plan.strategy == 4:
            sync_kv = pl.astype(0, pl.DT_INT64)
    elif deter_family == 16 or deter_family == 18:
        if plan.strategy == 5:
            sync_kv = pl.astype(0, pl.DT_INT64)
            dq_group = pl.astype(4, pl.DT_INT64)
    if plan.normalized == 1:
        sync_dq = pl.astype(1, pl.DT_INT64)
        sync_kv = pl.astype(1, pl.DT_INT64)
        dq_group = pl.astype(1, pl.DT_INT64)
    return sync_dq, sync_kv, dq_group


def deter_dq_token_edges(physical_round, first_round, max_round, lag, group, sync_dq):
    """Use source rounds, including empty slots, for grouped dQ ordering.

    The compute pipeline is lag rounds behind the decoder. Flush a partial
    final group before the common drain so every initialized token is consumed.
    """
    wait = pl.astype(sync_dq, pl.DT_INT64)
    signal = pl.astype(sync_dq, pl.DT_INT64)
    if group > 1:
        logical = physical_round - lag
        wait = pl.astype(0, pl.DT_INT64)
        signal = pl.astype(0, pl.DT_INT64)
        if logical >= first_round and logical < max_round:
            if (logical - first_round) % group == 0:
                wait = pl.astype(sync_dq, pl.DT_INT64)
            if (logical - first_round + 1) % group == 0 or logical + 1 == max_round:
                signal = pl.astype(sync_dq, pl.DT_INT64)
    return wait, signal


def ci_fixed_rounds(constInfo):
    return pl.astype(constInfo.deterMaxRound, pl.DT_INT64)


def deter_round_bounds(constInfo, coreNum, tnd_plan):
    """All cores share a physical-round interval selected by the active family."""
    if (deter_family >= 1 and deter_family <= 11) or (
        deter_family >= 21 and deter_family <= 26
    ):
        return pl.astype(0, pl.DT_INT64), ci_fixed_rounds(constInfo)
    if layout == 1:
        return pl.astype(0, pl.DT_INT64), tnd_plan.maxRound
    if tnd_plan.gqaDenseRound > 0:
        return pl.astype(0, pl.DT_INT64), pl.astype(
            tnd_plan.gqaDenseRound * constInfo.s1Outer, pl.DT_INT64
        )
    if swizzle == 0:
        return deter_lin_round_bounds(constInfo, coreNum)
    return pl.astype(0, pl.DT_INT64), deter_max_round(constInfo, coreNum)


def skip_empty_deter(
    constInfo,
    cBlockIdx,
    coreNum,
    roundId,
    maxRound,
    used_q_tensor=None,
    used_kv_tensor=None,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    tnd_plan=None,
    metadata_tensor=None,
):
    """Decode exactly one physical round; empty slots do not start a tile.

    已到期的跨轮流水仍需消费，且必须透传对应 FIX token，不能整轮 continue。
    当前轮解码与有效任务前瞻分开，不能用前瞻游标推进流水物理轮次。
    """
    dead = pl.min(pl.max(roundId + 1 - maxRound, 0), 1)
    if (deter_family >= 1 and deter_family <= 11) or (
        deter_family >= 21 and deter_family <= 26
    ):
        gIdx = deter_fixed_gidx(
            constInfo,
            cBlockIdx,
            coreNum,
            roundId,
            tnd_plan.gqaDenseK,
            tnd_plan.gqaDenseRound,
            tnd_plan.gqaDenseT1,
            tnd_plan.gqaDenseT2,
            tnd_plan,
        )
    elif deter_family >= 12 and deter_family <= 20:
        gIdx = tnd_metadata_gidx(
            constInfo,
            cBlockIdx,
            coreNum,
            roundId,
            tnd_plan,
            metadata_tensor,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
        )
    elif layout == 1:
        if mask_mode == 0:
            gIdx = tnd_dense_gidx(
                constInfo,
                cBlockIdx,
                coreNum,
                roundId,
                tnd_plan,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
            )
        else:
            gIdx = tnd_sparse_gidx(
                constInfo,
                cBlockIdx,
                coreNum,
                roundId,
                tnd_plan,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
            )
        valid = tnd_deter_block_valid(
            constInfo,
            gIdx,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
        )
        gIdx = gIdx * valid + pl.astype(-1, pl.DT_INT64) * (1 - valid)
    else:
        gIdx = deter_lin_gidx(
            constInfo,
            cBlockIdx,
            coreNum,
            roundId,
            tnd_plan.gqaDenseK,
            tnd_plan.gqaDenseRound,
            tnd_plan.gqaDenseT1,
            tnd_plan.gqaDenseT2,
        )
        if constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
            valid = seqused_block_valid(constInfo, gIdx, used_q_tensor, used_kv_tensor)
            gIdx = gIdx * valid + pl.astype(-1, pl.DT_INT64) * (1 - valid)
    gOut = gIdx * (1 - dead) + pl.astype(-1, pl.DT_INT64) * dead
    return pl.astype(gOut, pl.DT_INT64), pl.astype(roundId + 1, pl.DT_INT64)


def find_next_valid_deter(
    constInfo,
    cBlockIdx,
    coreNum,
    startRound,
    maxRound,
    used_q_tensor=None,
    used_kv_tensor=None,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    tnd_plan=None,
    metadata_tensor=None,
):
    """返回下一个有效任务及其原始物理轮次；无任务时返回 (-1, maxRound)。

    调用方缓存结果，只在消费有效任务后继续前瞻，各物理轮只解码一次。
    此游标不控制 readyRound/FIX；空轮仍由主循环逐轮透传同步。
    """
    searchRound = pl.astype(startRound, pl.DT_INT64)
    # Dense MHA starts at round zero. Prove a core permanently idle only on
    # that initial lookup; active cores need no repeated geometry checks.
    if startRound == 0 and deter_family == 0 and layout != 1 and swizzle == 0:
        dense_grid = constInfo.sequsedQOptional + constInfo.sequsedKvOptional
        if mask_mode == 0:
            dense_grid = 1
        if (
            dense_grid > 0
            and constInfo.gSize == 1
            and coreNum > 0
            and constInfo.bSize > 0
            and constInfo.n2Size > 0
            and constInfo.s1Outer > 0
            and constInfo.s2Outer > 0
        ):
            active_cores = pl.min(
                coreNum, constInfo.bSize * constInfo.n2Size * constInfo.s1Outer
            )
            if cBlockIdx >= active_cores:
                # Only stop task lookahead. The caller still advances every
                # physical round and propagates the original FIX tokens.
                searchRound = pl.astype(maxRound, pl.DT_INT64)
    nextGIdx = pl.const(-1, pl.DT_INT64)
    nextRound = pl.astype(maxRound, pl.DT_INT64)
    while searchRound < maxRound:
        candidateGIdx, followingRound = skip_empty_deter(
            constInfo,
            cBlockIdx,
            coreNum,
            searchRound,
            maxRound,
            used_q_tensor,
            used_kv_tensor,
            cu_q_tensor,
            cu_kv_tensor,
            tnd_plan,
            metadata_tensor,
        )
        if candidateGIdx >= 0:
            nextGIdx = candidateGIdx
            nextRound = searchRound
            break
        searchRound = followingRound
    return nextGIdx, nextRound


def next_work_gidx(
    constInfo,
    cBlockIdx,
    coreNum,
    blockStart,
    blockEnd,
    localIdx,
    denseCursor,
    numBlocks,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    """下一块全局下标；本核没有活块时 -1。TND 时 cu 张量透传给 skip_invalid。0930 移植。"""
    if swizzle == 0 or constInfo.normalSwizzleEnabled == 0:
        return skip_invalid(
            constInfo,
            denseCursor,
            blockEnd,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
            batch_cursor,
        )
    if localIdx >= numBlocks:
        return pl.astype(-1, pl.DT_INT64)
    return block_idx_of(constInfo, cBlockIdx, coreNum, blockStart, localIdx)


def fetch_work_gidx(
    constInfo,
    meta,
    fagOff,
    cBlockIdx,
    coreNum,
    blockStart,
    blockEnd,
    localIdx,
    denseCursor,
    numBlocks,
    roundId,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    """Non-deterministic runner fetch; deterministic runners use physical rounds.

    Keep the two protocols separate: a valid-task cursor cannot drive the
    deterministic pipeline's empty-slot relay and delayed reduction drain.
    """
    gIdx = next_work_gidx(
        constInfo,
        cBlockIdx,
        coreNum,
        blockStart,
        blockEnd,
        localIdx,
        denseCursor,
        numBlocks,
        cu_q_tensor,
        cu_kv_tensor,
        used_q_tensor,
        used_kv_tensor,
        batch_cursor,
    )
    if swizzle == 0 or constInfo.normalSwizzleEnabled == 0:
        if gIdx >= 0:
            return pl.astype(gIdx, pl.DT_INT64), pl.astype(gIdx + 1, pl.DT_INT64)
        return pl.astype(gIdx, pl.DT_INT64), pl.astype(blockEnd, pl.DT_INT64)
    return pl.astype(gIdx, pl.DT_INT64), pl.astype(denseCursor, pl.DT_INT64)


# ==================================================================
#  Swizzle 分核 —— ASC 连续有效块、大 dense 旋转、大 causal 双 head 折叠
#  判据在 host（op_host/plan/flash_attn_grad_tiling_swizzle.cpp），
#  这里只负责「已经决定开 swizzle」之后的下标映射。
# ==================================================================


def swizzle_idx(constInfo, cBlockIdx, coreNum, localIdx, diagonal_rotation=True):
    """ASC large-dense columns with a diagonal S1 rotation inside each column."""
    k = localIdx // constInfo.s1Outer
    s1Idx = localIdx % constInfo.s1Outer
    t = cBlockIdx + k * coreNum
    head = t // constInfo.s2Outer
    s2Idx = t % constInfo.s2Outer
    if diagonal_rotation and deter == 0:
        s1Idx = (s2Idx + s1Idx) % constInfo.s1Outer
    return head * constInfo.s1oS2o + s2Idx * constInfo.s1Outer + s1Idx


# ==================================================================
#  AICPU metadata 布局 —— 与 flash_attn_grad_metadata_split.h 对齐
# ==================================================================

# Reader offsets live in fag_common; this layer only consumes metadata.
# The tensor view uses Host metadataWordCount, including the optional TND tail.


def block_range_of(constInfo, cBlockIdx, coreNum, blockStart, blockEnd):
    """Exact useful-task count; linear and unproved TND use metadata intervals."""
    if swizzle == 0 or constInfo.normalSwizzleEnabled == 0:
        return pl.astype(pl.max(blockEnd - blockStart, 0), pl.DT_INT64)
    if constInfo.normalSwizzleKind == 2:
        return pl.astype(causal_core_count(constInfo, cBlockIdx, coreNum), pl.DT_INT64)
    if constInfo.normalSwizzleKind == 1 or deter == 1:
        numColTasks = (
            constInfo.bSize * constInfo.n2Size * constInfo.gSize * constInfo.s2Outer
        )
        rem = pl.max(numColTasks - cBlockIdx, 0)
        return pl.astype(
            ((rem + coreNum - 1) // coreNum) * constInfo.s1Outer, pl.DT_INT64
        )
    return pl.astype(col_core_count(constInfo, cBlockIdx, coreNum), pl.DT_INT64)


def block_idx_of(constInfo, cBlockIdx, coreNum, blockStart, localIdx):
    """局部块序号 -> 全局块下标。线性支起点取自 metadata。"""
    if swizzle == 0 or constInfo.normalSwizzleEnabled == 0:
        return pl.astype(blockStart + localIdx, pl.DT_INT64)
    if constInfo.normalSwizzleKind == 2:
        return pl.astype(
            causal_swizzle_idx(constInfo, cBlockIdx, coreNum, localIdx), pl.DT_INT64
        )
    if constInfo.normalSwizzleKind == 1 or deter == 1:
        return pl.astype(
            swizzle_idx(constInfo, cBlockIdx, coreNum, localIdx), pl.DT_INT64
        )
    gidx = col_local_to_global(constInfo, cBlockIdx, coreNum, localIdx)
    if mask_mode == 0:
        return pl.astype(gidx, pl.DT_INT64)
    return pl.astype(col_packed_to_dense(constInfo, gidx), pl.DT_INT64)


# ==================================================================
#  K/V 复用与 dK/dV L0C 驻留的开关
# ==================================================================


def s2_group_of(constInfo, idx):
    """(b, n2, g, s2o) 的组合编号：同一编号内 dK/dV 可在 L0C 上持续累加。

    块网格是 b -> n2 -> g -> s2o -> s1o，一组恰为「连续 s1Outer 个块」，
    故组号就是 idx // s1Outer —— 已融合 (b,n2,g,s2o) 四维，无需再逐维还原。
    GQA 下跨 g 即换组(g 在 s2o 外侧)，dK/dV 会先 flush 再重新累加；G 个 Q
    head 对同一 KV head 的贡献靠 GM 上的 atomicAdd 合并。
    """
    return idx // constInfo.s1Outer


def need_load_kv(constInfo, gIdx, localIdx):
    """是否需要重新从 GM 载入 K/V(1=需要，0=沿用 L1 里的)。

    只有换组那一块要重载，组内其余块 K/V 相同。dkvAcc = 1 - loadKV，
    不必再算一遍。
    8192 场景 s1Outer=64，K/V 的 GM 读次数降到 1/64。
    """
    if swizzle == 0:
        isFirst = 1 - pl.min(localIdx, 1)
        gCur = s2_group_of(constInfo, gIdx)
        gPrev = s2_group_of(constInfo, pl.max(gIdx - 1, 0))
        return pl.max(isFirst, pl.min(gCur - gPrev, 1))
    # Safe defaults until apply_l0c_s2 sees the actual adjacent decoded task.
    return 1


def next_s2_same(constInfo, gIdx, localIdx):
    """下一块是否仍落在同一个 s2 块内(1=是，0=否)。"""
    if swizzle == 0:
        return 1 - pl.min(
            s2_group_of(constInfo, gIdx + 1) - s2_group_of(constInfo, gIdx), 1
        )
    return 0


# ==================================================================
#  逐 task 的 RunInfo
# ==================================================================


def _decode_padded_run_coords(constInfo, runInfo, index):
    bDimTail = index % constInfo.n2GS1oS2o
    n2DimTail = bDimTail % constInfo.gS1oS2o
    gDimTail = index % constInfo.s1oS2o
    runInfo.boIdx = index // constInfo.n2GS1oS2o
    runInfo.n2oIdx = bDimTail // constInfo.gS1oS2o
    runInfo.goIdx = n2DimTail // constInfo.s1oS2o
    runInfo.s2oIdx = gDimTail // constInfo.s1Outer
    runInfo.s1oIdx = gDimTail % constInfo.s1Outer


def _decode_tnd_run_coords(runInfo, index, prefix, w, g, m, n):
    delta = index - prefix
    mn = m * n
    gmn = g * mn
    runInfo.boIdx = w
    runInfo.n2oIdx = delta // gmn
    bTail = delta % gmn
    runInfo.goIdx = bTail // mn
    gTail = bTail % mn
    runInfo.s2oIdx = gTail // m
    runInfo.s1oIdx = gTail % m


def _set_public_vector_split(runInfo, s1Real, subIdx):
    runInfo.firstHalfS1RealSize = VECTOR_BASEM
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
        runInfo.firstHalfS1RealSize = (s1Real + 1) // 2
    runInfo.halfS1RealSize = pl.max(
        pl.min(
            s1Real - subIdx * runInfo.firstHalfS1RealSize,
            runInfo.firstHalfS1RealSize,
        ),
        0,
    )


def set_run_info(
    constInfo,
    runInfo,
    taskId,
    index,
    subIdx,
    localIdx,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    # TND（layout==1）复用本核 batch cursor，再做 per-batch divmod。
    # cu 张量由 dispatcher 循环外创建后传入；非 TND 时为 None，不读。0930 移植。
    if layout == 1:
        if deter == 1:
            # Pass literal None: forwarding an omitted optional scalar alias
            # is unsupported by the 0930 inline-argument binder.
            _set_run_info_tnd_varlen(
                constInfo,
                runInfo,
                taskId,
                index,
                subIdx,
                localIdx,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
                None,
            )
        else:
            _set_run_info_tnd_varlen(
                constInfo,
                runInfo,
                taskId,
                index,
                subIdx,
                localIdx,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
                batch_cursor,
            )
        return
    if deter == 0:
        if constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
            # Metadata packs effective grids; GM remains the original padded layout.
            index = seqused_to_padded_index(
                constInfo, index, used_q_tensor, used_kv_tensor
            )
    # index -> (boIdx, n2oIdx, goIdx, s2oIdx, s1oIdx)，S1 为最快轴。
    # 维度顺序 b -> n2 -> g -> s2o -> s1o（g 在 n2 之内、s2o 之外）。
    _decode_padded_run_coords(constInfo, runInfo, index)

    # Effective lengths only clip compute tiles; all tensor strides stay physical.
    sq = padded_sequence_length(
        used_q_tensor, runInfo.boIdx, constInfo.sequsedQOptional, constInfo.s1Size
    )
    sk = padded_sequence_length(
        used_kv_tensor, runInfo.boIdx, constInfo.sequsedKvOptional, constInfo.s2Size
    )
    runInfo.batchS1Len = sq
    runInfo.batchS2Len = sk
    s1Real = pl.min(CUBE_BASEM, sq - runInfo.s1oIdx * CUBE_BASEM)
    s2Real = pl.min(CUBE_BASEN, sk - runInfo.s2oIdx * CUBE_BASEN)
    runInfo.s1RealSize = s1Real
    runInfo.s2RealSize = s2Real
    # dK/dV 的 DualModeSplitM 按这一列的有效行对半。vec0 拿这么多行，
    # vec1 的 GM 行偏移也是它。两个子核各自算自己的 runInfo。
    runInfo.firstHalfS2RealSize = (s2Real + 1) // 2

    # GS 的 C1/C2 使用实际 M，DualModeSplitM 将 M 补偶数后均分。
    # 奇数尾块的第二个 AIV 不计算最后一行 padding；仅目标 BN2 actual-M 族
    # 随 C1/C2 使用相同切点，其它 BN2 仍保持固定 64 行分界。
    _set_public_vector_split(runInfo, s1Real, subIdx)

    # BSND 布局的 GM 偏移
    runInfo.queryOffset = runInfo.s1oIdx * CUBE_BASEM
    runInfo.keyOffset = runInfo.s2oIdx * CUBE_BASEN

    # 统一四维视图上的两个非 S 轴索引，纯算术无分支。
    # Q 侧的 head 是 n2oIdx*gSize + goIdx（共 n1 个），KV 侧是 n2oIdx（共 n2 个）
    # Q 偏移含 g，KV 偏移不含。
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    runInfo.idx0Q = runInfo.boIdx * constInfo.coefB0Q + headQ * constInfo.coefN0
    runInfo.idx2Q = headQ * constInfo.coefN2
    runInfo.idx0KV = (
        runInfo.boIdx * constInfo.coefB0KV + runInfo.n2oIdx * constInfo.coefN0
    )
    runInfo.idx2KV = runInfo.n2oIdx * constInfo.coefN2
    if nz_out == 1:
        runInfo.nzQTile = (
            runInfo.boIdx * constInfo.n1Size + headQ
        ) * constInfo.s1Outer + runInfo.s1oIdx
        runInfo.nzKvTile = (
            runInfo.boIdx * constInfo.n2Size + runInfo.n2oIdx
        ) * constInfo.s2Outer + runInfo.s2oIdx
    rowBase = runInfo.queryOffset + subIdx * runInfo.firstHalfS1RealSize
    runInfo.loadBase = pl.max(pl.min(rowBase, constInfo.s1Size - VECTOR_BASEM), 0)
    runInfo.rowShift = rowBase - runInfo.loadBase
    # TND：lse 行地址 = boIdx * S1 + loadBase（全局 token 行，lse 是 2D
    # [N1, B*S1] token 主序视图）。BSND 时 loadBaseTnd == loadBase，
    # 统一用 loadBaseTnd 消除 kernel 侧 layout 分支。0930 bisect-sink 移植。
    runInfo.loadBaseTnd = runInfo.boIdx * constInfo.s1Size + runInfo.loadBase
    if (
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
        # Match actual-M SplitM: each AIV loads exactly its owned Q rows.
        runInfo.loadBase = pl.min(rowBase, constInfo.s1ViewSize - 1)
        runInfo.rowShift = 0
        runInfo.loadBaseTnd = runInfo.boIdx * constInfo.s1Size + runInfo.loadBase

    runInfo.loadKV = need_load_kv(constInfo, index, localIdx)
    # 与 loadKV 互补：列首覆写 L0C，列中累加。消费点必须 == 1，不要用 == 0。
    runInfo.dkvAcc = 1 - runInfo.loadKV
    runInfo.flushDkv = 1 - next_s2_same(constInfo, index, localIdx)
    # swizzle 表同轮可能撞 (b,n2,s2)，L0C 跨块累加不成立。非 swizzle ASC
    # 闭式本核独占 (b,n2,s2)，由 apply_s2_idx_no_change 开累加。
    if deter == 1:
        if swizzle == 1:
            runInfo.loadKV = 1
            runInfo.dkvAcc = 0
            runInfo.flushDkv = 1
    runInfo.gIdx = index
    runInfo.dqFirst = 1
    runInfo.dqLast = 1
    set_bn2_s2_range(constInfo, runInfo)


def padded_sequence_length(used_tensor, batch, has_used, physical_length):
    """Read an optional effective prefix without changing the padded GM stride."""
    length = physical_length
    if has_used == 1:
        length = pl.astype(pl.getval(used_tensor, batch), pl.DT_INT64)
    return length


def seqused_to_padded_index(constInfo, index, used_q_tensor, used_kv_tensor):
    """Map normal metadata's compact effective grid to a fixed-layout tile index.

    Deterministic scheduling already emits physical-grid indices and must not
    call this converter. Empty batches consume no compact indices.
    """
    batch = 0
    prefix = 0
    for _t in pl.range(0, constInfo.bSize):
        sq = padded_sequence_length(
            used_q_tensor, batch, constInfo.sequsedQOptional, constInfo.s1Size
        )
        sk = padded_sequence_length(
            used_kv_tensor, batch, constInfo.sequsedKvOptional, constInfo.s2Size
        )
        m = (sq + CUBE_BASEM - 1) // CUBE_BASEM
        n = (sk + CUBE_BASEN - 1) // CUBE_BASEN
        count = constInfo.n2Size * constInfo.gSize * m * n
        advance = pl.min(pl.max(index - prefix - count + 1, 0), 1)
        advance = advance * pl.min(pl.max(constInfo.bSize - 1 - batch, 0), 1)
        batch = batch + advance
        prefix = prefix + advance * count
    sq = padded_sequence_length(
        used_q_tensor, batch, constInfo.sequsedQOptional, constInfo.s1Size
    )
    sk = padded_sequence_length(
        used_kv_tensor, batch, constInfo.sequsedKvOptional, constInfo.s2Size
    )
    m = (sq + CUBE_BASEM - 1) // CUBE_BASEM
    n = (sk + CUBE_BASEN - 1) // CUBE_BASEN
    mn = pl.max(m * n, 1)
    delta = index - prefix
    head = delta // mn
    spatial = delta % mn
    s2o = spatial // pl.max(m, 1)
    s1o = spatial % pl.max(m, 1)
    return (
        batch * constInfo.n2GS1oS2o
        + head * constInfo.s1oS2o
        + s2o * constInfo.s1Outer
        + s1o
    )


def seqused_block_valid(constInfo, index, used_q_tensor, used_kv_tensor):
    """Effective per-batch geometry on a physical padded tile grid (also drain-safe)."""
    safeIndex = pl.max(index, 0)
    batch = pl.min(safeIndex // pl.max(constInfo.n2GS1oS2o, 1), constInfo.bSize - 1)
    spatial = safeIndex % pl.max(constInfo.s1oS2o, 1)
    s2o = spatial // pl.max(constInfo.s1Outer, 1)
    s1o = spatial % pl.max(constInfo.s1Outer, 1)
    sq = padded_sequence_length(
        used_q_tensor, batch, constInfo.sequsedQOptional, constInfo.s1Size
    )
    sk = padded_sequence_length(
        used_kv_tensor, batch, constInfo.sequsedKvOptional, constInfo.s2Size
    )
    qLeft = s1o * CUBE_BASEM
    kLeft = s2o * CUBE_BASEN
    qRight = pl.min(qLeft + CUBE_BASEM, sq)
    kRight = pl.min(kLeft + CUBE_BASEN, sk)
    valid = pl.min(pl.max(index + 1, 0), 1)
    valid = valid * pl.min(pl.max(constInfo.bSize * constInfo.n2GS1oS2o - index, 0), 1)
    valid = valid * pl.min(pl.max(sq - qLeft, 0), 1) * pl.min(pl.max(sk - kLeft, 0), 1)
    if mask_mode == 3:
        valid = valid * pl.min(pl.max(qRight + sk - sq - kLeft, 0), 1)
    if mask_mode == 4:
        maxDelta = constInfo.s1Size - constInfo.s2Size
        pre = constInfo.s1Token - maxDelta + sq - sk
        following = constInfo.s2Token + maxDelta - sq + sk
        valid = valid * pl.min(pl.max(qRight + following - kLeft, 0), 1)
        valid = valid * pl.min(pl.max(kRight + pre - qLeft, 0), 1)
    return valid


def tnd_sequence_length(cu_tensor, used_tensor, batch, has_used):
    """Effective geometry comes from seqused; cu always retains physical offsets.

    The host validates optional presence and metadata validates length bounds.
    A null optional tensor is never read. Keep one return for loop-phi lowering.
    """
    begin = pl.astype(pl.getval(cu_tensor, batch), pl.DT_INT64)
    end = pl.astype(pl.getval(cu_tensor, batch + 1), pl.DT_INT64)
    length = pl.max(end - begin, 0)
    if has_used == 1:
        length = pl.astype(pl.getval(used_tensor, batch), pl.DT_INT64)
    return length


def tnd_deter_block_valid(
    constInfo,
    index,
    cu_q_tensor,
    cu_kv_tensor,
    used_q_tensor,
    used_kv_tensor,
):
    """Validate a synthetic newDeter coordinate against actual TND geometry.

    Sparse virtual rectangles may include masked tiles. Only the logical index
    uses max-S strides; input/output storage remains packed by physical cu.
    Clamp the lookup for empty/drain slots so no cu or seqused read is out of range.
    """
    safeIndex = pl.max(index, 0)
    batch = pl.min(safeIndex // pl.max(constInfo.n2GS1oS2o, 1), constInfo.bSize - 1)
    sq = tnd_sequence_length(
        cu_q_tensor, used_q_tensor, batch, constInfo.sequsedQOptional
    )
    sk = tnd_sequence_length(
        cu_kv_tensor, used_kv_tensor, batch, constInfo.sequsedKvOptional
    )
    tail = safeIndex % pl.max(constInfo.s1oS2o, 1)
    s1o = tail % pl.max(constInfo.s1Outer, 1)
    s2o = tail // pl.max(constInfo.s1Outer, 1)
    inGrid = pl.min(pl.max(index + 1, 0), 1)
    inGrid = inGrid * pl.min(pl.max(constInfo.totalBlockNum - index, 0), 1)
    inGrid = inGrid * pl.min(pl.max(sq - CUBE_BASEM * s1o, 0), 1)
    inGrid = inGrid * pl.min(pl.max(sk - CUBE_BASEN * s2o, 0), 1)
    if mask_mode == 0:
        return inGrid
    if mask_mode == 3:
        end = pl.min(pl.max(sk - sq + CUBE_BASEM * (s1o + 1), 0), sk)
        return inGrid * pl.min(pl.max(end - CUBE_BASEN * s2o, 0), 1)
    maxD = constInfo.s1Size - constInfo.s2Size
    calcS1 = constInfo.s1Token - maxD + sq - sk
    calcS2 = constInfo.s2Token + maxD - sq + sk
    colL = CUBE_BASEN * s2o
    colR = pl.min(colL + CUBE_BASEN, sk)
    leftIP = pl.max(colL - calcS2, 0)
    m = (sq + CUBE_BASEM - 1) // CUBE_BASEM
    overLeft = pl.min(pl.max(leftIP - sq, 0), 1)
    begin = (1 - overLeft) * (leftIP // CUBE_BASEM) + overLeft * m
    end = (pl.min(sq, pl.max(colR + calcS1, 0)) + CUBE_BASEM - 1) // CUBE_BASEM
    ge = pl.min(pl.max(s1o - begin + 1, 0), 1)
    lt = pl.min(pl.max(end - s1o, 0), 1)
    return inGrid * ge * lt


def init_tnd_batch_cursor():
    """Per-pipeline scalar cache; never stored in constInfo or shared workspace."""
    return pl.struct(
        "FagTndBatchCursor",
        ready=pl.astype(0, pl.DT_INT64),
        batch=pl.astype(0, pl.DT_INT64),
        begin=pl.astype(0, pl.DT_INT64),
        end=pl.astype(0, pl.DT_INT64),
        sq=pl.astype(0, pl.DT_INT64),
        sk=pl.astype(0, pl.DT_INT64),
        m=pl.astype(0, pl.DT_INT64),
        n=pl.astype(0, pl.DT_INT64),
        qBase=pl.astype(0, pl.DT_INT64),
        kvBase=pl.astype(0, pl.DT_INT64),
    )


def load_tnd_batch_cursor(ci, cursor, cuq, cuk, usedq, usedk):
    """Refresh only when entering a batch; geometry is effective, offsets physical."""
    cursor.qBase = pl.astype(pl.getval(cuq, cursor.batch), pl.DT_INT64)
    cursor.kvBase = pl.astype(pl.getval(cuk, cursor.batch), pl.DT_INT64)
    sq = pl.astype(pl.getval(cuq, cursor.batch + 1), pl.DT_INT64) - cursor.qBase
    sk = pl.astype(pl.getval(cuk, cursor.batch + 1), pl.DT_INT64) - cursor.kvBase
    if ci.sequsedQOptional == 1:
        sq = pl.astype(pl.getval(usedq, cursor.batch), pl.DT_INT64)
    if ci.sequsedKvOptional == 1:
        sk = pl.astype(pl.getval(usedk, cursor.batch), pl.DT_INT64)
    cursor.sq = pl.max(sq, 0)
    cursor.sk = pl.max(sk, 0)
    cursor.m = (cursor.sq + CUBE_BASEM - 1) // CUBE_BASEM
    cursor.n = (cursor.sk + CUBE_BASEN - 1) // CUBE_BASEN
    cursor.end = cursor.begin + ci.n2Size * ci.gSize * cursor.m * cursor.n
    cursor.ready = pl.astype(1, pl.DT_INT64)


def locate_tnd_batch(ci, index, cursor, cuq, cuk, usedq, usedk):
    """Locate the packed rectangular tile index without restarting at batch zero.

    Normal streams advance monotonically, so each crossed batch is loaded once.
    The backward correction also supports revisits, without assuming swizzle
    support or sorting/reordering tasks. Duplicate prefixes (empty batches) are
    skipped with a bounded batch index; the all-empty drain remains in range.
    """
    if cursor.ready == 0:
        load_tnd_batch_cursor(ci, cursor, cuq, cuk, usedq, usedk)
    while index < cursor.begin and cursor.batch > 0:
        previousEnd = cursor.begin
        cursor.batch = cursor.batch - 1
        load_tnd_batch_cursor(ci, cursor, cuq, cuk, usedq, usedk)
        count = cursor.end - cursor.begin
        cursor.begin = previousEnd - count
        cursor.end = previousEnd
    while index >= cursor.end and cursor.batch + 1 < ci.bSize:
        cursor.begin = cursor.end
        cursor.batch = cursor.batch + 1
        load_tnd_batch_cursor(ci, cursor, cuq, cuk, usedq, usedk)


def _is_block_valid_tnd(
    constInfo,
    index,
    cu_q_tensor,
    cu_kv_tensor,
    used_q_tensor,
    used_kv_tensor,
    batch_cursor,
):
    """TND 变长 + mask 的块有效性，与 AICPU DoFagTndSplit 的 BEGIN/END
    逐项对齐。0930 bisect-sink 定案移植。
    """
    if mask_mode == 0:
        return 1
    n2 = constInfo.n2Size
    g = constInfo.gSize
    locate_tnd_batch(
        constInfo,
        index,
        batch_cursor,
        cu_q_tensor,
        cu_kv_tensor,
        used_q_tensor,
        used_kv_tensor,
    )
    sq = batch_cursor.sq
    sk = batch_cursor.sk
    m = batch_cursor.m
    n = batch_cursor.n
    delta = index - batch_cursor.begin
    # An all-empty grid can still be visited by the unconditional drain fetch.
    # Use nonzero divisors and an explicit range predicate, without adding a
    # runtime early return around cursor updates (PyPTO loop-phi constraint).
    inGrid = pl.min(pl.max(n2 * g * m * n - delta, 0), 1)
    inGrid = inGrid * pl.min(pl.max(delta + 1, 0), 1)
    mn = pl.max(m * n, 1)
    gmn = g * mn
    bTail = delta % gmn
    gTail = bTail % mn
    s2o = gTail // pl.max(m, 1)
    s1o = gTail % pl.max(m, 1)
    if mask_mode == 3:
        s2EndLen = pl.min(pl.max(sk - sq + CUBE_BASEM * (s1o + 1), 0), sk)
        return inGrid * pl.min(pl.max(s2EndLen - CUBE_BASEN * s2o, 0), 1)
    maxD = constInfo.s1Size - constInfo.s2Size
    calcS1 = constInfo.s1Token - maxD + sq - sk
    calcS2 = constInfo.s2Token + maxD - sq + sk
    colL = CUBE_BASEN * s2o
    colOk = pl.min(pl.max(sk - colL, 0), 1)
    colR = pl.min(colL + CUBE_BASEN, sk)
    leftIP = pl.max(colL - calcS2, 0)
    overLeft = pl.min(pl.max(leftIP - sq, 0), 1)
    begin = (1 - overLeft) * (leftIP // CUBE_BASEM) + overLeft * m
    end = (pl.min(sq, pl.max(colR + calcS1, 0)) + CUBE_BASEM - 1) // CUBE_BASEM
    ge = pl.min(pl.max(s1o - begin + 1, 0), 1)
    lt = pl.min(pl.max(end - s1o, 0), 1)
    return inGrid * colOk * ge * lt


def _set_run_info_tnd_varlen(
    constInfo,
    runInfo,
    taskId,
    index,
    subIdx,
    localIdx,
    cu_q_tensor,
    cu_kv_tensor,
    used_q_tensor,
    used_kv_tensor,
    batch_cursor,
):
    """变长 TND 解码：拼接空间 gIdx → (bo,n2,g,s2o,s1o) + GM token 基址。

    非确定性复用 packed 网格的核内 batch cursor；确定性
    调度已定位 batch，以 max-shape 网格编码坐标，直接解码而不重复扫描。
    非确定性再用有效 outer 做块内 divmod，仅跨 batch 时读取几何；
    确定性用 max-shape outer 解码。GM 基址始终来自 cu，不能以有效长度
    求和替代物理偏移。
    寻址约定（与 [1, T, n, d] 视图配套）：
      idx0Q = 0，queryOffset = tokenBaseQ + s1oIdx * 128（全局 token 行）。
    0930 bisect-sink 定案移植。
    """
    n2 = constInfo.n2Size
    g = constInfo.gSize
    w = 0
    prefix = 0
    if deter == 1:
        w = index // constInfo.n2GS1oS2o
        prefix = w * constInfo.n2GS1oS2o
        tokenBaseQ = pl.astype(pl.getval(cu_q_tensor, w), pl.DT_INT64)
        batchS1Len = tnd_sequence_length(
            cu_q_tensor, used_q_tensor, w, constInfo.sequsedQOptional
        )
        tokenBaseKV = pl.astype(pl.getval(cu_kv_tensor, w), pl.DT_INT64)
        batchS2Len = tnd_sequence_length(
            cu_kv_tensor, used_kv_tensor, w, constInfo.sequsedKvOptional
        )
        # Synthetic coordinates use max-S divisors, never packed-grid prefixes.
        m = constInfo.s1Outer
        n = constInfo.s2Outer
    else:
        locate_tnd_batch(
            constInfo,
            index,
            batch_cursor,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
        )
        w = batch_cursor.batch
        prefix = batch_cursor.begin
        tokenBaseQ = batch_cursor.qBase
        tokenBaseKV = batch_cursor.kvBase
        batchS1Len = batch_cursor.sq
        batchS2Len = batch_cursor.sk
        m = batch_cursor.m
        n = batch_cursor.n
    if template == Tmpl.BN2S2 and deter == 0:
        # Invert the same packed (head, column, row) index on AIC and AIV.
        delta = index - prefix
        column = delta // m
        head = column // n
        runInfo.boIdx = w
        runInfo.n2oIdx = head // g
        runInfo.goIdx = head - runInfo.n2oIdx * g
        runInfo.s2oIdx = column - head * n
        runInfo.s1oIdx = delta - column * m
    else:
        _decode_tnd_run_coords(runInfo, index, prefix, w, g, m, n)
    s1Real = pl.max(pl.min(CUBE_BASEM, batchS1Len - runInfo.s1oIdx * CUBE_BASEM), 0)
    s2Real = pl.max(pl.min(CUBE_BASEN, batchS2Len - runInfo.s2oIdx * CUBE_BASEN), 0)
    runInfo.s1RealSize = s1Real
    runInfo.s2RealSize = s2Real
    runInfo.firstHalfS2RealSize = (s2Real + 1) // 2
    _set_public_vector_split(runInfo, s1Real, subIdx)
    runInfo.queryOffset = tokenBaseQ + runInfo.s1oIdx * CUBE_BASEM
    runInfo.keyOffset = tokenBaseKV + runInfo.s2oIdx * CUBE_BASEN
    headQ = runInfo.n2oIdx * g + runInfo.goIdx
    runInfo.idx0Q = 0
    runInfo.idx2Q = headQ
    runInfo.idx0KV = 0
    runInfo.idx2KV = runInfo.n2oIdx
    runInfo.tokenBaseQ = tokenBaseQ
    runInfo.tokenBaseKV = tokenBaseKV
    if nz_out == 1:
        physicalQ = pl.astype(pl.getval(cu_q_tensor, w + 1), pl.DT_INT64) - tokenBaseQ
        physicalKv = (
            pl.astype(pl.getval(cu_kv_tensor, w + 1), pl.DT_INT64) - tokenBaseKV
        )
        runInfo.nzQTile = (
            (tokenBaseQ // 128 + w) * constInfo.n1Size
            + headQ * ((physicalQ + 127) // 128)
            + runInfo.s1oIdx
        )
        runInfo.nzKvTile = (
            (tokenBaseKV // 128 + w) * constInfo.n2Size
            + runInfo.n2oIdx * ((physicalKv + 127) // 128)
            + runInfo.s2oIdx
        )
    runInfo.batchS1Len = batchS1Len
    runInfo.batchS2Len = batchS2Len
    rowBase = runInfo.queryOffset + subIdx * runInfo.firstHalfS1RealSize
    batchEndQ = tokenBaseQ + batchS1Len
    batchLoadBase = pl.max(pl.min(rowBase, batchEndQ - VECTOR_BASEM), tokenBaseQ)
    # Keep the vector y/dy/lse window inside the physical packed tensor.
    # A short final batch can borrow preceding rows: rowShift and
    # halfS1RealSize exclude them from computation. If the entire tensor
    # is shorter than VECTOR_BASEM, the vector loads also shrink their
    # valid row count; moving the base alone cannot make a full window fit.
    tensorCapQ = pl.max(constInfo.t1Size - VECTOR_BASEM, 0)
    runInfo.loadBase = pl.min(batchLoadBase, tensorCapQ)
    runInfo.loadBaseTnd = runInfo.loadBase
    runInfo.rowShift = rowBase - runInfo.loadBase
    if (
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
        # Packed-TND addresses stay global; never borrow another batch's rows.
        runInfo.loadBase = pl.min(rowBase, constInfo.s1ViewSize - 1)
        runInfo.loadBaseTnd = runInfo.loadBase
        runInfo.rowShift = 0
    # The packed TND grid uses each batch's own s1Outer, not maxS1's grid.
    # apply_l0c_s2 refines these safe defaults from adjacent decoded RunInfo.
    runInfo.loadKV = 1
    runInfo.dkvAcc = 0
    runInfo.flushDkv = 1
    runInfo.gIdx = index
    runInfo.dqFirst = 1
    runInfo.dqLast = 1
    set_bn2_s2_range(constInfo, runInfo)


def _set_bn2_row_edges(constInfo, runInfo, cube_m, cube_n):
    s2o = runInfo.s2oIdx
    s1o = runInfo.s1oIdx
    nextLeft = (s2o + 1) * cube_n
    nextRight = (s2o + 2) * cube_n
    if mask_mode == 0:
        if s2o == constInfo.s2Outer - 1:
            runInfo.dqLast = 1
        else:
            runInfo.dqLast = 0
        if s2o == 0:
            runInfo.dqFirst = 1
        else:
            runInfo.dqFirst = 0
        return
    if mask_mode == 3:
        s2Ignored = constInfo.s1Size - cube_m * (s1o + 1)
        s2EndLen = pl.min(pl.max(constInfo.s2Size - s2Ignored, 0), constInfo.s2Size)
        if nextLeft < s2EndLen:
            runInfo.dqLast = 0
        else:
            runInfo.dqLast = 1
        if s2o == 0:
            runInfo.dqFirst = 1
        else:
            runInfo.dqFirst = 0
        return
    s2SparseLeft = pl.max(cube_m * s1o - constInfo.s1Token, 0)
    s2SparseLeft = (s2SparseLeft // 64) * 64
    rowEnd = pl.min(cube_m * (s1o + 1), constInfo.s1Size)
    s2SparseRight = ((rowEnd + constInfo.s2Token + 63) // 64) * 64
    s2SparseRight = pl.min(s2SparseRight, constInfo.s2Size)
    if nextLeft < s2SparseRight:
        if s2SparseLeft < nextRight:
            runInfo.dqLast = 0
        else:
            runInfo.dqLast = 1
    else:
        runInfo.dqLast = 1
    if s2o > 0:
        preLeft = (s2o - 1) * cube_n
        preRight = s2o * cube_n
        if preLeft < s2SparseRight:
            if s2SparseLeft < preRight:
                runInfo.dqFirst = 0
            else:
                runInfo.dqFirst = 1
        else:
            runInfo.dqFirst = 1
    else:
        runInfo.dqFirst = 1


def set_bn2_s2_range(constInfo, runInfo):
    """本 s1 列的首/末有效 s2 块（``dqFirst``/``dqLast``）。

    只有 BN2 的 MultiBlk 需要逐 s2 判首/末；其它模板/单块档保持默认 1，直接返回。
    Host 只在 Q/KV 等长时才选 BN2，故这里可以按 uniform s1/s2 算。
    """
    if template != Tmpl.BN2:
        return
    if is_bn2_multiblk != 1:
        return
    _set_bn2_row_edges(constInfo, runInfo, CUBE_BASEM, CUBE_BASEN)


def apply_l0c_s2(constInfo, runInfos, curInfo, taskId, ring):
    """GS 按相邻有效任务的 (b,n2,s2o) 复用 K/V 并驻留 dK/dV。

    G 不属于输出块身份：dense 的 S2Outer=1 跨 G 时也继续累加。
    普通 swizzle 的组首、跨 head、causal 折叠都不能由 localIdx%M 推测。
    当前任务默认 flush；实际的下一任务解码后才允许取消上一任务的 flush。
    """
    if template == Tmpl.BN2GS1S2 and swizzle == 0:
        apply_s2_idx_no_change(runInfos, curInfo, taskId, ring)
        return
    if template == Tmpl.BN2GS1S2 and deter == 0:
        apply_s2_idx_no_change(runInfos, curInfo, taskId, ring)
        return
    if layout == 1:
        apply_s2_idx_no_change(runInfos, curInfo, taskId, ring)
        return
    if mask_mode != 0:
        apply_s2_idx_no_change(runInfos, curInfo, taskId, ring)
        return
    if constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
        apply_s2_idx_no_change(runInfos, curInfo, taskId, ring)
        return
    if deter == 1:
        if swizzle == 0:
            apply_s2_idx_no_change(runInfos, curInfo, taskId, ring)


def dkv_target_of(constInfo, index):
    """Pack the dK/dV owner ``(b, n2, s2o)``; G is intentionally excluded."""
    safeIndex = pl.max(index, 0)
    bTail = safeIndex % constInfo.n2GS1oS2o
    n2Tail = bTail % constInfo.gS1oS2o
    bIdx = safeIndex // constInfo.n2GS1oS2o
    n2Idx = bTail // constInfo.gS1oS2o
    s2Idx = (safeIndex % constInfo.s1oS2o) // constInfo.s1Outer
    return (bIdx * constInfo.n2Size + n2Idx) * constInfo.s2Outer + s2Idx


def same_dkv_target(constInfo, lhs, rhs):
    """Scalar 0/1 equality for two valid fused indices' dK/dV destinations."""
    lhsValid = pl.min(pl.max(lhs + 1, 0), 1)
    rhsValid = pl.min(pl.max(rhs + 1, 0), 1)
    diff = dkv_target_of(constInfo, lhs) - dkv_target_of(constInfo, rhs)
    different = pl.min(pl.min(pl.max(diff, 0), 1) + pl.min(pl.max(-diff, 0), 1), 1)
    return lhsValid * rhsValid * (1 - different)


def set_deter_l0c_neighbors(constInfo, runInfo, prevGIdx, curGIdx, nextGIdx):
    """Use adjacent valid tasks; an empty physical round preserves residency."""
    prevSame = same_dkv_target(constInfo, prevGIdx, curGIdx)
    nextSame = same_dkv_target(constInfo, curGIdx, nextGIdx)
    runInfo.loadKV = 1 - prevSame
    runInfo.dkvAcc = prevSame
    runInfo.flushDkv = 1 - nextSame


def apply_s2_idx_no_change(runInfos, curInfo, taskId, ring, resource_bn2s2=False):
    """与上一有效块比 (s2o, b, n2)，不含 g。"""
    prevR = runInfos[(taskId + ring - 1) % ring]
    if taskId == 0:
        curInfo.loadKV = 1
        curInfo.dkvAcc = 0
        curInfo.flushDkv = 1
        return
    if (
        prevR.s2oIdx == curInfo.s2oIdx
        and prevR.boIdx == curInfo.boIdx
        and prevR.n2oIdx == curInfo.n2oIdx
    ):
        curInfo.loadKV = 0
        curInfo.dkvAcc = 1
        prevR.flushDkv = 0
    else:
        curInfo.loadKV = 1
        curInfo.dkvAcc = 0
        prevR.flushDkv = 1
    curInfo.flushDkv = 1
    if deter == 1:
        if swizzle == 1 and (not resource_bn2s2 or template != Tmpl.BN2S2):
            curInfo.loadKV = 1
            curInfo.dkvAcc = 0
            curInfo.flushDkv = 1
            prevR.flushDkv = 1


# BN2S2-specific implementation. Names remain distinct from the public family.
# Resource layout, metadata protocol and task order are preserved.

import pypto_pro.language as pl

# --- generated imports (fag_fiximports.py) ---
from fag_colsplit import (
    col_core_count_bn2s2,
    col_local_to_global_bn2s2,
    col_packed_to_dense_bn2s2,
)
from fag_common import CUBE_BASEM_bn2s2, CUBE_BASEN_bn2s2, VECTOR_BASEM_bn2s2
from schedule.deterministic.tnd.fag_deter_tnd_plan import (
    PlanSplit_bn2s2,
    decode_plan_coordinate_bn2s2,
    plan_fixed_rectangle_bn2s2,
    plan_special_column_bn2s2,
    seq_i64_bn2s2,
    deter_lin_gidx_bn2s2,
    deter_lin_max_round_bn2s2,
)
from schedule.deterministic.tnd.fag_deter_tnd_dense import bn2s2_gidx
from schedule.deterministic.tnd.fag_deter_tnd_dense import tnd_dense_gidx
from schedule.deterministic.tnd.fag_deter_tnd_sparse import tnd_sparse_gidx
from fag_tiling import Tmpl
# --- end generated imports ---


# ==================================================================
#  块级有效性与「下一块」
# ==================================================================


def deter_max_round_bn2s2(constInfo, coreNum):
    """本核循环上界。非 swizzle 闭式估轮；swizzle 取 core0 的列数（lockstep）。"""
    if swizzle == 0 or template == Tmpl.BN2S2:
        return deter_lin_max_round_bn2s2(constInfo, coreNum)
    return block_range_of_bn2s2(constInfo, 0, coreNum, 0, 0)


def deter_swz_gidx_bn2s2(constInfo, cBlockIdx, coreNum, roundId):
    """deter+swizzle：与 ``block_idx_of`` 同映射，越界空槽 -1。"""
    if mask_mode == 0:
        n_col = constInfo.bSize * constInfo.n2Size * constInfo.gSize * constInfo.s2Outer
        t = cBlockIdx + (roundId // constInfo.s1Outer) * coreNum
        if t >= n_col:
            return pl.astype(-1, pl.DT_INT64)
        return pl.astype(
            swizzle_idx_bn2s2(constInfo, cBlockIdx, coreNum, roundId), pl.DT_INT64
        )
    packed = cBlockIdx + roundId * coreNum
    limit = (
        constInfo.bSize
        * constInfo.n2Size
        * constInfo.gSize
        * constInfo.totalPerBatchNum
    )
    if packed >= limit:
        return pl.astype(-1, pl.DT_INT64)
    return pl.astype(packed_to_dense_idx(constInfo, packed), pl.DT_INT64)


def skip_empty_deter_bn2s2(
    constInfo,
    cBlockIdx,
    coreNum,
    roundId,
    maxRound,
    used_q_tensor=None,
    used_kv_tensor=None,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    tnd_plan=None,
):
    """deter+swizzle0：只算当前物理轮，空槽不启动新 tile。

    已到期的跨轮流水仍需消费，且必须透传对应 FIX token，不能整轮 continue。
    当前轮解码与有效任务前瞻分开，不能用前瞻游标推进流水物理轮次。
    """
    dead = pl.min(pl.max(roundId + 1 - maxRound, 0), 1)
    if template == Tmpl.BN2S2:
        # BN2S2 列轮转映射；列被单核严格独占，L0C 列内累加天然单写者。
        # mask/变长无效块由 tnd_deter_block_valid 过滤成空槽。
        gIdx = bn2s2_gidx(
            constInfo,
            cBlockIdx,
            coreNum,
            roundId,
            tnd_plan,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
        )
        valid = tnd_deter_block_valid(
            constInfo,
            gIdx,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
        )
        gIdx = gIdx * valid + pl.astype(-1, pl.DT_INT64) * (1 - valid)
    else:
        # 该分支仅BN2GS1S2模板能走进来
        if layout == 1:
            if mask_mode == 0:
                gIdx = tnd_dense_gidx(
                    constInfo,
                    cBlockIdx,
                    coreNum,
                    roundId,
                    tnd_plan,
                    cu_q_tensor,
                    cu_kv_tensor,
                    used_q_tensor,
                    used_kv_tensor,
                )
            else:
                gIdx = tnd_sparse_gidx(
                    constInfo,
                    cBlockIdx,
                    coreNum,
                    roundId,
                    tnd_plan,
                    cu_q_tensor,
                    cu_kv_tensor,
                    used_q_tensor,
                    used_kv_tensor,
                )
            valid = tnd_deter_block_valid(
                constInfo,
                gIdx,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
            )
            gIdx = gIdx * valid + pl.astype(-1, pl.DT_INT64) * (1 - valid)
        else:
            gIdx = deter_lin_gidx_bn2s2(constInfo, cBlockIdx, coreNum, roundId)
            if constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
                valid = seqused_block_valid(
                    constInfo, gIdx, used_q_tensor, used_kv_tensor
                )
                gIdx = gIdx * valid + pl.astype(-1, pl.DT_INT64) * (1 - valid)
    gOut = gIdx * (1 - dead) + pl.astype(-1, pl.DT_INT64) * dead
    return pl.astype(gOut, pl.DT_INT64), pl.astype(roundId + 1, pl.DT_INT64)


def next_work_gidx_bn2s2(
    constInfo,
    cBlockIdx,
    coreNum,
    blockStart,
    blockEnd,
    localIdx,
    denseCursor,
    numBlocks,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    """下一块全局下标；本核没有活块时 -1。TND 时 cu 张量透传给 skip_invalid。0930 移植。"""
    if swizzle == 0 or template == Tmpl.BN2S2:
        return skip_invalid(
            constInfo,
            denseCursor,
            blockEnd,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
            batch_cursor,
        )
    if localIdx >= numBlocks:
        return pl.astype(-1, pl.DT_INT64)
    return block_idx_of_bn2s2(constInfo, cBlockIdx, coreNum, blockStart, localIdx)


def fetch_work_gidx_bn2s2(
    constInfo,
    meta,
    fagOff,
    cBlockIdx,
    coreNum,
    blockStart,
    blockEnd,
    localIdx,
    denseCursor,
    numBlocks,
    roundId,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    """主循环唯一取块入口，返回 (gIdx, nextCursor)。

    deter+swizzle0：核上闭式跳空槽，nextCursor 是下一轮下标。
    deter+swizzle1：按轮 lockstep，空槽保持 -1，cursor 原样。
    非 deter：next_work_gidx；线性支 nextCursor = gIdx+1。
    """
    if deter == 1:
        if swizzle == 1 and template != Tmpl.BN2S2:
            return (
                deter_swz_gidx_bn2s2(constInfo, cBlockIdx, coreNum, roundId),
                pl.astype(denseCursor, pl.DT_INT64),
            )
        return skip_empty_deter_bn2s2(
            constInfo,
            cBlockIdx,
            coreNum,
            denseCursor,
            deter_max_round_bn2s2(constInfo, coreNum),
        )
    if (
        template == Tmpl.BN2S2
        and deter == 0
        and layout == 1
        and swizzle == 1
        and (
            (mask_mode == 0 and d_align == 256)
            or (mask_mode == 3 and d_align != 256)
            or (mask_mode == 4 and d_align != 192 and d_align != 256)
        )
    ):
        if constInfo.normalSwizzleEnabled == 1:
            position = pl.astype(localIdx, pl.DT_INT64)
            if mask_mode != 0 and constInfo.fixedGsHasPadding == 1:
                position = pl.astype(denseCursor, pl.DT_INT64)
                if localIdx == 0:
                    position = pl.const(0, pl.DT_INT64)
                while position < numBlocks:
                    index = block_idx_of(
                        constInfo, cBlockIdx, coreNum, blockStart, position
                    )
                    valid = is_block_valid_i(
                        constInfo,
                        index,
                        cu_q_tensor,
                        cu_kv_tensor,
                        used_q_tensor,
                        used_kv_tensor,
                        batch_cursor,
                    )
                    if valid == 1:
                        break
                    position = position + 1
            if position < numBlocks:
                index = block_idx_of(
                    constInfo, cBlockIdx, coreNum, blockStart, position
                )
                return pl.astype(index, pl.DT_INT64), pl.astype(
                    position + 1, pl.DT_INT64
                )
            return pl.const(-1, pl.DT_INT64), pl.astype(position, pl.DT_INT64)
    gIdx = next_work_gidx_bn2s2(
        constInfo,
        cBlockIdx,
        coreNum,
        blockStart,
        blockEnd,
        localIdx,
        denseCursor,
        numBlocks,
        cu_q_tensor,
        cu_kv_tensor,
        used_q_tensor,
        used_kv_tensor,
        batch_cursor,
    )
    if swizzle == 0 or template == Tmpl.BN2S2:
        if gIdx >= 0:
            return pl.astype(gIdx, pl.DT_INT64), pl.astype(gIdx + 1, pl.DT_INT64)
        return pl.astype(gIdx, pl.DT_INT64), pl.astype(blockEnd, pl.DT_INT64)
    return pl.astype(gIdx, pl.DT_INT64), pl.astype(denseCursor, pl.DT_INT64)


# ==================================================================
#  Swizzle 分核 —— 让同时在跑的核落在同一 head 内，命中 L2
#  判据在 host（op_host/plan/flash_attn_grad_tiling_swizzle.cpp），
#  这里只负责「已经决定开 swizzle」之后的下标映射。
# ==================================================================


def swizzle_idx_bn2s2(constInfo, cBlockIdx, coreNum, localIdx):
    """BN2S2 policy retains the original unrotated column order."""
    return swizzle_idx(constInfo, cBlockIdx, coreNum, localIdx, False)


# ==================================================================
#  AICPU metadata 布局 —— 与 flash_attn_grad_metadata_split.h 对齐
# ==================================================================

META_HEAD_FAG_START_INDEX_bn2s2 = 7
META_FAG_BLOCK_STARTS_OFFSET_bn2s2 = 8
META_FAG_BLOCK_ENDS_OFFSET_bn2s2 = 44
META_FAG_PRE_AIV_NUM_bn2s2 = 121
META_FAG_DQ_ROWS_bn2s2 = 122
META_FAG_DKV_ROWS_bn2s2 = 123
META_FAG_DQ_ROW_STARTS_bn2s2 = 124
META_FAG_DQ_ROW_ENDS_bn2s2 = 196
META_FAG_DKV_ROW_STARTS_bn2s2 = 268
META_FAG_DKV_ROW_ENDS_bn2s2 = 340
META_FAG_AIV_LIST_NUM_bn2s2 = 72
# BN2S2 运行期回退 GS1S2 标记（扩区专用字段，offset 412）：metadata AICPU
# （split.h DoFagSparse）判 TND 值不饱和/切核失败后置 1——老仓 DoSparse 饱和
# 回退（normal_regbase.cpp:1634）的运行期迁移；host 读不到 cu_seqlens 判不
# 了饱和、tilingkey 已固化 BN2S2，只能靠本标记切 GS 流程。0=保持 BN2S2。
META_FAG_FALLBACK_TO_GS_bn2s2 = 412
META_FAG_FIXED_GS_MODE_bn2s2 = 413
META_FAG_FIXED_GS_Q_TILES_bn2s2 = 414
META_FAG_FIXED_GS_KV_TILES_bn2s2 = 415


def fixed_gs_token_blocks(token):
    adjusted = pl.astype(token, pl.DT_INT64) + 127
    return pl.max(adjusted, 0) // 128 - pl.max(-adjusted, 0) // 128


def set_fixed_gs_schedule_bn2s2(ci, meta, base):
    if (
        template == Tmpl.BN2S2
        and deter == 0
        and layout == 1
        and swizzle == 1
        and (
            (mask_mode == 0 and d_align == 256)
            or (mask_mode == 3 and d_align != 256)
            or (mask_mode == 4 and d_align != 192 and d_align != 256)
        )
    ):
        mode = pl.getval(meta, base + META_FAG_FIXED_GS_MODE_bn2s2)
        if mode > 0:
            q_tiles = pl.astype(
                pl.getval(meta, base + META_FAG_FIXED_GS_Q_TILES_bn2s2), pl.DT_INT64
            )
            kv_tiles = pl.astype(
                pl.getval(meta, base + META_FAG_FIXED_GS_KV_TILES_bn2s2), pl.DT_INT64
            )
            span = (
                (q_tiles * 10 + kv_tiles * 12) * 128 * ci.dSize * ci.bSize * ci.n1Size
            )
            if span > ci.normalL2Bytes:
                limit = pl.const(6, pl.DT_INT64)
                if ci.s1Size >= 16384:
                    limit = (ci.s1Size // 16384) * 8
                factor = pl.astype(pl.getval(meta, base + 2), pl.DT_INT64)
                ci.normalSwizzleEnabled = 1
                ci.fixedGsHasPadding = mode // 4
                ci.normalSwizzleKind = (mode % 4) - 1
                if mask_mode != 0:
                    m = pl.astype(ci.s1Outer, pl.DT_INT64)
                    n = pl.astype(ci.s2Outer, pl.DT_INT64)
                    p = pl.min(m - 1, fixed_gs_token_blocks(ci.s1Token))
                    q = pl.min(n - 1, fixed_gs_token_blocks(ci.s2Token))
                    if mask_mode == 3:
                        p = m - 1
                        if ci.s1Size <= ci.s2Size:
                            q = n - m + 1
                            if ci.s1Size == ci.s2Size:
                                q = 0
                    if mask_mode == 4:
                        if ci.s1Token >= ci.s1Size and ci.s2Token >= ci.s2Size:
                            p = m - 1
                            q = n - 1
                        elif ci.s1Token >= ci.s1Size and ci.s2Token == 0:
                            p = m - 1
                            q = 0
                    ci.colP = p
                    ci.colQ = q
                    ci.packedPerHead = col_prefix_body(m, p, q, n)
                ci.normalContinuousBlocks = pl.max(pl.min(factor, limit), 1)


FAG_FALLBACK_TO_GS_bn2s2 = 1
# fagOff 随 FA 段（aic * stride * B * N2）变大，与 shape 表无关。
META_VIEW_LEN_bn2s2 = 1048576


def bn2s2_next_bn2s2(
    constInfo,
    cursor,
    blockEnd,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
):
    """列步进取下一块：TND+mask 时跳过全无效块（mask 几何收缩的 kernel 侧
    移植，对齐老仓 GetNextValidIdxForTndSwizzle*——只消费与 causal 三角/
    band 梯形相交的块，半有效边界块保留做块内 mask）。跳过的全无效列不再
    走完整 CV 流水，其 16-bit 输出由 pre 阶段的 kv_out_zero 兜底（见
    fag_kernel.py TND pre）；L0C 列驻留按相邻**被消费**块比较
    （apply_s2_idx_no_change），跳块天然兼容。

    dense（layout==0）与 mask0 恒有效（路由/打包网格已保证），走线性快路
    ——判断折成立即数，零运行期成本。cube/vector 两侧共用本函数、各自独立
    的 batch cursor 做同样的纯函数判定，跳块序列逐块一致——这是两侧
    set/wait 仍能逐任务配对的前提，任何只改一侧的跳块都会直接死锁。
    """
    if layout == 1 and mask_mode != 0:
        idx = cursor
        while idx < blockEnd:
            valid = _is_block_valid_tnd(
                constInfo,
                idx,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
                batch_cursor,
            )
            if valid == 1:
                return pl.astype(idx, pl.DT_INT64), pl.astype(idx + 1, pl.DT_INT64)
            idx = idx + 1
        return pl.astype(-1, pl.DT_INT64), pl.astype(blockEnd, pl.DT_INT64)
    if cursor >= blockEnd:
        return pl.astype(-1, pl.DT_INT64), pl.astype(blockEnd, pl.DT_INT64)
    return pl.astype(cursor, pl.DT_INT64), pl.astype(cursor + 1, pl.DT_INT64)


def block_range_of_bn2s2(constInfo, cBlockIdx, coreNum, blockStart, blockEnd):
    """本核负责的块数。线性支读 metadata；swizzle 仍走列跨核公式。

    mask+swizzle 按 packed 有效块跨核发，不再用整矩形列跨核。
    分核开关用 tilingkey 裸名 ``swizzle``，与 ``next_work_gidx`` 一样先判
    ``swizzle == 0``，死支公式不进线性核。
    """
    if swizzle == 0 or template == Tmpl.BN2S2:
        return pl.astype(pl.max(blockEnd - blockStart, 0), pl.DT_INT64)
    if mask_mode == 0:
        numColTasks = (
            constInfo.bSize * constInfo.n2Size * constInfo.gSize * constInfo.s2Outer
        )
        rem = pl.max(numColTasks - cBlockIdx, 0)
        return pl.astype(
            ((rem + coreNum - 1) // coreNum) * constInfo.s1Outer, pl.DT_INT64
        )
    return pl.astype(col_core_count_bn2s2(constInfo, cBlockIdx, coreNum), pl.DT_INT64)


def block_idx_of_bn2s2(constInfo, cBlockIdx, coreNum, blockStart, localIdx):
    """局部块序号 -> 全局块下标。线性支起点取自 metadata。"""
    if swizzle == 0 or template == Tmpl.BN2S2:
        return pl.astype(blockStart + localIdx, pl.DT_INT64)
    if mask_mode == 0:
        return pl.astype(
            swizzle_idx_bn2s2(constInfo, cBlockIdx, coreNum, localIdx), pl.DT_INT64
        )
    gidx = col_local_to_global_bn2s2(constInfo, cBlockIdx, coreNum, localIdx)
    return pl.astype(col_packed_to_dense_bn2s2(constInfo, gidx), pl.DT_INT64)


# ==================================================================
#  K/V 复用与 dK/dV L0C 驻留的开关
# ==================================================================


def need_load_kv_bn2s2(constInfo, gIdx, localIdx):
    """是否需要重新从 GM 载入 K/V(1=需要，0=沿用 L1 里的)。

    只有换组那一块要重载，组内其余块 K/V 相同。dkvAcc = 1 - loadKV，
    不必再算一遍。
    8192 场景 s1Outer=64，K/V 的 GM 读次数降到 1/64。
    """
    if swizzle == 0 or template == Tmpl.BN2S2:
        isFirst = 1 - pl.min(localIdx, 1)
        gCur = s2_group_of(constInfo, gIdx)
        gPrev = s2_group_of(constInfo, pl.max(gIdx - 1, 0))
        return pl.max(isFirst, pl.min(gCur - gPrev, 1))
    return 1 - pl.min(localIdx % constInfo.s1Outer, 1)


def next_s2_same_bn2s2(constInfo, gIdx, localIdx):
    """下一块是否仍落在同一个 s2 块内(1=是，0=否)。"""
    if swizzle == 0 or template == Tmpl.BN2S2:
        return 1 - pl.min(
            s2_group_of(constInfo, gIdx + 1) - s2_group_of(constInfo, gIdx),
            1,
        )
    return pl.min((localIdx + 1) % constInfo.s1Outer, 1)


# ==================================================================
#  逐 task 的 RunInfo
# ==================================================================


def set_run_info_bn2s2(
    constInfo,
    runInfo,
    taskId,
    index,
    subIdx,
    localIdx,
    cu_q_tensor=None,
    cu_kv_tensor=None,
    used_q_tensor=None,
    used_kv_tensor=None,
    batch_cursor=None,
    actual_m=False,
):
    # TND（layout==1）复用本核 batch cursor，再做 per-batch divmod。
    # cu 张量由 dispatcher 循环外创建后传入；非 TND 时为 None，不读。0930 移植。
    if layout == 1:
        if deter == 1:
            # Pass literal None: forwarding an omitted optional scalar alias
            # is unsupported by the 0930 inline-argument binder.
            _set_run_info_tnd_varlen_bn2s2(
                constInfo,
                runInfo,
                taskId,
                index,
                subIdx,
                localIdx,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
                None,
                actual_m,
            )
        else:
            _set_run_info_tnd_varlen_bn2s2(
                constInfo,
                runInfo,
                taskId,
                index,
                subIdx,
                localIdx,
                cu_q_tensor,
                cu_kv_tensor,
                used_q_tensor,
                used_kv_tensor,
                batch_cursor,
                actual_m,
            )
        return
    if deter == 0:
        if constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
            # Metadata packs effective grids; GM remains the original padded layout.
            index = seqused_to_padded_index(
                constInfo, index, used_q_tensor, used_kv_tensor
            )
    # index -> (boIdx, n2oIdx, goIdx, s2oIdx, s1oIdx)，S1 为最快轴。
    # 维度顺序 b -> n2 -> g -> s2o -> s1o（g 在 n2 之内、s2o 之外）。
    _decode_padded_run_coords(constInfo, runInfo, index)

    # Effective lengths only clip compute tiles; all tensor strides stay physical.
    sq = padded_sequence_length(
        used_q_tensor, runInfo.boIdx, constInfo.sequsedQOptional, constInfo.s1Size
    )
    sk = padded_sequence_length(
        used_kv_tensor, runInfo.boIdx, constInfo.sequsedKvOptional, constInfo.s2Size
    )
    runInfo.batchS1Len = sq
    runInfo.batchS2Len = sk
    s1Real = pl.min(CUBE_BASEM_bn2s2, sq - runInfo.s1oIdx * CUBE_BASEM_bn2s2)
    s2Real = pl.min(CUBE_BASEN_bn2s2, sk - runInfo.s2oIdx * CUBE_BASEN_bn2s2)
    runInfo.s1RealSize = s1Real
    runInfo.s2RealSize = s2Real
    # dK/dV 的 DualModeSplitM 按这一列的有效行对半。vec0 拿这么多行，
    # vec1 的 GM 行偏移也是它。两个子核各自算自己的 runInfo。
    runInfo.firstHalfS2RealSize = (s2Real + 1) // 2

    # BSND 布局的 GM 偏移
    runInfo.queryOffset = runInfo.s1oIdx * CUBE_BASEM_bn2s2
    runInfo.keyOffset = runInfo.s2oIdx * CUBE_BASEN_bn2s2

    # 统一四维视图上的两个非 S 轴索引，纯算术无分支。
    # Q 侧的 head 是 n2oIdx*gSize + goIdx（共 n1 个），KV 侧是 n2oIdx（共 n2 个）
    # Q 偏移含 g，KV 偏移不含。
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    runInfo.idx0Q = runInfo.boIdx * constInfo.coefB0Q + headQ * constInfo.coefN0
    runInfo.idx2Q = headQ * constInfo.coefN2
    runInfo.idx0KV = (
        runInfo.boIdx * constInfo.coefB0KV + runInfo.n2oIdx * constInfo.coefN0
    )
    runInfo.idx2KV = runInfo.n2oIdx * constInfo.coefN2
    set_vector_rows_bn2s2(constInfo, runInfo, subIdx, actual_m)

    runInfo.loadKV = need_load_kv_bn2s2(constInfo, index, localIdx)
    # 与 loadKV 互补：列首覆写 L0C，列中累加。消费点必须 == 1，不要用 == 0。
    runInfo.dkvAcc = 1 - runInfo.loadKV
    runInfo.flushDkv = 1 - next_s2_same_bn2s2(constInfo, index, localIdx)
    # swizzle 表同轮可能撞 (b,n2,s2)，L0C 跨块累加不成立。非 swizzle ASC
    # 闭式本核独占 (b,n2,s2)，由 apply_s2_idx_no_change 开累加。
    if deter == 1:
        if swizzle == 1 and template != Tmpl.BN2S2:
            runInfo.loadKV = 1
            runInfo.dkvAcc = 0
            runInfo.flushDkv = 1
    runInfo.gIdx = index
    runInfo.dqFirst = 1
    runInfo.dqLast = 1
    if is_bn2_multiblk == 1:
        check_s1_range_in_bn2_bn2s2(constInfo, runInfo)


def _set_run_info_tnd_varlen_bn2s2(
    constInfo,
    runInfo,
    taskId,
    index,
    subIdx,
    localIdx,
    cu_q_tensor,
    cu_kv_tensor,
    used_q_tensor,
    used_kv_tensor,
    batch_cursor,
    actual_m=False,
):
    """变长 TND 解码：拼接空间 gIdx → (bo,n2,g,s2o,s1o) + GM token 基址。

    非确定性复用 packed 网格的核内 batch cursor；确定性
    调度已定位 batch，以 max-shape 网格编码坐标，直接解码而不重复扫描。
    非确定性再用有效 outer 做块内 divmod，仅跨 batch 时读取几何；
    确定性用 max-shape outer 解码。GM 基址始终来自 cu，不能以有效长度
    求和替代物理偏移。
    寻址约定（与 [1, T, n, d] 视图配套）：
      idx0Q = 0，queryOffset = tokenBaseQ + s1oIdx * 128（全局 token 行）。
    0930 bisect-sink 定案移植。
    """
    n2 = constInfo.n2Size
    g = constInfo.gSize
    w = 0
    prefix = 0
    if deter == 1:
        w = index // constInfo.n2GS1oS2o
        prefix = w * constInfo.n2GS1oS2o
        tokenBaseQ = pl.astype(pl.getval(cu_q_tensor, w), pl.DT_INT64)
        batchS1Len = tnd_sequence_length(
            cu_q_tensor, used_q_tensor, w, constInfo.sequsedQOptional
        )
        tokenBaseKV = pl.astype(pl.getval(cu_kv_tensor, w), pl.DT_INT64)
        batchS2Len = tnd_sequence_length(
            cu_kv_tensor, used_kv_tensor, w, constInfo.sequsedKvOptional
        )
        # Synthetic coordinates use max-S divisors, never packed-grid prefixes.
        m = constInfo.s1Outer
        n = constInfo.s2Outer
    else:
        locate_tnd_batch(
            constInfo,
            index,
            batch_cursor,
            cu_q_tensor,
            cu_kv_tensor,
            used_q_tensor,
            used_kv_tensor,
        )
        w = batch_cursor.batch
        prefix = batch_cursor.begin
        tokenBaseQ = batch_cursor.qBase
        tokenBaseKV = batch_cursor.kvBase
        batchS1Len = batch_cursor.sq
        batchS2Len = batch_cursor.sk
        m = batch_cursor.m
        n = batch_cursor.n
    _decode_tnd_run_coords(runInfo, index, prefix, w, g, m, n)
    s1Real = pl.max(
        pl.min(CUBE_BASEM_bn2s2, batchS1Len - runInfo.s1oIdx * CUBE_BASEM_bn2s2), 0
    )
    s2Real = pl.max(
        pl.min(CUBE_BASEN_bn2s2, batchS2Len - runInfo.s2oIdx * CUBE_BASEN_bn2s2), 0
    )
    runInfo.s1RealSize = s1Real
    runInfo.s2RealSize = s2Real
    runInfo.firstHalfS2RealSize = (s2Real + 1) // 2
    runInfo.queryOffset = tokenBaseQ + runInfo.s1oIdx * CUBE_BASEM_bn2s2
    runInfo.keyOffset = tokenBaseKV + runInfo.s2oIdx * CUBE_BASEN_bn2s2
    headQ = runInfo.n2oIdx * g + runInfo.goIdx
    runInfo.idx0Q = 0
    runInfo.idx2Q = headQ
    runInfo.idx0KV = 0
    runInfo.idx2KV = runInfo.n2oIdx
    runInfo.tokenBaseQ = tokenBaseQ
    runInfo.tokenBaseKV = tokenBaseKV
    runInfo.batchS1Len = batchS1Len
    runInfo.batchS2Len = batchS2Len
    set_vector_rows_bn2s2(constInfo, runInfo, subIdx, actual_m)
    # The packed TND grid uses each batch's own s1Outer, not maxS1's grid.
    # apply_l0c_s2 refines these safe defaults from adjacent decoded RunInfo.
    runInfo.loadKV = 1
    runInfo.dkvAcc = 0
    runInfo.flushDkv = 1
    runInfo.gIdx = index
    runInfo.dqFirst = 1
    runInfo.dqLast = 1
    if is_bn2_multiblk == 1:
        # Host permits BN2 only when both Q and KV are uniformly sized.
        check_s1_range_in_bn2_bn2s2(constInfo, runInfo)


def check_s1_range_in_bn2_bn2s2(constInfo, runInfo):
    """本 s1 列的首/末有效 s2 块。"""
    _set_bn2_row_edges(constInfo, runInfo, CUBE_BASEM_bn2s2, CUBE_BASEN_bn2s2)


def apply_l0c_s2_bn2s2(constInfo, runInfos, curInfo, taskId, ring):
    """TND、mask 或 deter+非 swizzle：按上一有效块的 (s2,b,n2) 开 L0C 累加。"""
    if layout == 1:
        apply_s2_idx_no_change_bn2s2(runInfos, curInfo, taskId, ring)
        return
    if mask_mode != 0:
        apply_s2_idx_no_change_bn2s2(runInfos, curInfo, taskId, ring)
        return
    if constInfo.sequsedQOptional + constInfo.sequsedKvOptional > 0:
        apply_s2_idx_no_change_bn2s2(runInfos, curInfo, taskId, ring)
        return
    if deter == 1:
        if swizzle == 0 or template == Tmpl.BN2S2:
            apply_s2_idx_no_change_bn2s2(runInfos, curInfo, taskId, ring)


def apply_s2_idx_no_change_bn2s2(runInfos, curInfo, taskId, ring):
    apply_s2_idx_no_change(runInfos, curInfo, taskId, ring, True)


# ==================================================================
# BN2S2 TND metadata 分核：任务展开与双槽步进
# ==================================================================


def plan_task_valid_bn2s2(h, cuq, cuk, batch, row, col, mask_mode):
    """IsValidDeterForTnd, restricted to the admitted MHA mask 0/3/4 domain."""
    valid = False
    if batch >= 0 and batch < h.batch and row >= 0 and col >= 0:
        sq = seq_i64_bn2s2(cuq, batch + 1) - seq_i64_bn2s2(cuq, batch)
        sk = seq_i64_bn2s2(cuk, batch + 1) - seq_i64_bn2s2(cuk, batch)
        left = col * 128
        right = pl.min(left + 128, sk)
        valid = row * 128 < sq and left < sk
        if mask_mode != 0:
            pre = h.tokenQ + sq - sk
            post = h.tokenKv - sq + sk
            sparse_left = pl.max(row * 128 - pre, 0) // 64 * 64
            # Match AlignTo64 and the source's max-S1 bound, including signed
            # right-window arithmetic. Do not replace by a tile-level triangle.
            sparse_right = (pl.min((row + 1) * 128, h.maxQ) + post + 63) // 64 * 64
            sparse_right = pl.min(sparse_right, sk)
            valid = valid and left < sparse_right and right > sparse_left
    return valid


def plan_find_next_task_bn2s2(words, h, cuq, cuk, core, start_round, mask_mode):
    """CalDeterIndex: return the first valid physical round, or the drain slot."""
    next_round = h.maxRound
    batch = pl.astype(-1, pl.DT_INT64)
    head = pl.astype(-1, pl.DT_INT64)
    row = pl.astype(-1, pl.DT_INT64)
    col = pl.astype(-1, pl.DT_INT64)
    candidate = start_round
    if mask_mode == 0 and h.fixedSq > 0:
        # Every in-range tile of the admitted dense rectangle is live. Columns
        # grow by wave, so an idle core here stays idle for the remaining waves.
        if candidate < h.maxRound and core < h.heads * h.fixedRows:
            b, n, x, y = plan_fixed_rectangle_bn2s2(h, core, candidate)
            if b >= 0:
                next_round = candidate
                batch = b
                head = n
                row = x
                col = y
        return next_round, batch, head, row, col
    while candidate < h.maxRound:
        b, n, x, y = decode_plan_coordinate_bn2s2(
            words, h, cuq, cuk, core, candidate, mask_mode
        )
        if plan_task_valid_bn2s2(h, cuq, cuk, b, x, y, mask_mode):
            next_round = candidate
            batch = b
            head = n
            row = x
            col = y
            break
        candidate = candidate + 1
    return next_round, batch, head, row, col


def plan_column_key_bn2s2(h, cuk, batch, head, col, mask_mode):
    """Original packed TND keyOffset, in input elements, for SpecialS2Index."""
    key = pl.astype(-1, pl.DT_INT64)
    if batch >= 0:
        if mask_mode == 0 and h.fixedSq > 0:
            base = h.fixedKvBase
        else:
            base = seq_i64_bn2s2(cuk, batch)
        key = ((base + col * 128) * h.heads + head) * h.dimQ
    return key


def plan_bind_run_bn2s2(
    words,
    h,
    cuq,
    cuk,
    info,
    task_id,
    physical_round,
    sub_idx,
    batch,
    head,
    row,
    col,
    previous_key,
    next_key,
    previous,
    mask_mode,
):
    """Bind the common matmul/vector views to one already validated coordinate."""
    if mask_mode == 0 and h.fixedSq > 0:
        qb = h.fixedQBase
        kb = h.fixedKvBase
        sq = h.fixedSq
        sk = h.fixedSk
    elif (
        h.logicalSplit == PlanSplit_bn2s2.GS and task_id > 0 and previous.boIdx == batch
    ):
        qb = previous.tokenBaseQ
        kb = previous.tokenBaseKV
        sq = previous.batchS1Len
        sk = previous.batchS2Len
    else:
        qb = seq_i64_bn2s2(cuq, batch)
        kb = seq_i64_bn2s2(cuk, batch)
        sq = seq_i64_bn2s2(cuq, batch + 1) - qb
        sk = seq_i64_bn2s2(cuk, batch + 1) - kb
    info.boIdx = batch
    info.n2oIdx = head
    info.goIdx = pl.astype(0, pl.DT_INT64)
    info.s1oIdx = row
    info.s2oIdx = col
    info.s1RealSize = pl.min(128, sq - row * 128)
    info.s2RealSize = pl.min(128, sk - col * 128)
    # ASC SetRunInfoDeterForTND: both tail dimensions split at ceil(real/2).
    # Matmul/FIX and the NZ insert must use this same split, including odd tails.
    info.firstHalfS1RealSize = (info.s1RealSize + 1) // 2
    info.firstHalfS2RealSize = (info.s2RealSize + 1) // 2
    info.halfS1RealSize = pl.min(
        info.firstHalfS1RealSize, info.s1RealSize - sub_idx * info.firstHalfS1RealSize
    )
    info.halfS2RealSize = pl.min(
        info.firstHalfS2RealSize, info.s2RealSize - sub_idx * info.firstHalfS2RealSize
    )
    info.tokenBaseQ = qb
    info.tokenBaseKV = kb
    info.batchS1Len = sq
    info.batchS2Len = sk
    info.queryOffset = qb + row * 128
    info.keyOffset = kb + col * 128
    info.idx0Q = pl.astype(0, pl.DT_INT64)
    info.idx0KV = pl.astype(0, pl.DT_INT64)
    info.idx2Q = head
    info.idx2KV = head
    row_base = info.queryOffset + sub_idx * info.firstHalfS1RealSize
    batch_base = pl.max(pl.min(row_base, qb + sq - 64), qb)
    info.loadBase = pl.min(batch_base, pl.max(h.totalQ - 64, 0))
    info.loadBaseTnd = info.loadBase
    info.rowShift = row_base - info.loadBase
    info.columnKey = (info.keyOffset * h.heads + head) * h.dimQ
    info.dkOutOffset = info.columnKey
    info.dvOutOffset = (info.keyOffset * h.heads + head) * h.dimV
    info.loadKV = pl.astype(info.columnKey != previous_key, pl.DT_INT64)
    info.dkvAcc = 1 - info.loadKV
    info.flushDkv = pl.astype(info.columnKey != next_key, pl.DT_INT64)
    info.taskId = task_id
    info.readyRound = physical_round + 1
    info.specialIndex = pl.astype(-1, pl.DT_INT64)
    if h.logicalSplit == PlanSplit_bn2s2.BN2S2:
        info.specialIndex = plan_special_column_bn2s2(words, info.columnKey)
    info.sinkOffset = (
        (head * h.batch + batch) * h.sinkS1Outer * h.sinkS2Outer
        + (row * 2 + sub_idx) * h.sinkS2Outer
        + col
    )


def plan_pipeline_step_bn2s2(
    words, h, cuq, cuk, core, sub, state, slots, physical_round, mask_mode
):
    state.issue = pl.astype(0, pl.DT_INT64)
    state.retire = pl.astype(0, pl.DT_INT64)
    if physical_round == -1 or physical_round == state.nextRound:
        batch = state.nextBatch
        head = state.nextHead
        row = state.nextRow
        col = state.nextCol
        nr, nb, nn, nx, ny = plan_find_next_task_bn2s2(
            words, h, cuq, cuk, core, physical_round + 1, mask_mode
        )
        state.nextRound = nr
        state.nextBatch = nb
        state.nextHead = nn
        state.nextRow = nx
        state.nextCol = ny
        if batch >= 0:
            state.issueSlot = state.fetched % 2
            current = slots[state.issueSlot]
            next_key = plan_column_key_bn2s2(h, cuk, nb, nn, ny, mask_mode)
            plan_bind_run_bn2s2(
                words,
                h,
                cuq,
                cuk,
                current,
                state.fetched,
                physical_round,
                sub,
                batch,
                head,
                row,
                col,
                state.previousKey,
                next_key,
                slots[1 - state.issueSlot],
                mask_mode,
            )
            state.previousKey = current.columnKey
            state.fetched = state.fetched + 1
            state.issue = pl.astype(1, pl.DT_INT64)
    if state.retired < state.fetched:
        state.retireSlot = state.retired % 2
        previous = slots[state.retireSlot]
        if previous.readyRound <= physical_round:
            state.retire = pl.astype(1, pl.DT_INT64)


def set_vector_rows_bn2s2(constInfo, runInfo, subIdx, actual_m=False):
    """Couple score SplitM, Vector rows and the clamped Y/dY/LSE load window.

    actual_m is a compile-time choice paired with the Cube score M. Legacy
    fixed-M callers retain a 64-row split; actual-M callers use ceil(M/2).
    Deterministic metadata-plan decoding keeps its separate owner/round logic.
    """
    firstHalf = VECTOR_BASEM_bn2s2
    if actual_m:
        firstHalf = (runInfo.s1RealSize + 1) // 2
        runInfo.firstHalfS1RealSize = firstHalf
    runInfo.halfS1RealSize = pl.max(
        pl.min(runInfo.s1RealSize - subIdx * firstHalf, firstHalf), 0
    )
    rowBase = runInfo.queryOffset + subIdx * firstHalf
    if layout == 1:
        batchEndQ = runInfo.tokenBaseQ + runInfo.batchS1Len
        batchLoadBase = pl.max(
            pl.min(rowBase, batchEndQ - VECTOR_BASEM_bn2s2), runInfo.tokenBaseQ
        )
        # Short final batches borrow earlier physical rows. rowShift and the
        # valid VF row count exclude them; short tensors also shrink the load.
        tensorCapQ = pl.max(constInfo.t1Size - VECTOR_BASEM_bn2s2, 0)
        runInfo.loadBase = pl.min(batchLoadBase, tensorCapQ)
        runInfo.loadBaseTnd = runInfo.loadBase
    else:
        runInfo.loadBase = pl.max(
            pl.min(rowBase, constInfo.s1Size - VECTOR_BASEM_bn2s2), 0
        )
        runInfo.loadBaseTnd = runInfo.boIdx * constInfo.s1Size + runInfo.loadBase
    runInfo.rowShift = rowBase - runInfo.loadBase


def set_normal_bn2s2_run_info_bn2s2(
    ci, info, task, index, sub, local, cuq, cuk, usedq, usedk, cursor
):
    """Decode ordinary/GS-fallback tasks with actual-M score and Vector halves."""
    set_run_info_bn2s2(
        ci, info, task, index, sub, local, cuq, cuk, usedq, usedk, cursor, True
    )


# Ordinary TND swizzle keeps its own metadata protocol and scalar cursor.
NORMAL_SWIZZLE_MAGIC_bn2s2 = 0x46534E31
TND_TAIL_MAGIC_bn2s2 = 0x46545031


def normal_meta_i64_bn2s2(meta, word):
    lo = pl.astype(pl.getval(meta, word), pl.DT_UINT32)
    hi = pl.astype(pl.getval(meta, word + 1), pl.DT_INT64)
    return pl.astype(lo, pl.DT_INT64) + hi * 4294967296


def make_normal_swizzle_bn2s2(meta, constInfo):
    state = pl.struct(
        "FagBn2s2NormalSwizzle",
        active=pl.const(0, pl.DT_INT64),
        tail=pl.const(0, pl.DT_INT64),
        segment=pl.const(0, pl.DT_INT64),
        loop=pl.const(0, pl.DT_INT64),
        deltaCnt=pl.const(0, pl.DT_INT64),
        total=pl.const(0, pl.DT_INT64),
        kind=pl.const(0, pl.DT_INT64),
        batch=pl.const(0, pl.DT_INT64),
        gridBase=pl.const(0, pl.DT_INT64),
        begin=pl.const(0, pl.DT_INT64),
        rows=pl.const(0, pl.DT_INT64),
        cols=pl.const(0, pl.DT_INT64),
        end=pl.const(0, pl.DT_INT64),
        pre=pl.const(0, pl.DT_INT64),
        nxt=pl.const(0, pl.DT_INT64),
    )
    if layout == 1 and swizzle == 1:
        tail = pl.astype(pl.getval(meta, 12), pl.DT_INT64)
        words = pl.astype(pl.getval(meta, 13), pl.DT_INT64)
        if (
            tail >= pl.getval(meta, 7) + 416
            and tail % 16 == 0
            and words >= 16
            and tail + words <= constInfo.normalMetadataWords
            and pl.getval(meta, 14) == 3
            and pl.getval(meta, 15) == TND_TAIL_MAGIC_bn2s2
        ):
            relative = pl.astype(pl.getval(meta, tail + 12), pl.DT_INT64)
            size = pl.astype(pl.getval(meta, tail + 13), pl.DT_INT64)
            if (
                relative >= 16
                and size == 290
                and relative + size <= words
                and pl.getval(meta, tail + 14) == 1
                and pl.getval(meta, tail + 15) == NORMAL_SWIZZLE_MAGIC_bn2s2
            ):
                segment = tail + relative
                if (
                    normal_meta_i64_bn2s2(meta, segment) == NORMAL_SWIZZLE_MAGIC_bn2s2
                    and normal_meta_i64_bn2s2(meta, segment + 2) == 1
                ):
                    eligible = normal_meta_i64_bn2s2(meta, segment + 4)
                    working = normal_meta_i64_bn2s2(meta, segment + 12)
                    largeInvalid = normal_meta_i64_bn2s2(meta, segment + 14)
                    kind = normal_meta_i64_bn2s2(meta, segment + 6)
                    count = normal_meta_i64_bn2s2(meta, segment + 8)
                    total = normal_meta_i64_bn2s2(meta, segment + 10)
                    batch = normal_meta_i64_bn2s2(meta, segment + 26)
                    heads = normal_meta_i64_bn2s2(meta, segment + 28)
                    cores = normal_meta_i64_bn2s2(meta, segment + 30)
                    validIdentity = (
                        count == constInfo.bSize + 1
                        and batch == constInfo.bSize
                        and heads == constInfo.n1Size
                        and cores == constInfo.coreNum
                    )
                    if (
                        eligible == 1
                        and validIdentity
                        and total > 0
                        and kind >= 0
                        and kind <= 2
                        and (working > constInfo.normalL2Bytes or largeInvalid != 0)
                    ):
                        state.active = 1
                        state.tail = tail
                        state.segment = segment
                        state.kind = kind
                        state.total = total
                        state.end = normal_meta_i64_bn2s2(meta, segment + 34)
                        state.rows = (
                            normal_meta_i64_bn2s2(meta, tail + 122) + 127
                        ) // 128
                        state.cols = (
                            normal_meta_i64_bn2s2(meta, tail + 124) + 127
                        ) // 128
                        state.pre = normal_meta_i64_bn2s2(meta, segment + 22)
                        state.nxt = normal_meta_i64_bn2s2(meta, segment + 24)
    return state


def normal_swizzle_coordinate_bn2s2(
    kind, core, cores, heads, m, n, pre, nxt, delta, leftup=False
):
    """ASC kernel_base.h Dense/Casual/Band coordinate transform.

    valid=0 means a padding slot to skip; valid=-1 means this core has no
    column in this batch, and ASC probes the following batch at this loop.
    """
    head = 0
    row = 0
    col = 0
    valid = 1
    if kind == 0:
        row = delta % m
        linear = delta // m * cores + core
        head = linear // n
        col = linear % n
        if linear >= n * heads:
            valid = -1
    elif kind == 1:
        rows = pl.min(m, n)
        cols = 2 * n - rows + 1
        nActual = n
        mActual = rows
        if leftup:
            nActual = pl.min(m, n)
            mActual = m
            rows = 2 * mActual - nActual + 1
            cols = nActual
        row = delta % rows
        linear = delta // rows * cores + core
        head = linear // cols
        col = linear % cols
        if linear >= cols * (heads // 2):
            valid = -1
        elif leftup:
            if row >= col + mActual - nActual + 1:
                row = row - (mActual - nActual + 1)
                head = head * 2 + 1
            else:
                row = mActual - 1 - row
                col = nActual - 1 - col
                head = head * 2
        elif col >= row + nActual - mActual + 1:
            head = head * 2 + 1
            if col < nActual:
                col = 2 * nActual - mActual - col
                row = row + col - nActual + mActual
            else:
                col = 2 * nActual - mActual - col
        else:
            head = head * 2
    else:
        preBlocks = pl.min((pre + 127) // 128 + 1, m)
        nextBlocks = pl.min((nxt + 127) // 128 + 1, n)
        if preBlocks + nextBlocks <= m:
            increasing = nextBlocks - 1
            equal = pl.min(n - nextBlocks + 1, m + 2 - preBlocks - nextBlocks)
            decreasing = pl.max(
                0, pl.min(preBlocks + n - m - 1, preBlocks + nextBlocks - 2)
            )
            rows = m
            if decreasing == 0:
                rows = preBlocks + nextBlocks + equal - 2
            cols = increasing + equal + decreasing
            height = preBlocks + nextBlocks - 1
            linear = delta // height * cores + core
            if decreasing == 0 or cols - rows < 1:
                row = delta % height
                head = linear // cols
                col = linear % cols
                row = row + col - nextBlocks + 1
                if linear >= cols * heads:
                    valid = -1
            else:
                diagonal = delta % height
                temp = diagonal + linear + 1 - nextBlocks
                if temp < 0:
                    head = heads - 1
                    col = temp + rows
                    row = linear + rows
                else:
                    head = temp // rows
                    row = temp % rows
                    col = row + nextBlocks - 1 - diagonal
                    if head == heads - 1 and col >= rows:
                        valid = -1
        else:
            rows = m
            cols = pl.min(m - 1 + nextBlocks, n)
            row = delta % rows
            linear = delta // rows * cores + core
            # Preserve ASC's denominator here; coverage audit guards its domain.
            head = linear // rows
            col = linear - cols * head
            if linear >= cols * heads:
                valid = -1
        if valid == 1:
            if (
                row < 0
                or col < 0
                or row >= m
                or col >= n
                or row - col >= preBlocks
                or col - row >= nextBlocks
            ):
                valid = 0
    return head, row, col, valid


def next_normal_swizzle_bn2s2(meta, constInfo, state, core, cuQ, cuKv):
    # ASC bandLoopIdx skips geometric padding internally. Dense/Causal probe
    # later batches at the same loop and carry deltaCnt across boundaries.
    result = pl.const(-1, pl.DT_INT64)
    while state.loop < state.total:
        loop = state.loop
        # Commit only prefixes exhausted by the monotone physical loop.
        # A speculative probe into a later batch must not advance this cursor.
        while loop >= state.end and state.batch + 1 < constInfo.bSize:
            state.gridBase = state.gridBase + constInfo.n1Size * state.rows * state.cols
            state.batch = state.batch + 1
            state.begin = state.end
            state.end = normal_meta_i64_bn2s2(
                meta, state.segment + 32 + 2 * (state.batch + 1)
            )
            state.rows = (
                normal_meta_i64_bn2s2(meta, state.tail + 122 + 32 * state.batch) + 127
            ) // 128
            state.cols = (
                normal_meta_i64_bn2s2(meta, state.tail + 124 + 32 * state.batch) + 127
            ) // 128
        if mask_mode == 0:
            # A valid current-batch dense slot needs no speculative batch
            # search. Padding still takes the original cross-batch probe.
            if state.kind == 0 and state.rows > 0 and loop < state.end:
                original_delta = loop - state.begin + state.deltaCnt
                dense_delta = pl.max(original_delta, 0)
                linear = dense_delta // state.rows * constInfo.coreNum + core
                if linear < state.cols * constInfo.n1Size:
                    state.deltaCnt = state.deltaCnt + pl.max(-original_delta, 0)
                    result = (
                        state.gridBase + linear * state.rows + dense_delta % state.rows
                    )
                    state.loop = loop + 1
                    break
        gridPrefix = state.gridBase
        found = 0
        for batch in pl.range(state.batch, constInfo.bSize):
            m = state.rows
            n = state.cols
            begin = state.begin
            end = state.end
            if batch != state.batch:
                m = (
                    normal_meta_i64_bn2s2(meta, state.tail + 122 + 32 * batch) + 127
                ) // 128
                n = (
                    normal_meta_i64_bn2s2(meta, state.tail + 124 + 32 * batch) + 127
                ) // 128
                begin = normal_meta_i64_bn2s2(meta, state.segment + 32 + 2 * batch)
                end = normal_meta_i64_bn2s2(meta, state.segment + 32 + 2 * (batch + 1))
            if loop < end and end > begin:
                delta = loop - begin + state.deltaCnt
                if delta < 0:
                    state.deltaCnt = state.deltaCnt - delta
                    delta = 0
                head, row, col, valid = normal_swizzle_coordinate_bn2s2(
                    state.kind,
                    core,
                    constInfo.coreNum,
                    constInfo.n1Size,
                    m,
                    n,
                    state.pre,
                    state.nxt,
                    delta,
                    mask_mode == 4,
                )
                if valid >= 0:
                    found = 1
                    if valid == 1:
                        result = gridPrefix + (head * n + col) * m + row
                    break
            gridPrefix = gridPrefix + constInfo.n1Size * m * n
        state.loop = state.loop + 1
        if result >= 0 or found == 0:
            break
    return result
