# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Vector 侧编排。

阶段编号与 CV 同步 flag 一致::

    V1  sfmg = rowsum(dy * y)
    V2  P    = softmax(S*scale - lse) [+mask]
    V3  dS   = (dP - sfmg) * P -> cast -> L1
    V4  P    -> cast -> L1
    V5/V6    dQ/dK 的 muls(scale) + cast

本文件只做**搬运与编排**：GM->UB 的 load、UB->L1 的 insert、选哪一档 VF；
真正的寄存器级计算在 :mod:`fag_vector_api` 与 :mod:`attenmask`。

pre / post 两个阶段也放在这里：它们同样是纯 vector、同样按 metadata 给的
行区间切，只是被 ``sync_all`` 与主循环完全隔开。
"""

import pypto_pro.language as pl

# --- generated imports (fag_fiximports.py) ---
from fag_common import (
    CUBE_BASEM,
    D_CHUNK_MAXW,
    SINK_REDUCE_BATCH,
    D_NCHUNKS,
    NZP_ROWS,
    SP_ROWS,
    VECTOR_BASEM,
    VECTOR_BASEN,
    VF_LANES,
    VF_LANES_F16,
    chunk_off,
    chunk_valid_d,
)
from memory.fag_mem import (
    POST_ELEMS,
    PRE_ELEMS,
)
from compute.fag_vector_api import (
    broadcast_sub_mul_cast_vf,
    cast_p_vf,
    dsink_partial_vf,
    dsink_reduce_add_vf,
    fill_scaled_zero_bn2s2_vf,
    fill_zero_f16_vf,
    fill_zero_vf,
    muls_cast_vf,
    softmax_grad_front_row192_vf,
    softmax_grad_front_row256_vf,
    softmax_grad_front_row384_vf,
    softmax_grad_front_row512_vf,
    softmax_grad_front_row640_vf,
    softmax_grad_front_row768_vf,
    softmax_grad_front_row_vf,
)
from fag_common import (
    META_FAG_DKV_ROW_ENDS,
    META_FAG_DKV_ROW_STARTS,
    META_FAG_DQ_ROW_ENDS,
    META_FAG_DQ_ROW_STARTS,
)
from fag_tiling import Tmpl
from compute.attenmask import (
    copy_in_atten_mask,
    copy_in_atten_mask_rows,
    muls_sel_simple_softmax_vf,
)
# --- end generated imports ---

# 窗口循环内每窗交错穿插的清零 chunk 上限。大 N 场景（zTotal >> nIter，如
# N32×S4096 时 kBatchRaw≈31）若不封顶，每窗穿插 31 个清零 chunk 严重打断
# MTE3 流水（板测 MTE3 有效带宽 2.8→1.8 TB/s，降 36%）。封顶 2：每窗最多
# 穿插 2 个 chunk 保住 sfmg 载入与清零写的基本重叠，余量全部落排空段——
# sfmg 窗口完成后 MTE3 独占流水，清零 chunk 背靠背发（不穿插 sfmg load/VF），
# 接近 fasg 的顺序大 chunk 清零效率。
_KBATCH_CAP = 2

# ==================================================================
#  UB -> CV 共享 L1
# ==================================================================


def copy_ub2l1(l1_tile, nzp_ub, row_offset, rows):
    """把 NZ+1 缓冲搬进 CV 共享 L1，跳过 bank 错开用的填充块。

    GS 的 nzp_ub 是物理 65 行的非 compact NZ，收窄有效行数
    不改变 VF 的 65 行分形步长。两个 AIV 写到 0 / ceil(s1Real/2)，只搬
    各自有效行；L1 仍按完整 128 行物理步长排布，兼容同址 ZN 转置视图。
    其它路径调用时 rows=64，保持原来的 RowPlusOne 布局。
    """
    pl.set_validshape(nzp_ub, [rows, VECTOR_BASEN])
    pl.insert(l1_tile, nzp_ub, [row_offset, 0])
    pl.set_validshape(nzp_ub, [NZP_ROWS, VECTOR_BASEN])


# ==================================================================
#  V1 —— sfmg = rowsum(dy * y)
# ==================================================================


def _vector_load_rows(constInfo, runInfo):
    """Bound y/dy/lse GM reads by the physical Q view, not the used length.

    s1ViewSize is packed T for TND and physical S for BSND/BNSD. Keep the
    tile's physical shape/stride unchanged; rowShift selects the valid rows
    inside this load window, including windows borrowed across TND batches.
    """
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
        # A zero-row sibling keeps a legal one-row load; VFs/stores remain empty.
        return pl.max(runInfo.halfS1RealSize, 1)
    return pl.min(VECTOR_BASEM, pl.max(constInfo.s1ViewSize - runInfo.loadBase, 0))


def _sfmg_tile_compute(
    constInfo,
    tensor_y,
    tensor_dy,
    idx0,
    loadBase,
    idx2,
    loadRows,
    halfRows,
    rowShift,
    y_group,
    dy_group,
    sfmg_ub,
    rowStride,
    wideRows,
):
    """Shared pre/inline SFMG with the ASC per-lane FP32 addition tree.

    Wide rows are loaded in full, using 32-row pre or 8-row inline slots.
    The output remains the caller's original macro window: rowOff is local
    to that window, so workspace indexing and pre/zero interleaving do not
    change. Narrow loads retain their existing physical row layout.
    """
    if dv_align >= 512:
        for rowOff in pl.range(0, halfRows, wideRows):
            rows = pl.min(wideRows, halfRows - rowOff)
            y_ub = y_group.next()
            dy_ub = dy_group.next()
            pl.set_validshape(y_ub, [rows, constInfo.dvSize])
            pl.set_validshape(dy_ub, [rows, constInfo.dvSize])
            pl.load(
                y_ub,
                tensor_y,
                [idx0, loadBase + rowShift + rowOff, idx2, 0],
                order=[1, 3],
            )
            pl.load(
                dy_ub,
                tensor_dy,
                [idx0, loadBase + rowShift + rowOff, idx2, 0],
                order=[1, 3],
            )
            # Select outside VF: no unsupported runtime VF branch. Each
            # branch performs exactly one reduction per row, after the
            # corresponding ASC tree, including explicitly zeroed tail lanes.
            if constInfo.dvSize <= 384:
                tail = constInfo.dvSize - 256
                softmax_grad_front_row384_vf(
                    sfmg_ub,
                    y_ub,
                    dy_ub,
                    rows,
                    0,
                    dv_align,
                    (tail + 1) // 2,
                    tail // 2,
                    rowOff,
                )
            elif constInfo.dvSize <= 512:
                tail = constInfo.dvSize - 384
                softmax_grad_front_row512_vf(
                    sfmg_ub,
                    y_ub,
                    dy_ub,
                    rows,
                    0,
                    dv_align,
                    (tail + 1) // 2,
                    tail // 2,
                    rowOff,
                )
            elif dv_align == 768:
                # Keep wider VF reads out of the physical 512-column key.
                if constInfo.dvSize <= 640:
                    tail = constInfo.dvSize - 512
                    softmax_grad_front_row640_vf(
                        sfmg_ub,
                        y_ub,
                        dy_ub,
                        rows,
                        0,
                        dv_align,
                        (tail + 1) // 2,
                        tail // 2,
                        rowOff,
                    )
                else:
                    tail = constInfo.dvSize - 640
                    softmax_grad_front_row768_vf(
                        sfmg_ub,
                        y_ub,
                        dy_ub,
                        rows,
                        0,
                        dv_align,
                        (tail + 1) // 2,
                        tail // 2,
                        rowOff,
                    )
    else:
        y_ub = y_group.next()
        dy_ub = dy_group.next()
        pl.set_validshape(y_ub, [loadRows, constInfo.dvSize])
        pl.set_validshape(dy_ub, [loadRows, constInfo.dvSize])
        pl.load(y_ub, tensor_y, [idx0, loadBase, idx2, 0], order=[1, 3])
        pl.load(dy_ub, tensor_dy, [idx0, loadBase, idx2, 0], order=[1, 3])
        if dv_align == 256:
            tail = constInfo.dvSize - 128
            softmax_grad_front_row256_vf(
                sfmg_ub,
                y_ub,
                dy_ub,
                halfRows,
                rowShift,
                rowStride,
                (tail + 1) // 2,
                tail // 2,
            )
        elif dv_align == 192:
            tail = constInfo.dvSize - 128
            softmax_grad_front_row192_vf(
                sfmg_ub,
                y_ub,
                dy_ub,
                halfRows,
                rowShift,
                rowStride,
                (tail + 1) // 2,
                tail // 2,
            )
        else:
            softmax_grad_front_row_vf(
                sfmg_ub,
                y_ub,
                dy_ub,
                halfRows,
                rowShift,
                rowStride,
                constInfo.dvEvenSize,
                constInfo.dvOddSize,
            )


def load_precomputed_sfmg(constInfo, runInfo, tensor_sfmg_ws, sfmg_ub):
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    rows = runInfo.halfS1RealSize
    if rows > 0:
        # 只搬有效行（rows），并 clamp 确保不超出 ws 的 (head,batch) 段：
        # rowShift 使读取起点 = rowBase（全局行），rows 使终点 = rowBase+rows
        # 不会超出本 batch 的物理范围。剩余 sfmg_ub[rows..63] 为残留，
        # GS 的 V3 按有效行计算；与内联路径的 VF 写入范围一致。
        # validshape/order 与 lse 同一套：[64,1] DN tile 取 [rows,1]，
        # order=[1,0] 把 tile 行轴映到 ws 的扁平 token 轴。
        pl.set_validshape(sfmg_ub, [rows, 1])
        pl.load(
            sfmg_ub,
            tensor_sfmg_ws,
            [0, headQ * constInfo.t1Size + runInfo.loadBaseTnd + runInfo.rowShift],
            order=[1, 0],
        )
        # 恢复物理形状 [64,1]；GS 的 V3 仍只读取 rows 行。
        pl.set_validshape(sfmg_ub, [VECTOR_BASEM, 1])


def process_vec1(
    constInfo,
    runInfo,
    subIdx,
    tensor_y,
    tensor_dy,
    tensor_sfmg_ws,
    y_ub,
    dy_ub,
    sfmg_ub,
    rowStride,
):
    """softmaxGradFront: sfmg = rowsum(dy * y)。一趟 VECTOR_BASEM 行。

    dv_align>=128 走前移（对齐老 FAG 的 preSfmg）：sfmg 一行一值、与 s2
    无关，主循环每个 tile 重算会被重复 s2Outer 遍，大 D 档是 V 侧的主要
    浪费。pre 阶段(process_presfmg)已按 (head, 全局 token 行) 算好全量存
    ws，这里只搬本子核有效行：起点 loadBaseTnd+rowShift 即 rowBase 的
    全局行；GS 的 V3 只读取本子核有效行，不消费尾部残留。
    dv_align=64 保持内联（一趟半宽 VF，前移的 GM 往返不划算）。
    ★门必须与写入方逐一对齐（读未初始化的 ws 是灾难）：写侧
    process_presfmg 与读侧共用 constInfo.enablePreSfmg（host 按 fasg
    enablePreSfmg 判据统一裁决，见 flash_attn_grad_tiling_plan.cpp 的
    DecidePreSfmg）。两端同标志是结构保证，不再各自用 template/dv_align
    拼条件。
    """
    if constInfo.enablePreSfmg == 1:
        load_precomputed_sfmg(constInfo, runInfo, tensor_sfmg_ws, sfmg_ub)
    else:
        # 未前移：主循环内联重算，数值与 pre 路径一致（同一套 VF）。
        loadRows = _vector_load_rows(constInfo, runInfo)
        _sfmg_tile_compute(
            constInfo,
            tensor_y,
            tensor_dy,
            runInfo.idx0Q,
            runInfo.loadBase,
            runInfo.idx2Q,
            loadRows,
            runInfo.halfS1RealSize,
            runInfo.rowShift,
            y_ub,
            dy_ub,
            sfmg_ub,
            rowStride,
            8,
        )


def _zero_step(zt, zdq, zdk, zdv, dqStart, zS1, dkStart, zS2, dvStart, zS3, zc):
    """issue 下一个统一清零任务（三段逻辑连续编号，运行期条件分发）。

    zc 为全局元素游标（dq/dk/dv 三段拼接的连续下标）：每任务从 zc 起
    清 min(chunk, 段尾−zc) 个，返回值推进实际清零量 n——段尾用运行期
    min 裁 validshape，保证 zc 恰好落在段边界上，下一任务自动进入下一
    段且段内偏移归零，绝不越段（越段会误清 dsink/sfmg 段，清零对它们
    不幂等）。zc 越过 zS3 时 n=0 空转、游标冻结（排空段允许 kBatch 过量）。
    供 process_presfmg 的交错清零：清零任务与窗口 load/VF 交替 issue，
    MTE3 不被整段清零灌满阻塞 sfmg 写。
    """
    chunk = PRE_ELEMS[d_align]
    n = 0
    if zc < zS1:
        n = pl.min(chunk, zS1 - zc)
        pl.set_validshape(zt, [1, n])
        pl.store(zdq, zt, [0, dqStart + zc])
    else:
        if zc < zS2:
            n = pl.min(chunk, zS2 - zc)
            pl.set_validshape(zt, [1, n])
            pl.store(zdk, zt, [0, dkStart + (zc - zS1)])
        else:
            if zc < zS3:
                n = pl.min(chunk, zS3 - zc)
                pl.set_validshape(zt, [1, n])
                pl.store(zdv, zt, [0, dvStart + (zc - zS2)])
    return zc + n


def normal_wide_tnd_presfmg_bn2s2(
    ci, y, dy, ws, y_group, dy_group, out_group, flat_group, row_stride, ws_offset
):
    # Split physical token/head pairs at row granularity. The workspace is
    # head-major; Y and dY are token-major and keep their original head stride.
    total = ci.t1Size * ci.n1Size
    aiv = pl.get_block_idx()
    # Keep each 32-byte GM write block under one AIV. Only the physical end
    # and head crossings within a single AIV may need a short store.
    share = (total + 8 * ci.aivCount - 1) // (8 * ci.aivCount) * 8
    # The first row may be inside a 32-byte GM block. Shift each interior
    # boundary forward to the next absolute block, preserving the first row.
    prefix = (8 - ws_offset % 8) % 8
    position = aiv * share + pl.min(aiv, 1) * prefix
    end = pl.min((aiv + 1) * share + prefix, total)
    # An integral number of 32-row input slots avoids an extra short VF
    # inside every full output window. The two output views share a mutex.
    window = SP_ROWS[dv_align] // 32 * 32
    while position < end:
        head = position // ci.t1Size
        token = position - head * ci.t1Size
        rows = pl.min(window, pl.min(end - position, ci.t1Size - token))
        out = out_group.next()
        flat = flat_group.next()
        _sfmg_tile_compute(
            ci,
            y,
            dy,
            0,
            token,
            head,
            rows,
            rows,
            0,
            y_group,
            dy_group,
            out,
            row_stride,
            32,
        )
        pl.set_validshape(flat, [1, rows])
        pl.store(ws, flat, [0, position])
        position = position + rows


def normal_causal_kv_covered_bn2s2(ci, cuq, cuk, sw_active):
    covered = pl.const(0, pl.DT_INT64)
    if (
        ci.wideKvSlots == 1
        and sw_active == 0
        and ci.gSize == 1
        and ci.sequsedQOptional == 0
        and ci.sequsedKvOptional == 0
    ):
        # For right-down causal the final real query can see every real key.
        # The native column owner publishes s2RealSize rows at its last task.
        # An orphan KV batch is the sole gap in the untruncated physical grid.
        covered = 1
        for b in pl.range(0, ci.bSize):
            if pl.getval(cuq, b + 1) == pl.getval(cuq, b) and pl.getval(
                cuk, b + 1
            ) > pl.getval(cuk, b):
                covered = 0
    return covered


def process_presfmg(
    constInfo,
    tensor_y,
    tensor_dy,
    tensor_sfmg_ws,
    y_ub,
    dy_ub,
    sfmg_ub,
    sfmg_flat,
    rowStride,
    zt,
    zdq,
    zdk,
    zdv,
    dqStart,
    zS1,
    dkStart,
    zS2,
    dvStart,
    zS3,
):
    """softmaxGrad 前移：pre 阶段一次算完全部行的 sfmg 写入 workspace。

    行空间 [N1, T1]（token 主序扁平，与 TND lse 同构）按窗口行块切给全部
    AIV 均分（窗口行数按档取 SP_ROWS）；每块复用 _sfmg_tile_compute（与
    主循环 V1 同一套 VF 分档），结果 store 到 ws 的 head*T1+全局行 处。

    sp_out 与 sp_out_flat 双槽（depth 2）：循环内同步 .next() 乒乓轮转，
    VF(N) 写与 store(N-1) 读并行；sp_y/sp_dy 也各双槽（.next() 乒乓）：
    load(N+1) 写下一槽、VF(N) 读本槽，物理隔离无需跨队列 wait。
    **不能退回 sp_y/sp_dy 单槽（.current() 固定单 tile）**：pypto 对
    同一 tile 的跨队列操作不插 wait，load(N+1) 会覆盖 VF(N) 正在读的
    窗口（上板 sfmg 大面积错值 + 非逐位 + NaN）。

    块内地址分解：
      TND —— 全局 token 行即 packed 物理行，窗口任意切都合法；
      BSND/BNSD —— 按 (head, b, s 块) 切，窗口不跨 batch（s1Size 裁剪），
        视图系数 coefB0Q/coefN0/coefN2 与 set_run_info 同一套。
    ws 行号与 process_vec1 前移分支的读取口径 (loadBaseTnd) 一致——ws 是
    扁平缓存，写侧窗口行数与读侧 64 行窗口无需一致。
    落盘走 sfmg_flat（[1,SP_ROWS] 扁平视图，同址同号）：store 的 order 必须
    升序，[rows,1] DN tile 无法用 order=[1,0] 映到扁平轴，改走 zt/sink
    的恒等映射口径（validshape [1,rows] + 无 order）。
    """
    t1Size = constInfo.t1Size
    spRows = SP_ROWS[dv_align]
    aivIdx = pl.get_block_idx()
    zc = 0
    chunk = PRE_ELEMS[d_align]
    # 任务数三段分计：段尾裁宽后每段末任务步长 < chunk，全局 ceil(zS3/chunk)
    # 会少计 0~2 个任务（每段各可能少 1 个），段尾就清不干净。
    zTotal = (
        (zS1 + chunk - 1) // chunk
        + ((zS2 - zS1) + chunk - 1) // chunk
        + ((zS3 - zS2) + chunk - 1) // chunk
    )
    if layout == 1:
        tBlocks = (t1Size + spRows - 1) // spRows
        totalBlocks = constInfo.n1Size * tBlocks
        perCore = (totalBlocks + constInfo.aivCount - 1) // constInfo.aivCount
        biStart = pl.min(aivIdx * perCore, totalBlocks)
        nIter = pl.max(pl.min(biStart + perCore, totalBlocks) - biStart, 0)
        nIterSafe = pl.max(nIter, 1)
        # kBatch 封顶 _KBATCH_CAP：保住 sfmg 载入与清零写的基本重叠，
        # 余量全部落排空段集中背靠背发（设计依据见 _KBATCH_CAP）。
        kBatch = pl.min((zTotal + nIterSafe - 1) // nIterSafe, _KBATCH_CAP)
        for i in pl.range(0, nIter):
            for _ in pl.range(0, kBatch):
                zc = _zero_step(
                    zt, zdq, zdk, zdv, dqStart, zS1, dkStart, zS2, dvStart, zS3, zc
                )
            bi = biStart + i
            head = bi // tBlocks
            t0 = (bi - head * tBlocks) * spRows
            rows = pl.min(spRows, t1Size - t0)
            # enablePreSfmg=0 时只保留清零（主循环 V1 走内联重算），
            # sp 载入/VF/store 全部跳过；标志按 launch 恒定，乒乓轮转一致。
            if constInfo.enablePreSfmg == 1:
                out_t = sfmg_ub.next()
                flat_t = sfmg_flat.next()
                _sfmg_tile_compute(
                    constInfo,
                    tensor_y,
                    tensor_dy,
                    0,
                    t0,
                    head,
                    rows,
                    rows,
                    0,
                    y_ub,
                    dy_ub,
                    out_t,
                    rowStride,
                    32,
                )
                # 扁平视图落盘：[1,rows] 恒等映射，只落有效 rows 个元素。
                pl.set_validshape(flat_t, [1, rows])
                pl.store(tensor_sfmg_ws, flat_t, [0, head * t1Size + t0])
    else:
        sBlocks = (constInfo.s1Size + spRows - 1) // spRows
        bsBlocks = constInfo.bSize * sBlocks
        totalBlocks = constInfo.n1Size * bsBlocks
        perCore = (totalBlocks + constInfo.aivCount - 1) // constInfo.aivCount
        biStart = pl.min(aivIdx * perCore, totalBlocks)
        nIter = pl.max(pl.min(biStart + perCore, totalBlocks) - biStart, 0)
        nIterSafe = pl.max(nIter, 1)
        # 同 TND 分支：kBatch 封顶 _KBATCH_CAP，余量落排空段集中发。
        kBatch = pl.min((zTotal + nIterSafe - 1) // nIterSafe, _KBATCH_CAP)
        for i in pl.range(0, nIter):
            for _ in pl.range(0, kBatch):
                zc = _zero_step(
                    zt, zdq, zdk, zdv, dqStart, zS1, dkStart, zS2, dvStart, zS3, zc
                )
            bi = biStart + i
            head = bi // bsBlocks
            rem = bi - head * bsBlocks
            b = rem // sBlocks
            s = (rem - b * sBlocks) * spRows
            rows = pl.min(spRows, constInfo.s1Size - s)
            # 同 TND 分支：enablePreSfmg=0 时只清零，不搬 y/dO、不算 sfmg。
            if constInfo.enablePreSfmg == 1:
                out_t = sfmg_ub.next()
                flat_t = sfmg_flat.next()
                _sfmg_tile_compute(
                    constInfo,
                    tensor_y,
                    tensor_dy,
                    b * constInfo.coefB0Q + head * constInfo.coefN0,
                    s,
                    head * constInfo.coefN2,
                    rows,
                    rows,
                    0,
                    y_ub,
                    dy_ub,
                    out_t,
                    rowStride,
                    32,
                )
                # 同上：扁平视图 [1,rows] 恒等映射，落有效 rows 个元素。
                pl.set_validshape(flat_t, [1, rows])
                pl.store(
                    tensor_sfmg_ws,
                    flat_t,
                    [0, head * t1Size + b * constInfo.s1Size + s],
                )

    # 排空段：kBatch*nIter 可能 < zTotal（ceil 截断）或 nIter=0（无窗任务
    # 的核全部任务在此）；亦可能 > zTotal（_zero_step 的越界保护空转）。
    # zS3=0（BN2 调用点）zTotal=0 → zTail=0，编译期零循环。
    zTail = pl.max(zTotal - kBatch * nIter, 0)
    for _ in pl.range(0, zTail):
        zc = _zero_step(zt, zdq, zdk, zdv, dqStart, zS1, dkStart, zS2, dvStart, zS3, zc)


# ==================================================================
#  V2 —— P = simpleSoftmax(S * scale, lse) [+ atten mask]
# ==================================================================


def load_vec_lse_rows(constInfo, runInfo, tensor_lse, lse_ub, loadRows):
    """Shared LSE addressing; callers retain their physical/actual-row policy."""
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    pl.set_validshape(lse_ub, [loadRows, 1])
    if layout == 1:
        # TND：lse 是 (N1, T1) token 主序。变长 TND 的 loadBase 已是全局
        # token 行（tokenBaseQ + 块内偏移，由 _set_run_info_tnd_varlen 计算）；
        # 等长 TND 的 loadBase 也是全局坐标（tokenBase = boIdx*s1 已折叠）。
        # 0930 bisect-sink 定案移植。
        rowT = runInfo.loadBase
        pl.load(lse_ub, tensor_lse, [headQ, rowT], order=[1, 0])
    else:
        pl.load(
            lse_ub, tensor_lse, [runInfo.boIdx, headQ, runInfo.loadBase], order=[2, 0]
        )


def prefetch_vec_lse(constInfo, runInfo, tensor_lse, lse_ub):
    load_vec_lse_rows(
        constInfo, runInfo, tensor_lse, lse_ub, _vector_load_rows(constInfo, runInfo)
    )


def process_vec2(
    constInfo,
    runInfo,
    subIdx,
    tensor_lse,
    mm2_ub,
    lse_ub,
    tensor_mask,
    mask_ub,
    mask_pre_ub,
    lse_preloaded=False,
):
    if not lse_preloaded:
        prefetch_vec_lse(constInfo, runInfo, tensor_lse, lse_ub)
    if mask_mode != 0:
        copy_in_atten_mask(
            constInfo, runInfo, subIdx, tensor_mask, mask_ub, mask_pre_ub
        )
    muls_sel_simple_softmax_vf(
        mm2_ub,
        lse_ub,
        mask_ub,
        constInfo.scaleValue,
        runInfo.halfS1RealSize,
        runInfo.rowShift,
    )


# ==================================================================
#  V3 —— dS = (dP - sfmg) * P，cast + ND2NZ 落 L1（mm4 读同址 ZN 视图）
# ==================================================================


def process_vec3(constInfo, runInfo, subIdx, ds_l1, mm1_ub, mm2_ub, sfmg_ub, nzp_ub):
    rowOff = subIdx * runInfo.firstHalfS1RealSize
    rows = VECTOR_BASEM
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
        rows = runInfo.halfS1RealSize
    if rows > 0:
        broadcast_sub_mul_cast_vf(
            nzp_ub, mm1_ub, mm2_ub, sfmg_ub, constInfo.scaleValue, rows
        )
        # L1 物理步长不变；mm4 仍从同址 ZN 视图读取 dS^T。
        copy_ub2l1(ds_l1, nzp_ub, rowOff, rows)


# ==================================================================
#  V4 —— P cast + ND2NZ 落 L1（mm5 读同址 ZN 视图）
# ==================================================================


def process_vec4(constInfo, runInfo, subIdx, p_l1, mm2_ub, nzp_ub):
    rowOff = subIdx * runInfo.firstHalfS1RealSize
    rows = VECTOR_BASEM
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
        rows = runInfo.halfS1RealSize
    if rows > 0:
        cast_p_vf(nzp_ub, mm2_ub, rows)
        copy_ub2l1(p_l1, nzp_ub, rowOff, rows)


# ==================================================================
#  Pre —— dq/dk/dv 累加区清零
# ==================================================================


def process_sink_partial(
    constInfo,
    runInfo,
    tensor_slots,
    aiv_idx,
    p_ub,
    dp_ub,
    lse_ub,
    sink_part_ub,
    sink_val,
):
    """本 s2 块的 dsink 贡献，原子加到槽位区本 AIV 行的 headQ 列。"""
    rows = runInfo.halfS1RealSize
    if rows <= 0:
        return
    s2Real = runInfo.s2RealSize
    if s2Real <= 0:
        return
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    loopTimes = (s2Real + VF_LANES - 1) // VF_LANES - 1
    tailCols = s2Real - loopTimes * VF_LANES
    dsink_partial_vf(
        sink_part_ub,
        p_ub,
        dp_ub,
        lse_ub,
        sink_val,
        rows,
        loopTimes,
        tailCols,
        runInfo.rowShift,
    )
    pl.set_validshape(sink_part_ub, [1, 1])
    pl.store(
        tensor_slots,
        sink_part_ub,
        [0, aiv_idx * constInfo.n1Size + headQ],
        atomic=pl.AtomicType.AtomicAdd,
    )
    pl.set_validshape(sink_part_ub, [1, VECTOR_BASEN])


def process_sink_reduce(
    constInfo, tensor_slots, tensor_dsink, acc_ub, row_ub, aiv_count
):
    """AIV0 按行序把各 AIV 的 dsink 部分和加到输出。

    ``row_ub`` 是 SINK_REDUCE_BATCH 个独立的 1 行 tile：每批发 BATCH 次 load
    （偏移与原逐行版完全相同，不依赖任何行距假设），只等 1 次 bar_mte2。
    实测逐行版每次 load 后都排空流水，aivCount=72 时这一项占 sink 固定开销
    13.8us 里的 7.5us。
    """
    row0 = row_ub[0].current()
    row1 = row_ub[1].current()
    row2 = row_ub[2].current()
    row3 = row_ub[3].current()
    n1Size = constInfo.n1Size
    nsegs = VECTOR_BASEN // VF_LANES
    nChunks = (n1Size + VECTOR_BASEN - 1) // VECTOR_BASEN
    nRest = aiv_count - 1
    nBatch = (nRest + SINK_REDUCE_BATCH - 1) // SINK_REDUCE_BATCH
    for c in pl.range(0, nChunks):
        w = pl.min(VECTOR_BASEN, n1Size - c * VECTOR_BASEN)
        pl.set_validshape(acc_ub, [1, w])
        pl.load(acc_ub, tensor_slots, [0, c * VECTOR_BASEN])
        pl.system.bar_mte2()
        for b in pl.range(0, nBatch):
            r0 = 1 + b * SINK_REDUCE_BATCH
            k = pl.min(SINK_REDUCE_BATCH, aiv_count - r0)
            pl.set_validshape(row0, [1, w])
            pl.load(row0, tensor_slots, [0, (r0 + 0) * n1Size + c * VECTOR_BASEN])
            if k > 1:
                pl.set_validshape(row1, [1, w])
                pl.load(row1, tensor_slots, [0, (r0 + 1) * n1Size + c * VECTOR_BASEN])
            if k > 2:
                pl.set_validshape(row2, [1, w])
                pl.load(row2, tensor_slots, [0, (r0 + 2) * n1Size + c * VECTOR_BASEN])
            if k > 3:
                pl.set_validshape(row3, [1, w])
                pl.load(row3, tensor_slots, [0, (r0 + 3) * n1Size + c * VECTOR_BASEN])
            pl.system.bar_mte2()
            dsink_reduce_add_vf(acc_ub, row0, nsegs)
            if k > 1:
                dsink_reduce_add_vf(acc_ub, row1, nsegs)
            if k > 2:
                dsink_reduce_add_vf(acc_ub, row2, nsegs)
            if k > 3:
                dsink_reduce_add_vf(acc_ub, row3, nsegs)
        pl.set_validshape(acc_ub, [1, w])
        pl.store(tensor_dsink, acc_ub, [0, c * VECTOR_BASEN])


def process_sink_zero(tensor_slots, constInfo, zt, aiv_idx):
    """清本 AIV 的 dsink 槽位行，供主循环 atomicAdd。"""
    fill_zero_vf(zt, (constInfo.n1Size + VF_LANES - 1) // VF_LANES)
    process_pre(tensor_slots, zt, aiv_idx * constInfo.n1Size, constInfo.n1Size)


def process_pre_chunked(tensor_ws, zt, elemStart, nElems, chunk):
    """Clear a flat range with the caller's resource-sized chunk."""
    nFull = nElems // chunk
    rem = nElems - nFull * chunk
    off = elemStart
    pl.set_validshape(zt, [1, chunk])
    for _ in pl.range(0, nFull):
        pl.store(tensor_ws, zt, [0, off])
        off = off + chunk
    nTail = (rem + chunk - 1) // chunk
    for _ in pl.range(0, nTail):
        pl.set_validshape(zt, [1, rem])
        pl.store(tensor_ws, zt, [0, off])


def process_pre(tensor_ws, zt, elemStart, nElems):
    process_pre_chunked(tensor_ws, zt, elemStart, nElems, PRE_ELEMS[d_align])


def process_pre_from_meta_chunked(
    meta, fagOff, tensor_dq, tensor_dk, tensor_dv, dQ, dV, zt, chunk, clear_kv=1
):
    """Shared metadata row ownership; resource capacity stays a caller argument."""
    vBlockIdx = pl.get_block_idx()
    one64 = pl.const(1, pl.DT_INT64)
    dqRowStart = pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdx) * one64
    dqRowEnd = pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdx) * one64
    dkvRowStart = pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdx) * one64
    dkvRowEnd = pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdx) * one64
    dqNElems = (dqRowEnd - dqRowStart) * dQ
    kvRows = (dkvRowEnd - dkvRowStart) * clear_kv
    dkvNElems = kvRows * dQ
    dvNElems = kvRows * dV
    preElems = pl.min(chunk, pl.max(dqNElems, pl.max(dkvNElems, dvNElems)))
    pl.set_validshape(zt, [1, chunk])
    if bn2_need_zero == 1:
        fill_zero_f16_vf(zt, (preElems + VF_LANES_F16 - 1) // VF_LANES_F16)
    else:
        fill_zero_vf(zt, (preElems + VF_LANES - 1) // VF_LANES)
    process_pre_chunked(tensor_dq, zt, dqRowStart * dQ, dqNElems, chunk)
    process_pre_chunked(tensor_dk, zt, dkvRowStart * dQ, dkvNElems, chunk)
    process_pre_chunked(tensor_dv, zt, dkvRowStart * dV, dvNElems, chunk)


def process_pre_from_meta(meta, fagOff, tensor_dq, tensor_dk, tensor_dv, dQ, dV, zt):
    process_pre_from_meta_chunked(
        meta, fagOff, tensor_dq, tensor_dk, tensor_dv, dQ, dV, zt, PRE_ELEMS[d_align]
    )


def zero_tnd_empty_batch(
    constInfo, cu_q, cu_kv, tensor_dq, tensor_dk, tensor_dv, zt, clear_dq=True
):
    """变长 TND：把「空 batch」对侧的 out 行清零。

    packed 索引空间按每 batch 的有效块数拼起来，空 batch（一侧长度为 0）
    不产生任何任务，主循环不会写它的 out；host 读不到 cu_seqlens，无法像
    asc 那样据此退回 GS1S2，只能在这里按 batch 补零，语义等同 GS 的
    pre 清零 + post 全量写回。

    行空间是 token 主序紧凑视图：
      dq      [T1*N1, D]   元素偏移 = token * N1 * D
      dk/dv   [T2*N2, D/Dv] 元素偏移 = token * N2 * D
    一个 batch 的 token 区间对全部 head 连续，直接按 [token0, token1) 整段清。
    各 AIV 按 batch 均分；只有空 batch 才真正产生清零任务。
    """
    aivIdx = pl.get_block_idx()
    aivNum = constInfo.aivCount
    bSize = constInfo.bSize
    start = bSize * aivIdx // aivNum
    end = bSize * (aivIdx + 1) // aivNum
    n1 = constInfo.n2Size * constInfo.gSize
    n2 = constInfo.n2Size
    dQ = constInfo.dSize
    dV = constInfo.dvSize
    one64 = pl.const(1, pl.DT_INT64)
    pl.set_validshape(zt, [1, PRE_ELEMS[d_align]])
    fill_zero_f16_vf(zt, PRE_ELEMS[d_align] // VF_LANES_F16)
    for i in pl.range(start, end):
        q0 = pl.getval(cu_q, i) * one64
        q1 = pl.getval(cu_q, i + 1) * one64
        k0 = pl.getval(cu_kv, i) * one64
        k1 = pl.getval(cu_kv, i + 1) * one64
        # KV 为空：dq 整段为 0（没有 key 参与 softmax）。
        if clear_dq:
            if q1 > q0:
                if k1 == k0:
                    process_pre(tensor_dq, zt, q0 * n1 * dQ, (q1 - q0) * n1 * dQ)
        # Q 为空：dk/dv 整段为 0（该 batch 对 KV 没有贡献）。
        if k1 > k0:
            if q1 == q0:
                process_pre(tensor_dk, zt, k0 * n2 * dQ, (k1 - k0) * n2 * dQ)
                process_pre(tensor_dv, zt, k0 * n2 * dV, (k1 - k0) * n2 * dV)


def process_nz_pre(tensor_ws, elems, aiv_count, zt):
    """Clear padding and TND gaps as well as valid rows before NZ AtomicAdd."""
    per_core = ((elems + aiv_count * 128 - 1) // (aiv_count * 128)) * 128
    begin = pl.min(pl.get_block_idx() * per_core, elems)
    count = pl.min(per_core, elems - begin)
    pl.set_validshape(zt, [1, PRE_ELEMS[d_align]])
    fill_zero_vf(zt, (pl.min(count, PRE_ELEMS[d_align]) + VF_LANES - 1) // VF_LANES)
    process_pre(tensor_ws, zt, begin, count)


# ==================================================================
#  Post —— dQ/dK/dV 的 scale + cast
# ==================================================================


def process_post(tensor_ws, tensor_out, f32_ub, f16_ub, scale, elemStart, nElems):
    """fp32 workspace -> muls(scale) -> cast 16-bit，按一维元素切。

    本核 [elemStart, elemStart+nElems)。full tile 的 VF 段数是编译期常量；
    尾块用 trip count 0/1 的第二个 pl.range。
    """
    chunk = POST_ELEMS[d_align]
    nFull = nElems // chunk
    rem = nElems - nFull * chunk
    nsegsFull = chunk // VECTOR_BASEN
    off = elemStart
    for _ in pl.range(0, nFull):
        ft = f32_ub.next()
        ht = f16_ub.next()
        pl.set_validshape(ft, [1, chunk])
        pl.set_validshape(ht, [1, chunk])
        pl.load(ft, tensor_ws, [0, off])
        muls_cast_vf(ht, ft, scale, nsegsFull)
        pl.store(tensor_out, ht, [0, off])
        off = off + chunk
    nTail = (rem + chunk - 1) // chunk
    nsegsTail = (rem + VECTOR_BASEN - 1) // VECTOR_BASEN
    for _ in pl.range(0, nTail):
        ft = f32_ub.next()
        ht = f16_ub.next()
        pl.set_validshape(ft, [1, rem])
        pl.set_validshape(ht, [1, rem])
        pl.load(ft, tensor_ws, [0, off])
        muls_cast_vf(ht, ft, scale, nsegsTail)
        pl.store(tensor_out, ht, [0, off])


# ==================================================================
#  BN2 模板的 vector 阶段
# ==================================================================


def muls_cast_ub_out(
    tensor_out, src_ub, dst_ub, idx0, row, idx2, rows, scale, dim_size, d_off=0
):
    """UB 里的 fp32 结果 → Muls(scale) + Cast → out GM。

    整块 ``[VECTOR_BASEM, VECTOR_BASEN]`` 一次 cast，再按 ``rows``/``dim_size``
    收窄落盘。BN2 narrow 的 v5/v6 与 192 档 dK 的每个 D 分块（``d_off`` 指定列
    区间）都用这条；行区间由调用点算（``halfS1RealSize`` / ``firstHalfS2RealSize``，
    DualModeSplitM 的切点不能钉 64）。
    """
    if rows <= 0:
        return
    pl.set_validshape(src_ub, [VECTOR_BASEM, VECTOR_BASEN])
    pl.set_validshape(dst_ub, [VECTOR_BASEM, VECTOR_BASEN])
    muls_cast_vf(dst_ub, src_ub, scale, VECTOR_BASEM * VECTOR_BASEN // VF_LANES_F16)
    pl.set_validshape(dst_ub, [rows, dim_size])
    pl.store(
        tensor_out,
        dst_ub,
        [idx0, row, idx2, d_off],
        order=[1, 3],
        atomic=pl.AtomicType.AtomicNone,
    )
    pl.set_validshape(dst_ub, [VECTOR_BASEM, VECTOR_BASEN])


def muls_cast_ws_out(
    tensor_out,
    tensor_ws,
    ws_load_ub,
    ws_cast_ub,
    idx0,
    row,
    idx2,
    rows,
    scale,
    dim_size,
):
    """4D fp32 workspace → 逐 D 分块 load → Muls(scale) + Cast → out GM。

    BN2 wide（d_align>128）：dQ 三档、dK 256/768、dV 768。cube 侧用
    ``dqFirst``/``dkvAcc`` 保证首写 AtomicNone，故这里不需要 pre 清零，也没有
    全局 post。整块 cast（chunk=96/128 列）让 16-bit 行宽达到 192B/256B。
    """
    if rows <= 0:
        return
    tile_w = D_CHUNK_MAXW[d_align]
    nSegs = tile_w * VECTOR_BASEM // VF_LANES_F16
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
        nSegs = (tile_w * rows + VF_LANES_F16 - 1) // VF_LANES_F16
    for i in pl.range(0, D_NCHUNKS[d_align]):
        dOff = chunk_off(i)
        w = chunk_valid_d(dim_size, i)
        if w > 0:
            pl.set_validshape(ws_load_ub, [rows, w])
            pl.set_validshape(ws_cast_ub, [rows, w])
            pl.load(ws_load_ub, tensor_ws, [idx0, row, idx2, dOff], order=[1, 3])
            muls_cast_vf(ws_cast_ub, ws_load_ub, scale, nSegs)
            pl.store(
                tensor_out,
                ws_cast_ub,
                [idx0, row, idx2, dOff],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicNone,
            )
    pl.set_validshape(ws_load_ub, [VECTOR_BASEM, tile_w])
    pl.set_validshape(ws_cast_ub, [VECTOR_BASEM, tile_w])


def process_dq_multiblk(
    tensor_dq, tensor_dq_ws, src_ub, dst_ub, constInfo, runInfo, subIdx, cBlockIdx
):
    """只在本 s1 列的最后一个有效 s2 块（``dqLast``）从 per-core 2D 条带 load。

    累加已由 cube 按 needAtomic 写在 workspace 上。src_ub 是 load 缓冲。
    """
    rows = runInfo.halfS1RealSize
    rowOff = (
        cBlockIdx * constInfo.s1Outer + runInfo.s1oIdx
    ) * CUBE_BASEM + subIdx * VECTOR_BASEM
    pl.set_validshape(src_ub, [VECTOR_BASEM, VECTOR_BASEN])
    if rows > 0:
        pl.set_validshape(src_ub, [rows, constInfo.dSize])
        pl.load(src_ub, tensor_dq_ws, [rowOff, 0])
        pl.set_validshape(src_ub, [VECTOR_BASEM, VECTOR_BASEN])
    muls_cast_ub_out(
        tensor_dq,
        src_ub,
        dst_ub,
        runInfo.idx0Q,
        runInfo.queryOffset + subIdx * VECTOR_BASEM,
        runInfo.idx2Q,
        rows,
        constInfo.scaleValue,
        constInfo.dSize,
    )


# BN2S2 implementation: separate resource and synchronization semantics.
from compute.attenmask import copy_in_atten_mask
from memory import fag_mem as mem
from fag_common import (
    CUBE_BASEM_bn2s2,
    D_NCHUNKS_bn2s2,
    NZP_ROWS_bn2s2,
    VECTOR_BASEM_bn2s2,
    VECTOR_BASEN_bn2s2,
    VF_LANES_bn2s2,
    VF_LANES_F16_bn2s2,
    plan_run_info_bn2s2,
)
from schedule.deterministic.tnd.fag_deter_tnd_plan import (
    PF_bn2s2,
    PlanSplit_bn2s2,
    plan_i64_bn2s2,
    seq_i64_bn2s2,
)
from memory.fag_mem import (
    BN2_CAST_W_bn2s2,
    PLAN_POST_ROWS_bn2s2,
    PLAN_PRE_ELEMENTS_bn2s2,
    POST_ELEMS_bn2s2,
    PRE_ELEMS_bn2s2,
)
from compute.fag_vector_api import (
    broadcast_sub_mul_cast_vf,
    cast_p_vf,
    dsink_partial_bn2s2_vf,
    dsink_reduce_add_vf,
    fill_zero_f16_vf,
    fill_zero_vf,
    muls_cast_vf,
    plan_cast_only_vf,
    plan_merge_band_mask_vf,
    plan_sink_negate_vf,
    plan_sink_partial_vf,
    plan_sink_reduce_chunk_vf,
    softmax_grad_front_row192_bn2s2_vf,
    softmax_grad_front_row256_acc_vf,
    softmax_grad_front_row256_bn2s2_vf,
    softmax_grad_front_row320_single_vf,
    softmax_grad_front_row_bn2s2_vf,
    softmax_grad_front_single_bn2s2_vf,
)
from schedule.fag_schedule import (
    META_FAG_DKV_ROW_ENDS_bn2s2,
    META_FAG_DKV_ROW_STARTS_bn2s2,
    META_FAG_DQ_ROW_ENDS_bn2s2,
    META_FAG_DQ_ROW_STARTS_bn2s2,
)
from compute.attenmask import (
    compute_offset_for_causal,
    _load_compress_mask,
    muls_sel_simple_softmax_vf,
)

# --- end generated imports ---

# ==================================================================
#  UB -> CV 共享 L1
# ==================================================================


# ==================================================================
#  V1 —— sfmg = rowsum(dy * y)
# ==================================================================


def _vector_load_rows_bn2s2(constInfo, runInfo):
    """Bound y/dy/lse GM reads by the physical Q view, not the used length.

    s1ViewSize is packed T for TND and physical S for BSND/BNSD. Keep the
    tile's physical shape/stride unchanged; rowShift selects the valid rows
    inside this load window, including windows borrowed across TND batches.
    """
    rows = pl.min(
        VECTOR_BASEM_bn2s2, pl.max(constInfo.s1ViewSize - runInfo.loadBase, 0)
    )
    if (
        template == Tmpl.BN2S2
        and deter == 0
        and sink == 0
        and (
            (d_align == 128 and dv_align == 128) or (d_align == 192 and dv_align == 192)
        )
    ):
        # D72/D161 use their real Dv and the existing physical UB pitch.
        # VF consumes only [rowShift, rowShift + halfS1RealSize); retain the
        # borrowed prefix but do not transfer the unused trailing rows.
        # Keep the existing producer/consumer generation on empty subcores.
        rows = pl.min(rows, pl.max(runInfo.rowShift + runInfo.halfS1RealSize, 1))
    return rows


def process_plan_vec1_bn2s2(
    constInfo, runInfo, subIdx, tensor_y, tensor_dy, y_ub, dy_ub, sfmg_ub, rowStride
):
    # D64 plan path: retain the borrowed prefix and skip unused trailing rows.
    if runInfo.halfS1RealSize > 0:
        # D192/Dv128 copies the exact active rows as ASC ProcessVec1 does.
        # Keep RunInfo unchanged: LSE and all other consumers still use its offset.
        loadRows = runInfo.rowShift + runInfo.halfS1RealSize
        loadBase = runInfo.loadBase
        vfRowShift = runInfo.rowShift
        if d_align == 192 and dv_align == 128:
            loadRows = runInfo.halfS1RealSize
            loadBase = loadBase + vfRowShift
            vfRowShift = pl.astype(0, pl.DT_INT64)
        pl.set_validshape(y_ub, [loadRows, constInfo.dvSize])
        pl.set_validshape(dy_ub, [loadRows, constInfo.dvSize])
        pl.load(
            y_ub, tensor_y, [runInfo.idx0Q, loadBase, runInfo.idx2Q, 0], order=[1, 3]
        )
        pl.load(
            dy_ub, tensor_dy, [runInfo.idx0Q, loadBase, runInfo.idx2Q, 0], order=[1, 3]
        )
        softmax_grad_front_row_bn2s2_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            runInfo.halfS1RealSize,
            vfRowShift,
            rowStride,
            constInfo.dvEvenSize,
            constInfo.dvOddSize,
        )


def _sfmg_whole_at_width_bn2s2(
    ci, info, tensor_y, tensor_dy, y_slot, dy_slot, sfmg, row_width, slot_elements
):
    # Retain the slot and mutex selected by the caller. The last VF writes a
    # scalar per valid output row at its existing offset in the macro tile.
    step = slot_elements // row_width
    y_tile = pl.reinterpret(y_slot, shape=[step, row_width])
    dy_tile = pl.reinterpret(dy_slot, shape=[step, row_width])
    remainder = ci.dvSize - (row_width - 128)
    even = (remainder + 1) // 2
    odd = remainder // 2
    for off in pl.range(0, info.halfS1RealSize, step):
        rows = pl.min(step, info.halfS1RealSize - off)
        src_row = info.loadBase + info.rowShift + off
        pl.set_validshape(y_tile, [rows, ci.dvSize])
        pl.set_validshape(dy_tile, [rows, ci.dvSize])
        pl.load(y_tile, tensor_y, [info.idx0Q, src_row, info.idx2Q, 0], order=[1, 3])
        pl.load(dy_tile, tensor_dy, [info.idx0Q, src_row, info.idx2Q, 0], order=[1, 3])
        if row_width == 384:
            softmax_grad_front_row384_vf(
                sfmg, y_tile, dy_tile, rows, 0, row_width, even, odd, off
            )
        else:
            softmax_grad_front_row512_vf(
                sfmg, y_tile, dy_tile, rows, 0, row_width, even, odd, off
            )


def _sfmg_whole_wide_bn2s2(ci, info, y, dy, y_slot, dy_slot, sfmg, slot_elements):
    if ci.dvSize <= 384:
        _sfmg_whole_at_width_bn2s2(
            ci, info, y, dy, y_slot, dy_slot, sfmg, 384, slot_elements
        )
    else:
        _sfmg_whole_at_width_bn2s2(
            ci, info, y, dy, y_slot, dy_slot, sfmg, 512, slot_elements
        )


def process_vec1_bn2s2(
    constInfo,
    runInfo,
    subIdx,
    tensor_y,
    tensor_dy,
    y_ub,
    dy_ub,
    sfmg_ub,
    rowStride,
    exact_192_rows=False,
):
    """softmaxGradFront: sfmg = rowsum(dy * y)。一趟 VECTOR_BASEM 行。"""
    if dv_align >= 512:
        if constInfo.dvSize > 256 and constInfo.dvSize < 512:
            _sfmg_whole_wide_bn2s2(
                constInfo,
                runInfo,
                tensor_y,
                tensor_dy,
                y_ub,
                dy_ub,
                sfmg_ub,
                VECTOR_BASEM_bn2s2 * 256,
            )
        elif dv_align == 512:
            _process_vec1_d512_bn2s2(
                constInfo, runInfo, tensor_y, tensor_dy, y_ub, dy_ub, sfmg_ub, rowStride
            )
        else:
            _process_vec1_d768_bn2s2(
                constInfo, runInfo, tensor_y, tensor_dy, y_ub, dy_ub, sfmg_ub, rowStride
            )
        return
    loadBase = runInfo.loadBase
    vfShift = runInfo.rowShift
    if exact_192_rows:
        # Keep both MTE2 producers on a zero-row subcore. The D192 VF skips
        # its row loop there, while its buffer mutex generation still retires.
        loadBase = pl.min(loadBase + vfShift, constInfo.s1ViewSize - 1)
        loadRows = pl.max(runInfo.halfS1RealSize, 1)
        vfShift = pl.astype(0, pl.DT_INT64)
    else:
        loadRows = _vector_load_rows_bn2s2(constInfo, runInfo)
    pl.set_validshape(y_ub, [loadRows, constInfo.dvSize])
    pl.set_validshape(dy_ub, [loadRows, constInfo.dvSize])
    pl.load(
        y_ub,
        tensor_y,
        [runInfo.idx0Q, loadBase, runInfo.idx2Q, 0],
        order=[1, 3],
    )
    pl.load(
        dy_ub,
        tensor_dy,
        [runInfo.idx0Q, loadBase, runInfo.idx2Q, 0],
        order=[1, 3],
    )
    # 列宽跟 dv_align：256/192 各一档 VF；64 与 128（含 D=192,Dv=128）共用
    # 一次 128-lane load，用 even/odd mask 收真实 Dv。
    if dv_align == 256:
        w0 = pl.min(constInfo.dvSize, 128)
        w1 = pl.max(constInfo.dvSize - 128, 0)
        softmax_grad_front_row256_bn2s2_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            runInfo.halfS1RealSize,
            runInfo.rowShift,
            rowStride,
            (w0 + 1) // 2,
            w0 // 2,
            (w1 + 1) // 2,
            w1 // 2,
        )
    elif dv_align == 192:
        w0 = pl.min(constInfo.dvSize, 128)
        w1 = pl.max(constInfo.dvSize - 128, 0)
        softmax_grad_front_row192_bn2s2_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            runInfo.halfS1RealSize,
            vfShift,
            rowStride,
            (w0 + 1) // 2,
            w0 // 2,
            (w1 + 1) // 2,
            w1 // 2,
        )
    elif (
        d_align == 64
        and dv_align == 64
        and deter == 0
        and layout == 1
        and constInfo.dvSize == 1
    ):
        softmax_grad_front_single_bn2s2_vf(
            sfmg_ub, y_ub, dy_ub, runInfo.halfS1RealSize, runInfo.rowShift, rowStride
        )
    else:
        softmax_grad_front_row_bn2s2_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            runInfo.halfS1RealSize,
            runInfo.rowShift,
            rowStride,
            constInfo.dvEvenSize,
            constInfo.dvOddSize,
        )


def _process_vec1_d768_bn2s2(
    constInfo, runInfo, tensor_y, tensor_dy, y_ub, dy_ub, sfmg_ub, rowStride
):
    """768 档 V1：3 趟 VF256，后两趟累加到同一 sfmg。工作列宽 256。"""
    loadRows = _vector_load_rows_bn2s2(constInfo, runInfo)
    for k in pl.range(0, 3):
        colOff = k * 256
        rem = pl.max(constInfo.dvSize - colOff, 0)
        winW = pl.min(256, rem)
        if rem > 0:
            pl.set_validshape(y_ub, [loadRows, winW])
            pl.set_validshape(dy_ub, [loadRows, winW])
            pl.load(
                y_ub,
                tensor_y,
                [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, colOff],
                order=[1, 3],
            )
            pl.load(
                dy_ub,
                tensor_dy,
                [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, colOff],
                order=[1, 3],
            )
            w0 = pl.min(rem, 128)
            w1 = pl.max(rem - 128, 0)
            even0 = (w0 + 1) // 2
            odd0 = w0 // 2
            even1 = (w1 + 1) // 2
            odd1 = w1 // 2
            if k == 0:
                softmax_grad_front_row256_bn2s2_vf(
                    sfmg_ub,
                    y_ub,
                    dy_ub,
                    runInfo.halfS1RealSize,
                    runInfo.rowShift,
                    rowStride,
                    even0,
                    odd0,
                    even1,
                    odd1,
                )
            else:
                softmax_grad_front_row256_acc_vf(
                    sfmg_ub,
                    y_ub,
                    dy_ub,
                    runInfo.halfS1RealSize,
                    runInfo.rowShift,
                    rowStride,
                    even0,
                    odd0,
                    even1,
                    odd1,
                )


def _process_vec1_d512_bn2s2(
    constInfo, runInfo, tensor_y, tensor_dy, y_ub, dy_ub, sfmg_ub, rowStride
):
    """512/768 档 V1：按 256 列窗多趟 VF256（512=2 趟、768=3 趟），后趟累加到
    同一 sfmg。工作列宽 256，趟数 = D_NCHUNKS // 2（块数每窗 2 块）。"""
    loadRows = _vector_load_rows_bn2s2(constInfo, runInfo)
    nWin = 2
    for k in pl.range(0, nWin):
        colOff = k * 256
        rem = pl.max(constInfo.dvSize - colOff, 0)
        winW = pl.min(256, rem)
        if rem > 0:
            pl.set_validshape(y_ub, [loadRows, winW])
            pl.set_validshape(dy_ub, [loadRows, winW])
            pl.load(
                y_ub,
                tensor_y,
                [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, colOff],
                order=[1, 3],
            )
            pl.load(
                dy_ub,
                tensor_dy,
                [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, colOff],
                order=[1, 3],
            )
            w0 = pl.min(rem, 128)
            w1 = pl.max(rem - 128, 0)
            even0 = (w0 + 1) // 2
            odd0 = w0 // 2
            even1 = (w1 + 1) // 2
            odd1 = w1 // 2
            if k == 0:
                softmax_grad_front_row256_bn2s2_vf(
                    sfmg_ub,
                    y_ub,
                    dy_ub,
                    runInfo.halfS1RealSize,
                    runInfo.rowShift,
                    rowStride,
                    even0,
                    odd0,
                    even1,
                    odd1,
                )
            else:
                softmax_grad_front_row256_acc_vf(
                    sfmg_ub,
                    y_ub,
                    dy_ub,
                    runInfo.halfS1RealSize,
                    runInfo.rowShift,
                    rowStride,
                    even0,
                    odd0,
                    even1,
                    odd1,
                )


# ==================================================================
#  V2 —— P = simpleSoftmax(S * scale, lse) [+ atten mask]
# ==================================================================


# ==================================================================
#  V3 —— dS = (dP - sfmg) * P，cast + ND2NZ 落 L1（mm4 读同址 ZN 视图）
# ==================================================================


# ==================================================================
#  V4 —— P cast + ND2NZ 落 L1（mm5 读同址 ZN 视图）
# ==================================================================


# ==================================================================
#  Pre —— dq/dk/dv 累加区清零
# ==================================================================


def process_sink_partial_bn2s2(
    constInfo,
    runInfo,
    tensor_slots,
    aiv_idx,
    p_ub,
    dp_ub,
    lse_ub,
    sink_part_ub,
    sink_val,
):
    """本 s2 块的 dsink 贡献，原子加到槽位区本 AIV 行的 headQ 列。"""
    rows = runInfo.halfS1RealSize
    if rows <= 0:
        return
    s2Real = runInfo.s2RealSize
    if s2Real <= 0:
        return
    headQ = runInfo.n2oIdx * constInfo.gSize + runInfo.goIdx
    loopTimes = (s2Real + VF_LANES_bn2s2 - 1) // VF_LANES_bn2s2 - 1
    tailCols = s2Real - loopTimes * VF_LANES_bn2s2
    dsink_partial_bn2s2_vf(
        sink_part_ub,
        p_ub,
        dp_ub,
        lse_ub,
        sink_val,
        rows,
        loopTimes,
        tailCols,
        runInfo.rowShift,
    )
    pl.set_validshape(sink_part_ub, [1, 1])
    pl.store(
        tensor_slots,
        sink_part_ub,
        [0, aiv_idx * constInfo.n1Size + headQ],
        atomic=pl.AtomicType.AtomicAdd,
    )
    pl.set_validshape(sink_part_ub, [1, VECTOR_BASEN_bn2s2])


def process_sink_reduce_bn2s2(
    constInfo, tensor_slots, tensor_dsink, acc_ub, row_ub, aiv_count
):
    """Batch-load four independent rows, then add them in original AIV order.

    row_ub is the same four-buffer group consumed by the public reducer.
    Only the final load/barrier schedule is shared; the BN2S2 partial-sum
    VF and its floating-point reduction order remain unchanged.
    """
    process_sink_reduce(
        constInfo, tensor_slots, tensor_dsink, acc_ub, row_ub, aiv_count
    )


def process_sink_zero_bn2s2(tensor_slots, constInfo, zt, aiv_idx):
    """清本 AIV 的 dsink 槽位行，供主循环 atomicAdd。"""
    fill_zero_vf(zt, (constInfo.n1Size + VF_LANES_bn2s2 - 1) // VF_LANES_bn2s2)
    process_pre_bn2s2(tensor_slots, zt, aiv_idx * constInfo.n1Size, constInfo.n1Size)


def process_pre_bn2s2(tensor_ws, zt, elemStart, nElems):
    process_pre_chunked(tensor_ws, zt, elemStart, nElems, PRE_ELEMS_bn2s2[d_align])


def store_isolated_zero_bn2s2(out, f16, scale, is_dv, begin, count, chunk):
    if count > 0:
        tile = f16.next()
        pl.set_validshape(tile, [1, chunk])
        vectors = (pl.min(count, chunk) + 127) // 128
        if is_dv:
            fill_zero_f16_vf(tile, vectors)
        else:
            fill_scaled_zero_bn2s2_vf(tile, scale, vectors)
        process_pre_chunked(out, tile, begin, count, chunk)


def process_dense_dq_rows_bn2s2(
    ci, cuq, cuk, ws, out, start, end, zero, f32, f16, post
):
    # Clip only physically Q-only intervals against this worker's metadata rows.
    # Uninterrupted active rows keep the original bulk DMA and accumulation.
    cursor = start
    one64 = pl.const(1, pl.DT_INT64)
    for b in pl.range(0, ci.bSize):
        if pl.getval(cuk, b) == pl.getval(cuk, b + 1):
            q0 = pl.getval(cuq, b) * one64 * ci.n1Size
            q1 = pl.getval(cuq, b + 1) * one64 * ci.n1Size
            lo = pl.max(start, q0)
            hi = pl.min(end, q1)
            if hi > lo:
                if post:
                    process_post(
                        ws,
                        out,
                        f32,
                        f16,
                        ci.scaleValue,
                        cursor * ci.dSize,
                        (lo - cursor) * ci.dSize,
                    )
                    store_isolated_zero_bn2s2(
                        out,
                        f16,
                        ci.scaleValue,
                        False,
                        lo * ci.dSize,
                        (hi - lo) * ci.dSize,
                        POST_ELEMS_bn2s2[d_align],
                    )
                else:
                    process_pre_bn2s2(
                        ws, zero, cursor * ci.dSize, (lo - cursor) * ci.dSize
                    )
                cursor = hi
    if post:
        process_post(
            ws,
            out,
            f32,
            f16,
            ci.scaleValue,
            cursor * ci.dSize,
            (end - cursor) * ci.dSize,
        )
    else:
        process_pre_bn2s2(ws, zero, cursor * ci.dSize, (end - cursor) * ci.dSize)


def pre_dense_tnd_bn2s2(ci, cuq, cuk, meta, fagOff, dq_ws, dq, dk, dv, zero, zero16):
    aiv = pl.get_block_idx()
    one64 = pl.const(1, pl.DT_INT64)
    start = pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS_bn2s2 + aiv) * one64
    end = pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS_bn2s2 + aiv) * one64
    capacity = PRE_ELEMS_bn2s2[d_align]
    pl.set_validshape(zero, [1, capacity])
    fill_zero_vf(zero, (pl.min(capacity, (end - start) * ci.dSize) + 63) // 64)
    process_dense_dq_rows_bn2s2(
        ci, cuq, cuk, dq_ws, dq, start, end, zero, zero, zero, False
    )
    # AIVs divide EACH orphan KV span by physical head rows. Both dimensions
    # are 16-element aligned at admission, including worker boundary stores.
    filled = pl.const(0, pl.DT_INT64)
    for b in pl.range(0, ci.bSize):
        if pl.getval(cuq, b) == pl.getval(cuq, b + 1):
            first = pl.getval(cuk, b) * one64 * ci.n2Size
            last = pl.getval(cuk, b + 1) * one64 * ci.n2Size
            rows = last - first
            lo = first + rows * aiv // ci.aivCount
            hi = first + rows * (aiv + 1) // ci.aivCount
            if hi > lo:
                needed = pl.min(capacity, (hi - lo) * pl.max(ci.dSize, ci.dvSize))
                if needed > filled:
                    pl.set_validshape(zero16, [1, capacity])
                    fill_zero_f16_vf(zero16, (needed + 127) // 128)
                    filled = needed
                process_pre_bn2s2(dk, zero16, lo * ci.dSize, (hi - lo) * ci.dSize)
                process_pre_bn2s2(dv, zero16, lo * ci.dvSize, (hi - lo) * ci.dvSize)


def process_pre_from_meta_bn2s2(
    meta, fagOff, tensor_dq, tensor_dk, tensor_dv, dQ, dV, zt, clear_kv
):
    process_pre_from_meta_chunked(
        meta,
        fagOff,
        tensor_dq,
        tensor_dk,
        tensor_dv,
        dQ,
        dV,
        zt,
        PRE_ELEMS_bn2s2[d_align],
        clear_kv,
    )


def process_pre_kv_out_zero_bn2s2(meta, fagOff, tensor_dk, tensor_dv, dQ, dV, zt, need):
    """BN2S2 直出档（d_align<=128）：按 metadata 的 dkv 行区间清 16-bit dK/dV 输出。

    直出路径下无效列/零长序列的 dK/dV 行不会被主循环写，必须在 pre 清零
    兜底（对齐老 FAG pre 的 hasInvalidCol/sValueZeroUnderTND 分支）。dQ 输出
    由 post 全量 cast 覆盖（ws 已清零），无需清。zt 需为 16-bit zero UB。
    ``need`` 为 0 时元素数折叠为 0、清零循环空转；定长场景仅
    零 Q 任务（s1==0）与 seqused 变长需要兜底，主路径传 0 保住整份 kv 输出
    的写带宽。
    """
    vBlockIdx = pl.get_block_idx()
    one64 = pl.const(1, pl.DT_INT64)
    rowStart = (
        pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS_bn2s2 + vBlockIdx) * one64
    )
    rowEnd = pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS_bn2s2 + vBlockIdx) * one64
    dkNElems = (rowEnd - rowStart) * dQ * need
    dvNElems = (rowEnd - rowStart) * dV * need
    preElems = pl.min(PRE_ELEMS_bn2s2[d_align], pl.max(dkNElems, dvNElems))
    fill_zero_f16_vf(zt, (preElems + VF_LANES_F16_bn2s2 - 1) // VF_LANES_F16_bn2s2)
    process_pre_bn2s2(tensor_dk, zt, rowStart * dQ, dkNElems)
    process_pre_bn2s2(tensor_dv, zt, rowStart * dV, dvNElems)


# ==================================================================
#  Post —— dQ/dK/dV 的 scale + cast
# ==================================================================


# ==================================================================
#  BN2 模板的 vector 阶段
# ==================================================================


def _store_splitm_dkv_bn2s2(tensor_out, dst_ub, runInfo, subIdx, out_w):
    """按 runInfo.firstHalfS2RealSize 把本子核的行写出。

    切点在 set_run_info 里已经算好。vec0 从 keyOffset 起写这么多行，
    vec1 的 GM 行偏移就是这份，行数是 s2RealSize 减去它。UB 行都从 0 排起。
    """
    first = runInfo.firstHalfS2RealSize
    row_off = first * subIdx
    if subIdx == 0:
        rows = first
    else:
        rows = pl.max(runInfo.s2RealSize - first, 0)
    if rows > 0:
        seq_off = runInfo.keyOffset + row_off
        pl.set_validshape(dst_ub, [rows, out_w])
        pl.store(
            tensor_out,
            dst_ub,
            [runInfo.idx0KV, seq_off, runInfo.idx2KV, 0],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicNone,
        )
        pl.set_validshape(dst_ub, [VECTOR_BASEM_bn2s2, BN2_CAST_W_bn2s2])


def process_muls_cast_dk_bn2_bn2s2(
    tensor_dk, src_ub, dst_ub, constInfo, runInfo, subIdx
):
    """v6：mm4 的 fp32 UB → muls(scale) → cast 16-bit → dK GM。

    BN2 与 BN2S2 共用。行区间用 runInfo.firstHalfS2RealSize。
    """
    pl.set_validshape(src_ub, [VECTOR_BASEM_bn2s2, BN2_CAST_W_bn2s2])
    pl.set_validshape(dst_ub, [VECTOR_BASEM_bn2s2, BN2_CAST_W_bn2s2])
    muls_cast_vf(
        dst_ub,
        src_ub,
        constInfo.scaleValue,
        VECTOR_BASEM_bn2s2 * BN2_CAST_W_bn2s2 // VECTOR_BASEN_bn2s2,
    )
    _store_splitm_dkv_bn2s2(tensor_dk, dst_ub, runInfo, subIdx, constInfo.dSize)


# （曾有 _cast_ws_col / process_muls_cast_dkv_from_ws 做 d>=192 的"列末直出"
# 消费——从 GM ws 读 fp32 经 Muls/Cast 直写 16-bit GM。实测跨核事件不保证
# FIX 写 GM 对对端 MTE2 读的可见性（写穿透竞态导致 dK 前 64 列错值），已
# 随消费端撤回 post 全三段整体删除，勿再引入；老仓 DqkvMulsAndCastFromGM
# 的同款算术跑在 post/sync_all 之后，两者时机不同。）


# ==================================================================
#  BN2S2 TND metadata 确定性：V2/V3/V4、列写回与 pre/post
# ==================================================================
# 屏障由 kernel 编排；此处保持原搬运、逐列写回和 sink 归约顺序。


def plan_copy_mask_bn2s2(h, info, sub, tensor_mask, mask, mask_pre):
    if info.halfS1RealSize > 0:
        vec_off = sub * info.firstHalfS1RealSize
        local_delta = (info.s1oIdx - info.s2oIdx) * 128
        skew = info.batchS2Len - info.batchS1Len
        if mask_mode == 3:
            offset = compute_offset_for_causal(local_delta + skew, vec_off)
            _load_compress_mask(
                tensor_mask, mask, offset, info.halfS1RealSize, info.s2RealSize
            )
        else:
            pre = h.tokenQ - skew
            post = h.tokenKv + skew
            next_delta = local_delta + post
            pre_delta = local_delta - pre - 1
            offset = compute_offset_for_causal(next_delta, vec_off)
            _load_compress_mask(
                tensor_mask, mask, offset, info.halfS1RealSize, info.s2RealSize
            )
            if pre_delta + 129 > 0:
                pre_offset = compute_offset_for_causal(pre_delta, vec_off)
                _load_compress_mask(
                    tensor_mask,
                    mask_pre,
                    pre_offset,
                    info.halfS1RealSize,
                    info.s2RealSize,
                )
                plan_merge_band_mask_vf(mask, mask_pre, info.halfS1RealSize)


def plan_load_lse_bn2s2(h, info, tensor_lse, lse):
    load_rows = pl.min(64, pl.max(h.totalQ - info.loadBase, 0))
    load_base = info.loadBase
    if d_align == 192 and dv_align == 128 and sink == 0:
        # Keep one harmless in-bounds row for the zero-work subcore so its
        # MTE2 producer/VF consumer mutex generation still pairs normally.
        load_rows = pl.max(info.halfS1RealSize, 1)
        load_base = pl.min(info.loadBase + info.rowShift, h.totalQ - 1)
    pl.set_validshape(lse, [load_rows, 1])
    pl.load(lse, tensor_lse, [info.n2oIdx, load_base], order=[1, 0])


def plan_vec2_bn2s2(ci, h, info, sub, mm2, lse, tensor_mask, mask, mask_pre):
    if mask_mode != 0:
        plan_copy_mask_bn2s2(h, info, sub, tensor_mask, mask, mask_pre)
    lse_shift = info.rowShift
    if d_align == 192 and dv_align == 128 and sink == 0:
        lse_shift = pl.astype(0, pl.DT_INT64)
    muls_sel_simple_softmax_vf(
        mm2, lse, mask, ci.scaleValue, info.halfS1RealSize, lse_shift
    )


def plan_insert_tail_bn2s2(info, sub, l1, nzp):
    if info.halfS1RealSize > 0:
        copy_ub2l1(l1, nzp, sub * info.firstHalfS1RealSize, info.halfS1RealSize)


def plan_vec3_bn2s2(ci, info, sub, ds_l1, mm1, mm2, sfmg, nzp):
    broadcast_sub_mul_cast_vf(nzp, mm1, mm2, sfmg, ci.scaleValue, info.halfS1RealSize)
    plan_insert_tail_bn2s2(info, sub, ds_l1, nzp)


def plan_vec4_bn2s2(info, sub, p_l1, mm2, nzp):
    cast_p_vf(nzp, mm2, info.halfS1RealSize)
    plan_insert_tail_bn2s2(info, sub, p_l1, nzp)


def plan_sink_task_bn2s2(info, tensor_sinks, tensor_partial, p, dp, lse, partial):
    if info.halfS1RealSize > 0:
        sink_value = pl.getval(tensor_sinks, info.n2oIdx)
        loops = (info.s2RealSize + 63) // 64 - 1
        tail = info.s2RealSize - loops * 64
        plan_sink_partial_vf(
            partial,
            p,
            dp,
            lse,
            sink_value,
            info.halfS1RealSize,
            loops,
            tail,
            info.rowShift,
        )
        pl.set_validshape(partial, [1, 1])
        pl.store(
            tensor_partial,
            partial,
            [0, info.sinkOffset],
            atomic=pl.AtomicType.AtomicNone,
        )
        pl.set_validshape(partial, [1, 128])


def plan_column_chunked_from_gm_bn2s2(
    ws_ptr,
    base,
    stride,
    tensor_out,
    src,
    dst,
    key_token,
    head,
    rows,
    sub_idx,
    dim,
    scale,
    is_dv,
    v1w,
):
    """Copy disjoint ASC ceil(real/2) rows; base/stride are FP32 element units.

    Works for per-core scratch and the original interleaved special slot.
    Chunking the elementwise conversion changes no FP32 accumulation order.
    D tails load/store only valid columns; physical UB row width stays 128.
    """
    first = (rows + 1) // 2
    half_rows = pl.min(first, rows - sub_idx * first)
    source = pl.make_tensor(
        pl.addptr(ws_ptr, base), [128, dim], [stride, 1], dtype=pl.DT_FP32
    )
    if d_align == 192 and dv_align == 192:
        # Entire D in each row: no second 128-column DMA/conversion pass.
        # Physical stride remains 192; invalid columns never reach GM.
        for block in pl.range(0, (half_rows + 31) // 32):
            row = block * 32
            valid_rows = pl.min(32, half_rows - row)
            pl.set_validshape(src, [valid_rows, dim])
            pl.load(src, source, [sub_idx * first + row, 0])
            pl.set_validshape(src, [32, 192])
            pl.set_validshape(dst, [32, 192])
            segments = (valid_rows * 192 + 127) // 128
            if is_dv:
                plan_cast_only_vf(dst, src, segments)
            else:
                muls_cast_vf(dst, src, scale, segments)
            pl.set_validshape(dst, [valid_rows, dim])
            pl.store(
                tensor_out,
                dst,
                [0, key_token + sub_idx * first + row, head, 0],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicNone,
            )
    else:
        for block in pl.range(
            0, (half_rows + PLAN_POST_ROWS_bn2s2[v1w] - 1) // PLAN_POST_ROWS_bn2s2[v1w]
        ):
            row = block * PLAN_POST_ROWS_bn2s2[v1w]
            valid_rows = pl.min(PLAN_POST_ROWS_bn2s2[v1w], half_rows - row)
            for chunk in pl.range(0, (dim + 127) // 128):
                col = chunk * 128
                width = pl.min(128, dim - col)
                pl.set_validshape(src, [valid_rows, width])
                pl.load(src, source, [sub_idx * first + row, col])
                pl.set_validshape(src, [PLAN_POST_ROWS_bn2s2[v1w], 128])
                pl.set_validshape(dst, [64, 128])
                if is_dv:
                    plan_cast_only_vf(dst, src, PLAN_POST_ROWS_bn2s2[v1w])
                else:
                    muls_cast_vf(dst, src, scale, PLAN_POST_ROWS_bn2s2[v1w])
                pl.set_validshape(dst, [valid_rows, width])
                pl.store(
                    tensor_out,
                    dst,
                    [0, key_token + sub_idx * first + row, head, col],
                    order=[1, 3],
                    atomic=pl.AtomicType.AtomicNone,
                )


def plan_column_full320_from_gm_bn2s2(
    ws_ptr,
    base,
    stride,
    tensor_out,
    src,
    dst,
    key_token,
    head,
    rows,
    sub_idx,
    dim,
    scale,
    is_dv,
):
    # The 768 family has 32768 bytes of ordinary column input and at least
    # 16384 bytes of cast storage. Reinterpret only 24*320 elements at the
    # original address; original mutex ownership is retained by the views.
    whole_src = pl.reinterpret(src, shape=[24, 320])
    whole_dst = pl.reinterpret(dst, shape=[24, 320])
    first = (rows + 1) // 2
    half_rows = pl.min(first, rows - sub_idx * first)
    source = pl.make_tensor(
        pl.addptr(ws_ptr, base), [128, dim], [stride, 1], dtype=pl.DT_FP32
    )
    for block in pl.range(0, (half_rows + 23) // 24):
        row = block * 24
        valid_rows = pl.min(24, half_rows - row)
        pl.set_validshape(whole_src, [valid_rows, dim])
        pl.load(whole_src, source, [sub_idx * first + row, 0])
        pl.set_validshape(whole_src, [24, 320])
        pl.set_validshape(whole_dst, [24, 320])
        segments = (valid_rows * 320 + 127) // 128
        if is_dv:
            plan_cast_only_vf(whole_dst, whole_src, segments)
        else:
            muls_cast_vf(whole_dst, whole_src, scale, segments)
        pl.set_validshape(whole_dst, [valid_rows, dim])
        pl.store(
            tensor_out,
            whole_dst,
            [0, key_token + sub_idx * first + row, head, 0],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicNone,
        )


def normal_wide_slot_read_bn2s2(
    ws,
    base,
    stride,
    output,
    src,
    dst,
    token,
    head,
    rows,
    sub,
    dim,
    scale,
    is_dv,
    sw_active,
    padded_384=False,
):
    if d_align == 256 and sw_active == 1:
        plan_column_full_d_from_gm_bn2s2(
            ws,
            base,
            stride,
            output,
            src,
            dst,
            token,
            head,
            rows,
            sub,
            dim,
            scale,
            is_dv,
            32,
            256,
        )
    elif d_align > 256 and dim <= 384:
        if padded_384 and stride == 384:
            plan_column_full_d_from_gm_bn2s2(
                ws,
                base,
                stride,
                output,
                src,
                dst,
                token,
                head,
                rows,
                sub,
                dim,
                scale,
                is_dv,
                21,
                384,
                True,
            )
        else:
            plan_column_full_d_from_gm_bn2s2(
                ws,
                base,
                stride,
                output,
                src,
                dst,
                token,
                head,
                rows,
                sub,
                dim,
                scale,
                is_dv,
                21,
                384,
            )
    elif d_align > 256 and sw_active == 1 and dim <= 512:
        plan_column_full_d_from_gm_bn2s2(
            ws,
            base,
            stride,
            output,
            src,
            dst,
            token,
            head,
            rows,
            sub,
            dim,
            scale,
            is_dv,
            16,
            512,
        )
    else:
        plan_column_chunked_from_gm_bn2s2(
            ws,
            base,
            stride,
            output,
            src,
            dst,
            token,
            head,
            rows,
            sub,
            dim,
            scale,
            is_dv,
            256,
        )


def plan_column_full_d_from_gm_bn2s2(
    ws_ptr,
    base,
    stride,
    tensor_out,
    src,
    dst,
    key_token,
    head,
    rows,
    sub_idx,
    dim,
    scale,
    is_dv,
    physical_rows,
    physical_width,
    contiguous_slot=False,
):
    # ASC ceil(real/2) ownership, full-D transfer, balanced rows per iteration.
    # The post phase has >=32KiB/16KiB; main has a separate32KiB input arena.
    if contiguous_slot:
        # Each row in this private slot owns its padding; cast and scale are lane-wise.
        whole_src = pl.reinterpret(src, shape=[1, physical_rows * physical_width])
        source = pl.make_tensor(
            pl.addptr(ws_ptr, base),
            [1, 128 * physical_width],
            [128 * physical_width, 1],
            dtype=pl.DT_FP32,
        )
    else:
        whole_src = pl.reinterpret(src, shape=[physical_rows, physical_width])
        source = pl.make_tensor(
            pl.addptr(ws_ptr, base), [128, dim], [stride, 1], dtype=pl.DT_FP32
        )
    whole_dst = pl.reinterpret(dst, shape=[physical_rows, physical_width])
    first = (rows + 1) // 2
    half_rows = pl.min(first, rows - sub_idx * first)
    loops = (half_rows + physical_rows - 1) // physical_rows
    if loops > 0:
        block_rows = (half_rows + loops - 1) // loops
        for block in pl.range(0, loops):
            row = block * block_rows
            valid_rows = pl.min(block_rows, half_rows - row)
            if contiguous_slot:
                pl.set_validshape(whole_src, [1, valid_rows * physical_width])
                pl.load(
                    whole_src, source, [0, (sub_idx * first + row) * physical_width]
                )
            else:
                pl.set_validshape(whole_src, [valid_rows, dim])
                pl.load(whole_src, source, [sub_idx * first + row, 0])
                pl.set_validshape(whole_src, [physical_rows, physical_width])
            pl.set_validshape(whole_dst, [physical_rows, physical_width])
            segments = (valid_rows * physical_width + 127) // 128
            if is_dv:
                plan_cast_only_vf(whole_dst, whole_src, segments)
            else:
                muls_cast_vf(whole_dst, whole_src, scale, segments)
            pl.set_validshape(whole_dst, [valid_rows, dim])
            pl.store(
                tensor_out,
                whole_dst,
                [0, key_token + sub_idx * first + row, head, 0],
                order=[1, 3],
                atomic=pl.AtomicType.AtomicNone,
            )


def plan_column_from_gm_bn2s2(
    ws_ptr,
    base,
    stride,
    tensor_out,
    src,
    dst,
    key_token,
    head,
    rows,
    sub_idx,
    dim,
    scale,
    is_dv,
    v1w,
):
    if d_align == 192 and dv_align == 128:
        if is_dv:
            plan_column_full_d_from_gm_bn2s2(
                ws_ptr,
                base,
                stride,
                tensor_out,
                src,
                dst,
                key_token,
                head,
                rows,
                sub_idx,
                dim,
                scale,
                is_dv,
                64,
                128,
            )
        else:
            plan_column_full_d_from_gm_bn2s2(
                ws_ptr,
                base,
                stride,
                tensor_out,
                src,
                dst,
                key_token,
                head,
                rows,
                sub_idx,
                dim,
                scale,
                is_dv,
                42,
                192,
            )
    elif d_align == 768:
        if dim <= 320:
            plan_column_full320_from_gm_bn2s2(
                ws_ptr,
                base,
                stride,
                tensor_out,
                src,
                dst,
                key_token,
                head,
                rows,
                sub_idx,
                dim,
                scale,
                is_dv,
            )
        else:
            plan_column_chunked_from_gm_bn2s2(
                ws_ptr,
                base,
                stride,
                tensor_out,
                src,
                dst,
                key_token,
                head,
                rows,
                sub_idx,
                dim,
                scale,
                is_dv,
                v1w,
            )
    else:
        plan_column_chunked_from_gm_bn2s2(
            ws_ptr,
            base,
            stride,
            tensor_out,
            src,
            dst,
            key_token,
            head,
            rows,
            sub_idx,
            dim,
            scale,
            is_dv,
            v1w,
        )


def plan_ordinary_column_bn2s2(
    ci, policy, info, ws_ptr, tensor_out, src, dst, sub_idx, is_dv, v1w
):
    base = policy.dkBase + policy.core * policy.scratchStride
    dim = ci.dSize
    stride = policy.dkStride
    if is_dv:
        base = policy.dvBase + policy.core * policy.scratchStride
        dim = ci.dvSize
        stride = policy.dvStride
    plan_column_from_gm_bn2s2(
        ws_ptr,
        base,
        stride,
        tensor_out,
        src,
        dst,
        info.keyOffset,
        info.idx2KV,
        info.s2RealSize,
        sub_idx,
        dim,
        ci.scaleValue,
        is_dv,
        v1w,
    )


def plan_special_column_post_bn2s2(
    ci, policy, saved, ws_ptr, tensor_out, src, dst, sub_idx, is_dv, v1w
):
    # saved comes from a retired owner task or final-table recovery after sync.
    # Non-owners keep rows=0; sparse-empty owners can still flush contributions
    # submitted by other cores into their original special workspace slot.
    base = policy.specialBase + policy.core * 2 * 128 * policy.align
    dim = ci.dSize
    stride = (dim + 15) // 16 * 16
    if is_dv:
        base = base + 128 * policy.align
        dim = ci.dvSize
        stride = (dim + 15) // 16 * 16
    plan_column_from_gm_bn2s2(
        ws_ptr,
        base,
        stride,
        tensor_out,
        src,
        dst,
        saved.keyToken,
        saved.head,
        saved.rows,
        sub_idx,
        dim,
        ci.scaleValue,
        is_dv,
        v1w,
    )


def plan_direct_dk_bn2s2(ci, info, tensor_out, mm2, dst, sub_idx):
    pl.set_validshape(mm2, [64, 128])
    pl.set_validshape(dst, [64, 128])
    muls_cast_vf(dst, mm2, ci.scaleValue, 64)
    if info.halfS2RealSize > 0:
        pl.set_validshape(dst, [info.halfS2RealSize, ci.dSize])
        pl.store(
            tensor_out,
            dst,
            [0, info.keyOffset + sub_idx * info.firstHalfS2RealSize, info.idx2KV, 0],
            order=[1, 3],
            atomic=pl.AtomicType.AtomicNone,
        )


def plan_pre_range_bn2s2(words, factor_field, total_field, elements, aiv):
    factor = plan_i64_bn2s2(words, factor_field)
    begin = aiv * factor
    count = pl.astype(0, pl.DT_INT64)
    if aiv < plan_i64_bn2s2(words, total_field):
        count = pl.max(0, pl.min(factor, elements - begin))
    return begin, count


def plan_isolated_prefix_bn2s2(h, cuq, cuk):
    # All physical rows preceding the first batch with a nonempty rectangle
    # have no GS writer. Subsequent batches retain the original pre/post.
    q = pl.const(0, pl.DT_INT64)
    k = pl.const(0, pl.DT_INT64)
    b = pl.const(0, pl.DT_INT64)
    if h.logicalSplit == PlanSplit_bn2s2.GS and h.dimQ % 16 == 0 and h.dimV % 16 == 0:
        if mask_mode == 0 and d_align == 192 and dv_align == 192 and h.fixedSq > 0:
            return h.fixedQBase * h.heads, h.fixedKvBase * h.heads
        while b < h.batch:
            next_q = pl.getval(cuq, b + 1) * pl.const(1, pl.DT_INT64)
            next_k = pl.getval(cuk, b + 1) * pl.const(1, pl.DT_INT64)
            if q == next_q or k == next_k:
                q = next_q
                k = next_k
                b = b + 1
            else:
                b = h.batch
    return q * h.heads, k * h.heads


def plan_clip_prefix_bn2s2(begin, count, prefix):
    skipped = pl.min(count, pl.max(0, prefix - begin))
    return begin + skipped, count - skipped


def plan_clear_bn2s2(tensor, zero, begin, count):
    for part in pl.range(
        0,
        (count + mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg] - 1)
        // mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg],
    ):
        offset = part * mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg]
        size = pl.min(mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg], count - offset)
        pl.set_validshape(zero, [1, size])
        pl.store(tensor, zero, [0, begin + offset], atomic=pl.AtomicType.AtomicNone)


def plan_clear_pre_bn2s2(words, h, aiv, gm, zero, zero16):
    # Factors are ASC pre worker partitions, not metadata's old row ranges.
    q_begin, q_count = plan_pre_range_bn2s2(
        words,
        PF_bn2s2.qPreBlockFactor,
        PF_bn2s2.qPreBlockTotal,
        plan_i64_bn2s2(words, PF_bn2s2.qPostBlockTotal),
        aiv,
    )
    k_begin, k_count = plan_pre_range_bn2s2(
        words,
        PF_bn2s2.kPreBlockFactor,
        PF_bn2s2.kPreBlockTotal,
        plan_i64_bn2s2(words, PF_bn2s2.kPostBlockTotal),
        aiv,
    )
    v_begin, v_count = plan_pre_range_bn2s2(
        words,
        PF_bn2s2.vPreBlockFactor,
        PF_bn2s2.vPreBlockTotal,
        plan_i64_bn2s2(words, PF_bn2s2.vPostBlockTotal),
        aiv,
    )
    if sink != 0 and (d_align == 192 or d_align == 256):
        q_begin, q_count = plan_clip_prefix_bn2s2(q_begin, q_count, gm.qPrefix)
        k_begin, k_count = plan_clip_prefix_bn2s2(k_begin, k_count, gm.kPrefix)
        v_begin, v_count = plan_clip_prefix_bn2s2(v_begin, v_count, gm.vPrefix)
    if sink != 0:
        # ASC pads the element count by 512 here (workspace aligns bytes).
        # The excess reaches only the still-zero/uninitialized special region
        # or the 20 MiB tail. Preserve it; validate against whole user capacity.
        sink_elements = h.heads * plan_i64_bn2s2(words, PF_bn2s2.sinkReduceAxis)
        padded = (sink_elements + 512) // 512 * 512
        begin, count = plan_pre_range_bn2s2(
            words, PF_bn2s2.sinkPreBlockFactor, PF_bn2s2.sinkPreBlockTotal, padded, aiv
        )
    capacity = mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg]
    clear_elems = pl.astype(capacity, pl.DT_INT64)
    if h.logicalSplit == PlanSplit_bn2s2.GS:
        longest = pl.max(q_count, pl.max(k_count, v_count))
        if sink != 0:
            longest = pl.max(longest, count)
        clear_elems = pl.min(capacity, longest)
    pl.set_validshape(zero, [1, capacity])
    if clear_elems > 0:
        fill_zero_vf(zero, (clear_elems + 63) // 64)
    plan_clear_bn2s2(gm.dqWs, zero, q_begin, q_count)
    if h.logicalSplit == PlanSplit_bn2s2.GS:
        plan_clear_bn2s2(gm.dkWs, zero, k_begin, k_count)
        plan_clear_bn2s2(gm.dvWs, zero, v_begin, v_count)
    if sink != 0:
        plan_clear_bn2s2(gm.sinkPre, zero, begin, count)
    if h.logicalSplit == PlanSplit_bn2s2.BN2S2:
        special_size = 128 * plan_i64_bn2s2(words, PF_bn2s2.alignQ)
        plan_clear_bn2s2(gm.special, zero, aiv * special_size, special_size)
        if (
            plan_i64_bn2s2(words, PF_bn2s2.hasZeroSequence) != 0
            or plan_i64_bn2s2(words, PF_bn2s2.sValueZeroUnderTnd) != 0
            or plan_i64_bn2s2(words, PF_bn2s2.invalidCol) != 0
        ):
            pl.set_validshape(zero16, [1, mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg]])
            fill_zero_f16_vf(zero16, mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg] // 128)
            plan_clear_bn2s2(gm.dkOut, zero16, k_begin, k_count)
            plan_clear_bn2s2(gm.dvOut, zero16, v_begin, v_count)


def plan_post_range_bn2s2(words, factor_field, total_field, aiv):
    block = plan_i64_bn2s2(words, PF_bn2s2.postBaseElements)
    span = plan_i64_bn2s2(words, factor_field) * block
    begin = aiv * span
    end = pl.min(
        begin + span,
        plan_i64_bn2s2(words, total_field),
    )
    return begin, end


def plan_post_tile_bn2s2(tensor_ws, tensor_out, f32, f16, scale, is_dv, offset, size):
    src = f32.next()
    dst = f16.next()
    pl.set_validshape(src, [1, size])
    pl.set_validshape(dst, [1, size])
    pl.load(src, tensor_ws, [0, offset])
    if is_dv:
        plan_cast_only_vf(dst, src, (size + 127) // 128)
    else:
        muls_cast_vf(dst, src, scale, (size + 127) // 128)
    pl.store(tensor_out, dst, [0, offset], atomic=pl.AtomicType.AtomicNone)


def plan_post_tensor_bn2s2(
    words,
    aiv,
    factor_field,
    total_field,
    tensor_ws,
    tensor_out,
    f32,
    f16,
    scale,
    is_dv,
    prefix,
):
    begin, end = plan_post_range_bn2s2(words, factor_field, total_field, aiv)
    if sink != 0 and (d_align == 192 or d_align == 256):
        if begin < prefix and begin < end:
            mid = pl.min(prefix, end)
            store_isolated_zero_bn2s2(
                tensor_out, f16, scale, is_dv, begin, mid - begin, 16384
            )
            begin = mid
    # ASC REGBASE_POST_BASE, preserving post workers and ping->pong order.
    if end > begin and end - begin <= 16384:
        plan_post_tile_bn2s2(
            tensor_ws, tensor_out, f32, f16, scale, is_dv, begin, end - begin
        )
    else:
        for offset in pl.range(begin, end, 16384):
            size = pl.min(16384, end - offset)
            plan_post_tile_bn2s2(
                tensor_ws, tensor_out, f32, f16, scale, is_dv, offset, size
            )


def plan_post_sink_bn2s2(words, aiv, tensor_partial, tensor_out, source, total):
    factor = plan_i64_bn2s2(words, PF_bn2s2.sinkPostBlockFactor)
    begin = aiv * factor
    end = pl.min(begin + factor, plan_i64_bn2s2(words, PF_bn2s2.sinkPostBlockTotal))
    axis = plan_i64_bn2s2(words, PF_bn2s2.sinkReduceAxis)
    for head in pl.range(begin, end):
        pl.set_validshape(total, [1, 128])
        fill_zero_vf(total, 1)
        for part in pl.range(0, (axis + 16383) // 16384):
            size = pl.min(16384, axis - part * 16384)
            src = source.next()
            pl.set_validshape(src, [1, size])
            pl.load(src, tensor_partial, [0, head * axis + part * 16384])
            loops = (size + 63) // 64 - 1
            plan_sink_reduce_chunk_vf(total, src, loops, size - loops * 64)
        plan_sink_negate_vf(total)
        pl.set_validshape(total, [1, 1])
        pl.store(tensor_out, total, [0, head], atomic=pl.AtomicType.AtomicNone)


def plan_sfmg_offset_bn2s2(h, info, sub):
    return (
        info.n2oIdx * h.totalQ
        + info.tokenBaseQ
        + info.s1oIdx * 128
        + sub * info.firstHalfS1RealSize
    )


def plan_load_sfmg_bn2s2(h, info, sfmg_tensor, sfmg_ub, sub):
    if info.halfS1RealSize > 0:
        offset = plan_sfmg_offset_bn2s2(h, info, sub)
        pl.set_validshape(sfmg_ub, [info.halfS1RealSize, 1])
        pl.load(sfmg_ub, sfmg_tensor, [0, offset], order=[1, 0])
        pl.set_validshape(sfmg_ub, [64, 1])


def plan_compute_presfmg_bn2s2(
    ci,
    words,
    h,
    cuq,
    aiv,
    y,
    dy,
    sfmg_tensor,
    y_group,
    dy_group,
    out_group,
    flat_group,
    v1w,
):
    used = plan_i64_bn2s2(words, PF_bn2s2.sfmgUsedCoreNum)
    normal = plan_i64_bn2s2(words, PF_bn2s2.sfmgNormalCoreRows)
    total_rows = plan_i64_bn2s2(words, PF_bn2s2.sfmgRows)
    window_rows = mem.PLAN_SFMG_ROWS_bn2s2[dv_align]
    if dv_align == 768:
        if ci.dvSize > 256 and ci.dvSize <= 320:
            window_rows = 80
        elif (
            mask_mode == 3
            and h.logicalSplit == PlanSplit_bn2s2.GS
            and ci.dvSize > 320
            and ci.dvSize <= 384
        ):
            window_rows = 64
    if aiv < used:
        position = aiv * normal
        end = pl.min(position + normal, total_rows)
        info = plan_run_info_bn2s2()
        info.idx0Q = pl.astype(0, pl.DT_INT64)
        info.rowShift = pl.astype(0, pl.DT_INT64)
        while position < end:
            head = position // h.totalQ
            token = position - head * h.totalQ
            rows = pl.min(window_rows, pl.min(end - position, h.totalQ - token))
            info.idx2Q = head
            info.loadBase = token
            info.halfS1RealSize = rows
            y_ub = y_group.next()
            dy_ub = dy_group.next()
            out_ub = out_group.next()
            flat_ub = flat_group.next()
            if dv_align == 768:
                if ci.dvSize > 256 and ci.dvSize <= 320:
                    plan_presfmg_full320_bn2s2(
                        ci, info, y, dy, y_ub, dy_ub, out_ub, rows
                    )
                elif (
                    mask_mode == 3
                    and h.logicalSplit == PlanSplit_bn2s2.GS
                    and ci.dvSize > 320
                    and ci.dvSize <= 384
                ):
                    # Two 104x256 slots per input. Reinterpret the selected
                    # ping/pong slot only; the row scalar remains at offset 0.
                    y_full = pl.reinterpret(y_ub, shape=[64, 384])
                    dy_full = pl.reinterpret(dy_ub, shape=[64, 384])
                    pl.set_validshape(y_full, [rows, ci.dvSize])
                    pl.set_validshape(dy_full, [rows, ci.dvSize])
                    pl.load(y_full, y, [0, token, head, 0], order=[1, 3])
                    pl.load(dy_full, dy, [0, token, head, 0], order=[1, 3])
                    tail = ci.dvSize - 256
                    softmax_grad_front_row384_vf(
                        out_ub,
                        y_full,
                        dy_full,
                        rows,
                        0,
                        384,
                        (tail + 1) // 2,
                        tail // 2,
                        0,
                    )
                else:
                    plan_presfmg_reduce_rows_bn2s2(
                        ci, info, 0, y, dy, y_ub, dy_ub, out_ub, v1w, rows
                    )
            else:
                plan_presfmg_reduce_rows_bn2s2(
                    ci, info, 0, y, dy, y_ub, dy_ub, out_ub, v1w, rows
                )
            pl.set_validshape(flat_ub, [1, rows])
            pl.store(
                sfmg_tensor, flat_ub, [0, position], atomic=pl.AtomicType.AtomicNone
            )
            position = position + rows


def plan_presfmg_reduce_rows_bn2s2(
    constInfo,
    runInfo,
    subIdx,
    tensor_y,
    tensor_dy,
    y_ub,
    dy_ub,
    sfmg_ub,
    rowStride,
    loadRows,
):
    """softmaxGradFront: sfmg = rowsum(dy * y)。一趟 VECTOR_BASEM 行。"""
    if dv_align == 768:
        plan_presfmg_reduce_rows_d768_bn2s2(
            constInfo,
            runInfo,
            tensor_y,
            tensor_dy,
            y_ub,
            dy_ub,
            sfmg_ub,
            rowStride,
            loadRows,
        )
        return
    pl.set_validshape(y_ub, [loadRows, constInfo.dvSize])
    pl.set_validshape(dy_ub, [loadRows, constInfo.dvSize])
    pl.load(
        y_ub,
        tensor_y,
        [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, 0],
        order=[1, 3],
    )
    pl.load(
        dy_ub,
        tensor_dy,
        [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, 0],
        order=[1, 3],
    )
    # 列宽跟 dv_align：256/192 各一档 VF；64 与 128（含 D=192,Dv=128）共用
    # 一次 128-lane load，用 even/odd mask 收真实 Dv。
    if dv_align == 256:
        w0 = pl.min(constInfo.dvSize, 128)
        w1 = pl.max(constInfo.dvSize - 128, 0)
        softmax_grad_front_row256_bn2s2_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            runInfo.halfS1RealSize,
            runInfo.rowShift,
            rowStride,
            (w0 + 1) // 2,
            w0 // 2,
            (w1 + 1) // 2,
            w1 // 2,
        )
    elif dv_align == 192:
        w0 = pl.min(constInfo.dvSize, 128)
        w1 = pl.max(constInfo.dvSize - 128, 0)
        softmax_grad_front_row192_bn2s2_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            runInfo.halfS1RealSize,
            runInfo.rowShift,
            rowStride,
            (w0 + 1) // 2,
            w0 // 2,
            (w1 + 1) // 2,
            w1 // 2,
        )
    else:
        softmax_grad_front_row_bn2s2_vf(
            sfmg_ub,
            y_ub,
            dy_ub,
            runInfo.halfS1RealSize,
            runInfo.rowShift,
            rowStride,
            constInfo.dvEvenSize,
            constInfo.dvOddSize,
        )


def plan_presfmg_reduce_rows_d768_bn2s2(
    constInfo, runInfo, tensor_y, tensor_dy, y_ub, dy_ub, sfmg_ub, rowStride, loadRows
):
    """768 档 V1：3 趟 VF256，后两趟累加到同一 sfmg。工作列宽 256。"""
    for k in pl.range(0, 3):
        colOff = k * 256
        rem = pl.max(constInfo.dvSize - colOff, 0)
        winW = pl.min(256, rem)
        if rem > 0:
            pl.set_validshape(y_ub, [loadRows, winW])
            pl.set_validshape(dy_ub, [loadRows, winW])
            pl.load(
                y_ub,
                tensor_y,
                [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, colOff],
                order=[1, 3],
            )
            pl.load(
                dy_ub,
                tensor_dy,
                [runInfo.idx0Q, runInfo.loadBase, runInfo.idx2Q, colOff],
                order=[1, 3],
            )
            w0 = pl.min(rem, 128)
            w1 = pl.max(rem - 128, 0)
            even0 = (w0 + 1) // 2
            odd0 = w0 // 2
            even1 = (w1 + 1) // 2
            odd1 = w1 // 2
            if k == 0:
                softmax_grad_front_row256_bn2s2_vf(
                    sfmg_ub,
                    y_ub,
                    dy_ub,
                    runInfo.halfS1RealSize,
                    runInfo.rowShift,
                    rowStride,
                    even0,
                    odd0,
                    even1,
                    odd1,
                )
            else:
                softmax_grad_front_row256_acc_vf(
                    sfmg_ub,
                    y_ub,
                    dy_ub,
                    runInfo.halfS1RealSize,
                    runInfo.rowShift,
                    rowStride,
                    even0,
                    odd0,
                    even1,
                    odd1,
                )


def plan_presfmg_full320_bn2s2(ci, info, y, dy, y_ub, dy_ub, out_ub, rows):
    # Alias only: original104x256 slots and mutex lifetimes are retained.
    y_whole = pl.reinterpret(y_ub, shape=[81, 320])
    dy_whole = pl.reinterpret(dy_ub, shape=[81, 320])
    pl.set_validshape(y_whole, [rows, ci.dvSize])
    pl.set_validshape(dy_whole, [rows, ci.dvSize])
    pl.load(y_whole, y, [info.idx0Q, info.loadBase, info.idx2Q, 0], order=[1, 3])
    pl.load(dy_whole, dy, [info.idx0Q, info.loadBase, info.idx2Q, 0], order=[1, 3])
    tail = ci.dvSize - 256
    softmax_grad_front_row320_single_vf(
        out_ub, y_whole, dy_whole, rows, (tail + 1) // 2, tail // 2
    )


def plan_presfmg_gs_interleaved_bn2s2(
    ci,
    words,
    h,
    aiv,
    gm,
    zero,
    y,
    dy,
    sfmg_tensor,
    y_group,
    dy_group,
    out_group,
    flat_group,
):
    # Only GS / pre-SFMG / no-sink / Dv257..320 enters this path.
    # Original metadata partitions and original disjoint zero arena are retained.
    qb, qc = plan_pre_range_bn2s2(
        words,
        PF_bn2s2.qPreBlockFactor,
        PF_bn2s2.qPreBlockTotal,
        plan_i64_bn2s2(words, PF_bn2s2.qPostBlockTotal),
        aiv,
    )
    kb, kc = plan_pre_range_bn2s2(
        words,
        PF_bn2s2.kPreBlockFactor,
        PF_bn2s2.kPreBlockTotal,
        plan_i64_bn2s2(words, PF_bn2s2.kPostBlockTotal),
        aiv,
    )
    vb, vc = plan_pre_range_bn2s2(
        words,
        PF_bn2s2.vPreBlockFactor,
        PF_bn2s2.vPreBlockTotal,
        plan_i64_bn2s2(words, PF_bn2s2.vPostBlockTotal),
        aiv,
    )
    pl.set_validshape(zero, [1, mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg]])
    fill_zero_vf(zero, mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg] // 64)
    used = plan_i64_bn2s2(words, PF_bn2s2.sfmgUsedCoreNum)
    normal = plan_i64_bn2s2(words, PF_bn2s2.sfmgNormalCoreRows)
    total_rows = plan_i64_bn2s2(words, PF_bn2s2.sfmgRows)
    position = aiv * normal
    end = position
    if aiv < used:
        end = pl.min(position + normal, total_rows)
    offset = pl.astype(0, pl.DT_INT64)
    info = plan_run_info_bn2s2()
    info.idx0Q = pl.astype(0, pl.DT_INT64)
    info.rowShift = pl.astype(0, pl.DT_INT64)
    while position < end or offset < qc or offset < kc or offset < vc:
        if position < end:
            head = position // h.totalQ
            token = position - head * h.totalQ
            rows = pl.min(80, pl.min(end - position, h.totalQ - token))
            info.idx2Q = head
            info.loadBase = token
            info.halfS1RealSize = rows
            y_ub = y_group.next()
            dy_ub = dy_group.next()
            out_ub = out_group.next()
            flat_ub = flat_group.next()
            plan_presfmg_full320_bn2s2(ci, info, y, dy, y_ub, dy_ub, out_ub, rows)
            pl.set_validshape(flat_ub, [1, rows])
            pl.store(
                sfmg_tensor, flat_ub, [0, position], atomic=pl.AtomicType.AtomicNone
            )
            position = position + rows
        if offset < qc:
            size = pl.min(mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg], qc - offset)
            pl.set_validshape(zero, [1, size])
            pl.store(gm.dqWs, zero, [0, qb + offset], atomic=pl.AtomicType.AtomicNone)
        if offset < kc:
            size = pl.min(mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg], kc - offset)
            pl.set_validshape(zero, [1, size])
            pl.store(gm.dkWs, zero, [0, kb + offset], atomic=pl.AtomicType.AtomicNone)
        if offset < vc:
            size = pl.min(mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg], vc - offset)
            pl.set_validshape(zero, [1, size])
            pl.store(gm.dvWs, zero, [0, vb + offset], atomic=pl.AtomicType.AtomicNone)
        offset = offset + mem.PLAN_ZERO_ELEMENTS_bn2s2[pre_sfmg]


def process_normal_vec3_bn2s2(ci, info, sub, ds, c1, c2, sfmg, nzp):
    rows = info.halfS1RealSize
    if rows > 0:
        broadcast_sub_mul_cast_vf(nzp, c1, c2, sfmg, ci.scaleValue, rows)
        copy_ub2l1(ds, nzp, sub * info.firstHalfS1RealSize, rows)


def process_normal_vec4_bn2s2(ci, info, sub, p, c2, nzp):
    rows = info.halfS1RealSize
    if rows > 0:
        cast_p_vf(nzp, c2, rows)
        copy_ub2l1(p, nzp, sub * info.firstHalfS1RealSize, rows)


def process_normal_vec2_bn2s2(
    constInfo,
    runInfo,
    subIdx,
    tensor_lse,
    mm2_ub,
    lse_ub,
    tensor_mask,
    mask_ub,
    mask_pre_ub,
):
    if mask_mode != 0:
        copy_in_atten_mask(
            constInfo, runInfo, subIdx, tensor_mask, mask_ub, mask_pre_ub
        )
    muls_sel_simple_softmax_vf(
        mm2_ub,
        lse_ub,
        mask_ub,
        constInfo.scaleValue,
        runInfo.halfS1RealSize,
        runInfo.rowShift,
    )


def load_normal_lse_bn2s2(constInfo, runInfo, tensor_lse, lse_ub):
    load_vec_lse_rows(
        constInfo,
        runInfo,
        tensor_lse,
        lse_ub,
        _vector_load_rows_bn2s2(constInfo, runInfo),
    )
