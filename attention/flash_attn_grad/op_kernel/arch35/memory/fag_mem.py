# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""片上内存布局与 buffer-id 分配。

pypto 的 tile 地址与 mutex id 都必须是 trace 期常量，所以布局在
**import 期**一次算完，导出平表给 :mod:`fag_buffers`。

**一块 buffer 只声明一次。** ``arena.buf(名字, 字节数, 槽位数)`` 同时给出地址
和 buffer-id —— arena 持有本侧的 :class:`_MutexPool`，两者不可能对不上，也不
存在「在 A 处列一遍名单、在 B 处再列一遍」。

**mutex id 不由人指定。** 编译器只按「两块 buffer 的 id 是否相交」做分组
（``_call_parser._group_refs_by_mutex_overlap`` 的并查集），不看数值大小，也不
要求一块 buffer 的号连续，所以任何单射编号都等价，手工挑号只会挑出撞号。

本模块是地址与 id 的**唯一出处**：kernel 主体不再出现裸地址或裸 id。

读这个文件的三种方式：

* 想知道某块 buffer 放在哪 —— 看 ``_build_*``，它就是布局说明书；
* 想看完整内存图（地址 + 大小 + 锁）—— :func:`describe_layout`；
* 想知道 id 被谁占了、还剩几个 —— :func:`audit_mutex`。

::

    cd attention/flash_attn_grad/op_kernel/arch35
    python -c "import fag_mem; print(fag_mem.describe_layout(192))"
    python -c "import fag_mem; print(fag_mem.audit_mutex(192))"
    python -c "import fag_mem; print(fag_mem.describe_layout(256))"

容量由 ``_Arena`` 守住（L1 512KB / L0A·L0B 各 64KB / L0C 256KB / UB 256KB），
id 上限由 ``_MutexPool`` 守住（每核 32 个），越界都会在 import 期失败。
"""

# --- generated imports (fag_fiximports.py) ---
from fag_common import (
    CUBE_BASEM,
    CUBE_BASEN,
    D_CHUNK_MAXW,
    D_NCHUNKS,
    D_PHYS_W,
    HEAD_DIM_ALIGN,
    NZP_ROWS,
    SP_ROWS,
    SP_WORK_W,
    VECTOR_BASEM,
    VECTOR_BASEN,
    SINK_REDUCE_BATCH,
)
# --- end generated imports ---

from memory.fag_layout import _ArenaBase, _MutexPoolBase, _addr, _addrs, _mutex


# d_align 的分档。所有地址表与 id 表都以它为键。
_TIERS = (64, 128, 192, 256, 512, 768)

_L1_BYTES = 512 * 1024
_L0AB_BYTES = 64 * 1024
_L0C_BYTES = 256 * 1024
_UB_BYTES = 256 * 1024


# ==================================================================
#  buffer-id (mutex) 分配
# ==================================================================
# id 空间是**扁平的 [0, 32)，不按内存空间分**：L1 / L0A / L0B / L0C / UB 共用
# 同一批号。生成代码里就是 get_buf(PIPE_x, id, 0)，除了 pipe 和 id 再没有第三个
# 维度。唯一的分离维度是**核**：cube 与 vector 是两颗物理核，各有一套 32 个
# token，同号互不相干。
#
# CV 共享的 6 块 buffer 声明在两个 section 之外，只有一份 mutex_ids，两侧都要
# 用，所以必须两侧同号。把它们钉在号段最前面，两个池子都从这段之后开始发号，
# 谁也不会踩到 —— 这样 cube 池和 vector 池就能各自独立编号。
_CV_SLOTS = {
    "mm1_res": 2,  # cube FIX 写、vector V 读
    "mm2_res": 2,
    "ds": 1,  # vector 写、cube mm3/mm4 读
    "p": 1,
    # dS^T / P^T 与 ds / p 同址但**另发一个号**：它们是同一块 L1 的 ZN 视图，
    # 有独立游标，读写序由 CV flag(SYNC_V3_TO_C3 等)而不是 mutex 保证。
    "ds_t": 1,
    "p_t": 1,
}


def _build_cv_mutex(slots):
    table, nxt = {}, 0
    for name, n in slots.items():
        table[name] = list(range(nxt, nxt + n))
        nxt += n
    return table, nxt


_CV_MTX, _CV_RESERVED = _build_cv_mutex(_CV_SLOTS)


class _MutexPool(_MutexPoolBase):
    """一侧核的 buffer-id 顺序发号器。

    一律顺序发号、绝不复用：id 只要单射就正确，而「同一地址挂了不同号」才是
    真正的 bug（互斥直接消失，且不会当场报错）。别名视图不在这里申请号，它们
    直接引用被别名那块 buffer 的记录，同号是结构保证而非人为约定。
    """

    def __init__(self, side, tag):
        super().__init__(side, tag, _CV_RESERVED, _CV_MTX)

    def take(self, label, count):
        return self._take(label, count, check_reserved=True)

    def reserve(self, label, ids):
        """Reserve fixed CV ids without changing other tiers' sequential ids."""
        for i in ids:
            if i < 0 or i >= self.LIMIT or i in self.owners:
                raise AssertionError(
                    f"{self._side}({self._tag}) invalid mutex reservation: {i}"
                )
            self.owners[i] = label


# ==================================================================
#  片上内存的 bump 分配器
# ==================================================================
class _Arena(_ArenaBase):
    """一块片上内存的顺序铺排。

    只在 import 期执行，绝不进 trace：导出的永远是 int / list[int] 的平表，
    因为 tile 的 ``addrs=`` / ``mutex_ids=`` 只接受 trace 期常量。

    ``capacity`` 是**绝对上限**（含 ``base`` 之前的保留区），越界即断言失败。
    """

    def buf(self, label, nbytes, slots=1, align=1, cv=None, ids=None):
        return self._declare_buffer(label, nbytes, slots, align, cv, _CV_MTX, ids)


def _flat(table, field):
    """槽位数与档无关的 buffer 导出成扁平列表。

    vector 侧的 buffer 名单与 d_align 无关，四档发号顺序一致，取哪档都一样；
    真要改动发号顺序，用 audit_mutex() 对一下各档是否仍然对齐。
    """
    return list(table[HEAD_DIM_ALIGN][field]["mutex"])


# ==================================================================
#  Cube 侧：L1 (512KB) + L0A/L0B (各 64KB) + L0C (256KB)
# ==================================================================
def _build_l1(width, pool):
    """铺排一档的 L1，返回 ``{名字: buffer}``。

    Q/dO 是多槽 double buffer：tile_group 填多个 addrs + 多个 mutex_id 后，
    框架会自动插核内同步，使下一轮 load 与本轮 matmul 重叠。
    槽位数不是 2 而是 3+：cube 领先 vector 两个 task，mm2 在
    task N 载入 Q_N，而 mm4 要的是 Q_{N-2}；3 槽时 N%3 != (N-2)%3，两者不会
    互相覆写，于是 mm4/mm5 可以直接复用 mm2/mm1 搬好的数据，无需再读 GM。
    """
    ar = _Arena(f"L1 d_align={width}", _L1_BYTES, pool)
    blk = (
        CUBE_BASEM * D_PHYS_W[width] * 2
    )  # 64 档 16KB，128 档 32KB，192 档 48KB，256/768 档 64KB
    dsp = CUBE_BASEM * CUBE_BASEN * 2  # dS/P 是 [M,N]，与 D 无关，恒 32KB
    # 64/128 档 4 槽；192 档 3 槽；256/768 档 1-ahead 只要 2 槽（物理宽 256）。
    if width <= HEAD_DIM_ALIGN:
        nq = 4
    elif width == 192:
        nq = 3
    else:
        nq = 2

    q = ar.buf("q", blk, slots=nq)  # q_lag_view 复用这份记录，同址同号
    do = ar.buf("do", blk, slots=nq)  # do_lag_view 同理
    k = ar.buf("k", blk)
    # BN2/wide-D load C2/C3 separately. GS through D256 treats these two
    # allocations as a shared two-slot K buffer (see exports).
    # C2 uses ZN[D,S2], C3 uses the byte-equivalent NZ[S2,D] view.
    k_nd = ar.buf("k_nd", blk)
    v = ar.buf("v", blk)
    # dS^T / P^T 不单独占地址，是 ds / p 上的 ZN 视图（见 make_ds_t_l1）。
    ds = ar.buf("ds", dsp, cv="ds")
    p = ar.buf("p", dsp, cv="p")
    # Narrow GS normal kernels can pipeline two V->C tasks. Keep slot 0
    # addresses unchanged for BN2/deterministic/swizzle kernels. The added
    # slots take L1 from 416KB to 480KB, below its 512KB capacity.
    ds_db1 = ds
    p_db1 = p
    if width <= HEAD_DIM_ALIGN:
        ds_db1 = ar.buf("ds_db1", dsp, ids=[28])
        p_db1 = ar.buf("p_db1", dsp, ids=[29])

    return {
        "q": q,
        "do": do,
        "k": k,
        "k_nd": k_nd,
        "v": v,
        "ds": ds,
        "p": p,
        "ds_db1": ds_db1,
        "p_db1": p_db1,
        "_arenas": [ar],
    }


# acc / L0A / L0B 的列宽：MM2/MM1 输出 [M,s2]，MM3 输出 dQ 的一个 D 分块，
# 取两者较大值，一块 tile 同时够用。三档都等于 128。
ACC_COLS = {w: max(CUBE_BASEN, D_CHUNK_MAXW[w]) for w in _TIERS}


def _build_l0ab(width, pool, cols=None, arena_type=_Arena, capacity=_L0AB_BYTES):
    """L0A 与 L0B —— 两块独立的 64KB，各开双缓冲，布局相同但各有一套号。

    5 个 matmul 串行复用同一块 L0A/L0B 时，pl.move 填充期间 mac 只能干等
    （实测单缓冲 mte1 占用率 0.309）；双缓冲让下一个 matmul 的
    L1->L0 搬运与当前 matmul 的计算重叠。

    间距必须按 **tile 实际声明的宽度** ACC_COLS 算，而不是 D 分块宽度：
    L0A/L0B 要同时装下 MM3/MM4/MM5 的 [M,N] 左矩阵，故声明成 ACC_COLS 列。
    若按 192 档的分块宽 96 算间距（24KB），两个槽位会重叠（每块实占 32KB）。
    """
    blk = CUBE_BASEM * (ACC_COLS[width] if cols is None else cols) * 2
    a = arena_type(f"L0A d_align={width}", capacity, pool)
    b = arena_type(f"L0B d_align={width}", capacity, pool)
    # 两块布局相同，地址表一致，故对外只导出 l0a 那份（L0AB_ADDRS）。
    l0a = a.buf("l0a", blk, slots=2)
    l0b = b.buf("l0b", blk, slots=2)
    return {"l0a": l0a, "l0b": l0b, "_arenas": [a, b]}


def _build_l0c(width, pool, bn2_d192=False):
    """L0C：acc 槽位 + dK/dV 的分块块。

    dV 要在同一 s2 块内跨 s1 轮次累加（GS1S2 ``d_align<=192``、BN2 192/256），
    s2 切换时才 fixpipe 出 GM，所以每个 D 分块都得有自己的常驻 L0C：192/256
    档 dV 各占两个分块槽（2×48KB / 2×64KB）。dK 只在 192/256 档的各个组合里
    走 fp32 workspace（GS1S2 192/256、BN2 256；与 AscendC headDim∈[192,256] 的
    "only Dv L0C resident" 做法一致），matmul 输出直接复用 acc 的槽位；BN2 192
    例外，dK/dV 都常驻（见下方分支）；其余档 dK 只有单分块槽（64/128 常驻、
    768 单 scratch）。

    acc 单缓冲时 fixpipe 写出期间下一个 matmul 只能干等（实测 fixpipe 0.458
    而 mac 仅 0.783），故**除 BN2 192 外所有档位至少两槽**。预算：
    D 64 = acc 3×64 + dk 32+ dv 32 = 256KB；128 = acc 2×64 + dk 64 + dv 64 = 256KB；
    GS 192 = acc 2×64 +dv 2×48 = 224KB；BN2 192 = acc 1×64 + dk 2×48 + dv 2×48 = 256KB；
    256 = acc 2×64 + dv 2×64 = 256KB；768 = acc 2×64 + dk 64 + dv 64 = 256KB。
    """
    ar = _Arena(f"L0C d_align={width}", _L0C_BYTES, pool)
    accBlk = CUBE_BASEM * ACC_COLS[width] * 4  # acc 要装 MM2/MM1 的 [M,s2]
    dkvBlk = CUBE_BASEN * D_CHUNK_MAXW[width] * 4  # dK/dV 只装一个 D 分块

    # 256/768 档 C4/C5 原本都不驻留：每 task atomic 出 GM，只要 acc 双缓冲。
    # 方案 A 把 192/256 档的 dV 改成常驻（对齐 AscendC FlashAttentionScoreGrad
    # headDim∈[192,256] 的 "only Dv L0C resident" 做法），见下方分支；GS1S2
    # 192/256 与 BN2 256 的 dK 走 fp32 workspace，matmul 输出复用 acc 槽位，不再
    # 单独开块（BN2 192 例外：dK 也常驻，走下面的 bn2_d192 分支）。
    if width == 192 and bn2_d192:
        # （BN2 192）：dK/dV 都在 L0C 跨 s1 累加，每个 D 分块一个常驻槽，
        # 预算 acc 1×64 + dk 2×48 + dv 2×48 = 256KB。acc 因此退回单槽 ——
        # 实测（2026-10-29）BN2 在 1-ahead 流水下用不上第二槽，而 dK 写 ws
        # 会让 fixpipe 流量按 s1Outer 倍增长（FIX 30.6%→37.6%）。
        acc = ar.buf("acc", accBlk, slots=1)
        nChunk = D_NCHUNKS[width]
        dk = ar.buf("dk", dkvBlk, slots=nChunk)
        dv = ar.buf("dv", dkvBlk, slots=nChunk)
        return {"acc": acc, "dk": dk, "dv": dv, "_arenas": [ar]}

    if width == 64:
        # D=64：dK/dV 按真实 64 列声明（各 32KB），让出的 64KB 给 acc 第 3 槽。
        acc = ar.buf("acc", accBlk, slots=3)
    else:
        acc = ar.buf("acc", accBlk, slots=2)
    if width == 192 or width == 256:
        # dV 两个 D 分块常驻 L0C（192→2×48KB，256→2×64KB）：BN2 在 flush 时
        # fixpipe 直出 16-bit，GS1S2 累加回 fp32 ws 交给全局 post。
        dv = ar.buf("dv", dkvBlk, slots=2)
        dk = acc  # 同址同号：dK 与 acc 共用，靠同一 mutex 串行
    else:
        # 64/128 单分块常驻、768 单 scratch；dV 同样只有一个分块。
        dk = ar.buf("dk", dkvBlk, slots=1)
        dv = ar.buf("dv", dkvBlk, slots=1)
    return {"acc": acc, "dk": dk, "dv": dv, "_arenas": [ar]}


def _build_l0c_d192dv128(pool, gs, bn2):
    """GS without swizzle: N128+64 dQ/dK and one Dv128 resident tile.

    This layout and the normal GS layout are mutually exclusive tiling keys.
    Each 96KiB acc/dK allocation has two compact subviews at offsets 0/64KiB,
    sharing one mutex. Full-M results can be flushed through one full-N view;
    tail-M results retain two compact regions and use two FIX stores.
    dK remains resident across tasks for the same owner. The active budget is
    acc96 + dK96 + dV64 = 256KiB; L0A/L0B retain their two 32KiB slots.
    """
    ar = _Arena("L0C GS d_align=192 dv_align=128", _L0C_BYTES, pool)
    acc = ar.buf("acc_dv128", CUBE_BASEM * 192 * 4, ids=gs["acc"]["mutex"][:1])
    dk = ar.buf("dk192", CUBE_BASEN * 192 * 4, ids=bn2["dk"]["mutex"][:1])
    dv = ar.buf("dv128", CUBE_BASEN * 128 * 4, ids=gs["dv"]["mutex"][:1])
    return {"acc": acc, "dk": dk, "dv": dv, "_arenas": [ar]}


# ==================================================================
#  Vector 侧：UB (256KB)
# ==================================================================
# 头部留给 codegen 的 UB[0] tiling 中转：BN2 没有 pre/sync_all，不留这块会和
# cube 的 fixpipe 抢 UB[0]。所有 UB arena 都从它之后开始。
UB_HEAD_RESERVE = 0x1000
# [64,1] fp32 只有 256B，但按 512B 一格排，保证后面的槽位对齐。
_UB_ROW_SLOT = 0x200

# pre/post 走一维元素，不按 [行, D] 切。行区间已由 metadata 按 nAiv 均分，
# 块大不再空核。zt(pre_zero) 单槽只占 [0x1000,0x9000) 的 32KB（与 mm1
# 首槽同段，pre/post 阶段被 sync_all 与主循环隔开，可整块借用）：刻意与
# sfmg 前移窗口 [0x9000,0x40000) 完全避开——zt 与 sp 不同 mutex 且无地址
# 相交，框架不插 wait，pre_zero 清零写(MTE3)与 presfmg 载入(MTE2)/VF 得以
# 在同一 section 并行（交错 pre，见 fag_kernel）。板测哨兵 192MB 清零
# ~64µs≈3TB/s 聚合写带宽，32KB 背靠背突发效率仍高于顺序大块的拐点。
# 各档 elems 一致且被 128 整除，full-tile 的 VF 段数是编译期常量。
# ⚠️ 不能扩回 248KB@头部：zt 会横跨 sp 窗口，窗口阶段 sp 覆写该地址，
# 晚于窗口的清零 store 读 zt 会把 sp 残留当「零」写进 ws 中段。
# ⚠️ sp 借 mm1/mm2 + 主图段（[0x9000,0x40000)）需上板重新验证（见 sp arena
# 注释）。zt 缩到 32KB 后清零 chunk 减半、任务数翻倍，属预期代价。
PRE_ELEMS = {
    64: 8192,
    128: 8192,
    192: 8192,
    256: 8192,
    512: 8192,
    768: 8192,
}  # 单槽 fp32
# f16 视图（BN2 need_zero）占一半，见 _build_ub 的 pre arena。
POST_ELEMS = {
    64: 20480,
    128: 21120,
    192: 21120,
    256: 21120,
    512: 21120,
    768: 21120,
}  # depth-2，fp32 + 16-bit


def _build_ub(width, pool):
    """一档的 vector 侧全部 UB。

    三张图共用这 256KB，靠执行阶段错开：

    * 主图（y/dy/dS/P/sfmg/lse，BN2 用 nzp/cast）—— 从 0x21000 起；
    * mm1/mm2 结果区 —— CV 共享，cube fixpipe 写、vector 读，从头部起；
    * pre/post —— 与主循环被 sync_all 完全隔开，整块借用头部。

    V1 一趟 VECTOR_BASEM 行、按行 reduce，不再开 prod/tmp。列宽
    ``v1w`` 就是 ``dv_align``（与 ``d_align`` 相同，仅 D=192 且 Dv=128 时为 128）。
    512/768 档保留 256 列主区，SFMG 使用 8 行完整 Dv 的双槽视图。
    attenmask 复用 y/dy 这两条记录：uint8[64,128]=8KB，而 y/dy 在 v1w>=64
    时至少 8KB，装得下。
    复用的是**同一条记录**，所以「同址必须同号」是结构保证，不靠人守。

    BN2 的 cast/l0c_load 与 GS 的 P 输出是编译期互斥的两种视图。
    GS 的 dS 保留 nzp，P 借用 cast 起始的 16640B；BN2 仍用原布局
    （narrow 传 HEAD_DIM_ALIGN 取档）。两种 GS 输出各有独立地址和 mutex。
    """
    main = _Arena(f"UB v1w={width}", _UB_BYTES, pool, base=UB_HEAD_RESERVE + 0x20000)
    v1Alloc = 256 if width >= 512 else width
    y = main.buf("y", VECTOR_BASEM * v1Alloc * 2)
    dy = main.buf("dy", VECTOR_BASEM * v1Alloc * 2)
    # NZ+1 行：末行是 bank 错开的填充，见 fag_common.NZP_BLOCK_STRIDE。
    nzp = main.buf("nzp", NZP_ROWS * VECTOR_BASEN * 2)
    sfmg = main.buf("sfmg", _UB_ROW_SLOT, align=_UB_ROW_SLOT)
    lse = main.buf("lse", _UB_ROW_SLOT, slots=2)
    sink_part = main.buf("sink_part", _UB_ROW_SLOT, align=_UB_ROW_SLOT)
    cast = main.buf("cast", VECTOR_BASEM * VECTOR_BASEN * 2)
    # BN2 192 档 dK 的 chunk1 从 L0C 搬进 UB 后在此 cast。声明成与
    # mm2_res 同宽 [64,128]，保证 muls_cast_vf 的 flat stride 一致（32KB）。
    # 其它档不用它，只占 [64,64]=16KB，避免撑爆 768 档预算。
    _load_w = VECTOR_BASEN if width == 192 else VECTOR_BASEN // 2
    l0c_load_ub = main.buf("l0c_load_ub", VECTOR_BASEM * _load_w * 4)
    # GS 不创建 BN2 的 cast/l0c_load tile，可将这段空间作为独立 P 输出：
    # 65*128*2B 比 cast 的 64*128*2B 多 256B，借用相邻 l0c_load 的头部。
    # mutex 复用 cast 的记录，既不增加 id，也不改 BN2 的地址/号序。
    # 这不是可与 BN2 scratch 同时使用的别名；仅 GS factory 引用 p_out。
    # pre/SFMG/post 与主循环仍由原 sync_all 隔开，保持原来的分时复用。
    gs_out_bytes = NZP_ROWS * VECTOR_BASEN * 2
    bn2_scratch_end = l0c_load_ub["addrs"][0] + VECTOR_BASEM * _load_w * 4
    assert bn2_scratch_end <= _UB_BYTES
    gs_out = _Arena(
        f"UB GS P output v1w={width} (aliases BN2 scratch)",
        bn2_scratch_end,
        pool,
        base=cast["addrs"][0],
    )
    p_out = gs_out.buf("p_out_gs", gs_out_bytes, ids=cast["mutex"])
    # All other live GS main buffers end before the BN2-only scratch interval.
    assert sink_part["addrs"][0] + _UB_ROW_SLOT <= p_out["addrs"][0]
    assert nzp["addrs"][0] + gs_out_bytes <= p_out["addrs"][0]
    assert set(nzp["mutex"]).isdisjoint(p_out["mutex"])
    # BN2 wide（d_align>128）的载入/出口 tile 不再单独占 UB：载入 tile
    # [64, chunk] fp32 alias 到 y（V2 之后 y 已死），16-bit 出口 alias 到 cast
    # （BN2 wide 二进制里 GS 的 cast 不用）。整块 cast（chunk=96/128 列）是为了
    # 让 16-bit 写的行宽达到 192B/256B，避免 64 列(128B)行的短 burst。

    shared = _Arena("UB mm1/mm2", _UB_BYTES, pool, base=UB_HEAD_RESERVE)
    resBlk = VECTOR_BASEM * VECTOR_BASEN * 4  # fp32[64,128] = 32KB
    mm1 = shared.buf("mm1_res", resBlk, cv="mm1_res")
    mm2 = shared.buf("mm2_res", resBlk, cv="mm2_res")

    # pre 的 16-bit 支（BN2 need_zero）只占一半，按 fp32 算上限。
    pre = _Arena(f"UB pre d_align={width}", _UB_BYTES, pool, base=UB_HEAD_RESERVE)

    # post 的 fp32 输入与 16-bit 输出各用一套号，框架据此判定「下一块的 load」与
    # 「本块的 cast/store」无冲突，从而让 MTE2 / V / MTE3 三条流水重叠。
    # f32/f16 先于 pre_zero 发号：zero 要复用 f32[0] 的号（见下）。
    post = _Arena(f"UB post d_align={width}", _UB_BYTES, pool, base=UB_HEAD_RESERVE)
    n = POST_ELEMS[width]
    f32 = post.buf("post_f32", n * 4, slots=2)
    f16 = post.buf("post_f16", n * 2, slots=2)
    # pre_zero 与 post_f32 生命期不重叠（pre 段 vs post 段，中间隔着主循环
    # 前后的 sync_all），显式复用 f32[0] 的号，给 sp_y/sp_dy 双槽腾位。
    zero = pre.buf("pre_zero", PRE_ELEMS[width] * 4, ids=f32["mutex"][:1])
    # 收尾归约用 SINK_REDUCE_BATCH 个独立的 1 行 tile（同一批只等一次 barrier）
    sink_row_post = [
        post.buf(f"sink_row_post{i}", _UB_ROW_SLOT, align=_UB_ROW_SLOT)
        for i in range(SINK_REDUCE_BATCH)
    ]
    sink_red_post = post.buf("sink_red_post", _UB_ROW_SLOT, align=_UB_ROW_SLOT)

    # sfmg 前移缓冲：窗口 [0x9000,0x40000)=220KB（zt 缩到 32KB 后 sp 前移），
    # 覆盖 mm1 次槽、mm2_res 与主图 y/dy/nzp/sfmg/lse/cast/l0c_load_ub 的地址段。
    # 这些 tile 在 pre 阶段全部不活跃（pre 与主循环之间隔着 sync_all，见
    # fag_kernel），故地址可复用。
    #
    # 依赖方式（关键）：sp 的槽位 mutex 号**复用主图 tile 的号**（同址同号的
    # 保守形式），让“pre 最后一次 store → 主循环首次 load”有显式依赖，而不是
    # 完全独立的号。注意 4 个槽位各 52~54KB，必然跨多个主图 tile（单个主图
    # tile 只有 16~32KB），因此每个槽位只能绑定其覆盖范围内的一个主图 tile；
    # 其余重叠部分靠 sync_all 保证顺序。此布局必须上板验证。
    #
    # 三缓冲全双槽（slots=2）：slot 间独立 mutex，乒乓轮转（调用侧 .next()）
    # 让 load(N+1)/VF(N)/store(N-1) 三条流水真正并行。**不能退回 sp_y/sp_dy
    # 单槽**：pypto 对同一 tile（.current() 固定单槽）的跨队列操作不插 wait，
    # load(N+1) 会覆盖 VF(N) 正在读的窗口——上板复现 sfmg 大面积错值 +
    # 非逐位 + NaN（g2_batch_b8_n32_d256 / g2_nhead_n256_d768）。
    # **必须排在 pre/post 之后发号**：zero/f32/f16/sink_* 是跨档扁平导出，
    # 各档 id 必须一致，sp_* 只按档导出、放最后不影响前面的号序。
    sfmg_pre = {}
    if width >= 128:
        sp = _Arena(
            f"UB sfmg_pre {width}",
            _UB_BYTES,
            pool,
            base=UB_HEAD_RESERVE + 0x8000,
        )
        slot = SP_ROWS[width] * SP_WORK_W[width] * 2
        # sp_out 是 [rows,1] fp32，槽位字节数随行数变化（旧 512B 固定值在
        # rows>128 时会越界），按 rows*4 向上取 512B 对齐。
        out_bytes = max(_UB_ROW_SLOT, SP_ROWS[width] * 4)
        # 4 个槽位复用主图 tile 的号（y/dy/nzp/cast）——保守同号，
        # 保证 pre store 与主循环这些 tile 的首次 load 有显式依赖。
        sp_y = sp.buf("sp_y", slot, slots=2, ids=[y["mutex"][0], dy["mutex"][0]])
        sp_dy = sp.buf("sp_dy", slot, slots=2, ids=[nzp["mutex"][0], cast["mutex"][0]])
        sp_out = sp.buf(
            "sp_out",
            out_bytes,
            align=_UB_ROW_SLOT,
            ids=[sfmg["mutex"][0], lse["mutex"][0]],
        )
        sfmg_pre = {"y": sp_y, "dy": sp_dy, "out": sp_out, "_arenas": [sp]}

    # Wide inline SFMG keeps the original y/dy 32KB address ranges.
    # Each 8x768 half-precision slot uses 12KB within a 16KB reservation.
    # Allocate only the second-slot ids here, after the common cross-tier
    # prefix: no existing mutex id or UB address changes. The first slots
    # retain the mask aliases' ids; a full-width BN2 y alias owns both ids.
    sfmg_wide = {}
    if width >= 512:
        wide_y = _Arena(
            "UB sfmg wide y", y["addrs"][0] + 0x8000, pool, base=y["addrs"][0]
        )
        wide_dy = _Arena(
            "UB sfmg wide dy", dy["addrs"][0] + 0x8000, pool, base=dy["addrs"][0]
        )
        wide_y_buf = wide_y.buf(
            "sfmg_wide_y",
            0x4000,
            slots=2,
            ids=y["mutex"] + pool.take("sfmg_wide_y_slot1", 1),
        )
        wide_dy_buf = wide_dy.buf(
            "sfmg_wide_dy",
            0x4000,
            slots=2,
            ids=dy["mutex"] + pool.take("sfmg_wide_dy_slot1", 1),
        )
        assert 8 * width * 2 <= 0x4000
        assert 32 * width * 2 <= SP_ROWS[width] * SP_WORK_W[width] * 2
        sfmg_wide = {"y": wide_y_buf, "dy": wide_dy_buf, "_arenas": [wide_y, wide_dy]}

    return {
        "y": y,
        "dy": dy,
        "sfmg_wide": sfmg_wide,
        "nzp": nzp,
        "p_out": p_out,
        "sfmg": sfmg,
        "lse": lse,
        "sink_part": sink_part,
        "cast": cast,
        "l0c_load_ub": l0c_load_ub,
        "mm1": mm1,
        "mm2": mm2,
        "sfmg_pre": sfmg_pre,
        "zero": zero,
        "f32": f32,
        "f16": f16,
        "sink_row_post": sink_row_post,
        "sink_red_post": sink_red_post,
        "_arenas": [main, gs_out, shared, pre, post]
        + sfmg_pre.get("_arenas", [])
        + sfmg_wide.get("_arenas", []),
    }


# ==================================================================
#  分档铺排：每档一套 arena，cube / vector 各一个号池
# ==================================================================
_CUBE_POOL = {w: _MutexPool("cube", f"d_align={w}") for w in _TIERS}
_VEC_POOL = {w: _MutexPool("vector", f"d_align={w}") for w in _TIERS}

# The common CV prefix remains eight ids: increasing it would overflow the
# D192 cube pool. Only narrow tiers reserve the four otherwise unused ids.
for _w in (64, 128):
    for _pool in (_CUBE_POOL[_w], _VEC_POOL[_w]):
        _pool.reserve("ds_db1(CV)", [28])
        _pool.reserve("p_db1(CV)", [29])
        _pool.reserve("ds_t_db1(CV)", [30])
        _pool.reserve("p_t_db1(CV)", [31])

_L1 = {w: _build_l1(w, _CUBE_POOL[w]) for w in _TIERS}
_L0AB = {w: _build_l0ab(w, _CUBE_POOL[w]) for w in _TIERS}
_L0C = {}
for _w in _TIERS:
    if _w == 192:
        # 192 档要两套：GS1S2 用 acc 双槽（224KB），BN2 用 dK/dV 常驻（256KB）
        _L0C[(_w, "gs")] = _build_l0c(_w, _CUBE_POOL[_w], False)
        _L0C[(_w, "bn2")] = _build_l0c(_w, _CUBE_POOL[_w], True)
        _L0C[(_w, "gs_dv128")] = _build_l0c_d192dv128(
            _CUBE_POOL[_w], _L0C[(_w, "gs")], _L0C[(_w, "bn2")]
        )
    else:
        _lay = _build_l0c(_w, _CUBE_POOL[_w], False)
        _L0C[(_w, "gs")] = _lay
        _L0C[(_w, "bn2")] = _lay
_L0C_KEYS = list(_L0C.keys())
_UB = {w: _build_ub(w, _VEC_POOL[w]) for w in _TIERS}


# ---- L1 ----
L1_Q_ADDRS = _addrs(_L1, "q")
L1_Q_MUTEX = _mutex(_L1, "q")
L1_DO_ADDRS = _addrs(_L1, "do")
L1_DO_MUTEX = _mutex(_L1, "do")
L1_K_ADDR = _addr(_L1, "k")
MTX_L1_K = _mutex(_L1, "k")
L1_KND_ADDR = _addr(_L1, "k_nd")
MTX_L1_K_ND = _mutex(_L1, "k_nd")
# GS one-ahead pipelines keep at most current C2 and previous-task C3 live.
# Swizzle preload3 consumes C3(N-2) before C2(N), retaining its two-round delay
# without a third K slot. Empty deterministic rounds do not advance K cursors.
# Reuse the existing K/KND allocations and ids; no extra L1 bytes or mutex ids.
L1_K_REUSE_ADDRS = {w: _L1[w]["k"]["addrs"] + _L1[w]["k_nd"]["addrs"] for w in _TIERS}
L1_K_REUSE_MUTEX = {w: _L1[w]["k"]["mutex"] + _L1[w]["k_nd"]["mutex"] for w in _TIERS}
L1_V_ADDR = _addr(_L1, "v")
MTX_L1_V = _mutex(_L1, "v")
L1_DS_ADDR = _addr(_L1, "ds")
L1_P_ADDR = _addr(_L1, "p")
# CV 共享段的号钉死，三档一致，导出成扁平列表。
MTX_L1_DS = list(_CV_MTX["ds"])
MTX_L1_P = list(_CV_MTX["p"])
MTX_L1_DS_T = list(_CV_MTX["ds_t"])  # 与 ds 同址、另发一号，见 _CV_SLOTS
MTX_L1_P_T = list(_CV_MTX["p_t"])
L1_DS_DB_ADDRS = {w: [L1_DS_ADDR[w], _L1[w]["ds_db1"]["addrs"][0]] for w in (64, 128)}
L1_P_DB_ADDRS = {w: [L1_P_ADDR[w], _L1[w]["p_db1"]["addrs"][0]] for w in (64, 128)}
MTX_L1_DS_DB = [MTX_L1_DS[0], 28]
MTX_L1_P_DB = [MTX_L1_P[0], 29]
MTX_L1_DS_T_DB = [MTX_L1_DS_T[0], 30]
MTX_L1_P_T_DB = [MTX_L1_P_T[0], 31]

# ---- L0A / L0B ----
L0AB_ADDRS = _addrs(_L0AB, "l0a")
MTX_L0A = _mutex(_L0AB, "l0a")
MTX_L0B = _mutex(_L0AB, "l0b")


# ---- L0C ----
# 192 档两套布局：默认（GS1S2 / 其它档共用）= acc 双槽；_B2 = BN2 192 的
# dK/dV 常驻版。两套表都按 d_align 取，构造函数按编译期条件选表。
def _l0c_tabs(field, sub=None):
    d = {}
    b2 = {}
    for w in _TIERS:
        gs = _L0C[(w, "gs")][field]
        bn = _L0C[(w, "bn2")][field]
        d[w] = gs[sub] if sub else gs
        b2[w] = bn[sub] if sub else bn
    return d, b2


L0C_ACC_ADDRS, L0C_ACC_ADDRS_B2 = _l0c_tabs("acc", "addrs")
L0C_ACC_MUTEX, L0C_ACC_MUTEX_B2 = _l0c_tabs("acc", "mutex")
L0C_DK0_ADDR, L0C_DK0_ADDR_B2 = (
    {w: _L0C[(w, "gs")]["dk"]["addrs"][0] for w in _TIERS},
    {w: _L0C[(w, "bn2")]["dk"]["addrs"][0] for w in _TIERS},
)
L0C_DK1_ADDR, L0C_DK1_ADDR_B2 = (
    {
        w: _L0C[(w, "gs")]["dk"]["addrs"][len(_L0C[(w, "gs")]["dk"]["addrs"]) - 1]
        for w in _TIERS
    },
    {
        w: _L0C[(w, "bn2")]["dk"]["addrs"][len(_L0C[(w, "bn2")]["dk"]["addrs"]) - 1]
        for w in _TIERS
    },
)
L0C_DV0_ADDR, L0C_DV0_ADDR_B2 = (
    {w: _L0C[(w, "gs")]["dv"]["addrs"][0] for w in _TIERS},
    {w: _L0C[(w, "bn2")]["dv"]["addrs"][0] for w in _TIERS},
)
L0C_DV1_ADDR, L0C_DV1_ADDR_B2 = (
    {
        w: _L0C[(w, "gs")]["dv"]["addrs"][len(_L0C[(w, "gs")]["dv"]["addrs"]) - 1]
        for w in _TIERS
    },
    {
        w: _L0C[(w, "bn2")]["dv"]["addrs"][len(_L0C[(w, "bn2")]["dv"]["addrs"]) - 1]
        for w in _TIERS
    },
)
MTX_L0C_DK0, MTX_L0C_DK0_B2 = _l0c_tabs("dk", "mutex")
MTX_L0C_DK0 = {w: MTX_L0C_DK0[w][0] for w in _TIERS}
MTX_L0C_DK0_B2 = {w: MTX_L0C_DK0_B2[w][0] for w in _TIERS}
MTX_L0C_DV0, MTX_L0C_DV0_B2 = _l0c_tabs("dv", "mutex")
MTX_L0C_DV0 = {w: MTX_L0C_DV0[w][0] for w in _TIERS}
MTX_L0C_DV0_B2 = {w: MTX_L0C_DV0_B2[w][0] for w in _TIERS}
MTX_L0C_DK1, MTX_L0C_DK1_B2 = _l0c_tabs("dk", "mutex")
MTX_L0C_DK1 = {w: MTX_L0C_DK1[w][-1] for w in _TIERS}
MTX_L0C_DK1_B2 = {w: MTX_L0C_DK1_B2[w][-1] for w in _TIERS}
MTX_L0C_DV1, MTX_L0C_DV1_B2 = _l0c_tabs("dv", "mutex")
MTX_L0C_DV1 = {w: MTX_L0C_DV1[w][-1] for w in _TIERS}
MTX_L0C_DV1_B2 = {w: MTX_L0C_DV1_B2[w][-1] for w in _TIERS}
L0C_ACC_D192DV128_ADDRS = _L0C[(192, "gs_dv128")]["acc"]["addrs"]
L0C_ACC_D192DV128_MUTEX = _L0C[(192, "gs_dv128")]["acc"]["mutex"]
L0C_ACC_D192DV128_N64_ADDR = L0C_ACC_D192DV128_ADDRS[0] + CUBE_BASEM * 128 * 4
L0C_DK_D192DV128_ADDRS = _L0C[(192, "gs_dv128")]["dk"]["addrs"]
L0C_DK_D192DV128_MUTEX = _L0C[(192, "gs_dv128")]["dk"]["mutex"]
L0C_DK_D192DV128_N64_ADDR = L0C_DK_D192DV128_ADDRS[0] + CUBE_BASEN * 128 * 4
L0C_DV128_ADDR = _L0C[(192, "gs_dv128")]["dv"]["addrs"][0]
MTX_L0C_DV128 = _L0C[(192, "gs_dv128")]["dv"]["mutex"][0]

# ---- UB ----
# vector 侧的 buffer 名单与档无关，三档发出的号必然一致，故导出扁平列表。
UB_MM1_0, UB_MM1_1 = _UB[HEAD_DIM_ALIGN]["mm1"]["addrs"]
UB_MM2_0, UB_MM2_1 = _UB[HEAD_DIM_ALIGN]["mm2"]["addrs"]
MTX_MM1_RES = list(_CV_MTX["mm1_res"])
MTX_MM2_RES = list(_CV_MTX["mm2_res"])

UB_Y_A = _addr(_UB, "y")
UB_DY_A = _addr(_UB, "dy")
UB_NZP_A = _addr(_UB, "nzp")
UB_P_OUT_A = _addr(_UB, "p_out")  # GS only: BN2 scratch is inactive
UB_SFMG_A = _addr(_UB, "sfmg")
UB_LSE0_A = {w: _UB[w]["lse"]["addrs"][0] for w in _TIERS}
UB_LSE1_A = {w: _UB[w]["lse"]["addrs"][1] for w in _TIERS}
# attenmask 复用 y/dy 的地址与号（vec1 之后 vec2 再用）。
UB_MASK_A = UB_Y_A
UB_MASK_PRE_A = UB_DY_A
MTX_UB_Y = _flat(_UB, "y")  # attenmask 复用
MTX_UB_DY = _flat(_UB, "dy")  # attenmask(pre) 复用
MTX_UB_NZP = _flat(_UB, "nzp")
MTX_UB_P_OUT = _flat(_UB, "p_out")
MTX_UB_SFMG = _flat(_UB, "sfmg")
MTX_UB_LSE = _flat(_UB, "lse")
MTX_UB_CAST = _flat(_UB, "cast")  # 仅 BN2
MTX_UB_L0C_LOAD = _flat(_UB, "l0c_load_ub")  # 仅 BN2 192
# 仅 wide：载入 tile 复用 y、出口复用 cast（同址必须同号）。
MTX_UB_WSF32 = _flat(_UB, "y")
# Both wide keys reserve the same two 16KiB slots; only the declared row
# pitch differs (512 or 768). The flat aliases below must own the same ids.
for _wide_field in ("y", "dy"):
    assert _UB[512]["sfmg_wide"][_wide_field] == _UB[768]["sfmg_wide"][_wide_field]
UB_SFMG_WIDE_Y_A = _UB[768]["sfmg_wide"]["y"]["addrs"]
UB_SFMG_WIDE_DY_A = _UB[768]["sfmg_wide"]["dy"]["addrs"]
MTX_UB_SFMG_WIDE_Y = _UB[768]["sfmg_wide"]["y"]["mutex"]
MTX_UB_SFMG_WIDE_DY = _UB[768]["sfmg_wide"]["dy"]["mutex"]
# One full-range alias tile must wait for both independently rotating slots.
MTX_UB_WSF32_WIDE = [MTX_UB_SFMG_WIDE_Y]

MTX_UB_WSF16 = _flat(_UB, "cast")
UB_SINK_PART_A = _addr(_UB, "sink_part")
MTX_UB_SINK_PART = _flat(_UB, "sink_part")
# sink_row_post 是 SINK_REDUCE_BATCH 个独立的 1 行 tile（不是单个 buffer）
UB_SINK_ROW_POST_A = {
    w: [b["addrs"][0] for b in _UB[w]["sink_row_post"]] for w in _TIERS
}
UB_SINK_RED_POST_A = _addr(_UB, "sink_red_post")
MTX_UB_SINK_ROW_POST = [
    m for b in _UB[HEAD_DIM_ALIGN]["sink_row_post"] for m in b["mutex"]
]
MTX_UB_SINK_RED_POST = _flat(_UB, "sink_red_post")

UB_PRE_ZERO = _UB[HEAD_DIM_ALIGN]["zero"]["addrs"][0]

# sfmg 前移独立缓冲（仅 64 档不存在；构造时 sfmg_pre 为空 dict）。
# 全档 double buffer：导出完整地址/mutex 列表（[slot0, slot1]），
# 与 post_f32/post_f16 的 POST_F32_ADDRS / MTX_UB_POST_F32 同款口径。
UB_SP_Y_A = {w: _UB[w]["sfmg_pre"]["y"]["addrs"] for w in _TIERS if w >= 128}
UB_SP_DY_A = {w: _UB[w]["sfmg_pre"]["dy"]["addrs"] for w in _TIERS if w >= 128}
UB_SP_OUT_A = {w: _UB[w]["sfmg_pre"]["out"]["addrs"] for w in _TIERS if w >= 128}
MTX_UB_SP_Y = {w: _UB[w]["sfmg_pre"]["y"]["mutex"] for w in _TIERS if w >= 128}
MTX_UB_SP_DY = {w: _UB[w]["sfmg_pre"]["dy"]["mutex"] for w in _TIERS if w >= 128}
MTX_UB_SP_OUT = {w: _UB[w]["sfmg_pre"]["out"]["mutex"] for w in _TIERS if w >= 128}
MTX_UB_ZERO = _flat(_UB, "zero")
POST_F32_ADDRS = _addrs(_UB, "f32")
POST_F16_ADDRS = _addrs(_UB, "f16")
MTX_UB_POST_F32 = _flat(_UB, "f32")
MTX_UB_POST_F16 = _flat(_UB, "f16")


# ==================================================================
#  模板相关的 UB 槽位 / 别名
# ==================================================================
# BN2 narrow（d_align 64/128）的 V1 与 GS1S2 共用 softmax_grad_front_row_vf，
# 且 y/dy 最宽就是 [64,128]，所以**整张图直接沿用 v1w=128 那一份**：调用点给
# 工厂传 ``HEAD_DIM_ALIGN``，同地址、同号。不能混用按档取地址的图：
# d=64 档的 cast 会与 128 档的 nzp 重叠。narrow 与 wide 每个 tilingkey 只编
# 一个 template，两套用法不会同时活跃。
CAST_W = VECTOR_BASEN  # 16-bit cast 出口的列宽（整块 [64,128]）
UB_L0C_LOAD_A = _addr(_UB, "l0c_load_ub")  # dK chunk1 的 L0C→UB 搬入槽
UB_CAST_A = _addr(_UB, "cast")  # 16-bit cast 出口（按档取图）
UB_WSF32 = _addr(_UB, "y")  # wide 的 ws 载入 tile alias 到 y（V2 之后已死）
UB_WSF16 = _addr(_UB, "cast")  # wide 的 16-bit 出口 alias 到 cast


# ==================================================================
#  自查
# ==================================================================
def describe_layout(tier=HEAD_DIM_ALIGN, dv_tier=None):
    """打印一档的完整片上内存图：地址 + 大小 + 持有的 buffer-id。"""
    arenas = []
    for group in (_L1, _L0AB, _UB):
        arenas.extend(group[tier]["_arenas"])
    # L0C 表按 (档, 类) 取，默认看 GS1S2 那套；BN2 192 的另一套用 _L0C[(192,"bn2")]
    variant = "gs_dv128" if tier == 192 and dv_tier == 128 else "gs"
    arenas.extend(_L0C[(tier, variant)]["_arenas"])
    return "\n\n".join(a.describe() for a in arenas)


def audit_mutex(tier=HEAD_DIM_ALIGN):
    """打印一档的 buffer-id 占用：每个号归谁、每侧还剩几个。"""
    return _CUBE_POOL[tier].report() + "\n\n" + _VEC_POOL[tier].report()


# ==================================================================
# BN2S2-specific implementation: names are isolated; resource policy is retained.
# ==================================================================

# --- generated imports (fag_fiximports.py) ---
from fag_common import (
    CUBE_BASEM_bn2s2,
    CUBE_BASEN_bn2s2,
    D_CHUNK_MAXW_bn2s2,
    D_NCHUNKS_bn2s2,
    D_PHYS_W_bn2s2,
    HEAD_DIM_ALIGN_bn2s2,
    NZP_ROWS_bn2s2,
    VECTOR_BASEM_bn2s2,
    VECTOR_BASEN_bn2s2,
)
# --- end generated imports ---

from memory.fag_layout import _ArenaBase, _MutexPoolBase, _addr, _addrs, _mutex


# d_align 的分档。所有地址表与 id 表都以它为键。
_TIERS_bn2s2 = (64, 128, 192, 256, 512, 768)

_L1_BYTES_bn2s2 = 512 * 1024
_L0AB_BYTES_bn2s2 = 64 * 1024
_L0C_BYTES_bn2s2 = 256 * 1024
_UB_BYTES_bn2s2 = 256 * 1024


# ==================================================================
#  buffer-id (mutex) 分配
# ==================================================================
# id 空间是**扁平的 [0, 32)，不按内存空间分**：L1 / L0A / L0B / L0C / UB 共用
# 同一批号。生成代码里就是 get_buf(PIPE_x, id, 0)，除了 pipe 和 id 再没有第三个
# 维度。唯一的分离维度是**核**：cube 与 vector 是两颗物理核，各有一套 32 个
# token，同号互不相干。
#
# CV 共享的 6 块 buffer 声明在两个 section 之外，只有一份 mutex_ids，两侧都要
# 用，所以必须两侧同号。把它们钉在号段最前面，两个池子都从这段之后开始发号，
# 谁也不会踩到 —— 这样 cube 池和 vector 池就能各自独立编号。
_CV_SLOTS_bn2s2 = {
    "mm1_res": 2,  # cube FIX 写、vector V 读
    "mm2_res": 2,
    "ds": 1,  # vector 写、cube mm3/mm4 读
    "p": 1,
    # dS^T / P^T 与 ds / p 同址但**另发一个号**：它们是同一块 L1 的 ZN 视图，
    # 有独立游标，读写序由 CV flag(SYNC_V3_TO_C3 等)而不是 mutex 保证。
    "ds_t": 1,
    "p_t": 1,
}


_CV_MTX_bn2s2, _CV_RESERVED_bn2s2 = _build_cv_mutex(_CV_SLOTS_bn2s2)


class _MutexPool_bn2s2(_MutexPoolBase):
    """一侧核的 buffer-id 顺序发号器。

    一律顺序发号、绝不复用：id 只要单射就正确，而「同一地址挂了不同号」才是
    真正的 bug（互斥直接消失，且不会当场报错）。别名视图不在这里申请号，它们
    直接引用被别名那块 buffer 的记录，同号是结构保证而非人为约定。
    """

    def __init__(self, side, tag):
        super().__init__(side, tag, _CV_RESERVED_bn2s2, _CV_MTX_bn2s2)

    def take(self, label, count):
        return self._take(label, count, check_reserved=False)


# ==================================================================
#  片上内存的 bump 分配器
# ==================================================================
class _Arena_bn2s2(_ArenaBase):
    """一块片上内存的顺序铺排。

    只在 import 期执行，绝不进 trace：导出的永远是 int / list[int] 的平表，
    因为 tile 的 ``addrs=`` / ``mutex_ids=`` 只接受 trace 期常量。

    ``capacity`` 是**绝对上限**（含 ``base`` 之前的保留区），越界即断言失败。
    """

    def buf(self, label, nbytes, slots=1, align=1, cv=None):
        return self._declare_buffer(label, nbytes, slots, align, cv, _CV_MTX_bn2s2)


# ==================================================================
#  Cube 侧：L1 (512KB) + L0A/L0B (各 64KB) + L0C (256KB)
# ==================================================================
def _build_l1_bn2s2(width, pool):
    """铺排一档的 L1，返回 ``{名字: buffer}``。

    Q/dO 是多槽 double buffer：tile_group 填多个 addrs + 多个 mutex_id 后，
    框架会自动插核内同步，使下一轮 load 与本轮 matmul 重叠。
    槽位数不是 2 而是 3+：cube 领先 vector 两个 task，mm2 在
    task N 载入 Q_N，而 mm4 要的是 Q_{N-2}；3 槽时 N%3 != (N-2)%3，两者不会
    互相覆写，于是 mm4/mm5 可以直接复用 mm2/mm1 搬好的数据，无需再读 GM。
    """
    ar = _Arena_bn2s2(f"L1 d_align={width}", _L1_BYTES_bn2s2, pool)
    if width == 768:
        # Wide D has no cross-matmul Q/dY retention in ASC. Reuse two
        # physical common buffers, with shared IDs for every alias.
        a = ar.buf("common_a", CUBE_BASEM_bn2s2 * 768 * 2)
        b = ar.buf("common_b", CUBE_BASEM_bn2s2 * 768 * 2)
        ds = ar.buf("ds", CUBE_BASEM_bn2s2 * CUBE_BASEN_bn2s2 * 2, cv="ds")
        p = ar.buf("p", CUBE_BASEM_bn2s2 * CUBE_BASEN_bn2s2 * 2, cv="p")
        return {
            "q": a,
            "do": a,
            "k": b,
            "k_nd": b,
            "v": b,
            "ds": ds,
            "p": p,
            "_arenas": [ar],
        }
    blk = (
        CUBE_BASEM_bn2s2 * D_PHYS_W_bn2s2[width] * 2
    )  # 64/128 档 32KB，192 档 48KB，256/768 档 64KB
    dsp = CUBE_BASEM_bn2s2 * CUBE_BASEN_bn2s2 * 2  # dS/P 是 [M,N]，与 D 无关，恒 32KB
    # 64/128 档 4 槽；192 档 3 槽；256/768 档 1-ahead 只要 2 槽（物理宽 256）。
    if width <= HEAD_DIM_ALIGN_bn2s2:
        nq = 4
    elif width == 192:
        nq = 3
    else:
        nq = 2

    q = ar.buf("q", blk, slots=nq)  # q_lag_view 复用这份记录，同址同号
    do = ar.buf("do", blk, slots=nq)  # do_lag_view 同理
    k = ar.buf("k", blk)
    # Legacy QK/dQ loads use separate physical K slots. The normal D<512
    # preload chain reuses these as a two-slot ring through make_plan_k_l1
    # and make_plan_k_normal_l1: same NZ bytes/mutexes, paired view cursors.
    # Do not alias independently loaded layouts without that full contract.
    k_nd = ar.buf("k_nd", blk)
    v = ar.buf("v", blk)
    # dS^T / P^T 不单独占地址，是 ds / p 上的 ZN 视图（见 make_ds_t_l1）。
    ds = ar.buf("ds", dsp, cv="ds")
    p = ar.buf("p", dsp, cv="p")

    return {
        "q": q,
        "do": do,
        "k": k,
        "k_nd": k_nd,
        "v": v,
        "ds": ds,
        "p": p,
        "_arenas": [ar],
    }


# acc / L0A / L0B 的列宽：MM2/MM1 输出 [M,s2]，MM3 输出 dQ 的一个 D 分块，
# 取两者较大值，一块 tile 同时够用。三档都等于 128。
ACC_COLS_bn2s2 = {w: max(CUBE_BASEN_bn2s2, D_CHUNK_MAXW_bn2s2[w]) for w in _TIERS_bn2s2}


def _build_l0c_bn2s2(width, pool):
    """L0C：acc 槽位 + dK/dV 各 D 分块的常驻块。

    dK/dV 要在同一 s2 块内跨 s1 轮次累加（IS_DKV_RESIDENT_L0C），s2 切换时才
    fixpipe 出 GM，所以每个 D 分块都得有自己的常驻 L0C —— 不能像 acc 那样复用，
    故一个分块一个号。

    acc 单缓冲时 fixpipe 写出期间下一个 matmul 只能干等（实测 fixpipe 0.458
    而 mac 仅 0.783），故 64/128 档开两槽；192 档预算被 dK/dV 分块吃掉，退回
    单槽。64/128/192 都正好用满 256KB；256 档 acc 双缓冲 + dk/dv 各 1 块填表。
    """
    ar = _Arena_bn2s2(f"L0C d_align={width}", _L0C_BYTES_bn2s2, pool)
    accBlk = CUBE_BASEM_bn2s2 * ACC_COLS_bn2s2[width] * 4  # acc 要装 MM2/MM1 的 [M,s2]
    dkvBlk = CUBE_BASEN_bn2s2 * D_CHUNK_MAXW_bn2s2[width] * 4  # dK/dV 只装一个 D 分块
    nChunk = D_NCHUNKS_bn2s2[width]

    # 256/768 档 C4/C5 不驻留：每 task atomic 出 GM，只要 acc 双缓冲。仍留 1 块
    # dk/dv scratch 填表，主循环不引用。
    accSlots = 1 if width == 192 else 2
    nPhys = 1 if width >= 256 else nChunk
    acc = ar.buf("acc", accBlk, slots=accSlots)
    dk = ar.buf("dk", dkvBlk, slots=nPhys)
    dv = ar.buf("dv", dkvBlk, slots=nPhys)
    return {"acc": acc, "dk": dk, "dv": dv, "_arenas": [ar]}


# ==================================================================
#  Vector 侧：UB (256KB)
# ==================================================================
# 头部留给 codegen 的 UB[0] tiling 中转：BN2 没有 pre/sync_all，不留这块会和
# cube 的 fixpipe 抢 UB[0]。所有 UB arena 都从它之后开始。
UB_HEAD_RESERVE_bn2s2 = 0x1000
# [64,1] fp32 只有 256B，但按 512B 一格排，保证后面的槽位对齐。
_UB_ROW_SLOT_bn2s2 = 0x200

# pre/post 走一维元素，不按 [行, D] 切。三档都吃满可用 UB，且每档
# elems 都能被 128 整除，full-tile 的 VF 段数是编译期常量。行区间已由 metadata
# 按 nAiv 均分，块大不再空核。
PRE_ELEMS_bn2s2 = {
    64: 63488,
    128: 63488,
    192: 63360,
    256: 63360,
    512: 63360,
    768: 63360,
}  # 单槽 fp32
POST_ELEMS_bn2s2 = {
    64: 20480,
    128: 21120,
    192: 21120,
    256: 21120,
    512: 21120,
    768: 21120,
}  # depth-2，fp32 + 16-bit


def _build_ub_bn2s2(width, pool):
    """一档的 vector 侧全部 UB。

    三张图共用这 256KB，靠执行阶段错开：

    * 主图（y/dy/nzp/sfmg/lse/cast）—— 主循环用，从 0x21000 起；
    * mm1/mm2 结果区 —— CV 共享，cube fixpipe 写、vector 读，从头部起；
    * pre/post —— 与主循环被 sync_all 完全隔开，整块借用头部。

    V1 一趟 VECTOR_BASEM 行、按行 reduce，不再开 prod/tmp。列宽
    ``v1w`` 就是 ``dv_align``（与 ``d_align`` 相同，仅 D=192 且 Dv=128 时为 128）。
    768 档 y/dy 仍按 256 列分配，V1 三趟 VF256。
    attenmask 复用 y/dy 这两条记录：uint8[64,128]=8KB，而 y/dy 在 v1w>=64
    时至少 8KB，装得下。
    复用的是**同一条记录**，所以「同址必须同号」是结构保证，不靠人守。

    cast 只有 BN2 的 v5/v6 用，但三档都留：它排在最后，不占别人的地方，
    留着可以让 BN2 直接复用这张图（见 BN2_UB_*）。
    """
    main = _Arena_bn2s2(
        f"UB v1w={width}", _UB_BYTES_bn2s2, pool, base=UB_HEAD_RESERVE_bn2s2 + 0x20000
    )
    v1Alloc = 256 if width >= 512 else width
    y = main.buf("y", VECTOR_BASEM_bn2s2 * v1Alloc * 2)
    dy = main.buf("dy", VECTOR_BASEM_bn2s2 * v1Alloc * 2)
    # NZ+1 行：末行是 bank 错开的填充，见 fag_common.NZP_BLOCK_STRIDE。
    nzp = main.buf("nzp", NZP_ROWS_bn2s2 * VECTOR_BASEN_bn2s2 * 2)
    sfmg = main.buf("sfmg", _UB_ROW_SLOT_bn2s2, align=_UB_ROW_SLOT_bn2s2)
    lse = main.buf("lse", _UB_ROW_SLOT_bn2s2, slots=2)
    sink_part = main.buf("sink_part", _UB_ROW_SLOT_bn2s2, align=_UB_ROW_SLOT_bn2s2)
    cast = main.buf("cast", VECTOR_BASEM_bn2s2 * VECTOR_BASEN_bn2s2 * 2)

    shared = _Arena_bn2s2(
        "UB mm1/mm2", _UB_BYTES_bn2s2, pool, base=UB_HEAD_RESERVE_bn2s2
    )
    resBlk = VECTOR_BASEM_bn2s2 * VECTOR_BASEN_bn2s2 * 4  # fp32[64,128] = 32KB
    mm1 = shared.buf("mm1_res", resBlk, cv="mm1_res")
    mm2 = shared.buf("mm2_res", resBlk, cv="mm2_res")

    # pre 的 16-bit 支（BN2 need_zero）只占一半，按 fp32 算上限。
    pre = _Arena_bn2s2(
        f"UB pre d_align={width}", _UB_BYTES_bn2s2, pool, base=UB_HEAD_RESERVE_bn2s2
    )
    zero = pre.buf("pre_zero", PRE_ELEMS_bn2s2[width] * 4)

    # post 的 fp32 输入与 16-bit 输出各用一套号，框架据此判定「下一块的 load」与
    # 「本块的 cast/store」无冲突，从而让 MTE2 / V / MTE3 三条流水重叠。
    post = _Arena_bn2s2(
        f"UB post d_align={width}", _UB_BYTES_bn2s2, pool, base=UB_HEAD_RESERVE_bn2s2
    )
    n = POST_ELEMS_bn2s2[width]
    f32 = post.buf("post_f32", n * 4, slots=2)
    f16 = post.buf("post_f16", n * 2, slots=2)
    # Post-only rows: batch loads, then preserve the original rowwise add order.
    sink_row_post = [
        post.buf(f"sink_row_post{i}", _UB_ROW_SLOT_bn2s2, align=_UB_ROW_SLOT_bn2s2)
        for i in range(SINK_REDUCE_BATCH)
    ]
    sink_red_post = post.buf(
        "sink_red_post", _UB_ROW_SLOT_bn2s2, align=_UB_ROW_SLOT_bn2s2
    )

    return {
        "y": y,
        "dy": dy,
        "nzp": nzp,
        "sfmg": sfmg,
        "lse": lse,
        "sink_part": sink_part,
        "cast": cast,
        "mm1": mm1,
        "mm2": mm2,
        "zero": zero,
        "f32": f32,
        "f16": f16,
        "sink_row_post": sink_row_post,
        "sink_red_post": sink_red_post,
        "_arenas": [main, shared, pre, post],
    }


# ==================================================================
#  分档铺排：每档一套 arena，cube / vector 各一个号池
# ==================================================================
_CUBE_POOL_bn2s2 = {w: _MutexPool_bn2s2("cube", f"d_align={w}") for w in _TIERS_bn2s2}
_VEC_POOL_bn2s2 = {w: _MutexPool_bn2s2("vector", f"d_align={w}") for w in _TIERS_bn2s2}

_L1_bn2s2 = {w: _build_l1_bn2s2(w, _CUBE_POOL_bn2s2[w]) for w in _TIERS_bn2s2}
_L0AB_bn2s2 = {
    w: _build_l0ab(
        w, _CUBE_POOL_bn2s2[w], ACC_COLS_bn2s2[w], _Arena_bn2s2, _L0AB_BYTES_bn2s2
    )
    for w in _TIERS_bn2s2
}
_L0C_bn2s2 = {w: _build_l0c_bn2s2(w, _CUBE_POOL_bn2s2[w]) for w in _TIERS_bn2s2}
_UB_bn2s2 = {w: _build_ub_bn2s2(w, _VEC_POOL_bn2s2[w]) for w in _TIERS_bn2s2}


# ---- L1 ----
L1_Q_ADDRS_bn2s2 = _addrs(_L1_bn2s2, "q")
L1_Q_MUTEX_bn2s2 = _mutex(_L1_bn2s2, "q")
L1_DO_ADDRS_bn2s2 = _addrs(_L1_bn2s2, "do")
L1_DO_MUTEX_bn2s2 = _mutex(_L1_bn2s2, "do")
L1_K_ADDR_bn2s2 = _addr(_L1_bn2s2, "k")
MTX_L1_K_bn2s2 = _mutex(_L1_bn2s2, "k")
L1_KND_ADDR_bn2s2 = _addr(_L1_bn2s2, "k_nd")
MTX_L1_K_ND_bn2s2 = _mutex(_L1_bn2s2, "k_nd")
L1_V_ADDR_bn2s2 = _addr(_L1_bn2s2, "v")
MTX_L1_V_bn2s2 = _mutex(_L1_bn2s2, "v")
L1_DS_ADDR_bn2s2 = _addr(_L1_bn2s2, "ds")
L1_P_ADDR_bn2s2 = _addr(_L1_bn2s2, "p")
# CV 共享段的号钉死，三档一致，导出成扁平列表。
MTX_L1_DS_bn2s2 = list(_CV_MTX_bn2s2["ds"])
MTX_L1_P_bn2s2 = list(_CV_MTX_bn2s2["p"])
MTX_L1_DS_T_bn2s2 = list(_CV_MTX_bn2s2["ds_t"])  # 与 ds 同址、另发一号，见 _CV_SLOTS
MTX_L1_P_T_bn2s2 = list(_CV_MTX_bn2s2["p_t"])

# ---- L0A / L0B ----
L0AB_ADDRS_bn2s2 = _addrs(_L0AB_bn2s2, "l0a")
MTX_L0A_bn2s2 = _mutex(_L0AB_bn2s2, "l0a")
MTX_L0B_bn2s2 = _mutex(_L0AB_bn2s2, "l0b")

# ---- L0C ----
L0C_ACC_ADDRS_bn2s2 = _addrs(_L0C_bn2s2, "acc")
L0C_ACC_MUTEX_bn2s2 = _mutex(_L0C_bn2s2, "acc")
# 每个 D 分块单独取一个标量地址/id。不能在 kernel 里写 `TABLE[d_align][-1]`：
# parser 不支持负下标（报 "GetItemExpr index -1 out of bounds"）。
_DK_bn2s2 = {w: _L0C_bn2s2[w]["dk"] for w in _TIERS_bn2s2}
_DV_bn2s2 = {w: _L0C_bn2s2[w]["dv"] for w in _TIERS_bn2s2}
L0C_DK0_ADDR_bn2s2 = {w: _DK_bn2s2[w]["addrs"][0] for w in _TIERS_bn2s2}
L0C_DV0_ADDR_bn2s2 = {w: _DV_bn2s2[w]["addrs"][0] for w in _TIERS_bn2s2}
MTX_L0C_DK0_bn2s2 = {w: _DK_bn2s2[w]["mutex"][0] for w in _TIERS_bn2s2}
MTX_L0C_DV0_bn2s2 = {w: _DV_bn2s2[w]["mutex"][0] for w in _TIERS_bn2s2}
# 第二个分块。单块档位下这些 tile 永不被真的使用（调用路径被 trace 期的
# `if D_NCHUNKS > 1` 整块剪掉），故让它别名到第 0 块即可。
L0C_DK1_ADDR_bn2s2 = {
    w: _DK_bn2s2[w]["addrs"][len(_DK_bn2s2[w]["addrs"]) - 1] for w in _TIERS_bn2s2
}
L0C_DV1_ADDR_bn2s2 = {
    w: _DV_bn2s2[w]["addrs"][len(_DV_bn2s2[w]["addrs"]) - 1] for w in _TIERS_bn2s2
}
MTX_L0C_DK1_bn2s2 = {
    w: _DK_bn2s2[w]["mutex"][len(_DK_bn2s2[w]["mutex"]) - 1] for w in _TIERS_bn2s2
}
MTX_L0C_DV1_bn2s2 = {
    w: _DV_bn2s2[w]["mutex"][len(_DV_bn2s2[w]["mutex"]) - 1] for w in _TIERS_bn2s2
}

# Ordinary BN2S2 D256 retains dV in two 64 KiB blocks. Keep the existing
# DK/DV records and their IDs unchanged: the deterministic plan already uses
# them as distinct resources. Only the ordinary factory aliases the unused
# DK scratch as dV chunk 0; chunk 1 keeps the original DV scratch. C4 uses
# the separate common accumulator, and runtime GS fallback does not consume
# these resident blocks for D256.
NORM_DV0_ADDR_bn2s2 = dict(L0C_DV0_ADDR_bn2s2)
NORM_DV1_ADDR_bn2s2 = dict(L0C_DV1_ADDR_bn2s2)
NORM_DV0_MUTEX_bn2s2 = dict(MTX_L0C_DV0_bn2s2)
NORM_DV1_MUTEX_bn2s2 = dict(MTX_L0C_DV1_bn2s2)
NORM_DV0_ADDR_bn2s2[256] = L0C_DK0_ADDR_bn2s2[256]
NORM_DV0_MUTEX_bn2s2[256] = MTX_L0C_DK0_bn2s2[256]
assert NORM_DV0_ADDR_bn2s2[256] + 128 * 128 * 4 <= NORM_DV1_ADDR_bn2s2[256]
assert NORM_DV1_ADDR_bn2s2[256] + 128 * 128 * 4 <= _L0C_BYTES_bn2s2
assert NORM_DV0_MUTEX_bn2s2[256] != NORM_DV1_MUTEX_bn2s2[256]

# BN2 不切 D，但它只编 64/128 两档，那两档 D_NCHUNKS 本来就是 1，所以直接用
# 上面的第 0 块即可，不需要另一套常量。

# ---- UB ----
# vector 侧的 buffer 名单与档无关，三档发出的号必然一致，故导出扁平列表。
UB_MM1_0_bn2s2, UB_MM1_1_bn2s2 = _UB_bn2s2[HEAD_DIM_ALIGN_bn2s2]["mm1"]["addrs"]
UB_MM2_0_bn2s2, UB_MM2_1_bn2s2 = _UB_bn2s2[HEAD_DIM_ALIGN_bn2s2]["mm2"]["addrs"]
MTX_MM1_RES_bn2s2 = list(_CV_MTX_bn2s2["mm1_res"])
MTX_MM2_RES_bn2s2 = list(_CV_MTX_bn2s2["mm2_res"])

UB_Y_A_bn2s2 = _addr(_UB_bn2s2, "y")
UB_DY_A_bn2s2 = _addr(_UB_bn2s2, "dy")
UB_NZP_A_bn2s2 = _addr(_UB_bn2s2, "nzp")
UB_SFMG_A_bn2s2 = _addr(_UB_bn2s2, "sfmg")
UB_LSE0_A_bn2s2 = {w: _UB_bn2s2[w]["lse"]["addrs"][0] for w in _TIERS_bn2s2}
UB_LSE1_A_bn2s2 = {w: _UB_bn2s2[w]["lse"]["addrs"][1] for w in _TIERS_bn2s2}
UB_CAST_A_bn2s2 = _addr(_UB_bn2s2, "cast")
# attenmask 复用 y/dy 的地址与号（vec1 之后 vec2 再用）。
UB_MASK_A_bn2s2 = UB_Y_A_bn2s2
UB_MASK_PRE_A_bn2s2 = UB_DY_A_bn2s2
MTX_UB_Y_bn2s2 = _flat(_UB_bn2s2, "y")  # attenmask 复用
MTX_UB_DY_bn2s2 = _flat(_UB_bn2s2, "dy")  # attenmask(pre) 复用
MTX_UB_NZP_bn2s2 = _flat(_UB_bn2s2, "nzp")
MTX_UB_SFMG_bn2s2 = _flat(_UB_bn2s2, "sfmg")
MTX_UB_LSE_bn2s2 = _flat(_UB_bn2s2, "lse")
MTX_UB_CAST_bn2s2 = _flat(_UB_bn2s2, "cast")  # 仅 BN2
UB_SINK_PART_A_bn2s2 = _addr(_UB_bn2s2, "sink_part")
MTX_UB_SINK_PART_bn2s2 = _flat(_UB_bn2s2, "sink_part")
UB_SINK_ROW_POST_A_bn2s2 = {
    w: [b["addrs"][0] for b in _UB_bn2s2[w]["sink_row_post"]] for w in _TIERS
}
UB_SINK_RED_POST_A_bn2s2 = _addr(_UB_bn2s2, "sink_red_post")
MTX_UB_SINK_ROW_POST_bn2s2 = [
    m for b in _UB_bn2s2[HEAD_DIM_ALIGN_bn2s2]["sink_row_post"] for m in b["mutex"]
]
MTX_UB_SINK_RED_POST_bn2s2 = _flat(_UB_bn2s2, "sink_red_post")

UB_PRE_ZERO_bn2s2 = _UB_bn2s2[HEAD_DIM_ALIGN_bn2s2]["zero"]["addrs"][0]
MTX_UB_ZERO_bn2s2 = _flat(_UB_bn2s2, "zero")
POST_F32_ADDRS_bn2s2 = _addrs(_UB_bn2s2, "f32")
POST_F16_ADDRS_bn2s2 = _addrs(_UB_bn2s2, "f16")
MTX_UB_POST_F32_bn2s2 = _flat(_UB_bn2s2, "f32")
MTX_UB_POST_F16_bn2s2 = _flat(_UB_bn2s2, "f16")


# ==================================================================
#  BN2 模板专用的 UB 槽位
# ==================================================================
# BN2 的 V1 与 GS1S2 共用 softmax_grad_front_row_vf，且只在 d_align 64/128 两档
# 编译（y/dy 最宽就是 [64,128]），所以**整张图直接沿用 v1w=128 那一份** ——
# 同地址、同号、连 cast 都是同一块。每个 tilingkey 只编一个 template，两套用法
# 不会同时活跃。
BN2_V1_ROWS_bn2s2 = VECTOR_BASEM_bn2s2
BN2_CAST_W_bn2s2 = VECTOR_BASEN_bn2s2
_BN2_bn2s2 = _UB_bn2s2[HEAD_DIM_ALIGN_bn2s2]
BN2_UB_Y_bn2s2 = _BN2_bn2s2["y"]["addrs"][0]
BN2_UB_DY_bn2s2 = _BN2_bn2s2["dy"]["addrs"][0]
BN2_UB_NZP_bn2s2 = _BN2_bn2s2["nzp"]["addrs"][0]
BN2_UB_SFMG_bn2s2 = _BN2_bn2s2["sfmg"]["addrs"][0]
BN2_UB_LSE_0_bn2s2, BN2_UB_LSE_1_bn2s2 = _BN2_bn2s2["lse"]["addrs"]
BN2_UB_CAST_bn2s2 = _BN2_bn2s2["cast"]["addrs"][0]


# ==================================================================
#  BN2S2 TND metadata 确定性：独立布局表与容量检查
# ==================================================================
# 仅该确定性入口使用；旧 BN2 / 独立 GS 的布局表及 mutex 不变。
# GM column post aliases only the y input allocation after V1/V2 finish.
# It never borrows a live C1/C2 result slot. D=192/Dv=128 uses the 128-wide
# vector arena, so size this table by v1w, not the query template width.
PLAN_POST_ROWS_bn2s2 = {64: 16, 128: 32, 192: 48, 256: 64}
for _width_bn2s2, _rows_bn2s2 in PLAN_POST_ROWS_bn2s2.items():
    assert _rows_bn2s2 * 128 * 4 <= 64 * _width_bn2s2 * 2

PLAN_PRE_ELEMENTS_bn2s2 = 32768

PLAN_ACC_ADDRS_bn2s2 = {}
PLAN_ACC_MUTEX_bn2s2 = {}
PLAN_DV0_ADDR_bn2s2 = {}
PLAN_DV1_ADDR_bn2s2 = {}
PLAN_DV0_MUTEX_bn2s2 = {}
PLAN_DV1_MUTEX_bn2s2 = {}
for _d_bn2s2, _v_bn2s2 in (
    (64, 64),
    (128, 128),
    (192, 128),
    (192, 192),
    (256, 256),
    (768, 768),
):
    _key_bn2s2 = _d_bn2s2 * 1024 + _v_bn2s2
    PLAN_ACC_ADDRS_bn2s2[_key_bn2s2] = list(L0C_ACC_ADDRS_bn2s2[_d_bn2s2])
    PLAN_ACC_MUTEX_bn2s2[_key_bn2s2] = list(L0C_ACC_MUTEX_bn2s2[_d_bn2s2])
    PLAN_DV0_ADDR_bn2s2[_key_bn2s2], PLAN_DV1_ADDR_bn2s2[_key_bn2s2] = (
        L0C_DV0_ADDR_bn2s2[_d_bn2s2],
        L0C_DV1_ADDR_bn2s2[_d_bn2s2],
    )
    PLAN_DV0_MUTEX_bn2s2[_key_bn2s2], PLAN_DV1_MUTEX_bn2s2[_key_bn2s2] = (
        MTX_L0C_DV0_bn2s2[_d_bn2s2],
        MTX_L0C_DV1_bn2s2[_d_bn2s2],
    )
PLAN_ACC_ADDRS_bn2s2[192 * 1024 + 192] = [0, 65536]
PLAN_ACC_MUTEX_bn2s2[192 * 1024 + 192] = [
    L0C_ACC_MUTEX_bn2s2[192][0],
    MTX_L0C_DK0_bn2s2[192],
]
PLAN_DV0_ADDR_bn2s2[192 * 1024 + 192], PLAN_DV1_ADDR_bn2s2[192 * 1024 + 192] = (
    131072,
    180224,
)
PLAN_DV0_ADDR_bn2s2[256 * 1024 + 256], PLAN_DV1_ADDR_bn2s2[256 * 1024 + 256] = (
    131072,
    196608,
)
PLAN_DV0_MUTEX_bn2s2[256 * 1024 + 256], PLAN_DV1_MUTEX_bn2s2[256 * 1024 + 256] = (
    MTX_L0C_DK0_bn2s2[256],
    MTX_L0C_DV0_bn2s2[256],
)


# D192 resident outputs occupy one compact NZ accumulator each, as in ASC.
PLAN_ACC_WIDTH_bn2s2 = {key: 128 for key in PLAN_ACC_ADDRS_bn2s2}
PLAN_ACC_WIDTH_bn2s2[192 * 1024 + 128] = 192
PLAN_ACC_WIDTH_bn2s2[768 * 1024 + 768] = 512
PLAN_ACC_ADDRS_bn2s2[768 * 1024 + 768] = [0]
PLAN_ACC_MUTEX_bn2s2[768 * 1024 + 768] = [L0C_ACC_MUTEX_bn2s2[768][0]]
PLAN_DK0_ADDR_bn2s2 = dict(L0C_DK0_ADDR_bn2s2)
PLAN_DK0_ADDR_bn2s2[192] = 98304
PLAN_DK0_WIDTH_bn2s2 = dict(D_CHUNK_MAXW_bn2s2)
PLAN_DK0_WIDTH_bn2s2[192] = 192
PLAN_DV0_WIDTH_bn2s2 = {
    key: D_CHUNK_MAXW_bn2s2[key // 1024] for key in PLAN_DV0_ADDR_bn2s2
}
PLAN_DV0_WIDTH_bn2s2[192 * 1024 + 128] = 128
PLAN_DV0_WIDTH_bn2s2[192 * 1024 + 192] = 192
PLAN_DV0_ADDR_bn2s2[192 * 1024 + 128] = 196608


def _check_plan_l0c_layouts_bn2s2():
    """Import-time capacity/overlap audit of the groups each static key uses."""
    for d, v in ((64, 64), (128, 128), (192, 128), (192, 192), (256, 256), (768, 768)):
        key = d * 1024 + v
        allocations = [
            (a, 128 * PLAN_ACC_WIDTH_bn2s2[key] * 4, i)
            for a, i in zip(PLAN_ACC_ADDRS_bn2s2[key], PLAN_ACC_MUTEX_bn2s2[key])
        ]
        if d <= 128 or (d == 192 and v <= 128):
            allocations.append(
                (
                    PLAN_DK0_ADDR_bn2s2[d],
                    128 * PLAN_DK0_WIDTH_bn2s2[d] * 4,
                    MTX_L0C_DK0_bn2s2[d],
                )
            )
        if d <= 256:
            size = 128 * PLAN_DV0_WIDTH_bn2s2[key] * 4
            allocations.append(
                (PLAN_DV0_ADDR_bn2s2[key], size, PLAN_DV0_MUTEX_bn2s2[key])
            )
            if d == 256:
                allocations.append(
                    (PLAN_DV1_ADDR_bn2s2[key], size, PLAN_DV1_MUTEX_bn2s2[key])
                )
        other_ids = set(range(_CV_RESERVED_bn2s2))
        for space in (_L1_bn2s2[d], _L0AB_bn2s2[d]):
            for name, record in space.items():
                if name != "_arenas":
                    other_ids.update(record["mutex"])
        end = 0
        used_ids = set()
        for address, size, mutex in sorted(allocations):
            if address < end or address + size > 256 * 1024:
                raise ValueError(
                    f"Plan L0C overlap/overflow for {d}/{v}: {allocations}"
                )
            if mutex in used_ids or mutex in other_ids or not 0 <= mutex < 32:
                raise ValueError(f"Plan L0C mutex collision for {d}/{v}: {mutex}")
            end = address + size
            used_ids.add(mutex)


# D64 metadata resident outputs need at most 128 x 64 fp32 each.
# Keep L0A/B DB and all arithmetic/commit protocols; spend the freed 64 KiB
# on a third common accumulator, matching the inspected ASC D64 layout.
PLAN_ACC_ADDRS_bn2s2[64 * 1024 + 64] = [0, 65536, 131072]
PLAN_ACC_MUTEX_bn2s2[64 * 1024 + 64] = list(L0C_ACC_MUTEX_bn2s2[64]) + _CUBE_POOL_bn2s2[
    64
].take("plan_acc_third", 1)
PLAN_DK0_ADDR_bn2s2[64] = 196608
PLAN_DK0_WIDTH_bn2s2[64] = 64
PLAN_DV0_ADDR_bn2s2[64 * 1024 + 64] = 229376
PLAN_DV0_WIDTH_bn2s2[64 * 1024 + 64] = 64

_check_plan_l0c_layouts_bn2s2()


# ==================================================================
#  自查
# ==================================================================
def describe_layout_bn2s2(tier=HEAD_DIM_ALIGN_bn2s2):
    """打印一档的完整片上内存图：地址 + 大小 + 持有的 buffer-id。"""
    arenas = []
    for group in (_L1_bn2s2, _L0AB_bn2s2, _L0C_bn2s2, _UB_bn2s2):
        arenas.extend(group[tier]["_arenas"])
    return "\n\n".join(a.describe() for a in arenas)


def audit_mutex_bn2s2(tier=HEAD_DIM_ALIGN_bn2s2):
    """打印一档的 buffer-id 占用：每个号归谁、每侧还剩几个。"""
    return _CUBE_POOL_bn2s2[tier].report() + "\n\n" + _VEC_POOL_bn2s2[tier].report()


# Metadata-backed PreSFMG: separate zero arena, two Y/dY/output slots.
# Keep the entry tiling scratch at [0, 0x1000) untouched. The usable UB bound
# is 253952 bytes, including the head reserve. PR15483's larger 128/192 rows
# cannot be copied verbatim under this bound.
PLAN_SFMG_ROWS_bn2s2 = {128: 208, 192: 136, 256: 104, 768: 104}
PLAN_SFMG_WIDTH_bn2s2 = {128: 128, 192: 192, 256: 256, 768: 256}
# The 192 VF reads 128 B beyond its last named row before masking. Include a
# physical padding row in the tile itself, so auto-mutex sees the whole read.
PLAN_SFMG_PHYSICAL_ROWS_bn2s2 = {
    w: r + int(w == 192) for w, r in PLAN_SFMG_ROWS_bn2s2.items()
}
PLAN_ZERO_ELEMENTS_bn2s2 = {0: PLAN_PRE_ELEMENTS_bn2s2, 1: 8192}
PLAN_SFMG_Y_ADDRS_bn2s2 = {}
PLAN_SFMG_DY_ADDRS_bn2s2 = {}
PLAN_SFMG_OUT_ADDRS_bn2s2 = {}
PLAN_SFMG_INPUT_BYTES_bn2s2 = {}
PLAN_SFMG_OUTPUT_BYTES_bn2s2 = {}
PLAN_SFMG_END_bn2s2 = {}
for _w_bn2s2, _r_bn2s2 in PLAN_SFMG_ROWS_bn2s2.items():
    _slot_bn2s2 = (
        (
            PLAN_SFMG_PHYSICAL_ROWS_bn2s2[_w_bn2s2]
            * PLAN_SFMG_WIDTH_bn2s2[_w_bn2s2]
            * 2
            + 511
        )
        // 512
        * 512
    )
    _out_bn2s2 = (_r_bn2s2 * 4 + 511) // 512 * 512
    _base_bn2s2 = UB_HEAD_RESERVE_bn2s2 + 8192 * 4
    PLAN_SFMG_Y_ADDRS_bn2s2[_w_bn2s2] = [_base_bn2s2, _base_bn2s2 + _slot_bn2s2]
    PLAN_SFMG_DY_ADDRS_bn2s2[_w_bn2s2] = [
        _base_bn2s2 + 2 * _slot_bn2s2,
        _base_bn2s2 + 3 * _slot_bn2s2,
    ]
    PLAN_SFMG_OUT_ADDRS_bn2s2[_w_bn2s2] = [
        _base_bn2s2 + 4 * _slot_bn2s2,
        _base_bn2s2 + 4 * _slot_bn2s2 + _out_bn2s2,
    ]
    PLAN_SFMG_INPUT_BYTES_bn2s2[_w_bn2s2] = _slot_bn2s2
    PLAN_SFMG_OUTPUT_BYTES_bn2s2[_w_bn2s2] = _out_bn2s2
    PLAN_SFMG_END_bn2s2[_w_bn2s2] = _base_bn2s2 + 4 * _slot_bn2s2 + 2 * _out_bn2s2
# PR15483's six distinct pre slots reuse main-phase IDs; the full C/V barrier
# separates these phases. Zero has its own disjoint physical range and ID.
PLAN_SFMG_Y_MUTEX_bn2s2 = [MTX_UB_Y_bn2s2[0], MTX_UB_DY_bn2s2[0]]
PLAN_SFMG_DY_MUTEX_bn2s2 = [MTX_UB_NZP_bn2s2[0], MTX_UB_CAST_bn2s2[0]]
PLAN_SFMG_OUT_MUTEX_bn2s2 = [MTX_UB_SFMG_bn2s2[0], MTX_UB_LSE_bn2s2[0]]

# Key combines existing static fields; it does not add runtime key dimensions.
PLAN_MAIN_Y_bn2s2 = {}
PLAN_MAIN_DY_bn2s2 = {}
PLAN_MAIN_NZP_bn2s2 = {}
PLAN_MAIN_SFMG_bn2s2 = {}
PLAN_MAIN_LSE0_bn2s2 = {}
PLAN_MAIN_LSE1_bn2s2 = {}
PLAN_MAIN_CAST_bn2s2 = {}
PLAN_MAIN_END_bn2s2 = {}
PLAN_MAIN_ALLOCATIONS_bn2s2 = {}
for _d_bn2s2, _dv_bn2s2 in (
    (64, 64),
    (128, 128),
    (192, 128),
    (192, 192),
    (256, 256),
    (768, 768),
):
    _v_bn2s2 = min(_dv_bn2s2, 256)
    for _pre_bn2s2 in (0, 1):
        for _mask_bn2s2 in (0, 3, 4):
            _key_bn2s2 = (
                _pre_bn2s2 * 16777216
                + _mask_bn2s2 * 1048576
                + _d_bn2s2 * 1024
                + _dv_bn2s2
            )
            _layout_bn2s2 = _UB_bn2s2[_v_bn2s2]
            if _pre_bn2s2 == 0:
                _y_bn2s2, _dy_bn2s2 = UB_Y_A_bn2s2[_v_bn2s2], UB_DY_A_bn2s2[_v_bn2s2]
                _nzp_bn2s2, _sfmg_bn2s2 = (
                    UB_NZP_A_bn2s2[_v_bn2s2],
                    UB_SFMG_A_bn2s2[_v_bn2s2],
                )
                _lse0_bn2s2, _lse1_bn2s2 = (
                    UB_LSE0_A_bn2s2[_v_bn2s2],
                    UB_LSE1_A_bn2s2[_v_bn2s2],
                )
                _cast_bn2s2 = UB_CAST_A_bn2s2[_v_bn2s2]
                _end_bn2s2 = _cast_bn2s2 + 16384
                _allocations_bn2s2 = []
            else:
                _pos_bn2s2 = UB_HEAD_RESERVE_bn2s2 + 0x20000
                _column_bytes_bn2s2 = (
                    PLAN_POST_ROWS_bn2s2[_v_bn2s2] * 128 * 4 if _d_bn2s2 > 128 else 0
                )
                _y_bytes_bn2s2 = max(_column_bytes_bn2s2, 8192 if _mask_bn2s2 else 0)
                _dy_bytes_bn2s2 = 8192 if _mask_bn2s2 else 0
                _y_bn2s2 = _pos_bn2s2
                _pos_bn2s2 += _y_bytes_bn2s2
                _dy_bn2s2 = _pos_bn2s2
                _pos_bn2s2 += _dy_bytes_bn2s2
                _nzp_bn2s2 = _pos_bn2s2
                _pos_bn2s2 += 65 * 128 * 2
                _pos_bn2s2 = (_pos_bn2s2 + 511) // 512 * 512
                _sfmg_bn2s2 = _pos_bn2s2
                _pos_bn2s2 += 512
                _lse0_bn2s2 = _pos_bn2s2
                _pos_bn2s2 += 512
                _lse1_bn2s2 = _pos_bn2s2
                _pos_bn2s2 += 512
                _cast_bn2s2 = _pos_bn2s2
                _pos_bn2s2 += 16384
                _end_bn2s2 = _pos_bn2s2
                _allocations_bn2s2 = [
                    ("mask_column", _y_bn2s2, _y_bytes_bn2s2, MTX_UB_Y_bn2s2[0]),
                    ("mask_pre", _dy_bn2s2, _dy_bytes_bn2s2, MTX_UB_DY_bn2s2[0]),
                    ("nzp", _nzp_bn2s2, 65 * 128 * 2, MTX_UB_NZP_bn2s2[0]),
                    ("sfmg", _sfmg_bn2s2, 512, MTX_UB_SFMG_bn2s2[0]),
                    ("lse0", _lse0_bn2s2, 512, MTX_UB_LSE_bn2s2[0]),
                    ("lse1", _lse1_bn2s2, 512, MTX_UB_LSE_bn2s2[1]),
                    ("cast", _cast_bn2s2, 16384, MTX_UB_CAST_bn2s2[0]),
                ]
            PLAN_MAIN_Y_bn2s2[_key_bn2s2], PLAN_MAIN_DY_bn2s2[_key_bn2s2] = (
                _y_bn2s2,
                _dy_bn2s2,
            )
            PLAN_MAIN_NZP_bn2s2[_key_bn2s2], PLAN_MAIN_SFMG_bn2s2[_key_bn2s2] = (
                _nzp_bn2s2,
                _sfmg_bn2s2,
            )
            PLAN_MAIN_LSE0_bn2s2[_key_bn2s2], PLAN_MAIN_LSE1_bn2s2[_key_bn2s2] = (
                _lse0_bn2s2,
                _lse1_bn2s2,
            )
            PLAN_MAIN_CAST_bn2s2[_key_bn2s2], PLAN_MAIN_END_bn2s2[_key_bn2s2] = (
                _cast_bn2s2,
                _end_bn2s2,
            )
            PLAN_MAIN_ALLOCATIONS_bn2s2[_key_bn2s2] = _allocations_bn2s2


# D64 metadata main loop: separate P's Vec producer from dS's MTE3 reader.
# Pre/post use the same physical UB only across the existing sync_all barriers.
# Preserve the software-usable 248 KiB bound and the entry head reservation.
PLAN_P_NZP_bn2s2 = {}
PLAN_P_NZP_MUTEX_bn2s2 = {
    v: _VEC_POOL_bn2s2[v].take("plan_p_nzp", 1) for v in (64, 128, 192, 256)
}
for _d_bn2s2, _dv_bn2s2 in (
    (64, 64),
    (128, 128),
    (192, 128),
    (192, 192),
    (256, 256),
    (768, 768),
):
    for _pre_bn2s2 in (0, 1):
        for _mask_bn2s2 in (0, 3, 4):
            _key_bn2s2 = (
                _pre_bn2s2 * 16777216
                + _mask_bn2s2 * 1048576
                + _d_bn2s2 * 1024
                + _dv_bn2s2
            )
            _p_start_bn2s2 = (
                (
                    max(
                        PLAN_MAIN_END_bn2s2[_key_bn2s2],
                        UB_SINK_PART_A_bn2s2[min(_dv_bn2s2, 256)] + 512,
                    )
                    + 511
                )
                // 512
                * 512
            )
            # A shared dS/P tile with one mutex remains valid when the
            # phase's live allocations leave no complete second 65-row NZ tile.
            if _p_start_bn2s2 + 65 * 128 * 2 <= 253952:
                PLAN_P_NZP_bn2s2[_key_bn2s2] = _p_start_bn2s2
PLAN_HAS_P_NZP_bn2s2 = {key: key in PLAN_P_NZP_bn2s2 for key in PLAN_MAIN_END_bn2s2}


# Full-D column conversion reuses the existing D192 24KiB source and 16KiB cast
# allocations and their mutexes. No extra UB or cross-phase alias is introduced.
PLAN_COLUMN_ROWS_bn2s2 = {
    d * 1024 + dv: PLAN_POST_ROWS_bn2s2[min(dv, 256)]
    for d, dv in ((64, 64), (128, 128), (192, 128), (192, 192), (256, 256), (768, 768))
}
PLAN_COLUMN_WIDTH_bn2s2 = {key: 128 for key in PLAN_COLUMN_ROWS_bn2s2}
PLAN_COLUMN_ROWS_bn2s2[192 * 1024 + 192] = 32
PLAN_COLUMN_WIDTH_bn2s2[192 * 1024 + 192] = 192
PLAN_COLUMN_CAST_ROWS_bn2s2 = {key: 64 for key in PLAN_COLUMN_ROWS_bn2s2}
PLAN_COLUMN_CAST_ROWS_bn2s2[192 * 1024 + 192] = 32
PLAN_COLUMN_POST_ROWS_bn2s2 = dict(PLAN_COLUMN_CAST_ROWS_bn2s2)
assert 32 * 192 * 4 == PLAN_POST_ROWS_bn2s2[192] * 128 * 4
assert 32 * 192 * 2 <= 64 * 128 * 2


# Stack2 D192/Dv128: ASC-sized full-row conversion arena, after independent P.
# Main, pre and post remain separated by existing sync_all phase boundaries.
PLAN_COLUMN_MAIN_ADDR_bn2s2 = dict(PLAN_MAIN_Y_bn2s2)
PLAN_COLUMN_FULL_MUTEX_bn2s2 = _VEC_POOL_bn2s2[128].take("plan_column_full_d", 1)
PLAN_COLUMN_FULL_END_bn2s2 = {}
for _pre_bn2s2 in (0, 1):
    for _mask_bn2s2 in (0, 3, 4):
        _key_bn2s2 = _pre_bn2s2 * 16777216 + _mask_bn2s2 * 1048576 + 192 * 1024 + 128
        _begin_bn2s2 = (PLAN_P_NZP_bn2s2[_key_bn2s2] + 65 * 128 * 2 + 511) // 512 * 512
        assert _begin_bn2s2 + 32768 <= 253952
        PLAN_COLUMN_MAIN_ADDR_bn2s2[_key_bn2s2] = _begin_bn2s2
        PLAN_COLUMN_FULL_END_bn2s2[_key_bn2s2] = _begin_bn2s2 + 32768
PLAN_COLUMN_ROWS_bn2s2[192 * 1024 + 128] = 1
PLAN_COLUMN_WIDTH_bn2s2[192 * 1024 + 128] = 8192
PLAN_COLUMN_CAST_ROWS_bn2s2[192 * 1024 + 128] = 1
PLAN_COLUMN_POST_ROWS_bn2s2[192 * 1024 + 128] = 1
assert 42 * 192 * 4 <= 32768 and 42 * 192 * 2 <= 16384


# Ordinary BN2S2 independent P occupies genuine main-loop slack. It does not
# alias C1/C2, live cast/sink, mask/Y/dY, or the 0x1000 entry scratch reserve.
NORMAL_P_NZP_ADDR_bn2s2 = {}
for _v_bn2s2 in (64, 128, 192, 256):
    _start_bn2s2 = (
        (
            max(UB_CAST_A_bn2s2[_v_bn2s2] + 16384, UB_SINK_PART_A_bn2s2[_v_bn2s2] + 512)
            + 511
        )
        // 512
    ) * 512
    assert _start_bn2s2 + 65 * 128 * 2 <= 253952
    NORMAL_P_NZP_ADDR_bn2s2[_v_bn2s2] = _start_bn2s2
