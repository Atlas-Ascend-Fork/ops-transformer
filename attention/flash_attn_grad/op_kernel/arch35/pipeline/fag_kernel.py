# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""两个分核模板的入口与编译期分发。

本文件只负责**编排**：建 GM/L1 视图和 ``pl.make_tuple`` 袋子、跑 pre/post、
按 ``deter`` / ``template`` 分发到主循环。真正的 while 在
:mod:`fag_kernel_norm` 与 :mod:`fag_kernel_deter`；计算在
:mod:`fag_block_cube` 与 :mod:`fag_block_vec`，分核映射在 :mod:`fag_schedule`，
片上地址在 :mod:`fag_mem`。

两个模板：

* :func:`flash_attn_grad_preload3` —— ``Tmpl.BN2GS1S2`` 且 D≤192：
  cube 领先 vector 两个 task（``PRELOAD_TIMES=3``），自带 pre/post。
  192 档 dK 走 fp32 workspace（与 256 档同口径）、dV 仍常驻 L0C。
* :func:`flash_attn_grad_preload2` —— 1-ahead。``Tmpl.BN2`` 走
  :mod:`fag_kernel_norm` 的 BN2 循环；GS1S2 再按 ``deter`` 分到
  normal / deterministic。

CV 通信全部使用手动 set/wait cross-core，且正反向成对：

  正向 C->V : SYNC_C2_TO_V2_FLAG[2] / SYNC_C1_TO_V2_FLAG[2]  (mm2/mm1 结果就绪)
  正向 V->C : SYNC_V3_TO_C3_FLAG / SYNC_V4_TO_C5_FLAG        (dS/P 已落 L1)
  反向 C->V : SYNC_C4_TO_V3_FLAG / SYNC_C5_TO_V4_FLAG        (L1 已被 cube 读完)
  反向 V->C : SYNC_V2_TO_C1_FLAG[2] / SYNC_V2_TO_C2_FLAG[2]  (mm1/mm2 UB 已读完)

AIC 侧对每个 V->C flag 用 ``sync_mode=INTRA_BLOCK`` 等一次，覆盖两个
vector 子核各自发来的信号。

当前裁剪范围：BSND/BNSD/TND、FP16/BF16。mask_mode 0=dense，3=causal，
4=band。tilingkey ``swizzle`` 区分线性 skip 无效块与 packed 有效块分核。
``deter`` 只做编译期分发。非 swizzle 核上闭式分核，与现网同套跨轮流水和
L0C 累加，不每轮 ``sync_all``。BN2 支持 mask，禁止 BN2+swizzle，且与
GS1S2 deterministic 完全隔离。
"""

from pipeline.fag_bindings import bind_codegen_helpers as _bind_codegen_helpers
from pipeline.fag_kernel_deter import run_plan_cube_bn2s2, run_plan_vector_bn2s2
from pipeline.fag_kernel_norm import (
    normal_score_pair_bn2s2,
    normal_vector_retire_bn2s2,
    _run_preload2_cube_bn2s2,
    _run_preload2_vector_bn2s2,
    _run_preload2_cube_gs_normal_bn2s2,
    _run_preload2_vector_gs_normal_bn2s2,
)

import pypto_pro.language as pl

from compute import fag_block_cube
from compute import fag_block_vec
from memory import fag_buffers
from memory.fag_mem import PLAN_HAS_P_NZP_bn2s2
import fag_common
import fag_nzout
from schedule.deterministic import fag_deter_formulas, fag_deter_legacy, fag_deter_state
from schedule.deterministic import fag_deter_fixed
from schedule.deterministic import fag_deter_baseline
from schedule.deterministic import fag_deter_band_owner
from schedule.deterministic import fag_deter_band_plan
from schedule.deterministic.tnd import fag_deter_tnd_dense
from schedule.deterministic.tnd import fag_deter_tnd_sparse
from schedule.deterministic.tnd import fag_deter_tnd_plan
from schedule.deterministic.tnd import fag_deter_tnd_normalized
from schedule import fag_schedule
from fag_tiling import FlashAttnGradTilingData, Tmpl
from pipeline.fag_kernel_deter import (
    _run_preload2_cube_gs_deter,
    _run_preload2_vector_gs_deter,
    _run_preload3_cube_deter,
    _run_preload3_vector_deter,
)
from pipeline.fag_kernel_norm import (
    _run_preload2_cube_bn2,
    _run_preload2_cube_gs_normal,
    _run_preload2_vector_bn2,
    _run_preload2_vector_gs_normal,
    _run_preload3_cube_normal,
    _run_preload3_vector_normal,
)

# codegen 内联时读的是本模块 globals() 里的裸名，不是 from-import 清单。
# module.func() 会被当成 IR op，所以这里把 helper / 常量绑成裸名。
# 其它 list 不灌进来，避免和必须具名的 ping-pong flag 抢提升。
_PINGPONG_LISTS = frozenset(
    {
        "SYNC_C1_TO_V2_FLAG",
        "SYNC_C2_TO_V2_FLAG",
        "SYNC_V2_TO_C1_FLAG",
        "SYNC_V2_TO_C2_FLAG",
        "SYNC_C1_TO_V2_FLAG_bn2s2",
        "SYNC_C2_TO_V2_FLAG_bn2s2",
        "SYNC_V2_TO_C1_FLAG_bn2s2",
        "SYNC_V2_TO_C2_FLAG_bn2s2",
        "SYNC_C4_TO_V6_PP_FLAG_bn2s2",
        "SYNC_DKV_RELEASE_PP_FLAG_bn2s2",
        "WIDE_DK_READY_bn2s2",
        "WIDE_DV_READY_bn2s2",
        "WIDE_KV_READ_DONE_bn2s2",
    }
)


def _bind_helpers():
    _bind_codegen_helpers(
        globals(),
        (
            fag_block_cube,
            fag_block_vec,
            fag_buffers,
            fag_common,
            fag_deter_formulas,
            fag_deter_legacy,
            fag_deter_state,
            fag_deter_fixed,
            fag_deter_baseline,
            fag_deter_band_owner,
            fag_deter_band_plan,
            fag_deter_tnd_dense,
            fag_deter_tnd_sparse,
            fag_deter_tnd_plan,
            fag_deter_tnd_normalized,
            fag_schedule,
            fag_nzout,
        ),
        _PINGPONG_LISTS,
    )


_bind_helpers()


def flash_attn_grad_preload3(
    # 13 个输入，顺序严格对齐 op_host/flash_attn_grad_def.cpp 的 Input() 声明
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    # 以下为可选输入。TND 的 cu_seqlens 给物理偏移，seqused 给有效长度。
    # sinks：tilingkey sink=1 时按 head 读标量，算 dsink。
    # attn_mask：mask_mode 3/4 时按 [2048,2048] uint8 压缩矩阵搬 UB。
    # metadata：FAG 区含分核 [start,end) 与 pre/post 行区间（412 int32）。
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    # 4 个输出。dsink 是 [N1] fp32，仅在传入 sinks 时写出有效结果。
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    dsink: pl.Ptr[pl.DT_UINT8],
    # 单块 workspace，内部再切 dq/dk/dv 累加区和 dsink 槽位
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
    d_align,
    dv_align,
    swizzle,
    deter,
    sink,
):
    # 入口只建视图 / buffer / pre-post，cube/vector 主循环按 deter 编译期分发。
    # coreNum 从内建 API 取而非 tiling：TilingData 无此字段。
    # get_block_num() 是 host SetBlockDim 启动的 cube 数（dense 下等于
    # metadata blockOuter），cube/vector 两侧同一口径，set/wait 才能配对。
    coreNum = pl.get_block_num()
    # swizzle 来自 tilingkey，是编译期常量 —— 每种分核各生成一份二进制，
    # `if swizzle` 的两路在编译期折叠，运行期没有判断开销。
    cu_q_tensor = pl.make_tensor(cu_seqlens_q, [tiling.b + 1], [1], dtype=pl.DT_INT32)
    cu_kv_tensor = pl.make_tensor(cu_seqlens_kv, [tiling.b + 1], [1], dtype=pl.DT_INT32)
    used_q_tensor = pl.make_tensor(seqused_q, [tiling.b], [1], dtype=pl.DT_INT32)
    used_kv_tensor = pl.make_tensor(seqused_kv, [tiling.b], [1], dtype=pl.DT_INT32)
    # constInfo 内统一计算实际 token 数，区分逻辑序列长度与 GM 视图长度。
    constInfo = set_const_info(tiling, coreNum, cu_q_tensor, cu_kv_tensor)

    # ---- workspace 分区 ----
    # host(GetWorkspaceSizes) 给的是单块连续内存，这里按 fp32 元素切成
    # dq/dk/dv 三段累加区。ND 三段大小与 host 的 CalcWorkSpace 逐一对齐；
    # NZ 内核按运行期 ceil16(D) 紧凑分段，host 按 128 列档预留容量上界。
    # dq 段按 n1(=n2*G) 个 head，dk/dv 段按 n2 —— 与 host 的 dqElems/dkvElems
    # 一致。GQA 下若这里误用 n2，dk/dv 的起始偏移就会算错。
    # TND 按物理 packed token 分段，非 TND 的 T 仍为 B*S；seqused 不改变跨度。
    # host 只在三段之后多留 128 行保护垫，不改这里的基地址。
    wsRowsQ = constInfo.t1Size * constInfo.n1Size
    wsRowsKV = constInfo.t2Size * constInfo.n2Size
    dqWsElems = wsRowsQ * tiling.d
    dkWsElems = wsRowsKV * tiling.d
    dvWsElems = wsRowsKV * tiling.dv
    wsPadCols = pl.max(tiling.d, tiling.dv)
    if nz_out == 1:
        qTiles = (
            nz_workspace_tiles(constInfo.t1Size, tiling.b, tiling.s1) * constInfo.n1Size
        )
        kvTiles = (
            nz_workspace_tiles(constInfo.t2Size, tiling.b, tiling.s2) * constInfo.n2Size
        )
        dqWsElems = qTiles * 128 * constInfo.dAlign
        dkWsElems = kvTiles * 128 * constInfo.dAlign
        dvWsElems = dkWsElems
        wsPadCols = constInfo.dAlign
    ws_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    dq_workspace = ws_ptr
    # 段宽按各自的 head dim：dq/dk 是 D，dv 是 Dv(可以短于 D)。
    # ND 与 host 的 dqElems/dkElems/dvElems 对齐；NZ 的所有内核消费者
    # 统一使用上述紧凑分段，host 的容量上界不作为内核分段偏移。
    dk_workspace = pl.addptr(dq_workspace, dqWsElems)
    dv_workspace = pl.addptr(dk_workspace, dkWsElems)
    # 128 行保护垫之后是 dsink 槽位 [aivCount x N1]；NZ 也按紧凑段尾计算。
    dsink_workspace = pl.addptr(dv_workspace, dvWsElems + 128 * wsPadCols)
    # dsink 槽位之后是 sfmg 前移缓存 [N1*T1] fp32；host 为全段预留足够容量。
    # dv_align>=128 时 pre 阶段全量写入、主循环 V1 只读现成值。
    sfmg_workspace = pl.addptr(dsink_workspace, constInfo.aivCount * constInfo.n1Size)

    # ---- 统一四维视图 [viewD0, S, viewD2, D] ----
    # BSND 与 BNSD 共用同一套 make_tensor / pl.load(order=...) 代码，
    # 布局差异只体现在 viewD0/viewD2 与索引系数上(见 set_run_info 的 idx0/idx2)：
    #   BSND: [B,   S, N, D]，idx = [b,     s, n, 0]
    #   BNSD: [B*N, S, 1, D]，idx = [b*N+n, s, 0, 0]
    # 两者都是紧凑排布 —— 这一点是必须的：make_tensor 对非紧凑 stride 会静默
    # 忽略并按紧凑跨距去读(不报错)，所以不能用「倒置 stride」表达 BNSD。
    # Q 侧(n1 个 head)与 KV 侧(n2 个 head)各一套视图参数
    # 最内轴宽度取各张量**真实的** head dim，不能用分配宽度 d_align：
    # 这是 GM 上的实际排布，Q/K/dq/dk 侧是 D，V/dO/attn_out/dv 侧是 Dv。
    # (片上 tile 才按 d_align 分配、按真实宽度填。)
    dQ = constInfo.dSize
    dV = constInfo.dvSize
    viewD2Q = constInfo.viewD2Q
    d2dQ = viewD2Q * dQ
    s1ndQ = constInfo.s1ViewSize * d2dQ
    viewD2KV = constInfo.viewD2KV
    d2dKV = viewD2KV * dQ
    s2ndKV = constInfo.s2ViewSize * d2dKV
    # Dv 宽的那几个张量另有一套行距
    d2dQv = viewD2Q * dV
    s1ndQv = constInfo.s1ViewSize * d2dQv
    d2dKVv = viewD2KV * dV
    s2ndKVv = constInfo.s2ViewSize * d2dKVv

    tensor_q = pl.make_tensor(
        q,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
        [s1ndQ, d2dQ, dQ, 1],
        dtype=input_dtype,
    )
    tensor_k = pl.make_tensor(
        k,
        [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=input_dtype,
    )
    # MM3(dQ = dS@K) 需要「未转置」的 K，而 MM2 需要 K^T。
    # 每个 GM tensor 只会推导出一种 Layout：tensor_k 因 MM2 的转置读被定成
    # Layout::DN，MM3 若共用它，plain 读也会被按 DN 解释，实测等价于拿到 K^T
    # (partial 与 K[s2o]^T 吻合到 1e-4)。故为 MM3 单独建一份同 buffer 的
    # tensor 视图，让它独立推导 layout。
    tensor_k_nt = pl.make_tensor(
        k,
        [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=input_dtype,
    )
    # V/dO/attn_out 是 Dv 宽
    tensor_v = pl.make_tensor(
        v,
        [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
        [s2ndKVv, d2dKVv, dV, 1],
        dtype=input_dtype,
    )
    tensor_dy = pl.make_tensor(
        dout,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=input_dtype,
    )
    tensor_y = pl.make_tensor(
        attn_out,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=input_dtype,
    )
    # softmax_lse : [B,N1,S1] —— 按 Q 的 head 数(n1 = n2*G)存，因为每个 Q head
    # 的 softmax 归一化因子不同，G 个 Q head 不能共享同一份 lse。
    # lse offset = ((b*n2 + n2o)*G + g)*S1。
    n1Size = constInfo.n2Size * constInfo.gSize
    if layout == 1:
        # TND LSE 为 [N1,T1]，loadBase 已是全局 token 行。
        # head 间距必须取实际 T1，不能用 B*maxSeqlenQ。
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [n1Size, constInfo.t1Size],
            [constInfo.t1Size, 1],
            dtype=pl.DT_FP32,
        )
    else:
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [constInfo.bSize, n1Size, constInfo.s1Size],
            [n1Size * constInfo.s1Size, constInfo.s1Size, 1],
            dtype=pl.DT_FP32,
        )
    if mask_mode != 0:
        tensor_mask = pl.make_tensor(
            attn_mask,
            [ATTEN_MASK_COMPRESS, ATTEN_MASK_COMPRESS],
            [ATTEN_MASK_COMPRESS, 1],
            dtype=pl.DT_UINT8,
        )
    else:
        tensor_mask = tensor_lse
    # sfmg 前移缓存：[1, N1*T1] fp32 扁平，行号 = head*T1 + 全局 token 行
    # （与 TND lse 的 [N1,T1] token 主序同构，读侧统一用 loadBaseTnd 口径）。
    # dv_align=64 不前移：占位必须是 rank-2 扁平视图（与非 TND 的
    # tensor_lse [B,N1,S1] 三维不同 rank，pypto 严格校验 load 的 offsets
    # 数与源 rank 一致，直接占位 lse 会在 trace 期报 INVALID_SHAPE）。
    # process_vec1 走内联分支时不解引用本占位。
    if dv_align >= 128:
        sfmgElems = n1Size * constInfo.t1Size
        tensor_sfmg_ws = pl.make_tensor(
            sfmg_workspace, [1, sfmgElems], [sfmgElems, 1], dtype=pl.DT_FP32
        )
    else:
        lseElems = n1Size * constInfo.t1Size
        tensor_sfmg_ws = pl.make_tensor(
            softmax_lse, [1, lseElems], [lseElems, 1], dtype=pl.DT_FP32
        )
    # dq/dk/dv workspace 为 fp32，供 atomicAdd 累加
    # dq/dk 是 D 宽，dv 是 Dv 宽 —— 与 workspace 的分段宽度一致
    if nz_out == 1:
        tensor_dq_ws = pl.make_tensor(
            dq_workspace, [1, dqWsElems], [dqWsElems, 1], dtype=pl.DT_FP32
        )
        tensor_dk_ws = pl.make_tensor(
            dk_workspace, [1, dkWsElems], [dkWsElems, 1], dtype=pl.DT_FP32
        )
        tensor_dv_ws = pl.make_tensor(
            dv_workspace, [1, dvWsElems], [dvWsElems, 1], dtype=pl.DT_FP32
        )
    else:
        tensor_dq_ws = pl.make_tensor(
            dq_workspace,
            [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
            [s1ndQ, d2dQ, dQ, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dk_ws = pl.make_tensor(
            dk_workspace,
            [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
            [s2ndKV, d2dKV, dQ, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dv_ws = pl.make_tensor(
            dv_workspace,
            [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
            [s2ndKVv, d2dKVv, dV, 1],
            dtype=pl.DT_FP32,
        )
    # ---- pre/post：一维视图 ----
    # InitOutput / Muls+Cast 按实际 T*N*D 个元素线性切（非 TND 的 T=B*S）。
    # dq 侧 head 数是 n1(=n2*G)，dk/dv 侧是 n2。
    # ws 与 out 扁平视图都按真实 d/dv（无 padding）。
    dqRows = constInfo.t1Size * n1Size
    dkvRows = constInfo.t2Size * constInfo.n2Size
    dqElems = dqRows * dQ
    dkElems = dkvRows * dQ
    dvElems = dkvRows * dV
    tensor_dq_ws_flat = pl.make_tensor(
        dq_workspace, [1, dqWsElems], [dqWsElems, 1], dtype=pl.DT_FP32
    )
    tensor_dk_ws_flat = pl.make_tensor(
        dk_workspace, [1, dkWsElems], [dkWsElems, 1], dtype=pl.DT_FP32
    )
    tensor_dv_ws_flat = pl.make_tensor(
        dv_workspace, [1, dvWsElems], [dvWsElems, 1], dtype=pl.DT_FP32
    )
    if nz_out == 1:
        tensor_dq_out = pl.make_tensor(
            dq,
            [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
            [s1ndQ, d2dQ, dQ, 1],
            dtype=input_dtype,
        )
        tensor_dk_out = pl.make_tensor(
            dk,
            [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
            [s2ndKV, d2dKV, dQ, 1],
            dtype=input_dtype,
        )
        tensor_dv_out = pl.make_tensor(
            dv,
            [constInfo.viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
            [s2ndKVv, d2dKVv, dV, 1],
            dtype=input_dtype,
        )
    else:
        tensor_dq_out = pl.make_tensor(
            dq, [1, dqElems], [dqElems, 1], dtype=input_dtype
        )
        tensor_dk_out = pl.make_tensor(
            dk, [1, dkElems], [dkElems, 1], dtype=input_dtype
        )
        tensor_dv_out = pl.make_tensor(
            dv, [1, dvElems], [dvElems, 1], dtype=input_dtype
        )
    tensor_sinks = pl.make_tensor(sinks, [n1Size], [1], dtype=pl.DT_FP32)
    dsinkElems = constInfo.aivCount * n1Size
    tensor_dsink_ws = pl.make_tensor(
        dsink_workspace, [1, dsinkElems], [dsinkElems, 1], dtype=pl.DT_FP32
    )
    # dsink : [N1] fp32，post 阶段由 AIV0 固定次序归约后全量写出。
    tensor_dsink_out = pl.make_tensor(dsink, [1, n1Size], [n1Size, 1], dtype=pl.DT_FP32)
    # V1 列宽就是 dv_align。D=256 不进本函数。
    v1w = dv_align

    # cBlockIdx 必须按物理核取：AIV 侧 get_block_idx() 按 vector 子核计数
    # (0..2*coreNum-1)，AIC 侧按物理核计数。两侧口径不一致会算出不同的
    # numBlocks 不同 -> 循环次数不同 -> set/wait 配不上 -> 死锁。
    # 核数直接用 tiling 传入，避免再依赖 get_block_num 的口径。
    cBlockIdx = pl.get_block_idx() // pl.get_subblock_num()
    coreNum = constInfo.coreNum

    # 分核区间取自 metadata。swizzle 支仍走公式，原因见 block_range_of。
    meta = pl.make_tensor(
        metadata, [pl.max(constInfo.metadataWordCount, 1)], [1], dtype=pl.DT_INT32
    )
    one64 = pl.const(1, pl.DT_INT64)
    fagOff = pl.getval(meta, META_HEAD_FAG_START_INDEX) * one64
    blockStart = (
        pl.getval(meta, fagOff + META_FAG_BLOCK_STARTS_OFFSET + cBlockIdx) * one64
    )
    blockEnd = pl.getval(meta, fagOff + META_FAG_BLOCK_ENDS_OFFSET + cBlockIdx) * one64
    numBlocks = block_range_of(constInfo, cBlockIdx, coreNum, blockStart, blockEnd)

    # ---- CV 共享区：mm1/mm2 结果 UB + dS/P 的 L1 及其 ZN 转置视图 ----
    # vector 侧写、cube 侧读，故必须声明在两个 section 之外。
    mm1_res = make_mm1_res()
    mm2_res = make_mm2_res()
    ds_l1 = make_ds_l1()
    p_l1 = make_p_l1()
    ds_t_l1 = make_ds_t_l1()
    p_t_l1 = make_p_t_l1()

    # ==== Pre 阶段：清零 dq/dk/dv workspace ====
    # 三个 workspace 都用 atomicAdd 累加，进主循环前必须为 0。
    # 清完用 sync_all 隔开：主循环的 atomicAdd 必须看到已清零的 GM。
    # dv_align>=128 时整段走"交错清零"路径；是否真正追加 softmaxGrad 前移由
    # 运行时 constInfo.enablePreSfmg 决定（host 按 fasg enablePreSfmg 判据，
    # 见 flash_attn_grad_tiling_plan.cpp 的 DecidePreSfmg）。前移时一次算完
    # 全部行的 rowsum(dO⊙O) 存 ws、主循环 V1 只读；否则主循环 V1 内联重算。
    # 交错 pre：zt 单槽 32KB（[0x1000,0x9000)，见 fag_mem.PRE_ELEMS），与
    # sp 窗口 [0x9000,0x40000) 完全避开（sp 借 mm1 次槽 + mm2 + 主图段，
    # mutex 复用主图 tile 的号，见 fag_mem 的 sfmg_pre 注释），pre 与主循环
    # 由 sync_all 分时。zero 与 sp
    # 不同 mutex，框架不插 wait——清零写(MTE3)与 presfmg 载入(MTE2)/VF
    # 两条流水并行，且分块交替 issue：三段清零按统一 chunk 切成任务流，穿插
    # 在 presfmg 的窗口循环里（每窗 kBatch 个，见 process_presfmg 的
    # _zero_step）——避免整段清零先灌满 MTE3 队列、拖延 sfmg 写的双缓冲
    # 周转（实测中小 S 用例清零占 pre 搬运 1/3~1/2）。
    if nz_out == 1:
        with pl.section_vector():
            zero_ub = make_pre_zero_buffer_f32()
            # NZ 布局按整段 tile 口径清零（覆盖 padding 与 TND gap）。sfmg
            # 前移的交错清零按 ND 行区间寻址，与 NZ tile 布局不兼容——这里
            # 清零段传 0，复用 process_presfmg 的纯计算形态（同 BN2 调用点：
            # zS 全 0 → zTotal=0 → 清零任务零循环），仅保留 y/dO 窗口计算与
            # sfmg 落盘；enablePreSfmg=0 时内部自动跳过计算。
            process_nz_pre(
                tensor_dq_ws_flat, dqWsElems, constInfo.aivCount, zero_ub.current()
            )
            process_nz_pre(
                tensor_dk_ws_flat, dkWsElems, constInfo.aivCount, zero_ub.current()
            )
            process_nz_pre(
                tensor_dv_ws_flat, dvWsElems, constInfo.aivCount, zero_ub.current()
            )
            if sink == 1:
                process_sink_zero(
                    tensor_dsink_ws, constInfo, zero_ub.current(), pl.get_block_idx()
                )
            sp_y_ub = make_sp_y_ub(v1w)
            sp_dy_ub = make_sp_dy_ub(v1w)
            sp_out_ub = make_sp_out_ub(v1w)
            sp_out_flat = make_sp_out_flat(v1w)
            process_presfmg(
                constInfo,
                tensor_y,
                tensor_dy,
                tensor_sfmg_ws,
                sp_y_ub,
                sp_dy_ub,
                sp_out_ub,
                sp_out_flat,
                fag_common.SP_WORK_W[v1w],
                zero_ub.current(),
                tensor_dq_ws_flat,
                tensor_dk_ws_flat,
                tensor_dv_ws_flat,
                0,
                0,
                0,
                0,
                0,
                0,
            )
    elif dv_align >= 128:
        with pl.section_vector():
            zero_ub = make_pre_zero_buffer_f32()
            zt = zero_ub.current()
            # 行区间与三段边界（原 process_pre_from_meta 的读法，见其文档）。
            vBlockIdx = pl.get_block_idx()
            one64 = pl.const(1, pl.DT_INT64)
            dqRowStart = (
                pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdx) * one64
            )
            dqRowEnd = (
                pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdx) * one64
            )
            dkvRowStart = (
                pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdx) * one64
            )
            dkvRowEnd = (
                pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdx) * one64
            )
            dqNElems = (dqRowEnd - dqRowStart) * dQ
            dkNElems = (dkvRowEnd - dkvRowStart) * dQ
            dvNElems = (dkvRowEnd - dkvRowStart) * dV
            preElems = pl.min(
                PRE_ELEMS[d_align], pl.max(dqNElems, pl.max(dkNElems, dvNElems))
            )
            fill_zero_vf(zt, (preElems + VF_LANES - 1) // VF_LANES)
            if sink == 1:
                process_sink_zero(tensor_dsink_ws, constInfo, zt, pl.get_block_idx())
            sp_y_ub = make_sp_y_ub(v1w)
            sp_dy_ub = make_sp_dy_ub(v1w)
            sp_out_ub = make_sp_out_ub(v1w)
            sp_out_flat = make_sp_out_flat(v1w)
            process_presfmg(
                constInfo,
                tensor_y,
                tensor_dy,
                tensor_sfmg_ws,
                sp_y_ub,
                sp_dy_ub,
                sp_out_ub,
                sp_out_flat,
                fag_common.SP_WORK_W[v1w],
                zt,
                tensor_dq_ws_flat,
                tensor_dk_ws_flat,
                tensor_dv_ws_flat,
                dqRowStart * dQ,
                dqNElems,
                dkvRowStart * dQ,
                dqNElems + dkNElems,
                dkvRowStart * dV,
                dqNElems + dkNElems + dvNElems,
            )
    else:
        with pl.section_vector():
            zero_ub = make_pre_zero_buffer_f32()
            process_pre_from_meta(
                meta,
                fagOff,
                tensor_dq_ws_flat,
                tensor_dk_ws_flat,
                tensor_dv_ws_flat,
                dQ,
                dV,
                zero_ub.current(),
            )
            if sink == 1:
                process_sink_zero(
                    tensor_dsink_ws, constInfo, zero_ub.current(), pl.get_block_idx()
                )
    pl.system.sync_all()

    with pl.section_cube():
        q_l1 = make_q_l1()
        k_l1 = make_k_l1()
        k_l1_n = make_k_nd_l1()
        v_l1 = make_v_l1()
        do_l1 = make_do_l1()
        left = make_l0a()
        right = make_l0b()
        # 192 档 L0C 的两套布局由工厂内部按 template/d_align 自己选（BN2 用 dK/dV
        # 常驻表，GS1S2 用 acc 双槽表），调用点不传标志位。
        acc = make_l0c_acc()
        q_l1_dk = make_q_lag_view()
        do_l1_dv = make_do_lag_view()
        dk_acc0 = make_dk_l0c_chunk0()
        dv_acc0 = make_dv_l0c_chunk0()
        dk_acc1 = make_dk_l0c_chunk1()
        dv_acc1 = make_dv_l0c_chunk1()
        runInfos = make_run_infos()
        sched = pl.make_tuple(
            meta=meta,
            fagOff=fagOff,
            cBlockIdx=cBlockIdx,
            coreNum=coreNum,
            cuQ=cu_q_tensor,
            cuKv=cu_kv_tensor,
            usedQ=used_q_tensor,
            usedKv=used_kv_tensor,
            blockStart=blockStart,
            blockEnd=blockEnd,
            numBlocks=numBlocks,
        )
        gm = pl.make_tuple(
            tensor_q=tensor_q,
            tensor_k=tensor_k,
            tensor_k_nt=tensor_k_nt,
            tensor_dy=tensor_dy,
            tensor_v=tensor_v,
            tensor_dq_ws=tensor_dq_ws,
            tensor_dk_ws=tensor_dk_ws,
            tensor_dv_ws=tensor_dv_ws,
        )
        buf = pl.make_tuple(
            mm1_res=mm1_res,
            mm2_res=mm2_res,
            q_l1=q_l1,
            k_l1=k_l1,
            k_l1_n=k_l1_n,
            v_l1=v_l1,
            do_l1=do_l1,
            q_l1_dk=q_l1_dk,
            do_l1_dv=do_l1_dv,
            left=left,
            right=right,
            acc=acc,
            dk_acc0=dk_acc0,
            dv_acc0=dv_acc0,
            dk_acc1=dk_acc1,
            dv_acc1=dv_acc1,
            ds_l1=ds_l1,
            ds_t_l1=ds_t_l1,
            p_t_l1=p_t_l1,
        )
        if deter == 1:
            _run_preload3_cube_deter(constInfo, sched, gm, buf, runInfos)
        else:
            _run_preload3_cube_normal(constInfo, sched, gm, buf, runInfos)

    with pl.section_vector():
        subIdx = pl.get_subblock_idx()
        y_ub = make_y_ub(v1w)
        dy_ub = make_dy_ub(v1w)
        ds_out_ub = make_ds_out_ub(v1w)
        p_out_ub = make_p_out_ub(v1w)
        if (
            template == Tmpl.BN2GS1S2
            and d_align == 192
            and dv_align == 128
            and deter == 1
            and layout == 0
            and mask_mode == 0
            and sink == 0
            and (
                (swizzle == 1 and deter_family == 1)
                or (swizzle == 0 and deter_family == 21)
            )
        ):
            sfmg_ub = make_sfmg_prefetch_ub(v1w)
        else:
            sfmg_ub = make_sfmg_ub(v1w)
        lse_ub = make_lse_ub(v1w)
        sink_part_ub = make_sink_part_ub(v1w)
        mask_ub = make_mask_view(v1w)
        mask_pre_ub = make_mask_pre_view(v1w)

        runInfos = make_run_infos()
        sched = pl.make_tuple(
            meta=meta,
            fagOff=fagOff,
            cBlockIdx=cBlockIdx,
            coreNum=coreNum,
            cuQ=cu_q_tensor,
            cuKv=cu_kv_tensor,
            usedQ=used_q_tensor,
            usedKv=used_kv_tensor,
            blockStart=blockStart,
            blockEnd=blockEnd,
            numBlocks=numBlocks,
            subIdx=subIdx,
            v1w=v1w,
        )
        gm = pl.make_tuple(
            tensor_y=tensor_y,
            tensor_dy=tensor_dy,
            tensor_lse=tensor_lse,
            tensor_mask=tensor_mask,
            tensor_sinks=tensor_sinks,
            tensor_dsink_ws=tensor_dsink_ws,
            tensor_sfmg_ws=tensor_sfmg_ws,
        )
        buf = pl.make_tuple(
            mm1_res=mm1_res,
            mm2_res=mm2_res,
            y_ub=y_ub,
            dy_ub=dy_ub,
            ds_out_ub=ds_out_ub,
            p_out_ub=p_out_ub,
            sfmg_ub=sfmg_ub,
            lse_ub=lse_ub,
            sink_part_ub=sink_part_ub,
            mask_ub=mask_ub,
            mask_pre_ub=mask_pre_ub,
            ds_l1=ds_l1,
            p_l1=p_l1,
        )
        if deter == 1:
            _run_preload3_vector_deter(constInfo, sched, gm, buf, runInfos, sink)
        else:
            _run_preload3_vector_normal(constInfo, sched, gm, buf, runInfos, sink)

    # ==== Post 阶段：dQ/dK/dV 的 scale + cast fp32->16-bit ====
    # scale 在矩阵乘之后、fp32 结果上施加(而非提前乘进 dS)：先乘会让
    # 16-bit 舍入发生在不同量级上，网络训练中误差会累积。
    # 只有 dQ/dK 乘 scale，dV 不乘（传 1.0，fp32 乘 1.0 精确）。
    # 进 post 前必须 sync_all：要等所有核的 atomicAdd 都落盘。
    pl.system.sync_all()

    with pl.section_vector():
        vBlockIdxP = pl.get_block_idx()
        dqRowStartP = (
            pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdxP) * one64
        )
        dqRowEndP = pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdxP) * one64
        dkvRowStartP = (
            pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdxP) * one64
        )
        dkvRowEndP = (
            pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdxP) * one64
        )
        if nz_out == 1:
            post_f32 = make_nz_post_f32_buffer()
            post_f16 = make_nz_post_f16_buffer()
            process_nz_post(
                constInfo,
                cu_q_tensor,
                used_q_tensor,
                constInfo.sequsedQOptional,
                tensor_dq_ws_flat,
                tensor_dq_out,
                post_f32,
                post_f16,
                constInfo.n1Size,
                constInfo.s1Size,
                constInfo.coefB0Q,
                constInfo.scaleValue,
            )
            process_nz_post(
                constInfo,
                cu_kv_tensor,
                used_kv_tensor,
                constInfo.sequsedKvOptional,
                tensor_dk_ws_flat,
                tensor_dk_out,
                post_f32,
                post_f16,
                constInfo.n2Size,
                constInfo.s2Size,
                constInfo.coefB0KV,
                constInfo.scaleValue,
            )
            process_nz_post(
                constInfo,
                cu_kv_tensor,
                used_kv_tensor,
                constInfo.sequsedKvOptional,
                tensor_dv_ws_flat,
                tensor_dv_out,
                post_f32,
                post_f16,
                constInfo.n2Size,
                constInfo.s2Size,
                constInfo.coefB0KV,
                1.0,
            )
        else:
            post_f32 = make_post_f32_buffer()
            post_f16 = make_post_f16_buffer()
            process_post(
                tensor_dq_ws_flat,
                tensor_dq_out,
                post_f32,
                post_f16,
                constInfo.scaleValue,
                dqRowStartP * dQ,
                (dqRowEndP - dqRowStartP) * dQ,
            )
            process_post(
                tensor_dk_ws_flat,
                tensor_dk_out,
                post_f32,
                post_f16,
                constInfo.scaleValue,
                dkvRowStartP * dQ,
                (dkvRowEndP - dkvRowStartP) * dQ,
            )
            process_post(
                tensor_dv_ws_flat,
                tensor_dv_out,
                post_f32,
                post_f16,
                1.0,
                dkvRowStartP * dV,
                (dkvRowEndP - dkvRowStartP) * dV,
            )
        if sink == 1:
            if pl.get_block_idx() == 0:
                sink_red_ub = make_sink_red_post_ub(v1w)
                sink_row_ub = make_sink_row_post_ub(v1w)
                process_sink_reduce(
                    constInfo,
                    tensor_dsink_ws,
                    tensor_dsink_out,
                    sink_red_ub.current(),
                    sink_row_ub,
                    constInfo.aivCount,
                )


def flash_attn_grad_preload2(
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    # TND 变长：与 preload3 同款，循环外创建视图，非 TND 时为 null（不读）。
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    dsink: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
    d_align,
    dv_align,
    swizzle,
    deter,
    sink,
):
    """1-ahead 入口：建视图、pre/post，并按 template/deter 分发主循环。

    cube/vector 主循环已拆成 BN2、GS normal、GS deterministic 三套。
    路径按编译期 ``template`` / ``deter`` 分发，不是按 D 档。

    ``template!=Tmpl.BN2``（GS1S2，含 D=192/256/768 与 64/128 两档）：
    store 到 fp32 workspace；进循环前清 workspace，循环后再 scale+cast。
    不含 BN2 的 UB 直出（v5/v6）。
    ``Tmpl.BN2``：一个核独占整个 (b, n2) head；不跑全局 workspace pre/post
    （仅 ``bn2_need_zero`` 清 16-bit out，带 sink 时另清 dsink）。
    ``d_align<=128`` 的结果在 UB 里 Muls+Cast 直出；``d_align>128``（wide）只有
    经 4D fp32 workspace 的那部分由 vector 逐 D 分块读回 cast：dQ 三档都走
    workspace；dK 仅 256/768 走（192 驻留 L0C，flush 时搬 UB 再 cast）；dV 仅
    768 走（192/256 驻留 L0C，flush 时 cube fixpipe 直出）。首写仍用 AtomicNone
    覆盖，所以同样不需要全局清零。

    MultiBlk（is_bn2_multiblk）：核内按 s2 逐块累加，dQ 首有效 s2 覆盖
    workspace、其余 AtomicAdd，vector 在 ``dqLast`` 时从 4D workspace /
    per-core 2D 条带 load 后 Muls+Cast；dK/dV 在 L0C 累加到换 s2 再写回
    （dK 只有 256/768 档直接累加进 workspace；dV 192/256 仍走 L0C）。
    """
    coreNum = pl.get_block_num()
    # swizzle 来自 tilingkey。BN2 is_valid 保证为 0，与原先写死 0 等价。
    cu_q_tensor = pl.make_tensor(cu_seqlens_q, [tiling.b + 1], [1], dtype=pl.DT_INT32)
    cu_kv_tensor = pl.make_tensor(cu_seqlens_kv, [tiling.b + 1], [1], dtype=pl.DT_INT32)
    used_q_tensor = pl.make_tensor(seqused_q, [tiling.b], [1], dtype=pl.DT_INT32)
    used_kv_tensor = pl.make_tensor(seqused_kv, [tiling.b], [1], dtype=pl.DT_INT32)
    # 与 preload3 共用逻辑长度 / GM 视图长度的初始化约定。
    constInfo = set_const_info(tiling, coreNum, cu_q_tensor, cu_kv_tensor)

    dQ = constInfo.dSize
    viewD2Q = constInfo.viewD2Q
    d2dQ = viewD2Q * dQ
    s1ndQ = constInfo.s1ViewSize * d2dQ
    if template == Tmpl.BN2:
        # BN2 has no GQA; Q/K share head axes, but V/dY/dV use the actual Dv.
        dV = constInfo.dvSize
        viewD0KV = constInfo.viewD0Q
        viewD2KV = constInfo.viewD2Q
    else:
        dV = constInfo.dvSize
        viewD0KV = constInfo.viewD0KV
        viewD2KV = constInfo.viewD2KV
    d2dKV = viewD2KV * dQ
    s2ndKV = constInfo.s2ViewSize * d2dKV
    d2dQv = viewD2Q * dV
    s1ndQv = constInfo.s1ViewSize * d2dQv
    d2dKVv = viewD2KV * dV
    s2ndKVv = constInfo.s2ViewSize * d2dKVv

    tensor_q = pl.make_tensor(
        q,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
        [s1ndQ, d2dQ, dQ, 1],
        dtype=input_dtype,
    )
    tensor_k = pl.make_tensor(
        k,
        [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=input_dtype,
    )
    # MM3 需要未转置的 K，必须与 MM2 的 K^T 视图分开(layout 各自推导)
    tensor_k_nt = pl.make_tensor(
        k,
        [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=input_dtype,
    )
    tensor_v = pl.make_tensor(
        v,
        [viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
        [s2ndKVv, d2dKVv, dV, 1],
        dtype=input_dtype,
    )
    tensor_dy = pl.make_tensor(
        dout,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=input_dtype,
    )
    tensor_y = pl.make_tensor(
        attn_out,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=input_dtype,
    )
    # BN2 无 GQA（g=1），n1==n2，与原先 [B,n2,S1] 等价。
    n1Size = constInfo.n2Size * constInfo.gSize
    if layout == 1:
        # TND：LSE 为 [N1,T1]，head stride 必须是实际 token 总数。
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [n1Size, constInfo.t1Size],
            [constInfo.t1Size, 1],
            dtype=pl.DT_FP32,
        )
    else:
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [constInfo.bSize, n1Size, constInfo.s1Size],
            [n1Size * constInfo.s1Size, constInfo.s1Size, 1],
            dtype=pl.DT_FP32,
        )

    ws_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    # 与 preload3 / host 共用物理 token 口径；BN2 也用它定位 dsink 槽位。
    wsRowsQ = constInfo.t1Size * constInfo.n1Size
    wsRowsKV = constInfo.t2Size * constInfo.n2Size
    if template == Tmpl.BN2:
        # BN2 直出 16-bit。d_align<=128 的中间结果在 mm1/mm2 UB（MultiBlk 的
        # dQ 走 per-core 2D 条带）；d_align>128 放不下的段改走与 GS1S2 同布局的
        # 4D fp32 workspace：dQ 三档全走，dK 走 256/768，dV 只走 768（192 的
        # dK 与 192/256 的 dV 驻留 L0C）。workspace 只作本核 staging，不做全局
        # pre/post：cube 首写 AtomicNone、vector 内联 cast。
        tensor_dq_out = pl.make_tensor(
            dq,
            [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
            [s1ndQ, d2dQ, dQ, 1],
            dtype=input_dtype,
        )
        tensor_dk_out = pl.make_tensor(
            dk,
            [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
            [s2ndKV, d2dKV, dQ, 1],
            dtype=input_dtype,
        )
        tensor_dv_out = pl.make_tensor(
            dv,
            [viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
            [s2ndKVv, d2dKVv, dV, 1],
            dtype=input_dtype,
        )
        if d_align > 128:
            # BN2 wide：dq/dk/dv 的 4D fp32 workspace。分段与 GS1S2 一致，
            # host BuildWorkspacePlan 不必改。
            dq_workspace = ws_ptr
            dk_workspace = pl.addptr(dq_workspace, wsRowsQ * tiling.d)
            dv_workspace = pl.addptr(dk_workspace, wsRowsKV * tiling.d)
            tensor_dq_ws = pl.make_tensor(
                dq_workspace,
                [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
                [s1ndQ, d2dQ, dQ, 1],
                dtype=pl.DT_FP32,
            )
            tensor_dk_ws = pl.make_tensor(
                dk_workspace,
                [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
                [s2ndKV, d2dKV, dQ, 1],
                dtype=pl.DT_FP32,
            )
            tensor_dv_ws = pl.make_tensor(
                dv_workspace,
                [viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
                [s2ndKVv, d2dKVv, dV, 1],
                dtype=pl.DT_FP32,
            )
            # Keep one D-based UB layout for the wide BN2 main/scratch aliases.
            # D192/Dv128 loads 128 valid columns into 192-column physical rows.
            v1w = 256 if d_align >= 512 else d_align
        else:
            # 单块 BN2 的 C3 被 tilingkey 剪掉 store，不能把 UB
            # 或 16-bit out 冒充成 fp32 GM 参数。dQ 用 per-core 条带。
            alignS1 = constInfo.s1Outer * CUBE_BASEM
            tensor_dq_ws = pl.make_tensor(
                ws_ptr, [constInfo.coreNum * alignS1, dQ], [dQ, 1], dtype=pl.DT_FP32
            )
    else:
        # GS1S2：三段 fp32 累加区，分段必须与 host CalcWorkSpace 对齐。
        dq_workspace = ws_ptr
        dk_workspace = pl.addptr(dq_workspace, wsRowsQ * tiling.d)
        dv_workspace = pl.addptr(dk_workspace, wsRowsKV * tiling.d)
        tensor_dq_ws = pl.make_tensor(
            dq_workspace,
            [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
            [s1ndQ, d2dQ, dQ, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dk_ws = pl.make_tensor(
            dk_workspace,
            [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
            [s2ndKV, d2dKV, dQ, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dv_ws = pl.make_tensor(
            dv_workspace,
            [viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
            [s2ndKVv, d2dKVv, dV, 1],
            dtype=pl.DT_FP32,
        )
        dqElems = constInfo.t1Size * n1Size * dQ
        dkElems = constInfo.t2Size * constInfo.n2Size * dQ
        dvElems = constInfo.t2Size * constInfo.n2Size * dV
        tensor_dq_ws_flat = pl.make_tensor(
            dq_workspace, [1, dqElems], [dqElems, 1], dtype=pl.DT_FP32
        )
        tensor_dk_ws_flat = pl.make_tensor(
            dk_workspace, [1, dkElems], [dkElems, 1], dtype=pl.DT_FP32
        )
        tensor_dv_ws_flat = pl.make_tensor(
            dv_workspace, [1, dvElems], [dvElems, 1], dtype=pl.DT_FP32
        )
        tensor_dq_out = pl.make_tensor(
            dq, [1, dqElems], [dqElems, 1], dtype=input_dtype
        )
        tensor_dk_out = pl.make_tensor(
            dk, [1, dkElems], [dkElems, 1], dtype=input_dtype
        )
        tensor_dv_out = pl.make_tensor(
            dv, [1, dvElems], [dvElems, 1], dtype=input_dtype
        )
        # V1 列宽：768 档工作 tile 仍是 256，其余等于 dv_align。
        v1w = 256 if dv_align >= 512 else dv_align

    if mask_mode != 0:
        tensor_mask = pl.make_tensor(
            attn_mask,
            [ATTEN_MASK_COMPRESS, ATTEN_MASK_COMPRESS],
            [ATTEN_MASK_COMPRESS, 1],
            dtype=pl.DT_UINT8,
        )
    else:
        tensor_mask = tensor_lse

    dsink_workspace = pl.addptr(
        ws_ptr,
        wsRowsQ * tiling.d
        + wsRowsKV * tiling.d
        + wsRowsKV * tiling.dv
        + 128 * pl.max(tiling.d, tiling.dv),
    )
    tensor_sinks = pl.make_tensor(sinks, [constInfo.n1Size], [1], dtype=pl.DT_FP32)
    dsinkElems = constInfo.aivCount * constInfo.n1Size
    tensor_dsink_ws = pl.make_tensor(
        dsink_workspace, [1, dsinkElems], [dsinkElems, 1], dtype=pl.DT_FP32
    )
    tensor_dsink_out = pl.make_tensor(
        dsink, [1, constInfo.n1Size], [constInfo.n1Size, 1], dtype=pl.DT_FP32
    )
    # sfmg 前移缓存：[1, N1*T1] fp32 扁平（head*T1 + 全局 token 行），与
    # host 的 plan.sfmg 对齐。是否真正写/读由运行时 constInfo.enablePreSfmg
    # 决定（host 按 fasg enablePreSfmg 判据裁决）：
    #   GS1S2 + dv_align>=128 —— 可能前移，建 ws 视图；
    #   BN2（含 multiblk）/ dv_align=64 —— 不前移，占位必须是 rank-2 扁平
    #     视图（非 TND 的 tensor_lse 是 [B,N1,S1] 三维，直接占位会因 load
    #     offsets 数与源 rank 不一致在 trace 期报 INVALID_SHAPE），内联
    #     分支不解引用。
    if template != Tmpl.BN2GS1S2 or dv_align < 128:
        lseElems = n1Size * constInfo.t1Size
        tensor_sfmg_ws = pl.make_tensor(
            softmax_lse, [1, lseElems], [lseElems, 1], dtype=pl.DT_FP32
        )
    else:
        sfmgElems = n1Size * constInfo.t1Size
        tensor_sfmg_ws = pl.make_tensor(
            pl.addptr(dsink_workspace, dsinkElems),
            [1, sfmgElems],
            [sfmgElems, 1],
            dtype=pl.DT_FP32,
        )

    # AIV get_block_idx 按 vector 子核计数，必须换算成物理核，否则
    # cube/vector 的 numBlocks 不一致，set/wait 配不上。
    cBlockIdx = pl.get_block_idx() // pl.get_subblock_num()
    coreNum = constInfo.coreNum
    meta = pl.make_tensor(
        metadata, [pl.max(constInfo.metadataWordCount, 1)], [1], dtype=pl.DT_INT32
    )
    one64 = pl.const(1, pl.DT_INT64)
    fagOff = pl.getval(meta, META_HEAD_FAG_START_INDEX) * one64
    blockStart = (
        pl.getval(meta, fagOff + META_FAG_BLOCK_STARTS_OFFSET + cBlockIdx) * one64
    )
    blockEnd = pl.getval(meta, fagOff + META_FAG_BLOCK_ENDS_OFFSET + cBlockIdx) * one64
    # swizzle==0（含 BN2）时就是 max(end-start, 0)。
    numBlocks = block_range_of(constInfo, cBlockIdx, coreNum, blockStart, blockEnd)

    mm1_res = make_mm1_res()
    mm2_res = make_mm2_res()
    ds_l1 = make_ds_l1()
    p_l1 = make_p_l1()
    ds_t_l1 = make_ds_t_l1()
    p_t_l1 = make_p_t_l1()

    if template == Tmpl.BN2:
        # skip_invalid 不写整行/整列无效 tile，out GM 会留脏数据。
        # 三个 out 都清，不分行/列；不清 MultiBlk workspace。
        if bn2_need_zero == 1:
            with pl.section_vector():
                n1Size = constInfo.n2Size * constInfo.gSize
                dqElems = constInfo.t1Size * n1Size * constInfo.dSize
                dkElems = constInfo.t2Size * constInfo.n2Size * constInfo.dSize
                dvElems = constInfo.t2Size * constInfo.n2Size * constInfo.dvSize
                tensor_dq_flat = pl.make_tensor(
                    dq, [1, dqElems], [dqElems, 1], dtype=input_dtype
                )
                tensor_dk_flat = pl.make_tensor(
                    dk, [1, dkElems], [dkElems, 1], dtype=input_dtype
                )
                tensor_dv_flat = pl.make_tensor(
                    dv, [1, dvElems], [dvElems, 1], dtype=input_dtype
                )
                zero_ub = make_pre_zero_buffer_f16()
                if layout == 1 and is_bn2_multiblk == 0:
                    # 变长 TND：只清“空 batch 对侧”的行，运行期按 cu_seqlens 定位。
                    zero_tnd_empty_batch(
                        constInfo,
                        cu_q_tensor,
                        cu_kv_tensor,
                        tensor_dq_flat,
                        tensor_dk_flat,
                        tensor_dv_flat,
                        zero_ub.current(),
                    )
                else:
                    process_pre_from_meta(
                        meta,
                        fagOff,
                        tensor_dq_flat,
                        tensor_dk_flat,
                        tensor_dv_flat,
                        constInfo.dSize,
                        constInfo.dvSize,
                        zero_ub.current(),
                    )
                if sink == 1:
                    sink_zero_ub = make_pre_zero_buffer_f32()
                    process_sink_zero(
                        tensor_dsink_ws,
                        constInfo,
                        sink_zero_ub.current(),
                        pl.get_block_idx(),
                    )
            pl.system.sync_all()
        else:
            # No output pre-clear and no sink: BN2 needs no pre barrier.
            if sink == 1:
                with pl.section_vector():
                    sink_zero_ub = make_pre_zero_buffer_f32()
                    process_sink_zero(
                        tensor_dsink_ws,
                        constInfo,
                        sink_zero_ub.current(),
                        pl.get_block_idx(),
                    )
                pl.system.sync_all()
        # BN2（含 multiblk）不做 sfmg 前移：对齐 fasg enablePreSfmg 的
        # splitAxis 门（仅 BN2GS1S2 / BN2S2；FAG 无 BN2S2）。BN2 的 rowsum
        # 仍在主循环 V1 内联，多 s2Outer 的重复计算保留，行为与 fasg 一致。
        # 上面的 need_zero/sink 子支各自带 sync_all（或空跑无清零），进主
        # 循环前无需额外 barrier。
    else:
        # GS1S2：三段 fp32 累加区清零；dv_align>=128 走交错 pre，是否追加
        # softmaxGrad 前移由运行时 constInfo.enablePreSfmg 决定（对齐 fasg
        # enablePreSfmg）：为 1 时一次算全 rowsum(dO⊙O) 存 ws、V1 只读，
        # 为 0 时主循环 V1 内联重算（本分支只做清零）。
        # 交错 pre（同 preload3）：zt 32KB 避开 sp 的窗口，无 UB 冒险；
        # zero 与 sp 不同 mutex，框架不插 wait——清零写(MTE3)与 presfmg
        # 载入(MTE2)/VF 并行，且分块交替 issue（见 process_presfmg 的
        # _zero_step）。本函数（preload2）的 GS1S2 租户是 256/768 档——sp
        # 用半窗塞进空洞。
        if dv_align >= 128:
            with pl.section_vector():
                zero_ub = make_pre_zero_buffer_f32()
                zt = zero_ub.current()
                # 行区间与三段边界（同 preload3，见 process_pre_from_meta 文档）。
                vBlockIdx = pl.get_block_idx()
                one64 = pl.const(1, pl.DT_INT64)
                dqRowStart = (
                    pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdx) * one64
                )
                dqRowEnd = (
                    pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdx) * one64
                )
                dkvRowStart = (
                    pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdx)
                    * one64
                )
                dkvRowEnd = (
                    pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdx) * one64
                )
                dqNElems = (dqRowEnd - dqRowStart) * dQ
                dkNElems = (dkvRowEnd - dkvRowStart) * dQ
                dvNElems = (dkvRowEnd - dkvRowStart) * dV
                preElems = pl.min(
                    PRE_ELEMS[d_align], pl.max(dqNElems, pl.max(dkNElems, dvNElems))
                )
                fill_zero_vf(zt, (preElems + VF_LANES - 1) // VF_LANES)
                if sink == 1:
                    process_sink_zero(
                        tensor_dsink_ws, constInfo, zt, pl.get_block_idx()
                    )
                sp_y_ub = make_sp_y_ub(v1w)
                sp_dy_ub = make_sp_dy_ub(v1w)
                sp_out_ub = make_sp_out_ub(v1w)
                sp_out_flat = make_sp_out_flat(v1w)
                process_presfmg(
                    constInfo,
                    tensor_y,
                    tensor_dy,
                    tensor_sfmg_ws,
                    sp_y_ub,
                    sp_dy_ub,
                    sp_out_ub,
                    sp_out_flat,
                    fag_common.SP_WORK_W[v1w],
                    zt,
                    tensor_dq_ws_flat,
                    tensor_dk_ws_flat,
                    tensor_dv_ws_flat,
                    dqRowStart * dQ,
                    dqNElems,
                    dkvRowStart * dQ,
                    dqNElems + dkNElems,
                    dkvRowStart * dV,
                    dqNElems + dkNElems + dvNElems,
                )
        else:
            with pl.section_vector():
                zero_ub = make_pre_zero_buffer_f32()
                process_pre_from_meta(
                    meta,
                    fagOff,
                    tensor_dq_ws_flat,
                    tensor_dk_ws_flat,
                    tensor_dv_ws_flat,
                    dQ,
                    dV,
                    zero_ub.current(),
                )
                if sink == 1:
                    process_sink_zero(
                        tensor_dsink_ws,
                        constInfo,
                        zero_ub.current(),
                        pl.get_block_idx(),
                    )
        pl.system.sync_all()

    with pl.section_cube():
        q_l1 = make_q_l1()
        k_l1 = make_k_l1()
        k_l1_n = make_k_nd_l1()
        v_l1 = make_v_l1()
        do_l1 = make_do_l1()
        left = make_l0a()
        right = make_l0b()
        # 192 档 L0C 的两套布局由工厂内部按 template/d_align 选（BN2 用 dK/dV
        # 常驻 256KB，GS1S2 用 acc 双槽 224KB）；其它档两套指向同一份布局。
        acc = make_l0c_acc()
        if template == Tmpl.BN2:
            # dK/dV 每个 D 分块一个槽（c0 = 第 0 块、c1 = 第 1 块）；64/128/768 只有
            # 一个分块，c1 别名到 c0、trace 期剪掉。L0C 占用（单位 KB，额 256KB）：
            #   64/128   : acc 2×64 + dK 1×64 + dV 1×64 = 256，各自常驻累加
            #   192 BN2  : acc 1×64 + dK 2×48 + dV 2×48 = 256，dK/dV 常驻 L0C
            #   192 GS1S2: acc 2×64 + dV 2×48 = 224，dK 与 acc 同址（mm4 → fp32 ws）
            #   256      : acc 2×64 + dV 2×64 = 256，dK 与 acc 第 0 槽同址（→ fp32 ws）
            #   768      : acc 2×64 + dK 1×64 + dV 1×64 = 256，dK/dV 单 scratch → fp32 ws
            dk_c0 = make_dk_l0c_chunk0()
            dk_c1 = make_dk_l0c_chunk1()
            dv_c0 = make_dv_l0c_chunk0()
            dv_c1 = make_dv_l0c_chunk1()
            # BN2 192档使用，dK 从 L0C 搬到 UB，chunk0 → mm2_res、chunk1 → l0c_load_ub
            l0c_load_ub = make_l0c_load_ub()
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                fagOff=fagOff,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                blockStart=blockStart,
                blockEnd=blockEnd,
                numBlocks=numBlocks,
            )
            if d_align > 128:
                gm = pl.make_tuple(
                    tensor_q=tensor_q,
                    tensor_k=tensor_k,
                    tensor_k_nt=tensor_k_nt,
                    tensor_dy=tensor_dy,
                    tensor_v=tensor_v,
                    tensor_dq_ws=tensor_dq_ws,
                    tensor_dk_ws=tensor_dk_ws,
                    tensor_dv_ws=tensor_dv_ws,
                    tensor_dv_out=tensor_dv_out,
                )
            else:
                gm = pl.make_tuple(
                    tensor_q=tensor_q,
                    tensor_k=tensor_k,
                    tensor_k_nt=tensor_k_nt,
                    tensor_dy=tensor_dy,
                    tensor_v=tensor_v,
                    tensor_dq_ws=tensor_dq_ws,
                    tensor_dv_out=tensor_dv_out,
                )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                q_l1=q_l1,
                k_l1=k_l1,
                k_l1_n=k_l1_n,
                v_l1=v_l1,
                do_l1=do_l1,
                left=left,
                right=right,
                acc=acc,
                dk_c0=dk_c0,
                dk_c1=dk_c1,
                dv_c0=dv_c0,
                dv_c1=dv_c1,
                l0c_load_ub=l0c_load_ub,
                ds_l1=ds_l1,
                ds_t_l1=ds_t_l1,
                p_t_l1=p_t_l1,
            )
            _run_preload2_cube_bn2(constInfo, sched, gm, buf, runInfos)
        elif deter == 1:
            q_l1_dk = make_q_lag_view()
            do_l1_dv = make_do_lag_view()
            dv_acc0 = make_dv_l0c_chunk0()
            dv_acc1 = make_dv_l0c_chunk1()
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
            )
            gm = pl.make_tuple(
                tensor_q=tensor_q,
                tensor_k=tensor_k,
                tensor_k_nt=tensor_k_nt,
                tensor_dy=tensor_dy,
                tensor_v=tensor_v,
                tensor_dq_ws=tensor_dq_ws,
                tensor_dk_ws=tensor_dk_ws,
                tensor_dv_ws=tensor_dv_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                q_l1=q_l1,
                k_l1=k_l1,
                k_l1_n=k_l1_n,
                v_l1=v_l1,
                do_l1=do_l1,
                q_l1_dk=q_l1_dk,
                do_l1_dv=do_l1_dv,
                dv_acc0=dv_acc0,
                dv_acc1=dv_acc1,
                left=left,
                right=right,
                acc=acc,
                ds_l1=ds_l1,
                ds_t_l1=ds_t_l1,
                p_t_l1=p_t_l1,
            )
            _run_preload2_cube_gs_deter(constInfo, sched, gm, buf, runInfos)
        else:
            dv_acc0 = make_dv_l0c_chunk0()
            dv_acc1 = make_dv_l0c_chunk1()
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                fagOff=fagOff,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                blockStart=blockStart,
                blockEnd=blockEnd,
                numBlocks=numBlocks,
            )
            gm = pl.make_tuple(
                tensor_q=tensor_q,
                tensor_k=tensor_k,
                tensor_k_nt=tensor_k_nt,
                tensor_dy=tensor_dy,
                tensor_v=tensor_v,
                tensor_dq_ws=tensor_dq_ws,
                tensor_dk_ws=tensor_dk_ws,
                tensor_dv_ws=tensor_dv_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                q_l1=q_l1,
                k_l1=k_l1,
                k_l1_n=k_l1_n,
                v_l1=v_l1,
                do_l1=do_l1,
                dv_acc0=dv_acc0,
                dv_acc1=dv_acc1,
                left=left,
                right=right,
                acc=acc,
                ds_l1=ds_l1,
                ds_t_l1=ds_t_l1,
                p_t_l1=p_t_l1,
            )
            _run_preload2_cube_gs_normal(constInfo, sched, gm, buf, runInfos)

    with pl.section_vector():
        subIdx = pl.get_subblock_idx()
        if template == Tmpl.BN2 and d_align > 128:
            y_ub = make_y_ub(v1w)
            dy_ub = make_dy_ub(v1w)
            nzp_ub = make_nzp_ub(v1w)
            sfmg_ub = make_sfmg_ub(v1w)
            lse_ub = make_lse_ub(v1w)
            mask_ub = make_mask_view(v1w)
            mask_pre_ub = make_mask_pre_view(v1w)
            # gm_ws -> ws_load_ub -> muls_cast_vf -> ws_cast_ub -> gm_out
            ws_load_ub = make_ws_load_f32_ub()
            ws_cast_ub = make_ws_cast_f16_ub()
            # BN2 192档使用，dK 从 L0C 搬到 UB，chunk0 → mm2_res、chunk1 → l0c_load_ub
            l0c_load_ub = make_l0c_load_ub()
            l0c_cast_ub = make_l0c_cast_ub()
            sink_part_ub = make_sink_part_ub(v1w)
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                fagOff=fagOff,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                blockStart=blockStart,
                blockEnd=blockEnd,
                numBlocks=numBlocks,
                subIdx=subIdx,
                dv_align=dv_align,
                v1w=v1w,
            )
            gm = pl.make_tuple(
                tensor_y=tensor_y,
                tensor_dy=tensor_dy,
                tensor_lse=tensor_lse,
                tensor_mask=tensor_mask,
                tensor_dq_out=tensor_dq_out,
                tensor_dk_out=tensor_dk_out,
                tensor_dv_out=tensor_dv_out,
                tensor_dq_ws=tensor_dq_ws,
                tensor_dk_ws=tensor_dk_ws,
                tensor_dv_ws=tensor_dv_ws,
                tensor_sinks=tensor_sinks,
                tensor_dsink_ws=tensor_dsink_ws,
                tensor_sfmg_ws=tensor_sfmg_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                y_ub=y_ub,
                dy_ub=dy_ub,
                nzp_ub=nzp_ub,
                sfmg_ub=sfmg_ub,
                lse_ub=lse_ub,
                sink_part_ub=sink_part_ub,
                mask_ub=mask_ub,
                mask_pre_ub=mask_pre_ub,
                ws_load_ub=ws_load_ub,
                ws_cast_ub=ws_cast_ub,
                l0c_cast_ub=l0c_cast_ub,
                l0c_load_ub=l0c_load_ub,
                ds_l1=ds_l1,
                p_l1=p_l1,
            )
            _run_preload2_vector_bn2(constInfo, sched, gm, buf, runInfos, sink)
        elif template == Tmpl.BN2:
            # narrow：d==dv 由 host 门 Bn2FamilyFeasible 保证，故 v1w == d_align。
            # 这里一律传 d_align（tilingkey 常量）而不是 v1w：pypto 的定值分析认为
            # v1w 在本分支「可能未定义」（旧代码此分支传的是常量 HEAD_DIM_ALIGN，
            # 所以没暴露过），会报 NameNotFound: __inline_0_v1w。
            # 地址档由各工厂内部按 template/d_align 判成 HEAD_DIM_ALIGN。
            y_ub = make_y_ub(d_align)
            dy_ub = make_dy_ub(d_align)
            nzp_ub = make_nzp_ub(d_align)
            sfmg_ub = make_sfmg_ub(d_align)
            lse_ub = make_lse_ub(d_align)
            mask_ub = make_mask_view(d_align)
            mask_pre_ub = make_mask_pre_view(d_align)
            l0c_cast_ub = make_l0c_cast_ub()
            # 同上：narrow 不实际使用，但 buf 字段要齐（函数顶部无条件 unpack）
            l0c_load_ub = make_l0c_load_ub()
            ws_cast_ub = make_ws_cast_f16_ub()
            sink_part_ub = make_sink_part_ub(d_align)
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                fagOff=fagOff,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                blockStart=blockStart,
                blockEnd=blockEnd,
                numBlocks=numBlocks,
                subIdx=subIdx,
                dv_align=dv_align,
            )
            gm = pl.make_tuple(
                tensor_y=tensor_y,
                tensor_dy=tensor_dy,
                tensor_lse=tensor_lse,
                tensor_mask=tensor_mask,
                tensor_dq_out=tensor_dq_out,
                tensor_dk_out=tensor_dk_out,
                tensor_dq_ws=tensor_dq_ws,
                tensor_sinks=tensor_sinks,
                tensor_dsink_ws=tensor_dsink_ws,
                tensor_sfmg_ws=tensor_sfmg_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                y_ub=y_ub,
                dy_ub=dy_ub,
                nzp_ub=nzp_ub,
                sfmg_ub=sfmg_ub,
                lse_ub=lse_ub,
                sink_part_ub=sink_part_ub,
                mask_ub=mask_ub,
                mask_pre_ub=mask_pre_ub,
                ws_cast_ub=ws_cast_ub,
                l0c_cast_ub=l0c_cast_ub,
                l0c_load_ub=l0c_load_ub,
                ds_l1=ds_l1,
                p_l1=p_l1,
            )
            _run_preload2_vector_bn2(constInfo, sched, gm, buf, runInfos, sink)
        elif deter == 1:
            y_ub = make_y_ub(v1w)
            dy_ub = make_dy_ub(v1w)
            ds_out_ub = make_ds_out_ub(v1w)
            p_out_ub = make_p_out_ub(v1w)
            sfmg_ub = make_sfmg_ub(v1w)
            lse_ub = make_lse_ub(v1w)
            sink_part_ub = make_sink_part_ub(v1w)
            mask_ub = make_mask_view(v1w)
            mask_pre_ub = make_mask_pre_view(v1w)
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                subIdx=subIdx,
                v1w=v1w,
            )
            gm = pl.make_tuple(
                tensor_y=tensor_y,
                tensor_dy=tensor_dy,
                tensor_lse=tensor_lse,
                tensor_mask=tensor_mask,
                tensor_sinks=tensor_sinks,
                tensor_dsink_ws=tensor_dsink_ws,
                tensor_sfmg_ws=tensor_sfmg_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                y_ub=y_ub,
                dy_ub=dy_ub,
                ds_out_ub=ds_out_ub,
                p_out_ub=p_out_ub,
                sfmg_ub=sfmg_ub,
                lse_ub=lse_ub,
                sink_part_ub=sink_part_ub,
                mask_ub=mask_ub,
                mask_pre_ub=mask_pre_ub,
                ds_l1=ds_l1,
                p_l1=p_l1,
            )
            _run_preload2_vector_gs_deter(constInfo, sched, gm, buf, runInfos, sink)
        else:
            y_ub = make_y_ub(v1w)
            dy_ub = make_dy_ub(v1w)
            ds_out_ub = make_ds_out_ub(v1w)
            p_out_ub = make_p_out_ub(v1w)
            sfmg_ub = make_sfmg_ub(v1w)
            lse_ub = make_lse_ub(v1w)
            sink_part_ub = make_sink_part_ub(v1w)
            mask_ub = make_mask_view(v1w)
            mask_pre_ub = make_mask_pre_view(v1w)
            runInfos = make_run_infos_2()
            sched = pl.make_tuple(
                meta=meta,
                fagOff=fagOff,
                cBlockIdx=cBlockIdx,
                coreNum=coreNum,
                cuQ=cu_q_tensor,
                cuKv=cu_kv_tensor,
                usedQ=used_q_tensor,
                usedKv=used_kv_tensor,
                blockStart=blockStart,
                blockEnd=blockEnd,
                numBlocks=numBlocks,
                subIdx=subIdx,
                v1w=v1w,
            )
            gm = pl.make_tuple(
                tensor_y=tensor_y,
                tensor_dy=tensor_dy,
                tensor_lse=tensor_lse,
                tensor_mask=tensor_mask,
                tensor_sinks=tensor_sinks,
                tensor_dsink_ws=tensor_dsink_ws,
                tensor_sfmg_ws=tensor_sfmg_ws,
            )
            buf = pl.make_tuple(
                mm1_res=mm1_res,
                mm2_res=mm2_res,
                y_ub=y_ub,
                dy_ub=dy_ub,
                ds_out_ub=ds_out_ub,
                p_out_ub=p_out_ub,
                sfmg_ub=sfmg_ub,
                lse_ub=lse_ub,
                sink_part_ub=sink_part_ub,
                mask_ub=mask_ub,
                mask_pre_ub=mask_pre_ub,
                ds_l1=ds_l1,
                p_l1=p_l1,
            )
            _run_preload2_vector_gs_normal(constInfo, sched, gm, buf, runInfos, sink)

    if template != Tmpl.BN2:
        pl.system.sync_all()
        with pl.section_vector():
            vBlockIdxP = pl.get_block_idx()
            dqRowStartP = (
                pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS + vBlockIdxP) * one64
            )
            dqRowEndP = (
                pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS + vBlockIdxP) * one64
            )
            dkvRowStartP = (
                pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS + vBlockIdxP) * one64
            )
            dkvRowEndP = (
                pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS + vBlockIdxP) * one64
            )
            post_f32 = make_post_f32_buffer()
            post_f16 = make_post_f16_buffer()
            process_post(
                tensor_dq_ws_flat,
                tensor_dq_out,
                post_f32,
                post_f16,
                constInfo.scaleValue,
                dqRowStartP * dQ,
                (dqRowEndP - dqRowStartP) * dQ,
            )
            process_post(
                tensor_dk_ws_flat,
                tensor_dk_out,
                post_f32,
                post_f16,
                constInfo.scaleValue,
                dkvRowStartP * dQ,
                (dkvRowEndP - dkvRowStartP) * dQ,
            )
            process_post(
                tensor_dv_ws_flat,
                tensor_dv_out,
                post_f32,
                post_f16,
                1.0,
                dkvRowStartP * dV,
                (dkvRowEndP - dkvRowStartP) * dV,
            )
            if sink == 1:
                if pl.get_block_idx() == 0:
                    sink_red_ub = make_sink_red_post_ub(v1w)
                    sink_row_ub = make_sink_row_post_ub(v1w)
                    process_sink_reduce(
                        constInfo,
                        tensor_dsink_ws,
                        tensor_dsink_out,
                        sink_red_ub.current(),
                        sink_row_ub,
                        constInfo.aivCount,
                    )
    if template == Tmpl.BN2:
        # BN2 没有全局 workspace post：dQ/dK/dV 都在本核内联 scale/cast 落盘；
        # 只有 dsink 需要一次全局归约。
        if sink == 1:
            pl.system.sync_all()
            with pl.section_vector():
                if pl.get_block_idx() == 0:
                    sink_red_ub = make_sink_red_post_ub(HEAD_DIM_ALIGN)
                    sink_row_ub = make_sink_row_post_ub(HEAD_DIM_ALIGN)
                    process_sink_reduce(
                        constInfo,
                        tensor_dsink_ws,
                        tensor_dsink_out,
                        sink_red_ub.current(),
                        sink_row_ub,
                        constInfo.aivCount,
                    )


# BN2S2 entry/pipeline: separate resource and synchronization semantics.


def process_bn2s2(
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dout: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    softmax_lse: pl.Ptr[pl.DT_UINT8],
    # TND 变长：与 preload3 同款，循环外创建视图，非 TND 时为 null（不读）。
    cu_seqlens_q: pl.Ptr[pl.DT_UINT8],
    cu_seqlens_kv: pl.Ptr[pl.DT_UINT8],
    seqused_q: pl.Ptr[pl.DT_UINT8],
    seqused_kv: pl.Ptr[pl.DT_UINT8],
    attn_mask: pl.Ptr[pl.DT_UINT8],
    metadata: pl.Ptr[pl.DT_UINT8],
    sinks: pl.Ptr[pl.DT_UINT8],
    dq: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    dsink: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: FlashAttnGradTilingData,
    d_align,
    dv_align,
    swizzle,
    deter,
    sink,
):
    """1-ahead 入口：建视图、pre/post，并按 template/deter 分发主循环。

    cube/vector 主循环已拆成 BN2、GS normal、GS deterministic、BN2S2 deterministic 四套。
    路径按编译期 template/deter 分发。

    BN2：一个核独占整个 (b, n2) head；不跑 ws pre/post（仅 ``bn2_need_zero`` 清 16-bit out）。

    GS1S2 / BN2S2：dq 走三段 fp32 ws 累加区，进循环前清 ws，循环后 post scale+cast。

    当前 BN2S2 + deter 在主入口优先进入 process_deter_plan，保留独立的
    metadata 计划、同步和片上资源。此处留存旧分支，不替代该确定性实现。

    BN2S2 + 非deter 复用 fag_kernel_norm 列步进主循环：d<=128 列末直出；
    d==192 列末 flush ws；d==256 的 dV 驻留而 dK 逐 task 原子写 ws；
    宽 D 双逐 task 原子写 ws。d>=192 在 sync_all 后 post 三段收尾，
    不使用曾撤回的列末事件+GM读取直出机制。

    MultiBlk（is_bn2_multiblk）：dQ 首有效 s2 覆盖 workspace，其余 AtomicAdd，
    vector 只在 isLastS1Outer 从 GM Muls+Cast；dK/dV 在 L0C 累加到换 s2 再写回。
    """
    coreNum = pl.get_block_num()
    # swizzle 来自 tilingkey。BN2 is_valid 保证为 0，与原先写死 0 等价。
    cu_q_tensor = pl.make_tensor(cu_seqlens_q, [tiling.b + 1], [1], dtype=pl.DT_INT32)
    cu_kv_tensor = pl.make_tensor(cu_seqlens_kv, [tiling.b + 1], [1], dtype=pl.DT_INT32)
    used_q_tensor = pl.make_tensor(seqused_q, [tiling.b], [1], dtype=pl.DT_INT32)
    used_kv_tensor = pl.make_tensor(seqused_kv, [tiling.b], [1], dtype=pl.DT_INT32)
    # 与 preload3 共用逻辑长度 / GM 视图长度的初始化约定。
    constInfo = set_const_info_bn2s2(tiling, coreNum, cu_q_tensor, cu_kv_tensor)

    dQ = constInfo.dSize
    viewD2Q = constInfo.viewD2Q
    d2dQ = viewD2Q * dQ
    s1ndQ = constInfo.s1ViewSize * d2dQ
    dV = constInfo.dvSize
    viewD0KV = constInfo.viewD0KV
    viewD2KV = constInfo.viewD2KV
    d2dKV = viewD2KV * dQ
    s2ndKV = constInfo.s2ViewSize * d2dKV
    d2dQv = viewD2Q * dV
    s1ndQv = constInfo.s1ViewSize * d2dQv
    d2dKVv = viewD2KV * dV
    s2ndKVv = constInfo.s2ViewSize * d2dKVv

    tensor_q = pl.make_tensor(
        q,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
        [s1ndQ, d2dQ, dQ, 1],
        dtype=input_dtype,
    )
    tensor_k = pl.make_tensor(
        k,
        [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=input_dtype,
    )
    # MM3 需要未转置的 K，必须与 MM2 的 K^T 视图分开(layout 各自推导)
    tensor_k_nt = pl.make_tensor(
        k,
        [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=input_dtype,
    )
    tensor_v = pl.make_tensor(
        v,
        [viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
        [s2ndKVv, d2dKVv, dV, 1],
        dtype=input_dtype,
    )
    tensor_dy = pl.make_tensor(
        dout,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=input_dtype,
    )
    tensor_y = pl.make_tensor(
        attn_out,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dV],
        [s1ndQv, d2dQv, dV, 1],
        dtype=input_dtype,
    )
    # BN2 无 GQA（g=1），n1==n2，与原先 [B,n2,S1] 等价。
    n1Size = constInfo.n2Size * constInfo.gSize
    if layout == 1:
        # TND：LSE 为 [N1,T1]，head stride 必须是实际 token 总数。
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [n1Size, constInfo.t1Size],
            [constInfo.t1Size, 1],
            dtype=pl.DT_FP32,
        )
    else:
        tensor_lse = pl.make_tensor(
            softmax_lse,
            [constInfo.bSize, n1Size, constInfo.s1Size],
            [n1Size * constInfo.s1Size, constInfo.s1Size, 1],
            dtype=pl.DT_FP32,
        )

    ws_ptr = pl.make_ptr(workspace, dtype=pl.DT_FP32)
    # 与 preload3 / host 共用物理 token 口径；BN2 也用它定位 dsink 槽位。
    wsRowsQ = constInfo.t1Size * constInfo.n1Size
    wsRowsKV = constInfo.t2Size * constInfo.n2Size
    pass
    dq_workspace = ws_ptr
    tensor_dq_ws = pl.make_tensor(
        dq_workspace,
        [constInfo.viewD0Q, constInfo.s1ViewSize, viewD2Q, dQ],
        [s1ndQ, d2dQ, dQ, 1],
        dtype=pl.DT_FP32,
    )
    dqElems = constInfo.t1Size * n1Size * dQ
    tensor_dq_ws_flat = pl.make_tensor(
        dq_workspace, [1, dqElems], [dqElems, 1], dtype=pl.DT_FP32
    )
    tensor_dq_out = pl.make_tensor(dq, [1, dqElems], [dqElems, 1], dtype=input_dtype)
    tensor_dk_out4d = pl.make_tensor(
        dk,
        [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
        [s2ndKV, d2dKV, dQ, 1],
        dtype=input_dtype,
    )
    tensor_dv_out4d = pl.make_tensor(
        dv,
        [viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
        [s2ndKVv, d2dKVv, dV, 1],
        dtype=input_dtype,
    )
    if layout == 1:
        # TND 专属：dk/dv ws 段视图 + flats（deter 分档的 flush/atomic
        # 目标、post 的 ws→输出 cast）+ 输出 flats。非确定场景不消费。
        dk_workspace = pl.addptr(dq_workspace, wsRowsQ * tiling.d)
        dv_workspace = pl.addptr(dk_workspace, wsRowsKV * tiling.d)
        tensor_dk_ws = pl.make_tensor(
            dk_workspace,
            [viewD0KV, constInfo.s2ViewSize, viewD2KV, dQ],
            [s2ndKV, d2dKV, dQ, 1],
            dtype=pl.DT_FP32,
        )
        tensor_dv_ws = pl.make_tensor(
            dv_workspace,
            [viewD0KV, constInfo.s2ViewSize, viewD2KV, dV],
            [s2ndKVv, d2dKVv, dV, 1],
            dtype=pl.DT_FP32,
        )
        dkElems = constInfo.t2Size * constInfo.n2Size * dQ
        dvElems = constInfo.t2Size * constInfo.n2Size * dV
        tensor_dk_ws_flat = pl.make_tensor(
            dk_workspace, [1, dkElems], [dkElems, 1], dtype=pl.DT_FP32
        )
        tensor_dv_ws_flat = pl.make_tensor(
            dv_workspace, [1, dvElems], [dvElems, 1], dtype=pl.DT_FP32
        )
        tensor_dk_out = pl.make_tensor(
            dk, [1, dkElems], [dkElems, 1], dtype=input_dtype
        )
        tensor_dv_out = pl.make_tensor(
            dv, [1, dvElems], [dvElems, 1], dtype=input_dtype
        )
    # v1w：BN2S2 各场景 dv_align ∈ {64,128,192,256,512,768}（512/768 仅
    # TND d>256 可达且 D==Dv），dense 门保证 D==Dv，故与原 dense 专用的
    # d_align 取值恒等。512/768 档 y/dy 按 256 列窗分配，V1 按窗多趟。
    v1w = 256 if dv_align >= 512 else dv_align

    if mask_mode != 0:
        tensor_mask = pl.make_tensor(
            attn_mask,
            [ATTEN_MASK_COMPRESS_bn2s2, ATTEN_MASK_COMPRESS_bn2s2],
            [ATTEN_MASK_COMPRESS_bn2s2, 1],
            dtype=pl.DT_UINT8,
        )
    else:
        tensor_mask = tensor_lse

    if template == Tmpl.BN2S2 and layout == 0:
        # 与 host BuildWorkspacePlan 的段布局逐字镜像（tmpl==BN2S2 && !isTnd →
        # 单段）。deter 不参与判断：dense+BN2S2+deter 不可达（路由 bn2S2NotTndLimit
        # 要求非确定 + is_valid 剪掉该组合键），故无需显式排除。
        dsinkOffset = wsRowsQ * tiling.d + 128 * pl.max(tiling.d, tiling.dv)
    else:
        dsinkOffset = (
            wsRowsQ * tiling.d
            + wsRowsKV * tiling.d
            + wsRowsKV * tiling.dv
            + 128 * pl.max(tiling.d, tiling.dv)
        )
    dsink_workspace = pl.addptr(ws_ptr, dsinkOffset)
    tensor_sinks = pl.make_tensor(sinks, [constInfo.n1Size], [1], dtype=pl.DT_FP32)
    dsinkElems = constInfo.aivCount * constInfo.n1Size
    tensor_dsink_ws = pl.make_tensor(
        dsink_workspace, [1, dsinkElems], [dsinkElems, 1], dtype=pl.DT_FP32
    )
    sfmgElems = constInfo.n1Size * constInfo.t1Size
    tensor_sfmg_ws = pl.make_tensor(
        pl.addptr(dsink_workspace, dsinkElems),
        [1, sfmgElems],
        [sfmgElems, 1],
        dtype=pl.DT_FP32,
    )
    tensor_dsink_out = pl.make_tensor(
        dsink, [1, constInfo.n1Size], [constInfo.n1Size, 1], dtype=pl.DT_FP32
    )

    # AIV get_block_idx 按 vector 子核计数，必须换算成物理核，否则
    # cube/vector 的 numBlocks 不一致，set/wait 配不上。
    cBlockIdx = pl.get_block_idx() // pl.get_subblock_num()
    coreNum = constInfo.coreNum
    meta = pl.make_tensor(metadata, [META_VIEW_LEN_bn2s2], [1], dtype=pl.DT_INT32)
    one64 = pl.const(1, pl.DT_INT64)
    fagOff = pl.getval(meta, META_HEAD_FAG_START_INDEX_bn2s2) * one64
    blockStart = (
        pl.getval(meta, fagOff + META_FAG_BLOCK_STARTS_OFFSET_bn2s2 + cBlockIdx) * one64
    )
    blockEnd = (
        pl.getval(meta, fagOff + META_FAG_BLOCK_ENDS_OFFSET_bn2s2 + cBlockIdx) * one64
    )
    numBlocks = block_range_of_bn2s2(
        constInfo, cBlockIdx, coreNum, blockStart, blockEnd
    )
    if (
        layout == 1
        and deter == 0
        and d_align == 192
        and (mask_mode == 0 or mask_mode == 3)
    ):
        if pl.getval(meta, fagOff + fag_schedule.META_FAG_FIXED_GS_MODE_bn2s2) == -1:
            constInfo.completeKvOwner = 1
    if (
        layout == 1
        and deter == 0
        and swizzle == 1
        and (
            (mask_mode == 0 and d_align == 256)
            or (mask_mode == 3 and d_align != 256)
            or (mask_mode == 4 and d_align != 192 and d_align != 256)
        )
    ):
        set_fixed_gs_schedule_bn2s2(constInfo, meta, fagOff)
        if constInfo.normalSwizzleEnabled == 1:
            numBlocks = block_range_of(
                constInfo, cBlockIdx, coreNum, blockStart, blockEnd
            )

    if layout == 1 and deter == 0 and d_align >= 256:
        sw = make_normal_swizzle_bn2s2(meta, constInfo)
        ownsColumn = pl.const(1, pl.DT_INT64)
        if sw.active == 1 and sw.kind != 0:
            ownsColumn = 0
            if sw.kind == 2 and sw.pre >= 0 and sw.nxt >= 0:
                ownsColumn = 1
                for batch in pl.range(0, constInfo.bSize):
                    qBlocks = (
                        normal_meta_i64_bn2s2(meta, sw.tail + 122 + 32 * batch) + 127
                    ) // 128
                    kBlocks = (
                        normal_meta_i64_bn2s2(meta, sw.tail + 124 + 32 * batch) + 127
                    ) // 128
                    if qBlocks != kBlocks or qBlocks == 0:
                        ownsColumn = 0
        if (
            pl.getval(meta, fagOff + META_FAG_FALLBACK_TO_GS_bn2s2) == 0
            and ownsColumn == 1
            and constInfo.gSize == 1
            and (dQ + 15) // 16 * 16 <= d_align
            and (dV + 15) // 16 * 16 <= d_align
            and wsRowsKV * dQ >= coreNum * 128 * d_align
            and wsRowsKV * dV >= coreNum * 128 * d_align
        ):
            constInfo.wideKvSlots = 1
            if d_align == 512 and dv_align == 512 and mask_mode == 3 and sink == 0:
                if (
                    sw.active == 0
                    and constInfo.enablePreSfmg == 1
                    and dQ > 256
                    and dQ <= 384
                    and dV > 256
                    and dV <= 384
                    and wsRowsKV * dQ >= 2 * coreNum * 128 * d_align
                    and wsRowsKV * dV >= 2 * coreNum * 128 * d_align
                ):
                    constInfo.wideKvDoubleBuffer = 1

    mm1_res = make_mm1_res(True)
    mm2_res = make_mm2_res(True)
    ds_l1 = make_ds_l1_bn2s2()
    p_l1 = make_p_l1_bn2s2()
    ds_t_l1 = make_ds_t_l1_bn2s2()
    p_t_l1 = make_p_t_l1_bn2s2()

    pass
    if layout == 1:
        # Wide D and the internal GS fallback accumulate K/V in FP32 workspace.
        # Their post covers every physical output row, including empty batches.
        with pl.section_vector():
            zero_ub = make_pre_zero_buffer_f32(True)
            if d_align <= 128:
                clearKvWs = pl.getval(meta, fagOff + META_FAG_FALLBACK_TO_GS_bn2s2)
            elif deter == 0 and d_align >= 256:
                clearKvWs = 1 - constInfo.wideKvSlots
            elif deter == 0 and d_align == 192 and (mask_mode == 0 or mask_mode == 3):
                clearKvWs = 1 - constInfo.completeKvOwner
            else:
                clearKvWs = 1
            isolatedDense = False
            if (
                template == Tmpl.BN2S2
                and deter == 0
                and d_align == 128
                and mask_mode == 0
            ):
                isolatedDense = (
                    clearKvWs == 0
                    and constInfo.sequsedQOptional + constInfo.sequsedKvOptional == 0
                    and dQ % 16 == 0
                    and dV % 16 == 0
                )
            if isolatedDense:
                dense_zero16 = make_pre_zero_buffer_f16(True)
                pre_dense_tnd_bn2s2(
                    constInfo,
                    cu_q_tensor,
                    cu_kv_tensor,
                    meta,
                    fagOff,
                    tensor_dq_ws_flat,
                    tensor_dq_out,
                    tensor_dk_out,
                    tensor_dv_out,
                    zero_ub.current(),
                    dense_zero16.current(),
                )
            else:
                process_pre_from_meta_bn2s2(
                    meta,
                    fagOff,
                    tensor_dq_ws_flat,
                    tensor_dk_ws_flat,
                    tensor_dv_ws_flat,
                    dQ,
                    dV,
                    zero_ub.current(),
                    clearKvWs,
                )
            if d_align <= 128 or (deter == 0 and d_align >= 256):
                zero16_ub = make_pre_zero_buffer_f16(True)
                if clearKvWs == 0 and isolatedDense == 0:
                    # Non-sparse direct stores cover all K/V columns of a
                    # nonempty query batch when no seqused truncates it.
                    if (
                        mask_mode == 0
                        and constInfo.sequsedQOptional + constInfo.sequsedKvOptional
                        == 0
                    ):
                        zero_tnd_empty_batch(
                            constInfo,
                            cu_q_tensor,
                            cu_kv_tensor,
                            tensor_dq_out,
                            tensor_dk_out,
                            tensor_dv_out,
                            zero16_ub.current(),
                            False,
                        )
                    else:
                        covered = pl.const(0, pl.DT_INT64)
                        if (
                            deter == 0
                            and mask_mode == 3
                            and d_align == 512
                            and dv_align == 512
                        ):
                            covered = normal_causal_kv_covered_bn2s2(
                                constInfo, cu_q_tensor, cu_kv_tensor, sw.active
                            )
                        if covered == 0:
                            process_pre_kv_out_zero_bn2s2(
                                meta,
                                fagOff,
                                tensor_dk_out,
                                tensor_dv_out,
                                dQ,
                                dV,
                                zero16_ub.current(),
                                1,
                            )
            if sink == 1:
                process_sink_zero_bn2s2(
                    tensor_dsink_ws,
                    constInfo,
                    zero_ub.current(),
                    pl.get_block_idx(),
                )
    else:
        # dense（仅非确定可达）：host 只有 dq 段，仅清 dq ws。dk/dv 直出档
        # 补一条按需清零：零 Q 任务（s1==0，metadata fagS1Outer_==0 早退
        # 回退 GS 零块）与 seqused 有效长度外的列不会被主循环写，16-bit
        # 输出必须 pre 兜底；s1>0 且无 seqused 时列必写（非稀疏 + 饱和），
        # need=0 把清零行程折叠为空，保住主路径整份 kv 输出的写带宽。
        with pl.section_vector():
            zero_ub = make_pre_zero_buffer_f32(True)
            zt = zero_ub.current()
            dqElems = constInfo.t1Size * constInfo.n1Size * dQ
            aivIdx = pl.get_block_idx()
            aivCount = constInfo.aivCount
            share = (dqElems + aivCount - 1) // aivCount
            elemStart = aivIdx * share
            nElems = pl.max(pl.min(share, dqElems - elemStart), 0)
            preElems = pl.min(PRE_ELEMS_bn2s2[d_align], pl.max(nElems, 1))
            pl.set_validshape(zt, [1, PRE_ELEMS_bn2s2[d_align]])
            fill_zero_vf(zt, (preElems + VF_LANES_bn2s2 - 1) // VF_LANES_bn2s2)
            process_pre_bn2s2(tensor_dq_ws_flat, zt, elemStart, nElems)
            # 需要 dk/dv 输出清零兜底的两类输入：零 Q 任务、seqused 变长。
            s1Empty = 1 - pl.min(pl.max(constInfo.s1Size, 0), 1)
            hasUsed = pl.min(
                constInfo.sequsedQOptional + constInfo.sequsedKvOptional, 1
            )
            needKvZero = pl.min(s1Empty + hasUsed, 1)
            dkElems = constInfo.t2Size * constInfo.n2Size * dQ
            dvElems = constInfo.t2Size * constInfo.n2Size * dV
            tensor_dk_flat = pl.make_tensor(
                dk, [1, dkElems], [dkElems, 1], dtype=input_dtype
            )
            tensor_dv_flat = pl.make_tensor(
                dv, [1, dvElems], [dvElems, 1], dtype=input_dtype
            )
            zero16_ub = make_pre_zero_buffer_f16(True)
            process_pre_kv_out_zero_bn2s2(
                meta,
                fagOff,
                tensor_dk_flat,
                tensor_dv_flat,
                dQ,
                dV,
                zero16_ub.current(),
                needKvZero,
            )
            if sink == 1:
                process_sink_zero_bn2s2(tensor_dsink_ws, constInfo, zt, aivIdx)

    if dv_align >= 128 and sink == 0:
        with pl.section_vector():
            if constInfo.enablePreSfmg == 1:
                # The previous zero-fill uses this AIV's full UB. Drain its stores
                # before reusing that space for the double-buffered row producer.
                pl.system.bar_all()
                sp_y = make_sp_y_ub(v1w)
                sp_dy = make_sp_dy_ub(v1w)
                sp_out = make_sp_out_ub(v1w)
                sp_flat = make_sp_out_flat(v1w)
                if (
                    layout == 1
                    and deter == 0
                    and mask_mode == 3
                    and d_align == 512
                    and dv_align == 512
                ):
                    normal_wide_tnd_presfmg_bn2s2(
                        constInfo,
                        tensor_y,
                        tensor_dy,
                        tensor_sfmg_ws,
                        sp_y,
                        sp_dy,
                        sp_out,
                        sp_flat,
                        fag_common.SP_WORK_W[v1w],
                        dsinkOffset + dsinkElems,
                    )
                else:
                    zero = make_pre_zero_buffer_f32()
                    process_presfmg(
                        constInfo,
                        tensor_y,
                        tensor_dy,
                        tensor_sfmg_ws,
                        sp_y,
                        sp_dy,
                        sp_out,
                        sp_flat,
                        fag_common.SP_WORK_W[v1w],
                        zero.current(),
                        tensor_dq_ws_flat,
                        tensor_dq_ws_flat,
                        tensor_dq_ws_flat,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    )
    # All cores publish zero fills and completed per-row scalars before Vec1.
    pl.system.sync_all()

    with pl.section_cube():
        q_l1 = make_q_l1(True)
        k_l1 = make_k_l1_bn2s2()
        k_l1_n = make_k_nd_l1_bn2s2()
        v_l1 = make_v_l1(True)
        do_l1 = make_do_l1(True)
        left = make_l0a(True)
        right = make_l0b_bn2s2()
        acc = make_l0c_acc_bn2s2()
        dk_acc = make_dk_l0c_chunk0_bn2s2()
        dv_acc = make_bn2s2_norm_dv_l0c_chunk0_bn2s2()
        # 192/256 档真两块（192=dk/dv 各 96 宽、256=dv 双 128 宽而 dk
        # 哑元别名 dv）；64/128 档 chunk1 同址别名 chunk0，未用即剪。
        dk_acc1 = make_dk_l0c_chunk1_bn2s2()
        dv_acc1 = make_bn2s2_norm_dv_l0c_chunk1_bn2s2()
        runInfos = make_normal_run_infos_bn2s2()
        sched = pl.make_tuple(
            meta=meta,
            cBlockIdx=cBlockIdx,
            coreNum=coreNum,
            cuQ=cu_q_tensor,
            cuKv=cu_kv_tensor,
            usedQ=used_q_tensor,
            usedKv=used_kv_tensor,
            blockStart=blockStart,
            blockEnd=blockEnd,
        )
        gm = pl.make_tuple(
            tensor_q=tensor_q,
            tensor_k=tensor_k,
            tensor_k_nt=tensor_k_nt,
            tensor_dy=tensor_dy,
            tensor_v=tensor_v,
            tensor_dq_ws=tensor_dq_ws,
            tensor_dv_out=tensor_dv_out4d,
            dk_slot_base=dk_workspace if layout == 1 else dq_workspace,
            dv_slot_base=dv_workspace if layout == 1 else dq_workspace,
            # 192/256 档（恒 TND）消费；其余档位哑元占位、不读。
            tensor_dk_ws=tensor_dk_ws if d_align > 128 else tensor_dv_out4d,
            tensor_dv_ws=tensor_dv_ws if d_align > 128 else tensor_dv_out4d,
        )
        buf = pl.make_tuple(
            mm1_res=mm1_res,
            mm2_res=mm2_res,
            q_l1=q_l1,
            k_l1=k_l1,
            k_l1_n=k_l1_n,
            v_l1=v_l1,
            do_l1=do_l1,
            left=left,
            right=right,
            acc=acc,
            dk_acc=dk_acc,
            dv_acc=dv_acc,
            dk_acc1=dk_acc1,
            dv_acc1=dv_acc1,
            ds_l1=ds_l1,
            ds_t_l1=ds_t_l1,
            p_t_l1=p_t_l1,
        )
        if layout == 1:
            # TND：metadata 运行期回退标记（split.h DoFagSparse——老仓
            # DoSparse 饱和回退的迁移，host 读不到 cu_seqlens 判不了）。
            # 标记置位时 metadata 已按 GS1S2 口径重切块，此处切 GS 非确定
            # 主循环：区间解码随 GS 线性化，dK/dV 走 ws 累加（pre 已清三
            # 段、post 由 doKvPost 补全）。dense 不走此路：host step4 同式
            # 判饱和，不饱和 shape 本就路由 GS1S2，且 dense 无 dk/dv ws
            # 视图（引用会在 trace 期 NameError）。
            fbGs = pl.getval(meta, fagOff + META_FAG_FALLBACK_TO_GS_bn2s2)
            if fbGs == FAG_FALLBACK_TO_GS_bn2s2:
                gsSched = pl.make_tuple(
                    meta=meta,
                    fagOff=fagOff,
                    cBlockIdx=cBlockIdx,
                    coreNum=coreNum,
                    cuQ=cu_q_tensor,
                    cuKv=cu_kv_tensor,
                    usedQ=used_q_tensor,
                    usedKv=used_kv_tensor,
                    blockStart=blockStart,
                    blockEnd=blockEnd,
                    numBlocks=numBlocks,
                )
                gsGm = pl.make_tuple(
                    tensor_q=tensor_q,
                    tensor_k=tensor_k,
                    tensor_k_nt=tensor_k_nt,
                    tensor_dy=tensor_dy,
                    tensor_v=tensor_v,
                    tensor_dq_ws=tensor_dq_ws,
                    tensor_dk_ws=tensor_dk_ws,
                    tensor_dv_ws=tensor_dv_ws,
                )
                _run_preload2_cube_gs_normal_bn2s2(
                    constInfo, gsSched, gsGm, buf, runInfos
                )
            else:
                _run_preload2_cube_bn2s2(constInfo, sched, gm, buf, runInfos)
        else:
            _run_preload2_cube_bn2s2(constInfo, sched, gm, buf, runInfos)

    with pl.section_vector():
        subIdx = pl.get_subblock_idx()
        y_ub = make_y_ub_bn2s2(v1w)
        dy_ub = make_dy_ub_bn2s2(v1w)
        nzp_ub = make_normal_nzp_bn2s2(v1w)
        p_nzp_ub = make_normal_p_nzp_bn2s2(v1w)
        sfmg_ub = make_sfmg_ub_bn2s2(v1w)
        lse_ub = make_lse_ub_bn2s2(v1w)
        sink_part_ub = make_sink_part_ub_bn2s2(v1w)
        mask_ub = make_mask_view_bn2s2(v1w)
        mask_pre_ub = make_mask_pre_view_bn2s2(v1w)
        cast_ub = make_cast_ub_bn2s2()
        runInfos = make_normal_run_infos_bn2s2()
        sched = pl.make_tuple(
            meta=meta,
            cBlockIdx=cBlockIdx,
            coreNum=coreNum,
            cuQ=cu_q_tensor,
            cuKv=cu_kv_tensor,
            usedQ=used_q_tensor,
            usedKv=used_kv_tensor,
            blockStart=blockStart,
            blockEnd=blockEnd,
            subIdx=subIdx,
            v1w=v1w,
        )
        gm = pl.make_tuple(
            tensor_y=tensor_y,
            tensor_dy=tensor_dy,
            tensor_sfmg_ws=tensor_sfmg_ws,
            tensor_lse=tensor_lse,
            dk_slot_base=dk_workspace if layout == 1 else dq_workspace,
            dv_slot_base=dv_workspace if layout == 1 else dq_workspace,
            tensor_dv_out=tensor_dv_out4d,
            tensor_mask=tensor_mask,
            tensor_dk_out=tensor_dk_out4d,
            tensor_sinks=tensor_sinks,
            tensor_dsink_ws=tensor_dsink_ws,
        )
        buf = pl.make_tuple(
            mm1_res=mm1_res,
            mm2_res=mm2_res,
            y_ub=y_ub,
            dy_ub=dy_ub,
            nzp_ub=nzp_ub,
            p_nzp_ub=p_nzp_ub,
            sfmg_ub=sfmg_ub,
            lse_ub=lse_ub,
            sink_part_ub=sink_part_ub,
            mask_ub=mask_ub,
            mask_pre_ub=mask_pre_ub,
            cast_ub=cast_ub,
            ds_l1=ds_l1,
            p_l1=p_l1,
        )
        if layout == 1:
            # TND：同 cube 侧的运行期回退（META_FAG_FALLBACK_TO_GS==1 时
            # metadata 已按 GS 口径重切，vector 切 GS 主循环；UB 缓冲与
            # BN2S2 分支同款 v1w 构造，直接复用，buf 为超集）。
            fbGsV = pl.getval(meta, fagOff + META_FAG_FALLBACK_TO_GS_bn2s2)
            if fbGsV == FAG_FALLBACK_TO_GS_bn2s2:
                gsSchedV = pl.make_tuple(
                    meta=meta,
                    fagOff=fagOff,
                    cBlockIdx=cBlockIdx,
                    coreNum=coreNum,
                    cuQ=cu_q_tensor,
                    cuKv=cu_kv_tensor,
                    usedQ=used_q_tensor,
                    usedKv=used_kv_tensor,
                    blockStart=blockStart,
                    blockEnd=blockEnd,
                    numBlocks=numBlocks,
                    subIdx=subIdx,
                    v1w=v1w,
                )
                gsGmV = pl.make_tuple(
                    tensor_y=tensor_y,
                    tensor_dy=tensor_dy,
                    tensor_sfmg_ws=tensor_sfmg_ws,
                    tensor_lse=tensor_lse,
                    tensor_mask=tensor_mask,
                    tensor_sinks=tensor_sinks,
                    tensor_dsink_ws=tensor_dsink_ws,
                )
                _run_preload2_vector_gs_normal_bn2s2(
                    constInfo, gsSchedV, gsGmV, buf, runInfos, sink
                )
            else:
                _run_preload2_vector_bn2s2(constInfo, sched, gm, buf, runInfos, sink)
        else:
            _run_preload2_vector_bn2s2(constInfo, sched, gm, buf, runInfos, sink)

    """ BN2S2 与 GS1S2 共用：meta 行区间 dq 收尾 + 条件 dk/dv + sink。
        dq 分区统一走 metadata 行区间（AICPU 无条件写入：
        dense 按 b*s1*n1 行、TND 按物理 token*n1 行，FagSplitRows 均分），
        BN2S2 非确定原先的 flat 自分区取消——两者覆盖同一元素集合、饱和时
        worker 数相同，仅切分粒度（行 vs 元素）不同；TND 非确定的 pre 段
        已在用同一套行区间，改后 pre/post 工人模型对齐。"""
    pl.system.sync_all()
    with pl.section_vector():
        post_f32 = make_post_f32_buffer(True)
        post_f16 = make_post_f16_buffer(True)
        vBlockIdxP = pl.get_block_idx()
        dqRowStartP = (
            pl.getval(meta, fagOff + META_FAG_DQ_ROW_STARTS_bn2s2 + vBlockIdxP) * one64
        )
        dqRowEndP = (
            pl.getval(meta, fagOff + META_FAG_DQ_ROW_ENDS_bn2s2 + vBlockIdxP) * one64
        )
        if (
            layout == 1
            and template == Tmpl.BN2S2
            and deter == 0
            and d_align == 128
            and mask_mode == 0
        ):
            if isolatedDense == 1:
                process_dense_dq_rows_bn2s2(
                    constInfo,
                    cu_q_tensor,
                    cu_kv_tensor,
                    tensor_dq_ws_flat,
                    tensor_dq_out,
                    dqRowStartP,
                    dqRowEndP,
                    post_f32,
                    post_f32,
                    post_f16,
                    True,
                )
            else:
                process_post(
                    tensor_dq_ws_flat,
                    tensor_dq_out,
                    post_f32,
                    post_f16,
                    constInfo.scaleValue,
                    dqRowStartP * dQ,
                    (dqRowEndP - dqRowStartP) * dQ,
                )
        else:
            process_post(
                tensor_dq_ws_flat,
                tensor_dq_out,
                post_f32,
                post_f16,
                constInfo.scaleValue,
                dqRowStartP * dQ,
                (dqRowEndP - dqRowStartP) * dQ,
            )
        if template == Tmpl.BN2S2 and layout == 1 and deter == 0 and d_align >= 256:
            doKvPost = constInfo.wideKvSlots == 0
        elif template != Tmpl.BN2S2 or d_align > 128:
            doKvPost = True
        elif layout == 1 and deter == 0:
            """
               BN2S2 直出档（d<=128）TND 非确定：正常路径主循环列末直出，
               post 只收 dq；运行期回退（FAG_FALLBACK_TO_GS==1）时主循环
               已切 GS 流程、dK/dV 改 ws 累加，post 补全三段。
            """
            doKvPost = (
                pl.getval(meta, fagOff + fag_schedule.META_FAG_FALLBACK_TO_GS_bn2s2)
                == fag_schedule.FAG_FALLBACK_TO_GS_bn2s2
            )
        else:
            doKvPost = False
        if doKvPost:
            """
               doKvPost=False 仅剩两种命中：BN2S2 直出档（d<=128）的
               正常路径（deter 与非确定同判，对齐老算子 SPLIT_AXIS==5
               的 break）与 BN2S2+d<=128 的 dense（非 TND 不触发 AICPU
               回退置位，直出恒成立）。其余（GS、BN2S2 的 192/256/512/
               768 档、TND 运行期回退）一律走 ws + post 全三段——对齐老仓
               DqkvMulsAndCastFromGM 的时机（sync_all 全局屏障后收尾，
               GM 写穿透有保证；列末事件粒度的跨核 GM 读已实测存在可见
               性竞态，撤回）。dK/dV 的写入端（192 驻留+flush / 256 dV
               单驻留 / 512+768 逐 task atomic / 回退的驻留三件套）保持
               不变，均为 post 消费服务。
               d>128 档的判定是 trace 期常量：dk/dv 行区间只在命中的二进
               制里被 trace。d<=128+TND+非确定档为运行期双路径。
            """
            dkvRowStartP = (
                pl.getval(meta, fagOff + META_FAG_DKV_ROW_STARTS_bn2s2 + vBlockIdxP)
                * one64
            )
            dkvRowEndP = (
                pl.getval(meta, fagOff + META_FAG_DKV_ROW_ENDS_bn2s2 + vBlockIdxP)
                * one64
            )
            process_post(
                tensor_dk_ws_flat,
                tensor_dk_out,
                post_f32,
                post_f16,
                constInfo.scaleValue,
                dkvRowStartP * dQ,
                (dkvRowEndP - dkvRowStartP) * dQ,
            )
            if (
                d_align == 192
                and layout == 1
                and deter == 0
                and (mask_mode == 0 or mask_mode == 3)
            ):
                doDvPost = constInfo.completeKvOwner == 0
            else:
                doDvPost = True
            if doDvPost:
                process_post(
                    tensor_dv_ws_flat,
                    tensor_dv_out,
                    post_f32,
                    post_f16,
                    1.0,
                    dkvRowStartP * dV,
                    (dkvRowEndP - dkvRowStartP) * dV,
                )
        if sink == 1:
            if pl.get_block_idx() == 0:
                sink_red_ub = make_sink_red_post_ub(v1w, True)
                sink_row_ub = make_sink_row_post_ub(v1w, True)
                process_sink_reduce_bn2s2(
                    constInfo,
                    tensor_dsink_ws,
                    tensor_dsink_out,
                    sink_red_ub.current(),
                    sink_row_ub,
                    constInfo.aivCount,
                )


def process_deter_plan_bn2s2(
    q: pl.Ptr[pl.DT_UINT8],
    k: pl.Ptr[pl.DT_UINT8],
    v: pl.Ptr[pl.DT_UINT8],
    dy: pl.Ptr[pl.DT_UINT8],
    dk: pl.Ptr[pl.DT_UINT8],
    dv: pl.Ptr[pl.DT_UINT8],
    cuq: pl.Ptr[pl.DT_INT32],
    cuk: pl.Ptr[pl.DT_INT32],
    metadata: pl.Ptr[pl.DT_INT32],
    workspace: pl.Ptr[pl.DT_FP32],
    y: pl.Ptr[pl.DT_UINT8],
    lse: pl.Ptr[pl.DT_FP32],
    sinks: pl.Ptr[pl.DT_FP32],
    mask: pl.Ptr[pl.DT_UINT8],
    scale: pl.DT_FP32,
    dq: pl.Ptr[pl.DT_UINT8],
    dsink: pl.Ptr[pl.DT_FP32],
    expected,
    capacity,
    plan_bytes,
):
    if plan_bytes >= (16 + 412 + PM_bn2s2.I32_WORDS) * 4:
        words = metadata_plan_views_bn2s2(
            metadata, plan_bytes, expected.maskPresent, sink
        )
        valid = plan_header_matches_bn2s2(words, pl.get_block_num(), capacity)
        if valid:
            h = plan_execution_header_bn2s2(words)
            ci = pl.struct(
                "FagPlanConstInfo",
                dSize=h.dimQ,
                dvSize=h.dimV,
                s1Size=h.maxQ,
                s2Size=h.maxKv,
                s1ViewSize=h.totalQ,
                dvEvenSize=(h.dimV + 1) // 2,
                dvOddSize=h.dimV // 2,
                scaleValue=scale,
            )
            cu_q = pl.make_tensor(cuq, [h.batch + 1], [1], dtype=pl.DT_INT32)
            cu_k = pl.make_tensor(cuk, [h.batch + 1], [1], dtype=pl.DT_INT32)
            if template == Tmpl.BN2S2 and d_align == 192 and dv_align == 192:
                plan_admit_asc_pair_swizzle_bn2s2(h, cu_q, cu_k)
            if template == Tmpl.BN2S2 and (
                (d_align == 192 and mask_mode == 0)
                or (d_align == 768 and mask_mode == 3)
            ):
                reuse_dense = (
                    mask_mode == 0
                    and dv_align == 192
                    and sink != 0
                    and input_dtype == pl.DT_BF16
                )
                plan_admit_fixed_rectangle_bn2s2(h, cu_q, cu_k, mask_mode, reuse_dense)
            total_kv = plan_i64_bn2s2(words, PF_bn2s2.totalKv)
            tq = pl.make_tensor(
                q,
                [1, h.totalQ, h.heads, h.dimQ],
                [h.totalQ * h.heads * h.dimQ, h.heads * h.dimQ, h.dimQ, 1],
                dtype=input_dtype,
            )
            tk = pl.make_tensor(
                k,
                [1, total_kv, h.heads, h.dimQ],
                [total_kv * h.heads * h.dimQ, h.heads * h.dimQ, h.dimQ, 1],
                dtype=input_dtype,
            )
            tv = pl.make_tensor(
                v,
                [1, total_kv, h.heads, h.dimV],
                [total_kv * h.heads * h.dimV, h.heads * h.dimV, h.dimV, 1],
                dtype=input_dtype,
            )
            tdy = pl.make_tensor(
                dy,
                [1, h.totalQ, h.heads, h.dimV],
                [h.totalQ * h.heads * h.dimV, h.heads * h.dimV, h.dimV, 1],
                dtype=input_dtype,
            )
            tdk = pl.make_tensor(
                dk,
                [1, total_kv, h.heads, h.dimQ],
                [total_kv * h.heads * h.dimQ, h.heads * h.dimQ, h.dimQ, 1],
                dtype=input_dtype,
            )
            tdv = pl.make_tensor(
                dv,
                [1, total_kv, h.heads, h.dimV],
                [total_kv * h.heads * h.dimV, h.heads * h.dimV, h.dimV, 1],
                dtype=input_dtype,
            )
            tdq = pl.make_tensor(
                pl.addptr(workspace, plan_i64_bn2s2(words, PF_bn2s2.dqOffset) // 4),
                [1, h.totalQ, h.heads, h.dimQ],
                [h.totalQ * h.heads * h.dimQ, h.heads * h.dimQ, h.dimQ, 1],
                dtype=pl.DT_FP32,
            )
            q_count = h.totalQ * h.heads * h.dimQ
            k_count = total_kv * h.heads * h.dimQ
            v_count = total_kv * h.heads * h.dimV
            q_ws_flat = pl.make_tensor(
                pl.addptr(workspace, plan_i64_bn2s2(words, PF_bn2s2.dqOffset) // 4),
                [1, q_count],
                [q_count, 1],
                dtype=pl.DT_FP32,
            )
            k_ws_flat = pl.make_tensor(
                pl.addptr(workspace, plan_i64_bn2s2(words, PF_bn2s2.dkOffset) // 4),
                [1, k_count],
                [k_count, 1],
                dtype=pl.DT_FP32,
            )
            v_ws_flat = pl.make_tensor(
                pl.addptr(workspace, plan_i64_bn2s2(words, PF_bn2s2.dvOffset) // 4),
                [1, v_count],
                [v_count, 1],
                dtype=pl.DT_FP32,
            )
            q_out_flat = pl.make_tensor(
                dq, [1, q_count], [q_count, 1], dtype=input_dtype
            )
            k_out_flat = pl.make_tensor(
                dk, [1, k_count], [k_count, 1], dtype=input_dtype
            )
            v_out_flat = pl.make_tensor(
                dv, [1, v_count], [v_count, 1], dtype=input_dtype
            )
            sink_elements = h.heads * plan_i64_bn2s2(words, PF_bn2s2.sinkReduceAxis)
            sink_padded = (sink_elements + 512) // 512 * 512
            sink_pre = pl.make_tensor(
                pl.addptr(workspace, plan_i64_bn2s2(words, PF_bn2s2.sinkOffset) // 4),
                [1, sink_padded],
                [sink_padded, 1],
                dtype=pl.DT_FP32,
            )
            special_elements = plan_i64_bn2s2(words, PF_bn2s2.specialBytes) // 4
            special = pl.make_tensor(
                pl.addptr(
                    workspace, plan_i64_bn2s2(words, PF_bn2s2.specialOffset) // 4
                ),
                [1, special_elements],
                [special_elements, 1],
                dtype=pl.DT_FP32,
            )
            with pl.section_vector():
                pre_aiv = pl.get_block_idx()
                q_prefix = pl.const(0, pl.DT_INT64)
                k_prefix = pl.const(0, pl.DT_INT64)
                if sink != 0 and (d_align == 192 or d_align == 256):
                    q_prefix, k_prefix = plan_isolated_prefix_bn2s2(h, cu_q, cu_k)
                pre_zero = make_plan_zero_bn2s2(pl.DT_FP32)
                pre_zero16 = make_plan_zero_bn2s2(input_dtype)
                pre_gm = pl.make_tuple(
                    qPrefix=q_prefix * h.dimQ,
                    kPrefix=k_prefix * h.dimQ,
                    vPrefix=k_prefix * h.dimV,
                    dqWs=q_ws_flat,
                    dkWs=k_ws_flat,
                    dvWs=v_ws_flat,
                    dkOut=k_out_flat,
                    dvOut=v_out_flat,
                    sinkPre=sink_pre,
                    special=special,
                )
                if pre_sfmg != 0:
                    pre_y = pl.make_tensor(
                        y,
                        [1, h.totalQ, h.heads, h.dimV],
                        [h.totalQ * h.heads * h.dimV, h.heads * h.dimV, h.dimV, 1],
                        dtype=input_dtype,
                    )
                    pre_cache = pl.make_tensor(
                        pl.addptr(
                            workspace, plan_i64_bn2s2(words, PF_bn2s2.sfmgOffset) // 4
                        ),
                        [1, plan_i64_bn2s2(words, PF_bn2s2.sfmgRows)],
                        [plan_i64_bn2s2(words, PF_bn2s2.sfmgRows), 1],
                        dtype=pl.DT_FP32,
                    )
                    pre_y_ub = make_plan_presfmg_y_bn2s2()
                    pre_dy_ub = make_plan_presfmg_dy_bn2s2()
                    pre_sfmg_ub = make_plan_presfmg_out_bn2s2()
                    pre_sfmg_flat = make_plan_presfmg_flat_bn2s2()
                    if d_align == 768 and dv_align == 768 and sink == 0:
                        if (
                            h.logicalSplit == PlanSplit_bn2s2.GS
                            and ci.dvSize > 256
                            and ci.dvSize <= 320
                        ):
                            plan_presfmg_gs_interleaved_bn2s2(
                                ci,
                                words,
                                h,
                                pre_aiv,
                                pre_gm,
                                pre_zero.current(),
                                pre_y,
                                tdy,
                                pre_cache,
                                pre_y_ub,
                                pre_dy_ub,
                                pre_sfmg_ub,
                                pre_sfmg_flat,
                            )
                        else:
                            plan_clear_pre_bn2s2(
                                words,
                                h,
                                pre_aiv,
                                pre_gm,
                                pre_zero.current(),
                                pre_zero16.current(),
                            )
                            plan_compute_presfmg_bn2s2(
                                ci,
                                words,
                                h,
                                cu_q,
                                pre_aiv,
                                pre_y,
                                tdy,
                                pre_cache,
                                pre_y_ub,
                                pre_dy_ub,
                                pre_sfmg_ub,
                                pre_sfmg_flat,
                                min(dv_align, 256),
                            )
                    else:
                        plan_clear_pre_bn2s2(
                            words,
                            h,
                            pre_aiv,
                            pre_gm,
                            pre_zero.current(),
                            pre_zero16.current(),
                        )
                        plan_compute_presfmg_bn2s2(
                            ci,
                            words,
                            h,
                            cu_q,
                            pre_aiv,
                            pre_y,
                            tdy,
                            pre_cache,
                            pre_y_ub,
                            pre_dy_ub,
                            pre_sfmg_ub,
                            pre_sfmg_flat,
                            min(dv_align, 256),
                        )
                else:
                    plan_clear_pre_bn2s2(
                        words,
                        h,
                        pre_aiv,
                        pre_gm,
                        pre_zero.current(),
                        pre_zero16.current(),
                    )
            pl.system.sync_all()
            mm1 = make_mm1_res(True)
            mm2 = make_mm2_res(True)
            ds = make_ds_l1_bn2s2()
            ds_t = make_ds_t_l1_bn2s2()
            p_t = make_p_t_l1_bn2s2()
            p_l1 = make_p_l1_bn2s2()
            with pl.section_cube():
                core = pl.get_block_idx()
                policy = plan_commit_header_bn2s2(words, h, core)
                slots = plan_run_infos_bn2s2()
                left = make_l0a(True)
                if d_align == 768 or (d_align == 192 and dv_align == 128):
                    left_read = make_wide_l0a_reader_bn2s2()
                else:
                    left_read = left
                right = make_l0b_bn2s2()
                acc = make_plan_acc_bn2s2()
                if d_align <= 128 or (d_align == 192 and dv_align <= 128):
                    dk0 = make_plan_dk0_bn2s2()
                    if d_align == 192:
                        dk1 = dk0
                    else:
                        dk1 = make_plan_dk1_bn2s2()
                else:
                    dk0 = acc
                    dk1 = acc
                if d_align <= 256:
                    dv0 = make_plan_dv0_bn2s2()
                    if d_align == 192:
                        dv1 = dv0
                    else:
                        dv1 = make_plan_dv1_bn2s2()
                else:
                    dv0 = acc
                    dv1 = acc
                if d_align == 768:
                    q_l1 = make_plan_common_l1_bn2s2()
                    k_l1 = make_plan_common_zn_l1_bn2s2()
                    k_normal = make_k_nd_l1_bn2s2()
                else:
                    q_l1 = make_q_l1(True)
                    if d_align == 192:
                        k_l1 = make_plan_k_l1_bn2s2()
                        k_normal = make_plan_k_normal_l1_bn2s2()
                    else:
                        k_l1 = make_k_l1_bn2s2()
                        k_normal = make_k_nd_l1_bn2s2()
                dy_l1 = make_do_l1(True)
                v_l1 = make_v_l1(True)
                q_lag = make_q_lag_view(True)
                dy_lag = make_do_lag_view(True)
                mm2_lag = make_mm2_lag_view_bn2s2()
                gm = pl.make_tuple(
                    q=tq,
                    k=tk,
                    v=tv,
                    dy=tdy,
                    dk=tdk,
                    dv=tdv,
                    dqWorkspace=tdq,
                    workspace=workspace,
                )
                buf = pl.make_tuple(
                    q=q_l1,
                    dy=dy_l1,
                    k=k_l1,
                    v=v_l1,
                    kNormal=k_normal,
                    qLag=q_lag,
                    dyLag=dy_lag,
                    mm1=mm1,
                    mm2=mm2,
                    mm2Lag=mm2_lag,
                    ds=ds,
                    dsT=ds_t,
                    pT=p_t,
                    left=left,
                    leftRead=left_read,
                    right=right,
                    acc=acc,
                    dk0=dk0,
                    dk1=dk1,
                    dv0=dv0,
                    dv1=dv1,
                )
                run_plan_cube_bn2s2(
                    ci, words, h, cu_q, cu_k, core, policy, gm, buf, slots
                )
            with pl.section_vector():
                core_v = pl.get_block_idx() // pl.get_subblock_num()
                sub = pl.get_subblock_idx()
                v1w = min(dv_align, 256)
                policy_v = plan_commit_header_bn2s2(words, h, core_v)
                slots_v = plan_run_infos_bn2s2()
                ty = pl.make_tensor(
                    y,
                    [1, h.totalQ, h.heads, h.dimV],
                    [h.totalQ * h.heads * h.dimV, h.heads * h.dimV, h.dimV, 1],
                    dtype=input_dtype,
                )
                tlse = pl.make_tensor(
                    lse, [h.heads, h.totalQ], [h.totalQ, 1], dtype=pl.DT_FP32
                )
                tsinks = pl.make_tensor(sinks, [h.heads], [1], dtype=pl.DT_FP32)
                tmask = pl.make_tensor(mask, [2048, 2048], [2048, 1], dtype=pl.DT_UINT8)
                sfmg = pl.make_tensor(
                    pl.addptr(
                        workspace, plan_i64_bn2s2(words, PF_bn2s2.sfmgOffset) // 4
                    ),
                    [1, plan_i64_bn2s2(words, PF_bn2s2.sfmgRows)],
                    [plan_i64_bn2s2(words, PF_bn2s2.sfmgRows), 1],
                    dtype=pl.DT_FP32,
                )
                sink_count = h.heads * plan_i64_bn2s2(words, PF_bn2s2.sinkReduceAxis)
                partial = pl.make_tensor(
                    pl.addptr(
                        workspace, plan_i64_bn2s2(words, PF_bn2s2.sinkOffset) // 4
                    ),
                    [1, sink_count],
                    [sink_count, 1],
                    dtype=pl.DT_FP32,
                )
                nzp = make_plan_nzp_bn2s2(v1w)
                if PLAN_HAS_P_NZP_bn2s2[
                    pre_sfmg * 16777216
                    + mask_mode * 1048576
                    + d_align * 1024
                    + dv_align
                ]:
                    p_nzp = make_plan_p_nzp_bn2s2(v1w)
                else:
                    p_nzp = nzp
                sfmg_ub = make_plan_sfmg_ub_bn2s2(v1w)
                lse_ub = make_plan_lse_ub_bn2s2(v1w)
                cast = make_plan_cast_bn2s2(v1w)
                if mask_mode != 0:
                    mask_ub = make_plan_mask_view_bn2s2(v1w)
                    mask_pre = make_plan_mask_pre_view_bn2s2(v1w)
                else:
                    mask_ub = sfmg_ub
                    mask_pre = sfmg_ub
                if pre_sfmg == 0:
                    y_ub = make_y_ub_bn2s2(v1w)
                    dy_ub = make_dy_ub_bn2s2(v1w)
                else:
                    y_ub = sfmg_ub
                    dy_ub = sfmg_ub
                if sink != 0:
                    sink_ub = make_sink_part_ub_bn2s2(v1w)
                else:
                    sink_ub = sfmg_ub
                if d_align > 128:
                    column = make_plan_column_f32_bn2s2(v1w)
                else:
                    column = cast
                vgm = pl.make_tuple(
                    y=ty,
                    dy=tdy,
                    dk=tdk,
                    dv=tdv,
                    lse=tlse,
                    mask=tmask,
                    sinks=tsinks,
                    sfmg=sfmg,
                    sinkPartial=partial,
                    workspace=workspace,
                )
                vbuf = pl.make_tuple(
                    y=y_ub,
                    dy=dy_ub,
                    nzp=nzp,
                    pNzp=p_nzp,
                    sfmg=sfmg_ub,
                    lse=lse_ub,
                    mask=mask_ub,
                    maskPre=mask_pre,
                    sink=sink_ub,
                    column=column,
                    cast=cast,
                    ds=ds,
                    p=p_l1,
                    mm1=mm1,
                    mm2=mm2,
                )
                saved = run_plan_vector_bn2s2(
                    ci,
                    words,
                    h,
                    cu_q,
                    cu_k,
                    core_v,
                    sub,
                    policy_v,
                    vgm,
                    vbuf,
                    slots_v,
                    v1w,
                )
            pl.system.sync_all()
            with pl.section_vector():
                post_aiv = pl.get_block_idx()
                post_f32 = make_post_f32_buffer(True)
                post_f16 = make_post_f16_buffer(True)
                special_src = make_plan_post_column_bn2s2()
                special_dst = make_plan_post_cast_bn2s2()
                if h.logicalSplit == PlanSplit_bn2s2.BN2S2:
                    plan_restore_special_owner_bn2s2(words, h, cu_k, core_v, saved)
                    plan_special_column_post_bn2s2(
                        ci,
                        policy_v,
                        saved,
                        workspace,
                        tdk,
                        special_src.current(),
                        special_dst.current(),
                        sub,
                        False,
                        v1w,
                    )
                    plan_special_column_post_bn2s2(
                        ci,
                        policy_v,
                        saved,
                        workspace,
                        tdv,
                        special_src.current(),
                        special_dst.current(),
                        sub,
                        True,
                        v1w,
                    )
                plan_post_tensor_bn2s2(
                    words,
                    post_aiv,
                    PF_bn2s2.qPostBlockFactor,
                    PF_bn2s2.qPostBlockTotal,
                    q_ws_flat,
                    q_out_flat,
                    post_f32,
                    post_f16,
                    scale,
                    False,
                    q_prefix * h.dimQ,
                )
                if h.logicalSplit == PlanSplit_bn2s2.GS:
                    plan_post_tensor_bn2s2(
                        words,
                        post_aiv,
                        PF_bn2s2.kPostBlockFactor,
                        PF_bn2s2.kPostBlockTotal,
                        k_ws_flat,
                        k_out_flat,
                        post_f32,
                        post_f16,
                        scale,
                        False,
                        k_prefix * h.dimQ,
                    )
                    plan_post_tensor_bn2s2(
                        words,
                        post_aiv,
                        PF_bn2s2.vPostBlockFactor,
                        PF_bn2s2.vPostBlockTotal,
                        v_ws_flat,
                        v_out_flat,
                        post_f32,
                        post_f16,
                        scale,
                        True,
                        k_prefix * h.dimV,
                    )
                if sink != 0:
                    sink_total = make_sink_red_post_ub(d_align, True)
                    sink_out = pl.make_tensor(
                        dsink, [1, h.heads], [h.heads, 1], dtype=pl.DT_FP32
                    )
                    plan_post_sink_bn2s2(
                        words,
                        post_aiv,
                        partial,
                        sink_out,
                        post_f32,
                        sink_total.current(),
                    )
