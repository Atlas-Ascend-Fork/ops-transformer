/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_route.h"

#include "flash_attn_grad_block_outer.h"
#include "flash_attn_grad_sparse.h"
#include "log/log.h"

namespace optiling {
namespace {

// ---------------------------------------------------------------------------
// 能力边界常量：这些数字来自 kernel 实现边界，不是性能标定，改动必须伴随 kernel 改动
// ---------------------------------------------------------------------------
constexpr int64_t BN2_MAX_S = 128;          // 单块 BN2 的 S 上限
constexpr int64_t BN2_MAX_D = MAX_HEAD_DIM; // BN2 一系列模板的 D 上限（=768）
constexpr int64_t BN2_MULTIBLK_SEQ = 640;   // MultiBlk 的 S 上限
constexpr int64_t BN2S2_MAX_D = 512;        // Ordinary BN2S2 limit; deterministic admission uses useDeterPlan.
constexpr int64_t BN2S2_MAX_S = 1024;       // BN2S2 的 S2 上限（S1 靠 s1<s2 约束）
constexpr int64_t BN2S2_WRITE_UB_D = 128;   // BN2S2 的写入 UB D 上限

// ---------------------------------------------------------------------------
// 偏好门限常量：全部来自上板实测，改这里不影响功能正确性，只影响快慢
// ---------------------------------------------------------------------------
constexpr int64_t BN2_MULTIBLK_BN_128 = 128;
constexpr int64_t BN2_MULTIBLK_BN_256 = 256;
constexpr int64_t BN2_ALIGN128 = 128;

// ===== 能力谓词 =============================================================
// 变长 TND 的 BN2 单块准入：host 读不到 cu_seqlens 的数据，但 max_seqlen 是
// 每 batch 长度的上界——两侧都 ≤ BN2_MAX_S 时，任何 batch 都装得进一个
// BN2 基本块（运行期 s1Outer=s2Outer=1），kernel 用 cu_seqlens 逐 batch 寻址。
// 该档与 asc 的 "变长 TND 实际 max≤128 → BN2 单块" 对齐；MultiBlk 仍不接
// 变长 TND（其 per-core 条带寻址依赖固定 s1Outer），见 Feasible_Bn2MultiBlk。
// mask 4 的滑动窗口在变长下可能出现整块无效（skip_invalid 整块跳过、该
// batch 的 out 行没有写者），本轮不放开，保持 GS1S2 兜底；mask 3 右下对齐
// 时每个非空 batch 至少有一行有效，不会出现整块无效。
bool TndVarlenBn2SingleFeasible(const FagParsedInfo& p)
{
    if (p.maskMode == MASK_MODE_WINDOW) {
        return false;
    }
    return p.maxSeqlenQ > 0 && p.maxSeqlenKv > 0 && p.maxSeqlenQ <= BN2_MAX_S && p.maxSeqlenKv <= BN2_MAX_S;
}

// GQA and D>MAX_HEAD_DIM stay outside BN2. Short variable-length TND
// keeps the existing single-block route; D/Dv is checked per path.
bool Bn2FamilyFeasible(const FagParsedInfo& p)
{
    // 等长 TND（tndAllSame，parser 用 shape 算术判定）的 GM 排布与 BSND
    // 四维视图逐元素重合，可走 BN2（layout=1 二进制，lse 走 TND 2D 视图）；
    // 变长 TND 仅在 max_seqlen 可给出单块上界时放开。0930 bisect-sink 定案移植。
    // Effective lengths are runtime values. Even physically equal TND batches
    // need the variable-length GS grid when either seqused input is supplied.
    if (p.hasSeqused || (p.isTnd && !p.tndAllSame && !TndVarlenBn2SingleFeasible(p))) {
        return false;
    }
    // Host and AICPU must agree on the single-block D192/Dv128 admission.
    // Zero-task launches still need GS pre/post to clear a nonempty opposite
    // output side. Single-token sequences remain eligible for BN2.
    return p.s1 > 0 && p.s2 > 0 && p.n1 == p.n2 && p.d <= BN2_MAX_D;
}

bool Feasible_Bn2Single(const FagParsedInfo& p)
{
    // ASC single-block BN2 supports unequal D/Dv; wide TND remains excluded.
    const bool supportedDims = p.d == p.dv || (!p.isTnd && p.d == 192 && p.dv == 128);
    return Bn2FamilyFeasible(p) && supportedDims && p.s1 <= BN2_MAX_S && p.s2 <= BN2_MAX_S;
}

bool Feasible_Bn2MultiBlk(const FagParsedInfo& p)
{
    // S 落在 (128, 640]：这些都是 MultiBlk kernel 的实现前提。
    // mask 3/4 也可以走 MultiBlk；无效行/列由 bn2NeedZero 另开一份二进制。
    // Variable-length TND stays single-block; multi-block requires D==Dv.
    return !(p.isTnd && !p.tndAllSame) && Bn2FamilyFeasible(p) && p.d == p.dv &&
           (p.s1 > BN2_MAX_S || p.s2 > BN2_MAX_S) && p.s1 <= BN2_MULTIBLK_SEQ && p.s2 <= BN2_MULTIBLK_SEQ;
}

// Deterministic BN2S2 uses the existing metadata plan and admission unchanged.
// The ordinary path below integrates the non-deterministic master routing only.
bool Feasible_Bn2S2(const FagParsedInfo& p)
{
    if (p.isDeterministic) {
        return p.useDeterPlan;
    }
    /* step1：所有场景基础门限 */
    if (p.d > BN2S2_MAX_D || p.n1 != p.n2) {
        return false;
    }

    // A shape-only necessary bound for ordinary TND column packing. Actual
    // sequence-dependent saturation and GS fallback are resolved by metadata.
    if (p.layout == LAYOUT_TND) {
        int64_t s1Outer = 0;
        int64_t s2Outer = 0;
        FagS1S2Outer(p.s1, p.s2, s1Outer, s2Outer);
        return p.b * p.n2 * p.g * s2Outer >= static_cast<int64_t>(p.aicNum);
    }

    // Effective lengths force metadata to use a TND/GS grid. The non-TND
    // ordinary BN2S2 binary requires whole-column ownership for direct stores.
    if (p.hasSeqused) {
        return false;
    }

    /*
       step3：非 TND 场景门限。（老FAG的bn2S2NotTndLimit）
       sparse 判定用 parser 已按右对齐修正的s1Token/s2Token（FagFillSparseInfo）：mask0 -> (INT32_MAX, INT32_MAX)
       恒不剪； causal -> (INF, 0) 恒剪；band 用修正窗比 S —— 与老 SetSparseParams 的 !isSparse 逐口径一致（修正窗盖住 S
       的 band 老算子放行，这里同样放行）。 s1==0 的零 Q 任务必须落 GS：metadata DoFagBn2s2DenseSplit 的 fagS1Outer_==0
       早退会触发 Scenario 1 回退 GS（零块），且 BN2S2 dense 的 pre/post 只清/收
       dq 段——dK/dV 直出依赖 s1>0 时每列必写，零 Q 会让 dK/dV 留脏数据。
       与 Bn2FamilyFeasible 的 s1>0 同口径，零任务一律交 GS 兜底。
     */
    if (!(p.s1 > 0 && p.s1 < p.s2 && p.s2 <= BN2S2_MAX_S && p.s2 - p.s1 >= BN2_MAX_S && p.d <= BN2S2_WRITE_UB_D &&
          p.s1 <= p.s1Token && p.s2 <= p.s2Token && !p.isDeterministic)) {
        return false;
    }

    /*
       step4：“定长+确定性+TND”和“BSND”最后门限
       本算子特有（非老算子路由条件，不可省）：老算子 BN2S2 切不满片时会回退 GS1S2 并收缩 blockDim；
       本算子 host tiling 先于 AICPU metadata 运行，
       tilingkey 已固化为 BN2S2，metadata 回退 GS 口径切块会让 kernel 块语义错配。
       故定长场景在路由阶段就要求列分核铺满整片，不饱和的 shape 落 GS1S2。
    */
    int64_t s1Outer = 0;
    int64_t s2Outer = 0;
    FagS1S2Outer(p.s1, p.s2, s1Outer, s2Outer);
    uint32_t usedCube = static_cast<uint32_t>(
        FagUsedCubeCores(TMPL_BN2S2, p.b, p.n2, p.g, s1Outer, s2Outer, static_cast<int64_t>(p.aicNum)));
    return usedCube >= p.aicNum;
}

bool Feasible_Always(const FagParsedInfo&)
{
    return true;
}

// ===== 偏好谓词 =============================================================
bool Prefer_Always(const FagParsedInfo&)
{
    return true;
}

// BN 够大时 MultiBlk 才划算：BN>=256，或 BN>=128 且 S1/S2 都按 128 对齐。
bool Prefer_Bn2BnGate(const FagParsedInfo& p)
{
    const int64_t bn = p.b * p.n1;
    return (bn >= BN2_MULTIBLK_BN_256) ||
           (bn >= BN2_MULTIBLK_BN_128 && (p.s1 % BN2_ALIGN128 == 0) && (p.s2 % BN2_ALIGN128 == 0));
}

// ===== 候选表 ===============================================================
// 顺序即性能优先级，越靠前越希望命中；末行必须恒可行，保证一定选得出来。
// 新增 kernel 模板 = 加一行；实测结论变化 = 调顺序或改 prefer。
struct FagRoute {
    const char* name;
    int64_t tmpl;
    int64_t isBn2MultiBlk;
    bool (*feasible)(const FagParsedInfo&);
    bool (*prefer)(const FagParsedInfo&);
    const char* calibration; // 该行偏好门限的依据出处
};

constexpr FagRoute kRoutes[] = {
    {"BN2_single", TMPL_BN2, 0, Feasible_Bn2Single, Prefer_Always,
     "S<=128 时不存在多核累加，省掉 pre 清零与 post cast"},
    {"BN2_multiblk", TMPL_BN2, 1, Feasible_Bn2MultiBlk, Prefer_Bn2BnGate, "BN>=256，或 BN>=128 且 S1/S2 对齐 128"},
    {"BN2S2", TMPL_BN2S2, 0, Feasible_Bn2S2, Prefer_Always,
     "KV 比 Q 长至少一块时按 BNS2 切核，S1 核内串行，省掉 S1 轴分核开销"},
    {"GS1S2", TMPL_BN2GS1S2, 0, Feasible_Always, Prefer_Always, "兜底，恒可行"},
};

constexpr size_t kRouteNum = sizeof(kRoutes) / sizeof(kRoutes[0]);
static_assert(kRouteNum <= FAG_TRACE_SLOT_NUM, "trace slots must cover all routes");

// ===== D / Dv 分档 ==========================================================
// D 决定每块片上 tile 的宽度，而 tile shape 在 pypto 是 trace 期常量，所以必须
// 做成编译期分档。一律按最大档分配不只是
// 慢：L0C 会超出 256KB 预算，物理上放不下。
// Dv 只决定已分配 tile 里填多少列，是纯运行期数值；但 y/dy 的 UB 宽度由 Dv
// 派生，故仍需一个分配档次。dAlign 与 dvAlign 取值相同；唯一分叉是
// D=192 且 Dv=128 → dvAlign=128。其余 D!=Dv 已在 CheckHeadDim 拦截。
ge::graphStatus SelectDBucket(gert::TilingContext* context, const FagParsedInfo& p, FagKernelPlan& plan)
{
    OP_CHECK_IF(p.d > MAX_HEAD_DIM,
                OP_LOGE(context->GetNodeName(), "head dim of q (%ld) is not supported; only D <= %ld is implemented.",
                        p.d, MAX_HEAD_DIM),
                return ge::GRAPH_PARAM_INVALID);

    if (p.d <= 64) {
        plan.dAlign = 64;
    } else if (p.d <= 128) {
        plan.dAlign = 128;
    } else if (p.d <= 192) {
        plan.dAlign = 192;
    } else if (p.d <= 256) {
        plan.dAlign = 256;
    } else if (!p.isDeterministic && p.d <= 512) {
        // The ordinary 512 bucket must not change the validated deterministic
        // 768 physical layout, including deterministic GS fallbacks.
        plan.dAlign = 512;
    } else {
        plan.dAlign = 768;
    }
    plan.dvAlign = (p.d == 192 && p.dv == 128) ? 128 : plan.dAlign;
    return ge::GRAPH_SUCCESS;
}

} // namespace

ge::graphStatus SelectKernelPlan(gert::TilingContext* context, const FagParsedInfo& info, FagKernelPlan& plan,
                                 FagRouteTrace& trace)
{
    // The WJF metadata plan owns its complete deterministic domain, including
    // the GS runtime fallback. Do not let a mainline family steal that key.
    if (info.useDeterPlan) {
        plan.tmpl = TMPL_BN2S2;
        plan.isBn2MultiBlk = 0;
        plan.bn2NeedZero = 0;
        trace.chosen = "bn2s2-deterministic-metadata";
        return SelectDBucket(context, info, plan);
    }
    bool matched = false;
    for (size_t i = 0; i < kRouteNum; ++i) {
        const FagRoute& route = kRoutes[i];
        if (!route.feasible(info)) {
            trace.Reject(route.name, "capability");
            continue;
        }
        if (!route.prefer(info)) {
            trace.Reject(route.name, "preference");
            continue;
        }
        // MultiBlk 多核写同一 dq 行。确定性改走 GS1S2 核上闭式，不另编 MultiBlk deter。
        if (info.isDeterministic && route.isBn2MultiBlk == 1) {
            trace.Reject(route.name, "deterministic-gs1s2");
            continue;
        }
        plan.tmpl = route.tmpl;
        plan.isBn2MultiBlk = route.isBn2MultiBlk;
        trace.chosen = route.name;
        matched = true;
        break;
    }

    // 末行恒可行，正常不会走到这里；留断言防止有人删掉兜底行。
    OP_CHECK_IF(!matched, OP_LOGE(context->GetNodeName(), "no feasible route found, the fallback route is missing."),
                return ge::GRAPH_FAILED);
    // D5 兜底：保留取值 2 绝不能流到 tilingkey。
    OP_CHECK_IF(
        plan.tmpl != TMPL_BN2GS1S2 && plan.tmpl != TMPL_BN2 && plan.tmpl != TMPL_BN2S2,
        OP_LOGE(context->GetNodeName(), "route produced reserved template %ld, only %ld/%ld/%ld are implemented.",
                plan.tmpl, TMPL_BN2GS1S2, TMPL_BN2, TMPL_BN2S2),
        return ge::GRAPH_FAILED);

    if (plan.isBn2MultiBlk == 1 && (info.maskMode == MASK_MODE_CAUSAL || info.maskMode == MASK_MODE_WINDOW) &&
        FagBn2HasInvalidOuter(info.s1, info.s2, info.s1Token, info.s2Token)) {
        plan.bn2NeedZero = 1;
    }

    // 变长 TND 的 BN2 单块：host 读不到 cu_seqlens，无法像 asc 那样在存在空
    // batch 时退回 GS1S2；kernel 的 packed 索引空间不为空 batch 产生任务，
    // 需要 pre 阶段把「空 batch 对侧的 out 行」补零。用 bn2NeedZero（bit13）
    // 区分出这份二进制，同时让 tiling 打开 BATCH_MODE（未满核启动时
    // sync_all 必须靠它）。等长 TND 不可能有跨 batch 的空序列，保持
    // need_zero=0，不为既有路径增加清零开销。
    if (plan.tmpl == TMPL_BN2 && plan.isBn2MultiBlk == 0 && info.isTnd && !info.tndAllSame) {
        plan.bn2NeedZero = 1;
    }

    return SelectDBucket(context, info, plan);
}

} // namespace optiling
