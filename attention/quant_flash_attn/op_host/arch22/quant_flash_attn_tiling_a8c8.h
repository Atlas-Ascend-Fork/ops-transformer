/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_HOST_TILING_A8C8_H_
#define QUANT_FLASH_ATTN_HOST_TILING_A8C8_H_

#include "../../op_kernel/arch22/quant_flash_attn_tiling_data_int8.h"
#include "../../../common/op_host/split_core.h"
#include <limits>
#include <cmath>
#include <tuple>
#include <array>

namespace optiling {

namespace quant_flash_attn {
namespace a8c8_plan {
struct QfaPlanRecipe {
    uint32_t m = 16U;
    uint32_t unit = 512U;
    uint32_t fdRows = 16U;
    uint32_t r1 = 8U;
    uint32_t r2 = 8U;
    uint32_t columns = 1U;
    uint32_t budget = 1U;
    uint32_t headGroupSize = 1U;
    uint32_t headMajorOrder = 0U;
    uint32_t splitCap = 32U;
};

inline uint32_t QfaVecRows(uint32_t rows)
{
    const uint32_t first = rows <= 8U ? rows : (((rows + 1U) / 2U + 7U) & ~7U);
    return std::min(first, rows);
}

inline uint64_t QfaUbBytes(uint32_t m, uint32_t d, uint32_t r1, uint32_t r2, uint32_t scratch = 32768U,
                           bool compactD256 = false)
{
    const uint32_t alignedD = (d + 31U) & ~31U;
    if (d != 128U && d != 256U) {
        return 2ULL * r1 * 512U * 5U + 2ULL * r2 * alignedD * 4U + (std::max(64U, m) / 2U) * alignedD * 4ULL +
               r2 * alignedD * 2ULL + scratch + 24576U;
    }
    const uint32_t rows = (QfaVecRows(m) + 7U) & ~7U;
    const uint64_t score = std::max(2ULL * r1 * 512U * 4U, 2ULL * 16U * alignedD * 4U);
    const bool wideScale = d == 128U && r1 == 16U;
    const uint64_t rowScale = compactD256 ? 2048ULL : std::max(4096ULL, rows * (wideScale ? 256ULL : 32ULL));
    // Match HpInit; GM workspace pitch and FD padding remain independent of UB rows.
    return score + 2ULL * r1 * 512U + 2ULL * r2 * alignedD * 4U + rows * alignedD * 4ULL + r2 * alignedD * 2ULL +
           scratch + (compactD256 ? 3072U : 4096U) + 512U + rowScale + 4096U + (wideScale ? 256U : 0U) + rows * 32ULL;
}

inline bool QfaUsePairedHeads(const BaseInfo& base, uint32_t d, uint32_t layout)
{
    const uint64_t rows = static_cast<uint64_t>(base.s1Size) * base.gSize;
    const uint64_t panels = (static_cast<uint64_t>(base.s2Size) + 511U) / 512U;
    return d == 128U && layout == 0U && (base.s1Size == 1U || (rows > 32U && rows <= 64U)) && base.gSize <= 32U &&
           (base.n2Size == 2U || base.n2Size == 4U) &&
           (base.bSize >= 8U || panels >= 32U || (base.bSize >= 4U && panels >= 18U));
}

// Both entry points pass zero for an unspecified bound. Only the explicit bound can tighten the launch budget.
inline bool QfaMakeRecipe(const BaseInfo& base, uint32_t d, uint32_t layout, uint32_t aicNum, uint32_t heads,
                          QfaPlanRecipe& recipe)
{
    const uint64_t rows = static_cast<uint64_t>(std::max(1U, base.s1Size)) * base.gSize;
    if (rows == 0U || base.bSize == 0U || base.n2Size == 0U || aicNum == 0U || base.n2Size % heads != 0U) {
        return false;
    }
    const bool optimized = d == 64U || d == 128U || d == 256U;
    recipe.unit = optimized ? 512U : 128U;
    recipe.headGroupSize = heads;
    recipe.m = rows <= 16U ? 16U : rows <= 32U ? 32U : rows <= 64U ? 64U : 128U;
    const uint64_t panels = (static_cast<uint64_t>(base.s2Size) + recipe.unit - 1U) / recipe.unit;
    const bool splitWideTask = rows == 128U && base.bSize == 1U && base.n2Size == 1U && panels >= 8U;
    if (splitWideTask) {
        recipe.m = 64U;
    }
    if (heads == 2U) {
        recipe.r1 = recipe.r2 = rows <= 8U ? 8U : 16U;
    } else if (optimized && QfaVecRows(static_cast<uint32_t>(std::min<uint64_t>(rows, recipe.m))) > 8U &&
               !(d == 256U && recipe.m == 128U)) {
        const uint32_t r2 = d == 128U ? 16U : 8U;
        if (QfaUbBytes(recipe.m, d, 16U, r2) <= 192U * 1024U) {
            recipe.r1 = 16U;
            recipe.r2 = r2;
        }
    }
    if (QfaUbBytes(recipe.m * heads, d, recipe.r1, recipe.r2) > 192U * 1024U) {
        recipe.m = 64U;
    }
    if (d == 256U && recipe.m == 128U) {
        recipe.splitCap = 16U;
    }
    const uint64_t tasks =
        static_cast<uint64_t>(base.bSize) * (base.n2Size / heads) * ((rows + recipe.m - 1U) / recipe.m);
    const bool compactWide = d == 256U && recipe.m == 128U && panels > recipe.splitCap;
    constexpr uint64_t MIN_KV_PANELS_FOR_SPLIT_BUDGET = 8U;
    constexpr uint64_t MIN_KV_PANELS_FOR_PAIRED_TASK_BUDGET = 32U;
    constexpr uint32_t PAIRED_TASK_PANELS_PER_PART = 5U;
    constexpr uint32_t WIDE_TASK_PANELS_PER_PART = 3U;
    constexpr uint32_t DEFAULT_PANELS_PER_PART = 2U;
    const bool usePairedTaskBudget = heads == 2U && tasks == 1U && panels >= MIN_KV_PANELS_FOR_PAIRED_TASK_BUDGET;
    const uint32_t panelsPerPart =
        usePairedTaskBudget ? PAIRED_TASK_PANELS_PER_PART :
                              (compactWide || splitWideTask ? WIDE_TASK_PANELS_PER_PART : DEFAULT_PANELS_PER_PART);
    // Round up only when budgeting a single paired-head task with long KV.
    const uint32_t roundingBias = usePairedTaskBudget ? panelsPerPart - 1U : 0U;
    const uint64_t maxParts = panels >= MIN_KV_PANELS_FOR_SPLIT_BUDGET ?
                                  std::min<uint64_t>(recipe.splitCap, (panels + roundingBias) / panelsPerPart) :
                                  1U;
    recipe.budget = base.s1Size == 0U || base.s2Size == 0U ? aicNum : std::min<uint64_t>(aicNum, tasks * maxParts);
    if ((d == 256U && !compactWide && tasks < aicNum && (base.s2Size == 0U || panels >= 8U)) ||
        (d == 128U && heads == 1U && base.bSize == 1U && rows > 32U && rows <= 64U && panels >= 8U)) {
        recipe.fdRows = 8U;
    }
    recipe.headMajorOrder = heads == 1U && d == 128U && layout == 0U && base.n2Size == 4U &&
                                    base.bSize >= recipe.budget && recipe.budget % base.n2Size == 0U ?
                                1U :
                                0U;
    return QfaUbBytes(recipe.m * heads, d, recipe.r1, recipe.r2) <= 192U * 1024U;
}

// A batch contributes one prefix entry. Head-major order reuses that prefix for each head.
class QfaTaskRanges {
public:
    QfaTaskRanges(const BaseInfo& base, const QfaPlanRecipe& recipe)
        : base_(base),
          recipe_(recipe)
    {
        const uint32_t heads = base.n2Size / recipe.headGroupSize;
        headRepeats_ = recipe.headMajorOrder == 0U ? 1U : heads;
        batchHeads_ = heads / headRepeats_;
        for (uint32_t b = 0U; b < base.bSize; ++b) {
            const uint64_t rows = static_cast<uint64_t>(base.actualSeqS1Size[b]) * base.gSize;
            blocks_[b] = (rows + recipe.m - 1U) / recipe.m;
            panels_[b] = (static_cast<uint64_t>(base.actualSeqS2Size[b]) + recipe.unit - 1U) / recipe.unit;
            taskPrefix_[b + 1U] = taskPrefix_[b] + static_cast<uint64_t>(blocks_[b]) * batchHeads_;
            workPrefix_[b + 1U] = workPrefix_[b] + (taskPrefix_[b + 1U] - taskPrefix_[b]) * panels_[b];
            if (blocks_[b] != 0U) {
                longest = std::max(longest, panels_[b]);
            }
        }
    }

    uint64_t Tasks() const
    {
        return taskPrefix_[base_.bSize] * headRepeats_;
    }
    uint64_t Work() const
    {
        return workPrefix_[base_.bSize] * headRepeats_;
    }

    uint64_t WorkAt(uint64_t task) const
    {
        if (task == Tasks()) {
            return Work();
        }
        const uint64_t cycle = task / taskPrefix_[base_.bSize];
        const uint64_t offset = task % taskPrefix_[base_.bSize];
        const uint32_t b = BatchAt(offset);
        return cycle * workPrefix_[base_.bSize] + workPrefix_[b] + (offset - taskPrefix_[b]) * panels_[b];
    }

    void Locate(uint64_t task, uint32_t& bn, uint32_t& gs, uint32_t& rows, uint32_t& panels) const
    {
        if (task == Tasks()) {
            bn = base_.bSize * (base_.n2Size / recipe_.headGroupSize);
            gs = 0U;
            return;
        }
        const uint64_t cycle = task / taskPrefix_[base_.bSize];
        const uint64_t offset = task % taskPrefix_[base_.bSize];
        const uint32_t b = BatchAt(offset);
        const uint64_t local = offset - taskPrefix_[b];
        bn = recipe_.headMajorOrder == 0U ? b * batchHeads_ + local / blocks_[b] : cycle * base_.bSize + b;
        gs = local % blocks_[b];
        rows = std::min<uint64_t>(recipe_.m, static_cast<uint64_t>(base_.actualSeqS1Size[b]) * base_.gSize -
                                                 gs * static_cast<uint64_t>(recipe_.m));
        panels = panels_[b];
    }

    uint32_t longest = 0U;

private:
    uint32_t BatchAt(uint64_t offset) const
    {
        return std::upper_bound(taskPrefix_.begin(), taskPrefix_.begin() + base_.bSize + 1U, offset) -
               taskPrefix_.begin() - 1U;
    }

    const BaseInfo& base_;
    const QfaPlanRecipe& recipe_;
    uint32_t headRepeats_ = 1U;
    uint32_t batchHeads_ = 1U;
    std::array<uint32_t, QFA_PLAN_MAX_BATCH> blocks_{};
    std::array<uint32_t, QFA_PLAN_MAX_BATCH> panels_{};
    std::array<uint64_t, QFA_PLAN_MAX_BATCH + 1U> taskPrefix_{};
    std::array<uint64_t, QFA_PLAN_MAX_BATCH + 1U> workPrefix_{};
};

inline void QfaSplitFd(SplitResult& result)
{
    uint32_t totalFDLoad = 0;
    uint32_t totalFDHeadSplit = 0;
    // 计算FD的总数据量
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        totalFDLoad += result.fdRes.s2SplitNumOfFdHead[i] * result.fdRes.gS1SplitNumOfFdHead[i];
        totalFDHeadSplit += result.fdRes.gS1SplitNumOfFdHead[i];
    }

    // 基于FA开核数量，计算每个Vector需要计算的FD数据量
    // FD均衡的最小单位为一个归约任务的一个split，所以最多占用totalFDHeadSplit个vector
    uint32_t maxVectorNum = std::min(totalFDHeadSplit, result.usedCoreNum * result.vecCubeRatio);
    double loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum);
    int64_t loadOfCurVector = 0;
    uint32_t curCoreIndex = 0;
    uint32_t preTmpFDIndexEndOfFdHead = 0;
    uint32_t preTmpFDIndexEndOfFdHeadSplit = 0;
    for (uint32_t i = 0; i < result.numOfFdHead; i++) {
        uint32_t fDKVSplitNum = result.fdRes.s2SplitNumOfFdHead[i];
        for (uint32_t gS1SplitIdx = 0; gS1SplitIdx < result.fdRes.gS1SplitNumOfFdHead[i]; gS1SplitIdx++) {
            double remainSpace = loadThrOfVector - static_cast<double>(loadOfCurVector); // 计算当前vector剩余负载空间
            // 判断是否放在当前vector的标准是剩余空间是否能容纳一半当前归约块
            if (fDKVSplitNum > remainSpace * FD_TOLERANCE_RATIO) {
                result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
                result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
                curCoreIndex += 1U;
                totalFDLoad -= static_cast<uint32_t>(loadOfCurVector); // 当前未分配的总负载
                // 根据剩余负载和剩余可用vector更新负载上限，保证最后一个vector能分配所有负载
                loadThrOfVector = static_cast<double>(totalFDLoad) / static_cast<double>(maxVectorNum - curCoreIndex);
                loadOfCurVector = 0;
            }
            loadOfCurVector += fDKVSplitNum;
            preTmpFDIndexEndOfFdHead = i;
            preTmpFDIndexEndOfFdHeadSplit = gS1SplitIdx;
        }
    }
    result.fdRes.gS1IdxEndOfFdHead[curCoreIndex] = preTmpFDIndexEndOfFdHead;
    result.fdRes.gS1IdxEndOfFdHeadSplit[curCoreIndex] = preTmpFDIndexEndOfFdHeadSplit;
    result.usedVecNumOfFd = curCoreIndex + 1;
}

inline void QfaFinalizeTopology(SplitResult& topology, const QfaPlanRecipe& recipe)
{
    for (uint32_t g = 0U; g < topology.numOfFdHead; ++g) {
        const uint32_t rows =
            (topology.fdRes.gS1SplitNumOfFdHead[g] - 1U) * 16U + topology.fdRes.gS1LastPartSizeOfFdHead[g];
        topology.fdRes.gS1SplitNumOfFdHead[g] =
            (rows + recipe.fdRows - 1U) / recipe.fdRows * recipe.columns * recipe.headGroupSize;
        topology.fdRes.gS1LastPartSizeOfFdHead[g] = (rows - 1U) % recipe.fdRows + 1U;
    }
    if (topology.numOfFdHead != 0U)
        QfaSplitFd(topology);
}

inline bool QfaFillPlan(const BaseInfo& physical, const QfaPlanRecipe& recipe, const SplitResult& topology,
                        QfaHostPlanData& plan)
{
    const uint32_t groupedKvHeads = physical.n2Size / recipe.headGroupSize;
    plan.m = recipe.m;
    plan.kvUnit = recipe.unit;
    plan.fdRows = recipe.fdRows;
    plan.fdColumns = recipe.columns;
    plan.hpRows = recipe.r1;
    plan.hpVec2Rows = recipe.r2;
    plan.headGroupSize = recipe.headGroupSize;
    plan.headMajorOrder = recipe.headMajorOrder;
    plan.cores = std::max(1U, topology.usedCoreNum);
    plan.fdGroups = topology.numOfFdHead;
    plan.fdVectors = topology.usedVecNumOfFd;
    for (uint32_t b = 0U; b < physical.bSize; ++b) {
        plan.qBlocks[b] = (physical.actualSeqS1Size[b] * physical.gSize + recipe.m - 1U) / recipe.m;
        plan.kvPanels[b] = (physical.actualSeqS2Size[b] + recipe.unit - 1U) / recipe.unit;
    }
    std::fill(plan.group0, plan.group0 + 48U, ~0U);
    std::fill(plan.group1, plan.group1 + 48U, ~0U);
    const auto coord = [](uint32_t bn, uint32_t gs, uint32_t k) { return std::make_tuple(bn, gs, k); };
    for (uint32_t c = 0U; c < topology.usedCoreNum; ++c) {
        plan.bnEnd[c] = topology.bN2End[c];
        plan.gsEnd[c] = topology.gS1End[c];
        plan.kvEnd[c] = topology.s2End[c];
    }
    uint32_t prefix = 0U;
    for (uint32_t j = 0U; j < topology.numOfFdHead; ++j) {
        plan.fdBN[j] = topology.fdRes.bN2IdxOfFdHead[j];
        plan.fdGS[j] = topology.fdRes.gS1IdxOfFdHead[j];
        plan.fdSplits[j] = topology.fdRes.s2SplitNumOfFdHead[j];
        plan.fdPrefix[j] = prefix;
        plan.fdItems[j] = topology.fdRes.gS1SplitNumOfFdHead[j];
        plan.fdLastRows[j] = topology.fdRes.gS1LastPartSizeOfFdHead[j];
        const auto begin = coord(plan.fdBN[j], plan.fdGS[j], 0U);
        const uint32_t batch =
            plan.headMajorOrder != 0U ? plan.fdBN[j] % physical.bSize : plan.fdBN[j] / groupedKvHeads;
        const auto end = coord(plan.fdBN[j], plan.fdGS[j], plan.kvPanels[batch]);
        uint32_t index = 0U;
        for (uint32_t c = 0U; c < topology.usedCoreNum; ++c) {
            const auto coreBegin = c == 0U ?
                                       coord(0U, 0U, 0U) :
                                       coord(topology.bN2End[c - 1U], topology.gS1End[c - 1U], topology.s2End[c - 1U]);
            const auto coreEnd = coord(topology.bN2End[c], topology.gS1End[c], topology.s2End[c]);
            if (coreBegin < end && coreEnd > begin) {
                if (plan.group0[c] == ~0U) {
                    plan.group0[c] = j;
                    plan.slot0[c] = prefix + index;
                } else {
                    if (plan.group1[c] != ~0U)
                        return false;
                    plan.group1[c] = j;
                    plan.slot1[c] = prefix + index;
                }
                ++index;
            }
        }
        if (index != plan.fdSplits[j])
            return false;
        prefix += plan.fdSplits[j];
    }
    plan.partials = prefix * recipe.headGroupSize;
    for (uint32_t v = 0U; v < topology.usedVecNumOfFd; ++v) {
        plan.fdEndGroup[v] = topology.fdRes.gS1IdxEndOfFdHead[v];
        plan.fdEndItem[v] = topology.fdRes.gS1IdxEndOfFdHeadSplit[v];
    }
    return true;
}

inline void QfaSplitWholeTasks(const QfaTaskRanges& tasks, uint32_t cap, SplitResult& topology)
{
    const uint64_t total = tasks.Tasks();
    topology.usedCoreNum = std::max<uint64_t>(1U, std::min<uint64_t>(cap, total));
    uint64_t start = 0U;
    for (uint32_t c = 0U; c < topology.usedCoreNum; ++c) {
        const uint32_t remaining = topology.usedCoreNum - c;
        uint64_t end = total;
        if (remaining > 1U) {
            const uint64_t assigned = tasks.WorkAt(start);
            const uint64_t target = assigned + (tasks.Work() - assigned) / remaining;
            uint64_t lo = start + 1U;
            end -= remaining - 1U;
            while (lo < end) {
                const uint64_t mid = (lo + end) / 2U;
                if (tasks.WorkAt(mid) < target)
                    lo = mid + 1U;
                else
                    end = mid;
            }
            if (end > start + 1U && tasks.WorkAt(end) > target &&
                target - tasks.WorkAt(end - 1U) <= tasks.WorkAt(end) - target)
                --end;
        }
        uint32_t rows = 0U, panels = 0U;
        tasks.Locate(end, topology.bN2End[c], topology.gS1End[c], rows, panels);
        start = end;
    }
}

inline void QfaSplitKvTasks(const QfaTaskRanges& tasks, const QfaPlanRecipe& recipe, uint32_t divisor,
                            SplitResult& topology)
{
    const uint32_t taskCount = tasks.Tasks();
    std::array<uint32_t, QFA_PLAN_MAX_CORES + 1U> demand{};
    for (uint32_t t = 0U; t < taskCount; ++t) {
        uint32_t bn, gs, rows, panels;
        tasks.Locate(t, bn, gs, rows, panels);
        const uint32_t parts = panels < 8U ? 1U : std::min(recipe.splitCap, panels / divisor);
        demand[t + 1U] = demand[t] + parts - 1U;
    }
    const uint32_t totalDemand = demand[taskCount];
    const uint32_t extra = std::min(recipe.budget - taskCount, totalDemand);
    const bool equalPanels = tasks.Work() == tasks.Tasks() * tasks.longest;
    for (uint32_t t = 0U; t < taskCount; ++t) {
        uint32_t bn, gs, rows, panels;
        tasks.Locate(t, bn, gs, rows, panels);
        const uint32_t added = totalDemand == 0U ? 0U :
                               equalPanels       ? extra / taskCount :
                                                   demand[t + 1U] * extra / totalDemand - demand[t] * extra / totalDemand;
        const uint32_t parts = 1U + added;
        for (uint32_t part = 1U; part <= parts; ++part) {
            const uint32_t core = topology.usedCoreNum++;
            if (part == parts) {
                uint32_t nextRows, nextPanels;
                tasks.Locate(t + 1U, topology.bN2End[core], topology.gS1End[core], nextRows, nextPanels);
            } else {
                topology.bN2End[core] = bn;
                topology.gS1End[core] = gs;
                topology.s2End[core] = static_cast<uint64_t>(part) * panels / parts;
            }
        }
        if (parts > 1U) {
            const uint32_t group = topology.numOfFdHead++;
            topology.fdRes.bN2IdxOfFdHead[group] = bn;
            topology.fdRes.gS1IdxOfFdHead[group] = gs;
            topology.fdRes.s2SplitNumOfFdHead[group] = parts;
            topology.fdRes.gS1SplitNumOfFdHead[group] = (rows + 15U) / 16U;
            topology.fdRes.gS1LastPartSizeOfFdHead[group] = (rows - 1U) % 16U + 1U;
        }
    }
}

inline void QfaSplitPanelStream(const QfaTaskRanges& tasks, const QfaPlanRecipe& recipe, SplitResult& topology)
{
    const uint64_t work = tasks.Work();
    const uint64_t minPanels = (tasks.longest + recipe.splitCap - 2U) / (recipe.splitCap - 1U);
    topology.usedCoreNum = std::min<uint64_t>(recipe.budget, std::max<uint64_t>(1U, work / minPanels));
    uint64_t task = 0U;
    uint32_t parts = 0U;
    for (uint32_t core = 0U; core < topology.usedCoreNum; ++core) {
        const uint64_t coreEnd = work * (core + 1U) / topology.usedCoreNum;
        uint64_t lo = task, hi = tasks.Tasks();
        while (lo < hi) {
            const uint64_t mid = lo + (hi - lo + 1U) / 2U;
            if (tasks.WorkAt(mid) <= coreEnd)
                lo = mid;
            else
                hi = mid - 1U;
        }
        uint32_t rows = 0U, panels = 0U;
        tasks.Locate(lo, topology.bN2End[core], topology.gS1End[core], rows, panels);
        topology.s2End[core] = coreEnd - tasks.WorkAt(lo);
        if (topology.s2End[core] != 0U) {
            if (lo != task || parts == 0U) {
                const uint32_t group = topology.numOfFdHead++;
                topology.fdRes.bN2IdxOfFdHead[group] = topology.bN2End[core];
                topology.fdRes.gS1IdxOfFdHead[group] = topology.gS1End[core];
                topology.fdRes.gS1SplitNumOfFdHead[group] = (rows + 15U) / 16U;
                topology.fdRes.gS1LastPartSizeOfFdHead[group] = (rows - 1U) % 16U + 1U;
                parts = 1U;
            }
            topology.fdRes.s2SplitNumOfFdHead[topology.numOfFdHead - 1U] = ++parts;
        } else {
            parts = 0U;
        }
        task = lo;
    }
}

inline bool QfaMaterializeDirect(const BaseInfo& base, const QfaPlanRecipe& recipe, uint32_t d, QfaHostPlanData& plan)
{
    const QfaTaskRanges tasks(base, recipe);
    const uint32_t cap = recipe.budget;
    SplitResult topology(cap, 2U);
    const uint64_t count = tasks.Tasks();
    const uint64_t splitLoad = (tasks.Work() + cap - 1U) / cap;
    const uint64_t wholeLoad = (count + cap - 1U) / cap * tasks.longest;
    if (cap > 1U && tasks.longest >= 8U && wholeLoad >= splitLoad + 3U) {
        const bool compactWide = d == 256U && recipe.m == 128U && tasks.longest > recipe.splitCap;
        const bool shortDualTask = count == 2U && tasks.longest == 14U && tasks.Work() == 28U;
        if (count > 1U &&
            (count >= cap || cap < 2U * count || recipe.headGroupSize == 2U || (compactWide && count > 2U))) {
            QfaSplitPanelStream(tasks, recipe, topology);
        } else {
            QfaSplitKvTasks(tasks, recipe, count == 1U || shortDualTask || compactWide ? 2U : 3U, topology);
        }
        QfaFinalizeTopology(topology, recipe);
    } else {
        QfaSplitWholeTasks(tasks, cap, topology);
    }
    if (!QfaFillPlan(base, recipe, topology, plan)) {
        return false;
    }
    for (uint32_t c = plan.cores; c < cap; ++c) {
        plan.bnEnd[c] = plan.bnEnd[plan.cores - 1U];
        plan.gsEnd[c] = plan.gsEnd[plan.cores - 1U];
        plan.kvEnd[c] = plan.kvEnd[plan.cores - 1U];
    }
    return true;
}

template <typename ScratchBytes>
inline bool QfaBuildLaunchConfig(const BaseInfo& base, uint32_t d, uint32_t layout, uint32_t pageSize, uint32_t aicNum,
                                 QfaLaunchConfig& plan, bool& useOptimizedVectorService, bool allowPairedHeads,
                                 ScratchBytes scratchBytes)
{
    const uint32_t heads = allowPairedHeads && pageSize == 128U && QfaUsePairedHeads(base, d, layout) ? 2U : 1U;
    QfaPlanRecipe recipe;
    if (!QfaMakeRecipe(base, d, layout, aicNum, heads, recipe)) {
        return false;
    }
    const uint32_t rows = std::min<uint64_t>(recipe.m, static_cast<uint64_t>(std::max(1U, base.s1Size)) * base.gSize);
    useOptimizedVectorService =
        (heads == 2U ? rows : QfaVecRows(rows)) >= 4U && (d == 64U || d == 72U || d == 128U || d == 256U);
    plan.m = recipe.m;
    plan.kvUnit = recipe.unit;
    plan.fdRows = recipe.fdRows;
    plan.fdColumns = recipe.columns;
    plan.hpRows = recipe.r1;
    plan.hpVec2Rows = recipe.r2;
    plan.hpScratchBytes = scratchBytes(plan.hpRows, 32768U);
    plan.cores = recipe.budget;
    plan.headGroupSize = heads;
    plan.headMajorOrder = recipe.headMajorOrder;
    if (useOptimizedVectorService && d == 256U && plan.m == 128U && heads == 1U) {
        const uint32_t scratch = scratchBytes(16U, 16384U);
        if (scratch == 16384U && QfaUbBytes(plan.m, d, 16U, 8U, scratch, true) <= 192U * 1024U) {
            plan.hpRows = 16U;
            plan.hpScratchBytes = scratch;
            return true;
        }
    }
    return QfaUbBytes(plan.m * heads, d, plan.hpRows, plan.hpVec2Rows, plan.hpScratchBytes) <= 192U * 1024U;
}

} // namespace a8c8_plan
} // namespace quant_flash_attn
} // namespace optiling

#endif // QUANT_FLASH_ATTN_HOST_TILING_A8C8_H_
