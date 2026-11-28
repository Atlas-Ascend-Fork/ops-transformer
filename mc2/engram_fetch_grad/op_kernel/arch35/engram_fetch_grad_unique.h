/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENGRAM_FETCH_GRAD_UNIQUE_H
#define ENGRAM_FETCH_GRAD_UNIQUE_H

#include "kernel_operator.h"
// 注：核内多流水同步暂保留 SetFlag/WaitFlag（950 代际存量口径），后续迭代迁移 AscendC::Mutex。

#include "adv_api/reduce/reduce.h"
#include "../engram_fetch_grad_utils.h"

namespace EngramFetchGradUnique {

constexpr uint32_t ENTRY_BATCH_CAP_HALF = 2U;
constexpr uint32_t RUN_START_UB_OFFSET = 2U;
constexpr uint32_t RUN_LEN_UB_OFFSET = 2U;
constexpr uint32_t ACCUM_LIST_UB_OFFSET = 4U;
constexpr uint32_t COMPACT_INDEX_UB_OFFSET = 3U;
constexpr uint32_t DIRECT_FLAG_UB_OFFSET = 4U;
constexpr uint32_t UNIQUE_DIFF_UB_OFFSET = 2U;
// Dup/Run 链表与 CompactIdx、DirectFlag/AccumList 分时复用同一槽位, 槽内再拆前后半槽
constexpr uint32_t DUP_START_UB_OFFSET = 3U;
constexpr uint32_t DUP_END_UB_OFFSET = 4U;
constexpr uint32_t RUN_COMPACT_UB_OFFSET = 4U;
constexpr uint32_t CHUNK_EVT_BUFS = 4U; // 分块列通道的乒乓事件缓冲数

template <AscendC::HardEvent event>
__aicore__ inline void SyncFunc(AscendC::TPipe& pipe)
{
    int32_t eventID = static_cast<int32_t>(pipe.FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

static __aicore__ inline uint32_t GetDtypeSize(int32_t dtype)
{
    if (dtype == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
        return sizeof(bfloat16_t);
    }
    if (dtype == Mc2Kernel::ENGRAM_DT_FLOAT16) {
        return sizeof(half);
    }
    return sizeof(float);
}

constexpr uint32_t VF_CHUNK_INT32 = 64U;

struct EngramVfTables {
    static constexpr uint32_t GUARD = 0U;
    static constexpr uint32_t ISNEWN = 32U;
    static constexpr uint32_t ISNEW = 1056U;
    static constexpr uint32_t SCAN_PAD = 2080U;
    static constexpr uint32_t SCAN_DAT = 2112U;
    static constexpr uint32_t PREFIX = 2176U;
    static constexpr uint32_t DRF = 3200U;
    static constexpr uint32_t RUNSF = 4224U;
    static constexpr uint32_t RID = 5248U;
    static constexpr uint32_t DUPSF = 6272U;
    static constexpr uint32_t GID = 7296U;
    static constexpr uint32_t RUNEF = 8320U;
    static constexpr uint32_t DUPEF = 9344U;
    static constexpr uint32_t E2 = 10368U;
    static constexpr uint32_t TOTAL_INT32 = 11392U;
};
static_assert(EngramVfTables::TOTAL_INT32 * sizeof(int32_t) <= Mc2Kernel::GRAD_BUF_BYTES,
              "vf scratch tables must fit inside gradBuf (walk reuses it after judging)");

__simd_callee__ __aicore__ inline void EngramVfScanTable(__ubuf__ int32_t* src, __ubuf__ int32_t* dst,
                                                         __ubuf__ int32_t* scanDat, uint32_t b)
{
    const uint32_t chunks = (b + VF_CHUNK_INT32 - 1U) / VF_CHUNK_INT32;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t>();
    AscendC::Reg::RegTensor<int32_t> zeroReg;
    AscendC::Reg::Duplicate(zeroReg, (int32_t)0, allMask);
    for (uint32_t c = 0U; c < chunks; c++) {
        const uint32_t base = c * VF_CHUNK_INT32;
        const uint32_t n = (b - base > VF_CHUNK_INT32) ? VF_CHUNK_INT32 : (b - base);
        AscendC::Reg::RegTensor<int32_t> xReg;
        AscendC::Reg::LoadAlign(xReg, src + base);
        for (uint32_t s = 1U; s < VF_CHUNK_INT32; s <<= 1U) {
            AscendC::Reg::StoreAlign(scanDat, xReg, allMask);
            AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
            AscendC::Reg::RegTensor<int32_t> shReg;
            AscendC::Reg::Load<int32_t>(shReg, scanDat - s);
            AscendC::Reg::Add<int32_t>(xReg, xReg, shReg, allMask);
        }
        if (c > 0U) {
            AscendC::Reg::RegTensor<int32_t> carryIdx;
            AscendC::Reg::Duplicate(carryIdx, static_cast<int32_t>(base) - 1, allMask);
            AscendC::Reg::RegTensor<int32_t> carryReg;
            AscendC::Reg::Gather<int32_t>(carryReg, dst, (AscendC::Reg::RegTensor<uint32_t>&)carryIdx, allMask);
            AscendC::Reg::Add<int32_t>(xReg, xReg, carryReg, allMask);
        }
        uint32_t nVal = n;
        AscendC::Reg::MaskReg nMask = AscendC::Reg::UpdateMask<int32_t>(nVal);
        AscendC::Reg::RegTensor<int32_t> xClean;
        AscendC::Reg::Select(xClean, xReg, zeroReg, nMask);
        AscendC::Reg::StoreAlign(dst + base, xClean, allMask);
    }
}

// 以下 __simd_callee__ 辅助承接 EngramVfBatchUniqueVF 的各处理阶段：
// 各阶段寄存器/掩码就地重建，LocalMemBar 位于阶段边界。

// 逐 chunk 生成 nn（entry[i]!=entry[i+1]）写入 isnewNP，末元素按 0 处理
__simd_callee__ __aicore__ inline __attribute__((always_inline)) void EngramVfFillNewNext(__ubuf__ int32_t* entry,
                                                                                          __ubuf__ int32_t* vfBase,
                                                                                          uint32_t b, uint32_t chunks)
{
    __ubuf__ int32_t* e2P = vfBase + EngramVfTables::E2;
    __ubuf__ int32_t* isnewNP = vfBase + EngramVfTables::ISNEWN;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t>();
    AscendC::Reg::RegTensor<int32_t> zeroReg;
    AscendC::Reg::RegTensor<int32_t> onesReg;
    AscendC::Reg::Duplicate(zeroReg, (int32_t)0, allMask);
    AscendC::Reg::Duplicate(onesReg, (int32_t)1, allMask);
    for (uint32_t c = 0U; c < chunks; c++) {
        const uint32_t base = c * VF_CHUNK_INT32;
        AscendC::Reg::RegTensor<int32_t> e2Reg;
        AscendC::Reg::RegTensor<int32_t> enReg;
        AscendC::Reg::LoadAlign(e2Reg, e2P + base);
        AscendC::Reg::LoadAlign(enReg, entry + base);
        AscendC::Reg::MaskReg neMask;
        AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::NE>(neMask, e2Reg, enReg, allMask);
        AscendC::Reg::RegTensor<int32_t> nnReg;
        AscendC::Reg::Select(nnReg, onesReg, zeroReg, neMask);
        const int32_t keep = static_cast<int32_t>(b) - 1 - static_cast<int32_t>(base);
        if (keep > 0) {
            uint32_t keepN =
                (static_cast<uint32_t>(keep) > VF_CHUNK_INT32) ? VF_CHUNK_INT32 : static_cast<uint32_t>(keep);
            AscendC::Reg::MaskReg keepMask = AscendC::Reg::UpdateMask<int32_t>(keepN);
            AscendC::Reg::RegTensor<int32_t> nnClean;
            AscendC::Reg::Select(nnClean, nnReg, zeroReg, keepMask);
            AscendC::Reg::StoreAlign(isnewNP + base, nnClean, allMask);
        } else {
            AscendC::Reg::StoreAlign(isnewNP + base, zeroReg, allMask);
        }
    }
}

// isnewP[0] 播种（firstElem 或 entry[0]!=prevEntry），后续 chunk 右移一位接续
__simd_callee__ __aicore__ inline __attribute__((always_inline)) void EngramVfSeedIsNewHead(
    __ubuf__ int32_t* entry, __ubuf__ int32_t* vfBase, uint32_t chunks, uint32_t firstElem, int32_t prevEntry)
{
    __ubuf__ int32_t* isnewNP = vfBase + EngramVfTables::ISNEWN;
    __ubuf__ int32_t* isnewP = vfBase + EngramVfTables::ISNEW;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t>();
    AscendC::Reg::RegTensor<int32_t> zeroReg;
    AscendC::Reg::RegTensor<int32_t> onesReg;
    AscendC::Reg::Duplicate(zeroReg, (int32_t)0, allMask);
    AscendC::Reg::Duplicate(onesReg, (int32_t)1, allMask);
    {
        AscendC::Reg::RegTensor<int32_t> shReg;
        AscendC::Reg::Load<int32_t>(shReg, isnewNP - 1);
        AscendC::Reg::RegTensor<int32_t> arReg;
        AscendC::Reg::Arange<int32_t>(arReg, (int32_t)0);
        AscendC::Reg::MaskReg laneZero;
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(laneZero, arReg, (int32_t)0, allMask);
        AscendC::Reg::RegTensor<int32_t> isReg;
        if (firstElem != 0U) {
            AscendC::Reg::Select(isReg, onesReg, shReg, laneZero);
        } else {
            AscendC::Reg::RegTensor<int32_t> enReg;
            AscendC::Reg::RegTensor<int32_t> prevReg;
            AscendC::Reg::LoadAlign(enReg, entry);
            AscendC::Reg::Duplicate(prevReg, prevEntry, allMask);
            AscendC::Reg::MaskReg neMask;
            AscendC::Reg::Compare<int32_t, AscendC::CMPMODE::NE>(neMask, enReg, prevReg, allMask);
            AscendC::Reg::RegTensor<int32_t> cmpOnes;
            AscendC::Reg::Select(cmpOnes, onesReg, zeroReg, neMask);
            AscendC::Reg::Select(isReg, cmpOnes, shReg, laneZero);
        }
        AscendC::Reg::StoreAlign(isnewP, isReg, allMask);
    }
    for (uint32_t c = 1U; c < chunks; c++) {
        const uint32_t base = c * VF_CHUNK_INT32;
        AscendC::Reg::RegTensor<int32_t> shReg;
        AscendC::Reg::Load<int32_t>(shReg, isnewNP + base - 1U);
        AscendC::Reg::StoreAlign(isnewP + base, shReg, allMask);
    }
}

// 按 prefix 前缀索引把首现 entry 值散射进 uniqueT
__simd_callee__ __aicore__ inline __attribute__((always_inline)) void EngramVfScatterUnique(
    __ubuf__ int32_t* entry, __ubuf__ int32_t* vfBase, __ubuf__ int32_t* uniqueT, uint32_t chunks, int32_t rankBase)
{
    __ubuf__ int32_t* isnewP = vfBase + EngramVfTables::ISNEW;
    __ubuf__ int32_t* prefixP = vfBase + EngramVfTables::PREFIX;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t>();
    AscendC::Reg::RegTensor<int32_t> zeroReg;
    AscendC::Reg::Duplicate(zeroReg, (int32_t)0, allMask);
    for (uint32_t c = 0U; c < chunks; c++) {
        const uint32_t base = c * VF_CHUNK_INT32;
        AscendC::Reg::RegTensor<int32_t> isReg;
        AscendC::Reg::LoadAlign(isReg, isnewP + base);
        AscendC::Reg::MaskReg newMask;
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(newMask, isReg, (int32_t)0, allMask);
        AscendC::Reg::RegTensor<int32_t> preReg;
        AscendC::Reg::LoadAlign(preReg, prefixP + base);
        AscendC::Reg::RegTensor<int32_t> idxReg;
        AscendC::Reg::Adds(idxReg, preReg, (int32_t)-1, allMask);
        AscendC::Reg::MaskReg posMask;
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(posMask, preReg, (int32_t)0, allMask);
        AscendC::Reg::RegTensor<int32_t> idxClamped;
        AscendC::Reg::Select(idxClamped, idxReg, zeroReg, posMask);
        AscendC::Reg::RegTensor<int32_t> enReg;
        AscendC::Reg::LoadAlign(enReg, entry + base);
        AscendC::Reg::RegTensor<int32_t> valReg;
        AscendC::Reg::Adds(valReg, enReg, -rankBase, allMask);
        AscendC::Reg::Scatter<int32_t>(uniqueT, valReg, (AscendC::Reg::RegTensor<uint32_t>&)idxClamped, newMask);
    }
}

// drf = isnew & nn；runS/runE 为 direct-run 起止标记（首块 lane0 特判）
__simd_callee__ __aicore__ inline __attribute__((always_inline)) void EngramVfBuildRunFlags(__ubuf__ int32_t* vfBase,
                                                                                            uint32_t chunks)
{
    __ubuf__ int32_t* isnewP = vfBase + EngramVfTables::ISNEW;
    __ubuf__ int32_t* isnewNP = vfBase + EngramVfTables::ISNEWN;
    __ubuf__ int32_t* drfP = vfBase + EngramVfTables::DRF;
    __ubuf__ int32_t* runSFP = vfBase + EngramVfTables::RUNSF;
    __ubuf__ int32_t* runEFP = vfBase + EngramVfTables::RUNEF;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t>();
    for (uint32_t c = 0U; c < chunks; c++) {
        const uint32_t base = c * VF_CHUNK_INT32;
        AscendC::Reg::RegTensor<int32_t> isReg;
        AscendC::Reg::RegTensor<int32_t> nnReg;
        AscendC::Reg::RegTensor<int32_t> isPrevReg;
        AscendC::Reg::RegTensor<int32_t> nnNextReg;
        AscendC::Reg::LoadAlign(isReg, isnewP + base);
        AscendC::Reg::LoadAlign(nnReg, isnewNP + base);
        AscendC::Reg::Load<int32_t>(isPrevReg, isnewP + base - 1U);
        AscendC::Reg::Load<int32_t>(nnNextReg, isnewNP + base + 1U);
        AscendC::Reg::RegTensor<int32_t> drfReg;
        AscendC::Reg::RegTensor<int32_t> drfPrevReg;
        AscendC::Reg::RegTensor<int32_t> drfNextReg;
        AscendC::Reg::And<int32_t>(drfReg, isReg, nnReg, allMask);
        AscendC::Reg::And<int32_t>(drfPrevReg, isPrevReg, isReg, allMask);
        AscendC::Reg::And<int32_t>(drfNextReg, nnReg, nnNextReg, allMask);
        AscendC::Reg::StoreAlign(drfP + base, drfReg, allMask);

        AscendC::Reg::RegTensor<int32_t> notPrev;
        AscendC::Reg::Not(notPrev, drfPrevReg, allMask);
        AscendC::Reg::RegTensor<int32_t> runSFReg;
        AscendC::Reg::And<int32_t>(runSFReg, drfReg, notPrev, allMask);
        if (c == 0U) {
            AscendC::Reg::RegTensor<int32_t> arReg;
            AscendC::Reg::Arange<int32_t>(arReg, (int32_t)0);
            AscendC::Reg::MaskReg laneZero;
            AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(laneZero, arReg, (int32_t)0, allMask);
            AscendC::Reg::RegTensor<int32_t> fixed;
            AscendC::Reg::Select(fixed, drfReg, runSFReg, laneZero);
            runSFReg = fixed;
        }
        AscendC::Reg::StoreAlign(runSFP + base, runSFReg, allMask);

        AscendC::Reg::RegTensor<int32_t> notNext;
        AscendC::Reg::Not(notNext, drfNextReg, allMask);
        AscendC::Reg::RegTensor<int32_t> runEFReg;
        AscendC::Reg::And<int32_t>(runEFReg, drfReg, notNext, allMask);
        AscendC::Reg::StoreAlign(runEFP + base, runEFReg, allMask);
    }
}

// dupS/dupE 为需累加段起止标记（首块 lane0 / 末块 last lane 特判）；drf/notDrf 就地重算
__simd_callee__ __aicore__ inline __attribute__((always_inline)) void EngramVfBuildDupFlags(__ubuf__ int32_t* vfBase,
                                                                                            uint32_t b, uint32_t chunks)
{
    __ubuf__ int32_t* isnewP = vfBase + EngramVfTables::ISNEW;
    __ubuf__ int32_t* isnewNP = vfBase + EngramVfTables::ISNEWN;
    __ubuf__ int32_t* dupSFP = vfBase + EngramVfTables::DUPSF;
    __ubuf__ int32_t* dupEFP = vfBase + EngramVfTables::DUPEF;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t>();
    AscendC::Reg::RegTensor<int32_t> zeroReg;
    AscendC::Reg::RegTensor<int32_t> onesReg;
    AscendC::Reg::Duplicate(zeroReg, (int32_t)0, allMask);
    AscendC::Reg::Duplicate(onesReg, (int32_t)1, allMask);
    for (uint32_t c = 0U; c < chunks; c++) {
        const uint32_t base = c * VF_CHUNK_INT32;
        AscendC::Reg::RegTensor<int32_t> isReg;
        AscendC::Reg::RegTensor<int32_t> nnReg;
        AscendC::Reg::LoadAlign(isReg, isnewP + base);
        AscendC::Reg::LoadAlign(nnReg, isnewNP + base);
        AscendC::Reg::RegTensor<int32_t> drfReg;
        AscendC::Reg::And<int32_t>(drfReg, isReg, nnReg, allMask);
        AscendC::Reg::RegTensor<int32_t> notDrf;
        AscendC::Reg::Not(notDrf, drfReg, allMask);
        AscendC::Reg::RegTensor<int32_t> dupSFReg;
        if (c == 0U) {
            AscendC::Reg::RegTensor<int32_t> arReg;
            AscendC::Reg::RegTensor<int32_t> laneZeroVec;
            AscendC::Reg::RegTensor<int32_t> orReg;
            AscendC::Reg::Arange<int32_t>(arReg, (int32_t)0);
            AscendC::Reg::MaskReg laneZero;
            AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(laneZero, arReg, (int32_t)0, allMask);
            AscendC::Reg::Select(laneZeroVec, onesReg, zeroReg, laneZero);
            AscendC::Reg::Or<int32_t>(orReg, isReg, laneZeroVec, allMask);
            AscendC::Reg::And<int32_t>(dupSFReg, notDrf, orReg, allMask);
        } else {
            AscendC::Reg::And<int32_t>(dupSFReg, notDrf, isReg, allMask);
        }
        AscendC::Reg::StoreAlign(dupSFP + base, dupSFReg, allMask);

        AscendC::Reg::RegTensor<int32_t> dupEFReg;
        if (c + 1U == chunks) {
            AscendC::Reg::RegTensor<int32_t> arReg;
            AscendC::Reg::RegTensor<int32_t> isLastVec;
            AscendC::Reg::RegTensor<int32_t> orReg;
            AscendC::Reg::Arange<int32_t>(arReg, (int32_t)0);
            AscendC::Reg::MaskReg lastMask;
            AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(lastMask, arReg, static_cast<int32_t>(b - 1U - base),
                                                                  allMask);
            AscendC::Reg::Select(isLastVec, onesReg, zeroReg, lastMask);
            AscendC::Reg::Or<int32_t>(orReg, nnReg, isLastVec, allMask);
            AscendC::Reg::And<int32_t>(dupEFReg, notDrf, orReg, allMask);
        } else {
            AscendC::Reg::And<int32_t>(dupEFReg, notDrf, nnReg, allMask);
        }
        AscendC::Reg::StoreAlign(dupEFP + base, dupEFReg, allMask);
    }
}

// 按 rid/gid 前缀索引散射 run/dup 六张输出表
__simd_callee__ __aicore__ inline __attribute__((always_inline)) void EngramVfScatterRunDup(
    __ubuf__ int32_t* vfBase, __ubuf__ int32_t* runStartT, __ubuf__ int32_t* runEndT, __ubuf__ int32_t* runCompactT,
    __ubuf__ int32_t* dupStartT, __ubuf__ int32_t* dupEndT, __ubuf__ int32_t* dupCompactT, uint32_t chunks,
    int32_t runningOffset)
{
    __ubuf__ int32_t* prefixP = vfBase + EngramVfTables::PREFIX;
    __ubuf__ int32_t* ridP = vfBase + EngramVfTables::RID;
    __ubuf__ int32_t* gidP = vfBase + EngramVfTables::GID;
    __ubuf__ int32_t* runSFP = vfBase + EngramVfTables::RUNSF;
    __ubuf__ int32_t* runEFP = vfBase + EngramVfTables::RUNEF;
    __ubuf__ int32_t* dupSFP = vfBase + EngramVfTables::DUPSF;
    __ubuf__ int32_t* dupEFP = vfBase + EngramVfTables::DUPEF;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t>();
    AscendC::Reg::RegTensor<int32_t> zeroReg;
    AscendC::Reg::Duplicate(zeroReg, (int32_t)0, allMask);
    for (uint32_t c = 0U; c < chunks; c++) {
        const uint32_t base = c * VF_CHUNK_INT32;
        AscendC::Reg::RegTensor<int32_t> posReg;
        AscendC::Reg::Arange<int32_t>(posReg, (int32_t)0);
        if (base != 0U) {
            AscendC::Reg::Adds(posReg, posReg, static_cast<int32_t>(base), allMask);
        }
        AscendC::Reg::RegTensor<int32_t> pos1Reg;
        AscendC::Reg::Adds(pos1Reg, posReg, (int32_t)1, allMask);
        AscendC::Reg::RegTensor<int32_t> ridReg;
        AscendC::Reg::RegTensor<int32_t> gidReg;
        AscendC::Reg::RegTensor<int32_t> preReg;
        AscendC::Reg::LoadAlign(ridReg, ridP + base);
        AscendC::Reg::LoadAlign(gidReg, gidP + base);
        AscendC::Reg::LoadAlign(preReg, prefixP + base);
        AscendC::Reg::RegTensor<int32_t> ridIdx;
        AscendC::Reg::RegTensor<int32_t> gidIdx;
        AscendC::Reg::RegTensor<int32_t> cmpVal;
        AscendC::Reg::RegTensor<int32_t> ridIdxC;
        AscendC::Reg::RegTensor<int32_t> gidIdxC;
        AscendC::Reg::Adds(ridIdx, ridReg, (int32_t)-1, allMask);
        AscendC::Reg::Adds(gidIdx, gidReg, (int32_t)-1, allMask);
        AscendC::Reg::Adds(cmpVal, preReg, runningOffset - 1, allMask);
        AscendC::Reg::MaskReg ridPosMask, gidPosMask;
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(ridPosMask, ridReg, (int32_t)0, allMask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(gidPosMask, gidReg, (int32_t)0, allMask);
        AscendC::Reg::Select(ridIdxC, ridIdx, zeroReg, ridPosMask);
        AscendC::Reg::Select(gidIdxC, gidIdx, zeroReg, gidPosMask);
        AscendC::Reg::RegTensor<int32_t> runSFReg;
        AscendC::Reg::RegTensor<int32_t> runEFReg;
        AscendC::Reg::RegTensor<int32_t> dupSFReg;
        AscendC::Reg::RegTensor<int32_t> dupEFReg;
        AscendC::Reg::LoadAlign(runSFReg, runSFP + base);
        AscendC::Reg::LoadAlign(runEFReg, runEFP + base);
        AscendC::Reg::LoadAlign(dupSFReg, dupSFP + base);
        AscendC::Reg::LoadAlign(dupEFReg, dupEFP + base);
        AscendC::Reg::MaskReg runSFMask, runEFMask, dupSFMask, dupEFMask;
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(runSFMask, runSFReg, (int32_t)0, allMask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(runEFMask, runEFReg, (int32_t)0, allMask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(dupSFMask, dupSFReg, (int32_t)0, allMask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(dupEFMask, dupEFReg, (int32_t)0, allMask);
        AscendC::Reg::Scatter<int32_t>(runStartT, posReg, (AscendC::Reg::RegTensor<uint32_t>&)ridIdxC, runSFMask);
        AscendC::Reg::Scatter<int32_t>(runEndT, pos1Reg, (AscendC::Reg::RegTensor<uint32_t>&)ridIdxC, runEFMask);
        AscendC::Reg::Scatter<int32_t>(runCompactT, cmpVal, (AscendC::Reg::RegTensor<uint32_t>&)ridIdxC, runSFMask);
        AscendC::Reg::Scatter<int32_t>(dupStartT, posReg, (AscendC::Reg::RegTensor<uint32_t>&)gidIdxC, dupSFMask);
        AscendC::Reg::Scatter<int32_t>(dupEndT, pos1Reg, (AscendC::Reg::RegTensor<uint32_t>&)gidIdxC, dupEFMask);
        AscendC::Reg::Scatter<int32_t>(dupCompactT, cmpVal, (AscendC::Reg::RegTensor<uint32_t>&)gidIdxC, dupSFMask);
    }
}

// 主管线：nn 生成 → isnew 播种/移位 → prefix 扫描 → unique 散射 → run/dup 标记 → 双扫描 → 六表散射
__simd_vf__ __aicore__ inline void EngramVfBatchUniqueVF(__ubuf__ int32_t* vfBase, __ubuf__ int32_t* entry,
                                                         __ubuf__ int32_t* uniqueT, __ubuf__ int32_t* runStartT,
                                                         __ubuf__ int32_t* runEndT, __ubuf__ int32_t* runCompactT,
                                                         __ubuf__ int32_t* dupStartT, __ubuf__ int32_t* dupEndT,
                                                         __ubuf__ int32_t* dupCompactT, uint32_t b, int32_t prevEntry,
                                                         uint32_t firstElem, int32_t rankBase, int32_t runningOffset)
{
    __ubuf__ int32_t* isnewP = vfBase + EngramVfTables::ISNEW;
    __ubuf__ int32_t* scanPad = vfBase + EngramVfTables::SCAN_PAD;
    __ubuf__ int32_t* scanDat = vfBase + EngramVfTables::SCAN_DAT;
    __ubuf__ int32_t* prefixP = vfBase + EngramVfTables::PREFIX;
    __ubuf__ int32_t* runSFP = vfBase + EngramVfTables::RUNSF;
    __ubuf__ int32_t* ridP = vfBase + EngramVfTables::RID;
    __ubuf__ int32_t* dupSFP = vfBase + EngramVfTables::DUPSF;
    __ubuf__ int32_t* gidP = vfBase + EngramVfTables::GID;

    const uint32_t chunks = (b + VF_CHUNK_INT32 - 1U) / VF_CHUNK_INT32;

    EngramVfFillNewNext(entry, vfBase, b, chunks);

    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    EngramVfSeedIsNewHead(entry, vfBase, chunks, firstElem, prevEntry);

    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    {
        AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<int32_t>();
        AscendC::Reg::RegTensor<int32_t> zeroReg;
        AscendC::Reg::Duplicate(zeroReg, (int32_t)0, allMask);
        uint32_t padLen = 32U;
        AscendC::Reg::MaskReg padMask = AscendC::Reg::UpdateMask<int32_t>(padLen);
        AscendC::Reg::StoreAlign(scanPad, zeroReg, padMask);
    }
    EngramVfScanTable(isnewP, prefixP, scanDat, b);

    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    EngramVfScatterUnique(entry, vfBase, uniqueT, chunks, rankBase);

    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    EngramVfBuildRunFlags(vfBase, chunks);
    EngramVfBuildDupFlags(vfBase, b, chunks);

    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    EngramVfScanTable(runSFP, ridP, scanDat, b);
    EngramVfScanTable(dupSFP, gidP, scanDat, b);

    AscendC::Reg::LocalMemBar<AscendC::Reg::MemType::VEC_STORE, AscendC::Reg::MemType::VEC_LOAD>();
    EngramVfScatterRunDup(vfBase, runStartT, runEndT, runCompactT, dupStartT, dupEndT, dupCompactT, chunks,
                          runningOffset);
}

class EngramFetchGradUnique {
public:
    __aicore__ inline EngramFetchGradUnique() = default;

    __aicore__ inline void Init(uint32_t aivId, uint32_t totalBlocks, uint32_t rankId, uint32_t numRanks,
                                int32_t numEntriesPerRank, int64_t hiddenDim, int64_t hiddenBytes, int32_t inputDtype,
                                int32_t outputDtype, AscendC::TPipe* pipe, AscendC::TBuf<>& entryBuf,
                                AscendC::TBuf<>& gradBuf, AscendC::TBuf<>& indicesBuf, AscendC::TBuf<>& tempBuf,
                                AscendC::TBuf<>& statusBuf, AscendC::TBuf<>& castBuf, AscendC::TBuf<>& accumBuf);

    __aicore__ inline void ZeroGradUnique(uint32_t numRecv, GM_ADDR gradUniqueOutGM);

    __aicore__ inline void CountUniquesParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR coreStartGM,
                                                GM_ADDR segCountGM);
    __aicore__ inline void RunScatterCast(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM,
                                          GM_ADDR numUniqueOutGM, GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM,
                                          GM_ADDR coreStartGM, GM_ADDR segCountGM, GM_ADDR sortCompanionGM);

    __aicore__ inline void WriteNumUniqueZero(GM_ADDR numUniqueOutGM);

    __aicore__ inline void SetCastBuf(AscendC::TBuf<>& castBuf)
    {
        castBuf_ = &castBuf;
    }
    __aicore__ inline void SetAccumBuf(AscendC::TBuf<>& accumBuf)
    {
        accumBuf_ = &accumBuf;
    }
    __aicore__ inline void SetGradSubBatch(uint32_t batch)
    {
        gradSubBatch_ = batch;
    }
    __aicore__ inline void SetEntryBufBytes(uint32_t bytes)
    {
        entryBufBytes_ = bytes;
    }
    __aicore__ inline void SetChunkElems(uint32_t elems)
    {
        chunkElems_ = elems;
        if (chunkElems_ > 0U) {
            uint32_t inSize = GetDtypeSize(inputDtype_);
            uint32_t outSize = GetDtypeSize(outputDtype_);
            chunkInStride_ =
                (chunkElems_ * inSize + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
            chunkOutStride_ =
                (chunkElems_ * outSize + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
            chunkFp32Stride_ =
                (chunkElems_ * sizeof(float) + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
            numChunks_ = (static_cast<uint32_t>(hiddenDim_) + chunkElems_ - 1U) / chunkElems_;
            accumFloats_ = chunkFp32Stride_ / sizeof(float);
        } else {
            numChunks_ = 0U;
        }
    }

    // entryBuf_ int32 slot layout: one ENTRY_BATCH_CAP-sized slot per array.
    __aicore__ inline AscendC::LocalTensor<int32_t> CompUb()
    {
        return entryBuf_->Get<int32_t>();
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> UniqueUb()
    {
        return entryBuf_->Get<int32_t>()[Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> CompactIdxUb()
    {
        return entryBuf_->Get<int32_t>()[COMPACT_INDEX_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> DirectFlagUb()
    {
        return entryBuf_->Get<int32_t>()[DIRECT_FLAG_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> RunStartUb()
    {
        return entryBuf_->Get<int32_t>()[RUN_START_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> RunLenUb()
    {
        return entryBuf_->Get<int32_t>()[RUN_LEN_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP +
                                         Mc2Kernel::ENTRY_BATCH_CAP / ENTRY_BATCH_CAP_HALF];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> AccumListUb()
    {
        return entryBuf_->Get<int32_t>()[ACCUM_LIST_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> DupStartUb()
    {
        return entryBuf_->Get<int32_t>()[DUP_START_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> DupCompactUb()
    {
        return entryBuf_->Get<int32_t>()[DUP_START_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP +
                                         Mc2Kernel::ENTRY_BATCH_CAP / ENTRY_BATCH_CAP_HALF];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> DupEndUb()
    {
        return entryBuf_->Get<int32_t>()[DUP_END_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP];
    }
    __aicore__ inline AscendC::LocalTensor<int32_t> RunCompactUb()
    {
        return entryBuf_->Get<int32_t>()[RUN_COMPACT_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP +
                                         Mc2Kernel::ENTRY_BATCH_CAP / ENTRY_BATCH_CAP_HALF];
    }

private:
    __aicore__ inline void ScatterAccumulateParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                     GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
                                                     GM_ADDR recvGradGM, GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                     GM_ADDR sortCompanionGM);
    __aicore__ inline void WriteNumUnique(GM_ADDR segCountGM, GM_ADDR numUniqueOutGM);
    __aicore__ inline void LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM, GM_ADDR segCountGM, uint32_t& start,
                                         uint32_t& end, int32_t& preCoreOffset);
    __aicore__ inline uint32_t ProcessScatterBatch(uint32_t cur, uint32_t end, int32_t& runningOffset,
                                                   int32_t& runningUniqueOffset, int32_t& prevEntry,
                                                   bool& isFirstElement, GM_ADDR recvLocalEntryOutGM,
                                                   GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
                                                   GM_ADDR recvGradGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void ChunkedScatterRange(uint32_t start, uint32_t end, int32_t preCoreOffset,
                                               GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM,
                                               GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void ChunkColumnPass(uint32_t chunkIdx, uint32_t start, uint32_t end, int32_t preCoreOffset,
                                           GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM,
                                           GM_ADDR recvGradGM, GM_ADDR sortCompanionGM, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void ProcessChunkTile(uint32_t subStart, uint32_t subLen, uint32_t chunkIdx,
                                            uint32_t chunkSrcElems, AscendC::LocalTensor<uint8_t>& gradRaw,
                                            uint32_t& accumCursor, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void ProcessAccumRowChunk(int32_t pos, uint32_t subStart, uint32_t chunkIdx,
                                                uint32_t chunkSrcElems, AscendC::LocalTensor<uint8_t>& gradRaw,
                                                AscendC::LocalTensor<float>& castChunk, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void FlushAccumChunk(uint32_t chunkIdx, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void RunScatterTilesV1(uint32_t batchLen, uint32_t maxGradPerBatch, bool singleRowMode,
                                             bool canDirectCopy, AscendC::LocalTensor<uint8_t>& gradPing,
                                             GM_ADDR recvGradGM, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void LoadGradRowsSubBatch(AscendC::LocalTensor<uint8_t>& gradRaw, uint32_t subStart,
                                                uint32_t subLen, GM_ADDR recvGradGM);
    __aicore__ inline void AccumulateScatterSubBatch(AscendC::LocalTensor<uint8_t>& gradRaw, uint32_t bufIdx,
                                                     uint32_t subStart, uint32_t subLen, bool canDirectCopy,
                                                     uint32_t maxGradPerBatch, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void FlushDirectRunsFromFlags(AscendC::LocalTensor<uint8_t>& gradRaw, uint32_t subStart,
                                                    uint32_t subLen, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void DrainScatterEvtV1(uint32_t tileIdx, bool singleRowMode, bool canDirectCopy,
                                             event_t evtMte2V_0, event_t evtMte2V_1, event_t evtVMte2_0,
                                             event_t evtVMte2_1, event_t evtMte2Mte3_0, event_t evtMte2Mte3_1,
                                             event_t evtMte3Mte2_0, event_t evtMte3Mte2_1);
    __aicore__ inline uint32_t FindBatchStartBoundary(const AscendC::GlobalTensor<int32_t>& sortedEntryGM,
                                                      uint32_t rawStart, uint32_t rawEnd);
    __aicore__ inline uint32_t CountLocalUniquesInRange(const AscendC::GlobalTensor<int32_t>& sortedEntryGM,
                                                        uint32_t start, uint32_t rawEnd);
    __aicore__ inline void WriteSegCountStatus(uint32_t start, uint32_t localUniqueCount, GM_ADDR coreStartGM,
                                               GM_ADDR segCountGM);
    __aicore__ inline void AllocChunkColEvents(event_t* evtMte2V, event_t* evtVMte2, event_t* evtMte2Mte3,
                                               event_t* evtMte3Mte2, uint32_t evtBufs);
    __aicore__ inline void LoadPrevEntryBoundary(GM_ADDR recvLocalEntryOutGM, uint32_t start, int32_t& prevEntry,
                                                 bool& isFirstElement);
    __aicore__ inline void RunChunkSubTiles(uint32_t batchLen, uint32_t maxRows, uint32_t chunkIdx,
                                            uint32_t chunkSrcElems, uint32_t chunkSrcBytes, uint64_t srcByteOff,
                                            AscendC::LocalTensor<uint8_t>& gradBase, uint32_t& tileIdx,
                                            uint32_t& accumCursor, GM_ADDR recvGradGM, GM_ADDR gradUniqueOutGM,
                                            event_t* evtMte2V, event_t* evtVMte2, event_t* evtMte2Mte3,
                                            event_t* evtMte3Mte2, uint32_t evtBufs);
    __aicore__ inline void DrainChunkColEvents(uint32_t tileIdx, event_t* evtMte2V, event_t* evtVMte2,
                                               event_t* evtMte2Mte3, event_t* evtMte3Mte2, uint32_t evtBufs);
    __aicore__ inline void ClampCoreRange(uint32_t& start, uint32_t& end, uint32_t numRecv);
    __aicore__ inline void FlushAccumFp32(AscendC::LocalTensor<float>& accum, uint32_t bufIdx, uint64_t gmByteOffset,
                                          GM_ADDR gradUniqueOutGM);
    __aicore__ inline AscendC::LocalTensor<uint8_t> CastAccumToStaging(AscendC::LocalTensor<float>& accum,
                                                                       uint32_t bufIdx);
    __aicore__ inline void CopyStagedAccumOut(AscendC::LocalTensor<uint8_t>& flushCastBuf, uint32_t bufIdx,
                                              uint64_t gmByteOffset, GM_ADDR gradUniqueOutGM);
    __aicore__ inline uint32_t ProcessBatchUnique(uint32_t cur, uint32_t batchLen, int32_t runningOffset,
                                                  int32_t& prevEntry, bool& isFirstElement, int32_t& inclusiveSum,
                                                  GM_ADDR recvLocalEntryOutGM, GM_ADDR sortCompanionGM, bool emitLists,
                                                  bool sparseCompact);
    __aicore__ inline uint32_t ProcessBatchUniqueVF(uint32_t cur, uint32_t batchLen, int32_t runningOffset,
                                                    int32_t& prevEntry, bool& isFirstElement, int32_t& inclusiveSum,
                                                    GM_ADDR recvLocalEntryOutGM, GM_ADDR sortCompanionGM);
    __aicore__ inline void WalkBatchUniqueRuns(const AscendC::LocalTensor<int32_t>& entryUb, uint32_t batchLen,
                                               int32_t runningOffset, int32_t rankBase, bool& isFirstElement,
                                               int32_t& prevEntry, int32_t& inclusiveSum, uint32_t& tileUniqueCnt,
                                               uint32_t& runCnt, uint32_t& accumCnt, bool& inRun,
                                               uint32_t& runStartPos);
    __aicore__ inline void CloseBatchUniqueRuns(uint32_t batchLen, uint32_t& runCnt, uint32_t& accumCnt, bool& inRun,
                                                uint32_t& runStartPos);
    __aicore__ inline void WalkBatchUniqueDirectFlags(const AscendC::LocalTensor<int32_t>& entryUb, uint32_t batchLen,
                                                      int32_t runningOffset, bool& isFirstElement, int32_t& prevEntry,
                                                      int32_t& inclusiveSum, uint32_t& tileUniqueCnt);
    __aicore__ inline void WriteBatchUnique(uint32_t tileUniqueCnt, int32_t& runningUniqueOffset,
                                            GM_ADDR uniqueLocalEntryOutGM);

    __aicore__ inline void FlushAccum(GM_ADDR gradUniqueOutGM);
    __aicore__ inline void AccumulateSubBatch(AscendC::LocalTensor<float>& gradFp32, uint32_t rowStrideFloats,
                                              uint32_t subStart, uint32_t subLen, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void AccumulateDirectSubBatch(AscendC::LocalTensor<uint8_t>& gradRaw, uint32_t subStart,
                                                    uint32_t subLen, AscendC::LocalTensor<float>& castRow,
                                                    GM_ADDR gradUniqueOutGM);
    __aicore__ inline void ProcessNonDirectRow(int32_t pos, uint32_t subStart, AscendC::LocalTensor<uint8_t>& gradRaw,
                                               AscendC::LocalTensor<float>& castRow, GM_ADDR gradUniqueOutGM,
                                               int32_t compactIdx);
    __aicore__ inline void AccumulateDupListWalk(uint32_t subStart, uint32_t subLen, uint32_t& cursor,
                                                 AscendC::LocalTensor<uint8_t>& gradRaw,
                                                 AscendC::LocalTensor<float>& castRow, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void EmitRunsFromDupList(uint32_t subStart, uint32_t subLen, uint32_t& cursor,
                                               AscendC::LocalTensor<uint8_t>& gradRaw, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void ProcessDirectSubBatchLoopV2(uint32_t batchLen, uint32_t maxGradPerBatch,
                                                       AscendC::LocalTensor<uint8_t>& gradBase, GM_ADDR recvGradGM,
                                                       GM_ADDR gradUniqueOutGM);
    __aicore__ inline void FlushDirectRun(AscendC::LocalTensor<uint8_t>& gradRaw, uint32_t runStart, uint32_t runLen,
                                          int32_t compactFirst, GM_ADDR gradUniqueOutGM);
    __aicore__ inline void CastToFP32(AscendC::LocalTensor<float> outT, AscendC::LocalTensor<uint8_t> gradRaw,
                                      uint32_t count);

    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};
    int32_t numEntriesPerRank_{0};
    int64_t hiddenDim_{0};
    int64_t hiddenBytes_{0};
    uint32_t inRowStride_{0};   // GM 行字节数向 32B 上取整（UB 内行排布 stride）
    uint32_t fp32RowStride_{0}; // fp32 行字节数向 32B 上取整
    uint32_t chunkElems_{0};
    uint32_t chunkInStride_{0};
    uint32_t chunkOutStride_{0};
    uint32_t chunkFp32Stride_{0};
    uint32_t numChunks_{0};
    uint32_t numRecv_{0};
    int32_t inputDtype_{0};
    int32_t outputDtype_{0};
    AscendC::TPipe* pipe_{nullptr};
    AscendC::TBuf<>* entryBuf_{nullptr};
    AscendC::TBuf<>* gradBuf_{nullptr};
    AscendC::TBuf<>* indicesBuf_{nullptr};
    AscendC::TBuf<>* tempBuf_{nullptr};
    AscendC::TBuf<>* statusBuf_{nullptr};
    AscendC::TBuf<>* castBuf_{nullptr};
    AscendC::TBuf<>* accumBuf_{nullptr};
    uint32_t gradSubBatch_{Mc2Kernel::GRAD_SUB_BATCH};
    uint32_t entryBufBytes_{Mc2Kernel::ENTRY_BUF_BYTES};

    int32_t myPreCoreOffset_{0};
    int32_t mySegCount_{0};

    int32_t accumCompactIdx_{-1};
    bool accumDirty_{false};
    uint32_t accumFloats_{0};
    uint32_t accumIdx_{0};
    bool flushPending_[2]{false, false};
    int32_t flushEvtVMte3Arr_[2]{0, 0};
    int32_t flushEvtMte3VArr_[2]{0, 0};
    uint32_t runCnt_{0};   // 本批 direct-run 数
    uint32_t accumCnt_{0}; // 本批 accum 行数(≤ENTRY_BATCH_CAP)
    uint32_t dupCnt_{0};
    uint32_t accumRowCnt_{0}; // 当前 accum 组已累加行数
};

__aicore__ inline void EngramFetchGradUnique::Init(uint32_t aivId, uint32_t totalBlocks, uint32_t rankId,
                                                   uint32_t numRanks, int32_t numEntriesPerRank, int64_t hiddenDim,
                                                   int64_t hiddenBytes, int32_t inputDtype, int32_t outputDtype,
                                                   AscendC::TPipe* pipe, AscendC::TBuf<>& entryBuf,
                                                   AscendC::TBuf<>& gradBuf, AscendC::TBuf<>& indicesBuf,
                                                   AscendC::TBuf<>& tempBuf, AscendC::TBuf<>& statusBuf,
                                                   AscendC::TBuf<>& castBuf, AscendC::TBuf<>& accumBuf)
{
    aivId_ = aivId;
    totalBlocks_ = totalBlocks;
    rankId_ = rankId;
    numRanks_ = numRanks;
    numEntriesPerRank_ = numEntriesPerRank;
    hiddenDim_ = hiddenDim;
    hiddenBytes_ = hiddenBytes;
    inRowStride_ =
        (static_cast<uint32_t>(hiddenBytes) + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
    fp32RowStride_ = (static_cast<uint32_t>(hiddenDim) * sizeof(float) + Mc2Kernel::UB_ALIGN - 1U) /
                     Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
    inputDtype_ = inputDtype;
    outputDtype_ = outputDtype;
    pipe_ = pipe;
    entryBuf_ = &entryBuf;
    gradBuf_ = &gradBuf;
    indicesBuf_ = &indicesBuf;
    tempBuf_ = &tempBuf;
    statusBuf_ = &statusBuf;
    castBuf_ = &castBuf;
    accumBuf_ = &accumBuf;
    accumFloats_ = (static_cast<uint32_t>(hiddenDim) * sizeof(float) + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN *
                   Mc2Kernel::UB_ALIGN / sizeof(float);
}

__aicore__ inline void EngramFetchGradUnique::RunScatterCast(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                             GM_ADDR uniqueLocalEntryOutGM, GM_ADDR numUniqueOutGM,
                                                             GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM,
                                                             GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                             GM_ADDR sortCompanionGM)
{
    flushEvtVMte3Arr_[0] = static_cast<int32_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>());
    flushEvtVMte3Arr_[1] = static_cast<int32_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE3>());
    flushEvtMte3VArr_[0] = static_cast<int32_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>());
    flushEvtMte3VArr_[1] = static_cast<int32_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE3_V>());
    numRecv_ = numRecv;
    accumIdx_ = 0;
    flushPending_[0] = false;
    flushPending_[1] = false;

    ScatterAccumulateParallel(numRecv, recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM, recvGradGM,
                              coreStartGM, segCountGM, sortCompanionGM);

    if (accumDirty_) {
        FlushAccum(gradUniqueOutGM);
        accumDirty_ = false;
    }

    for (uint32_t i = 0; i < 2U; i++) {
        if (flushPending_[i]) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[i]);
            flushPending_[i] = false;
        }
    }

    WriteNumUnique(segCountGM, numUniqueOutGM);

    pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[0]);
    pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[1]);
    pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[0]);
    pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[1]);
}

__aicore__ inline void EngramFetchGradUnique::WriteNumUniqueZero(GM_ADDR numUniqueOutGM)
{
    if (aivId_ != 0) {
        AscendC::SyncAll<true>();
        return;
    }
    AscendC::GlobalTensor<int32_t> numUniqueGM;
    numUniqueGM.SetGlobalBuffer((__gm__ int32_t*)numUniqueOutGM);
    AscendC::LocalTensor<int32_t> tmp = statusBuf_->Get<int32_t>();
    tmp.SetValue(0, 0);
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::DataCopyParams params = {1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(numUniqueGM, tmp, params);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
    AscendC::SyncAll<true>();
}

__aicore__ inline void EngramFetchGradUnique::ZeroGradUnique(uint32_t numRecv, GM_ADDR gradUniqueOutGM)
{
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint64_t totalBytes = static_cast<uint64_t>(numRecv) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize;
    uint64_t chunk = (totalBytes + totalBlocks_ - 1U) / totalBlocks_;
    uint64_t start = static_cast<uint64_t>(aivId_) * chunk;
    uint64_t end = start + chunk;
    if (end > totalBytes) {
        end = totalBytes;
    }

    if (start < end) {
        AscendC::LocalTensor<float> zeroBuf = gradBuf_->Get<float>();
        uint32_t tileFloats = Mc2Kernel::TILE_BYTES / sizeof(float);
        constexpr uint32_t maxBlockFloats = Mc2Kernel::MAX_BLOCK_BYTES / sizeof(float);
        if (tileFloats > maxBlockFloats) {
            tileFloats = maxBlockFloats;
        }
        AscendC::Duplicate<float>(zeroBuf, 0.0f, tileFloats);
        SyncFunc<AscendC::HardEvent::V_MTE3>(*pipe_);

        // Chunk stride must equal the zeroed span (tileFloats*4 <= 65532): DataCopyParams
        // blockLen is uint16_t, and TILE_BYTES(64KB) as a full chunk wraps to 0 in the
        // uint16 cast -> the zero pass silently wrote nothing for every full 64KB chunk
        // (regression from the TILE_BYTES 32K->64K change; masked only by pre-zeroed
        // output buffers, exposed by generalization cases).
        uint32_t chunkBytes = tileFloats * sizeof(float);
        AscendC::GlobalTensor<uint8_t> dstGM;
        dstGM.SetGlobalBuffer((__gm__ uint8_t*)gradUniqueOutGM);
        for (uint64_t off = start; off < end; off += chunkBytes) {
            uint64_t thisLen = end - off;
            if (thisLen > chunkBytes) {
                thisLen = chunkBytes;
            }
            AscendC::DataCopyParams params{1U, static_cast<uint16_t>(thisLen), 0U, 0U};
            AscendC::DataCopyPad(dstGM[off], zeroBuf.ReinterpretCast<uint8_t>(), params);
        }
    }
}

// 批首边界探测：从 rawStart-1 起向前找第一个与前值不同的元素，返回本核计数起点
__aicore__ inline uint32_t EngramFetchGradUnique::FindBatchStartBoundary(
    const AscendC::GlobalTensor<int32_t>& sortedEntryGM, uint32_t rawStart, uint32_t rawEnd)
{
    AscendC::LocalTensor<int32_t> probeUb = tempBuf_->Get<int32_t>();
    AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    uint32_t probeBase = rawStart - 1U;
    uint32_t start = rawEnd;
    while (probeBase < rawEnd) {
        uint32_t probeLen = rawEnd - probeBase;
        if (probeLen > Mc2Kernel::ENTRY_BATCH_CAP) {
            probeLen = Mc2Kernel::ENTRY_BATCH_CAP;
        }
        AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(probeLen * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(probeUb, sortedEntryGM[probeBase], cpParams, cpPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

        int32_t boundaryEntry = probeUb.GetValue(0);
        uint32_t p = 1U;
        while (p < probeLen && probeUb.GetValue(p) == boundaryEntry) {
            p++;
        }
        if (p < probeLen) {
            start = probeBase + p;
            break;
        }
        if (probeBase + probeLen >= rawEnd) {
            break;
        }
        probeBase += probeLen - 1U;
    }
    return start;
}

// 区间逐批计数 unique（Sub/Mins/ReduceSum 向量归约，批间衔接首元素特判）
__aicore__ inline uint32_t EngramFetchGradUnique::CountLocalUniquesInRange(
    const AscendC::GlobalTensor<int32_t>& sortedEntryGM, uint32_t start, uint32_t rawEnd)
{
    // 注意: V算子(Sub/ReduceSum)的UB操作数必须32B对齐(底层LoadAlign), 槽位偏移勿做逐元素错位视图
    AscendC::LocalTensor<int32_t> xUb = gradBuf_->Get<int32_t>();
    AscendC::LocalTensor<int32_t> yUb = xUb[Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::LocalTensor<int32_t> fUb = xUb[UNIQUE_DIFF_UB_OFFSET * Mc2Kernel::ENTRY_BATCH_CAP];
    AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    uint32_t localUniqueCount = 0;
    uint32_t cur = start;
    while (cur < rawEnd) {
        uint32_t batchLen = rawEnd - cur;
        if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
            batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
        }
        AscendC::DataCopyExtParams yParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(yUb, sortedEntryGM[cur], yParams, cpPad);
        bool isFirstBatch = (cur == 0U);
        uint32_t xBase = isFirstBatch ? (cur + 1U) : (cur - 1U);
        AscendC::DataCopyExtParams cpX{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(xUb, sortedEntryGM[xBase], cpX, cpPad);
        SyncFunc<AscendC::HardEvent::MTE2_V>(*pipe_);

        uint32_t cntLen = isFirstBatch ? (batchLen - 1U) : batchLen;
        if (cntLen > 0U) {
            if (isFirstBatch) {
                AscendC::Sub<int32_t>(fUb, xUb, yUb, static_cast<int32_t>(cntLen));
            } else {
                AscendC::Sub<int32_t>(fUb, yUb, xUb, static_cast<int32_t>(cntLen));
            }
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins<int32_t>(fUb, fUb, 1, static_cast<int32_t>(cntLen));
            AscendC::PipeBarrier<PIPE_V>();

            const uint32_t rshape[] = {1U, cntLen};
            AscendC::LocalTensor<uint8_t> rsTmp = tempBuf_->Get<uint8_t>();
            AscendC::ReduceSum<int32_t, AscendC::Pattern::Reduce::AR>(xUb, fUb, rsTmp, rshape, false);
            SyncFunc<AscendC::HardEvent::V_S>(*pipe_);
            localUniqueCount += static_cast<uint32_t>(xUb.GetValue(0));
        }
        if (isFirstBatch) {
            localUniqueCount += 1U;
        }
        cur += batchLen;
    }
    return localUniqueCount;
}

// 计数结果落 coreStart/segCount 两 GM 槽并缓存 mySegCount_
__aicore__ inline void EngramFetchGradUnique::WriteSegCountStatus(uint32_t start, uint32_t localUniqueCount,
                                                                  GM_ADDR coreStartGM, GM_ADDR segCountGM)
{
    AscendC::LocalTensor<int32_t> cntUb = statusBuf_->Get<int32_t>();
    cntUb.SetValue(0, static_cast<int32_t>(start));
    cntUb.SetValue(Mc2Kernel::STATE_OFFSET / sizeof(int32_t), static_cast<int32_t>(localUniqueCount));
    mySegCount_ = static_cast<int32_t>(localUniqueCount);
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::GlobalTensor<int32_t> coreStartGMT;
    coreStartGMT.SetGlobalBuffer((__gm__ int32_t*)coreStartGM);
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t*)segCountGM);
    AscendC::DataCopyParams cntParams{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(coreStartGMT[aivId_], cntUb, cntParams);
    AscendC::DataCopyPad(segCountGMT[aivId_], cntUb[Mc2Kernel::STATE_OFFSET / sizeof(int32_t)], cntParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
}

__aicore__ inline void EngramFetchGradUnique::CountUniquesParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                                   GM_ADDR coreStartGM, GM_ADDR segCountGM)
{
    uint32_t chunk = (numRecv + totalBlocks_ - 1U) / totalBlocks_;
    uint32_t rawStart = aivId_ * chunk;
    uint32_t rawEnd = rawStart + chunk;
    if (rawEnd > numRecv) {
        rawEnd = numRecv;
    }

    AscendC::GlobalTensor<int32_t> sortedEntryGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t*)recvLocalEntryOutGM);

    uint32_t start = rawStart;
    if (rawStart > 0 && rawStart < rawEnd) {
        start = FindBatchStartBoundary(sortedEntryGM, rawStart, rawEnd);
    }

    uint32_t localUniqueCount = 0;
    if (start < rawEnd) {
        localUniqueCount = CountLocalUniquesInRange(sortedEntryGM, start, rawEnd);
    }

    WriteSegCountStatus(start, localUniqueCount, coreStartGM, segCountGM);
}
__aicore__ inline void EngramFetchGradUnique::ChunkedScatterRange(uint32_t start, uint32_t end, int32_t preCoreOffset,
                                                                  GM_ADDR recvLocalEntryOutGM,
                                                                  GM_ADDR uniqueLocalEntryOutGM,
                                                                  GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM,
                                                                  GM_ADDR sortCompanionGM)
{
    for (uint32_t c = 0U; c < numChunks_; c++) {
        accumCompactIdx_ = -1;
        accumDirty_ = false;
        accumRowCnt_ = 0U;
        ChunkColumnPass(c, start, end, preCoreOffset, recvLocalEntryOutGM, uniqueLocalEntryOutGM, recvGradGM,
                        sortCompanionGM, gradUniqueOutGM);
        if (accumDirty_) {
            FlushAccumChunk(c, gradUniqueOutGM);
            accumDirty_ = false;
        }
    }
}
// 分块列通道：预取 4 组 MTE2/V/MTE3 事件
__aicore__ inline void EngramFetchGradUnique::AllocChunkColEvents(event_t* evtMte2V, event_t* evtVMte2,
                                                                  event_t* evtMte2Mte3, event_t* evtMte3Mte2,
                                                                  uint32_t evtBufs)
{
    // evtBufs 同时是 UB 槽数与 4 元素事件数组的写上界：越界会溢出栈数组/越过分段容量
    ascendc_assert(evtBufs > 0U && evtBufs <= CHUNK_EVT_BUFS, "evtBufs out of range: %u", evtBufs);
    for (uint32_t evtBufIdx = 0U; evtBufIdx < evtBufs; evtBufIdx++) {
        evtMte2V[evtBufIdx] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
        evtVMte2[evtBufIdx] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
        evtMte2Mte3[evtBufIdx] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
        evtMte3Mte2[evtBufIdx] = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    }
}

// 分块列通道：读 start-1 处 entry 作为批间衔接的 prevEntry
__aicore__ inline void EngramFetchGradUnique::LoadPrevEntryBoundary(GM_ADDR recvLocalEntryOutGM, uint32_t start,
                                                                    int32_t& prevEntry, bool& isFirstElement)
{
    AscendC::GlobalTensor<int32_t> sortedEntryGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t*)recvLocalEntryOutGM);
    AscendC::LocalTensor<int32_t> prevUb = tempBuf_->Get<int32_t>();
    AscendC::DataCopyPadExtParams<int32_t> prevPad{false, 0, 0, 0};
    AscendC::DataCopyExtParams prevParams{1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(prevUb, sortedEntryGM[start - 1U], prevParams, prevPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
    prevEntry = prevUb.GetValue(0);
    isFirstElement = false;
}

// 分块列通道：一个批内的子批乒乓流水（grad 搬入 → ProcessChunkTile → 事件推进）
__aicore__ inline void EngramFetchGradUnique::RunChunkSubTiles(
    uint32_t batchLen, uint32_t maxRows, uint32_t chunkIdx, uint32_t chunkSrcElems, uint32_t chunkSrcBytes,
    uint64_t srcByteOff, AscendC::LocalTensor<uint8_t>& gradBase, uint32_t& tileIdx, uint32_t& accumCursor,
    GM_ADDR recvGradGM, GM_ADDR gradUniqueOutGM, event_t* evtMte2V, event_t* evtVMte2, event_t* evtMte2Mte3,
    event_t* evtMte3Mte2, uint32_t evtBufs)
{
    ascendc_assert(evtBufs > 0U && evtBufs <= CHUNK_EVT_BUFS, "evtBufs out of range: %u", evtBufs);
    constexpr uint32_t kBufBytes = Mc2Kernel::GRAD_PING_BYTES / 2U;
    AscendC::DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};
    for (uint32_t subStart = 0U; subStart < batchLen; subStart += maxRows) {
        uint32_t subLen = batchLen - subStart;
        if (subLen > maxRows) {
            subLen = maxRows;
        }
        uint32_t evtBufIdx = tileIdx % evtBufs;
        AscendC::LocalTensor<uint8_t> gradRaw = gradBase[evtBufIdx * kBufBytes];
        if (tileIdx >= evtBufs) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[evtBufIdx]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[evtBufIdx]);
        }
        for (uint32_t j = 0U; j < subLen; j++) {
            int32_t recvIdx = CompUb().GetValue(subStart + j);
            if (recvIdx < 0 || static_cast<uint32_t>(recvIdx) >= numRecv_) {
                recvIdx = 0;
            }
            GM_ADDR gradAddr = recvGradGM + static_cast<uint64_t>(recvIdx) * hiddenBytes_ + srcByteOff;
            AscendC::GlobalTensor<uint8_t> gradSrcGM;
            gradSrcGM.SetGlobalBuffer((__gm__ uint8_t*)gradAddr);
            AscendC::DataCopyExtParams gradParams{1U, chunkSrcBytes, 0U, 0U, 0U};
            AscendC::DataCopyPad(gradRaw[static_cast<uint64_t>(j) * chunkInStride_], gradSrcGM, gradParams, gradPad);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V[evtBufIdx]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[evtBufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V[evtBufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[evtBufIdx]);
        ProcessChunkTile(subStart, subLen, chunkIdx, chunkSrcElems, gradRaw, accumCursor, gradUniqueOutGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[evtBufIdx]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[evtBufIdx]);
        tileIdx++;
    }
}

// 分块列通道：排空未消费事件并归还全部事件 ID
__aicore__ inline void EngramFetchGradUnique::DrainChunkColEvents(uint32_t tileIdx, event_t* evtMte2V,
                                                                  event_t* evtVMte2, event_t* evtMte2Mte3,
                                                                  event_t* evtMte3Mte2, uint32_t evtBufs)
{
    ascendc_assert(evtBufs > 0U && evtBufs <= CHUNK_EVT_BUFS, "evtBufs out of range: %u", evtBufs);
    uint32_t drainFrom = (tileIdx > evtBufs) ? (tileIdx - evtBufs) : 0U;
    for (uint32_t t = drainFrom; t < tileIdx; t++) {
        uint32_t evtBufIdx = t % evtBufs;
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[evtBufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[evtBufIdx]);
    }
    for (uint32_t evtBufIdx = 0U; evtBufIdx < evtBufs; evtBufIdx++) {
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(evtMte2V[evtBufIdx]);
        pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(evtVMte2[evtBufIdx]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[evtBufIdx]);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[evtBufIdx]);
    }
}

__aicore__ inline void EngramFetchGradUnique::ChunkColumnPass(uint32_t chunkIdx, uint32_t start, uint32_t end,
                                                              int32_t preCoreOffset, GM_ADDR recvLocalEntryOutGM,
                                                              GM_ADDR uniqueLocalEntryOutGM, GM_ADDR recvGradGM,
                                                              GM_ADDR sortCompanionGM, GM_ADDR gradUniqueOutGM)
{
    uint32_t inDtypeSize = GetDtypeSize(inputDtype_);
    uint32_t chunkSrcElems =
        (chunkIdx + 1U == numChunks_) ? (static_cast<uint32_t>(hiddenDim_) - chunkIdx * chunkElems_) : chunkElems_;
    uint32_t chunkSrcBytes = chunkSrcElems * inDtypeSize;
    uint64_t srcByteOff = static_cast<uint64_t>(chunkIdx) * chunkElems_ * inDtypeSize;
    event_t evtMte2V[CHUNK_EVT_BUFS];
    event_t evtVMte2[CHUNK_EVT_BUFS];
    event_t evtMte2Mte3[CHUNK_EVT_BUFS];
    event_t evtMte3Mte2[CHUNK_EVT_BUFS];
    AllocChunkColEvents(evtMte2V, evtVMte2, evtMte2Mte3, evtMte3Mte2, CHUNK_EVT_BUFS);
    AscendC::LocalTensor<uint8_t> gradBase = gradBuf_->Get<uint8_t>();
    constexpr uint32_t kBufBytes = Mc2Kernel::GRAD_PING_BYTES / 2U;
    uint32_t bufRows = kBufBytes / chunkInStride_;
    if (bufRows < 1U) {
        bufRows = 1U;
    }
    uint32_t maxRows = bufRows;
    if (maxRows > gradSubBatch_) {
        maxRows = gradSubBatch_;
    }
    if (maxRows < 1U) {
        maxRows = 1U;
    }
    int32_t runningOffset = preCoreOffset;
    int32_t runningUniqueOffset = preCoreOffset;
    int32_t prevEntry = 0;
    bool isFirstElement = true;
    if (start > 0U) {
        LoadPrevEntryBoundary(recvLocalEntryOutGM, start, prevEntry, isFirstElement);
    }
    uint32_t tileIdx = 0U;
    uint32_t cur = start;
    while (cur < end) {
        uint32_t batchLen = end - cur;
        if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
            batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
        }
        int32_t inclusiveSum = 0;
        uint32_t tileUniqueCnt = ProcessBatchUnique(cur, batchLen, runningOffset, prevEntry, isFirstElement,
                                                    inclusiveSum, recvLocalEntryOutGM, sortCompanionGM, true, false);
        if (chunkIdx == 0U && tileUniqueCnt > 0) {
            int32_t uniqOff = runningUniqueOffset;
            WriteBatchUnique(tileUniqueCnt, uniqOff, uniqueLocalEntryOutGM);
            runningUniqueOffset = uniqOff;
        }
        runningOffset += inclusiveSum;
        uint32_t accumCursor = 0U;
        RunChunkSubTiles(batchLen, maxRows, chunkIdx, chunkSrcElems, chunkSrcBytes, srcByteOff, gradBase, tileIdx,
                         accumCursor, recvGradGM, gradUniqueOutGM, evtMte2V, evtVMte2, evtMte2Mte3, evtMte3Mte2,
                         CHUNK_EVT_BUFS);
        cur += batchLen;
    }
    DrainChunkColEvents(tileIdx, evtMte2V, evtVMte2, evtMte2Mte3, evtMte3Mte2, CHUNK_EVT_BUFS);
}
__aicore__ inline void EngramFetchGradUnique::ProcessChunkTile(uint32_t subStart, uint32_t subLen, uint32_t chunkIdx,
                                                               uint32_t chunkSrcElems,
                                                               AscendC::LocalTensor<uint8_t>& gradRaw,
                                                               uint32_t& accumCursor, GM_ADDR gradUniqueOutGM)
{
    bool directAllowed = (inputDtype_ == outputDtype_);
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint32_t inDtypeSize = GetDtypeSize(inputDtype_);
    uint32_t chunkSrcBytes = chunkSrcElems * inDtypeSize;
    uint64_t outByteOff = static_cast<uint64_t>(chunkIdx) * chunkElems_ * outDtypeSize;
    AscendC::LocalTensor<float> castChunk = castBuf_->Get<float>();
    for (uint32_t j = 0U; j < subLen; j++) {
        uint32_t pos = subStart + j;
        bool isAccum;
        if (!directAllowed) {
            isAccum = true;
        } else {
            while (accumCursor < accumCnt_ && static_cast<uint32_t>(AccumListUb().GetValue(accumCursor)) < pos) {
                accumCursor++;
            }
            isAccum = (accumCursor < accumCnt_ && static_cast<uint32_t>(AccumListUb().GetValue(accumCursor)) == pos);
        }
        if (isAccum) {
            ProcessAccumRowChunk(static_cast<int32_t>(pos), subStart, chunkIdx, chunkSrcElems, gradRaw, castChunk,
                                 gradUniqueOutGM);
        } else {
            int32_t compactIdx = CompactIdxUb().GetValue(pos);
            if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
                continue;
            }
            uint64_t gmByteOffset =
                static_cast<uint64_t>(compactIdx) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize + outByteOff;
            AscendC::GlobalTensor<uint8_t> dstGM;
            dstGM.SetGlobalBuffer((__gm__ uint8_t*)(gradUniqueOutGM + gmByteOffset));
            AscendC::DataCopyParams params{1U, static_cast<uint16_t>(chunkSrcBytes), 0U, 0U};
            AscendC::DataCopyPad(dstGM, gradRaw[j * chunkInStride_], params);
        }
    }
}
__aicore__ inline void EngramFetchGradUnique::ProcessAccumRowChunk(int32_t pos, uint32_t subStart, uint32_t chunkIdx,
                                                                   uint32_t chunkSrcElems,
                                                                   AscendC::LocalTensor<uint8_t>& gradRaw,
                                                                   AscendC::LocalTensor<float>& castChunk,
                                                                   GM_ADDR gradUniqueOutGM)
{
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();
    int32_t compactIdx = CompactIdxUb().GetValue(pos);
    if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
        return;
    }
    if (compactIdx != accumCompactIdx_) {
        if (accumDirty_) {
            FlushAccumChunk(chunkIdx, gradUniqueOutGM);
            accumIdx_ ^= 1U;
            if (flushPending_[accumIdx_]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[accumIdx_]);
                flushPending_[accumIdx_] = false;
            }
        }
        accumCompactIdx_ = compactIdx;
        accumRowCnt_ = 0;
    }
    AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
    uint32_t rowOff = static_cast<uint32_t>(pos - static_cast<int32_t>(subStart)) * chunkInStride_;
    if (accumRowCnt_ == 0U) {
        if (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) {
            CastToFP32(accum, gradRaw[rowOff], chunkSrcElems);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::LocalTensor<float> gradFp32 = gradRaw[rowOff].ReinterpretCast<float>();
            AscendC::Adds<float>(accum, gradFp32, 0.0f, chunkSrcElems);
            AscendC::PipeBarrier<PIPE_V>();
        }
    } else if (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) {
        CastToFP32(castChunk, gradRaw[rowOff], chunkSrcElems);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(accum, accum, castChunk, chunkSrcElems);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        AscendC::LocalTensor<float> gradFp32 = gradRaw[rowOff].ReinterpretCast<float>();
        AscendC::Add<float>(accum, accum, gradFp32, chunkSrcElems);
        AscendC::PipeBarrier<PIPE_V>();
    }
    accumRowCnt_++;
    accumDirty_ = true;
}
__aicore__ inline void EngramFetchGradUnique::FlushAccumChunk(uint32_t chunkIdx, GM_ADDR gradUniqueOutGM)
{
    uint32_t bufIdx = accumIdx_;
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();
    AscendC::LocalTensor<float> accum = accumBase[bufIdx * accumFloats_];
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint32_t chunkSrcElems =
        (chunkIdx + 1U == numChunks_) ? (static_cast<uint32_t>(hiddenDim_) - chunkIdx * chunkElems_) : chunkElems_;
    uint32_t chunkOutBytes = chunkSrcElems * outDtypeSize;
    if (accumCompactIdx_ < 0 || accumCompactIdx_ >= numEntriesPerRank_) {
        return;
    }
    uint64_t gmByteOffset = static_cast<uint64_t>(accumCompactIdx_) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize +
                            static_cast<uint64_t>(chunkIdx) * chunkElems_ * outDtypeSize;
    if (outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT) {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
        AscendC::GlobalTensor<float> dstGM;
        dstGM.SetGlobalBuffer((__gm__ float*)(gradUniqueOutGM + gmByteOffset));
        AscendC::DataCopyParams params{1U, static_cast<uint16_t>(chunkOutBytes), 0U, 0U};
        AscendC::DataCopyPad(dstGM, accum, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
    } else {
        AscendC::LocalTensor<uint8_t> entryRaw = entryBuf_->Get<uint8_t>();
        uint32_t flushCastOffset = Mc2Kernel::FLUSH_CAST_HEAD_BYTES;
        uint32_t castTailBytes = flushCastOffset + 2U * chunkOutStride_;
        ascendc_assert(castTailBytes <= entryBufBytes_,
                       "FlushAccumChunk staging overflow: need %u bytes, entryBuf=%u bytes", castTailBytes,
                       entryBufBytes_);
        AscendC::LocalTensor<uint8_t> flushCastBuf = entryRaw[flushCastOffset + bufIdx * chunkOutStride_];
        if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
            AscendC::LocalTensor<bfloat16_t> outT = flushCastBuf.ReinterpretCast<bfloat16_t>();
            AscendC::Cast(outT, accum, AscendC::RoundMode::CAST_RINT, chunkSrcElems);
        } else {
            AscendC::LocalTensor<half> outT = flushCastBuf.ReinterpretCast<half>();
            AscendC::Cast(outT, accum, AscendC::RoundMode::CAST_RINT, chunkSrcElems);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
        AscendC::GlobalTensor<uint8_t> dstGM;
        dstGM.SetGlobalBuffer((__gm__ uint8_t*)(gradUniqueOutGM + gmByteOffset));
        AscendC::DataCopyParams params{1U, static_cast<uint16_t>(chunkOutBytes), 0U, 0U};
        AscendC::DataCopyPad(dstGM, flushCastBuf, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
    }
    flushPending_[bufIdx] = false;
}

// 核区间钳制：非法 start/end 收敛到 [0, numRecv] 且 start<=end
__aicore__ inline void EngramFetchGradUnique::ClampCoreRange(uint32_t& start, uint32_t& end, uint32_t numRecv)
{
    if (end > numRecv || start > numRecv || start > end) {
        if (end > numRecv) {
            end = numRecv;
        }
        if (start > numRecv) {
            start = numRecv;
        }
        if (start > end) {
            start = end;
        }
    }
}

__aicore__ inline void EngramFetchGradUnique::LoadCoreRange(uint32_t numRecv, GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                            uint32_t& start, uint32_t& end, int32_t& preCoreOffset)
{
    AscendC::GlobalTensor<int32_t> coreStartGMT;
    coreStartGMT.SetGlobalBuffer((__gm__ int32_t*)coreStartGM);
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t*)segCountGM);

    AscendC::LocalTensor<int32_t> ub = tempBuf_->Get<int32_t>();
    uint32_t totalBytes = totalBlocks_ * static_cast<uint32_t>(sizeof(int32_t));
    uint32_t segOffElems = (totalBytes + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN *
                           (Mc2Kernel::UB_ALIGN / static_cast<uint32_t>(sizeof(int32_t)));
    AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
    AscendC::DataCopyExtParams params{1U, totalBytes, 0U, 0U, 0U};
    AscendC::DataCopyPad(ub, coreStartGMT, params, pad);
    AscendC::LocalTensor<int32_t> segUb = ub[segOffElems];
    AscendC::DataCopyPad(segUb, segCountGMT, params, pad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    int32_t nextValidStart = -1;
    preCoreOffset = 0;
    for (int32_t core = static_cast<int32_t>(totalBlocks_) - 1; core >= 0; core--) {
        int32_t segCnt = segUb.GetValue(core);
        if (segCnt == 0) {
            if (nextValidStart >= 0) {
                ub.SetValue(core, nextValidStart);
            } else {
                ub.SetValue(core, static_cast<int32_t>(numRecv));
            }
        } else {
            nextValidStart = ub.GetValue(core);
        }
        if (static_cast<uint32_t>(core) < aivId_) {
            preCoreOffset += segCnt;
        }
    }

    start = static_cast<uint32_t>(ub.GetValue(aivId_));
    if (aivId_ == totalBlocks_ - 1U) {
        end = numRecv;
    } else {
        end = static_cast<uint32_t>(ub.GetValue(aivId_ + 1U));
    }

    ClampCoreRange(start, end, numRecv);
    myPreCoreOffset_ = preCoreOffset;
}

// fp32 输出路径：accum 直拷回 GM（按 MAX_BLOCK_BYTES 分块）
__aicore__ inline void EngramFetchGradUnique::FlushAccumFp32(AscendC::LocalTensor<float>& accum, uint32_t bufIdx,
                                                             uint64_t gmByteOffset, GM_ADDR gradUniqueOutGM)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);

    AscendC::GlobalTensor<float> dstGM;
    dstGM.SetGlobalBuffer((__gm__ float*)(gradUniqueOutGM + gmByteOffset));
    uint32_t totalBytes = static_cast<uint32_t>(hiddenDim_) * sizeof(float);
    uint32_t numBlocks = (totalBytes + Mc2Kernel::MAX_BLOCK_BYTES - 1U) / Mc2Kernel::MAX_BLOCK_BYTES;
    for (uint32_t b = 0; b < numBlocks; b++) {
        uint32_t byteOffset = b * Mc2Kernel::MAX_BLOCK_BYTES;
        uint32_t elemOffset = byteOffset / sizeof(float);
        uint32_t remaining = totalBytes - byteOffset;
        uint16_t blkBytes =
            static_cast<uint16_t>(remaining > Mc2Kernel::MAX_BLOCK_BYTES ? Mc2Kernel::MAX_BLOCK_BYTES : remaining);
        AscendC::DataCopyParams params{1U, blkBytes, 0U, 0U};
        AscendC::DataCopyPad(dstGM[elemOffset], accum[elemOffset], params);
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
}

// Non-fp32 path: quantize accum into the ping-pong scratch at the tail of entryBuf_ and return that view.
// flushCastBuf MUST NOT live in castBuf_: castBuf_ is the fp32 grad staging
// (castPingF/castPongF) for the in-flight sub-batch; casting the accumulated row
// here overwrites gradFp32 row 0 and FlushAccum fires mid-AccumulateSubBatch
// (at compactIdx switches, before Add reads gradFp32[0]) -> corrupted accum.
// Use the unused tail of entryBuf_ (pingInt32 only occupies the first
// 5*ENTRY_BATCH_CAP*4 bytes), as the pre-merge baseline did.
__aicore__ inline AscendC::LocalTensor<uint8_t> EngramFetchGradUnique::CastAccumToStaging(
    AscendC::LocalTensor<float>& accum, uint32_t bufIdx)
{
    AscendC::LocalTensor<uint8_t> entryRaw = entryBuf_->Get<uint8_t>();
    uint32_t flushCastOffset = Mc2Kernel::FLUSH_CAST_HEAD_BYTES;
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint32_t castHalfBytes = (static_cast<uint32_t>(hiddenDim_) * outDtypeSize + Mc2Kernel::UB_ALIGN - 1U) /
                             Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
    // 双缓冲借用区必须完整落在 entryBuf_ 尾部内（Host 侧已按 hiddenDim 上界拒绝超限 shape，此处兜底）
    uint32_t castTailBytes = flushCastOffset + 2U * castHalfBytes;
    ascendc_assert(castTailBytes <= entryBufBytes_,
                   "FlushAccum cast staging overflow: need %u bytes, entryBuf=%u bytes, hiddenDim=%u", castTailBytes,
                   entryBufBytes_, static_cast<uint32_t>(hiddenDim_));
    AscendC::LocalTensor<uint8_t> flushCastBuf = entryRaw[flushCastOffset + bufIdx * castHalfBytes];
    uint32_t castCount = static_cast<uint32_t>(hiddenDim_);

    if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
        AscendC::LocalTensor<bfloat16_t> outT = flushCastBuf.ReinterpretCast<bfloat16_t>();
        AscendC::Cast(outT, accum, AscendC::RoundMode::CAST_RINT, castCount);
    } else {
        AscendC::LocalTensor<half> outT = flushCastBuf.ReinterpretCast<half>();
        AscendC::Cast(outT, accum, AscendC::RoundMode::CAST_RINT, castCount);
    }
    return flushCastBuf;
}

// 非 fp32 路径：把定点化暂存行按 dtype 分块拷回 GM
__aicore__ inline void EngramFetchGradUnique::CopyStagedAccumOut(AscendC::LocalTensor<uint8_t>& flushCastBuf,
                                                                 uint32_t bufIdx, uint64_t gmByteOffset,
                                                                 GM_ADDR gradUniqueOutGM)
{
    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(flushEvtVMte3Arr_[bufIdx]);

    if (outputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
        AscendC::GlobalTensor<bfloat16_t> dstGM;
        dstGM.SetGlobalBuffer((__gm__ bfloat16_t*)(gradUniqueOutGM + gmByteOffset));
        uint32_t totalBytes = static_cast<uint32_t>(hiddenDim_) * sizeof(bfloat16_t);
        uint32_t numBlocks = (totalBytes + Mc2Kernel::MAX_BLOCK_BYTES - 1U) / Mc2Kernel::MAX_BLOCK_BYTES;
        AscendC::LocalTensor<bfloat16_t> outT = flushCastBuf.ReinterpretCast<bfloat16_t>();
        for (uint32_t b = 0; b < numBlocks; b++) {
            uint32_t byteOffset = b * Mc2Kernel::MAX_BLOCK_BYTES;
            uint32_t elemOffset = byteOffset / sizeof(bfloat16_t);
            uint32_t remaining = totalBytes - byteOffset;
            uint16_t blkBytes =
                static_cast<uint16_t>(remaining > Mc2Kernel::MAX_BLOCK_BYTES ? Mc2Kernel::MAX_BLOCK_BYTES : remaining);
            AscendC::DataCopyParams params{1U, blkBytes, 0U, 0U};
            AscendC::DataCopyPad(dstGM[elemOffset], outT[elemOffset], params);
        }
    } else {
        AscendC::GlobalTensor<half> dstGM;
        dstGM.SetGlobalBuffer((__gm__ half*)(gradUniqueOutGM + gmByteOffset));
        uint32_t totalBytes = static_cast<uint32_t>(hiddenDim_) * sizeof(half);
        uint32_t numBlocks = (totalBytes + Mc2Kernel::MAX_BLOCK_BYTES - 1U) / Mc2Kernel::MAX_BLOCK_BYTES;
        AscendC::LocalTensor<half> outT = flushCastBuf.ReinterpretCast<half>();
        for (uint32_t b = 0; b < numBlocks; b++) {
            uint32_t byteOffset = b * Mc2Kernel::MAX_BLOCK_BYTES;
            uint32_t elemOffset = byteOffset / sizeof(half);
            uint32_t remaining = totalBytes - byteOffset;
            uint16_t blkBytes =
                static_cast<uint16_t>(remaining > Mc2Kernel::MAX_BLOCK_BYTES ? Mc2Kernel::MAX_BLOCK_BYTES : remaining);
            AscendC::DataCopyParams params{1U, blkBytes, 0U, 0U};
            AscendC::DataCopyPad(dstGM[elemOffset], outT[elemOffset], params);
        }
    }

    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[bufIdx]);
}

__aicore__ inline void EngramFetchGradUnique::FlushAccum(GM_ADDR gradUniqueOutGM)
{
    uint32_t bufIdx = accumIdx_;
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();
    AscendC::LocalTensor<float> accum = accumBase[bufIdx * accumFloats_];
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint64_t gmByteOffset = static_cast<uint64_t>(accumCompactIdx_) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize;
    if (accumCompactIdx_ < 0 || accumCompactIdx_ >= numEntriesPerRank_) {
        return;
    }

    if (outputDtype_ == Mc2Kernel::ENGRAM_DT_FLOAT) {
        FlushAccumFp32(accum, bufIdx, gmByteOffset, gradUniqueOutGM);
    } else {
        AscendC::LocalTensor<uint8_t> flushCastBuf = CastAccumToStaging(accum, bufIdx);
        CopyStagedAccumOut(flushCastBuf, bufIdx, gmByteOffset, gradUniqueOutGM);
    }
    flushPending_[bufIdx] = true;
}

__aicore__ inline void EngramFetchGradUnique::CastToFP32(AscendC::LocalTensor<float> outT,
                                                         AscendC::LocalTensor<uint8_t> gradRaw, uint32_t count)
{
    if (inputDtype_ == Mc2Kernel::ENGRAM_DT_BFLOAT16) {
        AscendC::LocalTensor<bfloat16_t> inT = gradRaw.ReinterpretCast<bfloat16_t>();
        AscendC::Cast(outT, inT, AscendC::RoundMode::CAST_NONE, count);
    } else {
        AscendC::LocalTensor<half> inT = gradRaw.ReinterpretCast<half>();
        AscendC::Cast(outT, inT, AscendC::RoundMode::CAST_NONE, count);
    }
}

__aicore__ inline void EngramFetchGradUnique::AccumulateSubBatch(AscendC::LocalTensor<float>& gradFp32,
                                                                 uint32_t rowStrideFloats, uint32_t subStart,
                                                                 uint32_t subLen, GM_ADDR gradUniqueOutGM)
{
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();

    bool canDirectCopy = (inputDtype_ == outputDtype_);

    for (uint32_t j = 0; j < subLen; j++) {
        if (canDirectCopy && DirectFlagUb().GetValue(subStart + j) != 0) {
            continue;
        }
        int32_t compactIdx = CompactIdxUb().GetValue(subStart + j);
        if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
            continue;
        }
        if (compactIdx != accumCompactIdx_) {
            if (accumDirty_) {
                FlushAccum(gradUniqueOutGM);
                accumIdx_ ^= 1U;
                if (flushPending_[accumIdx_]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[accumIdx_]);
                    flushPending_[accumIdx_] = false;
                }
            }
            AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
            AscendC::Duplicate<float>(accum, 0.0f, static_cast<uint32_t>(hiddenDim_));
            AscendC::PipeBarrier<PIPE_V>();
            accumCompactIdx_ = compactIdx;
        }
        AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
        AscendC::Add<float>(accum, accum, gradFp32[j * rowStrideFloats], static_cast<uint32_t>(hiddenDim_));
        AscendC::PipeBarrier<PIPE_V>();
        accumDirty_ = true;
    }
}

__aicore__ inline void EngramFetchGradUnique::AccumulateDirectSubBatch(AscendC::LocalTensor<uint8_t>& gradRaw,
                                                                       uint32_t subStart, uint32_t subLen,
                                                                       AscendC::LocalTensor<float>& castRow,
                                                                       GM_ADDR gradUniqueOutGM)
{
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();
    uint32_t hiddenDim = static_cast<uint32_t>(hiddenDim_);

    for (uint32_t j = 0; j < subLen; j++) {
        if (DirectFlagUb().GetValue(subStart + j) != 0) {
            continue;
        }
        int32_t compactIdx = CompactIdxUb().GetValue(subStart + j);
        if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
            continue;
        }
        if (compactIdx != accumCompactIdx_) {
            if (accumDirty_) {
                FlushAccum(gradUniqueOutGM);
                accumIdx_ ^= 1U;
                if (flushPending_[accumIdx_]) {
                    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[accumIdx_]);
                    flushPending_[accumIdx_] = false;
                }
            }
            AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
            AscendC::Duplicate<float>(accum, 0.0f, hiddenDim);
            AscendC::PipeBarrier<PIPE_V>();
            accumCompactIdx_ = compactIdx;
        }
        AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
        CastToFP32(castRow, gradRaw[j * inRowStride_], hiddenDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(accum, accum, castRow, hiddenDim);
        AscendC::PipeBarrier<PIPE_V>();
        accumDirty_ = true;
    }
}

__aicore__ inline void EngramFetchGradUnique::FlushDirectRun(AscendC::LocalTensor<uint8_t>& gradRaw, uint32_t runStart,
                                                             uint32_t runLen, int32_t compactFirst,
                                                             GM_ADDR gradUniqueOutGM)
{
    uint32_t outDtypeSize = GetDtypeSize(outputDtype_);
    uint32_t hb = static_cast<uint32_t>(hiddenBytes_);
    uint64_t gmByteOffset = static_cast<uint64_t>(compactFirst) * static_cast<uint64_t>(hiddenDim_) * outDtypeSize;
    AscendC::GlobalTensor<uint8_t> dstGM;
    dstGM.SetGlobalBuffer((__gm__ uint8_t*)(gradUniqueOutGM + gmByteOffset));
    if (inRowStride_ == hb) {
        uint32_t totalBytes = runLen * hb;
        AscendC::DataCopyParams params{1U, static_cast<uint16_t>(totalBytes), 0U, 0U};
        AscendC::DataCopyPad(dstGM, gradRaw[runStart * inRowStride_], params);
        return;
    }
    constexpr uint32_t kMaxBatch = 32U;
    uint32_t done = 0U;
    while (done < runLen) {
        uint32_t thisRows = runLen - done;
        if (thisRows > kMaxBatch) {
            thisRows = kMaxBatch;
        }
        AscendC::DataCopyExtParams p{static_cast<uint16_t>(thisRows), hb, 0U, 0U, 0U};
        AscendC::DataCopyPad(dstGM[static_cast<uint64_t>(done) * hb],
                             gradRaw[static_cast<uint64_t>(runStart + done) * inRowStride_], p);
        done += thisRows;
    }
}

// emitLists 路径：逐元素遍历生成 unique/compact 表并维护 direct-run 状态机（NBNC/圈复杂度整改自 ProcessBatchUnique）
__aicore__ inline void EngramFetchGradUnique::WalkBatchUniqueRuns(
    const AscendC::LocalTensor<int32_t>& entryUb, uint32_t batchLen, int32_t runningOffset, int32_t rankBase,
    bool& isFirstElement, int32_t& prevEntry, int32_t& inclusiveSum, uint32_t& tileUniqueCnt, uint32_t& runCnt,
    uint32_t& accumCnt, bool& inRun, uint32_t& runStartPos)
{
    AscendC::LocalTensor<int32_t> uniqueT = UniqueUb();
    AscendC::LocalTensor<int32_t> compactT = CompactIdxUb();
    AscendC::LocalTensor<int32_t> runStartT = RunStartUb();
    AscendC::LocalTensor<int32_t> runLenT = RunLenUb();
    AscendC::LocalTensor<int32_t> accumT = AccumListUb();
    bool prevIsNewUnique = false;
    int32_t prevCompact = 0;
    for (uint32_t i = 0; i < batchLen; i++) {
        int32_t entry = entryUb.GetValue(i);
        bool isNewUnique = isFirstElement || (entry != prevEntry);
        isFirstElement = false;
        prevEntry = entry;
        if (isNewUnique) {
            inclusiveSum++;
            uniqueT.SetValue(tileUniqueCnt, entry - rankBase);
            tileUniqueCnt++;
        }
        int32_t compactIdx = runningOffset + inclusiveSum - 1;
        compactT.SetValue(i, compactIdx);
        if (i > 0) {
            bool dPrev = prevIsNewUnique && prevCompact != compactIdx;
            if (dPrev) {
                if (!inRun) {
                    inRun = true;
                    runStartPos = i - 1U;
                }
            } else {
                if (inRun) {
                    runStartT.SetValue(runCnt, static_cast<int32_t>(runStartPos));
                    runLenT.SetValue(runCnt, static_cast<int32_t>(i - 1U - runStartPos));
                    runCnt++;
                    inRun = false;
                }
                accumT.SetValue(accumCnt, static_cast<int32_t>(i - 1U));
                accumCnt++;
            }
        }
        prevIsNewUnique = isNewUnique;
        prevCompact = compactIdx;
    }
}

// emitLists 路径：批尾收拢最后一条 direct-run 与最后一个 accum 槽
__aicore__ inline void EngramFetchGradUnique::CloseBatchUniqueRuns(uint32_t batchLen, uint32_t& runCnt,
                                                                   uint32_t& accumCnt, bool& inRun,
                                                                   uint32_t& runStartPos)
{
    AscendC::LocalTensor<int32_t> runStartT = RunStartUb();
    AscendC::LocalTensor<int32_t> runLenT = RunLenUb();
    AscendC::LocalTensor<int32_t> accumT = AccumListUb();
    if (batchLen > 0) {
        if (inRun) {
            runStartT.SetValue(runCnt, static_cast<int32_t>(runStartPos));
            runLenT.SetValue(runCnt, static_cast<int32_t>(batchLen - 1U - runStartPos));
            runCnt++;
            inRun = false;
        }
        accumT.SetValue(accumCnt, static_cast<int32_t>(batchLen - 1U));
        accumCnt++;
    }
}

// 非 emitLists 路径：仅生成 compactIdx 与 direct 标志（延迟一拍在主循环内直接生成最终 flag）
__aicore__ inline void EngramFetchGradUnique::WalkBatchUniqueDirectFlags(const AscendC::LocalTensor<int32_t>& entryUb,
                                                                         uint32_t batchLen, int32_t runningOffset,
                                                                         bool& isFirstElement, int32_t& prevEntry,
                                                                         int32_t& inclusiveSum, uint32_t& tileUniqueCnt)
{
    bool canDirectCopy = (inputDtype_ == outputDtype_);
    bool prevIsNewUnique = false;
    int32_t prevCompact = 0;
    for (uint32_t i = 0; i < batchLen; i++) {
        int32_t entry = entryUb.GetValue(i);
        bool isNewUnique = isFirstElement || (entry != prevEntry);
        isFirstElement = false;
        prevEntry = entry;
        if (isNewUnique) {
            inclusiveSum++;
            UniqueUb().SetValue(
                tileUniqueCnt,
                static_cast<int32_t>(static_cast<int64_t>(entry) - static_cast<int64_t>(rankId_) * numEntriesPerRank_));
            tileUniqueCnt++;
        }
        int32_t compactIdx = runningOffset + inclusiveSum - 1;
        CompactIdxUb().SetValue(i, compactIdx);
        DirectFlagUb().SetValue(i, isNewUnique ? 1 : 0);
        if (i > 0) {
            DirectFlagUb().SetValue(i - 1, (canDirectCopy && prevIsNewUnique && prevCompact != compactIdx) ? 1 : 0);
        }
        prevIsNewUnique = isNewUnique;
        prevCompact = compactIdx;
    }
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessBatchUnique(
    uint32_t cur, uint32_t batchLen, int32_t runningOffset, int32_t& prevEntry, bool& isFirstElement,
    int32_t& inclusiveSum, GM_ADDR recvLocalEntryOutGM, GM_ADDR sortCompanionGM, bool emitLists, bool sparseCompact)
{
    if (cur >= numRecv_) {
        return 0;
    }
    if (cur + batchLen > numRecv_) {
        batchLen = numRecv_ - cur;
    }
    if (emitLists && sparseCompact) {
        return ProcessBatchUniqueVF(cur, batchLen, runningOffset, prevEntry, isFirstElement, inclusiveSum,
                                    recvLocalEntryOutGM, sortCompanionGM);
    }

    AscendC::LocalTensor<int32_t> entryUb = indicesBuf_->Get<int32_t>();

    AscendC::GlobalTensor<int32_t> sortedEntryGM;
    AscendC::GlobalTensor<int32_t> compGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t*)recvLocalEntryOutGM);
    compGM.SetGlobalBuffer((__gm__ int32_t*)sortCompanionGM);
    AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batchLen * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    AscendC::DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
    AscendC::DataCopyPad(CompUb(), compGM[cur], cpParams, cpPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    inclusiveSum = 0;
    uint32_t tileUniqueCnt = 0;
    if (emitLists) {
        int32_t rankBase = static_cast<int32_t>(static_cast<int64_t>(rankId_) * numEntriesPerRank_);
        uint32_t runCnt = 0;
        uint32_t accumCnt = 0;
        bool inRun = false;
        uint32_t runStartPos = 0;
        WalkBatchUniqueRuns(entryUb, batchLen, runningOffset, rankBase, isFirstElement, prevEntry, inclusiveSum,
                            tileUniqueCnt, runCnt, accumCnt, inRun, runStartPos);
        CloseBatchUniqueRuns(batchLen, runCnt, accumCnt, inRun, runStartPos);
        runCnt_ = runCnt;
        accumCnt_ = accumCnt;
        return tileUniqueCnt;
    }
    // flag[i] = isNewUnique[i] && i+1<batchLen && compact[i]!=compact[i+1]；compact 仅在 isNewUnique 时
    // 递增，故 compact[i]!=compact[i+1] ⟺ isNewUnique[i+1]。延迟一拍在主循环内直接生成最终 flag，
    // 消除原本逐元素遍历的第二遍循环（PERF-1）
    WalkBatchUniqueDirectFlags(entryUb, batchLen, runningOffset, isFirstElement, prevEntry, inclusiveSum,
                               tileUniqueCnt);
    if (batchLen > 0) {
        DirectFlagUb().SetValue(batchLen - 1, 0);
    }
    return tileUniqueCnt;
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessBatchUniqueVF(uint32_t cur, uint32_t batchLen,
                                                                       int32_t runningOffset, int32_t& prevEntry,
                                                                       bool& isFirstElement, int32_t& inclusiveSum,
                                                                       GM_ADDR recvLocalEntryOutGM,
                                                                       GM_ADDR sortCompanionGM)
{
    AscendC::LocalTensor<int32_t> entryUb = indicesBuf_->Get<int32_t>();
    AscendC::GlobalTensor<int32_t> sortedEntryGM;
    AscendC::GlobalTensor<int32_t> compGM;
    sortedEntryGM.SetGlobalBuffer((__gm__ int32_t*)recvLocalEntryOutGM);
    compGM.SetGlobalBuffer((__gm__ int32_t*)sortCompanionGM);
    if (cur >= numRecv_) {
        return 0;
    }
    if (cur + batchLen > numRecv_) {
        batchLen = numRecv_ - cur;
    }
    if (batchLen == 0U) {
        return 0;
    }
    const uint32_t b = batchLen;
    int32_t rankBase = static_cast<int32_t>(static_cast<int64_t>(rankId_) * numEntriesPerRank_);

    AscendC::PipeBarrier<PIPE_ALL>();

    AscendC::LocalTensor<int32_t> vfBase = gradBuf_->Get<int32_t>();
    AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(b * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    AscendC::DataCopyPad(entryUb, sortedEntryGM[cur], cpParams, cpPad);
    AscendC::DataCopyPad(CompUb(), compGM[cur], cpParams, cpPad);
    if (b > 1U) {
        AscendC::DataCopyExtParams e2Params{1U, static_cast<uint32_t>((b - 1U) * sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(vfBase[EngramVfTables::E2], sortedEntryGM[cur + 1U], e2Params, cpPad);
    }
    SyncFunc<AscendC::HardEvent::MTE2_V>(*pipe_);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    asc_vf_call<EngramVfBatchUniqueVF>(
        (__ubuf__ int32_t*)vfBase.GetPhyAddr(), (__ubuf__ int32_t*)entryUb.GetPhyAddr(),
        (__ubuf__ int32_t*)UniqueUb().GetPhyAddr(), (__ubuf__ int32_t*)RunStartUb().GetPhyAddr(),
        (__ubuf__ int32_t*)RunLenUb().GetPhyAddr(), (__ubuf__ int32_t*)RunCompactUb().GetPhyAddr(),
        (__ubuf__ int32_t*)DupStartUb().GetPhyAddr(), (__ubuf__ int32_t*)DupEndUb().GetPhyAddr(),
        (__ubuf__ int32_t*)DupCompactUb().GetPhyAddr(), b, prevEntry, isFirstElement ? 1U : 0U, rankBase,
        runningOffset);

    SyncFunc<AscendC::HardEvent::V_S>(*pipe_);
    SyncFunc<AscendC::HardEvent::V_MTE3>(*pipe_);

    const int32_t total = vfBase[EngramVfTables::PREFIX].GetValue(b - 1U);
    const int32_t runTotal = vfBase[EngramVfTables::RID].GetValue(b - 1U);
    const int32_t dupTotal = vfBase[EngramVfTables::GID].GetValue(b - 1U);
    inclusiveSum = total;
    runCnt_ = static_cast<uint32_t>(runTotal);
    dupCnt_ = static_cast<uint32_t>(dupTotal);
    accumCnt_ = 0;
    prevEntry = entryUb.GetValue(b - 1U);
    isFirstElement = false;
    return static_cast<uint32_t>(total);
}

__aicore__ inline void EngramFetchGradUnique::WriteBatchUnique(uint32_t tileUniqueCnt, int32_t& runningUniqueOffset,
                                                               GM_ADDR uniqueLocalEntryOutGM)
{
    AscendC::GlobalTensor<int32_t> uniqueEntryOutGM;
    uniqueEntryOutGM.SetGlobalBuffer((__gm__ int32_t*)uniqueLocalEntryOutGM);
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::DataCopyParams ueParams{1U, static_cast<uint16_t>(tileUniqueCnt * sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(uniqueEntryOutGM[runningUniqueOffset], UniqueUb(), ueParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
    runningUniqueOffset += static_cast<int32_t>(tileUniqueCnt);
}

__aicore__ inline void EngramFetchGradUnique::ProcessNonDirectRow(int32_t pos, uint32_t subStart,
                                                                  AscendC::LocalTensor<uint8_t>& gradRaw,
                                                                  AscendC::LocalTensor<float>& castRow,
                                                                  GM_ADDR gradUniqueOutGM, int32_t compactIdx)
{
    AscendC::LocalTensor<float> accumBase = accumBuf_->Get<float>();
    uint32_t hiddenDim = static_cast<uint32_t>(hiddenDim_);

    if (compactIdx < 0 || compactIdx >= numEntriesPerRank_) {
        return;
    }
    if (compactIdx != accumCompactIdx_) {
        if (accumDirty_) {
            FlushAccum(gradUniqueOutGM);
            accumIdx_ ^= 1U;
            if (flushPending_[accumIdx_]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(flushEvtMte3VArr_[accumIdx_]);
                flushPending_[accumIdx_] = false;
            }
        }
        accumCompactIdx_ = compactIdx;
        accumRowCnt_ = 0;
    }
    AscendC::LocalTensor<float> accum = accumBase[accumIdx_ * accumFloats_];
    uint32_t rowOff = static_cast<uint32_t>(pos - static_cast<int32_t>(subStart)) * inRowStride_;
    if (accumRowCnt_ == 0U) {
        if (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) {
            CastToFP32(accum, gradRaw[rowOff], hiddenDim);
            AscendC::PipeBarrier<PIPE_V>();
        } else {
            AscendC::LocalTensor<float> gradFp32 = gradRaw[rowOff].ReinterpretCast<float>();
            AscendC::Adds<float>(accum, gradFp32, 0.0f, hiddenDim);
            AscendC::PipeBarrier<PIPE_V>();
        }
    } else if (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT) {
        CastToFP32(castRow, gradRaw[rowOff], hiddenDim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add<float>(accum, accum, castRow, hiddenDim);
        AscendC::PipeBarrier<PIPE_V>();
    } else {
        AscendC::LocalTensor<float> gradFp32 = gradRaw[rowOff].ReinterpretCast<float>();
        AscendC::Add<float>(accum, accum, gradFp32, hiddenDim);
        AscendC::PipeBarrier<PIPE_V>();
    }
    accumRowCnt_++;
    accumDirty_ = true;
}
__aicore__ inline void EngramFetchGradUnique::AccumulateDupListWalk(uint32_t subStart, uint32_t subLen,
                                                                    uint32_t& cursor,
                                                                    AscendC::LocalTensor<uint8_t>& gradRaw,
                                                                    AscendC::LocalTensor<float>& castRow,
                                                                    GM_ADDR gradUniqueOutGM)
{
    uint32_t subEnd = subStart + subLen;
    while (cursor < dupCnt_) {
        int32_t g0 = DupStartUb().GetValue(cursor);
        int32_t gEnd = DupEndUb().GetValue(cursor);
        if (g0 < 0 || gEnd < g0) {
            cursor++;
            continue;
        }
        if (gEnd <= static_cast<int32_t>(subStart)) {
            cursor++;
            continue;
        }
        if (static_cast<uint32_t>(g0) >= subEnd) {
            break;
        }
        int32_t compactIdx = DupCompactUb().GetValue(cursor);
        uint32_t lo = (static_cast<uint32_t>(g0) > subStart) ? static_cast<uint32_t>(g0) : subStart;
        uint32_t hi = (static_cast<uint32_t>(gEnd) < subEnd) ? static_cast<uint32_t>(gEnd) : subEnd;
        for (uint32_t p = lo; p < hi; p++) {
            ProcessNonDirectRow(static_cast<int32_t>(p), subStart, gradRaw, castRow, gradUniqueOutGM, compactIdx);
        }
        if (gEnd > static_cast<int32_t>(subEnd)) {
            break;
        }
        cursor++;
    }
}
__aicore__ inline void EngramFetchGradUnique::EmitRunsFromDupList(uint32_t subStart, uint32_t subLen, uint32_t& cursor,
                                                                  AscendC::LocalTensor<uint8_t>& gradRaw,
                                                                  GM_ADDR gradUniqueOutGM)
{
    uint32_t subEnd = subStart + subLen;
    while (cursor < runCnt_) {
        int32_t a = RunStartUb().GetValue(cursor);
        int32_t e = RunLenUb().GetValue(cursor);
        if (a < 0 || e <= a) {
            cursor++;
            continue;
        }
        if (static_cast<uint32_t>(e) <= subStart) {
            cursor++;
            continue;
        }
        if (static_cast<uint32_t>(a) >= subEnd) {
            break;
        }
        uint32_t b = static_cast<uint32_t>(e);
        uint32_t lo = (static_cast<uint32_t>(a) > subStart) ? static_cast<uint32_t>(a) : subStart;
        uint32_t hi = (b < subEnd) ? b : subEnd;
        if (lo >= hi) {
            continue;
        }
        int32_t compactFirst = RunCompactUb().GetValue(cursor) + static_cast<int32_t>(lo - static_cast<uint32_t>(a));
        if (compactFirst < 0 || compactFirst >= numEntriesPerRank_) {
            continue;
        }
        FlushDirectRun(gradRaw, lo - subStart, hi - lo, compactFirst, gradUniqueOutGM);
        if (b > subEnd) {
            break;
        }
        cursor++;
    }
}
__aicore__ inline void EngramFetchGradUnique::ProcessDirectSubBatchLoopV2(uint32_t batchLen, uint32_t maxGradPerBatch,
                                                                          AscendC::LocalTensor<uint8_t>& gradBase,
                                                                          GM_ADDR recvGradGM, GM_ADDR gradUniqueOutGM)
{
    event_t evtMte2V[CHUNK_EVT_BUFS];
    event_t evtVMte2[CHUNK_EVT_BUFS];
    event_t evtMte2Mte3[CHUNK_EVT_BUFS];
    event_t evtMte3Mte2[CHUNK_EVT_BUFS];
    AllocChunkColEvents(evtMte2V, evtVMte2, evtMte2Mte3, evtMte3Mte2, CHUNK_EVT_BUFS);
    AscendC::LocalTensor<float> castRow = castBuf_->Get<float>();
    constexpr uint32_t kBufBytes = Mc2Kernel::GRAD_PING_BYTES / 2U;
    uint32_t bufRows = kBufBytes / inRowStride_;
    if (bufRows < 1U) {
        bufRows = 1U;
    }
    if (bufRows < maxGradPerBatch) {
        maxGradPerBatch = bufRows;
    }

    uint32_t accumCursor = 0;
    uint32_t runCursor = 0;
    uint32_t tileIdx = 0;
    for (uint32_t subStart = 0; subStart < batchLen; subStart += maxGradPerBatch) {
        uint32_t subLen = batchLen - subStart;
        if (subLen > maxGradPerBatch) {
            subLen = maxGradPerBatch;
        }
        uint32_t evtBufIdx = tileIdx % CHUNK_EVT_BUFS;
        AscendC::LocalTensor<uint8_t> gradRaw = gradBase[evtBufIdx * kBufBytes];

        if (tileIdx >= CHUNK_EVT_BUFS) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[evtBufIdx]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[evtBufIdx]);
        }

        LoadGradRowsSubBatch(gradRaw, subStart, subLen, recvGradGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2V[evtBufIdx]);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[evtBufIdx]);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2V[evtBufIdx]);

        AccumulateDupListWalk(subStart, subLen, accumCursor, gradRaw, castRow, gradUniqueOutGM);

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3[evtBufIdx]);
        EmitRunsFromDupList(subStart, subLen, runCursor, gradRaw, gradUniqueOutGM);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2[evtBufIdx]);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evtVMte2[evtBufIdx]);
        tileIdx++;
    }

    DrainChunkColEvents(tileIdx, evtMte2V, evtVMte2, evtMte2Mte3, evtMte3Mte2, CHUNK_EVT_BUFS);
}

// 非 list 路径：把一个 sub 批的 grad 行从 GM 搬入 UB（行按 inRowStride_ 排布）
__aicore__ inline void EngramFetchGradUnique::LoadGradRowsSubBatch(AscendC::LocalTensor<uint8_t>& gradRaw,
                                                                   uint32_t subStart, uint32_t subLen,
                                                                   GM_ADDR recvGradGM)
{
    AscendC::DataCopyPadExtParams<uint8_t> gradPad{false, 0, 0, 0};
    for (uint32_t j = 0; j < subLen; j++) {
        int32_t recvIdx = CompUb().GetValue(subStart + j);
        if (recvIdx < 0 || static_cast<uint32_t>(recvIdx) >= numRecv_) {
            recvIdx = 0;
        }
        GM_ADDR gradAddr = recvGradGM + static_cast<uint64_t>(recvIdx) * hiddenBytes_;
        AscendC::DataCopyExtParams gradParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
        AscendC::GlobalTensor<uint8_t> gradSrcGM;
        gradSrcGM.SetGlobalBuffer((__gm__ uint8_t*)gradAddr);
        AscendC::DataCopyPad(gradRaw[static_cast<uint64_t>(j) * inRowStride_], gradSrcGM, gradParams, gradPad);
    }
}

// 非 list 路径：一个 sub 批的累加三分支（direct 累加 / 逐行 cast 后累加 / fp32 直读累加）
__aicore__ inline void EngramFetchGradUnique::AccumulateScatterSubBatch(AscendC::LocalTensor<uint8_t>& gradRaw,
                                                                        uint32_t bufIdx, uint32_t subStart,
                                                                        uint32_t subLen, bool canDirectCopy,
                                                                        uint32_t maxGradPerBatch,
                                                                        GM_ADDR gradUniqueOutGM)
{
    bool needCast = (inputDtype_ != Mc2Kernel::ENGRAM_DT_FLOAT);
    uint32_t castHalfFloats = 0;
    uint32_t fp32StrideFloats = fp32RowStride_ / sizeof(float);
    if (needCast) {
        uint32_t castHalfBytes =
            (fp32RowStride_ * maxGradPerBatch + Mc2Kernel::UB_ALIGN - 1U) / Mc2Kernel::UB_ALIGN * Mc2Kernel::UB_ALIGN;
        castHalfFloats = castHalfBytes / sizeof(float);
    }

    if (canDirectCopy && needCast) {
        AscendC::LocalTensor<float> castRow = castBuf_->Get<float>();
        AccumulateDirectSubBatch(gradRaw, subStart, subLen, castRow, gradUniqueOutGM);
    } else if (needCast) {
        AscendC::LocalTensor<float> castPingF = castBuf_->Get<float>();
        AscendC::LocalTensor<float> castPongF = castPingF[castHalfFloats];
        AscendC::LocalTensor<float> gradFp32 = (bufIdx == 0U) ? castPingF : castPongF;
        // gradRaw 行带 32B 对齐 stride，cast 必须逐行进行（行间存在 padding 字节）
        for (uint32_t j = 0; j < subLen; j++) {
            CastToFP32(gradFp32[j * fp32StrideFloats], gradRaw[j * inRowStride_], static_cast<uint32_t>(hiddenDim_));
        }
        AscendC::PipeBarrier<PIPE_V>();
        AccumulateSubBatch(gradFp32, fp32StrideFloats, subStart, subLen, gradUniqueOutGM);
    } else {
        AscendC::LocalTensor<float> gradFp32 = gradRaw.ReinterpretCast<float>();
        AccumulateSubBatch(gradFp32, inRowStride_ / sizeof(float), subStart, subLen, gradUniqueOutGM);
    }
}

// 非 list 路径：按 DirectFlag 连续段直拷（while 走 run，跳过非法 compact）
__aicore__ inline void EngramFetchGradUnique::FlushDirectRunsFromFlags(AscendC::LocalTensor<uint8_t>& gradRaw,
                                                                       uint32_t subStart, uint32_t subLen,
                                                                       GM_ADDR gradUniqueOutGM)
{
    uint32_t j = 0;
    while (j < subLen) {
        if (DirectFlagUb().GetValue(subStart + j) != 0) {
            uint32_t runStart = j;
            j++;
            while (j < subLen && DirectFlagUb().GetValue(subStart + j) != 0) {
                j++;
            }
            int32_t compactFirst = CompactIdxUb().GetValue(subStart + runStart);
            if (compactFirst < 0 || compactFirst >= numEntriesPerRank_) {
                continue;
            }
            FlushDirectRun(gradRaw, runStart, j - runStart, compactFirst, gradUniqueOutGM);
        } else {
            j++;
        }
    }
}

// 非 list 路径：ping/pong 事件流水线的 tile 主循环（grad 搬入 → 累加/直拷 → 事件推进）
__aicore__ inline void EngramFetchGradUnique::RunScatterTilesV1(uint32_t batchLen, uint32_t maxGradPerBatch,
                                                                bool singleRowMode, bool canDirectCopy,
                                                                AscendC::LocalTensor<uint8_t>& gradPing,
                                                                GM_ADDR recvGradGM, GM_ADDR gradUniqueOutGM)
{
    AscendC::LocalTensor<uint8_t> gradPong = gradPing[Mc2Kernel::GRAD_PING_BYTES];

    event_t evtMte2V_0 = static_cast<event_t>(0);
    event_t evtMte2V_1 = static_cast<event_t>(0);
    event_t evtVMte2_0 = static_cast<event_t>(0);
    event_t evtVMte2_1 = static_cast<event_t>(0);
    event_t evtMte2Mte3_0 = static_cast<event_t>(0);
    event_t evtMte2Mte3_1 = static_cast<event_t>(0);
    event_t evtMte3Mte2_0 = static_cast<event_t>(0);
    event_t evtMte3Mte2_1 = static_cast<event_t>(0);

    evtMte2V_0 = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
    evtMte2V_1 = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_V>());
    evtVMte2_0 = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
    evtVMte2_1 = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::V_MTE2>());
    if (canDirectCopy) {
        evtMte2Mte3_0 = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
        evtMte2Mte3_1 = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE3>());
        evtMte3Mte2_0 = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
        evtMte3Mte2_1 = static_cast<event_t>(pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>());
    }

    uint32_t tileIdx = 0;
    for (uint32_t subStart = 0; subStart < batchLen; subStart += maxGradPerBatch) {
        uint32_t subLen = batchLen - subStart;
        if (subLen > maxGradPerBatch) {
            subLen = maxGradPerBatch;
        }
        uint32_t bufIdx = tileIdx % 2U;
        AscendC::LocalTensor<uint8_t> gradRaw = (bufIdx == 0U || singleRowMode) ? gradPing : gradPong;
        bool useEvt0 = singleRowMode || bufIdx == 0U;

        if (tileIdx >= (singleRowMode ? 1U : 2U)) {
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(useEvt0 ? evtVMte2_0 : evtVMte2_1);
            if (canDirectCopy) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(useEvt0 ? evtMte3Mte2_0 : evtMte3Mte2_1);
            }
        }

        LoadGradRowsSubBatch(gradRaw, subStart, subLen, recvGradGM);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(useEvt0 ? evtMte2V_0 : evtMte2V_1);
        if (canDirectCopy) {
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(useEvt0 ? evtMte2Mte3_0 : evtMte2Mte3_1);
        }

        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(useEvt0 ? evtMte2V_0 : evtMte2V_1);

        AccumulateScatterSubBatch(gradRaw, bufIdx, subStart, subLen, canDirectCopy, maxGradPerBatch, gradUniqueOutGM);

        if (canDirectCopy) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(useEvt0 ? evtMte2Mte3_0 : evtMte2Mte3_1);
            FlushDirectRunsFromFlags(gradRaw, subStart, subLen, gradUniqueOutGM);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(useEvt0 ? evtMte3Mte2_0 : evtMte3Mte2_1);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(useEvt0 ? evtVMte2_0 : evtVMte2_1);
        tileIdx++;
    }

    DrainScatterEvtV1(tileIdx, singleRowMode, canDirectCopy, evtMte2V_0, evtMte2V_1, evtVMte2_0, evtVMte2_1,
                      evtMte2Mte3_0, evtMte2Mte3_1, evtMte3Mte2_0, evtMte3Mte2_1);
}

// 非 list 路径：排空尾批事件并归还全部事件 ID
__aicore__ inline void EngramFetchGradUnique::DrainScatterEvtV1(uint32_t tileIdx, bool singleRowMode,
                                                                bool canDirectCopy, event_t evtMte2V_0,
                                                                event_t evtMte2V_1, event_t evtVMte2_0,
                                                                event_t evtVMte2_1, event_t evtMte2Mte3_0,
                                                                event_t evtMte2Mte3_1, event_t evtMte3Mte2_0,
                                                                event_t evtMte3Mte2_1)
{
    if (tileIdx >= 1U) {
        uint32_t lastEvt = singleRowMode ? 0U : ((tileIdx - 1U) % 2U);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(lastEvt == 0U ? evtVMte2_0 : evtVMte2_1);
        if (canDirectCopy) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(lastEvt == 0U ? evtMte3Mte2_0 : evtMte3Mte2_1);
        }
    }
    if (!singleRowMode && tileIdx >= 2U) {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(tileIdx % 2U == 0U ? evtVMte2_0 : evtVMte2_1);
        if (canDirectCopy) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(tileIdx % 2U == 0U ? evtMte3Mte2_0 : evtMte3Mte2_1);
        }
    }
    pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(evtMte2V_0);
    pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_V>(evtMte2V_1);
    pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(evtVMte2_0);
    pipe_->ReleaseEventID<AscendC::HardEvent::V_MTE2>(evtVMte2_1);
    if (canDirectCopy) {
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3_0);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evtMte2Mte3_1);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2_0);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evtMte3Mte2_1);
    }
}

__aicore__ inline uint32_t EngramFetchGradUnique::ProcessScatterBatch(
    uint32_t cur, uint32_t end, int32_t& runningOffset, int32_t& runningUniqueOffset, int32_t& prevEntry,
    bool& isFirstElement, GM_ADDR recvLocalEntryOutGM, GM_ADDR uniqueLocalEntryOutGM, GM_ADDR gradUniqueOutGM,
    GM_ADDR recvGradGM, GM_ADDR sortCompanionGM)
{
    uint32_t batchLen = end - cur;
    if (batchLen > Mc2Kernel::ENTRY_BATCH_CAP) {
        batchLen = Mc2Kernel::ENTRY_BATCH_CAP;
    }

    int32_t inclusiveSum = 0;
    bool singleRowMode = inRowStride_ > Mc2Kernel::GRAD_PING_BYTES;
    bool canDirectCopy = (inputDtype_ == outputDtype_);
    bool useListPath = canDirectCopy && !singleRowMode && inRowStride_ <= Mc2Kernel::GRAD_PING_BYTES / 2U;
    uint32_t tileUniqueCnt = ProcessBatchUnique(cur, batchLen, runningOffset, prevEntry, isFirstElement, inclusiveSum,
                                                recvLocalEntryOutGM, sortCompanionGM, useListPath, useListPath);

    uint32_t maxGradPerBatch = Mc2Kernel::GRAD_PING_BYTES / inRowStride_;
    if (maxGradPerBatch < 1U) {
        maxGradPerBatch = 1U;
    }
    // 行宽超过 32KB 半缓冲时禁用 ping/pong 拆分：整缓冲单行、跨 tile 用 evt_0 串行，
    // 否则 tileIdx=1 写 gradPing[32K] 处的单行会越过 gradBuf_ 污染池内相邻缓冲
    bool canDirectCopyEarly = canDirectCopy;
    if (!canDirectCopyEarly && maxGradPerBatch > gradSubBatch_) {
        maxGradPerBatch = gradSubBatch_;
    }
    AscendC::LocalTensor<uint8_t> gradPing = gradBuf_->Get<uint8_t>();

    if (useListPath) {
        ProcessDirectSubBatchLoopV2(batchLen, maxGradPerBatch, gradPing, recvGradGM, gradUniqueOutGM);
    } else {
        RunScatterTilesV1(batchLen, maxGradPerBatch, singleRowMode, canDirectCopy, gradPing, recvGradGM,
                          gradUniqueOutGM);
    }

    if (tileUniqueCnt > 0) {
        WriteBatchUnique(tileUniqueCnt, runningUniqueOffset, uniqueLocalEntryOutGM);
    }
    runningOffset += inclusiveSum;
    return cur + batchLen;
}

__aicore__ inline void EngramFetchGradUnique::ScatterAccumulateParallel(uint32_t numRecv, GM_ADDR recvLocalEntryOutGM,
                                                                        GM_ADDR uniqueLocalEntryOutGM,
                                                                        GM_ADDR gradUniqueOutGM, GM_ADDR recvGradGM,
                                                                        GM_ADDR coreStartGM, GM_ADDR segCountGM,
                                                                        GM_ADDR sortCompanionGM)
{
    uint32_t start;
    uint32_t end;
    int32_t preCoreOffset;
    LoadCoreRange(numRecv, coreStartGM, segCountGM, start, end, preCoreOffset);
    if (start >= end) {
        return;
    }
    accumCompactIdx_ = -1;
    accumDirty_ = false;
    if (chunkElems_ > 0U) {
        ChunkedScatterRange(start, end, preCoreOffset, recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM,
                            recvGradGM, sortCompanionGM);
        return;
    }
    int32_t runningOffset = preCoreOffset;
    int32_t runningUniqueOffset = preCoreOffset;
    int32_t prevEntry = 0;
    bool isFirstElement = true;
    if (start > 0U) {
        AscendC::GlobalTensor<int32_t> sortedEntryGM;
        sortedEntryGM.SetGlobalBuffer((__gm__ int32_t*)recvLocalEntryOutGM);
        AscendC::LocalTensor<int32_t> prevUb = tempBuf_->Get<int32_t>();
        AscendC::DataCopyPadExtParams<int32_t> prevPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams prevParams{1U, static_cast<uint32_t>(sizeof(int32_t)), 0U, 0U, 0U};
        AscendC::DataCopyPad(prevUb, sortedEntryGM[start - 1U], prevParams, prevPad);
        SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
        prevEntry = prevUb.GetValue(0);
        isFirstElement = false;
    }

    uint32_t cur = start;
    while (cur < end) {
        cur = ProcessScatterBatch(cur, end, runningOffset, runningUniqueOffset, prevEntry, isFirstElement,
                                  recvLocalEntryOutGM, uniqueLocalEntryOutGM, gradUniqueOutGM, recvGradGM,
                                  sortCompanionGM);
    }
}

__aicore__ inline void EngramFetchGradUnique::WriteNumUnique(GM_ADDR segCountGM, GM_ADDR numUniqueOutGM)
{
    // 仅core0汇总各核segCount并普通回写numUnique(numUniqueOutGM未初始化, 不能原子加)。
    // 其余核直接返回, 省掉全核的GM加载+标量求和+尾部SyncAll(稳态约2K cyc/核);
    // 各核segCount早已由CountUniquesParallel后的SyncAll保证可见, numUnique由kernel
    // 结束顺序(host在kernel完成后读取)保证可见, 无需额外同步。
    if (aivId_ != 0) {
        return;
    }
    AscendC::GlobalTensor<int32_t> segCountGMT;
    segCountGMT.SetGlobalBuffer((__gm__ int32_t*)segCountGM);
    AscendC::LocalTensor<int32_t> segCountUb = tempBuf_->Get<int32_t>();
    AscendC::DataCopyPadExtParams<int32_t> scPad{false, 0, 0, 0};
    AscendC::DataCopyExtParams scParams{1U, static_cast<uint32_t>(totalBlocks_ * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPad(segCountUb, segCountGMT, scParams, scPad);
    SyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);

    int32_t numUnique = 0;
    for (uint32_t core = 0; core < totalBlocks_; core++) {
        numUnique += segCountUb.GetValue(core);
    }

    AscendC::LocalTensor<int32_t> tmp = statusBuf_->Get<int32_t>();
    tmp.SetValue(0, numUnique);
    SyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
    AscendC::GlobalTensor<int32_t> numUniqueGM;
    numUniqueGM.SetGlobalBuffer((__gm__ int32_t*)numUniqueOutGM);
    AscendC::DataCopyParams p{1U, static_cast<uint16_t>(sizeof(int32_t)), 0U, 0U};
    AscendC::DataCopyPad(numUniqueGM, tmp, p);
    SyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
}

} // namespace EngramFetchGradUnique

#endif
