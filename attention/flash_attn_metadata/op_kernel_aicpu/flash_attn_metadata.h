/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attn_metadata.h
 * \brief
 */

#ifndef FLASH_ATTN_METADATA_H
#define FLASH_ATTN_METADATA_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <cassert>

namespace optiling {

// Constants
using FA_METADATA_T = uint32_t;
constexpr uint32_t METADATA_STRIDE = 16U;

// Head Metadata Index Definitions
constexpr uint32_t HEAD_SECTION_NUM_INDEX = 0U;
constexpr uint32_t HEAD_IS_FD_INDEX = 1U;
constexpr uint32_t HEAD_M_BASE_SIZE_INDEX = 2U;
constexpr uint32_t HEAD_S2_BASE_SIZE_INDEX = 3U;
constexpr uint32_t HEAD_AIC_NUM_INDEX = 4U;
constexpr uint32_t HEAD_AIV_NUM_INDEX = 5U;
constexpr uint32_t HEAD_OUTPUT_LAYOUT_INDEX = 6U;
// head[7] is the FAG offset. head[8]: 0 = initialize, 1 = skip, 2 = compare head[9] with physical Q S.
// Mode 2 is only valid for non-TND layouts; older kernels initialize for this mode.
constexpr uint32_t HEAD_SKIP_INIT_OUTPUT_INDEX = 8U;
constexpr uint32_t HEAD_UNIFORM_Q_LEN_INDEX = 9U;

// FA Metadata Index Definitions
constexpr uint32_t FA_BN_START_INDEX = 0U;
constexpr uint32_t FA_M_START_INDEX = 1U;
constexpr uint32_t FA_S2_START_INDEX = 2U;
constexpr uint32_t FA_BN_END_INDEX = 3U;
constexpr uint32_t FA_M_END_INDEX = 4U;
constexpr uint32_t FA_S2_END_INDEX = 5U;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6U;

// FD Metadata Index Definitions
constexpr uint32_t FD_BN_IDX_INDEX = 0U;
constexpr uint32_t FD_M_IDX_INDEX = 1U;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 2U;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 3U;
constexpr uint32_t FD_M_START_INDEX = 4U;
constexpr uint32_t FD_M_NUM_INDEX = 5U;

// FAG (Flash Attn Grad) Metadata Index Definitions — head[7] records FAG start offset
// (head[4]/[5]/[6] are taken by HEAD_AIC_NUM_INDEX/HEAD_AIV_NUM_INDEX/HEAD_OUTPUT_LAYOUT_INDEX)
constexpr uint32_t HEAD_FAG_START_OFFSET_INDEX = 7U;
constexpr uint32_t FAG_METADATA_SIZE = 416U;
constexpr uint32_t FAG_AIV_SLOT_NUM = 72U;
constexpr uint32_t FAG_CORE_LIST_NUM = 36U;

// FAG Metadata field indices (relative to fagStartOffset)
constexpr uint32_t FAG_SPLIT_AXIS_INDEX = 0U;
constexpr uint32_t FAG_BLOCK_OUTER_INDEX = 1U;
constexpr uint32_t FAG_BLOCK_FACTOR_INDEX = 2U;
constexpr uint32_t FAG_S1_OUTER_INDEX = 3U;
constexpr uint32_t FAG_S2_OUTER_INDEX = 4U;
constexpr uint32_t FAG_LAYOUT_TYPE_INDEX = 5U;
constexpr uint32_t FAG_IS_SPARSE_INDEX = 6U;
constexpr uint32_t FAG_SPARSE_MODE_INDEX = 7U;
constexpr uint32_t FAG_BLOCK_STARTS_OFFSET = 8U;
constexpr uint32_t FAG_BLOCK_ENDS_OFFSET = 44U;
constexpr uint32_t FAG_TND_START_BIDX_OFFSET = 80U;
constexpr uint32_t FAG_MASK_MODE_INDEX = 116U;
constexpr uint32_t FAG_WIN_LEFT_INDEX = 117U;
constexpr uint32_t FAG_WIN_RIGHT_INDEX = 118U;
constexpr uint32_t FAG_MAX_SEQLEN_Q_INDEX = 119U;
constexpr uint32_t FAG_MAX_SEQLEN_KV_INDEX = 120U;
// pre/post 行区间：通常 nAiv = blockOuter * 2；零计算任务仍保留清零/post workers。
constexpr uint32_t FAG_PRE_AIV_NUM_INDEX = 121U;
constexpr uint32_t FAG_DQ_ROWS_INDEX = 122U;
constexpr uint32_t FAG_DKV_ROWS_INDEX = 123U;
constexpr uint32_t FAG_DQ_ROW_STARTS_OFFSET = 124U;  // [72]
constexpr uint32_t FAG_DQ_ROW_ENDS_OFFSET = 196U;    // [72]
constexpr uint32_t FAG_DKV_ROW_STARTS_OFFSET = 268U; // [72]
constexpr uint32_t FAG_DKV_ROW_ENDS_OFFSET = 340U;   // [72]  // 340+72=412
// Ordinary BN2S2 fallback flag in the original padding. The deterministic
// extension remains at the same next 16-word boundary.
constexpr uint32_t FAG_FALLBACK_TO_GS_INDEX = 412U;
// A signed consumer reads -1 for a complete, unique ordinary KV owner and +1..7 for GS geometry.
constexpr uint32_t FAG_FIXED_GS_MODE_INDEX = 413U;
constexpr int32_t FAG_COMPLETE_KV_OWNER = -1;
constexpr uint32_t FAG_FIXED_GS_Q_TILES_INDEX = 414U;
constexpr uint32_t FAG_FIXED_GS_KV_TILES_INDEX = 415U;
static_assert(FAG_FIXED_GS_KV_TILES_INDEX + 1 == FAG_METADATA_SIZE);

// FAG internal constants
constexpr uint32_t FAG_S1CV_RATIO_DEFAULT = 2U;
constexpr uint32_t FAG_S2CV_RATIO_DEFAULT = 1U;
constexpr int64_t FAG_ALIGN64 = 64;
constexpr uint32_t FAG_ARRAY_LENGTH = 3U;
constexpr uint32_t FAG_BATCH_MAX_SIZE = 2048U;
constexpr uint32_t FAG_MAX_S2_OUTER = 1024U;

// FAG layout type values (matching flash_attention_score_grad tiling common_regbase.h)
constexpr uint32_t FAG_INPUT_FORMAT_BS2N2GD = 1U; // BSND
constexpr uint32_t FAG_INPUT_FORMAT_BN2GS2D = 3U; // BNSD
constexpr uint32_t FAG_INPUT_FORMAT_TND = 4U;     // TND

// FAG sparse mode values (matching flash_attention_score_grad SparseMode enum)
constexpr uint32_t FAG_SPARSE_NO_MASK = 0U;
constexpr uint32_t FAG_SPARSE_RIGHT_DOWN_CAUSAL = 3U;
constexpr uint32_t FAG_SPARSE_BAND = 4U;

// FAG split axis values (matching SplitAxisEnum)
constexpr uint32_t FAG_SPLIT_AXIS_BN2GS1S2 = 0U;
constexpr uint32_t FAG_SPLIT_AXIS_BN2 = 1U;
constexpr uint32_t FAG_SPLIT_AXIS_BN2S2 = 2U;

namespace detail {
struct FaMetadata {
    uint32_t sectionNum;
    uint32_t aicNum;
    uint32_t aivNum;
    FA_METADATA_T* headMetadata; // [METADATA_STRIDE];
    FA_METADATA_T* faMetadata;   // [sectionNum][aicNum][METADATA_STRIDE];
    FA_METADATA_T* fdMetadata;   // [sectionNum][aivNum][METADATA_STRIDE];
    FaMetadata(uint32_t aicNum, uint32_t aivNum, uint32_t sectionNum, void* metadataPtr)
        : sectionNum(sectionNum),
          aicNum(aicNum),
          aivNum(aivNum),
          headMetadata(static_cast<FA_METADATA_T*>(metadataPtr)),
          faMetadata(headMetadata + METADATA_STRIDE),
          fdMetadata(faMetadata + static_cast<size_t>(sectionNum) * static_cast<size_t>(aicNum * METADATA_STRIDE))
    {
        headMetadata[0] = sectionNum;
    }

    void Clear()
    {
        for (size_t i = 0; i < METADATA_STRIDE; ++i) {
            headMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * aicNum * METADATA_STRIDE; ++i) {
            faMetadata[i] = 0U;
        }
        for (size_t i = 0; i < sectionNum * aivNum * METADATA_STRIDE; ++i) {
            fdMetadata[i] = 0U;
        }
    }

    void SetHeadMetadata(uint32_t metaIdx, uint32_t val)
    {
        assert(metaIdx < METADATA_STRIDE);
        headMetadata[metaIdx] = val;
    }

    uint32_t GetHeadMetadata(uint32_t metaIdx)
    {
        assert(metaIdx < METADATA_STRIDE);
        return headMetadata[metaIdx];
    }

    void SetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicNum);
        assert(metaIdx < METADATA_STRIDE);
        size_t offset = static_cast<size_t>(sectionIdx) * static_cast<size_t>(aicNum * METADATA_STRIDE) +
                        static_cast<size_t>(aicIdx * METADATA_STRIDE);
        faMetadata[offset + static_cast<size_t>(metaIdx)] = val;
    }

    uint32_t GetFaMetadata(uint32_t sectionIdx, uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aicIdx < aicNum);
        assert(metaIdx < METADATA_STRIDE);
        size_t offset = static_cast<size_t>(sectionIdx) * static_cast<size_t>(aicNum * METADATA_STRIDE) +
                        static_cast<size_t>(aicIdx * METADATA_STRIDE);
        return faMetadata[offset + static_cast<size_t>(metaIdx)];
    }

    void SetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aicNum);
        assert(metaIdx < METADATA_STRIDE);
        size_t offset = static_cast<size_t>(sectionIdx) * static_cast<size_t>(aivNum * METADATA_STRIDE) +
                        static_cast<size_t>(aivIdx * METADATA_STRIDE);
        fdMetadata[offset + static_cast<size_t>(metaIdx)] = val;
    }

    uint32_t GetFdMetadata(uint32_t sectionIdx, uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(sectionIdx < sectionNum);
        assert(aivIdx < aivNum);
        assert(metaIdx < METADATA_STRIDE);
        size_t offset = static_cast<size_t>(sectionIdx) * static_cast<size_t>(aivNum * METADATA_STRIDE) +
                        static_cast<size_t>(aivIdx * METADATA_STRIDE);
        return fdMetadata[offset + static_cast<size_t>(metaIdx)];
    }
};
} // namespace detail

} // namespace optiling

// Shared deterministic FAG metadata fields and descriptor-only workspace rules.
// Actual cu-dependent scheduling belongs to flash_attn_metadata_aicpu.cpp.
namespace optiling::fag_deter {

constexpr int64_t PLAN_MAGIC = 0x46414744504C414ELL;
constexpr int64_t PLAN_VERSION = 5;
// Shared L2 planning estimate for the swizzle heuristic, not a hardware capacity constraint.
constexpr uint64_t PLAN_L2_ESTIMATE_BYTES = 134217728;
// Fixed Ascend950 usable UB capacity. Core counts remain invocation-specific.
constexpr int64_t ASCEND950_UB_BYTES = 253952;
constexpr int64_t MAX_CORES = 36;
constexpr int64_t PREFIX_CAPACITY = 132;
constexpr int64_t SWIZZLE_PREFIX_CAPACITY = 129;

// These are wire values, not the different ASC SplitAxisEnum values.
enum class Split : int64_t {
    GS = 0,
    BN2 = 1,
    BN2S2 = 3
};
// Metadata has no Q tensor. Both supported input types have identical schedule
// and workspace geometry; F16_ANY denotes that shared capability, not a tensor dtype.
enum class DType : int64_t {
    FP16 = 0,
    BF16 = 1,
    F16_ANY = 2
};
constexpr int64_t SUPPORTED_DTYPE_MASK = 3; // bit 0: fp16, bit 1: bf16
enum class PlanStatus : int64_t {
    INVALID = 0,
    READY = 1
};
enum class WriteFormat : int64_t {
    UNKNOWN = -1,
    ND = 0,
    NZ = 1
};
enum class Schedule : int64_t {
    DENSE = 0,
    CAUSAL = 1,
    BAND = 2,
    LINE = 3,
    DENSE_SWIZZLE = 4
};
enum class Fallback : int64_t {
    NONE = 0,
    SUPPORT_BN2S2 = 1,
    SYNC_COST = 2,
    UNSATURATED = 4,
    CAUSAL = 8,
    LINE = 16
};

// Wire fields use signed 64-bit words; keep the matching offsets in fag_deter.py.
// Append fields only; incompatible changes require a version increment.
enum class Field : size_t {
    magic,
    version,
    byteSize,
    status,
    compiledFamily,
    initialSplit,
    logicalSplit,
    fallbackReasons,
    batch,
    totalQ,
    totalKv,
    maxQ,
    maxKv,
    headsQ,
    headsKv,
    dimQ,
    dimV,
    alignQ,
    alignV,
    supportedDtypeMask,
    maskMode,
    windowLeft,
    windowRight,
    sink,
    aicNum,
    aivNum,
    l2Bytes,
    reservedPlanMode, // Zero: plan does not depend on runtime deterministic level.
    effectiveLayout,
    sparseType,
    deterType,
    s1Token,
    s2Token,
    s1Inner,
    s2Inner,
    s1CvRatio,
    s2CvRatio,
    scheduleKind,
    isSwizzle,
    lineDeter,
    coreDivide,
    maxRound,
    blockOuter,
    blockFactor,
    prefixStep,
    tailZeroCount,
    hasZeroSequence,
    invalidRow,
    invalidCol,
    ascWriteFormat,
    writeFormat,
    workspaceBytes,
    dqOffset,
    dqBytes,
    dkOffset,
    dkBytes,
    dvOffset,
    dvBytes,
    scratchStride,
    specialOffset,
    specialBytes,
    specialCount,
    enablePreSfmg,
    sfmgOffset,
    sfmgBytes,
    sfmgRows,
    sfmgStride,
    sinkOffset,
    sinkBytes,
    sinkS1Outer,
    sinkS2Outer,
    sinkReduceAxis,
    qPreBlockFactor,
    qPreBlockTotal,
    kPreBlockFactor,
    kPreBlockTotal,
    vPreBlockFactor,
    vPreBlockTotal,
    qPostBlockFactor,
    qPostBlockTotal,
    kPostBlockFactor,
    kPostBlockTotal,
    vPostBlockFactor,
    vPostBlockTotal,
    blockStarts,
    blockEnds,
    startBatch,
    startNeedSyncRound,
    endNeedSyncRound,
    separateDkOffset,
    deterPrefix,
    deterPrefixAlign,
    deterPrefix0,
    deterPrefix1,
    deterPrefix2,
    swizzleAreaPrefix,
    swizzleAlignAreaPrefix,
    swizzleRoundPrefix,
    libapiBytes,
    ubBytes,
    maskPresent,
    maskShapeType,
    sValueZeroUnderTnd,
    s1Outer,
    s2Outer,
    sfmgUsedCoreNum,
    sfmgNormalCoreRows,
    sfmgTailCoreRows,
    sfmgLoopRows,
    sfmgDyBufferLen,
    sfmgOutputBufferLen,
    postBaseElements,
    sinkPreBlockFactor,
    sinkPreBlockTotal,
    sinkPostBlockFactor,
    sinkPostBlockTotal,
    COUNT
};

// Element counts in Field order. Array boundaries are checked below.
constexpr std::array<size_t, static_cast<size_t>(Field::COUNT)> FIELD_COUNTS = {{
    1,   // magic
    1,   // version
    1,   // byteSize
    1,   // status
    1,   // compiledFamily
    1,   // initialSplit
    1,   // logicalSplit
    1,   // fallbackReasons
    1,   // batch
    1,   // totalQ
    1,   // totalKv
    1,   // maxQ
    1,   // maxKv
    1,   // headsQ
    1,   // headsKv
    1,   // dimQ
    1,   // dimV
    1,   // alignQ
    1,   // alignV
    1,   // supportedDtypeMask
    1,   // maskMode
    1,   // windowLeft
    1,   // windowRight
    1,   // sink
    1,   // aicNum
    1,   // aivNum
    1,   // l2Bytes
    1,   // reservedPlanMode
    1,   // effectiveLayout
    1,   // sparseType
    1,   // deterType
    1,   // s1Token
    1,   // s2Token
    1,   // s1Inner
    1,   // s2Inner
    1,   // s1CvRatio
    1,   // s2CvRatio
    1,   // scheduleKind
    1,   // isSwizzle
    1,   // lineDeter
    1,   // coreDivide
    1,   // maxRound
    1,   // blockOuter
    1,   // blockFactor
    1,   // prefixStep
    1,   // tailZeroCount
    1,   // hasZeroSequence
    1,   // invalidRow
    1,   // invalidCol
    1,   // ascWriteFormat
    1,   // writeFormat
    1,   // workspaceBytes
    1,   // dqOffset
    1,   // dqBytes
    1,   // dkOffset
    1,   // dkBytes
    1,   // dvOffset
    1,   // dvBytes
    1,   // scratchStride
    1,   // specialOffset
    1,   // specialBytes
    1,   // specialCount
    1,   // enablePreSfmg
    1,   // sfmgOffset
    1,   // sfmgBytes
    1,   // sfmgRows
    1,   // sfmgStride
    1,   // sinkOffset
    1,   // sinkBytes
    1,   // sinkS1Outer
    1,   // sinkS2Outer
    1,   // sinkReduceAxis
    1,   // qPreBlockFactor
    1,   // qPreBlockTotal
    1,   // kPreBlockFactor
    1,   // kPreBlockTotal
    1,   // vPreBlockFactor
    1,   // vPreBlockTotal
    1,   // qPostBlockFactor
    1,   // qPostBlockTotal
    1,   // kPostBlockFactor
    1,   // kPostBlockTotal
    1,   // vPostBlockFactor
    1,   // vPostBlockTotal
    36,  // blockStarts
    36,  // blockEnds
    36,  // startBatch
    36,  // startNeedSyncRound
    36,  // endNeedSyncRound
    36,  // separateDkOffset
    132, // deterPrefix
    132, // deterPrefixAlign
    132, // deterPrefix0
    132, // deterPrefix1
    132, // deterPrefix2
    129, // swizzleAreaPrefix
    129, // swizzleAlignAreaPrefix
    129, // swizzleRoundPrefix
    1,   // libapiBytes
    1,   // ubBytes
    1,   // maskPresent
    1,   // maskShapeType
    1,   // sValueZeroUnderTnd
    1,   // s1Outer
    1,   // s2Outer
    1,   // sfmgUsedCoreNum
    1,   // sfmgNormalCoreRows
    1,   // sfmgTailCoreRows
    1,   // sfmgLoopRows
    1,   // sfmgDyBufferLen
    1,   // sfmgOutputBufferLen
    1,   // postBaseElements
    1,   // sinkPreBlockFactor
    1,   // sinkPreBlockTotal
    1,   // sinkPostBlockFactor
    1,   // sinkPostBlockTotal
}};

constexpr size_t WordOffset(Field field)
{
    size_t result = 0;
    for (size_t i = 0; i < static_cast<size_t>(field); ++i) {
        result += FIELD_COUNTS[i];
    }
    return result;
}

constexpr size_t PLAN_WORDS = WordOffset(Field::COUNT);
constexpr size_t PLAN_BYTES = PLAN_WORDS * sizeof(int64_t);
using PlanWords = std::array<int64_t, PLAN_WORDS>;

inline int64_t& At(PlanWords& plan, Field field, size_t index = 0)
{
    return plan[WordOffset(field) + index];
}

inline int64_t At(const PlanWords& plan, Field field, size_t index = 0)
{
    return plan[WordOffset(field) + index];
}

inline void InitializePlan(PlanWords& plan)
{
    plan.fill(0);
    At(plan, Field::magic) = PLAN_MAGIC;
    At(plan, Field::version) = PLAN_VERSION;
    At(plan, Field::byteSize) = static_cast<int64_t>(PLAN_BYTES);
    At(plan, Field::status) = static_cast<int64_t>(PlanStatus::INVALID);
    for (size_t i = 0; i < static_cast<size_t>(MAX_CORES); ++i) {
        At(plan, Field::separateDkOffset, i) = -1;
    }
}

inline bool HasReadyHeader(const int64_t* words, size_t bytes)
{
    return words != nullptr && bytes >= PLAN_BYTES && words[WordOffset(Field::magic)] == PLAN_MAGIC &&
           words[WordOffset(Field::version)] == PLAN_VERSION &&
           words[WordOffset(Field::byteSize)] == static_cast<int64_t>(PLAN_BYTES) &&
           words[WordOffset(Field::status)] == static_cast<int64_t>(PlanStatus::READY) &&
           words[WordOffset(Field::compiledFamily)] == static_cast<int64_t>(Split::BN2S2) &&
           words[WordOffset(Field::initialSplit)] == static_cast<int64_t>(Split::BN2S2) &&
           (words[WordOffset(Field::logicalSplit)] == static_cast<int64_t>(Split::BN2S2) ||
            words[WordOffset(Field::logicalSplit)] == static_cast<int64_t>(Split::GS));
}

static_assert(PLAN_WORDS == 1365 && PLAN_BYTES == 10920, "Plan wire size must remain unchanged");
static_assert(WordOffset(Field::blockStarts) == 84, "Per-core fields must start at word 84");
static_assert(WordOffset(Field::deterPrefix) == 300, "Deterministic prefixes must start at word 300");
static_assert(WordOffset(Field::swizzleAreaPrefix) == 960, "Swizzle prefixes must start at word 960");
static_assert(WordOffset(Field::libapiBytes) == 1347, "Trailing scalar fields must start at word 1347");
static_assert(sizeof(PlanWords) == PLAN_BYTES, "Wire plan must have no object padding");
static_assert(sizeof(int64_t) == 8, "Wire words must have 64 bits");

struct RouteInputs {
    int64_t batch = 0;
    int64_t totalQ = 0;
    int64_t totalKv = 0;
    int64_t maxQ = -1;
    int64_t maxKv = -1;
    int64_t headsQ = 0;
    int64_t headsKv = 0;
    int64_t dimQ = 0;
    int64_t dimV = 0;
    DType dtype = DType::FP16;
    int64_t maskMode = 0;
    bool allLayoutsTnd = false;
    bool deterministic = false;
    bool hasSequsedQ = false;
    bool hasSequsedKv = false;
};

enum class Admission : int64_t {
    ELIGIBLE = 0,
    NOT_TND_DETER,
    DTYPE,
    HEADS,
    DIMS,
    MASK,
    SEQUSED,
    MISSING_MAX,
    INVALID_SHAPE,
    ARITHMETIC_OVERFLOW,
    ALL_SAME,
    POSSIBLE_BN2
};

inline bool CheckedMul(int64_t a, int64_t b, int64_t& product)
{
    if (a < 0 || b < 0 || (b != 0 && a > std::numeric_limits<int64_t>::max() / b)) {
        return false;
    }
    product = a * b;
    return true;
}

// This proves initial ASC BN2S2 eligibility; it does not predict its final split.
// Caller must enforce that explicitly supplied maxima are actual physical maxima.
// Source: ASC arch35 common_regbase.cpp SetSplitAxis / SupportTrans2BS2N2GD.
inline Admission Admit(const RouteInputs& p)
{
    if (!p.allLayoutsTnd || !p.deterministic) {
        return Admission::NOT_TND_DETER;
    }
    if (p.dtype != DType::FP16 && p.dtype != DType::BF16 && p.dtype != DType::F16_ANY) {
        return Admission::DTYPE;
    }
    if (p.headsQ <= 0 || p.headsQ != p.headsKv) {
        return Admission::HEADS;
    }
    if (p.dimQ <= 0 || p.dimQ > 512 || !(p.dimV == p.dimQ || (p.dimQ == 192 && p.dimV == 128))) {
        return Admission::DIMS;
    }
    if (p.maskMode != 0 && p.maskMode != 3 && p.maskMode != 4) {
        return Admission::MASK;
    }
    if (p.hasSequsedQ || p.hasSequsedKv) {
        return Admission::SEQUSED;
    }
    if (p.maxQ <= 0 || p.maxKv <= 0) {
        return Admission::MISSING_MAX;
    }
    if (p.batch <= 0 || p.totalQ < p.maxQ || p.totalKv < p.maxKv) {
        return Admission::INVALID_SHAPE;
    }
    int64_t paddedQ = 0;
    int64_t paddedKv = 0;
    if (!CheckedMul(p.batch, p.maxQ, paddedQ) || !CheckedMul(p.batch, p.maxKv, paddedKv)) {
        return Admission::ARITHMETIC_OVERFLOW;
    }
    if (p.totalQ > paddedQ || p.totalKv > paddedKv) {
        return Admission::INVALID_SHAPE;
    }
    if (p.totalQ == paddedQ && p.totalKv == paddedKv) {
        return Admission::ALL_SAME;
    }
    if (p.maxQ <= 128 && p.maxKv <= 128 && p.dimQ <= 128) {
        return Admission::POSSIBLE_BN2;
    }
    return Admission::ELIGIBLE;
}

constexpr int64_t AlignHead(int64_t dim)
{
    return dim <= 0 || dim > 512 ? 0 : dim <= 64 ? 64 : dim <= 128 ? 128 : dim <= 192 ? 192 : dim <= 256 ? 256 : 768;
}

// PR15483 admission, shared by the FAG host and metadata producer.
// Callers decide supported compiled family. No per-batch empty-length gate.
// Plan v4: head-major packed-token SFMG cache, fixed DB windows.
// Worker row intervals remain metadata-owned; the kernel clips each window
// to both its worker interval and its current head's end.
constexpr int64_t PreSfmgWindowRows(int64_t alignV)
{
    return alignV == 128 ? 208 : alignV == 192 ? 136 : (alignV == 256 || alignV == 768) ? 104 : 0;
}

constexpr int64_t PreSfmgInputBytes(int64_t alignV)
{
    const int64_t rows = PreSfmgWindowRows(alignV) + (alignV == 192 ? 1 : 0);
    const int64_t width = alignV > 256 ? 256 : alignV;
    return (rows * width * 2 + 511) / 512 * 512;
}

constexpr int64_t PreSfmgOutputBytes(int64_t alignV)
{
    return (PreSfmgWindowRows(alignV) * 4 + 511) / 512 * 512;
}

inline bool AllPackedBatchesHaveRows(int64_t total, int64_t maximum, int64_t batch)
{
    // total > (batch - 1) * maximum forces every packed batch to contain a row.
    return total > 0 && maximum > 0 && batch > 0 && (total - 1) / maximum >= batch - 1;
}

inline int64_t PreSfmgSmallWorkWidth(const RouteInputs& r)
{
    if (r.allLayoutsTnd && r.dimQ == 192 && r.dimV == 128 && !r.hasSequsedQ && !r.hasSequsedKv &&
        AllPackedBatchesHaveRows(r.totalQ, r.maxQ, r.batch) && AllPackedBatchesHaveRows(r.totalKv, r.maxKv, r.batch)) {
        return r.dimQ;
    }
    return r.dimV;
}

inline bool DecidePreSfmg(const RouteInputs& r, int64_t alignV, bool sink, int64_t maxKv, int64_t s1Outer)
{
    if (sink || r.dimV <= 64 || r.dimV > 768 || alignV < 128) {
        return false;
    }
    // Only the comparison with1024 matters. Saturation avoids signed overflow
    // on already-positive public shape values without changing that predicate.
    const int64_t blocks =
        std::min<int64_t>(r.batch, 1024) * std::min<int64_t>(r.headsQ, 1024) * std::min<int64_t>(s1Outer, 1024);
    return !(PreSfmgSmallWorkWidth(r) <= 128 && maxKv <= 256 && blocks >= 1024);
}

constexpr bool CanCompileDirectStore(int64_t alignQ, int64_t alignV)
{
    return (alignQ == 64 && alignV == 64) || (alignQ == 128 && alignV == 128);
}

inline bool CanCommitDirect(Split logicalSplit, int64_t alignQ, int64_t alignV, bool wholeColumnOwned,
                            bool specialColumn, bool residentUntilFinal, bool resourceLifetimeValid)
{
    return logicalSplit == Split::BN2S2 && CanCompileDirectStore(alignQ, alignV) && wholeColumnOwned &&
           !specialColumn && residentUntilFinal && resourceLifetimeValid;
}

struct ScheduleInputs {
    RouteInputs route;
    int64_t aicNum = 0;
    int64_t aivNum = 0;
    int64_t windowLeft = -1;
    int64_t windowRight = -1;
    bool hasMask = false;
    uint64_t l2Bytes = 0;
    // ASC mask shape enum: 0=BN1S1S2, 1=BS1S2, 2=S1S2, 99=TT.
    uint32_t maskShapeType = 0;
};

struct PlanInputs {
    ScheduleInputs schedule;
    int64_t ubBytes = 0;
    bool sink = false;
    // Supplied by the reference/format integration, not inferred by the ND planner.
    WriteFormat ascWriteFormat = WriteFormat::UNKNOWN;
};

struct Region {
    int64_t offset = 0;
    int64_t bytes = 0;
};

struct WorkspaceLayout {
    Region dq, dk, dv, sfmg, sink, special;
    int64_t scratchStride = 0;
    int64_t specialHalfBytes = 0;
    int64_t bytes = 0;
};

namespace workspace_detail {
constexpr int64_t GM_ALIGN = 512;
constexpr int64_t RESERVED_BYTES = 64 * 1024;
constexpr int64_t TAIL_BYTES = 20 * 1024 * 1024;

struct Arithmetic {
    bool valid = true;
    int64_t Mul(int64_t a, int64_t b)
    {
        int64_t value = 0;
        valid = CheckedMul(a, b, value) && valid;
        return value;
    }
    int64_t Add(int64_t a, int64_t b)
    {
        if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b) {
            valid = false;
            return 0;
        }
        return a + b;
    }
    int64_t Align(int64_t n, int64_t alignment)
    {
        return Add(n, alignment - 1) / alignment * alignment;
    }
    // ASC adds a full GM_ALIGN, including for already-aligned regions. Preserve
    // that guard padding; replacing this by ordinary round-up changes offsets.
    int64_t Guard(int64_t bytes)
    {
        return Add(bytes, GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    }
};
} // namespace workspace_detail

inline bool MakeWorkspace(const PlanInputs& in, Split split, bool preSfmg, WorkspaceLayout& layout)
{
    using namespace workspace_detail;
    layout = WorkspaceLayout{};
    const auto& r = in.schedule.route;
    if (Admit(r) != Admission::ELIGIBLE || (split != Split::BN2S2 && split != Split::GS)) {
        return false;
    }
    Arithmetic math;
    const int64_t qRows = math.Align(math.Mul(r.totalQ, r.headsQ), 128);
    const int64_t kvRows = math.Align(math.Mul(r.totalKv, r.headsKv), 128);
    const int64_t dAlign = AlignHead(r.dimQ);
    // Source: GetWorkspaceSize + GetWorkspaceSize4Deter (TND ND fp16/bf16).
    // Both ASC and the generated PTO binary entry call GetUserWorkspace before
    // dispatching the implementation. Adding libapi here would count it twice.
    int64_t cursor = RESERVED_BYTES;
    auto append = [&](Region& region, int64_t bytes) {
        region.offset = cursor;
        region.bytes = math.Guard(bytes);
        cursor = math.Add(cursor, region.bytes);
    };
    append(layout.dq, math.Mul(math.Mul(qRows, r.dimQ), 4));
    if (split == Split::BN2S2) {
        layout.scratchStride = math.Mul(128 * 4, dAlign);
        const int64_t scratch = math.Mul(layout.scratchStride, MAX_CORES);
        append(layout.dk, scratch);
        append(layout.dv, scratch);
    } else {
        append(layout.dk, math.Mul(math.Mul(kvRows, r.dimQ), 4));
        append(layout.dv, math.Mul(math.Mul(kvRows, r.dimV), 4));
    }
    if (preSfmg) {
        // Deliberately the ASC max-shape reserve, even though its TND pre-kernel
        // computes totalQ * headsQ rows. Do not silently replace it by totalQ.
        const int64_t bn = math.Mul(r.batch, r.headsQ);
        const int64_t rows = math.Add(math.Mul(bn - 1, r.maxQ), math.Align(r.maxQ, 128));
        append(layout.sfmg, math.Mul(rows, 8 * 4));
    }
    if (in.sink) {
        const int64_t s1 = (r.maxQ + 127) / 128;
        const int64_t s2 = (r.maxKv + 127) / 128;
        const int64_t elements = math.Mul(math.Mul(math.Mul(r.batch, r.headsQ), s1 * 2), s2);
        append(layout.sink, math.Mul(elements, 4));
    }
    if (split == Split::BN2S2) {
        layout.special.offset = cursor;
        layout.specialHalfBytes = math.Guard(math.Mul(math.Mul(128 * 4, dAlign), MAX_CORES));
        layout.special.bytes = math.Mul(layout.specialHalfBytes, 2);
        cursor = math.Add(cursor, layout.special.bytes);
    }
    layout.bytes = math.Add(cursor, TAIL_BYTES);
    return math.valid;
}

inline bool WorkspaceCapacity(const PlanInputs& in, int64_t& bytes)
{
    bytes = 0;
    WorkspaceLayout bn, gs;
    // Every actual preSfmg layout is bounded by enabling its reserve. Sink
    // excludes preSfmg in ASC, so those two regions never need simultaneous space.
    const bool reserveSfmg = !in.sink && in.schedule.route.dimV > 64;
    if (!MakeWorkspace(in, Split::BN2S2, reserveSfmg, bn) || !MakeWorkspace(in, Split::GS, reserveSfmg, gs)) {
        return false;
    }
    bytes = std::max(bn.bytes, gs.bytes);
    return true;
}

// head[8..9] belongs to FA output initialization, head[12..15] to the TND
// extension. The compact FAG directory uses only head[10..11]: its size is
// fixed and its payload already carries the version/status. Offsets remain
// int32 elements; payload layout/alignment are unchanged. Directory v3 must
// be paired with its matching FAG consumer (the payload remains version 2).
constexpr uint32_t METADATA_HEAD_OFFSET_INDEX = 10;
constexpr uint32_t METADATA_HEAD_MAGIC_INDEX = 11;
constexpr uint32_t METADATA_HEAD_MAGIC = 0x46414733U;
static_assert(METADATA_HEAD_OFFSET_INDEX > HEAD_UNIFORM_Q_LEN_INDEX && METADATA_HEAD_MAGIC_INDEX < 12,
              "FAG directory must not overlap FA initialization or TND directory fields");
constexpr int64_t METADATA_MAGIC = 0x4641474D45544132LL;
constexpr int64_t METADATA_VERSION = 2;
constexpr size_t METADATA_HEADER_WORDS = 16;
constexpr size_t METADATA_ARRAY_BEGIN = WordOffset(Field::blockStarts);
constexpr size_t METADATA_ARRAY_END = WordOffset(Field::libapiBytes);
constexpr size_t METADATA_ARRAY_WORDS = METADATA_ARRAY_END - METADATA_ARRAY_BEGIN;
constexpr size_t METADATA_SCALAR_WORDS = PLAN_WORDS - METADATA_ARRAY_WORDS;
constexpr size_t METADATA_SINK_VARIANTS = 2;
constexpr size_t METADATA_BANK_COUNT = 2;
constexpr size_t METADATA_BANK_WORDS = METADATA_ARRAY_WORDS + METADATA_SINK_VARIANTS * METADATA_SCALAR_WORDS;
constexpr size_t METADATA_BYTES =
    ((METADATA_HEADER_WORDS + METADATA_BANK_COUNT * METADATA_BANK_WORDS) * 8 + 63) / 64 * 64;
constexpr size_t METADATA_I32_WORDS = METADATA_BYTES / sizeof(int32_t);
using MetadataWords = std::array<int64_t, METADATA_BYTES / sizeof(int64_t)>;
static_assert(METADATA_ARRAY_BEGIN == 84 && METADATA_ARRAY_END == 1347 && METADATA_SCALAR_WORDS == 102,
              "Update the metadata consumer when the logical plan layout changes");
static_assert(METADATA_BYTES == 23616, "Metadata allocator and consumer must share extension capacity");

} // namespace optiling::fag_deter

#endif
