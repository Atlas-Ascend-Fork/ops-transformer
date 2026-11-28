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
 * \file flash_attn_metadata_aicpu.h
 * \brief
 */

#ifndef FLASH_ATTN_METADATA_AICPU_H
#define FLASH_ATTN_METADATA_AICPU_H

#include <string>
#include <cstring>
#include <vector>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <set>
#include <tuple>
#include "cpu_context.h"
#include "cpu_kernel.h"
#include "cpu_tensor.h"
#include "flash_attn_metadata.h"
#include "../../common/op_kernel/load_balance/section_stream_k/section_stream_k.h"

// Deterministic FAG planning is private to the existing metadata AICPU kernel.
namespace optiling::fag_deter {

enum class ScheduleError {
    OK,
    ADMISSION,
    CORE_COUNT,
    CU_SIZE,
    CU_ORIGIN,
    CU_ORDER,
    CU_TOTAL,
    CU_MAX,
    BATCH_CAPACITY,
    ARITHMETIC_OVERFLOW,
    WINDOW,
    L2_CAPACITY
};

// ASC names and types are retained in this isolated namespace to make each
// imported expression reviewable. No gert/ACL tensor is accessed by this layer.
namespace asc {
constexpr uint32_t CORE_LIST_NUM = 36;
constexpr size_t DETER_PREFIX_ARRAY_SIZE = 132;
constexpr uint32_t INPUT_FORMAT_TND = 4;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t NUM_THREE = 3;
constexpr int64_t ALIGN64 = 64;
constexpr int64_t COMPRESS_ATTEN_MASK_SIZE = 2048 * 2048;
constexpr int64_t OUTINDEX = -1;
constexpr int64_t TND_DETER_SWIZZLE_MIN_SLOT_UTILIZATION_PERCENT = 90;
constexpr int64_t TND_DETER_SWIZZLE_MAX_ROUND_GROWTH_PERCENT = 3;
constexpr int64_t TND_DETER_SWIZZLE_PERCENT_BASE = 100;
constexpr uint32_t S1CV_RATIO_DEFAULT = 2, S2CV_RATIO_DEFAULT = 1, EMPTY_TENSOR = 0;
constexpr uint32_t TND_SWIZZLE_PREFIX_NUM = 129, TOTAL_BLOCK_DIMENSION = 2, CALCULATED_BLOCK_DIMENSION = 4;
constexpr uint32_t BEGIN_IDX = 0, END_IDX = 1, SUM_S1S2 = 2, SUM_ALL = 3, MULT_BASE = 2;
constexpr int32_t FP16_BYTES = 2, FP32_BYTES = 4;
constexpr uint32_t INPUT_FORMAT_BN2GS2D = 3, INPUT_FORMAT_S2BN2GD = 2, INPUT_FORMAT_BS2N2GD = 1;
enum class ConstAxisTemplateNum {
    NUM16 = 16
};
enum class SparseMode : uint32_t {
    NO_MASK = 0,
    ALL_MASK = 1,
    LEFT_UP_CAUSAL = 2,
    RIGHT_DOWN_CAUSAL = 3,
    BAND = 4,
    PREFIX = 5,
    PREFIX_COMPRESS = 6,
    RIGHT_DOWN_CASUAL_BAND = 7,
    BAND_LEFT_UP_CASUAL = 8
};
enum class SparseType : uint8_t {
    DENSE = 0,
    CASUAL = 1,
    BAND = 2,
    UNSUPPORTED = 3
};
enum class AttenMaskShapeType : uint32_t {
    ATTENMASKBN2GS1S2 = 0,
    ATTENMASKBS1S2 = 1,
    ATTENMASKS1S2 = 2
};
namespace ge {
using graphStatus = int;
constexpr graphStatus GRAPH_SUCCESS = 0;
} // namespace ge
enum class DeterSparseType : uint32_t {
    DETER_DENSE = 2,
    DETER_CAUSAL = 3,
    DETER_BAND = 4
};
enum class SplitAxisEnum {
    BN2GS1S2,
    BN2S2
};
struct DeterParamRegbase {
    static constexpr uint8_t SCHEDULE_DEFAULT = 0;
};

struct DeterPrefixData {
    std::vector<int64_t> prefix0{0}, prefix1{0}, prefix2{0}, deterPrefix{0}, deterPrefixAlign{0};
    std::vector<int64_t> mNewList, nNewList, pNewList, qNewList;
};

struct TndBandDeterRoundInfo {
    int64_t lastBatchId = 0;
    uint64_t lastValidRound = 0;
    uint64_t coreFirstBatchLastRound = 0;
    uint64_t coreLastBatchStartRound = 0;
};

struct FuzzyBaseInfoParamsRegbase {
    uint64_t aicNum = 0;
    uint64_t l2CacheSize = 0;
    int64_t b = 0, s1 = 0, s2 = 0, n1 = 0, n2 = 0, g = 1, d = 0;
    // ASC 128x128 cube tile is 64x128 per AIV, with S1CV_RATIO_DEFAULT=2.
    int64_t s1Inner = 64, s2Inner = 128;
    uint32_t s1CvRatio = 2, s2CvRatio = 1;
    uint32_t layoutType = INPUT_FORMAT_TND;
    uint32_t sparseMode = 0, deterSparseType = 2;
    uint8_t sparseType = 0;
    int64_t s1CvInner = 128, cvS2Inner = 128, s1Outer = 0, s2Outer = 0;
    uint32_t attenMaskOptional = 0, attenMaskShapeType = 0, bandIdx = 0;
    uint32_t prefixN[2048]{}; // Unreachable PREFIX branch retained in imported source.
    uint32_t blockOuter = 0;
    int64_t blockFactor = 0, maxValidBBLen = 0;
    int64_t blockStarts[CORE_LIST_NUM]{}, blockEnds[CORE_LIST_NUM]{};
    bool isBn2MultiBlk = false, isInvalidCol = false, isInvalidRow = false, sValueZeroUnderTND = false;
    bool isSparse = false, enableSwizzle = false, isSwizzle = false;
    int64_t s1Token = INT32_MAX, s2Token = INT32_MAX;
    bool isS1S2Same = false;
    std::vector<int64_t> actualSeqQlen, actualSeqKvlen;
    SplitAxisEnum splitAxis = SplitAxisEnum::BN2S2;
    int64_t deterMaxRound = 0, deterPrefixThreshold = 128, deterPrefixStep = 1;
    uint8_t tndLineDeter = 0, tndGqaScheduleMode = 0, tndGqaHeadPairGroup = 0;
    int64_t deterColStart[CORE_LIST_NUM + 1]{};
    int64_t deterPrefix[DETER_PREFIX_ARRAY_SIZE]{};
    int64_t deterPrefixAlign[DETER_PREFIX_ARRAY_SIZE]{};
    int64_t deterPrefix0[DETER_PREFIX_ARRAY_SIZE]{};
    int64_t deterPrefix1[DETER_PREFIX_ARRAY_SIZE]{};
    int64_t deterPrefix2[DETER_PREFIX_ARRAY_SIZE]{};
    uint64_t startNeedSyncRound[CORE_LIST_NUM]{};
    uint64_t endNeedSyncRound[CORE_LIST_NUM]{};
    int64_t separateDkOffset[CORE_LIST_NUM]{};
};

struct TndBaseInfo {
    uint64_t tndSwizzleS1S2PrefixSum[TND_SWIZZLE_PREFIX_NUM]{};
    uint64_t tndSwizzleS1S2AlignPrefixSum[TND_SWIZZLE_PREFIX_NUM]{};
    uint64_t tndS2BlockPrefixSum[TND_SWIZZLE_PREFIX_NUM]{};
    uint64_t tndStartBIdx[CORE_LIST_NUM]{}, tndS1S2PrefixSum[CORE_LIST_NUM]{};
    uint64_t tndS1S2AlignPrefixSum[CORE_LIST_NUM]{}, tndPrefixSum[CORE_LIST_NUM]{};
    uint64_t normalMaxValidBlockCount = 0;
    bool isS1GreaterThanS2 = true, isS1LessThanS2 = true, isSeqExistZero = false;
};

// Exact arithmetic conventions used by the ASC source, including signed divide.
template <class T>
inline T CeilDivideBy(T num1, T num2)
{
    return num2 == 0 ? 0 : (num1 + num2 - 1) / num2;
}
template <class T>
inline T AlignTo(T n, T size)
{
    return size == 0 ? 0 : (n + size - 1) & (~(size - 1));
}
inline int64_t AbsCeil(int64_t a, int64_t b)
{
    const int64_t result = (std::abs(a) + std::abs(b) - 1) / std::abs(b);
    return a < 0 || b < 0 ? -result : result;
}
inline int64_t Gcd(int64_t a, int64_t b)
{
    while (b > 0) {
        const int64_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}
template <class T>
inline std::vector<T> SliceVector(const std::vector<T>& arr, int64_t step)
{
    if (step == 1) {
        return arr;
    }
    std::vector<T> result;
    for (size_t i = 0; i < arr.size(); i += static_cast<size_t>(step)) {
        result.push_back(arr[i]);
    }
    return result;
}
inline void CalcleActualToken(FuzzyBaseInfoParamsRegbase& p, int64_t batch, int64_t& qToken, int64_t& kvToken)
{
    qToken = p.s1Token;
    kvToken = p.s2Token;
    if (p.sparseMode == 3 || p.sparseMode == 4) {
        qToken += p.actualSeqQlen[batch] - p.actualSeqKvlen[batch];
        kvToken += p.actualSeqKvlen[batch] - p.actualSeqQlen[batch];
    }
}

struct Scheduler {
    FuzzyBaseInfoParamsRegbase fBaseParams;
    TndBaseInfo tndBaseInfo;
    std::vector<int64_t> kvEnds;
    int64_t fallbackReasons = 0;

    bool CheckSparseLeftAndRight(int64_t s1, int64_t left, int64_t right, int64_t batch)
    {
        // Exactly the TND branch of the ASC normal implementation.
        return CheckUnpadSparseLeftAndRight(s1, left, right, batch);
    }
    bool IsValid(int64_t block)
    {
        return IsValidUnpad(block);
    }

    void CalcleTNDDeterParam();

    void CalcleTNDDenseDeterParam();

    void CalcleTNDCausalDeterParam();

    void CalcleTNDBandDeterParam();

    void CalcleTNDDenseBns2DeterParam(DeterPrefixData& deterPrefixData);

    void CalcleTNDCausalDeterPrefix(DeterPrefixData& deterPrefixData, int64_t& m0Max, int64_t& m1Max, int64_t& m2Max);

    void CalcleTNDCausalDeterParamNormal(DeterPrefixData& deterPrefixData, const int64_t m0Max, const int64_t m1Max,
                                         const int64_t m2Max);

    void CalcleTNDBandDeterPrefix(DeterPrefixData& deterPrefixData, int64_t N11, int64_t& mnMax);

    void CalcleTNDBandBns2DeterParam(DeterPrefixData& deterPrefixData);

    void CalcleTNDDenseDeterSplitDkOffset(DeterPrefixData& deterPrefixData,
                                          std::vector<std::pair<uint64_t, uint64_t>>& syncRounds,
                                          std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges);

    bool SupportTNDBns2(DeterPrefixData& deterPrefixData, int64_t round);

    void CalcleTNDBandDeterSplitDkOffset(DeterPrefixData& deterPrefixData,
                                         std::vector<std::pair<uint64_t, uint64_t>>& syncRounds,
                                         std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges);

    template <const uint32_t deterSparseType>
    std::tuple<int64_t, int64_t, int64_t> CalTNDDenseIndex(DeterPrefixData& deterPrefixData, int64_t coreId,
                                                           int64_t roundId, int64_t N1)
    {
        int64_t unPadRoundMax{deterPrefixData.prefix1[fBaseParams.b + 1]}, ID{(coreId - 1) * unPadRoundMax + roundId},
            w{0};
        while (w < fBaseParams.b && ID > deterPrefixData.prefix1[w + 1] * N1) {
            w += 1;
        }
        int64_t delta = ID - deterPrefixData.prefix1[w] * N1;

        if (w >= fBaseParams.b) {
            return std::make_tuple(-1, -1, -1);
        }

        int64_t m{deterPrefixData.mNewList[w]}, n{deterPrefixData.nNewList[w]}, p, q;
        if constexpr (deterSparseType == static_cast<uint32_t>(DeterSparseType::DETER_BAND)) {
            p = deterPrefixData.pNewList[w];
            q = deterPrefixData.qNewList[w];
            if (p + q <= m) {
                if (n >= m) {
                    n = p + q - 1;
                } else {
                    m = p + q - 1;
                }
            } else {
                if (p + q <= n) {
                    n = p + q - 1;
                }
            }
        }

        int64_t currentBaseNum = m * n;
        int64_t batchId = w + 1;
        int64_t deltaN = (delta - 1) / currentBaseNum + 1;
        delta = delta % currentBaseNum;
        delta = delta != 0 ? delta : currentBaseNum;

        int64_t g = Gcd(m, unPadRoundMax);
        if (g == 0) {
            return std::make_tuple(-1, -1, -1);
        }
        int64_t t1{unPadRoundMax / g}, t2{m / g}, x{((delta - 1) % m) + 1}, y{(delta - 1) / m + 1};
        if (t1 < n) {
            int64_t n1 = n % t1;
            n1 = n1 == 0 ? t1 : n1;
            if (y <= n - n1) {
                int64_t deltaAdj = CeilDivideBy(y, t1);
                delta += deltaAdj;
                if (delta > deltaAdj * t2 * unPadRoundMax) {
                    delta -= t2 * unPadRoundMax;
                }
                x = ((delta - 1) % m) + 1;
                y = (delta - 1) / m + 1;
            }
        }
        return std::make_tuple((batchId - 1) * N1 + deltaN, x, y);
    }

    void UpdateSeparateDkOffsetLargeM(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
        TndBandDeterRoundInfo& tndBandDeterRoundInfo);

    void UpdateSeparateDkOffsetSmallM(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
        TndBandDeterRoundInfo& tndBandDeterRoundInfo);

    void UpdateSeparateDkOffset(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo,
        TndBandDeterRoundInfo& tndBandDeterRoundInfo);

    bool IsSeparateS2(
        std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>& coordinateInfo);

    int64_t GetKeyOffset(const int64_t* kvValue, int64_t w, int64_t y);

    void CalcleTNDDeterSyncRounds(std::vector<std::pair<uint64_t, uint64_t>>& syncRounds,
                                  std::vector<std::pair<uint64_t, uint64_t>>& syncRoundRanges);

    std::vector<uint64_t> CalculateSyncRound(std::vector<std::pair<uint64_t, uint64_t>> syncRounds);

    void SetCoreRoundInfo(TndBandDeterRoundInfo& tndBandDeterRoundInfo, uint64_t round, int64_t batchId);

    void CalcTNDSwizzleParam();

    bool GetBlockInfoOfBNS4TND();

    float BinarySearchMaxBlockNumPerCore(const std::vector<std::vector<float>>& acturalBlockInfo);

    void FillColumnValidBlockCount(std::vector<std::vector<uint64_t>>& validBlockCount);

    bool CaclePerCoreBlockInfo(const std::vector<std::vector<int64_t>>& totalBlockInfo,
                               const std::vector<std::vector<float>>& acturalBlockInfo,
                               const std::vector<std::vector<uint64_t>>& validBlockCount,
                               const float maxBlockNumPerCore, int64_t (&blockStarts)[CORE_LIST_NUM],
                               int64_t (&blockEnds)[CORE_LIST_NUM]);

    void SetTndSwizzleParam(int64_t bIdx, int64_t s1OuterTmp, int64_t s2OuterTmp);

    bool IsPossible(const std::vector<std::vector<float>>& acturalBlockInfo, const float possibleMax);

    bool CheckUnpadSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx);

    bool IsValidUnpad(int64_t blockIdx);

    ge::graphStatus GetSparseUnpadBlockInfo();

    void FillBlockInfo(std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo,
                       std::vector<std::vector<int64_t>>& totalBlockInfo);

    void CalValidUnpadBlockInfo(int64_t batchIdx, std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo);

    void GetUnpadS1S2OuterIndex(int64_t& s1oIdx, int64_t& s2oIdx, int64_t gTail, int64_t bIdx,
                                std::vector<std::vector<std::vector<int64_t>>>& calculatedBlockInfo);

    void FillBlockInfoLoadBalance(std::vector<std::vector<int64_t>>& totalBlockInfo,
                                  std::vector<std::vector<float>>& acturalBlockInfo);

    uint8_t GetSparseType();

    bool CheckExceedL2Cache();
};
} // namespace asc

struct ScheduleResult {
    asc::FuzzyBaseInfoParamsRegbase state;
    asc::TndBaseInfo tnd;
    int64_t fallbackReasons = 0;
    int64_t tailZeroCount = 0;
    bool hasZeroSequence = false;
};

// Implements CalcleDeterParam's TND arithmetic stage. The subsequent ASC
// saturation/L2/swizzle/pre/post stages must complete before a plan becomes READY.
ScheduleError BuildInitialSchedule(const ScheduleInputs& inputs, const int32_t* cuQ, size_t cuQSize,
                                   const int32_t* cuKv, size_t cuKvSize, ScheduleResult& result);

// Completes DoSparse's saturation decision and DoOpTiling's TND swizzle decision.
// Workspace/pre/post and wire validation remain separate stages.
ScheduleError BuildSchedule(const ScheduleInputs& inputs, const int32_t* cuQ, size_t cuQSize, const int32_t* cuKv,
                            size_t cuKvSize, ScheduleResult& result);

enum class PlanError {
    OK,
    SCHEDULE,
    RESOURCE,
    WORKSPACE,
    CONTRACT
};

struct PlanResult {
    PlanError error = PlanError::OK;
    ScheduleError scheduleError = ScheduleError::OK;
};

PlanResult BuildPlan(const PlanInputs& in, const int32_t* cuQ, size_t cuQSize, const int32_t* cuKv, size_t cuKvSize,
                     int64_t workspaceCapacity, PlanWords& plan);
// Reuse an ASC schedule for configurations that differ only in sink handling.
PlanResult BuildPlanFromSchedule(const PlanInputs& in, const ScheduleResult& schedule, int64_t workspaceCapacity,
                                 PlanWords& plan);
bool ValidatePlan(const PlanWords& plan, const PlanInputs& in, int64_t workspaceCapacity);

// Inputs contain actual sequence/resource context; mask presence and sink are
// selected later by FAG because neither tensor is an input of FlashAttnMetadata.
PlanResult BuildMetadataPlans(const PlanInputs& inputs, const int32_t* cuQ, size_t cuQSize, const int32_t* cuKv,
                              size_t cuKvSize, MetadataWords& output);

} // namespace optiling::fag_deter

namespace aicpu {

using optiling::FAG_CORE_LIST_NUM;
using optiling::FAG_ARRAY_LENGTH;
using optiling::FAG_SPLIT_AXIS_BN2GS1S2;
using optiling::FAG_SPARSE_NO_MASK;
using optiling::FAG_INPUT_FORMAT_BS2N2GD;

class FlashAttnMetadataCpuKernel : public CpuKernel {
public:
    FlashAttnMetadataCpuKernel() = default;
    ~FlashAttnMetadataCpuKernel() = default;
    uint32_t Compute(CpuKernelContext& ctx) override;

private:
    bool Prepare(CpuKernelContext& ctx);
    bool BalanceSchedule(load_balance::SectionStreamKResult& splitRes);
    bool GenMetadata(load_balance::SectionStreamKResult& splitRes);

    bool CheckNeedInitOutput(uint32_t& uniformQLen) const;
    bool ParamsCheck();
    bool CheckActualQuerySeq();
    bool CheckActualKvSeq();

    bool ParamsInit();
    void InitDeviceInfo();
    void InitBaseInfo();
    void InitLoadBalanceParams();
    void LoadActualQuerySeq();
    void LoadActualKvSeq();
    static int64_t CostFunc(uint32_t basicM, uint32_t basicS2);

    void SetMetadataHead(const load_balance::SectionStreamKResult& splitRes, optiling::detail::FaMetadata& faMetadata);
    void SetMetadataFa(const load_balance::SectionStreamKResult& splitRes, optiling::detail::FaMetadata& faMetadata);
    void SetMetadataFd(const load_balance::SectionStreamKResult& splitRes, optiling::detail::FaMetadata& faMetadata);

private:
    CpuKernelContext* context_ = nullptr;
    // input tensor
    Tensor* cuSeqlensQ_ = nullptr;
    Tensor* cuSeqlensKv_ = nullptr;
    Tensor* sequsedQ_ = nullptr;
    Tensor* sequsedKv_ = nullptr;
    // output tensor
    Tensor* metadata_ = nullptr;

    // input attr
    int32_t batchSize_ = 0;
    int32_t maxSeqlenQ_ = -1;
    int32_t maxSeqlenKv_ = -1;
    int32_t numHeadsQ_ = 0;
    int32_t numHeadsKv_ = 0;
    int32_t headDim_ = 0;
    int32_t headDimV_ = -1; // -1: 未指定, 归一化后等于 headDim_
    int32_t maskMode_ = 1;
    float dropoutP_ = 0.0f;
    int64_t winLeft_ = -1;
    int64_t winRight_ = -1;
    std::string layoutQ_ = "BSND";
    std::string layoutKv_ = "BSND";
    std::string layoutOut_ = "BSND";
    std::string socVersion_ = "";
    int32_t aicCoreNum_ = 36U; // 36: default aic num
    int32_t aivCoreNum_ = 72U; // 72: default aiv num

    // BaseInfo
    bool isActualSeqlenQAccum_ = false;
    bool isActualSeqlenKvAccum_ = false;
    std::vector<int64_t> actualSeqlenQ_{};
    std::vector<int64_t> actualSeqlenKv_{};

    // SplitParams
    uint32_t groupSize_ = 0;
    uint32_t mBaseSize_ = 64;      // 64: default value
    uint32_t s2BaseSize_ = 128;    // 128: default value
    uint32_t maxUsedAicCores_ = 0; // used AIC num from load balance params
    load_balance::DeviceInfo deviceInfo;
    load_balance::BaseInfo baseInfo;
    load_balance::SectionStreamKParam param;

    // FAG split params (Flash Attn Grad metadata)
    int64_t fagS1Inner_ = 64;
    int64_t fagS2Inner_ = 128;
    int64_t fagS1CvInner_ = 128;
    int64_t fagCvS2Inner_ = 128;
    int64_t fagS1Outer_ = 0;
    int64_t fagS2Outer_ = 0;
    int64_t fagG_ = 0;
    int64_t fagBlockStarts_[FAG_CORE_LIST_NUM] = {0};
    int64_t fagBlockEnds_[FAG_CORE_LIST_NUM] = {0};
    int64_t fagTndStartBIdx_[FAG_CORE_LIST_NUM] = {0};
    uint32_t fagBlockOuter_ = 0;
    int64_t fagBlockFactor_ = 0;
    uint32_t fagSplitAxis_ = FAG_SPLIT_AXIS_BN2GS1S2;
    bool fagIsSparse_ = false;
    uint32_t fagSparseMode_ = FAG_SPARSE_NO_MASK;
    uint32_t fagLayoutType_ = FAG_INPUT_FORMAT_BS2N2GD;
    int64_t fagS1Token_ = -1;
    int64_t fagS2Token_ = -1;
    bool fagIsBn2_ = false;
    bool fagIsBn2MultiBlk_ = false;
    bool fagIsAllSame_ = false;
    bool fagIsInvalidCol_ = false;
    bool fagIsInvalidRow_ = false;
    bool fagIsGradEnabled_ = false;
    int64_t fagDeterMaxQ_ = -1;
    int64_t fagDeterMaxKv_ = -1;
    bool fagHasSeqused_ = false;
    int64_t fagPhysicalTokensQ_ = 0;
    int64_t fagPhysicalTokensKv_ = 0;
    bool fagUseTndSplit_ = false;
    bool fagIsSeqExistZero_ = false;
    bool fagFallbackToGs_ = false; // Ordinary TND BN2S2 runtime fallback; deterministic extension is independent.

    // FAG method declarations
    void InitFagParams();
    void InitFagBaseDims();
    void InitFagLayoutAndSparse();
    void InitFagActualSeqlen();
    void DoFagSparse();
    void GenFagMetadata(uint32_t sectionNum);
    bool GenFagDeterMetadata();
    bool GenFagTndMetadata();

    // FAG(BN2GS1S2) split implementations (defined in flash_attn_grad_metadata_split.h)
    void SetFagSplitAxis();
    void SupportTransBSND();
    void DoFagDenseSplit();
    void DoFagTndSplit();
    void DoFagSparseBlockInfo();
    void GetFagParseS1S2OuterInfo(std::vector<std::vector<int64_t>>& parseInfo);
    bool FagCheckSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx);
    bool FagIsValid(int64_t blockIdx);
    bool FagIsValidUnpad(int64_t blockIdx);
    bool FagCheckUnpadSparseLeftAndRight(int64_t s1oDimIdx, int64_t s2IdxLeft, int64_t s2IdxRight, int64_t bIdx);
    void FagFillBlockInfoLoadBalance(std::vector<std::vector<int64_t>>& totalBlockInfo,
                                     std::vector<std::vector<float>>& acturalBlockInfo);

    // BN2S2 split implementations
    void TryBn2s2Sparse();
    void DoFagBn2s2DenseSplit();
    void DoFagBn2s2SparseTndSplit();
    float Bn2s2BinarySearchMaxBlockNumPerCore(int64_t b, int64_t n2, int64_t g, int64_t aicNum,
                                              std::vector<std::vector<int64_t>>& totalBlockInfo,
                                              std::vector<std::vector<float>>& acturalBlockInfo);
    bool Bn2s2IsPossible(int64_t b, int64_t n2, int64_t g, int64_t aicNum, float possibleMax,
                         std::vector<std::vector<float>>& acturalBlockInfo);
    bool Bn2s2CaclePerCoreBlockInfo(int64_t b, int64_t n2, int64_t g, int64_t aicNum,
                                    std::vector<std::vector<int64_t>>& totalBlockInfo,
                                    std::vector<std::vector<float>>& acturalBlockInfo, float maxBlockNumPerCore);

    // BN2 split implementations
    bool TryBn2MultiBlkSparse();
    void DoFagBn2DenseSplit();
    bool DoFagBn2SparseBlockInfo();

private:
    enum ParamId {
        // input
        cuSeqlensQ = 0,
        cuSeqlensKv = 1,
        sequsedQ = 2,
        sequsedKv = 3,
        // output
        metaData = 0,
    };
};
} // namespace aicpu

#endif
