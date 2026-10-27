/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_UNPERMUTE_H
#define MEGA_MOE_UNPERMUTE_H

#include "../common/mega_moe_sync.h"
#include "../common/mega_moe_mxfp8_utils.h"
#include "../common/mega_moe_utils.h"
#include "../common/mega_moe_copy.h"

namespace MegaMoeImpl {

using namespace AscendC;

struct TokenUnpermuteConfig {
    AivJobContext job;
    uint32_t quantTokenSizeBytes;
    uint32_t fullTokenChunkJobCount;
    MegaMoeUnpermuteBufferConfig fullTokenChunkConfig;
    MegaMoeUnpermuteBufferConfig tailTokenChunkConfig;
};

struct TokenUnpermuteScratch {
    __ubuf__ bfloat16_t* dataResTensor = nullptr;
    __ubuf__ float* dataResFp32Tensor = nullptr;
    __ubuf__ float* topKWeightsTensor = nullptr;
    __ubuf__ float* fp32ScaleTensor = nullptr;
    __ubuf__ bfloat16_t* bf16ScaleTensor = nullptr;
    __ubuf__ bfloat16_t* topKWeightsBf16Tensor = nullptr;
};

// 加载并转换一批 top-k 权重。
template <typename TopkWeightsType>
__aicore__ inline void LoadTokenUnpermuteWeights(const MoeStageCommonConfig& common, const Params& params,
                                                 TokenUnpermuteScratch& scratch, int32_t jobOffset,
                                                 int32_t batchTokenOffset, int32_t batchTokenCount)
{
    if constexpr (Std::IsSame<TopkWeightsType, float>::value) {
        __gm__ float* topKWeightsGm = reinterpret_cast<__gm__ float*>(params.probsGmAddr);
        CopyGmToUb(scratch.topKWeightsTensor,
                   (topKWeightsGm + (static_cast<uint64_t>(jobOffset + batchTokenOffset) * common.topK)),
                   static_cast<uint32_t>(batchTokenCount * common.topK * sizeof(float)));
        PipeNotify<PIPE_MTE2, PIPE_S>(EVENT_ID0);
        PipeWait<PIPE_MTE2, PIPE_S>(EVENT_ID0);
    }
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        __gm__ bfloat16_t* topKWeightsGm = reinterpret_cast<__gm__ bfloat16_t*>(params.probsGmAddr);
        CopyGmToUb(scratch.topKWeightsBf16Tensor,
                   (topKWeightsGm + (static_cast<uint64_t>(jobOffset + batchTokenOffset) * common.topK)),
                   static_cast<uint32_t>(batchTokenCount * common.topK * sizeof(bfloat16_t)));
        PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();
        {
            LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(scratch.topKWeightsTensor),
                                   batchTokenCount * common.topK);
            LocalTensor<bfloat16_t> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(scratch.topKWeightsBf16Tensor),
                                        batchTokenCount * common.topK);
            Cast(dst, src, RoundMode::CAST_NONE, batchTokenCount * common.topK);
        }
        PipeNotify<PIPE_V, PIPE_S>(EVENT_ID0);
        PipeWait<PIPE_V, PIPE_S>(EVENT_ID0);
    }
}

// 将一个 MoE 专家的输入搬入 UB；Arch35 非量化输入由累加 VF 转换，量化输入先完成 MXFP8 反量化。
template <uint8_t CombineMode>
__aicore__ inline void LoadMoeExpertInput(const TokenUnpermuteConfig& context, const MoeStageCommonConfig& common,
                                          TokenUnpermuteScratch& scratch, __gm__ bfloat16_t* expandedX,
                                          uint64_t expertInputIndex, TEventID event, __ubuf__ bfloat16_t* dataInBf16,
                                          __ubuf__ float* dataInFp32)
{
    PipeWait<PIPE_V, PIPE_MTE2>(event);
    if constexpr (CombineMode == COMBINE_NO_QUANT) {
        CopyGmToUb(dataInBf16, (expandedX + (expertInputIndex * common.tokenHiddenDim)),
                   (common.tokenHiddenDim) * sizeof(bfloat16_t));
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
#if defined(MEGA_MOE_ARCH92_DEVICE)
        LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32), common.tokenHiddenDim);
        LocalTensor<bfloat16_t> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInBf16), common.tokenHiddenDim);
        Cast(dst, src, RoundMode::CAST_NONE, common.tokenHiddenDim);
#endif
    } else {
        uint32_t quantTokenElementCount = context.quantTokenSizeBytes / sizeof(bfloat16_t);
        CopyGmToUb(dataInBf16, (expandedX + (expertInputIndex * quantTokenElementCount)),
                   (quantTokenElementCount) * sizeof(bfloat16_t));
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
        uint32_t nScale = Ops::Base::CeilDiv(common.tokenHiddenDim, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        using Fp8Type = typename std::conditional<CombineMode == MXFP8_E4M3_COMM_QUANT, fp8_e4m3fn_t, fp8_e5m2_t>::type;
        Mxfp8::DeQuantMxFp8<Fp8Type, bfloat16_t>(dataInBf16, dataInFp32, scratch.bf16ScaleTensor,
                                                 scratch.fp32ScaleTensor, nScale, common.tokenHiddenDim);
    }
}

#if !defined(MEGA_MOE_ARCH92_DEVICE)
// 一次处理一个专家槽，转换和乘权结果留在寄存器；首项直接初始化，不读取旧累加值。
template <bool InitAccum, bool ApplyWeight>
__simd_vf__ inline void AccumulateBf16ExpertSlotVF(__ubuf__ float* accumUb, __ubuf__ bfloat16_t* inputUb,
                                                   float expertScale, uint32_t elementCount)
{
    // Host 保证 H 按 64 对齐且不超过 8192，完整向量加载在输入槽内，循环次数最多 128。
    constexpr uint32_t ELEMENTS_PER_VECTOR = static_cast<uint32_t>(asc_get_vf_len()) / sizeof(float);
    const uint16_t repeatCount = static_cast<uint16_t>((elementCount + ELEMENTS_PER_VECTOR - 1U) / ELEMENTS_PER_VECTOR);
    vector_bfloat16_t inputReg;
    vector_float valueReg;
    for (uint16_t i = 0U; i < repeatCount; ++i) {
        const uint32_t offset = static_cast<uint32_t>(i) * ELEMENTS_PER_VECTOR;
        vector_bool active = asc_update_mask_b32(elementCount);
        // 64 个连续 BF16 放在偶数 b16 lane，转换后与 64 个 FP32 累加值逐项对应。
        asc_loadalign_unpack(inputReg, inputUb + offset);
        asc_bfloat162float(valueReg, inputReg, active, ASC_POSITION_EVEN);
        if constexpr (ApplyWeight) {
            // 保持独立 FP32 乘法和加法，不使用 FMA 或乘加后转半精度的接口。
            asc_mul_scalar(valueReg, valueReg, expertScale, active);
        }
        if constexpr (!InitAccum) {
            vector_float accumReg;
            asc_loadalign(accumReg, accumUb + offset);
            asc_add(valueReg, accumReg, valueReg, active);
        }
        asc_storealign(accumUb + offset, valueReg, active);
    }
}

__aicore__ inline void AccumulateBf16ExpertSlotInput(const MoeStageCommonConfig& common, TokenUnpermuteScratch& scratch,
                                                     __ubuf__ bfloat16_t* dataInBf16, int32_t localIdx,
                                                     int32_t expertIdx, bool applyWeight, bool accumStarted)
{
    auto* accumUb = scratch.dataResFp32Tensor;
    auto* inputUb = dataInBf16;
    if (applyWeight) {
        float expertScale = scratch.topKWeightsTensor[localIdx * common.topK + expertIdx];
        if (accumStarted) {
            AccumulateBf16ExpertSlotVF<false, true>(accumUb, inputUb, expertScale, common.tokenHiddenDim);
        } else {
            AccumulateBf16ExpertSlotVF<true, true>(accumUb, inputUb, expertScale, common.tokenHiddenDim);
        }
    } else {
        if (accumStarted) {
            AccumulateBf16ExpertSlotVF<false, false>(accumUb, inputUb, 0.0f, common.tokenHiddenDim);
        } else {
            AccumulateBf16ExpertSlotVF<true, false>(accumUb, inputUb, 0.0f, common.tokenHiddenDim);
        }
    }
}
#endif

// 读回本 token 的 topK 目标卡号（专家→卡 = expertId / moeExpertPerRank）。
// 结果放函数栈小数组（topK≤32 有 host 校验；unpermute UB 已满载，UB 化需动整段布局，暂保留栈存储）。
__aicore__ inline void LoadTokenExpertRanks(const MoeStageCommonConfig& common, const Params& params, int32_t tokenIdx,
                                            int32_t (&expertRanks)[32])
{
    __gm__ int32_t* topkIdsBase =
        reinterpret_cast<__gm__ int32_t*>(params.expertIdxGmAddr) + static_cast<uint64_t>(tokenIdx) * common.topK;
    for (int32_t k = 0; k < static_cast<int32_t>(common.topK); ++k) {
        expertRanks[k] = GmRead(topkIdsBase + k) / static_cast<int32_t>(common.moeExpertPerRank);
    }
}

// canonical 槽判定：该槽是其目标卡在本 token topK 里的首个 k 才会到达（与生产卡 minTopkIdx 约定
// 一致）。返回 false = 该槽已并入同卡首槽、不会到达；sameRankCount 顺手数同卡命中数，tkw=0 下
// 决定该槽是否还需本端乘权（==1 是 PLAIN 行、发送端未乘；>=2 是合并行、发送端已按成员权重乘加）。
__aicore__ inline bool JudgeDedupCanonicalSlot(const int32_t (&expertRanks)[32], int32_t topK, int32_t expertIdx,
                                               int32_t& sameRankCount)
{
    sameRankCount = 1;
    for (int32_t otherIdx = 0; otherIdx < topK; ++otherIdx) {
        if (otherIdx != expertIdx && expertRanks[otherIdx] == expertRanks[expertIdx]) {
            if (otherIdx < expertIdx) {
                return false;
            }
            ++sameRankCount;
        }
    }
    return true;
}

// 把一个专家槽的输入累加进 dataResFp32Tensor（首槽直拷/乘权，后续槽乘权后加）。
__aicore__ inline void AccumulateExpertSlotInput(const MoeStageCommonConfig& common, TokenUnpermuteScratch& scratch,
                                                 __ubuf__ float* dataInFp32, int32_t localIdx, int32_t expertIdx,
                                                 bool applyWeight, bool accumStarted)
{
    auto* accum = scratch.dataResFp32Tensor;
    if (applyWeight) {
        float expertScale = scratch.topKWeightsTensor[localIdx * common.topK + expertIdx];
        auto* dst = accumStarted ? dataInFp32 : accum;
        LocalTensor<float> dstTensor(TPosition::VECCALC, reinterpret_cast<uint64_t>(dst), common.tokenHiddenDim);
        LocalTensor<float> srcTensor(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32), common.tokenHiddenDim);
        Muls(dstTensor, srcTensor, expertScale, common.tokenHiddenDim);
    } else {
        if (!accumStarted) {
            CopyUbToUb(accum, dataInFp32, common.tokenHiddenDim * sizeof(float));
        }
    }
    if (accumStarted) {
        LocalTensor<float> accumTensor(TPosition::VECCALC, reinterpret_cast<uint64_t>(accum), common.tokenHiddenDim);
        LocalTensor<float> inputTensor(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32),
                                       common.tokenHiddenDim);
        Add(accumTensor, accumTensor, inputTensor, common.tokenHiddenDim);
    }
}

// 累加一个 token 对应的全部 MoE 专家输入。
// combine 去重模式：生产卡已按卡合并（同卡多专家的贡献落在该卡首个 topk 槽、其余槽不再到达），
// 归属卡只累加 canonical 槽。
template <uint8_t CombineMode, bool TopkWeightsPrefetch>
__aicore__ inline void AccumulateMoeExpertsForToken(const TokenUnpermuteConfig& context,
                                                    const MoeStageCommonConfig& common, const Params& params,
                                                    TokenUnpermuteScratch& scratch, int32_t tokenIdx, int32_t localIdx,
                                                    __gm__ bfloat16_t* expandedX,
                                                    const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    const bool dedupCombineActive = IsCombineDedupOn(params.tilingData->compute.dedupMode);
    int32_t expertRanks[32];
    if (dedupCombineActive) {
        LoadTokenExpertRanks(common, params, tokenIdx, expertRanks);
    }
    bool accumStarted = false;
    for (int32_t expertIdx = 0; expertIdx < static_cast<int32_t>(common.topK); ++expertIdx) {
        int32_t sameRankCount = 1;
        if (dedupCombineActive &&
            !JudgeDedupCanonicalSlot(expertRanks, static_cast<int32_t>(common.topK), expertIdx, sameRankCount)) {
            continue; // 该槽不会到达（已并入同卡首槽）
        }
        int32_t accumulationItemIdx = localIdx * static_cast<int32_t>(common.topK + common.sharedExpertNum) + expertIdx;
        int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
        TEventID event = static_cast<TEventID>(inputBufferIdx);
        __ubuf__ bfloat16_t* dataInBf16 =
            (scratch.dataResTensor + ((inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount));
        __ubuf__ float* dataInFp32 = nullptr;
#if !defined(MEGA_MOE_ARCH92_DEVICE)
        if constexpr (CombineMode != COMBINE_NO_QUANT)
#endif
        {
            dataInFp32 = scratch.dataResFp32Tensor + (inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount;
        }
        uint64_t expertInputIndex = static_cast<uint64_t>(tokenIdx) * common.topK + expertIdx;
        LoadMoeExpertInput<CombineMode>(context, common, scratch, expandedX, expertInputIndex, event, dataInBf16,
                                        dataInFp32);
        const bool applyWeight = !TopkWeightsPrefetch && (!dedupCombineActive || sameRankCount == 1);
#if !defined(MEGA_MOE_ARCH92_DEVICE)
        if constexpr (CombineMode == COMBINE_NO_QUANT) {
            AccumulateBf16ExpertSlotInput(common, scratch, dataInBf16, localIdx, expertIdx, applyWeight, accumStarted);
        } else
#endif
        {
            PipeNotify<PIPE_S, PIPE_V>(event);
            PipeWait<PIPE_S, PIPE_V>(event);
            AccumulateExpertSlotInput(common, scratch, dataInFp32, localIdx, expertIdx, applyWeight, accumStarted);
        }
        accumStarted = true;
        PipeNotify<PIPE_V, PIPE_MTE2>(event);
    }
}

// 等待该共享专家的 GMM2 结果就绪后累加进当前 token（等待与累加是一体动作，调用方无需分两步）。
template <uint32_t Gmm1TileM>
__aicore__ inline void AccumulateSharedExpertForToken(const MoeStageCommonConfig& common, const Params& params,
                                                      TokenUnpermuteScratch& scratch, int32_t tokenIdx,
                                                      int32_t localIdx, uint32_t sharedExpertIdx,
                                                      const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    const auto& workspace = params.workspace.layout;
    const auto& buffer = params.workspace.base;
    uint32_t tokenGroupIndex = static_cast<uint32_t>(tokenIdx) / Gmm1TileM;
    uint32_t tokenGroupCount = Ops::Base::CeilDiv(common.tokenNum, Gmm1TileM);
    uint64_t sharedExpertStride = static_cast<uint64_t>(tokenGroupCount) * INT_CACHELINE;
    __gm__ int32_t* counterAddr =
        GetCombineSyncCounterAddress(workspace.mte.sync.sharedGmm2Tiles.get_ptr<int32_t>(buffer) +
                                         static_cast<uint64_t>(sharedExpertIdx) * sharedExpertStride,
                                     tokenGroupIndex);
    uint32_t gmm2NTilesPerGroup = Ops::Base::CeilDiv(common.tokenHiddenDim, L1_TILE_N);
    WaitUntilGmFlagEquals(counterAddr, static_cast<int32_t>(gmm2NTilesPerGroup));

    __gm__ bfloat16_t* sharedResult = workspace.shared.gmm2Output.get_ptr<bfloat16_t>(buffer);
    int32_t accumulationItemIdx = localIdx * static_cast<int32_t>(common.topK + common.sharedExpertNum) +
                                  static_cast<int32_t>(common.topK + sharedExpertIdx);
    int32_t inputBufferIdx = accumulationItemIdx % bufferConfig.inputBufferCount;
    TEventID event = static_cast<TEventID>(inputBufferIdx);
    __ubuf__ bfloat16_t* dataInBf16 =
        (scratch.dataResTensor + ((inputBufferIdx + 1) * bufferConfig.bf16SlotElementCount));
    PipeWait<PIPE_V, PIPE_MTE2>(event);
    CopyGmToUb(dataInBf16,
               (sharedResult +
                ((static_cast<uint64_t>(sharedExpertIdx) * common.tokenNum + tokenIdx) * common.tokenHiddenDim)),
               (common.tokenHiddenDim) * sizeof(bfloat16_t));
    PipeNotify<PIPE_MTE2, PIPE_V>(event);
    PipeWait<PIPE_MTE2, PIPE_V>(event);
    PipeNotify<PIPE_S, PIPE_V>(event);
    PipeWait<PIPE_S, PIPE_V>(event);
#if defined(MEGA_MOE_ARCH92_DEVICE)
    __ubuf__ float* dataInFp32 = scratch.dataResFp32Tensor + (inputBufferIdx + 1) * bufferConfig.fp32SlotElementCount;
    {
        LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32), common.tokenHiddenDim);
        LocalTensor<bfloat16_t> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInBf16), common.tokenHiddenDim);
        Cast(dst, src, RoundMode::CAST_NONE, common.tokenHiddenDim);
    }

    {
        LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(scratch.dataResFp32Tensor),
                               common.tokenHiddenDim);
        LocalTensor<float> rhs(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataInFp32), common.tokenHiddenDim);
        Add(dst, dst, rhs, common.tokenHiddenDim);
    }

#else
    AccumulateBf16ExpertSlotVF<false, false>(scratch.dataResFp32Tensor, dataInBf16, 0.0f, common.tokenHiddenDim);
#endif
    PipeNotify<PIPE_V, PIPE_MTE2>(event);
}

template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch, uint32_t Gmm1TileM>
__aicore__ inline void ProcessTokenUnpermuteBatch(const TokenUnpermuteConfig& context,
                                                  const MoeStageCommonConfig& common, const Params& params,
                                                  TokenUnpermuteScratch& scratch,
                                                  const MegaMoeUnpermuteBufferConfig& bufferConfig,
                                                  __gm__ bfloat16_t* expandedX, __gm__ bfloat16_t* output,
                                                  int32_t jobOffset, int32_t batchTokenOffset, int32_t batchTokenCount,
                                                  TEventID outputBufferEvent)
{
    if constexpr (!TopkWeightsPrefetch) {
        LoadTokenUnpermuteWeights<TopkWeightsType>(common, params, scratch, jobOffset, batchTokenOffset,
                                                   batchTokenCount);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<TEventID>(bufferIdx));
    }
    for (int32_t localIdx = 0; localIdx < batchTokenCount; ++localIdx) {
        int32_t tokenIdx = jobOffset + batchTokenOffset + localIdx;
        AccumulateMoeExpertsForToken<CombineMode, TopkWeightsPrefetch>(context, common, params, scratch, tokenIdx,
                                                                       localIdx, expandedX, bufferConfig);
        for (uint32_t sharedExpertIdx = 0; sharedExpertIdx < common.sharedExpertNum; ++sharedExpertIdx) {
            AccumulateSharedExpertForToken<Gmm1TileM>(common, params, scratch, tokenIdx, localIdx, sharedExpertIdx,
                                                      bufferConfig);
        }
        PipeWait<PIPE_MTE3, PIPE_V>(outputBufferEvent);
        {
            LocalTensor<bfloat16_t> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(scratch.dataResTensor),
                                        common.tokenHiddenDim);
            LocalTensor<float> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(scratch.dataResFp32Tensor),
                                   common.tokenHiddenDim);
            Cast(dst, src, RoundMode::CAST_RINT, common.tokenHiddenDim);
        }
        PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID3>();
        CopyUbToGm((output + (static_cast<uint64_t>(tokenIdx) * common.tokenHiddenDim)), scratch.dataResTensor,
                   (common.tokenHiddenDim) * sizeof(bfloat16_t));
        PipeNotify<PIPE_MTE3, PIPE_V>(outputBufferEvent);
    }
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        PipeWait<PIPE_V, PIPE_MTE2>(static_cast<TEventID>(bufferIdx));
    }
}

/*
 * 构造 Unpermute 使用的 UB 视图，并返回当前 AIV 对应的 buffer 配置（普通/wave/a8w4 编排共用）。
 * UB 从地址 0 起按下列顺序连续排布（N = inputBufferCount，每段起点=上一段末尾）：
 *   [dataRes    ] (N+1) 个 bf16 槽 × bf16SlotElementCount —— 累加结果 bf16 输出
 *   [dataResFp32] 1 个 fp32 累加槽；Arch92/量化 combine 另有 N 个 fp32 输入临时槽
 *   [topKWeights] topKWeightsBufferBytes —— 本批 token 的 topk 权重（fp32）
 *   [权重转换区 ] 仅 TopkWeightsType==bf16 时存在（topKWeightsConversionBufferBytes）
 *   [bf16Scale  ] 仅量化 combine 时存在，Align32(ceil(H/32)·2B·扩展系数)
 *   [fp32Scale  ] 仅量化 combine 时存在，Align32(ceil(H/32)·4B·扩展系数)
 */
template <typename TopkWeightsType, int32_t CombineQuantMode>
__aicore__ inline MegaMoeUnpermuteBufferConfig CreateTokenUnpermuteBuffers(const TokenUnpermuteConfig& context,
                                                                           uint32_t tokenHiddenDim,
                                                                           TokenUnpermuteScratch& scratch)
{
    MegaMoeUnpermuteBufferConfig bufferConfig = context.job.jobIndex < context.fullTokenChunkJobCount ?
                                                    context.fullTokenChunkConfig :
                                                    context.tailTokenChunkConfig;

    uint32_t bf16ScaleBufAlign = 0U;
    uint32_t fp32ScaleBufAlign = 0U;
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        uint32_t scaleNum = Ops::Base::CeilDiv(tokenHiddenDim, static_cast<uint32_t>(ALIGN_32));
        bf16ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(bfloat16_t)) * static_cast<uint32_t>(DEQUANT_BF16_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
        fp32ScaleBufAlign = Ops::Base::CeilAlign(
            scaleNum * static_cast<uint32_t>(sizeof(float)) * static_cast<uint32_t>(DEQUANT_FP32_SCALE_EXPANSION),
            static_cast<uint32_t>(ALIGN_32));
    }

    uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    uint32_t dataResBufBytes = (bufferConfig.inputBufferCount + 1) * bf16SlotBytes;
    uint32_t dataResFp32BufBytes = (bufferConfig.inputBufferCount + 1) * fp32SlotBytes;
#if !defined(MEGA_MOE_ARCH92_DEVICE)
    if constexpr (CombineQuantMode == COMBINE_NO_QUANT) {
        dataResFp32BufBytes = fp32SlotBytes;
    }
#endif
    uint32_t dataResAddr = 0U;
    scratch.dataResTensor = reinterpret_cast<__ubuf__ bfloat16_t*>(dataResAddr);
    uint32_t dataResFp32Addr = dataResAddr + dataResBufBytes;
    scratch.dataResFp32Tensor = reinterpret_cast<__ubuf__ float*>(dataResFp32Addr);
    uint32_t tempAddr = dataResFp32Addr + dataResFp32BufBytes;

    scratch.topKWeightsTensor = reinterpret_cast<__ubuf__ float*>(tempAddr);
    tempAddr += bufferConfig.topKWeightsBufferBytes;
    if constexpr (Std::IsSame<TopkWeightsType, bfloat16_t>::value) {
        scratch.topKWeightsBf16Tensor = reinterpret_cast<__ubuf__ bfloat16_t*>(tempAddr);
        tempAddr += bufferConfig.topKWeightsConversionBufferBytes;
    }
    if constexpr (CombineQuantMode != COMBINE_NO_QUANT) {
        scratch.bf16ScaleTensor = reinterpret_cast<__ubuf__ bfloat16_t*>(tempAddr);
        tempAddr += bf16ScaleBufAlign;
        scratch.fp32ScaleTensor = reinterpret_cast<__ubuf__ float*>(tempAddr);
    }
    return bufferConfig;
}

// 执行当前 token 分区的完整 Unpermute 流水。
template <uint8_t CombineMode, typename TopkWeightsType, bool TopkWeightsPrefetch, uint32_t Gmm1TileM>
__aicore__ inline void UnpermuteTokens(const TokenUnpermuteConfig& context, const MoeStageCommonConfig& common,
                                       const Params& params, TokenUnpermuteScratch& scratch,
                                       const MegaMoeUnpermuteBufferConfig& bufferConfig)
{
    const auto& peermem = params.communication.layout;
    const auto& commBuffer = params.communication.base;
    if constexpr (g_coreType == AIC) {
        return;
    }
    WorkRange tokenRange = TilingByJobContext(common.tokenNum, context.job.jobIndex, context.job.totalJobs, 1U);
    if (tokenRange.count == 0U) {
        return;
    }
    int32_t jobLen = static_cast<int32_t>(tokenRange.count);
    int32_t jobOffset = static_cast<int32_t>(tokenRange.start);
    __gm__ bfloat16_t* expandedX = peermem.combineSend.template get_ptr<bfloat16_t>(commBuffer);
    __gm__ bfloat16_t* output = reinterpret_cast<__gm__ bfloat16_t*>(params.y2GmAddr);

    constexpr TEventID outputBufferEvent = EVENT_ID0;
    PipeNotify<PIPE_MTE3, PIPE_V>(outputBufferEvent);
    for (int32_t batchTokenOffset = 0; batchTokenOffset < jobLen; batchTokenOffset += bufferConfig.tokensPerBatch) {
        int32_t batchTokenCount = batchTokenOffset + bufferConfig.tokensPerBatch > jobLen ? jobLen - batchTokenOffset :
                                                                                            bufferConfig.tokensPerBatch;
        ProcessTokenUnpermuteBatch<CombineMode, TopkWeightsType, TopkWeightsPrefetch, Gmm1TileM>(
            context, common, params, scratch, bufferConfig, expandedX, output, jobOffset, batchTokenOffset,
            batchTokenCount, outputBufferEvent);
    }
    PipeWait<PIPE_MTE3, PIPE_V>(outputBufferEvent);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_UNPERMUTE_H
