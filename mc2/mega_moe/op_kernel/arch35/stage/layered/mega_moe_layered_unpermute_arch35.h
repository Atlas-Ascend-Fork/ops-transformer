/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_UNPERMUTE_ARCH35_H
#define MEGA_MOE_LAYERED_UNPERMUTE_ARCH35_H

namespace MegaMoeImpl {

/* 单个权重 chunk：搬入并转换权重、逐 token 累加、转换输出并搬出。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteTokenChunk(
    const WorkRange& tokenRange, const MegaMoeUnpermuteBufferConfig& bufferConfig, uint32_t& inputBufferIdx,
    uint32_t& readySharedGroup)
{
    const bool TopkWeightsPrefetch = params_.tilingData->isTopkWeightsPrefetch;
    constexpr uint8_t UNPERMUTE_WEIGHT_MUTEX_ID = 0U;
    constexpr uint8_t UNPERMUTE_OUTPUT_MUTEX_ID = 1U;
    const int32_t chunkStart = static_cast<int32_t>(tokenRange.start);
    const int32_t chunkTokenCnt = static_cast<int32_t>(tokenRange.count);
    const int32_t chunkEnd = chunkStart + chunkTokenCnt;
    __gm__ bfloat16_t* expandedX =
        params_.communication.layout.combineSend.template get_ptr<bfloat16_t>(params_.communication.base);
    __gm__ bfloat16_t* output = reinterpret_cast<__gm__ bfloat16_t*>(params_.y2GmAddr);
    if (!TopkWeightsPrefetch) {
        constexpr bool isBf16Weights = Std::IsSame<TopkWeightsType, bfloat16_t>::value;
        const uint32_t weightCount = chunkTokenCnt * topK_;
        __gm__ TopkWeightsType* weightsGm = reinterpret_cast<__gm__ TopkWeightsType*>(params_.probsGmAddr);
        __ubuf__ TopkWeightsType* weightsUb = isBf16Weights ?
                                                  reinterpret_cast<__ubuf__ TopkWeightsType*>(topKWeightsTempAddr_) :
                                                  reinterpret_cast<__ubuf__ TopkWeightsType*>(topKWeightsTensor_);
        // 等上一批最后一次权重读取完成，再覆盖权重区及 BF16 转换暂存区。
        asc_lock(PIPE_MTE2, UNPERMUTE_WEIGHT_MUTEX_ID);
        CopyGmToUb(weightsUb, weightsGm + chunkStart * topK_,
                   weightCount * static_cast<uint32_t>(sizeof(TopkWeightsType)));
        asc_unlock(PIPE_MTE2, UNPERMUTE_WEIGHT_MUTEX_ID);
        // 等权重搬入，保护转换及本批乘权；最后一个 token 乘权后释放。
        asc_lock(PIPE_V, UNPERMUTE_WEIGHT_MUTEX_ID);
        if constexpr (isBf16Weights) {
            LocalTensor<float> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(topKWeightsTensor_),
                                   chunkTokenCnt * topK_);
            LocalTensor<bfloat16_t> src(TPosition::VECCALC, topKWeightsTempAddr_, chunkTokenCnt * topK_);
            Cast(dst, src, RoundMode::CAST_NONE, chunkTokenCnt * topK_);
        }
    }
    for (int32_t tokenIdx = chunkStart; tokenIdx < chunkEnd; tokenIdx++) {
        int32_t localIdx = tokenIdx - chunkStart;
        AccumulateUnpermuteExperts(expandedX, tokenIdx, localIdx, bufferConfig, inputBufferIdx);
        if (!TopkWeightsPrefetch && tokenIdx + 1 == chunkEnd) {
            // 最后一次乘权后释放，下一批权重搬入可与共享累加和输出 Cast 重叠。
            asc_unlock(PIPE_V, UNPERMUTE_WEIGHT_MUTEX_ID);
        }
        // 共享专家结果累加（直接加，不乘 topk_weight）
        if (sharedExpertNum_ > 0) {
            UnpermuteSharedExpert(tokenIdx, readySharedGroup, bufferConfig, inputBufferIdx);
        }
        // 输出区 [0,k) 在最终 Cast 处复用；复用等待放在 Cast 前，让输入搬入和 FP32 累加
        // 与上一 token 的搬出重叠。
        asc_lock(PIPE_V, UNPERMUTE_OUTPUT_MUTEX_ID);
        // 将 FP32 累加结果转换为 BF16。
        {
            LocalTensor<bfloat16_t> dst(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataResTensor_), k_);
            LocalTensor<float> src(TPosition::VECCALC, reinterpret_cast<uint64_t>(dataResFp32Tensor_), k_);
            Cast(dst, src, RoundMode::CAST_RINT, k_);
        }
        asc_unlock(PIPE_V, UNPERMUTE_OUTPUT_MUTEX_ID);
        asc_lock(PIPE_MTE3, UNPERMUTE_OUTPUT_MUTEX_ID);
        CopyUbToGm((output + (tokenIdx * k_)), dataResTensor_, (k_) * sizeof(bfloat16_t));
        asc_unlock(PIPE_MTE3, UNPERMUTE_OUTPUT_MUTEX_ID);
    }
}

/* 回传完成后，按 Combine 模式输出本 AIV 负责的 token。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::Unpermute()
{
    if (IsCombineServerDedupEnabled()) {
        UnpermuteServerCombine();
        return;
    }
    const auto bufferConfig = UnpermuteBuffInit();
    const WorkRange tokenRange = GetBalancedWorkRange(m_, {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_});
    if (tokenRange.count == 0U) {
        return;
    }
    const uint32_t tokenEnd = tokenRange.start + tokenRange.count;
    const uint32_t batchTokens = bufferConfig.tokensPerBatch;
    // 普通专家、共享专家和后续 token 共用游标，跨权重 chunk 也不重置。
    uint32_t readySharedGroup = UINT32_MAX; // 本次调用内跨权重 chunk 缓存；首 token 必须等待。
    uint32_t inputBufferIdx = 0U;
    // 输入 ring 使用事件 0..5；权重和输出由 mutex 保护。
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<TEventID>(bufferIdx));
    }
    for (uint32_t chunkStart = tokenRange.start; chunkStart < tokenEnd; chunkStart += batchTokens) {
        const uint32_t remainingTokens = tokenEnd - chunkStart;
        const WorkRange chunkRange{chunkStart, batchTokens < remainingTokens ? batchTokens : remainingTokens};
        UnpermuteTokenChunk(chunkRange, bufferConfig, inputBufferIdx, readySharedGroup);
    }
    // 等待最后一次输出搬出，再回收权重、输出和输入 ring 的同步资源。
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID3>();
    for (int32_t bufferIdx = 0; bufferIdx < bufferConfig.inputBufferCount; ++bufferIdx) {
        PipeWait<PIPE_V, PIPE_MTE2>(static_cast<TEventID>(bufferIdx));
    }
}

} // namespace MegaMoeImpl
#endif // MEGA_MOE_LAYERED_UNPERMUTE_ARCH35_H
