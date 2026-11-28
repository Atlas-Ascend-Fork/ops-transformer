/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_LAYERED_UNPERMUTE_H
#define MEGA_MOE_LAYERED_UNPERMUTE_H

namespace MegaMoeImpl {

struct LayeredDequantScaleBufferBytes {
    uint32_t bf16;
    uint32_t fp32;
};

/*
 * 按公共 DeQuantMxFp8 的实际存储轮次计算两块 scale scratch 的字节数，包含尾轮无效元素。
 * 设 VL 为向量寄存器字节数：DIST_INTLV_B16/B32 忽略 mask，每轮交织写出 2 * VL 字节。
 * BF16 阶段每轮转换 VL / sizeof(bfloat16_t) 个原始 scale，并交织复制为两份。
 * FP32 阶段的输入已有 scaleNum * 2 个 BF16 元素，每轮转换 VL / sizeof(float) 个，再复制为两份。
 * 因此先将每阶段转换后的单份数据字节数向上对齐到 VL（计算完整轮数），再乘 2（双寄存器存储）。
 * 不能先乘 2 再仅按 VL 对齐：例如 VL=256B、scaleNum=34，BF16 阶段一轮实际写 512B，
 * Align256(34 * 2B) * 2 = 512B，而 Align256(34 * 2B * 2) 只有 256B。
 */
__aicore__ inline LayeredDequantScaleBufferBytes CalcLayeredDequantScaleBufferBytes(uint32_t scaleNum)
{
    constexpr uint32_t vectorBytes = AscendC::VECTOR_REG_WIDTH;
    uint32_t bf16Bytes = Ops::Base::CeilAlign(scaleNum * static_cast<uint32_t>(sizeof(bfloat16_t)), vectorBytes) * 2U;
    uint32_t fp32Bytes = Ops::Base::CeilAlign(scaleNum * 2U * static_cast<uint32_t>(sizeof(float)), vectorBytes) * 2U;
    return {bf16Bytes, fp32Bytes};
}

/* 按 Host 配置绑定普通 Unpermute 的输入 ring、累加区和权重区。 */
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline MegaMoeUnpermuteBufferConfig MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteBuffInit()
{
    const uint8_t CombineQuantMode = params_.tilingData->combineQuantMode;
    const auto* tilingData = params_.tilingData;
    const auto bufferConfig = aivCoreIdx_ < tilingData->unpermuteFullTokenChunkCoreCount ?
                                  tilingData->unpermuteConfigForFullTokenChunk :
                                  tilingData->unpermuteConfigForTailTokenChunk;
    if (bufferConfig.tokensPerBatch == 0) {
        return bufferConfig;
    }
    const uint32_t bf16SlotBytes = bufferConfig.bf16SlotElementCount * sizeof(bfloat16_t);
    const uint32_t fp32SlotBytes = bufferConfig.fp32SlotElementCount * sizeof(float);
    const uint32_t dataResBytes = (bufferConfig.inputBufferCount + 1U) * bf16SlotBytes;
    uint32_t dataResFp32Bytes = fp32SlotBytes;
    if (CombineQuantMode != COMBINE_NO_QUANT) {
        dataResFp32Bytes += bufferConfig.inputBufferCount * fp32SlotBytes;
    }
    // BF16 槽 0 用于输出，其余槽用于输入；FP32 槽 0 累加，通信量化另有输入槽。
    dataResTensor_ = reinterpret_cast<__ubuf__ bfloat16_t*>(0U);
    dataResFp32Tensor_ = reinterpret_cast<__ubuf__ float*>(dataResBytes);
    uint32_t tempAddr = dataResBytes + dataResFp32Bytes;
    topKWeightsTensor_ = reinterpret_cast<__ubuf__ float*>(tempAddr);
    tempAddr += bufferConfig.topKWeightsBufferBytes;
    topKWeightsTempAddr_ = tempAddr;
    tempAddr += bufferConfig.topKWeightsConversionBufferBytes;
    if (CombineQuantMode != COMBINE_NO_QUANT) {
        const uint32_t scaleNum = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
        const auto scaleBytes = CalcLayeredDequantScaleBufferBytes(scaleNum);
        bf16ScaleTensor_ = reinterpret_cast<__ubuf__ bfloat16_t*>(tempAddr);
        tempAddr += scaleBytes.bf16;
        fp32ScaleTensor_ = reinterpret_cast<__ubuf__ float*>(tempAddr);
    }
    return bufferConfig;
}

// ===============================================================
// UnpermuteSharedExpert：共享专家结果累加到当前 token 的 fp32 累加器
// ===============================================================
template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::WaitForSharedExpertResult(
    uint32_t sharedExpertIdx, uint32_t tokenIdx)
{
    const uint32_t tokenGroupCount = Ops::Base::CeilDiv(m_, COMBINE_TOKEN_GROUP_SIZE);
    auto* expertCounters =
        params_.workspace.layout.urma.sync.sharedGmm2Tiles.template get_ptr<int32_t>(params_.workspace.base) +
        static_cast<uint64_t>(sharedExpertIdx) * tokenGroupCount * INT_CACHELINE;
    auto* counter = GetCombineSyncCounterAddress(expertCounters, tokenIdx / COMBINE_TOKEN_GROUP_SIZE);
    const uint32_t outputTileCount = Ops::Base::CeilDiv(k_, static_cast<uint32_t>(L1_TILE_N));
    // FIX 写完该组的全部输出列块后，才允许 Unpermute 读取共享结果。
    WaitUntilGmFlagEquals(counter, static_cast<int32_t>(outputTileCount));
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteSharedExpert(
    int32_t tokenIdx, uint32_t& readySharedGroup, const MegaMoeUnpermuteBufferConfig& bufferConfig,
    uint32_t& inputBufferIdx)
{
    const uint32_t tokenGroup = static_cast<uint32_t>(tokenIdx) / COMBINE_TOKEN_GROUP_SIZE;
    const bool waitForGroup = tokenGroup != readySharedGroup;
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    __gm__ bfloat16_t* sharedResult = workspace.shared.gmm2Output.template get_ptr<bfloat16_t>(buffer);
    for (uint32_t sharedIdx = 0; sharedIdx < sharedExpertNum_; sharedIdx++) {
        if (waitForGroup) {
            WaitForSharedExpertResult(sharedIdx, static_cast<uint32_t>(tokenIdx));
        }
        auto event = static_cast<TEventID>(inputBufferIdx);
        auto dataInBf16 = dataResTensor_ + (inputBufferIdx + 1U) * bufferConfig.bf16SlotElementCount;
        PipeWait<PIPE_V, PIPE_MTE2>(event);
        CopyGmToUb(dataInBf16, (sharedResult + ((sharedIdx * m_ + tokenIdx) * k_)), (k_) * sizeof(bfloat16_t));
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
        AccumulateUnpermuteExpertVF<bfloat16_t, false, false>(dataResFp32Tensor_, dataInBf16,
                                                              static_cast<__ubuf__ float*>(nullptr), k_);
        PipeNotify<PIPE_V, PIPE_MTE2>(event);
        inputBufferIdx = (inputBufferIdx + 1U) % static_cast<uint32_t>(bufferConfig.inputBufferCount);
    }
    readySharedGroup = tokenGroup; // 全部共享专家已就绪，后续 token 复用本组结果。
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::LoadUnpermuteExpertInput(
    __gm__ bfloat16_t* expandedX, int32_t tokenIdx, int32_t expId, TEventID event, __ubuf__ bfloat16_t* dataInBf16,
    __ubuf__ float* dataInFp32)
{
    const uint8_t CombineQuantMode = params_.tilingData->combineQuantMode;
    if (CombineQuantMode == COMBINE_NO_QUANT) {
        PipeWait<PIPE_V, PIPE_MTE2>(event);
        CopyGmToUb(dataInBf16, (expandedX + ((tokenIdx * topK_ + expId) * k_)), (k_) * sizeof(bfloat16_t));
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
    } else {
        uint32_t nScale = Ops::Base::CeilDiv(k_, uint32_t(MXFP_SCALE_GROUP_NUM));
        // 与发送端及 Peermem 容量使用相同的 padded record stride，包含真正的 scale 区。
        uint32_t quantTokenSize = static_cast<uint32_t>(CalcCombineTokenBytes(k_, sizeof(bfloat16_t), true));
        uint32_t quantEleNum = quantTokenSize / sizeof(bfloat16_t);
        PipeWait<PIPE_V, PIPE_MTE2>(event);
        CopyGmToUb(dataInBf16, (expandedX + ((tokenIdx * topK_ + expId) * quantEleNum)),
                   (quantEleNum) * sizeof(bfloat16_t));
        PipeNotify<PIPE_MTE2, PIPE_V>(event);
        PipeWait<PIPE_MTE2, PIPE_V>(event);
        PipeNotify<PIPE_S, PIPE_V>(event);
        PipeWait<PIPE_S, PIPE_V>(event);
        if (CombineQuantMode == MXFP8_E4M3_COMM_QUANT) {
            Mxfp8::DeQuantMxFp8<fp8_e4m3fn_t, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_, fp32ScaleTensor_,
                                                          nScale, k_);
        } else {
            Mxfp8::DeQuantMxFp8<fp8_e5m2_t, bfloat16_t>(dataInBf16, dataInFp32, bf16ScaleTensor_, fp32ScaleTensor_,
                                                        nScale, k_);
        }
    }
}

// 仅为 VF 输入类型及原有标量开关保留独立特化。
template <typename InputType>
__aicore__ inline void AccumulateLayeredUnpermuteInput(__ubuf__ float* accum, __ubuf__ InputType* input,
                                                       __ubuf__ float* weight, uint32_t elementCount, bool applyWeight,
                                                       bool initAccum)
{
    SelectRuntimeBool(
        applyWeight, [&](auto weightTag) __attribute__((cce_aicore)) {
            if (initAccum) {
                AccumulateUnpermuteExpertVF<InputType, decltype(weightTag)::value, true>(accum, input, weight,
                                                                                         elementCount);
            } else {
                AccumulateUnpermuteExpertVF<InputType, decltype(weightTag)::value, false>(accum, input, weight,
                                                                                          elementCount);
            }
        });
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::AccumulateUnpermuteExperts(
    __gm__ bfloat16_t* expandedX, int32_t tokenIdx, int32_t localIdx, const MegaMoeUnpermuteBufferConfig& bufferConfig,
    uint32_t& inputBufferIdx)
{
    const bool TopkWeightsPrefetch = params_.tilingData->isTopkWeightsPrefetch;
    const uint8_t CombineQuantMode = params_.tilingData->combineQuantMode;
    for (int32_t expId = 0; expId < topK_; ++expId) {
        auto event = static_cast<TEventID>(inputBufferIdx);
        auto dataInBf16 = dataResTensor_ + (inputBufferIdx + 1U) * bufferConfig.bf16SlotElementCount;
        __ubuf__ float* dataInFp32 = nullptr;
        if (CombineQuantMode != COMBINE_NO_QUANT) {
            dataInFp32 = dataResFp32Tensor_ + (inputBufferIdx + 1U) * bufferConfig.fp32SlotElementCount;
        }
        LoadUnpermuteExpertInput(expandedX, tokenIdx, expId, event, dataInBf16, dataInFp32);

        __ubuf__ float* weightUb = nullptr;
        if (!TopkWeightsPrefetch) {
            weightUb = topKWeightsTensor_ + localIdx * topK_ + expId;
        }
        // 首槽与后续槽分别实例化，避免 VF 的每次向量迭代都判断初始化状态。
        if (CombineQuantMode == COMBINE_NO_QUANT) {
            AccumulateLayeredUnpermuteInput(dataResFp32Tensor_, dataInBf16, weightUb, k_, !TopkWeightsPrefetch,
                                            expId == 0);
        } else {
            AccumulateLayeredUnpermuteInput(dataResFp32Tensor_, dataInFp32, weightUb, k_, !TopkWeightsPrefetch,
                                            expId == 0);
        }
        PipeNotify<PIPE_V, PIPE_MTE2>(event);
        inputBufferIdx = (inputBufferIdx + 1U) % static_cast<uint32_t>(bufferConfig.inputBufferCount);
    }
}

template <TemplateMegaMoeLayeredTypeClass>
__aicore__ inline void MegaMoeLayered<TemplateMegaMoeLayeredTypeFunc>::UnpermuteServerCombine()
{
    const auto& workspace = params_.workspace.layout;
    const auto& buffer = params_.workspace.base;
    const auto& peermem = params_.communication.layout;
    const auto& commBuffer = params_.communication.base;
    const uint32_t floatBytes =
        Ops::Base::CeilAlign(k_ * static_cast<uint32_t>(sizeof(float)), static_cast<uint32_t>(ALIGN_32));
    LocalTensor<float> inputTensor(TPosition::VECIN, ALIGN_512, k_);
    LocalTensor<float> sumTensor(TPosition::VECCALC, ALIGN_512 + floatBytes, k_);
    LocalTensor<bfloat16_t> bf16Tensor(TPosition::VECIN, ALIGN_512 + 2U * floatBytes, k_);
    __gm__ OutputType* outputGlobalTensor = nullptr;
    outputGlobalTensor = reinterpret_cast<__gm__ OutputType*>(params_.y2GmAddr);
    LocalTensor<OutputType> outputTensor = bf16Tensor.template ReinterpretCast<OutputType>();
    __gm__ bfloat16_t* combineResultsGlobalTensor = nullptr;
    combineResultsGlobalTensor = reinterpret_cast<__gm__ bfloat16_t*>(peermem.combineReceive.get_ptr(commBuffer));
    __gm__ bfloat16_t* sharedResultsGlobalTensor = nullptr;
    sharedResultsGlobalTensor = reinterpret_cast<__gm__ bfloat16_t*>(workspace.shared.gmm2Output.get_ptr(buffer));
    const WorkRange tokenRange = GetBalancedWorkRange(m_, {.jobIndex = aivCoreIdx_, .totalJobs = blockAivNum_});
    uint32_t readySharedGroup = UINT32_MAX; // 每次调用重新开始，不复用上轮就绪状态。
    for (uint32_t tokenOffset = 0U; tokenOffset < tokenRange.count; ++tokenOffset) {
        const uint32_t token = tokenRange.start + tokenOffset;
        Duplicate(sumTensor, 0.0f, k_);
        PipeFence<PIPE_MTE3, PIPE_V, SYNC_EVENT_ID2>();
        // 预置 bf16 缓冲可搬入事件，Server 和共享专家循环共用此链。
        PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
        for (uint32_t server = 0; server < serverNum_; ++server) {
            auto bitmapPtr = reinterpret_cast<__gm__ uint64_t*>(peermem.sourceReady.get_ptr(commBuffer) +
                                                                RelayFlagOffset(server, token));
            if (static_cast<uint32_t>(GmRead(bitmapPtr)) == 0U) {
                continue;
            }
            PipeWait<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
            CopyGmToUb(reinterpret_cast<__ubuf__ bfloat16_t*>((bf16Tensor).GetPhyAddr()),
                       (combineResultsGlobalTensor + ((static_cast<uint64_t>(token) * serverNum_ + server) * k_)),
                       k_ * static_cast<uint32_t>(sizeof(bfloat16_t)));
            PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();
            Cast(inputTensor, bf16Tensor, RoundMode::CAST_NONE, k_);
            PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
            Add(sumTensor, sumTensor, inputTensor, k_);
        }
        const uint32_t tokenGroup = token / COMBINE_TOKEN_GROUP_SIZE;
        const bool waitForGroup = tokenGroup != readySharedGroup;
        for (uint32_t expert = 0; expert < sharedExpertNum_; ++expert) {
            if (waitForGroup) {
                WaitForSharedExpertResult(expert, token);
            }
            PipeWait<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
            CopyGmToUb(reinterpret_cast<__ubuf__ bfloat16_t*>((bf16Tensor).GetPhyAddr()),
                       (sharedResultsGlobalTensor + ((static_cast<uint64_t>(expert) * m_ + token) * k_)),
                       k_ * static_cast<uint32_t>(sizeof(bfloat16_t)));
            PipeFence<PIPE_MTE2, PIPE_V, SYNC_EVENT_ID2>();
            Cast(inputTensor, bf16Tensor, RoundMode::CAST_NONE, k_);
            PipeNotify<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
            Add(sumTensor, sumTensor, inputTensor, k_);
        }
        readySharedGroup = tokenGroup;
        // 回收最后一个事件；没有输入时回收入口预置事件。
        PipeWait<PIPE_V, PIPE_MTE2>(static_cast<event_t>(SYNC_EVENT_ID2));
        Cast(outputTensor, sumTensor, RoundMode::CAST_RINT, k_);
        PipeFence<PIPE_V, PIPE_MTE3, SYNC_EVENT_ID2>();
        CopyUbToGm((outputGlobalTensor + (static_cast<uint64_t>(token) * k_)),
                   reinterpret_cast<__ubuf__ OutputType*>((outputTensor).GetPhyAddr()),
                   k_ * static_cast<uint32_t>(sizeof(OutputType)));
    }
    // 返回前等待本核最后一次输出搬运完成。
    PipeFence<PIPE_MTE3, PIPE_S, SYNC_EVENT_ID2>();
}

} // namespace MegaMoeImpl
#endif // MEGA_MOE_LAYERED_UNPERMUTE_H
