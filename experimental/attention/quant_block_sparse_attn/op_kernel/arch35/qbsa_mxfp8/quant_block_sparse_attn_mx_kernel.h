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
 * \file quant_block_sparse_attn_mx_kernel.h
 * \brief Fully quantized MXFP8 kernel entry for QuantBlockSparseAttn.
 */
#ifndef QUANT_BLOCK_SPARSE_ATTN_MX_KERNEL_H_
#define QUANT_BLOCK_SPARSE_ATTN_MX_KERNEL_H_

#include "kernel_operator.h"
#include "qbsa_mxfp8_tiling_data.h"
#include "../common/quant_block_sparse_attn_common_arch35.h"
#include "../common/quant_block_sparse_attn_attenmask.h"
#include "../common/quant_block_sparse_attn_kvcache.h"
#include "quant_block_sparse_attn_mx_block_cube.h"
#include "quant_block_sparse_attn_mx_block_vec.h"
#include "../../common/attn_buffer_manager.h"
#include "../../common/buffers_policy.h"
#include "qbsa_mxfp8_runtime.h"

using namespace AscendC;
using namespace optiling;

namespace BaseApi {
// Independent MX tiling/kernel path: two S2_SPLIT C1/V1 subLoops produce P[128,S2_BASE], followed by one C2/V2.
template <typename inputType, typename mmType, typename outputType, QBSALayout layout, QBSALayout kvLayout,
          S1TemplateType s1TemplateType, S2TemplateType s2TemplateType, DTemplateType dTemplateType,
          DTemplateType dVTemplateType, bool hasAtten, bool hasLse, bool isPa, bool useDn = true>
class QuantBlockSparseAttnMxKernel {
public:
    using INPUT_T = inputType;
    using MM_T = mmType;
    using OUTPUT_T = outputType;
    static constexpr bool USE_DN = useDn;
    static constexpr bool HAS_ATTEN = hasAtten;
    static constexpr bool HAS_LSE = hasLse;
    static constexpr bool IS_PA = isPa;
    static constexpr QBSALayout LAYOUT = layout;
    static constexpr QBSALayout KV_LAYOUT = kvLayout;
    static constexpr uint32_t M_BASE = static_cast<uint32_t>(s1TemplateType);
    static constexpr uint32_t S2_BASE = static_cast<uint32_t>(s2TemplateType);
    static constexpr uint32_t S2_SPLIT = S2_BASE / 2U;
    static constexpr uint32_t D_BASE = static_cast<uint32_t>(dTemplateType);
    static constexpr uint32_t DV_BASE = static_cast<uint32_t>(dVTemplateType);
    static_assert((LAYOUT == QBSALayout::TND || LAYOUT == QBSALayout::BSND || LAYOUT == QBSALayout::BNSD) &&
                      KV_LAYOUT == QBSALayout::PA_BNBD && IS_PA,
                  "MXFullQuantMode currently supports TND/BSND/BNSD query and PA_BNBD KV");
    static_assert(M_BASE == 128U && S2_BASE == (D_BASE == 256U ? 256U : 512U) && D_BASE == DV_BASE &&
                      (D_BASE == 64U || D_BASE == 128U || D_BASE == 256U),
                  "MXFullQuantMode requires S1=128, S2=(D256 ? 256 : 512) and D=DV in {64,128,256}");
    // C2 is delayed by two tasks and V2 by one more. A single BMM2 UB acts as a pipeline register across iterations:
    // AIC executes C1(i) / C2(i-2); AIV executes V2(i-3) / V1(i).
    static constexpr uint32_t C2_DELAY = 2U;
    static constexpr uint32_t V2_DELAY = 3U;
    static constexpr uint32_t PIPELINE_TASK_CACHE_SIZE = V2_DELAY + 1U;

    __aicore__ inline QuantBlockSparseAttnMxKernel() = default;

    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value,
                                __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseSeqLen, __gm__ uint8_t* attenMask,
                                __gm__ uint8_t* metadata, __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t*, __gm__ uint8_t*,
                                __gm__ uint8_t* seqUsedKv, __gm__ uint8_t* blockTable, __gm__ uint8_t* qScale,
                                __gm__ uint8_t* kScale, __gm__ uint8_t* vScale, __gm__ uint8_t* pScale,
                                __gm__ uint8_t* softmaxLse, __gm__ uint8_t* attentionOut, __gm__ uint8_t*,
                                const QuantBlockSparseAttnMxfp8TilingData* __restrict tiling, TPipe* tPipe)
    {
        // Cross-core buffer IDs are generated from core-local state, which ACL graph replay reuses. Reset the counters
        // at each launch to ensure AIC and AIV use the same event ID sequence.
        fa_base_matmul::idCounterNum = 0;
        // Cube transfers Q/K/V through PA addressing using block_table.
        pipe = tPipe;
        tilingData = tiling;
        queryPtr = query;
        keyPtr = key;
        valuePtr = value;
        sparseIndicesGm.SetGlobalBuffer((__gm__ int32_t*)sparseIndices);
        sparseSeqLenGm.SetGlobalBuffer((__gm__ int32_t*)sparseSeqLen);
        metadataGm.SetGlobalBuffer((__gm__ int32_t*)metadata);
        if constexpr (LAYOUT == QBSALayout::TND) {
            cuSeqlensQGm.SetGlobalBuffer((__gm__ int32_t*)cuSeqlensQ);
            dc_preload(reinterpret_cast<__gm__ uint64_t*>(cuSeqlensQ), 0);
        }
        seqUsedKvGm.SetGlobalBuffer((__gm__ int32_t*)seqUsedKv);
        // Prefetch sequence length metadata to overlap HBM latency with InitBuffers.
        dc_preload(reinterpret_cast<__gm__ uint64_t*>(seqUsedKv), 0);
        InitConstInfo();
        InitMMResBuf();
        cubeBlock.Init(pipe, &l1BufferManager, queryPtr, keyPtr, valuePtr, blockTable, qScale, kScale, vScale);
        __gm__ uint8_t* pScalePtr = (tilingData->commonParams.baseParams.pScaleShapeSize == 0U) ? nullptr : pScale;
        vecBlock.Init(pipe, pScalePtr, softmaxLse, attentionOut, attenMask,
                      tilingData->commonParams.maskParams.attenMaskS2Size, tilingData->mxParams.pScaleDtype);
    }

    __aicore__ inline void Process()
    {
        if (constInfo.aicIdx >= constInfo.coreNum || constInfo.aicIdx >= QBSA_FA_AIC_CORE_NUM) {
            return;
        }
        CrossCoreBufferInit();
        if ASCEND_IS_AIC {
            cubeBlock.InitBuffers();
        } else {
            vecBlock.InitBuffers();
        }
        ProcessMainLoop();
        CrossCoreBufferUninit();
        // Only active cores reach InitBuffers and this matching cleanup.
        if ASCEND_IS_AIC {
            cubeBlock.UninitBuffers();
        } else {
            vecBlock.UninitBuffers();
        }
    }

private:
    __aicore__ inline void InitConstInfo()
    {
        // Two AIV subBlocks share one logical AIC ID.
        if ASCEND_IS_AIC {
            constInfo.aicIdx = GetBlockIdx();
        } else {
            constInfo.aicIdx = GetBlockIdx() / GetSubBlockNum();
            constInfo.subBlockIdx = GetSubBlockIdx();
        }

        const auto& baseParams = tilingData->commonParams.baseParams;
        const auto& pageAttentionParams = tilingData->commonParams.paParams;
        const auto& sparseParams = tilingData->commonParams.sparseParams;
        const auto& scaleParams = tilingData->mxParams;

        constInfo.qSeqSize = scaleParams.qSeqSize;
        constInfo.n2Size = baseParams.n2Size;
        constInfo.gSize = baseParams.gSize;
        constInfo.realN2Size = baseParams.n2Size * baseParams.gSize;
        constInfo.dSize = baseParams.dSize;
        constInfo.dSizeV = baseParams.dSizeV;
        constInfo.scaleValue = baseParams.scaleValue;
        constInfo.coreNum = baseParams.coreNum;
        constInfo.n2GD = constInfo.realN2Size * constInfo.dSize;
        constInfo.attentionOutStride = (constInfo.realN2Size - 1U) * constInfo.dSizeV * sizeof(OUTPUT_T);
        constInfo.softmaxLseStride = (constInfo.realN2Size - 1U) * sizeof(float);

        constInfo.maxBlockNumPerBatch = pageAttentionParams.blockTableDim2;
        constInfo.paBlockStride = pageAttentionParams.paBlockStride;
        constInfo.qSparseBlockSize = pageAttentionParams.qBlockSize;
        constInfo.kvSparseBlockSize = pageAttentionParams.kvBlockSize;
        constInfo.paBlockSize = pageAttentionParams.blockSize;

        constInfo.maxQb = sparseParams.sparseSeqLenStride;
        constInfo.maxKb = sparseParams.sparseIndicesStride;

        // QScale uses [T,N,D/64,2]; K/V scales use BNBD mapping for PA.
        constInfo.scaleLastDim = scaleParams.scaleLastDim;
        constInfo.queryScaleDSize = scaleParams.queryScaleDSize;
        constInfo.keyScaleDSize = scaleParams.keyScaleDSize;
        constInfo.valueScaleDSize = scaleParams.valueScaleDSize;
        constInfo.qScaleN1D = constInfo.realN2Size * constInfo.queryScaleDSize * constInfo.scaleLastDim;
        constInfo.queryRowStride = constInfo.n2GD;
        constInfo.queryScaleRowStride = constInfo.qScaleN1D;
        if constexpr (LAYOUT == QBSALayout::BNSD) {
            constInfo.queryRowStride = constInfo.dSize;
            constInfo.queryScaleRowStride = constInfo.queryScaleDSize * constInfo.scaleLastDim;
            constInfo.attentionOutStride = 0U;
        }
        if constexpr (LAYOUT == QBSALayout::BSND || LAYOUT == QBSALayout::BNSD) {
            constInfo.softmaxLseStride = 0U;
        }
        constInfo.kScaleN2D = static_cast<uint32_t>(scaleParams.kScaleStrides.n2Stride);
        constInfo.valueScaleN2D = static_cast<uint32_t>(scaleParams.vScaleStrides.n2Stride);
    }

    __aicore__ inline void InitMMResBuf()
    {
        // C1 UB is [64,S2_SPLIT] in FP32; C2 UB is [64,DV_BASE] in FP32.
        constexpr uint32_t mm1ResultSize = M_BASE / CV_RATIO * S2_BASE * sizeof(MM_T) / 2U;
        constexpr uint32_t mm2ResultSize = M_BASE / CV_RATIO * DV_BASE * sizeof(MM_T);
        // L1 stores P(e4m3) and PScale(e8m0) contiguously for C2 dequantization during transfer.
        constexpr uint32_t mm2LeftSize = M_BASE * S2_BASE * sizeof(INPUT_T) + M_BASE * S2_BASE / 32U;
        l1BufferManager.Init(pipe, 512U * 1024U);
        l1PBuffers.Init(l1BufferManager, mm2LeftSize);
        ubBufferManager.Init(pipe, mm1ResultSize * 2U + mm2ResultSize);
        bmm1Buffers.Init(ubBufferManager, mm1ResultSize);
        bmm2Buffers.Init(ubBufferManager, mm2ResultSize);
    }

    __aicore__ inline void CrossCoreBufferInit()
    {
        if ASCEND_IS_AIV {
            bmm1Buffers.Get().SetCrossCore();
            bmm1Buffers.Get().SetCrossCore();
            bmm2Buffers.Get().SetCrossCore();
        }
    }

    __aicore__ inline void CrossCoreBufferUninit()
    {
        if ASCEND_IS_AIC {
            bmm1Buffers.Get().WaitCrossCore();
            bmm1Buffers.Get().WaitCrossCore();
            bmm2Buffers.Get().WaitCrossCore();
        }
    }

    __aicore__ inline uint32_t GetS1LoopStart(uint32_t bn1Idx, uint32_t bn1StartIdx, uint32_t s1StartIdx) const
    {
        return bn1Idx == bn1StartIdx ? s1StartIdx : 0U;
    }

    __aicore__ inline uint32_t GetS1LoopEnd(uint32_t bn1Idx, uint32_t bn1EndIdx, uint32_t s1EndIdx,
                                            uint32_t actualS1Size) const
    {
        const uint32_t metadataEnd = bn1Idx == bn1EndIdx - 1U ? s1EndIdx : 0U;
        return MxS1LoopEnd(actualS1Size, constInfo.qSparseBlockSize, metadataEnd);
    }

    __aicore__ inline bool ShouldMoveSparseBlockBack(int64_t previousBlockIdx, int64_t currentBlockIdx) const
    {
        // Sort valid blocks in ascending order and place invalid blocks last.
        if (previousBlockIdx < 0) {
            return currentBlockIdx >= 0;
        }
        if (currentBlockIdx < 0) {
            return false;
        }
        return previousBlockIdx > currentBlockIdx;
    }

    __aicore__ inline void SortSparseBlocks(MxRunInfo& runInfo) const
    {
        // Up to 8 blocks; generate offsets and mask state after sorting.
        for (uint32_t i = 1U; i < runInfo.sparseBlockCount; ++i) {
            const int64_t currentBlockIdx = runInfo.sparseBlockIdx[i];
            uint32_t insertPos = i;
            while (insertPos > 0U &&
                   ShouldMoveSparseBlockBack(runInfo.sparseBlockIdx[insertPos - 1U], currentBlockIdx)) {
                runInfo.sparseBlockIdx[insertPos] = runInfo.sparseBlockIdx[insertPos - 1U];
                --insertPos;
            }
            runInfo.sparseBlockIdx[insertPos] = currentBlockIdx;
        }
    }

    __aicore__ inline void FillRunInfo(MxRunInfo& runInfo, uint64_t loop, uint32_t mLoop, uint32_t bIdx, uint32_t n1Idx,
                                       uint32_t n2Idx, uint32_t queryTokenBase, uint32_t actualS1Size,
                                       uint32_t actualS2Size, uint32_t s1OuterIdx, uint32_t s2LoopIdx,
                                       uint32_t s2LoopEnd, uint32_t sparseBlockCount, uint64_t sparseBase,
                                       bool isFirstValidS2Loop)
    {
        // Build runtime parameters for a logical S2 tile of S2_BASE tokens.
        // Zero only the required fields.
        runInfo.actSingleLoopS2Size = 0U;
        runInfo.loop = static_cast<uint32_t>(loop);
        runInfo.mLoop = mLoop;
        runInfo.isValid = true;
        runInfo.bIdx = bIdx;
        runInfo.realN2Idx = n1Idx;
        runInfo.n2Idx = n2Idx;
        // s1OuterIdx is the sparse Q block index. MX still supports a maximum M of 128;
        // with q_block_size=64, each task processes only its own 64 rows to avoid crossing two sparse lists.
        runInfo.s1Idx = s1OuterIdx * constInfo.qSparseBlockSize;
        runInfo.queryTokenBase = queryTokenBase;
        runInfo.actS1Size = actualS1Size;
        runInfo.actS2Size = actualS2Size;
        runInfo.actMSize = constInfo.qSparseBlockSize;
        if (unlikely(runInfo.s1Idx + constInfo.qSparseBlockSize > runInfo.actS1Size)) {
            runInfo.actMSize = static_cast<uint32_t>(runInfo.actS1Size) - runInfo.s1Idx;
        }
        if ASCEND_IS_AIV {
            const uint32_t actMSizeAlign32 = (runInfo.actMSize + 31U) >> 5U << 5U;
            runInfo.actVecMSize = runInfo.actMSize <= 16U ? runInfo.actMSize : (actMSizeAlign32 >> 1U);
            runInfo.vecMbaseIdx = 0U;
            if (constInfo.subBlockIdx == 1U) {
                runInfo.vecMbaseIdx = runInfo.actVecMSize;
                runInfo.actVecMSize = runInfo.actMSize - runInfo.actVecMSize;
            }
            if constexpr (HAS_ATTEN) {
                runInfo.attenMaskSubLoopBits = 0U;
            }
        }
        runInfo.sparseBlockCount = sparseBlockCount;
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            runInfo.sparseBlockIdx[i] = sparseIndicesGm.GetValue(sparseBase + s2LoopIdx + i);
        }
        SortSparseBlocks(runInfo);
        for (uint32_t i = 0U; i < runInfo.sparseBlockCount; ++i) {
            const int64_t sparseBlkIdx = runInfo.sparseBlockIdx[i];
            uint64_t sparseTokenOffset = 0U;
            uint32_t sparseBlockRealSize = 0U;
            runInfo.sparseBlockTileOffset[i] = runInfo.actSingleLoopS2Size;
            // Exclude negative indices from concatenation.
            if (sparseBlkIdx >= 0) {
                sparseTokenOffset = static_cast<uint64_t>(sparseBlkIdx) * constInfo.kvSparseBlockSize;
                const int64_t remainS2 =
                    static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(sparseTokenOffset);
                sparseBlockRealSize = remainS2 > static_cast<int64_t>(constInfo.kvSparseBlockSize) ?
                                          constInfo.kvSparseBlockSize :
                                          (remainS2 > 0 ? static_cast<uint32_t>(remainS2) : 0U);
            }
            runInfo.sparseBlockTokenOffset[i] = sparseTokenOffset;
            runInfo.sparseBlockRealSize[i] = sparseBlockRealSize;
            if constexpr (HAS_ATTEN) {
                if ASCEND_IS_AIV {
                    runInfo.sparseBlockPartialMask[i] = false;
                    if (likely(sparseBlockRealSize != 0U)) {
                        const int64_t causalDelta =
                            static_cast<int64_t>(runInfo.actS2Size) - static_cast<int64_t>(runInfo.actS1Size);
                        int64_t sparseBlkIdxForMask = sparseBlkIdx;
                        AttentionMaskFullProcessingOrRequired(
                            static_cast<int64_t>(runInfo.s1Idx), static_cast<int64_t>(sparseTokenOffset), causalDelta,
                            sparseBlockRealSize, sparseBlkIdxForMask, runInfo.sparseBlockPartialMask[i]);
                        if (unlikely(runInfo.sparseBlockPartialMask[i])) {
                            const uint32_t blockTileStart = runInfo.sparseBlockTileOffset[i];
                            const uint32_t blockTileEnd = blockTileStart + sparseBlockRealSize;
                            if (blockTileStart < S2_SPLIT) {
                                runInfo.attenMaskSubLoopBits |= 0x1U;
                            }
                            if (blockTileEnd > S2_SPLIT) {
                                runInfo.attenMaskSubLoopBits |= 0x2U;
                            }
                        }
                    }
                }
            }
            runInfo.actSingleLoopS2Size += sparseBlockRealSize;
        }
        runInfo.isFirstS2Loop = isFirstValidS2Loop;
        runInfo.isLastS2Loop = s2LoopIdx + runInfo.sparseBlockCount - 1U >= s2LoopEnd;
        if ASCEND_IS_AIC {
            // GM element offsets may exceed 32 bits; widen from the token base and calculate offsets within each head
            // in 64 bits.
            if constexpr (LAYOUT == QBSALayout::TND) {
                const uint64_t queryTokenOffset = runInfo.queryTokenBase + runInfo.s1Idx;
                runInfo.queryOffset = queryTokenOffset * static_cast<uint64_t>(constInfo.n2GD) +
                                      static_cast<uint64_t>(runInfo.realN2Idx) * constInfo.dSize;
                runInfo.queryScaleOffset =
                    queryTokenOffset * static_cast<uint64_t>(constInfo.qScaleN1D) +
                    static_cast<uint64_t>(runInfo.realN2Idx) * constInfo.queryScaleDSize * constInfo.scaleLastDim;
            } else {
                const uint64_t querySlot = MxQuerySlot<LAYOUT>(runInfo.queryTokenBase, runInfo.s1Idx, runInfo.realN2Idx,
                                                               constInfo.realN2Size, constInfo.qSeqSize);
                runInfo.queryOffset = querySlot * constInfo.dSize;
                runInfo.queryScaleOffset = querySlot * constInfo.queryScaleDSize * constInfo.scaleLastDim;
            }
        }
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, MxRunInfo taskRunInfo[PIPELINE_TASK_CACHE_SIZE])
    {
        MxRunInfo& runInfoCur = taskRunInfo[loop % PIPELINE_TASK_CACHE_SIZE];
        if ASCEND_IS_AIC {
            // AIC first produces the current C1, then consumes P from two iterations earlier to produce C2 for the next
            // V2.
            if (likely(runInfoCur.isValid)) {
                if (likely(runInfoCur.actSingleLoopS2Size > S2_SPLIT)) {
                    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& c1Out0 = bmm1Buffers.Get();
                    Buffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH>& c1Out1 = bmm1Buffers.Get();
                    cubeBlock.IterateBmm1ReuseQ(c1Out0, c1Out1, runInfoCur, constInfo);
                } else {
                    cubeBlock.IterateBmm1(bmm1Buffers.Get(), runInfoCur, constInfo);
                }
            }
            if (likely(loop >= C2_DELAY)) {
                MxRunInfo& runInfoC2 = taskRunInfo[(loop - C2_DELAY) % PIPELINE_TASK_CACHE_SIZE];
                if (likely(runInfoC2.isValid)) {
                    cubeBlock.IterateBmm2(bmm2Buffers.Get(), l1PBuffers, runInfoC2, constInfo);
                    runInfoC2.isValid = false;
                }
            }
        } else {
            // AIV first consumes the C2 already available from the previous iteration, overlapping useful V2 work with
            // current C1 production.
            if (likely(loop >= V2_DELAY)) {
                MxRunInfo& runInfoV2 = taskRunInfo[(loop - V2_DELAY) % PIPELINE_TASK_CACHE_SIZE];
                if (likely(runInfoV2.isValid)) {
                    vecBlock.ProcessVec2(bmm2Buffers.Get(), runInfoV2, constInfo);
                    runInfoV2.isValid = false;
                }
            }
            if (likely(runInfoCur.isValid)) {
                Buffer<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD>& pBuf = l1PBuffers.GetVec();
                // ProcessMainLoop filters fully invalid sections; valid tasks always start with subLoop0.
                vecBlock.template ProcessVec1<0U>(pBuf, bmm1Buffers.Get(), runInfoCur, constInfo);
                if (likely(runInfoCur.actSingleLoopS2Size > S2_SPLIT)) {
                    vecBlock.template ProcessVec1<1U>(pBuf, bmm1Buffers.Get(), runInfoCur, constInfo);
                }
            }
        }
    }

    __aicore__ inline void ProcessMainLoop()
    {
        // Four task state slots cover the current task, C2(i-2), and V2(i-3).
        MxRunInfo taskRunInfo[PIPELINE_TASK_CACHE_SIZE] = {};
        uint64_t loop = 0U;
        uint32_t mLoop = 0U;
        // S2_BASE=256/512, kvSparseBlockSize=64/128; calculate sparse blocks per task from template parameters.
        const uint32_t sparseBlocksPerTask = S2_BASE / constInfo.kvSparseBlockSize;
        // Metadata may contain multiple sections; each provides independent BN1/S1 boundaries for the current AIC.
        const uint32_t sectionNum = GetBsaSectionNum(metadataGm);
        uint32_t cachedBIdx = static_cast<uint32_t>(-1);
        uint32_t queryTokenBase = 0U;
        uint32_t actualS1Size = 0U;
        uint32_t actualS2Size = 0U;
        for (uint32_t sectionIdx = 0U; sectionIdx < sectionNum; ++sectionIdx) {
            // MX only uses BN1/S1 boundaries; read enable first and skip the other 7 fields for disabled sections.
            const uint64_t metadataBase = GetBsaCoreMetadataOffset(sectionIdx, constInfo.aicIdx);
            if (unlikely(metadataGm.GetValue(metadataBase + QBSA_FA_CORE_ENABLE_INDEX) == 0U)) {
                continue;
            }
            const uint32_t bn1StartIdx = metadataGm.GetValue(metadataBase + QBSA_FA_BN1_START_INDEX);
            uint32_t bn1EndIdx = metadataGm.GetValue(metadataBase + QBSA_FA_BN1_END_INDEX);
            const uint32_t s1StartIdx = metadataGm.GetValue(metadataBase + QBSA_FA_S1_START_INDEX);
            const uint32_t s1EndIdx = metadataGm.GetValue(metadataBase + QBSA_FA_S1_END_INDEX);
            if (s1EndIdx != 0U) {
                ++bn1EndIdx;
            }
            if (unlikely(bn1EndIdx <= bn1StartIdx)) {
                continue;
            }

            for (uint32_t bn1Idx = bn1StartIdx; bn1Idx < bn1EndIdx; ++bn1Idx) {
                const uint32_t bIdx = bn1Idx / constInfo.realN2Size;
                const uint32_t n1Idx = bn1Idx % constInfo.realN2Size;
                const uint32_t n2Idx = n1Idx / constInfo.gSize;
                // Reuse sequence lengths across all N1 heads of the same batch to avoid repeated GM accesses.
                if (unlikely(bIdx != cachedBIdx)) {
                    if constexpr (LAYOUT == QBSALayout::BSND || LAYOUT == QBSALayout::BNSD) {
                        queryTokenBase = bIdx * constInfo.qSeqSize;
                        actualS1Size = constInfo.qSeqSize;
                    } else {
                        queryTokenBase = static_cast<uint32_t>(cuSeqlensQGm.GetValue(bIdx));
                        actualS1Size = static_cast<uint32_t>(cuSeqlensQGm.GetValue(bIdx + 1U)) - queryTokenBase;
                    }
                    actualS2Size = static_cast<uint32_t>(seqUsedKvGm.GetValue(bIdx));
                    cachedBIdx = bIdx;
                }
                const uint32_t s1LoopStart = GetS1LoopStart(bn1Idx, bn1StartIdx, s1StartIdx);
                const uint32_t s1LoopEnd = GetS1LoopEnd(bn1Idx, bn1EndIdx, s1EndIdx, actualS1Size);
                for (uint32_t s1OuterIdx = s1LoopStart; s1OuterIdx < s1LoopEnd; ++s1OuterIdx) {
                    // Number of valid sparse blocks in the current query block.
                    const int32_t sparseLen =
                        sparseSeqLenGm.GetValue(static_cast<uint64_t>(bn1Idx) * constInfo.maxQb + s1OuterIdx);
                    bool hasValidS2Task = false;
                    if (likely(sparseLen > 0)) {
                        // sparseBase points to sparseIndices[B,N1,Qb,0].
                        const uint64_t sparseBase = static_cast<uint64_t>(bn1Idx) * constInfo.maxQb * constInfo.maxKb +
                                                    static_cast<uint64_t>(s1OuterIdx) * constInfo.maxKb;
                        const uint32_t s2LoopEnd = static_cast<uint32_t>(sparseLen - 1);
                        for (uint32_t s2LoopIdx = 0U; s2LoopIdx <= s2LoopEnd; s2LoopIdx += sparseBlocksPerTask) {
                            MxRunInfo& runInfo = taskRunInfo[loop % PIPELINE_TASK_CACHE_SIZE];
                            const uint32_t remainSparseBlockCount = static_cast<uint32_t>(sparseLen) - s2LoopIdx;
                            const uint32_t sparseBlockCount = remainSparseBlockCount > sparseBlocksPerTask ?
                                                                  sparseBlocksPerTask :
                                                                  remainSparseBlockCount;
                            FillRunInfo(runInfo, loop, mLoop, bIdx, n1Idx, n2Idx, queryTokenBase, actualS1Size,
                                        actualS2Size, s1OuterIdx, s2LoopIdx, s2LoopEnd, sparseBlockCount, sparseBase,
                                        !hasValidS2Task);
                            if (likely(runInfo.actSingleLoopS2Size != 0U && runInfo.actMSize != 0U)) {
                                ExecuteTask(loop, taskRunInfo);
                                hasValidS2Task = true;
                                ++loop;
                            } else {
                                runInfo.isValid = false;
                            }
                        }
                    }
                    if (unlikely(!hasValidS2Task)) {
                        if ASCEND_IS_AIV {
                            vecBlock.ClearEmptyQBlock(n1Idx, queryTokenBase, actualS1Size, s1OuterIdx, constInfo);
                        }
                    } else {
                        // Only Q blocks entering the pipeline occupy a set of softmax state slots.
                        ++mLoop;
                    }
                }
            }
        }
        // After the last valid task, C2/V2 lag by 2/3 steps; no repeated scans of the four state slots are needed.
        if (loop != 0U) {
            for (uint32_t drainIdx = 0U; drainIdx < V2_DELAY; ++drainIdx) {
                ExecuteTask(loop, taskRunInfo);
                ++loop;
            }
        }
    }

    TPipe* pipe = nullptr;
    const QuantBlockSparseAttnMxfp8TilingData* __restrict tilingData = nullptr;
    regbasemx::MxConstInfo constInfo;
    __gm__ uint8_t* queryPtr = nullptr;
    __gm__ uint8_t* keyPtr = nullptr;
    __gm__ uint8_t* valuePtr = nullptr;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> sparseSeqLenGm;
    GlobalTensor<int32_t> metadataGm;
    GlobalTensor<int32_t> cuSeqlensQGm;
    GlobalTensor<int32_t> seqUsedKvGm;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    BuffersPolicySingleBuffer<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm2Buffers;
    BuffersPolicy3buff<BufferType::L1, SyncType::CROSS_CORE_SYNC_FORWARD> l1PBuffers;
    QuantBlockSparseAttnMxBlockCube<INPUT_T, MM_T, OUTPUT_T, LAYOUT, KV_LAYOUT, s1TemplateType, s2TemplateType,
                                    dTemplateType, dVTemplateType, IS_PA, USE_DN>
        cubeBlock;
    QuantBlockSparseAttnMxBlockVec<INPUT_T, MM_T, OUTPUT_T, LAYOUT, KV_LAYOUT, s1TemplateType, s2TemplateType,
                                   dTemplateType, dVTemplateType, HAS_ATTEN, HAS_LSE, IS_PA, USE_DN>
        vecBlock;
};
} // namespace BaseApi

#endif // QUANT_BLOCK_SPARSE_ATTN_MX_KERNEL_H_
