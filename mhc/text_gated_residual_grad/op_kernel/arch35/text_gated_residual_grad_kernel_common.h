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
 * \file text_gated_residual_grad_kernel_common.h
 * \brief Shared TextGatedResidualGrad initialization and Cube/Vector execution.
 */
#ifndef TEXT_GATED_RESIDUAL_GRAD_KERNEL_COMMON_H
#define TEXT_GATED_RESIDUAL_GRAD_KERNEL_COMMON_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "text_gated_residual_grad_tiling_data_arch35.h"
#include "text_gated_residual_grad_pipeline_arch35.h"
#include "text_gated_residual_grad_chunk_state.h"
#include "text_gated_residual_grad_vector_arch35.h"
#include "text_gated_residual_grad_credit_sync.h"

namespace TextGatedResidualGrad {

// Ascend 950 and Atlas A2/A3 share Init/Process/RunCube. Each entry header supplies its Cube backend
// and tiles. USE_DUP_T/G56_PACKED_T remain available for transpose fallback paths.
template <typename D_T, int HAS_GAMMA, class GemmT, bool USE_DUP_T, bool G56_PACKED_T = true>
class KernelTextGatedResidualGrad {
public:
    __aicore__ inline void Init(GM_ADDR gradMixed, GM_ADDR x, GM_ADDR wDown, GM_ADDR wUp, GM_ADDR invRms,
                                GM_ADDR downPreAct, GM_ADDR up, GM_ADDR gradInject, GM_ADDR wGate, GM_ADDR gamma,
                                GM_ADDR inject, GM_ADDR dx, GM_ADDR dwDown, GM_ADDR dwUp, GM_ADDR dwGate,
                                GM_ADDR dgamma, GM_ADDR workspace,
                                const TextGatedResidualGradTilingDataArch35* tilingData, AscendC::TPipe* pipe)
    {
        taskRows_ = static_cast<uint32_t>(tilingData->taskRows);
        fullBatchRows_ = static_cast<uint32_t>(tilingData->fullBatchRows);
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A2 && HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        lowpDx_ = tilingData->lowpDx != 0;
#endif
        n_ = static_cast<uint32_t>(tilingData->n);
        c_ = static_cast<uint32_t>(tilingData->c);
        r_ = static_cast<uint32_t>(tilingData->r);
        d_ = static_cast<uint32_t>(tilingData->d);
        taskCount_ = static_cast<uint32_t>(tilingData->taskCount);
        workspaceSlots_ = static_cast<uint32_t>(tilingData->workspaceSlots);
        splitkFactor_ = static_cast<uint32_t>(tilingData->splitkFactor);
        hasGate_ = static_cast<uint32_t>(tilingData->hasGate);
        scheduleBeatCount_ = static_cast<uint32_t>(tilingData->scheduleBeatCount);
        aicNum_ = static_cast<uint32_t>(tilingData->aicNum);
        workspace_ = workspace;
        // A5 only chooses a full-batch chunk for inputs with gamma.
        fullBatch_ = (HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5 || HAS_GAMMA) &&
                     tilingData->scheduleMode == HC_GRAD_FULL_BATCH_SCHEDULE;
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A2 && HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        producerConsumer_ = (tilingData->scheduleMode != 0) && !USE_DUP_T && !G56_PACKED_T;
#endif
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        singleTask_ = taskCount_ == 1U && tilingData->nnMatmul == 0;
#endif
        gmWUp_ = wUp;
        gmWDown_ = wDown;
        gmWGate_ = wGate;
        const uint32_t retainedRows = fullBatch_ ? fullBatchRows_ : 0U;
        const uint32_t denseColumns = r_ + (hasGate_ != 0U ? c_ : 0U);
        const uint32_t dxColumns =
            HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5 && fullBatch_ ? HcGradAlign16(r_ + HcGradAlign16(c_)) : denseColumns;
        layout_ = BuildHcGradWorkspaceLayout(workspaceSlots_, splitkFactor_, static_cast<uint32_t>(tilingData->aivNum),
                                             c_, static_cast<uint32_t>(tilingData->h), r_, d_, sizeof(D_T),
                                             HAS_GAMMA != 0, USE_DUP_T, taskRows_, retainedRows, fullBatch_, dxColumns,
                                             tilingData->recomputeDirect != 0, fullBatch_, fullBatch_ && !lowpDx_,
                                             fullBatch_);
        gmDwDown_ = Pointer(layout_.dwDownAccum);
        gmDwUp_ = Pointer(layout_.dwUpAccum);
        gmDwGate_ = Pointer(layout_.dwGateAccum);
        g56Rows_ = layout_.g56Rows;
        gateRows_ = layout_.gateRows;
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        InitC1aSlab(tilingData);
#endif
        credit_.Init(workspace_, 0U, taskCount_, workspaceSlots_, aicNum_, static_cast<uint32_t>(tilingData->aivNum));
        if ASCEND_IS_AIC {
            gemm_.Init(aicNum_, tilingData, workspace);
        }
        if ASCEND_IS_AIV {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            gemm_.Init(aicNum_, tilingData, workspace);
#elif HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
            if (fullBatch_) {
                gemm_.Init(aicNum_, tilingData, workspace);
            }
#endif
            vector_.Init(gradMixed, x, wDown, wUp, invRms, downPreAct, up, gradInject, wGate, gamma, inject, dx, dwDown,
                         dwUp, dwGate, dgamma, workspace, layout_, tilingData, pipe);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            if constexpr (HAS_GAMMA) {
                vector_.SetBlockedG56Xn(gemm_.UsesBlockedG56Xn(), gemm_.UsesPreparedG56A());
            }
#endif
        }
    }

    __aicore__ inline void Process()
    {
        // Empty token input still requires zeroing nonempty dw/dgamma outputs, but must bypass credit/ring scheduling.
        // With workspaceSlots=0, TaskPointer modulo operations or a DGAMMA step can hang the kernel.
        if (taskCount_ == 0U) {
            if ASCEND_IS_AIV {
                vector_.Prepare();
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
            AscendC::SyncAll<false>();
#endif
            return;
        }
        if ASCEND_IS_AIV {
            vector_.Prepare();
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            if (!fullBatch_) {
                vector_.PrepareWeights();
            }
#else
            vector_.PrepareWeights();
#endif
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        if (fullBatch_) {
            // The first V1 publication orders all writes from Prepare and V1.
            ProcessA2ChunkCredits();
            return;
        }
        if constexpr (!producerConsumer_) {
            AscendC::SyncAll<false>();
        }
#elif HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if constexpr (HAS_GAMMA) {
            if (fullBatch_) {
                ProcessChunkedFullBatch();
                return;
            }
        }
        if constexpr (!producerConsumer_) {
            AscendC::SyncAll<false>();
        }
#else
        AscendC::SyncAll<false>();
#endif
        ProcessTokenPipeline();
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A2 && HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        if (producerConsumer_) {
            AscendC::SyncAll<false>();
        }
#endif
        // Bounded mode has already reduced every per-task dW partial into the FP32 accumulators.
        if ASCEND_IS_AIV {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            // Reduction and cast assign different ranges to each Vector core.
            credit_.JoinAivWritesForReads();
#endif
            vector_.StoreWeightGradients();
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        AscendC::SyncAll<false>();
#endif
    }

private:
    __aicore__ inline void ProcessTokenPipeline()
    {
        HcGradScheduler<USE_DUP_T, G56_PACKED_T> scheduler(taskCount_, workspaceSlots_, HAS_GAMMA != 0);
        for (uint32_t beatIndex = 0; beatIndex < scheduleBeatCount_ && !scheduler.Done(); ++beatIndex) {
            const HcGradBeat beat = scheduler.Next();
            if (!producerConsumer_) {
                ProcessSynchronizedBeat(beat);
                continue;
            }
            // Both sides share the beat sequence; credits provide cross-side readiness and slot reuse.
            ProcessCubeCredits(beat);
            ProcessVectorCredits(beat);
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A2
            if constexpr (HAS_GAMMA) {
                SynchronizeDgamma(beat);
            }
#endif
        }
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void FinishFullBatchStage()
    {
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        AscendC::SyncAll<false>();
    }

    template <uint32_t STAGE>
    __aicore__ inline void ConvertFullBatchInputs(GM_ADDR a, GM_ADDR b)
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        const uint32_t plan = gemm_.ConversionPlan(chunk_.rows);
        if (!gemm_.NeedsNnConversion(STAGE, plan)) {
            // V3 and DX already published their outputs immediately before these two calls.
            if constexpr (STAGE != HC_GRAD_NN_STAGE_DX && STAGE != HC_GRAD_NN_STAGE_C1B) {
                FinishFullBatchStage();
            }
            return;
        }
        if ASCEND_IS_AIV {
            gemm_.template ConvertNnInputs<STAGE>(plan, a, b, vector_.NnScratch());
            vector_.FinishNnScratch();
        }
#else
        // A5 Blaze consumes the retained ND operands directly. Its callers already
        // publish V1/V3/DX at the enclosing stage boundary, so no conversion barrier is needed.
        (void)a;
        (void)b;
#endif
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
        FinishFullBatchStage();
#endif
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
    static constexpr uint32_t C1A_SLABS = 2U;
    static constexpr uint32_t A2_CHUNK_TASK = 0U;
    static constexpr uint32_t A2_FIRST_ROW = 0U;

    __aicore__ inline void InitC1aSlab(const TextGatedResidualGradTilingDataArch35* data)
    {
        constexpr uint32_t fullPlan = 0U;
        constexpr uint32_t unsplitK = 1U;
        constexpr uint32_t c1aMask = 1U << HC_GRAD_NN_STAGE_C1A;
        if (!HAS_GAMMA || !fullBatch_ || !layout_.duplicateV1Operands || data->a2WeightNzOffset == 0U ||
            c_ != HC_GRAD_TUNED_CHANNELS || d_ != HC_GRAD_TUNED_FEATURES || r_ != HC_GRAD_TUNED_PROJECTION_SIZE ||
            data->aivNum != HC_GRAD_AIV_PER_AIC * data->aicNum || data->aivNum % c_ != 0U ||
            data->vectorUbBytes < HC_GRAD_A2_NN_UB_BYTES || splitkFactor_ != unsplitK ||
            (data->nnMixMask[fullPlan] & c1aMask) == 0U) {
            return;
        }
        const auto& nn = data->nnBase[fullPlan][HC_GRAD_NN_STAGE_C1A];
        // Each group must cover whole M tiles with the original full-K accumulation.
        const uint32_t tiles = nn.tileL2cacheTiling.mTileCntL2;
        const uint32_t rowsPerTile = nn.tileL2cacheTiling.mTileBlock * nn.matmulTiling.singleCoreM;
        if (nn.matmulRunInfo.nd2nzA == 0U && nn.matmulRunInfo.nd2nzB == 1U && nn.matmulRunInfo.transA == 0U &&
            nn.matmulRunInfo.transB == 0U && nn.matmulTiling.M == fullBatchRows_ && nn.matmulTiling.N == r_ &&
            nn.matmulTiling.singleCoreK == d_ && tiles != 0U && tiles % C1A_SLABS == 0U &&
            static_cast<uint64_t>(tiles) * rowsPerTile == fullBatchRows_) {
            c1aTilesPerSlab_ = tiles / C1A_SLABS;
        }
    }

    __aicore__ inline bool StreamsA2C1aRows() const
    {
        return c1aTilesPerSlab_ != 0U && chunk_.rows == fullBatchRows_;
    }

    __aicore__ inline void RunA2C1aGroup(uint32_t firstRow, uint32_t rows)
    {
        const uint64_t inputOffset = static_cast<uint64_t>(firstRow) * d_ * sizeof(D_T);
        const uint64_t outputOffset = static_cast<uint64_t>(firstRow) * r_ * sizeof(float);
        GM_ADDR dup = TaskPointer(layout_.dup, A2_CHUNK_TASK, d_) + inputOffset;
        GM_ADDR partial = Pointer(layout_.dactPartial);
        gemm_.RunC1aRowGroup(dup, partial + outputOffset, rows, c1aTilesPerSlab_);
    }

    static constexpr uint32_t A2_FIRST_M_TILE = 0U;
    static constexpr uint32_t A2_EARLY_C1B_M_TILES = 1U;

    __aicore__ inline bool HasA2EarlyC1b() const
    {
        return StreamsA2C1aRows() && gemm_.C1bRowTileCount() > A2_EARLY_C1B_M_TILES;
    }

    __aicore__ inline void RunA2C1bTiles(uint32_t firstTile, uint32_t endTile)
    {
        gemm_.RunC1bTiles(TaskPointer(layout_.dup, A2_CHUNK_TASK, d_), Pointer(layout_.dwUpPartial), firstTile, endTile,
                          chunk_.rowBegin != 0U);
    }

    __aicore__ inline void PrepareA2EarlyC1b()
    {
        ConvertA2ChunkInputs<HC_GRAD_NN_STAGE_C1B, true>(TaskPointer(layout_.dup, A2_CHUNK_TASK, d_),
                                                         TaskPointer(layout_.act, A2_CHUNK_TASK, r_));
        credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_C1B_INPUT_READY);
    }

    __aicore__ inline void ComputeA2ChunkC1a()
    {
        if (StreamsA2C1aRows()) {
            const uint32_t groupRows = chunk_.rows / C1A_SLABS;
            if ASCEND_IS_AIC {
                RunA2C1aGroup(A2_FIRST_ROW, groupRows);
                credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_C1A_SUFFIX_READY);
                RunA2C1aGroup(groupRows, groupRows);
            } else {
                vector_.ProcessFullV1Slab(groupRows, groupRows);
                credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_C1A_SUFFIX_READY);
            }
        } else if ASCEND_IS_AIC {
            RunC1a(A2_CHUNK_TASK, chunk_.rows);
        }
    }
#endif

    __aicore__ inline void SetChunkState(uint32_t rowBegin, uint32_t chunk)
    {
        const uint32_t remaining = n_ - rowBegin;
        chunk_.rowBegin = rowBegin;
        chunk_.rows = remaining < fullBatchRows_ ? remaining : fullBatchRows_;
        chunk_.taskCount = (chunk_.rows + taskRows_ - 1U) / taskRows_;
        chunk_.parity = chunk & 1U;
        if ASCEND_IS_AIV {
            vector_.SetChunk(rowBegin, chunk_.rows, chunk_.taskCount, chunk_.parity);
        }
    }

    __aicore__ inline void ProduceV1()
    {
        if ASCEND_IS_AIV {
            if (!vector_.TryProcessFullV1()) {
                for (uint32_t task = 0; task < chunk_.taskCount; ++task) {
                    vector_.Process(HcGradVectorOp::V1, task);
                }
            }
        }
    }

    __aicore__ inline void ComputeC1a()
    {
        ConvertFullBatchInputs<HC_GRAD_NN_STAGE_C1A>(TaskPointer(layout_.dup, 0U, d_), gmWUp_);
        if ASCEND_IS_AIC {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_V1_READY);
#endif
            RunC1a(0U, chunk_.rows);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.PublishChunkOutput<HcGradCreditSync::CHUNK_C1A_READY>();
#endif
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        FinishFullBatchStage();
#endif
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void ProcessC1aV3Gm()
    {
        if (layout_.duplicateV1Operands || c1aTilesPerSlab_ == 0U || chunk_.rows != fullBatchRows_) {
            ComputeC1a();
        }
        if ASCEND_IS_AIV {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            // At chunk i>0 every publishing AIC has also finished G56(i-1).
            // V3 may now reuse the common packed storage.
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_C1A_READY);
#endif
            if (!vector_.TryProcessV3FullBatch()) {
                for (uint32_t task = 0; task < chunk_.taskCount; ++task) {
                    vector_.Process(HcGradVectorOp::REDUCE_DACT, task);
                    vector_.Process(HcGradVectorOp::V3, task);
                }
            }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.PublishChunkInput(HcGradCreditSync::CHUNK_V3_READY);
#endif
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        FinishFullBatchStage();
#endif
    }

    __aicore__ inline void ProcessC1aV3Ub()
    {
        const bool needsPreviousG56 = chunk_.rowBegin != 0U;
        if ASCEND_IS_AIC {
            if (needsPreviousG56) {
                credit_.PublishChunkOutput<HcGradCreditSync::CHUNK_G56_READ_DONE>();
            }
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_V1_READY);
            gemm_.RunC1aToV3Ub(TaskPointer(layout_.dup, 0U, d_), gmWUp_);
        }
        if ASCEND_IS_AIV {
            const uint32_t cube = AscendC::GetBlockIdx() / 2U;
            if (cube < HC_GRAD_C1A_UB_CORES) {
                // Drain V1 and opening WConcat before granting write access
                // to the local UB. The next chunk issues a fresh credit.
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::CrossCoreSetFlag<4, PIPE_MTE3>(HC_GRAD_C1A_UB_FREE);
            }
            if (needsPreviousG56) {
                // The previous G56 owns the shared prepared operand until its final read.
                credit_.WaitChunkStage(HcGradCreditSync::CHUNK_G56_READ_DONE);
            }
            if (cube < HC_GRAD_C1A_UB_CORES) {
                const uint32_t tileBegin = cube * HC_GRAD_C1A_UB_TILE_M;
                const uint32_t pairRows = AscendC::Std::min(HC_GRAD_C1A_UB_TILE_M, chunk_.rows - tileBegin);
                const uint32_t rows = pairRows / 2U;
                const uint32_t begin = tileBegin + AscendC::GetSubBlockIdx() * rows;
                AscendC::CrossCoreWaitFlag<4, PIPE_V>(HC_GRAD_C1A_UB_READY);
                vector_.ProcessV3FromC1aUb(begin, rows);
            }
            credit_.PublishChunkInput(HcGradCreditSync::CHUNK_V3_READY);
        }
    }
#endif

    __aicore__ inline void ProcessAfterV1()
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if constexpr (HAS_GAMMA) {
            if (gemm_.CanRunC1aToV3Ub(chunk_.rows)) {
                ProcessC1aV3Ub();
            } else {
                ProcessC1aV3Gm();
            }
        } else {
#endif
            if (layout_.duplicateV1Operands || c1aTilesPerSlab_ == 0U || chunk_.rows != fullBatchRows_) {
                ComputeC1a();
            }
            if ASCEND_IS_AIV {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                // At chunk i>0 every publishing AIC has also finished G56(i-1).
                // V3 may now reuse the common packed storage.
                credit_.WaitChunkStage(HcGradCreditSync::CHUNK_C1A_READY);
#endif
                if (!vector_.TryProcessV3FullBatch()) {
                    for (uint32_t task = 0; task < chunk_.taskCount; ++task) {
                        vector_.Process(HcGradVectorOp::REDUCE_DACT, task);
                        vector_.Process(HcGradVectorOp::V3, task);
                    }
                }
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                credit_.PublishChunkInput(HcGradCreditSync::CHUNK_V3_READY);
#endif
            }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
            FinishFullBatchStage();
#endif
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        }
#endif
        ConvertFullBatchInputs<HC_GRAD_NN_STAGE_DX>(DxMatrix(0U), hasGate_ != 0U ? Pointer(layout_.wConcat) : gmWDown_);
        if ASCEND_IS_AIC {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_V3_READY);
#endif
            RunDx(0U, chunk_.rows);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.PublishChunkOutput<HcGradCreditSync::CHUNK_DX_READY>();
#endif
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        // Publish the intermediate DX before V6 reads it.
        FinishFullBatchStage();
#endif
        ConvertFullBatchInputs<HC_GRAD_NN_STAGE_C1B>(TaskPointer(layout_.dup, 0U, d_),
                                                     TaskPointer(layout_.act, 0U, r_));
        if ASCEND_IS_AIC {
            RunC1b();
        }
        if ASCEND_IS_AIV {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_DX_READY);
#endif
            vector_.ResumeDgammaAfterNn(chunk_.rowBegin != 0U);
            vector_.ProcessFullV6();
            if constexpr (HAS_GAMMA) {
                vector_.Process(HcGradVectorOp::DGAMMA, HC_GRAD_NO_TASK);
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                if (chunk_.rows == n_ - chunk_.rowBegin) {
                    credit_.JoinAivWritesForReads();
                }
#endif
            }
        }
#if HC_GRAD_NPU_ARCH != HC_GRAD_ARCH_A5
        FinishFullBatchStage();
#endif
    }

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2
    static constexpr uint32_t A2_DX_V1_PREFETCH_ROWS = 1024U;

    template <uint32_t STAGE, bool READS_PEER_OPERANDS = false>
    __aicore__ inline void ConvertA2ChunkInputs(GM_ADDR a, GM_ADDR b)
    {
        const uint32_t plan = gemm_.ConversionPlan(chunk_.rows);
        if (!gemm_.NeedsNnConversion(STAGE, plan)) {
            return;
        }
        if constexpr (READS_PEER_OPERANDS) {
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
            credit_.SameKindBarrier();
        }
        gemm_.template ConvertNnInputs<STAGE>(plan, a, b, vector_.NnScratch());
        vector_.FinishNnScratch();
    }

    template <uint32_t STAGE, uint16_t READY, bool READS_PEER_OPERANDS = false>
    __aicore__ inline void PrepareA2ChunkGemm(GM_ADDR a, GM_ADDR b, bool convertInputs = true)
    {
        if ASCEND_IS_AIV {
            if (convertInputs) {
                ConvertA2ChunkInputs<STAGE, READS_PEER_OPERANDS>(a, b);
            }
            credit_.PublishA2ChunkEdge(READY);
        } else {
            credit_.WaitA2ChunkEdge(READY);
        }
    }

    __aicore__ inline void PrepareA2C1b()
    {
        PrepareA2ChunkGemm<HC_GRAD_NN_STAGE_C1B, HcGradCreditSync::A2_C1B_INPUT_READY>(
            TaskPointer(layout_.dup, 0U, d_), TaskPointer(layout_.act, 0U, r_));
    }

    __aicore__ inline void PrepareA2G56()
    {
        PrepareA2ChunkGemm<HC_GRAD_NN_STAGE_G56, HcGradCreditSync::A2_G56_INPUT_READY>(G56Matrix(),
                                                                                       TaskPointer(layout_.xn, 0U, d_));
    }

    __aicore__ inline void PrepareA2WeightInputs()
    {
        // V1 and V3 published these operands before DX was issued. Both NZ regions are free after C1a.
        ConvertA2ChunkInputs<HC_GRAD_NN_STAGE_C1B>(TaskPointer(layout_.dup, 0U, d_), TaskPointer(layout_.act, 0U, r_));
        ConvertA2ChunkInputs<HC_GRAD_NN_STAGE_G56>(G56Matrix(), TaskPointer(layout_.xn, 0U, d_));
        credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_C1B_INPUT_READY);
    }

    __aicore__ inline uint32_t PrefetchA2NextV1(uint32_t nextRow, uint32_t nextChunk)
    {
        if (!layout_.recomputeDirect || hasGate_ == 0U || !HAS_GAMMA || nextRow >= n_ || !vector_.CanUseFullV1Slabs()) {
            return 0U;
        }
        const uint32_t rows = AscendC::Std::min(fullBatchRows_, n_ - nextRow);
        const uint32_t tasks = (rows + taskRows_ - 1U) / taskRows_;
        const uint32_t prefetched = AscendC::Std::min(rows, A2_DX_V1_PREFETCH_ROWS);
        vector_.SetChunk(nextRow, rows, tasks, nextChunk & 1U);
        vector_.ProcessFullV1Slab(0U, prefetched);
        vector_.SetChunk(chunk_.rowBegin, chunk_.rows, chunk_.taskCount, chunk_.parity);
        return prefetched;
    }

    __aicore__ inline void FinishA2V1Inputs(uint32_t prefetchedRows, bool deferAct)
    {
        if (StreamsA2C1aRows()) {
            if ASCEND_IS_AIV {
                const uint32_t groupRows = chunk_.rows / C1A_SLABS;
                if (prefetchedRows < groupRows) {
                    vector_.ProcessFullV1Slab(prefetchedRows, groupRows - prefetchedRows);
                }
            }
            return;
        }
        if (!deferAct && prefetchedRows == 0U) {
            ProduceV1();
            return;
        }
        if (!deferAct) {
            vector_.ProcessFullV1Act();
        }
        vector_.ProcessFullV1Slab(prefetchedRows, chunk_.rows - prefetchedRows);
    }

    __aicore__ inline void ProcessA2ChunkCredits()
    {
        const bool preconvertWeights = gemm_.HasSeparateWeightNz();
        bool deferAct = false;
        if ASCEND_IS_AIV {
            deferAct = preconvertWeights && vector_.CanUseFullV1Slabs();
        }
        SetChunkState(0U, 0U);
        constexpr uint32_t noPrefetchedRows = 0U;
        FinishA2V1Inputs(noPrefetchedRows, deferAct);
        for (uint32_t row = 0U, index = 0U; row < n_; row += fullBatchRows_, ++index) {
            const uint32_t next = row + fullBatchRows_;
            const bool hasNext = next < n_;
            PrepareA2ChunkGemm<HC_GRAD_NN_STAGE_C1A, HcGradCreditSync::A2_C1A_INPUT_READY, true>(
                TaskPointer(layout_.dup, 0U, d_), gmWUp_, !preconvertWeights || row == 0U);
            ComputeA2ChunkC1a();
            const bool earlyC1b = HasA2EarlyC1b();
            if ASCEND_IS_AIC {
                credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_C1A_OUTPUT_READY);
                if (earlyC1b) {
                    credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_C1B_INPUT_READY);
                    // This independent dWUp tile covers the V3 and DX-conversion latency.
                    RunA2C1bTiles(A2_FIRST_M_TILE, A2_EARLY_C1B_M_TILES);
                }
            } else {
                // C1a consumes dup. Act is first read by C1b after the DX-input publication.
                if (deferAct) {
                    vector_.ProcessFullV1Act();
                }
                if (preconvertWeights && index != 0U) {
                    // The previous C1b has finished reading the dedicated Act NZ buffer.
                    credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_C1B_READ_DONE);
                }
                if (earlyC1b) {
                    PrepareA2EarlyC1b();
                }
                credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_C1A_OUTPUT_READY);
                if (!vector_.TryProcessV3FullBatch()) {
                    for (uint32_t task = 0U; task < chunk_.taskCount; ++task) {
                        vector_.Process(HcGradVectorOp::REDUCE_DACT, task);
                        vector_.Process(HcGradVectorOp::V3, task);
                    }
                }
            }
            PrepareA2ChunkGemm<HC_GRAD_NN_STAGE_DX, HcGradCreditSync::A2_DX_INPUT_READY, true>(
                DxMatrix(0U), hasGate_ != 0U ? Pointer(layout_.wConcat) : gmWDown_);
            if ASCEND_IS_AIC {
                RunDx(0U, chunk_.rows);
                credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_DX_OUTPUT_READY);
                if (earlyC1b) {
                    RunA2C1bTiles(A2_EARLY_C1B_M_TILES, gemm_.C1bRowTileCount());
                } else {
                    PrepareA2C1b();
                    RunC1b();
                }
                if (!preconvertWeights || hasNext) {
                    credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_C1B_READ_DONE);
                }
                if (!preconvertWeights) {
                    PrepareA2G56();
                } else if (earlyC1b) {
                    credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_G56_INPUT_READY);
                }
                RunG56(index != 0U);
                if (!preconvertWeights || !hasNext) {
                    credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_G56_READ_DONE);
                }
                if (hasNext) {
                    SetChunkState(next, index + 1U);
                }
            } else {
                if (earlyC1b) {
                    ConvertA2ChunkInputs<HC_GRAD_NN_STAGE_G56>(G56Matrix(), TaskPointer(layout_.xn, 0U, d_));
                    credit_.PublishA2ChunkEdge(HcGradCreditSync::A2_G56_INPUT_READY);
                } else if (preconvertWeights) {
                    PrepareA2WeightInputs();
                }
                const uint32_t prefetchedRows = PrefetchA2NextV1(next, index + 1U);
                credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_DX_OUTPUT_READY);
                if (!preconvertWeights) {
                    PrepareA2C1b();
                }
                vector_.ResumeDgammaAfterNn(row != 0U);
                vector_.ProcessFullV6();
                if constexpr (HAS_GAMMA) {
                    vector_.Process(HcGradVectorOp::DGAMMA, HC_GRAD_NO_TASK);
                    if (!hasNext) {
                        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                        credit_.SameKindBarrier();
                        vector_.Process(HcGradVectorOp::REDUCE_DGAMMA, HC_GRAD_NO_TASK);
                    }
                }
                if (!preconvertWeights) {
                    // The fallback shares C1b and G56 scratch.
                    credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_C1B_READ_DONE);
                    PrepareA2G56();
                }
                if (hasNext) {
                    SetChunkState(next, index + 1U);
                    FinishA2V1Inputs(prefetchedRows, deferAct);
                }
                if (!preconvertWeights || !hasNext) {
                    // Separate NZ buffers can be reused after the next C1a completion.
                    // The final G56 release still orders StoreWeightGradients against all Cube writers.
                    credit_.WaitA2ChunkEdge(HcGradCreditSync::A2_G56_READ_DONE);
                }
            }
        }
        if ASCEND_IS_AIV {
            vector_.StoreWeightGradients();
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
        AscendC::SyncAll<false>();
    }
#endif

#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    __aicore__ inline void ProcessChunkedFullBatch()
    {
        for (uint32_t rowBegin = 0U, chunk = 0U; rowBegin < n_; rowBegin += fullBatchRows_, ++chunk) {
            SetChunkState(rowBegin, chunk);
            // AIV can fill this chunk's operand bank while AIC finishes the previous G56.
            ProduceV1();
            if ASCEND_IS_AIV {
                credit_.PublishChunkInput(HcGradCreditSync::CHUNK_V1_READY);
                if (chunk == 0U) {
                    // V3 publishes the DX weights after C1a has consumed WUp.
                    AscendC::PipeBarrier<PIPE_ALL>();
                    vector_.PrepareWeights();
                }
            }
            ProcessAfterV1();
            const bool lastChunk = chunk_.rows == n_ - rowBegin;
            if ASCEND_IS_AIV {
                if constexpr (HAS_GAMMA) {
                    if (lastChunk) {
                        vector_.Process(HcGradVectorOp::REDUCE_DGAMMA, HC_GRAD_NO_TASK);
                    }
                }
            }
            if ASCEND_IS_AIC {
                RunG56(chunk != 0U);
                if (lastChunk) {
                    credit_.PublishChunkOutput<HcGradCreditSync::CHUNK_DW_READY>();
                }
            }
        }
        if ASCEND_IS_AIV {
            // All Cube dW writers have drained before Vector reads their accumulators.
            credit_.WaitChunkStage(HcGradCreditSync::CHUNK_DW_READY);
            vector_.StoreWeightGradients();
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
    }
#endif
#endif

    __aicore__ inline void RunC1b()
    {
        if (fullBatch_) {
            gemm_.RunC1b(TaskPointer(layout_.dup, 0U, d_), TaskPointer(layout_.act, 0U, r_),
                         Pointer(layout_.dwUpPartial), chunk_.rows, r_, d_, chunk_.rowBegin != 0U);
        } else {
            gemm_.RunC1bDirect(TaskPointer(layout_.dup, 0U, d_), TaskPointer(layout_.act, 0U, r_), gmDwUp_, n_, r_, d_);
        }
    }

    __aicore__ inline void RunG56(bool accumulate = false)
    {
        const uint32_t stride = fullBatch_ && hasGate_ == 0 ? r_ : g56Rows_;
        const uint32_t rows = fullBatch_ ? chunk_.rows : n_;
        gemm_.RunG56MergedPacked(G56Matrix(), TaskPointer(layout_.xn, 0U, d_), Pointer(layout_.dwPartial), rows, r_, d_,
                                 stride, gateRows_, hasGate_ != 0, accumulate);
    }

    __aicore__ inline GM_ADDR G56Matrix() const
    {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
        if (gemm_.UsesPreparedG56A()) {
            return Pointer(layout_.dact);
        }
#endif
        // Without a gate, G56 and DX both consume the same dense ddown matrix.
        return fullBatch_ && hasGate_ == 0 ? DxMatrix(0U) : TaskPointer(layout_.packed, 0U, g56Rows_);
    }

    __aicore__ inline void ProcessSynchronizedBeat(const HcGradBeat& beat)
    {
        if ASCEND_IS_AIC {
            RunCube(beat.cubeOp, beat.cubeTask);
            RunCube(beat.cubeOp2, beat.cubeTask2);
        }
        if ASCEND_IS_AIV {
            for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
                if (beat.vectorOp[i] != HcGradVectorOp::NONE) {
                    vector_.Process(beat.vectorOp[i], beat.vectorTask[i]);
                }
            }
            AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
        }
        AscendC::SyncAll<false>();
    }

    __aicore__ inline void ProcessCubeCredits(const HcGradBeat& beat)
    {
        if ASCEND_IS_AIC {
            const bool hasCube = (beat.cubeOp != HcGradCubeOp::NONE) || (beat.cubeOp2 != HcGradCubeOp::NONE);
            credit_.WaitCubeDeps(beat.cubeOp, beat.cubeTask);
            RunCube(beat.cubeOp, beat.cubeTask);
            credit_.WaitCubeDeps(beat.cubeOp2, beat.cubeTask2);
            RunCube(beat.cubeOp2, beat.cubeTask2);
            if (hasCube) {
                credit_.SameKindBarrier();
                credit_.MarkCube(beat.cubeOp, beat.cubeTask);
                credit_.MarkCube(beat.cubeOp2, beat.cubeTask2);
            }
        }
    }

    __aicore__ inline void ProcessVectorCredits(const HcGradBeat& beat)
    {
        if ASCEND_IS_AIV {
            bool hasVector = false;
            for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
                if (beat.vectorOp[i] != HcGradVectorOp::NONE) {
                    credit_.WaitVectorDeps(beat.vectorOp[i], beat.vectorTask[i]);
                    vector_.Process(beat.vectorOp[i], beat.vectorTask[i]);
                    hasVector = true;
                }
            }
            if (hasVector) {
                AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
                credit_.SameKindBarrier();
                for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
                    credit_.MarkVector(beat.vectorOp[i], beat.vectorTask[i]);
                }
            }
        }
    }

    __aicore__ inline void SynchronizeDgamma(const HcGradBeat& beat)
    {
        // DGAMMA publishes per-core partials before the next beat reads them.
        for (uint32_t i = 0; i < HC_GRAD_VECTOR_OPS_PER_BEAT; ++i) {
            if (beat.vectorOp[i] == HcGradVectorOp::DGAMMA) {
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
                if ASCEND_IS_AIV {
                    credit_.JoinAivWritesForReads();
                }
#else
                AscendC::SyncAll<false>();
#endif
                break;
            }
        }
    }

    __aicore__ inline GM_ADDR TaskPointer(uint64_t offset, uint32_t task, uint32_t rowStride) const
    {
        return workspace_ + layout_.TaskBufferOffset(offset, task, workspaceSlots_,
                                                     static_cast<uint64_t>(taskRows_) * rowStride * sizeof(D_T),
                                                     chunk_.parity);
    }

    __aicore__ inline GM_ADDR TaskPointerF32(uint64_t offset, uint32_t task, uint32_t rowStride) const
    {
        return workspace_ + layout_.TaskBufferOffset(offset, task, workspaceSlots_,
                                                     static_cast<uint64_t>(taskRows_) * rowStride * sizeof(float),
                                                     chunk_.parity);
    }

    __aicore__ inline GM_ADDR SlotPointerF32(uint64_t offset, uint32_t task, uint64_t elements) const
    {
        return workspace_ + layout_.TaskBufferOffset(offset, task, workspaceSlots_, elements * sizeof(float));
    }

    __aicore__ inline GM_ADDR Pointer(uint64_t offset) const
    {
        return workspace_ + offset;
    }

    __aicore__ inline uint32_t Rows(uint32_t task) const
    {
        const uint32_t rowBegin = task * taskRows_;
        return n_ - rowBegin < taskRows_ ? n_ - rowBegin : taskRows_;
    }

    __aicore__ inline void RunC1a(uint32_t task, uint32_t rows)
    {
        GM_ADDR dactPartial =
            SlotPointerF32(layout_.dactPartial, task, static_cast<uint64_t>(splitkFactor_) * taskRows_ * r_);
        gemm_.RunC1aSplitK(TaskPointer(layout_.dup, task, d_), gmWUp_, dactPartial, rows, r_, d_, splitkFactor_);
    }

    __aicore__ inline void RunC1(uint32_t task, uint32_t rows)
    {
        RunC1a(task, rows);
        // Bounded mode finishes dW for this task before its ring slot can be reused.
        const uint64_t c1bA = USE_DUP_T ? layout_.dupT : layout_.dup;
        if (singleTask_) {
            gemm_.RunC1bDirect(TaskPointer(c1bA, task, d_), TaskPointer(layout_.act, task, r_), gmDwUp_, rows, r_, d_);
        } else {
            gemm_.RunC1b(TaskPointer(c1bA, task, d_), TaskPointer(layout_.act, task, r_),
                         SlotPointerF32(layout_.dwPartial, task, layout_.dwSlotElems), rows, r_, d_);
        }
    }

    __aicore__ inline GM_ADDR DxMatrix(uint32_t task) const
    {
        return layout_.denseDx != 0U ? Pointer(layout_.denseDx) : TaskPointer(layout_.packed, task, g56Rows_);
    }

    __aicore__ inline void RunDx(uint32_t task, uint32_t rows)
    {
        const bool dense = layout_.denseDx != 0U;
        const uint32_t stride = dense ? r_ + (hasGate_ != 0U ? c_ : 0U) : g56Rows_;
        GM_ADDR dxOut = TaskPointerF32(layout_.dx2, task, d_);
        gemm_.RunDx(DxMatrix(task), gmWDown_, gmWGate_, Pointer(layout_.wConcat), dxOut, rows, d_, r_, c_, stride,
                    hasGate_ != 0U);
    }

    __aicore__ inline void RunCube(HcGradCubeOp op, uint32_t task)
    {
        const uint32_t rows = Rows(task);
        switch (op) {
            case HcGradCubeOp::C1: {
                RunC1(task, rows);
                break;
            }
            case HcGradCubeOp::G56: {
                GM_ADDR packedT = TaskPointer(layout_.packedT, task, g56Rows_);
                GM_ADDR packed = TaskPointer(layout_.packed, task, g56Rows_);
                GM_ADDR xn = TaskPointer(layout_.xn, task, d_);
                GM_ADDR dwPartial = SlotPointerF32(layout_.dwPartial, task, layout_.dwSlotElems);
                if constexpr (G56_PACKED_T) {
                    if (singleTask_) {
                        gemm_.RunG56Direct(packedT, xn, gmDwDown_, dwPartial + layout_.dwGateRowElems * sizeof(float),
                                           rows, r_, d_, gateRows_, hasGate_ != 0);
                    } else {
                        gemm_.RunG56Merged(packedT, xn, dwPartial, rows, r_, d_, gateRows_, hasGate_ != 0);
                    }
                } else {
                    if (singleTask_) {
                        gemm_.RunG56DirectPacked(packed, xn, gmDwDown_,
                                                 dwPartial + layout_.dwGateRowElems * sizeof(float), rows, r_, d_,
                                                 g56Rows_, gateRows_, hasGate_ != 0);
                    } else {
                        gemm_.RunG56MergedPacked(packed, xn, dwPartial, rows, r_, d_, g56Rows_, gateRows_,
                                                 hasGate_ != 0);
                    }
                }
                break;
            }
            case HcGradCubeOp::DX:
                RunDx(task, rows);
                break;
            default:
                break;
        }
    }

    uint32_t n_{0};
    uint32_t c_{0};
    uint32_t r_{0};
    uint32_t d_{0};
    uint32_t taskRows_{HC_GRAD_TASK_ROWS};
    uint32_t taskCount_{0};
    uint32_t workspaceSlots_{0};
    uint32_t splitkFactor_{0};
    uint32_t hasGate_{0};
    uint32_t g56Rows_{0};
    uint32_t gateRows_{0};
    uint32_t scheduleBeatCount_{0};
    uint32_t aicNum_{0};
    GM_ADDR workspace_{nullptr};
    GM_ADDR gmWUp_{nullptr};
    GM_ADDR gmWDown_{nullptr};
    GM_ADDR gmWGate_{nullptr};
    GM_ADDR gmDwUp_{nullptr};
    GM_ADDR gmDwDown_{nullptr};
    GM_ADDR gmDwGate_{nullptr};
    uint32_t c1aTilesPerSlab_{0};
    uint32_t fullBatchRows_{0};
    ChunkState chunk_{};
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A2 || HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    static constexpr bool lowpDx_ = false;
    // Both host tilings emit nonzero schedule modes for chunk and token execution.
    static constexpr bool producerConsumer_ = !USE_DUP_T && !G56_PACKED_T;
#else
    bool lowpDx_{false};
    bool producerConsumer_{false};
#endif
#if HC_GRAD_NPU_ARCH == HC_GRAD_ARCH_A5
    static constexpr bool singleTask_ = false;
#else
    bool singleTask_{false};
#endif
    bool fullBatch_{false};
    HcGradWorkspaceLayout layout_;
    HcGradCreditSync credit_;
    GemmT gemm_;
    HcGradVector<D_T, HAS_GAMMA> vector_;
};

} // namespace TextGatedResidualGrad

#endif // TEXT_GATED_RESIDUAL_GRAD_KERNEL_COMMON_H
