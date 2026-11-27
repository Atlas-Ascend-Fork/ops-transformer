/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// A5: actual NN Blaze kernels and scheduler, with fusion stride and synchronization adapters.
#ifndef TEXT_GATED_RESIDUAL_GRAD_GEMM_ARCH35_H
#define TEXT_GATED_RESIDUAL_GRAD_GEMM_ARCH35_H
#include "text_gated_residual_grad_tiling_data_arch35.h"
#include "text_gated_residual_grad_pipeline_arch35.h"
#include "../text_gated_residual_grad_constants.h"
#include "nn_a5/mat_mul_pingpong_basic.h"
#include "nn_a5/mat_mul_al1_full_load.h"
namespace TextGatedResidualGrad {
template <typename D_T, int HAS_GAMMA>
class HcGradGemm {
public:
    __aicore__ inline void Init(uint32_t aicNum, const TextGatedResidualGradTilingDataArch35* data, GM_ADDR workspace)
    {
        data_ = data;
        scratch_ = workspace + data->nnScratchOffset;
    }

    __aicore__ inline bool UsesBlockedG56Xn() const
    {
        constexpr uint32_t XN_TILE_N = HC_GRAD_G56_BLOCKED_XN_N;
        constexpr uint32_t PACKED_G56_ROWS = HcGradAlign16(HC_GRAD_TUNED_PROJECTION_SIZE + HC_GRAD_TUNED_CHANNELS);
        constexpr uint32_t FULL_K_PARTS = 1U;
        constexpr uint32_t FULL_AND_TAIL_PLAN_COUNT = 2U;
        constexpr uint64_t TRANSPOSED_BASIC_KEY = HC_GRAD_A5_BASIC_GEMM_KEY | HC_GRAD_A5_TRANSPOSE_A_KEY_BIT;
        constexpr uint32_t G56_L1_K = HC_GRAD_A5_L1_K_TILES * HC_GRAD_A5_WEIGHT_TILE_K;
        constexpr uint32_t L1_BYTES = 524288U;
        constexpr uint32_t L0_OPERAND_BYTES = 65536U;
        constexpr uint32_t L0C_BYTES = 262144U;
        static_assert(HC_GRAD_A5_DOUBLE_BUFFER * (PACKED_G56_ROWS + XN_TILE_N) * G56_L1_K * sizeof(D_T) <= L1_BYTES);
        static_assert(HC_GRAD_A5_DOUBLE_BUFFER * PACKED_G56_ROWS * HC_GRAD_A5_WEIGHT_TILE_K * sizeof(D_T) <=
                      L0_OPERAND_BYTES);
        static_assert(HC_GRAD_A5_DOUBLE_BUFFER * XN_TILE_N * HC_GRAD_A5_WEIGHT_TILE_K * sizeof(D_T) <=
                      L0_OPERAND_BYTES);
        static_assert(PACKED_G56_ROWS * XN_TILE_N * sizeof(float) <= L0C_BYTES);
        if (data_->n < HC_GRAD_CHUNK_ROWS || data_->fullBatchRows != HC_GRAD_CHUNK_ROWS ||
            data_->aicNum != HC_GRAD_A5_TUNED_AIC_COUNT || data_->aivNum != HC_GRAD_A5_TUNED_AIV_COUNT ||
            data_->c != HC_GRAD_TUNED_CHANNELS || data_->r != HC_GRAD_TUNED_PROJECTION_SIZE ||
            data_->h != HC_GRAD_TUNED_HIDDEN_SIZE || data_->d != HC_GRAD_TUNED_FEATURES || data_->hasGate == 0 ||
            data_->hasGamma == 0 || data_->splitkFactor != FULL_K_PARTS || data_->lowpDx != 0) {
            return false;
        }
        const uint32_t tailRows = static_cast<uint32_t>((data_->n - 1U) % HC_GRAD_CHUNK_ROWS + 1U);
        for (uint32_t plan = 0U; plan < FULL_AND_TAIL_PLAN_COUNT; ++plan) {
            const auto& t = data_->nnA5[plan][HC_GRAD_NN_STAGE_G56];
            if (data_->nnA5Key[plan][HC_GRAD_NN_STAGE_G56] != TRANSPOSED_BASIC_KEY || t.m != PACKED_G56_ROWS ||
                t.n != HC_GRAD_TUNED_FEATURES || t.k != (plan == 0U ? HC_GRAD_CHUNK_ROWS : tailRows) ||
                t.usedCoreNum != HC_GRAD_A5_TUNED_AIC_COUNT || t.baseM != HC_GRAD_A5_WEIGHT_TILE_M ||
                t.baseN != HC_GRAD_A5_WEIGHT_TILE_N || t.baseK != HC_GRAD_A5_WEIGHT_TILE_K || t.kL1 != G56_L1_K ||
                t.mL1 != HC_GRAD_A5_WEIGHT_TILE_M || t.nL1 != HC_GRAD_A5_WEIGHT_TILE_N ||
                t.l1BufferNum != HC_GRAD_A5_DOUBLE_BUFFER || t.l0cDB != HC_GRAD_A5_SINGLE_BUFFER ||
                t.mTailCnt != HC_GRAD_A5_G56_M_TAIL_SPLITS || t.nTailCnt != HC_GRAD_A5_UNSPLIT_TAIL ||
                t.mBaseTailSplitCnt != HC_GRAD_A5_UNSPLIT_TAIL || t.nBaseTailSplitCnt != HC_GRAD_A5_UNSPLIT_TAIL ||
                t.l2CacheDisable != 0U) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline bool UsesPreparedG56A() const
    {
        // With two rows per Vector block, V3 uses the C1a partial directly.
        // The chunk and its tail must both end on a complete ZN group.
        return data_->n % 16U == 0U && data_->scheduleMode == HC_GRAD_FULL_BATCH_SCHEDULE &&
               data_->pipeStages == HC_GRAD_PIPE_MAX_STAGES && data_->vectorRowBlock == 2U && UsesBlockedG56Xn();
    }

    __aicore__ inline bool CanRunC1aToV3Ub(uint32_t rows) const
    {
        if (rows != HC_GRAD_CHUNK_ROWS || data_->vectorUbBytes < HC_GRAD_C1A_UB_LIMIT || !UsesPreparedG56A()) {
            return false;
        }
        const auto& t = data_->nnA5[0][0];
        return data_->nnA5Key[0][0] == 0x2U && t.m == HC_GRAD_CHUNK_ROWS && t.n == HC_GRAD_C1A_UB_TILE_N &&
               t.k == 10240U && t.usedCoreNum == HC_GRAD_C1A_UB_CORES && t.mL1 == HC_GRAD_C1A_UB_TILE_M &&
               t.nL1 == HC_GRAD_C1A_UB_TILE_N && t.kL1 == 128U && t.baseM == t.mL1 && t.baseN == t.nL1 &&
               t.baseK == 32U && t.mTailCnt == 1U && t.nTailCnt == 1U && t.mBaseTailSplitCnt == 1U &&
               t.nBaseTailSplitCnt == 1U && t.l1BufferNum == 2U && t.l0cDB == 1U && t.l2CacheDisable == 1U &&
               t.sliceM == HC_GRAD_C1A_UB_TILE_M && t.srcNdStride == 1U;
    }

    __aicore__ inline void RunC1aToV3Ub(GM_ADDR dup, GM_ADDR wUp)
    {
        const uint32_t core = AscendC::GetBlockIdx();
        if (core < HC_GRAD_C1A_UB_CORES) {
            const uint32_t begin = core * HC_GRAD_C1A_UB_TILE_M;
            const uint32_t rows = AscendC::Std::min(HC_GRAD_C1A_UB_TILE_M, HC_GRAD_CHUNK_ROWS - begin);
            HcGradNnA5::MatMulC1aToV3Ub<D_T>(dup, wUp, data_->nnA5[0][0], begin, rows, HC_GRAD_C1A_UB_OFFSET);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void RunC1aSplitK(GM_ADDR dup, GM_ADDR wUp, GM_ADDR partial, uint32_t rows, uint32_t r,
                                        uint32_t d, uint32_t splitkFactor)
    {
        // The host selects Basic tiling with a complete K per output tile.
        RunNn<false>(0, rows, rows, r, d, dup, d, wUp, r, partial, r);
    }
    // C1b: dwUpPartial = dup^T @ act, reading dup as ColumnMajor [D,rows], as on A5.
    __aicore__ inline void RunC1b(GM_ADDR dup, GM_ADDR act, GM_ADDR dwUpPartial, uint32_t rows, uint32_t r, uint32_t d,
                                  bool accumulate = false)
    {
        if (accumulate) {
            AscendC::SetAtomicAdd<float>();
        }
        RunNn<true>(1, rows, d, r, rows, dup, d, act, r, dwUpPartial, r);
        if (accumulate) {
            AscendC::SetAtomicNone();
        }
    }

    // For multiple tasks, use one M=R+gateRows GEMM to avoid rereading xn in a separate small gate GEMM.
    __aicore__ inline void RunG56Merged(GM_ADDR packedT, GM_ADDR xn, GM_ADDR dwOut, uint32_t rows, uint32_t r,
                                        uint32_t d, uint32_t gateRows, bool hasGate)
    {
        const uint32_t m = hasGate ? (r + gateRows) : r;
        RunNn<false>(HC_GRAD_NN_STAGE_G56, rows, m, d, rows, packedT, rows, xn, d, dwOut, d);
    }

    __aicore__ inline void RunG56MergedPacked(GM_ADDR packed, GM_ADDR xn, GM_ADDR dwOut, uint32_t rows, uint32_t r,
                                              uint32_t d, uint32_t packedStride, uint32_t gateRows, bool hasGate,
                                              bool accumulate = false)
    {
        const uint32_t m = hasGate ? (r + gateRows) : r;
        if (accumulate) {
            AscendC::SetAtomicAdd<float>();
        }
        RunNn<true>(HC_GRAD_NN_STAGE_G56, rows, m, d, rows, packed, packedStride, xn, d, dwOut, d);
        if (accumulate) {
            AscendC::SetAtomicNone();
        }
    }

    __aicore__ inline void RunG56DirectPacked(GM_ADDR packed, GM_ADDR xn, GM_ADDR dwDown, GM_ADDR dwGate, uint32_t rows,
                                              uint32_t r, uint32_t d, uint32_t packedStride, uint32_t gateRows,
                                              bool hasGate)
    {
        RunNn<true>(HC_GRAD_NN_STAGE_G56, rows, r, d, rows, packed, packedStride, xn, d, dwDown, d);
        if (hasGate) {
            RunNn<true>(HC_GRAD_NN_STAGE_GATE, rows, gateRows, d, rows, packed + r * sizeof(D_T), packedStride, xn, d,
                        dwGate, d);
        }
    }

    // Single-task fast paths: accumulate straight into the FP32 outputs.
    __aicore__ inline void RunC1bDirect(GM_ADDR dup, GM_ADDR act, GM_ADDR dwUp, uint32_t rows, uint32_t r, uint32_t d)
    {
        RunNn<true>(1, rows, d, r, rows, dup, d, act, r, dwUp, r);
    }

    __aicore__ inline void RunG56Direct(GM_ADDR packedT, GM_ADDR xn, GM_ADDR dwDown, GM_ADDR dwGate, uint32_t rows,
                                        uint32_t r, uint32_t d, uint32_t gateRows, bool hasGate)
    {
        RunNn<false>(HC_GRAD_NN_STAGE_G56, rows, r, d, rows, packedT, rows, xn, d, dwDown, d);
        if (hasGate) {
            // As on A5, write gateRows aligned to 16 into dwGatePartial, which has room for all padded rows.
            // Vector REDUCE_WD copies only the first c*d elements into the dwGate FP32 accumulator.
            RunNn<false>(HC_GRAD_NN_STAGE_GATE, rows, gateRows, d, rows, packedT + r * rows * sizeof(D_T), rows, xn, d,
                         dwGate, d);
        }
    }

    // dxnorm = ddown @ w_down + dlogits @ w_gate: concatenate weights along K into wConcat for one GEMM.
    __aicore__ inline void RunDx(GM_ADDR packed, GM_ADDR wDown, GM_ADDR wGate, GM_ADDR wConcat, GM_ADDR dxOut,
                                 uint32_t rows, uint32_t d, uint32_t r, uint32_t c, uint32_t packedStride, bool hasGate)
    {
        if (hasGate) {
            RunNn<false>(HC_GRAD_NN_STAGE_DX, rows, rows, d, r + c, packed, packedStride, wConcat, d, dxOut, d);
        } else {
            RunNn<false>(HC_GRAD_NN_STAGE_DX, rows, rows, d, r, packed, packedStride, wDown, d, dxOut, d);
        }
    }

private:
    template <bool TRANS_A>
    __aicore__ inline void RunNn(uint32_t stage, uint32_t rows, uint32_t m, uint32_t n, uint32_t k, GM_ADDR a,
                                 uint32_t lda, GM_ADDR b, uint32_t ldb, GM_ADDR c, uint32_t ldc)
    {
        if ASCEND_IS_AIV {
            return;
        }
        const uint32_t fullRows = data_->fullBatchRows != 0 ? data_->fullBatchRows :
                                                              (data_->n < data_->taskRows ? data_->n : data_->taskRows);
        const uint32_t task = rows == fullRows ? 0 : 1;
        auto t = data_->nnA5[task][stage];
        t.rowStride = lda;
        RunBasic<TRANS_A>(stage, rows, n, k, a, b, c, data_->nnA5Key[task][stage], t);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataSyncBarrier<AscendC::MemDsbT::DDR>();
    }

    template <bool TRANS_A>
    __aicore__ inline void RunBasic(uint32_t stage, uint32_t rows, uint32_t n, uint32_t k, GM_ADDR a, GM_ADDR b,
                                    GM_ADDR c, uint64_t key, HcGradA5NnTiling& t)
    {
        using ALayout = AscendC::Std::conditional_t<TRANS_A, asc::te::dn_ext_layout_ptn, asc::te::nd_ext_layout_ptn>;
        using ND = asc::te::nd_ext_layout_ptn;
        // TileA5Nn selects Basic and transposed Basic. The tuned variants require a gamma chunk.
        if constexpr (HAS_GAMMA) {
            if constexpr (TRANS_A) {
                if (stage == HC_GRAD_NN_STAGE_G56 && UsesBlockedG56Xn()) {
                    // Each tile owns all M rows and its entire K, so B is read once.
                    t.mL1 = t.baseM = 336U;
                    t.nL1 = t.baseN = HC_GRAD_G56_BLOCKED_XN_N;
                    t.mTailCnt = t.nTailCnt = 1U;
                    t.mTailMain = t.m;
                    t.nTailMain = t.n % t.nL1;
                    t.aswWindowLen = 1U;
                    if (UsesPreparedG56A()) {
                        using ZN = asc::te::zn_layout_ptn;
                        HcGradNnA5::MatMulBasicTrafficKernel<D_T, float, ZN, ND, ND, HC_GRAD_CHUNK_ROWS, false>(a, b, c,
                                                                                                                t);
                    } else {
                        HcGradNnA5::MatMulBasicTrafficKernel<D_T, float, ALayout, ND, ND, HC_GRAD_CHUNK_ROWS, false>(
                            a, b, c, t);
                    }
                    return;
                }
            } else if (stage == HC_GRAD_NN_STAGE_DX && key == HC_GRAD_A5_BASIC_GEMM_KEY && n == 10240U && k == 324U &&
                       data_->fullBatchRows == HC_GRAD_CHUNK_ROWS && t.sliceM <= 1U) {
                // Retain the installed Basic K tessellation and output shape on every tile.
                if (rows == HC_GRAD_CHUNK_ROWS && data_->hasGate != 0 && data_->aicNum == 28 && t.usedCoreNum == 28U &&
                    t.m == HC_GRAD_CHUNK_ROWS && t.n == n && t.k == k && t.rowStride >= k && t.mL1 == 256U &&
                    t.nL1 == 256U && t.kL1 == 256U && t.baseM == 256U && t.baseN == 256U && t.baseK == 64U &&
                    t.mTailCnt == 1U && t.nTailCnt == 1U && t.mBaseTailSplitCnt == 1U && t.nBaseTailSplitCnt == 1U &&
                    t.l1BufferNum == 2U && t.l0cDB == 1U && t.l2CacheDisable == 0U) {
                    HcGradNnA5::MatMulDxAPanelL1Kernel<D_T>(a, b, c, t);
                    return;
                }
                // A non-allocating output keeps this large one-use DX intermediate out of L2.
                if (rows >= 3072U) {
                    HcGradNnA5::MatMulBasicTrafficKernel<D_T, float, ALayout, ND, ND, 0U, true>(a, b, c, t);
                    return;
                }
            }
        }
        HcGradNnA5::MatMulBasicKernel<D_T, D_T, float, float, ALayout, ND, ND>(a, b, nullptr, c, scratch_, t);
    }

    const TextGatedResidualGradTilingDataArch35* data_{nullptr};
    GM_ADDR scratch_{nullptr};
};
} // namespace TextGatedResidualGrad
#endif
