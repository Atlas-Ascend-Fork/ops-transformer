/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SHARED_EXPERT_HPP
#define MEGA_MOE_SHARED_EXPERT_HPP

#include "a2_a3_copy_helpers.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm_coord.hpp"
#include "utils/get_tensor_addr.hpp"
#include "utils/block_epilogue_pertoken_swiglu.hpp"
#include "utils/block_epilogue_pertoken_row.hpp"

namespace Catlass::Gemm::Kernel {

// Dense shared FFNs for A3 combinations with independent width or weight type.
// Every flag is consumed before routed kernels reuse the same hardware resources.
// Workspace sizes/order deliberately match host sharedExpertWorkspace.
template <class BlockMmad, class BlockScheduler, class BlockEpilogue1, class BlockEpilogue2, class SharedBf16Mmad_,
          bool SharedGmm2Bf16_>
class SharedExpert {
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using ElementC = typename BlockMmad::ElementC;
    using ElementD1 = typename BlockEpilogue1::ElementD;
    using ElementD2 = typename BlockEpilogue2::ElementD;
    using ElementScale = uint64_t;
    using LayoutA = typename BlockMmad::LayoutA;
    using LayoutB = typename BlockMmad::LayoutB;
    using LayoutC = typename BlockMmad::LayoutC;
    using LayoutScale = layout::VectorLayout;
    static constexpr bool kRoutingIsQuant =
        std::is_same_v<ElementB, int8_t> || std::is_same_v<ElementB, AscendC::int4b_t>;
    static_assert(!SharedGmm2Bf16_ || (std::is_same_v<ElementB, int8_t> && std::is_same_v<ElementD1, bfloat16_t> &&
                                       std::is_same_v<ElementD2, bfloat16_t>),
                  "Mixed shared GMM2 requires INT8 routing and BF16 activation/output");
    static constexpr uint16_t SYNCFLAG_SHARED_GMM1_C2V = 11;
    static constexpr uint16_t SYNCFLAG_SHARED_INPUT_V2C = 12;
    static constexpr uint16_t SYNCFLAG_SHARED_GMM1_V2C = 12;
    static constexpr uint16_t SYNCFLAG_SHARED_GMM2_C2V = 13;
    static constexpr uint16_t SYNCFLAG_SHARED_RESULT_V2C = 12;

public:
    struct Params {
        GemmCoord problemShape;
        GM_ADDR ptrA;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrSharedB1;
        GM_ADDR ptrSharedB2;
        GM_ADDR ptrSharedScale1;
        GM_ADDR ptrSharedScale2;
        GM_ADDR ptrSharedBias1;
        GM_ADDR ptrSharedBias2;
        LayoutA layoutA2;
        LayoutB layoutB1;
        LayoutB layoutB2;
        typename SharedBf16Mmad_::LayoutB layoutSharedB2;
        LayoutScale layoutScale1;
        LayoutScale layoutScale2;
        uint64_t sharedWorkspaceOffset;
        uint32_t sharedExpertNum;
        uint32_t epilogueCoreNum;
        float swigluLimit;
        uint32_t activationCode;
        float activationParams1;
        float activationParams2;
    };

    CATLASS_DEVICE
    SharedExpert(Params const& params)
    {
        coreIdx = AscendC::GetBlockIdx();
        coreNum = AscendC::GetBlockNum();
        if ASCEND_IS_AIV {
            coreIdx = get_block_idx() + get_subblockid() * get_block_num();
            coreNum = get_block_num() * get_subblockdim();
        }
        const uint64_t m = params.problemShape.m();
        const uint64_t n = params.problemShape.n();
        const uint64_t k = params.problemShape.k();
        const uint64_t experts = params.sharedExpertNum;
        constexpr uint64_t msd = std::is_same_v<ElementB, AscendC::int4b_t> ? 2U : 1U;
        auto& w = workspaceInfo;
        w.ptrSharedGmmBuffer = params.ptrWorkspace + params.sharedWorkspaceOffset;
        w.ptrSharedActivation =
            w.ptrSharedGmmBuffer + AlignBuffer(experts * m * (n > k ? n : k) * msd * sizeof(ElementC));
        GM_ADDR tail = w.ptrSharedActivation + AlignBuffer(experts * m * (n / 2U) * sizeof(ElementD1));
        if constexpr (kRoutingIsQuant) {
            w.ptrSharedInput = tail;
            tail += AlignBuffer(m * k * sizeof(int8_t));
            w.ptrSharedInputScale = tail;
            tail += AlignBuffer(m * sizeof(float));
            w.ptrSharedActivationScale = tail;
            tail += AlignBuffer(experts * m * sizeof(float));
            w.ptrSharedResult = tail;
            gmSharedInputScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w.ptrSharedInputScale));
            gmSharedActivationScale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(w.ptrSharedActivationScale));
        } else {
            w.ptrSharedResult = w.ptrSharedGmmBuffer;
        }
        gmSharedGmmBuffer.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(w.ptrSharedGmmBuffer));
        gmSharedActivation.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD1*>(w.ptrSharedActivation));
        gmSharedResult.SetGlobalBuffer(reinterpret_cast<__gm__ ElementD2*>(w.ptrSharedResult));
    }

    CATLASS_DEVICE GM_ADDR Result() const
    {
        return workspaceInfo.ptrSharedResult;
    }

    CATLASS_DEVICE void Process(Params const& params)
    {
        if ASCEND_IS_AIC {
            if constexpr (kRoutingIsQuant) {
                AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAG_SHARED_INPUT_V2C);
            }
            SharedGMM1(params);
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAG_SHARED_GMM1_V2C);
            SharedGMM2(params);
            // All AIVs have completed the GMM2 epilogue before routed work starts.
            AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAG_SHARED_RESULT_V2C);
        }
        if ASCEND_IS_AIV {
            PrepareSharedInput(params);
            SharedGMM1Epilogue(params);
            SharedGMM2Epilogue(params);
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAG_SHARED_RESULT_V2C);
        }
    }

private:
    CATLASS_DEVICE static uint64_t AlignBuffer(uint64_t bytes)
    {
        return (bytes + 511UL) / 512UL * 512UL;
    }

    CATLASS_DEVICE
    void PrepareSharedInput(Params const& params)
    {
        if constexpr (!kRoutingIsQuant) {
            return;
        } else {
            const uint32_t m = params.problemShape.m();
            const uint32_t k = params.problemShape.k();
            const uint32_t rowsPerCore = m / coreNum;
            const uint32_t remainder = m % coreNum;
            const uint32_t rowCount = coreIdx < remainder ? rowsPerCore + 1U : rowsPerCore;
            const uint32_t rowStart = coreIdx * rowsPerCore + (coreIdx < remainder ? coreIdx : remainder);

            Detail::PrepareSharedQuantInput<std::is_same_v<ElementB, AscendC::int4b_t>, ElementD2>(
                resource, reinterpret_cast<GM_ADDR>(params.ptrA), workspaceInfo.ptrSharedInput,
                workspaceInfo.ptrSharedInputScale, rowStart, rowCount, k);
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAG_SHARED_INPUT_V2C);
        }
    }

    CATLASS_DEVICE
    void RunSharedGmm1Tiles(Params const& params, BlockMmad& blockMmad, BlockScheduler& blockScheduler,
                            AscendC::GlobalTensor<ElementB>& sharedWeight,
                            AscendC::GlobalTensor<ElementScale>& sharedWeightScale,
                            AscendC::GlobalTensor<ElementA>& sharedInput, uint32_t sharedIdx, uint32_t computeM,
                            uint32_t n, uint32_t startLoopIdx, uint32_t coreLoops, const LayoutA& sharedLayoutA,
                            const LayoutC& sharedLayoutC)
    {
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            const int64_t gmOffsetA = sharedLayoutA.GetOffset(offsetA);
            const int64_t gmOffsetB = params.layoutB1.GetOffset(offsetB);
            const int64_t gmOffsetC = static_cast<int64_t>(sharedIdx) * computeM * n + sharedLayoutC.GetOffset(offsetC);
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                const int64_t gmOffsetS = blockCoord.n() * L1TileShape::N;
                blockMmad(sharedWeightScale[gmOffsetS], params.layoutScale1, sharedInput[gmOffsetA], sharedLayoutA,
                          sharedWeight[gmOffsetB], params.layoutB1, gmSharedGmmBuffer[gmOffsetC], sharedLayoutC,
                          actualBlockShape, Callback{}, Callback{});
            } else if constexpr (std::is_same_v<ElementB, int8_t>) {
                const int64_t gmOffsetS = blockCoord.n() * L1TileShape::N;
                blockMmad(sharedInput[gmOffsetA], sharedLayoutA, sharedWeight[gmOffsetB], params.layoutB1,
                          gmSharedGmmBuffer[gmOffsetC], sharedLayoutC, sharedWeightScale[gmOffsetS],
                          params.layoutScale1, actualBlockShape);
            } else {
                blockMmad(sharedInput[gmOffsetA], sharedLayoutA, sharedWeight[gmOffsetB], params.layoutB1,
                          gmSharedGmmBuffer[gmOffsetC], sharedLayoutC, gmS, params.layoutScale1, actualBlockShape);
            }
        }
    }

    CATLASS_DEVICE
    void SharedGMM1(Params const& params)
    {
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);
        AscendC::GlobalTensor<ElementA> sharedInput;
        AscendC::GlobalTensor<ElementB> sharedWeight;
        AscendC::GlobalTensor<ElementScale> sharedWeightScale;
        if constexpr (kRoutingIsQuant) {
            sharedInput.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(workspaceInfo.ptrSharedInput));
        } else {
            sharedInput.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(params.ptrA));
        }

        uint32_t startCoreIdx = 0U;
        const uint32_t m = params.problemShape.m();
        constexpr uint32_t msdFactor = std::is_same_v<ElementB, AscendC::int4b_t> ? 2U : 1U;
        const uint32_t computeM = m * msdFactor;
        const uint32_t n = params.problemShape.n();
        const uint32_t k = params.problemShape.k();
        for (uint32_t sharedIdx = 0U; sharedIdx < params.sharedExpertNum; ++sharedIdx) {
            sharedWeight.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(
                GetTensorAddr<ElementB>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedB1))));
            if constexpr (kRoutingIsQuant) {
                sharedWeightScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale*>(
                    GetTensorAddr<int64_t>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedScale1))));
            }
            if (m <= L1TileShape::M) {
                sharedWeight.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            GemmCoord sharedShape{computeM, n, k};
            // Shared X is dense, including interleaved W4 high/low rows;
            // routed A2 GMM1 instead reserves a max(K, N/2) token slot.
            LayoutA sharedLayoutA{computeM, k};
            LayoutC sharedLayoutC{computeM, n};
            blockScheduler.Update(sharedShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            const uint32_t coreLoops = blockScheduler.GetCoreLoops();
            const uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            RunSharedGmm1Tiles(params, blockMmad, blockScheduler, sharedWeight, sharedWeightScale, sharedInput,
                               sharedIdx, computeM, n, startLoopIdx, coreLoops, sharedLayoutA, sharedLayoutC);
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        blockMmad.Finalize(0, SYNCFLAG_SHARED_GMM1_C2V);
    }

    CATLASS_DEVICE
    void SharedGMM2Bf16(Params const& params)
    {
        using SharedMmad = SharedBf16Mmad_;
        using SharedTile = typename SharedMmad::L1TileShape;
        BlockScheduler scheduler;
        SharedMmad mm(resource);
        const uint32_t m = params.problemShape.m();
        const uint32_t n = params.problemShape.k();
        const uint32_t k = params.problemShape.n() / 2U;
        const GemmCoord shape{m, n, k};
        const LayoutA layoutA{m, k};
        const LayoutC layoutC{m, n};
        AscendC::GlobalTensor<bfloat16_t> activation, weight, output;
        activation.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceInfo.ptrSharedActivation));
        output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(workspaceInfo.ptrSharedResult));
        uint32_t startCore = 0U;
        for (uint32_t expert = 0U; expert < params.sharedExpertNum; ++expert) {
            weight.SetGlobalBuffer(GetTensorAddr<bfloat16_t>(expert, reinterpret_cast<GM_ADDR>(params.ptrSharedB2)));
            if (m <= SharedTile::M) {
                weight.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }
            scheduler.Update(shape, MakeCoord(SharedTile::M, SharedTile::N));
            const uint32_t loops = scheduler.GetCoreLoops();
            const uint32_t first = ((coreIdx < startCore) ? coreIdx + coreNum : coreIdx) - startCore;
            for (uint32_t loop = first; loop < loops; loop += coreNum) {
                const GemmCoord block = scheduler.GetBlockCoord(loop);
                const GemmCoord actual = scheduler.GetActualBlockShape(block);
                const MatrixCoord a{block.m() * SharedTile::M, block.k() * SharedTile::K};
                const MatrixCoord b{block.k() * SharedTile::K, block.n() * SharedTile::N};
                const MatrixCoord c{block.m() * SharedTile::M, block.n() * SharedTile::N};
                const uint64_t aOffset = static_cast<uint64_t>(expert) * m * k + layoutA.GetOffset(a);
                const uint64_t cOffset = static_cast<uint64_t>(expert) * m * n + layoutC.GetOffset(c);
                mm(activation[aOffset], layoutA, weight[params.layoutSharedB2.GetOffset(b)], params.layoutSharedB2,
                   output[cOffset], layoutC, gmS2, params.layoutScale2, actual);
            }
            startCore = (startCore + loops) % coreNum;
        }
        if constexpr (SharedMmad::DispatchPolicy::ASYNC) {
            mm.SynchronizeBlock();
        }
        mm.Finalize(0, SYNCFLAG_SHARED_GMM2_C2V);
    }

    CATLASS_DEVICE
    void RunSharedGmm2Tiles(Params const& params, BlockMmad& blockMmad, BlockScheduler& blockScheduler,
                            AscendC::GlobalTensor<ElementB>& sharedWeight,
                            AscendC::GlobalTensor<ElementScale>& sharedWeightScale,
                            AscendC::GlobalTensor<ElementA>& sharedActivation,
                            AscendC::GlobalTensor<ElementC>& sharedOutput, uint32_t sharedIdx, uint32_t computeM,
                            uint32_t n, uint32_t k, uint32_t startLoopIdx, uint32_t coreLoops,
                            const LayoutA& sharedLayoutA, const LayoutC& sharedLayoutC)
    {
        for (uint32_t loopIdx = startLoopIdx; loopIdx < coreLoops; loopIdx += coreNum) {
            GemmCoord blockCoord = blockScheduler.GetBlockCoord(loopIdx);
            GemmCoord actualBlockShape = blockScheduler.GetActualBlockShape(blockCoord);
            MatrixCoord offsetA{blockCoord.m() * L1TileShape::M, blockCoord.k() * L1TileShape::K};
            MatrixCoord offsetB{blockCoord.k() * L1TileShape::K, blockCoord.n() * L1TileShape::N};
            MatrixCoord offsetC{blockCoord.m() * L1TileShape::M, blockCoord.n() * L1TileShape::N};
            const int64_t gmOffsetA = static_cast<int64_t>(sharedIdx) * computeM * k + sharedLayoutA.GetOffset(offsetA);
            const int64_t gmOffsetB = params.layoutB2.GetOffset(offsetB);
            const int64_t gmOffsetC = static_cast<int64_t>(sharedIdx) * computeM * n + sharedLayoutC.GetOffset(offsetC);
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                const int64_t gmOffsetS = blockCoord.n() * L1TileShape::N;
                blockMmad(sharedWeightScale[gmOffsetS], params.layoutScale2, sharedActivation[gmOffsetA], sharedLayoutA,
                          sharedWeight[gmOffsetB], params.layoutB2, sharedOutput[gmOffsetC], sharedLayoutC,
                          actualBlockShape, Callback{}, Callback{});
            } else if constexpr (std::is_same_v<ElementB, int8_t>) {
                const int64_t gmOffsetS = blockCoord.n() * L1TileShape::N;
                blockMmad(sharedActivation[gmOffsetA], sharedLayoutA, sharedWeight[gmOffsetB], params.layoutB2,
                          sharedOutput[gmOffsetC], sharedLayoutC, sharedWeightScale[gmOffsetS], params.layoutScale2,
                          actualBlockShape);
            } else {
                blockMmad(sharedActivation[gmOffsetA], sharedLayoutA, sharedWeight[gmOffsetB], params.layoutB2,
                          sharedOutput[gmOffsetC], sharedLayoutC, gmS2, params.layoutScale2, actualBlockShape);
            }
        }
    }

    CATLASS_DEVICE
    void SharedGMM2(Params const& params)
    {
        if constexpr (SharedGmm2Bf16_) {
            SharedGMM2Bf16(params);
            return;
        }
        BlockScheduler blockScheduler;
        BlockMmad blockMmad(resource);
        AscendC::GlobalTensor<ElementA> sharedActivation;
        AscendC::GlobalTensor<ElementB> sharedWeight;
        AscendC::GlobalTensor<ElementScale> sharedWeightScale;
        AscendC::GlobalTensor<ElementC> sharedOutput;
        sharedActivation.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA*>(workspaceInfo.ptrSharedActivation));
        if constexpr (kRoutingIsQuant) {
            sharedOutput.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(workspaceInfo.ptrSharedGmmBuffer));
        } else {
            sharedOutput.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC*>(workspaceInfo.ptrSharedResult));
        }

        uint32_t startCoreIdx = 0U;
        const uint32_t m = params.problemShape.m();
        constexpr uint32_t msdFactor = std::is_same_v<ElementB, AscendC::int4b_t> ? 2U : 1U;
        const uint32_t computeM = m * msdFactor;
        const uint32_t n = params.problemShape.k();
        const uint32_t k = params.problemShape.n() / 2U;
        for (uint32_t sharedIdx = 0U; sharedIdx < params.sharedExpertNum; ++sharedIdx) {
            sharedWeight.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB*>(
                GetTensorAddr<ElementB>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedB2))));
            if constexpr (kRoutingIsQuant) {
                sharedWeightScale.SetGlobalBuffer(reinterpret_cast<__gm__ ElementScale*>(
                    GetTensorAddr<int64_t>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedScale2))));
            }
            if (m <= L1TileShape::M) {
                sharedWeight.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            }

            GemmCoord sharedShape{computeM, n, k};
            LayoutA sharedLayoutA = params.layoutA2.GetTileLayout(sharedShape.GetCoordMK());
            LayoutC sharedLayoutC{computeM, n};
            blockScheduler.Update(sharedShape, MakeCoord(L1TileShape::M, L1TileShape::N));
            const uint32_t coreLoops = blockScheduler.GetCoreLoops();
            const uint32_t startLoopIdx = ((coreIdx < startCoreIdx) ? (coreIdx + coreNum) : coreIdx) - startCoreIdx;

            RunSharedGmm2Tiles(params, blockMmad, blockScheduler, sharedWeight, sharedWeightScale, sharedActivation,
                               sharedOutput, sharedIdx, computeM, n, k, startLoopIdx, coreLoops, sharedLayoutA,
                               sharedLayoutC);
            startCoreIdx = (startCoreIdx + coreLoops) % coreNum;
        }

        if constexpr (BlockMmad::DispatchPolicy::ASYNC) {
            blockMmad.SynchronizeBlock();
        }
        blockMmad.Finalize(0, SYNCFLAG_SHARED_GMM2_C2V);
    }

    CATLASS_DEVICE
    void SharedGMM1Epilogue(Params const& params)
    {
        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAG_SHARED_GMM1_C2V);
        AscendC::SyncAll<true>();
        const uint32_t m = params.problemShape.m();
        const uint32_t n = params.problemShape.n();
        const uint32_t k2 = n / 2U;
        constexpr uint32_t msdFactor = std::is_same_v<ElementB, AscendC::int4b_t> ? 2U : 1U;
        BlockEpilogue1 sharedEpilogue(resource, n);
        for (uint32_t sharedIdx = 0U; sharedIdx < params.sharedExpertNum; ++sharedIdx) {
            if constexpr (std::is_same_v<ElementC, int32_t>) {
                sharedEpilogue.SetWeightScales(GetTensorAddr<uint64_t>(sharedIdx, params.ptrSharedScale1));
            }
            MatrixCoord shapeC{m, n};
            const int64_t gmOffsetC = static_cast<int64_t>(sharedIdx) * m * n * msdFactor;
            const int64_t gmOffsetD = static_cast<int64_t>(sharedIdx) * m * k2;
            if constexpr (SharedGmm2Bf16_) {
                sharedEpilogue(gmSharedGmmBuffer[gmOffsetC], shapeC, gmSharedActivation[gmOffsetD], resource,
                               params.epilogueCoreNum, params.swigluLimit, params.activationCode,
                               params.activationParams1, params.activationParams2, n, Callback{}, &gmSharedInputScale);
            } else if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                AscendC::GlobalTensor<float> sharedBias;
                sharedBias.SetGlobalBuffer(
                    GetTensorAddr<float>(sharedIdx, reinterpret_cast<GM_ADDR>(params.ptrSharedBias1)));
                sharedEpilogue.SharedExpert(
                    gmSharedGmmBuffer[gmOffsetC], shapeC, gmSharedInputScale, sharedBias, gmSharedActivation[gmOffsetD],
                    gmSharedActivationScale[static_cast<uint64_t>(sharedIdx) * m], resource, params.epilogueCoreNum,
                    params.swigluLimit, params.activationCode, params.activationParams1, params.activationParams2, n);
            } else if constexpr (std::is_same_v<ElementB, int8_t>) {
                sharedEpilogue(gmSharedGmmBuffer[gmOffsetC], shapeC, gmSharedInputScale, gmSharedActivation[gmOffsetD],
                               gmSharedActivationScale[static_cast<uint64_t>(sharedIdx) * m], resource,
                               params.epilogueCoreNum, params.swigluLimit, params.activationCode,
                               params.activationParams1, params.activationParams2, n);
            } else {
                sharedEpilogue(gmSharedGmmBuffer[gmOffsetC], shapeC, gmSharedActivation[gmOffsetD], resource,
                               params.epilogueCoreNum, params.swigluLimit, params.activationCode,
                               params.activationParams1, params.activationParams2, n);
            }
        }
        sharedEpilogue.Finalize();
        // Each shared expert owns a disjoint GM range. Drain the local
        // epilogue pipeline first, then synchronize once before AIC consumes
        // any of the published activation rows.
        AscendC::SyncAll<true>();
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(SYNCFLAG_SHARED_GMM1_V2C);
    }

    CATLASS_DEVICE void SharedGMM2Epilogue(Params const& params)
    {
        AscendC::CrossCoreWaitFlag<0x2>(SYNCFLAG_SHARED_GMM2_C2V);
        AscendC::SyncAll<true>();
        if constexpr (kRoutingIsQuant && !SharedGmm2Bf16_) {
            const uint32_t m = params.problemShape.m();
            const uint32_t n = params.problemShape.k();
            typename BlockEpilogue2::Params p{};
            if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                p.n0 = L1TileShape::N;
            }
            p.n2 = n;
            BlockEpilogue2 epilogue(resource, p);
            for (uint32_t expert = 0; expert < params.sharedExpertNum; ++expert) {
                if constexpr (std::is_same_v<ElementC, int32_t>) {
                    epilogue.SetWeightScales(GetTensorAddr<uint64_t>(expert, params.ptrSharedScale2));
                }
                const uint64_t offset = static_cast<uint64_t>(expert) * m * n;
                const uint64_t scaleOffset = static_cast<uint64_t>(expert) * m;
                if constexpr (std::is_same_v<ElementB, AscendC::int4b_t>) {
                    AscendC::GlobalTensor<float> bias;
                    bias.SetGlobalBuffer(GetTensorAddr<float>(expert, params.ptrSharedBias2));
                    epilogue.SharedExpert(gmSharedGmmBuffer[2U * offset], gmSharedActivationScale[scaleOffset], bias,
                                          gmSharedResult[offset], m, n, params.epilogueCoreNum);
                } else {
                    epilogue.SharedExpert(gmSharedGmmBuffer[offset], MatrixCoord{m, n},
                                          gmSharedActivationScale[scaleOffset], gmSharedResult[offset],
                                          params.epilogueCoreNum);
                }
            }
            AscendC::PipeBarrier<PIPE_MTE3>();
            AscendC::SyncAll<true>();
        }
    }

    struct Workspace {
        GM_ADDR ptrSharedGmmBuffer{nullptr};
        GM_ADDR ptrSharedActivation{nullptr};
        GM_ADDR ptrSharedInput{nullptr};
        GM_ADDR ptrSharedInputScale{nullptr};
        GM_ADDR ptrSharedActivationScale{nullptr};
        GM_ADDR ptrSharedResult{nullptr};
    } workspaceInfo;
    uint32_t coreIdx;
    uint32_t coreNum;
    Arch::Resource<ArchTag> resource;
    AscendC::GlobalTensor<ElementC> gmSharedGmmBuffer;
    AscendC::GlobalTensor<ElementD1> gmSharedActivation;
    AscendC::GlobalTensor<ElementD2> gmSharedResult;
    AscendC::GlobalTensor<float> gmSharedInputScale;
    AscendC::GlobalTensor<float> gmSharedActivationScale;
    AscendC::GlobalTensor<ElementScale> gmS, gmS2;
};
} // namespace Catlass::Gemm::Kernel
#endif
