/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_allto_all_quant_matmul.hpp
 * \brief
 */

#ifndef CATLASS_MATMUL_ALLTO_ALL_QUANT_MATMUL_HPP
#define CATLASS_MATMUL_ALLTO_ALL_QUANT_MATMUL_HPP

#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_cross_core_sync.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/layout/tla_layout_layout.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/detail/tla_detail_callback.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_matrix_coord.hpp"
#include "matmul_allto_all_util_arch22.h"

namespace Catlass::Gemm::Kernel {
template <class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, uint32_t WORKSPACE_STAGES_>
class QuantMatmulAllToAllKernel {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementA = typename BlockMmad::ElementA;
    using LayoutA = typename BlockMmad::LayoutA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutB = typename BlockMmad::LayoutB;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;
    using ElementAccumulator = typename BlockMmad::ElementAccumulator;

    using BlockEpilogue = BlockEpilogue_;
    using ElementScale = typename BlockEpilogue::ElementScale;
    using LayoutScale = typename BlockEpilogue::LayoutScale;
    using ElementPerTokenScale = typename BlockEpilogue::ElementPerTokenScale;
    using LayoutPerTokenScale = typename BlockEpilogue::LayoutPerTokenScale;
    using ElementBias = typename BlockEpilogue::ElementBias;
    using LayoutBias = typename BlockEpilogue::LayoutBias;
    using ElementD = typename BlockEpilogue::ElementD;
    using LayoutD = typename BlockEpilogue::LayoutD;
    using EpilogueParams = typename BlockEpilogue::Params;

    using BlockScheduler = BlockScheduler_;
    static constexpr uint32_t WORKSPACE_STAGES = WORKSPACE_STAGES_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrScale;
        LayoutScale layoutScale;
        GM_ADDR ptrPerTokenScale;
        LayoutPerTokenScale layoutPerTokenScale;
        GM_ADDR ptrBias;
        LayoutBias layoutBias;
        GM_ADDR ptrPeerMem;
        LayoutD layoutD;
        GM_ADDR ptrWorkspace;
        GM_ADDR ptrOut;
        CommBase commUtil;
        int32_t pipeDepth;
        bool transB;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const& problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrScale_, LayoutScale layoutScale_, GM_ADDR ptrPerTokenScale_,
               LayoutPerTokenScale layoutPerTokenScale_, GM_ADDR ptrBias_, LayoutBias layoutBias_, GM_ADDR ptrPeerMem_,
               LayoutD layoutD_, GM_ADDR ptrWorkspace_, GM_ADDR ptrOut_, CommBase commUtil_, int32_t pipeDepth_,
               bool transB_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrScale(ptrScale_),
              layoutScale(layoutScale_),
              ptrPerTokenScale(ptrPerTokenScale_),
              layoutPerTokenScale(layoutPerTokenScale_),
              ptrBias(ptrBias_),
              layoutBias(layoutBias_),
              ptrPeerMem(ptrPeerMem_),
              layoutD(layoutD_),
              ptrWorkspace(ptrWorkspace_),
              ptrOut(ptrOut_),
              commUtil(commUtil_),
              pipeDepth(pipeDepth_),
              transB(transB_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    QuantMatmulAllToAllKernel()
    {
        Arch::FlagID flagId = 0;
        for (uint32_t stageId = 0; stageId < WORKSPACE_STAGES; ++stageId) {
            flagAicFinishStoreList[stageId] = Arch::CrossCoreFlag(flagId++);
            flagAivFinishComputeList[stageId] = Arch::CrossCoreFlag(flagId++);
            aicWaitFuncList[stageId] = {this, stageId};
            aicSetFuncList[stageId] = {this, stageId};
        }
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params& params);

#if 1
    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params& params)
    {
        AscendC::GlobalTensor<ElementA> gmA;
        gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        AscendC::GlobalTensor<ElementB> gmB;
        gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);

        BlockMmad blockMmad(resource);

        int32_t coreIdx = AscendC::GetBlockIdx();
        int32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

        int32_t mLoops = (params.problemShape.m() + L1TileShape::M - 1) / L1TileShape::M;

        int32_t commCount = (mLoops + params.commUtil.p_value - 1) / params.commUtil.p_value;

        for (int32_t commIdx = 0; commIdx < commCount; commIdx++) {
            RunAicCommStage(commIdx, commCount, mLoops, coreIdx, coreNum, blockMmad, gmA, gmB, gmC, layoutC, params);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(Params& params)
    {
        BlockEpilogue blockEpilogue(resource);

        uint32_t coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        uint32_t coreNum = AscendC::GetBlockNum();

        AscendC::GlobalTensor<ElementD> gmD;
        gmD.SetGlobalBuffer((__gm__ ElementD*)params.ptrPeerMem);
        AscendC::GlobalTensor<ElementC> gmC;
        gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrWorkspace);
        auto layoutC = layout::RowMajor{L1TileShape::M * coreNum * WORKSPACE_STAGES, L1TileShape::N};

        LayoutScale layoutScale = params.layoutScale;
        LayoutScale layoutBias = params.layoutBias;
        LayoutPerTokenScale layoutPerTokenScale =
            params.layoutPerTokenScale.GetTileLayout(params.problemShape.template GetCoordByAxis<0>());
        LayoutD layoutD = params.layoutD;
        EpilogueParams epilogueParams{params.ptrScale,     layoutScale,    params.ptrPerTokenScale,
                                      layoutPerTokenScale, params.ptrBias, layoutBias};
        blockEpilogue.UpdateParams(epilogueParams);

        int32_t mLoops = (params.problemShape.m() + L1TileShape::M - 1) / L1TileShape::M;

        int32_t commCount = (mLoops + params.commUtil.p_value - 1) / params.commUtil.p_value;

        params.commUtil.ResetIpcFlags(2);
        PipeBarrier<PIPE_ALL>();
        for (int32_t commIdx = 0; commIdx < commCount; commIdx++) {
            RunAivCommStage(commIdx, commCount, mLoops, coreIdx, coreNum, blockEpilogue, gmD, gmC, layoutC, layoutD,
                            params);
        }
        params.commUtil.ResetIpcFlags(1);
    }
#endif

private:
    // 单轮 comm 的循环参数
    struct CommLoopInfo {
        int32_t actualLoopNum{0};
        int32_t nLoopPerRank{0};
        uint32_t actualPValue{0};
        uint32_t nLoops{0};
        int64_t gmAPingpongSize{0};
    };

    CATLASS_DEVICE static CommLoopInfo GetCommLoopInfo(int32_t commIdx, int32_t commCount, int32_t mLoops,
                                                       Params& params)
    {
        CommLoopInfo info;
        int32_t tailN = (params.problemShape.n() / params.commUtil.rank_size) % L1TileShape::N;
        uint32_t nLoops = params.problemShape.n() / params.commUtil.rank_size / L1TileShape::N;
        if (tailN) {
            nLoops += 1;
        }
        nLoops *= params.commUtil.rank_size;

        uint32_t actualPValue = params.commUtil.p_value;
        if (commIdx == commCount - 1) {
            actualPValue = mLoops - commIdx * params.commUtil.p_value;
        }
        info.nLoops = nLoops;
        info.actualPValue = actualPValue;
        info.actualLoopNum = actualPValue * nLoops;
        info.nLoopPerRank = nLoops / params.commUtil.rank_size;
        info.gmAPingpongSize = static_cast<int64_t>(L1TileShape::M) * params.commUtil.p_value * params.problemShape.n();
        return info;
    }

    CATLASS_DEVICE static uint32_t GetMActual(int32_t commIdx, int32_t commCount, int32_t mLoops, uint32_t blockM,
                                              Params& params)
    {
        bool isLastComm = (commIdx == commCount - 1);
        bool isTailBlock = (blockM == static_cast<uint32_t>((mLoops - 1) % params.commUtil.p_value));
        bool isTailM = (params.problemShape.m() % L1TileShape::M != 0);
        if (!isLastComm || !isTailBlock || !isTailM) {
            return L1TileShape::M;
        }
        return static_cast<uint32_t>(params.problemShape.m() - commIdx * params.commUtil.p_value * L1TileShape::M -
                                     blockM * L1TileShape::M);
    }

    CATLASS_DEVICE static uint32_t GetNActual(uint32_t blockN, int32_t nLoopPerRank, Params& params)
    {
        uint32_t tailOffset = blockN % static_cast<uint32_t>(nLoopPerRank);
        if (tailOffset != static_cast<uint32_t>(nLoopPerRank - 1)) {
            return L1TileShape::N;
        }
        return params.problemShape.n() / params.commUtil.rank_size - tailOffset * L1TileShape::N;
    }

    CATLASS_DEVICE static int64_t GetGmOffsetD(uint64_t flagIdx, int64_t gmAPingpongSize,
                                               GemmCoord const& blockIdxCoord, int32_t nLoopPerRank, Params& params)
    {
        int64_t rankOffset = gmAPingpongSize / params.commUtil.rank_size;
        int64_t rankIdx = blockIdxCoord.n() / nLoopPerRank;
        return static_cast<int64_t>(flagIdx) * L1TileShape::M * params.commUtil.p_value * params.problemShape.n() +
               rankIdx * rankOffset +
               static_cast<int64_t>(blockIdxCoord.m()) * L1TileShape::M *
                   (params.problemShape.n() / params.commUtil.rank_size) +
               blockIdxCoord.n() % nLoopPerRank * L1TileShape::N;
    }

    CATLASS_DEVICE void RunAicCommStage(int32_t commIdx, int32_t commCount, int32_t mLoops, int32_t coreIdx,
                                        int32_t coreNum, BlockMmad& blockMmad, AscendC::GlobalTensor<ElementA>& gmA,
                                        AscendC::GlobalTensor<ElementB>& gmB, AscendC::GlobalTensor<ElementC>& gmC,
                                        layout::RowMajor layoutC, Params& params)
    {
        uint64_t flagIdx = commIdx % params.pipeDepth;
        CommLoopInfo info = GetCommLoopInfo(commIdx, commCount, mLoops, params);
        if (commIdx >= params.pipeDepth) {
            WaitEvent(flagIdx + WORKSPACE_STAGES * 2); // 避开matmul-dequant使用的flag_idx(0-3)
        }
        GemmCoord calSwizzleShape{info.actualPValue, info.nLoops, 1};
        BlockScheduler matmulBlockScheduler(calSwizzleShape,
                                            MakeCoord(static_cast<uint32_t>(1), static_cast<uint32_t>(1)));

        uint32_t stageId = 0;
        uint32_t stageUsed = 0;
        for (int32_t loopOffset = 0; loopOffset < info.actualLoopNum; loopOffset++) {
            int32_t loopIdx = commIdx * params.commUtil.p_value * info.nLoops + loopOffset;
            if (loopIdx % coreNum != coreIdx) {
                continue;
            }
            RunAicBlock(commIdx, commCount, mLoops, loopOffset, coreIdx, coreNum, info, matmulBlockScheduler, blockMmad,
                        gmA, gmB, gmC, layoutC, stageId, stageUsed, params);
        }
        blockMmad.SynchronizeBlock();

        while (stageUsed > 0) {
            uint32_t aivComputeStageId =
                (stageId >= stageUsed) ? (stageId - stageUsed) : (stageId + WORKSPACE_STAGES - stageUsed);
            Arch::CrossCoreWaitFlag(flagAivFinishComputeList[aivComputeStageId]);
            --stageUsed;
        }
    }

    CATLASS_DEVICE void RunAicBlock(int32_t commIdx, int32_t commCount, int32_t mLoops, int32_t loopOffset,
                                    int32_t coreIdx, int32_t coreNum, CommLoopInfo const& info,
                                    BlockScheduler& matmulBlockScheduler, BlockMmad& blockMmad,
                                    AscendC::GlobalTensor<ElementA>& gmA, AscendC::GlobalTensor<ElementB>& gmB,
                                    AscendC::GlobalTensor<ElementC>& gmC, layout::RowMajor layoutC, uint32_t& stageId,
                                    uint32_t& stageUsed, Params& params)
    {
        Callback callbackBeforeFixpipe{};
        if (stageUsed == WORKSPACE_STAGES) {
            callbackBeforeFixpipe = MakeCallback(&aicWaitFuncList[stageId]);
        } else {
            ++stageUsed;
        }
        Callback callbackAfterFixpipe = MakeCallback(&aicSetFuncList[stageId]);

        GemmCoord blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopOffset);
        uint32_t mActual = GetMActual(commIdx, commCount, mLoops, blockIdxCoord.m(), params);
        uint32_t nActual = GetNActual(blockIdxCoord.n(), info.nLoopPerRank, params);
        GemmCoord actualBlockShape{mActual, nActual, params.problemShape.k()};

        MatrixCoord offsetA{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.k() * L1TileShape::K};
        MatrixCoord offsetC{(stageId * coreNum + coreIdx) * L1TileShape::M, 0};
        int64_t gmOffsetA = params.layoutA.GetOffset(offsetA) + static_cast<int64_t>(commIdx) * L1TileShape::M *
                                                                    params.commUtil.p_value * params.problemShape.k();
        int64_t gmOffsetB = (static_cast<int64_t>(blockIdxCoord.n() / info.nLoopPerRank) *
                                 (params.problemShape.n() / params.commUtil.rank_size) +
                             static_cast<int64_t>(blockIdxCoord.n() % info.nLoopPerRank) * L1TileShape::N) *
                            (params.transB ? params.problemShape.k() : 1);
        int64_t gmOffsetC = layoutC.GetOffset(offsetC);

        blockMmad(gmA[gmOffsetA], params.layoutA, gmB[gmOffsetB], params.layoutB, gmC[gmOffsetC], layoutC,
                  actualBlockShape, callbackBeforeFixpipe, callbackAfterFixpipe);
        stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
    }

    CATLASS_DEVICE void RunAivCommStage(int32_t commIdx, int32_t commCount, int32_t mLoops, uint32_t coreIdx,
                                        uint32_t coreNum, BlockEpilogue& blockEpilogue,
                                        AscendC::GlobalTensor<ElementD>& gmD, AscendC::GlobalTensor<ElementC>& gmC,
                                        layout::RowMajor layoutC, LayoutD layoutD, Params& params)
    {
        uint64_t flagIdx = commIdx % params.pipeDepth;
        CommLoopInfo info = GetCommLoopInfo(commIdx, commCount, mLoops, params);
        GemmCoord calSwizzleShape{info.actualPValue, info.nLoops, 1};
        BlockScheduler matmulBlockScheduler(calSwizzleShape,
                                            MakeCoord(static_cast<uint32_t>(1), static_cast<uint32_t>(1)));
        uint64_t flag_idx = commIdx % MAX_BLOCK_COUNT;
        uint32_t stageId = 0;
        for (int32_t loopOffset = 0; loopOffset < info.actualLoopNum; loopOffset++) {
            int32_t loopIdx = commIdx * params.commUtil.p_value * info.nLoops + loopOffset;
            if (loopIdx % coreNum != coreIdx) {
                continue;
            }
            RunAivBlock(commIdx, commCount, mLoops, loopOffset, coreIdx, coreNum, flagIdx, info, matmulBlockScheduler,
                        blockEpilogue, gmD, gmC, layoutC, layoutD, stageId, params);
        }
        RunAivCommSync(commIdx, commCount, flag_idx, params);
    }

    CATLASS_DEVICE void RunAivBlock(int32_t commIdx, int32_t commCount, int32_t mLoops, int32_t loopOffset,
                                    uint32_t coreIdx, uint32_t coreNum, uint64_t flagIdx, CommLoopInfo const& info,
                                    BlockScheduler& matmulBlockScheduler, BlockEpilogue& blockEpilogue,
                                    AscendC::GlobalTensor<ElementD>& gmD, AscendC::GlobalTensor<ElementC>& gmC,
                                    layout::RowMajor layoutC, LayoutD layoutD, uint32_t& stageId, Params& params)
    {
        GemmCoord blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopOffset);
        uint32_t mActual = GetMActual(commIdx, commCount, mLoops, blockIdxCoord.m(), params);
        uint32_t nActual = GetNActual(blockIdxCoord.n(), info.nLoopPerRank, params);
        GemmCoord actualBlockShape{mActual, nActual, params.problemShape.k()};
        int64_t mOffset = static_cast<int64_t>(commIdx) * L1TileShape::M * params.commUtil.p_value +
                          static_cast<int64_t>(blockIdxCoord.m()) * L1TileShape::M;
        int64_t nOffset =
            (blockIdxCoord.n() / info.nLoopPerRank) * (params.problemShape.n() / params.commUtil.rank_size) +
            (blockIdxCoord.n() % info.nLoopPerRank) * L1TileShape::N;
        MatrixCoord blockOffset{mOffset, nOffset};
        MatrixCoord offsetC{static_cast<uint32_t>((stageId * coreNum + coreIdx) * L1TileShape::M),
                            static_cast<uint32_t>(0)};
        int64_t gmOffsetC = layoutC.GetOffset(offsetC);

        auto gmBlockC = gmC[gmOffsetC];
        auto layoutBlockC = layoutC.GetTileLayout(actualBlockShape.GetCoordMN());

        int64_t gmOffsetD = GetGmOffsetD(flagIdx, info.gmAPingpongSize, blockIdxCoord, info.nLoopPerRank, params);
        Arch::CrossCoreWaitFlag(flagAicFinishStoreList[stageId]);
        blockEpilogue(blockOffset, actualBlockShape, gmBlockC, layoutBlockC, gmD[gmOffsetD], layoutD);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(flagAivFinishComputeList[stageId]);

        stageId = (stageId + 1 < WORKSPACE_STAGES) ? (stageId + 1) : 0;
    }

    CATLASS_DEVICE void RunAivCommSync(int32_t commIdx, int32_t commCount, uint64_t flag_idx, Params& params)
    {
        int64_t token_total = static_cast<int64_t>(params.commUtil.p_value) * L1TileShape::M * params.problemShape.n();
        if (commIdx == commCount - 1) {
            token_total =
                (params.problemShape.m() - static_cast<int64_t>(commIdx) * L1TileShape::M * params.commUtil.p_value) *
                params.problemShape.n();
        }
        int64_t token_per_rank = token_total / params.commUtil.rank_size;

        params.commUtil.SetAndWaitAivSync(flag_idx + WORKSPACE_STAGES);
        params.commUtil.CrossRankSyncV1(FLAG_ZERO_IDX, commIdx + 1);
        params.commUtil.SetAndWaitAivSync(flag_idx + WORKSPACE_STAGES);

        int64_t rank_offset =
            static_cast<int64_t>(params.problemShape.m()) * params.problemShape.n() / params.commUtil.rank_size;
        if (params.commUtil.aiv_idx == 0 && params.commUtil.core_idx < params.commUtil.rank_size) {
            int64_t src_offset = static_cast<int64_t>(flag_idx) * params.commUtil.gm_a_pingpong_size +
                                 params.commUtil.gm_a_pingpong_size / params.commUtil.rank_size * params.commUtil.rank;
            int64_t dst_offset = params.commUtil.core_idx * rank_offset +
                                 static_cast<int64_t>(commIdx) * L1TileShape::M * params.commUtil.p_value *
                                     (params.problemShape.n() / params.commUtil.rank_size);
            params.commUtil.CopyGMToGM((__gm__ ElementD*)params.commUtil.buff[params.commUtil.core_idx] + src_offset,
                                       reinterpret_cast<__gm__ ElementD*>(params.ptrOut) + dst_offset, token_per_rank);
        }

        params.commUtil.SetAndWaitAivSync(flag_idx + WORKSPACE_STAGES);
        params.commUtil.CrossRankSyncV1(FLAG_ONE_IDX, commIdx + 1);
        params.commUtil.SetAndWaitAivSync(flag_idx + WORKSPACE_STAGES);
        if (commIdx < commCount - 2) {
            params.commUtil.SetAicSync(flag_idx + WORKSPACE_STAGES * 2); // 避开matmul-dequant使用的flag_idx(0-3)
        }
    }

private:
    friend struct AicWaitFunc;
    friend struct AicSetFunc;

    struct AicWaitFunc {
        using MatmulKernel = QuantMatmulAllToAllKernel<BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES>;

        CATLASS_DEVICE
        AicWaitFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreWaitFlag(ptr->flagAivFinishComputeList[stageId]);
        }

        MatmulKernel* ptr{nullptr};
        uint32_t stageId;
    };

    struct AicSetFunc {
        using MatmulKernel = QuantMatmulAllToAllKernel<BlockMmad, BlockEpilogue, BlockScheduler, WORKSPACE_STAGES>;

        CATLASS_DEVICE
        AicSetFunc() = default;

        CATLASS_DEVICE
        void operator()() const
        {
            Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(ptr->flagAicFinishStoreList[stageId]);
        }

        MatmulKernel* ptr{nullptr};
        uint32_t stageId;
    };

    Arch::CrossCoreFlag flagAicFinishStoreList[WORKSPACE_STAGES];
    Arch::CrossCoreFlag flagAivFinishComputeList[WORKSPACE_STAGES];

    AicWaitFunc aicWaitFuncList[WORKSPACE_STAGES];
    AicSetFunc aicSetFuncList[WORKSPACE_STAGES];
    Arch::Resource<ArchTag> resource;
};
} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_MATMUL_ALLTO_ALL_QUANT_MATMUL_HPP
