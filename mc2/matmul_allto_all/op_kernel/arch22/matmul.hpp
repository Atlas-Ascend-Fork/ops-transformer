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
 * \file matmul.hpp
 * \brief
 */

#ifndef CATLASS_GEMM_KERNEL_MATMUL_ALLTOALL_HPP
#define CATLASS_GEMM_KERNEL_MATMUL_ALLTOALL_HPP

#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_catlass.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_gemm_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_matrix_coord.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_resource.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/arch/tla_arch_cross_core_sync.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_gm_to_ub.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/epilogue/tile/tla_epilogue_copy_ub_to_gm.hpp"
#include "../../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/gemm/kernel/tla_gemm_kernel_padding_matmul.hpp"

namespace Catlass::Gemm::Kernel {
template <class PrologueA, class PrologueB, class BlockMmad_, class BlockEpilogue_, class BlockScheduler_, bool HasBias>
class MatmulAlltoAllKernel : public CommBase {
public:
    using BlockMmad = BlockMmad_;
    using ArchTag = typename BlockMmad::ArchTag;
    using ElementA = typename BlockMmad::ElementA;
    using ElementB = typename BlockMmad::ElementB;
    using LayoutWA = typename BlockMmad::LayoutA;
    using LayoutWB = typename BlockMmad::LayoutB;

    template <bool condition, class mmad>
    struct BiasTypeHelper {
        using type = typename mmad::ElementBias;
    };

    template <class mmad>
    struct BiasTypeHelper<false, mmad> {
        using type = float;
    };

    template <class T>
    struct LayoutHelper {
        using type = typename T::LayoutIn;
    };
    template <>
    struct LayoutHelper<void> {
        using type = void;
    };

    using ElementBias = typename BiasTypeHelper<HasBias, BlockMmad>::type;
    using LayoutA = std::conditional_t<std::is_void_v<PrologueA>, LayoutWA, typename LayoutHelper<PrologueA>::type>;
    using LayoutB = std::conditional_t<std::is_void_v<PrologueB>, LayoutWB, typename LayoutHelper<PrologueB>::type>;

    using L1TileShape = typename BlockMmad::L1TileShape;
    using ElementC = typename BlockMmad::ElementC;
    using LayoutC = typename BlockMmad::LayoutC;

    using BlockScheduler = BlockScheduler_;

    /// Parameters structure
    struct Params {
        // Data members
        GemmCoord problemShape;
        GM_ADDR ptrA;
        LayoutA layoutA;
        GM_ADDR ptrB;
        LayoutB layoutB;
        GM_ADDR ptrBias;
        GM_ADDR ptrC;
        LayoutC layoutC;
        int32_t pValue;
        int32_t rankSize;
        int32_t pipeDepth;
        bool transB;

        // Methods
        CATLASS_HOST_DEVICE
        Params() {}

        CATLASS_HOST_DEVICE
        Params(GemmCoord const& problemShape_, GM_ADDR ptrA_, LayoutA layoutA_, GM_ADDR ptrB_, LayoutB layoutB_,
               GM_ADDR ptrBias_, GM_ADDR ptrC_, LayoutC layoutC_, int32_t pValue_, int32_t rankSize_,
               int32_t pipeDepth_, bool transB_)
            : problemShape(problemShape_),
              ptrA(ptrA_),
              layoutA(layoutA_),
              ptrB(ptrB_),
              layoutB(layoutB_),
              ptrBias(ptrBias_),
              ptrC(ptrC_),
              layoutC(layoutC_),
              pValue(pValue_),
              rankSize(rankSize_),
              pipeDepth(pipeDepth_),
              transB(transB_)
        {}
    };

    // Methods
    CATLASS_DEVICE
    MatmulAlltoAllKernel() {}

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params)
    {
        if constexpr (!std::is_void_v<PrologueA> || !std::is_void_v<PrologueB>) {
            Catlass::Arch::CrossCoreWaitFlag(flagAivFinishPadding);
        }

        GmTensors tensors;
        InitGmTensors(tensors, params);

        BlockMmad blockMmad(resource);

        int32_t coreIdx = AscendC::GetBlockIdx();
        int32_t coreNum = AscendC::GetBlockNum();
        int32_t mLoops = (params.problemShape.m() + L1TileShape::M - 1) / L1TileShape::M;
        int32_t commCount = (mLoops + params.pValue - 1) / params.pValue;

        for (int32_t commIdx = 0; commIdx < commCount; commIdx++) {
            LoopCtx ctx;
            ctx.commIdx = commIdx;
            ctx.commCount = commCount;
            ctx.mLoops = mLoops;
            ctx.coreIdx = coreIdx;
            ctx.coreNum = coreNum;
            ctx.flagIdx = commIdx % params.pipeDepth;

            RunCommBlock(ctx, blockMmad, tensors, params);
            FFTSCrossCoreSync<PIPE_FIX, 2>(ctx.flagIdx);
        }
    }

private:
    // 每次 matmul 前的全局张量视图
    struct GmTensors {
        AscendC::GlobalTensor<ElementA> gmA;
        AscendC::GlobalTensor<ElementB> gmB;
        AscendC::GlobalTensor<ElementC> gmC;
        AscendC::GlobalTensor<ElementBias> gmBias;
    };

    // 单个 comm 轮次的循环参数
    struct LoopCtx {
        int32_t commIdx{0};
        int32_t commCount{0};
        int32_t mLoops{0};
        int32_t actualLoopNum{0};
        int32_t nLoopPerRank{0};
        int32_t coreIdx{0};
        int32_t coreNum{0};
        uint64_t flagIdx{0};
        int64_t gmAPingpongSize{0};
    };

    CATLASS_DEVICE static void InitGmTensors(GmTensors& tensors, Params const& params)
    {
        // Represent the full gm
        tensors.gmA.SetGlobalBuffer((__gm__ ElementA*)params.ptrA);
        tensors.gmB.SetGlobalBuffer((__gm__ ElementB*)params.ptrB);
        tensors.gmC.SetGlobalBuffer((__gm__ ElementC*)params.ptrC);
        if constexpr (HasBias) {
            tensors.gmBias.SetGlobalBuffer((__gm__ ElementBias*)params.ptrBias);
        }
    }

    CATLASS_DEVICE static uint32_t GetMActual(int32_t commIdx, int32_t commCount, int32_t mLoops, uint32_t blockM,
                                              Params const& params)
    {
        bool isLastComm = (commIdx == commCount - 1);
        bool isLastBlock = (blockM == static_cast<uint32_t>((mLoops - 1) % params.pValue));
        bool isTailM = (params.problemShape.m() % L1TileShape::M != 0);
        if (!isLastComm || !isLastBlock || !isTailM) {
            return L1TileShape::M;
        }
        return static_cast<uint32_t>(params.problemShape.m() - commIdx * params.pValue * L1TileShape::M -
                                     blockM * L1TileShape::M);
    }

    CATLASS_DEVICE static uint32_t GetNActual(uint32_t blockN, int32_t nLoopPerRank, Params const& params)
    {
        uint32_t tailOffset = blockN % static_cast<uint32_t>(nLoopPerRank);
        if (tailOffset != static_cast<uint32_t>(nLoopPerRank - 1)) {
            return L1TileShape::N;
        }
        return params.problemShape.n() / static_cast<uint32_t>(params.rankSize) - tailOffset * L1TileShape::N;
    }

    CATLASS_DEVICE static int64_t GetGmOffsetA(int32_t commIdx, GemmCoord const& blockIdxCoord, Params const& params)
    {
        MatrixCoord offsetA{blockIdxCoord.m() * L1TileShape::M, blockIdxCoord.k() * L1TileShape::K};
        return params.layoutA.GetOffset(offsetA) +
               static_cast<int64_t>(commIdx) * L1TileShape::M * params.pValue * params.problemShape.k();
    }

    CATLASS_DEVICE static int64_t GetGmOffsetB(GemmCoord const& blockIdxCoord, int32_t nLoopPerRank,
                                               Params const& params)
    {
        return (static_cast<int64_t>(blockIdxCoord.n() / nLoopPerRank) * (params.problemShape.n() / params.rankSize) +
                static_cast<int64_t>(blockIdxCoord.n() % nLoopPerRank) * L1TileShape::N) *
               (params.transB ? params.problemShape.k() : 1);
    }

    CATLASS_DEVICE static int64_t GetGmOffsetC(uint64_t flagIdx, int64_t gmAPingpongSize,
                                               GemmCoord const& blockIdxCoord, int32_t nLoopPerRank,
                                               Params const& params)
    {
        int64_t rankOffset = gmAPingpongSize / params.rankSize;
        int64_t rankIdx = blockIdxCoord.n() / nLoopPerRank;
        return static_cast<int64_t>(flagIdx) * L1TileShape::M * params.pValue * params.problemShape.n() +
               rankIdx * rankOffset +
               static_cast<int64_t>(blockIdxCoord.m()) * L1TileShape::M * (params.problemShape.n() / params.rankSize) +
               blockIdxCoord.n() % nLoopPerRank * L1TileShape::N;
    }

    CATLASS_DEVICE static int64_t GetGmOffsetBias(GemmCoord const& blockIdxCoord, int32_t nLoopPerRank,
                                                  Params const& params)
    {
        return (blockIdxCoord.n() / nLoopPerRank) * (params.problemShape.n() / params.rankSize) +
               (blockIdxCoord.n() % nLoopPerRank) * L1TileShape::N;
    }

    CATLASS_DEVICE void RunCommBlock(LoopCtx& ctx, BlockMmad& blockMmad, GmTensors& tensors, Params const& params)
    {
        int32_t tailN = (params.problemShape.n() / params.rankSize) % L1TileShape::N;
        uint32_t nLoops = params.problemShape.n() / params.rankSize / L1TileShape::N;
        if (tailN) {
            nLoops += 1;
        }
        nLoops *= params.rankSize;

        uint32_t actualPValue = params.pValue;
        if (ctx.commIdx == ctx.commCount - 1) {
            actualPValue = ctx.mLoops - ctx.commIdx * params.pValue;
        }
        if (ctx.commIdx >= params.pipeDepth) {
            WaitEvent(ctx.flagIdx);
        }
        ctx.gmAPingpongSize = static_cast<int64_t>(L1TileShape::M) * params.pValue * params.problemShape.n();
        ctx.actualLoopNum = actualPValue * nLoops;
        ctx.nLoopPerRank = nLoops / params.rankSize;
        GemmCoord calSwizzleShape{actualPValue, nLoops, 1};
        BlockScheduler matmulBlockScheduler(calSwizzleShape,
                                            MakeCoord(static_cast<uint32_t>(1), static_cast<uint32_t>(1)));

        for (int32_t loopOffset = 0; loopOffset < ctx.actualLoopNum; loopOffset++) {
            int32_t loopIdx = ctx.commIdx * params.pValue * nLoops + loopOffset;
            if (loopIdx % ctx.coreNum != ctx.coreIdx) {
                continue;
            }
            RunBlock(ctx, loopOffset, matmulBlockScheduler, blockMmad, tensors, params);
        }
    }

    CATLASS_DEVICE void RunBlock(LoopCtx const& ctx, int32_t loopOffset, BlockScheduler& matmulBlockScheduler,
                                 BlockMmad& blockMmad, GmTensors& tensors, Params const& params)
    {
        GemmCoord blockIdxCoord = matmulBlockScheduler.GetBlockCoord(loopOffset);
        uint32_t mActual = GetMActual(ctx.commIdx, ctx.commCount, ctx.mLoops, blockIdxCoord.m(), params);
        uint32_t nActual = GetNActual(blockIdxCoord.n(), ctx.nLoopPerRank, params);
        GemmCoord actualBlockShape{mActual, nActual, params.problemShape.k()};
        int64_t gmOffsetA = GetGmOffsetA(ctx.commIdx, blockIdxCoord, params);
        int64_t gmOffsetB = GetGmOffsetB(blockIdxCoord, ctx.nLoopPerRank, params);
        int64_t gmOffsetC = GetGmOffsetC(ctx.flagIdx, ctx.gmAPingpongSize, blockIdxCoord, ctx.nLoopPerRank, params);

        if constexpr (HasBias) {
            int64_t gmOffsetBias = GetGmOffsetBias(blockIdxCoord, ctx.nLoopPerRank, params);
            blockMmad(tensors.gmA[gmOffsetA], params.layoutA, tensors.gmB[gmOffsetB], params.layoutB,
                      tensors.gmC[gmOffsetC], params.layoutC, tensors.gmBias[gmOffsetBias], actualBlockShape);
        } else {
            bool isFirstBlock = (loopOffset < ctx.coreNum);
            bool hasNextBlock = false;
            GemmCoord nextBlockIdCoord;
            GemmCoord nextActualBlockShape;
            if (loopOffset + ctx.coreNum < ctx.actualLoopNum) {
                hasNextBlock = true;
                nextBlockIdCoord = matmulBlockScheduler.GetBlockCoord(loopOffset + ctx.coreNum);
                uint32_t mActualNext = GetMActual(ctx.commIdx, ctx.commCount, ctx.mLoops, nextBlockIdCoord.m(), params);
                uint32_t nActualNext = GetNActual(nextBlockIdCoord.n(), ctx.nLoopPerRank, params);
                nextActualBlockShape = GemmCoord{mActualNext, nActualNext, params.problemShape.k()};
            }
            int64_t gmOffsetNextA = GetGmOffsetA(ctx.commIdx, nextBlockIdCoord, params);
            int64_t gmOffsetNextB = GetGmOffsetB(nextBlockIdCoord, ctx.nLoopPerRank, params);

            blockMmad(tensors.gmA[gmOffsetA], params.layoutA, tensors.gmB[gmOffsetB], params.layoutB,
                      tensors.gmC[gmOffsetC], params.layoutC, tensors.gmA[gmOffsetNextA], tensors.gmB[gmOffsetNextB],
                      actualBlockShape, nextActualBlockShape, isFirstBlock, hasNextBlock);
        }
    }

private:
    static constexpr Arch::FlagID FLAG_AIV_FINISH_STORE = 0;
    Arch::CrossCoreFlag flagAivFinishPadding{FLAG_AIV_FINISH_STORE};
    Arch::Resource<ArchTag> resource;
};

} // namespace Catlass::Gemm::Kernel

#endif // CATLASS_GEMM_KERNEL_MATMUL_ALLTO_ALL_HPP
