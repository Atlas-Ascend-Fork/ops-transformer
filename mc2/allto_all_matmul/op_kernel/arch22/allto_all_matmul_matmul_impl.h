/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// matmul member definitions for AlltoAllMatmul.
#pragma once

#include "allto_all_matmul.h"

namespace Mc2Kernel {

template <typename ArchTag, typename AType_, typename BType_, typename CType_, typename BiasType_>
struct AlltoAllMatmulTileCopy : public Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_> {
    using Base = Catlass::Gemm::Tile::TileCopy<ArchTag, AType_, BType_, CType_, BiasType_>;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementAccumulator = typename Base::ElementAccumulator;

    using CopyGmToL1A = typename Base::CopyGmToL1A;
    using CopyGmToL1B = typename Base::CopyGmToL1B;

    using CopyL1ToL0A = typename Base::CopyL1ToL0A;
    using CopyL1ToL0B = typename Base::CopyL1ToL0B;

    using CopyL0CToGm = typename Base::CopyL0CToGm;
    using BiasTypeSelector = typename Base::BiasTypeSelector;
    using CopyGmToL1Bias = typename Base::CopyGmToL1Bias;
    using CopyL1ToBT = typename Base::CopyL1ToBT;
};

// 不同的matmul数据类型对应的L1TileShape/L0TileShape不同
template <typename ElementB>
constexpr uint32_t GetAlltoAllL1TileShapeK()
{
    return std::is_same<ElementB, int4b_t>::value ? 1024 : (std::is_same<ElementB, int8_t>::value ? 512 : 256);
}

template <typename ElementB>
constexpr uint32_t GetAlltoAllL0TileShapeK()
{
    return std::is_same<ElementB, int4b_t>::value ? 256 : (std::is_same<ElementB, int8_t>::value ? 128 : 64);
}

template <TemplateA2AMMClass, uint32_t TileM, uint32_t TileN>
struct AlltoAllMatmulTraits {
    static constexpr bool ENABLE_UNIT_FLAG = false;
    static constexpr bool ENABLE_SHUFFLE_K = false;
    // 计算量化后的矩阵乘，bias不由CatlassMatmul负责
    static constexpr bool aicCalBias = (QuantType == MC2_NON_QUANT) && hasBias;

    // 非量化场景、量化场景，A、B的入参类型一致；伪量化场景，A需要动态量化成BType
    using ElementA = BType;
    using ElementB = BType;
    // 非量化场景，Btype和CType一致；量化场景计算结果为int32_t
    using ElementC = std::conditional_t<QuantType != MC2_NON_QUANT, int32_t, CType>;
    using ElementBias = BiasType;
    using LayoutA = layout::RowMajor;
    // B转置
    using LayoutB = std::conditional_t<transB, layout::ColumnMajor, layout::RowMajor>;
    using LayoutC = layout::RowMajor;
    using LayoutBias = layout::VectorLayout;

    using DispatchPolicy = std::conditional_t<aicCalBias, Gemm::MmadAtlasA2PingpongBias<ENABLE_UNIT_FLAG>,
                                              Gemm::MmadAtlasA2Preload<ENABLE_UNIT_FLAG, ENABLE_SHUFFLE_K>>;

    using AType_ = Gemm::GemmType<ElementA, LayoutA>;
    using BType_ = Gemm::GemmType<ElementB, LayoutB>;
    using CType_ = Gemm::GemmType<ElementC, LayoutC>;
    using BiasType_ = std::conditional_t<aicCalBias, Gemm::GemmType<ElementBias, LayoutBias>, void>;
    using TileCopy = AlltoAllMatmulTileCopy<Arch::AtlasA2, AType_, BType_, CType_, BiasType_>;

    using L1TileShape = GemmShape<TileM, TileN, GetAlltoAllL1TileShapeK<ElementB>()>;
    using L0TileShape = GemmShape<TileM, TileN, GetAlltoAllL0TileShapeK<ElementB>()>;
    using BlockMmadOpt =
        Gemm::Block::BlockMmad<DispatchPolicy, L1TileShape, L0TileShape, AType_, BType_, CType_, BiasType_, TileCopy>;
    using BlockScheduler30 = typename Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
    using MatmulKernel =
        Gemm::Kernel::AlltoAllMatmulKernel<void, void, BlockMmadOpt, void, BlockScheduler30, aicCalBias>;
};

template <typename MatmulKernel, typename LayoutA_, typename LayoutB_, typename LayoutC_>
__aicore__ inline typename MatmulKernel::Params BuildAlltoAllMatmulParams(GM_ADDR biasGM, int32_t rankSize,
                                                                          const GemmCoord& processSize, GM_ADDR srcGM,
                                                                          GM_ADDR bGM, GM_ADDR matmulResultGM,
                                                                          LayoutA_ layoutA, LayoutB_ layoutB,
                                                                          LayoutC_ layoutC, int32_t pValue)
{
    return typename MatmulKernel::Params{processSize, srcGM,  layoutA, bGM, layoutB,  biasGM,         matmulResultGM,
                                         layoutC,     pValue, 3,       0,   rankSize, MAX_BLOCK_COUNT};
}

template <TemplateA2AMMClass>
template <uint32_t TileM, uint32_t TileN>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::CatlassMatmulImpl()
{
    using Traits = AlltoAllMatmulTraits<TemplateA2AMMFunc, TileM, TileN>;

    uint32_t realM = m / rankSize;
    uint32_t realK = k * rankSize;
    typename Traits::LayoutA layoutA{static_cast<uint32_t>(realM), static_cast<uint32_t>(realK)};
    typename Traits::LayoutB layoutB{static_cast<uint32_t>(realK), static_cast<uint32_t>(n)};
    typename Traits::LayoutC layoutC{static_cast<uint32_t>(realM), static_cast<uint32_t>(n)};
    typename Traits::LayoutBias layoutBias{static_cast<uint32_t>(n)};

    GemmCoord processSize{static_cast<uint32_t>(realM), static_cast<uint32_t>(n), static_cast<uint32_t>(realK)};

    // 动态量化时，需要更改左矩阵读取位置
    GM_ADDR srcGM =
        (QuantType == MC2_DYNAMIC_QUANT) ? reinterpret_cast<GM_ADDR>(quantAGM_) : reinterpret_cast<GM_ADDR>(gmPeerMem_);
    // 量化矩阵乘法时，需要修改c矩阵存放地址
    GM_ADDR matmulResultGM = (QuantType == MC2_NON_QUANT) ? cGM_ : reinterpret_cast<GM_ADDR>(dequantCGM_);

    typename Traits::MatmulKernel matmul_op;
    typename Traits::MatmulKernel::Params params = BuildAlltoAllMatmulParams<typename Traits::MatmulKernel>(
        biasGM_, static_cast<int32_t>(rankSize), processSize, srcGM, bGM_, matmulResultGM, layoutA, layoutB, layoutC,
        pValue);
    matmul_op(params);
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::CatlassMatmul()
{
    if ASCEND_IS_AIC {
        if (m0 == 128) {
            CatlassMatmulImpl<128, 256>();
        } else {
            CatlassMatmulImpl<256, 128>();
        }
    }
}

template <typename ArchTag, typename CType_, typename ScaleGType, typename PerTokenScaleGType, typename BiasGType,
          typename DType>
struct AlltoAllDequantTileCopy
    : public Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleGType, PerTokenScaleGType, DType> {
    using Base = Catlass::Epilogue::Tile::TileCopy<ArchTag, CType_, ScaleGType, PerTokenScaleGType, DType>;
    using ElementC = typename Base::ElementC;
    using ElementScale = typename Base::ElementX;
    using ElementPerTokenScale = typename Base::ElementY;
    using ElementBias = typename BiasGType::Element;
    using ElementD = typename Base::ElementD;

    using CopyGmToUbC = typename Base::CopyGmToUbC;
    using CopyGmToUbScale = typename Base::CopyGmToUbX;
    using CopyGmToUbPerTokenScale = typename Base::CopyGmToUbY;
    using CopyGmToUbBias = Catlass::Epilogue::Tile::CopyGm2Ub<ArchTag, BiasGType>;
    using CopyUbToGmD = typename Base::CopyUbToGmD;
};

template <TemplateA2AMMClass, typename EpilogueTileShape>
struct AlltoAllDequantTraits {
    using ElementC = int32_t;
    using ElementBias = BiasType;
    using ElementD = CType;
    using LayoutD = layout::RowMajor;

    using CType_ = Gemm::GemmType<ElementC, layout::RowMajor>;

    static constexpr uint32_t ubStages = 2;
    using EpilogueDispatchPolicy = Epilogue::EpilogueAtlasA2PerTokenDequant<ubStages>;
    using ScaleGType = Gemm::GemmType<ScaleType, layout::VectorLayout>;
    using PerTokenScaleGType = Gemm::GemmType<float, layout::VectorLayout>; // 全量化时，也限定为float
    using BiasGType = Gemm::GemmType<BiasType, layout::VectorLayout>;
    using DType = Gemm::GemmType<CType, layout::RowMajor>;

    using RowBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;
    using RowBroadcastAddType = Gemm::GemmType<float, layout::RowMajor>;
    using BroadcastOneBlkType = Gemm::GemmType<float, layout::RowMajor>;
    using OneBlkColumnBroadcastMulType = Gemm::GemmType<float, layout::RowMajor>;

    using TileCopyDequant =
        AlltoAllDequantTileCopy<Arch::AtlasA2, CType_, ScaleGType, PerTokenScaleGType, BiasGType, DType>;

    using EpilogueTileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;

    using TileRowBroadcastMul =
        Epilogue::Tile::TileRowBroadcastMul<Arch::AtlasA2, RowBroadcastMulType, EpilogueTileShape>;
    using TileRowBroadcastAdd =
        Epilogue::Tile::TileRowBroadcastAdd<Arch::AtlasA2, RowBroadcastAddType, EpilogueTileShape>;
    using TileBroadcastOneBlk =
        Epilogue::Tile::TileBroadcastOneBlk<Arch::AtlasA2, BroadcastOneBlkType, EpilogueTileShape::ROW>;
    using TileOneBlkColumnBroadcastMul =
        Epilogue::Tile::TileOneBlkColumnBroadcastMul<Arch::AtlasA2, OneBlkColumnBroadcastMulType, EpilogueTileShape>;
    using QuantBlockEpilogue =
        Epilogue::Block::BlockEpilogue<EpilogueDispatchPolicy, CType_, ScaleGType, PerTokenScaleGType, BiasGType, DType,
                                       TileRowBroadcastMul, TileRowBroadcastAdd, TileBroadcastOneBlk,
                                       TileOneBlkColumnBroadcastMul, TileCopyDequant, EpilogueTileScheduler>;
};

template <TemplateA2AMMClass>
template <typename EpilogueTileShape>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::DequantImpl()
{
    using Traits = AlltoAllDequantTraits<TemplateA2AMMFunc, EpilogueTileShape>;
    using ElementC = typename Traits::ElementC;
    using ElementBias = typename Traits::ElementBias;
    using ElementD = typename Traits::ElementD;
    using LayoutD = typename Traits::LayoutD;

    uint32_t realM = m / rankSize;
    uint32_t realK = k * rankSize;
    LayoutD layoutD{static_cast<uint32_t>(realM), static_cast<uint32_t>(n)};

    layout::VectorLayout layoutScale{static_cast<uint32_t>(n)};
    layout::VectorLayout layoutPerTokenScale{static_cast<uint32_t>(realM)};
    layout::VectorLayout layoutBias{static_cast<uint32_t>(n)};

    GemmCoord problemShape{static_cast<uint32_t>(realM), static_cast<uint32_t>(n), static_cast<uint32_t>(realK)};

    AscendC::GlobalTensor<ElementD> gmD;
    gmD.SetGlobalBuffer((__gm__ ElementD*)cGM_);
    AscendC::GlobalTensor<ElementC> gmC;
    gmC.SetGlobalBuffer((__gm__ ElementC*)dequantCGM_);

    AscendC::GlobalTensor<ElementBias> gmBias;
    if (biasGM_ != nullptr) {
        gmBias.SetGlobalBuffer((__gm__ ElementBias*)biasGM_);
    }

    uint32_t rowsPerCore = DivCeil(problemShape.m(), blockNum);
    uint32_t rowsThisCore = rowsPerCore;
    uint32_t stRowPerCore = aicIdx * rowsPerCore;
    if (stRowPerCore < problemShape.m()) {
        if (rowsThisCore + stRowPerCore > problemShape.m()) {
            rowsThisCore = problemShape.m() - stRowPerCore;
        }
    } else {
        rowsThisCore = 0;
    }
    MatrixCoord coreOffset(stRowPerCore, 0u);
    auto layoutC = layout::RowMajor{problemShape.m(), n};
    int64_t gmOffsetC = layoutC.GetOffset(coreOffset);
    GemmCoord actualBlockShape{rowsThisCore, n, 1};

    GM_ADDR x1ScaleGM =
        (QuantType == MC2_DYNAMIC_QUANT) ? quantScaleGM_ : x1ScaleGM_; // 根据是否是全量化，决定从哪里获得pertokenScale

    typename Traits::QuantBlockEpilogue blockEpilogue(resource);

    using EpilogueParams = typename Traits::QuantBlockEpilogue::Params;
    EpilogueParams epilogueParams{
        scaleGM_, layoutScale, x1ScaleGM, layoutPerTokenScale.GetTileLayout(problemShape.template GetCoordByAxis<0>()),
        biasGM_,  layoutBias};
    blockEpilogue.UpdateParams(epilogueParams);

    blockEpilogue(coreOffset, actualBlockShape, gmC[gmOffsetC], layoutC, gmD[gmOffsetC], layoutC);
}

template <TemplateA2AMMClass>
__aicore__ inline void AlltoAllMatmul<TemplateA2AMMFunc>::Dequant()
{
    if (m0 == 128) {
        DequantImpl<MatrixShape<32, 256>>();
    } else {
        DequantImpl<MatrixShape<64, 128>>();
    }
}

} // namespace Mc2Kernel
