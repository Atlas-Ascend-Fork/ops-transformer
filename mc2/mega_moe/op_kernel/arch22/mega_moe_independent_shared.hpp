/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_INDEPENDENT_SHARED_HPP
#define MEGA_MOE_INDEPENDENT_SHARED_HPP

namespace MegaMoeImpl {

template <typename Weight, bool Nz, bool Transpose, bool Mixed, bool SharedW2Nz>
struct IndependentSharedTypes {
    static constexpr bool w4 = std::is_same_v<Weight, AscendC::int4b_t> || std::is_same_v<Weight, int32_t>;
    static constexpr bool w8 = std::is_same_v<Weight, int8_t>;
    using B = std::conditional_t<w4, AscendC::int4b_t, Weight>;
    using A = std::conditional_t<w4, AscendC::int4b_t, std::conditional_t<w8, int8_t, bfloat16_t>>;
    using C = std::conditional_t<w8 && !Mixed, int32_t, std::conditional_t<w4 || w8, float16_t, bfloat16_t>>;
    using D = std::conditional_t<Mixed || !(w4 || w8), bfloat16_t, int8_t>;
    using ArchTag = Catlass::Arch::AtlasA2;
    using LA = Catlass::layout::RowMajor;
    using LB =
        std::conditional_t<Nz, Catlass::layout::zN, std::conditional_t<Transpose, Catlass::layout::ColumnMajor, LA>>;
    using AT = Catlass::Gemm::GemmType<A, LA>;
    using BT = Catlass::Gemm::GemmType<B, LB>;
    using CT = Catlass::Gemm::GemmType<C, LA>;
    using DT = Catlass::Gemm::GemmType<D, LA>;
    using Out = Catlass::Gemm::GemmType<bfloat16_t, LA>;
    using Scale = Catlass::Gemm::GemmType<uint64_t, Catlass::layout::VectorLayout>;
    using TokenScale = Catlass::Gemm::GemmType<float, Catlass::layout::VectorLayout>;
    using Policy = Catlass::Gemm::MmadDispatchPolicyFor_t<B, false, 1, 2, 2, 2, 1, false, true>;
    using Tile = std::conditional_t<
        w4, Catlass::GemmShape<128, 256, 1024>,
        std::conditional_t<w8, Catlass::GemmShape<128, 256, 512>, Catlass::GemmShape<128, 256, 256>>>;
    using L0 =
        std::conditional_t<w4, Catlass::GemmShape<128, 256, 256>,
                           std::conditional_t<w8, Catlass::GemmShape<128, 256, 128>, Catlass::GemmShape<128, 256, 64>>>;
    using QuantCopy = Catlass::Gemm::Tile::QuantTileCopy<ArchTag, AT, BT, CT, void,
                                                         Catlass::Gemm::Tile::ScaleGranularity::PER_CHANNEL>;
    using Mmad = std::conditional_t<w4, Catlass::Gemm::Block::BlockMmad<Policy, Tile, L0, AT, BT, CT, void, QuantCopy>,
                                    Catlass::Gemm::Block::BlockMmad<Policy, Tile, L0, AT, BT, CT>>;
    using Mul = Catlass::Epilogue::Tile::TileElemWiseMuls<ArchTag, Catlass::Gemm::GemmType<float, LA>, 0>;
    using Copy1 = Catlass::Epilogue::Tile::TileCopy<ArchTag, CT, Scale, TokenScale, DT>;
    using Copy2 = Catlass::Epilogue::Tile::TileCopy<ArchTag, CT, Scale, TokenScale, Out>;
    using P1 = std::conditional_t<w4, Catlass::Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantSwigluQuant<2>,
                                  Catlass::Epilogue::EpilogueAtlasA2PerTokenDequantSwigluQuant<2>>;
    using P2 = std::conditional_t<w4, Catlass::Epilogue::EpilogueAtlasA2W4A8PostPerTokenDequantV2<2>,
                                  Catlass::Epilogue::EpilogueAtlasA2PerTokenDequant<2>>;
    using E1 = Catlass::Epilogue::Block::BlockEpilogue<P1, CT, TokenScale, DT, Mul, Copy1>;
    using E2 = std::conditional_t<w4,
                                  Catlass::Epilogue::Block::BlockEpilogue<P2, CT, TokenScale, Out, Copy2,
                                                                          Catlass::Epilogue::Block::IS_A2_T<false>>,
                                  Catlass::Epilogue::Block::BlockEpilogue<P2, CT, TokenScale, Out, Copy2>>;
    using Bf = Catlass::Gemm::GemmType<bfloat16_t, LA>;
    using LBF = std::conditional_t<Mixed && !SharedW2Nz, LA, LB>;
    using BfW = Catlass::Gemm::GemmType<bfloat16_t, LBF>;
    using BfPolicy = Catlass::Gemm::MmadDispatchPolicyFor_t<bfloat16_t, false, 1, 2, 2, 2, 1, false, true>;
    using BfMmad = Catlass::Gemm::Block::BlockMmad<BfPolicy, Catlass::GemmShape<128, 256, 256>,
                                                   Catlass::GemmShape<128, 256, 64>, Bf, BfW, Bf>;
    using Scheduler = Catlass::Gemm::Block::GemmIdentityBlockSwizzle<9, 1>;
    using Kernel = Catlass::Gemm::Kernel::SharedExpert<Mmad, Scheduler, E1, E2, BfMmad, Mixed>;
};

// A serial, fully drained stage within the same mixed kernel. Routing keeps its
// own type/layout and original fast path; this stage owns the shared types/I.
template <typename Weight, bool Nz, bool Transpose, bool Mixed, bool SharedW2Nz>
__aicore__ inline GM_ADDR IndependentShared(const MegaMoeA2A3TilingData& common, GM_ADDR input, GM_ADDR workspace,
                                            GM_ADDR w1, GM_ADDR w2, GM_ADDR s1, GM_ADDR s2, GM_ADDR b1, GM_ADDR b2,
                                            uint32_t hidden, uint32_t activation, float clamp, float param1,
                                            float param2)
{
    using Types = IndependentSharedTypes<Weight, Nz, Transpose, Mixed, SharedW2Nz>;
    using Kernel = typename Types::Kernel;
    using LB = typename Types::LB;
    using LBF = typename Types::LBF;
    using B = typename Types::B;
    const uint32_t n = common.sharedN;
    typename Kernel::Params params{Catlass::GemmCoord{common.M, n, hidden},
                                   input,
                                   workspace,
                                   w1,
                                   w2,
                                   s1,
                                   s2,
                                   b1,
                                   b2,
                                   Catlass::layout::RowMajor{common.M, n / 2},
                                   LayoutBInitializer<LB, B>::create(hidden, n),
                                   LayoutBInitializer<LB, B>::create(n / 2, hidden),
                                   LayoutBInitializer<LBF, bfloat16_t>::create(n / 2, hidden),
                                   Catlass::layout::VectorLayout{n},
                                   Catlass::layout::VectorLayout{hidden},
                                   common.sharedWorkspaceOffset,
                                   common.sharedExpertNum,
                                   common.aivNum,
                                   clamp,
                                   activation,
                                   param1,
                                   param2};
    Kernel kernel(params);
    kernel.Process(params);
    return kernel.Result();
}
} // namespace MegaMoeImpl
#endif
