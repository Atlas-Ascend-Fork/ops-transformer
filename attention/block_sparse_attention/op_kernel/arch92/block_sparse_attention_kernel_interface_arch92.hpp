/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_sparse_attention_kernel_interface_arch92.hpp
 * \brief Block Sparse Attention Interface
 */
#pragma once

#include "block_sparse_attention_kernel_arch92_regular.h"
#include "block_sparse_attention_kernel_arch92_full_quant.h"
#include "block_sparse_attention_kernel_arch92_mxfp4_full_quant.h"
#include "block_sparse_attention_kernel_arch92_mxfp8_full_quant.h"
#include "block_sparse_attention_kernel_arch92_hif8_full_quant.h"

using namespace NpuArch;

using namespace BsaKernelArch92;

template <class InDtype, class SMDtype, class REDtype, Format qFormat, Format kvFormat, Epilogue::LseMode lseMode,
          Epilogue::LseFormat lseFormat, bool transposedMm1>
__global__ __aicore__ void BsaInferIntfRegular(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables,
                                               GM_ADDR o, GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen,
                                               GM_ADDR blockSparseMask, GM_ADDR workspace, GM_ADDR lse, GM_ADDR tiling)
{
    using ArchTag = Arch::Ascend960DT;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InDtype;
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype;
    using ElementP = InDtype;
    using ElementO = InDtype;
    using ElementOTmp = REDtype;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    using LayoutPDummy = std::conditional_t<transposedMm1, layout::nZ, layout::zN>;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;
    // block mask pre-process
    using DispatchPolicyMask2Idx = Epilogue::Arch92::EpilogueBsaMask2Idx;
    using EpilogueMask2Idx =
        Epilogue::Block::BlockEpilogue<DispatchPolicyMask2Idx, ElementSparseMask, ElementSparseIdx, ElementSparseCount>;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::Arch92::MmadBsaQK<transposedMm1, false>;
    using TileCopyQK =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, layout::RowMajor, ElementK, layout::ColumnMajor, ElementS,
                                          LayoutS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;
    // online softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::Arch92::EpilogueOnlineSoftmaxBsa<transposedMm1>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::Arch92::MmadBsaPV<transposedMm1>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::Arch92::EpilogueBsaRescaleO<lseMode, lseFormat, transposedMm1, false>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementK, TileCopyRescaleO, Arch::PositionL0C>;

    using BsaRegularKernelArch92 =
        BsaRegularKernelArch92<EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                               qFormat, kvFormat, transposedMm1>;
    BsaKernelParamsArch92 params{q, k,         v,   mask,  blockTables, actualQseqlen, actualKvseqlen, blockSparseMask,
                                 o, workspace, lse, tiling};
    BsaRegularKernelArch92 bsaRegularKernelArch92;
    bsaRegularKernelArch92(params);
}

template <class InDtype, class SMDtype, class REDtype, Format qFormat, Format kvFormat, Epilogue::LseMode lseMode,
          Epilogue::LseFormat lseFormat>
__global__ __aicore__ void BsaInferInterfaceFullQuant(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                      GM_ADDR blockSparseMask, GM_ADDR attenMask, GM_ADDR blockTable,
                                                      GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKv,
                                                      GM_ADDR qDequantScale, GM_ADDR kDequantScale,
                                                      GM_ADDR vDequantScale, GM_ADDR attentionOut, GM_ADDR workspace,
                                                      GM_ADDR lse, GM_ADDR tiling)
{
    using ArchTag = Arch::Ascend960DT;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InDtype;    // fp8_e4m3fn_t
    using ElementK = InDtype;    // fp8_e4m3fn_t
    using ElementV = InDtype;    // fp8_e4m3fn_t
    using ElementS = SMDtype;    // half/bfloat16_t
    using ElementP = InDtype;    // fp8_e4m3fn_t
    using ElementO = SMDtype;    // half/bfloat16_t
    using ElementOTmp = REDtype; // float
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;
    // block mask pre-process
    using DispatchPolicyMask2Idx = Epilogue::Arch92::EpilogueBsaMask2Idx;
    using EpilogueMask2Idx =
        Epilogue::Block::BlockEpilogue<DispatchPolicyMask2Idx, ElementSparseMask, ElementSparseIdx, ElementSparseCount>;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    // mmad qk
    using DispatchPolicyQK = Gemm::Arch92::MmadBsaQK<false, false>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                                         Gemm::Tile::ScaleGranularity::PER_TENSOR>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;
    // online softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::Arch92::EpilogueOnlineSoftmaxBsa<false>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;
    // 处理单个tile内P和V的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    // mmad pv
    using DispatchPolicyPV = Gemm::Arch92::MmadBsaPV<false>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                          Gemm::Tile::ScaleGranularity::PER_TENSOR>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;
    // rescale o
    using DispatchPolicyRescaleO = Epilogue::Arch92::EpilogueBsaRescaleO<lseMode, lseFormat, false, false>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementK, TileCopyRescaleO, Arch::PositionL0C>;
    BsaFullQuantKernelParamsArch92 params{query,
                                          key,
                                          value,
                                          blockSparseMask,
                                          attenMask,
                                          blockTable,
                                          actualSeqLengths,
                                          actualSeqLengthsKv,
                                          qDequantScale,
                                          kDequantScale,
                                          vDequantScale,
                                          attentionOut,
                                          workspace,
                                          lse,
                                          tiling};
    using BsaFullQuantKernelArch92 =
        BsaFullQuantKernelArch92<EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                                 qFormat, kvFormat, false, false>;
    BsaFullQuantKernelArch92 bsaFullQuantKernelArch92;
    bsaFullQuantKernelArch92(params);
}

template <class InDtype, class SMDtype, class REDtype, Format qFormat, Format kvFormat, Epilogue::LseMode lseMode,
          Epilogue::LseFormat lseFormat, bool transposedMm1, bool zNOnlineSoftmax>
__global__ __aicore__ void BsaInferIntfB8FullQuantOpt(GM_ADDR query, GM_ADDR key, GM_ADDR value,
                                                      GM_ADDR blockSparseMask, GM_ADDR attenMask, GM_ADDR blockTable,
                                                      GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKv,
                                                      GM_ADDR qDequantScale, GM_ADDR kDequantScale,
                                                      GM_ADDR vDequantScale, GM_ADDR attentionOut, GM_ADDR workspace,
                                                      GM_ADDR lse, GM_ADDR tiling)
{
    using ArchTag = Arch::Ascend960DT;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InDtype;    // fp8_e4m3fn_t
    using ElementK = InDtype;    // fp8_e4m3fn_t
    using ElementV = InDtype;    // fp8_e4m3fn_t
    using ElementS = SMDtype;    // half/bfloat16_t
    using ElementP = InDtype;    // fp8_e4m3fn_t
    using ElementO = SMDtype;    // half/bfloat16_t
    using ElementOTmp = REDtype; // float
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor or zN on UB(dst)
    using LayoutS = std::conditional_t<zNOnlineSoftmax, layout::zN, layout::RowMajor>;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    using LayoutPDummy = layout::zN;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;
    // block mask pre-process
    using DispatchPolicyMask2Idx = Epilogue::Arch92::EpilogueBsaMask2Idx;
    using EpilogueMask2Idx =
        Epilogue::Block::BlockEpilogue<DispatchPolicyMask2Idx, ElementSparseMask, ElementSparseIdx, ElementSparseCount>;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    // mmad qk
    using DispatchPolicyQK = Gemm::Arch92::MmadBsaQK<transposedMm1, zNOnlineSoftmax>;
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                                         Gemm::Tile::ScaleGranularity::PER_TENSOR>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;
    // online softmax
    using DispatchPolicyOnlineSoftmax =
        Epilogue::Arch92::EpilogueB8FullQuantOnlineSoftmaxBsa<transposedMm1, zNOnlineSoftmax>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;
    // 处理单个tile内P和V的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    // mmad pv
    using DispatchPolicyPV = Gemm::Arch92::MmadBsaPV<false>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                          Gemm::Tile::ScaleGranularity::PER_TENSOR>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;
    // rescale o
    using DispatchPolicyRescaleO = Epilogue::Arch92::EpilogueBsaRescaleO<lseMode, lseFormat, false, zNOnlineSoftmax>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementK, TileCopyRescaleO, Arch::PositionL0C>;
    BsaFullQuantKernelParamsArch92 params{query,
                                          key,
                                          value,
                                          blockSparseMask,
                                          attenMask,
                                          blockTable,
                                          actualSeqLengths,
                                          actualSeqLengthsKv,
                                          qDequantScale,
                                          kDequantScale,
                                          vDequantScale,
                                          attentionOut,
                                          workspace,
                                          lse,
                                          tiling};
    using BsaFullQuantKernelArch92 =
        BsaFullQuantKernelArch92<EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV, EpilogueRescaleO,
                                 qFormat, kvFormat, transposedMm1, zNOnlineSoftmax>;
    BsaFullQuantKernelArch92 bsaFullQuantKernelArch92;
    bsaFullQuantKernelArch92(params);
}

template <class InDtype, class SMDtype, class REDtype, class Otype, Format qFormat, Format kvFormat,
          Epilogue::LseMode lseMode, Epilogue::LseFormat lseFormat, Epilogue::MXQuantMode mxQuantMode,
          bool transposedMm1>
__global__ __aicore__ void BsaInferInterfaceMXFP4FullQuant(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockSparseMask, GM_ADDR attenMask, GM_ADDR blockTable,
    GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKv, GM_ADDR qDequantScale, GM_ADDR kDequantScale,
    GM_ADDR vDequantScale, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR lse, GM_ADDR tiling)
{
    using ArchTag = Arch::Ascend960DT;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InDtype;     // fp4x2_e2m1_t
    using ElementK = InDtype;     // fp4x2_e2m1_t
    using ElementV = InDtype;     // fp4x2_e2m1_t
    using ElementS = SMDtype;     // half
    using ElementP = InDtype;     // fp4x2_e2m1_t
    using ElementScale = uint8_t; // e8m0
    using ElementOTmp = REDtype;  // float
    using ElementDm = REDtype;
    using ElementO = Otype; // bfloat16_t / fp16
    using ElementGroupMax = SMDtype;
    // layout tags
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    // S is rowMajor on UB(dst)
    using LayoutS = layout::RowMajor;
    // P is actually zN on UB(src), since there is no nd2nz in MTE1
    using LayoutPDummy = layout::zN;
    using LayoutPScale = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    // OTmp is rowMajor on UB(dst)
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;
    // block mask pre-process
    using DispatchPolicyMask2Idx = Epilogue::Arch92::EpilogueBsaMask2Idx;
    using EpilogueMask2Idx =
        Epilogue::Block::BlockEpilogue<DispatchPolicyMask2Idx, ElementSparseMask, ElementSparseIdx, ElementSparseCount>;
    // 处理单个tile内Q和K的matmul
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::Arch92::MmadBsaQKMX<transposedMm1>;
    using TileCopyQK =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, layout::RowMajor, ElementK, layout::ColumnMajor, ElementS,
                                          LayoutS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                          Gemm::Tile::ScaleGranularity::PER_GROUP>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;
    // online softmax
    using DispatchPolicyOnlineSoftmax = Epilogue::Arch92::EpilogueOnlineSoftmaxBsaMX<transposedMm1, mxQuantMode>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using PScaleType = Gemm::GemmType<ElementScale, LayoutPScale>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, PScaleType>;
    // compute PScale from group max: 输入 ElementGroupMax, 输出 PScaleType
    using DispatchPolicyComputePScale = Epilogue::Arch92::EpilogueComputePScaleBsaMX<transposedMm1, mxQuantMode>;
    using EpilogueComputePScale =
        Epilogue::Block::BlockEpilogue<DispatchPolicyComputePScale, ElementGroupMax, ElementDm, PScaleType>;
    // 处理单个tile内P和Value的matmul
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::Arch92::MmadBsaPVMX<transposedMm1>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                          Gemm::Tile::ScaleGranularity::PER_GROUP>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;
    // rescale O
    using DispatchPolicyRescaleO = Epilogue::Arch92::EpilogueBsaRescaleOMX<lseMode, lseFormat, transposedMm1>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementDm,
                                                            TileCopyRescaleO, Arch::PositionL0C>;

    using BsaMXFP4FullQuantKernelArch92 =
        BsaMXFP4FullQuantKernelArch92<EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV,
                                      EpilogueRescaleO, EpilogueComputePScale, qFormat, kvFormat, transposedMm1>;

    BsaFullQuantKernelParamsArch92 params{query,
                                          key,
                                          value,
                                          blockSparseMask,
                                          attenMask,
                                          blockTable,
                                          actualSeqLengths,
                                          actualSeqLengthsKv,
                                          qDequantScale,
                                          kDequantScale,
                                          vDequantScale,
                                          attentionOut,
                                          workspace,
                                          lse,
                                          tiling};
    BsaMXFP4FullQuantKernelArch92 bsaMXFP4FullQuantKernelArch92;
    bsaMXFP4FullQuantKernelArch92(params);
}

template <class InDtype, class SMDtype, class REDtype, class Otype, Format qFormat, Format kvFormat,
          Epilogue::LseMode lseMode, Epilogue::LseFormat lseFormat, Epilogue::MXQuantMode mxQuantMode,
          bool transposedMm1>
__global__ __aicore__ void BsaInferInterfaceMXFP8FullQuant(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockSparseMask, GM_ADDR attenMask, GM_ADDR blockTable,
    GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKv, GM_ADDR qDequantScale, GM_ADDR kDequantScale,
    GM_ADDR vDequantScale, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR lse, GM_ADDR tiling)
{
    using ArchTag = Arch::Ascend960DT;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InDtype; // fp8_e4m3fn_t
    using ElementK = InDtype;
    using ElementV = InDtype;
    using ElementS = SMDtype; // half
    using ElementP = InDtype;
    using ElementScale = uint8_t; // e8m0
    using ElementOTmp = REDtype;  // float
    using ElementDm = REDtype;
    using ElementO = Otype;
    using ElementGroupMax = SMDtype;
    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutPDummy = layout::zN;
    using LayoutPScale = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;
    using DispatchPolicyMask2Idx = Epilogue::Arch92::EpilogueBsaMask2Idx;
    using EpilogueMask2Idx =
        Epilogue::Block::BlockEpilogue<DispatchPolicyMask2Idx, ElementSparseMask, ElementSparseIdx, ElementSparseCount>;
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::Arch92::MmadBsaQKMxfp8<transposedMm1>;
    using TileCopyQK =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, layout::RowMajor, ElementK, layout::ColumnMajor, ElementS,
                                          LayoutS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                          Gemm::Tile::ScaleGranularity::PER_GROUP>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;
    using DispatchPolicyOnlineSoftmax = Epilogue::Arch92::EpilogueOnlineSoftmaxBsaMxfp8<transposedMm1, mxQuantMode>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using PScaleType = Gemm::GemmType<ElementScale, LayoutPScale>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, PScaleType>;
    using DispatchPolicyComputePScale = Epilogue::Arch92::EpilogueComputePScaleBsaMxfp8<transposedMm1, mxQuantMode>;
    using EpilogueComputePScale =
        Epilogue::Block::BlockEpilogue<DispatchPolicyComputePScale, ElementGroupMax, ElementDm, PScaleType>;
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using DispatchPolicyPV = Gemm::Arch92::MmadBsaPVMxfp8<transposedMm1>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                          Gemm::Tile::ScaleGranularity::PER_GROUP>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;
    using DispatchPolicyRescaleO = Epilogue::Arch92::EpilogueBsaRescaleOMxfp8<lseMode, lseFormat, transposedMm1>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementDm,
                                                            TileCopyRescaleO, Arch::PositionL0C>;

    using BsaMXFP8FullQuantKernelArch92 =
        BsaMXFP8FullQuantKernelArch92<EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV,
                                      EpilogueRescaleO, EpilogueComputePScale, qFormat, kvFormat, transposedMm1>;

    BsaFullQuantKernelParamsArch92 params{query,
                                          key,
                                          value,
                                          blockSparseMask,
                                          attenMask,
                                          blockTable,
                                          actualSeqLengths,
                                          actualSeqLengthsKv,
                                          qDequantScale,
                                          kDequantScale,
                                          vDequantScale,
                                          attentionOut,
                                          workspace,
                                          lse,
                                          tiling};
    BsaMXFP8FullQuantKernelArch92 bsaMXFP8FullQuantKernelArch92;
    bsaMXFP8FullQuantKernelArch92(params);
}

template <class InDtype, class SMDtype, class REDtype, class Otype, Format qFormat, Format kvFormat,
          Epilogue::LseMode lseMode, Epilogue::LseFormat lseFormat, bool transposedMm1>
__global__ __aicore__ void BsaInferInterfaceHIF8FullQuant(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR blockSparseMask, GM_ADDR attenMask, GM_ADDR blockTable,
    GM_ADDR actualSeqLengths, GM_ADDR actualSeqLengthsKv, GM_ADDR qDequantScale, GM_ADDR kDequantScale,
    GM_ADDR vDequantScale, GM_ADDR pQuantScale, GM_ADDR attentionOut, GM_ADDR workspace, GM_ADDR lse, GM_ADDR tiling)
{
    static_assert(transposedMm1, "HIF8 full quantization supports only the DN K * Q^T MM1 path");

    using ArchTag = Arch::Ascend960DT;
    using ElementSparseMask = uint8_t;
    using ElementSparseIdx = int32_t;
    using ElementSparseCount = int32_t;
    using ElementQ = InDtype;    // hifloat8_t
    using ElementK = InDtype;    // hifloat8_t
    using ElementV = InDtype;    // hifloat8_t
    using ElementS = SMDtype;    // float
    using ElementP = InDtype;    // hifloat8_t
    using ElementOTmp = REDtype; // float
    using ElementO = Otype;      // half / bfloat16_t

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    // K * Q^T produces transposed S. The existing high-precision softmax writes
    // P as nZ for the transposed PV path.
    using LayoutPDummy = layout::nZ;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;

    using DispatchPolicyMask2Idx = Epilogue::Arch92::EpilogueBsaMask2Idx;
    using EpilogueMask2Idx =
        Epilogue::Block::BlockEpilogue<DispatchPolicyMask2Idx, ElementSparseMask, ElementSparseIdx, ElementSparseCount>;

    // HIF8 MM1: K[256, 128] * Q^T[128, 128] -> S^T[256, 128].
    using L1TileShapeQK = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapeQK = Shape<Int<256>, Int<128>, Int<128>>;
    using DispatchPolicyQK = Gemm::Arch92::MmadBsaQKFp8<true>;
    // A6 sends the full tile to one AIV. The vector epilogues perform HIF8 dequantization.
    using TileCopyQK = Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementQ, LayoutQ, ElementK, LayoutK, ElementS,
                                                         LayoutS, void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                                         Gemm::Tile::ScaleGranularity::NO_QUANT>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, ElementQ, ElementK,
                                                  ElementS, void, TileCopyQK>;

    // Online softmax: high-precision transposed-S block with HIF8 P scaling,
    // four-lane Cast packing and QFA DN_INDEX gather.
    using DispatchPolicyOnlineSoftmax = Epilogue::Arch92::EpilogueOnlineSoftmaxBsa<true>;
    using PType = Gemm::GemmType<ElementP, LayoutPDummy>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType>;

    // HIF8 MM2: P[128, 256] * V[256, 128] -> OTmp[128, 128].
    using L1TileShapePV = Shape<Int<128>, Int<128>, Int<128>>;
    using L0TileShapePV = Shape<Int<128>, Int<128>, Int<256>>;
    using DispatchPolicyPV = Gemm::Arch92::MmadBsaPVFp8<true>;
    using TileCopyPV =
        Gemm::Tile::PackedTileCopyTlaToUB<ArchTag, ElementP, LayoutPDummy, ElementV, LayoutV, ElementOTmp, LayoutOTmp,
                                          void, Gemm::Tile::CopyL0CToUBMode::NO_SPLIT, false,
                                          Gemm::Tile::ScaleGranularity::NO_QUANT>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, ElementP, ElementV,
                                                  ElementOTmp, void, TileCopyPV>;

    // RescaleO: high-precision output rescale; HIF8 LSE subtracts ln(pScale).
    using DispatchPolicyRescaleO = Epilogue::Arch92::EpilogueBsaRescaleO<lseMode, lseFormat, true, false>;
    using TileCopyRescaleO = Epilogue::Tile::TileCopyRescaleO<ArchTag, ElementO, LayoutO, LayoutOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, ElementO, ElementOTmp, ElementS,
                                                            ElementV, TileCopyRescaleO, Arch::PositionL0C>;

    using BsaHif8Kernel =
        BsaHif8FullQuantKernelArch92<EpilogueMask2Idx, BlockMmadQK, EpilogueOnlineSoftmax, BlockMmadPV,
                                     EpilogueRescaleO, qFormat, kvFormat, transposedMm1, false>;
    BsaFullQuantKernelParamsArch92 params{query,
                                          key,
                                          value,
                                          blockSparseMask,
                                          attenMask,
                                          blockTable,
                                          actualSeqLengths,
                                          actualSeqLengthsKv,
                                          qDequantScale,
                                          kDequantScale,
                                          vDequantScale,
                                          attentionOut,
                                          workspace,
                                          lse,
                                          tiling};
    BsaHif8Kernel kernel;
    kernel(params, pQuantScale);
}
