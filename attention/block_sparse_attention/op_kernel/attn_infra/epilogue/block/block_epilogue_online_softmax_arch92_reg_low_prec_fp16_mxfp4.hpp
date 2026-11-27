/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH92_REG_LOW_PREC_FP16_MXFP4_HPP
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH92_REG_LOW_PREC_FP16_MXFP4_HPP

#include "../../bsa_base_defs.hpp"
#include "../../arch/bsa_resource.hpp"
#include "../bsa_epilogue_dispatch_policy.hpp"
#include "../tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../bsa_gemm_coord.hpp"
#include "../../bsa_matrix_coord.hpp"
#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "mxfp4_vf/vf_init_nd2nz_indexes_duplicate_mxfp4.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_qs64_kvs256.h"
#include "mxfp4_vf/vf_mm1_res_pre_padding_align_kvs32_multi_qs64.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_align_qs64_kvs32_multi.h"
#include "mxfp4_vf/vf_softmax_dn_cast_nz_mxfp4_align_qs64_kvs32.h"

namespace NpuArch::Epilogue::Block {

using namespace MXFP4Kernel;

template <MXQuantMode MX_QUANT_MODE_, class OutputType_, class LayoutS_, class ScaleType_>
class BlockEpilogue<Arch92::EpilogueOnlineSoftmaxBsaMX<true, MX_QUANT_MODE_>, OutputType_,
                    Gemm::GemmType<half, LayoutS_>, ScaleType_> {
public:
    using DispatchPolicy = Arch92::EpilogueOnlineSoftmaxBsaMX<true, MX_QUANT_MODE_>;
    static constexpr MXQuantMode MX_QUANT_MODE = DispatchPolicy::MX_QUANT_MODE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementOutput = typename OutputType_::Element; // P
    using ElementInput = half;                           // S
    using ElementMax = ElementInput;
    using ElementDisguiseP = uint8_t;
    using ElementPScale = typename ScaleType_::Element;
    using ElementIndex = uint8_t;
    using LayoutPL1 = typename OutputType_::Layout;
    using LayoutPUB = layout::RowMajor;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, ElementInput NEG_LOG2_CX)
    {
        // FIXPIPE<->V 区
        for (uint32_t i = 0; i < MXFP4::UB_S_BUF_CNT; i++) {
            sUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementInput>(MXFP4::UB_S_BUF_OFFSET +
                                                                                 MXFP4::UB_S_INNER_BUF_OFFSET * i);
        }

        // 输出缓冲区
        for (uint32_t i = 0; i < MXFP4::UB_P_BUF_CNT; i++) {
            pUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDisguiseP>(MXFP4::UB_P_BUF_OFFSET +
                                                                                     MXFP4::UB_P_INNER_BUF_OFFSET * i);
        }

        // 常驻 buffer
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GROUP_MAX_CNT; i++) {
            localGroupMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_LOCAL_GROUP_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GROUP_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementMax>(
                MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        indexUBTensor = resource.ubBuf.template GetBufferByByte<ElementIndex>(MXFP4::UB_INDEX_BUF_OFFSET);

        NEG_LOG2_CX_ = NEG_LOG2_CX;
        //  Init Tensor
        Mxfp4VF::InitIndexesAndDuplicateCallVF<ElementMax>(indexUBTensor, localGlobalMaxUBTensor[0]);
    }
    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorL1P>
    __aicore__ inline void operator()(TensorL1P& l1PTensorTla, GemmCoord& actualBlockShape, TileInfo const& tileInfo)
    {
        uint32_t spBuffIdx = GetSPBufferIdx(tileInfo.loop);
        // Acquire the local max slot once per KV tile group.
        if (tileInfo.isTileGoupFirstTile) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_GMAX_UB_TO_L1_BUF0_FLAG + tileInfo.tileMaxIdx);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);

        const uint32_t actKvsTile = actualBlockShape.m();
        const uint32_t groupMaxIdx = GetLocalGroupMaxBufIdx(tileInfo.loop);
        if (actKvsTile == MXFP4::KVS_BASE_SIZE) {
            // qs=64, kvs=256 softmax
            Mxfp4VF::SoftmaxWithGroupMaxQs64Kvs256CallVF<MX_QUANT_MODE, true, ElementInput, ElementDisguiseP,
                                                         MXFP4::KVS_BASE_SIZE, MXFP4::QS_BASE_SIZE>(
                pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_);
        } else {
            // kvs padding 32 multi
            if (actKvsTile != tileInfo.kvsActBaseTileAlign32) {
                // qs=64
                Mxfp4VF::Mm1ResPrePaddingAlignKvs32MultiQs64CallVF<ElementInput>(
                    sUBTensor[spBuffIdx], static_cast<uint16_t>(actKvsTile),
                    static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32));
                AscendC::PipeBarrier<PIPE_V>();
            }
            // softmax
            if (tileInfo.kvsActBaseTileAlign32 == MXFP4::DATA_BLOCK_BYTE) {
                // softmax padding 32, qs=64
                Mxfp4VF::SoftmaxWithGroupMaxAlignQs64Kvs32CallVF<MX_QUANT_MODE, true, ElementInput, ElementDisguiseP>(
                    pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                    localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_);
            } else {
                // softmax padding 32 multi >= 64, qs=64
                Mxfp4VF::SoftmaxWithGroupMaxAlignQs64Kvs32MultiCallVF<MX_QUANT_MODE, true, ElementInput,
                                                                      ElementDisguiseP, MXFP4::QS_BASE_SIZE>(
                    pUBTensor[spBuffIdx], sUBTensor[spBuffIdx], localGroupMaxUBTensor[groupMaxIdx],
                    localGlobalMaxUBTensor[tileInfo.tileMaxIdx], indexUBTensor, NEG_LOG2_CX_,
                    static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32),
                    static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign64));
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);

        // DataCopy UB -> L1 搬运
        auto ubPLayoutTla =
            tla::MakeLayout<ElementDisguiseP, LayoutPUB>(MXFP4::KVS_BASE_SIZE / 4, MXFP4::QS_BASE_SIZE * 4);
        auto ubPTensorTla = tla::MakeTensor(pUBTensor[0], ubPLayoutTla, Arch::PositionUB{});

        CopyPUBToPL1(l1PTensorTla, ubPTensorTla, tileInfo, spBuffIdx);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG + spBuffIdx);
    }

    template <class TensorDst, class TensorSrc>
    __aicore__ inline void CopyPUBToPL1(TensorDst const& l1PTensorTla, TensorSrc const& ubPTensorTla,
                                        TileInfo const& tileInfo, uint32_t spBuffIdx)
    {
        const uint32_t rowBytes = tla::get<1>(ubPTensorTla.shape()) / 2;
        const uint16_t blocksPerRow = rowBytes / MXFP4::DATA_BLOCK_BYTE;
        AscendC::DataCopyParams params;
        params.blockCount = static_cast<uint16_t>(tileInfo.kvsActBaseTileAlign32 / blocksPerRow);
        params.blockLen = blocksPerRow;
        params.srcStride = blocksPerRow;
        params.dstStride = 0;

        // The two P slots occupy alternating row segments in UB and are contiguous in L1.
        auto tile = tla::GetTile(ubPTensorTla, tla::MakeCoord(0u, spBuffIdx * MXFP4::UB_P_INNER_BUF_ELEMENT_OFFSET),
                                 tla::MakeShape(params.blockCount, rowBytes));
        AscendC::DataCopy(l1PTensorTla.data(), tile.data()[tile.layout()(tile.coord())], params);
    }

    // buffer id 获取
    __aicore__ inline uint32_t GetSPBufferIdx(const uint32_t loop)
    {
        return loop % 2;
    }

    __aicore__ inline uint32_t GetLocalGroupMaxBufIdx(uint32_t loop)
    {
        return loop * 2 % MXFP4::UB_LOCAL_GROUP_MAX_CNT;
    }

private:
    // UB tensors（FIXPIPE<->V 及常驻 buffer）
    AscendC::LocalTensor<ElementInput> sUBTensor[MXFP4::UB_S_BUF_CNT];                       // mm1Res(S)
    AscendC::LocalTensor<ElementDisguiseP> pUBTensor[MXFP4::UB_P_BUF_CNT];                   // vec1Res(P)
    AscendC::LocalTensor<ElementMax> localGroupMaxUBTensor[MXFP4::UB_LOCAL_GROUP_MAX_CNT];   // LocalGroupMax
    AscendC::LocalTensor<ElementMax> localGlobalMaxUBTensor[MXFP4::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
    AscendC::LocalTensor<ElementIndex> indexUBTensor;                                        // Index
    ElementInput NEG_LOG2_CX_;
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_ARCH92_REG_LOW_PREC_FP16_MXFP4_HPP
