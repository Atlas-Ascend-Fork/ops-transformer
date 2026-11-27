/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH92_MXFP4
#define EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH92_MXFP4

#include "../../bsa_base_defs.hpp"
#include "../../arch/bsa_resource.hpp"
#include "../bsa_epilogue_dispatch_policy.hpp"
#include "../tile_common/bsa_epilogue_tile_copy.hpp"
#include "../../bsa_gemm_coord.hpp"
#include "../../bsa_matrix_coord.hpp"
#include "../../../tla/tensor_bsa.hpp"
#include "../../../tla/layout_bsa.hpp"
#include "block_epilogue_arch35_utils.hpp"
#include "mxfp4_vf/vf_compute_pscale_arch92_mxfp4.h"

namespace NpuArch::Epilogue::Block {

using namespace MXFP4Kernel;

template <MXQuantMode MX_QUANT_MODE_, class ElementGroupMax_, class ElementDm_, class PScaleType_>
class BlockEpilogue<Arch92::EpilogueComputePScaleBsaMX<true, MX_QUANT_MODE_>, ElementGroupMax_, ElementDm_,
                    PScaleType_> {
public:
    using DispatchPolicy = Arch92::EpilogueComputePScaleBsaMX<true, MX_QUANT_MODE_>;
    static constexpr MXQuantMode MX_QUANT_MODE = DispatchPolicy::MX_QUANT_MODE;
    using ArchTag = typename DispatchPolicy::ArchTag;
    using ElementGroupMax = ElementGroupMax_;
    using PScaleType = PScaleType_;
    using ElementPScale = typename PScaleType::Element;
    using LayoutPScale = typename PScaleType::Layout;
    using ElementDm = ElementDm_;

    __aicore__ inline BlockEpilogue(Arch::Resource<ArchTag>& resource, ElementGroupMax LOG2_CX_CEIL)
    {
        pscaleUBTensor = resource.ubBuf.template GetBufferByByte<ElementPScale>(MXFP4::UB_P_SCALE_BUF_OFFSET);
        softmaxMaxUBTensor = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(MXFP4::UB_SOFTMAX_MAX_BUF_OFFSET);
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GROUP_MAX_CNT; i++) {
            localGroupMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(
                MXFP4::UB_LOCAL_GROUP_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GROUP_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_LOCAL_GLOBAL_MAX_CNT; i++) {
            localGlobalMaxUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementGroupMax>(
                MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_OFFSET + MXFP4::UB_LOCAL_GLOBAL_MAX_BUF_SIZE * i);
        }
        for (uint32_t i = 0; i < MXFP4::UB_UPDATE_SCALE_CNT; i++) {
            dmUBTensor[i] = resource.ubBuf.template GetBufferByByte<ElementDm>(MXFP4::UB_UPDATE_SCALE_BUF_OFFSET +
                                                                               MXFP4::UB_UPDATE_SCALE_BUF_SIZE * i);
        }
        LOG2_CX_CEIL_ = LOG2_CX_CEIL;
    }
    __aicore__ inline ~BlockEpilogue() {}

    template <class TensorL1Pscale>
    __aicore__ inline void operator()(TensorL1Pscale& l1PScaleTensor, TileInfo const& tileInfo)
    {
        AscendC::LocalTensor<ElementGroupMax> localGlobalMax = localGlobalMaxUBTensor[tileInfo.tileMaxIdx];
        AscendC::LocalTensor<ElementGroupMax> softmaxMaxOld = softmaxMaxUBTensor;
        AscendC::LocalTensor<ElementDm> dm = dmUBTensor[tileInfo.updateScaleIdx];

        const uint16_t groupMaxSlot = (tileInfo.loop * 2) % MXFP4::UB_LOCAL_GROUP_MAX_CNT;
        auto groupMax = localGroupMaxUBTensor[groupMaxSlot];

        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF1_FLAG);

        // MXFP4 has up to 64 valid lanes at a physical group stride of 128.
        AscendC::PipeBarrier<PIPE_V>();
        if (tileInfo.curKvsTileLoopIdx / MXFP4Kernel::TILE_GROUP_N == 0) {
            Mxfp4VF::ComputePScaleTile<MX_QUANT_MODE, true, ElementGroupMax, MXFP4::QS_BASE_SIZE>(
                pscaleUBTensor, groupMax, localGlobalMax, softmaxMaxOld, dm, LOG2_CX_CEIL_);
        } else {
            Mxfp4VF::ComputePScaleTile<MX_QUANT_MODE, false, ElementGroupMax, MXFP4::QS_BASE_SIZE>(
                pscaleUBTensor, groupMax, localGlobalMax, softmaxMaxOld, dm, LOG2_CX_CEIL_);
        }

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_P_BUF0_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_P_BUF1_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_P_BUF0_FLAG);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(MXFP4::SYNC_P_BUF1_FLAG);
        // The V->MTE3 wait above completes the max reset before softmax can reuse this slot.
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_GMAX_UB_TO_L1_BUF0_FLAG + tileInfo.tileMaxIdx);

        CopyPScaleUbToL1(l1PScaleTensor, tileInfo.loop);

        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF0_FLAG);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(MXFP4::SYNC_P_BUF1_FLAG);
    }

    template <class TensorDst>
    __aicore__ inline void CopyPScaleUbToL1(TensorDst& l1PScaleTensor, uint32_t loop)
    {
        const uint32_t width = tla::get<1>(l1PScaleTensor.shape());
        const uint32_t slot = loop % tla::get<0>(l1PScaleTensor.shape());
        auto tile = tla::GetTile(l1PScaleTensor, tla::MakeCoord(slot, 0), tla::MakeShape(1, width));
        AscendC::DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = static_cast<uint16_t>(width / 2 / MXFP4::DATA_BLOCK_BYTE);
        params.srcStride = 0;
        params.dstStride = 0;
        AscendC::DataCopy(tile.data()[tile.layout()(tile.coord())], pscaleUBTensor, params);
    }

private:
    AscendC::LocalTensor<ElementPScale> pscaleUBTensor;                                           // pScale(复用P)
    AscendC::LocalTensor<ElementGroupMax> softmaxMaxUBTensor;                                     // softmaxMax
    AscendC::LocalTensor<ElementGroupMax> localGroupMaxUBTensor[MXFP4::UB_LOCAL_GROUP_MAX_CNT];   // LocalGroupMax
    AscendC::LocalTensor<ElementGroupMax> localGlobalMaxUBTensor[MXFP4::UB_LOCAL_GLOBAL_MAX_CNT]; // LocalGlobalMax
    AscendC::LocalTensor<ElementDm> dmUBTensor[MXFP4::UB_UPDATE_SCALE_CNT];                       // updateScale
    ElementGroupMax LOG2_CX_CEIL_;
};

} // namespace NpuArch::Epilogue::Block

#endif // EPILOGUE_BLOCK_BLOCK_EPILOGUE_COMPUTE_PSCALE_ARCH92_MXFP4
