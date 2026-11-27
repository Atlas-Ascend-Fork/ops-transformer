/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VF_COMPUTE_PSCALE_ARCH92_MXFP4_H
#define VF_COMPUTE_PSCALE_ARCH92_MXFP4_H

#include "kernel_tensor.h"
#include "vf_common_def.h"
#include "../../bsa_epilogue_dispatch_policy.hpp"

namespace NpuArch::Epilogue::Block::Mxfp4VF {
using namespace AscendC;
using namespace Reg;

template <MXQuantMode MX_QUANT_MODE, bool firstKvGroup, typename T, uint16_t groupStride>
__simd_vf__ inline void ComputePScaleTileVF(__ubuf__ uint8_t* pScale, __ubuf__ T* groupMax, __ubuf__ T* tileMax,
                                            __ubuf__ T* runningMax, __ubuf__ float* dm, const T log2CxCeil)
{
    MaskReg preg_all_32 = CreateMask<uint32_t, MaskPattern::ALL>();
    MaskReg preg_VL32_32 = CreateMask<uint32_t, MaskPattern::VL32>();
    MaskReg preg_all16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_VL64_16 = CreateMask<uint16_t, MaskPattern::VL64>();
    MaskReg preg_VL128_8 = CreateMask<uint8_t, MaskPattern::VL128>();

    RegTensor<half> vreg_gmax, vreg_max1, vreg_rs16, vreg_gmax_old;
    RegTensor<float> vreg_rs1, vreg_rs2;

    LoadAlign(vreg_gmax, tileMax);
    if constexpr (firstKvGroup) {
        StoreAlign(runningMax, vreg_gmax, preg_VL64_16);
    } else {
        LoadAlign(vreg_gmax_old, runningMax);
        Max(vreg_gmax, vreg_gmax, vreg_gmax_old, preg_VL64_16);
        StoreAlign(runningMax, vreg_gmax, preg_VL64_16);

        Sub(vreg_rs16, vreg_gmax_old, vreg_gmax, preg_VL64_16);
        Adds(vreg_rs16, vreg_rs16, NUM_127, preg_VL64_16);
        Maxs(vreg_rs16, vreg_rs16, ZERO_VALUE, preg_VL64_16);
        Cast<int32_t, T, h2iZero>((RegTensor<int32_t>&)vreg_rs1, vreg_rs16, preg_VL64_16);
        Cast<int32_t, T, h2iOne>((RegTensor<int32_t>&)vreg_rs2, vreg_rs16, preg_VL64_16);
        ShiftLefts((RegTensor<int32_t>&)vreg_rs1, (RegTensor<int32_t>&)vreg_rs1, SHIFT_VALUE, preg_VL32_32);
        ShiftLefts((RegTensor<int32_t>&)vreg_rs2, (RegTensor<int32_t>&)vreg_rs2, SHIFT_VALUE, preg_VL32_32);

        Interleave((RegTensor<int32_t>&)vreg_rs1, (RegTensor<int32_t>&)vreg_rs2, (RegTensor<int32_t>&)vreg_rs1,
                   (RegTensor<int32_t>&)vreg_rs2);
        StoreAlign(dm, vreg_rs1, preg_all_32);
    }

    Duplicate(vreg_max1, MIN_VALUE);
    StoreAlign(tileMax, vreg_max1, preg_all16);
    if constexpr (MX_QUANT_MODE == MXQuantMode::OCP) {
        Adds(vreg_gmax, vreg_gmax, NUM_NEG_125, preg_VL64_16);
    } else {
        Adds(vreg_gmax, vreg_gmax, NUM_NEG_127, preg_VL64_16);
        Adds(vreg_gmax, vreg_gmax, log2CxCeil, preg_VL64_16);
    }

    RegTensor<half> vreg_scale1, vreg_scale2, vreg_scale3, vreg_scale4, vreg_scale5, vreg_scale6, vreg_scale7,
        vreg_scale8;
    LoadAlign(vreg_scale1, groupMax);
    LoadAlign(vreg_scale2, groupMax + groupStride);
    LoadAlign(vreg_scale3, groupMax + 2 * groupStride);
    LoadAlign(vreg_scale4, groupMax + 3 * groupStride);
    LoadAlign(vreg_scale5, groupMax + 4 * groupStride);
    LoadAlign(vreg_scale6, groupMax + 5 * groupStride);
    LoadAlign(vreg_scale7, groupMax + 6 * groupStride);
    LoadAlign(vreg_scale8, groupMax + 7 * groupStride);

    Sub(vreg_scale1, vreg_scale1, vreg_gmax, preg_VL64_16);
    Sub(vreg_scale2, vreg_scale2, vreg_gmax, preg_VL64_16);
    Sub(vreg_scale3, vreg_scale3, vreg_gmax, preg_VL64_16);
    Sub(vreg_scale4, vreg_scale4, vreg_gmax, preg_VL64_16);
    Sub(vreg_scale5, vreg_scale5, vreg_gmax, preg_VL64_16);
    Sub(vreg_scale6, vreg_scale6, vreg_gmax, preg_VL64_16);
    Sub(vreg_scale7, vreg_scale7, vreg_gmax, preg_VL64_16);
    Sub(vreg_scale8, vreg_scale8, vreg_gmax, preg_VL64_16);

    // SAT Mode, if < 0, then = 0
    RegTensor<uint8_t> vreg_mxscale1, vreg_mxscale2, vreg_mxscale3, vreg_mxscale4, vreg_mxscale5, vreg_mxscale6,
        vreg_mxscale7, vreg_mxscale8;
    Cast<uint8_t, T, castTraitZero>(vreg_mxscale1, vreg_scale1, preg_VL64_16);
    Cast<uint8_t, T, castTraitOne>(vreg_mxscale2, vreg_scale2, preg_VL64_16);
    Cast<uint8_t, T, castTraitZero>(vreg_mxscale3, vreg_scale3, preg_VL64_16);
    Cast<uint8_t, T, castTraitOne>(vreg_mxscale4, vreg_scale4, preg_VL64_16);
    Cast<uint8_t, T, castTraitZero>(vreg_mxscale5, vreg_scale5, preg_VL64_16);
    Cast<uint8_t, T, castTraitOne>(vreg_mxscale6, vreg_scale6, preg_VL64_16);
    Cast<uint8_t, T, castTraitZero>(vreg_mxscale7, vreg_scale7, preg_VL64_16);
    Cast<uint8_t, T, castTraitOne>(vreg_mxscale8, vreg_scale8, preg_VL64_16);

    Or(vreg_mxscale1, vreg_mxscale1, vreg_mxscale2, preg_VL128_8);
    Or(vreg_mxscale3, vreg_mxscale3, vreg_mxscale4, preg_VL128_8);
    Or(vreg_mxscale5, vreg_mxscale5, vreg_mxscale6, preg_VL128_8);
    Or(vreg_mxscale7, vreg_mxscale7, vreg_mxscale8, preg_VL128_8);
    StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(pScale, vreg_mxscale1, 5, preg_VL128_8);
    StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(pScale + 32, vreg_mxscale3, 5, preg_VL128_8);
    StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(pScale + 32 * 2, vreg_mxscale5, 5, preg_VL128_8);
    StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(pScale + 32 * 3, vreg_mxscale7, 5, preg_VL128_8);
}

template <MXQuantMode MX_QUANT_MODE, bool firstKvGroup, typename T, uint16_t groupStride>
__aicore__ inline void ComputePScaleTile(const LocalTensor<uint8_t>& pScale, const LocalTensor<T>& groupMax,
                                         const LocalTensor<T>& tileMax, const LocalTensor<T>& runningMax,
                                         const LocalTensor<float>& dm, const T log2CxCeil)
{
    ComputePScaleTileVF<MX_QUANT_MODE, firstKvGroup, T, groupStride>(
        (__ubuf__ uint8_t*)pScale.GetPhyAddr(), (__ubuf__ T*)groupMax.GetPhyAddr(), (__ubuf__ T*)tileMax.GetPhyAddr(),
        (__ubuf__ T*)runningMax.GetPhyAddr(), (__ubuf__ float*)dm.GetPhyAddr(), log2CxCeil);
}
} // namespace NpuArch::Epilogue::Block::Mxfp4VF

#endif // VF_COMPUTE_PSCALE_ARCH92_MXFP4_H
