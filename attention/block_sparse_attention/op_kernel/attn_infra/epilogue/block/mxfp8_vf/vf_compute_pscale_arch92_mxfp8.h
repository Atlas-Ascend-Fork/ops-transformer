/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VF_COMPUTE_PSCALE_ARCH92_MXFP8_H
#define VF_COMPUTE_PSCALE_ARCH92_MXFP8_H

#include "kernel_tensor.h"
#include "vf_common_def_mxfp8.h"
#include "../../bsa_epilogue_dispatch_policy.hpp"

namespace NpuArch::Epilogue::Block::Mxfp8VF {
using namespace AscendC;
using namespace MicroAPI;

template <bool firstKvGroup, typename T, uint16_t groupStride>
__simd_vf__ inline void ComputePScaleTileVF(__ubuf__ uint8_t* pScale, __ubuf__ T* groupMax, __ubuf__ T* tileMax,
                                            __ubuf__ T* runningMax, __ubuf__ float* dm)
{
    MaskReg preg_all16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all8 = CreateMask<uint8_t, MaskPattern::ALL>();

    RegTensor<half> vreg_gmax, vreg_max1, vreg_rs16, vreg_gmax_old;
    RegTensor<float> vreg_rs1, vreg_rs2;

    LoadAlign(vreg_gmax, tileMax);
    if constexpr (firstKvGroup) {
        StoreAlign(runningMax, vreg_gmax, preg_all16);
    } else {
        LoadAlign(vreg_gmax_old, runningMax);
        Max(vreg_gmax, vreg_gmax, vreg_gmax_old, preg_all16);
        StoreAlign(runningMax, vreg_gmax, preg_all16);

        Sub(vreg_rs16, vreg_gmax_old, vreg_gmax, preg_all16);
        Adds(vreg_rs16, vreg_rs16, NUM_127, preg_all16);
        Maxs(vreg_rs16, vreg_rs16, ZERO_VALUE, preg_all16);
        Cast<int32_t, T, h2iZero>((RegTensor<int32_t>&)vreg_rs1, vreg_rs16, preg_all16);
        Cast<int32_t, T, h2iOne>((RegTensor<int32_t>&)vreg_rs2, vreg_rs16, preg_all16);
        ShiftLefts((RegTensor<int32_t>&)vreg_rs1, (RegTensor<int32_t>&)vreg_rs1, SHIFT_VALUE, preg_all16);
        ShiftLefts((RegTensor<int32_t>&)vreg_rs2, (RegTensor<int32_t>&)vreg_rs2, SHIFT_VALUE, preg_all16);

        Interleave((RegTensor<int32_t>&)vreg_rs1, (RegTensor<int32_t>&)vreg_rs2, (RegTensor<int32_t>&)vreg_rs1,
                   (RegTensor<int32_t>&)vreg_rs2);
        StoreAlign(dm, vreg_rs1, preg_all16);
    }

    Duplicate(vreg_max1, MIN_VALUE);
    StoreAlign(tileMax, vreg_max1, preg_all16);
    Adds(vreg_gmax, vreg_gmax, NUM_NEG_119, preg_all16);

    RegTensor<half> vreg_scale1, vreg_scale2, vreg_scale3, vreg_scale4;
    LoadAlign(vreg_scale1, groupMax);
    LoadAlign(vreg_scale2, groupMax + groupStride);
    LoadAlign(vreg_scale3, groupMax + 2 * groupStride);
    LoadAlign(vreg_scale4, groupMax + 3 * groupStride);

    Sub(vreg_scale1, vreg_scale1, vreg_gmax, preg_all16);
    Sub(vreg_scale2, vreg_scale2, vreg_gmax, preg_all16);
    Sub(vreg_scale3, vreg_scale3, vreg_gmax, preg_all16);
    Sub(vreg_scale4, vreg_scale4, vreg_gmax, preg_all16);

    // SAT Mode, if < 0, then = 0
    RegTensor<uint8_t> vreg_mxscale1, vreg_mxscale2, vreg_mxscale3, vreg_mxscale4;
    Cast<uint8_t, T, castTraitZero>(vreg_mxscale1, vreg_scale1, preg_all16);
    Cast<uint8_t, T, castTraitOne>(vreg_mxscale2, vreg_scale2, preg_all16);
    Cast<uint8_t, T, castTraitZero>(vreg_mxscale3, vreg_scale3, preg_all16);
    Cast<uint8_t, T, castTraitOne>(vreg_mxscale4, vreg_scale4, preg_all16);

    Or(vreg_mxscale1, vreg_mxscale1, vreg_mxscale2, preg_all8);
    Or(vreg_mxscale3, vreg_mxscale3, vreg_mxscale4, preg_all8);
    StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(pScale, vreg_mxscale1, 3, preg_all8);
    StoreAlign<uint8_t, DataCopyMode::DATA_BLOCK_COPY>(pScale + 32, vreg_mxscale3, 3, preg_all8);
}

template <bool firstKvGroup, typename T, uint16_t groupStride>
__aicore__ inline void ComputePScaleTile(const LocalTensor<uint8_t>& pScale, const LocalTensor<T>& groupMax,
                                         const LocalTensor<T>& tileMax, const LocalTensor<T>& runningMax,
                                         const LocalTensor<float>& dm)
{
    ComputePScaleTileVF<firstKvGroup, T, groupStride>(
        (__ubuf__ uint8_t*)pScale.GetPhyAddr(), (__ubuf__ T*)groupMax.GetPhyAddr(), (__ubuf__ T*)tileMax.GetPhyAddr(),
        (__ubuf__ T*)runningMax.GetPhyAddr(), (__ubuf__ float*)dm.GetPhyAddr());
}
} // namespace NpuArch::Epilogue::Block::Mxfp8VF

#endif // VF_COMPUTE_PSCALE_ARCH92_MXFP8_H
