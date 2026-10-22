/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
using namespace AscendC;
using namespace matmul;
#include "../../grouped_matmul_swiglu_quant_v2/op_kernel/arch35/grouped_matmul_tensor_api_swiglu_quant_v2_mxfp8_kernel.h"
#include "arch35/grouped_matmul_activation_tiling_key.h"
template <uint64_t QUANT_B_TRANS>
__global__ __aicore__ void grouped_matmul_activation(GM_ADDR x, GM_ADDR xScale, GM_ADDR groupList, GM_ADDR weight,
                                                     GM_ADDR weightScale, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace,
                                                     GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GroupedMatmulSwigluQuantV2TensorApi::GMMSwigluQuantV2TensorApiTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    (void)bias;
    constexpr uint32_t overflowCtrl = 60;
    int64_t overflowMode = AscendC::GetCtrlSpr<overflowCtrl, overflowCtrl>();
    AscendC::SetCtrlSpr<overflowCtrl, overflowCtrl>(0);
    if constexpr (QUANT_B_TRANS == 0) {
        GmmTensorApiSwigluQuantMxFp8Kernel<asc::te::nd_ext_layout_ptn, asc::te::nz_layout_ptn, true>(
            x, weight, weightScale, xScale, nullptr, nullptr, groupList, y, nullptr, workspace, tiling);
    } else {
        GmmTensorApiSwigluQuantMxFp8Kernel<asc::te::nd_ext_layout_ptn, asc::te::zn_layout_ptn, true>(
            x, weight, weightScale, xScale, nullptr, nullptr, groupList, y, nullptr, workspace, tiling);
    }
    AscendC::SetCtrlSpr<overflowCtrl, overflowCtrl>(overflowMode);
}
