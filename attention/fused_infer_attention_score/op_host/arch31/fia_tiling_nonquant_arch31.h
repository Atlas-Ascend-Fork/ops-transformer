/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file fia_tiling_nonquant_arch31.h
 * \brief arch31 (KirinX90 / Kirin9030) non-quant Tiling 类声明
 */
#ifndef FIA_TILING_NONQUANT_ARCH31_H
#define FIA_TILING_NONQUANT_ARCH31_H

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "../fused_infer_attention_score_tiling.h"

namespace optiling {
class FiaTilingNonQuantArch31 {
public:
    explicit FiaTilingNonQuantArch31(gert::TilingContext* context)
        : context_(context)
    {}
    ~FiaTilingNonQuantArch31() = default;

    ge::graphStatus DoTiling();

private:
    ge::graphStatus CalcTilingParams(uint32_t& mBase, uint32_t& nBase, uint32_t& dInner, bool& needSplitD,
                                     uint32_t& dTile, uint32_t& kTileQK, uint32_t& nTileV, uint32_t& mBaseDeal,
                                     uint32_t& kTileV);
    uint64_t CalcTilingKey(uint32_t sparseMode, uint32_t layout, bool attenMaskTensorFlag, uint32_t hpCode);
    ge::graphStatus CalcWorkspaceSize(uint64_t& workspaceSize, uint32_t mBase, uint32_t nBase, uint32_t dInner);
    gert::TilingContext* context_;
    FusedInferAttentionScoreTilingData tilingData_;
};
} // namespace optiling

#endif // FIA_TILING_NONQUANT_ARCH31_H
