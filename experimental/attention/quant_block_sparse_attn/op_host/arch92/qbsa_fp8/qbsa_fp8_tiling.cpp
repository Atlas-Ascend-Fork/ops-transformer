/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include "qbsa_fp8_tiling.h"
#include "../../../op_kernel/arch92/qbsa_fp8/qbsa_fp8_tiling_key.h"
#include "../../../op_kernel/arch35/common/quant_block_sparse_attn_tiling_key_registry.h"

namespace optiling {
ge::graphStatus QbsaFp8Arch92Tiling::BuildTilingData()
{
    // Reuse validated FP8 shape/stride/mask facts without changing A5 serialization.
    if (QbsaFp8Arch35Tiling::BuildTilingData() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    constexpr uint32_t maxCoreNum = QBSA_ARCH92_CORE_SPLIT_NUM - 1U;
    usedCoreNum_ = static_cast<uint32_t>(std::min<uint64_t>(GetAicCoreNum(maxCoreNum), totalTaskNum_));
    std::memset(&arch92TilingData_, 0, sizeof(arch92TilingData_));
    arch92TilingData_.commonParams = tilingData_.commonParams;
    arch92TilingData_.commonParams.baseParams.coreNum = usedCoreNum_;
    const auto& source = tilingData_.fp8Params;
    auto& target = arch92TilingData_.fp8Params;
    target.bandIndex = source.bandIndex;
    target.seqUsedQlenSize = source.seqUsedQlenSize;
    target.seqUsedKvlenSize = source.seqUsedKvlenSize;
    target.layoutType = source.layoutType;
    target.isKvContinuous = source.isKvContinuous;
    target.fromFused = source.fromFused;
    target.isGqa = source.isGqa;
    target.s1OuterSize = source.s1OuterSize;
    target.needInit = source.needInit;
    for (uint32_t boundary = 0U; boundary <= usedCoreNum_; ++boundary) {
        const uint64_t taskOffset = totalTaskNum_ * boundary / usedCoreNum_;
        target.bnStartIdx[boundary] = static_cast<uint32_t>(taskOffset / tilingInfo_.gS1OuterSize);
    }
    // A6 uses one vector core per cube core; A5 continues to use two.
    target.singleCoreSize = static_cast<uint32_t>(
        (arch92TilingData_.commonParams.outputParams.totalOutputSize + usedCoreNum_ - 1) / usedCoreNum_);
    tilingKey_ =
        BuildQbsaFp8Arch92TilingKey(tilingInfo_.layoutQValue, tilingInfo_.maskModeVal, tilingInfo_.returnSoftmaxLseVal);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus QbsaFp8Arch92Tiling::SaveTilingData()
{
    return PublishTilingData(arch92TilingData_);
}
} // namespace optiling
