/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qbsa_mxfp8_tiling.h"
#include <algorithm>
#include "../../../op_kernel/arch35/common/quant_block_sparse_attn_const.h"
#include "log/log.h"

namespace optiling {
ge::graphStatus QbsaMxfp8Arch92Tiling::BuildTilingData()
{
    if (tilingInfo_.dSize != 128U || tilingInfo_.layoutQValue != static_cast<uint32_t>(QBSALayout::TND)) {
        OP_LOGE("QuantBlockSparseAttn", "960 MXFullQuantMode currently supports only TND query layout with D=128");
        return ge::GRAPH_FAILED;
    }
    if (QbsaMxfp8Arch35Tiling::BuildTilingData() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    usedCoreNum_ = static_cast<uint32_t>(std::min<uint64_t>(GetAicCoreNum(64U), totalTaskNum_));
    tilingData_.commonParams.baseParams.coreNum = usedCoreNum_;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
