/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v3_tiling_a5.cpp
 * \brief
 */

#include "moe_distribute_combine_v3_tiling_a5.h"
#include <cstring>
#include "../../../../moe_distribute_combine_v2/op_host/op_tiling/arch35/moe_distribute_combine_tiling_arch35.h"
#include "op_host/op_tiling/mc2_tiling_utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "mc2_log.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "mc2_hcom_topo_info.h"
#include "mc2_exception_dump.h"
namespace optiling {
ge::graphStatus MoeDistributeCombineV3TilingFuncA5::MoeDistributeCombineV2TilingFuncNew(gert::TilingContext* context,
                                                                                        const CombineV2Config& config)
{
    MoeDistributeCombineV2TilingFuncA5 funcA5;
    auto attrs = context->GetAttrs();
    OP_TILING_CHECK(attrs == nullptr, OP_LOGE(context->GetNodeName(), "attrs is null"), return ge::GRAPH_FAILED);
    auto commAlg = attrs->GetAttrPointer<char>(config.attrCommAlgIndex);
    auto tilingConfig = config;
    if (mc2tiling::GetNpuArch(context) == Ops::Base::DAV_3510 && commAlg != nullptr &&
        std::strcmp(commAlg, "hierarchy") == 0) {
        auto ranksPerServer = attrs->GetAttrPointer<int64_t>(config.attrRankSizePerServerIndex);
        auto epWorldSize = attrs->GetAttrPointer<int64_t>(config.attrEpWorldSizeIndex);
        OP_TILING_CHECK(ranksPerServer == nullptr || epWorldSize == nullptr || *ranksPerServer <= 0 ||
                            *epWorldSize <= 0 || *epWorldSize % *ranksPerServer != 0,
                        OP_LOGE(context->GetNodeName(), "ROCE requires rank_size_per_server to divide ep_world_size"),
                        return ge::GRAPH_FAILED);
        tilingConfig.isRoce = true;
        tilingConfig.roceServerNum = static_cast<uint32_t>(*epWorldSize / *ranksPerServer);
        OP_TILING_CHECK(tilingConfig.roceServerNum <= 1U,
                        OP_LOGE(context->GetNodeName(), "ROCE requires multiple servers, got serverNum=%u",
                                tilingConfig.roceServerNum),
                        return ge::GRAPH_FAILED);
    }
    return funcA5.MoeDistributeCombineV2TilingFuncNew(context, tilingConfig);
}

struct MoeDistributeCombineV3CompileInfo {};
static ge::graphStatus TilingParseForMoeDistributeCombineV3(gert::TilingParseContext* context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus MoeDistributeCombineV3TilingFuncImplA5(gert::TilingContext* context)
{
    MoeDistributeCombineV3TilingFuncA5 impl;
    return impl.MoeDistributeCombineV3TilingFunc(context);
}

IMPL_OP_OPTILING(MoeDistributeCombineV3)
    .Tiling(MoeDistributeCombineV3TilingFuncImplA5)
    .TilingParse<MoeDistributeCombineV3CompileInfo>(TilingParseForMoeDistributeCombineV3);
} // namespace optiling
