/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_init_routing_v3_tiling_base.cpp
 * \brief
 */
#include <string>
#include "moe_init_routing_v3_tiling.h"
#include "register/op_def_registry.h"
#include "log/log.h"

using Ops::Transformer::OpTiling::TilingRegistry;

namespace optiling {
static ge::graphStatus TilingForMoeInitRoutingV3(gert::TilingContext* context)
{
    return TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForMoeInitRountingV3(gert::TilingParseContext* context)
{
    OP_LOGD(context, "TilingPrepareForMoeInitRoutingV3 enter.");

    auto compileInfo = context->GetCompiledInfo<MoeInitRoutingV3CompileInfo>();
    OP_CHECK_IF(compileInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "compileInfo",
                    "Required compileInfo metadata is null; "
                    "context->GetCompiledInfo<MoeInitRoutingV3CompileInfo>() must provide a valid object."),
                return ge::GRAPH_FAILED);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_IF(platformInfo == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "platformInfo",
                    "Required platformInfo metadata is null; context->GetPlatformInfo() must provide a valid object."),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->aivNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF((compileInfo->aivNum <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "compileInfo->aivNum",
                                                      std::to_string(compileInfo->aivNum), "failed to get core num"),
                return ge::GRAPH_FAILED);

    uint64_t ubSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    compileInfo->ubSize = static_cast<int64_t>(ubSize);
    compileInfo->socVersion = ascendcPlatform.GetSocVersion();
    OP_CHECK_IF((compileInfo->ubSize <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize",
                                                      std::to_string(compileInfo->ubSize), "failed to get ub size"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MoeInitRoutingV3)
    .Tiling(TilingForMoeInitRoutingV3)
    .TilingParse<MoeInitRoutingV3CompileInfo>(TilingPrepareForMoeInitRountingV3);
} // namespace optiling
