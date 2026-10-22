/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../grouped_matmul_swiglu_quant_v2/op_host/op_tiling/arch35/grouped_matmul_swiglu_quant_v2_basic_tiling.h"
#include "register/op_impl_registry.h"
namespace optiling {
ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForGMMSwigluQuantV2(gert::TilingParseContext* context);
class GroupedMatmulActivationTiling950 : public GroupedMatmulSwigluQuantV2Tiling950 {
public:
    explicit GroupedMatmulActivationTiling950(gert::TilingContext* context)
        : GroupedMatmulSwigluQuantV2Tiling950(context, true)
    {}

protected:
    uint64_t GetTilingKey() const override
    {
        // The independent kernel has only the transpose template argument.
        return static_cast<uint64_t>(inputParams_.transB);
    }
};
static ge::graphStatus GroupedMatmulActivationTiling(gert::TilingContext* context)
{
    OP_CHECK_NULL_WITH_CONTEXT(context, context);
    auto* info = context->GetCompileInfo<GMMSwigluV2CompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, info);
    OP_CHECK_IF(info->npuArch_ != static_cast<int32_t>(NpuArch::DAV_3510),
                OP_LOGE(context->GetNodeName(), "only supports Ascend 950"), return ge::GRAPH_FAILED);
    GroupedMatmulActivationTiling950 tiling(context);
    return tiling.DoTiling();
}
IMPL_OP_OPTILING(GroupedMatmulActivation)
    .Tiling(GroupedMatmulActivationTiling)
    .TilingParse<GMMSwigluV2CompileInfo>(TilingPrepareForGMMSwigluQuantV2);
} // namespace optiling
