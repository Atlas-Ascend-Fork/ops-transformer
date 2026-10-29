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
 * \file permute_div_tiling.cpp
 * \brief PermuteDiv tiling
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "tiling/platform/platform_ascendc.h"
#include "permute_div_tiling.h"

namespace {
constexpr int DIVISOR_ATTR_INDEX = 0;
constexpr int MAX_UB = 184 * 1024;

template <typename T1, typename T2>
inline T2 CeilDiv(const T1 a, const T2 b)
{
    return static_cast<T2>((a + static_cast<T1>(b) - 1) / static_cast<T1>(b));
}
} // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    PermuteDivTilingData tiling;

    const gert::Shape& indiceShape = context->GetInputShape(0)->GetStorageShape();

    // check dim
    if (indiceShape.GetDimNum() != 1) {
        OP_LOGE(context, "Dim of input not right");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    int divisor = *(context->GetAttrs()->GetAttrPointer<int>(DIVISOR_ATTR_INDEX));
    if (divisor < 1) {
        OP_LOGE(context, "Invalid divisor: %d", divisor);
        return ge::GRAPH_FAILED;
    }

    // check ub space
    int indiceNum = indiceShape.GetDim(0);
    auto indiceSplit = CeilDiv(indiceNum, coreNum);
    if (indiceSplit * sizeof(int32_t) > static_cast<uint32_t>(MAX_UB)) {
        OP_LOGE(context, "numIndice out of range: %d", indiceNum);
        return ge::GRAPH_FAILED;
    }
    tiling.set_divisor(divisor);
    tiling.set_coreNum(coreNum);
    tiling.set_indiceNum(indiceNum);

    // pure vector op, will launch the same size of vector cores as SetBlockDim does.
    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepareForPermuteDiv(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<PermuteDivCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    auto platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo->ubSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(PermuteDiv).Tiling(TilingFunc).TilingParse<PermuteDivCompileInfo>(TilingPrepareForPermuteDiv);

} // namespace optiling
