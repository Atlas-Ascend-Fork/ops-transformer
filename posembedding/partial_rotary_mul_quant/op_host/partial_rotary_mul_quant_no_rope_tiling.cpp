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
#include "partial_rotary_mul_quant_tiling.h"

namespace optiling {
namespace {
constexpr uint64_t ELEMENT_ALIGNMENT = 256;
constexpr uint64_t MIN_ELEMENTS_PER_CORE = 2048;
constexpr uint64_t TILE_ELEMENTS = 4096;
constexpr uint64_t BUFFER_COUNT = 2;
constexpr uint64_t SCALE_BUFFER_BYTES = 32;
constexpr uint64_t NO_ROPE_LAYOUT_KEY = 20050;
} // namespace

bool PartialRotaryMulQuantNoRopeTiling::IsCapable()
{
    return IsRegbaseSocVersion() && yDtype_ == 0 && noRope_;
}

ge::graphStatus PartialRotaryMulQuantNoRopeTiling::DoOpTiling()
{
    uint64_t elements = static_cast<uint64_t>(context_->GetInputShape(0)->GetStorageShape().GetShapeSize());
    OP_CHECK_IF(elements == 0 || aicoreParams_.blockDim == 0,
                OP_LOGE(context_->GetNodeName(), "Empty input or no vector cores"), return ge::GRAPH_FAILED);

    // Flatten all four dimensions. Align core boundaries for both the input
    // and byte output, while keeping tiny tensors on a small number of cores.
    uint64_t cores = std::min(aicoreParams_.blockDim, CeilDiv(elements, MIN_ELEMENTS_PER_CORE));
    uint64_t blockElements = CeilAlign(CeilDiv(elements, cores), ELEMENT_ALIGNMENT);
    blockNum_ = static_cast<uint32_t>(CeilDiv(elements, blockElements));

    uint64_t bytesPerElement = BUFFER_COUNT * (ge::GetSizeByDataType(dtype_) + sizeof(uint8_t));
    OP_CHECK_IF(aicoreParams_.ubSize <= SCALE_BUFFER_BYTES,
                OP_LOGE(context_->GetNodeName(), "Insufficient UB for scale"), return ge::GRAPH_FAILED);
    uint64_t ubElements = FloorAlign((aicoreParams_.ubSize - SCALE_BUFFER_BYTES) / bytesPerElement, ELEMENT_ALIGNMENT);
    uint64_t tileElements = std::min({TILE_ELEMENTS, blockElements, ubElements});
    OP_CHECK_IF(tileElements == 0, OP_LOGE(context_->GetNodeName(), "Insufficient UB for double buffering"),
                return ge::GRAPH_FAILED);
    tilingData_.set_totalElements(elements);
    tilingData_.set_blockElements(blockElements);
    tilingData_.set_tileElements(static_cast<uint32_t>(tileElements));
    tilingData_.set_scale(scaleAttr_);
    tilingData_.set_hasScale(hasScale_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus PartialRotaryMulQuantNoRopeTiling::GetWorkspaceSize()
{
    workspaceSize_ = 0;
    return ge::GRAPH_SUCCESS;
}

uint64_t PartialRotaryMulQuantNoRopeTiling::GetTilingKey() const
{
    return QuantTilingKey(NO_ROPE_LAYOUT_KEY);
}

ge::graphStatus PartialRotaryMulQuantNoRopeTiling::PostTiling()
{
    context_->SetBlockDim(blockNum_);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspaces is null"), return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(PartialRotaryMulQuant, PartialRotaryMulQuantNoRopeTiling, 50);
} // namespace optiling
