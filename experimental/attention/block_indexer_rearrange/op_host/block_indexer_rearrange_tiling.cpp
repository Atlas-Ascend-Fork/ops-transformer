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
 * \file block_indexer_rearrange_tiling.cpp
 * \brief
 */

#include <string>
#include <cmath>
#include <numeric>
#include <sstream>
#include "block_indexer_rearrange_tiling.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

// tensor inputs
constexpr size_t TOPK_INPUT_INDEX = 0;
constexpr size_t SEQ_LENQ_INDEX = 1;
constexpr size_t SEQ_LENKV_INDEX = 2;

// attrs inputs
constexpr size_t CAUSAL_ATTR_INDEX = 0;
constexpr size_t BLOCK_SIZE_ATTR_INDEX = 1;

class BlockIndexerRearrangeTiling {
public:
    explicit BlockIndexerRearrangeTiling(gert::TilingContext* context)
        : context_(context)
    {}
    ~BlockIndexerRearrangeTiling() = default;
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus ProcessInput();
    ge::graphStatus DoBlockTiling();
    ge::graphStatus CalcWorkSpace();
    ge::graphStatus CalcBlockDim();
    ge::graphStatus GenTilingKey();
    ge::graphStatus FillTiling();

protected:
    // platform info
    uint32_t sysWorkspaceSize_;
    uint32_t aicNum_;
    uint32_t aivNum_;
    uint32_t coreNum_;
    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0bSize_ = 0;
    size_t l0cSize_ = 0;

    // input info
    bool causal_ = false;
    int32_t blockSize_ = 1;
    int32_t totalSeqlen_;
    int32_t headNum_;
    int32_t blockCount_;

    // split info
    uint32_t usedCoreNum_ = 0;
    uint32_t formerCoreProcessNums_ = 0;
    uint32_t remainCoreNum_ = 0;
    uint32_t workspaceSize_ = 0;

    gert::TilingContext* context_ = nullptr;
    const char* opName_ = nullptr;
    BlockIndexerRearrangeTilingData tilingData_;
};

ge::graphStatus BlockIndexerRearrangeTiling::GetPlatformInfo()
{
    if (context_->GetNodeName() == nullptr) {
        OP_LOGE("BlockIndexerRearrange", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    opName_ = context_->GetNodeName();

    OP_CHECK_IF(context_->GetPlatformInfo() == nullptr, OP_LOGE(opName_, "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    sysWorkspaceSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aicNum_ = ascendcPlatform.GetCoreNumAic();
    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    coreNum_ = aivNum_; // default aiv num
    OP_LOGD(opName_, "BlockIndexerRearrange, total vector core num is %u", coreNum_);
    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0, OP_LOGE(opName_, "num of core obtained is 0."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockIndexerRearrangeTiling::ProcessInput()
{
    auto topkShape = context_->GetInputShape(TOPK_INPUT_INDEX);
    auto topkDesc = context_->GetInputDesc(TOPK_INPUT_INDEX);
    auto seqlenQShape = context_->GetInputShape(SEQ_LENQ_INDEX);
    auto seqlenQDesc = context_->GetInputDesc(SEQ_LENQ_INDEX);
    auto seqlenKvShape = context_->GetInputShape(SEQ_LENKV_INDEX);
    auto seqlenKvDesc = context_->GetInputDesc(SEQ_LENKV_INDEX);

    OP_CHECK_IF(topkShape == nullptr, OP_LOGE(opName_, "Shape of tensor topk is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(topkDesc == nullptr, OP_LOGE(opName_, "Desc of tensor topk is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqlenQShape == nullptr, OP_LOGE(opName_, "Shape of tensor seqlenQ is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqlenQDesc == nullptr, OP_LOGE(opName_, "Desc of tensor seqlenQ is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqlenKvShape == nullptr, OP_LOGE(opName_, "Shape of tensor seqlenKV is nullptr"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(seqlenKvDesc == nullptr, OP_LOGE(opName_, "Desc of tensor seqlenKV is nullptr"),
                return ge::GRAPH_FAILED);

    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(opName_, "attrs got from TilingContext is nullptr"), return ge::GRAPH_FAILED);
    auto causalPtr = attrs->GetAttrPointer<bool>(CAUSAL_ATTR_INDEX);
    auto blockSizePtr = attrs->GetAttrPointer<int>(BLOCK_SIZE_ATTR_INDEX);
    OP_CHECK_IF(causalPtr == nullptr, OP_LOGE(opName_, "Attr causal is nullptr"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockSizePtr == nullptr, OP_LOGE(opName_, "Attr block_size is nullptr"), return ge::GRAPH_FAILED);

    causal_ = *causalPtr;
    blockSize_ = *blockSizePtr;

    // only support varlen now
    totalSeqlen_ = topkShape->GetStorageShape().GetDim(0);
    headNum_ = topkShape->GetStorageShape().GetDim(1);
    blockCount_ = topkShape->GetStorageShape().GetDim(2);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockIndexerRearrangeTiling::DoBlockTiling()
{
    int32_t totalBlocks = totalSeqlen_;
    int32_t aivNum = static_cast<int32_t>(aivNum_);
    usedCoreNum_ = static_cast<uint32_t>(totalBlocks < aivNum ? totalBlocks : aivNum);

    formerCoreProcessNums_ = (totalBlocks + aivNum_ - 1) / aivNum_;
    remainCoreNum_ = formerCoreProcessNums_ * aivNum_ - totalBlocks;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockIndexerRearrangeTiling::CalcWorkSpace()
{
    workspaceSize_ = sysWorkspaceSize_;

    size_t* workSpaces = context_->GetWorkspaceSizes(1);
    workSpaces[0] = workspaceSize_;

    FillTiling();

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockIndexerRearrangeTiling::CalcBlockDim()
{
    context_->SetBlockDim(aivNum_);
    OP_LOGD(opName_, "BlockIndexerRearrange tiling block dim: %u", aivNum_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockIndexerRearrangeTiling::GenTilingKey()
{
    uint64_t tilingKey = 1; // only support int32 index
    OP_LOGD(opName_, "BlockIndexerRearrange tilingKey:%lu", tilingKey);
    context_->SetTilingKey(tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockIndexerRearrangeTiling::FillTiling()
{
    // fill base params
    tilingData_.baseParams.set_totalSeqlen(totalSeqlen_);
    tilingData_.baseParams.set_headNum(headNum_);
    tilingData_.baseParams.set_blockCount(blockCount_);
    tilingData_.baseParams.set_blockSize(blockSize_);
    tilingData_.baseParams.set_causal(causal_);

    OP_LOGD(opName_, "totalSeqlen is: %ld", tilingData_.baseParams.get_totalSeqlen());
    OP_LOGD(opName_, "headNum is: %ld", tilingData_.baseParams.get_headNum());
    OP_LOGD(opName_, "blockCount is: %ld", tilingData_.baseParams.get_blockCount());
    OP_LOGD(opName_, "blockSize is: %ld", tilingData_.baseParams.get_blockSize());
    OP_LOGD(opName_, "causal is: %ld", tilingData_.baseParams.get_causal());

    // fill tiling params
    tilingData_.coreParams.set_usedCoreNum(usedCoreNum_);
    tilingData_.coreParams.set_formerCoreNum(aivNum_ - remainCoreNum_);
    tilingData_.coreParams.set_formerCoreProcessNums(formerCoreProcessNums_);
    tilingData_.coreParams.set_remainCoreProcessNums(formerCoreProcessNums_ - 1);

    OP_LOGD(opName_, "usedCoreNum is: %u", tilingData_.coreParams.get_usedCoreNum());
    OP_LOGD(opName_, "formerCoreNum is: %d", tilingData_.coreParams.get_formerCoreNum());
    OP_LOGD(opName_, "formerCoreProcessNums is: %u", tilingData_.coreParams.get_formerCoreProcessNums());
    OP_LOGD(opName_, "remainCoreProcessNums is: %d", tilingData_.coreParams.get_remainCoreProcessNums());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockIndexerRearrangeTiling::RunTiling()
{
    if (GetPlatformInfo() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ProcessInput() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (DoBlockTiling() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CalcWorkSpace() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CalcBlockDim() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (GenTilingKey() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                OP_LOGE(opName_, "RawTilingData got from ge context is nullptr."), return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BlockIndexerRearrangeTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("BlockIndexerRearrange", "Context is nullptr."), return ge::GRAPH_FAILED);
    BlockIndexerRearrangeTiling tilingFunc(context);
    return tilingFunc.RunTiling();
}

IMPL_OP_OPTILING(BlockIndexerRearrange).Tiling(BlockIndexerRearrangeTilingFunc);

} // namespace optiling
