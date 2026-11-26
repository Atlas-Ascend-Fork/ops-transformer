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
 * \file nsa_compress_attention_infer_proto.cpp
 * \brief
 */

#include <register/op_impl_registry.h>
#include <vector>
#include "log/log.h"
#include "platform/platform_info.h"

using namespace ge;

namespace ops {

constexpr uint32_t QUERY_INPUT_INDEX = 0;
constexpr uint32_t VALUE_INPUT_INDEX = 2;
constexpr uint32_t ATTEN_OUTPUT_INDEX = 0;
constexpr uint32_t TOPK_OUTPUT_INDEX = 1;
constexpr uint32_t KV_NUM_HEADS_ATTR_INDEX = 1;
constexpr uint32_t SELECT_BLOCK_ATTR_INDEX = 3;
constexpr uint32_t LAYOUT_ATTR_INDEX = 7;
static const uint64_t DIM_NUM_1 = 1;
static const uint64_t DIM_NUM_2 = 2;
static const uint64_t DIM_NUM_3 = 3;
static const uint64_t DIM_NUM_4 = 4;

static void ReportInferShapeFailure(const char* reason)
{
    REPORT_PREDEFINED_ERR_MSG("EZ1014", (std::vector<const char*>{"opName", "reason"}),
                              (std::vector<const char*>{"NsaCompressAttentionInfer", reason}));
}

static ge::graphStatus InferShapeNsaCompressAttentionInfer(gert::InferShapeContext* context)
{
    OP_CHECK_IF(context == nullptr, ReportInferShapeFailure("InferShape context is null."), return ge::GRAPH_FAILED);
    const gert::Shape* queryShape = context->GetInputShape(QUERY_INPUT_INDEX);
    OP_CHECK_IF(queryShape == nullptr, ReportInferShapeFailure("query input shape is null."), return GRAPH_FAILED);
    gert::Shape* outShape = context->GetOutputShape(ATTEN_OUTPUT_INDEX);
    OP_CHECK_IF(outShape == nullptr, ReportInferShapeFailure("attention output shape is null."), return GRAPH_FAILED);
    const gert::Shape* valueShape = context->GetInputShape(VALUE_INPUT_INDEX);
    OP_CHECK_IF(valueShape == nullptr, ReportInferShapeFailure("value input shape is null."), return GRAPH_FAILED);

    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, ReportInferShapeFailure("InferShape context attributes are null."),
                return GRAPH_FAILED);
    const int64_t* kvHeadNumsPtr = attrs->GetAttrPointer<int64_t>(KV_NUM_HEADS_ATTR_INDEX);
    OP_CHECK_IF(kvHeadNumsPtr == nullptr, ReportInferShapeFailure("numKeyValueHeads attribute is null."),
                return GRAPH_FAILED);
    const int64_t* selectBlockCountPtr = attrs->GetAttrPointer<int64_t>(SELECT_BLOCK_ATTR_INDEX);
    OP_CHECK_IF(selectBlockCountPtr == nullptr, ReportInferShapeFailure("selectBlockCount attribute is null."),
                return GRAPH_FAILED);
    const char* layOutPtr = attrs->GetAttrPointer<char>(LAYOUT_ATTR_INDEX);
    OP_CHECK_IF(layOutPtr == nullptr, ReportInferShapeFailure("InferShape context layoutOptional is null."),
                return GRAPH_FAILED);

    const int64_t kvHeadNums = *kvHeadNumsPtr;
    const int64_t selectBlockCount = *selectBlockCountPtr;
    std::string layOutStr = std::string(layOutPtr);
    if (layOutStr != "TND" && layOutStr != "BSND") {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("NsaCompressAttentionInfer", "layoutOptional", layOutStr,
                                              "Only TND and BSND are supported.");
        return GRAPH_FAILED;
    }

    OP_CHECK_IF(
        kvHeadNums <= 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("NsaCompressAttentionInfer", "numKeyValueHeads",
                                              std::to_string(kvHeadNums), "numKeyValueHeads must be greater than 0."),
        return GRAPH_FAILED);
    OP_CHECK_IF(selectBlockCount <= 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("NsaCompressAttentionInfer", "selectBlockCount",
                                                      std::to_string(selectBlockCount),
                                                      "selectBlockCount must be greater than 0."),
                return GRAPH_FAILED);
    const bool isTndLayout = (layOutStr == "TND");
    const uint64_t expectedQueryDimNum = isTndLayout ? DIM_NUM_3 : DIM_NUM_4;
    OP_CHECK_IF(queryShape->GetDimNum() != expectedQueryDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                    "NsaCompressAttentionInfer", "query", std::to_string(queryShape->GetDimNum()),
                    layOutStr + " layout requires " + std::to_string(expectedQueryDimNum) + " dimensions."),
                return GRAPH_FAILED);
    OP_CHECK_IF(valueShape->GetDimNum() < DIM_NUM_3,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("NsaCompressAttentionInfer", "value",
                                                         std::to_string(valueShape->GetDimNum()),
                                                         "value must have at least 3 dimensions."),
                return GRAPH_FAILED);

    const int64_t shapeD2 = valueShape->GetDim(DIM_NUM_2);
    // 动态维（-1）不参与整除，保持未知，避免 -1 / N 被截断为 0
    const int64_t valueHeadDim = shapeD2 < 0 ? shapeD2 : shapeD2 / kvHeadNums;
    *outShape = *queryShape;

    gert::Shape* topkIndicesOutShape = context->GetOutputShape(TOPK_OUTPUT_INDEX);
    OP_CHECK_IF(topkIndicesOutShape == nullptr, ReportInferShapeFailure("topkIndicesOut shape is null."),
                return GRAPH_FAILED);
    *topkIndicesOutShape = *queryShape;

    if (isTndLayout) {
        outShape->SetDim(DIM_NUM_2, valueHeadDim);
        topkIndicesOutShape->SetDim(DIM_NUM_1, kvHeadNums);
        topkIndicesOutShape->SetDim(DIM_NUM_2, selectBlockCount);
    } else if (layOutStr == "BSND") {
        outShape->SetDim(DIM_NUM_3, valueHeadDim);
        topkIndicesOutShape->SetDim(DIM_NUM_2, kvHeadNums);
        topkIndicesOutShape->SetDim(DIM_NUM_3, selectBlockCount);
    }
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeNsaCompressAttentionInfer(gert::InferDataTypeContext* context)
{
    OP_CHECK_IF(context == nullptr, ReportInferShapeFailure("InferDataType context is null."), return ge::GRAPH_FAILED);
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DataType::DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(NsaCompressAttentionInfer)
    .InferShape(InferShapeNsaCompressAttentionInfer)
    .InferDataType(InferDataTypeNsaCompressAttentionInfer);
} // namespace ops
