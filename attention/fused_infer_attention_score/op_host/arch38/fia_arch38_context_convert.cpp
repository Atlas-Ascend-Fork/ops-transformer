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
 * \file fia_arch38_context_convert.cpp
 * \brief PFA/IFA 上下文转换与 tensorlist 校验
 */

#include "fia_arch38_context_convert.h"
#include "fia_arch38_tiling_utils.h"
#include "fia_arch38_prompt_tiling.h"
#include "../fused_infer_attention_score_tiling_index.h"

using namespace ge;
using namespace AscendC;
namespace optiling {
namespace arch38 {

static bool CheckEmptyTensorList(FiaArch38ContextParams& contextKeyParams, int64_t validBatchOfK)
{
    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetShapeSize() != 0) {
            return false;
        }
        if (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetShapeSize() != 0) {
            return false;
        }
    }
    contextKeyParams.emptyTensor = 1;
    return true;
}

static bool GetKeyRopeSeqLenAndCheckBatch(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams,
                                          int64_t validBatchOfK, int64_t keyRopeSDimIdx, int64_t& keyRopeS)
{
    keyRopeS = 0;
    if (contextKeyParams.keyRopeInputShape != nullptr) {
        keyRopeS = contextKeyParams.keyRopeInputShape->GetStorageShape().GetDim(keyRopeSDimIdx);
        OP_CHECK_IF(contextKeyParams.keyRopeInputShape->GetStorageShape().GetDim(0) != validBatchOfK,
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        context->GetNodeName(), "key,key_rope",
                        std::to_string(validBatchOfK) + ", " +
                            std::to_string(contextKeyParams.keyRopeInputShape->GetStorageShape().GetDim(0)),
                        "Batch of Key does NOT equal to Batch of KeyRope under tensorlist mode!"),
                    return false);
    }
    return true;
}

static bool CheckNormalTensorListBSH(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams,
                                     int64_t validBatchOfK)
{
    auto standardKH = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(KV_DIM_2);
    auto standardVH = contextKeyParams.vTensorList[0]->GetStorageShape().GetDim(KV_DIM_2);
    int64_t keyRopeS = 0;
    if (!GetKeyRopeSeqLenAndCheckBatch(context, contextKeyParams, validBatchOfK, 1, keyRopeS)) {
        return false;
    }

    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) != standardKH) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "key",
                std::to_string(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2)) +
                    " (first batch: " + std::to_string(standardKH) + ")",
                "H of Key in the current batch is different from H of Key in the first batch under tensorlist mode!");
            return false;
        }
        if (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) != standardVH) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "value",
                std::to_string(contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2)) +
                    " (first batch: " + std::to_string(standardVH) + ")",
                "H of Value in the current batch is different from H of Value in the first batch under tensorlist "
                "mode!");
            return false;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) !=
            contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "key,value",
                std::to_string(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)) + ", " +
                    std::to_string(contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)),
                "S for Key and Value does NOT equal!");
            return false;
        }
        if (contextKeyParams.keyRopeInputShape != nullptr) {
            if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) != keyRopeS) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "key,key_rope",
                    std::to_string(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)) + ", " +
                        std::to_string(keyRopeS),
                    "S for Key and keyRope does NOT equal but they should!");
                return false;
            }
        }
        contextKeyParams.maxKVs =
            std::max(contextKeyParams.maxKVs,
                     static_cast<uint32_t>(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)));
    }
    return true;
}

static bool CheckNormalTensorListBNSD(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams,
                                      int64_t validBatchOfK)
{
    auto standardN = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(1);
    auto standardKD = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(KV_DIM_3);
    auto standardVD = contextKeyParams.vTensorList[0]->GetStorageShape().GetDim(KV_DIM_3);
    int64_t tmpNKv =
        (*contextKeyParams.numKeyValueHeads != 0) ? *contextKeyParams.numKeyValueHeads : *contextKeyParams.headsNumber;
    int64_t keyRopeS = 0;
    if (!GetKeyRopeSeqLenAndCheckBatch(context, contextKeyParams, validBatchOfK, SHAPE_INDEX_TWO, keyRopeS)) {
        return false;
    }

    if (tmpNKv != standardN) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "key,num_key_value_heads",
                                               std::to_string(standardN) + ", " + std::to_string(tmpNKv),
                                               "N of Key in the first batch is different from num_key_value_heads!");
        return false;
    }

    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
        if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) != standardN) ||
            (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1) != standardN)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "key,value,num_key_value_heads",
                std::to_string(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)) + ", " +
                    std::to_string(contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)) + ", " +
                    std::to_string(standardN),
                "N of Key and Value in the current batch is different from num_key_value_heads under tensorlist mode!");
            return false;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3) != standardKD) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "key",
                std::to_string(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3)) +
                    " (first batch: " + std::to_string(standardKD) + ")",
                "D of Key in the current batch is different from D of Key in the first batch under tensorlist mode!");
            return false;
        }
        if (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3) != standardVD) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "value",
                std::to_string(contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3)) +
                    " (first batch: " + std::to_string(standardVD) + ")",
                "D of Value in the current batch is different from D of Value in the first batch under tensorlist "
                "mode!");
            return false;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) !=
            contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "key,value",
                std::to_string(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2)) + ", " +
                    std::to_string(contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2)),
                "S from Key and Value does NOT equal but they should!");
            return false;
        }
        if (contextKeyParams.keyRopeInputShape != nullptr) {
            if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) != keyRopeS) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "key,key_rope",
                    std::to_string(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2)) + ", " +
                        std::to_string(keyRopeS),
                    "S for Key and keyRope does NOT equal but they should!");
                return false;
            }
        }
        contextKeyParams.maxKVs =
            std::max(contextKeyParams.maxKVs,
                     static_cast<uint32_t>(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2)));
    }
    return true;
}

static bool CheckNormalTensorListBSND(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams,
                                      int64_t validBatchOfK)
{
    auto standardN = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(KV_DIM_2);
    auto standardKD = contextKeyParams.kTensorList[0]->GetStorageShape().GetDim(KV_DIM_3);
    auto standardVD = contextKeyParams.vTensorList[0]->GetStorageShape().GetDim(KV_DIM_3);
    int64_t tmpNKv =
        (*contextKeyParams.numKeyValueHeads != 0) ? *contextKeyParams.numKeyValueHeads : *contextKeyParams.headsNumber;
    int64_t keyRopeS = 0;
    if (!GetKeyRopeSeqLenAndCheckBatch(context, contextKeyParams, validBatchOfK, 1, keyRopeS)) {
        return false;
    }

    if (tmpNKv != standardN) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context->GetNodeName(), "key,num_key_value_heads",
                                               std::to_string(standardN) + ", " + std::to_string(tmpNKv),
                                               "N of Key in the first batch is different from num_key_value_heads!");
        return false;
    }

    for (int64_t tmpIdx = 0; tmpIdx < validBatchOfK; ++tmpIdx) {
        if ((contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) != standardN) ||
            (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2) != standardN)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "key,value,num_key_value_heads",
                std::to_string(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2)) + ", " +
                    std::to_string(contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_2)) + ", " +
                    std::to_string(standardN),
                "N of Key and Value in the current batch is different from num_key_value_heads under tensorlist mode!");
            return false;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3) != standardKD) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "key",
                std::to_string(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3)) +
                    " (first batch: " + std::to_string(standardKD) + ")",
                "D of Key in the current batch is different from D of Key in the first batch under tensorlist mode!");
            return false;
        }
        if (contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3) != standardVD) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "value",
                std::to_string(contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(KV_DIM_3)) +
                    " (first batch: " + std::to_string(standardVD) + ")",
                "D of Value in the current batch is different from D of Value in the first batch under tensorlist "
                "mode!");
            return false;
        }
        if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) !=
            contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)) {
            OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                context->GetNodeName(), "key,value",
                std::to_string(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)) + ", " +
                    std::to_string(contextKeyParams.vTensorList[tmpIdx]->GetStorageShape().GetDim(1)),
                "S from Key and Value does NOT equal but they should!");
            return false;
        }
        if (contextKeyParams.keyRopeInputShape != nullptr) {
            if (contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1) != keyRopeS) {
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context->GetNodeName(), "key,key_rope",
                    std::to_string(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)) + ", " +
                        std::to_string(keyRopeS),
                    "S for Key and keyRope does NOT equal but they should!");
                return false;
            }
        }
        contextKeyParams.maxKVs =
            std::max(contextKeyParams.maxKVs,
                     static_cast<uint32_t>(contextKeyParams.kTensorList[tmpIdx]->GetStorageShape().GetDim(1)));
    }
    return true;
}

static bool CheckNormalTensorList(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams,
                                  const string layoutStr, int64_t validBatchOfK)
{
    bool ret = false;
    if (layoutStr == "BSH") {
        ret = CheckNormalTensorListBSH(context, contextKeyParams, validBatchOfK);
    } else if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND") {
        ret = CheckNormalTensorListBNSD(context, contextKeyParams, validBatchOfK);
    } else {
        ret = CheckNormalTensorListBSND(context, contextKeyParams, validBatchOfK);
    }
    if (ret) {
        contextKeyParams.isKvContinuous = 0;
    }
    return ret;
}

static bool CheckTensorListBatchIsOne(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams,
                                      int64_t validBatchOfK, int64_t validBatchOfV)
{
    for (int i = 0; i < validBatchOfK; i++) {
        if (contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(0) != 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "key",
                std::to_string(contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(0)),
                "Batch value of Key is NOT 1 but should be 1 under tensorlist mode!");
            return false;
        }
    }

    for (int i = 0; i < validBatchOfV; i++) {
        if (contextKeyParams.vTensorList[i]->GetStorageShape().GetDim(0) != 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                context->GetNodeName(), "value",
                std::to_string(contextKeyParams.vTensorList[i]->GetStorageShape().GetDim(0)),
                "Batch value of Value is NOT 1 but should be 1 under tensorlist mode!");
            return false;
        }
    }
    return true;
}

static bool CheckTensorList(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams,
                            const string layoutStr, int64_t batchOfQ)
{
    int64_t validBatchOfK = 0;
    int64_t validBatchOfV = 0;
    while (context->GetDynamicInputShape(KEY_INDEX, validBatchOfK) != nullptr) {
        validBatchOfK++;
    }
    while (context->GetDynamicInputShape(VALUE_INDEX, validBatchOfV) != nullptr) {
        validBatchOfV++;
    }

    OP_CHECK_IF(
        (batchOfQ != validBatchOfK) || (validBatchOfK != validBatchOfV),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "query,key,value",
            std::to_string(batchOfQ) + ", " + std::to_string(validBatchOfK) + ", " + std::to_string(validBatchOfV),
            "Batch of Query does NOT equal to Batch of Key and Value under tensorlist mode!"),
        return false);

    contextKeyParams.kTensorList.resize(batchOfQ);
    contextKeyParams.vTensorList.resize(batchOfQ);
    for (int64_t i = 0; i < validBatchOfK; i++) {
        contextKeyParams.kTensorList[i] = context->GetDynamicInputShape(KEY_INDEX, i);
    }
    for (int64_t i = 0; i < validBatchOfV; i++) {
        contextKeyParams.vTensorList[i] = context->GetDynamicInputShape(VALUE_INDEX, i);
    }

    OP_CHECK_IF(
        (batchOfQ > BLIMIT),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "query", std::to_string(batchOfQ),
                                              "Batch of Query must NOT be larger than 65535 under tensorlist mode!"),
        return false);

    if (CheckEmptyTensorList(contextKeyParams, validBatchOfK)) {
        return true;
    }

    if (!CheckTensorListBatchIsOne(context, contextKeyParams, validBatchOfK, validBatchOfV)) {
        return false;
    }

    if (!CheckNormalTensorList(context, contextKeyParams, layoutStr, validBatchOfK)) {
        return false;
    }
    return true;
}

static bool CheckKVPaddingCrossover(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams)
{
    OP_CHECK_IF(((contextKeyParams.queryPaddingSize != nullptr) &&
                 (contextKeyParams.queryPaddingSize->GetStorageShape().GetShapeSize() != 1 ||
                  contextKeyParams.queryPaddingSize->GetStorageShape().GetDimNum() != 1)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "query_padding_size",
                    std::to_string(contextKeyParams.queryPaddingSize->GetStorageShape().GetShapeSize()) + " dim " +
                        std::to_string(contextKeyParams.queryPaddingSize->GetStorageShape().GetDimNum()),
                    "Query PaddingSize input is invalid, the shape size of query paddingsize is not 1 or dim number of "
                    "query padding size is not 1!"),
                return false);
    OP_CHECK_IF(((contextKeyParams.kvPaddingSize != nullptr) &&
                 (contextKeyParams.kvPaddingSize->GetStorageShape().GetShapeSize() != 1 ||
                  contextKeyParams.kvPaddingSize->GetStorageShape().GetDimNum() != 1)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    context->GetNodeName(), "kv_padding_size",
                    std::to_string(contextKeyParams.kvPaddingSize->GetStorageShape().GetShapeSize()) + " dim " +
                        std::to_string(contextKeyParams.kvPaddingSize->GetStorageShape().GetDimNum()),
                    "KV PaddingSize input is invalid, the shape size of kv padding size is not 1 or dim number of kv "
                    "padding size is not 1!"),
                return false);
    OP_CHECK_IF(((contextKeyParams.blockTable != nullptr) &&
                 ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr))),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(),
                                                         "block_table and query_padding_size and kv_padding_size",
                                                         "When page attention is used, left padding is not supported!"),
                return false);
    OP_CHECK_IF(((contextKeyParams.queryPaddingSize != nullptr) && (contextKeyParams.actualSequenceLengthQ == nullptr)),
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "actual_seq_lengths"), return false);
    OP_CHECK_IF(((contextKeyParams.kvPaddingSize != nullptr) && (contextKeyParams.actualSequenceLengthKV == nullptr)),
                OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "actual_seq_lengths_kv"), return false);
    return true;
}

static ge::graphStatus ConvertQuantOptionalInputs(const gert::TilingContext* context,
                                                  FiaArch38ContextParams& contextKeyParams)
{
    contextKeyParams.deqScale1Shape = context->GetOptionalInputShape(DEQUANT_SCALE1_INDEX);
    contextKeyParams.scale1Shape = context->GetOptionalInputShape(QUANT_SCALE1_INDEX);
    contextKeyParams.deqScale2Shape = context->GetOptionalInputShape(DEQUANT_SCALE2_INDEX);
    contextKeyParams.scale2Shape = context->GetOptionalInputShape(QUANT_SCALE2_INDEX);
    contextKeyParams.offset2Shape = context->GetOptionalInputShape(QUANT_OFFSET2_INDEX);
    contextKeyParams.antiquantScaleShape = context->GetOptionalInputShape(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffsetShape = context->GetOptionalInputShape(ANTIQUANT_OFFSET_INDEX);

    contextKeyParams.quantScale2Type = (context->GetOptionalInputDesc(QUANT_SCALE2_INDEX) != nullptr) ?
                                           context->GetOptionalInputDesc(QUANT_SCALE2_INDEX)->GetDataType() :
                                           ge::DT_FLOAT;
    contextKeyParams.quantOffset2Type = (context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX) != nullptr) ?
                                            context->GetOptionalInputDesc(QUANT_OFFSET2_INDEX)->GetDataType() :
                                            ge::DT_FLOAT;
    contextKeyParams.dequantScaleQueryShape = context->GetOptionalInputShape(DEQUANT_SCALE_QUERY_INDEX);
    contextKeyParams.keyAntiquantScaleShape = context->GetOptionalInputShape(KEY_ANTIQUANT_SCALE_INDEX);
    contextKeyParams.keyAntiquantOffsetShape = context->GetOptionalInputShape(KEY_ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.valueAntiquantScaleShape = context->GetOptionalInputShape(VALUE_ANTIQUANT_SCALE_INDEX);
    contextKeyParams.valueAntiquantOffsetShape = context->GetOptionalInputShape(VALUE_ANTIQUANT_OFFSET_INDEX);

    contextKeyParams.dequantScaleQueryType =
        (context->GetOptionalInputDesc(DEQUANT_SCALE_QUERY_INDEX) != nullptr) ?
            context->GetOptionalInputDesc(DEQUANT_SCALE_QUERY_INDEX)->GetDataType() :
            ge::DT_FLOAT;
    contextKeyParams.keyAntiquantScaleType =
        (context->GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX) != nullptr) ?
            context->GetOptionalInputDesc(KEY_ANTIQUANT_SCALE_INDEX)->GetDataType() :
            ge::DT_FLOAT16;
    contextKeyParams.valueAntiquantScaleType =
        (context->GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX) != nullptr) ?
            context->GetOptionalInputDesc(VALUE_ANTIQUANT_SCALE_INDEX)->GetDataType() :
            ge::DT_FLOAT16;

    contextKeyParams.dequantScaleQuery = context->GetOptionalInputTensor(DEQUANT_SCALE_QUERY_INDEX);
    contextKeyParams.keyAntiquantScale = context->GetOptionalInputTensor(KEY_ANTIQUANT_SCALE_INDEX);
    contextKeyParams.valueAntiquantScale = context->GetOptionalInputTensor(VALUE_ANTIQUANT_SCALE_INDEX);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertOptionalInputTensorsPFA(gert::TilingContext* context,
                                                      FiaArch38ContextParams& contextKeyParams)
{
    contextKeyParams.pseShift = context->GetOptionalInputTensor(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMask = context->GetOptionalInputTensor(ATTEN_MASK_INDEX);
    OP_CHECK_IF((contextKeyParams.attentionMask != nullptr) &&
                    (context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_BOOL) &&
                    (context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_INT8) &&
                    (context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() != ge::DT_UINT8),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    context->GetNodeName(), "atten_mask",
                    arch38::GetPfaDataTypeStr(context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType()),
                    "Invalid attention mask dtype! Only support BOOL, INT8 and UINT8"),
                return ge::GRAPH_FAILED);
    contextKeyParams.actualSequenceLengthQ = context->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
    contextKeyParams.actualSequenceLengthKV = context->GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX);
    contextKeyParams.antiquantScale = context->GetOptionalInputTensor(ANTIQUANT_SCALE_INDEX);
    contextKeyParams.antiquantOffset = context->GetOptionalInputTensor(ANTIQUANT_OFFSET_INDEX);
    contextKeyParams.queryPaddingSize = context->GetOptionalInputTensor(QUERY_PADDING_SIZE_INDEX);
    contextKeyParams.kvPaddingSize = context->GetOptionalInputTensor(KV_PADDING_SIZE_INDEX);
    contextKeyParams.blockTable = context->GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    contextKeyParams.keySharedPrefix = context->GetOptionalInputTensor(KEY_SHARED_PREFIX_INDEX);
    contextKeyParams.valueSharedPrefix = context->GetOptionalInputTensor(VALUE_SHARED_PREFIX_INDEX);
    contextKeyParams.actualSharedPrefixLen = context->GetOptionalInputTensor(ACTUAL_SHARED_PREFIX_LEN_INDEX);
    contextKeyParams.learnableSink = context->GetOptionalInputTensor(LEARNABLE_SINK_INDEX);
    contextKeyParams.learnableSinkShape = context->GetOptionalInputShape(LEARNABLE_SINK_INDEX);
    contextKeyParams.hasLearnableSink =
        ((contextKeyParams.learnableSink != nullptr) && (contextKeyParams.learnableSinkShape != nullptr) &&
         (contextKeyParams.learnableSinkShape->GetStorageShape().GetShapeSize() != 0)) ?
            true :
            false;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertInputDtypesAndShapesPFA(gert::TilingContext* context,
                                                      FiaArch38ContextParams& contextKeyParams)
{
    contextKeyParams.inputDataType = context->GetInputDesc(QUERY_INDEX)->GetDataType();
    contextKeyParams.kDataType = context->GetInputDesc(KEY_INDEX)->GetDataType();
    contextKeyParams.vDataType = context->GetInputDesc(VALUE_INDEX)->GetDataType();
    contextKeyParams.pseShiftDataType = (contextKeyParams.pseShift != nullptr) ?
                                            context->GetOptionalInputDesc(PSE_SHIFT_INDEX)->GetDataType() :
                                            contextKeyParams.inputDataType;
    contextKeyParams.maskDataType = (contextKeyParams.attentionMask != nullptr) ?
                                        context->GetOptionalInputDesc(ATTEN_MASK_INDEX)->GetDataType() :
                                        contextKeyParams.inputDataType;
    contextKeyParams.blockTableType = (context->GetOptionalInputDesc(BLOCK_TABLE_INDEX) != nullptr) ?
                                          context->GetOptionalInputDesc(BLOCK_TABLE_INDEX)->GetDataType() :
                                          ge::DT_INT32;
    contextKeyParams.outputDataType = context->GetOutputDesc(ATTENTION_OUT_INDEX)->GetDataType();
    contextKeyParams.queryInputShape = context->GetInputShape(QUERY_INDEX);
    contextKeyParams.keyInputShape = context->GetDynamicInputShape(KEY_INDEX, 0);
    contextKeyParams.valueInputShape = context->GetDynamicInputShape(VALUE_INDEX, 0);
    contextKeyParams.pseShiftShape = context->GetOptionalInputShape(PSE_SHIFT_INDEX);
    contextKeyParams.attentionMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INDEX);
    contextKeyParams.blockTableShape = context->GetOptionalInputShape(BLOCK_TABLE_INDEX);
    contextKeyParams.outputShape = context->GetOutputShape(ATTENTION_OUT_INDEX);
    contextKeyParams.lseoutputShape = context->GetOutputShape(SOFTMAX_LSE_INDEX);
    contextKeyParams.keySharedPrefixDataType =
        (contextKeyParams.keySharedPrefix != nullptr) ?
            context->GetOptionalInputDesc(KEY_SHARED_PREFIX_INDEX)->GetDataType() :
            contextKeyParams.inputDataType;
    contextKeyParams.valueSharedPrefixDataType =
        (contextKeyParams.valueSharedPrefix != nullptr) ?
            context->GetOptionalInputDesc(VALUE_SHARED_PREFIX_INDEX)->GetDataType() :
            contextKeyParams.inputDataType;
    contextKeyParams.learnableSinkDataType = (contextKeyParams.learnableSink != nullptr) ?
                                                 context->GetOptionalInputDesc(LEARNABLE_SINK_INDEX)->GetDataType() :
                                                 contextKeyParams.inputDataType;

    auto convertQuantRet = ConvertQuantOptionalInputs(context, contextKeyParams);
    if (convertQuantRet != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                               "Error occurred while converting quant related tilingContext to PFA context!");
        return convertQuantRet;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ConvertAttrsPFA(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams)
{
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, ReportNullPointer(context, __func__, "attrs"), return ge::GRAPH_FAILED);
    contextKeyParams.queryQuantMode = attrs->GetAttrPointer<int64_t>(ATTR_QUERY_QUANT_MODE_INDEX);
    contextKeyParams.keyAntiquantMode = attrs->GetAttrPointer<int64_t>(ATTR_KEY_ANTIQUANT_MODE_INDEX);
    contextKeyParams.valueAntiquantMode = attrs->GetAttrPointer<int64_t>(ATTR_VALUE_ANTIQUANT_MODE_INDEX);
    contextKeyParams.innerPrecisePtr = attrs->GetAttrPointer<int64_t>(ATTR_INNER_PRECISE_INDEX);
    contextKeyParams.headsNumber = attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
    contextKeyParams.sparseMode = attrs->GetAttrPointer<int64_t>(ATTR_SPARSE_MODE_INDEX);
    contextKeyParams.preToken = attrs->GetAttrPointer<int64_t>(ATTR_PRE_TOKEN_INDEX);
    contextKeyParams.nextToken = attrs->GetAttrPointer<int64_t>(ATTR_NEXT_TOKEN_INDEX);
    contextKeyParams.scaleValue = attrs->GetAttrPointer<float>(ATTR_SCALE_INDEX);
    contextKeyParams.layout = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    contextKeyParams.numKeyValueHeads = attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX);
    OP_CHECK_IF(((contextKeyParams.layout == nullptr) || (contextKeyParams.headsNumber == nullptr) ||
                 (contextKeyParams.numKeyValueHeads == nullptr) || (contextKeyParams.scaleValue == nullptr) ||
                 (contextKeyParams.preToken == nullptr) || (contextKeyParams.nextToken == nullptr)),
                ReportNullPointer(context, __func__,
                                  "input_layout/num_heads/num_key_value_heads/scale_value/pre_tokens/next_tokens"),
                return ge::GRAPH_FAILED);
    contextKeyParams.blockSize = attrs->GetAttrPointer<int64_t>(ATTR_BLOCK_SIZE_INDEX);
    contextKeyParams.workspaceSize = context->GetWorkspaceSizes(1);
    contextKeyParams.isBSNDOut = (string(contextKeyParams.layout) == "BNSD_BSND") ? 1 : 0;
    contextKeyParams.transposeLayout = GetTransposeLayout(string(contextKeyParams.layout));
    contextKeyParams.softmaxLseFlag = attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    contextKeyParams.isSoftMaxLseEnable =
        (contextKeyParams.softmaxLseFlag == nullptr) ? false : *contextKeyParams.softmaxLseFlag;
    contextKeyParams.queryRopeInputShape = context->GetOptionalInputShape(QUERY_ROPE_INDEX);
    contextKeyParams.qRopeDataType = (contextKeyParams.queryRopeInputShape != nullptr) ?
                                         context->GetOptionalInputDesc(QUERY_ROPE_INDEX)->GetDataType() :
                                         contextKeyParams.inputDataType;
    contextKeyParams.keyRopeInputShape = context->GetOptionalInputShape(KEY_ROPE_INDEX);
    contextKeyParams.kRopeDataType = (contextKeyParams.keyRopeInputShape != nullptr) ?
                                         context->GetOptionalInputDesc(KEY_ROPE_INDEX)->GetDataType() :
                                         contextKeyParams.kDataType;
    contextKeyParams.qStartIdx = context->GetOptionalInputTensor(Q_START_IDX_INDEX);
    contextKeyParams.kvStartIdx = context->GetOptionalInputTensor(KV_START_IDX_INDEX);
    contextKeyParams.pseType = attrs->GetAttrPointer<int64_t>(ATTR_PSE_TYPE_INDEX);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetBatchOfQ(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams,
                                   const std::string& layoutStr, bool isMaxWorkspace, int64_t& batchOfQ)
{
    batchOfQ = 1;
    if (layoutStr != "NSD") {
        if (layoutStr != "TND" && layoutStr != "NTD" && layoutStr != "NTD_NTD" && layoutStr != "TND_NTD") {
            batchOfQ = contextKeyParams.queryInputShape->GetStorageShape().GetDim(QUERY_DIM_0);
        } else {
            if (!isMaxWorkspace) {
                const gert::Tensor* actSeqLenData = contextKeyParams.actualSequenceLengthQ;
                const gert::StorageShape* actSeqLenShape = context->GetOptionalInputShape(ACTUAL_SEQ_Q_INDEX);
                int64_t actSeqLenDims =
                    (actSeqLenShape != nullptr) ? actSeqLenShape->GetStorageShape().GetShapeSize() : 0;
                OP_CHECK_IF(((actSeqLenData == nullptr) || (actSeqLenDims == 0) ||
                             (actSeqLenData->GetData<int64_t>() == nullptr)),
                            OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "actual_seq_lengths"),
                            return ge::GRAPH_FAILED);
                const gert::Tensor* actSeqLenDataKV = contextKeyParams.actualSequenceLengthKV;
                const gert::StorageShape* actSeqLenKVShape = context->GetOptionalInputShape(ACTUAL_SEQ_KV_INDEX);
                int64_t actSeqLenKVDims =
                    (actSeqLenKVShape != nullptr) ? actSeqLenKVShape->GetStorageShape().GetShapeSize() : 0;
                OP_CHECK_IF(((actSeqLenDataKV == nullptr) || (actSeqLenKVDims == 0) ||
                             (actSeqLenDataKV->GetData<int64_t>() == nullptr)),
                            OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "actual_seq_lengths_kv"),
                            return ge::GRAPH_FAILED);
                batchOfQ = actSeqLenDims;
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckTensorListScene(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams,
                                            const std::string& layoutStr, int64_t batchOfQ)
{
    // Obtain the actual number of K input elements and determine whether they belong to the tensorlist scene
    int64_t validBatchOfK = 0;
    while (context->GetDynamicInputShape(KEY_INDEX, validBatchOfK) != nullptr) {
        validBatchOfK++;
        // If there are more than 1, break. When the input is large, it saves time.
        if (validBatchOfK > 1) {
            break;
        }
    }
    if (validBatchOfK > 1) { // k tensor 超过一个, 认为是tensorlist场景
        OP_CHECK_IF(
            (contextKeyParams.blockTable != nullptr),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "block_table and key and value",
                                                     "When tensorlist is used, page attention is not supported!"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((layoutStr == "TND" || layoutStr == "NTD" || layoutStr == "NTD_TND" || layoutStr == "TND_NTD"),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "key and value and input_layout",
                        "When tensorlist is used, layout TND/NTD/TND_NTD/NTD_TND is not supported!"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((!CheckTensorList(context, contextKeyParams, layoutStr, batchOfQ)),
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Check Tensorlist failed!"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConvertContextToParamsPFA(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams,
                                          bool isMaxWorkspace)
{
    contextKeyParams.opName = context->GetNodeName();

    contextKeyParams.isKvContinuous = 1;
    contextKeyParams.emptyTensor = 0;
    contextKeyParams.fromFused = FROM_FUSED_FLAG;
    contextKeyParams.maxKVs = 0;
    if (ConvertOptionalInputTensorsPFA(context, contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ConvertInputDtypesAndShapesPFA(context, contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (ConvertAttrsPFA(context, contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    const string layoutStr = string(contextKeyParams.layout);
    int64_t batchOfQ = 1;
    if (GetBatchOfQ(context, contextKeyParams, layoutStr, isMaxWorkspace, batchOfQ) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckTensorListScene(context, contextKeyParams, layoutStr, batchOfQ) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(((contextKeyParams.isKvContinuous == 0) &&
                 ((context->GetOptionalInputDesc(QUERY_PADDING_SIZE_INDEX) != nullptr) ||
                  (context->GetOptionalInputDesc(KV_PADDING_SIZE_INDEX) != nullptr))),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(),
                                                         "key and value and query_padding_size and kv_padding_size",
                                                         "When tensorlist is used, left padding is not supported!"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!CheckKVPaddingCrossover(context, contextKeyParams),
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "Check KV PaddingSize failed!"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(contextKeyParams.workspaceSize == nullptr, ReportNullPointer(context, __func__, "workspaceSize"),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConvertContextToParamsIFA(gert::TilingContext& context, FiaArch38IncreContext& ifaContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE_WITHOUT_REPORT("FusedInferAttentionScore", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }
    ifaContext.query.desc = context.GetInputDesc(QUERY_INDEX);
    ifaContext.query.shape = context.GetInputShape(QUERY_INDEX);
    ifaContext.key.desc = context.GetInputDesc(KEY_INDEX);
    ifaContext.key.shape = context.GetInputShape(KEY_INDEX);
    OP_CHECK_IF((ifaContext.query.shape == nullptr) || (ifaContext.key.shape == nullptr),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context.GetNodeName(), "query,key", "nullptr",
                                                       "shape of query or shape of key is null."),
                return ge::GRAPH_FAILED);

    ifaContext.value.desc = context.GetInputDesc(VALUE_INDEX);
    ifaContext.value.shape = context.GetInputShape(VALUE_INDEX);
    ifaContext.attenMask.desc = context.GetOptionalInputDesc(ATTEN_MASK_INDEX);
    ifaContext.attenMask.tensor = context.GetOptionalInputTensor(ATTEN_MASK_INDEX);
    ifaContext.attenOut.desc = context.GetOutputDesc(ATTENTION_OUT_INDEX);
    ifaContext.attenOut.shape = context.GetOutputShape(ATTENTION_OUT_INDEX);
    ifaContext.blockTable.tensor = context.GetOptionalInputTensor(BLOCK_TABLE_INDEX);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, ReportNullPointer(&context, __func__, "attrs"), return ge::GRAPH_FAILED);

    ifaContext.numHeads = attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
    ifaContext.kvHeadNums = attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX);

    // IFA（伪量化）模板当前不支持 learnable sink 输入特性
    OP_CHECK_IF(context.GetOptionalInputTensor(LEARNABLE_SINK_INDEX) != nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context.GetNodeName(), "learnable_sink",
                                                         "Learnable sink only supports no-quantized GQA mode."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool GetMaxWorkspaceFlag(gert::TilingContext& context)
{
    bool res = (context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX) != nullptr &&
                context.GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX)->GetData<int64_t>() == nullptr) ||
               (context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX) != nullptr &&
                context.GetOptionalInputTensor(ACTUAL_SEQ_KV_INDEX)->GetData<int64_t>() == nullptr);
    return res;
}

} // namespace arch38
} // namespace optiling
