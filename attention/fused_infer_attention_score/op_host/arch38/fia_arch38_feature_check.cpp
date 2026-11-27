/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fia_arch38_prompt_tiling.h"
#include "../fused_infer_attention_score_tiling_index.h"

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {

namespace arch38 {

static ge::graphStatus ReportInvalidValue(const char* opName, const std::string& param, const std::string& value,
                                          const std::string& reason)
{
    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(opName, param.c_str(), value.c_str(), reason.c_str());
    return ge::GRAPH_FAILED;
}

static ge::graphStatus CheckArch38ShapeInputs(const char* opName, const gert::Shape& queryShape,
                                              const gert::Shape& keyShape, const gert::Shape& valueShape)
{
    if ((queryShape.GetDimNum() != BNSD_DIM_NUM) || (keyShape.GetDimNum() != BNSD_DIM_NUM) ||
        (valueShape.GetDimNum() != BNSD_DIM_NUM)) {
        return ReportInvalidValue(opName, "query, key, value",
                                  std::to_string(queryShape.GetDimNum()) + ", " + std::to_string(keyShape.GetDimNum()) +
                                      ", " + std::to_string(valueShape.GetDimNum()),
                                  "BNSD layout requires 4-dim query, key and value shapes.");
    }
    const int64_t sQ = queryShape.GetDim(BNSD_DIM_S);
    const int64_t sKV = keyShape.GetDim(BNSD_DIM_S);
    if ((sQ != 1) || (sKV <= 1) || (sKV > ARCH38_IFA_SUPPORTED_MAX_SKV)) {
        return ReportInvalidValue(
            opName, "query, key", std::to_string(sQ) + ", " + std::to_string(sKV),
            "query S must be 1, kv S must be in (1, " + std::to_string(ARCH38_IFA_SUPPORTED_MAX_SKV) + "].");
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckArch38HeadInputs(const char* opName, int64_t nQ, int64_t nKV)
{
    if ((nQ <= 0) || (nQ > ARCH38_IFA_SUPPORTED_MAX_N) || (nKV <= 0) || (nKV > ARCH38_IFA_SUPPORTED_MAX_N)) {
        return ReportInvalidValue(opName, "numHeads, numKeyValueHeads", std::to_string(nQ) + ", " + std::to_string(nKV),
                                  "head num must be in (0, " + std::to_string(ARCH38_IFA_SUPPORTED_MAX_N) + "].");
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckArch38IncreInputs(const gert::TilingContext& context)
{
    const char* opName = context.GetNodeName();
    if (opName == nullptr) {
        OP_LOGE_WITHOUT_REPORT("FusedInferAttentionScore", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    const auto* queryShapePtr = context.GetInputShape(QUERY_INDEX);
    const auto* keyShapePtr = context.GetInputShape(KEY_INDEX);
    const auto* valueShapePtr = context.GetInputShape(VALUE_INDEX);
    const auto* blockTable = context.GetOptionalInputTensor(BLOCK_TABLE_INDEX);
    const auto* attrs = context.GetAttrs();
    const auto* numHeads = (attrs == nullptr) ? nullptr : attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
    const auto* kvHeadNums = (attrs == nullptr) ? nullptr : attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX);
    if ((queryShapePtr == nullptr) || (keyShapePtr == nullptr) || (valueShapePtr == nullptr) || (attrs == nullptr) ||
        (numHeads == nullptr)) {
        ReportNullPointer(&context, __func__, "query/key/value.shape, attrs, numHeads");
        return ge::GRAPH_FAILED;
    }

    // 本平台不支持 PagedAttention（K 按块存放，形状无法给出实际 KV 序列长度）与 learnable sink。
    if ((blockTable != nullptr) || (context.GetOptionalInputTensor(LEARNABLE_SINK_INDEX) != nullptr)) {
        return ReportInvalidValue(opName, "blockTable, learnable_sink", "not null", "not supported on this platform.");
    }

    const auto& queryShape = queryShapePtr->GetStorageShape();
    const auto& keyShape = keyShapePtr->GetStorageShape();
    const auto& valueShape = valueShapePtr->GetStorageShape();
    const int64_t nQ = *numHeads;
    const int64_t nKV = ((kvHeadNums != nullptr) && (*kvHeadNums != 0)) ? *kvHeadNums : nQ;
    if (CheckArch38ShapeInputs(opName, queryShape, keyShape, valueShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckArch38HeadInputs(opName, nQ, nKV) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

bool FiaArch38PromptTiling::CheckPFAMerge(FiaArch38ContextParams& contextKeyParams,
                                          const PFAShapeInfo& queryShapeInfo) const
{
    if ((queryShapeInfo.s <= 1U) || (queryShapeInfo.s > pfaMergeQsLimit)) {
        return false;
    }

    const int64_t nQ = *contextKeyParams.headsNumber;
    const int64_t nKV = *contextKeyParams.numKeyValueHeads;
    if ((nKV > 0) && (static_cast<uint32_t>(nQ / nKV) > pfaMergeGLimit)) {
        return false;
    }

    if (enableMask || enablePseShift || enablePA || enableAlibiPse) {
        return false;
    }
    return true;
}

bool FiaArch38PromptTiling::CheckIO(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                    PFAShapeInfo& valueShapeInfo)
{
    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* outShape = contextKeyParams.outputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    // check dtype
    if (!CheckIODataType(contextKeyParams)) {
        return false;
    }

    // check layout
    OP_CHECK_IF((!SetInputLayout(contextKeyParams.layout)),
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(contextKeyParams.opName, "input_layout", contextKeyParams.layout,
                                                       "input layout must be one of TND, BSH, BNSD, BSND or BNSD_BSND"),
                return false);

    // check query shape size and bnsd
    OP_CHECK_IF((!GetAndCheckShape(contextKeyParams, queryShapeInfo, queryShape, "query")),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check query shape failed."), return false);
    if (queryShapeInfo.s == 1 && !enableAlibiPse) {
        enableIFA = true;
    }
    enablePFAMerge = CheckPFAMerge(contextKeyParams, queryShapeInfo);
    // check value shape
    if (enablePA) { // only get shape
        PFAShapeDims dims;
        if (!SetShape(contextKeyParams, valueShape, "value", dims)) {
            return false;
        }
        valueShapeInfo.d = dims.d;
    } else {
        OP_CHECK_IF((!GetAndCheckShape(contextKeyParams, valueShapeInfo, valueShape, "value")),
                    OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check value shape failed."), return false);
    }
    if (queryShapeInfo.d == MLA_QKD_SIZE && valueShapeInfo.d == MLA_VD_SIZE) {
        enablePFAMLA = true;
        enablePFAMerge = false;
    }
    // check query and output consistency
    OP_CHECK_IF((!CheckQueryOutParamsConsistency(contextKeyParams, queryShape, outShape)),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Query and output consistency check failed."),
                return false);

    return true;
}

bool FiaArch38PromptTiling::CheckKV(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& keyShapeInfo,
                                    PFAShapeInfo& valueShapeInfo) const
{
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;

    // check dtype
    if (!CheckKVDataType(contextKeyParams)) {
        return false;
    }

    // check key shape size and bnsd
    OP_CHECK_IF((!enablePA && !GetAndCheckShape(contextKeyParams, keyShapeInfo, keyShape, "key")),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check key shape failed."), return false);

    // check value shape size and bnsd
    OP_CHECK_IF((!enablePA && !GetAndCheckShape(contextKeyParams, valueShapeInfo, valueShape, "value")),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check value shape failed."), return false);

    // check key and value consistency
    OP_CHECK_IF((!CheckKeyValueParamsConsistency(contextKeyParams, keyShape, valueShape)),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Key and value consistency check failed."),
                return false);

    return true;
}

bool FiaArch38PromptTiling::CheckQueryAndKey(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                             PFAShapeInfo& keyShapeInfo, FiaArch38FlashAttentionTilingData& tilingData)
{
    // check numhead ratio
    if (!SetAndCheckHeadNumRatio(contextKeyParams, tilingData)) {
        return false;
    }

    // check b and s for not PA
    OP_CHECK_IF((contextKeyParams.fromTilingSink == 0) && (queryShapeInfo.b != keyShapeInfo.b) && (!enableTensorList) &&
                    (!enablePA),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, "query, key, value",
                    (std::to_string(queryShapeInfo.b) + ", " + std::to_string(keyShapeInfo.b)).c_str(),
                    "query batch must be equal to key and value batch"),
                return false);
    // check d size
    OP_CHECK_IF((contextKeyParams.fromTilingSink == 0) && (queryShapeInfo.d != keyShapeInfo.d) && (!enableTensorList) &&
                    (!enablePA),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, "query, key",
                    (std::to_string(queryShapeInfo.d) + ", " + std::to_string(keyShapeInfo.d)).c_str(),
                    "query d size must be equal to key d size"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckIFAMLA(FiaArch38ContextParams& contextKeyParams,
                                        const PFAShapeInfo& queryShapeInfo) const
{
    constexpr uint32_t maxQuerySeqLenInIfaMla = 16U; // ifa mla场景qS最大支持16
    OP_CHECK_IF((queryShapeInfo.s > maxQuerySeqLenInIfaMla || queryShapeInfo.s < 1),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    contextKeyParams.opName, "query", std::to_string(queryShapeInfo.s).c_str(),
                    "when enable ifa mla, query sequence length must be in range [1, maxQuerySeqLenInIfaMla]"),
                return false);
    static const std::set<uint32_t> supportNumHeadInIfaMla = {32U, 64U, 128U}; // ifa mla场景qN支持范围
    OP_CHECK_IF((supportNumHeadInIfaMla.find(queryShapeInfo.n) == supportNumHeadInIfaMla.end()),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "num_heads",
                                                      std::to_string(queryShapeInfo.n).c_str(),
                                                      "when enable ifa mla, query heads num must be 32, 64 or 128"),
                return false);
    const int64_t nKV =
        *contextKeyParams.numKeyValueHeads; // ifa mla场景不支持g = 1, 因此在nKV用默认值0, nQ替代也属于异常场景
    OP_CHECK_IF((nKV != 1U),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "num_key_value_heads",
                                                      std::to_string(nKV).c_str(),
                                                      "when enable ifa mla, key and value heads num must be 1"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckRope(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                      PFAShapeInfo& keyShapeInfo, PFAShapeInfo& queryRopeShapeInfo)
{
    if (contextKeyParams.queryRopeInputShape == nullptr && contextKeyParams.keyRopeInputShape == nullptr) {
        return true;
    }
    OP_CHECK_IF((contextKeyParams.queryRopeInputShape == nullptr && contextKeyParams.keyRopeInputShape != nullptr),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query_rope, key_rope",
                                                       "queryRope absent, keyRope present",
                                                       "queryRope and keyRope must be both present or both absent"),
                return false);
    OP_CHECK_IF((contextKeyParams.queryRopeInputShape != nullptr && contextKeyParams.keyRopeInputShape == nullptr),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query_rope, key_rope",
                                                       "queryRope present, keyRope absent",
                                                       "queryRope and keyRope must be both present or both absent"),
                return false);
    const gert::StorageShape* queryRopeShape = contextKeyParams.queryRopeInputShape;
    const gert::StorageShape* keyRopeShape = contextKeyParams.keyRopeInputShape;

    // check queryRope shape
    OP_CHECK_IF((!GetAndCheckRopeShape(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, queryRopeShape,
                                       {"query", "queryRope"})),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check queryRope shape failed."), return false);
    // check keyRope shape
    PFAShapeInfo keyRopeShapeInfo;
    OP_CHECK_IF((!enablePA && !GetAndCheckRopeShape(contextKeyParams, keyShapeInfo, keyRopeShapeInfo, keyRopeShape,
                                                    {"key", "keyRope"})),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check queryRope shape failed."), return false);
    enableIFA = false;
    enablePFAMerge = false;
    if (queryShapeInfo.d == QUERY_SHAPE_DIM_D_128_TILING_V2) {
        enablePFARope = true;
    } else {
        enableIFAMLA = true;
    }
    OP_LOGI(contextKeyParams.opName, "enableIFAMLA is %d, enablePA is %d, enableMask is %d", enableIFAMLA, enablePA,
            enableMask);
    return true;
}

bool FiaArch38PromptTiling::CheckQuant(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                       PFAShapeInfo& keyShapeInfo, const PFAShapeInfo& valueShapeInfo) const
{
    const gert::StorageShape* deqScale1Shape = contextKeyParams.deqScale1Shape;
    const gert::StorageShape* quantScale1Shape = contextKeyParams.scale1Shape;
    const gert::StorageShape* deqScale2Shape = contextKeyParams.deqScale2Shape;
    // per-tensor quant check
    if (enablePertensorQuant) {
        OP_CHECK_IF(!CheckPerTensorQuantParams(contextKeyParams),
                    OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "per-tensor quant params check failed!"),
                    return false);
    }

    // per-block quant check
    if (enablePerblockQuant) {
        OP_CHECK_IF(!CheckPerblockQuantParams(contextKeyParams, queryShapeInfo, keyShapeInfo, valueShapeInfo),
                    OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "per-block quant params check failed!"),
                    return false);
    }

    OP_CHECK_IF(
        ((contextKeyParams.inputDataType == ge::DT_INT8) && (contextKeyParams.outputDataType == ge::DT_FLOAT16) &&
         ((contextKeyParams.scale2Shape != nullptr) || (contextKeyParams.offset2Shape != nullptr))),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            contextKeyParams.opName, "quant_scale2, quant_offset2", "present",
            "when query dtype is int8 and output dtype is fp16, quant_scale2 and quant_offset2 must be null"),
        return false);

    // quant/dequant dtype check
    if ((inputType == ge::DT_FLOAT16) && enablePostQuant) {
        OP_CHECK_IF((deqScale1Shape != nullptr) || (quantScale1Shape != nullptr) || (deqScale2Shape != nullptr),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        contextKeyParams.opName, "dequant_scale1, quant_scale1, dequant_scale2", "present",
                        "when input dtype is fp16 and postQuant is enabled, dequantScale1, quantScale1 and "
                        "dequantScale2 must be null"),
                    return false);
    }

    // post-quant check
    if (enablePostQuant) {
        OP_CHECK_IF(!CheckPostQuantParams(contextKeyParams, queryShapeInfo, valueShapeInfo),
                    OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "post quant params check failed!"), return false);
    }
    OP_CHECK_IF(enableKVAntiquant && !CheckAntiquantParamsShape(contextKeyParams),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "antiquant params check failed!"), return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPrefix(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                        PFAShapeInfo& keyShapeInfo, FiaArch38FlashAttentionTilingData& tilingData)
{
    PFAShapeInfo prefixShapeInfo;
    tilingData.fiaArch38AttentionBaseParams.set_prefixSeqInnerSize(0);
    if (!isKVHasPrefix) {
        tilingData.fiaArch38AttentionBaseParams.set_isActualSharedPrefixLenNull(1);
        actualSharedPrefixLen = 0;
        return true;
    }
    // The prefix does not support tensorlist, PA, or left padding
    OP_CHECK_IF(enableTensorList,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "tensorlist, prefix",
                                                       "tensorlist with system prefix",
                                                       "when tensorlist is used, system prefix is not supported"),
                return false);
    OP_CHECK_IF(enableLeftPadding,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "prefix, leftPadding",
                                                       "system prefix with left padding",
                                                       "when system prefix is used, left padding is not supported"),
                return false);
    OP_CHECK_IF((contextKeyParams.inputDataType == ge::DT_INT8) && (contextKeyParams.kDataType == ge::DT_INT8),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "query, key/value", "both int8",
                    "when system prefix is used, query and key/value should not both be int8"),
                return false);

    // get prefix shape
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;

    OP_CHECK_IF(!GetAndCheckPrefixShape(contextKeyParams, queryShapeInfo, prefixShapeInfo, tilingData),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Get and check prefix shape failed."), return false);

    // check prefix kv consistency
    OP_CHECK_IF(!CheckKeyValuePrefixConsistency(contextKeyParams, keyShape),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "key value prefix consistency check failed."),
                return false);

    // check actSharedPrefix
    if ((contextKeyParams.fromTilingSink == 0U) && (contextKeyParams.actualSharedPrefixLen != nullptr) &&
        (contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetShapeSize() > 0) &&
        !CheckActSharedPrefix(contextKeyParams, prefixShapeInfo.s, keyShapeInfo.s)) {
        tilingData.fiaArch38AttentionBaseParams.set_isActualSharedPrefixLenNull(0);
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_isActualSharedPrefixLenNull(1);
        actualSharedPrefixLen = prefixShapeInfo.s;
    }
    tilingData.fiaArch38AttentionBaseParams.set_prefixSeqInnerSize(prefixShapeInfo.s);
    return true;
}

bool FiaArch38PromptTiling::CheckActSeq(const FiaArch38ContextParams& contextKeyParams,
                                        const PFAShapeInfo& queryShapeInfo) const
{
    if (inputLayout != InputLayout::TND) {
        return true;
    }

    const gert::Tensor* actSeqLen = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;
    OP_CHECK_IF(actSeqLen == nullptr, OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "actual_seq_lengths"),
                return false);
    OP_CHECK_IF(actSeqLenKV == nullptr, OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "actual_seq_lengths_kv"),
                return false);

    auto batchOfQuery = actSeqLen->GetShapeSize();
    auto batchOfKey = actSeqLenKV->GetShapeSize();
    OP_CHECK_IF(
        batchOfQuery != batchOfKey,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
            contextKeyParams.opName, "actual_seq_lengths, actual_seq_lengths_kv",
            (std::to_string(batchOfQuery) + ", " + std::to_string(batchOfKey)).c_str(),
            "when layout is TND, the batch size of actualSequenceLengthQ and actualSequenceLengthKV must be equal"),
        return false);

    uint32_t batchSize = queryShapeInfo.b; // actSeqLengthSize and actSeqLengthKVSize are equal
    if (!CheckActSeqMonotonic(contextKeyParams, actSeqLen, actSeqLenKV, batchSize)) {
        return false;
    }

    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    OP_CHECK_IF(actSeqLen->GetData<int64_t>()[batchSize - 1] != queryShape->GetStorageShape().GetDim(0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    contextKeyParams.opName, "actual_seq_lengths",
                    std::to_string(actSeqLen->GetData<int64_t>()[batchSize - 1]).c_str(),
                    "when layout is TND, the last element of actual_seq_lengths must equal T"),
                return false);
    if (!enablePA) {
        OP_CHECK_IF(actSeqLenKV->GetData<int64_t>()[batchSize - 1] != keyShape->GetStorageShape().GetDim(0),
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        contextKeyParams.opName, "actual_seq_lengths_kv",
                        std::to_string(actSeqLenKV->GetData<int64_t>()[batchSize - 1]).c_str(),
                        "when layout is TND, the last element of actual_seq_lengths_kv must equal T"),
                    return false);
    }

    return true;
}

bool FiaArch38PromptTiling::CheckActSeqMonotonic(const FiaArch38ContextParams& contextKeyParams,
                                                 const gert::Tensor* actSeqLen, const gert::Tensor* actSeqLenKV,
                                                 uint32_t batchSize) const
{
    int64_t lastActSeq = 0;
    int64_t lastActSeqKV = 0;
    for (uint32_t i = LOOP_BEGIN_NUM; i < batchSize; ++i) {
        int64_t curActSeq = actSeqLen->GetData<int64_t>()[i];
        int64_t curActSeqKV = actSeqLenKV->GetData<int64_t>()[i];
        if (curActSeq < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths",
                                                  std::to_string(curActSeq).c_str(), "actualSeqLengths must be >= 0");
            return false;
        }
        if ((i >= 1) && (curActSeq < lastActSeq)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths",
                                                  std::to_string(curActSeq).c_str(),
                                                  "when layout is TND, actualSeqLengths must not be decreasing");
            return false;
        }
        if (curActSeqKV < 0) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths_kv",
                                                  std::to_string(curActSeqKV).c_str(),
                                                  "actualSeqLengthsKv must be >= 0");
            return false;
        }
        if (!enablePA && (i >= 1) && (curActSeqKV < lastActSeqKV)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths_kv",
                                                  std::to_string(curActSeqKV).c_str(),
                                                  "when layout is TND, actualSeqLengthsKv must not be decreasing");
            return false;
        }
        lastActSeq = curActSeq;
        lastActSeqKV = curActSeqKV;
    }
    return true;
}

bool FiaArch38PromptTiling::CheckActSeqLen(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                           const PFAShapeInfo& keyShapeInfo)
{
    if (enableActSeqLen) {
        if (!CheckActSeqLenQ(contextKeyParams, queryShapeInfo)) {
            return false;
        }
    }
    if (enableActSeqLenKV) {
        if (!CheckActSeqLenKV(contextKeyParams, queryShapeInfo, keyShapeInfo)) {
            return false;
        }
    }
    return true;
}

bool FiaArch38PromptTiling::CheckActSeqLenQ(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo)
{
    const gert::Tensor* actSeqLen = contextKeyParams.actualSequenceLengthQ;
    constexpr int64_t actSeqLenDimsQMin = 1; // The min value of the length of actualSeqQ is 1

    std::string layoutStr(contextKeyParams.layout);
    OP_CHECK_IF(enableIFAMLA && (inputLayout != InputLayout::TND),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "input_layout, actual_seq_lengths",
                                                       (layoutStr + ", present").c_str(),
                                                       "actual_seq_lengths is only supported when layout is TND"),
                return false);
    OP_CHECK_IF((actSeqLenDims < queryShapeInfo.b) && (actSeqLenDims > actSeqLenDimsQMin),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    contextKeyParams.opName, "actual_seq_lengths", std::to_string(actSeqLenDims).c_str(),
                    "the dim of actual_seq_lengths must be 1 or greater than or equal to batch size"),
                return false);
    uint32_t actSeqLengthSize = std::min(static_cast<uint32_t>(actSeqLenDims), queryShapeInfo.b);
    for (uint32_t i = LOOP_BEGIN_NUM; i < actSeqLengthSize; ++i) {
        int64_t actSeqTmp = actSeqLen->GetData<int64_t>()[i];
        if (inputLayout == InputLayout::TND && i >= 1) {
            actSeqTmp -= actSeqLen->GetData<int64_t>()[i - 1];
        }
        if (actSeqTmp < 0 || actSeqTmp > queryShapeInfo.s) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths",
                                                  std::to_string(actSeqTmp).c_str(),
                                                  "actual_seq_lengths must be in range [0, s]");
            return false;
        }
        // query act seq len padding情况下不支持合轴
        if (actSeqTmp < queryShapeInfo.s) {
            enablePFAMerge = false;
        }
    }
    return true;
}

bool FiaArch38PromptTiling::CheckActSeqLenKV(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                             const PFAShapeInfo& keyShapeInfo)
{
    const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;
    constexpr int64_t actSeqLenDimsKVMin = 1; // The min value of the length of actualSeqKV is 1

    // Batch of key is 1 under tensorlist mode, the number of actual_seq_lengthsKV should be compared
    // with batch of query, batch of key is equal to query under non-tensorlist mode.
    OP_CHECK_IF((actSeqLenKVDims < queryShapeInfo.b) && (actSeqLenKVDims > actSeqLenDimsKVMin),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    contextKeyParams.opName, "actual_seq_lengths_kv", std::to_string(actSeqLenKVDims).c_str(),
                    "the dim of actual_seq_lengths_kv must be 1 or greater than or equal to batch size"),
                return false);
    uint32_t actSeqLengthKVSize = std::min(static_cast<uint32_t>(actSeqLenKVDims), queryShapeInfo.b);
    for (uint32_t i = LOOP_BEGIN_NUM; i < actSeqLengthKVSize; ++i) {
        int64_t actSeqKVTmp = static_cast<int64_t>(actSeqLenKV->GetData<int64_t>()[i]);
        if (inputLayout == InputLayout::TND && !enablePA && i >= 1) {
            actSeqKVTmp -= static_cast<int64_t>(actSeqLenKV->GetData<int64_t>()[i - 1]);
        }
        if (!enableTensorList && !enablePA) {
            int64_t actSeqKVMax = static_cast<int64_t>(keyShapeInfo.s);
            OP_CHECK_IF((actSeqKVTmp < 0) || (actSeqKVTmp > actSeqKVMax),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            contextKeyParams.opName, "actual_seq_lengths_kv", std::to_string(actSeqKVTmp).c_str(),
                            "actual_seq_lengths_kv must be in range [0, actSeqKVMax]"),
                        return false);
        } else if (enableTensorList) {
            int64_t actSeqKVMax;
            if ((inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::BSH)) {
                actSeqKVMax = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1);
            } else {
                actSeqKVMax = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2); // 2: Dim S for BNSD
            }
            OP_CHECK_IF((actSeqKVTmp < 0) || (actSeqKVTmp > actSeqKVMax),
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            contextKeyParams.opName, "actual_seq_lengths_kv", std::to_string(actSeqKVTmp).c_str(),
                            "actual_seq_lengths_kv must be in range [0, actSeqKVMax]"),
                        return false);
        }
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPseShiftTypeAndShape(FiaArch38ContextParams& contextKeyParams, uint32_t b, uint32_t n,
                                                      uint32_t s1, uint32_t s2)
{
    const gert::StorageShape* pseShiftShape = contextKeyParams.pseShiftShape;
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, pseShiftShape, "pseShift")) {
        return false;
    }
    pseShiftElemType = contextKeyParams.pseShiftDataType;

    if (!CheckPseShiftDtype(contextKeyParams)) {
        return false;
    }
    if (!CheckPseShiftShapeDim(contextKeyParams, b, n, s1, s2)) {
        return false;
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPseShiftDtype(const FiaArch38ContextParams& contextKeyParams)
{
    OP_CHECK_IF((inputType == ge::DT_FLOAT16 && pseShiftElemType != ge::DT_FLOAT16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                      GetPfaDataTypeStr(pseShiftElemType).c_str(),
                                                      "when query dtype is fp16, pse shift type must be fp16"),
                return false);

    OP_CHECK_IF((inputType == ge::DT_BF16 && pseShiftElemType != ge::DT_BF16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                      GetPfaDataTypeStr(pseShiftElemType).c_str(),
                                                      "when query dtype is bf16, pse shift type must be bf16"),
                return false);

    OP_CHECK_IF((inputType == ge::DT_INT8 && pseShiftElemType != ge::DT_FLOAT16),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                      GetPfaDataTypeStr(pseShiftElemType).c_str(),
                                                      "when query dtype is int8, pse shift type must be fp16"),
                return false);

    if (pseShiftElemType == ge::DT_FLOAT16) {
        pseShiftElemSize = FLOAT16_SIZE;
    } else if (pseShiftElemType == ge::DT_BF16) {
        pseShiftElemSize = BFLOAT16_SIZE;
    } else if (pseShiftElemType == ge::DT_FLOAT) {
        pseShiftElemSize = FLOAT32_SIZE;
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                              GetPfaDataTypeStr(pseShiftElemType).c_str(),
                                              "pse shift type must be fp16, bf16 or float");
        return false;
    }
    pseShiftTypeByteNum = BYTE_BLOCK / pseShiftElemSize;
    return true;
}

bool FiaArch38PromptTiling::CheckPseShiftShapeDim(const FiaArch38ContextParams& contextKeyParams, uint32_t b,
                                                  uint32_t n, uint32_t s1, uint32_t s2)
{
    const gert::StorageShape* pseShiftShape = contextKeyParams.pseShiftShape;
    size_t pseShiftDim = pseShiftShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((pseShiftDim != PSESHIFTDIM_4),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                         std::to_string(pseShiftDim).c_str(),
                                                         "pse shift shape must be 4 dimension"),
                return false);

    pseShiftBatch = pseShiftShape->GetStorageShape().GetDim(0);
    int64_t pseShiftN = pseShiftShape->GetStorageShape().GetDim(1); // 1: The sirst dimension is N.
    pseShiftS1 = pseShiftShape->GetStorageShape().GetDim(2);        // 2: The second dimension is S1.
    pseShiftS2 = pseShiftShape->GetStorageShape().GetDim(3);        // 3: The third dimension is S2.
    OP_CHECK_IF(
        ((pseShiftBatch != 1 && pseShiftBatch != b) || (pseShiftN != n) || (pseShiftS1 < s1) || (pseShiftS2 < s2)),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            contextKeyParams.opName, "pse_shift",
            ("[" + std::to_string(pseShiftBatch) + ", " + std::to_string(pseShiftN) + ", " +
             std::to_string(pseShiftS1) + ", " + std::to_string(pseShiftS2) + "]")
                .c_str(),
            "pse shift shape must be [1 or b, n, >=s1, >=s2]"),
        return false);
    return true;
}

bool FiaArch38PromptTiling::CheckInnerPrecise(FiaArch38ContextParams& contextKeyParams,
                                              FiaArch38FlashAttentionTilingData& tilingData)
{
    // innerPrecise 有效值为 0/1/2/3，超出此范围非法
    constexpr int32_t innerPreciseMax = 4;
    if (innerPrecise >= innerPreciseMax || innerPrecise < 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "inner_precise",
                                              std::to_string(innerPrecise).c_str(),
                                              "inner_precise should be 0, 1, 2 or 3");
        return false;
    }
    // Determine if the bit1 bit of innerPrecise requires invalid correction.
    tilingData.fiaArch38AttentionBaseParams.set_isRowInvalid(
        (static_cast<uint32_t>(static_cast<uint32_t>(innerPrecise) >> 1U)) & 1U);
    // Determine the bit0 bit of innerPrecise, high-performance or high-precision mode.
    innerPrecise = HIGH_PERFORMANCE;

    // FP16 pse is forced to enter high-precision mode.
    if ((contextKeyParams.pseShift != nullptr) && (inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PERFORMANCE)) {
        innerPrecise = HIGH_PRECISION;
        OP_LOGW(contextKeyParams.opName,
                "when the input is fp16 and enable pseshift, the mode is forcibly switched to high-precision!");
    }
    if (inputType != ge::DT_FLOAT16) {
        OP_LOGW(contextKeyParams.opName, "innerPrecise will not take effect when input type is %s!",
                GetPfaDataTypeStr(inputType).c_str());
    }
    if ((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PERFORMANCE)) {
        softmaxDataTypeSize = FLOAT16_SIZE; // The default size is fp32.
    }
    return true;
}

bool FiaArch38PromptTiling::ParseActualSeqLengths(FiaArch38ContextParams& contextKeyParams,
                                                  PFAShapeInfo& queryShapeInfo, std::vector<int64_t>& actualSeqLengths,
                                                  std::vector<int64_t>& actualSeqLengthsKV)
{
    uint32_t lenDims = queryShapeInfo.b; // The current length of the actSeqLen array is equal to batch size b.
    const gert::Tensor* actSeqLenData = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenDataKV = contextKeyParams.actualSequenceLengthKV;
    actSeqLenDims = (actSeqLenData != nullptr) ? actSeqLenData->GetShapeSize() : 0;
    actSeqLenKVDims = (actSeqLenDataKV != nullptr) ? actSeqLenDataKV->GetShapeSize() : 0;

    if (inputLayout == InputLayout::TND) {
        if ((actSeqLenData == nullptr) || (actSeqLenDataKV == nullptr)) {
            return false;
        }
        middleActualSeqLengths = static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[lenDims - 1]);
    }
    for (uint32_t i = LOOP_BEGIN_NUM; i < lenDims; i++) {
        if (inputLayout == InputLayout::TND) {
            actualSeqLengths[i] = (i == 0) ? static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[0]) :
                                             static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i]) -
                                                 static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i - 1]);
            actualSeqLengthsKV[i] = static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[i]);
            if (enableIFAMLA) {
                actualSeqLengths[i] *= gSize;
            }
            if (!enablePA && i >= 1) {
                actualSeqLengthsKV[i] -= static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[i - 1]);
            }
            if (actualSeqLengthsKV[i] != S2) {
                needInit = 1;
            }
            continue;
        }

        if (!ParseActualSeqLengthsQueryNonTnd(contextKeyParams, queryShapeInfo, actualSeqLengths, i)) {
            return false;
        }
        middleActualSeqLengths += actualSeqLengths[i];
        ParseActualSeqLengthsKvNonTnd(contextKeyParams, actualSeqLengthsKV, i);
    }

    return true;
}

bool FiaArch38PromptTiling::ParseActualSeqLengthsQueryNonTnd(FiaArch38ContextParams& contextKeyParams,
                                                             PFAShapeInfo& queryShapeInfo,
                                                             std::vector<int64_t>& actualSeqLengths, uint32_t i)
{
    const gert::Tensor* actSeqLenData = contextKeyParams.actualSequenceLengthQ;
    if (!enableActSeqLen) {
        actualSeqLengths[i] = queryShapeInfo.s;
    } else if (enableIFAMLA) {
        actualSeqLengths[i] = (actSeqLenDims > 1) ? static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i]) :
                                                    static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[0]);
        if (actualSeqLengths[i] != queryShapeInfo.s / gSize) {
            needInit = 1;
            OP_CHECK_IF((sparseNextTokens < 0) &&
                            (sparseNextTokens * (-1) >= static_cast<int32_t>(actualSeqLengths[i])) &&
                            isDefaultSparseMode,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            contextKeyParams.opName, "next_tokens", std::to_string(sparseNextTokens).c_str(),
                            "nexttoken absolute value should be smaller than the actual length of query"),
                        return false);
        }
        if (inputLayout == InputLayout::BSND || inputLayout == InputLayout::BSH) {
            actualSeqLengths[i] *= gSize;
        } else {
            actualSeqLengths[i] = queryShapeInfo.s;
        }
    } else {
        actualSeqLengths[i] = (actSeqLenDims > 1) ? static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[i]) :
                                                    static_cast<uint32_t>(actSeqLenData->GetData<int64_t>()[0]);
        if (actualSeqLengths[i] != queryShapeInfo.s) {
            needInit = 1;
            OP_CHECK_IF((sparseNextTokens < 0) &&
                            (sparseNextTokens * (-1) >= static_cast<int32_t>(actualSeqLengths[i])) &&
                            isDefaultSparseMode,
                        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                            contextKeyParams.opName, "next_tokens", std::to_string(sparseNextTokens).c_str(),
                            "nexttoken absolute value should be smaller than the actual length of query"),
                        return false);
        }
    }
    return true;
}

void FiaArch38PromptTiling::ParseActualSeqLengthsKvNonTnd(FiaArch38ContextParams& contextKeyParams,
                                                          std::vector<int64_t>& actualSeqLengthsKV, uint32_t i)
{
    const gert::Tensor* actSeqLenDataKV = contextKeyParams.actualSequenceLengthKV;
    if (!enableActSeqLenKV) { // The user did not input act_seq_kv
        if (!enableTensorList) {
            actualSeqLengthsKV[i] = S2;
        } else {
            if ((inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::BSH)) {
                actualSeqLengthsKV[i] = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(1);
            } else { // 2: Obtain the second dimension
                actualSeqLengthsKV[i] = contextKeyParams.kTensorList[i]->GetStorageShape().GetDim(2);
            }
        }
    } else {
        actualSeqLengthsKV[i] = (actSeqLenKVDims > 1) ? static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[i]) :
                                                        static_cast<uint32_t>(actSeqLenDataKV->GetData<int64_t>()[0]);
        if (actualSeqLengthsKV[i] != S2) {
            needInit = 1;
        }
    }
    maxActualseqKV = std::max(maxActualseqKV, actualSeqLengthsKV[i]);
}

bool FiaArch38PromptTiling::CheckMultiFeatureCrossover(FiaArch38ContextParams& contextKeyParams,
                                                       PFAShapeInfo& queryShapeInfo,
                                                       std::vector<int64_t>& actualSeqLengths,
                                                       std::vector<int64_t>& actualSeqLengthsKV,
                                                       FiaArch38FlashAttentionTilingData& tilingData)
{
    if (!ParseActualSeqLengths(contextKeyParams, queryShapeInfo, actualSeqLengths, actualSeqLengthsKV)) {
        return false;
    }

    uint32_t lenDims = queryShapeInfo.b; // The current length of the actSeqLen array is equal to batch size b.
    for (uint32_t i = LOOP_BEGIN_NUM; i < lenDims; i++) {
        if (!CheckMultiFeatureCrossoverPerBatch(contextKeyParams, actualSeqLengths, actualSeqLengthsKV, tilingData,
                                                i)) {
            return false;
        }
    }
    return true;
}

bool FiaArch38PromptTiling::CheckMultiFeatureCrossoverPerBatch(FiaArch38ContextParams& contextKeyParams,
                                                               std::vector<int64_t>& actualSeqLengths,
                                                               std::vector<int64_t>& actualSeqLengthsKV,
                                                               FiaArch38FlashAttentionTilingData& tilingData,
                                                               uint32_t i)
{
    OP_CHECK_IF(isDefaultSparseMode && (sparsePreTokens < 0) &&
                    (sparsePreTokens * (-1) >= (actualSeqLengthsKV[i] + actualSharedPrefixLen)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    contextKeyParams.opName, "pre_tokens", std::to_string(sparsePreTokens).c_str(),
                    "preToken absolute value should be smaller than the actual length of key and value"),
                return false);
    int64_t preTokensPerbatch = 0;
    int64_t nextTokensPerbatch = 0;
    CalcPerbatchTokens(actualSeqLengths, actualSeqLengthsKV, i, preTokensPerbatch, nextTokensPerbatch);
    if ((nextTokensPerbatch < 0) ||
        (actualSeqLengths[i] > (actualSeqLengthsKV[i] + actualSharedPrefixLen + preTokensPerbatch))) {
        needInit = 1;
    }

    OP_LOGI(contextKeyParams.opName, "preTokensPerbatch[%u] is %ld, nextTokensPerbatch[%u] is %ld", i,
            preTokensPerbatch, i, nextTokensPerbatch);

    if (!CheckBandModeTokens(contextKeyParams, actualSeqLengths, actualSeqLengthsKV, i)) {
        return false;
    }

    OP_LOGI(contextKeyParams.opName,
            "actualSeqLengths[%u] is %ld, actualSeqLengthsKV[%u] is %ld, actualSharedPrefixLen is %ld, needInit is %u",
            i, actualSeqLengths[i], i, actualSeqLengthsKV[i], actualSharedPrefixLen, needInit);

    tilingData.fiaArch38AttentionInitOutputParams.set_needInit(needInit);

    if (enableAlibiPse) {
        OP_CHECK_IF((actualSeqLengths[i] != actualSeqLengthsKV[i]),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        contextKeyParams.opName, "actual_seq_lengths, actual_seq_lengths_kv",
                        (std::to_string(actualSeqLengths[i]) + ", " + std::to_string(actualSeqLengthsKV[i])).c_str(),
                        "when pseType is 2/3, actualSeqLengths must be equal to actualSeqLengthsKv"),
                    return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckBandModeTokens(const FiaArch38ContextParams& contextKeyParams,
                                                const std::vector<int64_t>& actualSeqLengths,
                                                const std::vector<int64_t>& actualSeqLengthsKV, uint32_t i) const
{
    OP_CHECK_IF((isBandMode && (sparseNextTokens < 0) &&
                 (sparseNextTokens * (-1) >= actualSeqLengthsKV[i] + actualSharedPrefixLen)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    contextKeyParams.opName, "next_tokens", std::to_string(sparseNextTokens).c_str(),
                    "nextTokens absolute value should be smaller than the actual length of key and value "
                    "in band mode"),
                return false);

    OP_CHECK_IF((isBandMode && (sparsePreTokens < 0) && (sparsePreTokens * (-1) >= actualSeqLengths[i])),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    contextKeyParams.opName, "pre_tokens", std::to_string(sparsePreTokens).c_str(),
                    "preTokens absolute value should be smaller than the actual length of query in band mode"),
                return false);
    return true;
}

void FiaArch38PromptTiling::CalcPerbatchTokens(const std::vector<int64_t>& actualSeqLengths,
                                               const std::vector<int64_t>& actualSeqLengthsKV, uint32_t i,
                                               int64_t& preTokensPerbatch, int64_t& nextTokensPerbatch)
{
    if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
        preTokensPerbatch = SPARSE_MODE_INT_MAX;
        if (enableIFAMLA) {
            nextTokensPerbatch = actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i] / gSize;
        } else {
            nextTokensPerbatch = actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i];
        }
    } else if (sparseModeVal == SPARSE_MODE_BAND) {
        preTokensPerbatch = sparsePreTokens - actualSeqLengthsKV[i] - actualSharedPrefixLen + actualSeqLengths[i];
        nextTokensPerbatch = sparseNextTokens + actualSeqLengthsKV[i] + actualSharedPrefixLen - actualSeqLengths[i];
    } else {
        preTokensPerbatch = sparsePreTokens;
        nextTokensPerbatch = sparseNextTokens;
    }
}

bool FiaArch38PromptTiling::CheckPerblockCrossover(FiaArch38ContextParams& contextKeyParams)
{
    if (!enablePerblockQuant) {
        return true;
    }
    if (!CheckPerblockQuantCrossoverBase(contextKeyParams) || !CheckPerblockQuantCrossoverFeature(contextKeyParams)) {
        return false;
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPerblockQuantCrossoverBase(const FiaArch38ContextParams& contextKeyParams) const
{
    OP_CHECK_IF(enableActSeqLen,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths, per-block quant",
                                                       "actualSeqLen present with per-block quant",
                                                       "ActSeqLen is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(
        enableActSeqLenKV,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths_kv, per-block quant",
                                               "actualSeqLenKV present with per-block quant",
                                               "ActSeqLenKV is not supported in per-block quant scenario"),
        return false);
    OP_CHECK_IF((innerPrecise == 2) || (innerPrecise == 3),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "inner_precise",
                                                      std::to_string(innerPrecise).c_str(),
                                                      "innerPrecise must be 0 or 1 in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enablePA,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "block_table, per-block quant",
                                                       "PA enabled with per-block quant",
                                                       "PA is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enableTensorList,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "tensorlist, per-block quant",
                                                       "tensorlist with per-block quant",
                                                       "tensorlist is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enableLeftPadding,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "leftPadding, per-block quant",
                                                       "left padding with per-block quant",
                                                       "leftpadding is not supported in per-block quant scenario"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPerblockQuantCrossoverFeature(const FiaArch38ContextParams& contextKeyParams) const
{
    OP_CHECK_IF(enablePFAMLA,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query_rope/key_rope, per-block quant",
                                                       "PFAMLA with per-block quant",
                                                       "PFAMLA is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enablePFARope,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query_rope/key_rope, per-block quant",
                                                       "PFARope with per-block quant",
                                                       "PFARope is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enableMask,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "atten_mask, per-block quant",
                                                       "mask with per-block quant",
                                                       "mask is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enablePseShift,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "pse_shift, per-block quant",
                                                       "PseShift with per-block quant",
                                                       "PseShift is not supported in per-block quant scenario"),
                return false);
    OP_CHECK_IF(enableAlibiPse,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "pse_shift, per-block quant",
                                                       "AlibiPse with per-block quant",
                                                       "AlibiPse is not supported in per-block quant scenario"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckAlibiPseShiftTypeAndShape(FiaArch38ContextParams& contextKeyParams, uint32_t n)
{
    const gert::StorageShape* pseShape = contextKeyParams.pseShiftShape;
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, pseShape, "pseShift")) {
        return false;
    }
    auto& inputParams = faTilingAdapter.inputParamsRegbase;

    pseShiftElemType = contextKeyParams.pseShiftDataType;

    OP_CHECK_IF((pseShiftElemType != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                      GetPfaDataTypeStr(pseShiftElemType).c_str(),
                                                      "when pseType is 2/3, pse shift type must be float"),
                return false);

    // 0: (B,N2,G,S1,S2), 1: (B,N2,G,1,S2)
    PfaPseShapeType pseShapeType = PfaPseShapeType::PSE_B_N2_G_1_S2;
    if (pseShape != nullptr && pseShape->GetStorageShape().GetDimNum() != 0) {
        auto& pseShapeDims = pseShape->GetStorageShape();
        int64_t pseDimNum = pseShapeDims.GetDimNum();
        int64_t pseBSize = 0;
        if (pseType == static_cast<int64_t>(PfaPseType::PSE_INNER_MUL_ADD_TYPE) ||
            pseType == static_cast<int64_t>(PfaPseType::PSE_INNER_MUL_ADD_SQRT_TYPE)) {
            OP_CHECK_IF(pseDimNum != SLOPE_N_DIM_NUM,
                        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "pse_shift",
                                                                 std::to_string(pseDimNum).c_str(),
                                                                 "when pseType is 2/3, the dim of pseShift must be 1"),
                        return false);
            int64_t pseShiftN = pseShape->GetStorageShape().GetDim(0); // 0: The first dimension is N.
            OP_CHECK_IF(pseShiftN != n,
                        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                            contextKeyParams.opName, "pse_shift", std::to_string(pseShiftN).c_str(),
                            "when pseType is 2/3, the length of pseShift must be equal to query head number"),
                        return false);
            if (pseDimNum == 1) {
                pseShapeType = PfaPseShapeType::PSE_1_N2_G_SLOPE;
                pseBSize = 1;
            }
        }
        inputParams.set_pseBSize(static_cast<uint32_t>(pseBSize));
    }

    inputParams.set_pseShapeType(static_cast<uint8_t>(pseShapeType));
    return true;
}

bool FiaArch38PromptTiling::CheckAlibiPseCrossover(FiaArch38ContextParams& contextKeyParams)
{
    if (!enableAlibiPse) {
        return true;
    }

    OP_CHECK_IF(enableLeftPadding,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "pse_type, leftPadding",
                                                       "pseType 2/3 with left padding",
                                                       "when pseType is 2/3, left padding is not supported"),
                return false);

    OP_CHECK_IF((enableIFAMLA || enablePFARope),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "pse_type, query_rope/key_rope",
                                                       (std::to_string(pseType) + ", present").c_str(),
                                                       "when pseType is 2/3, rope is not supported"),
                return false);

    OP_CHECK_IF(
        (enablePFAMLA),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(contextKeyParams.opName, "query, key, value", "192, 192, 128",
                                                   "when pseType is 2/3, query, key and value D must be the same"),
        return false);

    return true;
}

} // namespace arch38
} // namespace optiling
