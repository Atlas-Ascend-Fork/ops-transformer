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

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {

namespace arch38 {

bool FiaArch38PromptTiling::CheckMaskShape(FiaArch38ContextParams& contextKeyParams, MaskShapeInfo& info,
                                           const uint32_t sQ, const uint32_t sK, const uint32_t batchSize) const
{
    const int64_t* sparseMode = contextKeyParams.sparseMode;
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    size_t attenMaskDim = attenMaskShape->GetStorageShape().GetDimNum();
    int64_t attenMaskN = 1U;
    if (attenMaskDim == MASKDIM_2) {
        if (enableIFA) {
            info.attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
            info.attenMaskS1 = 1;
            info.attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(1);
            info.strMaskShape = std::to_string(info.attenMaskBatch) + ", " + std::to_string(info.attenMaskS2);
        } else {
            info.attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(0);
            info.attenMaskS2 = attenMaskShape->GetStorageShape().GetDim(1);
            info.strMaskShape = std::to_string(info.attenMaskS1) + ", " + std::to_string(info.attenMaskS2);
        }
    } else if (attenMaskDim == MASKDIM_3) {
        info.attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        info.attenMaskS1 = attenMaskShape->GetStorageShape().GetDim(1);
        info.attenMaskS2 =
            attenMaskShape->GetStorageShape().GetDim(2); // 2: When the dim is 3, the second dimension is S2.
        info.strMaskShape = std::to_string(info.attenMaskBatch) + ", " + std::to_string(info.attenMaskS1) + ", " +
                            std::to_string(info.attenMaskS2);
    } else if (attenMaskDim == MASKDIM_4) {
        info.attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
        attenMaskN = attenMaskShape->GetStorageShape().GetDim(1);
        info.attenMaskS1 =
            attenMaskShape->GetStorageShape().GetDim(2); // 2: When the dim is 4, the second dimension is S1.
        info.attenMaskS2 =
            attenMaskShape->GetStorageShape().GetDim(3); // 3: When the dim is 4, the third dimension is S2.
        info.strMaskShape = std::to_string(info.attenMaskBatch) + ", " + std::to_string(attenMaskN) + ", " +
                            std::to_string(info.attenMaskS1) + ", " + std::to_string(info.attenMaskS2);
    } else {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "atten_mask",
                                                 std::to_string(attenMaskDim).c_str(),
                                                 "attenMask dim must be 2, 3 or 4");
        return false;
    }

    if (enableIFA) {
        info.checkMask = (info.attenMaskBatch == batchSize) && (info.attenMaskS1 == 1) && (info.attenMaskS2 >= S2);
    } else if (isDefaultSparseMode || (sparseMode != nullptr && *sparseMode == SPARSE_MODE_ALL_MASK)) {
        info.checkMask = (info.attenMaskS1 >= sQ) && (info.attenMaskS2 >= sK) &&
                         (info.attenMaskBatch == 1 || info.attenMaskBatch == batchSize);
    } else if ((sparseMode != nullptr) &&
               ((*sparseMode == SPARSE_MODE_LEFT_UP) || (*sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
                (*sparseMode == SPARSE_MODE_BAND))) {
        info.checkMask = (info.attenMaskBatch == 1) && (attenMaskN == 1) &&
                         (info.attenMaskS1 == SPARSE_OPTIMIZE_ATTENTION_SIZE) &&
                         (info.attenMaskS2 == SPARSE_OPTIMIZE_ATTENTION_SIZE);
    }
    return true;
}

void FiaArch38PromptTiling::SetSparseModeData(FiaArch38ContextParams& contextKeyParams,
                                              FiaArch38FlashAttentionTilingData& tilingData)
{
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    const int64_t* sparseMode = contextKeyParams.sparseMode;
    const int64_t* preTokens = contextKeyParams.preToken;
    const int64_t* nextTokens = contextKeyParams.nextToken;
    SetAttenMaskBatch(attenMaskShape, tilingData);
    ClampSparseTokens(preTokens, nextTokens);
    if ((attenMaskShape != nullptr) && (sparseMode != nullptr)) {
        if (*sparseMode == SPARSE_MODE_LEFT_UP) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = 0;
            // Right down tokens are calculated on the kernel side.
        } else if (*sparseMode == SPARSE_MODE_RIGHT_DOWN) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
        } else if (*sparseMode == SPARSE_MODE_ALL_MASK) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
        } else if (*sparseMode == SPARSE_MODE_BAND) {
            isBandMode = true;
        }
        if (enableIFA && !enableIFAMLA) {
            sparseModeVal = SPARSE_MODE_NO_MASK;
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
        }
        sparseModeVal = static_cast<int32_t>(*sparseMode);
        OP_LOGI(contextKeyParams.opName, "sparseMode is %d.", sparseModeVal);
    }
    if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_NO_MASK)) {
        // sparse mode need process attention mask when empty tensor scenes as same.
        if (((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0)) ||
            (attenMaskShape == nullptr)) {
            sparsePreTokens = SPARSE_MODE_INT_MAX;
            sparseNextTokens = SPARSE_MODE_INT_MAX;
            sparseModeVal = static_cast<int32_t>(*sparseMode);
        }
    }
    if (isDefaultSparseMode && enableLeftPadding) {
        // For scenes with sparse mode=0 and left padding, the attention mask part is fully calculated
        sparsePreTokens = SPARSE_MODE_INT_MAX;
        sparseNextTokens = SPARSE_MODE_INT_MAX;
    }
}

void FiaArch38PromptTiling::SetAttenMaskBatch(const gert::StorageShape* attenMaskShape,
                                              FiaArch38FlashAttentionTilingData& tilingData)
{
    size_t attenMaskBatch = 1;
    if ((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetDimNum() != MASKDIM_2)) {
        // 2: target dimension of attenMask
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0);
    }
    if ((attenMaskShape != nullptr) && enableIFA && (attenMaskShape->GetStorageShape().GetDimNum() == MASKDIM_2)) {
        attenMaskBatch = attenMaskShape->GetStorageShape().GetDim(0); // IFA mask shape (B,S2)
    }
    tilingData.fiaArch38AttentionSingleCoreParams.set_attenMaskBatch(attenMaskBatch);
}

void FiaArch38PromptTiling::ClampSparseTokens(const int64_t* preTokens, const int64_t* nextTokens)
{
    if (*preTokens > SPARSE_MODE_INT_MAX) {
        sparsePreTokens = SPARSE_MODE_INT_MAX;
    } else if (*preTokens < -(SPARSE_MODE_INT_MAX)) {
        sparsePreTokens = -(SPARSE_MODE_INT_MAX);
    } else {
        sparsePreTokens = *preTokens;
    }

    if (*nextTokens > SPARSE_MODE_INT_MAX) {
        sparseNextTokens = SPARSE_MODE_INT_MAX;
    } else if (*nextTokens < -(SPARSE_MODE_INT_MAX)) {
        sparseNextTokens = -(SPARSE_MODE_INT_MAX);
    } else {
        sparseNextTokens = *nextTokens;
    }
}

bool FiaArch38PromptTiling::CheckMaskShapeCrossSparse(FiaArch38ContextParams& contextKeyParams,
                                                      const int64_t* sparseMode, uint64_t sQ, const uint64_t sK,
                                                      const uint32_t batchSize)
{
    if ((contextKeyParams.fromTilingSink != 0) || (!enableMask)) {
        return true;
    }
    if (enableIFA || enableIFAMLA || enablePFAMerge) {
        sQ /= gSize; // 合轴场景使用原始的seq长度校验
    }
    MaskShapeInfo info;
    if (!CheckMaskShape(contextKeyParams, info, sQ, sK, batchSize)) {
        return false;
    }
    if (enableIFA) {
        OP_CHECK_IF(!info.checkMask,
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        contextKeyParams.opName, "atten_mask",
                        (std::to_string(info.attenMaskBatch) + ", " + std::to_string(info.attenMaskS1) + ", " +
                         std::to_string(info.attenMaskS2))
                            .c_str(),
                        "attenMask batch, Q_S and KV_S are invalid for the given sparse mode"),
                    return false);
    } else {
        if (isDefaultSparseMode || ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_ALL_MASK))) {
            OP_CHECK_IF(!info.checkMask,
                        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                            contextKeyParams.opName, "atten_mask",
                            (std::to_string(info.attenMaskBatch) + ", " + std::to_string(info.attenMaskS1) + ", " +
                             std::to_string(info.attenMaskS2))
                                .c_str(),
                            "attenMask batch, Q_S and KV_S are invalid for the given sparse mode"),
                        return false);
        }
        if ((sparseMode != nullptr) &&
            ((*sparseMode == SPARSE_MODE_LEFT_UP) || (*sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
             (*sparseMode == SPARSE_MODE_BAND)) &&
            !info.checkMask) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                contextKeyParams.opName, "atten_mask", info.strMaskShape.c_str(),
                "attenMask shape must be (2048, 2048), (1, 2048, 2048) or (1, 1, 2048, 2048) for this sparse mode");
            return false;
        }
    }
    attenMaskShapeType = info.attenMaskBatch > 1 ? 1 : 2; // 1 for multi-batch and 2 for 1 batch, same as fa
    return true;
}

bool FiaArch38PromptTiling::CheckMaskTypeAndShape(FiaArch38ContextParams& contextKeyParams,
                                                  FiaArch38FlashAttentionTilingData& tilingData)
{
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    OP_CHECK_IF((attenMaskShape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
                ReportTilingFailure(context_, "get the shape size of atten_mask failed"), return false);

    if (!CheckMaskDtype(contextKeyParams)) {
        return false;
    }

    const size_t attenMaskDimNum = attenMaskShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((attenMaskDimNum < 2),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "atten_mask",
                                                         std::to_string(attenMaskDimNum).c_str(),
                                                         "atten_mask dim must be at least 2"),
                return false);

    // 1: last frist dim, 2: last second dim
    int64_t maskKVsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskDimNum - 1);
    int64_t maskQsSize = attenMaskShape->GetStorageShape().GetDim(attenMaskDimNum - 2); // 2 for Q dim index

    if (enableIFA) {
        maskQsSize = 1;
    }

    tilingData.fiaArch38AttentionBaseParams.set_maskKVsSize(maskKVsSize);
    tilingData.fiaArch38AttentionBaseParams.set_maskQsSize(maskQsSize);
    return true;
}

bool FiaArch38PromptTiling::CheckMaskDtype(const FiaArch38ContextParams& contextKeyParams)
{
    auto maskDataType = contextKeyParams.maskDataType;
    std::vector<ge::DataType> allowedMaskDtypes = {ge::DT_FLOAT16, ge::DT_BOOL, ge::DT_INT8, ge::DT_UINT8};
    std::vector<uint32_t> maskTypeSizeArray = {FLOAT16_SIZE, BOOL_SIZE, INT8_SIZE, UINT8_SIZE};

    auto maskTypeCheck = std::find(allowedMaskDtypes.begin(), allowedMaskDtypes.end(), maskDataType);

    OP_CHECK_IF((maskTypeCheck == allowedMaskDtypes.end()),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "atten_mask",
                                                      GetPfaDataTypeStr(maskDataType).c_str(),
                                                      "maskType should be DT_FLOAT16, DT_BOOL, DT_INT8 or DT_UINT8"),
                return false);
    int32_t maskTypeIndex = std::distance(allowedMaskDtypes.begin(), maskTypeCheck);
    maskElemSize = maskTypeSizeArray[maskTypeIndex];

    // When in fp16 high-precision mode, the mask type only supports bool or int8.
    OP_CHECK_IF(((inputType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) && (maskDataType != ge::DT_BOOL) &&
                    (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    contextKeyParams.opName, "atten_mask", GetPfaDataTypeStr(maskDataType).c_str(),
                    "when innerprecise is 2/3 and input type is fp16, maskType should be bool, int8 or uint8"),
                return false);
    // When bf16, the mask type only supports bool or int8.
    OP_CHECK_IF((inputType == ge::DT_BF16) && (maskDataType != ge::DT_BOOL) && (maskDataType != ge::DT_INT8) &&
                    (maskDataType != ge::DT_UINT8),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    contextKeyParams.opName, "atten_mask", GetPfaDataTypeStr(maskDataType).c_str(),
                    "when input type is bfloat16, maskType should be bool, int8 or uint8"),
                return false);
    return true;
}

void FiaArch38PromptTiling::SetSparseType(uint32_t qS)
{
    if (sparseModeVal == SPARSE_MODE_NO_MASK) {
        if (sparsePreTokens >= qS && sparseNextTokens == 0) {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_CAUSAL);
        } else if (sparsePreTokens >= qS && !enablePA && sparseNextTokens >= S2) {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_ALL);
        } else {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_BAND);
        }
    } else if (sparseModeVal == SPARSE_MODE_ALL_MASK) {
        sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_ALL);
    } else if (sparseModeVal == SPARSE_MODE_LEFT_UP) {
        sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_CAUSAL);
    } else if (sparseModeVal == SPARSE_MODE_RIGHT_DOWN) {
        if (!enablePA && qS == S2) {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_CAUSAL);
        } else {
            sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_BAND);
        }
    } else if (sparseModeVal == SPARSE_MODE_BAND) {
        sparseType = static_cast<uint8_t>(PfaSparseEnum::PFA_BAND);
    }
}

bool FiaArch38PromptTiling::CheckSparseMode(FiaArch38ContextParams& contextKeyParams, uint32_t qS,
                                            FiaArch38FlashAttentionTilingData& tilingData)
{
    const int64_t* sparseMode = contextKeyParams.sparseMode;

    bool sparseCheck = false;
    if (sparseMode != nullptr) {
        sparseCheck = ((*sparseMode != SPARSE_MODE_NO_MASK) && (*sparseMode != SPARSE_MODE_LEFT_UP) &&
                       (*sparseMode != SPARSE_MODE_RIGHT_DOWN) && (*sparseMode != SPARSE_MODE_ALL_MASK) &&
                       (*sparseMode != SPARSE_MODE_BAND));
        OP_CHECK_IF(sparseCheck,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                        contextKeyParams.opName, "sparse_mode", std::to_string(*sparseMode).c_str(),
                        "sparse_mode is out of range, only 0, 1, 2, 3, 4 are supported"),
                    return false);
    }
    SetSparseModeData(contextKeyParams, tilingData);

    if (!CheckSparseTokensRange(contextKeyParams, qS)) {
        return false;
    }

    SetSparseType(qS);
    return true;
}

bool FiaArch38PromptTiling::CheckSparseTokensRange(const FiaArch38ContextParams& contextKeyParams, uint32_t qS) const
{
    const int64_t* sparseMode = contextKeyParams.sparseMode;
    const int64_t* nextTokens = contextKeyParams.nextToken;
    const int64_t* preTokens = contextKeyParams.preToken;
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;

    OP_CHECK_IF(((attenMaskShape != nullptr) && (sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_BAND) &&
                 (*preTokens < 0 && *nextTokens < 0)),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "pre_tokens, next_tokens",
                    (std::to_string(*preTokens) + ", " + std::to_string(*nextTokens)).c_str(),
                    "preTokens and nextTokens must not be negative in band mode"),
                return false);

    OP_CHECK_IF((sparsePreTokens < 0) && (sparseNextTokens < 0),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "pre_tokens, next_tokens",
                    (std::to_string(sparsePreTokens) + ", " + std::to_string(sparseNextTokens)).c_str(),
                    "preTokens and nextTokens must not be negative"),
                return false);

    OP_CHECK_IF((sparseNextTokens * (-1)) > sparsePreTokens,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "pre_tokens, next_tokens",
                    (std::to_string(sparsePreTokens) + ", " + std::to_string(sparseNextTokens)).c_str(),
                    "nexttoken line should be higher than pretoken line"),
                return false);

    OP_CHECK_IF(isDefaultSparseMode && (sparseNextTokens < 0) && (sparseNextTokens * (-1)) >= static_cast<int32_t>(qS),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    contextKeyParams.opName, "next_tokens", std::to_string(sparseNextTokens).c_str(),
                    "nextTokens absolute value should be smaller than the length of query"),
                return false);

    OP_CHECK_IF(
        isDefaultSparseMode && (sparsePreTokens < 0) &&
            (sparsePreTokens * (-1) >= (static_cast<int32_t>(S2) + static_cast<int32_t>(actualSharedPrefixLen))),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            contextKeyParams.opName, "pre_tokens", std::to_string(sparsePreTokens).c_str(),
            "preTokens absolute value should be smaller than the length of key and value"),
        return false);

    return true;
}

bool FiaArch38PromptTiling::CheckPACrossover(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo)
{
    if (enablePA) {
        OP_CHECK_IF((isKVHasPrefix),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        contextKeyParams.opName, "prefix, block_table", "system prefix with page attention",
                        "when system prefix is used, page attention is not supported"),
                    return false);
        OP_CHECK_IF((enableKVAntiquant),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "block_table, antiquant_scale",
                                                           "block_table present with antiquant",
                                                           "antiquant is not supported when blockTable is not null"),
                    return false);
    }
    if (enableActSeqLenKV && !enableTensorList) {
        const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;
        uint32_t actSeqLenKVSize = std::min(static_cast<uint32_t>(actSeqLenKVDims), queryShapeInfo.b);
        for (uint32_t i = LOOP_BEGIN_NUM; i < actSeqLenKVSize; ++i) {
            if (enablePA && (actSeqLenKV->GetData<int64_t>()[i] < 0)) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_seq_lengths_kv",
                                                      std::to_string(actSeqLenKV->GetData<int64_t>()[i]).c_str(),
                                                      "actual_seq_lengths_kv must be >= 0");
                return false;
            }
        }
    }
    return true;
}

bool FiaArch38PromptTiling::CheckMaskCrossIFAMLA(FiaArch38ContextParams& contextKeyParams, const int64_t* sparseMode,
                                                 uint64_t queryS)
{
    if (sparseMode == nullptr) {
        return true;
    }
    if (queryS == 1U) {
        OP_CHECK_IF(!((*sparseMode == SPARSE_MODE_NO_MASK) && (!enableMask)),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        contextKeyParams.opName, "sparse_mode, atten_mask",
                        (std::to_string(*sparseMode) + ", " + std::string(enableMask ? "present" : "absent")).c_str(),
                        "only sparse 0 without mask is supported when ifa mla and query sequence length is 1"),
                    return false);
    } else {
        OP_CHECK_IF(!(((*sparseMode == SPARSE_MODE_RIGHT_DOWN) && (enableMask)) ||
                      ((*sparseMode == SPARSE_MODE_NO_MASK) && (!enableMask))),
                    OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                        contextKeyParams.opName, "sparse_mode, atten_mask",
                        (std::to_string(*sparseMode) + ", " + std::string(enableMask ? "present" : "absent")).c_str(),
                        "only sparse 3 with mask or sparse 0 without mask is supported when ifa mla and "
                        "query sequence length is > 1"),
                    return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckMaskCrossover(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                               FiaArch38FlashAttentionTilingData& tilingData)
{
    auto maskDataType = contextKeyParams.maskDataType;
    const int64_t* sparseMode = contextKeyParams.sparseMode;
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    if ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_LEFT_UP || *sparseMode == SPARSE_MODE_RIGHT_DOWN ||
                                    *sparseMode == SPARSE_MODE_ALL_MASK || *sparseMode == SPARSE_MODE_BAND)) {
        OP_CHECK_IF(((attenMaskShape != nullptr) && (attenMaskShape->GetStorageShape().GetShapeSize() == 0)) ||
                        (attenMaskShape == nullptr),
                    OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "atten_mask"), return false);

        // When sparse=2, 3, 4, the mask type only supports bool, int8, uint8
        OP_CHECK_IF((*sparseMode != SPARSE_MODE_ALL_MASK) && (maskDataType != ge::DT_BOOL) &&
                        (maskDataType != ge::DT_INT8) && (maskDataType != ge::DT_UINT8),
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        contextKeyParams.opName, "atten_mask", GetPfaDataTypeStr(maskDataType).c_str(),
                        "maskType should be bool, int8 or uint8 for this sparse mode"),
                    return false);
    }
    // FP16 mask type does not support invalid line correction.
    OP_CHECK_IF((maskDataType == ge::DT_FLOAT16 &&
                 static_cast<bool>(tilingData.fiaArch38AttentionBaseParams.get_isRowInvalid())),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "atten_mask",
                                                      GetPfaDataTypeStr(maskDataType).c_str(),
                                                      "maskType should not be float16 when innerPrecise is 2 or 3"),
                return false);
    if (!CheckMaskShapeCrossSparse(contextKeyParams, sparseMode, queryShapeInfo.s, S2 + actualSharedPrefixLen,
                                   queryShapeInfo.b)) {
        return false;
    }
    if (enableIFAMLA && (!CheckMaskCrossIFAMLA(contextKeyParams, sparseMode, queryShapeInfo.s / gSize))) {
        return false;
    }
    return true;
}

bool FiaArch38PromptTiling::CheckTNDLayoutCrossover(FiaArch38ContextParams& contextKeyParams)
{
    if (inputLayout != InputLayout::TND) {
        return true;
    }

    OP_CHECK_IF(enableLeftPadding,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "input_layout, leftPadding",
                                                       "TND with left padding",
                                                       "when layout is TND, left padding is not supported"),
                return false);

    OP_CHECK_IF(enableTensorList,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "input_layout, tensorlist",
                                                       "TND with tensorlist",
                                                       "when layout is TND, tensorlist is not supported"),
                return false);

    OP_CHECK_IF(enablePseShift,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "input_layout, pse_shift",
                                                       "TND with pse", "when layout is TND, pse is not supported"),
                return false);

    return true;
}

} // namespace arch38
} // namespace optiling
