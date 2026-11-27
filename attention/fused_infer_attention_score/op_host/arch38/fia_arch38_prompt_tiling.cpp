/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fia_arch38_prompt_tiling.h"
#include <queue>
#include <vector>

#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "log/log.h"
#include "err/ops_err.h"
#include "../../op_kernel/arch38/fia_arch38_policy.h"
#include <unordered_map>
#include <string>

namespace {
static const std::unordered_map<ge::DataType, std::string> g_strDataTypeArch38 = {
    {ge::DT_FLOAT, "DT_FLOAT"},
    {ge::DT_FLOAT16, "DT_FLOAT16"},
    {ge::DT_INT8, "DT_INT8"},
    {ge::DT_INT16, "DT_INT16"},
    {ge::DT_UINT16, "DT_UINT16"},
    {ge::DT_UINT8, "DT_UINT8"},
    {ge::DT_INT32, "DT_INT32"},
    {ge::DT_INT64, "DT_INT64"},
    {ge::DT_UINT32, "DT_UINT32"},
    {ge::DT_UINT64, "DT_UINT64"},
    {ge::DT_BOOL, "DT_BOOL"},
    {ge::DT_DOUBLE, "DT_DOUBLE"},
    {ge::DT_STRING, "DT_STRING"},
    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8V"},
    {ge::DT_COMPLEX64, "DT_COMPLEX64"},
    {ge::DT_COMPLEX128, "DT_COMPLEX128"},
    {ge::DT_QINT8, "DT_QINT8"},
    {ge::DT_QINT16, "DT_QINT16"},
    {ge::DT_QINT32, "DT_QINT32"},
    {ge::DT_QUINT8, "DT_QUINT8"},
    {ge::DT_QUINT16, "DT_QUINT16"},
    {ge::DT_RESOURCE, "DT_RESOURCE"},
    {ge::DT_STRING_REF, "DT_STRING_REF"},
    {ge::DT_DUAL, "DT_DUAL"},
    {ge::DT_VARIANT, "DT_VARIANT"},
    {ge::DT_BF16, "DT_BF16"},
    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
    {ge::DT_UNDEFINED, "DT_UNDEFINED"},
};
} // namespace

namespace optiling {
namespace arch38 {
std::string GetPfaDataTypeStr(ge::DataType type)
{
    ge::DataType findDype = (g_strDataTypeArch38.find(type) == g_strDataTypeArch38.end()) ? ge::DT_UNDEFINED : type;
    return g_strDataTypeArch38.at(findDype);
}

} // namespace arch38
} // namespace optiling

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {

namespace arch38 {

bool FiaArch38PromptTiling::CheckNonEmptyShapeExceptions(const FiaArch38ContextParams& contextKeyParams,
                                                         const gert::StorageShape* shape,
                                                         const std::string& sName) const
{
    OP_CHECK_IF(shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, sName.c_str()), return false);
    OP_CHECK_IF(shape->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue,
                ReportTilingFailure(context_, "get the shape size of tensor failed"), return false);
    return true;
}

ge::graphStatus FiaArch38PromptTiling::SetPlatMemoryInfo(FiaArch38ContextParams& contextKeyParams)
{
    auto compileInfoPtr = contextKeyParams.compileInfoPtr;
    contextKeyParamsPtr = &contextKeyParams;

    OP_CHECK_IF(compileInfoPtr == nullptr, ReportNullPointer(context_, __func__, "compileInfoPtr"),
                return ge::GRAPH_FAILED);

    coreNum = compileInfoPtr->aivNum;
    OP_CHECK_IF(coreNum == 0, ReportTilingFailure(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    aivNum = compileInfoPtr->aivNum;
    OP_CHECK_IF(aivNum == 0, ReportTilingFailure(context_, "aivNum is 0"), return ge::GRAPH_FAILED);
    aicNum = compileInfoPtr->aicNum;
    OP_CHECK_IF(aicNum == 0, ReportTilingFailure(context_, "aicNum is 0"), return ge::GRAPH_FAILED);
    ascendPlatformInfo.socVersion = compileInfoPtr->socShortName;
    ascendPlatformInfo.l1Size = compileInfoPtr->l1Size;
    ascendPlatformInfo.l0CSize = compileInfoPtr->l0CSize;
    ascendPlatformInfo.l0ASize = compileInfoPtr->l0ASize;
    ascendPlatformInfo.l0BSize = compileInfoPtr->l0BSize;
    ascendPlatformInfo.ubSize = compileInfoPtr->ubSize;
    OP_LOGI(contextKeyParams.opName,
            "ascendPlatformInfo:aivNum = %u, aicNum = %u, l1Size = %lu, l0CSize = %lu,"
            "l0ASize = %lu, l0BSize = %lu, ubSize = %lu!",
            aivNum, aicNum, compileInfoPtr->l1Size, compileInfoPtr->l0CSize, compileInfoPtr->l0ASize,
            compileInfoPtr->l0BSize, compileInfoPtr->ubSize);
    cvRatio = 1U; // Vector : Cube
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::SetAttributeInfo(FiaArch38ContextParams& contextKeyParams)
{
    if (SetAntiquantAndQuantInfo(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (SetPrefixAndSeqInfo(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (SetPseAndMaskInfo(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::SetAntiquantAndQuantInfo(FiaArch38ContextParams& contextKeyParams)
{
    if (contextKeyParams.hasKeyAntiquantScale || contextKeyParams.hasValueAntiquantScale) {
        enableKVAntiquant = false;
    } else {
        if (contextKeyParams.inputDataType == ge::DT_FLOAT16 && contextKeyParams.kDataType == ge::DT_INT8) {
            enableKVAntiquant = true;
        }
        OP_CHECK_IF(contextKeyParams.inputDataType == ge::DT_BF16 && contextKeyParams.kDataType == ge::DT_INT8,
                    OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "key_antiquant_scale, value_antiquant_scale"),
                    return ge::GRAPH_FAILED);
    }

    // postQuant check
    if (contextKeyParams.outputDataType != ge::DT_BF16 && contextKeyParams.outputDataType != ge::DT_FLOAT16) {
        enablePostQuant = true;
    }

    // per-tensor or per-block check
    if (contextKeyParams.inputDataType == ge::DT_INT8) {
        enablePertensorQuant = true;
    }

    const int64_t* keyAntiquantMode = contextKeyParams.keyAntiquantMode;
    const int64_t* queryQuantMode = contextKeyParams.queryQuantMode;
    const int64_t* valueAntiquantMode = contextKeyParams.valueAntiquantMode;
    if (contextKeyParams.inputDataType == ge::DT_HIFLOAT8 || contextKeyParams.inputDataType == ge::DT_FLOAT8_E4M3FN) {
        OP_CHECK_IF(((keyAntiquantMode == nullptr) || (queryQuantMode == nullptr) || (valueAntiquantMode == nullptr)),
                    OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName,
                                               "key_antiquant_mode, query_quant_mode, value_antiquant_mode"),
                    return ge::GRAPH_FAILED);
        if (*keyAntiquantMode == 7 && *queryQuantMode == 7 && *valueAntiquantMode == 7) { // 7: FP8 perblock quant
            enablePerblockQuant = true;
        } else {
            enablePertensorQuant = true;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::SetPrefixAndSeqInfo(FiaArch38ContextParams& contextKeyParams)
{
    // prefix check
    isKVHasPrefix =
        (contextKeyParams.keySharedPrefix != nullptr) && (contextKeyParams.valueSharedPrefix != nullptr) ? true : false;
    OP_CHECK_IF((!isKVHasPrefix &&
                 ((contextKeyParams.keySharedPrefix != nullptr) || contextKeyParams.valueSharedPrefix != nullptr)),
                OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "key_shared_prefix, value_shared_prefix"),
                return ge::GRAPH_FAILED);

    // actSeqLen check
    const gert::Tensor* actSeqLenData = contextKeyParams.actualSequenceLengthQ;
    const gert::Tensor* actSeqLenDataKV = contextKeyParams.actualSequenceLengthKV;
    actSeqLenDims = (actSeqLenData != nullptr) ? actSeqLenData->GetShapeSize() : 0;
    actSeqLenKVDims = (actSeqLenDataKV != nullptr) ? actSeqLenDataKV->GetShapeSize() : 0;
    enableActSeqLen =
        !((actSeqLenDims == 0) || (actSeqLenData == nullptr) || (actSeqLenData->GetData<int64_t>() == nullptr));
    enableActSeqLenKV =
        !((actSeqLenKVDims == 0) || (actSeqLenDataKV == nullptr) || (actSeqLenDataKV->GetData<int64_t>() == nullptr));

    // PA check
    if (contextKeyParams.blockTable != nullptr) {
        OP_CHECK_IF(contextKeyParams.blockSize == nullptr,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(contextKeyParams.opName, "block_size",
                                                             "block_size is required when PA is enabled"),
                    return ge::GRAPH_FAILED);
        enablePA = true;
    }

    // tensorlist check
    enableTensorList = contextKeyParams.isKvContinuous == 1 ? false : true;

    // LeftPadding check
    enableLeftPadding = ((contextKeyParams.queryPaddingSize != nullptr) || (contextKeyParams.kvPaddingSize != nullptr));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::SetPseAndMaskInfo(FiaArch38ContextParams& contextKeyParams)
{
    // Pse
    const gert::StorageShape* pseShiftShape = contextKeyParams.pseShiftShape;
    enablePseShift = (contextKeyParams.fromTilingSink == 0) && (contextKeyParams.pseShift != nullptr) &&
                     (pseShiftShape != nullptr) && (pseShiftShape->GetStorageShape().GetShapeSize() > 0);

    if (contextKeyParams.pseType != nullptr) {
        pseType = *contextKeyParams.pseType;
        OP_CHECK_IF(
            (pseType != 0) && (pseType != 2) && (pseType != 3),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "pse_type", std::to_string(pseType).c_str(),
                                                  "pse_type is not supported, must be 0, 2 or 3"),
            return ge::GRAPH_FAILED);
        faTilingAdapter.inputParamsRegbase.set_pseType(pseType);
    }
    if (pseType == PSE_TYPE_2_TILING_V2 || pseType == PSE_TYPE_3_TILING_V2) {
        enableAlibiPse = true;
        enablePseShift = false;
    }

    // innerprecise, 910B defaults to high-performance.
    const int64_t* innerPrecisePtr = contextKeyParams.innerPrecisePtr;
    innerPrecise = innerPrecisePtr != nullptr ? static_cast<int32_t>(*innerPrecisePtr) : HIGH_PERFORMANCE;

    // mask check
    const gert::StorageShape* attenMaskShape = contextKeyParams.attentionMaskShape;
    enableMask = (contextKeyParams.attentionMask != nullptr) && (attenMaskShape != nullptr) &&
                 (attenMaskShape->GetStorageShape().GetShapeSize() != 0);

    // sparsemode check
    const int64_t* sparseMode = contextKeyParams.sparseMode;
    isDefaultSparseMode = (sparseMode == nullptr) || ((sparseMode != nullptr) && (*sparseMode == SPARSE_MODE_NO_MASK));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckRopeInvalid(const FiaArch38ContextParams& contextKeyParams) const
{
    const gert::StorageShape* queryRopeShapeParam = contextKeyParams.queryRopeInputShape;
    const gert::StorageShape* keyRopeShapeParam = contextKeyParams.keyRopeInputShape;
    if ((queryRopeShapeParam != nullptr) && (queryRopeShapeParam->GetStorageShape().GetShapeSize() == 0)) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            contextKeyParams.opName, "query_rope", "0",
            "queryRope must not be an empty tensor when queryRopeShape is not null");
        return ge::GRAPH_FAILED;
    }
    if ((keyRopeShapeParam != nullptr) && (keyRopeShapeParam->GetStorageShape().GetShapeSize() == 0)) {
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, "key_rope", "0",
                                                  "keyRope must not be an empty tensor when keyRopeShape is not null");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckTensorInvalid(const FiaArch38ContextParams& contextKeyParams) const
{
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.queryInputShape, "query")) {
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((contextKeyParams.queryInputShape->GetStorageShape().GetShapeSize() == 0) &&
                    (contextKeyParams.outputShape->GetStorageShape().GetShapeSize() != 0),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query, attention_out",
                                                       "query empty, attention_out not empty",
                                                       "query input is empty but output is not empty"),
                return ge::GRAPH_FAILED);
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.keyInputShape, "key")) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.valueInputShape, "value")) {
        return ge::GRAPH_FAILED;
    }
    if (!CheckNonEmptyShapeExceptions(contextKeyParams, contextKeyParams.outputShape, "output")) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckSingleAttribute(FiaArch38ContextParams& contextKeyParams,
                                                            PFAShapeSet& shapes,
                                                            FiaArch38FlashAttentionTilingData& tilingData)
{
    if (CheckShapeStage(contextKeyParams, shapes, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckFeatureStage(contextKeyParams, shapes, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckShapeStage(FiaArch38ContextParams& contextKeyParams, PFAShapeSet& shapes,
                                                       FiaArch38FlashAttentionTilingData& tilingData)
{
    if (!CheckIO(contextKeyParams, *shapes.query, *shapes.value)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check query/ouput failed!");
        return ge::GRAPH_FAILED;
    }
    if (!CheckKV(contextKeyParams, *shapes.key, *shapes.value)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check key/value failed!");
        return ge::GRAPH_FAILED;
    }
    if (!CheckRope(contextKeyParams, *shapes.query, *shapes.key, *shapes.queryRope)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check queryRope/keyRope failed!");
        return ge::GRAPH_FAILED;
    }
    if (!CheckQueryAndKey(contextKeyParams, *shapes.query, *shapes.key, tilingData)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check query and key consistency failed!");
        return ge::GRAPH_FAILED;
    }
    if (enableIFAMLA && (!CheckIFAMLA(contextKeyParams, *shapes.query))) {
        return ge::GRAPH_FAILED;
    }
    // print shape info
    OP_LOGI(contextKeyParams.opName,
            "Tiling Info: Q_B is %u, KV_B is %u, Q_N is %u, KV_N is %u, Q_S is %u, KV_S is %u, H is %u, D is %u, "
            "headNumRatio = %u",
            shapes.query->b, shapes.key->b, shapes.query->n, shapes.key->n, shapes.query->s, S2, shapes.query->h,
            shapes.query->d, tilingData.fiaArch38AttentionBaseParams.get_headNumRatio());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckFeatureStage(FiaArch38ContextParams& contextKeyParams, PFAShapeSet& shapes,
                                                         FiaArch38FlashAttentionTilingData& tilingData)
{
    // quant/dequant/antiquant, prefix, actSeq, actSeqLen
    if (CheckBaseFeatureStage(contextKeyParams, shapes, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // PA check
    if (CheckPAStage(contextKeyParams, shapes, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // pse check
    if (CheckPseShiftStage(contextKeyParams, shapes) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // innerPrecise check
    if (!CheckInnerPrecise(contextKeyParams, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // mask check
    if (enableMask) {
        if (!CheckMaskTypeAndShape(contextKeyParams, tilingData)) {
            return ge::GRAPH_FAILED;
        }
        tilingData.fiaArch38AttentionBaseParams.set_useMask(1);
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_useMask(0);
    }

    // sparseMode check
    if (!CheckSparseMode(contextKeyParams, shapes.query->s, tilingData)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckBaseFeatureStage(FiaArch38ContextParams& contextKeyParams,
                                                             PFAShapeSet& shapes,
                                                             FiaArch38FlashAttentionTilingData& tilingData)
{
    // quant/dequant/antiquant
    if (!CheckQuant(contextKeyParams, *shapes.query, *shapes.key, *shapes.value)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check quant failed!");
        return ge::GRAPH_FAILED;
    }

    // prefix check
    if (!CheckPrefix(contextKeyParams, *shapes.query, *shapes.key, tilingData)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check prefix failed!");
        return ge::GRAPH_FAILED;
    }

    // actSeq check
    if (!CheckActSeq(contextKeyParams, *shapes.query)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check actual sequence failed!");
        return ge::GRAPH_FAILED;
    }

    // actSeqLen check
    if ((contextKeyParams.fromTilingSink == static_cast<uint32_t>(0)) &&
        !CheckActSeqLen(contextKeyParams, *shapes.query, *shapes.key)) {
        OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "Check actual sequence length failed!");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckPAStage(FiaArch38ContextParams& contextKeyParams, PFAShapeSet& shapes,
                                                    FiaArch38FlashAttentionTilingData& tilingData)
{
    if (!enablePA) {
        if (enableTensorList) {
            S2 = contextKeyParams.maxKVs;
        } else {
            S2 = shapes.key->s;
        }
    } else if (!CheckPATypeAndShape(contextKeyParams, *shapes.query, *shapes.queryRope, tilingData)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckPseShiftStage(FiaArch38ContextParams& contextKeyParams, PFAShapeSet& shapes)
{
    if (enablePseShift) {
        if (!CheckPseShiftTypeAndShape(contextKeyParams, shapes.query->b, shapes.query->n, shapes.query->s, S2)) {
            return ge::GRAPH_FAILED;
        }
        usePseShift = 1;
    } else if (enableAlibiPse) {
        if (!CheckAlibiPseShiftTypeAndShape(contextKeyParams, shapes.query->n)) {
            return ge::GRAPH_FAILED;
        }
        if (pseType == PSE_TYPE_2_TILING_V2) {
            usePseShift = static_cast<uint32_t>(PSE_TYPE_2_TILING_V2);
        } else if (pseType == PSE_TYPE_3_TILING_V2) {
            usePseShift = static_cast<uint32_t>(PSE_TYPE_3_TILING_V2);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckCrossoverAttribute(FiaArch38ContextParams& contextKeyParams,
                                                               PFAShapeInfo& queryShapeInfo,
                                                               std::vector<int64_t>& actualSeqLengths,
                                                               std::vector<int64_t>& actualSeqLengthsKV,
                                                               FiaArch38FlashAttentionTilingData& tilingData)
{
    // PA and prefix,antiquant,actseqlenKV features crossover
    if (!CheckPACrossover(contextKeyParams, queryShapeInfo)) {
        return ge::GRAPH_FAILED;
    }

    // Mask and innerprecise,sparseMode feature crossover
    if (!CheckMaskCrossover(contextKeyParams, queryShapeInfo, tilingData)) {
        return ge::GRAPH_FAILED;
    }

    // TNDLayout and left padding, tensorlist, pse feature crossover
    if (!CheckTNDLayoutCrossover(contextKeyParams)) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckAlibiPseCrossover(contextKeyParams)) {
        return ge::GRAPH_FAILED;
    }

    if ((contextKeyParams.fromTilingSink == 0) &&
        !CheckMultiFeatureCrossover(contextKeyParams, queryShapeInfo, actualSeqLengths, actualSeqLengthsKV,
                                    tilingData)) {
        return ge::GRAPH_FAILED;
    }

    if (!CheckPerblockCrossover(contextKeyParams)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::ComputeTilingData(FiaArch38ContextParams& contextKeyParams,
                                                         std::vector<int64_t>& actualSeqLengths,
                                                         std::vector<int64_t>& actualSeqLengthsKV,
                                                         FiaArch38FlashAttentionTilingData& tilingData)
{
    // Compute tiling data.
    bool isAttenMaskUsed = (contextKeyParams.attentionMaskShape != nullptr);
    FiaArch38SplitNBSeq(tilingData, actualSeqLengths, actualSeqLengthsKV, isAttenMaskUsed);

    if (needInit == 1) {
        FiaArch38InitOutputSplit(contextKeyParams.outputShape->GetStorageShape().GetShapeSize(), tilingData);
    }

    if (contextKeyParams.isSoftMaxLseEnable) {
        FiaArch38InitSoftmaxLseOutputSplit(contextKeyParams.lseoutputShape->GetStorageShape().GetShapeSize(),
                                           tilingData);
    }

    if (enableIFA && !enablePFAMerge) {
        FiaArch38AttentionSingleCoreParams* singleCoreParams = &tilingData.fiaArch38AttentionSingleCoreParams;
        uint32_t sOuterSize = singleCoreParams->get_singleProcessSOuterSize();
        uint64_t batchSize = tilingData.fiaArch38AttentionBaseParams.get_batchSize();
        uint64_t headNumKVSize = tilingData.fiaArch38AttentionBaseParams.get_headNumSize(); // IFA kv N
        uint64_t bng = batchSize * headNumKVSize * (gSize + sOuterSize - 1) / sOuterSize;
        if (IsFlashDecode()) {
            enableFlashDecode = true;
            return SplitBNS(tilingData, bng);
        }
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::ComputeTilingKey(uint64_t& tilingKey, FiaArch38ContextParams& contextKeyParams,
                                                        uint32_t& numBlocksToBeSet,
                                                        FiaArch38FlashAttentionTilingData& tilingData)
{
    bool tilingRet = TilingGetTilingKeyAttentionAscendC(tilingKey, contextKeyParams, tilingData);
    OP_CHECK_IF(!tilingRet, ReportTilingFailure(context_, "Get tilingKey fail"), return ge::GRAPH_FAILED);

    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, ReportNullPointer(context_, __func__, "platformInfoPtr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    numBlocksToBeSet = ascendcPlatform.CalcTschBlockDim(aivNum, aicNum, aivNum);

    size_t* workspaces = contextKeyParams.workspaceSize;
    workspaces[0] = GetPFAWorkSpaceSize(tilingData);
    OP_LOGI(contextKeyParams.opName, "The Tiling key is %lu", tilingKey);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::FiaArch38SetTilingData(gert::TilingContext* context)
{
    faTilingAdapter.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(faTilingAdapter.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::RunBigKernelTilingWithParams(FiaArch38ContextParams& contextKeyParams,
                                                                    uint64_t& tilingKey, uint32_t& numBlocksToBeSet,
                                                                    FiaArch38FlashAttentionTilingData& tilingData)
{
    // set memory parameters
    if (SetPlatMemoryInfo(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // set sttributes switch
    if (SetAttributeInfo(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // rope ptr is empty or rope is invalid
    if (CheckRopeInvalid(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // tensor ptr is empty or tensor is invalid
    if (CheckTensorInvalid(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    } else if (CheckEmptyTensor(contextKeyParams)) {
        return SetEmptyTensor(contextKeyParams, tilingKey, numBlocksToBeSet, tilingData);
    }
    // Input, output check
    PFAShapeInfo queryShapeInfo;
    PFAShapeInfo keyShapeInfo;
    PFAShapeInfo valueShapeInfo;
    PFAShapeInfo queryRopeShapeInfo;

    // Check kinds of single attribute
    PFAShapeSet shapes;
    shapes.query = &queryShapeInfo;
    shapes.key = &keyShapeInfo;
    shapes.value = &valueShapeInfo;
    shapes.queryRope = &queryRopeShapeInfo;
    if (CheckSingleAttribute(contextKeyParams, shapes, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    // 同IFA合轴实现
    if (enablePFAMerge) {
        if (*contextKeyParams.numKeyValueHeads > 0) {
            gSize = (*contextKeyParams.headsNumber) / (*contextKeyParams.numKeyValueHeads);
        } else {
            gSize = 1U;
        }
        tilingData.fiaArch38AttentionBaseParams.set_headNumRatio(1);
        tilingData.fiaArch38AttentionBaseParams.set_gOfMla(gSize);
    }
    if (enableIFAMLA || enableIFA || enablePFAMerge) {
        queryShapeInfo.n = queryShapeInfo.n / gSize;
        queryShapeInfo.s = queryShapeInfo.s * gSize;
    }

    // multi feature crossover check
    std::vector<int64_t> actualSeqLengths(queryShapeInfo.b);
    std::vector<int64_t> actualSeqLengthsKV(queryShapeInfo.b);

    // Check crossover attribute
    if (CheckCrossoverAttribute(contextKeyParams, queryShapeInfo, actualSeqLengths, actualSeqLengthsKV, tilingData) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // needInit is available only after CheckCrossoverAttribute, so the scenario gate is placed here.
    if (!CheckArch38ScenarioSupported(contextKeyParams, queryShapeInfo, valueShapeInfo)) {
        return ge::GRAPH_FAILED;
    }

    // print shape info
    OP_LOGI(contextKeyParams.opName,
            "inputLayout is %s, innerPrecise is %d, scaleValue is %f, preTokens is %ld, nextTokens is %ld",
            string(contextKeyParams.layout).c_str(), innerPrecise, *contextKeyParams.scaleValue,
            *contextKeyParams.preToken, *contextKeyParams.nextToken);

    // set tilingData
    SetTilingData(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, valueShapeInfo, tilingData);

    // Infering whether the tiling mode is S2 full load, CV diff, and whether to use the matmul norm template.
    InferTilingMod(contextKeyParams, actualSeqLengths, actualSeqLengthsKV, queryShapeInfo.b, queryShapeInfo.d);
    InferSplitCoreMode();
    // Whether to enable constant templates
    InferConstantization();

    if (AdjustTilingData(contextKeyParams, tilingData, queryShapeInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // DN check
    GetEnableDN(tilingData, queryShapeInfo, valueShapeInfo, actualSeqLengths, actualSeqLengthsKV);

    if (ComputeTilingData(contextKeyParams, actualSeqLengths, actualSeqLengthsKV, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // Compute tiling key.
    if (ComputeTilingKey(tilingKey, contextKeyParams, numBlocksToBeSet, tilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    PFATilingDataconvert(tilingData);
    if (SetQKVStartIdx(contextKeyParams) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::DoSubOpTiling(FiaArch38FlashAttentionTilingData& tilingData,
                                                     FiaArch38ContextParams& contextParamsForPFATiling)
{
    uint64_t tilingKey = 7;
    uint32_t numBlocksToBeSet;
    auto ret = RunBigKernelTilingWithParams(contextParamsForPFATiling, tilingKey, numBlocksToBeSet, tilingData);
    tilingKey += BENCHMARK_TILING_KEY;
    context_->SetTilingKey(tilingKey);
    context_->SetBlockDim(numBlocksToBeSet);
    FiaArch38SetTilingData(context_);
    // 使用SyncAll，需要设置为batchmode模式，所有核同时启动，否则多流方式下执行可能会卡死
    context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    return ret;
}

} // namespace arch38
} // namespace optiling
