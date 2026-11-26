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

void FiaArch38PromptTiling::FiaArch38InitOutputSplit(int64_t totalSize, FiaArch38FlashAttentionTilingData& tilingData)
{
    FiaArch38AttentionInitOutputParams* initParams = &tilingData.fiaArch38AttentionInitOutputParams;
    // Upward rounding, coreNum has been verified to be non-zero when obtained.
    uint32_t singleCoreSize = (totalSize + coreNum - 1) / (coreNum);

    if (enablePostQuant) {
        // requiring that the number of points allocated to each kernel must be even.
        singleCoreSize = ((singleCoreSize + 1) / 2) * 2; // 2 : fill in 0
    }

    initParams->set_singleCoreSize(singleCoreSize);
    initParams->set_totalOutputSize(totalSize);
}

bool FiaArch38PromptTiling::CheckEmptyTensor(FiaArch38ContextParams& contextKeyParams) const
{
    return (contextKeyParams.keyInputShape->GetStorageShape().GetShapeSize() == 0) ||
           (contextKeyParams.valueInputShape->GetStorageShape().GetShapeSize() == 0) ||
           (contextKeyParams.outputShape->GetStorageShape().GetShapeSize() == 0) ||
           (contextKeyParams.emptyTensor == 1U);
}

ge::graphStatus FiaArch38PromptTiling::SetEmptyTensor(FiaArch38ContextParams& contextKeyParams, uint64_t& tilingKey,
                                                      uint32_t& numBlocksToBeSet,
                                                      FiaArch38FlashAttentionTilingData& tilingData)
{
    tilingKey = EMPTY_KV_TILING_KEY;
    FiaArch38InitOutputSplit(contextKeyParams.outputShape->GetStorageShape().GetShapeSize(), tilingData);
    tilingData.fiaArch38AttentionInitOutputParams.set_needInit(1);

    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, ReportNullPointer(context_, __func__, "platformInfoPtr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    numBlocksToBeSet = ascendcPlatform.CalcTschBlockDim(coreNum, aicNum, coreNum);

    size_t* workspace = contextKeyParams.workspaceSize;
    const size_t sysWorkspaceSize = 16 * 1024 * 1024; // minimum size required by workspace
    workspace[0] = sysWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}

bool FiaArch38PromptTiling::CheckIODataType(FiaArch38ContextParams& contextKeyParams)
{
    outputType = contextKeyParams.outputDataType;
    inputType = contextKeyParams.inputDataType;

    std::vector<ge::DataType> allowedDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_HIFLOAT8,
                                               ge::DT_FLOAT8_E4M3FN};
    std::vector<uint32_t> dataTypeSizeArray = {FLOAT16_SIZE, BFLOAT16_SIZE, INT8_SIZE,
                                               FLOAT8_SIZE,  FLOAT8_SIZE,   FLOAT8_SIZE};

    auto inputTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), inputType);
    auto outputTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), outputType);
    if (inputTypeCheck != allowedDtypes.end()) {
        uint32_t inputTypeIndex = std::distance(allowedDtypes.begin(), inputTypeCheck);
        dataTypeSize = dataTypeSizeArray[inputTypeIndex];
    }
    maskElemSize = dataTypeSize;
    if (outputTypeCheck != allowedDtypes.end()) {
        uint32_t outputTypeIndex = std::distance(allowedDtypes.begin(), outputTypeCheck);
        outputDataTypeSize = dataTypeSizeArray[outputTypeIndex];
    }

    if (inputType != ge::DT_FLOAT16) {
        OP_LOGW(contextKeyParams.opName, "innerprecise will not take effect when input type is %s!",
                GetPfaDataTypeStr(inputType).c_str());
    }
    OP_CHECK_IF((inputTypeCheck == allowedDtypes.end() || outputTypeCheck == allowedDtypes.end()),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    contextKeyParams.opName, "inputType, outputType",
                    (GetPfaDataTypeStr(inputType) + ", " + GetPfaDataTypeStr(outputType)).c_str(),
                    "inputType and outputType should be DT_FLOAT16, DT_BF16 or DT_INT8"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::SetInputLayout(const char* layout)
{
    if (layout == nullptr) {
        inputLayout = InputLayout::BSH;
        return true;
    }

    std::string layoutStr(layout);
    if (layoutStr == "" || layoutStr == "BSH") {
        inputLayout = InputLayout::BSH;
    } else if (layoutStr == "TND") {
        inputLayout = InputLayout::TND;
    } else if (layoutStr == "BSND") {
        inputLayout = InputLayout::BSND;
    } else if (layoutStr == "BNSD" || layoutStr == "BNSD_BSND") { // Reuse BNSD process for BNSD_BSND
        inputLayout = InputLayout::BNSD;
    } else {
        return false;
    }

    return true;
}

int64_t GetMaxSeq(const gert::Tensor* actualSeqLength)
{
    int64_t max = actualSeqLength->GetData<int64_t>()[0];
    for (int i = 1; i < actualSeqLength->GetShapeSize(); ++i) {
        max = std::max(max, actualSeqLength->GetData<int64_t>()[i] - actualSeqLength->GetData<int64_t>()[i - 1]);
    }
    return max;
}

int64_t GetKVHeadNum(const FiaArch38ContextParams& contextKeyParams)
{
    return (*contextKeyParams.numKeyValueHeads > 0) ? *contextKeyParams.numKeyValueHeads :
                                                      *contextKeyParams.headsNumber;
}

bool FiaArch38PromptTiling::SetShape(FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* shape,
                                     const std::string inputName, PFAShapeDims& dims) const
{
    if (enablePA && (inputName != "query")) {
        return SetShapePA(contextKeyParams, shape, dims);
    }
    switch (inputLayout) {
        case InputLayout::BNSD:
            return SetShapeBNSD(shape, dims);
        case InputLayout::BSH:
            return SetShapeBSH(contextKeyParams, shape, inputName, dims);
        case InputLayout::BSND:
            return SetShapeBSND(shape, dims);
        case InputLayout::TND:
            return SetShapeTND(contextKeyParams, shape, inputName, dims);
        default:
            return false;
    }
}

bool FiaArch38PromptTiling::SetShapePA(const FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* shape,
                                       PFAShapeDims& dims) const
{
    // when enablePA, k_v has two shapes:
    // 1. [>=blockNums, N, blockSize, D]
    // 2. [>=blockNums, blockSize, H]
    if (shape->GetStorageShape().GetDimNum() == 3) { // 3 for dim num
        dims.b = shape->GetStorageShape().GetDim(0);
        dims.s = shape->GetStorageShape().GetDim(1);
        dims.h = shape->GetStorageShape().GetDim(2); // 2 for H dim
        dims.n = GetKVHeadNum(contextKeyParams);
        dims.d = dims.n > 0 ? dims.h / dims.n : 0;
    } else {
        dims.b = shape->GetStorageShape().GetDim(0);
        dims.n = shape->GetStorageShape().GetDim(1);
        dims.s = shape->GetStorageShape().GetDim(2); // 2 for Sequence length
        dims.d = shape->GetStorageShape().GetDim(3); // 3 for D dim
        dims.h = dims.n * dims.d;
    }
    return true;
}

bool FiaArch38PromptTiling::SetShapeBNSD(const gert::StorageShape* shape, PFAShapeDims& dims) const
{
    dims.b = shape->GetStorageShape().GetDim(0);
    dims.n = shape->GetStorageShape().GetDim(1);
    dims.s = shape->GetStorageShape().GetDim(2); // 2 for Sequence length
    dims.d = shape->GetStorageShape().GetDim(3); // 3 for D dim
    dims.h = dims.n * dims.d;
    return true;
}

bool FiaArch38PromptTiling::SetShapeBSH(const FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* shape,
                                        const std::string& inputName, PFAShapeDims& dims) const
{
    dims.b = shape->GetStorageShape().GetDim(0);
    dims.s = shape->GetStorageShape().GetDim(1);
    dims.h = shape->GetStorageShape().GetDim(2); // 2 for H dim
    dims.n =
        (inputName == "query") ? static_cast<int64_t>(*contextKeyParams.headsNumber) : GetKVHeadNum(contextKeyParams);
    dims.d = dims.n > 0 ? dims.h / dims.n : 0;
    return true;
}

bool FiaArch38PromptTiling::SetShapeBSND(const gert::StorageShape* shape, PFAShapeDims& dims) const
{
    dims.b = shape->GetStorageShape().GetDim(0);
    dims.s = shape->GetStorageShape().GetDim(1);
    dims.n = shape->GetStorageShape().GetDim(2); // 2 for head dim
    dims.d = shape->GetStorageShape().GetDim(3); // 3 for D dim
    dims.h = dims.n * dims.d;
    return true;
}

bool FiaArch38PromptTiling::SetShapeTND(const FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* shape,
                                        const std::string& inputName, PFAShapeDims& dims) const
{
    OP_CHECK_IF(
        (contextKeyParams.actualSequenceLengthQ == nullptr) || (contextKeyParams.actualSequenceLengthKV == nullptr) ||
            (contextKeyParams.actualSequenceLengthQ->GetData<int64_t>() == nullptr) ||
            (contextKeyParams.actualSequenceLengthKV->GetData<int64_t>() == nullptr),
        OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "actual_seq_lengths, actual_seq_lengths_kv"), return false);

    dims.b = static_cast<int64_t>(contextKeyParams.actualSequenceLengthQ->GetShapeSize());
    dims.s = (inputName == "query") ? GetMaxSeq(contextKeyParams.actualSequenceLengthQ) :
                                      GetMaxSeq(contextKeyParams.actualSequenceLengthKV);
    dims.t = shape->GetStorageShape().GetDim(0);
    dims.n = shape->GetStorageShape().GetDim(1);
    dims.d = shape->GetStorageShape().GetDim(2); // 2 for D dim
    dims.h = dims.n * dims.d;
    return true;
}

bool FiaArch38PromptTiling::GetAndCheckShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& shapeInfo,
                                             const gert::StorageShape* shape, const std::string& sName) const
{
    std::string layoutStr(contextKeyParams.layout);
    OP_CHECK_IF(
        (shape->GetStorageShape().GetDimNum() != 3) &&
            (inputLayout == InputLayout::BSH || inputLayout == InputLayout::TND),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            contextKeyParams.opName, sName.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
            (std::string("dim num of ") + sName + " must be 3 when inputLayout is " + layoutStr).c_str()),
        return false);
    OP_CHECK_IF(
        (shape->GetStorageShape().GetDimNum() != 4) &&
            (inputLayout != InputLayout::BSH && inputLayout != InputLayout::TND),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            contextKeyParams.opName, sName.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
            (std::string("dim num of ") + sName + " must be 4 when inputLayout is " + layoutStr).c_str()),
        return false);

    PFAShapeDims dims;
    if (!SetShape(contextKeyParams, shape, sName, dims)) {
        return false;
    }
    if (!CheckShapeSizeLimits(contextKeyParams, sName, dims)) {
        return false;
    }
    shapeInfo.b = static_cast<uint32_t>(dims.b);
    shapeInfo.n = static_cast<uint64_t>(dims.n);
    shapeInfo.s = static_cast<uint64_t>(dims.s);
    shapeInfo.d = static_cast<uint32_t>(dims.d);
    shapeInfo.h = static_cast<uint64_t>(dims.h);
    shapeInfo.t = static_cast<uint32_t>(dims.t);
    return true;
}

bool FiaArch38PromptTiling::CheckShapeSizeLimits(const FiaArch38ContextParams& contextKeyParams,
                                                 const std::string& sName, const PFAShapeDims& dims) const
{
    OP_CHECK_IF((dims.b > BLIMIT || dims.b <= 0),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(),
                                                          std::to_string(dims.b).c_str(),
                                                          "batch size of tensor must be within (0, BLIMIT]"),
                return false);
    OP_CHECK_IF((dims.s <= 0),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(),
                                                          std::to_string(dims.s).c_str(),
                                                          "seq size of tensor must be greater than 0"),
                return false);
    if (dims.s > SLIMIT) {
        OP_LOGW(contextKeyParams.opName, "sequence size of %s should <= 20M, but seq = %ld.", sName.c_str(), dims.s);
    }
    if (!CheckShapeDAndNLimits(contextKeyParams, sName, dims)) {
        return false;
    }
    if (((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND)) && (dims.h > HLIMIT)) {
        OP_LOGW(contextKeyParams.opName,
                "layout = %s, h(%ld) size of %s should be less than or equal to 65535, "
                "which may cause precision problem! Please use BNSD or BNSD_BSND instead.",
                contextKeyParams.layout, dims.h, sName.c_str());
    }
    OP_CHECK_IF((inputLayout == InputLayout::TND) && (dims.t <= 0),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(),
                                                          std::to_string(dims.t).c_str(),
                                                          "when layout is TND, t must be greater than 0"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckShapeDAndNLimits(const FiaArch38ContextParams& contextKeyParams,
                                                  const std::string& sName, const PFAShapeDims& dims) const
{
    if (inputLayout == InputLayout::BSH) {
        OP_CHECK_IF((dims.d > DLIMIT || dims.d <= 0),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(),
                                                              std::to_string(dims.d).c_str(),
                                                              "d size of tensor must be within (0, DLIMIT]"),
                    return false);
        OP_CHECK_IF((dims.n > NLIMIT || dims.n <= 0),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(),
                                                              std::to_string(dims.n).c_str(),
                                                              "n size of tensor must be within (0, NLIMIT]"),
                    return false);
    } else {
        OP_CHECK_IF((dims.n > NLIMIT || dims.n <= 0),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(),
                                                              std::to_string(dims.n).c_str(),
                                                              "n size of tensor must be within (0, NLIMIT]"),
                    return false);
        OP_CHECK_IF((dims.d > DLIMIT || dims.d <= 0),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, sName.c_str(),
                                                              std::to_string(dims.d).c_str(),
                                                              "d size of tensor must be within (0, DLIMIT]"),
                    return false);
    }
    return true;
}

bool FiaArch38PromptTiling::GetAndCheckRopeShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& shapeInfo,
                                                 PFAShapeInfo& ropeShapeInfo, const gert::StorageShape* shape,
                                                 const RopeNamePair& names) const
{
    if (!CheckRopeShapeDim(contextKeyParams, shape, names.srcName, shapeInfo)) {
        return false;
    }

    PFAShapeDims dims;
    if (!SetShape(contextKeyParams, shape, names.srcName, dims)) {
        return false;
    }
    int64_t b = dims.b;
    int64_t n = dims.n;
    int64_t s = dims.s;
    int64_t d = dims.d;
    int64_t h = dims.h;
    int64_t t = dims.t;

    if (!CheckRopeDimsD(contextKeyParams, names.ropeName, dims)) {
        return false;
    }

    if (!CheckRopeShapeConsistency(contextKeyParams, names.srcName, names.ropeName, shapeInfo, dims)) {
        return false;
    }

    if (((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND)) &&
        (static_cast<uint32_t>(h) > HLIMIT)) {
        OP_LOGW(contextKeyParams.opName,
                "layout = %s, h(%ld) size of %s should be less than or equal to 65535, "
                "which may cause precision problem! Please use BNSD or BNSD_BSND instead.",
                contextKeyParams.layout, h, names.srcName.c_str());
    }
    ropeShapeInfo.b = static_cast<uint32_t>(b);
    ropeShapeInfo.n = static_cast<uint64_t>(n);
    ropeShapeInfo.s = static_cast<uint64_t>(s);
    ropeShapeInfo.d = static_cast<uint32_t>(d);
    ropeShapeInfo.h = static_cast<uint64_t>(h);
    ropeShapeInfo.t = static_cast<uint32_t>(t);
    return true;
}

bool FiaArch38PromptTiling::CheckRopeShapeDim(const FiaArch38ContextParams& contextKeyParams,
                                              const gert::StorageShape* shape, const std::string& sName,
                                              const PFAShapeInfo& shapeInfo) const
{
    std::string layoutStr(contextKeyParams.layout);
    OP_CHECK_IF(
        (shape->GetStorageShape().GetDimNum() != 3) &&
            (inputLayout == InputLayout::BSH || inputLayout == InputLayout::TND),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            contextKeyParams.opName, sName.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
            (std::string("dim num of ") + sName + " must be 3 when inputLayout is " + layoutStr).c_str()),
        return false);
    OP_CHECK_IF(
        (shape->GetStorageShape().GetDimNum() != 4) &&
            (inputLayout != InputLayout::BSH && inputLayout != InputLayout::TND),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            contextKeyParams.opName, sName.c_str(), std::to_string(shape->GetStorageShape().GetDimNum()).c_str(),
            (std::string("dim num of ") + sName + " must be 4 when inputLayout is " + layoutStr).c_str()),
        return false);

    OP_CHECK_IF((shapeInfo.d != 128 && shapeInfo.d != 512),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    contextKeyParams.opName, sName.c_str(), std::to_string(shapeInfo.d).c_str(),
                    "when queryRope and keyRope is not null, d must be 128 or 512"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckRopeDimsD(const FiaArch38ContextParams& contextKeyParams, const std::string& rName,
                                           const PFAShapeDims& dims) const
{
    int64_t d = dims.d;
    if (inputLayout == InputLayout::BSH) {
        OP_CHECK_IF((d != 64),
                    OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                        contextKeyParams.opName, rName.c_str(), std::to_string(d).c_str(),
                        "when queryRope and keyRope is not null, d must be 64 when layout is BSH"),
                    return false);
    } else {
        OP_CHECK_IF(
            (d != 64),
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, rName.c_str(), std::to_string(d).c_str(),
                                                      "when queryRope and keyRope is not null, d must be 64"),
            return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckRopeShapeConsistency(const FiaArch38ContextParams& contextKeyParams,
                                                      const std::string& sName, const std::string& rName,
                                                      const PFAShapeInfo& shapeInfo, const PFAShapeDims& dims) const
{
    if (!enableTensorList || sName == "query") {
        OP_CHECK_IF((dims.b != shapeInfo.b),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        contextKeyParams.opName, (rName + ", " + sName).c_str(),
                        (std::to_string(dims.b) + ", " + std::to_string(shapeInfo.b)).c_str(),
                        "batch size of rope must be equal to batch size of tensor"),
                    return false);
    }
    OP_CHECK_IF((dims.n != shapeInfo.n),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, (rName + ", " + sName).c_str(),
                    (std::to_string(dims.n) + ", " + std::to_string(shapeInfo.n)).c_str(),
                    "head number of rope must be equal to head number of tensor"),
                return false);
    OP_CHECK_IF((dims.s != shapeInfo.s),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, (rName + ", " + sName).c_str(),
                    (std::to_string(dims.s) + ", " + std::to_string(shapeInfo.s)).c_str(),
                    "seq length of rope must be equal to seq length of tensor"),
                return false);
    OP_CHECK_IF((dims.t != shapeInfo.t),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, (rName + ", " + sName).c_str(),
                    (std::to_string(dims.t) + ", " + std::to_string(shapeInfo.t)).c_str(),
                    "t of rope must be equal to t of tensor"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckQueryOutParamsConsistency(const FiaArch38ContextParams& contextKeyParams,
                                                           const gert::StorageShape* queryShape,
                                                           const gert::StorageShape* outShape) const
{
    const size_t queryDimNum = queryShape->GetStorageShape().GetDimNum();
    const size_t outDimNum = outShape->GetStorageShape().GetDimNum();
    int64_t tmpqueryDim = 0;
    int64_t outDim = 0;
    std::string layoutStr(contextKeyParams.layout);

    OP_CHECK_IF(queryDimNum != outDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    contextKeyParams.opName, "query, attention_out",
                    (std::to_string(queryDimNum) + ", " + std::to_string(outDimNum)).c_str(),
                    "query dim num must be consistent with output dim num"),
                return false);
    OP_CHECK_IF(
        (queryDimNum < INPUT_QKV_SHAPE_MIN_DIMS) || (queryDimNum > INPUT_QKV_SHAPE_MAX_DIMS),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "query", std::to_string(queryDimNum).c_str(),
                                                 "query dim num is invalid, only support 3 or 4 dimensions"),
        return false);

    for (uint32_t i = 0; i < queryDimNum; ++i) {
        if ((i == queryDimNum - 1) && enablePFAMLA) {
            continue;
        }
        if (layoutStr == "BNSD_BSND") {
            if (i == 1) { // query:N, output:S
                tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
                outDim = outShape->GetStorageShape().GetDim(i + 1);
            } else if (i == 2) { // 2 for current queryDimNum; Q:N, Output:S
                tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
                outDim = outShape->GetStorageShape().GetDim(i - 1);
            }
        } else {
            tmpqueryDim = queryShape->GetStorageShape().GetDim(i);
            outDim = outShape->GetStorageShape().GetDim(i);
        }
        OP_CHECK_IF(tmpqueryDim != outDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        contextKeyParams.opName, "query, attention_out",
                        (std::to_string(tmpqueryDim) + ", " + std::to_string(outDim)).c_str(),
                        "query and output shape must be equal in the same dim"),
                    return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckKVDataType(FiaArch38ContextParams& contextKeyParams) const
{
    ge::DataType keyDataType = contextKeyParams.kDataType;
    ge::DataType valueDataType = contextKeyParams.vDataType;

    std::vector<ge::DataType> allowedDtypes = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_INT8, ge::DT_HIFLOAT8,
                                               ge::DT_FLOAT8_E4M3FN};

    auto keyTypeCheck = std::find(allowedDtypes.begin(), allowedDtypes.end(), keyDataType);

    OP_CHECK_IF((keyDataType != valueDataType),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(contextKeyParams.opName, "key, value",
                                                       (GetPfaDataTypeStr(contextKeyParams.kDataType) + ", " +
                                                        GetPfaDataTypeStr(contextKeyParams.vDataType))
                                                           .c_str(),
                                                       "key and value must have the same data type"),
                return false);
    OP_CHECK_IF((keyTypeCheck == allowedDtypes.end()),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "key",
                                                      GetPfaDataTypeStr(contextKeyParams.kDataType).c_str(),
                                                      "key dtype only supports DT_FLOAT16, DT_BF16 or DT_INT8"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckKeyValueParamsConsistency(FiaArch38ContextParams& contextKeyParams,
                                                           const gert::StorageShape* keyShape,
                                                           const gert::StorageShape* valueShape) const
{
    if (enableTensorList) {
        return true;
    }
    const size_t keyDimNum = keyShape->GetStorageShape().GetDimNum();
    const size_t valueDimNum = valueShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(keyDimNum != valueDimNum,
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    contextKeyParams.opName, "key, value",
                    (std::to_string(keyDimNum) + ", " + std::to_string(valueDimNum)).c_str(),
                    "key dim num must be consistent with value dim num"),
                return false);
    OP_CHECK_IF(
        (keyDimNum < INPUT_QKV_SHAPE_MIN_DIMS) || (keyDimNum > INPUT_QKV_SHAPE_MAX_DIMS),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "key", std::to_string(keyDimNum).c_str(),
                                                 "key dim num is invalid, only support 3 or 4 dimensions"),
        return false);
    for (uint32_t i = 0; i < keyDimNum; ++i) {
        if ((i == keyDimNum - 1) && enablePFAMLA) {
            continue;
        }
        int64_t tmpKeyDim = keyShape->GetStorageShape().GetDim(i);
        int64_t tmpValueDim = valueShape->GetStorageShape().GetDim(i);
        OP_CHECK_IF(tmpKeyDim != tmpValueDim,
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        contextKeyParams.opName, "key, value",
                        (std::to_string(tmpKeyDim) + ", " + std::to_string(tmpValueDim)).c_str(),
                        "key and value shape must be equal in the same dim"),
                    return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckInputDimAndHeadNum(FiaArch38ContextParams& contextKeyParams, const uint32_t nQAttr,
                                                    const uint32_t nKVAttr)
{
    uint32_t nQ = nQAttr;
    uint32_t nKV = nKVAttr;
    if (nKVAttr == 0U) { // Detected that nKVAttr is the default value, which means that the customer did not pass in.
        nKV = nQAttr;
    }

    uint32_t queryShapeHeadNum = nQ;
    uint32_t keyShapeHeadNum = nKV;
    uint32_t valueShapeHeadNum = nKV;
    if (!ReadQkvShapeHeadNum(contextKeyParams, queryShapeHeadNum, keyShapeHeadNum, valueShapeHeadNum)) {
        return false;
    }

    OP_CHECK_IF(nQ > 256U, // The maximum limit for head is 256.
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "num_heads", std::to_string(nQ).c_str(),
                                                      "numHeads should not be more than 256"),
                return false);
    OP_CHECK_IF(
        queryShapeHeadNum != nQ,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "query, num_heads",
                                               (std::to_string(queryShapeHeadNum) + ", " + std::to_string(nQ)).c_str(),
                                               "numHeads in query shape must be equal to numHeads in attr"),
        return false);
    OP_CHECK_IF(
        keyShapeHeadNum != nKV,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "key, num_key_value_heads",
                                               (std::to_string(keyShapeHeadNum) + ", " + std::to_string(nKV)).c_str(),
                                               "numHeads in key shape must match numKeyValueHeads in attr"),
        return false);
    OP_CHECK_IF(
        valueShapeHeadNum != nKV,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "value, num_key_value_heads",
                                               (std::to_string(valueShapeHeadNum) + ", " + std::to_string(nKV)).c_str(),
                                               "numHeads in value shape must match numKeyValueHeads in attr"),
        return false);
    if (enableIFAMLA || enableIFA) {
        gSize = nQ / nKV;
    }
    if ((nQ / nKV) > pfaMergeGLimit) {
        enablePFAMerge = false;
    }
    return true;
}

bool FiaArch38PromptTiling::ReadQkvShapeHeadNum(const FiaArch38ContextParams& contextKeyParams,
                                                uint32_t& queryShapeHeadNum, uint32_t& keyShapeHeadNum,
                                                uint32_t& valueShapeHeadNum)
{
    const gert::StorageShape* queryShape = contextKeyParams.queryInputShape;
    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const size_t queryDim = queryShape->GetStorageShape().GetDimNum();
    const size_t keyDim = keyShape->GetStorageShape().GetDimNum();
    const size_t valueDim = valueShape->GetStorageShape().GetDimNum();
    const size_t nIdx =
        (inputLayout == InputLayout::BNSD || inputLayout == InputLayout::TND) ? 1U : 2U; // BNSD/TND: 1; BSND:2

    if (((inputLayout == InputLayout::BNSD) || (inputLayout == InputLayout::BSND)) && (!enablePA)) {
        if ((queryDim == 4) && (keyDim == 4) && (valueDim == 4)) { // dim num: 4
            queryShapeHeadNum = queryShape->GetStorageShape().GetDim(nIdx);
            keyShapeHeadNum = keyShape->GetStorageShape().GetDim(nIdx);
            valueShapeHeadNum = valueShape->GetStorageShape().GetDim(nIdx);
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                contextKeyParams.opName, "query, key, value",
                (std::to_string(queryDim) + ", " + std::to_string(keyDim) + ", " + std::to_string(valueDim)).c_str(),
                "query, key and value must be 4D for BNSD or BSND format");
            return false;
        }
    } else if ((inputLayout == InputLayout::TND) && (!enablePA)) {
        if ((queryDim == 3) && (keyDim == 3) && (valueDim == 3)) { // dim num: 3
            queryShapeHeadNum = queryShape->GetStorageShape().GetDim(nIdx);
            keyShapeHeadNum = keyShape->GetStorageShape().GetDim(nIdx);
            valueShapeHeadNum = valueShape->GetStorageShape().GetDim(nIdx);
        } else {
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                contextKeyParams.opName, "query, key, value",
                (std::to_string(queryDim) + ", " + std::to_string(keyDim) + ", " + std::to_string(valueDim)).c_str(),
                "query, key and value must be 3D for TND format");
            return false;
        }
    }
    return true;
}

bool FiaArch38PromptTiling::SetAndCheckHeadNumRatio(FiaArch38ContextParams& contextKeyParams,
                                                    FiaArch38FlashAttentionTilingData& tilingData)
{
    const int64_t nQ = *contextKeyParams.headsNumber;
    const int64_t nKV = *contextKeyParams.numKeyValueHeads;

    if ((nQ < 0) || (nKV < 0)) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "num_heads, num_key_value_heads",
                                               (std::to_string(nQ) + ", " + std::to_string(nKV)).c_str(),
                                               "numHeads and numKeyValueHeads must not be negative");
        return false;
    }

    if (!CheckInputDimAndHeadNum(contextKeyParams, nQ, nKV)) {
        return false;
    }

    if (nKV == 0) { // Detected that nKV is the default value, which means that the customer did not pass in.
        tilingData.fiaArch38AttentionBaseParams.set_headNumRatio(1);
        tilingData.fiaArch38AttentionBaseParams.set_gOfMla(1);
        return true;
    }

    if (enableIFAMLA || enableIFA) {
        tilingData.fiaArch38AttentionBaseParams.set_headNumRatio(1);
        tilingData.fiaArch38AttentionBaseParams.set_gOfMla(gSize);
        return true;
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_gOfMla(1);
    }

    if (nQ % nKV != 0) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "num_heads, num_key_value_heads",
                                               (std::to_string(nQ) + ", " + std::to_string(nKV)).c_str(),
                                               "numHeads must be divisible by numKeyValueHeads");
        return false;
    } else {
        if (nQ / nKV > 64 && (!enablePFAMLA)) { // G cannot be greater than 64.
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "num_heads / num_key_value_heads",
                                                  std::to_string(nQ / nKV).c_str(),
                                                  "numHeads / numKeyValueHeads must not be larger than 64");
            return false;
        }
        tilingData.fiaArch38AttentionBaseParams.set_headNumRatio(nQ / nKV);
        return true;
    }
}

} // namespace arch38
} // namespace optiling
