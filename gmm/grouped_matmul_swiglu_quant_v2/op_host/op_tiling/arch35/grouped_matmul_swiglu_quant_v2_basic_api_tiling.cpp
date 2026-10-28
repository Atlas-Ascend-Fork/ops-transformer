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
 * \file grouped_matmul_swiglu_quant_v2_basic_api_tiling.cpp
 * \brief
 */

#include "grouped_matmul_swiglu_quant_v2_basic_api_tiling.h"

#include <algorithm>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "common/include/op_host/op_const_def_transformer.h"
#include "../../../op_kernel/arch35/grouped_matmul_swiglu_quant_v2_tiling_key.h"
#include "grouped_matmul_swiglu_quant_tensor_api_tiling_common.h"

namespace optiling {
namespace {
using namespace GroupedMatmulSwigluQuantV2Tiling;

constexpr size_t SHAPE_DIM_ONE = 1;
constexpr size_t SHAPE_DIM_TWO = 2;
constexpr size_t SHAPE_DIM_THREE = 3;
constexpr size_t SHAPE_DIM_FOUR = 4;
constexpr uint64_t MX_SCALE_K_ALIGN = 64UL;
constexpr int64_t MX_QUANT_MODE = 2L;

} // namespace

void GroupedMatmulSwigluQuantV2BasicApiTiling::Reset()
{
    tilingData_ = {};
    auto* rawTiling = context_ == nullptr ? nullptr : context_->GetRawTilingData();
    hif4TilingData_.SetDataPtr(rawTiling == nullptr ? nullptr : rawTiling->GetData());
    basicTiling_ = GQmmBasicTiling();
    xScaleDtype_ = ge::DT_UNDEFINED;
    quantDtype_ = ge::DT_UNDEFINED;
    xScaleFormat_ = static_cast<ge::Format>(-1);
    weightScaleFormat_ = static_cast<ge::Format>(-1);
    yScaleFormat_ = static_cast<ge::Format>(-1);
    aivNum_ = 0U;
    platformMemoryReady_ = false;
    return;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        const auto* compileInfo = context_->GetCompileInfo<GMMSwigluV2CompileInfo>();
        OP_CHECK_IF(compileInfo == nullptr, OP_LOGE(context_->GetNodeName(), "Compile info is nullptr."),
                    return ge::GRAPH_FAILED);
        aicoreParams_.aicNum = compileInfo->aicNum_;
        aivNum_ = compileInfo->aivNum_;
        aicoreParams_.ubSize = compileInfo->ubSize_;
        npuArch_ = static_cast<NpuArch>(compileInfo->npuArch_);
        if (IsHif4Input()) {
            aicoreParams_.l1Size = compileInfo->l1Size_;
            aicoreParams_.l0cSize = compileInfo->l0CSize_;
        } else {
            OP_LOGD(context_->GetNodeName(),
                    "Platform memory information is unavailable; Tensor API tiling will fall back.");
        }
        return ge::GRAPH_SUCCESS;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        aicoreParams_.aicNum = ascendcPlatform.GetCoreNumAic();
        aivNum_ = ascendcPlatform.GetCoreNumAiv();
        npuArch_ = ascendcPlatform.GetCurNpuArch();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, aicoreParams_.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, aicoreParams_.l0bSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
        platformMemoryReady_ = aicoreParams_.l1Size > 0 && aicoreParams_.l0cSize > 0;
    }
    OP_LOGD(context_->GetNodeName(),
            "Tensor API platform info: aicNum=%lu, aivNum=%u, ub=%lu, l1=%lu, l0a=%lu, l0b=%lu, l0c=%lu.",
            aicoreParams_.aicNum, aivNum_, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0aSize,
            aicoreParams_.l0bSize, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling::GetShapeAttrsInfo()
{
    inputParams_.Reset();
    inputParams_.opName = context_->GetNodeName();
    inputParams_.opType = GetOpType();
    OP_CHECK_IF(!AnalyzeDtype() || !AnalyzeAttrs() || !AnalyzeInputs(),
                OP_LOGE(inputParams_.opName, "Failed to analyze swiglu quant tiling inputs."), return ge::GRAPH_FAILED);
    inputParams_.initFlag = true;
    return ge::GRAPH_SUCCESS;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling::AnalyzeDtype()
{
    auto xDesc = context_->GetInputDesc(X_INDEX);
    auto xScaleDesc = context_->GetInputDesc(X_SCALE_INDEX);
    auto weightDesc = context_->GetDynamicInputDesc(WEIGHT_INDEX, 0);
    auto weightScaleDesc = context_->GetDynamicInputDesc(WEIGHT_SCALE_INDEX, 0);
    auto yDesc = context_->GetOutputDesc(Y_INDEX);
    auto yScaleDesc = context_->GetOutputDesc(Y_SCALE_INDEX);
    OP_CHECK_IF(xDesc == nullptr || xScaleDesc == nullptr || weightDesc == nullptr || weightScaleDesc == nullptr ||
                    yDesc == nullptr || yScaleDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "Tensor API input or output desc is nullptr."), return false);

    inputParams_.aDtype = xDesc->GetDataType();
    inputParams_.bDtype = weightDesc->GetDataType();
    xScaleDtype_ = xScaleDesc->GetDataType();
    inputParams_.scaleDtype = weightScaleDesc->GetDataType();
    inputParams_.outDataDtype = yDesc->GetDataType();
    inputParams_.outScaleDtype = yScaleDesc->GetDataType();
    // HIF4 keeps the existing FP16 intermediate used by the low-level tiling strategy.
    if (IsHif4Input()) {
        inputParams_.perTokenScaleDtype = xScaleDtype_;
    } else {
        inputParams_.cDtype = inputParams_.outDataDtype;
    }
    inputParams_.aFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(xDesc->GetFormat().GetStorageFormat()));
    inputParams_.bFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(weightDesc->GetFormat().GetStorageFormat()));
    xScaleFormat_ = static_cast<ge::Format>(ge::GetPrimaryFormat(xScaleDesc->GetFormat().GetStorageFormat()));
    weightScaleFormat_ = static_cast<ge::Format>(ge::GetPrimaryFormat(weightScaleDesc->GetFormat().GetStorageFormat()));
    inputParams_.cFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(yDesc->GetFormat().GetStorageFormat()));
    yScaleFormat_ = static_cast<ge::Format>(ge::GetPrimaryFormat(yScaleDesc->GetFormat().GetStorageFormat()));

    auto biasShape = context_->GetOptionalInputShape(BIAS_INDEX);
    inputParams_.hasBias = biasShape != nullptr && biasShape->GetStorageShape().GetShapeSize() != 0;
    auto biasDesc = context_->GetOptionalInputDesc(BIAS_INDEX);
    OP_CHECK_IF(inputParams_.hasBias && biasDesc == nullptr,
                OP_LOGE(context_->GetNodeName(), "Bias shape is present but bias desc is nullptr."), return false);
    if (biasDesc != nullptr) {
        inputParams_.biasDtype = biasDesc->GetDataType();
    }
    return true;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context_->GetNodeName(), "Attrs is nullptr."), return false);

    const int64_t* dequantMode = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_MODE);
    const int64_t* dequantDtype = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_DEQUANT_DTYPE);
    const int64_t* quantMode = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_MODE);
    const int64_t* quantDtype = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_QUANT_DTYPE);
    OP_CHECK_IF(dequantMode == nullptr || dequantDtype == nullptr || quantMode == nullptr || quantDtype == nullptr,
                OP_LOGE(context_->GetNodeName(), "Required quantization attrs are nullptr."), return false);
    const int64_t quantModeExpected = IsHif4Input() ? GroupedMatmulSwigluQuantParamsV2::Hif4QuantMode : MX_QUANT_MODE;
    OP_CHECK_IF(*dequantMode != quantModeExpected || *quantMode != quantModeExpected,
                OP_LOGE(context_->GetNodeName(), "dequant_mode and quant_mode must match the input quantization type."),
                return false);
    OP_CHECK_IF(static_cast<ge::DataType>(*dequantDtype) != ge::DT_FLOAT,
                OP_LOGE(context_->GetNodeName(), "dequant_dtype must be DT_FLOAT for MX quant."), return false);
    quantDtype_ = static_cast<ge::DataType>(*quantDtype);
    OP_CHECK_IF(IsHif4Input() && quantDtype_ != Ops::Transformer::DT_HIFLOAT4,
                OP_LOGE(context_->GetNodeName(), "HIF4 quant_dtype must be DT_HIFLOAT4."), return false);
    const bool* transposeWeight = attrs->GetAttrPointer<bool>(ATTR_INDEX_TRANSPOSE_WEIGHT);
    const int64_t* groupListType = attrs->GetAttrPointer<int64_t>(ATTR_INDEX_GROUPLIST_TYPE);
    inputParams_.transA = false;
    inputParams_.transB = transposeWeight != nullptr ? *transposeWeight : false;
    OP_CHECK_IF(IsHif4Input() && !inputParams_.transB,
                OP_LOGE(context_->GetNodeName(), "HIF4 BasicAPI requires transpose_weight=true."), return false);
    inputParams_.groupListType = groupListType != nullptr ? static_cast<int8_t>(*groupListType) : 0;
    OP_CHECK_IF(inputParams_.groupListType != 0 && inputParams_.groupListType != 1,
                OP_LOGE(context_->GetNodeName(), "group_list_type must be 0 or 1."), return false);
    inputParams_.groupType = static_cast<int8_t>(GroupedMatmul::SPLIT_M);
    inputParams_.aQuantMode = IsHif4Input() ? QuantMode::HIF4_PERGROUP_MODE : QuantMode::MX_PERGROUP_MODE;
    inputParams_.bQuantMode = inputParams_.aQuantMode;
    return true;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling::AnalyzeInputs()
{
    auto xStorageShape = context_->GetInputShape(X_INDEX);
    auto weightStorageShape = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    auto groupListStorageShape = context_->GetOptionalInputShape(GROUPLIST_INDEX);
    OP_CHECK_IF(xStorageShape == nullptr || weightStorageShape == nullptr || groupListStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "x, weight or group_list shape is nullptr."), return false);

    const auto& xShape = xStorageShape->GetOriginShape();
    const auto& weightShape = weightStorageShape->GetOriginShape();
    const auto& groupListShape = groupListStorageShape->GetStorageShape();
    OP_CHECK_IF(xShape.GetDimNum() < SHAPE_DIM_TWO || weightShape.GetDimNum() < SHAPE_DIM_TWO,
                OP_LOGE(context_->GetNodeName(), "x and weight must have at least two dimensions."), return false);
    OP_CHECK_IF(groupListShape.GetDimNum() != SHAPE_DIM_ONE,
                OP_LOGE(context_->GetNodeName(), "group_list must be one-dimensional."), return false);

    const size_t xDimNum = xShape.GetDimNum();
    const size_t weightDimNum = weightShape.GetDimNum();
    const int64_t m = xShape.GetDim(xDimNum - 2);
    const int64_t k = xShape.GetDim(xDimNum - 1);
    const int64_t weightK =
        inputParams_.transB ? weightShape.GetDim(weightDimNum - 1) : weightShape.GetDim(weightDimNum - 2);
    const int64_t n = inputParams_.transB ? weightShape.GetDim(weightDimNum - 2) : weightShape.GetDim(weightDimNum - 1);
    const int64_t groupNum = groupListShape.GetDim(0);
    OP_CHECK_IF(
        m <= 0 || n <= 0 || k <= 0 || weightK != k || groupNum <= 0 ||
            groupNum > static_cast<int64_t>(GmmConstant::GMM_MAX_GROUP_LIST_SIZE),
        OP_LOGE(context_->GetNodeName(), "Invalid Tensor API shape: M=%ld, N=%ld, K=%ld, weightK=%ld, groupNum=%ld.", m,
                n, k, weightK, groupNum),
        return false);

    inputParams_.mSize = static_cast<uint64_t>(m);
    inputParams_.nSize = static_cast<uint64_t>(n);
    inputParams_.kSize = static_cast<uint64_t>(k);
    inputParams_.groupNum = static_cast<uint64_t>(groupNum);
    if (IsHif4Input()) {
        OP_CHECK_IF(!CheckInputShapes(),
                    OP_LOGE(context_->GetNodeName(),
                            "HIF4 requires one weight tensor and one weightScale tensor, x rank %zu, "
                            "weight rank %zu, and weight expert count equal to groupNum=%lu.",
                            SHAPE_DIM_TWO, SHAPE_DIM_THREE, inputParams_.groupNum),
                    return false);
        return CheckScaleShapes() && CheckHif4Dims();
    }
    inputParams_.isSingleX = true;
    inputParams_.isSingleW = GetDynamicInputCount(WEIGHT_INDEX) == 1U;
    inputParams_.isSingleY = true;
    return true;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling::IsFp8(ge::DataType dtype) const
{
    return dtype == ge::DT_FLOAT8_E4M3FN || dtype == ge::DT_FLOAT8_E5M2;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling::IsSupportedFormat(ge::Format format) const
{
    return format == ge::FORMAT_ND || format == ge::FORMAT_NCL || format == ge::FORMAT_NCHW;
}

size_t GroupedMatmulSwigluQuantV2BasicApiTiling::GetDynamicInputCount(uint32_t inputIndex) const
{
    size_t count = 0U;
    while (count < GmmConstant::GMM_MAX_GROUP_LIST_SIZE &&
           context_->GetDynamicInputShape(inputIndex, count) != nullptr) {
        ++count;
    }
    return count;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling::CheckInputShapes() const
{
    auto xShapePtr = context_->GetInputShape(X_INDEX);
    auto weightShapePtr = context_->GetDynamicInputShape(WEIGHT_INDEX, 0);
    if (xShapePtr == nullptr || weightShapePtr == nullptr || GetDynamicInputCount(WEIGHT_INDEX) != 1U ||
        GetDynamicInputCount(WEIGHT_SCALE_INDEX) != 1U) {
        return false;
    }
    const auto& xShape = xShapePtr->GetOriginShape();
    const auto& weightShape = weightShapePtr->GetOriginShape();
    // HIF4 keeps the explicit expert axis required by its ACLNN input contract.
    if (IsHif4Input() && weightShape.GetDimNum() != SHAPE_DIM_THREE) {
        return false;
    }
    if (xShape.GetDimNum() != SHAPE_DIM_TWO ||
        (weightShape.GetDimNum() != SHAPE_DIM_TWO && weightShape.GetDimNum() != SHAPE_DIM_THREE)) {
        return false;
    }
    return (weightShape.GetDimNum() == SHAPE_DIM_TWO && inputParams_.groupNum == 1U) ||
           (weightShape.GetDimNum() == SHAPE_DIM_THREE &&
            weightShape.GetDim(0) == static_cast<int64_t>(inputParams_.groupNum));
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling::CheckScaleShapes() const
{
    auto xScaleShapePtr = context_->GetInputShape(X_SCALE_INDEX);
    auto weightScaleShapePtr = context_->GetDynamicInputShape(WEIGHT_SCALE_INDEX, 0);
    if (xScaleShapePtr == nullptr || weightScaleShapePtr == nullptr) {
        return false;
    }
    const auto& xScaleShape = xScaleShapePtr->GetOriginShape();
    const auto& weightScaleShape = weightScaleShapePtr->GetOriginShape();
    if (xScaleShape.GetDimNum() != SHAPE_DIM_THREE || weightScaleShape.GetDimNum() != SHAPE_DIM_FOUR) {
        return false;
    }

    // 2：MX 模式每组包含 2 个 E8M0 scale，HIF4 每组仅一个 scale。
    const int64_t scaleLastDim = IsHif4Input() ? 1L : 2L;
    const int64_t kBlocks = static_cast<int64_t>(GroupedMatmul::CeilDiv(inputParams_.kSize, MX_SCALE_K_ALIGN));
    if (xScaleShape.GetDim(0) != static_cast<int64_t>(inputParams_.mSize) || xScaleShape.GetDim(1) != kBlocks ||
        // 2：xScale 的末维索引。
        xScaleShape.GetDim(2) != scaleLastDim) {
        return false;
    }
    const int64_t weightScaleN = inputParams_.transB ? weightScaleShape.GetDim(1) : weightScaleShape.GetDim(2);
    const int64_t weightScaleK = inputParams_.transB ? weightScaleShape.GetDim(2) : weightScaleShape.GetDim(1);
    return weightScaleShape.GetDim(0) == static_cast<int64_t>(inputParams_.groupNum) &&
           weightScaleN == static_cast<int64_t>(inputParams_.nSize) && weightScaleK == kBlocks &&
           // 3：weightScale 的末维索引。
           weightScaleShape.GetDim(3) == scaleLastDim;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling::IsCapable()
{
    if (IsHif4Input()) {
        return inputParams_.aQuantMode == QuantMode::HIF4_PERGROUP_MODE;
    }
    if (!platformMemoryReady_) {
        return false;
    }
    // Keep Tensor API-specific dtype restrictions in capability checking so unsupported cases can fall back.
    const bool quantDtypeSupported = IsFp8(quantDtype_) && quantDtype_ == inputParams_.outDataDtype;
    const bool dtypeSupported = IsFp8(inputParams_.aDtype) && IsFp8(inputParams_.bDtype) &&
                                xScaleDtype_ == ge::DT_FLOAT8_E8M0 && inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0 &&
                                IsFp8(inputParams_.outDataDtype) && inputParams_.outScaleDtype == ge::DT_FLOAT8_E8M0;
    const bool formatSupported = IsSupportedFormat(inputParams_.aFormat) && IsSupportedFormat(inputParams_.bFormat) &&
                                 IsSupportedFormat(xScaleFormat_) && IsSupportedFormat(weightScaleFormat_) &&
                                 IsSupportedFormat(inputParams_.cFormat) && IsSupportedFormat(yScaleFormat_);
    const bool checkTensorApiShapes = CheckInputShapes();
    const bool checkTensorApiScaleShapes = CheckScaleShapes();
    const uint64_t averageMPerGroup = inputParams_.mSize / inputParams_.groupNum;
    const uint64_t mPerGroupLowerLimit =
        inputParams_.transB ? GroupedMatmulSwigluQuantTensorApiTiling::TRANS_B_M_PER_GROUP_LOWER_LIMIT :
                              GroupedMatmulSwigluQuantTensorApiTiling::NOT_TRANS_B_M_PER_GROUP_LOWER_LIMIT;
    const bool commonCapable = GroupedMatmulSwigluQuantTensorApiTiling::IsShapeAndPlatformCapable(
        {inputParams_.mSize, inputParams_.nSize, inputParams_.kSize, inputParams_.groupNum, aicoreParams_.aicNum,
         aivNum_, inputParams_.transB, platformMemoryReady_});
    const bool capable = dtypeSupported && quantDtypeSupported && formatSupported && checkTensorApiShapes &&
                         checkTensorApiScaleShapes && commonCapable;
    OP_LOGD(
        context_->GetNodeName(),
        "Tiling conditions: dtype=%d, quantDtypeSupported=%d, format=%d, groupNum=%lu, M=%lu, "
        "averageMPerGroup=%lu, mPerGroupLowerLimit=%lu, transB=%d, N=%lu, "
        "cores=%lu:%u, "
        "checkTensorApiShapes=%d, checkKSize=%d, checkKSizeAlign=%d, "
        "checkTensorApiScaleShapes=%d, checkMSizeGroupNumRatio=%d, checkBNoTransMLimlit=%d, capable=%d.",
        dtypeSupported, quantDtypeSupported, formatSupported, inputParams_.groupNum, inputParams_.mSize,
        averageMPerGroup, mPerGroupLowerLimit, inputParams_.transB, inputParams_.nSize, aicoreParams_.aicNum, aivNum_,
        checkTensorApiShapes, inputParams_.kSize > GroupedMatmulSwigluQuantTensorApiTiling::K_LOWER_LIMIT,
        inputParams_.kSize % GroupedMatmulSwigluQuantTensorApiTiling::K_ALIGN == 0UL, checkTensorApiScaleShapes,
        averageMPerGroup > mPerGroupLowerLimit,
        inputParams_.transB || inputParams_.mSize > GroupedMatmulSwigluQuantTensorApiTiling::NOT_TRANS_B_M_LOWER_LIMIT,
        capable);
    return capable;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling::DoOpTiling()
{
    if (IsHif4Input()) {
        OP_CHECK_IF(aicoreParams_.l1Size == 0 || aicoreParams_.l0cSize == 0,
                    OP_LOGE(context_->GetNodeName(), "HIF4 requires L1 and L0C platform memory information."),
                    return ge::GRAPH_FAILED);
        auto* rawTiling = context_->GetRawTilingData();
        OP_CHECK_IF(rawTiling == nullptr || rawTiling->GetData() == nullptr ||
                        rawTiling->GetCapacity() < hif4TilingData_.GetDataSize(),
                    OP_LOGE(context_->GetNodeName(), "HIF4 tiling buffer is unavailable or too small."),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF(memset_s(rawTiling->GetData(), rawTiling->GetCapacity(), 0, hif4TilingData_.GetDataSize()) != EOK,
                    OP_LOGE(context_->GetNodeName(), "Failed to clear HIF4 tiling buffer."), return ge::GRAPH_FAILED);
        hif4TilingData_.gmmSwigluQuantParams.set_groupNum(inputParams_.groupNum);
        hif4TilingData_.gmmSwigluQuantParams.set_groupListType(static_cast<uint8_t>(inputParams_.groupListType));
        hif4TilingData_.gmmSwigluQuantParams.set_quantDtype(static_cast<uint8_t>(quantDtype_));
    } else {
        GroupedMatmulSwigluQuantTensorApiTiling::FillQuantParams(tilingData_.gmmQuantParams, inputParams_);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling::DoLibApiTiling()
{
    GroupedQmmTiling::CalBasicBlock();
    if (IsHif4Input()) {
        basicTiling_.baseM = GroupedMatmul::CeilAlign(std::min(basicTiling_.baseM, GmmConstant::MAX_BASEM_ONE_VECTOR),
                                                      GmmConstant::CUBE_BLOCK);
    } else {
        GroupedMatmulSwigluQuantTensorApiTiling::PrepareBasicBlockForL1(basicTiling_);
    }
    OP_CHECK_IF(GroupedQmmBasicApiTiling::CalL1Tiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CalL1Tiling failed."), return ge::GRAPH_FAILED);
    if (!IsHif4Input()) {
        GroupedMatmulSwigluQuantTensorApiTiling::RestoreBasicBlockAfterL1(basicTiling_);
    }
    GroupedMatmulSwigluQuantTensorApiTiling::FillMatmulTiling(tilingData_.mmTilingData, inputParams_, basicTiling_,
                                                              basicTiling_.scaleFactorA, basicTiling_.scaleFactorB);
    if (IsHif4Input()) {
        // Serialize the same compact payload consumed by the HIF4 kernel.
        const auto& mm = tilingData_.mmTilingData;
        hif4TilingData_.basicTilingData.set_m(mm.m);
        hif4TilingData_.basicTilingData.set_n(mm.n);
        hif4TilingData_.basicTilingData.set_k(mm.k);
        hif4TilingData_.basicTilingData.set_baseM(mm.baseM);
        hif4TilingData_.basicTilingData.set_baseN(mm.baseN);
        hif4TilingData_.basicTilingData.set_baseK(mm.baseK);
        hif4TilingData_.basicTilingData.set_kAL1(mm.kAL1);
        hif4TilingData_.basicTilingData.set_kBL1(mm.kBL1);
        hif4TilingData_.basicTilingData.set_dbL0C(mm.dbL0C);
        hif4TilingData_.basicTilingData.set_isBias(0);
        hif4TilingData_.basicTilingData.set_scaleKAL1(mm.scaleKAL1);
        hif4TilingData_.basicTilingData.set_scaleKBL1(mm.scaleKBL1);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = GmmConstant::SYS_WORKSPACE_SIZES;
    return ge::GRAPH_SUCCESS;
}

uint64_t GroupedMatmulSwigluQuantV2BasicApiTiling::GetTilingKey() const
{
    const uint64_t kernelType =
        IsHif4Input() ? GMM_SWIGLU_HIF4_QUANT_KERNEL_TYPE : GMM_SWIGLU_QUANT_TENSOR_LEVEL_KERNEL_TYPE;
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(inputParams_.transB), static_cast<uint64_t>(inputParams_.transA),
                              kernelType);
}

ge::graphStatus GroupedMatmulSwigluQuantV2BasicApiTiling::PostTiling()
{
    if (IsHif4Input()) {
        context_->SetBlockDim(aicoreParams_.aicNum);
        context_->SetScheduleMode(1);
        OP_CHECK_IF(hif4TilingData_.GetDataSize() % sizeof(uint64_t) != 0,
                    OP_LOGE(context_->GetNodeName(), "HIF4 tiling data must be aligned to 8 bytes."),
                    return ge::GRAPH_FAILED);
        hif4TilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                                     context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(hif4TilingData_.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }
    context_->SetScheduleMode(BATCH_MODE_SCHEDULE);
    return SaveTilingDataToContext(tilingData_);
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling::IsHif4Input() const
{
    const auto* x = context_->GetInputDesc(GroupedMatmulSwigluQuantV2Tiling::X_INDEX);
    const auto* w = context_->GetDynamicInputDesc(GroupedMatmulSwigluQuantV2Tiling::WEIGHT_INDEX, 0);
    return x != nullptr && w != nullptr && x->GetDataType() == Ops::Transformer::DT_HIFLOAT4 &&
           w->GetDataType() == Ops::Transformer::DT_HIFLOAT4;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling::CheckHif4Dtype() const
{
    OP_CHECK_IF(inputParams_.scaleDtype != Ops::Transformer::DT_HIFLOAT4_SCALE ||
                    inputParams_.perTokenScaleDtype != Ops::Transformer::DT_HIFLOAT4_SCALE ||
                    inputParams_.outDataDtype != Ops::Transformer::DT_HIFLOAT4 ||
                    inputParams_.outScaleDtype != Ops::Transformer::DT_HIFLOAT4_SCALE,
                OP_LOGE(inputParams_.opName, "HIF4 requires HIFLOAT4 output and HIFLOAT4_SCALE input/output scales."),
                return false);
    return true;
}

bool GroupedMatmulSwigluQuantV2BasicApiTiling::CheckHif4Dims() const
{
    OP_CHECK_IF(inputParams_.kSize <= GroupedMatmulSwigluQuantParamsV2::HIF4_K_MIN_VALUE ||
                    inputParams_.kSize % GroupedMatmulSwigluQuantParamsV2::B4_DATACOPY_MIN_NUM != 0,
                OP_LOGE(inputParams_.opName, "HIF4 requires even K greater than 2."), return false);
    OP_CHECK_IF(inputParams_.nSize == 0 || inputParams_.nSize % GmmConstant::BASIC_BLOCK_SIZE_128 != 0,
                OP_LOGE(inputParams_.opName, "HIF4 requires positive N aligned to 128."), return false);
    return true;
}
} // namespace optiling
