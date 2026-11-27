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
 * \file grouped_matmul_add_tiling.cpp
 * \brief
 */

#include <string>
#include "grouped_matmul_add_tiling.h"
#include <climits>
#include <limits>
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "log/log.h"
#include <vector>

#include "arch35/grouped_matmul_add_compile_info.h"
#include "arch35/grouped_matmul_add_no_quant_tiling.h"
#include "arch35/grouped_matmul_add_platform_common.h"

namespace optiling {
constexpr uint64_t BEST_L1_PARTA = 128UL * 1024UL;
constexpr uint64_t BEST_L1_PARTB = 256UL * 1024UL;
constexpr int32_t BEST_BASEN = 256;
constexpr int32_t MAX_BASEM = 256;
constexpr uint32_t DATATYPE_SIZE = 2;
constexpr uint32_t FP32_DATATYPE_SIZE = 4;
constexpr uint64_t TILING_KEY_DEFAULT_FP16 = 0;
constexpr uint64_t TILING_KEY_DEFAULT_BF16 = 1;
// Preserve keys 0/1 for the existing transposed, FP32-output path.
constexpr uint64_t TILING_KEY_NONTRANSPOSE_FP32 = 2;
constexpr uint64_t TILING_KEY_TRANSPOSE_16BIT = 4;
constexpr uint64_t TILING_KEY_NONTRANSPOSE_16BIT = 6;
constexpr uint64_t DOUBLE_BUFFER_L0A_L0B = 2;
constexpr uint32_t DOUBLE_BUFFER_STEPKA_STEPKB = 2U;
constexpr int32_t MAX_TENSOR_CONT = 128;
constexpr int64_t INDEX_GROUP_LIST = 2;
constexpr size_t INDEX_IN_X = 0;
constexpr size_t INDEX_IN_W = 1;
constexpr size_t INDEX_IN_GROUP_LIST = 2;
constexpr size_t INDEX_IN_YREF = 3;
constexpr int32_t DIM0 = 0;
constexpr int32_t DIM1 = 1;
constexpr int32_t DIM2 = 2;
constexpr int32_t DIMNUM_1D = 1;
constexpr int32_t DIMNUM_2D = 2;
constexpr int32_t DIMNUM_3D = 3;
constexpr int32_t ATTR_TRANSPOSE_X_INDEX = 0;
constexpr int32_t ATTR_TRANSPOSE_W_INDEX = 1;
constexpr int32_t ATTR_GROUP_TYPE_INDEX = 2;
constexpr int32_t ATTR_GROUP_LIST_TYPE_INDEX_3 = 3; // newly added code uses this naming style.
constexpr int32_t AIC_AIV_RATION = 2;
constexpr int64_t GROUP_LIST_TYPE_CUMSUM_0 = 0;

static bool GetTransposeX(const gert::TilingContext* context)
{
    const auto* attrs = context->GetAttrs();
    const bool* transposeX = attrs == nullptr ? nullptr : attrs->GetAttrPointer<bool>(ATTR_TRANSPOSE_X_INDEX);
    return transposeX == nullptr ? true : *transposeX;
}

static inline uint32_t SixteenAlign(uint32_t a, bool up = false)
{
    if (up) {
        a += 15U; // 15: 16 bytes up-align
    }
    return a & ~15U; // ~15: 16 bytes down-align
};

static ge::graphStatus CalTCubeTiling(const gert::TilingContext* context, GroupedMatmulAddTilingData& tiling, int32_t m,
                                      int32_t k, int32_t n, int32_t baseM, int32_t baseN, int32_t baseK)
{
    auto xType = context->GetInputDesc(INDEX_IN_X)->GetDataType();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto nodeName = context->GetNodeName();
    matmul_tiling::DataType matmulDtype = static_cast<matmul_tiling::DataType>(xType);
    matmul_tiling::MultiCoreMatmulTiling mm(ascendcPlatform);
    // Preserve the existing tiling configuration for the original FP32 path.
    const bool transposeX = GetTransposeX(context);
    const bool legacyPath = transposeX && context->GetInputDesc(INDEX_IN_YREF)->GetDataType() == ge::DT_FLOAT;
    mm.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype,
                legacyPath ? false : transposeX);
    mm.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmulDtype, false);
    mm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    mm.SetBias(false);
    mm.SetOrgShape(m, n, k);
    mm.SetShape(m, baseN, k);
    mm.SetFixSplit(baseM, baseN, baseK);

    uint64_t l1Size = 0UL;
    uint64_t l0_cSize = 0UL;
    uint64_t ubSize = 0UL;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0_cSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    mm.SetBufferSpace(l1Size, l0_cSize, ubSize);
    if (mm.GetTiling(tiling.mmTilingData) == -1) {
        OP_LOGE_LIBOPAPI_REPORT(Ops::Base::GetOpInfo(nodeName), "%s", "GroupedMatmulAdd GetTiling error");
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", std::vector<const char*>({"opName", "reason"}),
            std::vector<const char*>({Ops::Base::GetOpInfo(nodeName), "GroupedMatmulAdd GetTiling error"}));
        return ge::GRAPH_FAILED;
    }
    uint64_t productMK = static_cast<uint64_t>(baseM) * static_cast<uint64_t>(baseK);
    uint64_t productNK = static_cast<uint64_t>(baseN) * static_cast<uint64_t>(baseK);
    if (productMK > UINT32_MAX || productNK > UINT32_MAX) {
        const std::string errorReason =
            ("productMK or productNK > uint32_t max value, productMK:" + std::to_string(productMK) +
             ", productNK:" + std::to_string(productNK));
        OP_LOGE_LIBOPAPI_REPORT(Ops::Base::GetOpInfo(nodeName), "%s", errorReason.c_str());
        REPORT_PREDEFINED_ERR_MSG("EZ1008", std::vector<const char*>({"opName", "reason"}),
                                  std::vector<const char*>({Ops::Base::GetOpInfo(nodeName), errorReason.c_str()}));
        return ge::GRAPH_FAILED;
    }
    uint32_t mmStepKa =
        static_cast<uint32_t>((BEST_L1_PARTA >> 1) / (productMK * static_cast<uint64_t>(DATATYPE_SIZE)));
    uint32_t mmStepKb =
        static_cast<uint32_t>((BEST_L1_PARTB >> 1) / (productNK * static_cast<uint64_t>(DATATYPE_SIZE)));
    if (mmStepKa > mmStepKb) {
        mmStepKa = mmStepKa / mmStepKb * mmStepKb;
    } else if (mmStepKa < mmStepKb) {
        mmStepKb = mmStepKb / mmStepKa * mmStepKa;
    }
    constexpr uint32_t stepM = 1; // 1: stepM set fixed value 1
    constexpr uint32_t stepN = 1; // 1: stepN set fixed value 1
    uint32_t mmDepthA1 = mmStepKa * DOUBLE_BUFFER_STEPKA_STEPKB * stepM;
    uint32_t mmDepthB1 = mmStepKb * DOUBLE_BUFFER_STEPKA_STEPKB * stepN;

    tiling.mmTilingData.set_stepKa(mmStepKa);   // set precomputed mmStepKa
    tiling.mmTilingData.set_depthA1(mmDepthA1); // set precomputed mmDepthA1
    tiling.mmTilingData.set_stepKb(mmStepKb);   // set precomputed mmStepKb
    tiling.mmTilingData.set_depthB1(mmDepthB1); // set precomputed mmDepthB1
    tiling.mmTilingData.set_stepM(stepM);       // set precomputed stepM
    tiling.mmTilingData.set_stepN(stepN);       // set precomputed stepN
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CalMmTiling(gert::TilingContext* context, GroupedMatmulAddTilingData& tiling, int32_t m,
                                   int32_t k, int32_t n)
{
    int32_t baseN = BEST_BASEN;
    int32_t baseK = 0;
    int32_t baseM = 0;
    int32_t maxM = 0;
    uint64_t l0_A_size = 0;
    uint64_t l0_B_size = 0;
    uint64_t l0_C_size = 0;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // 先根据 baseN 和 L0_B的大小确定baseK
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0_B_size);
    baseK = static_cast<int32_t>((l0_B_size / DOUBLE_BUFFER_L0A_L0B) /
                                 static_cast<uint64_t>(baseN * static_cast<int32_t>(DATATYPE_SIZE)));
    baseK = static_cast<int32_t>(SixteenAlign(static_cast<uint32_t>(baseK)));
    // L0_C大小会限制 BaseM
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0_C_size);
    uint32_t maxBaseM = static_cast<uint32_t>(l0_C_size / static_cast<uint64_t>(baseN * FP32_DATATYPE_SIZE));
    // L0_A大小会限制 BaseM
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0_A_size);
    baseM = std::min<uint32_t>(
        static_cast<uint32_t>((l0_A_size / DOUBLE_BUFFER_L0A_L0B)) / (static_cast<uint32_t>(baseK) * DATATYPE_SIZE),
        maxBaseM);
    maxM = m;
    if (baseM > maxM) {
        baseM = static_cast<int32_t>(SixteenAlign(static_cast<uint32_t>(maxM), true));
    } else {
        baseM = static_cast<int32_t>(SixteenAlign(static_cast<uint32_t>(baseM)));
    }
    if (baseM > MAX_BASEM) {
        baseM = MAX_BASEM;
    }

    // 设置矩阵相关tiling参数
    auto xType = context->GetInputDesc(INDEX_IN_X)->GetDataType();
    auto yType = context->GetInputDesc(INDEX_IN_YREF)->GetDataType();
    if ((xType != ge::DataType::DT_FLOAT16 && xType != ge::DataType::DT_BF16) ||
        (yType != ge::DataType::DT_FLOAT && yType != xType)) {
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x, y",
                                                 "GroupedMatmulAdd: unsupported input/output type combination");
        return ge::GRAPH_FAILED;
    }
    const bool transposeX = GetTransposeX(context);
    uint64_t key = xType == ge::DataType::DT_FLOAT16 ? TILING_KEY_DEFAULT_FP16 : TILING_KEY_DEFAULT_BF16;
    if (yType == ge::DataType::DT_FLOAT) {
        key += transposeX ? 0 : TILING_KEY_NONTRANSPOSE_FP32;
    } else {
        key += transposeX ? TILING_KEY_TRANSPOSE_16BIT : TILING_KEY_NONTRANSPOSE_16BIT;
    }
    context->SetTilingKey(key);
    auto ret = CalTCubeTiling(context, tiling, m, k, n, baseM, baseN, baseK);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != ret,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "GroupedMatmulAdd CalTCubeTiling error"), return ret);
    tiling.mmTilingData.set_shareMode(0);
    tiling.mmTilingData.set_baseM(baseM); // set precomputed baseM
    tiling.mmTilingData.set_baseN(baseN); // set precomputed baseN
    tiling.mmTilingData.set_baseK(baseK); // set precomputed baseK

    return ge::GRAPH_SUCCESS;
}

static void PrintInfo(const gert::TilingContext* context, GroupedMatmulAddTilingData& tiling)
{
    auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Start to print GroupedMatmulAdd tiling data <<<<<<<<<<<<<<<<");
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [GmmBaseParams]: groupNum = %ld", tiling.gmmBaseParams.get_groupNum());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [GmmBaseParams]: coreNum = %ld", tiling.gmmBaseParams.get_coreNum());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [GmmBaseParams]: groupType = %ld", tiling.gmmBaseParams.get_groupType());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [GmmBaseParams]: groupListType = %ld",
            tiling.gmmBaseParams.get_groupListType());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: usedCoreNum = %d", tiling.mmTilingData.get_usedCoreNum());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: M = %d", tiling.mmTilingData.get_M());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: N = %d", tiling.mmTilingData.get_N());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: Ka = %d", tiling.mmTilingData.get_Ka());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: Kb = %d", tiling.mmTilingData.get_Kb());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: singleCoreM = %d", tiling.mmTilingData.get_singleCoreM());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: singleCoreN = %d", tiling.mmTilingData.get_singleCoreN());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: singleCoreK = %d", tiling.mmTilingData.get_singleCoreK());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: baseM = %d", tiling.mmTilingData.get_baseM());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: baseN = %d", tiling.mmTilingData.get_baseN());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: baseK = %d", tiling.mmTilingData.get_baseK());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: depthA1 = %d", tiling.mmTilingData.get_depthA1());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: depthB1 = %d", tiling.mmTilingData.get_depthB1());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: stepM = %d", tiling.mmTilingData.get_stepM());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: stepN = %d", tiling.mmTilingData.get_stepN());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: stepKa = %d", tiling.mmTilingData.get_stepKa());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: stepKb = %d", tiling.mmTilingData.get_stepKb());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: isBias = %d", tiling.mmTilingData.get_isBias());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: transLength = %d", tiling.mmTilingData.get_transLength());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: iterateOrder = %d", tiling.mmTilingData.get_iterateOrder());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: dbL0A = %d", tiling.mmTilingData.get_dbL0A());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: dbL0B = %d", tiling.mmTilingData.get_dbL0B());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TCubeTiling]: dbL0C = %d", tiling.mmTilingData.get_dbL0C());
    OP_LOGD(nodeName, ">>> GroupedMatmulAdd [TilingKey]: %lu", context->GetTilingKey());
    OP_LOGD(nodeName, ">>>>>>>>>>>>>>> Print GroupedMatmulAdd tiling data end <<<<<<<<<<<<<<<<");
}

static ge::graphStatus TilingCheck4GroupedMatmulAdd(const gert::TilingContext* context)
{
    auto xShape = context->GetInputShape(INDEX_IN_X);
    auto wShape = context->GetInputShape(INDEX_IN_W);
    auto groupListShape = context->GetInputShape(INDEX_IN_GROUP_LIST);
    auto yRefShape = context->GetInputShape(INDEX_IN_YREF);

    // check nullptr
    OP_CHECK_IF(xShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "x"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(wShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "weight"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(groupListShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "group_list"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(yRefShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y"), return ge::GRAPH_FAILED);

    auto xOrgShape = xShape->GetOriginShape();
    auto wOrgShape = wShape->GetOriginShape();
    size_t xDimNum = xOrgShape.GetDimNum();
    size_t wDimNum = wOrgShape.GetDimNum();
    size_t groupListDimNum = groupListShape->GetOriginShape().GetDimNum();
    size_t yRefDimNum = yRefShape->GetOriginShape().GetDimNum();
    // 检查输入dim是否符合预期
    OP_CHECK_IF(DIMNUM_2D != xDimNum,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x",
                                                         "xShape error xShape.GetDimNum():" + std::to_string(xDimNum)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(DIMNUM_2D != wDimNum,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "weight",
                                                         "wShape error wShape.GetDimNum():" + std::to_string(wDimNum)),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(DIMNUM_1D != groupListDimNum,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "group_list",
                    "groupListShape error groupListShape.GetDimNum():" + std::to_string(groupListDimNum)),
                return ge::GRAPH_FAILED);
    if (DIMNUM_3D != yRefDimNum) {
        OP_LOGW(context->GetNodeName(), "yRefShape error yRefShape.GetDimNum():%lu", yRefDimNum);
    }

    // 检查shape是否可以对应
    auto xk = xOrgShape.GetDim(GetTransposeX(context) ? DIM0 : DIM1);
    auto wk = wOrgShape.GetDim(DIM0);
    OP_CHECK_IF(
        xk != wk,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x, weight",
                                                 "shape Error xk:" + std::to_string(xk) + " wk:" + std::to_string(wk)),
        return ge::GRAPH_FAILED);

    // Until unaligned 16-bit output writes are validated, keep each output row 32-byte aligned.
    const auto* yDesc = context->GetInputDesc(INDEX_IN_YREF);
    OP_CHECK_IF(yDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "y"), return ge::GRAPH_FAILED);
    const auto yType = yDesc->GetDataType();
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    if (platform.GetCurNpuArch() == NpuArch::DAV_2201 && (!GetTransposeX(context) || yType != ge::DT_FLOAT)) {
        const auto m = xOrgShape.GetDim(GetTransposeX(context) ? DIM1 : DIM0);
        const auto n = wOrgShape.GetDim(DIM1);
        const auto groups = groupListShape->GetOriginShape().GetDim(DIM0);
        constexpr int64_t DIM_LIMIT = 65536;
        constexpr int64_t INT32_LIMIT = std::numeric_limits<int32_t>::max();
        OP_CHECK_IF(m < 0 || m > INT32_LIMIT || xk < 0 || xk > INT32_LIMIT || n < 0 || n > INT32_LIMIT || groups < 0 ||
                        groups > INT32_LIMIT || xOrgShape.GetDim(DIM1) >= DIM_LIMIT || n >= DIM_LIMIT,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "x, weight, group_list",
                                                             "Expanded path dimensions exceed supported limits"),
                    return ge::GRAPH_FAILED);
        const auto maxSize = std::numeric_limits<int64_t>::max();
        OP_CHECK_IF(
            m != 0 && n > maxSize / m,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "y", "Output element count exceeds int64"),
            return ge::GRAPH_FAILED);
        const int64_t matrixSize = m * n;
        OP_CHECK_IF(
            matrixSize != 0 && groups > maxSize / matrixSize,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "y", "Output element count exceeds int64"),
            return ge::GRAPH_FAILED);
        int64_t actualSize = 1;
        const auto yShape = yRefShape->GetOriginShape();
        for (size_t i = 0; i < yRefDimNum; ++i) {
            const auto dim = yShape.GetDim(i);
            OP_CHECK_IF(
                dim < 0 || (actualSize != 0 && dim > maxSize / actualSize),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "y", "Output dimensions are negative or element count exceeds int64"),
                return ge::GRAPH_FAILED);
            actualSize *= dim;
        }
        // In ACLNN a contiguous y view can have a larger backing origin shape
        // and a nonzero view offset. The tiling context sees that backing
        // shape, not the logical view. Reject a buffer that is definitely too
        // small; ACLNN separately checks the exact logical view size.
        OP_CHECK_IF(actualSize < matrixSize * groups,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        context->GetNodeName(), "y", "Output storage has fewer elements than M * N * group count"),
                    return ge::GRAPH_FAILED);
    }
    constexpr int64_t ELEMENTS_PER_32B_16BIT = 16;
    OP_CHECK_IF(
        (yType == ge::DT_FLOAT16 || yType == ge::DT_BF16) && wOrgShape.GetDim(DIM1) % ELEMENTS_PER_32B_16BIT != 0,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "y",
                                                 "16-bit output requires N to be a multiple of 16"),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4GroupedMatmulAdd(gert::TilingContext* context)
{
    if (IsAdvancedNpuArch(context)) {
        GroupedMatmulAddNoQuantTiling gmmAddTiling;
        OP_CHECK_IF(!gmmAddTiling.SetTiling(context),
                    OPS_REPORT_VECTOR_INNER_ERR(context->GetNodeName(), "SetTiling failed."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }
    GroupedMatmulAddTilingData tiling;

    OP_CHECK_IF(ge::GRAPH_SUCCESS != TilingCheck4GroupedMatmulAdd(context),
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "GroupedMatmulAdd TilingCheck4GroupedMatmulAdd error"),
                return ge::GRAPH_FAILED);

    // input shapes was checked in TilingCheck4GroupedMatmulAdd. Fix: refactering to avoid repeated cal.
    auto xShape = context->GetInputShape(INDEX_IN_X)->GetOriginShape();
    auto wShape = context->GetInputShape(INDEX_IN_W)->GetOriginShape();
    auto groupListShape = context->GetInputShape(INDEX_IN_GROUP_LIST)->GetOriginShape();
    const bool transposeX = GetTransposeX(context);
    int64_t m = xShape.GetDim(transposeX ? DIM1 : DIM0);
    int64_t k = xShape.GetDim(transposeX ? DIM0 : DIM1);
    int64_t n = wShape.GetDim(DIM1);

    // 先设置最大核数
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const int64_t aicNum = ascendcPlatform.GetCoreNumAic();

    // 设置gmmBaseParams
    tiling.gmmBaseParams.set_groupNum(groupListShape.GetDim(DIM0));
    // Fix: remove coreNum. kernel can get coreNum.
    tiling.gmmBaseParams.set_coreNum(aicNum * AIC_AIV_RATION);
    tiling.gmmBaseParams.set_groupType(INDEX_GROUP_LIST);
    // set groupListType attr.
    auto attrs = context->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    context->GetNodeName(), "attrs",
                    "Required attrs metadata is null; context->GetAttrs() must provide a valid object."),
                return ge::GRAPH_FAILED);
    const int64_t* groupListTypePtr = attrs->GetAttrPointer<int64_t>(ATTR_GROUP_LIST_TYPE_INDEX_3);
    int64_t groupListType = groupListTypePtr != nullptr ? *groupListTypePtr : GROUP_LIST_TYPE_CUMSUM_0;
    tiling.gmmBaseParams.set_groupListType(groupListType);

    // 设置其余
    auto ret = CalMmTiling(context, tiling, m, k, n);
    OP_CHECK_IF(ge::GRAPH_SUCCESS != ret,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "GroupedMatmulAdd CalMmTiling error"), return ret);
    // Kernel task indices and the carry between groups are uint32. Reserve
    // room both for the carry and for the final core-stride increment.
    if (ascendcPlatform.GetCurNpuArch() == NpuArch::DAV_2201 &&
        (!transposeX || context->GetInputDesc(INDEX_IN_YREF)->GetDataType() != ge::DT_FLOAT)) {
        const uint64_t baseM = tiling.mmTilingData.get_baseM();
        const uint64_t baseN = tiling.mmTilingData.get_baseN();
        OP_CHECK_IF(baseM == 0 || baseN == 0 || aicNum <= 0,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "tiling",
                                                             "Invalid block size or core count"),
                    return ge::GRAPH_FAILED);
        const uint64_t dimM = (static_cast<uint64_t>(m) + baseM - 1) / baseM;
        const uint64_t dimN = (static_cast<uint64_t>(n) + baseN - 1) / baseN;
        const uint64_t coreStride = static_cast<uint64_t>(aicNum) * AIC_AIV_RATION;
        OP_CHECK_IF(dimM * dimN + 2 * coreStride > std::numeric_limits<uint32_t>::max(),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "tiling",
                                                             "Block count exceeds uint32 scheduling range"),
                    return ge::GRAPH_FAILED);
    }
    context->SetBlockDim(aicNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    auto baseM = tiling.mmTilingData.get_baseM();
    auto baseN = tiling.mmTilingData.get_baseN();
    uint32_t userWorkspaceSize = baseM * baseN * FP32_DATATYPE_SIZE * aicNum * DOUBLE_BUFFER_L0A_L0B;
    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = static_cast<size_t>(sysWorkspaceSize + userWorkspaceSize);

    PrintInfo(context, tiling);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingPrepare4GroupedMatmulAdd(gert::TilingParseContext* context)
{
    if (IsAdvancedNpuArch(context)) {
        return gmm_add_advanced::InitCompileInfo(context);
    }
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GroupedMatmulAdd)
    .Tiling(Tiling4GroupedMatmulAdd)
    .TilingParse<GMMCompileInfo>(TilingPrepare4GroupedMatmulAdd);
} // namespace optiling
