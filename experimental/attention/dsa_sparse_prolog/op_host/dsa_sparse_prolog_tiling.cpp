/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dsa_sparse_prolog_tiling.h"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <functional>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "err/ops_err.h"
#include "register/op_def_registry.h"

using namespace ge;
using namespace AscendC;
namespace optiling {

std::unordered_map<ge::DataType, uint32_t> DTYPE_TO_SIZE{{ge::DT_BF16, 2}, {ge::DT_FLOAT16, 2}, {ge::DT_INT32, 4}};

template <typename T>
inline T CeilDiv(T a, T b)
{
    if (b == 0) {
        return b;
    }
    return (a + b - 1) / b;
}

template <typename T>
inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

int64_t GetShapeSize(const gert::Shape& shape)
{
    size_t dimNum{shape.GetDimNum()};
    if (dimNum == 0U) {
        return 0;
    }
    int64_t shapeSize{1};
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        shapeSize *= shape.GetDim(i);
    }
    return shapeSize;
}

ge::graphStatus DsaSparsePrologTiling::GetNpuInfo()
{
    OP_CHECK_IF(context_->platformInfo == nullptr, OP_LOGE(context_->opName, "GetPlatformInfo is nullptr."),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    libapiSize_ = ascendcPlatform.GetLibApiWorkSpaceSize();

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize_);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0bSize_);

    aivNum_ = ascendcPlatform.GetCoreNumAiv();
    aicNum_ = ascendcPlatform.GetCoreNumAic();

    OP_CHECK_IF(aicNum_ == 0 || aivNum_ == 0, OP_LOGE(context_->opName, "num of core obtained is 0."),
                return GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

bool DsaSparsePrologTiling::GetMatmulType(ge::DataType getype, matmul_tiling::DataType* mmType)
{
    static struct {
        ge::DataType a;
        matmul_tiling::DataType b;
    } typeTrans[] = {{ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
                     {ge::DT_BF16, matmul_tiling::DataType::DT_BF16}};

    for (uint32_t i = 0; i < sizeof(typeTrans) / sizeof(typeTrans[0]); i++) {
        if (typeTrans[i].a == getype) {
            *mmType = typeTrans[i].b;
            return true;
        }
    }
    return false;
}

uint32_t DsaSparsePrologTiling::CalcSingleCoreN(uint32_t n, uint32_t coreNum, uint32_t alignNum)
{
    return CeilDiv(n, alignNum * coreNum) * alignNum;
}

// mm1.m = stepBatchSize
// mm1.n = singlecoreHeadSizeCq
// mm1.k = headSizeX
ge::graphStatus DsaSparsePrologTiling::FillMatmul1Tiling()
{
    uint32_t M = stepBatchSize_;
    auto dataType = context_->weightDq.desc->GetDataType();
    singlecoreHeadSizeCq_ = CalcSingleCoreN(headSizeCq_, aicNum_, BLOCK_SIZE / DTYPE_TO_SIZE[dataType]);
    mm1BlockNum_ = CeilDiv(headSizeCq_, singlecoreHeadSizeCq_);

    matmul_tiling::DataType bmm1DataType;

    if (!GetMatmulType(mmDateType_, &bmm1DataType)) {
        OP_LOGE(context_->opName, "get matmul type error");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);

    bmm1.SetShape(M, singlecoreHeadSizeCq_, headSizeX_);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1DataType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, bmm1DataType, false);

    bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1DataType);

    bmm1.SetOrgShape(M, singlecoreHeadSizeCq_, headSizeX_, headSizeX_);
    bmm1.SetBias(false);

    if (bmm1.GetTiling(*bmm1TilingData_) == -1) {
        OP_LOGE(context_->opName, "bmm1 get tiling fail");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// mm2.m = stepBatchSize
// mm2.n = singlecoreHeadSizeCkvKr
// mm2.k = headSizeX
ge::graphStatus DsaSparsePrologTiling::FillMatmul2Tiling()
{
    uint32_t M = stepBatchSize_;
    if (aicNum_ >= 9) {
        uint32_t baseN = 64U;
        mm2BlockNum_ = CeilDiv(headSizeCkv_ + dimHeadRope_, baseN);
        singlecoreHeadSizeCkvKr_ = baseN;
    } else {
        auto dataType = context_->weightDkvKr.desc->GetDataType();
        singlecoreHeadSizeCkvKr_ =
            CalcSingleCoreN(headSizeCkv_ + dimHeadRope_, aicNum_, BLOCK_SIZE / DTYPE_TO_SIZE[dataType]);
        mm2BlockNum_ = CeilDiv(headSizeCkv_ + dimHeadRope_, singlecoreHeadSizeCkvKr_);
    }

    matmul_tiling::DataType bmm2DataType;

    if (!GetMatmulType(mmDateType_, &bmm2DataType)) {
        OP_LOGE(context_->opName, "get matmul type error");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);

    bmm2.SetShape(M, singlecoreHeadSizeCkvKr_, headSizeX_);
    bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2DataType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, bmm2DataType, false);

    bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2DataType);

    bmm2.SetOrgShape(M, singlecoreHeadSizeCkvKr_, headSizeX_, headSizeX_);
    bmm2.SetBias(false);

    if (bmm2.GetTiling(*bmm2TilingData_) == -1) {
        OP_LOGE(context_->opName, "bmm2 get tiling fail");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// mm3.m = stepBatchSize
// mm3.n = singlecoreHeadSizeQcQr
// mm3.k = headSizeCq
ge::graphStatus DsaSparsePrologTiling::FillMatmul3Tiling()
{
    uint32_t M = stepBatchSize_;
    auto dataType = context_->weightUqQr.desc->GetDataType();
    auto oriM = numHeadSize_ * (dimHeadSizeQc_ + dimHeadRope_);
    singlecoreHeadSizeQcQr_ = CalcSingleCoreN(oriM, aicNum_, BLOCK_SIZE / DTYPE_TO_SIZE[dataType]);
    mm3BlockNum_ = CeilDiv(oriM, singlecoreHeadSizeQcQr_);

    matmul_tiling::DataType bmm3DataType;
    if (!GetMatmulType(mmDateType_, &bmm3DataType)) {
        OP_LOGE(context_->opName, "get matmul type error");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->platformInfo);
    matmul_tiling::MatmulApiTiling bmm3(ascendcPlatform);

    bmm3.SetShape(M, singlecoreHeadSizeQcQr_, headSizeCq_);
    bmm3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm3DataType, false);
    bmm3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::NZ, bmm3DataType, false);

    bmm3.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm3DataType);

    bmm3.SetOrgShape(M, singlecoreHeadSizeQcQr_, headSizeCq_, headSizeCq_);
    bmm3.SetBias(false);

    if (bmm3.GetTiling(*bmm3TilingData_) == -1) {
        OP_LOGE(context_->opName, "bmm3 get tiling fail");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::CheckAllInputInfo()
{
    auto tokenXType = context_->tokenX.desc->GetDataType();
    auto tokenXDimNum = context_->tokenX.shape->GetStorageShape().GetDimNum();
    OP_LOGD(context_->opName, "tokenX dtype is %s, dimnum is %lu",
            TypeUtils::DataTypeToSerialString(tokenXType).c_str(), tokenXDimNum);

    auto weightUqQrDimNum = context_->weightUqQr.shape->GetStorageShape().GetDimNum();
    OP_LOGD(context_->opName, "weightUqQr dimnum is %lu", weightUqQrDimNum);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::CheckRequiresParams()
{
    OP_CHECK_IF((context_->tokenX.shape == nullptr) || (context_->tokenX.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of tokenX is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->weightDq.shape == nullptr) || (context_->weightDq.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of weightDq is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->weightUqQr.shape == nullptr) || (context_->weightUqQr.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of weightUqQr is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->weightDkvKr.shape == nullptr) || (context_->weightDkvKr.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of weightDkvKr is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->rmsnormGammaCq.shape == nullptr) || (context_->rmsnormGammaCq.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of rmsnormGammaCq is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->rmsnormGammaCkv.shape == nullptr) || (context_->rmsnormGammaCkv.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of rmsnormGammaCkv is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->ropeSin.shape == nullptr) || (context_->ropeSin.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of ropeSin is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->ropeCos.shape == nullptr) || (context_->ropeCos.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of ropeCos is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->qNopeOut.shape == nullptr) || (context_->qNopeOut.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of qNopeOut is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->qPeOut.shape == nullptr) || (context_->qPeOut.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of qPeOut is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->ctKvOut.shape == nullptr) || (context_->ctKvOut.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of ctKvOut is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->kPeOut.shape == nullptr) || (context_->kPeOut.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of kPeOut is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->qrOut.shape == nullptr) || (context_->qrOut.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of qrOut is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->qMOut.shape == nullptr) || (context_->qMOut.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of qMOut is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->qRstdOut.shape == nullptr) || (context_->qRstdOut.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of qRstdOut is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->ckvMOut.shape == nullptr) || (context_->ckvMOut.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of ckvMOut is null."), return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->ckvRstdOut.shape == nullptr) || (context_->ckvRstdOut.desc == nullptr),
                OP_LOGE(context_->opName, "shape or desc of ckvRstdOut is null."), return ge::GRAPH_FAILED);

    OP_CHECK_IF((context_->rmsNormEspilonCq == nullptr), OP_LOGE(context_->opName, "rmsNormEspilonCq is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->rmsNormEspilonCkv == nullptr), OP_LOGE(context_->opName, "rmsNormEspilonCkv is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->mlaScaleQLora == nullptr), OP_LOGE(context_->opName, "mlaScaleQLora is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->mlaScaleKvLora == nullptr), OP_LOGE(context_->opName, "mlaScaleKvLora is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->qkNopeHeadDim == nullptr), OP_LOGE(context_->opName, "qkNopeHeadDim is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->isTypeDemotion == nullptr), OP_LOGE(context_->opName, "isTypeDemotion is null."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->applyRopeFusion == nullptr), OP_LOGE(context_->opName, "applyRopeFusion is null."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::CheckFormat()
{
    auto weightDqFormat = static_cast<ge::Format>(ge::GetPrimaryFormat(context_->weightDq.desc->GetStorageFormat()));
    OP_CHECK_IF(
        (weightDqFormat != ge::FORMAT_FRACTAL_NZ && weightDqFormat != ge::FORMAT_ND),
        OP_LOGE(context_->opName, "weightDq format is %s, witch should be NZ.", ge::GetFormatName(weightDqFormat)),
        return ge::GRAPH_FAILED);
    auto weightUqQrFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context_->weightUqQr.desc->GetStorageFormat()));
    OP_CHECK_IF(
        (weightUqQrFormat != ge::FORMAT_FRACTAL_NZ && weightUqQrFormat != ge::FORMAT_ND),
        OP_LOGE(context_->opName, "weightUqQr format is %s, witch should be NZ.", ge::GetFormatName(weightUqQrFormat)),
        return ge::GRAPH_FAILED);
    auto weightDkvKrFormat =
        static_cast<ge::Format>(ge::GetPrimaryFormat(context_->weightDkvKr.desc->GetStorageFormat()));
    OP_CHECK_IF((weightDkvKrFormat != ge::FORMAT_FRACTAL_NZ && weightDkvKrFormat != ge::FORMAT_ND),
                OP_LOGE(context_->opName, "weightDkvKr format is %s, witch should be NZ.",
                        ge::GetFormatName(weightDkvKrFormat)),
                return ge::GRAPH_FAILED);

    // 3 张权重必须同为 ND 或同为 NZ; 布局 (weightLayout) 在 ProcessBaseInputs 里结合 shape 方向一并确定
    weightIsNd_ = (weightDqFormat == ge::FORMAT_ND) ? 1U : 0U;
    OP_CHECK_IF((weightIsNd_ != ((weightUqQrFormat == ge::FORMAT_ND) ? 1U : 0U)) ||
                    (weightIsNd_ != ((weightDkvKrFormat == ge::FORMAT_ND) ? 1U : 0U)),
                OP_LOGE(context_->opName, "weightDq/weightUqQr/weightDkvKr must be all ND or all FRACTAL_NZ."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::CheckDataType()
{
    auto tokenXType = context_->tokenX.desc->GetDataType();
    mmDateType_ = tokenXType;
    OP_CHECK_IF(((tokenXType != ge::DT_BF16 && tokenXType != ge::DT_FLOAT16)),
                OP_LOGE(context_->opName, "Data type of tokenX is %s, which is not supported.",
                        TypeUtils::DataTypeToSerialString(tokenXType).c_str()),
                return ge::GRAPH_FAILED);

    auto weightDqType = context_->weightDq.desc->GetDataType();
    OP_CHECK_IF(((weightDqType != mmDateType_)),
                OP_LOGE(context_->opName, "Data type of weightDq(%s) should be same with tokenX(%s).",
                        TypeUtils::DataTypeToSerialString(weightDqType).c_str(),
                        TypeUtils::DataTypeToSerialString(mmDateType_).c_str()),
                return ge::GRAPH_FAILED);

    auto weightUqQrType = context_->weightUqQr.desc->GetDataType();
    OP_CHECK_IF(((weightUqQrType != mmDateType_)),
                OP_LOGE(context_->opName, "Data type of weightUqQr(%s) should be same with tokenX(%s).",
                        TypeUtils::DataTypeToSerialString(weightUqQrType).c_str(),
                        TypeUtils::DataTypeToSerialString(mmDateType_).c_str()),
                return ge::GRAPH_FAILED);

    auto weightDkvKrType = context_->weightDkvKr.desc->GetDataType();
    OP_CHECK_IF(((weightDkvKrType != mmDateType_)),
                OP_LOGE(context_->opName, "Data type of weightDkvKr(%s) should be same with tokenX(%s).",
                        TypeUtils::DataTypeToSerialString(weightDkvKrType).c_str(),
                        TypeUtils::DataTypeToSerialString(mmDateType_).c_str()),
                return ge::GRAPH_FAILED);

    auto rmsnormGammaCqType = context_->rmsnormGammaCq.desc->GetDataType();
    OP_CHECK_IF(rmsnormGammaCqType != ge::DT_BF16 && rmsnormGammaCqType != ge::DT_FLOAT16,
                OP_LOGE(context_->opName, "data type of rmsnormGammaCq should be bf16/fp16, but is %s.",
                        TypeUtils::DataTypeToSerialString(rmsnormGammaCqType).c_str()),
                return ge::GRAPH_FAILED);
    auto rmsnormGammaCkvType = context_->rmsnormGammaCkv.desc->GetDataType();
    OP_CHECK_IF(rmsnormGammaCkvType != ge::DT_BF16 && rmsnormGammaCkvType != ge::DT_FLOAT16,
                OP_LOGE(context_->opName, "data type of rmsnormGammaCkv should be bf16/fp16, but is %s.",
                        TypeUtils::DataTypeToSerialString(rmsnormGammaCkvType).c_str()),
                return ge::GRAPH_FAILED);

    auto ropeSinType = context_->ropeSin.desc->GetDataType();
    OP_CHECK_IF(ropeSinType != ge::DT_BF16 && ropeSinType != ge::DT_FLOAT16,
                OP_LOGE(context_->opName, "data type of ropeSin should be bf16/fp16, but is %s.",
                        TypeUtils::DataTypeToSerialString(ropeSinType).c_str()),
                return ge::GRAPH_FAILED);

    auto ropeCosType = context_->ropeCos.desc->GetDataType();
    OP_CHECK_IF(ropeCosType != ge::DT_BF16 && ropeCosType != ge::DT_FLOAT16,
                OP_LOGE(context_->opName, "data type of ropeCos should be bf16/fp16, but is %s.",
                        TypeUtils::DataTypeToSerialString(ropeCosType).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->qNopeOut.desc->GetDataType() != tokenXType,
                OP_LOGE(context_->opName, "data type of qNopeOut should be same with tokenX."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->qPeOut.desc->GetDataType() != tokenXType,
                OP_LOGE(context_->opName, "data type of qPeOut should be same with tokenX."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ctKvOut.desc->GetDataType() != tokenXType,
                OP_LOGE(context_->opName, "data type of ctKvOut should be same with tokenX."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->kPeOut.desc->GetDataType() != tokenXType,
                OP_LOGE(context_->opName, "data type of kPeOut should be same with tokenX."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->qrOut.desc->GetDataType() != tokenXType,
                OP_LOGE(context_->opName, "data type of qrOut should be same with tokenX."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->qMOut.desc->GetDataType() != tokenXType,
                OP_LOGE(context_->opName, "data type of qMOut should be same with tokenX."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->qRstdOut.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context_->opName, "data type of qRstdOut should be float32."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ckvMOut.desc->GetDataType() != tokenXType,
                OP_LOGE(context_->opName, "data type of ckvMOut should be same with tokenX."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->ckvRstdOut.desc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE(context_->opName, "data type of ckvRstdOut should be float32."), return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::CheckDimNum()
{
    auto tokenXDimNum = context_->tokenX.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (tokenXDimNum != DSA_SPARSE_PROLOG_DIM_NUM_2) && (tokenXDimNum != DSA_SPARSE_PROLOG_DIM_NUM_3),
        OP_LOGE(context_->opName, "tokenX dim num should be 2([T, He]) or 3([B, S, He]), but is %lu.", tokenXDimNum),
        return ge::GRAPH_FAILED);

    auto weightDqDimNum = context_->weightDq.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (weightDqDimNum != DSA_SPARSE_PROLOG_DIM_NUM_4) && (weightDqDimNum != DSA_SPARSE_PROLOG_DIM_NUM_2),
        OP_LOGE(context_->opName, "weightDq dim num should be 4([Hcq/16, He/16, 16, 16]) or 2([He, Hcq]), but is %lu.",
                weightDqDimNum),
        return ge::GRAPH_FAILED);

    auto weightUqQrDimNum = context_->weightUqQr.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (weightUqQrDimNum != DSA_SPARSE_PROLOG_DIM_NUM_4) && (weightUqQrDimNum != DSA_SPARSE_PROLOG_DIM_NUM_2),
        OP_LOGE(context_->opName,
                "weightUqQr dim num should be 4([N*(D+Dr)/16, Hcq/16, 16, 16]) or 2([Hcq, N*(D+Dr)]), but is %lu.",
                weightUqQrDimNum),
        return ge::GRAPH_FAILED);

    auto weightDkvKrDimNum = context_->weightDkvKr.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        (weightDkvKrDimNum != DSA_SPARSE_PROLOG_DIM_NUM_4) && (weightDkvKrDimNum != DSA_SPARSE_PROLOG_DIM_NUM_2),
        OP_LOGE(context_->opName,
                "weightDkvKr dim num should be 4([(Hckv+Dr)/16, He/16, 16, 16]) or 2([He, Hckv+Dr]), but is %lu.",
                weightDkvKrDimNum),
        return ge::GRAPH_FAILED);

    auto rmsnormGammaCqDimNum = context_->rmsnormGammaCq.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((rmsnormGammaCqDimNum != DSA_SPARSE_PROLOG_DIM_NUM_1),
                OP_LOGE(context_->opName, "rmsnormGammaCq dim num should be 1, but is %lu.", rmsnormGammaCqDimNum),
                return ge::GRAPH_FAILED);

    auto rmsnormGammaCkvDimNum = context_->rmsnormGammaCkv.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((rmsnormGammaCkvDimNum != DSA_SPARSE_PROLOG_DIM_NUM_1),
                OP_LOGE(context_->opName, "rmsnormGammaCkv dim num should be 1, but is %lu.", rmsnormGammaCkvDimNum),
                return ge::GRAPH_FAILED);

    auto ropeSinDimNum = context_->ropeSin.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((ropeSinDimNum != tokenXDimNum),
                OP_LOGE(context_->opName, "ropeSin dim num should be same with tokenX(%lu), but is %lu.", tokenXDimNum,
                        ropeSinDimNum),
                return ge::GRAPH_FAILED);

    auto ropeCosDimNum = context_->ropeCos.shape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((ropeCosDimNum != tokenXDimNum),
                OP_LOGE(context_->opName, "ropeCos dim num should be same with tokenX(%lu), but is %lu.", tokenXDimNum,
                        ropeCosDimNum),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus OpsErrIfEmpty(const char* opName, const gert::Shape& shape, const std::string&& inputName)
{
    OP_CHECK_IF(GetShapeSize(shape) == 0,
                OP_LOGE(opName, "%s is an empty tensor, please check your inputs!", inputName.c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::PreventEmptyTensor()
{
    using StatusFunction = std::function<ge::graphStatus()>;

    std::vector<StatusFunction> PreventEmptyFuncs{
        std::bind(&OpsErrIfEmpty, context_->opName, context_->tokenX.shape->GetStorageShape(), "tokenX"),
        std::bind(&OpsErrIfEmpty, context_->opName, context_->weightDq.shape->GetStorageShape(), "weightDq"),
        std::bind(&OpsErrIfEmpty, context_->opName, context_->weightUqQr.shape->GetStorageShape(), "weightUqQr"),
        std::bind(&OpsErrIfEmpty, context_->opName, context_->weightDkvKr.shape->GetStorageShape(), "weightDkvKr"),
        std::bind(&OpsErrIfEmpty, context_->opName, context_->rmsnormGammaCq.shape->GetStorageShape(),
                  "rmsnormGammaCq"),
        std::bind(&OpsErrIfEmpty, context_->opName, context_->rmsnormGammaCkv.shape->GetStorageShape(),
                  "rmsnormGammaCkv"),
        std::bind(&OpsErrIfEmpty, context_->opName, context_->ropeSin.shape->GetStorageShape(), "ropeSin"),
        std::bind(&OpsErrIfEmpty, context_->opName, context_->ropeCos.shape->GetStorageShape(), "ropeCos")};

    for (const auto& func : PreventEmptyFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::ProcessBaseInputs()
{
    // tokenX : [B, S, He] or [T, He]; tokenSize = product of leading dims
    auto tokenXDimNum = context_->tokenX.shape->GetStorageShape().GetDimNum();
    tokenSize_ = 1;
    for (size_t i = 0; i + 1 < tokenXDimNum; i++) {
        tokenSize_ *= context_->tokenX.shape->GetStorageShape().GetDim(i);
    }
    headSizeX_ = context_->tokenX.shape->GetStorageShape().GetDim(tokenXDimNum - 1);

    // weightDq : [K,N]=[He,Hcq] 或 [N,K]=[Hcq,He]
    // NZ 4D: [K,N] 为 [Hcq/axisSize, He/16, 16, axisSize], [N,K] 为 [He/16, Hcq/axisSize, axisSize, 16]
    weightDqAxisSize_ = 32U / ge::GetSizeByDataType(context_->weightDq.desc->GetDataType());
    auto weightDqDimNum = context_->weightDq.shape->GetStorageShape().GetDimNum();
    auto weightDqDim0 = context_->weightDq.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_0);
    auto weightDqDim1 = context_->weightDq.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_1);
    uint32_t weightDqNk = 0;
    if (weightDqDimNum == DSA_SPARSE_PROLOG_DIM_NUM_4) {
        if (weightDqDim1 == static_cast<int64_t>(headSizeX_ / fixAxisSize_)) {
            headSizeCq_ = weightDqAxisSize_ * weightDqDim0;
        } else {
            weightDqNk = 1;
            headSizeCq_ = weightDqAxisSize_ * weightDqDim1;
        }
    } else {
        if (weightDqDim0 == static_cast<int64_t>(headSizeX_)) {
            headSizeCq_ = weightDqDim1;
        } else {
            weightDqNk = 1;
            headSizeCq_ = weightDqDim0;
        }
    }

    // rmsnormGammaCkv : [Hckv]
    headSizeCkv_ = context_->rmsnormGammaCkv.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_0);

    // ropeSin/ropeCos : [..., Dr]
    dimHeadRope_ = context_->ropeCos.shape->GetStorageShape().GetDim(tokenXDimNum - 1);
    headSizeKr_ = dimHeadRope_;

    // dimHeadSizeQc from attr qkNopeHeadDim
    dimHeadSizeQc_ = static_cast<uint32_t>(*context_->qkNopeHeadDim);
    OP_CHECK_IF(dimHeadSizeQc_ == 0, OP_LOGE(context_->opName, "qkNopeHeadDim should not be 0."),
                return ge::GRAPH_FAILED);

    // weightUqQr : [K,N]=[Hcq, N*(D+Dr)] 或 [N,K]=[N*(D+Dr), Hcq]; 规则同 weightDq (K = Hcq)
    weightUqQrAxisSize_ = 32U / ge::GetSizeByDataType(context_->weightUqQr.desc->GetDataType());
    auto weightUqQrDimNum = context_->weightUqQr.shape->GetStorageShape().GetDimNum();
    auto weightUqQrDim0 = context_->weightUqQr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_0);
    auto weightUqQrDim1 = context_->weightUqQr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_1);
    uint32_t weightUqQrNk = 0;
    uint32_t weightUqQrWidth = 0;
    if (weightUqQrDimNum == DSA_SPARSE_PROLOG_DIM_NUM_4) {
        if (weightUqQrDim1 == static_cast<int64_t>(headSizeCq_ / fixAxisSize_)) {
            weightUqQrWidth = weightUqQrAxisSize_ * weightUqQrDim0;
        } else {
            weightUqQrNk = 1;
            weightUqQrWidth = weightUqQrAxisSize_ * weightUqQrDim1;
        }
    } else {
        if (weightUqQrDim0 == static_cast<int64_t>(headSizeCq_)) {
            weightUqQrWidth = weightUqQrDim1;
        } else {
            weightUqQrNk = 1;
            weightUqQrWidth = weightUqQrDim0;
        }
    }
    uint32_t headDimQcQr = dimHeadSizeQc_ + dimHeadRope_;
    OP_CHECK_IF((weightUqQrWidth % headDimQcQr) != 0,
                OP_LOGE(context_->opName, "width of weightUqQr(%u) should be divisible by (D + Dr)(%u).",
                        weightUqQrWidth, headDimQcQr),
                return ge::GRAPH_FAILED);
    numHeadSize_ = weightUqQrWidth / headDimQcQr;

    headSizeQc_ = dimHeadSizeQc_ * numHeadSize_;
    headSizeQr_ = dimHeadRope_ * numHeadSize_;

    OP_CHECK_IF((headSizeCq_ == 0) || (headSizeCkv_ == 0),
                OP_LOGE(context_->opName, "Hcq(%u) and Hckv(%u) should not be 0", headSizeCq_, headSizeCkv_),
                return ge::GRAPH_FAILED);
    reciprocalCq_ = 1.0f / (headSizeCq_);
    epsilonCq_ = *context_->rmsNormEspilonCq;
    reciprocalCkv_ = 1.0f / (headSizeCkv_);
    epsilonCkv_ = *context_->rmsNormEspilonCkv;

    // scale attr <= 0 : derive sqrt(He / Hcq), sqrt(He / Hckv)
    mlaScaleQLora_ = *context_->mlaScaleQLora;
    if (mlaScaleQLora_ <= 1e-6f) {
        mlaScaleQLora_ = std::sqrt(static_cast<float>(headSizeX_) / static_cast<float>(headSizeCq_));
    }
    mlaScaleKvLora_ = *context_->mlaScaleKvLora;
    if (mlaScaleKvLora_ <= 1e-6f) {
        mlaScaleKvLora_ = std::sqrt(static_cast<float>(headSizeX_) / static_cast<float>(headSizeCkv_));
    }

    isTypeDemotion_ = *(context_->isTypeDemotion);
    applyRopeFusion_ = *(context_->applyRopeFusion);

    stepBatchSize_ = std::min(128U, tokenSize_);
    vectorBlockNum_ = std::min(stepBatchSize_, aivNum_);

    // weightDkvKr 方向检测 (K = He): 2D [He, Hckv+Dr] / [Hckv+Dr, He]; NZ 4D 规则同 weightDq
    weightDkvKrAxisSize_ = 32U / ge::GetSizeByDataType(context_->weightDkvKr.desc->GetDataType());
    auto weightDkvKrDimNum = context_->weightDkvKr.shape->GetStorageShape().GetDimNum();
    uint32_t weightDkvKrNk = 0;
    if (weightDkvKrDimNum == DSA_SPARSE_PROLOG_DIM_NUM_4) {
        weightDkvKrNk = (context_->weightDkvKr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_1) ==
                         static_cast<int64_t>(headSizeX_ / fixAxisSize_)) ?
                            0U :
                            1U;
    } else {
        weightDkvKrNk = (context_->weightDkvKr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_0) ==
                         static_cast<int64_t>(headSizeX_)) ?
                            0U :
                            1U;
    }

    // 3 张权重逻辑 shape 必须同为 [K,N] 或同为 [N,K]; weightLayout: bit0=ND, bit1=[N,K]
    OP_CHECK_IF(
        (weightDqNk != weightUqQrNk) || (weightDqNk != weightDkvKrNk),
        OP_LOGE(context_->opName, "weightDq/weightUqQr/weightDkvKr logical shape must be all [K,N] or all [N,K]."),
        return ge::GRAPH_FAILED);
    weightIsNk_ = weightDqNk;
    baseParams_->set_weightLayout(weightIsNd_ | (weightIsNk_ << 1U));

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::CheckShape()
{
    // tokenX : [T, He] or [B, S, He]
    auto tokenXDimNum = context_->tokenX.shape->GetStorageShape().GetDimNum();
    auto headSizeXDim = context_->tokenX.shape->GetStorageShape().GetDim(tokenXDimNum - 1);
    OP_CHECK_IF(headSizeXDim != headSizeX_,
                OP_LOGE(context_->opName, "last dim of tokenX should be %u, but is %lu.", headSizeX_, headSizeXDim),
                return ge::GRAPH_FAILED);

    // weightDq : [K,N]=[He,Hcq] / [N,K]=[Hcq,He]; NZ: [K,N]=[Hcq/axis, He/16, 16, axis] / [N,K]=[He/16, Hcq/axis, axis,
    // 16]
    auto headSizeCqAxisSize = headSizeCq_ / weightDqAxisSize_;
    auto headSizeXAxisSize = headSizeX_ / fixAxisSize_;
    auto weightDqDimNum = context_->weightDq.shape->GetStorageShape().GetDimNum();
    auto weightDqDim0 = context_->weightDq.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_0);
    auto weightDqDim1 = context_->weightDq.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_1);
    if (weightDqDimNum == DSA_SPARSE_PROLOG_DIM_NUM_2) {
        auto expectDim0 = (weightIsNk_ != 0) ? headSizeCq_ : headSizeX_;
        auto expectDim1 = (weightIsNk_ != 0) ? headSizeX_ : headSizeCq_;
        OP_CHECK_IF(
            (weightDqDim0 != static_cast<int64_t>(expectDim0)) || (weightDqDim1 != static_cast<int64_t>(expectDim1)),
            OP_LOGE(context_->opName, "shape of weightDq should be [%u, %u], but is [%lu, %lu].", expectDim0,
                    expectDim1, weightDqDim0, weightDqDim1),
            return ge::GRAPH_FAILED);
    } else {
        auto weightDqDim2 = context_->weightDq.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_2);
        auto weightDqDim3 = context_->weightDq.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_3);
        auto expectDim0 = (weightIsNk_ != 0) ? headSizeXAxisSize : headSizeCqAxisSize;
        auto expectDim1 = (weightIsNk_ != 0) ? headSizeCqAxisSize : headSizeXAxisSize;
        auto expectDim2 = (weightIsNk_ != 0) ? weightDqAxisSize_ : fixAxisSize_;
        auto expectDim3 = (weightIsNk_ != 0) ? fixAxisSize_ : weightDqAxisSize_;
        OP_CHECK_IF((weightDqDim0 != static_cast<int64_t>(expectDim0)) ||
                        (weightDqDim1 != static_cast<int64_t>(expectDim1)) ||
                        (weightDqDim2 != static_cast<int64_t>(expectDim2)) ||
                        (weightDqDim3 != static_cast<int64_t>(expectDim3)),
                    OP_LOGE(context_->opName,
                            "shape of weightDq should be [%u, %u, %u, %u], but is [%lu, %lu, %lu, %lu].", expectDim0,
                            expectDim1, expectDim2, expectDim3, weightDqDim0, weightDqDim1, weightDqDim2, weightDqDim3),
                    return ge::GRAPH_FAILED);
    }

    // weightUqQr : [K,N]=[Hcq, N*(D+Dr)] / [N,K]=[N*(D+Dr), Hcq]; NZ 规则同 weightDq (K = Hcq)
    auto weightUqQrDimNum = context_->weightUqQr.shape->GetStorageShape().GetDimNum();
    auto weightUqQrDim0 = context_->weightUqQr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_0);
    auto weightUqQrDim1 = context_->weightUqQr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_1);
    auto numHeadQcAndRope = numHeadSize_ * (dimHeadSizeQc_ + dimHeadRope_);
    auto numHeadQcAndRopeAxisSize = numHeadQcAndRope / weightUqQrAxisSize_;
    auto headSizeCqAxisSize1 = headSizeCq_ / fixAxisSize_;
    if (weightUqQrDimNum == DSA_SPARSE_PROLOG_DIM_NUM_2) {
        auto expectDim0 = (weightIsNk_ != 0) ? numHeadQcAndRope : headSizeCq_;
        auto expectDim1 = (weightIsNk_ != 0) ? headSizeCq_ : numHeadQcAndRope;
        OP_CHECK_IF((weightUqQrDim0 != static_cast<int64_t>(expectDim0)) ||
                        (weightUqQrDim1 != static_cast<int64_t>(expectDim1)),
                    OP_LOGE(context_->opName, "shape of weightUqQr should be [%u, %u], but is [%lu, %lu].", expectDim0,
                            expectDim1, weightUqQrDim0, weightUqQrDim1),
                    return ge::GRAPH_FAILED);
    } else {
        auto weightUqQrDim2 = context_->weightUqQr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_2);
        auto weightUqQrDim3 = context_->weightUqQr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_3);
        auto expectDim0 = (weightIsNk_ != 0) ? headSizeCqAxisSize1 : numHeadQcAndRopeAxisSize;
        auto expectDim1 = (weightIsNk_ != 0) ? numHeadQcAndRopeAxisSize : headSizeCqAxisSize1;
        auto expectDim2 = (weightIsNk_ != 0) ? weightUqQrAxisSize_ : fixAxisSize_;
        auto expectDim3 = (weightIsNk_ != 0) ? fixAxisSize_ : weightUqQrAxisSize_;
        OP_CHECK_IF(
            (weightUqQrDim0 != static_cast<int64_t>(expectDim0)) ||
                (weightUqQrDim1 != static_cast<int64_t>(expectDim1)) ||
                (weightUqQrDim2 != static_cast<int64_t>(expectDim2)) ||
                (weightUqQrDim3 != static_cast<int64_t>(expectDim3)),
            OP_LOGE(context_->opName, "shape of weightUqQr should be [%u, %u, %u, %u], but is [%lu, %lu, %lu, %lu].",
                    expectDim0, expectDim1, expectDim2, expectDim3, weightUqQrDim0, weightUqQrDim1, weightUqQrDim2,
                    weightUqQrDim3),
            return ge::GRAPH_FAILED);
    }

    // weightDkvKr : [K,N]=[He, Hckv+Dr] / [N,K]=[Hckv+Dr, He]; NZ 规则同 weightDq
    auto ckvKrSize = headSizeCkv_ + dimHeadRope_;
    auto ckvKrAxisSize = ckvKrSize / weightDkvKrAxisSize_;
    auto weightDkvKrDimNum = context_->weightDkvKr.shape->GetStorageShape().GetDimNum();
    auto weightDkvKrDim0 = context_->weightDkvKr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_0);
    auto weightDkvKrDim1 = context_->weightDkvKr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_1);
    if (weightDkvKrDimNum == DSA_SPARSE_PROLOG_DIM_NUM_2) {
        auto expectDim0 = (weightIsNk_ != 0) ? ckvKrSize : headSizeX_;
        auto expectDim1 = (weightIsNk_ != 0) ? headSizeX_ : ckvKrSize;
        OP_CHECK_IF((weightDkvKrDim0 != static_cast<int64_t>(expectDim0)) ||
                        (weightDkvKrDim1 != static_cast<int64_t>(expectDim1)),
                    OP_LOGE(context_->opName, "shape of weightDkvKr should be [%u, %u], but is [%lu, %lu].", expectDim0,
                            expectDim1, weightDkvKrDim0, weightDkvKrDim1),
                    return ge::GRAPH_FAILED);
    } else {
        auto weightDkvKrDim2 = context_->weightDkvKr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_2);
        auto weightDkvKrDim3 = context_->weightDkvKr.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_3);
        auto expectDim0 = (weightIsNk_ != 0) ? headSizeXAxisSize : ckvKrAxisSize;
        auto expectDim1 = (weightIsNk_ != 0) ? ckvKrAxisSize : headSizeXAxisSize;
        auto expectDim2 = (weightIsNk_ != 0) ? weightDkvKrAxisSize_ : fixAxisSize_;
        auto expectDim3 = (weightIsNk_ != 0) ? fixAxisSize_ : weightDkvKrAxisSize_;
        OP_CHECK_IF(
            (weightDkvKrDim0 != static_cast<int64_t>(expectDim0)) ||
                (weightDkvKrDim1 != static_cast<int64_t>(expectDim1)) ||
                (weightDkvKrDim2 != static_cast<int64_t>(expectDim2)) ||
                (weightDkvKrDim3 != static_cast<int64_t>(expectDim3)),
            OP_LOGE(context_->opName, "shape of weightDkvKr should be [%u, %u, %u, %u], but is [%lu, %lu, %lu, %lu].",
                    expectDim0, expectDim1, expectDim2, expectDim3, weightDkvKrDim0, weightDkvKrDim1, weightDkvKrDim2,
                    weightDkvKrDim3),
            return ge::GRAPH_FAILED);
    }

    // rmsnormGammaCq : [Hcq]
    auto rmsnormGammaCqDim0 = context_->rmsnormGammaCq.shape->GetStorageShape().GetDim(DSA_SPARSE_PROLOG_DIM_INDEX_0);
    OP_CHECK_IF(rmsnormGammaCqDim0 != headSizeCq_,
                OP_LOGE(context_->opName, "shape of rmsnormGammaCq should be [%u], but is [%lu].", headSizeCq_,
                        rmsnormGammaCqDim0),
                return ge::GRAPH_FAILED);

    // ropeSin/ropeCos : [..., Dr], leading dims product == tokenSize
    auto ropeSinDimNum = context_->ropeSin.shape->GetStorageShape().GetDimNum();
    uint32_t ropeTokenSize = 1;
    for (size_t i = 0; i + 1 < ropeSinDimNum; i++) {
        ropeTokenSize *= context_->ropeSin.shape->GetStorageShape().GetDim(i);
    }
    auto ropeSinLastDim = context_->ropeSin.shape->GetStorageShape().GetDim(ropeSinDimNum - 1);
    OP_CHECK_IF((ropeTokenSize != tokenSize_) || (ropeSinLastDim != dimHeadRope_),
                OP_LOGE(context_->opName, "shape of ropeSin should be [%u, %u](after flatten), but last dim is %lu.",
                        tokenSize_, dimHeadRope_, ropeSinLastDim),
                return ge::GRAPH_FAILED);
    auto ropeCosDimNum = context_->ropeCos.shape->GetStorageShape().GetDimNum();
    uint32_t ropeCosTokenSize = 1;
    for (size_t i = 0; i + 1 < ropeCosDimNum; i++) {
        ropeCosTokenSize *= context_->ropeCos.shape->GetStorageShape().GetDim(i);
    }
    auto ropeCosLastDim = context_->ropeCos.shape->GetStorageShape().GetDim(ropeCosDimNum - 1);
    OP_CHECK_IF((ropeCosTokenSize != tokenSize_) || (ropeCosLastDim != dimHeadRope_),
                OP_LOGE(context_->opName, "shape of ropeCos should be [%u, %u](after flatten), but last dim is %lu.",
                        tokenSize_, dimHeadRope_, ropeCosLastDim),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::FillTiling()
{
    baseParams_->set_stepBatchSize(stepBatchSize_);
    baseParams_->set_ubBufferSize(ubBufferSize_);
    baseParams_->set_tokenSize(tokenSize_);
    baseParams_->set_headSizeX(headSizeX_);
    baseParams_->set_headSizeCq(headSizeCq_);
    baseParams_->set_headSizeCkv(headSizeCkv_);
    baseParams_->set_headSizeQc(headSizeQc_);
    baseParams_->set_headSizeQr(headSizeQr_);
    baseParams_->set_headSizeKr(headSizeKr_);
    baseParams_->set_numHeadSize(numHeadSize_);
    baseParams_->set_dimHeadSizeQc(dimHeadSizeQc_);
    baseParams_->set_dimHeadRope(dimHeadRope_);
    baseParams_->set_mm1BlockNum(mm1BlockNum_);
    baseParams_->set_mm2BlockNum(mm2BlockNum_);
    baseParams_->set_mm3BlockNum(mm3BlockNum_);
    baseParams_->set_mm1SingleCoreN(singlecoreHeadSizeCq_);
    baseParams_->set_mm2SingleCoreN(singlecoreHeadSizeCkvKr_);
    baseParams_->set_mm3SingleCoreN(singlecoreHeadSizeQcQr_);
    baseParams_->set_vectorBlockNum(vectorBlockNum_);
    baseParams_->set_reciprocalCq(reciprocalCq_);
    baseParams_->set_epsilonCq(epsilonCq_);
    baseParams_->set_reciprocalCkv(reciprocalCkv_);
    baseParams_->set_epsilonCkv(epsilonCkv_);
    baseParams_->set_isTypeDemotion(isTypeDemotion_);
    baseParams_->set_applyRopeFusion(applyRopeFusion_);
    baseParams_->set_mlaScaleQLora(mlaScaleQLora_);
    baseParams_->set_mlaScaleKvLora(mlaScaleKvLora_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::CalcWorkSpace()
{
    workspaceSize_ = libapiSize_;
    // mmCqRes + rmsNormCqRes
    workspaceSize_ += stepBatchSize_ * headSizeCq_ * NUM_BYTES_BF16 * 2;
    // mmCkvKrRes
    workspaceSize_ += stepBatchSize_ * (headSizeCkv_ + dimHeadRope_) * NUM_BYTES_BF16;
    // mmQcQrRes (fp32 accumulate buffer reused by scale/rope stages)
    workspaceSize_ += stepBatchSize_ * numHeadSize_ * (dimHeadSizeQc_ + dimHeadRope_) * NUM_BYTES_FP32;
    // rstd scratch（每个 AIV 核 8 fp32 = 32B 独占槽位，q_rstd/ckv_rstd 分时复用）
    workspaceSize_ += vectorBlockNum_ * 8 * NUM_BYTES_FP32;

    if (context_->workSpaces) {
        context_->workSpaces[0] = workspaceSize_;
    }
    OP_LOGD(context_->opName, "Tiling info: workspaceSize_ = %zu", workspaceSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::CalcUbBufferSize()
{
    uint32_t gatherBufferSize = 2 * dimHeadRope_ * NUM_BYTES_BF16 * CeilDiv(stepBatchSize_, aivNum_);
    uint32_t rmsNormCqBufferSize = headSizeCq_ * (NUM_BYTES_FP32 * 3 + NUM_BYTES_BF16) + 256;
    uint32_t scaleCol = headSizeQc_ + headSizeQr_;
    uint32_t scaleQcQrBufferSize = scaleCol * NUM_BYTES_FP32 + scaleCol * NUM_BYTES_BF16;
    uint32_t rmsNormCkvBufSize = 8 * 1024 + 128 * 4;
    uint32_t ropeKrBufferSize = 8 * 1024;
    uint32_t ropeQrBufferSize = headSizeQr_ * (4 * NUM_BYTES_FP32 + NUM_BYTES_BF16);
    std::vector<uint32_t> bufferSizeList = {gatherBufferSize,  rmsNormCqBufferSize, scaleQcQrBufferSize,
                                            rmsNormCkvBufSize, ropeKrBufferSize,    ropeQrBufferSize};
    ubBufferSize_ = *std::max_element(bufferSizeList.begin(), bufferSizeList.end());
    if (ubBufferSize_ > ubSize_) {
        uint32_t ubSize = ubSize_;
        OP_LOGE(context_->opName, "exceed ub buffer limits, need %u, only have %u", ubBufferSize_, ubSize);
        return ge::GRAPH_FAILED;
    }
    OP_LOGD(context_->opName, "Tiling info: ubBufferSize_ = %zu", ubBufferSize_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::GenTilingKey()
{
    uint32_t typeValue = 0;
    switch (mmDateType_) {
        case ge::DT_FLOAT16:
            typeValue = 0;
            break;
        case ge::DT_BF16:
            typeValue = 1U;
            break;
        default:
            OP_LOGE(context_->opName, "not support tokenX Type %d", mmDateType_);
            return ge::GRAPH_FAILED;
    }

    context_->tilingKey = DSA_SPARSE_PROLOG_TILINGKEY_BASE_OFFSET + typeValue * DSA_SPARSE_PROLOG_TYPE_OFFSET;
    OP_LOGD(context_->opName, "DsaSparseProlog tilingKey:%llu", context_->tilingKey);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::RunBigKernelTiling(DsaSparsePrologContext& context,
                                                          DsaSparsePrologTilingData& tilingData, bool isWorkspace)
{
    this->context_ = &context;
    this->bmm1TilingData_ = &tilingData.bmm1TilingData;
    this->bmm2TilingData_ = &tilingData.bmm2TilingData;
    this->bmm3TilingData_ = &tilingData.bmm3TilingData;
    this->baseParams_ = &tilingData.baseParams;
    this->isWorkspace_ = isWorkspace;

    using StatusFunction = std::function<ge::graphStatus()>;

    std::vector<StatusFunction> tilingFuncs{std::bind(&DsaSparsePrologTiling::GetNpuInfo, this),
                                            std::bind(&DsaSparsePrologTiling::CheckAllInputInfo, this),
                                            std::bind(&DsaSparsePrologTiling::CheckRequiresParams, this),
                                            std::bind(&DsaSparsePrologTiling::CheckDataType, this),
                                            std::bind(&DsaSparsePrologTiling::CheckFormat, this),
                                            std::bind(&DsaSparsePrologTiling::CheckDimNum, this),
                                            std::bind(&DsaSparsePrologTiling::PreventEmptyTensor, this),
                                            std::bind(&DsaSparsePrologTiling::ProcessBaseInputs, this),
                                            std::bind(&DsaSparsePrologTiling::CheckShape, this),
                                            std::bind(&DsaSparsePrologTiling::FillMatmul1Tiling, this),
                                            std::bind(&DsaSparsePrologTiling::FillMatmul2Tiling, this),
                                            std::bind(&DsaSparsePrologTiling::FillMatmul3Tiling, this),
                                            std::bind(&DsaSparsePrologTiling::CalcUbBufferSize, this),
                                            std::bind(&DsaSparsePrologTiling::FillTiling, this),
                                            std::bind(&DsaSparsePrologTiling::CalcWorkSpace, this),
                                            std::bind(&DsaSparsePrologTiling::GenTilingKey, this)};

    for (const auto& func : tilingFuncs) {
        if (func() != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    context_->blockDim = aicNum_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::ConvertContext(gert::TilingContext& context,
                                                      DsaSparsePrologContext& mlaPrologContext)
{
    if (context.GetNodeName() == nullptr) {
        OP_LOGE("DsaSparseProlog", "opName got from TilingContext is nullptr");
        return ge::GRAPH_FAILED;
    }

    mlaPrologContext.opName = context.GetNodeName();
    mlaPrologContext.platformInfo = context.GetPlatformInfo();
    mlaPrologContext.tokenX.desc = context.GetInputDesc(TOKEN_X_INPUT_INDEX);
    mlaPrologContext.tokenX.shape = context.GetInputShape(TOKEN_X_INPUT_INDEX);
    mlaPrologContext.weightDq.desc = context.GetInputDesc(WEIGHT_DQ_INPUT_INDEX);
    mlaPrologContext.weightDq.shape = context.GetInputShape(WEIGHT_DQ_INPUT_INDEX);
    mlaPrologContext.weightUqQr.desc = context.GetInputDesc(WEIGHT_UQ_QR_INPUT_INDEX);
    mlaPrologContext.weightUqQr.shape = context.GetInputShape(WEIGHT_UQ_QR_INPUT_INDEX);
    mlaPrologContext.weightDkvKr.desc = context.GetInputDesc(WEIGHT_DKV_KR_INPUT_INDEX);
    mlaPrologContext.weightDkvKr.shape = context.GetInputShape(WEIGHT_DKV_KR_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCq.desc = context.GetInputDesc(RMSNORM_GAMMA_CQ_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCq.shape = context.GetInputShape(RMSNORM_GAMMA_CQ_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCkv.desc = context.GetInputDesc(RMS_NORM_GAMMA_CKV_INPUT_INDEX);
    mlaPrologContext.rmsnormGammaCkv.shape = context.GetInputShape(RMS_NORM_GAMMA_CKV_INPUT_INDEX);
    mlaPrologContext.ropeSin.desc = context.GetInputDesc(ROPE_SIN_INPUT_INDEX);
    mlaPrologContext.ropeSin.shape = context.GetInputShape(ROPE_SIN_INPUT_INDEX);
    mlaPrologContext.ropeCos.desc = context.GetInputDesc(ROPE_COS_INPUT_INDEX);
    mlaPrologContext.ropeCos.shape = context.GetInputShape(ROPE_COS_INPUT_INDEX);
    mlaPrologContext.qNopeOut.desc = context.GetOutputDesc(Q_NOPE_OUTPUT_INDEX);
    mlaPrologContext.qNopeOut.shape = context.GetOutputShape(Q_NOPE_OUTPUT_INDEX);
    mlaPrologContext.qPeOut.desc = context.GetOutputDesc(Q_PE_OUTPUT_INDEX);
    mlaPrologContext.qPeOut.shape = context.GetOutputShape(Q_PE_OUTPUT_INDEX);
    mlaPrologContext.ctKvOut.desc = context.GetOutputDesc(CT_KV_OUTPUT_INDEX);
    mlaPrologContext.ctKvOut.shape = context.GetOutputShape(CT_KV_OUTPUT_INDEX);
    mlaPrologContext.kPeOut.desc = context.GetOutputDesc(K_PE_OUTPUT_INDEX);
    mlaPrologContext.kPeOut.shape = context.GetOutputShape(K_PE_OUTPUT_INDEX);
    mlaPrologContext.qrOut.desc = context.GetOutputDesc(QR_OUTPUT_INDEX);
    mlaPrologContext.qrOut.shape = context.GetOutputShape(QR_OUTPUT_INDEX);
    mlaPrologContext.qMOut.desc = context.GetOutputDesc(Q_M_OUTPUT_INDEX);
    mlaPrologContext.qMOut.shape = context.GetOutputShape(Q_M_OUTPUT_INDEX);
    mlaPrologContext.qRstdOut.desc = context.GetOutputDesc(Q_RSTD_OUTPUT_INDEX);
    mlaPrologContext.qRstdOut.shape = context.GetOutputShape(Q_RSTD_OUTPUT_INDEX);
    mlaPrologContext.ckvMOut.desc = context.GetOutputDesc(CKV_M_OUTPUT_INDEX);
    mlaPrologContext.ckvMOut.shape = context.GetOutputShape(CKV_M_OUTPUT_INDEX);
    mlaPrologContext.ckvRstdOut.desc = context.GetOutputDesc(CKV_RSTD_OUTPUT_INDEX);
    mlaPrologContext.ckvRstdOut.shape = context.GetOutputShape(CKV_RSTD_OUTPUT_INDEX);

    auto attrs = context.GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE(context.GetNodeName(), "attrs got from ge is nullptr"),
                return ge::GRAPH_FAILED);
    mlaPrologContext.rmsNormEspilonCq = attrs->GetAttrPointer<float>(RMS_NORM_EPSILON_CQ_ATTR_INDEX);
    mlaPrologContext.rmsNormEspilonCkv = attrs->GetAttrPointer<float>(RMS_NORM_EPSILON_CKV_ATTR_INDEX);
    mlaPrologContext.mlaScaleQLora = attrs->GetAttrPointer<float>(MLA_SCALE_Q_LORA_ATTR_INDEX);
    mlaPrologContext.mlaScaleKvLora = attrs->GetAttrPointer<float>(MLA_SCALE_KV_LORA_ATTR_INDEX);
    mlaPrologContext.qkNopeHeadDim = attrs->GetAttrPointer<int64_t>(QK_NOPE_HEAD_DIM_ATTR_INDEX);
    mlaPrologContext.isTypeDemotion = attrs->GetAttrPointer<bool>(TYPE_DEMOTION_INDEX);
    mlaPrologContext.applyRopeFusion = attrs->GetAttrPointer<bool>(ROPE_FUSION_INDEX);

    OP_CHECK_IF(context.GetWorkspaceSizes(1) == nullptr,
                OP_LOGE(context.GetNodeName(), "workSpaceSize got from ge is nullptr"), return ge::GRAPH_FAILED);
    mlaPrologContext.workSpaces = context.GetWorkspaceSizes(1);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DsaSparsePrologTiling::DsaSparsePrologSetTilingData(gert::TilingContext& context,
                                                                    DsaSparsePrologTilingData& tilingData)
{
    OP_CHECK_IF(context.GetRawTilingData() == nullptr,
                OP_LOGE(context.GetNodeName(), "RawTilingData got from ge context is nullptr."),
                return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(context.GetRawTilingData()->GetData(), context.GetRawTilingData()->GetCapacity());
    context.GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

MLA_EXTERN_C ge::graphStatus TilingDsaSparseProlog(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("DsaSparseProlog", "Context is nullptr."), return ge::GRAPH_FAILED);

    DsaSparsePrologContext mlaPrologContext{};
    if (DsaSparsePrologTiling::ConvertContext(*context, mlaPrologContext) != ge::GRAPH_SUCCESS) {
        OP_LOGE(context->GetNodeName(), "Error occurred while converting tilingContext to DsaSparseProlog context");
        return ge::GRAPH_FAILED;
    }

    DsaSparsePrologTiling mlaPrologTiling;
    DsaSparsePrologTilingData tilingData;
    if (mlaPrologTiling.RunBigKernelTiling(mlaPrologContext, tilingData) == ge::SUCCESS) {
        context->SetTilingKey(mlaPrologContext.tilingKey);
        context->SetBlockDim(mlaPrologContext.blockDim);
        context->SetScheduleMode(1); // 1: batchmode模式
        mlaPrologTiling.DsaSparsePrologSetTilingData(*context, tilingData);
        return ge::GRAPH_SUCCESS;
    }
    return ge::GRAPH_FAILED;
}
} // namespace optiling
