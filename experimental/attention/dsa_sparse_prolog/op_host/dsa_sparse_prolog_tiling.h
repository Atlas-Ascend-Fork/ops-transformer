/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DSA_SPARSE_PROLOG_TILING_H
#define DSA_SPARSE_PROLOG_TILING_H

#include <cstdint>
#include <string>
#include <vector>
#include <queue>
#include <map>
#include <unordered_map>
#include <set>
#include "register/tilingdata_base.h"
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>
#include "exe_graph/runtime/tiling_context.h"
#include "register/op_def_registry.h"

#ifdef ASCENDC_OP_TEST
#define MLA_EXTERN_C extern "C"
#else
#define MLA_EXTERN_C
#endif

namespace optiling {

// INPUT
constexpr uint32_t TOKEN_X_INPUT_INDEX = 0;
constexpr uint32_t WEIGHT_DQ_INPUT_INDEX = 1;
constexpr uint32_t WEIGHT_UQ_QR_INPUT_INDEX = 2;
constexpr uint32_t WEIGHT_DKV_KR_INPUT_INDEX = 3;
constexpr uint32_t RMSNORM_GAMMA_CQ_INPUT_INDEX = 4;
constexpr uint32_t RMS_NORM_GAMMA_CKV_INPUT_INDEX = 5;
constexpr uint32_t ROPE_SIN_INPUT_INDEX = 6;
constexpr uint32_t ROPE_COS_INPUT_INDEX = 7;

// OUTPUT
constexpr uint32_t Q_NOPE_OUTPUT_INDEX = 0;
constexpr uint32_t Q_PE_OUTPUT_INDEX = 1;
constexpr uint32_t CT_KV_OUTPUT_INDEX = 2;
constexpr uint32_t K_PE_OUTPUT_INDEX = 3;
constexpr uint32_t QR_OUTPUT_INDEX = 4;
constexpr uint32_t Q_M_OUTPUT_INDEX = 5;
constexpr uint32_t Q_RSTD_OUTPUT_INDEX = 6;
constexpr uint32_t CKV_M_OUTPUT_INDEX = 7;
constexpr uint32_t CKV_RSTD_OUTPUT_INDEX = 8;

// ATTR
constexpr uint32_t RMS_NORM_EPSILON_CQ_ATTR_INDEX = 0;
constexpr uint32_t RMS_NORM_EPSILON_CKV_ATTR_INDEX = 1;
constexpr uint32_t MLA_SCALE_Q_LORA_ATTR_INDEX = 2;
constexpr uint32_t MLA_SCALE_KV_LORA_ATTR_INDEX = 3;
constexpr uint32_t QK_NOPE_HEAD_DIM_ATTR_INDEX = 4;
constexpr uint32_t TYPE_DEMOTION_INDEX = 5;
constexpr uint32_t ROPE_FUSION_INDEX = 6;

constexpr uint32_t DSA_SPARSE_PROLOG_DIM_INDEX_0 = 0;
constexpr uint32_t DSA_SPARSE_PROLOG_DIM_INDEX_1 = 1;
constexpr uint32_t DSA_SPARSE_PROLOG_DIM_INDEX_2 = 2;
constexpr uint32_t DSA_SPARSE_PROLOG_DIM_INDEX_3 = 3;

constexpr uint32_t DSA_SPARSE_PROLOG_DIM_NUM_0 = 0;
constexpr uint32_t DSA_SPARSE_PROLOG_DIM_NUM_1 = 1;
constexpr uint32_t DSA_SPARSE_PROLOG_DIM_NUM_2 = 2;
constexpr uint32_t DSA_SPARSE_PROLOG_DIM_NUM_3 = 3;
constexpr uint32_t DSA_SPARSE_PROLOG_DIM_NUM_4 = 4;

constexpr uint32_t DSA_SPARSE_PROLOG_SHAPE_NUM_1 = 1;

BEGIN_TILING_DATA_DEF(DsaSparsePrologBaseParams)
TILING_DATA_FIELD_DEF(uint32_t, stepBatchSize);  // batch size per step 32
TILING_DATA_FIELD_DEF(uint32_t, ubBufferSize);   // ub share buffer size
TILING_DATA_FIELD_DEF(uint32_t, tokenSize);      // token size = product of leading dims of tokenX
TILING_DATA_FIELD_DEF(uint32_t, headSizeX);      // head size of Input Hidden 8192
TILING_DATA_FIELD_DEF(uint32_t, headSizeCq);     // head size of Latent Query 1536
TILING_DATA_FIELD_DEF(uint32_t, headSizeCkv);    // head size of Latent KeyValue 512
TILING_DATA_FIELD_DEF(uint32_t, headSizeQc);     // head size of Query = dimHeadSizeQc * numHeadSize
TILING_DATA_FIELD_DEF(uint32_t, headSizeQr);     // head size of Query Rope = dimHeadRope * numHeadSize
TILING_DATA_FIELD_DEF(uint32_t, headSizeKr);     // head size of Key Rope 64
TILING_DATA_FIELD_DEF(uint32_t, numHeadSize);    // number of head 64
TILING_DATA_FIELD_DEF(uint32_t, dimHeadSizeQc);  // dim size per query head 128
TILING_DATA_FIELD_DEF(uint32_t, dimHeadRope);    // dim size per rope head 64
TILING_DATA_FIELD_DEF(uint32_t, mm1BlockNum);    // Cq
TILING_DATA_FIELD_DEF(uint32_t, mm2BlockNum);    // CkvKr
TILING_DATA_FIELD_DEF(uint32_t, mm3BlockNum);    // QcQr
TILING_DATA_FIELD_DEF(uint32_t, vectorBlockNum); // 48
TILING_DATA_FIELD_DEF(uint32_t, mm1SingleCoreN); // single headSizeCq
TILING_DATA_FIELD_DEF(uint32_t, mm2SingleCoreN); // single headSizeCkv+headSizeKr
TILING_DATA_FIELD_DEF(uint32_t, mm3SingleCoreN); // single headSizeQc+headSizeQr
TILING_DATA_FIELD_DEF(uint32_t,
                      weightLayout); // 0=NZ[K,N], 1=ND[K,N], 2=NZ[N,K], 3=ND[N,K]; bit0=ND, bit1=[N,K] 算子内吸收转置
TILING_DATA_FIELD_DEF(float, reciprocalCq); // 1 / headSizeCq
TILING_DATA_FIELD_DEF(float, epsilonCq);
TILING_DATA_FIELD_DEF(float, reciprocalCkv); // 1 / headSizeCkv
TILING_DATA_FIELD_DEF(float, epsilonCkv);
TILING_DATA_FIELD_DEF(int8_t, isTypeDemotion);
TILING_DATA_FIELD_DEF(int8_t, applyRopeFusion);
TILING_DATA_FIELD_DEF(float, mlaScaleQLora);
TILING_DATA_FIELD_DEF(float, mlaScaleKvLora);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(DsaSparsePrologBaseParamsOp, DsaSparsePrologBaseParams)

BEGIN_TILING_DATA_DEF(DsaSparsePrologTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm1TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm2TilingData);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, bmm3TilingData);
TILING_DATA_FIELD_DEF_STRUCT(DsaSparsePrologBaseParams, baseParams);
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(DsaSparseProlog, DsaSparsePrologTilingData)

struct DsaSparsePrologCompileInfo {
    int64_t core_num;
};

struct RequiredParaInfo {
    const gert::CompileTimeTensorDesc* desc;
    const gert::StorageShape* shape;
};

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t NUM_BYTES_BF16 = 2;
constexpr uint32_t NUM_BYTES_FP32 = 4;
constexpr uint64_t DSA_SPARSE_PROLOG_TILINGKEY_BASE_OFFSET = uint64_t(10000000000000000UL); // 10^16
constexpr uint64_t DSA_SPARSE_PROLOG_TYPE_OFFSET = uint64_t(10UL);

struct DsaSparsePrologContext {
    const char* opName;
    fe::PlatFormInfos* platformInfo;
    RequiredParaInfo tokenX;
    RequiredParaInfo weightDq;
    RequiredParaInfo weightUqQr;
    RequiredParaInfo weightDkvKr;
    RequiredParaInfo rmsnormGammaCq;
    RequiredParaInfo rmsnormGammaCkv;
    RequiredParaInfo ropeSin;
    RequiredParaInfo ropeCos;
    RequiredParaInfo qNopeOut;
    RequiredParaInfo qPeOut;
    RequiredParaInfo ctKvOut;
    RequiredParaInfo kPeOut;
    RequiredParaInfo qrOut;
    RequiredParaInfo qMOut;
    RequiredParaInfo qRstdOut;
    RequiredParaInfo ckvMOut;
    RequiredParaInfo ckvRstdOut;

    const float* rmsNormEspilonCq;
    const float* rmsNormEspilonCkv;
    const float* mlaScaleQLora;
    const float* mlaScaleKvLora;
    const int64_t* qkNopeHeadDim;
    const bool* isTypeDemotion;
    const bool* applyRopeFusion;

    size_t* workSpaces;
    uint64_t tilingKey;
    uint32_t blockDim;
};

class DsaSparsePrologTiling {
public:
    DsaSparsePrologTiling() = default;
    ~DsaSparsePrologTiling() = default;

    ge::graphStatus DoTiling(gert::TilingContext& context);

    ge::graphStatus RunBigKernelTiling(DsaSparsePrologContext& context, DsaSparsePrologTilingData& tilingData,
                                       bool isWorkspace = false);
    ge::graphStatus DsaSparsePrologSetTilingData(gert::TilingContext& context, DsaSparsePrologTilingData& tilingData);
    static ge::graphStatus ConvertContext(gert::TilingContext& context, DsaSparsePrologContext& mlaPrologContext);

private:
    ge::graphStatus GetNpuInfo();
    ge::graphStatus CheckAllInputInfo();
    ge::graphStatus CheckRequiresParams();
    ge::graphStatus CheckDataType();
    ge::graphStatus CheckFormat();
    ge::graphStatus CheckDimNum();
    ge::graphStatus PreventEmptyTensor();
    ge::graphStatus CheckShape();
    ge::graphStatus ProcessBaseInputs();
    ge::graphStatus FillTiling();
    ge::graphStatus FillMatmul1Tiling();
    ge::graphStatus FillMatmul2Tiling();
    ge::graphStatus FillMatmul3Tiling();
    uint32_t CalcSingleCoreN(uint32_t n, uint32_t coreNum, uint32_t alignNum = 16);
    bool GetMatmulType(ge::DataType getype, matmul_tiling::DataType* mmType);
    ge::graphStatus CalcWorkSpace();
    ge::graphStatus CalcUbBufferSize();
    ge::graphStatus GenTilingKey();

    uint32_t tokenSize_ = 0;
    uint32_t stepBatchSize_ = 0;
    uint32_t ubBufferSize_ = 0;
    uint32_t headSizeX_ = 0;
    uint32_t headSizeCq_ = 0;
    uint32_t headSizeCkv_ = 0;
    uint32_t headSizeQc_ = 0;
    uint32_t headSizeQr_ = 0;
    uint32_t headSizeKr_ = 0;
    uint32_t numHeadSize_ = 0;
    uint32_t dimHeadSizeQc_ = 0;
    uint32_t dimHeadRope_ = 0;
    uint32_t mm1BlockNum_ = 0;
    uint32_t mm2BlockNum_ = 0;
    uint32_t mm3BlockNum_ = 0;
    uint32_t vectorBlockNum_ = 0;

    uint32_t singlecoreHeadSizeCq_ = 0;
    uint32_t singlecoreHeadSizeQcQr_ = 0;
    uint32_t singlecoreHeadSizeCkvKr_ = 0;

    float reciprocalCq_ = 0.00001f;
    float epsilonCq_ = 1.0;
    float reciprocalCkv_ = 0.00001f;
    float epsilonCkv_ = 1.0;
    float mlaScaleQLora_ = 1.0;
    float mlaScaleKvLora_ = 1.0;
    bool isTypeDemotion_ = false;
    bool applyRopeFusion_ = false;

    ge::DataType mmDateType_ = ge::DT_BF16;
    uint32_t weightDqAxisSize_ = 0;
    uint32_t weightUqQrAxisSize_ = 0;
    uint32_t weightDkvKrAxisSize_ = 0;
    uint32_t fixAxisSize_ = 16;
    uint32_t weightIsNd_ = 0; // 3 权重同为 ND=1 / NZ=0 (CheckFormat 填)
    uint32_t weightIsNk_ = 0; // 3 权重逻辑 shape 同为 [N,K]=1 / [K,N]=0 (ProcessBaseInputs 填)

    size_t ubSize_ = 0;
    size_t l1Size_ = 0;
    size_t l0cSize_ = 0;
    size_t l0bSize_ = 0;
    uint32_t coreNum_ = 0;
    uint32_t aicNum_ = 0;
    uint32_t aivNum_ = 0;
    size_t libapiSize_ = 0;
    size_t workspaceSize_ = 0;

    DsaSparsePrologContext* context_ = nullptr;
    TCubeTiling* bmm1TilingData_ = nullptr;
    TCubeTiling* bmm2TilingData_ = nullptr;
    TCubeTiling* bmm3TilingData_ = nullptr;
    DsaSparsePrologBaseParams* baseParams_ = nullptr;
    bool isWorkspace_ = false;
};

ge::graphStatus TilingPrepareForDsaSparseProlog(gert::TilingParseContext* context);
MLA_EXTERN_C ge::graphStatus TilingDsaSparseProlog(gert::TilingContext* context);
} // namespace optiling

#endif
