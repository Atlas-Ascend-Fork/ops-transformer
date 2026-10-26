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
 * \file swin_attention_score_quant_tiling.cc
 * \brief SwinAttentionScoreQuant tiling impl
 */

#include "swin_attention_score_quant_tiling.h"

namespace optiling {
const uint32_t INT8_NOMASK_MODE = 0;
const uint32_t INT8_MASK_MODE = 1;
const uint32_t Q_IDX = 0;
const uint32_t K_IDX = 1;
const uint32_t V_IDX = 2;
const uint32_t SCALE_QUANT_IDX = 3;
const uint32_t SCALE_DEQUANT1_IDX = 4;
const uint32_t SCALE_DEQUANT2_IDX = 5;
const uint32_t BIAS_QUANT_IDX = 6;
const uint32_t BIAS_DEQUANT1_IDX = 7;
const uint32_t MASK1_IDX = 9;
const uint32_t MASK2_IDX = 10;
const uint32_t QKV_DIM_SIZE = 4;
const uint32_t SMALL_S_DIM = 128;
const uint32_t CRITICAL_S_DIM = 970;
const uint32_t WORKSPACE_SIZE = 16777216;
const uint32_t BLOCK_NUM_PER_FRACTAL = 16;
const uint32_t NUM_PER_FRACTAL = 512;
const uint32_t BLOCK_SIZE_32 = 32;
const uint32_t BLOCK_SIZE_16 = 16;
const int64_t MIN_S_DIM = 1;
const int64_t MAX_S_DIM = 1024;
const int64_t MIN_H_DIM = 32;
const int64_t MAX_H_DIM = 64;

inline uint32_t DivUp(uint32_t num, uint32_t align)
{
    if (align > 0) {
        return (num + align - 1) / align;
    }
    return 0;
}

inline uint32_t RoundUp(uint32_t num, uint32_t align)
{
    return DivUp(num, align) * align;
}

struct SwinAttentionScoreQuantCompileInfo {};

// 合法输入下Tiling失败时，按EZ1008正式首报（Arglist: op_name, reason），并输出PLOG上下文
inline void ReportTilingFailure(gert::TilingContext* context, const char* reason)
{
    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "%s\n", reason);
    const std::vector<const char*> msgKey = {"op_name", "reason"};
    const std::vector<const char*> msgvalue = {context->GetNodeName(), reason};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgvalue);
}

inline bool CheckQTensor(gert::TilingContext* context, gert::Shape& qOriShape)
{
    OP_CHECK_IF(
        qOriShape.GetDimNum() != QKV_DIM_SIZE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "query", std::to_string(qOriShape.GetDimNum()),
                                                 "the dim num of query must be 4"),
        return false);
    int64_t dimB = qOriShape.GetDim(0);
    int64_t dimN = qOriShape.GetDim(1);
    int64_t dimS = qOriShape.GetDim(2); // query的第2维是S
    int64_t dimH = qOriShape.GetDim(3); // query的第3维是H
    int64_t dimBN = dimB * dimN;
    OP_CHECK_IF(dimS < MIN_S_DIM || dimS > MAX_S_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "query", std::to_string(dimS),
                                                         "the S dim of query should be in range [1, 1024]"),
                return false);
    OP_CHECK_IF(dimH != MIN_H_DIM && dimH != MAX_H_DIM,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "query", std::to_string(dimH),
                                                         "the H dim of query should be 32 or 64"),
                return false);
    OP_CHECK_IF(dimBN < 0 || dimBN > (int64_t)UINT32_MAX / dimS,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(context->GetNodeName(), "query", std::to_string(dimBN),
                                                          "B * N * S of query exceeds the uint32 upper limit"),
                return false);
    return true;
}

inline bool CheckQKVDimSize(gert::TilingContext* context, const char* kvName, gert::Shape& qOriShape,
                            gert::Shape& kvOriShape)
{
    OP_CHECK_IF(
        kvOriShape.GetDimNum() != QKV_DIM_SIZE,
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), kvName, std::to_string(kvOriShape.GetDimNum()),
                                                 "the dim num of key/value must be 4, same as query"),
        return false);
    for (uint32_t i = 0; i < qOriShape.GetDimNum(); i++) {
        OP_CHECK_IF(qOriShape.GetDim(i) != kvOriShape.GetDim(i),
                    OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                        context->GetNodeName(), (std::string("query, ") + kvName).c_str(),
                        (std::to_string(qOriShape.GetDim(i)) + " and " + std::to_string(kvOriShape.GetDim(i))).c_str(),
                        "the dims of key/value must be equal to the dims of query"),
                    return false);
    }
    return true;
}

inline bool CheckKVTensor(gert::TilingContext* context, gert::Shape& qOriShape)
{
    for (uint32_t i = K_IDX; i <= V_IDX; i++) {
        const char* kvName = (i == K_IDX) ? "key" : "value";
        auto kvShape = context->GetInputShape(i);
        if (kvShape != nullptr) {
            auto kvOriShape = kvShape->GetOriginShape();
            OP_CHECK_IF(!CheckQKVDimSize(context, kvName, qOriShape, kvOriShape),
                        OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "input tensor %s dim check failed\n", kvName),
                        return false);
        } else {
            OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), kvName);
            return false;
        }
    }
    return true;
}

inline bool CheckQuantTensor(gert::TilingContext* context)
{
    const char* quantNames[] = {"scale_quant", "scale_dequant1", "scale_dequant2", "bias_quant", "bias_dequant1"};
    for (uint32_t i = 0; i < sizeof(quantNames) / sizeof(quantNames[0]); i++) {
        auto quantShape = context->GetOptionalInputShape(SCALE_QUANT_IDX + i);
        OP_CHECK_IF(quantShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), quantNames[i]),
                    return false);
    }
    return true;
}

inline bool CheckMaskDimSize(gert::TilingContext* context, gert::Shape& qOriShape, gert::Shape& maskOriShape)
{
    OP_CHECK_IF(maskOriShape.GetDimNum() != QKV_DIM_SIZE,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "padding_mask1",
                                                         std::to_string(maskOriShape.GetDimNum()),
                                                         "the dim num of padding_mask1 must be 4, same as query"),
                return false);
    OP_CHECK_IF(maskOriShape.GetDim(0) != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "padding_mask1",
                                                         std::to_string(maskOriShape.GetDim(0)),
                                                         "the dim 0 of padding_mask1 must be 1"),
                return false);
    for (uint32_t i = 1; i < QKV_DIM_SIZE - 1; i++) {
        OP_CHECK_IF(
            qOriShape.GetDim(i) != maskOriShape.GetDim(i),
            OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                context->GetNodeName(), "query, padding_mask1",
                (std::to_string(qOriShape.GetDim(i)) + " and " + std::to_string(maskOriShape.GetDim(i))).c_str(),
                "the dim of padding_mask1 must be equal to the dim of query"),
            return false);
    }
    OP_CHECK_IF(
        qOriShape.GetDim(QKV_DIM_SIZE - 2) != maskOriShape.GetDim(QKV_DIM_SIZE - 1),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(context->GetNodeName(), "query, padding_mask1",
                                                  (std::to_string(qOriShape.GetDim(QKV_DIM_SIZE - 2)) + " and " +
                                                   std::to_string(maskOriShape.GetDim(QKV_DIM_SIZE - 1)))
                                                      .c_str(),
                                                  "the last dim of padding_mask1 must be equal to the S dim of query"),
        return false); // Mask的最后2维大小相等
    return true;
}

inline bool CheckMaskTensor(gert::TilingContext* context, gert::Shape& qOriShape)
{
    auto maskShape1 = context->GetOptionalInputShape(MASK1_IDX);
    if (maskShape1 != nullptr) {
        auto maskOriShape = maskShape1->GetOriginShape();
        OP_CHECK_IF(!CheckMaskDimSize(context, qOriShape, maskOriShape),
                    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "input padding_mask1 dim check failed\n"),
                    return false);
    }
    return true;
}

inline bool CheckInTensor(gert::TilingContext* context)
{
    auto qShape = context->GetInputShape(Q_IDX);
    OP_CHECK_IF(qShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "query"), return false);
    auto qOriShape = qShape->GetOriginShape();
    OP_CHECK_IF(!CheckQTensor(context, qOriShape),
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "input query check failed\n"), return false);
    OP_CHECK_IF(!CheckKVTensor(context, qOriShape),
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "input key/value check failed\n"), return false);
    OP_CHECK_IF(!CheckQuantTensor(context),
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "input quant check failed\n"), return false);
    OP_CHECK_IF(!CheckMaskTensor(context, qOriShape),
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "input padding_mask1 check failed\n"), return false);

    return true;
}

inline void GetSingleCoreM(uint32_t s, uint32_t& singleCoreM)
{
    if (s > SMALL_S_DIM) {
        singleCoreM = BLOCK_NUM_PER_FRACTAL;
    } else {
        singleCoreM = RoundUp(s, BLOCK_NUM_PER_FRACTAL);
    }
}

inline ge::graphStatus SwinAttentionCfgTiling(SwinAttentionScoreQuantTilingData& tiling, uint32_t& coreNum,
                                              uint32_t& tmpSize, gert::TilingContext* context, bool hasMask)
{
    uint32_t cubeSharedUbSize;
    uint32_t vecSharedUbSize;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    coreNum = ascendcPlatform.GetCoreNum();
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0CSize;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto qOriShape = inputShape->GetOriginShape();
    uint32_t dimB = qOriShape.GetDim(0);
    uint32_t dimN = qOriShape.GetDim(1);
    uint32_t dimS = qOriShape.GetDim(2); // query的第2维是S
    uint32_t dimH = qOriShape.GetDim(3); // query的第3维是H
    uint32_t round16S = RoundUp(dimS, BLOCK_SIZE_16);
    uint32_t round32S = RoundUp(dimS, BLOCK_SIZE_32);
    tmpSize = 0;

    uint32_t singleCoreM;
    GetSingleCoreM(dimS, singleCoreM);

    uint32_t numPerS = DivUp(dimS, singleCoreM);
    uint32_t coreLoops = dimB;
    if (hasMask) {
        coreLoops = dimB;
    } else {
        coreLoops = dimB * dimN * numPerS;
    }
    tiling.set_coreLoops(coreLoops);
    coreNum = (coreNum < coreLoops) ? coreNum : coreLoops;

    // qk tiling
    matmul_tiling::MatmulApiTiling qkBmm(ascendcPlatform);
    matmul_tiling::MatmulApiTiling pvBmm(ascendcPlatform);
    qkBmm.SetAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8,
                   false);
    qkBmm.SetBType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8,
                   true);
    qkBmm.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::NZ,
                   matmul_tiling::DataType::DT_FLOAT16);
    qkBmm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    qkBmm.SetShape(singleCoreM, dimS, dimH);
    qkBmm.SetOrgShape(singleCoreM, round32S, dimH);
    qkBmm.SetBias(true);
    qkBmm.SetBufferSpace(l1Size / 2, l0CSize); // 使用1/2的l1空间
    OP_CHECK_IF(qkBmm.GetTiling(tiling.qkBmmTilingData) == -1,
                ReportTilingFailure(context, "failed to get qk matmul tiling under current hardware and shape"),
                return ge::GRAPH_FAILED);
    matmul_tiling::SysTilingTempBufSize qkBufSize;
    OP_CHECK_IF(MatmulGetTmpBufSize(tiling.qkBmmTilingData, qkBufSize) == -1,
                ReportTilingFailure(context, "failed to get qk matmul tmp buffer size"), return ge::GRAPH_FAILED);
    cubeSharedUbSize = qkBufSize.ubSize;

    // pv tiling
    pvBmm.SetAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8,
                   false);
    pvBmm.SetBType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::NZ, matmul_tiling::DataType::DT_INT8,
                   true);
    pvBmm.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);
    pvBmm.SetBiasType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_INT32);
    pvBmm.SetShape(singleCoreM, dimH, dimS);
    pvBmm.SetOrgShape(singleCoreM, dimH, round32S, round32S);
    pvBmm.SetBias(true);
    pvBmm.SetBufferSpace(l1Size / 2, l0CSize); // 使用1/2的l1空间
    OP_CHECK_IF(pvBmm.GetTiling(tiling.pvBmmTilingData) == -1,
                ReportTilingFailure(context, "failed to get pv matmul tiling under current hardware and shape"),
                return ge::GRAPH_FAILED);
    uint32_t pvUbSize = tiling.pvBmmTilingData.get_transLength() + tiling.pvBmmTilingData.get_transLength();
    uint32_t pvUbSizeTmp = round32S * dimH + round32S * dimH;
    pvUbSize = std::max(pvUbSize, pvUbSizeTmp);
    cubeSharedUbSize = std::max(cubeSharedUbSize, pvUbSize);

    auto qkL0 = tiling.qkBmmTilingData.get_shareL0CSize();
    auto qkL1 = tiling.qkBmmTilingData.get_shareL1Size();
    auto pvL0 = tiling.pvBmmTilingData.get_shareL0CSize();
    auto pvL1 = tiling.pvBmmTilingData.get_shareL1Size();

    auto bmmMaxL0 = std::max(qkL0, pvL0);
    auto bmmMaxL1 = std::max(qkL1, pvL1);

    tiling.qkBmmTilingData.set_shareL0CSize(bmmMaxL0);
    tiling.qkBmmTilingData.set_shareL1Size(bmmMaxL1);
    tiling.pvBmmTilingData.set_shareL0CSize(bmmMaxL0);
    tiling.pvBmmTilingData.set_shareL1Size(bmmMaxL1);

    // softmax
    uint32_t softmaxTmpSize;
    const ge::Shape softmaxShape = ge::Shape({singleCoreM, round16S});
    if (hasMask && dimS > CRITICAL_S_DIM) {
        softmaxTmpSize = 0;
    } else {
        softmaxTmpSize = AscendC::GetSoftMaxMaxTmpSize(softmaxShape, sizeof(uint16_t), true);
        AscendC::SoftMaxTilingFunc(softmaxShape, sizeof(uint16_t), softmaxTmpSize, tiling.softmaxTilingData);
    }
    vecSharedUbSize = softmaxTmpSize;

    uint32_t qSize = RoundUp(singleCoreM * dimH, NUM_PER_FRACTAL);
    uint32_t kSize = RoundUp(dimH * round32S, NUM_PER_FRACTAL);
    uint32_t pSize = RoundUp(singleCoreM * round32S, NUM_PER_FRACTAL);
    uint32_t vSize = RoundUp(round32S * dimH, NUM_PER_FRACTAL);

    tiling.set_dimB(dimB);
    tiling.set_dimN(dimN);
    tiling.set_dimS(dimS);
    tiling.set_dimH(dimH);
    tiling.set_qSize(qSize);
    tiling.set_kSize(kSize);
    tiling.set_pSize(pSize);
    tiling.set_vSize(vSize);
    tiling.set_cubeSharedUbSize(cubeSharedUbSize);
    tiling.set_vecSharedUbSize(vecSharedUbSize);

    return ge::GRAPH_SUCCESS;
}

void TilingLog(SwinAttentionScoreQuantTilingData& tiling)
{
    OP_LOGI("[SwinAttentionScoreQuant]", "[coreLoops]: %u", tiling.get_coreLoops());
    OP_LOGI("[SwinAttentionScoreQuant]", "[dimB]: %u", tiling.get_dimB());
    OP_LOGI("[SwinAttentionScoreQuant]", "[dimN]: %u", tiling.get_dimN());
    OP_LOGI("[SwinAttentionScoreQuant]", "[dimB]: %u", tiling.get_dimS());
    OP_LOGI("[SwinAttentionScoreQuant]", "[dimN]: %u", tiling.get_dimH());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qSize]: %u", tiling.get_qSize());
    OP_LOGI("[SwinAttentionScoreQuant]", "[kSize]: %u", tiling.get_kSize());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pSize]: %u", tiling.get_pSize());
    OP_LOGI("[SwinAttentionScoreQuant]", "[vSize]: %u", tiling.get_vSize());
    OP_LOGI("[SwinAttentionScoreQuant]", "[cubeSharedUbSize]: %u", tiling.get_cubeSharedUbSize());
    OP_LOGI("[SwinAttentionScoreQuant]", "[vecSharedUbSize]: %u", tiling.get_vecSharedUbSize());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qkBmm.M]: %u", tiling.qkBmmTilingData.get_M());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qkBmm.N]: %u", tiling.qkBmmTilingData.get_N());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qkBmm.Ka]: %u", tiling.qkBmmTilingData.get_Ka());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qkBmm.Kb]: %u", tiling.qkBmmTilingData.get_Kb());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qkBmm.singleCoreM]: %u", tiling.qkBmmTilingData.get_singleCoreM());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qkBmm.singleCoreN]: %u", tiling.qkBmmTilingData.get_singleCoreN());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qkBmm.singleCoreK]: %u", tiling.qkBmmTilingData.get_singleCoreK());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qkBmm.baseM]: %u", tiling.qkBmmTilingData.get_baseM());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qkBmm.baseN]: %u", tiling.qkBmmTilingData.get_baseN());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qkBmm.baseK]: %u", tiling.qkBmmTilingData.get_baseK());
    OP_LOGI("[SwinAttentionScoreQuant]", "[qkBmm.isBias]: %u", tiling.qkBmmTilingData.get_isBias());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pvBmm.M]: %u", tiling.pvBmmTilingData.get_M());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pvBmm.N]: %u", tiling.pvBmmTilingData.get_N());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pvBmm.Ka]: %u", tiling.pvBmmTilingData.get_Ka());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pvBmm.Kb]: %u", tiling.pvBmmTilingData.get_Kb());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pvBmm.singleCoreM]: %u", tiling.pvBmmTilingData.get_singleCoreM());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pvBmm.singleCoreN]: %u", tiling.pvBmmTilingData.get_singleCoreN());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pvBmm.singleCoreK]: %u", tiling.pvBmmTilingData.get_singleCoreK());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pvBmm.baseM]: %u", tiling.pvBmmTilingData.get_baseM());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pvBmm.baseN]: %u", tiling.pvBmmTilingData.get_baseN());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pvBmm.baseK]: %u", tiling.pvBmmTilingData.get_baseK());
    OP_LOGI("[SwinAttentionScoreQuant]", "[pvBmm.isBias]: %u", tiling.pvBmmTilingData.get_isBias());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.srcM]: %u", tiling.softmaxTilingData.get_srcM());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.srcK]: %u", tiling.softmaxTilingData.get_srcK());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.srcSize]: %u", tiling.softmaxTilingData.get_srcSize());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.outMaxM]: %u", tiling.softmaxTilingData.get_outMaxM());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.outMaxK]: %u", tiling.softmaxTilingData.get_outMaxK());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.outMaxSize]: %u",
            tiling.softmaxTilingData.get_outMaxSize());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.splitM]: %u", tiling.softmaxTilingData.get_splitM());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.splitK]: %u", tiling.softmaxTilingData.get_splitK());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.splitSize]: %u", tiling.softmaxTilingData.get_splitSize());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.reduceM]: %u", tiling.softmaxTilingData.get_reduceM());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.reduceK]: %u", tiling.softmaxTilingData.get_reduceK());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.reduceSize]: %u",
            tiling.softmaxTilingData.get_reduceSize());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.rangeM]: %u", tiling.softmaxTilingData.get_rangeM());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.tailM]: %u", tiling.softmaxTilingData.get_tailM());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.tailSplitSize]: %u",
            tiling.softmaxTilingData.get_tailSplitSize());
    OP_LOGI("[SwinAttentionScoreQuant]", "[softmaxTilingData.tailReduceSize]: %u",
            tiling.softmaxTilingData.get_tailReduceSize());
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    ge::graphStatus res = ge::GRAPH_SUCCESS;
    SwinAttentionScoreQuantTilingData tiling;
    uint32_t coreNum;
    uint32_t tmpSize;

    OP_CHECK_IF(!CheckInTensor(context), OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "input tensor check failed\n"),
                return ge::GRAPH_FAILED);

    auto maskShape = context->GetOptionalInputShape(MASK1_IDX);
    if (maskShape == nullptr) {
        OP_LOGI(context->GetNodeName(), "without mask\n");
        context->SetTilingKey(INT8_NOMASK_MODE);
        res = SwinAttentionCfgTiling(tiling, coreNum, tmpSize, context, false);
    } else {
        OP_LOGI(context->GetNodeName(), "with mask\n");
        context->SetTilingKey(INT8_MASK_MODE);
        res = SwinAttentionCfgTiling(tiling, coreNum, tmpSize, context, true);
    }

    OP_CHECK_IF(res == ge::GRAPH_FAILED,
                OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "SwinAttentionCfgTiling failed\n"),
                return ge::GRAPH_FAILED);

    TilingLog(tiling);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = tmpSize + WORKSPACE_SIZE;
    context->SetBlockDim(coreNum);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return res;
}

ge::graphStatus TilingPrepareForSwinAttentionScoreQuant(gert::TilingParseContext* context)
{
    auto compileInfo = context->GetCompiledInfo<SwinAttentionScoreQuantCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compileInfo);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SwinAttentionScoreQuant)
    .Tiling(TilingFunc)
    .TilingParse<SwinAttentionScoreQuantCompileInfo>(TilingPrepareForSwinAttentionScoreQuant); // 向框架注册入口函数
} // namespace optiling
