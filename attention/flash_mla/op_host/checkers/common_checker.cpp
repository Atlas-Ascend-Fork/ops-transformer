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
 * \file common_checker.cpp
 * \brief Common checker for layout, shape, dtype, and scalar attr parameters
 */

#include <map>
#include <numeric>
#include <vector>
#include <algorithm>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_mla_tiling_info.h"
#include "common_checker.h"
#include "../../op_kernel/flash_mla_shared.h"

namespace optiling {
namespace flash_mla {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35MLA;
using namespace Ops::Base;

ge::graphStatus CommonChecker::CheckSingleParaLayout(const FlashMlaTilingInfo& faInfo)
{
    const std::vector<FlashMlaLayout> supportedQLayouts = {FlashMlaLayout::BSND, FlashMlaLayout::BNSD,
                                                           FlashMlaLayout::TND};
    const std::vector<FlashMlaLayout> supportedKvLayouts = {FlashMlaLayout::PA_NZ, FlashMlaLayout::PA_BBND,
                                                            FlashMlaLayout::PA_BNBD};
    const std::vector<FlashMlaLayout> supportedOutLayouts = {FlashMlaLayout::BSND, FlashMlaLayout::BNSD,
                                                             FlashMlaLayout::TND, FlashMlaLayout::NTD};

    if (std::find(supportedQLayouts.begin(), supportedQLayouts.end(), faInfo.qLayout) == supportedQLayouts.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "layout_q", LayoutToSerialString(faInfo.qLayout).c_str(),
                                              "The value of layout_q must be BSND, BNSD or TND");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(
        std::find(supportedKvLayouts.begin(), supportedKvLayouts.end(), faInfo.kvLayout) == supportedKvLayouts.end(),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "layout_kv", LayoutToSerialString(faInfo.kvLayout).c_str(),
                                              "The value of layout_kv must be PA_NZ, PA_BBND or PA_BNBD"),
        return ge::GRAPH_FAILED);

    if (std::find(supportedOutLayouts.begin(), supportedOutLayouts.end(), faInfo.outLayout) ==
        supportedOutLayouts.end()) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "layout_out",
                                              LayoutToSerialString(faInfo.outLayout).c_str(),
                                              "The value of layout_out must be BSND, BNSD, TND or NTD");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckSinglePara(const FlashMlaTilingInfo& faInfo)
{
    if (CheckSingleParaLayout(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckParaExistence(const FlashMlaTilingInfo& faInfo)
{
    OP_CHECK_IF(faInfo.opParamInfo.query.desc == nullptr || faInfo.opParamInfo.query.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "query"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.kCache.desc == nullptr || faInfo.opParamInfo.kCache.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "k_cache"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.opParamInfo.attnOut.desc == nullptr || faInfo.opParamInfo.attnOut.shape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "attn_out"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckNonQuantDataType(const FlashMlaTilingInfo& faInfo)
{
    if (ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.kCache.desc, K_CACHE_NAME) ||
        ge::GRAPH_SUCCESS != CheckDtypeSupport(faInfo.opParamInfo.attnOut.desc, ATTN_OUT_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.query.desc, QUERY_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.kCache.desc, K_CACHE_NAME) ||
        ge::GRAPH_SUCCESS != CheckFormatSupport(faInfo.opParamInfo.attnOut.desc, ATTN_OUT_NAME)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckDtypeConsistency(const FlashMlaTilingInfo& faInfo)
{
    if (faInfo.inputKvType != faInfo.inputQType) {
        std::string reason =
            "The dtype of k_cache must be the same as dtype(" + ToString(faInfo.inputQType) + ") of query";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "k_cache", ToString(faInfo.inputKvType).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }
    if (faInfo.outputType != faInfo.inputQType) {
        std::string reason =
            "The dtype of attn_out must be the same as dtype(" + ToString(faInfo.inputQType) + ") of query";
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(faInfo.opName, "attn_out", ToString(faInfo.outputType).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckNonQuantHeadNum(const FlashMlaTilingInfo& faInfo)
{
    if ((faInfo.n1Size <= 0) || (faInfo.n2Size <= 0)) {
        std::string shapeStr = ToString(faInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                               ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(faInfo.opName, "query and k_cache", shapeStr.c_str(),
                                               "N of query and k_cache must be greater than 0");
        return ge::GRAPH_FAILED;
    }
    if (faInfo.n1Size < faInfo.n2Size) {
        std::string shapeStr = ToString(faInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                               ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(faInfo.opName, "query and k_cache", shapeStr.c_str(),
                                               "N of query must be greater than or equal to the same axis of k_cache");
        return ge::GRAPH_FAILED;
    }
    if (faInfo.n1Size % faInfo.n2Size != 0) {
        std::string shapeStr = ToString(faInfo.opParamInfo.query.shape->GetStorageShape()) + " and " +
                               ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape());
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(faInfo.opName, "query and k_cache", shapeStr.c_str(),
                                               "N of query must be an integer multiple of the same axis of k_cache");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckAxis(const FlashMlaTilingInfo& faInfo)
{
    if (faInfo.bSize >= B_LIMIT || faInfo.bSize <= 0) {
        std::string reason = "The value of B must be within the range (0, " + std::to_string(B_LIMIT) + ")";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "axis B", std::to_string(faInfo.bSize).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }

    if (faInfo.qLayout == FlashMlaLayout::TND) {
        OP_CHECK_IF(faInfo.qTSize <= 0,
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                        faInfo.opName, "query", ToString(faInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                        "T of query must be greater than 0"),
                    return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF(faInfo.n1Size < 1 || faInfo.n1Size > 128,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    faInfo.opName, "query", ToString(faInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                    "N of query must be in [1, 128]"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.n2Size <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    faInfo.opName, "k_cache", ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape()).c_str(),
                    "N of k_cache must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s1Size <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    faInfo.opName, "query", ToString(faInfo.opParamInfo.query.shape->GetStorageShape()).c_str(),
                    "S of query must be greater than 0"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(faInfo.s2Size <= 0,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    faInfo.opName, "k_cache", ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape()).c_str(),
                    "S of k_cache must be greater than 0"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMlaGeometry(const FlashMlaTilingInfo& faInfo)
{
    OP_CHECK_IF(faInfo.n2Size != FlashMlaCommon::MLA_KV_HEADS,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    faInfo.opName, "k_cache", ToString(faInfo.opParamInfo.kCache.shape->GetStorageShape()).c_str(),
                    "kvHeadNum(N of k_cache) must be 1 (MLA single latent KV head)"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.qkNopeHeadDim != FlashMlaCommon::MLA_HEAD_DIM_QK_NOPE,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "nope head dimension of query and k_cache",
                                                      std::to_string(faInfo.qkNopeHeadDim).c_str(),
                                                      "qkNopeHeadDim must be 512 in MLA"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(
        faInfo.vHeadDim != FlashMlaCommon::MLA_HEAD_DIM_V,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "axis D of value", std::to_string(faInfo.vHeadDim).c_str(),
                                              "vHeadDim must be 512 in MLA (== head_dim_v)"),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(faInfo.qkNopeHeadDim + faInfo.ropeHeadDim != FlashMlaCommon::MLA_HEAD_DIM_QK,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "query head dimension",
                                                      std::to_string(faInfo.qkNopeHeadDim + faInfo.ropeHeadDim).c_str(),
                                                      "QK nope and rope dimensions must sum to 576"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

struct LayoutConstraintConfig {
    std::vector<FlashMlaLayout> supportedKvLayouts;
    std::vector<FlashMlaLayout> supportedOutLayouts;
};

static const std::map<FlashMlaLayout, LayoutConstraintConfig> LAYOUT_CONSTRAINT_TABLE = {
    {FlashMlaLayout::BSND,
     {{FlashMlaLayout::PA_NZ, FlashMlaLayout::PA_BBND, FlashMlaLayout::PA_BNBD}, {FlashMlaLayout::BSND}}},
    {FlashMlaLayout::BNSD,
     {{FlashMlaLayout::PA_NZ, FlashMlaLayout::PA_BBND, FlashMlaLayout::PA_BNBD}, {FlashMlaLayout::BNSD}}},
    {FlashMlaLayout::TND,
     {{FlashMlaLayout::PA_NZ, FlashMlaLayout::PA_BBND, FlashMlaLayout::PA_BNBD},
      {FlashMlaLayout::TND, FlashMlaLayout::NTD}}},
};

ge::graphStatus CommonChecker::CheckMultiParaLayout(const FlashMlaTilingInfo& faInfo)
{
    auto it = LAYOUT_CONSTRAINT_TABLE.find(faInfo.qLayout);
    OP_CHECK_IF(it == LAYOUT_CONSTRAINT_TABLE.end(),
                OP_LOGE(faInfo.opName, "layout_q %s is not supported", LayoutToSerialString(faInfo.qLayout).c_str()),
                return ge::GRAPH_FAILED);

    const auto& config = it->second;
    const std::string qLayoutStr = LayoutToSerialString(faInfo.qLayout);

    OP_CHECK_IF(std::find(config.supportedKvLayouts.begin(), config.supportedKvLayouts.end(), faInfo.kvLayout) ==
                    config.supportedKvLayouts.end(),
                OP_LOGE(faInfo.opName, "When layout_q is %s, layout_kv must match constraint, but got %s",
                        qLayoutStr.c_str(), LayoutToSerialString(faInfo.kvLayout).c_str()),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(std::find(config.supportedOutLayouts.begin(), config.supportedOutLayouts.end(), faInfo.outLayout) ==
                    config.supportedOutLayouts.end(),
                OP_LOGE(faInfo.opName, "When layout_q is %s, layout_out must match constraint, but got %s",
                        qLayoutStr.c_str(), LayoutToSerialString(faInfo.outLayout).c_str()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void CommonChecker::SetFaShapeCompare(const FlashMlaTilingInfo& faInfo)
{
    queryShapeCmp_ = std::make_shared<FlashMlaTilingShapeCompare>(faInfo.opParamInfo.query.shape->GetStorageShape(),
                                                                  faInfo.qLayout, QUERY_NAME, faInfo.opName);
    keyShapeCmp_ = std::make_shared<FlashMlaTilingShapeCompare>(faInfo.opParamInfo.kCache.shape->GetStorageShape(),
                                                                faInfo.kvLayout, K_CACHE_NAME, faInfo.opName);
    attnOutShapeCmp_ = std::make_shared<FlashMlaTilingShapeCompare>(faInfo.opParamInfo.attnOut.shape->GetStorageShape(),
                                                                    faInfo.outLayout, ATTN_OUT_NAME, faInfo.opName);
}

ge::graphStatus CommonChecker::CheckQueryShape(const FlashMlaTilingInfo& faInfo) const
{
    FlashMlaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(faInfo.bSize);
    shapeParams.N = static_cast<int64_t>(faInfo.n1Size);
    shapeParams.S = static_cast<int64_t>(faInfo.s1Size);
    shapeParams.D = static_cast<int64_t>(faInfo.qkNopeHeadDim + MLA_ROPE_D_DIM_64);
    if (faInfo.qLayout == FlashMlaLayout::TND) {
        shapeParams.T = static_cast<int64_t>(faInfo.qTSize);
    }
    return queryShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus CommonChecker::CheckKVShapeForPageAttention(const FlashMlaTilingInfo& faInfo) const
{
    ge::DataType kvDtype = faInfo.inputKvType;
    uint32_t kvBlockElemNum = 32 / FlashMlaBaseChecker::GetTypeSize(kvDtype);

    if (faInfo.blockSize % kvBlockElemNum != 0) {
        std::string reason = "The value of block_size must be a multiple of " + std::to_string(kvBlockElemNum) +
                             " (32 / sizeof(kv_dtype))";
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "block_size", std::to_string(faInfo.blockSize).c_str(),
                                              reason.c_str());
        return ge::GRAPH_FAILED;
    }

    FlashMlaTilingShapeCompareParam shapeParams;
    shapeParams.Bn = static_cast<int64_t>(faInfo.totalBlockNum);
    shapeParams.N = static_cast<int64_t>(faInfo.n2Size);
    shapeParams.Bs = static_cast<int64_t>(faInfo.blockSize);
    shapeParams.D = static_cast<int64_t>(faInfo.qkNopeHeadDim + MLA_ROPE_D_DIM_64);
    shapeParams.D0 = static_cast<int64_t>(kvBlockElemNum);

    return keyShapeCmp_->CompareShape(shapeParams, __func__);
}

ge::graphStatus CommonChecker::CheckKVShape(const FlashMlaTilingInfo& faInfo) const
{
    if (faInfo.kvLayout == FlashMlaLayout::PA_NZ || faInfo.kvLayout == FlashMlaLayout::PA_BBND ||
        faInfo.kvLayout == FlashMlaLayout::PA_BNBD) {
        if (faInfo.pageAttentionFlag) {
            return CheckKVShapeForPageAttention(faInfo);
        }
        std::string reason = "block_table cannot be empty when layout_kv is " + LayoutToSerialString(faInfo.kvLayout);
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(faInfo.opName, "block_table", reason.c_str());
        return ge::GRAPH_FAILED;
    }

    std::string reason = "layout_kv: " + LayoutToSerialString(faInfo.kvLayout) +
                         " is not supported (only paged KV PA_NZ/PA_BBND/PA_BNBD)";
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "layout_kv", LayoutToSerialString(faInfo.kvLayout).c_str(),
                                          reason.c_str());
    return ge::GRAPH_FAILED;
}

ge::graphStatus CommonChecker::CheckAttnOutShape(const FlashMlaTilingInfo& faInfo) const
{
    FlashMlaTilingShapeCompareParam shapeParams;
    shapeParams.B = static_cast<int64_t>(faInfo.bSize);
    shapeParams.N = static_cast<int64_t>(faInfo.n1Size);
    shapeParams.S = static_cast<int64_t>(faInfo.s1Size);
    shapeParams.D = static_cast<int64_t>(faInfo.vHeadDim);
    if (faInfo.outLayout == FlashMlaLayout::TND || faInfo.outLayout == FlashMlaLayout::NTD) {
        shapeParams.T = static_cast<int64_t>(faInfo.qTSize);
    }
    if (attnOutShapeCmp_->CompareShape(shapeParams, __func__) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckShapeConsistency(const FlashMlaTilingInfo& faInfo)
{
    SetFaShapeCompare(faInfo);
    if (ge::GRAPH_SUCCESS != CheckQueryShape(faInfo) || ge::GRAPH_SUCCESS != CheckKVShape(faInfo)) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CommonChecker::CheckMultiPara(const FlashMlaTilingInfo& faInfo)
{
    if (CheckMultiParaLayout(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonQuantDataType(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckNonQuantHeadNum(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckDtypeConsistency(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckShapeConsistency(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAxis(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckMlaGeometry(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckAttnOutShape(faInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_mla
} // namespace optiling
