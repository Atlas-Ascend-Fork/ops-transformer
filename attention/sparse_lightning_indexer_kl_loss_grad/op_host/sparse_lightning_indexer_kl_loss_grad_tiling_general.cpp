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
 * \file sparse_lightning_indexer_kl_loss_grad_tiling_general.cpp
 * \brief
 */

#include "log/log.h"
#include "sparse_lightning_indexer_kl_loss_grad_tiling_general.h"
#include "../op_kernel/arch22/sparse_lightning_indexer_kl_loss_grad_metadata_arch22.h"
#include <cstdint>
#include <cstring>
#include <vector>
#include <tiling/tiling_api.h>

using namespace ge;
using namespace AscendC;

namespace optiling {

static const int64_t PING_PONG_VALUE = 2L;
static constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
static constexpr int64_t PRIVATE_SCATTER_MAX_RATIO = 128;
static constexpr int64_t PRIVATE_SCATTER_RATIO_K_MAX = 131072;
static constexpr int64_t PRIVATE_SCATTER_MAX_B = 64;
static constexpr int64_t PRIVATE_SCATTER_MAX_AVG_S2 = 32 * 1024;

static int64_t CalcWorkspaceSize(int64_t scatterAddOutSize, int64_t scatterBankNum, int64_t multiCoreBase)
{
    return multiCoreBase + scatterAddOutSize * scatterBankNum + static_cast<int64_t>(WORK_SPACE_RESERVE_SIZE);
}

static bool IsPrivateScatter(int64_t scatterElems, int64_t totalS1, int64_t batchSize, int64_t topK)
{
    if (scatterElems <= 0 || totalS1 <= 0 || batchSize <= 0 || topK <= 0) {
        return false;
    }
    if (batchSize > PRIVATE_SCATTER_MAX_B) {
        return false;
    }
    const int64_t avgS2 = scatterElems / batchSize;
    if (avgS2 > PRIVATE_SCATTER_MAX_AVG_S2) {
        return false;
    }
    if (scatterElems > PRIVATE_SCATTER_MAX_RATIO * totalS1) {
        return false;
    }
    const int64_t kReal = topK < avgS2 ? topK : avgS2;
    return scatterElems * kReal <= PRIVATE_SCATTER_RATIO_K_MAX * totalS1;
}

static constexpr uint32_t BUFFER_SIZE_BYTE_2K = 2 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_8K = 8 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_32K = 32 * 1024;
static constexpr uint32_t BUFFER_SIZE_BYTE_33K = 33 * 1024;

static constexpr int64_t MASK_MODE_NO_MASK = 0;
static constexpr int64_t MASK_MODE_CAUSAL = 3;

template <typename T>
static auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingBase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF(attrs == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "attributes",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrs->GetAttrPointer<char>(LAYOUT_Q_ATTR_INDEX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "layout_q",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrs->GetAttrPointer<char>(LAYOUT_K_ATTR_INDEX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "layout_k",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrs->GetAttrPointer<int64_t>(MASK_MODE_ATTR_INDEX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "mask_mode",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrs->GetAttrPointer<int64_t>(CMP_RATIO_ATTR_INDEX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "cmp_ratio",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetWorkspaceSizes(1) == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{Ops::Base::GetOpInfo(context_).c_str(),
                                              "context_->GetWorkspaceSizes(1) is null; Tiling cannot continue"})),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context_->GetInputShape(QUERY_INPUT_INDEX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "query",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(KEY_INPUT_INDEX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "key",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(WEIGHT_INPUT_INDEX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "weight",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "sparse indices",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetInputShape(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "attn softmax l1 norm",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOutputShape(D_QUERY_OUTPUT_INDEX) == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context_).c_str(),
                                                                    "d query is null; Tiling cannot continue"})),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOutputShape(D_KEY_OUTPUT_INDEX) == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context_).c_str(),
                                                                    "d key is null; Tiling cannot continue"})),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOutputShape(D_WEIGHTS_OUTPUT_INDEX) == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context_).c_str(),
                                                                    "d weights is null; Tiling cannot continue"})),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetOutputShape(SOFTMAX_OUT_OUTPUT_INDEX) == nullptr,
                REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                          (std::vector<const char*>{Ops::Base::GetOpInfo(context_).c_str(),
                                                                    "softmax out is null; Tiling cannot continue"})),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((context_->GetOptionalInputShape(SEQUSED_QUERY_INPUT_INDEX) != nullptr ||
                 context_->GetOptionalInputShape(SEQUSED_KEY_INPUT_INDEX) != nullptr) &&
                    context_->GetOptionalInputShape(METADATA_INPUT_INDEX) == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    Ops::Base::GetOpInfo(context_).c_str(), "metadata",
                    "metadata is required when seqused_q or seqused_k is provided on arch22."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(context_->GetRawTilingData() == nullptr,
                REPORT_PREDEFINED_ERR_MSG(
                    "EZ1008", (std::vector<const char*>{"opName", "reason"}),
                    (std::vector<const char*>{Ops::Base::GetOpInfo(context_).c_str(),
                                              "context_->GetRawTilingData() is null; Tiling cannot continue"})),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        context_->GetRawTilingData()->GetData() == nullptr,
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{Ops::Base::GetOpInfo(context_).c_str(),
                                      "context_->GetRawTilingData()->GetData() is null; Tiling cannot continue"})),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

bool SparseLightningIndexerKLLossGradTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    inputLayout = attrs->GetAttrPointer<char>(LAYOUT_Q_ATTR_INDEX);
    keyLayout = attrs->GetAttrPointer<char>(LAYOUT_K_ATTR_INDEX);
    sparseMode = *attrs->GetAttrPointer<int64_t>(MASK_MODE_ATTR_INDEX);
    cmpRatio = *attrs->GetAttrPointer<int64_t>(CMP_RATIO_ATTR_INDEX);
    deterministic = (context_->GetDeterministic() == 1);
    scaleValue = 1.0f;
    hasSoftmaxInput = 1;
    hasRope = false;
    dQueryRopeSize = 0;
    dKeyRopeSize = 0;

    OP_CHECK_IF(
        sparseMode != MASK_MODE_NO_MASK && sparseMode != MASK_MODE_CAUSAL,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName, "attributes", (std::string() + std::to_string(sparseMode)).c_str(),
            (std::string() + "mask_mode only supports 0 or 3, but got [" + std::to_string(sparseMode) + "].").c_str()),
        return false);
    OP_CHECK_IF(
        cmpRatio < 1 || cmpRatio > 128,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName, "attributes", (std::string() + std::to_string(cmpRatio)).c_str(),
            (std::string() + "cmp_ratio must be in [1, 128], but got [" + std::to_string(cmpRatio) + "].").c_str()),
        return false);
    auto cmpResidualK = context_->GetOptionalInputShape(CMP_RESIDUAL_KEY_INPUT_INDEX);
    OP_CHECK_IF(
        sparseMode == MASK_MODE_CAUSAL && cmpRatio != 1 && cmpResidualK == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
            opName, "cmp_residual_k", "cmp_residual_k is required when mask_mode is 3 with cmp_ratio not equal to 1"),
        return false);
    OP_LOGD(context_, "attrs: layout_q[%s], layout_k[%s], mask_mode[%d], cmp_ratio[%ld], deterministic[%d].",
            inputLayout, keyLayout, sparseMode, cmpRatio, deterministic);
    return true;
}

bool SparseLightningIndexerKLLossGradTilingBase::AnalyzeDtype()
{
    auto qDesc = context_->GetInputDesc(QUERY_INPUT_INDEX);
    auto kDesc = context_->GetInputDesc(KEY_INPUT_INDEX);
    auto wDesc = context_->GetInputDesc(WEIGHT_INPUT_INDEX);
    auto sparseDesc = context_->GetInputDesc(SPARSE_INDICES_INPUT_INDEX);
    auto softmaxDesc = context_->GetInputDesc(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX);
    auto cuSeqQDesc = context_->GetOptionalInputDesc(CU_SEQLENS_QUERY_INPUT_INDEX);
    auto cuSeqKDesc = context_->GetOptionalInputDesc(CU_SEQLENS_KEY_INPUT_INDEX);
    auto seqUsedQDesc = context_->GetOptionalInputDesc(SEQUSED_QUERY_INPUT_INDEX);
    auto seqUsedKDesc = context_->GetOptionalInputDesc(SEQUSED_KEY_INPUT_INDEX);
    auto cmpResidualKDesc = context_->GetOptionalInputDesc(CMP_RESIDUAL_KEY_INPUT_INDEX);
    auto metadataDesc = context_->GetOptionalInputDesc(METADATA_INPUT_INDEX);
    OP_CHECK_IF(qDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "q",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(kDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "k",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(wDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "w",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(sparseDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "sparse",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(softmaxDesc == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(Ops::Base::GetOpInfo(context_).c_str(), "attn_softmax_l1_norm",
                                                         "required shape, descriptor or attribute is missing"),
                return ge::GRAPH_FAILED);

    auto qDtype = qDesc->GetDataType();
    auto kDtype = kDesc->GetDataType();
    OP_CHECK_IF(
        !((qDtype == ge::DT_FLOAT16 && kDtype == ge::DT_FLOAT16) || (qDtype == ge::DT_BF16 && kDtype == ge::DT_BF16)),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
            (std::string() + std::string(ge::TypeUtils::DataTypeToSerialString(qDtype).c_str()) + ", " +
             std::string(ge::TypeUtils::DataTypeToSerialString(kDtype).c_str()))
                .c_str(),
            (std::string() + "q/k dtype must be both fp16 or both bf16, but got q[" +
             std::string(ge::TypeUtils::DataTypeToSerialString(qDtype).c_str()) + "], k[" +
             std::string(ge::TypeUtils::DataTypeToSerialString(kDtype).c_str()) + "].")
                .c_str()),
        return false);
    OP_CHECK_IF(
        wDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
            (std::string() + std::string(ge::TypeUtils::DataTypeToSerialString(wDesc->GetDataType()).c_str())).c_str(),
            (std::string() + "w dtype must be fp32, but got [" +
             std::string(ge::TypeUtils::DataTypeToSerialString(wDesc->GetDataType()).c_str()) + "].")
                .c_str()),
        return false);
    OP_CHECK_IF(
        sparseDesc->GetDataType() != ge::DT_INT32,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName, "sparse_indices",
            (std::string() + std::string(ge::TypeUtils::DataTypeToSerialString(sparseDesc->GetDataType()).c_str()))
                .c_str(),
            (std::string() + "sparse_indices dtype must be int32, but got [" +
             std::string(ge::TypeUtils::DataTypeToSerialString(sparseDesc->GetDataType()).c_str()) + "].")
                .c_str()),
        return false);
    OP_CHECK_IF(
        softmaxDesc->GetDataType() != ge::DT_FLOAT,
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            opName, "attn_softmax_l1_norm",
            (std::string() + std::string(ge::TypeUtils::DataTypeToSerialString(softmaxDesc->GetDataType()).c_str()))
                .c_str(),
            (std::string() + "attn_softmax_l1_norm dtype must be fp32, but got [" +
             std::string(ge::TypeUtils::DataTypeToSerialString(softmaxDesc->GetDataType()).c_str()) + "].")
                .c_str()),
        return false);

    const gert::CompileTimeTensorDesc* int32OptionalInputs[] = {cuSeqQDesc,   cuSeqKDesc,       seqUsedQDesc,
                                                                seqUsedKDesc, cmpResidualKDesc, metadataDesc};
    const char* inputNames[] = {"cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k", "cmp_residual_k", "metadata"};
    for (size_t i = 0; i < sizeof(int32OptionalInputs) / sizeof(int32OptionalInputs[0]); ++i) {
        if (int32OptionalInputs[i] != nullptr && int32OptionalInputs[i]->GetDataType() != ge::DT_INT32) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
                (std::string() + std::string(inputNames[i]) + ", " +
                 std::string(ge::TypeUtils::DataTypeToSerialString(int32OptionalInputs[i]->GetDataType()).c_str()))
                    .c_str(),
                (std::string() + std::string(inputNames[i]) + " dtype must be int32, but got [" +
                 std::string(ge::TypeUtils::DataTypeToSerialString(int32OptionalInputs[i]->GetDataType()).c_str()) +
                 "].")
                    .c_str());
            return false;
        }
    }
    return true;
}

bool SparseLightningIndexerKLLossGradTilingBase::AnalyzeLayout()
{
    auto& qShape = context_->GetInputShape(QUERY_INPUT_INDEX)->GetStorageShape();
    auto& kShape = context_->GetInputShape(KEY_INPUT_INDEX)->GetStorageShape();
    auto& wShape = context_->GetInputShape(WEIGHT_INPUT_INDEX)->GetStorageShape();
    auto& sparseShape = context_->GetInputShape(SPARSE_INDICES_INPUT_INDEX)->GetStorageShape();
    auto& softmaxShape = context_->GetInputShape(ATTN_SOFTMAX_L1_NORM_INPUT_INDEX)->GetStorageShape();

    size_t qLayoutLen = strlen(inputLayout);
    size_t kLayoutLen = strlen(keyLayout);
    OP_CHECK_IF(
        qShape.GetDimNum() != qLayoutLen || kShape.GetDimNum() != kLayoutLen,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            opName, "attributes", (std::string() + std::string(inputLayout) + ", " + std::string(keyLayout)).c_str(),
            (std::string() + "Invalid q/k layout. layout_q[" + std::string(inputLayout) + "], layout_k[" +
             std::string(keyLayout) + "].")
                .c_str()),
        return false);
    OP_CHECK_IF(qLayoutLen != kLayoutLen,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                    opName, "attributes", "layout_q/layout_k rank must be the same in current implementation."),
                return false);
    OP_CHECK_IF(wShape.GetDimNum() + 1 != qShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
                                                         "w rank must be q rank - 1."),
                return false);
    OP_CHECK_IF(sparseShape.GetDimNum() != qShape.GetDimNum() || softmaxShape.GetDimNum() != sparseShape.GetDimNum(),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "sparse_indices, attn_softmax_l1_norm",
                                                         "sparse_indices and attn_softmax_l1_norm rank mismatch."),
                return false);

    bool qTnd = (qLayoutLen == 3UL && inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D');
    bool qBsnd = (qLayoutLen == 4UL && inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' &&
                  inputLayout[3] == 'D');
    bool kTnd = (kLayoutLen == 3UL && keyLayout[0] == 'T' && keyLayout[1] == 'N' && keyLayout[2] == 'D');
    bool kBsnd =
        (kLayoutLen == 4UL && keyLayout[0] == 'B' && keyLayout[1] == 'S' && keyLayout[2] == 'N' && keyLayout[3] == 'D');
    OP_CHECK_IF(
        !(qTnd || qBsnd) || !(kTnd || kBsnd),
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "attributes", "layout_q/layout_k only support TND or BSND."),
        return false);

    if (qTnd) {
        auto cuSeqQShape = context_->GetOptionalInputShape(CU_SEQLENS_QUERY_INPUT_INDEX);
        auto cuSeqKShape = context_->GetOptionalInputShape(CU_SEQLENS_KEY_INPUT_INDEX);
        OP_CHECK_IF(
            cuSeqQShape == nullptr || cuSeqKShape == nullptr,
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
                                                     "TND layout requires cu_seqlens_q/cu_seqlens_k tensor inputs."),
            return false);
        auto& cuSeqQStorageShape = cuSeqQShape->GetStorageShape();
        auto& cuSeqKStorageShape = cuSeqKShape->GetStorageShape();
        OP_CHECK_IF(cuSeqQStorageShape.GetDimNum() != 1 || cuSeqKStorageShape.GetDimNum() != 1,
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName, "attributes", "cu_seqlens_q/cu_seqlens_k must be 1D tensors in TND layout."),
                    return false);
        OP_CHECK_IF(cuSeqQStorageShape.GetDim(0) <= 1 || cuSeqQStorageShape.GetDim(0) != cuSeqKStorageShape.GetDim(0),
                    OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                        opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
                        "cu_seqlens_q/cu_seqlens_k length must be equal and larger than 1."),
                    return false);
        bSize = cuSeqQStorageShape.GetDim(0) - 1;
        realT1Size = qShape.GetDim(0);
        accumS1 = qShape.GetDim(0);
        accumS2 = kShape.GetDim(0);
        s1Size = qShape.GetDim(0);
        s2Size = kShape.GetDim(0);
        n2Size = kShape.GetDim(1);
        gSizeQueryIndex = qShape.GetDim(1) / n2Size;
        dSizeQueryIndex = qShape.GetDim(2);
        dSizeQuery = dSizeQueryIndex;
        gSizeQuery = gSizeQueryIndex;
        kSize = sparseShape.GetDim(2);
        tilingData->baseParams.set_layoutType(LAYOUT_TND);
        tilingKeyLayout = LayoutType::LAYOUT_TND;
    } else {
        bSize = qShape.GetDim(0);
        s1Size = qShape.GetDim(1);
        s2Size = kShape.GetDim(1);
        n2Size = kShape.GetDim(2);
        gSizeQueryIndex = qShape.GetDim(2) / n2Size;
        dSizeQueryIndex = qShape.GetDim(3);
        dSizeQuery = dSizeQueryIndex;
        gSizeQuery = gSizeQueryIndex;
        kSize = sparseShape.GetDim(3);
        tilingData->baseParams.set_layoutType(LAYOUT_BSND);
        tilingKeyLayout = LayoutType::LAYOUT_BSND;
    }

    auto checkOptionalVector = [this](uint32_t inputIndex, const char* inputName, int64_t expectedElementNum) -> bool {
        auto optionalShape = context_->GetOptionalInputShape(inputIndex);
        if (optionalShape == nullptr) {
            return true;
        }
        auto& storageShape = optionalShape->GetStorageShape();
        if (storageShape.GetDimNum() != 1 || storageShape.GetDim(0) != expectedElementNum) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                Ops::Base::GetOpInfo(context_).c_str(), "q, k, w, sparse_indices, attn_softmax_l1_norm",
                (std::string() + std::string(inputName) + ", " + std::to_string(expectedElementNum)).c_str(),
                (std::string() + std::string(inputName) + " must be a 1D tensor with " +
                 std::to_string(expectedElementNum) + " elements.")
                    .c_str());
            return false;
        }
        return true;
    };
    OP_CHECK_IF(!checkOptionalVector(SEQUSED_QUERY_INPUT_INDEX, "seqused_q", bSize) ||
                    !checkOptionalVector(SEQUSED_KEY_INPUT_INDEX, "seqused_k", bSize) ||
                    !checkOptionalVector(CMP_RESIDUAL_KEY_INPUT_INDEX, "cmp_residual_k", bSize) ||
                    !checkOptionalVector(METADATA_INPUT_INDEX, "metadata", SLI_METADATA_SIZE),
                OP_LOGE_WITHOUT_REPORT(opName, "optional input shape validation failed."), return false);

    OP_CHECK_IF(n2Size <= 0 || qShape.GetDim(qLayoutLen - 2) % n2Size != 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
                                                         "q N dimension must be divisible by k N dimension."),
                return false);
    OP_CHECK_IF(
        n2Size != 1,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            opName, "q, k, w, sparse_indices, attn_softmax_l1_norm", (std::string() + std::to_string(n2Size)).c_str(),
            (std::string() + "current kernel path only supports N2 == 1, but got [" + std::to_string(n2Size) + "].")
                .c_str()),
        return false);
    OP_CHECK_IF(gSizeQueryIndex != 8 && gSizeQueryIndex != 16 && gSizeQueryIndex != 32 && gSizeQueryIndex != 64,
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                    opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
                    (std::string() + std::to_string(gSizeQueryIndex)).c_str(),
                    (std::string() + "current kernel path only supports N1/N2 in {8, 16, 32, 64}, but got [" +
                     std::to_string(gSizeQueryIndex) + "].")
                        .c_str()),
                return false);
    OP_CHECK_IF(dSizeQueryIndex != kShape.GetDim(kLayoutLen - 1),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
                                                         "q/k D dimension must be equal."),
                return false);
    OP_CHECK_IF(dSizeQueryIndex != 128,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
                                                         "q/k D dimension must be equal to 128."),
                return false);
    OP_CHECK_IF(wShape.GetDim(wShape.GetDimNum() - 1) != qShape.GetDim(qLayoutLen - 2),
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
                                                         "w N dimension must equal q N dimension."),
                return false);
    OP_CHECK_IF(sparseShape.GetDim(sparseShape.GetDimNum() - 2) != n2Size ||
                    softmaxShape.GetDim(softmaxShape.GetDimNum() - 2) != n2Size ||
                    softmaxShape.GetDim(softmaxShape.GetDimNum() - 1) != kSize,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "sparse_indices, attn_softmax_l1_norm",
                                                         "sparse_indices/attn_softmax_l1_norm N2/K shape mismatch."),
                return false);
    OP_CHECK_IF(kSize < 1 || kSize > BUFFER_SIZE_BYTE_8K,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(opName, "sparse_indices", std::to_string(kSize).c_str(),
                                                      "K must be in range [1, 8192]"),
                return false);
    topkSize = (kSize <= static_cast<int32_t>(BUFFER_SIZE_BYTE_2K)) ? TopKRange::RANGE_0_2K : TopKRange::RANGE_2K_8K;
    return true;
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingBase::GetPlatformInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr =
            reinterpret_cast<const SparseLightningIndexerKLLossGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr,
                    REPORT_PREDEFINED_ERR_MSG("EZ1008", (std::vector<const char*>{"opName", "reason"}),
                                              (std::vector<const char*>{opName, "compileInfoPtr is null."})),
                    return ge::GRAPH_FAILED);
        aivNum = compileInfoPtr->aivNum;
        aicNum = compileInfoPtr->aicNum;
        aicoreParams_.ubSize = compileInfoPtr->ubSize;
        aicoreParams_.l1Size = compileInfoPtr->l1Size;
        aicoreParams_.l0cSize = compileInfoPtr->l0cSize;
        l2CacheSize = compileInfoPtr->l2CacheSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        aivNum = ascendcPlatform.GetCoreNumAiv();
        aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, aicoreParams_.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, aicoreParams_.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, aicoreParams_.l0cSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2CacheSize);
    }
    OP_LOGI(context_, "get platform from compileInfo.aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu).",
            aivNum, aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: %s.", GetTilingContextDebugStr().c_str());
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT(opName, "invalid context."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout(),
                OP_LOGE_WITHOUT_REPORT(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);

    sliGradkllossBaseParams_->set_bSize(bSize);
    sliGradkllossBaseParams_->set_n2Size(n2Size);
    sliGradkllossBaseParams_->set_gSizeQuery(gSizeQuery);
    sliGradkllossBaseParams_->set_gSizeQueryIndex(gSizeQueryIndex);
    sliGradkllossBaseParams_->set_s1Size(s1Size);
    sliGradkllossBaseParams_->set_s2Size(s2Size);
    sliGradkllossBaseParams_->set_dSizeQuery(dSizeQuery);
    sliGradkllossBaseParams_->set_dSizeQueryIndex(dSizeQueryIndex);
    sliGradkllossBaseParams_->set_kSize(kSize);
    sliGradkllossBaseParams_->set_sparseMode(sparseMode);
    sliGradkllossBaseParams_->set_scaleValue(scaleValue);
    sliGradkllossBaseParams_->set_cmpRatio(cmpRatio);
    sliGradkllossBaseParams_->set_hasSoftmaxInput(hasSoftmaxInput);

    OP_LOGW(context_,
            "INPUTPARAM bSize:[%d], n2Size:[%d], gSizeQuery:[%d], gSizeQueryIndex:[%d], s1Size:[%d], s2Size:[%d], "
            "dSize:[%d], kSize:[%d], maskMode:[%d], cmpRatio:[%ld].",
            bSize, n2Size, gSizeQuery, gSizeQueryIndex, s1Size, s2Size, dSizeQueryIndex, kSize, sparseMode, cmpRatio);
    return ge::GRAPH_SUCCESS;
}

int64_t SparseLightningIndexerKLLossGradTilingBase::CalcTotalSize()
{
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        return realT1Size;
    }
    if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
        return static_cast<int64_t>(bSize) * s1Size;
    }
    return 0;
}

void SparseLightningIndexerKLLossGradTilingBase::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t coreNum)
{
    int64_t actualUsedCoreNum = std::min(totalSize, std::min(coreNum, static_cast<int64_t>(MAX_CORE_NUM)));
    sliGradkllossMultiCoreParams_->set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    sliGradkllossMultiCoreParams_->set_totalSize(totalSize);
    sliGradkllossMultiCoreParams_->set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    int64_t splitFactorSize = sliGradkllossMultiCoreParams_->get_splitFactorSize();
    int64_t* bS1Index = sliGradkllossMultiCoreParams_->get_bS1Ptr();
    for (int64_t idx = 0; idx < static_cast<int64_t>(MAX_CORE_NUM); ++idx) {
        bS1Index[idx] = totalSize;
    }
    for (int64_t idx = 0; idx < actualUsedCoreNum; ++idx) {
        bS1Index[idx] = std::min(idx * splitFactorSize, totalSize);
    }
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingBase::InitOutputSplit()
{
    SLIKLLossGradInitOutputParams* initoutput = &tilingData->initOutputParams;
    auto& dKeyShape = context_->GetOutputShape(D_KEY_OUTPUT_INDEX)->GetStorageShape();
    int64_t totalSize = 0;
    if (tilingKeyLayout == LayoutType::LAYOUT_TND) {
        totalSize = dKeyShape.GetDim(0) * dKeyShape.GetDim(2);
    } else if (tilingKeyLayout == LayoutType::LAYOUT_BSND) {
        totalSize = static_cast<int64_t>(bSize) * dKeyShape.GetDim(1) * dKeyShape.GetDim(3);
    }
    int64_t singleCoreSize = CeilDivision(totalSize, static_cast<int64_t>(aivNum));
    if (singleCoreSize > UINT32_MAX) {
        REPORT_PREDEFINED_ERR_MSG(
            "EZ1008", (std::vector<const char*>{"opName", "reason"}),
            (std::vector<const char*>{
                Ops::Base::GetOpInfo(context_).c_str(),
                (std::string() + "singleCoreSize(" + std::to_string(singleCoreSize) + ") exceeds UINT32_MAX limit.")
                    .c_str()}));
        return ge::GRAPH_FAILED;
    }
    initoutput->set_singleCoreSize(static_cast<uint32_t>(singleCoreSize));
    initoutput->set_totalOutputSize(totalSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingBase::DoOpTiling()
{
    OP_LOGD(context_, "try template[%s]", templateName);
    int64_t totalSize = CalcTotalSize();
    OP_CHECK_IF(totalSize <= 0,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(opName, "q, k, w, sparse_indices, attn_softmax_l1_norm",
                                                         "totalSize should be larger than 0."),
                return ge::GRAPH_FAILED);
    SetMultiCoreParamsRegbase(totalSize, static_cast<int64_t>(aicNum));
    context_->SetBlockDim(sliGradkllossMultiCoreParams_->get_coreNum());

    int64_t kAlign8Softmax = ((static_cast<int64_t>(kSize) + 7) / 8) * 8;
    std::vector<int64_t> shapeVec = {1, kAlign8Softmax};
    ge::Shape srcShape(shapeVec);
    int64_t softmaxTmpBufferSize = (kSize > BUFFER_SIZE_BYTE_2K) ? BUFFER_SIZE_BYTE_33K : BUFFER_SIZE_BYTE_32K;
    SoftMaxTilingFunc(srcShape, sizeof(float), softmaxTmpBufferSize, tilingData->vectorParams.softmaxYTilingData);
    SoftMaxTilingFunc(srcShape, sizeof(float), softmaxTmpBufferSize, tilingData->vectorParams.simpleSoftmaxPTilingData);

    auto ret = InitOutputSplit();
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    OP_LOGD(context_, "ending template[%s]", templateName);
    return ge::GRAPH_SUCCESS;
}

uint64_t SparseLightningIndexerKLLossGradTilingBase::GetTilingKey() const
{
    LayoutType qLayout = (tilingKeyLayout == LayoutType::LAYOUT_TND) ? LayoutType::LAYOUT_TND : LayoutType::LAYOUT_BSND;
    LayoutType kLayout =
        (keyLayout != nullptr && keyLayout[0] == 'T') ? LayoutType::LAYOUT_TND : LayoutType::LAYOUT_BSND;
    bool hasSequsedQ = context_->GetOptionalInputTensor(SEQUSED_QUERY_INPUT_INDEX) != nullptr;
    bool hasSequsedK = context_->GetOptionalInputTensor(SEQUSED_KEY_INPUT_INDEX) != nullptr;
    return GET_TPL_TILING_KEY(
        static_cast<uint8_t>(false), static_cast<uint32_t>(topkSize), static_cast<uint8_t>(qLayout),
        static_cast<uint8_t>(kLayout), static_cast<uint8_t>(sparseMode), static_cast<uint8_t>(hasSequsedQ),
        static_cast<uint8_t>(hasSequsedK), static_cast<uint8_t>(deterministic), static_cast<uint8_t>(privateScatter));
}

ge::graphStatus SparseLightningIndexerKLLossGradTilingBase::GetWorkspaceSize()
{
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    int64_t kAlign16 = ((static_cast<int64_t>(kSize) + 15) / 16) * 16;
    int64_t pSize = static_cast<int64_t>(kSize) * dSizeQuery * sizeof(uint16_t);
    int64_t sySize = static_cast<int64_t>(kSize) * dSizeQueryIndex * sizeof(uint16_t);
    int64_t bmm1Size = static_cast<int64_t>(gSizeQuery) * kAlign16 * sizeof(float);
    int64_t bmm2Size = static_cast<int64_t>(gSizeQueryIndex) * kAlign16 * sizeof(float);
    int64_t reluGradSize = static_cast<int64_t>(gSizeQueryIndex) * kAlign16 * sizeof(float);
    int64_t psySyncSize = (kAlign16 * 2 + 32 / sizeof(float)) * sizeof(float);
    int64_t bmm3Size = static_cast<int64_t>(kSize) * dSizeQueryIndex * sizeof(float);

    int64_t scatterElems = (tilingKeyLayout == LayoutType::LAYOUT_TND) ? accumS2 : static_cast<int64_t>(bSize) * s2Size;
    int64_t scatterAddOutSize =
        scatterElems * static_cast<int64_t>(dSizeQueryIndex) * static_cast<int64_t>(sizeof(float));

    int64_t coreNum = static_cast<int64_t>(sliGradkllossMultiCoreParams_->get_coreNum());
    int64_t singleCoreTotalSize =
        PING_PONG_VALUE * (pSize + bmm1Size + bmm2Size + reluGradSize + sySize + psySyncSize + bmm3Size);
    int64_t multiCoreBase = singleCoreTotalSize * coreNum;

    privateScatter = false;
    int64_t scatterBankNum = 1;
    int64_t totalS1 = 0;
    if (deterministic) {
        totalS1 = CalcTotalSize();
        const int64_t batchSize = static_cast<int64_t>(bSize);
        privateScatter = IsPrivateScatter(scatterElems, totalS1, batchSize, static_cast<int64_t>(kSize));
        scatterBankNum = privateScatter ? coreNum : static_cast<int64_t>(SLI_DETER_SCATTER_BANK_NUM);
    }

    int64_t workspaceSize = CalcWorkspaceSize(scatterAddOutSize, scatterBankNum, multiCoreBase);
    const int64_t multiCoreTotalSize = multiCoreBase + scatterAddOutSize * scatterBankNum;
    workspaces[0] = static_cast<size_t>(workspaceSize);
    OP_LOGW(context_,
            "workspace size:[%zu], multicoreTotalSize:[%ld], scatterElems:[%ld], totalS1:[%ld], kSize:[%d], "
            "scatterBankNum:[%ld], privateScatter:[%d]",
            workspaces[0], multiCoreTotalSize, scatterElems, totalS1, kSize, scatterBankNum,
            static_cast<int>(privateScatter));
    return ge::GRAPH_SUCCESS;
}

} // namespace optiling
