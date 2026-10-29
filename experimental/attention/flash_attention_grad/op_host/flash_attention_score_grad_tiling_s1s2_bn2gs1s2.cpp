/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2.h"
#include "base/tiling_type.h"
#include "base/tiling_templates_registry.h"

namespace optiling {

bool FlashAttentionScoreGradTilingUs1s2Bs2::IsCapable()
{
    // 基础模板 全部支持
    bool deterministic = *context_->GetAttrs()->GetAttrPointer<int>(ATTR_DETERMINISTIC_INDEX);
    if (deterministic) {
        return false;
    }
    return true;
}

uint64_t FlashAttentionScoreGradTilingUs1s2Bs2::GetTilingKey() const
{
    auto dtypeValue = fBaseParams.mode == BF16 ? DtypeEnum::BFLOAT16 : DtypeEnum::FLOAT16_PRECISION;
    auto unpackValue = fBaseParams.layoutType == INPUT_FROAMT_TND ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    auto isNZOut = fBaseParams.isNZOut && unpackValue != OptionEnum::ENABLE ? OptionEnum::ENABLE : OptionEnum::DISABLE;
    uint64_t tilingKey = GET_TILINGKEY(AxisEnum::S2, AxisEnum::S1, AxisEnum::S2, dtypeValue, LayoutEnum::BSND,
                                       SparseEnum::ALL, unpackValue, isNZOut);
    OP_LOGD(opName, "FAGTiling Us1s2Bbn2gs1s2 DoTiling success, tiling is %lu.", tilingKey);
    return tilingKey;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2::GetPlatformInfo()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    uint32_t coreNum = CORE_INIT_NUM; // 40 is init core num
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, TILING_INNER_ERR_REPORT(context_->GetNodeName(), "compile_info is null"),
                    return ge::GRAPH_FAILED);

        fBaseParams.coreNum = compileInfoPtr->aivNum;
        fBaseParams.aicNum = compileInfoPtr->aicNum;
        fBaseParams.ubSize = compileInfoPtr->ubSize;
        fBaseParams.l1Size = compileInfoPtr->l1Size;
        fBaseParams.l0aSize = compileInfoPtr->l0aSize;
        fBaseParams.l0cSize = compileInfoPtr->l0cSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
        coreNum = ascendcPlatform.GetCoreNumAiv();
        OP_CHECK_IF(coreNum == 0, TILING_INNER_ERR_REPORT(context_->GetNodeName(), "num of coreNum is 0."),
                    return ge::GRAPH_FAILED);

        fBaseParams.coreNum = coreNum;
        fBaseParams.aicNum = ascendcPlatform.GetCoreNumAic();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, fBaseParams.ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, fBaseParams.l1Size);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, fBaseParams.l0aSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, fBaseParams.l0cSize);
    }

    fBaseParams.ubSize -= MATMUL_SIZE;
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreGradTilingUs1s2Bs2::SetSparseParams()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    if (fBaseParams.layoutType == INPUT_FROAMT_TND) {
        OP_LOGD(opName, "in the TND scenario,isSparse is true by default");
        return true;
    }

    auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK_INDEX);
    auto attenMaskDesc = context_->GetOptionalInputDesc(ATTEN_MASK_INDEX);
    if (fBaseParams.sparseMode == ALL_MASK || attenMaskShape == nullptr || attenMaskDesc == nullptr) {
        OP_LOGD(opName, "in the ALL_MASK or attenMask is none scenario,isSparse is false");
        return false;
    }

    // 兼容老版本，未配置sparseMode或配置sparseMode为0的处理
    if (fBaseParams.sparseMode == NO_MASK) {
        if (int64_t(fBaseParams.s1) > fBaseParams.s1Token ||
            int64_t(fBaseParams.s2) > fBaseParams.s2Token) { // band场景，包含causal
            OP_LOGD(opName, "in the NONE_MASK  and token is band scenario,isSparse is true");
            return true;
        } else {
            OP_LOGD(opName, "in the NONE_MASK  and token is not band scenario,isSparse is false");
            return false;
        }
    }

    if (fBaseParams.sparseMode == LEFT_UP_CAUSAL || fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL) {
        OP_LOGD(opName, "in the LEFT_UP_CAUSAL or RIGHT_DOWN_CAUSAL scenario,isSparse is true");
        return true;
    }

    if (fBaseParams.sparseMode == BAND &&
        (int64_t(fBaseParams.s1) > fBaseParams.s1Token || int64_t(fBaseParams.s2) > fBaseParams.s2Token)) {
        OP_LOGD(opName, "in the BAND  and token is band scenario,isSparse is true ");
        return true;
    }

    OP_LOGD(opName, "no scenario is hit, isSparse is false ");
    return false;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2::GetShapeAttrsInfo()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    const gert::StorageShape* queryShape = context_->GetInputShape(QUARY_INDEX); // [B, N2, G, S1, D]
    const gert::StorageShape* keyShape = context_->GetInputShape(KEY_INDEX);     // [B, N2, 1, S2, D]
    const gert::StorageShape* valueShape = context_->GetInputShape(VALUE_INDEX); // [B, N2, 1, S2, DV]

    uint32_t head_num = *context_->GetAttrs()->GetAttrPointer<int>(ATTR_HEAD_NUM_INDEX);
    const char* inputLayout = context_->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);

    if (strcmp(inputLayout, "SBH") == 0) {
        OP_LOGD(opName, "GetShapeAttrsInfo, inputLayout == SBH queryShape");
        fBaseParams.layoutType = INPUT_FROAMT_S2BN2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.n2 = head_num / fBaseParams.g; // 跟se和mde讨论  按 head_num=n1 计算
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / head_num;  // H=N*D
        fBaseParams.dV = valueShape->GetStorageShape().GetDim(INPUT_DIM_2) / head_num; // H=N*D
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_0);
    } else if (strcmp(inputLayout, "BSH") == 0) {
        OP_LOGD(opName, "GetShapeAttrsInfo, inputLayout == BSH queryShape");
        fBaseParams.layoutType = INPUT_FROAMT_BS2N2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.n2 = head_num / fBaseParams.g;
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / head_num;  // H=N*D
        fBaseParams.dV = valueShape->GetStorageShape().GetDim(INPUT_DIM_2) / head_num; // H=N*D
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
    } else if (strcmp(inputLayout, "BNSD") == 0) {
        OP_LOGD(opName, "GetShapeAttrsInfo, inputLayout == BNSD queryShape");
        fBaseParams.layoutType = INPUT_FROAMT_BN2GS2D;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_1) / keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.dV = valueShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        OP_LOGD(opName, "GetShapeAttrsInfo, inputLayout == BNSD queryShape", "%ld, %ld, %ld, %ld,",
                queryShape->GetStorageShape().GetDim(INPUT_DIM_0), queryShape->GetStorageShape().GetDim(INPUT_DIM_1),
                queryShape->GetStorageShape().GetDim(INPUT_DIM_2), queryShape->GetStorageShape().GetDim(INPUT_DIM_3));
    } else if (strcmp(inputLayout, "TND") == 0) {
        OP_LOGD(opName, "GetShapeAttrsInfo, inputLayout == TND");
        fBaseParams.layoutType = INPUT_FROAMT_TND;
        auto attrs = context_->GetAttrs();
        const gert::ContinuousVector* seqQData =
            attrs->GetAttrPointer<gert::ContinuousVector>(ACTUAL_SEQ_Q_LEN_ATTRS_INDEX);
        const gert::ContinuousVector* seqKVData =
            attrs->GetAttrPointer<gert::ContinuousVector>(ACTUAL_SEQ_KV_LEN_ATTRS_INDEX);
        if (seqQData == nullptr) {
            OP_LOGE(opName, "GetShapeAttrsInfo, actualSeqLenQ is null pointer");
            return ge::GRAPH_PARAM_INVALID;
        }
        if (seqKVData == nullptr) {
            OP_LOGE(opName, "GetShapeAttrsInfo, actualSeqLenKV is null pointer");
            return ge::GRAPH_PARAM_INVALID;
        }

        // get seq tensor data input
        const gert::StorageShape* seqQTensorShape = context_->GetInputShape(ACTUAL_SEQ_Q_LEN_INDEX);
        const gert::StorageShape* seqKvTensorShape = context_->GetInputShape(ACTUAL_SEQ_KV_LEN_INDEX);
        if (seqQTensorShape->GetStorageShape().GetDimNum() != 1 ||
            seqKvTensorShape->GetStorageShape().GetDimNum() != 1) {
            OP_LOGE(opName, "GetShapeAttrsInfo, actualSeqLenK tensor input dim size should be 1");
            return ge::GRAPH_PARAM_INVALID;
        }

        const size_t seqQShapeSize = seqQData->GetSize();
        const size_t kvSeqShapeSize = seqKVData->GetSize();
        if (seqQShapeSize != kvSeqShapeSize) {
            OP_LOGE(opName, "GetShapeAttrsInfo, actualSeqQlenTensor shapeSize is not equal actualSeqKvlenTensor");
            return ge::GRAPH_PARAM_INVALID;
        }
        if (seqQShapeSize < 2) {
            OP_LOGE(opName, "actualSeqQ array size cannot be less than 2");
            return ge::GRAPH_PARAM_INVALID;
        }
        if (seqQShapeSize != seqQTensorShape->GetStorageShape().GetDim(0)) { // tensor size != attr input size
            OP_LOGE(opName,
                    "GetShapeAttrsInfo, actualSeqlenTensor shapeSize is not equal actualSeqlenAttrs input size");
            return ge::GRAPH_PARAM_INVALID;
        }

        const int64_t* qValue = reinterpret_cast<const int64_t*>(seqQData->GetData());
        const int64_t* kvValue = reinterpret_cast<const int64_t*>(seqKVData->GetData());
        OP_CHECK_IF((qValue == nullptr || kvValue == nullptr),
                    TILING_INNER_ERR_REPORT("FlashAttentionGradTiling", "qValue or kvValue is nullptr."),
                    return ge::GRAPH_FAILED);

        // seq data is accmulated seq len begin with 0 as gpu
        for (size_t i = 0; i < seqQShapeSize - 1; i++) {
            fBaseParams.actualSeqQlen.push_back(qValue[i + 1] - qValue[i]);
            fBaseParams.actualSeqKvlen.push_back(kvValue[i + 1] - kvValue[i]);
            fBaseParams.sumS1S2Product += fBaseParams.actualSeqQlen[i] * fBaseParams.actualSeqKvlen[i];
            OP_LOGD(opName, "CurrentSeqLenQ %d is %ld.", i, fBaseParams.actualSeqQlen[i]);
            OP_LOGD(opName, "CurrentSeqLenKV %d is %ld.", i, fBaseParams.actualSeqKvlen[i]);
        }

        fBaseParams.b = seqQShapeSize - 1;
        fBaseParams.t1 = qValue[fBaseParams.b];
        fBaseParams.t2 = kvValue[fBaseParams.b];
        fBaseParams.s1 = *std::max_element(fBaseParams.actualSeqQlen.begin(), fBaseParams.actualSeqQlen.end());
        fBaseParams.s2 = *std::max_element(fBaseParams.actualSeqKvlen.begin(), fBaseParams.actualSeqKvlen.end());

        OP_LOGD(opName, "Seq Nums is %d, SeqQ is %ld, SeqK is %ld, max SeqQlen is %ld, max SeqKlen is %ld.",
                fBaseParams.b, fBaseParams.t1, fBaseParams.t2, fBaseParams.s1, fBaseParams.s2);

        if (keyShape->GetStorageShape().GetDimNum() == 4) {
            fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
            fBaseParams.g =
                queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
            fBaseParams.d = queryShape->GetStorageShape().GetDim(INPUT_DIM_3);
            fBaseParams.dV = valueShape->GetStorageShape().GetDim(INPUT_DIM_3);
        } else {
            fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
            fBaseParams.g =
                queryShape->GetStorageShape().GetDim(INPUT_DIM_1) / keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
            fBaseParams.d = queryShape->GetStorageShape().GetDim(INPUT_DIM_2);
            fBaseParams.dV = valueShape->GetStorageShape().GetDim(INPUT_DIM_2);
        }

    } else {
        OP_LOGD(opName, "GetShapeAttrsInfo, inputLayout == BSND queryShape");
        // inputLayout = "BSND"
        fBaseParams.layoutType = INPUT_FROAMT_BS2N2GD;
        fBaseParams.b = queryShape->GetStorageShape().GetDim(INPUT_DIM_0);
        fBaseParams.n2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.g =
            queryShape->GetStorageShape().GetDim(INPUT_DIM_2) / keyShape->GetStorageShape().GetDim(INPUT_DIM_2);
        fBaseParams.s1 = queryShape->GetStorageShape().GetDim(INPUT_DIM_1);
        fBaseParams.d = queryShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.dV = valueShape->GetStorageShape().GetDim(INPUT_DIM_3);
        fBaseParams.s2 = keyShape->GetStorageShape().GetDim(INPUT_DIM_1);
    }

    fBaseParams.n1 = fBaseParams.n2 * fBaseParams.g;
    fBaseParams.s1Align = (fBaseParams.s1 + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;
    fBaseParams.s2Align = (fBaseParams.s2 + INPUT_ALIGN - 1) / INPUT_ALIGN * INPUT_ALIGN;
    fBaseParams.qSize = fBaseParams.t1 * static_cast<int64_t>(fBaseParams.n2 * fBaseParams.g * fBaseParams.d);
    fBaseParams.kSize = fBaseParams.t2 * static_cast<int64_t>(fBaseParams.n2 * 1 * fBaseParams.d);
    fBaseParams.vSize = fBaseParams.t2 * static_cast<int64_t>(fBaseParams.n2 * 1 * fBaseParams.dV);

    // mBaseParams is used for matmal tiling module
    auto queryType = context_->GetInputDesc(0)->GetDataType();
    fBaseParams.queryType = queryType;
    fBaseParams.isBf16 = queryType == ge::DT_BF16 ? true : false;
    if (queryType == ge::DT_FLOAT) {
        fBaseParams.dataTypeSize = FP32_BYTES; // init date type fp32 is 4
        fBaseParams.dataBlockNum = FP32_BLOCK_NUMS;
        fBaseParams.calTypeSize = FP32_BYTES; // init cal type fp32 is 4
        fBaseParams.calBlockNum = FP32_BLOCK_NUMS;
    } else {
        fBaseParams.dataTypeSize = FP16_BYTES; // init date type fp32 is 4
        fBaseParams.dataBlockNum = FP16_BLOCK_NUMS;
        fBaseParams.calTypeSize = FP32_BYTES; // init cal type fp32 is 4
        fBaseParams.calBlockNum = FP32_BLOCK_NUMS;
    }

    // LSE CASE
    auto softmaxLseShape = context_->GetOptionalInputShape(SOFTMAX_LSE_INDEX);
    if (softmaxLseShape == nullptr || softmaxLseShape->GetStorageShape().GetShapeSize() == 0) {
        fBaseParams.softmaxLseOptional = EMPTY_TENSOR;
    } else {
        fBaseParams.softmaxLseOptional = NORMAL_TENSOR;
    }

    auto softmaxMaxShape = context_->GetOptionalInputShape(SOFTMAX_MAX_INDEX);
    auto softmaxSumShape = context_->GetOptionalInputShape(SOFTMAX_SUM_INDEX);
    if (softmaxMaxShape != nullptr && softmaxMaxShape->GetStorageShape().GetShapeSize() != 0 &&
        softmaxSumShape != nullptr && softmaxSumShape->GetStorageShape().GetShapeSize() != 0) {
        fBaseParams.softmaxLseOptional = EMPTY_TENSOR;
    } else {
        fBaseParams.softmaxLseOptional = NORMAL_TENSOR;
    }

    if (fBaseParams.s2 % S2_NZ_SIZE == 0) {
        fBaseParams.isNZOut = false;
    } else {
        fBaseParams.isNZOut = true;
    }

    fBaseParams.dataBlockNum = BYTE_BLOCK / fBaseParams.dataTypeSize;
    fBaseParams.calBlockNum = BYTE_BLOCK / fBaseParams.calTypeSize;

    fBaseParams.scaleValue = *(context_->GetAttrs()->GetAttrPointer<float>(ATTR_SCALE_VALUE_INDEX));

    // token_info
    fBaseParams.s1Token = *(context_->GetAttrs()->GetAttrPointer<int64_t>(ATTR_PRE_TOKENS_INDEX));
    fBaseParams.s2Token = *(context_->GetAttrs()->GetAttrPointer<int64_t>(ATTR_NEXT_TOKENS_INDEX));
    if (-fBaseParams.s1Token > int64_t(fBaseParams.s2) || -fBaseParams.s2Token > int64_t(fBaseParams.s1) ||
        (fBaseParams.s1Token + fBaseParams.s2Token) <= 0) {
        OP_LOGE(
            opName,
            "GetShapeAttrsInfo, pre_token and next_token is invalid, got s1 %u, s2 %u, pre_token %ld, next_token %ld",
            fBaseParams.s1, fBaseParams.s2, fBaseParams.s1Token, fBaseParams.s2Token);
        return ge::GRAPH_FAILED;
    }

    // 新增SPARSE_MODE属性，上库兼容处理
    auto attrs = context_->GetAttrs();
    fBaseParams.sparseMode = NO_MASK;
    if (attrs->GetAttrNum() > static_cast<size_t>(ATTR_SPARSE_MODE_INDEX)) {
        fBaseParams.sparseMode = *(attrs->GetAttrPointer<int>(ATTR_SPARSE_MODE_INDEX));
    }
    fBaseParams.attenMaskCompressMode = 0;
    auto attenMaskShape = context_->GetOptionalInputShape(ATTEN_MASK_INDEX);
    if (attenMaskShape == nullptr || attenMaskShape->GetStorageShape().GetShapeSize() == 0) {
        fBaseParams.attenMaskOptional = EMPTY_TENSOR;
    } else {
        fBaseParams.attenMaskOptional = NORMAL_TENSOR;
        auto storageShape = attenMaskShape->GetStorageShape();
        size_t dimNum = storageShape.GetDimNum();
        if (dimNum == ATTEN_MASK_DIM_LENGTH_2) {
            fBaseParams.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_SS;
        } else if (dimNum == ATTEN_MASK_DIM_LENGTH_4) {
            auto dim0 = attenMaskShape->GetStorageShape().GetDim(ATTEN_MASK_SHAPE_DIMS_0);
            auto dim1 = attenMaskShape->GetStorageShape().GetDim(ATTEN_MASK_SHAPE_DIMS_1);
            if ((dim0 == fBaseParams.b) && (dim1 == fBaseParams.n2 * fBaseParams.g)) {
                fBaseParams.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_BNSS;
            } else if ((dim0 == fBaseParams.b) && (dim1 == 1)) {
                fBaseParams.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_B1SS;
            } else if ((dim0 == 1) && (dim1 == 1)) {
                fBaseParams.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_SS;
            } else {
                OP_LOGE("FAG attenMask", "dim value error, dim0 = %ld, dim1 = %ld", dim0, dim1);
            }
        } else if (dimNum == ATTEN_MASK_DIM_LENGTH_1) {
            fBaseParams.attenMaskShapeType = ATTEN_MASK_SHAPE_TYPE_B1SS;
        } else {
            OP_LOGE("FAG attenMask", "dim num error, dimNum = %ld", dimNum);
        }
        if (dimNum >= 1) {
            fBaseParams.attenMaskS2Size = storageShape.GetDim(dimNum - LAST_AXIS_IDX);
            if (dimNum >= ATTEN_MASK_DIM_LENGTH_2) {
                fBaseParams.attenMaskS1Size = storageShape.GetDim(dimNum - SEC_LAST_AXIS_IDX);
            }
        }
        if (fBaseParams.sparseMode == LEFT_UP_CAUSAL) {
            fBaseParams.attenMaskCompressMode = LEFT_UP_CAUSAL_MODE;
        } else if (fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL) {
            fBaseParams.attenMaskCompressMode = RIGHT_DOWN_CAUSAL_MODE;
        } else if (fBaseParams.sparseMode == BAND) {
            fBaseParams.attenMaskCompressMode = BAND_EQUAL_S_MODE;
        }

        auto attenMask = context_->GetOptionalInputDesc(ATTEN_MASK_INDEX);
        if (attenMask != nullptr) {
            if (attenMask->GetDataType() == queryType) {
                fBaseParams.attenMaskDtype = ATTEN_MASK_TYPE_SAME;
            } else {
                fBaseParams.attenMaskDtype = ATTEN_MASK_TYPE_U8_BOOL;
            }
        }
    }

    auto prmMaskShape = context_->GetOptionalInputShape(PRM_MASK_INDEX);
    if (prmMaskShape != nullptr && prmMaskShape->GetStorageShape().GetDimNum() != 0) {
        fBaseParams.prmMaskOptional = 1;
    }

    auto sinkShape = context_->GetOptionalInputShape(SINK_IN_INDEX);
    if (sinkShape != nullptr && sinkShape->GetStorageShape().GetDimNum() != 0) {
        fBaseParams.sinkOptional = 1;
        fBaseParams.softmaxLseOptional = 0;
    }
    fBaseParams.sinkRequiresGrad = *(context_->GetAttrs()->GetAttrPointer<bool>(ATTR_SINK_REQUIRES_GRAD_INDEX));

    fBaseParams.isSparse = SetSparseParams();
    OP_LOGD(opName, "GetShapeAttrsInfo, FAG Us1s2Bbn2gs1s2 sparse mode = %u, sparse %s.", fBaseParams.sparseMode,
            fBaseParams.isSparse ? "enable" : "disable");
    OP_LOGD(opName,
            "GetShapeAttrsInfo, softmaxLse: %d, attenMask flag: %d, prmMask flag: %d, sink flag: %d, sinkRequiresGrad "
            "is %d",
            fBaseParams.softmaxLseOptional, fBaseParams.attenMaskOptional, fBaseParams.prmMaskOptional,
            fBaseParams.sinkOptional, fBaseParams.sinkRequiresGrad);

    if (CheckAttenMaskShape() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2::DoOpTiling()
{
    DoSplit();
    DoSparse();
    DoPreTiling();
    DoPostTiling();
    DetermineMode();

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bs2::DoSplit()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    // s1CvRatio set to 1, max tile s1 is 128
    fBaseParams.s1CvRatio = S1CV_RATIO_DEFAULT;
    fBaseParams.s2CvRatio = S2CV_RATIO_DEFAULT;
    // @lg: 这里先沿用dk
    if (fBaseParams.d == 64) { // d size is 64
        fBaseParams.s2CvRatio = CV_RATIO_16;
    }
    std::tuple<uint32_t, uint32_t, uint32_t> bestSplitRes = FuzzyForBestSplit();
    // s1 inner size, 每个core上每次处理vector 数据量
    uint32_t s1Inner = std::get<0>(bestSplitRes);
    // s1 inner size, 每个core上每次处理cube 数据量
    uint32_t s1CvInner = s1Inner * fBaseParams.s1CvRatio; // max is 128
    uint32_t s1Outer = (fBaseParams.s1 + s1CvInner - 1) / s1CvInner;
    uint32_t s1TailTmp = fBaseParams.s1 % s1Inner;
    uint32_t s1CvTailTmp = fBaseParams.s1 % s1CvInner;
    fBaseParams.s1Tail = s1TailTmp == 0 ? s1Inner : s1TailTmp;
    fBaseParams.s1CvTail = s1CvTailTmp == 0 ? s1CvInner : s1CvTailTmp;
    fBaseParams.s1Inner = s1Inner;
    fBaseParams.s1CvInner = s1CvInner;
    fBaseParams.s1Outer = s1Outer;

    uint32_t s2Inner = std::get<1>(bestSplitRes);
    uint32_t cvS2Inner = s2Inner * fBaseParams.s2CvRatio;
    uint32_t s2Outer = (fBaseParams.s2 + cvS2Inner - 1) / cvS2Inner;
    uint32_t s2TailTmp = fBaseParams.s2 % s2Inner;
    uint32_t s2Tail = s2TailTmp == 0 ? s2Inner : s2TailTmp;
    fBaseParams.s2Outer = s2Outer;
    fBaseParams.cvS2Inner = cvS2Inner;
    fBaseParams.s2Inner = s2Inner;
    fBaseParams.s2Tail = s2Tail;
    fBaseParams.baseMN = s1Inner * s2Inner;

    uint32_t sfmgdInner = std::get<2>(bestSplitRes);
    uint32_t sfmgdOuter = (fBaseParams.dV + sfmgdInner - 1) / sfmgdInner;
    uint32_t sfmgdTailTmp = fBaseParams.dV % sfmgdInner;
    uint32_t sfmgdTail = sfmgdTailTmp == 0 ? sfmgdInner : sfmgdTailTmp;
    fBaseParams.sfmgdOuter = sfmgdOuter;
    fBaseParams.sfmgdInner = sfmgdInner;
    fBaseParams.sfmgdTail = sfmgdTail;

    for (uint32_t i = 0; i < fBaseParams.b; i++) {
        uint32_t actualS1Len = fBaseParams.actualSeqQlen[i];
        fBaseParams.totalS1TileNums += (actualS1Len + fBaseParams.s1CvInner - 1) / fBaseParams.s1CvInner;
    }
}

void FlashAttentionScoreGradTilingUs1s2Bs2::DoSparse()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    if (fBaseParams.isSparse) {
        if (fBaseParams.layoutType == INPUT_FROAMT_TND) {
            GetSparseUnpadBlockInfo();
        } else {
            GetSparseBlockInfo();
        }
    } else {
        int64_t blockStarts[CORE_LIST_NUM];
        int64_t blockEnds[CORE_LIST_NUM];
        // block split
        int64_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.s1Outer * fBaseParams.s2Outer;
        int64_t blockFactor = (fusedOuter + fBaseParams.coreNum - 1) / fBaseParams.coreNum;
        int64_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

        fBaseParams.blockOuter = blockOuter;
        fBaseParams.blockFactor = blockFactor;

        for (int64_t i = 0; i < blockOuter; i++) {
            blockStarts[i] = blockFactor * i;
            blockEnds[i] = std::min(blockFactor * (i + 1), fusedOuter);
        }
        for (int64_t i = blockOuter; i < CORE_LIST_NUM; i++) {
            blockStarts[i] = 0;
            blockEnds[i] = 0;
        }

        std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
        std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));
    }
}

std::tuple<uint32_t, uint32_t, uint32_t> FlashAttentionScoreGradTilingUs1s2Bs2::FuzzyForBestSplit()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    uint32_t s1Inner = std::min<uint32_t>(INITIAL_S1_SPLIT_NUM, fBaseParams.s1Align);
    uint32_t s2Inner = std::min<uint32_t>(INITIAL_S2_SPLIT_NUM, fBaseParams.s2Align);

    bool left = true;
    while (!CheckFuzzyArgsLegal(s1Inner, s2Inner)) {
        if (left) {
            s1Inner = s1Inner - FRACTAL_NUM;
        } else {
            s2Inner = s2Inner - FRACTAL_NUM;
        }
        left = !left;
    }
    s2Inner = s2Inner > SOFT_MAX_PERF ? s2Inner / SOFT_MAX_PERF * SOFT_MAX_PERF : s2Inner;
    uint32_t first = s1Inner;
    uint32_t second = s2Inner;
    uint32_t tmpBufferSize =
        (fBaseParams.ubSize - first * second * basicBlockMultiple - first * SHAPE_INFO * fBaseParams.calTypeSize) /
        BYTE_BLOCK * BYTE_BLOCK;
    if (fBaseParams.isNZOut) {
        tmpBufferSize = tmpBufferSize - TEMP_BUFFER_REMAIN_SIZE;
    }

    fBaseParams.tmpBufferSize = tmpBufferSize;
    OP_LOGD(opName, "s1Inner = %d, s2Inner = %d, tmpBufferSize = %d", first, second, tmpBufferSize);

    // softmaxfront
    // init dV split factor use s2Inner
    uint32_t third = 0;
    uint32_t dInner = std::min(s2Inner, fBaseParams.dV);
    while (dInner > 0) {
        auto softmaxgradShape = ge::Shape({s1Inner, dInner});
        uint32_t softmaxgradTmpSize =
            AscendC::GetSoftMaxGradMinTmpSize(softmaxgradShape, fBaseParams.calTypeSize, true, false);
        if (fBaseParams.tmpBufferSize < softmaxgradTmpSize) {
            dInner -= FRACTAL_NUM;
        } else {
            third = dInner;
            break;
        }
    }

    third = third > SOFT_MAX_PERF ? third / SOFT_MAX_PERF * SOFT_MAX_PERF : third;
    return std::tie(std::min(first, 128u), std::min(second, 64u), std::min(third, 64u));
}

bool FlashAttentionScoreGradTilingUs1s2Bs2::CheckFuzzyArgsLegal(uint32_t s1Inner, uint32_t s2Inner)
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    OP_LOGD(opName, "Enter s1Inner = %d, s2Inner = %d", s1Inner, s2Inner);
    uint32_t baseMNSize = s1Inner * s2Inner * fBaseParams.calTypeSize;
    if (baseMNSize > fBaseParams.ubSize) {
        return false;
    }

    // simplesoftmax and dropout
    uint32_t cvS2Inner = s2Inner * fBaseParams.s2CvRatio;
    uint32_t s2VSize = cvS2Inner > 256 ? 256 : cvS2Inner;
    // ascend api attenmask and dropout last dim 32 align
    if (fBaseParams.attenMaskOptional == NORMAL_TENSOR) {
        s2VSize = (s2VSize + API_BOOL_ALIGN - 1) / API_BOOL_ALIGN * API_BOOL_ALIGN;
    }
    uint32_t s1VecSize = std::min(((INITIAL_S1_SPLIT_NUM * INITIAL_S2_SPLIT_NUM + s2VSize - 1) / s2VSize), s1Inner);

    auto softmaxShape = ge::Shape({s1VecSize, s2VSize});
    auto selectWithBytesMaskShape1 = ge::Shape({s1VecSize, s2VSize});
    auto selectWithBytesMaskShape2 =
        ge::Shape({s1VecSize, (s2VSize + BOOL_BLOCK_NUMS - 1) / BOOL_BLOCK_NUMS * BOOL_BLOCK_NUMS});
    uint32_t softmaxTmpSize = AscendC::GetSoftMaxMinTmpSize(softmaxShape, fBaseParams.calTypeSize, true);
    uint32_t selectWithBytesMaskTmpSize =
        AscendC::GetSelectWithBytesMaskMinTmpSize(selectWithBytesMaskShape1, ge::Shape({1}), fBaseParams.calTypeSize,
                                                  selectWithBytesMaskShape2, sizeof(uint8_t), true);
    uint32_t maxTmpBufferSize = softmaxTmpSize;
    maxTmpBufferSize = std::max(maxTmpBufferSize, selectWithBytesMaskTmpSize);
    if (ASCENDC_API_TEMP_BUFFER < maxTmpBufferSize) {
        return false;
    }

    // Loc buffer
    uint32_t bufferSizeL0c = baseMNSize;

    if (bufferSizeL0c <= fBaseParams.l0cSize) {
        return true;
    }
    return false;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2::DoLibApiTiling()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    // mm tiling
    matmul_tiling::MatmulApiTiling mm1;
    matmul_tiling::MatmulApiTiling mm2;
    matmul_tiling::MatmulApiTiling mm3;
    mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 false);
    mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 true);
    mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    uint32_t mm_s1 = fBaseParams.s1;
    if (fBaseParams.isNZOut) {
        mm_s1 = fBaseParams.s1Inner * fBaseParams.s1CvRatio;
    }
    // format left[B, N2, G, S1, D] right[B, N2, 1, S2, D] result[B, N2, G, S1, S2]
    if (fBaseParams.layoutType == INPUT_FROAMT_BN2GS2D || fBaseParams.layoutType == INPUT_FROAMT_AAG3BSH ||
        fBaseParams.layoutType == INPUT_FROAMT_AAG3SBH) {
        mm1.SetOrgShape(mm_s1, fBaseParams.s2, fBaseParams.d);
    } else if (fBaseParams.layoutType == INPUT_FROAMT_S2BN2GD) {
        mm1.SetOrgShape(mm_s1, fBaseParams.s2, fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.d,
                        fBaseParams.b * fBaseParams.n2 * fBaseParams.d);
    } else if (fBaseParams.layoutType == INPUT_FROAMT_BS2N2GD) {
        mm1.SetOrgShape(mm_s1, fBaseParams.s2, fBaseParams.n2 * fBaseParams.g * fBaseParams.d,
                        fBaseParams.n2 * fBaseParams.d);
    }
    mm1.SetShape(fBaseParams.s1Inner * fBaseParams.s1CvRatio, fBaseParams.s2Inner * fBaseParams.s2CvRatio,
                 fBaseParams.d);
    mm1.SetBias(false);
    if (fBaseParams.cvS2Inner > 128) {                       // 128 for perf when s2 cv ratio
        if (fBaseParams.d > 64) {                            // 64 for d
            mm1.SetFixSplit(fBaseParams.s1CvInner, 128, -1); // 128 for baseN
        } else {
            mm1.SetFixSplit(fBaseParams.s1CvInner, 256, -1); // 256 for baseN
        }
    } else {
        mm1.SetFixSplit(-1, -1, -1);
    }
    mm1.GetTiling(tilingData.mm1TilingData);
    SetMatmulTilingBufferInfo(tilingData.mm1TilingData);

    // format left[B, N2, G, S1, S2] right[B, N2, G, S1, D] result[B, N2, G, S2, D]
    // matmal3/5
    mm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 true);
    mm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 false);
    mm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    if (fBaseParams.layoutType == INPUT_FROAMT_BN2GS2D || fBaseParams.layoutType == INPUT_FROAMT_AAG3BSH ||
        fBaseParams.layoutType == INPUT_FROAMT_AAG3SBH) {
        // M/N/K
        mm2.SetOrgShape(fBaseParams.s2, fBaseParams.d, fBaseParams.s1);
    } else if (fBaseParams.layoutType == INPUT_FROAMT_S2BN2GD) {
        mm2.SetOrgShape(fBaseParams.s2, fBaseParams.b * fBaseParams.n2 * fBaseParams.g * fBaseParams.d, fBaseParams.s1,
                        fBaseParams.s1);
    } else if (fBaseParams.layoutType == INPUT_FROAMT_BS2N2GD) {
        mm2.SetOrgShape(fBaseParams.s2, fBaseParams.n2 * fBaseParams.g * fBaseParams.d, fBaseParams.s1, fBaseParams.s1);
    }
    mm2.SetShape(fBaseParams.s2Inner * fBaseParams.s2CvRatio, fBaseParams.d,
                 fBaseParams.s1Inner * fBaseParams.s1CvRatio);
    mm2.SetBias(false);
    mm2.SetFixSplit(-1, -1, -1);
    mm2.GetTiling(tilingData.mm2TilingData);
    SetMatmulTilingBufferInfo(tilingData.mm2TilingData);

    // format left[B, N2, G, S1, S2] right[B, N2, 1, S2, D] result[B, N2, G, S1, D]
    // matmal4
    mm3.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 false);
    mm3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_BFLOAT16,
                 false);
    mm3.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
    if (fBaseParams.layoutType == INPUT_FROAMT_BN2GS2D || fBaseParams.layoutType == INPUT_FROAMT_AAG3BSH ||
        fBaseParams.layoutType == INPUT_FROAMT_AAG3SBH) {
        // M/N/K
        mm3.SetOrgShape(fBaseParams.s1, fBaseParams.d, fBaseParams.s2);
    } else if (fBaseParams.layoutType == INPUT_FROAMT_S2BN2GD) {
        mm3.SetOrgShape(fBaseParams.s1, fBaseParams.b * fBaseParams.n2 * fBaseParams.d, fBaseParams.s2, fBaseParams.s2);
    } else if (fBaseParams.layoutType == INPUT_FROAMT_BS2N2GD) {
        mm3.SetOrgShape(fBaseParams.s1, fBaseParams.n2 * fBaseParams.d, fBaseParams.s2, fBaseParams.s2);
    }
    mm3.SetShape(fBaseParams.s1Inner * fBaseParams.s1CvRatio, fBaseParams.d,
                 fBaseParams.s2Inner * fBaseParams.s2CvRatio);
    mm3.SetBias(false);
    mm3.SetFixSplit(-1, -1, -1);
    mm3.GetTiling(tilingData.mm3TilingData);
    SetMatmulTilingBufferInfo(tilingData.mm3TilingData);

    uint32_t cvS2Inner = fBaseParams.s2Inner * fBaseParams.s2CvRatio;
    uint32_t s2VSize = cvS2Inner > 256 ? 256 : cvS2Inner;
    uint32_t s1VecSize =
        std::min(((INITIAL_S1_SPLIT_NUM * INITIAL_S2_SPLIT_NUM + s2VSize - 1) / s2VSize), fBaseParams.s1Inner);

    auto softmaxShape = ge::Shape({s1VecSize, s2VSize});
    AscendC::SoftMaxTilingFunc(softmaxShape, fBaseParams.calTypeSize, fBaseParams.tmpBufferSize,
                               tilingData.softmaxTilingData);
    AscendC::SoftMaxGradTilingFunc(softmaxShape, fBaseParams.calTypeSize, fBaseParams.tmpBufferSize,
                                   tilingData.softmaxGradTilingData, true);

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bs2::SetMatmulTilingBufferInfo(TCubeTiling& mmTiling)
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    mmTiling.set_shareMode(0);
    mmTiling.set_shareL1Size(fBaseParams.l1Size);
    mmTiling.set_shareL0CSize(fBaseParams.l0cSize);
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2::GetWorkspaceSize()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    size_t workspaceSize = MUL_CORE_SYNC_BUFFER;
    // matmal3 q
    workspaceSize = (workspaceSize + fBaseParams.qSize * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 k
    workspaceSize = (workspaceSize + fBaseParams.kSize * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // matmal3 v
    workspaceSize = (workspaceSize + fBaseParams.vSize * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;
    // dsik workspace
    workspaceSize =
        (workspaceSize + fBaseParams.n1 * fBaseParams.totalS1TileNums * FP32_BYTES + GM_ALIGN) / GM_ALIGN * GM_ALIGN;

    // matmal1/matmal2 workspace size
    size_t vectorCoreNum = fBaseParams.coreNum;
    workspaceSize = (workspaceSize +
                     vectorCoreNum * fBaseParams.s1CvRatio * fBaseParams.s2CvRatio * fBaseParams.baseMN * FP32_BYTES *
                         MATMAL_INPUT_NUMS +
                     GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;
    // CV ratio workspace size fp16
    // mask workspace
    workspaceSize = (workspaceSize +
                     vectorCoreNum * fBaseParams.s1CvRatio * fBaseParams.s2CvRatio * fBaseParams.baseMN * FP16_BYTES *
                         2 // 2 means pingpong
                     + GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;
    // mul workspace size
    workspaceSize = (workspaceSize +
                     vectorCoreNum * fBaseParams.s1CvRatio * fBaseParams.s2CvRatio * fBaseParams.baseMN * FP16_BYTES *
                         2 // 2 means pingpong
                     + GM_ALIGN) /
                    GM_ALIGN * GM_ALIGN;

    workspaceSize = workspaceSize + WORKSPACE_BUFFER;
    workspaces[0] = workspaceSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2::PostTiling()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    SaveToTilingData();
    auto blockdim = CalcTschBlockDim(tilingData.s1s2BNGS1S2SplitCoreParams.get_blockOuter(), fBaseParams.aicNum,
                                     fBaseParams.coreNum);
    OP_CHECK_IF(blockdim == 0,
                TILING_INNER_ERR_REPORT(context_->GetNodeName(), "blockdim is 0, aicNum is %ld, aivNum is %ld.",
                                        fBaseParams.aicNum, fBaseParams.coreNum),
                return ge::GRAPH_FAILED);
    context_->SetBlockDim(blockdim);
    context_->SetScheduleMode(1); // 1: batchmode模式

    tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2::SaveToTilingData()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    tilingData.s1s2BNGS1S2BaseParams.set_coreNum(fBaseParams.coreNum);
    // set tilingdata baseinfo
    tilingData.s1s2BNGS1S2BaseParams.set_b(fBaseParams.b);
    tilingData.s1s2BNGS1S2BaseParams.set_n2(fBaseParams.n2);
    tilingData.s1s2BNGS1S2BaseParams.set_g(fBaseParams.g);
    tilingData.s1s2BNGS1S2BaseParams.set_s1(fBaseParams.s1);
    tilingData.s1s2BNGS1S2BaseParams.set_d(fBaseParams.d);
    tilingData.s1s2BNGS1S2BaseParams.set_dV(fBaseParams.dV);
    tilingData.s1s2BNGS1S2BaseParams.set_s2(fBaseParams.s2);

    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskOptional(fBaseParams.attenMaskOptional);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskShapeType(fBaseParams.attenMaskShapeType);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskDtype(fBaseParams.attenMaskDtype);
    tilingData.s1s2BNGS1S2BaseParams.set_layout(fBaseParams.layoutType);
    tilingData.s1s2BNGS1S2BaseParams.set_scaleValue(fBaseParams.scaleValue);
    tilingData.s1s2BNGS1S2BaseParams.set_s1Token(fBaseParams.s1Token);
    tilingData.s1s2BNGS1S2BaseParams.set_s2Token(fBaseParams.s2Token);
    tilingData.s1s2BNGS1S2BaseParams.set_sparseMode(fBaseParams.sparseMode);
    tilingData.s1s2BNGS1S2BaseParams.set_isSparse(fBaseParams.isSparse);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskS2Size(fBaseParams.attenMaskS2Size);
    tilingData.s1s2BNGS1S2BaseParams.set_attenMaskCompressMode(fBaseParams.attenMaskCompressMode);
    tilingData.s1s2BNGS1S2BaseParams.set_softmaxLseOptional(fBaseParams.softmaxLseOptional);
    tilingData.s1s2BNGS1S2BaseParams.set_prmMaskOptional(fBaseParams.prmMaskOptional);
    tilingData.s1s2BNGS1S2BaseParams.set_sinkOptional(fBaseParams.sinkOptional);
    tilingData.s1s2BNGS1S2BaseParams.set_sinkRequiresGrad(fBaseParams.sinkRequiresGrad);

    // s1/s2 split
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1CvRatio(fBaseParams.s1CvRatio);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1Outer(fBaseParams.s1Outer);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1Inner(fBaseParams.s1Inner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1CvInner(fBaseParams.s1CvInner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1Tail(fBaseParams.s1Tail);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s1CvTail(fBaseParams.s1CvTail);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2Outer(fBaseParams.s2Outer);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2CvRatio(fBaseParams.s2CvRatio);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2Inner(fBaseParams.s2Inner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_s2Tail(fBaseParams.s2Tail);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_sfmgdOuter(fBaseParams.sfmgdOuter);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_sfmgdFactor(fBaseParams.sfmgdInner);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_sfmgdTail(fBaseParams.sfmgdTail);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_totalS1TileNums(fBaseParams.totalS1TileNums);

    tilingData.s1s2BNGS1S2SplitCoreParams.set_baseMN(fBaseParams.baseMN);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_tmpBufferSize(fBaseParams.tmpBufferSize);
    tilingData.s1s2BNGS1S2BlockNumList.set_blockStarts(fBaseParams.blockStarts);
    tilingData.s1s2BNGS1S2BlockNumList.set_blockEnds(fBaseParams.blockEnds);
    tilingData.s1s2BNGS1S2SplitCoreParams.set_blockOuter(fBaseParams.blockOuter);
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bs2::GetParseS1S2OuterInfo(uint32_t (*parseInfo)[ARRAY_LENGTH])
{
    for (uint32_t i = 0; i < fBaseParams.s2Outer; i++) {
        parseInfo[i][BEGIN_IDX] =
            uint32_t(std::min(std::max(0L, int64_t(fBaseParams.cvS2Inner * i) - fBaseParams.s2Token),
                              int64_t(fBaseParams.s1))) /
            fBaseParams.s1Inner;
        parseInfo[i][END_IDX] =
            uint32_t(std::min(std::max(0L, int64_t(fBaseParams.cvS2Inner * (i + 1)) + fBaseParams.s1Token),
                              int64_t(fBaseParams.s1)) +
                     fBaseParams.s1Inner - 1) /
            fBaseParams.s1Inner;
        uint32_t tmpSize =
            (parseInfo[i][END_IDX] > parseInfo[i][BEGIN_IDX]) ? parseInfo[i][END_IDX] - parseInfo[i][BEGIN_IDX] : 0;
        if (i == 0) {
            parseInfo[i][LENGTH_IDX] = tmpSize;
        } else {
            parseInfo[i][LENGTH_IDX] = parseInfo[i - 1][LENGTH_IDX] + tmpSize;
        }
        OP_LOGD(opName, "GetParseS1S2OuterInfo, idx = %d: Begin = %d, End = %d, Length = %d, total_Length = %d", i,
                parseInfo[i][0], parseInfo[i][1], tmpSize, parseInfo[i][LENGTH_IDX]);
    }
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2::GetSparseBlockInfo()
{
    // 自动校正left和right causal的token值，token信息仅用于sparse分核计算
    if (fBaseParams.sparseMode == LEFT_UP_CAUSAL || fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL) {
        fBaseParams.s1Token = INT32_MAX;
        fBaseParams.s2Token = 0;
    }

    // sparse_mode =4 (band)时 或者sparse_mode ==3 (RIGHT_DOWN_CAUSAL) 时，token以右下角为基准，需要校正
    if (fBaseParams.sparseMode == RIGHT_DOWN_CAUSAL || fBaseParams.sparseMode == BAND) {
        fBaseParams.s1Token = fBaseParams.s1Token + fBaseParams.s1 - fBaseParams.s2;
        fBaseParams.s2Token = fBaseParams.s2Token - fBaseParams.s1 + fBaseParams.s2;
    }

    OP_LOGD(opName, "GetSparseBlockInfo, the corrected s1Token = %ld, s2Token %ld.", fBaseParams.s1Token,
            fBaseParams.s2Token);

    // [s2OuterIdx][begin, end, length]
    uint32_t(*parseInfo)[ARRAY_LENGTH] = new uint32_t[fBaseParams.s2Outer][ARRAY_LENGTH];
    GetParseS1S2OuterInfo(parseInfo);
    uint32_t s1s2oCount = parseInfo[fBaseParams.s2Outer - 1][LENGTH_IDX];

    // block split
    uint32_t fusedOuter = fBaseParams.b * fBaseParams.n2 * fBaseParams.g * s1s2oCount;
    uint32_t blockFactor = (fusedOuter + fBaseParams.coreNum - 1) / fBaseParams.coreNum;
    uint32_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;
    uint32_t blockTailTmp = fusedOuter % blockFactor;
    uint32_t blockTail = blockTailTmp == 0 ? blockFactor : blockTailTmp;
    OP_LOGD(opName, "GetSparseBlockInfo, Sparse parseInfo fusedOuter = %d: blockFactor = %d, blockTail = %d",
            fusedOuter, blockFactor, blockTail);
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;

    uint32_t bIdx = 0;
    uint32_t bTail = 0;
    uint32_t n2Idx = 0;
    uint32_t n2Tail = 0;
    uint32_t gIdx = 0;
    uint32_t gTail = 0;
    int64_t s1oIdx = 0;
    int64_t s2oIdx = 0;

    uint32_t n2gs1s2o = fBaseParams.n2 * fBaseParams.g * s1s2oCount;
    uint32_t gs1s2o = fBaseParams.g * s1s2oCount;

    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    blockStarts[0] = 0;
    blockEnds[blockOuter - 1] = static_cast<int64_t>(fBaseParams.b) * fBaseParams.n2 * fBaseParams.g *
                                fBaseParams.s1Outer * fBaseParams.s2Outer;
    for (uint32_t c = 1; c < blockOuter; c++) {
        // cal indx for total bngs1os2o(sparse)
        uint32_t currentIdx = std::min(c * blockFactor, fusedOuter);
        bIdx = currentIdx / n2gs1s2o;
        bTail = currentIdx % n2gs1s2o;
        n2Idx = bTail / gs1s2o;
        n2Tail = bTail % gs1s2o;
        gIdx = n2Tail / s1s2oCount;
        gTail = n2Tail % s1s2oCount;

        OP_LOGD(opName,
                "GetSparseBlockInfo, Sparse parseInfo currentIdx = %d: bIdx = %d, bTail = %d, n2Idx = %d, n2Tail = %d, "
                "gIdx = %d, gTail = %d",
                currentIdx, bIdx, bTail, n2Idx, n2Tail, gIdx, gTail);
        uint32_t preSize = 0;
        uint32_t nextSize = 0;
        for (uint32_t i = 0; i < fBaseParams.s2Outer; i++) {
            if (gTail >= preSize) {
                nextSize = parseInfo[i][LENGTH_IDX];
                if (gTail < nextSize) {
                    s2oIdx = i;
                    s1oIdx = parseInfo[i][BEGIN_IDX] + gTail - preSize - 1;
                    OP_LOGD(opName, "GetSparseBlockInfo, s1oIdx = %d, s2oIdx = %d, preSize = %d, nextSize = %d", s1oIdx,
                            s2oIdx, preSize, nextSize);
                    break;
                }
                preSize = parseInfo[i][LENGTH_IDX];
            }
        }

        // total indx in bngs1os2o (range is [))
        blockStarts[c] = (((bIdx * fBaseParams.n2 + n2Idx) * fBaseParams.g + gIdx) * fBaseParams.s2Outer + s2oIdx) *
                             fBaseParams.s1Outer +
                         s1oIdx + 1;
        blockEnds[c - 1] = blockStarts[c];
        OP_LOGD(opName, "GetSparseBlockInfo, blockStarts[c] = %d:", blockStarts[c]);
    }
    for (uint32_t c = blockOuter; c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

    // free tensor
    delete[] parseInfo;
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bs2::FillBlockInfo(
    std::vector<std::vector<std::vector<uint32_t>>>& calculatedBlockInfo,
    std::vector<std::vector<uint32_t>>& totalBlockInfo)
{
    for (uint32_t i = 0; i < fBaseParams.b; i++) {
        uint32_t actualS1Len = fBaseParams.actualSeqQlen[i];
        uint32_t actualS2Len = fBaseParams.actualSeqKvlen[i];

        auto actualS1Outer = (actualS1Len + fBaseParams.s1Inner - 1) / fBaseParams.s1Inner;
        auto actualS2Outer = (actualS2Len + fBaseParams.cvS2Inner - 1) / fBaseParams.cvS2Inner;
        totalBlockInfo[i][0] = actualS1Outer * actualS2Outer;

        calculatedBlockInfo[i][0][BEGIN_IDX] = 0; // BEGIN_IDX 行的有效起始位置
        calculatedBlockInfo[i][0][END_IDX] =      // END_IDX 行的有效终止位置
            uint32_t(std::min(int64_t(actualS1Len), std::max(0L, fBaseParams.cvS2Inner + fBaseParams.s1Token)) +
                     fBaseParams.s1Inner - 1) /
            fBaseParams.s1Inner;
        calculatedBlockInfo[i][0][SUM_S1S2] =
            calculatedBlockInfo[i][0][END_IDX]; // SUM_S1S2 当前batch上， 列的有效块总和

        // unpad 场景下s2Outer是按照最大的s2计算得到的
        for (uint32_t j = 1; j < fBaseParams.s2Outer; j++) {
            if (fBaseParams.cvS2Inner * j >= actualS2Len) {
                calculatedBlockInfo[i][j][BEGIN_IDX] = 0;
                calculatedBlockInfo[i][j][END_IDX] = 0;
            } else {
                calculatedBlockInfo[i][j][BEGIN_IDX] =
                    uint32_t(std::min(std::max(int64_t(fBaseParams.cvS2Inner * j) - fBaseParams.s2Token, 0L),
                                      int64_t(actualS1Len))) /
                    fBaseParams.s1Inner;
                calculatedBlockInfo[i][j][END_IDX] =
                    uint32_t(std::min(int64_t(actualS1Len),
                                      std::max(int64_t(fBaseParams.cvS2Inner * (j + 1)) + fBaseParams.s1Token, 0L)) +
                             fBaseParams.s1Inner - 1) /
                    fBaseParams.s1Inner;
            }
            calculatedBlockInfo[i][j][SUM_S1S2] = calculatedBlockInfo[i][j - 1][SUM_S1S2] +
                                                  calculatedBlockInfo[i][j][END_IDX] -
                                                  calculatedBlockInfo[i][j][BEGIN_IDX];
            calculatedBlockInfo[i][j][SUM_ALL] = 0; // 初始化清零
        }

        if (i == 0) {
            calculatedBlockInfo[0][0][SUM_ALL] = // SUM_ALL 为从0到当前batch的所有有效块总和
                fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[0][fBaseParams.s2Outer - 1][SUM_S1S2];
            totalBlockInfo[0][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[0][0];
        } else {
            calculatedBlockInfo[i][0][SUM_ALL] =
                fBaseParams.n2 * fBaseParams.g * calculatedBlockInfo[i][fBaseParams.s2Outer - 1][SUM_S1S2] +
                calculatedBlockInfo[i - 1][0][SUM_ALL];
            totalBlockInfo[i][1] = fBaseParams.n2 * fBaseParams.g * totalBlockInfo[i][0] + totalBlockInfo[i - 1][1];
        }
    }
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2::GetSparseUnpadBlockInfo()
{
    std::vector<std::vector<std::vector<uint32_t>>> calculatedBlockInfo(
        fBaseParams.b,
        std::vector<std::vector<uint32_t>>(fBaseParams.s2Outer, std::vector<uint32_t>(CALCULATED_BLOCK_DIMENSION)));
    std::vector<std::vector<uint32_t>> totalBlockInfo(fBaseParams.b, std::vector<uint32_t>(TOTAL_BLOCK_DIMENSION));
    FillBlockInfo(calculatedBlockInfo, totalBlockInfo);

    // block split
    uint32_t fusedOuter = calculatedBlockInfo[fBaseParams.b - 1][0][SUM_ALL];
    uint32_t blockFactor = (fusedOuter + fBaseParams.coreNum - 1) / fBaseParams.coreNum;
    uint32_t blockOuter = (fusedOuter + blockFactor - 1) / blockFactor;

    OP_LOGD(opName, "GetSparseUnpadBlockInfo, fusedOuter = %u: blockFactor = %u, blockOuter = %u", fusedOuter,
            blockFactor, blockOuter);
    fBaseParams.blockOuter = blockOuter;
    fBaseParams.blockFactor = blockFactor;

    uint32_t bIdx = 0;
    uint32_t bTail = 0;
    uint32_t n2Idx = 0;
    uint32_t n2Tail = 0;
    uint32_t gIdx = 0;
    uint32_t gTail = 0;
    int64_t s1oIdx = 0;
    int64_t s1oTail = 0;
    int64_t s2oIdx = 0;

    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    blockStarts[0] = 0;
    blockEnds[blockOuter - 1] = totalBlockInfo[fBaseParams.b - 1][1];

    uint32_t s1OuterTmp = 0;

    for (uint32_t c = 1; c < blockOuter; c++) {
        uint32_t currentIdx = std::min(c * blockFactor, fusedOuter);

        for (uint32_t b = 0; b < fBaseParams.b; b++) {
            if (calculatedBlockInfo[b][0][SUM_ALL] >= currentIdx) {
                bIdx = b;
                auto s1os2o = calculatedBlockInfo[b][fBaseParams.s2Outer - 1][SUM_S1S2];
                auto gs1os2o = s1os2o * fBaseParams.g;
                bTail = (b == 0) ? currentIdx : currentIdx - calculatedBlockInfo[b - 1][0][SUM_ALL];
                n2Idx = bTail / gs1os2o;
                n2Tail = bTail % gs1os2o;
                gIdx = n2Tail / s1os2o;
                gTail = n2Tail % s1os2o;
                for (uint32_t i = 0; i < fBaseParams.s2Outer; i++) {
                    if (calculatedBlockInfo[b][i][SUM_S1S2] >= gTail) {
                        s2oIdx = i;
                        s1oTail = (i == 0) ? gTail : gTail - calculatedBlockInfo[b][i - 1][SUM_S1S2];
                        s1oIdx = calculatedBlockInfo[b][i][BEGIN_IDX] + s1oTail;
                        break;
                    }
                }
                s1OuterTmp = (fBaseParams.actualSeqQlen[b] + fBaseParams.s1Inner - 1) / fBaseParams.s1Inner;
                break;
            }
        }
        if (bIdx == 0) {
            blockStarts[c] = (n2Idx * fBaseParams.g + gIdx) * totalBlockInfo[bIdx][0] + s2oIdx * s1OuterTmp + s1oIdx;
        } else {
            blockStarts[c] = totalBlockInfo[bIdx - 1][1] + (n2Idx * fBaseParams.g + gIdx) * totalBlockInfo[bIdx][0] +
                             s2oIdx * s1OuterTmp + s1oIdx;
        }
        blockEnds[c - 1] = blockStarts[c];
    }
    for (uint32_t c = blockOuter; c < CORE_LIST_NUM; c++) {
        blockStarts[c] = 0;
        blockEnds[c] = 0;
    }
    std::copy(std::begin(blockStarts), std::end(blockStarts), std::begin(fBaseParams.blockStarts));
    std::copy(std::begin(blockEnds), std::end(blockEnds), std::begin(fBaseParams.blockEnds));

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreGradTilingUs1s2Bs2::DoPreTiling()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    int64_t qPreBlockFactor = (fBaseParams.qSize + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;
    int64_t qPreBlockTotal = (fBaseParams.qSize + qPreBlockFactor - 1) / qPreBlockFactor;
    int64_t qPreTailNumTmp = fBaseParams.qSize % qPreBlockFactor;
    int64_t qPreTailNum = qPreTailNumTmp == 0 ? qPreBlockFactor : qPreTailNumTmp;

    int64_t kPreBlockFactor = (fBaseParams.kSize + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;
    int64_t kPreBlockTotal = (fBaseParams.kSize + kPreBlockFactor - 1) / kPreBlockFactor;
    int64_t kPreTailNumTmp = fBaseParams.kSize % kPreBlockFactor;
    int64_t kPreTailNum = kPreTailNumTmp == 0 ? kPreBlockFactor : kPreTailNumTmp;

    int64_t vPreBlockFactor = (fBaseParams.vSize + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;
    int64_t vPreBlockTotal = (fBaseParams.vSize + vPreBlockFactor - 1) / vPreBlockFactor;
    int64_t vPreTailNumTmp = fBaseParams.vSize % vPreBlockFactor;
    int64_t vPreTailNum = vPreTailNumTmp == 0 ? vPreBlockFactor : vPreTailNumTmp;

    int64_t sinkSize = fBaseParams.n1 * fBaseParams.totalS1TileNums;
    int64_t sinkPreBlockFactor = (sinkSize + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;
    int64_t sinkPreBlockTotal = (sinkSize + sinkPreBlockFactor - 1) / sinkPreBlockFactor;
    int64_t sinkPreTailNumTmp = sinkSize % sinkPreBlockFactor;
    int64_t sinkPreTailNum = sinkPreTailNumTmp == 0 ? sinkPreBlockFactor : sinkPreTailNumTmp;

    tilingData.preTilingData.set_qPreBlockFactor(qPreBlockFactor);
    tilingData.preTilingData.set_qPreBlockTotal(qPreBlockTotal);
    tilingData.preTilingData.set_qPreBlockTail(qPreTailNum);
    tilingData.preTilingData.set_kPreBlockFactor(kPreBlockFactor);
    tilingData.preTilingData.set_kPreBlockTotal(kPreBlockTotal);
    tilingData.preTilingData.set_kPreBlockTail(kPreTailNum);
    tilingData.preTilingData.set_vPreBlockFactor(vPreBlockFactor);
    tilingData.preTilingData.set_vPreBlockTotal(vPreBlockTotal);
    tilingData.preTilingData.set_vPreBlockTail(vPreTailNum);
    tilingData.preTilingData.set_sinkPreBlockFactor(sinkPreBlockFactor);
    tilingData.preTilingData.set_sinkPreBlockTotal(sinkPreBlockTotal);
    tilingData.preTilingData.set_sinkPreBlockTail(sinkPreTailNum);
    tilingData.preTilingData.set_dsinkSize(sinkSize);
}

void FlashAttentionScoreGradTilingUs1s2Bs2::DoPostTiling()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    uint32_t postUbBaseSize = (fBaseParams.ubSize) / postCoexNode / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN;
    uint32_t qPostBaseNum = postUbBaseSize / FP16_BYTES;
    uint64_t qPostBlockTotal = fBaseParams.qSize;
    uint64_t qSizeAlign = (qPostBlockTotal + WORKSPACE_NUM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * FP16_BYTES;
    uint32_t qPostTailNumTmp = qPostBlockTotal % qPostBaseNum;
    uint32_t qPostTailNum = qPostTailNumTmp == 0 ? qPostBaseNum : qPostTailNumTmp;
    uint32_t qPostBlockOuterTotal = (qPostBlockTotal + qPostBaseNum - 1) / qPostBaseNum;
    uint32_t qPostBlockFactor = (qPostBlockOuterTotal + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;

    uint32_t kPostBaseNum = postUbBaseSize / FP16_BYTES;
    uint64_t kPostBlockTotal = fBaseParams.kSize;
    uint64_t kSizeAlign = (kPostBlockTotal + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * FP16_BYTES;
    uint32_t kPostTailNumTmp = kPostBlockTotal % kPostBaseNum;
    uint32_t kPostTailNum = kPostTailNumTmp == 0 ? kPostBaseNum : kPostTailNumTmp;
    uint32_t kPostBlockOuterTotal = (kPostBlockTotal + kPostBaseNum - 1) / kPostBaseNum;
    uint32_t kPostBlockFactor = (kPostBlockOuterTotal + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;

    uint32_t vPostBaseNum = postUbBaseSize / FP16_BYTES;
    uint64_t vPostBlockTotal = fBaseParams.vSize;
    uint64_t vSizeAlign = (vPostBlockTotal + GM_ALIGN - 1) / GM_ALIGN * GM_ALIGN * FP16_BYTES;
    uint32_t vPostTailNumTmp = vPostBlockTotal % vPostBaseNum;
    uint32_t vPostTailNum = vPostTailNumTmp == 0 ? vPostBaseNum : vPostTailNumTmp;
    uint32_t vPostBlockOuterTotal = (vPostBlockTotal + vPostBaseNum - 1) / vPostBaseNum;
    uint32_t vPostBlockFactor = (vPostBlockOuterTotal + fBaseParams.blockOuter - 1) / fBaseParams.blockOuter;

    tilingData.postTilingData.set_postUbBaseSize(postUbBaseSize);
    tilingData.postTilingData.set_qPostBlockFactor(qPostBlockFactor);
    tilingData.postTilingData.set_qPostBaseNum(qPostBaseNum);
    tilingData.postTilingData.set_qPostTailNum(qPostTailNum);
    tilingData.postTilingData.set_qPostBlockTotal(qPostBlockTotal);
    tilingData.postTilingData.set_qSizeAlign(qSizeAlign);

    tilingData.postTilingData.set_kPostBlockFactor(kPostBlockFactor);
    tilingData.postTilingData.set_kPostBlockTotal(kPostBlockTotal);
    tilingData.postTilingData.set_kPostBaseNum(kPostBaseNum);
    tilingData.postTilingData.set_kPostTailNum(kPostTailNum);
    tilingData.postTilingData.set_kSizeAlign(kSizeAlign);

    tilingData.postTilingData.set_vPostBlockFactor(vPostBlockFactor);
    tilingData.postTilingData.set_vPostBlockTotal(vPostBlockTotal);
    tilingData.postTilingData.set_vPostBaseNum(vPostBaseNum);
    tilingData.postTilingData.set_vPostTailNum(vPostTailNum);
    tilingData.postTilingData.set_vSizeAlign(vSizeAlign);

    uint32_t bufferNums = 5;
    uint32_t dsinkBaseSize = (fBaseParams.ubSize) / bufferNums / WORKSPACE_NUM_ALIGN * WORKSPACE_NUM_ALIGN;
    tilingData.postTilingData.set_dsinkBaseSize(dsinkBaseSize);
}

void FlashAttentionScoreGradTilingUs1s2Bs2::DetermineMode()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    // 当前fp16都走高精度
    if (fBaseParams.queryType == ge::DT_FLOAT) {
        fBaseParams.mode = FP32;
    } else if (fBaseParams.queryType == ge::DT_BF16) {
        fBaseParams.mode = BF16;
    } else if (fBaseParams.queryType == ge::DT_FLOAT16) {
        fBaseParams.mode = INHP;
    } else {
        fBaseParams.mode = FP16;
    }
}

ge::graphStatus FlashAttentionScoreGradTilingUs1s2Bs2::CheckAttenMaskShape()
{
    FuzzyBaseInfoParams& fBaseParams = GetBaseParams();
    // check atten_mask shape when enable atten_mask_compress
    if (fBaseParams.attenMaskCompressMode != 0) {
        if (fBaseParams.attenMaskS1Size != fBaseParams.attenMaskS2Size) {
            OP_LOGE("Atten Mask Compress", "atten mask shape is not square.");
            return ge::GRAPH_FAILED;
        }

        if (fBaseParams.attenMaskS2Size != ATTEN_MASK_COMPRESS_LIMIT) {
            OP_LOGE("Atten Mask Compress", "atten mask shape is invalid, try setting it to [2048, 2048].");
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

REGISTER_TILING_TEMPLATE("FlashAttentionGrad", FlashAttentionScoreGradTilingUs1s2Bs2, 16000);

} // namespace optiling
