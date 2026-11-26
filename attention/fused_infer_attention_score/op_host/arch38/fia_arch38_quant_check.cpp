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

bool FiaArch38PromptTiling::CheckPostQuantDtype(const FiaArch38ContextParams& contextKeyParams,
                                                const gert::StorageShape* quantOffset2Shape,
                                                const ge::DataType quantScale2Type) const
{
    OP_CHECK_IF(
        (quantOffset2Shape != nullptr) && (quantScale2Type != contextKeyParams.quantOffset2Type),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            contextKeyParams.opName, "quant_scale2, quant_offset2",
            (GetPfaDataTypeStr(quantScale2Type) + ", " + GetPfaDataTypeStr(contextKeyParams.quantOffset2Type)).c_str(),
            "post quant scale and offset must have the same data type"),
        return false);

    OP_CHECK_IF((quantScale2Type != ge::DT_BF16) && (quantScale2Type != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "quant_scale2",
                                                      GetPfaDataTypeStr(quantScale2Type).c_str(),
                                                      "post quant scale dtype only supports bf16 and fp32"),
                return false);

    if (inputType == ge::DT_BF16) {
        OP_CHECK_IF(quantScale2Type != ge::DT_FLOAT && quantScale2Type != ge::DT_BF16,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        contextKeyParams.opName, "quant_scale2", GetPfaDataTypeStr(quantScale2Type).c_str(),
                        "post quant scale dtype only supports float32 and bf16 for this query dtype"),
                    return false);
    } else {
        OP_CHECK_IF(quantScale2Type != ge::DT_FLOAT,
                    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                        contextKeyParams.opName, "quant_scale2", GetPfaDataTypeStr(quantScale2Type).c_str(),
                        "post quant scale dtype only supports float32 for this query dtype"),
                    return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPostQuantShapeSize(const FiaArch38ContextParams& contextKeyParams,
                                                    const gert::StorageShape* quantOffset2Shape,
                                                    int64_t quantScale2ShapeSize, const PFAShapeInfo& queryShapeInfo,
                                                    const PFAShapeInfo& valueShapeInfo) const
{
    if (quantOffset2Shape != nullptr) {
        int64_t quantOffset2ShapeSize = quantOffset2Shape->GetStorageShape().GetShapeSize();
        OP_CHECK_IF(quantScale2ShapeSize != quantOffset2ShapeSize,
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        contextKeyParams.opName, "quant_scale2, quant_offset2",
                        (std::to_string(quantScale2ShapeSize) + ", " + std::to_string(quantOffset2ShapeSize)).c_str(),
                        "quant_scale2 and quant_offset2 dimension multiply results must be equal"),
                    return false);
    }

    uint64_t quantScale2ShapeSizePerChannel =
        static_cast<uint64_t>(queryShapeInfo.n) * static_cast<uint64_t>(valueShapeInfo.d);
    OP_CHECK_IF(
        quantScale2ShapeSize < 0 || ((static_cast<uint64_t>(quantScale2ShapeSize) != 1U) &&
                                     (static_cast<uint64_t>(quantScale2ShapeSize) != quantScale2ShapeSizePerChannel)),
        OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
            contextKeyParams.opName, "quant_scale2, quant_offset2", std::to_string(quantScale2ShapeSize).c_str(),
            "post quant scale2/offset2 dimension multiply result only supports 1 or qN * vD"),
        return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPerTensorQuantParams(const FiaArch38ContextParams& contextKeyParams) const
{
    const gert::StorageShape* deqScale1Shape = contextKeyParams.deqScale1Shape;
    const gert::StorageShape* quantScale1Shape = contextKeyParams.scale1Shape;
    const gert::StorageShape* deqScale2Shape = contextKeyParams.deqScale2Shape;
    const ge::DataType inputParamsType = contextKeyParams.inputDataType;
    OP_CHECK_IF((inputParamsType != ge::DT_INT8) && (inputParamsType != ge::DT_HIFLOAT8) &&
                    (inputParamsType != ge::DT_FLOAT8_E4M3FN),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                    contextKeyParams.opName, "query", GetPfaDataTypeStr(contextKeyParams.inputDataType).c_str(),
                    "input type must be INT8, HIFLOAT8 or FLOAT8_E4M3FN in per-tensor quant scenario"),
                return false);
    OP_CHECK_IF((deqScale1Shape == nullptr) || (quantScale1Shape == nullptr) || (deqScale2Shape == nullptr),
                OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "deq_scale1, quant_scale1, deq_scale2"),
                return false);
    OP_CHECK_IF(
        (deqScale1Shape != nullptr && deqScale1Shape->GetStorageShape().GetShapeSize() == 0) ||
            (quantScale1Shape != nullptr && quantScale1Shape->GetStorageShape().GetShapeSize() == 0) ||
            (deqScale2Shape != nullptr && deqScale2Shape->GetStorageShape().GetShapeSize() == 0),
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(contextKeyParams.opName, "deq_scale1, quant_scale1, deq_scale2", "0",
                                                   "quant scale inputs must not be empty in per-tensor quant scenario"),
        return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPerblockQuantParams(const FiaArch38ContextParams& contextKeyParams,
                                                     const PFAShapeInfo& queryShapeInfo,
                                                     const PFAShapeInfo& keyShapeInfo,
                                                     const PFAShapeInfo& valueShapeInfo) const
{
    if (!CheckPerblockQuantDtype(contextKeyParams)) {
        return false;
    }
    PerblockScaleCheckCtx dequantCtx = {
        contextKeyParams.dequantScaleQueryShape, 128U, "dequant_scale_query",
        "dequantScaleQueryShape must be (b, n1, ceil(s1/128), 1) in per-block quant scenario"};
    if (!CheckPerblockScaleShape(contextKeyParams, queryShapeInfo, dequantCtx)) {
        return false;
    }
    PerblockScaleCheckCtx keyCtx = {
        contextKeyParams.keyAntiquantScaleShape, 256U, "key_antiquant_scale",
        "keyAntiquantScaleShape must be (b, n2, ceil(s2/256), 1) in per-block quant scenario"};
    if (!CheckPerblockScaleShape(contextKeyParams, keyShapeInfo, keyCtx)) {
        return false;
    }
    PerblockScaleCheckCtx valueCtx = {
        contextKeyParams.valueAntiquantScaleShape, 256U, "value_antiquant_scale",
        "valueAntiquantScaleShape must be (b, n2, ceil(s2/256), 1) when enablePertensorQuant is true"};
    if (!CheckPerblockScaleShape(contextKeyParams, valueShapeInfo, valueCtx)) {
        return false;
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPerblockQuantDtype(const FiaArch38ContextParams& contextKeyParams) const
{
    const ge::DataType dequantScaleQueryType = contextKeyParams.dequantScaleQueryType;
    const ge::DataType keyAntiquantScaleType = contextKeyParams.keyAntiquantScaleType;
    const ge::DataType valueAntiquantScaleType = contextKeyParams.valueAntiquantScaleType;
    OP_CHECK_IF(
        (contextKeyParams.inputDataType != ge::DT_HIFLOAT8) && (contextKeyParams.inputDataType != ge::DT_FLOAT8_E4M3FN),
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
            contextKeyParams.opName, "query", GetPfaDataTypeStr(inputType).c_str(),
            "input type must be HIFLOAT8 or FLOAT8_E4M3FN in per-block quant scenario"),
        return false);
    OP_CHECK_IF((dequantScaleQueryType != ge::DT_FLOAT) || (keyAntiquantScaleType != ge::DT_FLOAT) ||
                    (valueAntiquantScaleType != ge::DT_FLOAT),
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    contextKeyParams.opName, "dequant_scale_query, key_antiquant_scale, value_antiquant_scale",
                    (GetPfaDataTypeStr(dequantScaleQueryType) + ", " + GetPfaDataTypeStr(keyAntiquantScaleType) + ", " +
                     GetPfaDataTypeStr(valueAntiquantScaleType))
                        .c_str(),
                    "dequant scale type must be DT_FLOAT in per-block quant scenario"),
                return false);
    OP_CHECK_IF((contextKeyParams.dequantScaleQueryShape == nullptr) ||
                    (contextKeyParams.keyAntiquantScaleShape == nullptr) ||
                    (contextKeyParams.valueAntiquantScaleShape == nullptr),
                OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName,
                                           "dequant_scale_query, key_antiquant_scale, value_antiquant_scale"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPerblockScaleShape(const FiaArch38ContextParams& contextKeyParams,
                                                    const PFAShapeInfo& shapeInfo,
                                                    const PerblockScaleCheckCtx& ctx) const
{
    OP_CHECK_IF(
        (ctx.scaleShape->GetStorageShape().GetDim(0) != shapeInfo.b) ||
            (ctx.scaleShape->GetStorageShape().GetDim(1) != shapeInfo.n) ||
            (ctx.scaleShape->GetStorageShape().GetDim(2) !=
             CeilDivision(shapeInfo.s, static_cast<uint64_t>(ctx.blockSize))) || // 2 is the dim of scale along s.
            (ctx.scaleShape->GetStorageShape().GetDim(3) != 1),                  // 3 is the dim of scale along d.
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(contextKeyParams.opName, ctx.paramName.c_str(),
                                              ("[" + std::to_string(ctx.scaleShape->GetStorageShape().GetDim(0)) +
                                               ", " + std::to_string(ctx.scaleShape->GetStorageShape().GetDim(1)) +
                                               ", " + std::to_string(ctx.scaleShape->GetStorageShape().GetDim(2)) +
                                               ", " + std::to_string(ctx.scaleShape->GetStorageShape().GetDim(3)) + "]")
                                                  .c_str(),
                                              ctx.expectShape.c_str()),
        return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPostQuantParams(const FiaArch38ContextParams& contextKeyParams,
                                                 const PFAShapeInfo& queryShapeInfo,
                                                 const PFAShapeInfo& valueShapeInfo) const
{
    const gert::StorageShape* quantScale2Shape = contextKeyParams.scale2Shape;
    const gert::StorageShape* quantOffset2Shape = contextKeyParams.offset2Shape;
    const ge::DataType quantScale2Type = contextKeyParams.quantScale2Type;
    uint64_t h = queryShapeInfo.h;
    uint64_t n = queryShapeInfo.n;

    int64_t quantScale2ShapeSize = 0;
    uint32_t quantD = 0;
    uint64_t queryD = h / n;

    OP_CHECK_IF(outputType != ge::DT_INT8 && outputType != ge::DT_FLOAT8_E4M3FN && outputType != ge::DT_HIFLOAT8,
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "attention_out",
                                                      GetPfaDataTypeStr(outputType).c_str(),
                                                      "output type only supports int8, fp8_e4m3fn or hifloat8"),
                return false);

    // Basic verification: quantScale2 must be inputted and not an empty tensor
    OP_CHECK_IF(quantScale2Shape == nullptr, OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "quant_scale2"),
                return false);
    quantScale2ShapeSize = quantScale2Shape->GetStorageShape().GetShapeSize();
    quantD = quantScale2ShapeSize / n;
    OP_CHECK_IF(quantScale2ShapeSize == 0,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, "quant_scale2", "0",
                                                          "quant_scale2 must not be empty in post quant scenario"),
                return false);

    // altert unsupported situation(post quant per-tensor + BF16 + BSH + D unalign)
    if ((contextKeyParams.inputDataType == ge::DT_BF16) && (quantScale2ShapeSize == 1) &&
        (inputLayout == InputLayout::BSH) && (queryD % BYTE_BLOCK != 0)) {
        OP_LOGW(contextKeyParams.opName,
                "post quant per-tensor doesn't support D unaligned(%u), when qkv is bf16 and layout is BSH.", queryD);
    }
    OP_CHECK_IF(!CheckPostQuantDtype(contextKeyParams, quantOffset2Shape, quantScale2Type) ||
                    !CheckPostQuantShapeSize(contextKeyParams, quantOffset2Shape, quantScale2ShapeSize, queryShapeInfo,
                                             valueShapeInfo),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "post quant params check failed!"), return false);
    return true;
}

bool FiaArch38PromptTiling::CheckAntiquantParamsShape(FiaArch38ContextParams& contextKeyParams) const
{
    const gert::StorageShape* antiquantScaleShape = contextKeyParams.antiquantScaleShape;
    // 伪量化收编至ifa tiling，pfa接口kv若传入伪量化dtype均会在此被拒绝
    OP_CHECK_IF(contextKeyParams.antiquantScale == nullptr || antiquantScaleShape == nullptr,
                OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "antiquant_scale"), return false);
    return true;
}

} // namespace arch38
} // namespace optiling
