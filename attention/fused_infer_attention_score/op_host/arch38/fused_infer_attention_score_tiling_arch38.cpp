/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

#include "op_host/util/op_const_def.h"
#include "fused_infer_attention_score_tiling_arch38.h"
#include "fia_arch38_context_convert.h"
#include "fia_arch38_prompt_tiling.h"
#include "../fused_infer_attention_score_tiling_index.h"
#include "log/log.h"
#include "log/error_code.h"
#include "err/ops_err.h"
#include "tiling/tiling_api.h"
#include "platform/platform_info.h"

using namespace ge;
using namespace AscendC;
using namespace optiling::arch38;
namespace optiling {
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1000000000000000090, FlashAttentionScoreSimplifiedTilingData)

REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002122000000021012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001121000000021012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000000021012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000000021012, FlashAttentionScoreSimplifiedTilingData)

REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000040001212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000040021212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000000001212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000000021212, FlashAttentionScoreSimplifiedTilingData)

// attenMask使能场景，mask占tilingKey第8位(1e8)，且使能mask时DN(4e7)必然关闭
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000100001212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000100021212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000100001212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000100021212, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002122000100021012, FlashAttentionScoreSimplifiedTilingData)

REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000040001012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1002312000100001012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1003113000100001012, FlashAttentionScoreSimplifiedTilingData)
REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_1001311000100001012, FlashAttentionScoreSimplifiedTilingData)

static ge::graphStatus CheckQueryOutShapeNSD(gert::TilingContext* context, const QKVShapeCheckCtx& ctx,
                                             uint32_t& queryD, uint32_t& valueD)
{
    OP_CHECK_IF((ctx.tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "query",
                                                         std::to_string(ctx.tempQ->GetStorageShape().GetDimNum()),
                                                         "input shape dim should be 3!"),
                return ge::GRAPH_FAILED);
    queryD = ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
    valueD = ctx.tempV->GetStorageShape().GetDim(VALUE_DIM_2);
    OP_CHECK_IF(
        ((queryD == valueD) && (ctx.tempQ->GetStorageShape() != ctx.tempOut->GetStorageShape())),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "query,attention_out",
            GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " + GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
            "Layout is NSD and Query shape does NOT match Attention Out shape"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckQueryOutShapeTND(gert::TilingContext* context, const QKVShapeCheckCtx& ctx,
                                             uint32_t& queryD, uint32_t& valueD)
{
    OP_CHECK_IF(
        (ctx.tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "query",
            std::to_string(ctx.tempQ->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", input query shape dim should be 3!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (ctx.tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_3),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "attention_out",
            std::to_string(ctx.tempOut->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", attention out shape dim should be 3!"),
        return ge::GRAPH_FAILED);
    queryD = ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
    valueD = ctx.tempV->GetStorageShape().GetDim(VALUE_DIM_2);
    if (ctx.inputLayoutStr == "TND") {
        OP_CHECK_IF(
            ((queryD == valueD) && (ctx.tempQ->GetStorageShape() != ctx.tempOut->GetStorageShape())),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "query,attention_out",
                                                   GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " +
                                                       GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
                                                   "Layout is TND and Query shape does NOT match Attention Out shape"),
            return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(((queryD == valueD) &&
                     ((ctx.tempQ->GetStorageShape().GetDim(0) != ctx.tempOut->GetStorageShape().GetDim(1)) ||
                      (ctx.tempQ->GetStorageShape().GetDim(1) != ctx.tempOut->GetStorageShape().GetDim(0)) ||
                      (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_2) !=
                       ctx.tempOut->GetStorageShape().GetDim(QUERY_DIM_2)))),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "query,attention_out",
                        GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " +
                            GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
                        "Layout is TND_NTD and Query shape does NOT match Attention Out shape"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckQueryOutShapeNTD(gert::TilingContext* context, const QKVShapeCheckCtx& ctx,
                                             uint32_t& queryD, uint32_t& valueD)
{
    OP_CHECK_IF(
        (ctx.tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "query",
            std::to_string(ctx.tempQ->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", input query shape dim should be 3!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (ctx.tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_3),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "attention_out",
            std::to_string(ctx.tempOut->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", attention out shape dim should be 3!"),
        return ge::GRAPH_FAILED);
    queryD = ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
    valueD = ctx.tempV->GetStorageShape().GetDim(VALUE_DIM_2);
    if (ctx.inputLayoutStr == "NTD") {
        OP_CHECK_IF(
            ((queryD == valueD) && (ctx.tempQ->GetStorageShape() != ctx.tempOut->GetStorageShape())),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "query,attention_out",
                                                   GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " +
                                                       GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
                                                   "Layout is NTD and Query shape does NOT match Attention Out shape"),
            return ge::GRAPH_FAILED);
    } else if (ctx.inputLayoutStr == "NTD_TND") {
        OP_CHECK_IF(
            ((queryD == valueD) &&
             ((ctx.tempQ->GetStorageShape().GetDim(0) != ctx.tempOut->GetStorageShape().GetDim(1)) ||
              (ctx.tempQ->GetStorageShape().GetDim(1) != ctx.tempOut->GetStorageShape().GetDim(0)) ||
              (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_2)))),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "query,attention_out",
                GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " +
                    GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
                "Layout is NTD_TND and Query shape does NOT match Attention Out shape"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckQueryOutShapeBSH(gert::TilingContext* context, const QKVShapeCheckCtx& ctx,
                                             uint32_t& queryD, uint32_t& valueD)
{
    OP_CHECK_IF(
        (ctx.tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "query",
            std::to_string(ctx.tempQ->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", input query shape dim should be 3!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (ctx.tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_3),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "attention_out",
            std::to_string(ctx.tempOut->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", attention out shape dim should be 3!"),
        return ge::GRAPH_FAILED);
    queryD = ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_2) / ctx.tempN;
    valueD = ctx.tempV->GetStorageShape().GetDim(VALUE_DIM_2) / ctx.tempKVN;
    OP_CHECK_IF(
        ((queryD == valueD) && (ctx.tempQ->GetStorageShape() != ctx.tempOut->GetStorageShape())),
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            context->GetNodeName(), "query,attention_out",
            GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " + GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
            "Layout is BSH and Query shape does NOT match Attention Out shape"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckQueryOutShapeBNSD(gert::TilingContext* context, const QKVShapeCheckCtx& ctx,
                                              uint32_t& queryD, uint32_t& valueD)
{
    OP_CHECK_IF(
        (ctx.tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_4),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "query",
            std::to_string(ctx.tempQ->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", input query shape dim should be 4!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (ctx.tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_4),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "attention_out",
            std::to_string(ctx.tempOut->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", attention out shape dim should be 4!"),
        return ge::GRAPH_FAILED);
    queryD = ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_3);
    valueD = ctx.tempV->GetStorageShape().GetDim(VALUE_DIM_3);
    if (ctx.inputLayoutStr == "BNSD_BSND" || ctx.inputLayoutStr == "BSND_BNSD") {
        OP_CHECK_IF(
            ((queryD == valueD) &&
             ((ctx.tempQ->GetStorageShape().GetDim(0) != ctx.tempOut->GetStorageShape().GetDim(0)) ||
              (ctx.tempQ->GetStorageShape().GetDim(1) != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
              (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != ctx.tempOut->GetStorageShape().GetDim(1)) ||
              (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_3) != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_3)))),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "query,attention_out",
                GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " +
                    GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
                "Layout is " + ctx.inputLayoutStr + " and Query shape does NOT match Attention Out shape"),
            return ge::GRAPH_FAILED);
    } else if (ctx.inputLayoutStr == "BNSD") {
        OP_CHECK_IF(
            ((queryD == valueD) && (ctx.tempQ->GetStorageShape() != ctx.tempOut->GetStorageShape())),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "query,attention_out",
                                                   GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " +
                                                       GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
                                                   "Layout is BNSD and Query shape does NOT match Attention Out shape"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckQueryOutShapeBSHBNSD(gert::TilingContext* context, const QKVShapeCheckCtx& ctx,
                                                 uint32_t& queryD, uint32_t& valueD)
{
    OP_CHECK_IF(
        (ctx.tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_3),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "query",
            std::to_string(ctx.tempQ->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", input query shape dim should be 3!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (ctx.tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_4),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "attention_out",
            std::to_string(ctx.tempOut->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", attention out shape dim should be 4!"),
        return ge::GRAPH_FAILED);
    queryD = ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_2) / ctx.tempN;
    valueD = ctx.tempV->GetStorageShape().GetDim(VALUE_DIM_2) / ctx.tempKVN;
    if (ctx.inputLayoutStr == "BSH_BNSD") {
        OP_CHECK_IF(((queryD == valueD) && ((ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_0) !=
                                             ctx.tempOut->GetStorageShape().GetDim(0)) ||
                                            (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_1) !=
                                             ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
                                            queryD != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_3)) ||
                     ctx.tempN != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_1)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "query,attention_out",
                        GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " +
                            GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
                        "Layout is " + ctx.inputLayoutStr + " and Query shape does NOT match Attention Out shape"),
                    return ge::GRAPH_FAILED);
    } else if (ctx.inputLayoutStr == "BSH_NBSD") {
        OP_CHECK_IF(((queryD == valueD) && ((ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_0) !=
                                             ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_1)) ||
                                            (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_1) !=
                                             ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
                                            queryD != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_3)) ||
                     ctx.tempN != ctx.tempOut->GetStorageShape().GetDim(0)),
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "query,attention_out",
                        GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " +
                            GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
                        "Layout is " + ctx.inputLayoutStr + " and Query shape does NOT match Attention Out shape"),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckQueryOutShapeBSNDNBSD(gert::TilingContext* context, const QKVShapeCheckCtx& ctx,
                                                  uint32_t& queryD, uint32_t& valueD)
{
    OP_CHECK_IF(
        (ctx.tempQ->GetStorageShape().GetDimNum() != QUERY_DIM_4),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "query",
            std::to_string(ctx.tempQ->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", input query shape dim should be 4!"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        (ctx.tempOut->GetStorageShape().GetDimNum() != QUERY_DIM_4),
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            context->GetNodeName(), "attention_out",
            std::to_string(ctx.tempOut->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
            "The current layout is " + ctx.inputLayoutStr + ", attention out shape dim should be 4!"),
        return ge::GRAPH_FAILED);
    queryD = ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_3);
    valueD = ctx.tempV->GetStorageShape().GetDim(VALUE_DIM_3);
    if (ctx.inputLayoutStr == "BSND_NBSD") {
        OP_CHECK_IF(
            ((queryD == valueD) &&
             ((ctx.tempQ->GetStorageShape().GetDim(0) != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_1)) ||
              (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_1) != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
              (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != ctx.tempOut->GetStorageShape().GetDim(0)) ||
              (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_3) != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_3)))),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "query,attention_out",
                GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " +
                    GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
                "Layout is " + ctx.inputLayoutStr + " and Query shape does NOT match Attention Out shape"),
            return ge::GRAPH_FAILED);
    } else if (ctx.inputLayoutStr == "BNSD_NBSD") {
        OP_CHECK_IF(
            ((queryD == valueD) &&
             ((ctx.tempQ->GetStorageShape().GetDim(0) != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_1)) ||
              (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_1) != ctx.tempOut->GetStorageShape().GetDim(0)) ||
              (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_2) != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_2)) ||
              (ctx.tempQ->GetStorageShape().GetDim(QUERY_DIM_3) != ctx.tempOut->GetStorageShape().GetDim(OUT_DIM_3)))),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                context->GetNodeName(), "query,attention_out",
                GetShapeDebugStr(ctx.tempQ->GetStorageShape()) + " vs " +
                    GetShapeDebugStr(ctx.tempOut->GetStorageShape()),
                "Layout is " + ctx.inputLayoutStr + " and Query shape does NOT match Attention Out shape"),
            return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckSoftmaxLseShape(gert::TilingContext* context, const LseShapeCheckCtx& ctx)
{
    if (ctx.inputLayoutStr == "TND" || ctx.inputLayoutStr == "TND_NTD") {
        OP_CHECK_IF(((ctx.tempLse->GetStorageShape().GetDimNum() != QUERY_DIM_3)), // 3：lse shape TN1
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "softmax_lse",
                                                             std::to_string(ctx.tempLse->GetStorageShape().GetDimNum()),
                                                             "Layout is TND/TND_NTD SoftmaxLse shape dim should be 3"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((((ctx.tempLse->GetStorageShape().GetDim(QUERY_DIM_0) != ctx.t) ||
                      (ctx.tempLse->GetStorageShape().GetDim(QUERY_DIM_1) !=
                       ctx.tempN) || // 0: the first dimension 1: the second dimension
                      (ctx.tempLse->GetStorageShape().GetDim(QUERY_DIM_2) != 1))), // 2: the third dimension
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "softmax_lse",
                        GetShapeDebugStr(ctx.tempLse->GetStorageShape()) + " vs TN1[" + std::to_string(ctx.t) + ", " +
                            std::to_string(ctx.tempN) + ", 1]",
                        "Layout is TND SoftmaxLse shape does not match TN1"), // 2: the third dimension
                    return ge::GRAPH_FAILED);
    } else if (ctx.inputLayoutStr == "NTD" || ctx.inputLayoutStr == "NTD_TND") {
        OP_CHECK_IF(((ctx.tempLse->GetStorageShape().GetDimNum() != QUERY_DIM_3)), // 3：lse shape TN1
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(context->GetNodeName(), "softmax_lse",
                                                             std::to_string(ctx.tempLse->GetStorageShape().GetDimNum()),
                                                             "Layout is NTD/NTD_TND SoftmaxLse shape dim should be 3"),
                    return ge::GRAPH_FAILED);
        OP_CHECK_IF((((ctx.tempLse->GetStorageShape().GetDim(QUERY_DIM_1) != ctx.tempN) ||
                      (ctx.tempLse->GetStorageShape().GetDim(QUERY_DIM_0) !=
                       ctx.t) || // 0: the first dimension 1: the second dimension
                      (ctx.tempLse->GetStorageShape().GetDim(QUERY_DIM_2) != 1))), // 2: the third dimension
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "softmax_lse",
                        GetShapeDebugStr(ctx.tempLse->GetStorageShape()) + " vs TN1[" + std::to_string(ctx.t) + ", " +
                            std::to_string(ctx.tempN) + ", 1]",
                        "Layout is NTD SoftmaxLse shape does not match TN1"), // 2: the third dimension
                    return ge::GRAPH_FAILED);
    } else {
        OP_CHECK_IF(
            ((ctx.tempLse->GetStorageShape().GetDimNum() != QUERY_DIM_4)), // 4：lse shape BNS1
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                context->GetNodeName(), "softmax_lse",
                std::to_string(ctx.tempLse->GetStorageShape().GetDimNum()) + " (layout is " + ctx.inputLayoutStr + ")",
                "Layout is " + ctx.inputLayoutStr + ", SoftmaxLse shape dim should be 4"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF((((ctx.tempLse->GetStorageShape().GetDim(QUERY_DIM_0) != ctx.b) ||
                      (ctx.tempLse->GetStorageShape().GetDim(QUERY_DIM_1) !=
                       ctx.tempN) || // 0: the first dimension 1: the second dimension
                      (ctx.tempLse->GetStorageShape().GetDim(QUERY_DIM_2) != ctx.s) ||
                      (ctx.tempLse->GetStorageShape().GetDim(QUERY_DIM_3) !=
                       1))), // 2: the third dimension 3: the fourth dimension
                    OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                        context->GetNodeName(), "softmax_lse",
                        GetShapeDebugStr(ctx.tempLse->GetStorageShape()) + " vs BNS1[" + std::to_string(ctx.b) + ", " +
                            std::to_string(ctx.tempN) + ", " + std::to_string(ctx.s) + ", 1]",
                        "SoftmaxLse shape does not match BNS1"), // 2: the third dimension 3: the fourth dimension
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingFusedInferAttentionScoreArch38(gert::TilingContext* context)
{
    FusedInferAttentionScoreTilingArch38 FIATilingArch38(context);
    auto ret = FIATilingArch38.DoTiling(nullptr);
    return ret;
}

ge::graphStatus FusedInferAttentionScoreTilingArch38::DoOpTiling()
{
    if (context_ == nullptr) {
        OP_LOGE_WITHOUT_REPORT("FusedInferAttentionScore", "tiling context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    bool isMaxWorkspace = GetMaxWorkspaceFlag(*context_);

    auto tempQ = context_->GetInputShape(QUERY_INDEX);
    auto tempV = context_->GetDynamicInputShape(VALUE_INDEX, 0);
    auto tempOut = context_->GetOutputShape(ATTENTION_OUT_INDEX);
    auto tempLse = context_->GetOutputShape(SOFTMAX_LSE_INDEX);
    bool qOutEmptyTensor = false;
    uint32_t queryD = 1U;
    uint32_t valueD = 1U;
    OP_CHECK_IF((tempQ == nullptr), OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "query"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempV == nullptr), OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "value"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((tempOut == nullptr), OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "attention_out"),
                return ge::GRAPH_FAILED);
    auto attrs = context_->GetAttrs();
    OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == 0 && tempOut->GetStorageShape().GetShapeSize() != 0) ||
                    (tempQ->GetStorageShape().GetShapeSize() != 0 && tempOut->GetStorageShape().GetShapeSize() == 0),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    context_->GetNodeName(), "query,attention_out",
                    std::to_string(tempQ->GetStorageShape().GetShapeSize()) + ", " +
                        std::to_string(tempOut->GetStorageShape().GetShapeSize()),
                    "query shape size and attention Out shape size cannot be empty while the other is not"),
                return ge::GRAPH_FAILED);
    if (tempQ->GetStorageShape().GetShapeSize() == 0 && tempOut->GetStorageShape().GetShapeSize() == 0) {
        qOutEmptyTensor = true;
    }

    OP_CHECK_IF((tempQ->GetStorageShape().GetShapeSize() == gert::Shape::kInvalidDimValue),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "query",
                                                      std::to_string(tempQ->GetStorageShape().GetShapeSize()),
                                                      "Get the shape size of Query failed!"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(attrs == nullptr, ReportNullPointer(context_, __func__, "attrs"), return ge::GRAPH_FAILED);

    const int64_t* numHeadsAttrPtr = attrs->GetAttrPointer<int64_t>(ATTR_N_INDEX);
    const int64_t* numKvHeadsAttrPtr = attrs->GetAttrPointer<int64_t>(ATTR_NUM_KV_HEADS_INDEX);
    const char* inputLayoutAttrPtr = attrs->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
    const bool* lseFlagAttrPtr = attrs->GetAttrPointer<bool>(ATTR_SOFTMAX_LSE_FLAG_INDEX);
    OP_CHECK_IF(((numHeadsAttrPtr == nullptr) || (numKvHeadsAttrPtr == nullptr) || (inputLayoutAttrPtr == nullptr) ||
                 (lseFlagAttrPtr == nullptr)),
                ReportNullPointer(context_, __func__, "num_heads/num_key_value_heads/input_layout/softmax_lse_flag"),
                return ge::GRAPH_FAILED);
    int64_t tempN = *numHeadsAttrPtr;
    int64_t tempKVN = *numKvHeadsAttrPtr;
    OP_CHECK_IF(tempN == 0,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "num_heads", std::to_string(tempN),
                                                      "Q numhead is 0!"),
                return ge::GRAPH_FAILED);
    if (tempKVN == 0) {
        tempKVN = tempN;
    }
    const string inputLayoutStr = string(inputLayoutAttrPtr);
    int64_t s = 0;
    int64_t b = tempQ->GetStorageShape().GetDim(QUERY_DIM_0);
    int64_t t = 0;
    bool lseFlag = *lseFlagAttrPtr;
    if (inputLayoutStr == "BNSD" || inputLayoutStr == "BNSD_BSND") {
        s = tempQ->GetStorageShape().GetDim(QUERY_DIM_2);
    } else if (inputLayoutStr == "TND" || inputLayoutStr == "TND_NTD") {
        if (isMaxWorkspace) {
            t = tempQ->GetStorageShape().GetDim(QUERY_DIM_0);
            s = tempQ->GetStorageShape().GetDim(QUERY_DIM_0);
        } else {
            const gert::Tensor* actualSeqLength = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
            const gert::StorageShape* actualSeqLengthShape = context_->GetOptionalInputShape(ACTUAL_SEQ_Q_INDEX);
            int64_t actSeqLenDims =
                (actualSeqLengthShape != nullptr) ? actualSeqLengthShape->GetStorageShape().GetShapeSize() : 0;
            OP_CHECK_IF(((actualSeqLength == nullptr) || (actSeqLenDims == 0) ||
                         (actualSeqLength->GetData<int64_t>() == nullptr)),
                        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "actual_seq_lengths"),
                        return ge::GRAPH_FAILED);
            s = actualSeqLength->GetData<int64_t>()[0];
            for (int i = 1; i < actSeqLenDims; ++i) {
                s = std::max(s, actualSeqLength->GetData<int64_t>()[i] - actualSeqLength->GetData<int64_t>()[i - 1]);
            }
            t = actualSeqLength->GetData<int64_t>()[actSeqLenDims - 1];
        }
    } else if (inputLayoutStr == "NTD" || inputLayoutStr == "NTD_TND") {
        if (isMaxWorkspace) {
            t = tempQ->GetStorageShape().GetDim(QUERY_DIM_1);
            s = tempQ->GetStorageShape().GetDim(QUERY_DIM_1);
        } else {
            const gert::Tensor* actualSeqLength = context_->GetOptionalInputTensor(ACTUAL_SEQ_Q_INDEX);
            const gert::StorageShape* actualSeqLengthShape = context_->GetOptionalInputShape(ACTUAL_SEQ_Q_INDEX);
            int64_t actSeqLenDims =
                (actualSeqLengthShape != nullptr) ? actualSeqLengthShape->GetStorageShape().GetShapeSize() : 0;
            OP_CHECK_IF(((actualSeqLength == nullptr) || (actSeqLenDims == 0) ||
                         (actualSeqLength->GetData<int64_t>() == nullptr)),
                        OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "actual_seq_lengths"),
                        return ge::GRAPH_FAILED);
            s = actualSeqLength->GetData<int64_t>()[0];
            for (int i = 1; i < actSeqLenDims; ++i) {
                s = std::max(s, actualSeqLength->GetData<int64_t>()[i] - actualSeqLength->GetData<int64_t>()[i - 1]);
            }
            t = actualSeqLength->GetData<int64_t>()[actSeqLenDims - 1];
        }
    } else {
        s = tempQ->GetStorageShape().GetDim(1);
    }
    QKVShapeCheckCtx qkvCtx;
    qkvCtx.tempQ = tempQ;
    qkvCtx.tempV = tempV;
    qkvCtx.tempOut = tempOut;
    qkvCtx.tempN = tempN;
    qkvCtx.tempKVN = tempKVN;
    qkvCtx.inputLayoutStr = inputLayoutStr;
    if (inputLayoutStr == "NSD") { // 当前已没有NSD, 回主线后在FIA tiling Arch38内删除
        b = 1;
        if (CheckQueryOutShapeNSD(context_, qkvCtx, queryD, valueD) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (inputLayoutStr == "TND" || inputLayoutStr == "TND_NTD") {
        if (CheckQueryOutShapeTND(context_, qkvCtx, queryD, valueD) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (inputLayoutStr == "NTD" || inputLayoutStr == "NTD_TND") {
        if (CheckQueryOutShapeNTD(context_, qkvCtx, queryD, valueD) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (inputLayoutStr == "BSH") {
        if (CheckQueryOutShapeBSH(context_, qkvCtx, queryD, valueD) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (inputLayoutStr == "BNSD_BSND" || inputLayoutStr == "BNSD" || inputLayoutStr == "BSND" ||
               inputLayoutStr == "BSND_BNSD") {
        if (CheckQueryOutShapeBNSD(context_, qkvCtx, queryD, valueD) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (inputLayoutStr == "BSH_BNSD" || inputLayoutStr == "BSH_NBSD") {
        if (CheckQueryOutShapeBSHBNSD(context_, qkvCtx, queryD, valueD) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else if (inputLayoutStr == "BSND_NBSD" || inputLayoutStr == "BNSD_NBSD") {
        if (CheckQueryOutShapeBSNDNBSD(context_, qkvCtx, queryD, valueD) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context_->GetNodeName(), "input_layout", inputLayoutStr,
            "Invalid input layout. Currently only "
            "TND/NTD/BSH/BNSD/BSND/BSND_BNSD/BNSD_BSND/BSH_BNSD/BSND_NBSD/BNSD_NBSD/BSH_NBSD layout are supported!");
        return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF((queryD > DLIMIT),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "query", std::to_string(queryD),
                                                      "D of query should be less than or equal to 512"),
                return ge::GRAPH_FAILED);
    bool inputOutputIsNullPtr =
        (context_->GetInputDesc(QUERY_INDEX) == nullptr) || (context_->GetInputDesc(KEY_INDEX) == nullptr) ||
        (context_->GetInputDesc(VALUE_INDEX) == nullptr) || (context_->GetOutputDesc(ATTENTION_OUT_INDEX) == nullptr) ||
        (context_->GetInputShape(QUERY_INDEX) == nullptr) || (context_->GetInputShape(KEY_INDEX) == nullptr) ||
        (context_->GetInputShape(VALUE_INDEX) == nullptr) || (context_->GetOutputShape(ATTENTION_OUT_INDEX) == nullptr);
    OP_CHECK_IF(inputOutputIsNullPtr,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(context_->GetNodeName(), "query,key,value,attention_out",
                                                       "nullptr", "q, k, v or attenOut is nullptr!"),
                return ge::GRAPH_FAILED);
    auto qDType = context_->GetInputDesc(QUERY_INDEX)->GetDataType();
    auto kDType = context_->GetInputDesc(KEY_INDEX)->GetDataType();
    // arch38 仅注册非伪量化模板（q == k），伪量化（q != k）无模板可分发，直接拒绝
    OP_CHECK_IF(qDType != kDType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context_->GetNodeName(), "query, key",
                    (std::string(GetPfaDataTypeStr(qDType)) + ", " + std::string(GetPfaDataTypeStr(kDType))).c_str(),
                    "pseudo-quantization (query dtype != key dtype) is not supported on this platform."),
                return ge::GRAPH_FAILED);
    // IFA 非 MLA 场景（s=1/BNSD/无 rope）复用 PFA 模板，仅做 IFA 门禁
    const bool isIfaNonMlaScenario = (s == 1) && (inputLayoutStr == "BNSD") &&
                                     (context_->GetOptionalInputShape(QUERY_ROPE_INDEX) == nullptr) &&
                                     (context_->GetOptionalInputShape(KEY_ROPE_INDEX) == nullptr);
    // 进入公共tiling流程前，先校验decode incre场景的输入
    if (isIfaNonMlaScenario) {
        OP_CHECK_IF(CheckArch38IncreInputs(*context_) != ge::GRAPH_SUCCESS,
                    OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "unsupported IFA scenario on this platform."),
                    return ge::GRAPH_FAILED);
    }

    constexpr int64_t D_ALIGN_32 = 32;
    constexpr int64_t D_ALIGN_16 = 16;

    FiaArch38FlashAttentionTilingData pfaTilingData;
    FiaArch38PromptTiling pfaTiling(context_);
    FiaArch38ContextParams contextParamsForPFATiling;
    FiaArch38CompileInfo tempCompileInfoPtr;

    OP_CHECK_IF((attrs->GetAttrPointer<uint64_t>(ATTR_ANTIQUANT_MODE_INDEX) != nullptr) &&
                    (*attrs->GetAttrPointer<uint64_t>(ATTR_ANTIQUANT_MODE_INDEX) != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    context_->GetNodeName(), "antiquant_mode",
                    std::to_string(*attrs->GetAttrPointer<uint64_t>(ATTR_ANTIQUANT_MODE_INDEX)),
                    "antiquant_mode is not supported!"),
                return ge::GRAPH_FAILED);

    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, ReportNullPointer(context_, __func__, "platformInfoPtr"),
                return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    tempCompileInfoPtr.aivNum = ascendcPlatform.GetCoreNumAiv();
    tempCompileInfoPtr.aicNum = ascendcPlatform.GetCoreNumAic();
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, tempCompileInfoPtr.ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, tempCompileInfoPtr.l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, tempCompileInfoPtr.l0CSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, tempCompileInfoPtr.l0ASize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, tempCompileInfoPtr.l0BSize);
    tempCompileInfoPtr.socShortName = ascendcPlatform.GetSocVersion();
    tempCompileInfoPtr.defaultSysWorkspaceSize = 0;

    contextParamsForPFATiling.compileInfoPtr = &tempCompileInfoPtr;
    auto ret = ConvertContextToParamsPFA(context_, contextParamsForPFATiling, isMaxWorkspace);
    if (ret != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(),
                               "Error occurred while converting tilingContext to PFA context!");
        return ret;
    }
    if (lseFlag != false) {
        if (!pfaTiling.CheckNonEmptyShapeExceptions(contextParamsForPFATiling, contextParamsForPFATiling.lseoutputShape,
                                                    "softmaxLse")) {
            return ge::GRAPH_FAILED;
        }
        OP_CHECK_IF(((tempLse == nullptr)), OP_LOGE_WITH_INVALID_INPUT(context_->GetNodeName(), "softmax_lse"),
                    return ge::GRAPH_FAILED);

        if (!qOutEmptyTensor) { // q、out为空时，lse为空则不输出，不为空则输出inf，跳过lse形状校验
            LseShapeCheckCtx lseCtx;
            lseCtx.tempLse = tempLse;
            lseCtx.inputLayoutStr = inputLayoutStr;
            lseCtx.tempN = tempN;
            lseCtx.b = b;
            lseCtx.s = s;
            lseCtx.t = t;
            if (CheckSoftmaxLseShape(context_, lseCtx) != ge::GRAPH_SUCCESS) {
                return ge::GRAPH_FAILED;
            }
        }
    }
    if (ascendcPlatform.GetCurNpuArch() != Ops::Base::DAV_5102) {
        OP_CHECK_IF(
            (((contextParamsForPFATiling.inputDataType == ge::DT_INT8) ||
              (contextParamsForPFATiling.kDataType == ge::DT_INT8) ||
              (contextParamsForPFATiling.outputDataType == ge::DT_INT8)) &&
             (queryD % D_ALIGN_32 != 0)),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "query", std::to_string(queryD),
                                                  "D of query should be 32 elements aligned when int8 is involved!"),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(
            (queryD % D_ALIGN_16 != 0),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context_->GetNodeName(), "query", std::to_string(queryD),
                                                  "D should be 16 elements aligned when FLOAT16/BFLOAT16 dtype!"),
            return ge::GRAPH_FAILED);
    }
    ret = pfaTiling.DoSubOpTiling(pfaTilingData, contextParamsForPFATiling);
    OP_CHECK_IF(ret == ge::GRAPH_FAILED, OP_LOGE_WITHOUT_REPORT(context_->GetNodeName(), "failed in FIA DoSubOpTiling"),
                return ge::GRAPH_FAILED);
    return ret;
}
} // namespace optiling
