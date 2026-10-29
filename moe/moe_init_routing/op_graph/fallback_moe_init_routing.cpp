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
 * \file fallback_moe_init_routing.cpp
 * \brief
 */
#include "log/log.h"
#include <string>
#include "fallback/fallback.h"
#include "fallback/fallback_comm.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace fallback {
using namespace ge;
using namespace gert;
constexpr size_t xInputIndex = 0;
constexpr size_t rowIdxInputIndex = 1;
constexpr size_t expertIdxInputIndex = 2;

constexpr size_t expandedXOutputIndex = 0;
constexpr size_t expandedRowIdxOutputIndex = 1;
constexpr size_t expandedExpertIdxOutputIndex = 2;

static graphStatus MoeInitRoutingExecuteFunc(OpExecuteContext* host_api_ctx)
{
    OP_CHECK_IF(host_api_ctx == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "host_api_ctx", "host_api_ctx is null"),
                return GRAPH_FAILED);

    auto x_ge = host_api_ctx->GetInputTensor(xInputIndex);
    OP_CHECK_IF(x_ge == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "x_ge", "x_ge is null"),
                return GRAPH_FAILED);
    auto row_idx_ge = host_api_ctx->GetInputTensor(rowIdxInputIndex);
    OP_CHECK_IF(row_idx_ge == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "row_idx_ge", "row_idx_ge is null"),
                return GRAPH_FAILED);
    auto expert_idx_ge = host_api_ctx->GetInputTensor(expertIdxInputIndex);
    OP_CHECK_IF(expert_idx_ge == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "expert_idx_ge", "expert_idx_ge is null"),
                return GRAPH_FAILED);

    auto expanded_x_ge = host_api_ctx->GetOutputTensor(expandedXOutputIndex);
    OP_CHECK_IF(expanded_x_ge == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "expanded_x_ge", "expanded_x_ge is null"),
                return GRAPH_FAILED);
    auto expanded_row_idx_ge = host_api_ctx->GetOutputTensor(expandedRowIdxOutputIndex);
    OP_CHECK_IF(
        expanded_row_idx_ge == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "expanded_row_idx_ge", "expanded_row_idx_ge is null"),
        return GRAPH_FAILED);
    auto expanded_expert_idx_ge = host_api_ctx->GetOutputTensor(expandedExpertIdxOutputIndex);
    OP_CHECK_IF(expanded_expert_idx_ge == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "expanded_expert_idx_ge",
                                                         "expanded_expert_idx_ge is null"),
                return GRAPH_FAILED);

    auto attrs = host_api_ctx->GetAttrs();
    OP_CHECK_IF(attrs == nullptr, OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "attrs", "attrs is null"),
                return GRAPH_FAILED);
    const int64_t* active_num = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_IF(active_num == nullptr,
                OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("aclnnfallback", "active_num", "active_num is null"),
                return GRAPH_FAILED);
    // execute opapi
    auto api_ret = EXEC_OPAPI_CMD(aclnnMoeInitRouting, x_ge, row_idx_ge, expert_idx_ge, *active_num, expanded_x_ge,
                                  expanded_row_idx_ge, expanded_expert_idx_ge);
    OP_CHECK_IF(api_ret != GRAPH_SUCCESS, OP_LOGE_WITHOUT_REPORT("aclnnfallback", "api_ret faild:%u", api_ret),
                return GRAPH_FAILED);

    return GRAPH_SUCCESS;
}

IMPL_OP(MoeInitRouting).OpExecuteFunc(MoeInitRoutingExecuteFunc);

} // namespace fallback

#ifdef __cplusplus
}
#endif
