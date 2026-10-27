/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "../../../tq_score_compute/op_kernel/tq_score_compute_config.h"
#include "infer_shape_case_executor.h"

namespace {
using Attr = gert::InfershapeContextPara::OpAttr;

std::vector<Attr> QueryPackAttrs(int64_t bits)
{
    return {{"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bits)}};
}

std::vector<Attr> ScoreComputeAttrs(int64_t bits, int64_t maxKvLength)
{
    return {
        {"mse_bits", Ops::Transformer::AnyValue::CreateFrom<int64_t>(bits)},
        {"max_kv_len", Ops::Transformer::AnyValue::CreateFrom<int64_t>(maxKvLength)},
    };
}
} // namespace

TEST(TurboQuantAttentionScoreLayout, MetadataStrideMatchesKernelLoadLayout)
{
    EXPECT_EQ(tqv2::GetMetadataStride(4), 4U);
    EXPECT_EQ(tqv2::GetMetadataStride(16), 16U);
    EXPECT_EQ(tqv2::GetMetadataStride(17), 16U);
    EXPECT_EQ(tqv2::GetMetadataStride(32), 16U);
}

TEST(TurboQuantAttentionScoreInfershape, QueryPackAlignsTokens)
{
    gert::InfershapeContextPara para("TqQueryPack",
                                     {
                                         {{{17, 16, 128}, {17, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{17, 16, 128}, {17, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {{{{}, {}}, ge::DT_BF16, ge::FORMAT_ND}}, QueryPackAttrs(3));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{16, 128, 512}});
}

TEST(TurboQuantAttentionScoreInfershape, ScoreUsesBlockTableCapacity)
{
    gert::InfershapeContextPara para("TqScoreCompute",
                                     {
                                         {{{1, 16, 128}, {1, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{1, 16, 128}, {1, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{32, 128, 4, 48}, {32, 128, 4, 48}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{32, 128, 4, 16}, {32, 128, 4, 16}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{32, 128, 4}, {32, 128, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{32, 128, 4}, {32, 128, 4}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{1, 32}, {1, 32}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{1, 16, 128}, {1, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, ScoreComputeAttrs(3, -1));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{1, 16, 4096}});
}

TEST(TurboQuantAttentionScoreInfershape, ScoreUsesExplicitLength)
{
    gert::InfershapeContextPara para("TqScoreCompute",
                                     {
                                         {{{8, 16, 128}, {8, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{8, 16, 128}, {8, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{32, 128, 4, 64}, {32, 128, 4, 64}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{32, 128, 4, 16}, {32, 128, 4, 16}}, ge::DT_UINT8, ge::FORMAT_ND},
                                         {{{32, 128, 4}, {32, 128, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{32, 128, 4}, {32, 128, 4}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                         {{{8, 32}, {8, 32}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
                                         {{{8, 16, 128}, {8, 16, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {{{{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND}}, ScoreComputeAttrs(4, 1024));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{8, 16, 1024}});
}
