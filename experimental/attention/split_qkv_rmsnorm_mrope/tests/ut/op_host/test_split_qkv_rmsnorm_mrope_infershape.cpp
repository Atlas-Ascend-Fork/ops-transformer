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

#include "infer_shape_case_executor.h"

namespace {
using Attr = gert::InfershapeContextPara::OpAttr;

std::vector<Attr> MakeAttrs(int64_t qHeads, int64_t kvHeads, bool hasGate)
{
    return {
        {"num_q_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(qHeads)},
        {"num_kv_heads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(kvHeads)},
        {"eps", Ops::Transformer::AnyValue::CreateFrom<float>(1.0e-6F)},
        {"interleaved", Ops::Transformer::AnyValue::CreateFrom<bool>(true)},
        {"has_gate", Ops::Transformer::AnyValue::CreateFrom<bool>(hasGate)},
    };
}
} // namespace

TEST(SplitQkvRmsnormMropeInfershape, GateEnabled)
{
    gert::InfershapeContextPara para("SplitQkvRmsnormMrope",
                                     {
                                         {{{16, 10240}, {16, 10240}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{3, 16, 64}, {3, 16, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     MakeAttrs(16, 4, true));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{16, 4096}, {16, 1024}, {16, 1024}, {16, 4096}});
}

TEST(SplitQkvRmsnormMropeInfershape, GateDisabled)
{
    gert::InfershapeContextPara para("SplitQkvRmsnormMrope",
                                     {
                                         {{{9, 6144}, {9, 6144}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{3, 9, 64}, {3, 9, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     MakeAttrs(16, 4, false));
    ExecuteTestCase(para, ge::GRAPH_SUCCESS, {{9, 4096}, {9, 1024}, {9, 1024}, {9, 0}});
}

TEST(SplitQkvRmsnormMropeInfershape, RejectsNonTwoDimensionalQkv)
{
    gert::InfershapeContextPara para("SplitQkvRmsnormMrope",
                                     {
                                         {{{1, 16, 10240}, {1, 16, 10240}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{3, 16, 64}, {3, 16, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     MakeAttrs(16, 4, true));
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST(SplitQkvRmsnormMropeInfershape, RejectsUnsupportedHeadCount)
{
    gert::InfershapeContextPara para("SplitQkvRmsnormMrope",
                                     {
                                         {{{1, 18432}, {1, 18432}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{3, 1, 64}, {3, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     MakeAttrs(32, 4, true));
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}

TEST(SplitQkvRmsnormMropeInfershape, RejectsMismatchedQkvWidth)
{
    gert::InfershapeContextPara para("SplitQkvRmsnormMrope",
                                     {
                                         {{{1, 10239}, {1, 10239}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{256}, {256}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{3, 1, 64}, {3, 1, 64}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     {
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                         {{{}, {}}, ge::DT_BF16, ge::FORMAT_ND},
                                     },
                                     MakeAttrs(16, 4, true));
    ExecuteTestCase(para, ge::GRAPH_FAILED, {});
}
