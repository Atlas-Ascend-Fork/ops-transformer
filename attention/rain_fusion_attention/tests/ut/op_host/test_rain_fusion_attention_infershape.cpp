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
 * \file test_rain_fusion_attention_infershape.cpp
 * \brief UT for RainFusionAttention infershape. Each negative case targets exactly one OP_LOGE report branch.
 */
#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>

#include "infer_shape_context_faker.h"
#include "infer_shape_case_executor.h"

using namespace std;
using namespace ge;

namespace {

struct RfaIsCase {
    std::string qLayout = "TND";
    std::string kvLayout = "TND";
    int64_t numKvHeads = 2;
    // TND
    int64_t t = 4;
    int64_t kvT = 4;
    // BNSD
    int64_t b = 2;
    int64_t s = 2;
    int64_t kvS = 2;
    int64_t n = 8;
    int64_t kvN = 2;
    int64_t d = 128;
    // 覆盖key/value维度的开关
    int64_t keyD = -1;   // -1表示与d一致
    int64_t valueD = -1; // -1表示与keyD(或d)一致
};

gert::InfershapeContextPara BuildPara(const RfaIsCase& c)
{
    bool isBnsd = (c.qLayout == "BNSD");
    int64_t realKeyD = (c.keyD == -1) ? c.d : c.keyD;
    int64_t realValueD = (c.valueD == -1) ? realKeyD : c.valueD;

    // 输入：0 query / 1 key / 2 value
    gert::InfershapeContextPara::TensorDescription queryDesc({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    if (isBnsd) {
        queryDesc = gert::InfershapeContextPara::TensorDescription({{c.b, c.n, c.s, c.d}, {c.b, c.n, c.s, c.d}},
                                                                   ge::DT_FLOAT16, ge::FORMAT_ND);
    } else {
        queryDesc = gert::InfershapeContextPara::TensorDescription({{c.t, c.n, c.d}, {c.t, c.n, c.d}}, ge::DT_FLOAT16,
                                                                   ge::FORMAT_ND);
    }
    gert::InfershapeContextPara::TensorDescription keyDesc({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription valueDesc({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    if (isBnsd) {
        keyDesc = gert::InfershapeContextPara::TensorDescription(
            {{c.b, c.kvN, c.kvS, realKeyD}, {c.b, c.kvN, c.kvS, realKeyD}}, ge::DT_FLOAT16, ge::FORMAT_ND);
        valueDesc = gert::InfershapeContextPara::TensorDescription(
            {{c.b, c.kvN, c.kvS, realValueD}, {c.b, c.kvN, c.kvS, realValueD}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    } else {
        keyDesc = gert::InfershapeContextPara::TensorDescription({{c.kvT, c.kvN, realKeyD}, {c.kvT, c.kvN, realKeyD}},
                                                                 ge::DT_FLOAT16, ge::FORMAT_ND);
        valueDesc = gert::InfershapeContextPara::TensorDescription(
            {{c.kvT, c.kvN, realValueD}, {c.kvT, c.kvN, realValueD}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    }
    std::vector<gert::InfershapeContextPara::TensorDescription> inputs = {queryDesc, keyDesc, valueDesc};

    // 输出：0 attentionOut / 1 softmaxLse
    gert::InfershapeContextPara::TensorDescription attentionOutDesc({{}, {}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    gert::InfershapeContextPara::TensorDescription softmaxLseDesc({{}, {}}, ge::DT_FLOAT, ge::FORMAT_ND);
    std::vector<gert::InfershapeContextPara::TensorDescription> outputs = {attentionOutDesc, softmaxLseDesc};

    std::vector<gert::InfershapeContextPara::OpAttr> attrs = {
        {"qInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>(c.qLayout)},
        {"kvInputLayout", Ops::Transformer::AnyValue::CreateFrom<std::string>(c.kvLayout)},
        {"numKeyValueHeads", Ops::Transformer::AnyValue::CreateFrom<int64_t>(c.numKvHeads)}};

    return gert::InfershapeContextPara("RainFusionAttention", inputs, outputs, attrs);
}

void RunCase(const RfaIsCase& c, ge::graphStatus expectStatus,
             const std::vector<std::vector<int64_t>>& expectOutputShape = {})
{
    auto para = BuildPara(c);
    ExecuteTestCase(para, expectStatus, expectOutputShape);
}
} // namespace

class RainFusionAttentionInfershape : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "--- RainFusionAttentionInfershape UT SetUp ---" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "--- RainFusionAttentionInfershape UT TearDown ---" << std::endl;
    }
};

// ============================ 成功用例 ============================

// TND格式成功用例
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_tnd_success)
{
    RfaIsCase c;
    RunCase(c, ge::GRAPH_SUCCESS, {{c.t, c.n, c.d}, {c.t, c.n}});
}

// BNSD格式成功用例
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_bnsd_success)
{
    RfaIsCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    RunCase(c, ge::GRAPH_SUCCESS, {{c.b, c.n, c.s, c.d}, {c.b, c.n, c.s}});
}

// ============================ 负向用例：算子逻辑报错 ============================

// 对应 InferShape（TND query）：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("query", "Layout TND, queryDims must be 3")
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_tnd_query_dims_invalid)
{
    RfaIsCase c;
    // 手工构造4维query（TND layout）
    auto para = BuildPara(c);
    para.inputTensorDesc_[0] = gert::InfershapeContextPara::TensorDescription(
        {{c.b, c.n, c.s, c.d}, {c.b, c.n, c.s, c.d}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 对应 InferShape（BNSD query）：OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("query", "Layout BNSD, queryDims must be 4")
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_bnsd_query_dims_invalid)
{
    RfaIsCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    auto para = BuildPara(c);
    // 手工构造3维query（BNSD layout）
    para.inputTensorDesc_[0] = gert::InfershapeContextPara::TensorDescription({{c.t, c.n, c.d}, {c.t, c.n, c.d}},
                                                                              ge::DT_FLOAT16, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 对应 InferShape：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("qInputLayout", "Only TND and BNSD are supported")
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_q_layout_invalid)
{
    RfaIsCase c;
    c.qLayout = "BSHD";
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 InferShape：OP_LOGE_FOR_INVALID_VALUES_WITH_REASON("qInputLayout, kvInputLayout", layout必须一致)
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_layout_mismatch)
{
    RfaIsCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "TND";
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 InferShape（TND KV）：OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON("key and value", "Layout TND, KV dims must be
// 3")
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_tnd_kv_dims_invalid)
{
    RfaIsCase c;
    auto para = BuildPara(c);
    // 手工构造4维key
    para.inputTensorDesc_[1] = gert::InfershapeContextPara::TensorDescription(
        {{c.b, c.kvN, c.kvS, c.d}, {c.b, c.kvN, c.kvS, c.d}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 对应 InferShape（TND）：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("numKeyValueHeads", 与key的N维不匹配)
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_tnd_kv_heads_mismatch)
{
    RfaIsCase c;
    c.kvN = 4; // attr numKeyValueHeads=2
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 InferShape（TND）：OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("key and value", K/V head dim不匹配)
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_tnd_kv_d_mismatch)
{
    RfaIsCase c;
    c.keyD = 128;
    c.valueD = 64;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 InferShape（BNSD KV）：OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON("key and value", "Layout BNSD, KV dims must be
// 4")
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_bnsd_kv_dims_invalid)
{
    RfaIsCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    auto para = BuildPara(c);
    // 手工构造3维key
    para.inputTensorDesc_[1] = gert::InfershapeContextPara::TensorDescription(
        {{c.kvT, c.kvN, c.d}, {c.kvT, c.kvN, c.d}}, ge::DT_FLOAT16, ge::FORMAT_ND);
    ExecuteTestCase(para, ge::GRAPH_FAILED);
}

// 对应 InferShape（BNSD）：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("numKeyValueHeads", 与key的N维不匹配)
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_bnsd_kv_heads_mismatch)
{
    RfaIsCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.kvN = 4; // attr numKeyValueHeads=2
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 InferShape（BNSD）：OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON("key and value", K/V head dim不匹配)
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_bnsd_kv_d_mismatch)
{
    RfaIsCase c;
    c.qLayout = "BNSD";
    c.kvLayout = "BNSD";
    c.keyD = 128;
    c.valueD = 64;
    RunCase(c, ge::GRAPH_FAILED);
}

// 对应 InferShape：OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("kvInputLayout", "Only TND and BNSD are supported")
TEST_F(RainFusionAttentionInfershape, rain_fusion_attention_infershape_kv_layout_invalid)
{
    RfaIsCase c;
    c.kvLayout = "BSHD";
    RunCase(c, ge::GRAPH_FAILED);
}
