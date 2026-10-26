/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "tiling_case_executor.h"

class SwinAttentionScoreQuantTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "SwinAttentionScoreQuantTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "SwinAttentionScoreQuantTiling TearDown" << std::endl;
    }
};

namespace {

struct SwinAttentionScoreQuantCompileInfo {
    uint32_t coreNum = 8;
    uint64_t ubSize = 196608;
    uint64_t l1Size = 524288;
    uint64_t l2Size = 33554432;
};

// 参数化tiling用例输入，默认值为可成功tiling的合法输入
struct TilingCase {
    int64_t b = 2048;
    int64_t n = 3;
    int64_t s = 49;
    int64_t h = 32;
    bool query_present = true;
    bool key_present = true;
    bool scale_present = true;
    bool bias_present = true;
    bool mask1_present = true;
    int64_t key_s = -1;         // key的S维，-1表示与query相同
    int64_t value_n = -1;       // value的N维，-1表示与query相同
    int64_t query_rank = 4;     // query的维度数
    int64_t key_rank = 4;       // key/value的维度数
    int64_t mask_rank = 4;      // padding_mask1的维度数
    int64_t mask_dim0 = 1;      // padding_mask1的第0维
    int64_t mask_n = -1;        // padding_mask1的N维，-1表示与query相同
    int64_t mask_last_dim = -1; // padding_mask1的最后一维，-1表示与query的S相同
};

gert::StorageShape ToStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape storageShape;
    gert::Shape shape;
    for (int64_t dim : dims) {
        shape.AppendDim(dim);
    }
    storageShape.MutableOriginShape() = shape;
    storageShape.MutableStorageShape() = shape;
    return storageShape;
}

gert::TilingContextPara::TensorDescription MakeTensorDesc(const std::vector<int64_t>& dims, ge::DataType dataType,
                                                          bool present)
{
    if (!present) {
        return gert::TilingContextPara::TensorDescription(gert::StorageShape(), dataType, ge::FORMAT_ND);
    }
    return gert::TilingContextPara::TensorDescription(ToStorageShape(dims), dataType, ge::FORMAT_ND);
}

gert::TilingContextPara BuildTilingPara(const TilingCase& tc)
{
    static SwinAttentionScoreQuantCompileInfo compileInfo;
    const int64_t keyS = tc.key_s < 0 ? tc.s : tc.key_s;
    const int64_t valueN = tc.value_n < 0 ? tc.n : tc.value_n;
    const int64_t maskN = tc.mask_n < 0 ? tc.n : tc.mask_n;
    const int64_t maskLastDim = tc.mask_last_dim < 0 ? tc.s : tc.mask_last_dim;

    std::vector<int64_t> qDims =
        tc.query_rank == 4 ? std::vector<int64_t>{tc.b, tc.n, tc.s, tc.h} : std::vector<int64_t>{tc.b, tc.n, tc.s};
    std::vector<int64_t> kDims =
        tc.key_rank == 4 ? std::vector<int64_t>{tc.b, tc.n, keyS, tc.h} : std::vector<int64_t>{tc.b, tc.n, keyS};
    std::vector<int64_t> vDims =
        tc.key_rank == 4 ? std::vector<int64_t>{tc.b, valueN, tc.s, tc.h} : std::vector<int64_t>{tc.b, valueN, tc.s};
    std::vector<int64_t> maskDims = tc.mask_rank == 4 ? std::vector<int64_t>{tc.mask_dim0, maskN, tc.s, maskLastDim} :
                                                        std::vector<int64_t>{maskN, tc.s, maskLastDim};

    return gert::TilingContextPara(
        "SwinAttentionScoreQuant",
        {
            MakeTensorDesc(qDims, ge::DT_INT8, tc.query_present),
            MakeTensorDesc(kDims, ge::DT_INT8, tc.key_present),
            MakeTensorDesc(vDims, ge::DT_INT8, true),
            MakeTensorDesc({1, tc.s}, ge::DT_FLOAT16, tc.scale_present),
            MakeTensorDesc({1, tc.s}, ge::DT_UINT64, true),
            MakeTensorDesc({1, tc.h}, ge::DT_UINT64, true),
            MakeTensorDesc({1, tc.s}, ge::DT_FLOAT16, tc.bias_present),
            MakeTensorDesc({1, tc.s}, ge::DT_INT32, tc.bias_present),
            MakeTensorDesc({1, tc.h}, ge::DT_INT32, tc.bias_present),
            MakeTensorDesc(maskDims, ge::DT_FLOAT16, tc.mask1_present),
            MakeTensorDesc({1, tc.n, tc.s, tc.s}, ge::DT_FLOAT16, true),
        },
        {{{{tc.b, tc.n, tc.s, tc.h}, {tc.b, tc.n, tc.s, tc.h}}, ge::DT_FLOAT16, ge::FORMAT_ND}},
        {
            {"query_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"key_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"value_transpose", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
            {"softmax_axes", Ops::Transformer::AnyValue::CreateFrom<int64_t>(-1)},
        },
        &compileInfo);
}

void ExpectTilingResult(const gert::TilingContextPara& para, bool expect_success)
{
    TilingInfo tilingInfo;
    bool ok = ExecuteTiling(para, tilingInfo);
    EXPECT_EQ(ok, expect_success);
    if (expect_success) {
        EXPECT_NE(tilingInfo.tilingKey, -1);
        EXPECT_FALSE(tilingInfo.workspaceSizes.empty());
    }
}

} // namespace

// 正向：带mask
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_with_mask_001)
{
    ExpectTilingResult(BuildTilingPara(TilingCase{}), true);
}

// 正向：不带mask
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_without_mask_002)
{
    TilingCase tc;
    tc.mask1_present = false;
    ExpectTilingResult(BuildTilingPara(tc), true);
}

// 负向：bias_quant（实现上必填）为空，EZ0004
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_null_bias_003)
{
    TilingCase tc;
    tc.bias_present = false;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：query的S维超上限1024，EZ0012
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_s_004)
{
    TilingCase tc;
    tc.s = 1025;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：query的H维不是32/64，EZ0012
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_h_005)
{
    TilingCase tc;
    tc.h = 33;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：key的S维与query不一致，EZ0013
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_key_006)
{
    TilingCase tc;
    tc.key_s = 50;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：value的N维与query不一致，EZ0013
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_value_007)
{
    TilingCase tc;
    tc.value_n = 4;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：padding_mask1最后一维与query的S维不一致，EZ0013
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_mask_008)
{
    TilingCase tc;
    tc.mask_last_dim = 50;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：query维度数不为4，EZ0012
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_query_rank_009)
{
    TilingCase tc;
    tc.query_rank = 3;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：key维度数不为4，EZ0012
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_key_rank_010)
{
    TilingCase tc;
    tc.key_rank = 3;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：B*N*S超过uint32上限，EZ0015
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_size_overflow_011)
{
    TilingCase tc;
    tc.b = 4194304;
    tc.n = 1;
    tc.s = 1024;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：query缺失。UT harness会剔除空TensorDescription导致必选输入索引压缩，
// 实际首错为索引偏移后value取到scale_quant的EZ0012维度校验；本用例回归验证query缺失时tiling拒绝
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_null_query_012)
{
    TilingCase tc;
    tc.query_present = false;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：key缺失。UT harness会剔除空TensorDescription导致必选输入索引压缩，
// 实际首错为索引偏移后value取到scale_quant的EZ0012维度校验；本用例回归验证key缺失时tiling拒绝
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_null_key_013)
{
    TilingCase tc;
    tc.key_present = false;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：必选输入scale_quant为空，EZ0004
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_null_scale_014)
{
    TilingCase tc;
    tc.scale_present = false;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：padding_mask1维度数不为4，EZ0012
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_mask_rank_015)
{
    TilingCase tc;
    tc.mask_rank = 3;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：padding_mask1第0维不为1，EZ0012
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_mask_dim0_016)
{
    TilingCase tc;
    tc.mask_dim0 = 2;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：padding_mask1的N维与query不一致，EZ0013
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_wrong_mask_n_017)
{
    TilingCase tc;
    tc.mask_n = 4;
    ExpectTilingResult(BuildTilingPara(tc), false);
}

// 负向：S=1边界值，探测matmul tiling失败分支（若为合法输入则应成功）
TEST_F(SwinAttentionScoreQuantTiling, swin_attention_score_quant_tiling_s_one_018)
{
    TilingCase tc;
    tc.s = 1;
    ExpectTilingResult(BuildTilingPara(tc), true);
}
