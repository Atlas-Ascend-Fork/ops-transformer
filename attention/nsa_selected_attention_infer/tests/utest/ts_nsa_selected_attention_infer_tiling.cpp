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
 * \file ts_nsa_selected_attention_infer_tiling.cpp
 * \brief NsaSelectedAttentionInfer tiling用例.
 */

#include "ts_nsa_selected_attention_infer.h"

namespace {

// 将 topkIndices 改为 3 维，覆盖 BSH 下 dim 校验失败分支（不改公共 param）
void PreTilingRunCbf_CorruptTopkTo3D(NsaSelectAttentionInferCase::DoTilingParam& tilingParam)
{
    if (tilingParam.topkIndicesTensor == nullptr) {
        return;
    }
    auto& storageShape = tilingParam.topkIndicesTensor->MutableStorageShape();
    // BSH 合法为 4 维 [B,S,N2,selectBlockCount]，压成 3 维 [B,S,selectBlockCount]
    if (storageShape.GetDimNum() >= 4) {
        storageShape.SetDimNum(3);
        storageShape.SetDim(0, 1);
        storageShape.SetDim(1, 1);
        storageShape.SetDim(2, 2);
    }
}

TEST_P(Ts_NsaSelectAttentionInfer_WithParam_Ascend910B1, Tc_Tiling_NsaSelectAttentionInfer)
{
    ASSERT_TRUE(case_->Init());
    ASSERT_EQ(case_->Run(), case_->mOpInfo.mExp.mSuccess);
}

const auto Tc_NsaSelectAttentionInfer_Tiling_Case = ::testing::Values(
    NsaSelectAttentionInferCase(
        "NsaSelectAttentionInfer_Case_q3_tnd", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),                 /* RunTiling, RunKernel */
               ExpectInfo(false,                         /* ExpectSuccess */
                          1,                             /* ExpectTilingKey */
                          48)),                          /* ExpectTilingBlockDim */
        NsaSelectAttentionInferParam(
            30, 3, 1, 192, 2, /* batchSize, qSeqSize, headSize, headDim, maxBlockNumPerBatch*/
            128, 256, 128, 1, /* blockSize kvSeqSize, headDimV, headSizeV*/
            128, 2, 0, 1.0f,  /* selectedBlockSize selectedBlockCount, sparseMode, scaleValue*/
            "TND",            /* inputLayout */
            ge::DT_FLOAT16,   /* optionalDataType */
            {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}, /* actualSeqQLenList */
            {256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
             256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256}, /* actualSeqKVLenList */
            90                                                                           /* numtokens */
            )),
    NsaSelectAttentionInferCase(
        "NsaSelectAttentionInfer_Case_q1_bsnd_fp16", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),                       /* RunTiling, RunKernel */
               ExpectInfo(false,                               /* ExpectSuccess */
                          0,                                   /* ExpectTilingKey */
                          48)),                                /* ExpectTilingBlockDim */
        NsaSelectAttentionInferParam(
            30, 1, 1, 192, 2, /* batchSize, qSeqSize, headSize, headDim, maxBlockNumPerBatch*/
            128, 256, 128, 1, /* blockSize kvSeqSize, headDimV, headSizeV*/
            128, 2, 0, 1.0f,  /* selectedBlockSize selectedBlockCount, sparseMode, scaleValue*/
            "BSND",           /* inputLayout */
            ge::DT_FLOAT16,   /* optionalDataType */
            {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
             3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}, /* actualSeqQLenList */
            {256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
             256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256}, /* actualSeqKVLenList */
            30                                                                           /* numtokens */
            )),
    // BSH + 4D topk：QKV 为 3 维，topkIndices 仍为 4 维，tiling 应成功
    NsaSelectAttentionInferCase("NsaSelectAttentionInfer_Case_q1_bsh_fp16_topk4d_success", true, "",
                                OpInfo(ControlInfo(true, false), ExpectInfo(true, 0, 48)),
                                NsaSelectAttentionInferParam(1, 1, 1, 192, 2, 128, 256, 128, 1, 128, 2, 0, 1.0f, "BSH",
                                                             ge::DT_FLOAT16, {1}, {256}, 1)),
    NsaSelectAttentionInferCase(
        "NsaSelectAttentionInfer_Case_q1_tnd_fp16", true, "", /* CaseName, Enable, DebugInfo */
        OpInfo(ControlInfo(true, false),                      /* RunTiling, RunKernel */
               ExpectInfo(false,                              /* ExpectSuccess */
                          1,                                  /* ExpectTilingKey */
                          48)),                               /* ExpectTilingBlockDim */
        NsaSelectAttentionInferParam(1, 1, 1, 192, 2,  /* batchSize, qSeqSize, headSize, headDim, maxBlockNumPerBatch*/
                                     128, 256, 128, 1, /* blockSize kvSeqSize, headDimV, headSizeV*/
                                     128, 2, 0, 1.0f,  /* selectedBlockSize selectedBlockCount, sparseMode, scaleValue*/
                                     "TND",            /* inputLayout */
                                     ge::DT_FLOAT16,   /* optionalDataType */
                                     {1},              /* actualSeqQLenList */
                                     {256},            /* actualSeqKVLenList */
                                     1                 /* numtokens */
                                     )));

INSTANTIATE_TEST_SUITE_P(NsaSelectedAttentionInfer, Ts_NsaSelectAttentionInfer_WithParam_Ascend910B1,
                         Tc_NsaSelectAttentionInfer_Tiling_Case);

// BSH + 3D topk：tiling 维度校验失败（对应 issue 6615）
TEST_F(Ts_NsaSelectAttentionInfer_Ascend910B1, Tc_Tiling_Bsh_Topk3d_Fail)
{
    NsaSelectAttentionInferCase tilingCase("NsaSelectAttentionInfer_Case_q1_bsh_fp16_topk3d_fail", true, "",
                                           OpInfo(ControlInfo(true, false), ExpectInfo(false, 0, 48)),
                                           NsaSelectAttentionInferParam(1, 1, 1, 192, 2, 128, 256, 128, 1, 128, 2, 0,
                                                                        1.0f, "BSH", ge::DT_FLOAT16, {1}, {256}, 1));
    tilingCase.mPreTilingRunCbf = PreTilingRunCbf_CorruptTopkTo3D;
    ASSERT_TRUE(tilingCase.Init());
    ASSERT_EQ(tilingCase.Run(), false);
}

} // namespace
