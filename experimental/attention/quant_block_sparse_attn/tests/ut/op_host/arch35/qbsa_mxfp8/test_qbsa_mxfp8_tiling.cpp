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
#include <iostream>
#include <stdexcept>
#include <set>
#include "../../../../../op_kernel/arch35/common/quant_block_sparse_attn_const.h"

#include "../../test_utils/qbsa_tiling_case_executor.h"

namespace QuantBlockSparseAttnUT {

class QbsaMxfp8Arch35TilingTest : public testing::TestWithParam<QuantBlockSparseAttnTilingUtParam> {};

TEST_P(QbsaMxfp8Arch35TilingTest, Tiling)
{
    const auto param = GetParam();
    std::cout << "[TEST_CASE] " << param.case_name << std::endl;
    ASSERT_EQ(param.quant_mode, optiling::QBSA_QUANT_MODE_MXFP8_FULL_QUANT);

    TilingInfo tilingInfo;
    const bool ok = ExecuteQbsaTilingCase(param, tilingInfo, sizeof(optiling::QuantBlockSparseAttnMxfp8TilingData));
    if (param.expectResult != ge::GRAPH_SUCCESS) {
        EXPECT_FALSE(ok);
        return;
    }

    ASSERT_TRUE(ok);
    EXPECT_GT(tilingInfo.blockNum, 0U);
    const auto& queryShape = param.query.shape_.GetStorageShape();
    const uint32_t headDim = queryShape.GetDim(queryShape.GetDimNum() - 1U);
    const uint32_t layout =
        param.layout_q == "BSND" ? QBSA_LAYOUT_BSND : (param.layout_q == "BNSD" ? QBSA_LAYOUT_BNSD : QBSA_LAYOUT_TND);
    EXPECT_EQ(tilingInfo.tilingKey, static_cast<int64_t>(optiling::BuildQbsaMxfp8TilingKey(
                                        layout, param.mask_mode, param.return_softmax_lse, headDim)));
    ASSERT_EQ(tilingInfo.tilingDataSize, sizeof(optiling::QuantBlockSparseAttnMxfp8TilingData));
    const auto data = DecodeQbsaTilingData<optiling::QuantBlockSparseAttnMxfp8TilingData>(tilingInfo);
    ExpectQbsaCommonTilingData(param, tilingInfo, data.commonParams);
    const auto queryHeads = param.query.shape_.GetStorageShape().GetDim(param.layout_q == "BSND" ? 2U : 1U);
    const auto kvHeads = param.key.shape_.GetStorageShape().GetDim(1U);
    ASSERT_GT(kvHeads, 0);
    EXPECT_EQ(data.commonParams.baseParams.n2Size, kvHeads);
    EXPECT_EQ(data.commonParams.baseParams.gSize, queryHeads / kvHeads);
    EXPECT_EQ(data.mxParams.scaleGroupSize, 64U);
    EXPECT_EQ(data.mxParams.scaleLastDim, 2U);
    EXPECT_EQ(data.mxParams.queryScaleDSize, headDim / 64U);
    EXPECT_EQ(data.mxParams.keyScaleDSize, headDim / 64U);
    EXPECT_EQ(data.mxParams.kScaleStrides.n2Stride,
              static_cast<uint64_t>(data.commonParams.paParams.blockSize) * (headDim / 64U) * 2U);
    EXPECT_EQ(data.mxParams.vScaleStrides.n2Stride,
              static_cast<uint64_t>(data.commonParams.paParams.blockSize) / 64U * headDim * 2U);

    auto omittedParam = param;
    TilingInfo omittedInfo;
    ASSERT_TRUE(ExecuteQbsaTilingCase(omittedParam, omittedInfo, 4096U, true));
    EXPECT_EQ(omittedInfo.tilingKey, tilingInfo.tilingKey);
    ASSERT_EQ(omittedInfo.tilingDataSize, tilingInfo.tilingDataSize);
    EXPECT_EQ(std::memcmp(omittedInfo.tilingData.get(), tilingInfo.tilingData.get(), tilingInfo.tilingDataSize), 0);

    auto unsupportedLayout = param;
    unsupportedLayout.layout_q = "NTD";
    unsupportedLayout.layout_out = "TND";
    TilingInfo unsupportedLayoutInfo;
    EXPECT_FALSE(ExecuteQbsaTilingCase(unsupportedLayout, unsupportedLayoutInfo));

    // A valid case must fit its own payload exactly and reject an undersized buffer.
    TilingInfo undersizedTilingInfo;
    EXPECT_FALSE(
        ExecuteQbsaTilingCase(param, undersizedTilingInfo, sizeof(optiling::QuantBlockSparseAttnMxfp8TilingData) - 1U));
}

INSTANTIATE_TEST_SUITE_P(
    QbsaMxfp8Arch35Tiling, QbsaMxfp8Arch35TilingTest,
    testing::ValuesIn(GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<QuantBlockSparseAttnTilingUtParam>);

static QuantBlockSparseAttnTilingUtParam MakePaddedStrideCase(const std::string& layout, int64_t batch,
                                                              int64_t sequence, int64_t heads)
{
    const auto cases = GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__));
    for (auto param : cases) {
        if (param.case_name != "mxfp8_tiling_bnsd_bns_lse") {
            continue;
        }
        const int64_t dim1 = layout == "BSND" ? sequence : heads;
        const int64_t dim2 = layout == "BSND" ? heads : sequence;
        param.layout_q = layout;
        param.layout_out = layout;
        param.mask_mode = 0;
        param.attenMask = TensorDesc(gert::StorageShape({0}, {0}), ge::DT_UINT8, ge::FORMAT_ND);
        param.query = TensorDesc(gert::StorageShape({batch, dim1, dim2, 128}, {batch, dim1, dim2, 128}),
                                 ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND);
        param.qDescale = TensorDesc(gert::StorageShape({batch, dim1, dim2, 2, 2}, {batch, dim1, dim2, 2, 2}),
                                    ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
        param.key = TensorDesc(gert::StorageShape({4, heads, 128, 128}, {4, heads, 128, 128}), ge::DT_FLOAT8_E4M3FN,
                               ge::FORMAT_ND);
        param.value = param.key;
        param.kDescale = TensorDesc(gert::StorageShape({4, heads, 128, 2, 2}, {4, heads, 128, 2, 2}),
                                    ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
        param.vDescale = TensorDesc(gert::StorageShape({4, heads, 2, 128, 2}, {4, heads, 2, 128, 2}),
                                    ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
        param.sequsedKv = TensorDesc(gert::StorageShape({batch}, {batch}), ge::DT_INT32, ge::FORMAT_ND);
        param.sparseIndices =
            TensorDesc(gert::StorageShape({batch, heads, 1, 4}, {batch, heads, 1, 4}), ge::DT_INT32, ge::FORMAT_ND);
        param.sparseSeqLen =
            TensorDesc(gert::StorageShape({batch, heads, 1}, {batch, heads, 1}), ge::DT_INT32, ge::FORMAT_ND);
        param.blockTable = TensorDesc(gert::StorageShape({batch, 8}, {batch, 8}), ge::DT_INT32, ge::FORMAT_ND);
        param.attentionOut = TensorDesc(gert::StorageShape({batch, dim1, dim2, 128}, {batch, dim1, dim2, 128}),
                                        ge::DT_BF16, ge::FORMAT_ND);
        param.softmaxLse = TensorDesc(gert::StorageShape({batch, heads, sequence}, {batch, heads, sequence}),
                                      ge::DT_FLOAT, ge::FORMAT_ND);
        for (auto* input : {&param.query, &param.qDescale}) {
            const auto& shape = input->shape_.GetStorageShape();
            int64_t stride = 1;
            for (size_t i = shape.GetDimNum(); i > 0U; --i) {
                input->stride_.SetStride(i - 1U, stride);
                stride *= shape.GetDim(i - 1U);
            }
            input->stride_.SetDimNum(shape.GetDimNum());
            input->hasStride_ = true;
        }
        return param;
    }
    throw std::runtime_error("Missing padded-layout baseline tiling case");
}

TEST(QuantBlockSparseAttnPaddedStrideTest, AcceptsSingletonDimensionStrides)
{
    // Includes LAYOUT006 (B=2, N=1, S=31), singleton B/S, and a non-singleton control.
    const int64_t dimensions[][3] = {{2, 31, 1}, {1, 31, 4}, {2, 1, 4}, {2, 31, 4}};
    for (const std::string layout : {"BSND", "BNSD"}) {
        for (const auto& dims : dimensions) {
            // Exercise Q and QScale independently, then together; mode 0 is canonical.
            for (uint32_t mode = 0U; mode < 4U; ++mode) {
                SCOPED_TRACE(layout + " B=" + std::to_string(dims[0]) + " S=" + std::to_string(dims[1]) +
                             " N=" + std::to_string(dims[2]) + " mode=" + std::to_string(mode));
                auto param = MakePaddedStrideCase(layout, dims[0], dims[1], dims[2]);
                TensorDesc* inputs[] = {&param.query, &param.qDescale};
                for (size_t inputIndex = 0U; inputIndex < 2U; ++inputIndex) {
                    if ((mode & (1U << inputIndex)) == 0U) {
                        continue;
                    }
                    auto& input = *inputs[inputIndex];
                    const auto& shape = input.shape_.GetStorageShape();
                    for (size_t dim = 0U; dim < shape.GetDimNum(); ++dim) {
                        if (shape.GetDim(dim) == 1) {
                            // A singleton index is always zero, so its stride cannot change an address.
                            input.stride_.SetStride(dim, inputIndex == 0U ? 128 : 4);
                        }
                    }
                }
                TilingInfo tilingInfo;
                EXPECT_TRUE(ExecuteQbsaTilingCase(param, tilingInfo));
            }
        }
    }
}

TEST(QuantBlockSparseAttnPaddedStrideTest, RejectsNonContiguousInputsAndWrongStrideRank)
{
    for (const std::string layout : {"BSND", "BNSD"}) {
        for (size_t inputIndex = 0U; inputIndex < 2U; ++inputIndex) {
            // Keep N=1 so the relaxed singleton rule must not hide a real stride error.
            auto baseline = MakePaddedStrideCase(layout, 2, 31, 1);
            const auto& shape = (inputIndex == 0U ? baseline.query : baseline.qDescale).shape_.GetStorageShape();
            for (size_t dim = 0U; dim < shape.GetDimNum(); ++dim) {
                if (shape.GetDim(dim) == 1) {
                    continue;
                }
                auto param = baseline;
                auto& input = inputIndex == 0U ? param.query : param.qDescale;
                input.stride_.SetStride(dim, input.stride_.GetStride(dim) + 1);
                TilingInfo tilingInfo;
                EXPECT_FALSE(ExecuteQbsaTilingCase(param, tilingInfo));
            }
            auto& input = inputIndex == 0U ? baseline.query : baseline.qDescale;
            input.stride_.SetDimNum(shape.GetDimNum() - 1U);
            TilingInfo tilingInfo;
            EXPECT_FALSE(ExecuteQbsaTilingCase(baseline, tilingInfo));
        }
    }
}

TEST(QbsaMxfp8OptionalInputs, SparseInputsRemainAccessibleAfterOmittedOptionalSlots)
{
    const auto cases = GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__));
    for (auto param : cases) {
        if (param.case_name != "mxfp8_tiling_basic") {
            continue;
        }
        // Match the reported TND case: no P scale, cu_seqlens_kv, seqused_q or mask.
        param.mask_mode = 0;
        param.pScale = TensorDesc(gert::StorageShape({0}, {0}), ge::DT_FLOAT8_E8M0, ge::FORMAT_ND);
        param.attenMask = TensorDesc(gert::StorageShape({0}, {0}), ge::DT_UINT8, ge::FORMAT_ND);
        TilingInfo emptyTensorInfo;
        ASSERT_TRUE(ExecuteQbsaTilingCase(param, emptyTensorInfo));
        TilingInfo omittedInputInfo;
        ASSERT_TRUE(ExecuteQbsaTilingCase(param, omittedInputInfo, 4096U, true));
        EXPECT_EQ(omittedInputInfo.tilingKey, emptyTensorInfo.tilingKey);
        ASSERT_EQ(omittedInputInfo.tilingDataSize, emptyTensorInfo.tilingDataSize);
        EXPECT_EQ(std::memcmp(omittedInputInfo.tilingData.get(), emptyTensorInfo.tilingData.get(),
                              emptyTensorInfo.tilingDataSize),
                  0);
        // Missing required sparse inputs must still be rejected after the fix.
        auto missingIndices = param;
        missingIndices.sparseIndices = TensorDesc(gert::StorageShape(), ge::DT_INT32, ge::FORMAT_ND);
        TilingInfo missingIndicesInfo;
        EXPECT_FALSE(ExecuteQbsaTilingCase(missingIndices, missingIndicesInfo, 4096U, true));
        auto missingLengths = param;
        missingLengths.sparseSeqLen = TensorDesc(gert::StorageShape(), ge::DT_INT32, ge::FORMAT_ND);
        TilingInfo missingLengthsInfo;
        EXPECT_FALSE(ExecuteQbsaTilingCase(missingLengths, missingLengthsInfo, 4096U, true));
        return;
    }
    FAIL() << "Missing MXFP8 baseline tiling case";
}

TEST(QbsaMxfp8TilingKey, EveryMxLayoutHeadMaskLseCombinationHasADistinctKey)
{
    EXPECT_EQ(QBSA_LAYOUT_TND, static_cast<uint32_t>(QBSALayout::TND));
    EXPECT_EQ(QBSA_LAYOUT_BSND, static_cast<uint32_t>(QBSALayout::BSND));
    EXPECT_EQ(QBSA_LAYOUT_BNSD, static_cast<uint32_t>(QBSALayout::BNSD));
    std::set<uint64_t> keys;
    for (const uint32_t layout : {QBSA_LAYOUT_TND, QBSA_LAYOUT_BSND, QBSA_LAYOUT_BNSD}) {
        for (const uint32_t headDim : {64U, 128U, 256U}) {
            for (const uint32_t mask : {0U, 3U}) {
                for (const bool lse : {false, true}) {
                    EXPECT_TRUE(keys.insert(optiling::BuildQbsaMxfp8TilingKey(layout, mask, lse, headDim)).second);
                }
            }
        }
    }
    EXPECT_EQ(keys.size(), 36U);
}
TEST(QbsaMxfp8Architecture, RejectsPaddedQueryOnArch92)
{
    const auto param = MakePaddedStrideCase("BNSD", 2, 31, 4);
    TilingInfo a5;
    TilingInfo a6;
    EXPECT_TRUE(ExecuteQbsaTilingCase(param, a5));
    EXPECT_FALSE(ExecuteQbsaTilingCase(param, a6, 4096U, false, "Ascend960dt"));
}

TEST(QbsaMxfp8Architecture, Arch92UsesMxPayloadForSupportedTndCase)
{
    const auto cases = GetCasesFromCsv<QuantBlockSparseAttnTilingUtParam>(ReplaceFileExtension2Csv(__FILE__));
    ASSERT_FALSE(cases.empty());
    const auto param = cases.front();
    TilingInfo a5;
    TilingInfo a6;
    ASSERT_TRUE(ExecuteQbsaTilingCase(param, a5));
    ASSERT_TRUE(ExecuteQbsaTilingCase(param, a6, 4096U, false, "Ascend960dt"));
    EXPECT_EQ(a5.tilingKey, a6.tilingKey);
    EXPECT_EQ(a6.tilingDataSize, sizeof(optiling::QuantBlockSparseAttnMxfp8TilingData));
    const auto data = DecodeQbsaTilingData<optiling::QuantBlockSparseAttnMxfp8TilingData>(a6);
    EXPECT_EQ(data.commonParams.baseParams.coreNum, a6.blockNum);
    EXPECT_LE(a6.blockNum, 64U);
    EXPECT_EQ(data.commonParams.baseParams.dSize, 128U);
    EXPECT_EQ(data.mxParams.kScaleStrides.n2Stride, static_cast<uint64_t>(data.commonParams.paParams.blockSize) * 4U);
}

} // namespace QuantBlockSparseAttnUT
