/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <gtest/gtest.h>
#include "exe_graph/runtime/shape.h"
#include "../../../op_host/merge_chunk_shape.h"

namespace {
using MergeChunkSupport::Layout;
using MergeChunkSupport::ReadLayout;
using MergeChunkSupport::SUPPORTED_CHUNKS;
using MergeChunkSupport::SUPPORTED_PATHS;
constexpr int64_t FIRST_ROWS = 4;
constexpr int64_t SECOND_ROWS = 3;
constexpr int64_t FIRST_WIDTH = 16;
constexpr int64_t SECOND_WIDTH = 32;
constexpr int64_t EXPERTS = 3;

class MergeChunkLayoutTest : public testing::Test {
protected:
    const char* Read(Layout& layout, int64_t paths = SUPPORTED_PATHS, int64_t chunks = SUPPORTED_CHUNKS)
    {
        return ReadLayout(
            paths, chunks, [this](size_t i) { return i < data.size() ? &data[i] : nullptr; },
            [this](size_t i) { return i < counts.size() ? &counts[i] : nullptr; }, layout);
    }

    const char* Infer(MergeChunkSupport::InferredLayout& result)
    {
        return MergeChunkSupport::InferLayout([this](size_t i) { return i < data.size() ? &data[i] : nullptr; },
                                              [this](size_t i) { return i < counts.size() ? &counts[i] : nullptr; },
                                              result);
    }

    std::vector<gert::Shape> data{
        {FIRST_ROWS, FIRST_WIDTH}, {SECOND_ROWS, FIRST_WIDTH}, {FIRST_ROWS, SECOND_WIDTH}, {SECOND_ROWS, SECOND_WIDTH}};
    std::array<gert::Shape, SUPPORTED_CHUNKS> counts{{{EXPERTS}, {EXPERTS}}};
};

TEST_F(MergeChunkLayoutTest, ReadsTwoPathsAndTwoChunks)
{
    Layout layout;
    ASSERT_EQ(Read(layout), nullptr);
    EXPECT_EQ(layout.paths, SUPPORTED_PATHS);
    EXPECT_EQ(layout.chunks, SUPPORTED_CHUNKS);
    EXPECT_EQ(layout.experts, EXPERTS);
    EXPECT_EQ(layout.totalRows, FIRST_ROWS + SECOND_ROWS);
    EXPECT_EQ(layout.chunkRows[0], FIRST_ROWS);
    EXPECT_EQ(layout.chunkRows[1], SECOND_ROWS);
    EXPECT_EQ(layout.hidden[0], FIRST_WIDTH);
    EXPECT_EQ(layout.hidden[1], SECOND_WIDTH);
}

TEST_F(MergeChunkLayoutTest, RejectsMismatchedRowsAcrossPaths)
{
    Layout layout;
    data[SUPPORTED_CHUNKS + 1].SetDim(0, SECOND_ROWS - 1);
    EXPECT_NE(Read(layout), nullptr);
}

TEST_F(MergeChunkLayoutTest, RejectsMismatchedExpertCountsAcrossChunks)
{
    Layout layout;
    counts[1].SetDim(0, EXPERTS + 1);
    EXPECT_NE(Read(layout), nullptr);
}

TEST_F(MergeChunkLayoutTest, RejectsAnIncorrectListLength)
{
    Layout layout;
    data.pop_back();
    EXPECT_NE(Read(layout), nullptr);
}
TEST_F(MergeChunkLayoutTest, PropagatesUnknownDimensionsOnlyDuringInference)
{
    data = {{-1, -1}, {SECOND_ROWS, -1}, {-1, SECOND_WIDTH}, {SECOND_ROWS, SECOND_WIDTH}};
    counts = {{{-1}, {-1}}};
    MergeChunkSupport::InferredLayout inferred;
    ASSERT_EQ(Infer(inferred), nullptr);
    EXPECT_EQ(inferred.totalRows, -1);
    EXPECT_EQ(inferred.hidden[0], -1);
    EXPECT_EQ(inferred.hidden[1], SECOND_WIDTH);
    EXPECT_EQ(inferred.experts, -1);
    Layout runtime;
    EXPECT_NE(Read(runtime), nullptr);
    data[SUPPORTED_CHUNKS].SetDim(0, FIRST_ROWS);
    data[1].SetDim(1, FIRST_WIDTH);
    counts[1].SetDim(0, EXPERTS);
    ASSERT_EQ(Infer(inferred), nullptr);
    EXPECT_EQ(inferred.totalRows, FIRST_ROWS + SECOND_ROWS);
    EXPECT_EQ(inferred.hidden[0], FIRST_WIDTH);
    EXPECT_EQ(inferred.experts, EXPERTS);
}

TEST_F(MergeChunkLayoutTest, RejectsInvalidKnownWidthsDuringInference)
{
    MergeChunkSupport::InferredLayout inferred;
    for (const auto width : {0, 17, 90128}) {
        data[2].SetDim(1, -1);
        data[3].SetDim(1, width);
        EXPECT_NE(Infer(inferred), nullptr);
    }
}
} // namespace
