/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2.h"

namespace optiling {

class FlashAttentionGradTilingSBn2gs1s2 : public FlashAttentionScoreGradTilingUs1s2Bs2 {
public:
    explicit FlashAttentionGradTilingSBn2gs1s2(gert::TilingContext* context_)
        : FlashAttentionScoreGradTilingUs1s2Bs2(context_)
    {}
    ~FlashAttentionGradTilingSBn2gs1s2() {};

protected:
    // override base
    bool IsCapable() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;

    void DoSplit() override;
    void DoSparse() override;
    ge::graphStatus SaveToTilingData() override;

    // new method
    ge::graphStatus GetSparseSplitS1BlockInfo();
    ge::graphStatus GetSparseSplitS2BlockInfo();
    void FillSplitS1BlockInfo(std::vector<std::vector<std::vector<uint32_t>>>& calculatedBlockInfo,
                              std::vector<std::vector<uint32_t>>& totalBlockInfo);

    void FillSplitS2BlockInfo(std::vector<std::vector<std::vector<uint32_t>>>& calculatedBlockInfo,
                              std::vector<std::vector<uint32_t>>& totalBlockInfo);
    void SplitS1Naive(std::vector<std::vector<std::vector<uint32_t>>> calculatedBlockInfo,
                      std::vector<std::vector<uint32_t>> totalBlockInfo,
                      std::vector<std::vector<uint32_t>> blockCluster, std::vector<std::vector<uint32_t>> bIdxCluster);
    void SplitS1Balanced(std::vector<std::vector<std::vector<uint32_t>>> calculatedBlockInfo,
                         std::vector<std::vector<uint32_t>> totalBlockInfo,
                         std::vector<std::vector<uint32_t>> blockCluster,
                         std::vector<std::vector<uint32_t>> bIdxCluster);
    FuzzyBaseInfoParams fBaseParams;
    FuzzyBaseInfoParams& GetBaseParams() override
    {
        return fBaseParams;
    }
};

} // namespace optiling
