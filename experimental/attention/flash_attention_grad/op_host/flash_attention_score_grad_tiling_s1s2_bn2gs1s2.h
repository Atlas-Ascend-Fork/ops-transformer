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

#include "flash_attention_grad_tiling_common.h"
#include "base/tiling_base.h"
#include "flash_attention_score_grad_tiling_s1s2_bn2gs1s2_def.h"

namespace optiling {

class FlashAttentionScoreGradTilingUs1s2Bs2 : public TilingBaseClass {
public:
    explicit FlashAttentionScoreGradTilingUs1s2Bs2(gert::TilingContext* context_)
        : TilingBaseClass(context_)
    {}
    ~FlashAttentionScoreGradTilingUs1s2Bs2() {}
    FlashAttentionScoreGradTilingDataUs1s2Bbn2gs1s2 tilingData;

protected:
    // 通用
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoLibApiTiling() override;
    ge::graphStatus PostTiling() override;
    // 子类重载
    bool IsCapable() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus GetWorkspaceSize() override;
    uint64_t GetTilingKey() const override;

    virtual void DoSplit();
    virtual void DoSparse();
    virtual ge::graphStatus SaveToTilingData();

    bool SetSparseParams();
    bool CheckFuzzyArgsLegal(uint32_t s1Inner, uint32_t s2Inner);
    std::tuple<uint32_t, uint32_t, uint32_t> FuzzyForBestSplit();
    void SetMatmulTilingBufferInfo(TCubeTiling& mmTiling);
    void DoPreTiling();
    void DoPostTiling();
    void DetermineMode();
    ge::graphStatus CheckAttenMaskShape();

    ge::graphStatus GetSparseBlockInfo();
    ge::graphStatus GetSparseUnpadBlockInfo();
    void FillBlockInfo(std::vector<std::vector<std::vector<uint32_t>>>& calculatedBlockInfo,
                       std::vector<std::vector<uint32_t>>& totalBlockInfo);
    void GetParseS1S2OuterInfo(uint32_t (*parseInfo)[ARRAY_LENGTH]);

    FuzzyBaseInfoParams fBaseParams;
    virtual FuzzyBaseInfoParams& GetBaseParams()
    {
        return fBaseParams;
    }
};

} // namespace optiling
