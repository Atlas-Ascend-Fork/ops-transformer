/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file fused_infer_attention_score_tiling_arch31.h
 * \brief arch31 (KirinX90 / Kirin9030) Tiling 入口声明
 */
#ifndef FUSED_INFER_ATTENTION_SCORE_TILING_ARCH31_H
#define FUSED_INFER_ATTENTION_SCORE_TILING_ARCH31_H

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"

namespace optiling {
// arch31 Tiling 入口
ge::graphStatus TilingFusedInferAttentionScoreArch31(gert::TilingContext* context);
// arch31 Tiling 数据结构（简化版，仅含 arch31 路径所需字段）
class FusedInferAttentionScoreTilingArch31 {
public:
    explicit FusedInferAttentionScoreTilingArch31(gert::TilingContext* context)
        : context_(context)
    {}
    ~FusedInferAttentionScoreTilingArch31() = default;

    ge::graphStatus DoOpTiling();

private:
    gert::TilingContext* context_;
};
} // namespace optiling

#endif // FUSED_INFER_ATTENTION_SCORE_TILING_ARCH31_H
