/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_gate_graph_infer.cpp
 * \brief EngramGate dtype 推导
 */
#include <cstddef>
#include "register/op_impl_registry.h"

namespace ops {
namespace {
constexpr size_t OUTPUT_INDEX = 0;
constexpr size_t DOT_INDEX = 1;
constexpr size_t GATE_SCORE_INDEX = 2;
constexpr size_t RSTD_X_INDEX = 3;
constexpr size_t RSTD_K_INDEX = 4;
} // namespace

ge::graphStatus InferDataType4EngramGate(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(OUTPUT_INDEX, context->GetInputDataType(0));
    context->SetOutputDataType(DOT_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(GATE_SCORE_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(RSTD_X_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(RSTD_K_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(EngramGate).InferDataType(InferDataType4EngramGate);
} // namespace ops
