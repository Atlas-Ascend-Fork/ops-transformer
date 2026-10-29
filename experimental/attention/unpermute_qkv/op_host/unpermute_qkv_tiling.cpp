/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file unpermute_qkv_tiling.cpp
 * \brief
 */

#include "unpermute_qkv_tiling.h"
#include "register/op_def_registry.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

// tiling keys
constexpr uint64_t TILING_KEY_BF16 = 0;
constexpr uint64_t TILING_KEY_FP16 = 1;

// some constants
constexpr int UB_ALIGN = 32;
constexpr uint32_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;
constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    UnpermuteQkvTilingData tiling;
    const gert::Shape& input_shape = context->GetInputShape(0)->GetStorageShape();
    const gert::Shape& q_shape = context->GetOutputShape(0)->GetStorageShape();
    const gert::Shape& k_shape = context->GetOutputShape(1)->GetStorageShape();
    const gert::Shape& v_shape = context->GetOutputShape(2)->GetStorageShape();

    if (input_shape.GetDimNum() != 5) {
        OP_LOGE(context->GetNodeName(), "Dim of input not right");
        return ge::GRAPH_FAILED;
    }
    if (q_shape.GetDimNum() != 4 || k_shape.GetDimNum() != 4 || v_shape.GetDimNum() != 4) {
        OP_LOGE(context->GetNodeName(), "Dim of input not right");
        return ge::GRAPH_FAILED;
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = ascendcPlatform.GetCoreNumAiv();

    auto inputType = context->GetInputDesc(0)->GetDataType();
    auto outputType = context->GetOutputDesc(0)->GetDataType();
    (void)outputType;

    const uint32_t s_c = q_shape.GetDim(0);
    const uint32_t b = q_shape.GetDim(1);
    const uint32_t q_num_heads = q_shape.GetDim(2);
    const uint32_t n_k = k_shape.GetDim(2);
    const uint32_t kv_group = q_num_heads / n_k;
    const uint32_t h = q_shape.GetDim(3);
    const uint32_t split_size = input_shape.GetDim(0);

    if (q_num_heads % n_k != 0) {
        OP_LOGE(context->GetNodeName(), "n_k must be divisible by query dim 2");
        return ge::GRAPH_FAILED;
    }
    if ((h * GetSizeByDataType(inputType)) % UB_ALIGN != 0) {
        OP_LOGE(context->GetNodeName(), "h dimension must be divisible by 16");
        return ge::GRAPH_FAILED;
    }
    const uint32_t split_n_k = n_k / split_size;
    if (n_k % split_size != 0) {
        OP_LOGE(context->GetNodeName(), "split_size must be divisible by n_k");
        return ge::GRAPH_FAILED;
    }

    tiling.set_splitSize(split_size);
    tiling.set_sC(s_c);
    tiling.set_b(b);
    tiling.set_kvGroup(kv_group);
    tiling.set_nK(n_k);
    tiling.set_splitNK(split_n_k);
    tiling.set_h(h);

    context->SetBlockDim(coreNum);

    // tiling key
    if (inputType == ge::DT_BF16) {
        context->SetTilingKey(TILING_KEY_BF16);
    } else if (inputType == ge::DT_FLOAT16) {
        context->SetTilingKey(TILING_KEY_FP16);
    } else {
        OP_LOGE(context->GetNodeName(), "Dtype of input is not in fp16 / bf16.");
        return ge::GRAPH_FAILED;
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = SYS_WORKSPACE_SIZE;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(UnpermuteQkv).Tiling(TilingFunc);
} // namespace optiling
