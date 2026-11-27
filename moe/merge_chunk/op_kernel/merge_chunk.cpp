/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "merge_chunk_impl.h"
#include "kernel_operator_list_tensor_intf.h"

extern "C" __global__ __aicore__ void merge_chunk(GM_ADDR inputChunks, GM_ADDR groupListChunk, GM_ADDR outChunks,
                                                  GM_ADDR outGroupList, GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AscendC::AIC) {
        return;
    }
    GET_TILING_DATA(data, tiling);
    AscendC::ListTensorDesc inputs(reinterpret_cast<__gm__ void*>(inputChunks));
    AscendC::ListTensorDesc counts(reinterpret_cast<__gm__ void*>(groupListChunk));
    AscendC::ListTensorDesc outputs(reinterpret_cast<__gm__ void*>(outChunks));
    AscendC::TPipe pipe;
    // Token rows are copied bit for bit for both FLOAT16 and BFLOAT16.
    MergeChunkSupport::KernelMergeChunk<uint16_t, decltype(data)> kernel(data, pipe);
    kernel.Process(inputs.GetDataPtr<uint8_t>(0), inputs.GetDataPtr<uint8_t>(1), inputs.GetDataPtr<uint8_t>(2),
                   inputs.GetDataPtr<uint8_t>(3), counts.GetDataPtr<uint8_t>(0), counts.GetDataPtr<uint8_t>(1),
                   outputs.GetDataPtr<uint8_t>(0), outputs.GetDataPtr<uint8_t>(1), outGroupList);
}
