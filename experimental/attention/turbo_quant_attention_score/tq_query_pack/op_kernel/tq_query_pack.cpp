/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
using namespace AscendC;
namespace {
constexpr uint32_t QUERY_VECTOR_BYTES = 128 * sizeof(bfloat16_t);
constexpr uint32_t QUERY_PACK_ROW_BYTES = 2 * QUERY_VECTOR_BYTES;
// For GM -> UB DataCopyPad on Ascend910B, srcStride is in bytes while dstStride is in 32-byte blocks.
constexpr uint32_t QUERY_PACK_DST_GAP_BLOCKS = (QUERY_PACK_ROW_BYTES - QUERY_VECTOR_BYTES) / 32;
static_assert(QUERY_PACK_DST_GAP_BLOCKS == 8, "query-pack row gap must be eight 32-byte blocks");
} // namespace
namespace optiling {
struct TqQueryPackTilingData {
    uint32_t payload[16400];
};
} // namespace optiling
extern "C" __global__ __aicore__ void tq_query_pack(GM_ADDR q, GM_ADDR s, GM_ADDR out, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(optiling::TqQueryPackTilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GlobalTensor<uint32_t> h;
    h.SetGlobalBuffer((__gm__ uint32_t*)tiling);
    uint32_t tokens = h.GetValue(0), heads = h.GetValue(1), cores = h.GetValue(2);
    TPipe pipe;
    TBuf<TPosition::VECCALC> qb, ob, ib;
    pipe.InitBuffer(qb, 32 * QUERY_PACK_ROW_BYTES);
    pipe.InitBuffer(ob, 32 * 512 * 2);
    pipe.InitBuffer(ib, 32 * 512 * 4);
    GlobalTensor<bfloat16_t> qr, qs, packed;
    qr.SetGlobalBuffer((__gm__ bfloat16_t*)q);
    qs.SetGlobalBuffer((__gm__ bfloat16_t*)s);
    packed.SetGlobalBuffer((__gm__ bfloat16_t*)out);
    auto local = qb.Get<bfloat16_t>();
    auto dst = ob.Get<bfloat16_t>();
    auto idx = ib.Get<uint32_t>();
    DataCopy(idx, h[4], 32 * 512);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    uint32_t groups = (tokens + 31) / 32, padded = (tokens + 127) / 128 * 128;
    for (uint32_t task = GetBlockIdx(); task < heads * groups; task += cores) {
        uint32_t head = task / groups, token = task % groups * 32, rows = tokens - token < 32 ? tokens - token : 32;
        uint64_t src = (uint64_t(token) * heads + head) * 128;
        DataCopyExtParams cp{uint16_t(rows), QUERY_VECTOR_BYTES, (heads - 1) * QUERY_VECTOR_BYTES,
                             QUERY_PACK_DST_GAP_BLOCKS, 0};
        DataCopyPad(local, qr[src], cp, DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        DataCopyPad(local[128], qs[src], cp, DataCopyPadExtParams<bfloat16_t>{false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Gather(dst, local, idx, 0, rows * 512);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(packed[(uint64_t(head) * padded + token) * 512], dst, rows * 512);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}
