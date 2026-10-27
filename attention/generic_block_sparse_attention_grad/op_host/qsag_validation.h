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

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace qsag {
// Architecture-independent descriptor validation shared by ACLNN and graph tiling.
// Kernel limits and tiling parameters belong to each architecture's tiling.
// Device values (prefix sums, counts and indices) remain the documented input contract.
struct TensorInfo {
    std::vector<int64_t> shape;
    int dtype = -1;
    bool present = false;
};
struct Config {
    std::string layoutQ, layoutKv;
    std::vector<int64_t> block;
    int64_t residual = 0, mask = 1, precision = 0, left = -1, right = -1;
    bool consistent = false;
    float scale = 1;
};
struct ShapeInfo {
    uint64_t batch, sq, sk, nq, nk, d, group, qcap, topk, bs, layout;
};
inline bool Product(uint64_t a, uint64_t b, uint64_t& out)
{
    if (b != 0 && a > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / b)
        return false;
    out = a * b;
    return true;
}
inline bool ValidShape(const TensorInfo& t)
{
    if (!t.present || t.shape.empty())
        return false;
    uint64_t size = 1;
    for (int64_t d : t.shape) {
        if (d < 0 || !Product(size, static_cast<uint64_t>(d), size))
            return false;
    }
    // Bound byte offsets, including padded float workspaces.
    return size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / 64;
}
inline std::string Validate(const std::array<TensorInfo, 17>& t, const Config& a, int fp16, int bf16, int fp32,
                            int int32, int int64, ShapeInfo& s)
{
    for (size_t i : {0, 1, 2, 3, 4, 5, 6, 7, 14, 15, 16}) {
        if (!ValidShape(t[i]))
            return "required tensor missing, negative dimension or shape overflow";
    }
    if (a.layoutQ != a.layoutKv || (a.layoutQ != "TND" && a.layoutQ != "BNSD" && a.layoutQ != "BSND"))
        return "QSAG requires matching TND/BNSD/BSND layouts";
    if (a.residual != 1)
        return "QSAG requires residual_block_mode=1";
    if ((a.mask != 0 && a.mask != 1) || a.precision != 0 || a.left != -1 || a.right != -1 || !std::isfinite(a.scale))
        return "QSAG supports mask 0/1, precision 0, windows -1 and finite scale";
    if (t[8].present || t[9].present)
        return "QSAG does not consume metadata or atten_mask; pass null";
    const auto& q = t[0].shape;
    const auto& k = t[1].shape;
    bool packed = a.layoutQ == "TND";
    size_t rank = packed ? 3 : 4;
    if (q.size() != rank || k.size() != rank)
        return "Q/K rank does not match layout";
    if (t[2].shape != k || t[3].shape != q || t[4].shape != q || t[14].shape != q || t[15].shape != k ||
        t[16].shape != k)
        return "Q/dout/out/dQ and K/V/dK/dV shapes must match";
    if (t[0].dtype != fp16 && t[0].dtype != bf16)
        return "QSAG supports FP16/BF16 only";
    for (size_t i : {1, 2, 3, 4, 14, 15, 16})
        if (t[i].dtype != t[0].dtype)
            return "Q/K/V and gradient dtypes must match";
    if (t[5].dtype != fp32 || t[6].dtype != int32 || t[7].dtype != int32)
        return "LSE must be FP32 and sparse tensors INT32";
    auto lse = q;
    lse.back() = 1;
    auto squeezed = q;
    squeezed.pop_back();
    if (t[5].shape != lse && t[5].shape != squeezed)
        return "LSE must follow Q layout with trailing singleton or without D axis";
    if (t[6].shape.size() != 4 || t[7].shape.size() != 3)
        return "indices/count ranks must be 4/3";
    s.layout = packed ? 0 : (a.layoutQ == "BNSD" ? 1 : 2);
    size_t h = packed || s.layout == 1 ? 1 : 2, seq = packed ? 0 : (s.layout == 1 ? 2 : 1);
    s.nq = q[h];
    s.nk = k[h];
    s.d = q.back();
    s.sq = q[seq];
    s.sk = k[seq];
    s.batch = t[6].shape[0];
    s.qcap = t[6].shape[2];
    s.topk = t[6].shape[3];
    if (!s.batch || !s.nq || !s.nk || s.nq % s.nk || !s.d || k.back() != static_cast<int64_t>(s.d))
        return "positive batch/heads, divisible head counts and common positive D required";
    s.group = s.nq / s.nk;
    if (t[6].shape[1] != static_cast<int64_t>(s.nk) || !s.topk || s.topk > INT32_MAX ||
        t[7].shape != std::vector<int64_t>(
                          {static_cast<int64_t>(s.batch), static_cast<int64_t>(s.nk), static_cast<int64_t>(s.qcap)}))
        return "indices [B,N2,Qcap,topK] and counts [B,N2,Qcap] must match; 1<=topK<=INT32_MAX";
    if (!packed && (q[0] != static_cast<int64_t>(s.batch) || k[0] != q[0] || s.qcap != s.sq))
        return "dense batch and Qcap must match Q/K";
    if (packed) {
        for (size_t i : {10, 11})
            if (!ValidShape(t[i]) || t[i].dtype != int64 ||
                t[i].shape != std::vector<int64_t>({static_cast<int64_t>(s.batch + 1)}))
                return "TND requires INT64 cu lengths [B+1]";
        for (size_t i : {12, 13})
            if (t[i].present && (!ValidShape(t[i]) || t[i].dtype != int32 ||
                                 t[i].shape != std::vector<int64_t>({static_cast<int64_t>(s.batch)})))
                return "seqused must be INT32 [B]";
    } else {
        for (size_t i : {10, 11, 12, 13})
            if (t[i].present)
                return "dense QSAG does not accept cu lengths or seqused";
    }
    if (a.block.size() != 2 || a.block[0] != 1 || a.block[1] <= 0)
        return "QSAG block_shape must be [1,Bs] with Bs>0";
    s.bs = a.block[1];
    return {};
}
} // namespace qsag
