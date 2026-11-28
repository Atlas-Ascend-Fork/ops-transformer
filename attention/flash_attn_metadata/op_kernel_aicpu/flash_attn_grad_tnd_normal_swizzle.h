/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLASH_ATTN_GRAD_TND_NORMAL_SWIZZLE_H
#define FLASH_ATTN_GRAD_TND_NORMAL_SWIZZLE_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace aicpu::fag_pto_normal_swizzle {
constexpr int64_t MAGIC = 0x46534E31;
constexpr int64_t VERSION = 1;
constexpr int64_t HEADER_I64 = 16;
constexpr int64_t PREFIX_COUNT = 129;
constexpr int64_t WORDS = 2 * (HEADER_I64 + PREFIX_COUNT);
using Wire = std::array<int64_t, HEADER_I64 + PREFIX_COUNT>;

inline int64_t Ceil(int64_t x, int64_t y)
{
    return x >= 0 ? x / y + (x % y != 0) : x / y;
}

// ASC varlen CheckUnpadSparseLeftAndRight, for this API's masks 0/3/4.
inline bool Valid(int64_t row, int64_t col, int64_t q, int64_t kv, int64_t mask, int64_t pre, int64_t next, bool sparse,
                  int64_t maxQ)
{
    if (!sparse) {
        return true;
    }
    const int64_t colLeft = col * 128;
    const int64_t colRight = std::min((col + 1) * 128, kv);
    pre += q - kv;
    next += kv - q;
    const int64_t left = Ceil(std::max(row * 128 - pre, int64_t{0}), 64) * 64;
    const int64_t right = std::min(Ceil(std::min((row + 1) * 128, maxQ) + next, 64) * 64, kv);
    return colLeft < right && colRight > left;
}

// Runtime producer: no deterministic-mode prediction and no device-L2 constant.
// Gate L2 against tiling.planL2Bytes on device. Final split/full-chip and actual
// lengths are known only here, after ordinary DoFagSparse has completed.
inline Wire Build(const std::vector<int64_t>& q, const std::vector<int64_t>& kv, int64_t n1, int64_t d, int64_t cores,
                  int64_t mask, int64_t left, int64_t right, bool sparse, bool finalBn2s2Tnd, int64_t blockOuter,
                  const int64_t* starts, const int64_t* ends)
{
    Wire out{};
    out[0] = MAGIC;
    out[1] = VERSION;
    out[3] = 3; // ASC UNSUPPORTED
    out[10] = mask;
    out[11] = left;
    out[12] = right;
    out[13] = q.size();
    out[14] = n1;
    out[15] = cores;
    if (!finalBn2s2Tnd || q.empty() || q.size() >= PREFIX_COUNT || q.size() != kv.size() || n1 <= 0 || d <= 0 ||
        cores <= 0 || cores > 36 || blockOuter != cores || starts == nullptr || ends == nullptr) {
        return out;
    }
    int64_t maxQ = 0, maxKv = 0;
    bool same = true, less = true, greater = true;
    for (size_t b = 0; b < q.size(); ++b) {
        if (q[b] < 0 || kv[b] < 0 || q[b] > INT32_MAX || kv[b] > INT32_MAX) {
            return out;
        }
        if (q[b] == 0 || kv[b] == 0) {
            if (mask != 0 || sparse) {
                return out;
            }
            continue;
        }
        maxQ = std::max(maxQ, q[b]);
        maxKv = std::max(maxKv, kv[b]);
        same = same && q[b] == kv[b];
        less = less && q[b] <= kv[b];
        greater = greater && q[b] >= kv[b];
    }
    const int64_t pre = mask == 3 || left < 0 ? INT32_MAX : left;
    const int64_t next = mask == 3 ? 0 : (right < 0 ? INT32_MAX : right);
    // ASC GetSparseType, restricted to TND and the public mask domain.
    const bool causal = n1 % 2 == 0 && ((mask == 3 && less && (same || maxQ % 128 == 0)) ||
                                        (mask == 4 && pre >= maxQ && next == 0 && greater));
    const bool band = mask == 4 && pre >= 0 && next >= 0 && same;
    const int64_t kind = !sparse ? 0 : causal ? 1 : band ? 2 : 3;
    out[3] = kind;
    if (kind == 3) {
        return out;
    }
    out[4] = q.size() + 1;
    std::array<int64_t, 36> validPerCore{};
    int64_t densePrefix = 0;
    __int128 workingBytes = 0;
    for (size_t b = 0; b < q.size(); ++b) {
        const int64_t m = Ceil(q[b], 128), n = Ceil(kv[b], 128);
        if (m == 0 || n == 0) {
            // Preserve physical batch indices while giving empty tile grids no work.
            out[HEADER_I64 + b + 1] = out[HEADER_I64 + b];
            continue;
        }
        int64_t rounds = 0;
        if (kind == 0) {
            rounds = Ceil(n * n1, cores) * m;
        } else if (kind == 1) {
            const int64_t rm = mask == 3 ? std::min(m, n) : m;
            const int64_t rn = mask == 3 ? n : std::min(m, n);
            rounds = mask == 3 ? Ceil((2 * rn - rm + 1) * (n1 / 2), cores) * rm :
                                 Ceil(rn * (n1 / 2), cores) * (2 * rm - rn + 1);
        } else {
            const int64_t p = std::min(Ceil(pre + q[b] - kv[b], 128) + 1, m);
            const int64_t t = std::min(Ceil(next - q[b] + kv[b], 128) + 1, n);
            rounds = Ceil(n * n1, cores) * std::min(p + t - 1, m);
        }
        out[HEADER_I64 + b + 1] = out[HEADER_I64 + b] + rounds;
        std::vector<uint8_t> usedRows(m, 0);
        int64_t usedColumns = 0;
        // Count the same effective tiles that ASC CalValidBlockCount feeds to
        // CaclePerCoreBlockInfo. The partition boundaries remain the original
        // ordinary BN2S2 split, not a GS allocation formula.
        for (int64_t col = 0; col < n; ++col) {
            std::vector<int64_t> rows;
            for (int64_t row = 0; row < m; ++row) {
                if (Valid(row, col, q[b], kv[b], mask, pre, next, sparse, maxQ)) {
                    rows.push_back(row);
                    usedRows[row] = 1;
                }
            }
            usedColumns += !rows.empty();
            for (int64_t head = 0; head < n1; ++head) {
                for (auto row : rows) {
                    const int64_t idx = densePrefix + head * m * n + col * m + row;
                    const auto end = std::upper_bound(ends, ends + blockOuter, idx);
                    const int64_t core = end - ends;
                    if (core < blockOuter && idx >= starts[core]) {
                        ++validPerCore[core];
                    }
                }
            }
        }
        const int64_t usedRowCount = std::count(usedRows.begin(), usedRows.end(), uint8_t{1});
        // This ASC estimate deliberately uses D for all inputs/outputs. It is
        // an admission heuristic, not a UB/L1 allocation or an identity check.
        workingBytes += static_cast<__int128>(usedRowCount) * n1 * 128 * d * 10;
        workingBytes += static_cast<__int128>(usedColumns) * n1 * 128 * d * 12;
        densePrefix += m * n * n1;
    }
    if (mask == 3 || mask == 4) {
        workingBytes += 2048 * 2048;
    }
    out[5] = out[HEADER_I64 + q.size()];
    out[6] = static_cast<int64_t>(std::min(workingBytes, static_cast<__int128>(INT64_MAX)));
    // CheckIsLargeInvalidBlk applies only to LEFT_UP_CAUSAL, absent in this API.
    out[7] = 0;
    out[8] = *std::max_element(validPerCore.begin(), validPerCore.end());
    out[9] = out[5];
    const bool longSeq = maxQ >= 2048 || (maxKv > 128 && maxQ >= 1024);
    const bool shortBenefit = out[5] > 0 && static_cast<__int128>(out[5]) * 100 <= static_cast<__int128>(out[8]) * 80;
    out[2] = out[5] > 0 && (longSeq || shortBenefit);
    return out;
}
} // namespace aicpu::fag_pto_normal_swizzle
#endif
