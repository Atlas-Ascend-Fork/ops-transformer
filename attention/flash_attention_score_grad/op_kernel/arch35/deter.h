/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file deter.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_GRAD_DETER_H_
#define FLASH_ATTENTION_SCORE_GRAD_DETER_H_

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "flash_attention_score_grad_common.h"
#define FAG_GQA_BAND_INLINE __aicore__ inline
#include "deter_gqa_band_index.h"

namespace commondef {
constexpr int64_t DETER_PREFIX_NUM = 132;
constexpr int64_t DETER_PREFIX_THRESHOLD = 128;
constexpr int64_t M_BASE_64 = 64;
constexpr int64_t N_BASE_128 = 128;

__aicore__ inline int64_t Square(int64_t num)
{
    return num * num;
}

__aicore__ inline int64_t Gcd(int64_t a, int64_t b)
{
    int64_t r;
    while (b > 0) {
        r = a % b;
        a = b;
        b = r;
    }
    return a;
}

__aicore__ inline int64_t AbsCeil(int64_t num1, int64_t num2)
{
    bool isNegative = (num1 < 0) || (num2 < 0);
    int64_t result = (abs(num1) + abs(num2) - 1) / abs(num2);
    return isNegative ? -result : result;
}

struct CoordinateInfo {
    int64_t batchId = 0;
    int64_t n2Idx = 0;
    int64_t gIdx = 0;
    int64_t s1Idx = 0;
    int64_t s2Idx = 0;

    int64_t s1Outer = 0;
    int64_t s2Outer = 0;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    int64_t sparseMode = NO_MASK;

    int64_t mOffset = 0;
    int64_t nOffset = 0;

    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
};

template <uint8_t mode = 0>
__aicore__ inline void TransDeterRound(int64_t& r, bool& useEvenCol, int64_t rMax)
{
    if constexpr (mode == 0) {
        return;
    } else if constexpr (mode == 1) {
        useEvenCol = r % NUM_TWO == 1;
        r = (r + 1) / NUM_TWO;
        return;
    } else {
        if (r > rMax) {
            r = r - rMax;
            useEvenCol = true;
        }
        return;
    }
}

template <uint8_t mode = 0>
__aicore__ inline void TransTilingSplitMode(int64_t& m, int64_t& n, int64_t& r, bool& useEvenCol, int64_t rMax)
{
    if constexpr (mode == 0) {
        return;
    } else if constexpr (mode == 1) {
        m = Ceil<int64_t>(m, static_cast<int64_t>(NUM_TWO));
        TransDeterRound<mode>(r, useEvenCol, rMax);
        return;
    } else {
        n = Ceil<int64_t>(n, static_cast<int64_t>(NUM_TWO));
        TransDeterRound<mode>(r, useEvenCol, rMax);
        return;
    }
}

template <uint8_t mode = 0, const bool IS_TND = false>
__aicore__ inline void TransTilingSplitModeBack(CoordinateInfo& coordinateInfo, bool useEvenCol)
{
    if constexpr (mode == 0) {
        return;
    } else if constexpr (mode == 1) {
        coordinateInfo.s1Idx = useEvenCol ? coordinateInfo.s1Idx * NUM_TWO - 1 : coordinateInfo.s1Idx * NUM_TWO;
        if constexpr (IS_TND) {
            coordinateInfo.s1Outer = Ceil<int64_t>(coordinateInfo.actualS1Len, M_BASE_64);
        }
        return;
    } else {
        coordinateInfo.s2Idx = useEvenCol ? NUM_TWO * coordinateInfo.s2Idx : NUM_TWO * coordinateInfo.s2Idx - 1;
        if constexpr (IS_TND) {
            coordinateInfo.s2Outer = Ceil<int64_t>(coordinateInfo.actualS2Len, N_BASE_128);
        }
        return;
    }
}

__aicore__ inline void InitCoordinateInfo(int64_t s1Outer, int64_t s2Outer, int64_t mOffset, int64_t nOffset,
                                          CoordinateInfo& coordinateInfo)
{
    coordinateInfo.s1Outer = s1Outer;
    coordinateInfo.s2Outer = s2Outer;
    coordinateInfo.mOffset = mOffset;
    coordinateInfo.nOffset = nOffset;
}

__aicore__ inline void CalDenseIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                     CoordinateInfo& coordinate)
{
    k = Min(k, b * m);
    if (j > k) {
        coordinate.batchId = -1;
        return;
    }

    int64_t p = (Ceil<int64_t>(r, m) - 1) * k + j;
    // Determine w and y
    int64_t w = p % b;
    w = (w != 0) ? w : b;
    int64_t y = Ceil<int64_t>(p, b);

    int64_t y1 = y % m;
    y1 = (y1 != 0) ? y1 : m;
    int64_t r1 = r % m;
    r1 = (r1 != 0) ? r1 : m;

    // Calculate x
    int64_t x = y1 + r1 - 1;
    if (x > m) {
        x -= m;
    }

    // Check if all values are within the valid ranges
    if (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) {
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    } else {
        coordinate.batchId = -1;
    }
    return;
}

__aicore__ inline void CalDenseIndexForSingleN(int64_t k, int64_t m, int64_t b, int64_t j, int64_t r, int64_t R,
                                               CoordinateInfo& coordinate)
{
    int64_t n = 1;
    int64_t ID = (j - 1) * R + r;
    int64_t num = m * n;

    int64_t delta1 = (ID - 1) / num + 1;
    int64_t delta = ID % num;
    delta = delta == 0 ? num : delta;

    int64_t g = Gcd(m, R);
    int64_t t1 = R / g;
    int64_t t2 = m / g;

    int64_t x = ((delta - 1) % m) + 1;
    int64_t y = (delta - 1) / m + 1;

    if (t1 < n) {
        int64_t n1 = (n % t1);
        n1 = n1 == 0 ? t1 : n1;
        if (y <= n - n1) {
            int64_t delta_adj = Ceil<int64_t>(y, t1);
            delta += delta_adj;
            if (delta > delta_adj * t2 * R) {
                delta -= t2 * R;
            }
            x = ((delta - 1) % m) + 1;
            y = (delta - 1) / m + 1;
        }
    }
    coordinate.batchId = delta1;
    coordinate.s1Idx = x;
    coordinate.s2Idx = y;
}

__aicore__ inline void CalDenseSwizzleIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                            CoordinateInfo& coordinate)
{
    j = j - 1;
    r = r - 1;
    k = Min(k, b * m);
    if (j > k) {
        coordinate.batchId = -1;
        return;
    }
    int64_t p = (r / m) * k + j;
    int64_t w = p / n;
    int64_t y = p % n;
    int64_t x = (y + r) % m;
    if (x >= m) {
        x -= m;
    }
    w += 1;
    x += 1;
    y += 1;
    // Check if all values are within the valid ranges
    if (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) {
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    } else {
        coordinate.batchId = -1;
    }
    return;
}

__aicore__ inline void CalGQADenseIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t core_id, int64_t round_id,
                                        int64_t g, CoordinateInfo& coordinate, int64_t denseRound = 0)
{
    coordinate.batchId = -1;
    k = Min(Min(k, b * g * m), b * n);
    int64_t R = denseRound > 0 ? denseRound : Max(Max(Ceil<int64_t>(b * n * g, k), Ceil<int64_t>(n, m)), g);
    if (core_id < 1 || core_id > k || round_id < 1 || round_id > R * m) {
        return;
    }
    int64_t ID = (core_id - 1) * R + Ceil<int64_t>(round_id, m);
    int64_t local_id = round_id % m;
    local_id = local_id != 0 ? local_id : m;

    int64_t num = g * n;
    if (ID > num * b) {
        return;
    }

    int64_t N = b * g;
    int64_t b_id = ID % N;
    b_id = b_id != 0 ? b_id : N;
    b_id = Ceil<int64_t>(b_id, g);
    int64_t y = Ceil<int64_t>(ID, N);
    int64_t w = ID % g;
    w = w != 0 ? w : g;

    int64_t gcd = Gcd(N, R);
    int64_t t1 = R / gcd;
    // 实际不需要计算t2也可以 因为t2*R就是t1*b
    int64_t t2 = N / gcd;

    int64_t t1_new = t1 * m;
    int64_t y1 = y % t1_new;
    y1 = y1 != 0 ? y1 : t1_new;
    int64_t offset = Ceil<int64_t>(y1, t1);

    if (t1_new < n) {
        int64_t n1 = (n % t1_new);
        n1 = n1 != 0 ? n1 : t1_new;
        if (y <= n - n1) {
            int64_t delta = Ceil<int64_t>(y, t1_new);
            ID += delta;
            if (ID > (delta - 1) * t2 * m * R + offset * t2 * R) {
                ID -= t2 * R;
            }
            b_id = ID % N;
            b_id = b_id != 0 ? b_id : N;
            b_id = Ceil<int64_t>(b_id, g);
            w = ID % g;
            w = w != 0 ? w : g;
            y = Ceil<int64_t>(ID, N);
        }
    }

    int64_t x = local_id + offset - 1;
    if (x > m) {
        x -= m;
    }

    coordinate.batchId = w + (b_id - 1) * g;
    coordinate.s1Idx = x;
    coordinate.s2Idx = y;
}
__aicore__ inline void CalCausalG2kSingleBatchDeterIndex(int64_t k, int64_t j, int64_t a, int64_t L1, int64_t offset,
                                                         CoordinateInfo& coordinate)
{
    int64_t x, y = 0;
    if (j % NUM_TWO == 1) { // 奇核分支
        if (a <= L1 - j + 1) {
            y = j + offset;
            x = y + a - 1;
        } else {
            y = NUM_TWO * k + 1 - j + offset;
            x = y + NUM_TWO * L1 - NUM_TWO * k + 1 - a;
        }
    } else { // 偶核分支
        if (a >= L1 - NUM_TWO * k + 1 + j) {
            y = j + offset;
            x = y + NUM_TWO * L1 - NUM_TWO * k + 1 - a;
        } else {
            y = NUM_TWO * k + 1 - j + offset;
            x = y + a - 1;
        }
    }
    coordinate.batchId = 0;
    coordinate.s1Idx = x;
    coordinate.s2Idx = y;
    return;
}

__aicore__ inline void CalCausalNoRecSingleBatchDeterIndex(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r,
                                                           CoordinateInfo& coordinate)
{
    coordinate.batchId = -1;

    // 核心编号超过 ⌊n/2⌋ + 1 时无任务可处理
    if (j > (n / NUM_TWO) + 1) {
        return;
    }

    int64_t x, y = 0;
    if (j % NUM_TWO == 1) { // 奇核
        if (r + j <= n + 1) {
            x = r + j - 1;
            y = j;
        } else {
            x = NUM_TWO * n + NUM_TWO - j - r;
            y = n + NUM_THREE - j - (n % NUM_TWO);
        }
    } else { // 偶核
        if (j <= r + 1 - (n % NUM_TWO)) {
            x = n + j - r - 1 + n % NUM_TWO;
            y = j;
        } else {
            x = n + NUM_TWO + r - j - (n % NUM_TWO);
            y = n + NUM_THREE - j - (n % NUM_TWO);
        }
    }

    // 检查 (x, y) 是否在下三角范围内
    if (y >= 1 && y <= m && y <= x && x <= m) {
        coordinate.batchId = 0;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    }
    return;
}

// 矩形
__aicore__ inline void CalCausalRecSingleBatchDeterIndex(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r,
                                                         CoordinateInfo& coordinate)
{
    int64_t x, y = 0;
    // Choose method based on core count
    if (NUM_TWO * k < m + 1 && k < n) {
        CalCausalG2kSingleBatchDeterIndex(k, j, r, m, 0, coordinate);
        x = coordinate.s1Idx;
        y = coordinate.s2Idx;
    } else {
        CalCausalNoRecSingleBatchDeterIndex(k, m, m, j, r, coordinate);
        if (coordinate.batchId != -1) {
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;
        } else {
            x = m + 1;
            y = m + 1;
        }
    }

    // Check if the computed position is within the valid submatrix
    if (y >= 1 && y <= n && y <= x && x <= m) {
        coordinate.batchId = 0;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    } else {
        coordinate.batchId = -1;
    }
    return;
}

// 计算单batch切分
__aicore__ inline void CalCausalSingleBatchDeterIndex(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r,
                                                      CoordinateInfo& coordinate, bool exactRoundBoundary = false)
{
    // 0. special case
    if (k >= (n / NUM_TWO) + 1) {
        CalCausalRecSingleBatchDeterIndex(k, m, n, j, r, coordinate);
        return;
    }

    // 1. compute ell and t1, t2, t3
    int64_t ell = n % k;
    int64_t t1 = n / (NUM_TWO * k);
    int64_t t2 = 0;
    int64_t t3 = (n / k) % NUM_TWO;

    int64_t rm, L1, a, offset;
    // 2. 2k-block groups
    int64_t bound1 = (NUM_TWO * m + 1) * t1 - NUM_TWO * k * t1 * t1;
    if (r <= bound1) {
        // solve for group index i
        int64_t i;
        if (exactRoundBoundary) {
            // The batch-major path must not depend on rounding a floating-point square root.
            int64_t groupLow = 1;
            int64_t groupHigh = t1;
            while (groupLow < groupHigh) {
                int64_t groupMid = groupLow + (groupHigh - groupLow) / NUM_TWO;
                int64_t endRound = (NUM_TWO * m + 1) * groupMid - NUM_TWO * k * groupMid * groupMid;
                if (r <= endRound) {
                    groupHigh = groupMid;
                } else {
                    groupLow = groupMid + 1;
                }
            }
            i = groupLow;
        } else {
            int64_t disc = Square((NUM_TWO * m + 1)) - NUM_EIGHT * k * r;
            i = Ceil<int64_t>(((NUM_TWO * m + 1) - sqrt(disc)), (NUM_FOUR * k));
        }
        rm = (NUM_TWO * m + 1) * (i - 1) - NUM_TWO * k * (i - 1) * (i - 1);
        offset = NUM_TWO * k * (i - 1);
        L1 = m - NUM_TWO * k * (i - 1);
        a = r - rm;
        CalCausalG2kSingleBatchDeterIndex(k, j, a, L1, offset, coordinate);
        return;
    } else {
        rm = bound1;
        offset = NUM_TWO * k * t1;
    }

    // 4. remaining block
    int64_t rem = t3 * k + ell;
    coordinate.batchId = -1;
    if (rem > 0) {
        L1 = m - n + rem;
        a = r - rm;
        if (t3 == 0 && j <= rem) {
            coordinate.batchId = 0;
            coordinate.s2Idx = offset + j;
            coordinate.s1Idx = coordinate.s2Idx + a - 1;
            return;
        } else if (t3 == 1) {
            CalCausalRecSingleBatchDeterIndex(k, L1, L1, j, a, coordinate);
            if (coordinate.batchId == -1) {
                return;
            }

            int64_t shift = NUM_TWO * k * t1 + NUM_THREE * k * t2;
            coordinate.s1Idx += shift;
            coordinate.s2Idx += shift;
            return;
        } else {
            return;
        }
    }
    return;
}

__aicore__ inline void CalCausalIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                      CoordinateInfo& coordinate)
{
    // 1. b1, b2, rm1
    int64_t b1 = b / k;
    int64_t b2 = b % k;
    int64_t delta = m - n;
    int64_t sizeTri = n * (m + delta + 1) / NUM_TWO;
    int64_t rm1 = b1 * sizeTri;

    int64_t w, x, y = 0;
    // --- 第一段：完整矩形 ---
    if (r <= rm1) {
        int64_t a = r % sizeTri;
        a = a != 0 ? a : sizeTri;
        w = Ceil<int64_t>(r, sizeTri) + b1 * (j - 1);
        int64_t n1 = n / NUM_TWO * NUM_TWO;
        int64_t L = NUM_TWO * m - n1 + 1;
        int64_t rmLocal = (n1 * L) / NUM_TWO;
        if (a <= rmLocal) {
            y = Ceil<int64_t>(a, L);
            int64_t r1 = a % L != 0 ? a % L : L;
            x = r1 + y - 1;
            if (x > m) {
                x = NUM_TWO * m + 1 - x;
                y = n1 + 1 - y;
            }
        } else {
            int64_t a1 = a - rmLocal;
            y = n;
            x = a1 - 1 + y;
        }
        w = ((w - 1) % b1) * k + (w - 1) / b1 + 1;
        w = ((w - 1) / k) * k + ((y - 1 + (w - 1)) % k) + 1;
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
        return;
    }

    // 2. 行块压平分给 floor(b2/2) 个 batch
    int64_t t = n / k;
    int64_t ell = n % k;
    int64_t a2 = r - rm1;
    int64_t half_ = b2 / NUM_TWO;
    int64_t rm2 = (NUM_TWO * m - t * k + 1) * t * half_;
    if (a2 >= 1 && a2 <= rm2) {
        // 调用 dense 子块
        CalDenseIndex(k, NUM_TWO * m - t * k + 1, t * k, half_, j, a2, coordinate);
        int64_t wSub = coordinate.batchId;
        int64_t xSub = coordinate.s1Idx;
        int64_t ySub = coordinate.s2Idx;
        // 新增范围检查
        int64_t max_x = NUM_TWO * m - t * k + 1;
        int64_t max_y = t * k;
        int64_t max_w = half_;
        if (xSub >= 1 && xSub <= max_x && ySub >= 1 && ySub <= max_y && wSub >= 1 && wSub <= max_w) {
            // 计算映射
            if (xSub - ySub <= m - t * k) {
                w = NUM_TWO * wSub - 1 + b1 * k;
                x = m + 1 - xSub;
                y = t * k + 1 - ySub;
            } else {
                w = NUM_TWO * wSub + b1 * k;
                x = xSub - m + t * k - 1;
                y = ySub;
            }
            coordinate.batchId = w;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
        } else {
            coordinate.batchId = -1;
        }
        return;
    }

    // 3. 剩余，分两种：b2 mod2 == 1 或 else
    int64_t a3 = r - rm1 - rm2;
    int64_t rm3 = 0;
    if (b2 % NUM_TWO == 1) {
        int64_t t1 = n / (NUM_TWO * k);
        int64_t t3 = t % NUM_TWO;
        // 奇数 b2
        if (t3 == 1) {
            int64_t m1 = m - t1 * NUM_TWO * k;
            rm3 = (m + m1 + 1) * t1;
            rm3 += ell == 0 ? m1 : Max(m1, NUM_TWO * m1 - NUM_TWO * k + 1);
            if (a3 >= 1 && a3 <= rm3) {
                CalCausalSingleBatchDeterIndex(k, m, n, j, a3, coordinate);
                if (coordinate.batchId != -1) {
                    coordinate.batchId = b;
                    return;
                } else {
                    return;
                }
            }
            b -= 1;
            b2 -= 1;
        } else {
            // 偶数 b2
            rm3 = (NUM_TWO * m - t * k + 1) * t / NUM_TWO;
            if (a3 >= 1 && a3 <= rm3) {
                CalCausalSingleBatchDeterIndex(k, m, t * k, j, a3, coordinate);
                if (coordinate.batchId != -1) {
                    coordinate.batchId = b;
                    return;
                } else {
                    return;
                }
            }
        }
    }

    // 4. 零碎余数 ℓ 部分
    int64_t a4 = a3 - rm3;
    int64_t p = Ceil<int64_t>(ell, NUM_TWO);
    int64_t ell1 = ell + 1 - (ell % NUM_TWO);
    int64_t block = (b2 * p) / k;
    int64_t res0 = (b2 * p) % k;
    // 情况 A
    if (a4 > block * (ell1 + NUM_TWO * delta) && res0 <= k / NUM_TWO) {
        int64_t offs = a4 - block * (ell1 + NUM_TWO * delta);
        // 前 res0 cores
        if (j >= 1 && j <= res0) {
            int64_t limit = (res0 - j) / b2 + Ceil<int64_t>(ell1, NUM_TWO) + delta;
            if (offs <= limit) {
                w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j) % b2;
                w = w != 0 ? w : b2;
                y = p - (res0 - j) / b2;
                x = y + offs - 1;
                if (y >= 1 && y <= ell && y <= x && x <= (ell + delta)) {
                    coordinate.batchId = b1 * k + w;
                    coordinate.s1Idx = x + t * k;
                    coordinate.s2Idx = y + t * k;
                    return;
                }
            }
        }

        // 后 res0 cores
        if (k - res0 + 1 <= j && j <= k) {
            int64_t idx = j - (k - res0 + 1);
            int64_t limit = (ell1 - 1) / NUM_TWO - idx / b2 + delta;
            if (offs <= limit) {
                w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + k + 1 - j) % b2;
                w = w != 0 ? w : b2;
                y = p + 1 + idx / b2;
                x = y + offs - 1;
                if (y >= 1 && y <= ell && y <= x && x <= (ell + delta)) {
                    coordinate.batchId = b1 * k + w;
                    coordinate.s1Idx = x + t * k;
                    coordinate.s2Idx = y + t * k;
                    return;
                }
            }
        }
        coordinate.batchId = -1;
        return;
    } else {
        // 情况 B
        w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j) % b2;
        w = w != 0 ? w : b2;
        int64_t g = Ceil<int64_t>((k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j), b2);
        if (g >= 1 && g <= p) {
            int64_t a5 = a4 % (ell1 + NUM_TWO * delta);
            a5 = a5 != 0 ? a5 : (ell1 + NUM_TWO * delta);
            int64_t x0, y0 = 0;
            if (g % NUM_TWO == 1) {
                if (a5 <= ell - g + 1 + delta) {
                    x0 = g + a5 - 1;
                    y0 = g;
                } else {
                    x0 = NUM_TWO * ell + NUM_TWO * delta + NUM_TWO - g - a5;
                    y0 = ell + 1 + (ell % NUM_TWO) - g;
                }
            } else {
                if (a5 >= (g + 1 + delta - (ell % NUM_TWO))) {
                    x0 = g + ell + NUM_TWO * delta + 1 - (ell % NUM_TWO) - a5;
                    y0 = g;
                } else {
                    x0 = a5 + ell - g + (ell % NUM_TWO);
                    y0 = ell + 1 + (ell % NUM_TWO) - g;
                }
            }

            if (y0 >= 1 && y0 <= ell && y0 <= x0 && x0 <= ell + delta) {
                coordinate.batchId = b1 * k + w;
                coordinate.s1Idx = x0 + t * k;
                coordinate.s2Idx = y0 + t * k;
                return;
            }
        }
    }
    coordinate.batchId = -1;
    return;
}

__aicore__ inline void CalCausalSwizzleIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                             CoordinateInfo& coordinate)
{
    // 按照相邻B或者N拼接成一个完整S1S2
    int64_t nNew = n + 1;
    if (m != n) {
        nNew = (n - m + NUM_TWO) + (n + 1);
    }
    int64_t bNew = b >> 1;
    CalDenseSwizzleIndex(k, m, nNew, bNew, j, r, coordinate);
    if (coordinate.batchId == -1) {
        return;
    }

    int64_t w = coordinate.batchId;
    int64_t x = coordinate.s1Idx;
    int64_t y = coordinate.s2Idx;
    if (m == n) {
        if (y >= x + 1) {
            y = (n << 1) - m - y + NUM_TWO;
            x = m + 1 - x;
            w = (w << 1);
        } else {
            w = (w << 1) - 1;
        }
    } else {
        if (y >= x + (n + 1) - m + 1) {
            y = ((n + 1) << 1) - m - y + NUM_TWO;
            x = m + 1 - x;
            w = (w << 1);
        } else {
            w = (w << 1) - 1;
        }
    }

    // Check if all values are within the valid ranges
    if (w >= 1 && w <= b && x >= 1 && x <= m && y >= 1 && y <= n) {
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
    } else {
        coordinate.batchId = -1;
    }
}

__aicore__ inline int64_t CalLeftUpCausalSwizzleMaxRound(int64_t k, int64_t m, int64_t n, int64_t b)
{
    int64_t pairCount = b >> 1;
    if (k <= 0 || m <= 0 || n <= 0 || pairCount <= 0) {
        return 0;
    }

    int64_t virtualM = m;
    int64_t virtualN = m + 1;
    int64_t activeK = Min(k, m * pairCount);
    if (m > n) {
        virtualM = NUM_TWO * m - n + 1;
        virtualN = n;
        activeK = Min(k, n * pairCount);
    }
    return virtualM * Ceil<int64_t>(virtualN * pairCount, activeK);
}

__aicore__ inline void CalLeftUpCausalSwizzleIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                                   CoordinateInfo& coordinate)
{
    coordinate.batchId = -1;
    int64_t pairCount = b >> 1;
    if (k <= 0 || m <= 0 || n <= 0 || pairCount <= 0 || j < 1 || r < 1) {
        return;
    }

    if (m <= n) {
        int64_t activeK = Min(k, m * pairCount);
        if (j > activeK) {
            return;
        }
        CalCausalSwizzleIndex(activeK, m, m, b, j, r, coordinate);
        return;
    }

    int64_t virtualM = NUM_TWO * m - n + 1;
    int64_t activeK = Min(k, n * pairCount);
    if (j > activeK) {
        return;
    }
    int64_t columnId = (r - 1) / virtualM * activeK + j - 1;
    if (columnId >= n * pairCount) {
        return;
    }

    int64_t pairId = columnId / n + 1;
    int64_t virtualS2 = columnId % n + 1;
    int64_t virtualS1 = (r - 1) % virtualM + 1;
    int64_t oddBatchLen = m - virtualS2 + 1;
    if (virtualS1 <= oddBatchLen) {
        coordinate.batchId = NUM_TWO * pairId - 1;
        coordinate.s1Idx = virtualS2 + virtualS1 - 1;
        coordinate.s2Idx = virtualS2;
        return;
    }

    int64_t evenBatchOffset = virtualS1 - oddBatchLen;
    coordinate.batchId = NUM_TWO * pairId;
    coordinate.s1Idx = m - evenBatchOffset + 1;
    coordinate.s2Idx = n - virtualS2 + 1;
}

__aicore__ inline void CalGQACausalIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r, int64_t g,
                                         CoordinateInfo& coordinate, bool exactRoundBoundary = false)
{
    // 1. b1, b2, rm1
    int64_t b1 = b / k;
    int64_t b2 = b % k;
    int64_t delta = m - n;
    int64_t sizeTri = n * (m + delta + 1) / NUM_TWO;
    int64_t sizeTriGroup = sizeTri * g;
    int64_t rm1 = b1 * sizeTriGroup;

    int64_t w, x, y = 0;
    // --- 第一段：完整矩形 ---
    if (r <= rm1) {
        int64_t a = r % sizeTriGroup;
        a = a != 0 ? a : sizeTriGroup;
        int64_t N1_id = Ceil<int64_t>(a, sizeTri);
        a = r % sizeTri;
        a = a != 0 ? a : sizeTri;
        int64_t b_id = k * (Ceil<int64_t>(r, sizeTriGroup) - 1) + j;

        int64_t n1 = n / NUM_TWO * NUM_TWO;
        int64_t L = NUM_TWO * m - n1 + 1;
        int64_t rmLocal = (n1 * L) / NUM_TWO;
        if (a <= rmLocal) {
            y = Ceil<int64_t>(a, L);
            int64_t r1 = a % L != 0 ? a % L : L;
            x = r1 + y - 1;
            if (x > m) {
                x = NUM_TWO * m + 1 - x;
                y = n1 + 1 - y;
            }
        } else {
            int64_t a1 = a - rmLocal;
            y = n;
            x = a1 - 1 + y;
        }

        b_id = ((b_id - 1) / k) * k + ((y - 1 + (b_id - 1)) % k) + 1;
        w = (b_id - 1) * g + N1_id;

        // w = ((w - 1) % b1) * k + (w - 1) / b1 + 1;
        // w = ((w - 1) / k) * k + ((y - 1 + (w - 1)) % k) + 1;
        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
        return;
    }

    // 2. 行块压平分给 floor(b2/2) 个 batch
    int64_t t = n / k;
    int64_t ell = n % k;
    int64_t a2 = r - rm1;
    int64_t half_ = b2 / NUM_TWO;
    int64_t rm2 = (NUM_TWO * m - t * k + 1) * t * half_ * g;
    if (a2 >= 1 && a2 <= rm2) {
        // 调用 dense 子块
        CalGQADenseIndex(k, NUM_TWO * m - t * k + 1, t * k, half_, j, a2, g, coordinate);
        int64_t wSub = coordinate.batchId;
        int64_t xSub = coordinate.s1Idx;
        int64_t ySub = coordinate.s2Idx;
        // 新增范围检查
        int64_t max_x = NUM_TWO * m - t * k + 1;
        int64_t max_y = t * k;
        int64_t max_w = half_ * g;
        if (xSub >= 1 && xSub <= max_x && ySub >= 1 && ySub <= max_y && wSub >= 1 && wSub <= max_w) {
            // 计算映射
            int64_t b_id = Ceil<int64_t>(wSub, g);
            int64_t N1_id = wSub % g;
            N1_id = N1_id != 0 ? N1_id : g;
            if (xSub - ySub <= m - t * k) {
                b_id = NUM_TWO * b_id - 1;
                x = m + 1 - xSub;
                y = t * k + 1 - ySub;
            } else {
                b_id = NUM_TWO * b_id;
                x = xSub - m + t * k - 1;
                y = ySub;
            }
            w = b1 * k * g + (b_id - 1) * g + N1_id;
            coordinate.batchId = w;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
        } else {
            coordinate.batchId = -1;
        }
        return;
    }

    // 3. 剩余，分两种：b2 mod2 == 1 或 else
    int64_t a3 = r - rm1 - rm2;
    int64_t rm3 = 0;
    if (b2 % NUM_TWO == 1) {
        int64_t t1 = n / (NUM_TWO * k);
        int64_t t3 = t % NUM_TWO;
        // 奇数 b2
        if (t3 == 1) {
            int64_t m1 = m - t1 * NUM_TWO * k;
            rm3 = (m + m1 + 1) * t1;
            rm3 += ell == 0 ? m1 : Max(m1, NUM_TWO * m1 - NUM_TWO * k + 1);
            if (a3 >= 1 && a3 <= rm3 * g) {
                int64_t N1_id = Ceil<int64_t>(a3, rm3);
                a3 = a3 % rm3;
                a3 = a3 != 0 ? a3 : rm3;
                CalCausalSingleBatchDeterIndex(k, m, n, j, a3, coordinate, exactRoundBoundary);
                if (coordinate.batchId != -1) {
                    coordinate.batchId = (b - 1) * g + N1_id;
                    return;
                } else {
                    return;
                }
            }
            b -= 1;
            b2 -= 1;
        } else {
            // 偶数 b2
            rm3 = (NUM_TWO * m - t * k + 1) * t / NUM_TWO;
            if (a3 >= 1 && a3 <= rm3 * g) {
                int64_t N1_id = Ceil<int64_t>(a3, rm3);
                a3 = a3 % rm3;
                a3 = a3 != 0 ? a3 : rm3;
                CalCausalSingleBatchDeterIndex(k, m, t * k, j, a3, coordinate, exactRoundBoundary);
                if (coordinate.batchId != -1) {
                    coordinate.batchId = (b - 1) * g + N1_id;
                    return;
                } else {
                    return;
                }
            }
        }
    }

    // 4. 零碎余数 ℓ 部分
    int64_t a4 = a3 - rm3 * g;
    int64_t N1_id = a4 % g;
    N1_id = N1_id != 0 ? N1_id : g;
    a4 = Ceil<int64_t>(a4, g);
    int64_t p = Ceil<int64_t>(ell, NUM_TWO);
    int64_t ell1 = ell + 1 - (ell % NUM_TWO);
    int64_t block = (b2 * p) / k;
    int64_t res0 = (b2 * p) % k;
    // 情况 A
    if (a4 > block * (ell1 + NUM_TWO * delta) && res0 <= k / NUM_TWO) {
        int64_t offs = a4 - block * (ell1 + NUM_TWO * delta);
        // 前 res0 cores
        if (j >= 1 && j <= res0) {
            int64_t limit = (res0 - j) / b2 + Ceil<int64_t>(ell1, NUM_TWO) + delta;
            if (offs <= limit) {
                w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j) % b2;
                w = w != 0 ? w : b2;
                y = p - (res0 - j) / b2;
                x = y + offs - 1;
                if (y >= 1 && y <= ell && y <= x && x <= (ell + delta)) {
                    w = (w - 1) * g + N1_id + b1 * k * g;
                    coordinate.batchId = w;
                    coordinate.s1Idx = x + t * k;
                    coordinate.s2Idx = y + t * k;
                    return;
                }
            }
        }

        // 后 res0 cores
        if (k - res0 + 1 <= j && j <= k) {
            int64_t idx = j - (k - res0 + 1);
            int64_t limit = (ell1 - 1) / NUM_TWO - idx / b2 + delta;
            if (offs <= limit) {
                w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + k + 1 - j) % b2;
                w = w != 0 ? w : b2;
                y = p + 1 + idx / b2;
                x = y + offs - 1;
                if (y >= 1 && y <= ell && y <= x && x <= (ell + delta)) {
                    w = (w - 1) * g + N1_id + b1 * k * g;
                    coordinate.batchId = w;
                    coordinate.s1Idx = x + t * k;
                    coordinate.s2Idx = y + t * k;
                    return;
                }
            }
        }
        coordinate.batchId = -1;
        return;
    } else {
        // 情况 B
        w = (k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j) % b2;
        w = w != 0 ? w : b2;
        int64_t gTail = Ceil<int64_t>((k * ((a4 - 1) / (ell1 + NUM_TWO * delta)) + j), b2);
        if (gTail >= 1 && gTail <= p) {
            int64_t a5 = a4 % (ell1 + NUM_TWO * delta);
            a5 = a5 != 0 ? a5 : (ell1 + NUM_TWO * delta);
            int64_t x0, y0 = 0;
            if (gTail % NUM_TWO == 1) {
                if (a5 <= ell - gTail + 1 + delta) {
                    x0 = gTail + a5 - 1;
                    y0 = gTail;
                } else {
                    x0 = NUM_TWO * ell + NUM_TWO * delta + NUM_TWO - gTail - a5;
                    y0 = ell + 1 + (ell % NUM_TWO) - gTail;
                }
            } else {
                if (a5 >= (gTail + 1 + delta - (ell % NUM_TWO))) {
                    x0 = gTail + ell + NUM_TWO * delta + 1 - (ell % NUM_TWO) - a5;
                    y0 = gTail;
                } else {
                    x0 = a5 + ell - gTail + (ell % NUM_TWO);
                    y0 = ell + 1 + (ell % NUM_TWO) - gTail;
                }
            }

            if (y0 >= 1 && y0 <= ell && y0 <= x0 && x0 <= ell + delta) {
                w = (w - 1) * g + N1_id + b1 * k * g;
                coordinate.batchId = w;
                coordinate.s1Idx = x0 + t * k;
                coordinate.s2Idx = y0 + t * k;
                return;
            }
        }
    }
    coordinate.batchId = -1;
    return;
}

// Keep the single-batch GQA column schedule and serialize the flattened (B, N2) batches.
// j, r and the returned coordinates are 1-based; roundsPerBatch includes all g query heads.
__aicore__ inline void CalGQACausalBatchMajorIndex(int64_t k, int64_t m, int64_t n, int64_t b, int64_t j, int64_t r,
                                                   int64_t g, int64_t roundsPerBatch, CoordinateInfo& coordinate)
{
    coordinate.batchId = -1;
    if (k <= 0 || n <= 0 || m < n || b <= 0 || g <= 0 || j < 1 || j > k || r <= 0 || roundsPerBatch <= 0) {
        return;
    }
    int64_t batchIdx = (r - 1) / roundsPerBatch;
    if (batchIdx >= b) {
        return;
    }
    int64_t localRound = (r - 1) % roundsPerBatch + 1;
    CalGQACausalIndex(k, m, n, 1, j, localRound, g, coordinate, true);
    // Never offset the invalid-slot sentinel into a valid flattened head index.
    if (coordinate.batchId < 1 || coordinate.batchId > g || coordinate.s2Idx < 1 || coordinate.s2Idx > n ||
        coordinate.s1Idx < coordinate.s2Idx || coordinate.s1Idx > m) {
        coordinate.batchId = -1;
        return;
    }
    coordinate.batchId += batchIdx * g;
}

struct BandInfo {
    int64_t k = 0;
    int64_t m = 0;
    int64_t n = 0;
    int64_t p = 0;
    int64_t q = 0;
    int64_t b = 0;
    int64_t b1 = 0;
    int64_t b2 = 0;
    int64_t L1 = 0;
    int64_t L2 = 0;
    int64_t L3 = 0;
    int64_t nSeg = 0;
    int64_t R1 = 0;
    int64_t R2 = 0;
    int64_t R3 = 0;
    int64_t Rm = 0;
    int64_t rm = 0;
    int64_t rm2 = 0;
    int64_t hybridPairCount = 0;
};

__aicore__ inline void GenBandHybridInfo(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b,
                                         BandInfo& bandInfo)
{
    p = Min(p, m);
    q = Min(q, n);
    m = Min(m, n + p - 1);
    n = Min(n, m + q - 1);

    bandInfo.k = k;
    bandInfo.m = m;
    bandInfo.n = n;
    bandInfo.p = p;
    bandInfo.q = q;
    bandInfo.b = b;
    if (p + q <= m) {
        bandInfo.L1 = q - 1;
        bandInfo.L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        bandInfo.L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));
    } else {
        bandInfo.L1 = m - p;
        bandInfo.L2 = p + q - m;
        bandInfo.L3 = Min(n - q, m - 1);
    }
    bandInfo.nSeg = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
    bandInfo.hybridPairCount = Max(0, Min(bandInfo.L1, bandInfo.L3 - p + 1));
}

// Both orientations retain the causal mapper's fixed owner for each KV column and unique
// Q row per round. For upper BAND, rotate both axes; a transpose would exchange these invariants.
__aicore__ inline void CalBandCausalIndex(const BandInfo& bandInfo, int64_t j, int64_t r, bool upper,
                                          CoordinateInfo& coordinate)
{
    const int64_t k = bandInfo.k;
    const int64_t b = bandInfo.b;
    const int64_t causalShift = upper ? bandInfo.p - 1 : bandInfo.q - 1;
    const int64_t causalSize = (upper ? bandInfo.n : bandInfo.m) + causalShift;
    const int64_t causalPairCount = b / NUM_TWO;
    const int64_t pairedBatch = causalPairCount * NUM_TWO;
    const int64_t causalK = Min(k, causalSize * causalPairCount);
    const int64_t pairRound =
        causalPairCount > 0 ? Ceil<int64_t>((causalSize + 1) * causalPairCount, causalK) * causalSize : 0;
    if ((b & 1) != 0 && r > pairRound) {
        // Keep the tail group boundary exact for both orientations, including the first round.
        CalCausalSingleBatchDeterIndex(k, causalSize, causalSize, j, r - pairRound, coordinate, true);
        if (coordinate.batchId != -1) {
            coordinate.batchId = b;
        }
    } else {
        CalCausalSwizzleIndex(k, causalSize, causalSize, pairedBatch, j, r, coordinate);
    }
    if (coordinate.batchId == -1) {
        return;
    }
    if (upper) {
        coordinate.s1Idx = causalSize + 1 - coordinate.s1Idx;
        coordinate.s2Idx = bandInfo.n + 1 - coordinate.s2Idx;
    } else {
        coordinate.s1Idx -= causalShift;
    }
    if (coordinate.s1Idx < 1 || coordinate.s1Idx > bandInfo.m || coordinate.s2Idx < 1 ||
        coordinate.s2Idx > bandInfo.n) {
        coordinate.batchId = -1;
    }
}

__aicore__ inline bool SetBandHybridCoordinate(const BandInfo& bandInfo, int64_t w, int64_t y, int64_t localRound,
                                               CoordinateInfo& coordinate)
{
    int64_t x = y + localRound - bandInfo.q;
    int64_t xMin = Max(1, y - bandInfo.q + 1);
    int64_t xMax = Min(bandInfo.m, y + bandInfo.p - 1);
    if (x < xMin || x > xMax) {
        return false;
    }
    coordinate.batchId = w;
    coordinate.s1Idx = x;
    coordinate.s2Idx = y;
    return true;
}

__aicore__ inline void CalBandHybridIndex(const BandInfo& bandInfo, int64_t j, int64_t r, CoordinateInfo& coordinate)
{
    coordinate.batchId = -1;
    int64_t slot = bandInfo.p + bandInfo.q - 1;
    int64_t colsPerBatch = bandInfo.nSeg - bandInfo.hybridPairCount;
    if (j < 1 || j > bandInfo.k || r < 1 || slot <= 0 || colsPerBatch <= 0) {
        return;
    }

    int64_t layer = (r - 1) / slot;
    int64_t localRound = (r - 1) % slot + 1;
    int64_t globalColumn = layer * bandInfo.k + j;
    if (globalColumn > bandInfo.b * colsPerBatch) {
        return;
    }
    int64_t w = (globalColumn - 1) / colsPerBatch + 1;
    int64_t localColumn = (globalColumn - 1) % colsPerBatch + 1;

    if (localColumn <= bandInfo.hybridPairCount) {
        int64_t y = localColumn;
        if (SetBandHybridCoordinate(bandInfo, w, y, localRound, coordinate)) {
            return;
        }
        int64_t seg3Index = bandInfo.p + localColumn - 1;
        y = bandInfo.L1 + bandInfo.L2 + seg3Index;
        SetBandHybridCoordinate(bandInfo, w, y, localRound, coordinate);
        return;
    }

    if (localColumn <= bandInfo.hybridPairCount + bandInfo.L2) {
        int64_t y = bandInfo.L1 + localColumn - bandInfo.hybridPairCount;
        SetBandHybridCoordinate(bandInfo, w, y, localRound, coordinate);
        return;
    }

    int64_t singleIndex = localColumn - bandInfo.hybridPairCount - bandInfo.L2;
    int64_t unpairedSeg1Count = bandInfo.L1 - bandInfo.hybridPairCount;
    if (singleIndex <= unpairedSeg1Count) {
        int64_t y = bandInfo.hybridPairCount + singleIndex;
        SetBandHybridCoordinate(bandInfo, w, y, localRound, coordinate);
        return;
    }

    int64_t unpairedSeg3Index = singleIndex - unpairedSeg1Count;
    int64_t unpairedSeg3Prefix = Min(bandInfo.p - 1, bandInfo.L3);
    int64_t seg3Index =
        unpairedSeg3Index <= unpairedSeg3Prefix ? unpairedSeg3Index : unpairedSeg3Index + bandInfo.hybridPairCount;
    int64_t y = bandInfo.L1 + bandInfo.L2 + seg3Index;
    SetBandHybridCoordinate(bandInfo, w, y, localRound, coordinate);
}

__aicore__ inline void GenBandInfo(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, BandInfo& bandInfo)
{
    bandInfo.k = k;
    bandInfo.m = m;
    bandInfo.n = n;
    bandInfo.p = p;
    bandInfo.q = q;
    bandInfo.b = b;

    // 1. initial b1, b2
    bandInfo.b1 = b / k;
    bandInfo.b2 = b % k;
    // p+q>m的分支
    if (p + q > m) {
        // 2. lengths
        bandInfo.L1 = m - p;
        bandInfo.L2 = p + q - m;
        bandInfo.L3 = Min(m - 1, n - q);
        // 3. redefine nSeg
        bandInfo.nSeg = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
        // 4. segment sizes
        bandInfo.R1 = (p + m - 1) * bandInfo.L1 / NUM_TWO;
        bandInfo.R2 = m * bandInfo.L2;
        bandInfo.R3 = (NUM_TWO * m - 1 - bandInfo.L3) * bandInfo.L3 / NUM_TWO;
        bandInfo.Rm = bandInfo.R1 + bandInfo.R2 + bandInfo.R3;
        // 5. rm = b1 * Rm
        bandInfo.rm = bandInfo.b1 * bandInfo.Rm;
        bandInfo.rm2 = m * Ceil<int64_t>(n * b, Min(k, b * m));
    } else {
        // p+q<=m的分支
        // 2. lengths
        bandInfo.L1 = q - 1;
        bandInfo.L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        bandInfo.L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));
        // 3. redefine nSeg
        bandInfo.nSeg = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
        // 4. segment sizes
        bandInfo.R1 = (NUM_TWO * p - NUM_TWO + q) * bandInfo.L1 / NUM_TWO;
        bandInfo.R2 = (p + q - 1) * bandInfo.L2;
        bandInfo.R3 = (p + q - NUM_TWO) * bandInfo.L3 - (bandInfo.L3 * (bandInfo.L3 - 1)) / NUM_TWO;
        bandInfo.Rm = bandInfo.R1 + bandInfo.R2 + bandInfo.R3;
        // 5. rm = b1 * Rm
        bandInfo.rm = bandInfo.b1 * bandInfo.Rm;
        bandInfo.rm2 = Ceil<int64_t>(n * b, k) * (p + q - 1);
    }
}

__aicore__ inline void CalGQADenseIndexNoTune(int64_t k, int64_t m, int64_t n, int64_t b, int64_t core_id,
                                              int64_t round_id, int64_t g, CoordinateInfo& coordinate)
{
    coordinate.batchId = -1;
    k = Min(Min(k, b * g * m), b * n);
    int64_t R = Max(Max(Ceil<int64_t>(b * n * g, k), Ceil<int64_t>(n, m)), g);
    if (core_id < 1 || core_id > k || round_id < 1 || round_id > R * m) {
        return;
    }

    int64_t ID = (core_id - 1) * R + Ceil<int64_t>(round_id, m);
    int64_t local_id = round_id % m;
    local_id = local_id != 0 ? local_id : m;
    if (ID > g * n * b) {
        return;
    }

    int64_t N = b * g;
    int64_t b_id = ID % N;
    b_id = b_id != 0 ? b_id : N;
    b_id = Ceil<int64_t>(b_id, g);
    coordinate.s1Idx = local_id;
    coordinate.s2Idx = Ceil<int64_t>(ID, N);
    int64_t w = ID % g;
    w = w != 0 ? w : g;
    coordinate.batchId = w + (b_id - 1) * g;
}

// Phase geometry and round boundaries are selected and serialized by the Host.
// All coordinates remain 1-based until CalBandDeterIndex restores the physical tensor indices.
template <bool IncludeColumnOwners = true>
__aicore__ inline void CalGQABandPiecewiseIndex(const __gm__ optiling::fag::GqaBandSchedule& plan, int64_t j, int64_t r,
                                                CoordinateInfo& coordinate)
{
    coordinate.batchId = -1;
    if (j < 1 || j > plan.k || r < 1 || r > plan.totalRounds || plan.groupsPerChunk <= 0 || plan.fullRounds <= 0) {
        return;
    }
    const int64_t fullChunks = plan.b / plan.groupsPerChunk;
    const int64_t fullRounds = fullChunks * plan.fullRounds;
    const bool isTail = r > fullRounds;
    const int64_t chunk = isTail ? fullChunks : (r - 1) / plan.fullRounds;
    const int64_t groups = isTail ? plan.b % plan.groupsPerChunk : plan.groupsPerChunk;
    int64_t localRound = isTail ? r - fullRounds : (r - 1) % plan.fullRounds + 1;
    const bool useTailPhases = isTail && plan.tailPhaseCount > 0;
    const int64_t phaseCount = useTailPhases ? plan.tailPhaseCount : plan.phaseCount;
    for (int64_t phaseIdx = 0; phaseIdx < phaseCount; ++phaseIdx) {
        const __gm__ optiling::fag::GqaBandPhase& gmPhase =
            useTailPhases ? plan.tailPhases[phaseIdx] : plan.phases[phaseIdx];
        const int64_t phaseRounds = isTail ? gmPhase.tailRounds : gmPhase.fullRounds;
        if (localRound > phaseRounds) {
            localRound -= phaseRounds;
            continue;
        }
        // Load only the selected phase's geometry; never materialize both phase arrays locally.
        optiling::fag::GqaBandPhase phase;
        phase.kind = gmPhase.kind;
        phase.m = gmPhase.m;
        phase.n = gmPhase.n;
        phase.rowOffset = gmPhase.rowOffset;
        phase.colOffset = gmPhase.colOffset;
        // Current host routes single-domain owners to dedicated template keys.
        // Keep the complete mapper available for host/device mapping oracles.
        if constexpr (IncludeColumnOwners) {
            if (phase.kind == optiling::fag::GqaBandPhaseKind::FOLDED_UPPER_COLUMN_OWNER ||
                phase.kind == optiling::fag::GqaBandPhaseKind::FOLDED_LOWER_COLUMN_OWNER ||
                phase.kind == optiling::fag::GqaBandPhaseKind::BAND_FOLDED_COLUMN_OWNER) {
                const auto domain = optiling::fag::DecodeGqaBandFoldedDomain(phase);
                optiling::fag::GqaBandFoldedColumnOwnerIndex(plan.k, groups, plan.g, domain.m, domain.n, domain.p,
                                                             domain.q, j, localRound, coordinate.batchId,
                                                             coordinate.s1Idx, coordinate.s2Idx);
                if (coordinate.batchId > 0) {
                    coordinate.batchId += chunk * plan.groupsPerChunk * plan.g;
                    coordinate.s1Idx += domain.rowOrigin;
                    coordinate.s2Idx += domain.colOrigin;
                }
                return;
            }
            if (phase.kind == optiling::fag::GqaBandPhaseKind::BAND_COLUMN_OWNER) {
                // A whole-band phase stores p/q in the offset fields and returns physical block coordinates.
                if (plan.groupsPerChunk == plan.b && plan.phaseCount == 1 && plan.tailPhaseCount == 0) {
                    optiling::fag::GqaBandColumnOwnerIndex(plan.k, groups, plan.g, phase.m, phase.n, phase.rowOffset,
                                                           phase.colOffset, j, localRound, coordinate.batchId,
                                                           coordinate.s1Idx, coordinate.s2Idx);
                }
                return;
            }
            if (phase.kind == optiling::fag::GqaBandPhaseKind::LOWER_TRIANGLE_COLUMN_OWNER) {
                if (plan.groupsPerChunk == plan.b && plan.phaseCount == 1 && plan.tailPhaseCount == 0 &&
                    phase.m == phase.n) {
                    optiling::fag::GqaBandColumnOwnerIndex(plan.k, groups, plan.g, phase.m, phase.n, phase.m, 1, j,
                                                           localRound, coordinate.batchId, coordinate.s1Idx,
                                                           coordinate.s2Idx);
                    if (coordinate.batchId > 0) {
                        coordinate.s1Idx += phase.rowOffset;
                        coordinate.s2Idx += phase.colOffset;
                    }
                }
                return;
            }
        }
        const bool isTriangle = phase.kind == optiling::fag::GqaBandPhaseKind::LOWER_TRIANGLE ||
                                phase.kind == optiling::fag::GqaBandPhaseKind::UPPER_TRIANGLE;
        const bool isEdgePair = optiling::fag::IsGqaBandEdgePair(phase);
        int64_t localGroups = groups;
        int64_t phaseGroupOffset = 0;
        if (isTriangle) {
            // Reorder whole rounds on every core together, preserving shared dK/dV uniqueness.
            const int64_t sourceRound =
                optiling::fag::GqaBandTriangleSourceRound(plan.k, phase.m, groups, plan.g, localRound);
            CalGQACausalIndex(plan.k, phase.m, phase.n, groups, j, sourceRound, plan.g, coordinate, true);
        } else if (phase.kind == optiling::fag::GqaBandPhaseKind::DIAGONAL_COLUMN_OWNER) {
            optiling::fag::GqaBandDiagonalColumnIndex(plan.k, phase.m, groups, plan.g, j, localRound,
                                                      coordinate.batchId, coordinate.s1Idx, coordinate.s2Idx);
        } else if (phase.kind == optiling::fag::GqaBandPhaseKind::DIAGONAL) {
            // Keep the local row equal on all cores before the column-dependent row translation.
            CalGQADenseIndexNoTune(plan.k, phase.m, phase.n, groups, j, localRound, plan.g, coordinate);
        } else {
            int64_t sourceRound = localRound;
            if (isEdgePair) {
                sourceRound = optiling::fag::GqaBandEdgePairSourceRound(phase, plan.k, groups, plan.g, localRound);
            } else if (phase.kind == optiling::fag::GqaBandPhaseKind::RECTANGLE_BATCH_MAJOR) {
                // The host planner proves equal rounds and identical per-column core ownership.
                const int64_t roundsPerGroup = phaseRounds / groups;
                phaseGroupOffset = (localRound - 1) / roundsPerGroup;
                sourceRound = (localRound - 1) % roundsPerGroup + 1;
                localGroups = 1;
            }
            CalGQADenseIndex(plan.k, phase.m, phase.n, localGroups, j, sourceRound, plan.g, coordinate);
        }
        // Never translate an invalid slot into a valid flattened query-head index.
        if (coordinate.batchId < 1 || coordinate.batchId > localGroups * plan.g || coordinate.s1Idx < 1 ||
            coordinate.s1Idx > phase.m || coordinate.s2Idx < 1 || coordinate.s2Idx > phase.n ||
            (isTriangle && coordinate.s1Idx < coordinate.s2Idx)) {
            coordinate.batchId = -1;
            return;
        }
        if (phase.kind == optiling::fag::GqaBandPhaseKind::UPPER_TRIANGLE) {
            // Reverse both axes; transposing a GQA schedule would violate shared dK/dV uniqueness.
            coordinate.s1Idx = phase.m + 1 - coordinate.s1Idx;
            coordinate.s2Idx = phase.n + 1 - coordinate.s2Idx;
        } else if (phase.kind == optiling::fag::GqaBandPhaseKind::DIAGONAL) {
            coordinate.s1Idx += coordinate.s2Idx - 1;
        }
        bool applyOffset = true;
        if (isEdgePair) {
            applyOffset = coordinate.s1Idx > optiling::fag::GqaBandEdgePairSplit(phase) + coordinate.s2Idx;
        }
        if (applyOffset) {
            coordinate.s1Idx += phase.rowOffset;
            coordinate.s2Idx += phase.colOffset;
        }
        coordinate.batchId += (chunk * plan.groupsPerChunk + phaseGroupOffset) * plan.g;
        return;
    }
}

// Overlap disjoint rectangle and triangle columns inside each physical KV group.
// Every hardware core still participates in the existing global round/flush protocol;
// an unused core or an earlier-finishing group returns an empty coordinate only.
__aicore__ inline void CalGQABandOverlapIndex(const __gm__ optiling::fag::GqaBandSchedule& plan, int64_t j, int64_t r,
                                              CoordinateInfo& coordinate)
{
    coordinate.batchId = -1;
    const int64_t h = plan.groupsPerChunk;
    if (h <= 0 || h > plan.b || plan.fullRounds <= 0 || plan.phaseCount != 1 || plan.tailPhaseCount < 0 ||
        plan.tailPhaseCount > 1 || r < 1 || r > plan.totalRounds || j < 1 || j > plan.k) {
        return;
    }
    const int64_t fullChunks = plan.b / h;
    const int64_t fullEnd = fullChunks * plan.fullRounds;
    const bool tail = r > fullEnd;
    const int64_t chunk = tail ? fullChunks : (r - 1) / plan.fullRounds;
    const int64_t groups = tail ? plan.b % h : h;
    const int64_t localRound = tail ? r - fullEnd : r - chunk * plan.fullRounds;
    const auto& phase = tail && plan.tailPhaseCount == 1 ? plan.tailPhases[0] : plan.phases[0];
    const int64_t phaseRounds = tail ? phase.tailRounds : phase.fullRounds;
    const bool upper = phase.kind == optiling::fag::GqaBandPhaseKind::OVERLAP_UPPER;
    const int64_t m = phase.m;
    const int64_t n = phase.n;
    if ((!upper && phase.kind != optiling::fag::GqaBandPhaseKind::OVERLAP_LOWER) || localRound > phaseRounds ||
        !optiling::fag::CanUseGqaBandOverlap(plan.k, groups, plan.g, m, n)) {
        return;
    }
    int64_t group = -1;
    int64_t localCore = 0;
    int64_t localK = 0;
    if (!optiling::fag::GqaBandOverlapCoreIndex(plan.k, groups, m, n, j, group, localCore, localK)) {
        return;
    }
    const int64_t triangleCores = m / 2;
    const int64_t stripeColumns = localK - triangleCores;
    const int64_t prefixRounds = m * plan.g;
    if (localRound <= prefixRounds) {
        if (localCore <= triangleCores) {
            const int64_t side = m - 1;
            if (localRound > optiling::fag::GqaBandOverlapTriangleRounds(m, plan.g)) {
                return;
            }
            const int64_t sourceRound =
                optiling::fag::GqaBandTriangleSourceRound(triangleCores, side, 1, plan.g, localRound);
            CalGQACausalIndex(triangleCores, side, side, 1, localCore, sourceRound, plan.g, coordinate, true);
            if (coordinate.batchId > 0) {
                coordinate.s1Idx += 1;
                coordinate.s2Idx += n;
            }
        } else {
            const int64_t lane = localCore - triangleCores - 1;
            // Shift the stripe's query head by one relative to the triangle.
            // Their KV columns are disjoint; distinct heads separate dQ writes.
            coordinate.batchId = localRound % plan.g + 1;
            coordinate.s1Idx = ((localRound - 1) / plan.g + lane) % m + 1;
            coordinate.s2Idx = lane + 1;
        }
    } else {
        if (n <= stripeColumns) {
            return;
        }
        // Dense's own round bound also handles groups that finish before the
        // common chunk length. Do not invoke Dense for a zero-width suffix.
        CalGQADenseIndex(localK, m, n - stripeColumns, 1, localCore, localRound - prefixRounds, plan.g, coordinate);
        if (coordinate.batchId > 0) {
            coordinate.s2Idx += stripeColumns;
        }
    }
    if (coordinate.batchId < 1) {
        return;
    }
    if (upper) {
        coordinate.s1Idx = m + 1 - coordinate.s1Idx;
        coordinate.s2Idx = n + m - coordinate.s2Idx;
    }
    coordinate.s1Idx += phase.rowOffset;
    coordinate.s2Idx += phase.colOffset;
    coordinate.batchId += (chunk * h + group) * plan.g;
}

// A new owner template contains one complete-domain phase and no generic phase dispatcher.
template <bool Balanced, bool TranslatedTriangle = false, bool Folded = false>
__aicore__ inline void CalGQABandOwnerIndex(const __gm__ optiling::fag::GqaBandSchedule& plan, int64_t j, int64_t r,
                                            CoordinateInfo& coordinate)
{
    coordinate.batchId = -1;
    const int64_t h = plan.groupsPerChunk;
    if (h <= 0 || plan.fullRounds <= 0 || r < 1 || r > plan.totalRounds || j < 1 || j > plan.k) {
        return;
    }
    const int64_t fullChunks = plan.b / h;
    const int64_t fullEnd = fullChunks * plan.fullRounds;
    const bool tail = r > fullEnd;
    const int64_t chunk = tail ? fullChunks : (r - 1) / plan.fullRounds;
    const int64_t groups = tail ? plan.b % h : h;
    const int64_t localRound = tail ? r - fullEnd : r - chunk * plan.fullRounds;
    const auto& phase = plan.phases[0];
    if constexpr (Folded) {
        // Decode a local geometry value; the serialized phase remains in GM.
        optiling::fag::GqaBandPhase foldedPhase;
        foldedPhase.kind = phase.kind;
        foldedPhase.m = phase.m;
        foldedPhase.n = phase.n;
        foldedPhase.rowOffset = phase.rowOffset;
        foldedPhase.colOffset = phase.colOffset;
        const auto domain = optiling::fag::DecodeGqaBandFoldedDomain(foldedPhase);
        optiling::fag::GqaBandFoldedColumnOwnerIndex(plan.k, groups, plan.g, domain.m, domain.n, domain.p, domain.q, j,
                                                     localRound, coordinate.batchId, coordinate.s1Idx,
                                                     coordinate.s2Idx);
        if (coordinate.batchId > 0) {
            coordinate.batchId += chunk * h * plan.g;
            coordinate.s1Idx += domain.rowOrigin;
            coordinate.s2Idx += domain.colOrigin;
        }
        return;
    } else if constexpr (TranslatedTriangle) {
        optiling::fag::GqaBandColumnOwnerIndex(plan.k, groups, plan.g, phase.m, phase.n, phase.m, 1, j, localRound,
                                               coordinate.batchId, coordinate.s1Idx, coordinate.s2Idx);
    } else if constexpr (Balanced) {
        optiling::fag::GqaBandBalancedColumnOwnerIndex(plan.k, groups, plan.g, phase.m, phase.n, phase.rowOffset,
                                                       phase.colOffset, j, localRound, coordinate.batchId,
                                                       coordinate.s1Idx, coordinate.s2Idx);
    } else {
        optiling::fag::GqaBandColumnOwnerIndex(plan.k, groups, plan.g, phase.m, phase.n, phase.rowOffset,
                                               phase.colOffset, j, localRound, coordinate.batchId, coordinate.s1Idx,
                                               coordinate.s2Idx);
    }
    if (coordinate.batchId > 0) {
        coordinate.batchId += chunk * h * plan.g;
        if constexpr (TranslatedTriangle) {
            coordinate.s1Idx += phase.rowOffset;
            coordinate.s2Idx += phase.colOffset;
        }
    }
}

__aicore__ inline void GenGQABandInfo(int64_t k, int64_t m, int64_t n, int64_t p, int64_t q, int64_t b, int64_t g,
                                      BandInfo& bandInfo)
{
    bandInfo.k = k;
    bandInfo.m = m;
    bandInfo.n = n;
    bandInfo.p = p;
    bandInfo.q = q;
    bandInfo.b = b;

    // 1. initial b1, b2
    bandInfo.b1 = b / k;
    bandInfo.b2 = b % k;
    if (p + q > m) {
        bandInfo.L1 = m - p;
        bandInfo.L2 = p + q - m;
        bandInfo.L3 = Min(m - 1, n - q);
        // redefine n
        n = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
        bandInfo.R1 = (p + m - 1) * bandInfo.L1 / NUM_TWO;
        bandInfo.R2 = m * bandInfo.L2;
    } else {
        bandInfo.L1 = q - 1;
        bandInfo.L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        bandInfo.L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));
        // redefine m, n
        if (bandInfo.L3 == 0) {
            m = p + q + bandInfo.L2 - NUM_TWO;
        }
        n = bandInfo.L1 + bandInfo.L2 + bandInfo.L3;
    }
    bandInfo.n = n;
    // Rm_group内有列的轮次约束
    bandInfo.Rm = m * n - (m - p) * (m - p + 1) / NUM_TWO - (n - q) * (n - q + 1) / NUM_TWO;
    int64_t Rm_group = bandInfo.Rm * g;
    bandInfo.rm = bandInfo.b1 * Rm_group;

    if (p + q > m) {
        if (p + q <= n) {
            int64_t n1 = p + q - 1;
            bandInfo.rm2 = m * Max(Max(Ceil<int64_t>(bandInfo.b2 * n1 * g, k), Ceil<int64_t>(n1, m)), g);
        } else {
            bandInfo.rm2 = m * Max(Max(Ceil<int64_t>(bandInfo.b2 * n * g, k), Ceil<int64_t>(n, m)), g);
        }
    } else {
        if (n > m) {
            bandInfo.R1 =
                (p + q - 1) * Max(Max(Ceil<int64_t>(bandInfo.b2 * n * g, k), Ceil<int64_t>(n, (p + q - 1))), g);
            bandInfo.R2 =
                m * Max(Max(Ceil<int64_t>(bandInfo.b2 * (p + q - 1) * g, k), Ceil<int64_t>((p + q - 1), m)), g);
            bandInfo.rm2 = Min(bandInfo.R1, bandInfo.R2);
        } else {
            int64_t m1 = p + q - 1;
            bandInfo.rm2 = m1 * Max(Max(Ceil<int64_t>(bandInfo.b2 * n * g, k), Ceil<int64_t>(n, m1)), g);
        }
    }
}

__aicore__ inline void CalBandIndex(const BandInfo& bandInfo, int64_t j, int64_t r, CoordinateInfo& coordinate)
{
    int64_t w, x, y = 0;
    coordinate.batchId = -1;

    int64_t k = bandInfo.k;
    int64_t m = bandInfo.m;
    int64_t n = bandInfo.n;
    int64_t p = bandInfo.p;
    int64_t q = bandInfo.q;
    int64_t b = bandInfo.b;
    int64_t b1 = bandInfo.b1;
    int64_t b2 = bandInfo.b2;
    int64_t L1 = bandInfo.L1;
    int64_t L2 = bandInfo.L2;
    int64_t L3 = bandInfo.L3;
    int64_t nSeg = bandInfo.nSeg;
    int64_t R1 = bandInfo.R1;
    int64_t R2 = bandInfo.R2;
    int64_t R3 = bandInfo.R3;
    int64_t Rm = bandInfo.Rm;
    int64_t rm = bandInfo.rm;
    int64_t a;
    // p+q>m的分支
    if (p + q > m) {
        if (r <= rm) {
            a = r % Rm;
            w = Ceil<int64_t>(r, Rm) + b1 * (j - 1);
            if (a == 0) {
                a = Rm;
            }

            // subsegment selection
            if (a <= R1) {
                int64_t L11 = L1 / NUM_TWO * NUM_TWO;
                int64_t L = NUM_TWO * p + L11 - 1;
                int64_t local_round = L11 * L / NUM_TWO;
                if (a <= local_round) {
                    y = Ceil<int64_t>(a, L);
                    int64_t r1 = a % L;
                    r1 = r1 != 0 ? r1 : L;
                    x = p + y - r1;
                    if (x < 1) {
                        y = L11 + 1 - y;
                        x = 1 - x;
                    }
                } else {
                    x = a - local_round;
                    y = L1;
                }
            } else if (a <= R1 + R2) {
                int64_t a2 = a - R1;
                y = Ceil<int64_t>(a2, m);
                x = a2 % m;
                if (x == 0) {
                    x = m;
                }
                y = y + L1;
            } else {
                int64_t a3 = a - R1 - R2;

                int64_t L31 = L3 / NUM_TWO * NUM_TWO;
                int64_t L = NUM_TWO * m - L31 - 1;
                int64_t local_round = L31 * L / NUM_TWO;
                if (a3 <= local_round) {
                    y = Ceil<int64_t>(a3, L);
                    int64_t r1 = a3 % L;
                    r1 = r1 != 0 ? r1 : L;
                    x = y + r1;
                    if (x > m) {
                        y = L31 + 1 - y;
                        x = NUM_TWO * m + 1 - x;
                    }
                } else {
                    x = m - a3 + local_round + 1;
                    y = L3;
                }
                y = y + L1 + L2;
            }
            w = ((w - 1) % b1) * k + (w - 1) / b1 + 1;
            w = ((w - 1) / k) * k + ((y - 1 + (w - 1)) % k) + 1;
            coordinate.batchId = w;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
            return;
        }

        // --- second phase: overflow rows ---
        a = r - rm;

        CalDenseIndex(k, m, n, b2, j, a, coordinate);
        w = coordinate.batchId;
        if (w != -1) {
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;
            if (y <= m - p && x >= p + y) {
                return;
            }
            if (y > L1 + L2 && x <= y - L1 - L2) {
                return;
            }
            if (w > 0 && w < b2 + 1 && x > 0 && x < m + 1 && y > 0 && y < n + 1) {
                coordinate.batchId = b1 * k + w;
                coordinate.s1Idx = x;
                coordinate.s2Idx = y;
                return;
            }
        }
        return;
    }

    // p+q<=m的分支
    if (L3 == 0) {
        m = p + q + L2 - NUM_TWO;
    }
    if (r <= rm) {
        a = r % Rm;
        w = Ceil<int64_t>(r, Rm) + b1 * (j - 1);
        if (a == 0) {
            a = Rm;
        }

        if (a <= R1) {
            int64_t L11 = L1 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * p + L11 - 1;
            int64_t local_round = L11 * L / NUM_TWO;
            if (a <= local_round) {
                y = Ceil<int64_t>(a, L);
                int64_t r1 = a % L;
                r1 = r1 != 0 ? r1 : L;
                x = p + y - r1;
                if (x < 1) {
                    y = L11 + 1 - y;
                    x = 1 - x;
                }
            } else {
                x = a - local_round;
                y = L1;
            }
        } else if (a <= R1 + R2) {
            int64_t a2 = a - R1;
            y = Ceil<int64_t>(a2, (p + q - 1));
            x = a2 % (p + q - 1) + (y - 1);
            if (x == y - 1) {
                x = (p + q - 1) + y - 1;
            }
            y = y + L1;
        } else {
            int64_t a3 = a - R1 - R2;

            int64_t L31 = L3 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * (p + q) - L31 - NUM_THREE;
            int64_t local_round = L31 * L / NUM_TWO;
            if (a3 <= local_round) {
                y = Ceil<int64_t>(a3, L);
                int64_t r1 = a3 % L;
                r1 = r1 != 0 ? r1 : L;
                x = y + r1 + 1 + m - (p + q);
                if (x > m) {
                    y = L31 + 1 - y;
                    x = NUM_TWO * m + 1 - x;
                }
            } else {
                x = m - a3 + local_round + 1;
                y = L3;
            }
            y = y + L1 + L2;
        }

        coordinate.batchId = w;
        coordinate.s1Idx = x;
        coordinate.s2Idx = y;
        return;
    }

    if (b2 == 0) {
        coordinate.batchId = -1;
        return;
    }

    // --- second phase: overflow rows ---
    a = r - rm;
    int64_t seg = p + q - 1;
    int64_t a1 = Ceil<int64_t>(a, seg);
    int64_t a2 = a % seg;
    a2 = a2 == 0 ? seg : a2;

    // if no L3 or nSeg - m < 1
    if (L3 == 0 || nSeg - m < 1) {
        int64_t idx = (a1 - 1) * k + j;
        w = Ceil(idx, nSeg);
        y = idx % nSeg;
        if (y == 0) {
            y = nSeg;
        }

        x = y + a2 - q;
        if (x >= 1 && x <= m) {
            coordinate.batchId = w + b1 * k;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
            return;
        }
        return;
    }

    // else L3>0 and nSeg-m >=1
    if (L3 > 0 && nSeg - m >= 1) {
        y = (a1 - 1) * k + j;
        x = y + a2 - q;
        if (x < 1) {
            coordinate.batchId = b;
            coordinate.s1Idx = x + m;
            coordinate.s2Idx = y + m;
            return;
        }

        w = Ceil<int64_t>(x, m);
        x = x % m;
        if (x == 0) {
            x = m;
        }
        y = x + q - a2;
        if (w == b2 && y > m) {
            return;
        }
        if (y >= 1 && y <= n) {
            coordinate.batchId = w + b1 * k;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
            return;
        }
        return;
    }
    return;
}

template <bool isAlign = false>
__aicore__ inline int64_t GetPrefixByBidx(const __gm__ uint8_t* actualSeqQlenAddr,
                                          const __gm__ uint8_t* actualSeqKvlenAddr, const __gm__ int64_t* prefix,
                                          int64_t bIdx, int64_t step)
{
    int64_t w = bIdx / step;

    int64_t prefixNum = prefix[w];
    if (bIdx % step > 0) {
        w = w * step;
        int64_t lastSeqQLen = w == 0 ? 0 : ((__gm__ int64_t*)actualSeqQlenAddr)[w - 1];
        int64_t lastSeqKvLen = w == 0 ? 0 : ((__gm__ int64_t*)actualSeqKvlenAddr)[w - 1];
        int64_t currentSeqQLen, currentSeqKvLen;
        while (w < bIdx) {
            currentSeqQLen = ((__gm__ int64_t*)actualSeqQlenAddr)[w];
            currentSeqKvLen = ((__gm__ int64_t*)actualSeqKvlenAddr)[w];
            prefixNum += (currentSeqQLen - lastSeqQLen) *
                         (isAlign ? AlignTo16(currentSeqKvLen - lastSeqKvLen) : (currentSeqKvLen - lastSeqKvLen));
            w += 1;
            lastSeqQLen = currentSeqQLen;
            lastSeqKvLen = currentSeqKvLen;
        }
    }
    return prefixNum;
}

__aicore__ inline void GetSeqQlenKvlenByBidx(const __gm__ uint8_t* actualSeqQlenAddr,
                                             const __gm__ uint8_t* actualSeqKvlenAddr, int64_t bIdx,
                                             int64_t& actualSeqQlen, int64_t& actualSeqKvlen)
{
    if (unlikely(bIdx == 0)) {
        actualSeqQlen = ((__gm__ int64_t*)actualSeqQlenAddr)[0];
        actualSeqKvlen = ((__gm__ int64_t*)actualSeqKvlenAddr)[0];
    } else {
        actualSeqQlen = ((__gm__ int64_t*)actualSeqQlenAddr)[bIdx] - ((__gm__ int64_t*)actualSeqQlenAddr)[bIdx - 1];
        actualSeqKvlen = ((__gm__ int64_t*)actualSeqKvlenAddr)[bIdx] - ((__gm__ int64_t*)actualSeqKvlenAddr)[bIdx - 1];
    }
}

__aicore__ inline void CalVirtualIndex(int64_t flag, int64_t& m, int64_t& n)
{
    if (m < n) {
        n = m;
    }
    if (flag == 0) {
        m = NUM_TWO * m - n + 1;
    } else if (flag == 1) {
        m = m - (n + 1) / NUM_TWO + 1;
        n = n / NUM_TWO;
    } else {
        m = m - n / NUM_TWO;
        n = (n + 1) / NUM_TWO;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN, const bool isDense, const bool IS_N_EQUAL>
__aicore__ inline void UpdateMNPQ(int64_t actualCalcS1Token, int64_t actualCalcS2Token, CoordinateInfo& coordinateInfo,
                                  int64_t& actualM, int64_t& actualN)
{
    // sparse_mode == band 或者 RIGHT_DOWN_CASUAL时，token以右下角为基本，需要校正
    int64_t actualS1Len = coordinateInfo.actualS1Len;
    int64_t actualS2Len = coordinateInfo.actualS2Len;
    if (coordinateInfo.sparseMode == BAND || coordinateInfo.sparseMode == RIGHT_DOWN_CAUSAL) {
        actualCalcS1Token = actualCalcS1Token + actualS1Len - actualS2Len;
        actualCalcS2Token = actualCalcS2Token - actualS1Len + actualS2Len;
    }

    int64_t m = coordinateInfo.s1Outer;
    int64_t n = coordinateInfo.s2Outer;
    int64_t p = Ceil<int64_t>(actualCalcS1Token, CUBE_BASEM) + 1;
    int64_t q = Ceil<int64_t>(actualCalcS2Token, CUBE_BASEN) + 1;
    p = p > m ? m : p;
    q = q > n ? n : q;

    // 负数场景变换
    if (p < 0) {
        coordinateInfo.mOffset = 0;
        coordinateInfo.nOffset = -p;
        n = n + p;
        q = p + q;
        p = 1;
    } else if (q < 0) {
        coordinateInfo.mOffset = -q;
        coordinateInfo.nOffset = 0;
        m = m + q;
        p = p + q;
        q = 1;
    } else {
        coordinateInfo.mOffset = 0;
        coordinateInfo.nOffset = 0;
    }

    if (p + q <= m) {
        int64_t L1 = q - 1;
        int64_t L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        int64_t L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));

        if (L3 == 0) {
            actualM = p + q + L2 - NUM_TWO;
        } else {
            actualM = m;
        }
        actualN = L1 + L2 + L3;
    } else {
        actualM = m;
        actualN = Min(m - 1 + q, n);
    }

    coordinateInfo.m = actualM;
    coordinateInfo.n = actualN;
    coordinateInfo.p = p;
    coordinateInfo.q = q;

    if constexpr (!isDense) {
        return;
    }

    if (p + q <= actualM) {
        if (actualN > actualM || (IS_N_EQUAL && actualN == actualM)) {
            actualN = p + q - 1;
        } else {
            actualM = p + q - 1;
        }
    } else {
        if (p + q <= actualN) {
            actualN = p + q - 1;
        }
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDDenseSwizzleIndex(const __gm__ uint8_t* actualSeqQlenAddr,
                                               const __gm__ uint8_t* actualSeqKvlenAddr,
                                               const __gm__ uint64_t* tndS2BlockPrefixSum, int64_t b, int64_t n2,
                                               int64_t g, int64_t j, int64_t k, int64_t r,
                                               CoordinateInfo& coordinateInfo)
{
    j -= 1;
    r -= 1;
    coordinateInfo.batchId = -1;
    int64_t n1 = n2 * g;
    for (int64_t bIdx = 0; bIdx < b; bIdx++) {
        if (r < tndS2BlockPrefixSum[bIdx + 1]) {
            int64_t actualS1Len = 0;
            int64_t actualS2Len = 0;
            GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, bIdx, actualS1Len, actualS2Len);
            int64_t s1OuterTmp = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            int64_t s2OuterTmp = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            int64_t batchPrefix = static_cast<int64_t>(tndS2BlockPrefixSum[bIdx]);
            int64_t delta = r - batchPrefix;
            int64_t linearIdx = delta / s1OuterTmp * k + j;
            if (linearIdx >= s2OuterTmp * n1) {
                break;
            }
            // Keep this coordinate order identical to Python mode 2: head-major S2, then rotate S1.
            int64_t n1Idx = linearIdx / s2OuterTmp;
            int64_t s2Idx = linearIdx % s2OuterTmp;
            int64_t s1Idx = (s2Idx + delta) % s1OuterTmp;
            coordinateInfo.actualS1Len = actualS1Len;
            coordinateInfo.actualS2Len = actualS2Len;
            coordinateInfo.s1Outer = s1OuterTmp;
            coordinateInfo.s2Outer = s2OuterTmp;
            coordinateInfo.batchId = bIdx + 1;
            coordinateInfo.n2Idx = n1Idx / g + 1;
            coordinateInfo.gIdx = 1;
            coordinateInfo.s1Idx = s1Idx + 1;
            coordinateInfo.s2Idx = s2Idx + 1;
            break;
        }
    }
}

// Both schedules assign complete ownership units. Prefixes contain per-batch tile
// counts before multiplying by N2*G, at the existing compressed-prefix stride.
// Coordinates returned here are one-based, as in the existing TND decoders.
template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN, const bool CAUSAL_MIRROR>
__aicore__ inline void CalTNDGqaOwnedIndex(const __gm__ uint8_t* actualSeqQlenAddr,
                                           const __gm__ uint8_t* actualSeqKvlenAddr, const __gm__ int64_t* prefix,
                                           const __gm__ int64_t* unitStart, int64_t deterMaxRound, int64_t b,
                                           int64_t n2, int64_t g, int64_t j, int64_t r, int64_t step, int64_t pairGroup,
                                           CoordinateInfo& coordinateInfo)
{
    coordinateInfo.batchId = -1;
    if (r > deterMaxRound || unitStart[j - 1] + r > unitStart[j]) {
        return;
    }
    const int64_t task = unitStart[j - 1] + r - 1;
    const int64_t n1 = n2 * g;
    int64_t group = 0;
    if constexpr (CAUSAL_MIRROR) {
        while ((group + 1) * step < b && task >= prefix[group + 1] * n1) {
            ++group;
        }
    } else {
        // Find the last batch-group start not greater than this zero-based task.
        // The endpoint prefix, when present, is not a valid batch-group start.
        int64_t high = (b - 1) / step;
        while (group < high) {
            const int64_t middle = group + (high - group + 1) / NUM_TWO;
            if (task >= prefix[middle] * n1) {
                group = middle;
            } else {
                high = middle - 1;
            }
        }
    }
    int64_t batch = group * step;
    int64_t local = task - prefix[group] * n1;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    int64_t m = 0;
    int64_t n = 0;
    // Resolve at most one compressed-prefix interval, including its last batch.
    while (batch < b) {
        GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batch, actualS1Len, actualS2Len);
        m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
        n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
        int64_t batchTasks;
        if constexpr (CAUSAL_MIRROR) {
            batchTasks = m * (m + 1) / NUM_TWO * n1;
        } else {
            batchTasks = m * n * n1;
        }
        if (local < batchTasks) {
            break;
        }
        local -= batchTasks;
        ++batch;
    }
    if (batch >= b || m == 0 || n == 0) {
        return;
    }
    coordinateInfo.actualS1Len = actualS1Len;
    coordinateInfo.actualS2Len = actualS2Len;
    coordinateInfo.s1Outer = m;
    coordinateInfo.s2Outer = n;
    coordinateInfo.batchId = batch + 1;
    if constexpr (CAUSAL_MIRROR) {
        // N2=1, equal Q/KV lengths, even G, and pairGroup|(G/2) are Host guards.
        const int64_t virtualRows = m + 1;
        const int64_t columnTasks = virtualRows * (g / NUM_TWO);
        const int64_t columnRank = local / columnTasks;
        // Unit order is {1,m}, {2,m-1}, ...; an odd centre column occurs once.
        const int64_t y = columnRank % NUM_TWO == 0 ? columnRank / NUM_TWO + 1 : m - columnRank / NUM_TWO;
        // The common logical round, not local task offset, prevents dQ writers
        // on distinct virtual columns from colliding within the same round.
        const int64_t z = (r - 1 + y) % columnTasks;
        const int64_t a = z / pairGroup;
        const int64_t headPair = pairGroup * (a / virtualRows) + z % pairGroup;
        const int64_t x = a % virtualRows + 1;
        coordinateInfo.n2Idx = 1;
        if (x >= y + 1) {
            coordinateInfo.gIdx = NUM_TWO * headPair + 1;
            coordinateInfo.s1Idx = x - 1;
            coordinateInfo.s2Idx = y;
        } else {
            coordinateInfo.gIdx = NUM_TWO * headPair + NUM_TWO;
            coordinateInfo.s1Idx = m + 1 - x;
            coordinateInfo.s2Idx = m + 1 - y;
        }
    } else {
        const int64_t groupTasks = m * n * g;
        coordinateInfo.n2Idx = local / groupTasks + 1;
        local %= groupTasks;
        // Keep one S2 column in the existing L1 reuse path across all G heads.
        coordinateInfo.s2Idx = local / (g * m) + 1;
        coordinateInfo.gIdx = (local % (g * m)) / m + 1;
        coordinateInfo.s1Idx = local % m + 1;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN, const uint8_t DETER_SPARSE_TYPE = DETER_DENSE,
          const bool IS_N_EQUAL = true>
__aicore__ inline void CalTNDDenseIndex(const __gm__ uint8_t* actualSeqQlenAddr,
                                        const __gm__ uint8_t* actualSeqKvlenAddr, const __gm__ int64_t* prefix,
                                        int64_t deterMaxRound, int64_t b, int64_t n2, int64_t g, int64_t j, int64_t r,
                                        uint8_t flag, int64_t step, CoordinateInfo& coordinateInfo,
                                        const __gm__ int64_t* colStart = nullptr, int64_t headGroup = 1)
{
    coordinateInfo.batchId = -1;
    if (r > deterMaxRound) {
        return;
    }
    int64_t n1 = n2 * g;
    // 列独占时每个核起点各不相同，区间长度也不等，超出本核区间的轮次直接空转
    int64_t ID = (j - 1) * deterMaxRound + r;
    if (colStart != nullptr) {
        ID = colStart[j - 1] + r;
        if (ID > colStart[j]) {
            return;
        }
    }
    int64_t w = 0;
    int64_t batchId = 0;

    int64_t prefixMaxIndex = Ceil<int64_t>(b + 1, step);

    // GQA 需要将 b n2 g 转换为 b*n2 g
    if constexpr (!IS_N_EQUAL) {
        if (colStart != nullptr) {
            // Preserve the one-based strict comparison and the linear scan's
            // endpoint bound. In particular, the round-count suffix is not a
            // batch prefix, including when the last interval is compressed.
            int64_t high = Min(Ceil<int64_t>(b, step), prefixMaxIndex - 1);
            while (w < high) {
                const int64_t middle = w + (high - w + 1) / NUM_TWO;
                if (ID > prefix[middle] * n1) {
                    w = middle;
                } else {
                    high = middle - 1;
                }
            }
        } else {
            while (w * step < b && prefixMaxIndex > w + 1 && ID > prefix[w + 1] * n1) {
                w += 1;
            }
        }
        batchId = w;
    } else {
        while ((w + 1) * step < b && ID > prefix[w + 1] * n1) {
            w += 1;
        }
        batchId = w;
    }

    int64_t delta = ID - prefix[batchId] * n1;

    w = w * step;
    if (w >= b) {
        return;
    }
    int64_t m, n, p, q;
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, w, actualS1Len, actualS2Len);

    m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
    n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
    coordinateInfo.actualS1Len = actualS1Len;
    coordinateInfo.actualS2Len = actualS2Len;
    coordinateInfo.s1Outer = m;
    coordinateInfo.s2Outer = n;
    int64_t actualCalcS1Token = coordinateInfo.p;
    int64_t actualCalcS2Token = coordinateInfo.q;

    if constexpr (DETER_SPARSE_TYPE == DETER_BAND) {
        UpdateMNPQ<CUBE_BASEM, CUBE_BASEN, true, IS_N_EQUAL>(actualCalcS1Token, actualCalcS2Token, coordinateInfo, m,
                                                             n);
        p = coordinateInfo.p;
        q = coordinateInfo.q;
    }

    if constexpr (DETER_SPARSE_TYPE == DETER_CAUSAL) {
        CalVirtualIndex(flag, m, n);
    }

    if (unlikely(step > 1)) {
        int64_t batchBaseNum = m * n * n1;
        while (delta > batchBaseNum && w < b) {
            delta = delta - batchBaseNum;
            w += 1;
            if (w >= b) {
                return;
            }
            GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, w, actualS1Len, actualS2Len);
            m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            coordinateInfo.s1Outer = m;
            coordinateInfo.s2Outer = n;
            if constexpr (DETER_SPARSE_TYPE == DETER_CAUSAL) {
                CalVirtualIndex(flag, m, n);
            } else if constexpr (DETER_SPARSE_TYPE == DETER_BAND) {
                coordinateInfo.actualS1Len = actualS1Len;
                coordinateInfo.actualS2Len = actualS2Len;
                UpdateMNPQ<CUBE_BASEM, CUBE_BASEN, true, IS_N_EQUAL>(actualCalcS1Token, actualCalcS2Token,
                                                                     coordinateInfo, m, n);
                p = coordinateInfo.p;
                q = coordinateInfo.q;
            }

            batchBaseNum = m * n * n1;
        }
        coordinateInfo.actualS1Len = actualS1Len;
        coordinateInfo.actualS2Len = actualS2Len;
    }
    int64_t currentBaseNum = m * n;
    batchId = w + 1;

    int64_t x, y;
    if constexpr (!IS_N_EQUAL && DETER_SPARSE_TYPE == DETER_CAUSAL) {
        if (colStart != nullptr && headGroup == 128) {
            // F4 preserves C4's exact coordinates and four-round dQ ordering.
            // delta-1 = pair*(m*n*g) + column*(m*g) + offset, so one
            // column quotient resolves both pair and column. Keep task/prefix
            // arithmetic and all head-count operations at signed 64 bits.
            const int64_t column = (delta - 1) / (m * g);
            const int64_t t = r - 1;
            // Host proves r<=R<UINT32_MAX, j<=36 and actual Q<=INT32_MAX
            // with 128-row blocks: a<=2^30-1 and m<=16777217. Only these
            // two nonnegative phase divisions are narrowed; a+j<2^31.
            const uint32_t a = static_cast<uint32_t>(t >> 2);
            const uint32_t virtualRows = static_cast<uint32_t>(m);
            const int64_t h = NUM_FOUR * (static_cast<int64_t>(a / virtualRows) % (g >> 2)) + (t & 3);
            const int64_t u = (a + static_cast<uint32_t>(j)) % virtualRows + 1;
            coordinateInfo.batchId = batchId;
            coordinateInfo.n2Idx = column / n + 1;
            coordinateInfo.gIdx = h + 1;
            coordinateInfo.s1Idx = u;
            coordinateInfo.s2Idx = column % n + 1;
            return;
        }
    }
    if constexpr (IS_N_EQUAL) {
        int64_t deltaN = (delta - 1) / currentBaseNum + 1;
        delta = delta % currentBaseNum;
        delta = delta != 0 ? delta : currentBaseNum;
        int64_t gcd = Gcd(m, deterMaxRound);
        int64_t t1 = deterMaxRound / gcd;
        int64_t t2 = m / gcd;

        x = ((delta - 1) % m) + 1;
        y = (delta - 1) / m + 1;
        if (t1 < n) {
            int64_t nTail = n % t1;
            nTail = nTail == 0 ? t1 : nTail;
            if (y <= n - nTail) {
                int64_t delta_adj = Ceil<int64_t>(y, t1);
                delta += delta_adj;
                if (delta > delta_adj * t2 * deterMaxRound) {
                    delta -= t2 * deterMaxRound;
                }
                x = ((delta - 1) % m) + 1;
                y = (delta - 1) / m + 1;
            }
        }
        coordinateInfo.batchId = (batchId - 1) * n1 + deltaN;
    } else {
        currentBaseNum = currentBaseNum * g;
        int64_t deltaN = (delta - 1) / currentBaseNum + 1;
        delta = delta % currentBaseNum;
        delta = delta != 0 ? delta : currentBaseNum;

        int64_t m_new = m * g;
        x = delta % m_new;
        x = x != 0 ? x : m_new;
        y = Ceil<int64_t>(delta, m_new);

        if (colStart == nullptr) {
            int64_t gcd = Gcd(m_new, deterMaxRound);
            int64_t t1 = deterMaxRound / gcd;
            int64_t t2 = m_new / gcd;
            if (t1 < n) {
                int64_t nTail = (n % t1);
                nTail = nTail != 0 ? nTail : t1;
                if (y <= n - nTail) {
                    int64_t delta_adj = Ceil<int64_t>(y, t1);
                    delta += delta_adj;
                    if (delta > delta_adj * t2 * deterMaxRound) {
                        delta -= t2 * deterMaxRound;
                    }
                    x = delta % m_new;
                    x = x != 0 ? x : m_new;
                    y = Ceil<int64_t>(delta, m_new);
                }
            }
        } else if (headGroup > 1) {
            // Host admits one common H for the launch, with H|G and complete
            // columns/cuts/rounds divisible by H. Use the global logical round,
            // not the local column offset: within an H-round group every core
            // has a fixed virtual row and distinct query heads. Core phases
            // then keep dQ writers on different cores disjoint for the group.
            // The caller supplies r>=1 (r>R already returned above). The
            // mode1 helper restricts H to 2/4/8 here, and Host proves H|G,
            // G>=H. Thus both shifted operands are nonnegative and G/H>0.
            // 16/32 encode H=8 with a 2/4-row token window. Host proves that
            // every shared KV-pair has C*stride<=m; different core windows
            // therefore remain disjoint even across internal column boundaries.
            const int64_t rowStride = headGroup == 32 ? 4 : (headGroup == 16 ? 2 : 1);
            const int64_t hGroup = headGroup == 64 ? NUM_FOUR : (headGroup >= NUM_EIGHT ? NUM_EIGHT : headGroup);
            const int64_t headShift = hGroup == NUM_EIGHT ? 3 : (hGroup == NUM_FOUR ? 2 : 1);
            const int64_t t = r - 1;
            const int64_t a = t >> headShift;
            int64_t h;
            int64_t u;
            if (headGroup == 64) {
                // A complete column covers the M*(G/4) (row, head-group)
                // states. Consecutive core phases are distinct on this ring,
                // so each global four-round group has disjoint dQ writers.
                const int64_t groups = g >> headShift;
                const int64_t z = a + j;
                h = hGroup * (z % groups) + (t & (hGroup - 1));
                u = (z / groups) % m + 1;
            } else {
                h = hGroup * ((a / m) % (g >> headShift)) + (t & (hGroup - 1));
                u = (a + rowStride * j) % m + 1;
            }
            x = h * m + u;
        } else {
            // 列独占的切点落在列边界上，同一 batch 内各核的 x 相位会完全重合，原来那段
            // Gcd 偏移不再适用。按虚拟列号旋转 x 即可错开：列独占保证同一轮里落在同一
            // batch 的两个核必定处于不同的列，而 y 与 y' 都在 [1, n] 内且 n < m_new。
            x = ((x - 1 + y) % m_new) + 1;
        }

        int64_t N1_id = Ceil<int64_t>(x, m);
        x = x % m;
        x = x != 0 ? x : m;
        coordinateInfo.batchId = batchId;
        coordinateInfo.n2Idx = deltaN;
        coordinateInfo.gIdx = N1_id;
    }
    coordinateInfo.s1Idx = x;
    coordinateInfo.s2Idx = y;
    return;
}

__aicore__ inline void CalCausalPosWholeBatch(int64_t m, int64_t n, int64_t a, CoordinateInfo& coordinateInfo)
{
    int64_t n1 = n / NUM_TWO * NUM_TWO;
    int64_t L = NUM_TWO * m - n1 + 1;
    int64_t rm_local = (n1 * L) / NUM_TWO;
    int64_t x, y;
    if (a <= rm_local) {
        y = Ceil<int64_t>(a, L);
        int64_t r1 = a % L;
        r1 = r1 != 0 ? r1 : L;
        x = r1 + y - 1;
        if (x > m) {
            x = NUM_TWO * m + 1 - x;
            y = n1 + 1 - y;
        }
    } else {
        int64_t a1 = a - rm_local;
        y = n;
        x = a1 - 1 + y;
    }
    coordinateInfo.s1Idx = x;
    coordinateInfo.s2Idx = y;
    return;
}

__aicore__ inline void UpdataCausalMN(CoordinateInfo& coordinateInfo, int64_t& m, int64_t& n)
{
    coordinateInfo.s1Outer = m;
    coordinateInfo.s2Outer = n;
    n = m < n ? m : n;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDCausalIndex(const __gm__ uint8_t* actualSeqQlenAddr,
                                         const __gm__ uint8_t* actualSeqKvlenAddr, const __gm__ int64_t* prefix0,
                                         const __gm__ int64_t* prefix1, const __gm__ int64_t* prefix2, int64_t b,
                                         int64_t N1, int64_t k, int64_t j, int64_t r, int64_t step,
                                         CoordinateInfo& coordinateInfo)
{
    int64_t maxRoundIndex = b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
    int64_t N10 = N1 / k;
    int64_t N11 = N1 % k / NUM_TWO;
    int64_t R01 = prefix0[maxRoundIndex];
    int64_t R02 = prefix0[maxRoundIndex + 1];
    int64_t R0 = R01 + R02;
    int64_t R1 = prefix1[maxRoundIndex];
    int64_t R2 = prefix2[maxRoundIndex];
    coordinateInfo.batchId = -1;
    if (r <= R01) {
        int64_t a_judge = Ceil<int64_t>(r * NUM_TWO, N10);
        int64_t w = 0;
        while ((w + 1) * step < b && a_judge > prefix0[w + 1]) {
            w += 1;
        }
        int64_t batch_id = w * step;
        int64_t a = r - prefix0[w] * N10 / NUM_TWO;

        if (batch_id >= b) {
            return;
        }

        int64_t actualS1Len, actualS2Len, m, n;
        int64_t round_batch;

        GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batch_id, actualS1Len, actualS2Len);
        m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
        n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
        UpdataCausalMN(coordinateInfo, m, n);
        round_batch = (NUM_TWO * m - n + 1) * n / NUM_TWO;

        while (a > round_batch * N10) {
            batch_id += 1;
            if (batch_id >= b) {
                return;
            }
            GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batch_id, actualS1Len, actualS2Len);
            m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            UpdataCausalMN(coordinateInfo, m, n);
            a = a - round_batch * N10;
            round_batch = (NUM_TWO * m - n + 1) * n / NUM_TWO;
        }

        int64_t a0 = a % round_batch;
        a0 = a0 != 0 ? a0 : round_batch;
        CalCausalPosWholeBatch(m, n, a0, coordinateInfo);
        int64_t x = coordinateInfo.s1Idx;
        int64_t y = coordinateInfo.s2Idx;

        w = (a - 1) / round_batch * k + j;
        // b1组，每组k个batch行交错
        w = ((w - 1) / k) * k + ((y - 1 + (w - 1)) % k) + 1;
        batch_id = batch_id * N1 + w;
        coordinateInfo.batchId = batch_id;
        coordinateInfo.actualS1Len = actualS1Len;
        coordinateInfo.actualS2Len = actualS2Len;
        return;
    } else if (r > R01 && r <= R0) {
        int64_t a = r - R01;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix0, R02, b,
                                                               N11, 1, j, a, 0, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer}, x{coordinateInfo.s1Idx}, y{coordinateInfo.s2Idx};
        n = m < n ? m : n;
        int64_t batch_id1 = Ceil<int64_t>(coordinateInfo.batchId, N11);
        int64_t batch_id2 = coordinateInfo.batchId % N11;
        batch_id2 = batch_id2 != 0 ? batch_id2 : N11;
        int64_t x_new;
        if (x >= y + m - n + 1) {
            x_new = x - (m - n + 1);
            batch_id2 = NUM_TWO * batch_id2 - 1;
        } else {
            x_new = m + 1 - x;
            y = n + 1 - y;
            batch_id2 = NUM_TWO * batch_id2;
        }
        coordinateInfo.batchId = (batch_id1 - 1) * N1 + N10 * k + batch_id2;
        coordinateInfo.s1Idx = x_new;
        coordinateInfo.s2Idx = y;
        return;
    } else if (R0 < r && r <= R0 + R1) {
        int64_t a = r - R0;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix1, R1, b, 1,
                                                               1, j, a, 1, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer}, x{coordinateInfo.s1Idx}, y{coordinateInfo.s2Idx};
        n = m < n ? m : n;
        int64_t x_new;
        if (x >= y + m - n + 1) {
            x_new = x - (m - n + 1);
        } else {
            x_new = m + 1 - x;
            y = n + 1 - y;
        }
        coordinateInfo.batchId = coordinateInfo.batchId * N1;
        coordinateInfo.s1Idx = x_new;
        coordinateInfo.s2Idx = y;
        return;
    } else {
        int64_t a = r - R1 - R0;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix2, R2, b, 1,
                                                               1, j, a, NUM_TWO, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer};
        n = m < n ? m : n;
        coordinateInfo.s1Idx += n / NUM_TWO;

        coordinateInfo.batchId = coordinateInfo.batchId * N1;
        coordinateInfo.s1Idx = coordinateInfo.s1Idx;
        return;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDGQACausalIndex(const __gm__ uint8_t* actualSeqQlenAddr,
                                            const __gm__ uint8_t* actualSeqKvlenAddr, const __gm__ int64_t* prefix0,
                                            const __gm__ int64_t* prefix1, const __gm__ int64_t* prefix2, int64_t b,
                                            int64_t g, int64_t N2, int64_t k, int64_t j, int64_t r, int64_t step,
                                            bool coreDivide, CoordinateInfo& coordinateInfo,
                                            const __gm__ int64_t* colStart = nullptr, int64_t headGroup = 1)
{
    int64_t maxRoundIndex = b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
    int64_t N2_1 = N2 / NUM_TWO;
    int64_t R0 = prefix0[maxRoundIndex];
    int64_t R1 = prefix1[maxRoundIndex];
    int64_t R2 = prefix2[maxRoundIndex];

    coordinateInfo.batchId = -1;
    bool isFlag0{false}, isFlag1{false}, isFlag2{false};
    int64_t k2 = Ceil<int64_t>(k, N2);
    int64_t k1 = k - k2;
    int64_t curRoundId{0}, curCoreId{j};
    if ((!coreDivide || (coreDivide && (1 <= j && j <= k1))) && r <= R0) {
        isFlag0 = true;
        curRoundId = r;
    } else if (((!coreDivide && R0 < r && r <= R0 + R1) || (coreDivide && k1 < j && j <= k && r <= R1))) {
        isFlag1 = true;
        curRoundId = coreDivide ? r : r - R0;
        curCoreId = coreDivide ? j - k1 : j;
    } else if (!coreDivide || (coreDivide && k1 < j && j <= k)) {
        isFlag2 = true;
        curRoundId = coreDivide ? r - R1 : r - R0 - R1;
        curCoreId = coreDivide ? j - k1 : j;
    }

    if (isFlag0) {
        // 列独占只覆盖 N2 为偶数的场景，此时 flag1 / flag2 为空，切点数组只在 flag0 生效
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL, false>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix0,
                                                                      R0, b, N2_1, g, j, curRoundId, 0, step,
                                                                      coordinateInfo, colStart, headGroup);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer}, x{coordinateInfo.s1Idx}, y{coordinateInfo.s2Idx};
        n = m < n ? m : n;
        int64_t x_new;
        if (x >= y + m - n + 1) {
            x_new = x - (m - n + 1);
            coordinateInfo.n2Idx = NUM_TWO * coordinateInfo.n2Idx - 1;
        } else {
            x_new = m + 1 - x;
            y = n + 1 - y;
            coordinateInfo.n2Idx = NUM_TWO * coordinateInfo.n2Idx;
        }
        coordinateInfo.s1Idx = x_new;
        coordinateInfo.s2Idx = y;
        return;
    } else if (isFlag1) {
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL, false>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix1,
                                                                      R1, b, 1, g, curCoreId, curRoundId, 1, step,
                                                                      coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t x_new;
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer}, x{coordinateInfo.s1Idx}, y{coordinateInfo.s2Idx};
        n = m < n ? m : n;
        if (x >= y + m - n + 1) {
            x_new = x - (m - n + 1);
        } else {
            x_new = m + 1 - x;
            y = n + 1 - y;
        }
        coordinateInfo.n2Idx = N2;
        coordinateInfo.s1Idx = x_new;
        coordinateInfo.s2Idx = y;
        return;
    } else if (isFlag2) {
        int64_t a = r - R1 - R0;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_CAUSAL, false>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix2,
                                                                      R2, b, 1, g, curCoreId, curRoundId, NUM_TWO, step,
                                                                      coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t m{coordinateInfo.s1Outer}, n{coordinateInfo.s2Outer};
        n = m < n ? m : n;
        coordinateInfo.s1Idx += n / NUM_TWO;

        coordinateInfo.n2Idx = N2;
        coordinateInfo.s1Idx = coordinateInfo.s1Idx;
        return;
    } else {
        coordinateInfo.batchId = -1;
    }
}

__aicore__ inline int64_t BinarySearch(const __gm__ int64_t* prefix, int64_t b, int64_t rThreshold, int64_t step)
{
    int64_t w = 0;
    while ((w + 1) * step < b && rThreshold > prefix[w + 1]) {
        w += 1;
    }
    return w;
}

__aicore__ inline void CalPosWholeBatch(int64_t m, int64_t n, int64_t p, int64_t q, int64_t a,
                                        CoordinateInfo& coordinateInfo)
{
    int64_t L1, L2, L3, R1, R2, R3, Rm;
    int64_t x, y;
    if (p + q > m) {
        L1 = m - p;
        L2 = p + q - m;
        L3 = Min(m - 1, n - q);
        R1 = (p + m - 1) * L1 / NUM_TWO;
        R2 = m * L2;
        R3 = (NUM_TWO * m - 1 - L3) * L3 / NUM_TWO;
        Rm = R1 + R2 + R3;

        if (a <= R1) {
            int64_t L11 = L1 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * p + L11 - 1;
            int64_t local_round = L11 * L / NUM_TWO;
            if (a <= local_round) {
                y = Ceil<int64_t>(a, L);
                int64_t r1 = a % L;
                r1 = r1 != 0 ? r1 : L;
                x = p + y - r1;
                if (x < 1) {
                    y = L11 + 1 - y;
                    x = 1 - x;
                }
            } else {
                x = a - local_round;
                y = L1;
            }
        } else if (a <= R1 + R2 && m != 0) {
            int64_t a2 = a - R1;
            y = Ceil<int64_t>(a2, m);
            x = a2 % m;
            if (x == 0) {
                x = m;
            }
            y = y + L1;
        } else {
            int64_t a3 = a - R1 - R2;

            int64_t L31 = L3 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * m - L31 - 1;
            int64_t local_round = L31 * L / NUM_TWO;
            if (a3 <= local_round) {
                y = Ceil<int64_t>(a3, L);
                int64_t r1 = a3 % L;
                r1 = r1 != 0 ? r1 : L;
                x = y + r1;
                if (x > m) {
                    y = L31 + 1 - y;
                    x = x - m + y;
                }
            } else {
                x = a3 - local_round + L3;
                y = L3;
            }
            y = y + L1 + L2;
        }
        coordinateInfo.s1Idx = x;
        coordinateInfo.s2Idx = y;
        return;
    } else {
        L1 = q - 1;
        L2 = Min(n - q + 1, m + NUM_TWO - p - q);
        L3 = Max(0, Min(p + n - m - 1, p + q - NUM_TWO));
        R1 = (NUM_TWO * p - NUM_TWO + q) * L1 / NUM_TWO;
        R2 = (p + q - 1) * L2;
        R3 = (p + q - NUM_TWO) * L3 - (L3 * (L3 - 1)) / NUM_TWO;
        Rm = R1 + R2 + R3;
        if (a <= R1) {
            int64_t L11 = L1 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * p + L11 - 1;
            int64_t local_round = L11 * L / NUM_TWO;
            if (a <= local_round) {
                y = Ceil<int64_t>(a, L);
                int64_t r1 = a % L;
                r1 = r1 != 0 ? r1 : L;
                x = p + y - r1;
                if (x < 1) {
                    y = L11 + 1 - y;
                    x = 1 - x;
                } else {
                    x = p + y - x;
                }
            } else {
                x = a - local_round;
                y = L1;
            }
        } else if (a <= R1 + R2) {
            int64_t a2 = a - R1;
            y = Ceil<int64_t>(a2, (p + q - 1));
            x = a2 % (p + q - 1) + (y - 1);
            if (x == y - 1) {
                x = (p + q - 1) + y - 1;
            }
            y = y + L1;
        } else {
            int64_t a3 = a - R1 - R2;

            int64_t L31 = L3 / NUM_TWO * NUM_TWO;
            int64_t L = NUM_TWO * (p + q) - L31 - NUM_THREE;
            int64_t local_round = L31 * L / NUM_TWO;
            if (a3 <= local_round) {
                y = Ceil<int64_t>(a3, L);
                int64_t r1 = a3 % L;
                r1 = r1 != 0 ? r1 : L;
                x = y + r1 + 1 + m - (p + q);
                if (x > m) {
                    y = L31 + 1 - y;
                    x = x - (p + q - 1) + y;
                }
            } else {
                x = a3 - local_round + L3 + m - (p + q - 1);
                y = L3;
            }
            y = y + L1 + L2;
        }
        coordinateInfo.s1Idx = x;
        coordinateInfo.s2Idx = y;
        return;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDBandIndex(const __gm__ uint8_t* actualSeqQlenAddr,
                                       const __gm__ uint8_t* actualSeqKvlenAddr, const __gm__ int64_t* prefix0,
                                       const __gm__ int64_t* prefix1, int64_t b, int64_t N1, int64_t k, int64_t j,
                                       int64_t r, int64_t step, CoordinateInfo& coordinateInfo)
{
    int64_t maxRoundIndex = b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
    int64_t R0 = prefix0[maxRoundIndex];
    int64_t R1 = prefix1[maxRoundIndex];
    int64_t N11 = N1 / k;
    int64_t N12 = N1 % k;

    int64_t m, n, p, q, x, y;
    if (r <= R0) {
        int64_t batch_id = BinarySearch(prefix0, b, r, step);
        int64_t a = r - prefix0[batch_id];
        int64_t round_batch = 0;
        batch_id = batch_id * step - 1;
        int64_t actualCalcS1Token = coordinateInfo.p;
        int64_t actualCalcS2Token = coordinateInfo.q;
        do {
            batch_id++;
            a = a - round_batch * N11;
            GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batch_id, coordinateInfo.actualS1Len,
                                  coordinateInfo.actualS2Len);
            coordinateInfo.s1Outer = (coordinateInfo.actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
            coordinateInfo.s2Outer = (coordinateInfo.actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
            // 重新计算m、n、p、q
            UpdateMNPQ<CUBE_BASEM, CUBE_BASEN, false, true>(actualCalcS1Token, actualCalcS2Token, coordinateInfo, m, n);
            p = coordinateInfo.p;
            q = coordinateInfo.q;
            round_batch = m * n - (m - p) * (m - p + 1) / NUM_TWO - (n - q) * (n - q + 1) / NUM_TWO;
        } while (batch_id < b && a > round_batch * N11);

        int64_t a0 = a % round_batch;
        a0 = a0 != 0 ? a0 : round_batch;
        CalPosWholeBatch(m, n, p, q, a0, coordinateInfo);
        y = coordinateInfo.s2Idx;
        // 错位
        int64_t w = (a - 1) / round_batch * k + j;
        // b1组，每组k个batch行交错
        coordinateInfo.batchId = batch_id * N1 + w;
        return;
    } else if (R0 < r && r <= R0 + R1) {
        int64_t a = r - R0;
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_BAND>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix1, R1, b, N12,
                                                             1, j, a, 0, step, coordinateInfo);
        if (coordinateInfo.batchId < 0) {
            return;
        }
        m = coordinateInfo.m;
        n = coordinateInfo.n;
        p = coordinateInfo.p;
        q = coordinateInfo.q;
        x = coordinateInfo.s1Idx;
        y = coordinateInfo.s2Idx;
        int64_t batch_id = coordinateInfo.batchId;

        int64_t batch_id2 = batch_id % N12;
        batch_id2 = batch_id2 != 0 ? batch_id2 : N12;
        batch_id = (Ceil<int64_t>(batch_id, N12) - 1) * N1 + N11 * k + batch_id2;

        if (p + q <= m) {
            if (n >= m) {
                if (y - q + 1 <= x && x <= p + y - 1) {
                    coordinateInfo.batchId = batch_id;
                } else {
                    y = AbsCeil((x - (p + y - 1)), (p + q - 1)) * (p + q - 1) + y;
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                }
                return;
            } else {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;

                } else {
                    x = AbsCeil((y - (q + x - 1)), (p + q - 1)) * (p + q - 1) + x;
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                }
                return;
            }
        } else {
            if (p + q <= n) {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;
                } else if (y < x - p + 1 && y + p + q - 1 <= n) {
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s2Idx = y + p + q - 1;
                } else {
                    coordinateInfo.batchId = -1;
                }
                return;
            } else {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                    return;
                }
            }
        }
    }
    coordinateInfo.batchId = -1;
    return;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDGQABandIndex(const __gm__ uint8_t* actualSeqQlenAddr,
                                          const __gm__ uint8_t* actualSeqKvlenAddr, const __gm__ int64_t* prefix0,
                                          const __gm__ int64_t* prefix1, int64_t b, int64_t g, int64_t N2, int64_t k,
                                          int64_t j, int64_t r, int64_t step, CoordinateInfo& coordinateInfo)
{
    int64_t maxRoundIndex = b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
    int64_t R1 = prefix1[maxRoundIndex];

    int64_t m, n, p, q, x, y;
    if (r <= R1) {
        CalTNDDenseIndex<CUBE_BASEM, CUBE_BASEN, DETER_BAND, false>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix1, R1,
                                                                    b, N2, g, j, r, 0, step, coordinateInfo);

        if (coordinateInfo.batchId < 0) {
            return;
        }
        int64_t batch_id = coordinateInfo.batchId;
        m = coordinateInfo.m;
        n = coordinateInfo.n;
        p = coordinateInfo.p;
        q = coordinateInfo.q;
        x = coordinateInfo.s1Idx;
        y = coordinateInfo.s2Idx;

        if (p + q <= m) {
            if (n > m) {
                if (y - q + 1 <= x && x <= p + y - 1) {
                    coordinateInfo.batchId = batch_id;
                } else {
                    y = AbsCeil((x - (p + y - 1)), (p + q - 1)) * (p + q - 1) + y;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                    coordinateInfo.batchId = batch_id;
                }
                return;
            } else {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;

                } else {
                    x = AbsCeil((y - (q + x - 1)), (p + q - 1)) * (p + q - 1) + x;
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                    coordinateInfo.batchId = batch_id;
                }
                return;
            }
        } else {
            if (p + q <= n) {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.batchId = batch_id;
                } else if (y < x - p + 1 && y + p + q - 1 <= n) {
                    coordinateInfo.s2Idx = y + p + q - 1;
                    coordinateInfo.batchId = batch_id;
                } else {
                    coordinateInfo.batchId = -1;
                }
                return;
            } else {
                if (x - p + 1 <= y && y <= x + q - 1) {
                    coordinateInfo.s1Idx = x;
                    coordinateInfo.s2Idx = y;
                    coordinateInfo.batchId = batch_id;
                    return;
                }
            }
        }
    }
    coordinateInfo.batchId = -1;
    return;
}

__aicore__ inline void CalGQABandIndex(const BandInfo& bandInfo, int64_t j, int64_t r, int64_t N1,
                                       CoordinateInfo& coordinate)
{
    int64_t w, x, y = 0;
    coordinate.batchId = -1;

    int64_t k = bandInfo.k;
    int64_t m = bandInfo.m;
    int64_t n = bandInfo.n;
    int64_t p = bandInfo.p;
    int64_t q = bandInfo.q;
    int64_t b = bandInfo.b;
    int64_t L1 = bandInfo.L1;
    int64_t L2 = bandInfo.L2;
    int64_t L3 = bandInfo.L3;
    int64_t Rm = bandInfo.Rm;
    int64_t rm1 = bandInfo.rm;
    int64_t Rm_group = Rm * N1;
    int64_t a, b_id, N1_id;

    int64_t b1 = b / k;
    int64_t b2 = b % k;
    if (r <= rm1) {
        a = r % Rm_group;
        a = a != 0 ? a : Rm_group;
        N1_id = Ceil<int64_t>(a, Rm);
        a = a % Rm;
        a = a != 0 ? a : Rm;
        b_id = k * (Ceil<int64_t>(r, Rm_group) - 1) + j;
        // 计算最终坐标
        CalPosWholeBatch(m, n, p, q, a, coordinate);
        x = coordinate.s1Idx;
        y = coordinate.s2Idx;
        b_id = ((b_id - 1) / k) * k + ((y - 1 + (b_id - 1)) % k) + 1;
        w = (b_id - 1) * N1 + N1_id;
        coordinate.batchId = w;
        return;
    }

    // 第二部分
    a = r - rm1;
    if (b2 == 0) {
        return;
    }
    if (p + q > m) {
        int64_t R1 = bandInfo.R1;
        int64_t R2 = bandInfo.R2;
        // 对称性好的情况特殊处理
        if (NUM_TWO * b2 == k && L1 == L3 && m > 1 && L1 > 0) {
            b_id = (j + 1) / NUM_TWO;
            if (L2 % NUM_TWO == 1) {
                int64_t R0 = Max(NUM_TWO * p, m);
                R0 = m == 1 ? 1 : R0;
                // 此处1-R0的轮次优先处理
                if (a <= R0 * N1) {
                    int64_t N1_id = Ceil<int64_t>(a, R0);
                    a = a % R0;
                    a = a != 0 ? a : R0;
                    if (j % NUM_TWO == 1) {
                        if (n >= NUM_THREE) {
                            if (a <= p) {
                                coordinate.batchId = b1 * k * N1 + (b_id - 1) * N1 + N1_id;
                                coordinate.s1Idx = a;
                                coordinate.s2Idx = 1;
                                return;
                            } else if (p < a && a <= NUM_TWO * p) {
                                coordinate.batchId = b1 * k * N1 + (b_id - 1) * N1 + N1_id;
                                coordinate.s1Idx = L3 + a - p;
                                coordinate.s2Idx = n;
                                return;
                            }
                            return;
                        }
                    } else {
                        y = (n + 1) / NUM_TWO;
                        if (a <= p) {
                            coordinate.batchId = b1 * k * N1 + (b_id - 1) * N1 + N1_id;
                            coordinate.s1Idx = a + m - p;
                            coordinate.s2Idx = y;
                            return;
                        } else {
                            x = a - p;
                            if (x > m - p) {
                                coordinate.batchId = -1;
                                return;
                            } else {
                                coordinate.batchId = b1 * k * N1 + (b_id - 1) * N1 + N1_id;
                                coordinate.s1Idx = x;
                                coordinate.s2Idx = y;
                                return;
                            }
                        }
                    }
                }
                a -= (R0 - p) * N1;
                R2 -= m;
            }

            if (a <= R1 * N1) {
                if (L2 % NUM_TWO == 1) {
                    N1_id = Ceil<int64_t>((a - p * N1), (R1 - p));
                    a = (a - p * N1) % (R1 - p);
                    a = a != 0 ? a : (R1 - p);
                    a += p;
                } else {
                    N1_id = Ceil<int64_t>(a, R1);
                    a = a % R1;
                    a = a != 0 ? a : R1;
                }

                int64_t L11 = L1 / 2 * 2;
                int64_t L = 2 * p + L11 - 1;
                int64_t local_round = L11 * L / 2;
                if (a <= local_round) {
                    y = Ceil<int64_t>(a, L);
                    int64_t r1 = a % L;
                    r1 = r1 != 0 ? r1 : L;
                    x = p + y - r1;
                    if (x < 1) {
                        y = L11 + 1 - y;
                        x = 1 - x;
                    } else {
                        x = p + y - x;
                    }
                } else {
                    x = a - local_round;
                    y = L1;
                }
            } else if (a <= (R1 + (R2 / NUM_TWO)) * N1) {
                a -= R1 * N1;
                int64_t R_local = R2 / 2;
                N1_id = Ceil<int64_t>(a, R_local);
                a = a % R_local;
                a = a != 0 ? a : R_local;

                y = Ceil<int64_t>(a, m);
                x = a % m;
                if (x == 0) {
                    x = m;
                }
                y = y + L1;
            } else {
                return;
            }

            if (j % NUM_TWO == 0) {
                y = n + 1 - y;
                if (y <= q) {
                    x = x + 1;
                    if (x > m) {
                        x = x - m;
                    }
                } else {
                    x = L3 + y + x - n;
                }
            }
            coordinate.batchId = b1 * k * N1 + (b_id - 1) * N1 + N1_id;
            coordinate.s1Idx = x;
            coordinate.s2Idx = y;
            return;
        }

        // 拼接
        if (p + q <= n) {
            CalGQADenseIndex(k, m, p + q - 1, b2, j, a, N1, coordinate);
            if (coordinate.batchId == -1) {
                return;
            }
            w = coordinate.batchId;
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;
            if (x - p + 1 <= y && y <= x + q - 1) {
                coordinate.batchId = b1 * k * N1 + w;
                coordinate.s1Idx = x;
                coordinate.s2Idx = y;
                return;
            } else if (y < x - p + 1 && 1 <= y + p + q - 1 && y + p + q - 1 <= n) {
                coordinate.batchId = b1 * k * N1 + w;
                coordinate.s1Idx = x;
                coordinate.s2Idx = y + p + q - 1;
                return;
            } else {
                coordinate.batchId = -1;
                return;
            }
        } else {
            CalGQADenseIndex(k, m, n, b2, j, a, N1, coordinate);
            if (coordinate.batchId == -1) {
                return;
            }
            w = coordinate.batchId;
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;

            if (x - p + 1 <= y && y <= x + q - 1) {
                coordinate.batchId = b1 * k * N1 + w;
                coordinate.s1Idx = x;
                coordinate.s2Idx = y;
                return;
            } else {
                coordinate.batchId = -1;
                return;
            }
        }
    } else {
        if (n > m) {
            int64_t r_case1 = bandInfo.R1;
            int64_t r_case2 = bandInfo.R2;
            if (r_case2 <= r_case1) {
                CalGQADenseIndex(k, m, p + q - 1, b2, j, a, N1, coordinate);
                if (coordinate.batchId == -1) {
                    return;
                }
                w = coordinate.batchId;
                x = coordinate.s1Idx;
                y = coordinate.s2Idx;

                if (y - q + 1 <= x && x <= p + y - 1) {
                    coordinate.batchId = b1 * k * N1 + w;
                    coordinate.s1Idx = x;
                    coordinate.s2Idx = y;
                    return;
                } else {
                    y = Ceil<int64_t>((x - (p + y - 1)), (p + q - 1)) * (p + q - 1) + y;
                    if (1 <= y && y <= n) {
                        coordinate.batchId = b1 * k * N1 + w;
                        coordinate.s1Idx = x;
                        coordinate.s2Idx = y;
                        return;
                    } else {
                        coordinate.batchId = -1;
                        return;
                    }
                }
            } else {
                CalGQADenseIndexNoTune(k, p + q - 1, n, b2, j, a, N1, coordinate);
                if (coordinate.batchId == -1) {
                    return;
                }
                w = coordinate.batchId;
                x = coordinate.s1Idx;
                y = coordinate.s2Idx;

                x = x + y - q;
                if (1 <= x && x <= m) {
                    coordinate.batchId = b1 * k * N1 + w;
                    coordinate.s1Idx = x;
                    coordinate.s2Idx = y;
                    return;
                } else {
                    coordinate.batchId = -1;
                    return;
                }
            }
        } else {
            CalGQADenseIndex(k, p + q - 1, n, b2, j, a, N1, coordinate);
            if (coordinate.batchId == -1) {
                return;
            }
            w = coordinate.batchId;
            x = coordinate.s1Idx;
            y = coordinate.s2Idx;

            if (x - p + 1 <= y && y <= x + q - 1) {
                coordinate.batchId = b1 * k * N1 + w;
                coordinate.s1Idx = x;
                coordinate.s2Idx = y;
                return;
            } else {
                x = Ceil<int64_t>((y - (q + x - 1)), (p + q - 1)) * (p + q - 1) + x;
                if (1 <= x && x <= m) {
                    coordinate.batchId = b1 * k * N1 + w;
                    coordinate.s1Idx = x;
                    coordinate.s2Idx = y;
                    return;
                } else {
                    coordinate.batchId = -1;
                    return;
                }
            }
        }
    }
    coordinate.batchId = -1;
    return;
}
__aicore__ inline bool CalTNDDenseLine(int64_t k, int64_t m, int64_t n, int64_t copies, int64_t j, int64_t r,
                                       int64_t& w, int64_t& x, int64_t& y)
{
    if (j < 1 || j > k || r < 1 || copies <= 0 || m <= 0 || n <= 0) {
        return false;
    }
    const int64_t delta = r - 1;
    const int64_t y1 = (delta / m) * k + j;
    if (y1 > n * copies) {
        return false;
    }
    w = (y1 - 1) / n + 1;
    y = (y1 - 1) % n + 1;
    x = (y + delta - 1) % m + 1;
    return true;
}

__aicore__ inline void CalTNDG2k(int64_t k, int64_t j, int64_t a, int64_t l1, int64_t offset, int64_t& x, int64_t& y)
{
    if (j % NUM_TWO == 1) {
        if (a <= l1 - j + 1) {
            y = j + offset;
            x = y + a - 1;
        } else {
            y = NUM_TWO * k + 1 - j + offset;
            x = y + NUM_TWO * l1 - NUM_TWO * k + 1 - a;
        }
    } else {
        if (a >= l1 - NUM_TWO * k + 1 + j) {
            y = j + offset;
            x = y + NUM_TWO * l1 - NUM_TWO * k + 1 - a;
        } else {
            y = NUM_TWO * k + 1 - j + offset;
            x = y + a - 1;
        }
    }
}

__aicore__ inline bool CalTNDCase0(int64_t m, int64_t n, int64_t j, int64_t r, int64_t& x, int64_t& y)
{
    if (j > (n / NUM_TWO) + 1 || j < 1 || r < 1) {
        return false;
    }
    if (j % NUM_TWO == 1) {
        if (r + j <= n + 1) {
            x = r + j - 1;
            y = j;
        } else {
            x = NUM_TWO * n + NUM_TWO - j - r;
            y = n + NUM_THREE - j - (n % NUM_TWO);
        }
    } else {
        if (j <= r + 1 - (n % NUM_TWO)) {
            x = n + j - r - 1 + n % NUM_TWO;
            y = j;
        } else {
            x = n + NUM_TWO + r - j - (n % NUM_TWO);
            y = n + NUM_THREE - j - (n % NUM_TWO);
        }
    }
    return y >= 1 && y <= m && y <= x && x <= m;
}

__aicore__ inline bool CalTNDCase0Rec(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r, int64_t& x, int64_t& y)
{
    if (NUM_TWO * k < m + 1 && k < n) {
        CalTNDG2k(k, j, r, m, 0, x, y);
    } else {
        if (!CalTNDCase0(m, m, j, r, x, y)) {
            return false;
        }
    }
    return y >= 1 && y <= n && y <= x && x <= m;
}

__aicore__ inline int64_t CalTNDCase0RecRounds(int64_t k, int64_t m, int64_t n)
{
    if (n <= 0) {
        return 0;
    }
    if (NUM_TWO * k < m + 1 && k < n) {
        return Max(static_cast<int64_t>(0), NUM_TWO * m - NUM_TWO * k + 1);
    }
    return m;
}

__aicore__ inline int64_t CalTNDSingleCausalRounds(int64_t k, int64_t m, int64_t n)
{
    const int64_t t = n / (NUM_TWO * k);
    const int64_t ell = n % (NUM_TWO * k);
    const int64_t rounds2k = (NUM_TWO * m + 1) * t - NUM_TWO * k * t * t;
    if (ell == 0) {
        return rounds2k;
    }
    if (t == 0) {
        return CalTNDCase0RecRounds(k, m, n);
    }
    const int64_t l = m - NUM_TWO * k * t;
    const int64_t phaseCount = Ceil<int64_t>(ell, k);
    int64_t roundsTail = 0;
    for (int64_t s = 0; s < phaseCount; ++s) {
        roundsTail += Max(static_cast<int64_t>(0), l - s * k);
    }
    return rounds2k + roundsTail;
}

__aicore__ inline bool CalTNDSingleCausalPos(int64_t k, int64_t m, int64_t n, int64_t j, int64_t r, int64_t& x,
                                             int64_t& y)
{
    if (j < 1 || j > k || r < 1) {
        return false;
    }
    const int64_t t = n / (NUM_TWO * k);
    const int64_t ell = n % (NUM_TWO * k);
    const int64_t bound2k = (NUM_TWO * m + 1) * t - NUM_TWO * k * t * t;
    if (r <= bound2k) {
        int64_t a = r;
        int64_t offset = 0;
        for (int64_t i = 0; i < t; ++i) {
            const int64_t l = m - offset;
            const int64_t blockRounds = NUM_TWO * l - NUM_TWO * k + 1;
            if (a <= blockRounds) {
                CalTNDG2k(k, j, a, l, offset, x, y);
                return y >= 1 && y <= n && y <= x && x <= m;
            }
            a -= blockRounds;
            offset += NUM_TWO * k;
        }
        return false;
    }
    int64_t a = r - bound2k;
    if (ell == 0) {
        return false;
    }
    const int64_t offset = NUM_TWO * k * t;
    const int64_t l = m - offset;
    if (t == 0) {
        return CalTNDCase0Rec(k, m, n, j, a, x, y);
    }
    const int64_t phaseCount = Ceil<int64_t>(ell, k);
    for (int64_t s = 0; s < phaseCount; ++s) {
        const int64_t phaseRounds = Max(static_cast<int64_t>(0), l - s * k);
        if (a <= phaseRounds) {
            const int64_t yLocal = s * k + j;
            const int64_t xLocal = yLocal + a - 1;
            if (yLocal >= 1 && yLocal <= ell && yLocal <= xLocal && xLocal <= l) {
                x = offset + xLocal;
                y = offset + yLocal;
                return true;
            }
            return false;
        }
        a -= phaseRounds;
    }
    return false;
}

__aicore__ inline bool IsTNDRightDownValid(int64_t m, int64_t n, int64_t x, int64_t y)
{
    return x >= 1 && x <= m && y >= 1 && y <= n && y <= x + n - m + 1;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void LoadTNDOuterMN(const __gm__ uint8_t* actualSeqQlenAddr, const __gm__ uint8_t* actualSeqKvlenAddr,
                                      int64_t batchType, int64_t& m, int64_t& n, CoordinateInfo& coordinateInfo)
{
    int64_t actualS1Len = 0;
    int64_t actualS2Len = 0;
    GetSeqQlenKvlenByBidx(actualSeqQlenAddr, actualSeqKvlenAddr, batchType, actualS1Len, actualS2Len);
    if (actualS1Len <= 0 || actualS2Len <= 0) {
        m = 0;
        n = 0;
    } else {
        m = (actualS1Len + CUBE_BASEM - 1) / CUBE_BASEM;
        n = (actualS2Len + CUBE_BASEN - 1) / CUBE_BASEN;
        if (m < 1) {
            m = 1;
        }
        if (n < 1) {
            n = 1;
        }
    }
    coordinateInfo.actualS1Len = actualS1Len;
    coordinateInfo.actualS2Len = actualS2Len;
    coordinateInfo.s1Outer = m;
    coordinateInfo.s2Outer = n;
}

__aicore__ inline int64_t GetTNDRightDownRowShift(int64_t m, int64_t n)
{
    return Max(static_cast<int64_t>(0), m - n - 1);
}

__aicore__ inline void TrimTNDCausalMN(int64_t& m, int64_t& n, bool leftUp)
{
    if (leftUp) {
        if (m < n) {
            n = m;
        }
    } else {
        m -= GetTNDRightDownRowShift(m, n);
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void ExpandTNDCausalRun(const __gm__ uint8_t* actualSeqQlenAddr,
                                          const __gm__ uint8_t* actualSeqKvlenAddr, int64_t b, int64_t bIdx,
                                          bool leftUp, int64_t& runHead, int64_t& runEnd, int64_t& m, int64_t& n,
                                          CoordinateInfo& coordinateInfo)
{
    LoadTNDOuterMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, bIdx, m, n, coordinateInfo);
    TrimTNDCausalMN(m, n, leftUp);
    runHead = bIdx;
    while (runHead > 0) {
        int64_t prevM = 0;
        int64_t prevN = 0;
        CoordinateInfo prevInfo;
        LoadTNDOuterMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead - 1, prevM, prevN,
                                               prevInfo);
        TrimTNDCausalMN(prevM, prevN, leftUp);
        if (prevM != m || prevN != n) {
            break;
        }
        runHead -= 1;
    }
    runEnd = bIdx;
    while (runEnd + 1 < b) {
        int64_t nextM = 0;
        int64_t nextN = 0;
        CoordinateInfo nextInfo;
        LoadTNDOuterMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runEnd + 1, nextM, nextN,
                                               nextInfo);
        TrimTNDCausalMN(nextM, nextN, leftUp);
        if (nextM != m || nextN != n) {
            break;
        }
        runEnd += 1;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void LoadTNDBandMN(const __gm__ uint8_t* actualSeqQlenAddr, const __gm__ uint8_t* actualSeqKvlenAddr,
                                     int64_t bIdx, int64_t& m, int64_t& n, int64_t& p, int64_t& q,
                                     CoordinateInfo& coordinateInfo)
{
    LoadTNDOuterMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, bIdx, m, n, coordinateInfo);
    if (m < 1 || n < 1) {
        p = 0;
        q = 0;
        return;
    }
    int64_t actualP = coordinateInfo.p;
    int64_t actualQ = coordinateInfo.q;
    int64_t dummyM = 0;
    int64_t dummyN = 0;
    UpdateMNPQ<CUBE_BASEM, CUBE_BASEN, false, true>(actualP, actualQ, coordinateInfo, dummyM, dummyN);
    p = coordinateInfo.p;
    q = coordinateInfo.q;
    m = coordinateInfo.m > 0 ? coordinateInfo.m : m;
    n = coordinateInfo.n > 0 ? coordinateInfo.n : n;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void ExpandTNDBandRun(const __gm__ uint8_t* actualSeqQlenAddr,
                                        const __gm__ uint8_t* actualSeqKvlenAddr, int64_t b, int64_t bIdx,
                                        int64_t& runHead, int64_t& runEnd, int64_t& m, int64_t& n, int64_t& p,
                                        int64_t& q, CoordinateInfo& coordinateInfo)
{
    const int64_t rawP = coordinateInfo.p;
    const int64_t rawQ = coordinateInfo.q;
    const int64_t rawSparseMode = coordinateInfo.sparseMode;
    LoadTNDBandMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, bIdx, m, n, p, q, coordinateInfo);
    runHead = bIdx;
    while (runHead > 0) {
        int64_t prevM = 0;
        int64_t prevN = 0;
        int64_t prevP = 0;
        int64_t prevQ = 0;
        CoordinateInfo prevInfo;
        prevInfo.sparseMode = rawSparseMode;
        prevInfo.p = rawP;
        prevInfo.q = rawQ;
        LoadTNDBandMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead - 1, prevM, prevN, prevP,
                                              prevQ, prevInfo);
        if (prevM != m || prevN != n || prevP != p || prevQ != q) {
            break;
        }
        runHead -= 1;
    }
    runEnd = bIdx;
    while (runEnd + 1 < b) {
        int64_t nextM = 0;
        int64_t nextN = 0;
        int64_t nextP = 0;
        int64_t nextQ = 0;
        CoordinateInfo nextInfo;
        nextInfo.sparseMode = rawSparseMode;
        nextInfo.p = rawP;
        nextInfo.q = rawQ;
        LoadTNDBandMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runEnd + 1, nextM, nextN, nextP,
                                              nextQ, nextInfo);
        if (nextM != m || nextN != n || nextP != p || nextQ != q) {
            break;
        }
        runEnd += 1;
    }
}

__aicore__ inline int64_t GetTNDPrefixTailIndex(int64_t b, int64_t step)
{
    return b > DETER_PREFIX_THRESHOLD ? Ceil<int64_t>(b + 1, step) : b + 1;
}

__aicore__ inline void GetTNDLeftUpVirt(int64_t m, int64_t n, int64_t& virtM, int64_t& virtN)
{
    if (m < n) {
        n = m;
    }
    virtM = NUM_TWO * m - n + 1;
    virtN = n;
}

__aicore__ inline void GetTNDRightDownVirt(int64_t m, int64_t n, int64_t& virtM, int64_t& virtN)
{
    m -= GetTNDRightDownRowShift(m, n);
    if (m < 1 || n < 1) {
        virtM = 0;
        virtN = 0;
        return;
    }
    virtM = m;
    virtN = NUM_TWO * n - m + NUM_THREE;
}

constexpr int64_t TND_LINE_TINY_AREA = 8;

__aicore__ inline bool TNDRunIsLineWorthy(int64_t k, int64_t virtM, int64_t virtN, int64_t pairCount)
{
    if (virtM <= 0 || virtN <= 0 || virtM * virtN <= TND_LINE_TINY_AREA) {
        return false;
    }
    return pairCount > 0 && virtN * pairCount >= k && virtM >= Min(k, virtN);
}

__aicore__ inline int64_t CalTNDBandWideCols(int64_t m, int64_t n, int64_t p, int64_t q)
{
    const int64_t nNew = Min(m - 1 + q, n);
    const int64_t l1 = m - p;
    const int64_t l2 = p + q - m;
    const int64_t l3 = nNew - l1 - l2;
    const int64_t foldCols = Max(static_cast<int64_t>(0), l3 - p + 1);
    return nNew - foldCols;
}

__aicore__ inline int64_t CalTNDBandNarrowRounds(int64_t k, int64_t n1, int64_t m, int64_t n, int64_t p, int64_t q)
{
    const int64_t l1 = q - 1;
    const int64_t l2 = Min(n - q + 1, m + NUM_TWO - p - q);
    const int64_t l3 = Max(static_cast<int64_t>(0), Min(p + n - m - 1, p + q - NUM_TWO));
    int64_t overlap = 0;
    int64_t nNew = l1 + l2 + l3;
    if (l3 != 0 && p <= l3) {
        overlap = Min(l3 - p + 1, l1);
    }
    const int64_t seg = p + q - 1;
    return seg * Ceil<int64_t>((nNew - overlap) * n1 + overlap, k);
}

__aicore__ inline int64_t CalTNDBandLineRounds(int64_t k, int64_t n1, int64_t m, int64_t n, int64_t p, int64_t q)
{
    if (p + q < NUM_TWO) {
        return 0;
    }
    if (p >= m) {
        return Ceil<int64_t>(n * n1, k) * m;
    }
    if (p + q <= m) {
        return CalTNDBandNarrowRounds(k, n1, m, n, p, q);
    }
    return Ceil<int64_t>(CalTNDBandWideCols(m, n, p, q) * n1, k) * m;
}

__aicore__ inline void MapTNDTriToBatch(int64_t tri, int64_t n1, int64_t runHead, int64_t& batchId)
{
    batchId = (runHead + tri / n1) * n1 + (tri % n1) + 1;
}

__aicore__ inline void MapTNDCopyToBatch(int64_t copyId, int64_t n1, int64_t runHead, int64_t& batchId)
{
    MapTNDTriToBatch(copyId - 1, n1, runHead, batchId);
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN, const bool LEFT_UP>
__aicore__ inline void BindTNDCausalMappedBatch(const __gm__ uint8_t* actualSeqQlenAddr,
                                                const __gm__ uint8_t* actualSeqKvlenAddr, int64_t runHead, int64_t n1,
                                                int64_t n1Id, int64_t x, int64_t y, CoordinateInfo& coordinateInfo)
{
    MapTNDTriToBatch(n1Id - 1, n1, runHead, coordinateInfo.batchId);
    int64_t realM = 0;
    int64_t realN = 0;
    LoadTNDOuterMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead + (n1Id - 1) / n1, realM,
                                           realN, coordinateInfo);
    coordinateInfo.s1Idx = LEFT_UP ? x : (x + GetTNDRightDownRowShift(realM, realN));
    coordinateInfo.s2Idx = y;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void BindTNDBandMappedBatch(const __gm__ uint8_t* actualSeqQlenAddr,
                                              const __gm__ uint8_t* actualSeqKvlenAddr, int64_t runHead, int64_t n1,
                                              int64_t copyId, int64_t x, int64_t y, int64_t rawP, int64_t rawQ,
                                              CoordinateInfo& coordinateInfo)
{
    MapTNDCopyToBatch(copyId, n1, runHead, coordinateInfo.batchId);
    coordinateInfo.p = rawP;
    coordinateInfo.q = rawQ;
    int64_t dummyM = 0;
    int64_t dummyN = 0;
    int64_t dummyP = 0;
    int64_t dummyQ = 0;
    LoadTNDBandMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead + (copyId - 1) / n1, dummyM,
                                          dummyN, dummyP, dummyQ, coordinateInfo);
    coordinateInfo.s1Idx = x;
    coordinateInfo.s2Idx = y;
}

__aicore__ inline bool MapTNDLeftUp(int64_t m, int64_t n, int64_t pairId, int64_t pairBatchCount, int64_t& x,
                                    int64_t& y, int64_t& n1Id)
{
    if (x > m) {
        y = n + 1 - y;
        x = NUM_TWO * m + 1 - x;
        n1Id = NUM_TWO * pairId;
    } else if (x < y) {
        y = n + 1 - y;
        x = n - x;
        n1Id = NUM_TWO * pairId;
    } else {
        n1Id = NUM_TWO * pairId - 1;
    }
    return n1Id >= 1 && n1Id <= pairBatchCount && y >= 1 && y <= n && y <= x && x <= m;
}

__aicore__ inline bool MapTNDRightDown(int64_t m, int64_t n, int64_t pairId, int64_t pairBatchCount, int64_t& x,
                                       int64_t& y, int64_t& n1Id)
{
    const int64_t nExt = n + 1;
    if (y >= x + nExt - m + 1) {
        y = NUM_TWO * nExt - m - y + NUM_TWO;
        x = m + 1 - x;
        n1Id = NUM_TWO * pairId;
    } else {
        n1Id = NUM_TWO * pairId - 1;
    }
    return n1Id >= 1 && n1Id <= pairBatchCount && IsTNDRightDownValid(m, n, x, y);
}

__aicore__ inline bool MapTNDBandWide(int64_t m, int64_t n, int64_t p, int64_t q, int64_t& x, int64_t& y)
{
    const int64_t l1 = m - p;
    const int64_t l2 = p + q - m;
    const int64_t l3 = n - l1 - l2;
    const int64_t foldCols = Max(static_cast<int64_t>(0), l3 - p + 1);
    if (foldCols > 0 && y <= foldCols && x >= p + y) {
        y = n - foldCols + y;
    } else if (y <= l1 && x >= p + y) {
        return false;
    }
    if (y > l1 + l2 && x <= y - l1 - l2) {
        return false;
    }
    return x >= 1 && x <= m && y >= 1 && y <= n;
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN, const bool LEFT_UP>
__aicore__ inline void CalTNDCausalLineRunIndex(const __gm__ uint8_t* actualSeqQlenAddr,
                                                const __gm__ uint8_t* actualSeqKvlenAddr, const __gm__ int64_t* prefix,
                                                int64_t b, int64_t n1, int64_t k, int64_t j, int64_t r, int64_t step,
                                                CoordinateInfo& coordinateInfo)
{
    coordinateInfo.batchId = -1;
    const int64_t batchTypeBase = BinarySearch(prefix, b, r, step);
    int64_t a = r - prefix[batchTypeBase];
    int64_t bIdx = batchTypeBase * step;
    if (bIdx >= b || a < 1) {
        return;
    }
    while (bIdx < b) {
        int64_t runHead = bIdx;
        int64_t runEnd = bIdx;
        int64_t m = 0;
        int64_t n = 0;
        ExpandTNDCausalRun<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, b, bIdx, LEFT_UP, runHead,
                                                   runEnd, m, n, coordinateInfo);
        const int64_t groupSize = runEnd - runHead + 1;
        if (bIdx > runHead) {
            bIdx = runEnd + 1;
            continue;
        }
        if (m <= 0 || n <= 0 || (m == 1 && n == 1)) {
            bIdx = runEnd + 1;
            continue;
        }
        int64_t virtM = 0;
        int64_t virtN = 0;
        if (LEFT_UP) {
            GetTNDLeftUpVirt(m, n, virtM, virtN);
        } else {
            GetTNDRightDownVirt(m, n, virtM, virtN);
        }
        const int64_t total = n1 * groupSize;
        const int64_t runPairs = total / NUM_TWO;
        const int64_t pairSpace = runPairs * NUM_TWO;
        const int64_t pairRounds = (runPairs > 0 && virtN > 0) ? (Ceil<int64_t>(virtN * runPairs, k) * virtM) : 0;
        const int64_t singleRounds =
            (total % NUM_TWO == 1) ? (LEFT_UP ? CalTNDSingleCausalRounds(k, m, n) : (Ceil<int64_t>(n, k) * m)) : 0;
        const int64_t runRounds = pairRounds + singleRounds;
        if (a <= runRounds) {
            if (a <= pairRounds && runPairs > 0) {
                int64_t pairId = 0;
                int64_t x = 0;
                int64_t y = 0;
                const bool ok = CalTNDDenseLine(k, virtM, virtN, runPairs, j, a, pairId, x, y);
                int64_t n1Id = 0;
                const bool mapped = ok && (LEFT_UP ? MapTNDLeftUp(m, n, pairId, pairSpace, x, y, n1Id) :
                                                     MapTNDRightDown(m, n, pairId, pairSpace, x, y, n1Id));
                if (mapped) {
                    BindTNDCausalMappedBatch<CUBE_BASEM, CUBE_BASEN, LEFT_UP>(actualSeqQlenAddr, actualSeqKvlenAddr,
                                                                              runHead, n1, n1Id, x, y, coordinateInfo);
                }
                return;
            }
            const int64_t singleRound = a - pairRounds;
            int64_t x = 0;
            int64_t y = 0;
            bool ok = false;
            if (LEFT_UP) {
                ok = CalTNDSingleCausalPos(k, m, n, j, singleRound, x, y);
            } else {
                int64_t w = 0;
                ok = CalTNDDenseLine(k, m, n, 1, j, singleRound, w, x, y) && IsTNDRightDownValid(m, n, x, y);
            }
            if (ok) {
                BindTNDCausalMappedBatch<CUBE_BASEM, CUBE_BASEN, LEFT_UP>(actualSeqQlenAddr, actualSeqKvlenAddr,
                                                                          runHead, n1, total, x, y, coordinateInfo);
            }
            return;
        }
        a -= runRounds;
        bIdx = runEnd + 1;
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDCausalLineIndex(const __gm__ uint8_t* actualSeqQlenAddr,
                                             const __gm__ uint8_t* actualSeqKvlenAddr, const __gm__ int64_t* prefix0,
                                             const __gm__ int64_t* prefixSingle, int64_t b, int64_t N1, int64_t k,
                                             int64_t j, int64_t r, int64_t step, int64_t sparseMode,
                                             CoordinateInfo& coordinateInfo)
{
    if (r <= prefix0[0]) {
        coordinateInfo.batchId = -1;
        const int64_t task = (r - 1) * k + j;
        const int64_t bucket = BinarySearch(prefixSingle, b, task, step);
        int64_t remaining = task - prefixSingle[bucket];
        for (int64_t batch = bucket * step; batch < b; ++batch) {
            int64_t m = 0;
            int64_t n = 0;
            LoadTNDOuterMN<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, batch, m, n, coordinateInfo);
            TrimTNDCausalMN(m, n, sparseMode != RIGHT_DOWN_CAUSAL);
            if (m != 1 || n != 1) {
                continue;
            }
            if (remaining <= N1) {
                BindTNDCausalMappedBatch<CUBE_BASEM, CUBE_BASEN, true>(actualSeqQlenAddr, actualSeqKvlenAddr, batch, N1,
                                                                       remaining, 1, 1, coordinateInfo);
                return;
            }
            remaining -= N1;
        }
        return;
    }
    if (sparseMode == RIGHT_DOWN_CAUSAL) {
        CalTNDCausalLineRunIndex<CUBE_BASEM, CUBE_BASEN, false>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix0, b, N1,
                                                                k, j, r, step, coordinateInfo);
    } else {
        CalTNDCausalLineRunIndex<CUBE_BASEM, CUBE_BASEN, true>(actualSeqQlenAddr, actualSeqKvlenAddr, prefix0, b, N1, k,
                                                               j, r, step, coordinateInfo);
    }
}

template <const int64_t CUBE_BASEM, const int64_t CUBE_BASEN>
__aicore__ inline void CalTNDBandLineIndex(const __gm__ uint8_t* actualSeqQlenAddr,
                                           const __gm__ uint8_t* actualSeqKvlenAddr, const __gm__ int64_t* prefix,
                                           int64_t b, int64_t n1, int64_t k, int64_t j, int64_t r, int64_t step,
                                           CoordinateInfo& coordinateInfo)
{
    coordinateInfo.batchId = -1;
    const int64_t rawP = coordinateInfo.p;
    const int64_t rawQ = coordinateInfo.q;
    const int64_t batchTypeBase = BinarySearch(prefix, b, r, step);
    int64_t a = r - prefix[batchTypeBase];
    int64_t bIdx = batchTypeBase * step;
    if (bIdx >= b || a < 1) {
        return;
    }

    while (bIdx < b) {
        int64_t runHead = bIdx;
        int64_t runEnd = bIdx;
        int64_t m = 0;
        int64_t n = 0;
        int64_t p = 0;
        int64_t q = 0;
        ExpandTNDBandRun<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, b, bIdx, runHead, runEnd, m, n,
                                                 p, q, coordinateInfo);
        const int64_t groupSize = runEnd - runHead + 1;
        if (bIdx > runHead) {
            bIdx = runEnd + 1;
            continue;
        }
        const int64_t copies = n1 * groupSize;
        const int64_t soloRounds = CalTNDBandLineRounds(k, copies, m, n, p, q);
        if (a > soloRounds) {
            a -= soloRounds;
            bIdx = runEnd + 1;
            continue;
        }

        if (p >= m) {
            int64_t w = 0;
            int64_t x = 0;
            int64_t y = 0;
            if (CalTNDDenseLine(k, m, n, copies, j, a, w, x, y) && w >= 1 && w <= copies && x >= 1 && x <= m &&
                y >= 1 && y <= n && y <= x + q - 1) {
                BindTNDBandMappedBatch<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead, n1, w, x,
                                                               y, rawP, rawQ, coordinateInfo);
            }
            return;
        }

        if (p + q > m) {
            const int64_t denseN = CalTNDBandWideCols(m, n, p, q);
            int64_t w = 0;
            int64_t x = 0;
            int64_t y = 0;
            if (CalTNDDenseLine(k, m, denseN, copies, j, a, w, x, y) && MapTNDBandWide(m, n, p, q, x, y) && w >= 1 &&
                w <= copies) {
                BindTNDBandMappedBatch<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead, n1, w, x,
                                                               y, rawP, rawQ, coordinateInfo);
            }
            return;
        }

        const int64_t l3 = Max(static_cast<int64_t>(0), Min(p + n - m - 1, p + q - NUM_TWO));
        const int64_t seg = p + q - 1;
        const int64_t a1 = Ceil<int64_t>(a, seg);
        int64_t a2 = a % seg;
        a2 = a2 != 0 ? a2 : seg;
        if (l3 == 0 || n - m < 1) {
            const int64_t idx = (a1 - 1) * k + j;
            const int64_t w = Ceil<int64_t>(idx, n);
            int64_t y = idx % n;
            y = y != 0 ? y : n;
            const int64_t x = y + a2 - q;
            if (w >= 1 && w <= copies && x >= 1 && x <= m && y >= 1 && y <= n) {
                BindTNDBandMappedBatch<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead, n1, w, x,
                                                               y, rawP, rawQ, coordinateInfo);
            }
            return;
        }
        const int64_t yAbs = (a1 - 1) * k + j;
        const int64_t xAbs = yAbs + a2 - q;
        if (xAbs < 1) {
            return;
        }
        const int64_t w = Ceil<int64_t>(xAbs, m);
        int64_t x = xAbs % m;
        x = x != 0 ? x : m;
        const int64_t y = x + q - a2;
        if (w >= 1 && w <= copies && x >= 1 && x <= m && y >= 1 && y <= n) {
            BindTNDBandMappedBatch<CUBE_BASEM, CUBE_BASEN>(actualSeqQlenAddr, actualSeqKvlenAddr, runHead, n1, w, x, y,
                                                           rawP, rawQ, coordinateInfo);
        }
        return;
    }
}

} // namespace commondef
#endif // _FLASH_ATTENTION_SCORE_GRAD_DETER_H_
