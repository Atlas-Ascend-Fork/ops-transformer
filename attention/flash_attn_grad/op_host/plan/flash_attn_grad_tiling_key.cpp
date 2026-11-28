/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attn_grad_tiling_key.h"

namespace optiling {
namespace {

constexpr uint64_t KEY_SHIFT_LAYOUT = 2;
constexpr uint64_t KEY_SHIFT_MASK_MODE = 3;
constexpr uint64_t KEY_SHIFT_SWIZZLE = 5;
constexpr uint64_t KEY_SHIFT_D_ALIGN = 6;
constexpr uint64_t KEY_SHIFT_DV_ALIGN = 9;
constexpr uint64_t KEY_SHIFT_BN2_MULTIBLK = 12;
constexpr uint64_t KEY_SHIFT_BN2_NEED_ZERO = 13;
constexpr uint64_t KEY_SHIFT_DETER = 14;
constexpr uint64_t KEY_SHIFT_SINK = 15;
constexpr uint64_t KEY_SHIFT_PRE_SFMG = 16;
constexpr uint64_t KEY_SHIFT_DETER_FAMILY = 17;
constexpr uint64_t KEY_SHIFT_NZ_OUT = 22;

uint64_t AlignBits(int64_t align)
{
    if (align == 64) {
        return 0ULL;
    }
    if (align == 128) {
        return 1ULL;
    }
    if (align == 192) {
        return 2ULL;
    }
    if (align == 256) {
        return 3ULL;
    }
    if (align == 512) {
        return 5ULL; // 追加尾部，不动 768 的码点 4（老仓 NUM512 档移植）
    }
    return 4ULL; // 768
}

} // namespace

uint64_t EncodeTilingKey(const FagParsedInfo& info, const FagKernelPlan& kernel, const FagSchedulePlan& schedule)
{
    uint64_t maskModeBits = 0ULL;
    if (info.hasAttenMask && info.maskMode == MASK_MODE_CAUSAL) {
        maskModeBits = 1ULL;
    } else if (info.hasAttenMask && info.maskMode == MASK_MODE_WINDOW) {
        maskModeBits = 2ULL;
    }
    const uint64_t deterBit =
        (info.isDeterministic && (kernel.tmpl == TMPL_BN2GS1S2 || kernel.tmpl == TMPL_BN2S2)) ? 1ULL : 0ULL;
    // This candidate specializes the existing metadata-backed BN2S2 family.
    // Existing non-plan/GS variants retain their original key and behavior.
    const bool preSfmg = info.useDeterPlan && kernel.tmpl == TMPL_BN2S2 &&
                         fag_deter::DecidePreSfmg(info.deterConfig.schedule.route, kernel.dvAlign, info.hasSinks,
                                                  info.s2, schedule.s1Outer);
    return static_cast<uint64_t>(kernel.tmpl) | (static_cast<uint64_t>(info.layout) << KEY_SHIFT_LAYOUT) |
           (maskModeBits << KEY_SHIFT_MASK_MODE) | (static_cast<uint64_t>(schedule.swizzle) << KEY_SHIFT_SWIZZLE) |
           (AlignBits(kernel.dAlign) << KEY_SHIFT_D_ALIGN) | (AlignBits(kernel.dvAlign) << KEY_SHIFT_DV_ALIGN) |
           (static_cast<uint64_t>(kernel.isBn2MultiBlk) << KEY_SHIFT_BN2_MULTIBLK) |
           (static_cast<uint64_t>(kernel.bn2NeedZero) << KEY_SHIFT_BN2_NEED_ZERO) | (deterBit << KEY_SHIFT_DETER) |
           (static_cast<uint64_t>(info.hasSinks ? 1 : 0) << KEY_SHIFT_SINK) |
           (static_cast<uint64_t>(preSfmg) << KEY_SHIFT_PRE_SFMG) |
           (static_cast<uint64_t>(schedule.deter.family) << KEY_SHIFT_DETER_FAMILY) |
           (static_cast<uint64_t>(kernel.nzOut) << KEY_SHIFT_NZ_OUT);
}

} // namespace optiling
