/*
 * Adapted from ops-transformer attention/common/op_kernel/arch35/vf/
 * vf_basic_block_unaligned64_update.h, vf_mul_sel_softmaxflashv2_cast_nz.h,
 * and vf_basic_block_fullquant_unaligned256_update.h (NZ block processing).
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the repository root.
 */
#pragma once
#include "common.h"
namespace Sage {
// Production consumes prebalanced FP16 scales (PREPARED=true). Block-local
// preparation is retained for isolated diagnostics. Scores/max remain FP16.
template <uint32_t ROWS = 64, uint32_t COLS = 256, bool FULL = false, bool NZ_OUTPUT = false, bool PREPARED = false,
          uint32_t NZ_PAD_ROWS = 0>
__simd_vf__ inline void SoftmaxHalfVF(__ubuf__ half* scores, __ubuf__ float* ks, __ubuf__ float* qs,
                                      __ubuf__ half* maxima, __ubuf__ float* denom, __ubuf__ float* alpha,
                                      __ubuf__ float8_e4m3_t* out, __ubuf__ half* qhalf, __ubuf__ half* khalf,
                                      __ubuf__ float* scratch, uint32_t qstart, uint32_t sq, uint32_t start,
                                      uint32_t sk, uint32_t causal, float scale)
{
    static_assert(ROWS == 64 && COLS == 256, "half softmax requires a 64x256 AIV tile");
    using namespace AscendC::MicroAPI;
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait toHalf{RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait toFp8{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto f = CreateMask<float, MaskPattern::ALL>();
    auto h = CreateMask<half, MaskPattern::ALL>();
    if constexpr (!PREPARED) {
        uint32_t nr = qstart < sq ? Min(uint32_t(ROWS), sq - qstart) : 0;
        auto qrMask = UpdateMask<float>(nr);
        RegTensor<float> qv, kv, qmax, kmax, tmp, zero, one;
        Duplicate(zero, 0.0f);
        Duplicate(one, 1.0f);
        LoadAlign(qv, qs);
        Select(tmp, qv, zero, qrMask);
        ReduceMax(qmax, tmp, f);
        Duplicate(kmax, 0.0f);
        for (uint16_t col = 0; col < COLS; col += 64) {
            uint32_t at = start + col, valid = at < sk ? Min(uint32_t(64), sk - at) : 0;
            auto mask = UpdateMask<float>(valid);
            LoadAlign(kv, ks + col);
            Select(kv, kv, zero, mask);
            Max(kmax, kmax, kv, f);
        }
        ReduceMax(kmax, kmax, f);
        Duplicate(qmax, qmax, f);
        Duplicate(kmax, kmax, f);
        // Empty AIV halves have no valid scores; use finite harmless denominators.
        Maxs(qmax, qmax, 1.0e-30f, f);
        Maxs(kmax, kmax, 1.0e-30f, f);
        RegTensor<float> rootQ, rootK, common, qfactor, kfactor;
        Sqrt(rootQ, qmax, f);
        Sqrt(rootK, kmax, f);
        Mul(common, rootQ, rootK, f);
        Duplicate(tmp, scale);
        Abs(tmp, tmp, f);
        Muls(tmp, tmp, 2048.0f, f);
        Sqrt(tmp, tmp, f);
        Mul(common, common, tmp, f);
        Div(qfactor, common, qmax, f);
        Div(kfactor, common, kmax, f);
        Duplicate(tmp, scale);
        MaskReg negScale;
        Compare<float, CMPMODE::LT>(negScale, tmp, zero, f);
        Muls(tmp, qfactor, -1.0f, f);
        Select(qfactor, tmp, qfactor, negScale);
        RegTensor<half> packed;
        Mul(qv, qv, qfactor, f);
        Cast<half, float, toHalf>(packed, qv, f);
        StoreAlign<half, StoreDist::DIST_PACK_B32>(qhalf, packed, f);
        for (uint16_t col = 0; col < COLS; col += 64) {
            LoadAlign(kv, ks + col);
            Mul(kv, kv, kfactor, f);
            Cast<half, float, toHalf>(packed, kv, f);
            StoreAlign<half, StoreDist::DIST_PACK_B32>(khalf + col, packed, f);
        }
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    }
    auto blockMax = (__ubuf__ half*)alpha;
    UnalignReg maxStore;
    RegTensor<half> kh0, kh1, negative;
    LoadAlign(kh0, khalf);
    LoadAlign(kh1, khalf + 128);
    Duplicate(negative, half(-__builtin_inff()));
    for (uint16_t row = 0; row < ROWS; ++row) {
        RegTensor<half> qh, x, coeff, rowMax, m;
        LoadAlign<half, LoadDist::DIST_BRC_B16>(qh, qhalf + row);

#pragma unroll
        for (uint16_t col = 0; col < COLS; col += 128) {
            LoadAlign(x, scores + row * COLS + col);
            if (col == 0)
                Mul(coeff, qh, kh0, h);
            else
                Mul(coeff, qh, kh1, h);
            Mul(x, x, coeff, h);
            if constexpr (!FULL) {
                uint32_t at = start + col, pos = qstart + row, valid = at < sk ? Min(uint32_t(128), sk - at) : 0;
                if (causal)
                    valid = Min(valid, pos >= at ? pos - at + 1 : 0);
                if (pos >= sq)
                    valid = 0;
                auto mask = UpdateMask<half>(valid);
                Select(x, x, negative, mask);
            }
            StoreAlign(scores + row * COLS + col, x, h);
            // Seed from the first fragment; oldMax still protects fully masked rows.
            if (col == 0)
                rowMax = x;
            else
                Max(rowMax, rowMax, x, h);
        }
        ReduceMax(m, rowMax, h);
        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m, maxStore, 1);
    }
    StoreUnAlignPost(blockMax, maxStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    RegTensor<half> oldMax, newMax;
    RegTensor<float> ratio, oldDen;
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(oldMax, maxima);
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(newMax, (__ubuf__ half*)alpha);
    LoadAlign(oldDen, denom);
    // Keep the FP16 shifted maximum consistent across probabilities, alpha and LSE.
    Adds(newMax, newMax, half(-6.104793232414985f), h);
    Max(newMax, newMax, oldMax, h);
    ExpSub<float, half, RegLayout::ZERO>(ratio, oldMax, newMax, h);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(maxima, newMax, f);
    StoreAlign(alpha, ratio, f);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    uint32_t halfChunk = 32;
    auto first32 = UpdateMask<float>(halfChunk);
    MaskReg last32;
    Not(last32, first32, f);
    auto blockSum = scratch;
    UnalignReg sumStore;
    for (uint16_t row = 0; row < ROWS; ++row) {
        // Independent fragments expose load/ExpSub/Cast overlap. Keep the
        // original sum order, and pack directly into the (optionally padded) NZ planes.
        RegTensor<half> m, x0, x1, x2, x3;
        RegTensor<float> p0, p1, p2, p3, rowSum, sum;
        RegTensor<float8_e4m3_t> packed0, packed1, packed2, packed3;
        LoadAlign<half, LoadDist::DIST_BRC_B16>(m, maxima + row);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x0, scores + row * COLS + 0);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x1, scores + row * COLS + 64);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x2, scores + row * COLS + 128);
        LoadAlign<half, LoadDist::DIST_UNPACK_B16>(x3, scores + row * COLS + 192);
        ExpSub<float, half, RegLayout::ZERO>(p0, x0, m, h);
        ExpSub<float, half, RegLayout::ZERO>(p1, x1, m, h);
        ExpSub<float, half, RegLayout::ZERO>(p2, x2, m, h);
        ExpSub<float, half, RegLayout::ZERO>(p3, x3, m, h);
        Add(rowSum, p0, p1, f);
        Add(rowSum, rowSum, p2, f);
        Add(rowSum, rowSum, p3, f);
        Cast<float8_e4m3_t, float, toFp8>(packed0, p0, f);
        Cast<float8_e4m3_t, float, toFp8>(packed1, p1, f);
        Cast<float8_e4m3_t, float, toFp8>(packed2, p2, f);
        Cast<float8_e4m3_t, float, toFp8>(packed3, p3, f);
        if constexpr (NZ_OUTPUT) {
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 0 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32,
                                                                 packed0, first32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 1 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32 - 32,
                                                                 packed0, last32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 2 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32,
                                                                 packed1, first32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 3 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32 - 32,
                                                                 packed1, last32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 4 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32,
                                                                 packed2, first32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 5 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32 - 32,
                                                                 packed2, last32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 6 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32,
                                                                 packed3, first32);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + 7 * (ROWS + NZ_PAD_ROWS) * 32 + row * 32 - 32,
                                                                 packed3, last32);
        } else {
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row * COLS + 0, packed0, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row * COLS + 64, packed1, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row * COLS + 128, packed2, f);
            StoreAlign<float8_e4m3_t, StoreDist::DIST_PACK4_B32>(out + row * COLS + 192, packed3, f);
        }
        ReduceSum(sum, rowSum, f);
        StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(blockSum, sum, sumStore, 1);
    }
    StoreUnAlignPost(blockSum, sumStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    RegTensor<float> sum;
    LoadAlign(sum, scratch);
    MulDstAdd(oldDen, ratio, sum, f);
    StoreAlign(denom, oldDen, f);
}

// NZ scores retain eight rows in each FP16 register. ReduceDataBlock produces
// eight row statistics; FP8 P is packed into full 8x32 NZ output blocks.
template <bool MASK = false>
__simd_callee__ inline void SoftmaxWideMaxCore(__ubuf__ half* scores, __ubuf__ half* scoresNext, __ubuf__ half* maxima,
                                               __ubuf__ float* alpha, __ubuf__ half* qhalf, __ubuf__ half* khalf,
                                               int32_t queryInGroup = 0)
{
    using namespace AscendC::MicroAPI;
    // QK_READY and the scale MTE2_V event make input scores/scales visible.
    // The fence below the reduction also orders the previous maxima store.
    auto h = CreateMask<half, MaskPattern::ALL>();
    auto f = CreateMask<float, MaskPattern::ALL>();
    // NZ lane i belongs to row i/16 and column i%16. Use coordinates
    // relative to this N512 group so the signed 16-bit comparison is exact.
    RegTensor<int16_t> laneOffset;
    RegTensor<half> negative;
    if constexpr (MASK) {
        RegTensor<int16_t> lane, rowIndex, colIndex, lowBits;
        Arange(lane, int16_t(0));
        ShiftRights(rowIndex, lane, int16_t(4), h);
        Duplicate(lowBits, int16_t(15));
        And(colIndex, lane, lowBits, h);
        Sub(laneOffset, colIndex, rowIndex, h);
        Duplicate(negative, half(-__builtin_inff()));
    }
    auto blockMax = (__ubuf__ half*)alpha;
    UnalignReg maxStore;
    for (uint16_t row = 0; row < 64; row += 16) {
        RegTensor<half> q0, m0;
        LoadAlign<half, LoadDist::DIST_E2B_B16>(q0, qhalf + row + 0);
        Duplicate(m0, half(-65504.0f));
        RegTensor<half> q1, m1;
        LoadAlign<half, LoadDist::DIST_E2B_B16>(q1, qhalf + row + 8);
        Duplicate(m1, half(-65504.0f));
        for (uint16_t col = 0; col < 16; ++col) {
            {
                RegTensor<half> k, x0, x1, c0, c1;
                LoadAlign<half, LoadDist::DIST_BLK>(k, khalf + 0 + col * 16);
                LoadAlign(x0, scores + col * 64 * 16 + row * 16 + 0);
                Mul(c0, q0, k, h);
                LoadAlign(x1, scores + col * 64 * 16 + row * 16 + 128);
                Mul(c1, q1, k, h);
                Mul(x0, x0, c0, h);
                if constexpr (MASK) {
                    MaskReg valid;
                    CompareScalar<int16_t, CMPMODE::LE>(valid, laneOffset, int16_t(queryInGroup + row - col * 16), h);
                    Select(x0, x0, negative, valid);
                }
                Max(m0, m0, x0, h);
                StoreAlign(scores + col * 64 * 16 + row * 16 + 0, x0, h);
                Mul(x1, x1, c1, h);
                if constexpr (MASK) {
                    MaskReg valid;
                    CompareScalar<int16_t, CMPMODE::LE>(valid, laneOffset, int16_t(queryInGroup + row + 8 - col * 16),
                                                        h);
                    Select(x1, x1, negative, valid);
                }
                Max(m1, m1, x1, h);
                StoreAlign(scores + col * 64 * 16 + row * 16 + 128, x1, h);
            }
            {
                RegTensor<half> k, x0, x1, c0, c1;
                LoadAlign<half, LoadDist::DIST_BLK>(k, khalf + 256 + col * 16);
                LoadAlign(x0, scoresNext + col * 64 * 16 + row * 16 + 0);
                Mul(c0, q0, k, h);
                LoadAlign(x1, scoresNext + col * 64 * 16 + row * 16 + 128);
                Mul(c1, q1, k, h);
                Mul(x0, x0, c0, h);
                if constexpr (MASK) {
                    MaskReg valid;
                    CompareScalar<int16_t, CMPMODE::LE>(valid, laneOffset, int16_t(queryInGroup + row - col * 16 - 256),
                                                        h);
                    Select(x0, x0, negative, valid);
                }
                Max(m0, m0, x0, h);
                StoreAlign(scoresNext + col * 64 * 16 + row * 16 + 0, x0, h);
                Mul(x1, x1, c1, h);
                if constexpr (MASK) {
                    MaskReg valid;
                    CompareScalar<int16_t, CMPMODE::LE>(valid, laneOffset,
                                                        int16_t(queryInGroup + row + 8 - col * 16 - 256), h);
                    Select(x1, x1, negative, valid);
                }
                Max(m1, m1, x1, h);
                StoreAlign(scoresNext + col * 64 * 16 + row * 16 + 128, x1, h);
            }
        }
        ReduceDataBlock<ReduceType::MAX>(m0, m0, h);
        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m0, maxStore, 8);
        ReduceDataBlock<ReduceType::MAX>(m1, m1, h);
        StoreUnAlign<half, PostLiteral::POST_MODE_UPDATE>(blockMax, m1, maxStore, 8);
    }
    StoreUnAlignPost(blockMax, maxStore, 0);
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    RegTensor<half> oldMax, newMax;
    RegTensor<float> ratio;
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(oldMax, maxima);
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(newMax, (__ubuf__ half*)alpha);
    // Keep the FP16 shifted maximum consistent across probabilities, alpha and LSE.
    Adds(newMax, newMax, half(-6.104793232414985f), h);
    Max(newMax, newMax, oldMax, h);
    ExpSub<float, half, RegLayout::ZERO>(ratio, oldMax, newMax, h);
    StoreAlign<half, StoreDist::DIST_PACK_B32>(maxima, newMax, f);
    StoreAlign(alpha, ratio, f);
}

template <bool FIRST, bool UPDATE_DEN = true>
__simd_callee__ inline void SoftmaxWideProbCore(__ubuf__ half* scores, __ubuf__ half* maxima, __ubuf__ float* denom,
                                                __ubuf__ float* alpha, __ubuf__ float8_e4m3_t* out,
                                                __ubuf__ float* scratch, __ubuf__ float* partialSums)
{
    using namespace AscendC::MicroAPI;
    // FIRST reads MaxCore's maxima; SECOND reads FIRST's partial lane sums.
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    static constexpr CastTrait cast0{RegLayout::ZERO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast1{RegLayout::ONE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast2{RegLayout::TWO, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    static constexpr CastTrait cast3{RegLayout::THREE, SatMode::SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
    auto f = CreateMask<float, MaskPattern::ALL>();
    auto h = CreateMask<half, MaskPattern::ALL>();
    auto b = CreateMask<uint8_t, MaskPattern::ALL>();
    auto sumMask = CreateMask<float, MaskPattern::VL8>();
    for (uint16_t row = 0; row < 64; row += 8) {
        RegTensor<half> m;
        RegTensor<float> rowSum;
        LoadAlign<half, LoadDist::DIST_E2B_B16>(m, maxima + row);
        if constexpr (FIRST)
            Duplicate(rowSum, 0.0f);
        else
            LoadAlign(rowSum, partialSums + row * 8);
        for (uint16_t col = 0; col < 8; col += 2) {
            // Load the next NZ pair before computing the current pair so its
            // probability instructions can overlap the current FP8 packing.
            RegTensor<half> next0, next1;
            LoadAlign(next0, scores + (col + 1) * 64 * 32 + row * 16);
            LoadAlign(next1, scores + (col + 1) * 64 * 32 + 64 * 16 + row * 16);
            {
                RegTensor<half> x0, x1;
                RegTensor<float> p0, p1, p2, p3, pair0, pair1;
                RegTensor<float8_e4m3_t> packed0, packed1, packed2, packed3;
                RegTensor<uint8_t> merge0, merge1, merged;
                LoadAlign(x0, scores + (col + 0) * 64 * 32 + row * 16);
                LoadAlign(x1, scores + (col + 0) * 64 * 32 + 64 * 16 + row * 16);
                ExpSub<float, half, RegLayout::ZERO>(p0, x0, m, h);
                ExpSub<float, half, RegLayout::ONE>(p1, x0, m, h);
                ExpSub<float, half, RegLayout::ZERO>(p2, x1, m, h);
                ExpSub<float, half, RegLayout::ONE>(p3, x1, m, h);
                Add(pair0, p0, p1, f);
                Add(pair1, p2, p3, f);
                Add(pair0, pair0, pair1, f);
                Add(rowSum, rowSum, pair0, f);
                Cast<float8_e4m3_t, float, cast0>(packed0, p0, f);
                Cast<float8_e4m3_t, float, cast2>(packed1, p1, f);
                Cast<float8_e4m3_t, float, cast1>(packed2, p2, f);
                Cast<float8_e4m3_t, float, cast3>(packed3, p3, f);
                Or(merge0, (RegTensor<uint8_t>&)packed0, (RegTensor<uint8_t>&)packed1, b);
                Or(merge1, (RegTensor<uint8_t>&)packed2, (RegTensor<uint8_t>&)packed3, b);
                Or(merged, merge0, merge1, b);
                // Overwrite only the first FP16 stripe just loaded for these 8
                // rows. Other rows and column blocks remain disjoint. MTE3
                // skips the unused second stripe when copying packed P.
                StoreAlign((__ubuf__ uint8_t*)out + (col + 0) * 64 * 64 + row * 32, merged, b);
            }
            {
                RegTensor<float> p0, p1, p2, p3, pair0, pair1;
                RegTensor<float8_e4m3_t> packed0, packed1, packed2, packed3;
                RegTensor<uint8_t> merge0, merge1, merged;
                ExpSub<float, half, RegLayout::ZERO>(p0, next0, m, h);
                ExpSub<float, half, RegLayout::ONE>(p1, next0, m, h);
                ExpSub<float, half, RegLayout::ZERO>(p2, next1, m, h);
                ExpSub<float, half, RegLayout::ONE>(p3, next1, m, h);
                Add(pair0, p0, p1, f);
                Add(pair1, p2, p3, f);
                Add(pair0, pair0, pair1, f);
                Add(rowSum, rowSum, pair0, f);
                Cast<float8_e4m3_t, float, cast0>(packed0, p0, f);
                Cast<float8_e4m3_t, float, cast2>(packed1, p1, f);
                Cast<float8_e4m3_t, float, cast1>(packed2, p2, f);
                Cast<float8_e4m3_t, float, cast3>(packed3, p3, f);
                Or(merge0, (RegTensor<uint8_t>&)packed0, (RegTensor<uint8_t>&)packed1, b);
                Or(merge1, (RegTensor<uint8_t>&)packed2, (RegTensor<uint8_t>&)packed3, b);
                Or(merged, merge0, merge1, b);
                // Overwrite only the first FP16 stripe just loaded for these 8
                // rows. Other rows and column blocks remain disjoint. MTE3
                // skips the unused second stripe when copying packed P.
                StoreAlign((__ubuf__ uint8_t*)out + (col + 1) * 64 * 64 + row * 32, merged, b);
            }
        }
        if constexpr (FIRST) {
            StoreAlign(partialSums + row * 8, rowSum, f);
        } else {
            ReduceDataBlock<ReduceType::SUM>(rowSum, rowSum, f);
            StoreAlign(scratch + row, rowSum, sumMask);
        }
    }
    if constexpr (!FIRST && UPDATE_DEN) {
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> oldDen, ratio;
        LoadAlign(oldDen, denom);
        LoadAlign(ratio, alpha);
        RegTensor<float> sum;
        LoadAlign(sum, scratch);
        MulDstAdd(oldDen, ratio, sum, f);
        StoreAlign(denom, oldDen, f);
    }
}
template <bool MASK = false>
__simd_vf__ inline void SoftmaxWideFirstVF(__ubuf__ half* scores, __ubuf__ half* scoresNext, __ubuf__ half* maxima,
                                           __ubuf__ float* denom, __ubuf__ float* alpha, __ubuf__ float8_e4m3_t* out,
                                           __ubuf__ half* qhalf, __ubuf__ half* khalf, __ubuf__ float* scratch,
                                           __ubuf__ float* partialSums, int32_t queryInGroup = 0)
{
    SoftmaxWideMaxCore<MASK>(scores, scoresNext, maxima, alpha, qhalf, khalf, queryInGroup);
    SoftmaxWideProbCore<true>(scores, maxima, denom, alpha, out, scratch, partialSums);
}
__simd_vf__ inline void SoftmaxWideSecondVF(__ubuf__ half* scores, __ubuf__ half* maxima, __ubuf__ float* denom,
                                            __ubuf__ float* alpha, __ubuf__ float8_e4m3_t* out, __ubuf__ float* scratch,
                                            __ubuf__ float* partialSums)
{
    SoftmaxWideProbCore<false, false>(scores, maxima, denom, alpha, out, alpha + 64, partialSums);
}
} // namespace Sage
