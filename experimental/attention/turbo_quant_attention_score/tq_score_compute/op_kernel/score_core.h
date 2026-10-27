/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TURBO_QUANT_ATTENTION_SCORE_TQ_SCORE_COMPUTE_SCORE_CORE_H_
#define TURBO_QUANT_ATTENTION_SCORE_TQ_SCORE_COMPUTE_SCORE_CORE_H_

#include "tq_score_compute_config.h"
#include "kernel_operator.h"
using namespace tqv2;
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
using namespace AscendC;
using AType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using BType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using CType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulConfig CONFIG = GetNormalConfig();
constexpr auto MM_TILING = GetMatmulApiTiling<AType, BType, CType, CType>(CONFIG);
template <HardEvent E>
__aicore__ inline void Fence()
{
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}
// Each AIV decodes 64 keys; W keys per vector microtile.
// Layout offsets are generated once by the host, never scalarized per key.
template <int BITS, int K, typename Meta>
class Decoder {
    TPipe p;
    TBuf<TPosition::VECCALC> rawBuf, halfBuf, intsBuf, offsetBuf, laneBuf, tempBuf, fpBuf, bfBuf, metaBuf, queryBuf,
        lutBuf, pairOffsetsBuf;
    GlobalTensor<uint8_t> idx, signs;
    GlobalTensor<Meta> norm, gamma;
    GlobalTensor<bfloat16_t> qr, qq, a, b;
    GlobalTensor<int32_t> table, lens;
    GlobalTensor<float> out;
    uint32_t heads, kvh, pages, length, cores, groups, groupTiles, physicalBlocks;
    uint32_t currentHead = 0;
    __aicore__ inline uint32_t MetaBase()
    {
        return kvh <= 16 ? currentHead * 4 : 0;
    }

public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR s, GM_ADDR i, GM_ADDR j, GM_ADDR n, GM_ADDR g, GM_ADDR t, GM_ADDR l,
                                GM_ADDR o, GM_ADDR aw, GM_ADDR bw, GM_ADDR offsets, uint32_t kh, uint32_t pg,
                                uint32_t len, uint32_t nc, uint32_t qheads, uint32_t cacheBlocks)
    {
        kvh = kh;
        heads = qheads;
        groups = heads / kh;
        groupTiles = (groups + 3) / 4;
        pages = pg;
        length = len;
        cores = nc;
        physicalBlocks = cacheBlocks;
        qr.SetGlobalBuffer((__gm__ bfloat16_t*)q);
        qq.SetGlobalBuffer((__gm__ bfloat16_t*)s);
        idx.SetGlobalBuffer((__gm__ uint8_t*)i);
        signs.SetGlobalBuffer((__gm__ uint8_t*)j);
        norm.SetGlobalBuffer((__gm__ Meta*)n);
        gamma.SetGlobalBuffer((__gm__ Meta*)g);
        table.SetGlobalBuffer((__gm__ int32_t*)t);
        lens.SetGlobalBuffer((__gm__ int32_t*)l);
        out.SetGlobalBuffer((__gm__ float*)o);
        a.SetGlobalBuffer((__gm__ bfloat16_t*)aw);
        b.SetGlobalBuffer((__gm__ bfloat16_t*)bw);
        p.InitBuffer(rawBuf, RAW_BYTES);
        p.InitBuffer(halfBuf, RAW_BYTES * 2);
        p.InitBuffer(intsBuf, RAW_BYTES * 4);
        p.InitBuffer(offsetBuf, OFFSET_COUNT * 4);
        p.InitBuffer(laneBuf, E * 4);
        p.InitBuffer(tempBuf, BITS == 3 && K == 512 ? W * 8 * 4 : E * 4);
        p.InitBuffer(fpBuf, E * 4);
        p.InitBuffer(bfBuf, BITS == 3 && K == 512 ? 4 * K * 2 : E * 2);
        p.InitBuffer(metaBuf, W * 64);
        p.InitBuffer(queryBuf, 4 * K * 2);
        p.InitBuffer(lutBuf, 64);
        if constexpr (BITS == 3 && K == 512) {
            p.InitBuffer(pairOffsetsBuf, W * 8 * 4 * 2);
            auto indices = tempBuf.Get<int32_t>();
            CreateVecIndex(indices, int32_t(0), W);
            PipeBarrier<PIPE_V>();
            ShiftLeft(indices, indices, 5, W);
            PipeBarrier<PIPE_V>();
            Brcb(pairOffsetsBuf.Get<int32_t>(), indices, W / 8, BrcbRepeatParams{1, 8});
            PipeBarrier<PIPE_V>();
            ShiftRight(indices, indices, 3, W);
            PipeBarrier<PIPE_V>();
            Brcb(pairOffsetsBuf.Get<int32_t>()[W * 8], indices, W / 8, BrcbRepeatParams{1, 8});
            PipeBarrier<PIPE_V>();
        }
        GlobalTensor<uint32_t> off;
        off.SetGlobalBuffer((__gm__ uint32_t*)offsets);
        DataCopy(offsetBuf.Get<uint32_t>(), off, OFFSET_COUNT);
        auto lut = lutBuf.Get<float>();
        constexpr float v2[4] = {-.1335033178f, -.04002048075f, .04002048075f, .1335033178f};
        constexpr float v3[8] = {-.19020693f,   -.1187859178f, -.06682205945f, -.02166347019f,
                                 .02166347019f, .06682205945f, .1187859178f,   .19020693f};
        constexpr float v4[16] = {-.2414890379f,  -.1828317791f,  -.1429702938f,  -.1109927073f,
                                  -.08325428516f, -.05802082643f, -.03428063914f, -.01134236995f,
                                  .01134236995f,  .03428063914f,  .05802082643f,  .08325428516f,
                                  .1109927073f,   .1429702938f,   .1828317791f,   .2414890379f};
        for (int x = 0; x < (1 << BITS); ++x) {
            lut.SetValue(x, BITS == 2 ? v2[x] : (BITS == 3 ? v3[x] : v4[x]));
        }
        Fence<HardEvent::S_V>();
        Fence<HardEvent::MTE2_V>();
    }
    __aicore__ inline void Chunk(uint64_t row, uint64_t dest)
    {
        auto raw = rawBuf.Get<uint8_t>();
        auto h = halfBuf.Get<half>();
        auto ii = intsBuf.Get<int32_t>();
        auto offsets = offsetBuf.Get<uint32_t>();
        // The byte-to-int Cast below finishes before Gather reuses this buffer.
        auto words = halfBuf.Get<int32_t>();
        auto tmp = words[WORDS];
        auto sw = words[2 * WORDS];
        auto lane = laneBuf.Get<int32_t>();
        auto work = tempBuf.Get<float>();
        auto fp = fpBuf.Get<float>();
        auto meta = metaBuf.Get<Meta>();
        DataCopyExtParams ip{W, 16 * BITS, (kvh - 1) * 16 * BITS, 0, 0};
        DataCopyPad(raw, idx[row * 16 * BITS], ip, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        DataCopyExtParams sp{W, 16, (kvh - 1) * 16, 0, 0};
        DataCopyPad(raw[W * 64], signs[row * 16], sp, DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
        if (kvh <= 16) {
            // Read all heads contiguously: two larger transfers replace 128
            // tiny strided metadata rows. Select this head in UB afterwards.
            DataCopy(meta, norm[row - currentHead], W * kvh);
            DataCopy(meta[WORDS], gamma[row - currentHead], W * kvh);
        } else {
            // Each 2-byte GM block occupies one 32-byte UB row on 910B.  The
            // host metadata offsets therefore use a 16-element stride.
            DataCopyExtParams mp{W, 2, (kvh - 1) * 2, 0, 0};
            DataCopyPad(meta, norm[row], mp, DataCopyPadExtParams<Meta>{false, 0, 0, 0});
            DataCopyPad(meta[WORDS], gamma[row], mp, DataCopyPadExtParams<Meta>{false, 0, 0, 0});
        }
        Fence<HardEvent::MTE2_V>();
        Cast(h, raw, RoundMode::CAST_NONE, RAW_BYTES);
        PipeBarrier<PIPE_V>();
        Cast(ii, h, RoundMode::CAST_RINT, RAW_BYTES);
        PipeBarrier<PIPE_V>();
        Gather(words, ii, offsets, 0, WORDS);
        Gather(sw, ii[W * 64], offsets[4 * WORDS], 0, WORDS);
        PipeBarrier<PIPE_V>();
        for (int byte = 1; byte < BITS; ++byte) {
            Gather(tmp, ii, offsets[byte * WORDS], 0, WORDS);
            PipeBarrier<PIPE_V>();
            ShiftLeft(tmp, tmp, byte * 8, WORDS);
            PipeBarrier<PIPE_V>();
            Or(words.ReinterpretCast<uint16_t>(), words.ReinterpretCast<uint16_t>(), tmp.ReinterpretCast<uint16_t>(),
               2 * WORDS);
            PipeBarrier<PIPE_V>();
        }
        if constexpr (BITS == 3 && K == 512) {
            EmitPairs(dest);
            Fence<HardEvent::MTE3_MTE2>();
            return;
        }
        // Emit [key, bit_lane, packed_word] directly; query uses the same
        // coordinate permutation, avoiding a full decoded-key Gather transpose.
        for (int x = 0; x < 8; ++x) {
            ShiftRight(lane[x * 16], words, int32_t(x * BITS), uint64_t(16), W, UnaryRepeatParams{1, 1, 16, 2});
        }
        Duplicate(work.ReinterpretCast<int32_t>(), int32_t((1 << BITS) - 1), E);
        PipeBarrier<PIPE_V>();
        And(lane.ReinterpretCast<uint16_t>(), lane.ReinterpretCast<uint16_t>(), work.ReinterpretCast<uint16_t>(),
            E * 2);
        PipeBarrier<PIPE_V>();
        ShiftLeft(lane, lane, 2, E);
        PipeBarrier<PIPE_V>();
        Gather(fp, lutBuf.Get<float>(), lane.ReinterpretCast<uint32_t>(), 0, E);
        PipeBarrier<PIPE_V>();
        Cast(lane.ReinterpretCast<float>(), meta, RoundMode::CAST_NONE, WORDS);
        PipeBarrier<PIPE_V>();
        Gather(work, lane.ReinterpretCast<float>(), offsets[5 * WORDS + QUERY_OFFSETS], MetaBase(), W);
        PipeBarrier<PIPE_V>();
        Brcb(lane.ReinterpretCast<float>(), work, W / 8, BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();
        Mul(fp, fp, lane.ReinterpretCast<float>(), uint64_t(64), W, BinaryRepeatParams{1, 1, 0, 16, 16, 1});
        Mul(fp[64], fp[64], lane.ReinterpretCast<float>(), uint64_t(64), W, BinaryRepeatParams{1, 1, 0, 16, 16, 1});
        PipeBarrier<PIPE_V>();
        Store(dest);
        for (int x = 0; x < 8; ++x) {
            ShiftRight(lane[x * 16], sw, int32_t(x), uint64_t(16), W, UnaryRepeatParams{1, 1, 16, 2});
        }
        Duplicate(work.ReinterpretCast<int32_t>(), int32_t(1), E);
        PipeBarrier<PIPE_V>();
        And(lane.ReinterpretCast<uint16_t>(), lane.ReinterpretCast<uint16_t>(), work.ReinterpretCast<uint16_t>(),
            E * 2);
        PipeBarrier<PIPE_V>();
        Cast(fp, lane, RoundMode::CAST_NONE, E);
        PipeBarrier<PIPE_V>();
        Muls(fp, fp, 2.0f, E);
        PipeBarrier<PIPE_V>();
        Adds(fp, fp, -1.0f, E);
        PipeBarrier<PIPE_V>();
        Cast(lane.ReinterpretCast<float>(), meta[WORDS], RoundMode::CAST_NONE, WORDS);
        PipeBarrier<PIPE_V>();
        Gather(work, lane.ReinterpretCast<float>(), offsets[5 * WORDS + QUERY_OFFSETS], MetaBase(), W);
        PipeBarrier<PIPE_V>();
        Muls(work, work, 0.009791516697777346f, W);
        PipeBarrier<PIPE_V>();
        Brcb(lane.ReinterpretCast<float>(), work, W / 8, BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();
        Mul(fp, fp, lane.ReinterpretCast<float>(), uint64_t(64), W, BinaryRepeatParams{1, 1, 0, 16, 16, 1});
        Mul(fp[64], fp[64], lane.ReinterpretCast<float>(), uint64_t(64), W, BinaryRepeatParams{1, 1, 0, 16, 16, 1});
        PipeBarrier<PIPE_V>();
        Store(dest + 128);
        Fence<HardEvent::MTE3_MTE2>();
    }
    // Pack two BF16 values into each uint32 lookup entry. The low 16 bits
    // hold the rounded value and the high 16 bits hold its residual.
    __aicore__ inline void PackPairs(LocalTensor<uint32_t> dst, uint32_t count)
    {
        auto fp = fpBuf.Get<float>();
        auto work = tempBuf.Get<float>();
        auto bf = bfBuf.Get<bfloat16_t>();
        Cast(bf, fp, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Cast(work, bf, RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Sub(fp, fp, work, count);
        PipeBarrier<PIPE_V>();
        ShiftRight(work.ReinterpretCast<uint32_t>(), work.ReinterpretCast<uint32_t>(), uint32_t(16), count);
        Cast(bf[count], fp, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        Cast(fp, bf[count], RoundMode::CAST_NONE, count);
        PipeBarrier<PIPE_V>();
        Or(dst.ReinterpretCast<uint16_t>(), work.ReinterpretCast<uint16_t>(), fp.ReinterpretCast<uint16_t>(),
           count * 2);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AddKeyOffsets(uint32_t shift)
    {
        auto lane = laneBuf.Get<int32_t>();
        auto base = pairOffsetsBuf.Get<int32_t>()[shift == 5 ? 0 : W * 8];
        Add(lane, lane, base, uint64_t(64), W, BinaryRepeatParams{1, 1, 0, 16, 16, 1});
        Add(lane[64], lane[64], base, uint64_t(64), W, BinaryRepeatParams{1, 1, 0, 16, 16, 1});
        PipeBarrier<PIPE_V>();
    }

    // Exact same two-BF16 approximation as Store(), but evaluate it for only
    // 8 centroids and 2 signs per key, not for every decoded coordinate.
    __aicore__ inline void EmitPairs(uint64_t dest)
    {
        auto words = halfBuf.Get<int32_t>();
        auto sw = words[2 * WORDS];
        auto lane = laneBuf.Get<int32_t>();
        auto work = tempBuf.Get<float>();
        auto fp = fpBuf.Get<float>();
        auto pairs = intsBuf.Get<uint32_t>(); // Byte conversion is finished.
        auto meta = metaBuf.Get<Meta>();
        auto metaOffsets = offsetBuf.Get<uint32_t>()[5 * WORDS + QUERY_OFFSETS];
        // Extract the code already scaled to a 4-byte lookup offset.
        ShiftLeft(lane, words, int32_t(2), uint64_t(16), W, UnaryRepeatParams{1, 1, 16, 2});
        for (int x = 1; x < 8; ++x) {
            ShiftRight(lane[x * 16], words, int32_t(x * 3 - 2), uint64_t(16), W, UnaryRepeatParams{1, 1, 16, 2});
        }
        Duplicate(work.ReinterpretCast<int32_t>(), int32_t(28), 8);
        PipeBarrier<PIPE_V>();
        And(lane.ReinterpretCast<uint16_t>(), lane.ReinterpretCast<uint16_t>(), work.ReinterpretCast<uint16_t>(),
            uint64_t(128), E / 64, BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        Cast(fp, meta, RoundMode::CAST_NONE, WORDS);
        PipeBarrier<PIPE_V>();
        Gather(work, fp, metaOffsets, MetaBase(), W);
        PipeBarrier<PIPE_V>();
        Brcb(fp, work, W / 8, BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();
        Mul(fp, fp, lutBuf.Get<float>(), uint64_t(64), W / 8, BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        PackPairs(pairs, W * 8);
        AddKeyOffsets(5); // Each key owns eight 4-byte packed pairs.
        Gather(fp.ReinterpretCast<uint32_t>(), pairs, lane.ReinterpretCast<uint32_t>(), 0, E);
        Fence<HardEvent::V_MTE3>();
        DataCopy(b[dest], fp.ReinterpretCast<bfloat16_t>(), DataCopyParams{W, 16, 0, 16});
        Fence<HardEvent::MTE3_V>();
        // Start from -gamma. A set QJL bit toggles both BF16 sign bits,
        // yielding +gamma without a per-coordinate floating-point multiply.
        Cast(fp, meta[WORDS], RoundMode::CAST_NONE, WORDS);
        PipeBarrier<PIPE_V>();
        Gather(work, fp, metaOffsets, MetaBase(), W);
        PipeBarrier<PIPE_V>();
        Muls(fp, work, -0.009791516697777346f, W);
        PipeBarrier<PIPE_V>();
        PackPairs(pairs, W);
        Brcb(words.ReinterpretCast<uint32_t>(), pairs, W / 8, BrcbRepeatParams{1, 8});
        PipeBarrier<PIPE_V>();
        // Duplicate each packed sign byte into both uint16 halves once.
        Muls(sw, sw, int32_t(65537), WORDS);
        PipeBarrier<PIPE_V>();
        for (int x = 0; x < 8; ++x) {
            ShiftLeft(lane[x * 16], sw, int32_t(15 - x), uint64_t(16), W, UnaryRepeatParams{1, 1, 16, 2});
        }
        Duplicate(work.ReinterpretCast<uint16_t>(), uint16_t(0x8000), 16);
        PipeBarrier<PIPE_V>();
        And(lane.ReinterpretCast<uint16_t>(), lane.ReinterpretCast<uint16_t>(), work.ReinterpretCast<uint16_t>(),
            uint64_t(128), E / 64, BinaryRepeatParams{1, 1, 0, 8, 8, 0});
        PipeBarrier<PIPE_V>();
        auto mask16 = lane.ReinterpretCast<uint16_t>();
        auto gamma16 = words.ReinterpretCast<uint16_t>();
        auto out16 = fp.ReinterpretCast<uint16_t>();
        // Modulo-16-bit addition of 0x8000 toggles the sign bit of each BF16
        // half independently, including negative residual halves.
        Add(out16.ReinterpretCast<int16_t>(), mask16.ReinterpretCast<int16_t>(), gamma16.ReinterpretCast<int16_t>(),
            uint64_t(128), W, BinaryRepeatParams{1, 1, 0, 16, 16, 1});
        Add(out16[128].ReinterpretCast<int16_t>(), mask16[128].ReinterpretCast<int16_t>(),
            gamma16.ReinterpretCast<int16_t>(), uint64_t(128), W, BinaryRepeatParams{1, 1, 0, 16, 16, 1});
        Fence<HardEvent::V_MTE3>();
        DataCopy(b[dest + 256], fp.ReinterpretCast<bfloat16_t>(), DataCopyParams{W, 16, 0, 16});
        Fence<HardEvent::MTE3_V>();
    }

    __aicore__ inline void Store(uint64_t dest)
    {
        auto bf = bfBuf.Get<bfloat16_t>();
        auto fp = fpBuf.Get<float>();
        auto work = tempBuf.Get<float>();
        Fence<HardEvent::MTE3_V>();
        Cast(bf, fp, RoundMode::CAST_RINT, E);
        Fence<HardEvent::V_MTE3>();
        DataCopy(b[dest], bf, DataCopyParams{W, 8, 0, uint16_t(K / 16 - 8)});
        if constexpr (K == 512) {
            Cast(work, bf, RoundMode::CAST_NONE, E);
            PipeBarrier<PIPE_V>();
            Sub(fp, fp, work, E);
            PipeBarrier<PIPE_V>();
            Fence<HardEvent::MTE3_V>();
            Cast(bf, fp, RoundMode::CAST_RINT, E);
            Fence<HardEvent::V_MTE3>();
            DataCopy(b[dest + 256], bf, DataCopyParams{W, 8, 0, uint16_t(K / 16 - 8)});
        }
        Fence<HardEvent::MTE3_V>();
    }
    __aicore__ inline void Run(uint32_t batch)
    {
        uint32_t core = GetBlockIdx() / 2, sub = GetBlockIdx() % 2;
        uint32_t tiles = (length + 127) / 128, tasks = batch * kvh * groupTiles * tiles;
        uint32_t count = core < tasks ? (tasks - 1 - core) / cores + 1 : 0;
        uint32_t queryTag[2] = {0xffffffffu, 0xffffffffu};
        for (uint32_t it = 0; it < count; ++it) {
            uint32_t slot = it % 2, task = core + it * cores;
            if (it >= 2) {
                CrossCoreWaitFlag(4 + slot);
            }
            uint32_t seq = task / (kvh * groupTiles * tiles), hg = (task / tiles) % (kvh * groupTiles);
            uint32_t head = hg / groupTiles, qbase = head * groups + (hg % groupTiles) * 4;
            uint32_t active = groups - (hg % groupTiles) * 4;
            if (active > 4) {
                active = 4;
            }
            uint32_t tile = task % tiles;
            uint32_t sl = lens.GetValue(seq), start = tile * 128;
            uint64_t base = (uint64_t(core) * 2 + slot);
            int32_t physicalBlock = table.GetValue(seq * pages + tile);
            bool validPhysicalBlock = physicalBlock >= 0 && static_cast<uint32_t>(physicalBlock) < physicalBlocks;
            // Initialize padding independently of the Cube's valid N tail.
            if (!validPhysicalBlock || start + 128 > sl) {
                auto pad = fpBuf.Get<float>();
                Duplicate(pad, -__builtin_inff(), 128);
                Fence<HardEvent::V_MTE3>();
                uint32_t lo = start + sub * 64;
                uint32_t end = lo + 64 < length ? lo + 64 : length;
                if (validPhysicalBlock && lo < sl) {
                    lo = sl;
                }
                if (lo < end) {
                    for (int h = 0; h < active; ++h) {
                        DataCopyPad(out[(uint64_t(seq) * heads + qbase + h) * length + lo], pad,
                                    DataCopyExtParams{1, (end - lo) * 4, 0, 0, 0});
                    }
                }
                Fence<HardEvent::MTE3_V>();
            }
            if (start < sl && validPhysicalBlock) {
                if (sub == 0 && queryTag[slot] != seq * (kvh * groupTiles) + hg) {
                    queryTag[slot] = seq * (kvh * groupTiles) + hg;
                    auto query = queryBuf.Get<bfloat16_t>();
                    for (int h = 0; h < active; ++h) {
                        uint64_t qoff = (uint64_t(seq) * heads + qbase + h) * 128;
                        DataCopy(query[h * K], qr[qoff], 128);
                        DataCopy(query[h * K + 128], qq[qoff], 128);
                    }
                    Fence<HardEvent::MTE2_V>();
                    Gather(bfBuf.Get<bfloat16_t>(), query, offsetBuf.Get<uint32_t>()[5 * WORDS], 0, 4 * K);
                    Fence<HardEvent::V_MTE3>();
                    DataCopy(a[base * 4 * K], bfBuf.Get<bfloat16_t>(), 4 * K);
                    Fence<HardEvent::MTE3_V>();
                }
                currentHead = head;
                for (int chunk = 0; chunk < 64 / W; ++chunk) {
                    uint32_t key = sub * 64 + chunk * W;
                    Chunk((uint64_t(static_cast<uint32_t>(physicalBlock)) * 128 + key) * kvh + head,
                          base * 128 * K + key * K);
                }
            }
            NotifyEvent<PIPE_MTE3>(6 + slot);
        }
        // Drain both slots before TPipe storage is released.
        if (count >= 2) {
            CrossCoreWaitFlag(4 + (count - 2) % 2);
        }
        if (count >= 1) {
            CrossCoreWaitFlag(4 + (count - 1) % 2);
        }
    }
    // Each core owns two query scratch slots. Packing and consumption are
    // synchronized by the same page producer/consumer flags, without a grid barrier.
    __aicore__ inline void ProcessPrefill(uint32_t tokens)
    {
        uint32_t core = GetBlockIdx() / 2, sub = GetBlockIdx() % 2;
        uint32_t tiles = (length + 127) / 128, tasks = kvh * groupTiles * tiles;
        uint32_t count = core < tasks ? (tasks - 1 - core) / cores + 1 : 0;
        uint32_t padded = (tokens + TILE_M - 1) / TILE_M * TILE_M;
        uint32_t validLength = lens.GetValue(0);
        for (uint32_t it = 0; it < count; ++it) {
            uint32_t slot = it % 2, task = core + it * cores;
            if (it >= 2) {
                CrossCoreWaitFlag(4 + slot);
            }
            uint32_t hg = task / tiles, head = hg / groupTiles, page = task % tiles;
            uint32_t qbase = head * groups + (hg % groupTiles) * 4;
            uint32_t active = groups - (hg % groupTiles) * 4;
            if (active > 4) {
                active = 4;
            }
            uint64_t base = uint64_t(core) * 2 + slot;
            uint32_t start = page * 128;
            int32_t physicalBlock = table.GetValue(page);
            bool validPhysicalBlock = physicalBlock >= 0 && static_cast<uint32_t>(physicalBlock) < physicalBlocks;
            auto pad = fpBuf.Get<float>();
            uint32_t lo = start + sub * 64, end = lo + 64 < length ? lo + 64 : length;
            if (validPhysicalBlock && lo < validLength) {
                lo = validLength;
            }
            if (lo < end) {
                Duplicate(pad, -__builtin_inff(), 128);
                Fence<HardEvent::V_MTE3>();
                for (uint32_t token = 0; token < tokens; ++token) {
                    for (uint32_t h = 0; h < active; ++h) {
                        DataCopyPad(out[(uint64_t(token) * heads + qbase + h) * length + lo], pad,
                                    DataCopyExtParams{1, (end - lo) * 4, 0, 0, 0});
                    }
                }
                Fence<HardEvent::MTE3_V>();
            }
            if (start < validLength && validPhysicalBlock) {
                currentHead = head;
                Chunk((uint64_t(static_cast<uint32_t>(physicalBlock)) * 128 + sub * 64) * kvh + head,
                      base * 128 * K + sub * 64 * K);
            }
            NotifyEvent<PIPE_MTE3>(6 + slot);
        }
        if (count >= 2) {
            CrossCoreWaitFlag(4 + (count - 2) % 2);
        }
        if (count >= 1) {
            CrossCoreWaitFlag(4 + (count - 1) % 2);
        }
    }
};

template <int BITS, int K, typename Meta>
__aicore__ inline void Run(GM_ADDR q, GM_ADDR s, GM_ADDR i, GM_ADDR j, GM_ADDR n, GM_ADDR g, GM_ADDR t, GM_ADDR l,
                           GM_ADDR o, GM_ADDR aw, GM_ADDR bw, GM_ADDR offsets, GM_ADDR tiling, uint32_t batch,
                           uint32_t kh, uint32_t pg, uint32_t len, uint32_t cores, bool prefill, uint32_t qheads,
                           uint32_t physicalBlocks)
{
    if ASCEND_IS_AIV {
        Decoder<BITS, K, Meta> d;
        d.Init(q, s, i, j, n, g, t, l, o, aw, bw, offsets, kh, pg, len, cores, qheads, physicalBlocks);
        if (prefill) {
            d.ProcessPrefill(batch);
        } else {
            d.Run(batch);
        }
    }
    if ASCEND_IS_AIC {
        TPipe pipe;
        Matmul<AType, BType, CType, CType, MM_TILING> mm;
        GlobalTensor<bfloat16_t> a, b;
        GlobalTensor<float> c;
        GlobalTensor<int32_t> table, lens;
        a.SetGlobalBuffer((__gm__ bfloat16_t*)aw);
        b.SetGlobalBuffer((__gm__ bfloat16_t*)bw);
        c.SetGlobalBuffer((__gm__ float*)o);
        table.SetGlobalBuffer((__gm__ int32_t*)t);
        lens.SetGlobalBuffer((__gm__ int32_t*)l);
        mm.Init((__gm__ const tiling::TCubeTiling*)tiling, &pipe);
        if (prefill) {
            uint32_t groups = qheads / kh, groupTiles = (groups + 3) / 4;
            uint32_t core = GetBlockIdx(), tiles = (len + 127) / 128, tasks = kh * groupTiles * tiles;
            uint32_t padded = (batch + TILE_M - 1) / TILE_M * TILE_M;
            uint32_t count = core < tasks ? (tasks - 1 - core) / cores + 1 : 0;
            for (uint32_t it = 0; it < count; ++it) {
                uint32_t task = core + it * cores;
                uint32_t slot = it % 2;
                uint32_t hg = task / tiles, head = hg / groupTiles, page = task % tiles;
                uint32_t qbase = head * groups + (hg % groupTiles) * 4;
                uint32_t active = groups - (hg % groupTiles) * 4;
                if (active > 4) {
                    active = 4;
                }
                uint32_t sl = lens.GetValue(0), start = page * 128;
                uint32_t valid = sl > start ? (sl - start < 128 ? sl - start : 128) : 0;
                int32_t physicalBlock = table.GetValue(page);
                bool validPhysicalBlock = physicalBlock >= 0 && static_cast<uint32_t>(physicalBlock) < physicalBlocks;
                CrossCoreWaitFlag(6 + slot);
                for (uint32_t h = 0; h < active && valid > 0 && validPhysicalBlock; ++h) {
                    for (uint32_t token = 0; token < batch; token += TILE_M) {
                        uint32_t rows = batch - token < TILE_M ? batch - token : TILE_M;
                        mm.SetOrgShape(TILE_M, qheads * len, K);
                        mm.SetSingleShape(rows, valid, K);
                        mm.SetTensorA(a[(uint64_t(qbase + h) * padded + token) * K]);
                        mm.SetTensorB(b[(uint64_t(core) * 2 + slot) * 128 * K], true);
                        mm.IterateAll(c[(uint64_t(token) * qheads + qbase + h) * len + page * 128]);
                        mm.End();
                    }
                }
                NotifyEvent<PIPE_FIX>(4 + slot);
            }
            return;
        }
        uint32_t groups = qheads / kh, groupTiles = (groups + 3) / 4;
        uint32_t core = GetBlockIdx(), tiles = (len + 127) / 128, tasks = batch * kh * groupTiles * tiles;
        uint32_t count = core < tasks ? (tasks - 1 - core) / cores + 1 : 0;
        for (uint32_t it = 0; it < count; ++it) {
            uint32_t slot = it % 2, task = core + it * cores, seq = task / (kh * groupTiles * tiles),
                     hg = (task / tiles) % (kh * groupTiles), page = task % tiles, start = page * 128;
            uint32_t qbase = (hg / groupTiles) * groups + (hg % groupTiles) * 4;
            uint32_t active = groups - (hg % groupTiles) * 4;
            if (active > 4) {
                active = 4;
            }
            CrossCoreWaitFlag(6 + slot);
            uint32_t sl = lens.GetValue(seq);
            int32_t physicalBlock = table.GetValue(seq * pg + page);
            bool validPhysicalBlock = physicalBlock >= 0 && static_cast<uint32_t>(physicalBlock) < physicalBlocks;
            if (start < sl && validPhysicalBlock) {
                uint64_t base = uint64_t(core) * 2 + slot;
                mm.SetOrgShape(4, len, K);
                mm.SetSingleShape(active, sl - start < 128 ? sl - start : 128, K);
                mm.SetTensorA(a[base * 4 * K]);
                mm.SetTensorB(b[base * 128 * K], true);
                mm.IterateAll(c[(uint64_t(seq) * qheads + qbase) * len + start]);
                mm.End();
            }
            NotifyEvent<PIPE_FIX>(4 + slot);
        }
    }
}
#endif // TURBO_QUANT_ATTENTION_SCORE_TQ_SCORE_COMPUTE_SCORE_CORE_H_
