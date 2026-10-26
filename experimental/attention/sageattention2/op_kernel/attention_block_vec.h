/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "attention_common.h"
#include "softmax.h"
#include "output_update.h"
namespace Sage {
template <class T, uint32_t DP, bool SMOOTH>
__simd_vf__ inline void LseVF(__ubuf__ float* out, __ubuf__ float* denom, __ubuf__ half* maxima, __ubuf__ T* query,
                              __ubuf__ float* mean, uint32_t rows, float scale)
{
    using namespace AscendC::MicroAPI;
    static_assert(DP == 64 || DP == 128, "LSE requires DP64 or DP128");
    static constexpr CastTrait toFloat{RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
    auto full = CreateMask<float, MaskPattern::ALL>();
    uint32_t remaining = rows;
    auto valid = UpdateMask<float>(remaining);
    RegTensor<float> denominator, maximum, result;
    RegTensor<half> maximumHalf;
    LoadAlign(denominator, denom);
    LoadAlign<half, LoadDist::DIST_UNPACK_B16>(maximumHalf, maxima);
    Cast<float, half, toFloat>(maximum, maximumHalf, valid);
    Ln(result, denominator, valid);
    Add(result, result, maximum, valid);
    if constexpr (SMOOTH) {
        RegTensor<float> meanLow, meanHigh;
        LoadAlign(meanLow, mean);
        if constexpr (DP == 128)
            LoadAlign(meanHigh, mean + 64);
        UnalignReg compact;
        __ubuf__ float* corrections = out;
        for (uint16_t row = 0; row < rows; ++row) {
            RegTensor<T> raw;
            RegTensor<float> low, high, correction;
            LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, query + row * DP);
            Cast<float, T, toFloat>(low, raw, full);
            Mul(low, low, meanLow, full);
            if constexpr (DP == 128) {
                LoadAlign<T, LoadDist::DIST_UNPACK_B16>(raw, query + row * DP + 64);
                Cast<float, T, toFloat>(high, raw, full);
                Mul(high, high, meanHigh, full);
                Add(low, low, high, full);
            }
            ReduceSum(correction, low, full);
            StoreUnAlign<float, PostLiteral::POST_MODE_UPDATE>(corrections, correction, compact, 1);
        }
        StoreUnAlignPost(corrections, compact, 0);
        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        RegTensor<float> correction;
        LoadAlign(correction, out);
        Muls(correction, correction, scale, valid);
        Add(result, result, correction, valid);
    }
    StoreAlign(out, result, valid);
}

struct DefaultVectorLayout {
    static constexpr uint32_t ACCUM_UB = Sage::ACCUM_UB;
    static constexpr uint32_t LSE_UB = Sage::LSE_UB;
    static constexpr uint32_t MU_UB = Sage::MU_UB;
    static constexpr uint32_t P_UB = Sage::P_UB;
    static constexpr uint32_t QS_UB = Sage::QS_UB;
    static constexpr uint32_t MAX_UB = Sage::MAX_UB;
    static constexpr uint32_t DEN_UB = Sage::DEN_UB;
    static constexpr uint32_t KS_UB = Sage::KS_UB;
    static constexpr uint32_t SCORE_UB = Sage::SCORE_UB;
    static constexpr uint32_t ALPHA_UB = Sage::ALPHA_UB;
    static constexpr uint32_t SUM_UB = Sage::SUM_UB;
    static constexpr uint32_t PV_UB = Sage::PV_UB;
    static constexpr uint32_t PV_SLOT_BYTES = Sage::PV_SLOT_BYTES;
};
// AIV stages own scale prefetch, softmax/P transport, output update and LSE.
// All methods inline into the mixed entry to preserve AIC/AIV specialization.
template <class T, uint32_t DP, class Layout = DefaultVectorLayout>
class AttentionBlockVec {
public:
    __aicore__ inline __attribute__((always_inline)) void Init(FixedResource& resource, GM_ADDR q, GM_ADDR qs,
                                                               GM_ADDR ks, GM_ADDR mean, GM_ADDR out, GM_ADDR lse,
                                                               SageParams params)
    {
        resource_ = &resource;
        params_ = params;
        subBlockIdx_ = GetSubBlockIdx();
        qsg_.SetGlobalBuffer((__gm__ half*)qs);
        ksg_.SetGlobalBuffer((__gm__ half*)ks);
        mg_.SetGlobalBuffer((__gm__ float*)mean);
        lg_.SetGlobalBuffer((__gm__ float*)lse);
        qg_.SetGlobalBuffer((__gm__ T*)q);
        og_.SetGlobalBuffer((__gm__ T*)out);

        accum_ = resource_->ubBuf.GetBufferByByte<float>(ACCUM_UB);
        lse_ = resource_->ubBuf.GetBufferByByte<float>(LSE_UB);
        mu_ = resource_->ubBuf.GetBufferByByte<float>(MU_UB);
        qo_ = resource_->ubBuf.GetBufferByByte<T>(P_UB);
        qsu_ = resource_->ubBuf.GetBufferByByte<half>(QS_UB);
        maxima_ = resource_->ubBuf.GetBufferByByte<half>(MAX_UB);
        denom_ = resource_->ubBuf.GetBufferByByte<float>(DEN_UB);
    }
    __aicore__ inline __attribute__((always_inline)) void BeginTile(uint32_t b, uint32_t kh, uint64_t qb,
                                                                    uint32_t qstart)
    {
        Sync<HardEvent::V_MTE2>();
        Duplicate(accum_, 0.0f, HALF * params_.dp);
        DataCopy(mu_, mg_[(uint64_t(b) * params_.hk + kh) * params_.dp], params_.dp);
        DataCopy(qsu_, qsg_[qb + qstart + subBlockIdx_ * HALF], HALF);
        Sync<HardEvent::MTE2_S>();
        Duplicate(maxima_, half(-65504.0f), 128);
        Duplicate(denom_, 0.0f, 64);
        PipeBarrier<PIPE_V>();
    }
    __aicore__ inline __attribute__((always_inline)) void PrefetchScales(uint64_t kb, uint32_t count)
    {
        // IDs 4/5 own the two immutable K-scale slots. Every prefetch
        // consumes a free token and publishes one MTE2->V ready token.
        SetFlag<HardEvent::V_MTE2>(SCALE_EVENT_BASE);
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(SCALE_EVENT_BASE + 1));
        for (uint32_t pre = 0; pre < Min(count, uint32_t(2)); ++pre) {
            event_t ev = static_cast<event_t>(SCALE_EVENT_BASE + pre);
            WaitFlag<HardEvent::V_MTE2>(ev);
            DataCopy(resource_->ubBuf.GetBufferByByte<half>(KS_UB + pre * N * 2), ksg_[kb + pre * N], N);
            SetFlag<HardEvent::MTE2_V>(ev);
        }
    }
    __aicore__ inline __attribute__((always_inline)) void Softmax(uint32_t step, uint32_t qstart, uint64_t kb,
                                                                  uint32_t count)
    {
        uint32_t start = step * N, parity = step % PIPE_SLOTS;
        auto scores = resource_->ubBuf.GetBufferByByte<half>(SCORE_UB + parity * HALF * N * 2);
        auto p8 = resource_->ubBuf.GetBufferByByte<float8_e4m3_t>(P_UB + parity * P_SLOT_BYTES);
        auto alpha = resource_->ubBuf.GetBufferByByte<float>(ALPHA_UB + (step % ALPHA_SLOTS) * 512);
        CrossCoreWaitFlag<4, PIPE_V>(QK_READY);
        auto ksu = resource_->ubBuf.GetBufferByByte<half>(KS_UB + (step & 1) * N * 2);
        event_t scaleEvent = static_cast<event_t>(SCALE_EVENT_BASE + (step & 1));
        WaitFlag<HardEvent::MTE2_V>(scaleEvent);
        if (qstart + subBlockIdx_ * HALF + HALF <= params_.sq && start + N <= params_.sk &&
            (!params_.causal || start + N <= qstart + subBlockIdx_ * HALF + 1)) {
            SoftmaxHalfVF<HALF, N, true, true, true, P_NZ_PAD_ROWS>(
                (__ubuf__ half*)scores.GetPhyAddr(), nullptr, nullptr, (__ubuf__ half*)maxima_.GetPhyAddr(),
                (__ubuf__ float*)denom_.GetPhyAddr(), (__ubuf__ float*)alpha.GetPhyAddr(),
                (__ubuf__ float8_e4m3_t*)p8.GetPhyAddr(), (__ubuf__ half*)qsu_.GetPhyAddr(),
                (__ubuf__ half*)ksu.GetPhyAddr(),
                (__ubuf__ float*)resource_->ubBuf.GetBufferByByte<float>(SUM_UB).GetPhyAddr(),
                qstart + subBlockIdx_ * HALF, params_.sq, start, params_.sk, params_.causal, params_.smScale);
        } else {
            SoftmaxHalfVF<HALF, N, false, true, true, P_NZ_PAD_ROWS>(
                (__ubuf__ half*)scores.GetPhyAddr(), nullptr, nullptr, (__ubuf__ half*)maxima_.GetPhyAddr(),
                (__ubuf__ float*)denom_.GetPhyAddr(), (__ubuf__ float*)alpha.GetPhyAddr(),
                (__ubuf__ float8_e4m3_t*)p8.GetPhyAddr(), (__ubuf__ half*)qsu_.GetPhyAddr(),
                (__ubuf__ half*)ksu.GetPhyAddr(),
                (__ubuf__ float*)resource_->ubBuf.GetBufferByByte<float>(SUM_UB).GetPhyAddr(),
                qstart + subBlockIdx_ * HALF, params_.sq, start, params_.sk, params_.causal, params_.smScale);
        }
        SetFlag<HardEvent::V_MTE2>(scaleEvent);
        if (step + 2 < count) {
            WaitFlag<HardEvent::V_MTE2>(scaleEvent);
            DataCopy(ksu, ksg_[kb + (step + 2) * N], N);
            SetFlag<HardEvent::MTE2_V>(scaleEvent);
        }
        Sync<HardEvent::V_MTE3>();
        auto pl1 = resource_->l1Buf.GetBufferByByte<float8_e4m3_t>(P_L1 + parity * M * N);
        DataCopyParams cp{N / 32, HALF, P_NZ_PAD_ROWS, HALF};
        DataCopy(pl1[subBlockIdx_ * HALF * 32], p8, cp);
        CrossCoreSetFlag<4, PIPE_MTE3>(P_READY);
    }
    __aicore__ inline __attribute__((always_inline)) void UpdateOutput(uint32_t index, uint32_t count)
    {
        if (index & 1) {
            // Both PV slots remain owned by this pair until both reads finish.
            CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
            CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
            auto alpha0 = resource_->ubBuf.GetBufferByByte<float>(ALPHA_UB + ((index - 1) % ALPHA_SLOTS) * 512);
            auto alpha1 = resource_->ubBuf.GetBufferByByte<float>(ALPHA_UB + (index % ALPHA_SLOTS) * 512);
            AccumulateFp32PairVF<DP>(
                (__ubuf__ float*)accum_.GetPhyAddr(),
                (__ubuf__ float*)resource_->ubBuf.GetBufferByByte<float>(PV_UB).GetPhyAddr(),
                (__ubuf__ float*)resource_->ubBuf.GetBufferByByte<float>(PV_UB + PV_SLOT_BYTES).GetPhyAddr(),
                (__ubuf__ float*)alpha0.GetPhyAddr(), (__ubuf__ float*)alpha1.GetPhyAddr());
            CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
            CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
        } else if (index + 1 == count) {
            CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
            auto alpha = resource_->ubBuf.GetBufferByByte<float>(ALPHA_UB + (index % ALPHA_SLOTS) * 512);
            AccumulateFp32VF<DP>((__ubuf__ float*)accum_.GetPhyAddr(),
                                 (__ubuf__ float*)resource_->ubBuf.GetBufferByByte<float>(PV_UB).GetPhyAddr(),
                                 (__ubuf__ float*)alpha.GetPhyAddr(), HALF);
            CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
        }
    }
    template <bool FusedCast = false, bool DrainScales = true>
    __aicore__ inline __attribute__((always_inline)) void FinalizeTile(uint32_t qstart, uint32_t b, uint32_t h)
    {
        if constexpr (DrainScales) {
            WaitFlag<HardEvent::V_MTE2>(SCALE_EVENT_BASE);
            WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(SCALE_EVENT_BASE + 1));
        }
        if constexpr (FusedCast) {
            NormalizeCastVF<T, DP>(
                (__ubuf__ T*)qo_.GetPhyAddr(), (__ubuf__ float*)accum_.GetPhyAddr(),
                (__ubuf__ float*)denom_.GetPhyAddr(),
                (__ubuf__ float*)resource_->ubBuf.template GetBufferByByte<float>(SUM_UB).GetPhyAddr());
        } else {
            NormalizeVF((__ubuf__ float*)accum_.GetPhyAddr(), (__ubuf__ float*)denom_.GetPhyAddr(), params_.dp, HALF);
            PipeBarrier<PIPE_V>();
            Cast(qo_, accum_, RoundMode::CAST_RINT, HALF * params_.dp);
        }
        StoreTileOutput(qstart, b, h);
    }
    __aicore__ inline __attribute__((always_inline)) void StoreTileOutput(uint32_t qstart, uint32_t b, uint32_t h)
    {
        PipeBarrier<PIPE_V>();
        uint32_t first = qstart + subBlockIdx_ * HALF;
        uint32_t validRows = first < params_.sq ? Min(HALF, params_.sq - first) : 0;
        if (validRows) {
            uint64_t offset = ((uint64_t(b) * params_.hq + h) * params_.sq + first) * params_.d;
            WriteRows(og_, offset, qo_, validRows, params_.d, params_.dp);
        }
        if (params_.returnLse)
            StoreTileLse(first, validRows, b, h);
    }

protected:
    __aicore__ inline __attribute__((always_inline)) void StoreTileLse(uint32_t first, uint32_t validRows, uint32_t b,
                                                                       uint32_t h)
    {
        if (!validRows)
            return;
        auto query = accum_.template ReinterpretCast<T>();
        auto output = (__ubuf__ float*)lse_.GetPhyAddr();
        auto denominator = (__ubuf__ float*)denom_.GetPhyAddr();
        auto maximum = (__ubuf__ half*)maxima_.GetPhyAddr();
        auto input = (__ubuf__ T*)query.GetPhyAddr();
        auto mean = (__ubuf__ float*)mu_.GetPhyAddr();
        if (params_.smoothK) {
            uint64_t offset = Offset(params_.qs, b, h, first);
            ReadRows(query, qg_, offset, validRows, params_.qs[2], params_.d, DP);
            LseVF<T, DP, true>(output, denominator, maximum, input, mean, validRows, params_.smScale);
        } else {
            LseVF<T, DP, false>(output, denominator, maximum, input, mean, validRows, params_.smScale);
        }
        uint64_t offset = (uint64_t(b) * params_.hq + h) * params_.sq + first;
        Sync<HardEvent::V_MTE3>();
        DataCopyExtParams copy{1, validRows * uint32_t(sizeof(float)), 0, 0, 0};
        DataCopyPad(lg_[offset], lse_, copy);
        Sync<HardEvent::MTE3_V>();
    }
    static constexpr uint32_t ACCUM_UB = Layout::ACCUM_UB;
    static constexpr uint32_t LSE_UB = Layout::LSE_UB;
    static constexpr uint32_t MU_UB = Layout::MU_UB;
    static constexpr uint32_t P_UB = Layout::P_UB;
    static constexpr uint32_t QS_UB = Layout::QS_UB;
    static constexpr uint32_t MAX_UB = Layout::MAX_UB;
    static constexpr uint32_t DEN_UB = Layout::DEN_UB;
    static constexpr uint32_t KS_UB = Layout::KS_UB;
    static constexpr uint32_t SCORE_UB = Layout::SCORE_UB;
    static constexpr uint32_t ALPHA_UB = Layout::ALPHA_UB;
    static constexpr uint32_t SUM_UB = Layout::SUM_UB;
    static constexpr uint32_t PV_UB = Layout::PV_UB;
    static constexpr uint32_t PV_SLOT_BYTES = Layout::PV_SLOT_BYTES;
    FixedResource* resource_;
    SageParams params_;
    uint32_t subBlockIdx_;
    GlobalTensor<half> qsg_, ksg_;
    GlobalTensor<float> mg_, lg_;
    GlobalTensor<T> qg_, og_;
    LocalTensor<float> accum_, lse_, mu_, denom_;
    LocalTensor<half> qsu_, maxima_;
    LocalTensor<T> qo_;
};

struct WideVectorLayout {
    static constexpr uint32_t SCORE_UB = 0, PV_UB = 4 * HALF * N * 2;
    static constexpr uint32_t P_PART_BYTES = HALF * N, PV_SLOT_BYTES = 2 * P_PART_BYTES,
                              ACCUM_UB = PV_UB + 2 * PV_SLOT_BYTES;
    static constexpr uint32_t P_UB = ACCUM_UB + HALF * 128 * 4;
    static constexpr uint32_t MU_UB = P_UB + P_SLOT_BYTES;
    static constexpr uint32_t KS_UB = MU_UB + 512, QS_UB = KS_UB + 2 * 512 * 2;
    static constexpr uint32_t MAX_UB = QS_UB + 512, DEN_UB = MAX_UB + 512, ALPHA_UB = DEN_UB + 512;
    static constexpr uint32_t SUM_UB = ALPHA_UB + 3 * 512;
    static constexpr uint32_t PARTIAL_SUMS_UB = P_UB, END = SUM_UB + HALF * 4;
    static constexpr uint32_t LSE_UB = P_UB;
    static_assert(PV_SLOT_BYTES >= HALF * 128 * 4, "PV slot cannot hold FP32 PV");
    static_assert(P_SLOT_BYTES >= HALF * 8 * 4, "output scratch cannot hold softmax partial sums");
    static_assert(END <= Arch::UB_SIZE, "wide softmax exceeds UB");
};

template <class T, uint32_t DP>
class AttentionBlockVecWide : public AttentionBlockVec<T, DP, WideVectorLayout> {
    using Base = AttentionBlockVec<T, DP, WideVectorLayout>;

public:
    __aicore__ inline __attribute__((always_inline)) void PrefetchWideScales(uint64_t kb, uint32_t count)
    {
        SetFlag<HardEvent::V_MTE2>(SCALE_EVENT_BASE);
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(SCALE_EVENT_BASE + 1));
        for (uint32_t pre = 0; pre < Min(count, uint32_t(2)); ++pre) {
            event_t ev = static_cast<event_t>(SCALE_EVENT_BASE + pre);
            WaitFlag<HardEvent::V_MTE2>(ev);
            DataCopy(resource_->ubBuf.template GetBufferByByte<half>(KS_UB + pre * 512 * 2), ksg_[kb + pre * 512], 512);
            SetFlag<HardEvent::MTE2_V>(ev);
        }
    }
    template <bool CAUSAL = false, bool STREAM = false>
    __aicore__ inline __attribute__((always_inline)) void SoftmaxWide(uint32_t step, uint64_t kb, uint32_t count,
                                                                      uint32_t qstart = 0, uint32_t group = 0,
                                                                      uint64_t nextScale = ~uint64_t(0))
    {
        const uint32_t ring = STREAM ? group : step;
        auto scores = resource_->ubBuf.template GetBufferByByte<half>(SCORE_UB + (ring & 1) * 2 * HALF * N * 2);
        // Each FP8 NZ block replaces the consumed first FP16 stripe of its
        // 32-column score block. Keep the second stripe as a 2048-byte gap.
        auto p8 = scores.template ReinterpretCast<float8_e4m3_t>();
        auto alpha = resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + (ring % 3) * 512);
        CrossCoreWaitFlag<4, PIPE_V>(QK_READY);
        CrossCoreWaitFlag<4, PIPE_V>(QK_READY);
        // P_READY for fragment f follows its MTE3 copy. Cube waits for this
        // token before PV(f)'s MTE1->M, which precedes QK(f+4)'s M->FIX.
        // Thus score reuse cannot overwrite P before the copy completes.
        auto ksu = resource_->ubBuf.template GetBufferByByte<half>(KS_UB + (ring & 1) * 512 * 2);
        event_t ev = static_cast<event_t>(SCALE_EVENT_BASE + (ring & 1));
        WaitFlag<HardEvent::MTE2_V>(ev);
        auto scoresNext =
            resource_->ubBuf.template GetBufferByByte<half>(SCORE_UB + ((ring & 1) * 2 + 1) * HALF * N * 2);
        auto scratch = resource_->ubBuf.template GetBufferByByte<float>(SUM_UB);
        // Output scratch is idle until the complete softmax group finishes.
        auto partialSums = resource_->ubBuf.template GetBufferByByte<float>(WideVectorLayout::PARTIAL_SUMS_UB);
        if (CAUSAL && step * 512 + 512 > qstart + subBlockIdx_ * HALF + 1) {
            SoftmaxWideFirstVF<true>((__ubuf__ half*)scores.GetPhyAddr(), (__ubuf__ half*)scoresNext.GetPhyAddr(),
                                     (__ubuf__ half*)maxima_.GetPhyAddr(), (__ubuf__ float*)denom_.GetPhyAddr(),
                                     (__ubuf__ float*)alpha.GetPhyAddr(), (__ubuf__ float8_e4m3_t*)p8.GetPhyAddr(),
                                     (__ubuf__ half*)qsu_.GetPhyAddr(), (__ubuf__ half*)ksu.GetPhyAddr(),
                                     (__ubuf__ float*)scratch.GetPhyAddr(), (__ubuf__ float*)partialSums.GetPhyAddr(),
                                     int32_t(qstart + subBlockIdx_ * HALF) - int32_t(step * 512));
        } else {
            SoftmaxWideFirstVF((__ubuf__ half*)scores.GetPhyAddr(), (__ubuf__ half*)scoresNext.GetPhyAddr(),
                               (__ubuf__ half*)maxima_.GetPhyAddr(), (__ubuf__ float*)denom_.GetPhyAddr(),
                               (__ubuf__ float*)alpha.GetPhyAddr(), (__ubuf__ float8_e4m3_t*)p8.GetPhyAddr(),
                               (__ubuf__ half*)qsu_.GetPhyAddr(), (__ubuf__ half*)ksu.GetPhyAddr(),
                               (__ubuf__ float*)scratch.GetPhyAddr(), (__ubuf__ float*)partialSums.GetPhyAddr());
        }
        SetFlag<HardEvent::V_MTE2>(ev);
        if constexpr (STREAM) {
            if (nextScale != ~uint64_t(0)) {
                WaitFlag<HardEvent::V_MTE2>(ev);
                DataCopy(ksu, ksg_[nextScale], 512);
                SetFlag<HardEvent::MTE2_V>(ev);
            }
        } else if (step + 2 < count) {
            WaitFlag<HardEvent::V_MTE2>(ev);
            DataCopy(ksu, ksg_[kb + (step + 2) * 512], 512);
            SetFlag<HardEvent::MTE2_V>(ev);
        }
        Sync<HardEvent::V_MTE3>();
        auto pl1 = resource_->l1Buf.template GetBufferByByte<float8_e4m3_t>(P_L1 + (ring & 1) * 2 * M * N);
        DataCopyParams cp{N / 32, HALF, HALF, HALF};
        DataCopy(pl1[subBlockIdx_ * HALF * 32], p8, cp);
        CrossCoreSetFlag<4, PIPE_MTE3>(P_READY);
        auto p8Next = scoresNext.template ReinterpretCast<float8_e4m3_t>();
        SoftmaxWideSecondVF((__ubuf__ half*)scoresNext.GetPhyAddr(), (__ubuf__ half*)maxima_.GetPhyAddr(),
                            (__ubuf__ float*)denom_.GetPhyAddr(), (__ubuf__ float*)alpha.GetPhyAddr(),
                            (__ubuf__ float8_e4m3_t*)p8Next.GetPhyAddr(), (__ubuf__ float*)scratch.GetPhyAddr(),
                            (__ubuf__ float*)partialSums.GetPhyAddr());
        Sync<HardEvent::V_MTE3>();
        DataCopy(pl1[M * N + subBlockIdx_ * HALF * 32], p8Next, cp);
        CrossCoreSetFlag<4, PIPE_MTE3>(P_READY);
    }
    __aicore__ inline __attribute__((always_inline)) void UpdateWideOutput(uint32_t step)
    {
        CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
        AccumulateWideFp32VF<DP>(
            (__ubuf__ float*)accum_.GetPhyAddr(),
            (__ubuf__ float*)resource_->ubBuf.template GetBufferByByte<float>(PV_UB + (step & 1) * PV_SLOT_BYTES)
                .GetPhyAddr(),
            (__ubuf__ float*)resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + (step % 3) * 512).GetPhyAddr(),
            (__ubuf__ float*)denom_.GetPhyAddr());
        CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
    }

    __aicore__ inline __attribute__((always_inline)) void BeginStream(uint64_t firstScale, uint64_t secondScale)
    {
        SetFlag<HardEvent::V_MTE2>(SCALE_EVENT_BASE);
        SetFlag<HardEvent::V_MTE2>(static_cast<event_t>(SCALE_EVENT_BASE + 1));
        for (uint32_t slot = 0; slot < 2; ++slot) {
            uint64_t offset = slot ? secondScale : firstScale;
            if (offset != ~uint64_t(0)) {
                event_t ev = static_cast<event_t>(SCALE_EVENT_BASE + slot);
                WaitFlag<HardEvent::V_MTE2>(ev);
                DataCopy(resource_->ubBuf.template GetBufferByByte<half>(KS_UB + slot * 1024), ksg_[offset], 512);
                SetFlag<HardEvent::MTE2_V>(ev);
            }
        }
    }
    __aicore__ inline __attribute__((always_inline)) void SelectStreamQuery(uint32_t ordinal, uint64_t qb,
                                                                            uint32_t qstart, bool first)
    {
        // MAX_UB already reserves 512 bytes: two independent 256-byte slots.
        maxima_ = resource_->ubBuf.template GetBufferByByte<half>(Base::MAX_UB + (ordinal & 1) * 256);
        if (first) {
            Sync<HardEvent::V_MTE2>();
            DataCopy(qsu_, Base::qsg_[qb + qstart + subBlockIdx_ * HALF], HALF);
            Sync<HardEvent::MTE2_V>();
            Duplicate(maxima_, half(-65504.0f), 128);
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline __attribute__((always_inline)) void LoadLseMean(uint32_t b, uint32_t kh)
    {
        if (!Base::params_.returnLse || !Base::params_.smoothK)
            return;
        Sync<HardEvent::V_MTE2>();
        DataCopy(Base::mu_, Base::mg_[(uint64_t(b) * Base::params_.hk + kh) * DP], DP);
        Sync<HardEvent::MTE2_V>();
    }
    __aicore__ inline __attribute__((always_inline)) void UpdateStreamOutput(uint32_t group, bool first, bool last)
    {
        CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
        auto pv = resource_->ubBuf.template GetBufferByByte<float>(PV_UB + (group & 1) * PV_SLOT_BYTES);
        auto alpha = resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + (group % 3) * 512);
        if (last) {
            auto scratch = resource_->ubBuf.template GetBufferByByte<float>(SUM_UB);
            if (first)
                FinalizeWideFp32VF<T, DP, true>(
                    (__ubuf__ T*)Base::qo_.GetPhyAddr(), (__ubuf__ float*)accum_.GetPhyAddr(),
                    (__ubuf__ float*)pv.GetPhyAddr(), (__ubuf__ float*)alpha.GetPhyAddr(),
                    (__ubuf__ float*)denom_.GetPhyAddr(), (__ubuf__ float*)scratch.GetPhyAddr());
            else
                FinalizeWideFp32VF<T, DP, false>(
                    (__ubuf__ T*)Base::qo_.GetPhyAddr(), (__ubuf__ float*)accum_.GetPhyAddr(),
                    (__ubuf__ float*)pv.GetPhyAddr(), (__ubuf__ float*)alpha.GetPhyAddr(),
                    (__ubuf__ float*)denom_.GetPhyAddr(), (__ubuf__ float*)scratch.GetPhyAddr());
        } else if (first)
            InitializeWideFp32VF<DP>((__ubuf__ float*)accum_.GetPhyAddr(), (__ubuf__ float*)pv.GetPhyAddr(),
                                     (__ubuf__ float*)alpha.GetPhyAddr(), (__ubuf__ float*)denom_.GetPhyAddr());
        else
            AccumulateWideFp32VF<DP>((__ubuf__ float*)accum_.GetPhyAddr(), (__ubuf__ float*)pv.GetPhyAddr(),
                                     (__ubuf__ float*)alpha.GetPhyAddr(), (__ubuf__ float*)denom_.GetPhyAddr());
        CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
    }
    __aicore__ inline __attribute__((always_inline)) void AcquireStreamPv()
    {
        CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
    }
    template <bool FIRST, bool LAST, bool FIRST_READY>
    __aicore__ inline __attribute__((always_inline)) void UpdateStreamPairImpl(uint64_t group)
    {
        if constexpr (!FIRST_READY)
            CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
        CrossCoreWaitFlag<4, PIPE_V>(PV_READY);
        auto pv0 = resource_->ubBuf.template GetBufferByByte<float>(PV_UB + (group & 1) * PV_SLOT_BYTES);
        auto pv1 = resource_->ubBuf.template GetBufferByByte<float>(PV_UB + ((group + 1) & 1) * PV_SLOT_BYTES);
        auto a0 = resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + (group % 3) * 512);
        auto a1 = resource_->ubBuf.template GetBufferByByte<float>(ALPHA_UB + ((group + 1) % 3) * 512);
        auto scratch = resource_->ubBuf.template GetBufferByByte<float>(SUM_UB);
        UpdateWidePairVF<T, DP, FIRST, LAST>((__ubuf__ T*)Base::qo_.GetPhyAddr(), (__ubuf__ float*)accum_.GetPhyAddr(),
                                             (__ubuf__ float*)pv0.GetPhyAddr(), (__ubuf__ float*)pv1.GetPhyAddr(),
                                             (__ubuf__ float*)a0.GetPhyAddr(), (__ubuf__ float*)a1.GetPhyAddr(),
                                             (__ubuf__ float*)denom_.GetPhyAddr(),
                                             (__ubuf__ float*)scratch.GetPhyAddr());
        CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
        CrossCoreSetFlag<4, PIPE_V>(PV_RELEASE);
    }
    template <bool FIRST_READY = false>
    __aicore__ inline __attribute__((always_inline)) void UpdateStreamPair(uint64_t group, bool first, bool last)
    {
        if (first) {
            if (last)
                UpdateStreamPairImpl<true, true, FIRST_READY>(group);
            else
                UpdateStreamPairImpl<true, false, FIRST_READY>(group);
        } else {
            if (last)
                UpdateStreamPairImpl<false, true, FIRST_READY>(group);
            else
                UpdateStreamPairImpl<false, false, FIRST_READY>(group);
        }
    }
    __aicore__ inline __attribute__((always_inline)) void FinishStreamQuery(uint32_t ordinal, uint32_t qstart,
                                                                            uint32_t b, uint32_t h)
    {
        maxima_ = resource_->ubBuf.template GetBufferByByte<half>(Base::MAX_UB + (ordinal & 1) * 256);
        Base::StoreTileOutput(qstart, b, h);
    }
    __aicore__ inline __attribute__((always_inline)) void EndStream()
    {
        WaitFlag<HardEvent::V_MTE2>(SCALE_EVENT_BASE);
        WaitFlag<HardEvent::V_MTE2>(static_cast<event_t>(SCALE_EVENT_BASE + 1));
    }

private:
    using Base::resource_;
    using Base::ksg_;
    using Base::maxima_;
    using Base::denom_;
    using Base::qsu_;
    using Base::subBlockIdx_;
    using Base::accum_;
    using Base::KS_UB;
    using Base::SCORE_UB;
    using Base::PV_UB;
    using Base::PV_SLOT_BYTES;
    using Base::ALPHA_UB;
    using Base::SUM_UB;
};
} // namespace Sage
