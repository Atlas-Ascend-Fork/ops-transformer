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
 * \file compressor_grad_gemm.h
 * \brief CompressorGrad arch22 AIC (cube) kernel：Phase2 gemm
 *   d_x(T,H)  = dn_kv @ wkv + dn_sc @ wgate
 *   d_wkv(C,H) = dn_kv^T @ x；d_wgate 同理
 *
 * B 操作数按自然 (K,N) tile 经 GM->L1 Nd2Nz + load3dv2 加载；wkv/wgate/x 由
 * cast kernel 以 blocked-NZ 落盘，tile (kb,nb) 位于 nb*Tiles+kb，单次连续 32KB 搬运。
 * dw 的 A 操作数（dn^T）用 LoadDataWithTranspose。
 * dn：fp16 HiLo 双平面（每 k-tile 两次 Mmad），bf16 单平面。
 */

#ifndef COMPRESSOR_GRAD_GEMM_H
#define COMPRESSOR_GRAD_GEMM_H

#include "kernel_operator.h"
#include "compressor_grad_common_arch22.h"
#include "compressor_grad_tiling_data.h"

using namespace AscendC;

// dn arrives blocked-NZ pre-tiled from the scatter kernel: every A tile is
// one contiguous 32KB DataCopy whose L1 image is byte-identical to the
// Nd2Nz(nValue=128, dValue=128, dstNzC0Stride=128, dstNzNStride=1) product.

// L1(Nz) -> L0A(Zz): A tile (m, k), non-transposed (A = src)
template <typename T>
__aicore__ inline void LoadNzL1ToZzL0A(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t m, uint32_t k,
                                       uint32_t colC0Stride)
{
    LoadData3DParamsV2<T> params;
    params.l1H = 1;
    params.l1W = colC0Stride;
    params.channelSize = k;
    params.kExtension = k;
    params.mExtension = m;
    params.strideH = 1;
    params.strideW = 1;
    params.filterH = 1;
    params.filterW = 1;
    params.dilationFilterH = 1;
    params.dilationFilterW = 1;
    LoadData(dst, src, params);
}

// L1(Nz of matrix (k, m)) -> L0A(Zz of (m, k)) via per-fractal 16x16 transpose
// (b16 types): fractal content transposes; fractal positions stay, because the
// L1 NZ fractal id (m-blk-major: mblk*(k/16)+kblk) equals the L0A ZZ fractal id.
// src NZ layout: m-direction C0(16) blocks; block j holds (k, 16) row-major at j*k*16.
// dst ZZ layout: k-direction C0(16) blocks;  block f holds (m, 16) row-major at f*m*16.
template <typename T>
__aicore__ inline void LoadNzL1ToZzL0ATranspose(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t m,
                                                uint32_t k)
{
    constexpr uint32_t FRAC = 16;
    LoadData2dTransposeParams params;
    params.startIndex = 0;
    params.repeatTimes = k / FRAC;
    params.srcStride = 1;
    params.dstGap = 0;
    params.dstFracGap = 0;
    params.addrMode = false;
    for (uint32_t j = 0; j < m / FRAC; j++) {
        LoadDataWithTranspose(dst[j * k * FRAC], src[j * k * FRAC], params);
    }
}

// 自然 (K,N) B tile 的 L1 NZ -> L0B ZN：load3dv2 + fmatrix
// （LoadDataToL0, isBTranspose=false），channelSize=CeilAlign(N,16)，
// extConfig=(Koff<<48)|(Noff<<32)|(Align(K,16)<<16)|Align(N,16)；
// 前置一次 SetFmatrix(1, Align(K,16), pads, FMATRIX_RIGHT)。
// 此处 bk/bn 编译期 128，已 16 对齐，无 padding。
template <typename T>
__aicore__ inline void LoadNzL1ToZnL0BNatural(const LocalTensor<T>& dst, const LocalTensor<T>& src, uint32_t bk,
                                              uint32_t bn)
{
    LoadData3DParamsV2Pro params;
    params.channelSize = bn;
    params.extConfig = (static_cast<uint64_t>(bk) << 16) | bn;
    params.fMatrixCtrl = true;
    LoadData<T>(dst, src, params);
}

// Fixpipe on-the-fly cast mode for L0C(fp32) -> GM(T)
template <typename T>
__aicore__ inline QuantMode_t FixQuantPre()
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return QuantMode_t::F322BF16;
    } else {
        return QuantMode_t::F322F16;
    }
}

constexpr uint32_t CG_GEMM_BM = 128;
constexpr uint32_t CG_GEMM_BN = 128;
constexpr uint32_t CG_GEMM_BK = 128;

// L1 A1/B1 queue depths: one full k-step tile set in flight (dx needs
// 4 A / 4 B, dw needs 4 A / 1 B) plus lookahead into the next step.
constexpr uint32_t CG_L1_A_DEPTH = 4;
constexpr uint32_t CG_L1_B_DEPTH = 4;

template <typename ioT>
class CompressorGradGemmKernel {
public:
    __aicore__ inline CompressorGradGemmKernel() {}

    __aicore__ inline void Init(GM_ADDR dX, GM_ADDR dWkv, GM_ADDR dWgate, GM_ADDR userWs,
                                const optiling::CompressorGradTilingDataA3* tilingData, TPipe* pipe)
    {
        xF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->xF16WsOff));
        wkvF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->wkvF16WsOff));
        wgateF16GM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->wgateF16WsOff));
        dnKvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnKvWsOff));
        dnScGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(userWs + tilingData->dnScWsOff));
        dXGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(dX));
        dWkvGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(dWkv));
        dWgateGM_.SetGlobalBuffer(reinterpret_cast<__gm__ ioT*>(dWgate));
        tLen_ = static_cast<uint32_t>(tilingData->tLen);
        hSize_ = static_cast<uint32_t>(tilingData->hiddenSize);
        cSize_ = static_cast<uint32_t>(tilingData->totalHeadDim);
        tPad_ = static_cast<uint32_t>(tilingData->tPad);
        gDx_ = static_cast<uint32_t>(tilingData->gemmDxCores);
        // dn is stored in blocked-NZ tiles at C-strip-major index
        // cTile*dnTilesT_ + tTile. fp16 keeps the HiLo dual planes
        // (dn = hi + lo, 64KB [hi tile | lo tile] units); bf16 is
        // single-plane (one 32KB hi tile per unit).
        hilo_ = std::is_same_v<ioT, half>;
        // dnPlanes（host 侧真值）必须与模板推导的平面数一致（fp16 HiLo = 2，
        // bf16 = 1），否则 dnUnit 平面步长与 scatter 写侧布局脱节会错读 GM。
        assert(tilingData->dnPlanes == (hilo_ ? 2 : 1), "dnPlanes != dtype-derived plane count");
        dnTilesT_ = CeilDivU32(tPad_, CG_GEMM_BK); // T-tiles per C-strip
        pipe->InitBuffer(queA1_, CG_L1_A_DEPTH, CG_GEMM_BM * CG_GEMM_BK * sizeof(half));
        pipe->InitBuffer(queB1_, CG_L1_B_DEPTH, CG_GEMM_BK * CG_GEMM_BN * sizeof(half));
        // L0 double buffering: 2x32KB fills L0A/L0B (64KB each), 2x64KB fills
        // L0C (128KB); the pair unit keeps both L0C slots alive (cA/cB resp.
        // cW/cG) across the k-loop.
        pipe->InitBuffer(queL0A_, 2, CG_GEMM_BM * CG_GEMM_BK * sizeof(half));
        pipe->InitBuffer(queL0B_, 2, CG_GEMM_BK * CG_GEMM_BN * sizeof(half));
        pipe->InitBuffer(queL0C_, 2, CG_GEMM_BM * CG_GEMM_BN * sizeof(float));
        // One-time fmatrix setup for the natural-B load3dv2 path (official
        // Prepare semantic; constant params: K=N=128, zero padding).
        const uint8_t padList[4] = {0, 0, 0, 0};
        SetFmatrix(1, CG_GEMM_BK, padList, FmatrixMode::FMATRIX_RIGHT);
    }

    __aicore__ inline void Process()
    {
        // KERNEL_TYPE_MIX_AIC_1_2: blockDim counts AIC cores; on the AIC side
        // GetBlockIdx()/GetBlockNum() already index the AIC space.
        uint32_t cid = static_cast<uint32_t>(GetBlockIdx());
        uint32_t aicTotal = static_cast<uint32_t>(GetBlockNum());
        if (cid < gDx_) {
            ProcessDx(cid);
        } else if (cid < aicTotal) {
            ProcessDw(cid - gDx_, aicTotal - gDx_);
        }
    }

private:
    // Contiguous balanced partition of `total` units over `n` indices:
    // index i takes [start, start+count) with counts differing by <= 1.
    __aicore__ inline void BalancedRange(uint32_t idx, uint32_t n, uint32_t total, uint32_t& start,
                                         uint32_t& count) const
    {
        uint32_t base = total / n;
        uint32_t rem = total % n;
        start = idx * base + MinU32(idx, rem);
        count = base + (idx < rem ? 1U : 0U);
    }

    // Issue the L1->L0A ZZ copy (non-transposed) of the head of queA1_ and
    // free the L1 slot. The L0A tensor is queued but NOT dequeued, so
    // split-stage callers (DxTerm) can keep issuing further MTE1 copies
    // before taking the MTE1_M wait at the consume side.
    __aicore__ inline void EnQueAToL0A()
    {
        LocalTensor<half> aL1 = queA1_.DeQue<half>();
        LocalTensor<half> aL0 = queL0A_.AllocTensor<half>();
        LoadNzL1ToZzL0A(aL0, aL1, CG_GEMM_BM, CG_GEMM_BK, CG_GEMM_BM);
        queL0A_.EnQue(aL0);
        queA1_.FreeTensor(aL1);
    }

    // DeQue the head of queA1_ and copy it to an L0A ZZ tile (non-transposed);
    // the L1 slot is freed once the MTE1 copy is queued, and the L0A tensor is
    // handed back already waited (MTE1_M) for immediate-Mmad callers.
    __aicore__ inline LocalTensor<half> DeQueAToL0A()
    {
        EnQueAToL0A();
        return queL0A_.DeQue<half>();
    }

    // Same for the transposed x tile (NZ of (K,M)) -> L0A ZZ of (M,K).
    __aicore__ inline LocalTensor<half> DeQueAToL0ATranspose()
    {
        LocalTensor<half> aL1 = queA1_.DeQue<half>();
        LocalTensor<half> aL0 = queL0A_.AllocTensor<half>();
        LoadNzL1ToZzL0ATranspose(aL0, aL1, CG_GEMM_BM, CG_GEMM_BK);
        queL0A_.EnQue(aL0);
        queA1_.FreeTensor(aL1);
        return queL0A_.DeQue<half>();
    }

    // DeQue the head of queB1_ and copy it to an L0B ZN tile via the
    // natural-(K,N) load3dv2+fmatrix path (dx w tiles and dw x tiles).
    __aicore__ inline LocalTensor<half> DeQueBToL0Natural()
    {
        LocalTensor<half> bL1 = queB1_.DeQue<half>();
        LocalTensor<half> bL0 = queL0B_.AllocTensor<half>();
        LoadNzL1ToZnL0BNatural(bL0, bL1, CG_GEMM_BK, CG_GEMM_BN);
        queL0B_.EnQue(bL0);
        queB1_.FreeTensor(bL1);
        return queL0B_.DeQue<half>();
    }

    __aicore__ inline void DoMmad(LocalTensor<float>& cL0, const LocalTensor<half>& aL0, const LocalTensor<half>& bL0,
                                  bool init)
    {
        MmadParams mp;
        mp.m = CG_GEMM_BM;
        mp.n = CG_GEMM_BN;
        mp.k = CG_GEMM_BK;
        mp.cmatrixInitVal = init;
        mp.cmatrixSource = false;
        // DeQue on queL0A_/queL0B_ already waits MTE1_M; and per the official
        // Mmad doc, back-to-back K-accumulation into the same L0C tile needs
        // no PipeBarrier<PIPE_M> when (m/16)*(n/16) >= 10 (here 8*8=64) —
        // the cube scheduler handles the L0C write-read dependency.
        Mmad(cL0, aL0, bL0, mp);
    }

    __aicore__ inline void FixOut(const GlobalTensor<ioT>& dst, uint64_t dstOff, LocalTensor<float>& cL0,
                                  uint32_t mValid, uint32_t rowStride)
    {
        SetWaitFlag<HardEvent::M_FIX>();
        FixpipeParamsV220 fp;
        fp.mSize = mValid;
        fp.nSize = CG_GEMM_BN;
        fp.srcStride = CG_GEMM_BM;
        fp.dstStride = rowStride;
        fp.ndNum = 1;
        fp.srcNdStride = 0;
        fp.dstNdStride = 0;
        fp.quantPre = FixQuantPre<ioT>();
        Fixpipe(dst[dstOff], cL0, fp);
        SetWaitFlag<HardEvent::FIX_MTE2>();
    }

    // 一个 dx GEMM term（kv 或 sc）的一个 k-step，split-stage 形式：
    // stage 1 连续下发的 L1->L0 拷贝（A,B0,B1 在 MTE1 上流水）；
    // stage 2 在 consume 侧等 MTE1_M 后做 Mmad。H 512 对齐（host 校验），
    // nH 恒偶，每个 pair unit 都含两个 N tile。
    __aicore__ inline void DxTerm(LocalTensor<float>& cA, LocalTensor<float>& cB, bool init)
    {
        // ---- stage 1: L1 -> L0 issue
        EnQueAToL0A();
        LocalTensor<half> b0L1 = queB1_.DeQue<half>();
        LocalTensor<half> b0L0 = queL0B_.AllocTensor<half>();
        LoadNzL1ToZnL0BNatural(b0L0, b0L1, CG_GEMM_BK, CG_GEMM_BN);
        queL0B_.EnQue(b0L0);
        queB1_.FreeTensor(b0L1);
        LocalTensor<half> b1L1 = queB1_.DeQue<half>();
        LocalTensor<half> b1L0 = queL0B_.AllocTensor<half>();
        LoadNzL1ToZnL0BNatural(b1L0, b1L1, CG_GEMM_BK, CG_GEMM_BN);
        queL0B_.EnQue(b1L0);
        queB1_.FreeTensor(b1L1);
        // ---- stage 2: wait + Mmad (kv hi[, lo] order unchanged)
        LocalTensor<half> aL0d = queL0A_.DeQue<half>();
        LocalTensor<half> b0L0d = queL0B_.DeQue<half>();
        LocalTensor<half> b1L0d = queL0B_.DeQue<half>();
        DoMmad(cA, aL0d, b0L0d, init);
        DoMmad(cB, aL0d, b1L0d, init);
        queL0A_.FreeTensor(aL0d);
        if (hilo_) {
            LocalTensor<half> aL0LoD = DeQueAToL0A();
            DoMmad(cA, aL0LoD, b0L0d, false);
            DoMmad(cB, aL0LoD, b1L0d, false);
            queL0A_.FreeTensor(aL0LoD);
        }
        queL0B_.FreeTensor(b0L0d);
        queL0B_.FreeTensor(b1L0d);
    }

    // Issue the GM->L1 loads of one dx term (kv or sc) of k-step k.
    // Per-queue stream order per step: A = kv hi[,lo], sc hi[,lo];
    // B = b0w, b1w, b0g, b1g — the DeQue side (DxTerm) consumes kv then sc.
    // Both pair tiles always load: H is 512-aligned (host check) so nH is
    // even and every unit carries both N tiles.
    __aicore__ inline void DxEnqueueTerm(uint32_t m0, uint32_t n0, uint32_t k, bool isSc, uint32_t cTiles)
    {
        const GlobalTensor<half>& dn = isSc ? dnScGM_ : dnKvGM_;
        // dn is blocked-NZ from the scatter kernel: tile (m0/128, k) is
        // one contiguous 32KB burst at C-strip-major index
        // k*dnTilesT_ + m0/128, with the lo tile 32KB behind the hi tile
        // (fp16 dual-plane units; bf16 units are the hi tile alone).
        const uint64_t dnUnit = static_cast<uint64_t>(hilo_ ? 2U : 1U) * CG_GEMM_BM * CG_GEMM_BK;
        uint64_t aUnit = (static_cast<uint64_t>(k) * dnTilesT_ + m0 / CG_GEMM_BM) * dnUnit;
        LocalTensor<half> a = queA1_.AllocTensor<half>();
        DataCopy(a, dn[aUnit], CG_GEMM_BM * CG_GEMM_BK);
        queA1_.EnQue(a);
        if (hilo_) {
            LocalTensor<half> aLo = queA1_.AllocTensor<half>();
            DataCopy(aLo, dn[aUnit + CG_GEMM_BM * CG_GEMM_BK], CG_GEMM_BM * CG_GEMM_BK);
            queA1_.EnQue(aLo);
        }
        // wkv/wgate are blocked-NZ from the cast kernel: tile
        // (k, n0/128) is one contiguous 32KB burst at strip-major index
        // nb*cTiles + k (cTiles = C/128).
        const GlobalTensor<half>& w = isSc ? wgateF16GM_ : wkvF16GM_;
        uint64_t b0Tile = (static_cast<uint64_t>(n0) / CG_GEMM_BN) * cTiles + k;
        LocalTensor<half> b0w = queB1_.AllocTensor<half>();
        DataCopy(b0w, w[b0Tile * (CG_GEMM_BK * CG_GEMM_BN)], CG_GEMM_BK * CG_GEMM_BN);
        queB1_.EnQue(b0w);
        LocalTensor<half> b1w = queB1_.AllocTensor<half>();
        DataCopy(b1w, w[(b0Tile + cTiles) * (CG_GEMM_BK * CG_GEMM_BN)], CG_GEMM_BK * CG_GEMM_BN);
        queB1_.EnQue(b1w);
    }

    // branch B unit：d_wkv / d_wgate 的 (m0,n0) tile
    //   d_w*(C,H) = dn_*(T,C)^T @ x(T,H)，M=C, N=H, K=T_pad，FixOut 直写最终输出。
    // x B tile（blocked-NZ 32KB 突发）由每个 k-step 的 cW/cG 两个 Mmad 共用；
    // dn A tile（转置 L0A 加载）按 term 流式加载。
    __aicore__ inline void DwPair(uint32_t m0, uint32_t n0)
    {
        uint32_t tTiles = CeilDivU32(tPad_, CG_GEMM_BK);
        LocalTensor<float> cW = queL0C_.AllocTensor<float>();
        LocalTensor<float> cG = queL0C_.AllocTensor<float>();
        for (uint32_t k = 0; k < tTiles; k++) {
            bool init = (k == 0);
            // ---- GM -> L1: dnKv/dnSc A tiles (blocked-NZ (T,C) tiles at
            // (k,m0/128), transposed on the way to L0A), then shared x B
            // tile. k is the inner loop and the tile index is
            // (m0/128)*tTiles + k, so the dw A stream walks consecutive
            // hi+lo units (64KB fp16 / 32KB bf16) within one C-strip.
            const uint64_t dnUnit = static_cast<uint64_t>(hilo_ ? 2U : 1U) * CG_GEMM_BM * CG_GEMM_BK;
            uint64_t aUnit = (static_cast<uint64_t>(m0) / CG_GEMM_BM * tTiles + k) * dnUnit;
            LocalTensor<half> aW = queA1_.AllocTensor<half>();
            DataCopy(aW, dnKvGM_[aUnit], CG_GEMM_BK * CG_GEMM_BM);
            queA1_.EnQue(aW);
            if (hilo_) {
                LocalTensor<half> aWLo = queA1_.AllocTensor<half>();
                DataCopy(aWLo, dnKvGM_[aUnit + CG_GEMM_BK * CG_GEMM_BM], CG_GEMM_BK * CG_GEMM_BM);
                queA1_.EnQue(aWLo);
            }
            LocalTensor<half> aG = queA1_.AllocTensor<half>();
            DataCopy(aG, dnScGM_[aUnit], CG_GEMM_BK * CG_GEMM_BM);
            queA1_.EnQue(aG);
            if (hilo_) {
                LocalTensor<half> aGLo = queA1_.AllocTensor<half>();
                DataCopy(aGLo, dnScGM_[aUnit + CG_GEMM_BK * CG_GEMM_BM], CG_GEMM_BK * CG_GEMM_BM);
                queA1_.EnQue(aGLo);
            }
            LocalTensor<half> bX = queB1_.AllocTensor<half>();
            // x is blocked-NZ from the cast kernel: tile (k, n0/128) is one
            // contiguous 32KB burst (strip-major: nb*tTiles + k).
            uint64_t bTile = static_cast<uint64_t>(n0 / CG_GEMM_BN) * tTiles + k;
            DataCopy(bX, xF16GM_[bTile * (CG_GEMM_BK * CG_GEMM_BN)], CG_GEMM_BK * CG_GEMM_BN);
            queB1_.EnQue(bX);
            // ---- L1 -> L0 -> Mmad. x L0B tile feeds every Mmad of this step;
            // A queue pops kv hi[, kv lo, sc hi, sc lo] matching enqueue order.
            LocalTensor<half> xL0 = DeQueBToL0Natural();
            LocalTensor<half> aWL0 = DeQueAToL0ATranspose();
            DoMmad(cW, aWL0, xL0, init);
            queL0A_.FreeTensor(aWL0);
            if (hilo_) {
                LocalTensor<half> aWLoL0 = DeQueAToL0ATranspose();
                DoMmad(cW, aWLoL0, xL0, false);
                queL0A_.FreeTensor(aWLoL0);
            }
            LocalTensor<half> aGL0 = DeQueAToL0ATranspose();
            DoMmad(cG, aGL0, xL0, init);
            queL0A_.FreeTensor(aGL0);
            if (hilo_) {
                LocalTensor<half> aGLoL0 = DeQueAToL0ATranspose();
                DoMmad(cG, aGLoL0, xL0, false);
                queL0A_.FreeTensor(aGLoL0);
            }
            queL0B_.FreeTensor(xL0);
        }
        // C % 128 == 0 and H % 128 == 0 (host checks) -> full tiles,
        // mValid is always G_BM.
        queL0C_.EnQue(cW);
        cW = queL0C_.DeQue<float>();
        FixOut(dWkvGM_, static_cast<uint64_t>(m0) * hSize_ + n0, cW, CG_GEMM_BM, hSize_);
        queL0C_.FreeTensor(cW);
        queL0C_.EnQue(cG);
        cG = queL0C_.DeQue<float>();
        FixOut(dWgateGM_, static_cast<uint64_t>(m0) * hSize_ + n0, cG, CG_GEMM_BM, hSize_);
        queL0C_.FreeTensor(cG);
    }

    // branch A: d_x = dn_kv @ wkv + dn_sc @ wgate (M=T, N=H, K=C)
    // unit 循环展开为单一 term 流，distance-2 预取可跨 pair 边界；
    // 每 unit 的累加器轮转 / Mmad 次序 / FixOut 次序不变。
    __aicore__ inline void ProcessDx(uint32_t cid)
    {
        uint32_t nM = CeilDivU32(tLen_, CG_GEMM_BM);
        uint32_t nH = CeilDivU32(hSize_, CG_GEMM_BN);
        uint32_t nPairRow = CeilDivU32(nH, 2); // tile pairs per m-row (nH always even)
        uint32_t total = nM * nPairRow;
        uint32_t start, count;
        BalancedRange(cid, gDx_, total, start, count);
        if (count == 0) {
            return;
        }
        uint32_t cTiles = CeilDivU32(cSize_, CG_GEMM_BK);
        uint32_t tpu = cTiles * 2U; // terms per unit (kv + sc)
        uint32_t end = start + count;
        // enqueue cursor (eU,eT) leads the consume cursor (cU,cT) by <=2 terms
        uint32_t eU = start, eT = 0;
        for (uint32_t i = 0; i < 2U && eU < end; i++) { // prologue
            uint32_t mi = eU / nPairRow, pj = eU % nPairRow;
            DxEnqueueTerm(mi * CG_GEMM_BM, pj * 2U * CG_GEMM_BN, eT >> 1, (eT & 1U) != 0U, cTiles);
            eT++;
            if (eT == tpu) {
                eT = 0;
                eU++;
            }
        }
        uint32_t cU = start, cT = 0;
        uint32_t mi = cU / nPairRow, pj = cU % nPairRow;
        uint32_t cM0 = mi * CG_GEMM_BM, cN0 = pj * 2U * CG_GEMM_BN;
        LocalTensor<float> cA = queL0C_.AllocTensor<float>();
        LocalTensor<float> cB = queL0C_.AllocTensor<float>();
        while (cU < end) {
            DxTerm(cA, cB, cT == 0);
            // refill the freed slots with the next stream term (may belong to
            // the next unit — issued before this unit's FixOut on purpose)
            if (eU < end) {
                uint32_t emi = eU / nPairRow, epj = eU % nPairRow;
                DxEnqueueTerm(emi * CG_GEMM_BM, epj * 2U * CG_GEMM_BN, eT >> 1, (eT & 1U) != 0U, cTiles);
                eT++;
                if (eT == tpu) {
                    eT = 0;
                    eU++;
                }
            }
            cT++;
            if (cT == tpu) { // unit done: drain + FixOut, rotate accumulators
                cT = 0;
                uint32_t mValid = MinU32(CG_GEMM_BM, tLen_ - cM0);
                queL0C_.EnQue(cA);
                cA = queL0C_.DeQue<float>();
                FixOut(dXGM_, static_cast<uint64_t>(cM0) * hSize_ + cN0, cA, mValid, hSize_);
                queL0C_.FreeTensor(cA);
                queL0C_.EnQue(cB);
                cB = queL0C_.DeQue<float>();
                FixOut(dXGM_, static_cast<uint64_t>(cM0) * hSize_ + cN0 + CG_GEMM_BN, cB, mValid, hSize_);
                queL0C_.FreeTensor(cB);
                cU++;
                if (cU < end) {
                    mi = cU / nPairRow;
                    pj = cU % nPairRow;
                    cM0 = mi * CG_GEMM_BM;
                    cN0 = pj * 2U * CG_GEMM_BN;
                    cA = queL0C_.AllocTensor<float>();
                    cB = queL0C_.AllocTensor<float>();
                }
            }
        }
    }

    // branch B: d_wkv / d_wgate (C, H) direct — one unit = (m0,n0) x {cW,cG}
    __aicore__ inline void ProcessDw(uint32_t gid, uint32_t coreNumW)
    {
        uint32_t nCn = CeilDivU32(cSize_, CG_GEMM_BM); // M tiles over C
        uint32_t nHm = CeilDivU32(hSize_, CG_GEMM_BN); // N tiles over H
        uint32_t total = nCn * nHm;
        uint32_t start, count;
        BalancedRange(gid, coreNumW, total, start, count);
        // n-fastest unit order keeps a core on the same m0 for nHm consecutive
        // units, so its dn panel re-reads hit L2.
        for (uint32_t u = start; u < start + count; u++) {
            DwPair((u / nHm) * CG_GEMM_BM, (u % nHm) * CG_GEMM_BN);
        }
    }

    GlobalTensor<half> xF16GM_, wkvF16GM_, wgateF16GM_, dnKvGM_, dnScGM_;
    GlobalTensor<ioT> dXGM_, dWkvGM_, dWgateGM_;
    uint32_t tLen_, hSize_, cSize_, tPad_, gDx_, dnTilesT_;
    bool hilo_;
    TQue<TPosition::A1, CG_L1_A_DEPTH> queA1_;
    TQue<TPosition::B1, CG_L1_B_DEPTH> queB1_;
    TQue<TPosition::A2, 2> queL0A_;
    TQue<TPosition::B2, 2> queL0B_;
    TQue<TPosition::CO1, 2> queL0C_;
};

#endif // COMPRESSOR_GRAD_GEMM_H
