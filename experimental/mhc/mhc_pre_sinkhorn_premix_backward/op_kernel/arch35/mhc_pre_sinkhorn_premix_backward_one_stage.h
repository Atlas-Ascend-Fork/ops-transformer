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
 * \file mhc_pre_sinkhorn_premix_backward_one_stage.h
 * \brief mhc_pre_sinkhorn_premix_backward
 */

#ifndef MHC_PRE_SINKHORN_PREMIX_BACKWARD_ONE_STAGE_H
#define MHC_PRE_SINKHORN_PREMIX_BACKWARD_ONE_STAGE_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "mhc_pre_sinkhorn_premix_backward_simd_vf.h"
#include "mhc_pre_sinkhorn_premix_backward_simt_vf.h"
#include "mhc_pre_sinkhorn_premix_backward_data_arch35.h"

using namespace AscendC;
using namespace Reg;

namespace {
constexpr int32_t DOUBLE_BUF = 2;
constexpr int32_t BYTE_SIZE_PER_BLOCK = 32;
constexpr int32_t ELEMENTS_SIZE_PER_BLOCK = BYTE_SIZE_PER_BLOCK / sizeof(float);
constexpr int32_t FP32_PER_REPEAT = 64;

constexpr MatmulConfig MHC_PRE_GRAD_MM1_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
constexpr MatmulConfig MHC_PRE_GRAD_MM2_CFG = GetMDLConfig(false, false, 0, true, false, false, true);
// 收尾 pass 每行(c 维)处理的元素数；需 % 64 == 0 保证三次 CopyIn 与 CopyOut 全部 32B 对齐
constexpr int32_t GRADX_FIN_CHUNK = 4096;

// bf16 路径 gradX 精度修复：vec 部分不再先舍入写 bf16 GM（旧结构为 vec 舍入写 + mm1 原子加，
// 双重舍入叠加 HF32 截断导致 bf16 输出相对 fp64 真值约 28% 元素差 1 ulp，超 ATK default 的
// bf16 阈值 4e-3）；改为 mm1 非原子写 fp32 workspace + 收尾 pass fp32 合成单次舍入，
// 与 deterministic(key1)/A3(arch22) 的 gradX 装配结构同构。mm1 输出 dtype 随之切 fp32。
template <typename T, typename U>
struct Mm1OutSelector {
    using type = U;
};
template <typename T>
struct Mm1OutSelector<T, bfloat16_t> {
    using type = T;
};
template <typename T>
struct Mm1OutSelector<T, half> {
    using type = T;
};
} // namespace

template <typename T, typename U>
class MhcPreSinkhornPremixBackwardOneStage {
public:
    // 全 dtype 统一 fp32 workspace 单次舍入装配（对齐 arch22/golden 语义）。
    // fp16 旧路径（HF32 截断 + fp16 原子累加双重舍入）在 ATK 原生复检
    // （accuracy_lt 双标杆对称性）下系统性放大噪声导致 FAIL，已弃用
    static constexpr bool kGradXWsFp32_ = true;
    using A0Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using A1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using BType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;
    using C1Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, typename Mm1OutSelector<T, U>::type>;
    using C2Type = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T>;

    matmul::MatmulImpl<A0Type, BType, C1Type, C1Type, MHC_PRE_GRAD_MM1_CFG> mm1_;
    matmul::MatmulImpl<A1Type, BType, C2Type, C2Type, MHC_PRE_GRAD_MM2_CFG> mm2_;

    __aicore__ inline MhcPreSinkhornPremixBackwardOneStage() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
                                GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR preMix, GM_ADDR gradPre, GM_ADDR gradX,
                                GM_ADDR gradPhi, GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR gradPreMix,
                                GM_ADDR workspace, const MhcPreSinkhornPremixBackwardArch35TilingData* tilingData,
                                TPipe* pipe)
    {
        pipe_ = pipe;
        blkIdx_ = GetBlockIdx();

        InitTiling(tilingData);
        InitGM(x, phi, hPre, gradHin, gradHPost, gradHRes, alpha, bias, hcBeforeNorm, invRms, sumOut, normOut, preMix,
               gradPre, gradX, gradPhi, gradAlpha, gradBias, gradPreMix, workspace);
        InitBuffer();
    }
    __aicore__ inline void Process();

protected:
    int8_t ping1_ = 0;
    int64_t blkIdx_, aivNum_, aicNum_;
    TPipe* pipe_;
    int64_t batchSize_, seqLength_, totalTasks_, totalTasksAligned_;
    int64_t c_, n_, c0_, c1_;
    int64_t tile_, tileAligned_;
    int64_t skIterCount_;
    int64_t mm1K_, mm1M_, mm1N_;
    int64_t mm2K_, mm2M_, mm2N_;
    int64_t mm2SliceN_;
    T alphaPre_, alphaPost_, alphaRes_;

    TBuf<TPosition::VECCALC> gradHinBuf_, hPreBuf_, hResBuf_, xBuf_;
    TBuf<TPosition::VECCALC> gradPreMixStageBuf_;
    TBuf<TPosition::VECCALC> pmGradPreSlotBuf_;
    TBuf<TPosition::VECCALC> xFp32Buf_, gradXBuf_;
    TBuf<TPosition::VECCALC> fusedGradHPreAndGradHPostBuf_, gradHResBuf_;
    TBuf<TPosition::VECCALC> alphaBuf_, biasBuf_, invRmsBuf_, gradInvRmsBuf_;
    // 小输出(gradAlpha/gradBias)每核本地累加:逐 tile 原子 -> 每核一次原子归并
    TBuf<TPosition::VECCALC> alphaAccBuf_, biasAccBuf_, alphaResAccBuf_, biasResAccBuf_, smallOutBuf_;
    TBuf<TPosition::VECCALC> smallPartBuf_;
    TBuf<TPosition::VECCALC> fusedHPre2AndHPost2Buf_;
    TBuf<TPosition::VECCALC> sigmoidGradBuf1_, sigmoidGradBuf2_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> gradRMSNormBuf_;
    TBuf<TPosition::VECCALC> skNormPreBuf_, skSumPreBuf_;
    TBuf<TPosition::VECOUT> gradH2Buf_;
    TBuf<TPosition::VECOUT> gradAlphaBuf_, gradBiasBuf_;
    // bf16：逐 token 标量积 gradRsqrt*gradRMSNorm 的暂存（主循环 spill，收尾 pass 读回）
    TBuf<TPosition::VECCALC> sSpillBuf_;
    // bf16 收尾 pass 缓冲（pipe_->Reset() 后分配，独占 UB）
    TQue<QuePosition::VECIN, DOUBLE_BUF> xFinQue_, gradHinFinQue_, mFinQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUF> gradXFinQue_;
    TBuf<TPosition::VECCALC> xFinTBuf_, ghFinTBuf_, tmpFinTBuf_, hPreFinBuf_, sFinBuf_;

    LocalTensor<U> xLocal_;
    LocalTensor<T> xFp32Local_;
    LocalTensor<U> gradHinLocal_;
    LocalTensor<U> gradXLocal_;
    LocalTensor<T> hPreLocal_;
    LocalTensor<T> gradPreMixStageLocal_;
    LocalTensor<T> pmGradPreSlotLocal_;
    LocalTensor<T> fusedGradHPreAndGradHPostLocal_, gradHResLocal_;
    LocalTensor<T> biasLocal_, alphaLocal_;
    LocalTensor<T> gradBiasLocal_, gradAlphaLocal_;

    LocalTensor<T> invRmsLocal_, gradInvRmsLocal_;
    LocalTensor<T> gradH2Local_;
    LocalTensor<T> fusedHPre2AndHPost2Local_, hRes2Local_;
    LocalTensor<T> tmpLocal_;
    LocalTensor<T> gradSigmoidLocal1_;
    LocalTensor<T> gradSigmoidLocal2_;
    LocalTensor<T> gradRMSNormLocal_;
    LocalTensor<T> alphaAccLocal_, biasAccLocal_, alphaResAccLocal_, biasResAccLocal_, smallOutLocal_;
    LocalTensor<T> smallPartLocal_;
    LocalTensor<T> skNormPreLocal_, skSumPreLocal_;
    LocalTensor<T> sSpillLocal_;
    LocalTensor<T> xFinT_, ghFinT_, tmpFinT_, hPreFinLocal_, sFinLocal_;
    int64_t spillSlotElems_ = 0; // 逐 tile spill 槽位跨度：AlignUp(tile_, 8)，保证 GM 写 32B 对齐

    GlobalTensor<U> xGlobal_;
    GlobalTensor<T> xFp32Global_;
    GlobalTensor<U> gradHinGlobal_;
    GlobalTensor<T> hPreGlobal_;
    GlobalTensor<U> gradXGlobal_;
    GlobalTensor<T> gradHPostGlobal_;
    GlobalTensor<T> alphaGlobal_, biasGlobal_;
    GlobalTensor<T> invRmsGlobal_;
    GlobalTensor<T> gradAlphaGlobal_, gradBiasGlobal_;
    GlobalTensor<T> h2Global_;
    GlobalTensor<T> skNormGlobal_, skSumGlobal_;
    GlobalTensor<T> preMixGlobal_, gradPreGlobal_, gradPreMixGlobal_;
    GlobalTensor<T> smallPartWsGlobal_;
    bool hasPreMix_ = false;
    bool hasGradPre_ = false;
    float eps_ = 0.f;
    GlobalTensor<T> gradHResGlobal_;
    GlobalTensor<T> gradH2Global_;
    GlobalTensor<T> gradPhiGlobal_;
    GlobalTensor<T> phiGlobal_;
    GlobalTensor<T> gradXMWsGlobal_; // bf16：mm1 非原子输出的 fp32 workspace plane
    GlobalTensor<T> sSpillGlobal_; // bf16：标量积 spill 区（fp16 workspace 不含此两段，仅赋值不解引用）

private:
    __aicore__ inline void InitTiling(const MhcPreSinkhornPremixBackwardArch35TilingData* tilingData)
    {
        batchSize_ = tilingData->batchSize;
        seqLength_ = tilingData->seqLength;
        aivNum_ = tilingData->aivNum;
        aicNum_ = tilingData->aicNum;
        hasPreMix_ = tilingData->hasPreMix != 0;
        hasGradPre_ = tilingData->hasGradPre != 0;
        eps_ = tilingData->eps;
        c_ = tilingData->c;
        n_ = tilingData->n;
        c0_ = tilingData->c0;
        c1_ = tilingData->c1;
        skIterCount_ = tilingData->skIterCount;
        tile_ = tilingData->tileSize;
        tileAligned_ = AlignUp(tile_, BYTE_SIZE_PER_BLOCK / sizeof(T));
        totalTasks_ = batchSize_ * seqLength_;
        totalTasksAligned_ = AlignUp(totalTasks_, aivNum_ * tile_);
        spillSlotElems_ = AlignUp(tile_, ELEMENTS_SIZE_PER_BLOCK);

        if ASCEND_IS_AIC {
            mm1K_ = n_ * n_ + 2 * n_;
            mm1M_ = tile_ * 2;
            mm1N_ = n_ * c_;

            mm2K_ = batchSize_ * seqLength_;
            mm2M_ = n_ * n_ + 2 * n_;
            mm2N_ = n_ * c_;
            // CeilDiv 切 N 维，不整除时由 ProcessMatmul2 的尾核截断兜底
            mm2SliceN_ = (mm2N_ + aicNum_ - 1) / aicNum_;
        }
    }

    __aicore__ inline void InitGM(GM_ADDR x, GM_ADDR phi, GM_ADDR hPre, GM_ADDR gradHin, GM_ADDR gradHPost,
                                  GM_ADDR gradHRes, GM_ADDR alpha, GM_ADDR bias, GM_ADDR hcBeforeNorm, GM_ADDR invRms,
                                  GM_ADDR sumOut, GM_ADDR normOut, GM_ADDR preMix, GM_ADDR gradPre, GM_ADDR gradX,
                                  GM_ADDR gradPhi, GM_ADDR gradAlpha, GM_ADDR gradBias, GM_ADDR gradPreMix,
                                  GM_ADDR workspace)
    {
        // cube 和 vector 共用
        xGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(x));             // fp32
        xFp32Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace)); // fp32
        gradPhiGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradPhi)); // fp32
        phiGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(phi));         // fp32
        gradXGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradX));     // fp32
        gradH2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) + batchSize_ * seqLength_ * n_ * c_);
        // workspace 布局：[xFp32: bs*n*c][gradH2: bs*(n*n+2n)]，bf16 追加 [gradXM: bs*n*c][sSpill]
        gradXMWsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) +
                                        batchSize_ * seqLength_ * (n_ * c_ + n_ * n_ + 2 * n_));
        sSpillGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) +
                                      batchSize_ * seqLength_ * (2 * n_ * c_ + n_ * n_ + 2 * n_));
        // 小输出每核 partial:[aivNum][32] fp32,blk0 归并
        smallPartWsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(workspace) +
                                           batchSize_ * seqLength_ * (2 * n_ * c_ + n_ * n_ + 2 * n_) +
                                           ((batchSize_ * seqLength_ + tile_ - 1) / tile_) * spillSlotElems_);

        if ASCEND_IS_AIV {
            hPreGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hPre)); // fp32
            // v2 可选通路:pre_mix 替代 h_pre 作 y 折权重;grad_pre 在合并点三态;grad_pre_mix 输出
            preMixGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(preMix));   // fp32(可空)
            gradPreGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradPre)); // fp32(可空)
            gradPreMixGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradPreMix));
            gradHinGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(gradHin)); // fp32

            gradHPostGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradHPost)); // fp32

            alphaGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(alpha)); // fp32
            biasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(bias));   // fp32

            invRmsGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(invRms)); // fp32
            gradAlphaGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradAlpha));
            gradBiasGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradBias));

            h2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(hcBeforeNorm)); // fp32

            skNormGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(normOut)); // fp32
            skSumGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sumOut));   // fp32

            gradHResGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(gradHRes)); // fp32
        }
    }

    __aicore__ inline void InitBuffer()
    {
        if ASCEND_IS_AIV {
            pipe_->InitBuffer(xBuf_, n_ * c_ * sizeof(U));
            pipe_->InitBuffer(xFp32Buf_, DOUBLE_BUF * c_ * sizeof(T));
            pipe_->InitBuffer(gradHinBuf_, DOUBLE_BUF * c_ * sizeof(U));
            if constexpr (kGradXWsFp32_) {
                // bf16 不在主循环舍入写 gradX，gradXBuf_ 不分配；改配标量 spill 暂存
                pipe_->InitBuffer(sSpillBuf_, AlignUp(tile_, ELEMENTS_SIZE_PER_BLOCK) * sizeof(T));
            } else {
                pipe_->InitBuffer(gradXBuf_, DOUBLE_BUF * c_ * sizeof(U));
            }
            pipe_->InitBuffer(hPreBuf_, DOUBLE_BUF * 2 * n_ * sizeof(T));
            // v2:grad_pre_mix 紧凑暂存(hasPreMix 路径)
            pipe_->InitBuffer(gradPreMixStageBuf_, tile_ * n_ * sizeof(T));
            // v2:pm&&gp 时 grad_pre 的 ping-pong 宽槽(8 元素/槽,16B 实际 32B 块写溢出落本槽 padding)
            pipe_->InitBuffer(pmGradPreSlotBuf_, DOUBLE_BUF * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(hResBuf_, tile_ * n_ * n_ * sizeof(T));
            pipe_->InitBuffer(invRmsBuf_, tile_ * sizeof(T));
            pipe_->InitBuffer(gradH2Buf_, tile_ * (n_ * n_ + 2 * n_) * sizeof(T));

            int32_t sigmoidGradBufSize = AlignUp(tile_ * 2 * n_ * sizeof(T), VEC_LENGTH);

            pipe_->InitBuffer(sigmoidGradBuf1_, sigmoidGradBufSize);
            pipe_->InitBuffer(sigmoidGradBuf2_, sigmoidGradBufSize);

            pipe_->InitBuffer(biasBuf_, (n_ * n_ + 2 * n_) * sizeof(T));
            pipe_->InitBuffer(alphaBuf_, (n_ * n_ + 2 * n_) * sizeof(T));

            pipe_->InitBuffer(tmpBuf_, tile_ * (n_ * n_ + 2 * n_) * sizeof(T));
            pipe_->InitBuffer(gradInvRmsBuf_, tile_ * sizeof(T));
            pipe_->InitBuffer(gradHResBuf_, tile_ * n_ * n_ * sizeof(T));
            pipe_->InitBuffer(fusedGradHPreAndGradHPostBuf_, tile_ * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(fusedHPre2AndHPost2Buf_, tile_ * 2 * n_ * sizeof(T));
            pipe_->InitBuffer(gradAlphaBuf_, 8 * sizeof(T));
            pipe_->InitBuffer(gradBiasBuf_, 2 * n_ * sizeof(T));
            pipe_->InitBuffer(gradRMSNormBuf_, tile_ * sizeof(T));
            pipe_->InitBuffer(alphaAccBuf_, 2 * sizeof(T));
            pipe_->InitBuffer(biasAccBuf_, 2 * n_ * sizeof(T));
            pipe_->InitBuffer(alphaResAccBuf_, sizeof(T));
            pipe_->InitBuffer(biasResAccBuf_, n_ * n_ * sizeof(T));
            pipe_->InitBuffer(smallOutBuf_, 32 * sizeof(T));
            pipe_->InitBuffer(smallPartBuf_, aivNum_ * 32 * sizeof(T));
            pipe_->InitBuffer(skNormPreBuf_, skIterCount_ * tile_ * n_ * n_ * sizeof(T));
            // DataCopyPad 多块写 UB 的块距按 32B 对齐，tile_ 为奇数时每块多占一档，buffer 按对齐口径分配
            pipe_->InitBuffer(skSumPreBuf_,
                              2 * skIterCount_ * AlignUp(tile_ * n_, ELEMENTS_SIZE_PER_BLOCK) * sizeof(T));

            xLocal_ = xBuf_.Get<U>();
            xFp32Local_ = xFp32Buf_.Get<T>();
            gradHinLocal_ = gradHinBuf_.Get<U>();

            fusedGradHPreAndGradHPostLocal_ = fusedGradHPreAndGradHPostBuf_.Get<T>();
            fusedHPre2AndHPost2Local_ = fusedHPre2AndHPost2Buf_.Get<T>();

            biasLocal_ = biasBuf_.Get<T>();
            alphaLocal_ = alphaBuf_.Get<T>();

            gradH2Local_ = gradH2Buf_.Get<T>();
            tmpLocal_ = tmpBuf_.Get<T>();

            gradHResLocal_ = gradHResBuf_.Get<T>();

            invRmsLocal_ = invRmsBuf_.Get<T>();
            gradInvRmsLocal_ = gradInvRmsBuf_.Get<T>();

            gradSigmoidLocal1_ = sigmoidGradBuf1_.Get<T>();
            gradSigmoidLocal2_ = sigmoidGradBuf2_.Get<T>();

            gradBiasLocal_ = gradBiasBuf_.Get<T>();
            gradAlphaLocal_ = gradAlphaBuf_.Get<T>();

            gradRMSNormLocal_ = gradRMSNormBuf_.Get<T>();
            alphaAccLocal_ = alphaAccBuf_.Get<T>();
            biasAccLocal_ = biasAccBuf_.Get<T>();
            alphaResAccLocal_ = alphaResAccBuf_.Get<T>();
            biasResAccLocal_ = biasResAccBuf_.Get<T>();
            smallOutLocal_ = smallOutBuf_.Get<T>();
            smallPartLocal_ = smallPartBuf_.Get<T>();
            if constexpr (kGradXWsFp32_) {
                sSpillLocal_ = sSpillBuf_.Get<T>();
            } else {
                gradXLocal_ = gradXBuf_.Get<U>();
            }

            hPreLocal_ = hPreBuf_.Get<T>();
            gradPreMixStageLocal_ = gradPreMixStageBuf_.Get<T>();
            pmGradPreSlotLocal_ = pmGradPreSlotBuf_.Get<T>();

            hRes2Local_ = hResBuf_.Get<T>();

            skNormPreLocal_ = skNormPreBuf_.Get<T>();
            skSumPreLocal_ = skSumPreBuf_.Get<T>();
        }
    }

    __aicore__ inline void GetAlphaAndBias();
    __aicore__ inline void ComputeGradResHHat2(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline void ComputeGradSigmoid(const int32_t taskOffset, const int32_t tileTaskCount);
    __aicore__ inline T ComputeGradInvRms(const int32_t i);
    __aicore__ inline void ComputeGradRMSNorm(const int32_t tileTaskCount);
    __aicore__ inline void ComputeGradXVector(const T gradRsqrtVal, const T gradRMSNormVal, const int32_t taskOffset,
                                              bool copyNext);
    __aicore__ inline void ProcessMatmul1(const int32_t taskOffset, const int32_t mm1M);
    __aicore__ inline void ProcessMatmul2();
    __aicore__ inline void ProcessSmallOutFinal();
    // bf16 收尾 pass：fp32 合成 gradX 并单次舍入写出（仅 kGradXWsFp32_ 时调用）
    __aicore__ inline void ProcessGradXFinal();
};

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornPremixBackwardOneStage<T, U>::GetAlphaAndBias()
{
    // HcBase 常驻 UB
    alphaPre_ = alphaGlobal_.GetValue(0);
    alphaPost_ = alphaGlobal_.GetValue(1);
    alphaRes_ = alphaGlobal_.GetValue(2);
    DataCopyPad(biasLocal_, biasGlobal_,
                {static_cast<uint16_t>(1), static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});

    for (int32_t i = 0; i < n_; i++) {
        alphaLocal_.SetValue(i, alphaPre_);
        alphaLocal_.SetValue(i + n_, alphaPost_);
    }
}

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornPremixBackwardOneStage<T, U>::ComputeGradResHHat2(const int32_t taskOffset,
                                                                                       const int32_t tileTaskCount)
{
    // Compute sigmoidGrad
    DataCopyPad(hRes2Local_, h2Global_[taskOffset * (n_ * n_ + 2 * n_) + 2 * n_],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(n_ * n_ * sizeof(T)),
                 static_cast<int64_t>(2 * n_ * sizeof(T)), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(gradHResLocal_, gradHResGlobal_[taskOffset * n_ * n_],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(n_ * n_ * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(invRmsLocal_, invRmsGlobal_[taskOffset],
                {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * sizeof(T)), 0, 0, 0},
                {false, 0, 0, 0});

    // Sinkhorn 全部迭代数据一次性批量预载到 UB，消除 SIMT 内逐轮 GM 标量装载的延迟链。
    // GM 中迭代块以 totalTasks_*2 行为间隔正向等距排布，UB 内块 j 对应 SIMT 第 skIterCount_-1-j 轮；
    // sum 的 col 段紧随 row 段（GM 源起始偏移 totalTasks_*n_），UB 侧块间距按 32B 对齐（SIMT 内同口径）。
    DataCopyPad(skNormPreLocal_, skNormGlobal_[taskOffset * n_ * n_],
                {static_cast<uint16_t>(skIterCount_), static_cast<uint32_t>(tileTaskCount * n_ * n_ * sizeof(T)),
                 static_cast<int64_t>((totalTasks_ * 2 - tileTaskCount) * n_ * n_ * sizeof(T)), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(skSumPreLocal_, skSumGlobal_[taskOffset * n_],
                {static_cast<uint16_t>(skIterCount_), static_cast<uint32_t>(tileTaskCount * n_ * sizeof(T)),
                 static_cast<int64_t>((totalTasks_ * 2 - tileTaskCount) * n_ * sizeof(T)), 0, 0},
                {false, 0, 0, 0});
    DataCopyPad(skSumPreLocal_[skIterCount_ * AlignUp(tileTaskCount * n_, ELEMENTS_SIZE_PER_BLOCK)],
                skSumGlobal_[(totalTasks_ + taskOffset) * n_],
                {static_cast<uint16_t>(skIterCount_), static_cast<uint32_t>(tileTaskCount * n_ * sizeof(T)),
                 static_cast<int64_t>((totalTasks_ * 2 - tileTaskCount) * n_ * sizeof(T)), 0, 0},
                {false, 0, 0, 0});

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    uint32_t thread_num = min(static_cast<uint32_t>(THREAD_NUM), static_cast<uint32_t>(tileTaskCount * n_ * n_));
    uint32_t xThreadCount = n_ * n_;
    uint32_t yThreadCount = thread_num / xThreadCount;

    Duplicate(gradInvRmsLocal_, 0.f, tileTaskCount);
    Simt::VF_CALL<SinkhornKnoppSimt<T>>(
        Simt::Dim3{xThreadCount, yThreadCount}, (__ubuf__ T*)gradH2Local_.GetPhyAddr(),
        (__ubuf__ T*)gradInvRmsLocal_.GetPhyAddr(), (__ubuf__ T*)alphaResAccLocal_.GetPhyAddr(),
        (__ubuf__ T*)biasResAccLocal_.GetPhyAddr(), (__ubuf__ T*)skSumPreLocal_.GetPhyAddr(),
        (__ubuf__ T*)skNormPreLocal_.GetPhyAddr(), (__ubuf__ T*)gradHResLocal_.GetPhyAddr(),
        (__ubuf__ T*)hRes2Local_.GetPhyAddr(), (__ubuf__ T*)invRmsLocal_.GetPhyAddr(),
        (__ubuf__ T*)biasLocal_[n_ * 2].GetPhyAddr(), (__ubuf__ T*)tmpLocal_.GetPhyAddr(), alphaRes_, tileTaskCount,
        skIterCount_, n_, eps_);
}

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornPremixBackwardOneStage<T, U>::ComputeGradSigmoid(const int32_t taskOffset,
                                                                                      const int32_t tileTaskCount)
{
    DataCopyPad(fusedHPre2AndHPost2Local_, h2Global_[taskOffset * (n_ * n_ + 2 * n_)],
                {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(2 * n_ * sizeof(T)),
                 static_cast<int64_t>(n_ * n_ * sizeof(T)), 0, 0},
                {false, 0, 0, 0});

    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    uint16_t repeatTimes = Ceil(tileTaskCount * n_ * 2, FP32_PER_VL);
    ComputeGradSigmoidVf<T>(gradSigmoidLocal1_, gradSigmoidLocal2_, fusedHPre2AndHPost2Local_, invRmsLocal_,
                            alphaLocal_, biasLocal_, repeatTimes, tileTaskCount * n_ * 2);
}

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornPremixBackwardOneStage<T, U>::ComputeGradRMSNorm(const int32_t tileTaskCount)
{
    Mul(gradRMSNormLocal_, invRmsLocal_, invRmsLocal_, tileTaskCount);
    Mul(gradRMSNormLocal_, gradRMSNormLocal_, invRmsLocal_, tileTaskCount);
    Muls(gradRMSNormLocal_, gradRMSNormLocal_, -1.f / static_cast<T>(n_ * c_), tileTaskCount);
}

template <typename T, typename U>
__aicore__ inline T MhcPreSinkhornPremixBackwardOneStage<T, U>::ComputeGradInvRms(const int32_t i)
{
    Mul(gradSigmoidLocal1_[i * 2 * n_], fusedGradHPreAndGradHPostLocal_[i * 2 * n_], gradSigmoidLocal1_[i * 2 * n_],
        2 * n_);
    Mul(gradSigmoidLocal2_[i * 2 * n_], fusedGradHPreAndGradHPostLocal_[i * 2 * n_], gradSigmoidLocal2_[i * 2 * n_],
        2 * n_);
    Mul(tmpLocal_[i * 2 * n_], gradSigmoidLocal1_[i * 2 * n_], alphaLocal_, 2 * n_);
    Muls(gradH2Local_[i * (2 * n_ + n_ * n_)], tmpLocal_[i * 2 * n_], invRmsLocal_.GetValue(i), 2 * n_);
    Mul(tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], fusedHPre2AndHPost2Local_[i * 2 * n_], 2 * n_);
    ReduceSum<T>(tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], tmpLocal_[i * 2 * n_], 2 * n_);
    T gradRsqrtVal = tmpLocal_.GetValue(i * 2 * n_) + gradInvRmsLocal_.GetValue(i);
    return gradRsqrtVal;
}

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornPremixBackwardOneStage<T, U>::ComputeGradXVector(const T gradRsqrtVal,
                                                                                      const T gradRMSNormVal,
                                                                                      const int32_t taskOffset,
                                                                                      bool copyNext)
{
    int8_t ping = 0;

    SetFlag<HardEvent::MTE3_V>(0);
    SetFlag<HardEvent::MTE3_V>(1);
    for (int32_t i = 0; i < n_; i++) {
        WaitFlag<HardEvent::MTE3_V>(ping);

        if constexpr (!kGradXWsFp32_) {
            ComputeGradXVf<T, U>(gradXLocal_[ping * c_], xLocal_[i * c_], gradHinLocal_[ping1_ * c_],
                                 hPreLocal_[ping1_ * 8 + i], gradRsqrtVal, gradRMSNormVal, c_, c1_);
        }
        Cast(xFp32Local_[ping * c_], xLocal_[i * c_], RoundMode::CAST_NONE, c_);

        SetFlag<HardEvent::V_MTE3>(0);
        SetFlag<HardEvent::V_MTE2>(0);
        WaitFlag<HardEvent::V_MTE3>(0);

        if constexpr (!kGradXWsFp32_) {
            DataCopyPad(gradXGlobal_[taskOffset * n_ * c_ + i * c_], gradXLocal_[ping * c_],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0});
        }
        DataCopyPad(xFp32Global_[taskOffset * n_ * c_ + i * c_], xFp32Local_[ping * c_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(T)), 0, 0, 0});

        SetFlag<HardEvent::MTE3_V>(ping);
        WaitFlag<HardEvent::V_MTE2>(0);

        SetFlag<HardEvent::MTE3_MTE2>(0);
        WaitFlag<HardEvent::MTE3_MTE2>(0);

        if (copyNext) {
            DataCopyPad(xLocal_[i * c_], xGlobal_[(taskOffset + 1) * n_ * c_ + i * c_],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0}, {false, 0, 0, 0});
        }

        ping = 1 - ping;
    }
    WaitFlag<HardEvent::MTE3_V>(0);
    WaitFlag<HardEvent::MTE3_V>(1);
}

/**
    约束：
    c: c >= 64 && c % 64 == 0
    n: n == 4
*/
template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornPremixBackwardOneStage<T, U>::Process()
{
    if ASCEND_IS_AIV {
        // 每核本地累加器清零(所有 AIV),输出全局区由 blk0 清零后全核屏障
        Duplicate(alphaAccLocal_, float(0), 2);
        Duplicate(biasAccLocal_, float(0), 2 * n_);
        Duplicate(alphaResAccLocal_, float(0), 1);
        Duplicate(biasResAccLocal_, float(0), n_ * n_);
    }
    SyncAll<false>();

    if ASCEND_IS_AIV {
        int8_t ping = 0;
        GetAlphaAndBias();

        for (int32_t taskOffset = blkIdx_ * tile_; taskOffset < totalTasksAligned_; taskOffset += aivNum_ * tile_) {
            int32_t tileTaskCount = min(tile_, totalTasks_ - taskOffset);

            if (tileTaskCount > 0) {
                ComputeGradResHHat2(taskOffset, tileTaskCount);
                ComputeGradSigmoid(taskOffset, tileTaskCount);
                ComputeGradRMSNorm(tileTaskCount);

                DataCopyPad(fusedGradHPreAndGradHPostLocal_, gradHPostGlobal_[taskOffset * n_],
                            {static_cast<uint16_t>(tileTaskCount), static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0},
                            {true, static_cast<uint8_t>(ELEMENTS_SIZE_PER_BLOCK - n_), 0, 0});
                DataCopyPad(xLocal_, xGlobal_[taskOffset * n_ * c_],
                            {static_cast<uint16_t>(n_), static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0},
                            {false, 0, 0, 0});

                SetFlag<HardEvent::V_MTE2>(0);
                SetFlag<HardEvent::V_MTE2>(1);
                for (int32_t i = 0; i < tileTaskCount; i++) {
                    WaitFlag<HardEvent::V_MTE2>(ping1_);
                    // v2:y 折权重源二选一(hasPreMix 时 pre_mix 替代 h_pre,golden.py 口径)
                    GlobalTensor<T>& mixWeightSrc = hasPreMix_ ? preMixGlobal_ : hPreGlobal_;
                    DataCopyPad(hPreLocal_[ping1_ * 8], mixWeightSrc[(taskOffset + i) * n_],
                                {static_cast<uint16_t>(1), static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0},
                                {false, 0, 0, 0});
                    DataCopyPad(gradHinLocal_[ping1_ * c_], gradHinGlobal_[(taskOffset + i) * c_],
                                {static_cast<uint16_t>(1), static_cast<uint32_t>(c_ * sizeof(U)), 0, 0, 0},
                                {false, 0, 0, 0});
                    // v2 装载:grad_pre 一律入 ping-pong 宽槽(MTE2 16B 请求实测按 32B 块搬,
                    // 直写 fused 的 4 元素 pre 槽会溢出覆盖 post 段 = 下一 token 数据,实测踩坑);
                    // 宽槽溢出落本槽 padding,随后 V 侧按 n 计数取出
                    if (hasGradPre_) {
                        DataCopyPad(pmGradPreSlotLocal_[ping1_ * 8], gradPreGlobal_[(taskOffset + i) * n_],
                                    {static_cast<uint16_t>(1), static_cast<uint32_t>(n_ * sizeof(T)), 0, 0, 0},
                                    {false, 0, 0, 0});
                    }

                    SetFlag<HardEvent::MTE2_V>(0);
                    WaitFlag<HardEvent::MTE2_V>(0);

                    // v2:d_mix_for_y 的落点二选一(hasPreMix 时直写紧凑暂存,pre 槽不承载)
                    ComputeGradPreVf<T>(
                        hasPreMix_ ? gradPreMixStageLocal_[i * n_] : fusedGradHPreAndGradHPostLocal_[i * 2 * n_],
                        xLocal_, gradHinLocal_[ping1_ * c_], n_, c_ / FP32_PER_VL);
                    if (hasPreMix_) {
                        // pre 槽 = grad_pre(或 0):V 侧按 n 计数装配,避免 MTE2 块粒度溢出
                        if (hasGradPre_) {
                            Duplicate(fusedGradHPreAndGradHPostLocal_[i * 2 * n_], float(0), n_);
                            Add(fusedGradHPreAndGradHPostLocal_[i * 2 * n_],
                                fusedGradHPreAndGradHPostLocal_[i * 2 * n_], pmGradPreSlotLocal_[ping1_ * 8], n_);
                        } else {
                            Duplicate(fusedGradHPreAndGradHPostLocal_[i * 2 * n_], float(0), n_);
                        }
                    } else if (hasGradPre_) {
                        Add(fusedGradHPreAndGradHPostLocal_[i * 2 * n_], fusedGradHPreAndGradHPostLocal_[i * 2 * n_],
                            pmGradPreSlotLocal_[ping1_ * 8], n_);
                    }
                    T gradRsqrtVal = ComputeGradInvRms(i);
                    if constexpr (kGradXWsFp32_) {
                        // bf16：暂存逐 token 标量积，收尾 pass 以 fp32 重算 vec 部分并单次舍入
                        sSpillLocal_.SetValue(i, gradRsqrtVal * gradRMSNormLocal_.GetValue(i));
                    }
                    ComputeGradXVector(gradRsqrtVal, gradRMSNormLocal_.GetValue(i), taskOffset + i,
                                       i != tileTaskCount - 1);
                    SetFlag<HardEvent::V_MTE2>(ping1_);
                    ping1_ = 1 - ping1_;
                }
                if constexpr (kGradXWsFp32_) {
                    // spill 槽位起点按 AlignUp(tile_,8) 对齐，保证 GM 写 32B 对齐；
                    // MTE3_S 就地等待避免下一 tile 的 SetValue 与本次写出竞争
                    SetFlag<HardEvent::S_MTE3>(0);
                    WaitFlag<HardEvent::S_MTE3>(0);
                    DataCopyPad(sSpillGlobal_[(taskOffset / tile_) * spillSlotElems_], sSpillLocal_,
                                {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * sizeof(T)), 0, 0, 0});
                    SetFlag<HardEvent::MTE3_S>(0);
                    WaitFlag<HardEvent::MTE3_S>(0);
                }
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);

                if (hasPreMix_) {
                    // grad_pre_mix = d_mix_for_y(紧凑暂存一遍拉,避免逐 token 16B GM 写)
                    DataCopyPad(
                        gradPreMixGlobal_[taskOffset * n_], gradPreMixStageLocal_,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * n_ * sizeof(T)), 0, 0, 0});
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                }

                DataCopyPad(gradH2Global_[taskOffset * (n_ * n_ + 2 * n_)], gradH2Local_,
                            {static_cast<uint16_t>(tileTaskCount),
                             static_cast<uint32_t>((n_ * n_ + 2 * n_) * sizeof(T)), 0, 0, 0});

                WaitFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(1);
            }

            CrossCoreSetFlag<0x2, PIPE_MTE3>(ping);
            ping = (ping + 1) % 10;

            if (tileTaskCount > 0) {
                int32_t dupSize = AlignUp(tile_ * n_ * 2, FP32_PER_REPEAT) - tileTaskCount * 2 * n_;
                Duplicate(gradSigmoidLocal1_[tileTaskCount * 2 * n_], 0.f, dupSize);
                Duplicate(gradSigmoidLocal2_[tileTaskCount * 2 * n_], 0.f, dupSize);
                // compute gradHcScale
                int32_t repeatTimes = Ceil(tileTaskCount * n_ * 2, FP32_PER_REPEAT);
                uint64_t preMask[] = {0X0F0F0F0F0F0F0F0F};
                uint64_t postMask[] = {0XF0F0F0F0F0F0F0F0};
                ReduceSum<T>(gradAlphaLocal_, gradSigmoidLocal2_, tmpLocal_, preMask, repeatTimes, 8);
                ReduceSum<T>(gradAlphaLocal_[1], gradSigmoidLocal2_, tmpLocal_, postMask, repeatTimes, 8);
                uint32_t shape[] = {static_cast<uint32_t>(tileTaskCount), static_cast<uint32_t>(2 * n_)};
                ReduceSum<T, Pattern::Reduce::RA, true>(gradBiasLocal_, gradSigmoidLocal1_,
                                                        tmpLocal_.template ReinterpretCast<uint8_t>(), shape, true);

                // pre/post 贡献并入本核累加器(V 序内),不再逐 tile 原子写全局输出
                Add(alphaAccLocal_, alphaAccLocal_, gradAlphaLocal_, 2);
                Add(biasAccLocal_, biasAccLocal_, gradBiasLocal_, 2 * n_);
            }
        }
    }

    if ASCEND_IS_AIV {
        // 每核装配 32 元素 partial 写入 workspace(V->MTE3 同步后一次 128B 写),
        // blk0 在屏障后做补偿归并,彻底消除跨核原子舍入链
        Duplicate(smallOutLocal_, float(0), 32);
        smallOutLocal_.SetValue(0, alphaAccLocal_.GetValue(0));
        smallOutLocal_.SetValue(1, alphaAccLocal_.GetValue(1));
        smallOutLocal_.SetValue(2, alphaResAccLocal_.GetValue(0));
        DataCopy(smallOutLocal_[8], biasAccLocal_, 2 * n_);
        DataCopy(smallOutLocal_[8 + 2 * n_], biasResAccLocal_, n_ * n_);
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(smallPartWsGlobal_[blkIdx_ * 32], smallOutLocal_,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(32 * sizeof(T)), 0, 0, 0});
    }

    if ASCEND_IS_AIC {
        int8_t ping = 0;
        for (int32_t taskOffset = blkIdx_ * 2 * tile_; taskOffset < totalTasksAligned_;
             taskOffset += aicNum_ * 2 * tile_) {
            int32_t tileTaskCount = min(2 * tile_, totalTasks_ - taskOffset);
            CrossCoreWaitFlag<0x2>(ping);
            ping = (ping + 1) % 10;

            if (tileTaskCount > 0) {
                ProcessMatmul1(taskOffset, tileTaskCount);
            }
        }
    }

    // AIV+AIC 全核屏障：mm2 要按全量 K=bs 读所有核产出的 gradH2/xFp32，
    // 逐组 flag 只保证本组 AIV 的 tile 落盘，不足以为 mm2 提供全量可见性
    SyncAll<false>();

    if ASCEND_IS_AIV {
        if (blkIdx_ == 0) {
            ProcessSmallOutFinal();
        }
        if constexpr (kGradXWsFp32_) {
            // bf16：mm1 的 fp32 部分和已全数落盘，合成 vec 部分并单次舍入写出 gradX
            ProcessGradXFinal();
        }
    } else {
        // 各 AIC 按 N 维取不相交切片，直接非原子写 gradPhi，消除逐核原子写放大
        ProcessMatmul2();
    }
}

// mm1K_ = n_ * n_ + 2 * n_;
// mm1M_ = tile_ * 2;
// mm1N_ = n_ * c_;

// mm2K_ = batchSize_ * seqLength_;
// mm2M_ = n_ * n_ + 2 * n_;
// mm2N_ = n_ * c_;  // org N；单核实算 N = min(mm2SliceN_, mm2N_ - blkIdx_ * mm2SliceN_)
template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornPremixBackwardOneStage<T, U>::ProcessMatmul1(const int32_t taskOffset,
                                                                                  const int32_t mm1M)
{
    if (mm1M <= 0)
        return;

    mm1_.SetTensorA(gradH2Global_[taskOffset * mm1K_]);
    mm1_.SetTensorB(phiGlobal_);
    // 全 dtype：关 HF32 截断，非原子直写 fp32 workspace，收尾 pass 单次舍入
    mm1_.SetHF32(false, 1);
    mm1_.SetSingleShape(mm1M, mm1N_, mm1K_);
    mm1_.template IterateAll<false>(gradXMWsGlobal_[taskOffset * (n_ * c_)], 0);
    mm1_.End();
}

// bf16 gradX 收尾 pass：gradX[task,k,:] = cast_bf16(gradHin[task,:]·hPre[task,k] + x[task,k,:]·s[task]
// + mm1Ws[task,k,:])，fp32 求和单次舍入。注意 gradHin 是 [B,S,C] 逐 token 的（n_ 行共享），
// x/WS_m/gradX 是 [B,S,N,C] 逐行的。tile 分配与主循环一致，spill/hPre 按 tile 批量装载。
// 先于 pipe_->Reset() 排空主循环的 MTE2/MTE3（flag 触发即对应 pipe 队列排空），再独占 UB。
template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornPremixBackwardOneStage<T, U>::ProcessGradXFinal()
{
    SetFlag<HardEvent::MTE3_V>(6);
    WaitFlag<HardEvent::MTE3_V>(6);
    SetFlag<HardEvent::MTE2_V>(6);
    WaitFlag<HardEvent::MTE2_V>(6);
    pipe_->Reset();

    pipe_->InitBuffer(xFinQue_, DOUBLE_BUF, GRADX_FIN_CHUNK * sizeof(U));
    pipe_->InitBuffer(gradHinFinQue_, DOUBLE_BUF, GRADX_FIN_CHUNK * sizeof(U));
    pipe_->InitBuffer(mFinQue_, DOUBLE_BUF, GRADX_FIN_CHUNK * sizeof(T));
    pipe_->InitBuffer(gradXFinQue_, DOUBLE_BUF, GRADX_FIN_CHUNK * sizeof(U));
    pipe_->InitBuffer(xFinTBuf_, GRADX_FIN_CHUNK * sizeof(T));
    pipe_->InitBuffer(ghFinTBuf_, GRADX_FIN_CHUNK * sizeof(T));
    pipe_->InitBuffer(tmpFinTBuf_, GRADX_FIN_CHUNK * sizeof(T));
    pipe_->InitBuffer(hPreFinBuf_, AlignUp(tile_ * n_, ELEMENTS_SIZE_PER_BLOCK) * sizeof(T));
    pipe_->InitBuffer(sFinBuf_, spillSlotElems_ * sizeof(T));
    xFinT_ = xFinTBuf_.Get<T>();
    ghFinT_ = ghFinTBuf_.Get<T>();
    tmpFinT_ = tmpFinTBuf_.Get<T>();
    hPreFinLocal_ = hPreFinBuf_.Get<T>();
    sFinLocal_ = sFinBuf_.Get<T>();

    for (int32_t taskOffset = blkIdx_ * tile_; taskOffset < totalTasksAligned_; taskOffset += aivNum_ * tile_) {
        int32_t tileTaskCount = min(tile_, totalTasks_ - taskOffset);
        if (tileTaskCount <= 0) {
            continue;
        }
        DataCopyPad(hPreFinLocal_, (hasPreMix_ ? preMixGlobal_ : hPreGlobal_)[taskOffset * n_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * n_ * sizeof(T)), 0, 0, 0},
                    {false, 0, 0, 0});
        DataCopyPad(sFinLocal_, sSpillGlobal_[(taskOffset / tile_) * spillSlotElems_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(tileTaskCount * sizeof(T)), 0, 0, 0},
                    {false, 0, 0, 0});
        SetFlag<HardEvent::MTE2_S>(1);
        WaitFlag<HardEvent::MTE2_S>(1);

        for (int32_t t = 0; t < tileTaskCount; t++) {
            T sVal = sFinLocal_.GetValue(t);
            int64_t tokenBase = static_cast<int64_t>(taskOffset + t);
            for (int32_t cOff = 0; cOff < c_; cOff += GRADX_FIN_CHUNK) {
                int32_t len = min(static_cast<int64_t>(GRADX_FIN_CHUNK), c_ - cOff);
                // gradHin 逐 token 装载一次，n_ 行复用（[B,S,C] 布局）
                LocalTensor<U> ghU = gradHinFinQue_.AllocTensor<U>();
                DataCopyPad(ghU, gradHinGlobal_[tokenBase * c_ + cOff],
                            {static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(U)), 0, 0, 0},
                            {false, 0, 0, 0});
                gradHinFinQue_.EnQue(ghU);
                ghU = gradHinFinQue_.DeQue<U>();
                Cast(ghFinT_, ghU, RoundMode::CAST_NONE, len);
                gradHinFinQue_.FreeTensor(ghU);
                for (int32_t k = 0; k < n_; k++) {
                    T hPreVal = hPreFinLocal_.GetValue(t * n_ + k);
                    int64_t rowOffset = (tokenBase * n_ + k) * c_ + cOff;
                    LocalTensor<U> xU = xFinQue_.AllocTensor<U>();
                    DataCopyPad(xU, xGlobal_[rowOffset],
                                {static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(U)), 0, 0, 0},
                                {false, 0, 0, 0});
                    xFinQue_.EnQue(xU);
                    LocalTensor<T> mT = mFinQue_.AllocTensor<T>();
                    DataCopyPad(mT, gradXMWsGlobal_[rowOffset],
                                {static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0},
                                {false, 0, 0, 0});
                    mFinQue_.EnQue(mT);
                    xU = xFinQue_.DeQue<U>();
                    mT = mFinQue_.DeQue<T>();
                    LocalTensor<U> outU = gradXFinQue_.AllocTensor<U>();
                    Cast(xFinT_, xU, RoundMode::CAST_NONE, len);
                    Muls(xFinT_, xFinT_, sVal, len);
                    Muls(tmpFinT_, ghFinT_, hPreVal, len);
                    Add(xFinT_, xFinT_, tmpFinT_, len);
                    Add(xFinT_, xFinT_, mT, len);
                    // 与被替换路径的 castTrait2 一致：最近舍入（RINT）fp32 -> bf16
                    Cast(outU, xFinT_, RoundMode::CAST_RINT, len);
                    gradXFinQue_.EnQue(outU);
                    xFinQue_.FreeTensor(xU);
                    mFinQue_.FreeTensor(mT);
                    outU = gradXFinQue_.DeQue<U>();
                    DataCopyPad(gradXGlobal_[rowOffset], outU,
                                {static_cast<uint16_t>(1), static_cast<uint32_t>(len * sizeof(U)), 0, 0, 0});
                    gradXFinQue_.FreeTensor(outU);
                }
            }
        }
        // 下一 tile 的批量装载会覆盖 hPreFin/sFin：先确保本 tile 的标量读（S pipe）完成
        SetFlag<HardEvent::S_MTE2>(1);
        WaitFlag<HardEvent::S_MTE2>(1);
    }
}

// gradPhi = gradH2^T @ xFp32，A/B/C 的 GM 行主序行首 stride 均由 SetOrgShape 给出（orgN = n_ * c_），
// 本核只算/写 N 维切片 [nOffset, nOffset + curSliceN)，切片间不相交，无需原子加与清零；
// mm2SliceN_ 为 CeilDiv，不整除时尾核由 curSliceN 截断、越界核直接返回
// 小输出最终归并:blk0 读全部每核 partial,32 通道并行 Neumaier 补偿求和,
// 结果整块非原子写出(无原子舍入链、无跨核次序噪声;向量化消除逐通道标量读的固定开销)
template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornPremixBackwardOneStage<T, U>::ProcessSmallOutFinal()
{
    const int64_t partElems = aivNum_ * 32;
    DataCopyPad(smallPartLocal_, smallPartWsGlobal_,
                {static_cast<uint16_t>(1), static_cast<uint32_t>(partElems * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
    SetFlag<HardEvent::MTE2_V>(0);
    WaitFlag<HardEvent::MTE2_V>(0);

    __local_mem__ T* partPtr = (__local_mem__ T*)smallPartLocal_.GetPhyAddr();
    __local_mem__ T* outPtr = (__local_mem__ T*)smallOutLocal_.GetPhyAddr();
    __VEC_SCOPE__
    {
        RegTensor<T> sReg, cReg, pReg, tReg, e1Reg, e2Reg, tmpAReg, tmpBReg, absSReg, absPReg;
        MaskReg maskAll = CreateMask<T, MaskPattern::ALL>();
        MaskReg geMask;
        Duplicate(sReg, 0.f, maskAll);
        Duplicate(cReg, 0.f, maskAll);
        for (uint16_t core = 0; core < static_cast<uint16_t>(aivNum_); core++) {
            DataCopy(pReg, partPtr + core * 32);
            Add(tReg, sReg, pReg, maskAll);
            Abs(absSReg, sReg, maskAll);
            Abs(absPReg, pReg, maskAll);
            Compare<T, CMPMODE::GE>(geMask, absSReg, absPReg, maskAll);
            Sub(tmpAReg, sReg, tReg, maskAll);
            Add(e1Reg, tmpAReg, pReg, maskAll);
            Sub(tmpBReg, pReg, tReg, maskAll);
            Add(e2Reg, tmpBReg, sReg, maskAll);
            Select(e1Reg, e1Reg, e2Reg, geMask);
            Add(cReg, cReg, e1Reg, maskAll);
            Adds(sReg, tReg, 0.f, maskAll);
        }
        Add(sReg, sReg, cReg, maskAll);
        Store(outPtr, sReg, 32);
    }

    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    DataCopyPad(gradAlphaGlobal_, smallOutLocal_,
                {static_cast<uint16_t>(1), static_cast<uint32_t>(3 * sizeof(T)), 0, 0, 0});
    DataCopyPad(gradBiasGlobal_, smallOutLocal_[8],
                {static_cast<uint16_t>(1), static_cast<uint32_t>((2 * n_ + n_ * n_) * sizeof(T)), 0, 0, 0});
}

template <typename T, typename U>
__aicore__ inline void MhcPreSinkhornPremixBackwardOneStage<T, U>::ProcessMatmul2()
{
    const int64_t nOffset = blkIdx_ * mm2SliceN_;
    const int64_t curSliceN = min(mm2SliceN_, mm2N_ - nOffset);
    if (mm2K_ <= 0 || curSliceN <= 0)
        return;

    // 大 K(>512)两级 split-K:16 段部分和,段内 cube 累加、段间本核顺序原子合并。
    // 全 K 单级累加的误差约为 CPU SGEMM 的 2 倍(实测),两级归并可降至 ~0.9 倍;
    // 原子只发生在同一核心顺序执行的段间,不引入跨核次序噪声。小 K 保持单发直写。
    const int64_t kSplit = (mm2K_ > 512) ? (mm2K_ + 15) / 16 : mm2K_;
    mm2_.SetHF32(false, 1);
    mm2_.SetOrgShape(mm2M_, mm2N_, mm2K_);
    for (int64_t kOffset = 0; kOffset < mm2K_; kOffset += kSplit) {
        const int64_t kLen = min(kSplit, mm2K_ - kOffset);
        mm2_.SetTensorA(gradH2Global_[kOffset * mm2M_], true);
        mm2_.SetTensorB(xFp32Global_[kOffset * mm2N_ + nOffset]);
        mm2_.SetSingleShape(mm2M_, curSliceN, kLen);
        mm2_.template IterateAll<false>(gradPhiGlobal_[nOffset], kOffset == 0 ? 0 : 1);
        mm2_.End();
    }
}

#endif // MHC_PRE_SINKHORN_PREMIX_BACKWARD_ONE_STAGE_H
