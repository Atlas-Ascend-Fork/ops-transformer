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
 * \file kernel_dsa_sparse_prolog_vec_s1_cub_s2.h
 * \brief
 */

#ifndef KERNEL_DSA_SPARSE_PROLOG_VEC_S1_CUB_S2_H
#define KERNEL_DSA_SPARSE_PROLOG_VEC_S1_CUB_S2_H

#include "dsa_sparse_prolog_comm.h"
#include "service_matmul.h"
#include "service_rms_norm.h"
#include "service_gather_sin_cos.h"
#include "service_rotary_position_embedding.h"

namespace DsaSparseProlog {

// 标准矩阵乘法配置
constexpr auto GetMmCfgNorm(int sm, int sn, int sk, int bm, int bn, int bk)
{
    auto CFG = CFG_NORM;
    CFG.singleCoreM = sm;
    CFG.singleCoreN = sn;
    CFG.singleCoreK = sk;
    CFG.basicM = bm;
    CFG.basicN = bn;
    CFG.basicK = bk;
    CFG.enUnitFlag = false;

    CFG.enableSetBias = false;
    CFG.enableQuantVector = false;
    CFG.enableSetDefineData = false;
    return CFG;
}

constexpr auto GetMmCfgMdl(int sm, int sn, int sk, int bm, int bn, int bk)
{
    auto CFG = CFG_MDL;
    CFG.singleCoreM = sm;
    CFG.singleCoreN = sn;
    CFG.singleCoreK = sk;
    CFG.basicM = bm;
    CFG.basicN = bn;
    CFG.basicK = bk;
    CFG.enUnitFlag = false;

    CFG.enableSetBias = false;
    CFG.enableQuantVector = false;
    CFG.enableSetDefineData = false;
    return CFG;
}

constexpr static auto GetMMCqCkvKrTiling(const MatmulApiStaticTiling& mmTiling)
{
    auto tiling = mmTiling;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 4;
    tiling.stepKb = 4;
    tiling.depthA1 = 8;
    tiling.depthB1 = 8;
    return tiling;
}

constexpr static auto GetMMQcQrTiling(const MatmulApiStaticTiling& mmTiling)
{
    auto tiling = mmTiling;
    tiling.stepM = 1;
    tiling.stepN = 1;
    tiling.stepKa = 24;
    tiling.stepKb = 4;
    tiling.depthA1 = 24;
    tiling.depthB1 = 8;
    return tiling;
}

template <typename MLAPT>
class DsaSparsePrologVecS1CubS2 {
public:
    __aicore__ inline DsaSparsePrologVecS1CubS2(TPipe* pipe, const DsaSparsePrologTilingData* __restrict tilingData,
                                                const DsaSparsePrologBaseParams* __restrict baseParams)
        : pipe_(pipe),
          tilingData_(tilingData),
          baseParams_(baseParams)
    {}

    __aicore__ inline void Init(__gm__ uint8_t* tokenX, __gm__ uint8_t* weightDq, __gm__ uint8_t* weightUqQr,
                                __gm__ uint8_t* weightDkvKr, __gm__ uint8_t* rmsnormGammaCq,
                                __gm__ uint8_t* rmsnormGammaCkv, __gm__ uint8_t* ropeSin, __gm__ uint8_t* ropeCos,
                                __gm__ uint8_t* qNopeOut, __gm__ uint8_t* qPeOut, __gm__ uint8_t* ctKvOut,
                                __gm__ uint8_t* kPeOut, __gm__ uint8_t* qrOut, __gm__ uint8_t* qMOut,
                                __gm__ uint8_t* qRstdOut, __gm__ uint8_t* ckvMOut, __gm__ uint8_t* ckvRstdOut,
                                __gm__ uint8_t* workspace);
    __aicore__ inline void Process();
    __aicore__ inline void MMParamInit();
    __aicore__ inline void UpdateStepBatchParams(int64_t curStepBatchSize);

private:
    __aicore__ inline void MatmulCq(int64_t tokenXOffset, int64_t weightDqOffset, int64_t cqResOffset);
    __aicore__ inline void MatmulCkvKr(int64_t tokenXOffset, int64_t weightDkvKrOffset, int64_t ckvKrResOffset);
    __aicore__ inline void MatmulQcQr(int64_t weightUqQrOffset, int64_t qcQrResOffset);
    __aicore__ inline void MatmulQcQrWightPreload(int64_t weightUqQrOffset);
    __aicore__ inline void GetSinCos(int64_t tokenIndex, int64_t curVecToken);
    __aicore__ inline void RmsNormCq(int64_t tokenIndex, int64_t rmsNormCqOffset, int64_t curVecToken);
    __aicore__ inline void CopyQrOutput(int64_t rmsNormCqOffset, int64_t qrOutOffset, int64_t curVecToken);
    __aicore__ inline void GatherRstdOutput(GlobalTensor<float>& rstdOutGm, int64_t outOffset);
    __aicore__ inline void Scale(int64_t scaleQcQrOffset, int64_t qNopeOutOffset, int64_t curVecToken);
    __aicore__ inline void RmsNormRopeCkvKr(int64_t tokenIndex, int64_t rmsNormCkvOffset, int64_t ropeKrOffset,
                                            int64_t curVecToken);
    __aicore__ inline void RopeQr(int64_t ropeQrOffset, int64_t qPeResOffset, int64_t curVecToken);
    __aicore__ inline void ComputeAiVOffset(int64_t& curVecToken, int64_t& curBlockTokenOffset,
                                            int64_t& rmsNormCqOffset, int64_t& scaleQcQrOffset, int64_t& qNopeOutOffset,
                                            int64_t& rmsNormCkvOffset, int64_t& ropeKrOffset, int64_t& ropeQrOffset,
                                            int64_t& qPeResOffset, int64_t batchOffset);

public:
    using mmInputType = typename MLAPT::mmInputType;
    using mmQcQrInputType = typename MLAPT::mmQcQrInputType;
    using mmCqOutputType = typename MLAPT::mmCqOutputType;
    using mmCkvKrOutputType = typename MLAPT::mmCkvKrOutputType;
    using mmQcQrOutputType = typename MLAPT::mmQcQrOutputType;
    using rmsNormGammaType = typename MLAPT::rmsNormGammaType;
    using rmsNormComputType = typename MLAPT::rmsNormComputType;
    using rmsNormCqOutputType = typename MLAPT::rmsNormCqOutputType;
    using scaleQcQrOutputType = typename MLAPT::scaleQcQrOutputType;
    using scaleQcQrComputType = typename MLAPT::scaleQcQrComputType;
    using rmsNormCkvOutputType = typename MLAPT::rmsNormCkvOutputType;
    using ropeSinCosType = typename MLAPT::ropeSinCosType;
    using ropeComputType = typename MLAPT::ropeComputType;
    using ropeOutputType = typename MLAPT::ropeOutputType;
    using cacheType = typename MLAPT::cacheType;

    typedef MatmulType<TPosition::A1, CubeFormat::NZ, mmInputType, true> aCqCkvKrType;
    typedef MatmulType<TPosition::B1, CubeFormat::NZ, mmInputType, true> bCqCkvKrType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, mmCqOutputType> cCqCkvKrType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, mmInputType> biasCqCkvKrType;
    constexpr static auto CFGCqCkvKr = GetMmCfgMdl(128, 128, 256, 128, 128, 128);
    constexpr static auto mmCqCkvKrTiling =
        GetMatmulApiTiling<aCqCkvKrType, bCqCkvKrType, cCqCkvKrType, biasCqCkvKrType>(CFGCqCkvKr);
    constexpr static auto mmCqCkvKrTilingUpdate = GetMMCqCkvKrTiling(mmCqCkvKrTiling);
    using mmCqCkvKrType = MatmulImpl<aCqCkvKrType, bCqCkvKrType, cCqCkvKrType, biasCqCkvKrType,
                                     mmCqCkvKrTilingUpdate>; // 各个matmul需要调整配置

    typedef MatmulType<TPosition::A1, CubeFormat::NZ, mmQcQrInputType, true> aQcQrType;
    typedef MatmulType<TPosition::B1, CubeFormat::NZ, mmQcQrInputType, true> bQcQrType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, mmQcQrOutputType> cQcQrType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, mmQcQrInputType> biasQcQrType;
    constexpr static auto CFGQcQr = GetMmCfgMdl(128, 1024, 1536, 128, 256, 128);
    constexpr static auto mmQcQrTiling = GetMatmulApiTiling<aQcQrType, bQcQrType, cQcQrType, biasQcQrType>(CFGQcQr);
    constexpr static auto mmQcQrTilingUpdate = GetMMQcQrTiling(mmQcQrTiling);
    using mmQcQrType = MatmulImpl<aQcQrType, bQcQrType, cQcQrType, biasQcQrType, mmQcQrTilingUpdate>;

    mmCqCkvKrType mmCqCkvKr_;
    mmQcQrType mmQcQr_;

    MMParams mmCqParam_;
    MMParams mmCkvKrParam_;
    MMParams mmQcQrParam_;

private:
    TPipe* pipe_;
    const DsaSparsePrologTilingData* __restrict tilingData_;
    const DsaSparsePrologBaseParams* __restrict baseParams_;
    uint32_t blockIdx_ = 0U;
    int64_t vectorRow_ = 1;
    int64_t curVectorBlockNum_;
    int64_t vectorCoreNum_;
    uint32_t curStepVecFrontToken_;
    uint32_t curStepVecFrontListNum_;
    uint32_t curStepVecBackToken_;
    uint32_t curVecTokenMax_;

    // GM
    GlobalTensor<mmInputType> tokenXGm_;
    GlobalTensor<mmInputType> weightDqGm_;
    GlobalTensor<mmQcQrInputType> weightUqQrGm_;
    GlobalTensor<mmInputType> weightDkvKrGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCqGm_;
    GlobalTensor<rmsNormGammaType> rmsnormGammaCkvGm_;
    GlobalTensor<ropeSinCosType> ropeSinGm_;
    GlobalTensor<ropeSinCosType> ropeCosGm_;
    GlobalTensor<ropeOutputType> qPeOutGm_;     // q_pe: RoPE 后的 query rope 部分
    GlobalTensor<cacheType> qNopeOutGm_;        // q_nope: Scale 后的 query nope 部分
    GlobalTensor<cacheType> ctKvOutGm_;         // ct_kv: RmsNorm(Ckv) * mlaScaleKvLora
    GlobalTensor<cacheType> kPeOutGm_;          // k_pe: RoPE 后的 key rope 部分
    GlobalTensor<rmsNormCqOutputType> qrOutGm_; // qr: RmsNorm(Cq)（post-norm，供 Indexer）
    GlobalTensor<mmCqOutputType> qMOutGm_;      // q_m: q_a_proj 输出（RMSNorm 前），反向梯度中间量
    GlobalTensor<float> qRstdOutGm_;            // q_rstd: q_a_layernorm 的 rstd（fp32），反向梯度中间量
    GlobalTensor<mmCkvKrOutputType> ckvMOutGm_; // ckv_m: kv_a_proj 输出 split 的 ct_kv（RMSNorm 前），反向梯度中间量
    GlobalTensor<float> ckvRstdOutGm_; // ckv_rstd: kv_a_layernorm 的 rstd（fp32），反向梯度中间量

    GlobalTensor<rmsNormCqOutputType> rmsNormCqResGm_;
    GlobalTensor<scaleQcQrOutputType> scaleQcQrResGm_;
    GlobalTensor<mmCqOutputType> mmCqResGm_;
    GlobalTensor<mmCkvKrOutputType> mmCkvKrResGm_;
    GlobalTensor<mmQcQrOutputType> mmQcQrResGm_;
    GlobalTensor<float> rstdScratchGm_; // rstd 收集 scratch：每 AIV 核 8 fp32 (32B) 独占槽位

    // UB
    TQue<QuePosition::VECOUT, 1> rmsNormCqOutQueue_;
    TQue<QuePosition::VECOUT, 1> scaleQcQrOutQueue_;
    TQue<QuePosition::VECOUT, 1> ropeQrOutQueue_;
    TQue<QuePosition::VECOUT, 1> ckvKrOutQueue_;

    TBuf<TPosition::VECCALC> sincosBuffer_;
    TBuf<TPosition::VECCALC> shareBuffer_;
    TBuf<TPosition::VECCALC> rstdBuffer_; // rstd 单值暂存（UB 标量写 -> DataCopyPad 到 scratch）

    LocalTensor<float> rstdStagingUb_;

    LocalTensor<ropeComputType> cosLocal_;
    LocalTensor<ropeComputType> sinLocal_;

    TBuf<TPosition::A1> aBufL1_;
    TBuf<TPosition::B1> bBufL1_;
    LocalTensor<mmInputType> aL1Tensor_;
    LocalTensor<mmInputType> bL1Tensor_;
    MMBufParams bufParam_;
};

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::Init(
    __gm__ uint8_t* tokenX, __gm__ uint8_t* weightDq, __gm__ uint8_t* weightUqQr, __gm__ uint8_t* weightDkvKr,
    __gm__ uint8_t* rmsnormGammaCq, __gm__ uint8_t* rmsnormGammaCkv, __gm__ uint8_t* ropeSin, __gm__ uint8_t* ropeCos,
    __gm__ uint8_t* qNopeOut, __gm__ uint8_t* qPeOut, __gm__ uint8_t* ctKvOut, __gm__ uint8_t* kPeOut,
    __gm__ uint8_t* qrOut, __gm__ uint8_t* qMOut, __gm__ uint8_t* qRstdOut, __gm__ uint8_t* ckvMOut,
    __gm__ uint8_t* ckvRstdOut, __gm__ uint8_t* workspace)
{
    blockIdx_ = GetBlockIdx(); // cube:0-23  vec:0-47
    curVectorBlockNum_ = int64_t(baseParams_->stepBatchSize);
    vectorCoreNum_ = int64_t(baseParams_->vectorBlockNum); // aivNum 48
    curVecTokenMax_ = (curVectorBlockNum_ + vectorCoreNum_ - 1) / vectorCoreNum_;

    MMParamInit();

    // GM
    tokenXGm_.SetGlobalBuffer((__gm__ mmInputType*)tokenX);
    weightDqGm_.SetGlobalBuffer((__gm__ mmInputType*)weightDq);         // NZ
    weightUqQrGm_.SetGlobalBuffer((__gm__ mmQcQrInputType*)weightUqQr); // NZ
    weightDkvKrGm_.SetGlobalBuffer((__gm__ mmInputType*)weightDkvKr);   // NZ
    rmsnormGammaCqGm_.SetGlobalBuffer((__gm__ rmsNormGammaType*)rmsnormGammaCq);
    rmsnormGammaCkvGm_.SetGlobalBuffer((__gm__ rmsNormGammaType*)rmsnormGammaCkv);
    ropeSinGm_.SetGlobalBuffer((__gm__ ropeSinCosType*)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ ropeSinCosType*)ropeCos);
    qNopeOutGm_.SetGlobalBuffer((__gm__ cacheType*)qNopeOut);
    qPeOutGm_.SetGlobalBuffer((__gm__ ropeOutputType*)qPeOut);
    ctKvOutGm_.SetGlobalBuffer((__gm__ cacheType*)ctKvOut);
    kPeOutGm_.SetGlobalBuffer((__gm__ cacheType*)kPeOut);
    qrOutGm_.SetGlobalBuffer((__gm__ rmsNormCqOutputType*)qrOut);
    qMOutGm_.SetGlobalBuffer((__gm__ mmCqOutputType*)qMOut);
    qRstdOutGm_.SetGlobalBuffer((__gm__ float*)qRstdOut);
    ckvMOutGm_.SetGlobalBuffer((__gm__ mmCkvKrOutputType*)ckvMOut);
    ckvRstdOutGm_.SetGlobalBuffer((__gm__ float*)ckvRstdOut);

    // workspace
    int64_t workspaceOffset = 0;
    mmCqResGm_.SetGlobalBuffer((__gm__ mmCqOutputType*)workspace);                             // bf16   stepBatch * Hcq
    rmsNormCqResGm_.SetGlobalBuffer((__gm__ rmsNormCqOutputType*)workspace + workspaceOffset); // 复用 mmCqResGm_
    workspaceOffset +=
        int64_t(baseParams_->stepBatchSize) * int64_t(baseParams_->headSizeCq) * sizeof(rmsNormCqOutputType);
    mmCkvKrResGm_.SetGlobalBuffer((__gm__ mmCkvKrOutputType*)(workspace + workspaceOffset));

    workspaceOffset += int64_t(baseParams_->stepBatchSize) *
                       int64_t(baseParams_->headSizeCkv + baseParams_->dimHeadRope) * sizeof(mmCkvKrOutputType);
    scaleQcQrResGm_.SetGlobalBuffer((__gm__ scaleQcQrOutputType*)(workspace + workspaceOffset)); // 复用mmQcQrResGm_
    mmQcQrResGm_.SetGlobalBuffer(
        (__gm__ mmQcQrOutputType*)(workspace + workspaceOffset)); // bf16   stepBatch * N * (D+Dr)

    workspaceOffset += int64_t(baseParams_->stepBatchSize) * int64_t(baseParams_->numHeadSize) *
                       int64_t(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) * sizeof(mmQcQrOutputType);
    rstdScratchGm_.SetGlobalBuffer((__gm__ float*)(workspace + workspaceOffset)); // vectorBlockNum * 8 fp32

    // 初始化AIV相关缓冲区
    if ASCEND_IS_AIV {
        pipe_->InitBuffer(rmsNormCqOutQueue_, 1,
                          vectorRow_ * baseParams_->headSizeCq * sizeof(rmsNormCqOutputType)); // [1, Hcq] bf16
        pipe_->InitBuffer(scaleQcQrOutQueue_, 1,
                          vectorRow_ * (baseParams_->headSizeQc + baseParams_->headSizeQr) *
                              sizeof(scaleQcQrOutputType)); // [1, N*(D+Dr)] bf16
        pipe_->InitBuffer(ropeQrOutQueue_, 1,
                          baseParams_->headSizeQr * sizeof(ropeOutputType)); // [N, Dr] bf16
        pipe_->InitBuffer(ckvKrOutQueue_, 1,
                          vectorRow_ * baseParams_->headSizeCkv * sizeof(rmsNormCkvOutputType)); // [1, Hckv] bf16

        pipe_->InitBuffer(sincosBuffer_,
                          2 * baseParams_->dimHeadRope * sizeof(ropeComputType) * curVecTokenMax_); // [2, Dr] float
        pipe_->InitBuffer(shareBuffer_, baseParams_->ubBufferSize); // 除 sincos 外的 sharebuffer 共享，不会同时使用
        pipe_->InitBuffer(rstdBuffer_, 8 * sizeof(float));          // rstd 单值暂存（32B 对齐）

        rstdStagingUb_ = rstdBuffer_.Get<float>();

        cosLocal_ = sincosBuffer_.Get<ropeComputType>();
        sinLocal_ = cosLocal_[baseParams_->dimHeadRope * curVecTokenMax_];
    } else {
        pipe_->InitBuffer(aBufL1_, L1_A_SIZE * 2);
        pipe_->InitBuffer(bBufL1_, L1_B_SIZE * 2);
        mmCqCkvKr_.SetSubBlockIdx(0);
        mmCqCkvKr_.Init(&tilingData_->bmm1TilingData, pipe_);
        mmQcQr_.SetSubBlockIdx(0);
        mmQcQr_.Init(&tilingData_->bmm3TilingData, pipe_);

        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
        SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
        SetFlag<HardEvent::MTE1_MTE2>(B_EVENT1);
        aL1Tensor_ = aBufL1_.Get<mmInputType>();
        bL1Tensor_ = bBufL1_.Get<mmInputType>();
        bufParam_.aL1BufAddr = aBufL1_.GetBufferAddr(aL1Tensor_.GetBufferHandle());
        bufParam_.bL1BufAddr = bBufL1_.GetBufferAddr(bL1Tensor_.GetBufferHandle());
        ;
    }
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::MMParamInit()
{
    mmCqParam_.m = baseParams_->stepBatchSize; // 32
    if (blockIdx_ == baseParams_->mm1BlockNum - 1) {
        mmCqParam_.n = baseParams_->headSizeCq - baseParams_->mm1SingleCoreN * blockIdx_;
    } else {
        mmCqParam_.n = baseParams_->mm1SingleCoreN; // 1536 / 24 = 64
    }
    mmCqParam_.k = baseParams_->headSizeX; // 8192
    mmCqParam_.needSetOrgShape = 1;
    mmCqParam_.orgM = mmCqParam_.m;
    mmCqParam_.orgN = mmCqParam_.n;
    mmCqParam_.orgKa = mmCqParam_.k;
    mmCqParam_.orgKb = mmCqParam_.k;
    mmCqParam_.orgKc = baseParams_->headSizeCq; // 1536
    mmCqParam_.orgM = mmCqParam_.m;
    mmCqParam_.baseK = 128;
    mmCqParam_.baseN = 128;
    mmCqParam_.stepK = 4;
    mmCqParam_.bLayout = baseParams_->weightLayout;

    mmCkvKrParam_.m = baseParams_->stepBatchSize; // 32
    if (blockIdx_ == baseParams_->mm2BlockNum - 1) {
        mmCkvKrParam_.n = baseParams_->headSizeCkv + baseParams_->headSizeKr - baseParams_->mm2SingleCoreN * blockIdx_;
    } else {
        mmCkvKrParam_.n = baseParams_->mm2SingleCoreN;
    }
    mmCkvKrParam_.k = baseParams_->headSizeX; // 8192
    mmCkvKrParam_.needSetOrgShape = 1;
    mmCkvKrParam_.orgM = mmCkvKrParam_.m;
    mmCkvKrParam_.orgN = mmCkvKrParam_.n;
    mmCkvKrParam_.orgKa = mmCkvKrParam_.k;
    mmCkvKrParam_.orgKb = mmCkvKrParam_.k;
    mmCkvKrParam_.orgKc = (baseParams_->headSizeCkv + baseParams_->dimHeadRope); // 576
    mmCkvKrParam_.baseK = 128;
    mmCkvKrParam_.baseN = 128;
    mmCkvKrParam_.stepK = 4;
    mmCkvKrParam_.bLayout = baseParams_->weightLayout;

    mmQcQrParam_.m = baseParams_->stepBatchSize; // 32
    if (blockIdx_ == baseParams_->mm3BlockNum - 1) {
        mmQcQrParam_.n = baseParams_->headSizeQc + baseParams_->headSizeQr - baseParams_->mm3SingleCoreN * blockIdx_;
    } else {
        mmQcQrParam_.n = baseParams_->mm3SingleCoreN;
    }
    mmQcQrParam_.k = baseParams_->headSizeCq; // 1536
    mmQcQrParam_.needSetOrgShape = 1;
    mmQcQrParam_.orgM = mmQcQrParam_.m;
    mmQcQrParam_.orgN = mmQcQrParam_.n;
    mmQcQrParam_.orgKa = mmQcQrParam_.k;
    mmQcQrParam_.orgKb = mmQcQrParam_.k;
    mmQcQrParam_.orgKc = (baseParams_->headSizeQc + baseParams_->headSizeQr); // (D * N + Dr * N)
    mmQcQrParam_.baseK = 64;
    mmQcQrParam_.baseN = 256;
    mmQcQrParam_.stepK = 4;
    mmQcQrParam_.bLayout = baseParams_->weightLayout;
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::UpdateStepBatchParams(int64_t curStepBatchSize)
{
    mmCqParam_.m = curStepBatchSize;
    mmCkvKrParam_.m = curStepBatchSize;
    mmQcQrParam_.m = curStepBatchSize;
    curVectorBlockNum_ = curStepBatchSize;
    mmCqParam_.orgM = curStepBatchSize;
    mmCkvKrParam_.orgM = curStepBatchSize;
    mmQcQrParam_.orgM = curStepBatchSize;
}

// DsaSparseProlog算子计算流程（吸收模式，无 weight_uk / MatmulQn）
// token_x ──> MatmulCq ──> RmsNorm(Cq) ──> MatmulQcQr ──> Scale ──> q_nope_out（按头抽前 D 列）
//        │                       │                                   └──> Rope(Qr) ──> q_pe_out
//        │                       └──> qr_out（post-norm，供 Indexer）
//        └──> MatmulCkvKr ──> RmsNorm(Ckv)*scale ──> ct_kv_out
//                        └──> Rope(Kr) ──> k_pe_out
template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::Process()
{
    // AIC的offset参数 (按 weightLayout 4 布局)
    int64_t weightDqOffset;
    int64_t weightDkvKrOffset;
    int64_t weightUqQrOffset;
    if (baseParams_->weightLayout == B_LAYOUT_ND_KN) {
        // ND [K,N] 行主序: 每核列条偏移 = singleCoreN * blockIdx (元素)
        weightDqOffset = int64_t(baseParams_->mm1SingleCoreN) * blockIdx_;
        weightDkvKrOffset = int64_t(baseParams_->mm2SingleCoreN) * blockIdx_;
        weightUqQrOffset = int64_t(baseParams_->mm3SingleCoreN) * blockIdx_;
    } else if (baseParams_->weightLayout == B_LAYOUT_NZ_NK) {
        // NZ of [N,K]: 每核 n1 块偏移 = singleCoreN * 16 * blockIdx (元素)
        weightDqOffset = int64_t(baseParams_->mm1SingleCoreN) * 16 * blockIdx_;
        weightDkvKrOffset = int64_t(baseParams_->mm2SingleCoreN) * 16 * blockIdx_;
        weightUqQrOffset = int64_t(baseParams_->mm3SingleCoreN) * 16 * blockIdx_;
    } else if (baseParams_->weightLayout == B_LAYOUT_ND_NK) {
        // ND [N,K] 行主序: 每核行偏移 = singleCoreN * K * blockIdx (元素)
        weightDqOffset = int64_t(baseParams_->mm1SingleCoreN) * blockIdx_ * baseParams_->headSizeX;
        weightDkvKrOffset = int64_t(baseParams_->mm2SingleCoreN) * blockIdx_ * baseParams_->headSizeX;
        weightUqQrOffset = int64_t(baseParams_->mm3SingleCoreN) * blockIdx_ * baseParams_->headSizeCq;
    } else {
        weightDqOffset = int64_t(baseParams_->headSizeX) * baseParams_->mm1SingleCoreN * blockIdx_; // 8192 * 64 * idx
        weightDkvKrOffset =
            int64_t(baseParams_->headSizeX) * baseParams_->mm2SingleCoreN * blockIdx_; // 8192 * 64 * idx
        weightUqQrOffset =
            int64_t(baseParams_->headSizeCq) * baseParams_->mm3SingleCoreN * blockIdx_; // 1536 * 512 * idx
    }
    int64_t cqResOffset = baseParams_->mm1SingleCoreN * blockIdx_;    // 64 * idx
    int64_t ckvKrResOffset = baseParams_->mm2SingleCoreN * blockIdx_; // 64 * idx
    int64_t qcQrResOffset = baseParams_->mm3SingleCoreN * blockIdx_;  // 512 * idx

    // AIV的offset参数
    int64_t curVecToken = blockIdx_ < curStepVecFrontListNum_ ? curStepVecFrontToken_ : curStepVecBackToken_;
    int64_t curBlockTokenOffset = blockIdx_ < curStepVecFrontListNum_ ?
                                      blockIdx_ * curVecToken :
                                      blockIdx_ * curVecToken + curStepVecFrontListNum_;

    int64_t rmsNormCqOffset;
    int64_t scaleQcQrOffset;
    int64_t qNopeOutOffset;
    int64_t rmsNormCkvOffset;
    int64_t ropeKrOffset;
    int64_t ropeQrOffset;
    int64_t qPeResOffset;
    ComputeAiVOffset(curVecToken, curBlockTokenOffset, rmsNormCqOffset, scaleQcQrOffset, qNopeOutOffset,
                     rmsNormCkvOffset, ropeKrOffset, ropeQrOffset, qPeResOffset, 0);
    int64_t bsSize = int64_t(baseParams_->tokenSize);
    // 需要考虑BS合轴的尾块情况
    for (int64_t batchOffset = 0; batchOffset < bsSize; batchOffset += baseParams_->stepBatchSize) {
        if (bsSize - batchOffset < baseParams_->stepBatchSize) {
            UpdateStepBatchParams(bsSize - batchOffset);
            if ((bsSize - batchOffset) != baseParams_->stepBatchSize) {
                ComputeAiVOffset(curVecToken, curBlockTokenOffset, rmsNormCqOffset, scaleQcQrOffset, qNopeOutOffset,
                                 rmsNormCkvOffset, ropeKrOffset, ropeQrOffset, qPeResOffset, batchOffset);
            }
        }
        bool hasNextStep = batchOffset + baseParams_->stepBatchSize < bsSize;
        if ASCEND_IS_AIC {
            int64_t tokenXOffset = batchOffset * baseParams_->headSizeX;
            MatmulCq(tokenXOffset, weightDqOffset, cqResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x6);
            MatmulCkvKr(tokenXOffset, weightDkvKrOffset, ckvKrResOffset);
            MatmulQcQrWightPreload(weightUqQrOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x7);
            // wait RmsNormCq
            // MatmulQcQr依赖RmsNormCq的输出，需要插入CV核间同步
            CrossCoreWaitFlag(0x6);
            MatmulQcQr(weightUqQrOffset, qcQrResOffset);
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
            if (hasNextStep) {
                // q_nope/q_pe 的 workspace 会在下一个 step 复用。等待 vector
                // 完成本 step 的 Scale/RoPE 读取，避免 cube 提前覆盖数据。
                CrossCoreWaitFlag(0xA);
            }
        }
        if ASCEND_IS_AIV {
            int64_t tokenIndex = batchOffset + curBlockTokenOffset;
            GetSinCos(tokenIndex, curVecToken);

            // wait MatmulCq
            CrossCoreWaitFlag(0x6);
            // wait all vector
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x9);
            CrossCoreWaitFlag(0x9);
            RmsNormCq(tokenIndex, rmsNormCqOffset, curVecToken);
            int64_t qrOutOffset = (batchOffset + curBlockTokenOffset) * int64_t(baseParams_->headSizeCq);
            CopyQrOutput(rmsNormCqOffset, qrOutOffset, curVecToken);
            // wait all cube
            // 由于RmsNormCq和MatmulQcQr的分核策略不一样，需要等所有vector上的RmsNormCq执行完成后才能启动MatmulQcQr
            // 需要所有vector核上的RmsNormCq执行完成后，才发起MatmulQcQr的执行
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x9);
            CrossCoreWaitFlag(0x9);
            // 保障MatmulQcQr等RmsNormCq（先放行 cube，再做 rstd 汇总，gather 不在 cube 关键路径上）
            CrossCoreSetFlag<0x2, PIPE_MTE3>(0x6);
            // 汇总 q_rstd（scratch 各核独占槽位已在 barrier 前写完）
            GatherRstdOutput(qRstdOutGm_, batchOffset);

            // wait MatmulCkvKr
            CrossCoreWaitFlag(0x7);
            // wait all vector
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x9);
            CrossCoreWaitFlag(0x9);
            RmsNormRopeCkvKr(tokenIndex, rmsNormCkvOffset, ropeKrOffset, curVecToken);

            // wait MatmulQcQr
            CrossCoreWaitFlag(0x8);
            // wait all vector
            CrossCoreSetFlag<0x0, PIPE_MTE3>(0x9);
            CrossCoreWaitFlag(0x9);
            // 汇总 ckv_rstd
            GatherRstdOutput(ckvRstdOutGm_, batchOffset);
            Scale(scaleQcQrOffset, qNopeOutOffset, curVecToken);
            // Scale 通过 MTE3 回写 scaleQcQrResGm_，RopeQr 通过 MTE2 读取，需要同核 MTE3->MTE2 同步
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            RopeQr(ropeQrOffset, qPeResOffset, curVecToken);
            if (hasNextStep) {
                CrossCoreSetFlag<0x2, PIPE_MTE3>(0xA);
            }
            qNopeOutOffset += int64_t(baseParams_->stepBatchSize) * int64_t(baseParams_->headSizeQc);
            qPeResOffset += int64_t(baseParams_->stepBatchSize) * int64_t(baseParams_->headSizeQr);
        }
    }
    if ASCEND_IS_AIC {
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT1);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0);
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT1);
    }
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::ComputeAiVOffset(
    int64_t& curVecToken, int64_t& curBlockTokenOffset, int64_t& rmsNormCqOffset, int64_t& scaleQcQrOffset,
    int64_t& qNopeOutOffset, int64_t& rmsNormCkvOffset, int64_t& ropeKrOffset, int64_t& ropeQrOffset,
    int64_t& qPeResOffset, int64_t batchOffset)
{
    curStepVecBackToken_ = curVectorBlockNum_ / uint32_t(vectorCoreNum_);
    curStepVecFrontListNum_ = curVectorBlockNum_ % uint32_t(vectorCoreNum_);
    curStepVecFrontToken_ = curStepVecFrontListNum_ == 0 ? curStepVecBackToken_ : curStepVecBackToken_ + 1;

    curVecToken = blockIdx_ < curStepVecFrontListNum_ ? curStepVecFrontToken_ : curStepVecBackToken_;
    curBlockTokenOffset = blockIdx_ < curStepVecFrontListNum_ ? blockIdx_ * curVecToken :
                                                                blockIdx_ * curVecToken + curStepVecFrontListNum_;
    rmsNormCqOffset = baseParams_->headSizeCq * curBlockTokenOffset; //  1536 * idx
    scaleQcQrOffset = (baseParams_->headSizeQc + baseParams_->headSizeQr) * curBlockTokenOffset;
    qNopeOutOffset = int64_t(baseParams_->headSizeQc) * (curBlockTokenOffset + batchOffset); // q_nope 全局输出偏移
    rmsNormCkvOffset = (baseParams_->headSizeCkv + baseParams_->dimHeadRope) * curBlockTokenOffset; // (512 + 64) * idx
    ropeKrOffset = baseParams_->headSizeCkv + (baseParams_->headSizeCkv + baseParams_->dimHeadRope) *
                                                  curBlockTokenOffset; // 512 + (512 + 64) * idx
    ropeQrOffset = baseParams_->dimHeadSizeQc + int64_t(baseParams_->numHeadSize) *
                                                    int64_t(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope) *
                                                    curBlockTokenOffset; // 128 + 64 * (128 + 64) * idx
    qPeResOffset =
        int64_t(baseParams_->headSizeQr) * (curBlockTokenOffset + batchOffset); // q_pe 全局输出偏移，按BS合轴切分step
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::MatmulCq(int64_t tokenXOffset, int64_t weightDqOffset,
                                                                  int64_t cqResOffset)
{
    if (blockIdx_ >= baseParams_->mm1BlockNum) {
        return;
    }
    // MatmulCq ──> RmsNorm(Cq)
    // [stepBatch, 8192] * [8192, 1536] = [stepBatch, 1536]
    MatmulImplNormal<mmInputType, mmCqOutputType, mmCqCkvKrType>(mmCqCkvKr_, tokenXGm_[tokenXOffset],
                                                                 weightDqGm_[weightDqOffset], mmCqResGm_[cqResOffset],
                                                                 mmCqParam_, &bufParam_, blockIdx_);
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::MatmulCkvKr(int64_t tokenXOffset, int64_t weightDkvKrOffset,
                                                                     int64_t ckvKrResOffset)
{
    if (blockIdx_ >= baseParams_->mm2BlockNum) {
        return;
    }
    // MatmulCkvKr ──> RmsNorm(Ckv)
    //            └──> Rope(Kr)
    // [stepBatch, 8192] * [8192, 512+64] = [stepBatch, 576]
    MatmulImplNormal<mmInputType, mmCkvKrOutputType, mmCqCkvKrType>(
        mmCqCkvKr_, tokenXGm_[tokenXOffset], weightDkvKrGm_[weightDkvKrOffset], mmCkvKrResGm_[ckvKrResOffset],
        mmCkvKrParam_, &bufParam_);
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::MatmulQcQrWightPreload(int64_t weightUqQrOffset)
{
    if (blockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }
    QcQrWeightPreload<rmsNormCqOutputType, mmQcQrOutputType, mmQcQrType>(weightUqQrGm_[weightUqQrOffset], mmQcQrParam_,
                                                                         &bufParam_);
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::MatmulQcQr(int64_t weightUqQrOffset, int64_t qcQrResOffset)
{
    if (blockIdx_ >= baseParams_->mm3BlockNum) {
        return;
    }
    // RmsNorm(Cq) ──> MatmulQcQr ──> Scale ──> q_nope_out
    //                          └──> Rope(Qr) ──> q_pe_out
    // [stepBatch, 1536] * [1536, 64*(128+64)] = [stepBatch, 64*192]
    constexpr uint32_t mSize = 32;
    bool isAFullLoad = (mmQcQrParam_.m <= mSize) ? true : false;
    if (isAFullLoad) {
        MatmulImplQcQr<rmsNormCqOutputType, mmQcQrOutputType, mmQcQrType>(
            mmQcQr_, rmsNormCqResGm_, // 复用mmCqResGm_(即rmsNormCqResGm_) workspace
            weightUqQrGm_[weightUqQrOffset], mmQcQrResGm_[qcQrResOffset], mmQcQrParam_, &bufParam_);
    } else {
        MatmulImplNormalPreload<rmsNormCqOutputType, mmQcQrOutputType, mmQcQrType>(
            mmQcQr_, rmsNormCqResGm_, // 复用mmCqResGm_（即rmsNormCqResGm_） workspace
            weightUqQrGm_[weightUqQrOffset], mmQcQrResGm_[qcQrResOffset], mmQcQrParam_, &bufParam_);
    }
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::GetSinCos(int64_t tokenIndex, int64_t curVecToken)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
    // [1, 64]
    GatherSinCos<ropeSinCosType, ropeComputType>(ropeCosGm_, ropeSinGm_, tokenIndex, curVecToken, shareTmpUb,
                                                 vectorRow_, baseParams_->dimHeadRope, cosLocal_, sinLocal_);
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::RmsNormCq(int64_t tokenIndex, int64_t rmsNormCqOffset,
                                                                   int64_t curVecToken)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    int64_t headSizeCq = baseParams_->headSizeCq;
    LocalTensor<mmCqOutputType> qMTmpBuf = shareBuffer_.Get<mmCqOutputType>();
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulCq ──> RmsNorm(Cq) ──> MatmulQcQr
        // RmsNormCq 内部不乘 scale（qr 为 post-norm，s_q 仅在 Scale 中对 QcQr 乘一次）
        float scale = baseParams_->mlaScaleQLora;
        LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
        LocalTensor<rmsNormCqOutputType> outputLocal = rmsNormCqOutQueue_.AllocTensor<rmsNormCqOutputType>();
        // 等上一轮 q_m 的 MTE3 写完 qMTmpBuf（复用 shareBuffer_）
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        // [1, 1536]，rstd 由返回值带出（fp32 标量）
        float rstdValue =
            ::DsaSparseProlog::RmsNormCq<mmCqOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCqOutputType>(
                mmCqResGm_[rmsNormCqOffset], rmsnormGammaCqGm_, baseParams_->reciprocalCq, baseParams_->epsilonCq,
                scale, vectorRow_, headSizeCq, shareTmpUb, outputLocal, baseParams_->isTypeDemotion);
        rstdStagingUb_.SetValue(curVecTokenIdx, rstdValue);

        // q_m: mmCqResGm_ 与 rmsNormCqResGm_ 共享 workspace，需在回写 norm 结果前把 pre-norm matmul 输出搬到 q_m
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        DataCopy(qMTmpBuf, mmCqResGm_[rmsNormCqOffset], headSizeCq);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(qMOutGm_[tokenIndex * headSizeCq], qMTmpBuf, headSizeCq);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);

        rmsNormCqOutQueue_.EnQue(outputLocal);
        outputLocal = rmsNormCqOutQueue_.DeQue<rmsNormCqOutputType>();
        // RmsNorm(Cq)的结果拷进mmCqResGm_中，用于MatmulQcQr的A矩阵
        DataCopy(rmsNormCqResGm_[rmsNormCqOffset], outputLocal, headSizeCq);
        rmsNormCqOutQueue_.FreeTensor(outputLocal);
        rmsNormCqOffset += headSizeCq;
        tokenIndex += 1;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    // q_rstd: UB 标量 -> scratch 独占槽位（每核 8 fp32 = 32B 对齐），由 0 号核在 barrier 后汇总写出
    SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
    DataCopyExtParams rstdCopyParams{1, static_cast<uint32_t>(curVecToken * sizeof(float)), 0, 0, 0};
    DataCopyPad(rstdScratchGm_[blockIdx_ * 8], rstdStagingUb_, rstdCopyParams);
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::CopyQrOutput(int64_t rmsNormCqOffset, int64_t qrOutOffset,
                                                                      int64_t curVecToken)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    LocalTensor<rmsNormCqOutputType> tmpBuf = shareBuffer_.Get<rmsNormCqOutputType>();
    int64_t headSizeCq = baseParams_->headSizeCq;
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    for (int64_t t = 0; t < curVecToken; t++) {
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        DataCopy(tmpBuf, rmsNormCqResGm_[rmsNormCqOffset], headSizeCq);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(qrOutGm_[qrOutOffset], tmpBuf, headSizeCq);
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        rmsNormCqOffset += headSizeCq;
        qrOutOffset += headSizeCq;
    }
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::GatherRstdOutput(GlobalTensor<float>& rstdOutGm,
                                                                          int64_t outOffset)
{
    // 单核汇总写出 rstd。stepBatchSize(min(128, T)) 个 token 按 ComputeAiVOffset 块连续分到 48 核
    // （前 front 核 vFront 个、其余 vBack 个，每核 <=3 <= 槽位容量 8），
    // scratch 槽位 blockIdx*8 fp32（32B 对齐、核间独占，避免标量写 GM 的 cache line 竞争），
    // 目标 outOffset=batchOffset 为 stepBatchSize 的倍数（128B 对齐），DataCopyPad 一次写出
    if (blockIdx_ != 0) {
        return;
    }
    int64_t count = curVectorBlockNum_;
    LocalTensor<float> gatherUb = shareBuffer_.Get<float>();
    DataCopy(gatherUb, rstdScratchGm_, vectorCoreNum_ * 8); // 48*8 fp32 = 1536B
    SetFlag<HardEvent::MTE2_S>(EVENT_ID2);
    WaitFlag<HardEvent::MTE2_S>(EVENT_ID2);
    // 压实到独立目的区（非原地）。token->核映射与 ComputeAiVOffset 一致：
    // 前 front 核各 vFront 个 token（块连续），其余核各 vBack 个；槽位 k 内第 j 个 = 该核第 j 个 token。
    // 核序遍历 + 游标递增，避免逐元素 int64 div/mod
    LocalTensor<float> compactUb = gatherUb[vectorCoreNum_ * 8];
    int64_t front = curStepVecFrontListNum_;
    int64_t vFront = curStepVecFrontToken_;
    int64_t vBack = curStepVecBackToken_;
    int64_t cursor = 0;
    for (int64_t k = 0; k < front; k++) {
        for (int64_t j = 0; j < vFront; j++) {
            compactUb.SetValue(cursor, gatherUb.GetValue(k * 8 + j));
            cursor++;
        }
    }
    // count = front*vFront + backCores*vBack；vBack=0 时无后部核（count<48 全在前部）
    int64_t backCores = vBack > 0 ? (count - front * vFront) / vBack : 0;
    for (int64_t k = 0; k < backCores; k++) {
        for (int64_t j = 0; j < vBack; j++) {
            compactUb.SetValue(cursor, gatherUb.GetValue((front + k) * 8 + j));
            cursor++;
        }
    }
    SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPad(rstdOutGm[outOffset], compactUb, copyParams);
    // shareBuffer_ 后续被 MTE2 复用，等本次 MTE3 读完
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::Scale(int64_t scaleQcQrOffset, int64_t qNopeOutOffset,
                                                               int64_t curVecToken)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    // q_nope 从 [1, N*(D+Dr)] 中按头抽取前 D 列，stride 写 [T, N, D]
    DataCopyParams qNopeCopyParams{static_cast<uint16_t>(baseParams_->numHeadSize),
                                   static_cast<uint16_t>(baseParams_->dimHeadSizeQc * sizeof(scaleQcQrOutputType) / 32),
                                   static_cast<uint16_t>(baseParams_->dimHeadRope * sizeof(scaleQcQrOutputType) / 32),
                                   0};
    int64_t headSizeQcQr = baseParams_->headSizeQc + baseParams_->headSizeQr;
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulQcQr  ──> Scale(QcQr)  ──> q_nope_out
        float scale = baseParams_->mlaScaleQLora;
        LocalTensor<uint8_t> shareTmpUb = shareBuffer_.Get<uint8_t>();
        LocalTensor<scaleQcQrOutputType> outputLocal = scaleQcQrOutQueue_.AllocTensor<scaleQcQrOutputType>();
        // [1, N*(D+Dr)]
        ::DsaSparseProlog::Scale<mmQcQrOutputType, scaleQcQrComputType, scaleQcQrOutputType>(
            mmQcQrResGm_[scaleQcQrOffset], scale, vectorRow_, headSizeQcQr, shareTmpUb, outputLocal);

        scaleQcQrOutQueue_.EnQue(outputLocal);
        outputLocal = scaleQcQrOutQueue_.DeQue<scaleQcQrOutputType>();
        // scale(QcQr)的结果拷进mmQcQrResGm_中，用于RopeQr读取 Qr 部分
        DataCopy(scaleQcQrResGm_[scaleQcQrOffset], outputLocal, headSizeQcQr);
        // q_nope: 每个头取前 D 列，目的端连续
        DataCopy(qNopeOutGm_[qNopeOutOffset], outputLocal, qNopeCopyParams);
        scaleQcQrOutQueue_.FreeTensor(outputLocal);
        scaleQcQrOffset += headSizeQcQr;
        qNopeOutOffset += baseParams_->headSizeQc;
    }
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::RmsNormRopeCkvKr(int64_t tokenIndex, int64_t rmsNormCkvOffset,
                                                                          int64_t ropeKrOffset, int64_t curVecToken)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    int64_t headSizeCkv = baseParams_->headSizeCkv;
    LocalTensor<mmCkvKrOutputType> ckvMTmpBuf = shareBuffer_.Get<mmCkvKrOutputType>();
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        LocalTensor<rmsNormCkvOutputType> outputLocal = ckvKrOutQueue_.AllocTensor<rmsNormCkvOutputType>();
        // RmsNorm(Ckv) * mlaScaleKvLora
        // MatmulCkvKr ──> RmsNorm(Ckv) ──> ct_kv_out 由于当前只处理1行数据，不需要stride
        float scale = baseParams_->mlaScaleKvLora;
        LocalTensor<uint8_t> rmsNormShareTmpUb = shareBuffer_.Get<uint8_t>();
        // rstd 由返回值带出（fp32 标量）
        float rstdValue = RmsNorm<mmCkvKrOutputType, rmsNormGammaType, rmsNormComputType, rmsNormCkvOutputType>(
            mmCkvKrResGm_[rmsNormCkvOffset], rmsnormGammaCkvGm_, baseParams_->reciprocalCkv, baseParams_->epsilonCkv,
            scale, vectorRow_, headSizeCkv, rmsNormShareTmpUb, outputLocal, baseParams_->isTypeDemotion);
        rstdStagingUb_.SetValue(curVecTokenIdx, rstdValue);

        // ckv_m: pre-norm 的 ct_kv（kv_a_proj 输出 split 的前 Hckv 列），搬到 ckv_m 输出
        SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
        DataCopy(ckvMTmpBuf, mmCkvKrResGm_[rmsNormCkvOffset], headSizeCkv);
        SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        DataCopy(ckvMOutGm_[tokenIndex * headSizeCkv], ckvMTmpBuf, headSizeCkv);

        ckvKrOutQueue_.EnQue(outputLocal);
        outputLocal = ckvKrOutQueue_.DeQue<rmsNormCkvOutputType>();
        // RmsNorm(Ckv) ──> ct_kv_out [T, Hckv]
        DataCopy(ctKvOutGm_[tokenIndex * int64_t(baseParams_->headSizeCkv)], outputLocal, baseParams_->headSizeCkv);
        ckvKrOutQueue_.FreeTensor(outputLocal);

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        outputLocal = ckvKrOutQueue_.AllocTensor<rmsNormCkvOutputType>();
        // Rope(Kr)
        // MatmulCkvKr ──> Rope(Kr) ──> k_pe_out
        LocalTensor<uint8_t> ropeShareTmpUb = shareBuffer_.Get<uint8_t>();
        int64_t stride = baseParams_->headSizeCkv + baseParams_->headSizeKr; // 512 + 64
        if (baseParams_->applyRopeFusion) {
            RotaryPosEmb_noQuant_rope_fusion<mmCkvKrOutputType, ropeComputType, rmsNormCkvOutputType>(
                mmCkvKrResGm_[ropeKrOffset], cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx],
                sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], stride, vectorRow_, baseParams_->dimHeadRope,
                ropeShareTmpUb, outputLocal);
        } else {
            RotaryPosEmb_noQuant<mmCkvKrOutputType, ropeComputType, rmsNormCkvOutputType>(
                mmCkvKrResGm_[ropeKrOffset], cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx],
                sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], stride, vectorRow_, baseParams_->dimHeadRope,
                ropeShareTmpUb, outputLocal, GlobalTensor<ropeComputType>(), baseParams_->isTypeDemotion);
        }

        ckvKrOutQueue_.EnQue(outputLocal);
        outputLocal = ckvKrOutQueue_.DeQue<rmsNormCkvOutputType>();
        // Rope(Kr) ──> k_pe_out [T, 1, Dr]（MQA 单头，连续存储）
        DataCopy(kPeOutGm_[tokenIndex * int64_t(baseParams_->dimHeadRope)], outputLocal, baseParams_->dimHeadRope);
        ckvKrOutQueue_.FreeTensor(outputLocal);

        tokenIndex += 1;
        rmsNormCkvOffset += baseParams_->headSizeCkv + baseParams_->dimHeadRope;
        ropeKrOffset += baseParams_->headSizeCkv + baseParams_->dimHeadRope;
    }
    // ckv_rstd: UB 标量 -> scratch 独占槽位（与 q_rstd 分时复用，中间隔了 0x9 barrier），由 0 号核汇总写出
    SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
    DataCopyExtParams rstdCopyParams{1, static_cast<uint32_t>(curVecToken * sizeof(float)), 0, 0, 0};
    DataCopyPad(rstdScratchGm_[blockIdx_ * 8], rstdStagingUb_, rstdCopyParams);
}

template <typename MLAPT>
__aicore__ inline void DsaSparsePrologVecS1CubS2<MLAPT>::RopeQr(int64_t ropeQrOffset, int64_t qPeResOffset,
                                                                int64_t curVecToken)
{
    if (blockIdx_ >= curVectorBlockNum_) {
        return;
    }
    for (int64_t curVecTokenIdx = 0; curVecTokenIdx < curVecToken; curVecTokenIdx++) {
        // MatmulQcQr ──> Scale ──> Rope(Qr) ──> q_pe_out
        LocalTensor<uint8_t> ropeShareTmpUb = shareBuffer_.Get<uint8_t>();
        LocalTensor<ropeOutputType> outputLocal = ropeQrOutQueue_.AllocTensor<ropeOutputType>();
        int64_t stride = baseParams_->dimHeadRope + baseParams_->dimHeadSizeQc;
        if (baseParams_->applyRopeFusion) {
            RotaryPosEmb_noQuant_rope_fusion<mmQcQrOutputType, ropeComputType, ropeOutputType>(
                mmQcQrResGm_[ropeQrOffset], cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx],
                sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], stride, baseParams_->numHeadSize,
                baseParams_->dimHeadRope, ropeShareTmpUb, outputLocal);
        } else {
            RotaryPosEmb_noQuant<mmQcQrOutputType, ropeComputType, ropeOutputType>(
                mmQcQrResGm_[ropeQrOffset], cosLocal_[baseParams_->dimHeadRope * curVecTokenIdx],
                sinLocal_[baseParams_->dimHeadRope * curVecTokenIdx], stride, baseParams_->numHeadSize,
                baseParams_->dimHeadRope, ropeShareTmpUb, outputLocal, GlobalTensor<ropeComputType>(),
                baseParams_->isTypeDemotion);
        }

        ropeQrOutQueue_.EnQue(outputLocal);
        outputLocal = ropeQrOutQueue_.DeQue<ropeOutputType>();
        // [N, Dr]
        DataCopy(qPeOutGm_[qPeResOffset], outputLocal, baseParams_->headSizeQr);
        ropeQrOutQueue_.FreeTensor(outputLocal);

        ropeQrOffset +=
            int64_t(baseParams_->numHeadSize) * int64_t(baseParams_->dimHeadSizeQc + baseParams_->dimHeadRope);
        qPeResOffset += int64_t(baseParams_->headSizeQr);
    }
}

} // namespace DsaSparseProlog

#endif // DSA_SPARSE_PROLOG_VEC_S1_CUB_S2_H
