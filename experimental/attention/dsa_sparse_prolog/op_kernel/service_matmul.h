/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* *
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file service_matmul.h
 * \brief
 */

#ifndef SERVICE_MATMUL_H
#define SERVICE_MATMUL_H

#include "dsa_sparse_prolog_comm.h"

namespace DsaSparseProlog {

template <typename SrcT>
__aicore__ inline constexpr uint32_t GetC0Num()
{
    if (sizeof(SrcT) == sizeof(float)) {
        return 8;
    } else if (sizeof(SrcT) == sizeof(int8_t)) {
        return 32;
    }
    return 16;
}

template <typename T>
__aicore__ inline T CeilDivT(T num1, T num2)
{
    return (num2 == 0) ? 0 : (num1 + num2 - 1) / num2;
}

template <typename T>
__aicore__ inline void CopyNDGmToL1(LocalTensor<T>& l1Tensor, const GlobalTensor<T>& gmSrcTensor, uint32_t srcN,
                                    uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = CeilDivT(srcN, GetC0Num<T>()) * GetC0Num<T>(); // 对齐到16 单位 block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename T>
__aicore__ inline void CopyNZGmToL1(LocalTensor<T>& l1Tensor, const GlobalTensor<T>& gmSrcTensor, uint32_t srcN,
                                    uint32_t srcD, uint32_t srcNstride)
{
    DataCopyParams param;
    param.blockCount = CeilDivT(srcD, GetC0Num<T>());
    param.blockLen = srcN;                 // 单位为32B srcN*16/16
    param.srcStride = (srcNstride - srcN); // 单位为32B (srcNstride - srcN)*16/16
    param.dstStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, param);
}

// B 侧一个 L1 tile 的 GM→L1 装载, 按 bLayout 4 布局分支:
//   NZ [K,N]: n1 行 stride = kInput (全 K), tile 布局 (N1,K1,K0,N0), L1→L0B ifTranspose=true
//   ND [K,N]: 行主序 [k,n], 行 stride = orgNfull, Nd2Nz 后同上
//   NZ [N,K]: k1 行 stride = orgNfull (全 N), tile 布局 (K1,N1,N0,K0), L1→L0B ifTranspose=false (transB)
//   ND [N,K]: 行主序 [n,k], 行 stride = kInput, Nd2Nz 后同上
// nTileSize = 本 tile 实际 N 宽; nL1SplitSize = nL1 步进; orgNfull = 权重全宽 N (orgKc); kInput = 全 K
template <typename T>
__aicore__ inline void CopyBTileGmToL1(LocalTensor<T>& bL1, const GlobalTensor<T>& tensorBGm, uint32_t bLayout,
                                       int64_t kIndex, int64_t nL1, uint32_t kL1StepSize, uint32_t nTileSize,
                                       uint32_t nL1SplitSize, uint32_t kInput, uint32_t orgNfull)
{
    int64_t tensorBGmOffset;
    if (bLayout == B_LAYOUT_ND_KN) {
        tensorBGmOffset = kIndex * kL1StepSize * orgNfull + nL1 * nL1SplitSize;
        CopyNDGmToL1(bL1, tensorBGm[tensorBGmOffset], kL1StepSize, nTileSize, orgNfull);
    } else if (bLayout == B_LAYOUT_NZ_NK) {
        tensorBGmOffset = kIndex * kL1StepSize * orgNfull + nL1 * nL1SplitSize * GetC0Num<T>();
        CopyNZGmToL1(bL1, tensorBGm[tensorBGmOffset], nTileSize, kL1StepSize, orgNfull);
    } else if (bLayout == B_LAYOUT_ND_NK) {
        tensorBGmOffset = nL1 * nL1SplitSize * kInput + kIndex * kL1StepSize;
        CopyNDGmToL1(bL1, tensorBGm[tensorBGmOffset], nTileSize, kL1StepSize, kInput);
    } else {
        tensorBGmOffset = kInput * nL1 * nL1SplitSize + kIndex * kL1StepSize * GetC0Num<T>();
        CopyNZGmToL1(bL1, tensorBGm[tensorBGmOffset], kL1StepSize, nTileSize, kInput);
    }
}

#ifdef USE_MM_API_MLAP

template <typename T, typename O, typename MMType>
__aicore__ inline void Matmul(MMType& mm, const GlobalTensor<T>& tensorAGm, const GlobalTensor<T>& tensorBGm,
                              const GlobalTensor<O>& tensorCGm, const MMParams& para)
{
    // mmQn实际计算单核k=128, 输入orgk为192
    if (para.needSetOrgShape) {
        mm.SetOrgShape(para.orgM, para.orgN, para.orgKa, para.orgKb, para.orgKc);
    }
    mm.SetSingleShape(para.m, para.n, para.k);
    mm.SetTensorA(tensorAGm);
    mm.SetTensorB(tensorBGm);
    mm.IterateAll(tensorCGm);
    mm.End();
}

template <typename T, typename O, typename MMType>
__aicore__ inline void MatmulImplNormal(MMType& mm, const GlobalTensor<T>& tensorAGm, const GlobalTensor<T>& tensorBGm,
                                        const GlobalTensor<O>& tensorCGm, const MMParams& para, MMBufParams* bufParam,
                                        int32_t blockIdx = 0)
{
    // 全局L1管理
    uint32_t mInput = para.m;
    uint32_t nInput = para.n;
    uint32_t kInput = para.k;
    uint32_t stepK = para.stepK;
    uint32_t kL1SplitSize = para.baseK;
    uint32_t nL1SplitSize = para.baseN;
    uint32_t kL1StepSize = kL1SplitSize * stepK;
    uint32_t mSize = Align(mInput, GetC0Num<T>());
    uint32_t kOuterloops = CeilDivT(kInput, kL1StepSize);
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);

    uint32_t aOffsetUnit = mSize * kL1SplitSize;
    uint32_t bOffsetUnit = GetC0Num<T>() * kL1SplitSize;

    LocalTensor<T> aL1Tensor;
    aL1Tensor.SetAddr(bufParam->aL1BufAddr);
    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam->bL1BufAddr);

    uint32_t subNL1SplitSize = nL1SplitSize;
    mm.SetOrgShape(para.orgM, para.orgN, kL1StepSize, kL1StepSize, para.orgKc);
    // [N,K] 布局: L1 tile 为 NZ-of-[N,K], k1 stride = tile N 宽, L1→L0B 走 ifTranspose=false + mad transB
    const bool bTrans = (para.bLayout >= B_LAYOUT_NZ_NK);

    uint32_t kStart = 0;
    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        if (bTrans) {
            // trans 路径 L1→L0B srcStride = orgN/16 (fractal 单位), 必须等于 tile N 宽; C 写出走 orgKc 不受影响
            mm.SetOrgShape(para.orgM, subNL1SplitSize, kL1StepSize, kL1StepSize, para.orgKc);
        }
        mm.SetSingleShape(mInput, subNL1SplitSize, kL1SplitSize);
        // trans tile k1 stride = subNL1SplitSize*16 元素, 每 inner 步进 baseK/16 个 k1 块
        uint32_t bOffsetStep = bTrans ? kL1SplitSize * subNL1SplitSize : bOffsetUnit;
        for (uint32_t kL1 = kStart; kL1 < kStart + kOuterloops; kL1++) {
            int64_t kIndex = kL1 % kOuterloops;
            WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
            LocalTensor<T> aL1 = aL1Tensor[(bufParam->aL1BufIter % 2) * (L1_A_SIZE / sizeof(T))];
            auto tensorAGmForL1 = tensorAGm[kIndex * kL1StepSize];
            CopyNDGmToL1(aL1, tensorAGmForL1, mInput, kL1StepSize, kInput);
            SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));
            WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));

            WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            LocalTensor<T> bL1 = bL1Tensor[(bufParam->bL1BufIter % 2) * (L1_B_SIZE / sizeof(T))];
            CopyBTileGmToL1(bL1, tensorBGm, para.bLayout, kIndex, nL1, kL1StepSize, subNL1SplitSize, nL1SplitSize,
                            kInput, para.orgKc);
            SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            int64_t aOffset = 0;
            int64_t bOffset = 0;
            for (int64_t kInner = 0; kInner < stepK; kInner++) {
                mm.SetTensorA(aL1[aOffset]);
                mm.SetTensorB(bL1[bOffset], bTrans);
                mm.Iterate((kL1 != kStart) || (kInner != 0)); // enAtomic
                aOffset += aOffsetUnit;
                bOffset += bOffsetStep;
            }
            SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            bufParam->bL1BufIter++;
            SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
            bufParam->aL1BufIter++;
        }
        mm.GetTensorC(tensorCGm[nL1 * nL1SplitSize]);
    }
    mm.End();
}

template <typename T, typename O, typename MMType>
__aicore__ inline void MatmulImplNormalPreload(MMType& mm, const GlobalTensor<T>& tensorAGm,
                                               const GlobalTensor<T>& tensorBGm, const GlobalTensor<O>& tensorCGm,
                                               const MMParams& para, MMBufParams* bufParam, int32_t blockIdx = 0)
{
    // 全局L1管理
    uint32_t mInput = para.m;
    uint32_t nInput = para.n;
    uint32_t kInput = para.k;
    uint32_t stepK = para.stepK;
    uint32_t kL1SplitSize = para.baseK;
    uint32_t nL1SplitSize = para.baseN;
    uint32_t kL1StepSize = kL1SplitSize * stepK;
    uint32_t mSize = Align(mInput, GetC0Num<T>());
    uint32_t kOuterloops = CeilDivT(kInput, kL1StepSize);
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);

    uint32_t aOffsetUnit = mSize * kL1SplitSize;
    uint32_t bOffsetUnit = GetC0Num<T>() * kL1SplitSize;

    LocalTensor<T> aL1Tensor;
    aL1Tensor.SetAddr(bufParam->aL1BufAddr);
    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam->bL1BufAddr);

    uint32_t subNL1SplitSize = nL1SplitSize;
    mm.SetOrgShape(para.orgM, para.orgN, kL1StepSize, kL1StepSize, para.orgKc);
    const bool bTrans = (para.bLayout >= B_LAYOUT_NZ_NK);

    uint32_t kStart = 0;
    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        if (bTrans) {
            mm.SetOrgShape(para.orgM, subNL1SplitSize, kL1StepSize, kL1StepSize, para.orgKc);
        }
        mm.SetSingleShape(mInput, subNL1SplitSize, kL1SplitSize);
        uint32_t bOffsetStep = bTrans ? kL1SplitSize * subNL1SplitSize : bOffsetUnit;
        for (uint32_t kL1 = kStart; kL1 < kStart + kOuterloops; kL1++) {
            int64_t kIndex = kL1 % kOuterloops;
            WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
            LocalTensor<T> aL1 = aL1Tensor[(bufParam->aL1BufIter % 2) * (L1_A_SIZE / sizeof(T))];
            auto tensorAGmForL1 = tensorAGm[kIndex * kL1StepSize];
            CopyNDGmToL1(aL1, tensorAGmForL1, mInput, kL1StepSize, kInput);
            SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));
            WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));

            LocalTensor<T> bL1 = bL1Tensor[(bufParam->bL1BufIter % 2) * (L1_B_SIZE / sizeof(T))];
            if (nL1 != 0 || kL1 > 1) {
                WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
                CopyBTileGmToL1(bL1, tensorBGm, para.bLayout, kIndex, nL1, kL1StepSize, subNL1SplitSize, nL1SplitSize,
                                kInput, para.orgKc);
                SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
                WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            }
            int64_t aOffset = 0;
            int64_t bOffset = 0;
            for (int64_t kInner = 0; kInner < stepK; kInner++) {
                mm.SetTensorA(aL1[aOffset]);
                mm.SetTensorB(bL1[bOffset], bTrans);
                mm.Iterate((kL1 != kStart) || (kInner != 0)); // enAtomic
                aOffset += aOffsetUnit;
                bOffset += bOffsetStep;
            }
            SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            bufParam->bL1BufIter++;
            SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
            bufParam->aL1BufIter++;
        }
        mm.GetTensorC(tensorCGm[nL1 * nL1SplitSize]);
    }
    mm.End();
}

template <typename T, typename O, typename MMType>
__aicore__ inline void QcQrWeightPreload(const GlobalTensor<T>& tensorBGm, const MMParams& para, MMBufParams* bufParam)
{
    // 全局L1管理
    uint32_t kInput = para.k;
    uint32_t stepK = para.stepK;
    uint32_t kL1SplitSize = para.baseK;
    // uint32_t nL1SplitSize = para.baseN;
    uint32_t nL1SplitSize = (para.n < para.baseN) ? para.n : para.baseN;
    uint32_t kL1StepSize = kL1SplitSize * stepK;

    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam->bL1BufAddr);

    int64_t nL1 = 0;
    for (uint32_t kL1 = 0; kL1 < 2; kL1++) {
        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
        LocalTensor<T> bL1 = bL1Tensor[(bufParam->bL1BufIter % 2) * (L1_B_SIZE / sizeof(T))];
        CopyBTileGmToL1(bL1, tensorBGm, para.bLayout, kL1, nL1, kL1StepSize, nL1SplitSize, nL1SplitSize, kInput,
                        para.orgKc);
        SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
        bufParam->bL1BufIter++;
    }
}

template <typename T, typename O, typename MMType>
__aicore__ inline void MatmulImplQcQr(MMType& mm, const GlobalTensor<T>& tensorAGm, const GlobalTensor<T>& tensorBGm,
                                      const GlobalTensor<O>& tensorCGm, const MMParams& para, MMBufParams* bufParam)
{
    // 全局L1管理
    uint32_t mInput = para.m;
    uint32_t nInput = para.n;
    uint32_t kInput = para.k;
    uint32_t stepK = para.stepK;
    uint32_t kL1SplitSize = para.baseK;
    uint32_t nL1SplitSize = para.baseN;
    uint32_t kL1StepSize = kL1SplitSize * stepK;
    uint32_t mSize = Align(mInput, GetC0Num<T>());
    uint32_t kOuterloops = CeilDivT(kInput, kL1StepSize);
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);

    uint32_t aOffsetUnit = mSize * kL1SplitSize;
    uint32_t bOffsetUnit = GetC0Num<T>() * kL1SplitSize;

    LocalTensor<T> aL1Tensor;
    aL1Tensor.SetAddr(bufParam->aL1BufAddr);
    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam->bL1BufAddr);

    WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    LocalTensor<T> aL1 = aL1Tensor[(bufParam->aL1BufIter % 2) * (L1_A_SIZE / sizeof(T))];
    CopyNDGmToL1(aL1, tensorAGm, mInput, kInput, kInput);
    SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));

    uint32_t subNL1SplitSize = nL1SplitSize;
    mm.SetOrgShape(para.orgM, para.orgN, kInput, kL1StepSize, para.orgKc);
    const bool bTrans = (para.bLayout >= B_LAYOUT_NZ_NK);

    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        if (bTrans) {
            mm.SetOrgShape(para.orgM, subNL1SplitSize, kInput, kL1StepSize, para.orgKc);
        }
        mm.SetSingleShape(mInput, subNL1SplitSize, kL1SplitSize);
        uint32_t bOffsetStep = bTrans ? kL1SplitSize * subNL1SplitSize : bOffsetUnit;
        int64_t aOffset = 0;
        for (uint32_t kL1 = 0; kL1 < kOuterloops; kL1++) {
            LocalTensor<T> bL1 = bL1Tensor[(bufParam->bL1BufIter % 2) * (L1_B_SIZE / sizeof(T))];
            if (nL1 != 0 || kL1 > 1) {
                WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
                CopyBTileGmToL1(bL1, tensorBGm, para.bLayout, kL1, nL1, kL1StepSize, subNL1SplitSize, nL1SplitSize,
                                kInput, para.orgKc);
                SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
                WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            }
            int64_t bOffset = 0;
            for (int64_t kInner = 0; kInner < stepK; kInner++) {
                mm.SetTensorA(aL1[aOffset]);
                mm.SetTensorB(bL1[bOffset], bTrans);
                mm.Iterate((kL1 != 0) || (kInner != 0)); // enAtomic
                aOffset += aOffsetUnit;
                bOffset += bOffsetStep;
            }
            SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            bufParam->bL1BufIter++;
        }
        mm.GetTensorC(tensorCGm[nL1 * nL1SplitSize]);
    }
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    bufParam->aL1BufIter++;
    mm.End();
}

template <typename T, typename O, typename MMType>
__aicore__ inline void MatmulImplAFullLoad(MMType& mm, const GlobalTensor<T>& tensorAGm,
                                           const GlobalTensor<T>& tensorBGm, const GlobalTensor<O>& tensorCGm,
                                           const MMParams& para, MMBufParams* bufParam)
{
    // 全局L1管理
    uint32_t mInput = para.m;
    uint32_t nInput = para.n;
    uint32_t kInput = para.k;
    uint32_t stepK = para.stepK;
    uint32_t kL1SplitSize = para.baseK;
    uint32_t nL1SplitSize = para.baseN;
    uint32_t kL1StepSize = kL1SplitSize * stepK;
    uint32_t kOffesetUnit = kL1StepSize * GetC0Num<T>();
    uint32_t mSize = Align(mInput, GetC0Num<T>());
    uint32_t kOuterloops = CeilDivT(kInput, kL1StepSize);
    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    uint32_t aOffsetUnit = mSize * kL1SplitSize;
    uint32_t bOffsetUnit = GetC0Num<T>() * kL1SplitSize;

    LocalTensor<T> aL1Tensor;
    aL1Tensor.SetAddr(bufParam->aL1BufAddr);
    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam->bL1BufAddr);

    WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    LocalTensor<T> aL1 = aL1Tensor[(bufParam->aL1BufIter % 2) * (L1_A_SIZE / sizeof(T))];
    CopyNDGmToL1(aL1, tensorAGm, mInput, kInput, kInput);
    SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));

    uint32_t subNL1SplitSize = nL1SplitSize;
    mm.SetOrgShape(para.orgM, para.orgN, kInput, kL1StepSize, para.orgKc);

    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        if (nL1 == nL1loops - 1) {
            subNL1SplitSize = nInput - (nL1loops - 1) * nL1SplitSize;
        }
        mm.SetSingleShape(mInput, subNL1SplitSize, kL1SplitSize);
        int64_t aOffset = 0;
        for (uint32_t kL1 = 0; kL1 < kOuterloops; kL1++) {
            WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            LocalTensor<T> bL1 = bL1Tensor[(bufParam->bL1BufIter % 2) * (L1_B_SIZE / sizeof(T))];
            int64_t tensorBGmOffset = kInput * nL1 * nL1SplitSize + kL1 * kOffesetUnit;
            auto tensorBGmForL1 = tensorBGm[tensorBGmOffset];
            CopyNZGmToL1(bL1, tensorBGmForL1, kL1StepSize, subNL1SplitSize, kInput);
            SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));

            int64_t bOffset = 0;
            for (int kInner = 0; kInner < stepK; kInner++) {
                mm.SetTensorA(aL1[aOffset]);
                mm.SetTensorB(bL1[bOffset]);
                mm.Iterate((kL1 != 0) || (kInner != 0)); // enAtomic
                aOffset += aOffsetUnit;
                bOffset += bOffsetUnit;
            }
            SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            bufParam->bL1BufIter++;
        }
        mm.GetTensorC(tensorCGm[nL1 * nL1SplitSize]);
    }
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    bufParam->aL1BufIter++;
    mm.End();
}

template <typename T, typename O, typename MMType>
__aicore__ inline void QnWeightPreload(const GlobalTensor<T>& tensorBGm, const MMParams& para, MMBufParams* bufParam)
{
    uint32_t nInput = para.n;
    uint32_t kInput = para.k;

    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam->bL1BufAddr);

    WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
    LocalTensor<T> bL1 = bL1Tensor[(bufParam->bL1BufIter % 2) * (L1_B_SIZE / sizeof(T))];
    CopyNDGmToL1(bL1, tensorBGm, kInput, nInput, nInput);
    SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
    WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
    bufParam->bL1BufIter++;
}

template <typename T, typename O, typename MMType>
__aicore__ inline void MatmulImplQn(MMType& mm, const GlobalTensor<T>& tensorAGm, const GlobalTensor<T>& tensorBGm,
                                    const GlobalTensor<O>& tensorCGm, const MMParams& para, MMBufParams* bufParam)
{
    uint32_t mInput = para.m;
    uint32_t nInput = para.n;
    uint32_t kInput = para.k;
    uint32_t mSize = Align(mInput, GetC0Num<T>());

    LocalTensor<T> aL1Tensor;
    aL1Tensor.SetAddr(bufParam->aL1BufAddr);
    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam->bL1BufAddr);

    WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    LocalTensor<T> aL1 = aL1Tensor[(bufParam->aL1BufIter % 2) * (L1_A_SIZE / sizeof(T))];
    CopyNDGmToL1(aL1, tensorAGm, mInput, kInput, para.orgKa);
    SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));

    LocalTensor<T> bL1 = bL1Tensor[(bufParam->bL1BufIter % 2) * (L1_B_SIZE / sizeof(T))];

    mm.SetOrgShape(para.orgM, para.orgN, para.k, para.k, para.orgKc);
    mm.SetSingleShape(para.m, para.n, para.k);
    mm.SetTensorA(aL1);
    mm.SetTensorB(bL1);
    mm.IterateAll(tensorCGm);
    mm.End();
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
    bufParam->bL1BufIter++;
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    bufParam->aL1BufIter++;
}

template <typename T, typename O, typename MMType>
__aicore__ inline void MatmulImplABFullLoad(MMType& mm, const GlobalTensor<T>& tensorAGm,
                                            const GlobalTensor<T>& tensorBGm, const GlobalTensor<O>& tensorCGm,
                                            const MMParams& para, MMBufParams* bufParam)
{
    uint32_t mInput = para.m;
    uint32_t nInput = para.n;
    uint32_t kInput = para.k;
    uint32_t mSize = Align(mInput, GetC0Num<T>());

    LocalTensor<T> aL1Tensor;
    aL1Tensor.SetAddr(bufParam->aL1BufAddr);
    LocalTensor<T> bL1Tensor;
    bL1Tensor.SetAddr(bufParam->bL1BufAddr);

    WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    LocalTensor<T> aL1 = aL1Tensor[(bufParam->aL1BufIter % 2) * (L1_A_SIZE / sizeof(T))];
    CopyNDGmToL1(aL1, tensorAGm, mInput, kInput, para.orgKa);
    SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));

    WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
    LocalTensor<T> bL1 = bL1Tensor[(bufParam->bL1BufIter % 2) * (L1_B_SIZE / sizeof(T))];
    CopyNDGmToL1(bL1, tensorBGm, kInput, nInput, nInput);
    SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
    WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));

    mm.SetOrgShape(para.orgM, para.orgN, para.k, para.k, para.orgKc);
    mm.SetSingleShape(para.m, para.n, para.k);
    mm.SetTensorA(aL1);
    mm.SetTensorB(bL1);
    mm.IterateAll(tensorCGm);
    mm.End();
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
    bufParam->bL1BufIter++;
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    bufParam->aL1BufIter++;
}

#else

template <typename T>
__aicore__ inline void LoadDataAL0(LocalTensor<T>& aL0Tensor, LocalTensor<T>& aL1Tensor, uint32_t mSize, uint32_t kSize)
{
    uint32_t mloops = mSize / 16;

    for (uint32_t i = 0; i < mloops; i++) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = kSize / (32 / sizeof(T));

        loadData2DParams.srcStride = mSize / 16;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;

        LocalTensor<T> tmpSrcTensor;
        tmpSrcTensor = aL1Tensor[i * 16 * (32 / sizeof(T))];
        LoadData(aL0Tensor[i * 16 * kSize], tmpSrcTensor, loadData2DParams);
    }
}

template <typename T>
__aicore__ inline void LoadDataBL0(LocalTensor<T>& bL0Tensor, LocalTensor<T>& bL1Tensor, uint32_t nSplitSize,
                                   uint32_t nIdx, uint32_t subNSize, uint32_t kSize)
{
    uint32_t kloops = kSize / 16;

    LocalTensor<T> srcTensor = bL1Tensor[kSize * nSplitSize * nIdx];
    for (uint32_t i = 0; i < kloops; i++) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = subNSize / (32 / sizeof(T));

        loadData2DParams.srcStride = kSize / 16;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = true;

        LocalTensor<T> tmpSrcTensor;
        tmpSrcTensor = srcTensor[i * 16 * (32 / sizeof(T))];
        LoadData(bL0Tensor[i * 16 * subNSize], tmpSrcTensor, loadData2DParams);
    }
}

template <typename T>
__aicore__ inline void LoadDataAL0K(LocalTensor<T>& aL0Tensor, LocalTensor<T>& aL1Tensor, uint32_t mSize,
                                    uint32_t subKidx, uint32_t subKSize, uint32_t kSplitSize)
{
    uint32_t mLoops = mSize / 16;
    LocalTensor<T> aL1SrcTensor = aL1Tensor[mSize * kSplitSize * subKidx];

    for (uint32_t i = 0; i < mLoops; i++) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = subKSize / (32 / sizeof(T));
        loadData2DParams.srcStride = mSize / 16;
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = false;

        LocalTensor<T> tmpSrcTensor;
        tmpSrcTensor = aL1SrcTensor[i * 16 * (32 / sizeof(T))];
        LoadData(aL0Tensor[i * 16 * subKSize], tmpSrcTensor, loadData2DParams);
    }
}

template <typename T>
__aicore__ inline void LoadDataBL0K(LocalTensor<T>& bL0Tensor, LocalTensor<T>& bL1Tensor, uint32_t kSplitSize,
                                    uint32_t kIdx, uint32_t nSize, uint32_t kL1Size)
{
    // L1 128 * 256, L0 64 * 128
    uint32_t kloops = kSplitSize / 16;

    LocalTensor<T> srcTensor = bL1Tensor[kIdx * kSplitSize * 16];
    for (uint32_t i = 0; i < kloops; i++) {
        LoadData2DParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = nSize / (32 / sizeof(T));

        loadData2DParams.srcStride = kL1Size / 16; // 512
        loadData2DParams.dstGap = 0;
        loadData2DParams.ifTranspose = true;

        LocalTensor<T> tmpSrcTensor;
        tmpSrcTensor = srcTensor[i * 16 * (32 / sizeof(T))];
        LoadData(bL0Tensor[i * 16 * nSize], tmpSrcTensor, loadData2DParams);
    }
}

template <typename T>
__aicore__ inline void LoadDataBL0KQuant(LocalTensor<T>& bL0Tensor, LocalTensor<T>& bL1Tensor, uint32_t kSplitSize,
                                         uint32_t kIdx, uint32_t nSize, uint32_t kL1Size)
{
    // L1 128 * 256, L0 64 * 128
    uint32_t kloops = kSplitSize / (32 / sizeof(T));

    LocalTensor<T> srcTensor = bL1Tensor[kIdx * kSplitSize * (32 / sizeof(T))];
    for (uint32_t i = 0; i < kloops; i++) {
        LoadData2dTransposeParams loadData2DParams;
        loadData2DParams.startIndex = 0;
        loadData2DParams.repeatTimes = nSize / (32 / sizeof(T));
        loadData2DParams.srcStride = kL1Size / 32; // 512
        loadData2DParams.dstGap = 1;
        loadData2DParams.dstFracGap = 0;

        LocalTensor<T> tmpSrcTensor;
        tmpSrcTensor = srcTensor[i * 32 * (32 / sizeof(T))];
        LoadDataWithTranspose(bL0Tensor[i * 32 * nSize], tmpSrcTensor, loadData2DParams);
    }
}

// L1切K， L0切K
template <typename T, typename O>
__aicore__ inline void KKMatmul(const GlobalTensor<T>& tensorAGm, const GlobalTensor<T>& tensorBGm,
                                const GlobalTensor<O>& tensorCGm, const MMParams& para, MMBufParams<T, float>* bufParam)
{
    uint32_t mInput = para.m;
    uint32_t nInput = para.n;
    uint32_t kInput = para.k;
    uint32_t kL1SplitSize = para.kL1SplitSize;
    uint32_t kSplitSize = para.kSplitSize;

    uint32_t mSize = Align(mInput, 16U);

    // 原始矩阵k较大，切k到512， 保证加载到L1有足够空间
    uint32_t kL1Loops = (kInput + kL1SplitSize - 1) / kL1SplitSize;
    uint32_t kL1Tail = kInput - (kL1Loops - 1) * kL1SplitSize;
    uint32_t subKL1Size = kL1SplitSize;

    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + bufParam->cL0BufIter % 2);
    LocalTensor<float> cL0Tensor =
        bufParam->cL0TensorPingPong[(bufParam->cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(float))];

    for (int64_t kL1 = 0; kL1 < kL1Loops; kL1++) {
        if (kL1 == kL1Loops - 1) {
            subKL1Size = Align(kL1Tail, 16U);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
        LocalTensor<T> aL1 = bufParam->aL1Tensor[(bufParam->aL1BufIter % 2) * (L1_A_SIZE / sizeof(T))];
        auto tensorAGmForL1 = tensorAGm[kL1 * kL1SplitSize];
        CopyNDGmToL1(aL1, tensorAGmForL1, mInput, subKL1Size, kInput);
        SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));

        WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
        LocalTensor<T> bL1 = bufParam->bL1Tensor[(bufParam->bL1BufIter % 2) * (L1_B_SIZE / sizeof(T))];
        auto tensorBGmForL1 = tensorBGm[kL1 * kL1SplitSize * 16];
        CopyNZGmToL1(bL1, tensorBGmForL1, subKL1Size, nInput, kInput);
        SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
        WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));

        // 切k
        uint32_t kLoops = (subKL1Size + kSplitSize - 1) / kSplitSize;
        uint32_t kTail = subKL1Size - (kLoops - 1) * kSplitSize;
        uint32_t subKSize = kSplitSize;
        uint32_t subKSizeAct = kSplitSize;

        for (uint32_t k = 0; k < kLoops; k++) {
            if (k == kLoops - 1) {
                subKSizeAct = kTail;
                subKSize = Align(kTail, 16U);
            }
            WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + bufParam->aL0BufIter % 2);
            LocalTensor<T> aL0Tensor =
                bufParam->aL0TensorPingPong[(bufParam->aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(T))];
            LoadDataAL0K(aL0Tensor, aL1, mInput, k, subKSize, kSplitSize);
            SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + bufParam->aL0BufIter % 2);
            WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + bufParam->aL0BufIter % 2);

            WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bufParam->bL0BufIter % 2);
            LocalTensor<T> bL0Tensor =
                bufParam->bL0TensorPingPong[(bufParam->bL0BufIter % 2) * (L0B_PP_SIZE / sizeof(T))];
            LoadDataBL0K(bL0Tensor, bL1, subKSize, k, nInput, kL1SplitSize);
            SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bufParam->bL0BufIter % 2);
            WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bufParam->bL0BufIter % 2);

            MmadParams mmadParams;
            mmadParams.m = mInput;
            mmadParams.n = nInput;
            mmadParams.k = subKSize;
            mmadParams.cmatrixInitVal = ((kL1 == 0) && (k == 0));
            mmadParams.cmatrixSource = false;

            LocalTensor<float> destL0C = cL0Tensor;

            Mmad(destL0C, aL0Tensor, bL0Tensor, mmadParams);

            PipeBarrier<PIPE_M>();

            SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bufParam->bL0BufIter % 2);
            bufParam->bL0BufIter++;

            SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + bufParam->aL0BufIter % 2);
            bufParam->aL0BufIter++;
        }
        SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
        bufParam->bL1BufIter++;
        SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
        bufParam->aL1BufIter++;
    }

    SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + bufParam->cL0BufIter % 2);
    WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + bufParam->cL0BufIter % 2);

    FixpipeParamsV220 fixParams;
    fixParams.nSize = nInput;    // 实现切片大小
    fixParams.mSize = mInput;    // msdIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
    fixParams.srcStride = mSize; // ((fixParams.mSize + 15) / 16) * 16
    fixParams.dstStride = para.orgKc;
    fixParams.ndNum = 1;

    fixParams.quantPre = QuantMode_t::F322BF16;
    Fixpipe(tensorCGm, cL0Tensor, fixParams);

    SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + bufParam->cL0BufIter % 2);
    bufParam->cL0BufIter++;
}

// L1切K, L0切K
template <typename T, typename O>
__aicore__ inline void KKMatmulQuant(const GlobalTensor<T>& tensorAGm, const GlobalTensor<T>& tensorBGm,
                                     const GlobalTensor<O>& tensorCGm, const MMParams& para,
                                     MMBufParams<T, float>* bufParam)
{
    uint32_t mInput = para.m;
    uint32_t nInput = para.n;
    uint32_t kInput = para.k;
    constexpr uint32_t nL1SplitSize = 256;
    uint32_t kL1SplitSize = para.kL1SplitSize;
    uint32_t kSplitSize = para.kSplitSize;

    uint32_t mSize = Align(mInput, 32U);

    uint32_t nL1loops = CeilDivT(nInput, nL1SplitSize);
    uint32_t kL1Loops = (kInput + kL1SplitSize - 1) / kL1SplitSize;
    uint32_t kL1Tail = kInput - (kL1Loops - 1) * kL1SplitSize;
    uint32_t subKL1Size = kL1SplitSize;

    for (int64_t nL1 = 0; nL1 < nL1loops; nL1++) {
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + bufParam->cL0BufIter % 2);
        auto cL0TensorOri = bufParam->cL0TensorPingPong[(bufParam->cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(O))];
        LocalTensor<O> cL0Tensor = cL0TensorOri.template ReinterpretCast<O>();
        for (int64_t kL1 = 0; kL1 < kL1Loops; kL1++) {
            if (kL1 == kL1Loops - 1) {
                subKL1Size = Align(kL1Tail, 16U);
            }
            WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
            auto aL1Ori = bufParam->aL1Tensor[(bufParam->aL1BufIter % 2) * (L1_A_SIZE / 2 / sizeof(T))];
            LocalTensor<T> aL1 = aL1Ori.template ReinterpretCast<T>();

            CopyNDGmToL1(aL1, tensorAGm[kL1 * kL1SplitSize], mInput, subKL1Size, kInput);
            SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));
            WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));

            WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            auto bL1Ori = bufParam->bL1Tensor[(bufParam->bL1BufIter % 2) * (L1_B_SIZE / 2 / sizeof(T))];
            LocalTensor<T> bL1 = bL1Ori.template ReinterpretCast<T>();

            int64_t tensorBGmOffset = nL1 * nL1SplitSize * kInput + kL1 * kL1SplitSize * 32;
            auto tensorBGmForL1 = tensorBGm[tensorBGmOffset];
            CopyNZGmToL1(bL1, tensorBGmForL1, subKL1Size, nL1SplitSize, kInput);
            SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));

            // 切k
            uint32_t kLoops = (subKL1Size + kSplitSize - 1) / kSplitSize;
            uint32_t kTail = subKL1Size - (kLoops - 1) * kSplitSize;
            uint32_t subKSize = kSplitSize;
            uint32_t subKSizeAct = kSplitSize;

            for (uint32_t k = 0; k < kLoops; k++) {
                if (k == kLoops - 1) {
                    subKSizeAct = kTail;
                    subKSize = Align(kTail, 32U);
                }
                WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + bufParam->aL0BufIter % 2);
                auto aL0TensorOri =
                    bufParam->aL0TensorPingPong[(bufParam->aL0BufIter % 2) * (L0A_PP_SIZE / 2 / sizeof(T))];
                LocalTensor<T> aL0Tensor = aL0TensorOri.template ReinterpretCast<T>();
                LoadDataAL0K(aL0Tensor, aL1, mInput, k, subKSize, kSplitSize);
                SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + bufParam->aL0BufIter % 2);
                WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + bufParam->aL0BufIter % 2);

                WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bufParam->bL0BufIter % 2);
                auto bL0TensorOri =
                    bufParam->bL0TensorPingPong[(bufParam->bL0BufIter % 2) * (L0B_PP_SIZE / 2 / sizeof(T))];
                LocalTensor<T> bL0Tensor = bL0TensorOri.template ReinterpretCast<T>();
                LoadDataBL0KQuant(bL0Tensor, bL1, subKSize, k, nL1SplitSize, kL1SplitSize);
                SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bufParam->bL0BufIter % 2);
                WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bufParam->bL0BufIter % 2);

                MmadParams mmadParams;
                mmadParams.m = mInput;
                mmadParams.n = nL1SplitSize;
                mmadParams.k = subKSize;
                mmadParams.cmatrixInitVal = ((kL1 == 0) && (k == 0));
                mmadParams.cmatrixSource = false;

                LocalTensor<O> destL0C = cL0Tensor;

                Mmad(destL0C, aL0Tensor, bL0Tensor, mmadParams);

                PipeBarrier<PIPE_M>();

                SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bufParam->bL0BufIter % 2);
                bufParam->bL0BufIter++;

                SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + bufParam->aL0BufIter % 2);
                bufParam->aL0BufIter++;
            }
            SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
            bufParam->bL1BufIter++;
            SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
            bufParam->aL1BufIter++;
        }

        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + bufParam->cL0BufIter % 2);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + bufParam->cL0BufIter % 2);

        FixpipeParamsV220 fixParams;
        fixParams.nSize = nL1SplitSize; // 实现切片大小
        fixParams.mSize = mInput;    // msdIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
        fixParams.srcStride = mSize; // ((fixParams.mSize + 15) / 16) * 16
        fixParams.dstStride = para.orgKc;
        fixParams.ndNum = 1;

        Fixpipe(tensorCGm[nL1 * nL1SplitSize], cL0Tensor, fixParams);

        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + bufParam->cL0BufIter % 2);
        bufParam->cL0BufIter++;
    }
}

// L1切K, L0切K
template <typename T, typename O>
__aicore__ inline void NMatmul(const GlobalTensor<T>& tensorAGm, const GlobalTensor<T>& tensorBGm,
                               const GlobalTensor<O>& tensorCGm, const MMParams& para, MMBufParams<T, float>* bufParam)
{
    uint32_t mInput = para.m;
    uint32_t nInput = para.n;
    uint32_t kInput = para.k;
    uint32_t nSplitSize = para.nSplitSize;
    uint32_t mSize = Align(mInput, 16U);

    WaitFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    LocalTensor<T> aL1 = bufParam->aL1Tensor[(bufParam->aL1BufIter % 2) * (L1_A_SIZE / sizeof(T))];
    auto tensorAGmForL1 = tensorAGm;
    CopyNDGmToL1(aL1, tensorAGmForL1, mSize, kInput, para.orgKa); // (128 + 64)
    SetFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    WaitFlag<HardEvent::MTE2_MTE1>(A_EVENT0 + (bufParam->aL1BufIter % 2));

    WaitFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
    LocalTensor<T> bL1 = bufParam->bL1Tensor[(bufParam->bL1BufIter % 2) * (L1_B_SIZE / sizeof(T))];
    auto tensorBGmForL1 = tensorBGm;
    CopyNDGmToL1(bL1, tensorBGmForL1, kInput, nInput, nInput);
    SetFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));
    WaitFlag<HardEvent::MTE2_MTE1>(B_EVENT0 + (bufParam->bL1BufIter % 2));

    WaitFlag<HardEvent::M_MTE1>(L0A_EVENT0 + bufParam->aL0BufIter % 2);
    LocalTensor<T> aL0Tensor = bufParam->aL0TensorPingPong[(bufParam->aL0BufIter % 2) * (L0A_PP_SIZE / sizeof(T))];
    LoadDataAL0(aL0Tensor, aL1, mSize, kInput);
    SetFlag<HardEvent::MTE1_M>(L0A_EVENT0 + bufParam->aL0BufIter % 2);
    WaitFlag<HardEvent::MTE1_M>(L0A_EVENT0 + bufParam->aL0BufIter % 2);

    uint32_t nLoops = (nInput + nSplitSize - 1) / nSplitSize;
    uint32_t nTail = nInput - (nLoops - 1) * nSplitSize;
    uint32_t subNSize = nSplitSize;
    uint32_t subNSizeAct = nSplitSize;

    for (uint32_t n = 0; n < nLoops; n++) {
        if (n == nLoops - 1) {
            subNSizeAct = nTail;
            subNSize = Align(nTail, 16U);
        }
        WaitFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bufParam->bL0BufIter % 2);
        LocalTensor<T> bL0Tensor = bufParam->bL0TensorPingPong[(bufParam->bL0BufIter % 2) * (L0B_PP_SIZE / sizeof(T))];
        LoadDataBL0(bL0Tensor, bL1, nSplitSize, n, subNSize, kInput);
        SetFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bufParam->bL0BufIter % 2);
        WaitFlag<HardEvent::MTE1_M>(L0B_EVENT0 + bufParam->bL0BufIter % 2);

        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + bufParam->cL0BufIter % 2);
        LocalTensor<float> cL0Tensor =
            bufParam->cL0TensorPingPong[(bufParam->cL0BufIter % 2) * (L0C_PP_SIZE / sizeof(float))];
        MmadParams mmadParams;
        mmadParams.m = 32;
        mmadParams.n = 128;
        mmadParams.k = 128;
        mmadParams.cmatrixInitVal = 1;
        mmadParams.cmatrixSource = false;

        LocalTensor<float> destL0C = cL0Tensor;

        Mmad(destL0C, aL0Tensor, bL0Tensor, mmadParams);

        SetFlag<HardEvent::M_MTE1>(L0B_EVENT0 + bufParam->bL0BufIter % 2);
        bufParam->bL0BufIter++;

        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + bufParam->cL0BufIter % 2);
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + bufParam->cL0BufIter % 2);

        FixpipeParamsV220 fixParams;
        fixParams.nSize = subNSizeAct; // 实现切片大小
        fixParams.mSize = mInput;    // msdIterNum * gSize; // 有效数据不足16行，只需要输出部分行即可
        fixParams.srcStride = mSize; // ((fixParams.mSize + 15) / 16) * 16
        fixParams.dstStride = para.orgKc; // 32 * 512
        fixParams.ndNum = 1;

        fixParams.quantPre = QuantMode_t::F322BF16;
        int64_t tensorCGmOffset = n * nSplitSize;
        Fixpipe(tensorCGm[tensorCGmOffset], cL0Tensor, fixParams);

        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + bufParam->cL0BufIter % 2);
        bufParam->cL0BufIter++;
    }
    SetFlag<HardEvent::M_MTE1>(L0A_EVENT0 + bufParam->aL0BufIter % 2);
    bufParam->aL0BufIter++;
    SetFlag<HardEvent::MTE1_MTE2>(B_EVENT0 + (bufParam->bL1BufIter % 2));
    bufParam->bL1BufIter++;
    SetFlag<HardEvent::MTE1_MTE2>(A_EVENT0 + (bufParam->aL1BufIter % 2));
    bufParam->aL1BufIter++;
}

#endif
} // namespace DsaSparseProlog

#endif
