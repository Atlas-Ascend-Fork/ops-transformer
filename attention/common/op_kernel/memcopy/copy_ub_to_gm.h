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
 * \file copy_ub_to_gm.h
 * \brief
 */
#ifndef COPY_UB_TO_GM_H
#define COPY_UB_TO_GM_H

#include "gm_layout.h"
#include "gm_coord.h"
#include "fa_ub_tensor.h"
#include "offset_calculator_v2.h"
#include "../const_def.h"
// ----------------------------------------------SafeStrideCopy--------------------------------
template <typename OUT_T>
__aicore__ inline void SafeStrideCopy(GlobalTensor<OUT_T> gmTensor, const LocalTensor<OUT_T> ubTensor,
                                      uint32_t blockCount, uint32_t blockLen, uint32_t srcStride, uint64_t dstStride)
{
    DataCopyExtParams dataCopyParams;
    // B*S过大时，跳写参数dataCopyParams.dstStride(uint32_t)计算结果将溢出，使用for循环拷贝代替
    if (dstStride > UINT32_MAX) {
        uint64_t gmSingleStride = (dstStride + blockLen) / sizeof(OUT_T);
        uint64_t ubSingleStride = (srcStride * AttentionCommon::BYTE_BLOCK + blockLen) / sizeof(OUT_T);
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0; // 单位为Byte
        for (uint32_t i = 0; i < blockCount; i++) {
            DataCopyPad(gmTensor[i * gmSingleStride], ubTensor[i * ubSingleStride], dataCopyParams);
        }
    } else {
        // dataCopyParams.dstStride(uint32_t)没有溢出时，进行跳写
        dataCopyParams.blockCount = blockCount;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.srcStride = srcStride;
        dataCopyParams.dstStride = dstStride; // 单位为Byte
        DataCopyPad(gmTensor, ubTensor, dataCopyParams);
    }
}

// ----------------------------------------------CopyAttenOutUbToGm--------------------------------
template <typename OUT_T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyAttenOutUbToGm {};

template <typename OUT_T, GmFormat GM_FORMAT>
class CopyAttenOutUbToGm<OUT_T, GM_FORMAT, UbFormat::GS1> {
public:
    template <typename FaGmTensorType, bool ROW_PAD = true>
    __aicore__ inline void operator()(FaGmTensorType& dstTensor, FaUbTensor<OUT_T, ROW_PAD>& srcTensor,
                                      GmCoordGs1Merge& gmCoord)
    {
        uint32_t s1Size = 0;
        auto& offsetCalculator = dstTensor.offsetCalculator;
        if constexpr (GmLayoutParams<GM_FORMAT>::CATEGORY == FormatCategory::GM_Q_OUT_TND) {
            s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        } else {
            if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
                s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
            } else {
                s1Size = offsetCalculator.GetDimS1();
            }
        }
        uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
        uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / s1Size;
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % s1Size;

        uint64_t attenOutGmbaseOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, 0, 0);

        // 处理第一个S
        uint32_t headS1 = 0;
        if (gIdxStart == gIdxEnd) {
            headS1 = s1IdxEnd - s1IdxStart;
        } else {
            headS1 = s1Size - s1IdxStart;
        }
        uint64_t gmOffset = attenOutGmbaseOffset + s1IdxStart * offsetCalculator.GetStrideS1();
        uint64_t ubOffset = 0;
        uint32_t blockCount = headS1;
        uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
        uint32_t srcStride = (srcTensor.colCount - gmCoord.dDealSize) / (AttentionCommon::BYTE_BLOCK / sizeof(OUT_T));
        uint64_t dstStride = (offsetCalculator.GetStrideS1() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
        SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride,
                              dstStride);

        if (gIdxEnd - gIdxStart >= 1) {
            // 处理中间块
            gmOffset = attenOutGmbaseOffset + offsetCalculator.GetStrideG();
            ubOffset = headS1 * srcTensor.colCount;
            for (uint32_t i = gIdxStart + 1; i < gIdxEnd; i++) {
                blockCount = s1Size;
                SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                      srcStride, dstStride);
                gmOffset += offsetCalculator.GetStrideG();
                ubOffset += s1Size * srcTensor.colCount;
            }

            // 处理尾块
            if (s1IdxEnd > 0) {
                blockCount = s1IdxEnd;
                SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                      srcStride, dstStride);
            }
        }
    }
};

template <typename OUT_T, GmFormat GM_FORMAT>
class CopyAttenOutUbToGm<OUT_T, GM_FORMAT, UbFormat::S1G> {
public:
    template <typename FaGmTensorType, bool ROW_PAD = true>
    __aicore__ inline void operator()(FaGmTensorType& dstTensor, FaUbTensor<OUT_T, ROW_PAD>& srcTensor,
                                      GmCoordGs1Merge& gmCoord)
    {
        auto& offsetCalculator = dstTensor.offsetCalculator;
        uint32_t s1IdxStart = gmCoord.gS1Idx / offsetCalculator.GetDimG();
        uint32_t gIdxStart = gmCoord.gS1Idx % offsetCalculator.GetDimG();
        uint32_t s1IdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) / offsetCalculator.GetDimG();
        uint32_t gIdxEnd = (gmCoord.gS1Idx + gmCoord.gS1DealSize) % offsetCalculator.GetDimG();

        uint64_t attenOutGmbaseOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart, 0);
        uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
        uint32_t srcStride = (srcTensor.colCount - gmCoord.dDealSize) / (AttentionCommon::BYTE_BLOCK / sizeof(OUT_T));
        uint64_t dstStride = (offsetCalculator.GetStrideG() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte

        uint32_t headSize = 0;
        if (s1IdxStart == s1IdxEnd) {
            headSize = gIdxEnd - gIdxStart;
        } else {
            headSize = offsetCalculator.GetDimG() - gIdxStart;
        }
        uint64_t gmOffset = attenOutGmbaseOffset + gIdxStart * offsetCalculator.GetStrideG();
        uint64_t ubOffset = 0;
        uint32_t blockCount = headSize;

        SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen, srcStride,
                              dstStride);

        if (s1IdxEnd - s1IdxStart >= 1) {
            // 处理中间块
            gmOffset = attenOutGmbaseOffset + offsetCalculator.GetStrideS1();
            ubOffset = ((uint64_t)headSize) * ((uint64_t)srcTensor.colCount);
            if constexpr ((!ROW_PAD) && ((GM_FORMAT == GmFormat::BSNGD) || (GM_FORMAT == GmFormat::TNGD))) {
                if (s1IdxEnd > 1 + s1IdxStart) {
                    uint32_t blockCountTmp = s1IdxEnd - s1IdxStart - 1;
                    uint32_t blockLenTmp = gmCoord.dDealSize * sizeof(OUT_T) * offsetCalculator.GetDimG();
                    uint32_t srcStrideTmp = 0;
                    uint64_t dstStrideTmp = offsetCalculator.GetStrideS1() * sizeof(OUT_T) - blockLenTmp; // 单位为Byte

                    // blockCount = offsetCalculator.GetDimG() * (s1IdxEnd - s1IdxStart -1);
                    SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCountTmp,
                                          blockLenTmp, srcStrideTmp, dstStrideTmp);
                    gmOffset += offsetCalculator.GetStrideS1() * blockCountTmp;
                    ubOffset += offsetCalculator.GetDimG() * srcTensor.colCount * blockCountTmp;
                }
            } else {
                for (uint32_t i = s1IdxStart + 1; i < s1IdxEnd; i++) {
                    blockCount = offsetCalculator.GetDimG();
                    SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount,
                                          blockLen, srcStride, dstStride);
                    gmOffset += offsetCalculator.GetStrideS1();
                    ubOffset += offsetCalculator.GetDimG() * srcTensor.colCount;
                }
            }
            // 处理尾块
            if (gIdxEnd > 0) {
                blockCount = gIdxEnd;
                SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[ubOffset], blockCount, blockLen,
                                      srcStride, dstStride);
            }
        }
    }
};

template <typename OUT_T, GmFormat GM_FORMAT>
class CopyAttenOutUbToGm<OUT_T, GM_FORMAT, UbFormat::S1_ONLY> {
public:
    template <typename FaGmTensorType, bool ROW_PAD = true>
    __aicore__ inline void operator()(FaGmTensorType& dstTensor, FaUbTensor<OUT_T, ROW_PAD>& srcTensor,
                                      GmCoordS1Only& gmCoord)
    {
        auto& offsetCalculator = dstTensor.offsetCalculator;
        uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gmCoord.gIdx, gmCoord.s1Idx, 0);
        uint32_t blockCount = gmCoord.s1DealSize;
        uint32_t blockLen = gmCoord.dDealSize * sizeof(OUT_T);
        uint32_t srcStride = (srcTensor.colCount - gmCoord.dDealSize) / (AttentionCommon::BYTE_BLOCK / sizeof(OUT_T));
        uint64_t dstStride = (offsetCalculator.GetStrideS1() - gmCoord.dDealSize) * sizeof(OUT_T); // 单位为Byte
        SafeStrideCopy<OUT_T>(dstTensor.gmTensor[gmOffset], srcTensor.tensor[0], blockCount, blockLen, srcStride,
                              dstStride);
    }
};

// ----------------------------------------------CopyLseOutUbToGm--------------------------------
// LSE 在 UB 中按行存放, 每行 lse 值占 FP32_BLOCK_ELEMENT_NUM(BYTE_BLOCK / sizeof(float)) 个元素
// UB_FORMAT 决定 UB 行序与坐标类型, 与具体 GM 格式无关:
//   S1G    : [S1, G], gS1Idx = s1Idx * G + gIdx,         坐标 GmCoordGs1Merge
//   GS1    : [G, S1], gS1Idx = gIdx * actS1 + s1Idx,      坐标 GmCoordGs1Merge
//   S1_ONLY: DN 范式(realGSize==1)单头, 行按 S1 连续,      坐标 GmCoordS1Only
// GM 侧仅依赖计算器接口: 4 参 GetOffset(b, n2, g, s1) + GetDimG/GetStrideG/GetStrideS1,
// 且 S1(T) 落在最内轴; 排布等价的格式(如 LSE_TN2G/LSE_N2GT/LSE_BN2GS1 及逐头 scale 的 ANTIQ_NT/TN)均可复用
template <typename T, GmFormat GM_FORMAT, UbFormat UB_FORMAT>
class CopyLseOutUbToGm {};

template <typename T, GmFormat GM_FORMAT>
class CopyLseOutUbToGm<T, GM_FORMAT, UbFormat::S1G> {
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaGmTensorType& dstTensor, LocalTensor<T>& lseSrc, GmCoordGs1Merge& gmCoord)
    {
        if (gmCoord.gS1DealSize == 0) {
            return;
        }
        auto& offsetCalculator = dstTensor.offsetCalculator;
        uint64_t gSize = offsetCalculator.GetDimG();
        uint32_t s1IdxStart = gmCoord.gS1Idx / gSize;
        uint32_t gIdxStart = gmCoord.gS1Idx % gSize;
        // 末行索引(含), 避免 deal 恰好结束在 S1 边界时多写一个 S1
        uint32_t lastIdx = gmCoord.gS1Idx + gmCoord.gS1DealSize - 1;
        uint32_t s1IdxEnd = lastIdx / gSize;
        uint32_t gIdxEnd = lastIdx % gSize;
        uint32_t blockLen = sizeof(T);
        // LSE_TN2G: G 维连续(dstStride=0); LSE_N2GT: 相邻 G 间隔 T; LSE_BN2GS1: 相邻 G 间隔 S1
        uint64_t dstStride = (offsetCalculator.GetStrideG() - 1) * sizeof(T); // 单位为Byte
        uint64_t ubOffset = 0;

        // gSize==1(如 MHA, Nq==Nkv)时一行即一个 S1, 相邻行在 GM 上等距, 整段一次跳写
        if (gSize == 1) {
            uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, 0, s1IdxStart);
            uint64_t rowStride = offsetCalculator.GetStrideS1();
            SafeStrideCopy<T>(dstTensor.gmTensor[gmOffset], lseSrc, gmCoord.gS1DealSize, blockLen, 0,
                              (rowStride - 1) * sizeof(T));
            return;
        }

        // 单循环: 首个 S1 从 gIdxStart 起写, 中间写满 gSize 行, 末个 S1 写至 gIdxEnd
        uint32_t startG = gIdxStart;
        uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, startG, s1IdxStart);
        for (uint32_t s1Idx = s1IdxStart; s1Idx <= s1IdxEnd; s1Idx++) {
            uint32_t blockCount = (s1Idx != s1IdxEnd) ? (gSize - startG) : (gIdxEnd + 1 - startG);
            SafeStrideCopy<T>(dstTensor.gmTensor[gmOffset], lseSrc[ubOffset], blockCount, blockLen, 0, dstStride);
            // 下一 S1 的起始位置(gIdx=0): 跨过一个 S1 间距, 回退本段已消费的 startG 行
            gmOffset += offsetCalculator.GetStrideS1() - startG * offsetCalculator.GetStrideG();
            startG = 0;
            ubOffset += blockCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        }
    }
};

template <typename T, GmFormat GM_FORMAT>
class CopyLseOutUbToGm<T, GM_FORMAT, UbFormat::GS1> {
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaGmTensorType& dstTensor, LocalTensor<T>& lseSrc, GmCoordGs1Merge& gmCoord)
    {
        if (gmCoord.gS1DealSize == 0) {
            return;
        }
        auto& offsetCalculator = dstTensor.offsetCalculator;
        // GS1 排布下 UB 一行 S1 的长度为实际 q 序列长度
        // BN2GS1 未传 actualSeqLengths 时按完整 S1 处理; TN2G/N2GT 的 T 维由长度信息推导, 缺失时不搬运
        uint64_t s1Size = 0;
        if (offsetCalculator.actualSeqLensQParser.GetActualLenDims() != 0) {
            s1Size = offsetCalculator.actualSeqLensQParser.GetActualSeqLength(gmCoord.bIdx);
        } else if constexpr (GM_FORMAT == GmFormat::LSE_BN2GS1) {
            s1Size = offsetCalculator.GetDimS1();
        }
        if (s1Size == 0) {
            return;
        }
        uint32_t gIdxStart = gmCoord.gS1Idx / s1Size;
        uint32_t s1IdxStart = gmCoord.gS1Idx % s1Size;
        // 末行索引(含), 避免 deal 恰好结束在 G 边界时多写一个 G
        uint32_t lastIdx = gmCoord.gS1Idx + gmCoord.gS1DealSize - 1;
        uint32_t gIdxEnd = lastIdx / s1Size;
        uint32_t s1IdxEnd = lastIdx % s1Size;
        uint32_t blockLen = sizeof(T);
        // LSE_BN2GS1: S1 维连续(dstStride=0); LSE_N2GT: T 维连续(dstStride=0); LSE_TN2G: 相邻 T 间隔 N2*G
        uint64_t dstStride = 0;
        if constexpr (GM_FORMAT == GmFormat::LSE_TN2G) {
            dstStride = (offsetCalculator.GetStrideT() - 1) * sizeof(T); // 单位为Byte
        }
        uint64_t ubOffset = 0;

        // 单循环: 首个 G 从 s1IdxStart 起写, 中间写满 actS1 行, 末个 G 写至 s1IdxEnd
        uint32_t startS = s1IdxStart;
        uint64_t gmOffset = offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gIdxStart, startS);
        for (uint32_t gIdx = gIdxStart; gIdx <= gIdxEnd; gIdx++) {
            uint32_t blockCount = (gIdx != gIdxEnd) ? (s1Size - startS) : (s1IdxEnd + 1 - startS);
            SafeStrideCopy<T>(dstTensor.gmTensor[gmOffset], lseSrc[ubOffset], blockCount, blockLen, 0, dstStride);
            // 下一 G 的起始位置(s1Idx=0): 跨过一个 G 间距, 回退本段已消费的 startS 行
            gmOffset += offsetCalculator.GetStrideG() - startS * offsetCalculator.GetStrideS1();
            startS = 0;
            ubOffset += blockCount * AttentionCommon::FP32_BLOCK_ELEMENT_NUM;
        }
    }
};

// 单头(DN 范式, G 已合入 N2)整段搬出: 一行即一个 S1, 且 S1 落在 GM 最内轴, 天然连续
// 无分段循环, 仅一次 DataCopyPad, 供 DN kernel 编译期选用
template <typename T, GmFormat GM_FORMAT>
class CopyLseOutUbToGm<T, GM_FORMAT, UbFormat::S1_ONLY> {
public:
    template <typename FaGmTensorType>
    __aicore__ inline void operator()(FaGmTensorType& dstTensor, LocalTensor<T>& lseSrc, GmCoordS1Only& gmCoord)
    {
        if (gmCoord.s1DealSize == 0) {
            return;
        }
        uint64_t gmOffset =
            dstTensor.offsetCalculator.GetOffset(gmCoord.bIdx, gmCoord.n2Idx, gmCoord.gIdx, gmCoord.s1Idx);
        SafeStrideCopy<T>(dstTensor.gmTensor[gmOffset], lseSrc, gmCoord.s1DealSize, sizeof(T), 0, 0);
    }
};

#endif
