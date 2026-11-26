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
 * \file quant_sparse_lightning_indexer_service_vector_arch22.h
 * \brief
 */
#ifndef QUANT_SPARSE_LIGHTNING_INDEXER_SERVICE_VECTOR_H
#define QUANT_SPARSE_LIGHTNING_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "quant_sparse_lightning_indexer_common_arch22.h"
#include "quant_sparse_lightning_indexer_vector.h"

namespace QSLIKernel {
using namespace QSLICommon;
using namespace QSLIServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t BASE_TOPK_VALUE_IDX_SIZE = 4096;
constexpr uint32_t ELE_NUM_32 = 32;
constexpr uint32_t ELE_NUM_128 = 128;
constexpr uint32_t ELE_NUM_512 = 512;

// ============ tmpBuf_(64KB) 内分区表, 单位: float 元素下标 ============
// candidate 掩码/距离计算区的各段, 后段由前段释放的 scratch 复用 (见各段注释)
constexpr int64_t TMP_BUF_BYTES = 64 * 1024;
constexpr int64_t TMP_FLOAT_NUM = TMP_BUF_BYTES / sizeof(float); // 16384
constexpr int64_t TMP_CAND_SORT_TMP_OFF = 4096; // [tileBlkNum] SortAll 临时区, 随后复用为 blkIdxF
constexpr int64_t TMP_CAND_ACC_OFF = 4352;      // [tileBlkNum] acc, 到候选块的最小距离
constexpr int64_t TMP_CAND_DIFF_OFF = 4608;     // [tileBlkNum] diff, |blkIdx - cand| 暂存
constexpr int64_t TMP_CAND_INT_OFF = 12288;     // 候选整数输入, Cast 后复用为位置级 posDist
// S12 (方案 C') 二维比较归约区: candVals 紧凑候选值 + mat 广播比较矩阵
// candVals: [candBlocks=2048] fp32 = 8KB, 占 [6144, 8192) — 与 candSortTmp [4096,8192)
// 的尾段重叠, 但 SortAll 仅 tile0 跑且结果已落 candPairs(candBuf_), 时序上 GatherMask 在后, 安全
// mat: [TMP_CAND_MAT_ROWS=32][tileBlkNum=256] fp32 = 32KB, 占 [8192, 16384) — 与
// TMP_CAND_INT_OFF(12288) 的 candInt/posDist 及 isOutI32(14336..) 区重叠:
// mat 归约完成在 posDist(Brcb)/isOutI32(Cast) 写之前, candInt 入口即消费 — 复用安全
constexpr int64_t TMP_CAND_VALS_OFF = 6144; // [candBlocks=2048] 紧凑候选值 (GatherMask 抽取)
constexpr int64_t TMP_CAND_MAT_ROWS = 32;   // mat 行数 (候选批大小, 2 的幂便于折半归约)
constexpr int64_t TMP_CAND_MAT_OFF = 8192;  // mat 区 [8192, 16384), 恰为 tmpBuf_ 尾
constexpr int64_t TMP_CAND_MAT_ELEMS = TMP_CAND_MAT_ROWS * 256; // 最大 tileBlkNum=256 → 8192 fp32

template <typename QSLIT>
class QSLIVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QSLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QSLIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = QSLIT::pageAttention;
    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;
    using K_T = typename QSLIT::keyType;

    __aicore__ inline QSLIVector(){};
    __aicore__ inline void ProcessVec0(const QSLICommon::RunInfo& info);
    __aicore__ inline void ProcessVec1(const QSLICommon::RunInfo& info);
    // 候选任务流水（AIV 侧入口）：逐 token 逐 tile gather K/kscale → staging，并按 token 消费分数
    __aicore__ inline void RunCandTask(const QSLICommon::RunInfo& info);
    __aicore__ inline void InitBuffers(TPipe* pipe);
    __aicore__ inline void InitParams(const struct QSLICommon::ConstInfo& constInfo,
                                      const struct QSLICommon::LdSplitCoreInfo& ldInfo,
                                      const QSLITilingDataConsumer* __restrict tilingData);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm, GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                  GlobalTensor<float> vec1ResGm);
    __aicore__ inline void InitVecCandidateTensor(GlobalTensor<int32_t> candidateBlockIndicesInGm,
                                                  GlobalTensor<int32_t> candidateBlockLengthGm,
                                                  GlobalTensor<int32_t> outputIdxOffsetGm, bool outputIdxOffsetValid);
    // 候选 gather 的源与目标：K 源、staging 区、分数区、kscale 辅助区
    __aicore__ inline void InitCandGatherTensor(GlobalTensor<K_T> keyGm, GlobalTensor<int8_t> stagingKeyGm,
                                                GlobalTensor<float> candScoreGm, GlobalTensor<half> candAuxScaleGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<half> weightsGm, GlobalTensor<half> qScaleGm,
                                              GlobalTensor<half> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline int32_t AlignS2(int32_t cuS2Len);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe* pipe);
    // 核尾排空候选分数消费队列（跨任务延迟消费的最后几条记录）
    __aicore__ inline void DrainPendingCandScore();

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<half> weightsGm;
    GlobalTensor<half> qScaleGm;
    GlobalTensor<half> kScaleGm;
    GlobalTensor<half> vec0OutGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> candidateBlockIndicesInGm;
    GlobalTensor<int32_t> candidateBlockLengthGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<K_T> keyGm_;              // 候选 gather 的 K 源（PA 分页物理布局）
    GlobalTensor<int8_t> candStagingKeyGm; // AIV 聚集的候选 K（8 槽 × 512 token × 128B）
    GlobalTensor<float> candScoreGm;       // AIC 写的候选分数区（2 任务 parity × 128 tile × 512 float）
    GlobalTensor<half> candAuxScaleGm;     // AIV 写的逐块 kscale（4 任务 parity × 128 tile × 1KB）
    // =================================常量区=================================

private:
    __aicore__ inline void GetKeyScale(const QSLICommon::RunInfo& runInfo, const LocalTensor<half>& resUb,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // candidate (two-level topk)
    __aicore__ inline int32_t CountGE(const LocalTensor<float>& sortedDesc, int32_t n, float x);
    // 候选任务流水：AIV 逐 token 逐 tile gather K/kscale 到 staging，并按 token 消费分数
    __aicore__ inline void GatherCandTile(const QSLICommon::RunInfo& info, uint32_t tokenIdx, uint32_t absTile,
                                          uint64_t auxSlotBase);
    __aicore__ inline void ConsumeCandScoreToken(const QSLICommon::RunInfo& info, uint32_t tokenIdx,
                                                 uint64_t scoreSlotBase, uint64_t auxSlotBase);
    __aicore__ inline void RecordPendingCandScore(const QSLICommon::RunInfo& info, uint32_t tokenIdx);
    __aicore__ inline void ConsumePendingCandScore();
    __aicore__ inline void EmptyCandToken(const QSLICommon::RunInfo& info, uint32_t tokenIdx);
    __aicore__ inline bool OwnsCandToken(uint32_t tokenIdx) const
    {
        return (tokenIdx & 1U) == static_cast<uint32_t>(blockId_ & 1);
    }
    __aicore__ inline void FinishCandToken(const QSLICommon::RunInfo& info, int32_t row, uint32_t tokenIdx);
    __aicore__ inline void BuildCandidateMask(const QSLICommon::RunInfo& info, int32_t cuS1Idx, int32_t cuBaseS2Idx,
                                              int32_t innerS1Idx);
    // candidate 向量区间封装: 按 B32_VEC_ELM_NUM 分块 (本版本 Level-2 大 count 的 vmax 等指令会触发 aicore 异常)
    __aicore__ inline void VecMaxPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                      const LocalTensor<float>& src1, int32_t n);
    __aicore__ inline void VecMinPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                      const LocalTensor<float>& src1, int32_t n);
    __aicore__ inline void VecAbs(const LocalTensor<float>& dst, const LocalTensor<float>& src, int32_t n);
    __aicore__ inline void VecMinsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src, float v,
                                         int32_t n);
    __aicore__ inline void VecMaxsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src, float v,
                                         int32_t n);
    __aicore__ inline void VecSubPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                      const LocalTensor<float>& src1, int32_t n);
    __aicore__ inline void VecAddsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src, float v,
                                         int32_t n);
    __aicore__ inline void VecMulPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                      const LocalTensor<float>& src1, int32_t n);
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECIN, 1> inQueue_;
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    // tmp buff for candidate (two-level topk)
    TBuf<TPosition::VECCALC> candBuf_;       // consumer: 排序后的候选块索引 [2048, 2] (value+idx)
    TBuf<TPosition::VECCALC> candConstBuf_;  // consumer: NEG_INF 常量 [2048] (位型 0xFF800000)
    TBuf<TPosition::VECCALC> candPosBuf_;    // 候选驱动: 0..7 位置模式 [8]
    TBuf<TPosition::VECCALC> candGatherBuf_; // 候选驱动: gather 暂存（16 块 K=16KB + 16 块 kscale=512B）

    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<int32_t> candIsOut_; // consumer: position 级 int 0/1 (1=候选外), 供 int32 位域无效化链
    LocalTensor<float> candNegInf_;  // -inf 位型常量 (int32 视图消费)
    GlobalTensor<int32_t> outputIdxOffsetGm_; // A15: 每行输出索引偏移 (仅 sparse_indices)
    bool isOutputIdxOffsetValid_ = false;     // A15: offset 是否传入 (经 InitVecCandidateTensor 传递)
    bool candidateDriven_ = false;            // 非空 candidate_block_length: 按 512 token 候选 tile 执行

    int32_t blockId_ = -1;
    // P3: topk 自适应宽度 — 累加器/抽取/LD 归并按 sparseCount 收缩 (64 对齐),
    // 动态 topk (topk<2048) 时消除固定 2048 宽度的无效排序合并与搬运
    uint32_t topkWidth_ = BASE_TOPK; // Align(sparseCount, B32_VEC_ELM_NUM)
    uint32_t topkPairWidth_ = 4096;  // 2 * topkWidth_ (value+index 对)
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;

    // para for LD
    uint32_t mrgListNum_ = 4;

    // 候选流水：gather 生产序号（槽号 = 本值 % CAND_RING_SLOTS）与跨任务延迟消费队列
    uint32_t candProdTileSeq_ = 0;
    struct PendingCandScore {
        bool valid = false;
        QSLICommon::RunInfo runInfo{};
        uint32_t tokenIdx = 0;
    };
    PendingCandScore pendingScore_[CAND_PENDING_SCORE_SLOTS]{};
    uint32_t pendingHead_ = 0;
    uint32_t pendingCount_ = 0;

    struct QSLICommon::ConstInfo constInfo_;
    struct QSLICommon::LdSplitCoreInfo ldInfo_;
};

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::GetKeyScale(const QSLICommon::RunInfo& runInfo,
                                                      const LocalTensor<half>& resUb, int64_t batchId, int64_t startS2,
                                                      int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    if constexpr (PAGE_ATTENTION) {
        // A11: k_scale 0 轴非连续 — 块基址 = 表值 x keyDequantScaleStride0 (真实 stride);
        // 0 时兜底原紧凑公式 (kCacheBlockSize_), 现网行为不变 (R8)
        // P13: 页跨度提升 int64 后再乘 — paged 混合缓存可超 2^31 个 scale 元素,
        // blockId * stride 的 int32 乘法在物理块号 >= ceil(2^31/stride) 时溢出
        // (stride=73856 时首个溢出块 29077), 负偏移符号扩展后越界/静默错读。
        // 必须在乘法前提升: static_cast<int64_t>(blockId * stride32) 已溢出无效。
        // 回归用例: qli_cand_test/test_qli_cand.py::test_qli_p13_kscale_int32_overflow
        int64_t kScaleBlkStride = constInfo_.keyDequantScaleStride0 != 0 ?
                                      static_cast<int64_t>(constInfo_.keyDequantScaleStride0) :
                                      static_cast<int64_t>(kCacheBlockSize_);
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        // P13 同型防御: batchId * maxBlockNumPerBatch_ 的 int32 乘法在 block_table
        // 总项超 2^31 (≈8.6GB 表) 时溢出 — 提前提升 int64
        int64_t blockTableBatchOffset = static_cast<int64_t>(batchId) * maxBlockNumPerBatch_;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(half);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb, kScaleGm[blockId * kScaleBlkStride + startBlockTableOffset], copyInParams,
                                 padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(half);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(half);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPad(resUb[resUbBaseOffset + i * kCacheBlockSize_], kScaleGm[blockId * kScaleBlkStride],
                                 copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(half);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(resUb, kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::InitBuffers(TPipe* pipe)
{
    pipe->InitBuffer(inQueue_, 2, s2BaseSize_ * sizeof(float) * 2); // 32KB
    pipe->InitBuffer(outQueue_, 1, BASE_TOPK * sizeof(float));      // 8 KB
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));     // 8 KB
    pipe->InitBuffer(tmpBuf_, 64 * 1024);                           // 64KB
    // P3: 累加器按 topk 宽度分配 (topk<2048 时省 UB; 2048 时与原一致)
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * topkPairWidth_ * sizeof(float));

    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    globalTopkNum_ = 0;

    // consumer 算子: 仅 mode=2 (candidate consumer) UB
    // 两条互斥路径的 UB 分配：
    //  - 全量扫（空 candidate_block_length）：candBuf_ 按行分区（每 AIV 2 行，行间 tile0
    //    重排序会互相覆盖）2 行 × 2048 对 × 8B = 32KB + negInf 8KB；
    //  - 候选驱动：candBuf_/negInf 不使用，改为 gather 暂存 18KB（16 块 K=16KB + 16 块 kscale 512B）。
    pipe->InitBuffer(candPosBuf_, ELE_NUM_32); // 候选驱动: 0..7 位置模式 (32B)
    ArithProgression(candPosBuf_.Get<int32_t>(), 0, 1, 8);
    PipeBarrier<PIPE_V>();
    if (candidateDriven_) {
        pipe->InitBuffer(candGatherBuf_, 18 * 1024);
    } else {
        pipe->InitBuffer(candBuf_, CeilDiv(s1BaseSize_, 2) * BASE_TOPK * 2 * sizeof(float));
        pipe->InitBuffer(candConstBuf_, BASE_TOPK * sizeof(float)); // 8KB: negInf
        candNegInf_ = candConstBuf_.Get<float>();
        Duplicate(candNegInf_.template ReinterpretCast<int32_t>(), QSLIServiceVec::NEG_INF, BASE_TOPK);
        PipeBarrier<PIPE_V>();
    }

    // 基本块执行前初始化UB和GM
    // step1. 初始化一个有序索引 0 - s2BaseSize_
    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    // step2. globalTopkUb_ [CeilDiv(s1BaseSize_, 2), BASE_TOPK, 2]   -inf,-1
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * topkPairWidth_);
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::InitLDBuffers(TPipe* pipe)
{
    pipe->Reset();
    // P3: LD 归并缓冲按 topk 实际宽度收缩 (mrgListNum_ 列表 × 每列表 value+idx 对)
    pipe->InitBuffer(ldToBeMrgBuf_, topkPairWidth_ * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldTmpBuf_, topkPairWidth_ * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldOutValueBuf_, topkWidth_ * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, topkWidth_ * sizeof(int32_t));
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::InitParams(const struct QSLICommon::ConstInfo& constInfo,
                                                     const struct QSLICommon::LdSplitCoreInfo& ldInfo,
                                                     const QSLITilingDataConsumer* __restrict tilingData)
{
    this->constInfo_ = constInfo;
    this->ldInfo_ = ldInfo;
    candidateDriven_ = constInfo.candidateDriven;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize; // 4
    s2BaseSize_ = constInfo.s2BaseSize; // 2048
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    // P3: 按 sparseCount 计算实际 topk 工作宽度 (对齐 64; MergeSort/ExtractIndex 元素粒度需求)
    topkWidth_ =
        QSLICommon::Align(static_cast<uint32_t>(constInfo.sparseCount), static_cast<uint32_t>(B32_VEC_ELM_NUM));
    topkPairWidth_ = 2 * topkWidth_;
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::InitVecInputTensor(GlobalTensor<half> weightsGm, GlobalTensor<half> qScaleGm,
                                                             GlobalTensor<half> kScaleGm,
                                                             GlobalTensor<int32_t> indiceOutGm,
                                                             GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::InitVecWorkspaceTensor(GlobalTensor<half> vec0OutGm,
                                                                 GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                                 GlobalTensor<float> vec1ResGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec0OutGm = vec0OutGm;
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::InitVecCandidateTensor(GlobalTensor<int32_t> candidateBlockIndicesInGm,
                                                                 GlobalTensor<int32_t> candidateBlockLengthGm,
                                                                 GlobalTensor<int32_t> outputIdxOffsetGm,
                                                                 bool outputIdxOffsetValid)
{
    this->candidateBlockIndicesInGm = candidateBlockIndicesInGm;
    this->candidateBlockLengthGm = candidateBlockLengthGm;
    this->outputIdxOffsetGm_ = outputIdxOffsetGm; // A15
    this->isOutputIdxOffsetValid_ = outputIdxOffsetValid;
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::InitCandGatherTensor(GlobalTensor<K_T> keyGm,
                                                               GlobalTensor<int8_t> stagingKeyGm,
                                                               GlobalTensor<float> candScoreGm,
                                                               GlobalTensor<half> candAuxScaleGm)
{
    this->keyGm_ = keyGm;
    this->candStagingKeyGm = stagingKeyGm;
    this->candScoreGm = candScoreGm;
    this->candAuxScaleGm = candAuxScaleGm;
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::AllocEventID()
{}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::FreeEventID()
{}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    // init -1 and copy to output
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.sparseCount);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    QSLIServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.sparseCount);
    outQueue_.FreeTensor(valueULocal);
}

// 向量区间封装: 按 B32_VEC_ELM_NUM(一个 repeat 的 fp32 元素数)分块下发。
// 本版本 Level-2 count > 64 的 vmax/vmin/vabs 等指令会触发 aicore 异常, 故一律走分块。
// 前置条件: n 为 B32_VEC_ELM_NUM 的整数倍 (调用点均已按该粒度对齐)。
template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::VecMaxPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                     const LocalTensor<float>& src1, int32_t n)
{
    {
        const BinaryRepeatParams repeatParams = QSLIServiceVec::B32BinaryRepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QSLIServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Max(dst[off], src0[off], src1[off], mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::VecMinPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                     const LocalTensor<float>& src1, int32_t n)
{
    {
        const BinaryRepeatParams repeatParams = QSLIServiceVec::B32BinaryRepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QSLIServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Min(dst[off], src0[off], src1[off], mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::VecAbs(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                 int32_t n)
{
    {
        const UnaryRepeatParams repeatParams = QSLIServiceVec::B32RepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QSLIServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Abs(dst[off], src[off], mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::VecMinsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                        float v, int32_t n)
{
    {
        const UnaryRepeatParams repeatParams = QSLIServiceVec::B32RepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QSLIServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Mins(dst[off], src[off], v, mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::VecMaxsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                        float v, int32_t n)
{
    {
        const UnaryRepeatParams repeatParams = QSLIServiceVec::B32RepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QSLIServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Maxs(dst[off], src[off], v, mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::VecSubPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                     const LocalTensor<float>& src1, int32_t n)
{
    {
        const BinaryRepeatParams repeatParams = QSLIServiceVec::B32BinaryRepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QSLIServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Sub(dst[off], src0[off], src1[off], mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::VecAddsScalar(const LocalTensor<float>& dst, const LocalTensor<float>& src,
                                                        float v, int32_t n)
{
    {
        const UnaryRepeatParams repeatParams = QSLIServiceVec::B32RepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QSLIServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Adds(dst[off], src[off], v, mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::VecMulPair(const LocalTensor<float>& dst, const LocalTensor<float>& src0,
                                                     const LocalTensor<float>& src1, int32_t n)
{
    {
        const BinaryRepeatParams repeatParams = QSLIServiceVec::B32BinaryRepeatParams();
        int32_t off = 0;
        while (off < n) {
            uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
            uint8_t repeatTimes = 1;
            QSLIServiceVec::VecRepeatPlan(n - off, mask, repeatTimes);
            Mul(dst[off], src0[off], src1[off], mask, repeatTimes, repeatParams);
            off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
        }
    }
    PipeBarrier<PIPE_V>();
}

// 统计降序序列 (偶数 lane 为 value) 中 >= x 的元素个数, 二分实现
template <typename QSLIT>
__aicore__ inline int32_t QSLIVector<QSLIT>::CountGE(const LocalTensor<float>& sortedDesc, int32_t n, float x)
{
    int32_t lo = 0;
    int32_t hi = n;
    while (lo < hi) {
        int32_t mid = (lo + hi) / 2;
        if (sortedDesc.GetValue(2 * mid) >= x) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// consumer (use_candidate): 加载/排序候选行 (每行首个S2分片), 判定 tile 内块归属,
// 产出 position 级 isOut (fp32 0/1, 1=候选外) 与其 int32 形式
template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::BuildCandidateMask(const QSLICommon::RunInfo& info, int32_t cuS1Idx,
                                                             int32_t cuBaseS2Idx, int32_t innerS1Idx)
{
    int32_t candBlocks = static_cast<int32_t>(constInfo_.candidateTopkBlocks);
    int32_t blockSize = static_cast<int32_t>(constInfo_.candidateBlockSize);
    int32_t tileBlkNum = s2BaseSize_ / blockSize;
    int32_t tileBlockBase = cuBaseS2Idx / blockSize;
    // R6 修复: 同核多行 (每 AIV 处理 CeilDiv(s1BaseSize,2)=2 行) 共享 candBuf 时,
    // 后一行的 tile0 重排序会覆盖前行候选 (s2 内层循环按 gS1 块整体推进) — candBuf 按行分区
    LocalTensor<float> tmp = tmpBuf_.Get<float>();
    LocalTensor<float> candPairs = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    if (info.isFirstS2InnerLoop) {
        // 加载候选行到 tmp 尾部, 转 fp32 后与有序索引组成 [values | idx] 对并降序排序
        LocalTensor<int32_t> candInt = tmp[TMP_CAND_INT_OFF].template ReinterpretCast<int32_t>();
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = candBlocks * sizeof(int32_t);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        AscendC::DataCopyPad(
            candInt, candidateBlockIndicesInGm[info.candidateOutOffset + cuS1Idx * constInfo_.candidateTopkBlocks],
            copyInParams, padParams);
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
        PipeBarrier<PIPE_V>();
        Cast(candPairs, candInt, RoundMode::CAST_NONE, candBlocks);
        PipeBarrier<PIPE_V>();
        DataCopy(candPairs[candBlocks].template ReinterpretCast<int32_t>(), globalTopkIndice_, candBlocks);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> candSortTmp = tmp[TMP_CAND_SORT_TMP_OFF];
        QSLIServiceVec::SortAll(candPairs, candSortTmp, candBlocks);
        PipeBarrier<PIPE_V>();
        // V->S 围栏: 排序结果对后续标量 GetValue (CountGE 二分) 可见
        // (PipeBarrier<PIPE_V> 不足以保证标量读到 V 写数据; 参照 CANN topk_v200 的 SetFlag/WaitFlag<V_S> 模式)
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    }
    // 候选与 tile 块号 [tileBlockBase, tileBlockBase+tileBlkNum) 的最小距离, 0 即命中
    // (candBuf 按行分区, R6: 同核行间覆盖已修复)
    LocalTensor<float> candPairs2 = candBuf_.Get<float>()[innerS1Idx * candBlocks * 2];
    int32_t lo = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase));
    int32_t hi = CountGE(candPairs2, candBlocks, static_cast<float>(tileBlockBase + tileBlkNum));
    LocalTensor<float> blkIdxF = tmp[TMP_CAND_SORT_TMP_OFF]; // [tileBlkNum]
    LocalTensor<float> acc = tmp[TMP_CAND_ACC_OFF];          // [tileBlkNum]
    Cast(blkIdxF, globalTopkIndice_, RoundMode::CAST_NONE, tileBlkNum);
    PipeBarrier<PIPE_V>();
    Adds(blkIdxF, blkIdxF, static_cast<float>(tileBlockBase), tileBlkNum);
    PipeBarrier<PIPE_V>();
    Duplicate(acc.ReinterpretCast<int32_t>(), QSLIServiceVec::POS_INF_F32, tileBlkNum);
    PipeBarrier<PIPE_V>();
    // 降序排序下: 值 >= base+tileBlkNum 占 [0, hi), 落在 tile 内的候选占 [hi, lo), < base (含 -1 pad) 占 [lo, ...)
    // S12 (方案 C'): 逐候选串行循环 (每候选 1 次 UB 标量读 + 3 条向量 + 2 个 PipeBarrier)
    // 改为全向量化二维归约:
    //   1) GatherMask(pattern=1) 把 [value|idx] 对的偶数 lane (value) 一次抽成紧凑 candVals[candBlocks]
    //      — 消除逐候选 UB 标量读 (标量 pipe 热点之一)
    //   2) 命中段 [hi,lo) 的候选按 MAT_ROWS=32 一批广播减: mat[t][b] = blkIdxF[b] - candVals[k+t]
    //      (每行一条 向量-标量 Sub, 行间无依赖免 barrier), Abs 后对 t 维折半 Min 归约进 acc
    //      — 循环次数从 (lo-hi) 次串行 (带 barrier) 降为 ceil/32 次批, 每批仅折半层间 barrier
    // 语义等价: acc[b] = min_k |blkIdxF[b] - candVals[k]|, 距离 0 即命中 (与逐候选 min 一致)
    LocalTensor<float> candVals = tmp[TMP_CAND_VALS_OFF]; // [candBlocks] 紧凑候选值
    {
        AscendC::GatherMaskParams gmParams;
        gmParams.repeatTimes = Ceil(candBlocks * sizeof(float) * VALUE_AND_INDEX_NUM, VEC_REPEAT_BYTES);
        gmParams.src0BlockStride = 1;
        gmParams.src0RepeatStride = B32_VEC_REPEAT_STRIDE;
        gmParams.src1RepeatStride = 0;
        uint64_t rsvdCnt = 0;
        uint8_t src1Pattern = 1; // 固定模式1: 每个 repeat 取偶数索引元素 (value lane)
        AscendC::GatherMask(candVals, candPairs2, src1Pattern, false, static_cast<uint32_t>(0), gmParams, rsvdCnt);
        PipeBarrier<PIPE_V>();
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S); // 抽取结果对后续标量循环 (candVals.GetValue) 可见
    }
    LocalTensor<float> mat = tmp[TMP_CAND_MAT_OFF]; // [MAT_ROWS][tileBlkNum] 二维比较矩阵
    for (int32_t k = hi; k < lo; k += TMP_CAND_MAT_ROWS) {
        int32_t rows = (lo - k) < TMP_CAND_MAT_ROWS ? (lo - k) : TMP_CAND_MAT_ROWS;
        // 尾批 rows < MAT_ROWS 时, 折半归约会读到未写的行 — 先整矩阵填 +inf 垫底
        // (Min 归约对 +inf 幂等, 不影响有效行的结果)
        Duplicate(mat.ReinterpretCast<int32_t>(), QSLIServiceVec::POS_INF_F32, TMP_CAND_MAT_ELEMS);
        PipeBarrier<PIPE_V>();
        // 广播减: mat[t][.] = blkIdxF - candVals[k+t], 每行一条 Sub (行间写不同区域, 免 barrier)
        for (int32_t t = 0; t < rows; t++) {
            Adds(mat[t * tileBlkNum], blkIdxF, -candVals.GetValue(k + t), tileBlkNum);
        }
        PipeBarrier<PIPE_V>();
        VecAbs(mat, mat, rows * tileBlkNum);
        // t 维折半 Min 归约: 32→16→8→4→2→1, 比较长度 = stride*tileBlkNum, 层间 1 个 barrier (同地址读写)
        for (int32_t stride = TMP_CAND_MAT_ROWS / 2; stride >= 1; stride /= 2) {
            VecMinPair(mat, mat, mat[stride * tileBlkNum], stride * tileBlkNum);
            PipeBarrier<PIPE_V>();
        }
        // mat[0][.] 即本批各块的最近候选距离 — 并入全局 acc
        VecMinPair(acc, acc, mat, tileBlkNum);
        PipeBarrier<PIPE_V>();
    }
    // 距离 0 → 候选块; Brcb 展开到位置级; 距离 clamp 到 1 得 isOut (0/1)
    LocalTensor<float> posDist = tmp[TMP_CAND_INT_OFF]; // candInt 已释放, 可复用
    Brcb(posDist, acc, tileBlkNum / 8, {1, 8});
    PipeBarrier<PIPE_V>();
    VecMinsScalar(posDist, posDist, 1.0f, s2BaseSize_);
    PipeBarrier<PIPE_V>();
    // isOut 转为 int32 0/1 形态供位域无效化链使用: float 0.0/1.0 的位型是 0x00000000/0x3F800000,
    // 不能直接作为整数乘数; 值域仅 {0.0, 1.0}, Cast 后 0/1 精确。
    // 候选外位置（与不可达位置并轨）由 ProcessVec 的 int32 位域链压成 (-inf, -1), 不参与选择 —
    // 对齐模型 masked_fill(~candidates, -inf) 语义; 有效数不足 topk 的槽位输出 -1。
    LocalTensor<int32_t> isOutI32 = tmpBuf_.Get<int32_t>()[TMP_CAND_INT_OFF + s2BaseSize_];
    Cast(isOutI32, posDist, RoundMode::CAST_RINT, s2BaseSize_);
    PipeBarrier<PIPE_V>();
    candIsOut_ = isOutI32;
}

// 候选 gather：把本 (token, 候选 tile) 的 K 从 PA(Page Attention 分页注意力) 分页按块
// 聚集成连续的 512 token 段写入 staging；同时把逐块 kscale 写进辅助区（消费侧一次 DMA
// 读回，取代"每 tile 64 次 16B DMA + 64 次 block_table 标量读"）。
// 每个 AIV 负责本 tile 的一半（32 个候选块 = 32KB），按 16 块一个 chunk 复用 UB，
// 每 chunk 写完立即 MTE3 落 staging；两个 AIV 对同一 READY 的 set 合并为 1 个令牌/tile。
template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::GatherCandTile(const QSLICommon::RunInfo& info, uint32_t tokenIdx,
                                                         uint32_t absTile, uint64_t auxSlotBase)
{
    constexpr uint32_t CAND_BLOCKS_PER_TILE = QSLICommon::CAND_TILE_BLOCKS; // 64
    constexpr uint32_t CAND_BLOCKS_PER_AIV = CAND_BLOCKS_PER_TILE / 2U;     // 32
    constexpr uint32_t CAND_BLOCKS_PER_CHUNK = 16U;
    constexpr uint32_t CAND_TOKENS_PER_BLOCK = 8U;
    constexpr uint32_t CAND_BLOCK_BYTES = CAND_TOKENS_PER_BLOCK * 128U; // 1KB（int8 K, D=128）
    constexpr uint32_t CAND_BLOCK_HALVES = CAND_TOKENS_PER_BLOCK * 2U;  // kscale 每块 16 half（32B）
    const uint32_t aivHalf = static_cast<uint32_t>(blockId_ % 2);
    const uint32_t halfBlkBase = aivHalf * CAND_BLOCKS_PER_AIV;
    const uint32_t maxTiles = info.candidateMaxTiles;
    const int32_t row = static_cast<int32_t>(info.candidateRowBase + tokenIdx);
    int32_t len = candidateBlockLengthGm.GetValue(row);
    len = (len < 0) ?
              0 :
              ((len > static_cast<int32_t>(constInfo_.candidateTopkBlocks)) ? constInfo_.candidateTopkBlocks : len);
    // 本 tile 的有效 token 前缀与有效块数：超出 L 的块不搬（索引表尾部内容不可信，
    // 且这些位置的分数不会被消费）
    const uint32_t tileBase = absTile * QSLICommon::CAND_TILE_TOKENS;
    const uint32_t totalTokens = static_cast<uint32_t>(len) * CAND_TOKENS_PER_BLOCK;
    uint32_t validTokens = (totalTokens > tileBase) ? (totalTokens - tileBase) : 0U;
    if (validTokens > QSLICommon::CAND_TILE_TOKENS) {
        validTokens = QSLICommon::CAND_TILE_TOKENS;
    }
    const uint32_t validBlocks = (validTokens + CAND_TOKENS_PER_BLOCK - 1U) / CAND_TOKENS_PER_BLOCK;
    // 槽号 = 全局生产序号 % RING_SLOTS（与 AIC 的消费序号逐 tile 对齐：token 主序、tile 次序）
    const uint32_t ringSlot = candProdTileSeq_ % QSLICommon::CAND_RING_SLOTS;
    ++candProdTileSeq_;
    const uint64_t slotKeyBase = static_cast<uint64_t>(ringSlot) * QSLICommon::CAND_TILE_TOKENS * 128ULL +
                                 static_cast<uint64_t>(aivHalf) * (CAND_BLOCKS_PER_AIV * CAND_BLOCK_BYTES);
    // aux 区独立 4 周期寻址（与分数区的 2 周期脱钩，覆写链见 common 头 CAND_AUX_PERIOD）
    const uint64_t auxTileBase =
        2ULL * (auxSlotBase + static_cast<uint64_t>(tokenIdx * maxTiles + absTile) * QSLICommon::CAND_TILE_TOKENS) +
        static_cast<uint64_t>(aivHalf) * (CAND_BLOCKS_PER_AIV * CAND_BLOCK_HALVES);
    AscendC::CrossCoreWaitFlag(QSLICommon::CAND_FREE_FLAG);
    const uint32_t pageBlkNum = static_cast<uint32_t>(kCacheBlockSize_) / CAND_TOKENS_PER_BLOCK;
    if (pageBlkNum == 0U || validBlocks == 0U) {
        AscendC::CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(QSLICommon::CAND_READY_FLAG);
        return;
    }
    // 与 AIC 的 staging 地址口径一致：块基址 = 页表值 × keyStride0（0 时兜底紧凑 stride）
    const uint64_t blkStride = (constInfo_.keyStride0 != 0) ?
                                   static_cast<uint64_t>(constInfo_.keyStride0) :
                                   static_cast<uint64_t>(kCacheBlockSize_) * static_cast<uint64_t>(kHeadNum_) * 128ULL;
    const int64_t scaleStride = (constInfo_.keyDequantScaleStride0 != 0) ?
                                    static_cast<int64_t>(constInfo_.keyDequantScaleStride0) :
                                    static_cast<int64_t>(kCacheBlockSize_);
    const int64_t tableBase = static_cast<int64_t>(info.bIdx) * static_cast<int64_t>(maxBlockNumPerBatch_);
    const uint64_t rowOff = (static_cast<uint64_t>(row) * kHeadNum_ + info.n2Idx) * constInfo_.candidateTopkBlocks +
                            static_cast<uint64_t>(absTile) * CAND_BLOCKS_PER_TILE;
    LocalTensor<K_T> kUb = candGatherBuf_.Get<K_T>();
    LocalTensor<half> scaleUb = candGatherBuf_.Get<half>()[CAND_BLOCKS_PER_CHUNK * CAND_BLOCK_BYTES / 2];
    for (uint32_t chunk = 0U; chunk < CAND_BLOCKS_PER_AIV / CAND_BLOCKS_PER_CHUNK; chunk++) {
        const uint32_t blkBase = chunk * CAND_BLOCKS_PER_CHUNK;
        for (uint32_t i = 0U; i < CAND_BLOCKS_PER_CHUNK; i++) {
            const uint32_t blkInTile = halfBlkBase + blkBase + i;
            const int32_t c = (blkInTile < validBlocks) ? candidateBlockIndicesInGm.GetValue(rowOff + blkInTile) : -1;
            if (c < 0) {
                // 无效槽的 aux 必须显式写 0：陈旧位型可能是 NaN，score = raw × scale 会得 NaN
                AscendC::Duplicate(scaleUb[i * CAND_BLOCK_HALVES], static_cast<half>(0.0f), CAND_BLOCK_HALVES);
                continue;
            }
            const uint64_t page = static_cast<uint64_t>(c) / pageBlkNum;
            const uint64_t within = static_cast<uint64_t>(c) % pageBlkNum;
            const int32_t phys = blockTableGm.GetValue(tableBase + static_cast<int64_t>(page));
            const uint64_t kOff = static_cast<uint64_t>(phys) * blkStride + within * 8ULL * 128ULL;
            const uint64_t sOff = static_cast<uint64_t>(phys) * static_cast<uint64_t>(scaleStride) + within * 8ULL;
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            AscendC::DataCopyPadExtParams<K_T> kPad{false, 0, 0, 0};
            AscendC::DataCopyExtParams kParams;
            kParams.blockCount = 1;
            kParams.blockLen = CAND_BLOCK_BYTES * sizeof(K_T);
            kParams.srcStride = 0;
            kParams.dstStride = 0;
            kParams.rsv = 0;
            AscendC::DataCopyPad(kUb[i * CAND_BLOCK_BYTES], keyGm_[kOff], kParams, kPad);
            AscendC::DataCopyPadExtParams<half> sPad{false, 0, 0, 0};
            AscendC::DataCopyExtParams sParams;
            sParams.blockCount = 1;
            sParams.blockLen = CAND_TOKENS_PER_BLOCK * sizeof(half);
            sParams.srcStride = 0;
            sParams.dstStride = 0;
            sParams.rsv = 0;
            AscendC::DataCopyPad(scaleUb[i * CAND_BLOCK_HALVES], kScaleGm[sOff], sParams, sPad);
        }
        // 本 chunk 的 K/kscale 已搬进 UB：先与 MTE2 / V 管道对齐，再落 staging/aux
        SetWaitFlag<HardEvent::MTE2_MTE3>(HardEvent::MTE2_MTE3);
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3); // 覆盖无效槽的 V 管道清零写
        AscendC::DataCopyExtParams outParams;
        outParams.blockCount = 1;
        outParams.blockLen = CAND_BLOCKS_PER_CHUNK * CAND_BLOCK_BYTES;
        outParams.srcStride = 0;
        outParams.dstStride = 0;
        outParams.rsv = 0;
        AscendC::DataCopyPad(candStagingKeyGm[slotKeyBase + static_cast<uint64_t>(blkBase) * CAND_BLOCK_BYTES], kUb,
                             outParams);
        AscendC::DataCopyExtParams auxParams;
        auxParams.blockCount = 1;
        auxParams.blockLen = CAND_BLOCKS_PER_CHUNK * CAND_BLOCK_HALVES * sizeof(half);
        auxParams.srcStride = 0;
        auxParams.dstStride = 0;
        auxParams.rsv = 0;
        AscendC::DataCopyPad(candAuxScaleGm[auxTileBase + static_cast<uint64_t>(blkBase) * CAND_BLOCK_HALVES], scaleUb,
                             auxParams);
        // 下个 chunk 复用 UB 前必须等本 chunk 的 MTE3 读完（且与 V 管道的清零写对齐）
        SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
    AscendC::CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(QSLICommon::CAND_READY_FLAG);
}

// 候选任务末：把本 token 的 topk 累加器输出（非 LD）或溢写到 LD 工作区。
template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::FinishCandToken(const QSLICommon::RunInfo& info, int32_t row,
                                                          uint32_t tokenIdx)
{
    const uint32_t accIdx = tokenIdx % 2U;
    if (info.candidateNeedSpill) {
        AscendC::DataCopyExtParams copyWsParams;
        copyWsParams.blockLen = topkPairWidth_ * sizeof(float);
        copyWsParams.srcStride = 0;
        copyWsParams.dstStride = 0;
        copyWsParams.blockCount = 1;
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        const uint64_t wsOffset = static_cast<uint64_t>(info.candidateWsBase) * s1BaseSize_ * topkPairWidth_ +
                                  static_cast<uint64_t>(tokenIdx) * topkPairWidth_;
        AscendC::DataCopyPad(vec1ResGm[wsOffset], globalTopkUb_[accIdx * topkPairWidth_], copyWsParams);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        InitSortOutBuf(globalTopkUb_[accIdx * topkPairWidth_], topkPairWidth_);
        PipeBarrier<PIPE_V>();
    } else {
        LocalTensor<uint32_t> idxULocal = outQueue_.AllocTensor<uint32_t>();
        ExtractIndex(idxULocal, globalTopkUb_[accIdx * topkPairWidth_].template ReinterpretCast<uint32_t>(),
                     topkWidth_);
        PipeBarrier<PIPE_V>();
        InitSortOutBuf(globalTopkUb_[accIdx * topkPairWidth_], topkPairWidth_);
        outQueue_.EnQue<uint32_t>(idxULocal);
        idxULocal = outQueue_.DeQue<uint32_t>();
        QSLIServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + static_cast<uint64_t>(row) * constInfo_.sparseCount],
                                idxULocal.template ReinterpretCast<int32_t>(), constInfo_.sparseCount);
        outQueue_.FreeTensor(idxULocal);
    }
}

// 候选任务流水（AIV 侧）：逐 token 逐 tile gather K/kscale → staging，按 token 消费分数。
// 与 AIC 的 ComputeMm1Candidate 用 ring 的 READY/FREE + SCORE_READY 解耦，不再依赖全量扫的
// syncC1V1 逐 tile 握手，因此 gather 可以领先 compute 若干 tile。
template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::RunCandTask(const QSLICommon::RunInfo& info)
{
    // int4 候选驱动尚未实现：候选 K/kscale 的字节口径按 int8 固定 128B/token 编写。
    // tiling 与 metadata 已对 int4 关闭 candidateDriven；这里用 if constexpr 丢弃 int4 实例化，
    // 避免 int4b_t 触发 int8 专用的 DataCopyPad 重载（编译期）。
    if constexpr (!IsSameType<K_T, int4b_t>::value) {
        const uint32_t taskTokens = static_cast<uint32_t>(info.actMBaseSize / gSize_);
        if (taskTokens == 0U) {
            return;
        }
        const uint32_t taskSeq = info.candidateTaskSeq;
        const uint32_t maxTiles = info.candidateMaxTiles;
        const uint32_t tileBegin = info.candidateTileBegin;
        const uint32_t tileEnd = info.candidateTileEnd;
        // 分数区按任务奇偶双缓冲：任务 t 的 AIC 写会覆写任务 t-2 同 parity 区的分数，
        // 因此开跑前强制消费所有 taskSeq ≤ t-2 的队首记录（队列深度 ≤ 8，不会溢出）。
        while (pendingCount_ > 0U) {
            const PendingCandScore& front = pendingScore_[pendingHead_];
            if (!front.valid || front.runInfo.candidateTaskSeq + 2U > taskSeq) {
                break;
            }
            ConsumePendingCandScore();
        }
        if (maxTiles == 0U || tileEnd <= tileBegin) {
            // 空任务：没有 ring 流量与分数信号（AIC 侧同样提前返回），只做整行 -1 或空 partial
            for (uint32_t t = 0U; t < taskTokens; t++) {
                EmptyCandToken(info, t);
            }
            return;
        }
        const uint64_t scoreSlotBase =
            static_cast<uint64_t>(taskSeq & 1U) * QSLICommon::CAND_SCORE_TILES_PER_TASK * QSLICommon::CAND_TILE_TOKENS;
        const uint64_t auxSlotBase =
            static_cast<uint64_t>(taskSeq & 3U) * QSLICommon::CAND_SCORE_TILES_PER_TASK * QSLICommon::CAND_TILE_TOKENS;
        // 任务内 token 与 tile 交错：gather token t 后消费 token t-lag（滞后一拍保证分数写有
        // 落地余量）。分片 tile 数可小至 1，按 16 tile 的余量自适应，下限 2。
        const uint32_t sliceTiles = tileEnd - tileBegin;
        const uint32_t lagTokens = (sliceTiles >= 16U) ? 2U : ((16U + sliceTiles - 1U) / sliceTiles);
        for (uint32_t t = 0U; t < taskTokens; t++) {
            for (uint32_t tile = tileBegin; tile < tileEnd; tile++) {
                GatherCandTile(info, t, tile, auxSlotBase);
            }
            if (t >= lagTokens) {
                ConsumeCandScoreToken(info, t - lagTokens, scoreSlotBase, auxSlotBase);
            }
        }
        // drain：AIC 每任务 wait 的令牌数 = tile 数 + 1，最后一个 tile 也能错拍消费
        AscendC::CrossCoreSetFlag<QSLICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE3>(QSLICommon::CAND_READY_FLAG);
        // 末尾 lagTokens 个 token 入队，跨任务/核尾延迟消费
        const uint32_t pendBegin = (taskTokens > lagTokens) ? (taskTokens - lagTokens) : 0U;
        for (uint32_t t = pendBegin; t < taskTokens; t++) {
            RecordPendingCandScore(info, t);
        }
        while (pendingCount_ > QSLICommon::CAND_PENDING_SCORE_SLOTS - 1U) {
            ConsumePendingCandScore();
        }
    }
}

// 单 token 分数消费：逐候选 tile 读分数（512 float）与 aux kscale（1KB），乘 scale 后
// 排序归并进本 token 的 topk 累加器；任务末 tile 之后输出或溢写。
template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::ConsumeCandScoreToken(const QSLICommon::RunInfo& info, uint32_t tokenIdx,
                                                                uint64_t scoreSlotBase, uint64_t auxSlotBase)
{
    constexpr uint32_t TOKENS_PER_BLOCK = 8U;
    constexpr uint32_t CAND_TILE_TOKENS = QSLICommon::CAND_TILE_TOKENS;
    const uint32_t maxTiles = info.candidateMaxTiles;
    const uint32_t tileBegin = info.candidateTileBegin;
    const uint32_t tileEnd = info.candidateTileEnd;
    // AIC 每 token set 一次（广播给两个 AIV）；两个 AIV 都要 wait，保持计数守恒
    AscendC::CrossCoreWaitFlag(QSLICommon::CAND_SCORE_READY_FLAG);
    if (!OwnsCandToken(tokenIdx)) {
        return;
    }
    const int32_t row = static_cast<int32_t>(info.candidateRowBase + tokenIdx);
    int32_t len = candidateBlockLengthGm.GetValue(row);
    len = (len < 0) ?
              0 :
              ((len > static_cast<int32_t>(constInfo_.candidateTopkBlocks)) ? constInfo_.candidateTopkBlocks : len);
    // 两个口径必须分开：
    //  - candLimit = L×8：本行有效候选**槽位**数（循环上界；≥L 的表项为 -1 填充）；
    //  - visLimit：候选**位置**（块号×8 + 块内偏移）的可见上界 = 压缩后 K 长度（mask0），
    //    或因果可见长度 V（mask3，V = (压缩前 key 长度 - S1 长度 + 行号 + 1) / 压缩率，
    //    与全量扫路径的 cuRealAcSeq 同源）。
    // 候选块号是任意值，位置可远大于 L×8，故不能用 candLimit 当位置上界。
    const uint32_t candLimit = static_cast<uint32_t>(len) * TOKENS_PER_BLOCK;
    uint32_t visLimit = static_cast<uint32_t>(info.actS2Size);
    if (constInfo_.attenMaskFlag) {
        const int32_t s1Row = static_cast<int32_t>(info.gS1Idx * s1BaseSize_) + static_cast<int32_t>(tokenIdx);
        const int32_t base =
            static_cast<int32_t>(info.actS2SizeOrig) - static_cast<int32_t>(info.actS1Size) + s1Row + 1;
        int32_t v = (base > 0) ? (base / static_cast<int32_t>(constInfo_.cmpRatio)) : 0;
        if (v < 0) {
            v = 0;
        }
        visLimit = static_cast<uint32_t>(v);
    }
    const uint32_t accIdx = tokenIdx & 1U;
    // 累加器复位（空行也走同一路径，输出 -inf/-1 或空 partial）
    InitSortOutBuf(globalTopkUb_[accIdx * topkPairWidth_], topkPairWidth_);
    PipeBarrier<PIPE_V>();
    if (candLimit > 0U && maxTiles > 0U) {
        AscendC::DataCopyPadExtParams<float> scorePad{false, 0, 0, 0};
        AscendC::DataCopyExtParams scoreParams;
        scoreParams.blockCount = 1;
        scoreParams.srcStride = 0;
        scoreParams.dstStride = 0;
        scoreParams.rsv = 0;
        AscendC::DataCopyPadExtParams<half> auxPad{false, 0, 0, 0};
        AscendC::DataCopyExtParams auxParams;
        auxParams.blockCount = 1;
        auxParams.blockLen = QSLICommon::CAND_AUX_SCALE_HALVES_PER_TILE * sizeof(half); // 2KB
        auxParams.srcStride = 0;
        auxParams.dstStride = 0;
        auxParams.rsv = 0;
        uint64_t blkMask[2] = {0xFFULL, 0ULL}; // 每块前 8 个 fp32 的有效位
        // UB 分区（tmpBuf 内 float 下标）：scaleF 8192、idxTokF 9216、validF 9728、penF 10240、
        // oneF 10752、zeroF 11264、posPattern 11776；掩码缓冲取字节 47168/47680。
        LocalTensor<float> scaleF = tmpBuf_.Get<float>()[8192];       // [512] 去填充后的 kscale
        LocalTensor<float> idxTokF = tmpBuf_.Get<float>()[9216];      // [512] 逻辑 token 号（含 rowOff）
        LocalTensor<float> validF = tmpBuf_.Get<float>()[9728];       // [512] 位置有效 0/1
        LocalTensor<float> penF = tmpBuf_.Get<float>()[10240];        // [512] 1 - valid
        LocalTensor<float> oneF = tmpBuf_.Get<float>()[10752];        // [512] 1.0
        LocalTensor<float> zeroF = tmpBuf_.Get<float>()[11264];       // [512] 0.0
        LocalTensor<float> posPatternF = tmpBuf_.Get<float>()[11776]; // [8] 块内偏移 0..7
        LocalTensor<uint8_t> maskA = tmpBuf_.Get<uint8_t>()[47168];
        LocalTensor<uint8_t> maskB = tmpBuf_.Get<uint8_t>()[47680];
        // A15: 每行输出索引偏移（一并进入位置掩码，保证无效位置仍输出 -1）
        int32_t rowOff = 0;
        if (isOutputIdxOffsetValid_) {
            rowOff = outputIdxOffsetGm_.GetValue(info.outputIdxOffsetCoreOffset + row * kHeadNum_);
        }
        ArithProgression(posPatternF, 0.0f, 1.0f, 8);
        Duplicate(oneF, 1.0f, CAND_TILE_TOKENS);
        Duplicate(zeroF, 0.0f, CAND_TILE_TOKENS);
        PipeBarrier<PIPE_V>();
        const uint64_t candRowOff =
            (static_cast<uint64_t>(row) * kHeadNum_ + info.n2Idx) * constInfo_.candidateTopkBlocks;
        for (uint32_t tile = tileBegin; tile < tileEnd; tile++) {
            const uint32_t tileBase = tile * CAND_TILE_TOKENS;
            if (tileBase >= candLimit) {
                break;
            }
            uint32_t validTokens = candLimit - tileBase;
            if (validTokens > CAND_TILE_TOKENS) {
                validTokens = CAND_TILE_TOKENS;
            }
            const uint64_t scoreOff =
                scoreSlotBase + static_cast<uint64_t>(tokenIdx * maxTiles + tile) * CAND_TILE_TOKENS;
            const uint64_t auxOff = auxSlotBase + static_cast<uint64_t>(tokenIdx * maxTiles + tile) * CAND_TILE_TOKENS;
            // 1) 本 tile 的 512 分数 + 1KB kscale 一次读回（aux 每块 16 half，前 8 有效）
            LocalTensor<float> scoreUb = inQueue_.AllocTensor<float>();
            LocalTensor<half> auxUb = scoreUb[CAND_TILE_TOKENS].template ReinterpretCast<half>()[CAND_TILE_TOKENS];
            LocalTensor<float> padF = scoreUb[4 * CAND_TILE_TOKENS]; // [1024] aux 的 fp32 视图
            scoreParams.blockLen = validTokens * sizeof(float);
            AscendC::DataCopyPad(scoreUb, candScoreGm[scoreOff], scoreParams, scorePad);
            AscendC::DataCopyPad(auxUb, candAuxScaleGm[2ULL * auxOff], auxParams, auxPad);
            inQueue_.EnQue<float>(scoreUb);
            scoreUb = inQueue_.DeQue<float>();
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            // 2) aux 展开为 fp32 并按块去填充（16 half → 前 8 个）
            AscendC::Cast(padF, auxUb, RoundMode::CAST_NONE, QSLICommon::CAND_AUX_SCALE_HALVES_PER_TILE);
            PipeBarrier<PIPE_V>();
            for (uint32_t b = 0U; b < QSLICommon::CAND_TILE_BLOCKS; b++) {
                AscendC::Adds(scaleF[b * TOKENS_PER_BLOCK], padF[b * 16U], 0.0f, blkMask, static_cast<uint8_t>(1),
                              QSLIServiceVec::B32RepeatParams());
            }
            PipeBarrier<PIPE_V>();
            // 3) 位置表 + 掩码（位置级，对齐全量扫的 masked_fill 语义）：
            //   tok = 块号×8 + 块内偏移 + rowOff；无效块（表内 -1 或超出 L）取 -8+j（恒负）。
            //   valid = (tok >= rowOff) && (tok < rowOff + V)；无效位置索引置 -1、分数罚 1e31。
            for (uint32_t b = 0U; b < QSLICommon::CAND_TILE_BLOCKS; b++) {
                const uint32_t ci = tile * QSLICommon::CAND_TILE_BLOCKS + b;
                int32_t c = -1;
                if (static_cast<int32_t>(ci) < len) {
                    c = candidateBlockIndicesInGm.GetValue(candRowOff + ci);
                }
                const float seed =
                    (c >= 0) ? static_cast<float>(c * static_cast<int32_t>(TOKENS_PER_BLOCK) + rowOff) : -8.0f;
                AscendC::Adds(idxTokF[b * TOKENS_PER_BLOCK], posPatternF, seed, blkMask, static_cast<uint8_t>(1),
                              QSLIServiceVec::B32RepeatParams());
            }
            PipeBarrier<PIPE_V>();
            AscendC::Compares(maskA, idxTokF, static_cast<float>(rowOff), CMPMODE::GE, CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            AscendC::Compares(maskB, idxTokF, static_cast<float>(static_cast<int32_t>(visLimit) + rowOff), CMPMODE::LT,
                              CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            AscendC::Select(validF, maskA, oneF, zeroF, SELMODE::VSEL_TENSOR_TENSOR_MODE, CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            AscendC::Select(validF, maskB, validF, zeroF, SELMODE::VSEL_TENSOR_TENSOR_MODE, CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            Muls(penF, validF, -1.0f, CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            Adds(penF, penF, 1.0f, CAND_TILE_TOKENS); // penF = 1 - valid
            PipeBarrier<PIPE_V>();
            Mul(idxTokF, idxTokF, validF, CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            Sub(idxTokF, idxTokF, penF, CAND_TILE_TOKENS); // 无效位置 → -1
            PipeBarrier<PIPE_V>();
            // 4) 分数 = clamp(原始分) × kscale，再 clamp 后减去无效罚分
            //   （无效槽 staging 可能是陈旧/NaN 位型，先上界 clamp 防止 NaN 传播）
            Mins(scoreUb, scoreUb, 1.0e30f, CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            Mul(scoreUb, scoreUb, scaleF, CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            Mins(scoreUb, scoreUb, 1.0e30f, CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            Muls(penF, penF, 1.0e31f, CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            Sub(scoreUb, scoreUb, penF, CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            // 5) 排序并归并进本 token 的 topk 累加器
            const int32_t alignLen = AlignS2(static_cast<int32_t>(validTokens));
            LocalTensor<float> sortBuff = tmpBuf_.Get<float>();
            LocalTensor<float> sortIndiceUb = sortBuff[alignLen];
            Duplicate(sortBuff.template ReinterpretCast<int32_t>(), QSLIServiceVec::NEG_INF, alignLen);
            PipeBarrier<PIPE_V>();
            Adds(sortBuff, scoreUb, 0.0f, validTokens);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(scoreUb);
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            AscendC::Cast(sortIndiceUbInt, idxTokF, RoundMode::CAST_RINT, CAND_TILE_TOKENS);
            PipeBarrier<PIPE_V>();
            // 阈值剪枝：本 tile 最大分严格小于累加器当前第 topk 名时，本 tile 不可能改变
            // topk（相等仍归并，保证同分选择与不剪枝逐位一致）。
            LocalTensor<float> maxSlot = tmpBuf_.Get<float>()[12048]; // [64] ReduceMax 输出 + work
            AscendC::ReduceMax(maxSlot, sortBuff, maxSlot[32], static_cast<int32_t>(CAND_TILE_TOKENS), false);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
            const float tileMax = maxSlot.GetValue(0);
            const float curTail = globalTopkUb_.GetValue(accIdx * topkPairWidth_ + 2 * (topkWidth_ - 1));
            SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
            // v15：候选按 source 块分降序，tileMax 跨 tile 单调不增，curTail 单调不减。
            // 首次命中后，后续 tile 也不可能改变 topk，直接结束本行，与逐 tile 剪枝等价。
            if (tileMax < curTail) {
                break;
            }
            LocalTensor<float> tmpSortBuf = sortBuff[2 * alignLen];
            QSLIServiceVec::SortAll(sortBuff, tmpSortBuf, alignLen);
            PipeBarrier<PIPE_V>();
            QSLIServiceVec::MergeSort(globalTopkUb_[accIdx * topkPairWidth_], topkWidth_, sortBuff, alignLen,
                                      tmpSortBuf);
            PipeBarrier<PIPE_V>();
        }
    }
    FinishCandToken(info, row, tokenIdx);
}

// 空任务/空行：累加器复位后走统一的输出/溢写路径（-inf/-1 或空 partial）
template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::EmptyCandToken(const QSLICommon::RunInfo& info, uint32_t tokenIdx)
{
    if (!OwnsCandToken(tokenIdx)) {
        return;
    }
    const int32_t row = static_cast<int32_t>(info.candidateRowBase + tokenIdx);
    const uint32_t accIdx = tokenIdx & 1U;
    InitSortOutBuf(globalTopkUb_[accIdx * topkPairWidth_], topkPairWidth_);
    PipeBarrier<PIPE_V>();
    FinishCandToken(info, row, tokenIdx);
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::RecordPendingCandScore(const QSLICommon::RunInfo& info, uint32_t tokenIdx)
{
    PendingCandScore& rec = pendingScore_[(pendingHead_ + pendingCount_) % QSLICommon::CAND_PENDING_SCORE_SLOTS];
    rec.runInfo = info;
    rec.tokenIdx = tokenIdx;
    rec.valid = true;
    ++pendingCount_;
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::ConsumePendingCandScore()
{
    if (pendingCount_ == 0U) {
        return;
    }
    PendingCandScore& rec = pendingScore_[pendingHead_];
    if (rec.valid) {
        rec.valid = false;
        const uint64_t scoreSlotBase = static_cast<uint64_t>(rec.runInfo.candidateTaskSeq & 1U) *
                                       QSLICommon::CAND_SCORE_TILES_PER_TASK * QSLICommon::CAND_TILE_TOKENS;
        const uint64_t auxSlotBase = static_cast<uint64_t>(rec.runInfo.candidateTaskSeq & 3U) *
                                     QSLICommon::CAND_SCORE_TILES_PER_TASK * QSLICommon::CAND_TILE_TOKENS;
        ConsumeCandScoreToken(rec.runInfo, rec.tokenIdx, scoreSlotBase, auxSlotBase);
    }
    pendingHead_ = (pendingHead_ + 1U) % QSLICommon::CAND_PENDING_SCORE_SLOTS;
    --pendingCount_;
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::DrainPendingCandScore()
{
    while (pendingCount_ > 0U) {
        ConsumePendingCandScore();
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::ProcessVec0(const QSLICommon::RunInfo& info)
{
    // 只需要一个v核做
    if (blockId_ % 2 != 0) {
        return;
    }
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    // 计算输出w基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 4*64 + aic_offset
    int64_t vec0OutGmOffset = (info.loop % 2) * ((s1BaseSize_ * gSize_ * BLOCK_CUBE));
    // 计算输入weight的地址偏移，qScale的地址偏移与weight相同
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * qHeadNum_;
    // 当前需要计算的S1行数，处理尾块场景
    int32_t cuS1ProcNum = cuBaseS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuProcRealNum = cuS1ProcNum * gSize_;
    int32_t cuProcEleNum = QSLICommon::Align(cuProcRealNum, 32); // 32: UB对齐, 参照v1

    LocalTensor<half> inWeightsUb = inQueue_.AllocTensor<half>();
    LocalTensor<half> inQScaleUb = inWeightsUb[cuProcEleNum];
    AscendC::DataCopyPadExtParams<half> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.blockLen = cuProcRealNum * sizeof(half);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    AscendC::DataCopyPad(inWeightsUb, weightsGm[weightGmOffset], copyInParams, padParams);
    AscendC::DataCopyPad(inQScaleUb, qScaleGm[weightGmOffset], copyInParams, padParams);

    inQueue_.EnQue<half>(inWeightsUb);
    inWeightsUb = inQueue_.DeQue<half>();
    AscendC::Mul(inWeightsUb, inWeightsUb, inQScaleUb, cuProcEleNum);
    PipeBarrier<PIPE_V>();
    LocalTensor<half> resUb = outQueue_.AllocTensor<half>();
    AscendC::Brcb(resUb, inWeightsUb, static_cast<uint8_t>(cuProcEleNum / 8), {1, 8});
    inQueue_.FreeTensor(inWeightsUb);

    outQueue_.EnQue<half>(resUb);
    resUb = outQueue_.DeQue<half>();
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = cuProcRealNum * BLOCK_CUBE * sizeof(half);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    AscendC::DataCopyPad(vec0OutGm[vec0OutGmOffset], resUb, copyOutParams);
    outQueue_.FreeTensor(resUb);
}

template <typename QSLIT>
__aicore__ inline int32_t QSLIVector<QSLIT>::AlignS2(int32_t cuS2Len)
{
    // 限制：当前cuS2Len最大为2048，暂不考虑更长
    // 该函数目的是将cuS2Len对齐到形如 32*(4^n)*m 的形式 (m ∈ [1, 3])，方便后续sort/merge
    if (cuS2Len <= ELE_NUM_128) {
        return Align(cuS2Len, ELE_NUM_32);
    } else if (cuS2Len <= ELE_NUM_512) {
        return Align(cuS2Len, ELE_NUM_128);
    } else {
        return Align(cuS2Len, ELE_NUM_512);
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::ProcessVec1(const QSLICommon::RunInfo& info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    // 计算基本块基地址偏移 偶数循环 -> 0 + aic_offset  奇数循环 -> 4*2048 + aic_offset
    int64_t mmGmOffset = (info.loop % 2) * (s1BaseSize_ * s2BaseSize_);

    // cuS1BeginIdxPerAiv: 每个AIV的S1起始偏移
    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // cuS1ProcNumPerAiv: 每个AIv的S1计算量
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);
    // 基本块基地址偏移奇数核加一个S1地址偏移
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * s2BaseSize_;
    // 非首个基本块, M(S1)轴发生切换需要初始化
    if (info.loop != 0 && info.s2Idx == 0) {
        // globalTopkUb_ value,index=-inf,-1
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * topkPairWidth_);
        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    // cuRealAcSeq: 当前基本块S1对应的AcSeq
    int32_t cuRealAcSeq = info.actS2Size;
    int32_t cuRealAcSeqCount = 0;
    if (constInfo_.attenMaskFlag) {
        // attenMask true场景
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + cuS1BeginIdxPerAiv;
    }
    int32_t cuRealAcSeqIni = cuRealAcSeq;

    // LD输出S1方向偏移，保证2个Vector输出的内容连续
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeqCount += 1;
            cuRealAcSeq = (cuRealAcSeqCount + cuRealAcSeqIni) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        // 当前vec1ResGm对应S1的位置
        // P10: LD 布局 = 每核 s1Base 行预留区 (起点 = saveWorkSpaceIdx * s1Base * pairW),
        // 行内槽偏移必须用**块内相对行号** (cuS1Idx - cuBaseS1Idx, 取值 0..s1Base-1):
        // 归约侧 ProcessLD 按 metadata 的 fdMStart(块内 0 起) + j 寻址, 两侧坐标系必须一致。
        // 若用全局行号 cuS1Idx, gS1Idx>=1 时会整体多偏移 gS1Idx*s1Base 行 — 溢写落到
        // 其它分片的槽行区, 归约读到陈旧/错行数据 (相邻行互换、整行全 0)。
        int32_t blockRowIdx = cuS1Idx - cuBaseS1Idx;
        uint64_t wsOffset = static_cast<uint64_t>(info.saveWorkSpaceIdx) * s1BaseSize_ * topkPairWidth_ +
                            static_cast<uint64_t>(blockRowIdx) * topkPairWidth_;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = AlignS2(cuS2Len);
            LocalTensor<float> mmInUb = inQueue_.AllocTensor<float>();
            LocalTensor<float> kScaleUb = mmInUb[cuS2LenVecAlign];
            LocalTensor<half> kScaleTUb = kScaleUb.template ReinterpretCast<half>()[cuS2LenVecAlign];
            AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
            AscendC::DataCopyPadExtParams<half> padTParams{false, 0, 0, 0};
            AscendC::DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = cuS2Len * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            copyInParams.rsv = 0;
            AscendC::DataCopyPad(mmInUb, mm1ResGm[mmGmOffset + innerS1Idx * s2BaseSize_], copyInParams, padParams);
            GetKeyScale(info, kScaleTUb, info.bIdx, cuBaseS2Idx, cuS2Len);
            inQueue_.EnQue<float>(mmInUb);
            mmInUb = inQueue_.DeQue<float>();
            AscendC::Cast(kScaleUb, kScaleTUb, RoundMode::CAST_NONE, cuS2Len);
            PipeBarrier<PIPE_V>();
            AscendC::Mul(mmInUb, mmInUb, kScaleUb, cuS2Len);
            PipeBarrier<PIPE_V>();
            // consumer: 候选外位置无效化（int32 位域链, NaN-free — 同 PR 14077 方案）:
            //   score' = score + isOut×(0xFF800000 − score)：isOut=1 时模 2^32 回绕互消 ≡ -inf 位型;
            //           isOut=0 时 pen=0、位型不动（score 为 ±inf/NaN 的极端输入也逐位安全）
            //   idx'   = idx   − isOut×(idx + 1)          ：isOut=1 → -1; isOut=0 → 原值
            // 候选外可达位置与不可达位置并轨为同一档无效位 (-inf, -1)，不参与选择（对齐模型
            // masked_fill(~candidates, -inf)）; 有效数不足 topk 的槽位由累加器垫底对自然补齐为 -1。
            // 不能在 float 域用 -inf: (-inf − x)×0 = NaN 会污染有效位。
            // P11: 全候选快路径 — 若该批总块数 ≤ candidateTopkBlocks, 候选覆盖全部
            // 位置块, isOut 恒 0 (pen 恒 0, score 不降级), 整个 BuildCandidateMask+pen
            // 链可跳过 (生产 s2=16384: 2048 块 ≤ 2048 候选 → 恒命中)。逐块判定用
            // info.actS2SizeOrig (压缩前) — 候选块号域即压缩前位置域。
            // P11 撤销 (2026-09-17): 全候选快路径的前提"候选表覆盖全部块号"是数据属性,
            // 无法从 shape/attr 判定 (上游候选可含重复/-1, kernel 内严格判定需逐点读候选表,
            // 成本 ≈ 被跳过的 tile 内命中检测, 无净收益); randint 候选用例证明上游无覆盖契约。
            // 恢复无条件 BuildCandidateMask+无效化链 (正确性优先)。
            BuildCandidateMask(info, cuS1Idx, cuBaseS2Idx, innerS1Idx);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> penI = tmpBuf_.Get<int32_t>()[TMP_CAND_SORT_TMP_OFF]; // blkIdxF/acc/diff 已释放, 复用
            LocalTensor<int32_t> scoreI = mmInUb.template ReinterpretCast<int32_t>();
            LocalTensor<int32_t> negInfI = candNegInf_.template ReinterpretCast<int32_t>();
            const BinaryRepeatParams binRepeatParams = QSLIServiceVec::B32BinaryRepeatParams();
            // (a) score 无效化; count 覆盖到对齐长度: [cuS2Len, cuS2LenVecAlign) 在 mmInUb 中是
            //     无效残留, 但该段不会被拷入排序缓冲 (仅拷 [0, cuS2Len)), 处理无副作用
            for (int32_t off = 0; off < cuS2LenVecAlign;) {
                uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
                uint8_t repeatTimes = 1;
                QSLIServiceVec::VecRepeatPlan(cuS2LenVecAlign - off, mask, repeatTimes);
                Sub(penI[off], negInfI[off], scoreI[off], mask, repeatTimes, binRepeatParams);
                off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
            }
            PipeBarrier<PIPE_V>();
            for (int32_t off = 0; off < cuS2LenVecAlign;) {
                uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
                uint8_t repeatTimes = 1;
                QSLIServiceVec::VecRepeatPlan(cuS2LenVecAlign - off, mask, repeatTimes);
                Mul(penI[off], penI[off], candIsOut_[off], mask, repeatTimes, binRepeatParams);
                off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
            }
            PipeBarrier<PIPE_V>();
            for (int32_t off = 0; off < cuS2LenVecAlign;) {
                uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
                uint8_t repeatTimes = 1;
                QSLIServiceVec::VecRepeatPlan(cuS2LenVecAlign - off, mask, repeatTimes);
                Add(scoreI[off], scoreI[off], penI[off], mask, repeatTimes, binRepeatParams);
                off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
            }
            PipeBarrier<PIPE_V>();
            LocalTensor<float> sortBuff = tmpBuf_.Get<float>();
            LocalTensor<float> sortScoreUb = sortBuff;
            LocalTensor<float> sortIndiceUb = sortBuff[cuS2LenVecAlign];
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), QSLIServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, mmInUb, 0.0f, cuS2Len);
            PipeBarrier<PIPE_V>();
            inQueue_.FreeTensor(mmInUb);
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            // 无效数据索引填充为-1
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
                PipeBarrier<PIPE_V>();
            }
            // A15: output_idx_offset 在索引成形阶段施加（对齐 arch35: 偏移先于 -1 填充）——
            // 累加器的 (-inf,-1) 垫底对不经过本加法, 输出 padding 恒为 -1;
            // candidate_block_indices 走独立路径 (相对块号), 不受影响。
            int32_t rowOff = 0;
            if (isOutputIdxOffsetValid_) {
                rowOff = outputIdxOffsetGm_.GetValue(info.outputIdxOffsetCoreOffset + cuS1Idx * kHeadNum_);
            }
            Adds(sortIndiceUbInt, globalTopkIndice_, static_cast<int32_t>(cuBaseS2Idx) + rowOff, cuS2Len);
            PipeBarrier<PIPE_V>();
            // (b) idx 无效化: idx' = idx − isOut×(idx + 1) —— isOut=1 → -1; isOut=0 → 原值。
            //     pad 区 idx 已为 -1: pen = -1+1 = 0, 链对 pad 为恒等变换, 故可覆盖到对齐长度。
            const UnaryRepeatParams unRepeatParams = QSLIServiceVec::B32RepeatParams();
            for (int32_t off = 0; off < cuS2LenVecAlign;) {
                uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
                uint8_t repeatTimes = 1;
                QSLIServiceVec::VecRepeatPlan(cuS2LenVecAlign - off, mask, repeatTimes);
                Adds(penI[off], sortIndiceUbInt[off], 1, mask, repeatTimes, unRepeatParams);
                off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
            }
            PipeBarrier<PIPE_V>();
            for (int32_t off = 0; off < cuS2LenVecAlign;) {
                uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
                uint8_t repeatTimes = 1;
                QSLIServiceVec::VecRepeatPlan(cuS2LenVecAlign - off, mask, repeatTimes);
                Mul(penI[off], penI[off], candIsOut_[off], mask, repeatTimes, binRepeatParams);
                off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
            }
            PipeBarrier<PIPE_V>();
            for (int32_t off = 0; off < cuS2LenVecAlign;) {
                uint64_t mask = QSLIServiceVec::B32_MASK_FULL;
                uint8_t repeatTimes = 1;
                QSLIServiceVec::VecRepeatPlan(cuS2LenVecAlign - off, mask, repeatTimes);
                Sub(sortIndiceUbInt[off], sortIndiceUbInt[off], penI[off], mask, repeatTimes, binRepeatParams);
                off += static_cast<int32_t>(repeatTimes) * static_cast<int32_t>(B32_VEC_ELM_NUM);
            }
            PipeBarrier<PIPE_V>();
            // 候选外位置（含不可达位置）统一输出 -1; 仅候选块内可达位置保留真实位置号。
            // 与模型 where(idxs < compress_lens, idxs + offset, -1) + masked_fill(~candidates, -inf) 一致。
            LocalTensor<float> tmpSortBuf = sortBuff[2 * cuS2LenVecAlign];
            QSLIServiceVec::SortAll(sortBuff, tmpSortBuf, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            QSLIServiceVec::MergeSort(globalTopkUb_[innerS1Idx * topkPairWidth_], topkWidth_, sortBuff, cuS2LenVecAlign,
                                      tmpSortBuf); // P3
            PipeBarrier<PIPE_V>();
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            // 中间结果保存
            if (needCopyOutGm && !info.isNeedLD) {
                LocalTensor<uint32_t> idxULocal = outQueue_.AllocTensor<uint32_t>();
                // P3: 只抽取前 topkWidth_ 对 (降序排序下 2048 名之后永不入选, 截断语义等价)
                ExtractIndex(idxULocal, globalTopkUb_[innerS1Idx * topkPairWidth_].template ReinterpretCast<uint32_t>(),
                             topkWidth_);
                PipeBarrier<PIPE_V>();
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                outQueue_.EnQue<uint32_t>(idxULocal);
                idxULocal = outQueue_.DeQue<uint32_t>();
                QSLIServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount],
                                        idxULocal.template ReinterpretCast<int32_t>(), constInfo_.sparseCount);
                outQueue_.FreeTensor(idxULocal);
            }
            // LD拷贝到当前vector对应S1的位置
            if (info.isNeedLD && info.isLastS2InnerLoop) { // 当前核存在归约任务 且是最后处理的一段
                AscendC::DataCopyExtParams copyWsParams;
                copyWsParams.blockLen = topkPairWidth_ * sizeof(float); // P3: 按实际 topk 宽度
                copyWsParams.srcStride = 0;
                copyWsParams.dstStride = 0;
                copyWsParams.blockCount = 1;
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                AscendC::DataCopyPad(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * topkPairWidth_], copyWsParams);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                PipeBarrier<PIPE_V>();
            }
        } else if (cuRealAcSeq <= 0) {
            // 无效长度处理
            if (info.isNeedLD && info.isLastS2InnerLoop) {
                PipeBarrier<PIPE_V>();
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                QSLIServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * topkPairWidth_],
                                        topkPairWidth_);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            } else {
                CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.sparseCount);
            }
        } else if (cuS2Len <= 0) {
            // LD拷贝到当前vector对应S1的位置
            if (info.isNeedLD && info.isLastS2InnerLoop) { // 当前核存在归约任务 且是最后处理的一段
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                QSLIServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * topkPairWidth_],
                                        topkPairWidth_);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
                InitSortOutBuf(globalTopkUb_[innerS1Idx * topkPairWidth_], topkPairWidth_);
                PipeBarrier<PIPE_V>();
            }
        }
    }

    // BNSD场景无效S1 输出-1
    if (Q_LAYOUT_T == LI_LAYOUT::BSND) {
        // 最后一个S1的基本块, 需要 >= info.actS1Size
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        // blockS2StartIdx_ == 0 控制S2从开始的核去做冗余清理
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.sparseCount);
            }
        }

        int32_t invalidS1Num2 = info.actS1Size - info.actS2SizeOrig;
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.sparseCount);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        // S2最后一个Loop后, 下一个基本块初始从0开始
        blockS2StartIdx_ = 0;
    }
}

template <typename QSLIT>
__aicore__ inline void QSLIVector<QSLIT>::ProcessLD()
{
    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
    LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    AscendC::DataCopyPadExtParams<float> indexValuePadParams{true, 0, 0, 0};
    AscendC::DataCopyExtParams indexValueParams;
    // P3: 每列表搬运/间距按实际 topk 对宽 (vec1ResGm 内相邻列表间距 = 3 对宽)
    indexValueParams.blockLen = topkPairWidth_ * sizeof(int32_t); // bytes
    // P10: 分段槽按核分布 (核 i 的区起点 = i*s1Base 槽), 跨段间距 = s1Base 槽。
    // 本函数所有拷贝均按段显式寻址 (blockCount 恒为 1), srcStride 不参与寻址, 置 0。
    // 注意不要改回 blockCount>1 的批量拷贝: 两种 srcStride 口径都实测过 —
    //   绝对间距口径 s1Base*pairW*4B/32、块间空隙口径 (s1Base-1)*pairW*4B/32 —
    // 在尾段为 2 段 (ldTailLen=2) 时都会读到错误槽位 (部分行错乱/含重复索引, 某些
    // S2 触发 507015), 逐块拷贝才正确. 详见提交 40207d6be 的验证记录。
    indexValueParams.srcStride = 0;
    indexValueParams.dstStride = 0;

    uint32_t ldProWorkspaceNum = ldInfo_.workspaceNum;
    uint32_t ldProcessLen = 4; // 4: 4块归约任务做一次Merge
    uint32_t ldProcessNum = (ldProWorkspaceNum - 1) / (ldProcessLen - 1);
    uint32_t ldTailLen = ldProWorkspaceNum - (ldProcessNum * (ldProcessLen - 1) + 1); // 尾块长度

    for (uint32_t j = 0; j < ldInfo_.mNum; j++) {
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        // 拷贝第一块归约任务
        indexValueParams.blockCount = 1;
        uint64_t wsOffsetIni = static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * topkPairWidth_ +
                               static_cast<uint64_t>(ldInfo_.mStart + j) * topkPairWidth_;
        AscendC::DataCopyPad(curValueIdxUb, vec1ResGm[wsOffsetIni], indexValueParams, indexValuePadParams);
        uint64_t valueOffset = topkPairWidth_;
        SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);

        // 处理等于4的部分
        for (uint32_t i = 0; i < ldProcessNum; i++) {
            // LD处理偏移
            uint64_t wsOffset = static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * topkPairWidth_ +
                                static_cast<uint64_t>(i * (ldProcessLen - 1) + 1) * s1BaseSize_ * topkPairWidth_ +
                                static_cast<uint64_t>(ldInfo_.mStart + j) * topkPairWidth_;
            indexValueParams.blockCount = 1; // P10: 逐段拷贝 (见上)

            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            // P10: 绕开 srcStride 语义不确定性 — 3 个分段槽逐个显式拷贝
            for (uint32_t k = 0; k < ldProcessLen - 1U; k++) {
                uint64_t segSlot = wsOffset + static_cast<uint64_t>(k) * s1BaseSize_ * topkPairWidth_;
                AscendC::DataCopyPad(curValueIdxUb[valueOffset + k * topkPairWidth_], vec1ResGm[segSlot],
                                     indexValueParams, indexValuePadParams);
            }
            // merge参数 (P3: elementLengths 按实际 topk 宽度)
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = topkWidth_;
            params.elementLengths[1] = topkWidth_;
            params.elementLengths[2] = topkWidth_;
            params.elementLengths[3] = topkWidth_;
            params.ifExhaustedSuspension = true;
            params.validBit = MRG_VALID_BIT_4;
            params.repeatTimes = 1;

            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[topkPairWidth_];
            srcList.src3 = curValueIdxUb[2 * topkPairWidth_];
            srcList.src4 = curValueIdxUb[3 * topkPairWidth_];
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, topkPairWidth_);
            PipeBarrier<PIPE_V>();
        }

        // 处理不等于4的部分
        if (ldTailLen != 0) {
            // 搬运尾块
            uint64_t wsOffsetTail =
                static_cast<uint64_t>(ldInfo_.workspaceIdx) * s1BaseSize_ * topkPairWidth_ +
                static_cast<uint64_t>(ldProcessNum * (ldProcessLen - 1) + 1) * s1BaseSize_ * topkPairWidth_ +
                static_cast<uint64_t>(ldInfo_.mStart + j) * topkPairWidth_;
            // 尾段同样逐块显式拷贝, 不使用 DataCopyExtParams.srcStride
            // (与该函数 group 路径一致: srcStride 的"块间空隙/绝对间距"语义不确定)
            for (uint32_t k = 0; k < ldTailLen; k++) {
                indexValueParams.blockCount = 1;
                uint64_t segSlotTail = wsOffsetTail + static_cast<uint64_t>(k) * s1BaseSize_ * topkPairWidth_;
                SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
                AscendC::DataCopyPad(curValueIdxUb[valueOffset + k * topkPairWidth_], vec1ResGm[segSlotTail],
                                     indexValueParams, indexValuePadParams);
            }
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = topkWidth_;
            params.elementLengths[1] = topkWidth_;
            params.elementLengths[2] = topkWidth_;
            params.elementLengths[3] = topkWidth_;
            params.ifExhaustedSuspension = true;
            if (ldTailLen == 1) {
                params.validBit = MRG_VALID_BIT_2;
            } else if (ldTailLen == 2) {
                params.validBit = MRG_VALID_BIT_3;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[topkPairWidth_];
            srcList.src3 = curValueIdxUb[2 * topkPairWidth_];
            srcList.src4 = curValueIdxUb[3 * topkPairWidth_];
            PipeBarrier<PIPE_V>();
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, topkPairWidth_);
            PipeBarrier<PIPE_V>();
        }

        // 搬出 (P3: Extract repeat = topkWidth_/32, 每 repeat 消费 32 对)
        Extract(outValueUb, outIdxUb, curValueIdxUb, (topkWidth_ / 32));
        PipeBarrier<PIPE_V>();
        InitSortOutBuf(curValueIdxUb, topkPairWidth_);
        LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
        // A15: output_idx_offset 已在 ProcessVec 的索引成形阶段施加, LD 归并只做 (value,index) 合并,
        // -1 垫底槽不经过偏移（对齐 arch35: 偏移先于 -1 填充）。
        SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
        uint64_t outOffset =
            ldInfo_.indiceOutCoreOffset + (ldInfo_.mStart + j) * constInfo_.kHeadNum * constInfo_.sparseCount;
        AscendC::DataCopyPad(indiceOutGm[outOffset], idxULocal1, copyOutParams);
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
}

} // namespace QSLIKernel
#endif // QUANT_SPARSE_LIGHTNING_INDEXER_SERVICE_VECTOR_H
