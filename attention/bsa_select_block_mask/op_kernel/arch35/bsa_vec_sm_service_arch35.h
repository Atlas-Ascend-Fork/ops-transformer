/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_VEC_SM_SERVICE_ARCH35_H
#define BSA_VEC_SM_SERVICE_ARCH35_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "bsa_select_block_mask_common_arch35.h"
#include "bsa_select_block_mask_tiling_data_arch35.h"
#include "bsa_vec_post_pool_service_arch35.h"

static constexpr uint32_t BSA_SM_SLICES_MAX = BSA_POST_POOL_UNITS_MAX + 2;

static constexpr uint32_t NZ_FRAC_COLS = 16;
static constexpr uint32_t NZ_BLOCK_ELEMS = Q_CHUNK_SIZE * NZ_FRAC_COLS;
static constexpr uint32_t VEC = 64;
static constexpr uint32_t REDUCE_WIDTH = K_TILE_COLS;
static constexpr uint32_t L1_CHUNK_ROWS = 16;
static constexpr uint32_t MERGED_SLICE_ROWS = 32;
static constexpr uint32_t BATCH = 8;

static constexpr uint32_t SCORE_UB_ELEMS = (Q_CHUNK_SIZE / 2) * K_TILE_COLS;
static constexpr uint32_t MAX_GROUPS = K_TILE_COLS / 8;

static constexpr uint32_t GF_PS_UB_ELEMS = Q_CHUNK_SIZE * MAX_GROUPS;
static constexpr uint32_t GF_PM_ROW_OFF = GF_PS_UB_ELEMS;
static constexpr uint32_t GF_CORR_ROW_OFF = GF_PM_ROW_OFF + Q_CHUNK_SIZE;
static constexpr uint32_t GF_CORR_BROC_OFF = GF_CORR_ROW_OFF + Q_CHUNK_SIZE;
static constexpr uint32_t GF_UNIT_GRP_OFF = GF_CORR_BROC_OFF + Q_CHUNK_SIZE * FLOAT_DATA_BLOCK_NUM;
static constexpr uint32_t GF_ACC_BASE = GF_UNIT_GRP_OFF + MAX_GROUPS + 64;

static constexpr uint32_t GF_OUT_ROW_OFF = SCORE_UB_ELEMS - 1024;
static constexpr uint32_t GF_OUT_F16_OFF = GF_OUT_ROW_OFF + 512;
static constexpr uint32_t GF_COL_SCALE_OFF = GF_OUT_F16_OFF + 256;
static constexpr uint32_t GF_ACC_CAP = GF_OUT_ROW_OFF - GF_ACC_BASE;

static constexpr uint32_t BF_RA_TMP_ELEMS = 1024;
static constexpr uint32_t BF_FP16_ROW_OFF = BF_RA_TMP_ELEMS;
static constexpr uint32_t BF_INV_COLS_OFF = BF_FP16_ROW_OFF + 512;
static constexpr uint32_t BF_UNIT_RES_OFF = BF_INV_COLS_OFF + 32;
static constexpr uint32_t BF_UNIT_RES_ELEMS = 16 * BATCH * 32;
static constexpr uint32_t BF_CORR_ALL_OFF = BF_UNIT_RES_OFF + BF_UNIT_RES_ELEMS;
static constexpr uint32_t BF_CORR_ALL_OFF_FIXED = BF_UNIT_RES_OFF + 1024;

static constexpr uint32_t FOLD_OUT8_OFF = 4096;

static constexpr uint32_t GLOBAL_STATE_UB_OFFSET = 186 * 1024;
static constexpr uint32_t FP16OUT_UB_OFFSET = 154 * 1024;
static constexpr uint32_t VMASK_TABLE_OFFSET = GLOBAL_STATE_UB_OFFSET + 2 * Q_CHUNK_SIZE * sizeof(float);
static constexpr uint32_t VMASK_NEGINF_OFFSET = VMASK_TABLE_OFFSET + 8 * 8 * sizeof(float);
static constexpr uint32_t PENDING_UB_OFFSET = VMASK_NEGINF_OFFSET + 8 * 8 * sizeof(float);
static constexpr uint32_t PENDING_WIN_MAX = 2;
static constexpr uint32_t PENDING_GRP_MAX = 256;

template <typename BSAT>
class BSAVecSmService {
public:
    using T = float;
    using IN_T = typename BSAT::inputT;
    using OUT_T = half;
    using PostRowSlice = typename BSAVecPostPoolService<BSAT>::PostRowSlice;

    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV;
    static constexpr uint32_t SUB_ROWS = 32;
    static constexpr event_t EV_CROSS = (event_t)6;
    static constexpr event_t EVENT_FIN = (event_t)7;

    struct Pass1Band {
        uint32_t sliceS, sliceE;
        uint32_t localStart;
        uint32_t chunkStart;
        uint32_t rows;
        uint32_t nSub;
    };

    __aicore__ inline BSAVecSmService(){};

    __aicore__ inline void InitParams(const BSAConstInfo& constInfo,
                                      const optiling::BSASelectBlockMaskTilingData* __restrict tilingData);
    __aicore__ inline void InitBuffers(TBuf<>* uBuf_);
    __aicore__ inline void InitGM(GlobalTensor<T>& scoreFp32Gm, GlobalTensor<OUT_T>& attnScoreFp16Gm,
                                  GlobalTensor<OUT_T>& attnScoreGm, GlobalTensor<T>& postStagingGm,
                                  GlobalTensor<T>& partSumGm, GlobalTensor<T>& partMaxGm);

    __aicore__ inline void SoftmaxFinalize(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t batchIdx,
                                           uint32_t headIdx, uint32_t validXBlocks, uint32_t validYBlocks);
    __aicore__ inline void OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart,
                                                       uint32_t kChunkSize, uint32_t validXBlocks,
                                                       uint32_t validYBlocks);
    __aicore__ inline void SoftmaxSecondPassAndCast(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t kChunkStart,
                                                    uint32_t kChunkSize, uint32_t batchIdx, uint32_t headIdx,
                                                    uint32_t validXBlocks, uint32_t validYBlocks);
    __aicore__ inline void NzReadSubBand(uint64_t tileBase, uint32_t subChunkStart, uint32_t subRows,
                                         uint32_t kChunkSize, uint32_t kColsAlign, uint32_t slot, event_t ev);

private:
    __aicore__ inline void SoftmaxFinalizeBatch(const PostRowSlice* slices, uint32_t sliceNum, uint32_t qChunkStart,
                                                uint32_t batchIdx, uint32_t headIdx, uint32_t validXBlocks,
                                                uint32_t validYBlocks, uint32_t t0, uint32_t k, uint32_t groupsValid,
                                                bool boundary);
    __aicore__ inline void GeneralFinalize(const PostRowSlice* slices, uint32_t sliceNum, uint32_t qChunkStart,
                                           uint32_t batchIdx, uint32_t headIdx, uint32_t validXBlocks,
                                           uint32_t validYBlocks);
    __aicore__ inline void NzToNdCopyIn(LocalTensor<T>& dstUb, uint64_t tileBase, uint32_t rowStart, uint32_t rowCount,
                                        uint32_t validCols, uint32_t ubRowWidth);
    __aicore__ inline void RowMaxCascade(LocalTensor<T>& dstRow, LocalTensor<T>& srcTile, uint32_t rows,
                                         uint32_t rowStride, LocalTensor<uint8_t>& tmp);
    __aicore__ inline void RowSumCascade(LocalTensor<T>& dstRow, LocalTensor<T>& srcTile, uint32_t rows,
                                         uint32_t rowStride, LocalTensor<uint8_t>& tmp);
    __aicore__ inline void FillPass1Ctx(uint32_t qChunkStart, uint32_t qChunkSize, uint32_t validXBlocks);
    __aicore__ inline uint32_t MergeBands(uint32_t kChunkSizeAlign, Pass1Band* out);
    __aicore__ inline void PrefetchBandHead(uint64_t tileBase, const Pass1Band& bd, uint32_t kChunkSize,
                                            uint32_t foldCols);
    __aicore__ inline void MaskSubBandTailPad(LocalTensor<T> subUb, uint32_t subRows, uint32_t kChunkSize,
                                              uint32_t foldCols);
    __aicore__ inline void SoftmaxOnlineSlice(LocalTensor<T>& tile, const PostRowSlice& sl, uint32_t kChunkStart,
                                              uint32_t kChunkSize, uint32_t foldCols);
    __aicore__ inline void FoldSliceEmit(LocalTensor<T>& tile, const PostRowSlice& sl, uint64_t psSlotBase,
                                         uint32_t foldW, uint32_t foldCols);
    __aicore__ inline void FinalizeBand(const Pass1Band& bd, uint32_t bi, uint32_t bandNum, uint32_t sliceNum,
                                        uint32_t tileIdx, uint32_t kChunkStart, uint32_t kChunkSize,
                                        uint32_t validYBlocks, uint64_t pmSlotBase);
    __aicore__ inline void FinalizeBatchReadParts(const PostRowSlice* slices, uint32_t sliceNum, uint32_t k,
                                                  uint32_t t0);
    __aicore__ inline void FinalizeBatchCorrTile(uint32_t b, uint32_t ownRows);
    __aicore__ inline void FinalizeBatchFoldUnits(const PostRowSlice* slices, uint32_t sliceNum, uint32_t b, uint32_t k,
                                                  uint32_t qChunkStartFx);
    __aicore__ inline void FinalizeBatchScaleCast(const PostRowSlice* slices, uint32_t sliceNum, uint32_t k,
                                                  uint32_t nUnits, uint32_t qChunkStart, uint32_t validXBlocks,
                                                  bool invRowsUniform);
    __aicore__ inline void FinalizeBatchWriteOut(const PostRowSlice* slices, uint32_t sliceNum, uint32_t qChunkStart,
                                                 uint32_t k, uint32_t t0, uint32_t groupsValid, uint32_t validXBlocksBx,
                                                 uint32_t nUnitsFx, uint64_t pooledHeadOffset, uint32_t pooledRowStride,
                                                 GlobalTensor<OUT_T>& attnScoreGmTensor);
    __aicore__ inline void GeneralWinReadTile(const PostRowSlice* slices, uint32_t sliceNum, uint32_t t, uint32_t G);
    __aicore__ inline void GeneralWinCorr(uint32_t G, uint32_t ownRows);
    __aicore__ inline void GeneralWinFoldAcc(const PostRowSlice* slices, uint32_t sliceNum, uint32_t G, uint32_t tG,
                                             uint32_t gLo, uint32_t gHi, uint32_t unitRows, uint32_t groupsWinA,
                                             uint32_t qChunkStartFx, uint32_t nUnitsAccFold);
    __aicore__ inline void GeneralWinRowMean(const PostRowSlice* slices, uint32_t sliceNum, uint32_t qChunkStart,
                                             uint32_t validXBlocks, uint32_t nUnits, uint32_t winGroups,
                                             uint32_t groupsWinA);
    __aicore__ inline void GeneralWinAssembleWrite(const PostRowSlice* slices, uint32_t sliceNum, uint32_t qChunkStart,
                                                   uint32_t validYBlocks, uint32_t pbY, uint32_t P, uint32_t c0,
                                                   uint32_t c1, uint32_t gLo, uint32_t gHi, uint32_t winGroups,
                                                   uint32_t groupsWinA, uint32_t colsPerWinFx, uint32_t validXBlocksFx,
                                                   uint64_t pooledHeadOffset, uint32_t pooledRowStride,
                                                   GlobalTensor<OUT_T>& attnScoreGmTensor);
    __aicore__ inline void GeneralWinGeometry(uint32_t P, uint32_t nUnits, uint32_t& colsPerWin, uint32_t& groupsWinA);
    __aicore__ inline void Pass2LoadBand(uint64_t tileBase, uint32_t bandChunkStart, uint32_t bandRows,
                                         uint32_t kChunkSize, uint32_t kChunkSizeAlign);
    __aicore__ inline void Pass2NormalizeSlice(LocalTensor<T> tile, const PostRowSlice& sl, uint32_t kChunkStart,
                                               uint32_t kChunkSize, uint32_t kChunkSizeAlign);
    __aicore__ inline void Pass2BuildOutView(const PostRowSlice& sl, uint32_t qChunkStart, uint32_t kChunkStart,
                                             uint32_t batchIdx, uint32_t headIdx, uint32_t validYBlocks,
                                             GlobalTensor<OUT_T>& attnScoreChunkGm, uint32_t& p2RowStride);
    __aicore__ inline void Pass2CastWriteSlice(LocalTensor<T> tile, const PostRowSlice& sl, uint32_t qChunkStart,
                                               uint32_t kChunkStart, uint32_t kChunkSize, uint32_t kChunkSizeAlign,
                                               uint32_t validXBlocks, uint32_t validYBlocks, uint32_t p2RowStride,
                                               GlobalTensor<OUT_T>& attnScoreChunkGm);
    __aicore__ inline void Pass2WriteLastSlice(const PostRowSlice& sl, uint32_t kChunkStart, uint32_t kChunkSize,
                                               uint32_t kChunkSizeAlign, uint32_t validYBlocks, uint32_t p2RowStride,
                                               uint32_t rowBytes, GlobalTensor<OUT_T>& attnScoreChunkGm);

public:
    __aicore__ inline void VToSSync()
    {
        SetFlag<HardEvent::V_S>(static_cast<event_t>(2));
        WaitFlag<HardEvent::V_S>(static_cast<event_t>(2));
    }
    __aicore__ inline void SToVSync()
    {
        SetFlag<HardEvent::S_V>(static_cast<event_t>(6));
        WaitFlag<HardEvent::S_V>(static_cast<event_t>(6));
    }

    __aicore__ inline void ResetPending()
    {
        AscendC::Duplicate(pendingUb_, float(0.0f), PENDING_WIN_MAX * PENDING_GRP_MAX);
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    BSAConstInfo constInfo_;
    const optiling::BSASelectBlockMaskTilingData* __restrict tilingData_;
    BSAVecPostPoolService<BSAT> postPoolOP_;
    uint32_t vecSubBlockIdx_ = 0;

    GlobalTensor<OUT_T> attnScoreFp16GmTensor_;
    GlobalTensor<T> scoreFp32GmTensor_;
    GlobalTensor<T> partSumGmTensor_;
    GlobalTensor<T> partMaxGmTensor_;
    TBuf<>* uBufPtr_ = nullptr;

    LocalTensor<uint8_t> reduceSharedTemp_;
    LocalTensor<T> tmpMax_;
    LocalTensor<T> tmpSum_;
    LocalTensor<T> globalMax_;
    LocalTensor<T> globalSum_;
    LocalTensor<T> globalMaxBroc_;
    LocalTensor<T> globalSumBroc_;
    LocalTensor<T> diffMax_;
    LocalTensor<T> scoreUb_;
    LocalTensor<OUT_T> scoreFp16Ub_;
    LocalTensor<OUT_T> scoreFp16OutUb_;
    LocalTensor<T> vmaskTable_;
    LocalTensor<T> vmaskNegInf_;
    LocalTensor<T> pendingUb_;

    event_t eventSft_ = (event_t)3;
    event_t evSlot_[2] = {(event_t)4, (event_t)5};

    PostRowSlice cachedSlices_[BSA_SM_SLICES_MAX];
    uint32_t cachedSliceNum_ = 0;
    uint32_t cachedChunkKey_ = 0xFFFFFFFF;
    uint32_t cachedValidXBlocks_ = 0xFFFFFFFF;
    Pass1Band ctxBands_[BSA_SM_SLICES_MAX];
    uint32_t ctxBandNum_ = 0;
    uint64_t ctxChunkGlobalTile_ = 0;
    uint64_t ctxTileElems_ = 0;
    uint64_t ctxAicRingBase_ = 0;
    uint64_t ctxAivSlotStride_ = 0;
};

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitParams(
    const BSAConstInfo& constInfo, const optiling::BSASelectBlockMaskTilingData* __restrict tilingData)
{
    this->constInfo_ = constInfo;
    this->tilingData_ = tilingData;
    vecSubBlockIdx_ = constInfo_.subBlockIdx;
    postPoolOP_.InitParams(constInfo, tilingData);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitBuffers(TBuf<>* uBuf_)
{
    uint32_t ubOffset = 0;
    uint32_t chunkSize = Q_CHUNK_SIZE;

    globalMax_ = uBuf_->GetWithOffset<T>(chunkSize, GLOBAL_STATE_UB_OFFSET);
    globalSum_ = uBuf_->GetWithOffset<T>(chunkSize, GLOBAL_STATE_UB_OFFSET + chunkSize * sizeof(T));

    globalMaxBroc_ = uBuf_->GetWithOffset<T>(chunkSize * FLOAT_DATA_BLOCK_NUM, ubOffset);
    ubOffset += chunkSize * FLOAT_DATA_BLOCK_NUM * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    globalSumBroc_ = uBuf_->GetWithOffset<T>(chunkSize * FLOAT_DATA_BLOCK_NUM, ubOffset);
    ubOffset += chunkSize * FLOAT_DATA_BLOCK_NUM * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    tmpMax_ = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    diffMax_ = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    tmpSum_ = uBuf_->GetWithOffset<T>(chunkSize, ubOffset);
    ubOffset += chunkSize * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uint32_t scoreElements = (Q_CHUNK_SIZE / 2) * K_CHUNK_SIZE * CV_EXEC_RATIO;
    scoreUb_ = uBuf_->GetWithOffset<T>(scoreElements, ubOffset);
    scoreFp16Ub_ = uBuf_->GetWithOffset<OUT_T>(scoreElements, ubOffset);

    ubOffset += scoreElements * sizeof(T);
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uint32_t sharedReduceEles = BSAConstInfo::BUFFER_SIZE_BYTE_24K / sizeof(uint8_t);
    reduceSharedTemp_ = uBuf_->GetWithOffset<uint8_t>(sharedReduceEles, ubOffset);
    ubOffset += sharedReduceEles * sizeof(uint8_t);

    scoreFp16OutUb_ = uBuf_->GetWithOffset<OUT_T>(scoreElements / 2, FP16OUT_UB_OFFSET);

    pendingUb_ = uBuf_->GetWithOffset<T>(PENDING_WIN_MAX * PENDING_GRP_MAX, PENDING_UB_OFFSET);
    AscendC::Duplicate(pendingUb_, float(0.0f), PENDING_WIN_MAX * PENDING_GRP_MAX);

    vmaskTable_ = uBuf_->GetWithOffset<T>(8 * 8, VMASK_TABLE_OFFSET);
    for (uint32_t rem = 0; rem < 8u; rem++) {
        for (uint32_t c = 0; c < 8u; c++) {
            vmaskTable_.SetValue(rem * 8u + c, c < rem ? 1.0f : 0.0f);
        }
    }
    vmaskNegInf_ = uBuf_->GetWithOffset<T>(8 * 8, VMASK_NEGINF_OFFSET);
    for (uint32_t rem = 0; rem < 8u; rem++) {
        for (uint32_t c = 0; c < 8u; c++) {
            vmaskNegInf_.SetValue(rem * 8u + c, c < rem ? 0.0f : float(SOFTMAX_NEG_INF));
        }
    }
    ubOffset = BSAAlignTo(ubOffset, static_cast<uint32_t>(VEC_ALIGN_SIZE));

    uBufPtr_ = uBuf_;
    if (constInfo_.usePostBlockShape) {
        postPoolOP_.InitBuffers(uBuf_, ubOffset, reduceSharedTemp_);
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::InitGM(GlobalTensor<T>& ScoreFp32Gm, GlobalTensor<OUT_T>& attnScoreFp16Gm,
                                                     GlobalTensor<OUT_T>& attnScoreGm, GlobalTensor<T>& postStagingGm,
                                                     GlobalTensor<T>& partSumGm, GlobalTensor<T>& partMaxGm)
{
    this->attnScoreFp16GmTensor_ = attnScoreFp16Gm;
    this->scoreFp32GmTensor_ = ScoreFp32Gm;
    this->partSumGmTensor_ = partSumGm;
    this->partMaxGmTensor_ = partMaxGm;
    if (constInfo_.usePostBlockShape) {
        postPoolOP_.InitGM(attnScoreGm, postStagingGm);
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::NzToNdCopyIn(LocalTensor<T>& dstUb, uint64_t tileBase, uint32_t rowStart,
                                                           uint32_t rowCount, uint32_t validCols, uint32_t ubRowWidth)
{
    uint32_t blockNum = (validCols + NZ_FRAC_COLS - 1) / NZ_FRAC_COLS;
    for (uint32_t b = 0; b < blockNum; b++) {
        DataCopyPad(dstUb[b * NZ_FRAC_COLS],
                    scoreFp32GmTensor_[tileBase + static_cast<uint64_t>(b) * NZ_BLOCK_ELEMS +
                                       static_cast<uint64_t>(rowStart) * NZ_FRAC_COLS],
                    {static_cast<uint16_t>(rowCount), static_cast<uint32_t>(NZ_FRAC_COLS * sizeof(float)), 0,
                     static_cast<uint32_t>((ubRowWidth - NZ_FRAC_COLS) * sizeof(float) / 32), 0},
                    {false, 0, 0, 0});
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::RowMaxCascade(LocalTensor<T>& dstRow, LocalTensor<T>& srcTile,
                                                            uint32_t rows, uint32_t rowStride,
                                                            LocalTensor<uint8_t>& tmpBytes)
{
    LocalTensor<T> tmp = tmpBytes.ReinterpretCast<T>();
    uint32_t l1Elems = rows * (REDUCE_WIDTH / FLOAT_DATA_BLOCK_NUM);
    for (uint32_t r = 0; r < rows; r += L1_CHUNK_ROWS) {
        uint32_t cnt = BSAMin(L1_CHUNK_ROWS, rows - r);
        AscendC::BlockReduceMax<float, false>(tmp[r * (REDUCE_WIDTH / FLOAT_DATA_BLOCK_NUM)], srcTile[r * rowStride],
                                              cnt * rowStride / VEC, 0, 1, 1, 8);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceMax<float, false>(tmp[l1Elems], tmp, l1Elems / VEC, 0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    if (rows * FLOAT_DATA_BLOCK_NUM >= VEC) {
        uint32_t rep3 = (rows * FLOAT_DATA_BLOCK_NUM + VEC - 1) / VEC;
        AscendC::BlockReduceMax<float, false>(dstRow, tmp[l1Elems], rep3, 0, 1, 1, 8);
    } else {
        uint32_t l3Shape[] = {rows, FLOAT_DATA_BLOCK_NUM};
        AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, false>(dstRow, tmp[l1Elems], tmpBytes, l3Shape, false);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::RowSumCascade(LocalTensor<T>& dstRow, LocalTensor<T>& srcTile,
                                                            uint32_t rows, uint32_t rowStride,
                                                            LocalTensor<uint8_t>& tmpBytes)
{
    LocalTensor<T> tmp = tmpBytes.ReinterpretCast<T>();
    uint32_t l1Elems = rows * (REDUCE_WIDTH / FLOAT_DATA_BLOCK_NUM);
    for (uint32_t r = 0; r < rows; r += L1_CHUNK_ROWS) {
        uint32_t cnt = BSAMin(L1_CHUNK_ROWS, rows - r);
        AscendC::BlockReduceSum<float, false>(tmp[r * (REDUCE_WIDTH / FLOAT_DATA_BLOCK_NUM)], srcTile[r * rowStride],
                                              cnt * rowStride / VEC, 0, 1, 1, 8);
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BlockReduceSum<float, false>(tmp[l1Elems], tmp, l1Elems / VEC, 0, 1, 1, 8);
    AscendC::PipeBarrier<PIPE_V>();
    if (rows * FLOAT_DATA_BLOCK_NUM >= VEC) {
        uint32_t rep3 = (rows * FLOAT_DATA_BLOCK_NUM + VEC - 1) / VEC;
        AscendC::BlockReduceSum<float, false>(dstRow, tmp[l1Elems], rep3, 0, 1, 1, 8);
    } else {
        uint32_t l3Shape[] = {rows, FLOAT_DATA_BLOCK_NUM};
        AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, false>(dstRow, tmp[l1Elems], tmpBytes, l3Shape, false);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename BSAT>
__aicore__ inline uint32_t BSAVecSmService<BSAT>::MergeBands(uint32_t kChunkSizeAlign, Pass1Band* out)
{
    uint32_t bandCap = (Q_CHUNK_SIZE / 2) * K_CHUNK_SIZE * CV_EXEC_RATIO / kChunkSizeAlign;
    uint32_t num = 0;
    uint32_t s = 0;
    while (s < cachedSliceNum_) {
        uint32_t e = s + 1;
        while (e < cachedSliceNum_) {
            bool contiguous =
                (cachedSlices_[e].chunkRowStart ==
                 cachedSlices_[e - 1].chunkRowStart + cachedSlices_[e - 1].rowCount) &&
                (cachedSlices_[e].localRowStart == cachedSlices_[e - 1].localRowStart + cachedSlices_[e - 1].rowCount);
            uint32_t mergedRows = cachedSlices_[e - 1].localRowStart + cachedSlices_[e - 1].rowCount -
                                  cachedSlices_[s].localRowStart + cachedSlices_[e].rowCount;
            if (!contiguous || mergedRows > bandCap) {
                break;
            }
            e++;
        }
        uint32_t bandLocalStart = cachedSlices_[s].localRowStart;
        uint32_t bandChunkStart = cachedSlices_[s].chunkRowStart;
        uint32_t bandRows = cachedSlices_[e - 1].localRowStart + cachedSlices_[e - 1].rowCount - bandLocalStart;
        uint32_t nSub = (bandRows + SUB_ROWS - 1) / SUB_ROWS;
        out[num].sliceS = s;
        out[num].sliceE = e;
        out[num].localStart = bandLocalStart;
        out[num].chunkStart = bandChunkStart;
        out[num].rows = bandRows;
        out[num].nSub = nSub;
        num++;
        s = e;
    }
    return num;
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::FillPass1Ctx(uint32_t qChunkStart, uint32_t qChunkSize,
                                                           uint32_t validXBlocks)
{
    PostRowSlice raw[BSA_SM_SLICES_MAX];
    uint32_t rawNum = postPoolOP_.GetChunkRowSlices(qChunkStart, qChunkSize, validXBlocks, raw);
    cachedSliceNum_ = 0;
    for (uint32_t i = 0; i < rawNum;) {
        uint32_t rows = raw[i].rowCount;
        uint32_t j = i + 1;
        while (j < rawNum && rows < MERGED_SLICE_ROWS) {
            const PostRowSlice& a = raw[j - 1];
            const PostRowSlice& b = raw[j];
            if (b.chunkRowStart != a.chunkRowStart + a.rowCount || b.localRowStart != a.localRowStart + a.rowCount ||
                rows + b.rowCount > MERGED_SLICE_ROWS) {
                break;
            }
            rows += b.rowCount;
            j++;
        }
        cachedSlices_[cachedSliceNum_] = {raw[i].chunkRowStart, raw[i].localRowStart, rows};
        cachedSliceNum_++;
        i = j;
    }
    cachedChunkKey_ = qChunkStart;
    cachedValidXBlocks_ = validXBlocks;
    uint32_t ktiles = (constInfo_.yBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
    ctxChunkGlobalTile_ = (static_cast<uint64_t>(qChunkStart) / Q_CHUNK_SIZE) * ktiles;
    ctxTileElems_ = static_cast<uint64_t>(Q_CHUNK_SIZE) * K_TILE_COLS;
    uint64_t ringSlots =
        constInfo_.usePostBlockShape ? BSA_SCRATCH_RING_TILES : (constInfo_.yBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
    ctxAicRingBase_ = static_cast<uint64_t>(constInfo_.aicIdx) * ringSlots * ctxTileElems_;
    ctxAivSlotStride_ = static_cast<uint64_t>(constInfo_.aivIdx) * ktiles;
    ctxBandNum_ = MergeBands(K_TILE_COLS, ctxBands_);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::NzReadSubBand(uint64_t tileBase, uint32_t subChunkStart, uint32_t subRows,
                                                            uint32_t kChunkSize, uint32_t kColsAlign, uint32_t slot,
                                                            event_t ev)
{
    SetFlag<HardEvent::V_MTE2>(ev);
    WaitFlag<HardEvent::V_MTE2>(ev);
    LocalTensor<T> slotUb = scoreUb_[slot * SUB_ROWS * kColsAlign];
    NzToNdCopyIn(slotUb, tileBase, subChunkStart, subRows, kChunkSize, kColsAlign);
    SetFlag<HardEvent::MTE2_V>(ev);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::OnlineSoftmaxFirstPassChunk(uint32_t qChunkStart, uint32_t qChunkSize,
                                                                          uint32_t kChunkStart, uint32_t kChunkSize,
                                                                          uint32_t validXBlocks, uint32_t validYBlocks)
{
    if (cachedChunkKey_ != qChunkStart || cachedValidXBlocks_ != validXBlocks) {
        FillPass1Ctx(qChunkStart, qChunkSize, validXBlocks);
    }
    const PostRowSlice* slices = cachedSlices_;
    uint32_t sliceNum = cachedSliceNum_;
    if (sliceNum == 0) {
        return;
    }

    bool fullTile = (kChunkSize == K_TILE_COLS);
    uint32_t foldW = (constInfo_.usePostBlockShape && constInfo_.postBlockShapeY % 16 != 0) ? 8u : 16u;
    uint32_t foldG = K_TILE_COLS / foldW;
    uint32_t foldCols;
    if (constInfo_.usePostBlockShape) {
        foldCols = K_TILE_COLS;
    } else {
        uint32_t foldAlign = (foldW == 8) ? 64u : 128u;
        foldCols = (kChunkSize + foldAlign - 1) / foldAlign * foldAlign;
    }

    SetFlag<HardEvent::MTE3_MTE2>(eventSft_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft_);

    uint32_t tileIdx = kChunkStart / K_TILE_COLS;
    uint64_t tileBase =
        constInfo_.usePostBlockShape ?
            ctxAicRingBase_ + ((ctxChunkGlobalTile_ + tileIdx) % BSA_SCRATCH_RING_TILES) * ctxTileElems_ :
            ctxAicRingBase_ + static_cast<uint64_t>(tileIdx) * ctxTileElems_;
    uint64_t psSlotBase = (ctxAivSlotStride_ + tileIdx) * Q_CHUNK_SIZE * foldG;
    uint64_t pmSlotBase = (ctxAivSlotStride_ + tileIdx) * Q_CHUNK_SIZE;

    const Pass1Band* bands;
    Pass1Band tailBands[BSA_SM_SLICES_MAX];
    uint32_t bandNum;
    if (fullTile) {
        bands = ctxBands_;
        bandNum = ctxBandNum_;
    } else {
        bandNum = MergeBands(foldCols, tailBands);
        bands = tailBands;
    }
    for (uint32_t bi = 0; bi < bandNum; bi++) {
        const Pass1Band& bd = bands[bi];

        PrefetchBandHead(tileBase, bd, kChunkSize, foldCols);

        uint32_t sliceIdx = bd.sliceS;
        uint32_t sliceRemOff = 0;
        for (uint32_t j = 0; j < bd.nSub; j++) {
            uint32_t subLocalStart = bd.localStart + j * SUB_ROWS;
            uint32_t subChunkStart = bd.chunkStart + j * SUB_ROWS;
            uint32_t subRows = BSAMin(SUB_ROWS, bd.rows - j * SUB_ROWS);
            uint32_t slot = j & 1;
            LocalTensor<T> subUb = scoreUb_[slot * SUB_ROWS * foldCols];

            WaitFlag<HardEvent::MTE2_V>(evSlot_[slot]);
            MaskSubBandTailPad(subUb, subRows, kChunkSize, foldCols);

            while (sliceIdx < bd.sliceE) {
                const PostRowSlice& sl = slices[sliceIdx];
                uint32_t absStart = sl.localRowStart + sliceRemOff;
                if (absStart >= subLocalStart + subRows) {
                    break;
                }
                uint32_t rowsHere = BSAMin(sl.rowCount - sliceRemOff, subLocalStart + subRows - absStart);
                PostRowSlice partial;
                partial.localRowStart = absStart;
                partial.chunkRowStart = sl.chunkRowStart + sliceRemOff;
                partial.rowCount = rowsHere;
                LocalTensor<T> tile = subUb[(absStart - subLocalStart) * foldCols];
                SoftmaxOnlineSlice(tile, partial, kChunkStart, kChunkSize, foldCols);
                FoldSliceEmit(tile, partial, psSlotBase, foldW, foldCols);
                sliceRemOff += rowsHere;
                if (sliceRemOff >= sl.rowCount) {
                    sliceIdx++;
                    sliceRemOff = 0;
                }
            }
            if (j + 2 < bd.nSub) {
                NzReadSubBand(tileBase, subChunkStart + 2 * SUB_ROWS, BSAMin(SUB_ROWS, bd.rows - (j + 2) * SUB_ROWS),
                              kChunkSize, foldCols, slot, evSlot_[slot]);
            }
        }

        FinalizeBand(bd, bi, bandNum, sliceNum, tileIdx, kChunkStart, kChunkSize, validYBlocks, pmSlotBase);
    }

    if (kChunkStart + kChunkSize >= validYBlocks) {
        uint32_t rowsTotal = slices[sliceNum - 1].localRowStart + slices[sliceNum - 1].rowCount;
        uint32_t brocRepeat = BSACeilDiv(rowsTotal, static_cast<uint32_t>(BROC_BASE_NUM));
        Brcb(globalSumBroc_, globalSum_, brocRepeat, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::PrefetchBandHead(uint64_t tileBase, const Pass1Band& bd,
                                                               uint32_t kChunkSize, uint32_t foldCols)
{
    NzReadSubBand(tileBase, bd.chunkStart, BSAMin(SUB_ROWS, bd.rows), kChunkSize, foldCols, 0, evSlot_[0]);
    if (bd.nSub > 1) {
        NzReadSubBand(tileBase, bd.chunkStart + SUB_ROWS, BSAMin(SUB_ROWS, bd.rows - SUB_ROWS), kChunkSize, foldCols, 1,
                      evSlot_[1]);
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::MaskSubBandTailPad(LocalTensor<T> subUb, uint32_t subRows,
                                                                 uint32_t kChunkSize, uint32_t foldCols)
{
    if (kChunkSize >= foldCols) {
        return;
    }
    uint32_t padAlign = (kChunkSize + 7u) / 8u * 8u;
    uint32_t rem = kChunkSize % 8u;
    uint32_t maskBase = kChunkSize - rem;
    if (rem != 0 && maskBase < foldCols) {
        for (uint32_t r = 0; r < subRows; r++) {
            AscendC::Add(subUb[static_cast<uint32_t>(r) * foldCols + maskBase],
                         subUb[static_cast<uint32_t>(r) * foldCols + maskBase], vmaskNegInf_[rem * 8u], 8);
        }
    }
    if (padAlign < foldCols) {
        for (uint32_t r = 0; r < subRows; r++) {
            AscendC::Duplicate(subUb[static_cast<uint32_t>(r) * foldCols + padAlign], float(SOFTMAX_NEG_INF),
                               foldCols - padAlign);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::SoftmaxOnlineSlice(LocalTensor<T>& tile, const PostRowSlice& sl,
                                                                 uint32_t kChunkStart, uint32_t kChunkSize,
                                                                 uint32_t foldCols)
{
    Muls(tile, tile, (float)(constInfo_.scaleValue), sl.rowCount * foldCols);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t maxShape[] = {sl.rowCount, foldCols};
    bool isSrcInnerPad = true;
    if (foldCols == K_TILE_COLS && constInfo_.usePostBlockShape) {
        LocalTensor<T> dstMax = tmpMax_[sl.localRowStart];
        RowMaxCascade(dstMax, tile, sl.rowCount, foldCols, reduceSharedTemp_);
    } else {
        AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, false>(tmpMax_[sl.localRowStart], tile, reduceSharedTemp_,
                                                                   maxShape, isSrcInnerPad);
        AscendC::PipeBarrier<PIPE_V>();
    }

    if (kChunkStart != 0) {
        AscendC::Max(tmpMax_[sl.localRowStart], tmpMax_[sl.localRowStart], globalMax_[sl.localRowStart], sl.rowCount);
    } else {
        AscendC::Duplicate(globalMax_[sl.localRowStart], float(SOFTMAX_NEG_INF), sl.rowCount);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Duplicate(globalSum_[sl.localRowStart], float(0.0f), sl.rowCount);
    }
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::Sub(diffMax_[sl.localRowStart], globalMax_[sl.localRowStart], tmpMax_[sl.localRowStart], sl.rowCount);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(diffMax_[sl.localRowStart], diffMax_[sl.localRowStart], sl.rowCount);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mul(globalSum_[sl.localRowStart], globalSum_[sl.localRowStart], diffMax_[sl.localRowStart], sl.rowCount);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t brocRepeat = BSACeilDiv(sl.rowCount, static_cast<uint32_t>(BROC_BASE_NUM));
    Brcb(globalMaxBroc_[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], tmpMax_[sl.localRowStart], brocRepeat, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    BSASubRowBroadcast(tile, tile, globalMaxBroc_[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount, foldCols);
    AscendC::Exp(tile, tile, sl.rowCount * foldCols);
    AscendC::PipeBarrier<PIPE_V>();

    if (kChunkSize < foldCols) {
        uint32_t rem = kChunkSize % 8u;
        uint32_t maskBase = kChunkSize - rem;
        if (maskBase < foldCols) {
            LocalTensor<T> mskMix = vmaskTable_[static_cast<uint32_t>(rem) * 8u];
            for (uint32_t r = 0; r < sl.rowCount; r++) {
                AscendC::Mul(tile[static_cast<uint32_t>(r) * foldCols + maskBase],
                             tile[static_cast<uint32_t>(r) * foldCols + maskBase], mskMix, 8);
            }
            for (uint32_t off = maskBase + 8u; off < foldCols; off += 8u) {
                for (uint32_t r = 0; r < sl.rowCount; r++) {
                    AscendC::Duplicate(tile[static_cast<uint32_t>(r) * foldCols + off], 0.0f, 8);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    if (foldCols == K_TILE_COLS && constInfo_.usePostBlockShape) {
        LocalTensor<T> dstSum = tmpSum_[sl.localRowStart];
        RowSumCascade(dstSum, tile, sl.rowCount, foldCols, reduceSharedTemp_);
    } else {
        uint32_t sumShape[] = {sl.rowCount, foldCols};
        if (foldCols <= FLOAT_REPEAT_NUM) {
            AscendC::WholeReduceSum<T, true>(tmpSum_[sl.localRowStart], tile, static_cast<int32_t>(foldCols),
                                             static_cast<int32_t>(sl.rowCount), 1, 1,
                                             static_cast<int32_t>(foldCols / FLOAT_DATA_BLOCK_NUM));
        } else {
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::AR, false>(tmpSum_[sl.localRowStart], tile,
                                                                       reduceSharedTemp_, sumShape, isSrcInnerPad);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::Add(globalSum_[sl.localRowStart], tmpSum_[sl.localRowStart], globalSum_[sl.localRowStart], sl.rowCount);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(globalMax_[sl.localRowStart], tmpMax_[sl.localRowStart], globalMax_[sl.localRowStart], sl.rowCount);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::FoldSliceEmit(LocalTensor<T>& tile, const PostRowSlice& sl,
                                                            uint64_t psSlotBase, uint32_t foldW, uint32_t foldCols)
{
    if (!constInfo_.usePostBlockShape) {
        return;
    }
    if (!(foldW == 16 && constInfo_.usePostBlockShape)) {
        LocalTensor<T> foldOut8 = reduceSharedTemp_.ReinterpretCast<T>()[FOLD_OUT8_OFF];
        uint32_t grpW = foldCols / 8;
        for (uint32_t r0 = 0; r0 < sl.rowCount; r0 += SUB_ROWS) {
            uint32_t rc = BSAMin(SUB_ROWS, sl.rowCount - r0);
            for (uint32_t r = 0; r < rc; r += 16) {
                uint32_t cnt = BSAMin(uint32_t(16), rc - r);
                AscendC::BlockReduceSum<float, false>(foldOut8[r * grpW], tile[(r0 + r) * foldCols],
                                                      cnt * foldCols / 64, 0, 1, 1, 8);
                AscendC::PipeBarrier<PIPE_V>();
            }
            SetFlag<HardEvent::V_MTE3>(eventSft_);
            WaitFlag<HardEvent::V_MTE3>(eventSft_);
            DataCopyPad(partSumGmTensor_[psSlotBase + (sl.chunkRowStart + r0) * 64], foldOut8[0],
                        {static_cast<uint16_t>(rc), static_cast<uint32_t>(grpW * sizeof(float)), 0,
                         static_cast<uint32_t>((64 - grpW) * sizeof(float)), 0});
            SetFlag<HardEvent::MTE3_V>(eventSft_);
            WaitFlag<HardEvent::MTE3_V>(eventSft_);
        }
        return;
    }

    LocalTensor<T> groupTmp = reduceSharedTemp_.ReinterpretCast<T>();
    LocalTensor<T> foldOut = reduceSharedTemp_.ReinterpretCast<T>()[FOLD_OUT8_OFF];
    AscendC::Duplicate(foldOut[sl.localRowStart * 32], float(0.0f), 32 * sl.rowCount);
    AscendC::PipeBarrier<PIPE_V>();
    if (foldCols == K_TILE_COLS) {
        AscendC::ReducePairElem<AscendC::ReduceType::SUM, float, float, true>(
            foldOut[sl.localRowStart * 32], groupTmp[0], static_cast<int32_t>(sl.rowCount * 64),
            static_cast<int32_t>(sl.rowCount), 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(eventSft_);
        WaitFlag<HardEvent::V_MTE3>(eventSft_);
        uint32_t sliceElems = sl.rowCount * 32;
        for (uint32_t off = 0; off < sliceElems; off += 1024) {
            uint32_t n = BSAMin(uint32_t(1024), sliceElems - off);
            DataCopyPad(partSumGmTensor_[psSlotBase + sl.chunkRowStart * 32 + off],
                        foldOut[sl.localRowStart * 32 + off], {1, static_cast<uint16_t>(n * sizeof(float)), 0, 0});
        }
        return;
    }

    uint32_t g16 = foldCols / 16;
    uint32_t quadCalls = g16 / 8;
    LocalTensor<T> chunks = groupTmp[0];
    LocalTensor<T> compact = groupTmp[4096];
    AscendC::BinaryRepeatParams bpPair;
    bpPair.dstBlkStride = 1;
    bpPair.src0BlkStride = 2;
    bpPair.src1BlkStride = 2;
    bpPair.dstRepStride = foldCols / 16;
    bpPair.src0RepStride = foldCols / 8;
    bpPair.src1RepStride = foldCols / 8;
    for (uint32_t rb = 0; rb < sl.rowCount; rb += 16) {
        uint32_t rc = BSAMin(uint32_t(16), sl.rowCount - rb);
        for (uint32_t qb = 0; qb < quadCalls; qb++) {
            AscendC::Add<T, false>(chunks[qb * 64], tile[rb * foldCols + qb * 128], tile[rb * foldCols + qb * 128 + 8],
                                   static_cast<uint64_t>(0), static_cast<uint8_t>(rc), bpPair);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::BlockReduceSum<float, false>(compact[0], chunks[0], rc * g16 / 8, 0, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(eventSft_);
        WaitFlag<HardEvent::V_MTE3>(eventSft_);
        DataCopyPad(partSumGmTensor_[psSlotBase + (sl.chunkRowStart + rb) * 32], compact[0],
                    {static_cast<uint16_t>(rc), static_cast<uint32_t>(g16 * sizeof(float)), 0,
                     static_cast<uint32_t>((32 - g16) * sizeof(float)), 0});
        SetFlag<HardEvent::MTE3_V>(eventSft_);
        WaitFlag<HardEvent::MTE3_V>(eventSft_);
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::FinalizeBand(const Pass1Band& bd, uint32_t bi, uint32_t bandNum,
                                                           uint32_t sliceNum, uint32_t tileIdx, uint32_t kChunkStart,
                                                           uint32_t kChunkSize, uint32_t validYBlocks,
                                                           uint64_t pmSlotBase)
{
    SetFlag<HardEvent::V_MTE3>(eventSft_);
    WaitFlag<HardEvent::V_MTE3>(eventSft_);
    if (constInfo_.usePostBlockShape) {
        DataCopyPad(partMaxGmTensor_[pmSlotBase + bd.chunkStart], tmpMax_[bd.localStart],
                    {1, static_cast<uint16_t>(bd.rows * sizeof(float)), 0, 0});
    }
    SetFlag<HardEvent::MTE3_V>(EV_CROSS);
    WaitFlag<HardEvent::MTE3_V>(EV_CROSS);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::FinalizeBatchReadParts(const PostRowSlice* slices, uint32_t sliceNum,
                                                                     uint32_t k, uint32_t t0)
{
    uint32_t ktiles = (constInfo_.yBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
    LocalTensor<T> rs = postPoolOP_.GetReduceTemp();
    LocalTensor<T> pmT = rs[BF_CORR_ALL_OFF_FIXED + 2 * Q_CHUNK_SIZE];
    LocalTensor<T> psUb = scoreUb_;
    SetFlag<HardEvent::MTE3_MTE2>(eventSft_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft_);
    SetFlag<HardEvent::V_MTE2>(eventSft_);
    WaitFlag<HardEvent::V_MTE2>(eventSft_);
    for (uint32_t b = 0; b < k; b++) {
        uint64_t slotB = (static_cast<uint64_t>(constInfo_.aivIdx) * ktiles + t0 + b) * Q_CHUNK_SIZE * 32;
        uint64_t pmSlotB = (static_cast<uint64_t>(constInfo_.aivIdx) * ktiles + t0 + b) * Q_CHUNK_SIZE;
        for (uint32_t si = 0; si < sliceNum; si++) {
            const PostRowSlice& sl = slices[si];
            uint32_t segElems = sl.rowCount * 32;
            for (uint32_t off = 0; off < segElems; off += 1024) {
                uint32_t n = BSAMin(uint32_t(1024), segElems - off);
                DataCopyPad(psUb[b * 2048 + sl.localRowStart * 32 + off],
                            partSumGmTensor_[slotB + sl.chunkRowStart * 32 + off],
                            {1, static_cast<uint16_t>(n * sizeof(float)), 0, 0}, {false, 0, 0, 0});
            }
            DataCopyPad(pmT[b * Q_CHUNK_SIZE + sl.localRowStart], partMaxGmTensor_[pmSlotB + sl.chunkRowStart],
                        {1, static_cast<uint16_t>(((sl.rowCount + 7u) / 8u * 8u) * sizeof(float)), 0, 0},
                        {false, 0, 0, 0});
        }
    }
    SetFlag<HardEvent::MTE2_V>(eventSft_);
    WaitFlag<HardEvent::MTE2_V>(eventSft_);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::FinalizeBatchCorrTile(uint32_t b, uint32_t ownRows)
{
    LocalTensor<T> rs = postPoolOP_.GetReduceTemp();
    LocalTensor<T> corrAll = rs[BF_CORR_ALL_OFF_FIXED];
    LocalTensor<T> diffSlice = rs[BF_CORR_ALL_OFF_FIXED + Q_CHUNK_SIZE];
    LocalTensor<T> pmT = rs[BF_CORR_ALL_OFF_FIXED + 2 * Q_CHUNK_SIZE];
    LocalTensor<T> corrBroc = rs[BF_CORR_ALL_OFF_FIXED + 2 * Q_CHUNK_SIZE + BATCH * Q_CHUNK_SIZE];
    LocalTensor<T> psUb = scoreUb_;
    AscendC::Sub(diffSlice, pmT[b * Q_CHUNK_SIZE], globalMax_[0], ownRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(diffSlice, diffSlice, ownRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(corrAll, diffSlice, globalSum_[0], ownRows);
    AscendC::PipeBarrier<PIPE_V>();
    Brcb(corrBroc, corrAll, (ownRows + FLOAT_DATA_BLOCK_NUM - 1) / FLOAT_DATA_BLOCK_NUM, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    BSAvRowBroadcastMul(psUb[b * 2048], corrBroc, ownRows, 32);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::FinalizeBatchFoldUnits(const PostRowSlice* slices, uint32_t sliceNum,
                                                                     uint32_t b, uint32_t k, uint32_t qChunkStartFx)
{
    uint32_t unitRows = constInfo_.postBlockShapeX;
    LocalTensor<T> rs = postPoolOP_.GetReduceTemp();
    LocalTensor<uint8_t> raTmp = rs[0].ReinterpretCast<uint8_t>();
    LocalTensor<T> unitRes = scoreUb_[BATCH * 2048];
    LocalTensor<T> psUb = scoreUb_;
    for (uint32_t s = 0; s < sliceNum; s++) {
        const PostRowSlice& sl = slices[s];
        uint32_t gS0 = qChunkStartFx + sl.chunkRowStart;
        uint32_t firstPxL = (qChunkStartFx + slices[0].chunkRowStart) / unitRows;
        uint32_t uo = 0;
        while (uo < sl.rowCount) {
            uint32_t gS = gS0 + uo;
            uint32_t segEndA = (gS % unitRows == 0) ? (gS + unitRows) : (((gS / unitRows) + 1) * unitRows);
            uint32_t cnt = BSAMin(segEndA - gS, sl.rowCount - uo);
            uint32_t rel = sl.localRowStart + uo;
            uint32_t u = (gS / unitRows) - firstPxL;
            uint32_t uShape[] = {cnt, 32};
            AscendC::ReduceSum<T, AscendC::Pattern::Reduce::RA, true>(unitRes[(u * k + b) * 32],
                                                                      psUb[b * 2048 + rel * 32], raTmp, uShape, true);
            AscendC::PipeBarrier<PIPE_V>();
            uo += cnt;
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::FinalizeBatchScaleCast(const PostRowSlice* slices, uint32_t sliceNum,
                                                                     uint32_t k, uint32_t nUnits, uint32_t qChunkStart,
                                                                     uint32_t validXBlocks, bool invRowsUniform)
{
    uint32_t unitRows = constInfo_.postBlockShapeX;
    LocalTensor<T> rs = postPoolOP_.GetReduceTemp();
    LocalTensor<OUT_T> fp16Row = rs[BF_FP16_ROW_OFF].ReinterpretCast<OUT_T>();
    LocalTensor<T> invCols = rs[BF_INV_COLS_OFF];
    LocalTensor<T> unitRes = scoreUb_[BATCH * 2048];
    if (invRowsUniform) {
        AscendC::Muls(unitRes, unitRes, 1.0f / static_cast<float>(static_cast<int32_t>(unitRows)), k * nUnits * 32);
    } else {
        for (uint32_t s = 0; s < sliceNum; s++) {
            const PostRowSlice& sl = slices[s];
            uint32_t gS0 = qChunkStart + sl.chunkRowStart;
            uint32_t firstPxL = (qChunkStart + slices[0].chunkRowStart) / unitRows;
            uint32_t uo = 0;
            while (uo < sl.rowCount) {
                uint32_t gS = gS0 + uo;
                uint32_t segEndA = (gS % unitRows == 0) ? (gS + unitRows) : (((gS / unitRows) + 1) * unitRows);
                uint32_t cnt = BSAMin(segEndA - gS, sl.rowCount - uo);
                uint32_t px = gS / unitRows;
                uint32_t rowsInPx = BSAMin(unitRows, validXBlocks - px * unitRows);
                uint32_t u = px - firstPxL;
                for (uint32_t b = 0; b < k; b++) {
                    int32_t rowsInPxI = static_cast<int32_t>(rowsInPx);
                    AscendC::Muls(unitRes[(u * k + b) * 32], unitRes[(u * k + b) * 32],
                                  1.0f / static_cast<float>(rowsInPxI), 32);
                }
                uo += cnt;
            }
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::BinaryRepeatParams bpCols;
    bpCols.dstBlkStride = 1;
    bpCols.src0BlkStride = 1;
    bpCols.src1BlkStride = 1;
    bpCols.dstRepStride = 4;
    bpCols.src0RepStride = 4;
    bpCols.src1RepStride = 0;
    AscendC::Mul<T, false>(unitRes, unitRes, invCols, static_cast<uint64_t>(32), static_cast<uint8_t>(k * nUnits),
                           bpCols);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(fp16Row, unitRes, AscendC::RoundMode::CAST_ROUND, k * nUnits * 32);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::FinalizeBatchWriteOut(const PostRowSlice* slices, uint32_t sliceNum,
                                                                    uint32_t qChunkStart, uint32_t k, uint32_t t0,
                                                                    uint32_t groupsValid, uint32_t validXBlocksBx,
                                                                    uint32_t nUnitsFx, uint64_t pooledHeadOffset,
                                                                    uint32_t pooledRowStride,
                                                                    GlobalTensor<OUT_T>& attnScoreGmTensor)
{
    uint32_t unitRows = constInfo_.postBlockShapeX;
    LocalTensor<T> rs = postPoolOP_.GetReduceTemp();
    LocalTensor<OUT_T> fp16Row = rs[BF_FP16_ROW_OFF].ReinterpretCast<OUT_T>();
    LocalTensor<T> unitRes = scoreUb_[BATCH * 2048];
    SetFlag<HardEvent::V_MTE3>(eventSft_);
    WaitFlag<HardEvent::V_MTE3>(eventSft_);
    for (uint32_t s = 0; s < sliceNum; s++) {
        const PostRowSlice& sl = slices[s];
        uint32_t gS0 = qChunkStart + sl.chunkRowStart;
        uint32_t firstPxL = (qChunkStart + slices[0].chunkRowStart) / unitRows;
        uint32_t uo = 0;
        while (uo < sl.rowCount) {
            uint32_t gS = gS0 + uo;
            uint32_t segEndA = (gS % unitRows == 0) ? (gS + unitRows) : (((gS / unitRows) + 1) * unitRows);
            uint32_t cnt = BSAMin(segEndA - gS, sl.rowCount - uo);
            uint32_t px = gS / unitRows;
            uint32_t u = px - firstPxL;
            uint64_t gmCol = pooledHeadOffset + static_cast<uint32_t>(px) * pooledRowStride + t0 * (K_TILE_COLS / 16);
            uint32_t unitEndB = BSAMin((px + 1) * unitRows, validXBlocksBx);
            bool segIsTailB = ((gS + cnt) >= unitEndB);
            if (!segIsTailB) {
                AscendC::Add(pendingUb_[0], pendingUb_[0], unitRes[u * k * 32], 32);
                AscendC::PipeBarrier<PIPE_V>();
                uo += cnt;
                continue;
            }
            if (gS % unitRows != 0) {
                AscendC::Add(unitRes[u * k * 32], unitRes[u * k * 32], pendingUb_[0], 32);
                AscendC::Duplicate(pendingUb_[0], float(0.0f), 32);
                AscendC::PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(eventSft_);
                WaitFlag<HardEvent::V_MTE3>(eventSft_);
                AscendC::Cast(fp16Row, unitRes, AscendC::RoundMode::CAST_ROUND, nUnitsFx * k * 32);
                AscendC::PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::MTE3_V>(eventSft_);
                WaitFlag<HardEvent::MTE3_V>(eventSft_);
            }
            DataCopyPad(attnScoreGmTensor[gmCol], fp16Row[u * k * 32],
                        {1, static_cast<uint16_t>(groupsValid * k * sizeof(OUT_T)), 0, 0});
            uo += cnt;
        }
    }
    SetFlag<HardEvent::MTE3_V>(eventSft_);
    WaitFlag<HardEvent::MTE3_V>(eventSft_);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::SoftmaxFinalizeBatch(const PostRowSlice* slices, uint32_t sliceNum,
                                                                   uint32_t qChunkStart, uint32_t batchIdx,
                                                                   uint32_t headIdx, uint32_t validXBlocks,
                                                                   uint32_t validYBlocks, uint32_t t0, uint32_t k,
                                                                   uint32_t groupsValid, bool boundary)
{
    GlobalTensor<OUT_T> attnScoreGmTensor = attnScoreFp16GmTensor_;
    uint32_t validPostYCur = (validYBlocks + constInfo_.postBlockShapeY - 1) / constInfo_.postBlockShapeY;
    uint32_t pooledRowStride = validPostYCur;
    uint64_t pooledHeadOffset = 0;
    uint32_t unitRows = constInfo_.postBlockShapeX;
    LocalTensor<T> rs = postPoolOP_.GetReduceTemp();
    LocalTensor<T> invCols = rs[BF_INV_COLS_OFF];

    uint32_t ownRows = slices[sliceNum - 1].localRowStart + slices[sliceNum - 1].rowCount;
    uint32_t firstPx = (qChunkStart + slices[0].chunkRowStart) / unitRows;
    uint32_t lastPx =
        (qChunkStart + slices[sliceNum - 1].chunkRowStart + (slices[sliceNum - 1].rowCount - 1)) / unitRows;
    uint32_t nUnits = lastPx - firstPx + 1;
    bool invRowsUniform = (validXBlocks % unitRows == 0);

    FinalizeBatchReadParts(slices, sliceNum, k, t0);

    if (boundary) {
        uint32_t py0 = t0 * (K_TILE_COLS / 16);
        AscendC::Duplicate(invCols, 1.0f / 16.0f, 32);
        AscendC::PipeBarrier<PIPE_V>();
        uint32_t validGroups = (validYBlocks > py0 * 16) ? (validYBlocks - py0 * 16 + 15) / 16 : 0;
        if (validGroups > 0) {
            uint32_t lastG = BSAMin(validGroups, uint32_t(32)) - 1;
            uint32_t lastCols = BSAMin(uint32_t(16), validYBlocks - (py0 + lastG) * 16);
            if (lastCols != 16) {
                VToSSync();
                float v = invCols.GetValue(lastG);
                float lastColsF = static_cast<float>(static_cast<int32_t>(lastCols));
                invCols.SetValue(lastG, v * 16.0f / lastColsF);
                SToVSync();
            }
        }
    }

    for (uint32_t b = 0; b < k; b++) {
        FinalizeBatchCorrTile(b, ownRows);
        FinalizeBatchFoldUnits(slices, sliceNum, b, k, qChunkStart);
    }

    FinalizeBatchScaleCast(slices, sliceNum, k, nUnits, qChunkStart, validXBlocks, invRowsUniform);

    FinalizeBatchWriteOut(slices, sliceNum, qChunkStart, k, t0, groupsValid, validXBlocks, nUnits, pooledHeadOffset,
                          pooledRowStride, attnScoreGmTensor);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::SoftmaxFinalize(uint32_t qChunkStart, uint32_t qChunkSize,
                                                              uint32_t batchIdx, uint32_t headIdx,
                                                              uint32_t validXBlocks, uint32_t validYBlocks)
{
    if (!constInfo_.usePostBlockShape) {
        return;
    }
    if (cachedChunkKey_ != qChunkStart || cachedValidXBlocks_ != validXBlocks) {
        FillPass1Ctx(qChunkStart, qChunkSize, validXBlocks);
    }
    const PostRowSlice* slices = cachedSlices_;
    uint32_t sliceNum = cachedSliceNum_;
    if (sliceNum == 0) {
        return;
    }

    LocalTensor<T> rs0 = postPoolOP_.GetReduceTemp();
    LocalTensor<T> invCols = rs0[1536];
    AscendC::Duplicate(invCols, float(1.0f / 16.0f), 32);
    AscendC::PipeBarrier<PIPE_V>();

    uint32_t lastFull = validYBlocks / K_TILE_COLS;
    uint32_t ktilesBatch = (validYBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
    if (constInfo_.usePostBlockShape && constInfo_.postBlockShapeY != 16) {
        GeneralFinalize(slices, sliceNum, qChunkStart, batchIdx, headIdx, validXBlocks, validYBlocks);
        return;
    }
    for (uint32_t t0 = 0; t0 < lastFull; t0 += BATCH) {
        uint32_t k = BSAMin(BATCH, lastFull - t0);
        SoftmaxFinalizeBatch(slices, sliceNum, qChunkStart, batchIdx, headIdx, validXBlocks, validYBlocks, t0, k,
                             K_TILE_COLS / 16, false);
    }
    if (lastFull < ktilesBatch) {
        uint32_t groupsValid = (validYBlocks - lastFull * K_TILE_COLS + 15) / 16;
        SoftmaxFinalizeBatch(slices, sliceNum, qChunkStart, batchIdx, headIdx, validXBlocks, validYBlocks, lastFull, 1,
                             groupsValid, true);
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::GeneralWinGeometry(uint32_t P, uint32_t nUnits, uint32_t& colsPerWin,
                                                                 uint32_t& groupsWinA)
{
    uint32_t accBase = 9600;
    uint32_t accCap = 32768 - accBase - 768;
    uint32_t gcdP8 = P;
    uint32_t gcdR = 8;
    while (gcdR != 0) {
        uint32_t t = gcdP8 % gcdR;
        gcdP8 = gcdR;
        gcdR = t;
    }
    uint32_t stepCols = 8 / gcdP8;
    uint32_t colsCap = accCap / nUnits / P;
    colsPerWin = BSAMax(stepCols, (BSAMin(colsCap, uint32_t(256)) / stepCols) * stepCols);
    groupsWinA = (colsPerWin * P + 7) / 8 * 8;
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::GeneralWinReadTile(const PostRowSlice* slices, uint32_t sliceNum,
                                                                 uint32_t t, uint32_t G)
{
    uint32_t ktiles = (constInfo_.yBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
    LocalTensor<T> psUb = scoreUb_[0];
    LocalTensor<T> pmRow = scoreUb_[GF_PM_ROW_OFF];
    uint64_t slotB = (static_cast<uint64_t>(constInfo_.aivIdx) * ktiles + t) * Q_CHUNK_SIZE * G;
    uint64_t pmSlotB = (static_cast<uint64_t>(constInfo_.aivIdx) * ktiles + t) * Q_CHUNK_SIZE;
    SetFlag<HardEvent::V_MTE2>(EVENT_FIN);
    WaitFlag<HardEvent::V_MTE2>(EVENT_FIN);
    for (uint32_t si = 0; si < sliceNum; si++) {
        const PostRowSlice& sl = slices[si];
        uint32_t segElems = sl.rowCount * G;
        for (uint32_t off = 0; off < segElems; off += 1024) {
            uint32_t n = BSAMin(uint32_t(1024), segElems - off);
            DataCopyPad(psUb[sl.localRowStart * G + off], partSumGmTensor_[slotB + sl.chunkRowStart * G + off],
                        {1, static_cast<uint16_t>(n * sizeof(float)), 0, 0}, {false, 0, 0, 0});
        }
        DataCopyPad(pmRow[sl.localRowStart], partMaxGmTensor_[pmSlotB + sl.chunkRowStart],
                    {1, static_cast<uint16_t>(((sl.rowCount + 7u) / 8u * 8u) * sizeof(float)), 0, 0}, {false, 0, 0, 0});
    }
    SetFlag<HardEvent::MTE2_V>(EVENT_FIN);
    WaitFlag<HardEvent::MTE2_V>(EVENT_FIN);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::GeneralWinCorr(uint32_t G, uint32_t ownRows)
{
    LocalTensor<T> psUb = scoreUb_[0];
    LocalTensor<T> pmRow = scoreUb_[GF_PM_ROW_OFF];
    LocalTensor<T> corrRow = scoreUb_[GF_CORR_ROW_OFF];
    LocalTensor<T> corrBroc = scoreUb_[GF_CORR_BROC_OFF];
    AscendC::Sub(corrRow, pmRow, globalMax_[0], ownRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(corrRow, corrRow, ownRows);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(corrRow, corrRow, globalSum_[0], ownRows);
    AscendC::PipeBarrier<PIPE_V>();
    Brcb(corrBroc, corrRow, (ownRows + FLOAT_DATA_BLOCK_NUM - 1) / FLOAT_DATA_BLOCK_NUM, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    BSAvRowBroadcastMul(psUb, corrBroc, ownRows, G);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::GeneralWinFoldAcc(const PostRowSlice* slices, uint32_t sliceNum,
                                                                uint32_t G, uint32_t tG, uint32_t gLo, uint32_t gHi,
                                                                uint32_t unitRows, uint32_t groupsWinA,
                                                                uint32_t qChunkStartFx, uint32_t nUnitsAccFold)
{
    LocalTensor<T> rsG = postPoolOP_.GetReduceTemp();
    LocalTensor<uint8_t> raTmp0 = rsG[0].ReinterpretCast<uint8_t>();
    LocalTensor<T> psUb = scoreUb_[0];
    LocalTensor<T> unitGrp = scoreUb_[GF_UNIT_GRP_OFF];
    LocalTensor<T> acc = scoreUb_[GF_ACC_BASE];
    uint32_t lg0 = (gLo > tG) ? (gLo - tG) : 0;
    uint32_t lg1 = BSAMin(G, gHi - tG);
    for (uint32_t s = 0; s < sliceNum; s++) {
        const PostRowSlice& sl = slices[s];
        uint32_t gS0 = qChunkStartFx + sl.chunkRowStart;
        uint32_t firstPxL = (qChunkStartFx + slices[0].chunkRowStart) / unitRows;
        uint32_t uo = 0;
        while (uo < sl.rowCount) {
            uint32_t gS = gS0 + uo;
            uint32_t segEndA = (gS % unitRows == 0) ? (gS + unitRows) : (((gS / unitRows) + 1) * unitRows);
            uint32_t cnt = BSAMin(segEndA - gS, sl.rowCount - uo);
            uint32_t rel = sl.localRowStart + uo;
            uint32_t u = (gS / unitRows) - firstPxL;
            uint32_t cntP2 = 8u;
            while (cntP2 < cnt) {
                cntP2 <<= 1;
            }
            LocalTensor<T> foldTmp = scoreUb_[GF_ACC_BASE + nUnitsAccFold * groupsWinA];
            AscendC::Adds(foldTmp, psUb[rel * G], 0.0f, cnt * G);
            AscendC::PipeBarrier<PIPE_V>();
            if (cntP2 > cnt) {
                AscendC::Duplicate(foldTmp[cnt * G], float(0.0f), (cntP2 - cnt) * G);
                AscendC::PipeBarrier<PIPE_V>();
            }
            for (uint32_t half = cntP2 / 2; half > 0; half >>= 1) {
                AscendC::Add(foldTmp, foldTmp, foldTmp[half * G], half * G);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::Add(acc[u * groupsWinA + (tG + lg0 - gLo)], acc[u * groupsWinA + (tG + lg0 - gLo)], foldTmp[lg0],
                         lg1 - lg0);
            AscendC::PipeBarrier<PIPE_V>();
            uo += cnt;
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::GeneralWinRowMean(const PostRowSlice* slices, uint32_t sliceNum,
                                                                uint32_t qChunkStart, uint32_t validXBlocks,
                                                                uint32_t nUnits, uint32_t winGroups,
                                                                uint32_t groupsWinA)
{
    uint32_t unitRows = constInfo_.postBlockShapeX;
    LocalTensor<T> acc = scoreUb_[GF_ACC_BASE];
    if (validXBlocks % unitRows == 0) {
        AscendC::Muls(acc, acc, 1.0f / static_cast<float>(static_cast<int32_t>(unitRows)), nUnits * groupsWinA);
    } else {
        for (uint32_t s = 0; s < sliceNum; s++) {
            const PostRowSlice& sl = slices[s];
            uint32_t gS0 = qChunkStart + sl.chunkRowStart;
            uint32_t firstPxL = (qChunkStart + slices[0].chunkRowStart) / unitRows;
            uint32_t uo = 0;
            while (uo < sl.rowCount) {
                uint32_t gS = gS0 + uo;
                uint32_t segEndA = (gS % unitRows == 0) ? (gS + unitRows) : (((gS / unitRows) + 1) * unitRows);
                uint32_t cnt = BSAMin(segEndA - gS, sl.rowCount - uo);
                uint32_t px = gS / unitRows;
                uint32_t rowsInPx = BSAMin(unitRows, validXBlocks - px * unitRows);
                uint32_t u = px - firstPxL;
                AscendC::Muls(acc[u * groupsWinA], acc[u * groupsWinA],
                              1.0f / static_cast<float>(static_cast<int32_t>(rowsInPx)), winGroups);
                uo += cnt;
            }
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::GeneralWinAssembleWrite(
    const PostRowSlice* slices, uint32_t sliceNum, uint32_t qChunkStart, uint32_t validYBlocks, uint32_t pbY,
    uint32_t P, uint32_t c0, uint32_t c1, uint32_t gLo, uint32_t gHi, uint32_t winGroups, uint32_t groupsWinA,
    uint32_t colsPerWinFx, uint32_t validXBlocksFx, uint64_t pooledHeadOffset, uint32_t pooledRowStride,
    GlobalTensor<OUT_T>& attnScoreGmTensor)
{
    uint32_t unitRows = constInfo_.postBlockShapeX;
    LocalTensor<T> ub = scoreUb_;
    LocalTensor<T> acc = ub[GF_ACC_BASE];
    LocalTensor<T> outRow = ub[GF_OUT_ROW_OFF];
    LocalTensor<T> colScale = ub[GF_COL_SCALE_OFF];
    LocalTensor<OUT_T> outF16 = ub[GF_OUT_F16_OFF].ReinterpretCast<OUT_T>();
    VToSSync();
    uint32_t firstPxW = (qChunkStart + slices[0].chunkRowStart) / unitRows;
    for (uint32_t s = 0; s < sliceNum; s++) {
        const PostRowSlice& sl = slices[s];
        uint32_t gS0 = qChunkStart + sl.chunkRowStart;
        uint32_t uo = 0;
        while (uo < sl.rowCount) {
            uint32_t gS = gS0 + uo;
            uint32_t segEndA = (gS % unitRows == 0) ? (gS + unitRows) : (((gS / unitRows) + 1) * unitRows);
            uint32_t cnt = BSAMin(segEndA - gS, sl.rowCount - uo);
            uint32_t chunkRow = sl.chunkRowStart + uo;
            uint32_t px = gS / unitRows;
            uint32_t u = px - firstPxW;
            uint32_t segRows = cnt;
            uint32_t unitEnd = BSAMin((px + 1) * unitRows, validXBlocksFx);
            bool segIsTail = ((gS + segRows) >= unitEnd);
            uint32_t winIdx = c0 / BSAMax(colsPerWinFx, 1u);
            if (!segIsTail) {
                AscendC::Add(pendingUb_[winIdx * PENDING_GRP_MAX], pendingUb_[winIdx * PENDING_GRP_MAX],
                             acc[u * groupsWinA], winGroups);
                AscendC::PipeBarrier<PIPE_V>();
                uo += cnt;
                continue;
            }
            if (gS % unitRows != 0) {
                AscendC::Add(acc[u * groupsWinA], acc[u * groupsWinA], pendingUb_[winIdx * PENDING_GRP_MAX], winGroups);
                AscendC::Duplicate(pendingUb_[winIdx * PENDING_GRP_MAX], float(0.0f), winGroups);
                AscendC::PipeBarrier<PIPE_V>();
            }
            const LocalTensor<T> accRow = acc[u * groupsWinA];
            uint32_t winCols = c1 - c0;
            if (P == 1 && gHi == gLo + winCols) {
                AscendC::Adds(outRow, accRow, 0.0f, winGroups);
                AscendC::PipeBarrier<PIPE_V>();
            } else {
                for (uint32_t c = c0; c < c1; c++) {
                    uint32_t g0 = (c - c0) * P;
                    uint32_t gN = BSAMin(P, gHi - c * P);
                    float sum = 0.0f;
                    for (uint32_t pj = 0; pj < gN; pj++) {
                        sum += accRow.GetValue(g0 + pj);
                    }
                    outRow.SetValue(c - c0, sum);
                }
                SToVSync();
            }
            float pbYF = static_cast<float>(static_cast<int32_t>(pbY));
            AscendC::Muls(outRow, outRow, 1.0f / pbYF, BSAMax(uint32_t(8), winGroups));
            AscendC::PipeBarrier<PIPE_V>();
            uint32_t lastCol = winCols - 1;
            uint32_t tailWidth = validYBlocks - (c1 - 1) * pbY;
            if (tailWidth < pbY) {
                VToSSync();
                float cur = outRow.GetValue(lastCol);
                float tailWF = static_cast<float>(static_cast<int32_t>(tailWidth));
                outRow.SetValue(lastCol, cur * pbYF / tailWF);
                SToVSync();
            }
            AscendC::Cast(outF16, outRow, AscendC::RoundMode::CAST_ROUND, BSAMax(uint32_t(16), c1 - c0));
            AscendC::PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE3>(EVENT_FIN);
            WaitFlag<HardEvent::V_MTE3>(EVENT_FIN);
            DataCopyPad(attnScoreGmTensor[pooledHeadOffset + static_cast<uint64_t>(px) * pooledRowStride + c0], outF16,
                        {1, static_cast<uint16_t>((c1 - c0) * sizeof(OUT_T)), 0, 0});
            SetFlag<HardEvent::MTE3_V>(EVENT_FIN);
            WaitFlag<HardEvent::MTE3_V>(EVENT_FIN);
            uo += cnt;
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::GeneralFinalize(const PostRowSlice* slices, uint32_t sliceNum,
                                                              uint32_t qChunkStart, uint32_t batchIdx, uint32_t headIdx,
                                                              uint32_t validXBlocks, uint32_t validYBlocks)
{
    uint32_t pbY = constInfo_.postBlockShapeY;
    uint32_t foldW = (pbY % 16 == 0) ? 16u : 8u;
    uint32_t G = K_TILE_COLS / foldW;
    uint32_t P = pbY / foldW;
    uint32_t unitRows = constInfo_.postBlockShapeX;
    uint32_t validPostYCur = (validYBlocks + pbY - 1) / pbY;
    uint32_t globalGroups = (validYBlocks + foldW - 1) / foldW;
    uint32_t ktiles = (constInfo_.yBlocks + K_TILE_COLS - 1) / K_TILE_COLS;
    GlobalTensor<OUT_T> attnScoreGmTensor = attnScoreFp16GmTensor_;
    uint32_t pooledRowStride = validPostYCur;
    uint64_t pooledHeadOffset = 0;

    uint32_t ownRows = slices[sliceNum - 1].localRowStart + slices[sliceNum - 1].rowCount;
    uint32_t firstPx = (qChunkStart + slices[0].chunkRowStart) / unitRows;
    uint32_t lastPx =
        (qChunkStart + slices[sliceNum - 1].chunkRowStart + (slices[sliceNum - 1].rowCount - 1)) / unitRows;
    uint32_t nUnits = lastPx - firstPx + 1;

    uint32_t colsPerWin = 0;
    uint32_t groupsWinA = 0;
    GeneralWinGeometry(P, nUnits, colsPerWin, groupsWinA);
    LocalTensor<T> acc = scoreUb_[GF_ACC_BASE];

    for (uint32_t c0 = 0; c0 < validPostYCur; c0 += colsPerWin) {
        uint32_t c1 = BSAMin(c0 + colsPerWin, validPostYCur);
        uint32_t gLo = c0 * P;
        uint32_t gHi = BSAMin(c1 * P, globalGroups);
        uint32_t winGroups = gHi - gLo;

        SetFlag<HardEvent::MTE3_MTE2>(EVENT_FIN);
        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_FIN);
        SetFlag<HardEvent::V_MTE2>(EVENT_FIN);
        WaitFlag<HardEvent::V_MTE2>(EVENT_FIN);
        AscendC::Duplicate(acc, float(0.0f), nUnits * groupsWinA);
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t t = 0; t < ktiles; t++) {
            uint32_t tG = t * G;
            if (tG >= gHi || tG + G <= gLo) {
                continue;
            }
            GeneralWinReadTile(slices, sliceNum, t, G);
            GeneralWinCorr(G, ownRows);
            GeneralWinFoldAcc(slices, sliceNum, G, tG, gLo, gHi, unitRows, groupsWinA, qChunkStart, nUnits);
        }

        GeneralWinRowMean(slices, sliceNum, qChunkStart, validXBlocks, nUnits, winGroups, groupsWinA);

        GeneralWinAssembleWrite(slices, sliceNum, qChunkStart, validYBlocks, pbY, P, c0, c1, gLo, gHi, winGroups,
                                groupsWinA, colsPerWin, validXBlocks, pooledHeadOffset, pooledRowStride,
                                attnScoreGmTensor);
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::Pass2LoadBand(uint64_t tileBase, uint32_t bandChunkStart,
                                                            uint32_t bandRows, uint32_t kChunkSize,
                                                            uint32_t kChunkSizeAlign)
{
    SetFlag<HardEvent::V_MTE2>(eventSft_);
    WaitFlag<HardEvent::V_MTE2>(eventSft_);

    NzToNdCopyIn(scoreUb_, tileBase, bandChunkStart, bandRows, kChunkSize, kChunkSizeAlign);

    SetFlag<HardEvent::MTE2_V>(eventSft_);
    WaitFlag<HardEvent::MTE2_V>(eventSft_);

    if (kChunkSize < kChunkSizeAlign) {
        uint32_t padAlign2 = (kChunkSize + 7u) / 8u * 8u;
        if (padAlign2 < kChunkSizeAlign) {
            for (uint32_t r = 0; r < bandRows; r++) {
                AscendC::Duplicate(scoreUb_[static_cast<uint32_t>(r) * kChunkSizeAlign + padAlign2],
                                   float(SOFTMAX_NEG_INF), kChunkSizeAlign - padAlign2);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::Pass2NormalizeSlice(LocalTensor<T> tile, const PostRowSlice& sl,
                                                                  uint32_t kChunkStart, uint32_t kChunkSize,
                                                                  uint32_t kChunkSizeAlign)
{
    Muls(tile, tile, (float)(constInfo_.scaleValue), sl.rowCount * kChunkSizeAlign);
    AscendC::PipeBarrier<PIPE_V>();
    uint32_t brocRep2 = BSACeilDiv(sl.rowCount, static_cast<uint32_t>(BROC_BASE_NUM));
    Brcb(globalMaxBroc_[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], globalMax_[sl.localRowStart], brocRep2, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    Brcb(globalSumBroc_[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], globalSum_[sl.localRowStart], brocRep2, {1, 8});
    AscendC::PipeBarrier<PIPE_V>();
    BSASubRowBroadcast(tile, tile, globalMaxBroc_[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount,
                       kChunkSizeAlign);

    AscendC::Exp(tile, tile, sl.rowCount * kChunkSizeAlign);
    AscendC::PipeBarrier<PIPE_V>();
    if (kChunkSize < kChunkSizeAlign) {
        uint32_t rem2 = kChunkSize % 8u;
        uint32_t maskBase2 = kChunkSize - rem2;
        if (maskBase2 < kChunkSizeAlign) {
            for (uint32_t r = 0; r < sl.rowCount; r++) {
                AscendC::Mul(tile[static_cast<uint32_t>(r) * kChunkSizeAlign + maskBase2],
                             tile[static_cast<uint32_t>(r) * kChunkSizeAlign + maskBase2],
                             vmaskTable_[static_cast<uint32_t>(rem2) * 8u], 8);
            }
            for (uint32_t off = maskBase2 + 8u; off < kChunkSizeAlign; off += 8u) {
                for (uint32_t r = 0; r < sl.rowCount; r++) {
                    AscendC::Duplicate(tile[static_cast<uint32_t>(r) * kChunkSizeAlign + off], 0.0f, 8);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

    BSADivRowBroadcast(tile, tile, globalSumBroc_[sl.localRowStart * FLOAT_DATA_BLOCK_NUM], sl.rowCount,
                       kChunkSizeAlign);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::Pass2BuildOutView(const PostRowSlice& sl, uint32_t qChunkStart,
                                                                uint32_t kChunkStart, uint32_t batchIdx,
                                                                uint32_t headIdx, uint32_t validYBlocks,
                                                                GlobalTensor<OUT_T>& attnScoreChunkGm,
                                                                uint32_t& p2RowStride)
{
    uint32_t sftOutQRow = qChunkStart + sl.chunkRowStart;
    p2RowStride = validYBlocks;
    uint64_t attnScoreOffset = static_cast<uint64_t>(sftOutQRow) * p2RowStride + kChunkStart;
    attnScoreChunkGm.SetGlobalBuffer((__gm__ OUT_T*)(attnScoreFp16GmTensor_.GetPhyAddr() + attnScoreOffset));
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::Pass2CastWriteSlice(LocalTensor<T> tile, const PostRowSlice& sl,
                                                                  uint32_t qChunkStart, uint32_t kChunkStart,
                                                                  uint32_t kChunkSize, uint32_t kChunkSizeAlign,
                                                                  uint32_t validXBlocks, uint32_t validYBlocks,
                                                                  uint32_t p2RowStride,
                                                                  GlobalTensor<OUT_T>& attnScoreChunkGm)
{
    AscendC::Cast(scoreFp16OutUb_, tile, AscendC::RoundMode::CAST_ROUND, sl.rowCount * kChunkSizeAlign);
    AscendC::PipeBarrier<PIPE_V>();

    SetFlag<HardEvent::V_MTE3>(eventSft_);
    WaitFlag<HardEvent::V_MTE3>(eventSft_);

    uint32_t rowBytes = kChunkSize * sizeof(OUT_T);
    bool tinyRow = (rowBytes < 32u);
    bool isGlobalLastRow =
        (static_cast<uint64_t>(qChunkStart) + sl.chunkRowStart + sl.rowCount >= static_cast<uint64_t>(validXBlocks));
    if (tinyRow && !isGlobalLastRow) {
        for (uint32_t r = 0; r < sl.rowCount; r++) {
            DataCopyPad(attnScoreChunkGm[static_cast<uint64_t>(r) * p2RowStride],
                        scoreFp16OutUb_[static_cast<uint64_t>(r) * kChunkSizeAlign], {1, rowBytes, 0, 0, 0});
        }
    } else {
        Pass2WriteLastSlice(sl, kChunkStart, kChunkSize, kChunkSizeAlign, validYBlocks, p2RowStride, rowBytes,
                            attnScoreChunkGm);
    }
    SetFlag<HardEvent::MTE3_V>(eventSft_);
    WaitFlag<HardEvent::MTE3_V>(eventSft_);
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::Pass2WriteLastSlice(const PostRowSlice& sl, uint32_t kChunkStart,
                                                                  uint32_t kChunkSize, uint32_t kChunkSizeAlign,
                                                                  uint32_t validYBlocks, uint32_t p2RowStride,
                                                                  uint32_t rowBytes,
                                                                  GlobalTensor<OUT_T>& attnScoreChunkGm)
{
    (void)kChunkStart;
    (void)validYBlocks;
    for (uint32_t r = 0; r < sl.rowCount; r++) {
        DataCopyPad(attnScoreChunkGm[static_cast<uint64_t>(r) * p2RowStride],
                    scoreFp16OutUb_[static_cast<uint64_t>(r) * kChunkSizeAlign], {1, rowBytes, 0, 0, 0});
    }
}

template <typename BSAT>
__aicore__ inline void BSAVecSmService<BSAT>::SoftmaxSecondPassAndCast(uint32_t qChunkStart, uint32_t qChunkSize,
                                                                       uint32_t kChunkStart, uint32_t kChunkSize,
                                                                       uint32_t batchIdx, uint32_t headIdx,
                                                                       uint32_t validXBlocks, uint32_t validYBlocks)
{
    PostRowSlice slices[BSA_SM_SLICES_MAX];
    uint32_t sliceNum = postPoolOP_.GetChunkRowSlices(qChunkStart, qChunkSize, validXBlocks, slices);
    if (sliceNum == 0) {
        return;
    }

    uint32_t kChunkSizeAlign = (kChunkSize + 15) / 16 * 16;

    SetFlag<HardEvent::MTE3_MTE2>(eventSft_);
    WaitFlag<HardEvent::MTE3_MTE2>(eventSft_);

    uint64_t tileBase = BSAScratchTileOffset(constInfo_.aicIdx, qChunkStart, kChunkStart, constInfo_.yBlocks,
                                             constInfo_.usePostBlockShape);

    uint32_t bandCap = (Q_CHUNK_SIZE / 2) * K_CHUNK_SIZE * CV_EXEC_RATIO / kChunkSizeAlign;
    uint32_t s = 0;
    while (s < sliceNum) {
        uint32_t e = s + 1;
        while (e < sliceNum) {
            bool contiguous = (slices[e].chunkRowStart == slices[e - 1].chunkRowStart + slices[e - 1].rowCount) &&
                              (slices[e].localRowStart == slices[e - 1].localRowStart + slices[e - 1].rowCount);
            uint32_t mergedRows =
                slices[e - 1].localRowStart + slices[e - 1].rowCount - slices[s].localRowStart + slices[e].rowCount;
            if (!contiguous || mergedRows > bandCap) {
                break;
            }
            e++;
        }
        uint32_t bandLocalStart = slices[s].localRowStart;
        uint32_t bandChunkStart = slices[s].chunkRowStart;
        uint32_t bandRows = slices[e - 1].localRowStart + slices[e - 1].rowCount - bandLocalStart;

        Pass2LoadBand(tileBase, bandChunkStart, bandRows, kChunkSize, kChunkSizeAlign);

        for (uint32_t i = s; i < e; i++) {
            const PostRowSlice& sl = slices[i];
            LocalTensor<T> tile = scoreUb_[(sl.localRowStart - bandLocalStart) * kChunkSizeAlign];

            Pass2NormalizeSlice(tile, sl, kChunkStart, kChunkSize, kChunkSizeAlign);

            uint32_t p2RowStride = 0;
            GlobalTensor<OUT_T> attnScoreChunkGm;
            Pass2BuildOutView(sl, qChunkStart, kChunkStart, batchIdx, headIdx, validYBlocks, attnScoreChunkGm,
                              p2RowStride);
            Pass2CastWriteSlice(tile, sl, qChunkStart, kChunkStart, kChunkSize, kChunkSizeAlign, validXBlocks,
                                validYBlocks, p2RowStride, attnScoreChunkGm);
        }
        s = e;
    }
}

#endif
