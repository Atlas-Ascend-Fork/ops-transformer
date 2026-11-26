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
 * \file mhc_pre_split_bs.h
 * \brief BS-split Basic API implementation for MHC Pre
 */

#ifndef MHC_PRE_SPLIT_BS_H
#define MHC_PRE_SPLIT_BS_H

#include "mhc_pre_vector_compute.h"
#include "mhc_pre_cube_compute.h"

namespace MhcPre {

constexpr uint32_t MHC_PRE_BS_SEQUENTIAL_PARTIAL_K = 1024U;

#if defined(MHC_PRE_ENABLE_ASCEND950)
template <bool FIRST>
__simd_vf__ inline void MhcPreAccumulateBsMixWithCompensation(__ubuf__ float* sum, __ubuf__ float* correction,
                                                              __ubuf__ float* src, uint32_t elements, uint16_t loops)
{
    for (uint16_t index = 0U; index < loops; ++index) {
        const uint32_t offset = index * 64U;
        Reg::MaskReg mask = Reg::UpdateMask<float>(elements);
        Reg::RegTensor<float> partial, accum, error;
        Reg::LoadAlign(partial, src + offset);
        if constexpr (FIRST) {
            accum = partial;
            Reg::Duplicate(error, 0.0f, mask);
        } else {
            Reg::LoadAlign(accum, sum + offset);
            Reg::LoadAlign(error, correction + offset);
            MhcPreCompensatedAdd<false>(accum, error, partial, mask);
        }
        Reg::StoreAlign(sum + offset, accum, mask);
        Reg::StoreAlign(correction + offset, error, mask);
    }
}
#endif

template <class T, class P, int8_t RESI_MODE>
class MhcPreSplitBS {
public:
    __aicore__ inline MhcPreSplitBS() = default;

    __aicore__ inline void Init(InitParams initParams)
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        vector_.BindGlobalTensors(initParams);
        vector_.xGm_.template SetL2CacheHint<CacheRwMode::READ>(CacheMode::CACHE_MODE_PERSISTENT);
        vector_.InitFromTilingData(initParams.tilingData);
        stageRows_ = initParams.tilingData->bsStageRows;
        vector_.InitMNConfig();
        vector_.InitHMixBuffer(initParams);
        vector_.InitPipeAndCoreIdx(initParams.tPipeIn);
        alignedMixWidth_ = initParams.tilingData->fusionAlign;
        xQueueBytes_ = initParams.tilingData->bsXQueueBytes;

        // Fixpipe addresses the same UB offset on the AIC and both paired AIVs.
        uint32_t mixBytes = MhcPreAlign(((vector_.chunkTSize_ + 1U) / 2U) * alignedMixWidth_ * sizeof(P),
                                        MHC_PRE_BASIC_API_VL_FP32 * sizeof(P));
        partialSlots_ = vector_.implMode_ == MHC_PRE_IMPL_MODE_FP32 ? 2U : 1U;
        partialStride_ = mixBytes / sizeof(P);
        vector_.pipe_->InitBuffer(partialBuffer_, partialSlots_ * mixBytes);
        partialLocal_ = partialBuffer_.Get<P>();
        if ASCEND_IS_AIV {
            vector_.pipe_->InitBuffer(accumBuffer_, mixBytes);
            accumLocal_ = accumBuffer_.Get<P>();
        }
        InitUbBuffersBasicApi();
        vector_.pipe_->InitBuffer(l1Buffer_, MHC_PRE_BASIC_API_L1_ALLOC_SIZE);
        aL1_ = l1Buffer_.Get<float>();
        bL1_ = aL1_[MHC_PRE_BASIC_API_L1_BUF_NUM * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        if ASCEND_IS_AIV {
            if (stageRows_ == MHC_PRE_BS_DEFAULT_STAGE_ROWS) {
                vector_.pipe_->InitBuffer(nzBuffer_,
                                          (MHC_PRE_BS_DEFAULT_STAGE_ROWS + 1U) * vector_.ND_LENGTH * sizeof(P));
                nzLocal_ = nzBuffer_.Get<P>();
            }
        }
        SyncAll<false>();

        if ASCEND_IS_AIV {
            vector_.coreIdx_ = GetBlockIdx() / vector_.kDoubleBufferCount;
            vector_.AIVPreLoad();
        }
        if ASCEND_IS_AIC {
            mmService_.Init(vector_.implMode_);
        }
    }
#else
    {
        vector_.BindGlobalTensors(initParams);
        vector_.InitFromTilingData(initParams.tilingData);
        vector_.InitMNConfig();
        vector_.InitHMixBuffer(initParams);
        vector_.InitPipeAndCoreIdx(initParams.tPipeIn);

        InitUbBuffersBasicApi();
        SyncAll<false>();

        if ASCEND_IS_AIV {
            vector_.coreIdx_ = GetBlockIdx() / MHC_PRE_AIV_PER_AIC;
            vector_.AIVPreLoad();
        }
        if ASCEND_IS_AIC {
            vector_.pipe_->InitBuffer(l1Buffer_, MHC_PRE_BASIC_API_L1_ALLOC_SIZE);
            aL1_ = l1Buffer_.Get<float>();
            bL1_ = aL1_[MHC_PRE_BASIC_API_L1_BUF_NUM * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
            mmService_.Init(vector_.implMode_);
        }
    }
#endif

    __aicore__ inline void Process()
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        uint32_t tBlockNum = Ceil(vector_.totalLength_, vector_.chunkTSize_);

        for (uint64_t curblock = vector_.coreIdx_; curblock < tBlockNum; curblock += vector_.coreNum_) {
            vector_.InitBlockParams(curblock, tBlockNum);
            ConfigureDirectUbRowPartition();

            if ASCEND_IS_AIV {
                const uint32_t mixElements = vector_.vectorOffset_.singleCoreM * alignedMixWidth_;
                compensateMix_ = vector_.implMode_ == MHC_PRE_IMPL_MODE_FP32 && mixElements != 0U &&
                                 mixElements * sizeof(P) <= MHC_PRE_BS_OUT_QUEUE_BYTES;
                if (compensateMix_) {
                    mixCorrection_ = vector_.outQueue_.template AllocTensor<P>();
                }
            }

            if ASCEND_IS_AIC {
                vector_.mnConfig_.curSingleCoreK = vector_.mnConfig_.singleCoreK;
            }

            uint32_t partialIndex = 0U;
            for (uint32_t offsetNd = 0; offsetNd < vector_.matrixInfo_.nD; offsetNd += vector_.ND_LENGTH) {
                uint32_t curNdLen = MhcPreMin(vector_.ND_LENGTH, vector_.matrixInfo_.nD - offsetNd);
                bool isPartialEnd = IsSequentialPartialEnd(offsetNd, curNdLen);
                // Two X staging slots are free initially. From the third slice on,
                // AIV waits until AIC returns the slot through X_CONSUMED_FLAG.
                const uint32_t totalK = vector_.matrixInfo_.nD;
                const uint32_t groupK = BsProjectionGroupK(totalK);
                const bool grouped = vector_.implMode_ == MHC_PRE_IMPL_MODE_FP32 && totalK > groupK;
                if ASCEND_IS_AIV {
                    if (vector_.vectorCount_ >= 2) {
                        AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_CONSUMED_FLAG);
                    }
                    if (vector_.hasGamma_) {
                        V0PrologueL1<true>(curNdLen, offsetNd);
                    } else {
                        V0PrologueL1<false>(curNdLen, offsetNd);
                    }
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_READY_FLAG);
                    vector_.vectorCount_++;
                    if (grouped) {
                        // Publish the next X slice before draining the previous slice's projection.
                        if (offsetNd != 0U) {
                            ConsumeGroupedTile(offsetNd - vector_.ND_LENGTH, offsetNd, totalK, groupK, partialIndex);
                        }
                        if (offsetNd + curNdLen == totalK) {
                            ConsumeGroupedTile(offsetNd, totalK, totalK, groupK, partialIndex);
                        }
                    } else if (isPartialEnd) {
                        ConsumePartial(partialIndex == 0U);
                    }
                }

                if ASCEND_IS_AIC {
                    AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_READY_FLAG);
                    AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_READY_FLAG +
                                                                                        MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                    AICProcessBasicApi(offsetNd);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE1>(MHC_PRE_X_CONSUMED_FLAG +
                                                                              MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                }
                if (!grouped && isPartialEnd) {
                    partialIndex++;
                }
            }
            vector_.mmCount_++;
            if ASCEND_IS_AIV {
                if (compensateMix_) {
                    PipeBarrier<PIPE_V>();
                    vector_.outQueue_.FreeTensor(mixCorrection_);
                }
                vector_.V0PostProcess(curblock, tBlockNum);
                AIV1ProcessDirect();
                vector_.vec1Count_++;
            }
        }

        if ASCEND_IS_AIV {
            vector_.invRmsOutQueue_.FreeTensor(vector_.invRmsUb_);
            vector_.biasInQue_.FreeTensor(vector_.biasInUb_);
        } else {
            const uint32_t inFlight = MhcPreMin(partialSequence_, partialSlots_);
            for (uint32_t slot = 0; slot < inFlight; ++slot) {
                WaitPartialConsumed();
            }
            mmService_.End(vector_.implMode_);
        }
    }
#else
    {
        uint32_t tBlockNum = Ceil(vector_.totalLength_, vector_.chunkTSize_);

        for (uint64_t curblock = vector_.coreIdx_; curblock < tBlockNum; curblock += vector_.coreNum_) {
            vector_.InitBlockParams(curblock, tBlockNum);
            ConfigureDirectUbRowPartition();

            if ASCEND_IS_AIC {
                vector_.mnConfig_.curSingleCoreK = vector_.mnConfig_.singleCoreK;
            }

            uint32_t partialIndex = 0U;
            for (uint32_t offsetNd = 0; offsetNd < vector_.matrixInfo_.nD; offsetNd += vector_.ND_LENGTH) {
                uint32_t curNdLen = MhcPreMin(vector_.ND_LENGTH, vector_.matrixInfo_.nD - offsetNd);
                bool isPartialEnd = IsSequentialPartialEnd(offsetNd, curNdLen);
                // Two X staging slots are free initially. From the third slice on,
                // AIV waits until AIC returns the slot through X_CONSUMED_FLAG.
                if ASCEND_IS_AIV {
                    if (vector_.vectorCount_ >= 2) {
                        AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_CONSUMED_FLAG);
                    }
                    vector_.V0Prologue(curNdLen, offsetNd);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE3>(MHC_PRE_X_READY_FLAG);
                    vector_.vectorCount_++;
                }

                if ASCEND_IS_AIC {
                    AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_X_READY_FLAG);
                    if constexpr (MHC_PRE_AIV_PER_AIC == TWO_AIV_PER_AIC) {
                        AscendC::CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(
                            MHC_PRE_X_READY_FLAG + MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                    }
                    AICProcessBasicApi(offsetNd, partialIndex);
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_X_CONSUMED_FLAG);
                    if constexpr (MHC_PRE_AIV_PER_AIC == TWO_AIV_PER_AIC) {
                        CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_X_CONSUMED_FLAG +
                                                                                  MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                    }
                }
                if (isPartialEnd) {
                    partialIndex++;
                }
            }
            vector_.mmCount_++;
            if ASCEND_IS_AIC {
                CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_MM_READY_FLAG);
                if constexpr (MHC_PRE_AIV_PER_AIC == TWO_AIV_PER_AIC) {
                    CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_MM_READY_FLAG +
                                                                             MHC_PRE_SUBBLOCK_FLAG_OFFSET);
                }
            }

            if ASCEND_IS_AIV {
                vector_.V0PostProcess(curblock, tBlockNum);
                CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_MTE2>(MHC_PRE_MM_READY_FLAG);
                AIV1ProcessDirect();
                vector_.vec1Count_++;
            }
        }

        if ASCEND_IS_AIV {
            vector_.invRmsOutQueue_.FreeTensor(vector_.invRmsUb_);
            vector_.biasInQue_.FreeTensor(vector_.biasInUb_);
        } else {
            mmService_.End();
        }
    }
#endif

private:
#if defined(MHC_PRE_ENABLE_ASCEND950)
    uint32_t stageRows_ = MHC_PRE_BS_DEFAULT_STAGE_ROWS;
    uint32_t alignedMixWidth_ = 0U;
    uint32_t xQueueBytes_ = 0U;
    uint32_t rmsCorrectionOffset_ = 0U;
    uint32_t partialSlots_ = 1U;
    uint32_t partialStride_ = 0U;
    uint32_t partialSequence_ = 0U;
    __aicore__ static inline uint32_t BsProjectionGroupK(uint32_t totalK)
    {
        return MhcPreMin(MhcPreCubeCompute::ProjectionGroupK(totalK), 128U);
    }

    bool compensateMix_ = false;
    LocalTensor<P> mixCorrection_;
#endif

    __aicore__ inline void InitUbBuffersBasicApi()
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }

        vector_.pipe_->InitBuffer(vector_.xInQueue_, vector_.kDoubleBufferCount, xQueueBytes_);
        vector_.pipe_->InitBuffer(vector_.outQueue_, vector_.kDoubleBufferCount, MHC_PRE_BS_OUT_QUEUE_BYTES);

        uint32_t invRmsRows = (vector_.curSingleT_ + vector_.kHalfSplitDivisor - 1U) / vector_.kHalfSplitDivisor;
        uint32_t invRmsBytes = MhcPreAlign(invRmsRows, vector_.kAlignmentBytes / sizeof(P)) * sizeof(P);
        rmsCorrectionOffset_ = invRmsBytes / sizeof(P);
        vector_.pipe_->InitBuffer(vector_.invRmsOutQueue_, vector_.kSingleBufferCount, 2U * invRmsBytes);

        if (vector_.hasGamma_) {
            vector_.pipe_->InitBuffer(vector_.gammaInQueue_, vector_.kSingleBufferCount, vector_.ND_LENGTH * sizeof(P));
        }

        uint32_t parameterBufferBytes = MhcPreAlign(vector_.mnConfig_.n * sizeof(P), vector_.kAlignmentBytes);
        vector_.pipe_->InitBuffer(vector_.biasInQue_, vector_.kSingleBufferCount, parameterBufferBytes);
        vector_.pipe_->InitBuffer(vector_.alphaBuf_, parameterBufferBytes);
        vector_.alphaInUb_ = vector_.alphaBuf_.template Get<P>();

        uint32_t hSegmentBytes = MhcPreAlign(vector_.V1_BASE_T * vector_.N_ * sizeof(P), vector_.kAlignmentBytes);
        vector_.pipe_->InitBuffer(vector_.tmpBuff_, 2U * hSegmentBytes);
        vector_.hPreBuff_ =
            vector_.tmpBuff_.template GetWithOffset<P>(static_cast<uint32_t>(vector_.V1_BASE_T * vector_.N_), 0);
        vector_.hPostBuff_ = vector_.tmpBuff_.template GetWithOffset<P>(
            static_cast<uint32_t>(vector_.V1_BASE_T * vector_.N_), hSegmentBytes);
    }
#else
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }

        vector_.pipe_->InitBuffer(vector_.xInQueue_, vector_.kDoubleBufferCount, vector_.kXInQueueBufferBytes);
        vector_.pipe_->InitBuffer(vector_.outQueue_, vector_.kDoubleBufferCount, vector_.kOutQueueBufferBytes);

        uint32_t invRmsRows = MHC_PRE_AIV_PER_AIC == 1U ?
                                  vector_.curSingleT_ :
                                  (vector_.curSingleT_ + vector_.kHalfSplitDivisor - 1U) / vector_.kHalfSplitDivisor;
        uint32_t invRmsBytes = MhcPreAlign(invRmsRows, vector_.kAlignmentBytes / sizeof(P)) * sizeof(P);
        vector_.pipe_->InitBuffer(vector_.invRmsOutQueue_, vector_.kSingleBufferCount, invRmsBytes);

        if (vector_.hasGamma_) {
            vector_.pipe_->InitBuffer(vector_.gammaInQueue_, vector_.kSingleBufferCount, vector_.ND_LENGTH * sizeof(P));
        }

        uint32_t parameterBufferBytes = MhcPreAlign(vector_.mnConfig_.n * sizeof(P), vector_.kAlignmentBytes);
        vector_.pipe_->InitBuffer(vector_.biasInQue_, vector_.kSingleBufferCount, parameterBufferBytes);
        vector_.pipe_->InitBuffer(vector_.alphaBuf_, parameterBufferBytes);
        vector_.alphaInUb_ = vector_.alphaBuf_.template Get<P>();

        uint32_t hSegmentBytes = MhcPreAlign(vector_.V1_BASE_T * vector_.N_ * sizeof(P), vector_.kAlignmentBytes);
        vector_.pipe_->InitBuffer(vector_.tmpBuff_, 2U * hSegmentBytes);
        vector_.hPreBuff_ =
            vector_.tmpBuff_.template GetWithOffset<P>(static_cast<uint32_t>(vector_.V1_BASE_T * vector_.N_), 0);
        vector_.hPostBuff_ = vector_.tmpBuff_.template GetWithOffset<P>(
            static_cast<uint32_t>(vector_.V1_BASE_T * vector_.N_), hSegmentBytes);
    }
#endif

    __aicore__ inline void ConfigureDirectUbRowPartition()
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        uint64_t firstRows = (vector_.curSingleT_ + 1U) / 2U;
        if (vector_.subBlockIdx_ == 0) {
            vector_.vectorOffset_.singleCoreM = firstRows;
            vector_.vectorOffset_.offsetMStart = 0;
            vector_.vectorOffset_.offsetMEnd = firstRows;
        } else {
            vector_.vectorOffset_.singleCoreM = vector_.curSingleT_ - firstRows;
            vector_.vectorOffset_.offsetMStart = firstRows;
            vector_.vectorOffset_.offsetMEnd = vector_.curSingleT_;
        }
    }
#else
    {
        if ASCEND_IS_NOT_AIV {
            return;
        }
        if constexpr (MHC_PRE_AIV_PER_AIC == 1U) {
            vector_.vectorOffset_.singleCoreM = vector_.curSingleT_;
            vector_.vectorOffset_.offsetMStart = 0U;
            vector_.vectorOffset_.offsetMEnd = vector_.curSingleT_;
            return;
        }
        uint64_t firstRows = (vector_.curSingleT_ + 1U) / 2U;
        if (vector_.subBlockIdx_ == 0) {
            vector_.vectorOffset_.singleCoreM = firstRows;
            vector_.vectorOffset_.offsetMStart = 0;
            vector_.vectorOffset_.offsetMEnd = firstRows;
        } else {
            vector_.vectorOffset_.singleCoreM = vector_.curSingleT_ - firstRows;
            vector_.vectorOffset_.offsetMStart = firstRows;
            vector_.vectorOffset_.offsetMEnd = vector_.curSingleT_;
        }
    }
#endif

    __aicore__ inline bool IsSequentialPartialEnd(uint32_t offsetNd, uint32_t currentK) const
    {
        uint32_t endK = offsetNd + currentK;
        return endK >= vector_.matrixInfo_.nD || endK % MHC_PRE_BS_SEQUENTIAL_PARTIAL_K == 0U;
    }

#if defined(MHC_PRE_ENABLE_ASCEND950)
    __aicore__ inline void AICProcessBasicApi(uint32_t offsetNd)
    {
        if (offsetNd + vector_.mnConfig_.singleCoreK > vector_.mnConfig_.k) {
            vector_.mnConfig_.curSingleCoreK = vector_.mnConfig_.k - offsetNd;
        }

        uint32_t currentK = vector_.mnConfig_.curSingleCoreK;
        uint8_t aL1BufferId = vector_.cubeCount_ & 1;
        uint8_t bL1BufferId = mmService_.GetBL1BufferId();

        LocalTensor<float> currentAL1 = aL1_[aL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        LocalTensor<float> currentBL1 = bL1_[bL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        // X has already arrived in this core's L1 through its paired AIVs.
        mmService_.CopyInB1Nd2Nz(vector_.mnConfig_.k, currentK, vector_.mnConfig_.n, vector_.phiGm_[offsetNd],
                                 currentBL1);

        bool isFirstKL1 = offsetNd % MHC_PRE_BS_SEQUENTIAL_PARTIAL_K == 0U;
        bool isLastKL1 = IsSequentialPartialEnd(offsetNd, currentK);
        uint64_t mAlign = BasicApiAlign(vector_.mnConfig_.curSingleCoreM, AscendC::BLOCK_CUBE);
        uint64_t nAlign = BasicApiAlign(vector_.mnConfig_.n, AscendC::BLOCK_CUBE);
        // Fill available L0 from the live M/N footprint instead of hard-coding baseK for one shape.
        uint64_t baseK = (256 / AscendC::Std::max(mAlign, nAlign)) * 32;
        const uint32_t totalK = vector_.matrixInfo_.nD;
        const uint32_t groupK = BsProjectionGroupK(totalK);
        if (vector_.implMode_ == MHC_PRE_IMPL_MODE_FP32 && totalK > groupK) {
            const uint32_t tileEnd = offsetNd + currentK;
            for (uint32_t k = offsetNd; k < tileEnd;) {
                const uint32_t end = MhcPreCubeCompute::ProjectionGroupEnd(k, totalK, groupK);
                const uint32_t stop = MhcPreMin(end, tileEnd);
                mmService_.template ProcessProjectionSlice<true>(
                    vector_.mnConfig_.curSingleCoreM, vector_.mnConfig_.n, vector_.mnConfig_.curSingleCoreM, baseK,
                    k - offsetNd, stop - offsetNd,
                    k % groupK == 0U && (k == 0U || MhcPreCubeCompute::ProjectionGroupEnd(k - 1U, totalK, groupK) == k),
                    stop == end, currentAL1, currentBL1);
                if (stop == end) {
                    SendPartial();
                }
                k = stop;
            }
            mmService_.ReleaseProjectionB();
            vector_.cubeCount_++;
            return;
        }
        mmService_.template Process<true>(vector_.mnConfig_.curSingleCoreM, vector_.mnConfig_.n,
                                          vector_.mnConfig_.curSingleCoreM, baseK, isFirstKL1, isLastKL1, currentAL1,
                                          currentBL1);

        if (isLastKL1) {
            SendPartial();
        }
        vector_.cubeCount_++;
    }

    __aicore__ inline LocalTensor<P> PartialSlot()
    {
        return partialLocal_[(partialSequence_ % partialSlots_) * partialStride_];
    }

    __aicore__ inline void SendPartial()
    {
        if (partialSequence_ >= partialSlots_) {
            WaitPartialConsumed();
        }
        mmService_.CopyOut(PartialSlot());
        CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_BS_PARTIAL_READY_FLAG);
        CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_BS_PARTIAL_READY_FLAG +
                                                                 MHC_PRE_SUBBLOCK_FLAG_OFFSET);
        ++partialSequence_;
    }

    __aicore__ inline void WaitPartialConsumed()
    {
        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_BS_PARTIAL_CONSUMED_FLAG);
        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_FIX>(MHC_PRE_BS_PARTIAL_CONSUMED_FLAG +
                                                                  MHC_PRE_SUBBLOCK_FLAG_OFFSET);
    }

    __aicore__ inline void ConsumeGroupedTile(uint32_t start, uint32_t stop, uint32_t totalK, uint32_t groupK,
                                              uint32_t& partialIndex)
    {
        for (uint32_t end = MhcPreCubeCompute::ProjectionGroupEnd(start, totalK, groupK); end <= stop;
             end = MhcPreCubeCompute::ProjectionGroupEnd(end, totalK, groupK)) {
            ConsumePartial(partialIndex++ == 0U);
            if (end == totalK) {
                break;
            }
        }
    }

    __aicore__ inline void ConsumePartial(bool first)
    {
        CrossCoreWaitFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_V>(MHC_PRE_BS_PARTIAL_READY_FLAG);
        auto partial = PartialSlot();
        const uint32_t elements = vector_.vectorOffset_.singleCoreM * alignedMixWidth_;
        if (elements != 0U) {
            if (compensateMix_) {
                auto* sum = (__ubuf__ float*)accumLocal_.GetPhyAddr();
                auto* correction = (__ubuf__ float*)mixCorrection_.GetPhyAddr();
                auto* src = (__ubuf__ float*)partial.GetPhyAddr();
                const uint16_t loops = BasicApiCeilDiv(elements, 64U);
                if (first) {
                    asc_vf_call<MhcPreAccumulateBsMixWithCompensation<true>>(sum, correction, src, elements, loops);
                } else {
                    asc_vf_call<MhcPreAccumulateBsMixWithCompensation<false>>(sum, correction, src, elements, loops);
                }
            } else if (first) {
                MhcPreAccumulateBsPartial<true>(accumLocal_, partial, elements);
            } else {
                MhcPreAccumulateBsPartial<false>(accumLocal_, partial, elements);
            }
        }
        // Both paired AIVs return slots in the Cube's publication order.
        CrossCoreSetFlag<MHC_PRE_CROSS_CORE_SYNC_MODE, PIPE_V>(MHC_PRE_BS_PARTIAL_CONSUMED_FLAG);
        ++partialSequence_;
    }
#else
    __aicore__ inline void AICProcessBasicApi(uint32_t offsetNd, uint32_t partialIndex)
    {
        if (offsetNd + vector_.mnConfig_.singleCoreK > vector_.mnConfig_.k) {
            vector_.mnConfig_.curSingleCoreK = vector_.mnConfig_.k - offsetNd;
        }

        uint32_t currentK = vector_.mnConfig_.curSingleCoreK;
        uint64_t xOffset = vector_.chunkTSize_ * vector_.ND_LENGTH *
                           (vector_.coreIdx_ + (vector_.cubeCount_ % vector_.PARALLEL_NUM) * vector_.coreNum_);
        uint8_t aL1BufferId = vector_.cubeCount_ & 1;
        uint8_t bL1BufferId = mmService_.GetBL1BufferId();

        LocalTensor<float> currentAL1 = aL1_[aL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        LocalTensor<float> currentBL1 = bL1_[bL1BufferId * MHC_PRE_BASIC_API_L1_BUF_OFFSET];
        mmService_.CopyInA1Nd2Nz(vector_.mnConfig_.curSingleCoreM, currentK, vector_.xFloatGm_[xOffset], currentAL1);
        mmService_.CopyInB1Nd2Nz(vector_.mnConfig_.k, currentK, vector_.mnConfig_.n, vector_.phiGm_[offsetNd],
                                 currentBL1);

        bool isFirstKL1 = offsetNd % MHC_PRE_BS_SEQUENTIAL_PARTIAL_K == 0U;
        bool isLastKL1 = IsSequentialPartialEnd(offsetNd, currentK);
        uint64_t mAlign = BasicApiAlign(vector_.mnConfig_.curSingleCoreM, AscendC::BLOCK_CUBE);
        uint64_t nAlign = BasicApiAlign(vector_.mnConfig_.n, AscendC::BLOCK_CUBE);
        // Fill available L0 from the live M/N footprint instead of hard-coding baseK for one shape.
        uint64_t baseK = (256 / AscendC::Std::max(mAlign, nAlign)) * 32;
        mmService_.Process(vector_.mnConfig_.curSingleCoreM, vector_.mnConfig_.n, vector_.mnConfig_.curSingleCoreM,
                           baseK, isFirstKL1, isLastKL1, currentAL1, currentBL1);

        if (isLastKL1) {
            // Emit ordered K=1024 partials. Later partials atomically accumulate into the same compact
            // hMix rows, avoiding a separate vector reduction kernel while retaining the required order.
            if (partialIndex != 0U) {
                AscendC::SetAtomicAdd<float>();
            }
            uint64_t outputOffset = vector_.globalOffsetM_ * vector_.mnConfig_.n;
            mmService_.CopyOut(vector_.hMixGm_[outputOffset], vector_.mnConfig_.n);
            if (offsetNd + currentK >= vector_.matrixInfo_.nD) {
                AscendC::DisableDmaAtomic();
            }
        }
        vector_.cubeCount_++;
    }
#endif

    __aicore__ inline void AIV1ProcessDirect()
#if defined(MHC_PRE_ENABLE_ASCEND950)
    {
        if (vector_.vectorOffset_.singleCoreM == 0) {
            return;
        }

        if (vector_.outFlag_) {
            auto ready = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            const uint32_t mixWidth = vector_.mnConfig_.n;
            const uint64_t row = vector_.globalOffsetM_ + vector_.vectorOffset_.offsetMStart;
            DataCopyExtParams copy{static_cast<uint16_t>(vector_.vectorOffset_.singleCoreM),
                                   mixWidth * static_cast<uint32_t>(sizeof(P)), 0U, 0U, 0U};
            DataCopyPad(vector_.hMixGm_[row * mixWidth], accumLocal_, copy);
        }
        uint64_t localOffset = 0;
        for (uint64_t offsetT = vector_.vectorOffset_.offsetMStart; offsetT < vector_.vectorOffset_.offsetMEnd;
             offsetT += vector_.V1_BASE_T) {
            uint64_t lenT =
                AscendC::Std::min(static_cast<uint64_t>(vector_.V1_BASE_T), vector_.vectorOffset_.offsetMEnd - offsetT);
            vector_.AIV1PostProcessTile(accumLocal_[localOffset * alignedMixWidth_], offsetT, lenT,
                                        static_cast<uint32_t>(localOffset), alignedMixWidth_, xQueueBytes_,
                                        MHC_PRE_BS_OUT_QUEUE_BYTES);
            localOffset += lenT;
        }
        if (vector_.outFlag_) {
            auto done = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(done);
            WaitFlag<HardEvent::MTE3_V>(done);
        }
    }

    __aicore__ inline LocalTensor<P> GetNzTile(const LocalTensor<T>& x)
    {
        if constexpr (IsSameType<T, bfloat16_t>::value) {
            if (stageRows_ == MHC_PRE_BS_EXTENDED_STAGE_ROWS) {
                // X and NZ share a queue slot, which stays live until MTE3 finishes reading NZ.
                using VectorCompute = MhcPreVectorCompute<T, P, RESI_MODE>;
                constexpr uint32_t xElements = MHC_PRE_BS_EXTENDED_STAGE_ROWS * VectorCompute::ND_LENGTH;
                static_assert(xElements * sizeof(T) +
                                  (MHC_PRE_BS_EXTENDED_STAGE_ROWS + 1U) * VectorCompute::ND_LENGTH * sizeof(P) <=
                              MHC_PRE_BS_X_QUEUE_MAX_BYTES);
                return x[xElements].template ReinterpretCast<P>();
            }
        }
        return nzLocal_;
    }

    template <bool HAS_GAMMA>
    __aicore__ inline void V0PrologueL1(uint32_t currentK, uint32_t offsetK)
    {
        __ubuf__ P* gamma = nullptr;
        if constexpr (HAS_GAMMA) {
            vector_.gammaUb_ = vector_.gammaInQueue_.template AllocTensor<P>();
            vector_.DataCopyGamma(currentK, offsetK);
            vector_.gammaUb_ = vector_.gammaInQueue_.template DeQue<P>();
            gamma = (__ubuf__ P*)vector_.gammaUb_.GetPhyAddr();
        }
        // This core owns complete rows and traverses all K. Paired AIVs split only those rows,
        // preserving the per-row operation order when B/S and the number of M blocks change.
        const uint32_t begin = vector_.vectorOffset_.offsetMStart;
        const uint32_t end = vector_.vectorOffset_.offsetMEnd;
        if (begin < end) {
            vector_.xLocal_ = vector_.xInQueue_.template AllocTensor<T>();
            vector_.DataCopyX(MhcPreMin(uint32_t(stageRows_), end - begin), currentK, begin, offsetK);
        }
        for (uint32_t offsetM = begin; offsetM < end; offsetM += stageRows_) {
            uint32_t rows = MhcPreMin(uint32_t(stageRows_), end - offsetM);
            auto x = vector_.xInQueue_.template DeQue<T>();
            if (offsetM + rows < end) {
                vector_.xLocal_ = vector_.xInQueue_.template AllocTensor<T>();
                vector_.DataCopyX(MhcPreMin(uint32_t(stageRows_), end - offsetM - rows), currentK, offsetM + rows,
                                  offsetK);
            }
            auto nd = GetNzTile(x);
            auto rms = vector_.invRmsUb_[offsetM - begin];
            auto correction = (__ubuf__ P*)vector_.invRmsUb_[rmsCorrectionOffset_ + offsetM - begin].GetPhyAddr();
            bool packed = false;
            if constexpr (IsSameType<T, bfloat16_t>::value && !HAS_GAMMA) {
                if (currentK % 128U == 0U) {
                    if (offsetK == 0U) {
                        MhcPreMkVFPackNz<true, true>(nd, rms, x, rows, currentK, nullptr, correction);
                    } else {
                        MhcPreMkVFPackNz<false, true>(nd, rms, x, rows, currentK, nullptr, correction);
                    }
                    packed = true;
                }
            }
            if (!packed) {
                auto ndAddr = (__ubuf__ P*)nd.GetPhyAddr();
                auto rmsAddr = (__ubuf__ P*)rms.GetPhyAddr();
                auto xAddr = (__ubuf__ T*)x.GetPhyAddr();
                if (offsetK == 0U) {
                    vector_.template VFDoV0ProcessXIn<HAS_GAMMA, true, true, true, true>(
                        ndAddr, rmsAddr, xAddr, gamma, rows, currentK, 1U, correction);
                } else {
                    vector_.template VFDoV0ProcessXIn<HAS_GAMMA, false, true, true, true>(
                        ndAddr, rmsAddr, xAddr, gamma, rows, currentK, 1U, correction);
                }
            }
            auto ready = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(ready);
            WaitFlag<HardEvent::V_MTE3>(ready);
            DataCopyParams copy;
            copy.blockCount = currentK / 8U;
            copy.blockLen = rows;
            copy.srcStride = BasicApiAlign(rows, 8U) + 1U - rows;
            copy.dstStride = BasicApiAlign(vector_.curSingleT_, AscendC::BLOCK_CUBE) - rows;
            BasicApiCopyToL1(nd, aL1_[(vector_.vectorCount_ % 2U) * MHC_PRE_BASIC_API_L1_BUF_OFFSET + offsetM * 8U],
                             copy);
            auto free = static_cast<event_t>(vector_.pipe_->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(free);
            WaitFlag<HardEvent::MTE3_V>(free);
            vector_.xInQueue_.FreeTensor(x);
        }
        if constexpr (HAS_GAMMA) {
            vector_.gammaInQueue_.FreeTensor(vector_.gammaUb_);
        }
    }

    TBuf<TPosition::VECCALC> partialBuffer_;
    TBuf<TPosition::VECCALC> accumBuffer_;
    LocalTensor<P> partialLocal_;
    LocalTensor<P> accumLocal_;
    TBuf<TPosition::VECCALC> nzBuffer_;
    LocalTensor<P> nzLocal_;
#else
    {
        if (vector_.vectorOffset_.singleCoreM == 0) {
            return;
        }

        uint64_t localOffset = 0;
        for (uint64_t offsetT = vector_.vectorOffset_.offsetMStart; offsetT < vector_.vectorOffset_.offsetMEnd;
             offsetT += vector_.V1_BASE_T) {
            uint64_t lenT =
                AscendC::Std::min(static_cast<uint64_t>(vector_.V1_BASE_T), vector_.vectorOffset_.offsetMEnd - offsetT);
            uint64_t hMixOffset = (vector_.globalOffsetM_ + offsetT) * vector_.mnConfig_.n;
            vector_.HMixCopyIn(hMixOffset, lenT);
            LocalTensor<P> hMixLocal = vector_.xInQueue_.template DeQue<P>();
            vector_.AIV1PostProcessTile(hMixLocal, offsetT, lenT, static_cast<uint32_t>(localOffset),
                                        vector_.mnConfig_.n);
            vector_.xInQueue_.FreeTensor(hMixLocal);
            localOffset += lenT;
        }
    }
#endif

    MhcPreVectorCompute<T, P, RESI_MODE> vector_;
    MhcPreCubeCompute mmService_;
    TBuf<TPosition::A1> l1Buffer_;
    LocalTensor<float> aL1_;
    LocalTensor<float> bL1_;
};

} // namespace MhcPre

#endif // MHC_PRE_SPLIT_BS_H
