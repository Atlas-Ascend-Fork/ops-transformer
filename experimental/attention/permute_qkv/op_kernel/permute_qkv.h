/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

using namespace AscendC;

namespace PermuteQkv {
constexpr int MAX_UB = 180 * 1024;
constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue

template <typename inputDType>
class PermuteQkvKernel {
public:
    __aicore__ PermuteQkvKernel(const PermuteQkvTilingData& tiling, TPipe* pipe)
        : tiling_(tiling),
          pipe_(pipe)
    {}

    __aicore__ void Process(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR output)
    {
        // only vector can go through
        if (g_coreType == AIC) {
            return;
        }

        qGm_.SetGlobalBuffer((__gm__ inputDType*)q);
        kGm_.SetGlobalBuffer((__gm__ inputDType*)k);
        vGm_.SetGlobalBuffer((__gm__ inputDType*)v);
        outputGm_.SetGlobalBuffer((__gm__ inputDType*)output);

        // get block id
        const uint32_t blockIdx_ = GetBlockIdx();
        const uint32_t blockNum_ = GetBlockNum();

        const uint32_t split_size_ = tiling_.splitSize;
        const uint32_t s_c_ = tiling_.sC;
        const uint32_t b_ = tiling_.b;
        const uint32_t sc_b_ = s_c_ * b_;
        const uint32_t kv_group_ = tiling_.kvGroup;
        const uint32_t n_k_ = tiling_.nK;
        const uint32_t split_n_k_ = tiling_.splitNK;
        const uint32_t h_ = tiling_.h;

        int32_t THE_BASE_BLOCK_COUNT = 1; // tensor num for each queue
        if (sc_b_ % 2 == 0)
            THE_BASE_BLOCK_COUNT = 2;

        const uint32_t blockQSize_ = kv_group_ * n_k_ * h_;
        const uint32_t blockKvSize_ = n_k_ * h_;

        const uint32_t blockBaseQNum_ =
            MAX_UB / BUFFER_NUM / (blockQSize_ * THE_BASE_BLOCK_COUNT) / sizeof(inputDType) * THE_BASE_BLOCK_COUNT;
        const uint32_t blockBaseKvNum_ =
            MAX_UB / BUFFER_NUM / (blockKvSize_ * THE_BASE_BLOCK_COUNT) / sizeof(inputDType) * THE_BASE_BLOCK_COUNT;

        const uint32_t blockSingleNum_ =
            sc_b_ / (blockNum_ * THE_BASE_BLOCK_COUNT) * THE_BASE_BLOCK_COUNT +
            THE_BASE_BLOCK_COUNT * (blockIdx_ < (sc_b_ % (blockNum_ * THE_BASE_BLOCK_COUNT)) / THE_BASE_BLOCK_COUNT);

        const uint32_t loopQ_ = (blockSingleNum_ + blockBaseQNum_ - 1) / blockBaseQNum_;
        const uint32_t loopKv_ = (blockSingleNum_ + blockBaseKvNum_ - 1) / blockBaseKvNum_;

        const uint32_t offsetBlockQNums =
            blockSingleNum_ * blockIdx_ +
            (blockIdx_ < (sc_b_ % (blockNum_ * THE_BASE_BLOCK_COUNT)) / THE_BASE_BLOCK_COUNT ?
                 0 :
                 THE_BASE_BLOCK_COUNT * (sc_b_ % (blockNum_ * THE_BASE_BLOCK_COUNT)) / THE_BASE_BLOCK_COUNT);
        const uint32_t offsetBlockKvNums = offsetBlockQNums;

        const uint32_t tailBaseQNum_ = blockSingleNum_ % blockBaseQNum_;
        const uint32_t tailBaseKvNum_ = blockSingleNum_ % blockBaseKvNum_;

        const uint32_t offsetQ = offsetBlockQNums * blockQSize_;
        const uint32_t offsetKv = offsetBlockKvNums * blockKvSize_;

        const uint32_t maxBufferBlockNum = blockBaseQNum_ * blockQSize_ > blockBaseKvNum_ * blockKvSize_ ?
                                               blockBaseQNum_ * blockQSize_ :
                                               blockBaseKvNum_ * blockKvSize_;

        pipe_->InitBuffer(Queue, BUFFER_NUM, BUFFER_NUM * maxBufferBlockNum * sizeof(inputDType));

        DataCopyPadParams dataCopyPadParams = {false, 0, 0, 0};

        for (int loopIndex = 0; loopIndex < loopQ_; loopIndex++) // q
        {
            uint32_t currBaseQNum_ = (loopIndex == loopQ_ - 1 && tailBaseQNum_ != 0) ? tailBaseQNum_ : blockBaseQNum_;
            uint32_t cur_loop_offset_q = offsetQ + blockBaseQNum_ * loopIndex * blockQSize_;

            // b(读其中currBaseQNum_ 大小), kv_group * n_k, h)
            // -> ub: (kv_group * n_k, s*b = currBaseQNum_, h)
            // gm连续读 ub跨步写
            DataCopyParams dataCopyInParams_Q;
            dataCopyInParams_Q.blockCount = kv_group_ * n_k_;
            dataCopyInParams_Q.blockLen = h_ * sizeof(inputDType);
            dataCopyInParams_Q.srcStride = 0;
            dataCopyInParams_Q.dstStride = (currBaseQNum_ - 1) * h_ * sizeof(inputDType) / 32;

            LocalTensor<inputDType> dstUB_ = Queue.AllocTensor<inputDType>();
            for (int sc_b = 0; sc_b < currBaseQNum_; sc_b++) {
                DataCopyPad(dstUB_[sc_b * h_], qGm_[cur_loop_offset_q + sc_b * kv_group_ * n_k_ * h_],
                            dataCopyInParams_Q, dataCopyPadParams);
            }
            Queue.EnQue(dstUB_);

            uint32_t cur_loop_offset_output = offsetBlockQNums * h_ + blockBaseQNum_ * loopIndex * h_;
            // ub: (split_idx, kv_group/split_idx, n_k, s_c*b = currBaseQNum_, h)
            // gm: [split_idx, (kv_group/split_idx, n_k) + split_n_k +split_n_k, s_c*b(写其中currBaseQNum_大小), h)
            DataCopyParams dataCopyOutParams;
            dataCopyOutParams.blockCount = 1;
            dataCopyOutParams.blockLen = currBaseQNum_ * h_ * sizeof(inputDType);
            dataCopyOutParams.srcStride = 0;
            dataCopyOutParams.dstStride = 0;

            LocalTensor<inputDType> outLocal = Queue.DeQue<inputDType>();
            for (int split_idx = 0; split_idx < split_size_; split_idx++) {
                uint32_t cur_loop_offset_output_s =
                    cur_loop_offset_output + split_idx * (kv_group_ * split_n_k_ + split_n_k_ * 2) * sc_b_ * h_;
                for (int head = 0; head < kv_group_ * split_n_k_; head++) {
                    uint32_t cur_loop_offset_output_h = cur_loop_offset_output_s + head * sc_b_ * h_;
                    DataCopyPad(
                        outputGm_[cur_loop_offset_output_h],
                        outLocal[split_idx * kv_group_ * split_n_k_ * currBaseQNum_ * h_ + head * currBaseQNum_ * h_],
                        dataCopyOutParams);
                }
            }
            Queue.FreeTensor(outLocal);
        }

        for (int loopIndex = 0; loopIndex < loopKv_; loopIndex++) // k
        {
            int currBaseKvNum_ = (loopIndex == loopKv_ - 1 && tailBaseKvNum_ != 0) ? tailBaseKvNum_ : blockBaseKvNum_;
            uint32_t cur_loop_offset_Kv = offsetKv + blockBaseKvNum_ * loopIndex * blockKvSize_;

            DataCopyParams dataCopyInParams_Kv;
            dataCopyInParams_Kv.blockCount = n_k_;
            dataCopyInParams_Kv.blockLen = h_ * sizeof(inputDType);
            dataCopyInParams_Kv.srcStride = 0;
            dataCopyInParams_Kv.dstStride = (currBaseKvNum_ - 1) * h_ * sizeof(inputDType) / 32;

            LocalTensor<inputDType> dstUB_ = Queue.AllocTensor<inputDType>();
            for (int sc_b = 0; sc_b < currBaseKvNum_; sc_b++) {
                DataCopyPad(dstUB_[sc_b * h_], kGm_[cur_loop_offset_Kv + sc_b * n_k_ * h_], dataCopyInParams_Kv,
                            dataCopyPadParams);
            }
            Queue.EnQue(dstUB_);

            uint32_t cur_loop_offset_output =
                offsetBlockKvNums * h_ + blockBaseKvNum_ * loopIndex * h_ + kv_group_ * split_n_k_ * sc_b_ * h_;

            DataCopyParams dataCopyOutParams;
            dataCopyOutParams.blockCount = 1;
            dataCopyOutParams.blockLen = currBaseKvNum_ * h_ * sizeof(inputDType);
            dataCopyOutParams.srcStride = 0;
            dataCopyOutParams.dstStride = 0;

            LocalTensor<inputDType> outLocal = Queue.DeQue<inputDType>();
            for (int split_idx = 0; split_idx < split_size_; split_idx++) {
                int cur_loop_offset_output_s =
                    cur_loop_offset_output + split_idx * (kv_group_ + 2) * split_n_k_ * sc_b_ * h_;
                for (int split_nk_idx = 0; split_nk_idx < split_n_k_; split_nk_idx++) {
                    int cur_loop_offset_output_n = cur_loop_offset_output_s + split_nk_idx * sc_b_ * h_;

                    DataCopyPad(
                        outputGm_[cur_loop_offset_output_n],
                        outLocal[split_idx * split_n_k_ * currBaseKvNum_ * h_ + split_nk_idx * currBaseKvNum_ * h_],
                        dataCopyOutParams);
                }
            }
            Queue.FreeTensor(outLocal);
        }

        for (int loopIndex = 0; loopIndex < loopKv_; loopIndex++) // v
        {
            int currBaseKvNum_ = (loopIndex == loopKv_ - 1 && tailBaseKvNum_ != 0) ? tailBaseKvNum_ : blockBaseKvNum_;
            uint32_t cur_loop_offset_Kv = offsetKv + blockBaseKvNum_ * loopIndex * blockKvSize_;

            DataCopyParams dataCopyInParams_Kv;
            dataCopyInParams_Kv.blockCount = n_k_;
            dataCopyInParams_Kv.blockLen = h_ * sizeof(inputDType);
            dataCopyInParams_Kv.srcStride = 0;
            dataCopyInParams_Kv.dstStride = (currBaseKvNum_ - 1) * h_ * sizeof(inputDType) / 32;

            LocalTensor<inputDType> dstUB_ = Queue.AllocTensor<inputDType>();
            for (int sc_b = 0; sc_b < currBaseKvNum_; sc_b++) {
                DataCopyPad(dstUB_[sc_b * h_], vGm_[cur_loop_offset_Kv + sc_b * n_k_ * h_], dataCopyInParams_Kv,
                            dataCopyPadParams);
            }
            Queue.EnQue(dstUB_);

            uint32_t cur_loop_offset_output =
                offsetBlockKvNums * h_ + blockBaseKvNum_ * loopIndex * h_ + (kv_group_ + 1) * split_n_k_ * sc_b_ * h_;

            DataCopyParams dataCopyOutParams;
            dataCopyOutParams.blockCount = 1;
            dataCopyOutParams.blockLen = currBaseKvNum_ * h_ * sizeof(inputDType);
            dataCopyOutParams.srcStride = 0;
            dataCopyOutParams.dstStride = 0;

            LocalTensor<inputDType> outLocal = Queue.DeQue<inputDType>();
            for (int split_idx = 0; split_idx < split_size_; split_idx++) {
                int cur_loop_offset_output_s =
                    cur_loop_offset_output + split_idx * (kv_group_ + 2) * split_n_k_ * sc_b_ * h_;
                for (int split_nk_idx = 0; split_nk_idx < split_n_k_; split_nk_idx++) {
                    int cur_loop_offset_output_n = cur_loop_offset_output_s + split_nk_idx * sc_b_ * h_;
                    DataCopyPad(
                        outputGm_[cur_loop_offset_output_n],
                        outLocal[split_idx * split_n_k_ * currBaseKvNum_ * h_ + split_nk_idx * currBaseKvNum_ * h_],
                        dataCopyOutParams);
                }
            }
            Queue.FreeTensor(outLocal);
        }
    }

private:
    TPipe* pipe_;

    const PermuteQkvTilingData& tiling_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> Queue;
    GlobalTensor<inputDType> qGm_;
    GlobalTensor<inputDType> kGm_;
    GlobalTensor<inputDType> vGm_;
    GlobalTensor<inputDType> outputGm_;
};
} // namespace PermuteQkv
