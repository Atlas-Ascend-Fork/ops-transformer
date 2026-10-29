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

namespace UnpermuteQkv {
constexpr int MAX_UB = 180 * 1024;
constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue

template <typename inputDType>
class UnpermuteQkvKernel {
public:
    __aicore__ UnpermuteQkvKernel(const UnpermuteQkvTilingData& tiling, TPipe* pipe)
        : tiling_(tiling),
          pipe_(pipe)
    {}

    __aicore__ void Process(GM_ADDR input, GM_ADDR q, GM_ADDR k, GM_ADDR v)
    {
        // Skip processing if not on vector core
        if (g_coreType == AIC)
            return;

        // Initialize global tensors
        inputGm_.SetGlobalBuffer((__gm__ inputDType*)input);
        qGm_.SetGlobalBuffer((__gm__ inputDType*)q);
        kGm_.SetGlobalBuffer((__gm__ inputDType*)k);
        vGm_.SetGlobalBuffer((__gm__ inputDType*)v);

        // Extract tiling parameters
        const uint32_t split_size_ = tiling_.splitSize;
        const uint32_t s_c_ = tiling_.sC;
        const uint32_t b_ = tiling_.b;
        const uint32_t kv_group_ = tiling_.kvGroup;
        const uint32_t n_k_ = tiling_.nK;
        const uint32_t split_n_k_ = tiling_.splitNK; // n_k_ / split_size_
        const uint32_t h_ = tiling_.h;

        // Calculate block-related parameters
        const uint32_t blockIdx_ = GetBlockIdx();
        const uint32_t blockNum_ = GetBlockNum();
        const uint32_t blockInputSize_ = n_k_ * (kv_group_ + 2) * h_;

        int blockBaseInputNum_ = MAX_UB / BUFFER_NUM / blockInputSize_ / sizeof(inputDType);

        int blockSingleInputNum_ = s_c_ * b_ / blockNum_;

        const int tailInputCoreNum = s_c_ * b_ % blockNum_;
        blockSingleInputNum_ += blockIdx_ < tailInputCoreNum;

        const int loopInput_ = (blockSingleInputNum_ + blockBaseInputNum_ - 1) / blockBaseInputNum_;
        const int offsetBlockInputNums =
            blockSingleInputNum_ * blockIdx_ + (blockIdx_ < tailInputCoreNum ? 0 : tailInputCoreNum);
        const int tailBaseInputNum_ = blockSingleInputNum_ % blockBaseInputNum_;

        const uint32_t offsetInput = offsetBlockInputNums * (kv_group_ + 2) * split_n_k_ * h_;

        // Initialize pipe and allocate UB space
        pipe_->InitBuffer(Queue, BUFFER_NUM, BUFFER_NUM * blockBaseInputNum_ * blockInputSize_ * sizeof(inputDType));

        DataCopyPadParams dataCopyPadParams = {false, 0, 0, 0};

        for (int loopIndex = 0; loopIndex < loopInput_; loopIndex++) // q
        {
            int currBaseInputNum_ =
                (loopIndex == loopInput_ - 1 && tailBaseInputNum_ != 0) ? tailBaseInputNum_ : blockBaseInputNum_;
            uint32_t cur_loop_offset_input =
                offsetInput + blockBaseInputNum_ * loopIndex * (kv_group_ + 2) * split_n_k_ * h_;

            // gm:[split_size, s_c*b(读其中currBaseInputNum_), n_k * kv_group /
            // split_size + n_k / split_size + n_k / split_size, h]
            // -> ub: (s_c*b = currBaseInputNum_, (split_size, n_k * kv_group /
            // split_size) + (split_size, n_k / split_size) + (split_size, n_k /
            // split_size), h) gm连续读 ub跨步写
            DataCopyParams dataCopyInParams_Q;
            dataCopyInParams_Q.blockCount = 1;
            dataCopyInParams_Q.blockLen = n_k_ * kv_group_ / split_size_ * h_ * sizeof(inputDType);
            dataCopyInParams_Q.srcStride = 0;
            dataCopyInParams_Q.dstStride = 0;

            DataCopyParams dataCopyInParams_kv;
            dataCopyInParams_kv.blockCount = 2;
            dataCopyInParams_kv.blockLen = n_k_ / split_size_ * h_ * sizeof(inputDType);
            dataCopyInParams_kv.srcStride = 0;
            dataCopyInParams_kv.dstStride = (n_k_ - (n_k_ / split_size_)) * h_ * sizeof(inputDType) / 32;

            LocalTensor<inputDType> dstUB_ = Queue.AllocTensor<inputDType>();

            for (int curr_split = 0; curr_split < split_size_; curr_split++) { // q
                int cur_loop_offset_input_c =
                    cur_loop_offset_input + curr_split * s_c_ * b_ * (kv_group_ + 2) * n_k_ / split_size_ * h_;
                for (int sc_b = 0; sc_b < currBaseInputNum_; sc_b++) {
                    int cur_loop_offset_input_s =
                        cur_loop_offset_input_c + sc_b * (kv_group_ + 2) * n_k_ / split_size_ * h_;

                    DataCopyPad(
                        dstUB_[curr_split * n_k_ * kv_group_ / split_size_ * h_ + sc_b * (kv_group_ + 2) * n_k_ * h_],
                        inputGm_[cur_loop_offset_input_s], dataCopyInParams_Q, dataCopyPadParams);
                }
            }

            int cur_loop_offset_input_kv = cur_loop_offset_input + n_k_ * kv_group_ / split_size_ * h_;
            for (int curr_split = 0; curr_split < split_size_; curr_split++) { // k v
                int cur_loop_offset_input_c =
                    cur_loop_offset_input_kv + curr_split * s_c_ * b_ * (kv_group_ + 2) * n_k_ / split_size_ * h_;
                for (int sc_b = 0; sc_b < currBaseInputNum_; sc_b++) {
                    int cur_loop_offset_input_s =
                        cur_loop_offset_input_c + sc_b * (kv_group_ + 2) * n_k_ / split_size_ * h_;
                    DataCopyPad(dstUB_[curr_split * n_k_ / split_size_ * h_ + sc_b * (kv_group_ + 2) * n_k_ * h_ +
                                       kv_group_ * n_k_ * h_],
                                inputGm_[cur_loop_offset_input_s], dataCopyInParams_kv, dataCopyPadParams);
                }
            }

            Queue.EnQue(dstUB_);

            uint32_t cur_loop_offset_q =
                (offsetBlockInputNums + blockBaseInputNum_ * loopIndex) * n_k_ * kv_group_ * h_;
            uint32_t cur_loop_offset_kv = (offsetBlockInputNums + blockBaseInputNum_ * loopIndex) * n_k_ * h_;

            // ub: (s_c*b = currBaseInputNum_, (n_k * kv_group) + n_k + n_k, h)
            // gm: q [s_c*b (写其中currBaseInputNum_), n_k * kv_group, h]
            //     k [s_c*b (写其中currBaseInputNum_), n_k, h]
            //     v [s_c*b (写其中currBaseInputNum_), n_k, h]
            DataCopyParams dataCopyOutParams_q;
            dataCopyOutParams_q.blockCount = 1;
            dataCopyOutParams_q.blockLen = n_k_ * kv_group_ * h_ * sizeof(inputDType);
            dataCopyOutParams_q.srcStride = 0;
            dataCopyOutParams_q.dstStride = 0;

            DataCopyParams dataCopyOutParams_kv;
            dataCopyOutParams_kv.blockCount = 1;
            dataCopyOutParams_kv.blockLen = n_k_ * h_ * sizeof(inputDType);
            dataCopyOutParams_kv.srcStride = 0;
            dataCopyOutParams_kv.dstStride = 0;

            LocalTensor<inputDType> outLocal = Queue.DeQue<inputDType>();

            for (int sc_b = 0; sc_b < currBaseInputNum_; sc_b++) { // q
                DataCopyPad(qGm_[cur_loop_offset_q + sc_b * n_k_ * kv_group_ * h_],
                            outLocal[sc_b * n_k_ * (kv_group_ + 2) * h_], dataCopyOutParams_q);
            }

            for (int sc_b = 0; sc_b < currBaseInputNum_; sc_b++) { // k
                DataCopyPad(kGm_[cur_loop_offset_kv + sc_b * n_k_ * h_],
                            outLocal[sc_b * n_k_ * (kv_group_ + 2) * h_ + (kv_group_ * n_k_) * h_],
                            dataCopyOutParams_kv);
            }

            for (int sc_b = 0; sc_b < currBaseInputNum_; sc_b++) { // v
                DataCopyPad(vGm_[cur_loop_offset_kv + sc_b * n_k_ * h_],
                            outLocal[sc_b * n_k_ * (kv_group_ + 2) * h_ + (kv_group_ + 1) * n_k_ * h_],
                            dataCopyOutParams_kv);
            }

            Queue.FreeTensor(outLocal);
        }
    }

private:
    TPipe* pipe_;

    const UnpermuteQkvTilingData& tiling_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> Queue;
    GlobalTensor<inputDType> inputGm_;
    GlobalTensor<inputDType> qGm_;
    GlobalTensor<inputDType> kGm_;
    GlobalTensor<inputDType> vGm_;
};
} // namespace UnpermuteQkv
