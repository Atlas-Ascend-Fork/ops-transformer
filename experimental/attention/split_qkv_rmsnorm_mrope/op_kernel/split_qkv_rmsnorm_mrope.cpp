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
#include "split_qkv_rmsnorm_mrope_tiling_data.h"

using namespace AscendC;

namespace {
constexpr uint16_t BF16_HEAD_BLOCKS = 16;
constexpr uint16_t INTERLEAVED_HEAD_GAP_BLOCKS = 16;
} // namespace

template <HardEvent event>
__aicore__ inline void Sync()
{
    SetFlag<event>(EVENT_ID0);
    WaitFlag<event>(EVENT_ID0);
}

class SplitQkvRmsnormMrope {
public:
    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR qw, GM_ADDR qb, GM_ADDR kw, GM_ADDR kb, GM_ADDR cs, GM_ADDR qo,
                                GM_ADDR ko, GM_ADDR vo, GM_ADDR go, uint32_t tokens, uint32_t qh, uint32_t kh,
                                uint32_t blocks, float eps, bool qBias, bool kBias, bool gate, bool interleaved)
    {
        tokens_ = tokens;
        qh_ = qh;
        kh_ = kh;
        blocks_ = blocks;
        eps_ = eps;
        bias_[0] = qBias;
        bias_[1] = kBias;
        gate_ = gate;
        interleaved_ = interleaved;
        qs_ = qh * 256;
        ks_ = kh * 256;
        width_ = qs_ + (gate ? qs_ : 0) + 2 * ks_;

        input_.SetGlobalBuffer((__gm__ bfloat16_t*)qkv);
        weights_[0].SetGlobalBuffer((__gm__ bfloat16_t*)qw);
        weights_[1].SetGlobalBuffer((__gm__ bfloat16_t*)kw);
        if (qBias) {
            biases_[0].SetGlobalBuffer((__gm__ bfloat16_t*)qb);
        }
        if (kBias) {
            biases_[1].SetGlobalBuffer((__gm__ bfloat16_t*)kb);
        }
        rope_.SetGlobalBuffer((__gm__ bfloat16_t*)cs);
        outputs_[0].SetGlobalBuffer((__gm__ bfloat16_t*)qo);
        outputs_[1].SetGlobalBuffer((__gm__ bfloat16_t*)ko);
        passthrough_[0].SetGlobalBuffer((__gm__ bfloat16_t*)go);
        passthrough_[1].SetGlobalBuffer((__gm__ bfloat16_t*)vo);

        pipe_.InitBuffer(bfBuf_, 5120 * sizeof(bfloat16_t));
        pipe_.InitBuffer(outBfBuf_, 5120 * sizeof(bfloat16_t));
        pipe_.InitBuffer(passBuf_, 5120 * sizeof(bfloat16_t));
        pipe_.InitBuffer(rowBuf_, 10240 * sizeof(bfloat16_t));
        pipe_.InitBuffer(ropeQueue_, 2, 192 * sizeof(bfloat16_t));
        pipe_.InitBuffer(fpBuf_, 13056 * sizeof(float));
        bf_ = bfBuf_.Get<bfloat16_t>();
        outBf_ = outBfBuf_.Get<bfloat16_t>();
        passBf_ = passBuf_.Get<bfloat16_t>();
        rowBf_ = rowBuf_.Get<bfloat16_t>();
        auto fp = fpBuf_.Get<float>();
        x_ = fp;
        square_ = fp[5120];
        temp_ = fp[10240];
        weight_[0] = fp[11520];
        weight_[1] = fp[11776];
        biasFp_[0] = fp[12032];
        biasFp_[1] = fp[12288];
        cos_ = fp[12544];
        sin_ = fp[12608];
        sum_ = fp[12672];
        axes_ = fp[12736];
        cosOffset_ = fp[12928].ReinterpretCast<uint32_t>();
        sinOffset_ = fp[12992].ReinterpretCast<uint32_t>();

        if (interleaved_) {
            BuildInterleavedRopeOffsets();
        }
        LoadParameters();
    }

    __aicore__ inline void Process()
    {
        uint32_t block = GetBlockIdx();
        uint32_t count = tokens_ / blocks_ + (block < tokens_ % blocks_);
        uint32_t start = block * (tokens_ / blocks_) + (block < tokens_ % blocks_ ? block : tokens_ % blocks_);
        for (uint32_t token = start; token < start + count; ++token) {
            ProcessToken(token);
        }
    }

private:
    __aicore__ inline void BuildInterleavedRopeOffsets()
    {
        auto index = cosOffset_.ReinterpretCast<float>();
        auto axis = sinOffset_.ReinterpretCast<float>();
        ArithProgression(index, 0.0f, 1.0f, 32);
        Muls(axis, index, 1.0f / 3.0f, 32);
        PipeBarrier<PIPE_V>();
        Floor(axis, axis, 32);
        PipeBarrier<PIPE_V>();
        Muls(axis, axis, -768.0f, 32);
        Muls(index, index, 260.0f, 32);
        PipeBarrier<PIPE_V>();
        Add(index, index, axis, 32);
        PipeBarrier<PIPE_V>();
        Cast(cosOffset_.ReinterpretCast<int32_t>(), index, RoundMode::CAST_RINT, 32);
        PipeBarrier<PIPE_V>();
        Adds(cosOffset_[32].ReinterpretCast<int32_t>(), cosOffset_.ReinterpretCast<int32_t>(), 0, 32);
        Adds(sinOffset_.ReinterpretCast<int32_t>(), cosOffset_.ReinterpretCast<int32_t>(), static_cast<int32_t>(128),
             64);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadParameters()
    {
        DataCopy(bf_, weights_[0], 256);
        DataCopy(bf_[256], weights_[1], 256);
        if (bias_[0]) {
            DataCopy(bf_[512], biases_[0], 256);
        }
        if (bias_[1]) {
            DataCopy(bf_[768], biases_[1], 256);
        }
        Sync<HardEvent::MTE2_V>();
        Cast(weight_[0], bf_, RoundMode::CAST_NONE, 256);
        Cast(weight_[1], bf_[256], RoundMode::CAST_NONE, 256);
        if (bias_[0]) {
            Cast(biasFp_[0], bf_[512], RoundMode::CAST_NONE, 256);
        }
        if (bias_[1]) {
            Cast(biasFp_[1], bf_[768], RoundMode::CAST_NONE, 256);
        }
        Sync<HardEvent::V_S>();
    }

    __aicore__ inline void LoadRope(uint32_t token)
    {
        LocalTensor<bfloat16_t> ropeBf = ropeQueue_.AllocTensor<bfloat16_t>();
        for (uint32_t axis = 0; axis < 3; ++axis) {
            DataCopy(ropeBf[axis * 64], rope_[(static_cast<uint64_t>(axis) * tokens_ + token) * 64], 64);
        }
        ropeQueue_.EnQue(ropeBf);
        ropeBf = ropeQueue_.DeQue<bfloat16_t>();
        Cast(axes_, ropeBf, RoundMode::CAST_NONE, 192);
        ropeQueue_.FreeTensor(ropeBf);
        if (!interleaved_) {
            Sync<HardEvent::V_S>();
            for (uint32_t j = 0; j < 32; ++j) {
                uint32_t axis = j < 11 ? 0 : (j < 22 ? 1 : 2);
                uint32_t c = (axis * 64 + j) * sizeof(float);
                cosOffset_.SetValue(j, c);
                cosOffset_.SetValue(j + 32, c);
                uint32_t s = c + 32 * sizeof(float);
                sinOffset_.SetValue(j, s);
                sinOffset_.SetValue(j + 32, s);
            }
            Sync<HardEvent::S_V>();
        }
        Gather(cos_, axes_, cosOffset_, 0, 64);
        Gather(sin_, axes_, sinOffset_, 0, 64);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyPassthrough(uint64_t row, uint32_t token)
    {
        if (passPending_) {
            Sync<HardEvent::MTE3_MTE2>();
        }
        if (gate_) {
            // DataCopy srcStride is the gap after blockLen. 16 + 16 blocks = 1024 B between Q head starts.
            DataCopyParams gather{static_cast<uint16_t>(qh_), BF16_HEAD_BLOCKS, INTERLEAVED_HEAD_GAP_BLOCKS, 0};
            DataCopy(passBf_, input_[row + 256], gather);
        }

        uint64_t kbase = row + qs_ + (gate_ ? qs_ : 0);
        DataCopy(passBf_[qs_], input_[kbase + ks_], ks_);
        Sync<HardEvent::MTE2_MTE3>();
        if (gate_) {
            DataCopy(passthrough_[0][static_cast<uint64_t>(token) * qs_], passBf_, qs_);
        }
        DataCopy(passthrough_[1][static_cast<uint64_t>(token) * ks_], passBf_[qs_], ks_);
        if (tokens_ <= blocks_) {
            passPending_ = true;
        } else {
            Sync<HardEvent::MTE3_MTE2>();
        }
    }

    __aicore__ inline void LoadAllHeads(uint64_t row)
    {
        if (gate_) {
            DataCopyParams gather{static_cast<uint16_t>(qh_), BF16_HEAD_BLOCKS, INTERLEAVED_HEAD_GAP_BLOCKS, 0};
            DataCopy(bf_, input_[row], gather);
        } else {
            DataCopy(bf_, input_[row], qs_);
        }
        uint64_t kbase = row + qs_ + (gate_ ? qs_ : 0);
        DataCopy(bf_[qs_], input_[kbase], ks_);
        Sync<HardEvent::MTE2_V>();
        Cast(x_, bf_, RoundMode::CAST_NONE, qs_ + ks_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadFixedRow(uint64_t row, uint32_t token)
    {
        DataCopy(rowBf_, input_[row], width_);
        Sync<HardEvent::MTE2_V>();
        UnaryRepeatParams extract{1, 1, 32, 32};
        for (uint32_t part = 0; part < 4; ++part) {
            Cast(x_[part * 64], rowBf_[part * 64], RoundMode::CAST_NONE, 64, static_cast<uint8_t>(qh_), extract);
        }
        Cast(x_[qs_], rowBf_[2 * qs_], RoundMode::CAST_NONE, ks_);
        Sync<HardEvent::V_MTE3>();
        DataCopyParams gateCopy{static_cast<uint16_t>(qh_), BF16_HEAD_BLOCKS, INTERLEAVED_HEAD_GAP_BLOCKS, 0};
        DataCopy(passthrough_[0][static_cast<uint64_t>(token) * qs_], rowBf_[256], gateCopy);
        DataCopy(passthrough_[1][static_cast<uint64_t>(token) * ks_], rowBf_[2 * qs_ + ks_], ks_);
        Sync<HardEvent::MTE3_MTE2>();
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ApplyAffine(uint32_t offset, uint32_t heads, uint32_t kind)
    {
        BinaryRepeatParams affine{1, 1, 1, 32, 32, 0};
        for (uint32_t part = 0; part < 4; ++part) {
            Mul(x_[offset + part * 64], x_[offset + part * 64], weight_[kind][part * 64], 64, heads, affine);
        }
        if (bias_[kind]) {
            PipeBarrier<PIPE_V>();
            for (uint32_t part = 0; part < 4; ++part) {
                Add(x_[offset + part * 64], x_[offset + part * 64], biasFp_[kind][part * 64], 64, heads, affine);
            }
        }
    }

    __aicore__ inline void RmsNormAll()
    {
        uint32_t heads = qh_ + kh_;
        Mul(square_, x_, x_, heads * 256);
        PipeBarrier<PIPE_V>();

        BinaryRepeatParams pairA{1, 1, 1, 8, 32, 32};
        Add(temp_, square_, square_[128], 64, heads, pairA);
        BinaryRepeatParams pairB{1, 1, 1, 32, 32, 32};
        Add(square_[64], square_[64], square_[192], 64, heads, pairB);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams merge{1, 1, 1, 8, 8, 32};
        Add(temp_, temp_, square_[64], 64, heads, merge);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(sum_, temp_, 64, heads, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Muls(sum_, sum_, 1.0f / 256.0f, heads);
        PipeBarrier<PIPE_V>();
        Adds(sum_, sum_, eps_, heads);
        PipeBarrier<PIPE_V>();
        Sqrt(sum_, sum_, heads);
        PipeBarrier<PIPE_V>();
        Duplicate(temp_, 1.0f, heads);
        PipeBarrier<PIPE_V>();
        Div(sum_, temp_, sum_, heads);
        PipeBarrier<PIPE_V>();
        Brcb(temp_, sum_, static_cast<uint8_t>((heads + 7) / 8), {1, 8});
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams normalize{1, 1, 0, 32, 32, 1};
        for (uint32_t part = 0; part < 4; ++part) {
            Mul(x_[part * 64], x_[part * 64], temp_, 64, heads, normalize);
        }
        PipeBarrier<PIPE_V>();
        ApplyAffine(0, qh_, 0);
        ApplyAffine(qs_, kh_, 1);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Rope(uint32_t heads)
    {
        UnaryRepeatParams rotate{1, 1, 8, 32};
        Muls(temp_, x_[32], -1.0f, 32, static_cast<uint8_t>(heads), rotate);
        Adds(temp_[32], x_, 0.0f, 32, static_cast<uint8_t>(heads), rotate);
        PipeBarrier<PIPE_V>();

        BinaryRepeatParams xTrig{1, 1, 1, 32, 32, 0};
        BinaryRepeatParams tempTrig{1, 1, 1, 8, 8, 0};
        Mul(x_, x_, cos_, 64, heads, xTrig);
        Mul(temp_, temp_, sin_, 64, heads, tempTrig);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams combine{1, 1, 1, 32, 32, 8};
        Add(x_, x_, temp_, 64, heads, combine);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreAllHeads(uint32_t token)
    {
        if (outPending_) {
            Sync<HardEvent::MTE3_V>();
        }
        uint32_t heads = qh_ + kh_;
        UnaryRepeatParams castParams{1, 1, 4, 8};
        Cast(outBf_, x_, RoundMode::CAST_RINT, 64, static_cast<uint8_t>(heads * 4), castParams);
        Sync<HardEvent::V_MTE3>();
        DataCopy(outputs_[0][static_cast<uint64_t>(token) * qs_], outBf_, qs_);
        DataCopy(outputs_[1][static_cast<uint64_t>(token) * ks_], outBf_[qs_], ks_);
        outPending_ = true;
    }

    __aicore__ inline void ProcessToken(uint32_t token)
    {
        uint64_t row = static_cast<uint64_t>(token) * width_;
        LoadRope(token);
        if (gate_ && qh_ == 16 && (kh_ == 2 || kh_ == 4)) {
            LoadFixedRow(row, token);
        } else {
            CopyPassthrough(row, token);
            LoadAllHeads(row);
        }
        RmsNormAll();
        Rope(qh_ + kh_);
        StoreAllHeads(token);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> bfBuf_, outBfBuf_, passBuf_, rowBuf_, fpBuf_;
    TQue<QuePosition::VECIN, 2> ropeQueue_;
    LocalTensor<bfloat16_t> bf_, outBf_, passBf_, rowBf_;
    LocalTensor<float> x_, square_, temp_, weight_[2], biasFp_[2];
    LocalTensor<float> cos_, sin_, sum_, axes_;
    LocalTensor<uint32_t> cosOffset_, sinOffset_;
    GlobalTensor<bfloat16_t> input_, weights_[2], biases_[2], rope_;
    GlobalTensor<bfloat16_t> outputs_[2], passthrough_[2];

    uint32_t tokens_, qh_, kh_, blocks_, qs_, ks_, width_;
    float eps_;
    bool bias_[2], gate_, interleaved_;
    bool outPending_ = false;
    bool passPending_ = false;
};

template <uint32_t KH>
__aicore__ inline void split_qkv_rmsnorm_mrope_short_impl(GM_ADDR qkv, GM_ADDR qw, GM_ADDR kw, GM_ADDR cs, GM_ADDR qo,
                                                          GM_ADDR ko, GM_ADDR vo, GM_ADDR go, uint32_t tokens,
                                                          uint32_t blocks, float eps)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t QH = 16;
    constexpr uint32_t HEAD = 256;
    constexpr uint32_t QS = QH * HEAD;
    constexpr uint32_t KS = KH * HEAD;
    constexpr uint32_t HEADS = (QH + KH) / 2;
    constexpr uint32_t WIDTH = 2 * QS + 2 * KS;

    GlobalTensor<bfloat16_t> input, weights[2], rope, output[2], pass[2];
    input.SetGlobalBuffer((__gm__ bfloat16_t*)qkv);
    weights[0].SetGlobalBuffer((__gm__ bfloat16_t*)qw);
    weights[1].SetGlobalBuffer((__gm__ bfloat16_t*)kw);
    rope.SetGlobalBuffer((__gm__ bfloat16_t*)cs);
    output[0].SetGlobalBuffer((__gm__ bfloat16_t*)qo);
    output[1].SetGlobalBuffer((__gm__ bfloat16_t*)ko);
    pass[0].SetGlobalBuffer((__gm__ bfloat16_t*)go);
    pass[1].SetGlobalBuffer((__gm__ bfloat16_t*)vo);

    TPipe pipe;
    TBuf<TPosition::VECCALC> bfBuf, fpBuf;
    pipe.InitBuffer(bfBuf, 1024 * sizeof(bfloat16_t));
    pipe.InitBuffer(fpBuf, 1664 * sizeof(float));
    auto bf = bfBuf.Get<bfloat16_t>();
    auto fp = fpBuf.Get<float>();
    auto x = fp;
    auto w = fp[512];
    auto trig = fp[768];
    auto square = fp[960];
    auto temp = fp[1472];
    auto sum = fp[1600];

    for (uint32_t task = GetBlockIdx(); task < tokens * HEADS; task += blocks) {
        uint32_t token = task / HEADS;
        uint32_t group = task % HEADS;
        uint32_t kind = group >= QH / 2;
        uint32_t h = (kind ? group - QH / 2 : group) * 2;
        uint64_t row = static_cast<uint64_t>(token) * WIDTH;
        uint64_t src = row + (kind ? 2 * QS + h * HEAD : h * 2 * HEAD);
        uint64_t dst = (static_cast<uint64_t>(token) * (kind ? KH : QH) + h) * HEAD;

        if (kind) {
            DataCopy(bf, input[src], 2 * HEAD);
        } else {
            // Select q[h] and q[h + 1] from [q0, gate0, q1, gate1, ...].
            DataCopyParams gather{2, BF16_HEAD_BLOCKS, INTERLEAVED_HEAD_GAP_BLOCKS, 0};
            DataCopy(bf, input[src], gather);
        }
        DataCopy(bf[512], weights[kind], HEAD);
        for (uint32_t axis = 0; axis < 3; ++axis) {
            DataCopy(bf[768 + axis * 64], rope[(static_cast<uint64_t>(axis) * tokens + token) * 64], 64);
        }
        Sync<HardEvent::MTE2_V>();
        Cast(x, bf, RoundMode::CAST_NONE, 2 * HEAD);
        Cast(w, bf[512], RoundMode::CAST_NONE, HEAD);
        Cast(trig, bf[768], RoundMode::CAST_NONE, 192);
        PipeBarrier<PIPE_V>();

        Mul(square, x, x, 2 * HEAD);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams pairA{1, 1, 1, 8, 32, 32};
        Add(temp, square, square[128], 64, 2, pairA);
        BinaryRepeatParams pairB{1, 1, 1, 32, 32, 32};
        Add(square[64], square[64], square[192], 64, 2, pairB);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams merge{1, 1, 1, 8, 8, 32};
        Add(temp, temp, square[64], 64, 2, merge);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(sum, temp, 64, 2, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Muls(sum, sum, 1.0f / 256.0f, 2);
        PipeBarrier<PIPE_V>();
        Adds(sum, sum, eps, 2);
        PipeBarrier<PIPE_V>();
        Sqrt(sum, sum, 2);
        PipeBarrier<PIPE_V>();
        Duplicate(temp, 1.0f, 2);
        PipeBarrier<PIPE_V>();
        Div(sum, temp, sum, 2);
        PipeBarrier<PIPE_V>();
        Brcb(temp, sum, 1, {1, 8});
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams normalize{1, 1, 0, 32, 32, 1};
        for (uint32_t part = 0; part < 4; ++part) {
            Mul(x[part * 64], x[part * 64], temp, 64, 2, normalize);
        }
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams affine{1, 1, 1, 32, 32, 0};
        for (uint32_t part = 0; part < 4; ++part) {
            Mul(x[part * 64], x[part * 64], w[part * 64], 64, 2, affine);
        }
        Sync<HardEvent::V_S>();

        auto cosOffset = temp.ReinterpretCast<uint32_t>();
        auto sinOffset = sum.ReinterpretCast<uint32_t>();
        for (uint32_t j = 0; j < 32; ++j) {
            uint32_t axis = (j % 3 == 1) ? 1 : (j % 3 == 2 && j <= 30 ? 2 : 0);
            uint32_t c = (axis * 64 + j) * sizeof(float);
            cosOffset.SetValue(j, c);
            uint32_t s = c + 32 * sizeof(float);
            sinOffset.SetValue(j, s);
        }
        Sync<HardEvent::S_V>();
        Adds(cosOffset[32].ReinterpretCast<int32_t>(), cosOffset.ReinterpretCast<int32_t>(), 0, 32);
        Adds(sinOffset[32].ReinterpretCast<int32_t>(), sinOffset.ReinterpretCast<int32_t>(), 0, 32);
        PipeBarrier<PIPE_V>();
        Gather(w, trig, cosOffset, 0, 64);
        Gather(square, trig, sinOffset, 0, 64);
        PipeBarrier<PIPE_V>();
        UnaryRepeatParams rotate{1, 1, 8, 32};
        Muls(temp, x[32], -1.0f, 32, 2, rotate);
        Adds(temp[32], x, 0.0f, 32, 2, rotate);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams xTrig{1, 1, 1, 32, 32, 0};
        BinaryRepeatParams tempTrig{1, 1, 1, 8, 8, 0};
        Mul(temp, temp, square, 64, 2, tempTrig);
        Mul(x, x, w, 64, 2, xTrig);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams combine{1, 1, 1, 32, 32, 8};
        Add(x, x, temp, 64, 2, combine);
        PipeBarrier<PIPE_V>();
        UnaryRepeatParams castParams{1, 1, 4, 8};
        Cast(bf, x, RoundMode::CAST_RINT, 64, 8, castParams);
        Sync<HardEvent::V_MTE3>();
        DataCopy(output[kind][dst], bf, 2 * HEAD);
        Sync<HardEvent::MTE3_MTE2>();

        uint64_t copySrc = kind ? src + KS : src + HEAD;
        if (kind) {
            DataCopy(bf, input[copySrc], 2 * HEAD);
        } else {
            // Select gate[h] and gate[h + 1] with the same interleaved-head stride.
            DataCopyParams gather{2, BF16_HEAD_BLOCKS, INTERLEAVED_HEAD_GAP_BLOCKS, 0};
            DataCopy(bf, input[copySrc], gather);
        }
        Sync<HardEvent::MTE2_MTE3>();
        DataCopy(pass[kind][dst], bf, 2 * HEAD);
        Sync<HardEvent::MTE3_MTE2>();
    }
}

extern "C" __global__ __aicore__ void split_qkv_rmsnorm_mrope(GM_ADDR qkv, GM_ADDR qWeight, GM_ADDR kWeight,
                                                              GM_ADDR cosSin, GM_ADDR qBias, GM_ADDR kBias,
                                                              GM_ADDR qOutput, GM_ADDR kOutput, GM_ADDR vOutput,
                                                              GM_ADDR gateOutput, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    bool noBias = !data.hasQBias && !data.hasKBias;
    if (data.qHeads == 16 && noBias && data.hasGate && data.interleaved && data.tokens == 1 && data.kvHeads == 4) {
        split_qkv_rmsnorm_mrope_short_impl<4>(qkv, qWeight, kWeight, cosSin, qOutput, kOutput, vOutput, gateOutput,
                                              data.tokens, data.blocks, data.eps);
        return;
    }
    if (data.qHeads == 16 && noBias && data.hasGate && data.interleaved && data.tokens == 1 && data.kvHeads == 2) {
        split_qkv_rmsnorm_mrope_short_impl<2>(qkv, qWeight, kWeight, cosSin, qOutput, kOutput, vOutput, gateOutput,
                                              data.tokens, data.blocks, data.eps);
        return;
    }
    SplitQkvRmsnormMrope op;
    op.Init(qkv, qWeight, qBias, kWeight, kBias, cosSin, qOutput, kOutput, vOutput, gateOutput, data.tokens,
            data.qHeads, data.kvHeads, data.blocks, data.eps, data.hasQBias, data.hasKBias, data.hasGate,
            data.interleaved);
    op.Process();
}
