/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_split_qkv_rmsnorm_mrope.h"

namespace {
constexpr int DEVICE_ID = 0;
constexpr int64_t Q_HEADS = 16;
constexpr int64_t KV_HEADS = 4;
constexpr int64_t HEAD_DIM = 256;
constexpr int64_t Q_SIZE = Q_HEADS * HEAD_DIM;
constexpr int64_t KV_SIZE = KV_HEADS * HEAD_DIM;
constexpr int64_t INPUT_WIDTH = 2 * Q_SIZE + 2 * KV_SIZE;

#define CHECK_ACL(expression) \
    do { \
        const aclError status = (expression); \
        if (status != ACL_SUCCESS) { \
            std::printf("%s failed: %d\n", #expression, static_cast<int>(status)); \
            return status; \
        } \
    } while (0)

struct TensorResource {
    void* device = nullptr;
    aclTensor* tensor = nullptr;
};

int64_t Numel(const std::vector<int64_t>& shape)
{
    int64_t count = 1;
    for (int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

int CreateBf16Tensor(const std::vector<int64_t>& shape, TensorResource& resource)
{
    const size_t bytes = static_cast<size_t>(Numel(shape)) * sizeof(uint16_t);
    CHECK_ACL(aclrtMalloc(&resource.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(resource.device, bytes, 0, bytes));
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    resource.tensor = aclCreateTensor(shape.data(), shape.size(), ACL_BF16, strides.data(), 0, ACL_FORMAT_ND,
                                      shape.data(), shape.size(), resource.device);
    return resource.tensor == nullptr ? ACL_ERROR_INVALID_PARAM : ACL_SUCCESS;
}

void DestroyTensor(TensorResource& resource)
{
    if (resource.tensor != nullptr) {
        aclDestroyTensor(resource.tensor);
    }
    if (resource.device != nullptr) {
        aclrtFree(resource.device);
    }
}

uint16_t FloatToBf16(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<uint16_t>(bits >> 16U);
}

float Bf16ToFloat(uint16_t value)
{
    uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

aclError CopyToDevice(const std::vector<uint16_t>& source, TensorResource& destination)
{
    return aclrtMemcpy(destination.device, source.size() * sizeof(uint16_t), source.data(),
                       source.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE);
}

void ComputeHeadReference(const std::vector<uint16_t>& input, int64_t inputOffset, const std::vector<uint16_t>& weight,
                          const std::vector<uint16_t>* bias, const std::vector<uint16_t>& cosSin, int64_t token,
                          int64_t tokens, std::vector<float>& output, int64_t outputOffset)
{
    std::array<float, HEAD_DIM> affine{};
    float squareSum = 0.0F;
    for (int64_t dim = 0; dim < HEAD_DIM; ++dim) {
        const float value = Bf16ToFloat(input[inputOffset + dim]);
        squareSum += value * value;
    }
    const float inverseRms = 1.0F / std::sqrt(squareSum / static_cast<float>(HEAD_DIM) + 1.0e-6F);
    for (int64_t dim = 0; dim < HEAD_DIM; ++dim) {
        const float normalized = Bf16ToFloat(input[inputOffset + dim]) * inverseRms;
        const float biasValue = bias == nullptr ? 0.0F : Bf16ToFloat((*bias)[dim]);
        affine[dim] = normalized * Bf16ToFloat(weight[dim]) + biasValue;
        output[outputOffset + dim] = affine[dim];
    }
    for (int64_t dim = 0; dim < 32; ++dim) {
        const int64_t axis = dim % 3;
        const int64_t ropeOffset = (axis * tokens + token) * 64;
        const float cosValue = Bf16ToFloat(cosSin[ropeOffset + dim]);
        const float sinValue = Bf16ToFloat(cosSin[ropeOffset + 32 + dim]);
        output[outputOffset + dim] = affine[dim] * cosValue - affine[dim + 32] * sinValue;
        output[outputOffset + 32 + dim] = affine[dim + 32] * cosValue + affine[dim] * sinValue;
    }
}

float MaxAbsError(const std::vector<uint16_t>& actual, const std::vector<float>& expected)
{
    float maxError = 0.0F;
    for (size_t index = 0; index < actual.size(); ++index) {
        const float value = Bf16ToFloat(actual[index]);
        if (!std::isfinite(value)) {
            return INFINITY;
        }
        maxError = std::fmax(maxError, std::fabs(value - expected[index]));
    }
    return maxError;
}

aclError RunCase(int64_t tokens, bool withBias, aclrtStream stream)
{
    TensorResource qkv;
    TensorResource qWeight;
    TensorResource kWeight;
    TensorResource cosSin;
    TensorResource qBias;
    TensorResource kBias;
    TensorResource qOutput;
    TensorResource kOutput;
    TensorResource vOutput;
    TensorResource gateOutput;
    CHECK_ACL(CreateBf16Tensor({tokens, INPUT_WIDTH}, qkv));
    CHECK_ACL(CreateBf16Tensor({HEAD_DIM}, qWeight));
    CHECK_ACL(CreateBf16Tensor({HEAD_DIM}, kWeight));
    CHECK_ACL(CreateBf16Tensor({3, tokens, 64}, cosSin));
    if (withBias) {
        CHECK_ACL(CreateBf16Tensor({HEAD_DIM}, qBias));
        CHECK_ACL(CreateBf16Tensor({HEAD_DIM}, kBias));
    }
    CHECK_ACL(CreateBf16Tensor({tokens, Q_SIZE}, qOutput));
    CHECK_ACL(CreateBf16Tensor({tokens, KV_SIZE}, kOutput));
    CHECK_ACL(CreateBf16Tensor({tokens, KV_SIZE}, vOutput));
    CHECK_ACL(CreateBf16Tensor({tokens, Q_SIZE}, gateOutput));

    std::vector<uint16_t> input(static_cast<size_t>(tokens * INPUT_WIDTH), 0U);
    std::vector<uint16_t> qWeightData(HEAD_DIM);
    std::vector<uint16_t> kWeightData(HEAD_DIM);
    std::vector<uint16_t> qBiasData(HEAD_DIM);
    std::vector<uint16_t> kBiasData(HEAD_DIM);
    std::vector<uint16_t> cosSinData(static_cast<size_t>(3 * tokens * 64));
    std::vector<float> expectedQ(static_cast<size_t>(tokens * Q_SIZE));
    std::vector<float> expectedK(static_cast<size_t>(tokens * KV_SIZE));
    std::vector<uint16_t> expectedGate(static_cast<size_t>(tokens * Q_SIZE));
    std::vector<uint16_t> expectedV(static_cast<size_t>(tokens * KV_SIZE));
    for (int64_t dim = 0; dim < HEAD_DIM; ++dim) {
        qWeightData[dim] = FloatToBf16(0.75F + static_cast<float>(dim % 5) * 0.0625F);
        kWeightData[dim] = FloatToBf16(0.625F + static_cast<float>(dim % 7) * 0.03125F);
        qBiasData[dim] = FloatToBf16(static_cast<float>(dim % 5 - 2) * 0.015625F);
        kBiasData[dim] = FloatToBf16(static_cast<float>(dim % 7 - 3) * 0.0078125F);
    }
    constexpr float ROPE_COS[3] = {0.5F, 0.75F, 1.0F};
    constexpr float ROPE_SIN[3] = {0.25F, -0.5F, 0.125F};
    for (int64_t axis = 0; axis < 3; ++axis) {
        for (int64_t token = 0; token < tokens; ++token) {
            const int64_t offset = (axis * tokens + token) * 64;
            for (int64_t dim = 0; dim < 32; ++dim) {
                cosSinData[offset + dim] = FloatToBf16(ROPE_COS[axis]);
                cosSinData[offset + 32 + dim] = FloatToBf16(ROPE_SIN[axis]);
            }
        }
    }
    for (int64_t token = 0; token < tokens; ++token) {
        const int64_t row = token * INPUT_WIDTH;
        for (int64_t head = 0; head < Q_HEADS; ++head) {
            for (int64_t dim = 0; dim < HEAD_DIM; ++dim) {
                const float qValue = static_cast<float>((token * 11 + head * 7 + dim) % 31 - 15) * 0.03125F;
                input[row + 2 * head * HEAD_DIM + dim] = FloatToBf16(qValue);
                const uint16_t gateValue = static_cast<uint16_t>(0x3F00U + ((token * 19 + head * 7 + dim) & 0x7FU));
                input[row + (2 * head + 1) * HEAD_DIM + dim] = gateValue;
                expectedGate[token * Q_SIZE + head * HEAD_DIM + dim] = gateValue;
            }
        }
        const int64_t kBase = row + 2 * Q_SIZE;
        const int64_t vBase = kBase + KV_SIZE;
        for (int64_t index = 0; index < KV_SIZE; ++index) {
            const float kValue = static_cast<float>((token * 13 + index * 3) % 29 - 14) * 0.03125F;
            input[kBase + index] = FloatToBf16(kValue);
            const uint16_t value = static_cast<uint16_t>(0x4000U + ((token * 23 + index) & 0x7FU));
            input[vBase + index] = value;
            expectedV[token * KV_SIZE + index] = value;
        }
        for (int64_t head = 0; head < Q_HEADS; ++head) {
            ComputeHeadReference(input, row + 2 * head * HEAD_DIM, qWeightData, withBias ? &qBiasData : nullptr,
                                 cosSinData, token, tokens, expectedQ, token * Q_SIZE + head * HEAD_DIM);
        }
        for (int64_t head = 0; head < KV_HEADS; ++head) {
            ComputeHeadReference(input, kBase + head * HEAD_DIM, kWeightData, withBias ? &kBiasData : nullptr,
                                 cosSinData, token, tokens, expectedK, token * KV_SIZE + head * HEAD_DIM);
        }
    }
    CHECK_ACL(CopyToDevice(input, qkv));
    CHECK_ACL(CopyToDevice(qWeightData, qWeight));
    CHECK_ACL(CopyToDevice(kWeightData, kWeight));
    CHECK_ACL(CopyToDevice(cosSinData, cosSin));
    if (withBias) {
        CHECK_ACL(CopyToDevice(qBiasData, qBias));
        CHECK_ACL(CopyToDevice(kBiasData, kBias));
    }

    if (tokens == 1) {
        uint64_t rejectedWorkspaceSize = 0;
        aclOpExecutor* rejectedExecutor = nullptr;
        const aclnnStatus zeroEpsStatus = aclnnSplitQkvRmsnormMropeGetWorkspaceSize(
            qkv.tensor, qWeight.tensor, kWeight.tensor, cosSin.tensor, qBias.tensor, kBias.tensor, Q_HEADS, KV_HEADS,
            0.0, true, true, qOutput.tensor, kOutput.tensor, vOutput.tensor, gateOutput.tensor, &rejectedWorkspaceSize,
            &rejectedExecutor);
        const aclnnStatus invalidHeadStatus = aclnnSplitQkvRmsnormMropeGetWorkspaceSize(
            qkv.tensor, qWeight.tensor, kWeight.tensor, cosSin.tensor, qBias.tensor, kBias.tensor, 32, KV_HEADS, 1.0e-6,
            true, true, qOutput.tensor, kOutput.tensor, vOutput.tensor, gateOutput.tensor, &rejectedWorkspaceSize,
            &rejectedExecutor);
        if (zeroEpsStatus == ACL_SUCCESS || invalidHeadStatus == ACL_SUCCESS) {
            std::printf("invalid eps or head count was not rejected\n");
            return ACL_ERROR_INVALID_PARAM;
        }
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnSplitQkvRmsnormMropeGetWorkspaceSize(qkv.tensor, qWeight.tensor, kWeight.tensor, cosSin.tensor,
                                                        qBias.tensor, kBias.tensor, Q_HEADS, KV_HEADS, 1.0e-6, true,
                                                        true, qOutput.tensor, kOutput.tensor, vOutput.tensor,
                                                        gateOutput.tensor, &workspaceSize, &executor));
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnSplitQkvRmsnormMrope(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<uint16_t> qResult(static_cast<size_t>(tokens * Q_SIZE));
    std::vector<uint16_t> kResult(static_cast<size_t>(tokens * KV_SIZE));
    std::vector<uint16_t> gateResult(expectedGate.size());
    std::vector<uint16_t> vResult(expectedV.size());
    CHECK_ACL(aclrtMemcpy(qResult.data(), qResult.size() * sizeof(uint16_t), qOutput.device,
                          qResult.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(kResult.data(), kResult.size() * sizeof(uint16_t), kOutput.device,
                          kResult.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(gateResult.data(), gateResult.size() * sizeof(uint16_t), gateOutput.device,
                          gateResult.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(vResult.data(), vResult.size() * sizeof(uint16_t), vOutput.device,
                          vResult.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST));
    const float qMaxAbs = MaxAbsError(qResult, expectedQ);
    const float kMaxAbs = MaxAbsError(kResult, expectedK);
    constexpr float MAX_ABS_TOLERANCE = 0.01F;
    const bool passed = qMaxAbs < MAX_ABS_TOLERANCE && kMaxAbs < MAX_ABS_TOLERANCE && gateResult == expectedGate &&
                        vResult == expectedV;
    std::printf("SplitQkvRmsnormMrope tokens=%ld bias=%s: %s, q_max_abs=%.8f, k_max_abs=%.8f, workspace=%lu "
                "bytes\n",
                tokens, withBias ? "true" : "false", passed ? "PASSED" : "FAILED", qMaxAbs, kMaxAbs, workspaceSize);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    DestroyTensor(qkv);
    DestroyTensor(qWeight);
    DestroyTensor(kWeight);
    DestroyTensor(cosSin);
    DestroyTensor(qBias);
    DestroyTensor(kBias);
    DestroyTensor(qOutput);
    DestroyTensor(kOutput);
    DestroyTensor(vOutput);
    DestroyTensor(gateOutput);
    return passed ? ACL_SUCCESS : ACL_ERROR_INVALID_PARAM;
}
} // namespace

int main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(DEVICE_ID));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    CHECK_ACL(RunCase(1, false, stream));
    CHECK_ACL(RunCase(2, true, stream));
    aclrtDestroyStream(stream);
    aclrtResetDevice(DEVICE_ID);
    aclFinalize();
    return 0;
}
