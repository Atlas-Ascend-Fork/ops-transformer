/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_turbo_quant_attention_score.h"

namespace {
constexpr int DEVICE_ID = 0;
constexpr int64_t QUERY_HEADS = 2;
constexpr int64_t KEY_HEADS = 2;
constexpr int64_t HEAD_DIM = 128;
constexpr int64_t BLOCK_SIZE = 128;
constexpr int64_t BLOCKS = 2;
constexpr int64_t TABLE_PAGES = 2;
constexpr int64_t CAPACITY = TABLE_PAGES * BLOCK_SIZE;
constexpr int32_t SEQUENCE_LENGTH = 129;
constexpr float MAX_ABS_TOLERANCE = 1.0e-4F;

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

size_t TypeBytes(aclDataType dtype)
{
    if (dtype == ACL_UINT8) {
        return sizeof(uint8_t);
    }
    if (dtype == ACL_INT32 || dtype == ACL_FLOAT) {
        return sizeof(uint32_t);
    }
    return sizeof(uint16_t);
}

aclError CreateTensor(const std::vector<int64_t>& shape, aclDataType dtype, TensorResource& resource)
{
    const size_t bytes = static_cast<size_t>(Numel(shape)) * TypeBytes(dtype);
    CHECK_ACL(aclrtMalloc(&resource.device, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemset(resource.device, bytes, 0, bytes));
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t index = static_cast<int64_t>(shape.size()) - 2; index >= 0; --index) {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    resource.tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND, shape.data(),
                                      shape.size(), resource.device);
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

float Centroid(int bits, uint32_t index)
{
    static constexpr float VALUES_2[4] = {-0.1335033178F, -0.04002048075F, 0.04002048075F, 0.1335033178F};
    static constexpr float VALUES_3[8] = {-0.19020693F,   -0.1187859178F, -0.06682205945F, -0.02166347019F,
                                          0.02166347019F, 0.06682205945F, 0.1187859178F,   0.19020693F};
    static constexpr float VALUES_4[16] = {-0.2414890379F,  -0.1828317791F,  -0.1429702938F,  -0.1109927073F,
                                           -0.08325428516F, -0.05802082643F, -0.03428063914F, -0.01134236995F,
                                           0.01134236995F,  0.03428063914F,  0.05802082643F,  0.08325428516F,
                                           0.1109927073F,   0.1429702938F,   0.1828317791F,   0.2414890379F};
    return bits == 2 ? VALUES_2[index] : (bits == 3 ? VALUES_3[index] : VALUES_4[index]);
}

template <typename T>
aclError CopyToDevice(const std::vector<T>& source, TensorResource& destination)
{
    return aclrtMemcpy(destination.device, source.size() * sizeof(T), source.data(), source.size() * sizeof(T),
                       ACL_MEMCPY_HOST_TO_DEVICE);
}

void PackPrimary(std::vector<uint8_t>& packed, int64_t base, int bits, int64_t physical, int64_t offset, int64_t head)
{
    const uint32_t mask = (1U << bits) - 1U;
    for (int64_t dim = 0; dim < HEAD_DIM; ++dim) {
        const uint32_t value = static_cast<uint32_t>((physical * 17 + offset * 5 + head * 3 + dim) & mask);
        const int64_t bitOffset = dim * bits;
        for (int bit = 0; bit < bits; ++bit) {
            if (((value >> bit) & 1U) != 0U) {
                packed[base + (bitOffset + bit) / 8] |= static_cast<uint8_t>(1U << ((bitOffset + bit) % 8));
            }
        }
    }
}

void PackSigns(std::vector<uint8_t>& packed, int64_t base, int64_t physical, int64_t offset, int64_t head)
{
    for (int64_t dim = 0; dim < HEAD_DIM; ++dim) {
        const bool positive = ((physical * 11 + offset * 3 + head * 5 + dim) & 1) != 0;
        if (positive) {
            packed[base + dim / 8] |= static_cast<uint8_t>(1U << (dim % 8));
        }
    }
}

aclError RunCase(int bits, int64_t tokens, bool checkInvalidBlocks, aclrtStream stream)
{
    std::vector<TensorResource> tensors(9);
    CHECK_ACL(CreateTensor({tokens, QUERY_HEADS, HEAD_DIM}, ACL_BF16, tensors[0]));
    CHECK_ACL(CreateTensor({tokens, QUERY_HEADS, HEAD_DIM}, ACL_BF16, tensors[1]));
    CHECK_ACL(CreateTensor({BLOCKS, BLOCK_SIZE, KEY_HEADS, 16 * bits}, ACL_UINT8, tensors[2]));
    CHECK_ACL(CreateTensor({BLOCKS, BLOCK_SIZE, KEY_HEADS, 16}, ACL_UINT8, tensors[3]));
    CHECK_ACL(CreateTensor({BLOCKS, BLOCK_SIZE, KEY_HEADS}, ACL_BF16, tensors[4]));
    CHECK_ACL(CreateTensor({BLOCKS, BLOCK_SIZE, KEY_HEADS}, ACL_BF16, tensors[5]));
    CHECK_ACL(CreateTensor({1, TABLE_PAGES}, ACL_INT32, tensors[6]));
    CHECK_ACL(CreateTensor({1}, ACL_INT32, tensors[7]));
    CHECK_ACL(CreateTensor({tokens, QUERY_HEADS, CAPACITY}, ACL_FLOAT, tensors[8]));

    std::vector<uint16_t> queryRotated(static_cast<size_t>(tokens * QUERY_HEADS * HEAD_DIM));
    std::vector<uint16_t> queryQjl(queryRotated.size());
    std::vector<uint8_t> keyIdx(static_cast<size_t>(BLOCKS * BLOCK_SIZE * KEY_HEADS * 16 * bits), 0U);
    std::vector<uint8_t> keyQjl(static_cast<size_t>(BLOCKS * BLOCK_SIZE * KEY_HEADS * 16), 0U);
    std::vector<uint16_t> keyNorm(static_cast<size_t>(BLOCKS * BLOCK_SIZE * KEY_HEADS));
    std::vector<uint16_t> keyGamma(keyNorm.size());
    const std::vector<int32_t> blockTable = {1, 0};
    const std::vector<int32_t> sequenceLength = {SEQUENCE_LENGTH};

    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t head = 0; head < QUERY_HEADS; ++head) {
            for (int64_t dim = 0; dim < HEAD_DIM; ++dim) {
                const int64_t index = (token * QUERY_HEADS + head) * HEAD_DIM + dim;
                const float mainValue = static_cast<float>((token * 7 + head * 5 + dim) % 23 - 11) * 0.03125F;
                const float qjlValue = static_cast<float>((token * 3 + head * 11 + dim * 2) % 19 - 9) * 0.03125F;
                queryRotated[index] = FloatToBf16(mainValue);
                queryQjl[index] = FloatToBf16(qjlValue);
            }
        }
    }
    for (int64_t physical = 0; physical < BLOCKS; ++physical) {
        for (int64_t offset = 0; offset < BLOCK_SIZE; ++offset) {
            for (int64_t head = 0; head < KEY_HEADS; ++head) {
                const int64_t metadataIndex = (physical * BLOCK_SIZE + offset) * KEY_HEADS + head;
                const float norm = 0.75F + static_cast<float>((physical + offset + head) % 5) * 0.125F;
                const float gamma = 0.25F + static_cast<float>((physical * 3 + offset + head) % 4) * 0.0625F;
                keyNorm[metadataIndex] = FloatToBf16(norm);
                keyGamma[metadataIndex] = FloatToBf16(gamma);
                PackPrimary(keyIdx, metadataIndex * 16 * bits, bits, physical, offset, head);
                PackSigns(keyQjl, metadataIndex * 16, physical, offset, head);
            }
        }
    }

    CHECK_ACL(CopyToDevice(queryRotated, tensors[0]));
    CHECK_ACL(CopyToDevice(queryQjl, tensors[1]));
    CHECK_ACL(CopyToDevice(keyIdx, tensors[2]));
    CHECK_ACL(CopyToDevice(keyQjl, tensors[3]));
    CHECK_ACL(CopyToDevice(keyNorm, tensors[4]));
    CHECK_ACL(CopyToDevice(keyGamma, tensors[5]));
    CHECK_ACL(CopyToDevice(blockTable, tensors[6]));
    CHECK_ACL(CopyToDevice(sequenceLength, tensors[7]));

    std::vector<float> expected(static_cast<size_t>(tokens * QUERY_HEADS * CAPACITY), -INFINITY);
    const float coefficient = std::sqrt(std::acos(-1.0F) / 2.0F) / static_cast<float>(HEAD_DIM);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t head = 0; head < QUERY_HEADS; ++head) {
            for (int64_t key = 0; key < SEQUENCE_LENGTH; ++key) {
                const int64_t physical = blockTable[key / BLOCK_SIZE];
                const int64_t offset = key % BLOCK_SIZE;
                const int64_t metadataIndex = (physical * BLOCK_SIZE + offset) * KEY_HEADS + head;
                float main = 0.0F;
                float correction = 0.0F;
                for (int64_t dim = 0; dim < HEAD_DIM; ++dim) {
                    const int64_t queryIndex = (token * QUERY_HEADS + head) * HEAD_DIM + dim;
                    const uint32_t code =
                        static_cast<uint32_t>((physical * 17 + offset * 5 + head * 3 + dim) & ((1U << bits) - 1U));
                    const float sign = ((physical * 11 + offset * 3 + head * 5 + dim) & 1) != 0 ? 1.0F : -1.0F;
                    main += Bf16ToFloat(queryRotated[queryIndex]) * Centroid(bits, code);
                    correction += Bf16ToFloat(queryQjl[queryIndex]) * sign;
                }
                expected[(token * QUERY_HEADS + head) * CAPACITY + key] =
                    main * Bf16ToFloat(keyNorm[metadataIndex]) +
                    coefficient * correction * Bf16ToFloat(keyGamma[metadataIndex]);
            }
        }
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    CHECK_ACL(aclnnTurboQuantAttentionScoreGetWorkspaceSize(
        tensors[0].tensor, tensors[1].tensor, tensors[2].tensor, tensors[3].tensor, tensors[4].tensor,
        tensors[5].tensor, tensors[6].tensor, tensors[7].tensor, bits, tensors[8].tensor, &workspaceSize, &executor));
    void* workspace = nullptr;
    if (workspaceSize > 0) {
        CHECK_ACL(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    }
    CHECK_ACL(aclnnTurboQuantAttentionScore(workspace, workspaceSize, executor, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    std::vector<float> output(expected.size());
    CHECK_ACL(aclrtMemcpy(output.data(), output.size() * sizeof(float), tensors[8].device,
                          output.size() * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    bool passed = true;
    float maxAbs = 0.0F;
    for (size_t index = 0; index < output.size(); ++index) {
        if (std::isinf(expected[index])) {
            passed = passed && std::isinf(output[index]) && output[index] < 0.0F;
        } else {
            const float error = std::fabs(output[index] - expected[index]);
            maxAbs = std::fmax(maxAbs, error);
            passed = passed && std::isfinite(output[index]) && error < MAX_ABS_TOLERANCE;
        }
    }

    auto invalidPhysicalBlockIsMasked = [&](int32_t invalidPhysicalBlock) {
        const std::vector<int32_t> invalidBlockTable = {invalidPhysicalBlock, 0};
        if (CopyToDevice(invalidBlockTable, tensors[6]) != ACL_SUCCESS ||
            aclrtMemset(tensors[8].device, output.size() * sizeof(float), 0, output.size() * sizeof(float)) !=
                ACL_SUCCESS) {
            return false;
        }
        uint64_t requiredWorkspace = 0;
        aclOpExecutor* invalidExecutor = nullptr;
        if (aclnnTurboQuantAttentionScoreGetWorkspaceSize(tensors[0].tensor, tensors[1].tensor, tensors[2].tensor,
                                                          tensors[3].tensor, tensors[4].tensor, tensors[5].tensor,
                                                          tensors[6].tensor, tensors[7].tensor, bits, tensors[8].tensor,
                                                          &requiredWorkspace, &invalidExecutor) != ACL_SUCCESS ||
            requiredWorkspace > workspaceSize ||
            aclnnTurboQuantAttentionScore(workspace, requiredWorkspace, invalidExecutor, stream) != ACL_SUCCESS ||
            aclrtSynchronizeStream(stream) != ACL_SUCCESS ||
            aclrtMemcpy(output.data(), output.size() * sizeof(float), tensors[8].device, output.size() * sizeof(float),
                        ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            return false;
        }
        for (int64_t token = 0; token < tokens; ++token) {
            for (int64_t head = 0; head < QUERY_HEADS; ++head) {
                for (int64_t key = 0; key < CAPACITY; ++key) {
                    const float value = output[(token * QUERY_HEADS + head) * CAPACITY + key];
                    if (key == BLOCK_SIZE) {
                        const float reference = expected[(token * QUERY_HEADS + head) * CAPACITY + key];
                        if (!std::isfinite(value) || std::fabs(value - reference) >= MAX_ABS_TOLERANCE) {
                            return false;
                        }
                    } else if (!(std::isinf(value) && value < 0.0F)) {
                        return false;
                    }
                }
            }
        }
        return true;
    };
    if (checkInvalidBlocks) {
        passed = passed && invalidPhysicalBlockIsMasked(-1);
        passed = passed && invalidPhysicalBlockIsMasked(BLOCKS);
    }
    std::printf("TurboQuantAttentionScore bits=%d mode=%s: %s, max_abs=%.8f, workspace=%lu bytes\n", bits,
                tokens > 1 ? "prefill" : "decode", passed ? "PASSED" : "FAILED", maxAbs, workspaceSize);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    for (auto& tensor : tensors) {
        DestroyTensor(tensor);
    }
    return passed ? ACL_SUCCESS : ACL_ERROR_INVALID_PARAM;
}
} // namespace

int main()
{
    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(DEVICE_ID));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    for (int bits = 2; bits <= 4; ++bits) {
        CHECK_ACL(RunCase(bits, 1, false, stream));
        CHECK_ACL(RunCase(bits, 3, bits == 3, stream));
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(DEVICE_ID);
    aclFinalize();
    return 0;
}
