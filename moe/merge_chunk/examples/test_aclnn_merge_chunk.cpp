/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_merge_chunk.h"

namespace {
using DevicePtr = std::unique_ptr<void, decltype(&aclrtFree)>;
using TensorPtr = std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)>;
using TensorListPtr = std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)>;

void Check(int status, const char* operation)
{
    if (status != 0) {
        throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(status));
    }
}

struct Runtime {
    bool initialized = false;
    bool deviceSet = false;
    aclrtStream stream = nullptr;

    ~Runtime()
    {
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(0);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};

DevicePtr Allocate(size_t bytes)
{
    void* address = nullptr;
    if (bytes != 0) {
        Check(aclrtMalloc(&address, bytes, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
    }
    return DevicePtr(address, aclrtFree);
}

struct Tensor {
    DevicePtr storage;
    TensorPtr descriptor;

    Tensor(const std::vector<int64_t>& shape, aclDataType dtype, size_t bytes, const void* values = nullptr)
        : storage(Allocate(bytes)),
          descriptor(nullptr, aclDestroyTensor)
    {
        std::vector<int64_t> strides(shape.size(), 1);
        for (size_t i = shape.size(); i > 1; --i) {
            strides[i - 2] = strides[i - 1] * shape[i - 1];
        }
        descriptor.reset(aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                         shape.data(), shape.size(), storage.get()));
        if (!descriptor) {
            throw std::runtime_error("aclCreateTensor failed");
        }
        if (values != nullptr) {
            Check(aclrtMemcpy(storage.get(), bytes, values, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "upload");
        }
    }
};

TensorListPtr MakeList(std::initializer_list<Tensor*> tensors)
{
    std::vector<const aclTensor*> descriptors;
    for (auto* tensor : tensors) {
        descriptors.push_back(tensor->descriptor.get());
    }
    TensorListPtr list(aclCreateTensorList(descriptors.data(), descriptors.size()), aclDestroyTensorList);
    if (!list) {
        throw std::runtime_error("aclCreateTensorList failed");
    }
    for (auto* tensor : tensors) {
        tensor->descriptor.release();
    }
    return list;
}

constexpr size_t PATHS = 2;
constexpr size_t CHUNKS = 2;
constexpr size_t EXPERTS = 2;
constexpr size_t ROWS_PER_CHUNK = 2;
constexpr size_t OUTPUT_ROWS = CHUNKS * ROWS_PER_CHUNK;
constexpr size_t WIDTH = 16;
using TokenRows = std::array<aclFloat16, ROWS_PER_CHUNK * WIDTH>;

TokenRows Tokens(float first, float second)
{
    TokenRows rows;
    std::fill_n(rows.begin(), WIDTH, aclFloatToFloat16(first));
    std::fill_n(rows.begin() + WIDTH, WIDTH, aclFloatToFloat16(second));
    return rows;
}

void Run(aclrtStream stream)
{
    const auto first = Tokens(1, 2);
    const auto second = Tokens(3, 4);
    const std::array<int32_t, EXPERTS> counts{1, 1};
    std::array<aclFloat16, OUTPUT_ROWS * WIDTH> tokens{};
    std::array<int32_t, EXPERTS> outCounts{};
    Tensor chunk0({ROWS_PER_CHUNK, WIDTH}, ACL_FLOAT16, sizeof(first), first.data());
    Tensor chunk1({ROWS_PER_CHUNK, WIDTH}, ACL_FLOAT16, sizeof(second), second.data());
    const auto third = Tokens(5, 6);
    const auto fourth = Tokens(7, 8);
    Tensor chunk2({ROWS_PER_CHUNK, WIDTH}, ACL_FLOAT16, sizeof(third), third.data());
    Tensor chunk3({ROWS_PER_CHUNK, WIDTH}, ACL_FLOAT16, sizeof(fourth), fourth.data());
    Tensor counts0({EXPERTS}, ACL_INT32, sizeof(counts), counts.data());
    Tensor counts1({EXPERTS}, ACL_INT32, sizeof(counts), counts.data());
    Tensor merged({OUTPUT_ROWS, WIDTH}, ACL_FLOAT16, sizeof(tokens));
    Tensor mergedSecond({OUTPUT_ROWS, WIDTH}, ACL_FLOAT16, sizeof(tokens));
    Tensor mergedCounts({EXPERTS}, ACL_INT32, sizeof(outCounts));

    auto inputs = MakeList({&chunk0, &chunk1, &chunk2, &chunk3});
    auto groupLists = MakeList({&counts0, &counts1});
    auto outputs = MakeList({&merged, &mergedSecond});
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    Check(aclnnMergeChunkGetWorkspaceSize(inputs.get(), groupLists.get(), PATHS, CHUNKS, outputs.get(),
                                          mergedCounts.descriptor.get(), &workspaceSize, &executor),
          "aclnnMergeChunkGetWorkspaceSize");
    auto workspace = Allocate(workspaceSize);
    Check(aclnnMergeChunk(workspace.get(), workspaceSize, executor, stream), "aclnnMergeChunk");
    Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

    Check(aclrtMemcpy(tokens.data(), sizeof(tokens), merged.storage.get(), sizeof(tokens), ACL_MEMCPY_DEVICE_TO_HOST),
          "download tokens");
    Check(aclrtMemcpy(outCounts.data(), sizeof(outCounts), mergedCounts.storage.get(), sizeof(outCounts),
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "download counts");
    const std::array<float, OUTPUT_ROWS> expected{1, 3, 2, 4};
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] != aclFloatToFloat16(expected[i / WIDTH])) {
            throw std::runtime_error("unexpected token order");
        }
    }
    Check(aclrtMemcpy(tokens.data(), sizeof(tokens), mergedSecond.storage.get(), sizeof(tokens),
                      ACL_MEMCPY_DEVICE_TO_HOST),
          "download second path");
    const std::array<float, OUTPUT_ROWS> expectedSecond{5, 7, 6, 8};
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] != aclFloatToFloat16(expectedSecond[i / WIDTH])) {
            throw std::runtime_error("unexpected token order in second path");
        }
    }
    if (outCounts != std::array<int32_t, EXPERTS>{counts[0] + counts[0], counts[1] + counts[1]}) {
        throw std::runtime_error("unexpected expert counts");
    }
}
} // namespace

int main()
{
    Runtime runtime;
    try {
        Check(aclInit(nullptr), "aclInit");
        runtime.initialized = true;
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        runtime.deviceSet = true;
        Check(aclrtCreateStream(&runtime.stream), "aclrtCreateStream");
        Run(runtime.stream);
        std::cout << "MergeChunk example passed" << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
