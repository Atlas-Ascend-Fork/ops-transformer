/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_mega_mhc.h"

bool Check(int status, const char* step)
{
    if (status == 0) {
        return true;
    }
    std::cerr << step << " failed: " << status << std::endl;
    return false;
}

struct Resources {
    bool initialized = false;
    bool deviceSet = false;
    int32_t deviceId = 0;
    aclrtStream stream = nullptr;
    std::vector<aclTensor*> tensors;
    std::vector<void*> buffers;

    ~Resources()
    {
        // 等待异步任务完成，再释放其引用的资源。
        if (stream != nullptr) {
            Check(aclrtSynchronizeStream(stream), "cleanup aclrtSynchronizeStream");
        }
        for (auto* tensor : tensors) {
            Check(aclDestroyTensor(tensor), "aclDestroyTensor");
        }
        for (auto* buffer : buffers) {
            Check(aclrtFree(buffer), "aclrtFree");
        }
        if (stream != nullptr) {
            Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
        }
        if (deviceSet) {
            Check(aclrtResetDevice(deviceId), "aclrtResetDevice");
        }
        if (initialized) {
            Check(aclFinalize(), "aclFinalize");
        }
    }
};

// 使用0和1初始化输入，FLOAT16的1编码为0x3c00。
aclTensor* CreateTensor(Resources& resources, const std::vector<int64_t>& shape, aclDataType dtype, bool ones = false,
                        bool identity = false)
{
    int64_t count = 1;
    for (auto dim : shape) {
        count *= dim;
    }
    const uint64_t bytes = count * (dtype == ACL_FLOAT16 ? sizeof(uint16_t) : sizeof(float));
    void* deviceData = nullptr;
    if (!Check(aclrtMalloc(&deviceData, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc")) {
        return nullptr;
    }
    resources.buffers.push_back(deviceData);
    std::vector<float> fp32(count, ones ? 1.0f : 0.0f);
    std::vector<uint16_t> fp16(count, ones ? 0x3c00 : 0);
    if (identity) {
        const int64_t n = shape.back();
        for (int64_t offset = 0; offset < count; offset += n * n) {
            for (int64_t head = 0; head < n; ++head) {
                fp32[offset + head * n + head] = 1.0f;
            }
        }
    }
    const void* hostData =
        dtype == ACL_FLOAT16 ? static_cast<const void*>(fp16.data()) : static_cast<const void*>(fp32.data());
    if (!Check(aclrtMemcpy(deviceData, bytes, hostData, bytes, ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy")) {
        return nullptr;
    }
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t axis = static_cast<int64_t>(shape.size()) - 2; axis >= 0; --axis) {
        strides[axis] = strides[axis + 1] * shape[axis + 1];
    }
    aclTensor* tensor = aclCreateTensor(shape.data(), shape.size(), dtype, strides.data(), 0, ACL_FORMAT_ND,
                                        shape.data(), shape.size(), deviceData);
    if (tensor != nullptr) {
        resources.tensors.push_back(tensor);
    }
    return tensor;
}

int Run()
{
    // 初始化运行环境、设备及Stream。
    Resources resources;
    if (!Check(aclInit(nullptr), "aclInit")) {
        return 1;
    }
    resources.initialized = true;
    if (!Check(aclrtSetDevice(resources.deviceId), "aclrtSetDevice")) {
        return 1;
    }
    resources.deviceSet = true;
    if (!Check(aclrtCreateStream(&resources.stream), "aclrtCreateStream")) {
        return 1;
    }

    const int64_t t = 16;
    const int64_t n = 4;
    const int64_t d = 4096;
    const int64_t q = n * n + 2 * n;
    const bool withMid = false;
    const int64_t opImplMode = 0;
    const double rmsNormEps = 1e-6;
    const double hcNormEps = 1e-6;
    const double hcPreEps = 1e-6;
    const int64_t hcMult = n;
    const double hcPostScale = 2.0;
    const double sinkhornEps = 1e-6;
    const int64_t numSinkhornIters = 0;
    const double rmsNormScale = 1.0;
    const int64_t outputMode = 0;

    // 创建输入和输出Tensor。
    auto* x = CreateTensor(resources, {t, d}, ACL_FLOAT16);
    auto* residual = CreateTensor(resources, {t, n, d}, ACL_FLOAT16);
    auto* postMix = CreateTensor(resources, {t, n, 1}, ACL_FLOAT, true);
    auto* fn = CreateTensor(resources, {q, n * d}, ACL_FLOAT);
    auto* mixScales = CreateTensor(resources, {3}, ACL_FLOAT, true);
    auto* mixBases = CreateTensor(resources, {q}, ACL_FLOAT);
    auto* rmsNorm1Weight = CreateTensor(resources, {d}, ACL_FLOAT, true);
    auto* combResMixOptional = CreateTensor(resources, {t, n, n}, ACL_FLOAT, false, true);
    const aclTensor* shiftedPrevMixOptional = nullptr;
    auto* rmsNorm0WeightOptional = CreateTensor(resources, {d}, ACL_FLOAT, true);
    aclTensor* rmsNormMidWeightOptional = withMid ? CreateTensor(resources, {n * d}, ACL_FLOAT, true) : nullptr;
    auto* yOut = CreateTensor(resources, {t, d}, ACL_FLOAT16);
    auto* newPostMixOut = CreateTensor(resources, {t, n, 1}, ACL_FLOAT);
    auto* newCombResMixOut = CreateTensor(resources, {t, n, n}, ACL_FLOAT);
    auto* newResidualOut = CreateTensor(resources, {t, n, d}, ACL_FLOAT16);
    aclTensor* newPrevMixOut = nullptr;
    // 输出原精度结果，预留量化输出必须传入nullptr。
    aclTensor* yQuantOut = nullptr;
    aclTensor* yScaleOut = nullptr;
    if (x == nullptr || residual == nullptr || postMix == nullptr || fn == nullptr || mixScales == nullptr ||
        mixBases == nullptr || rmsNorm1Weight == nullptr || combResMixOptional == nullptr ||
        rmsNorm0WeightOptional == nullptr || (withMid && rmsNormMidWeightOptional == nullptr) || yOut == nullptr ||
        newPostMixOut == nullptr || newCombResMixOut == nullptr || newResidualOut == nullptr) {
        std::cerr << "CreateTensor failed" << std::endl;
        return 1;
    }

    // 第一段接口获取workspace大小及执行器。
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    if (!Check(aclnnMegaMhcGetWorkspaceSize(x, residual, postMix, fn, mixScales, mixBases, rmsNorm1Weight,
                                            combResMixOptional, shiftedPrevMixOptional, rmsNorm0WeightOptional,
                                            rmsNormMidWeightOptional, rmsNormEps, hcNormEps, hcPreEps, opImplMode,
                                            hcMult, hcPostScale, sinkhornEps, numSinkhornIters, rmsNormScale,
                                            outputMode, newResidualOut, newPostMixOut, newCombResMixOut, newPrevMixOut,
                                            yOut, yQuantOut, yScaleOut, &workspaceSize, &executor),
               "aclnnMegaMhcGetWorkspaceSize")) {
        return 1;
    }
    void* workspace = nullptr;
    if (workspaceSize != 0) {
        if (!Check(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "workspace allocation")) {
            return 1;
        }
        resources.buffers.push_back(workspace);
    }
    // 第二段接口执行计算，等待任务完成。
    if (!Check(aclnnMegaMhc(workspace, workspaceSize, executor, resources.stream), "aclnnMegaMhc") ||
        !Check(aclrtSynchronizeStream(resources.stream), "aclrtSynchronizeStream")) {
        return 1;
    }
    std::cout << "MegaMhc completed: " << (withMid ? "with_mid" : "no_mid") << ", T=" << t << ", N=" << n << ", D=" << d
              << std::endl;
    return 0;
}

int main()
{
    return Run();
}
