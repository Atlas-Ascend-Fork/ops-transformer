/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_gather_pa_kv_cache.cpp
 * \brief
 */
#include "data_utils.h"
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_gather_pa_kv_cache_tiling_def.h"

using namespace std;

extern "C" __global__ __aicore__ void gather_pa_kv_cache(GM_ADDR keyCacheInGm, GM_ADDR valueCacheInGm,
                                                         GM_ADDR blockTablesInGm, GM_ADDR seqLensInGm, GM_ADDR keyInGm,
                                                         GM_ADDR valueInGm, GM_ADDR seqOffsetInGm, GM_ADDR keyOutGm,
                                                         GM_ADDR valueOutGm, GM_ADDR workspace, GM_ADDR tiling);

class gather_pa_kv_cache_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "gather_pa_kv_cache_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "gather_pa_kv_cache_test TearDown\n" << endl;
    }
};

TEST_F(gather_pa_kv_cache_test, test_tiling_key_577)
{
    uint32_t keyCache_shape = 30 * 101 * 128 * 32;
    uint32_t valueCache_shape = 30 * 127 * 128 * 32;
    uint32_t blockTables_shape = 36 * 8;
    uint32_t seqLens_shape = 36;
    uint32_t key_shape = 18933 * 3232;
    uint32_t value_shape = 18933 * 4064;
    uint32_t seqOffset_shape = 1;

    size_t keyCache_size = keyCache_shape * sizeof(int8_t);
    size_t valueCache_size = valueCache_shape * sizeof(int8_t);
    size_t blockTables_size = blockTables_shape * sizeof(int32_t);
    size_t seqLens_size = seqLens_shape * sizeof(int32_t);
    size_t key_size = key_shape * sizeof(int8_t);
    size_t value_size = value_shape * sizeof(int8_t);
    size_t seqOffset_size = seqOffset_shape * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GatherPaKvCacheTilingDataTest);

    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCache_size);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCache_size);
    uint8_t* blockTables = (uint8_t*)AscendC::GmAlloc(blockTables_size);
    uint8_t* seqLens = (uint8_t*)AscendC::GmAlloc(seqLens_size);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* seqOffset = (uint8_t*)AscendC::GmAlloc(seqOffset_size);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* valueOut = (uint8_t*)AscendC::GmAlloc(value_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    GatherPaKvCacheTilingDataTest* tilingDatafromBin = reinterpret_cast<GatherPaKvCacheTilingDataTest*>(tiling);

    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->numTokens = 36;
    tilingDatafromBin->numblkTabCol = 8;
    tilingDatafromBin->tokenSizeK = 3232;
    tilingDatafromBin->tokenSizeV = 4064;
    tilingDatafromBin->typeByte = 1;
    tilingDatafromBin->hasSeqStarts = false;
    tilingDatafromBin->isSeqLensCumsum = false;
    tilingDatafromBin->kCacheBlockStride = 0;
    tilingDatafromBin->vCacheBlockStride = 0;

    ICPU_SET_TILING_KEY(577);
    ICPU_RUN_KF(gather_pa_kv_cache, blockDim, keyCache, valueCache, blockTables, seqLens, key, value, seqOffset, keyOut,
                valueOut, workspace, (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(blockTables);
    AscendC::GmFree(seqLens);
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(seqOffset);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(valueOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(gather_pa_kv_cache_test, test_tiling_key_619)
{
    uint32_t keyCache_shape = 128 * 128 * 16 * 144;
    uint32_t valueCache_shape = 128 * 128 * 16 * 128;
    uint32_t blockTables_shape = 16 * 12;
    uint32_t seqLens_shape = 16;
    uint32_t key_shape = 8931 * 16 * 144;
    uint32_t value_shape = 8931 * 16 * 128;
    uint32_t seqOffset_shape = 16;

    size_t keyCache_size = keyCache_shape * sizeof(half);
    size_t valueCache_size = valueCache_shape * sizeof(half);
    size_t blockTables_size = blockTables_shape * sizeof(int32_t);
    size_t seqLens_size = seqLens_shape * sizeof(int32_t);
    size_t key_size = key_shape * sizeof(half);
    size_t value_size = value_shape * sizeof(half);
    size_t seqOffset_size = seqOffset_shape * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GatherPaKvCacheTilingDataTest);

    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCache_size);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCache_size);
    uint8_t* blockTables = (uint8_t*)AscendC::GmAlloc(blockTables_size);
    uint8_t* seqLens = (uint8_t*)AscendC::GmAlloc(seqLens_size);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* seqOffset = (uint8_t*)AscendC::GmAlloc(seqOffset_size);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* valueOut = (uint8_t*)AscendC::GmAlloc(value_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    GatherPaKvCacheTilingDataTest* tilingDatafromBin = reinterpret_cast<GatherPaKvCacheTilingDataTest*>(tiling);

    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->numTokens = 16;
    tilingDatafromBin->numblkTabCol = 12;
    tilingDatafromBin->tokenSizeK = 16 * 144;
    tilingDatafromBin->tokenSizeV = 16 * 128;
    tilingDatafromBin->typeByte = 2;
    tilingDatafromBin->hasSeqStarts = false;
    tilingDatafromBin->isSeqLensCumsum = false;
    tilingDatafromBin->kCacheBlockStride = 0;
    tilingDatafromBin->vCacheBlockStride = 0;

    ICPU_SET_TILING_KEY(619);
    ICPU_RUN_KF(gather_pa_kv_cache, blockDim, keyCache, valueCache, blockTables, seqLens, key, value, seqOffset, keyOut,
                valueOut, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(blockTables);
    AscendC::GmFree(seqLens);
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(seqOffset);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(valueOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(gather_pa_kv_cache_test, test_tiling_key_618)
{
    uint32_t keyCache_shape = 128 * 128 * 16 * 144;
    uint32_t valueCache_shape = 128 * 128 * 16 * 128;
    uint32_t blockTables_shape = 16 * 12;
    uint32_t seqLens_shape = 16;
    uint32_t key_shape = 9547 * 16 * 144;
    uint32_t value_shape = 9547 * 16 * 128;
    uint32_t seqOffset_shape = 16;

    size_t keyCache_size = keyCache_shape * sizeof(int8_t);
    size_t valueCache_size = valueCache_shape * sizeof(int8_t);
    size_t blockTables_size = blockTables_shape * sizeof(int32_t);
    size_t seqLens_size = seqLens_shape * sizeof(int32_t);
    size_t key_size = key_shape * sizeof(int8_t);
    size_t value_size = value_shape * sizeof(int8_t);
    size_t seqOffset_size = seqOffset_shape * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GatherPaKvCacheTilingDataTest);

    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCache_size);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCache_size);
    uint8_t* blockTables = (uint8_t*)AscendC::GmAlloc(blockTables_size);
    uint8_t* seqLens = (uint8_t*)AscendC::GmAlloc(seqLens_size);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* seqOffset = (uint8_t*)AscendC::GmAlloc(seqOffset_size);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* valueOut = (uint8_t*)AscendC::GmAlloc(value_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;

    GatherPaKvCacheTilingDataTest* tilingDatafromBin = reinterpret_cast<GatherPaKvCacheTilingDataTest*>(tiling);

    tilingDatafromBin->blockSize = 128;
    tilingDatafromBin->numTokens = 16;
    tilingDatafromBin->numblkTabCol = 12;
    tilingDatafromBin->tokenSizeK = 2304;
    tilingDatafromBin->tokenSizeV = 2048;
    tilingDatafromBin->typeByte = 1;
    tilingDatafromBin->hasSeqStarts = false;
    tilingDatafromBin->isSeqLensCumsum = false;
    tilingDatafromBin->kCacheBlockStride = 0;
    tilingDatafromBin->vCacheBlockStride = 0;

    ICPU_SET_TILING_KEY(618);
    ICPU_RUN_KF(gather_pa_kv_cache, blockDim, keyCache, valueCache, blockTables, seqLens, key, value, seqOffset, keyOut,
                valueOut, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(blockTables);
    AscendC::GmFree(seqLens);
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(seqOffset);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(valueOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 非 32B 对齐尾字节守护：blockSize=1、INT8 head_size=17/33，原实现会整 token 未写/丢尾 1B
TEST_F(gather_pa_kv_cache_test, test_nd_i8_unaligned_tail)
{
    const uint32_t blockSize = 1;
    const uint32_t numTokens = 1;
    const uint32_t tokenSizeK = 17;
    const uint32_t tokenSizeV = 33;

    size_t keyCache_size = blockSize * tokenSizeK;
    size_t valueCache_size = blockSize * tokenSizeV;
    size_t key_size = numTokens * tokenSizeK;
    size_t value_size = numTokens * tokenSizeV;
    size_t blockTables_size = numTokens * sizeof(int32_t);
    size_t seqLens_size = numTokens * sizeof(int32_t);
    size_t seqOffset_size = sizeof(int32_t);
    size_t tiling_data_size = sizeof(GatherPaKvCacheTilingDataTest);

    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCache_size);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCache_size);
    uint8_t* blockTables = (uint8_t*)AscendC::GmAlloc(blockTables_size);
    uint8_t* seqLens = (uint8_t*)AscendC::GmAlloc(seqLens_size);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* seqOffset = (uint8_t*)AscendC::GmAlloc(seqOffset_size);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* valueOut = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    for (size_t i = 0; i < keyCache_size; i++) {
        keyCache[i] = static_cast<uint8_t>(i * 7 + 3);
    }
    for (size_t i = 0; i < valueCache_size; i++) {
        valueCache[i] = static_cast<uint8_t>(i * 11 + 5);
    }
    memset(keyOut, 0xE7, key_size);
    memset(valueOut, 0xE7, value_size);
    reinterpret_cast<int32_t*>(blockTables)[0] = 0;
    reinterpret_cast<int32_t*>(seqLens)[0] = 1;

    GatherPaKvCacheTilingDataTest* tilingDatafromBin = reinterpret_cast<GatherPaKvCacheTilingDataTest*>(tiling);
    tilingDatafromBin->blockSize = blockSize;
    tilingDatafromBin->numTokens = numTokens;
    tilingDatafromBin->numblkTabCol = 1;
    tilingDatafromBin->tokenSizeK = tokenSizeK;
    tilingDatafromBin->tokenSizeV = tokenSizeV;
    tilingDatafromBin->typeByte = 1;
    tilingDatafromBin->hasSeqStarts = false;
    tilingDatafromBin->isSeqLensCumsum = false;
    tilingDatafromBin->kCacheBlockStride = 0;
    tilingDatafromBin->vCacheBlockStride = 0;

    ICPU_SET_TILING_KEY(618);
    ICPU_RUN_KF(gather_pa_kv_cache, 1, keyCache, valueCache, blockTables, seqLens, key, value, seqOffset, keyOut,
                valueOut, workspace, (uint8_t*)(tilingDatafromBin));

    uint32_t keyMis = 0;
    uint32_t valueMis = 0;
    for (size_t i = 0; i < key_size; i++) {
        if (keyOut[i] != keyCache[i]) {
            keyMis++;
        }
    }
    for (size_t i = 0; i < value_size; i++) {
        if (valueOut[i] != valueCache[i]) {
            valueMis++;
        }
    }
    EXPECT_EQ(keyMis, 0u);
    EXPECT_EQ(valueMis, 0u);

    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(blockTables);
    AscendC::GmFree(seqLens);
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(seqOffset);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(valueOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// 尾块 + 非 32B 对齐：blockSize=16、20 token、token 34B/38B（等价 fp16 head_size=17/19 的字节视角），
// 覆盖尾块截断与尾字节补齐（UT 只实例化 int8 分支，字节布局与 2 字节 dtype 等价）
TEST_F(gather_pa_kv_cache_test, test_nd_unaligned_tail_block)
{
    const uint32_t blockSize = 16;
    const uint32_t numTokens = 1;
    const uint32_t seqLen = 20;
    const uint32_t numBlocks = 2;
    const uint32_t tokenSizeK = 34;
    const uint32_t tokenSizeV = 38;

    size_t keyCache_size = numBlocks * blockSize * tokenSizeK;
    size_t valueCache_size = numBlocks * blockSize * tokenSizeV;
    size_t key_size = seqLen * tokenSizeK;
    size_t value_size = seqLen * tokenSizeV;
    size_t blockTables_size = numTokens * 2 * sizeof(int32_t);
    size_t seqLens_size = numTokens * sizeof(int32_t);
    size_t seqOffset_size = sizeof(int32_t);
    size_t tiling_data_size = sizeof(GatherPaKvCacheTilingDataTest);

    uint8_t* keyCache = (uint8_t*)AscendC::GmAlloc(keyCache_size);
    uint8_t* valueCache = (uint8_t*)AscendC::GmAlloc(valueCache_size);
    uint8_t* blockTables = (uint8_t*)AscendC::GmAlloc(blockTables_size);
    uint8_t* seqLens = (uint8_t*)AscendC::GmAlloc(seqLens_size);
    uint8_t* key = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* value = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* seqOffset = (uint8_t*)AscendC::GmAlloc(seqOffset_size);
    uint8_t* keyOut = (uint8_t*)AscendC::GmAlloc(key_size);
    uint8_t* valueOut = (uint8_t*)AscendC::GmAlloc(value_size);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    for (size_t i = 0; i < keyCache_size; i++) {
        keyCache[i] = static_cast<uint8_t>(i * 7 + 3);
    }
    for (size_t i = 0; i < valueCache_size; i++) {
        valueCache[i] = static_cast<uint8_t>(i * 11 + 5);
    }
    memset(keyOut, 0xE7, key_size);
    memset(valueOut, 0xE7, value_size);
    int32_t* blkTab = reinterpret_cast<int32_t*>(blockTables);
    blkTab[0] = 1;
    blkTab[1] = 0;
    reinterpret_cast<int32_t*>(seqLens)[0] = seqLen;

    GatherPaKvCacheTilingDataTest* tilingDatafromBin = reinterpret_cast<GatherPaKvCacheTilingDataTest*>(tiling);
    tilingDatafromBin->blockSize = blockSize;
    tilingDatafromBin->numTokens = numTokens;
    tilingDatafromBin->numblkTabCol = 2;
    tilingDatafromBin->tokenSizeK = tokenSizeK;
    tilingDatafromBin->tokenSizeV = tokenSizeV;
    tilingDatafromBin->typeByte = 1;
    tilingDatafromBin->hasSeqStarts = false;
    tilingDatafromBin->isSeqLensCumsum = false;
    tilingDatafromBin->kCacheBlockStride = 0;
    tilingDatafromBin->vCacheBlockStride = 0;

    ICPU_SET_TILING_KEY(618);
    ICPU_RUN_KF(gather_pa_kv_cache, 1, keyCache, valueCache, blockTables, seqLens, key, value, seqOffset, keyOut,
                valueOut, workspace, (uint8_t*)(tilingDatafromBin));

    std::vector<uint8_t> expectK(key_size);
    std::vector<uint8_t> expectV(value_size);
    for (uint32_t t = 0; t < seqLen; t++) {
        int32_t blockId = blkTab[t / blockSize];
        uint32_t slot = t % blockSize;
        memcpy(expectK.data() + t * tokenSizeK, keyCache + blockId * blockSize * tokenSizeK + slot * tokenSizeK,
               tokenSizeK);
        memcpy(expectV.data() + t * tokenSizeV, valueCache + blockId * blockSize * tokenSizeV + slot * tokenSizeV,
               tokenSizeV);
    }
    EXPECT_EQ(memcmp(keyOut, expectK.data(), key_size), 0);
    EXPECT_EQ(memcmp(valueOut, expectV.data(), value_size), 0);

    AscendC::GmFree(keyCache);
    AscendC::GmFree(valueCache);
    AscendC::GmFree(blockTables);
    AscendC::GmFree(seqLens);
    AscendC::GmFree(key);
    AscendC::GmFree(value);
    AscendC::GmFree(seqOffset);
    AscendC::GmFree(keyOut);
    AscendC::GmFree(valueOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
