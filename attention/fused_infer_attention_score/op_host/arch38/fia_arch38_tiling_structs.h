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
 * \file fia_arch38_tiling_structs.h
 * \brief FIA arch38 tiling 传参打包结构（内部工具层，不参与序列化）
 *
 * 本头文件声明仅用于在校验函数之间传递/聚合入参的纯结构体（PACacheDimCtx/PFAShapeDims/BmmShapeCtx/
 * SplitNBSeqCtx/PFAShapeSet 等），目的是减少成员函数形参个数，属于 host 实现内部的数据交换，
 * 不参与算子 tiling 数据的序列化（序列化契约见 fia_arch38_tiling_common.h）。
 */

#ifndef FIA_ARCH38_TILING_STRUCTS_H
#define FIA_ARCH38_TILING_STRUCTS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include "exe_graph/runtime/storage_shape.h"

namespace optiling {
namespace arch38 {

// PA（page attention）形状校验入参：打包 key/value 维度与 block 参数
struct PACacheDimCtx {
    const gert::StorageShape* keyShape = nullptr;
    const gert::StorageShape* valueShape = nullptr;
    size_t keyDim = 0;
    int64_t keyDim1 = 0;
    const int64_t* blockSize = nullptr;
    int64_t blockNumValid = 0;
    int64_t headNumRatio = 0;
};

// PA 每 batch 的 block 数统计结果
struct PABlockNumInfo {
    int64_t blockNumValid = 0;
    int32_t maxBlockNumPerBatch = 0;
};

// 形状信息六元组（uint 最终态）：SetShape 布局解析 + 校验后得到的 batch/head/seq/headSize/... 值，
// 是 host 侧形状校验与 tiling 计算的最终入参（与 int64 中间态 PFAShapeDims 配套使用）
struct PFAShapeInfo {
    uint32_t b = 0;
    uint64_t n = 0;
    uint64_t s = 0;
    uint32_t d = 0;
    uint64_t h = 0;
    uint32_t t = 0;
};

// SetShape 布局解析的 int64 中间态六元组：保留 int64 以承载中间/负值，校验后再 cast 到 PFAShapeInfo
struct PFAShapeDims {
    int64_t b = 0;
    int64_t n = 0;
    int64_t s = 0;
    int64_t d = 0;
    int64_t h = 0;
    int64_t t = 0;
};

// CV diff 场景的 s_outer/s_inner/softmax_s_outer 因子集
struct CVFactorParams {
    uint32_t sOuterFactor = 0;
    uint32_t sInnerFactor = 0;
    uint32_t softmaxSOuterFactor = 0;
};

// BMM1/BMM2 计算中的共享维度/空间入参
struct BmmShapeCtx {
    int64_t l1SizeRemain = 0;
    int64_t l0CSize = 0;
    uint32_t sOuterFactor = 0;
    uint32_t sInnerFactor = 0;
    uint32_t dSplitFactor = 0;
    bool autoBaseMNK = false;
};

// NBS 按核切分时的分段索引与循环推进状态（数组长度在调用方按 tilingElementArrayLen 初始化）
struct SplitNBSeqCtx {
    std::vector<uint32_t> coreSposEnd;
    std::vector<uint32_t> coreSposStart;
    std::vector<uint32_t> coreSidEnd;
    std::vector<uint32_t> coreSidStart;
    std::vector<uint32_t> coreNidEnd;
    std::vector<uint32_t> coreNidStart;
    std::vector<uint32_t> bnStartIdx;
    std::vector<int64_t> gS1StartIdx;
    int64_t curWight = 0;
    uint32_t curCore = 0;
    uint32_t tmpCoreNidEnd = 0;
    uint32_t tmpCoreSidEnd = 0;
    uint32_t tmpCoreSposEnd = 0;
    uint32_t sOuterSize = 0;
    uint32_t sInnerSize = 0;
    double coreWightTarget = 0;
    const std::vector<int64_t>* actualSeqLengths = nullptr;
    const std::vector<int64_t>* actualSeqLengthsKV = nullptr;
};

// 单属性校验阶段使用的四类 shape 引用集合
struct PFAShapeSet {
    PFAShapeInfo* query = nullptr;
    PFAShapeInfo* key = nullptr;
    PFAShapeInfo* value = nullptr;
    PFAShapeInfo* queryRope = nullptr;
};

// CheckMaskShape 的掩码形状解析结果（5 个输出打包，减少形参个数）
struct MaskShapeInfo {
    int64_t attenMaskBatch = 1;
    int64_t attenMaskS1 = 0;
    int64_t attenMaskS2 = 0;
    bool checkMask = false;
    std::string strMaskShape;
};

// CheckPerblockScaleShape 的 scale 形状校验入参打包
struct PerblockScaleCheckCtx {
    const gert::StorageShape* scaleShape = nullptr;
    uint32_t blockSize = 0;
    std::string paramName;
    std::string expectShape;
};

// GetAndCheckRopeShape 的源/rope 名称对（仅用于错误消息）
struct RopeNamePair {
    std::string srcName;
    std::string ropeName;
};

} // namespace arch38
} // namespace optiling

#endif // FIA_ARCH38_TILING_STRUCTS_H
