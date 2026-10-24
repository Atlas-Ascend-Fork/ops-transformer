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
 * \file mega_moe_peermem.h
 * \brief peermem 跨卡对称窗口的布局与尺寸计算。
 *        窗口各区尺寸一律按全卡一致的 numMaxTokensPerRank 上界推导（可变 bs 下各卡真实 bs 可不同，
 *        但跨卡读写共用同一套偏移，布局必须逐字节一致）。
 *        CCL 通信布局公式在公共头维护，本文件绑定核内区域地址。
 */

#ifndef MEGA_MOE_PEERMEM_H
#define MEGA_MOE_PEERMEM_H

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__) || defined(__DAV_310R6_CUBE__) || \
    defined(__DAV_310R6_VEC__) || defined(__DAV_920R1_CUBE__) || defined(__DAV_920R1_VEC__) || defined(__NPU_ARCH__)
#ifndef MEGA_MOE_IS_KERNEL_COMPILE
#define MEGA_MOE_IS_KERNEL_COMPILE
#endif
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#else
#define GM_ADDR uint8_t*
// host 侧命名空间级自由函数需显式 inline 保 ODR；类内成员函数本就隐式 inline，加之无副作用。
// host 侧自洽：Ops::Base::CeilAlign/CeilDiv 的提供方，不再依赖包含方的 include 顺序。
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/bfloat16.h"
#endif
#include "../mega_moe_tiling.h"
#include "mega_moe_constants.h"
#include "mega_moe_memory_layout.h"

namespace MegaMoeImpl {

// 纯布局头提供共享尺寸常量及计数跨度；这里补齐两份架构 workspace 共用的 tiling 适配。
// Arch92 保留自己的完整布局生成逻辑，同时通过原有 guard 避免重复声明这些共享值。
#ifndef MEGA_MOE_WORKSPACE_VALUE_DEFINITIONS
#define MEGA_MOE_WORKSPACE_VALUE_DEFINITIONS

HOST_DEVICE bool IsW4GmmMode(uint8_t gmmMode)
{
    return gmmMode == GMM_MODE_A8W4_NZ || gmmMode == GMM_MODE_A4W4_ND || gmmMode == GMM_MODE_A4W4_NZ;
}

HOST_DEVICE bool IsA4W4GmmMode(uint8_t gmmMode)
{
    return gmmMode == GMM_MODE_A4W4_ND || gmmMode == GMM_MODE_A4W4_NZ;
}

HOST_DEVICE uint32_t GetActivationOutputDivisor(uint8_t actMode)
{
    return actMode == static_cast<uint8_t>(MegaMoeActMode::GELU) ? 1U : ACTIVATION_N_HALF;
}

HOST_DEVICE int64_t GetActivationOutputWidth(const MegaMoeTilingData* tilingData)
{
    return static_cast<int64_t>(tilingData->compute.hiddenDim) / GetActivationOutputDivisor(tilingData->actMode);
}

#endif // MEGA_MOE_WORKSPACE_VALUE_DEFINITIONS

// Combine 输出和 Server 归约记录使用 BF16。
#ifdef MEGA_MOE_IS_KERNEL_COMPILE
constexpr int64_t SIZE_OF_BF16 = sizeof(bfloat16_t);
#else
constexpr int64_t SIZE_OF_BF16 = sizeof(Ops::Base::bfloat16);
#endif

// 每个 (localExpert, srcRank) 槽直接保存有序 topkIndex。
// 同一 token 的 topK expert id 不重复，因此单专家从一张源卡最多接收 numMaxTokensPerRank 个 index。
HOST_DEVICE int64_t CalcDispatchRouteIndexAlignSize(const MegaMoeTilingData* tilingData)
{
    return Ops::Base::CeilAlign(
        static_cast<int64_t>(tilingData->compute.numMaxTokensPerRank) *
            CalcTopkIndexTypeBytes(tilingData->compute.numMaxTokensPerRank, tilingData->compute.topK),
        ALIGN_32);
}

// GM 区域保存相对基址的偏移、字节数和初始化时计算的地址。
struct GmMemoryRegion {
    int64_t offset{-1};
    int64_t size{0};
    GM_ADDR ptr{nullptr};
};

// 主机与核内保留相同的区域描述，地址绑定不再重复推进偏移。
struct PeermemInfo : CclCommLayoutBase<GmMemoryRegion> {
    GM_ADDR cclComputePtr{nullptr}; // CCL 本地计算区起点，相对窗口基址按 512B 对齐。

    HOST_DEVICE PeermemInfo() = default;

    HOST_DEVICE explicit PeermemInfo(const CclCommLayoutParams& params, GM_ADDR windowBase = nullptr)
        : CclCommLayoutBase<GmMemoryRegion>(CalcCclCommLayout<GmMemoryRegion>(params))
    {
        BindRegion(windowBase, rankSync);
        BindRegion(windowBase, syncCount);
        BindRegion(windowBase, maskRecv);
        BindRegion(windowBase, expertCountRecv);
        BindRegion(windowBase, quantTokenScale);
        BindRegion(windowBase, dispatchReceive);
        BindRegion(windowBase, dispatchFlag);
        BindRegion(windowBase, sourceReady);
        BindRegion(windowBase, contributionReady);
        BindRegion(windowBase, combineSend);
        BindRegion(windowBase, contribution);
        BindRegion(windowBase, combineReceive);
        cclComputePtr = windowBase == nullptr ?
                            nullptr :
                            windowBase + MemoryLayoutDetail::CeilAlign(totalSize, MemoryLayoutDetail::ALIGN_512);
    }

private:
    HOST_DEVICE static void BindRegion(GM_ADDR base, GmMemoryRegion& region)
    {
        region.ptr = base == nullptr || region.offset < 0 ? nullptr : base + region.offset;
    }
};

} // namespace MegaMoeImpl

#endif // MEGA_MOE_PEERMEM_H
