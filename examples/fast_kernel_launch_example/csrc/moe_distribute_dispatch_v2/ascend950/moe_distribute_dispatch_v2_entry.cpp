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
 * \file moe_distribute_dispatch_v2_entry.cpp
 * \brief A5 (ascend950/arch35) <<<>>> direct-launch entry with SuperKernel support.
 *        Kernel implementations are referenced in place from mc2 op_kernel/arch35
 *        (no local copies), resolved via the repo-root include path.
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"

#include "mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2.h"
#include "mc2/moe_distribute_dispatch_v2/op_kernel/moe_distribute_dispatch_v2_tiling.h"

#include "moe_distribute_dispatch_v2_entry.h"

using namespace Mc2Kernel;
using namespace AscendC;

/*
 * A5 example 本地补充（不引入 mc2 tiling_key_decl.h）：
 * tiling_key_decl.h 内含 host-only TPL 头 template_argument.h（FastEncodeTilingKeyDirect 用 printf），
 * 在含 __aicore__ printf 的 device TU 中展开会报 "call to __aicore__ function from __host__ function"，
 * 故 device 侧不复用该头，仅镜像其 TILINGKEY_* 声明值（与 ops 构建框架注入的
 * GET_TILING_DATA_WITH_STRUCT 语义等价：从 tilingGM 读出本地 TilingData 结构体）。
 */
#define GET_TILING_DATA_WITH_STRUCT(TilingDataStru, tilingData, tilingArg) \
    TilingDataStru tilingData; \
    { \
        const __gm__ uint32_t* __gm_src = reinterpret_cast<const __gm__ uint32_t*>(tilingArg); \
        uint32_t* __local_dst = reinterpret_cast<uint32_t*>(&tilingData); \
        constexpr uint32_t __words = sizeof(TilingDataStru) / sizeof(uint32_t); \
        for (uint32_t __i = 0; __i < __words; ++__i) { \
            __local_dst[__i] = __gm_src[__i]; \
        } \
    }

/*
 * A5 tilingkey说明（example自有编码，非mc2 TPL注册键，仅在本直调样例内使用）
 * 4位的十进制数，基值20000
 * 第1位（个位）：quantMode:
 *     0: 不量化, 2: perToken动态量化(int8输出)
 * 第2位（十位）：x输入类型:
 *     0: float16, 1: bfloat16
 * 第3位（百位）：是否有smoothScale(scales):
 *     0: 无, 1: 有
 * 仅支持 MTE 通信（非fullmesh）
 */

template <typename XType, typename ExpandXOutType, int32_t QuantMode, bool IsSmoothScaleExist>
__attribute__((always_inline)) __aicore__ __inline__ void moe_distribute_dispatch_v2_a5_impl(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR assistInfoOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM, GM_ADDR mc2Context)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
    TPipe pipe;
    // A3 式直接传值：mc2Context 为 HcclCombinOpParam 布局的上下文设备地址
    GM_ADDR contextGM0 = mc2Context;
    int64_t oriOverflowMode = AscendC::GetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>();
    // 仅支持 MTE 非fullmesh 模板
    MoeDistributeDispatchV2<Mc2Kernel::HcclContextHolder, XType, ExpandXOutType, QuantMode, IsSmoothScaleExist> op;
    op.Init(contextGM0, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut,
            dynamicScalesOut, assistInfoOut, expandScalesOut, expertTokenNumsOut, epSendCountsOut, workspaceGM, &pipe,
            &tilingData);
    op.Process();
    if ASCEND_IS_AIV {
        PipeBarrier<PIPE_ALL>();
    }
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(oriOverflowMode);
    pipe.DestroyWithoutPipeAll();
    return;
}

__aicore__ inline void moe_dispatch_a5_switch(int32_t tilingKey, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
                                              GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo,
                                              GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                              GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut,
                                              GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut,
                                              GM_ADDR workspaceGM, GM_ADDR tilingGM, GM_ADDR mc2Context)
{
    switch (tilingKey) {
        case 20000: // float16, 不量化, 无scales
            moe_distribute_dispatch_v2_a5_impl<float16_t, float16_t, Mc2Kernel::UNQUANT, false>(
                x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut,
                dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut,
                workspaceGM, tilingGM, mc2Context);
            break;

        case 20010: // bfloat16, 不量化, 无scales
            moe_distribute_dispatch_v2_a5_impl<bfloat16_t, bfloat16_t, Mc2Kernel::UNQUANT, false>(
                x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut,
                dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut,
                workspaceGM, tilingGM, mc2Context);
            break;

        case 20002: // float16, perToken动态量化(int8输出), 无smoothScale
            moe_distribute_dispatch_v2_a5_impl<float16_t, int8_t, Mc2Kernel::PERTOKEN_DYNAMIC_QUANT, false>(
                x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut,
                dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut,
                workspaceGM, tilingGM, mc2Context);
            break;

        case 20012: // bfloat16, perToken动态量化(int8输出), 无smoothScale
            moe_distribute_dispatch_v2_a5_impl<bfloat16_t, int8_t, Mc2Kernel::PERTOKEN_DYNAMIC_QUANT, false>(
                x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut,
                dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut,
                workspaceGM, tilingGM, mc2Context);
            break;

        case 20102: // float16, perToken动态量化(int8输出), 有smoothScale
            moe_distribute_dispatch_v2_a5_impl<float16_t, int8_t, Mc2Kernel::PERTOKEN_DYNAMIC_QUANT, true>(
                x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut,
                dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut,
                workspaceGM, tilingGM, mc2Context);
            break;

        case 20112: // bfloat16, perToken动态量化(int8输出), 有smoothScale
            moe_distribute_dispatch_v2_a5_impl<bfloat16_t, int8_t, Mc2Kernel::PERTOKEN_DYNAMIC_QUANT, true>(
                x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut,
                dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut,
                workspaceGM, tilingGM, mc2Context);
            break;

        default:
            AscendC::PRINTF("moe_distribute_dispatch_v2_a5 Error: invalid tilingKey = %d\n", tilingKey);
            break;
    }
}

extern "C" __global__ __aicore__ void moe_distribute_dispatch_v2_generic(
    int32_t tilingKey, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR assistInfoOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut,
    GM_ADDR workspaceGM, GM_ADDR tilingGM, GM_ADDR mc2Context)
{
    // 对齐 A5-MTE：SetHcclContext<0>(mc2Context) 供经 GetHcclContext 读取同一 HcclCombinOpParam 上下文
    AscendC::SetHcclContext<0>(mc2Context);
    moe_dispatch_a5_switch(tilingKey, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo,
                           expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                           tpSendCountsOut, expandScalesOut, workspaceGM, tilingGM, mc2Context);
}

// SK孪生模板函数：从SkArgs解包参数后复用moe_dispatch_a5_switch，与generic走相同分发逻辑
template <int32_t split_num>
__sk__ __vector__ void moe_distribute_dispatch_v2_generic_sk(MoeDistributeDispatchV2SkArgs* args)
{
    int32_t tilingKey = args->tilingKey;
    GM_ADDR x = args->x;
    GM_ADDR expertIds = args->expertIds;
    GM_ADDR scales = args->scales;
    GM_ADDR xActiveMask = args->xActiveMask;
    GM_ADDR expertScales = args->expertScales;
    GM_ADDR elasticInfo = args->elasticInfo;
    GM_ADDR performanceInfo = args->performanceInfo;
    GM_ADDR expandXOut = args->expandXOut;
    GM_ADDR dynamicScalesOut = args->dynamicScalesOut;
    GM_ADDR assistInfoOut = args->assistInfoOut;
    GM_ADDR expertTokenNumsOut = args->expertTokenNumsOut;
    GM_ADDR epSendCountsOut = args->epSendCountsOut;
    GM_ADDR tpSendCountsOut = args->tpSendCountsOut;
    GM_ADDR expandScalesOut = args->expandScalesOut;
    GM_ADDR workspaceGM = args->workspaceGM;
    GM_ADDR tilingGM = args->tilingGM;
    GM_ADDR mc2Context = args->mc2Context;

    // 对齐 A5-MTE：SetHcclContext<0>(mc2Context) 供经 GetHcclContext 读取同一 HcclCombinOpParam 上下文
    AscendC::SetHcclContext<0>(mc2Context);
    moe_dispatch_a5_switch(tilingKey, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo,
                           expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                           tpSendCountsOut, expandScalesOut, workspaceGM, tilingGM, mc2Context);
}

SK_BIND(moe_distribute_dispatch_v2_generic, 4, moe_distribute_dispatch_v2_generic_sk<0>,
        moe_distribute_dispatch_v2_generic_sk<1>, moe_distribute_dispatch_v2_generic_sk<2>,
        moe_distribute_dispatch_v2_generic_sk<3>);

// <<<>>>调用函数
void moe_distribute_dispatch_v2_entry(int32_t tilingKey, uint32_t blockDim, void* stream, GM_ADDR x, GM_ADDR expertIds,
                                      GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo,
                                      GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
                                      GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut,
                                      GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM,
                                      GM_ADDR tilingGM, GM_ADDR mc2Context)
{
    moe_distribute_dispatch_v2_generic<<<blockDim, nullptr, stream>>>(
        tilingKey, x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut,
        dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut,
        workspaceGM, tilingGM, mc2Context);
}
