/**
 * copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_ARCH92_QBSA_MATMUL_PARAMS_H
#define QBSA_ARCH92_QBSA_MATMUL_PARAMS_H

namespace fa_base_matmul {
constexpr uint32_t UNITFLAG_ENABLE = 2;
constexpr uint32_t UNITFLAG_EN_OUTER_LAST = 3;
static constexpr uint32_t FP16_ONE_FRACTAL_ELEMENT = 16; // 一个分形512B,16*16个fp16
static constexpr uint32_t INT4_ONE_FRACTAL_ELEMENT = 64; // 一个分形512B,16*64个fp16
static constexpr uint32_t ONE_FRACTAL_H_ELEMENT = 16;    // 一个分形512B,height方向为16个element
static constexpr uint32_t ONE_FRACTAL_W_BYTE = 32;       // 一个分形512B,weight方向为32B
static constexpr uint32_t LOAD3D_L1W_SIZE = 16;
static constexpr uint8_t LOAD3D_STRIDE_W = 1;
static constexpr uint8_t LOAD3D_STRIDE_H = 1;
static constexpr uint8_t LOAD3D_FILTER_W = 1;
static constexpr uint8_t LOAD3D_FILTER_H = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_W = 1;
static constexpr uint8_t LOAD3D_DILA_FILTER_H = 1;
static constexpr uint32_t K_STEP_ALIGN_BASE = 2;
static constexpr uint32_t M_STEP_ALIGN_BASE = 2;
static constexpr uint32_t MX_FP4_PTG_PCG_SCALE_PARAM = 32; // mxfp4场景随路伪量化中32个元素共用一个scale
static constexpr uint32_t HI_FP4_PTG_PCG_SCALE_PARAM = 64; // hifp4场景随路伪量化中64个元素共用一个scale

struct MMParam {
    uint32_t singleM;
    uint32_t singleN;
    uint32_t singleK;
    bool isLeftTranspose;
    bool isRightTranspose;
    bool cmatrixInitVal = true;
    bool isOutKFisrt = true; // 默认值为true， true：在L1切K轴的场景中，表示首轮K
    uint32_t unitFlag = 0;   // 0：disable: 不配置unitFlag
                           // 2：enable: 行为在切K接口中（MatmulK），会将mmadParams.unitFlag设置为 0b10
                           // 3：enable: 行为在切K接口中（MatmulK），在k的最后一轮循环，会将mmadParams.unitFlag设置为
                           // 0b11 外部使用时，在外层k循环的最后一轮将该参数配置为3
    uint32_t realM = 0; // bmm2以s1realsize为M轴，不赋值时不影响现有代码逻辑
};

__aicore__ inline MMParam MxMakeMMParam(uint32_t singleM, uint32_t singleN, uint32_t singleK, bool isLeftTranspose,
                                        bool isRightTranspose, bool cmatrixInitVal = true, bool isOutKFisrt = true,
                                        uint32_t unitFlag = 0, uint32_t realM = 0)
{
    // MX 调用点使用的小型构造函数，用于让 matmul 维度和转置标志更易读。
    // 在 MX 路径中，这些标志描述的是 LoadDataToL0*Mx 消费的 L1 视图，
    // 并不表示额外启动一个显式转置 kernel。
    MMParam param;
    param.singleM = singleM;
    param.singleN = singleN;
    param.singleK = singleK;
    param.isLeftTranspose = isLeftTranspose;
    param.isRightTranspose = isRightTranspose;
    param.cmatrixInitVal = cmatrixInitVal;
    param.isOutKFisrt = isOutKFisrt;
    param.unitFlag = unitFlag;
    param.realM = realM;
    return param;
}

enum class ABLayout {
    MK = 0,
    KM = 1,
    KN = 2,
    NK = 3,
};

template <typename T>
__aicore__ inline T AlignUp(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}
} // namespace fa_base_matmul

#endif // QBSA_ARCH92_QBSA_MATMUL_PARAMS_H
