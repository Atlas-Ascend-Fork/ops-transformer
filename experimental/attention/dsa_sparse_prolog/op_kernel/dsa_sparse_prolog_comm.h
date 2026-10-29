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
 * \file dsa_sparse_prolog_comm.h
 * \brief
 */

#ifndef DSA_SPARSE_PROLOG_COMM_H
#define DSA_SPARSE_PROLOG_COMM_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;
using AscendC::AIC;
using AscendC::AIV;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::SetFlag;
using AscendC::ShapeInfo;
using AscendC::SoftmaxConfig;
using AscendC::WaitFlag;
using matmul::Matmul;
using matmul::MatmulType;

#ifdef ENABLE_DUMP_DATA
#define DO_DUMP_DATA(srcTensor, id, len) AscendC::DumpTensor(srcTensor, id, len)
#else
#define DO_DUMP_DATA(srcTensor, id, len)
#endif

#define USE_MM_API_MLAP

// mte2 <> mte1
#define A_EVENT0 EVENT_ID4
#define A_EVENT1 EVENT_ID5
#define B_EVENT0 EVENT_ID6
#define B_EVENT1 EVENT_ID7

// m <> mte1
#define L0A_EVENT0 EVENT_ID3
#define L0A_EVENT1 EVENT_ID4
#define L0B_EVENT0 EVENT_ID5
#define L0B_EVENT1 EVENT_ID6

// fix <> m
#define L0C_EVENT0 EVENT_ID3
#define L0C_EVENT1 EVENT_ID4

#define L1_A_SIZE (128 * 1024) // Ping buffer size
#define L1_B_SIZE (128 * 1024)
#define L0A_PP_SIZE (32 * 1024)
#define L0B_PP_SIZE (32 * 1024)
#define L0C_PP_SIZE (64 * 1024)

template <typename T>
struct MLAPType {
    using mmInputType = T; // tokenX的类型与weight的类型一致
    using mmQcQrInputType = T;
    using mmCqOutputType = T;    // matmul计算Cq的输出类型
    using mmCkvKrOutputType = T; // matmul计算CkvKr的输出类型
    using mmQcQrOutputType = T;  // matmul计算QcQr的输出类型
    using rmsNormGammaType = T;  // gamma的输入类型
    using rmsNormComputType = float;
    using rmsNormCqOutputType = T;
    using scaleQcQrComputType = float;
    using scaleQcQrOutputType = T;
    using rmsNormCkvOutputType = T;
    using ropeSinCosType = T; // sin cos的输入类型
    using ropeComputType = float;
    using ropeOutputType = T;
    using cacheType = T; // q_nope q_pe ct_kv k_pe 的输出类型一致
};

using MMParams = struct MMParams {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t orgM;
    uint32_t orgN;
    uint32_t orgKa;
    uint32_t orgKb;
    uint32_t orgKc;
    uint32_t baseM;
    uint32_t baseN;
    uint32_t baseK;
    uint32_t stepK;
    uint32_t needSetOrgShape;
    uint32_t
        bLayout; // B 侧权重布局: 0=NZ[K,N], 1=ND[K,N], 2=NZ[N,K], 3=ND[N,K]; bit0=ND, bit1=[N,K] (转置由 kernel 吸收)
};

// bLayout 布局枚举, 与 host 侧 weightLayout 一致
constexpr uint32_t B_LAYOUT_NZ_KN = 0;
constexpr uint32_t B_LAYOUT_ND_KN = 1;
constexpr uint32_t B_LAYOUT_NZ_NK = 2;
constexpr uint32_t B_LAYOUT_ND_NK = 3;

struct MMBufParams {
    uint32_t aL1BufIter = 0;
    uint32_t bL1BufIter = 0;
    TBuffAddr aL1BufAddr;
    TBuffAddr bL1BufAddr;
};

constexpr MatmulConfig CFG_MDL_EXCEED_INIT{.doNorm = false,
                                           .doBasicBlock = false,
                                           .doMultiDataLoad = true,
                                           .basicM = 0,
                                           .basicN = 0,
                                           .basicK = 0,
                                           .intrinsicsCheck = true,
                                           .isNBatch = false,
                                           .enVecND2NZ = false,
                                           .doSpecialBasicBlock = false,
                                           .doMTE2Preload = false,
                                           .singleCoreM = 0,
                                           .singleCoreN = 0,
                                           .singleCoreK = 0,
                                           .stepM = 0,
                                           .stepN = 0,
                                           .baseMN = 0,
                                           .singleCoreMN = 0,
                                           .enUnitFlag = false,
                                           .isPerTensor = false,
                                           .hasAntiQuantOffset = false,
                                           .doIBShareNorm = false,
                                           .doSpecialMDL = false,
                                           .enableInit = false,
                                           .batchMode = BatchMode::NONE,
                                           .enableEnd = false,
                                           .enableGetTensorC = false,
                                           .enableSetOrgShape = true,
                                           .enableSetBias = false,
                                           .enableSetTail = true,
                                           .enableQuantVector = false,
                                           .enableSetDefineData = true,
                                           .iterateMode = IterateMode::ITERATE_MODE_ALL};
#endif
