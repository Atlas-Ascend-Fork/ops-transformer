/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file allto_all_matmul_v2_fit_balance_tiling.h
 * \brief AlltoAllMatmulV2 fit-balance 引擎接入——标定数据与分派（算子接入唯一编辑面）
 */

#ifndef __ALLTO_ALL_MATMUL_V2_FIT_BALANCE_TILING_H__
#define __ALLTO_ALL_MATMUL_V2_FIT_BALANCE_TILING_H__

#pragma once

#include "apace/tiling/fit_balance_tiling/fit_balance_tiling.h"

namespace MC2Tiling {

// 实测发现 LocalMMPosition::None 场景的模型能更精确的模拟 precisionMode = 1 分支
const FitBalanceCalibration::DataflowShape A2A_DATAFLOWS[] = {
    {CommComputeOrder::COMM_FIRST, LocalMMPosition::NONE, false},  // precisionMode = 0 / 1
    {CommComputeOrder::COMM_FIRST, LocalMMPosition::FRONT, false}, // precisionMode = 2
};

const FitBalanceCalibration::CommFitRow A2A_COMM_ROWS[] = {
    {TilingCommEngine::AIV_URMA, 4, 6.18, 8.0, 5.29},
};

const FitBalanceCalibration::MatmulFit A2A_MM_FIT = {{256, 256, 128}};

inline bool GetA2aFitBalanceCut(uint64_t mPerRank, uint64_t nValue, uint64_t ka, uint64_t rankDim, uint32_t aicCoreNum,
                                uint32_t precisionMode, ge::DataType x1Dtype, uint64_t inputADtypeSize,
                                uint64_t outputDtypeSize, const char* commMode, uint64_t realBaseM, uint64_t realBaseN,
                                CommCutResult& cut)
{
    const int dataFlowMode = (precisionMode == 2U) ? 1 : 0;
    const TilingCommEngine commEngine = ParseCommEngine(commMode);
    FitBalanceCalibration::MatmulFit mmFit = A2A_MM_FIT;
    if (realBaseM > 0UL) {
        mmFit.baseBlock.baseM = realBaseM;
    }
    if (realBaseN > 0UL) {
        mmFit.baseBlock.baseN = realBaseN;
    }
    const FitBalanceCalibration calib = MakeFitBalanceCalibration(A2A_DATAFLOWS[dataFlowMode], A2A_COMM_ROWS, mmFit);
    if (!IsCommCalibrated(static_cast<uint32_t>(rankDim), commEngine, calib)) {
        return false;
    }
    FitBalanceTilingArgs args{};
    args.mValue = mPerRank;
    args.nValue = nValue;
    args.kValue = ka * rankDim;
    args.rankDim = static_cast<uint32_t>(rankDim);
    args.aicCoreNum = aicCoreNum;
    args.geAType = x1Dtype;
    args.inputADtypeSize = (x1Dtype == ge::DT_FLOAT4_E2M1) ? 0UL : inputADtypeSize;
    args.outputDtypeSize = outputDtypeSize;
    args.commEngine = commEngine;
    cut = FitBalanceCut(args, calib);
    return true;
}

} // namespace MC2Tiling

#endif // __ALLTO_ALL_MATMUL_V2_FIT_BALANCE_TILING_H__
