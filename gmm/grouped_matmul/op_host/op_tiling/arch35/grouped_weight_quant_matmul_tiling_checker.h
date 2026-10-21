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
 * \file grouped_weight_quant_matmul_tiling_checker.h
 * \brief
 */

#pragma once

#include "../../grouped_matmul_host_util.h"
#include "grouped_weight_quant_matmul_tiling_tool.h"

using optiling::GMMWeightQuantTilingTool::GMMWeightQuantInputInfo;
using optiling::GMMWeightQuantTilingTool::GMMWeightQuantInputNum;
using optiling::GMMWeightQuantTilingTool::MKNDimInfo;

namespace optiling {
class GMMWeightQuantTilingChecker {
public:
    explicit GMMWeightQuantTilingChecker(gert::TilingContext* context, const GMMWeightQuantInputInfo& inputParams,
                                         const GMMWeightQuantInputNum& inputNum)
        : context_(context),
          inputParams_(inputParams),
          inputNum_(inputNum) {};
    ~GMMWeightQuantTilingChecker() {};
    bool CheckGroupedMatmulWeightQuant();

private:
    bool CheckUnsupportedDataFlow() const;
    bool CheckGroupScenario() const;
    bool CheckTensorListSize() const;
    bool CheckEmptyTensor() const;
    bool CheckInputDtype() const;
    bool CheckEveryTensor();
    bool CheckNotNull(size_t tensorIdx) const;
    bool CheckNotNullDesc(size_t inputIdx, size_t tensorIdx) const;
    bool CheckEveryTensorDtype(size_t tensorIdx) const;
    bool CheckTensorDtype(size_t inputIdx, size_t tensorIdx, const ge::DataType expectedDtype,
                          const std::string& tensorType) const;
    bool CheckEveryTensorShape(size_t tensorIdx);
    bool CheckEveryTensorDim(size_t tensorIdx) const;
    bool CheckTensorDimSingleXSingleWeightSingleY(size_t tensorIdx) const;
    bool CheckTensorDim(size_t inputIdx, size_t tensorIdx, size_t targetDim, const std::string& tensorType) const;
    bool GetMKN(size_t tensorIdx);
    bool CheckXAndWeightShape(size_t tensorIdx) const;
    bool CheckTensorShape(size_t inputIdx, size_t tensorIdx, const std::string& tensorType) const;
    bool CheckGroupSize(const gert::Shape& scaleShape, const std::string& tensorType) const;
    bool CheckPerTokenScale() const;

    std::string GetDataFlowString() const;

    gert::TilingContext* context_ = nullptr;
    GMMWeightQuantInputInfo inputParams_;
    GMMWeightQuantInputNum inputNum_;
    MKNDimInfo dimInfo_;
};
} // namespace optiling
