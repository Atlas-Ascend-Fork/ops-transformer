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
 * \file sinks_checker.cpp
 * \brief Checker for sinks parameter (文档参数名: sinks, 文档约束: LearnableSink参数组)
 */

#include <map>
#include <numeric>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../fa_tiling_info.h"
#include "sinks_checker_flash_attn.h"

namespace optiling {
namespace flash_attn {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35FA;

ge::graphStatus SinksChecker::CheckSinglePara(const FaTilingInfo& faInfo)
{
    if (!faInfo.sinksFlag) {
        return ge::GRAPH_SUCCESS;
    }

    auto& sinksTensor = faInfo.opParamInfo.sinks.tensor;
    OP_CHECK_IF(sinksTensor == nullptr, OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "sinks"), return ge::GRAPH_FAILED);

    const gert::CompileTimeTensorDesc* sinksDesc = faInfo.opParamInfo.sinks.desc;
    OP_CHECK_IF(sinksDesc == nullptr, OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "TensorDesc of sinks"),
                return ge::GRAPH_FAILED);

    auto dtype = sinksDesc->GetDataType();
    OP_CHECK_IF(dtype != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "sinks", Ops::Base::ToString(dtype).c_str(), "FLOAT32"),
                return ge::GRAPH_FAILED);

    auto& shape = sinksTensor->GetStorageShape();
    uint32_t dimNum = shape.GetDimNum();
    OP_CHECK_IF(dimNum != 1,
                OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "sinks", (std::to_string(dimNum) + "D").c_str(), "1D"),
                return ge::GRAPH_FAILED);

    if (shape.GetDim(0) != faInfo.n1Size) {
        std::string shapeStr = Ops::Base::ToString(shape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            faInfo.opName, "sinks", shapeStr.c_str(),
            ("The first dim of sinks must be equal to Q_N " + std::to_string(faInfo.n1Size)).c_str());
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace flash_attn
} // namespace optiling
