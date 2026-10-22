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
 * \brief Checker for optional sinks.
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_mla_tiling_info.h"
#include "flash_mla_sinks_checker.h"

namespace optiling {
namespace flash_mla {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35MLA;

ge::graphStatus SinksChecker::CheckSinglePara(const FlashMlaTilingInfo& faInfo)
{
    if (!faInfo.sinkFlag) {
        return ge::GRAPH_SUCCESS;
    }
    OP_CHECK_IF(faInfo.opParamInfo.sinks.desc == nullptr, OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "Desc of sinks"),
                return ge::GRAPH_FAILED);
    if (CheckFormatSupport(faInfo.opParamInfo.sinks.desc, "sinks") != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    auto dtype = faInfo.opParamInfo.sinks.desc->GetDataType();
    if (dtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "sinks", ge::TypeUtils::DataTypeToSerialString(dtype).c_str(),
                                  "FLOAT32");
        return ge::GRAPH_FAILED;
    }
    auto& shape = faInfo.opParamInfo.sinks.tensor->GetStorageShape();
    if (shape.GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(faInfo.opName, "sinks", std::to_string(shape.GetDimNum()).c_str(), "1");
        return ge::GRAPH_FAILED;
    }
    if (static_cast<uint64_t>(shape.GetDim(0)) != static_cast<uint64_t>(faInfo.n1Size)) {
        OP_LOGE_FOR_INVALID_SHAPESIZE(faInfo.opName, "sinks[0]", std::to_string(shape.GetDim(0)).c_str(),
                                      std::to_string(faInfo.n1Size).c_str());
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_mla
} // namespace optiling
