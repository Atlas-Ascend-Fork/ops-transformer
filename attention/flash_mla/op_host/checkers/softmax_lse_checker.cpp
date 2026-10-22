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
 * \file softmax_lse_checker.cpp
 * \brief Checker for softmax_lse parameter (文档约束: SoftmaxLSE参数组)
 */

#include <map>
#include <numeric>
#include <graph/utils/type_utils.h>
#include "log/log.h"
#include "log/error_code.h"
#include "register/op_def_registry.h"
#include "../flash_mla_tiling_info.h"
#include "softmax_lse_checker.h"

namespace optiling {
namespace flash_mla {
using std::map;
using std::pair;
using std::string;
using namespace ge;
using namespace AscendC;
using namespace arch35MLA;

ge::graphStatus SoftmaxLSEChecker::CheckSinglePara(const FlashMlaTilingInfo& faInfo)
{
    OP_CHECK_IF(faInfo.returnSoftmaxLse != 0 && faInfo.returnSoftmaxLse != 1,
                OP_LOGE_FOR_INVALID_VALUE(faInfo.opName, "return_softmax_lse",
                                          std::to_string(faInfo.returnSoftmaxLse).c_str(), "0 or 1"),
                return ge::GRAPH_FAILED);

    if (!faInfo.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }

    const gert::CompileTimeTensorDesc* lseOutDesc = faInfo.opParamInfo.lseOut.desc;
    if (lseOutDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(lseOutDesc->GetDataType() != ge::DT_FLOAT,
                OP_LOGE_FOR_INVALID_DTYPE(faInfo.opName, "lse_out",
                                          Ops::Base::ToString(lseOutDesc->GetDataType()).c_str(), "FP32"),
                return ge::GRAPH_FAILED);

    if (ge::GRAPH_SUCCESS != CheckFormatSupport(lseOutDesc, SOFTMAX_LSE_NAME)) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckParaExistence(const FlashMlaTilingInfo& faInfo)
{
    if (faInfo.softmaxLseFlag) {
        const gert::StorageShape* lseOutShape = faInfo.opParamInfo.lseOut.shape;
        OP_CHECK_IF(lseOutShape == nullptr || faInfo.opParamInfo.lseOut.desc == nullptr,
                    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(faInfo.opName, "lse_out", "empty",
                                                          "When softmaxLSE is enabled, lse_out cannot be empty"),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SoftmaxLSEChecker::CheckMultiPara(const FlashMlaTilingInfo& faInfo)
{
    if (!faInfo.softmaxLseFlag) {
        return ge::GRAPH_SUCCESS;
    }
    const auto* lseOutShape = faInfo.opParamInfo.lseOut.shape;
    OP_CHECK_IF(lseOutShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(faInfo.opName, "lse_out"), return ge::GRAPH_FAILED);

    const auto& shape = lseOutShape->GetStorageShape();
    const bool isTnd = faInfo.qLayout == FlashMlaLayout::TND;
    const std::vector<int64_t> expectedShape =
        isTnd ? std::vector<int64_t>{static_cast<int64_t>(faInfo.n1Size), static_cast<int64_t>(faInfo.qTSize)} :
                std::vector<int64_t>{static_cast<int64_t>(faInfo.bSize), static_cast<int64_t>(faInfo.n1Size),
                                     static_cast<int64_t>(faInfo.s1Size)};
    const char* reason = isTnd ? "lse_out shape must be [Q_N, Q_T] for TND query" :
                                 "lse_out shape must be [B, Q_N, Q_S] for BSND/BNSD query";
    OP_CHECK_IF(shape.GetDimNum() != expectedShape.size(),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "lse_out", GetShapeStr(shape).c_str(), reason),
                return ge::GRAPH_FAILED);
    for (size_t axis = 0; axis < expectedShape.size(); ++axis) {
        OP_CHECK_IF(shape.GetDim(axis) != expectedShape[axis],
                    OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(faInfo.opName, "lse_out", GetShapeStr(shape).c_str(), reason),
                    return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

} // namespace flash_mla
} // namespace optiling
