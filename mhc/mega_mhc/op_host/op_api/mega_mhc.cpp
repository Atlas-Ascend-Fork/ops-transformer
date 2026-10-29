/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mega_mhc.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "log/log.h"
#include <string>
using namespace op;
namespace l0op {
OP_TYPE_REGISTER(MegaMhc);
/** Pack inputs, outputs and attributes in OpDef order. */
bool MegaMhc(const aclTensor* x, const aclTensor* residual, const aclTensor* postMix, const aclTensor* fn,
             const aclTensor* mixScales, const aclTensor* mixBases, const aclTensor* rmsNorm1Weight,
             const aclTensor* combResMixOptional, const aclTensor* shiftedPrevMixOptional,
             const aclTensor* rmsNorm0WeightOptional, const aclTensor* rmsNormMidWeightOptional, double rmsNormEps,
             double hcNormEps, double hcPreEps, int64_t opImplMode, int64_t hcMult, double hcPostScale,
             double sinkhornEps, int64_t numSinkhornIters, double rmsNormScale, int64_t outputMode,
             const aclTensor* newResidualOut, const aclTensor* newPostMixOut, const aclTensor* newCombResMixOut,
             const aclTensor* newPrevMixOut, const aclTensor* yOut, const aclTensor* yQuantOut,
             const aclTensor* yScaleOut, aclOpExecutor* executor)
{
    L0_DFX(MegaMhc, x, residual, postMix, fn, mixScales, mixBases, rmsNorm1Weight, combResMixOptional,
           shiftedPrevMixOptional, rmsNorm0WeightOptional, rmsNormMidWeightOptional, rmsNormEps, hcNormEps, hcPreEps,
           opImplMode, hcMult, hcPostScale, sinkhornEps, numSinkhornIters, rmsNormScale, outputMode, newResidualOut,
           newPostMixOut, newCombResMixOut, newPrevMixOut, yOut, yQuantOut, yScaleOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        MegaMhc,
        OP_INPUT(x, residual, postMix, fn, mixScales, mixBases, rmsNorm1Weight, combResMixOptional,
                 shiftedPrevMixOptional, rmsNorm0WeightOptional, rmsNormMidWeightOptional),
        OP_OUTPUT(newResidualOut, newPostMixOut, newCombResMixOut, newPrevMixOut, yOut, yQuantOut, yScaleOut),
        OP_ATTR(static_cast<float>(rmsNormEps), static_cast<float>(hcNormEps), static_cast<float>(hcPreEps), opImplMode,
                hcMult, static_cast<float>(hcPostScale), static_cast<float>(sinkhornEps), numSinkhornIters,
                static_cast<float>(rmsNormScale), outputMode));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("MegaMhc", "launcher status",
                                              std::to_string(static_cast<int32_t>(ret)).c_str(),
                                              "ADD_TO_LAUNCHER_LIST_AICORE failed");
        return false;
    }
    return true;
}
} // namespace l0op
