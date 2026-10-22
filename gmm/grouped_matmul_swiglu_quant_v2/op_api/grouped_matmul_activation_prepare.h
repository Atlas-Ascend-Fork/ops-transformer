/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_MATMUL_ACTIVATION_PREPARE_H_
#define GROUPED_MATMUL_ACTIVATION_PREPARE_H_
#include "opdev/op_executor.h"
namespace gmm_activation {
struct PreparedInputs {
    const aclTensor* x = nullptr;
    const aclTensor* xScale = nullptr;
    const aclTensor* groupList = nullptr;
    const aclTensorList* weight = nullptr;
    const aclTensorList* weightScale = nullptr;
    bool transposeWeight = false;
};
aclnnStatus PrepareInputs(const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList,
                          const aclTensorList* weight, const aclTensorList* weightScale, int64_t groupListType,
                          const aclIntArray* tuningConfig, int64_t swigluMode, double clampLimit, double gluAlpha,
                          double gluBias, aclTensor* y, aclOpExecutor* executor, PreparedInputs& inputs);
} // namespace gmm_activation
#endif
