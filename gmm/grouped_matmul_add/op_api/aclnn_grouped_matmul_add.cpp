/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <string>
#include <utility>
#include "aclnn_grouped_matmul_add.h"
#include "aclnn_grouped_matmul_add_v2.h"

#include <dlfcn.h>
#include <limits>
#include <new>

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "log/log.h"
#include "gmm/common/op_host/log_format_util.h"

#include "../../grouped_matmul/op_api/grouped_matmul_util.h"
#include "grouped_matmul_add_util.h"
#include "grouped_matmul_add.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t ELEMENTS_PER_32B_16BIT = 16;

bool UsesArch35Kernel()
{
    const auto arch = op::GetCurrentPlatformInfo().GetCurNpuArch();
    return arch == NpuArch::DAV_3510 || arch == NpuArch::DAV_9201 || arch == NpuArch::DAV_9202;
}

static aclnnStatus CheckNotNull(gmm_add_advanced::GroupedMatmulAddParams params, const char* opName)
{
    CHECK_COND(params.x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [%s] must not be nullptr.", opName, "x");
    CHECK_COND(params.weight != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [%s] must not be nullptr.", opName,
               "weight");
    CHECK_COND(params.groupList != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [%s] must not be nullptr.", opName,
               "groupList");
    CHECK_COND(params.yRef != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [%s] must not be nullptr.", opName,
               "yRef");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(gmm_add_advanced::GroupedMatmulAddParams params, const char* opName)
{
    CHECK_COND(params.x->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, the format of [%s] is not supported, got [%s].", opName, "x",
               op::ToString(params.x->GetStorageFormat()).GetString());
    CHECK_COND(params.weight->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, the format of [%s] is not supported, got [%s].", opName, "weight",
               op::ToString(params.weight->GetStorageFormat()).GetString());
    CHECK_COND(params.groupList->GetStorageFormat() == Format::FORMAT_ND, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, the format of [%s] is not supported, got [%s].", opName, "groupList",
               op::ToString(params.groupList->GetStorageFormat()).GetString());
    CHECK_COND(
        params.yRef->GetStorageFormat() == Format::FORMAT_ND || params.yRef->GetStorageFormat() == Format::FORMAT_NCL,
        ACLNN_ERR_PARAM_INVALID, "In op [%s], when non-quant, the format of [%s] is not supported, got [%s].", opName,
        "yRef", op::ToString(params.yRef->GetStorageFormat()).GetString());
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(gmm_add_advanced::GroupedMatmulAddParams params, const char* opName)
{
    auto xViewShape = params.x->GetViewShape();
    auto wViewShape = params.weight->GetViewShape();
    auto yRefViewShape = params.yRef->GetViewShape();
    auto groupListViewShape = params.groupList->GetViewShape();
    auto weightDimNum = wViewShape.GetDimNum();
    auto xDimNum = xViewShape.GetDimNum();
    auto yRefDimNum = yRefViewShape.GetDimNum();
    auto groupListDimNum = groupListViewShape.GetDimNum();

    CHECK_COND(xDimNum == 2, ACLNN_ERR_PARAM_INVALID, // 2 max dim num
               "In op [%s], when non-quant, the shape of [%s] is not supported, got [dim num %zu]. "
               "Constraint:[dim num must be 2].",
               opName, "x", xDimNum);
    CHECK_COND(weightDimNum == 2, ACLNN_ERR_PARAM_INVALID, // 2 max dim num
               "In op [%s], when non-quant, the shape of [%s] is not supported, got [dim num %zu]. "
               "Constraint:[dim num must be 2].",
               opName, "weight", weightDimNum);
    CHECK_COND(groupListDimNum == 1, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, the shape of [%s] is not supported, got [dim num %zu]. "
               "Constraint:[dim num must be 1].",
               opName, "groupList", groupListDimNum);
    auto aKDim = xViewShape.GetDim(params.transposeX ? 0 : 1);
    auto aMDim = xViewShape.GetDim(params.transposeX ? 1 : 0);
    auto bKDim = wViewShape.GetDim(0);                // 0: w shape k direction
    auto bNDim = wViewShape.GetDim(1);                // 1: w shape m direction
    auto groupListDim = groupListViewShape.GetDim(0); // 0: groupList shape

    // The expanded DAV_2201 paths narrow the dimensions in the matmul tiling.
    // Validate before calculating sizes or passing dimensions to tiling.
    const bool expandedPath = op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 &&
                              (!params.transposeX || params.yRef->GetDataType() != DataType::DT_FLOAT);
    if (expandedPath) {
        constexpr int64_t DIM_LIMIT = 65536;
        constexpr int64_t INT32_LIMIT = std::numeric_limits<int32_t>::max();
        CHECK_COND(aMDim >= 0 && aMDim <= INT32_LIMIT && aKDim >= 0 && aKDim <= INT32_LIMIT && bNDim >= 0 &&
                       bNDim <= INT32_LIMIT && groupListDim >= 0 && groupListDim <= INT32_LIMIT,
                   ACLNN_ERR_PARAM_INVALID, "In op [%s], M, K, N and group count must fit in int32.", opName);
        CHECK_COND(xViewShape.GetDim(1) < DIM_LIMIT && bNDim < DIM_LIMIT, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], the last dimensions of x and weight must be less than 65536.", opName);
        const auto maxSize = std::numeric_limits<int64_t>::max();
        CHECK_COND(aMDim == 0 || bNDim <= maxSize / aMDim, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], output element count exceeds int64.", opName);
        const int64_t matrixSize = aMDim * bNDim;
        CHECK_COND(matrixSize == 0 || groupListDim <= maxSize / matrixSize, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], output element count exceeds int64.", opName);
        int64_t actualSize = 1;
        for (size_t i = 0; i < yRefDimNum; ++i) {
            const auto dim = yRefViewShape.GetDim(i);
            CHECK_COND(dim >= 0 && (actualSize == 0 || dim <= maxSize / actualSize), ACLNN_ERR_PARAM_INVALID,
                       "In op [%s], yRef element count exceeds int64 or has a negative dimension.", opName);
            actualSize *= dim;
        }
    }

    // Require a 32-byte-multiple row width for the initial 16-bit output path.
    // The FP32 path keeps its old contract.
    CHECK_COND(params.yRef->GetDataType() == DataType::DT_FLOAT || bNDim % ELEMENTS_PER_32B_16BIT == 0,
               ACLNN_ERR_PARAM_INVALID,
               "In op [%s], N must be a multiple of 16 when yRef is FLOAT16 or BFLOAT16, got %ld.", opName, bNDim);

    CHECK_COND(aMDim >= 0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, the shape of [%s] is not supported, got [M %ld]. Constraint:[M must "
               "not be negative].",
               opName, "x", aMDim);

    CHECK_COND(aKDim == bKDim, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, the tensor shapes of [%s...] are mismatched, the reason is: [K dim "
               "should be equal, but actual K dims are %ld and %ld].",
               opName, "x, weight", aKDim, bKDim);

    // group_matmul_add support the y shape is 3D actually. other is historical issue.
    constexpr size_t Dim3 = 3;
    if (yRefDimNum != Dim3) {
        OP_LOGW("Expected shape of y is 3 Dim.");
    }
    // only check total size for avoiding error interception. the y shape Dim is not interceped before.
    auto expectedYSize = aMDim * bNDim * groupListDim;
    auto actualYSize = yRefViewShape.GetShapeSize();
    CHECK_COND(expectedYSize == actualYSize, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, the shape of [%s] is not supported, got [shape size %ld]. "
               "Constraint:[shape size should be %ld].",
               opName, "yRef", actualYSize, expectedYSize);

    if (UsesArch35Kernel()) {
        auto groupNum = params.groupList->GetViewShape().GetDim(0);
        auto yDimNum = params.yRef->GetViewShape().GetDimNum();
        if (yDimNum != Dim3) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                opName, "y", Ops::Transformer::Gmm::FormatString("%zu", yDimNum).c_str(),
                Ops::Transformer::Gmm::FormatString("The shape dim of %s must be %zu", "y", Dim3).c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
        auto groupNumY = params.yRef->GetViewShape().GetDim(0);
        if (groupNum != groupNumY) {
            OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                opName, "groupList", Ops::Transformer::Gmm::FormatString("%ld", groupNum).c_str(),
                Ops::Transformer::Gmm::FormatString(
                    "When %s is %s, the shape size of %s must be equal to the size (%ld) of the %s axis of %s",
                    "groupType", "2", "groupList", groupNumY, "first", "y")
                    .c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

// check zero axis
static aclnnStatus CheckZeroShape(gmm_add_advanced::GroupedMatmulAddParams params)
{
    bool isEmptyX = params.x->IsEmpty() ? true : false;
    bool isEmptyWeight = params.weight->IsEmpty() ? true : false;
    // support return empty tensor
    aclnnStatus ret = isEmptyX || isEmptyWeight ? ACLNN_ERR_PARAM_INVALID : ACLNN_SUCCESS;
    return ret;
}

static aclnnStatus DataContiguous(const aclTensor*& tensor, aclOpExecutor* executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(gmm_add_advanced::GroupedMatmulAddParams& params, aclOpExecutor* executorPtr,
                                        const char* opName)
{
    if (!params.transposeX && UsesArch35Kernel()) {
        // Present [K, M] to the split-K kernel and materialize it with the contiguous step below.
        auto shape = params.x->GetViewShape();
        auto strides = params.x->GetViewStrides();
        std::swap(shape[0], shape[1]);
        std::swap(strides[0], strides[1]);
        params.x =
            executorPtr->CreateView(params.x, shape, params.x->GetStorageShape(), strides, params.x->GetViewOffset());
        CHECK_RET(params.x != nullptr, ACLNN_ERR_INNER_NULLPTR);
        params.transposeX = true;
    }
    CHECK_COND(DataContiguous(params.x, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, contiguous [%s] failed.", opName, "x");
    CHECK_COND(DataContiguous(params.weight, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, contiguous [%s] failed.", opName, "weight");
    CHECK_COND(DataContiguous(params.groupList, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, contiguous [%s] failed.", opName, "groupList");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(gmm_add_advanced::GroupedMatmulAddParams params, const char* opName)
{
    auto xDtype = params.x->GetDataType();
    auto weightDtype = params.weight->GetDataType();
    CHECK_COND(params.yRef->GetDataType() == DataType::DT_FLOAT || params.yRef->GetDataType() == xDtype,
               ACLNN_ERR_PARAM_INVALID, "In op [%s], when non-quant, the data type of [%s] is not supported, got [%s].",
               opName, "yRef", op::ToString(params.yRef->GetDataType()).GetString());
    if (params.yRef->GetDataType() != DataType::DT_FLOAT) {
        // The initial 16-bit kernel accesses a flat, 32-byte-aligned output view.
        CHECK_COND(IsContiguous(params.yRef), ACLNN_ERR_PARAM_INVALID, "In op [%s], 16-bit yRef must be contiguous.",
                   opName);
        const auto offset = params.yRef->GetViewOffset();
        CHECK_COND(offset >= 0 && offset % ELEMENTS_PER_32B_16BIT == 0, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], 16-bit yRef view offset must be a nonnegative multiple of 16 elements.", opName);
    }
    CHECK_COND(params.groupList->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, the data type of [%s] is not supported, got [%s].", opName, "groupList",
               op::ToString(params.groupList->GetDataType()).GetString());
    if ((xDtype != DataType::DT_FLOAT16 && xDtype != DataType::DT_BF16) || (xDtype != weightDtype)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            "GroupedMatmulAdd", "x, weight",
            std::string(op::ToString(xDtype).GetString()) + ", " + std::string(op::ToString(weightDtype).GetString()),
            "In op [" + std::string(opName) + "], when non-quant, the data types of [" + std::string("x, weight") +
                "...] are mismatched, the reason is: [x dtype " + std::string(op::ToString(xDtype).GetString()) +
                " and weight dtype " + std::string(op::ToString(weightDtype).GetString()) +
                " are not supported. This operator supports x/weight dtype FLOAT16 or BF16 only].");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckAttrs(gmm_add_advanced::GroupedMatmulAddParams params, const char* opName)
{
    CHECK_COND(params.groupListType == 0 || params.groupListType == 1, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, [%s] only supported 0/1, but got [%ld].", opName, "groupListType",
               params.groupListType);
    CHECK_COND(params.groupType == 2, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, [%s] must be 2, but got [%ld].", opName, "groupType", params.groupType);
    CHECK_COND(op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_2201 ||
                   params.yRef->GetDataType() == DataType::DT_FLOAT,
               ACLNN_ERR_PARAM_INVALID, "In op [%s], 16-bit yRef is supported only on DAV_2201.", opName);
    CHECK_COND(!params.transposeWeight, ACLNN_ERR_PARAM_INVALID, "In op [%s], when non-quant,  [%s] must be False.",
               opName, "transposeWeight");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(gmm_add_advanced::GroupedMatmulAddParams params, const char* opName)
{
    CHECK_RET(CheckNotNull(params, opName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckFormat(params, opName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params, opName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params, opName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttrs(params, opName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSizeCommon(gmm_add_advanced::GroupedMatmulAddParams params,
                                                               uint64_t* workspaceSize, aclOpExecutor** executor,
                                                               const char* opName)
{
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    // 固定写法，参数检查
    auto ret = CheckParams(params, opName);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    // any axis is zero return empty tensor
    if (CheckZeroShape(params) != ACLNN_SUCCESS) {
        *workspaceSize = 0UL;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    CHECK_COND(IsContiguous(params.yRef), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, [%s] must be a contiguous tensor, non-contiguous tensor is not supported.",
               opName, "yRef");
    CHECK_COND(ParamsDataContiguous(params, executorPtr, opName) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when non-quant, ParamsDataContiguous failed.", opName);
    auto result = l0op::GroupedMatmulAdd(params.x, params.weight, params.groupList, params.yRef, params.transposeX,
                                         params.transposeWeight, params.groupType, params.groupListType, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // The in-place operation already returns yRef. Avoid invalidating the executor cache
    // through a redundant self-copy; retain writeback if a separate result is returned.
    if (result != params.yRef) {
        auto viewCopyResult = l0op::ViewCopy(result, params.yRef, executorPtr);
        CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(const aclTensor* x, const aclTensor* weight,
                                                  const aclTensor* groupList, aclTensor* yRef, bool transposeX,
                                                  bool transposeWeight, int64_t groupType, uint64_t* workspaceSize,
                                                  aclOpExecutor** executor)
{
    const char* opName = "aclnnGroupedMatmulAddGetWorkspaceSize";
    gmm_add_advanced::GroupedMatmulAddParams params{x, weight, groupList, yRef, transposeX, transposeWeight, groupType};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnGroupedMatmulAdd, DFX_IN(x, weight, groupList, yRef, transposeX, transposeWeight, groupType),
                   DFX_OUT(yRef));
    return aclnnGroupedMatmulAddGetWorkspaceSizeCommon(params, workspaceSize, executor, opName);
}

aclnnStatus aclnnGroupedMatmulAddV2GetWorkspaceSize(const aclTensor* x, const aclTensor* weight,
                                                    const aclTensor* groupList, aclTensor* yRef, bool transposeX,
                                                    bool transposeWeight, int64_t groupType, int64_t groupListType,
                                                    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    const char* opName = "aclnnGroupedMatmulAddV2GetWorkspaceSize";
    gmm_add_advanced::GroupedMatmulAddParams params{x,          weight,          groupList, yRef,
                                                    transposeX, transposeWeight, groupType, groupListType};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnGroupedMatmulAddV2,
                   DFX_IN(x, weight, groupList, yRef, transposeX, transposeWeight, groupType, groupListType),
                   DFX_OUT(yRef));
    return aclnnGroupedMatmulAddGetWorkspaceSizeCommon(params, workspaceSize, executor, opName);
}

aclnnStatus aclnnGroupedMatmulAdd(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulAdd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "An error occurred when QuantGMMInplaceAdd launched on aicore.");
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulAddV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulAddV2);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "An error occurred when QuantGMMInplaceAdd launched on aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
