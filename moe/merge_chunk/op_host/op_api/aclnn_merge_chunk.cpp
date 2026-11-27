/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_merge_chunk.h"
#include "merge_chunk.h"
#include "../merge_chunk_shape.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
aclnnStatus CheckTensor(const aclTensor* tensor, DataType type)
{
    if (tensor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (tensor->GetDataType() != type || tensor->GetStorageFormat() != Format::FORMAT_ND || !IsContiguous(tensor)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckList(const aclTensorList* tensors, DataType type)
{
    for (size_t i = 0; i < tensors->Size(); ++i) {
        const auto status = CheckTensor((*tensors)[i], type);
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensorList* input, const aclTensorList* counts, int64_t paths, int64_t chunks,
                        const aclTensorList* output, const aclTensor* outCounts)
{
    if (input == nullptr || counts == nullptr || output == nullptr || outCounts == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    if (!MergeChunkSupport::ValidListDimensions(paths, chunks) ||
        input->Size() != static_cast<size_t>(paths * chunks) || counts->Size() != static_cast<size_t>(chunks) ||
        output->Size() != static_cast<size_t>(paths)) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((*input)[0] == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    const auto dtype = (*input)[0]->GetDataType();
    if (dtype != DataType::DT_FLOAT16 && dtype != DataType::DT_BF16) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (const auto status : {CheckList(input, dtype), CheckList(counts, DataType::DT_INT32), CheckList(output, dtype),
                              CheckTensor(outCounts, DataType::DT_INT32)}) {
        if (status != ACLNN_SUCCESS) {
            return status;
        }
    }
    MergeChunkSupport::Layout layout;
    if (MergeChunkSupport::ReadLayout(
            paths, chunks, [input](size_t i) { return i < input->Size() ? &(*input)[i]->GetViewShape() : nullptr; },
            [counts](size_t i) { return i < counts->Size() ? &(*counts)[i]->GetViewShape() : nullptr; },
            layout) != nullptr) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t path = 0; path < layout.paths; ++path) {
        const auto& shape = (*output)[path]->GetViewShape();
        if (shape.GetDimNum() != MergeChunkSupport::TOKEN_RANK || shape.GetDim(0) != layout.totalRows ||
            shape.GetDim(1) != layout.hidden[path]) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    const auto& countShape = outCounts->GetViewShape();
    return countShape.GetDimNum() == 1 && countShape.GetDim(0) == layout.experts ? ACLNN_SUCCESS :
                                                                                   ACLNN_ERR_PARAM_INVALID;
}
} // namespace

extern "C" aclnnStatus aclnnMergeChunkGetWorkspaceSize(const aclTensorList* inputChunks,
                                                       const aclTensorList* groupListChunk, int64_t numPaths,
                                                       int64_t numChunks, const aclTensorList* outChunks,
                                                       const aclTensor* outGroupList, uint64_t* workspaceSize,
                                                       aclOpExecutor** executor)
{
    if (workspaceSize == nullptr || executor == nullptr) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    *workspaceSize = 0;
    *executor = nullptr;
    const auto status = CheckParams(inputChunks, groupListChunk, numPaths, numChunks, outChunks, outGroupList);
    if (status != ACLNN_SUCCESS) {
        return status;
    }
    L2_DFX_PHASE_1(aclnnMergeChunk, DFX_IN(inputChunks, groupListChunk, numPaths, numChunks),
                   DFX_OUT(outChunks, outGroupList));
    auto owner = CREATE_EXECUTOR();
    if (owner.get() == nullptr) {
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    }
    const auto launchStatus =
        l0op::LaunchMergeChunk(inputChunks, groupListChunk, numPaths, numChunks, outChunks, outGroupList, owner.get());
    if (launchStatus != ACLNN_SUCCESS) {
        return launchStatus;
    }
    *workspaceSize = owner->GetWorkspaceSize();
    owner.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnMergeChunk(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                       aclrtStream stream)
{
    if (executor == nullptr || (workspaceSize != 0 && workspace == nullptr)) {
        return ACLNN_ERR_PARAM_NULLPTR;
    }
    L2_DFX_PHASE_2(aclnnMergeChunk);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
