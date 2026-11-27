/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_MERGE_CHUNK_H_
#define ACLNN_MERGE_CHUNK_H_

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnMergeChunk 的第一段接口，校验参数，计算所需的 workspace 大小并返回执行器。
 * @param [in] inputChunks: 各路的 token 数据，列表长度为 numPaths × numChunks。第 r 路的第 c 个 chunk
 * 位于下标 r * numChunks + c（下标从 0 开始）。各 Tensor 的行按专家编号分段，数据类型统一为 FLOAT16 或 BFLOAT16。
 * @param [in] groupListChunk: 每个 chunk 一份一维 INT32 计数，列表长度为 numChunks。第 c 项的第 e 个元素
 * 表示该 chunk 中专家 e 的 token 数，同一 chunk 的所有数据路共用这份计数。
 * @param [in] numPaths: 数据路数，当前版本仅支持 2。
 * @param [in] numChunks: 每路的 chunk 数，当前版本仅支持 2。
 * @param [out] outChunks: 调用方提供的输出 Tensor 列表，长度为 numPaths。第 r 项保存第 r 路按专家合并的
 * token 数据，行数为该路各 chunk 的行数之和，列数和数据类型与该路输入一致。
 * @param [out] outGroupList: 调用方提供的一维 INT32 输出 Tensor，第 e 个元素为各 chunk 中专家 e 的 token 数之和。
 * @param [out] workspaceSize: 返回第二段接口所需的设备侧 workspace 大小，单位为字节。
 * @param [out] executor: 返回供第二段接口使用的算子执行器。
 * @return aclnnStatus: 接口返回的状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMergeChunkGetWorkspaceSize(
    const aclTensorList* inputChunks, const aclTensorList* groupListChunk, int64_t numPaths, int64_t numChunks,
    const aclTensorList* outChunks, const aclTensor* outGroupList, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnMergeChunk 的第二段接口，将按专家合并各路 token 数据的任务提交到指定的 stream。
 * @param [in] workspace: 调用方申请的设备侧 workspace 起始地址，workspaceSize 为 0 时可传 nullptr。
 * @param [in] workspaceSize: 第一段接口返回的 workspace 大小，单位为字节。
 * @param [in] executor: 第一段接口返回的算子执行器。
 * @param [in] stream: 执行计算任务的 ACL Stream。
 * @return aclnnStatus: 接口返回的状态码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnMergeChunk(void* workspace, uint64_t workspaceSize,
                                                                   aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
