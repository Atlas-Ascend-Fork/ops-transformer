/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLNN_FUSED_FLOYD_ATTENTION_GRAD_H_
#define ACLNN_FUSED_FLOYD_ATTENTION_GRAD_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnFusedFloydAttentionGrad 的第一段接口，校验参数、计算 workspace 大小并获取执行器。
 * @param [in] query: Device 侧的 aclTensor，表示公式中的 Q。
 * 数据类型支持 FLOAT16、BFLOAT16，与 key1、value1、key2、value2 一致；数据格式为 ND，shape 为 [B,H,N,M,D]，支持非连续
 * Tensor。
 * @param [in] key1: Device 侧的 aclTensor，表示公式中的 K1。
 * 数据类型支持 FLOAT16、BFLOAT16，与 query 一致；数据格式为 ND，shape 为 [B,H,N,K,D]，支持非连续 Tensor。
 * @param [in] value1: Device 侧的 aclTensor，表示公式中的 V1。
 * 数据类型支持 FLOAT16、BFLOAT16，与 query 一致；数据格式为 ND，shape 与 key1 相同，为 [B,H,N,K,D]，支持非连续 Tensor。
 * @param [in] key2: Device 侧的 aclTensor，表示公式中的 K2。
 * 数据类型支持 FLOAT16、BFLOAT16，与 query 一致；数据格式为 ND，shape 为 [B,H,K,M,D]，支持非连续 Tensor。
 * @param [in] value2: Device 侧的 aclTensor，表示公式中的 V2。
 * 数据类型支持 FLOAT16、BFLOAT16，与 query 一致；数据格式为 ND，shape 与 key2 相同，为 [B,H,K,M,D]，支持非连续 Tensor。
 * @param [in] dy: Device 侧的 aclTensor，表示正向输出对应的上游梯度 dY。
 * 数据类型支持 FLOAT16、BFLOAT16，数据格式为 ND，shape 与 query 相同，为 [B,H,N,M,D]，支持非连续 Tensor。
 * @param [in] attenMaskOptional: Device 侧的注意力掩码 aclTensor，当前版本必须传入，不支持空指针。
 * 数据类型支持 BOOL、UINT8，数据格式为 ND，shape 为 [B,1,N,1,K]；取值 1 表示不参与计算，0 表示参与计算，支持非连续
 * Tensor。
 * @param [in] softmaxMaxOptional: Device 侧的 aclTensor，表示正向计算保存的 Softmax 最大值。
 * 数据类型为 FLOAT，数据格式为 ND，shape 为 [B,H,N,M,8]，与 softmaxSumOptional 的 shape 相同，支持非连续 Tensor。
 * @param [in] softmaxSumOptional: Device 侧的 aclTensor，表示正向计算保存的 Softmax 指数和。
 * 数据类型为 FLOAT，数据格式为 ND，shape 为 [B,H,N,M,8]，与 softmaxMaxOptional 的 shape 相同，支持非连续 Tensor。
 * @param [in] attentionInOptional: Device 侧的 aclTensor，表示注意力正向计算的最终输出。
 * 数据类型和 shape 与 query 相同，数据类型支持 FLOAT16、BFLOAT16，数据格式为 ND，shape 为 [B,H,N,M,D]，支持非连续
 * Tensor。
 * @param [in] scaleValue: 注意力分数的缩放系数，Host 侧 DOUBLE 标量。
 * @param [out] dqOut: Device 侧的 aclTensor，表示 query 的梯度 dQ。
 * 数据类型支持 FLOAT16、BFLOAT16，数据格式为 ND，shape 与 query 相同，为 [B,H,N,M,D]，不支持非连续 Tensor。
 * @param [out] dk1Out: Device 侧的 aclTensor，表示 key1 的梯度 dK1。
 * 数据类型支持 FLOAT16、BFLOAT16，数据格式为 ND，shape 与 key1 相同，为 [B,H,N,K,D]，不支持非连续 Tensor。
 * @param [out] dv1Out: Device 侧的 aclTensor，表示 value1 的梯度 dV1。
 * 数据类型支持 FLOAT16、BFLOAT16，数据格式为 ND，shape 与 value1 相同，为 [B,H,N,K,D]，不支持非连续 Tensor。
 * @param [out] dk2Out: Device 侧的 aclTensor，表示 key2 的梯度 dK2。
 * 数据类型支持 FLOAT16、BFLOAT16，数据格式为 ND，shape 与 key2 相同，为 [B,H,K,M,D]，不支持非连续 Tensor。
 * @param [out] dv2Out: Device 侧的 aclTensor，表示 value2 的梯度 dV2。
 * 数据类型支持 FLOAT16、BFLOAT16，数据格式为 ND，shape 与 value2 相同，为 [B,H,K,M,D]，不支持非连续 Tensor。
 * @param [out] workspaceSize: 返回调用方需要在 Device 侧申请的 workspace 大小，单位为字节。
 * @param [out] executor: 返回包含算子计算流程的执行器，供第二段接口使用。
 * @return aclnnStatus: 返回状态码，0 表示成功，其他值为错误码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedFloydAttentionGradGetWorkspaceSize(
    const aclTensor* query, const aclTensor* key1, const aclTensor* value1, const aclTensor* key2,
    const aclTensor* value2, const aclTensor* dy, const aclTensor* attenMaskOptional,
    const aclTensor* softmaxMaxOptional, const aclTensor* softmaxSumOptional, const aclTensor* attentionInOptional,
    double scaleValue, const aclTensor* dqOut, const aclTensor* dk1Out, const aclTensor* dv1Out,
    const aclTensor* dk2Out, const aclTensor* dv2Out, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnFusedFloydAttentionGrad 的第二段接口，在指定 stream 上执行计算。
 * @param [in] workspace: 调用方在 Device 侧申请的 workspace 内存起始地址。
 * @param [in] workspaceSize: Device 侧 workspace 大小，单位为字节，由第一段接口获取。
 * @param [in] executor: 第一段接口返回的执行器，包含算子计算流程。
 * @param [in] stream: 指定执行计算任务的 ACL stream。
 * @return aclnnStatus: 返回状态码，0 表示任务提交成功，其他值为错误码。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedFloydAttentionGrad(void* workspace, uint64_t workspaceSize,
                                                                                aclOpExecutor* executor,
                                                                                aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
