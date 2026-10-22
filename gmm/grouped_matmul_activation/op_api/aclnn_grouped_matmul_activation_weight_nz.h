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
 * \file aclnn_grouped_matmul_activation_weight_nz.h
 * \brief ACLNN interfaces for grouped matmul with direct BF16 SwiGLU output.
 */

#ifndef OP_API_ACLNN_GROUPED_MATMUL_ACTIVATION_WEIGHT_NZ_H
#define OP_API_ACLNN_GROUPED_MATMUL_ACTIVATION_WEIGHT_NZ_H

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnGroupedMatmulActivationWeightNz 的第一段接口，校验参数并计算 workspace 大小。
 * @domain aclnn_ops_infer
 *
 * 融合分组矩阵乘、MX 反量化和前后分半 SwiGLU 激活，直接输出 BF16，不进行输出 MX 量化。
 * 仅支持 Ascend 950PR&950DT系列产品的 MXFP4 E2M1 A4W4 WeightNZ 场景，K/N 均要求 128 对齐。
 *
 * @param [in] x: 激活矩阵，FLOAT4_E2M1、ND，逻辑 shape 为 (M,K)。
 * @param [in] xScale: 激活的 MX scale，FLOAT8_E8M0、ND，shape 为 (M,K/64,2)。
 * @param [in] groupList: 专家分组信息，INT64、ND，shape 为 (E)，数据位于 Device 内存。
 * @param [in] weight: 权重 TensorList，长度为 1，FLOAT4_E2M1、FRACTAL_NZ，包含全部 E 个专家。
 * 传入 Tensor 的 shape 为 (E,K,N)，非转置和转置的 stride 分别为 (K*N,N,1) 和 (K*N,1,K)。
 * @param [in] weightScale: 权重 MX scale TensorList，长度为 1，FLOAT8_E8M0、ND。
 * 传入 Tensor 的 shape 为 (E,K/64,N,2)，转置时使用 stride ((K/64)*N*2,2,(K/64)*2,1)。
 * 其转置状态必须与 weight 一致，接口内部完成维度转换，无需提前交换维度。
 * @param [in] bias: 矩阵乘偏置 TensorList，当前必须为 nullptr 或空列表。
 * @param [in] dequantMode: 反量化模式，仅支持 2，表示 MX 反量化。
 * @param [in] dequantDtype: GMM 中间结果类型，仅支持 0，表示 FLOAT32。
 * @param [in] actType: 激活类型，仅支持字符串 "swiglu"。
 * @param [in] groupListType: groupList 解释方式，0 为累计结束位置，1 为各组 token 数。
 * @param [in] tuningConfig: tiling 调优配置，当前必须为 nullptr 或空数组。
 * @param [in] swigluMode: 仅支持 2，将 GMM 结果前后分半。
 * 激活分支先做上限裁剪，再计算带 gluAlpha 的 sigmoid 激活；线性分支对称裁剪后加 gluBias，最后两分支相乘。
 * @param [in] clampLimit: 裁剪上限，须为 FLOAT32 范围内的有限值，转换为 float 后大于 0。
 * @param [in] gluAlpha: sigmoid 输入缩放系数，须为 FLOAT32 范围内的有限值，内部转换为 float。
 * @param [in] gluBias: 线性分支偏置，须为 FLOAT32 范围内的有限值，内部转换为 float。
 * @param [out] y: SwiGLU 结果，BFLOAT16、ND，shape 为 (M,N/2)，不产生 yScale 输出。
 * @param [out] workspaceSize: 返回需要在 NPU Device 侧申请的 workspace 大小，单位为字节。
 * @param [out] executor: 返回包含算子计算流程的执行器，传给第二段接口。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulActivationWeightNzGetWorkspaceSize(
    const aclTensor* x, const aclTensor* xScale, const aclTensor* groupList, const aclTensorList* weight,
    const aclTensorList* weightScale, const aclTensorList* bias, int64_t dequantMode, int64_t dequantDtype,
    const char* actType, int64_t groupListType, const aclIntArray* tuningConfig, int64_t swigluMode, double clampLimit,
    double gluAlpha, double gluBias, aclTensor* y, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnGroupedMatmulActivationWeightNz 的第二段接口，在指定 stream 上执行计算。
 * @param [in] workspace: 在 NPU Device 侧申请的 workspace 内存起址。
 * @param [in] workspaceSize: workspace 大小，由第一段接口获取，单位为字节。
 * @param [in] executor: 第一段接口返回的算子执行器，包含计算流程。
 * @param [in] stream: 执行算子的 ACL stream。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnGroupedMatmulActivationWeightNz(void* workspace, uint64_t workspaceSize,
                                                           aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_ACLNN_GROUPED_MATMUL_ACTIVATION_WEIGHT_NZ_H
