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
 * \file quant_block_sparse_attn.cpp
 * \brief QuantBlockSparseAttn kernel entry: dispatch into the kernel-side Process().
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "arch35/common/quant_block_sparse_attn_common.h"
#include "arch35/common/quant_block_sparse_attn_tiling_key_registry.h"
#if QBSA_ARCH92_FP8
#include "arch92/qbsa_fp8/quant_block_sparse_attn_kernel.h"
#define QBSA_FP8_KERNEL_TILING_TYPE optiling::QuantBlockSparseAttnFp8Arch92TilingData
#elif QBSA_ARCH_9102R1
#include "arch35/qbsa_mxfp8/qbsa_mxfp8_tiling_data.h"
#include "arch92/qbsa_mxfp8/quant_block_sparse_attn_mx_kernel.h"
#include "arch92/qbsa_mxfp8/quant_block_sparse_attn_mx_m128_v64_layout.h"
#if QBSA_9102_MX_M128_V64
#include "arch92/qbsa_mxfp8/quant_block_sparse_attn_mx_m128_v64_kernel.h"
#endif
#else
#include "arch35/qbsa_fp8/qbsa_fp8_tiling_data.h"
#include "arch35/qbsa_mxfp8/qbsa_mxfp8_tiling_data.h"
#include "arch35/qbsa_fp8/quant_block_sparse_attn_kernel.h"
#include "arch35/qbsa_mxfp8/quant_block_sparse_attn_mx_kernel.h"
#define QBSA_FP8_KERNEL_TILING_TYPE optiling::QuantBlockSparseAttnFp8TilingData
#endif

using namespace AscendC;
using namespace optiling;

// Follow the kernel invocation pattern in kv_quant_sparse_attn_sharedkv.cpp:
//   - Select real/Dummy Cube/Vec blocks by g_coreType: AIC runs real Cube + Dummy Vec; AIV runs Dummy Cube + real Vec.
//   - Instantiate QuantBlockSparseAttnKernel<CubeBlockType,VecBlockType>.
//   - GET_TILING_DATA -> InitBaseAPI -> Process()
// __VA_ARGS__ contains block template parameters in exactly the CUBE_BLOCK_TRAITS field order.

#define QBSA_OP_IMPL_WITH_TILING(tilingDataPtr, ...) \
    do { \
        using CubeBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::QBSABlockCube<__VA_ARGS__>, \
                                      BaseApi::QBSABlockCubeDummy<__VA_ARGS__>>::type; \
        using VecBlockType = \
            typename std::conditional<g_coreType == AscendC::AIC, BaseApi::QBSABlockVecDummy<__VA_ARGS__>, \
                                      BaseApi::QBSABlockVec<__VA_ARGS__>>::type; \
        BaseApi::QuantBlockSparseAttnKernel<CubeBlockType, VecBlockType> op; \
        op.InitBaseAPI(query, key, value, sparseIndices, sparseSeqLen, attenMask, metadata, cuSeqlensQ, cuSeqlensKv, \
                       seqUsedQ, seqUsedKv, blockTable, nullptr /* queryPaddingSize */, nullptr /* kvPaddingSize */, \
                       qScale /* deqScaleQ */, kScale /* deqScaleK */, vScale /* deqScaleV */, pScale /* pScale */, \
                       nullptr /* softmaxMax */, nullptr /* softmaxSum */, softmaxLse, attentionOut, \
                       user /* workspace */, tilingDataPtr, &tPipe); \
        op.Process(); \
    } while (0)

#define QBSA_OP_IMPL(...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(QBSA_FP8_KERNEL_TILING_TYPE, tilingDataIn, tiling); \
        const QBSA_FP8_KERNEL_TILING_TYPE* __restrict tilingData = &tilingDataIn; \
        QBSA_OP_IMPL_WITH_TILING(tilingData, __VA_ARGS__); \
    } while (0)

#if QBSA_ARCH_9102R1
#define QBSA_MX_RUN_KERNEL(kernelType, tilingDataPtr, ...) \
    do { \
        kernelType<__VA_ARGS__> op; \
        op.Init(query, key, value, sparseIndices, sparseSeqLen, attenMask, metadata, cuSeqlensQ, cuSeqlensKv, \
                seqUsedQ, seqUsedKv, blockTable, qScale, kScale, vScale, pScale, softmaxLse, attentionOut, user, \
                tilingDataPtr, &tPipe); \
        op.Process(); \
    } while (0)
#if QBSA_9102_MX_M128_V64
#define QBSA_MX_OP_IMPL_WITH_TILING(tilingDataPtr, ...) \
    do { \
        if ((tilingDataPtr)->commonParams.paParams.qBlockSize == 128U) { \
            QBSA_MX_RUN_KERNEL(BaseApi::QuantBlockSparseAttnMxM128V64Kernel, tilingDataPtr, __VA_ARGS__); \
        } else { \
            QBSA_MX_RUN_KERNEL(BaseApi::QuantBlockSparseAttnMxKernel, tilingDataPtr, __VA_ARGS__); \
        } \
    } while (0)
#else
#define QBSA_MX_OP_IMPL_WITH_TILING(tilingDataPtr, ...) \
    QBSA_MX_RUN_KERNEL(BaseApi::QuantBlockSparseAttnMxKernel, tilingDataPtr, __VA_ARGS__)
#endif
#else
#define QBSA_MX_OP_IMPL_WITH_TILING(tilingDataPtr, ...) \
    do { \
        BaseApi::QuantBlockSparseAttnMxKernel<__VA_ARGS__> op; \
        op.Init(query, key, value, sparseIndices, sparseSeqLen, attenMask, metadata, cuSeqlensQ, cuSeqlensKv, \
                seqUsedQ, seqUsedKv, blockTable, qScale, kScale, vScale, pScale, softmaxLse, attentionOut, user, \
                tilingDataPtr, &tPipe); \
        op.Process(); \
    } while (0)
#endif

#define QBSA_MX_OP_IMPL(...) \
    do { \
        GET_TILING_DATA_WITH_STRUCT(QuantBlockSparseAttnMxfp8TilingData, tilingDataIn, tiling); \
        const QuantBlockSparseAttnMxfp8TilingData* __restrict tilingData = &tilingDataIn; \
        QBSA_MX_OP_IMPL_WITH_TILING(tilingData, __VA_ARGS__); \
    } while (0)

template <uint32_t QKV_DTYPE, uint32_t LAYOUT_T, uint32_t KV_LAYOUT_T, uint32_t MASK_MODE, bool RETURN_SOFTMAX_LSE,
          uint32_t Config, uint32_t QUANT_MODE>
__global__ __aicore__ void quant_block_sparse_attn(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* qScale, __gm__ uint8_t* kScale,
    __gm__ uint8_t* vScale, __gm__ uint8_t* sparseIndices, __gm__ uint8_t* sparseSeqLen, __gm__ uint8_t* pScale,
    __gm__ uint8_t* cuSeqlensQ, __gm__ uint8_t* cuSeqlensKv, __gm__ uint8_t* seqUsedQ, __gm__ uint8_t* seqUsedKv,
    __gm__ uint8_t* blockTable, __gm__ uint8_t* attenMask, __gm__ uint8_t* metadata, __gm__ uint8_t* attentionOut,
    __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
#if QBSA_ARCH92_FP8
    REGISTER_TILING_DEFAULT(optiling::QuantBlockSparseAttnFp8Arch92TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#elif QBSA_ARCH_9102R1
    REGISTER_TILING_DEFAULT(optiling::QuantBlockSparseAttnMxfp8TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
    REGISTER_TILING_DEFAULT(optiling::QuantBlockSparseAttnFp8TilingData);
    REGISTER_TILING_FOR_TILINGKEY("QUANT_MODE == 2", optiling::QuantBlockSparseAttnMxfp8TilingData);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#endif

    // The tiling key selects compile-time templates for the mask mode and softmaxLse enable flag.
    TPipe tPipe;
    __gm__ uint8_t* user = GetUserWorkspace(workspace);

    constexpr QBSALayout layout = static_cast<QBSALayout>(LAYOUT_T);
    constexpr QBSALayout kvLayout = static_cast<QBSALayout>(KV_LAYOUT_T);
    // isPa: contiguous KV -> false; PA_ND -> true.
    constexpr bool bsaIsPa = true;
    constexpr bool bsaUseDn = BaseApi::IsDn();
    constexpr bool HAS_ATTENTION = (MASK_MODE == 3);
#if QBSA_ARCH92_FP8
    static_assert(QUANT_MODE == QBSA_FP8_ARCH92_QUANT_MODE, "arch92 supports FP8 only");
    static_assert(Config == QBSA_FP8_ARCH92_CONFIG, "arch92 FP8 requires S1=64, S2=512, D=DV=128");
    QBSA_OP_IMPL(fp8_e4m3fn_t, float, bfloat16_t, layout, kvLayout, S1TemplateType::Aligned64,
                 S2TemplateType::Aligned512, DTemplateType::Aligned128, DTemplateType::Aligned128, HAS_ATTENTION,
                 RETURN_SOFTMAX_LSE, bsaIsPa, bsaUseDn);
#else
    if constexpr (QUANT_MODE == QBSA_MXFP8_QUANT_MODE) {
#if QBSA_ARCH_9102R1
        static_assert(Config == QBSA_MXFP8_CONFIG, "960 MXFP8 requires S1=128, S2=512, D=DV=128");
        QBSA_MX_OP_IMPL(fp8_e4m3fn_t, float, bfloat16_t, layout, kvLayout, S1TemplateType::Aligned128,
                        S2TemplateType::Aligned512, DTemplateType::Aligned128, DTemplateType::Aligned128, HAS_ATTENTION,
                        RETURN_SOFTMAX_LSE, bsaIsPa, bsaUseDn);
#else
        static_assert(
            Config == QBSA_TILING_CONFIG_1 || Config == QBSA_TILING_CONFIG_2 || Config == QBSA_TILING_CONFIG_3,
            "MXFP8 requires D=DV in {64,128,256}");
        int64_t originalCastMode = 0;
        int64_t originalGlobalSatMode = 0;
        if ASCEND_IS_AIV {
            // MX Reg Casts use their own SAT/NO_SAT trait; global NO_SAT is reserved for Tensor Casts.
            originalCastMode =
                AscendC::GetCtrlSpr<QbsaVectorApi::CAST_SAT_MODE_CTRL_BIT, QbsaVectorApi::CAST_SAT_MODE_CTRL_BIT>();
            originalGlobalSatMode = AscendC::GetCtrlSpr<QbsaVectorApi::CAST_GLOBAL_SAT_MODE_CTRL_BIT,
                                                        QbsaVectorApi::CAST_GLOBAL_SAT_MODE_CTRL_BIT>();
            AscendC::SetCtrlSpr<QbsaVectorApi::CAST_GLOBAL_SAT_MODE_CTRL_BIT,
                                QbsaVectorApi::CAST_GLOBAL_SAT_MODE_CTRL_BIT>(QbsaVectorApi::CAST_GLOBAL_NO_SAT_MODE);
            AscendC::SetCtrlSpr<QbsaVectorApi::CAST_SAT_MODE_CTRL_BIT, QbsaVectorApi::CAST_SAT_MODE_CTRL_BIT>(
                QbsaVectorApi::CAST_USE_INSTRUCTION_SAT_MODE);
        }
        // Keep the baseline MX tiles: D64/D128 use S2=512; D256 uses S2=256.
        if constexpr (Config == QBSA_TILING_CONFIG_2) {
            QBSA_MX_OP_IMPL(fp8_e4m3fn_t, float, bfloat16_t, layout, kvLayout, S1TemplateType::Aligned128,
                            S2TemplateType::Aligned512, DTemplateType::Aligned64, DTemplateType::Aligned64,
                            HAS_ATTENTION, RETURN_SOFTMAX_LSE, bsaIsPa, bsaUseDn);
        } else if constexpr (Config == QBSA_TILING_CONFIG_3) {
            QBSA_MX_OP_IMPL(fp8_e4m3fn_t, float, bfloat16_t, layout, kvLayout, S1TemplateType::Aligned128,
                            S2TemplateType::Aligned256, DTemplateType::Aligned256, DTemplateType::Aligned256,
                            HAS_ATTENTION, RETURN_SOFTMAX_LSE, bsaIsPa, bsaUseDn);
        } else {
            QBSA_MX_OP_IMPL(fp8_e4m3fn_t, float, bfloat16_t, layout, kvLayout, S1TemplateType::Aligned128,
                            S2TemplateType::Aligned512, DTemplateType::Aligned128, DTemplateType::Aligned128,
                            HAS_ATTENTION, RETURN_SOFTMAX_LSE, bsaIsPa, bsaUseDn);
        }
        if ASCEND_IS_AIV {
            // Restore once at the MX kernel exit so the operator cannot leak CTRL state to subsequent work.
            AscendC::SetCtrlSpr<QbsaVectorApi::CAST_GLOBAL_SAT_MODE_CTRL_BIT,
                                QbsaVectorApi::CAST_GLOBAL_SAT_MODE_CTRL_BIT>(originalGlobalSatMode);
            AscendC::SetCtrlSpr<QbsaVectorApi::CAST_SAT_MODE_CTRL_BIT, QbsaVectorApi::CAST_SAT_MODE_CTRL_BIT>(
                originalCastMode);
        }
#endif
    }
#if !QBSA_ARCH_9102R1
    else if constexpr (QUANT_MODE == QBSA_FP8_QUANT_MODE) {
        static_assert(Config == QBSA_FP8_CONFIG, "FP8 must use the S1=128, S2=256, D=128 and DV=128 config");
        QBSA_OP_IMPL(fp8_e4m3fn_t, float, bfloat16_t, layout, kvLayout, S1TemplateType::Aligned128,
                     S2TemplateType::Aligned256, DTemplateType::Aligned128, DTemplateType::Aligned128, HAS_ATTENTION,
                     RETURN_SOFTMAX_LSE, bsaIsPa, bsaUseDn);
    } else {
        static_assert(QUANT_MODE == QBSA_FP8_QUANT_MODE || QUANT_MODE == QBSA_MXFP8_QUANT_MODE,
                      "Unsupported QBSA quant mode");
    }
#endif
#endif
}
