/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "err/ops_err.h"

using namespace ge;
namespace {
const uint32_t TOKEN_X_INDEX = 0;
const uint32_t WEIGHT_UQ_QR_INDEX = 2;
const uint32_t RMSNORM_GAMMA_CQ_INDEX = 4;
const uint32_t RMSNORM_GAMMA_CKV_INDEX = 5;
const uint32_t ROPE_SIN_INDEX = 6;
const uint32_t QK_NOPE_HEAD_DIM_ATTR_INDEX = 4;
const uint32_t Q_NOPE_INDEX = 0;
const uint32_t Q_PE_INDEX = 1;
const uint32_t CT_KV_INDEX = 2;
const uint32_t K_PE_INDEX = 3;
const uint32_t QR_INDEX = 4;
const uint32_t Q_M_INDEX = 5;
const uint32_t Q_RSTD_INDEX = 6;
const uint32_t CKV_M_INDEX = 7;
const uint32_t CKV_RSTD_INDEX = 8;
const size_t DIM_NUM_4 = 4;
} // namespace
namespace ops {
static ge::graphStatus InferShapeDsaSparseProlog(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter DsaSparseProlog infershape impl.");
    // tokenX shape : (B, S, He) | (T, He)
    const gert::Shape* tokenXShape = context->GetInputShape(TOKEN_X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, tokenXShape);
    const gert::Shape* weightUqQrShape = context->GetInputShape(WEIGHT_UQ_QR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, weightUqQrShape);
    const gert::Shape* rmsnormGammaCqShape = context->GetInputShape(RMSNORM_GAMMA_CQ_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, rmsnormGammaCqShape);
    const gert::Shape* rmsnormGammaCkvShape = context->GetInputShape(RMSNORM_GAMMA_CKV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, rmsnormGammaCkvShape);
    const gert::Shape* ropeSinShape = context->GetInputShape(ROPE_SIN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ropeSinShape);
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    ;
    const int64_t* qkNopeHeadDimPtr = attrs->GetInt(QK_NOPE_HEAD_DIM_ATTR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkNopeHeadDimPtr);
    ;

    const int64_t dimD = *qkNopeHeadDimPtr;                                    // qk_nope_head_dim
    const int64_t dimDr = ropeSinShape->GetDim(ropeSinShape->GetDimNum() - 1); // qk_rope_head_dim
    const int64_t headSizeCq = rmsnormGammaCqShape->GetDim(0);                 // Hcq
    const int64_t headSizeCkv = rmsnormGammaCkvShape->GetDim(0);               // Hckv
    // 无 weight_uk 时 N/D 无法仅从 shape 推导（N*(D+Dr) 分解不唯一），D 由 attr 显式传入，
    // N 由 weight_uq_qr 宽度 / (D+Dr) 推导
    // weight_uq_qr : 2D [Hcq, N*(D+Dr)] 或 4D NZ [N*(D+Dr)/16, Hcq/16, 16, 16]（bf16/fp16 内轴固定 16）
    int64_t qOut = 0;
    if (weightUqQrShape->GetDimNum() == DIM_NUM_4) {
        qOut = weightUqQrShape->GetDim(0) * weightUqQrShape->GetDim(3);
    } else {
        qOut = weightUqQrShape->GetDim(1); // N * (D + Dr)
    }
    if ((dimD + dimDr) == 0 || qOut % (dimD + dimDr) != 0) {
        OP_LOGE(context->GetNodeName(), "weight_uq_qr width should be divisible by (qk_nope_head_dim + Dr)");
        return ge::GRAPH_FAILED;
    }
    const int64_t numHead = qOut / (dimD + dimDr); // N

    // q_nope: (B, S, N, D) | (T, N, D)
    gert::Shape* qNopeShape = context->GetOutputShape(Q_NOPE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qNopeShape);
    // q_pe: (B, S, N, Dr) | (T, N, Dr)
    gert::Shape* qPeShape = context->GetOutputShape(Q_PE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qPeShape);
    // ct_kv: (B, S, Hckv) | (T, Hckv)
    gert::Shape* ctKvShape = context->GetOutputShape(CT_KV_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ctKvShape);
    // k_pe: (B, S, 1, Dr) | (T, 1, Dr)
    gert::Shape* kPeShape = context->GetOutputShape(K_PE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kPeShape);
    // qr: (B, S, Hcq) | (T, Hcq)
    gert::Shape* qrShape = context->GetOutputShape(QR_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qrShape);
    // q_m: (B, S, Hcq) | (T, Hcq)
    gert::Shape* qMShape = context->GetOutputShape(Q_M_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qMShape);
    // q_rstd: (B, S, 1) | (T, 1)
    gert::Shape* qRstdShape = context->GetOutputShape(Q_RSTD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, qRstdShape);
    // ckv_m: (B, S, Hckv) | (T, Hckv)
    gert::Shape* ckvMShape = context->GetOutputShape(CKV_M_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ckvMShape);
    // ckv_rstd: (B, S, 1) | (T, 1)
    gert::Shape* ckvRstdShape = context->GetOutputShape(CKV_RSTD_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, ckvRstdShape);

    const size_t batchDimNum = tokenXShape->GetDimNum() - 1;
    // q_nope / q_pe 尾部 [N, D] / [N, Dr]
    qNopeShape->SetDimNum(batchDimNum + 2);
    qPeShape->SetDimNum(batchDimNum + 2);
    for (size_t i = 0; i < batchDimNum; i++) {
        qNopeShape->SetDim(i, tokenXShape->GetDim(i));
        qPeShape->SetDim(i, tokenXShape->GetDim(i));
    }
    qNopeShape->SetDim(batchDimNum, numHead);
    qNopeShape->SetDim(batchDimNum + 1, dimD);
    qPeShape->SetDim(batchDimNum, numHead);
    qPeShape->SetDim(batchDimNum + 1, dimDr);
    // ct_kv / qr 尾部 [Hckv] / [Hcq]
    ctKvShape->SetDimNum(batchDimNum + 1);
    qrShape->SetDimNum(batchDimNum + 1);
    for (size_t i = 0; i < batchDimNum; i++) {
        ctKvShape->SetDim(i, tokenXShape->GetDim(i));
        qrShape->SetDim(i, tokenXShape->GetDim(i));
    }
    ctKvShape->SetDim(batchDimNum, headSizeCkv);
    qrShape->SetDim(batchDimNum, headSizeCq);
    // k_pe 尾部 [1, Dr]
    kPeShape->SetDimNum(batchDimNum + 2);
    for (size_t i = 0; i < batchDimNum; i++) {
        kPeShape->SetDim(i, tokenXShape->GetDim(i));
    }
    kPeShape->SetDim(batchDimNum, 1);
    kPeShape->SetDim(batchDimNum + 1, dimDr);
    // q_m 尾部 [Hcq]；q_rstd 尾部 [1]
    qMShape->SetDimNum(batchDimNum + 1);
    qRstdShape->SetDimNum(batchDimNum + 1);
    for (size_t i = 0; i < batchDimNum; i++) {
        qMShape->SetDim(i, tokenXShape->GetDim(i));
        qRstdShape->SetDim(i, tokenXShape->GetDim(i));
    }
    qMShape->SetDim(batchDimNum, headSizeCq);
    qRstdShape->SetDim(batchDimNum, 1);
    // ckv_m 尾部 [Hckv]；ckv_rstd 尾部 [1]
    ckvMShape->SetDimNum(batchDimNum + 1);
    ckvRstdShape->SetDimNum(batchDimNum + 1);
    for (size_t i = 0; i < batchDimNum; i++) {
        ckvMShape->SetDim(i, tokenXShape->GetDim(i));
        ckvRstdShape->SetDim(i, tokenXShape->GetDim(i));
    }
    ckvMShape->SetDim(batchDimNum, headSizeCkv);
    ckvRstdShape->SetDim(batchDimNum, 1);

    OP_LOGD(context->GetNodeName(), "DsaSparseProlog infershape end.");
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeDsaSparseProlog(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter DsaSparseProlog inferdatatype impl.");

    ge::DataType outputType = context->GetInputDataType(TOKEN_X_INDEX);
    context->SetOutputDataType(Q_NOPE_INDEX, outputType);
    context->SetOutputDataType(Q_PE_INDEX, outputType);
    context->SetOutputDataType(CT_KV_INDEX, outputType);
    context->SetOutputDataType(K_PE_INDEX, outputType);
    context->SetOutputDataType(QR_INDEX, outputType);
    context->SetOutputDataType(Q_M_INDEX, outputType);
    context->SetOutputDataType(Q_RSTD_INDEX, ge::DT_FLOAT);
    context->SetOutputDataType(CKV_M_INDEX, outputType);
    context->SetOutputDataType(CKV_RSTD_INDEX, ge::DT_FLOAT);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(DsaSparseProlog).InferShape(InferShapeDsaSparseProlog).InferDataType(InferDataTypeDsaSparseProlog);

} // namespace ops
