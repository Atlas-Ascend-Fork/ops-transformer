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
 * \file dsa_sparse_prolog.cpp
 * \brief DSA(DeepseekSparseAttention) 稀疏路径前向 prolog 算子定义（吸收模式）。
 *
 * 该算子的计算流程如下：
 *   Q : q_a_proj(weight_dq) -> RMSNorm(gamma_cq) -> qr（post-norm，供 Indexer 使用）
 *       qr -> q_b_proj(weight_uq_qr) -> view[*,N,D+Dr] -> split(q_nope, q_pe)
 *       -> 两半同乘 mla_scale_q_lora -> RoPE 作用于 q_pe -> 得 q_nope 与 q_pe
 *   KV: kv_a_proj_with_mqa(weight_dkv_kr) -> split(ct_kv, k_pe)
 *       -> RMSNorm(gamma_ckv) 仅作用 ct_kv -> 乘 mla_scale_kv_lora -> ct_kv 输出
 *       k_pe 为 MQA 单头，RoPE 后输出
 */

#include "register/op_def_registry.h"
#include "dsa_sparse_prolog_tiling.h"

namespace ops {
class DsaSparseProlog : public OpDef {
public:
    DsaSparseProlog(const char *name) : OpDef(name)
    {
        // token_x: (B, S, He) | (T, He)
        this->Input("token_x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // weight_dq: q_a_proj, (He, Hcq)
        this->Input("weight_dq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND})
            .AutoContiguous();
        // weight_uq_qr: q_b_proj, (Hcq, N*(D+Dr))
        this->Input("weight_uq_qr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND})
            .AutoContiguous();
        // weight_dkv_kr: kv_a_proj_with_mqa, (He, Hckv+Dr)
        this->Input("weight_dkv_kr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_FRACTAL_NZ, ge::FORMAT_ND})
            .AutoContiguous();
        // rmsnorm_gamma_cq: (Hcq,)
        this->Input("rmsnorm_gamma_cq")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // rmsnorm_gamma_ckv: (Hckv,)
        this->Input("rmsnorm_gamma_ckv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // rope_sin: (B, S, Dr) | (T, Dr)
        this->Input("rope_sin")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();
        // rope_cos: (B, S, Dr) | (T, Dr)
        this->Input("rope_cos")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND})
            .AutoContiguous();

        // q_nope: (B, S, N, D) | (T, N, D) —— q_nope * mla_scale_q_lora
        this->Output("q_nope")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        // q_pe: (B, S, N, Dr) | (T, N, Dr) —— RoPE(q_pe * mla_scale_q_lora)
        this->Output("q_pe")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        // ct_kv: (B, S, Hckv) | (T, Hckv) —— RMSNorm(ct_kv) * mla_scale_kv_lora
        this->Output("ct_kv")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        // k_pe: (B, S, 1, Dr) | (T, 1, Dr) —— MQA 单头 RoPE(k_pe)
        this->Output("k_pe")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        // qr: (B, S, Hcq) | (T, Hcq) —— q_a_proj 后 RMSNorm 的结果（post-norm，供 Indexer）
        this->Output("qr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        // q_m: (B, S, Hcq) | (T, Hcq) —— q_a_proj 输出（RMSNorm 前），反向梯度中间量
        this->Output("q_m")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        // q_rstd: (B, S, 1) | (T, 1) —— q_a_layernorm 的 rstd（fp32），反向梯度中间量
        this->Output("q_rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});
        // ckv_m: (B, S, Hckv) | (T, Hckv) —— kv_a_proj_with_mqa 输出 split 的 ct_kv（RMSNorm 前），反向梯度中间量
        this->Output("ckv_m")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16, ge::DT_FLOAT16})
            .FormatList({ge::FORMAT_ND});
        // ckv_rstd: (B, S, 1) | (T, 1) —— kv_a_layernorm 的 rstd（fp32），反向梯度中间量
        this->Output("ckv_rstd")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .FormatList({ge::FORMAT_ND});

        this->Attr("rmsnorm_epsilon_cq").AttrType(OPTIONAL).Float(1e-05f);
        this->Attr("rmsnorm_epsilon_ckv").AttrType(OPTIONAL).Float(1e-05f);
        // <=0 时按 sqrt(He/Hcq) 推导
        this->Attr("mla_scale_q_lora").AttrType(OPTIONAL).Float(0.0f);
        // <=0 时按 sqrt(He/Hckv) 推导
        this->Attr("mla_scale_kv_lora").AttrType(OPTIONAL).Float(0.0f);
        // qk_nope_head_dim：无 weight_uk 时 N/D 无法仅从 shape 推导，D 显式传入，N 由
        // weight_uq_qr 宽度 / (D+Dr) 推导
        this->Attr("qk_nope_head_dim").AttrType(OPTIONAL).Int(128);
        // True=与 torch 拼接实现（_process_sparse_inputs）精度一致（含手动降精）；
        // False=更高精度实现（中间量不手动降精）
        this->Attr("is_type_demotion").AttrType(OPTIONAL).Bool(false);
        // False=使用 npu_rotary_mul 对应的 RoPE 语义；True=另一种 RoPE 实现路径
        this->Attr("apply_rope_fusion").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicore_config;
        aicore_config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("aclnnSupport.value", "support_aclnn"); // set value of aclnn support
        this->AICore().AddConfig("ascend910b", aicore_config);
        this->AICore().AddConfig("ascend910_93", aicore_config);
    }
};
OP_ADD(DsaSparseProlog, optiling::DsaSparsePrologCompileInfo);
} // namespace ops
