/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_MLA_PARAM_H
#define FLASH_MLA_PARAM_H

#include <string>
#include <vector>
#include <sstream>
#include "op_host_csv_case_loader.h"
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"
#include "../../../op_host/flash_mla_tiling_common.h"

namespace FlashMlaUT {

struct FlashMlaHostUtParamBase : public HostUtParamBase {
    int64_t head_dim_v = 512;
    float softmax_scale = 1.0f;
    int64_t mask_mode = 0;
    int64_t win_left = -1;
    int64_t win_right = -1;
    int64_t max_seqlen_q = -1;
    int64_t max_seqlen_kv = -1;
    std::string layout_q = "BSND";
    std::string layout_kv = "PA_BBND";
    std::string layout_out = "BSND";
    int64_t return_softmax_lse = 0;

    explicit FlashMlaHostUtParamBase(const csv_map& values)
        : HostUtParamBase(values)
    {
        head_dim_v = std::stoll(ReadMap(values, "head_dim_v", "512"));
        softmax_scale = std::stof(ReadMap(values, "softmax_scale", "1.0"));
        mask_mode = std::stoll(ReadMap(values, "mask_mode", "0"));
        win_left = std::stoll(ReadMap(values, "win_left", "-1"));
        win_right = std::stoll(ReadMap(values, "win_right", "-1"));
        max_seqlen_q = std::stoll(ReadMap(values, "max_seqlen_q", "-1"));
        max_seqlen_kv = std::stoll(ReadMap(values, "max_seqlen_kv", "-1"));
        layout_q = ReadMap(values, "layout_q", "BSND");
        layout_kv = ReadMap(values, "layout_kv", "PA_BBND");
        layout_out = ReadMap(values, "layout_out", "BSND");
        return_softmax_lse = std::stoll(ReadMap(values, "return_softmax_lse", "0"));
    }
};

struct FlashMlaTilingUtParam : public FlashMlaHostUtParamBase {
    gert::TilingContextPara::TensorDescription q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription k_cache = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription block_table = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cache_seqlens = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cu_seqlens_q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription seqused_q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription sinks = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription attn_mask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription metadata = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription attn_out = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription softmax_lse = TD_DEFAULT;
    optiling::flash_mla::FlashMlaCompileInfo compileInfo = {
        64, 32, 262144, 524288, 262144, 33554432, platform_ascendc::SocVersion::ASCEND950, NpuArch::DAV_3510};

    explicit FlashMlaTilingUtParam(const csv_map& values)
        : FlashMlaHostUtParamBase(values)
    {
        inputInstance.emplace_back(GetTensorGE(values, "q_shape", "q_dtype", "q_format", q));
        inputInstance.emplace_back(GetTensorGE(values, "k_cache_shape", "k_cache_dtype", "k_cache_format", k_cache));
        ApplyStrideFromCsv(values, "k_cache_stride", k_cache);
        inputInstance.emplace_back(
            GetTensorGE(values, "block_table_shape", "block_table_dtype", "block_table_format", block_table));
        inputInstance.emplace_back(
            GetTensorGE(values, "cache_seqlens_shape", "cache_seqlens_dtype", "cache_seqlens_format", cache_seqlens));
        inputInstance.emplace_back(
            GetTensorGE(values, "cu_seqlens_q_shape", "cu_seqlens_q_dtype", "cu_seqlens_q_format", cu_seqlens_q));
        inputInstance.emplace_back(
            GetTensorGE(values, "seqused_q_shape", "seqused_q_dtype", "seqused_q_format", seqused_q));
        inputInstance.emplace_back(GetTensorGE(values, "sinks_shape", "sinks_dtype", "sinks_format", sinks));
        inputInstance.emplace_back(
            GetTensorGE(values, "attn_mask_shape", "attn_mask_dtype", "attn_mask_format", attn_mask));
        inputInstance.emplace_back(
            GetTensorGE(values, "metadata_shape", "metadata_dtype", "metadata_format", metadata));
        outputInstance.emplace_back(
            GetTensorGE(values, "attn_out_shape", "attn_out_dtype", "attn_out_format", attn_out));
        outputInstance.emplace_back(
            GetTensorGE(values, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", softmax_lse));
    }
};

struct FlashMlaInferShapeUtParam : public FlashMlaHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription k_cache = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription block_table = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription cache_seqlens = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription cu_seqlens_q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription seqused_q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription sinks = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription attn_mask = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription metadata = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription attn_out = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription softmax_lse = ID_DEFAULT;
    std::vector<std::vector<int64_t>> expectOutputShape;

    explicit FlashMlaInferShapeUtParam(const csv_map& values)
        : FlashMlaHostUtParamBase(values)
    {
        inputInstance.emplace_back(GetTensorGE(values, "q_shape", "q_dtype", "q_format", q));
        inputInstance.emplace_back(GetTensorGE(values, "k_cache_shape", "k_cache_dtype", "k_cache_format", k_cache));
        inputInstance.emplace_back(
            GetTensorGE(values, "block_table_shape", "block_table_dtype", "block_table_format", block_table));
        inputInstance.emplace_back(
            GetTensorGE(values, "cache_seqlens_shape", "cache_seqlens_dtype", "cache_seqlens_format", cache_seqlens));
        inputInstance.emplace_back(
            GetTensorGE(values, "cu_seqlens_q_shape", "cu_seqlens_q_dtype", "cu_seqlens_q_format", cu_seqlens_q));
        inputInstance.emplace_back(
            GetTensorGE(values, "seqused_q_shape", "seqused_q_dtype", "seqused_q_format", seqused_q));
        inputInstance.emplace_back(GetTensorGE(values, "sinks_shape", "sinks_dtype", "sinks_format", sinks));
        inputInstance.emplace_back(
            GetTensorGE(values, "attn_mask_shape", "attn_mask_dtype", "attn_mask_format", attn_mask));
        inputInstance.emplace_back(
            GetTensorGE(values, "metadata_shape", "metadata_dtype", "metadata_format", metadata));
        outputInstance.emplace_back(
            GetTensorGE(values, "attn_out_shape", "attn_out_dtype", "attn_out_format", attn_out));
        outputInstance.emplace_back(
            GetTensorGE(values, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", softmax_lse));
        if (expectResult == ge::GRAPH_SUCCESS) {
            for (const auto* key : {"expect_attn_out_shape", "expect_softmax_lse_shape"}) {
                std::stringstream text(ReadMap(values, key, ""));
                std::vector<int64_t> shape;
                int64_t dim;
                while (text >> dim) {
                    shape.push_back(dim);
                }
                expectOutputShape.push_back(shape);
            }
        }
    }
};

struct FlashMlaInferDTypeUtParam : public FlashMlaHostUtParamBase {
    ge::DataType q_dtype = ge::DT_UNDEFINED;
    ge::DataType k_cache_dtype = ge::DT_UNDEFINED;
    ge::DataType expect_attn_out_dtype = ge::DT_UNDEFINED;
    ge::DataType expect_softmax_lse_dtype = ge::DT_UNDEFINED;

    explicit FlashMlaInferDTypeUtParam(const csv_map& values)
        : FlashMlaHostUtParamBase(values)
    {
        GetDataTypeGE(values, "q_dtype", q_dtype);
        GetDataTypeGE(values, "k_cache_dtype", k_cache_dtype);
        GetDataTypeGE(values, "expect_attn_out_dtype", expect_attn_out_dtype);
        GetDataTypeGE(values, "expect_softmax_lse_dtype", expect_softmax_lse_dtype);
    }
};

} // namespace FlashMlaUT
#endif // FLASH_MLA_PARAM_H
