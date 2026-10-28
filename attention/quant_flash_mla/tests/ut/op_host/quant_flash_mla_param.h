/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_MLA_PARAM_H
#define QUANT_FLASH_MLA_PARAM_H

#include <memory>
#include <string>
#include <vector>
#include "op_host_csv_case_loader.h"
#include "../../../op_host/quant_flash_mla_tiling_common.h"

namespace QuantFlashMlaUT {

struct QuantFlashMlaHostUtParamBase : public HostUtParamBase {
    int64_t quant_compute_mode = 1;
    float softmax_scale = 1.0f;
    int64_t mask_mode = 0;
    int64_t max_seqlen_q = -1;
    int64_t max_seqlen_kv = -1;
    int64_t head_dim_v = 512;
    std::string layout_q = "BSND";
    std::string layout_kv = "PA_BNBD";
    std::string layout_out = "BSND";
    bool return_softmax_lse = false;

    QuantFlashMlaHostUtParamBase(const csv_map& csvMap)
        : HostUtParamBase(csvMap)
    {
        this->quant_compute_mode = std::stoll(ReadMap(csvMap, "quant_compute_mode", "1"));
        this->softmax_scale = std::stof(ReadMap(csvMap, "softmax_scale", "1.0"));
        this->mask_mode = std::stoll(ReadMap(csvMap, "mask_mode", "0"));
        this->max_seqlen_q = std::stoll(ReadMap(csvMap, "max_seqlen_q", "-1"));
        this->max_seqlen_kv = std::stoll(ReadMap(csvMap, "max_seqlen_kv", "-1"));
        this->head_dim_v = std::stoll(ReadMap(csvMap, "head_dim_v", "512"));
        this->layout_q = ReadMap(csvMap, "layout_q", "BSND");
        this->layout_kv = ReadMap(csvMap, "layout_kv", "PA_BNBD");
        this->layout_out = ReadMap(csvMap, "layout_out", "BSND");
        this->return_softmax_lse = StrToBoolIgnoreCase(ReadMap(csvMap, "return_softmax_lse", "false"));
    }
};

struct QuantFlashMlaTilingUtParam : public QuantFlashMlaHostUtParamBase {
    gert::TilingContextPara::TensorDescription q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription k_cache = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription q_descale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription k_descale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription block_table = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cache_seqlens = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription cu_seqlens_q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription seqused_q = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription attn_mask = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription metadata = TD_DEFAULT;

    gert::TilingContextPara::TensorDescription attn_out = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription softmax_lse = TD_DEFAULT;

    uint64_t expectTilingKey = UINT64_MAX;
    std::string expectTilingDataHash;

    QuantFlashMlaTilingUtParam(const csv_map& csvMap)
        : QuantFlashMlaHostUtParamBase(csvMap)
    {
        // 输入顺序与 quant_flash_mla_def.cpp 中 Input() 注册顺序保持一致
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "q_shape", "q_dtype", "q_format", this->q));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "k_cache_shape", "k_cache_dtype", "k_cache_format", this->k_cache));
        ApplyStrideFromCsv(csvMap, "k_cache_stride", this->k_cache);
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "q_descale_shape", "q_descale_dtype", "q_descale_format", this->q_descale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "k_descale_shape", "k_descale_dtype", "k_descale_format", this->k_descale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "block_table_shape", "block_table_dtype", "block_table_format", this->block_table));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "cache_seqlens_shape", "cache_seqlens_dtype",
                                                     "cache_seqlens_format", this->cache_seqlens));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "cu_seqlens_q_shape", "cu_seqlens_q_dtype", "cu_seqlens_q_format", this->cu_seqlens_q));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "seqused_q_shape", "seqused_q_dtype", "seqused_q_format", this->seqused_q));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_mask_shape", "attn_mask_dtype", "attn_mask_format", this->attn_mask));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "metadata_shape", "metadata_dtype", "metadata_format", this->metadata));

        // 输出
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_out_shape", "attn_out_dtype", "attn_out_format", this->attn_out));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", this->softmax_lse));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            const std::string tilingKeyStr = ReadMap(csvMap, "expectTilingKey");
            // 留空时使用 SKIP_TILING_KEY_VALIDATION(UINT64_MAX) 跳过 tiling key 校验
            this->expectTilingKey = tilingKeyStr.empty() ? UINT64_MAX : std::stoull(tilingKeyStr);
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }
};

struct QuantFlashMlaInferShapeUtParam : public QuantFlashMlaHostUtParamBase {
    gert::InfershapeContextPara::TensorDescription q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription k_cache = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription q_descale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription k_descale = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription block_table = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription cache_seqlens = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription cu_seqlens_q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription seqused_q = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription attn_mask = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription metadata = ID_DEFAULT;

    gert::InfershapeContextPara::TensorDescription attn_out = ID_DEFAULT;
    gert::InfershapeContextPara::TensorDescription softmax_lse = ID_DEFAULT;

    std::vector<std::vector<int64_t>> expectOutputShape;

    QuantFlashMlaInferShapeUtParam(const csv_map& csvMap)
        : QuantFlashMlaHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "q_shape", "q_dtype", "q_format", this->q));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "k_cache_shape", "k_cache_dtype", "k_cache_format", this->k_cache));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "q_descale_shape", "q_descale_dtype", "q_descale_format", this->q_descale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "k_descale_shape", "k_descale_dtype", "k_descale_format", this->k_descale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "block_table_shape", "block_table_dtype", "block_table_format", this->block_table));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "cache_seqlens_shape", "cache_seqlens_dtype",
                                                     "cache_seqlens_format", this->cache_seqlens));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "cu_seqlens_q_shape", "cu_seqlens_q_dtype", "cu_seqlens_q_format", this->cu_seqlens_q));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "seqused_q_shape", "seqused_q_dtype", "seqused_q_format", this->seqused_q));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_mask_shape", "attn_mask_dtype", "attn_mask_format", this->attn_mask));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "metadata_shape", "metadata_dtype", "metadata_format", this->metadata));

        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "attn_out_shape", "attn_out_dtype", "attn_out_format", this->attn_out));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "softmax_lse_shape", "softmax_lse_dtype", "softmax_lse_format", this->softmax_lse));

        if (this->expectResult == ge::GRAPH_SUCCESS) {
            this->expectOutputShape = {GetShapeArr(ReadMap(csvMap, "attn_out_shape")),
                                       GetShapeArr(ReadMap(csvMap, "softmax_lse_shape"))};
        }
    }
};

struct QuantFlashMlaInferDTypeUtParam : public QuantFlashMlaHostUtParamBase {
    ge::DataType q = ge::DT_UNDEFINED;
    ge::DataType k_cache = ge::DT_UNDEFINED;
    ge::DataType q_descale = ge::DT_UNDEFINED;
    ge::DataType k_descale = ge::DT_UNDEFINED;
    ge::DataType block_table = ge::DT_UNDEFINED;
    ge::DataType cache_seqlens = ge::DT_UNDEFINED;
    ge::DataType cu_seqlens_q = ge::DT_UNDEFINED;
    ge::DataType seqused_q = ge::DT_UNDEFINED;
    ge::DataType attn_mask = ge::DT_UNDEFINED;
    ge::DataType metadata = ge::DT_UNDEFINED;

    ge::DataType attn_out = ge::DT_UNDEFINED;
    ge::DataType softmax_lse = ge::DT_UNDEFINED;

    QuantFlashMlaInferDTypeUtParam(const csv_map& csvMap)
        : QuantFlashMlaHostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "q_dtype", this->q));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "k_cache_dtype", this->k_cache));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "q_descale_dtype", this->q_descale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "k_descale_dtype", this->k_descale));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "block_table_dtype", this->block_table));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "cache_seqlens_dtype", this->cache_seqlens));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "cu_seqlens_q_dtype", this->cu_seqlens_q));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "seqused_q_dtype", this->seqused_q));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "attn_mask_dtype", this->attn_mask));
        this->inputInstance.emplace_back(GetDataTypeGE(csvMap, "metadata_dtype", this->metadata));

        this->outputInstance.emplace_back(GetDataTypeGE(csvMap, "attn_out_dtype", this->attn_out));
        this->outputInstance.emplace_back(GetDataTypeGE(csvMap, "softmax_lse_dtype", this->softmax_lse));
    }
};

} // namespace QuantFlashMlaUT

#endif // QUANT_FLASH_MLA_PARAM_H
