/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_MATMULV2_HOST_UT_PARAM_H
#define ALL_GATHER_MATMULV2_HOST_UT_PARAM_H

#include <cstdint>
#include <string>
#include <vector>
#include <sstream>
#include "tiling_context_faker.h"
#include "infer_shape_context_faker.h"
#include "op_host_csv_case_loader.h"

// 与 def.cpp 的 ENABLE_ALL_GATHER_MATMUL_DT_HIFLOAT4 / apt_tiling_key.h 的
// ALL_GATHER_MATMUL_V2_HIF4_SUPPORTED 同款 devkit 版本守卫: devkit < 9.3 的平台层
// 不支持 Ascend960DT(9201), ut_hif4 用例(soc=9201)在该环境必然于 tiling 分发处失败,
// 由测试体按此宏 GTEST_SKIP 跳过
#if defined(ASC_DEVKIT_MAJOR)
#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR >= 3)
#define ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED 1
#endif
#elif defined(__has_include)
#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#if ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR >= 3)
#define ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED 1
#endif
#endif
#endif
#ifndef ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED
#define ALL_GATHER_MATMUL_V2_UT_HIF4_SUPPORTED 0
#endif

namespace all_gather_matmul_v2_ut {

struct AllGatherMatmulV2HostUtParamBase {
    std::string case_name;
    std::vector<uint32_t> inputInstance;
    std::vector<uint32_t> outputInstance;
    std::string group;
    bool is_trans_a;
    bool is_trans_b;
    int64_t gather_index;
    int64_t comm_turn;
    int64_t rank_size;
    int64_t block_size;
    int64_t group_size;
    bool is_gather_out;
    bool is_amax_out;
    ge::DataType y_dtype;
    std::string comm_mode;
    ge::graphStatus expectResult;

    explicit AllGatherMatmulV2HostUtParamBase(const csv_map& csvMap)
    {
        this->case_name = ReadMap(csvMap, "case_name");
        this->group = ReadMap(csvMap, "group");
        this->is_trans_a = stoi(ReadMap(csvMap, "is_trans_a"));
        this->is_trans_b = stoi(ReadMap(csvMap, "is_trans_b"));
        this->gather_index = stoll(ReadMap(csvMap, "gather_index"));
        this->comm_turn = stoll(ReadMap(csvMap, "comm_turn"));
        this->rank_size = stoll(ReadMap(csvMap, "rank_size"));
        this->block_size = stoll(ReadMap(csvMap, "block_size"));
        this->group_size = stoll(ReadMap(csvMap, "group_size"));
        this->is_gather_out = stoi(ReadMap(csvMap, "is_gather_out"));
        this->is_amax_out = stoi(ReadMap(csvMap, "is_amax_out"));
        GetDataTypeGE(csvMap, "y_dtype", y_dtype);
        this->comm_mode = ReadMap(csvMap, "comm_mode");
        this->expectResult = ReadMap(csvMap, "expectResult") == "SUCCESS" ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
    }
};

inline std::ostream& operator<<(std::ostream& os, const AllGatherMatmulV2HostUtParamBase& param)
{
    return os << param.case_name;
}

template <typename T>
inline std::string GetCaseInfoString(const testing::TestParamInfo<T>& info)
{
    return info.param.case_name;
}

const gert::TilingContextPara::TensorDescription TD_DEFAULT = {{}, ge::DT_UNDEFINED, ge::FORMAT_NULL};
struct AllGatherMatmulV2TilingUtParam : public AllGatherMatmulV2HostUtParamBase {
    gert::TilingContextPara::TensorDescription x1 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2 = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription bias = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x1_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription x2_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription quant_scale = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription y = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription gather_out = TD_DEFAULT;
    gert::TilingContextPara::TensorDescription amax_out = TD_DEFAULT;
    std::string soc;
    uint64_t coreNum;
    uint64_t ubsize;
    uint64_t expectTilingKey;
    std::string expectTilingDataHash;

    explicit AllGatherMatmulV2TilingUtParam(const csv_map& csvMap)
        : AllGatherMatmulV2HostUtParamBase(csvMap)
    {
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x1_shape", "x1_dtype", "x1_format", x1));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "x2_shape", "x2_dtype", "x2_format", x2));
        this->inputInstance.emplace_back(GetTensorGE(csvMap, "bias_shape", "bias_dtype", "bias_format", bias));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x1_scale_shape", "x1_scale_dtype", "x1_scale_format", x1_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "x2_scale_shape", "x2_scale_dtype", "x2_scale_format", x2_scale));
        this->inputInstance.emplace_back(
            GetTensorGE(csvMap, "quant_scale_shape", "quant_scale_dtype", "quant_scale_format", quant_scale));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "output_y_shape", "output_y_dtype", "output_y_format", y));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "gather_out_shape", "gather_out_dtype", "gather_out_format", gather_out));
        this->outputInstance.emplace_back(
            GetTensorGE(csvMap, "amax_out_shape", "amax_out_dtype", "amax_out_format", amax_out));
        this->soc = ReadMap(csvMap, "soc");
        this->coreNum = stoull(ReadMap(csvMap, "core_num"));
        this->ubsize = stoull(ReadMap(csvMap, "ubsize"));
        if (this->expectResult == ge::GRAPH_SUCCESS) {
            // key 留空 = 跳过 tiling key 断言（对齐框架 SKIP_TILING_KEY_VALIDATION 语义，
            // 与 allto_all_matmul 的 hif4 用例适配一致；hif4 正例已推导并维护真实 key，
            // 仅暂无法推导 key 的版本守卫场景才留空）
            std::string tilingKeyStr = ReadMap(csvMap, "expectTilingKey");
            if (!tilingKeyStr.empty()) {
                this->expectTilingKey = stoull(tilingKeyStr);
            } else {
                this->expectTilingKey = UINT64_MAX; // Skip validation marker
            }
            this->expectTilingDataHash = ReadMap(csvMap, "expectTilingDataHash");
        }
    }
};
} // namespace all_gather_matmul_v2_ut

#endif // ALL_GATHER_MATMULV2_HOST_UT_PARAM_H
