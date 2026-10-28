/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <iostream>

#include "../../../../op_host/compressor_grad_tiling.h"
#include "../../../../op_kernel/arch22/compressor_grad_tiling_data.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;

// Ascend910_93 (A3 / arch22) SoC info: 24 cube cores, 48 vector cores, UB 192KB
std::string CompressorGrad_tiling_A3SocInfo = "{\n"
                                              "  \"hardware_info\": {\n"
                                              "    \"BT_SIZE\": 0,\n"
                                              "    \"load3d_constraints\": \"1\",\n"
                                              "    \"Intrinsic_fix_pipe_l0c2out\": false,\n"
                                              "    \"Intrinsic_data_move_l12ub\": true,\n"
                                              "    \"Intrinsic_data_move_l0c2ub\": true,\n"
                                              "    \"Intrinsic_data_move_out2l1_nd2nz\": false,\n"
                                              "    \"4096\": 196608,\n"
                                              "    \"L2_SIZE\": 201326592,\n"
                                              "    \"L1_SIZE\": 524288,\n"
                                              "    \"L0A_SIZE\": 65536,\n"
                                              "    \"L0B_SIZE\": 65536,\n"
                                              "    \"L0C_SIZE\": 131072,\n"
                                              "    \"vector_core_cnt\": 48,\n"
                                              "    \"cube_core_cnt\": 24,\n"
                                              "    \"socVersion\": \"Ascend910_93\"\n"
                                              "  }\n"
                                              "}";

// ====================================================================
// arch22 tiling key space (op_kernel/arch22/compressor_grad_template_tiling_key.h):
//   single field D_TYPE (2bit, LIST {0, 1}): 0 = BF16, 1 = FP16
//   GET_TPL_TILING_KEY encodes the list index, so key == dtype code.
// ====================================================================

class CompressorGradTilingArch22Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "CompressorGradTilingArch22Test SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "CompressorGradTilingArch22Test TearDown" << std::endl;
    }
};

// ====================================================================
// BSH Layout Tiling Tests
// ====================================================================

// C4A bf16: B=2, S=8, H=4096, D=512, coff=2, cmp_ratio=4
TEST_F(CompressorGradTilingArch22Test, bsh_c4a_bf16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            // Required inputs
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // x
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // wgate
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},        // d_cmp_kv
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv
                                                                             // Optional inputs (empty for BSH)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cu_seqlens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // seqused
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // start_pos
        },
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // d_x
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // d_wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // d_wgate
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},      // d_ape
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 0; // BF16
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// C4A fp16: B=2, S=8, H=4096, D=512, coff=2, cmp_ratio=4
TEST_F(CompressorGradTilingArch22Test, bsh_c4a_fp16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 2, 512}, {2, 2, 512}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 1; // FP16
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// C4Li bf16: B=2, S=8, H=2048, D=128, coff=2, cmp_ratio=4
TEST_F(CompressorGradTilingArch22Test, bsh_c4li_bf16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 128}, {2, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 128}, {2, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 128}, {2, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// C128A bf16: B=2, S=256, H=4096, D=512, coff=1, cmp_ratio=128 (nblk=2; arch22 要求 S>=cr)
TEST_F(CompressorGradTilingArch22Test, bsh_c128a_bf16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 256, 4096}, {2, 256, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 128, 512}, {2, 2, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2, 128, 512}, {2, 2, 128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 256, 4096}, {2, 256, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{512, 4096}, {512, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 512}, {128, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(128)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// Non-divisible S: B=1, S=6, H=2048, D=128, coff=2, cmp_ratio=4 (nblk=1, nblkIn=2)
TEST_F(CompressorGradTilingArch22Test, bsh_non_divisible_s)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{1, 6, 2048}, {1, 6, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 128}, {1, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 8, 128}, {1, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 2, 8, 128}, {1, 2, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 6, 2048}, {1, 6, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 0;
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// ====================================================================
// TH Layout Tiling Tests (cu_seqlens based, A5-aligned)
// ====================================================================

// TH basic: T=8, B=2 (cu_seqlens 3), H=4096, D=512, coff=2, cmp_ratio=4;
// d_cmp_kv/softmax_score/kv rows = outputRows = min(T, T/cr+B) = 4
TEST_F(CompressorGradTilingArch22Test, th_basic_bf16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},       // x (T, H)
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wgate
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // d_cmp_kv (rows, D)
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // softmax_score (rows, coff*cr, D)
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // kv
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},                  // cu_seqlens (B+1)
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // seqused
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // start_pos
        },
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 0; // BF16
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// TH fp16 + seqused + start_pos (metadata arrays consumed by the kernel):
// T=128, B=2 (cu_seqlens 3), H=2048, D=128, coff=1, cr=2,
// rows = outputRows = min(T, T/cr+B) = 66
TEST_F(CompressorGradTilingArch22Test, th_seqused_startpos_fp16)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{66, 128}, {66, 128}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{66, 2, 128}, {66, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{66, 2, 128}, {66, 2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_FLOAT16, ge::FORMAT_ND},
                                                  {{{2, 128}, {2, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    int64_t expectTilingKey = 1; // FP16
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey);
}

// TH layout without cu_seqlens -> reject (existence check)
TEST_F(CompressorGradTilingArch22Test, err_th_without_cu_seqlens)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},       // x (T, H)
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wgate
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // d_cmp_kv (rows, D)
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // softmax_score
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                    // cu_seqlens missing
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// TH layout with row-count mismatch between softmax_score/kv and d_cmp_kv -> reject
TEST_F(CompressorGradTilingArch22Test, err_th_rows_mismatch)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},       // x (T, H)
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wgate
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // d_cmp_kv (rows, D)
            {{{2, 8, 512}, {2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // softmax_score rows=2 != 4
            {{{2, 8, 512}, {2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // kv rows=2 != 4
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},                  // cu_seqlens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// ====================================================================
// TH 元数据结构性负例（numel/dtype）
// ====================================================================

// TH cu_seqlens numel=1（B=0）-> reject：cu_seqlens 必须 numel == B+1 >= 2
// （numel=0 由 CheckEmptyTensor 拦截；numel=1 时 B=0 空批次不支持）
TEST_F(CompressorGradTilingArch22Test, err_th_cu_seqlens_numel_one_b0)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},       // x (T, H)
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wgate
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // d_cmp_kv (rows, D)
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // softmax_score
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // kv
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                  // cu_seqlens numel=1 -> B=0
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// TH cu_seqlens numel=0（空 tensor）-> reject（CheckEmptyTensor 扩展：B 下溢防护）
TEST_F(CompressorGradTilingArch22Test, err_th_cu_seqlens_numel_zero)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},       // x (T, H)
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wgate
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // d_cmp_kv (rows, D)
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // softmax_score
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // kv
            {{{0}, {0}}, ge::DT_INT32, ge::FORMAT_ND},                  // cu_seqlens numel=0
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// TH seqused numel=1 < B=2 -> reject（seqused numel 必须 == B）
TEST_F(CompressorGradTilingArch22Test, err_th_seqused_numel_short)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},       // x (T, H)
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wgate
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // d_cmp_kv (rows, D)
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // softmax_score
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // kv
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},                  // cu_seqlens (B=2)
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},                  // seqused numel=1 != 2
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// TH seqused dtype=INT64（仅支持 INT32）-> reject
TEST_F(CompressorGradTilingArch22Test, err_th_seqused_dtype_int64)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},       // x (T, H)
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wgate
            {{{4, 512}, {4, 512}}, ge::DT_BF16, ge::FORMAT_ND},         // d_cmp_kv (rows, D)
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // softmax_score
            {{{4, 8, 512}, {4, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // kv
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},                  // cu_seqlens (B=2)
            {{{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND},                  // seqused dtype != INT32
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// ====================================================================
// Error Cases — expected FAIL
// ====================================================================

// TH layout but 3D d_cmp_kv（dimnum 与 2D x 不一致；DIM_NUM_MAP 单查允许 {2,3}，
// 一致性检查前置后由 CheckDimNumConsistency 拒绝，不再落入 shape 分支的取维）
// -> reject
TEST_F(CompressorGradTilingArch22Test, err_th_dimnum_mismatch)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},       // x (T, H)
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND}, // wgate
            {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},   // d_cmp_kv 3D（应为 2D）
            {{{2, 8, 512}, {2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // softmax_score (B, coff*cmpRatio, D)
            {{{2, 8, 512}, {2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},  // kv
            {{{3}, {3}}, ge::DT_INT32, ge::FORMAT_ND},                  // cu_seqlens
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 4096}, {8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// BSH layout but 2D d_cmp_kv（x 3D，dimnum 不一致；单查 DIM_NUM_MAP 允许 {2,3}）。
// 一致性检查前置前，BSH shape 分支会对 2D d_cmp_kv GetDim(2) 越 rank；前置后由
// CheckDimNumConsistency 干净拒绝 -> reject
TEST_F(CompressorGradTilingArch22Test, err_bsh_dimnum_mismatch)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // x (B, S, H)
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // wkv
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},      // wgate
            {{{2, 512}, {2, 512}}, ge::DT_BF16, ge::FORMAT_ND},              // d_cmp_kv 2D（应为 3D [B,nblk,D]）
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // softmax_score
            {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND}, // kv
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},                         // cu_seqlens（BSH 必空）
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// S=2 < cmp_ratio=4（无完整块）不拒绝——nblk=0 合法，scatter 跳过全部块，
// dn/d_ape 预清零 + gemm 全行覆盖 => 四输出全零（A5 _zero_outputs / golden 空循环语义）
TEST_F(CompressorGradTilingArch22Test, no_complete_block_all_zero)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{1, 2, 2048}, {1, 2, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 128}, {1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 8, 128}, {1, 1, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1, 8, 128}, {1, 1, 8, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 2, 2048}, {1, 2, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{256, 2048}, {256, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 256}, {4, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0); // 全零输出，正常返回
}

// Unsupported cmp_ratio=3 (interval check: [2, 128])
TEST_F(CompressorGradTilingArch22Test, err_cmp_ratio_3)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// Unsupported coff=3 (only 1 and 2 supported)
TEST_F(CompressorGradTilingArch22Test, err_coff_3)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 4096}, {2, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(3)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// Unsupported headDim=256 (only 128 and 512 supported)
TEST_F(CompressorGradTilingArch22Test, err_head_dim_256)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara(
        "CompressorGrad",
        {
            {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 2048}, {512, 2048}}, ge::DT_BF16, ge::FORMAT_ND}, // coff*headDim = 2*256 = 512
            {{{512, 2048}, {512, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 256}, {2, 2, 256}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{2, 2, 8, 256}, {2, 2, 8, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 2, 8, 256}, {2, 2, 8, 256}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 2048}, {512, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{512, 2048}, {512, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
            {{{4, 512}, {4, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
            {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
        },
        &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// Unsupported hiddenSize=3000 (not 512-aligned, must be in [1024, 10240])
TEST_F(CompressorGradTilingArch22Test, err_hidden_size_not_aligned)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 8, 3000}, {2, 8, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 3000}, {1024, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 3000}, {1024, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 512}, {2, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2, 8, 512}, {2, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 3000}, {2, 8, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 3000}, {1024, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 3000}, {1024, 3000}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// Empty tensor: x has zero-size shape -> should fail (backward does not support empty tensors)
TEST_F(CompressorGradTilingArch22Test, err_empty_x_tensor)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{0, 8, 4096}, {0, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{0, 2, 512}, {0, 2, 512}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{0, 2, 8, 512}, {0, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{0, 2, 8, 512}, {0, 2, 8, 512}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{0, 8, 4096}, {0, 8, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1024, 4096}, {1024, 4096}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 1024}, {4, 1024}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(2)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_FAILED, 0);
}

// ====================================================================
// seqused / start_pos 值语义。tiling 侧为 kernel-only 设计：
// 只按"张量是否存在"置 flags 并取安全上界 nblk_ = min(S/cr+1, nblkIn)，
// 实际块数由 scatter 在设备端按 GM 值逐 batch 计算（j >= (sp+sq)/cr 跳过）。
// faker 的常量张量在本设计下只决定 desc 是否实例化（值不被 tiling 读）。
// 字段断言：直接 include tiling data 头，按命名成员取值（不按 int64 偏移
// reinterpret_cast——偏移断言与字段顺序强耦合，加字段即碎）
// ====================================================================

static void CheckNblkFields(const gert::TilingContextPara& p, int64_t expNblk, int64_t expSeqValid, int64_t expSpValid)
{
    TilingInfo info;
    bool ok = ExecuteTiling(p, info);
    ASSERT_TRUE(ok) << "tiling should succeed";
    const auto* td = reinterpret_cast<const optiling::CompressorGradTilingDataA3*>(info.tilingData.get());
    ASSERT_GE(info.tilingDataSize, static_cast<size_t>(sizeof(optiling::CompressorGradTilingDataA3)));
    EXPECT_EQ(td->nblk, expNblk) << "nblk";
    EXPECT_EQ(td->seqUsedValid, expSeqValid) << "seqUsedValid";
    EXPECT_EQ(td->startPosValid, expSpValid) << "startPosValid";
}

// seqused 值语义：B=2, S=8, cr=4, seqused=[8,5]（实际块 max(2,1)=2）；
// kernel-only tiling：nblk_ = min(S/cr+1, nblkIn) = min(3, 2) = 2
TEST_F(CompressorGradTilingArch22Test, seqused_value_nblk)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    int32_t seqUsed[2] = {8, 5};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 128}, {2, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 2, 4, 128}, {2, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 2, 4, 128}, {2, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsed},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 128}, {4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
    CheckNblkFields(tilingContextPara, 2, 1, 0);
}

// seqused 值 + dCmpKv 欠配（nblkIn=1 < 实际 2 块）：kernel-only tiling 不读值，
// 无法在 tiling 期判欠配 → 成功（nblk_ = min(3,1) = 1），kernel 只处理前缀块
// （欠配属调用方契约违约；A5 BSH 同样不校验 dim1）
TEST_F(CompressorGradTilingArch22Test, seqused_value_underalloc_prefix)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    int32_t seqUsed[2] = {8, 5};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 1, 128}, {2, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{2, 1, 4, 128}, {2, 1, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{2, 1, 4, 128}, {2, 1, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsed},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{2, 8, 2048}, {2, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 128}, {4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
    CheckNblkFields(tilingContextPara, 1, 1, 0);
}

// seqused 值越界（> S）：kernel-only tiling 不读值 → 成功（nblk_ = min(3,2) = 2），
// 设备端 UsedBlocks 对 sq 做 [0,S] clamp（越界值按 S 处理）
TEST_F(CompressorGradTilingArch22Test, seqused_value_range_clamped)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    int32_t seqUsed[1] = {9};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 128}, {1, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 4, 128}, {1, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 2, 4, 128}, {1, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsed},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 128}, {4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
    CheckNblkFields(tilingContextPara, 2, 1, 0);
}

// seqused 张量存在但值不可读（faker isConst=false → data=nullptr）→ kernel-only
// 回退：tiling 置 flag + nblk 取安全上界 min(S/cr+1, nblkIn)，设备端读值兜底
TEST_F(CompressorGradTilingArch22Test, seqused_value_unreadable_fallback)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 128}, {1, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 4, 128}, {1, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 2, 4, 128}, {1, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 128}, {4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    // S=8, cr=4 → shape 上界 2，nblkIn=2 → 回退 nblk = min(2+1, 2) = 2
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
    CheckNblkFields(tilingContextPara, 2, 1, 0);
}

// seqused 值 < cr（0 完整块）→ nblk=0 合法成功（全零输出）
TEST_F(CompressorGradTilingArch22Test, seqused_value_zero_blocks)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    int32_t seqUsed[1] = {3};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{1, 3, 2048}, {1, 3, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 128}, {1, 1, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 1, 4, 128}, {1, 1, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 1, 4, 128}, {1, 1, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsed},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1, 3, 2048}, {1, 3, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 128}, {4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
    // kernel-only：nblk_ 是安全上界 min(S/cr+1, nblkIn) = min(0+1, 1) = 1（非精确 0）；
    // 设备端 used=(0+3)/4=0 → 全部块跳过 → 输出全零（全零输出语义由 kernel 保证）
    CheckNblkFields(tilingContextPara, 1, 1, 0);
}

// start_pos 值语义：sp=3, S=8, cr=4 → nblk = (3+8)/4 = 2（首块为含 cache 行的部分块）
TEST_F(CompressorGradTilingArch22Test, startpos_value_anchor)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    int32_t startPos[1] = {3};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 128}, {1, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 4, 128}, {1, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 2, 4, 128}, {1, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, startPos},
                                              },
                                              {
                                                  {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 128}, {4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
    CheckNblkFields(tilingContextPara, 2, 0, 1);
}

// start_pos 负值：kernel-only tiling 不读值 → 成功（nblk_ = min(3,2) = 2）；
// 设备端 static_cast<uint32>(负值) % cr 回绕为 <cr 的合法偏移（垃圾容错，不越界）
TEST_F(CompressorGradTilingArch22Test, startpos_negative_clamped)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    int32_t startPos[1] = {-1};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 128}, {1, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 4, 128}, {1, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 2, 4, 128}, {1, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, startPos},
                                              },
                                              {
                                                  {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 128}, {4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
    CheckNblkFields(tilingContextPara, 2, 0, 1);
}

// start_pos + seqused 组合：sp=3(sq 无) → nblk=(3+8)/4=2；再加 seqused=[6] → (3+6)/4=2
TEST_F(CompressorGradTilingArch22Test, startpos_plus_seqused_value)
{
    optiling::CompressorGradCompileInfo compileInfo = {};
    int32_t startPos[1] = {3};
    int32_t seqUsed[1] = {6};
    gert::TilingContextPara tilingContextPara("CompressorGrad",
                                              {
                                                  {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 128}, {1, 2, 128}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{1, 2, 4, 128}, {1, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{1, 2, 4, 128}, {1, 2, 4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                                  {{{}, {}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, seqUsed},
                                                  {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND, true, startPos},
                                              },
                                              {
                                                  {{{1, 8, 2048}, {1, 8, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{128, 2048}, {128, 2048}}, ge::DT_BF16, ge::FORMAT_ND},
                                                  {{{4, 128}, {4, 128}}, ge::DT_FLOAT, ge::FORMAT_ND},
                                              },
                                              {
                                                  {"cmp_ratio", Ops::Transformer::AnyValue::CreateFrom<int64_t>(4)},
                                                  {"coff", Ops::Transformer::AnyValue::CreateFrom<int64_t>(1)},
                                              },
                                              &compileInfo, "Ascend910_93", CompressorGrad_tiling_A3SocInfo, 4096);
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, 0);
    CheckNblkFields(tilingContextPara, 2, 1, 1);
}
