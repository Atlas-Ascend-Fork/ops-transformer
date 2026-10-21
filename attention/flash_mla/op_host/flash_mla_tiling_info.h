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
 * \file flash_mla_tiling_info.h
 * \brief
 */

#ifndef FLASH_MLA_FA_TILING_INFO_H
#define FLASH_MLA_FA_TILING_INFO_H

#include <map>
#include <set>
#include <string>
#include <vector>
#include <cstdint>
#include "../../common/op_host/fia_tiling_base.h"
#include "../../common/op_host/fia_tiling_shape.h"

namespace optiling {
namespace flash_mla {
const std::string QUERY_NAME = "q";
const std::string K_CACHE_NAME = "k_cache";
const std::string BLOCK_TABLE_NAME = "block_table";
const std::string CACHE_SEQLENS_NAME = "cache_seqlens";
const std::string CU_SEQLENS_Q_NAME = "cu_seqlens_q";
const std::string SEQUSED_Q_NAME = "seqused_q";
const std::string METADATA_NAME = "metadata";
const std::string SOFTMAX_SCALE_NAME = "softmax_scale";
const std::string MASK_MODE_NAME = "mask_mode";
const std::string ATTN_MASK_NAME = "attn_mask";
const std::string HEAD_DIM_V_NAME = "head_dim_v";
const std::string LAYOUT_Q_NAME = "layout_q";
const std::string LAYOUT_KV_NAME = "layout_kv";
const std::string LAYOUT_OUT_NAME = "layout_out";
const std::string RETURN_SOFTMAX_LSE_NAME = "return_softmax_lse";
const std::string ATTN_OUT_NAME = "attn_out";
const std::string SOFTMAX_LSE_NAME = "softmax_lse";

constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t K_CACHE_INDEX = 1;
constexpr uint32_t BLOCK_TABLE_INDEX = 2;
constexpr uint32_t CACHE_SEQLENS_INDEX = 3;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 4;
constexpr uint32_t SEQUSED_Q_INDEX = 5;
constexpr uint32_t ATTN_MASK_INDEX = 6;
constexpr uint32_t METADATA_INDEX = 7;

constexpr uint32_t ATTR_HEAD_DIM_V_INDEX = 0;    // head_dim_v (nope/value width inside k_cache, default 512)
constexpr uint32_t ATTR_SOFTMAX_SCALE_INDEX = 1; // softmax_scale
constexpr uint32_t ATTR_MASK_MODE_INDEX = 2;     // mask_mode
constexpr uint32_t ATTR_MAX_SEQLEN_Q_INDEX = 3;  // max_seqlen_q
constexpr uint32_t ATTR_MAX_SEQLEN_KV_INDEX = 4; // max_seqlen_kv
constexpr uint32_t ATTR_LAYOUT_Q_INDEX = 5;      // layout_q
constexpr uint32_t ATTR_LAYOUT_KV_INDEX = 6;     // layout_kv
constexpr uint32_t ATTR_LAYOUT_OUT_INDEX = 7;    // layout_out
constexpr uint32_t ATTR_RETURN_LSE_INDEX = 8;    // return_softmax_lse

constexpr uint32_t ATTN_OUT_INDEX = 0;
constexpr uint32_t SOFTMAX_LSE_INDEX = 1;

constexpr int64_t MASK_MODE_INT_MAX = 2147483647;

enum class MaskMode : int32_t {
    NO_MASK = 0,
    CAUSAL = 3,
    BAND = 4
};

using FlashMlaLayout = FiaLayout;
using FlashMlaAxis = FiaAxis;
using FlashMlaCompareType = FiaCompareType;
using FlashMlaTilingShapeCompareParam = FiaTilingShapeCompareParam;
using FlashMlaTilingShape = FiaTilingShape;
using FlashMlaTilingShapeCompare = FiaTilingShapeCompare;

const std::map<std::string, FlashMlaLayout> layoutMap = {
    {"BSND", FlashMlaLayout::BSND},  {"BNSD", FlashMlaLayout::BNSD},       {"TND", FlashMlaLayout::TND},
    {"NTD", FlashMlaLayout::NTD},    {"PA_BBND", FlashMlaLayout::PA_BBND}, {"PA_BNBD", FlashMlaLayout::PA_BNBD},
    {"PA_NZ", FlashMlaLayout::PA_NZ}};

enum class FlashMlaQuantMode : uint32_t {
    NO_QUANT = 0,
    ANTI_QUANT,
    FULL_QUANT
};

enum class KvStorageMode : uint32_t {
    BATCH_CONTINUOUS = 0,
    PAGE_ATTENTION = 1
};

enum class FlashMlaTilingInOutMode : uint32_t {
    FP16_FP16 = 0,
    BF16_BF16 = 1
};

std::string QuantModeToSerialString(FlashMlaQuantMode faQuantMode);

struct FlashMlaRequiredParaInfo {
    const gert::CompileTimeTensorDesc* desc;
    const gert::StorageShape* shape;
};

struct FlashMlaOptionalParaInfo {
    const gert::CompileTimeTensorDesc* desc;
    const gert::Tensor* tensor;
};

struct FlashMlaParaInfo {
    FlashMlaRequiredParaInfo query = {nullptr, nullptr};
    FlashMlaRequiredParaInfo kCache = {nullptr, nullptr};

    FlashMlaOptionalParaInfo blockTable = {nullptr, nullptr};
    FlashMlaOptionalParaInfo cacheSeqlens = {nullptr, nullptr};
    FlashMlaOptionalParaInfo cuSeqlensQ = {nullptr, nullptr};
    FlashMlaOptionalParaInfo sequsedQ = {nullptr, nullptr};
    FlashMlaOptionalParaInfo attnMask = {nullptr, nullptr};
    FlashMlaOptionalParaInfo metadata = {nullptr, nullptr};

    FlashMlaOptionalParaInfo cuSeqlensKv = {nullptr, nullptr};
    FlashMlaOptionalParaInfo sequsedKv = {nullptr, nullptr};

    const int64_t* headDimV = nullptr;
    const float* softmaxScale = nullptr;
    const int64_t* maskMode = nullptr;
    const int64_t* maxSeqlenQ = nullptr;
    const int64_t* maxSeqlenKV = nullptr;
    const char* layoutQ = nullptr;
    const char* layoutKV = nullptr;
    const char* layoutOut = nullptr;
    const int64_t* returnSoftMaxLse = nullptr;

    FlashMlaRequiredParaInfo attnOut = {nullptr, nullptr};
    FlashMlaRequiredParaInfo lseOut = {nullptr, nullptr};
};

class FlashMlaTilingInfo : public TilingInfo {
public:
    const char* opName = nullptr;
    fe::PlatFormInfos* platformInfo = nullptr;
    FlashMlaParaInfo opParamInfo;

    int64_t bSize = 0;
    int64_t n1Size = 0;
    int64_t n2Size = 0;
    int64_t s1Size = 0;
    int64_t s2Size = 0;
    int64_t qkNopeHeadDim = 0;
    int64_t vHeadDim = 0;
    int64_t ropeHeadDim = 0; // rope 段宽（rope 合并进 q/k_cache 末 64 道, 默认 64）
    int64_t gSize = 0;
    int64_t qTSize = 0;
    int64_t kTSize = 0;
    float softmaxScale = 0;

    int64_t cuSeqlensQSize = 0;   // cu_seqlens_q 元素数 = bSize + 1（TND 累计语义, 含头部 0）
    int64_t cacheSeqlensSize = 0; // cache_seqlens 元素数 = bSize（BY_BATCH 每 batch 长度）

    int64_t maskMode = 0;
    bool attenMaskFlag = false;

    uint64_t totalOutputSize = 0;
    uint64_t totalLseSize = 0;

    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    int64_t blockTypeSize = 0;
    int64_t maxBlockNumPerBatch = 0;
    int64_t totalBlockNum = 0;

    int64_t seqUsedQSize = 0;
    int64_t cuSeqLenQDims = 0;
    int64_t maxSeqQ = 0;
    int64_t maxSeqKv = 0;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    int64_t winLeft = 0;
    int64_t winRight = 0;

    bool batchContinuousFlag = true;
    KvStorageMode kvStorageMode = KvStorageMode::BATCH_CONTINUOUS;
    int64_t returnSoftmaxLse = 0;
    bool softmaxLseFlag = false;
    bool emptyTensorFlag = false;

    // Without tensor-view information, opbase guarantees contiguous inputs.
    uint64_t keyBnStride = 0;
    uint64_t keyN2Stride = 0;
    uint64_t valueBnStride = 0;
    uint64_t valueN2Stride = 0;
    bool hasViewStride = true;
    // 非连续场景下首个非连续维, -1表示连续, 由parser计算供checker复用
    int32_t keyNonContigDim = -1;
    int32_t valueNonContigDim = -1;

    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKvType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_FLOAT16;

    FlashMlaLayout qLayout = FlashMlaLayout::BSND;
    FlashMlaLayout outLayout = FlashMlaLayout::BSND;
    FlashMlaLayout kvLayout = FlashMlaLayout::BSND;
    int64_t kernelOutputLayout = 0;
};

namespace arch35MLA {
constexpr uint32_t B_LIMIT = 65536;

constexpr uint32_t BLOCK_SIZE_ALIGN_SIZE_16 = 16;

constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;

constexpr uint32_t MLA_D_DIM_512 = 512;
constexpr uint32_t MLA_ROPE_D_DIM_64 = 64;
constexpr uint32_t KERNEL_LAYOUT_BSH = 0; // BSH/BSND
constexpr uint32_t KERNEL_LAYOUT_BNSD = 1;
constexpr uint32_t KERNEL_LAYOUT_TND = 3;
constexpr uint32_t KERNEL_LAYOUT_NTD = 5;

constexpr uint32_t BYTE_BLOCK = 32;

constexpr uint32_t CV_RATIO = 2;

const std::map<ge::DataType, std::string> DATATYPE_TO_STRING_MAP = {{ge::DT_UNDEFINED, "DT_UNDEFINED"},
                                                                    {ge::DT_FLOAT, "DT_FLOAT"},
                                                                    {ge::DT_FLOAT16, "DT_FLOAT16"},
                                                                    {ge::DT_INT8, "DT_INT8"},
                                                                    {ge::DT_INT16, "DT_INT16"},
                                                                    {ge::DT_UINT16, "DT_UINT16"},
                                                                    {ge::DT_UINT8, "DT_UINT8"},
                                                                    {ge::DT_INT32, "DT_INT32"},
                                                                    {ge::DT_INT64, "DT_INT64"},
                                                                    {ge::DT_UINT32, "DT_UINT32"},
                                                                    {ge::DT_UINT64, "DT_UINT64"},
                                                                    {ge::DT_BOOL, "DT_BOOL"},
                                                                    {ge::DT_DOUBLE, "DT_DOUBLE"},
                                                                    {ge::DT_DUAL, "DT_DUAL"},
                                                                    {ge::DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},
                                                                    {ge::DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},
                                                                    {ge::DT_COMPLEX32, "DT_COMPLEX32"},
                                                                    {ge::DT_COMPLEX64, "DT_COMPLEX64"},
                                                                    {ge::DT_COMPLEX128, "DT_COMPLEX128"},
                                                                    {ge::DT_QINT8, "DT_QINT8"},
                                                                    {ge::DT_QINT16, "DT_QINT16"},
                                                                    {ge::DT_QINT32, "DT_QINT32"},
                                                                    {ge::DT_QUINT8, "DT_QUINT8"},
                                                                    {ge::DT_QUINT16, "DT_QUINT16"},
                                                                    {ge::DT_RESOURCE, "DT_RESOURCE"},
                                                                    {ge::DT_STRING_REF, "DT_STRING_REF"},
                                                                    {ge::DT_STRING, "DT_STRING"},
                                                                    {ge::DT_VARIANT, "DT_VARIANT"},
                                                                    {ge::DT_BF16, "DT_BFLOAT16"},
                                                                    {ge::DT_INT4, "DT_INT4"},
                                                                    {ge::DT_UINT1, "DT_UINT1"},
                                                                    {ge::DT_INT2, "DT_INT2"},
                                                                    {ge::DT_UINT2, "DT_UINT2"},
                                                                    {ge::DT_HIFLOAT8, "DT_HIFLOAT8"},
                                                                    {ge::DT_FLOAT8_E4M3FN, "DT_FLOAT8_E4M3FN"},
                                                                    {ge::DT_FLOAT4_E2M1, "DT_FLOAT4_E2M1"}};

const std::map<std::string, std::vector<ge::DataType>> DTYPE_SUPPORT_MAP = {
    {QUERY_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {K_CACHE_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {ATTN_MASK_NAME, {ge::DT_INT8}},
    {BLOCK_TABLE_NAME, {ge::DT_INT32}},
    {CACHE_SEQLENS_NAME, {ge::DT_INT32}},
    {CU_SEQLENS_Q_NAME, {ge::DT_INT32}},
    {SEQUSED_Q_NAME, {ge::DT_INT32}},
    {METADATA_NAME, {ge::DT_INT32}},
    {ATTN_OUT_NAME, {ge::DT_FLOAT16, ge::DT_BF16}},
    {SOFTMAX_LSE_NAME, {ge::DT_FLOAT}},
};

const std::set<ge::Format> FORMAT_SUPPORT_SET = {ge::FORMAT_ND};
} // namespace arch35MLA

} // namespace flash_mla
} // namespace optiling
#endif // FLASH_MLA_FA_TILING_INFO_H
