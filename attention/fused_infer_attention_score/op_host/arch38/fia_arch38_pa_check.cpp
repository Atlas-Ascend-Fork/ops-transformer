/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fia_arch38_prompt_tiling.h"

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {

namespace arch38 {

bool FiaArch38PromptTiling::GetAndCheckPrefixShape(FiaArch38ContextParams& contextKeyParams,
                                                   const PFAShapeInfo& queryShapeInfo, PFAShapeInfo& prefixShapeInfo,
                                                   FiaArch38FlashAttentionTilingData& tilingData) const
{
    PFAShapeDims dims;
    if (!SetShape(contextKeyParams, contextKeyParams.queryInputShape, "query", dims)) {
        return false;
    }
    int64_t prefixSeqInnerSize = dims.s;
    int64_t bPrefix = dims.b;
    int64_t nPrefix = dims.n;
    int64_t dPrefix = dims.d;
    int64_t hPrefix = dims.h;
    int64_t tPrefix = dims.t;
    OP_CHECK_IF(
        (bPrefix != 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "prefix batch", std::to_string(bPrefix).c_str(),
                                              "prefix batch num only supports 1"),
        return false);
    if (inputLayout == InputLayout::BSH) {
        OP_CHECK_IF((hPrefix != queryShapeInfo.h / tilingData.fiaArch38AttentionBaseParams.get_headNumRatio()),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        contextKeyParams.opName, "key_shared_prefix, key",
                        (std::to_string(hPrefix) + ", " +
                         std::to_string(queryShapeInfo.h / tilingData.fiaArch38AttentionBaseParams.get_headNumRatio()))
                            .c_str(),
                        "prefix H must be the same as KV H"),
                    return false);
    } else {
        OP_CHECK_IF((nPrefix != queryShapeInfo.n / tilingData.fiaArch38AttentionBaseParams.get_headNumRatio()) ||
                        (dPrefix != queryShapeInfo.h / queryShapeInfo.n),
                    OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                        contextKeyParams.opName, "key_shared_prefix, key",
                        (std::to_string(nPrefix) + ", " + std::to_string(dPrefix) + ", " +
                         std::to_string(queryShapeInfo.n / tilingData.fiaArch38AttentionBaseParams.get_headNumRatio()) +
                         ", " + std::to_string(queryShapeInfo.h / queryShapeInfo.n))
                            .c_str(),
                        "prefix N and D must be the same as KV N and D"),
                    return false);
    }
    prefixShapeInfo.b = static_cast<uint32_t>(bPrefix);
    prefixShapeInfo.n = static_cast<uint64_t>(nPrefix);
    prefixShapeInfo.s = static_cast<uint32_t>(prefixSeqInnerSize);
    prefixShapeInfo.d = static_cast<uint32_t>(dPrefix);
    prefixShapeInfo.h = static_cast<uint64_t>(hPrefix);
    return true;
}

bool FiaArch38PromptTiling::CheckKeyValuePrefixConsistency(FiaArch38ContextParams& contextKeyParams,
                                                           const gert::StorageShape* keyShape) const
{
    size_t prefixKeyDim = contextKeyParams.keySharedPrefix->GetStorageShape().GetDimNum();
    size_t prefixValueDim = contextKeyParams.valueSharedPrefix->GetStorageShape().GetDimNum();
    size_t KVDim = keyShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF(
        ((prefixKeyDim != KVDim) || (prefixKeyDim != prefixValueDim)),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            contextKeyParams.opName, "key_shared_prefix, value_shared_prefix, key",
            (std::to_string(prefixKeyDim) + ", " + std::to_string(prefixValueDim) + ", " + std::to_string(KVDim))
                .c_str(),
            "dim num of key_shared_prefix and value_shared_prefix must be the same as KV"),
        return false);
    for (uint32_t i = 0; i < prefixKeyDim; i++) {
        int64_t tmpPrefixKeyDim = contextKeyParams.keySharedPrefix->GetStorageShape().GetDim(i);
        int64_t tmpPrefixValueDim = contextKeyParams.valueSharedPrefix->GetStorageShape().GetDim(i);
        if ((tmpPrefixKeyDim == 0) || (tmpPrefixValueDim == 0)) {
            OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                contextKeyParams.opName, "key_shared_prefix, value_shared_prefix",
                (std::to_string(tmpPrefixKeyDim) + ", " + std::to_string(tmpPrefixValueDim)).c_str(),
                "key_shared_prefix and value_shared_prefix do not support empty tensor");
            return false;
        }
        if (tmpPrefixKeyDim != tmpPrefixValueDim) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                contextKeyParams.opName, "key_shared_prefix, value_shared_prefix",
                (std::to_string(tmpPrefixKeyDim) + ", " + std::to_string(tmpPrefixValueDim)).c_str(),
                "shape of key_shared_prefix must be the same as value_shared_prefix");
            return false;
        }
    }
    return true;
}

bool FiaArch38PromptTiling::CheckActSharedPrefix(FiaArch38ContextParams& contextKeyParams, const uint32_t sPrefix,
                                                 const uint32_t sKV)
{
    size_t prefixDimNum = contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetDimNum();
    OP_CHECK_IF((prefixDimNum != 1),
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "actual_shared_prefix_len",
                                                         std::to_string(prefixDimNum).c_str(),
                                                         "actualSharedPrefixLen dim num must be 1"),
                return false);
    int64_t prefixShapeSize = contextKeyParams.actualSharedPrefixLen->GetStorageShape().GetShapeSize();
    OP_CHECK_IF((prefixShapeSize != 1),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(contextKeyParams.opName, "actual_shared_prefix_len",
                                                          std::to_string(prefixShapeSize).c_str(),
                                                          "actualSharedPrefixLen length must be 1"),
                return false);
    OP_CHECK_IF((contextKeyParams.actualSharedPrefixLen->GetData<int64_t>() == nullptr),
                OP_LOGE_WITHOUT_REPORT(contextKeyParams.opName, "input actualSharedPrefixLen GetData is nullptr!"),
                return false);
    actualSharedPrefixLen = contextKeyParams.actualSharedPrefixLen->GetData<int64_t>()[0];
    OP_CHECK_IF((actualSharedPrefixLen > sPrefix) || (actualSharedPrefixLen < 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "actual_shared_prefix_len",
                                                      std::to_string(actualSharedPrefixLen).c_str(),
                                                      "actualSharedPrefixLen must be in range [0, sPrefix]"),
                return false);

    if (actualSharedPrefixLen + sKV > SLIMIT) {
        OP_LOGW(contextKeyParams.opName, "Seq size of actualSharedPrefixLen(%ld) + kv_s(%u) should <= 20m",
                actualSharedPrefixLen, sKV);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPAKeyValueDimConsistency(const FiaArch38ContextParams& contextKeyParams,
                                                          const PACacheDimCtx& ctx) const
{
    int64_t keyDim2 = ctx.keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_1);
    int64_t keyDim3 = ctx.keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_2);
    int64_t keyDim4 = 0;
    int64_t valueDim1 = ctx.valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_0);
    int64_t valueDim2 = ctx.valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_1);
    int64_t valueDim3 = ctx.valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_2);
    int64_t valueDim4 = 0;
    if (ctx.keyDim == KV_CACHE_DIM_NUMS_4) {
        keyDim4 = ctx.keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_3);
        valueDim4 = ctx.valueShape->GetStorageShape().GetDim(KV_CACHE_DIM_3);
    }
    OP_CHECK_IF(
        (ctx.keyDim == KV_CACHE_DIM_NUMS_4) && ((ctx.keyDim1 != valueDim1) || (keyDim2 != valueDim2) ||
                                                (keyDim3 != valueDim3) || ((keyDim4 != valueDim4) && (!enablePFAMLA))),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            contextKeyParams.opName, "key, value",
            (std::to_string(ctx.keyDim1) + ", " + std::to_string(keyDim2) + ", " + std::to_string(keyDim3) + ", " +
             std::to_string(keyDim4) + ", " + std::to_string(valueDim1) + ", " + std::to_string(valueDim2) + ", " +
             std::to_string(valueDim3) + ", " + std::to_string(valueDim4))
                .c_str(),
            "the dim num of key and value are inconsistent when PA enable"),
        return false);
    OP_CHECK_IF(
        (ctx.keyDim == KV_CACHE_DIM_NUMS_3) &&
            ((ctx.keyDim1 != valueDim1) || (keyDim2 != valueDim2) || ((keyDim3 != valueDim3) && (!enablePFAMLA))),
        OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
            contextKeyParams.opName, "key, value",
            (std::to_string(ctx.keyDim1) + ", " + std::to_string(keyDim2) + ", " + std::to_string(keyDim3) + ", " +
             std::to_string(valueDim1) + ", " + std::to_string(valueDim2) + ", " + std::to_string(valueDim3))
                .c_str(),
            "the dim num of key and value are inconsistent when PA enable"),
        return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPAKeyValueLayout(const FiaArch38ContextParams& contextKeyParams,
                                                  const PFAShapeInfo& queryShapeInfo, const PACacheDimCtx& ctx) const
{
    int64_t keyDim2 = ctx.keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_1);
    int64_t keyDim3 = ctx.keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_2);
    int64_t keyDim4 = (ctx.keyDim == KV_CACHE_DIM_NUMS_4) ? ctx.keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_3) : 0;
    if (inputLayout == InputLayout::BNSD || inputLayout == InputLayout::TND) {
        OP_CHECK_IF(
            (ctx.keyDim != KV_CACHE_DIM_NUMS_3) && (ctx.keyDim != KV_CACHE_DIM_NUMS_4),
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(contextKeyParams.opName, "key, value",
                                                   (std::to_string(ctx.keyDim1) + ", " + std::to_string(keyDim2) +
                                                    ", " + std::to_string(keyDim3) + ", " + std::to_string(keyDim4))
                                                       .c_str(),
                                                   "the layout of key and value is invalid when PA enable"),
            return false);
    } else if (inputLayout == InputLayout::BSH || inputLayout == InputLayout::BSND) {
        OP_CHECK_IF(
            ctx.keyDim != 3,
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(contextKeyParams.opName, "key, value",
                                                   (std::to_string(ctx.keyDim1) + ", " + std::to_string(keyDim2) +
                                                    ", " + std::to_string(keyDim3) + ", " + std::to_string(keyDim4))
                                                       .c_str(),
                                                   "the layout of key and value is invalid when PA enable"),
            return false);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPACacheShape3D(const FiaArch38ContextParams& contextKeyParams,
                                                const PFAShapeInfo& shapeInfo, const gert::StorageShape* shape,
                                                const PACacheDimCtx& ctx, const std::string& sName)
{
    int64_t dim1 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_0);
    int64_t dim2 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_1);
    int64_t dim3 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_2);
    int64_t tempBlockSize = dim2;
    int64_t tempH = dim3;
    paLayoutType = 1; // If it is three-dimensional, paLayoutType = 1
    OP_CHECK_IF(
        ((dim1 < ctx.blockNumValid) || (tempBlockSize != *ctx.blockSize) || (tempH * ctx.headNumRatio != shapeInfo.h)),
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            contextKeyParams.opName, sName.c_str(),
            ("[" + std::to_string(dim1) + ", " + std::to_string(dim2) + ", " + std::to_string(dim3) + "]").c_str(),
            "the dim must be [>=blockNumValid, blockSize, h/headNumRatio] when PA enable"),
        return false);
    // In the BSH input of the PA scenario, the h of the KV matrix must not exceed 65535.
    OP_CHECK_IF(dim3 > HLIMIT,
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    contextKeyParams.opName, sName.c_str(), std::to_string(dim3).c_str(),
                    "layout of key/value is BSH, h must not be greater than 65535 when PA enable"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPACacheShape4D(const FiaArch38ContextParams& contextKeyParams,
                                                const PFAShapeInfo& shapeInfo, const gert::StorageShape* shape,
                                                const PACacheDimCtx& ctx, const std::string& sName)
{
    int64_t dim1 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_0);
    int64_t dim2 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_1);
    int64_t dim3 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_2);
    int64_t dim4 = shape->GetStorageShape().GetDim(KV_CACHE_DIM_3);
    int64_t tempN = dim2;
    int64_t tempBlockSize = dim3;
    int64_t tempD = dim4;
    paLayoutType = 0; // If it is four-dimensional, paLayoutType = 0
    OP_CHECK_IF(((dim1 < ctx.blockNumValid) || (tempN * ctx.headNumRatio != shapeInfo.n) ||
                 (tempBlockSize != *ctx.blockSize) || (tempD != (shapeInfo.h / shapeInfo.n))),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    contextKeyParams.opName, sName.c_str(),
                    ("[" + std::to_string(dim1) + ", " + std::to_string(dim2) + ", " + std::to_string(dim3) + ", " +
                     std::to_string(dim4) + "]")
                        .c_str(),
                    "the dim must be [>=blockNumValid, n/headNumRatio, blockSize, h/n] when PA enable"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPACacheShape(const FiaArch38ContextParams& contextKeyParams,
                                              const PFAShapeInfo& shapeInfo, const gert::StorageShape* shape,
                                              const PACacheDimCtx& ctx, const std::string& sName)
{
    return (ctx.keyDim == KV_CACHE_DIM_NUMS_3) ? CheckPACacheShape3D(contextKeyParams, shapeInfo, shape, ctx, sName) :
                                                 CheckPACacheShape4D(contextKeyParams, shapeInfo, shape, ctx, sName);
}

PABlockNumInfo FiaArch38PromptTiling::ComputePABlockNumInfo(const FiaArch38ContextParams& contextKeyParams,
                                                            const PFAShapeInfo& queryShapeInfo) const
{
    const gert::Tensor* actSeqLenKV = contextKeyParams.actualSequenceLengthKV;
    const int64_t* blockSize = contextKeyParams.blockSize;
    PABlockNumInfo info;
    for (uint32_t i = 0; i < queryShapeInfo.b; i++) {
        int32_t actualSeqKVPerBatch = (actSeqLenKV->GetShapeSize() > 1) ?
                                          static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[i]) :
                                          static_cast<int32_t>(actSeqLenKV->GetData<int64_t>()[0]);
        int32_t blockNumPerBatch = static_cast<int32_t>((actualSeqKVPerBatch + *blockSize - 1) / *blockSize);
        info.blockNumValid += blockNumPerBatch;
        if (blockNumPerBatch > info.maxBlockNumPerBatch) {
            info.maxBlockNumPerBatch = blockNumPerBatch;
        }
    }
    return info;
}

bool FiaArch38PromptTiling::CheckBlockTableDims(const FiaArch38ContextParams& contextKeyParams,
                                                const PFAShapeInfo& queryShapeInfo, int32_t maxBlockNumPerBatch) const
{
    const gert::StorageShape* blockTableShape = contextKeyParams.blockTableShape;
    size_t blockTableDim = blockTableShape->GetStorageShape().GetDimNum();
    int64_t blockTableDim1 = blockTableShape->GetStorageShape().GetDim(0);
    string blockTableStr = "";
    for (size_t i = 0; i < blockTableDim; i++) {
        blockTableStr += std::to_string(blockTableShape->GetStorageShape().GetDim(i));
        blockTableStr += ",";
    }
    blockTableStr.pop_back();

    OP_CHECK_IF(blockTableDim != 2, // PA BlockTable dim size
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(contextKeyParams.opName, "block_table",
                                                         std::to_string(blockTableDim).c_str(),
                                                         "the dim of block table must be 2 when PA enable"),
                return false);

    OP_CHECK_IF(((blockTableDim1 != queryShapeInfo.b) || (blockTableDim2 < maxBlockNumPerBatch)),
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    contextKeyParams.opName, "block_table",
                    ("[" + std::to_string(blockTableDim1) + ", " + std::to_string(blockTableDim2) + "]").c_str(),
                    "block table shape should be [b, >=maxBlockNumPerBatch] when PA enable"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckBlockTableShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                                 PFAShapeInfo& queryRopeShapeInfo,
                                                 FiaArch38FlashAttentionTilingData& tilingData)
{
    PABlockNumInfo blockNumInfo = ComputePABlockNumInfo(contextKeyParams, queryShapeInfo);

    const gert::StorageShape* keyShape = contextKeyParams.keyInputShape;
    const gert::StorageShape* valueShape = contextKeyParams.valueInputShape;
    const size_t keyDim = keyShape->GetStorageShape().GetDimNum();
    PACacheDimCtx ctx;
    ctx.keyShape = keyShape;
    ctx.valueShape = valueShape;
    ctx.keyDim = keyDim;
    ctx.keyDim1 = keyShape->GetStorageShape().GetDim(KV_CACHE_DIM_0);
    ctx.blockSize = contextKeyParams.blockSize;
    ctx.blockNumValid = blockNumInfo.blockNumValid;
    ctx.headNumRatio = (enableIFAMLA || enableIFA) ?
                           static_cast<int64_t>(gSize) :
                           static_cast<int64_t>(tilingData.fiaArch38AttentionBaseParams.get_headNumRatio());

    if (!CheckPAKeyValueDimConsistency(contextKeyParams, ctx) ||
        !CheckPAKeyValueLayout(contextKeyParams, queryShapeInfo, ctx)) {
        return false;
    }
    if (!CheckPACacheShape(contextKeyParams, queryShapeInfo, keyShape, ctx, "key")) {
        return false;
    }
    if (!CheckPARopeShape(contextKeyParams, queryRopeShapeInfo, ctx)) {
        return false;
    }

    if (!CheckBlockTableDims(contextKeyParams, queryShapeInfo, blockNumInfo.maxBlockNumPerBatch)) {
        return false;
    }

    paBlockNumSum = ctx.keyDim1;
    S2 = static_cast<uint32_t>(blockNumInfo.maxBlockNumPerBatch * (*contextKeyParams.blockSize));
    if (S2 > SLIMIT) {
        OP_LOGW(contextKeyParams.opName, "seq should <= 20m, but kv seq = %u when PA enable", S2);
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPARopeShape(const FiaArch38ContextParams& contextKeyParams,
                                             PFAShapeInfo& queryRopeShapeInfo, const PACacheDimCtx& ctx)
{
    if (!enableIFAMLA) {
        return true;
    }
    const gert::StorageShape* keyRopeShape = contextKeyParams.keyRopeInputShape;
    const size_t keyRopeDim = keyRopeShape->GetStorageShape().GetDimNum();
    OP_CHECK_IF((keyRopeDim != ctx.keyDim),
                OP_LOGE_FOR_INVALID_SHAPEDIMS_WITH_REASON(
                    contextKeyParams.opName, "key_rope, key",
                    (std::to_string(keyRopeDim) + ", " + std::to_string(ctx.keyDim)).c_str(),
                    "the dim of keyRope must be equal to the dim of key when PA enable"),
                return false);
    if (!CheckPACacheShape(contextKeyParams, queryRopeShapeInfo, keyRopeShape, ctx, "keyRope")) {
        return false;
    }
    return true;
}

bool FiaArch38PromptTiling::CheckPATypeAndShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                                PFAShapeInfo& queryRopeShapeInfo,
                                                FiaArch38FlashAttentionTilingData& tilingData)
{
    // The interception that is mutually exclusive with the left padding has been implemented in FIA.
    OP_CHECK_IF(enableTensorList,
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "block_table, tensorlist",
                                                       "block_table present with tensorlist",
                                                       "tensorlist is not supported when blockTable is not null"),
                return false);
    OP_CHECK_IF((!enableActSeqLenKV) && (contextKeyParams.fromTilingSink == 0),
                OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "actual_seq_lengths_kv"), return false);

    if (!CheckPAType(contextKeyParams)) {
        return false;
    }
    if (!CheckPABlockSize(contextKeyParams, queryShapeInfo)) {
        return false;
    }

    if (contextKeyParams.fromTilingSink != 0) {
        S2 = static_cast<uint32_t>(blockTableDim2 * (*contextKeyParams.blockSize));
        return true;
    }
    if (!CheckBlockTableShape(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, tilingData)) {
        return false;
    }

    tilingData.fiaArch38AttentionBaseParams.set_PAlayoutType(paLayoutType);
    return true;
}

bool FiaArch38PromptTiling::CheckPAType(const FiaArch38ContextParams& contextKeyParams)
{
    const gert::StorageShape* blockTableShape = contextKeyParams.blockTableShape;
    OP_CHECK_IF(blockTableShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(contextKeyParams.opName, "block_table"),
                return false);

    OP_CHECK_IF(
        blockTableShape->GetStorageShape().GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(contextKeyParams.opName, "block_table",
                                                   (std::to_string(blockTableShape->GetStorageShape().GetDim(0)) +
                                                    ", " + std::to_string(blockTableShape->GetStorageShape().GetDim(1)))
                                                       .c_str(),
                                                   "blockTable each dim must not be 0"),
        return false);

    ge::DataType blockTableType = contextKeyParams.blockTableType;
    OP_CHECK_IF((blockTableType != ge::DT_INT32),
                OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(contextKeyParams.opName, "block_table",
                                                      GetPfaDataTypeStr(blockTableType).c_str(),
                                                      "blockTable only supports int32 when PA enable"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckPABlockSize(const FiaArch38ContextParams& contextKeyParams,
                                             const PFAShapeInfo& queryShapeInfo)
{
    const int64_t* blockSize = contextKeyParams.blockSize;
    blockTableDim2 = static_cast<int32_t>(contextKeyParams.blockTableShape->GetStorageShape().GetDim(1));
    OP_CHECK_IF((!enableIFAMLA && !enableIFA && !(queryShapeInfo.s == 1 && enableAlibiPse) &&
                 (*blockSize % BLOCK_SIZE_BASE != 0 || *blockSize < BLOCK_SIZE_BASE || *blockSize > BLOCK_SIZE_MAX)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    contextKeyParams.opName, "block_size", std::to_string(*blockSize).c_str(),
                    "block size must be a multiple of BLOCK_SIZE_BASE and within the allowed range"),
                return false);
    ifaBlockSizeBase /= static_cast<int32_t>(dataTypeSize);
    OP_CHECK_IF(((enableIFAMLA || enableIFA || (queryShapeInfo.s == 1 && enableAlibiPse)) &&
                 (*blockSize % ifaBlockSizeBase != 0 || *blockSize < ifaBlockSizeBase || *blockSize > BLOCK_SIZE_MAX)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    contextKeyParams.opName, "block_size", std::to_string(*blockSize).c_str(),
                    "block size must be a multiple of ifaBlockSizeBase and within the allowed range"),
                return false);
    return true;
}

} // namespace arch38
} // namespace optiling
