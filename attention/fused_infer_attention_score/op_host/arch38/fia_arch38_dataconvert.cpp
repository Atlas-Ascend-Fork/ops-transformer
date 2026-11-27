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

void FiaArch38PromptTiling::SetMultiCoreParamsRegbase(int64_t totalSize, int64_t actualUsedCoreNum)
{
    faTilingAdapter.multiCoreParamsRegbase.set_coreNum(static_cast<int32_t>(actualUsedCoreNum));
    faTilingAdapter.multiCoreParamsRegbase.set_totalSize(totalSize);
    faTilingAdapter.multiCoreParamsRegbase.set_splitFactorSize(CeilDivision(totalSize, actualUsedCoreNum));
    faTilingAdapter.multiCoreParamsRegbase.set_splitFactorTailSize(
        CalcTailSize(totalSize, faTilingAdapter.multiCoreParamsRegbase.get_splitFactorSize()));
}

bool FiaArch38PromptTiling::CheckArch38PlatformSupported(const FiaArch38ContextParams& contextKeyParams,
                                                         const PFAShapeInfo& queryShapeInfo,
                                                         const PFAShapeInfo& valueShapeInfo) const
{
    // 以下约束只与平台已注册的模板/tilingKey 有关，不依赖 IFA/合轴改写后的 n/s，
    // 因此 IFA / MLA / 合轴路径也必须满足，不能放在 early-return 之后。
    OP_CHECK_IF((inputLayout != InputLayout::BNSD),
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(contextKeyParams.opName, "input_layout", contextKeyParams.layout,
                                                       "only BNSD layout is supported on this platform"),
                return false);

    OP_CHECK_IF((!IsArch38SupportedHeadSize(static_cast<int64_t>(queryShapeInfo.d)) ||
                 !IsArch38SupportedHeadSize(static_cast<int64_t>(valueShapeInfo.d))),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, "query, value",
                    (std::to_string(queryShapeInfo.d) + ", " + std::to_string(valueShapeInfo.d)).c_str(),
                    "head size only supports 64, 80, 128 or 256 on this platform"),
                return false);

    // 本平台注册的模板中 query d 恒等于 value d。
    OP_CHECK_IF((queryShapeInfo.d != valueShapeInfo.d),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "query.d, value.d",
                    (std::to_string(queryShapeInfo.d) + ", " + std::to_string(valueShapeInfo.d)).c_str(),
                    "query d and value d must be equal on this platform."),
                return false);

    // 仅放行已注册 tiling key 的 dtype 组合，避免分发到不存在的 kernel。
    OP_CHECK_IF(
        !IsArch38SupportedDataType(contextKeyParams.inputDataType, contextKeyParams.kDataType,
                                   contextKeyParams.outputDataType, static_cast<int64_t>(queryShapeInfo.d)),
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            contextKeyParams.opName, "query, key, attentionOut",
            (GetPfaDataTypeStr(contextKeyParams.inputDataType) + ", " + GetPfaDataTypeStr(contextKeyParams.kDataType) +
             ", " + GetPfaDataTypeStr(contextKeyParams.outputDataType))
                .c_str(),
            "only fp16/int8 query/key/value with fp16/int8 output is supported on this platform, "
            "and head size 256 only supports fp16 query/key/value with fp16 output."),
        return false);

    // head size 256 仅有带 mask 场景的 tiling key。
    OP_CHECK_IF(((static_cast<int64_t>(queryShapeInfo.d) == ARCH38_SUPPORTED_D_256) && (!enableMask)),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "atten_mask", "absent",
                                                      "head size 256 requires atten_mask on this platform."),
                return false);

    return true;
}

bool FiaArch38PromptTiling::CheckArch38ScenarioSupported(const FiaArch38ContextParams& contextKeyParams,
                                                         const PFAShapeInfo& queryShapeInfo,
                                                         const PFAShapeInfo& valueShapeInfo) const
{
    if (!CheckArch38UnsupportedFeatures(contextKeyParams)) {
        return false;
    }

    // 平台级约束（layout / head size / dQ == dV / dtype / D256 mask）对所有路径生效。
    if (!CheckArch38PlatformSupported(contextKeyParams, queryShapeInfo, valueShapeInfo)) {
        return false;
    }

    // 带 mask 的 band（sparseMode=4）会生成 +2e8 的 tilingKey，本平台没有注册；
    // 不带 mask 的 band 仍走普通的 no-mask key，IFA / 合轴可继续使用。
    OP_CHECK_IF(
        (enableMask && (sparseModeVal == SPARSE_MODE_BAND)),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "sparse_mode, atten_mask", "band mask present",
                                               "band mask is not supported on this platform"),
        return false);

    // IFA / 合轴在调用点前已按 gSize 改写 queryShapeInfo.n/.s，下面 qs == kvs、qs > 1 等
    // PFA 原始形状约束不适用，因此这里只跳过形状相关检查（平台级约束已在上方执行）。
    if (enableIFA || enablePFAMerge) {
        return true;
    }

    if (!CheckArch38ShapeConstraints(contextKeyParams, queryShapeInfo)) {
        return false;
    }

    if (!enableMask) {
        return true;
    }
    return CheckArch38MaskConstraints(contextKeyParams);
}

bool FiaArch38PromptTiling::CheckArch38UnsupportedFeatures(const FiaArch38ContextParams& contextKeyParams) const
{
    // 本平台没有 queryRope/keyRope 对应的模板：所有 tiling key 都以 hasRope == false 实例化，
    // 若放行 PFARope 场景会静默忽略 rope 输入。
    OP_CHECK_IF(enablePFARope,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "queryRope/keyRope", "not empty",
                                                      "queryRope/keyRope is not supported on this platform."),
                return false);

    // 本平台同样没有 MLA 对应的模板：entry 中不存在 +3e7（MLA）key，也没有 d=192/576 的组合，
    // 放行只会生成未注册的 key 并分发失败，此处显式拒绝。
    OP_CHECK_IF((enableIFAMLA || enablePFAMLA),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "queryRope/keyRope or query.d, value.d",
                                                      "MLA", "MLA is not supported on this platform."),
                return false);

    // 本平台没有 PSE 对应的模板：entry 中所有 key 的 PseType 恒为 PSE_NONE_TYPE，不存在 +1e9（pse）key。
    OP_CHECK_IF(enablePseShift,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "pse_shift", "not empty",
                                                      "pse_shift is not supported on this platform."),
                return false);

    // 本平台没有 alibi PSE 对应的模板：entry 中不存在 +2e9/+3e9（alibi）key。
    OP_CHECK_IF(enableAlibiPse,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "pse_type", "2 or 3",
                                                      "alibi pse (pse_type 2/3) is not supported on this platform."),
                return false);

    // 本平台没有 PA 对应的模板：entry 中不存在 +2e7（PA）key。
    OP_CHECK_IF(enablePA,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "block_table", "not null",
                                                      "PagedAttention is not supported on this platform."),
                return false);

    // 本平台没有 system prefix 对应的模板：entry 中不存在 +1e8（prefix）key，
    // 且该 +1e8 位与 mask 位冲突，放行会导致 key 歧义。
    OP_CHECK_IF(isKVHasPrefix,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "key_shared_prefix, value_shared_prefix",
                                                      "not null", "system prefix is not supported on this platform."),
                return false);

    // 本平台 kernel entry 没有 learnable_sink 参数（InitBaseAPI 将其置 nullptr），
    // 放行会静默忽略 sink 输入，此处显式拒绝。
    OP_CHECK_IF(contextKeyParams.hasLearnableSink,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "learnable_sink", "not null",
                                                      "learnable_sink is not supported on this platform."),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckArch38ShapeConstraints(const FiaArch38ContextParams& contextKeyParams,
                                                        const PFAShapeInfo& queryShapeInfo) const
{
    OP_CHECK_IF((queryShapeInfo.b != 1U && queryShapeInfo.b != 9U),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "query batch",
                                                      std::to_string(queryShapeInfo.b).c_str(),
                                                      "batch size must be 1 or 9 on this platform"),
                return false);

    OP_CHECK_IF((queryShapeInfo.s != static_cast<uint64_t>(S2)),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, "query, key/value",
                    (std::to_string(queryShapeInfo.s) + ", " + std::to_string(S2)).c_str(),
                    "query seq length must equal kv seq length on this platform"),
                return false);

    OP_CHECK_IF(((queryShapeInfo.s <= 1UL) || (S2 <= 1U)),
                OP_LOGE_FOR_INVALID_SHAPESIZES_WITH_REASON(
                    contextKeyParams.opName, "query, key/value",
                    (std::to_string(queryShapeInfo.s) + ", " + std::to_string(S2)).c_str(),
                    "query and kv seq length must be greater than 1 on this platform"),
                return false);

    OP_CHECK_IF((queryShapeInfo.s > static_cast<uint64_t>(ARCH38_SUPPORTED_MAX_SEQ)),
                OP_LOGE_FOR_INVALID_SHAPESIZE_WITH_REASON(
                    contextKeyParams.opName, "query", std::to_string(queryShapeInfo.s).c_str(),
                    "seq length must be less than or equal to ARCH38_SUPPORTED_MAX_SEQ on this platform"),
                return false);

    return CheckArch38HeadAndSparse(contextKeyParams);
}

bool FiaArch38PromptTiling::CheckArch38HeadAndSparse(const FiaArch38ContextParams& contextKeyParams) const
{
    const int64_t nQ = *contextKeyParams.headsNumber;
    const int64_t nKV = (*contextKeyParams.numKeyValueHeads > 0) ? *contextKeyParams.numKeyValueHeads : nQ;
    OP_CHECK_IF(
        ((nQ <= 0) || (nQ > ARCH38_SUPPORTED_MAX_N) || (nKV <= 0) || (nKV > ARCH38_SUPPORTED_MAX_N)),
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "num_heads, num_key_value_heads",
                                               (std::to_string(nQ) + ", " + std::to_string(nKV)).c_str(),
                                               "head num must be in (0, ARCH38_SUPPORTED_MAX_N] on this platform"),
        return false);

    // Guarantees the kernel never takes the InitOutput branch.
    OP_CHECK_IF((needInit != 0U),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "actual_seq_lengths, pre_tokens, leftPadding", "needInit = 1",
                    "output must be fully covered by the main loop, check actual_seq_lengths / preTokens / "
                    "left padding configuration"),
                return false);

    OP_CHECK_IF((sparseModeVal != SPARSE_MODE_NO_MASK),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "sparse_mode",
                                                      std::to_string(sparseModeVal).c_str(),
                                                      "attention mask only supports sparseMode = 0 on this platform"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::CheckArch38MaskConstraints(const FiaArch38ContextParams& contextKeyParams) const
{
    // With sparseMode = 0, nextTokensPerbatch always equals this attribute, no per-batch conversion needed.
    OP_CHECK_IF((sparseNextTokens != 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "next_tokens",
                                                      std::to_string(sparseNextTokens).c_str(),
                                                      "attention mask requires nextTokens = 0"),
                return false);

    OP_CHECK_IF((sparsePreTokens <= 0),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "pre_tokens",
                                                      std::to_string(sparsePreTokens).c_str(),
                                                      "attention mask requires preTokens > 0"),
                return false);
    return true;
}

bool FiaArch38PromptTiling::IsFlashDecode() const
{
    return false;
}

ge::graphStatus FiaArch38PromptTiling::SplitBNS(FiaArch38FlashAttentionTilingData& tilingData, uint64_t bng)
{
    uint64_t batchSize = tilingData.fiaArch38AttentionBaseParams.get_batchSize();
    uint64_t headNumSize = tilingData.fiaArch38AttentionBaseParams.get_headNumSize() * gSize;
    uint64_t headDimAlign = AlignUp(tilingData.fiaArch38AttentionBaseParams.get_vHeadSize(), BYTE_BLOCK);
    uint32_t kvSplitLimit = 256U; // 256: 经验值
    int64_t kvSplitPart = 0;
    if (bng != 0) {
        kvSplitPart = aicNum / bng;
    }
    if (kvSplitPart < 1) {
        kvSplitPart = 1; // kvSplitPart 至少为 1，避免后续按份切分时除零
    }
    while (((maxActualseqKV / kvSplitPart) < kvSplitLimit) && (kvSplitPart > 1)) {
        kvSplitPart--;
    }

    faTilingAdapter.inputParamsRegbase.set_kvSplitPart(kvSplitPart);
    faTilingAdapter.inputParamsRegbase.set_accumOutSize(batchSize * headNumSize * kvSplitPart * headDimAlign);
    faTilingAdapter.inputParamsRegbase.set_logSumExpSize(batchSize * headNumSize * kvSplitPart *
                                                         (BYTE_BLOCK / sizeof(float)));

    return ge::GRAPH_SUCCESS;
}

void FiaArch38PromptTiling::SetAttenMaskCompressMode()
{
    static std::map<uint32_t, uint8_t> sparseToCompressModeMap = {
        {SPARSE_MODE_NO_MASK, PfaAttenMaskCompressMode::PFA_NO_COMPRESS_MODE},
        {SPARSE_MODE_ALL_MASK, PfaAttenMaskCompressMode::PFA_NO_COMPRESS_MODE},
        {SPARSE_MODE_LEFT_UP, PfaAttenMaskCompressMode::PFA_LEFT_UP_CAUSAL_MODE},
        {SPARSE_MODE_RIGHT_DOWN, PfaAttenMaskCompressMode::PFA_RIGHT_DOWN_CAUSAL_MODE},
        {SPARSE_MODE_BAND, PfaAttenMaskCompressMode::PFA_BAND_MODE}};
    auto itr = sparseToCompressModeMap.find(sparseModeVal);
    if (itr == sparseToCompressModeMap.end()) {
        faTilingAdapter.inputParamsRegbase.set_attenMaskCompressMode(0);
    } else {
        faTilingAdapter.inputParamsRegbase.set_attenMaskCompressMode(itr->second);
    }
}

void FiaArch38PromptTiling::SetLayoutType()
{
    static std::map<InputLayout, LayoutType> layoutStrToLayoutTypeMap = {
        {InputLayout::BSH, LayoutType::LAYOUT_BSH},
        {InputLayout::TND, LayoutType::LAYOUT_TND},
        {InputLayout::BSND, LayoutType::LAYOUT_BSND},
        {InputLayout::BNSD, LayoutType::LAYOUT_BNSD},
    };
    auto itr = layoutStrToLayoutTypeMap.find(inputLayout);
    if (itr == layoutStrToLayoutTypeMap.end()) {
        faTilingAdapter.inputParamsRegbase.set_layoutType(static_cast<uint8_t>(0));
    } else {
        faTilingAdapter.inputParamsRegbase.set_layoutType(static_cast<uint8_t>(itr->second));
    }
}

ge::graphStatus FiaArch38PromptTiling::SetQKVStartIdx(FiaArch38ContextParams& contextKeyParams)
{
    auto& inputParams = faTilingAdapter.inputParamsRegbase;
    inputParams.set_qStartIdx(0);
    inputParams.set_kvStartIdx(0);
    if (!enableAlibiPse) {
        return ge::GRAPH_SUCCESS;
    }
    int64_t qStartIdx = 0;
    int64_t kvStartIdx = 0;
    auto qStartIdxTensor = contextKeyParams.qStartIdx;
    if (qStartIdxTensor != nullptr) {
        if (qStartIdxTensor->GetShapeSize() >= 1) {
            const int64_t* value = qStartIdxTensor->GetData<int64_t>();
            if (value != nullptr) {
                qStartIdx = value[0];
                OP_CHECK_IF(qStartIdx > INT32_MAX || qStartIdx < INT32_MIN,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "q_start_idx",
                                                                  std::to_string(qStartIdx).c_str(),
                                                                  "qStartIdx must be in range [INT32_MIN, INT32_MAX]"),
                            return ge::GRAPH_FAILED);
                inputParams.set_qStartIdx(qStartIdx);
            }
        }
    }

    auto kvStartIdxTensor = contextKeyParams.kvStartIdx;
    if (kvStartIdxTensor != nullptr) {
        if (kvStartIdxTensor->GetShapeSize() >= 1) {
            const int64_t* kvValue = kvStartIdxTensor->GetData<int64_t>();
            if (kvValue != nullptr) {
                kvStartIdx = kvValue[0];
                OP_CHECK_IF(kvStartIdx > INT32_MAX || kvStartIdx < INT32_MIN,
                            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(contextKeyParams.opName, "kv_start_idx",
                                                                  std::to_string(kvStartIdx).c_str(),
                                                                  "kvStartIdx must be in range [INT32_MIN, INT32_MAX]"),
                            return ge::GRAPH_FAILED);
                inputParams.set_kvStartIdx(kvStartIdx);
            }
        }
    }
    // 当kvStartIdx - qStartIdx超出范围后，由于编译器不支持大数值类型转换，kernel侧int_64转float类型时可能发生截断。
    OP_CHECK_IF(
        kvStartIdx - qStartIdx > INT32_MAX || kvStartIdx - qStartIdx < INT32_MIN,
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(contextKeyParams.opName, "q_start_idx, kv_start_idx",
                                               (std::to_string(qStartIdx) + ", " + std::to_string(kvStartIdx)).c_str(),
                                               "kvStartIdx - qStartIdx must be in range [INT32_MIN, INT32_MAX]"),
        return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void FiaArch38PromptTiling::PFATilingDataconvert(FiaArch38FlashAttentionTilingData& tilingData)
{
    SetLayoutType();
    SetInputParamsShape(tilingData);
    SetInputParamsPseAndMask(tilingData);
    SetInputParamsPA(tilingData);
    SetInitOutputParams(tilingData);
}

void FiaArch38PromptTiling::SetInputParamsShape(FiaArch38FlashAttentionTilingData& tilingData)
{
    auto& inputParams = faTilingAdapter.inputParamsRegbase;
    inputParams.set_bSize(tilingData.fiaArch38AttentionBaseParams.get_batchSize());
    // 将GS1合轴与不合轴场景下，有不同含义的n2Size、gSize与s1Size参数，转化为各自实际的值
    if (enableIFAMLA || enableIFA || enablePFAMerge) {
        inputParams.set_n2Size(tilingData.fiaArch38AttentionBaseParams.get_headNumSize());
        inputParams.set_gSize(tilingData.fiaArch38AttentionBaseParams.get_gOfMla());
        inputParams.set_s1Size(tilingData.fiaArch38AttentionBaseParams.get_seqSize() /
                               tilingData.fiaArch38AttentionBaseParams.get_gOfMla());
    } else {
        inputParams.set_n2Size(tilingData.fiaArch38AttentionBaseParams.get_headNumSize() /
                               tilingData.fiaArch38AttentionBaseParams.get_headNumRatio());
        inputParams.set_gSize(tilingData.fiaArch38AttentionBaseParams.get_headNumRatio());
        inputParams.set_s1Size(tilingData.fiaArch38AttentionBaseParams.get_seqSize());
    }
    inputParams.set_s2Size(tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize());
    inputParams.set_alignedS2(0); // 默认值
    inputParams.set_dSize(tilingData.fiaArch38AttentionBaseParams.get_headSize());
    inputParams.set_dSizeV(tilingData.fiaArch38AttentionBaseParams.get_vHeadSize());
    inputParams.set_dSizeRope(ROPE_DIMENSION_SIZE_TILING_V2);
    inputParams.set_scaleValue(tilingData.fiaArch38AttentionBaseParams.get_scaleValue());
    inputParams.set_preTokens(tilingData.fiaArch38AttentionBaseParams.get_preTokens());
    inputParams.set_nextTokens(tilingData.fiaArch38AttentionBaseParams.get_nextTokens());
    inputParams.set_pseS1Size(tilingData.fiaArch38AttentionBaseParams.get_pseShiftS1Size());
    inputParams.set_pseS2Size(tilingData.fiaArch38AttentionBaseParams.get_pseShiftS2Size());
}

void FiaArch38PromptTiling::SetInputParamsPseAndMask(FiaArch38FlashAttentionTilingData& tilingData)
{
    auto& inputParams = faTilingAdapter.inputParamsRegbase;
    if (!enableAlibiPse) {
        inputParams.set_pseBSize(tilingData.fiaArch38AttentionSingleCoreParams.get_pseShiftBatch());
        inputParams.set_pseShapeType(0); // 对应训练 PSE_B_N2_G_S1_S2
        inputParams.set_pseType(0);      // 对应训练 PSE_OUTER_MUL_ADD_TYPE
    }
    inputParams.set_bandIndex(0);      // 训练代码中在TND场景生效，用于计算s2方向循环的起始位置
    inputParams.set_pseEncodeType(0);  // 默认值
    inputParams.set_pseAlibiBaseS1(0); // 默认值
    inputParams.set_pseAlibiBaseS2(0); // 默认值
    inputParams.set_attenMaskShapeType(attenMaskShapeType);
    inputParams.set_attenMaskDataType(1); // 默认值
    SetAttenMaskCompressMode();
    inputParams.set_implMode(static_cast<uint8_t>(HIGH_PRECISION));
    inputParams.set_sparseType(sparseType);
    inputParams.set_needDropMaskOp(0); // 默认值
    inputParams.set_keepProb(0);       // 默认值
    inputParams.set_keepProbUint8(0);  // 默认值
    inputParams.set_dropMaskOuter(0);  // 默认值
    inputParams.set_remain(0);         // 默认值
    inputParams.set_attenMaskS2Size(tilingData.fiaArch38AttentionBaseParams.get_maskKVsSize());
    inputParams.set_rsv1(0);              // 默认值
    inputParams.set_s1SparseValidSize(0); // 临时默认值
    inputParams.set_s2SparseValidSize(0); // 临时默认值
    inputParams.set_seed(0);              // 默认值
    inputParams.set_offset(0);            // 默认值
}

void FiaArch38PromptTiling::SetInputParamsPA(FiaArch38FlashAttentionTilingData& tilingData)
{
    auto& inputParams = faTilingAdapter.inputParamsRegbase;
    // PFA
    inputParams.set_blockSize(tilingData.fiaArch38AttentionBaseParams.get_blockSize());
    inputParams.set_blockTableDim2(tilingData.fiaArch38AttentionBaseParams.get_blockTableDim2());
    inputParams.set_paBlockNumSum(tilingData.fiaArch38AttentionBaseParams.get_PABlockNumSum());
    inputParams.set_prefixSeqInnerSize(tilingData.fiaArch38AttentionBaseParams.get_prefixSeqInnerSize());
    inputParams.set_paLayoutType(tilingData.fiaArch38AttentionBaseParams.get_PAlayoutType());
    inputParams.set_attenMaskS1Size(tilingData.fiaArch38AttentionBaseParams.get_maskQsSize());
    inputParams.set_isActualSeqLengthsNull(tilingData.fiaArch38AttentionBaseParams.get_isActualSeqLengthsNull());
    inputParams.set_isActualSeqLengthsKVNull(tilingData.fiaArch38AttentionBaseParams.get_isActualSeqLengthsKVNull());
    inputParams.set_actualSeqLengthsSize(tilingData.fiaArch38AttentionBaseParams.get_actualSeqLengthsSize());
    inputParams.set_actualSeqLengthsKVSize(tilingData.fiaArch38AttentionBaseParams.get_actualSeqLengthsKVSize());
    inputParams.set_deqScaleFlag(tilingData.fiaArch38AttentionBaseParams.get_deqScaleFlag());
    inputParams.set_deqScale2Flag(tilingData.fiaArch38AttentionBaseParams.get_deqScale2Flag());
    inputParams.set_isKvContinuous(tilingData.fiaArch38AttentionBaseParams.get_isKvContinuous());
    inputParams.set_fromFused(tilingData.fiaArch38AttentionBaseParams.get_fromFused());
    inputParams.set_isBSNDOut(tilingData.fiaArch38AttentionBaseParams.get_isBSNDOut());
    inputParams.set_isGqa((tilingData.fiaArch38AttentionBaseParams.get_isIFA() != 0) || enablePFAMerge);
    inputParams.set_isSoftMaxLseEnable(tilingData.fiaArch38AttentionBaseParams.get_isSoftMaxLseEnable());
    inputParams.set_isActualSharedPrefixLenNull(
        tilingData.fiaArch38AttentionBaseParams.get_isActualSharedPrefixLenNull());
    inputParams.set_isQHasLeftPadding(tilingData.fiaArch38AttentionBaseParams.get_isQHasLeftPadding());
    inputParams.set_isKVHasLeftPadding(tilingData.fiaArch38AttentionBaseParams.get_isKVHasLeftPadding());
    inputParams.set_ropeHeadSize(tilingData.fiaArch38AttentionBaseParams.get_ropeHeadSize());
    inputParams.set_isRowInvalid(static_cast<uint8_t>(tilingData.fiaArch38AttentionBaseParams.get_isRowInvalid()));
    inputParams.set_headNumRatio(tilingData.fiaArch38AttentionBaseParams.get_headNumRatio());
}

void FiaArch38PromptTiling::SetInitOutputParams(FiaArch38FlashAttentionTilingData& tilingData)
{
    auto& initOutputParams = faTilingAdapter.initOutputParams;
    initOutputParams.set_singleCoreSize(tilingData.fiaArch38AttentionInitOutputParams.get_singleCoreSize());
    initOutputParams.set_totalOutputSize(tilingData.fiaArch38AttentionInitOutputParams.get_totalOutputSize());
    initOutputParams.set_totalSoftMaxLseOutputSize(
        tilingData.fiaArch38AttentionInitOutputParams.get_totalSoftMaxLseOutputSize());
    initOutputParams.set_needInit(static_cast<uint8_t>(tilingData.fiaArch38AttentionInitOutputParams.get_needInit()));
    initOutputParams.set_isOneN(static_cast<uint8_t>(tilingData.fiaArch38AttentionInitOutputParams.get_isOneN()));

    auto& inputParams = faTilingAdapter.inputParamsRegbase;
    inputParams.set_isPostQuantPerChnl(tilingData.fiaArch38AttentionBaseParams.get_isQuant2Perchannel());
    inputParams.set_isPostQuantBF16(tilingData.fiaArch38AttentionBaseParams.get_isQuant2BF16());
    inputParams.set_antiquantPerTensorFlag(0);
    inputParams.set_antiquantPerHeadFlag(0);
    inputParams.set_antiquantParaSeqSize(1);
}

} // namespace arch38
} // namespace optiling
