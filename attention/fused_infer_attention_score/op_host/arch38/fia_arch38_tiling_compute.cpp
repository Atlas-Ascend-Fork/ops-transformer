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
#include "../../op_kernel/arch38/fia_arch38_policy.h"

using namespace ge;
using namespace AscendC;
using namespace matmul_tiling;
namespace optiling {

namespace arch38 {

void FiaArch38PromptTiling::SetTilingDataAttribute(const FiaArch38ContextParams& contextKeyParams,
                                                   FiaArch38FlashAttentionTilingData& tilingData)
{
    tilingData.fiaArch38AttentionBaseParams.set_preTokens(sparsePreTokens);
    tilingData.fiaArch38AttentionBaseParams.set_nextTokens(sparseNextTokens);
    tilingData.fiaArch38AttentionBaseParams.set_sparseMode(static_cast<uint32_t>(sparseModeVal));

    bool isActualSeqLengthsNull = contextKeyParams.fromTilingSink == 0 ? !enableActSeqLen : true;
    bool isActualSeqLengthsKVNull = contextKeyParams.fromTilingSink == 0 ? !enableActSeqLenKV : true;
    tilingData.fiaArch38AttentionBaseParams.set_isActualSeqLengthsNull(static_cast<uint32_t>(isActualSeqLengthsNull));
    tilingData.fiaArch38AttentionBaseParams.set_isActualSeqLengthsKVNull(
        static_cast<uint32_t>(isActualSeqLengthsKVNull));
    tilingData.fiaArch38AttentionBaseParams.set_actualSeqLengthsSize(actSeqLenDims);
    tilingData.fiaArch38AttentionBaseParams.set_actualSeqLengthsKVSize(actSeqLenKVDims);

    tilingData.fiaArch38AttentionBaseParams.set_usePseShift(usePseShift);
    tilingData.fiaArch38AttentionBaseParams.set_pseShiftTypeByteNum(pseShiftTypeByteNum);
    tilingData.fiaArch38AttentionBaseParams.set_pseMaskMaxSize(pseMaskMaxSize);
    tilingData.fiaArch38AttentionSingleCoreParams.set_pseShiftBatch(pseShiftBatch);
    tilingData.fiaArch38AttentionBaseParams.set_pseShiftS1Size(pseShiftS1);
    tilingData.fiaArch38AttentionBaseParams.set_pseShiftS2Size(pseShiftS2);

    tilingData.fiaArch38AttentionBaseParams.set_isKvContinuous(contextKeyParams.isKvContinuous);
    tilingData.fiaArch38AttentionBaseParams.set_isQHasLeftPadding(contextKeyParams.queryPaddingSize != nullptr ? 1 : 0);
    tilingData.fiaArch38AttentionBaseParams.set_isKVHasLeftPadding(contextKeyParams.kvPaddingSize != nullptr ? 1 : 0);

    tilingData.fiaArch38AttentionBaseParams.set_fromFused((contextKeyParams.fromFused == FROM_FUSED_FLAG) ? 1 : 0);
    tilingData.fiaArch38AttentionBaseParams.set_isBSNDOut(contextKeyParams.isBSNDOut);
    tilingData.fiaArch38AttentionBaseParams.set_isSoftMaxLseEnable(contextKeyParams.isSoftMaxLseEnable);

    uint32_t originHeadSize = enableIFAMLA ? tilingData.fiaArch38AttentionBaseParams.get_headSize() :
                                             tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize();
    uint32_t blockElementCnt = BYTE_BLOCK / dataTypeSize;
    if (originHeadSize % blockElementCnt !=
        0) { // Determine if D is aligned with 32B, using fp16 type with 16 elements.
        tilingData.fiaArch38AttentionBaseParams.set_alignedHeadSize(
            ((originHeadSize + blockElementCnt - 1) / blockElementCnt) * blockElementCnt);
        isDNoTail = false;
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_alignedHeadSize(originHeadSize);
    }
}

void FiaArch38PromptTiling::GetEnableDN(FiaArch38FlashAttentionTilingData& tilingData,
                                        const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& valueShapeInfo,
                                        std::vector<int64_t>& actualSeqLengths,
                                        std::vector<int64_t>& actualSeqLengthsKV)
{
    // 使能 DN：镜像 kernel 模板条件 !hasAtten && s1BaseSize == 128 && dTemplate == Aligned128，
    // 即 !enableMask && AlignUp(d, 64) == 128 && sOuter == 128。
    (void)valueShapeInfo;
    (void)actualSeqLengths;
    (void)actualSeqLengthsKV;
    constexpr uint32_t dLimitDN = 128;
    constexpr uint32_t sOuterBaseSizeDN = 128;
    constexpr uint32_t sOuterLimitDN = 64;
    // kernel 侧由模板参数推导 useDn：!hasAtten && s1BaseSize == 128 && dTemplate == Aligned128。
    // 此处镜像同一规则，保证 DN 的 tilingKey 位始终与 kernel 路径一致
    // （sOuter == 128 <=> s1BaseSize == 128；不在此判断 IFA，kernel 侧看不到该信息）。
    const uint32_t sOuter = tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSOuterSize();
    enableDN = (AlignUp(queryShapeInfo.d, sOuterLimitDN) == dLimitDN) && !enableMask && (sOuter == sOuterBaseSizeDN);

    // 移植 arch35 时删除了两段 sInner=256 扩展（d==64 / perblock fp8），两者在 arch38 均不可达：
    // d==64 进不了 DN（AlignUp(64,64)!=128），fp8 perblock 被 dtype 门禁拒绝。
    // 未来放开 D64-DN 或 fp8 perblock 时，需一并恢复 sInner=256 及对应 tilingKey/模板。
}

void FiaArch38PromptTiling::SetTilingData(const FiaArch38ContextParams& contextKeyParams,
                                          const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& queryRopeShapeInfo,
                                          const PFAShapeInfo& valueShapeInfo,
                                          FiaArch38FlashAttentionTilingData& tilingData)
{
    SetTilingDataTypeByteNum(tilingData);
    SetTilingDataQuantParams(contextKeyParams, tilingData);
    SetTilingDataShapeParams(contextKeyParams, queryShapeInfo, queryRopeShapeInfo, valueShapeInfo, tilingData);
    SetTilingDataAttribute(contextKeyParams, tilingData);
}

void FiaArch38PromptTiling::SetTilingDataTypeByteNum(FiaArch38FlashAttentionTilingData& tilingData)
{
    typeByteNum = BYTE_BLOCK / dataTypeSize;
    outputTypeByteNum = BYTE_BLOCK / outputDataTypeSize;
    softmaxTypeByteNum = BYTE_BLOCK / softmaxDataTypeSize;
    maskTypeByteNum = BYTE_BLOCK / maskElemSize;
    tilingData.fiaArch38AttentionBaseParams.set_typeByteNum(typeByteNum);
    tilingData.fiaArch38AttentionBaseParams.set_maskTypeByteNum(maskTypeByteNum);
    tilingData.fiaArch38AttentionBaseParams.set_softmaxTypeByteNum(softmaxTypeByteNum);
    tilingData.fiaArch38AttentionBaseParams.set_outputTypeByteNum(outputTypeByteNum);
}

void FiaArch38PromptTiling::SetTilingDataQuantParams(const FiaArch38ContextParams& contextKeyParams,
                                                     FiaArch38FlashAttentionTilingData& tilingData)
{
    uint32_t deqScaleTypeFlag = (contextKeyParams.deqScaleType == DT_UINT64) ? 0U : 1U;
    uint32_t deqScale2TypeFlag = (contextKeyParams.deqScale2Type == DT_UINT64) ? 0U : 1U;
    tilingData.fiaArch38AttentionBaseParams.set_deqScaleFlag(deqScaleTypeFlag);
    tilingData.fiaArch38AttentionBaseParams.set_deqScale2Flag(deqScale2TypeFlag);
    tilingData.fiaArch38AttentionBaseParams.set_scaleValue(*contextKeyParams.scaleValue);
    // Perchannel judgment to be adapted, maintain the existing logic firstly.
    tilingData.fiaArch38AttentionBaseParams.set_isQuant2Perchannel(0);
    tilingData.fiaArch38AttentionBaseParams.set_isQuant2BF16(0);
    tilingData.fiaArch38AttentionBaseParams.set_isQuant2FP16(0);
    // IFA flag
    tilingData.fiaArch38AttentionBaseParams.set_isIFA(enableIFA);
    if (enablePostQuant) {
        if (contextKeyParams.scale2Shape->GetStorageShape().GetShapeSize() > 1) {
            tilingData.fiaArch38AttentionBaseParams.set_isQuant2Perchannel(1);
        }
        if (contextKeyParams.quantScale2Type == ge::DT_BF16) {
            tilingData.fiaArch38AttentionBaseParams.set_isQuant2BF16(1);
        }
        if (contextKeyParams.quantScale2Type == ge::DT_FLOAT16 && contextKeyParams.hasKeyAntiquantScale &&
            contextKeyParams.hasValueAntiquantScale) {
            tilingData.fiaArch38AttentionBaseParams.set_isQuant2FP16(1);
        }
    }
}

void FiaArch38PromptTiling::SetTilingDataShapeParams(const FiaArch38ContextParams& contextKeyParams,
                                                     const PFAShapeInfo& queryShapeInfo,
                                                     const PFAShapeInfo& queryRopeShapeInfo,
                                                     const PFAShapeInfo& valueShapeInfo,
                                                     FiaArch38FlashAttentionTilingData& tilingData)
{
    if (enablePA) {
        tilingData.fiaArch38AttentionBaseParams.set_blockSize(static_cast<int32_t>(*contextKeyParams.blockSize));
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_blockSize(BLOCK_SIZE_BASE);
    }
    tilingData.fiaArch38AttentionBaseParams.set_blockTableDim2(blockTableDim2);
    tilingData.fiaArch38AttentionBaseParams.set_PABlockNumSum(paBlockNumSum);
    uint32_t isLayoutSH = (inputLayout == InputLayout::SH) ? 1U : 0U;
    tilingData.fiaArch38AttentionBaseParams.set_isLayoutSH(isLayoutSH);
    tilingData.fiaArch38AttentionBaseParams.set_dimNumOfseq(queryShapeInfo.b);
    tilingData.fiaArch38AttentionBaseParams.set_headSize(queryShapeInfo.d);
    tilingData.fiaArch38AttentionBaseParams.set_ropeHeadSize(queryRopeShapeInfo.d);
    if (enableIFAMLA) {
        tilingData.fiaArch38AttentionBaseParams.set_qkHeadSize(queryShapeInfo.d + queryRopeShapeInfo.d);
    } else {
        tilingData.fiaArch38AttentionBaseParams.set_qkHeadSize(queryShapeInfo.d);
    }
    tilingData.fiaArch38AttentionBaseParams.set_vHeadSize(valueShapeInfo.d);
    tilingData.fiaArch38AttentionBaseParams.set_seqInnerSize(S2);
    tilingData.fiaArch38AttentionBaseParams.set_seqSize(queryShapeInfo.s);
    tilingData.fiaArch38AttentionBaseParams.set_headNumSize(queryShapeInfo.n);
    tilingData.fiaArch38AttentionBaseParams.set_batchSize(queryShapeInfo.b);
}

void FiaArch38PromptTiling::InferTilingMod(const FiaArch38ContextParams& contextKeyParams,
                                           std::vector<int64_t>& actualSeqLengths,
                                           std::vector<int64_t>& actualSeqLengthsKV, uint32_t actualSeqArrayLen,
                                           uint32_t d)
{
    // Determine whether to use the norm template
    int64_t minActualSeqLengths = INT64_MAX;
    int64_t minActualSeqLengthsKV = INT64_MAX;
    for (uint32_t i = 0; i < actualSeqArrayLen; ++i) {
        minActualSeqLengths = std::min(minActualSeqLengths, actualSeqLengths[i]);
        minActualSeqLengthsKV = std::min(minActualSeqLengthsKV, actualSeqLengthsKV[i]);
    }
    if ((minActualSeqLengths >= MATMUL_NORM_MIN_SEQ) && (minActualSeqLengthsKV >= MATMUL_NORM_MIN_SEQ) &&
        (d == MATMUL_NORM_MIN_HEADSIZE) && (inputType == ge::DT_FLOAT16) &&
        (contextKeyParams.kDataType == ge::DT_FLOAT16) && (contextKeyParams.maskDataType == ge::DT_BOOL) &&
        (outputType == ge::DT_FLOAT16) && (usePseShift == 0) && (inputLayout == InputLayout::BNSD) &&
        (sparseModeVal == SPARSE_MODE_BAND) && (!enablePA)) {
        // Currently, only the matmul norm template is open for the X1 scenario
        enableMatmulNorm = true;
    }
}

void FiaArch38PromptTiling::InferSplitCoreMode()
{
    splitCoreMode = SplitCoreMode::SPLIT_NBS_VECTOR;
}

void FiaArch38PromptTiling::InferConstantization()
{
    isConstantization = true;
}

void FiaArch38PromptTiling::GetMatMulType(matmul_tiling::DataType& mmInputType,
                                          matmul_tiling::DataType& mmOutputType) const
{
    if (inputType == ge::DT_FLOAT16 && innerPrecise == HIGH_PRECISION) {
        mmInputType = matmul_tiling::DataType::DT_FLOAT16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_BF16) {
        mmInputType = matmul_tiling::DataType::DT_BF16;
        mmOutputType = matmul_tiling::DataType::DT_FLOAT;
    } else if (inputType == ge::DT_INT8) {
        mmInputType = matmul_tiling::DataType::DT_INT8;
        mmOutputType = matmul_tiling::DataType::DT_INT32;
    }
}

bool FiaArch38PromptTiling::EnableMTE2BmmPipe(FiaArch38FlashAttentionTilingData& tilingData,
                                              matmul_tiling::MatmulApiTiling& bmm, TCubeTiling& bmmTilingData,
                                              uint32_t sOuterFactor, uint32_t sInnerFactor) const
{
    // When the size is greater than 16, use xiaoe speculative inference.
    if (tilingData.fiaArch38AttentionBaseParams.get_seqSize() > 16) {
        return true;
    }
    uint32_t baseK = 32U;
    uint32_t headSize = tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize();
    if (headSize % baseK != 0) {
        return true;
    }

    uint32_t baseM = std::min(static_cast<uint32_t>(BLOCK_SIZE_BASE), sOuterFactor);
    uint32_t baseN = std::min(static_cast<uint32_t>(512), sInnerFactor);
    if (enablePA) {
        baseN = BLOCK_SIZE_BASE;
    }
    int32_t ret = 0;
    ret = bmm.SetFixSplit(baseM, baseN, baseK);
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm SetFixSplit failed"), return false);
    bool res = bmm.GetTiling(bmmTilingData) != -1;
    return res;
}

void FiaArch38PromptTiling::EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData) const
{
    if ((bmmTilingData.get_depthA1() == 1) && (bmmTilingData.get_depthB1() == 1)) {
        bmmTilingData.set_depthA1(2); // 2 : depthA1
        bmmTilingData.set_depthB1(2); // 2 : depthB1
    }
}

bool FiaArch38PromptTiling::FiaArch38CheckBmm1(FiaArch38FlashAttentionTilingData& tilingData,
                                               TCubeTiling& bmm1TilingData, BmmShapeCtx& bmmCtx, bool autoBaseMNK)
{
    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) {
        bmmCtx.sOuterFactor = bmmCtx.sOuterFactor * cvRatio;
    }
    matmul_tiling::MatmulApiTiling bmm1(ascendPlatformInfo);

    matmul_tiling::DataType bmm1InputType = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType bmm1OutputType = matmul_tiling::DataType::DT_FLOAT16;
    GetMatMulType(bmm1InputType, bmm1OutputType);
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1InputType, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, bmm1OutputType);

    int32_t ret = bmm1.SetShape(bmmCtx.sOuterFactor, bmmCtx.sInnerFactor,
                                tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize());
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 SetShape failed"), return false);
    int32_t ratio = tilingData.fiaArch38AttentionBaseParams.get_headNumRatio();
    if (ratio == 0) {
        return false;
    }
    int32_t strideQ = tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize() *
                      tilingData.fiaArch38AttentionBaseParams.get_headNumSize();
    int32_t strideK = strideQ / ratio;

    SetBmm1OrgShape(bmm1, tilingData, strideQ, strideK);

    if (!Bmm1SetTiling(bmm1, bmm1TilingData, bmmCtx, autoBaseMNK)) {
        return false;
    }

    bmm1TilingData.set_shareMode(0);
    bmm1TilingData.set_shareL1Size(bmmCtx.l1SizeRemain);
    bmm1TilingData.set_shareL0CSize(bmmCtx.l0CSize);
    bmm1TilingData.set_shareUbSize(0);
    EnableBmmDoubleBuffer(bmm1TilingData); // Open the double buffer for BMM1 calculation, and BMM1's MTE2 can be bound.
    // Open MTE2 Matmul pipeline.
    bool res = EnableMTE2BmmPipe(tilingData, bmm1, bmm1TilingData, bmmCtx.sOuterFactor, bmmCtx.sInnerFactor);
    OP_CHECK_IF(res == false, OP_LOGE_WITHOUT_REPORT(contextKeyParamsPtr->opName, "EnableMTE2BmmPipe failed!"),
                return false);

    return true;
}

void FiaArch38PromptTiling::SetBmm1OrgShape(matmul_tiling::MatmulApiTiling& bmm1,
                                            FiaArch38FlashAttentionTilingData& tilingData, int32_t strideQ,
                                            int32_t strideK) const
{
    if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::TND)) {
        if (enableKVAntiquant || (inputLayout == InputLayout::TND && enablePA && paLayoutType == 0)) {
            bmm1.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize(), strideQ,
                             tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize());
        } else if (enableIFAMLA || enableIFA) {
            bmm1.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize(), strideK);
        } else {
            bmm1.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize(), strideQ, strideK);
        }
    } else if (inputLayout == InputLayout::BNSD) {
        if (enablePA && paLayoutType == 1) { // The left matrix of PA is BNSD, and the right matrix is BSH.
            bmm1.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize(), strideK);
        } else {
            bmm1.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize());
        }
    }
}

bool FiaArch38PromptTiling::Bmm1SetTiling(matmul_tiling::MatmulApiTiling& bmm1, TCubeTiling& bmm1TilingData,
                                          const BmmShapeCtx& bmmCtx, bool autoBaseMNK) const
{
    bmm1.SetBias(false);
    int32_t ret = bmm1.SetBufferSpace(bmmCtx.l1SizeRemain, bmmCtx.l0CSize);
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 SetBufferSpace failed"), return false);
    if (!enableIFAMLA && enablePA) {
        ret = bmm1.SetFixSplit(bmmCtx.sOuterFactor, BLOCK_SIZE_BASE);
    } else {
        ret = bmm1.SetFixSplit(bmmCtx.sOuterFactor, bmmCtx.sInnerFactor);
    }
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 SetFixSplit failed"), return false);

    ret = bmm1.GetTiling(bmm1TilingData);
    if (autoBaseMNK) {
        uint32_t baseM = std::min(static_cast<uint32_t>(BLOCK_SIZE_BASE), bmmCtx.sOuterFactor);
        uint32_t baseN = std::min(static_cast<uint32_t>(BLOCK_SIZE_BASE), bmmCtx.sInnerFactor);
        if (enableMatmulNorm) {
            uint32_t baseK = static_cast<uint32_t>(BLOCK_SIZE_BASE);
            ret = bmm1.SetFixSplit(baseM, baseN, baseK);
            OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 SetFixSplit failed"), return false);
            ret = bmm1.GetTiling(bmm1TilingData);
        } else {
            uint32_t baseK = 64U;
            if (enablePA) {
                baseN = BLOCK_SIZE_BASE;
            }
            if (ret != 0) {
                ret = bmm1.SetFixSplit(baseM, baseN, baseK);
                OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 SetFixSplit failed"), return false);
                ret = bmm1.GetTiling(bmm1TilingData);
            }
        }
    }
    // Get tiling fail for bmm1.
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm1 GetTiling failed"), return false);
    return true;
}

bool FiaArch38PromptTiling::AdjustCVTilingCVDiff(const FiaArch38ContextParams& contextKeyParams,
                                                 CVFactorParams& cvFactors,
                                                 FiaArch38FlashAttentionTilingData& tilingData,
                                                 const PFAShapeInfo& queryShapeInfo)
{
    cvFactors.sOuterFactor = SOUTER_FACTOR_DEFAULT;
    cvFactors.sInnerFactor = SINNER_FACTOR_DEFAULT;
    cvFactors.softmaxSOuterFactor = SOUTER_FACTOR_DEFAULT;

    auto compileInfoPtr = contextKeyParams.compileInfoPtr;
    bool res = FiaArch38ComputeCVDiffParams(tilingData, compileInfoPtr->l1Size, compileInfoPtr->l0CSize,
                                            cvFactors.sOuterFactor, cvFactors.sInnerFactor);
    OP_CHECK_IF(res == false,
                OP_LOGE_WITHOUT_REPORT(contextKeyParamsPtr->opName, "FiaArch38ComputeCVDiffParams failed!"),
                return false);

    if (tilingData.fiaArch38AttentionBaseParams.get_vHeadSize() <= ARCH38_SUPPORTED_D_128 &&
        !enablePFAMLA) { // 128 for D size
        AdjustCVFactorForSmallS(contextKeyParams, queryShapeInfo, tilingData, cvFactors);
    } else if (tilingData.fiaArch38AttentionBaseParams.get_vHeadSize() > ARCH38_SUPPORTED_D_128 && !enableIFAMLA &&
               !enableIFA) {
        AdjustCVFactorForLargeD(cvFactors);
    } else if (enableIFAMLA || (enableIFA && tilingData.fiaArch38AttentionBaseParams.get_vHeadSize() >
                                                 ARCH38_SUPPORTED_D_128)) { // IFA VD > 128
        AdjustCVFactorForIFAMLA(cvFactors);
    }

    cvFactors.sOuterFactor *= 2; // 2 for cv ratio equals 1:1, previous calculation based on cv ratio equals 1:2
    return true;
}

bool FiaArch38PromptTiling::AdjustCVFactorForSmallS(const FiaArch38ContextParams& contextKeyParams,
                                                    const PFAShapeInfo& queryShapeInfo,
                                                    FiaArch38FlashAttentionTilingData& tilingData,
                                                    CVFactorParams& cvFactors) const
{
    bool checkDtype = contextKeyParams.inputDataType == ge::DT_FLOAT16 || contextKeyParams.inputDataType == ge::DT_BF16;
    bool checkQueryAndValueS = queryShapeInfo.s <= SOUTER_FACTOR_DEFAULT;
    uint32_t sparseMode = tilingData.fiaArch38AttentionBaseParams.get_sparseMode();
    int32_t preTokens = tilingData.fiaArch38AttentionBaseParams.get_preTokens();
    int32_t nextTokens = tilingData.fiaArch38AttentionBaseParams.get_nextTokens();
    if (sparseMode == SPARSE_MODE_NO_MASK) {
        preTokens = (preTokens > 0) ? 0 : preTokens;
    } else if (sparseMode == SPARSE_MODE_BAND) {
        nextTokens = (nextTokens > 0) ? 0 : nextTokens;
    }
    // checkSparseMode：注意力窗口足够宽（带 mask / 全 mask / 右下 / 带宽 > 256）时走小 s（sInner=256）。
    // mask 场景若只看 preTokens+nextTokens，preTokens 已归 0 且 nextTokens=0，恒 < 256 会漏掉注册的小 s key，故显式按
    // enableMask 判宽。
    bool checkSparseMode =
        (enableMask || (sparseMode == SPARSE_MODE_ALL_MASK) || (sparseMode == SPARSE_MODE_RIGHT_DOWN) ||
         (((sparseMode == SPARSE_MODE_NO_MASK) || (sparseMode == SPARSE_MODE_BAND)) &&
          (preTokens + nextTokens > static_cast<int32_t>(SINNER_FACTOR_DEFAULT))));
    // 小 s（64/256）仅有 int8 输出的模板；fp16->fp16 固定走大 s（128/128）。
    if (checkDtype && checkQueryAndValueS && checkSparseMode && !enablePFARope && !enablePerblockQuant &&
        (contextKeyParams.outputDataType == ge::DT_INT8)) {
        cvFactors.sOuterFactor = SOUTER_FACTOR_SUB;
        cvFactors.sInnerFactor = SINNER_FACTOR_DOUBLE;
        cvFactors.softmaxSOuterFactor = SOUTER_FACTOR_SUB;
    }
    return true;
}

bool FiaArch38PromptTiling::AdjustCVFactorForLargeD(CVFactorParams& cvFactors) const
{
    // D>128 with atten_mask: enforce s_outer/s_inner = 64/128
    // (UB cannot accommodate 128/128 tiling + D256 features + mask data)
    if (enableMask) {
        cvFactors.sOuterFactor = SOUTER_FACTOR_SUB;
        cvFactors.sInnerFactor = SINNER_FACTOR_DEFAULT;
    } else {
        cvFactors.sOuterFactor = SOUTER_FACTOR_DEFAULT;
        cvFactors.sInnerFactor = SINNER_FACTOR_DEFAULT;
    }
    cvFactors.softmaxSOuterFactor = SOUTER_FACTOR_SUB;
    return true;
}

bool FiaArch38PromptTiling::AdjustCVFactorForIFAMLA(CVFactorParams& cvFactors) const
{
    if (enableIFAMLA || enableMask) {
        cvFactors.sOuterFactor = SOUTER_FACTOR_SUB;
    } else {
        cvFactors.sOuterFactor = SOUTER_FACTOR_DEFAULT;
    }
    cvFactors.sInnerFactor = SINNER_FACTOR_DEFAULT;
    cvFactors.softmaxSOuterFactor = SOUTER_FACTOR_SUB;
    return true;
}

bool FiaArch38PromptTiling::FiaArch38CheckBmm2(FiaArch38FlashAttentionTilingData& tilingData,
                                               TCubeTiling& bmm2TilingData, BmmShapeCtx& bmmCtx)
{
    int32_t ret = 0;
    matmul_tiling::MatmulApiTiling bmm2(ascendPlatformInfo);
    matmul_tiling::DataType bmm2InputType = matmul_tiling::DataType::DT_FLOAT16;
    matmul_tiling::DataType bmm2OutputType = matmul_tiling::DataType::DT_FLOAT16;
    GetMatMulType(bmm2InputType, bmm2OutputType);
    bmm2.SetAType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
    bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2InputType, false);
    bmm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND_ALIGN, bmm2OutputType);
    ret = bmm2.SetShape(bmmCtx.sOuterFactor, tilingData.fiaArch38AttentionBaseParams.get_vHeadSize(),
                        bmmCtx.sInnerFactor);
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm2 set SetShape failed"), return false);
    int32_t ratio = tilingData.fiaArch38AttentionBaseParams.get_headNumRatio();
    if (ratio == 0) {
        return false;
    }
    int32_t strideV = tilingData.fiaArch38AttentionBaseParams.get_vHeadSize() *
                      tilingData.fiaArch38AttentionBaseParams.get_headNumSize() / ratio;

    SetBmm2OrgShape(bmm2, tilingData, strideV);

    bmm2.SetBias(false);
    ret = bmm2.SetBufferSpace(bmmCtx.l1SizeRemain, bmmCtx.l0CSize);
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm2 set SetBufferSpace failed"), return false);

    if (!Bmm2GetTiling(bmm2, bmm2TilingData, tilingData, bmmCtx)) {
        return false;
    }

    bmm2TilingData.set_shareMode(0);
    bmm2TilingData.set_shareL1Size(bmmCtx.l1SizeRemain);
    bmm2TilingData.set_shareL0CSize(bmmCtx.l0CSize);
    bmm2TilingData.set_shareUbSize(0);
    return true;
}

void FiaArch38PromptTiling::SetBmm2OrgShape(matmul_tiling::MatmulApiTiling& bmm2,
                                            FiaArch38FlashAttentionTilingData& tilingData, int32_t strideV) const
{
    if ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND) || (inputLayout == InputLayout::TND)) {
        bmm2.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(), strideV,
                         tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize());
        if (enableKVAntiquant || (inputLayout == InputLayout::TND && enablePA && paLayoutType == 0)) {
            bmm2.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_vHeadSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize());
        }
    } else if (inputLayout == InputLayout::BNSD) {
        if (enablePA && paLayoutType == 1) { // The left matrix of PA is BNSD, and the right matrix is of PA is BSH.
            bmm2.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(), strideV,
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize());
        } else {
            bmm2.SetOrgShape(tilingData.fiaArch38AttentionBaseParams.get_seqSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_vHeadSize(),
                             tilingData.fiaArch38AttentionBaseParams.get_seqInnerSize());
        }
    }
}

bool FiaArch38PromptTiling::Bmm2GetTiling(matmul_tiling::MatmulApiTiling& bmm2, TCubeTiling& bmm2TilingData,
                                          FiaArch38FlashAttentionTilingData& tilingData,
                                          const BmmShapeCtx& bmmCtx) const
{
    int32_t ret = 0;
    if (bmmCtx.autoBaseMNK) {
        if (enableMatmulNorm) {
            uint32_t baseM = std::min(static_cast<uint32_t>(BLOCK_SIZE_BASE), bmmCtx.sOuterFactor);
            uint32_t baseN = std::min(static_cast<uint32_t>(BLOCK_SIZE_BASE),
                                      tilingData.fiaArch38AttentionBaseParams.get_vHeadSize());
            uint32_t baseK = static_cast<uint32_t>(BLOCK_SIZE_BASE);
            ret = bmm2.SetFixSplit(baseM, baseN, baseK);
            OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm2 SetFixSplit failed"), return false);
        }
        ret = bmm2.GetTiling(bmm2TilingData);
    } else {
        if ((isDNoTail) || (splitS2 == 0)) {
            ret = bmm2.SetFixSplit(bmmCtx.sOuterFactor, bmmCtx.dSplitFactor);
        } else {
            ret = bmm2.SetFixSplit(bmmCtx.sOuterFactor, tilingData.fiaArch38AttentionBaseParams.get_alignedHeadSize());
        }
        OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm2 SetFixSplit failed"), return false);
        ret = bmm2.GetTiling(bmm2TilingData);
    }
    OP_CHECK_IF(ret != 0, ReportTilingFailure(context_, "bmm2 set GetTiling failed"), return false);
    return true;
}

bool FiaArch38PromptTiling::FiaArch38ComputeCVDiffParams(FiaArch38FlashAttentionTilingData& tilingData, int64_t l1Size,
                                                         int64_t l0CSize, uint32_t& sOuterFactor,
                                                         uint32_t& sInnerFactor)
{
    constexpr uint32_t dSplitFactorBmm2 = 128U;
    BmmShapeCtx bmmCtx;
    bmmCtx.l1SizeRemain = l1Size;
    bmmCtx.l0CSize = l0CSize;
    bmmCtx.sOuterFactor = sOuterFactor;
    bmmCtx.sInnerFactor = sInnerFactor;
    bool res = FiaArch38CheckBmm1(tilingData, tilingData.bmm1TilingDataRect, bmmCtx, true);
    OP_CHECK_IF(!res, OP_LOGE_WITHOUT_REPORT(contextKeyParamsPtr->opName, "FiaArch38CheckBmm1 failed!"), return false);

    bmmCtx.dSplitFactor = dSplitFactorBmm2;
    bmmCtx.autoBaseMNK = true;
    res = FiaArch38CheckBmm2(tilingData, tilingData.bmm2TilingDataRect, bmmCtx);
    OP_CHECK_IF(!res, OP_LOGE_WITHOUT_REPORT(contextKeyParamsPtr->opName, "FiaArch38CheckBmm2 failed!"), return false);

    sOuterFactor = bmmCtx.sOuterFactor;
    sInnerFactor = bmmCtx.sInnerFactor;
    return true;
}

void FiaArch38PromptTiling::GetPreNextTokensLeftUp(FiaArch38FlashAttentionTilingData& tilingData,
                                                   int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                                   int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) const
{
    FiaArch38AttentionBaseParams* baseParams = &tilingData.fiaArch38AttentionBaseParams;
    if (baseParams->get_sparseMode() == SPARSE_MODE_RIGHT_DOWN) {
        preTokensLeftUp = SPARSE_MODE_INT_MAX;
        if (enableIFAMLA) {
            if (inputLayout == InputLayout::BSND || inputLayout == InputLayout::BSH ||
                inputLayout == InputLayout::TND) {
                nextTokensLeftUp = actualSeqLengthKV * gSize - actualSeqLength;
            } else { // BNSD场景下分核不做优化
                nextTokensLeftUp = SPARSE_MODE_INT_MAX;
            }
        } else {
            nextTokensLeftUp = actualSeqLengthKV - actualSeqLength;
        }
    } else if (baseParams->get_sparseMode() == SPARSE_MODE_BAND) {
        preTokensLeftUp = baseParams->get_preTokens() - actualSeqLengthKV + actualSeqLength;
        nextTokensLeftUp = baseParams->get_nextTokens() + actualSeqLengthKV - actualSeqLength;
    } else {
        preTokensLeftUp = baseParams->get_preTokens();
        nextTokensLeftUp = baseParams->get_nextTokens();
    }
}

void FiaArch38PromptTiling::UpdateTilingKeyMatmulCfg(uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeySplitCoreNBSVectorValue = 0;
    constexpr uint64_t tilingKeySplitCoreNBSCubeValue = static_cast<uint64_t>(1e6); // Position 6

    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_VECTOR) {
        tilingKey += tilingKeySplitCoreNBSVectorValue;
    } else if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) {
        tilingKey += tilingKeySplitCoreNBSCubeValue;
    } else {
        tilingKey += tilingKeySplitCoreNBSVectorValue;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeyMaskCfg(FiaArch38FlashAttentionTilingData& tilingData,
                                                   uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeyDisableMask = 0;
    constexpr uint64_t tilingKeyEnableMaskNoBand = static_cast<uint64_t>(1e8); // Position 8
    constexpr uint64_t tilingKeyEnableMaskBand = static_cast<uint64_t>(2e8);

    if (tilingData.fiaArch38AttentionBaseParams.get_useMask() == 0U) {
        tilingKey += tilingKeyDisableMask;
    } else if (tilingData.fiaArch38AttentionBaseParams.get_useMask() == 1U &&
               tilingData.fiaArch38AttentionBaseParams.get_sparseMode() == SPARSE_MODE_BAND) {
        tilingKey += tilingKeyEnableMaskBand;
    } else {
        tilingKey += tilingKeyEnableMaskNoBand;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeyPseCfg(uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeyDisablePse = 0;
    constexpr uint64_t tilingKeyEnablePse = static_cast<uint64_t>(1e9);          // Position 9
    constexpr uint64_t tilingKeyEnableAlibiPse = static_cast<uint64_t>(2e9);     // Position 9
    constexpr uint64_t tilingKeyEnableAlibiPseSqrt = static_cast<uint64_t>(3e9); // Position 9

    if (usePseShift == 0U) {
        tilingKey += tilingKeyDisablePse;
    } else if (usePseShift == 1U) {
        tilingKey += tilingKeyEnablePse;
    } else if (usePseShift == 2U) {
        tilingKey += tilingKeyEnableAlibiPse;
    } else if (usePseShift == 3U) {
        tilingKey += tilingKeyEnableAlibiPseSqrt;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeyDSizeConst(FiaArch38FlashAttentionTilingData& tilingData,
                                                      uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeyDSizeConstValue64 = static_cast<uint64_t>(1e12); // Position 12
    constexpr uint64_t tilingKeyDSizeConstValue128 = static_cast<uint64_t>(2e12);
    constexpr uint64_t tilingKeyDSizeConstValue256 = static_cast<uint64_t>(3e12);
    constexpr uint64_t tilingKeyDSizeConstValue512 = static_cast<uint64_t>(4e12);
    constexpr uint64_t tilingKeyDSizeConstValue576 = static_cast<uint64_t>(5e12); // for IFA MLA
    uint32_t dSize = tilingData.fiaArch38AttentionBaseParams.get_qkHeadSize();
    if (dSize <= FiaPolicy::DV_TEMPLATE_64) {
        tilingKey += tilingKeyDSizeConstValue64;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_128) {
        tilingKey += tilingKeyDSizeConstValue128;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_256) {
        tilingKey += tilingKeyDSizeConstValue256;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_512) {
        tilingKey += tilingKeyDSizeConstValue512;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_576) {
        tilingKey += tilingKeyDSizeConstValue576;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeySInnerConst(FiaArch38FlashAttentionTilingData& tilingData,
                                                       uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeySInnerConstValue128 = static_cast<uint64_t>(1e13); // Position 13
    constexpr uint64_t tilingKeySInnerConstValue256 = static_cast<uint64_t>(2e13);

    if (tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSInnerSize() == FiaPolicy::S2_TEMPLATE_128) {
        tilingKey += tilingKeySInnerConstValue128;
    } else if (tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSInnerSize() ==
               FiaPolicy::S2_TEMPLATE_256) {
        tilingKey += tilingKeySInnerConstValue256;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeySOuterConst(FiaArch38FlashAttentionTilingData& tilingData,
                                                       uint64_t& tilingKey)
{
    // arch38 的 sOuter 只会取 64（SOuterFactor=32*2）或 128（SOuterFactor=64*2）；
    // kernel 侧 S1TemplateType 无 Aligned96，故无 96 桶（对应 2e14 位）——arch35 的 96 分支已删除。
    constexpr uint64_t tilingKeySOuterConstValue64 = static_cast<uint64_t>(1e14); // Position 14
    constexpr uint64_t tilingKeySOuterConstValue128 = static_cast<uint64_t>(3e14);

    if (!isConstantization) {
        return;
    }
    if (tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSOuterSize() == FiaPolicy::S1_TEMPLATE_64) {
        tilingKey += tilingKeySOuterConstValue64;
    } else if (tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSOuterSize() ==
               FiaPolicy::S1_TEMPLATE_128) {
        tilingKey += tilingKeySOuterConstValue128;
    }
}

void FiaArch38PromptTiling::UpdateTilingKeyValueDSizeConst(FiaArch38FlashAttentionTilingData& tilingData,
                                                           uint64_t& tilingKey) const
{
    constexpr uint64_t tilingKeyDSizeConstValue64 = static_cast<uint64_t>(1e15); // Position 15
    constexpr uint64_t tilingKeyDSizeConstValue128 = static_cast<uint64_t>(2e15);
    constexpr uint64_t tilingKeyDSizeConstValue256 = static_cast<uint64_t>(3e15);
    constexpr uint64_t tilingKeyDSizeConstValue512 = static_cast<uint64_t>(4e15);
    uint32_t dSize = tilingData.fiaArch38AttentionBaseParams.get_vHeadSize();
    if (dSize <= FiaPolicy::DV_TEMPLATE_64) {
        tilingKey += tilingKeyDSizeConstValue64;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_128) {
        tilingKey += tilingKeyDSizeConstValue128;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_256) {
        tilingKey += tilingKeyDSizeConstValue256;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_512) {
        tilingKey += tilingKeyDSizeConstValue512;
    }
}

int64_t FiaArch38PromptTiling::GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd,
                                                  int64_t innerBlockNums) const
{
    int64_t sInnerBlockNums = 0;

    if (sInnerIndexEnd < 0) {
        sInnerBlockNums = 0;
    } else if (sInnerIndexEnd < innerBlockNums) {
        sInnerBlockNums = (sInnerIndexStart < 0) ? (sInnerIndexEnd + 1) : (sInnerIndexEnd - sInnerIndexStart + 1);
    } else {
        sInnerBlockNums = (sInnerIndexStart < 0) ?
                              innerBlockNums :
                              (sInnerIndexStart < innerBlockNums ? innerBlockNums - sInnerIndexStart : 0);
    }

    return sInnerBlockNums;
}

int64_t FiaArch38PromptTiling::GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner,
                                               int64_t sOuter, int64_t token) const
{
    // 等差数列求和公式 (首项 + 末项) * 项数 / 2 中的分母
    constexpr int32_t arithmeticSeqDivisor = 2;
    int64_t blockNums = 0;
    int64_t blockToken = 0;
    int64_t outDivIn = 0;
    if (sInner != 0) {
        blockToken = token > 0 ? ((token + sInner - 1) / sInner * sInner) : (token / sInner * sInner);
        outDivIn = sOuter > sInner ? sOuter / sInner : 1;
    }
    int64_t InDivOut = 0;
    if (sOuter != 0) {
        InDivOut = sInner > sOuter ? sInner / sOuter : 1;
    }
    int64_t tolerance = 0;
    int64_t smallSize = 0;
    if (outDivIn >= 1) {
        tolerance = outDivIn;
        smallSize = sInner;
    } else {
        tolerance = InDivOut;
        smallSize = sOuter;
    }
    int64_t innerCutBlockNums = (blockSeqLengthKV - blockToken) / smallSize - tolerance;
    int64_t innerCutBlockLeftNums = -blockToken / smallSize - tolerance;
    int64_t innerCutBlockDownNums = (blockSeqLengthKV - blockSeqLength - blockToken) / smallSize - tolerance;
    blockNums += (innerCutBlockNums > 0) ? (innerCutBlockNums % tolerance + innerCutBlockNums) *
                                               (innerCutBlockNums / tolerance + 1) / arithmeticSeqDivisor :
                                           0;
    blockNums -= (innerCutBlockLeftNums > 0) ? (innerCutBlockLeftNums % tolerance + innerCutBlockLeftNums) *
                                                   (innerCutBlockLeftNums / tolerance + 1) / arithmeticSeqDivisor :
                                               0;
    blockNums -= (innerCutBlockDownNums > 0) ? (innerCutBlockDownNums % tolerance + innerCutBlockDownNums) *
                                                   (innerCutBlockDownNums / tolerance + 1) / arithmeticSeqDivisor :
                                               0;
    return blockNums;
}

// 函数内部不处理prefix逻辑，prefix场景下入参需自行传入actualSeqLengthKV + prefix
void FiaArch38PromptTiling::FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV,
                                                   int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) const
{
    // 若出现行无效，需要重新计算nexttokens，pretokens，actualseqlen，以便正确计算分核核数
    int64_t nextTokensError = (nextTokensLeftUp < 0) ? -nextTokensLeftUp : 0;
    int64_t preTokensError = (actualSeqLength > actualSeqLengthKV + preTokensLeftUp) ?
                                 (actualSeqLength - actualSeqLengthKV - preTokensLeftUp) :
                                 0;

    // 若出现上方行无效，需要重新计算nexttokens，pretokens，actualseqlen
    nextTokensLeftUp += nextTokensError;
    preTokensLeftUp -= nextTokensError;
    actualSeqLength -= nextTokensError;

    // 若出现下方行无效，需要重新计算actualseqlen
    actualSeqLength -= preTokensError;
}

int64_t FiaArch38PromptTiling::GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV,
                                                       uint32_t sOuterSize, uint32_t sInnerSize,
                                                       int64_t preTokensLeftUp, int64_t nextTokensLeftUp,
                                                       bool isAttenMaskUsed) const
{
    if (!isAttenMaskUsed) {
        int64_t outerBlockNums = (actualSeqLength + sOuterSize - 1) / sOuterSize;
        int64_t innerBlockNums = (actualSeqLengthKV + sInnerSize - 1) / sInnerSize;
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        return toCalcBlockNums;
    } else {
        int64_t innerBlockNums =
            (actualSeqLengthKV + static_cast<int64_t>(sInnerSize) - 1) / static_cast<int64_t>(sInnerSize);
        int64_t blockSeqLengthKV = innerBlockNums * static_cast<int64_t>(sInnerSize);
        int64_t outerBlockNums =
            (actualSeqLength + static_cast<int64_t>(sOuterSize) - 1) / static_cast<int64_t>(sOuterSize);
        int64_t blockSeqLength = outerBlockNums * static_cast<int64_t>(sOuterSize);
        int64_t toCalcBlockNums = innerBlockNums * outerBlockNums;
        // Must meet this condition : pretoken + nexttoken > 0
        toCalcBlockNums -= GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
                                           static_cast<int64_t>(sOuterSize), nextTokensLeftUp);
        toCalcBlockNums -=
            GetCutBlockNums(blockSeqLengthKV, blockSeqLength, static_cast<int64_t>(sInnerSize),
                            static_cast<int64_t>(sOuterSize), blockSeqLengthKV - blockSeqLength + preTokensLeftUp);
        return toCalcBlockNums;
    }
}

void FiaArch38PromptTiling::ComputeSplitNBSeq(FiaArch38FlashAttentionTilingData& tilingData, uint32_t batchSize,
                                              const size_t tilingElementArrayLen, SplitNBSeqCtx& ctx)
{
    FiaArch38AttentionBaseParams* baseParams = &tilingData.fiaArch38AttentionBaseParams;
    ctx.coreSposEnd.resize(tilingElementArrayLen, 0U);
    ctx.coreSposStart.resize(tilingElementArrayLen, 0U);
    ctx.coreSidEnd.resize(tilingElementArrayLen, 0U);
    ctx.coreSidStart.resize(tilingElementArrayLen, 0U);
    ctx.coreNidEnd.resize(tilingElementArrayLen, 0U);
    ctx.coreNidStart.resize(tilingElementArrayLen, 0U);
    ctx.bnStartIdx.resize(tilingElementArrayLen, 0U);
    ctx.gS1StartIdx.resize(tilingElementArrayLen, 0L);
    ctx.curWight = 0;
    ctx.curCore = 0;
    ctx.tmpCoreNidEnd = 0; // actual seq为0时不分配核
    ctx.tmpCoreSidEnd = 0;
    ctx.tmpCoreSposEnd = 0;
    const uint32_t headNumSize = baseParams->get_headNumSize();
    for (uint32_t sIdx = 0; sIdx < batchSize; sIdx++) {
        for (uint32_t headNum = 0; headNum < headNumSize; headNum++) {
            SplitOneHeadBlock(tilingData, sIdx, headNum, headNumSize, ctx);
        }
    }
    ctx.coreNidEnd[ctx.curCore] = ctx.tmpCoreNidEnd;
    ctx.coreSidEnd[ctx.curCore] = ctx.tmpCoreSidEnd;
    ctx.coreSposEnd[ctx.curCore] = ctx.tmpCoreSposEnd;
    ctx.bnStartIdx[ctx.curCore + 1] = batchSize * headNumSize;
    ctx.gS1StartIdx[ctx.curCore + 1] = ctx.tmpCoreSposEnd;

    FiaArch38AttentionSeqParams* seqParams = &tilingData.fiaArch38AttentionSeqParams;
    seqParams->set_CoreHeadNumTail(ctx.coreNidStart.data());
    seqParams->set_actualS1(ctx.coreNidEnd.data());
    seqParams->set_actualCoreNums(ctx.coreSidStart.data());
    seqParams->set_singleCoreHeadNumSize(ctx.coreSidEnd.data());
    seqParams->set_coreSeqPosStart(ctx.coreSposStart.data());
    seqParams->set_coreSeqPosEnd(ctx.coreSposEnd.data());

    faTilingAdapter.multiCoreParamsRegbase.set_bnStartIdx(ctx.bnStartIdx.data());
    faTilingAdapter.multiCoreParamsRegbase.set_sparseStartIdx(ctx.gS1StartIdx.data());
}

void FiaArch38PromptTiling::SplitOneHeadBlock(FiaArch38FlashAttentionTilingData& tilingData, uint32_t sIdx,
                                              uint32_t headNum, uint32_t headNumSize, SplitNBSeqCtx& ctx) const
{
    // 针对行无效情况修正actualseqlen
    int64_t preTokensLeftUp = 0;
    int64_t nextTokensLeftUp = 0;
    GetPreNextTokensLeftUp(tilingData, (*ctx.actualSeqLengths)[sIdx],
                           (*ctx.actualSeqLengthsKV)[sIdx] + actualSharedPrefixLen, preTokensLeftUp, nextTokensLeftUp);
    int64_t actualSeqLength = (*ctx.actualSeqLengths)[sIdx];
    int64_t actualSeqLengthKV = (*ctx.actualSeqLengthsKV)[sIdx];
    FixParamWithRowInvalid(actualSeqLength, actualSeqLengthKV + actualSharedPrefixLen, preTokensLeftUp,
                           nextTokensLeftUp);

    int64_t outerBlockNums = (actualSeqLength + ctx.sOuterSize - 1) / ctx.sOuterSize;
    int64_t innerBlockNums = (actualSeqLengthKV + ctx.sInnerSize - 1) / ctx.sInnerSize +
                             (actualSharedPrefixLen + ctx.sInnerSize - 1) / ctx.sInnerSize;
    for (uint32_t sOuterIndex = 0; sOuterIndex < outerBlockNums; sOuterIndex++) {
        int64_t dif = static_cast<int64_t>(ctx.coreWightTarget * static_cast<double>(ctx.curCore + 1)) - ctx.curWight;
        int64_t sInnerIndexStart =
            -(preTokensLeftUp > 0 ?
                  (preTokensLeftUp + static_cast<int64_t>(ctx.sInnerSize) - 1) / static_cast<int64_t>(ctx.sInnerSize) :
                  preTokensLeftUp / static_cast<int64_t>(ctx.sInnerSize));
        int64_t sInnerIndexEnd = nextTokensLeftUp > 0 ? (nextTokensLeftUp + static_cast<int64_t>(ctx.sInnerSize) - 1) /
                                                            static_cast<int64_t>(ctx.sInnerSize) :
                                                        nextTokensLeftUp / static_cast<int64_t>(ctx.sInnerSize);

        // The number of innerBlock blocks in each outBlock row represents the calculation amount of each outBlock row.
        int64_t sInnerBlockNums = GetSInnerBlockNums(sInnerIndexStart, sInnerIndexEnd, innerBlockNums);
        if (sInnerBlockNums - dif > dif &&
            !(ctx.tmpCoreNidEnd == 0 && ctx.tmpCoreSidEnd == 0 && ctx.tmpCoreSposEnd == 0)) {
            ctx.coreNidEnd[ctx.curCore] = ctx.tmpCoreNidEnd;
            ctx.coreSidEnd[ctx.curCore] = ctx.tmpCoreSidEnd;
            ctx.coreSposEnd[ctx.curCore] = ctx.tmpCoreSposEnd;
            ctx.curCore += 1;
            ctx.coreNidStart[ctx.curCore] = headNum;
            ctx.coreSidStart[ctx.curCore] = sIdx;
            ctx.coreSposStart[ctx.curCore] = sOuterIndex;
            ctx.bnStartIdx[ctx.curCore] = sIdx * headNumSize + headNum;
            ctx.gS1StartIdx[ctx.curCore] = sOuterIndex;
        }
        ctx.tmpCoreNidEnd = headNum + 1;
        ctx.tmpCoreSidEnd = sIdx + 1;
        ctx.tmpCoreSposEnd = sOuterIndex + 1;

        ctx.curWight += sInnerBlockNums;
        preTokensLeftUp -= ctx.sOuterSize;
        nextTokensLeftUp += ctx.sOuterSize;
    }
}

void FiaArch38PromptTiling::FiaArch38SplitNBSeq(FiaArch38FlashAttentionTilingData& tilingData,
                                                std::vector<int64_t>& actualSeqLengths,
                                                std::vector<int64_t>& actualSeqLengthsKV, bool isAttenMaskUsed)
{
    FiaArch38AttentionBaseParams* baseParams = &tilingData.fiaArch38AttentionBaseParams;
    FiaArch38AttentionSingleCoreParams* singleCoreParams = &tilingData.fiaArch38AttentionSingleCoreParams;
    uint32_t curCoreNum = coreNum;
    uint32_t batchSize = baseParams->get_dimNumOfseq();
    uint32_t sOuterSize = singleCoreParams->get_singleProcessSOuterSize();
    uint32_t sInnerSize = singleCoreParams->get_singleProcessSInnerSize();

    if (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) { // From the perspective of cube
        sOuterSize = sOuterSize * cvRatio;
        curCoreNum = curCoreNum / cvRatio;
    }

    int64_t totalBlockNumsOneHead = 0; // The calculation amount of all sequences for a single head

    std::vector<uint32_t> sInnerLoopTimes(batchSize);
    uint32_t multiSmaxsInnerLoopTimes = 0U;
    for (uint32_t sIdx = 0; sIdx < batchSize; sIdx++) {
        int64_t actualSeqLengthsTmp = actualSeqLengths[sIdx]; // 用于存放减去行无效后，真实的actseqlen
        int64_t preTokensLeftUp = 0;
        int64_t nextTokensLeftUp = 0;
        GetPreNextTokensLeftUp(tilingData, actualSeqLengths[sIdx], actualSeqLengthsKV[sIdx] + actualSharedPrefixLen,
                               preTokensLeftUp, nextTokensLeftUp);

        // 计算各sparse mode情况下，减去行无效后真实的actseqlen
        FixParamWithRowInvalid(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx], preTokensLeftUp, nextTokensLeftUp);

        // sinner方向块数，prefix和origin是分开切的。
        sInnerLoopTimes[sIdx] = (actualSeqLengthsKV[sIdx] + sInnerSize - 1) / sInnerSize +
                                (actualSharedPrefixLen + sInnerSize - 1) / sInnerSize;
        multiSmaxsInnerLoopTimes = std::max(multiSmaxsInnerLoopTimes, sInnerLoopTimes[sIdx]);

        totalBlockNumsOneHead +=
            GetCalcBlockNumsOneHead(actualSeqLengthsTmp, actualSeqLengthsKV[sIdx], sOuterSize, sInnerSize,
                                    preTokensLeftUp, nextTokensLeftUp, isAttenMaskUsed);
    }
    singleCoreParams->set_multiSmaxsInnerLoopTimes(multiSmaxsInnerLoopTimes);

    // Amount of computation per core
    double coreWightTarget =
        (static_cast<double>(totalBlockNumsOneHead * baseParams->get_headNumSize()) / static_cast<double>(curCoreNum));

    int64_t s1OuterSize = (baseParams->get_seqSize() + sOuterSize - 1) / sOuterSize;
    faTilingAdapter.multiCoreParamsRegbase.set_s1OuterSize(s1OuterSize);

    // The tiling structure element needs to have a length greater than or equal to the length specified
    // by TILING_DATA_FIELD_DEF_ARR. If the tiling structure definition specifies a length of 64,
    // the vector definition needs to compare its size with coreNum and take the larger value
    const size_t tilingElementArrayLen =
        (static_cast<size_t>(curCoreNum) > 64UL) ? static_cast<size_t>(curCoreNum) : 64UL;
    SplitNBSeqCtx splitCtx;
    splitCtx.sOuterSize = sOuterSize;
    splitCtx.sInnerSize = sInnerSize;
    splitCtx.coreWightTarget = coreWightTarget;
    splitCtx.actualSeqLengths = &actualSeqLengths;
    splitCtx.actualSeqLengthsKV = &actualSeqLengthsKV;
    ComputeSplitNBSeq(tilingData, batchSize, tilingElementArrayLen, splitCtx);
    uint32_t curIndx = splitCtx.curCore;

    uint32_t actualCoreNums = (splitCoreMode == SplitCoreMode::SPLIT_NBS_CUBE) ? (curIndx + 1) * cvRatio : curIndx + 1;
    singleCoreParams->set_actualCoreNums(actualCoreNums);
    int64_t sinnerBlocknum = (baseParams->get_seqInnerSize() + sInnerSize - 1) / sInnerSize;
    SetMultiCoreParamsRegbase((totalBlockNumsOneHead / sinnerBlocknum) * baseParams->get_headNumSize(),
                              static_cast<int64_t>((curIndx + 1)));
}

void FiaArch38PromptTiling::FiaArch38InitSoftmaxLseOutputSplit(int64_t totalSize,
                                                               FiaArch38FlashAttentionTilingData& tilingData) const
{
    FiaArch38AttentionInitOutputParams* initParams = &tilingData.fiaArch38AttentionInitOutputParams;
    initParams->set_totalSoftMaxLseOutputSize(totalSize);
}

void FiaArch38PromptTiling::UpdateTilingKeyFlag(const FiaArch38ContextParams& contextKeyParams,
                                                uint64_t& tilingKey) const
{
    uint64_t binaryFlag = 0;
    auto queryDtype = contextKeyParams.inputDataType;
    auto kvDtype = contextKeyParams.kDataType;

    if ((queryDtype == ge::DT_FLOAT16) && (kvDtype == ge::DT_INT8)) {
        // 4bit flag bit, the leftmost side indicates whether to perform inverse quantization operation,
        // with a corresponding value of 2**3 = 8, and the remaining 3bit is reserved
        binaryFlag += 8U;
    }
    // If inverse quantization is performed, tilingKey should increase by 8*1e11.
    tilingKey += (binaryFlag * static_cast<uint64_t>(1e11));
    return;
}

bool FiaArch38PromptTiling::TilingGetTilingKeyAttentionAscendC(uint64_t& tilingKey,
                                                               FiaArch38ContextParams& contextKeyParams,
                                                               FiaArch38FlashAttentionTilingData& tilingData)
{
    auto attenMaskElemType = contextKeyParams.maskDataType;
    tilingData.fiaArch38AttentionBaseParams.set_attenMaskElemType(attenMaskElemType);

    tilingKey = 1012U; // 1012：CV diff, +1000; new_tiling, +10; not distinguishing between tail and total, +2.
    AddTilingKeyDtypeAndLayout(contextKeyParams, tilingKey);
    UpdateTilingKeyMatmulCfg(tilingKey);
    UpdateTilingKeyDSizeConst(tilingData, tilingKey);
    UpdateTilingKeyValueDSizeConst(tilingData, tilingKey);
    UpdateTilingKeySInnerConst(tilingData, tilingKey);
    UpdateTilingKeySOuterConst(tilingData, tilingKey);
    UpdateTilingKeyMaskCfg(tilingData, tilingKey);
    UpdateTilingKeyPseCfg(tilingKey);
    UpdateTilingKeyFlag(contextKeyParams, tilingKey);
    AddTilingKeyFeatureFlag(tilingKey);
    return true;
}

void FiaArch38PromptTiling::AddTilingKeyDtypeAndLayout(const FiaArch38ContextParams& contextKeyParams,
                                                       uint64_t& tilingKey) const
{
    auto inputDataType = contextKeyParams.inputDataType;   // input q
    auto outputDataType = contextKeyParams.outputDataType; // output tensor
    // fp16 high precision mode, regarded as a type 600.
    tilingKey += ((inputDataType == ge::DT_FLOAT16) && (innerPrecise == HIGH_PRECISION)) ? 600U : 0;
    tilingKey += (inputDataType == ge::DT_BF16) ? 100U : 0;          // 100: bf16
    tilingKey += (inputDataType == ge::DT_INT8) ? 200U : 0;          // 200: int8
    tilingKey += (inputDataType == ge::DT_HIFLOAT8) ? 400U : 0;      // 400: hifloat8
    tilingKey += (inputDataType == ge::DT_FLOAT8_E4M3FN) ? 900U : 0; // 900: float8_e4m3
    tilingKey += (enablePerblockQuant == true) ? 1U : 0;             // 0: FP8 perblock dequant
    // When the output dtype is bf16, tilingKey should increase by 10000.
    tilingKey += (outputDataType == ge::DT_BF16) ? static_cast<uint64_t>(1e4) : 0;
    tilingKey += (outputDataType == ge::DT_INT8) ? static_cast<uint64_t>(2e4) :
                                                   0; // 20000: The situation of outputDataType == ge::DT_INT8
    tilingKey += (outputDataType == ge::DT_HIFLOAT8) ? static_cast<uint64_t>(4e4) :
                                                       0; // 40000: The situation of outputDataType == ge::DT_HIFLOAT8
    tilingKey += (outputDataType == ge::DT_FLOAT8_E4M3FN) ?
                     static_cast<uint64_t>(6e4) :
                     0; // 60000: The situation of outputDataType == ge::DT_FLOAT8_E4M3FN
    // When the inputLayout is TND, plus 2e5.
    // When the inputLayout is BSH or BSND, plus 1e5.
    uint64_t tilingKeyLayout = (inputLayout == InputLayout::TND) ? static_cast<uint64_t>(2e5) :
                               ((inputLayout == InputLayout::BSH) || (inputLayout == InputLayout::BSND)) ?
                                                                   static_cast<uint64_t>(1e5) :
                                                                   0U;
    tilingKey += tilingKeyLayout;
}

void FiaArch38PromptTiling::AddTilingKeyFeatureFlag(uint64_t& tilingKey) const
{
    if (enablePA) {
        tilingKey += static_cast<uint64_t>(2e7); // 2e7: the situation of PA
    }

    if (enableIFAMLA) {
        tilingKey += static_cast<uint64_t>(3e7); // 3e7: the situation of IFA MLA
    }

    if (enableDN) {
        tilingKey += static_cast<uint64_t>(4e7); // 4e7: the situation of DN
    }

    if (isKVHasPrefix) {
        tilingKey += static_cast<uint64_t>(1e8); // 1e8: the situation of prefix
    }

    if (enableFlashDecode) {
        tilingKey += static_cast<uint64_t>(1e10); // 1e10: the situation of FD
    }
}

size_t FiaArch38PromptTiling::GetPFAWorkSpaceSize(FiaArch38FlashAttentionTilingData& tilingData)
{
    constexpr uint32_t S1_BASIC_BLOCK_FACTOR = 2;              // S1 基本块 = SOuter × 2
    constexpr uint32_t WORKSPACE_BMM_FACTOR = 3;               // bmm2/vec2 在 workspace 中占 3 份
    constexpr uint32_t PA_WORKSPACE_ALIGN_FACTOR = 2 * 2 * 64; // PA 场景额外 workspace 对齐（2 bmm × 2 × 64B）

    size_t sysWorkspaceSize = 0;
    size_t curWorkspaceSize = 0;
    const uint64_t defaultSysWorkspaceSize910B = 16U * 1024U * 1024U;
    sysWorkspaceSize = defaultSysWorkspaceSize910B; // sys workspace size default value

    uint32_t kvSplitPart = faTilingAdapter.inputParamsRegbase.get_kvSplitPart();
    auto batchSize = tilingData.fiaArch38AttentionBaseParams.get_batchSize();
    auto headNumSize = tilingData.fiaArch38AttentionBaseParams.get_headNumSize();
    auto vHeadSize = tilingData.fiaArch38AttentionBaseParams.get_vHeadSize();
    size_t accumOutSize = batchSize * gSize * headNumSize * kvSplitPart * vHeadSize * sizeof(float);
    size_t logSumExpSize = batchSize * gSize * headNumSize * kvSplitPart * BYTE_BLOCK * 2;

    int64_t bmm2Bytes = 0;
    int64_t vec2Bytes = 0;
    uint32_t dSize = tilingData.fiaArch38AttentionBaseParams.get_vHeadSize();
    uint32_t dVBasicBlock = ComputeDvBasicBlock(dSize);
    int64_t bmm2ResBlockSize = dVBasicBlock;
    if (dVBasicBlock > FiaPolicy::DV_TEMPLATE_256) {
        bmm2ResBlockSize = FiaPolicy::DV_TEMPLATE_512;
    }
    uint32_t s1BasicBlock =
        tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSOuterSize() * S1_BASIC_BLOCK_FACTOR;
    uint32_t sOuter = tilingData.fiaArch38AttentionSingleCoreParams.get_singleProcessSOuterSize();
    uint32_t calcTypeSize = softmaxDataTypeSize;
    // 与 kernel 共用同一判定（dV 模板桶 + S1 基本块 + mask），避免 host/kernel 条件漂移。
    bool forceBmm2UB = FiaPolicy::IsForceBmm2UB(enableMask, dVBasicBlock, sOuter);
    if (!forceBmm2UB &&
        ((!enableDN && dSize > FiaPolicy::DV_TEMPLATE_128) || (enableDN && dSize > FiaPolicy::DV_TEMPLATE_192))) {
        bmm2Bytes = static_cast<int64_t>(s1BasicBlock) * bmm2ResBlockSize * calcTypeSize;
        if (dVBasicBlock > FiaPolicy::DV_TEMPLATE_256) {
            vec2Bytes = static_cast<int64_t>(s1BasicBlock) * dVBasicBlock * calcTypeSize;
        }
    }
    bmm2Bytes = AlignUp(bmm2Bytes, GM_ALIGN);
    vec2Bytes = AlignUp(vec2Bytes, GM_ALIGN);
    curWorkspaceSize = static_cast<size_t>((bmm2Bytes + vec2Bytes) * WORKSPACE_BMM_FACTOR * coreNum) +
                       sysWorkspaceSize + accumOutSize + logSumExpSize;
    if (enablePA) {
        curWorkspaceSize += static_cast<uint64_t>(coreNum) * PA_WORKSPACE_ALIGN_FACTOR;
    }
    return curWorkspaceSize;
}

uint32_t FiaArch38PromptTiling::ComputeDvBasicBlock(uint32_t dSize) const
{
    if (dSize <= FiaPolicy::DV_TEMPLATE_64) {
        return FiaPolicy::DV_TEMPLATE_64;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_128) {
        return FiaPolicy::DV_TEMPLATE_128;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_256) {
        return FiaPolicy::DV_TEMPLATE_256;
    } else if (dSize <= FiaPolicy::DV_TEMPLATE_512) {
        return FiaPolicy::DV_TEMPLATE_512;
    }
    return 0;
}

ge::graphStatus FiaArch38PromptTiling::AdjustTilingData(FiaArch38ContextParams& contextKeyParams,
                                                        FiaArch38FlashAttentionTilingData& tilingData,
                                                        const PFAShapeInfo& queryShapeInfo)
{
    uint32_t sOuterFactor = 0;
    uint32_t sInnerFactor = 0;
    uint32_t softmaxSInnerFactor = 0;
    uint32_t softmaxSOuterFactor = 0;
    // Currently, there will be no D splitting scenario, and split D = 0 is default when splitting.
    CVFactorParams cvFactors;
    auto ret = AdjustCVTilingCVDiff(contextKeyParams, cvFactors, tilingData, queryShapeInfo);
    OP_CHECK_IF(!ret, ReportTilingFailure(context_, "adjust tiling cv diff fail"), return ge::GRAPH_FAILED);
    sOuterFactor = cvFactors.sOuterFactor;
    sInnerFactor = cvFactors.sInnerFactor;
    softmaxSOuterFactor = cvFactors.softmaxSOuterFactor;

    // fp16 输入的部分形状没有注册 tiling key，在此 fail fast。
    // 该判断需与 fia_arch38_entry_regbase.h 的 key 列表保持一致。
    if (CheckFp16TilingKeySupport(contextKeyParams, sOuterFactor, queryShapeInfo) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    softmaxSInnerFactor = sInnerFactor;

    tilingData.fiaArch38AttentionBaseParams.set_softmaxOuterSize(softmaxSOuterFactor);
    tilingData.fiaArch38AttentionSingleCoreParams.set_singleProcessSOuterSize(sOuterFactor);
    tilingData.fiaArch38AttentionSingleCoreParams.set_singleProcessSInnerSize(sInnerFactor);
    tilingData.fiaArch38AttentionBaseParams.set_splitS2(splitS2);

    sOuterFactorTiling = sOuterFactor;
    softmaxSInnerFactorTiling = softmaxSInnerFactor;
    softmaxSOuterFactorTiling = softmaxSOuterFactor;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FiaArch38PromptTiling::CheckFp16TilingKeySupport(const FiaArch38ContextParams& contextKeyParams,
                                                                 uint32_t sOuterFactor,
                                                                 const PFAShapeInfo& queryShapeInfo) const
{
    const bool isFp16Input = (inputType == ge::DT_FLOAT16);
    const bool isSmallSOuter = (sOuterFactor == 64U);
    const bool isHeadSize64 = (static_cast<int64_t>(queryShapeInfo.d) == ARCH38_SUPPORTED_D_64);
    // fp16 -> fp16：d = 64 且无 mask 的组合没有注册 key（小 s 已固定走大 s）。
    OP_CHECK_IF((isFp16Input && (outputType == ge::DT_FLOAT16) && isHeadSize64 && (!enableMask)),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "input_dtype, output_dtype, head_size, atten_mask",
                    (std::string(GetPfaDataTypeStr(contextKeyParams.inputDataType)) + ", " +
                     GetPfaDataTypeStr(outputType) + ", " + std::to_string(queryShapeInfo.d) + ", absent")
                        .c_str(),
                    "fp16->fp16 with head size 64 without atten_mask has no registered tiling key on this platform."),
                return ge::GRAPH_FAILED);
    // fp16 -> int8：只注册了小 s（64/256）模板，且 d = 64 带 mask 的组合没有注册 key。
    OP_CHECK_IF((isFp16Input && (outputType == ge::DT_INT8) && ((!isSmallSOuter) || (isHeadSize64 && enableMask))),
                OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
                    contextKeyParams.opName, "output_dtype, sOuter, head_size, atten_mask",
                    (std::string(GetPfaDataTypeStr(outputType)) + ", " + std::to_string(sOuterFactor) + ", " +
                     std::to_string(queryShapeInfo.d) + ", " + (enableMask ? "true" : "false"))
                        .c_str(),
                    "fp16->int8 only supports sOuter = 64, and head size 64 combined with atten_mask "
                    "has no registered tiling key on this platform."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

} // namespace arch38
} // namespace optiling
