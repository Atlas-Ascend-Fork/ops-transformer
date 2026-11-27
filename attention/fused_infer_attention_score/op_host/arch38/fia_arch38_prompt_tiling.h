/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FIA_ARCH38_PROMPT_TILING_H_
#define FIA_ARCH38_PROMPT_TILING_H_
#include "op_host/data_copy_transpose_tiling_def.h"
#include "op_host/data_copy_transpose_tiling.h"
#include "register/tilingdata_base.h"
#include "../../../common/op_host/fia_tiling_templates_registry.h"
#include "fia_arch38_context.h"
#include "fia_arch38_tiling_data.h"
#include "fia_arch38_tiling_common.h"
#include "fia_arch38_tiling_structs.h"
#include "fia_arch38_tiling_utils.h"

namespace optiling {
namespace arch38 {

std::string GetPfaDataTypeStr(ge::DataType type);

// IFA 增量（decode 单 token，复用 PFA 模板）场景门禁：校验 IFA 特有边界（blockTable/BNSD 维度/sQ==1/sKV/head
// 数/learnable_sink）， 平台级约束由 CheckArch38PlatformSupported 统一兜底。
ge::graphStatus CheckArch38IncreInputs(const gert::TilingContext& context);

// arch38 已验证的 head size 白名单，PFA 与 IFA 两条场景门禁共用
constexpr int64_t ARCH38_SUPPORTED_D_64 = 64;
constexpr int64_t ARCH38_SUPPORTED_D_80 = 80;
constexpr int64_t ARCH38_SUPPORTED_D_128 = 128;
constexpr int64_t ARCH38_SUPPORTED_D_256 = 256;

// IFA 增量场景边界（复用 PFA 模板前做门禁）
constexpr int64_t ARCH38_IFA_SUPPORTED_MAX_N = 128;
constexpr int64_t ARCH38_IFA_SUPPORTED_MAX_SKV = 4101;
constexpr size_t BNSD_DIM_NUM = 4U;
constexpr size_t BNSD_DIM_S = 2U;

// from_fused 场景标记值
constexpr uint32_t FROM_FUSED_FLAG = 71;

inline bool IsArch38SupportedHeadSize(int64_t d)
{
    return (d == ARCH38_SUPPORTED_D_64) || (d == ARCH38_SUPPORTED_D_80) || (d == ARCH38_SUPPORTED_D_128) ||
           (d == ARCH38_SUPPORTED_D_256);
}

// arch38 已注册模板覆盖的 dtype 组合，PFA 与 IFA 两条场景门禁共用：
// Q/K/V 需同为 fp16 或同为 int8，输出为 fp16 或 int8；d = 256 仅有 fp16 Q/K/V -> fp16 输出的模板。
inline bool IsArch38SupportedDataType(ge::DataType queryDtype, ge::DataType kvDtype, ge::DataType outputDtype,
                                      int64_t d)
{
    if ((queryDtype != kvDtype) || ((queryDtype != ge::DT_FLOAT16) && (queryDtype != ge::DT_INT8)) ||
        ((outputDtype != ge::DT_FLOAT16) && (outputDtype != ge::DT_INT8))) {
        return false;
    }
    return (d != ARCH38_SUPPORTED_D_256) || ((queryDtype == ge::DT_FLOAT16) && (outputDtype == ge::DT_FLOAT16));
}

class FiaArch38PromptTiling {
public:
    explicit FiaArch38PromptTiling(gert::TilingContext* context)
        : context_(context)
    {}
    ~FiaArch38PromptTiling() = default;
    ge::graphStatus RunBigKernelTilingWithParams(FiaArch38ContextParams& contextKeyParams, uint64_t& tilingKey,
                                                 uint32_t& numBlocksToBeSet,
                                                 FiaArch38FlashAttentionTilingData& tilingData);
    ge::graphStatus FiaArch38SetTilingData(gert::TilingContext* context);
    ge::graphStatus DoSubOpTiling(FiaArch38FlashAttentionTilingData& tilingData,
                                  FiaArch38ContextParams& contextParamsForPFATiling);
    bool CheckNonEmptyShapeExceptions(const FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* shape,
                                      const std::string& sName) const;

protected:
    void FiaArch38InitOutputSplit(int64_t totalSize, FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckEmptyTensor(FiaArch38ContextParams& contextKeyParams) const;
    ge::graphStatus SetEmptyTensor(FiaArch38ContextParams& contextKeyParams, uint64_t& tilingKey,
                                   uint32_t& numBlocksToBeSet, FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckIODataType(FiaArch38ContextParams& contextKeyParams);
    bool SetInputLayout(const char* layout);
    bool GetAndCheckShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& shapeInfo,
                          const gert::StorageShape* shape, const std::string& sName) const;
    bool GetAndCheckRopeShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& shapeInfo,
                              PFAShapeInfo& ropeShapeInfo, const gert::StorageShape* shape,
                              const RopeNamePair& names) const;
    bool CheckRopeShapeDim(const FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* shape,
                           const std::string& sName, const PFAShapeInfo& shapeInfo) const;
    bool CheckRopeDimsD(const FiaArch38ContextParams& contextKeyParams, const std::string& rName,
                        const PFAShapeDims& dims) const;
    bool SetShape(FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* shape,
                  const std::string inputName, PFAShapeDims& dims) const;
    bool SetShapePA(const FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* shape,
                    PFAShapeDims& dims) const;
    bool SetShapeBNSD(const gert::StorageShape* shape, PFAShapeDims& dims) const;
    bool SetShapeBSH(const FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* shape,
                     const std::string& inputName, PFAShapeDims& dims) const;
    bool SetShapeBSND(const gert::StorageShape* shape, PFAShapeDims& dims) const;
    bool SetShapeTND(const FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* shape,
                     const std::string& inputName, PFAShapeDims& dims) const;
    bool CheckShapeSizeLimits(const FiaArch38ContextParams& contextKeyParams, const std::string& sName,
                              const PFAShapeDims& dims) const;
    bool CheckShapeDAndNLimits(const FiaArch38ContextParams& contextKeyParams, const std::string& sName,
                               const PFAShapeDims& dims) const;
    bool CheckRopeShapeConsistency(const FiaArch38ContextParams& contextKeyParams, const std::string& sName,
                                   const std::string& rName, const PFAShapeInfo& shapeInfo,
                                   const PFAShapeDims& dims) const;
    bool CheckQueryOutParamsConsistency(const FiaArch38ContextParams& contextKeyParams,
                                        const gert::StorageShape* queryShape, const gert::StorageShape* outShape) const;
    bool CheckKVDataType(FiaArch38ContextParams& contextKeyParams) const;
    bool CheckKeyValueParamsConsistency(FiaArch38ContextParams& contextKeyParams, const gert::StorageShape* keyShape,
                                        const gert::StorageShape* valueShape) const;
    bool SetAndCheckHeadNumRatio(FiaArch38ContextParams& contextKeyParams,
                                 FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckInputDimAndHeadNum(FiaArch38ContextParams& contextKeyParams, const uint32_t nQAttr,
                                 const uint32_t nKVAttr);
    bool ReadQkvShapeHeadNum(const FiaArch38ContextParams& contextKeyParams, uint32_t& queryShapeHeadNum,
                             uint32_t& keyShapeHeadNum, uint32_t& valueShapeHeadNum);
    bool CheckPostQuantDtype(const FiaArch38ContextParams& contextKeyParams,
                             const gert::StorageShape* quantOffset2Shape, const ge::DataType quantScale2Type) const;
    bool CheckPostQuantShapeSize(const FiaArch38ContextParams& contextKeyParams,
                                 const gert::StorageShape* quantOffset2Shape, int64_t quantScale2ShapeSize,
                                 const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& valueShapeInfo) const;
    bool CheckPostQuantParams(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo,
                              const PFAShapeInfo& valueShapeInfo) const;
    bool CheckPerTensorQuantParams(const FiaArch38ContextParams& contextKeyParams) const;
    bool CheckPerblockQuantParams(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo,
                                  const PFAShapeInfo& keyShapeInfo, const PFAShapeInfo& valueShapeInfo) const;
    bool CheckPerblockQuantDtype(const FiaArch38ContextParams& contextKeyParams) const;
    bool CheckPerblockScaleShape(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& shapeInfo,
                                 const PerblockScaleCheckCtx& ctx) const;
    bool CheckAntiquantParamsShape(FiaArch38ContextParams& contextKeyParams) const;
    bool GetAndCheckPrefixShape(FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo,
                                PFAShapeInfo& prefixShapeInfo, FiaArch38FlashAttentionTilingData& tilingData) const;
    bool CheckKeyValuePrefixConsistency(FiaArch38ContextParams& contextKeyParams,
                                        const gert::StorageShape* keyShape) const;
    bool CheckActSharedPrefix(FiaArch38ContextParams& contextKeyParams, const uint32_t sPrefix, const uint32_t sKV);
    bool CheckPAKeyValueDimConsistency(const FiaArch38ContextParams& contextKeyParams, const PACacheDimCtx& ctx) const;
    bool CheckPAKeyValueLayout(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo,
                               const PACacheDimCtx& ctx) const;
    bool CheckPACacheShape3D(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& shapeInfo,
                             const gert::StorageShape* shape, const PACacheDimCtx& ctx, const std::string& sName);
    bool CheckPACacheShape4D(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& shapeInfo,
                             const gert::StorageShape* shape, const PACacheDimCtx& ctx, const std::string& sName);
    bool CheckPACacheShape(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& shapeInfo,
                           const gert::StorageShape* shape, const PACacheDimCtx& ctx, const std::string& sName);
    PABlockNumInfo ComputePABlockNumInfo(const FiaArch38ContextParams& contextKeyParams,
                                         const PFAShapeInfo& queryShapeInfo) const;
    bool CheckBlockTableDims(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo,
                             int32_t maxBlockNumPerBatch) const;
    bool CheckBlockTableShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                              PFAShapeInfo& queryRopeShapeInfo, FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckPARopeShape(const FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryRopeShapeInfo,
                          const PACacheDimCtx& ctx);
    bool CheckMaskShape(FiaArch38ContextParams& contextKeyParams, MaskShapeInfo& info, const uint32_t sQ,
                        const uint32_t sK, const uint32_t batchSize) const;
    void SetSparseModeData(FiaArch38ContextParams& contextKeyParams, FiaArch38FlashAttentionTilingData& tilingData);
    void SetAttenMaskBatch(const gert::StorageShape* attenMaskShape, FiaArch38FlashAttentionTilingData& tilingData);
    void ClampSparseTokens(const int64_t* preTokens, const int64_t* nextTokens);
    bool CheckMaskShapeCrossSparse(FiaArch38ContextParams& contextKeyParams, const int64_t* sparseMode, uint64_t sQ,
                                   const uint64_t sK, const uint32_t batchSize);
    bool CheckMaskCrossIFAMLA(FiaArch38ContextParams& contextKeyParams, const int64_t* sparseMode, uint64_t queryS);
    bool CheckIO(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& valueShapeInfo);
    bool CheckKV(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& keyShapeInfo,
                 PFAShapeInfo& valueShapeInfo) const;
    bool CheckQueryAndKey(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                          PFAShapeInfo& keyShapeInfo, FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckPFAMerge(FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const;
    bool CheckRope(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo,
                   PFAShapeInfo& queryRopeShapeInfo);
    bool CheckIFAMLA(FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const;
    bool CheckQuant(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo,
                    const PFAShapeInfo& valueShapeInfo) const;
    bool CheckPrefix(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo, PFAShapeInfo& keyShapeInfo,
                     FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckActSeq(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo) const;
    bool CheckActSeqMonotonic(const FiaArch38ContextParams& contextKeyParams, const gert::Tensor* actSeqLen,
                              const gert::Tensor* actSeqLenKV, uint32_t batchSize) const;
    bool CheckActSeqLen(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                        const PFAShapeInfo& keyShapeInfo);
    bool CheckActSeqLenQ(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo);
    bool CheckActSeqLenKV(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                          const PFAShapeInfo& keyShapeInfo);
    bool CheckPATypeAndShape(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                             PFAShapeInfo& queryRopeShapeInfo, FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckPAType(const FiaArch38ContextParams& contextKeyParams);
    bool CheckPABlockSize(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo);
    bool CheckPseShiftTypeAndShape(FiaArch38ContextParams& contextKeyParams, uint32_t b, uint32_t n, uint32_t s1,
                                   uint32_t s2);
    bool CheckPseShiftDtype(const FiaArch38ContextParams& contextKeyParams);
    bool CheckPseShiftShapeDim(const FiaArch38ContextParams& contextKeyParams, uint32_t b, uint32_t n, uint32_t s1,
                               uint32_t s2);
    bool CheckInnerPrecise(FiaArch38ContextParams& contextKeyParams, FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckMaskTypeAndShape(FiaArch38ContextParams& contextKeyParams, FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckMaskDtype(const FiaArch38ContextParams& contextKeyParams);
    void SetSparseType(uint32_t qS);
    bool CheckSparseMode(FiaArch38ContextParams& contextKeyParams, uint32_t qS,
                         FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckSparseTokensRange(const FiaArch38ContextParams& contextKeyParams, uint32_t qS) const;
    bool CheckPACrossover(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo);
    bool CheckMaskCrossover(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                            FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckTNDLayoutCrossover(FiaArch38ContextParams& contextKeyParams);
    bool ParseActualSeqLengths(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                               std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV);
    bool ParseActualSeqLengthsQueryNonTnd(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                          std::vector<int64_t>& actualSeqLengths, uint32_t i);
    void ParseActualSeqLengthsKvNonTnd(FiaArch38ContextParams& contextKeyParams,
                                       std::vector<int64_t>& actualSeqLengthsKV, uint32_t i);
    bool CheckMultiFeatureCrossover(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                    std::vector<int64_t>& actualSeqLengths, std::vector<int64_t>& actualSeqLengthsKV,
                                    FiaArch38FlashAttentionTilingData& tilingData);
    bool CheckMultiFeatureCrossoverPerBatch(FiaArch38ContextParams& contextKeyParams,
                                            std::vector<int64_t>& actualSeqLengths,
                                            std::vector<int64_t>& actualSeqLengthsKV,
                                            FiaArch38FlashAttentionTilingData& tilingData, uint32_t i);
    bool CheckBandModeTokens(const FiaArch38ContextParams& contextKeyParams,
                             const std::vector<int64_t>& actualSeqLengths,
                             const std::vector<int64_t>& actualSeqLengthsKV, uint32_t i) const;
    void CalcPerbatchTokens(const std::vector<int64_t>& actualSeqLengths,
                            const std::vector<int64_t>& actualSeqLengthsKV, uint32_t i, int64_t& preTokensPerbatch,
                            int64_t& nextTokensPerbatch);
    bool CheckPerblockCrossover(FiaArch38ContextParams& contextKeyParams);
    bool CheckPerblockQuantCrossoverBase(const FiaArch38ContextParams& contextKeyParams) const;
    bool CheckPerblockQuantCrossoverFeature(const FiaArch38ContextParams& contextKeyParams) const;
    void SetTilingDataAttribute(const FiaArch38ContextParams& contextKeyParams,
                                FiaArch38FlashAttentionTilingData& tilingData);
    void GetEnableDN(FiaArch38FlashAttentionTilingData& tilingData, const PFAShapeInfo& queryShapeInfo,
                     const PFAShapeInfo& valueShapeInfo, std::vector<int64_t>& actualSeqLengths,
                     std::vector<int64_t>& actualSeqLengthsKV);
    void SetTilingData(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo,
                       const PFAShapeInfo& queryRopeShapeInfo, const PFAShapeInfo& valueShapeInfo,
                       FiaArch38FlashAttentionTilingData& tilingData);
    void SetTilingDataTypeByteNum(FiaArch38FlashAttentionTilingData& tilingData);
    void SetTilingDataQuantParams(const FiaArch38ContextParams& contextKeyParams,
                                  FiaArch38FlashAttentionTilingData& tilingData);
    void SetTilingDataShapeParams(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo,
                                  const PFAShapeInfo& queryRopeShapeInfo, const PFAShapeInfo& valueShapeInfo,
                                  FiaArch38FlashAttentionTilingData& tilingData);
    void InferTilingMod(const FiaArch38ContextParams& contextKeyParams, std::vector<int64_t>& actualSeqLengths,
                        std::vector<int64_t>& actualSeqLengthsKV, uint32_t actualSeqArrayLen, uint32_t d);
    int64_t GetSInnerBlockNums(int64_t sInnerIndexStart, int64_t sInnerIndexEnd, int64_t innerBlockNums) const;
    int64_t GetCutBlockNums(int64_t blockSeqLengthKV, int64_t blockSeqLength, int64_t sInner, int64_t sOuter,
                            int64_t token) const;
    void FixParamWithRowInvalid(int64_t& actualSeqLength, int64_t actualSeqLengthKV, int64_t& preTokensLeftUp,
                                int64_t& nextTokensLeftUp) const;
    int64_t GetCalcBlockNumsOneHead(int64_t actualSeqLength, int64_t actualSeqLengthKV, uint32_t sOuterSize,
                                    uint32_t sInnerSize, int64_t preTokensLeftUp, int64_t nextTokensLeftUp,
                                    bool isAttenMaskUsed) const;
    void ComputeSplitNBSeq(FiaArch38FlashAttentionTilingData& tilingData, uint32_t batchSize,
                           const size_t tilingElementArrayLen, SplitNBSeqCtx& ctx);
    void SplitOneHeadBlock(FiaArch38FlashAttentionTilingData& tilingData, uint32_t sIdx, uint32_t headNum,
                           uint32_t headNumSize, SplitNBSeqCtx& ctx) const;
    void FiaArch38SplitNBSeq(FiaArch38FlashAttentionTilingData& tilingData, std::vector<int64_t>& actualSeqLengths,
                             std::vector<int64_t>& actualSeqLengthsKV, bool isAttenMaskUsed);
    void InferSplitCoreMode();
    void InferConstantization();
    bool AdjustCVTilingCVDiff(const FiaArch38ContextParams& contextKeyParams, CVFactorParams& cvFactors,
                              FiaArch38FlashAttentionTilingData& tilingData, const PFAShapeInfo& queryShapeInfo);
    bool AdjustCVFactorForSmallS(const FiaArch38ContextParams& contextKeyParams, const PFAShapeInfo& queryShapeInfo,
                                 FiaArch38FlashAttentionTilingData& tilingData, CVFactorParams& cvFactors) const;
    bool AdjustCVFactorForLargeD(CVFactorParams& cvFactors) const;
    bool AdjustCVFactorForIFAMLA(CVFactorParams& cvFactors) const;
    void GetMatMulType(matmul_tiling::DataType& mmInputType, matmul_tiling::DataType& mmOutputType) const;
    bool EnableMTE2BmmPipe(FiaArch38FlashAttentionTilingData& tilingData, matmul_tiling::MatmulApiTiling& bmm,
                           TCubeTiling& bmmTilingData, uint32_t sOuterFactor, uint32_t sInnerFactor) const;
    void EnableBmmDoubleBuffer(TCubeTiling& bmmTilingData) const;
    bool FiaArch38CheckBmm1(FiaArch38FlashAttentionTilingData& tilingData, TCubeTiling& bmm1TilingData,
                            BmmShapeCtx& bmmCtx, bool autoBaseMNK = false);
    void SetBmm1OrgShape(matmul_tiling::MatmulApiTiling& bmm1, FiaArch38FlashAttentionTilingData& tilingData,
                         int32_t strideQ, int32_t strideK) const;
    bool Bmm1SetTiling(matmul_tiling::MatmulApiTiling& bmm1, TCubeTiling& bmm1TilingData, const BmmShapeCtx& bmmCtx,
                       bool autoBaseMNK) const;
    bool FiaArch38CheckBmm2(FiaArch38FlashAttentionTilingData& tilingData, TCubeTiling& bmm2TilingData,
                            BmmShapeCtx& bmmCtx);
    void SetBmm2OrgShape(matmul_tiling::MatmulApiTiling& bmm2, FiaArch38FlashAttentionTilingData& tilingData,
                         int32_t strideV) const;
    bool Bmm2GetTiling(matmul_tiling::MatmulApiTiling& bmm2, TCubeTiling& bmm2TilingData,
                       FiaArch38FlashAttentionTilingData& tilingData, const BmmShapeCtx& bmmCtx) const;
    bool FiaArch38ComputeCVDiffParams(FiaArch38FlashAttentionTilingData& tilingData, int64_t l1Size, int64_t l0CSize,
                                      uint32_t& sOuterFactor, uint32_t& sInnerFactor);
    void GetPreNextTokensLeftUp(FiaArch38FlashAttentionTilingData& tilingData, int64_t actualSeqLength,
                                int64_t actualSeqLengthKV, int64_t& preTokensLeftUp, int64_t& nextTokensLeftUp) const;
    void UpdateTilingKeyMatmulCfg(uint64_t& tilingKey) const;
    void UpdateTilingKeyMaskCfg(FiaArch38FlashAttentionTilingData& tilingData, uint64_t& tilingKey) const;
    void UpdateTilingKeyPseCfg(uint64_t& tilingKey) const;
    void UpdateTilingKeyDSizeConst(FiaArch38FlashAttentionTilingData& tilingData, uint64_t& tilingKey) const;
    void UpdateTilingKeyValueDSizeConst(FiaArch38FlashAttentionTilingData& tilingData, uint64_t& tilingKey) const;
    void UpdateTilingKeySInnerConst(FiaArch38FlashAttentionTilingData& tilingData, uint64_t& tilingKey) const;
    void UpdateTilingKeySOuterConst(FiaArch38FlashAttentionTilingData& tilingData, uint64_t& tilingKey);
    void FiaArch38InitSoftmaxLseOutputSplit(int64_t totalSize, FiaArch38FlashAttentionTilingData& tilingData) const;
    void UpdateTilingKeyFlag(const FiaArch38ContextParams& contextKeyParams, uint64_t& tilingKey) const;
    bool TilingGetTilingKeyAttentionAscendC(uint64_t& tilingKey, FiaArch38ContextParams& contextKeyParams,
                                            FiaArch38FlashAttentionTilingData& tilingData);
    void AddTilingKeyDtypeAndLayout(const FiaArch38ContextParams& contextKeyParams, uint64_t& tilingKey) const;
    void AddTilingKeyFeatureFlag(uint64_t& tilingKey) const;
    size_t GetPFAWorkSpaceSize(FiaArch38FlashAttentionTilingData& tilingData);
    uint32_t ComputeDvBasicBlock(uint32_t dSize) const;
    ge::graphStatus SetPlatMemoryInfo(FiaArch38ContextParams& contextKeyParams);
    ge::graphStatus SetAttributeInfo(FiaArch38ContextParams& contextKeyParams);
    ge::graphStatus SetAntiquantAndQuantInfo(FiaArch38ContextParams& contextKeyParams);
    ge::graphStatus SetPrefixAndSeqInfo(FiaArch38ContextParams& contextKeyParams);
    ge::graphStatus SetPseAndMaskInfo(FiaArch38ContextParams& contextKeyParams);
    ge::graphStatus CheckTensorInvalid(const FiaArch38ContextParams& contextKeyParams) const;
    ge::graphStatus CheckRopeInvalid(const FiaArch38ContextParams& contextKeyParams) const;
    ge::graphStatus CheckSingleAttribute(FiaArch38ContextParams& contextKeyParams, PFAShapeSet& shapes,
                                         FiaArch38FlashAttentionTilingData& tilingData);
    ge::graphStatus CheckShapeStage(FiaArch38ContextParams& contextKeyParams, PFAShapeSet& shapes,
                                    FiaArch38FlashAttentionTilingData& tilingData);
    ge::graphStatus CheckFeatureStage(FiaArch38ContextParams& contextKeyParams, PFAShapeSet& shapes,
                                      FiaArch38FlashAttentionTilingData& tilingData);
    ge::graphStatus CheckBaseFeatureStage(FiaArch38ContextParams& contextKeyParams, PFAShapeSet& shapes,
                                          FiaArch38FlashAttentionTilingData& tilingData);
    ge::graphStatus CheckPAStage(FiaArch38ContextParams& contextKeyParams, PFAShapeSet& shapes,
                                 FiaArch38FlashAttentionTilingData& tilingData);
    ge::graphStatus CheckPseShiftStage(FiaArch38ContextParams& contextKeyParams, PFAShapeSet& shapes);
    ge::graphStatus CheckCrossoverAttribute(FiaArch38ContextParams& contextKeyParams, PFAShapeInfo& queryShapeInfo,
                                            std::vector<int64_t>& actualSeqLengths,
                                            std::vector<int64_t>& actualSeqLengthsKV,
                                            FiaArch38FlashAttentionTilingData& tilingData);
    ge::graphStatus AdjustTilingData(FiaArch38ContextParams& contextKeyParams,
                                     FiaArch38FlashAttentionTilingData& tilingData, const PFAShapeInfo& queryShapeInfo);
    ge::graphStatus CheckFp16TilingKeySupport(const FiaArch38ContextParams& contextKeyParams, uint32_t sOuterFactor,
                                              const PFAShapeInfo& queryShapeInfo) const;
    ge::graphStatus ComputeTilingData(FiaArch38ContextParams& contextKeyParams, std::vector<int64_t>& actualSeqLengths,
                                      std::vector<int64_t>& actualSeqLengthsKV,
                                      FiaArch38FlashAttentionTilingData& tilingData);
    ge::graphStatus ComputeTilingKey(uint64_t& tilingKey, FiaArch38ContextParams& contextKeyParams,
                                     uint32_t& numBlocksToBeSet, FiaArch38FlashAttentionTilingData& tilingData);
    void SetAttenMaskCompressMode();
    void SetLayoutType();
    void PFATilingDataconvert(FiaArch38FlashAttentionTilingData& tilingData);
    void SetInputParamsShape(FiaArch38FlashAttentionTilingData& tilingData);
    void SetInputParamsPseAndMask(FiaArch38FlashAttentionTilingData& tilingData);
    void SetInputParamsPA(FiaArch38FlashAttentionTilingData& tilingData);
    void SetInitOutputParams(FiaArch38FlashAttentionTilingData& tilingData);
    void SetMultiCoreParamsRegbase(int64_t totalSize, int64_t actualUsedCoreNum);
    bool IsFlashDecode() const;
    ge::graphStatus SplitBNS(FiaArch38FlashAttentionTilingData& tilingData, uint64_t bng);
    bool CheckAlibiPseShiftTypeAndShape(FiaArch38ContextParams& contextKeyParams, uint32_t n);
    ge::graphStatus SetQKVStartIdx(FiaArch38ContextParams& contextKeyParams);
    bool CheckAlibiPseCrossover(FiaArch38ContextParams& contextKeyParams);

    bool CheckArch38PlatformSupported(const FiaArch38ContextParams& contextKeyParams,
                                      const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& valueShapeInfo) const;
    bool CheckArch38ScenarioSupported(const FiaArch38ContextParams& contextKeyParams,
                                      const PFAShapeInfo& queryShapeInfo, const PFAShapeInfo& valueShapeInfo) const;
    bool CheckArch38UnsupportedFeatures(const FiaArch38ContextParams& contextKeyParams) const;
    bool CheckArch38ShapeConstraints(const FiaArch38ContextParams& contextKeyParams,
                                     const PFAShapeInfo& queryShapeInfo) const;
    bool CheckArch38HeadAndSparse(const FiaArch38ContextParams& contextKeyParams) const;
    bool CheckArch38MaskConstraints(const FiaArch38ContextParams& contextKeyParams) const;

protected:
    gert::TilingContext* context_ = nullptr;
    FiaArch38ContextParams* contextKeyParamsPtr = nullptr;
    int64_t ubSizeRemain = 1;
    bool enableFlashDecode = false;
    bool isSOuterNoTail = true;
    bool isSInnerNoTail = true;
    bool isDNoTail = true;
    bool enableTensorList = false;
    bool enableLeftPadding = false;
    bool enableActSeqLen = false;
    bool enableActSeqLenKV = false;
    bool enableKVAntiquant = false;
    bool enablePseShift = false;
    bool enableAlibiPse = false;
    bool enableMask = false;
    bool enableQuantBF16 = false;
    bool enableMatmulNorm = false;
    bool enablePA = false;
    bool enableSplitSeqOneN = false;
    bool isDefaultSparseMode = false;
    bool isKVHasPrefix = false;
    bool isBandMode = false;
    bool enableIFAMLA = false;
    bool enableIFA = false;
    // MLPerf合轴优化
    bool enablePFAMerge = false;
    uint32_t pfaMergeGLimit = 16;
    uint32_t pfaMergeQsLimit = 4;
    bool enablePFAMLA = false;
    bool enablePFARope = false;
    bool enableDN = false;
    bool enablePostQuant = false;
    bool enablePertensorQuant = false;
    bool enablePerblockQuant = false;
    uint64_t gSize = 1;
    InputLayout inputLayout = InputLayout::BSH;
    ge::DataType inputType{ge::DT_FLOAT16};
    ge::DataType outputType{ge::DT_FLOAT16};
    ge::DataType pseShiftElemType{ge::DT_FLOAT16};
    ge::DataType queryType{ge::DT_FLOAT};
    ge::DataType keyType{ge::DT_FLOAT};
    ge::DataType valueType{ge::DT_FLOAT};
    uint32_t dataTypeSize = FLOAT32_SIZE;
    uint32_t outputDataTypeSize = FLOAT32_SIZE;
    uint32_t maskElemSize = FLOAT32_SIZE;
    int32_t ifaBlockSizeBase = 32;
    uint32_t coreNum = 0;
    uint32_t aivNum = 0;
    uint32_t aicNum = 0;
    uint32_t typeByteNum = 0;
    uint32_t outputTypeByteNum = 0;
    uint32_t softmaxTypeByteNum = 0;
    uint32_t pseShiftTypeByteNum = 0;
    uint32_t pseShiftElemSize = 0;
    uint32_t pseMaskMaxSize = 0;
    int64_t pseShiftBatch = 0;
    int64_t pseShiftS1 = 0;
    int64_t pseShiftS2 = 0;
    int64_t actSeqLenDims = 0;
    int64_t actSeqLenKVDims = 0;
    int64_t middleActualSeqLengths = 0;
    int64_t actualSharedPrefixLen = 0;
    uint32_t needInit = 0U;
    uint32_t usePseShift = 0;
    // There is no S2 axis for PA. Use the change amount to normalize the S2 length in both PA and non PA scenarios
    uint32_t S2 = 0;
    int32_t blockTableDim2 = 1;
    int32_t paBlockNumSum = 1;
    uint32_t maskTypeByteNum = 0;
    uint32_t softmaxDataTypeSize = FLOAT32_SIZE; // BF16 calculates through FP32
    platform_ascendc::SocVersion curShortSocName;
    uint32_t layoutType = 0;
    uint32_t paLayoutType = 0;
    int64_t sparsePreTokens = 0;
    int64_t sparseNextTokens = 0;
    int32_t sparseModeVal = 0;
    int64_t maxActualseqKV = 0;
    SplitCoreMode splitCoreMode = SplitCoreMode::SPLIT_NBS_VECTOR;
    uint32_t cvRatio = 2U; // Vector : Cube
    bool isConstantization = false;
    uint32_t splitS2 = 1; // It can only be 0 when the D axis is split
    int32_t innerPrecise = HIGH_PERFORMANCE;
    uint32_t sOuterFactorTiling = 0;
    uint32_t softmaxSInnerFactorTiling = 0;
    uint32_t softmaxSOuterFactorTiling = 0;
    size_t defaultSysWorkspaceSize = 0;
    matmul_tiling::PlatformInfo ascendPlatformInfo;

    uint8_t attenMaskShapeType = 0; // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
    uint8_t sparseType = 0;
    int64_t pseType = 0;
    FlashAttentionScoreSimplifiedTilingData faTilingAdapter;
};
} // namespace arch38
} // namespace optiling

#endif // FIA_ARCH38_PROMPT_TILING_H_
