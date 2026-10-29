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
 * \file flash_attention_score_tiling_general.cpp
 * \brief
 */

#include <string>
#include <numeric>
#include "flash_attention_tiling.h"

namespace optiling {
namespace FA {

const size_t ACTUAL_SEQ_LENGTH_ATTRS_INDEX = 7UL;
const size_t ACTUAL_SEQ_LENGTH_KV_ATTRS_INDEX = 8UL;
const size_t ATTENTION_MASK_INPUT_INDEX = 5UL;
const size_t PRM_MASK_INPUT_INDEX = 6UL;
const size_t SINK_INPUT_INDEX = 7UL;
const size_t ATTEN_OUT_INDEX = 3UL;

const uint32_t BYTE_BLOCK = 32;
const uint32_t GM_ALIGN = 512;
const int64_t FRACTAL_NUM = 16L;
const int64_t PSE_DIM_NUM = 4L;
const int64_t S1_VEC2_BASE_SIZE_MAX = 512L;

const int64_t BYTE_BIT_NUM = 8UL;
const size_t ATTENTION_MASK_DIM_NUM = 4UL;
const int64_t BMM_SOFTMAX_RATIO = 4L;
const size_t DATA_COPY_TRANPOSE_DIM_NUM = 4L;
const int64_t MAX_AIV_NUM = 48L;
const int64_t DROP_MASK_ALIGN_UNIT = 256L; // input bits, and align to 32B in UB
const int64_t HIGH_PERF_BUFFER_NUM = 6L;
const int64_t HIGH_PERF_SUPPORT_D_LIMIT = 128L;
const int64_t HIGH_PERF_SUPPORT_S2_BASIC = 128L;
const int64_t HIGH_PERF_API_BUFFER_MULTIPLE = 2L;
const int64_t HIGH_PERF_WORKSPACE_MULTIPLE = 3L;
const int64_t HIGH_PERF_BLOCK_SIZE = 128L;
const uint32_t PSE_ALIBI_S_SIZE = 1024;

constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;
const int64_t MAX_PREFIX_N_SIZE = 24L;
const int64_t ATTEN_MASK_S1_REV_INDEX = 2L;
const int64_t ATTEN_MASK_COMPRESS_LIMIT = 2048L;
const int64_t MAX_VAR_LEN_SEQ_LEN = 65536L;
const int64_t ALIGN_NUM_16 = 16L;
const int64_t S2_REUSE_SIZE_512 = 512L;
const int64_t S2_REUSE_SIZE_1024 = 1024L;
const int64_t S1_REUSE_SIZE_3840 = 3840L;
const int64_t S1_REUSE_SIZE_2048 = 2048L;
const int64_t D_SPECIFIC_SIZE = 64L;
const int64_t BMM1_BASICBLOCK_M_128 = 128L;
const int64_t BMM1_BASICBLOCK_N_256 = 256L;
const int64_t BMM1_BASICBLOCK_K_64 = 64L;
const int64_t S2_NZTOND_SIZE_64 = 64L;
const int64_t SPACE_NUM_2 = 2L;
const int64_t SPACE_NUM_3 = 3L;
const int64_t SPACE_NUM_4 = 4L;
const int64_t BMM2_BASICBLOCK_M_64 = 64L;
const int64_t BMM2_BASICBLOCK_N_64 = 64L;
const int64_t BMM2_BASICBLOCK_K_256 = 256L;
const int64_t S2_SPECIFIC_SIZE_928 = 928L;
const int64_t S2_NZTOND_SIZE_128 = 128L;
enum LayoutType : uint8_t {
    None = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

enum AttenMaskShapeType : uint8_t {
    ATTEN_B_N2_G_S1_S2 = 0,
    ATTEN_B_1_1_S1_S2 = 1,
    ATTEN_1_1_1_S1_S2 = 2,
    ATTEN_1D_FLATTEN_S1_S2 = 3,
    ATTEN_1_1_1_T_T = 99,
};

enum SparseMode : uint8_t {
    NO_MASK = 0,
    ALL_MASK,
    LEFT_UP_CAUSAL,
    RIGHT_DOWN_CAUSAL,
    BAND,
    GLOBAL,
    DILATED,
    BLOCK_LOCAL
};

enum AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_MODE
};

class TilingKey {
public:
    TilingKey()
        : splitS1(0),
          splitS2(0),
          splitD(0),
          dtype(0),
          layoutType(0),
          sparseType(0),
          reserved(0)
    {}

    void Reset()
    {
        splitS1 = 0;
        splitS2 = 0;
        splitD = 0;
        dtype = 0;
        layoutType = 0;
        sparseType = 0;
        reserved = 0;
    }

    uint32_t GetRawTilingKey() const
    {
        return *(reinterpret_cast<const uint32_t*>(this));
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << " splitS1: " << splitS1 << " splitS2: " << splitS2 << " splitD: " << splitD;
        return ss.str();
    }

    uint32_t splitS1 : 1;
    uint32_t splitS2 : 1;
    uint32_t splitD : 1;
    uint32_t dtype : 2;
    uint32_t layoutType : 2;
    uint32_t sparseType : 2;
    uint32_t reserved : 23; // to fullfil 32 bit, if add new template bit then decrease this number
};

inline bool operator==(const TilingKey& left, const TilingKey& right)
{
    return left.GetRawTilingKey() == right.GetRawTilingKey();
}

using TemplateType = TilingKey;

class BufferNum {
public:
    // sum and max always use fp32, shape is (S1, 1), inner axis align 32B.
    size_t bufferS1S2Num; // unit: input dtype
    size_t bufferS1DNum;
    size_t bufferExpNum; // unit: input dtype, shape: [S1, 1], inner axis align 32B.
};

class FlashAttentionScoreTilingBase : public TilingBaseClass {
public:
    explicit FlashAttentionScoreTilingBase(gert::TilingContext* context)
        : TilingBaseClass(context)
    {
        Reset();
    }
    ~FlashAttentionScoreTilingBase() override = default;

    void Reset(gert::TilingContext* context) override
    {
        TilingBaseClass::Reset(context);
        Reset();
    }

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;
    virtual bool GetSparseInfo(SparseEnum& sparseType);
    bool SparseModeProcess(SparseEnum& sparseType);
    void SetSparseTilingInfo(SparseEnum& sparseType);
    bool SparseBandModeCheck(int64_t maxSVal, int64_t minSVal, SparseEnum& sparseType);
    bool PretokenAndNexttokenAdjustment(SparseEnum& sparseType);

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override = 0;
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

    virtual void GetBufferNum(BufferNum& bufferNum) const = 0;

    void Reset();

    void GetActualSeqLenData(int64_t inputIdx, std::vector<int64_t>& res, int64_t& actualLen) const;

    virtual int64_t GetNRatio();

    virtual int64_t GetMinS1BasicBlock() const
    {
        return std::min(64L, alignedS1);
    }

    virtual bool IsTemplateMatched() const
    {
        return expectTemplate == actualTemplate;
    }

    ge::graphStatus CheckContext();
    virtual bool AnalyzeDtype();
    bool AnalyzeAttrs();
    bool AnalyzeLayout();
    bool Analyze3DimLayout(const gert::Shape& queryShape, const gert::Shape& keyShape, size_t layoutLen);
    bool Analyze4DimLayout(const gert::Shape& queryShape, const gert::Shape& keyShape, size_t layoutLen);
    bool AnalyzeOptionalInput();
    bool MatchTemplate();
    virtual void CalcS1S2BasicBlock(const BufferNum& bufferNum);
    virtual void CalcDBasicBlock(const BufferNum& bufferNum);
    int64_t CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum& bufferNum) const;
    int64_t CalcMaxS2BasicBlockSize(const BufferNum& bufferNum, int64_t tmpS1BasicBlock) const;
    virtual bool CalcUBSize(const BufferNum& bufferNum, int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                            int64_t tmpDBasicBlock, int64_t batch = 1);
    virtual int64_t CalcMinApiTmpSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t batch);
    bool IsBasicBlockInSoftMax(const ge::Shape& shape) const;
    virtual bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock,
                                    int64_t batch, matmul_tiling::MatmulApiTiling& bmm1);
    virtual bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock,
                                    int64_t batch, matmul_tiling::MatmulApiTiling& bmm2) = 0;
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                         matmul_tiling::MatmulApiTiling& bmm1, matmul_tiling::MatmulApiTiling& bmm2);
    bool SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch = 1);
    virtual void SetCoreParams();
    virtual void SetMultiBatchCoreParams();
    virtual void SetMultiCoreParams();
    virtual void SetSoftMaxTiling();
    void SetDataCopyTransposeTiling();
    virtual void SetTensorSizeParams();
    virtual void SetSparseParams();
    virtual bool InitSparseValidArray(std::vector<int64_t>& sparseValidArray, int64_t bIdx);
    virtual bool SetSparseStartIdx(const std::vector<int64_t>& sparseValidArray, MultiCoreParams& multiCoreParams);
    void SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>>& sparseValidArray,
                                 MultiCoreParams& multiCoreParams);
    void PrintSparseMaxMinLoadPerCore(const std::vector<int64_t>& sparseValidArray, int64_t* sparseStartIdx,
                                      int32_t validAivNum, int64_t avgLoadSize);
    bool PartitionSparseData(const std::vector<int64_t>& sparseRollingArray, int64_t sparseRollingArraySum,
                             int64_t sparseArraySize, int64_t loadMaxEachCore, std::vector<int64_t>& partitionResult);

    uint32_t aivNum;
    uint32_t aicNum;
    int64_t apiMaxUBSize = 0;

    matmul_tiling::DataType bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    matmul_tiling::DataType bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    ge::DataType inputDtype;
    int64_t inputDtypeBytes;
    int64_t calcTypeSize;

    bool isHighPercision; // fp16 high percision mode

    DtypeEnum tilingKeyDType;
    LayoutEnum tilingKeyLayout;
    CubeFormatEnum tilingKeyBmm1Format = CubeFormatEnum::ND;
    CubeInputSourceEnum tilingKeyBmm1Source = CubeInputSourceEnum::GM;
    CubeInputSourceEnum tilingKeyBmm2Source = CubeInputSourceEnum::GM;

    int64_t bSize;
    int64_t gSize;
    int64_t dSize;
    int64_t vDsize;
    int64_t n1Size;
    int64_t n2Size;
    int64_t s1Size;
    int64_t s2Size;
    int64_t s1StrideSize;      // query Shape S inner axes, for bmm1
    int64_t s2StrideSize;      // key Shape S inner axes, for bmm1
    int64_t valueS2strideSize; // value Shape S inner axes, for bmm2
    int64_t preTokens;
    int64_t nextTokens;
    int64_t sparseMode;
    int64_t outputMaxLogits;
    int64_t maxS1Val;
    int64_t minS1Val;
    int64_t accumS1;
    int64_t accumS1BlockNum;
    int64_t dropTotalSize;
    int64_t maxS2Val;
    int64_t minS2Val;
    int64_t accumS2;
    int64_t accumTT;
    std::vector<int64_t> actualSeqLenData;
    std::vector<int64_t> actualSeqLenKvData;
    float scaleValue;
    uint8_t attenMaskCompressMode;
    uint8_t attenMaskExistFlag;
    uint8_t prmMaskExistFlag;
    uint8_t sinkExistFlag;

    int64_t alignedS1;
    int64_t alignedS2;
    int64_t alignedD;

    int64_t s1BasicBlock;
    int64_t s2BasicBlock;
    int64_t dBasicBlock;
    int64_t batchBasic;
    int64_t nRatio;

    int64_t minUsedUBSize;

    const char* templateName = "base";
    const char* opName;
    const char* inputLayout;
    TemplateType expectTemplate;
    TemplateType actualTemplate;

    FlashAttentionGeneralTilingData tilingData;
};

int64_t FlashAttentionScoreTilingBase::GetNRatio()
{
    return BMM_SOFTMAX_RATIO;
}

ge::graphStatus FlashAttentionScoreTilingBase::GetPlatformInfo()
{
    aicoreParams_.blockDim = 0;
    auto platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_IF(platformInfoPtr == nullptr, TILING_INNER_ERR_REPORT(opName, "platformInfoPtr is null"),
                return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    aivNum = ascendcPlatform.GetCoreNumAiv();
    aicNum = ascendcPlatform.GetCoreNumAic();
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0cSize;

    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0cSize);

    aicoreParams_.ubSize = ubSize;
    aicoreParams_.l1Size = l1Size;
    aicoreParams_.l0cSize = l0cSize;
    OP_LOGD(opName, "get platform from compileInfo. aivNum(%u) aicNum(%u) ubSize(%lu) l1Size(%lu) l0cSize(%lu)", aivNum,
            aicNum, aicoreParams_.ubSize, aicoreParams_.l1Size, aicoreParams_.l0cSize);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTilingBase::CheckContext()
{
    auto attrs = context_->GetAttrs();
    OPS_LOG_E_IF_NULL(opName, attrs, return ge::GRAPH_FAILED)
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto outputMaxLogitsPtr = attrs->GetAttrPointer<uint32_t>(idx++);

    OPS_LOG_E_IF_NULL(opName, scaleValuePtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(opName, preTokensPtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(opName, nextTokensPtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(opName, n1SizePtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(opName, inputLayoutPtr, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(opName, outputMaxLogitsPtr, return ge::GRAPH_FAILED)

    auto queryShape = context_->GetInputShape(0);
    auto queryDesc = context_->GetInputDesc(0);
    auto keyShape = context_->GetInputShape(1);
    auto attenOutShape = context_->GetOutputShape(ATTEN_OUT_INDEX);

    OPS_LOG_E_IF_NULL(opName, queryShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(opName, queryDesc, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(opName, keyShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(opName, attenOutShape, return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(opName, context_->GetRawTilingData(), return ge::GRAPH_FAILED)
    OPS_LOG_E_IF_NULL(opName, context_->GetRawTilingData()->GetData(), return ge::GRAPH_FAILED)
    OP_CHECK_IF(context_->GetRawTilingData()->GetCapacity() < tilingData.GetDataSize(),
                TILING_INNER_ERR_REPORT(opName, "context tiling data capacity %zu < actual tiling data size %zu",
                                        context_->GetRawTilingData()->GetCapacity(), tilingData.GetDataSize()),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreTilingBase::SetSparseTilingInfo(SparseEnum& sparseType)
{
    auto& inputParams = tilingData.inputParams;
    inputParams.set_attenMaskCompressMode(attenMaskCompressMode);
    inputParams.set_sparseType(static_cast<uint8_t>(sparseType));
    inputParams.set_preTokens(preTokens);
    inputParams.set_nextTokens(nextTokens);
}

bool FlashAttentionScoreTilingBase::PretokenAndNexttokenAdjustment(SparseEnum& sparseType)
{
    if (sparseMode == ALL_MASK) {
        if (preTokens < s1Size - 1 || nextTokens < s2Size - 1) {
            OP_LOGW(opName,
                    "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                    "preTokens and nextTokens will be reset max int value.",
                    preTokens, nextTokens, sparseMode);
            preTokens = std::numeric_limits<int32_t>::max();
            nextTokens = std::numeric_limits<int32_t>::max();
        }
        sparseType = static_cast<SparseEnum>(static_cast<uint8_t>(sparseMode));
    } else if (sparseMode == LEFT_UP_CAUSAL || sparseMode == RIGHT_DOWN_CAUSAL) {
        if (preTokens != s1Size || nextTokens != 0) {
            OP_LOGW(opName,
                    "preTokens[%ld] and nextTokens[%ld] not match sparseMode[%ld], "
                    "preTokens will be reset max int value and nextTokens will be reset 0.",
                    preTokens, nextTokens, sparseMode);
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
            nextTokens = 0;
        }
        sparseType = SparseEnum::CAUSAL;
    } else if (sparseMode == BAND) {
        int64_t minSVal = s1Size;
        if (tilingKeyLayout == LayoutEnum::TND) {
            minSVal = *std::min_element(actualSeqLenData.begin(), actualSeqLenData.begin() + bSize);
        }
        if (!SparseBandModeCheck(s1Size, minSVal, sparseType)) {
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::SparseBandModeCheck(int64_t maxSVal, int64_t minSVal, SparseEnum& sparseType)
{
    if (preTokens >= 0 && nextTokens >= 0) {
        if (preTokens >= maxSVal && nextTokens >= maxSVal) {
            OP_LOGE(opName,
                    "preTokens[%ld] and nextTokens[%ld] config error, should not both greater than maxSVal[%ld].",
                    preTokens, nextTokens, maxSVal);
            return false;
        }
        // preTokens = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
        // nextTokens = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
        preTokens = std::min(preTokens, s1Size);
        nextTokens = std::min(nextTokens, s2Size);
        sparseType = SparseEnum::BAND;
        return true;
    }

    if (preTokens < 0 && nextTokens < 0) {
        OP_LOGE(opName, "preTokens[%ld] and nextTokens[%ld] config error, there is no valid data block.", preTokens,
                nextTokens);
        return false;
    }

    if (preTokens < 0 && nextTokens >= 0) {
        int64_t absPreTokens = std::abs(preTokens);
        if (nextTokens >= absPreTokens && absPreTokens < minSVal) {
            // preTokens = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), 0L);
            // nextTokens = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            preTokens = std::min(preTokens, 0L);
            nextTokens = std::min(nextTokens, s2Size);
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OP_LOGE(opName,
                    "preTokens[%ld] and nextTokens[%ld] config error with S[%ld], there is no valid data block.",
                    preTokens, nextTokens, minSVal);
            return false;
        }
    }

    if (preTokens >= 0 && nextTokens < 0) {
        int64_t absNextTokens = std::abs(nextTokens);
        if (absNextTokens <= preTokens && absNextTokens < minSVal) {
            // preTokens = std::min(AlignUp(preTokens, HIGH_PERF_BLOCK_SIZE), s1Size);
            // nextTokens = std::min(AlignUp(nextTokens, HIGH_PERF_BLOCK_SIZE), 0L);
            preTokens = std::min(preTokens, s1Size);
            nextTokens = std::min(nextTokens, 0L);
            sparseType = SparseEnum::BAND;
            return true;
        } else {
            OP_LOGE(opName,
                    "preTokens[%ld] and nextTokens[%ld] config error with S[%ld], there is no valid data block.",
                    preTokens, nextTokens, minSVal);
            return false;
        }
    }
    return true;
}

bool FlashAttentionScoreTilingBase::SparseModeProcess(SparseEnum& sparseType)
{
    if (sparseMode > BAND) {
        OP_LOGE(opName, "Not support sparse mode of %ld.", sparseMode);
        return false;
    }

    if (!PretokenAndNexttokenAdjustment(sparseType)) {
        return false;
    }

    return true;
}

bool FlashAttentionScoreTilingBase::GetSparseInfo(SparseEnum& sparseType)
{
    OP_LOGD(opName, "check sparse feature: preTokens[%ld], nextTokens[%ld], s1[%ld], s2[%ld], attenMaskExistFlag[%d]",
            preTokens, nextTokens, s1Size, s2Size, attenMaskExistFlag);
    if (attenMaskExistFlag != 1) {
        return true;
    }

    if (sparseMode == NO_MASK) {
        if (preTokens >= s1Size && nextTokens == 0) {
            sparseType = SparseEnum::CAUSAL;
            preTokens = s1Size; // if sparse type is causal, template always need preTokens equal to s1Size
        } else {
            if (preTokens >= s1Size && nextTokens >= s2Size) {
                return true;
            }
            int64_t minSVal = s1Size;
            if (tilingKeyLayout == LayoutEnum::TND) {
                minSVal = *std::min_element(actualSeqLenData.begin(), actualSeqLenData.begin() + bSize);
            }
            if (!SparseBandModeCheck(s1Size, minSVal, sparseType)) {
                return false;
            }
        }
    } else {
        if (!SparseModeProcess(sparseType)) {
            return false;
        }
    }
    return true;
}

ge::graphStatus FlashAttentionScoreTilingBase::GetShapeAttrsInfo()
{
    opName = context_->GetNodeName();
    OP_LOGD(opName, "TilingContext: %s", GetTilingContextDebugStr().c_str());
    OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, TILING_INNER_ERR_REPORT(opName, "invalid context"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeLayout() || !AnalyzeOptionalInput(),
                TILING_INNER_ERR_REPORT(opName, "fail to analyze context info"), return ge::GRAPH_FAILED);
    if (vDsize == 0)
        vDsize = dSize;
    if (valueS2strideSize == 0)
        valueS2strideSize = s2StrideSize;
    alignedS1 = AlignUp(s1Size, FRACTAL_NUM);
    alignedS2 = AlignUp(s2Size, FRACTAL_NUM);
    alignedD = AlignUp(dSize, FRACTAL_NUM);
    auto& inputParams = tilingData.inputParams;
    inputParams.set_bSize(bSize);
    inputParams.set_n2Size(n2Size);
    inputParams.set_gSize(gSize);
    inputParams.set_s1Size(s1Size);
    inputParams.set_s2Size(s2Size);
    inputParams.set_accumS1(accumS1);
    inputParams.set_accumS2(accumS2);
    inputParams.set_dSize(dSize);
    inputParams.set_vDsize(vDsize);
    inputParams.set_scaleValue(scaleValue);
    inputParams.set_alignedS2(alignedS2);
    inputParams.set_outputMaxLogits(outputMaxLogits);
    OP_LOGD(opName, "input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld], vDsize[%ld], scaleValue[%f]", bSize, n2Size,
            gSize, s1Size, s2Size, dSize, vDsize, scaleValue);

    SparseEnum sparseType = SparseEnum::ALL;
    OP_CHECK_IF(!GetSparseInfo(sparseType), TILING_INNER_ERR_REPORT(opName, "fail to get sparse info"),
                return ge::GRAPH_FAILED);
    SetSparseTilingInfo(sparseType);

    return ge::GRAPH_SUCCESS;
}

void FlashAttentionScoreTilingBase::Reset()
{
    tilingData.SetDataPtr(context_->GetRawTilingData()->GetData());
    apiMaxUBSize = 0;

    bmmDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
    bmm2OutDtype = matmul_tiling::DataType::DT_FLOAT;

    inputDtype = ge::DT_FLOAT16;
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    calcTypeSize = inputDtypeBytes;

    tilingKeyDType = DtypeEnum::FLOAT16;
    tilingKeyLayout = LayoutEnum::BNSD;
    tilingKeyBmm1Format = CubeFormatEnum::ND;
    tilingKeyBmm1Source = CubeInputSourceEnum::GM;

    bSize = 0;
    gSize = 0;
    dSize = 0;
    vDsize = 0;
    n1Size = 0;
    n2Size = 0;
    s1Size = 0;
    s2Size = 0;
    maxS1Val = 0;
    minS1Val = 0;
    accumS1 = 0;
    accumS2 = 0;
    accumTT = 0;
    dropTotalSize = 0;
    maxS2Val = 0;
    minS2Val = 0;

    s1StrideSize = 0;
    s2StrideSize = 0;
    valueS2strideSize = 0;
    preTokens = std::numeric_limits<int32_t>::max();
    nextTokens = std::numeric_limits<int32_t>::max();
    sparseMode = static_cast<int64_t>(NO_MASK);

    outputMaxLogits = 0;

    scaleValue = 1.0f;
    attenMaskCompressMode = NO_COMPRESS_MODE;
    attenMaskExistFlag = 0;
    prmMaskExistFlag = 0;
    sinkExistFlag = 0;
    isHighPercision = true;

    alignedS1 = 0;
    alignedS2 = 0;
    alignedD = 0;

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    dBasicBlock = std::numeric_limits<int64_t>::max();
    nRatio = GetNRatio();

    minUsedUBSize = 0;
    batchBasic = 1;

    opName = nullptr;
    inputLayout = nullptr;

    actualTemplate.Reset();
}

bool FlashAttentionScoreTilingBase::AnalyzeDtype()
{
    inputDtype = context_->GetInputDesc(0)->GetDataType();
    inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
    calcTypeSize = ge::GetSizeByDataType(inputDtype);
    switch (inputDtype) {
        case ge::DT_FLOAT16:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT16;
            bmm1OutDtype = isHighPercision ? matmul_tiling::DataType::DT_FLOAT : matmul_tiling::DataType::DT_FLOAT16;
            tilingKeyDType = isHighPercision ? DtypeEnum::FLOAT16_PRECISION : DtypeEnum::FLOAT16;
            calcTypeSize = isHighPercision ? ge::GetSizeByDataType(ge::DT_FLOAT) : ge::GetSizeByDataType(inputDtype);
            break;
        case ge::DT_FLOAT:
            bmmDtype = matmul_tiling::DataType::DT_FLOAT;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::FLOAT32;
            isHighPercision = false;
            break;
        case ge::DT_BF16:
            bmmDtype = matmul_tiling::DataType::DT_BF16;
            bmm1OutDtype = matmul_tiling::DataType::DT_FLOAT;
            tilingKeyDType = DtypeEnum::BFLOAT16;
            calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
            isHighPercision = false;
            break;
        default:
            TILING_INNER_ERR_REPORT(opName, "not support input dtype: %s for now",
                                    ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
            return false;
    }

    bmm2OutDtype = bmm1OutDtype;
    OP_LOGD(opName, "get high precision flag: %d", isHighPercision);
    return true;
}

bool FlashAttentionScoreTilingBase::AnalyzeAttrs()
{
    auto attrs = context_->GetAttrs();
    size_t idx = 0;
    auto scaleValuePtr = attrs->GetAttrPointer<float>(idx++);
    auto preTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto nextTokensPtr = attrs->GetAttrPointer<int64_t>(idx++);
    auto n1SizePtr = attrs->GetAttrPointer<uint32_t>(idx++);
    auto inputLayoutPtr = attrs->GetAttrPointer<char>(idx++);
    auto outputMaxLogitsPtr = attrs->GetAttrPointer<uint32_t>(idx++);
    // for AA3：qkv is BNSD, attenOut is orignal layout(bsh/sbh)
    if (strcmp(inputLayoutPtr, "AA3BSH") == 0) {
        inputLayout = "BNSD";
        tilingData.inputParams.set_attenOutLayoutType(LAYOUT_BSH);
    } else if (strcmp(inputLayoutPtr, "AA3SBH") == 0) {
        inputLayout = "BNSD";
        tilingData.inputParams.set_attenOutLayoutType(LAYOUT_SBH);
    } else {
        inputLayout = inputLayoutPtr;
        tilingData.inputParams.set_attenOutLayoutType(None);
    }

    preTokens = *preTokensPtr;
    nextTokens = *nextTokensPtr;
    scaleValue = *scaleValuePtr;
    n1Size = *n1SizePtr;
    outputMaxLogits = *outputMaxLogitsPtr;
    OP_CHECK_IF(n1Size == 0, TILING_INNER_ERR_REPORT(opName, "head num is zero"), return false);

    if (attrs->GetAttrNum() > idx) {
        auto sparseModePtr = attrs->GetAttrPointer<int64_t>(idx++);
        sparseMode = *sparseModePtr;
        if (sparseMode == LEFT_UP_CAUSAL) {
            attenMaskCompressMode = LEFT_UP_CAUSAL_MODE;
        } else if (sparseMode == RIGHT_DOWN_CAUSAL) {
            attenMaskCompressMode = RIGHT_DOWN_CAUSAL_MODE;
        } else if (sparseMode == BAND) {
            attenMaskCompressMode = BAND_MODE;
        }
        OP_LOGD(opName, "The current value of attenMaskCompressMode is %u.", attenMaskCompressMode);
    }

    OP_LOGD(opName,
            "attrs: scale_value[%f] keep_prob[%f] pre_tockens[%ld] next_tockens[%ld] head_num[%ld] input_layout[%s] "
            "sparse_mode[%ld]",
            scaleValue, preTokens, nextTokens, n1Size, inputLayout, sparseMode);
    return true;
}

bool FlashAttentionScoreTilingBase::AnalyzeLayout()
{
    auto& queryShape = context_->GetInputShape(0)->GetStorageShape();
    auto& keyShape = context_->GetInputShape(1)->GetStorageShape();

    size_t layoutLen = strlen(inputLayout);
    OP_LOGD(opName, "get input_layout [%s]", inputLayout);

    // Support sbnh input which equals TND
    // Make sure sbnh data is totally same as TND data
    OP_CHECK_IF((std::string(inputLayout) != "TND") &&
                    (queryShape.GetDimNum() != layoutLen || keyShape.GetDimNum() != layoutLen),
                TILING_INNER_ERR_REPORT(opName, "invalid layout[%s]", inputLayout), return false);

    OP_CHECK_IF(
        !Analyze3DimLayout(queryShape, keyShape, layoutLen) || !Analyze4DimLayout(queryShape, keyShape, layoutLen),
        TILING_INNER_ERR_REPORT(opName, "get unsupported layout: %s", inputLayout), return false);

    OP_CHECK_IF(gSize == 0, TILING_INNER_ERR_REPORT(opName, "get unsupported layout: %s", inputLayout), return false);
    return true;
}
void FlashAttentionScoreTilingBase::GetActualSeqLenData(int64_t inputIdx, std::vector<int64_t>& res,
                                                        int64_t& actualLen) const
{
    auto attrs = context_->GetAttrs();
    const gert::ContinuousVector* seqData = attrs->GetAttrPointer<gert::ContinuousVector>(inputIdx);
    if (seqData == nullptr) {
        OP_LOGW(opName, "[%s]actualSeqLen is null pointer", templateName);
        return;
    }
    size_t seq_size = seqData->GetSize();
    /* Get Data from tensor. */
    const int64_t* value = reinterpret_cast<const int64_t*>(seqData->GetData());
    if (value == nullptr) {
        OP_LOGW(opName, "[%s]actualSeqLen data is null pointer", templateName);
        return;
    }
    // use acculated seq lens, start with 0 as gpu
    for (auto i = 0; i < seq_size - 1; ++i) {
        res.push_back(value[i + 1] - value[i]);
        actualLen++;
    }
}
bool FlashAttentionScoreTilingBase::Analyze3DimLayout(const gert::Shape& queryShape, const gert::Shape& keyShape,
                                                      size_t layoutLen)
{
    int64_t h1 = 0;
    int64_t h2 = 0;
    auto& valueShape = context_->GetInputShape(2)->GetStorageShape();
    if (layoutLen == 3UL) {
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'H') { // 2: H idx
            bSize = queryShape.GetDim(0);
            s1Size = queryShape.GetDim(1);
            s2Size = keyShape.GetDim(1);
            h1 = queryShape.GetDim(2); // 2: H idx
            h2 = keyShape.GetDim(2);   // 2: H idx
            s1StrideSize = h1;
            s2StrideSize = h2;
            tilingData.inputParams.set_layoutType(LAYOUT_BSH);
            tilingKeyLayout = LayoutEnum::BSND;
        } else if (inputLayout[0] == 'S' && inputLayout[1] == 'B' && inputLayout[2] == 'H') { // 2: H idx
            s1Size = queryShape.GetDim(0);
            s2Size = keyShape.GetDim(0);
            bSize = queryShape.GetDim(1);
            h1 = queryShape.GetDim(2); // 2: H idx
            h2 = keyShape.GetDim(2);   // 2: H idx
            s1StrideSize = h1 * bSize;
            s2StrideSize = h2 * bSize;
            tilingData.inputParams.set_layoutType(LAYOUT_SBH);
            tilingKeyLayout = LayoutEnum::SBND;
        } else if (inputLayout[0] == 'T' && inputLayout[1] == 'N' && inputLayout[2] == 'D') {
            int64_t actualSeqQLen = 0;
            int64_t actualSeqKVLen = 0;
            GetActualSeqLenData(ACTUAL_SEQ_LENGTH_ATTRS_INDEX, actualSeqLenData, actualSeqQLen);
            GetActualSeqLenData(ACTUAL_SEQ_LENGTH_KV_ATTRS_INDEX, actualSeqLenKvData, actualSeqKVLen);
            if (actualSeqQLen >
                MAX_VAR_LEN_SEQ_LEN) { // batch size limit, atten mask(s1 * s2) may lead memory exceed GM size
                OP_LOGD(opName, "actualSeqQTensor shape size %zu is large than %d", actualSeqQLen, MAX_VAR_LEN_SEQ_LEN);
                return ge::GRAPH_PARAM_INVALID;
            }
            OP_CHECK_IF(actualSeqQLen != actualSeqKVLen,
                        TILING_INNER_ERR_REPORT(opName, "VarLen case, seq nums of query is not equal kv's"),
                        return false);
            OP_CHECK_IF(
                actualSeqQLen < 1,
                TILING_INNER_ERR_REPORT(opName, "cu sequence case, sequence query size %d should be larger than 1",
                                        actualSeqQLen + 1),
                return false);
            for (int64_t i = 0; i < actualSeqQLen; i++) {
                accumTT += actualSeqLenData[i] * actualSeqLenKvData[i];
                if (attenMaskCompressMode != NO_COMPRESS_MODE) {
                    if (actualSeqLenData[i] > actualSeqLenKvData[i]) {
                        OP_LOGE(opName, "Unsupport seqlenQ %d and seqlenKV %d, seqlenKV must greater than seqlenQ.",
                                actualSeqLenData[i], actualSeqLenKvData[i]);
                        return ge::GRAPH_PARAM_INVALID;
                    }
                }
            }

            bSize = actualSeqQLen;
            maxS1Val = *std::max_element(actualSeqLenData.begin(), actualSeqLenData.end());
            accumS1 = std::accumulate(actualSeqLenData.begin(), actualSeqLenData.end(), 0LL);
            maxS2Val = *std::max_element(actualSeqLenKvData.begin(), actualSeqLenKvData.end());
            accumS2 = std::accumulate(actualSeqLenKvData.begin(), actualSeqLenKvData.end(), 0LL);
            s1Size = maxS1Val;
            s2Size = maxS2Val;

            if (queryShape.GetDimNum() == 4) {
                n1Size = queryShape.GetDim(2);
                n2Size = keyShape.GetDim(2);
                dSize = queryShape.GetDim(3);
                vDsize = valueShape.GetDim(3);
            } else {
                n1Size = queryShape.GetDim(1);
                n2Size = keyShape.GetDim(1);
                dSize = queryShape.GetDim(2);
                vDsize = valueShape.GetDim(2);
            }
            OP_CHECK_IF(n2Size == 0, TILING_INNER_ERR_REPORT(opName, "N2 is zero"), return false);
            gSize = n1Size / n2Size;
            h1 = n1Size * dSize;
            h2 = n2Size * dSize;
            s1StrideSize = gSize * n2Size * dSize;
            s2StrideSize = n2Size * dSize;
            valueS2strideSize = n2Size * vDsize;
            tilingData.inputParams.set_layoutType(LAYOUT_TND);
            tilingKeyLayout = LayoutEnum::TND;
        }
        OP_CHECK_IF(h1 == 0 || h2 == 0, TILING_INNER_ERR_REPORT(opName, "H is zero"), return false);
        dSize = h1 / n1Size;
        gSize = h1 / h2;
        n2Size = h2 / dSize;
        OP_CHECK_IF(n1Size > 512, TILING_INNER_ERR_REPORT(opName, "Not support head num bigger than 512"),
                    return false);
    }

    return true;
}

bool FlashAttentionScoreTilingBase::Analyze4DimLayout(const gert::Shape& queryShape, const gert::Shape& keyShape,
                                                      size_t layoutLen)
{
    if (layoutLen == 4UL) {
        // 2: N idx, 3: D idx
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'N' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            s1Size = queryShape.GetDim(1);
            s2Size = keyShape.GetDim(1);
            n2Size = keyShape.GetDim(2); // 2: N idx
            OP_CHECK_IF(n2Size == 0, TILING_INNER_ERR_REPORT(opName, "N2 is zero"), return false);
            gSize = queryShape.GetDim(2) / n2Size; // 2: N idx
            dSize = queryShape.GetDim(3);          // 3: D idx
            s1StrideSize = gSize * n2Size * dSize;
            s2StrideSize = n2Size * dSize;
            tilingData.inputParams.set_layoutType(LAYOUT_BSND);
            tilingKeyLayout = LayoutEnum::BSND;
        } else if (inputLayout[0] == 'B' && inputLayout[1] == 'N' &&
                   // 2: S idx, 3: N idx
                   inputLayout[2] == 'S' && inputLayout[3] == 'D') {
            bSize = queryShape.GetDim(0);
            n2Size = keyShape.GetDim(1); // 1: N idx
            OP_CHECK_IF(n2Size == 0, TILING_INNER_ERR_REPORT(opName, "N2 is zero"), return false);
            gSize = queryShape.GetDim(1) / n2Size;
            s1Size = queryShape.GetDim(2); // 2: S idx
            s2Size = keyShape.GetDim(2);   // 2: S idx
            dSize = queryShape.GetDim(3);  // 3: D idx
            s1StrideSize = dSize;
            s2StrideSize = dSize;
            tilingData.inputParams.set_layoutType(LAYOUT_BNSD);
            tilingKeyLayout = LayoutEnum::BNSD;
        }
    }

    return true;
}

bool FlashAttentionScoreTilingBase::AnalyzeOptionalInput()
{
    auto attenMaskInput = context_->GetOptionalInputDesc(ATTENTION_MASK_INPUT_INDEX);
    auto attenMaskShape = context_->GetOptionalInputShape(ATTENTION_MASK_INPUT_INDEX);

    if (attenMaskInput != nullptr && attenMaskShape != nullptr && attenMaskShape->GetStorageShape().GetDimNum() != 0) {
        attenMaskExistFlag = 1;
        auto attenMaskType = attenMaskInput->GetDataType();
        OP_CHECK_IF(attenMaskType != ge::DT_BOOL && attenMaskType != ge::DT_UINT8,
                    TILING_INNER_ERR_REPORT(opName, "invalid attenMask dtype[%s], only support bool or uint8",
                                            ge::TypeUtils::DataTypeToSerialString(attenMaskType).c_str()),
                    return false);

        tilingData.inputParams.set_attenMaskDataType(1);
        // 0: (B,N2,G,S1,S2), 1: (B,1,1,S1,S2), 2: (1,1,1,S1,S2)
        AttenMaskShapeType attenMaskShapeType = ATTEN_B_N2_G_S1_S2;
        auto& attenMaskStorageShape = attenMaskShape->GetStorageShape();
        if (attenMaskStorageShape.GetDimNum() == ATTENTION_MASK_DIM_NUM) {
            if (attenMaskStorageShape.GetDim(0) == 1 && bSize > 1) {
                attenMaskShapeType = ATTEN_1_1_1_S1_S2;
            } else if (attenMaskStorageShape.GetDim(1) == 1 && n1Size > 1) {
                attenMaskShapeType = ATTEN_B_1_1_S1_S2;
            }
        } else if ((attenMaskStorageShape.GetDimNum() > 1 &&
                    attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - ATTEN_MASK_S1_REV_INDEX) ==
                        s1Size &&
                    attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 1) == s2Size) ||
                   (attenMaskCompressMode != NO_COMPRESS_MODE)) {
            attenMaskShapeType = ATTEN_1_1_1_S1_S2; // maybe [S1, S2]
        } else if (attenMaskStorageShape.GetDimNum() > 1 &&
                   attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - ATTEN_MASK_S1_REV_INDEX) ==
                       accumS1 &&
                   attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 1) == accumS2) {
            attenMaskShapeType = ATTEN_1_1_1_T_T;
        } else if (attenMaskStorageShape.GetDimNum() == 1 &&
                   attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 1) == accumTT) {
            attenMaskShapeType = ATTEN_1D_FLATTEN_S1_S2;
        }

        tilingData.inputParams.set_attenMaskShapeType(attenMaskShapeType);

        if (attenMaskStorageShape.GetDimNum() > 1) {
            if ((attenMaskCompressMode != NO_COMPRESS_MODE) &&
                ((attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - ATTEN_MASK_S1_REV_INDEX) !=
                  ATTEN_MASK_COMPRESS_LIMIT) ||
                 (attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 1) != ATTEN_MASK_COMPRESS_LIMIT))) {
                OP_LOGE(opName, "In the attenmask compression, please set the atten_mask_shape to [2048,2048].");
                return false;
            }
            tilingData.inputParams.set_attenMaskS2Size(
                attenMaskStorageShape.GetDim(attenMaskStorageShape.GetDimNum() - 1));
        }
    }

    auto prmMaskShape = context_->GetOptionalInputShape(PRM_MASK_INPUT_INDEX);
    if (prmMaskShape != nullptr && prmMaskShape->GetStorageShape().GetDimNum() != 0) {
        prmMaskExistFlag = 1;
    }
    tilingData.inputParams.set_prmMaskFlag(prmMaskExistFlag);

    auto sinkInput = context_->GetOptionalInputDesc(SINK_INPUT_INDEX);
    auto sinkShape = context_->GetOptionalInputShape(SINK_INPUT_INDEX);
    if (sinkInput != nullptr && sinkShape != nullptr && sinkShape->GetStorageShape().GetDimNum() != 0) {
        sinkExistFlag = 1;
    }
    tilingData.inputParams.set_sinkFlag(sinkExistFlag);

    OP_LOGD(opName, "attenMaskExistFlag: %d, prmMaskExistFlag: %d, sinkExistFlag: %d", attenMaskExistFlag,
            prmMaskExistFlag, sinkExistFlag);
    return true;
}

ge::graphStatus FlashAttentionScoreTilingBase::DoOpTiling()
{
    auto& inputParams = tilingData.inputParams;
    OP_LOGD(opName, "[%s]try template[%s]", templateName, expectTemplate.ToString().c_str());
    if (!MatchTemplate()) {
        OP_LOGD(opName, "[%s]not match template[%s], input params: bn2gs1s2d[%ld, %ld, %ld, %ld, %ld, %ld]",
                templateName, expectTemplate.ToString().c_str(), inputParams.get_bSize(), inputParams.get_n2Size(),
                inputParams.get_gSize(), inputParams.get_s1Size(), inputParams.get_s2Size(), inputParams.get_dSize());
        return ge::GRAPH_PARAM_INVALID;
    }

    SetCoreParams();
    SetMultiCoreParams();
    SetTensorSizeParams();
    SetSparseParams();
    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTilingBase::MatchTemplate()
{
    // UB Size calc logic: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    BufferNum bufferNum;
    GetBufferNum(bufferNum);

    s1BasicBlock = std::numeric_limits<int64_t>::max();
    s2BasicBlock = std::numeric_limits<int64_t>::max();
    CalcS1S2BasicBlock(bufferNum);

    if (s2BasicBlock == std::numeric_limits<int64_t>::max()) {
        OP_LOGD(opName,
                "[%s]can't find proper S1S2 basic block for shape: S1[%ld] S2[%ld], D[%ld], minS1BasicBlock[%ld], "
                "dtype[%s], high precision[%d]",
                templateName, s1Size, s2Size, dSize, GetMinS1BasicBlock(),
                ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str(), isHighPercision);
        return false;
    }

    CalcDBasicBlock(bufferNum);
    actualTemplate.splitS1 = s1BasicBlock < alignedS1 ? 1 : 0;
    actualTemplate.splitS2 = s2BasicBlock < alignedS2 ? 1 : 0;
    actualTemplate.splitD = dBasicBlock < alignedD ? 1 : 0;

    if (IsTemplateMatched()) {
        (void)CalcUBSize(bufferNum, s1BasicBlock, s2BasicBlock, dBasicBlock);
        OP_LOGD(opName, "[%s]final basic block: [%ld, %ld, %ld], match template[%s]", templateName, s1BasicBlock,
                s2BasicBlock, dBasicBlock, actualTemplate.ToString().c_str());
        return true;
    }

    return false;
}

void FlashAttentionScoreTilingBase::CalcS1S2BasicBlock(const BufferNum& bufferNum)
{
    // calc s1 s2 first, we set d basic block as s2 now
    const int64_t actualD = expectTemplate.splitD == 0 ? alignedD : FRACTAL_NUM; // if split d we use min s2 16
    int64_t maxS1BasicBlock = CalcMaxS1BasicBlockSize(actualD, bufferNum);
    maxS1BasicBlock = std::min(maxS1BasicBlock, alignedS1);
    if (maxS1BasicBlock == 0) {
        return;
    }

    for (int64_t tmpS1BasicBlock = std::min(GetMinS1BasicBlock(), maxS1BasicBlock); tmpS1BasicBlock <= maxS1BasicBlock;
         tmpS1BasicBlock += FRACTAL_NUM) {
        int64_t tmpS2BasicBlock = CalcMaxS2BasicBlockSize(bufferNum, tmpS1BasicBlock);
        tmpS2BasicBlock = std::min(tmpS2BasicBlock, alignedS2);
        for (; tmpS2BasicBlock >= FRACTAL_NUM; tmpS2BasicBlock -= FRACTAL_NUM) {
            int64_t tmpDBasicBlock = expectTemplate.splitD == 1 ? std::min(tmpS2BasicBlock, alignedD) : alignedD;
            OP_LOGD(opName, "[%s]try basic block: [%ld, %ld]", templateName, tmpS1BasicBlock, tmpS2BasicBlock);
            if (CalcUBSize(bufferNum, tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, 1) &&
                SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock)) {
                break;
            }
        }

        // check whether is valid, if tmpS1BasicBlock is too big, then there is no proper tmpS2BasicBlock
        if (tmpS2BasicBlock < FRACTAL_NUM) {
            break;
        }

        OP_LOGD(opName, "[%s]get candidate basic block: [%ld, %ld]", templateName, tmpS1BasicBlock, tmpS2BasicBlock);
        if (s2BasicBlock == std::numeric_limits<int64_t>::max()) {
            s1BasicBlock = tmpS1BasicBlock;
            s2BasicBlock = tmpS2BasicBlock;
        } else if (s2BasicBlock == tmpS2BasicBlock && s1BasicBlock < tmpS1BasicBlock) {
            s1BasicBlock = tmpS1BasicBlock;
        } else {
            break;
        }
    }
}

void FlashAttentionScoreTilingBase::CalcDBasicBlock(const BufferNum& bufferNum)
{
    return;
}

int64_t FlashAttentionScoreTilingBase::CalcMaxS1BasicBlockSize(int64_t actualD, const BufferNum& bufferNum) const
{
    // if S2 basic block is min value 16, s1 basic block can reach max value, then we get:
    // s1 * 16 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp =>
    // s1 * (16 * X + D * Y + (expNum + 2) * (32 / sizeof(T))) * sizeof(T) + apiTmp
    // just ignore apiTmp now, consider it at last
    int64_t alignUnit = BYTE_BLOCK / inputDtypeBytes;
    int64_t maxS1BasicBlock =
        aicoreParams_.ubSize / inputDtypeBytes /
        (FRACTAL_NUM * bufferNum.bufferS1S2Num + actualD * bufferNum.bufferS1DNum +
         (bufferNum.bufferExpNum + 2) * alignUnit); // here 2 means FlashSoftMax sum and max output
    return AlignDown(maxS1BasicBlock, FRACTAL_NUM);
}

int64_t FlashAttentionScoreTilingBase::CalcMaxS2BasicBlockSize(const BufferNum& bufferNum,
                                                               int64_t tmpS1BasicBlock) const
{
    // used UB: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    // if D full load, use alignedD in above formula
    // if D not full load, use S2 basic block var in above formula
    // just ignore apiTmp now, consider it at last
    int64_t tmpS2BasicBlock;
    if (expectTemplate.splitD == 0) {
        // here 2 means FlashSoftMax sum and max output
        tmpS2BasicBlock = (aicoreParams_.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK -
                           tmpS1BasicBlock * alignedD * bufferNum.bufferS1DNum * inputDtypeBytes) /
                          (tmpS1BasicBlock * bufferNum.bufferS1S2Num * inputDtypeBytes);
    } else {
        // here 2 means FlashSoftMax sum and max output
        tmpS2BasicBlock = (aicoreParams_.ubSize - tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK) /
                          (tmpS1BasicBlock * (bufferNum.bufferS1DNum + bufferNum.bufferS1S2Num) * inputDtypeBytes);
    }
    return std::min(AlignDown(tmpS2BasicBlock, FRACTAL_NUM), alignedS2);
}

bool FlashAttentionScoreTilingBase::CalcUBSize(const BufferNum& bufferNum, int64_t tmpS1BasicBlock,
                                               int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch)
{
    // used UB: s1s2 * X * sizeof(T) + s1d * Y * sizeof(T) + s1 * expNum * 32 + s1 * 64 + apiTmp
    int64_t usedUBSize = 0;
    usedUBSize += batch * tmpS1BasicBlock * tmpS2BasicBlock * bufferNum.bufferS1S2Num * inputDtypeBytes;
    usedUBSize += batch * tmpS1BasicBlock * tmpDBasicBlock * bufferNum.bufferS1DNum * inputDtypeBytes;
    // here 2 means FlashSoftMax sum and max output
    usedUBSize += batch * tmpS1BasicBlock * (bufferNum.bufferExpNum + 2) * BYTE_BLOCK;

    int64_t apiMinUBSize = CalcMinApiTmpSize(tmpS1BasicBlock, tmpS2BasicBlock, batch);
    if (apiMinUBSize + usedUBSize <= static_cast<int64_t>(aicoreParams_.ubSize)) {
        apiMaxUBSize = aicoreParams_.ubSize - usedUBSize;
        minUsedUBSize = usedUBSize + apiMinUBSize;
        return true;
    } else {
        return false;
    }
}

int64_t FlashAttentionScoreTilingBase::CalcMinApiTmpSize(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                         int64_t batch)
{
    auto tmpShape = ge::Shape({batch, tmpS1BasicBlock, tmpS2BasicBlock});
    int64_t softmaxFlashTmpSize = 0;
    if (expectTemplate.splitS2 == 1) {
        softmaxFlashTmpSize = AscendC::GetSoftMaxFlashV2MinTmpSize(tmpShape, calcTypeSize, sizeof(float), true,
                                                                   IsBasicBlockInSoftMax(tmpShape));
    }
    int64_t softmaxTmpSize = AscendC::GetSoftMaxMinTmpSize(tmpShape, calcTypeSize, true);
    int64_t dropoutTmpSize = 0;
    int64_t attenMaskTmpSize = 0;
    if (attenMaskExistFlag == 1) {
        attenMaskTmpSize = AscendC::GetSelectWithBytesMaskMinTmpSize(tmpShape, ge::Shape({1}), calcTypeSize, tmpShape,
                                                                     sizeof(uint8_t), false);
    }
    return std::max({softmaxFlashTmpSize, softmaxTmpSize, dropoutTmpSize, attenMaskTmpSize});
}

bool FlashAttentionScoreTilingBase::IsBasicBlockInSoftMax(const ge::Shape& shape) const
{
    // 2 axes at least
    if (shape.GetDimNum() < 2) {
        return false;
    }

    int64_t lastAxis = shape.GetDim(shape.GetDimNum() - 1);
    // last axis should be less than 2048 and fullfil 64 times
    int64_t basicLastAxis = 64;
    int64_t lastAxisNum = 2048;
    if (lastAxis > lastAxisNum || lastAxis % basicLastAxis != 0) {
        return false;
    }

    int64_t preAxes = 1;
    for (size_t idx = 0; idx < shape.GetDimNum() - 1; ++idx) {
        preAxes *= shape.GetDim(idx);
    }

    // all axes except last one should be 8 times
    return preAxes % 8 == 0;
}

void FlashAttentionScoreTilingBase::SetCoreParams()
{
    auto& coreParams = tilingData.coreParams;
    coreParams.set_s1BaseSize(s1BasicBlock);
    coreParams.set_s1BaseTailSize(CalcTailSize(s1Size, s1BasicBlock));
    coreParams.set_s1OuterSize(CeilDivision(s1Size, s1BasicBlock));
    coreParams.set_s2BaseSize(s2BasicBlock);
    coreParams.set_s2BaseTailSize(CalcTailSize(s2Size, s2BasicBlock));
    coreParams.set_s2OuterSize(CeilDivision(s2Size, s2BasicBlock));
    if (expectTemplate.splitS2 == 1) {
        nRatio = std::min(GetNRatio(), coreParams.get_s2OuterSize());
        coreParams.set_s2OuterSize(CeilDivision(coreParams.get_s2OuterSize(), nRatio));
    } else if (expectTemplate.splitS1 == 1) {
        nRatio = std::min(GetNRatio(), coreParams.get_s1OuterSize());
    } else {
        nRatio = 1;
    }
    coreParams.set_nRatio(nRatio);

    coreParams.set_dBaseSize(dBasicBlock);
    coreParams.set_dBaseTailSize(CalcTailSize(dSize, dBasicBlock));
    coreParams.set_dOuterSize(CeilDivision(dSize, dBasicBlock));
    // 向下取整保证数据量不超32K
    int64_t s1Vec2BaseSize = 8 * 1024 / alignedD;
    coreParams.set_s1Vec2BaseSize(std::min(s1Vec2BaseSize, S1_VEC2_BASE_SIZE_MAX));
    coreParams.set_s1Vec2BaseTailSize(s1Size % coreParams.get_s1Vec2BaseSize());
    SetMultiBatchCoreParams();
}

void FlashAttentionScoreTilingBase::SetMultiBatchCoreParams()
{
    auto& coreParams = tilingData.coreParams;
    coreParams.set_bBaseSize(1);
    coreParams.set_bBaseTailSize(1);
    coreParams.set_bOuterSize(bSize);

    coreParams.set_n2BaseSize(1);
    coreParams.set_n2BaseTailSize(1);
    coreParams.set_n2OuterSize(n2Size);

    coreParams.set_gBaseSize(1);
    coreParams.set_gBaseTailSize(1);
    coreParams.set_gOuterSize(gSize);
}

void FlashAttentionScoreTilingBase::SetMultiCoreParams()
{
    auto& multiCoreParams = tilingData.multiCoreParams;
    auto& coreParams = tilingData.coreParams;
    int64_t totalSize = coreParams.get_bOuterSize() * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize() *
                        coreParams.get_s1OuterSize();
    int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
    multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAivNum));
    multiCoreParams.set_totalSize(totalSize);
    multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
    multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
}

ge::graphStatus FlashAttentionScoreTilingBase::DoLibApiTiling()
{
    (void)SetMatMulTiling(s1BasicBlock, s2BasicBlock, dBasicBlock, batchBasic);
    SetSoftMaxTiling();
    SetDataCopyTransposeTiling();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FlashAttentionScoreTilingBase::PostTiling()
{
    context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize()); // already check capcity in CheckContext
    auto blockDim = CalcTschBlockDim(tilingData.multiCoreParams.get_coreNum(), aicNum, aivNum);
    context_->SetBlockDim(blockDim);
    context_->SetScheduleMode(1); // 1: batchmode模式

    OP_LOGD(opName, "[%s]tiling data: %s", templateName, GetTilingDataDebugStr().c_str());
    OP_LOGD(opName, "[%s]tiling data size: %zu", templateName, tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

bool FlashAttentionScoreTilingBase::SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                       int64_t tmpDBasicBlock, int64_t batch,
                                                       matmul_tiling::MatmulApiTiling& bmm1)
{
    bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
    bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
    bmm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
    bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
    bmm1.SetOrgShape(s1Size, s2Size, s1StrideSize, s2StrideSize);
    bmm1.SetBias(false);
    bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
    bmm1.SetFixSplit(tmpS1BasicBlock, tmpS2BasicBlock);

    return true;
}

bool FlashAttentionScoreTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                    int64_t tmpDBasicBlock, int64_t batch,
                                                    matmul_tiling::MatmulApiTiling& bmm1,
                                                    matmul_tiling::MatmulApiTiling& bmm2)
{
    if (!SetBmm1TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm1) ||
        !SetBmm2TilingInput(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm2)) {
        return false;
    }

    if (bmm1.GetTiling(tilingData.bmm1TilingData) == -1) {
        return false;
    }
    tilingData.bmm1TilingData.set_shareMode(0);
    tilingData.bmm1TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm1TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    if (bmm2.GetTiling(tilingData.bmm2TilingData) == -1) {
        return false;
    }

    tilingData.bmm2TilingData.set_shareMode(0);
    tilingData.bmm2TilingData.set_shareL1Size(aicoreParams_.l1Size);
    tilingData.bmm2TilingData.set_shareL0CSize(aicoreParams_.l0cSize);

    return true;
}

bool FlashAttentionScoreTilingBase::SetMatMulTiling(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                                                    int64_t tmpDBasicBlock, int64_t batch)
{
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo != nullptr) {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        matmul_tiling::MatmulApiTiling bmm1(ascendcPlatform);
        matmul_tiling::MatmulApiTiling bmm2(ascendcPlatform);
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm1, bmm2);
    } else {
        OP_LOGD(opName, "platform info is null, use default info to generate matmul tiling.");
        matmul_tiling::MatmulApiTiling bmm1;
        matmul_tiling::MatmulApiTiling bmm2;
        return SetMatMulTiling(tmpS1BasicBlock, tmpS2BasicBlock, tmpDBasicBlock, batch, bmm1, bmm2);
    }
}

void FlashAttentionScoreTilingBase::SetSoftMaxTiling()
{
    auto softmaxShape = ge::Shape({batchBasic, std::min(s1BasicBlock, alignedS1), std::min(s2BasicBlock, alignedS2)});

    AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                      tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
}

void FlashAttentionScoreTilingBase::SetDataCopyTransposeTiling()
{
    auto& coreParams = tilingData.coreParams;
    auto transposeSrcShape = ge::Shape({coreParams.get_bBaseSize(), 1, std::min(s1BasicBlock, alignedS1),
                                        coreParams.get_gBaseSize() * std::min(dBasicBlock, alignedD)});
    auto transposeDstShape = ge::Shape({bSize, n1Size, s1Size, n1Size * vDsize});
    GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShape, inputDtypeBytes, tilingData.transposeTilingData);
}

void FlashAttentionScoreTilingBase::SetTensorSizeParams()
{
    auto& tensorSizeParams = tilingData.tensorSizeParams;
    auto& coreParams = tilingData.coreParams;
    int64_t batchInnerSize = coreParams.get_bBaseSize() * coreParams.get_n2BaseSize() * coreParams.get_gBaseSize();
    tensorSizeParams.set_bmm1ResUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
    tensorSizeParams.set_attenMaskUbSize(attenMaskExistFlag * batchInnerSize * s1BasicBlock * s2BasicBlock);

    if (inputDtype == ge::DT_BF16 || isHighPercision) {
        if (expectTemplate.splitS2 == 1) {
            tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * std::max(s2BasicBlock, dBasicBlock));
        } else {
            tensorSizeParams.set_castUbSize(batchInnerSize * s1BasicBlock * s2BasicBlock);
        }
    }
    tensorSizeParams.set_softmaxMaxUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxSumUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / sizeof(float)));
    tensorSizeParams.set_softmaxExpUbSize(batchInnerSize * s1BasicBlock * (BYTE_BLOCK / calcTypeSize));
    tensorSizeParams.set_apiTmpBufferBytes(apiMaxUBSize);

    tensorSizeParams.set_bmm2ResUbSize(batchInnerSize * s1BasicBlock * dBasicBlock);
}

bool FlashAttentionScoreTilingBase::InitSparseValidArray(std::vector<int64_t>& sparseValidArray, int64_t bIdx)
{
    OP_CHECK_IF(sparseValidArray.size() == 0,
                TILING_INNER_ERR_REPORT(opName, "Sparse valid array size should be larger than 0."), return false);
    uint8_t sparseType = tilingData.inputParams.get_sparseType();
    int64_t s2BlkNum = CeilDivision(s2Size, s2BasicBlock);
    int64_t validS1Size = CeilDivision(preTokens, s1BasicBlock);
    int64_t validS2Size = CeilDivision(nextTokens, s2BasicBlock);
    for (int64_t i = 0; i < static_cast<int64_t>(sparseValidArray.size()); i++) {
        int64_t reduceBlk =
            (i < validS1Size) ? 0 : (CeilDivision((i + 1) * s1BasicBlock - preTokens, s2BasicBlock) - 1);
        int64_t addBlk = std::min(s2BlkNum - validS2Size,
                                  CeilDivision((i + 1) * s1BasicBlock + nextTokens, s2BasicBlock) - validS2Size);
        int64_t validBlockNum = validS2Size - reduceBlk + addBlk;
        sparseValidArray[i] = validBlockNum > 0 ? validBlockNum : s2BlkNum;
    }
    return true;
}

bool FlashAttentionScoreTilingBase::PartitionSparseData(const std::vector<int64_t>& sparseRollingArray,
                                                        int64_t sparseRollingArraySum, int64_t sparseArraySize,
                                                        int64_t loadMaxEachCore, std::vector<int64_t>& partitionResult)
{
    OP_CHECK_IF(partitionResult.size() == 0,
                TILING_INNER_ERR_REPORT(opName, "partitionResult size should be larger than 0."), return false);

    OP_CHECK_IF(sparseRollingArraySum <= 0,
                TILING_INNER_ERR_REPORT(opName, "sparseRollingArraySum should be larger than 0."), return false);
    int64_t s1OuterCutEachCore = loadMaxEachCore / sparseRollingArraySum;
    int64_t s1OuterLoadEachCore = s1OuterCutEachCore * sparseRollingArraySum;
    int64_t s1OuterNumEachCore = s1OuterCutEachCore * sparseRollingArray.size();

    int64_t targetCoreNum = partitionResult.size();
    int64_t coreIdx = 0;
    int64_t rollingIdx = 0;
    int64_t loadSize = s1OuterLoadEachCore;
    partitionResult[0] = 0;
    for (int64_t i = s1OuterNumEachCore; i < sparseArraySize; i++, rollingIdx++) {
        rollingIdx = (static_cast<uint64_t>(rollingIdx) >= sparseRollingArray.size()) ? 0 : rollingIdx;
        int64_t loadNext = sparseRollingArray[rollingIdx];
        bool needOneMoreCore = (loadSize + loadNext) > loadMaxEachCore;
        if (needOneMoreCore && coreIdx >= (targetCoreNum - 1)) {
            return false;
        }

        if (needOneMoreCore) {
            partitionResult[++coreIdx] = i;
            i += s1OuterNumEachCore;
            i--;
            rollingIdx--;
            loadSize = s1OuterLoadEachCore;
            continue;
        }

        loadSize += loadNext;
    }

    std::fill(partitionResult.begin() + coreIdx + 1, partitionResult.end(), sparseArraySize);
    return true;
}

void FlashAttentionScoreTilingBase::SetPrefixSparseStartIdx(const std::vector<std::vector<int64_t>>& sparseValidArray,
                                                            MultiCoreParams& multiCoreParams)
{
    int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
    int64_t* sparseStartIdx = multiCoreParams.get_sparseStartIdx();
    for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }
    if (totalSize <= validAivNum) {
        int64_t idx = 0;
        for (; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }
        for (; idx < validAivNum; ++idx) {
            sparseStartIdx[idx] = totalSize;
        }
        return;
    }

    int64_t loadTotal = 0;
    /* Need to adapt when we split b. */
    for (int64_t i = 0; i < bSize; i++) {
        loadTotal += std::accumulate(sparseValidArray[i].begin(), sparseValidArray[i].end(), 0LL);
    }
    int64_t n2G = tilingData.coreParams.get_n2OuterSize() * tilingData.coreParams.get_gOuterSize();
    loadTotal *= n2G;

    auto loadEachCoreExpect = CeilDivision(loadTotal, validAivNum);
    int64_t s1OuterSize = tilingData.coreParams.get_s1OuterSize();
    int64_t tempBlock = 0;
    int64_t coreIdx = 0;
    int64_t loadStartIdx = 0;

    for (int64_t bNGS1Index = 0; bNGS1Index < bSize * n2G * s1OuterSize; ++bNGS1Index) {
        int64_t bIdx = bNGS1Index / (n2G * s1OuterSize);
        if (s1OuterSize == 0) {
            continue;
        }
        int64_t s1Idx = bNGS1Index % s1OuterSize;
        auto currBlockNum = sparseValidArray[bIdx][s1Idx];
        if (tempBlock >= loadEachCoreExpect) {
            if ((tempBlock + currBlockNum - loadEachCoreExpect) >= (loadEachCoreExpect - (tempBlock))) {
                /* 不累加当前block */
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = bNGS1Index;
                // 下一个核使用当前这个S1
                tempBlock = currBlockNum;
            } else {
                sparseStartIdx[coreIdx++] = loadStartIdx;
                loadStartIdx = (bNGS1Index + 1);
                tempBlock = 0;
            }
        } else {
            tempBlock += currBlockNum;
        }
    }

    if (tempBlock != 0) {
        sparseStartIdx[coreIdx++] = loadStartIdx;
    }

    /* 将没用到的核的start index置为最大值 */
    for (; coreIdx < MAX_AIV_NUM; ++coreIdx) {
        sparseStartIdx[coreIdx] = totalSize;
    }
}

bool FlashAttentionScoreTilingBase::SetSparseStartIdx(const std::vector<int64_t>& sparseValidArray,
                                                      MultiCoreParams& multiCoreParams)
{
    // to avoid buffer overflow, or maybe sometimes we want to only verify single core
    int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
    int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
    int64_t* sparseStartIdx = multiCoreParams.get_sparseStartIdx();
    for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
        sparseStartIdx[idx] = totalSize;
    }

    if (totalSize <= validAivNum) {
        for (int64_t idx = 0; idx < totalSize; ++idx) {
            sparseStartIdx[idx] = idx;
        }

        return true;
    }

    // Minimize the max load each core to find a load balance result.
    // The range of max load each core is (loadEachCoreLowerBound, loadEachCoreUpperBound].
    std::vector<int64_t> partitionResult(validAivNum, totalSize);
    std::vector<int64_t> lastValidPartitionResult(validAivNum, totalSize);
    int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
    int64_t loadTotal = sparseArraySum * (totalSize / sparseValidArray.size());
    OP_CHECK_IF(validAivNum <= 0, TILING_INNER_ERR_REPORT(opName, "validAivNum should be larger than 0."),
                return false);
    int64_t loadEachCoreLowerBound = loadTotal / validAivNum - 1;
    int64_t loadEachCoreUpperBound =
        CeilDivision(loadTotal, validAivNum) + (*std::max_element(sparseValidArray.begin(), sparseValidArray.end()));
    while (loadEachCoreLowerBound + 1 < loadEachCoreUpperBound) {
        int64_t loadMax = loadEachCoreLowerBound + (loadEachCoreUpperBound - loadEachCoreLowerBound) / 2;
        if ((loadMax * validAivNum >= loadTotal) &&
            PartitionSparseData(sparseValidArray, sparseArraySum, totalSize, loadMax, partitionResult)) {
            loadEachCoreUpperBound = loadMax;
            lastValidPartitionResult.swap(partitionResult);
            continue;
        }
        loadEachCoreLowerBound = loadMax;
    }

    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        sparseStartIdx[idx] = lastValidPartitionResult[idx];
    }

    if (CheckLogLevel(OP, DLOG_DEBUG) == 1) {
        PrintSparseMaxMinLoadPerCore(sparseValidArray, sparseStartIdx, validAivNum,
                                     CeilDivision(loadTotal, validAivNum));
    }
    return true;
}

void FlashAttentionScoreTilingBase::PrintSparseMaxMinLoadPerCore(const std::vector<int64_t>& sparseValidArray,
                                                                 int64_t* sparseStartIdx, int32_t validAivNum,
                                                                 int64_t avgLoadSize)
{
    int64_t maxLoadSize = 0;
    int64_t minLoadSize = std::numeric_limits<int64_t>::max();
    int64_t totalSize = tilingData.multiCoreParams.get_totalSize();
    int64_t s1OuterSize = tilingData.coreParams.get_s1OuterSize();
    if (s1OuterSize == 0) {
        return;
    }
    for (int64_t idx = 0; idx < validAivNum; ++idx) {
        int64_t startIdx = sparseStartIdx[idx];
        int64_t endIdx = totalSize;
        if (idx + 1 < validAivNum) {
            endIdx = sparseStartIdx[idx + 1];
        }

        if (startIdx >= endIdx) {
            minLoadSize = 0;
            break;
        }

        int64_t s1OuterStartIdx = startIdx % s1OuterSize;
        int64_t s1OuterEndIdx = endIdx % s1OuterSize;
        int64_t loadSize = 0LL;
        if (s1OuterEndIdx > s1OuterStartIdx) {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx,
                                       sparseValidArray.begin() + s1OuterEndIdx, 0LL);
        } else {
            loadSize = std::accumulate(sparseValidArray.begin() + s1OuterStartIdx, sparseValidArray.end(), 0LL);
            loadSize = std::accumulate(sparseValidArray.begin(), sparseValidArray.begin() + s1OuterEndIdx, loadSize);
        }

        int64_t s1OuterLoop = (endIdx / s1OuterSize) - (startIdx / s1OuterSize);
        if (s1OuterLoop > 1) {
            if (s1OuterEndIdx > s1OuterStartIdx) {
                loadSize += s1OuterLoop * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
            } else {
                loadSize += (s1OuterLoop - 1) * std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
            }
        }

        maxLoadSize = std::max(maxLoadSize, loadSize);
        minLoadSize = std::min(minLoadSize, loadSize);
    }

    OP_LOGD(opName, "[%s]each core load: max[%ld], min[%ld], avg[%ld]", templateName, maxLoadSize, minLoadSize,
            avgLoadSize);
}

void FlashAttentionScoreTilingBase::SetSparseParams()
{
    if (tilingData.inputParams.get_sparseType() == static_cast<uint8_t>(SparseEnum::ALL)) {
        return;
    }

    if (expectTemplate.splitS2 == 0) {
        OP_LOGD(opName, "[%s]match not split S2 template, close sparse feature", templateName);
        tilingData.inputParams.set_sparseType(static_cast<uint8_t>(SparseEnum::ALL));
        return;
    }

    auto& coreParams = tilingData.coreParams;
    coreParams.set_s1SparseValidSize(preTokens);
    coreParams.set_s2SparseValidSize(nextTokens);

    auto& multiCoreParams = tilingData.multiCoreParams;
    std::vector<int64_t> sparseValidArray(coreParams.get_s1OuterSize(), 0);
    InitSparseValidArray(sparseValidArray, 0);
    SetSparseStartIdx(sparseValidArray, multiCoreParams);
}

ge::graphStatus FlashAttentionScoreTilingBase::GetWorkspaceSize()
{
    auto& tensorSizeParams = tilingData.tensorSizeParams;
    auto& coreParams = tilingData.coreParams;

    size_t* workspaces = context_->GetWorkspaceSizes(1);
    int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
    int64_t bmm2Byetes = tensorSizeParams.get_bmm2ResUbSize() * calcTypeSize;
    workspaces[0] = static_cast<size_t>((bmm1Byetes + bmm2Byetes) * aivNum) + WORK_SPACE_RESERVE_SIZE;
    return ge::GRAPH_SUCCESS;
}

class FusionAttentionVarLenScoreTilingDBHighPerformance : public FlashAttentionScoreTilingBase {
public:
    explicit FusionAttentionVarLenScoreTilingDBHighPerformance(gert::TilingContext* context)
        : FlashAttentionScoreTilingBase(context)
    {
        expectTemplate.splitS1 = 1;
        expectTemplate.splitS2 = 1;
        templateName = "VarLenFASplitS1S2DBHighPerformance";
    }
    ~FusionAttentionVarLenScoreTilingDBHighPerformance() override = default;

protected:
    int64_t s2sizeLimitMax = 1024;

    int64_t GetNRatio() override
    {
        return 8L;
    }

    void GetBufferNum(BufferNum& bufferNum) const override
    {
        bufferNum.bufferS1S2Num = HIGH_PERF_BUFFER_NUM;
    }

    void CalcS1S2BasicBlock(const BufferNum& bufferNum) override
    {
        s1BasicBlock = std::min(128L, alignedS1);
        s2BasicBlock = std::min(128L, alignedS2);
    }

    void CalcDBasicBlock(const BufferNum& bufferNum) override
    {
        dBasicBlock = std::min(128L, alignedD);
    }

    bool SetBmm1TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling& bmm1) override
    {
        bmm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, true);
        bmm1.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm1OutDtype);
        // 分不满核，且稀疏场景，shape设置的较小能产生更好的tiling
        if (bSize * n1Size * gSize * CeilDiv(s1Size, s1BasicBlock) <= aivNum && attenMaskExistFlag == 1) {
            bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size), std::min(tmpS2BasicBlock, s2Size), dSize);
        } else {
            bmm1.SetShape(std::min(tmpS1BasicBlock, s1Size),
                          std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size), dSize);
        }
        bmm1.SetOrgShape(s1Size, tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s1StrideSize, s2StrideSize);
        bmm1.SetBias(false);
        bmm1.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);

        return true;
    }

    bool SetBmm2TilingInput(int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock, int64_t tmpDBasicBlock, int64_t batch,
                            matmul_tiling::MatmulApiTiling& bmm2) override
    {
        bmm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmmDtype, false);
        bmm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, bmm2OutDtype);
        bmm2.SetShape(std::min(tmpS1BasicBlock, s1Size), vDsize,
                      std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size));
        // m, n, ka, kb, kc
        // 对应原始矩阵的 mnk信息
        bmm2.SetOrgShape(s1Size, valueS2strideSize,
                         std::min(tmpS2BasicBlock * tilingData.coreParams.get_nRatio(), s2Size), valueS2strideSize);
        bmm2.SetBias(false);
        bmm2.SetBufferSpace(aicoreParams_.l1Size, aicoreParams_.l0cSize);
        return true;
    }

    uint64_t GetTilingKey() const override
    {
        // not care about layout in tiling key, pass BSND(enum value is 0)
        return GET_TILINGKEY(AxisEnum::S1, AxisEnum::S2, AxisEnum::NONE, tilingKeyDType, LayoutEnum::TND,
                             SparseEnum::ANY, PerformanceOrientedEnum::BIG_DOUBLE_BUFFER);
    }

    bool IsCapable() override
    {
        if (tilingKeyLayout != LayoutEnum::TND) {
            return false;
        }
        return true;
    }

    void SetMultiCoreParams() override
    {
        auto& multiCoreParams = tilingData.multiCoreParams;
        auto& coreParams = tilingData.coreParams;
        accumS1BlockNum = 0;
        for (int64_t i = 0; i < bSize; i++) {
            OP_LOGD(opName, "[%s]actualSeqLenData data %ld is %ld", templateName, i, actualSeqLenData[i]);
            OP_LOGD(opName, "[%s]actualSeqLenKvData data %ld is %ld", templateName, i, actualSeqLenKvData[i]);
            accumS1BlockNum += CeilDivision(actualSeqLenData[i], s1BasicBlock);
        }
        int64_t totalSize = accumS1BlockNum * coreParams.get_n2OuterSize() * coreParams.get_gOuterSize();
        int64_t actualUsedAivNum = std::min(totalSize, static_cast<int64_t>(aivNum));
        // int64_t actualUsedAivNum = 1;
        multiCoreParams.set_coreNum(static_cast<int32_t>(actualUsedAivNum));
        multiCoreParams.set_aivNum(static_cast<int32_t>(aivNum));
        multiCoreParams.set_totalSize(totalSize);
        multiCoreParams.set_splitFactorSize(CeilDivision(totalSize, actualUsedAivNum));
        multiCoreParams.set_splitFactorTailSize(CalcTailSize(totalSize, multiCoreParams.get_splitFactorSize()));
    }

    bool IsTemplateMatched() const override
    {
        return true;
    }

    bool CalcUBSize(const BufferNum& bufferNum, int64_t tmpS1BasicBlock, int64_t tmpS2BasicBlock,
                    int64_t tmpDBasicBlock, int64_t batch) override
    {
        apiMaxUBSize = HIGH_PERF_API_BUFFER_MULTIPLE * tmpS1BasicBlock * tmpS2BasicBlock * sizeof(float);
        return true;
    }

    ge::graphStatus GetWorkspaceSize() override
    {
        auto& tensorSizeParams = tilingData.tensorSizeParams;
        auto& coreParams = tilingData.coreParams;

        size_t* workspaces = context_->GetWorkspaceSizes(1);
        int64_t bmm1Byetes = coreParams.get_nRatio() * tensorSizeParams.get_bmm1ResUbSize() * calcTypeSize;
        // outer dSize小于64的场景，无需做 vec2 DB， workspace占用较小
        int64_t vDsizeAligned = AlignUp(vDsize, FRACTAL_NUM);
        ;
        if (vDsizeAligned <= 64) {
            // stage1占用2倍的空间，stage2占用2倍空间
            workspaces[0] =
                static_cast<size_t>((bmm1Byetes * 2 + 2 * coreParams.get_s1BaseSize() * vDsizeAligned * calcTypeSize) *
                                    aivNum) +
                n1Size * accumS1BlockNum * sizeof(float) + WORK_SPACE_RESERVE_SIZE;
        } else {
            // 切D场景，stage1占用2倍的空间，stage2需要占用4倍空间
            workspaces[0] =
                static_cast<size_t>((bmm1Byetes * 2 + 4 * coreParams.get_s1BaseSize() * vDsizeAligned * calcTypeSize) *
                                    aivNum) +
                n1Size * accumS1BlockNum * sizeof(float) + WORK_SPACE_RESERVE_SIZE;
        }

        return ge::GRAPH_SUCCESS;
    }

    void SetSoftMaxTiling() override
    {
        auto softmaxShape = ge::Shape({s1BasicBlock / GetNRatio(), s2BasicBlock * GetNRatio()});
        AscendC::SoftMaxFlashV2TilingFunc(softmaxShape, calcTypeSize, sizeof(float), apiMaxUBSize,
                                          tilingData.softmaxFlashTilingData, true, IsBasicBlockInSoftMax(softmaxShape));
    }

    bool InitSparseValidArray(std::vector<int64_t>& sparseValidArray, int64_t bIdx) override
    {
        OP_CHECK_IF(sparseValidArray.size() == 0,
                    TILING_INNER_ERR_REPORT(opName, "Sparse valid array size should be larger than 0."), return false);

        // 特殊系数, 代表s2Size=[128, 256, 384, 512, 640, 768, 896, 1024] 对应的真实耗时膨胀值
        // 如 {256, 384, 512, 640, 768, 896, 960, 1024}; {384, 512, 640, 768, 832, 896, 960, 1024};
        // 和 {256, 256, 512, 512, 768, 768, 1024, 1024};
        // cof数组可变化，以下为当前选择参数，可优化不对齐场景负载均衡
        const int64_t cof[] = {256, 384, 512, 640, 768, 896, 960, 1024};
        auto& coreParams = tilingData.coreParams;
        uint8_t sparseType = tilingData.inputParams.get_sparseType();
        int64_t localAccumS1BlockNum = 0;
        bool causal = false;
        if (sparseType == static_cast<uint8_t>(SparseEnum::CAUSAL) && s1Size == s2Size) {
            causal = true;
        }
        for (int32_t i = 0; i < bSize; i++) {
            int64_t n2G = coreParams.get_n2OuterSize() * coreParams.get_gOuterSize();
            int64_t s1BlockNum = CeilDivision(actualSeqLenData[i], s1BasicBlock);
            for (int64_t j = 0; j < s1BlockNum; j++) {
                // 此处暂时设置为1, 由于实测尾块1和128性能差距不大，理论上应该如下所示
                // 理论值: s1RealSize为std::min(s1BasicBlock, (actualSeqLenData[i] - s1BasicBlock * j))
                int64_t s1RealSize = 1;
                int64_t s2RealSize = (causal == true) ? (s1BasicBlock * (j + 1)) : s2Size;
                s2RealSize = std::min(s2RealSize, actualSeqLenKvData[i]);
                // 新增一个系数, 解决理论和实际的差异
                int64_t s2RemainSize = s2RealSize % s2sizeLimitMax; // s2sizeLimitMax = 1024
                s2RealSize = (s2RealSize / s2sizeLimitMax) * s2sizeLimitMax;
                s2RealSize += ((s2RemainSize > 0) ? cof[CeilDivision(s2RemainSize, 128L) - 1] : 0);

                // 每个s1方向上切分块的计算量
                for (int64_t k = 0; k < n2G; k++) {
                    sparseValidArray[localAccumS1BlockNum + k * s1BlockNum + j] = s1RealSize * s2RealSize;
                }
            }
            localAccumS1BlockNum += (s1BlockNum * n2G);
        }

        return true;
    }

    bool BalanceLoad(const std::vector<int64_t>& sparseValidArray, MultiCoreParams& multiCoreParams,
                     std::vector<int64_t>& localValue, std::vector<int64_t>& sparseStartIdx)
    {
        // to avoid buffer overflow, or maybe sometimes we want to only verify single core
        int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
        int64_t totalSize = multiCoreParams.get_totalSize();
        int64_t maxVal = *std::max_element(localValue.begin(), localValue.end());
        int64_t tmpMaxVal = maxVal;

        // 从前往后遍历
        for (int64_t idx = 1; idx < validAivNum; ++idx) {
            int64_t start = sparseStartIdx[idx];
            if (start < totalSize && start > 0 && ((localValue[idx - 1] + sparseValidArray[start]) < maxVal)) {
                localValue[idx - 1] += sparseValidArray[start];
                localValue[idx] -= sparseValidArray[start];
                sparseStartIdx[idx] += 1;
            } else if (start == totalSize) {
                break;
            }
        }
        tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

        // 从后往前遍历
        for (int64_t idx = validAivNum - 1; idx > 0; --idx) {
            int64_t start = sparseStartIdx[idx];
            if (start == totalSize) {
                if (sparseStartIdx[idx - 1] == totalSize) {
                    continue;
                }
                localValue[idx - 1] -= sparseValidArray[start - 1];
                localValue[idx] = sparseValidArray[start - 1];
                sparseStartIdx[idx] -= 1;
            } else if (start > 0) {
                if ((localValue[idx] + sparseValidArray[start - 1]) >= tmpMaxVal) {
                    continue;
                }
                localValue[idx - 1] -= sparseValidArray[start - 1];
                localValue[idx] += sparseValidArray[start - 1];
                sparseStartIdx[idx] -= 1;
            } else {
                break;
            }
        }
        tmpMaxVal = *std::max_element(localValue.begin(), localValue.end());

        return (tmpMaxVal >= maxVal) ? false : true;
    }

    inline bool InitLoadValue(const std::vector<int64_t>& sparseValidArray, int64_t validAivNum, int64_t totalSize,
                              const std::vector<int64_t>& sparseStartIdx, std::vector<int64_t>& localValue)
    {
        for (int64_t idx = 0; idx < validAivNum; ++idx) {
            int64_t start = sparseStartIdx[idx];
            int64_t end = ((idx + 1) < validAivNum) ? sparseStartIdx[idx + 1] : totalSize;
            if (start < totalSize) {
                localValue[idx] =
                    std::accumulate(sparseValidArray.begin() + start, sparseValidArray.begin() + end, 0LL);
            } else {
                break;
            }
        }
        return true;
    }

    bool SetSparseStartIdx(const std::vector<int64_t>& sparseValidArray, MultiCoreParams& multiCoreParams) override
    {
        // to avoid buffer overflow, or maybe sometimes we want to only verify single core
        int64_t validAivNum = std::min(static_cast<int64_t>(multiCoreParams.get_coreNum()), MAX_AIV_NUM);
        int64_t totalSize = multiCoreParams.get_totalSize(); // BN2GS1.o
        int64_t* sparseStartIdx = multiCoreParams.get_sparseStartIdx();

        OP_CHECK_IF(totalSize <= 0, TILING_INNER_ERR_REPORT(opName, "totalSize should be larger than 0."),
                    return false);

        // initLoad: 使用均分策略, 保证后续不会比均分差
        int64_t splitFactorSize = multiCoreParams.get_splitFactorSize();
        std::vector<int64_t> localSparseStartIdx(MAX_AIV_NUM, totalSize);
        for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
            localSparseStartIdx[idx] = std::min((idx * splitFactorSize), totalSize);
        }
        std::vector<int64_t> localValue(validAivNum, 0);
        InitLoadValue(sparseValidArray, validAivNum, totalSize, localSparseStartIdx, localValue);
        // 负载均衡粗调
        std::vector<int64_t> tmpLocalValue(validAivNum, 0);
        std::vector<int64_t> tmpsparseStartIdx(MAX_AIV_NUM, totalSize);
        int64_t sparseArraySum = std::accumulate(sparseValidArray.begin(), sparseValidArray.end(), 0LL);
        int64_t avgVal = CeilDivision(sparseArraySum, validAivNum);

        tmpsparseStartIdx[0] = 0;
        for (int64_t idx = 1; idx < MAX_AIV_NUM; ++idx) {
            int64_t start = tmpsparseStartIdx[idx - 1];
            int64_t singleLoadValue = 0;
            tmpsparseStartIdx[idx] = start;
            while (singleLoadValue < avgVal && tmpsparseStartIdx[idx] < totalSize) {
                singleLoadValue += sparseValidArray[tmpsparseStartIdx[idx]];
                tmpsparseStartIdx[idx] += 1;
            }

            if ((start + 1) < tmpsparseStartIdx[idx]) {
                int64_t redoSingleLoadValue = singleLoadValue - sparseValidArray[tmpsparseStartIdx[idx] - 1];
                tmpsparseStartIdx[idx] = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ?
                                             (tmpsparseStartIdx[idx] - 1) :
                                             (tmpsparseStartIdx[idx]);
                singleLoadValue = ((singleLoadValue - avgVal) > (avgVal - redoSingleLoadValue)) ? redoSingleLoadValue :
                                                                                                  singleLoadValue;
                sparseArraySum -= singleLoadValue;
                avgVal = CeilDivision(sparseArraySum, (validAivNum - idx));
            }
        }
        InitLoadValue(sparseValidArray, validAivNum, totalSize, tmpsparseStartIdx, tmpLocalValue);

        // 负载均衡精调
        while (BalanceLoad(sparseValidArray, multiCoreParams, tmpLocalValue, tmpsparseStartIdx)) {
            // 根据负载均衡是否能得到更好预测结果决定是否结束循环
        }

        // exchange initLoad and 负载均衡
        if ((*std::max_element(localValue.begin(), localValue.end())) >
            (*std::max_element(tmpLocalValue.begin(), tmpLocalValue.end()))) {
            localSparseStartIdx.swap(tmpsparseStartIdx);
            localValue.swap(tmpLocalValue);
        }

        for (int64_t idx = 0; idx < MAX_AIV_NUM; ++idx) {
            sparseStartIdx[idx] = localSparseStartIdx[idx];
        }

        return true;
    }

    void SetSparseParams() override
    {
        auto& coreParams = tilingData.coreParams;
        auto& multiCoreParams = tilingData.multiCoreParams;
        std::vector<int64_t> sparseValidArray(multiCoreParams.get_totalSize(), 0);
        InitSparseValidArray(sparseValidArray, 0);
        SetSparseStartIdx(sparseValidArray, multiCoreParams);

        int64_t s1SparseValidSize = preTokens;
        int64_t s2SparseValidSize = nextTokens;
        coreParams.set_s1SparseValidSize(s1SparseValidSize);
        coreParams.set_s2SparseValidSize(s2SparseValidSize);
    }
};

// NOTE manually initialize tiling data in hostapi scenario in highest priority template
REGISTER_TILING_TEMPLATE("FlashAttention", FusionAttentionVarLenScoreTilingDBHighPerformance, 94);
} // namespace FA
} // namespace optiling
