/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flash_attention_grad_tiling.h"
#include "flash_attention_grad_tiling_common.h"
#include "register/op_impl_registry.h"

using namespace ge;
using namespace AscendC;

namespace optiling {

constexpr uint32_t KEY_BN_F16 = 3;
constexpr uint32_t KEY_S1_S2_BN_F16 = 10001;
constexpr uint32_t KEY_S1_S2_BNS_F16 = 10003;
constexpr uint32_t KEY_BASE = 1000;
constexpr uint32_t KEY_S1_S2_BASE = 10000;

constexpr uint32_t FAG_EMPTY_TILING_KEY = 90;
constexpr uint32_t SPARSE_BNS_HIGH_HIGH_PERF_TILING_KEY = 7;
constexpr uint32_t SPECIAL_SPARSE_BNS_HIGH_HIGH_PERF_TILING_KEY = 107;
constexpr uint32_t BNS_BF16_TILING_KEY = 23;
constexpr uint32_t BN_HIGH_PRECISE_TILING_KEY = 1017;
constexpr uint32_t SPARSE_BNS_HIGH_PRECISE_TILING_KEY = 1023;
constexpr uint32_t SPECIAL_BNS_BF16_TILING_KEY = 2023;
constexpr uint32_t SPECIAL_BN_HIGH_PRECISE_TILING_KEY = 2017;
constexpr uint32_t SPECIAL_SPARSE_BNS_HIGH_PRECISE_TILING_KEY = 2223;
constexpr uint32_t SPECIAL_BNS_BF16_96_96_TILING_KEY = 2123;

constexpr uint32_t CUBE_POSTLOAD_BLOCK_NUM = 4;
constexpr uint32_t HIGH_PERFORMANCE_MAX_D_VALUE = 64;

constexpr uint32_t SIZT_OF_HALF = 2;
constexpr uint32_t QK_V_BMM_NUM = 2;

enum class TilingKeyType {
    BN_TRANSPOSE_OUT = 0,
    BN_NO_TRANSPOSE_OUT,
    BNS_TRANSPOSE_OUT,
    BNS_NO_TRANSPOSE_OUT
};

struct BaseInfoParams { // 频繁使用的基础参数
    uint32_t coreNum;
    uint32_t ubSize;
    uint32_t l1Size;
    uint32_t l0aSize;
    uint32_t l0bSize;
    uint32_t l0cSize;

    uint32_t b;
    uint32_t n;
    uint32_t s;
    uint32_t sq;
    uint32_t skv;
    uint32_t d;
    uint32_t dataTypeSize;
    uint32_t blockNum;
    uint32_t mask;

    uint32_t tilingKey;
    uint32_t usedUBSize;

    uint32_t layoutSBH;

    bool isSparse;
    int innerPrecise = 0; // high precision
    uint32_t dataType;
    ge::DataType dType;

    uint32_t apiClcQueueSize = 0;
    bool hasAttenMask = true;
    bool n12Flag = false;
};

inline uint32_t GcdFunc(uint32_t a, uint32_t b) // a >= b
{
    if (b > a) {
        return GcdFunc(b, a);
    }
    if (a % b == 0) {
        return b;
    }
    return GcdFunc(b, a % b);
}

uint32_t CeilFunc(uint32_t num1, uint32_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

uint32_t AlignFunc(uint32_t num1, uint32_t num2)
{
    return CeilFunc(num1, num2) * num2;
}

uint32_t CalculateTschBlockDim(uint32_t sliceNum, uint32_t aicCoreNum, uint32_t aivCoreNum)
{
    uint32_t ration;
    if (aicCoreNum == 0 || aivCoreNum == 0 || aicCoreNum > aivCoreNum) {
        return sliceNum;
    }
    ration = aivCoreNum / aicCoreNum;
    return (sliceNum + (ration - 1)) / ration;
}

// tiling func + tiling prepare
class FlashAttentionScoreGradTiling {
public:
    FlashAttentionScoreGradTilingData tilingData;
    FlashAttentionScoreGradTiling() {};

    ge::graphStatus RunEmptyTiling(gert::TilingContext* context)
    {
        uint64_t aicNum = 40; // 40: B3 default aicNum
        uint64_t aivNum = 20; // 20: B3 default aivNum
        auto platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compilePtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo*>(context->GetCompileInfo());
            OP_CHECK_IF(compilePtr == nullptr, TILING_INNER_ERR_REPORT(context->GetNodeName(), "compile_info is null"),
                        return ge::GRAPH_FAILED);
            aivNum = compilePtr->aivNum;
            aicNum = compilePtr->aicNum;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            aicNum = ascendcPlatform.GetCoreNumAic();
            aivNum = ascendcPlatform.GetCoreNumAiv();
        }
        OP_CHECK_IF(aivNum == 0, TILING_INNER_ERR_REPORT("flash_attention_score_grad", "num of aiv is 0."),
                    return GRAPH_FAILED);
        uint64_t dqNum = context->GetOutputShape(OUTPUT_DQ_INDEX)->GetStorageShape().GetShapeSize();
        if (dqNum % aivNum == 0) {
            tilingData.emptyTensorTilingData.set_formerDqNum(aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDqNum(dqNum / aivNum);
            tilingData.emptyTensorTilingData.set_tailCoreDqNum(0);
        } else {
            tilingData.emptyTensorTilingData.set_formerDqNum(dqNum % aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDqNum(dqNum / aivNum + 1);
            tilingData.emptyTensorTilingData.set_tailCoreDqNum(dqNum / aivNum);
        }
        uint64_t dkNum = context->GetOutputShape(OUTPUT_DK_INDEX)->GetStorageShape().GetShapeSize();
        if (dkNum % aivNum == 0) {
            tilingData.emptyTensorTilingData.set_formerDkNum(aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDkNum(dkNum / aivNum);
            tilingData.emptyTensorTilingData.set_tailCoreDkNum(0);
        } else {
            tilingData.emptyTensorTilingData.set_formerDkNum(dkNum % aivNum);
            tilingData.emptyTensorTilingData.set_singleCoreDkNum(dkNum / aivNum + 1);
            tilingData.emptyTensorTilingData.set_tailCoreDkNum(dkNum / aivNum);
        }

        context->SetTilingKey(FAG_EMPTY_TILING_KEY);
        auto sliceNum = (dqNum < aivNum && dkNum < aivNum) ? std::max(dqNum, dkNum) : aivNum;
        context->SetBlockDim(CalculateTschBlockDim(sliceNum, aicNum, aivNum));
        context->SetScheduleMode(1); // 1: batchmode模式
        size_t* workspaces = context->GetWorkspaceSizes(1);
        workspaces[0] = 100 * 1024 * 1024;
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus RunBigKernelTiling(gert::TilingContext* context)
    {
        // set essential params
        SetBaseInfoParams(context);

        // set tiling key
        uint32_t tilingKey = 0;

        FlashAttentionScoreGradSetTilingKeyFunc(context, tilingKey);

        if (mBaseParams.isSparse) {
            tilingKey += 4; // 4 is sparse
        }
        if (mBaseParams.dataTypeSize == 4) { // 4 is sizeof 32bit date type
            tilingKey += 8;                  // 8 is 32bit date type key
        }

        if (mBaseParams.dataType == 2) { // 2 is BF16
            tilingKey += 8;              // 8 is bf16 key
        }
        FlashAttentionScoreGradSetS1S2TilingKeyFunc(context, tilingKey);
        UpdateTilingKeyFunc(tilingKey);
        mBaseParams.tilingKey = tilingKey;

        uint32_t sBase = FRACTAL_NUM;

        // 根据shape、硬件size限制（参考CheckArgsLegal函数）、切分sOut、sFla 并设置；
        std::pair<uint32_t, uint32_t> bestSplitRes = BisectionForBestSplit(mBaseParams.s, mBaseParams.skv);
        uint32_t sOut = bestSplitRes.first;
        uint32_t sFla = bestSplitRes.second;
        uint32_t tailCoreSOut = bestSplitRes.first;
        uint32_t tailCoreSFla = bestSplitRes.second;

        tilingData.splitCoreParams.set_sOut(sOut);
        tilingData.splitCoreParams.set_sFla(sFla);

        tilingData.tailSplitCoreParams.set_sOut(tailCoreSOut);
        tilingData.tailSplitCoreParams.set_sFla(tailCoreSFla);

        uint32_t preTokensBlocks = CeilFunc(tilingData.baseParams.get_preTokens(), sOut);
        uint32_t nextTokensBlocks = CeilFunc(tilingData.baseParams.get_nextTokens(), sFla);
        tilingData.baseParams.set_preTokensBlocks(preTokensBlocks);
        tilingData.baseParams.set_nextTokensBlocks(nextTokensBlocks);

        sBase = sOut;

        // 根据tilingKey执行相应分核函数 填好分核参数 batchFactor
        // mTilingFuncMap[tilingKey](tilingData);
        SplitCoreTilingFunc(tilingKey % 4, sBase); // 4 reserve

        // 单核tiling
        SingleCoreTiling();

        SetApiTiling();

        // Cast Tiling
        CastTilingFunc();

        SaveTilingData(context);

        context->SetTilingKey(tilingKey);

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetBaseInfoParams(gert::TilingContext* context)
    {
        uint32_t coreNum = 40; // 40 is init core num

        auto platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compilePtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo*>(context->GetCompileInfo());
            OP_CHECK_IF(compilePtr == nullptr, TILING_INNER_ERR_REPORT(context->GetNodeName(), "compile_info is null"),
                        return ge::GRAPH_FAILED);
            coreNum = compilePtr->aivNum;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            coreNum = ascendcPlatform.GetCoreNumAiv();
        }
        tilingData.baseParams.set_coreNum(coreNum);

        mBaseParams.ubSize = 192 * 1024;                       // 192 * 1024 = 192kB
        mBaseParams.l1Size = 512 * 1024;                       // 512 * 1024 = 512kB
        mBaseParams.l0aSize = mBaseParams.l0bSize = 64 * 1024; // 64 * 1024 = 64kB
        mBaseParams.l0cSize = 128 * 1024;                      // 128 * 1024 = 128kB

        const gert::StorageShape* queryShape = context->GetInputShape(QUARY_INDEX);
        const gert::StorageShape* keyShape = context->GetInputShape(KEY_INDEX);
        const char* inputLayout = context->GetAttrs()->GetAttrPointer<char>(ATTR_INPUT_LAYOUT_INDEX);
        uint32_t bVal = 0;
        uint32_t nVal = *context->GetAttrs()->GetAttrPointer<int>(ATTR_HEAD_NUM_INDEX); // 4 is N index
        uint32_t sVal = 0;
        uint32_t dVal = 0;
        uint32_t hVal = 0;
        uint32_t sq = 0;
        uint32_t skv = 0;
        uint32_t layoutType = 0;
        if (inputLayout[0] == 'B' && inputLayout[1] == 'S' && inputLayout[2] == 'H') { // 2 is H index
            bVal = queryShape->GetStorageShape().GetDim(0);
            sVal = queryShape->GetStorageShape().GetDim(1);
            hVal = queryShape->GetStorageShape().GetDim(2); // 2 is H index
            sq = queryShape->GetStorageShape().GetDim(1);
            skv = keyShape->GetStorageShape().GetDim(1);
            layoutType = 0;
            dVal = hVal / nVal;
        } else if (inputLayout[0] == 'S' && inputLayout[1] == 'B' && inputLayout[2] == 'H') { // 2 is H index
            sVal = queryShape->GetStorageShape().GetDim(0);
            bVal = queryShape->GetStorageShape().GetDim(1);
            hVal = queryShape->GetStorageShape().GetDim(2); // 2 is H index
            sq = queryShape->GetStorageShape().GetDim(0);
            skv = keyShape->GetStorageShape().GetDim(0);
            layoutType = 1;
            dVal = hVal / nVal;
        } else if (inputLayout[0] == 'B' && inputLayout[1] == 'N' && inputLayout[2] == 'S' && inputLayout[3] == 'D') {
            bVal = queryShape->GetStorageShape().GetDim(0);
            nVal = queryShape->GetStorageShape().GetDim(1);
            sVal = queryShape->GetStorageShape().GetDim(2);
            dVal = queryShape->GetStorageShape().GetDim(3);
            sq = queryShape->GetStorageShape().GetDim(2);
            skv = keyShape->GetStorageShape().GetDim(2);
            layoutType = 2;
            hVal = nVal * dVal;
        }

        tilingData.baseParams.set_B(bVal);
        tilingData.baseParams.set_N(nVal);
        tilingData.baseParams.set_S(sVal);
        tilingData.baseParams.set_D(dVal);
        tilingData.baseParams.set_H(hVal);
        tilingData.baseParams.set_batch(bVal * nVal);
        tilingData.baseParams.set_layoutSBH(layoutType);
        tilingData.baseParams.set_Sq(sq);
        tilingData.baseParams.set_Skv(skv);

        auto queryType = context->GetInputDesc(QUARY_INDEX)->GetDataType();
        mBaseParams.dataTypeSize = 4; // init date type bf16 is 4
        mBaseParams.mask = 64;        // 64 is init mask
        mBaseParams.dataType = 0;

        if (queryType == ge::DT_FLOAT16) {
            mBaseParams.dataTypeSize = 2; // 2 is fp16
            mBaseParams.mask = 128;       // fp16 init mask is 128
            mBaseParams.dataType = 1;
        } else if (queryType == ge::DT_BF16) {
            mBaseParams.dataTypeSize = 4; // 4 is bf16
            mBaseParams.mask = 64;        // bf16 init mask is 64
            mBaseParams.dataType = 2;     // 2 is bf16
        }
        if (queryType == ge::DT_FLOAT16 && mBaseParams.innerPrecise == 0) {
            mBaseParams.dataTypeSize = 4;
            mBaseParams.mask = 64;
            mBaseParams.dataType = 2;
        }

        auto attenMaskDesc = context->GetInputDesc(ATTEN_MASK_INDEX);
        if (attenMaskDesc != nullptr) {
            auto attenMaskDtype = attenMaskDesc->GetDataType();
            if (attenMaskDtype == ge::DT_BOOL || attenMaskDtype == ge::DT_UINT8) {
                tilingData.baseParams.set_attenMaskType(ATTEN_MASK_TYPE_U8_BOOL);
            } else {
                tilingData.baseParams.set_attenMaskType(ATTEN_MASK_TYPE_SAME);
            }
        }
        auto attenMaskShape = context->GetOptionalInputShape(ATTEN_MASK_INDEX);
        if (attenMaskShape != nullptr) {
            auto dim0 = attenMaskShape->GetStorageShape().GetDim(ATTEN_MASK_SHAPE_TEMP_DIMS);
            if (dim0 != gert::Shape::kInvalidDimValue && dim0 == bVal) {
                tilingData.baseParams.set_attenMaskShapeType(ATTEN_MASK_SHAPE_TYPE_B1SS);
            } else {
                tilingData.baseParams.set_attenMaskShapeType(ATTEN_MASK_SHAPE_TYPE_SS);
            }
        } else {
            mBaseParams.hasAttenMask = false;
        }

        int preTokens = *context->GetAttrs()->GetAttrPointer<int>(ATTR_PRE_TOKENS_INDEX);
        int nextTokens = *context->GetAttrs()->GetAttrPointer<int>(ATTR_NEXT_TOKENS_INDEX);

        tilingData.baseParams.set_typeBlockNum(32 / sizeof(float)); // 32 reserve
        tilingData.baseParams.set_scaleValue(*(context->GetAttrs()->GetAttrPointer<float>(ATTR_SCALE_VALUE_INDEX)));

        tilingData.baseParams.set_preTokens(preTokens);
        tilingData.baseParams.set_nextTokens(nextTokens);
        tilingData.baseParams.set_splitBatchFactor(1);
        tilingData.baseParams.set_splitSeqFactor(1);
        tilingData.baseParams.set_allMatInputNum(sVal * dVal);
        tilingData.baseParams.set_allMatResNum(sVal * skv);
        tilingData.baseParams.set_allReduceNum(sVal * tilingData.baseParams.get_typeBlockNum());

        mBaseParams.coreNum = coreNum;
        mBaseParams.b = bVal;
        mBaseParams.n = nVal;
        mBaseParams.s = sVal;
        mBaseParams.sq = sq;
        mBaseParams.skv = skv;
        mBaseParams.d = dVal;
        mBaseParams.layoutSBH = layoutType;
        mBaseParams.isSparse = false;
        mBaseParams.dType = queryType;
        mBaseParams.blockNum = BYTE_BLOCK / mBaseParams.dataTypeSize;
        if ((uint32_t)preTokens < sVal || (uint32_t)nextTokens < skv) {
            mBaseParams.isSparse = true;
        }
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus FlashAttentionScoreGradSetTilingKeyFunc(gert::TilingContext* context, uint32_t& tilingKey)
    {
        uint32_t bVal = tilingData.baseParams.get_B();
        uint32_t nVal = tilingData.baseParams.get_N();
        uint32_t coreNum = tilingData.baseParams.get_coreNum();
        if (bVal * nVal >= coreNum && coreNum / GcdFunc(bVal, coreNum) <= nVal &&
            GcdFunc(coreNum, bVal) * GcdFunc(coreNum, nVal) >= CORE_THRESHOLD) {
            tilingKey = 1; // 1 bn key
            uint32_t batchRange = CeilFunc(bVal * nVal, coreNum);
            uint32_t usedCoreNum = CeilFunc(bVal * nVal, batchRange);
            if (usedCoreNum % 2 != 0) { // 2 is
                tilingKey = 3;          // 3 bns key
            }
        } else {
            tilingKey = 3; // 3 bns key
        }
        return ge::GRAPH_SUCCESS;
    }

    void FlashAttentionScoreGradSetS1S2TilingKeyFunc(gert::TilingContext* context, uint32_t& tilingKey)
    {
        uint32_t s = tilingData.baseParams.get_S();
        uint32_t skv = tilingData.baseParams.get_Skv();
        if (s != skv) {
            // tilingKey: fp16, flash, S不等长, 1+10000(BN), 3+10000(BNS)
            // tilingKey: fp16, flash, S不等长, high precision, 17+10000(BN), 19+10000(BNS)
            if (mBaseParams.dType == ge::DT_FLOAT16) {
                tilingKey += KEY_S1_S2_BASE;
            }
        } else {
            // tilingKey: fp16, flash, S等长, 1(BN), 3(BNS)
            // tilingKey: fp16, flash, S等长, high precision, 17+1000(BN), 19+1000(BNS)
            // tilingKey: fp16, flash, sparse, S等长, 21+1000(BN), 23+1000(BNS)
            if (mBaseParams.innerPrecise == 0 && mBaseParams.dType == ge::DT_FLOAT16) {
                tilingKey += KEY_BASE;
            }
        }
    }

    void UpdateTilingKeyFunc(uint32_t& tilingKey)
    {
        // SD: [24, 5, 9216, 64], [24, 10, 2304, 64]
        if (tilingKey == BN_HIGH_PRECISE_TILING_KEY &&
            ((mBaseParams.b == 24 && mBaseParams.n == 5 && mBaseParams.s == 9216 && mBaseParams.d == 64) ||
             (mBaseParams.b == 24 && mBaseParams.n == 10 && mBaseParams.s == 2304 && mBaseParams.d == 64))) {
            tilingKey = SPECIAL_BN_HIGH_PRECISE_TILING_KEY;
        }

        // [2, 5, 8192, 128], high performance
        if (tilingKey == SPARSE_BNS_HIGH_HIGH_PERF_TILING_KEY &&
            (mBaseParams.b == 2 && mBaseParams.n == 5 && mBaseParams.s == 8192 && mBaseParams.d == 128)) {
            tilingKey = SPECIAL_SPARSE_BNS_HIGH_HIGH_PERF_TILING_KEY;
        }

        // [2, 5, 8192, 128], high precise
        if (tilingKey == SPARSE_BNS_HIGH_PRECISE_TILING_KEY &&
            (mBaseParams.b == 2 && mBaseParams.n == 5 && mBaseParams.s == 8192 && mBaseParams.d == 128)) {
            tilingKey = SPECIAL_SPARSE_BNS_HIGH_PRECISE_TILING_KEY;
        }

        // [1, 12, 6144, 128], [1, 8, 6144, 128], [2, 5, 8192, 128]
        if (tilingKey == BNS_BF16_TILING_KEY) {
            if (mBaseParams.b == 2 && mBaseParams.n == 5 && mBaseParams.s == 8192 && mBaseParams.d == 128) {
                tilingKey = SPECIAL_BNS_BF16_TILING_KEY;
            } else if ((mBaseParams.b == 1 && mBaseParams.n == 12 && mBaseParams.s == 6144 && mBaseParams.d == 128) ||
                       (mBaseParams.b == 1 && mBaseParams.n == 8 && mBaseParams.s == 6144 && mBaseParams.d == 128)) {
                tilingKey = SPECIAL_BNS_BF16_96_96_TILING_KEY;
                if (mBaseParams.n == 12) {
                    mBaseParams.n12Flag = true;
                }
            }
        }
    }

    ge::graphStatus SplitCoreTilingFunc(uint32_t tilingKey, uint32_t sBase)
    {
        switch (tilingKey) {
            case 0: // 0 belong BN key
            case 1: // 1 belong BN key
                SplitCoreTilingBN();
                break;
            case 2: // 2 belong BNS key
            case 3: // 3 belong BNS key
                SplitCoreTilingBNS(sBase);
                break;
            default:
                return ge::GRAPH_FAILED;
        }
        return ge::GRAPH_SUCCESS;
    }

    uint32_t GetRealUsedCoreNum(uint32_t splitSeqFactor, uint32_t coreNum, uint32_t batch, bool isSBN)
    {
        uint32_t bnAndCoreNumCommonFactor;
        uint32_t batchRange;
        uint32_t splitBatchFactor;
        if (isSBN) {
            bnAndCoreNumCommonFactor = CeilFunc(coreNum, splitSeqFactor);
            batchRange = CeilFunc(batch, bnAndCoreNumCommonFactor); // ceil
            splitBatchFactor = CeilFunc(batch, batchRange);
            return splitBatchFactor * splitSeqFactor;
        }
        bnAndCoreNumCommonFactor = GcdFunc(batch, mBaseParams.coreNum);
        batchRange = CeilFunc(batch, bnAndCoreNumCommonFactor);
        return bnAndCoreNumCommonFactor * splitSeqFactor;
    }

    std::pair<uint32_t, uint32_t> SelectSbnOrBns(uint32_t sBase, uint32_t& batchRange, uint32_t& splitBatchFactor,
                                                 uint32_t& remainBatchRange)
    {
        uint32_t batch = tilingData.baseParams.get_batch();
        uint32_t bnAndCoreNumCommonFactor = 0;
        bool isSBN = false;
        uint32_t coreNum = tilingData.baseParams.get_coreNum();
        uint32_t seq = tilingData.baseParams.get_S();
        uint32_t sbnSplitSeqFactor = 0;
        uint32_t bnsSplitSeqFactor = 0;
        std::pair<uint32_t, uint32_t> splitedCoreSeqNum;

        // 根据空核数，选择SBN分核和BNS分核
        std::pair<uint32_t, uint32_t> sbnSplitedCoreSeqNum = CalcSplitSeqFirst(seq, sBase, sbnSplitSeqFactor); // SBN
        uint32_t sbnRealUsedCoreNum = GetRealUsedCoreNum(sbnSplitSeqFactor, coreNum, batch, true);
        std::pair<uint32_t, uint32_t> bnsSplitedCoreSeqNum = CalcSplitBnFirst(seq, sBase, bnsSplitSeqFactor); // BNS
        uint32_t bnsRealUsedCoreNum = GetRealUsedCoreNum(bnsSplitSeqFactor, coreNum, batch, false);
        if (bnsRealUsedCoreNum > sbnRealUsedCoreNum) {
            tilingData.baseParams.set_splitSeqFactor(bnsSplitSeqFactor);
            splitedCoreSeqNum = bnsSplitedCoreSeqNum;
        } else if (bnsRealUsedCoreNum == sbnRealUsedCoreNum) {
            if (bnsSplitSeqFactor > sbnSplitSeqFactor) {
                tilingData.baseParams.set_splitSeqFactor(sbnSplitSeqFactor);
                splitedCoreSeqNum = sbnSplitedCoreSeqNum;
                isSBN = true;
            } else {
                tilingData.baseParams.set_splitSeqFactor(bnsSplitSeqFactor);
                splitedCoreSeqNum = bnsSplitedCoreSeqNum;
            }
        } else {
            tilingData.baseParams.set_splitSeqFactor(sbnSplitSeqFactor);
            splitedCoreSeqNum = sbnSplitedCoreSeqNum;
            isSBN = true;
        }

        tilingData.baseParams.set_preSeqCoreNum(tilingData.baseParams.get_splitSeqFactor() - 1);
        if (mBaseParams.n12Flag) {
            splitedCoreSeqNum.first = 8 * sBase;
            splitedCoreSeqNum.second = 6 * sBase;
            tilingData.baseParams.set_preSeqCoreNum(2);
        }

        if (isSBN) { // 根据相应的分核计算splitBatchFactor和真实使用核数
            bnAndCoreNumCommonFactor = coreNum / tilingData.baseParams.get_splitSeqFactor();
            batchRange = CeilFunc(batch, bnAndCoreNumCommonFactor); // ceil
            splitBatchFactor = CeilFunc(batch, batchRange);
            remainBatchRange = batch - batchRange * (splitBatchFactor - 1);
        } else {
            bnAndCoreNumCommonFactor = GcdFunc(batch, mBaseParams.coreNum);
            batchRange = CeilFunc(batch, bnAndCoreNumCommonFactor);
            splitBatchFactor = bnAndCoreNumCommonFactor;
            remainBatchRange = batch - batchRange * (splitBatchFactor - 1);
        }

        return splitedCoreSeqNum;
    }

    ge::graphStatus SplitCoreTilingBNS(uint32_t sBase)
    {
        uint32_t seq = tilingData.baseParams.get_S(); // 1024
        uint32_t seqKv = tilingData.baseParams.get_Skv();
        uint32_t d = tilingData.baseParams.get_D(); // 64

        uint32_t batchRange;
        uint32_t splitBatchFactor;
        uint32_t remainBatchRange;
        std::pair<uint32_t, uint32_t> splitedCoreSeqNum =
            SelectSbnOrBns(sBase, batchRange, splitBatchFactor, remainBatchRange);

        tilingData.baseParams.set_splitBatchFactor(splitBatchFactor);
        tilingData.baseParams.set_coreNum(splitBatchFactor * tilingData.baseParams.get_splitSeqFactor());
        SplitCoreParams& splitCoreParams = tilingData.splitCoreParams;
        splitCoreParams.set_splitedBatchRange(batchRange);
        splitCoreParams.set_splitedCoreSeqNum(splitedCoreSeqNum.first);
        splitCoreParams.set_splitedCoreBatchNum(splitCoreParams.get_splitedBatchRange() * seq * d);
        splitCoreParams.set_splitAllMatInputNum(splitedCoreSeqNum.first * d);
        splitCoreParams.set_splitedCoreBatchMaskNum(splitCoreParams.get_splitedBatchRange() * seq * seqKv);
        splitCoreParams.set_splitAllMatResNum(splitedCoreSeqNum.first * seqKv);
        splitCoreParams.set_splitedCoreBatchReduceNum(splitCoreParams.get_splitedBatchRange() * seq * 32 /
                                                      sizeof(float));                        // 32 reserve
        splitCoreParams.set_splitAllReduceNum(splitedCoreSeqNum.first * 32 / sizeof(float)); // 32 reserve

        SplitCoreParams& tailSplitCoreParams = tilingData.tailSplitCoreParams;
        tailSplitCoreParams.set_splitedBatchRange(remainBatchRange);
        tailSplitCoreParams.set_splitedCoreSeqNum(splitedCoreSeqNum.second);
        tailSplitCoreParams.set_splitedCoreBatchNum(tailSplitCoreParams.get_splitedBatchRange() * seq * d);
        tailSplitCoreParams.set_splitAllMatInputNum(splitedCoreSeqNum.second * d);
        tailSplitCoreParams.set_splitedCoreBatchMaskNum(tailSplitCoreParams.get_splitedBatchRange() * seq * seqKv);
        tailSplitCoreParams.set_splitAllMatResNum(splitedCoreSeqNum.second * seqKv);
        tailSplitCoreParams.set_splitedCoreBatchReduceNum(tailSplitCoreParams.get_splitedBatchRange() * seq * 32 /
                                                          sizeof(float));                         // 32 reserve
        tailSplitCoreParams.set_splitAllReduceNum(splitedCoreSeqNum.second * 32 / sizeof(float)); // 32 reserve
        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SplitCoreTilingBN()
    {
        // 按照B * N分核，填充参数tailSplitCoreParams和splitCoreParams
        uint32_t batch = tilingData.baseParams.get_batch();
        uint32_t seq = tilingData.baseParams.get_S();
        uint32_t seqKv = tilingData.baseParams.get_Skv();
        uint32_t d = tilingData.baseParams.get_D();

        uint32_t batchRange = CeilFunc(batch, mBaseParams.coreNum); // ceil
        uint32_t usedCoreNum = CeilFunc(batch, batchRange);
        uint32_t remainBatchRange = batch - batchRange * (usedCoreNum - 1);

        FlashAttentionScoreGradBaseParams& baseParams = tilingData.baseParams;
        mBaseParams.coreNum = usedCoreNum;
        baseParams.set_coreNum(mBaseParams.coreNum);

        SplitCoreParams& splitCoreParams = tilingData.splitCoreParams;
        splitCoreParams.set_splitedBatchRange(batchRange); // 每个核处理总大小
        splitCoreParams.set_splitedCoreSeqNum(seq);
        splitCoreParams.set_splitedCoreBatchNum(batchRange * seq * d);
        splitCoreParams.set_splitAllMatInputNum(seq * d);
        splitCoreParams.set_splitedCoreBatchMaskNum(batchRange * seq * seqKv);
        splitCoreParams.set_splitAllMatResNum(seq * seqKv);
        splitCoreParams.set_splitedCoreBatchReduceNum(batchRange * seq * 32 / sizeof(float)); // 32 reserve
        splitCoreParams.set_splitAllReduceNum(seq * 32 / sizeof(float));                      // 32 reserve

        SplitCoreParams& tailSplitCoreParams = tilingData.tailSplitCoreParams;
        tailSplitCoreParams.set_splitedBatchRange(remainBatchRange); // 每个核处理总大小
        tailSplitCoreParams.set_splitedCoreSeqNum(seq);
        tailSplitCoreParams.set_splitedCoreBatchNum(remainBatchRange * seq * d);
        tailSplitCoreParams.set_splitAllMatInputNum(seq * d);
        tailSplitCoreParams.set_splitedCoreBatchMaskNum(remainBatchRange * seq * seqKv);
        tailSplitCoreParams.set_splitAllMatResNum(seq * seqKv);
        tailSplitCoreParams.set_splitedCoreBatchReduceNum(remainBatchRange * seq * 32 / sizeof(float)); // 32 reserve
        tailSplitCoreParams.set_splitAllReduceNum(seq * 32 / sizeof(float));                            // 32 reserve
        /** 待考虑有空核的情况 */

        return ge::GRAPH_SUCCESS;
    }

    std::pair<int, int> CalcSplitBnFirst(uint32_t s, uint32_t sBase, uint32_t& splitSeqFactor) // for BNS
    {
        uint32_t bnCommonFactor = GcdFunc(mBaseParams.b * mBaseParams.n, mBaseParams.coreNum);
        splitSeqFactor = CeilFunc(mBaseParams.coreNum, bnCommonFactor);
        uint32_t seqBlockNum = CeilFunc(s, sBase);
        std::pair<int, int> ans;
        uint32_t coreSeq = CeilFunc(seqBlockNum, splitSeqFactor);
        splitSeqFactor = CeilFunc(seqBlockNum, coreSeq);
        uint32_t tailCoreSeq = s - coreSeq * sBase * (splitSeqFactor - 1);

        if (splitSeqFactor == 1) {
            ans.first = s;
            ans.second = s;
        } else {
            ans.first = coreSeq * sBase;
            ans.second = tailCoreSeq;
        }
        return ans;
    }

    std::pair<uint32_t, uint32_t> CalcSplitSeqFirst(uint32_t s, uint32_t sBase, uint32_t& splitSeqFactor) // for SBN
    {
        uint32_t coreNum = tilingData.baseParams.get_coreNum();
        uint32_t seqBlockNum = CeilFunc(s, sBase);
        std::pair<uint32_t, uint32_t> ans;

        splitSeqFactor = GcdFunc(coreNum, seqBlockNum); // ceil
        uint32_t coreSeq = CeilFunc(seqBlockNum, splitSeqFactor);
        uint32_t tailCoreSeq = s - coreSeq * sBase * (splitSeqFactor - 1);
        if (splitSeqFactor == 1) {
            ans.first = s;
            ans.second = s;
        } else {
            ans.first = coreSeq * sBase;
            ans.second = tailCoreSeq;
        }
        return ans;
    }

    std::pair<uint32_t, uint32_t> CalcSplitedCoreSeqNum(uint32_t s, uint32_t splitSeqFactor, uint32_t sBase) // for BNS
    {
        uint32_t seqBlockNum = CeilFunc(s, sBase); // 11
        std::pair<uint32_t, uint32_t> ans;
        uint32_t coreSeq = CeilFunc(seqBlockNum, splitSeqFactor); // ceil
        uint32_t usedCoreNum = CeilFunc(seqBlockNum, coreSeq);
        uint32_t tailCoreSeq = s - coreSeq * sBase * (usedCoreNum - 1);

        if (usedCoreNum == 1) {
            ans.first = s;
            ans.second = s;
        } else {
            ans.first = coreSeq * sBase;
            ans.second = tailCoreSeq;
        }

        tilingData.baseParams.set_splitSeqFactor(usedCoreNum);
        mBaseParams.coreNum = usedCoreNum * tilingData.baseParams.get_splitBatchFactor();
        tilingData.baseParams.set_coreNum(mBaseParams.coreNum);

        return ans;
    }

    ge::graphStatus SingleCoreTiling()
    {
        // 填singleCoreParams和tailSingleCoreParams的参数
        SetSingleCoreParams(tilingData.splitCoreParams);
        SetSingleCoreParams(tilingData.tailSplitCoreParams);
        return ge::GRAPH_SUCCESS;
    }

    std::pair<uint32_t, uint32_t> BisectionForBestSplit(const uint32_t& seq, const uint32_t& seqKv)
    {
        std::pair<uint32_t, uint32_t> bestSplitRes; // sout, sfla

        uint32_t sOut = seq <= INITIAL_SOUT_SPLIT_NUM ? seq : INITIAL_SOUT_SPLIT_NUM;
        uint32_t sFla = seqKv <= INITIAL_SFLA_SPLIT_NUM ? seqKv : INITIAL_SFLA_SPLIT_NUM;

        uint32_t sOutDiv = 1;
        uint32_t sFlaDiv = 1;
        while (true) {
            if (!CheckArgsLegal(sOut, sFla)) {
                if (seqKv >= INITIAL_SFLA_SPLIT_NUM) {
                    sFlaDiv *= 2;                                // 2 increase
                    sFla = sFla / FRACTAL_NUM / 2 * FRACTAL_NUM; // 2 increase
                }

                sOutDiv *= 2;                                // 2 increase
                sOut = sOut / FRACTAL_NUM / 2 * FRACTAL_NUM; // 2 increase
            } else {
                bestSplitRes.first = sOut;
                bestSplitRes.second = sFla;
                break;
            }
        }
        if (mBaseParams.dataType == 2) { // 2 is bf16
            sOut = 64;                   // 64 reserve
            sFla = 128;                  // 128 reserve

            if (mBaseParams.tilingKey == SPECIAL_BNS_BF16_96_96_TILING_KEY) {
                sOut = 96;
                sFla = 96;
            }

            if (tilingData.baseParams.get_isHighPerformanceMode() == 1) {
                sOut = INITIAL_SOUT_SPLIT_NUM / 2;
            }
            (void)CheckArgsLegal(sOut, sFla); // update mBaseParams.apiClcQueueSize
            bestSplitRes.first = sOut;
            bestSplitRes.second = sFla;
        }

        return bestSplitRes;
    }

    bool CheckArgsLegal(uint32_t sOut, uint32_t sFla)
    {
        bool isLegal = false;
        // 计算ub buffer的总大小
        uint32_t vecInQue1Size = sOut * mBaseParams.d * mBaseParams.dataTypeSize * 2; // 2 is max node num in queue
        uint32_t vecInQue2Size = sOut * sFla * mBaseParams.dataTypeSize;
        uint32_t vecClc1Size = sOut * sFla * mBaseParams.dataTypeSize;
        uint32_t vecClc2Size = sOut * sFla * mBaseParams.dataTypeSize;

        if (mBaseParams.dataType == 2) {       // 2 is bf16
            vecInQue1Size = vecInQue1Size / 2; // bf16 queue size reduce 2
            vecInQue2Size = vecInQue2Size / 2; // bf16 queue size reduce 2
            vecClc1Size = mBaseParams.d > sFla ? sOut * mBaseParams.d * mBaseParams.dataTypeSize :
                                                 sOut * sFla * mBaseParams.dataTypeSize;
            vecClc2Size = vecClc1Size;
            if (tilingData.baseParams.get_isHighPerformanceMode() == 1) {
                vecInQue2Size = 0;
            }
        }

        uint32_t queBufferSizeUb = vecInQue1Size + vecInQue2Size + vecClc1Size + vecClc2Size;

        // softmaxGrad，softmax，dropout计算所需要的tmpSize
        uint32_t maxTmpBufferSize = GetTmpBufferSize(sOut, sFla);

        uint32_t bufferSizeUb = queBufferSizeUb + maxTmpBufferSize + SOFTMAX_REMAIN_SIZE;

        // L1 buffer上的占用
        uint32_t scmSize = sOut * sFla * mBaseParams.dataTypeSize * 2; // 2 reserve
        uint32_t bufferSizeL1 = scmSize;
        uint32_t bufferSizeL0c = std::max(sOut * mBaseParams.d * 4, sFla * mBaseParams.d * 4); // 4 reserve
        bufferSizeL0c = std::max(bufferSizeL0c, sOut * sFla * 4);                              // 4 reserve

        uint32_t bufferSizeL0a = std::max(sOut, sFla) * mBaseParams.d * mBaseParams.dataTypeSize;
        bufferSizeL0a = std::max(bufferSizeL0a, sFla * sOut * mBaseParams.dataTypeSize);

        mBaseParams.usedUBSize = queBufferSizeUb;

        if (bufferSizeUb <= mBaseParams.ubSize && bufferSizeL1 <= mBaseParams.l1Size &&
            bufferSizeL0c <= mBaseParams.l0cSize && bufferSizeL0a <= mBaseParams.l0aSize) {
            isLegal = true;
            mBaseParams.apiClcQueueSize = mBaseParams.ubSize - queBufferSizeUb - 4 * 1024; // 4*1024=4KB Reserved
            mBaseParams.apiClcQueueSize = std::max(mBaseParams.apiClcQueueSize, maxTmpBufferSize);
        }
        return isLegal;
    }

    uint32_t GetTmpBufferSize(uint32_t sOut, uint32_t sFla)
    {
        auto softmaxShape = Shape({sOut, sFla});
        auto softmaxGradShape = Shape({sOut, mBaseParams.d});
        auto dropoutShape = Shape({sOut, sFla});

        uint32_t softmaxTmpSize = GetSoftMaxMinTmpSize(softmaxShape, mBaseParams.dataTypeSize, true);
        uint32_t softmaxGradTmpSize = GetSoftMaxGradMinTmpSize(softmaxGradShape, mBaseParams.dataTypeSize, true, true);
        uint32_t dropoutTmpSize = GetDropOutMinTmpSize(dropoutShape, mBaseParams.dataTypeSize, true);

        return std::max(std::max(softmaxTmpSize, dropoutTmpSize), softmaxGradTmpSize);
    }

    ge::graphStatus SetSingleCoreParams(SplitCoreParams& splitCoreParams)
    {
        // 通过每个核的sOut、sFla计算其他相关参数
        uint32_t sOut = splitCoreParams.get_sOut();
        uint32_t sFla = splitCoreParams.get_sFla();
        uint32_t sOutAlign = AlignFunc(sOut, mBaseParams.blockNum);
        uint32_t sOutCubeAlign = AlignFunc(sOut, CUBE_ALIGN_NUM);
        uint32_t sFlaAlign = AlignFunc(sFla, mBaseParams.blockNum);

        uint32_t sOutDiv = CeilFunc(splitCoreParams.get_splitedCoreSeqNum(), sOut);
        uint32_t sOutTail = splitCoreParams.get_splitedCoreSeqNum() - sOut * (sOutDiv - 1);
        uint32_t sFlaDiv = CeilFunc(mBaseParams.skv, sFla);
        uint32_t sFlaTail = mBaseParams.skv - sFla * (sFlaDiv - 1);
        uint32_t sOutTailAlign = AlignFunc(sOutTail, mBaseParams.blockNum);
        uint32_t sOutTailCubeAlign = AlignFunc(sOutTail, CUBE_ALIGN_NUM);
        uint32_t sFlaTailAlign = AlignFunc(sFlaTail, mBaseParams.blockNum);

        splitCoreParams.set_flashKLoopRange(sFlaDiv);
        splitCoreParams.set_s1OutRange(sOutDiv);
        splitCoreParams.set_sOutTail(sOutTail);
        splitCoreParams.set_sFlaTail(sFlaTail);
        splitCoreParams.set_sOutAlign(sOutAlign);
        splitCoreParams.set_sFlaAlign(sFlaAlign);
        splitCoreParams.set_sOutTailAlign(sOutTailAlign);
        splitCoreParams.set_sFlaTailAlign(sFlaTailAlign);
        splitCoreParams.set_sOutCubeAlign(sOutCubeAlign);
        splitCoreParams.set_sOutTailCubeAlign(sOutTailCubeAlign);

        uint32_t size = mBaseParams.dataTypeSize;
        uint32_t blockNum = 32 / sizeof(float); // 32 date in block

        splitCoreParams.set_sInnerRange(1);
        uint32_t sInner = sOut / splitCoreParams.get_sInnerRange();
        splitCoreParams.set_sInner(sInner);

        splitCoreParams.set_innerReduceNum(sInner * blockNum);
        splitCoreParams.set_innerMatResNum(sInner * sFla);
        splitCoreParams.set_innerMatResTailNum(sInner * sFlaTail);
        splitCoreParams.set_innerMatResNumAlign(sInner * sFlaAlign);
        splitCoreParams.set_innerMatResTailNumAlign(sInner * sFlaTailAlign);
        splitCoreParams.set_innerMatInputNum(sInner * mBaseParams.d);
        splitCoreParams.set_innerMaskNum(sInner * mBaseParams.skv);

        splitCoreParams.set_outReduceNum(sOut * blockNum);
        splitCoreParams.set_outMatResNum(sOut * sFla);
        splitCoreParams.set_outMatResTailNum(sOut * sFlaTail);
        splitCoreParams.set_outMatInputNum(sOut * mBaseParams.d);
        splitCoreParams.set_outMatInputNumCubeAlign(sOutCubeAlign * mBaseParams.d);
        splitCoreParams.set_outMaskNum(sOut * mBaseParams.skv);
        splitCoreParams.set_sFlaMatInputNum(sFla * mBaseParams.d);
        splitCoreParams.set_sFlaMatInputTailNum(sFlaTail * mBaseParams.d);

        splitCoreParams.set_innerReduceSize(sInner * blockNum * size);
        splitCoreParams.set_innerMatResSize(sInner * sFla * size);
        splitCoreParams.set_innerMatResTailSize(sInner * sFlaTail * size);
        splitCoreParams.set_innerMatResSizeAlign(sInner * sFlaAlign * size);
        splitCoreParams.set_innerMatResTailSizeAlign(sInner * sFlaTailAlign * size);
        splitCoreParams.set_innerMatInputSize(sInner * mBaseParams.d * size);
        splitCoreParams.set_innerMaskSize(sInner * mBaseParams.skv);

        splitCoreParams.set_outReduceSize(sOut * blockNum * size);
        splitCoreParams.set_outMatResSize(sOut * sFla * size);
        splitCoreParams.set_outMatResTailSize(sOut * sFlaTail * size);
        splitCoreParams.set_outMatInputSize(sOut * mBaseParams.d * size);
        splitCoreParams.set_outMatInputSizeCubeAlign(sOutCubeAlign * mBaseParams.d * size);
        splitCoreParams.set_outMaskSize(sOut * mBaseParams.skv);
        splitCoreParams.set_sFlaMatInputSize(sFla * mBaseParams.d * size);
        splitCoreParams.set_sFlaMatInputTailSize(sFlaTail * mBaseParams.d * size);

        splitCoreParams.set_sFlaBlock(sFla * size / 32);                                  // 32 date in block
        splitCoreParams.set_sFlaTailBlock(sFlaTail * size / 32);                          // 32 date in block
        splitCoreParams.set_sSubSFlaStride((mBaseParams.skv - sFla) * size / 32);         // 32 date in block
        splitCoreParams.set_sSubSFlaTailStride((mBaseParams.skv - sFlaTail) * size / 32); // 32 date in block
        splitCoreParams.set_subMask(mBaseParams.mask);
        splitCoreParams.set_sFlaBlockNum(sFla * size / 32);         // 32 date in block
        splitCoreParams.set_sFlaBlockTailNum(sFlaTail * size / 32); // 32 date in block
        splitCoreParams.set_subRange(sInner / 8);                   // 8 reserve

        splitCoreParams.set_innerMatInputBlock(splitCoreParams.get_innerMatInputSize() / 32);     // 32 date in block
        splitCoreParams.set_innerMaskBlock(splitCoreParams.get_innerMaskSize() / 32);             // 32 date in block
        splitCoreParams.set_innerMatResBlock(splitCoreParams.get_innerMatResSize() / 32);         // 32 date in block
        splitCoreParams.set_innerMatResTailBlock(splitCoreParams.get_innerMatResTailSize() / 32); // 32 date in block
        splitCoreParams.set_dropStride((mBaseParams.skv - sFla) * size / 32);                     // 32 date in block

        // sOutTailParams
        splitCoreParams.set_tailSInnerRange(1);
        uint32_t tailsInner = sOutTail / splitCoreParams.get_tailSInnerRange();
        splitCoreParams.set_tailSInner(tailsInner);

        splitCoreParams.set_tailInnerReduceNum(tailsInner * blockNum);
        splitCoreParams.set_tailInnerMatResNum(tailsInner * sFla);
        splitCoreParams.set_tailInnerMatResTailNum(tailsInner * sFlaTail);
        splitCoreParams.set_tailInnerMatInputNum(tailsInner * mBaseParams.d);
        splitCoreParams.set_tailInnerMaskNum(tailsInner * mBaseParams.skv);

        splitCoreParams.set_tailOutReduceNum(sOutTail * blockNum);
        splitCoreParams.set_tailOutMatResNum(sOutTail * sFla);
        splitCoreParams.set_tailOutMatResTailNum(sOutTail * sFlaTail);
        splitCoreParams.set_tailOutMatInputNum(sOutTail * mBaseParams.d);
        splitCoreParams.set_tailOutMaskNum(sOutTail * mBaseParams.skv);
        splitCoreParams.set_tailSFlaMatInputNum(sFla * mBaseParams.d);
        splitCoreParams.set_tailSFlaMatInputTailNum(sFlaTail * mBaseParams.d);

        splitCoreParams.set_tailInnerReduceSize(tailsInner * blockNum * size);
        splitCoreParams.set_tailInnerMatResSize(tailsInner * sFla * size);
        splitCoreParams.set_tailInnerMatResTailSize(tailsInner * sFlaTail * size);
        splitCoreParams.set_tailInnerMatInputSize(tailsInner * mBaseParams.d * size);
        splitCoreParams.set_tailInnerMaskSize(tailsInner * mBaseParams.skv);

        splitCoreParams.set_tailOutReduceSize(sOutTail * blockNum * size);
        splitCoreParams.set_tailOutMatResSize(sOutTail * sFla * size);
        splitCoreParams.set_tailOutMatResTailSize(sOutTail * sFlaTail * size);
        splitCoreParams.set_tailOutMatInputSize(sOutTail * mBaseParams.d * size);
        splitCoreParams.set_tailOutMaskSize(sOutTail * mBaseParams.skv);
        splitCoreParams.set_tailSFlaMatInputSize(sFla * mBaseParams.d * size);
        splitCoreParams.set_tailSFlaMatInputTailSize(sFlaTail * mBaseParams.d * size);

        splitCoreParams.set_tailSFlaBlock(sFla * size / 32);                                  // 32blockdata
        splitCoreParams.set_tailSFlaTailBlock(sFlaTail * size / 32);                          // 32blockdata
        splitCoreParams.set_tailSSubSFlaStride((mBaseParams.skv - sFla) * size / 32);         // 32blockdata
        splitCoreParams.set_tailSSubSFlaTailStride((mBaseParams.skv - sFlaTail) * size / 32); // 32blockdata
        splitCoreParams.set_tailSubMask(mBaseParams.mask);
        splitCoreParams.set_tailSFlaBlockNum(sFla * size / 32);         // 32blockdata
        splitCoreParams.set_tailSFlaBlockTailNum(sFlaTail * size / 32); // 32blockdata
        splitCoreParams.set_tailSubRange(tailsInner / 8);               // 8 reserve

        splitCoreParams.set_tailDropStride((mBaseParams.skv - sFla) * size / 32); // 32blockdata

        if (mBaseParams.dataType == 2) { // 2 bf16
            // FP32 calculate
            splitCoreParams.set_sFlaBlock(sFla * size / 64);                                  // 64 date in block
            splitCoreParams.set_sFlaTailBlock(sFlaTail * size / 64);                          // 64 date in block
            splitCoreParams.set_sSubSFlaStride((mBaseParams.skv - sFla) * size / 64);         // 64 date in block
            splitCoreParams.set_sSubSFlaTailStride((mBaseParams.skv - sFlaTail) * size / 64); // 64 date in block

            splitCoreParams.set_tailSFlaBlock(sFla * size / 64);                                  // 64 date in block
            splitCoreParams.set_tailSFlaTailBlock(sFlaTail * size / 64);                          // 64 date in block
            splitCoreParams.set_tailSSubSFlaStride((mBaseParams.skv - sFla) * size / 64);         // 64 date in block
            splitCoreParams.set_tailSSubSFlaTailStride((mBaseParams.skv - sFlaTail) * size / 64); // 64 date in block
        }

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus SetApiTiling()
    {
        // mm tiling
        uint32_t sOut = tilingData.splitCoreParams.get_sOut();
        uint32_t sFla = tilingData.splitCoreParams.get_sFla();
        uint32_t tailCoreSOut = tilingData.tailSplitCoreParams.get_sOut();
        uint32_t tailCoreSFla = tilingData.tailSplitCoreParams.get_sFla();

        uint32_t scmSize = sOut * sFla * mBaseParams.dataTypeSize * mBaseParams.dataTypeSize;
        uint32_t tailCoreScmSize = tailCoreSOut * tailCoreSFla * mBaseParams.dataTypeSize * mBaseParams.dataTypeSize;

        uint32_t l1SizeRemain = mBaseParams.l1Size - scmSize;
        uint32_t tailCorel1SizeRemain = mBaseParams.l1Size - tailCoreScmSize;

        matmul_tiling::MatmulApiTiling mm1;
        matmul_tiling::MatmulApiTiling mm2;
        matmul_tiling::MatmulApiTiling mm3;
        matmul_tiling::MatmulApiTiling tailCoreMm1;
        matmul_tiling::MatmulApiTiling tailCoreMm2;
        SetMatmulTiling(mm1, mm2, sOut, sFla, l1SizeRemain, tilingData.mm1TilingData, tilingData.mm2TilingData);
        SetKvMatmulTiling(mm3, sOut, sFla, l1SizeRemain, tilingData.mm3TilingData);
        SetMatmulTiling(tailCoreMm1, tailCoreMm2, tailCoreSOut, tailCoreSFla, tailCorel1SizeRemain,
                        tilingData.mm1TilingDataTailCore, tilingData.mm2TilingDataTailCore);

        auto softmaxShape = Shape({tilingData.splitCoreParams.get_sOut(), tilingData.splitCoreParams.get_sFla()});
        auto softmaxShapeTail =
            Shape({tilingData.tailSplitCoreParams.get_sOut(), tilingData.tailSplitCoreParams.get_sFla()});

        SoftMaxTilingFunc(softmaxShape, sizeof(float), tilingData.splitCoreParams.get_innerMatResSize(),
                          tilingData.softmaxTilingData);
        SoftMaxGradTilingFunc(softmaxShape, mBaseParams.dataTypeSize, tilingData.splitCoreParams.get_innerMatResSize(),
                              tilingData.softmaxGradTilingData, true);
        // transpose tiling
        auto transposeSrcShape = Shape({1, 1, sOut, mBaseParams.d});
        auto transposeDstShape = Shape({mBaseParams.b, mBaseParams.n, mBaseParams.s, mBaseParams.d * mBaseParams.n});
        GetDataCopyTransposeTiling(transposeDstShape, transposeSrcShape, mBaseParams.dataTypeSize,
                                   tilingData.copyTransposeTiling);

        return ge::GRAPH_SUCCESS;
    }

    void SetMatmulTiling(matmul_tiling::MatmulApiTiling& mm1, matmul_tiling::MatmulApiTiling& mm2, const uint32_t& sOut,
                         const uint32_t& sFla, const uint32_t& l1SizeRemain, TCubeTiling& mm1Tiling,
                         TCubeTiling& mm2Tiling)
    {
        mm1.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                     false);
        mm1.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                     true);
        mm1.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT16);
        mm1.SetShape(tilingData.splitCoreParams.get_splitedCoreSeqNum(), sFla, mBaseParams.d);
        if (mBaseParams.layoutSBH == 0) {
            mm1.SetOrgShape(mBaseParams.s, mBaseParams.skv, mBaseParams.d * mBaseParams.n);
        } else if (mBaseParams.layoutSBH == 1) {
            mm1.SetOrgShape(mBaseParams.s, mBaseParams.skv, mBaseParams.b * mBaseParams.d * mBaseParams.n);
        } else if (mBaseParams.layoutSBH == 2) {
            mm1.SetOrgShape(mBaseParams.s, mBaseParams.skv, mBaseParams.d);
        }
        mm1.SetBias(false);
        mm1.SetBufferSpace(l1SizeRemain, mBaseParams.l0cSize);
        mm1.SetFixSplit(sOut, sFla, -1);
        mm1.GetTiling(mm1Tiling);

        mm1Tiling.set_shareMode(0);
        mm1Tiling.set_shareL1Size(l1SizeRemain);
        mm1Tiling.set_shareL0CSize(mBaseParams.l0cSize);

        mm2.SetAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                     true);
        mm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                     false);
        mm2.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT16);
        mm2.SetShape(sOut, mBaseParams.d, sFla);

        if (mBaseParams.layoutSBH == 0) {
            mm2.SetOrgShape(mBaseParams.s, mBaseParams.d * mBaseParams.n, mBaseParams.skv);
        } else if (mBaseParams.layoutSBH == 1) {
            mm2.SetOrgShape(mBaseParams.s, mBaseParams.b * mBaseParams.d * mBaseParams.n, mBaseParams.skv);
        } else if (mBaseParams.layoutSBH == 2) {
            mm2.SetOrgShape(mBaseParams.s, mBaseParams.d, mBaseParams.skv);
        }

        if (mBaseParams.tilingKey == SPECIAL_BN_HIGH_PRECISE_TILING_KEY ||
            mBaseParams.tilingKey == SPECIAL_BNS_BF16_TILING_KEY ||
            mBaseParams.tilingKey == SPECIAL_SPARSE_BNS_HIGH_PRECISE_TILING_KEY ||
            mBaseParams.tilingKey == SPECIAL_BNS_BF16_96_96_TILING_KEY) {
            mm2.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT16, true);
            mm2.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT16, false);
            mm2.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                         matmul_tiling::DataType::DT_FLOAT16);
            mm2.SetShape(sOut * CUBE_POSTLOAD_BLOCK_NUM, mBaseParams.d, sFla * CUBE_POSTLOAD_BLOCK_NUM);

            if (mBaseParams.layoutSBH == 0) {
                mm2.SetOrgShape(tilingData.splitCoreParams.get_sOut() * CUBE_POSTLOAD_BLOCK_NUM,
                                mBaseParams.d * mBaseParams.n, tilingData.splitCoreParams.get_sFla(), mBaseParams.skv);
            } else if (mBaseParams.layoutSBH == 1) {
                mm2.SetOrgShape(tilingData.splitCoreParams.get_sOut() * CUBE_POSTLOAD_BLOCK_NUM,
                                mBaseParams.b * mBaseParams.d * mBaseParams.n, tilingData.splitCoreParams.get_sFla(),
                                mBaseParams.skv);
            } else if (mBaseParams.layoutSBH == 2) {
                mm2.SetOrgShape(tilingData.splitCoreParams.get_sOut() * CUBE_POSTLOAD_BLOCK_NUM, mBaseParams.d,
                                tilingData.splitCoreParams.get_sFla(), mBaseParams.skv);
            }
        }

        mm2.SetBias(false);
        mm2.SetBufferSpace(l1SizeRemain, mBaseParams.l0cSize);
        mm2.SetFixSplit(sOut, mBaseParams.d, -1);
        mm2.GetTiling(mm2Tiling);

        mm2Tiling.set_shareMode(0);
        mm2Tiling.set_shareL1Size(l1SizeRemain);
        mm2Tiling.set_shareL0CSize(mBaseParams.l0cSize);
    }

    void SetKvMatmulTiling(matmul_tiling::MatmulApiTiling& mm3, const uint32_t& sOut, const uint32_t& sFla,
                           const uint32_t& l1SizeRemain, TCubeTiling& mm3Tiling)
    {
        mm3.SetAType(matmul_tiling::TPosition::TSCM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                     true);
        mm3.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16,
                     false);
        mm3.SetCType(matmul_tiling::TPosition::VECCALC, matmul_tiling::CubeFormat::ND,
                     matmul_tiling::DataType::DT_FLOAT16);
        mm3.SetShape(sFla, mBaseParams.d, sOut);
        if (mBaseParams.layoutSBH == 0) {
            mm3.SetOrgShape(mBaseParams.skv, mBaseParams.d * mBaseParams.n, mBaseParams.s);
        } else if (mBaseParams.layoutSBH == 1) {
            mm3.SetOrgShape(mBaseParams.skv, mBaseParams.b * mBaseParams.d * mBaseParams.n, mBaseParams.s);
        } else if (mBaseParams.layoutSBH == 2) {
            mm3.SetOrgShape(mBaseParams.skv, mBaseParams.d, mBaseParams.s);
        }
        mm3.SetBias(false);
        mm3.SetBufferSpace(l1SizeRemain, mBaseParams.l0cSize);
        mm3.SetFixSplit(sFla, mBaseParams.d, -1);
        mm3.GetTiling(mm3Tiling);
        mm3Tiling.set_shareMode(0);
        mm3Tiling.set_shareL1Size(l1SizeRemain);
        mm3Tiling.set_shareL0CSize(mBaseParams.l0cSize);
    }

    ge::graphStatus SaveTilingData(gert::TilingContext* context)
    {
        auto aicNum = 40; // 40: B3 default aicNum
        auto aivNum = 20; // 20: B3 default aivNum
        auto platformInfoPtr = context->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compilePtr = reinterpret_cast<const FlashAttentionScoreGradCompileInfo*>(context->GetCompileInfo());
            OP_CHECK_IF(compilePtr == nullptr, TILING_INNER_ERR_REPORT(context->GetNodeName(), "compile_info is null"),
                        return ge::GRAPH_FAILED);
            aivNum = compilePtr->aivNum;
            aicNum = compilePtr->aicNum;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            aicNum = ascendcPlatform.GetCoreNumAic();
            aivNum = ascendcPlatform.GetCoreNumAiv();
        }
        context->SetBlockDim(CalculateTschBlockDim(tilingData.baseParams.get_coreNum(), aicNum, aivNum));
        context->SetScheduleMode(1); // 1: batchmode模式

        size_t* workspaces = context->GetWorkspaceSizes(1);
        workspaces[0] =
            100 * 1024 * 1024 /* 100MB = 100 * 1024KB = 100 * 1024 * 1024 B */
            + tilingData.splitCoreParams.get_innerMatResSize() * tilingData.baseParams.get_coreNum() +
            tilingData.splitCoreParams.get_splitedCoreSeqNum() * tilingData.splitCoreParams.get_sFla() *
                tilingData.baseParams.get_coreNum() * 4 +
            tilingData.baseParams.get_B() *
                (tilingData.baseParams.get_Sq() + tilingData.baseParams.get_Skv() + tilingData.baseParams.get_Skv()) *
                tilingData.baseParams.get_H() * sizeof(float) +
            tilingData.baseParams.get_coreNum() * CUBE_POSTLOAD_BLOCK_NUM * 2 *
                tilingData.splitCoreParams.get_outMatResNum() * SIZT_OF_HALF * QK_V_BMM_NUM; // 4 is min core num
        tilingData.baseParams.set_apiClcQueueSize(mBaseParams.apiClcQueueSize);
        tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }

    ge::graphStatus CastTilingFunc()
    {
        uint32_t baseOutNum = 60 * 1024 / 2 / 2;
        uint32_t castUsedUbSize = baseOutNum * 2 * 3;

        // query
        uint32_t allNumQuery =
            tilingData.baseParams.get_B() * tilingData.baseParams.get_Sq() * tilingData.baseParams.get_H();
        uint32_t singleCoreNumQuery = CeilFunc(allNumQuery / 16, tilingData.baseParams.get_coreNum()) * 16;
        uint32_t tailCoreNumQuery = allNumQuery - singleCoreNumQuery * (tilingData.baseParams.get_coreNum() - 1);

        uint32_t singleCoreLoopTimesQuery = CeilFunc(singleCoreNumQuery / 16, baseOutNum / 16);
        uint32_t lastLoopNumQuery = singleCoreNumQuery - (singleCoreLoopTimesQuery - 1) * baseOutNum;

        uint32_t tailCoreLoopTimesQuery = CeilFunc(tailCoreNumQuery / 16, baseOutNum / 16);
        uint32_t tailCoreLastLoopNumQuery = tailCoreNumQuery - (tailCoreLoopTimesQuery - 1) * baseOutNum;

        // K V
        uint32_t allNumKv =
            tilingData.baseParams.get_B() * tilingData.baseParams.get_Skv() * tilingData.baseParams.get_H();
        uint32_t singleCoreNumKv = CeilFunc(allNumKv / 16, tilingData.baseParams.get_coreNum()) * 16;
        uint32_t tailCoreNumKv = allNumKv - singleCoreNumKv * (tilingData.baseParams.get_coreNum() - 1);

        uint32_t singleCoreLoopTimesKv = CeilFunc(singleCoreNumKv / 16, baseOutNum / 16);
        uint32_t lastLoopNumKv = singleCoreNumKv - (singleCoreLoopTimesKv - 1) * baseOutNum;

        uint32_t tailCoreLoopTimesKv = CeilFunc(tailCoreNumKv / 16, baseOutNum / 16);
        uint32_t tailCoreLastLoopNumKv = tailCoreNumKv - (tailCoreLoopTimesKv - 1) * baseOutNum;

        tilingData.castTiling.set_baseOutNum(baseOutNum);
        tilingData.castTiling.set_castUsedUbSize(castUsedUbSize);

        tilingData.castTiling.set_allNumQuery(allNumQuery);
        tilingData.castTiling.set_singleCoreNumQuery(singleCoreNumQuery);
        tilingData.castTiling.set_tailCoreNumQuery(tailCoreNumQuery);
        tilingData.castTiling.set_singleCoreLoopTimesQuery(singleCoreLoopTimesQuery);
        tilingData.castTiling.set_lastLoopNumQuery(lastLoopNumQuery);
        tilingData.castTiling.set_tailCoreLoopTimesQuery(tailCoreLoopTimesQuery);
        tilingData.castTiling.set_tailCoreLastLoopNumQuery(tailCoreLastLoopNumQuery);

        tilingData.castTiling.set_allNumKv(allNumKv);
        tilingData.castTiling.set_singleCoreNumKv(singleCoreNumKv);
        tilingData.castTiling.set_tailCoreNumKv(tailCoreNumKv);
        tilingData.castTiling.set_singleCoreLoopTimesKv(singleCoreLoopTimesKv);
        tilingData.castTiling.set_lastLoopNumKv(lastLoopNumKv);
        tilingData.castTiling.set_tailCoreLoopTimesKv(tailCoreLoopTimesKv);
        tilingData.castTiling.set_tailCoreLastLoopNumKv(tailCoreLastLoopNumKv);

        return ge::GRAPH_SUCCESS;
    }

private:
    std::map<TilingKeyType, std::function<ge::graphStatus(BaseInfoParams)>> mTilingFuncMap;
    BaseInfoParams mBaseParams;
};

bool IsEmptyOutput(gert::TilingContext* context)
{
    const gert::StorageShape* dqShape = context->GetOutputShape(OUTPUT_DQ_INDEX);
    const gert::StorageShape* dkShape = context->GetOutputShape(OUTPUT_DK_INDEX);
    const gert::StorageShape* dvShape = context->GetOutputShape(OUTPUT_DV_INDEX);
    if (dqShape != nullptr && dkShape != nullptr && dvShape != nullptr) {
        if (dqShape->GetStorageShape().GetShapeSize() == 0 || dkShape->GetStorageShape().GetShapeSize() == 0 ||
            dvShape->GetStorageShape().GetShapeSize() == 0) {
            return true;
        }
    }
    return false;
}

ge::graphStatus TilingFlashAttentionGrad(gert::TilingContext* context)
{
    if (IsEmptyOutput(context)) {
        FlashAttentionScoreGradTiling flashAttentionScoreGradTiling;
        return flashAttentionScoreGradTiling.RunEmptyTiling(context);
    } else {
        return TilingRegistry::GetInstance().DoTilingImpl(context);
    }
}

IMPL_OP_OPTILING(FlashAttentionGrad).Tiling(TilingFlashAttentionGrad);

} // namespace optiling
