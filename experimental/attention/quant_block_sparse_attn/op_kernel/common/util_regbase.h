/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file util_regbase.h
 * \brief
 */

#ifndef FLASH_ATTENTION_UTIL_REGBASE_H
#define FLASH_ATTENTION_UTIL_REGBASE_H

#include "util.h"

using AscendC::QuePosition;
using AscendC::TQue;

namespace regbaseutil {
constexpr uint16_t regBytes = 256;
constexpr int64_t MAX_PRE_NEXT_TOKENS = 0x7FFFFFFF;
constexpr float SINK_MIN_INF = -3.40E+38;
enum class VselrIndexEnum {
    GT_64_AND_LTE_128_INDEX = 0,
    GT_0_AND_LTE_64_INDEX = 1,
    DN_INDEX = 2,
    NZ_INDEX = 3
};
enum class DTemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned48 = 48,
    Aligned64 = 64,
    Aligned72 = 72,
    Aligned80 = 80,
    Aligned96 = 96,
    Aligned128 = 128,
    Aligned160 = 160,
    Aligned192 = 192,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned576 = 576,
    Aligned768 = 768,
    NotAligned,
};

enum class S1TemplateType {
    Aligned16 = 16,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    NotAligned,
};

enum class S2TemplateType {
    Aligned16 = 16,
    Aligned32 = 32,
    Aligned64 = 64,
    Aligned128 = 128,
    Aligned256 = 256,
    Aligned512 = 512,
    Aligned1024 = 1024,
    NotAligned,
};

enum class SparseType : uint8_t {
    DENSE = 0,
    CASUAL = 1,
    BAND = 2,
    UNSUPPORTED = 3 // The optimization for sizes exceeding L2 does not yet support sparse inputs.
};
template <bool isInfer = false>
struct RunParamStr;

#define COMMON_RUN_PARAM \
    int64_t boIdx; \
    int64_t s1oIdx; \
    int64_t n2oIdx; \
    int64_t goIdx; \
    int32_t s2LoopEndIdx;       /* S2 loop control information, determined at the souter level. */ \
    int64_t s2LineStartIdx = 0; /* Starting row position along S2. */ \
    int64_t s2LineEndIdx;       /* Ending row position along S2. */ \
    /* Cube's sOuter; for SAMEAB, cubeSOuterSize is twice halfS1RealSize, determined at the souter level. */ \
    uint32_t s1RealSize; \
    uint32_t s1RealSizeAlign32; /* Used in the DN case. */ \
    uint32_t s1RealSizeAlign64; /* Used in the DN case. */ \
    uint32_t halfS1RealSize; \
    uint32_t firstHalfS1RealSize; \
    int64_t tensorQOffset;      /* Query offset, determined at the souter level. */ \
    int64_t attentionOutOffset; /* attentionOut offset, determined at the souter level. */ \
    int64_t actualS1Size;       /* actualSeqLength of Q. */ \
    int64_t actualS2Size;       /* actualSeqLength of KV. */ \
    uint64_t b1SSOffset; \
    uint64_t b1SSAttenMaskOffset; \
    uint64_t b1SSOffsetAlign16; \
    int64_t qRopeNBGOffset; /* QueryRope offset. */ \
    int64_t kRopeNBGOffset; /* KeyRope offsets for different g values along G. */

template <>
struct RunParamStr<false> { // Parameters required for core partitioning and tiling.
    COMMON_RUN_PARAM;
};

template <>
struct RunParamStr<true> { // Parameters required for core partitioning and tiling.
    COMMON_RUN_PARAM;
    /* Added for inference. */
    int64_t s1LoopTimes;
    int64_t gS1Idx;
    // Data produced by the BN loop.
    int64_t s2InCurrentBatch; // For tensor lists, S2 lengths differ across batches; later used to calculate KvStride.
    int64_t preTokensPerBatch = MAX_PRE_NEXT_TOKENS;  // pretoken at the upper-left corner.
    int64_t nextTokensPerBatch = MAX_PRE_NEXT_TOKENS; // nexttoken at the upper-left corner.

    // Data produced by the NBS1 loop.
    int64_t sOuterOffset;        // souterIdx * halfS1RealSize within one S, determined at the souter level.
    int64_t cubeSOuterOffset;    // souterIdx * halfS1RealSize within one S, determined at the souter level.
    int64_t keyCoreOffset;       // Key offsets for different BN values, determined at the batch level.
    int64_t valueCoreOffset;     // Value offsets for different BN values, determined at the batch level.
    uint64_t pseShiftCoreOffset; // pseShift offsets for different souter values along Souter.
    int64_t keyOffset;           // MM1 Key offset; to be renamed KFinalOffset.

    // Offsets along different axes for Q/K/V/attenMask.
    // B-axis offset.
    int64_t qBOffset;     // bIdx * seqSize * multiHeadQ; to be renamed qBOffset, determined at the batch level.
    int64_t qRopeBOffset; // Determined at the batch level; related to actualSeqLengths.

    // Left padding.
    int64_t queryLeftPaddingSize; // Determined at the batch level.
    int64_t kvLeftPaddingSize;    // Determined at the batch level.

    // LSE output offset.
    int64_t softmaxLseOffset; // Determined at the souter level.

    // IFA_MLA
    int64_t actualSeqLengthOfMlaPerBatch = 0; // actualSeqLength of Q for MLA.
    int64_t nextTokensOfMlaPerBatch = 0;      // Upper-left nexttoken for MLA, used to calculate invalid BNSD rows.
    int64_t preTokensOfMlaPerBatch = 0;       // Upper-left nexttoken for MLA, used to calculate invalid BNSD rows.

    // prefix
    int64_t prefixCoreOffset = 0; // Store the prefix address offset along BN for the current loop.
};

#define COMMON_RUN_INFO \
    int64_t s2StartIdx; /* S2 start, which may be nonzero for sparse inputs. */ \
    int64_t s2EndIdx; \
    int64_t s2LoopCount; /* Current S2 loop index. */ \
    int64_t s2LoopLimit; \
    int64_t s1oIdx = 0; /* S1-axis index. */ \
    int64_t boIdx = 0;  /* B-axis index. */ \
    int64_t n2oIdx = 0; /* N2-axis index. */ \
    int64_t goIdx = 0;  /* G-axis index. */ \
    int32_t s1RealSize; \
    int32_t s1RealSizeAlign32; \
    int32_t s1RealSizeAlign64; \
    int32_t halfS1RealSize; /* Actual Vector-side S1 basic block size; halfS1RealSize=64 when the Cube basic block \
                               size is 128. */ \
    int32_t firstHalfS1RealSize; /* When s1RealSize is odd, V0 processes one fewer row than V1; use V0's S1 when \
                                    calculating the subBlock offset. size */ \
    int32_t s2RealSize;          /* Actual basic block length along S2. */ \
    int64_t s2AlignedSize;       /* Basic block length along S2 aligned to 16. */ \
    int32_t vec2S1BaseSize;      /* S1 size after splitting in the vector2 loop, e.g., split 64 into 32 and 32. */ \
    int32_t vec2S1RealSize; /* S1 tail size after splitting in the vector2 loop, e.g., split 63 into 32 and 31; the \
                             * second part has 31 rows. \
                             */ \
    int64_t vecCoreOffset;  /* Vector core's S1 offset relative to the Cube core's start. */ \
    int64_t queryOffset;    /* MM1 Query offset.*/ \
    int64_t keyOffset;      /* MM1 Key offset. */ \
    int64_t valueOffset;    /* MM2 Value offset.*/ \
    int64_t qRopeOffset; \
    int64_t kRopeOffset; \
\
    int64_t taskId; \
    int64_t multiCoreInnerIdx = 0; \
\
    int64_t attentionOutOffset; \
    uint64_t s1ScaleNumAcc; \
    uint64_t s2ScaleNumAcc; \
    int64_t s1SizeAcc;          /* For non-TND: boIdx * pseInfo.s2Size; \
                                   For TND: sum of S1 lengths of the preceding boIdx batches; S1 differs across batches. */ \
    int64_t s2SizeAcc;          /* For non-TND: boIdx * pseInfo.s2Size; \
                                   For TND: sum of S2 lengths of the preceding boIdx batches; S2 differs across batches.*/ \
    int64_t actualS1Size;       /* Non-TND: total s1Size; TND: S1 of the current batch. */ \
    int64_t actualS2Size;       /* Non-TND: total s2Size; TND: S2 of the current batch. */ \
    int64_t preTokensPerBatch;  /* vector2 upper-left pretoken. */ \
    int64_t nextTokensPerBatch; /* vector2 upper-left nexttoken. */ \
    int64_t b1SSOffset;         /* Non-TND: boIdx * s1 * s2; TND: sum of S1*S2 for the preceding boIdx batches. */ \
    /* Training: equal to b1SSOffset. Inference: non-TND masks may exceed s1*s2; TND masks are padded. */ \
    int64_t b1SSAttenMaskOffset; \
    int64_t b1SSOffsetAlign; /* For TND, sum S1*S2 of preceding batches after aligning S2 to 16. */ \
    int64_t deScaleKvOffset; /* GM offset of KV dequantization scales; original shape [B,N2,1,Ceil(S2,128),1]. */ \
    int64_t nextTokensOfMlaPerBatch = 0; /* Upper-left nexttoken for MLA, used to calculate invalid BNSD rows. */ \
    int64_t preTokensOfMlaPerBatch = 0;  /* Upper-left nexttoken for MLA, used to calculate invalid BNSD rows. */ \
    uint8_t taskIdMod2; \
    uint8_t taskIdMod3; \
    uint8_t multiCoreIdxMod2 = 0; \
    uint8_t multiCoreIdxMod3 = 0; \
    int64_t sOuterOffset

template <bool isInfer = false>
struct RunInfo;

template <>
struct RunInfo<true> {
    COMMON_RUN_INFO;
    // Added for inference.
    int64_t gS1Idx;
    uint64_t pseShiftOffset; // vector1 PSE offset.
    int64_t queryLeftPaddingSize;
    int64_t kvLeftPaddingSize;
    int64_t actualSeqLengthOfMlaPerBatch = 0; // actualSeqLength of Q for MLA.
    // LSE output offset.
    int64_t softmaxLseOffset;

    // Related to FD.
    int64_t flashDecodeS2Idx;

    // Related to tensor lists.
    int64_t s2InCurrentBatch;

    // Related to prefixes.
    int64_t prefixOffset; // Store the prefix address offset for the current loop.
};

template <>
struct RunInfo<false> {
    COMMON_RUN_INFO;
};

#define COMMON_CONST_INFO \
    /* Global basic block information. */ \
    uint32_t s1BaseSize; \
    uint32_t s2BaseSize; \
    int64_t bSize; \
    int64_t t1Size; \
    int64_t t2Size; \
    int64_t dSize; \
    int64_t dSizeV; \
    int64_t dBasicBlock; \
    int64_t dSizeRope; \
    int64_t gSize; /* Size of the g axis. */ \
    int64_t n2Size; \
    int64_t s1Size; /* Total S1 size. */ \
    int64_t s2Size; /* Total S2 size. */ \
    /* Product of axis sizes. */ \
    int64_t s1D; \
    int64_t gS1D; \
    int64_t n2GS1D; \
    int64_t s2D; \
    int64_t n2S2D; \
    int64_t s1Dv; \
    int64_t gS1Dv; \
    int64_t n2GS1Dv; \
    int64_t s2Dv; \
    int64_t n2S2Dv; \
    int64_t s1S2; \
    int64_t gS1; \
    int64_t gD; \
    int64_t n2D; \
    int64_t bN2D; \
    int64_t gDv; \
    int64_t n2Dv; \
    int64_t bN2Dv; \
    int64_t n2G; \
    int64_t n2GD; \
    int64_t bN2GD; \
    int64_t n2GDv; \
    int64_t bN2GDv; \
    int64_t gS2; \
    int64_t s1Dr; \
    int64_t gS1Dr; \
    int64_t n2GS1Dr; \
    int64_t s2Dr; \
    int64_t n2S2Dr; \
    int64_t gDr; \
    int64_t n2Dr; \
    int64_t bN2Dr; \
    int64_t n2GDr; \
    int64_t bN2GDr; \
    int32_t s2BaseN2D; \
    int32_t s1BaseN2GD; \
    int64_t s2BaseBN2D; \
    int64_t s1BaseBN2GD; \
    int32_t s1BaseD; \
    int32_t s2BaseD; \
    int64_t s2BaseN2Dv; \
    int64_t s2BaseBN2Dv; \
    int64_t s1BaseN2GDv; \
    int64_t s1BaseBN2GDv; \
    int32_t s1BaseDv; \
    int32_t s2BaseDv; \
    int64_t s1OuterSize; \
    /* Matmul parameters for strided reads. */ \
    int64_t mm1Ka; \
    int64_t mm1Kb; \
    int64_t mm2Kb; \
    /* Stride of dq or attentionOut. */ \
    int64_t attentionOutStride; \
    uint32_t aivIdx; \
    uint8_t layoutType; \
    uint8_t subBlockIdx; \
    bool softMaxCheckRes; \
    float keepProb; \
    float scaleValue; \
    int64_t matmulMSize; /* For matmul, distinguish the left matrix M size with and without GS1 axis merging. */ \
    bool learnableSinkFlag = false; /* attentionsink */ \
    float pScale; \
    float sinkValue = SINK_MIN_INF; \
    /* strides */ \
    uint64_t keyBnStride; \
    uint64_t keyN2Stride; \
    uint64_t valueBnStride; \
    uint64_t valueN2Stride; \
    uint64_t keyRopeBnStride; \
    uint64_t keyRopeN2Stride; \
    uint64_t kScaleBnStride; \
    uint64_t kScaleN2Stride; \
    uint64_t vScaleBnStride; \
    uint64_t vScaleN2Stride; \
    uint64_t kOffsetBnStride; \
    uint64_t kOffsetN2Stride; \
    uint64_t vOffsetBnStride; \
    uint64_t vOffsetN2Stride

#define ROPE_INFO \
    /* RoPE parameters. */ \
    int64_t s1DR; \
    int64_t gS1DR; \
    int64_t n2GS1DR; \
    int64_t s2DR; \
    int64_t n2S2DR; \
    int64_t gDR; \
    int64_t n2DR; \
    int64_t bN2DR; \
    int64_t n2GDR; \
    int64_t bN2GDR; \
    int64_t s2BaseN2DR; \
    int64_t s2BaseBN2DR; \
    int64_t s1BaseN2GDR; \
    int64_t s1BaseBN2GDR; \
    int64_t s1BaseDR; \
    int64_t s2BaseDR; \
    int64_t mm1RopeKa; /* RoPE matmul parameters for strided reads.*/ \
    int64_t mm1RopeKb

#define KVPREFIX_INFO \
    /* Prefix parameters. */ \
    bool isActualSharedPrefixLenNull = true; \
    int64_t actualKVPrefixSize = 0; /* Store the actual prefix length. */ \
    int64_t kvPrefixSize = 0;       /* Store the full prefix shape length. */ \
    int64_t prefixLoopCount = 0     /* Store the number of S2 loop iterations involving the prefix. */

#define INFER_CONST_INFO \
    /* Added for inference. */ \
    bool isRowInvalid;          /* Whether invalid rows are enabled. */ \
    bool isActualLenDimsNull;   /* Whether actualseq is present. */ \
    bool isActualLenDimsKVNull; /* Whether actualseq_kv is present. */ \
    bool isGqa; \
    bool isPfaGS1Merge; /* Whether PFA GS1 axes are merged. */ \
\
    uint32_t actualSeqLenSize;   /* Length of the user-provided actualseq. */ \
    uint32_t actualSeqLenKVSize; /* Length of the user-provided actualseq_kv. */ \
    uint32_t isKvContinuous;     /* Whether the input is a tensor list. */ \
    /* service mm1 mm2 pageAttention */ \
    uint32_t blockTableDim2; \
    uint32_t blockSize; \
    uint32_t paLayoutType; \
    uint32_t kvCacheNzD0; \
    uint32_t paBlockNumSum; \
    uint32_t transposeLayout; \
    /* With GS1 axis merging, outer loops cover B/N2 and inner loops cover G/S1; headNumRatio=1. */ \
    /* Without GS1 axis merging, outer loops cover B/N2/G and inner loops cover S1; headNumRatio=gSize. */ \
    uint32_t headNumRatio; \
    bool rsvd1; \
    bool isSoftmaxLseEnable; \
    /* Left padding. */ \
    bool isQHasLeftPadding; \
    bool isKVHasLeftPadding; \
    int64_t queryRightPaddingSize; \
    int64_t kvRightPaddingSize; \
    /* FD */ \
    int64_t sInnerLoopSize;        /* Total FD S2 size. */ \
    int64_t actualCombineLoopSize; /* Actual number of reduction blocks. */ \
    int64_t splitKVNum; \
    /* Post-quantization. */ \
    bool isPostQuantPerChnl; \
    bool isPostQuantBF16; \
    bool isPostQuantOffsetExist; \
    float postQuantScaleValue; \
    float postQuantOffsetValue

#define CV_SHARED_PARAMS \
    /* base params */ \
    uint32_t bSize; \
    int64_t t1Size; \
    int64_t t2Size; \
    uint32_t n2Size; \
    uint32_t gSize; \
    uint32_t s1Size; \
    uint32_t s2Size; \
    uint32_t dSize : 16; \
    uint32_t dSizeV : 16; \
    /* special params */ \
    int64_t preTokens; \
    int64_t nextTokens; \
    uint32_t attenMaskS1Size; \
    uint32_t attenMaskS2Size; \
    int64_t s1SparseValidSize; \
    int64_t s2SparseValidSize; \
    /* core params */ \
    volatile int64_t multiCoreInnerOffset; /* Variables assigned a second time must be volatile. */ \
    volatile int64_t multiCoreInnerLimit;  /* Variables assigned a second time must be volatile. */ \
    uint32_t s1OuterSize; \
    uint32_t bandIndex; \
    uint32_t compressMode : 4; \
    uint32_t implMode : 4; \
    uint32_t layoutType : 4; \
    uint32_t sparseType : 8; \
    uint32_t dSizeRope : 11; \
    uint32_t splitCoreMode : 1; \
    uint32_t coreNum

#define FAG_CV_SHARED_PARAMS \
    /* base params */ \
    float qScaleDs

struct FagCVSharedParams {
    FAG_CV_SHARED_PARAMS;
};

template <bool isInfer = false, bool hasRope = false>
struct ConstInfo;

template <>
struct ConstInfo<true, true> {
    COMMON_CONST_INFO;
    INFER_CONST_INFO;
    ROPE_INFO;
    KVPREFIX_INFO;
};

template <>
struct ConstInfo<true, false> {
    COMMON_CONST_INFO;
    INFER_CONST_INFO;
    KVPREFIX_INFO;
};

template <>
struct ConstInfo<false, true> {
    COMMON_CONST_INFO;
    ROPE_INFO;
    int64_t n2GS1o; // Training only.
    int64_t gS1o;
};

template <>
struct ConstInfo<false, false> {
    COMMON_CONST_INFO;
    int64_t n2GS1o; // Training only.
    int64_t gS1o;
};

/* only support b32 or b64 */
template <bool isInfer = false, bool isPa = false>
struct CVSharedParams;

template <>
struct CVSharedParams<false, false> {
    CV_SHARED_PARAMS;
    int64_t firstFullLoadS1OuterIdx;
    int64_t totalSize;
    float scaleValue;
};

/* CVSharedParams must fit within one cache line: 128 bytes. */
template <>
struct CVSharedParams<true, false> {
    CV_SHARED_PARAMS;
    uint32_t fromFused : 1;
    uint32_t isGqa : 1;
    uint32_t isPfaGS1Merge : 1;
    uint32_t isKvContinuous : 1;
    uint32_t isRowInvalid : 1;
    uint32_t isActualSeqLengthsNull : 1;
    uint32_t isActualSeqLengthsKVNull : 1;
    uint32_t isQHasLeftPadding : 1;
    uint32_t isKVHasLeftPadding : 1;
    uint32_t needInit : 1;
    uint32_t isPostQuantPerChnl : 1;
    uint32_t isPostQuantBF16 : 1;
    uint32_t headNumRatio : 20;

    uint32_t transposeLayout;
    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;
    uint32_t splitKVNum;

    uint32_t bnStartIdx;
    uint32_t bnEndIdx;

    uint32_t queryRightPaddingSize;
    uint32_t kvRightPaddingSize;

    // prefix
    bool isActualSharedPrefixLenNull;
    int64_t kvPrefixSize;

    int64_t totalSize;
};

template <>
struct CVSharedParams<true, true> {
    CV_SHARED_PARAMS;
    uint32_t fromFused : 1;
    uint32_t isGqa : 1;
    uint32_t isPfaGS1Merge : 1;
    uint32_t isKvContinuous : 1;
    uint32_t isRowInvalid : 1;
    uint32_t isActualSeqLengthsNull : 1;
    uint32_t isActualSeqLengthsKVNull : 1;
    uint32_t isQHasLeftPadding : 1;
    uint32_t isKVHasLeftPadding : 1;
    uint32_t needInit : 1;
    uint32_t isPostQuantPerChnl : 1;
    uint32_t isPostQuantBF16 : 1;
    uint32_t headNumRatio : 20;

    uint32_t transposeLayout;
    uint32_t actualSeqLengthsSize;
    uint32_t actualSeqLengthsKVSize;
    uint32_t splitKVNum;

    uint32_t bnStartIdx;
    uint32_t bnEndIdx;

    uint32_t queryRightPaddingSize;
    uint32_t kvRightPaddingSize;

    int32_t blockSize;
    int32_t blockTableDim2;
    int32_t paBlockNumSum;
    uint32_t paLayoutType;

    uint64_t keyBnStride;
    uint64_t keyN2Stride;
    uint64_t valueBnStride;
    uint64_t valueN2Stride;
    uint64_t keyRopeBnStride;
    uint64_t keyRopeN2Stride;
    uint64_t kScaleBnStride;
    uint64_t kScaleN2Stride;
    uint64_t vScaleBnStride;
    uint64_t vScaleN2Stride;
    uint64_t kOffsetBnStride;
    uint64_t kOffsetN2Stride;
    uint64_t vOffsetBnStride;
    uint64_t vOffsetN2Stride;

    // prefix
    bool isActualSharedPrefixLenNull;
    int64_t kvPrefixSize;

    int64_t totalSize;
};
} // namespace regbaseutil

#endif // FLASH_ATTENTION_UTIL_REGBASE_H
