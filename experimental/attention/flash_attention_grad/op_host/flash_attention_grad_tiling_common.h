/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <sstream>

namespace optiling {

constexpr uint32_t BYTE_BLOCK = 32;              // 32 B in block
constexpr uint32_t CORE_THRESHOLD = 32;          // 32 B in block
constexpr uint32_t FRACTAL_NUM = 16;             // 16 is 分形大小
constexpr uint32_t CUBE_ALIGN_NUM = 16;          // 16 is cube align num
constexpr uint32_t INITIAL_SOUT_SPLIT_NUM = 256; // sOut初始切分大小
constexpr uint32_t INITIAL_SFLA_SPLIT_NUM = 256; // sFla初始切分大小
constexpr uint32_t SOFTMAX_REMAIN_SIZE = 8 * 1024;
constexpr uint32_t API_RSDV_BUFFER_SIZE = 4 * 1024;
constexpr int64_t SINGLE_VEC_INST_DATASIZE = 256;
constexpr uint32_t DEFAULT_DATA_TYPE_SIZE = 4;
constexpr uint32_t DEFAULT_MASK = 64;
constexpr uint32_t FP16_DATA_TYPE_SIZE = 2;
constexpr uint32_t BF16_DATA_TYPE_SIZE = 2;
constexpr uint32_t FP16_MASK = 128;
constexpr uint32_t BF16_MASK = 64;
constexpr uint32_t BATCH_MAX_SIZE = 64;
constexpr uint32_t ATTEN_MASK_COMPRESS_LIMIT = 2048;

constexpr const char* BSH_STR = "BSH";
constexpr const char* SBH_STR = "SBH";
constexpr const char* BNSD_STR = "BNSD";
constexpr const char* BSND_STR = "BSND";
constexpr const char* TND_STR = "TND";
constexpr size_t DIM_0 = 0;
constexpr size_t DIM_1 = 1;
constexpr size_t DIM_2 = 2;
constexpr size_t DIM_3 = 3;
constexpr size_t DIM_4 = 4;
constexpr size_t LAST_AXIS_IDX = 1;
constexpr size_t SEC_LAST_AXIS_IDX = 2;
constexpr uint32_t MULT_BASE = 2;

// tiling common params
constexpr uint32_t INITIAL_S1_SPLIT_NUM = 128; // to avoid repeat max value 255
constexpr uint32_t INITIAL_S2_SPLIT_NUM = 64;
constexpr uint32_t MUL_CORE_SYNC_BUFFER = 16 * 1024;

constexpr uint32_t EMPTY_TENSOR = 0;
constexpr uint32_t NORMAL_TENSOR = 1;

constexpr uint32_t MAX_BASIC_BLOCK_SIZE = 1024;
constexpr uint32_t PSE_NORMAL_SHAPE_DIM = 4;
constexpr uint32_t PSE_COMMON_MODE = 0x10;
constexpr uint32_t PSE_PERFORMANCE_MODE = 0x12;
constexpr uint32_t TEMP_BUFFER_REMAIN_SIZE = 1024 * 2;

constexpr uint32_t ATTEN_MASK_SHAPE_DIMS_0 = 0;
constexpr uint32_t ATTEN_MASK_SHAPE_DIMS_1 = 1;
constexpr uint32_t ATTEN_MASK_DIM_LENGTH_1 = 1;
constexpr uint32_t ATTEN_MASK_DIM_LENGTH_2 = 2;
constexpr uint32_t ATTEN_MASK_DIM_LENGTH_4 = 4;

constexpr uint32_t INPUT_FROAMT_BN2GS2D = 0; // BNSD
constexpr uint32_t INPUT_FROAMT_S2BN2GD = 1; // SBH
constexpr uint32_t INPUT_FROAMT_BS2N2GD = 2; // BSH  BSND
constexpr uint32_t INPUT_FROAMT_TND = 3;     // TND
constexpr uint32_t INPUT_FROAMT_AAG3BSH = 4; // qkv:BNSD dy:BSH
constexpr uint32_t INPUT_FROAMT_AAG3SBH = 5; // qkv:BNSD dy:BSH
constexpr uint32_t INPUT_DIM_0 = 0;          // BSH  BSND
constexpr uint32_t INPUT_DIM_1 = 1;
constexpr uint32_t INPUT_DIM_2 = 2;
constexpr uint32_t INPUT_DIM_3 = 3;
constexpr uint32_t INPUT_DIM_4 = 4;

constexpr uint32_t CORE_INIT_NUM = 40;
constexpr uint32_t MATMUL_SIZE = 8 * 1024;

constexpr uint32_t INPUT_ALIGN = 16;
constexpr uint32_t WORKSPACE_NUM_ALIGN = 256;
constexpr uint32_t GM_ALIGN = 512;

constexpr uint32_t SOFT_MAX_PERF = 64;

constexpr uint32_t TOTAL_BLOCK_DIMENSION = 2;
constexpr uint32_t CALCULATED_BLOCK_DIMENSION = 4;
constexpr uint32_t BEGIN_IDX = 0;
constexpr uint32_t END_IDX = 1;
constexpr uint32_t SUM_S1S2 = 2;
constexpr uint32_t SUM_ALL = 3;
constexpr uint32_t LENGTH_IDX = 2;
constexpr uint32_t basicBlockMultiple = 15;
constexpr uint32_t preCoexNode = 5;
constexpr uint32_t postCoexNode = 3;

constexpr uint32_t FP16_BYTES = 2;
constexpr uint32_t FP16_BLOCK_NUMS = 16;
constexpr uint32_t FP32_BYTES = 4;
constexpr uint32_t FP32_BLOCK_NUMS = 8;
constexpr uint32_t BOOL_BLOCK_NUMS = 32;
constexpr uint32_t SHAPE_INFO = 32;

constexpr uint32_t MATMAL_INPUT_NUMS = 2;
constexpr uint32_t S1CV_RATIO_DEFAULT = 1;
constexpr uint32_t S2CV_RATIO_DEFAULT = 8;
constexpr uint32_t CV_RATIO_2 = 2;
constexpr uint32_t CV_RATIO_4 = 4;
constexpr uint32_t CV_RATIO_16 = 16;
constexpr uint32_t WORKSPACE_BUFFER = 20 * 1024 * 1024;
constexpr uint32_t BIT_NUMS = 8;
constexpr uint32_t S2_NZ_SIZE = 128;
constexpr uint32_t ASCENDC_API_TEMP_BUFFER = 32 * 1024 + 1024; // ND2ND NEED ANOTHER 1K
constexpr uint32_t API_BOOL_ALIGN = 32;                        // ASCEND API ATTENMASK OR DROPOUT LAST DIM ALIGN

template <class T>
inline T AlignTo(const T n, const T alignSize)
{
    return (n + alignSize - 1) & (~(alignSize - 1));
}

template <class T>
inline T CeilDivideBy(T num1, T num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

enum TilingDataType {
    FP16 = 1,
    BF16 = 2,
    FP32 = 3,
    INHP
};

enum class InputLayout {
    BSH = 0,
    SBH = 1,
    BNSD = 2,
    BSND = 3,
    TND
};

enum AxisIdx {
    B = 0,
    S = 1,
    H = 2
};

enum Axis4Idx {
    AXIS4_B = B,
    AXIS4_S = S,
    AXIS4_N = 2,
    AXIS4_D = 3
};

/* layout和b,s,h三根轴的位置关系映射 */
const std::vector<std::vector<size_t>> LAYOUT_TO_AXIS{
    // 3根轴对应dimid
    {0, 1, 2}, // BSH
    {1, 0, 2}, // SBH
    // 4根轴对应dimid
    {0, 2, 1, 3}, // BNSD
    {0, 1, 2, 3}  // BSND
};

enum InputIndex {
    DY_INDEX = 0,
    QUARY_INDEX,
    KEY_INDEX,
    VALUE_INDEX,
    ATTENTION_IN_INDEX,
    SOFTMAX_MAX_INDEX,
    SOFTMAX_SUM_INDEX,
    SOFTMAX_LSE_INDEX,
    ACTUAL_SEQ_Q_LEN_INDEX,
    ACTUAL_SEQ_KV_LEN_INDEX,
    ATTEN_MASK_INDEX,
    PRM_MASK_INDEX,
    SINK_IN_INDEX
};

enum AttrIndex {
    ATTR_SCALE_VALUE_INDEX = 0,
    ATTR_PRE_TOKENS_INDEX,
    ATTR_NEXT_TOKENS_INDEX,
    ATTR_HEAD_NUM_INDEX,
    ATTR_INPUT_LAYOUT_INDEX,
    ATTR_SPARSE_MODE_INDEX,
    ACTUAL_SEQ_Q_LEN_ATTRS_INDEX,
    ACTUAL_SEQ_KV_LEN_ATTRS_INDEX,
    ATTR_SINK_REQUIRES_GRAD_INDEX,
    ATTR_DETERMINISTIC_INDEX
};

enum OutputIndex {
    OUTPUT_DQ_INDEX = 0,
    OUTPUT_DK_INDEX,
    OUTPUT_DV_INDEX
};

enum AttenDataType {
    ATTEN_MASK_TYPE_SAME = 0,   // 0 表示 AttenMask 数据类型与 qkv 一致
    ATTEN_MASK_TYPE_U8_BOOL = 1 // 1 表示 AttenMask 数据类型为 u8 bool
};

enum AttenShapeType {
    ATTEN_MASK_SHAPE_TYPE_SS,
    ATTEN_MASK_SHAPE_TYPE_B1SS,
    ATTEN_MASK_SHAPE_TYPE_BNSS
};
constexpr uint32_t ATTEN_MASK_SHAPE_TEMP_DIMS = 0; // 0 是 B1SS 及 SS差异轴索引, S不可能为 1

enum AttenMaskCompressMode {
    NO_COMPRESS_MODE = 0,
    LEFT_UP_CAUSAL_MODE,
    RIGHT_DOWN_CAUSAL_MODE,
    BAND_EQUAL_S_MODE
};

enum SparseMode {
    NO_MASK = 0, // 未传入attenmask，不做mask操作
    ALL_MASK,
    LEFT_UP_CAUSAL,        // 左上角点划分的三角部分
    RIGHT_DOWN_CAUSAL = 3, // 右下角点划分的下三角部分
    BAND = 4,
    PREFIX = 5
};

struct TempParams { // 频繁使用的中间态临时变量
    uint32_t usedUBSize;
    uint32_t tilingKey;
    uint32_t apiClcQueueSize = 0;
};

inline uint32_t Gcd(uint32_t a, uint32_t b) // a >= b
{
    if (b > a) {
        return Gcd(b, a);
    }
    if (a % b == 0) {
        return b;
    }
    return Gcd(b, a % b);
}

inline int64_t CeilCommon(int64_t num1, int64_t num2)
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

inline int64_t Align(const int64_t n)
{
    return (n + BYTE_BLOCK - 1) & (~(BYTE_BLOCK - 1));
}

inline uint32_t AlignData(const uint32_t a, const uint32_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

// fag tiling base struct
constexpr uint32_t CORE_LIST_NUM = 50;
constexpr uint32_t ARRAY_LENGTH = 3;
struct FuzzyBaseInfoParams { // 频繁使用的基础参数
    uint64_t coreNum;
    uint64_t aicNum;
    uint64_t ubSize;
    uint64_t l1Size;
    uint64_t l0aSize;
    uint64_t l0cSize;

    // base info
    uint32_t b;
    uint32_t n1;
    uint32_t n2;
    uint32_t g;
    uint32_t d;
    uint32_t dV;
    int64_t s1;
    int64_t s2;
    int64_t t1 = 0;
    int64_t t2 = 0;
    int64_t s1Align;
    int64_t s2Align;
    int64_t s1Tail;
    int64_t s2Tail;

    // split info
    uint32_t s1Outer;
    uint32_t s1Inner;
    uint32_t s1CvInner;
    uint32_t s1CvTail;
    uint32_t s2Outer;
    uint32_t s1CvRatio = 1;
    uint32_t s2CvRatio = 1;
    uint32_t mm1CalRatio = 1;
    uint32_t cvS2Inner;
    uint32_t s2Inner;
    uint32_t sfmgdOuter;
    uint32_t sfmgdInner;
    uint32_t sfmgdTail;

    int64_t totalS1TileNums = 0;
    int64_t sumS1S2Product = 0;

    uint32_t queryType;
    uint32_t attenMaskOptional;
    uint32_t attenMaskShapeType = 0;
    uint32_t attenMaskDtype = 0;
    uint32_t attenMaskCompressMode = 0;
    uint32_t attenMaskS1Size = 0;
    uint32_t attenMaskS2Size = 0;
    uint32_t prmMaskOptional = 0;
    uint32_t sinkOptional = 0;
    int32_t sinkRequiresGrad = 0; // 0 inputs mean no grad
    uint32_t layoutType;
    float scaleValue;
    uint32_t softmaxLseOptional;

    uint32_t dataTypeSize;
    uint32_t dataBlockNum;
    uint32_t calTypeSize;
    uint32_t calBlockNum;
    uint32_t blockNum;
    uint32_t blockOuter;
    uint32_t blockFactor;
    int64_t s1Token;
    int64_t s2Token;

    int64_t qSize;
    int64_t kSize;
    int64_t vSize;

    // sparse mode split core info
    int64_t gs1Offset[CORE_LIST_NUM];
    int64_t splitS1BlockStarts[CORE_LIST_NUM];
    int64_t splitS1BlockEnds[CORE_LIST_NUM];
    int64_t splitS2BlockStarts[CORE_LIST_NUM];
    int64_t splitS2BlockEnds[CORE_LIST_NUM];
    int64_t blockStarts[CORE_LIST_NUM];
    int64_t blockEnds[CORE_LIST_NUM];
    uint32_t baseMN;
    uint32_t sparseMode;

    std::vector<int64_t> actualSeqQlen;
    std::vector<int64_t> actualSeqKvlen;

    bool isSparse;
    bool isBf16;
    bool isNZOut;

    uint32_t tmpBufferSize = 0;

    // dq存workspace相关参数
    uint32_t s1TileNum;
    uint32_t s2TileNum;
    uint32_t gs1TileNum = 0;
    uint64_t dqkvWorkspaceOffset;
    uint64_t dqkvWorkspaceSize;
    uint32_t saveToWorkspace;

    TilingDataType mode;
};

} // namespace optiling
