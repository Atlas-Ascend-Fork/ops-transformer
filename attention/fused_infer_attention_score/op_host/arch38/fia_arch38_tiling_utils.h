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
 * \file fia_arch38_tiling_utils.h
 * \brief FIA arch38 实现侧公共定义：本地常量、数值工具函数、枚举、DoOpTiling 校验入参结构与错误上报/辅助工具
 *
 * 本头文件是 arch38 实现内部（PFA 模板拆分出的多个 .cpp 与 DoOpTiling）共享的公共定义，
 * 不参与算子 tiling 数据的序列化契约（序列化契约见 fia_arch38_tiling_common.h）。
 * 内容按用途分块：
 *   1) 维度/形状相关常量；2) 限值常量；3) 稀疏模式常量；4) 分块/切分因子常量；5) MLA 及其它常量；
 *   6) 数值计算模板函数；7) 布局/稀疏/PSE 枚举；8) DoOpTiling 形状校验入参结构；9) 错误上报/辅助工具函数。
 */

#ifndef FIA_ARCH38_TILING_UTILS_H
#define FIA_ARCH38_TILING_UTILS_H

#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <exe_graph/runtime/tiling_context.h>
#include "log/log.h"
#include "err/ops_err.h"

namespace optiling {
namespace arch38 {

// ------------------------------------------------------------------
// 1) 维度/形状相关常量
// ------------------------------------------------------------------
// Inputs/Outputs 维度索引（DoOpTiling 形状校验使用）
constexpr uint32_t QUERY_DIM_0 = 0;
constexpr uint32_t QUERY_DIM_1 = 1;
constexpr uint32_t QUERY_DIM_2 = 2;
constexpr uint32_t QUERY_DIM_3 = 3;
constexpr uint32_t QUERY_DIM_4 = 4;
constexpr uint32_t VALUE_DIM_2 = 2;
constexpr uint32_t VALUE_DIM_3 = 3;
constexpr uint32_t KV_DIM_0 = 0;
constexpr uint32_t KV_DIM_2 = 2;
constexpr uint32_t KV_DIM_3 = 3;
constexpr uint32_t OUT_DIM_1 = 1;
constexpr uint32_t OUT_DIM_2 = 2;
constexpr uint32_t OUT_DIM_3 = 3;

constexpr uint32_t SHAPE_INDEX_TWO = 2;

// 输入 Q/K/V 形状维度数范围
constexpr uint32_t INPUT_QKV_SHAPE_MIN_DIMS = 3;
constexpr uint32_t INPUT_QKV_SHAPE_MAX_DIMS = 4;

// datacopy 按块搬移的块粒度（字节）
constexpr uint32_t BYTE_BLOCK = 32;

// 注意力掩码 / PSE 形状维度数
constexpr uint32_t MASKDIM_2 = 2;
constexpr uint32_t MASKDIM_3 = 3;
constexpr uint32_t MASKDIM_4 = 4;
constexpr uint32_t PSESHIFTDIM_4 = 4;

// PA 场景 KV cache 形状维度数及维度下标
constexpr uint32_t KV_CACHE_DIM_NUMS_4 = 4;
constexpr uint32_t KV_CACHE_DIM_NUMS_3 = 3;
constexpr uint32_t KV_CACHE_DIM_0 = 0;
constexpr uint32_t KV_CACHE_DIM_1 = 1;
constexpr uint32_t KV_CACHE_DIM_2 = 2;
constexpr uint32_t KV_CACHE_DIM_3 = 3;

// ------------------------------------------------------------------
// 2) 限值常量
// ------------------------------------------------------------------
constexpr uint32_t DLIMIT = 512;
constexpr uint32_t BLIMIT = 65536;
constexpr uint32_t NLIMIT = 256;      // n <= 256
constexpr uint32_t SLIMIT = 20971520; // s, kvs <= 20MB
constexpr uint32_t HLIMIT = 65535;    // warning: H <= 65536

// 本平台已验证的场景边界，越界场景显式拒绝而非静默误算
constexpr int64_t ARCH38_SUPPORTED_MAX_SEQ = 4096;
constexpr int64_t ARCH38_SUPPORTED_MAX_N = 128;

// ------------------------------------------------------------------
// 3) 稀疏模式常量
// ------------------------------------------------------------------
// 空 KV/空输出固定 tiling key（1e18 + 90），需与 kernel entry 及 REGISTER_TILING_DATA_CLASS 注册名一致。
constexpr uint64_t EMPTY_KV_TILING_KEY = 90;
constexpr uint32_t LOOP_BEGIN_NUM = 0;
constexpr uint32_t SPARSE_MODE_NO_MASK = 0;
constexpr uint32_t SPARSE_MODE_ALL_MASK = 1;
constexpr uint32_t SPARSE_MODE_LEFT_UP = 2;
constexpr uint32_t SPARSE_MODE_RIGHT_DOWN = 3;
constexpr uint32_t SPARSE_MODE_BAND = 4;
constexpr int64_t SPARSE_MODE_INT_MAX = 2147483647;
constexpr uint32_t SPARSE_OPTIMIZE_ATTENTION_SIZE = 2048;

// ------------------------------------------------------------------
// 4) 分块/切分因子常量
// ------------------------------------------------------------------
// mm base 取 128 的倍数，避免跨 block 处理
constexpr int32_t BLOCK_SIZE_BASE = 128;
constexpr int32_t BLOCK_SIZE_MAX = 512;

constexpr uint32_t SOUTER_FACTOR_SUB = 32;
constexpr uint32_t SOUTER_FACTOR_DEFAULT = 64;
constexpr uint32_t SINNER_FACTOR_SUB = 64;
constexpr uint32_t SINNER_FACTOR_DEFAULT = 128;
constexpr uint32_t SINNER_FACTOR_DOUBLE = 256;

constexpr uint32_t MATMUL_NORM_MIN_SEQ = 128;
constexpr uint32_t MATMUL_NORM_MIN_HEADSIZE = 128;

// ------------------------------------------------------------------
// 5) MLA 及其它常量
// ------------------------------------------------------------------
constexpr uint32_t MLA_QKD_SIZE = 192;
constexpr uint32_t MLA_VD_SIZE = 128; // PFA MLA 典型场景

constexpr int64_t GM_ALIGN = 512;
constexpr int64_t SLOPE_N_DIM_NUM = 1L;
constexpr uint64_t BENCHMARK_TILING_KEY = 1000000000000000000ULL;
constexpr uint32_t ATTR_SPARSE_MODE = 6;
constexpr uint32_t ATTR_INNER_PRECISE = 7;
constexpr int64_t PSE_TYPE_2_TILING_V2 = 2;
constexpr int64_t PSE_TYPE_3_TILING_V2 = 3;
constexpr uint32_t QUERY_SHAPE_DIM_D_128_TILING_V2 = 128;
constexpr int32_t ROPE_DIMENSION_SIZE_TILING_V2 = 64;
constexpr uint32_t BATCH_MODE_SCHEDULE = 1;

// ------------------------------------------------------------------
// 6) 数值计算模板函数
// ------------------------------------------------------------------
// 向上对齐到 num2 的整数倍
template <typename T>
inline auto AlignUp(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    if (num1 < 0) {
        return -(-num1 / num2) * num2;
    }
    return (num1 + num2 - 1) / num2 * num2;
}

// 向上取整除法
template <typename T>
inline auto CeilDivision(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    return (num1 + num2 - 1) / num2;
}

// 计算尾部大小（num1 % num2，整除时返回 num2）
template <typename T>
inline auto CalcTailSize(T num1, T num2) -> T
{
    if (num2 == 0) {
        return 0;
    }
    T mod = num1 % num2;
    return mod != 0 ? mod : num2;
}

// 向上取整除法（n2 为 0 时返回 n1）
template <typename T>
inline auto CeilDiv(const T n1, const T n2) -> T
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? (((n1 - 1) / n2) + 1) : n1;
}

// ------------------------------------------------------------------
// 7) 布局/稀疏/PSE 枚举
// ------------------------------------------------------------------
// 输入布局类型
enum class LayoutType : uint8_t {
    NONE = 0,
    LAYOUT_BSH = 1,
    LAYOUT_BSND = 1,
    LAYOUT_SBH = 2,
    LAYOUT_BNSD = 3,
    LAYOUT_TND = 4,
};

// 稀疏类型（对应 kernel 侧 PFA 稀疏枚举）
enum class PfaSparseEnum : uint8_t {
    PFA_ALL = 0,
    PFA_NONE = 1,
    PFA_ANY = 2,
    PFA_CAUSAL = 3,
    PFA_BAND = 4,
    PFA_PREFIX = 5,
    PFA_BAND_COMPRESS = 6,
    PFA_RIGHT_DOWN_CAUSAL = 7,
    PFA_RIGHT_DOWN_CAUSAL_BAND = 8,
    PFA_BAND_LEFT_UP_CAUSAL = 9
};

// 注意力掩码压缩模式
enum PfaAttenMaskCompressMode : uint8_t {
    PFA_NO_COMPRESS_MODE = 0,
    PFA_LEFT_UP_CAUSAL_MODE,
    PFA_RIGHT_DOWN_CAUSAL_MODE,
    PFA_BAND_MODE,
    PFA_PREFIX_MODE,
    PFA_RIGHT_DOWN_CAUSAL_BAND_MODE,
    PFA_BAND_LEFT_UP_CAUSAL_MODE
};

// PSE 类型
enum class PfaPseType : uint8_t {
    PSE_OUTER_MUL_ADD_TYPE = 0, // v2 default
    PSE_OUTER_ADD_MUL_TYPE,     // v1 current usage
    PSE_INNER_MUL_ADD_TYPE,
    PSE_INNER_MUL_ADD_SQRT_TYPE,
    PSE_INVALID_TYPE,
    PSE_NONE_TYPE = 9
};

// PSE 形状类型
enum class PfaPseShapeType : uint8_t {
    PSE_B_N2_G_S1_S2 = 0,
    PSE_B_N2_G_1_S2 = 1,
    PSE_B_N2_G_SLOPE,
    PSE_1_N2_G_SLOPE
};

// ------------------------------------------------------------------
// 8) DoOpTiling 形状校验入参结构
// ------------------------------------------------------------------
// DoOpTiling 形状校验的共享只读入参
struct QKVShapeCheckCtx {
    const gert::StorageShape* tempQ = nullptr;
    const gert::StorageShape* tempV = nullptr;
    const gert::StorageShape* tempOut = nullptr;
    int64_t tempN = 0;
    int64_t tempKVN = 0;
    std::string inputLayoutStr;
};

// softmax_lse 形状校验入参
struct LseShapeCheckCtx {
    const gert::StorageShape* tempLse = nullptr;
    std::string inputLayoutStr;
    int64_t tempN = 0;
    int64_t b = 0;
    int64_t s = 0;
    int64_t t = 0;
};

// ------------------------------------------------------------------
// 9) 错误上报/辅助工具函数
// ------------------------------------------------------------------
// 合法输入下 Tiling 失败时，按 EZ1008 正式错误码首报（Arglist: opName, reason），并输出 PLOG 上下文
inline void ReportTilingFailure(const gert::TilingContext* context, const char* reason)
{
    OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "%s\n", reason);
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgvalue = {context->GetNodeName(), reason};
    REPORT_PREDEFINED_ERR_MSG("EZ1008", msgKey, msgvalue);
}

// 必选输入/属性为空指针时，按 EZ1011 正式错误码首报（Arglist: functionName, paraName），并输出 PLOG 上下文
inline void ReportNullPointer(const gert::TilingContext* context, const char* functionName, const char* paraName)
{
    const char* opName = (context != nullptr && context->GetNodeName() != nullptr) ? context->GetNodeName() : "";
    OP_LOGE_WITHOUT_REPORT(opName, "%s failed because %s cannot be a NULL pointer.", functionName, paraName);
    const std::vector<const char*> msgKey = {"functionName", "paraName"};
    const std::vector<const char*> msgValue = {functionName, paraName};
    REPORT_PREDEFINED_ERR_MSG("EZ1011", msgKey, msgValue);
}

// 将 shape 各维拼接为 "[d0, d1, ...]" 描述字符串
inline std::string GetShapeDebugStr(const gert::Shape& shape)
{
    std::ostringstream oss;
    oss << "[";
    if (shape.GetDimNum() > 0) {
        for (size_t i = 0; i < shape.GetDimNum() - 1; ++i) {
            oss << shape.GetDim(i) << ", ";
        }
        oss << shape.GetDim(shape.GetDimNum() - 1);
    }
    oss << "]";
    return oss.str();
}

// 布局字符串 -> 转置布局编号；0 不转置; 1 BNSD_BSND; 2 BSND_BNSD; 3 BSH_BNSD; 4 BNSD_NBSD; 5 BSND_NBSD; 6 BSH_NBSD; 7
// NTD_TND; 8 TND_NTD
inline uint32_t GetTransposeLayout(const std::string& layout)
{
    const std::map<std::string, uint32_t> transposeLayoutMp = {{"BNSD_BSND", 1}, {"BSND_BNSD", 2}, {"BSH_BNSD", 3},
                                                               {"BNSD_NBSD", 4}, {"BSND_NBSD", 5}, {"BSH_NBSD", 6},
                                                               {"NTD_TND", 7},   {"TND_NTD", 8}};
    if (transposeLayoutMp.find(layout) != transposeLayoutMp.end()) {
        return transposeLayoutMp.at(layout);
    }
    return 0;
}

} // namespace arch38
} // namespace optiling

#endif // FIA_ARCH38_TILING_UTILS_H
