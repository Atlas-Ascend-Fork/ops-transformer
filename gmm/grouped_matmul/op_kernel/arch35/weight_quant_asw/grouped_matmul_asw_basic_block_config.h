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
 * \file grouped_matmul_asw_basic_block_config.h
 * \brief
 */

#pragma once

#include "weight_quant_tool.h"

using WeightQuantTools::GetKBUnit;
using WeightQuantTools::IsA16MxFp4;
using WeightQuantTools::IsA16MxHiF4;
using WeightQuantTools::IsMxA8HiF4;

namespace WeightQuantGroupedMatmul::Arch35 {
struct L1BufferCfg {
    uint32_t bL1SizeElem;            // weight L1单块buffer包含元素个数
    uint32_t aL1SizeElemWithBias;    // x L1单块buffer包含元素个数（含bias场景）
    uint32_t aL1SizeElemWithoutBias; // x L1单块buffer包含元素个数（不含bias场景）
    uint32_t aScaleL1SizeElem;       // x scale L1单块buffer包含元素个数
    uint32_t bScaleL1SizeElem;       // weight L1单块buffer包含元素个数
    uint32_t biasL1SizeElem;         // bias L1单块buffer包含元素个数
    uint32_t scaleKL1Size; // scale多倍载入对应的k轴大小，该值除以MX_GROUP_SIZE为对应的perGroup分组数
};

template <typename xType, typename wType, typename antiquantScaleType, typename pertokenScaleType, typename biasType>
__aicore__ constexpr L1BufferCfg GetMxA8W4BufferCfg()
{
    return {.bL1SizeElem = 64 * GetKBUnit<wType>(),
            .aL1SizeElemWithBias = 120 * GetKBUnit<xType>(),
            .aL1SizeElemWithoutBias = 128 * GetKBUnit<xType>(),
            .aScaleL1SizeElem = 32 * GetKBUnit<pertokenScaleType>(),
            .bScaleL1SizeElem = 32 * GetKBUnit<antiquantScaleType>(),
            .biasL1SizeElem = 1 * GetKBUnit<biasType>(),
            .scaleKL1Size = 4096U};
}

template <typename xType, typename wType, typename antiquantScaleType, typename pertokenScaleType, typename biasType>
__aicore__ constexpr L1BufferCfg GetMxA8HiF4BufferCfg()
{
    return {.bL1SizeElem = 64 * GetKBUnit<wType>(),
            .aL1SizeElemWithBias = 128 * GetKBUnit<xType>(),
            .aL1SizeElemWithoutBias = 128 * GetKBUnit<xType>(),
            .aScaleL1SizeElem = 16 * GetKBUnit<pertokenScaleType>(),
            .bScaleL1SizeElem = 32 * GetKBUnit<antiquantScaleType>(),
            .biasL1SizeElem = 1 * GetKBUnit<biasType>(),
            .scaleKL1Size = 2048U};
}

template <typename xType, typename wType, typename antiquantScaleType, typename biasType>
__aicore__ constexpr L1BufferCfg GetA16MxFp4BufferCfg()
{
    return {.bL1SizeElem = 64 * GetKBUnit<wType>(),
            .aL1SizeElemWithBias = 128 * GetKBUnit<xType>(),
            .aL1SizeElemWithoutBias = 128 * GetKBUnit<xType>(),
            .aScaleL1SizeElem = 0,
            .bScaleL1SizeElem = 32 * GetKBUnit<antiquantScaleType>(),
            .biasL1SizeElem = 1 * GetKBUnit<biasType>(),
            .scaleKL1Size = 4096U};
}

template <typename xType, typename wType, typename antiquantScaleType, typename biasType>
__aicore__ constexpr L1BufferCfg GetA16MxHiF4BufferCfg()
{
    return {.bL1SizeElem = 64 * GetKBUnit<wType>(),
            .aL1SizeElemWithBias = 112 * GetKBUnit<xType>(),
            .aL1SizeElemWithoutBias = 128 * GetKBUnit<xType>(),
            .aScaleL1SizeElem = 0,
            .bScaleL1SizeElem = 64 * GetKBUnit<antiquantScaleType>(),
            .biasL1SizeElem = 1 * GetKBUnit<biasType>(),
            .scaleKL1Size = 4096U};
}

template <typename xType, typename wType, typename antiquantScaleType, typename pertokenScaleType, typename biasType>
__aicore__ constexpr L1BufferCfg GetBufferConfig()
{
    if constexpr (IsMxA8HiF4<xType, wType, antiquantScaleType>()) {
        return GetMxA8HiF4BufferCfg<xType, wType, antiquantScaleType, pertokenScaleType, biasType>();
    } else if constexpr (IsA16MxFp4<xType, wType, antiquantScaleType>()) {
        return GetA16MxFp4BufferCfg<xType, wType, antiquantScaleType, biasType>();
    } else if constexpr (IsA16MxHiF4<xType, wType, antiquantScaleType>()) {
        return GetA16MxHiF4BufferCfg<xType, wType, antiquantScaleType, biasType>();
    } else {
        return GetMxA8W4BufferCfg<xType, wType, antiquantScaleType, pertokenScaleType, biasType>();
    }
}
} // namespace WeightQuantGroupedMatmul::Arch35
