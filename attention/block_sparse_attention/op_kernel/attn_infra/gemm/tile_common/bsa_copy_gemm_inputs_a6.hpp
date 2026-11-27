/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_COPY_GEMM_INPUTS_A6_HPP
#define BSA_COPY_GEMM_INPUTS_A6_HPP

#include "bsa_copy_gm_to_l1_a5.hpp"
#include "bsa_copy_l1_to_l0a_a5.hpp"
#include "bsa_copy_l1_to_l0b_a5.hpp"

namespace NpuArch::Gemm::Tile {

// 960DT uses the same GM-to-L1 and L1-to-L0 operand transfers and layouts as AtlasA5.
template <class TensorSrc, class TensorDst>
struct TileCopyTla<
    Arch::Ascend960DT, TensorSrc, TensorDst,
    std::enable_if_t<(TensorSrc::position == AscendC::TPosition::GM && TensorDst::position == AscendC::TPosition::A1) ||
                     (TensorSrc::position == AscendC::TPosition::A1 &&
                      (TensorDst::position == AscendC::TPosition::A2 ||
                       TensorDst::position == AscendC::TPosition::B2))>>
    : TileCopyTla<Arch::AtlasA5, TensorSrc, TensorDst> {};

} // namespace NpuArch::Gemm::Tile

#endif // BSA_COPY_GEMM_INPUTS_A6_HPP
