/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_GEMM_DISPATCH_POLICY_ARCH92_HPP
#define BSA_GEMM_DISPATCH_POLICY_ARCH92_HPP

#include "../arch/bsa_arch.hpp"

namespace NpuArch::Gemm::Arch92 {

template <bool Transposed>
struct MmadBsa {
    using ArchTag = Arch::Ascend960DT;
    static constexpr uint32_t L0_STAGES = 2;
    static constexpr bool transposedMm1 = Transposed;
};

template <bool Transposed, bool ZnOnlineSoftmax = false>
struct MmadBsaQK : MmadBsa<Transposed> {};

template <bool Transposed>
struct MmadBsaPV : MmadBsa<Transposed> {};

template <bool Transposed>
struct MmadBsaQKMX : MmadBsa<Transposed> {};

template <bool Transposed>
struct MmadBsaPVMX : MmadBsa<Transposed> {};

template <bool Transposed>
struct MmadBsaQKMxfp8 : MmadBsa<Transposed> {};

template <bool Transposed>
struct MmadBsaPVMxfp8 : MmadBsa<Transposed> {};

template <bool Transposed>
struct MmadBsaQKFp8 : MmadBsa<Transposed> {};

template <bool Transposed>
struct MmadBsaPVFp8 : MmadBsa<Transposed> {};

} // namespace NpuArch::Gemm::Arch92

#endif // BSA_GEMM_DISPATCH_POLICY_ARCH92_HPP
