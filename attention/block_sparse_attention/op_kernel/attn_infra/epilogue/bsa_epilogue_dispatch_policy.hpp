/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_EPILOGUE_DISPATCH_POLICY_HPP
#define BSA_EPILOGUE_DISPATCH_POLICY_HPP

#include "../../attn_infra/bsa_base_defs.hpp"
#include "../../attn_infra/arch/bsa_arch.hpp"

namespace NpuArch::Epilogue {

enum class LseMode {
    NONE = 0,
    OUT_ONLY = 1
};
enum class LseFormat {
    TN1 = 0,
    BNS1 = 1,
    BSN1 = 2
};
enum class MXQuantMode {
    NONE = 0,
    OCP = 1,
    CX = 2
};
// For AtlasA2, FA Infer online Softmax
template <LseMode LSE_MODE_, typename SM_DTYPE_>
struct EpilogueAtlasA2OnlineSoftmax {
    using ArchTag = Arch::AtlasA2;
    using IntermPrec = SM_DTYPE_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
};

// For AtlasA2, FA Infer RescaleO
template <LseMode LSE_MODE_, typename SM_DTYPE_>
struct EpilogueAtlasA2RescaleO {
    using ArchTag = Arch::AtlasA2;
    using IntermPrec = SM_DTYPE_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
};

template <typename ArchTag_>
struct EpilogueBsaMask2IdxPolicy {
    static constexpr uint32_t IO_STAGES = 2;
    using ArchTag = ArchTag_;
};

using EpilogueBsaMask2Idx = EpilogueBsaMask2IdxPolicy<Arch::AtlasA5>;

template <bool transposedMm1_ = false, typename ArchTag_ = Arch::AtlasA5>
struct EpilogueOnlineSoftmaxBsa {
    using ArchTag = ArchTag_;
    static constexpr bool transposedMm1 = transposedMm1_;
};

template <bool transposedMm1_ = false, bool zNOnlineSoftmax_ = false, typename ArchTag_ = Arch::AtlasA5>
struct EpilogueB8FullQuantOnlineSoftmaxBsa {
    using ArchTag = ArchTag_;
    static constexpr bool transposedMm1 = transposedMm1_;
    static constexpr bool zNOnlineSoftmax = zNOnlineSoftmax_;
};

template <bool transposedMm1_ = false, MXQuantMode MX_QUANT_MODE_ = MXQuantMode::NONE,
          typename ArchTag_ = Arch::AtlasA5>
struct EpilogueOnlineSoftmaxBsaMX {
    using ArchTag = ArchTag_;
    static constexpr bool transposedMm1 = transposedMm1_;
    static constexpr MXQuantMode MX_QUANT_MODE = MX_QUANT_MODE_;
};

template <bool transposedMm1_ = false, MXQuantMode MX_QUANT_MODE_ = MXQuantMode::NONE,
          typename ArchTag_ = Arch::AtlasA5>
struct EpilogueOnlineSoftmaxBsaMxfp8 {
    using ArchTag = ArchTag_;
    static constexpr bool transposedMm1 = transposedMm1_;
    static constexpr MXQuantMode MX_QUANT_MODE = MX_QUANT_MODE_;
};

template <bool transposedMm1_ = false, MXQuantMode MX_QUANT_MODE_ = MXQuantMode::NONE,
          typename ArchTag_ = Arch::AtlasA5>
struct EpilogueComputePScaleBsaMX {
    using ArchTag = ArchTag_;
    static constexpr bool transposedMm1 = transposedMm1_;
    static constexpr MXQuantMode MX_QUANT_MODE = MX_QUANT_MODE_;
};

template <bool transposedMm1_ = false, MXQuantMode MX_QUANT_MODE_ = MXQuantMode::NONE,
          typename ArchTag_ = Arch::AtlasA5>
struct EpilogueComputePScaleBsaMxfp8 {
    using ArchTag = ArchTag_;
    static constexpr bool transposedMm1 = transposedMm1_;
    static constexpr MXQuantMode MX_QUANT_MODE = MX_QUANT_MODE_;
};

struct EpilogueCopyGlobalMaxUbToL1BsaMX {
    using ArchTag = Arch::AtlasA5;
};

struct EpilogueCopyGlobalMaxUbToL1BsaMxfp8 {
    using ArchTag = Arch::AtlasA5;
};

template <LseMode LSE_MODE_ = LseMode::NONE, LseFormat LSE_FORMAT_ = LseFormat::BNS1, bool transposedMm1_ = false,
          bool zNOnlineSoftmax_ = false, typename ArchTag_ = Arch::AtlasA5>
struct EpilogueBsaRescaleO {
    using ArchTag = ArchTag_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
    static constexpr LseFormat LSE_FORMAT = LSE_FORMAT_;
    static constexpr bool transposedMm1 = transposedMm1_;
    static constexpr bool zNOnlineSoftmax = zNOnlineSoftmax_;
};

template <LseMode LSE_MODE_ = LseMode::NONE, LseFormat LSE_FORMAT_ = LseFormat::BNS1, bool transposedMm1_ = false,
          bool zNOnlineSoftmax_ = false>
using EpilogueAtlasA5BsaRescaleO =
    EpilogueBsaRescaleO<LSE_MODE_, LSE_FORMAT_, transposedMm1_, zNOnlineSoftmax_, Arch::AtlasA5>;

template <LseMode LSE_MODE_ = LseMode::NONE, LseFormat LSE_FORMAT_ = LseFormat::BNS1, bool transposedMm1_ = false,
          typename ArchTag_ = Arch::AtlasA5>
struct EpilogueBsaRescaleOMX {
    using ArchTag = ArchTag_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
    static constexpr LseFormat LSE_FORMAT = LSE_FORMAT_;
    static constexpr bool transposedMm1 = transposedMm1_;
};

template <LseMode LSE_MODE_ = LseMode::NONE, LseFormat LSE_FORMAT_ = LseFormat::BNS1, bool transposedMm1_ = false>
using EpilogueAtlasA5BsaRescaleOMX = EpilogueBsaRescaleOMX<LSE_MODE_, LSE_FORMAT_, transposedMm1_, Arch::AtlasA5>;

template <LseMode LSE_MODE_ = LseMode::NONE, LseFormat LSE_FORMAT_ = LseFormat::BNS1, bool transposedMm1_ = false,
          typename ArchTag_ = Arch::AtlasA5>
struct EpilogueBsaRescaleOMxfp8 {
    using ArchTag = ArchTag_;
    static constexpr LseMode LSE_MODE = LSE_MODE_;
    static constexpr LseFormat LSE_FORMAT = LSE_FORMAT_;
    static constexpr bool transposedMm1 = transposedMm1_;
};

template <LseMode LSE_MODE_ = LseMode::NONE, LseFormat LSE_FORMAT_ = LseFormat::BNS1, bool transposedMm1_ = false>
using EpilogueAtlasA5BsaRescaleOMxfp8 = EpilogueBsaRescaleOMxfp8<LSE_MODE_, LSE_FORMAT_, transposedMm1_, Arch::AtlasA5>;

namespace Arch92 {

using EpilogueBsaMask2Idx = EpilogueBsaMask2IdxPolicy<Arch::Ascend960DT>;

template <bool transposedMm1_ = false>
using EpilogueOnlineSoftmaxBsa = Epilogue::EpilogueOnlineSoftmaxBsa<transposedMm1_, Arch::Ascend960DT>;

template <bool transposedMm1_ = false, bool zNOnlineSoftmax_ = false>
using EpilogueB8FullQuantOnlineSoftmaxBsa =
    Epilogue::EpilogueB8FullQuantOnlineSoftmaxBsa<transposedMm1_, zNOnlineSoftmax_, Arch::Ascend960DT>;

template <bool transposedMm1_ = false, MXQuantMode MX_QUANT_MODE_ = MXQuantMode::NONE>
using EpilogueOnlineSoftmaxBsaMX =
    Epilogue::EpilogueOnlineSoftmaxBsaMX<transposedMm1_, MX_QUANT_MODE_, Arch::Ascend960DT>;

template <bool transposedMm1_ = false, MXQuantMode MX_QUANT_MODE_ = MXQuantMode::NONE>
using EpilogueOnlineSoftmaxBsaMxfp8 =
    Epilogue::EpilogueOnlineSoftmaxBsaMxfp8<transposedMm1_, MX_QUANT_MODE_, Arch::Ascend960DT>;

template <bool transposedMm1_ = false, MXQuantMode MX_QUANT_MODE_ = MXQuantMode::NONE>
using EpilogueComputePScaleBsaMX =
    Epilogue::EpilogueComputePScaleBsaMX<transposedMm1_, MX_QUANT_MODE_, Arch::Ascend960DT>;

template <bool transposedMm1_ = false, MXQuantMode MX_QUANT_MODE_ = MXQuantMode::NONE>
using EpilogueComputePScaleBsaMxfp8 =
    Epilogue::EpilogueComputePScaleBsaMxfp8<transposedMm1_, MX_QUANT_MODE_, Arch::Ascend960DT>;

template <LseMode LSE_MODE_ = LseMode::NONE, LseFormat LSE_FORMAT_ = LseFormat::BNS1, bool transposedMm1_ = false,
          bool zNOnlineSoftmax_ = false>
using EpilogueBsaRescaleO =
    Epilogue::EpilogueBsaRescaleO<LSE_MODE_, LSE_FORMAT_, transposedMm1_, zNOnlineSoftmax_, Arch::Ascend960DT>;

template <LseMode LSE_MODE_ = LseMode::NONE, LseFormat LSE_FORMAT_ = LseFormat::BNS1, bool transposedMm1_ = false>
using EpilogueBsaRescaleOMX =
    Epilogue::EpilogueBsaRescaleOMX<LSE_MODE_, LSE_FORMAT_, transposedMm1_, Arch::Ascend960DT>;

template <LseMode LSE_MODE_ = LseMode::NONE, LseFormat LSE_FORMAT_ = LseFormat::BNS1, bool transposedMm1_ = false>
using EpilogueBsaRescaleOMxfp8 =
    Epilogue::EpilogueBsaRescaleOMxfp8<LSE_MODE_, LSE_FORMAT_, transposedMm1_, Arch::Ascend960DT>;

} // namespace Arch92

} // namespace NpuArch::Epilogue

#endif // EPILOGUE_DISPATCH_POLICY_HPP
