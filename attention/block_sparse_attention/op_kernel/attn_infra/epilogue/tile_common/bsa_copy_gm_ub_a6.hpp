/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_COPY_GM_UB_A6_HPP
#define BSA_COPY_GM_UB_A6_HPP

#include "bsa_copy_gm_to_ub_tla.hpp"
#include "bsa_copy_ub_to_gm_tla.hpp"

namespace NpuArch::Epilogue::Tile {

template <class TensorSrc, class TensorDst>
struct CopyGm2UbTla<Arch::Ascend960DT, TensorSrc, TensorDst> : CopyGm2UbTla<Arch::AtlasA5, TensorSrc, TensorDst> {};

template <class TensorSrc, class TensorDst>
struct CopyUb2GmTla<Arch::Ascend960DT, TensorSrc, TensorDst> : CopyUb2GmTla<Arch::AtlasA5, TensorSrc, TensorDst> {};

} // namespace NpuArch::Epilogue::Tile

#endif // BSA_COPY_GM_UB_A6_HPP
