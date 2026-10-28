/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MHC_PRE_SINKHORN_BACKWARD_KEY_ARCH35_H
#define MHC_PRE_SINKHORN_BACKWARD_KEY_ARCH35_H

namespace MhcPreSinkhornBackwardNs {
constexpr uint64_t TILING_KEY_MK_SPLIT = 3000;
constexpr uint64_t TILING_KEY_M_SPLIT = 3001;
constexpr uint64_t TILING_KEY_M_SPLIT_DETERMINISTIC = 3003;

enum class SplitMode : int64_t {
    MK_SPLIT = 0,
    M_SPLIT = 1,
};
} // namespace MhcPreSinkhornBackwardNs

#endif // MHC_PRE_SINKHORN_BACKWARD_KEY_ARCH35_H
