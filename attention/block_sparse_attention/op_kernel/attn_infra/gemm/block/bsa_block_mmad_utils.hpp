/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_BLOCK_MMAD_UTILS_HPP
#define BSA_BLOCK_MMAD_UTILS_HPP

#include "../../../bsa_kernel_arch.h"

#if defined(BSA_ARCH_92)
#include "block_mmad_arch92_utils.hpp"
#else
#include "block_mmad_arch35_utils.hpp"
#endif

#endif // BSA_BLOCK_MMAD_UTILS_HPP
