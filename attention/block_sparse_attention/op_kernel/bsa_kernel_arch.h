/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BSA_KERNEL_ARCH_H
#define BSA_KERNEL_ARCH_H

// The 960DT compiler may also expose the 310 instruction-set macro.
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 9201
#define BSA_ARCH_92
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 310
#define BSA_ARCH_35
#elif defined(__CCE_AICORE__) && __CCE_AICORE__ == 220
#define BSA_ARCH_22
#endif

#endif // BSA_KERNEL_ARCH_H
