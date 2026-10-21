/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_ARCH92_QBSA_BLOCK_UTILS_H
#define QBSA_ARCH92_QBSA_BLOCK_UTILS_H

namespace BaseApi {
__aicore__ constexpr uint16_t Align64Func(uint16_t data)
{
    return (data + ADD_NUM_63) >> SHIFT_NUM_6 << SHIFT_NUM_6;
}

__aicore__ constexpr bool IsDn()
{
    return true;
}

__aicore__ constexpr TPosition GetC2Position()
{
    return TPosition::VECCALC;
}
} // namespace BaseApi

#endif // QBSA_ARCH92_QBSA_BLOCK_UTILS_H
