/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */
#ifndef MC2_CC_TILING_SHIM_H
#define MC2_CC_TILING_SHIM_H

#include <cstdint>

/*
 * A5 example 本地 shim（extern "C" ABI 中立接口）：
 * 包装 AscendC::Mc2CcTilingConfig（CANN 静态库 libtiling_api.a 以旧 libstdc++ ABI 编译，
 * example/torch 侧为新 ABI，符号无法直接互链），实现见 mc2_cc_tiling_shim.cpp（编译时强制旧 ABI）。
 */
#ifdef __cplusplus
extern "C" {
#endif

void* A5Mc2CcTilingCreate(const char* groupName, uint32_t opType, const char* algConfig);
void A5Mc2CcTilingSetCommEngine(void* cfg, uint8_t commEngine);
void A5Mc2CcTilingGetInitTiling(void* cfg, void* initTiling); // 填充 Mc2InitTiling (64B)
void A5Mc2CcTilingGetCcTiling(void* cfg, void* ccTiling);     // 填充 Mc2CcTiling (512B)
void A5Mc2CcTilingDestroy(void* cfg);

#ifdef __cplusplus
}
#endif

#endif // MC2_CC_TILING_SHIM_H
