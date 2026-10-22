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
#include <cstdint>
#include <string>

#include "tiling/tiling_api.h"
#include "mc2_cc_tiling_shim.h"

/*
 * A5 example 本地 shim 实现：以旧 libstdc++ ABI 编译本文件，链接 CANN 静态库
 * libtiling_api.a 中的 AscendC::Mc2CcTilingConfig 实现（example/torch 侧为新 ABI，无法直链）。
 */
extern "C" {

__attribute__((weak)) void* A5Mc2CcTilingCreate(const char* groupName, uint32_t opType, const char* algConfig)
{
    return new AscendC::Mc2CcTilingConfig(std::string(groupName), opType, std::string(algConfig));
}

__attribute__((weak)) void A5Mc2CcTilingSetCommEngine(void* cfg, uint8_t commEngine)
{
    static_cast<AscendC::Mc2CcTilingConfig*>(cfg)->SetCommEngine(commEngine);
}

__attribute__((weak)) void A5Mc2CcTilingGetInitTiling(void* cfg, void* initTiling)
{
    static_cast<AscendC::Mc2CcTilingConfig*>(cfg)->GetTiling(*reinterpret_cast<Mc2InitTiling*>(initTiling));
}

__attribute__((weak)) void A5Mc2CcTilingGetCcTiling(void* cfg, void* ccTiling)
{
    static_cast<AscendC::Mc2CcTilingConfig*>(cfg)->GetTiling(*reinterpret_cast<Mc2CcTiling*>(ccTiling));
}

__attribute__((weak)) void A5Mc2CcTilingDestroy(void* cfg)
{
    delete static_cast<AscendC::Mc2CcTilingConfig*>(cfg);
}

} // extern "C"
