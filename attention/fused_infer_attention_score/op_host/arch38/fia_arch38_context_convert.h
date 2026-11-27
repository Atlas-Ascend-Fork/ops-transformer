/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fia_arch38_context_convert.h
 * \brief PFA/IFA 上下文转换入口：将 gert::TilingContext 收敛为 PFA/IFA 上下文参数
 */

#ifndef FIA_ARCH38_CONTEXT_CONVERT_H
#define FIA_ARCH38_CONTEXT_CONVERT_H

#include "fia_arch38_context.h"

namespace optiling {
namespace arch38 {

ge::graphStatus ConvertContextToParamsPFA(gert::TilingContext* context, FiaArch38ContextParams& contextKeyParams,
                                          bool isMaxWorkspace);
bool GetMaxWorkspaceFlag(gert::TilingContext& context);

} // namespace arch38
} // namespace optiling

#endif // FIA_ARCH38_CONTEXT_CONVERT_H
