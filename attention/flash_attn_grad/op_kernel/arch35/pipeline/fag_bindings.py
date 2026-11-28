# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Bind codegen helpers without mixing template resource families.

PyPTO inlines helpers against the entry module's bare globals. Module order is
part of that contract: later exports override earlier ones, and only named
ping-pong arrays may be lifted into entry globals. This runs at import time.
"""

import inspect


def bind_codegen_helpers(namespace, modules, pingpong_lists):
    """Preserve the caller's ordered namespace and list-export policy."""
    for module in modules:
        for name, value in vars(module).items():
            if name.startswith("_") or inspect.ismodule(value):
                continue
            if isinstance(value, list) and name not in pingpong_lists:
                continue
            namespace[name] = value
