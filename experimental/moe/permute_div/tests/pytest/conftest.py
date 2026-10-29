# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import ctypes
import os
import sys
from pathlib import Path


def _preload_runtime_libraries():
    candidates = [
        Path(sys.prefix) / "lib" / "libstdc++.so.6",
        Path(sys.executable).resolve().parents[1] / "lib" / "libstdc++.so.6",
    ]
    for library in candidates:
        if library.exists():
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
            break

    lib_dirs = [str(path) for path in candidates if path.exists()]
    if lib_dirs:
        current = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            lib_dirs + ([current] if current else [])
        )


_preload_runtime_libraries()
