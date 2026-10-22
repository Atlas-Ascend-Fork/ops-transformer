# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
add_sources("--npu-arch=dav-3510")

# mc2_cc_tiling_shim.cpp 需以旧 libstdc++ ABI 编译（链接 libtiling_api.a），覆盖全局 -D_GLIBCXX_USE_CXX11_ABI=1
set_source_files_properties(mc2_cc_tiling_shim.cpp PROPERTIES COMPILE_FLAGS "--npu-arch=dav-3510 -xasc -D_GLIBCXX_USE_CXX11_ABI=0")
