#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e
[ "$1" == "single" ] || { echo "Usage: $0 single [-R result_path]"; exit 1; }
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        -R) export UQ_RESULT_SAVE_PATH="$2"; shift 2 ;;
        *) echo "错误: 未知参数 '$1'"; exit 1 ;;
    esac
done
python3 -m pytest -rA -s test_unpermute_qkv_single.py -v -m ci
