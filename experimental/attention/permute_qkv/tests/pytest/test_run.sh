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
SCRIPT_TYPE=${1:?Usage: $0 {single} [-R result_path]}
shift || true
R_VALUE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -R) R_VALUE="$2"; shift 2 ;;
        *) echo "错误: 未知参数 '$1'"; exit 1 ;;
    esac
done
if [ "$SCRIPT_TYPE" == "single" ]; then
    if [ -n "$R_VALUE" ]; then export PQ_RESULT_SAVE_PATH="$R_VALUE"; fi
    python3 -m pytest -rA -s test_permute_qkv_single.py -v -m ci
else
    echo "错误: 未知的脚本类型 '$SCRIPT_TYPE'"; exit 1
fi
