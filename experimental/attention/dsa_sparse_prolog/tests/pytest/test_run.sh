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

show_help() {
    cat << EOF
使用方法: $0 {single} [参数]

脚本选项:
  single         执行单跑功能
    -R           结果保存路径
    示例: bash $0 single -R "./result/dsa_prolog_result.xlsx"

通用选项:
  -h, --help   显示此帮助信息
EOF
}

if [ $# -lt 1 ]; then
    echo "错误: 需要至少一个参数"
    show_help
    exit 1
fi

SCRIPT_TYPE=$1
shift

if [[ "$SCRIPT_TYPE" == "-h" ]] || [[ "$SCRIPT_TYPE" == "--help" ]]; then
    show_help
    exit 0
fi

run_script_single() {
    echo "准备运行 单跑 脚本..."
    while [[ $# -gt 0 ]]; do
        case $1 in
            -R)
                if [ -z "$2" ] || [[ "$2" == -* ]]; then
                    echo "错误: -R 参数需要值"
                    exit 1
                fi
                R_VALUE="$2"
                shift 2
                ;;
            *)
                echo "错误: 未知参数 '$1'"
                exit 1
                ;;
        esac
    done

    if [ -n "$R_VALUE" ]; then
        echo "  结果存储至文件 $R_VALUE"
        export DSA_PROLOG_RESULT_SAVE_PATH="$R_VALUE"
    else
        echo "  结果存储至文件 ./result/dsa_prolog_result.xlsx"
    fi

    if [ ! -f "test_dsa_sparse_prolog_single.py" ]; then
        echo "错误: 找不到 test_dsa_sparse_prolog_single.py 脚本"
        exit 1
    fi
    python3 -m pytest -rA -s test_dsa_sparse_prolog_single.py -v -m ci
}

case "$SCRIPT_TYPE" in
    single)
        run_script_single "$@"
        ;;
    *)
        echo "错误: 未知的脚本类型 '$SCRIPT_TYPE'"
        show_help
        exit 1
        ;;
esac

echo "==============================="
echo "所有任务完成"
