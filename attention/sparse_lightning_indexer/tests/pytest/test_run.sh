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

# ====================== 执行区 ======================

# 单用例算子调测（paramset 参数集）
run_single() {
    local case_names="$1"
    echo "===== 执行单用例算子调测 ====="
    SLI_CASE_NAMES="${case_names}" \
        python3 -m pytest -rA -s test_sparse_lightning_indexer_single.py -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning
}

# host 校验拒绝用例（C1-C7 / R18）
run_reject() {
    echo "===== 执行 host 校验拒绝用例 ====="
    python3 -m pytest -rA -s test_sparse_lightning_indexer_reject.py -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 模型场景测试套（与 LIV2 模型套一一配对）
run_model() {
    echo "===== 执行模型场景测试套 ====="
    python3 -m pytest -rA -s test_sparse_lightning_indexer_model.py -v -m ci \
        -W ignore::UserWarning -W ignore::DeprecationWarning
}

# 显示帮助信息
show_help() {
    echo "用法: $0 <command> [options]"
    echo "命令说明："
    echo "  single [-C CASE[,CASE...]]   执行单用例参数集（test_sparse_lightning_indexer_single.py）"
    echo "  reject                       执行 host 校验拒绝用例（test_sparse_lightning_indexer_reject.py）"
    echo "  model                        执行模型场景测试套（test_sparse_lightning_indexer_model.py）"
    echo "示例："
    echo "  $0 single                    # 全量单用例"
    echo "  $0 single -C SLI_FP16_01     # 指定用例"
    echo "  $0 reject"
    echo "  $0 model"
}

# ====================== 主逻辑 ======================
if [ $# -lt 1 ]; then
    echo "错误：必须传入至少一个参数"
    show_help
    exit 1
fi

COMMAND="$1"
shift

case "$COMMAND" in
    single)
        case_names=""
        while [ $# -gt 0 ]; do
            case "$1" in
                -C|--cases)
                    if [ $# -lt 2 ]; then
                        echo "错误：$1 缺少参数值"
                        exit 1
                    fi
                    case_names="$2"
                    shift 2
                    ;;
                *)
                    echo "错误：未知 single 参数 '$1'"
                    show_help
                    exit 1
                    ;;
            esac
        done
        run_single "$case_names"
        ;;
    reject)
        run_reject
        ;;
    model)
        run_model
        ;;
    help)
        show_help
        ;;
    *)
        echo "错误：未知命令 '$COMMAND'"
        show_help
        exit 1
        ;;
esac

exit 0
