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

# Keep each profiler capture paired with the accuracy rows from the same pytest run.
result_var=$1
result_path=${!result_var:-$2}
test_script=$3
test_mark=$4
golden_path=$5
shift 5

run_dir=$(mktemp -d "$PWD/msprof_XXXXXXXX") || exit 1
run_result="$run_dir/cases.xlsx"

env "$result_var=$run_result" msprof --output="$run_dir" python3 -m pytest \
    -rA -s "$test_script" -v -m "$test_mark" \
    -W ignore::UserWarning -W ignore::DeprecationWarning "$@" 2>&1 | tee res.log
pytest_status=${PIPESTATUS[0]}

perf_args=(--test_result_path "$run_result" --profile_root "$run_dir" --merge_into "$result_path")
if [[ -f "$golden_path" ]]; then
    perf_args+=(--is_compare --perf_golden_path "$golden_path")
fi
python3 process_perf_data.py "${perf_args[@]}"
perf_status=$?

if (( pytest_status == 0 && perf_status == 0 )); then
    rm -- "$run_result"
else
    printf 'Performance run files: %s\n' "$run_dir" >&2
    if [[ -f "$run_result" ]]; then
        printf 'To retry processing from %s:\n' "$PWD" >&2
        printf '%q ' python3 process_perf_data.py "${perf_args[@]}" >&2
        printf '\n' >&2
    fi
    if (( pytest_status != 0 )); then
        exit "$pytest_status"
    fi
    exit "$perf_status"
fi
