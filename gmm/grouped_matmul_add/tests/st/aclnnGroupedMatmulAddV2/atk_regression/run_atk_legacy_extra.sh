#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

# Run rounds 4-8 of the common legacy path in an already configured CANN/ATK environment.
# Usage: bash run_atk_legacy_extra.sh BASE_VENDOR FINAL_VENDOR OUTPUT_DIR PHYSICAL_DEVICE
set -eo pipefail
base="$1"
current="$2"
root="$(mkdir -p "$3" && cd "$3" && pwd)"
device="$4"
assets="$(cd "$(dirname "$0")" && pwd)"

run_one() {
  local round="$1" label="$2" vendor="$3"
  local status output retained pid
  status="$(npu-smi info -t proc-mem -i "$device" -c 0)"
  echo "$status" | grep -q 'No process in device' || { echo "device busy: $status"; exit 42; }
  echo "START round=$round label=$label time=$(date -u +%FT%TZ)"
  output="$root/formal-r${round}-${label}"
  retained="$root/retained-r${round}-${label}"
  mkdir -p "$output" "$retained"
  (
    source "$vendor/bin/set_env.bash"
    export ASCEND_CUSTOM_OPP_PATH="$vendor"
    export ATK_CUSTOM_OPP_PATH="$vendor/op_api/lib/libcust_opapi.so"
    export LD_LIBRARY_PATH="$vendor/op_api/lib:${LD_LIBRARY_PATH:-}"
    export ASCEND_RT_VISIBLE_DEVICES="$device" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
    python -c 'import os; from atk.tasks.backends.lib_interface.acl_wrapper import get_opp_lib_path; p=get_opp_lib_path("aclnnGroupedMatmulAddV2GetWorkspaceSize"); print("ATK operator library:",p,flush=True); assert os.path.realpath(p)==os.path.realpath(os.environ["ATK_CUSTOM_OPP_PATH"])'
    exec atk aclnn "$assets/legacy6.json" --task performance_device --devices 0 --concurrency 1 --timeout 180 -p "$assets/executor.py" -o "$output"
  ) > "$root/formal-r${round}-${label}.log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    while IFS= read -r csv; do
      local relative="${csv#"$output"/}"
      mkdir -p "$retained/$(dirname "$relative")"
      cp "$csv" "$retained/$relative" 2>/dev/null || true
    done < <(find "$output" -name kernel_details.csv -print 2>/dev/null)
    sleep 0.2
  done
  wait "$pid"
  echo "DONE round=$round label=$label time=$(date -u +%FT%TZ)"
}

for round in 4 5 6 7 8; do
  if [ $((round % 2)) -eq 0 ]; then
    run_one "$round" final-legacy "$current"
    run_one "$round" base-legacy "$base"
  else
    run_one "$round" base-legacy "$base"
    run_one "$round" final-legacy "$current"
  fi
done
