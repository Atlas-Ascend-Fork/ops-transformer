#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -eo pipefail

# Use the configured CANN installation; CUSTOM_OPP_ROOT overrides its vendor path.
rotary_cann_root="${CANN_ROOT:-${ASCEND_HOME_PATH:-}}"
if [[ -z "${rotary_cann_root}" ]]; then
    printf 'Set CANN_ROOT or source the CANN set_env.sh before running this script.\n' >&2
    exit 1
fi
rotary_custom_opp="${CUSTOM_OPP_ROOT:-${rotary_cann_root}/opp/vendors/custom_transformer}"
source "${rotary_cann_root}/set_env.sh"
source "${rotary_custom_opp}/bin/set_env.bash"

rotary_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python "${rotary_script_dir}/test_torch_partial_rotary_mul_quant.py" "$@"
