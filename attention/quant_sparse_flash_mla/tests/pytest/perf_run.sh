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

exec bash "$(dirname "$0")/../../../sparse_flash_mla/tests/pytest/perf_run_common.sh" \
    QSMLA_RESULT_SAVE_PATH "result.xlsx" \
    "${QSMLA_PERF_TEST_SCRIPT:-test_quant_sparse_flash_mla_batch_graph.py}" \
    graph "${QSMLA_PERF_GOLDEN_PATH:-perf_golden.xlsx}" "$@"
