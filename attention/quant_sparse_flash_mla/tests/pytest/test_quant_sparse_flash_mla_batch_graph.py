#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run saved QSMLA cases in graph mode with the shared accuracy checks."""

import pytest

from test_quant_sparse_flash_mla_batch import qsmla, testcase_files, testcase_ids


@pytest.mark.graph
@pytest.mark.parametrize("testcase_files", testcase_files, ids=testcase_ids)
def test_quant_sparse_flash_mla(testcase_files):
    qsmla(testcase_files, run_graph=True)
