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

import datetime
import logging
import os
import sys

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def print_log(data=None, level="INFO"):
    print(
        "[%s] [%s]-%s:%s - %s"
        % (
            datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"),
            level,
            os.path.basename(sys._getframe().f_back.f_code.co_filename),
            str(sys._getframe().f_back.f_lineno).zfill(4),
            data,
        )
    )


def check_result(expect, npu_result):
    """逐元素精确比较 NPU 输出与 CPU golden（索引为整数，要求完全一致）。"""
    real_data = npu_result.cpu().to(torch.int64).numpy().flatten()
    data_compe = expect.cpu().to(torch.int64).numpy().flatten()

    if real_data.size != data_compe.size:
        print_log(
            "Error,the size of npu output[%s] and benchmark[%s] is not equal."
            % (real_data.size, data_compe.size)
        )
        return "Failed", 0.0

    total = real_data.size
    if total == 0:
        print_log('The output is empty, the result of data_compare is "Pass"')
        return "Pass", 100.0

    err_idx = np.where(real_data != data_compe)[0]
    fulfill_percent = float(total - err_idx.size) / float(total) * 100.0
    result = "Pass" if err_idx.size == 0 else "Failed"

    print_log(
        "---------------------------------------------------------------------------------------"
    )
    print_log("Total \t Mismatch \t MatchPct \t Result")
    print_log(
        "%d \t %d \t %.6f%% \t %s" % (total, err_idx.size, fulfill_percent, result)
    )
    if err_idx.size > 0:
        print_log("First 10 mismatch (idx: golden vs npu):")
        for i in err_idx[:10]:
            print_log("  %d: %d vs %d" % (i, data_compe[i], real_data[i]))
    return result, fulfill_percent
