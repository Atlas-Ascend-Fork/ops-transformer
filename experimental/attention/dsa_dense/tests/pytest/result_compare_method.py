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
import os
import sys

import torch

MEAN_DIFF_THD = 0.05


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


def check_result(expect, npu_result, name="output", mean_thd=MEAN_DIFF_THD):
    """按浮点误差比较 NPU 输出与 CPU golden：mean(abs diff) <= 阈值 即 Pass。"""
    real = npu_result.detach().cpu().to(torch.float32)
    comp = expect.detach().cpu().to(torch.float32)
    if real.shape != comp.shape:
        print_log(
            "%s shape mismatch: npu %s vs golden %s"
            % (name, tuple(real.shape), tuple(comp.shape))
        )
        return "Failed", float("inf"), float("inf")

    diff = torch.abs(real - comp)
    max_diff = float(torch.max(diff)) if diff.numel() > 0 else 0.0
    mean_diff = float(torch.mean(diff)) if diff.numel() > 0 else 0.0
    result = "Pass" if mean_diff <= mean_thd else "Failed"
    print_log(
        "%-6s total=%d max_diff=%.9f mean_diff=%.9f thd=%.4f result=%s"
        % (name, real.numel(), max_diff, mean_diff, mean_thd, result)
    )
    return result, max_diff, mean_diff
