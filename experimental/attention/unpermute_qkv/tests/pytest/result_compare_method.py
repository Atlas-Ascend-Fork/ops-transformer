#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

import datetime
import os
import sys

import torch


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


def check_result(expect, npu_result, name):
    real = npu_result.detach().cpu().reshape(-1)
    comp = expect.detach().cpu().reshape(-1)
    if real.numel() != comp.numel():
        print_log(
            "%s size mismatch: npu %d vs golden %d" % (name, real.numel(), comp.numel())
        )
        return "Failed", float("inf"), float("inf")
    mismatch = int((real != comp).sum())
    max_diff = (
        float(torch.max(torch.abs(real.float() - comp.float())))
        if real.numel() > 0
        else 0.0
    )
    result = "Pass" if mismatch == 0 else "Failed"
    print_log(
        "%s total=%d mismatch=%d max_diff=%.6e result=%s"
        % (name, real.numel(), mismatch, max_diff, result)
    )
    return result, max_diff, float(mismatch)
