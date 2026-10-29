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

from dsa_sparse_prolog_golden import OUT_NAMES, check_precision_row, rel_err


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


def check_outputs(op_outs, base_outs, fp64_outs, demote):
    """逐输出计算 op vs torch 基线 / op vs fp64 / torch vs fp64 的 L2 相对误差并判据。"""
    rows = []
    for name, op_o, base_o, t64 in zip(OUT_NAMES, op_outs, base_outs, fp64_outs):
        rows.append(
            {
                "out": name,
                "op_vs_torch": rel_err(op_o.cpu(), base_o.cpu()),
                "op_vs_fp64": rel_err(op_o.cpu(), t64),
                "torch_vs_fp64": rel_err(base_o.cpu(), t64),
            }
        )
    all_pass = all(check_precision_row(r, demote) for r in rows)
    for r in rows:
        mark = "" if check_precision_row(r, demote) else "  <-- FAIL"
        print_log(
            "%-8s op_vs_torch=%.3e op_vs_fp64=%.3e torch_vs_fp64=%.3e%s"
            % (r["out"], r["op_vs_torch"], r["op_vs_fp64"], r["torch_vs_fp64"], mark)
        )
    return all_pass, rows
