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

from pathlib import Path

import pandas as pd


def save_result(results, params, result_path="./result/fag_result.xlsx"):
    """results: {name: (result, max_diff, mean_diff)}"""
    result_path = Path(result_path)
    row_data = {
        "case_name": params[6],
        "n": params[2],
        "d": params[3],
        "causal": params[4],
        "dtype": str(params[5]),
    }
    all_pass = True
    for name in ["dq", "dk", "dv"]:
        result, max_diff, mean_diff = results[name]
        row_data[f"{name}_result"] = result
        row_data[f"{name}_max_diff"] = max_diff
        row_data[f"{name}_mean_diff"] = mean_diff
        if result != "Pass":
            all_pass = False
    row_data["result"] = "Pass" if all_pass else "Failed"

    result_path.parent.mkdir(parents=True, exist_ok=True)
    if result_path.exists():
        df = pd.read_excel(result_path)
        if set(df.columns) != set(row_data.keys()):
            print("警告：变量名与Excel列名不匹配！")
            return False
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
    else:
        df = pd.DataFrame([row_data])

    df.to_excel(result_path, index=False)
    return True
