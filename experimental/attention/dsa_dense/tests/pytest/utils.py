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


def save_result(results, params, result_path="./result/dsa_result.xlsx"):
    """results: {name: (result, max_diff, mean_diff)}"""
    result_path = Path(result_path)
    row_data = {
        "case_name": params[3],
        "batch": params[0],
        "t1_base": params[1],
        "n": params[2],
    }
    all_pass = True
    for name in ["loss", "dq", "dk", "dw"]:
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
            print(f"Excel列名: {list(df.columns)}")
            print(f"变量名: {list(row_data.keys())}")
            return False
        new_df = pd.DataFrame([row_data])
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = pd.DataFrame([row_data])

    df.to_excel(result_path, index=False)
    return True
