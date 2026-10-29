#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0

from pathlib import Path

import pandas as pd


def save_result(results, params, result_path="./result/uq_result.xlsx"):
    result_path = Path(result_path)
    row_data = {
        "case_name": params[7],
        "s_c": params[0],
        "b": params[1],
        "split_size": params[3],
        "n_k": params[5],
    }
    all_pass = True
    for name in ["q", "k", "v"]:
        result, max_diff, mismatch = results[name]
        row_data[f"{name}_result"] = result
        row_data[f"{name}_max_diff"] = max_diff
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
