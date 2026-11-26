# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Canonical MQSMLA testcase names, including historical QSMLA prefixes."""

import re
from pathlib import Path


def normalise_mqsmla_case_name(value):
    """Preserve case IDs while correcting legacy operator/file prefixes."""
    if not isinstance(value, str):
        return value
    name = Path(value).name
    if name.casefold().endswith(".pt"):
        name = name[:-3]

    def replace_prefix(match):
        token = match.group()
        if token.isupper():
            return "MQSMLA"
        if token.istitle():
            return "Mqsmla"
        return "mqsmla"

    return re.sub(r"(?<![a-z])qsmla", replace_prefix, name, flags=re.IGNORECASE)


def normalise_mqsmla_test_data(input_data):
    """Copy testcase labels without copying or changing tensors and golden data."""
    data = input_data.copy()
    if "Testcase_Name" in data:
        data["Testcase_Name"] = normalise_mqsmla_case_name(data["Testcase_Name"])
    if isinstance(data.get("params"), dict):
        data["params"] = data["params"].copy()
        if "Testcase_Name" in data["params"]:
            data["params"]["Testcase_Name"] = normalise_mqsmla_case_name(
                data["params"]["Testcase_Name"]
            )
    return data
