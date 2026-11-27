# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

"""Check that either ATK report summarizer rejects synthetic failures.

Usage: python check_summary_rejects_failure.py ORIGINAL_OUTPUT_DIR
Only copies the first report and necessary logs into a temporary directory; never modifies originals.
"""

import glob
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import openpyxl

assets = Path(__file__).resolve().parent
source = glob.glob(
    str(Path(sys.argv[1]) / "formal-r1-base-legacy" / "**" / "*.xlsx"), recursive=True
)
if len(source) != 1:
    raise RuntimeError(("expected one original report", source))
for script in ("summarize_atk_perf.py", "summarize_atk_legacy8.py"):
    for failure in ("status", "failed_case_zero", "dtype"):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            destination = root / "formal-r1-base-legacy" / "test.xlsx"
            destination.parent.mkdir()
            shutil.copyfile(source[0], destination)
            book = openpyxl.load_workbook(destination)
            if failure == "status":
                sheet = book["statistic"]
                headers = {cell.value: cell.column for cell in sheet[1]}
                sheet.cell(2, headers["运行结果"]).value = "FAILURE"
                expected = "ATK case failed"
            elif failure == "failed_case_zero":
                book["failed cases"].append([0])
                expected = "ATK failed cases sheet not empty"
            book.save(destination)
            if failure == "dtype":
                original_root = Path(sys.argv[1])
                shutil.copytree(
                    original_root / "retained-r1-base-legacy",
                    root / "retained-r1-base-legacy",
                )
                log = (original_root / "formal-r1-base-legacy.log").read_text(
                    errors="replace"
                )
                if "dtype=torch.float16" not in log:
                    raise RuntimeError(
                        "expected float16 input missing from original log"
                    )
                (root / "formal-r1-base-legacy.log").write_text(
                    log.replace("dtype=torch.float16", "dtype=torch.bfloat16", 1)
                )
                expected = "input dtype/shape mismatch with case JSON"
            process = subprocess.run(
                [sys.executable, str(assets / script), str(root)],
                capture_output=True,
                text=True,
            )
            if process.returncode == 0 or expected not in process.stderr:
                raise RuntimeError(
                    (
                        script,
                        failure,
                        process.returncode,
                        process.stdout,
                        process.stderr,
                    )
                )
            print("REJECTED", script, failure, "exit", process.returncode)
