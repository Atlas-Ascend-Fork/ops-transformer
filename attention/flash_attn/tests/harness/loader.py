# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""用例加载：JSON 目录/文件与 CSV，含测试集/用例名/跳过过滤。"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .case import Case, load_cases_file


class CaseSelectionError(RuntimeError):
    pass


def _match(case: Case, patterns: Sequence[str]) -> bool:
    """模式匹配：无通配符按子串；有通配符匹配全名/短名/可读前缀（hash 后缀不影响）。"""
    cid = case.id.lower()
    short = cid.split("/", 1)[-1]
    readable = short.split("__", 1)[0]
    for pat in patterns:
        target = pat.lower()
        if any(ch in target for ch in "*?["):
            if (
                fnmatch.fnmatch(cid, target)
                or fnmatch.fnmatch(short, target)
                or fnmatch.fnmatch(readable, target)
            ):
                return True
        elif target in cid:
            return True
    return False


def load_from_json(sources: Iterable[str]) -> List[Case]:
    cases: List[Case] = []
    for src in sources:
        path = Path(src)
        if path.is_dir():
            for file in sorted(path.glob("*.json")):
                cases.extend(load_cases_file(file))
        elif path.suffix.lower() == ".json":
            cases.extend(load_cases_file(path))
        else:
            raise CaseSelectionError(f"unsupported case source: {src}")
    return cases


def select(
    cases: Sequence[Case],
    case_ids: Optional[Sequence[str]] = None,
    testsets: Optional[Sequence[str]] = None,
    skip: Optional[Sequence[str]] = None,
) -> List[Case]:
    """按用例集(tag)/用例名(前缀或 glob)/跳过表过滤；顺序保持稳定。"""
    picked = list(cases)
    if testsets:
        picked = [case for case in picked if any(tag in testsets for tag in case.tags)]
    if case_ids:
        picked = [case for case in picked if _match(case, case_ids)]
    if skip:
        picked = [case for case in picked if not _match(case, skip)]
    # 去重（同一 case_id 只保留第一次出现）
    seen: Dict[str, Case] = {}
    for case in picked:
        seen.setdefault(case.id, case)
    return list(seen.values())


def load_skip_file(path: Optional[str]) -> List[str]:
    if not path:
        return []
    target = Path(path)
    if not target.exists():
        raise CaseSelectionError(f"skip file not found: {path}")
    patterns: List[str] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        patterns.append(text.split()[0])
    return patterns
