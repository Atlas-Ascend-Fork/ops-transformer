# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Case 数据模型：JSON 真源 + 稳定命名（算子无关）。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

SCHEMA = "case/v1"
NORM_VERSION = "n2"  # 归一化规则版本：规则变更需 bump，避免旧 id 被误复用


def _num(value: Any) -> Any:
    """规范化数值表示，保证跨机器/跨 Python 版本的 hash 稳定。"""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _num(value.tolist())
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return format(value, ".17g")
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_num(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _num(v) for k, v in sorted(value.items())}
    return value


def canonical_params(params: Dict[str, Any]) -> str:
    return json.dumps(
        _num(dict(params)), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def params_hash(params: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_params(params).encode("utf-8")).hexdigest()[:8]


def make_case_id(op: str, params: Dict[str, Any], name: str = "") -> str:
    # norm 版本参与 hash：归一化规则变化时 id 会变，避免与旧基线静默混用
    prefix = name or params_hash(params)
    return f"{op}/{prefix}__{params_hash({**params, '_norm': NORM_VERSION})}"


@dataclass
class Case:
    """一条用例：params 是唯一权威，其余为元数据。"""

    op: str
    params: Dict[str, Any]
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        from .ops.flash_attn import PACKAGE

        self.params = PACKAGE.normalize(self.params)

    @property
    def id(self) -> str:
        from .ops.flash_attn import PACKAGE

        return make_case_id(self.op, self.params, PACKAGE.case_name(self.params))

    @property
    def seed(self) -> int:
        return self.params.get("seed", int(params_hash(self.params), 16) % (2**31))

    def signature(self) -> Dict[str, Any]:
        """完整指纹：用于数据文件/缓存校验。"""
        return {
            "norm": NORM_VERSION,
            "op": self.op,
            "params": _num(dict(self.params)),
        }

    def signature_digest(self) -> str:
        raw = json.dumps(self.signature(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "case_id": self.id,
            "params": self.params,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Case":
        params = dict(data.get("params") or {})
        return cls(
            op=str(data.get("op") or "flash_attn"),
            params=params,
            tags=list(data.get("tags") or []),
        )


def _validate_case_payload(data, path):
    if isinstance(data, dict):
        schema = data.get("schema")
        if schema is not None and schema != SCHEMA:
            raise ValueError(f"invalid case file {path}: unsupported schema {schema!r}")
        rows = data.get("cases")
        if not isinstance(rows, list):
            raise ValueError(f"invalid case file {path}: cases must be a list")
        count = data.get("count")
        if count is not None:
            try:
                count = int(count)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid case file {path}: count must be an integer"
                ) from exc
            if count != len(rows):
                raise ValueError(
                    f"invalid case file {path}: count={count} != len(cases)={len(rows)}"
                )
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError(f"invalid case file {path}: top level must be mapping or list")

    for index, row in enumerate(rows):
        where = f"invalid case file {path}: cases[{index}]"
        if not isinstance(row, dict):
            raise ValueError(f"{where} must be an object")
        if not isinstance(row.get("params"), dict):
            raise ValueError(f"{where}.params must be an object")
        tags = row.get("tags")
        if tags is not None and (
            not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags)
        ):
            raise ValueError(f"{where}.tags must be a string array")
        op = row.get("op")
        if op is not None and (not isinstance(op, str) or not op.strip()):
            raise ValueError(f"{where}.op must be a non-empty string")
        case_schema = row.get("schema")
        if case_schema is not None and case_schema != SCHEMA:
            raise ValueError(f"{where}.schema must be {SCHEMA!r}")
    return rows


def load_cases_file(path) -> List[Case]:
    """JSON 文件：{"cases": [...]} 或直接 [...]。"""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = _validate_case_payload(data, path)
    return [Case.from_dict(row) for row in rows]


def dump_cases_file(cases: List[Case], path) -> None:
    """每条用例一行（紧凑 JSON），便于 diff 与人工阅读。"""
    lines = [
        "{",
        f'  "schema": "{SCHEMA}",',
        f'  "count": {len(cases)},',
        '  "cases": [',
    ]
    for index, case in enumerate(cases):
        body = json.dumps(case.to_dict(), ensure_ascii=False, separators=(",", ":"))
        comma = "," if index + 1 < len(cases) else ""
        lines.append(f"    {body}{comma}")
    lines.extend(["  ]", "}", ""])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
