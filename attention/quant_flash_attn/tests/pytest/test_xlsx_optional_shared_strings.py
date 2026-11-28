#!/usr/bin/python3
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
"""Regressions for the standard-library XLSX reader's optional string table."""

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT = "application/vnd.openxmlformats-officedocument.spreadsheetml"
DEFAULT = "xl/sharedStrings.xml"


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    source = Path(__file__).resolve().parents[1] / "assets" / "_xlsx_minireader.py"
    spec = importlib.util.spec_from_file_location(
        "xlsx_optional_strings_reader", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_workbook(
    tmp_path, cells, parts=None, shared_relation=None, external=False, omit_rels=False
):
    parts = parts or {}
    types = [
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
        '<Default Extension="xml" ContentType="application/xml"/>',
        f'<Override PartName="/xl/workbook.xml" ContentType="{CONTENT}.sheet.main+xml"/>',
        f'<Override PartName="/xl/worksheets/sheet1.xml" ContentType="{CONTENT}.worksheet+xml"/>',
    ]
    types.extend(
        f'<Override PartName="/{part}" ContentType="{CONTENT}.sharedStrings+xml"/>'
        for part in parts
    )
    relationships = [
        f'<Relationships xmlns="{PACKAGE_REL}">',
        f'<Relationship Id="rId1" Type="{REL}/worksheet" Target="worksheets/sheet1.xml"/>',
    ]
    if shared_relation is not None:
        target_mode = ' TargetMode="External"' if external else ""
        relationships.append(
            f'<Relationship Id="rId2" Type="{REL}/sharedStrings" '
            f'Target="{shared_relation}"{target_mode}/>'
        )
    path = tmp_path / "cases.xlsx"
    with zipfile.ZipFile(path, "w") as book:
        book.writestr("[Content_Types].xml", "\n".join([*types, "</Types>"]))
        book.writestr(
            "_rels/.rels",
            f'<Relationships xmlns="{PACKAGE_REL}">'
            f'<Relationship Id="rId1" Type="{REL}/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        book.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{MAIN}" xmlns:r="{REL}"><sheets>'
            '<sheet name="Cases" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        if not omit_rels:
            book.writestr(
                "xl/_rels/workbook.xml.rels",
                "\n".join([*relationships, "</Relationships>"]),
            )
        book.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{MAIN}"><sheetData><row r="1">'
            f"{cells}</row></sheetData></worksheet>",
        )
        for part, xml in parts.items():
            book.writestr(part, f'<sst xmlns="{MAIN}"><si>{xml}</si></sst>')
    return str(path)


def test_inline_strings_without_optional_dependency(tmp_path, reader):
    path = make_workbook(
        tmp_path,
        '<c r="A1" t="inlineStr"><is><t>用例名</t></is></c>'
        '<c r="B1" t="b"><v>1</v></c><c r="C1"><v>4096</v></c><c r="D1"/>',
    )
    assert reader.load_sheet1(path) == [{0: "用例名", 1: "True", 2: "4096", 3: None}]


def test_named_sheet_with_inline_rich_text(tmp_path, reader):
    path = make_workbook(
        tmp_path,
        '<c r="A1" t="inlineStr"><is><r><t>case_</t></r><r><t>one</t></r></is></c>',
    )
    assert reader.load_sheet(path, "Cases") == [{0: "case_one"}]


def test_numeric_only_sheet_without_shared_strings(tmp_path, reader):
    path = make_workbook(
        tmp_path, '<c r="A1"><v>0.125</v></c><c r="B1" t="b"><v>0</v></c>'
    )
    assert reader.load_sheet(path, 1) == [{0: "0.125", 1: "False"}]


def test_shared_strings_remain_supported(tmp_path, reader):
    path = make_workbook(
        tmp_path,
        '<c r="A1" t="s"><v>0</v></c><c r="B1" t="inlineStr"><is><t>inline</t></is></c>',
        {DEFAULT: "<r><t>shared_</t></r><r><t>value</t></r>"},
    )
    assert reader.load_sheet1(path) == [{0: "shared_value", 1: "inline"}]


@pytest.mark.parametrize(
    "target,part",
    [
        ("caseLabels.xml", "xl/caseLabels.xml"),
        ("/xl/strings/caseLabels.xml", "xl/strings/caseLabels.xml"),
    ],
)
def test_internal_relationship_supplies_missing_default(tmp_path, reader, target, part):
    path = make_workbook(
        tmp_path, '<c r="A1" t="s"><v>0</v></c>', {part: "<t>custom</t>"}, target
    )
    assert reader.load_sheet1(path) == [{0: "custom"}]


def test_default_part_wins_when_both_parts_exist(tmp_path, reader):
    path = make_workbook(
        tmp_path,
        '<c r="A1" t="s"><v>0</v></c>',
        {DEFAULT: "<t>default</t>", "xl/custom.xml": "<t>custom</t>"},
        "custom.xml",
    )
    assert reader.load_sheet1(path) == [{0: "default"}]


@pytest.mark.parametrize(
    "settings",
    [
        {"omit_rels": True},
        {"shared_relation": "absent.xml"},
        {"shared_relation": "https://example.invalid/strings.xml", "external": True},
    ],
)
def test_no_usable_relationship_preserves_inline_cells(tmp_path, reader, settings):
    path = make_workbook(
        tmp_path, '<c r="A1" t="inlineStr"><is><t>inline</t></is></c>', **settings
    )
    assert reader.load_sheet1(path) == [{0: "inline"}]


@pytest.mark.parametrize("index,parts", [(0, {}), (-1, {DEFAULT: "<t>label</t>"})])
def test_invalid_shared_string_index_has_context(tmp_path, reader, index, parts):
    path = make_workbook(tmp_path, f'<c r="A1" t="s"><v>{index}</v></c>', parts)
    with pytest.raises(
        ValueError, match=rf"shared string index {index}.*sharedStrings"
    ):
        reader.load_sheet1(path)
