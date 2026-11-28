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
"""Regressions for blank XLSX cached values in the standard-library reader."""

import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest

MAIN = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT = "application/vnd.openxmlformats-officedocument.spreadsheetml"


@pytest.fixture
def reader(monkeypatch):
    monkeypatch.setitem(sys.modules, "openpyxl", None)
    source = Path(__file__).resolve().parents[1] / "assets" / "_xlsx_minireader.py"
    spec = importlib.util.spec_from_file_location("xlsx_empty_cache_reader", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_workbook(tmp_path, value_cells):
    parts = {
        "[Content_Types].xml": (
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
            '<Default Extension="xml" ContentType="application/xml"/>\n'
            f'<Override PartName="/xl/workbook.xml" ContentType="{CONTENT}.sheet.main+xml"/>\n'
            f'<Override PartName="/xl/worksheets/sheet1.xml" ContentType="{CONTENT}.worksheet+xml"/>\n'
            f'<Override PartName="/xl/sharedStrings.xml" ContentType="{CONTENT}.sharedStrings+xml"/>\n'
            "</Types>"
        ),
        "_rels/.rels": (
            f'<Relationships xmlns="{PACKAGE_REL}">\n'
            f'<Relationship Id="rId1" Type="{REL}/officeDocument" Target="xl/workbook.xml"/>\n'
            "</Relationships>"
        ),
        "xl/workbook.xml": (
            f'<workbook xmlns="{MAIN}" xmlns:r="{REL}"><sheets>'
            '<sheet name="Cases" sheetId="1" r:id="rId1"/></sheets></workbook>'
        ),
        "xl/_rels/workbook.xml.rels": (
            f'<Relationships xmlns="{PACKAGE_REL}">\n'
            f'<Relationship Id="rId1" Type="{REL}/worksheet" Target="worksheets/sheet1.xml"/>\n'
            f'<Relationship Id="rId2" Type="{REL}/sharedStrings" Target="sharedStrings.xml"/>\n'
            "</Relationships>"
        ),
        "xl/sharedStrings.xml": (
            f'<sst xmlns="{MAIN}" count="1" uniqueCount="1"><si><t>case</t></si></sst>'
        ),
        "xl/worksheets/sheet1.xml": (
            f'<worksheet xmlns="{MAIN}"><sheetData><row r="1">'
            '<c r="A1" t="s"><v>0</v></c>'
            f"{value_cells}</row></sheetData></worksheet>"
        ),
    }
    path = tmp_path / "cases.xlsx"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, xml in parts.items():
            archive.writestr(name, xml)
    return str(path)


def check_sheet(reader, workbook, expected):
    assert reader.load_sheet1(workbook) == [{0: "case", **expected}]
    assert reader.load_sheet(workbook, "Cases") == [{0: "case", **expected}]


@pytest.mark.parametrize("cache", ["<v/>", "<v></v>", ""])
def test_formula_with_empty_or_missing_cache(tmp_path, reader, cache):
    path = write_workbook(tmp_path, f'<c r="B1"><f>1+2</f>{cache}</c>')
    check_sheet(reader, path, {1: None})


@pytest.mark.parametrize("value", ["<v/>", "<v></v>"])
@pytest.mark.parametrize("kind", ["n", "s", "b"])
def test_typed_cell_with_empty_value(tmp_path, reader, value, kind):
    path = write_workbook(tmp_path, f'<c r="B1" t="{kind}">{value}</c>')
    check_sheet(reader, path, {1: None})


def test_inline_text_has_priority_over_empty_cached_value(tmp_path, reader):
    path = write_workbook(
        tmp_path, '<c r="B1" t="inlineStr"><is><t>中文</t></is><v/></c>'
    )
    check_sheet(reader, path, {1: "中文"})


def test_populated_values_unchanged(tmp_path, reader):
    path = write_workbook(
        tmp_path,
        '<c r="B1"><f>1+2</f><v>3</v></c>'
        '<c r="C1"><v>0</v></c>'
        '<c r="D1" t="b"><v>0</v></c>'
        '<c r="E1" t="b"><v>1</v></c>'
        '<c r="F1" t="inlineStr"><is><t>中文</t></is></c>'
        '<c r="G1" t="str"><f>"text"</f><v>text</v></c>',
    )
    check_sheet(
        reader, path, {1: "3", 2: "0", 3: "False", 4: "True", 5: "中文", 6: "text"}
    )
