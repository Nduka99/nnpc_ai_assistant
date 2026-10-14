"""The spreadsheet gate: values drawn by seed from the workbooks as stored, saved
readings scored by row label and header. Every workbook and value here is invented."""

import json
from pathlib import Path
from typing import Any

import openpyxl
import pytest
import xlrd

from evaluation import __main__ as cli
from evaluation import sheetvalues

GAS: list[list[Any]] = [
    ["GAS DATA"],
    ["COMPANIES", "GAS PRODUCED", "GAS FLARED"],
    [None, "(MMSCF)", "(MMSCF)"],
    ["SHELL", 2285.25, 398.0],
    ["MOBIL", 25984.7, True],  # a flag is not a number
]


def xlsx(path: Path, rows: list[list[Any]]) -> None:
    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.title = "Gas"
    for row in rows:
        sheet.append(row)
    book.save(path)


def test_a_workbook_s_numeric_cells_are_drawn_by_seed(tmp_path: Path) -> None:
    path = tmp_path / ("a" * 64)
    xlsx(path, GAS)
    grids = sheetvalues.sheets(path)

    assert sheetvalues.numeric_cells(grids) == [
        ("Gas", 4, 2), ("Gas", 4, 3), ("Gas", 5, 2),
    ]  # fmt: skip
    books = {"a" * 64: grids, "b" * 64: {"S": [[1, 2, 3]]}}
    drawn = sheetvalues.draw(books, 5, 3)
    assert drawn == sheetvalues.draw(books, 5, 3)  # seeded
    assert len(drawn) == 3 and [item["number"] for item in drawn] == [1, 2, 3]
    assert sum(1 for item in drawn if item["sha256"] == "b" * 64) <= 2  # per workbook
    assert sheetvalues.draw(books, 5, 9)[-1]["number"] == 4  # all four allowed cells


def test_an_old_xls_workbook_is_read_from_its_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "old"
    path.write_bytes(sheetvalues.OLE2 + bytes(10))

    class Sheet:
        name, nrows = "Liftings", 2

        def row_values(self, number: int) -> list[Any]:
            rows: list[list[Any]] = [["S/N", "QTY"], [1, 949528.0]]
            return rows[number]

    class Book:
        def sheets(self) -> list[Sheet]:
            return [Sheet()]

    seen: list[bytes] = []

    def open_workbook(file_contents: bytes) -> Book:
        seen.append(file_contents)
        return Book()

    monkeypatch.setattr(xlrd, "open_workbook", open_workbook)

    assert sheetvalues.sheets(path) == {"Liftings": [["S/N", "QTY"], [1, 949528.0]]}
    assert seen == [path.read_bytes()]


def test_the_context_shows_the_top_rows_and_marks_the_drawn_cell() -> None:
    grid: list[list[Any]] = [["TITLE"], *[[n, None, ""] for n in range(2, 12)]]
    grid.append(["SHELL", 5.5])

    shown = sheetvalues.context(grid, 12, 2).splitlines()

    assert shown[0] == "[R1C1] 'TITLE'"
    assert len(shown) == 11  # rows 1 to 10 and the drawn row
    assert shown[-1] == "[R12C1] 'SHELL' | [R12C2] 5.5 <== DRAWN"
    assert sheetvalues.context(grid, 2, 1).count("DRAWN") == 1


READING = {
    "elements": [
        {"kind": "table", "cells": []},
        {"kind": "table_row", "cells": []},
        {"kind": "table_row", "cells": [
            ["S/N", "7"], ["COMPANIES", "SHELL"],
            ["GAS PRODUCED (MMSCF)", "2,285.25"], ["REMARK", "ok"]]},
    ]
}  # fmt: skip


def test_a_value_is_placed_by_its_row_label_and_every_header_part() -> None:
    item = {"label": "Shell", "header": ["GAS PRODUCED", "(MMSCF)"], "value": 2285.25}

    assert sheetvalues.placed(item, READING)
    assert sheetvalues.placed({"label": "", "header": ["S/N"], "value": 7}, READING)
    assert not sheetvalues.placed(item | {"label": "MOBIL"}, READING)
    assert not sheetvalues.placed(item | {"header": ["GAS FLARED"]}, READING)
    assert not sheetvalues.placed(item | {"value": 2285.2}, READING)


def test_the_command_line_draws_values_once_and_scores_a_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, registry = tmp_path / "eval", tmp_path / "sets.json"
    readings, store = tmp_path / "elements", tmp_path / "files"
    readings.mkdir()
    store.mkdir()
    sha = "a" * 64
    xlsx(store / sha, GAS)
    sheet_reading = {"fingerprint": {"packages": {"openpyxl": "3"}}, **READING}
    (readings / f"{sha}.json").write_text(json.dumps(sheet_reading))
    pdf = {"fingerprint": {"packages": {"pypdfium2": "5"}}}
    (readings / f"{'b' * 64}.json").write_text(json.dumps(pdf))
    (readings / "report.json").write_text("{}")
    common = ["--root", str(root), "--registry", str(registry)]
    draw = [*common, "sheet-sample", "sheets", "--seed", "3", "--count", "2",
            "--readings", str(readings), "--store", str(store)]  # fmt: skip

    assert cli.main(draw) == 0
    folder = root / "sheets"
    drawn = json.loads((folder / "values.json").read_text(encoding="utf-8"))
    assert drawn["seed"] == 3 and len(drawn["values"]) == 2
    assert len(list((folder / "context").glob("v*.txt"))) == 2
    assert "2 values from 1 workbooks" in capsys.readouterr().out
    assert cli.main(draw) == 1
    assert "drawn once" in capsys.readouterr().err

    values = [
        {"number": 1, "sha256": sha, "label": "SHELL",
         "header": ["GAS PRODUCED"], "value": 2285.25},
        {"number": 2, "sha256": sha, "label": "MOBIL", "header": [], "value": 1.0},
        {"number": 3, "sha256": sha, "exclude": "blank template row"},
    ]  # fmt: skip
    (folder / "truth.json").write_text(json.dumps({"values": values}))
    assert cli.main([*common, "register", "sheets"]) == 0
    out = tmp_path / "score.json"
    score = [*common, "sheet-score", "sheets", "--out", str(out),
             "--readings", str(readings)]  # fmt: skip

    assert cli.main(score) == 1
    printed = capsys.readouterr().out
    assert "values placed: 0.500 of 2 (bar 0.95); missed: 2; excluded: 3" in printed
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["values"] == {"1": True, "2": False} and not report["pass"]

    good = root / "good"
    good.mkdir()
    (good / "truth.json").write_text(json.dumps({"values": values[:1]}))
    assert cli.main([*common, "register", "good"]) == 0
    passing = [*common, "sheet-score", "good", "--out", str(out),
               "--readings", str(readings)]  # fmt: skip
    assert cli.main(passing) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "pass"
