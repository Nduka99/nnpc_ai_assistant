"""Spreadsheet values placed in their row and column (K16; Lime Green X8's "values
placed" measure, for workbooks).

Values are drawn by one seeded generator from the numeric cells of the workbooks as
stored, read here with openpyxl or xlrd directly, never through the reader measured.
For each drawn value the truth writer sees the raw grid around it and writes the
value's row label and the parts of its column header (a group label, the column's
name, a unit). A value is placed when the reading has a row holding the label in some
cell and whose cell holding the value sits under a header that contains every part.
A value with no row text and no header text is excluded by the truth writer. The
harness reads saved files only; it never imports the engine it measures.
"""

import math
import random
from io import BytesIO
from pathlib import Path
from typing import Any

from evaluation import parsing

PER_WORKBOOK = 2  # so one long sheet cannot fill the set
CONTEXT_ROWS = 10  # rows shown from a sheet's top, where its title and header are
OLE2 = b"\xd0\xcf\x11\xe0"  # an old .xls workbook; .xlsx files are zip packages

Grid = list[list[Any]]


def sheets(path: Path) -> dict[str, Grid]:
    """Every sheet's cells as stored, by sheet name."""
    data = path.read_bytes()
    if data[:4] == OLE2:
        import xlrd

        book = xlrd.open_workbook(file_contents=data)
        return {
            sheet.name: [sheet.row_values(n) for n in range(sheet.nrows)]
            for sheet in book.sheets()
        }
    import openpyxl

    # A stream, since a stored file has no extension for openpyxl to check.
    workbook = openpyxl.load_workbook(BytesIO(data), read_only=True, data_only=True)
    try:
        return {
            sheet.title: [list(row) for row in sheet.iter_rows(values_only=True)]
            for sheet in workbook.worksheets
        }
    finally:
        workbook.close()


def is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def numeric_cells(grids: dict[str, Grid]) -> list[tuple[str, int, int]]:
    """Every (sheet, row, column) holding a number, from 1, in sheet order."""
    found = []
    for name, grid in grids.items():
        for row, cells in enumerate(grid, 1):
            for column, value in enumerate(cells, 1):
                if is_number(value):
                    found.append((name, row, column))
    return found


def draw(
    workbooks: dict[str, dict[str, Grid]], seed: int, count: int
) -> list[dict[str, Any]]:
    """`count` numeric cells in a seeded order, at most PER_WORKBOOK per workbook,
    numbered in drawing order."""
    cells = [
        (sha256, sheet, row, column)
        for sha256 in sorted(workbooks)
        for sheet, row, column in numeric_cells(workbooks[sha256])
    ]
    random.Random(seed).shuffle(cells)
    taken: dict[str, int] = {}
    chosen: list[dict[str, Any]] = []
    for sha256, sheet, row, column in cells:
        if len(chosen) == count:
            break
        if taken.get(sha256, 0) == PER_WORKBOOK:
            continue
        taken[sha256] = taken.get(sha256, 0) + 1
        value = workbooks[sha256][sheet][row - 1][column - 1]
        chosen.append({"number": len(chosen) + 1, "sha256": sha256, "sheet": sheet,
                       "row": row, "column": column, "value": value})  # fmt: skip
    return chosen


def context(grid: Grid, row: int, column: int) -> str:
    """The grid's top rows and the drawn value's row, each cell with its address,
    for the truth writer; the drawn cell is marked."""
    shown = list(range(1, min(CONTEXT_ROWS, len(grid)) + 1))
    if row not in shown:
        shown.append(row)
    lines = []
    for number in shown:
        cells = []
        for index, value in enumerate(grid[number - 1], 1):
            if value is None or value == "":
                continue
            mark = " <== DRAWN" if (number, index) == (row, column) else ""
            cells.append(f"[R{number}C{index}] {value!r}{mark}")
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def same_number(cell: str, value: float) -> bool:
    """Whether a cell's text is the number `value` (placement is measured, not how a
    number is written)."""
    try:
        return math.isclose(float(cell.replace(",", "")), value, rel_tol=1e-12)
    except ValueError:
        return False


def placed(item: dict[str, Any], reading: dict[str, Any]) -> bool:
    """Whether the reading holds the item's value in a row holding its row label in
    some cell, under a column header containing every part of its header."""
    label = parsing.grid_fold(item["label"])
    parts = [parsing.grid_fold(part) for part in item["header"]]
    for element in reading["elements"]:
        cells = element["cells"]
        if element["kind"] != "table_row" or not cells:
            continue
        if not any(label in parsing.grid_fold(cell) for _, cell in cells):
            continue
        for header, cell in cells:  # the first column is a column too (S/N)
            folded = parsing.grid_fold(header)
            if same_number(cell, item["value"]) and all(p in folded for p in parts):
                return True
    return False
