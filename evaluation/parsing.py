"""Score PDF parsers against ground truth written from page images (experiment X8).

Each parser's output for a page is a list of text units in reading order (the text
layer's lines, or Docling's elements) plus any table rows it rebuilt. The measures
and the gate are the ones registered in evaluation/reports/X8-pdf-parsing.md.
Everything is compared with whitespace removed and case ignored, so spacing
differences (a superscript read as "m 2") never count as errors.
"""

import re
import statistics
import unicodedata
from collections import Counter
from typing import Any

Rows = list[tuple[tuple[str, str], ...]]  # table rows as (header, value) cells
Grid = tuple[tuple[str, ...], ...]  # a whole table's rows, header rows included
NUMBER = re.compile(r"[0-9]+(?:[.,][0-9]+)?")
# Punctuation at a word's edges: a rebuilt table row adds separators ("A1;").
EDGES = ".,;:!?()[]{}\"'‘’“”—–-"
GATE = {"cells": 0.95, "pairs": 0.95, "words": 0.99, "sentences": 0.95, "numbers": 0.99}
LOWEST_TABLE = 0.80
# Round 3's gate: values in their row and column of the whole table, whatever the
# parser flags as a header (with the lowest table's share at LOWEST_TABLE).
GRID_GATE = {"grid": 0.95, "words": 0.99, "numbers": 0.99}
# Round 2's bar for indexing a model's transcriptions as searchable text.
TRANSCRIPTION = {"recall": 0.90, "precision": 0.95}


# Docling's PDF parser writes typographic characters in their plain form (its
# default sanitisation), so both sides of every comparison are folded the same way.
TYPOGRAPHY = str.maketrans(dict.fromkeys('‘’“”"', "'") | dict.fromkeys("–—", "-"))


def fold(text: str) -> str:
    return text.translate(TYPOGRAPHY).casefold()


def squash(text: str) -> str:
    return "".join(fold(text).split())


def grid_fold(text: str) -> str:
    """For grid comparisons, compatibility forms are folded too (a subscript ₂ is a
    2), since truth is written from the image and PDFs encode such forms either way.
    The naira sign is an N: NNPC's PDFs draw ₦ as an N with a separate stroke, and
    their text layer holds the N (pdfs-v1 amendment 1)."""
    return squash(unicodedata.normalize("NFKC", text).replace("₦", "N"))


def placed(label: str, header: str, value: str, grids: list[Grid]) -> bool:
    """Whether a parsed table holds this value in a row whose first cell holds the
    label, in a column whose cell in some row above is the header (an empty label or
    header matches any row or column)."""
    for grid in grids:
        for index, row in enumerate(grid):
            if not row or grid_fold(label) not in grid_fold(row[0]):
                continue
            for column in range(1, len(row)):
                if grid_fold(row[column]) != grid_fold(value):
                    continue
                above = [line[column] for line in grid[:index] if column < len(line)]
                if not header or grid_fold(header) in map(grid_fold, above):
                    return True
    return False


def grid_cells(
    tables: list[dict[str, Any]], grids: list[Grid]
) -> tuple[int, int, list[float]]:
    """Truth values of tables written as whole grids (`grid`, with `header_rows`),
    found in their row and column; each table's share. The header of a value is its
    column's text in the lowest header row."""
    found = 0
    total = 0
    shares = []
    for table in tables:
        if "grid" not in table:
            continue
        rows = table["grid"]
        heads = table["header_rows"]
        hits = 0
        cells = 0
        for row in rows[heads:]:
            for column in range(1, len(row)):
                if not row[column]:
                    continue
                cells += 1
                header = rows[heads - 1][column] if heads else ""
                hits += placed(row[0], header, row[column], grids)
        found += hits
        total += cells
        shares.append(hits / cells if cells else 1.0)
    return found, total, shares


def cell_found(label: str, header: str, value: str, rows: Rows) -> bool:
    """Whether a parsed row with this label holds this value under this header. A
    parsed header keeps its levels ("Performance › Class i"); the truth header must be
    one of them ("Class i" must not match "Class ii"), or empty for any column."""
    for cells in rows:
        if not cells or squash(label) not in squash(cells[0][1]):
            continue
        for column_header, cell in cells[1:]:
            parts = {squash(part) for part in column_header.split(" › ")}
            same_column = not header or squash(header) in parts
            if same_column and squash(cell) == squash(value):
                return True
    return False


def table_cells(
    tables: list[dict[str, Any]], rows: Rows
) -> tuple[int, int, list[float]]:
    """Truth cells found as (row label, column header, value); each table's share."""
    found = 0
    total = 0
    shares = []
    for table in tables:
        header, body = header_and_rows(table)
        hits = 0
        cells = 0
        for row in body:
            for column in range(1, len(row)):
                if not row[column]:
                    continue
                cells += 1
                if cell_found(row[0], header[column], row[column], rows):
                    hits += 1
        found += hits
        total += cells
        shares.append(hits / cells if cells else 1.0)
    return found, total, shares


def header_and_rows(table: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    """A truth table's column headers and data rows, whether written as a header and
    rows (rounds 1 and 2) or as a whole grid (round 3: the lowest header row)."""
    if "grid" not in table:
        return table["header"], table["rows"]
    heads = table["header_rows"]
    rows = table["grid"]
    header = rows[heads - 1] if heads else [""] * len(rows[0])
    return header, rows[heads:]


def words(text: str) -> Counter[str]:
    """The text's words, without case or punctuation at their edges."""
    found = (word.strip(EDGES) for word in fold(text).split())
    return Counter(word for word in found if word)


def words_kept(reference: str, units: list[str]) -> tuple[int, int]:
    """Text-layer words that appear in the parser's text, counted as a multiset."""
    wanted = words(reference)
    have = words(" ".join(units))
    kept = sum(min(count, have[word]) for word, count in wanted.items())
    return kept, sum(wanted.values())


def sentences_whole(sentences: list[str], units: list[str]) -> tuple[int, int]:
    """Truth sentences found as continuous text in the parser's reading order."""
    text = squash(" ".join(units))
    return sum(squash(sentence) in text for sentence in sentences), len(sentences)


def table_numbers(tables: list[dict[str, Any]], units: list[str]) -> tuple[int, int]:
    """Numbers in truth table cells that appear in the parser's text."""
    text = squash(" ".join(units))
    numbers = [
        number
        for table in tables
        for row in header_and_rows(table)[1]
        for value in row
        for number in NUMBER.findall(value)
    ]
    return sum(number in text for number in numbers), len(numbers)


def pairs_kept(pairs: list[list[str]], units: list[str]) -> tuple[int, int]:
    """Label-value pairs whose value sits in the label's unit or the next one."""
    kept = 0
    squashed = [squash(unit) for unit in units]
    for label, value in pairs:
        for index, unit in enumerate(squashed):
            if squash(label) not in unit:
                continue
            following = squashed[index + 1] if index + 1 < len(squashed) else ""
            if squash(value) in unit or squash(value) in following:
                kept += 1
                break
    return kept, len(pairs)


def score_page(
    page: dict[str, Any],
    units: list[str],
    rows: Rows,
    reference: str,
    grids: list[Grid] | None = None,
) -> dict[str, Any]:
    """Every measure for one parser on one truth page, as (found, total) counts."""
    found, total, shares = table_cells(page["tables"], rows)
    placed_found, placed_total, grid_shares = grid_cells(page["tables"], grids or [])
    return {
        "cells": (found, total),
        "table_shares": shares,
        "grid": (placed_found, placed_total),
        "grid_shares": grid_shares,
        "pairs": pairs_kept(page["pairs"], units),
        "words": words_kept(reference, units),
        "sentences": sentences_whole(page["sentences"], units),
        "numbers": table_numbers(page["tables"], units),
    }


def pooled(pages: list[dict[str, Any]]) -> dict[str, float]:
    """Each measure's share over all pages (1.0 where a measure has nothing to find)."""
    result = {}
    for measure in [*GATE, "grid"]:
        found = sum(page[measure][0] for page in pages)
        total = sum(page[measure][1] for page in pages)
        result[measure] = found / total if total else 1.0
    return result


def gate(
    arm: dict[str, float], pypdf: dict[str, float], lowest: float
) -> dict[str, bool]:
    """The registered gate, item by item, for one arm."""
    return {
        "1. table cells and pairs": arm["cells"] >= GATE["cells"]
        and arm["pairs"] >= GATE["pairs"]
        and lowest >= LOWEST_TABLE,
        "2. text kept": arm["words"] >= GATE["words"],
        "3. sentences whole": arm["sentences"] >= GATE["sentences"]
        and arm["sentences"] >= pypdf["sentences"],
        "4. table numbers": arm["numbers"] >= GATE["numbers"],
    }


def grid_gate(arm: dict[str, float], lowest: float) -> dict[str, bool]:
    """Round 3's registered gate, item by item, for one arm."""
    return {
        "1. table values in their row and column": arm["grid"] >= GRID_GATE["grid"]
        and lowest >= LOWEST_TABLE,
        "2. text kept": arm["words"] >= GRID_GATE["words"],
        "3. table numbers": arm["numbers"] >= GRID_GATE["numbers"],
    }


def markdown(result: dict[str, Any]) -> str:
    """The scores of the text layer and the arm, the lowest table and the gate, as a
    report table (measures outside the gate are marked as reported)."""
    arm = result["arm"]
    thresholds = GRID_GATE if result["grid_mode"] else GATE
    lines = [
        f"| Measure | pypdf | {arm} | Gate |",
        "|---|---|---|---|",
    ]
    for measure in [*GATE, "grid"]:
        pypdf = result["pypdf"][measure]
        score = result[arm][measure]
        bar = f">= {thresholds[measure]}" if measure in thresholds else "reported"
        lines.append(f"| {measure} | {pypdf:.3f} | {score:.3f} | {bar} |")
    lines.append("")
    shares = ", ".join(f"{share:.3f}" for share in result["table_shares"])
    lines.append(
        f"Lowest table ({arm}): {result['lowest_table']:.3f} (>= {LOWEST_TABLE}); "
        f"each table: {shares}"
    )
    seconds = result["seconds"]
    lines.append(
        f"Seconds per page ({arm}): mean {seconds['mean']:.1f}, median "
        f"{seconds['median']:.1f}, first {seconds['first']:.1f} (loads the models)"
    )
    for item, passed in result["gate"].items():
        lines.append(f"- {item}: {'pass' if passed else 'FAIL'}")
    return "\n".join(lines)


def summarise(
    scores: dict[str, list[dict[str, Any]]], seconds: list[float], arm: str = "docling"
) -> dict[str, Any]:
    """Pooled measures for the text layer and the arm, the arm's lowest table and its
    gate: round 3's when the truth is written as whole grids, else rounds 1 and 2's."""
    measured = pooled(scores[arm])
    pypdf = pooled(scores["pypdf"])
    grid_mode = any(page["grid"][1] for page in scores[arm])
    kind = "grid_shares" if grid_mode else "table_shares"
    shares = [share for page in scores[arm] for share in page[kind]]
    lowest = min(shares, default=1.0)
    verdict = (
        grid_gate(measured, lowest) if grid_mode else gate(measured, pypdf, lowest)
    )
    return {
        "arm": arm,
        "grid_mode": grid_mode,
        "pypdf": pypdf,
        arm: measured,
        "lowest_table": lowest,
        "table_shares": shares,
        "seconds": {
            "mean": statistics.mean(seconds),
            "median": statistics.median(seconds),
            "first": seconds[0],
        },
        "gate": verdict,
    }


def transcription(truth: str, answer: str) -> dict[str, tuple[int, int]]:
    """Recall (truth words the transcription holds) and precision (transcribed
    words the truth holds), counted as multisets of words with a letter or digit,
    markup removed."""
    wanted = _alphanumeric(words(truth))
    have = _alphanumeric(words(re.sub(r"<[^>]+>", " ", answer)))
    common = sum((wanted & have).values())
    return {
        "recall": (common, sum(wanted.values())),
        "precision": (common, sum(have.values())),
    }


def _alphanumeric(counts: Counter[str]) -> Counter[str]:
    return Counter({w: n for w, n in counts.items() if any(c.isalnum() for c in w)})
