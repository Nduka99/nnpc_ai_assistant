"""Born-digital PDF pages scored against truth judged from their images (Lime
Green's X8 round 3 on NNPC's documents).

Candidates are every page with a text layer (at least MIN_WORDS visible words and
not flagged as a scan), shuffled by one seeded generator, at most PER_DOCUMENT pages
of a document and no page resembling an earlier one. The truth writer takes the
first candidates whose image shows a table, in order, and writes each table as a
whole grid with its header rows. The saved readings (data/elements) are then scored
with round 3's measures: table values in their row and column, the page's text kept
against its text layer, and table numbers kept. The harness reads saved files only;
it never imports the engine it measures.
"""

import random
from pathlib import Path
from typing import Any

from evaluation import parsing

MIN_WORDS = 10  # fewer visible words is a cover, a drawing or a scan (round 1's rule)
PER_DOCUMENT = 2  # so one long report cannot fill the set
NEAR_DUPLICATE = 0.8  # word-set resemblance at which two pages are the same page
SCANNED = "little or no text"  # the flag a reading gives a page with no text layer

Reading = dict[str, Any]


def page_text(reading: Reading, page: int) -> str:
    """The reading's text for one page, every element joined."""
    return " ".join(e["text"] for e in reading["elements"] if e["page"] == page)


def resemblance(first: str, second: str) -> float:
    """The share of words two texts have in common (Jaccard, over word sets)."""
    a = set(first.casefold().split())
    b = set(second.casefold().split())
    return len(a & b) / len(a | b) if a | b else 1.0


def text_pages(reading: Reading) -> list[int]:
    """The pages with a text layer: enough visible words, not flagged as a scan."""
    pages = reading["validation"]["pages"]
    return [
        page["page"]
        for page in pages
        if page["words"] >= MIN_WORDS and SCANNED not in page["flags"]
    ]


def candidates(
    readings: list[Reading], seed: int, used: set[tuple[str, int]]
) -> list[dict[str, Any]]:
    """Every page with a text layer in a seeded order, leaving out pages `used`
    before, pages beyond PER_DOCUMENT of one document and pages resembling an
    earlier one; numbered in that order."""
    ordered = sorted(readings, key=lambda reading: str(reading["sha256"]))
    pages = [(reading, n) for reading in ordered for n in text_pages(reading)]
    random.Random(seed).shuffle(pages)
    earlier: list[str] = []
    taken: dict[str, int] = {}
    chosen: list[dict[str, Any]] = []
    for reading, number in pages:
        sha256 = str(reading["sha256"])
        if (sha256, number) in used or taken.get(sha256, 0) == PER_DOCUMENT:
            continue
        text = page_text(reading, number)
        if any(resemblance(text, other) >= NEAR_DUPLICATE for other in earlier):
            continue
        earlier.append(text)
        taken[sha256] = taken.get(sha256, 0) + 1
        chosen.append({
            "sha256": sha256,
            "page": number,
            "url": reading["urls"][0],
            "number": len(chosen) + 1,
        })  # fmt: skip
    return chosen


def reference(path: Path, page: int) -> str:
    """The page's text layer as pdfium reads it, the reference for words kept."""
    import pypdfium2

    document = pypdfium2.PdfDocument(str(path))
    try:
        text = document[page - 1].get_textpage().get_text_bounded()
    finally:
        document.close()
    return str(text)


def score_page(
    page: dict[str, Any], reading: Reading, layer: str
) -> dict[str, dict[str, Any]]:
    """Round 3's measures for the text layer and the reading on one truth page. A
    whole table repeats its rows' text, so it is measured by its grid only."""
    found = [e for e in reading["elements"] if e["page"] == page["page"]]
    units = [e["text"] for e in found if e["text"] and e["kind"] != "table"]
    rows = [
        tuple((str(header), str(value)) for header, value in e["cells"])
        for e in found
        if e["kind"] == "table_row"
    ]
    grids = [e["grid"] for e in found if e["kind"] == "table"]
    lines = layer.splitlines()
    return {
        "pypdf": parsing.score_page(page, lines, [], layer),
        "reading": parsing.score_page(page, units, rows, layer, grids),
    }
