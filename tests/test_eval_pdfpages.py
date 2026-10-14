"""The gate for born-digital PDF pages: candidates drawn by seed, saved readings
scored against truth grids. Every document, page and text here is invented."""

import json
from pathlib import Path
from typing import Any

import pypdfium2
import pytest

from evaluation import __main__ as cli
from evaluation import pdfpages

A, B, C = "a" * 64, "b" * 64, "c" * 64
SCANNED = ["little or no text"]


def reading(sha256: str, pages: list[tuple[int, list[str], str]]) -> dict[str, Any]:
    """A reading whose pages have (visible words, flags, text)."""
    validation = [
        {"page": n, "words": words, "flags": flags}
        for n, (words, flags, _) in enumerate(pages, 1)
    ]
    elements = [
        {"page": n, "kind": "paragraph", "text": text, "cells": [], "grid": []}
        for n, (_, _, text) in enumerate(pages, 1)
    ]
    return {"sha256": sha256, "urls": [f"https://x/{sha256[:4]}.pdf?v=1"],
            "validation": {"pages": validation}, "elements": elements,
            "seconds": 2.5,
            "fingerprint": {"packages": {"pypdfium2": "5.13.0"}}}  # fmt: skip


READINGS = [
    reading(A, [(50, [], "alpha beta gamma"), (50, [], "delta epsilon zeta"),
                (50, [], "eta theta iota"), (5, [], "cover"),
                (50, SCANNED, "")]),
    reading(B, [(50, [], "alpha beta gamma"), (50, [], "kappa lambda mu")]),
    reading(C, [(50, [], "nu xi omicron")]),
]  # fmt: skip


def test_candidates_are_seeded_text_pages_two_per_document_without_duplicates() -> None:
    found = pdfpages.candidates(READINGS, 5, {(C, 1)})

    assert found == pdfpages.candidates(READINGS, 5, {(C, 1)})  # seeded
    pages = [(page["sha256"][0], page["page"]) for page in found]
    assert ("c", 1) not in pages  # used by an earlier set
    assert ("a", 4) not in pages and ("a", 5) not in pages  # a cover and a scan
    assert sum(1 for sha, _ in pages if sha == "a") == 2  # at most two per document
    texts = [pdfpages.page_text(r, p) for r in READINGS for p in [1]]
    assert texts[0] == texts[1]  # A and B's first pages are the same page:
    assert not {("a", 1), ("b", 1)} <= set(pages)  # only one of them is a candidate
    assert [page["number"] for page in found] == list(range(1, len(found) + 1))
    assert found[0]["url"].startswith("https://x/")


def text_pdf(path: Path, line: str) -> None:
    """A one-page PDF whose text layer holds `line` (pdfium repairs the xref)."""
    stream = f"BT /F1 12 Tf 10 50 Td ({line}) Tj ET".encode()
    parts = [
        b"%PDF-1.4\n",
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 100] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n",
        b"4 0 obj << /Length " + str(len(stream)).encode() + b" >> stream\n",
        stream + b"\nendstream endobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        b"trailer << /Root 1 0 R >>\n%%EOF\n",
    ]
    path.write_bytes(b"".join(parts))


def test_the_reference_is_the_page_s_text_layer(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    text_pdf(path, "Revenue 1,200")

    assert pdfpages.reference(path, 1).strip() == "Revenue 1,200"


TRUTH_PAGE = {
    "number": 1, "sha256": A, "page": 1, "pairs": [], "sentences": [],
    "tables": [{"header_rows": 1,
                "grid": [["", "2022 N'M"], ["Revenue", "1,200"], ["Costs", "(300)"]]}],
}  # fmt: skip
TABLE_READING = {
    "elements": [
        {"page": 1, "kind": "paragraph", "text": "Revenue table", "cells": [],
         "grid": []},
        {"page": 1, "kind": "table", "text": "whole table", "cells": [],
         "grid": [["", "2022 N'M"], ["Revenue", "1,200"], ["Costs", "(310)"]]},
        {"page": 1, "kind": "table_row", "text": "Revenue — 2022 N'M: 1,200",
         "cells": [["", "Revenue"], ["2022 N'M", "1,200"]], "grid": []},
        {"page": 2, "kind": "paragraph", "text": "another page", "cells": [],
         "grid": []},
    ]
}  # fmt: skip


def test_a_reading_is_scored_by_its_grid_its_text_and_its_numbers() -> None:
    scored = pdfpages.score_page(TRUTH_PAGE, TABLE_READING, "Revenue table")

    reading = scored["reading"]
    assert reading["grid"] == (1, 2)  # (310) is not the page's (300)
    assert reading["grid_shares"] == [0.5]
    assert reading["words"] == (2, 2)
    assert reading["numbers"] == (1, 2)  # body cells 1,200 and 300; only 1,200 read
    assert scored["pypdf"]["grid"] == (0, 2)  # the text layer holds no grid


def test_the_command_line_draws_candidates_once_and_scores_a_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, registry = tmp_path / "eval", tmp_path / "sets.json"
    readings, store = tmp_path / "elements", tmp_path / "files"
    readings.mkdir()
    store.mkdir()
    for found in READINGS:
        sha256 = found["sha256"]
        (readings / f"{sha256}.json").write_text(json.dumps(found))
        document = pypdfium2.PdfDocument.new()
        for _ in found["validation"]["pages"]:
            document.new_page(72, 72)
        document.save(store / sha256)
        document.close()
    sheet = {"fingerprint": {"packages": {"openpyxl": "3.1.5"}}}
    (readings / f"{'d' * 64}.json").write_text(json.dumps(sheet))  # not a PDF
    (readings / "report.json").write_text("{}")
    common = ["--root", str(root), "--registry", str(registry)]
    draw = [*common, "pdf-candidates", "pdfs", "--seed", "4", "--count", "2",
            "--readings", str(readings), "--store", str(store)]  # fmt: skip

    assert cli.main(draw) == 0
    drawn = json.loads((root / "pdfs" / "candidates.json").read_text(encoding="utf-8"))
    assert drawn["seed"] == 4 and len(drawn["pages"]) == 5
    assert len(list((root / "pdfs" / "images").glob("*.png"))) == 4
    assert "5 candidates, images of the first 2" in capsys.readouterr().out
    assert cli.main(draw) == 1
    assert "drawn once" in capsys.readouterr().err

    truth = {"pages": [TRUTH_PAGE]}
    (root / "pdfs" / "truth.json").write_text(json.dumps(truth))
    assert cli.main([*common, "register", "pdfs"]) == 0
    (readings / f"{A}.json").write_text(json.dumps({**READINGS[0], **TABLE_READING}))
    text_pdf(store / A, "Revenue table")
    out = tmp_path / "score.json"
    score = [*common, "pdf-score", "pdfs", "--out", str(out),
             "--readings", str(readings), "--store", str(store)]  # fmt: skip

    assert cli.main(score) == 1  # one value of two placed
    printed = capsys.readouterr().out
    assert "| grid | 0.000 | 0.500 | >= 0.95 |" in printed
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["arm"] == "reading" and report["lowest_table"] == 0.5

    again = [*common, "pdf-candidates", "more", "--seed", "4", "--count", "0",
             "--exclude-set", "pdfs", "--readings", str(readings),
             "--store", str(store)]  # fmt: skip
    assert cli.main(again) == 0
    later = json.loads((root / "more" / "candidates.json").read_text(encoding="utf-8"))
    assert (A, 1) not in {(p["sha256"], p["page"]) for p in later["pages"]}
