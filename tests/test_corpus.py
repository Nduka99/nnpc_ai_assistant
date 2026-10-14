"""An index version's passages built from the saved readings. Every page, document
and figure here is invented."""

import json
from pathlib import Path
from typing import Any

import pytest

from petroknowledge import config, corpus, fetch, sources, units

OWN = ["acme.test"]
# The smallest PDF pdfium reads, created on 20 December 2022 by its metadata.
DATED_PDF = (
    b"%PDF-1.4\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
    b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
    b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 10 10] >> endobj\n"
    b"4 0 obj << /CreationDate (D:20221220102933+01'00') >> endobj\n"
    b"trailer << /Root 1 0 R /Info 4 0 R >>\n%%EOF\n"
)


def element(page: int, kind: str, text: str, section: list[str]) -> dict[str, Any]:
    found: dict[str, Any] = {"page": page, "kind": kind, "text": text}
    return found | {"section": section, "bbox": None, "table": None, "row": None,
                    "cells": [], "grid": []}  # fmt: skip


def test_a_document_is_titled_by_its_file_name_and_a_source_by_its_channel() -> None:
    url = "https://cms.acme.test/uploads/ACME_FINANCIAL_STATEMENTS_2023_ff9ff543f9.pdf"
    assert (
        corpus.file_title(url + "?updated_at=2023") == "ACME FINANCIAL STATEMENTS 2023"
    )
    assert corpus.file_title("https://acme.test/docs/Annual%20Report.pdf") == (
        "Annual Report"
    )
    assert corpus.channel("https://retail.acme.test/about", OWN) == "company"
    assert corpus.channel("https://en.wikipedia.org/wiki/Acme", OWN) == "guidance"


SAVED = {
    "url": "https://acme.test/insights/new-pipeline",
    "title": "New Pipeline Opens",
    "kind": "post",
    "published": "2024-03-12",
    "sha256": "a" * 64,
    "elements": [
        {"kind": "paragraph", "text": "MARCH 12, 2024", "section": []},
        {"kind": "heading", "text": "Route", "level": 2, "section": []},
        {"kind": "paragraph", "text": "The line runs 614 km.", "section": ["Route"]},
        {"kind": "figure", "text": "Welders at work on the line", "section": ["Route"]},
        {"kind": "list_item", "text": "Ajaokuta", "section": ["Route"]},
    ],
}


def test_a_web_page_s_sections_are_packed_and_its_pictures_left_out() -> None:
    rows = corpus.web_rows(SAVED, OWN)

    assert len(rows) == 1
    row = rows[0]
    assert row["text"] == "MARCH 12, 2024\nRoute\nThe line runs 614 km.\nAjaokuta"
    assert (row["heading"], row["context"]) == ("New Pipeline Opens", "")
    assert (row["kind"], row["channel"], row["quotable"]) == ("post", "company", True)
    assert row["published"] == "2024-03-12" and row["page"] is None
    only_pictures = SAVED | {"elements": [SAVED["elements"][3]]}
    assert corpus.web_rows(only_pictures, OWN) == []


def test_a_document_becomes_page_passages_and_a_scan_is_never_quoted() -> None:
    elements = [
        element(1, "heading", "Results", []),
        element(1, "paragraph", "Revenue rose 5%.", ["Results"]),
        element(2, "paragraph", "Costs fell.", ["Results"]),
    ]
    url = "https://acme.test/report.pdf"

    rows = corpus.document_rows(elements, url, "pdf", OWN)

    assert [(r["page"], r["text"]) for r in rows] == [
        (1, "Results\nRevenue rose 5%."),
        (2, "Costs fell."),
    ]
    assert rows[1]["context"] == "Results" and rows[0]["title"] == "report"
    assert all(r["quotable"] for r in rows)
    scanned = corpus.document_rows(elements, url, "scan", OWN)
    assert not any(r["quotable"] for r in scanned)


def test_a_transcript_is_found_by_search_but_never_quoted() -> None:
    transcript = {"pages": {"3": {"answer": "## Remita\n\nPay online through Remita."}}}

    rows = corpus.transcript_rows(transcript, "https://acme.test/guide.pdf", OWN)

    assert [(r["page"], r["kind"], r["quotable"]) for r in rows] == [(3, "scan", False)]
    assert "Pay online through Remita." in rows[0]["text"]
    assert corpus.transcript_rows({"pages": {}}, "https://acme.test/x.pdf", OWN) == []


def test_an_index_is_built_from_every_saved_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages, readings, transcripts = (tmp_path / n for n in ("p", "e", "t"))
    for folder in (pages, readings, transcripts):
        folder.mkdir()
    (pages / f"{'a' * 64}.json").write_text(json.dumps(SAVED))
    (pages / f"{'f' * 64}.json").write_text(json.dumps(SAVED | {"sha256": "f" * 64}))
    (pages / "report-acme.json").write_text("{}")
    # The report was collected from two addresses; the manifest names the second.
    # Its page 2 has no usable text layer (a broken font's glyphs): its transcript
    # stands for it.
    flagged = {"pages": [{"page": 1, "flags": []},
                         {"page": 2, "flags": ["little or no text"]}]}  # fmt: skip
    pdf = {"sha256": "b" * 64, "fingerprint": {"packages": {"pypdfium2": "5"}},
           "urls": ["https://cms.acme.test/r.pdf", "https://acme.test/r.pdf"],
           "validation": flagged,
           "elements": [element(1, "paragraph", "Revenue rose 5%.", []),
                        element(2, "paragraph", "¾ǚƷژGȵȏɓȲ", [])]}  # fmt: skip
    sheet = {"sha256": "c" * 64, "fingerprint": {"packages": {"openpyxl": "3"}},
             "urls": ["https://acme.test/gas.xlsx"], "validation": {"pages": []},
             "elements": [element(1, "paragraph", "Gas 2,285", [])]}  # fmt: skip
    (readings / f"{'b' * 64}.json").write_text(json.dumps(pdf))
    (readings / f"{'c' * 64}.json").write_text(json.dumps(sheet))
    (readings / f"{'e' * 64}.json").write_text(json.dumps(pdf | {"sha256": "e" * 64}))
    scan = {"sha256": "b" * 64, "pages": {"2": {"answer": "Signed in Abuja."}}}
    (transcripts / f"{'b' * 64}.json").write_text(json.dumps(scan))
    stray = {"sha256": "d" * 64, "pages": {"1": {"answer": "Not collected."}}}
    (transcripts / f"{'d' * 64}.json").write_text(json.dumps(stray))
    manifest = [
        {"company": "acme", "sha256": "a" * 64, "url": SAVED["url"],
         "fetched_at": "2026-10-10T10:00:00+00:00"},
        {"company": "acme", "sha256": "b" * 64, "url": "https://acme.test/r.pdf",
         "fetched_at": "2026-10-10T11:00:00+00:00"},
        {"company": "acme", "sha256": "c" * 64, "url": "https://acme.test/gas.xlsx",
         "fetched_at": "2026-10-10T12:00:00+00:00"},
        {"company": "other", "sha256": "f" * 64, "url": "https://other.test/x",
         "fetched_at": "2026-10-10T13:00:00+00:00"},
        {"company": "acme", "url": "https://acme.test/blocked"},  # never stored
    ]  # fmt: skip
    files = tmp_path / "files"
    files.mkdir()
    (files / ("b" * 64)).write_bytes(DATED_PDF)
    monkeypatch.setattr(config, "FILE_STORE", files)
    monkeypatch.setattr(config, "PAGE_READINGS", pages)
    monkeypatch.setattr(config, "READINGS", readings)
    monkeypatch.setattr(config, "TRANSCRIPTS", transcripts)
    monkeypatch.setattr(fetch, "read_manifest", lambda: manifest)
    monkeypatch.setattr(sources, "load", lambda company_id: {"own_hosts": OWN})
    pictures = tmp_path / "images"
    pictures.mkdir()
    monkeypatch.setattr(config, "IMAGES", pictures)
    assert corpus.stored_pictures(pictures) == ([], {})  # none stored yet
    # The picture reader names the report's first address.
    place = {"id": "p1", "source": "https://cms.acme.test/r.pdf", "page": 1,
             "alt": "", "section": []}  # fmt: skip
    (pictures / "places.json").write_text(json.dumps([place, place | {"id": "p2"}]))
    (pictures / "p1.json").write_text(json.dumps({"ocr": "Crude oil exports"}))

    rows, documents = corpus.build("acme", compiled_units=True)  # no units page

    assert [(r["kind"], r["quotable"]) for r in rows] == [
        ("post", True), ("pdf", True), ("sheet", True), ("scan", False),
        ("picture", True), ("picture", True),
    ]  # fmt: skip
    assert [r["image"] for r in rows] == ["", "", "", "", "p1", "p2"]
    assert rows[4]["url"] == "https://acme.test/r.pdf"  # filed with its document
    assert rows[4]["context"] == "Picture\nCrude oil exports"
    assert rows[5]["context"] == "Picture"  # p2 has no reading
    assert "ǚ" not in "".join(r["text"] for r in rows)  # the flagged page's own text
    assert rows[3]["text"].endswith("Signed in Abuja.")  # its transcript instead
    assert [d["title"] for d in documents] == ["New Pipeline Opens", "r", "gas"]
    assert documents[1]["fetched_at"] == "2026-10-10T11:00:00+00:00"
    # Only the PDF is dated, by its file: a page and a spreadsheet are not.
    assert [d["file_date"] for d in documents] == ["", "2022-12-20", ""]


def test_a_pdf_is_dated_only_by_a_real_creation_date(tmp_path: Path) -> None:
    for name, found in [("no-date", b"/Producer"), ("unreal", b"/CreationDate")]:
        data = DATED_PDF.replace(b"/CreationDate", found)
        (tmp_path / name).write_bytes(data.replace(b"D:20221220", b"D:00001301"))
    (tmp_path / "early").write_bytes(DATED_PDF.replace(b"D:2022", b"D:1985"))

    assert corpus.pdf_date(tmp_path / "no-date") == ""
    assert corpus.pdf_date(tmp_path / "unreal") == ""  # month 13 of year 0
    assert corpus.pdf_date(tmp_path / "early") == ""  # before any collected file


def test_a_unit_s_page_without_its_name_in_the_title_takes_the_unit_s_name() -> None:
    listed = [{"name": "Acme E&P Limited", "acronym": "AEP",
               "url": "https://acme.test/aep"}]  # fmt: skip
    history = {"url": "https://acme.test/aep", "title": "HISTORICAL ACTIVITIES"}
    named = {"url": "https://acme.test/aep", "title": "Acme E&P Limited (AEP)"}

    assert corpus.page_title(history, listed) == "Acme E&P Limited (AEP)"
    assert corpus.page_title(named, listed) == "Acme E&P Limited (AEP)"
    assert corpus.page_title(SAVED, listed) == "New Pipeline Opens"  # not a unit's


PDF_URL = "https://acme.test/r.pdf"
POST_URL = str(SAVED["url"])
DOCUMENTS: list[dict[str, str]] = [
    {"url": PDF_URL, "title": "r", "fetched_at": "", "sha256": "b" * 64},
    {"url": POST_URL, "title": "New Pipeline Opens", "fetched_at": "",
     "sha256": "a" * 64},
]  # fmt: skip
ROWS: list[dict[str, Any]] = [
    {"url": PDF_URL, "page": 2, "text": "Revenue rose 5%.", "published": ""},
    {"url": POST_URL, "page": None, "text": "The line runs 614 km.",
     "published": "2024-03-12"},
]  # fmt: skip


def test_a_picture_is_one_passage_and_its_read_text_only_context() -> None:
    places: list[dict[str, Any]] = [
        {"id": "p1", "source": PDF_URL, "page": 2, "alt": "", "section": ["Results"]},
        # The same picture again: its longer alt text describes it, so it is filed here.
        {"id": "p1", "source": POST_URL, "page": None,
         "alt": "Welders  at work on the line", "section": []},
        {"id": "p2", "source": PDF_URL, "page": 2, "alt": "Chart",
         "section": ["Results"]},
        {"id": "p3", "source": "https://elsewhere.test/x", "page": None,
         "alt": "A picture from outside the build", "section": []},
    ]  # fmt: skip
    read = {"p1": "<td>Welding crew</td>\n```", "p2": "Revenue rose 5%"}

    found = corpus.picture_rows(places, read, ROWS, DOCUMENTS)

    assert found == [
        {"url": POST_URL, "title": "New Pipeline Opens", "heading": "Picture",
         "text": "Welders at work on the line", "context": "Picture\nWelding crew",
         "page": None, "kind": "picture", "channel": "picture", "quotable": True,
         "published": "2024-03-12", "image": "p1"},
        # Its read words are all on its page already, so they are left out.
        {"url": PDF_URL, "title": "r", "heading": "Picture", "text": "",
         "context": "Picture › Results\nChart", "page": 2, "kind": "picture",
         "channel": "picture", "quotable": True, "published": "", "image": "p2"},
    ]  # fmt: skip


def test_new_words_are_the_share_not_already_known() -> None:
    assert corpus.new_words("Gas GAS oil", "gas") == 1 / 3
    assert corpus.new_words("", "gas") == 0.0


def test_a_company_s_units_are_read_from_its_stored_units_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, pages = tmp_path / "files", tmp_path / "pages"
    files.mkdir()
    pages.mkdir()
    url = "https://acme.test/businesses"
    html = '<div class="body"><a href="/acme-gas">Acme Gas Limited (AGL)</a></div>'
    (files / ("a" * 64)).write_text(html, encoding="utf-8")
    line = {"kind": "paragraph", "text": "Acme Gas Limited (AGL)",
            "section": ["Our Businesses"]}  # fmt: skip
    reading = {"url": url, "elements": [line]}
    (pages / f"{'a' * 64}.json").write_text(json.dumps(reading))
    # The unit's own page says what it was called before.
    told = {"kind": "paragraph", "section": [], "text": "Acme Gas Limited (AGL), "
            "formerly the Acme Gas Company (AGC), sells gas."}  # fmt: skip
    own = {"url": "https://acme.test/acme-gas", "elements": [told]}
    (pages / f"{'g' * 64}.json").write_text(json.dumps(own))
    monkeypatch.setattr(config, "FILE_STORE", files)
    monkeypatch.setattr(config, "PAGE_READINGS", pages)
    page = {"url": url, "section": "Our Businesses"}
    monkeypatch.setattr(sources, "load", lambda company_id: {"units_page": page})
    captures = [{"url": url, "sha256": "0" * 64}, {"url": url, "sha256": "a" * 64},
                {"url": url}]  # fmt: skip
    monkeypatch.setattr(fetch, "read_manifest", lambda: captures)

    assert corpus.listed_units("acme") == [  # from the latest capture
        {
            "name": "Acme Gas Limited",
            "acronym": "AGL",
            "url": "https://acme.test/acme-gas",
            "aliases": "Acme Gas Company; AGC",
        }
    ]
    monkeypatch.setattr(sources, "load", lambda company_id: {})
    assert corpus.listed_units("acme") == []  # no units page, no units
    assert corpus.compiled_rows("acme", OWN) == []


def test_a_units_page_s_list_is_one_compiled_passage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files, pages = tmp_path / "files", tmp_path / "pages"
    files.mkdir()
    pages.mkdir()
    url = "https://acme.test/businesses"
    html = ('<div><div>Gas</div><div><a href="/acme-gas">Acme Gas Limited (AGL)</a>'
            "</div></div>")  # fmt: skip
    lines = ["Gas", "Acme Gas Limited (AGL)"]
    elements = [{"kind": "paragraph", "text": line, "section": ["Our Businesses"]}
                for line in lines]  # fmt: skip
    reading = {"url": url, "title": "Businesses", "kind": "page", "published": "",
               "elements": elements}  # fmt: skip
    (files / ("a" * 64)).write_text(html, encoding="utf-8")
    (pages / f"{'a' * 64}.json").write_text(json.dumps(reading))
    monkeypatch.setattr(config, "FILE_STORE", files)
    monkeypatch.setattr(config, "PAGE_READINGS", pages)
    page = {"url": url, "section": "Our Businesses"}
    monkeypatch.setattr(sources, "load", lambda company_id: {"units_page": page})
    monkeypatch.setattr(
        fetch, "read_manifest", lambda: [{"url": url, "sha256": "a" * 64}]
    )

    (row,) = corpus.compiled_rows("acme", OWN)

    assert row["text"].splitlines()[2] == "Gas, 1 unit: Acme Gas Limited (AGL)."
    assert (row["url"], row["title"], row["heading"], row["context"]) == (
        url, "Businesses", units.COMPILED, "Our Businesses")  # fmt: skip
    assert (row["kind"], row["channel"], row["quotable"]) == ("page", "company", True)
    (files / ("a" * 64)).write_text("<p>Gas</p>", encoding="utf-8")
    assert corpus.compiled_rows("acme", OWN) == []  # no list on the page
