"""Reading every stored PDF, each in its own process. Every document, reading and
process here is invented; no PDF library runs."""

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest

from petroknowledge import cli, documents, fetch, llm, pdf
from petroknowledge.elements import Element

WORDS = "Revenue for the year rose to four trillion naira on higher crude output"


def test_stored_pdfs_and_spreadsheets_are_grouped_by_their_content() -> None:
    records = [
        {"url": "https://x.test/a.pdf", "kind": "document", "sha256": "a",
         "content_type": "application/pdf"},
        {"url": "https://x.test/copy-of-a.pdf", "kind": "document", "sha256": "a",
         "content_type": "application/pdf; charset=binary"},
        {"url": "https://x.test/b.jpg", "kind": "picture", "sha256": "b",
         "content_type": "image/jpeg"},
        {"url": "https://x.test/c.pdf", "kind": "document", "sha256": "c",
         "content_type": "application/pdf", "added_by": "hand"},
        {"url": "https://x.test/gone.pdf", "kind": "document", "error": "HTTP 404"},
        {"url": "https://x.test/sheet.xlsx", "kind": "document", "sha256": "d",
         "content_type": "application/vnd.ms-excel"},
        {"url": "https://x.test/about", "kind": "page", "sha256": "e",
         "content_type": "text/html"},
        {"url": "https://x.test/data.xlsx", "kind": "document", "sha256": "f",
         "content_type": "application/vnd.openxmlformats-officedocument."
                         "spreadsheetml.sheet"},
        {"url": "https://x.test/letter.docx", "kind": "document", "sha256": "g",
         "content_type": "application/vnd.openxmlformats-officedocument."
                         "wordprocessingml.document"},
    ]  # fmt: skip

    assert documents.document_files(records) == {
        "a": ["https://x.test/a.pdf", "https://x.test/copy-of-a.pdf"],
        "c": ["https://x.test/c.pdf"],
        "d": ["https://x.test/sheet.xlsx"],
        "f": ["https://x.test/data.xlsx"],
    }


def test_a_fingerprint_names_the_code_packages_and_vision_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get(url: str, **kwargs: Any) -> httpx.Response:
        assert url == "http://vlm/v1/models"
        body = {"data": [{"id": "GLM-OCR-f16.gguf"}]}
        return httpx.Response(200, json=body, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(llm, "auth", lambda: {})

    plain = documents.fingerprint("")
    with_model = documents.fingerprint("http://vlm")

    assert len(plain["reader_sha256"]) == 64
    assert set(plain["packages"]) == set(documents.PACKAGES)
    assert plain["vision_model"] == ""
    assert with_model["vision_model"] == "GLM-OCR-f16.gguf"
    assert with_model["reader_sha256"] == plain["reader_sha256"]


def test_each_page_is_flagged_from_its_check_and_grade() -> None:
    checks = {
        1: {"words": 13, "kept": 13, "hidden": 0, "recovered": 0},
        2: {"words": 13, "kept": 3, "hidden": 0, "recovered": 1},
        3: {"words": 2, "kept": 2, "hidden": 4, "recovered": 0},
        4: {"words": 20, "kept": 20, "hidden": 0, "recovered": 0},
    }
    grades = {1: "excellent", 2: "fair", 4: "poor"}

    checked = documents.validate(checks, grades)

    pages = {page["page"]: page for page in checked["pages"]}
    assert pages[1] == {"page": 1, "words": 13, "kept": 13, "hidden": 0,
                        "recovered": 0, "grade": "excellent", "flags": []}  # fmt: skip
    assert pages[2]["flags"] == ["low coverage"]
    assert pages[3]["flags"] == ["little or no text"]
    assert pages[3]["grade"] == "unspecified"
    assert pages[4]["flags"] == ["poor confidence"]
    assert checked["flagged"] == 3


def test_one_document_is_read_with_its_tables_and_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def read_pdf(
        path: Path,
        vlm: str,
        stats: Counter[str],
        grades: dict[int, str],
        checks: dict[int, dict[str, int]],
    ) -> Any:
        assert (path.name, vlm) == ("abc", "http://vlm")
        stats["tables"] += 2
        grades[1] = "good"
        checks[1] = {"words": 13, "kept": 13, "hidden": 0, "recovered": 0}
        return [Element(1, "paragraph", WORDS)]

    monkeypatch.setattr(pdf, "read_pdf", read_pdf)
    monkeypatch.setattr(documents, "is_pdf", lambda path: True)

    reading = documents.read_one("abc", "http://vlm")

    assert reading["sha256"] == "abc"
    assert reading["tables"] == {"tables": 2}
    assert reading["validation"]["pages"][0]["grade"] == "good"
    assert reading["elements"][0]["text"] == WORDS
    assert isinstance(reading["seconds"], float)


class Child:
    """Stands in for subprocess.run: each document's outcomes, in order."""

    def __init__(self, outcomes: dict[str, list[str]]) -> None:
        self.outcomes = outcomes
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> Any:
        # The child's output is read as UTF-8, never in the locale's code page.
        assert (kwargs["encoding"], kwargs["errors"]) == ("utf-8", "replace")
        self.commands.append(command)
        sha256, part = command[4], Path(command[5])
        outcome = self.outcomes[sha256].pop(0)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if outcome == "ok":
            reading = {"sha256": sha256, "seconds": 2.5, "tables": {"tables": 1},
                       "validation": {"pages": [
                           {"page": 1, "words": 20, "kept": 20, "hidden": 0,
                            "recovered": 0, "removed": 0, "grade": "good",
                            "flags": []},
                           {"page": 2, "words": 20, "kept": 10, "hidden": 5,
                            "recovered": 3, "removed": 2, "grade": "fair",
                            "flags": ["low coverage"]}],
                           "flagged": 1},
                       "elements": []}  # fmt: skip
            part.write_text(json.dumps(reading), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, "", "")
        if outcome == "silent":
            return subprocess.CompletedProcess(command, 0, "", "")
        return subprocess.CompletedProcess(command, 127, "", "Windows fatal exception")


def test_a_run_reads_each_document_once_retries_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        documents, "fingerprint", lambda vlm, spreadsheet=False: {"vision_model": vlm}
    )
    monkeypatch.setattr(documents, "is_pdf", lambda path: True)
    done = {"sha256": "old", "seconds": 1.0, "tables": {}, "urls": ["u/old.pdf"],
            "validation": {"pages": [], "flagged": 0},
            "fingerprint": {"vision_model": "http://vlm"}}  # fmt: skip
    (tmp_path / "old.json").write_text(json.dumps(done), encoding="utf-8")
    files = {"old": ["u/old.pdf"], "new": ["u/new.pdf"], "retried": ["u/r.pdf"],
             "broken": ["u/b.pdf"], "stale": ["u/s.pdf"]}  # fmt: skip
    stale = done | {"fingerprint": {"vision_model": ""}}
    (tmp_path / "stale.json").write_text(json.dumps(stale), encoding="utf-8")
    child = Child({"new": ["ok"], "retried": ["crash", "ok"],
                   "broken": ["timeout", "silent"], "stale": ["ok"]})  # fmt: skip

    report = documents.run_all(files, "http://vlm", tmp_path, run=child)

    assert report["skipped"] == ["old"]
    assert report["read"] == ["new", "retried", "stale"]
    assert report["failed"]["broken"]["errors"] == [
        "timed out after 3600 s",
        "exit 0: ",
    ]
    assert child.commands[0][-2:] == ["--vlm", "http://vlm"]
    saved = json.loads((tmp_path / "new.json").read_text(encoding="utf-8"))
    assert saved["urls"] == ["u/new.pdf"]
    assert saved["fingerprint"] == {"vision_model": "http://vlm"}
    assert not (tmp_path / "new.part.json").exists()
    assert "Windows fatal exception" in documents.attempt(
        "x", tmp_path / "x.json", "", Child({"x": ["crash"]})
    )
    totals = report["summary"]["totals"]
    assert totals == {"documents": 4, "seconds": 8.5, "tables": 3, "pages": 6,
                      "recovered lines": 9, "words not shown": 15,
                      "hidden words removed": 6}  # fmt: skip
    assert [page["url"] for page in report["summary"]["flagged"]] == [
        "u/new.pdf",
        "u/r.pdf",
        "u/s.pdf",
    ]
    assert json.loads((tmp_path / "report.json").read_text())["read"] == report["read"]
    plain = Child({"x": ["ok"]})
    documents.attempt("x", tmp_path / "x.json", "", plain)
    assert "--vlm" not in plain.commands[0]


def test_the_command_line_reads_every_pdf_and_prints_the_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fetch, "read_manifest", lambda: [])
    report = {
        "read": ["a"], "skipped": [], "failed": {"b": {"urls": ["u/b.pdf"],
                                                      "errors": ["exit 127: fault"]}},
        "summary": {"totals": {"documents": 1, "seconds": 2.5},
                    "flagged": [{"url": "u/a.pdf", "page": 2, "flags": ["low coverage"],
                                 "kept": 10, "words": 20, "grade": "fair"}]},
    }  # fmt: skip
    monkeypatch.setattr(documents, "run_all", lambda files, vlm, out: report)
    monkeypatch.setattr(llm, "healthy", lambda url: url == "http://up")

    assert cli.main(["read-documents", "--vlm", "http://down"]) == 1
    assert "no vision model ready" in capsys.readouterr().err
    assert cli.main(["read-documents", "--vlm", "http://up"]) == 1
    printed = capsys.readouterr().out
    assert "0 documents: 1 read, 0 already read, 1 failed" in printed
    assert "failed u/b.pdf (b): exit 127: fault" in printed
    assert "seconds: 2.5" in printed and "documents: 1" in printed
    assert "flagged a.pdf p2: low coverage (10/20 words, fair)" in printed
    report["failed"] = {}
    assert cli.main(["read-documents"]) == 0

    monkeypatch.setattr(documents, "read_one", lambda sha, vlm: {"sha256": sha})
    out = tmp_path / "one.json"
    assert cli.main(["read-document", "abc", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8")) == {"sha256": "abc"}


def test_a_spreadsheet_is_read_by_the_sheet_reader_with_no_pages_to_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openpyxl

    book = openpyxl.Workbook()
    sheet = book.active
    assert sheet is not None
    sheet.append(["Company", "Barrels"])
    sheet.append(["Acme", 100])
    path = tmp_path / "data.xlsx"
    book.save(path)
    monkeypatch.setattr(fetch, "store_path", lambda sha256: path)

    reading = documents.read_one("data", "")

    assert reading["tables"] == {"tables": 1}
    assert reading["validation"] == {"pages": [], "flagged": 0}
    assert reading["elements"][-1]["text"].endswith("Acme — Barrels: 100")
    (tmp_path / "x.pdf").write_bytes(b"%PDF-1.7")
    assert documents.is_pdf(tmp_path / "x.pdf") and not documents.is_pdf(path)


def test_a_spreadsheet_fingerprint_names_its_own_reader_and_no_vision_model() -> None:
    sheet_print = documents.fingerprint("http://vlm", spreadsheet=True)

    assert set(sheet_print["packages"]) == set(documents.SHEET_PACKAGES)
    assert sheet_print["vision_model"] == ""  # no request to the vision server
    assert sheet_print["reader_sha256"] != documents.fingerprint("")["reader_sha256"]
