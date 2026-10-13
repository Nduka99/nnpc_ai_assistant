"""Collected PDFs and spreadsheets read into elements, adopted from the Lime Green
assistant (its ADR 0027). Spreadsheets are read by `sheets`, PDFs by `pdf`.

Each distinct file (by SHA-256) is read in its own child process, so a native fault
in the PDF libraries, or a document that never finishes, stops only that document;
it is tried once more, then reported as failed. Each reading is saved with a
fingerprint of everything that produced it (reader code, package versions, the
vision model), so a run skips documents already read the same way and resumes
after an interruption. Each reading carries a validation record per page: how much
of the words a reader sees (by pdfium) the parser's own text holds, the lines
recovered, the words not shown, and Docling's confidence grade; pages below the bar
are flagged in the run's report.
"""

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import httpx

from petroknowledge import config, fetch, llm, pdf, sheets

OUT = config.READINGS
ATTEMPTS = 2
TIMEOUT_SECONDS = 3600.0  # an annual report of several hundred pages takes a while
LOW_COVERAGE = 0.95  # a page whose own text holds less of its layer is flagged
MIN_WORDS = 10  # a page with fewer text-layer words has no text to check
PACKAGES = ("docling-slim", "docling-core", "docling-parse", "docling-ibm-models",
            "pypdfium2", "torch", "pylatexenc")  # fmt: skip
READER = (
    "elements.py",
    "pdf.py",
    "recovery.py",
    "rendering.py",
    "tables.py",
    "visibility.py",
)
# Spreadsheets have their own reader, so a change to it rereads only them.
SHEET_PACKAGES = ("openpyxl", "xlrd")
SHEET_READER = ("elements.py", "sheets.py")
READABLE = (
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
)

Run = Any  # subprocess.run, or a stand-in in tests


def document_files(records: list[fetch.Record]) -> dict[str, list[str]]:
    """Each stored PDF or spreadsheet by SHA-256, with every URL it came from
    (collected or added by hand)."""
    files: dict[str, list[str]] = {}
    for record in records:
        if record.get("kind") != "document" or "sha256" not in record:
            continue
        if str(record.get("content_type", "")).startswith(READABLE):
            files.setdefault(str(record["sha256"]), []).append(str(record["url"]))
    return files


def is_pdf(path: Path) -> bool:
    with path.open("rb") as file:
        return file.read(4) == b"%PDF"


def fingerprint(vlm: str, spreadsheet: bool = False) -> dict[str, Any]:
    """What a reading depends on: the reader's own code, the packages it uses and,
    when a PDF's tables are read again, the model the vision server has loaded."""
    names, packages = (
        (SHEET_READER, SHEET_PACKAGES) if spreadsheet else (READER, PACKAGES)
    )
    code = hashlib.sha256()
    for name in names:
        code.update((Path(pdf.__file__).parent / name).read_bytes())
    found: dict[str, Any] = {
        "reader_sha256": code.hexdigest(),
        "packages": {name: version(name) for name in packages},
        "vision_model": "",
    }
    if vlm and not spreadsheet:
        response = httpx.get(f"{vlm}/v1/models", headers=llm.auth(), timeout=30.0)
        response.raise_for_status()
        found["vision_model"] = response.json()["data"][0]["id"]
    return found


def validate(
    checks: dict[int, dict[str, int]], grades: dict[int, str]
) -> dict[str, Any]:
    """Per page: the words pdfium reads where the page shows them, how many the
    parser's own reading holds, words not shown (left out of the index), lines
    recovered, Docling's lowest confidence grade, and flags."""
    pages = []
    for number, check in sorted(checks.items()):
        flags = []
        if check["words"] < MIN_WORDS:
            flags.append("little or no text")
        elif check["kept"] / check["words"] < LOW_COVERAGE:
            flags.append("low coverage")
        if grades.get(number) == "poor":
            flags.append("poor confidence")
        grade = grades.get(number, "unspecified")
        pages.append({"page": number, **check, "grade": grade, "flags": flags})
    return {"pages": pages, "flagged": sum(bool(page["flags"]) for page in pages)}


def read_one(sha256: str, vlm: str) -> dict[str, Any]:
    """One document read into elements, with its validation (run in a child). A
    spreadsheet has no page layer to check, so its validation lists no pages."""
    stats: Counter[str] = Counter()
    grades: dict[int, str] = {}
    checks: dict[int, dict[str, int]] = {}
    started = time.perf_counter()
    path = fetch.store_path(sha256)
    if is_pdf(path):
        found = pdf.read_pdf(path, vlm=vlm, stats=stats, grades=grades, checks=checks)
    else:
        found = sheets.read_book(path)
        stats["tables"] = sum(1 for element in found if element.kind == "table")
    seconds = round(time.perf_counter() - started, 1)
    return {
        "sha256": sha256,
        "seconds": seconds,
        "tables": dict(stats),
        "validation": validate(checks, grades),
        "elements": [asdict(element) for element in found],
    }


def current(target: Path, found: dict[str, Any]) -> bool:
    """Whether a saved reading was made exactly as a new one would be."""
    if not target.exists():
        return False
    saved = json.loads(target.read_text(encoding="utf-8"))
    return bool(saved.get("fingerprint") == found)


def attempt(sha256: str, part: Path, vlm: str, run: Run) -> str:
    """Read one document in a child process; the error, or "" when it worked."""
    command = [
        sys.executable,
        "-m",
        "petroknowledge",
        "read-document",
        sha256,
        str(part),
    ]
    if vlm:
        command += ["--vlm", vlm]
    try:
        # The child's libraries may write bytes the locale's code page cannot
        # decode (cp1252 on Windows), so its output is read as UTF-8, replacing
        # anything unreadable rather than failing.
        done = run(
            command,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"timed out after {TIMEOUT_SECONDS:.0f} s"
    if done.returncode != 0 or not part.exists():
        tail = (done.stderr or "").strip().splitlines()[-1:] or [""]
        return f"exit {done.returncode}: {tail[0][:300]}"
    return ""


def run_all(
    files: dict[str, list[str]], vlm: str, out: Path, run: Run = subprocess.run
) -> dict[str, Any]:
    """Read every file not already read the same way; the run's report."""
    prints = {False: fingerprint(vlm), True: fingerprint(vlm, spreadsheet=True)}
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"fingerprint": prints[False], "read": [], "skipped": [],
                              "failed": {}}  # fmt: skip
    for sha256, urls in files.items():
        found = prints[not is_pdf(fetch.store_path(sha256))]
        target = out / f"{sha256}.json"
        if current(target, found):
            report["skipped"].append(sha256)
            continue
        part = out / f"{sha256}.part.json"
        errors = []
        for _ in range(ATTEMPTS):
            error = attempt(sha256, part, vlm, run)
            if not error:
                break
            errors.append(error)
        if len(errors) == ATTEMPTS:
            report["failed"][sha256] = {"urls": urls, "errors": errors}
            continue
        reading = json.loads(part.read_text(encoding="utf-8"))
        reading |= {"urls": urls, "fingerprint": found}
        text = json.dumps(reading, indent=1, ensure_ascii=False) + "\n"
        part.write_text(text, encoding="utf-8", newline="\n")
        part.replace(target)  # written whole, then moved into place
        report["read"].append(sha256)
    report["summary"] = summary(files, out)
    text = json.dumps(report, indent=1, ensure_ascii=False) + "\n"
    (out / "report.json").write_text(text, encoding="utf-8", newline="\n")
    return report


def summary(files: dict[str, list[str]], out: Path) -> dict[str, Any]:
    """Totals over every saved reading, and each flagged page with its document."""
    totals: Counter[str] = Counter()
    flagged = []
    for sha256, urls in files.items():
        target = out / f"{sha256}.json"
        if not target.exists():
            continue
        reading = json.loads(target.read_text(encoding="utf-8"))
        totals["documents"] += 1
        totals["seconds"] += reading["seconds"]
        totals.update(reading["tables"])
        for page in reading["validation"]["pages"]:
            totals["pages"] += 1
            totals["recovered lines"] += page["recovered"]
            totals["words not shown"] += page["hidden"]
            totals["hidden words removed"] += page["removed"]
            if page["flags"]:
                flagged.append({
                    "url": urls[0],
                    "page": page["page"],
                    "flags": page["flags"],
                    "kept": page["kept"],
                    "words": page["words"],
                    "grade": page["grade"],
                })  # fmt: skip
    return {"totals": dict(totals), "flagged": flagged}
