"""The passages one index version holds, built from what was read (plan step 8).

Each source keeps the form the Lime Green assistant measured: a PDF's or a
spreadsheet's pages as page-sized passages with their tables inline (X9's `page`
form), a web page's sections packed into passages with the first section's path as
context (X42 W3's `page` form). A scanned page's transcript becomes passages that
search can find but an answer never quotes (K12). Each picture becomes one passage
in the `picture` channel (X43 B); the text read in it is context only, because that
reading failed its gate (K16, pictures-v1).

A passage is a plain record (`Row`): where it comes from, what it says, its context
(searched, never quoted), which channel searches it, whether it may be quoted and
the picture it shows. The company's own sites and files form the `company` channel;
regulators and reference works (NUPRC, Wikipedia, the EIA) the `guidance` channel
(Lime Green ADR 0030). On request, the units page's whole list is also compiled into
one passage (K27, which failed its first gate, so it is off by default).
"""

import json
import re
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pypdfium2

from petroknowledge import (
    config,
    fetch,
    images,
    passages,
    sources,
    transcribe,
    units,
)

Row = dict[str, Any]
# Strapi, NNPC's CMS, adds an underscore and ten hex digits to each uploaded file.
UPLOAD_HASH = re.compile(r"_[0-9a-f]{10}$")
SHEET_PACKAGE = "openpyxl"  # a spreadsheet reading's fingerprint names it
PICTURE = "Picture"  # the heading of a passage made from a picture
DESCRIBED = 5  # words: an alt text this long describes its picture (Lime Green X42)
# The share of read words a page must lack for the text read in a picture to be kept
# (Lime Green X43 B2): a chart repeating its page's labels adds nothing.
NEW_WORDS = 0.2
WORD = re.compile(r"\w+")


def channel(url: str, own_hosts: list[str]) -> str:
    """The channel searching a source: the company's own sites, or guidance."""
    return "company" if sources.is_allowed(url, own_hosts) else "guidance"


def page_title(saved: dict[str, Any], listed: list[units.Unit]) -> str:
    """The title a page is indexed under: its own, or, for a unit's page whose title
    does not name the unit ("HISTORICAL ACTIVITIES" for NEPL's), the unit's name
    with its acronym (K29 F2), so search and the answer request know its subject."""
    for unit in listed:
        if unit["url"] == saved["url"] and not units.naming([unit])(saved["title"]):
            return units.described([unit])
    return str(saved["title"])


def pdf_date(path: Path) -> str:
    """A PDF's creation date from its own metadata (ISO date), or "" when it gives
    none or one that is not a real date. It dates the file, not its publication."""
    document = pypdfium2.PdfDocument(str(path))
    try:
        created = document.get_metadata_dict().get("CreationDate", "")
    finally:
        document.close()
    found = re.match(r"D:(\d{4})(\d{2})(\d{2})", created)
    if found is None:
        return ""
    try:
        day = date(int(found[1]), int(found[2]), int(found[3]))
    except ValueError:
        return ""
    return day.isoformat() if day.year >= 1990 else ""


def file_title(url: str) -> str:
    """A document's title from its file name: no extension, no CMS upload hash,
    underscores as spaces ("NNPCL FINANCIAL STATEMENTS FOR THE YEAR ENDED ...")."""
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1]).rsplit(".", 1)[0]
    return " ".join(UPLOAD_HASH.sub("", name).replace("_", " ").split())


def web_rows(saved: dict[str, Any], own_hosts: list[str]) -> list[Row]:
    """A saved web reading (`pages`) as passages: its sections packed, each passage
    with its first section's path as context. Pictures are left to their own
    channel, as Lime Green left alt texts out of web passages (X42 W5)."""
    kept = [e | {"page": 0} for e in saved["elements"] if e["kind"] != "figure"]
    if not kept:
        return []
    rows = []
    for heading, context, text, _ in passages.page_passages(kept, "table"):
        rows.append({
            "url": saved["url"], "title": saved["title"],
            "heading": heading or saved["title"], "text": text, "context": context,
            "page": None, "kind": saved["kind"],
            "channel": channel(saved["url"], own_hosts), "quotable": True,
            "published": saved["published"], "image": "",
        })  # fmt: skip
    return rows


def document_rows(
    elements: list[dict[str, Any]], url: str, kind: str, own_hosts: list[str]
) -> list[Row]:
    """A document's elements as page-sized passages, tables inline."""
    rows = []
    for heading, context, text, page in passages.pdf_passages(elements, "page"):
        rows.append({
            "url": url, "title": file_title(url), "heading": heading, "text": text,
            "context": context, "page": page, "kind": kind,
            "channel": channel(url, own_hosts), "quotable": kind != "scan",
            "published": "", "image": "",
        })  # fmt: skip
    return rows


def transcript_rows(
    transcript: dict[str, Any], url: str, own_hosts: list[str]
) -> list[Row]:
    """A scanned document's transcribed pages as passages search can find and an
    answer never quotes (K12): no transcription passed its gate."""
    found = [asdict(element) for element in transcribe.transcript_elements(transcript)]
    return document_rows(found, url, "scan", own_hosts) if found else []


def new_words(text: str, known: str) -> float:
    """The share of a text's words not already in `known` (case folded)."""
    words = [word.casefold() for word in WORD.findall(text)]
    have = {word.casefold() for word in WORD.findall(known)}
    return sum(word not in have for word in words) / len(words) if words else 0.0


def picture_rows(
    places: list[dict[str, Any]],
    read: dict[str, str],
    rows: list[Row],
    documents: list[dict[str, str]],
) -> list[Row]:
    """One passage per picture shown in a source of the build (`documents`), filed
    under the place whose alt text says most about it (the first, on a tie).

    Its text, the only part quoted, is an alt text of DESCRIBED words or more: the
    publisher's own words. Its context is lines, searched but never quoted: the
    section path, any shorter alt text, then the text read in it (`read`, by id)
    when at least NEW_WORDS of that is not already on its page (`rows`). So a
    picture has words of its own exactly when its text is not empty or its context
    has more than one line (`store.in_channel`)."""
    titles = {document["url"]: document["title"] for document in documents}
    shown: dict[tuple[str, int | None], str] = {}
    published: dict[str, str] = {}
    for row in rows:
        where = (row["url"], row["page"])
        shown[where] = shown.get(where, "") + " " + row["text"]
        published[row["url"]] = row["published"]
    chosen: dict[str, dict[str, Any]] = {}
    for place in places:
        best = chosen.get(place["id"])
        if place["source"] in titles and (
            best is None or len(place["alt"]) > len(best["alt"])
        ):
            chosen[place["id"]] = place
    found = []
    for identity, place in chosen.items():
        alt = " ".join(place["alt"].split())
        described = len(alt.split()) >= DESCRIBED
        context = [passages.PATH.join([PICTURE, *place["section"]])]
        if alt and not described:
            context.append(alt)
        text = "\n".join(images.ocr_lines(read.get(identity, "")))
        known = shown.get((place["source"], place["page"]), "")
        if text and new_words(text, known) >= NEW_WORDS:
            context.append(text)
        url = place["source"]
        found.append({
            "url": url, "title": titles[url], "heading": PICTURE,
            "text": alt if described else "", "context": "\n".join(context),
            "page": place["page"], "kind": "picture", "channel": "picture",
            "quotable": True, "published": published.get(url, ""), "image": identity,
        })  # fmt: skip
    return found


def stored_pictures(folder: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The places pictures are shown (`read-images`) and the text read in each, by
    id; none before the pictures are stored."""
    path = folder / "places.json"
    if not path.exists():
        return [], {}
    places = json.loads(path.read_text(encoding="utf-8"))
    read = {}
    for identity in dict.fromkeys(place["id"] for place in places):
        reading = folder / f"{identity}.json"
        if reading.exists():
            read[identity] = json.loads(reading.read_text(encoding="utf-8"))["ocr"]
    return places, read


def saved_json(folder: Path) -> list[dict[str, Any]]:
    """Every reading saved in a folder, by SHA-256 file name (reports left out)."""
    found = []
    for path in sorted(folder.glob("*.json")):
        if len(path.stem) == 64:
            found.append(json.loads(path.read_text(encoding="utf-8")))
    return found


def build(
    company_id: str, compiled_units: bool = False
) -> tuple[list[Row], list[dict[str, str]]]:
    """Every passage of a company's index version and every source it comes from
    (url, title, fetched_at, sha256): web pages and posts, documents (PDFs and
    spreadsheets), scanned pages' transcripts and the pictures shown in them;
    with `compiled_units`, the units page's list as one compiled passage too."""
    own = sources.load(company_id)["own_hosts"]
    records: dict[str, dict[str, Any]] = {}
    for entry in fetch.read_manifest():
        if entry.get("company") == company_id and "sha256" in entry:
            records[entry["sha256"]] = entry
    listed = listed_units(company_id)
    rows: list[Row] = []
    documents: list[dict[str, str]] = []
    # A file collected from several addresses is indexed under one of them; its
    # pictures are filed under the same one (the picture reader names the first).
    filed: dict[str, str] = {}
    for saved in saved_json(config.PAGE_READINGS):
        record = records.get(saved["sha256"])
        if record is None:
            continue
        titled = saved | {"title": page_title(saved, listed)}
        rows += web_rows(titled, own)
        documents.append(source(record, titled["title"]))
    if compiled_units:
        rows += compiled_rows(company_id, own)
    for reading in saved_json(config.READINGS):
        record = records.get(reading["sha256"])
        if record is None:
            continue
        sheet = SHEET_PACKAGE in reading["fingerprint"]["packages"]
        kind = "sheet" if sheet else "pdf"
        # A page the reader flags as having little or no text layer is indexed from
        # its transcript only: what text it holds can be a broken font's garbage
        # (133 pages of NNPC's 2025 annual report).
        scanned = set(transcribe.scanned_pages(reading))
        kept = [e for e in reading["elements"] if e["page"] not in scanned]
        rows += document_rows(kept, record["url"], kind, own)
        dated = "" if sheet else pdf_date(config.FILE_STORE / record["sha256"])
        documents.append(source(record, file_title(record["url"]), dated))
        for address in reading["urls"]:
            filed[address] = record["url"]
    for transcript in saved_json(config.TRANSCRIPTS):
        record = records.get(transcript["sha256"])
        if record is not None:
            rows += transcript_rows(transcript, record["url"], own)
    places, read = stored_pictures(config.IMAGES)
    for place in places:
        place["source"] = filed.get(place["source"], place["source"])
    rows += picture_rows(places, read, rows, documents)
    return rows, documents


def units_capture(
    company_id: str,
) -> tuple[dict[str, str], str, dict[str, Any]] | None:
    """A company's units page as registered, its latest stored capture and the
    reader's reading of it; None for a company without a units page."""
    page = sources.load(company_id).get("units_page")
    if page is None:
        return None
    captures = [r for r in fetch.read_manifest()
                if r.get("url") == page["url"] and "sha256" in r]  # fmt: skip
    sha256 = captures[-1]["sha256"]
    raw = (config.FILE_STORE / sha256).read_text(encoding="utf-8", errors="replace")
    reading_path = config.PAGE_READINGS / f"{sha256}.json"
    reading = json.loads(reading_path.read_text(encoding="utf-8"))
    return page, raw, reading


def listed_units(company_id: str) -> list[units.Unit]:
    """The units a company's units page lists (`units.listed`), each with the former
    names its own page gives it (`units.former_names`, K29 F2); none for a company
    without a units page."""
    capture = units_capture(company_id)
    if capture is None:
        return []
    page, raw, reading = capture
    found = units.listed(raw, page["url"], reading["elements"], page["section"])
    pages = {saved["url"]: saved for saved in saved_json(config.PAGE_READINGS)}
    for unit in found:
        own = pages.get(unit["url"], {"elements": []})
        text = "\n".join(element["text"] for element in own["elements"])
        unit["aliases"] = "; ".join(units.former_names(text, unit))
    return found


def listed_groups(company_id: str) -> list[units.Group]:
    """The units page's list by group (`units.grouped`); none for a company without a
    units page."""
    capture = units_capture(company_id)
    if capture is None:
        return []
    page, raw, reading = capture
    return units.grouped(raw, reading["elements"], page["section"])


def compiled_rows(company_id: str, own_hosts: list[str]) -> list[Row]:
    """The units page's whole list as one compiled passage, by group (K27); none for
    a company without a units page or a list on it."""
    groups = listed_groups(company_id)
    if not groups:
        return []
    capture = units_capture(company_id)
    assert capture is not None  # a list was read from it
    page, _, reading = capture
    return [{
        "url": reading["url"], "title": reading["title"], "heading": units.COMPILED,
        "text": units.compiled_text(page["section"], groups),
        "context": page["section"], "page": None, "kind": reading["kind"],
        "channel": channel(reading["url"], own_hosts), "quotable": True,
        "published": reading["published"], "image": "",
    }]  # fmt: skip


def source(record: dict[str, Any], title: str, file_date: str = "") -> dict[str, str]:
    """A collected file as the index records it, with its file's own date if it
    gives one (`pdf_date`, K29 F1)."""
    return {"url": record["url"], "title": title, "fetched_at": record["fetched_at"],
            "sha256": record["sha256"], "file_date": file_date}  # fmt: skip
