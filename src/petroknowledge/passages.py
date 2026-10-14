"""Passages built from PDF readings (X9 report): the units searched, cited and quoted.

A reading (petroknowledge.documents) holds a PDF's elements in reading order. On each
page, consecutive elements under the same section form a section (a heading starts one
once the current one holds more than headings, so stacked headings stay with the text
below them), and consecutive sections are packed into passages of at most
MAX_PASSAGE_CHARS, as element-based chunking merges small elements (X9 report). A
table forms passages of its own, in one of the forms experiment X9 compares; its cells
are joined by " | ", a mark that quote verification reads as a space. Running headers
and footers stay out. Each passage records its page and a context (section path, table
caption, column headers) that is searched and embedded with it but never quoted.
"""

import re
import textwrap
from typing import Any

from petroknowledge import config

# X9's table forms: whole tables, one passage per row, both, or whole pages.
FORMS = ("table", "rows", "both", "page")
CELL = " | "  # between a table's cells, as in elements.grid_text
PATH = " › "  # between the parts of a context
SKIPPED = {"furniture", "table_header", "table_row"}  # tables are read whole
Passage = tuple[str, str, str, int]  # heading, context, text, page
Section = tuple[tuple[str, ...], list[str]]  # a section path and its texts
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def pieces(paragraph: str, budget: int) -> list[str]:
    """A paragraph that fits the budget stays whole; a longer one is split at
    sentence ends, and a sentence that is still too long at spaces. Words and
    hyphenated words are never broken, so every quote survives unchanged."""
    if len(paragraph) <= budget:
        return [paragraph]
    parts: list[str] = []
    for sentence in SENTENCE_END.split(paragraph):
        if len(sentence) <= budget:
            parts.append(sentence)
        else:
            parts += textwrap.wrap(
                sentence, budget, break_long_words=False, break_on_hyphens=False
            )
    return parts


def pack(texts: list[str], budget: int) -> list[str]:
    """Texts joined by line breaks into passages of at most `budget` characters; a
    longer text is split as a web section's paragraph is (`pieces`)."""
    passages: list[list[str]] = [[]]
    for text in texts:
        for piece in pieces(text, budget):
            if passages[-1] and len("\n".join([*passages[-1], piece])) > budget:
                passages.append([])
            passages[-1].append(piece)
    return ["\n".join(chunk) for chunk in passages if chunk]


def caption(table: dict[str, Any]) -> str:
    """A table element's caption: its text's first line when the text holds one
    more line than the table has rows (`elements.grid_text`)."""
    lines = table["text"].split("\n")
    return lines[0] if len(lines) == len(table["grid"]) + 1 else ""


def table_lines(table: dict[str, Any], data: set[int]) -> tuple[list[str], list[str]]:
    """A table's header lines and data lines, cells joined by CELL. The data rows
    are those read as rows (`data`); the others are headers."""
    headers = []
    rows = []
    for index, row in enumerate(table["grid"]):
        line = CELL.join(row)
        if index in data:
            rows.append(line)
        else:
            headers.append(line)
    return headers, rows


def group_row(line: str) -> bool:
    """Whether a table row spans the whole table, every cell the same as a merged
    cell is read ("DOWNSTREAM | DOWNSTREAM | DOWNSTREAM"): it names the group of the
    rows below it."""
    cells = line.split(CELL)
    return len(cells) > 1 and len(set(cells)) == 1 and cells[0].strip() != ""


def whole_table(headers: list[str], rows: list[str]) -> list[str]:
    """A table as passages: its header lines, then its rows, split between rows
    when it is too long, each part starting with the header lines and, unless it
    starts with a group row, the last group row before it (K29 F6): a part still
    says which group its rows are in. A part never ends with a group row: it moves
    to the next part, with its rows."""
    budget = config.MAX_PASSAGE_CHARS - len("\n".join(headers)) - 1
    parts: list[list[str]] = [[]]
    carried = [""]  # the group row each part repeats
    group = ""
    for row in rows:
        room = budget - (len(carried[-1]) + 1 if carried[-1] else 0)
        if parts[-1] and len("\n".join([*parts[-1], row])) > room:
            last = parts[-1]
            moved = [last.pop()] if len(last) > 1 and group_row(last[-1]) else []
            parts.append(moved)
            carried.append("" if moved or group_row(row) else group)
        parts[-1].append(row)
        if group_row(row):
            group = row
    texts = []
    for repeated, part in zip(carried, parts, strict=True):
        texts.append("\n".join([*headers, *([repeated] if repeated else []), *part]))
    return texts


def table_passages(
    table: dict[str, Any], elements: list[dict[str, Any]], form: str
) -> list[Passage]:
    """A table's passages in the given form, its row and header elements found
    among the page's `elements`."""
    number = table["table"]
    rows = [e for e in elements if e["kind"] == "table_row" and e["table"] == number]
    headers = [
        e["text"]
        for e in elements
        if e["kind"] == "table_header" and e["table"] == number
    ]
    section = tuple(table["section"])
    heading = section[-1] if section else ""
    context = PATH.join(part for part in [*section, caption(table)] if part)
    head, data = table_lines(table, {row["row"] for row in rows})
    page = table["page"]
    found = []
    if form in ("table", "both", "page"):
        texts = whole_table(head, data)
        found += [(heading, context, text, page) for text in texts if text]
    if form in ("rows", "both"):
        # The column headers as read, else the table's header lines.
        names = headers[0] if headers else " / ".join(head)
        columns = f"{context}\nColumns: {names}" if names else context
        if head:
            found.append((heading, context, "\n".join(head), page))
        found += [(heading, columns, line, page) for line in data]
    return found


def passages_of(section: tuple[str, ...], texts: list[str], page: int) -> list[Passage]:
    """Texts under one section path packed into passages of the page."""
    heading = section[-1] if section else ""
    packed = pack(texts, config.MAX_PASSAGE_CHARS)
    return [(heading, PATH.join(section), text, page) for text in packed]


def merge(sections: list[Section], page: int) -> list[Passage]:
    """Consecutive sections packed into passages of at most MAX_PASSAGE_CHARS: a
    passage ends only where the next section would not fit, and takes the first
    section's path as its context. A section longer than that is split by `pack`."""
    found = []
    texts: list[str] = []
    first: tuple[str, ...] = ()
    for section, lines in sections:
        if texts and len("\n".join([*texts, *lines])) > config.MAX_PASSAGE_CHARS:
            found += passages_of(first, texts, page)
            texts = []
        if not texts:
            first = section
        texts += lines
    if texts:
        found += passages_of(first, texts, page)
    return found


def page_passages(elements: list[dict[str, Any]], form: str) -> list[Passage]:
    """One page's passages, its elements given in reading order: its sections merged
    (`merge`), and each table's own passages. In the `page` form a page's elements
    are packed together whatever their section, tables inline."""
    page = elements[0]["page"]
    found: list[Passage] = []
    sections: list[Section] = []
    texts: list[str] = []
    section: tuple[str, ...] = ()
    body = False  # whether the section being read holds more than headings

    def close() -> None:
        nonlocal texts, body
        if texts:
            sections.append((section, texts))
        texts = []
        body = False

    for element in elements:
        kind = element["kind"]
        text = str(element["text"]).strip()
        here = tuple(element["section"])
        if kind in SKIPPED or (not text and kind != "table"):
            continue
        if form == "page":
            section = section if texts else here
            if kind == "table":
                passages = table_passages(element, elements, form)
                texts += [passage[2] for passage in passages]
            else:
                texts.append(text)
            continue
        if kind == "table":
            close()
            found += merge(sections, page) + table_passages(element, elements, form)
            sections = []
            continue
        if kind == "heading":
            if body:
                close()
            section = (*here, text)  # until the text below it says otherwise
        else:
            if body and here != section:
                close()
            section = here
            body = True
        texts.append(text)
    close()
    return found + merge(sections, page)


def pdf_passages(elements: list[dict[str, Any]], form: str) -> list[Passage]:
    """A reading's passages in the given X9 form, page by page."""
    if form not in FORMS:
        raise ValueError(f"unknown table form {form!r}; choose one of {FORMS}")
    pages: dict[int, list[dict[str, Any]]] = {}
    for element in elements:
        pages.setdefault(element["page"], []).append(element)
    found = []
    for number in sorted(pages):
        found += page_passages(pages[number], form)
    return found
