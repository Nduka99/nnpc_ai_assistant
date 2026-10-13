"""Reading order across columns of one line (X43 A3a).

A key-value box (a label on the left, its value on the right) is drawn as one line,
but a layout model can read its columns as separate blocks and put the values after
all the labels. Text blocks that sit side by side on the same line of a page are read
as one line, left to right, where the first of them stood in reading order. A number
range a line break cuts after its hyphen keeps the hyphen the reader dropped (X44).
"""

import re
from typing import Any

from petroknowledge.elements import Element

# Blocks a layout model read. Recovered lines are already lines of running text,
# placed where they stand (petroknowledge.recovery), so they are never joined.
TEXT = {"paragraph", "heading", "list", "caption"}
OVERLAP = 0.5  # of the shorter block's height: how far apart two first lines may be
# Points: one line of text. A label is one line and its value at most two, so two
# columns of running text, whose paragraphs are taller, are never joined.
ONE_LINE = 20.0


# A number range cut by a line break after its hyphen, as the PDF's text layer has it
# ("12-" at a line's end, "36hrs" starting the next).
NUMBER_BREAK = re.compile(r"(\w*\d)-[ \t]*\r?\n[ \t]*(\d\w*)")


def number_breaks(page_text: str) -> list[tuple[str, str]]:
    """Each number range a page's text layer breaks after its hyphen, as its two
    halves; left out when the page also shows the halves joined as a word of its own,
    which then cannot tell the two apart."""
    found = []
    for left, right in NUMBER_BREAK.findall(page_text):
        if not re.search(rf"(?<![\w.]){re.escape(left + right)}(?!\w)", page_text):
            found.append((left, right))
    return found


def kept_hyphens(text: str, breaks: list[tuple[str, str]]) -> str:
    """The text with each such range the reader joined given back its hyphen: a
    hyphen after a digit never splits a word, so "1236hrs" was "12-36hrs" (X44). Only
    a whole token is mended, so "2023" is never read as a range."""
    for left, right in breaks:
        joined = re.compile(rf"(?<![\w.]){re.escape(left + right)}(?!\w)")
        text = joined.sub(f"{left}-{right}", text)
    return text


def same_line(left: Element, right: Element) -> bool:
    """Whether `right` sits to the right of `left` on the same line of the page."""
    if left.page != right.page or left.bbox is None or right.bbox is None:
        return False
    l_left, l_top, l_right, l_bottom = left.bbox
    r_left, r_top, r_right, r_bottom = right.bbox
    shorter = min(l_bottom - l_top, r_bottom - r_top)
    taller = max(l_bottom - l_top, r_bottom - r_top)
    # A label and its value: one line beside at most two, their first lines level.
    short = 0 < shorter <= ONE_LINE and taller <= 2 * ONE_LINE
    level = abs(l_top - r_top) <= OVERLAP * shorter
    return l_right <= r_left and short and level


def side_by_side(elements: list[Element]) -> list[Element]:
    """The elements with each line's side-by-side text blocks joined, left to right,
    at the first block's place; other elements unchanged and in order."""
    joined: set[int] = set()
    found: list[Element] = []
    for number, element in enumerate(elements):
        if number in joined:
            continue
        if element.kind not in TEXT:
            found.append(element)
            continue
        partners = [
            index
            for index, other in enumerate(elements)
            if index != number
            and index not in joined
            and other.kind in TEXT
            and (same_line(element, other) or same_line(other, element))
        ]
        if not partners:
            found.append(element)
            continue
        joined.update(partners)
        line = [element] + [elements[index] for index in partners]
        line.sort(key=lambda e: e.bbox[0] if e.bbox else 0.0)
        text = " ".join(e.text for e in line)
        found.append(
            Element(element.page, element.kind, text, element.section, element.bbox)
        )
    return found


def from_record(record: dict[str, Any]) -> Element:
    """An element saved in a reading (`dataclasses.asdict`) read back. (Kept out of
    `elements`, whose code is part of every reading's fingerprint.)"""
    box = record["bbox"]
    return Element(
        page=record["page"],
        kind=record["kind"],
        text=record["text"],
        section=tuple(record["section"]),
        bbox=(box[0], box[1], box[2], box[3]) if box else None,
        table=record["table"],
        row=record["row"],
        cells=tuple((header, value) for header, value in record["cells"]),
        grid=tuple(tuple(row) for row in record["grid"]),
    )
