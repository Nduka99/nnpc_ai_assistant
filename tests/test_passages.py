"""Passages built from PDF readings (X9). Every element here is invented."""

from typing import Any

import pytest

from petroknowledge import config, passages


def element(
    kind: str, text: str, section: tuple[str, ...] = (), page: int = 1, **more: Any
) -> dict[str, Any]:
    """An element as a saved reading holds it (plain JSON data)."""
    found = {"page": page, "kind": kind, "text": text, "section": list(section)}
    found.update({"table": None, "row": None, "cells": [], "grid": []})
    found.update(more)
    return found


GRID = [["Property", "Class"], ["Fire", "A1"], ["Strength", "M5"]]
TABLE = [
    element("table", "Table 1\nProperty | Class\nFire | A1\nStrength | M5",
            ("Mortex", "Performance"), table=1, grid=GRID),
    element("table_header", "Property | Class", ("Mortex", "Performance"), table=1),
    element("table_row", "Table 1 › Fire — Class: A1", ("Mortex", "Performance"),
            table=1, row=1),
    element("table_row", "Table 1 › Strength — Class: M5", ("Mortex", "Performance"),
            table=1, row=2),
]  # fmt: skip


def test_long_text_is_split_at_sentences_and_packed_within_the_budget() -> None:
    assert passages.pieces("One two. Three four.", 12) == ["One two.", "Three four."]
    assert passages.pieces("Fits whole.", 50) == ["Fits whole."]
    assert passages.pieces("abcdefgh ijklmnop", 10) == ["abcdefgh", "ijklmnop"]
    assert passages.pack(["aaaa", "bbbb", "cccc"], budget=9) == ["aaaa\nbbbb", "cccc"]
    assert passages.pack([], budget=9) == []


def test_a_table_is_one_passage_with_its_caption_as_context() -> None:
    found = passages.table_passages(TABLE[0], TABLE, "table")

    assert found == [("Performance", "Mortex › Performance › Table 1",
                      "Property | Class\nFire | A1\nStrength | M5", 1)]  # fmt: skip
    assert passages.caption(element("table", "a | b", grid=[["a", "b"]])) == ""


def test_a_long_table_is_split_between_rows_each_part_with_its_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_PASSAGE_CHARS", 32)

    assert passages.whole_table(
        ["Property | Class"], ["Fire | A1", "Strength | M5"]
    ) == [
        "Property | Class\nFire | A1",
        "Property | Class\nStrength | M5",
    ]


def test_a_split_table_part_repeats_the_group_row_its_rows_belong_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "MAX_PASSAGE_CHARS", 50)
    rows = ["GAS | GAS | GAS", "1 | AGL | sales", "2 | AGI | pipes", "3 | AGP | power",
            "OIL | OIL | OIL", "4 | AOL | wells"]  # fmt: skip
    ending = ["1 | AGL | sales", "OIL | OIL | OIL", "4 | AOL | wells"]

    assert passages.whole_table(["No | Unit | Work"], rows) == [
        "No | Unit | Work\nGAS | GAS | GAS\n1 | AGL | sales",
        "No | Unit | Work\nGAS | GAS | GAS\n2 | AGI | pipes",
        "No | Unit | Work\nGAS | GAS | GAS\n3 | AGP | power",
        "No | Unit | Work\nOIL | OIL | OIL\n4 | AOL | wells",
    ]
    assert passages.whole_table(["No | Unit | Work"], ending) == [
        "No | Unit | Work\n1 | AGL | sales",
        "No | Unit | Work\nOIL | OIL | OIL\n4 | AOL | wells",  # moved with its rows
    ]
    assert not passages.group_row("Fire")  # one cell is not a group row
    assert not passages.group_row(" |  | ")  # nor empty cells


def test_each_row_is_a_passage_with_the_column_headers_in_its_context() -> None:
    rows = passages.table_passages(TABLE[0], TABLE, "rows")
    both = passages.table_passages(TABLE[0], TABLE, "both")
    unnamed = passages.table_passages(TABLE[0], [TABLE[0], *TABLE[2:]], "rows")

    context = "Mortex › Performance › Table 1"
    assert rows == [
        ("Performance", context, "Property | Class", 1),
        ("Performance", f"{context}\nColumns: Property | Class", "Fire | A1", 1),
        ("Performance", f"{context}\nColumns: Property | Class", "Strength | M5", 1),
    ]
    assert both == passages.table_passages(TABLE[0], TABLE, "table") + rows
    # Without a header element, the table's header lines name the columns.
    assert unnamed[1][1] == f"{context}\nColumns: Property | Class"


def test_sections_are_merged_until_the_next_would_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page = [
        element("furniture", "Page 1 of 2"),
        element("heading", "Mixing", ("Mortex",)),
        element("heading", "By hand", ("Mortex", "Mixing")),  # stacked headings
        element("paragraph", "Add water.", ("Mortex", "Mixing", "By hand")),
        element("figure", ""),  # no caption: nothing to add
        element("heading", "Curing", ("Mortex",)),
        element("paragraph", "Keep damp for days.", ("Mortex", "Curing")),
        element("recovered", "Mist twice daily.", ("Mortex", "Curing")),
        element("paragraph", "Lime Green Products Ltd", ()),  # another section
    ]

    merged = passages.page_passages(page, "table")
    monkeypatch.setattr(config, "MAX_PASSAGE_CHARS", 45)
    split = passages.page_passages(page, "table")

    assert merged == [("By hand", "Mortex › Mixing › By hand",
                       "Mixing\nBy hand\nAdd water.\nCuring\nKeep damp for days.\n"
                       "Mist twice daily.\nLime Green Products Ltd", 1)]  # fmt: skip
    assert split == [
        ("By hand", "Mortex › Mixing › By hand", "Mixing\nBy hand\nAdd water.", 1),
        (
            "Curing",
            "Mortex › Curing",
            "Curing\nKeep damp for days.\nMist twice daily.",
            1,
        ),
        ("", "", "Lime Green Products Ltd", 1),
    ]


def test_a_table_ends_the_text_before_it_and_keeps_its_own_passages() -> None:
    page = [
        element("heading", "Performance", ("Mortex",)),
        element("paragraph", "Tested to EN 998.", ("Mortex", "Performance")),
        *TABLE,
        element("paragraph", "Values are typical.", ("Mortex", "Performance")),
    ]

    found = passages.page_passages(page, "table")

    assert [text for _, _, text, _ in found] == [
        "Performance\nTested to EN 998.",
        "Property | Class\nFire | A1\nStrength | M5",
        "Values are typical.",
    ]


def test_the_page_form_packs_a_page_whatever_its_sections_tables_inline() -> None:
    page = [
        element("paragraph", "Tested to EN 998.", ("Mortex", "Performance")),
        *TABLE,
        element("heading", "Curing", ("Mortex",)),
    ]

    found = passages.page_passages(page, "page")
    starting = passages.page_passages(TABLE, "page")

    assert found == [("Performance", "Mortex › Performance",
                      "Tested to EN 998.\nProperty | Class\nFire | A1\nStrength | M5\n"
                      "Curing", 1)]  # fmt: skip
    assert starting[0][1] == "Mortex › Performance"  # a page may start with a table


def test_a_reading_becomes_passages_page_by_page() -> None:
    reading = [
        element("paragraph", "Second page.", page=2),
        element("paragraph", "First page.", page=1),
    ]

    assert [p[3] for p in passages.pdf_passages(reading, "rows")] == [1, 2]
    with pytest.raises(ValueError, match="unknown table form 'grid'"):
        passages.pdf_passages(reading, "grid")
