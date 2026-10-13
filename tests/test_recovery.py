"""Recovering the words a page shows but Docling's reading lacks. Every page, line and
element here is invented."""

from collections import Counter
from typing import Any

from petroknowledge import recovery
from petroknowledge.elements import Element


def element(
    kind: str, text: str, box: tuple[float, ...] | None, section: str = ""
) -> Element:
    return Element(1, kind, text, (section,) if section else (), box)  # type: ignore[arg-type]


def test_a_recovered_line_goes_below_its_neighbour_in_its_column() -> None:
    placed = [
        element("heading", "Mixing", (10, 10, 90, 20), "Mixing"),
        element("table_row", "row 1", (10, 30, 90, 60), "Mixing"),
        element("table_row", "row 2", (10, 30, 90, 60), "Mixing"),
        element("paragraph", "side note", (200, 30, 300, 40), "Notes"),
    ]

    after_table = recovery.anchor(placed, (10, 70, 50, 80))
    above_all = recovery.anchor(placed[1:], (10, 0, 50, 5))
    elsewhere = recovery.anchor(placed, (400, 70, 450, 80))

    assert after_table == (3, ("Mixing",))  # after the table's last row
    assert above_all == (0, ("Mixing",))  # before the first element below it
    assert elsewhere == (4, ())  # no element in its column: the page's end
    assert recovery.anchor([element("figure", "", None)], (0, 0, 1, 1)) == (1, ())


def line(text: str, top: float, seen: bool = True, left: float = 10.0) -> Any:
    """A pdfium line of 10-point-wide words, 12 points apart, starting at `left`, each
    marked with whether a reader sees it."""
    return [
        (word, (left + 12 * i, top, left + 12 * i + 10, top + 5), seen)
        for i, word in enumerate(text.split())
    ]


def test_a_page_recovers_the_words_it_shows_and_counts_the_rest() -> None:
    found = [
        element("heading", "Mixing", (10, 10, 90, 20), "Mixing"),
        element("paragraph", "Water absorption 0.8kg/(m 2 .min 0.5 )", (10, 25, 90, 35),
                "Mixing"),
    ]  # fmt: skip
    partly = line("Shown", 80) + line("clipped", 80, seen=False, left=22)
    lines = [
        line("• Water absorption 0.8kg/(m2.min0.5)", 25),  # the bullet is not compared
        line("Mix for three minutes", 40),
        line("Contractor to report errors", 120, seen=False),  # off the page
        line("Clipped note here", 60, seen=False),  # drawn nowhere a reader sees
        partly,
    ]

    placed, check = recovery.recover_page(1, found, lines)

    assert [e.text for e in placed] == [
        "Mixing",
        "Water absorption 0.8kg/(m 2 .min 0.5 )",
        "Mix for three minutes",
        "Shown",  # its hidden word left out
    ]
    assert placed[2] == Element(1, "recovered", "Mix for three minutes", ("Mixing",),
                                (10.0, 40.0, 56.0, 45.0))  # fmt: skip
    assert placed[3].bbox == (10.0, 80.0, 20.0, 85.0)
    assert check == {"words": 8, "kept": 4, "hidden": 8, "recovered": 2}


def test_a_document_s_pages_are_recovered_in_order() -> None:
    pages: dict[int, Any] = {
        1: ((100.0, 100.0), [line("Mix for three minutes", 40)]),
        2: ((100.0, 100.0), []),
    }
    found = [
        Element(1, "heading", "Mixing", (), (10, 10, 90, 20)),
        Element(3, "paragraph", "A page pdfium did not read"),
    ]
    checks: dict[int, dict[str, int]] = {}

    ordered = recovery.recover(found, pages, Counter({1: 2}), checks)

    assert [(e.page, e.text) for e in ordered] == [
        (1, "Mixing"),
        (1, "Mix for three minutes"),
        (3, "A page pdfium did not read"),
    ]
    assert checks[1] == {"words": 4, "kept": 1, "hidden": 0, "recovered": 1,
                         "removed": 2}  # fmt: skip
    assert checks[2]["removed"] == checks[3]["words"] == 0
    assert len(recovery.recover(found, pages, Counter())) == 3
