"""Which words a reader sees on a PDF page. Every page and word here is invented; the
PDFs are drawn by pdfium in the tests."""

import ctypes
from collections import Counter
from pathlib import Path
from typing import Any

import pypdfium2
import pypdfium2.raw as pdfium_c
import pytest
from PIL import Image

from petroknowledge import visibility

WHITE = 255


def page_image(ink: tuple[int, int, int, int] | None = None) -> Image.Image:
    """A 200 x 200 grey page (100 x 100 points at scale 2), with ink in one square."""
    image = Image.new("L", (200, 200), WHITE)
    if ink:
        image.paste(0, ink)
    return image


def test_words_are_compared_without_case_or_edge_marks_and_symbols_never() -> None:
    assert [visibility.key(word) for word in ("Mix,", "THEN", "(wait.)")] == [
        "mix",
        "then",
        "wait",
    ]
    assert visibility.key("•") == visibility.key("—") == ""


class TextPage:
    """Stands in for pdfium's text page: each character 1 point wide on one row."""

    def __init__(self, text: str) -> None:
        self.text = text

    def get_text_range(self) -> str:
        return self.text

    def get_charbox(self, index: int) -> tuple[float, float, float, float]:
        return (10.0 + index, 700.0, 11.0 + index, 710.0)


def test_each_word_gets_the_box_around_its_characters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(visibility, "layer", lambda textpage, index: index == 0)

    lines = visibility.text_lines(
        TextPage("Ab\r\n  \r\ncd\t e\r\nself￾declared"), height=800
    )

    # "Ab" is characters 0-1, "cd" 8-9, "e" 12, "self-declared" 15-27.
    assert lines == [
        [("Ab", (10.0, 90.0, 12.0, 100.0), True)],
        [("cd", (18.0, 90.0, 20.0, 100.0), False),
         ("e", (22.0, 90.0, 23.0, 100.0), False)],
        [("self-declared", (25.0, 90.0, 38.0, 100.0), False)],
    ]  # fmt: skip


def test_a_word_is_visible_where_its_text_draws_or_a_layer_has_ink() -> None:
    inked = page_image(ink=(20, 20, 31, 31))
    drawn_text = (inked, page_image(ink=(20, 20, 31, 31)).point(lambda v: 255 - v))
    no_text = (inked, page_image().point(lambda v: 0))

    assert visibility.visible(("a", (10, 10, 15, 15), False), (100, 100), drawn_text)
    assert not visibility.visible(("a", (10, 10, 15, 15), False), (100, 100), no_text)
    assert visibility.visible(("a", (10, 10, 15, 15), True), (100, 100), no_text)
    assert not visibility.visible(("a", (40, 40, 45, 45), True), (100, 100), no_text)
    below = ("a", (10.0, 120.0, 15.0, 130.0), True)
    assert not visibility.visible(below, (100, 100), drawn_text)  # off the page
    # On a 100 x 150 page whose drawing ends at 100 points: nothing to look at.
    assert not visibility.visible(below, (100, 150), drawn_text)


def text_object(document: Any, text: str, x: float, y: float) -> Any:
    """A line of 12-point Helvetica placed at (x, y) from the bottom-left."""
    font = pdfium_c.FPDFText_LoadStandardFont(document, b"Helvetica")
    made = pdfium_c.FPDFPageObj_CreateTextObj(document, font, 12.0)
    encoded = ctypes.create_string_buffer((text + "\x00").encode("utf-16-le"))
    pdfium_c.FPDFText_SetText(made, ctypes.cast(encoded, pdfium_c.FPDF_WIDESTRING))
    pdfium_c.FPDFPageObj_Transform(made, 1, 0, 0, 1, x, y)
    return made


def rectangle(x: float, y: float, width: float, height: float, grey: int) -> Any:
    """A filled grey rectangle."""
    made = pdfium_c.FPDFPageObj_CreateNewRect(x, y, width, height)
    pdfium_c.FPDFPageObj_SetFillColor(made, grey, grey, grey, 255)
    pdfium_c.FPDFPath_SetDrawMode(made, pdfium_c.FPDF_FILLMODE_WINDING, False)
    return made


def sheet(path: Path) -> None:
    """A three-page PDF, 200 x 200 points. Page 2 holds text a reader sees, an
    invisible text layer over a black box, text under a white box and text off the
    page; page 3 is blank."""
    document = pypdfium2.PdfDocument.new()
    cover = document.new_page(200, 200)
    pdfium_c.FPDFPage_InsertObject(cover, text_object(document, "Cover", 10, 170))
    pdfium_c.FPDFPage_GenerateContent(cover)
    page = document.new_page(200, 200)
    layer = text_object(document, "Gamma delta", 10, 122)
    pdfium_c.FPDFTextObj_SetTextRenderMode(
        layer, pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE
    )
    for made in (
        text_object(document, "Alpha beta", 10, 170),
        rectangle(8, 118, 90, 16, 0),
        layer,
        text_object(document, "Epsilon zeta", 10, 80),
        rectangle(5, 70, 150, 25, 255),
        text_object(document, "Kappa omega", -150, 40),
    ):
        pdfium_c.FPDFPage_InsertObject(page, made)
    pdfium_c.FPDFPage_GenerateContent(page)
    document.new_page(200, 200)
    document.save(path)
    document.close()


def test_a_layer_is_invisible_text_or_a_type_3_font(tmp_path: Path) -> None:
    sheet(tmp_path / "sheet.pdf")
    document = pypdfium2.PdfDocument(tmp_path / "sheet.pdf")
    textpage = document[1].get_textpage()
    text = textpage.get_text_range()

    assert not visibility.layer(textpage, text.index("Alpha"))
    assert visibility.layer(textpage, text.index("Gamma"))
    textpage.close()
    document.close()


def test_each_word_of_a_page_is_marked_with_whether_a_reader_sees_it(
    tmp_path: Path,
) -> None:
    sheet(tmp_path / "sheet.pdf")

    pages = visibility.read_pages(tmp_path / "sheet.pdf", 2, 2)

    size, lines = pages[2]
    assert list(pages) == [2] and size == (200.0, 200.0)
    assert [(word[0], word[2]) for line in lines for word in line] == [
        ("Alpha", True), ("beta", True),  # drawn
        ("Gamma", True), ("delta", True),  # a layer over ink
        ("Epsilon", False), ("zeta", False),  # under a white box
        ("Kappa", False), ("omega", False),  # off the page
    ]  # fmt: skip
    everything = visibility.read_pages(tmp_path / "sheet.pdf", 0, 99)
    assert list(everything) == [1, 2, 3] and everything[3][1] == []


LINES = [
    [("PRODUCT", (10.0, 10.0, 50.0, 20.0), True),
     ("DESCRIPTION", (55.0, 10.0, 95.0, 20.0), True),
     ("VP-009", (100.0, 10.0, 120.0, 20.0), False)],  # a hidden field code
    [("the", (10.0, 30.0, 20.0, 40.0), True),
     ("the", (30.0, 30.0, 40.0, 40.0), False),  # shown elsewhere in the box
     ("•", (0.0, 50.0, 5.0, 55.0), False),  # never compared
     ("Outside", (300.0, 10.0, 320.0, 20.0), False),
     ("Below", (10.0, 200.0, 30.0, 210.0), False)],
]  # fmt: skip


def test_hidden_words_are_those_a_box_holds_only_where_they_are_not_seen() -> None:
    assert visibility.hidden_inside(LINES, (0, 0, 130, 60)) == {"vp-009"}
    assert visibility.hidden_inside(LINES, (290, 0, 330, 30)) == {"outside"}
    assert visibility.hidden_inside(LINES, (0, 100, 100, 150)) == set()


def test_a_box_shows_a_word_that_overlaps_it() -> None:
    # "PRODUCT" runs from 10 to 50: it overlaps this box, though centred outside it.
    assert visibility.shows_inside(LINES, (45.0, 15.0, 52.0, 25.0))
    assert not visibility.shows_inside(LINES, (290.0, 0.0, 330.0, 30.0))  # hidden
    assert not visibility.shows_inside(LINES, (0.0, 100.0, 50.0, 150.0))  # empty


def test_an_item_keeps_only_the_words_its_page_shows() -> None:
    pages = {1: ((600.0, 800.0), LINES), 2: ((600.0, 800.0), [])}
    removed: Counter[int] = Counter()

    def keep(number: int, bbox: Any, text: str) -> str:
        return visibility.keep_visible(pages, removed, number, bbox, text)

    heading = (0.0, 0.0, 130.0, 60.0)
    assert keep(1, heading, "PRODUCT DESCRIPTION VP-009") == "PRODUCT DESCRIPTION"
    assert keep(1, heading, "PRODUCT  DESCRIPTION") == "PRODUCT  DESCRIPTION"
    # The page shows nothing over this box: text read, or placed, where no one sees it.
    nowhere = (0.0, 100.0, 50.0, 150.0)
    assert keep(1, nowhere, "It is very important") == keep(1, nowhere, "") == ""
    # No box, no page read, or no pdfium word on the page: nothing is known hidden.
    assert keep(1, None, "VP-009") == keep(3, heading, "VP-009") == "VP-009"
    assert keep(2, heading, "VP-009") == "VP-009"
    assert removed == Counter({1: 5})
