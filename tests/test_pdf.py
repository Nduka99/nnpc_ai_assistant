"""Reading PDFs into elements. The Docling documents here are built by hand, so no
model runs; every text is invented."""

from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc.base import BoundingBox, CoordOrigin, Size
from docling_core.types.doc.common.content_layer import ContentLayer
from docling_core.types.doc.common.reference import ProvenanceItem
from docling_core.types.doc.document import DoclingDocument
from docling_core.types.doc.items.key_value import GraphCell, GraphData
from docling_core.types.doc.items.table.table import TableItem
from docling_core.types.doc.items.table.table_data import TableCell, TableData
from docling_core.types.doc.labels import DocItemLabel, GraphCellLabel
from docling_core.types.doc.page import BoundingRectangle, TextCell

from petroknowledge import pdf, recovery, tables, visibility
from petroknowledge.elements import Element, row_text


def prov(page: int = 1, top: float = 700.0) -> ProvenanceItem:
    """A box 100 points wide and 20 high, `top` points above the page's bottom."""
    box = BoundingBox(
        l=50, t=top, r=150, b=top - 20, coord_origin=CoordOrigin.BOTTOMLEFT
    )
    return ProvenanceItem(page_no=page, bbox=box, charspan=(0, 1))


def cell(text: str, row: int, column: int, header: bool = False) -> TableCell:
    return TableCell(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + 1,
        start_col_offset_idx=column,
        end_col_offset_idx=column + 1,
        column_header=header,
    )


def datasheet() -> DoclingDocument:
    """A two-page datasheet: title, sections, a list, a table with a two-row header
    and a blank corner, a drawing with a callout, a running footer."""
    doc = DoclingDocument(name="mortex")
    for page in (1, 2):
        doc.add_page(page_no=page, size=Size(width=600, height=800))
    doc.add_title(text="Mortex Mortar", prov=prov())
    doc.add_heading(text="Mixing", level=1, prov=prov(top=650))
    doc.add_text(
        label=DocItemLabel.TEXT, text="Add 4 litres of water.", prov=prov(top=620)
    )
    items = doc.add_list_group()
    doc.add_list_item(
        text="Mix for 3 minutes.",
        marker="1.",
        orig="1. Mix for 3 minutes.",
        parent=items,
        prov=prov(top=600),
    )
    doc.add_heading(text="Performance", level=1, prov=prov(top=560))
    cells = [
        cell("", 0, 0), cell("Grade", 0, 1, header=True),
        cell("Property", 1, 0, header=True), cell("M5", 1, 1, header=True),
        cell("Strength", 2, 0), cell("5 N/mm2", 2, 1),
        cell("Fire", 3, 0), cell("A1", 3, 1),
    ]  # fmt: skip
    caption = doc.add_text(
        label=DocItemLabel.CAPTION, text="Table 1", prov=prov(top=540)
    )
    doc.add_table(
        data=TableData(num_rows=4, num_cols=2, table_cells=cells),
        caption=caption,
        prov=prov(top=520),
    )
    doc.add_heading(text="Detail", level=2, prov=prov(page=2))
    drawing = doc.add_picture(prov=prov(page=2, top=600))
    doc.add_text(
        label=DocItemLabel.TEXT, text="Mesh here", prov=prov(2, 500), parent=drawing
    )
    doc.add_text(
        label=DocItemLabel.PAGE_FOOTER,
        text="Page 2 of 2",
        prov=prov(page=2, top=40),
        content_layer=ContentLayer.FURNITURE,
    )
    doc.add_text(label=DocItemLabel.TEXT, text="no position")  # no provenance
    doc.add_table(data=TableData(num_rows=0, num_cols=0), prov=prov(page=2, top=300))
    contents = TableData(
        num_rows=1, num_cols=2, table_cells=[cell("Mixing", 0, 0), cell("1", 0, 1)]
    )
    doc.add_table(
        data=contents, prov=prov(page=2, top=250), label=DocItemLabel.DOCUMENT_INDEX
    )
    pairs = [
        GraphCell(label=GraphCellLabel.KEY, cell_id=0, text="pH", orig="pH"),
        GraphCell(label=GraphCellLabel.VALUE, cell_id=1, text="11.2", orig="11.2"),
    ]
    doc.add_key_values(graph=GraphData(cells=pairs, links=[]), prov=prov(2, 200))
    return doc  # fmt: skip


def test_a_document_becomes_elements_in_reading_order() -> None:
    found = pdf.elements(datasheet())

    assert [(e.page, e.kind, e.text) for e in found] == [
        (1, "heading", "Mortex Mortar"),
        (1, "heading", "Mixing"),
        (1, "paragraph", "Add 4 litres of water."),
        (1, "list", "1. Mix for 3 minutes."),
        (1, "heading", "Performance"),
        (1, "caption", "Table 1"),
        (1, "table", "Table 1\n | Grade\nProperty | M5\nStrength | 5 N/mm2\nFire | A1"),
        (1, "table_header", "Property | Grade › M5"),
        (1, "table_row", "Table 1 › Strength — Grade › M5: 5 N/mm2"),
        (1, "table_row", "Table 1 › Fire — Grade › M5: A1"),
        (2, "heading", "Detail"),
        (2, "figure", ""),
        (2, "paragraph", "Mesh here"),
        (2, "furniture", "Page 2 of 2"),
        (2, "table", "Mixing | 1"),
        (2, "table_row", "Mixing — 1"),
        (2, "paragraph", "pH 11.2"),
    ]


def test_each_element_keeps_its_headings_box_and_cells() -> None:
    found = {e.text: e for e in pdf.elements(datasheet())}

    assert found["Add 4 litres of water."].section == ("Mortex Mortar", "Mixing")
    assert found["Mesh here"].section == ("Mortex Mortar", "Performance", "Detail")
    # 700 points above the bottom of an 800-point page is 100 from its top.
    assert found["Mortex Mortar"].bbox == (50.0, 100.0, 150.0, 120.0)
    row = found["Table 1 › Strength — Grade › M5: 5 N/mm2"]
    assert (row.table, row.row) == (1, 2)
    assert row.cells == (("Property", "Strength"), ("Grade › M5", "5 N/mm2"))
    assert found["Property | Grade › M5"].table == 1


def test_a_row_is_one_quotable_line() -> None:
    cells = (("Property", "Fire"), ("Class i", "A1"), ("", "EN 998"), ("Note", ""))

    assert row_text(cells) == "Fire — Class i: A1; EN 998"
    assert row_text((("", ""), ("Class i", "A1"))) == "Class i: A1"
    assert row_text((("Property", "Fire"),), "Table 2") == "Table 2 › Fire"


def test_a_pdf_is_read_through_the_converter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class Converter:
        def __init__(self, images_scale: float = 0.0) -> None:
            assert images_scale == 0.0  # no page images without a vision model

        def convert(self, path: Path, page_range: tuple[int, int]) -> Any:
            calls.append((path, page_range))
            return type("Result", (), {"document": datasheet()})

    recovered = []
    # "Detail" is shown on page 2; a hidden word lies in the footer's box.
    seen = [[("Detail", (50.0, 100.0, 80.0, 120.0), True),
             ("VP-001", (100.0, 765.0, 120.0, 775.0), False)]]  # fmt: skip
    pages = {2: ((600.0, 800.0), seen)}

    def read_pages(path: Path, first: int, last: int) -> Any:
        assert (path, first, last) == (Path("sheet.pdf"), 2, 2)
        return pages

    def recover(
        found: list[Element], read: Any, removed: Counter[int], checks: Any
    ) -> list[Element]:
        recovered.append((len(found), read, dict(removed), checks))
        return found + [Element(2, "recovered", "A line the reading lost")]

    monkeypatch.setattr(pdf, "converter", Converter)
    monkeypatch.setattr(visibility, "read_pages", read_pages)
    monkeypatch.setattr(recovery, "recover", recover)
    checks: dict[int, dict[str, int]] = {}

    found = pdf.read_pdf(Path("sheet.pdf"), first=2, last=2, checks=checks)

    assert calls == [(Path("sheet.pdf"), (2, 2))]
    assert all(isinstance(element, Element) for element in found)
    assert found[-1] == Element(2, "recovered", "A line the reading lost")
    # pdfium shows only "Detail" on page 2, so the five other items there (10 words)
    # lie where no reader sees them and are dropped; the figure stays.
    elements = len(pdf.elements(datasheet()))
    assert recovered == [(elements - 5, pages, {2: 10}, checks)]


HIDDEN = {"Mixing", "Mesh", "here", "M5"}


def keep_shown(page: int, bbox: Any, text: str) -> str:
    """Takes the words in HIDDEN out of any text, as if its page hid them."""
    words = text.split()
    kept = [word for word in words if word not in HIDDEN]
    return text if len(kept) == len(words) else " ".join(kept)


def test_words_a_page_hides_are_taken_out_of_every_item() -> None:
    found = pdf.elements(datasheet(), keep=keep_shown)

    texts = [(e.kind, e.text) for e in found]
    assert ("heading", "Mixing") not in texts  # wholly hidden: dropped
    assert ("paragraph", "Mesh here") not in texts
    assert ("figure", "") in texts  # a figure stays
    assert ("table_header", "Property | Grade") in texts  # "M5" hidden
    assert ("table_row", "Table 1 › Strength — Grade: 5 N/mm2") in texts
    assert ("table_row", "1") in texts  # the contents table without "Mixing"
    water = next(e for e in found if e.text == "Add 4 litres of water.")
    assert water.section == ("Mortex Mortar",)  # a hidden heading is no section


def test_a_table_the_page_hides_is_dropped_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = datasheet()
    table = next(item for item, _ in doc.iterate_items() if hasattr(item, "data"))

    def nothing(page: int, bbox: Any, text: str) -> str:
        return ""

    monkeypatch.setattr(TableItem, "get_image", lambda self, document: "image")
    monkeypatch.setattr(tables, "recognise", lambda image, url: ANSWER)
    words = {1: TABLE_WORDS}

    assert pdf.table_rows(table, 1, (), doc, nothing) == []
    assert pdf.vlm_table(table, 1, (), doc, words, "u", Counter(), nothing) == []


def test_the_converter_reads_text_cells_without_ocr() -> None:
    options: Any = pdf.converter().format_to_options[InputFormat.PDF].pipeline_options

    assert options.do_ocr is False
    assert options.table_structure_options.mode.value == "accurate"


def test_a_converter_for_a_vision_model_keeps_page_images_and_words() -> None:
    found = pdf.converter(pdf.IMAGES_SCALE).format_to_options[InputFormat.PDF]
    options: Any = found.pipeline_options
    plain: Any = pdf.converter().format_to_options[InputFormat.PDF].pipeline_options

    assert options.generate_page_images and options.generate_parsed_pages
    assert options.images_scale == pdf.IMAGES_SCALE
    assert not plain.generate_page_images and not plain.generate_parsed_pages


def test_a_page_is_rendered_by_the_converter(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered: dict[int, Any] = {3: SimpleNamespace(pil_image="image of page 3")}

    class Converter:
        def __init__(self, images_scale: float) -> None:
            assert images_scale == 2.0

        def convert(self, path: Path, page_range: tuple[int, int]) -> Any:
            page = page_range[0]
            pages = {page: SimpleNamespace(image=rendered.get(page))}
            return SimpleNamespace(document=SimpleNamespace(pages=pages))

    monkeypatch.setattr(pdf, "converter", Converter)

    assert pdf.page_image(Path("a.pdf"), 3, 2.0) == "image of page 3"
    with pytest.raises(ValueError, match="page 4 of a.pdf was not rendered"):
        pdf.page_image(Path("a.pdf"), 4, 2.0)


def word(text: str, left: float, top: float) -> TextCell:
    """A word 10 points high, `top` points above the bottom of an 800-point page."""
    box = BoundingBox(
        l=left, t=top, r=left + 20, b=top - 10, coord_origin=CoordOrigin.BOTTOMLEFT
    )
    return TextCell(
        text=text, orig=text, rect=BoundingRectangle.from_bounding_box(box),
        from_ocr=False,
    )  # fmt: skip


# The table's box runs from 280 to 300 points below the top of page 1 (see prov),
# so a word 515 points above the bottom is centred inside it.
TABLE_WORDS = [word(text, 60 + 5 * n, 515)
               for n, text in enumerate(["Property", "M5", "Strength", "5", "N/mm2",
                                         "Fire", "A1"])]  # fmt: skip
OUTSIDE = word("Mixing", 60, 650)
ANSWER = (
    "<fcel>Property<fcel>M5<nl><fcel>Strength<fcel>5 N/mm2<nl><fcel>Fire<fcel>A1<nl>"
)


def test_words_inside_a_box_and_docling_s_header_words() -> None:
    doc = datasheet()
    table = next(item for item, _ in doc.iterate_items() if hasattr(item, "data"))
    words = TABLE_WORDS + [OUTSIDE]

    assert pdf.words_inside(pdf.box(table, doc), words, 800) == [
        "Property", "M5", "Strength", "5", "N/mm2", "Fire", "A1",
    ]  # fmt: skip
    assert pdf.header_words(table) == {"grade", "property", "m5"}


def test_a_table_is_read_again_from_its_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    doc = datasheet()
    table = next(item for item, _ in doc.iterate_items() if hasattr(item, "data"))
    answers = [ANSWER, "No table here."]
    monkeypatch.setattr(TableItem, "get_image", lambda self, document: "image")
    monkeypatch.setattr(tables, "recognise", lambda image, url: answers.pop(0))
    stats: Counter[str] = Counter()
    words = {1: TABLE_WORDS + [OUTSIDE, word("Mixing", 120, 515)]}

    rows = pdf.vlm_table(table, 1, ("Performance",), doc, words, "http://vlm", stats)
    unread = pdf.vlm_table(table, 1, ("Performance",), doc, words, "http://vlm", stats)

    assert [(e.kind, e.text) for e in rows or []] == [
        ("table", "Table 1\nProperty | M5\nStrength | 5 N/mm2\nFire | A1"),
        ("table_header", "Property | M5"),
        ("table_row", "Table 1 › Strength — M5: 5 N/mm2"),
        ("table_row", "Table 1 › Fire — M5: A1"),
        ("recovered", "Mixing"),  # a word in the box that no cell holds
    ]
    assert unread is None
    assert stats == Counter(tables=2, cells=6, unread=1)
    answers[:] = [ANSWER]
    hidden = pdf.vlm_table(table, 1, (), doc, words, "http://vlm", stats, keep_shown)
    assert not any("M5" in e.text or "Mixing" in e.text for e in hidden or [])
    monkeypatch.setattr(TableItem, "get_image", lambda self, document: None)
    assert pdf.vlm_table(table, 1, (), doc, words, "http://vlm", stats) is None


def test_each_header_block_of_a_table_is_kept_and_heads_its_rows() -> None:
    doc = datasheet()
    table = next(item for item, _ in doc.iterate_items() if hasattr(item, "data"))
    grid = [["Item", "2023"], ["Calson", "2,310"], ["Item", "2022"], ["Calson", "-"]]
    rows = [(1, ["Calson", "2,310"], ["Item", "2023"]),
            (3, ["Calson", "-"], ["Item", "2022"]),
            (4, ["Notes", ""], ["", ""])]  # fmt: skip

    found = pdf.table_elements(grid, ["Item", "2023"], rows, table, 1, (), doc)

    assert [(e.kind, e.text) for e in found[1:]] == [
        ("table_header", "Item | 2023"),
        ("table_header", "Item | 2022"),  # a block with no header text adds none
        ("table_row", "Table 1 › Calson — 2023: 2,310"),
        ("table_row", "Table 1 › Calson — 2022: -"),
        ("table_row", "Table 1 › Notes"),
    ]


def test_a_pdf_read_with_a_vision_model_keeps_its_other_elements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = SimpleNamespace(word_cells=TABLE_WORDS)
    pages = [SimpleNamespace(page_no=1, parsed_page=parsed),
             SimpleNamespace(page_no=2, parsed_page=None)]  # fmt: skip

    class Converter:
        def __init__(self, images_scale: float) -> None:
            assert images_scale == pdf.IMAGES_SCALE

        def convert(self, path: Path, page_range: tuple[int, int]) -> Any:
            return SimpleNamespace(document=datasheet(), pages=pages)

    answers = [ANSWER, "", ""]
    monkeypatch.setattr(pdf, "converter", Converter)
    monkeypatch.setattr(recovery, "recover", lambda found, *rest: found)
    monkeypatch.setattr(visibility, "read_pages", lambda path, first, last: {})
    monkeypatch.setattr(TableItem, "get_image", lambda self, document: "image")
    monkeypatch.setattr(tables, "recognise", lambda image, url: answers.pop(0))
    stats: Counter[str] = Counter()

    found = pdf.read_pdf(Path("sheet.pdf"), vlm="http://vlm", stats=stats)

    texts = [element.text for element in found]
    assert "Table 1 › Strength — M5: 5 N/mm2" in texts
    assert "Mixing — 1" in texts  # the contents table: the model gave no table
    assert stats == Counter(tables=3, cells=6, unread=2)
    answers[:] = [ANSWER, "", ""]
    assert len(pdf.read_pdf(Path("sheet.pdf"), vlm="http://vlm")) == len(found)


def test_page_grades_come_from_docling_s_confidence_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = SimpleNamespace(
        pages={2: SimpleNamespace(low_grade=SimpleNamespace(value="fair"))}
    )

    class Converter:
        def __init__(self, images_scale: float = 0.0) -> None:
            pass

        def convert(self, path: Path, page_range: tuple[int, int]) -> Any:
            return SimpleNamespace(document=datasheet(), confidence=report)

    monkeypatch.setattr(pdf, "converter", Converter)
    monkeypatch.setattr(recovery, "recover", lambda found, *rest: found)
    monkeypatch.setattr(visibility, "read_pages", lambda path, first, last: {})
    grades: dict[int, str] = {}

    found = pdf.read_pdf(Path("sheet.pdf"), grades=grades)

    assert grades == {2: "fair"}
    assert not any(element.kind == "recovered" for element in found)
