"""Reading order across columns of one line (X43 A3a). Invented boxes."""

from petroknowledge import layout
from petroknowledge.elements import Element
from petroknowledge.layout import same_line, side_by_side


def block(text: str, left: float, top: float, right: float, bottom: float) -> Element:
    return Element(1, "paragraph", text, ("Box",), (left, top, right, bottom))


def test_a_label_and_its_value_on_one_line_are_read_together() -> None:
    elements = [
        Element(1, "figure", "", (), (0, 0, 50, 50)),
        block("Reaction to fire:", 72, 358, 148, 369),
        block("Adhesion:", 72, 381, 121, 392),
        block("Class A 1", 324, 358, 367, 369),
        block("0.9 N/mm2", 324, 381, 420, 392),
        block("Durability:", 72, 484, 221, 495),
        block("evaluation based on provisions valid", 324, 484, 511, 507),
        Element(2, "paragraph", "Class A 1", (), (324, 358, 367, 369)),
        Element(1, "paragraph", "No box"),
    ]

    found = side_by_side(elements)

    assert [e.text for e in found] == [
        "",
        "Reaction to fire: Class A 1",
        "Adhesion: 0.9 N/mm2",
        "Durability: evaluation based on provisions valid",
        "Class A 1",  # another page
        "No box",
    ]
    assert found[1].bbox == (72, 358, 148, 369) and found[1].section == ("Box",)


def test_two_columns_of_running_text_are_not_joined() -> None:
    left = block("A paragraph of several lines", 72, 100, 280, 160)
    right = block("Another paragraph beside it", 320, 100, 530, 170)
    above = block("Title", 72, 40, 200, 52)

    assert not same_line(left, right)
    assert not same_line(above, right)  # not on the same line
    assert not same_line(right, left)  # left of it, not right
    assert side_by_side([left, right, above]) == [left, right, above]


def test_a_number_range_cut_after_its_hyphen_gets_its_hyphen_back() -> None:
    page = "Drying time is 12-\r\n36hrs. EN 13501-\n1 class. Apply 2-\n3 coats, 23 mm."

    breaks = layout.number_breaks(page)

    assert breaks == [("12", "36hrs"), ("13501", "1")]  # "23" stands alone too
    assert layout.kept_hyphens("Drying is 1236hrs; EN 135011; 2023.", breaks) == (
        "Drying is 12-36hrs; EN 13501-1; 2023."
    )


def test_a_saved_element_reads_back_as_it_was() -> None:
    from dataclasses import asdict

    element = Element(
        3,
        "table",
        "Revenue | 100",
        ("Results",),
        (1.0, 2.0, 3.0, 4.0),
        table=1,
        row=2,
        cells=(("Item", "Revenue"), ("Value", "100")),
        grid=(("Item", "Value"), ("Revenue", "100")),
    )
    unboxed = Element(1, "paragraph", "No box", ())

    assert layout.from_record(asdict(element)) == element
    assert layout.from_record(asdict(unboxed)) == unboxed
