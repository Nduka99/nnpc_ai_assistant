"""Tables read again by a vision model. Every table, answer and word is invented."""

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

from petroknowledge import llm, tables
from petroknowledge.tables import Cell

HTML = (
    "<table><thead><tr><th rowspan='2'>Property</th><th colspan=\"2\">Class</th></tr>"
    "<tr><th>i</th><th>ii</th></tr></thead>"
    "<tr><td>Fire</td><td>A1</td><td>A 2</td></tr>"
    "<tr><td>Strength<br/>28 days</td><td colspan='2'>5 N/mm<sup>2</sup></td></tr>"
    "</table>"
)
OTSL = (
    "<fcel>Property<fcel>Class<lcel><nl>"
    "<ucel><fcel>i<fcel>ii<nl>"
    "<fcel>Fire<fcel>A1<ecel><nl>"
)


class Image:
    """Stands in for a PIL image: saving writes fixed bytes."""

    def save(self, buffer: Any, format: str) -> None:
        buffer.write(b"png")


IMAGE: Any = Image()


def test_the_model_is_asked_with_the_image_and_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}

    def post(url: str, **kwargs: Any) -> httpx.Response:
        sent.update(url=url, **kwargs)
        body = {"choices": [{"message": {"content": "<table></table>"}}]}
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(llm.CLIENT, "post", post)
    monkeypatch.setattr(llm, "auth", lambda: {"Authorization": "Bearer k"})

    answer = tables.recognise(IMAGE, "http://127.0.0.1:8083", "OCR:")

    assert answer == "<table></table>"
    assert sent["url"] == "http://127.0.0.1:8083/v1/chat/completions"
    content = sent["json"]["messages"][0]["content"]
    assert content[0]["image_url"]["url"] == "data:image/png;base64,cG5n"
    assert content[1] == {"type": "text", "text": "OCR:"}
    assert sent["json"]["max_tokens"] == tables.MAX_TOKENS
    assert sent["json"]["top_k"] == 1 and sent["json"]["repeat_penalty"] == 1.1
    assert sent["headers"] == {"Authorization": "Bearer k"}


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, json={"choices": []}),
    ],
)
def test_a_failed_or_malformed_answer_is_a_model_server_error(
    monkeypatch: pytest.MonkeyPatch, response: httpx.Response
) -> None:
    response.request = httpx.Request("POST", "http://x")
    monkeypatch.setattr(llm.CLIENT, "post", lambda url, **kwargs: response)

    with pytest.raises(llm.ModelServerError, match="vision model at http://x"):
        tables.recognise(IMAGE, "http://x")


def test_html_cells_keep_their_spans_headers_and_line_breaks() -> None:
    cells = tables.parse(HTML)

    assert cells == [
        Cell(0, 0, "Property", rows=2, header=True),
        Cell(0, 1, "Class", columns=2, header=True),
        Cell(1, 1, "i", header=True),
        Cell(1, 2, "ii", header=True),
        Cell(2, 0, "Fire"),
        Cell(2, 1, "A1"),
        Cell(2, 2, "A 2"),
        Cell(3, 0, "Strength 28 days"),
        Cell(3, 1, "5 N/mm2", columns=2),
    ]
    odd = tables.parse_html("<td>outside</td><TABLE><tr><td rowspan='x'>a</td></tr>")
    assert odd == [Cell(0, 0, "a")]


def test_otsl_cells_extend_left_up_or_both() -> None:
    assert tables.parse(OTSL) == [
        Cell(0, 0, "Property", rows=2),
        Cell(0, 1, "Class", columns=2),
        Cell(1, 1, "i"),
        Cell(1, 2, "ii"),
        Cell(2, 0, "Fire"),
        Cell(2, 1, "A1"),
        Cell(2, 2, ""),
    ]
    # The model writes a line break inside a cell as a backslash and an n.
    lines = tables.parse_otsl(r"<fcel>Cement\nLime\nSand<fcel>1\n0.25<nl>")
    assert [cell.text for cell in lines] == ["Cement Lime Sand", "1 0.25"]
    both = tables.parse_otsl("<fcel>a<lcel><nl><ucel><xcel><nl><fcel>b")
    assert both == [Cell(0, 0, "a", rows=2, columns=2), Cell(2, 0, "b")]
    assert tables.parse("Just some text.") == []


def test_a_cell_is_spelt_by_unused_words_in_reading_order() -> None:
    words = ["Fire", "A1", "0.8kg/(m", "2", ".min", "0.5", ")", "A1", "‘B’"]

    assert tables.fold("0.8 kg/(m²") == "0.8kg/(m2"
    assert tables.spell("0.8kg/(m2.min0.5)", words, set()) == [2, 3, 4, 5, 6]
    assert tables.spell("A1", words, {1}) == [7]
    assert tables.spell("A1", words, {1, 7}) is None
    assert tables.spell("'b'", words, set()) == [8]
    assert tables.spell("", words, set()) == []
    # A word that starts the text but leaves a rest no word spells is tried and
    # abandoned for the next word ("10" then "25" fails; "1025" spells it).
    assert tables.spell("1025", ["10", "2", "1025"], set()) == [2]
    assert tables.spell("Fire A1", words, {0}) is None
    # Words before the last one used are searched too, after those that follow.
    assert tables.spell("A1 Fire", words, set()) == [1, 0]


def test_the_structure_takes_headers_from_the_model_and_text_from_the_pdf() -> None:
    words = ["Property", "Class", "i", "ii", "Fire", "A1", "A2", "Strength", "28",
             "days", "5", "N/mm2"]  # fmt: skip
    cells = tables.parse(HTML)

    read = tables.structure(cells, words + ["o"], set())

    assert read.headers == ["Property", "Class › i", "Class › ii"]
    # "A 2" is spelt by the PDF's "A2"; the value spanning two columns applies to both.
    headers = ["Property", "Class › i", "Class › ii"]
    assert read.rows == [
        (2, ["Fire", "A1", "A2"], headers),
        (3, ["Strength 28 days", "5 N/mm2", "5 N/mm2"], headers),
    ]
    assert read.dropped == 0
    assert read.leftover == ["o"]  # a tick box no cell holds
    assert read.grid == [
        ["Property", "Class", "Class"],
        ["Property", "i", "ii"],
        ["Fire", "A1", "A2"],
        ["Strength 28 days", "5 N/mm2", "5 N/mm2"],
    ]


def test_without_marked_headers_docling_s_header_rows_are_used() -> None:
    words = ["Property", "Class", "i", "ii", "Fire", "A1", "Smoke", "s1", "Notes"]
    cells = tables.parse(OTSL) + [Cell(3, 0, "Smoke", rows=2), Cell(3, 1, "s1")]
    cells += [Cell(4, 1, "s2 invented"), Cell(5, 0, "Notes", columns=3)]
    # Docling flagged "Class" and "Property i ii" as its header rows: the same words,
    # split differently from the model's rows.
    header_words = {"class", "property", "i", "ii"}

    read = tables.structure(cells, words, header_words)

    assert read.headers == ["Property", "Class › i", "Class › ii"]
    # A value spanning rows repeats; a cell the PDF cannot spell is left empty; a
    # title spanning the whole width is written once.
    headers = ["Property", "Class › i", "Class › ii"]
    assert read.rows == [
        (2, ["Fire", "A1", ""], headers),
        (3, ["Smoke", "s1", ""], headers),
        (4, ["Smoke", "", ""], headers),
        (5, ["Notes", "", ""], headers),
    ]
    assert (read.dropped, read.leftover) == (1, [])
    assert tables.structure(cells, words, set()).headers == ["", "", ""]
    assert json.dumps(read.rows)  # plain, JSON-compatible data


def test_the_model_s_inline_latex_becomes_the_characters_a_pdf_prints() -> None:
    assert tables.unlatex("Thermal Conductivity $\lambda$ (90/90)") == (
        "Thermal Conductivity λ (90/90)"
    )
    assert tables.unlatex("5 N/mm$^2$ at 20-30 $^{\circ}$C") == "5 N/mm2 at 20-30 °C"
    assert tables.unlatex("CO\(_{2}\)e, $\mu$ and <0.1% & A1") == (
        "CO2e, μ and <0.1% & A1"
    )
    # The PDF's micro sign (µ) and the model's Greek mu (μ) fold alike.
    words = ["Water", "Vapour", "Permeability", "µ"]
    assert tables.spell(tables.unlatex("Permeability $\mu$"), words, set()) == [2, 3]


def test_punctuation_the_model_adds_or_drops_does_not_stop_a_cell() -> None:
    words = ["LENGTH", "(SINGLE/DUAL", "CARRIAGE", "Transfers-", "in", "·Updated"]

    # The PDF lacks the closing bracket; its words still spell the header.
    assert tables.spell("LENGTH(SINGLE/DUALCARRIAGE)", words, set()) == [0, 1, 2]
    assert tables.spell("Transfersin", words, set()) == [3, 4]  # a hyphen left out
    assert tables.spell("•Updated", words, set()) == [5]  # another bullet sign
    assert tables.spell("Transfers out", words, set()) is None  # words still count


def test_stacked_tables_read_as_one_keep_each_tables_own_headers() -> None:
    def row(number: int, *texts: str) -> list[Cell]:
        return [Cell(number, column, text) for column, text in enumerate(texts) if text]

    cells = [Cell(0, 0, "Group", columns=3, header=True)]
    cells += row(1, "", "At 1 January 2023", "Share of profit")
    cells = [replace(cell, header=True) for cell in cells]
    cells += row(2, "Calson", "2,310", "-")
    cells += row(3, "Section B notes")  # one text across the table: a row
    cells += row(4, "Hyson", "1,421", "(1)")
    cells += [Cell(5, 0, "Group", columns=3)]
    cells += row(6, "", "At 21 September 2021", "Transfers-in")
    cells += row(7, "Calson", "-", "4,516")
    cells += row(8, "I", "PIPELINES")  # a section line leaving a value column empty
    cells += row(9, "Anoh", "-", "87,091")
    cells += row(10, "Prepared by", "Finance")  # text no value follows: a row
    words = ["Group", "At", "1", "January", "2023", "Share", "of", "profit", "Calson",
             "2,310", "-", "Section", "B", "notes", "Hyson", "1,421", "(1)", "Group",
             "At", "21", "September", "2021", "Transfers-in", "Calson", "-", "4,516",
             "I", "PIPELINES", "Anoh", "-", "87,091", "Prepared", "by",
             "Finance"]  # fmt: skip

    read = tables.structure(cells, words, set())

    first = ["Group", "Group › At 1 January 2023", "Group › Share of profit"]
    second = ["Group", "At 21 September 2021", "Transfers-in"]
    assert read.headers == first
    assert [(number, values[0], block) for number, values, block in read.rows] == [
        (2, "Calson", first),
        (3, "Section B notes", first),
        (4, "Hyson", first),
        (7, "Calson", second),
        (8, "I", second),
        (9, "Anoh", second),
        (10, "Prepared by", second),
    ]  # fmt: skip


def test_a_row_the_model_wrote_twice_is_read_once() -> None:
    cells = [Cell(0, 0, "", rows=2), Cell(0, 1, "2023"),
             Cell(1, 1, "23,990,048"),
             Cell(2, 0, "Revenue"), Cell(2, 1, "23,990,048"),
             Cell(3, 1, "5,988,688"), Cell(4, 0, "Profit"),
             Cell(4, 1, "1")]  # fmt: skip
    words = ["2023", "Revenue", "23,990,048", "5,988,688", "Profit", "1"]

    read = tables.structure(cells, words, {"2023"})

    assert tables.repeated_rows(cells) == {1}  # row 3 differs from the row after it
    assert [values for _, values, _ in read.rows] == [
        ["Revenue", "23,990,048"],
        ["", "5,988,688"],
        ["Profit", "1"],
    ]
    assert read.dropped == 0
