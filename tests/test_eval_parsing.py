"""Scoring readers against ground truth written from page images (Lime Green's
X8, adopted with the module). Every page and text is invented."""

from typing import Any

from evaluation import parsing

TABLE = {"header": ["Property", "Class i", "Class ii"], "rows": [["Fire", "A1", "A2"]]}


GRID: dict[str, Any] = {
    "grid": [
        ["Mix", "Class", "Class"],
        ["Mix", "i", "ii"],
        ["Cement", "1", "1"],
        ["Water", "2 ₂", ""],
    ],
    "header_rows": 2,
}


def test_a_cell_is_found_by_its_row_label_column_header_and_value() -> None:
    rows: parsing.Rows = [
        (),
        (("Property", "Fire"), ("Performance › Class i", "A1"),
         ("Performance › Class ii", "A 2")),
    ]  # fmt: skip
    wrong: parsing.Rows = [
        (("Property", "Fire"), ("Class ii", "A1"), ("Class i", "A2"))
    ]

    assert parsing.table_cells([TABLE], rows) == (2, 2, [1.0])
    assert parsing.table_cells([TABLE], wrong) == (0, 2, [0.0])
    any_column = {"header": ["", ""], "rows": [["Fire", "A1"]]}
    assert parsing.table_cells([any_column], wrong) == (1, 1, [1.0])
    empty = {"header": ["", ""], "rows": [["Notes", ""]]}
    assert parsing.table_cells([empty], []) == (0, 0, [1.0])


def test_text_is_kept_word_by_word_and_sentences_in_reading_order() -> None:
    units = ["Mix 4 litres of", "water.", "Store dry. Use within 6 months."]

    assert parsing.words_kept("Mix 4 litres of water. water.", units) == (5, 6)
    sentences = ["Mix 4 litres of water.", "Store dry.", "Use within 12 months."]
    assert parsing.sentences_whole(sentences, units) == (2, 3)
    interleaved = ["Mix 4 litres Store dry.", "of water. Use within"]
    assert parsing.sentences_whole(sentences[:1], interleaved) == (0, 1)
    assert parsing.table_numbers([TABLE], ["Fire A1", "A 2"]) == (2, 2)
    # Docling writes typographic characters plainly; both sides are folded alike.
    typographic = ["Fit in a ‘brick bond’ – done."]
    plain = ["Fit in a 'brick bond' - done."]
    assert parsing.sentences_whole(typographic, plain) == (1, 1)


def test_a_pair_is_kept_on_one_line_or_the_next() -> None:
    units = ["Reaction to fire: Class A1", "pH", "11.2", "Colour"]
    pairs = [["Reaction to fire:", "Class A1"], ["pH", "11.2"], ["Colour", "White"]]

    assert parsing.pairs_kept(pairs, units) == (2, 3)


def page_scores(found: int) -> dict[str, object]:
    measures = {measure: (found, 2) for measure in parsing.GATE}
    return {**measures, "table_shares": [found / 2], "grid": (0, 0), "grid_shares": []}


def test_the_gate_needs_every_measure_and_no_weak_table() -> None:
    perfect = page_scores(2)

    passed = parsing.summarise(
        {"pypdf": [page_scores(1)], "docling": [perfect]}, [9.0, 1.0]
    )
    failed = parsing.summarise({"pypdf": [perfect], "docling": [page_scores(1)]}, [1.0])

    assert all(passed["gate"].values())
    assert passed["seconds"] == {"mean": 5.0, "median": 5.0, "first": 9.0}
    assert not any(failed["gate"].values())
    assert (
        parsing.pooled([{m: (0, 0) for m in [*parsing.GATE, "grid"]}])["cells"] == 1.0
    )
    text = parsing.markdown(failed)
    assert "| cells | 1.000 | 0.500 | >= 0.95 |" in text
    assert "Lowest table (docling): 0.500 (>= 0.8)" in text
    assert "- 2. text kept: FAIL" in text


def test_a_transcription_is_scored_by_its_words() -> None:
    truth = "lime | green T: 01952 728611 Much Wenlock ©"
    answer = "<p>lime green</p> T: 01952 728611 Much Wenlock Shropshire"

    assert parsing.transcription(truth, answer) == {
        "recall": (7, 7),
        "precision": (7, 8),
    }


def test_a_value_is_placed_by_its_row_label_and_a_header_above_it() -> None:
    # Docling-like grid: the header rows are not flagged, only laid out above.
    parsed = (("Mix", "Class", "Class"), ("", "i", "ii"), ("Cement", "1", "1"),
              ("Water", "2 2", ""))  # fmt: skip
    swapped = (("", "ii", "i"), ("Cement", "1", "0"), ("Water", "22", ""))

    assert parsing.grid_cells([GRID], [parsed]) == (3, 3, [1.0])
    assert parsing.grid_cells([GRID], [swapped])[:2] == (1, 3)
    no_header: dict[str, Any] = {"grid": [["Colour", "White"]], "header_rows": 0}
    assert parsing.grid_cells([no_header, TABLE], [(("Colour", "White"),)]) == (
        1, 1, [1.0],
    )  # fmt: skip
    assert parsing.grid_fold("CO₂e") == "co2e"
    assert parsing.header_and_rows(GRID) == (["Mix", "i", "ii"], GRID["grid"][2:])
    assert parsing.header_and_rows(no_header)[0] == ["", ""]


def test_round_3_is_gated_on_values_in_their_row_and_column() -> None:
    page = {"tables": [GRID], "pairs": [], "sentences": []}
    parsed = (("Mix", "i", "ii"), ("Cement", "1", "1"), ("Water", "22", ""))
    good = parsing.score_page(page, ["Cement 1 1 Water 22"], [], "Cement 1", [parsed])
    bad = parsing.score_page(page, ["Cement"], [], "Cement 1", [])

    passed = parsing.summarise({"pypdf": [bad], "docling": [good]}, [1.0])
    failed = parsing.summarise({"pypdf": [good], "docling": [bad]}, [1.0])

    assert passed["grid_mode"] and all(passed["gate"].values())
    assert list(failed["gate"]) == [
        "1. table values in their row and column",
        "2. text kept",
        "3. table numbers",
    ]
    assert not failed["gate"]["1. table values in their row and column"]
    text = parsing.markdown(failed)
    assert "| grid | 1.000 | 0.000 | >= 0.95 |" in text
    assert "| pairs | 1.000 | 1.000 | reported |" in text
