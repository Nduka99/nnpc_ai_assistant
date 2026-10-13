"""What the search models and the answer model read of a passage."""

from petroknowledge.records import Passage, as_read, described


def test_the_search_models_read_title_context_and_text_without_blanks() -> None:
    assert described("Annual report", "Results > Revenue", "Revenue rose.") == (
        "Annual report\nResults > Revenue\nRevenue rose."
    )
    assert described("About", "", "Text") == "About\nText"


def test_the_answer_model_reads_title_heading_and_text() -> None:
    passage = Passage(1, "https://x.test/", "About", "Profile", "Text", "2026-10-10")

    assert as_read(passage) == "About\nProfile\nText"
    assert as_read(Passage(2, "u", "About", "", "Text", "t")) == "About\nText"
