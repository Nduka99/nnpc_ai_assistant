"""What a reader sees: numbered dated sources, fixed notices, pointers, never a
removed claim. Every answer here is invented."""

from dataclasses import replace

from petroknowledge import units
from petroknowledge.answer import INSUFFICIENT, REFERRAL
from petroknowledge.records import Answer, Claim, Evidence, Passage, Rejection
from petroknowledge.view import INDEPENDENT, reply_text, view

LINE = Passage(1, "https://acme.test/gas/line", "Gas Line", "Route",
               "Route\nThe line runs 614 km.", "2026-10-10T10:00:00+00:00")  # fmt: skip
NEWS = Passage(2, "https://acme.test/news/flare", "Flare Cut", "Flaring",
               "Routine flaring fell.", "2026-10-10T11:00:00+00:00",
               published="2026-09-29")  # fmt: skip
SCAN = Passage(3, "https://acme.test/reports/annual report.pdf", "Annual Report",
               "Notes", "Revenue rose.", "d", 74, quotable=False)  # fmt: skip
PICTURE = Passage(4, "https://acme.test/news/flare", "Flare Cut", "Picture", "", "d",
                  context="Picture\nFLARE DOWN", image="a" * 64)  # fmt: skip


def evidence(passage: Passage, quote: str) -> Evidence:
    return Evidence(passage.id, passage.url, passage.title, passage.heading, quote,
                    f"{passage.url}#:~:text=x", passage.fetched_at,
                    passage.published)  # fmt: skip


ROUTE = evidence(LINE, "The line runs\n 614 km.")
FLARE = evidence(NEWS, "Routine flaring fell.")
ANSWERED = Answer(
    "How long is the line, and did flaring fall?",
    "answered",
    "",
    (Claim("The line runs 614 km.", (ROUTE,)),
     Claim("Routine flaring fell along the 614 km line.", (FLARE, ROUTE), 2)),
    (LINE, NEWS),
    (Rejection("The line is new.", "quote not in S1"),),
    pointers=(SCAN, PICTURE),
)  # fmt: skip
REFUSED = Answer("What does gas cost?", "insufficient_evidence", INSUFFICIENT, (),
                 (LINE, NEWS), ())  # fmt: skip
REFERRED = Answer("Should I buy shares?", "safety_referral", REFERRAL, (), (), ())


def test_each_distinct_quote_becomes_one_numbered_dated_source() -> None:
    result = view(ANSWERED)

    assert [claim["sources"] for claim in result["claims"]] == [[1], [2, 1]]
    first, second = result["sources"]
    assert (first["number"], first["title"], first["heading"]) == (1, "Gas Line",
                                                                   "Route")  # fmt: skip
    assert first["quote"] == "The line runs 614 km."  # one line
    assert first["link"] == "https://acme.test/gas/line#:~:text=x"
    assert (first["published"], first["captured"]) == ("", "2026-10-10")
    assert (second["published"], second["url"]) == ("2026-09-29", NEWS.url)
    assert not first["compiled"]


def test_a_quote_from_a_compiled_list_is_marked_compiled() -> None:
    listed = replace(LINE, id=9, heading=units.COMPILED)
    claim = Claim("The group has one gas unit.", (evidence(listed, "Gas"),))

    (source,) = view(replace(ANSWERED, claims=(claim,)))["sources"]

    assert source["compiled"]


def test_search_only_pages_are_pointed_to_once_each_never_quoted() -> None:
    same_page = replace(SCAN, id=5, text="Costs fell.")
    others = [replace(SCAN, id=n, page=n) for n in (80, 81)]
    pointing = replace(ANSWERED, pointers=(SCAN, same_page, PICTURE, *others))

    result = view(pointing)

    assert result["pointers"] == [
        {
            "title": "Annual Report",
            "kind": "scanned page",
            "link": "https://acme.test/reports/annual%20report.pdf#page=74",
        },
        {
            "title": "Flare Cut",
            "kind": "picture",
            "link": "https://acme.test/news/flare",
        },
        {
            "title": "Annual Report",
            "kind": "scanned page",
            "link": "https://acme.test/reports/annual%20report.pdf#page=80",
        },
    ]  # fmt: skip  (page 74 once; page 81 is past the three shown)
    assert "Revenue rose." not in str(result) and "FLARE DOWN" not in str(result)


def test_a_removed_claim_never_appears_and_only_a_refusal_lists_pages() -> None:
    assert "The line is new" not in str(view(ANSWERED))
    assert view(ANSWERED)["closest_pages"] == []
    assert view(REFUSED)["closest_pages"] == [
        {"title": "Gas Line", "url": LINE.url},
        {"title": "Flare Cut", "url": NEWS.url},
    ]


def test_a_referral_is_its_fixed_text_and_the_product_says_it_is_independent() -> None:
    result = view(REFERRED)

    assert (result["status"], result["notice"]) == ("safety_referral", REFERRAL)
    assert result["claims"] == [] and result["sources"] == [] == result["pointers"]
    assert "not affiliated with or endorsed by NNPC Limited" in INDEPENDENT


def test_the_reply_carried_into_a_conversation_is_what_the_reader_saw() -> None:
    shown = view(ANSWERED)

    assert reply_text(shown) == (
        "The line runs 614 km.\nRoutine flaring fell along the 614 km line."
    )
    assert reply_text(view(REFUSED)) == INSUFFICIENT
    assert reply_text(view(REFERRED)) == REFERRAL
