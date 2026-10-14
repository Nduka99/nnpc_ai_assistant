"""Dated role statements: read from a passage, kept only when the passage holds their
quote, grouped by role, and the latest decided by date (K24). Every passage, person
and date here is invented."""

from typing import Any

from petroknowledge import roles

TEXT = (
    "OCTOBER 9, 2026 PRESS RELEASE Acme Gas opened a new plant. "
    "Ada Obi, Executive Vice President, Downstream, Acme Ltd, said the plant runs."
)
GOOD = {"person": "Ada Obi", "role": "Executive Vice President, Downstream",
        "organisation": "Acme Ltd",
        "quote": "Ada Obi, Executive Vice President, Downstream"}  # fmt: skip


def test_only_passages_naming_a_role_title_are_read() -> None:
    assert roles.states_roles("The CFO spoke.") and roles.states_roles(TEXT)
    assert not roles.states_roles("The plant opened on time.")
    assert roles.text_sha256("T", "x") != roles.text_sha256("T", "y")


def test_a_statement_is_kept_only_with_its_quote_naming_the_person() -> None:
    elsewhere = GOOD | {"quote": "Ben Eze, Executive Vice President"}
    unnamed = GOOD | {"quote": "Executive Vice President, Downstream"}

    found = roles.read_statements({"statements": [GOOD, elsewhere, unnamed]}, TEXT)

    assert found == [GOOD]  # the passage's own wording of the quote
    assert roles.read_statements({"statements": []}, TEXT) == []
    assert roles.read_statements({"rows": []}, TEXT) is None
    assert roles.read_statements({"statements": [GOOD | {"person": " "}]}, TEXT) is None
    assert roles.read_statements({"statements": [{"person": "x"}]}, TEXT) is None


def test_a_passage_is_read_by_one_structured_request() -> None:
    asked: list[tuple[str, str, dict[str, Any]]] = []

    def chat(system: str, user: str, schema: dict[str, Any]) -> object:
        asked.append((system, user, schema))
        return {"statements": [GOOD]}

    assert roles.extract("Plant Opens", TEXT, chat) == [GOOD]
    system, user, schema = asked[0]
    assert system == roles.EXTRACT_PROMPT and schema == roles.EXTRACT_SCHEMA
    assert user == f'Passage from "Plant Opens":\n{TEXT}'


def stated(person: str, role: str, organisation: str, day: str, passage: int,
           kind: str = "published") -> dict[str, Any]:  # fmt: skip
    return {"person": person, "role": role, "organisation": organisation,
            "quote": person, "stated_on": day, "date_kind": kind,
            "passage_id": passage, "title": f"Release {passage}"}  # fmt: skip


STATEMENTS = [
    stated("Ben Eze", "Executive Vice President, Downstream", "Acme Ltd",
           "2024-11-13", 1),
    stated("Ada Obi", "Executive Vice President Downstream", "Acme Limited",
           "2026-09-10", 2),
    stated("Ben Eze", "Executive Vice President, Downstream", "ACME LTD",
           "2023-01-05", 3),
    stated("Cy Ude", "Chief Innovation Officer", "Research & Innovation (RAI)",
           "2023-11-23", 4),
    stated("Di Ola", "Chief Innovation Officer", "Research & Innovation (RAI)",
           "2026-10-10", 5, kind="captured"),
    stated("Ed Ani", "Chief Financial Officer", "Acme Ltd", "2025-04-04", 6),
    # Other EVPs: "executive", "vice" and "president" are then common title words.
    stated("Fe Ibe", "Executive Vice President, Upstream", "Acme Ltd",
           "2025-01-01", 7),
    stated("Gu Oko", "Executive Vice President, Gas and Power", "Acme Ltd",
           "2025-01-01", 8),
    stated("Ha Uka", "Executive Vice President, Business Services", "Acme Ltd",
           "2025-01-01", 9),
]  # fmt: skip
LATEST = roles.latest(STATEMENTS)


def test_the_newest_statement_of_a_role_is_current_and_the_rest_history() -> None:
    by_holder = {r["current"]["person"]: r for r in LATEST}

    assert sorted(by_holder)[:3] == ["Ada Obi", "Di Ola", "Ed Ani"]
    downstream = by_holder["Ada Obi"]
    # One person's earlier statements count once, newest first.
    assert [s["person"] for s in downstream["earlier"]] == ["Ben Eze"]
    assert downstream["earlier"][0]["stated_on"] == "2024-11-13"
    assert [s["person"] for s in by_holder["Di Ola"]["earlier"]] == ["Cy Ude"]
    assert roles.organisation_key("Research & Innovation (RAI)") == (
        "research innovation"
    )
    assert roles.initials("chief financial officer") == "cfo"


def test_a_question_names_a_role_by_its_whole_title_or_its_initials() -> None:
    named = roles.naming(LATEST)
    holder = {n: r["current"]["person"] for n, r in enumerate(LATEST)}

    def holders(text: str) -> list[str]:
        return [holder[n] for n in named(text)]

    assert holders("Who is Acme's chief financial officer?") == ["Ed Ani"]
    assert holders("Who is the CFO now?") == ["Ed Ani"]
    assert holders("Who is the executive vice president downstream?") == ["Ada Obi"]
    # A single title word never names a role: titles are made of ordinary words.
    assert holders("Who leads Acme's downstream business these days?") == []
    assert holders("How much oil is financial innovation worth?") == []
    assert holders("What does Acme refine?") == []
    # An organisation named keeps its own roles; naming none means the site's own.
    assert holders("RAI's chief innovation officer, and the CFO?") == ["Di Ola"]
    assert holders("chief innovation officer and CFO") == ["Ed Ani"]


def test_a_one_word_title_names_a_role_only_beside_its_organisation() -> None:
    said = [
        stated("Ada Obi", "Chairman", "Acme Ltd", "2026-01-01", 1),
        stated("Bo Ike", "Principal", "Girls School", "2025-01-01", 2),
    ]  # fmt: skip
    found = roles.latest(said)
    named = roles.naming(found)

    assert named("Which principal regulations apply?") == []
    assert named("Who is listed as chairman in the report?") == []
    assert [
        found[p]["current"]["person"] for p in named("Who is Acme's chairman?")
    ] == ["Ada Obi"]


def test_asked_roles_are_described_by_holder_and_source_without_dates() -> None:
    by_holder = {r["current"]["person"]: r for r in LATEST}

    found = roles.described([by_holder["Ada Obi"], by_holder["Di Ola"]])

    assert found == (
        "- Executive Vice President Downstream (Acme Limited): Ada Obi, in "
        '"Release 2"; earlier holders: Ben Eze.\n'
        "- Chief Innovation Officer (Research & Innovation (RAI)): Di Ola, in "
        '"Release 5"; earlier holders: Cy Ude.'
    )
    assert "2026" not in found  # a date given to the model ends up in a claim
    assert roles.described([]) == ""


def test_one_role_written_many_ways_is_one_group_dated_by_its_newest() -> None:
    said = [
        stated("Umar Ajiya", "CFO", "Acme Ltd", "2024-01-25", 11),
        stated("Umar Ajiya", "Chief Financial Officer (R)", "Acme Ltd",
               "2023-02-17", 12),
        stated("Ed Ani", "Chief Financial Officer", "ACME", "2026-04-29", 13),
        stated("Cy Ude", "Chief Innovation Officer, RAI", "Acme Ltd",
               "2023-11-23", 14),
        stated("Di Ola", "Chief Innovation Officer (CIO)", "Acme Ltd",
               "2026-10-10", 15),
        stated("Ada Obi", "Group Chief Executive Officer, Acme Ltd.", "Acme Ltd",
               "2026-09-29", 16),
        stated("Mo Kye", "GCEO", "Acme Ltd", "2024-12-03", 17),
        stated("Ben Eze", "EVP Downstream", "Acme Ltd", "2024-08-19", 18),
        stated("Ada Obi", "Executive Vice President of Downstream", "Acme Ltd",
               "2026-09-10", 19),
    ]  # fmt: skip
    spelled = roles.expansions(said)
    assert spelled["gceo"] == "group chief executive officer"
    assert roles.role_key(said[0], spelled) == "chief financial officer"
    assert roles.role_key(said[3], spelled) == "chief innovation officer"

    found = {r["key"]: r for r in roles.latest(said)}

    assert sorted(found) == ["chief financial officer", "chief innovation officer",
                             "executive vice president downstream",
                             "group chief executive officer"]  # fmt: skip
    # The date decides, never how often a holder is mentioned.
    assert found["chief financial officer"]["current"]["person"] == "Ed Ani"
    assert found["chief innovation officer"]["current"]["person"] == "Di Ola"
    assert found["group chief executive officer"]["current"]["person"] == "Ada Obi"
    downstream = found["executive vice president downstream"]
    assert [s["person"] for s in downstream["earlier"]] == ["Ben Eze"]


def test_a_nameless_organisation_is_the_site_s_own_and_variants_count_once() -> None:
    said = [
        stated("Ada Obi", "Group Chief Executive Officer", "Acme Ltd", "2026-09-29", 1),
        stated("Mo Kye", "Group Chief Executive Officer", "the Company",
               "2024-03-11", 2),
        stated("Ed Ani", "Chief Financial Officer", "Acme", "2026-04-29", 3),
        stated("Umar Ajiya", "Chief Finance Officer", "Acme Ltd", "2024-01-10", 4),
        stated("Cy Ude", "Chief Executive Officer", "Beta Ltd", "2025-01-01", 5),
    ]  # fmt: skip
    assert roles.organisation_key("the Company") == ""
    assert roles.same_person("Bayo Ojulari", "Bashir Bayo Ojulari")
    assert roles.same_person("Andy Odeh", "Andrew Odeh")
    assert not roles.same_person("Rasheed Ojulari", "Bashir Ojulari")
    assert not roles.same_person("Ada Obi", "Ada Eze") and not roles.same_person(
        "", "x"
    )

    found = roles.latest(said)
    named = roles.naming(found)

    chief = [r for r in found if r["key"] == "group chief executive officer"]
    assert len(chief) == 1  # "the Company" is Acme, the most named organisation
    assert [s["person"] for s in chief[0]["earlier"]] == ["Mo Kye"]
    # Two wordings of one role: only the newest is told.
    assert [found[p]["current"]["person"] for p in named("Who is the CFO?")] == [
        "Ed Ani"
    ]


UNITS = [
    {"name": "Acme Gas Investment Services", "acronym": "AGIS",
     "url": "https://acme.test/agis"},
    {"name": "Acme Trading Limited", "acronym": "ATL",
     "url": "https://acme.test/trading"},
    {"name": "Acme Health Organisation", "acronym": "AHO",
     "url": "https://acme.test/aho"},
    {"name": "Research and Innovation", "acronym": "RAI",
     "url": "https://acme.test/rai"},
]  # fmt: skip


def on_page(statement: dict[str, Any], url: str) -> dict[str, Any]:
    return statement | {"url": url}


def test_each_statement_is_resolved_to_the_entity_it_is_about() -> None:
    said = [
        # One officer, the unit spelled three ways.
        stated("Ben Eze", "Chief Gas Investment Officer", "AGIS", "2024-03-01", 1),
        stated("Ada Obi", "Chief Gas Investment Officer, Acme Gas Investment Services",
               "Acme Gas Investment Services (AGIS)", "2026-10-10", 2),
        stated("Fe Ibe", "Chief Gas Investment Officer", "", "2025-02-02", 3),
        # A unit named only in the title: it leaves the title.
        stated("Cy Ude", "Managing Director, Acme Trading Ltd", "Acme Ltd",
               "2026-01-05", 4),
        # On a unit's own page: nameless, or the company itself (a unit's team read
        # as the parent's).
        on_page(stated("Di Ola", "Company Secretary", "", "2026-10-10", 5),
                "https://acme.test/aho"),
        on_page(stated("Ed Ani", "Managing Director", "Acme Ltd", "2026-10-10", 6),
                "https://acme.test/aho"),
        # A company office heading a unit: the company's other pages hold it.
        on_page(stated("Gu Oko", "Chief Innovation Officer", "Acme Ltd",
                       "2026-10-10", 7), "https://acme.test/rai"),
        stated("Ha Uka", "Chief Innovation Officer", "Acme Ltd", "2023-08-10", 8),
        stated("Ij Ama", "Chief Innovation Officer", "RAI", "2023-11-23", 9),
        stated("Jo Eke", "Chief Executive Officer", "Beta Ltd", "2025-01-01", 10),
        stated("Ko Ani", "Group Chief Executive Officer", "Acme Ltd", "2026-09-29", 11),
    ]  # fmt: skip

    resolved = roles.resolve(said, UNITS)

    agis = "Acme Gas Investment Services (AGIS)"
    aho = "Acme Health Organisation (AHO)"
    assert [s["organisation"] for s in resolved] == [
        agis, agis, agis, "Acme Trading Limited (ATL)", aho, aho, "Acme Ltd",
        "Acme Ltd", "Acme Ltd", "Beta Ltd", "Acme Ltd",
    ]  # fmt: skip
    assert [s["role"] for s in resolved[:4]] == [
        "Chief Gas Investment Officer", "chief gas investment officer",
        "Chief Gas Investment Officer", "managing director",
    ]  # fmt: skip
    found = {(r["key"], r["organisation_key"]): r for r in roles.latest(resolved)}
    officer = found[("chief gas investment officer", "acme gas investment services")]
    assert officer["current"]["person"] == "Ada Obi"
    assert [s["person"] for s in officer["earlier"]] == ["Fe Ibe", "Ben Eze"]
    innovation = found[("chief innovation officer", "acme")]
    assert innovation["current"]["person"] == "Gu Oko"
    assert [s["person"] for s in innovation["earlier"]] == ["Ij Ama", "Ha Uka"]
    assert roles.without_unit("AGIS", UNITS[0]) == "AGIS"  # only the unit: kept


def test_an_organisation_its_passage_does_not_name_is_left_out() -> None:
    filler = stated("Ada Obi", "Minister of State", "not specified", "2026-01-01", 1)
    kept = stated("Ada Obi", "Director", "Acme Ltd", "2026-01-01", 1)

    assert roles.as_named(filler, "Ada Obi, Minister of State, spoke.")[
        "organisation"
    ] == ""  # fmt: skip
    assert roles.as_named(kept, "Ada Obi, Director at ACME, spoke.") is kept
    nameless = roles.latest([roles.as_named(filler, "Ada Obi spoke.")])
    assert "Minister of State (the company): Ada Obi" in roles.described(nameless)


def test_an_acting_holder_holds_the_role_and_names_may_change_order() -> None:
    acting = stated("Ada Obi", "Ag. Chief Upstream Investment Officer", "Acme",
                    "2026-10-10", 1)  # fmt: skip

    assert roles.role_key(acting, {}) == "chief upstream investment officer"
    assert roles.role_key(acting | {"role": "Acting"}, {}) == "acting"
    assert roles.same_person("Audu Ibrahim", "Ibrahim Audu")


def test_a_claim_naming_an_earlier_holder_in_place_of_the_latest_goes() -> None:
    stale = "Ben Eze is the Executive Vice President, Downstream."

    assert roles.earlier_holder(stale, LATEST) == (
        "names an earlier holder of Executive Vice President Downstream "
        "(Acme Limited); the latest statement names Ada Obi"
    )
    succeeded = "Ada Obi succeeded Ben Eze as Executive Vice President, Downstream."
    assert roles.earlier_holder(succeeded, LATEST) == ""
    # The latest holder named elsewhere in the answer: the earlier one is history.
    whole = f"Ada Obi is the Executive Vice President, Downstream. {stale}"
    assert roles.earlier_holder(stale, LATEST, whole) == ""
    assert roles.earlier_holder("Ben Eze spoke at the plant.", LATEST) == ""
    # Another organisation's role counts only where the claim names it.
    assert roles.earlier_holder("Cy Ude is the Chief Innovation Officer.", LATEST) == ""
    assert roles.earlier_holder(
        "Cy Ude is the Chief Innovation Officer of RAI.", LATEST
    ).startswith("names an earlier holder of Chief Innovation Officer")
    chair = roles.latest(
        [
            stated("Mo Kye", "Chairman", "Acme Ltd", "2024-01-01", 1),
            stated("Ada Obi", "Chairman", "Acme Ltd", "2026-01-01", 2),
        ]
    )
    assert roles.earlier_holder("Mo Kye is the Chairman.", chair) == ""  # one word
    assert roles.names_person("Ada Obi spoke", "Ada Obi")
    assert not roles.names_person("Ada spoke", "Ada Obi")
