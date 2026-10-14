"""Business units read from a company's own units page, named in a user's words,
and the passages in each unit's scope (K23). Every page and passage here is
invented."""

from petroknowledge import units

PAGE = "https://acme.test/businesses"
HTML = """<html><body>
<div class="nav"><a href="/about">Who we are</a><a href="/gas-arm">Gas</a></div>
<div class="body">
  <h2>Our Businesses</h2>
  <p>Upstream</p>
  <a href="/acme-gas-marketing-limited">Acme Gas Marketing Limited
     (<span>AGML</span>)</a>
  <a href="https://acme.test/acme-e-and-p">Acme E&amp;P Limited (AEPL)</a>
  <a href="/acme-foundation">Acme Foundation Limited/Gte</a>
  <a href="/acme-foundation">Acme Foundation Limited/Gte</a>
  <p>Acme Shipping Limited (ASL)</p>
  <h2>Learn More</h2>
  <a href="/contact">Contact Us</a>
</div></body></html>"""
ELEMENTS = [
    {"kind": "heading", "text": "Our Businesses", "section": []},
    {"kind": "paragraph", "text": "Upstream", "section": ["Our Businesses"]},
    {"kind": "paragraph", "text": "Acme Gas Marketing Limited (AGML)",
     "section": ["Our Businesses"]},
    {"kind": "paragraph", "text": "Acme E&P Limited (AEPL)",
     "section": ["Our Businesses"]},
    {"kind": "paragraph", "text": "Acme Foundation Limited/Gte",
     "section": ["Our Businesses"]},
    {"kind": "paragraph", "text": "Acme Shipping Limited (ASL)",
     "section": ["Our Businesses"]},
    {"kind": "paragraph", "text": "Contact Us", "section": ["Learn More"]},
]  # fmt: skip
UNITS = units.listed(HTML, PAGE, ELEMENTS, "Our Businesses")


def test_units_are_the_section_s_lines_that_link_to_a_page_of_their_own() -> None:
    assert UNITS == [
        {"name": "Acme Gas Marketing Limited", "acronym": "AGML",
         "url": "https://acme.test/acme-gas-marketing-limited"},
        {"name": "Acme E&P Limited", "acronym": "AEPL",
         "url": "https://acme.test/acme-e-and-p"},
        {"name": "Acme Foundation Limited/Gte", "acronym": "",
         "url": "https://acme.test/acme-foundation"},
    ]  # fmt: skip
    # Left out: navigation ("Gas"), an unlinked line (ASL), another section's link.


def test_names_are_folded_and_compared_without_the_legal_form() -> None:
    assert units.fold("Acme E&P – Gas/Power  Ltd.") == "acme e and p gas power ltd"
    assert units.short_name("Acme Foundation Limited/Gte") == "acme foundation"
    assert units.short_name("Company") == "company"  # one word is kept


def test_a_text_names_a_unit_by_acronym_or_short_name_as_whole_words() -> None:
    named = units.naming(UNITS)

    assert named("Is AGML's mandate gas marketing?") == [0]
    assert named("what does acme e&p do") == [1]
    assert named("ACME FOUNDATION scholarships, and AEPL") == [1, 2]
    assert named("Acme Gas Marketing Limited") == [0]
    assert named("AGMLX or Acme Gas alone") == []  # whole words only
    assert named("Who heads Acme Ltd?") == []  # the parent is not a unit


def test_a_unit_is_named_by_a_former_name_its_own_page_gives() -> None:
    page = ("Acme E&P Limited (AEPL), formerly the Acme Petroleum Development "
            "Company (APDC), produces oil. Acme Oil, formerly Zeta Oil (ZO), is a "
            "partner. AEPL was formerly part of Acme Holdings.")  # fmt: skip

    aliases = units.former_names(page, UNITS[1])
    named = units.naming([UNITS[1] | {"aliases": "; ".join(aliases)}])

    # Zeta Oil is another company's old name; "formerly part of" is no name.
    assert aliases == ["Acme Petroleum Development Company", "APDC"]
    assert named("Does APDC list stimulation?") == [0]
    assert named("Acme Petroleum Development wells") == [0]


def test_a_follow_up_takes_the_units_of_the_latest_message_naming_one() -> None:
    named = units.naming(UNITS)
    history = [("What does AGML do?", "It markets gas."),
               ("And AEPL?", "It produces oil."),
               ("Its start date?", "1990.")]  # fmt: skip

    assert units.asked(named, "Does Acme Foundation give grants?", history) == [2]
    assert units.asked(named, "and its staff?", history) == [1]
    assert units.asked(named, "and its staff?", []) == []


def test_a_passage_is_in_scope_by_its_page_or_by_naming_the_unit() -> None:
    passages = [
        (10, "https://acme.test/acme-e-and-p", "History", "Founded 1990."),
        (11, "https://acme.test/news/1", "AGML news", "Pipeline opened."),
        (12, "https://acme.test/news/2", "Update", "AEPL and Acme Foundation."),
        (13, "https://acme.test/news/3", "Update", "Group results."),
    ]

    assert units.members(UNITS, passages) == [(1, 10), (0, 11), (1, 12), (2, 12)]


def test_units_are_described_by_name_and_acronym() -> None:
    assert units.described(UNITS) == (
        "Acme Gas Marketing Limited (AGML); Acme E&P Limited (AEPL); "
        "Acme Foundation Limited/Gte"
    )
    assert units.described([]) == ""


GROUPED_HTML = """<div class="body"><h2>Our Businesses</h2>
<div class="block"><div>Upstream</div><p>To find oil.</p><p>Safely.</p>
  <div class="options"><a href="/acme-e-and-p">Acme E&amp;P Limited (AEPL)</a>
    <div class="is-disabled">Acme Shipping Limited (ASL)</div></div></div>
<div class="block"><div>Non-Energy</div><p>To build value.</p>
  <div class="options"><a href="/acme-foundation">Acme Foundation Limited/Gte</a>
  </div></div>
<p>Ask us</p></div>"""
GROUPED_LINES = ["Upstream", "To find oil.", "Safely.", "Acme E&P Limited (AEPL)",
                 "Acme Shipping Limited (ASL)", "Non-Energy", "To build value.",
                 "Acme Foundation Limited/Gte", "Ask us"]  # fmt: skip
GROUPED_ELEMENTS = [
    {"kind": "paragraph", "text": line, "section": ["Our Businesses"]}
    for line in GROUPED_LINES
]


def test_a_units_list_is_grouped_by_the_lines_that_lead_its_entries() -> None:
    groups = units.grouped(GROUPED_HTML, GROUPED_ELEMENTS, "Our Businesses")

    assert groups == [
        {"name": "Upstream", "words": ["To find oil.", "Safely."],
         "entries": ["Acme E&P Limited (AEPL)", "Acme Shipping Limited (ASL)"]},
        {"name": "Non-Energy", "words": ["To build value."],
         "entries": ["Acme Foundation Limited/Gte"]},
    ]  # fmt: skip
    # ASL is shown without a link but beside one, so it is an entry; "Ask us" leads
    # no entries, so it is no group.
    assert units.compiled_text("Our Businesses", groups) == "\n".join([
        units.COMPILED,
        "Our Businesses: 3 units in 2 groups.",
        "Upstream, 2 units: Acme E&P Limited (AEPL); Acme Shipping Limited (ASL).",
        "Upstream: To find oil. Safely.",
        "Non-Energy, 1 unit: Acme Foundation Limited/Gte.",
        "Non-Energy: To build value.",
    ])  # fmt: skip


def test_a_text_names_groups_by_their_names_as_whole_words() -> None:
    groups = units.grouped(GROUPED_HTML, GROUPED_ELEMENTS, "Our Businesses")
    names = [group["name"] for group in groups]

    assert units.structure_named("Is upstream bigger than non-energy?", names) == 2
    assert units.structure_named("Upstreaming crude", names) == 0  # whole words
    assert units.structure_named("Any text at all", [""]) == 0  # a nameless group


def test_a_list_without_group_lines_is_one_group_named_by_its_section() -> None:
    groups = units.grouped(HTML, ELEMENTS, "Our Businesses")

    assert groups == [{"name": "", "words": [], "entries": [
        "Upstream", "Acme Gas Marketing Limited (AGML)", "Acme E&P Limited (AEPL)",
        "Acme Foundation Limited/Gte", "Acme Shipping Limited (ASL)"]}]  # fmt: skip
    assert units.compiled_text("Our Businesses", groups).splitlines()[2] == (
        "Our Businesses, 5 units: Upstream; Acme Gas Marketing Limited (AGML); "
        "Acme E&P Limited (AEPL); Acme Foundation Limited/Gte; "
        "Acme Shipping Limited (ASL)."
    )
    assert units.grouped("<p>No links</p>", ELEMENTS, "Our Businesses") == []
