"""Claim verification policy. Every passage here is invented."""

from petroknowledge import prices, units
from petroknowledge.records import Claim, DraftClaim, DraftEvidence, Passage, Rejection
from petroknowledge.verify import (
    caption_numbers,
    check_claim,
    find_quote,
    labels,
    measures,
    quote_link,
    source_link,
    states_figure,
    verify,
)

LINE = Passage(
    11,
    "https://acme.test/gas/pipeline",
    "Gas Pipeline",
    "Gas Pipeline",
    "Gas Pipeline\nThe line is buried 3 to 6 m deep.\nIt is a low-pressure line.",
    "2026-10-10T10:00:00+00:00",
)
RULES = Passage(
    12,
    "https://acme.test/hse",
    "Safety",
    "Permits",
    "Permits\nLeaks are reported within one hour.\nThe Petroleum Industry Act "
    "applies to midstream work.",
    "2026-10-10T10:00:00+00:00",
)
SOURCES = {"S1": LINE, "S2": RULES}


def claim(text: str, *evidence: tuple[str, str]) -> DraftClaim:
    return DraftClaim(text, tuple(DraftEvidence(sid, quote) for sid, quote in evidence))


def reason(result: Claim | Rejection) -> str:
    assert isinstance(result, Rejection), result
    return result.reason


def passage(text: str, context: str = "", quotable: bool = True) -> Passage:
    return Passage(20, "u", "t", "h", text, "d", 2, context, quotable=quotable)


def test_a_valid_claim_and_exact_quote_produce_the_expected_citation() -> None:
    result = check_claim(
        claim("The line lies 3 to 6 m deep.", ("S1", "buried 3 to 6 m deep")),
        SOURCES,
    )

    assert isinstance(result, Claim)
    (evidence,) = result.evidence
    assert (evidence.passage_id, evidence.url) == (11, LINE.url)
    assert (evidence.title, evidence.heading) == ("Gas Pipeline", "Gas Pipeline")
    assert evidence.quote == "buried 3 to 6 m deep"
    assert evidence.link == (
        "https://acme.test/gas/pipeline#:~:text=buried%203%20to%206%20m%20deep"
    )
    assert evidence.fetched_at == "2026-10-10T10:00:00+00:00"


def test_quotes_may_differ_only_in_case_whitespace_and_typography() -> None:
    row = 'Gas ("dry") | MMscf/d\n3-6 m | m³ | ﬁeld'

    assert find_quote("THE LINE IS\n buried", LINE.text) == "The line is buried"
    assert find_quote("The line was buried", LINE.text) is None
    assert find_quote("  ", LINE.text) is None
    assert find_quote("Gas (“dry”)", row) == 'Gas ("dry")'
    assert find_quote("3–6 m", row) == "3-6 m"
    assert find_quote("m3", row) == "m³"
    assert find_quote("field", row) == "ﬁeld"
    # Invisible format characters are never copied into a quote (K30).
    team = "Ada Obi​\nManaging Director"
    assert find_quote("Ada Obi\nManaging Director", team) == team


def test_a_cell_mark_may_be_quoted_as_punctuation_between_words() -> None:
    row = "3 | NNPC Energy Services Limited | EnServ | 4 | 6"

    assert find_quote("NNPC Energy Services Limited; EnServ", row) == (
        "NNPC Energy Services Limited | EnServ"
    )
    assert find_quote("Limited: EnServ, 4", row) == "Limited | EnServ | 4"
    assert find_quote("4-6", row) is None  # two cells are not a range
    assert find_quote("4, 6", row) is None  # nor a number
    assert find_quote("4:6", row) is None  # nor a time
    assert find_quote("; EnServ", row) is None  # only between two words


def test_a_mark_may_be_added_or_left_out_between_words_never_digits() -> None:
    page = (
        "Gas & Power: To grow gas sales.\nAcme Gas Limited (AGL) is a unit, "
        "formed on May 16, 2016 to sell gas.\nAcme Trading Limited\nRoute 4 6"
    )

    assert find_quote("Gas & Power To grow gas", page) == "Gas & Power: To grow gas"
    assert find_quote("Acme Gas Limited (AGL), is a unit", page) == (
        "Acme Gas Limited (AGL) is a unit"
    )
    assert find_quote("Acme Trading Limited.", page) == "Acme Trading Limited"
    assert find_quote("sales Acme Gas", page) == "sales.\nAcme Gas"
    assert find_quote("Route 4, 6", page) is None  # a mark between digits is a number
    assert find_quote("May 16 2016", page) is None
    assert find_quote(". Acme Gas", page) is None  # never a quote's first mark


def test_a_quote_keeps_word_edges_spacing_and_whole_numbers() -> None:
    text = "Laid in trenches of 3 to 60 m. Gas | 4,223 | MMscf/d | 1.5 bn"

    assert find_quote("trenches of 3 to 6", text) is None
    assert find_quote("renches", text) is None
    assert find_quote("trenches of 3 to 60 m.", text) == "trenches of 3 to 60 m."
    assert find_quote("Gas 4,223 MMscf/d", text) == "Gas | 4,223 | MMscf/d"
    assert find_quote("4, 223", text) is None
    assert find_quote("1.5bn", text) == "1.5 bn"  # a digit meeting a letter


def test_links_highlight_the_quote_open_pdf_pages_and_escape_addresses() -> None:
    url = "https://acme.test/Reports/annual report.pdf?v=1 2"
    done = "https://acme.test/Reports/annual%20report.pdf?v=1%202"

    assert quote_link("https://acme.test/p", "low-pressure, 3  & dry") == (
        "https://acme.test/p#:~:text=low%2Dpressure%2C%203%20%26%20dry"
    )
    assert quote_link("https://acme.test/p", "Route\n\nAbuja, Kaduna-Kano") == (
        "https://acme.test/p#:~:text=Route,Abuja%2C%20Kaduna%2DKano"
    )
    assert quote_link(url, "gas", 3) == f"{done}#page=3"
    assert quote_link(done, "gas", 3) == f"{done}#page=3"  # escapes kept as they are


def test_a_picture_opens_its_page_where_it_is_shown() -> None:
    post = Passage(1, "https://acme.test/news", "N", "Picture", "HSE week", "",
                   image="p")  # fmt: skip
    figure = Passage(2, "https://acme.test/a b.pdf", "A", "Picture", "", "", 4,
                     image="q")  # fmt: skip

    assert source_link(post, "HSE week") == "https://acme.test/news"
    assert source_link(figure, "") == "https://acme.test/a%20b.pdf#page=4"


def test_a_compiled_list_opens_its_page_without_a_highlight() -> None:
    listed = Passage(3, "https://acme.test/our businesses", "B", units.COMPILED,
                     f"{units.COMPILED}\nGas, 1 unit: AGL.", "")  # fmt: skip

    assert source_link(listed, "Gas, 1 unit: AGL.") == (
        "https://acme.test/our%20businesses"
    )


def test_unknown_sources_missing_quotes_and_bare_claims_are_rejected() -> None:
    unknown = check_claim(claim("It is low pressure.", ("S9", "low-pressure")), SOURCES)
    missing = check_claim(claim("It is high pressure.", ("S1", "high-pressure")),
                          SOURCES)  # fmt: skip
    one_bad = check_claim(
        claim("It is buried and new.", ("S1", "The line is buried"),
              ("S1", "The line is new")),
        SOURCES,
    )  # fmt: skip

    assert reason(unknown) == "unknown source S9"
    assert reason(missing) == "quote not in S1: 'high-pressure'"
    assert reason(one_bad).startswith("quote not in S1")
    assert reason(check_claim(claim("It is safe."), SOURCES)) == "no quote"


def test_a_quote_cited_to_the_wrong_passage_is_cited_to_the_one_holding_it() -> None:
    result = check_claim(claim("It is low pressure.", ("S2", "low-pressure line")),
                         SOURCES)  # fmt: skip

    assert isinstance(result, Claim)
    assert result.evidence[0].passage_id == LINE.id


def test_a_scanned_page_is_never_quoted_even_when_it_holds_the_words() -> None:
    scan = passage("Revenue rose to N5.4 trillion.", quotable=False)
    typed = passage("The board approved the accounts.")
    sources = {"S1": scan, "S2": typed}

    refused = check_claim(claim("Revenue rose.", ("S1", "Revenue rose")), sources)
    elsewhere = check_claim(
        claim("The accounts were approved.", ("S1", "board approved the accounts")),
        sources,
    )

    assert reason(refused) == "quote from search-only S1 (a scan)"
    assert isinstance(elsewhere, Claim)  # the typed passage holds it
    assert elsewhere.evidence[0].quote == "board approved the accounts"


def test_numbers_must_be_quoted_with_their_sign_or_named_by_the_passage() -> None:
    text = "OML 118 lies offshore. Output fell by −5% and rose by +8%."
    field = Passage(14, "u", "OML118 Bonga", "Field", text, "d")
    sources = {"S1": field}

    def result(text: str, quote: str = "lies offshore") -> Claim | Rejection:
        return check_claim(claim(text, ("S1", quote)), sources)

    assert isinstance(result("OML118 lies offshore."), Claim)  # its title names it
    assert isinstance(result("OML 118 lies offshore."), Claim)  # its text names it
    assert reason(result("OML 119 lies offshore.")) == "number not in its quotes: 119"
    assert reason(result("Block 118B lies offshore.")) == (
        "number not in its quotes: 118"  # "118B" is not the passage's name
    )
    assert reason(result("Output fell by 4%.")) == "number not in its quotes: 4"
    assert reason(result("Output fell by -8%.", "fell by −5%")) == (
        "number not in its quotes: -8"
    )
    assert isinstance(result("Output fell by -5%.", "fell by −5%"), Claim)
    assert isinstance(result("It lies 3-6 m deep.", "3 to 6 m"), Rejection)
    ranged = check_claim(claim("It is buried 3-6 m deep.", ("S1", "3 to 6 m deep")),
                         SOURCES)  # fmt: skip
    assert isinstance(ranged, Claim)  # a dash between numbers is a range


def test_an_amount_keeps_its_currency_scale_and_unit() -> None:
    report = passage("Profit after tax was N5.4 trillion; gas sales averaged "
                     "4,223 mmscf/d and crude output 1.73 mmbopd.")  # fmt: skip
    sources = {"S1": report}

    def result(text: str) -> Claim | Rejection:
        quote = ("S1", report.text)
        return check_claim(claim(text, quote), sources)

    assert isinstance(result("Profit after tax was ₦5.4 trillion."), Claim)
    assert isinstance(result("Profit was 5.4 trillion naira."), Claim)
    assert isinstance(result("Output was 1.73 million barrels per day."), Claim)
    assert isinstance(result("Gas sales were 4,223 MMscf/d."), Claim)
    assert reason(result("Profit was ₦5.4 billion.")) == (
        "amount not as quoted: ₦5.4 billion"
    )
    assert reason(result("Profit was $5.4 trillion.")) == (
        "amount not as quoted: $5.4 trillion"
    )
    assert reason(result("Gas sales were 4,223 bcf/d.")) == (
        "amount not as quoted: 4,223 bcf/d"
    )
    assert reason(result("Output was 1.73 kbopd.")) == (
        "amount not as quoted: 1.73 kbopd"
    )


def test_a_bare_table_cell_takes_its_amount_from_the_column_headers() -> None:
    row = passage("Revenue | 2,774 | 2,548", context="Results › NGN Billion | 2025")
    sources = {"S1": row}

    cell = ("S1", "Revenue | 2,774")
    named = check_claim(claim("Revenue was ₦2,774 billion.", cell), sources)
    wrong = check_claim(claim("Revenue was ₦2,774 trillion.", cell), sources)

    assert isinstance(named, Claim)
    assert reason(wrong) == "amount not as quoted: ₦2,774 trillion"


def test_a_table_row_takes_the_numbers_its_caption_states() -> None:
    table = passage(
        "Overview\nTable 1. Energy overview, 2023\n | Gas | Total\n"
        "Generation (TWh) | 32.8 | 42.5\nPlain line, 2024"
    )  # fmt: skip
    sources = {"S1": table}
    row = ("S1", "Generation (TWh) | 32.8 | 42.5")

    captioned = check_claim(claim("Generation in 2023 was 42.5 TWh.", row), sources)
    other = check_claim(claim("Generation in 2022 was 42.5 TWh.", row), sources)
    prose = check_claim(claim("In 2024 it changed.", ("S1", "Plain line")), sources)

    assert isinstance(captioned, Claim)
    assert reason(other) == "number not in its quotes: 2022"
    assert reason(prose) == "number not in its quotes: 2024"
    assert caption_numbers("Generation (TWh)", table) == set()
    assert caption_numbers("No | such row", table) == set()
    assert caption_numbers("A | B", passage("A | B")) == set()


def test_amounts_and_labels_are_read_in_nnpc_s_forms() -> None:
    found = measures("US$3billion, N65 a litre, 614 km, 3 m deep, $2b and 15 kbpd")

    assert [(m["number"], m["currency"], m["scale"], m["unit"]) for m in found] == [
        ("3", "dollar", "billion", None),
        ("65", "naira", None, None),
        ("614", None, None, None),
        ("3", None, None, None),  # three metres, not three million
        ("2", "dollar", "billion", None),
        ("15", None, "thousand", "oil per day"),
    ]
    assert labels("Revenue (N billion) | Output (barrels per day) | m | NNPC") == {
        ("currency", "naira"), ("scale", "billion"), ("unit", "oil per day"),
    }  # fmt: skip
    assert labels("Gas (MMscf/d) | US$ | k") == {
        ("unit", "gas per day"), ("scale", "million"), ("currency", "dollar"),
    }  # fmt: skip


def test_the_requests_names_for_passages_are_not_part_of_a_claim() -> None:
    labelled = check_claim(
        claim("The line is buried 3 to 6 m deep (S1, S2).", ("S1", "3 to 6 m deep")),
        SOURCES,
    )
    line = ("S1", "It is a low-pressure line.")
    bare = check_claim(claim("According to S1, it is a low-pressure line.", line),
                       SOURCES)  # fmt: skip
    unknown = check_claim(claim("It is low pressure (S7).", ("S1", "low-pressure")),
                          SOURCES)  # fmt: skip

    assert isinstance(labelled, Claim)
    assert labelled.text == "The line is buried 3 to 6 m deep."
    assert isinstance(bare, Claim)
    assert bare.text == "According to Gas Pipeline, it is a low-pressure line."
    assert reason(unknown) == "number not in its quotes: 7"  # not one of the request's


def test_a_number_in_the_cited_sources_title_or_stated_date_counts() -> None:
    release = Passage(30, "u", "AGM 2023: Acme results", "Results",
                      "Gross profit rose 145 per cent.", "d",
                      published="2026-09-29")  # fmt: skip
    sources = {"S1": release}

    def result(text: str) -> Claim | Rejection:
        return check_claim(claim(text, ("S1", "Gross profit rose 145 per cent.")),
                           sources)  # fmt: skip

    assert isinstance(result("The 2023 AGM release says gross profit rose 145 per "
                             "cent."), Claim)  # fmt: skip
    assert isinstance(result("The September 2026 release says gross profit rose 145 "
                             "per cent."), Claim)  # fmt: skip
    assert isinstance(result("The 29 Sept. 2026 release says gross profit rose 145 "
                             "per cent."), Claim)  # fmt: skip
    # The source's own year, standing alone (K32); a figure beside it is the
    # grounding check's to judge.
    assert isinstance(result("In 2026, gross profit rose 145 per cent."), Claim)
    assert reason(result("In 2025, gross profit rose 145 per cent.")) == (
        "number not in its quotes: 2025"  # another year than the source's
    )
    assert reason(result("The 30 September 2026 release says gross profit rose 145 "
                         "per cent.")) == "number not in its quotes: 30"  # fmt: skip


def test_a_figure_is_a_number_other_than_a_date_or_a_year() -> None:
    assert states_figure("Production averaged 1.71 million barrels per day.")
    assert states_figure("The group has 21 units in 5 groups.")
    assert states_figure("In 2025, NNPC employed 1,023 full-time employees.")
    assert not states_figure("The current GCEO is Bayo Ojulari.")
    assert not states_figure("A post from July 7, 2023, names the managing director.")
    assert not states_figure("Appointed on 22nd December, 2023 and 1 May 2024.")
    assert not states_figure(
        "The 29 Sept. 2026 release names him; Oredo began in 1996."
    )
    assert states_figure("The Mayor said 3 vessels sail.")  # no month: a figure


def test_a_claim_may_give_its_source_s_date_as_the_request_shows_it() -> None:
    post = Passage(31, "u", "Record output", "", "NEPL produced 365,000 barrels.",
                   "d", published="2026-07-07")  # fmt: skip
    quote = ("S1", "NEPL produced 365,000 barrels.")

    shown = check_claim(claim("A post from 2026-07-07 says NEPL produced 365,000 "
                              "barrels.", quote), {"S1": post})  # fmt: skip
    other = check_claim(claim("A post from 2026-07-08 says NEPL produced 365,000 "
                              "barrels.", quote), {"S1": post})  # fmt: skip

    assert isinstance(shown, Claim)
    assert reason(other) == "number not in its quotes: 07, 08"  # not its date


def test_a_regulation_licence_or_approval_must_be_in_the_quotes() -> None:
    quote = ("S1", "The line is buried 3 to 6 m deep.")

    verdict = check_claim(claim("The line complies with the PIA.", quote), SOURCES)
    licence = check_claim(claim("NUPRC licensed the line.", quote), SOURCES)
    described = check_claim(
        claim("Midstream work falls under the Petroleum Industry Act.",
              ("S2", "The Petroleum Industry Act applies to midstream work.")),
        SOURCES,
    )  # fmt: skip

    assert reason(verdict) == (
        "regulation not in its quotes: compliance, petroleum industry act"
    )
    assert reason(licence) == "regulation not in its quotes: licence"
    assert isinstance(described, Claim)


def test_a_named_regulation_needs_that_name_not_another() -> None:
    gazette = passage("The Upstream Petroleum Safety Regulations, 2024 apply "
                      "offshore.")  # fmt: skip
    sources = {"S1": gazette}
    quote = ("S1", gazette.text)

    named = "The Upstream Petroleum Safety Regulations apply offshore."
    another = "The Upstream Petroleum Measurement Regulations apply."
    same = check_claim(claim(named, quote), sources)
    other = check_claim(claim(another, quote), sources)

    assert isinstance(same, Claim)
    assert reason(other) == (
        "regulation not in its quotes: upstream petroleum measurement regulations"
    )


def test_a_source_that_is_a_regulation_names_it_by_its_title() -> None:
    act = Passage(31, "u", "Petroleum Industry Act 2021", "Host communities",
                  "Settlors pay 3 per cent of operating expenditure.", "d")  # fmt: skip
    quote = ("S1", act.text)

    under = check_claim(claim("Under the PIA, settlors pay 3 per cent of operating "
                              "expenditure.", quote), {"S1": act})  # fmt: skip
    other = check_claim(claim("Under the Upstream Safety Regulations, settlors pay 3 "
                              "per cent.", quote), {"S1": act})  # fmt: skip

    assert isinstance(under, Claim)
    assert reason(other) == (
        "regulation not in its quotes: regulations, upstream safety regulations"
    )


def test_a_per_unit_price_is_never_stated_but_a_total_is() -> None:
    for text in ["₦945 per litre", "N617 a litre", "US$75/bbl", "$3 per MMBtu",
                 "N500 per copy"]:  # fmt: skip
        assert prices.states_price(text), text
    for text in ["₦5.4 trillion profit", "₦2 billion per year", "$500,000 a day",
                 "N2 trillion a year", "614 km per line", "the ₦ sign"]:  # fmt: skip
        assert not prices.states_price(text), text
    pump = passage("The pump price is ₦945 per litre in Lagos.")

    result = check_claim(claim("Petrol costs ₦945 per litre.", ("S1", pump.text)),
                         {"S1": pump})  # fmt: skip

    assert reason(result) == "states a per-unit price"


def test_verify_keeps_valid_claims_and_records_each_removal() -> None:
    good = claim("The line is low pressure.", ("S1", "It is a low-pressure line."))
    bad = claim("The line is 9 m deep.", ("S1", "It is a low-pressure line."))
    both = claim(
        "Leaks are reported within one hour on this low-pressure line.",
        ("S2", "Leaks are reported within one hour."),
        ("S1", "It is a low-pressure line."),
    )

    claims, rejected = verify([good, bad, both], SOURCES)

    assert [c.text for c in claims] == [good.text, both.text]
    assert [e.passage_id for e in claims[1].evidence] == [12, 11]
    assert rejected == (Rejection(bad.text, "number not in its quotes: 9"),)
