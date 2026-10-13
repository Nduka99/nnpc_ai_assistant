"""Claim verification policy. Every passage here is invented."""

from petroknowledge import prices, units
from petroknowledge.records import Claim, DraftClaim, DraftEvidence, Passage, Rejection
from petroknowledge.verify import (
    check_claim,
    find_quote,
    labels,
    measures,
    quote_link,
    source_link,
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
