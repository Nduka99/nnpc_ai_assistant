"""Check each claim the model returned against the passages it cites (Lime Green's
verify, carried over for NNPC).

A claim is shown only if every check passes:

1. it has at least one quote, and every source id names a supplied passage;
2. every quote appears in a passage that may be quoted, ignoring only case,
   whitespace and a punctuation mark added or left out between words
   (`find_quote`); a quote the model cited to the wrong passage is cited to the supplied
   passage holding it. A scanned page's transcript is searched but never quoted
   (K12), so a quote found only there is refused;
3. every number in the claim, with its sign, appears in one of its quotes, or, if
   unsigned, in a cited passage's title, heading or text together with the token it
   is part of ("OML118", "2,548.0bn") or the word before it ("OML 118"), or in the
   cited source's own title or heading, or in its publication date: its year
   wherever the claim states that year (K32), its day when the claim states the
   date with its month (K29 F4). The request's names for its passages
   ("S1") are not part of the claim: they are removed before checking and showing;
4. every amount keeps what it measures: a currency, scale or unit the claim gives a
   number ("₦5.4 trillion", "1.73 mmbopd", "4,223 mmscf/d") is given to that number
   in its quotes, or, for a number quoted bare from a table, named by the cited
   passage's column headers, so the digits cannot be kept while the amount changes;
5. every regulation, licence, approval or compliance term the claim uses, and every
   named regulation ("Upstream Petroleum Measurement Regulations"), is in its quotes
   or in the title or heading of the cited source, which may be that regulation;
6. it states no per-unit price, whatever it quotes (`prices`).

These are evidence checks: none of them tries to judge what a sentence means, so
they apply equally to any wording. A failing claim is removed whole and recorded
with its reason; nothing is repaired or regenerated.
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any
from urllib.parse import quote as percent_encode
from urllib.parse import urlsplit, urlunsplit

from petroknowledge import prices, units
from petroknowledge.records import Claim, DraftClaim, Evidence, Passage, Rejection

# A sign belongs to a number only where it cannot be a range dash: "-5 °C" has a
# sign, "3-6 mm" is a range. Unicode minus and en dash are read as a minus.
NUMBER = re.compile(r"(?<![\w.])[-+−–]?[0-9]+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?")

# Rules, licences and approvals, each counted under one name so that "complies" in a
# claim is matched by "comply" in its quote.
REGULATION_TERMS = {
    "petroleum industry act": r"\bpetroleum\s+industry\s+act\b|\bPIA\b",
    "regulations": r"\bregulations?\b|\bregs\b",
    "licence": r"\blicen[cs](?:e|es|ed|ing)\b",
    "lease": r"\bleases?\b",
    "permit": r"\bpermit(?:s|ted)?\b",
    "approval": r"\bapprov(?:e|es|ed|ing|al|als)\b",
    "compliance": r"\bcompl(?:y|ies|ied|ying|iant|iance)\b",
}
# A named regulation: the capitalised words before "Regulations", without a
# sentence's opening "The".
REGULATION_NAME = re.compile(r"((?:(?!The\b)[A-Z][\w'’-]*\s+){1,6})Regulations?\b")
# The request's own names for its passages, as a model writes them into a claim: a
# bracketed group ("(S1)", "[S2, S3]") or a bare one ("According to S8").
LABEL_GROUP = re.compile(r"\s*[(\[]\s*S\d+(?:\s*(?:,|;|and)\s*S\d+)*\s*[)\]]")
LABEL = re.compile(r"\bS\d+\b")
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December")  # fmt: skip
# A number standing alone: not part of a word or name ("2026", not "OML118" or
# "118B").
STANDALONE = re.compile(r"(?<![\w.])[0-9]+(?:\.[0-9]+)?(?![\w.]?\w)")
# A date ("July 7, 2023", "7 July 2023", "22nd December, 2023", "29 Sept. 2026") and a
# year standing alone: numbers that are no figure (K32, `states_figure`).
MONTH = "(?:" + "|".join(["Sept", *(f"{m[:3]}(?:{m[3:]})?" for m in MONTHS)]) + ")"
DATE = re.compile(
    rf"\b(?:\d{{1,2}}(?:st|nd|rd|th)?\s+(?:of\s+)?)?{MONTH}(?![a-z])\.?"
    rf"(?:\s+\d{{1,2}}(?:st|nd|rd|th)?\b,?)?(?:\s+\d{{4}})?"
)
YEAR = re.compile(r"(?<![\w.])(?:19|20)\d{2}(?![\w.]?\w)")

# What a number measures, each form folded to one name. Forms as NNPC writes them
# (measured on index version 1's passages).
CURRENCIES = {
    "₦": "naira", "n": "naira", "ngn": "naira", "naira": "naira",
    "$": "dollar", "us$": "dollar", "usd": "dollar", "dollar": "dollar",
    "dollars": "dollar", "£": "pound", "gbp": "pound", "€": "euro", "eur": "euro",
}  # fmt: skip
SCALES = {
    "thousand": "thousand", "k": "thousand", "million": "million", "mn": "million",
    "mln": "million", "m": "million", "billion": "billion", "bn": "billion",
    "b": "billion", "trillion": "trillion", "tn": "trillion", "trn": "trillion",
    "tr": "trillion",
}  # fmt: skip
SHORT_SCALES = {"k", "m", "b"}  # a scale only after a currency: "$3b", not "3 m"
# A unit: what it measures and the scale it carries ("kbopd": thousand barrels a day).
UNITS: dict[str, tuple[str, str | None]] = {
    "bpd": ("oil per day", None), "b/d": ("oil per day", None),
    "bbl/d": ("oil per day", None), "bopd": ("oil per day", None),
    "barrels per day": ("oil per day", None),
    "barrels of oil per day": ("oil per day", None),
    "kbpd": ("oil per day", "thousand"), "kbopd": ("oil per day", "thousand"),
    "kb/d": ("oil per day", "thousand"), "mmbpd": ("oil per day", "million"),
    "mmbopd": ("oil per day", "million"), "mmb/d": ("oil per day", "million"),
    "mbpd": ("oil per day, mbpd", None),  # thousand or million by writer: itself only
    "scf/d": ("gas per day", None), "mmscf/d": ("gas per day", "million"),
    "mmscfd": ("gas per day", "million"), "bcf/d": ("gas per day", "billion"),
    "bscf/d": ("gas per day", "billion"), "scf": ("gas", None),
    "mmscf": ("gas", "million"), "bcf": ("gas", "billion"), "bscf": ("gas", "billion"),
    "tcf": ("gas", "trillion"),
}  # fmt: skip


def alternation(words: Sequence[str]) -> str:
    """A regular-expression choice of the words, longest first."""
    return "|".join(re.escape(word) for word in sorted(words, key=len, reverse=True))


MEASURE = re.compile(
    rf"(?P<currency>₦|US\$|\$|£|€|\b(?:NGN|USD|GBP|EUR|N))?\s?"
    rf"(?P<number>[0-9][0-9,]*(?:\.[0-9]+)?)"
    rf"(?:\s?(?P<scale>{alternation(list(SCALES))})(?![\w/]))?"
    rf"(?:\s?(?P<unit>{alternation(list(UNITS))})(?![\w/]))?"
    rf"(?:\s(?P<named>naira|dollars?)\b)?",
    re.IGNORECASE,
)
# The gap between two quoted numbers: whitespace, which may hold the " | " this
# project puts between a table's cells (Lime Green X9).
NUMBER_GAP = r"\s+(?:\|\s+)*"
# Between two quoted words the passage may also hold a mark the quote leaves out
# ("Gas & Power: To unlock" quoted as "Gas & Power To unlock", K29 F5).
WORD_GAP = r"[,;:.]?" + NUMBER_GAP
# Where a digit meets a non-digit, or at punctuation, whitespace is optional: Docling
# spaces superscripts that the page and its text layer do not ("N/mm 2"). A mark the
# quote leaves out may stand there too, when spacing follows it.
OPTIONAL_GAP = r"\s*(?:[,;:.](?=\s)\s*)?(?:\|\s+)*"
# Marks a model writes where a table's cell mark stands ("Limited; EnServ" for
# "Limited | EnServ", K27c). Never a hyphen or slash, and never between digits:
# "4 | 6" is not the range "4-6", the number "4, 6" or the time "4:6".
CELL_PUNCTUATION = {";", ",", ":"}
# Marks a model adds or drops between words or at a quote's end ("(NGML), is" for
# "(NGML) is", "Limited." for a line ending "Limited"; K29 F5): they never change
# which words are quoted. Never between digits, where they make a number.
SOFT_MARKS = {",", ";", ":", "."}
# Typographic forms of one character; the page's curly quotes, which Docling writes
# straight, are the same quote marks.
TYPOGRAPHY = str.maketrans(
    dict.fromkeys("‘’‚′", "'")
    | dict.fromkeys("“”„″", '"')
    | dict.fromkeys("–—−‐‑", "-")
)
# The units a quote is matched by: a number, with its decimal point or comma, is one
# unit ("1.5" never matches "1. 5"); any other character is its own unit.
UNIT = re.compile(r"\d+(?:[.,]\d+)*|\S")


@lru_cache(maxsize=512)  # about 25 MB at most: passages are up to 1,500 characters
def folded(text: str) -> tuple[str, tuple[int, ...]]:
    """The text with typographic and Unicode compatibility forms folded (NFKC: "²"
    as "2", "ﬁ" as "fi"), and each folded character's position in the text.
    Invisible format characters (zero-width spaces, soft hyphens) are left out: a
    copied quote never holds them (K30: 10 passages, one lost role statement).
    Remembered, as the same passages are checked again and again."""
    chars = []
    where = []
    for index, char in enumerate(text):
        if unicodedata.category(char) == "Cf":
            continue
        form = unicodedata.normalize("NFKC", char.translate(TYPOGRAPHY))
        for piece in form.translate(TYPOGRAPHY):
            chars.append(piece)
            where.append(index)
    return "".join(chars), tuple(where)


def same_kind(before: str, after: str) -> bool:
    """Whether two neighbouring units are letters, or digits, where spacing is
    part of the words ("therapist" is not "the rapist", nor "36" "3 6")."""
    letters = before[-1].isalpha() and after[0].isalpha()
    return letters or (before[-1].isdigit() and after[0].isdigit())


def quote_pattern(quote: str) -> str:
    """The pattern a quote is found by (Lime Green X9, "Follow-up: quote matching")."""
    units = list(UNIT.finditer(folded(quote)[0]))
    # A point or comma between two digits belongs to a number: spacing around it
    # stays as the quote has it ("1, 2" is not "1,2").
    decimal = set()
    for index in range(1, len(units) - 1):
        before, mark, after = (unit.group() for unit in units[index - 1 : index + 2])
        if mark in ".," and before[-1].isdigit() and after[0].isdigit():
            decimal |= {index - 1, index}
    parts = []
    for index, unit in enumerate(units):
        if index:
            previous = units[index - 1].group()
            spaced = unit.start() > units[index - 1].end()
            numeric = previous[-1].isdigit() and unit.group()[0].isdigit()
            if index - 1 in decimal or numeric:
                parts.append(NUMBER_GAP if spaced else "")
            elif same_kind(previous, unit.group()):
                parts.append(WORD_GAP if spaced else "")
            else:
                parts.append(OPTIONAL_GAP)
        parts.append(unit_pattern(units, index))
    return r"(?<!\w)" + "".join(parts) + r"(?!\w)"


def unit_pattern(units: Sequence[re.Match[str]], index: int) -> str:
    """The pattern one quoted unit is found by: itself, or, for a mark between two
    units that are not both digits or at the quote's end, the mark or nothing
    (`SOFT_MARKS`); a semicolon, comma or colon between them may also be a table's
    cell mark."""
    text = units[index].group()
    inside = 0 < index < len(units) - 1
    last = 0 < index == len(units) - 1
    if text not in SOFT_MARKS or not (inside or last):
        return re.escape(text)
    if inside:
        before, after = units[index - 1].group(), units[index + 1].group()
        if before[-1].isdigit() and after[0].isdigit():
            return re.escape(text)
    cell = "|\\|" if inside and text in CELL_PUNCTUATION else ""
    return f"(?:{re.escape(text)}{cell})?"


def find_quote(quote: str, text: str) -> str | None:
    """The passage's own wording of `quote`, or None if the passage does not contain it.

    Case, typographic forms and Unicode compatibility forms may differ, and so may
    whitespace where a model re-wraps lines or where a digit meets a non-digit
    (superscripts); between letters, or digits, spacing must be the quote's. The
    table cell mark reads as whitespace, or as a semicolon, comma or colon between
    words (K27c). A comma, semicolon, colon or full stop may be added or left out
    between words and at the quote's end, never between digits (K29 F5). The quote
    must start and end at word edges, so "3 to 6" cannot match "3 to 60 mm".
    """
    if not quote.split():
        return None
    text_folded, where = folded(text)
    match = re.search(quote_pattern(quote), text_folded, re.IGNORECASE)
    if match is None:
        return None
    return text[where[match.start()] : where[match.end() - 1] + 1]


def encoded(url: str) -> str:
    """The address with its path and query percent-encoded (RFC 3986), escapes it
    already has kept: a path with spaces is cut at the first space wherever text is
    turned into links."""
    parts = urlsplit(url)
    path = percent_encode(parts.path, safe="/%!$&'()*+,;=:@-._~")
    query = percent_encode(parts.query, safe="/?%!$&'()*+,;=:@-._~")
    return urlunsplit((parts.scheme, parts.netloc, path, query, parts.fragment))


def directive_text(text: str) -> str:
    """Words for a text directive: percent-encoded, '-' included, as it is syntax."""
    return percent_encode(" ".join(text.split()), safe="").replace("-", "%2D")


def quote_link(url: str, quote: str, page: int | None = None) -> str:
    """A link that opens the page with the quote highlighted (a URL text fragment),
    or a PDF at the quote's page (RFC 8118 `page=`: browsers do not find text
    fragments in PDFs). A text directive matches inside one block of the page, so a
    quote whose wording spans lines of its passage is linked as a range from its
    first line to its last."""
    address = encoded(url)
    if page is not None:
        return f"{address}#page={page}"
    lines = [line for line in quote.split("\n") if line.strip()]
    if len(lines) > 1:
        start, end = directive_text(lines[0]), directive_text(lines[-1])
        return f"{address}#:~:text={start},{end}"
    return f"{address}#:~:text={directive_text(quote)}"


def source_link(passage: Passage, quote: str) -> str:
    """Where a citation opens: the quote highlighted on its page, except for a
    picture, whose words are not the page's text, so its page opens where it is
    shown (Lime Green X44 G6), and a compiled list, whose sentences are not on the
    page, so the page itself opens (K27)."""
    if passage.heading == units.COMPILED:
        return encoded(passage.url)
    if passage.image:
        page = f"#page={passage.page}" if passage.page is not None else ""
        return encoded(passage.url) + page
    return quote_link(passage.url, quote, passage.page)


def numbers(text: str) -> set[str]:
    """Every number in the text, preserving an explicit plus or minus sign."""
    found = NUMBER.findall(text)
    return {n.replace("−", "-").replace("–", "-") for n in found}


def states_figure(text: str) -> bool:
    """Whether a claim states a figure: a number other than a date's day or a year
    ("averaged 1.71 million barrels per day", "21 units"), where a substituted
    period, measure or entity is the error the grounding check catches (K32)."""
    return bool(numbers(YEAR.sub(" ", DATE.sub(" ", text))))


def number_units(claim: str, number: str) -> list[str]:
    """What an unsigned number in a claim is found with in a passage: each token it is
    part of ("OML118", "25kg"), or, where it stands alone, the word before it and the
    number ("OML 118"). A signed number has none: only its quotes can support it."""
    if not number[0].isdigit():
        return []
    units = []
    for match in re.finditer(rf"(?<![\d.]){re.escape(number)}(?!\.?\d)", claim):
        start, end = match.span()
        while start > 0 and claim[start - 1].isalnum():
            start -= 1
        while end < len(claim) and claim[end].isalnum():
            end += 1
        if (start, end) != match.span():
            units.append(claim[start:end])
            continue
        before = re.search(r"(\w+)\W*$", claim[:start])
        if before:
            units.append(claim[before.start(1) : end])
    return units


def without_labels(text: str, sources: Mapping[str, Passage]) -> str:
    """The claim without the request's own names for its passages ("S1"), which a
    model sometimes writes into a claim (K29 F4): a bracketed group of them is
    removed, a bare one is replaced by its passage's title."""

    def bracketed(match: re.Match[str]) -> str:
        names = re.findall(r"S\d+", match.group())
        return "" if all(name in sources for name in names) else match.group()

    def bare(match: re.Match[str]) -> str:
        passage = sources.get(match.group())
        return passage.title if passage is not None else match.group()

    return LABEL.sub(bare, LABEL_GROUP.sub(bracketed, text)).strip()


def source_numbers(claim: str, passages: Sequence[Passage]) -> set[str]:
    """The numbers a cited passage's source names (K29 F4): a number standing alone
    in its title or heading and in the claim ("the 2026 licensing amendment"; never
    digits inside a name such as "OML118"), its publication year wherever the claim
    states it ("at the 2023 commissioning", K32), its whole date as the request shows
    it ("2026-07-07", K32 B2b), and its publication day when the claim states the date
    with its month ("the 29 September 2026 release"). A year
    that is instead a figure's period ("in 2023, revenue was ₦5 trillion" from a 2023
    post about 2022) comes with a figure, which the grounding check judges."""
    named = "\n".join(f"{p.title}\n{p.heading}" for p in passages)
    standing = set(STANDALONE.findall(claim))
    found = set(STANDALONE.findall(named)) & standing
    for passage in passages:
        if not passage.published:
            continue
        year, month, day = passage.published[:10].split("-")
        if year in standing:
            found.add(year)
        # The date as the request shows it beside the passage ("2026-07-07", K32 B2b).
        if passage.published[:10] in claim:
            found |= {year, month, day}
        stated = re.search(
            rf"\b(?:(\d{{1,2}})\s+)?{MONTHS[int(month) - 1][:3]}[a-z]*\.?\s+"
            rf"(?:(\d{{1,2}}),?\s+)?{year}\b",
            claim,
            re.IGNORECASE,
        )
        if stated and str(int(day)) in (stated[1], stated[2]):
            found.add(str(int(day)))
    return found


def caption_numbers(quote: str, passage: Passage) -> set[str]:
    """The numbers in the caption of the table a quoted row comes from ("Table 1.
    Nigeria's energy overview, 2023"): the line just above the table's rows, which
    applies to every row, as its headers do (K20, K31). A quote that is not a table
    row has none."""
    if " | " not in quote:
        return set()
    lines = passage.text.split("\n")
    first = quote.split("\n")[0]
    at = next((n for n, line in enumerate(lines) if first in line), None)
    if at is None:
        return set()
    while at > 0 and " | " in lines[at - 1]:
        at -= 1
    if at == 0:
        return set()
    return set(STANDALONE.findall(lines[at - 1]))


def named_number(claim: str, number: str, passages: Sequence[Passage]) -> bool:
    """Whether a cited passage states the number as the claim uses it (check 3):
    digits inside names ("OML118") are rarely in the quoted words."""
    places = [f"{p.title}\n{p.heading}\n{p.text}" for p in passages]
    units = number_units(claim, number)
    return any(find_quote(unit, place) for unit in units for place in places)


def measures(text: str) -> list[dict[str, Any]]:
    """Every number in the text with what it measures: its currency, scale and unit,
    each folded to one name, or None where the text gives none."""
    found = []
    for match in MEASURE.finditer(text):
        currency = match["currency"] or match["named"]
        scale = (match["scale"] or "").casefold() or None
        if scale in SHORT_SCALES and not currency:
            scale = None  # "3 m" is three metres, not three million
        unit = None
        if match["unit"]:
            unit, carried = UNITS[match["unit"].casefold()]
            scale = carried or scale
        found.append({
            "text": match.group().strip(),
            "number": match["number"].replace(",", ""),
            "currency": CURRENCIES[currency.casefold()] if currency else None,
            "scale": SCALES.get(scale, scale) if scale else None,
            "unit": unit,
        })  # fmt: skip
    return found


def labels(text: str) -> set[tuple[str, str]]:
    """The currencies, scales and units a text names anywhere, as (kind, name) pairs:
    a table's column header ("Revenue (NGN Billion)"), a chart's axis ("N billion")."""
    lowered = " ".join(text.split()).casefold()
    # The cell mark is a token, so a currency never lends its scale across cells.
    tokens = re.findall(r"₦|us\$|\$|£|€|\||[a-z]+(?:/[a-z]+)?", lowered)
    found = set()
    for index, token in enumerate(tokens):
        before = tokens[index - 1] if index else ""
        after = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token in CURRENCIES and (token != "n" or after in SCALES):
            found.add(("currency", CURRENCIES[token]))
        if token in SCALES and (token not in SHORT_SCALES or before in CURRENCIES):
            found.add(("scale", SCALES[token]))
        if token in UNITS:
            unit, carried = UNITS[token]
            found.add(("unit", unit))
            if carried:
                found.add(("scale", carried))
    for phrase in (form for form in UNITS if " " in form):
        if phrase in lowered:
            found.add(("unit", UNITS[phrase][0]))
    return found


def changed_amounts(claim: str, quoted: str, headers: str) -> list[str]:
    """The claim's amounts whose currency, scale or unit no quote supports (check 4).
    A quote supports it by giving it to the same number, or, where it gives the
    number bare (a table cell), by the cited passages' title, heading or context
    naming it (`headers`: the table's column headers). A number the claim gives
    none of these is left to check 3."""
    given = measures(quoted)
    named = labels(headers)
    missing = []
    for amount in measures(claim):
        kinds = [k for k in ("currency", "scale", "unit") if amount[k] is not None]
        if not kinds:
            continue
        supported = False
        for other in given:
            if other["number"] != amount["number"]:
                continue
            if all(
                other[k] == amount[k] or (other[k] is None and (k, amount[k]) in named)
                for k in kinds
            ):
                supported = True
        if not supported:
            missing.append(amount["text"])
    return missing


def locate(
    quote: str, cited: Passage, sources: Mapping[str, Passage]
) -> tuple[Passage, str] | None:
    """The passage a quote is in and its wording there: the cited passage, else the
    first other supplied passage holding it word for word (a wrong source id). Only
    passages that may be quoted count."""
    for passage in (cited, *sources.values()):
        if not passage.quotable:
            continue
        wording = find_quote(quote, passage.text)
        if wording is not None:
            return passage, wording
    return None


def regulation_terms(text: str) -> set[str]:
    """The rules, licences and approvals the text mentions, each under one name, and
    every regulation it names."""
    found = {
        name
        for name, pattern in REGULATION_TERMS.items()
        if re.search(pattern, text, re.IGNORECASE)
    }
    for name in REGULATION_NAME.findall(text):
        found.add(" ".join(name.split()).casefold() + " regulations")
    return found


def unquoted_regulations(claim: str, quoted: str) -> list[str]:
    """The claim's regulation terms its quotes do not mention (check 5). A named
    regulation is matched in the quotes by its words, however they are wrapped."""
    folded_quotes = " ".join(quoted.split()).casefold()
    missing = []
    for term in sorted(regulation_terms(claim)):
        if term in REGULATION_TERMS:
            if term not in regulation_terms(quoted):
                missing.append(term)
        elif term.removesuffix(" regulations") not in folded_quotes:
            missing.append(term)
    return missing


def check_claim(draft: DraftClaim, sources: Mapping[str, Passage]) -> Claim | Rejection:
    """The verified claim with its citations, or the reason it must be removed. The
    claim is checked and shown without the request's names for its passages."""
    if not draft.evidence:
        return Rejection(draft.text, "no quote")
    if prices.states_price(draft.text):
        return Rejection(draft.text, "states a per-unit price")
    text = without_labels(draft.text, sources)
    evidence = []
    cited = []
    captions: set[str] = set()
    for item in draft.evidence:
        passage = sources.get(item.source_id)
        if passage is None:
            return Rejection(draft.text, f"unknown source {item.source_id}")
        found = locate(item.quote, passage, sources)
        if found is None:
            if find_quote(item.quote, passage.text) is not None:
                return Rejection(
                    draft.text, f"quote from search-only {item.source_id} (a scan)"
                )
            return Rejection(
                draft.text, f"quote not in {item.source_id}: {item.quote!r}"
            )
        passage, quote = found
        cited.append(passage)
        captions |= caption_numbers(quote, passage)
        evidence.append(
            Evidence(
                passage_id=passage.id,
                url=passage.url,
                title=passage.title,
                heading=passage.heading,
                quote=quote,
                link=source_link(passage, quote),
                fetched_at=passage.fetched_at,
                published=passage.published,
            )
        )
    quoted = " ".join(item.quote for item in evidence)
    unquoted_numbers = (
        numbers(text) - numbers(quoted) - source_numbers(text, cited) - captions
    )
    missing = [
        number
        for number in sorted(unquoted_numbers)
        if not named_number(text, number, cited)
    ]
    if missing:
        return Rejection(draft.text, f"number not in its quotes: {', '.join(missing)}")
    headers = "\n".join(f"{p.title}\n{p.heading}\n{p.context}" for p in cited)
    changed = changed_amounts(text, quoted, headers)
    if changed:
        return Rejection(draft.text, f"amount not as quoted: {', '.join(changed)}")
    # A regulation the cited source is ("Petroleum Industry Act 2021", the
    # "Licencing Round Amendment Regulations") is named by its title (K29 F4).
    named = "\n".join(f"{p.title}\n{p.heading}" for p in cited)
    unquoted = unquoted_regulations(text, f"{quoted}\n{named}")
    if unquoted:
        return Rejection(
            draft.text, f"regulation not in its quotes: {', '.join(unquoted)}"
        )
    return Claim(text, tuple(evidence), draft.part)


def verify(
    drafts: Sequence[DraftClaim], sources: Mapping[str, Passage]
) -> tuple[tuple[Claim, ...], tuple[Rejection, ...]]:
    """Split the model's claims into those to show and those removed, keeping order."""
    results = [check_claim(draft, sources) for draft in drafts]
    claims = tuple(result for result in results if isinstance(result, Claim))
    rejected = tuple(result for result in results if isinstance(result, Rejection))
    return claims, rejected
