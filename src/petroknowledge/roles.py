"""Dated roles (K24): who holds which role, as each dated source states it, so code
decides which statement is the latest. A model cannot reliably tell an outdated fact
from a current one; dates can.

A statement (person, role, organisation, quote) is read from one dated passage by
one structured request, and kept only when code finds its quote in the passage with
the person's name in it (K15's rule for model-proposed edges). It is as of its
passage's date: a post's publication date, else a page's capture date.

Statements are grouped by role and organisation. One role is written many ways
("CFO", "Chief Financial Officer (R)", "EVP Downstream", "Executive Vice President of
Downstream"), so a title is grouped by `role_key`: abbreviations are spelled out as
the titles themselves spell them. In each group the newest statement is current as
of its date and the others are history, as Wikidata ranks a changing value; the date
decides, never how often a holder is mentioned. A question names a role by its
whole title or its initials, never by a single title word, and an organisation as
units are named; a question naming no organisation is about the site's own.
"""

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from petroknowledge import units
from petroknowledge.verify import find_quote

Statement = dict[str, Any]
Chat = Callable[[str, str, dict[str, Any]], object]

# The nouns of a role title: a passage that names none of them states no role and is
# not read.
ROLE_NOUNS = re.compile(
    r"\b(?:officer|president|director|chairman|chairperson|chair|manager|secretary|"
    r"head|minister|commissioner|controller|governor|ceo|cfo|coo|cto|cio|gceo|evp|"
    r"md)\b",
    re.IGNORECASE,
)
MAX_STATEMENTS = 12  # statements read from one passage, at most
MAX_ROLES = 3  # roles a question is answered about, at most
EARLIER = 2  # earlier holders shown with a role, at most

EXTRACT_PROMPT = """\
You read one passage from a company's public website. List every statement in it that \
a named person holds a role (a job title or office) in an organisation, as the passage \
states it.

Rules:
1. person: the person's name as written, without honorifics such as Mr, Mrs, Dr, \
Engr. or Chief.
2. role: the full title as the passage gives it, with any abbreviation spelled out \
(EVP: Executive Vice President), without the organisation's name.
3. organisation: the organisation the role is in, as the passage names it.
4. quote: copy, character for character, the shortest span of the passage that names \
the person and the role together.
5. Leave out roles the passage says the person held before, and roles with no person \
named.
6. If the passage states no such role, return an empty list."""

TEXT = {"type": "string", "minLength": 1}
EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "statements": {
            "type": "array",
            "maxItems": MAX_STATEMENTS,
            "items": {
                "type": "object",
                "properties": {
                    "person": TEXT,
                    "role": TEXT,
                    "organisation": TEXT,
                    "quote": TEXT,
                },
                "required": ["person", "role", "organisation", "quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["statements"],
    "additionalProperties": False,
}
FIELDS = ("person", "role", "organisation", "quote")
SMALL_WORDS = frozenset({"of", "the", "and", "to"})  # not part of a role's key
# An acting holder holds the role: "Ag. Chief Upstream Investment Officer" is the
# chief upstream investment officer (NUIMS's own page, K30).
ACTING = frozenset({"acting", "ag"})


def states_roles(text: str) -> bool:
    """Whether a passage names a role title at all, so is worth reading."""
    return ROLE_NOUNS.search(text) is not None


def text_sha256(title: str, text: str) -> str:
    """A passage's reading is kept by its title and text."""
    return hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()


def grounded(statement: Mapping[str, str], text: str) -> dict[str, str] | None:
    """The statement with the passage's own wording of its quote, if the passage holds
    the quote and the quote names the person; else None."""
    quote = find_quote(statement["quote"], text)
    if quote is None:
        return None
    words = [word for word in units.fold(statement["person"]).split() if len(word) > 1]
    folded = units.fold(quote).split()
    if not words or not all(word in folded for word in words):
        return None
    found = {field: statement[field].strip() for field in FIELDS}
    found["quote"] = quote
    return found


def read_statements(output: object, text: str) -> list[dict[str, str]] | None:
    """The statements of a reply that the passage supports (`grounded`); None if the
    reply breaks its schema."""
    if not (isinstance(output, dict) and isinstance(output.get("statements"), list)):
        return None
    found = []
    for statement in output["statements"]:
        if not (
            isinstance(statement, dict)
            and set(statement) == set(FIELDS)
            and all(
                isinstance(statement[f], str) and statement[f].strip() for f in FIELDS
            )  # fmt: skip
        ):
            return None
        kept = grounded(statement, text)
        if kept is not None:
            found.append(kept)
    return found


def extract(title: str, text: str, chat: Chat) -> list[dict[str, str]] | None:
    """One passage's role statements (`read_statements`): one structured request."""
    user = f'Passage from "{title}":\n{text}'
    return read_statements(chat(EXTRACT_PROMPT, user, EXTRACT_SCHEMA), text)


def organisation_key(name: str) -> str:
    """An organisation as statements are grouped by it: its name without a bracketed
    acronym, its legal form and "of", "the", "and" and "to" ("NNPC Ltd" is "nnpc").
    An organisation named only by article and legal form ("the Company") gives ""."""
    match = units.BRACKETED.match(name.strip())
    words = units.short_name(match["name"] if match else name).split()
    return " ".join(word for word in words if word not in SMALL_WORDS)


def bare_title(title: str) -> str:
    """A title without its parenthetical parts ("(R)", "(CIO)")."""
    return re.sub(r"\([^)]*\)", " ", title)


def expansions(statements: Sequence[Statement]) -> dict[str, str]:
    """Abbreviations as the titles spell them out: the initials of each run of 2 to 6
    title words, mapped to the most frequent run with them ("gceo": "group chief
    executive officer")."""
    runs: dict[str, Counter[str]] = {}
    for statement in statements:
        title = units.fold(bare_title(statement["role"]))
        words = [word for word in title.split() if word not in SMALL_WORDS]
        for size in range(2, 7):
            for start in range(len(words) - size + 1):
                run = words[start : start + size]
                if all(len(word) > 1 for word in run):
                    spelled = " ".join(run)
                    runs.setdefault(initials(spelled), Counter())[spelled] += 1
    return {letters: found.most_common(1)[0][0] for letters, found in runs.items()}


def role_key(statement: Mapping[str, Any], spelled: Mapping[str, str]) -> str:
    """A title as statements are grouped by it: without parenthetical parts, without
    a trailing qualifier that is an acronym or names the statement's organisation
    (", RTI", ", NNPC Ltd."), with each all-caps abbreviation spelled out
    (`expansions`), folded, and without "of", "the", "and" and "to"."""
    title = bare_title(statement["role"])
    head, _, tail = title.rpartition(",")
    letters = re.sub(r"[^A-Za-z]", "", tail)
    own = organisation_key(tail) == organisation_key(statement["organisation"])
    if head.strip() and (letters.isupper() or own):
        title = head
    words = []
    for token in title.split():
        letters = re.sub(r"[^A-Za-z]", "", token)
        short = letters.isupper() and 2 <= len(letters) <= 6
        words.append(spelled.get(letters.casefold(), token) if short else token)
    folded = [word for word in units.fold(" ".join(words)).split()
              if word not in SMALL_WORDS]  # fmt: skip
    if len(folded) > 1 and folded[0] in ACTING:
        folded = folded[1:]
    return " ".join(folded)


def same_person(one: str, other: str) -> bool:
    """Whether two names in one role are one person written two ways: the same
    surname, and either one name's words within the other's ("Bayo Ojulari", "Bashir
    Bayo Ojulari") or the same first initial ("Andy Odeh", "Andrew Odeh"); or the
    same words in another order ("Audu Ibrahim", "Ibrahim Audu")."""
    first = units.fold(one).split()
    second = units.fold(other).split()
    if first and set(first) == set(second):
        return True
    if not first or not second or first[-1] != second[-1]:
        return False
    within = set(first) <= set(second) or set(second) <= set(first)
    return within or first[0][0] == second[0][0]


def resolve(
    statements: Sequence[Statement], listed: Sequence[units.Unit]
) -> list[Statement]:
    """Each statement with its organisation resolved to the entity it is about (K30
    G2), so one role written with several spellings of its organisation is one role:
    a unit its organisation names (by name, acronym or former name), whose names then
    leave the title; else a unit its title names, likewise; else, on a unit's own page,
    that unit when the organisation is the company itself (a unit's management team is
    read as the parent's: HMO) or has no name of its own ("the Company", or none,
    `as_named`), unless the company's other pages hold the role: then it is a company
    office heading the unit (RTI's Chief Innovation Officer), and the unit's own
    statements of that role are the company's too; else, with no name, the one
    organisation the role's other statements name, else the company itself
    (`latest`); else the organisation as written."""
    named = units.naming(listed)
    pages = {unit["url"]: place for place, unit in enumerate(listed)}
    own = own_key(statements)
    spelled = expansions(statements)
    company = company_roles(statements, named, pages, own, spelled)
    offices: dict[tuple[str, str], str] = {}  # (role, unit) -> the company's spelling
    found = []
    nameless = []
    for statement in statements:
        resolved = dict(statement)
        by_organisation = named(statement["organisation"])
        by_title = named(statement["role"])
        page = pages.get(statement.get("url", ""))
        key = organisation_key(statement["organisation"])
        if by_organisation:
            unit = listed[by_organisation[0]]
            title = without_unit(statement["role"], written(statement))
            resolved["organisation"] = units.described([unit])
            resolved["role"] = without_unit(title, unit)
        elif by_title:
            unit = listed[by_title[0]]
            resolved["organisation"] = units.described([unit])
            resolved["role"] = without_unit(statement["role"], unit)
        elif page is not None and key in ("", own):
            role = role_key(statement, spelled)
            unit_name = units.described([listed[page]])
            if role in company:
                offices[(role, unit_name)] = statement["organisation"] or own
            else:
                resolved["organisation"] = unit_name
        if not organisation_key(resolved["organisation"]):
            nameless.append(resolved)
        found.append(resolved)
    for resolved in found:
        office = (role_key(resolved, spelled), resolved["organisation"])
        if office in offices:
            resolved["organisation"] = offices[office]
    placed = holders([s for s in found if organisation_key(s["organisation"])])
    for statement in nameless:
        held = placed.get(role_key(statement, spelled), {})
        if len(held) == 1:
            statement["organisation"] = next(iter(held.values()))
    return found


def company_roles(
    statements: Sequence[Statement],
    named: Callable[[str], list[int]],
    pages: Mapping[str, int],
    own: str,
    spelled: Mapping[str, str],
) -> set[str]:
    """The role keys the company holds by statements outside its units' own pages
    that name no unit."""
    found = set()
    for statement in statements:
        elsewhere = statement.get("url", "") not in pages
        unnamed = not named(statement["organisation"]) and not named(statement["role"])
        its_own = organisation_key(statement["organisation"]) in ("", own)
        if elsewhere and unnamed and its_own:
            found.add(role_key(statement, spelled))
    return found


def own_key(statements: Sequence[Statement]) -> str:
    """The organisation key of the company whose site it is: the one the statements
    name most often."""
    named = Counter(organisation_key(s["organisation"]) for s in statements)
    named.pop("", None)
    return named.most_common(1)[0][0] if named else ""


def holders(statements: Sequence[Statement]) -> dict[str, dict[str, str]]:
    """Each role key with the organisations its statements name: by organisation key,
    the first spelling."""
    spelled = expansions(statements)
    found: dict[str, dict[str, str]] = {}
    for statement in statements:
        held = found.setdefault(role_key(statement, spelled), {})
        held.setdefault(organisation_key(statement["organisation"]),
                        statement["organisation"])  # fmt: skip
    return found


def without_unit(title: str, unit: units.Unit) -> str:
    """A title without the unit it names, folded ("Managing Director, NNPC Trading
    Ltd" is "managing director"): its short name, acronym and legal form left out. A
    title naming no unit, or only the unit, is kept as written."""
    folded = units.fold(title)
    left = folded
    for name in (units.short_name(unit["name"]), units.fold(unit["acronym"])):
        if name:
            left = re.sub(rf"\b{re.escape(name)}\b", " ", left)
    if left == folded:
        return title
    words = left.split()
    while words and words[-1] in units.LEGAL_FORMS:
        words.pop()
    return " ".join(words) or title


def latest(statements: Sequence[Statement]) -> list[dict[str, Any]]:
    """Each role of an organisation (`role_key`) with its current statement (the
    newest, by date then passage) and its earlier holders (newest first, one
    statement each). The date decides, never how often a holder is mentioned. A
    statement whose organisation has no name of its own ("the Company") is about the
    company whose site it is: the organisation the statements name most often."""
    spelled = expansions(statements)
    keys = [organisation_key(statement["organisation"]) for statement in statements]
    own = own_key(statements)
    groups: dict[tuple[str, str], list[Statement]] = {}
    for statement, organisation in zip(statements, keys, strict=True):
        key = (role_key(statement, spelled), organisation or own)
        groups.setdefault(key, []).append(statement)
    found = []
    for key in sorted(groups):
        dated = sorted(groups[key], key=lambda s: (s["stated_on"], s["passage_id"]))
        current = dated[-1]
        earlier: list[Statement] = []
        seen = [current["person"]]
        for statement in reversed(dated[:-1]):
            if not any(same_person(statement["person"], name) for name in seen):
                seen.append(statement["person"])
                earlier.append(statement)
        found.append({"role": current["role"], "key": key[0],
                      "organisation": current["organisation"],
                      "organisation_key": key[1], "own": key[1] == own,
                      "current": current, "earlier": earlier})  # fmt: skip
    return found


def initials(title: str) -> str:
    """A title's initials ("chief financial officer" is "cfo")."""
    return "".join(word[0] for word in title.split())


def naming(roles: Sequence[Mapping[str, Any]]) -> Callable[[str], list[int]]:
    """A function giving the roles a text asks about, by their place in `roles`: the
    role's whole title or its initials name it, never a single title word, since
    titles are made of ordinary words ("oil", "financial"); a one-word title names it
    only beside the role's organisation. When the text names some
    of those roles' organisations, only those roles are kept; when it names none, the
    site's own organisation's roles (`own`) if any. Roles of one organisation with the
    same initials are one role written two ways ("Chief Finance Officer"), so only
    the most recently stated is kept; at most MAX_ROLES, the most recently stated
    first."""
    titles = [role["key"] for role in roles]
    organisations = units.naming(
        [
            {
                "name": role["organisation_key"],
                "acronym": acronym(role),
                "url": "",
            }
            for role in roles
        ]  # fmt: skip
    )

    def named(text: str) -> list[int]:
        folded = units.fold(text)
        words = set(folded.split())
        found = []
        named_organisations = set(organisations(text))
        for place, title in enumerate(titles):
            letters = initials(title)
            whole = re.search(rf"\b{re.escape(title)}\b", folded) is not None
            # A one-word title ("chairman", "principal") is an ordinary word too: it
            # names a role only beside the role's organisation.
            if " " not in title and place not in named_organisations:
                whole = False
            if whole or (len(letters) >= 3 and letters in words):
                found.append(place)
        inside = [place for place in found if place in named_organisations]
        if not named_organisations:
            inside = [place for place in found if roles[place]["own"]]
        chosen = inside or found
        chosen.sort(key=lambda p: roles[p]["current"]["stated_on"], reverse=True)
        kept = []
        seen = set()
        for place in chosen:
            same = (roles[place]["organisation_key"], initials(titles[place]))
            if same not in seen:
                seen.add(same)
                kept.append(place)
        return kept[:MAX_ROLES]

    return named


def names_person(text: str, person: str) -> bool:
    """Whether a text names a person: every word of the name longer than one letter,
    folded, is a word of the text."""
    words = [word for word in units.fold(person).split() if len(word) > 1]
    found = set(units.fold(text).split())
    return bool(words) and all(word in found for word in words)


def written(role: Mapping[str, Any]) -> units.Unit:
    """A role's or statement's organisation as a unit is named: its name and the
    acronym it brackets, if any."""
    match = units.BRACKETED.match(role["organisation"].strip())
    name = match["name"] if match else role["organisation"]
    return {"name": name, "acronym": acronym(role), "url": ""}


def organisation_named(text: str, role: Mapping[str, Any]) -> bool:
    """Whether a text names a role's organisation, as units are named."""
    unit = written(role)
    return bool(unit["name"].strip()) and bool(units.naming([unit])(text))


def as_named(statement: Statement, text: str) -> Statement:
    """A statement as its passage names it: without an organisation the passage does
    not name. The schema requires one, so a passage naming none gets a filler ("not
    specified") or a guess ("Federal Government"): 110 of 2,011 statements (K30)."""
    if organisation_named(text, statement):
        return statement
    return dict(statement) | {"organisation": ""}


def earlier_holder(
    text: str, found: Sequence[Mapping[str, Any]], whole: str = ""
) -> str:
    """Why a claim must go because it names an earlier holder of a role it names,
    and neither it nor the rest of its answer (`whole`) names the latest holder (K30
    G4); "" when it does not. A role is named when every word of its title is in the
    claim ("Executive Vice President (EVP), Downstream") and its organisation is
    named, or is the site's own; the dates decided who is latest (`latest`)."""
    words = set(units.fold(text).split())
    for role in found:
        title = role["key"].split()
        if len(title) < 2 or not set(title) <= words:
            continue
        if not (role["own"] or organisation_named(text, role)):
            continue
        if names_person(f"{text} {whole}", role["current"]["person"]):
            continue
        for statement in role["earlier"]:
            if names_person(text, statement["person"]):
                return (f"names an earlier holder of {role['role']} "
                        f"({role['organisation'] or 'the company'}); the latest "
                        f"statement names {role['current']['person']}")  # fmt: skip
    return ""


def acronym(role: Mapping[str, Any]) -> str:
    """The acronym an organisation's name brackets, if any ("Research, Technology &
    Innovation (RTI)")."""
    match = units.BRACKETED.match(role["organisation"].strip())
    return match["acronym"] if match else ""


def described(roles: Sequence[Mapping[str, Any]]) -> str:
    """Asked roles as the answer request is told them: who holds each, by which
    source, and the earlier holders. No dates: code has compared them, and a date
    given to the model ends up in a claim whose quote does not hold it."""
    lines = []
    for role in roles:
        current = role["current"]
        organisation = role["organisation"] or "the company"
        line = (f"- {role['role']} ({organisation}): {current['person']}, "
                f"in \"{current['title']}\"")  # fmt: skip
        earlier = [statement["person"] for statement in role["earlier"][:EARLIER]]
        if earlier:
            line += "; earlier holders: " + "; ".join(earlier)
        lines.append(line + ".")
    return "\n".join(lines)
