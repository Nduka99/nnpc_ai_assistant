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
    folded = units.fold(" ".join(words)).split()
    return " ".join(word for word in folded if word not in SMALL_WORDS)


def same_person(one: str, other: str) -> bool:
    """Whether two names in one role are one person written two ways: the same
    surname, and either one name's words within the other's ("Bayo Ojulari", "Bashir
    Bayo Ojulari") or the same first initial ("Andy Odeh", "Andrew Odeh")."""
    first = units.fold(one).split()
    second = units.fold(other).split()
    if not first or not second or first[-1] != second[-1]:
        return False
    within = set(first) <= set(second) or set(second) <= set(first)
    return within or first[0][0] == second[0][0]


def latest(statements: Sequence[Statement]) -> list[dict[str, Any]]:
    """Each role of an organisation (`role_key`) with its current statement (the
    newest, by date then passage) and its earlier holders (newest first, one
    statement each). The date decides, never how often a holder is mentioned. A
    statement whose organisation has no name of its own ("the Company") is about the
    company whose site it is: the organisation the statements name most often."""
    spelled = expansions(statements)
    keys = [organisation_key(statement["organisation"]) for statement in statements]
    named = Counter(key for key in keys if key)
    own = named.most_common(1)[0][0] if named else ""
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
        line = (f"- {role['role']} ({role['organisation']}): {current['person']}, "
                f"in \"{current['title']}\"")  # fmt: skip
        earlier = [statement["person"] for statement in role["earlier"][:EARLIER]]
        if earlier:
            line += "; earlier holders: " + "; ".join(earlier)
        lines.append(line + ".")
    return "\n".join(lines)
