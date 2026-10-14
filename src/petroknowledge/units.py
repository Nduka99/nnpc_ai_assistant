"""Business units (K23): the units a company's own units page lists and links, which
of them a user's words name, and the passages in each unit's scope. They are the
graph's first entities (K15).

A unit is a line of the units page's named section, as the page was read, that
links to a page of its own: "NNPC Gas Marketing Limited (NGML)" with its link. They
are read from the company's structure, never from a hand-written list. A text names
a unit by its acronym or by its name without the legal form ("NNPC Gas Marketing"),
as whole words, with case, spacing, punctuation and "&"/"and" folded, or by a former
name its own page gives it ("NPDC" for NEPL, K29). A passage is
in a unit's scope when its document is the unit's page, or when its title or its
text names the unit.

The units a question asks about come from the user's own words, never from the
model's rewrite of them: in K22's runs, five of nine wrong answers began with a
rewrite that changed or dropped the unit named.

The units page's whole list is also compiled into one passage, by group (K27): read
as text, it was split across passages mid-group, so no passage held the list or tied
a unit to its group (Lime Green X12). The passage is compiled, not copied: its
heading says so, and a claim quoting it links to the page itself.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

# name, acronym ("" for none), url, and any former names its own page gives it
# ("aliases", "; " between them; K29 F2)
Unit = dict[str, str]
Group = dict[str, Any]  # name, words (lines), entries (lines), as the page shows them
LEGAL_FORMS = ("limited", "ltd", "gte", "plc", "company")
BRACKETED = re.compile(r"^(?P<name>.+?)\s*\(\s*(?P<acronym>[^()]+?)\s*\)$")
COMPILED = "Units listed on this page, by group (compiled from the page's list)"
# A former name as a unit's own page states it: "formerly (known as) (the) <Name>
# (<ACRONYM>)". "Formerly part of ..." is not a name: a name starts with a capital.
FORMERLY = re.compile(
    r"\bformerly\s+(?:known\s+as\s+)?(?:the\s+)?"
    r"(?P<name>[A-Z][\w&'’.-]*(?:\s+(?:[A-Z][\w&'’.-]*|of|and|for))*)"
    r"(?:\s*\((?P<acronym>[A-Z][\w&]{1,9})\))?"
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def fold(text: str) -> str:
    """Text as names are compared: case folded, "&" read as "and", and every run of
    punctuation, hyphens, slashes and spacing one space."""
    text = text.casefold().replace("&", " and ")
    return " ".join(re.sub(r"[\W_]+", " ", text).split())


def short_name(name: str) -> str:
    """A unit's name as people write it: folded, without its legal form ("NNPC
    Foundation Limited/Gte" is "nnpc foundation")."""
    words = fold(name).split()
    while len(words) > 1 and words[-1] in LEGAL_FORMS:
        words.pop()
    return " ".join(words)


def listed(
    raw_html: str, page_url: str, elements: Sequence[Mapping[str, Any]], section: str
) -> list[Unit]:
    """The units a units page lists: each line of its `section`, as the reader read
    the page, that a link of the page shows, in the page's order. A bracketed
    acronym is split from the name."""
    lines = set()
    for element in elements:
        if section in element.get("section", []):
            lines.add(fold(element["text"]))
    found: list[Unit] = []
    seen = set()
    soup = BeautifulSoup(raw_html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        text = " ".join(anchor.get_text(" ").split())
        if fold(text) not in lines or fold(text) in seen:
            continue
        seen.add(fold(text))
        match = BRACKETED.match(text)
        name, acronym = (match["name"], match["acronym"]) if match else (text, "")
        url = urljoin(page_url, str(anchor["href"]))
        found.append({"name": name, "acronym": acronym, "url": url})
    return found


def grouped(
    raw_html: str, elements: Sequence[Mapping[str, Any]], section: str
) -> list[Group]:
    """A units page's list by group, in the page's order and words. The list's
    entries are the lines of `section` shown beside a unit's link, in the same
    parent: an entry shown without a link counts too. Each run of other lines before
    a run of entries leads a group: its first line names it, the rest are its own
    words. Entries before any such line form a group without a name."""
    lines = [e["text"] for e in elements if section in e.get("section", [])]
    known = {fold(line) for line in lines}
    entries = set()
    soup = BeautifulSoup(raw_html, "html.parser")
    for anchor in soup.find_all("a", href=True):
        if fold(anchor.get_text(" ")) in known and anchor.parent is not None:
            for beside in anchor.parent.find_all(recursive=False):
                entries.add(fold(beside.get_text(" ")))
    groups: list[Group] = []
    for line in lines:
        if fold(line) in entries:
            if not groups:
                groups.append({"name": "", "words": [], "entries": []})
            groups[-1]["entries"].append(line)
        elif groups and not groups[-1]["entries"]:
            groups[-1]["words"].append(line)
        else:
            groups.append({"name": line, "words": [], "entries": []})
    return [group for group in groups if group["entries"]]


def compiled_text(section: str, groups: Sequence[Group]) -> str:
    """The compiled passage: its heading, how many units in how many groups, then
    each group's units and its own words. A group without a name goes by the list's
    section."""
    count = sum(len(group["entries"]) for group in groups)
    lines = [COMPILED, f"{section}: {count} units in {len(groups)} groups."]
    for group in groups:
        name = group["name"] or section
        noun = "unit" if len(group["entries"]) == 1 else "units"
        entries = "; ".join(group["entries"])
        lines.append(f"{name}, {len(group['entries'])} {noun}: {entries}.")
        if group["words"]:
            lines.append(f"{name}: {' '.join(group['words'])}")
    return "\n".join(lines)


def former_names(text: str, unit: Unit) -> list[str]:
    """The names a unit's own page says it formerly had, with their acronyms, from
    sentences that name the unit before saying so ("NNPC E&P Limited (NEPL),
    formerly the Nigerian Petroleum Development Company (NPDC)"; K29 F2)."""
    named = naming([unit])
    found: list[str] = []
    for sentence in SENTENCE_END.split(text):
        for match in FORMERLY.finditer(sentence):
            if not named(sentence[: match.start()]):
                continue
            for name in (match["name"], match["acronym"]):
                if name and name not in found:
                    found.append(name)
    return found


def structure_named(text: str, group_names: Sequence[str]) -> int:
    """How many of the groups' names a text names, as whole words, folded; a list
    without group lines has one nameless group, which no text names."""
    folded = fold(text)
    found = 0
    for name in group_names:
        if fold(name) and re.search(rf"\b{re.escape(fold(name))}\b", folded):
            found += 1
    return found


def naming(units: Sequence[Unit]) -> Callable[[str], list[int]]:
    """A function giving the units a text names, by their place in `units`: by
    name, acronym or a former name."""
    patterns = []
    for unit in units:
        names = [short_name(unit["name"])]
        if unit["acronym"]:
            names.append(fold(unit["acronym"]))
        names += [short_name(alias) for alias in unit.get("aliases", "").split("; ")
                  if alias]  # fmt: skip
        either = "|".join(re.escape(name) for name in names)
        patterns.append(re.compile(rf"\b(?:{either})\b"))

    def named(text: str) -> list[int]:
        folded = fold(text)
        return [n for n, pattern in enumerate(patterns) if pattern.search(folded)]

    return named


def members(
    units: Sequence[Unit], passages: Sequence[tuple[int, str, str, str]]
) -> list[tuple[int, int]]:
    """(unit place, passage id) for each passage (id, url, title, text) in a unit's
    scope: its document is the unit's page, or its title or its text names it."""
    named = naming(units)
    pages = {unit["url"]: n for n, unit in enumerate(units)}
    found = []
    for passage_id, url, title, text in passages:
        places = set(named(title)) | set(named(text))
        if url in pages:
            places.add(pages[url])
        found += [(n, passage_id) for n in sorted(places)]
    return found


def described(units: Sequence[Unit]) -> str:
    """Units as the model requests name them: each name with its acronym."""
    names = []
    for unit in units:
        acronym = f" ({unit['acronym']})" if unit["acronym"] else ""
        names.append(unit["name"] + acronym)
    return "; ".join(names)


def asked(
    named: Callable[[str], list[int]], message: str, history: Sequence[tuple[str, str]]
) -> list[int]:
    """The units a user's message names. A follow-up that names none asks about the
    units of the most recent earlier user message in `history` that names one."""
    for text in [message, *[earlier for earlier, _ in reversed(history)]]:
        found = named(text)
        if found:
            return found
    return []
