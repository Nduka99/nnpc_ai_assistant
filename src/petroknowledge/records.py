"""Plain data records passed between the modules."""

from dataclasses import dataclass
from typing import Literal

# The only three outcomes of a question. An unreachable model or a damaged index
# is an operational error, not a status.
Status = Literal["answered", "insufficient_evidence", "safety_referral"]


@dataclass(frozen=True)
class Passage:
    """One section of a web page or of a PDF page: the unit that is searched and
    cited."""

    id: int
    url: str
    title: str  # page or document title
    heading: str  # section heading; a web passage's text starts with it
    text: str  # the source's own words: the only part a quote may come from
    fetched_at: str  # when the page was captured, ISO 8601 UTC
    page: int | None = None  # a PDF passage's page
    # Searched and embedded with the text, never quoted: a PDF passage's section
    # path, table caption and column headers (X9); a web passage's section path in
    # the packed forms (X42 W3), else empty.
    context: str = ""
    # A passage made from a picture names it (the SHA-256 of its stored PNG); its text
    # is the picture's own words (X43). Empty for every other passage.
    image: str = ""
    # Whether an answer may quote it. A scanned page's transcript helps search find
    # the page but is never quoted: no transcription passed its gate (K12).
    quotable: bool = True
    # A post's publication date (ISO 8601), else empty: shown by the application, so
    # the model never has to write a date its quotes do not hold.
    published: str = ""


def described(title: str, context: str, text: str) -> str:
    """What the embedding model and the reranker read for a passage: its title,
    context and text, one per line (a web passage has no context)."""
    return "\n".join(part for part in (title, context, text) if part)


def as_read(passage: Passage) -> str:
    """What the answer model reads of a passage, without its markup: its page title,
    its section and its text, one per line (`answer.user_prompt`). Evidence is looked
    for here when measuring what reached the model (X44 M1)."""
    parts = (passage.title, passage.heading, passage.text)
    return "\n".join(part for part in parts if part)


@dataclass(frozen=True)
class Part:
    """One thing a question asks, as the first request wrote it: its search question
    and, when it asks about several things (several products, or several facts about
    one product), each of them as a search question naming its subject (X48)."""

    question: str
    items: tuple[str, ...] = ()


@dataclass(frozen=True)
class DraftEvidence:
    """A quote as the model returned it, before verification."""

    source_id: str  # "S1", "S2", ...: the passage's position in the prompt
    quote: str


@dataclass(frozen=True)
class DraftClaim:
    """A claim as the model returned it, before verification."""

    text: str
    evidence: tuple[DraftEvidence, ...]
    part: int = 1  # the number of the question part it answers (C2)


@dataclass(frozen=True)
class Evidence:
    """A verified quote and the stored details needed to cite it.

    Nothing here is written by the model: the quote is the passage's own wording
    and the rest comes from the index.
    """

    passage_id: int
    url: str
    title: str
    heading: str
    quote: str
    link: str  # opens the live page with the quote highlighted
    fetched_at: str
    published: str = ""  # the cited post's publication date, else empty


@dataclass(frozen=True)
class Claim:
    """A claim that passed verification, shown with its evidence."""

    text: str
    evidence: tuple[Evidence, ...]
    part: int = 1  # the number of the question part it answers (C2)


@dataclass(frozen=True)
class Rejection:
    """A claim removed by verification: kept for evaluation, never shown as fact."""

    text: str
    reason: str


@dataclass(frozen=True)
class Answer:
    """Core answer plus audit data; interfaces must never render `rejected`."""

    question: str
    status: Status
    notice: str  # fixed application text; empty when the claims need no caution
    claims: tuple[Claim, ...]
    passages: tuple[Passage, ...]  # the passages the model was given, best first
    rejected: tuple[Rejection, ...]
    # The search questions the first request wrote, shown as "Understood as" on a
    # follow-up; empty for a safety referral, which is never searched.
    understood: tuple[str, ...] = ()
    # Search-only matches (a scanned page's transcript, the words read in a picture):
    # pages the reader may open, never quoted (K12, K16).
    pointers: tuple[Passage, ...] = ()
