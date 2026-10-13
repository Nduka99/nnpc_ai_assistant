"""What a reader sees for one answer, as plain data shared by every interface (Lime
Green's view, carried over for NNPC).

The command line prints it, the web page renders it and the API returns it as JSON,
so all three show the same status, claims and sources. It holds only verified
claims, their numbered citations with each source's dates, the fixed notice, the
closest pages for a refusal, and the search-only pages an answer points to (a
scanned page, a picture), which are opened, never quoted. Claims removed by
verification are never shown.
"""

from typing import TypedDict

from petroknowledge import config, units
from petroknowledge.answer import closest_pages
from petroknowledge.records import Answer
from petroknowledge.verify import source_link

# Shown by every interface beside every answer (K02; NNPC's terms forbid implying
# affiliation).
INDEPENDENT = (
    "Independent research prototype, not affiliated with or endorsed by NNPC Limited. "
    "Answers quote public documents; check the linked source before relying on one."
)


class Source(TypedDict):
    """One verified quote, numbered so claims can point to it."""

    number: int
    title: str
    heading: str
    quote: str
    link: str  # opens the live page at the quote
    url: str
    published: str  # a post's publication date, YYYY-MM-DD, else empty
    captured: str  # capture date, YYYY-MM-DD
    compiled: bool  # quoted from a list compiled from the page, not the page (K27)


class ClaimView(TypedDict):
    text: str
    sources: list[int]  # numbers of the sources that support it


class Page(TypedDict):
    title: str
    url: str


class Pointer(TypedDict):
    """A page search found whose words cannot be quoted: opened, never quoted."""

    title: str
    link: str  # opens the page, a PDF at its page
    kind: str  # "scanned page" or "picture"


class AnswerView(TypedDict):
    question: str
    status: str
    notice: str  # fixed application text, possibly several lines; may be empty
    claims: list[ClaimView]
    sources: list[Source]
    closest_pages: list[Page]  # only for insufficient evidence
    pointers: list[Pointer]


def view(answer: Answer) -> AnswerView:
    """The reader's view of an answer; each distinct quote is one numbered source."""
    sources: list[Source] = []
    numbers: dict[tuple[int, str], int] = {}
    claims: list[ClaimView] = []
    for claim in answer.claims:
        refs: list[int] = []
        for evidence in claim.evidence:
            key = (evidence.passage_id, evidence.quote)
            if key not in numbers:
                numbers[key] = len(sources) + 1
                sources.append(
                    Source(
                        number=numbers[key],
                        title=evidence.title,
                        heading=evidence.heading,
                        quote=" ".join(evidence.quote.split()),
                        link=evidence.link,
                        url=evidence.url,
                        published=evidence.published,
                        captured=evidence.fetched_at[:10],
                        compiled=evidence.heading == units.COMPILED,
                    )
                )
            if numbers[key] not in refs:
                refs.append(numbers[key])
        claims.append(ClaimView(text=claim.text, sources=refs))
    closest = (
        [Page(title=p.title, url=p.url) for p in closest_pages(answer.passages)]
        if answer.status == "insufficient_evidence"
        else []
    )
    # Each page once, best first, as few as the closest pages of a refusal.
    pointers: list[Pointer] = []
    for passage in answer.pointers:
        link = source_link(passage, "")
        if link not in {pointer["link"] for pointer in pointers}:
            kind = "picture" if passage.image else "scanned page"
            pointers.append(Pointer(title=passage.title, link=link, kind=kind))
    return AnswerView(
        question=answer.question,
        status=answer.status,
        notice=answer.notice,
        claims=claims,
        sources=sources,
        closest_pages=closest,
        pointers=pointers[: config.CLOSEST_PAGES],
    )


def reply_text(shown: AnswerView) -> str:
    """An answer as the reader saw it, for the conversation that follows: its claims,
    then any notice (a referral or a refusal is its fixed text alone)."""
    lines = [claim["text"] for claim in shown["claims"]]
    if shown["notice"]:
        lines.append(shown["notice"])
    return "\n".join(lines)
