"""The web page gate: Lime Green's X42 method on NNPC's pages.

Pages are drawn by seed from the saved web readings, per stratum. Each is drawn by
headless Chromium from its stored HTML, with the site's own CSS, fonts and scripts, as
a visitor who reads all of it sees it: the cookie notice refused, the page scrolled
through, and the lines each click newly shows (another tab, a profile, a slide) added
after the page's visible text. The truth is judged from those renderings as an outline
over the numbered visible lines, and the reader's saved readings are scored against
it (X42 W2's measures).
"""

import random
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import pairwise
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from petroknowledge import config

VIEWPORT = {"width": 1280, "height": 900}
LOAD_MS = 60_000  # a page that has not loaded by then is reported, not judged
BLOCKED = (
    # robots.txt forbids crawlers on NNPC's CMS hosts: their pictures stay out.
    "cms1977.nnpcgroup.com",
    "staging-cms1977.nnpcgroup.com",
    "nnpc-web-cms.nnpcgroup.com",
    # Trackers and live widgets: not part of the page a visitor reads, and they keep
    # the network busy (NNPC's error tracker and price ticker).
    "sentry.io",
    "check-dc.com",
    "googletagmanager.com",
    "google-analytics.com",
    "doubleclick.net",
    "facebook.net",
    "youtube.com",
    "youtube-nocookie.com",
)
CONSENT = "Reject all"  # NNPC's cookie notice: the least a visitor accepts
CONSENT_MS = 2500  # the notice slides in after the page loads
SETTLE_MS = 150  # after each scroll step, for scroll-triggered text and counters
OPEN_MS = 700  # after each click, for its animation
# What a click reveals on NNPC's pages, found by probing before any truth was judged.
TABS = "[role=tab], .team-tab-menu-wrapper__menu"
CARDS = ".team-member"  # a profile card: a click opens the profile
PROFILE = ".profile-overlay.show"  # the profile a card opens
CLOSE = ".profile-overlay .close-icon"
SLIDES = ".w-slider-dot"
# Accordions a visitor opens with a click: each term, and its answer relative to it.
ACCORDIONS = (
    ("dl > dt", "xpath=following-sibling::dd[1]"),  # a definition-list FAQ (X42)
    (".accordion__title", "xpath=following-sibling::*[1]"),  # NNPC's legal pages
    (".collapsible", "xpath=following-sibling::*[1]"),  # NNPC Retail's FAQ
)
SHOW_OPENED = (
    "document.querySelectorAll('[data-opened]').forEach(e => Object.assign(e.style,"
    " {display: 'block', height: 'auto', maxHeight: 'none', overflow: 'visible'}))"
)
REVEALED = "▸ after clicking: "  # the marker line before a click's new lines
DESCRIBED = 5  # words: an alt text this long may describe its picture (X42)
QUOTAS = {"nuxt data": 5, "page": 8, "post": 10, "retail": 2, "careers": 2,
          "regulator": 1, "wikipedia": 2}  # fmt: skip
HELD_OUT = {"page": 3, "post": 4, "retail": 1, "careers": 1, "wikipedia": 1}
HOSTS = {"retail.nnpcgroup.com": "retail", "careers.nnpcgroup.com": "careers",
         "en.wikipedia.org": "wikipedia"}  # fmt: skip
# X42's bars for recall and precision; its list bar was relative to a baseline arm,
# so with one reader it is set below the 1.000 Lime Green's adopted reader reached.
GATE = {"recall": 0.99, "precision": 0.98, "lists": 0.95}
WORD = re.compile(r"\w+")


def stratum(reading: Mapping[str, Any]) -> str:
    """A saved web reading's stratum: read from its Nuxt data, else by host and kind."""
    host = urlsplit(reading["url"]).netloc
    if reading["origin"] == "nuxt data":
        return "nuxt data"
    if host in HOSTS:
        return HOSTS[host]
    if host.endswith("nnpcgroup.com"):
        return str(reading["kind"])
    return "regulator"


def draw(
    readings: Mapping[str, Mapping[str, Any]],
    quotas: Mapping[str, int],
    seed: int,
    exclude: set[str],
) -> list[str]:
    """Readings (by SHA-256) drawn by seed, each stratum to its quota, none from
    `exclude`."""
    rng = random.Random(seed)
    chosen: list[str] = []
    for name, quota in quotas.items():
        pool = sorted(
            sha256
            for sha256, reading in readings.items()
            if stratum(reading) == name and sha256 not in exclude
        )
        chosen += rng.sample(pool, min(quota, len(pool)))
    return chosen


def blocked(url: str) -> bool:
    """Whether a request goes to a blocked host or one of its subdomains."""
    host = urlsplit(url).netloc
    return any(host == name or host.endswith("." + name) for name in BLOCKED)


def router(served: dict[str, str]) -> Callable[[Any], None]:
    """A request handler: the page being rendered is served from the store; what a
    visitor's browser loads for it loads; blocked hosts are refused."""

    def handle(route: Any) -> None:
        url = route.request.url
        if url in served:
            route.fulfill(body=served[url], content_type="text/html; charset=utf-8")
        elif blocked(url):
            route.abort()
        else:
            route.continue_()

    return handle


def visible_lines(text: str) -> list[str]:
    """A rendered page's visible text as the outlines number it: non-empty lines."""
    return [line.strip() for line in text.split("\n") if line.strip()]


def new_lines(before: str, after: str) -> list[str]:
    """The lines `after` shows beyond those `before` showed, as a multiset, in
    `after`'s order."""
    left = Counter(visible_lines(before))
    found = []
    for line in visible_lines(after):
        if left[line]:
            left[line] -= 1
        else:
            found.append(line)
    return found


def shown(page: Any, selector: str) -> list[Any]:
    """The page's visible elements matching `selector`."""
    return [element for element in page.locator(selector).all() if element.is_visible()]


def label(element: Any) -> str:
    """A control's name for its marker line: its text on one line."""
    return " ".join(str(element.inner_text()).split()) or "a control"


def scroll_through(page: Any) -> None:
    """Scroll to the end in half-screen steps so scroll-triggered text appears."""
    height = int(page.evaluate("document.body.scrollHeight"))
    for top in range(0, height + VIEWPORT["height"], VIEWPORT["height"] // 2):
        page.evaluate(f"window.scrollTo(0, {top})")
        page.wait_for_timeout(SETTLE_MS)


def open_answers(page: Any) -> None:
    """Open what a visitor opens in place with a click: every disclosure widget
    (`details`, nested ones included), and each hidden answer of an accordion, by
    clicking its term. An accordion may show one answer at a time, so each answer that
    opened is marked, then all marked answers are shown together (Lime Green X42):
    nothing hidden by other means is revealed."""
    page.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
    for terms, answers in ACCORDIONS:
        for term in page.locator(terms).all():
            answer = term.locator(answers)
            if not term.is_visible() or answer.is_visible():
                continue
            term.click()
            page.wait_for_timeout(OPEN_MS)
            if answer.is_visible():
                answer.evaluate("e => e.dataset.opened = 'yes'")
    page.wait_for_timeout(OPEN_MS)
    page.evaluate(SHOW_OPENED)


def prepare(page: Any) -> None:
    """Show the page as a visitor first reads it: the cookie notice refused, the page
    scrolled through so scroll-triggered text appears, what opens in place opened
    (`open_answers`), then back at the top."""
    page.wait_for_timeout(CONSENT_MS)
    for button in page.get_by_role("button", name=CONSENT).all():
        if button.is_visible():
            button.click()
    scroll_through(page)
    open_answers(page)
    page.evaluate("window.scrollTo(0, 0)")


def clicks(page: Any) -> list[str]:
    """The lines each click newly shows, each click's lines under a marker line naming
    it: the slides first, then each tab and, under it, each profile. A line counts
    as new against the page just before the click; a profile is read whole from the
    panel it opens, in the order shown, then closed."""
    added: list[str] = []
    before = str(page.evaluate("document.body.innerText"))

    def click(control: Any, name: str) -> None:
        nonlocal before
        control.scroll_into_view_if_needed()
        control.click()
        page.wait_for_timeout(OPEN_MS)
        after = str(page.evaluate("document.body.innerText"))
        found = new_lines(before, after)
        if found:
            added.extend([REVEALED + name, *found])
        before = after

    for number, dot in enumerate(shown(page, SLIDES), 1):
        click(dot, f"slide {number}")
    for tab in shown(page, TABS) or [None]:
        if tab is not None:
            click(tab, label(tab))
        for card in shown(page, CARDS):
            card.scroll_into_view_if_needed()
            card.click()
            page.wait_for_timeout(OPEN_MS)
            for panel in shown(page, PROFILE):
                found = visible_lines(str(panel.inner_text()))
                added.extend([REVEALED + label(card), *found])
            for close in shown(page, CLOSE):
                close.click()
                page.wait_for_timeout(OPEN_MS)
            before = str(page.evaluate("document.body.innerText"))
    return added


def alt_texts(page: Any) -> list[str]:
    """The page's image alt texts of DESCRIBED words or more, each once."""
    found = page.evaluate("[...document.querySelectorAll('img[alt]')].map(i => i.alt)")
    kept: list[str] = []
    for text in found:
        text = " ".join(str(text).split())
        if len(text.split()) >= DESCRIBED and text not in kept:
            kept.append(text)
    return kept


def render(
    pages: Sequence[Mapping[str, str]], out: Path, browser: Any
) -> dict[str, int]:
    """Each page (`url`, `sha256`, `html`) drawn into `out`: a full-page screenshot and
    the visible text of the page as first shown, the lines its clicks reveal after
    that text, and its alt texts; its line count."""
    out.mkdir(parents=True, exist_ok=True)
    served: dict[str, str] = {}
    page = browser.new_page(viewport=VIEWPORT, user_agent=config.USER_AGENT)
    page.route("**/*", router(served))
    counts = {}
    for item in pages:
        served.clear()
        served[item["url"]] = item["html"]
        page.goto(item["url"], wait_until="load", timeout=LOAD_MS)
        prepare(page)
        name = item["sha256"]
        page.screenshot(path=str(out / f"{name}.png"), full_page=True)
        first = str(page.evaluate("document.body.innerText"))
        text = "\n".join([first, *clicks(page)])
        (out / f"{name}.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
        alts = "\n".join(alt_texts(page))
        (out / f"{name}.alts.txt").write_text(alts + "\n", encoding="utf-8")
        counts[name] = len(visible_lines(text))
        time.sleep(config.REQUEST_DELAY_SECONDS)
    page.close()
    return counts


@contextmanager
def chromium() -> Iterator[Any]:
    """Playwright's headless Chromium (the `evaluation` group)."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        yield browser
        browser.close()


def referenced(reference: int | list[int] | str) -> list[int]:
    """The lines a title or heading names: one line, a [first, last] range, or none
    when the judge writes the heading's text."""
    if isinstance(reference, str):
        return []
    if isinstance(reference, int):
        return [reference]
    return list(range(reference[0], reference[1] + 1))


def line_text(lines: Sequence[str], reference: int | list[int] | str) -> str:
    """A title's or heading's text: its lines joined by a space, as one heading a
    text-splitting script broke over several lines (amendment 1), or the text the
    judge wrote when furniture shares its line (amendment 5: Wikipedia's edit
    links)."""
    if isinstance(reference, str):
        return reference
    return " ".join(lines[number] for number in referenced(reference))


def outline_lines(outline: Mapping[str, Any]) -> list[int]:
    """The visible lines an outline's blocks use (the title apart), in its order."""
    used: list[int] = []
    for entry in outline["blocks"]:
        kind = entry[0]
        if kind == "h":
            used += referenced(entry[2])
        elif kind in ("p", "meta", "join"):
            used += range(entry[1], entry[-1] + 1)
        elif kind == "list":
            step = entry[3] if len(entry) > 3 else 1
            used += range(entry[1], entry[2] + 1, step)
    return used


def outline_problems(outline: Mapping[str, Any], count: int) -> list[str]:
    """Lines an outline uses beyond the page's `count` lines, twice or out of
    reading order: slips made while judging."""
    used = outline_lines(outline)
    title = referenced(outline["title"])
    problems = [f"line {n} beyond the page" for n in [*title, *used] if n >= count]
    problems += [f"line {b} after line {a}" for a, b in pairwise(used) if b <= a]
    problems += [f"title line {n} used again" for n in title if n in used]
    return problems


def build_truth(outline: Mapping[str, Any], lines: Sequence[str]) -> dict[str, Any]:
    """A page's truth from its outline over `lines`: the title, the headings with their
    levels, and the main content's blocks in order, each with its section path (the
    headings above it, outermost first). Entries: ["h", level, line], ["meta", line],
    ["p", first, last] (one paragraph per line), ["join", first, last] (one paragraph,
    the lines joined without a space: a counter's digits), ["list", first, last, step]
    (one block; the step, 1 if left out, skips lines between items), ["alt", text]
    and ["end", level] (closes the sections at that level and below). A title or a
    heading's line may be a [first, last] range."""
    path: list[tuple[int, str]] = []
    headings = []
    blocks: list[dict[str, Any]] = []
    for entry in outline["blocks"]:
        kind = entry[0]
        if kind == "end":
            path = [(n, h) for n, h in path if n < entry[1]]
            continue
        if kind == "h":
            level, text = entry[1], line_text(lines, entry[2])
            path = [(n, h) for n, h in path if n < level] + [(level, text)]
            headings.append({"level": level, "text": text})
            continue
        section = [heading for _, heading in path]
        if kind == "alt":
            blocks.append({"kind": kind, "text": entry[1], "path": section})
        elif kind in ("p", "meta"):
            for number in range(entry[1], entry[-1] + 1):
                blocks.append({"kind": kind, "text": lines[number], "path": section})
        elif kind == "join":
            text = "".join(lines[entry[1] : entry[2] + 1])
            blocks.append({"kind": "p", "text": text, "path": section})
        else:
            step = entry[3] if len(entry) > 3 else 1
            rows = list(lines[entry[1] : entry[2] + 1 : step])
            blocks.append({"kind": kind, "items": rows, "path": section})
    title = line_text(lines, outline["title"])
    return {"title": title, "headings": headings, "blocks": blocks}


def tokens(text: str) -> list[str]:
    """Words as W2 compares them: compatibility forms and case folded, runs of
    letters and digits (punctuation and typography do not count)."""
    return WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def squash(text: str) -> str:
    """A text's letters and digits only, for finding one text inside another."""
    return "".join(tokens(text))


def reading_of(saved: Mapping[str, Any]) -> dict[str, Any]:
    """A saved web reading (`pages`) as W2 scores it: its title, its headings, and its
    other elements as sections, consecutive elements under one path forming one."""
    headings = []
    sections: list[dict[str, Any]] = []
    for element in saved["elements"]:
        if element["kind"] == "heading":
            headings.append(element["text"])
            continue
        if not sections or sections[-1]["path"] != element["section"]:
            sections.append({"path": element["section"], "texts": []})
        sections[-1]["texts"].append(element["text"])
    return {"title": saved["title"], "headings": headings, "sections": sections}


def holder(text: str, sections: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The section holding a truth block's text (letters and digits compared), else
    the one sharing most of its words, else None."""
    target = squash(text)
    for section in sections:
        if target in squash(" ".join(section["texts"])):
            return section
    wanted = Counter(tokens(text))
    shared = [
        sum((wanted & Counter(tokens(" ".join(section["texts"])))).values())
        for section in sections
    ]
    best = max(shared, default=0)
    return sections[shared.index(best)] if best else None


def in_order(parts: list[str], text: str) -> bool:
    """Whether every part occurs in `text`, each after the one before."""
    position = 0
    for part in parts:
        found = text.find(part, position)
        if found < 0:
            return False
        position = found + len(part)
    return True


def page_score(page: Mapping[str, Any], reading: Mapping[str, Any]) -> dict[str, int]:
    """One page's counts for W2's measures (X42 report)."""
    blocks = page["blocks"]
    truth_text = [page["title"], *(heading["text"] for heading in page["headings"])]
    for block in blocks:
        if block["kind"] == "list":
            truth_text += block["items"]
        elif block["kind"] != "alt":
            truth_text.append(block["text"])
    alts = [block["text"] for block in blocks if block["kind"] == "alt"]
    sections = reading["sections"]
    kept = [
        reading["title"],
        *reading["headings"],
        *(text for section in sections for text in section["texts"]),
    ]
    truth = Counter(tokens(" ".join(truth_text)))
    words = Counter(tokens(" ".join(kept)))
    allowed = truth + Counter(tokens(" ".join(alts)))
    named = Counter(squash(heading["text"]) for heading in page["headings"])
    found_headings = named & Counter(squash(heading) for heading in reading["headings"])
    tested = placed = 0
    for block in blocks:
        if block["kind"] not in ("p", "list") or not block["path"]:
            continue
        tested += 1
        held = holder(block.get("text") or " ".join(block["items"]), sections)
        above = {squash(heading) for heading in held["path"]} if held else set()
        placed += {squash(heading) for heading in block["path"]} <= above
    lists = [block["items"] for block in blocks if block["kind"] == "list"]
    whole = sum(
        any(
            in_order([squash(item) for item in items], squash(" ".join(s["texts"])))
            for s in sections
        )
        for items in lists
    )
    everything = squash(" ".join(kept))
    return {
        "words": sum(truth.values()),
        "found": sum((truth & words).values()),
        "kept": sum(words.values()),
        "right": sum((allowed & words).values()),
        "headings": sum(named.values()),
        "headings_found": sum(found_headings.values()),
        "tested": tested,
        "placed": placed,
        "lists": len(lists),
        "lists_whole": whole,
        "alts": len(alts),
        "alts_kept": sum(squash(alt) in everything for alt in alts),
    }


def rate(part: int, whole: int) -> float:
    """A share, 1.0 when there is nothing to measure."""
    return part / whole if whole else 1.0


def scores(
    truth: Mapping[str, Any], readings: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Every page's counts, their totals, and W2's rates over the totals."""
    pages = {name: page_score(page, readings[name]) for name, page in truth.items()}
    total: Counter[str] = Counter()
    for counts in pages.values():
        total.update(counts)
    rates = {
        "recall": rate(total["found"], total["words"]),
        "precision": rate(total["right"], total["kept"]),
        "headings": rate(total["headings_found"], total["headings"]),
        "section_paths": rate(total["placed"], total["tested"]),
        "lists": rate(total["lists_whole"], total["lists"]),
        "alts": rate(total["alts_kept"], total["alts"]),
    }
    return {"rates": rates, "total": dict(total), "pages": pages}
