"""K16's web gate (X42's method): pages drawn by stratum, rendered as a visitor who
reads all of a page sees it, judged as outlines, and readings scored. Every page,
person and word here is invented."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from evaluation import __main__ as cli
from evaluation import webpages
from petroknowledge import config

SHA = {name: name * 64 for name in "abcdefghij"}


def saved(url: str, kind: str = "page", origin: str = "html") -> dict[str, Any]:
    return {"url": url, "kind": kind, "origin": origin, "title": "T", "elements": []}


READINGS = {
    SHA["a"]: saved("https://nnpcgroup.com/who-we-are", origin="nuxt data"),
    SHA["b"]: saved("https://nnpcgroup.com/upstream"),
    SHA["c"]: saved("https://nnpcgroup.com/insights/a-post", kind="post"),
    SHA["d"]: saved("https://retail.nnpcgroup.com/"),
    SHA["e"]: saved("https://careers.nnpcgroup.com/"),
    SHA["f"]: saved("https://en.wikipedia.org/wiki/NNPC_Limited"),
    SHA["g"]: saved("https://www.nuprc.gov.ng/laws/regulations/gazetted"),
    SHA["h"]: saved("https://nnpcgroup.com/insights/another-post", kind="post"),
}


def test_each_reading_has_one_stratum_and_the_draw_fills_each_quota() -> None:
    assert [webpages.stratum(r) for r in READINGS.values()] == [
        "nuxt data", "page", "post", "retail", "careers", "wikipedia", "regulator",
        "post",
    ]  # fmt: skip
    quotas = {"post": 1, "page": 3, "careers": 1}

    drawn = webpages.draw(READINGS, quotas, 42, set())

    assert drawn == webpages.draw(READINGS, quotas, 42, set())  # seeded
    assert len(drawn) == 3 and drawn[1:] == [SHA["b"], SHA["e"]]  # pools run short
    assert webpages.draw(READINGS, {"post": 2}, 1, {SHA["c"]}) == [SHA["h"]]


class Route:
    def __init__(self, url: str) -> None:
        self.request = type("Request", (), {"url": url})()
        self.done = ""

    def fulfill(self, body: str, content_type: str) -> None:
        self.done = f"stored {len(body)}"

    def continue_(self) -> None:
        self.done = "loaded"

    def abort(self) -> None:
        self.done = "blocked"


def test_rendering_serves_the_store_loads_assets_and_blocks_cms_and_trackers() -> None:
    handle = webpages.router({"https://nnpcgroup.com/who-we-are": "<html></html>"})
    routes = [
        Route("https://nnpcgroup.com/who-we-are"),
        Route("https://nnpcgroup.com/_nuxt/entry.js"),
        Route("https://nnpc-web-cms.nnpcgroup.com/uploads/kida.png"),
        Route("https://staging-cms1977.nnpcgroup.com/uploads/sheet.xlsx"),
        Route("https://o4504.ingest.us.sentry.io/api/1/envelope/"),
        Route("https://notsentry.io/script.js"),  # a lookalike host, not a subdomain
    ]
    for route in routes:
        handle(route)

    assert [r.done for r in routes] == [
        "stored 13", "loaded", "blocked", "blocked", "blocked", "loaded"
    ]  # fmt: skip


def test_new_lines_are_those_beyond_the_earlier_ones_counted_as_a_multiset() -> None:
    assert webpages.new_lines("a\n b\nb\n\n", "b\na\nc\nb\nb") == ["c", "b"]
    assert webpages.new_lines("a", "a") == []


class Control:
    """A clickable element: `action` runs on a click; `visible` decides is_visible."""

    def __init__(
        self, text: str, action: Callable[[], None], visible: bool = True
    ) -> None:
        self.text = text
        self.action = action
        self.visible = visible

    def is_visible(self) -> bool:
        return self.visible

    def click(self) -> None:
        self.action()

    def inner_text(self) -> str:
        return self.text

    def scroll_into_view_if_needed(self) -> None:
        pass


class Answer:
    def __init__(self, shown: bool) -> None:
        self.shown = shown
        self.marked = False

    def is_visible(self) -> bool:
        return self.shown

    def evaluate(self, script: str) -> None:
        self.marked = True


class Term:
    """An accordion term: clicking it opens its answer, or (`opens` false) not."""

    def __init__(self, answer: Answer, opens: bool, visible: bool = True) -> None:
        self.answer = answer
        self.opens = opens
        self.visible = visible

    def is_visible(self) -> bool:
        return self.visible

    def locator(self, selector: str) -> Answer:
        return self.answer

    def click(self) -> None:
        self.answer.shown = self.opens


class Site:
    """A page with team tabs, profile cards opening a profile, and a slider."""

    def __init__(self, interactive: bool) -> None:
        self.interactive = interactive
        self.tab = "board"
        self.profile = ""
        self.slide = 1
        self.refused = 0
        self.terms = [
            Term(Answer(shown=False), opens=True),
            Term(Answer(shown=True), opens=True),
            Term(Answer(shown=False), opens=False),
            Term(Answer(shown=False), opens=True, visible=False),
        ]

    def text(self) -> str:
        lines = ["Who We Are", "Our Leadership"]
        if self.interactive:
            lines += ["Board Of Directors", "Senior Management Team"]
            lines += ["Ada Obi", "Chair"] if self.tab == "board" else ["Ike Eze", "CEO"]
            lines += [self.profile] if self.profile else []
            lines.append("Efficient" if self.slide == 1 else "Resilient")
        return "\n".join(lines)

    def set(self, **changes: Any) -> Callable[[], None]:
        return lambda: self.__dict__.update(changes)

    def controls(self, selector: str) -> list[Any]:
        if not self.interactive:
            return []
        if selector == webpages.ACCORDIONS[0][0]:
            return self.terms
        if selector in (terms for terms, _ in webpages.ACCORDIONS):
            return []
        if selector == webpages.TABS:
            return [
                Control("Board Of Directors", self.set(tab="board")),
                Control("Senior Management Team", self.set(tab="senior")),
            ]
        if selector == webpages.CARDS:
            hidden = Control("Hidden Person", self.set(profile="never"), visible=False)
            if self.tab == "board":
                return [
                    Control("Ada Obi\nChair", self.set(profile="Ada's story")),
                    hidden,
                ]
            return [Control("Ike Eze\nCEO", self.set(profile="Ike's story"))]
        if selector == webpages.CLOSE:
            return [Control("", self.set(profile=""), visible=bool(self.profile))]
        if selector == webpages.PROFILE:
            # The panel shows the person's name and role above the story.
            person = "Ada Obi\nChair" if self.tab == "board" else "Ike Eze\nCEO"
            panel = f"{person}\n\n{self.profile}"
            return [Control(panel, lambda: None, visible=bool(self.profile))]
        return [Control("", self.set(slide=1)), Control("", self.set(slide=2))]


class Page:
    def __init__(self, sites: dict[str, Site]) -> None:
        self.sites = sites
        self.site = Site(False)
        self.visited: list[tuple[str, str, int]] = []

    def route(self, pattern: str, handler: object) -> None:
        self.handler = handler

    def goto(self, url: str, wait_until: str, timeout: int) -> None:
        self.visited.append((url, wait_until, timeout))
        self.site = self.sites[url]

    def get_by_role(self, role: str, name: str) -> object:
        site = self.site
        refuse = Control("Reject All", lambda: setattr(site, "refused", 1))
        unseen = Control("Reject All", lambda: setattr(site, "refused", 9), False)
        return type("Found", (), {"all": lambda found: [refuse, unseen]})()

    def locator(self, selector: str) -> object:
        controls = self.site.controls(selector)
        return type("Found", (), {"all": lambda found: controls})()

    def evaluate(self, script: str) -> object:
        if "scrollHeight" in script:
            return 1000
        if "innerText" in script:
            return self.site.text()
        if "img[alt]" in script:
            return ["A refinery at dusk on the river", "logo",
                    "A refinery  at dusk on the river"]  # fmt: skip
        return None

    def wait_for_timeout(self, ms: int) -> None:
        pass

    def screenshot(self, path: str, full_page: bool) -> None:
        Path(path).write_bytes(b"png")

    def close(self) -> None:
        pass


class Browser:
    def __init__(self, sites: dict[str, Site]) -> None:
        self.page = Page(sites)
        self.options: dict[str, Any] = {}

    def new_page(self, viewport: dict[str, int], user_agent: str) -> Page:
        self.options = {"viewport": viewport, "user_agent": user_agent}
        return self.page

    def close(self) -> None:
        pass


def test_a_visitor_opens_every_tab_profile_and_slide() -> None:
    site = Site(True)
    page = Page({"u": site})
    page.goto("u", "load", 1)

    webpages.prepare(page)
    added = webpages.clicks(page)

    # Slides come first. The first slide and tab are already shown: their clicks add
    # nothing. Each profile is read and closed; the hidden card is never clicked.
    assert added == [
        webpages.REVEALED + "slide 2", "Resilient",
        webpages.REVEALED + "Ada Obi Chair", "Ada Obi", "Chair", "Ada's story",
        webpages.REVEALED + "Senior Management Team", "Ike Eze", "CEO",
        webpages.REVEALED + "Ike Eze CEO", "Ike Eze", "CEO", "Ike's story",
    ]  # fmt: skip
    assert site.refused == 1 and site.profile == ""
    # Only the hidden answer that opened on a click is shown with the others.
    assert [term.answer.marked for term in site.terms] == [True, False, False, False]
    assert webpages.label(Control(" \n", lambda: None)) == "a control"


def test_each_page_is_drawn_with_its_revealed_lines_and_alt_texts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "REQUEST_DELAY_SECONDS", 0)
    browser = Browser({"https://x/team": Site(True), "https://x/post": Site(False)})
    pages = [
        {"url": "https://x/team", "sha256": SHA["a"], "html": "<p>team</p>"},
        {"url": "https://x/post", "sha256": SHA["b"], "html": "<p>post</p>"},
    ]

    counts = webpages.render(pages, tmp_path / "out", browser)

    assert counts == {SHA["a"]: 20, SHA["b"]: 2}
    assert browser.options["user_agent"] == config.USER_AGENT
    assert browser.page.visited[0] == ("https://x/team", "load", webpages.LOAD_MS)
    text = (tmp_path / "out" / f"{SHA['a']}.txt").read_text(encoding="utf-8")
    # The page as first shown (the first tab's people), then what the clicks show.
    assert text.startswith("Who We Are\nOur Leadership\nBoard Of Directors\n")
    assert "Ada Obi\nChair\nEfficient\n" + webpages.REVEALED + "slide 2\n" in text
    assert text.endswith(webpages.REVEALED + "Ike Eze CEO\nIke Eze\nCEO\nIke's story\n")
    assert (tmp_path / "out" / f"{SHA['b']}.txt").read_text() == (
        "Who We Are\nOur Leadership\n"
    )
    alts = (tmp_path / "out" / f"{SHA['a']}.alts.txt").read_text(encoding="utf-8")
    assert alts == "A refinery at dusk on the river\n"
    assert (tmp_path / "out" / f"{SHA['a']}.png").read_bytes() == b"png"


SHOWN = "\n".join(
    [
        "Posted 9 May 2026",
        "  ",
        "Gas Week",
        "Projects",
        "AKK pipeline",
        "Share",
        "OB3 pipeline",
        "Progress",
        "Welding is complete.",
        "Read more",
    ]
)
OUTLINE = {
    "title": 1,
    "blocks": [
        ["meta", 0, 0],
        ["h", 2, 2],
        ["list", 3, 5, 2],
        ["h", 3, 6],
        ["p", 7, 7],
        ["alt", "Welders at work on a pipeline"],
        ["end", 2],
        ["p", 8, 8],
    ],
}


def test_an_outline_over_the_visible_lines_gives_blocks_with_section_paths() -> None:
    lines = webpages.visible_lines(SHOWN)

    truth = webpages.build_truth(OUTLINE, lines)

    assert len(lines) == 9 and webpages.outline_problems(OUTLINE, len(lines)) == []
    assert truth["title"] == "Gas Week"
    assert truth["headings"] == [
        {"level": 2, "text": "Projects"}, {"level": 3, "text": "Progress"}
    ]  # fmt: skip
    assert truth["blocks"] == [
        {"kind": "meta", "text": "Posted 9 May 2026", "path": []},
        {"kind": "list", "items": ["AKK pipeline", "OB3 pipeline"],
         "path": ["Projects"]},
        {"kind": "p", "text": "Welding is complete.", "path": ["Projects", "Progress"]},
        {"kind": "alt", "text": "Welders at work on a pipeline",
         "path": ["Projects", "Progress"]},
        {"kind": "p", "text": "Read more", "path": []},
    ]  # fmt: skip


def test_a_split_heading_is_a_line_range_and_a_counter_s_digits_are_joined() -> None:
    lines = ["Energy", "efficiency and", "sustainability", "7", "0", "0", "0", "+",
             "Total workforce"]  # fmt: skip
    outline = {"title": [0, 2], "blocks": [["h", 2, [8, 8]], ["join", 3, 7]]}

    truth = webpages.build_truth(outline, lines)

    assert truth["title"] == "Energy efficiency and sustainability"
    assert truth["headings"] == [{"level": 2, "text": "Total workforce"}]
    assert truth["blocks"] == [
        {"kind": "p", "text": "7000+", "path": ["Total workforce"]}
    ]
    assert webpages.outline_problems(outline, len(lines)) == ["line 3 after line 8"]
    assert webpages.outline_problems({"title": [1, 2], "blocks": [["p", 2, 2]]}, 3) == [
        "title line 2 used again"
    ]
    # A heading whose line holds furniture too ("Backgroundedit") is written as text.
    written = {"title": 0, "blocks": [["h", 2, "Background"], ["p", 2, 2]]}
    truth = webpages.build_truth(written, ["Act", "Backgroundedit", "Passed."])
    assert truth["headings"] == [{"level": 2, "text": "Background"}]
    assert truth["blocks"] == [{"kind": "p", "text": "Passed.", "path": ["Background"]}]
    assert webpages.outline_problems(written, 3) == []


def test_outline_slips_are_reported() -> None:
    slipped = {"title": 9, "blocks": [["p", 3, 4], ["h", 2, 4], ["list", 9, 11]]}

    assert webpages.outline_problems(slipped, 10) == [
        "line 10 beyond the page",
        "line 11 beyond the page",
        "line 4 after line 4",
        "title line 9 used again",
    ]


def test_a_saved_reading_becomes_title_headings_and_sections_by_path() -> None:
    reading = {
        "title": "Gas Week",
        "elements": [
            {"kind": "heading", "text": "Projects", "level": 2, "section": []},
            {"kind": "list_item", "text": "AKK pipeline", "section": ["Projects"]},
            {"kind": "list_item", "text": "OB3 pipeline", "section": ["Projects"]},
            {"kind": "paragraph", "text": "Welding", "section": ["Progress"]},
            {"kind": "figure", "text": "Welders", "section": ["Progress"]},
        ],
    }

    assert webpages.reading_of(reading) == {
        "title": "Gas Week",
        "headings": ["Projects"],
        "sections": [
            {"path": ["Projects"], "texts": ["AKK pipeline", "OB3 pipeline"]},
            {"path": ["Progress"], "texts": ["Welding", "Welders"]},
        ],
    }


TRUTH_PAGE = {
    "title": "Gas Week",
    "headings": [{"level": 2, "text": "Projects"}, {"level": 3, "text": "Progress"}],
    "blocks": [
        {"kind": "meta", "text": "Posted 9 May", "path": []},
        {"kind": "p", "text": "Three lines.", "path": []},
        {"kind": "list", "items": ["AKK", "OB3"], "path": ["Projects"]},
        {"kind": "p", "text": "Welding is done.", "path": ["Projects", "Progress"]},
        {
            "kind": "p",
            "text": "Tests follow in June.",
            "path": ["Projects", "Progress"],
        },
        {"kind": "alt", "text": "Welders at work on a pipeline", "path": []},
    ],
}
READING = {
    "title": "Gas Week",
    "headings": ["Progress", "Extra"],
    "sections": [
        {"path": ["Gas Week"], "texts": ["Posted 9 May Three lines. AKK"]},
        {
            "path": ["Projects", "Progress"],
            "texts": ["OB3", "Welding is", "done. Menu"],
        },
    ],
}


def test_a_reading_is_scored_against_the_page_truth() -> None:
    found = webpages.page_score(TRUTH_PAGE, READING)

    # Words: 18 in the truth, 15 kept; "extra" and "menu" are not on the page. The
    # list is split between two sections and goes to the first (path "Gas Week":
    # wrong); "Welding is done." is held under both its headings; the June sentence
    # is lost.
    assert found == {
        "words": 18,
        "found": 13,
        "kept": 15,
        "right": 13,
        "headings": 2,
        "headings_found": 1,
        "tested": 3,
        "placed": 1,
        "lists": 1,
        "lists_whole": 0,
        "alts": 1,
        "alts_kept": 0,
    }
    scored = webpages.scores({"p": TRUTH_PAGE}, {"p": READING})
    assert scored["rates"]["recall"] == 13 / 18
    assert scored["rates"]["section_paths"] == 1 / 3
    assert webpages.rate(0, 0) == 1.0  # nothing to measure


def test_a_block_is_held_by_the_section_holding_its_text_or_most_of_its_words() -> None:
    sections = [
        {"path": ["Projects"], "texts": ["AKK runs"]},
        {"path": ["Progress"], "texts": ["runs north then south"]},
    ]

    assert webpages.holder("South", sections) == sections[1]
    assert webpages.holder("AKK runs north then", sections) == sections[1]
    assert webpages.holder("AKK runs north", sections) == sections[0]  # a tie
    assert webpages.holder("Ochre", sections) is None
    assert webpages.in_order(["a", "c"], "abc")
    assert not webpages.in_order(["c", "a"], "abc")


def test_the_command_line_draws_renders_judges_and_scores_a_web_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import playwright.sync_api

    root, registry = tmp_path / "eval", tmp_path / "sets.json"
    readings, store = tmp_path / "pages", tmp_path / "files"
    readings.mkdir()
    store.mkdir()
    for sha256, reading in READINGS.items():
        (readings / f"{sha256}.json").write_text(json.dumps(reading))
        (store / sha256).write_text("<p>stored</p>")
    (readings / "report-nnpc.json").write_text("{}")
    common = ["--root", str(root), "--registry", str(registry)]
    sample = [*common, "web-sample", "web", "--seed", "42", "--readings", str(readings)]

    assert cli.main(sample) == 0
    drawn = json.loads((root / "web" / "pages.json").read_text())["pages"]
    assert len(drawn) == 8 and drawn[0]["stratum"] == "nuxt data"
    assert cli.main(sample) == 1
    assert "drawn once" in capsys.readouterr().err
    held = [*common, "web-sample", "held", "--seed", "43", "--held-out-from", "web",
            "--readings", str(readings)]  # fmt: skip
    assert cli.main(held) == 0
    assert json.loads((root / "held" / "pages.json").read_text())["pages"] == []

    sites = {page["url"]: Site(False) for page in drawn}

    class Playwright:
        chromium = type("Chromium", (), {"launch": lambda self: Browser(sites)})()

        def __enter__(self) -> "Playwright":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    monkeypatch.setattr(config, "REQUEST_DELAY_SECONDS", 0)
    monkeypatch.setattr(playwright.sync_api, "sync_playwright", Playwright)
    assert cli.main([*common, "web-render", "web", "--store", str(store)]) == 0
    rendered = root / "web" / "rendered"
    assert len(list(rendered.glob("*.txt"))) == 16  # each page's text and alt texts

    outlines = root / "web" / "outlines"
    outlines.mkdir()
    (outlines / f"{SHA['a']}.json").write_text(json.dumps({"title": 5, "blocks": []}))
    assert cli.main([*common, "web-truth", "web"]) == 1
    assert "line 5 beyond the page" in capsys.readouterr().out
    outline = {"title": 0, "blocks": [["p", 1, 1]]}
    (outlines / f"{SHA['a']}.json").write_text(json.dumps(outline))
    assert cli.main([*common, "web-truth", "web"]) == 0
    assert cli.main([*common, "register", "web"]) == 0

    out = tmp_path / "score.json"
    score = [
        *common,
        "web-score",
        "web",
        "--out",
        str(out),
        "--readings",
        str(readings),
    ]
    assert cli.main(score) == 1  # the saved reading holds none of the page's words
    assert json.loads(out.read_text())["pass"] is False
    page = {"kind": "paragraph", "text": "Who We Are Our Leadership", "section": []}
    reading = READINGS[SHA["a"]] | {"title": "", "elements": [page]}
    (readings / f"{SHA['a']}.json").write_text(json.dumps(reading))
    assert cli.main(score) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "pass"
