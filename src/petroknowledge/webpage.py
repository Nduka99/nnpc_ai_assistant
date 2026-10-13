"""A web page read as a visitor reads it: the main content's elements in reading
order, each with the path of headings above it, as `petroknowledge.pdf` gives a PDF's
elements. Adopted from the Lime Green assistant (its experiment X42).

The text is laid out as a browser lays it out (the WHATWG `innerText` steps): runs
of text join inside inline elements, and a line ends where a block-level element
starts or ends and at `<br>`, so no text is lost because its tag was not expected.

Sites differ in where their content sits, so each host's reading rules are data
(companies/hosts.json, `sources.host_rules`): the main content's selector, site
furniture to remove, elements that serve as headings, and headings whose sections
are not content (a Wikipedia article's references). A host without rules is read
from <main> (or <body>) with standard furniture removed.
"""

import re
from typing import Any

from bs4 import BeautifulSoup, Tag
from bs4.element import Comment, NavigableString

Rules = dict[str, Any]
# What holds no readable text: scripts, styles and the like, and form controls.
NOT_TEXT = "script, style, noscript, svg, template, input, select, textarea"
# Furniture on every site: page chrome, forms (search and newsletter boxes) and what
# holds no text. A host whose portal wraps the whole content in a form says so
# (`forms_hold_content`, NNPC Retail's, web-v1); its forms are then read.
CHROME = "header, nav, footer"
STANDARD = f"{CHROME}, form, {NOT_TEXT}"
# Elements that start and end a line (block-level in browsers' default styles).
BLOCKS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "main",
    "ol",
    "p",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
HEADING_TAG = re.compile(r"^h([1-6])$")
HEADING_STYLE = re.compile(r"^h([1-6])-style$")  # a paragraph styled as a heading
BOLD = {"strong", "b"}
LEAD_IN = (":", ";")  # a line ending so introduces what follows: not a heading
ONE_LINE = 100  # characters: a heading fits on one line of the content column
SEPARATORS = (" | ", " - ", " – ")  # between a <title>'s page title and site name
LOOK = 3  # elements: a page shows its title, if anywhere, among its first ones
DESCRIBED = 5  # words: an image's alt text this long describes it (X42 truth rule)


def text_of(tag: Tag | None) -> str:
    return " ".join(tag.get_text(" ").split()) if tag else ""


def rendered_lines(root: Tag, every_image: bool = False) -> list[dict[str, Any]]:
    """The text as a browser lays it out: each line with the block element that holds
    it, whether all of it is bold, and images as lines of their own with the file
    they show: those with a describing alt text, or with `every_image` all of them."""
    lines: list[dict[str, Any]] = []
    runs: list[tuple[str, bool]] = []

    def end_line(block: Tag) -> None:
        text = " ".join("".join(run for run, _ in runs).split())
        if text:
            bold = all(heavy for run, heavy in runs if run.strip())
            lines.append({"text": text, "block": block, "bold": bold, "figure": False})
        runs.clear()

    def walk(node: Tag, block: Tag, bold: bool) -> None:
        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, NavigableString):
                runs.append((str(child), bold))
            elif isinstance(child, Tag) and child.name == "br":
                end_line(block)
            elif isinstance(child, Tag) and child.name == "img":
                alt = " ".join(str(child.get("alt") or "").split())
                source = str(child.get("data-src") or child.get("src") or "")
                if source and (every_image or len(alt.split()) >= DESCRIBED):
                    end_line(block)
                    lines.append({"text": alt, "block": block, "bold": False,
                                  "figure": True, "image": source})  # fmt: skip
            elif isinstance(child, Tag) and child.name in BLOCKS:
                end_line(block)
                walk(child, child, False)
                end_line(child)
            elif isinstance(child, Tag):
                walk(child, block, bold or child.name in BOLD)

    walk(root, root, False)
    end_line(root)
    return lines


def tagged_level(block: Tag) -> int | None:
    """A heading's level from its tag (h1-h6), or from a heading style on another
    element (some sites style paragraphs as headings)."""
    tag = HEADING_TAG.match(block.name)
    if tag:
        return int(tag[1])
    for name in block.get("class") or []:
        style = HEADING_STYLE.match(name)
        if style:
            return int(style[1])
    return None


def item_list(block: Tag) -> Tag | None:
    """The list a line belongs to: its own <li>'s list, or None."""
    item = block if block.name == "li" else block.find_parent("li")
    return item.find_parent(["ul", "ol"]) if item else None


def elements(
    lines: list[dict[str, Any]], title_block: list[Tag]
) -> list[dict[str, Any]]:
    """Lines as elements, each with the headings above it. A tagged heading takes its
    level; a definition term, or a bold line that fits on one line and does not end
    as a lead-in, sits one level below the last tagged heading. Lines of the title
    block are metadata."""
    found: list[dict[str, Any]] = []
    path: list[tuple[int, str]] = []
    base = 1  # the last tagged heading's level (the title is level 1)
    for line in lines:
        block = line["block"]
        text = line["text"]
        section = [heading for _, heading in path]
        level = tagged_level(block)
        if any(tag is part for tag in [block, *block.parents] for part in title_block):
            found.append({"kind": "meta", "text": text, "section": section})
            continue
        if line["figure"]:
            figure = {"kind": "figure", "text": text, "section": section}
            found.append(figure | {"image": line["image"]})
            continue
        derived = block.name == "dt" or (
            line["bold"]
            and len(text) <= ONE_LINE
            and not text.endswith(LEAD_IN)
            and not item_list(block)
        )
        if level is None and derived:
            level = base + 1
        elif level is not None:
            base = level
        if level is not None:
            path = [(n, heading) for n, heading in path if n < level]
            parents = [heading for _, heading in path]
            found.append(
                {"kind": "heading", "text": text, "level": level, "section": parents}
            )
            path.append((level, text))
            continue
        kind = "list_item" if item_list(block) else "paragraph"
        found.append({"kind": kind, "text": text, "section": section})
    return found


def without_stopped(
    found: list[dict[str, Any]], stops: list[str]
) -> list[dict[str, Any]]:
    """The elements outside sections that are not content (references, links)."""
    stopped = {stop.casefold() for stop in stops}
    kept = []
    for element in found:
        path = [heading.casefold() for heading in element["section"]]
        if element["kind"] == "heading":
            path.append(element["text"].casefold())
        if not stopped.intersection(path):
            kept.append(element)
    return kept


def titled(title: str, found: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """The page's title read once. A page without an h1 shows its title as some other
    element: one of its first LOOK elements whose text equals the title, or that the
    title holds beside a separator with the site's name ("NNPC Careers - How To
    Apply"). That text is the title, and the element is not read again (web-v1)."""
    folded = " ".join(title.split()).casefold()
    for index, element in enumerate(found[:LOOK]):
        text = " ".join(element["text"].split()).casefold()
        named = any(
            folded.endswith(separator + text) or folded.startswith(text + separator)
            for separator in SEPARATORS
        )
        if text and (folded == text or named):
            return element["text"], found[:index] + found[index + 1 :]
    return title, found


def read_page(
    raw_html: str, rules: Rules, every_image: bool = False
) -> tuple[str, list[dict[str, Any]]]:
    """The page's title and its main content's elements, by the host's rules. The
    title is the content's h1, else the page's h1 elsewhere (Wikipedia's title bar),
    else the element that shows the <title> (`titled`)."""
    soup = BeautifulSoup(raw_html, "html.parser")
    root = soup.select_one(rules["main"]) if rules.get("main") else None
    root = root or soup.find("main") or soup.body
    if root is None:
        raise ValueError("page has no <body>")
    standard = f"{CHROME}, {NOT_TEXT}" if rules.get("forms_hold_content") else STANDARD
    furniture = ", ".join(part for part in (standard, rules.get("furniture")) if part)
    for element in root.select(furniture):
        element.decompose()
    if rules.get("headings"):
        for element in root.select(rules["headings"]):
            element.name = "h2"  # a section's heading, below the page title
    heading = root.find("h1")
    shown = heading or soup.find("h1")
    title = text_of(shown) if shown else text_of(soup.title)
    if heading:
        heading.decompose()  # the title is the page's, not a section's
    found = elements(rendered_lines(root, every_image), [])
    found = without_stopped(found, rules.get("stop_headings", []))
    return (title, found) if shown else titled(title, found)
