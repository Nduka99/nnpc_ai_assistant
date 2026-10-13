"""Page text a Nuxt site sends as data, read as elements.

A Nuxt page carries the data it was built from in <script id="__NUXT_DATA__">,
serialised with devalue: a flat JSON array in which containers hold the indices of
their values, and a list starting with a type name ("Reactive", "Ref", "Set"...)
wraps a value. Some sites build part of their text from this data in the browser,
so their HTML holds none of it. NNPC's "Who we are" page, for example, has about
5,000 words in its data and none in its HTML. This module rebuilds the data and
reads the page's own content from it, in the shape `webpage` gives: headings with
their level, paragraphs and list items, each with the headings above it.

Only the page's fetched content (`data`) is read, never site-wide state (menus,
prices, settings). Media records, timestamps, identifiers and button labels are not
text. Structured content has no layout, so the order is the content's own order.
"""

import json
import re
from typing import Any

from bs4 import BeautifulSoup

from petroknowledge import webpage

Element = dict[str, Any]
# devalue's wrappers around a value, and its markers for an empty one.
WRAPPERS = {"Reactive", "ShallowReactive", "Ref", "ShallowRef", "NuxtError", "Island"}
EMPTY = {"EmptyRef", "EmptyShallowRef"}
# Fields holding identifiers, timestamps or file details, not text.
NOT_TEXT = {"id", "documentId", "createdAt", "updatedAt", "publishedAt", "locale",
            "slug", "url", "hash", "ext", "mime", "provider", "formats", "size",
            "width", "height", "previewUrl", "provider_metadata"}  # fmt: skip
# Field names for interface text (a button's label, a link's words), not content.
INTERFACE = re.compile(r"(^|_)(cta|button|link)(_|$)", re.IGNORECASE)
# A field naming its record: a title, a heading, or a name (a person's, a unit's).
HEADING_FIELD = re.compile(r"(title|heading|name)$", re.IGNORECASE)
ONE_LINE = 120  # characters: a title or heading field longer than this is prose
MARKUP = re.compile(r"<(p|h[1-6]|ul|ol|li|br|div|strong)\b", re.IGNORECASE)
# A value that is only a date or a timestamp, and one with no letter or digit ("#").
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}([T ][\d:.]+(Z|[+-]\d{2}:?\d{2})?)?$")
WORDLIKE = re.compile(r"[^\W_]")
LABEL_WORDS = 3  # a separate entry holding only this many words is a label
# A collection whose records share this share of fields (and at least 3) with a
# sibling record repeats its kind: a post's "other posts", not the post itself.
SIMILAR = 0.5


def payload(raw_html: str) -> list[Any] | None:
    """The page's flattened Nuxt data, or None when it has none."""
    script = BeautifulSoup(raw_html, "html.parser").find("script", id="__NUXT_DATA__")
    if script is None or not script.string:
        return None
    try:
        values = json.loads(script.string)
    except json.JSONDecodeError:
        return None
    return values if isinstance(values, list) and values else None


def hydrate(values: list[Any]) -> Any:
    """The data devalue flattened, rebuilt from its first value; None when the data
    is malformed (an index that is not a number or points past the end)."""
    built: dict[int, Any] = {}

    def value_at(index: int) -> Any:
        if index < 0:  # devalue's constants: undefined, holes, NaN, infinities
            return None
        if index in built:
            return built[index]
        value = values[index]
        if isinstance(value, list) and value and isinstance(value[0], str):
            if value[0] in WRAPPERS:
                found = value_at(value[1])
            elif value[0] in EMPTY:
                found = None
            elif value[0] == "Set":
                found = [value_at(item) for item in value[1:]]
            else:  # a Date, Map or other typed value: not text a page shows
                found = None
        elif isinstance(value, list):
            found = [value_at(item) for item in value]
        elif isinstance(value, dict):
            found = {key: value_at(item) for key, item in value.items()}
        else:
            found = value
        built[index] = found
        return found

    try:
        return value_at(0)
    except (TypeError, IndexError, RecursionError):
        return None


def is_media(node: dict[str, Any]) -> bool:
    """A file record (a picture or document upload), not text."""
    return "mime" in node and "url" in node


def is_heading(key: str, text: str) -> bool:
    return bool(HEADING_FIELD.search(key)) and len(text) <= ONE_LINE


def is_text(text: str) -> bool:
    """Words a page shows: not a bare date or timestamp, not a symbol alone."""
    return bool(WORDLIKE.search(text)) and not TIMESTAMP.match(text)


def own_keys(node: dict[str, Any]) -> set[str]:
    return {key for key in node if key not in NOT_TEXT}


def is_label(node: dict[str, Any]) -> bool:
    """A separate content entry (it has its own document id) holding one short
    label and nothing else: a type or category, such as "Company News"."""
    if "documentId" not in node:
        return False
    fields = [
        value
        for key, value in node.items()
        if key not in NOT_TEXT and value is not None and not isinstance(value, bool)
    ]
    return (
        len(fields) == 1
        and isinstance(fields[0], str)
        and len(fields[0].split()) <= LABEL_WORDS
    )


def items_of(value: Any) -> list[dict[str, Any]]:
    """The records a collection holds: a list of records, or a record wrapping one
    list of them (such as {"data": [...], "meta": {...}})."""
    candidates: Any = value
    if isinstance(value, dict):
        lists = [item for item in value.values() if isinstance(item, list)]
        candidates = lists[0] if len(lists) == 1 else []
    if not isinstance(candidates, list):
        return []
    return [item for item in candidates if isinstance(item, dict)]


def related(node: dict[str, Any]) -> set[str]:
    """The keys of collections repeating a sibling record's kind: a post's "other
    posts" are separate content entries (each with its own document id) holding the
    post's own fields. A list of page components (no document ids) is the page's own
    content, however alike its items are."""
    records = [
        own_keys(value)
        for value in node.values()
        if isinstance(value, dict) and not is_media(value) and len(own_keys(value)) >= 3
    ]
    found = set()
    for key, value in node.items():
        items = items_of(value)
        if not items or "documentId" not in items[0]:
            continue
        keys = own_keys(items[0])
        for other in records:
            shared = keys & other
            if len(shared) >= 3 and len(shared) / len(keys | other) >= SIMILAR:
                found.add(key)
    return found


def block_text(node: Any) -> str:
    """The words of a rich-text block and its children (a child may be plain text)."""
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    own = str(node.get("text") or "")
    children = node.get("children") or []
    return " ".join([own, *(block_text(child) for child in children)]).strip()


def markup_elements(html: str, section: list[str]) -> list[Element]:
    """A rich-text field written as HTML, read as `webpage` reads a page's lines; a
    style or script inside it is not text (a CMS field holding CSS, web-v1)."""
    root = BeautifulSoup(f"<div>{html}</div>", "html.parser")
    for tag in root.select(webpage.NOT_TEXT):
        tag.decompose()
    found = webpage.elements(webpage.rendered_lines(root), [])
    for element in found:
        element["section"] = section + element["section"]
        if element["kind"] == "heading":
            element["level"] = len(element["section"]) + 1
    return found


class Reader:
    """Walks content in order, keeping the headings above each element and leaving
    out a paragraph already read under the same headings."""

    def __init__(self) -> None:
        self.found: list[Element] = []
        self.seen: set[tuple[str, tuple[str, ...]]] = set()

    def add(self, kind: str, text: str, section: list[str]) -> None:
        """An element under `section`; a heading's level is one below it."""
        key = (text, tuple(section))
        if not text or (kind != "heading" and key in self.seen):
            return
        self.seen.add(key)
        element: Element = {"kind": kind, "text": text, "section": list(section)}
        if kind == "heading":
            element["level"] = len(section) + 1
        self.found.append(element)

    def walk(self, node: Any, section: list[str]) -> None:
        if isinstance(node, list):
            path = section
            for item in node:  # a heading block heads the blocks after it
                path = self.walk_item(item, section, path)
        elif isinstance(node, dict) and not is_media(node):
            self.walk_item(node, section, section)

    def walk_item(self, node: Any, section: list[str], path: list[str]) -> list[str]:
        """One item of a list; returns the path for the items after it."""
        if isinstance(node, dict) and "type" in node and "children" in node:
            return self.block(node, section, path)
        if isinstance(node, dict) and not is_media(node) and not is_label(node):
            self.record(node, path)
        elif isinstance(node, list):
            self.walk(node, path)
        return path

    def block(
        self, node: dict[str, Any], section: list[str], path: list[str]
    ) -> list[str]:
        """A rich-text block: a heading (returning the path it starts), a list item
        or a paragraph."""
        words = " ".join(block_text(node).split())
        if node["type"] == "heading":
            self.add("heading", words, section)
            return section + [words]
        if "list" in str(node["type"]):
            self.add("list_item", words, path)
        else:
            self.add("paragraph", words, path)
        return path

    def record(self, node: dict[str, Any], section: list[str]) -> None:
        """A content record. Its first heading field names its section, wherever
        the field stands; a later heading field names a subsection within it. Every
        other field goes under the latest heading. Collections of related records
        (a post's other posts) are skipped."""
        skipped = related(node)
        fields = [
            (key, value)
            for key, value in node.items()
            if key not in NOT_TEXT and key not in skipped and not INTERFACE.search(key)
        ]
        title = ""
        for key, value in fields:
            if isinstance(value, str) and is_heading(key, " ".join(value.split())):
                title = " ".join(value.split())
                break
        path = section
        if title:
            self.add("heading", title, section)
            path = section + [title]
        for key, value in fields:
            if isinstance(value, dict | list):
                self.walk(value, path)
                continue
            if not isinstance(value, str):
                continue
            text = " ".join(value.split())
            if text == title or not is_text(text):
                continue
            if MARKUP.search(value):
                self.found += markup_elements(value, path)
            elif is_heading(key, text) and title:
                self.add("heading", text, section + [title])
                path = section + [title, text]
            else:
                self.add("paragraph", text, path)


def page_elements(raw_html: str) -> list[Element]:
    """The page's own content from its Nuxt data, as elements; none without data."""
    values = payload(raw_html)
    if values is None:
        return []
    root = hydrate(values)
    if not isinstance(root, dict) or not isinstance(root.get("data"), dict):
        return []
    reader = Reader()
    reader.walk(root["data"], [])
    return reader.found


def files_of(node: Any) -> list[str]:
    """The upload URLs in content: each media record's own `url`, the original file
    rather than a resized copy in its `formats`, skipping collections of related
    records (a post's other posts)."""
    found: list[str] = []
    if isinstance(node, list):
        for item in node:
            found += files_of(item)
    elif isinstance(node, dict) and is_media(node):
        found.append(str(node["url"]))
    elif isinstance(node, dict):
        skipped = related(node)
        for key, value in node.items():
            if key not in skipped:
                found += files_of(value)
    return found


def media_of(node: Any) -> list[dict[str, str]]:
    """Each media record in content with its alt text (its caption when it has
    none), skipping collections of related records, as `files_of` does."""
    found: list[dict[str, str]] = []
    if isinstance(node, list):
        for item in node:
            found += media_of(item)
    elif isinstance(node, dict) and is_media(node):
        alt = node.get("alternativeText") or node.get("caption") or ""
        found.append({"url": str(node["url"]), "alt": str(alt)})
    elif isinstance(node, dict):
        skipped = related(node)
        for key, value in node.items():
            if key not in skipped:
                found += media_of(value)
    return found


def page_media(raw_html: str) -> list[dict[str, str]]:
    """The uploads the page's own content shows, each once with its alt text, as
    written in its Nuxt data (relative to the CMS's media host)."""
    values = payload(raw_html)
    root = hydrate(values) if values else None
    if not isinstance(root, dict) or not isinstance(root.get("data"), dict):
        return []
    unique: dict[str, dict[str, str]] = {}
    for media in media_of(root["data"]):
        unique.setdefault(media["url"], media)
    return list(unique.values())


def page_files(raw_html: str) -> list[str]:
    """The files the page's own content uploads (pictures, documents), as written in
    its Nuxt data: a client-built page shows or links them with no <img> or <a> in
    its HTML. Strapi writes them relative to its media host."""
    values = payload(raw_html)
    root = hydrate(values) if values else None
    if not isinstance(root, dict) or not isinstance(root.get("data"), dict):
        return []
    return list(dict.fromkeys(files_of(root["data"])))


DATE_FIELD = re.compile(r"date$", re.IGNORECASE)  # an editor's date, not a timestamp
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def page_date(raw_html: str) -> str:
    """The page's own date from its Nuxt data: the shallowest field named "...date"
    holding an ISO date, so a post's date wins over its related posts' dates; empty
    when there is none."""
    values = payload(raw_html)
    root = hydrate(values) if values else None
    data = root.get("data") if isinstance(root, dict) else None
    level: list[dict[str, Any] | list[Any]] = [data] if data else []
    while level:
        deeper: list[dict[str, Any] | list[Any]] = []
        for node in level:
            items = node.items() if isinstance(node, dict) else enumerate(node)
            for key, value in items:
                if isinstance(value, str) and DATE_FIELD.search(str(key)):
                    if ISO_DATE.match(value):
                        return value
                elif isinstance(value, dict | list):
                    deeper.append(value)
        level = deeper
    return ""


def prose_words(found: list[Element]) -> int:
    """Words in paragraphs and list items."""
    kinds = ("paragraph", "list_item")
    return sum(len(e["text"].split()) for e in found if e["kind"] in kinds)
