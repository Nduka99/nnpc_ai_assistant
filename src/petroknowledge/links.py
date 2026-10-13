"""The link tier: often-updated pages kept as links and read live when answering.

Only metadata is kept, in data/links/<company>.json: the URL, title, the page's own
description, its date, where the link came from and when it was seen. A company's
posts come from its link sitemaps (each post read once for its metadata, its body
never stored; read again when its sitemap lastmod changes or the metadata reader
improves). News items come from the shared RSS feeds and are kept for every
company whose name or alias they contain.
"""

import json
import re
from collections.abc import Sequence
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

from petroknowledge import config, fetch, nuxt
from petroknowledge.fetch import Record, Robots, local_name
from petroknowledge.sources import Company

# The version of `page_metadata`: a post read by an older one is read again.
# 2: the date from the page's Nuxt data when its markup states none.
READER = 2


def meta_content(soup: BeautifulSoup, attribute: str, value: str) -> str:
    tag = soup.find("meta", attrs={attribute: value})
    return " ".join(str(tag.get("content", "")).split()) if tag else ""


def page_metadata(raw_html: str) -> Record:
    """A page's own title, description and publication date (empty when absent)."""
    soup = BeautifulSoup(raw_html, "html.parser")
    title = meta_content(soup, "property", "og:title")
    if not title and soup.title:
        title = " ".join(soup.title.get_text().split())
    description = meta_content(soup, "name", "description") or meta_content(
        soup, "property", "og:description"
    )
    published = meta_content(soup, "property", "article:published_time")
    if not published:
        time_tag = soup.find("time", attrs={"datetime": True})
        published = str(time_tag["datetime"]) if time_tag else ""
    if not published:
        published = nuxt.page_date(raw_html)
    return {"title": title, "description": description, "published": published}


def plain_text(html: str) -> str:
    return " ".join(BeautifulSoup(html, "html.parser").get_text(" ").split())


def iso_date(rfc822: str) -> str:
    """An RSS date as ISO 8601, or empty when it cannot be read."""
    try:
        return parsedate_to_datetime(rfc822).isoformat()
    except (TypeError, ValueError):
        return ""


def feed_items(xml: str) -> list[Record]:
    """Each RSS 2.0 item: title, link, date and description as plain text."""
    items = []
    for element in ElementTree.fromstring(xml).iter():
        if local_name(element.tag) != "item":
            continue
        fields = {
            local_name(child.tag): (child.text or "").strip() for child in element
        }
        if not fields.get("link"):
            continue
        items.append(
            {
                "url": fields["link"],
                "title": plain_text(fields.get("title", "")),
                "description": plain_text(fields.get("description", "")),
                "published": iso_date(fields.get("pubDate", "")),
            }
        )
    return items


def mentions(text: str, aliases: Sequence[str]) -> bool:
    """Whether the text names one of the aliases as a whole word (any case)."""
    return any(
        re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE) for alias in aliases
    )


def read_registry(company_id: str) -> list[Record]:
    path = config.LINK_REGISTRY / f"{company_id}.json"
    if not path.exists():
        return []
    entries: list[Record] = json.loads(path.read_text(encoding="utf-8"))
    return entries


def write_registry(company_id: str, entries: list[Record]) -> None:
    config.LINK_REGISTRY.mkdir(parents=True, exist_ok=True)
    text = json.dumps(entries, indent=1, ensure_ascii=False) + "\n"
    (config.LINK_REGISTRY / f"{company_id}.json").write_text(text, encoding="utf-8")


def merged(existing: list[Record], found: list[Record]) -> list[Record]:
    """The registry with new links added and changed ones updated, by URL; a link
    keeps the time it was first seen."""
    by_url = {entry["url"]: entry for entry in existing}
    for entry in found:
        first_seen = by_url.get(entry["url"], entry)["first_seen"]
        by_url[entry["url"]] = {**entry, "first_seen": first_seen}
    return list(by_url.values())


def collect_posts(
    company: Company, transport: httpx.BaseTransport | None = None
) -> tuple[int, list[Record]]:
    """Read the metadata of every post a link sitemap lists that is new or has a
    new lastmod; returns how many were added or updated, and the failures."""
    known_posts = {entry["url"]: entry for entry in read_registry(company["id"])}
    found: list[Record] = []
    failures: list[Record] = []
    known: Robots = {}
    with fetch.polite_client(transport) as client:
        for sitemap in company["link_sitemaps"]:
            xml = fetch.get_text(client, known, sitemap)
            if isinstance(xml, dict):
                failures.append(xml)
                continue
            for entry in fetch.sitemap_entries(xml):
                url, lastmod = entry["loc"], entry.get("lastmod", "")
                seen = known_posts.get(url)
                if (
                    seen
                    and seen.get("listed_lastmod") == lastmod
                    and seen.get("reader") == READER
                ):
                    continue
                raw = fetch.get_text(client, known, url)
                if isinstance(raw, dict):
                    failures.append(raw)
                    continue
                stamp = fetch.now()
                found.append(
                    {
                        "url": url,
                        **page_metadata(raw),
                        "listed_lastmod": lastmod,
                        "source": "company site",
                        "reader": READER,
                        "seen_at": stamp,
                        "first_seen": stamp,
                    }
                )
    # Read again before writing: a news run may have added links meanwhile.
    write_registry(company["id"], merged(read_registry(company["id"]), found))
    return len(found), failures


def collect_news(
    companies: Sequence[Company],
    feeds: Sequence[dict[str, str]],
    transport: httpx.BaseTransport | None = None,
) -> tuple[dict[str, int], list[Record]]:
    """Read every feed once and keep each item for the companies it names; returns
    how many items each company gained or updated, and the failures."""
    found: dict[str, list[Record]] = {company["id"]: [] for company in companies}
    failures: list[Record] = []
    known: Robots = {}
    with fetch.polite_client(transport) as client:
        for feed in feeds:
            xml = fetch.get_text(client, known, feed["url"])
            if isinstance(xml, dict):
                failures.append(xml)
                continue
            stamp = fetch.now()
            for item in feed_items(xml):
                text = f"{item['title']} {item['description']}"
                for company in companies:
                    if mentions(text, [company["name"], *company["aliases"]]):
                        entry = {**item, "source": feed["name"], "seen_at": stamp}
                        found[company["id"]].append({**entry, "first_seen": stamp})
    for company_id, items in found.items():
        write_registry(company_id, merged(read_registry(company_id), items))
    return {company_id: len(items) for company_id, items in found.items()}, failures
