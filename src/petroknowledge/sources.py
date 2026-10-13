"""The source registry: what to collect for each company, read from
companies/<id>/sources.json.

A company file is plain JSON:

    {
      "id": "nnpc",
      "name": "NNPC Limited",
      "aliases": ["NNPC", ...],          names a news item must contain to be tagged
      "terms": "...",                     the reuse terms, quoted or summarised
      "allowed_hosts": ["nnpcgroup.com", ...],
      "own_hosts": ["nnpcgroup.com"],     the company's own sites (subdomains too):
                                          searched first, as its own channel
      "max_file_mb": 120,                 larger files are skipped and reported
      "core_sitemaps": [...],             pages read and indexed
      "post_sitemaps": [...],             the company's own posts, read and indexed
                                          with their publication dates
      "core_urls": [{"url": ..., "kind": "page" | "document", "licence": ...}],
      "link_sitemaps": [...]              pages kept as links, fetched when answering
    }

companies/news.json lists the RSS feeds every company shares:
{"feeds": [{"name", "url"}]}.
"""

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from petroknowledge import config

Company = dict[str, Any]
REQUIRED = {
    "id": str,
    "name": str,
    "aliases": list,
    "terms": str,
    "allowed_hosts": list,
    "own_hosts": list,
    "max_file_mb": int,
    "core_sitemaps": list,
    "post_sitemaps": list,
    "core_urls": list,
    "link_sitemaps": list,
}
KINDS = ("page", "document")


class SourceError(ValueError):
    """A source file is missing or malformed; the message says what is wrong."""


def problems(company: Company) -> list[str]:
    """What is wrong with a company's sources, or nothing."""
    found = []
    for key, kind in REQUIRED.items():
        if not isinstance(company.get(key), kind):
            found.append(f"{key}: expected {kind.__name__}")
    if found:
        return found
    for entry in company["core_urls"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            found.append(f"core_urls: each entry needs a url: {entry!r}")
        elif entry.get("kind") not in KINDS:
            found.append(f"core_urls: kind must be page or document: {entry['url']}")
    for host in company["own_hosts"]:
        if not is_allowed(f"https://{host}/", company["allowed_hosts"]):
            found.append(f"own_hosts: not an allowed host: {host}")
    page = company.get("units_page")
    if page is not None and not (
        isinstance(page, dict)
        and isinstance(page.get("url"), str)
        and isinstance(page.get("section"), str)
    ):
        found.append("units_page: needs a url and a section")
        page = None
    for url in (
        company["core_sitemaps"]
        + company["post_sitemaps"]
        + company["link_sitemaps"]
        + [entry["url"] for entry in company["core_urls"] if isinstance(entry, dict)]
        + ([page["url"]] if page else [])
    ):
        if isinstance(url, str) and not is_allowed(url, company["allowed_hosts"]):
            found.append(f"not on an allowed host: {url}")
    return found


def is_allowed(url: str, hosts: list[str]) -> bool:
    """Whether the URL is on one of the hosts (or a subdomain of one), over HTTPS.
    "www." is ignored on both sides, so "www.example.com" and "example.com" agree."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        return False
    host = parts.netloc.lower().removeprefix("www.")
    for allowed in hosts:
        allowed = allowed.lower().removeprefix("www.")
        if host == allowed or host.endswith("." + allowed):
            return True
    return False


def load(company_id: str, folder: Path | None = None) -> Company:
    """A company's sources, checked."""
    path = (folder or config.COMPANIES_DIR) / company_id / "sources.json"
    if not path.exists():
        raise SourceError(f"no source file: {path}")
    company: Company = json.loads(path.read_text(encoding="utf-8"))
    found = problems(company)
    if found:
        raise SourceError(f"{path}: " + "; ".join(found))
    return company


def company_ids(folder: Path | None = None) -> list[str]:
    """Every company whose folder has a sources.json (a company still being planned
    has none, so it is not collected yet)."""
    folder = folder or config.COMPANIES_DIR
    return sorted(path.parent.name for path in folder.glob("*/sources.json"))


def host_rules(url: str, path: Path | None = None) -> dict[str, Any]:
    """How the URL's host is read (companies/hosts.json); none for a host without
    rules. A subdomain takes its parent's rules unless it has its own."""
    path = path or config.HOST_RULES
    hosts: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))["hosts"]
    host = urlsplit(url).netloc.lower().removeprefix("www.")
    while host:
        if host in hosts:
            found: dict[str, Any] = hosts[host]
            return found
        host = host.partition(".")[2]
    return {}


def news_feeds(path: Path | None = None) -> list[dict[str, str]]:
    """The shared news feeds, each a name and an HTTPS URL."""
    path = path or config.NEWS_SOURCES
    feeds: list[dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))["feeds"]
    for feed in feeds:
        if urlsplit(feed["url"]).scheme != "https":
            raise SourceError(f"{path}: feeds must use HTTPS: {feed['url']}")
    return feeds
