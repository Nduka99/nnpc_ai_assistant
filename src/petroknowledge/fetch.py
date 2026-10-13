"""Collect public sources politely into a content-addressed store.

Every request identifies itself, obeys the host's robots.txt (and its crawl-delay)
and waits at least the polite delay before the next request to that host. Each file
is stored once, named by its SHA-256, and manifest.json records its URL, company,
kind, size, type, validators (ETag, Last-Modified) and when it was fetched. A failed
URL is reported and the run goes on; the next run tries it again. Nothing collected
is committed: data/ is git-ignored.
"""

import hashlib
import json
import os
import re
import shutil
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

from petroknowledge import config, nuxt
from petroknowledge.sources import host_rules, is_allowed

Record = dict[str, Any]  # one URL's outcome, plain JSON-compatible data
Robots = dict[str, RobotFileParser]  # rules already read, by host root
PICTURE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".gif")
WEB_KINDS = ("page", "post")  # stored web pages: site pages and the company's posts
# A CSS background picture: background: url(...) or background-image: url(...).
BACKGROUND = re.compile(r"background(?:-image)?\s*:[^;\"']*?url\(\s*['\"]?([^'\")]+)")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def polite_client(
    transport: httpx.BaseTransport | None = None, hosts: list[str] | None = None
) -> httpx.Client:
    """The one HTTP client for every request: identifying agent, timeout, retries.
    Given `hosts`, every request, redirects included, must stay on them: one that
    would leave them fails before it is sent."""

    def stay_on_hosts(request: httpx.Request) -> None:
        if hosts is not None and not is_allowed(str(request.url), hosts):
            raise httpx.RequestError(f"off the allowed hosts: {request.url}")

    return httpx.Client(
        headers={"User-Agent": config.USER_AGENT},
        timeout=config.REQUEST_TIMEOUT_SECONDS,
        transport=transport or httpx.HTTPTransport(retries=config.REQUEST_RETRIES),
        event_hooks={"request": [stay_on_hosts]},
    )


def host_root(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}/"


def load_robots(client: httpx.Client, root: str) -> RobotFileParser:
    """A host's robots.txt rules, following RFC 9309: redirects are followed, a 4xx
    response means no rules, and an unreachable file or a server error means the
    whole host is disallowed."""
    url = root + "robots.txt"
    robots = RobotFileParser(url)
    try:
        response = client.get(url, follow_redirects=True)
    except httpx.HTTPError:
        robots.parse(["User-agent: *", "Disallow: /"])
        return robots
    if response.status_code == 200:
        robots.parse(response.text.splitlines())
    elif 400 <= response.status_code < 500:
        robots.parse([])
    else:
        robots.parse(["User-agent: *", "Disallow: /"])
    return robots


def robots_for(client: httpx.Client, url: str, known: Robots) -> RobotFileParser:
    """The URL's host rules, read once per host (then the polite delay)."""
    root = host_root(url)
    if root not in known:
        known[root] = load_robots(client, root)
        time.sleep(config.REQUEST_DELAY_SECONDS)
    return known[root]


def delay_for(robots: RobotFileParser) -> float:
    """The wait after a request to this host: its crawl-delay if longer."""
    asked = robots.crawl_delay(config.USER_AGENT)
    return max(config.REQUEST_DELAY_SECONDS, float(asked or 0))


def local_name(tag: str) -> str:
    """An XML tag without its namespace: "{http://...}loc" -> "loc"."""
    return tag.rsplit("}", 1)[-1]


def sitemap_entries(xml: str) -> list[dict[str, str]]:
    """Each page a sitemap lists: its loc and, when given, its lastmod. Python's
    expat (2.4.1 and later) limits entity expansion in untrusted XML."""
    entries = []
    for element in ElementTree.fromstring(xml).iter():
        if local_name(element.tag) != "url":
            continue
        entry = {}
        for child in element:
            name = local_name(child.tag)
            if name in ("loc", "lastmod") and child.text and child.text.strip():
                entry[name] = child.text.strip()
        if "loc" in entry:
            entries.append(entry)
    return entries


def get_text(client: httpx.Client, known: Robots, url: str) -> str | Record:
    """A small text resource (a sitemap or a feed), or the reason it failed."""
    robots = robots_for(client, url, known)
    if not robots.can_fetch(config.USER_AGENT, url):
        return {"url": url, "error": "disallowed by robots.txt"}
    try:
        response = client.get(url, follow_redirects=True)
    except httpx.HTTPError as error:
        return {"url": url, "error": str(error)}
    finally:
        time.sleep(delay_for(robots))
    if response.status_code != 200:
        return {"url": url, "error": f"HTTP {response.status_code}"}
    return response.text


def read_limited(response: httpx.Response, max_bytes: int) -> bytes | None:
    """The body, or None once it passes max_bytes (stopped early, never stored)."""
    stated = response.headers.get("content-length")
    if stated is not None and int(stated) > max_bytes:
        return None
    body = bytearray()
    for chunk in response.iter_bytes():
        body.extend(chunk)
        if len(body) > max_bytes:
            return None
    return bytes(body)


def store_path(sha256: str) -> Path:
    return config.FILE_STORE / sha256


def read_manifest() -> list[Record]:
    path = config.FILE_STORE / "manifest.json"
    if not path.exists():
        return []
    records: list[Record] = json.loads(path.read_text(encoding="utf-8"))
    return records


def write_manifest(records: list[Record]) -> None:
    config.FILE_STORE.mkdir(parents=True, exist_ok=True)
    text = json.dumps(records, indent=1, ensure_ascii=False) + "\n"
    (config.FILE_STORE / "manifest.json").write_text(text, encoding="utf-8")


def fetch_file(
    client: httpx.Client,
    robots: RobotFileParser,
    url: str,
    source: Record,
    max_bytes: int,
) -> Record:
    """Download one file into the store; its record, or the reason it failed."""
    record: Record = {**source, "url": url, "fetched_at": now()}
    if not robots.can_fetch(config.USER_AGENT, url):
        return {**record, "error": "disallowed by robots.txt"}
    try:
        with client.stream("GET", url, follow_redirects=True) as response:
            if response.status_code != 200:
                return {**record, "error": f"HTTP {response.status_code}"}
            body = read_limited(response, max_bytes)
    except httpx.HTTPError as error:
        return {**record, "error": str(error)}
    if body is None:
        return {**record, "error": f"larger than {max_bytes // 1_000_000} MB"}
    sha256 = hashlib.sha256(body).hexdigest()
    path = store_path(sha256)
    if not path.exists():  # the same bytes under another URL are stored once
        path.write_bytes(body)
    return {
        **record,
        "sha256": sha256,
        "bytes": len(body),
        "content_type": response.headers.get("content-type", ""),
        "etag": response.headers.get("etag", ""),
        "last_modified": response.headers.get("last-modified", ""),
    }


def acquire(
    wanted: Sequence[Record],
    hosts: list[str],
    max_bytes: int,
    transport: httpx.BaseTransport | None = None,
) -> list[Record]:
    """Download every wanted file whose URL is not yet in the manifest; return this
    run's records. Each wanted record holds a url plus what the manifest should
    keep with it (company, kind, licence). The manifest keeps successes only and is
    saved after each file, so a stopped run resumes."""
    manifest = read_manifest()
    stored = {record["url"] for record in manifest}
    missing = {item["url"]: item for item in wanted if item["url"] not in stored}
    results: list[Record] = []
    if not missing:
        return results
    config.FILE_STORE.mkdir(parents=True, exist_ok=True)
    known: Robots = {}
    with polite_client(transport, hosts) as client:
        for url, item in missing.items():
            robots = robots_for(client, url, known)
            source = {key: value for key, value in item.items() if key != "url"}
            record = fetch_file(client, robots, url, source, max_bytes)
            time.sleep(delay_for(robots))
            results.append(record)
            if "sha256" in record:
                manifest.append(record)
                write_manifest(manifest)
    return results


def measure(
    urls: Sequence[str], transport: httpx.BaseTransport | None = None
) -> list[Record]:
    """Each URL's size from a HEAD request, so a download can be approved first."""
    results: list[Record] = []
    known: Robots = {}
    with polite_client(transport) as client:
        for url in dict.fromkeys(urls):
            robots = robots_for(client, url, known)
            if not robots.can_fetch(config.USER_AGENT, url):
                results.append({"url": url, "error": "disallowed by robots.txt"})
                continue
            try:
                response = client.head(url, follow_redirects=True)
            except httpx.HTTPError as error:
                results.append({"url": url, "error": str(error)})
                continue
            finally:
                time.sleep(delay_for(robots))
            if response.status_code != 200:
                results.append({"url": url, "error": f"HTTP {response.status_code}"})
                continue
            size = response.headers.get("content-length")
            results.append({"url": url, "bytes": int(size) if size else None})
    return results


def linked_files(
    raw_html: str, base: str, hosts: list[str], media_base: str = ""
) -> tuple[list[str], list[str]]:
    """The documents and pictures a page links on the allowed hosts, as absolute
    URLs. A lazily loaded picture names its file in `data-src`. A client-built page's
    own uploads come from its Nuxt data, written relative to the host's media server
    (`media_base`, a host rule), or to the page when it has none."""
    soup = BeautifulSoup(raw_html, "html.parser")
    documents = []
    for link in soup.find_all("a", href=True):
        url = urljoin(base, str(link["href"]).strip())
        path = urlsplit(url).path.lower()
        if path.endswith(config.DOCUMENT_TYPES) and is_allowed(url, hosts):
            documents.append(url)
    sources = []
    for image in soup.find_all("img"):
        source = image.get("data-src") or image.get("src")
        if source:
            sources.append(str(source).strip())
    sources += BACKGROUND.findall(raw_html)  # banners set by CSS, not <img>
    pictures = []
    for source in sources:
        url = urljoin(base, source.strip())
        path = urlsplit(url).path.lower()
        if path.endswith(PICTURE_TYPES) and is_allowed(url, hosts):
            pictures.append(url)
    for source in nuxt.page_files(raw_html):
        url = urljoin(media_base or base, source.strip())
        path = urlsplit(url).path.lower()
        if not is_allowed(url, hosts):
            continue
        if path.endswith(config.DOCUMENT_TYPES):
            documents.append(url)
        elif path.endswith(PICTURE_TYPES):
            pictures.append(url)
    return list(dict.fromkeys(documents)), list(dict.fromkeys(pictures))


def stored_links(company: str, hosts: list[str]) -> tuple[list[str], list[str]]:
    """Documents and pictures linked from every stored page and post of the
    company."""
    documents: list[str] = []
    pictures: list[str] = []
    for record in read_manifest():
        if record.get("company") != company or record.get("kind") not in WEB_KINDS:
            continue
        raw = (
            store_path(record["sha256"]).read_bytes().decode("utf-8", errors="replace")
        )
        media_base = host_rules(record["url"]).get("media_base", "")
        found = linked_files(raw, record["url"], hosts, media_base)
        documents += found[0]
        pictures += found[1]
    return list(dict.fromkeys(documents)), list(dict.fromkeys(pictures))


def readable_name(record: Record) -> str:
    """The file name the URL ends with, decoded (spaces instead of %20)."""
    return unquote(urlsplit(str(record["url"])).path.rsplit("/", 1)[-1]) or "unnamed"


def browse() -> int:
    """Give every stored file a readable name under data/browse/<company>/<kind>/,
    as a hard link (no extra space) or a copy where links are not possible. Two
    different files with the same name get the start of their hash added. Returns
    how many names were made."""
    made = 0
    taken: dict[Path, str] = {}
    for record in read_manifest():
        folder = (
            config.FILE_STORE.parent / "browse" / record["company"] / record["kind"]
        )
        name = readable_name(record)
        target = folder / name
        if taken.get(target, record["sha256"]) != record["sha256"]:
            stem, dot, suffix = name.rpartition(".")
            short = record["sha256"][:8]
            name = f"{stem}-{short}{dot}{suffix}" if dot else f"{name}-{short}"
            target = folder / name
        taken[target] = str(record["sha256"])
        if target.exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.link(store_path(str(record["sha256"])), target)
        except OSError:
            shutil.copyfile(store_path(str(record["sha256"])), target)
        made += 1
    return made
