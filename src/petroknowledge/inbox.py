"""Add files a person saved by hand, with their provenance.

Some hosts forbid crawlers in robots.txt while publishing documents for people to
read: NNPC's reports sit on such a host. The collector never fetches them. A person
may open those public links in a browser and save the files into data/inbox/<id>/,
each under the name it has at the end of its link. Adding them stores each file once
by SHA-256 with its original URL, the note "added by hand" and when it was added.
A file is accepted only if its name matches a document the company's sources list,
or a document or picture its stored pages link, so every added file has a known
public origin.

On request, `download_listed` fetches exactly the files a download
list names into the inbox, though their host's robots.txt asks crawlers to stay
away. It identifies itself as the collector does, waits five seconds between files
and crawls nothing. Its log (downloaded.json in the inbox folder) records each
file's URL and time, so adding the file records how it came.
"""

import hashlib
import json
import mimetypes
import time
from collections.abc import Sequence
from pathlib import Path

import httpx

from petroknowledge import config, fetch
from petroknowledge.fetch import Record, readable_name
from petroknowledge.sources import Company

LOG = "downloaded.json"
DOWNLOAD_DELAY_SECONDS = 5.0  # gentler than the crawler: these hosts ask bots away
BY_HAND = "hand"
BY_DOWNLOAD = "listed download (robots.txt disallows crawlers)"


def listed_urls(path: Path) -> list[str]:
    """The links a download list names, one per line; other lines are notes."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip().startswith("https://")]


def read_log(folder: Path) -> dict[str, Record]:
    path = folder / LOG
    if not path.exists():
        return {}
    log: dict[str, Record] = json.loads(path.read_text(encoding="utf-8"))
    return log


def download_one(client: httpx.Client, url: str, max_bytes: int) -> bytes | str:
    """One listed file's bytes, or the reason it could not be downloaded."""
    try:
        with client.stream("GET", url, follow_redirects=True) as response:
            if response.status_code != 200:
                return f"HTTP {response.status_code}"
            body = fetch.read_limited(response, max_bytes)
    except httpx.HTTPError as error:
        return str(error) or type(error).__name__
    if body is None:
        return f"larger than {max_bytes // 1_000_000} MB"
    return body


def download_listed(
    urls: Sequence[str],
    folder: Path,
    hosts: list[str],
    max_bytes: int,
    transport: httpx.BaseTransport | None = None,
) -> list[Record]:
    """Download each listed file not yet in the folder, saved under the name its
    link ends with; return a record per file tried."""
    folder.mkdir(parents=True, exist_ok=True)
    log = read_log(folder)
    results: list[Record] = []
    with fetch.polite_client(transport, hosts) as client:
        for url in dict.fromkeys(urls):
            name = readable_name({"url": url})
            if (folder / name).exists():
                continue
            found = download_one(client, url, max_bytes)
            time.sleep(DOWNLOAD_DELAY_SECONDS)
            if isinstance(found, str):
                results.append({"url": url, "error": found})
                continue
            (folder / name).write_bytes(found)
            log[name] = {"url": url, "downloaded_at": fetch.now()}
            (folder / LOG).write_text(
                json.dumps(log, indent=1) + "\n", encoding="utf-8"
            )
            results.append({"url": url, "bytes": len(found)})
    return results


def known_files(company: Company) -> dict[str, str]:
    """Each document the company's sources list, and each document or picture its
    stored pages link, by the file name at the end of its URL."""
    urls = [
        entry["url"] for entry in company["core_urls"] if entry["kind"] == "document"
    ]
    documents, pictures = fetch.stored_links(company["id"], company["allowed_hosts"])
    urls += documents + pictures
    return {readable_name({"url": url}): url for url in urls}


def add_folder(company: Company, folder: Path) -> list[Record]:
    """Store every file in the folder whose name matches a known document; return a
    record per file, or the reason it was not added."""
    if not folder.is_dir():
        return [{"url": str(folder), "error": "no such folder"}]
    known = known_files(company)
    log = read_log(folder)
    manifest = fetch.read_manifest()
    stored = {record["url"] for record in manifest}
    results: list[Record] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name == LOG:
            continue
        url = known.get(path.name)
        if url is None:
            results.append(
                {"url": path.name, "error": "no known link ends in this name"}
            )
            continue
        if url in stored:
            continue
        body = path.read_bytes()
        sha256 = hashlib.sha256(body).hexdigest()
        is_picture = path.suffix.lower() in fetch.PICTURE_TYPES
        if not fetch.store_path(sha256).exists():
            config.FILE_STORE.mkdir(parents=True, exist_ok=True)
            fetch.store_path(sha256).write_bytes(body)
        record = {
            "url": url,
            "company": company["id"],
            "kind": "picture" if is_picture else "document",
            "fetched_at": log.get(path.name, {}).get("downloaded_at") or fetch.now(),
            "added_by": BY_DOWNLOAD if path.name in log else BY_HAND,
            "sha256": sha256,
            "bytes": len(body),
            "content_type": mimetypes.guess_type(path.name)[0] or "",
        }
        manifest.append(record)
        fetch.write_manifest(manifest)
        results.append(record)
    return results
