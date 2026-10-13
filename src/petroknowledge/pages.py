"""Collected web pages read into elements, with a report: the pages' counterpart of
`documents` for PDFs.

Each stored page of a company is read by its host's rules (`webpage`). When the
page's Nuxt data holds more than twice the prose of its HTML, the browser builds
text the HTML lacks, so the page is read from its data instead (`nuxt`). Near-equal
amounts are the same text, read from the HTML a visitor sees. A company's posts are
read the same way and keep their publication date. Each reading is saved in
data/pages/<sha256>.json with the page's URL, title, where its text came from and
its word count; the report flags pages that hold almost no text.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from petroknowledge import config, fetch, links, nuxt, sources, webpage
from petroknowledge.fetch import WEB_KINDS, Record

THIN = 50  # words of prose: a page with fewer is flagged
DATA_SHARE = 2  # data with more than this many times the HTML's prose is read


def read_stored(record: Record) -> dict[str, Any]:
    """One stored page read by its host's rules, or from its Nuxt data when that
    holds more than twice the HTML's prose."""
    raw = fetch.store_path(record["sha256"]).read_bytes().decode("utf-8", "replace")
    title, found = webpage.read_page(raw, sources.host_rules(record["url"]))
    origin = "html"
    data = nuxt.page_elements(raw)
    if nuxt.prose_words(data) > DATA_SHARE * nuxt.prose_words(found):
        title, found = webpage.titled(title, data)
        origin = "nuxt data"
    published = (
        links.page_metadata(raw)["published"] if record["kind"] == "post" else ""
    )
    return {
        "url": record["url"],
        "sha256": record["sha256"],
        "kind": record["kind"],
        "fetched_at": record["fetched_at"],
        "published": published,
        "title": title,
        "origin": origin,
        "words": nuxt.prose_words(found),
        "elements": found,
    }


def read_all(company_id: str, out: Path | None = None) -> dict[str, Any]:
    """Read every stored page and post of the company; the run's report."""
    out = out or config.PAGE_READINGS
    out.mkdir(parents=True, exist_ok=True)
    totals: Counter[str] = Counter()
    thin = []
    for record in fetch.read_manifest():
        if record.get("company") != company_id or record.get("kind") not in WEB_KINDS:
            continue
        reading = read_stored(record)
        text = json.dumps(reading, indent=1, ensure_ascii=False) + "\n"
        target = out / f"{record['sha256']}.json"
        target.write_text(text, encoding="utf-8", newline="\n")
        totals[f"{record['kind']}s"] += 1
        totals[f"from {reading['origin']}"] += 1
        totals["words"] += reading["words"]
        if reading["words"] < THIN:
            thin.append({"url": reading["url"], "words": reading["words"]})
    report = {"company": company_id, "totals": dict(totals), "thin": thin}
    text = json.dumps(report, indent=1, ensure_ascii=False) + "\n"
    (out / f"report-{company_id}.json").write_text(text, encoding="utf-8")
    return report
