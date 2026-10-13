"""Polite collection into the content-addressed store, on the invented web."""

import hashlib
import os
from pathlib import Path

import httpx
import pytest

from petroknowledge import config, fetch
from tests.conftest import PDF, SITE, SITEMAP

HOSTS = ["example.test"]
MEDIA = "https://cms.example.test"


def test_a_sitemap_gives_each_page_and_its_lastmod_when_known() -> None:
    assert fetch.sitemap_entries(SITEMAP) == [
        {"loc": "https://example.test/about", "lastmod": "2026-10-01"},
        {"loc": "https://example.test/contact"},
    ]


def test_robots_rules_follow_rfc_9309(web: list[str]) -> None:
    with fetch.polite_client() as client:
        rules = fetch.load_robots(client, SITE)
        missing = fetch.load_robots(client, "https://norules.test/")
        broken = fetch.load_robots(client, "https://broken.test/")
        unreachable = fetch.load_robots(client, "https://down.test/")

    agent = config.USER_AGENT
    assert rules.can_fetch(agent, SITE + "about")
    assert not rules.can_fetch(agent, SITE + "private/a")
    assert missing.can_fetch(agent, "https://norules.test/private/a")
    assert not broken.can_fetch(agent, "https://broken.test/about")
    assert not unreachable.can_fetch(agent, "https://down.test/about")


def test_a_longer_crawl_delay_is_obeyed(web: list[str]) -> None:
    with fetch.polite_client() as client:
        slow = fetch.load_robots(client, "https://slow.test/")
        normal = fetch.load_robots(client, SITE)

    assert fetch.delay_for(slow) == 5.0
    assert fetch.delay_for(normal) == config.REQUEST_DELAY_SECONDS


def test_small_texts_are_read_or_their_failure_given(web: list[str]) -> None:
    known: fetch.Robots = {}
    with fetch.polite_client() as client:
        found = fetch.get_text(client, known, SITE + "contact")
        private = fetch.get_text(client, known, SITE + "private/sitemap.xml")
        missing = fetch.get_text(client, known, SITE + "nothing")
        refused = fetch.get_text(client, known, SITE + "docs/boom.pdf")

    assert found == "<p>Contact us</p>"
    assert private == {
        "url": SITE + "private/sitemap.xml",
        "error": "disallowed by robots.txt",
    }
    assert missing == {"url": SITE + "nothing", "error": "HTTP 404"}
    assert refused == {"url": SITE + "docs/boom.pdf", "error": "refused"}
    assert web.count("GET example.test/robots.txt") == 1  # read once per host


def wanted(*names: str) -> list[fetch.Record]:
    return [
        {"url": SITE + name, "company": "acme", "kind": "document"} for name in names
    ]


def test_files_are_stored_once_by_hash_and_failures_reported(web: list[str]) -> None:
    names = ["docs/report.pdf", "docs/moved.pdf", "docs/gone.pdf", "docs/boom.pdf"]
    names += ["private/a.pdf", "docs/huge.pdf", "docs/endless.pdf", "docs/offsite.pdf"]

    results = fetch.acquire(wanted(*names), HOSTS, max_bytes=1000)

    sha256 = hashlib.sha256(PDF).hexdigest()
    by_url = {record["url"]: record for record in results}
    assert by_url[SITE + "docs/report.pdf"]["sha256"] == sha256
    assert by_url[SITE + "docs/moved.pdf"]["sha256"] == sha256  # redirect followed
    errors = {url: record.get("error") for url, record in by_url.items()}
    assert errors[SITE + "docs/gone.pdf"] == "HTTP 404"
    assert errors[SITE + "docs/boom.pdf"] == "refused"
    assert errors[SITE + "private/a.pdf"] == "disallowed by robots.txt"
    assert errors[SITE + "docs/huge.pdf"] == "larger than 0 MB"
    assert errors[SITE + "docs/endless.pdf"] == "larger than 0 MB"
    assert errors[SITE + "docs/offsite.pdf"] == (
        "off the allowed hosts: https://elsewhere.test/x.pdf"
    )
    assert "GET elsewhere.test/x.pdf" not in web  # stopped before it was sent
    assert sorted(path.name for path in config.FILE_STORE.iterdir()) == [
        sha256,
        "manifest.json",
    ]
    manifest = fetch.read_manifest()
    assert [record["url"] for record in manifest] == [
        SITE + "docs/report.pdf",
        SITE + "docs/moved.pdf",
    ]
    assert manifest[0]["company"] == "acme" and manifest[0]["kind"] == "document"
    assert manifest[0]["content_type"] == "application/pdf"
    assert manifest[0]["etag"] == '"v1"' and manifest[0]["last_modified"] == ""
    assert manifest[0]["bytes"] == len(PDF)


def test_a_second_run_fetches_only_what_failed(web: list[str]) -> None:
    fetch.acquire(wanted("docs/report.pdf", "docs/gone.pdf"), HOSTS, 1000)
    web.clear()

    retried = fetch.acquire(wanted("docs/report.pdf", "docs/gone.pdf"), HOSTS, 1000)

    assert web == ["GET example.test/robots.txt", "GET example.test/docs/gone.pdf"]
    assert [record["url"] for record in retried] == [SITE + "docs/gone.pdf"]
    web.clear()
    assert fetch.acquire(wanted("docs/report.pdf"), HOSTS, 1000) == []
    assert web == []  # nothing left to fetch: not even robots.txt


def test_measuring_sends_only_head_requests(web: list[str]) -> None:
    urls = [SITE + name for name in ["docs/report.pdf", "img/unsized.webp"]]
    urls += [
        SITE + name for name in ["docs/gone.pdf", "private/a.pdf", "docs/boom.pdf"]
    ]

    results = fetch.measure(urls)

    assert results == [
        {"url": SITE + "docs/report.pdf", "bytes": len(PDF)},
        {"url": SITE + "img/unsized.webp", "bytes": None},
        {"url": SITE + "docs/gone.pdf", "error": "HTTP 404"},
        {"url": SITE + "private/a.pdf", "error": "disallowed by robots.txt"},
        {"url": SITE + "docs/boom.pdf", "error": "refused"},
    ]
    assert [line for line in web if line.startswith("GET")] == [
        "GET example.test/robots.txt"
    ]


def test_only_documents_and_pictures_on_allowed_hosts_are_linked() -> None:
    from tests.conftest import ABOUT

    documents, pictures = fetch.linked_files(ABOUT, SITE + "about", HOSTS)

    assert documents == [
        SITE + "docs/report.pdf",
        "https://cdn.example.test/docs/b.PDF",
    ]
    assert pictures == [
        SITE + "img/a.png",
        SITE + "img/lazy.webp",
        SITE + "img/banner.jpg?v=1",
    ]


def test_a_client_built_pages_uploads_are_linked_from_its_media_server() -> None:
    from tests.test_nuxt import FILES, page

    post = page(FILES)

    assert fetch.linked_files(post, SITE + "insights/eoi", HOSTS, MEDIA) == (
        [MEDIA + "/uploads/eoi_cd34.pdf"],
        [MEDIA + "/uploads/flyer_ab12.png"],
    )
    assert fetch.linked_files(post, SITE + "insights/eoi", HOSTS)[0] == [
        SITE + "uploads/eoi_cd34.pdf"
    ]
    assert fetch.linked_files(post, SITE, ["other.test"]) == ([], [])


def test_links_come_from_the_companys_stored_pages_only(web: list[str]) -> None:
    pages = [
        {"url": SITE + "about", "company": "acme", "kind": "page"},
        {"url": SITE + "contact", "company": "acme", "kind": "page"},
    ]
    fetch.acquire(pages, HOSTS, 10_000)
    fetch.acquire(
        [{"url": SITE + "news/one", "company": "other", "kind": "page"}], HOSTS, 10_000
    )

    documents, pictures = fetch.stored_links("acme", HOSTS)

    assert documents == [
        SITE + "docs/report.pdf",
        "https://cdn.example.test/docs/b.PDF",
    ]
    assert len(pictures) == 3
    assert fetch.stored_links("nobody", HOSTS) == ([], [])


def test_browse_gives_stored_files_readable_names(web: list[str]) -> None:
    fetch.acquire(wanted("docs/report.pdf", "docs/b.PDF"), HOSTS, 10_000)
    fetch.acquire(
        [{"url": SITE + "about", "company": "acme", "kind": "page"}], HOSTS, 10_000
    )

    made = fetch.browse()

    folder = config.FILE_STORE.parent / "browse" / "acme"
    assert sorted(path.name for path in (folder / "document").iterdir()) == [
        "b.PDF",
        "report.pdf",
    ]
    assert (folder / "document" / "report.pdf").read_bytes() == PDF
    assert [path.name for path in (folder / "page").iterdir()] == ["about"]
    assert made == 3
    assert fetch.browse() == 0  # names already made are left alone


def test_a_name_clash_gets_the_hash_and_a_copy_is_made_without_links(
    web: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    fetch.write_manifest(
        [
            {
                "url": SITE + "one/",
                "company": "acme",
                "kind": "page",
                "sha256": "a" * 64,
            },
            {
                "url": SITE + "two/",
                "company": "acme",
                "kind": "page",
                "sha256": "b" * 64,
            },
            {
                "url": SITE + "1/r.pdf",
                "company": "acme",
                "kind": "document",
                "sha256": "c" * 64,
            },
            {
                "url": SITE + "2/r.pdf",
                "company": "acme",
                "kind": "document",
                "sha256": "d" * 64,
            },
        ]
    )
    for letter in "abcd":
        fetch.store_path(letter * 64).write_bytes(letter.encode())

    def no_links(source: Path, target: Path) -> None:
        raise OSError("links not supported")

    monkeypatch.setattr(os, "link", no_links)

    assert fetch.browse() == 4
    folder = config.FILE_STORE.parent / "browse" / "acme"
    assert sorted(path.name for path in (folder / "page").iterdir()) == [
        "unnamed",
        f"unnamed-{'b' * 8}",
    ]
    assert sorted(path.name for path in (folder / "document").iterdir()) == [
        f"r-{'d' * 8}.pdf",
        "r.pdf",
    ]


def test_the_client_identifies_itself() -> None:
    with fetch.polite_client(
        httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client:
        assert client.headers["User-Agent"] == config.USER_AGENT
    assert "https://" in config.USER_AGENT  # contact details, as hosts ask
