"""The source registry: company files are checked before anything is collected."""

import json
from pathlib import Path

import pytest

from petroknowledge import sources
from tests.conftest import SITE, company, write_company


def test_a_valid_company_has_no_problems() -> None:
    assert sources.problems(company()) == []


def test_missing_or_mistyped_fields_are_named() -> None:
    found = company(max_file_mb="big")
    del found["aliases"]

    assert sources.problems(found) == [
        "aliases: expected list",
        "max_file_mb: expected int",
    ]


def test_core_urls_need_a_url_and_a_known_kind_on_an_allowed_host() -> None:
    found = company(
        core_urls=[
            "https://example.test/a.pdf",
            {"url": SITE + "b.pdf", "kind": "spreadsheet"},
            {"url": "https://elsewhere.test/c.pdf", "kind": "document"},
        ],
        link_sitemaps=["http://example.test/posts.xml"],
        own_hosts=["example.test", "elsewhere.test"],
    )

    assert sources.problems(found) == [
        "core_urls: each entry needs a url: 'https://example.test/a.pdf'",
        "core_urls: kind must be page or document: https://example.test/b.pdf",
        "own_hosts: not an allowed host: elsewhere.test",
        "not on an allowed host: http://example.test/posts.xml",
        "not on an allowed host: https://elsewhere.test/c.pdf",
    ]


def test_a_units_page_needs_a_url_on_an_allowed_host_and_a_section() -> None:
    page = {"url": SITE + "businesses", "section": "Our Businesses"}
    assert sources.problems(company(units_page=page)) == []

    assert sources.problems(company(units_page={"url": SITE + "b"})) == [
        "units_page: needs a url and a section"
    ]
    elsewhere = {"url": "https://elsewhere.test/b", "section": "Units"}
    assert sources.problems(company(units_page=elsewhere)) == [
        "not on an allowed host: https://elsewhere.test/b"
    ]


def test_subdomains_count_as_the_host_but_lookalikes_do_not() -> None:
    hosts = ["example.test"]

    assert sources.is_allowed("https://www.example.test/a", hosts)
    assert sources.is_allowed("https://cms.example.test/a", hosts)
    assert not sources.is_allowed("https://badexample.test/a", hosts)
    assert not sources.is_allowed("http://example.test/a", hosts)
    assert sources.is_allowed("https://example.test/a", ["www.example.test"])
    assert sources.is_allowed("https://WWW.Example.test/a", ["example.test"])


def test_loading_checks_the_file(tmp_path: Path) -> None:
    write_company(tmp_path, company())
    write_company(tmp_path, company("bad", name=3))

    assert sources.load("acme", tmp_path)["name"] == "Acme Limited"
    with pytest.raises(sources.SourceError, match="name: expected str"):
        sources.load("bad", tmp_path)
    with pytest.raises(sources.SourceError, match="no source file"):
        sources.load("missing", tmp_path)


def test_only_company_folders_with_sources_are_collected(tmp_path: Path) -> None:
    write_company(tmp_path, company("nnpc"))
    write_company(tmp_path, company("nlng"))
    (tmp_path / "dangote").mkdir()  # still being planned: no sources.json
    (tmp_path / "news.json").write_text("{}")

    assert sources.company_ids(tmp_path) == ["nlng", "nnpc"]


def test_host_rules_are_found_for_a_host_or_its_parent(tmp_path: Path) -> None:
    path = tmp_path / "hosts.json"
    rules = {"example.test": {"main": ".content"}, "cms.example.test": {"main": "#x"}}
    path.write_text(json.dumps({"hosts": rules}))

    assert sources.host_rules("https://www.example.test/a", path) == {
        "main": ".content"
    }
    assert sources.host_rules("https://news.example.test/a", path) == {
        "main": ".content"
    }
    assert sources.host_rules("https://cms.example.test/a", path) == {"main": "#x"}
    assert sources.host_rules("https://other.test/a", path) == {}


def test_the_real_host_rules_load() -> None:
    path = Path(__file__).parent.parent / "companies" / "hosts.json"

    assert "main" in sources.host_rules("https://en.wikipedia.org/wiki/NNPC", path)


def test_news_feeds_must_use_https(tmp_path: Path) -> None:
    path = tmp_path / "news.json"
    path.write_text(
        json.dumps({"feeds": [{"name": "A", "url": "https://a.test/feed/"}]})
    )
    assert sources.news_feeds(path) == [{"name": "A", "url": "https://a.test/feed/"}]

    path.write_text(
        json.dumps({"feeds": [{"name": "B", "url": "http://b.test/feed/"}]})
    )
    with pytest.raises(sources.SourceError, match="HTTPS"):
        sources.news_feeds(path)


def test_the_real_source_files_are_valid() -> None:
    repository = Path(__file__).parent.parent / "companies"

    for company_id in sources.company_ids(repository):
        assert sources.load(company_id, repository)["id"] == company_id
    assert len(sources.news_feeds(repository / "news.json")) >= 1
