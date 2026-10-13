"""Reading a company's stored pages: by host rules, or from Nuxt data when the
browser builds most of the text. Pages invented."""

import json
from pathlib import Path

import pytest

from petroknowledge import cli, config, fetch, pages
from tests.conftest import company, write_sources

PROSE = " ".join(["word"] * 60)


def nuxt_page(html_text: str, data_text: str) -> str:
    """A page with some text in its HTML and other text in its Nuxt data."""
    values = [{"data": 1}, {"page": 2}, {"title": 3, "body": 4}, "From data", data_text]
    script = (
        '<script type="application/json" id="__NUXT_DATA__">'
        f"{json.dumps(values)}</script>"
    )
    return f"<html><body><main><p>{html_text}</p></main>{script}</body></html>"


def store(url: str, html: str, company_id: str = "acme", kind: str = "page") -> None:
    sha256 = str(abs(hash(url)))
    config.FILE_STORE.mkdir(parents=True, exist_ok=True)
    fetch.store_path(sha256).write_text(html, encoding="utf-8")
    records = fetch.read_manifest()
    fetched = "2026-10-10T10:00:00+00:00"
    records.append({"url": url, "company": company_id, "kind": kind,
                    "sha256": sha256, "fetched_at": fetched})  # fmt: skip
    fetch.write_manifest(records)


@pytest.fixture
def stored(web: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rules = tmp_path / "hosts.json"
    rules.write_text(json.dumps({"hosts": {}}), encoding="utf-8")
    monkeypatch.setattr(config, "HOST_RULES", rules)
    monkeypatch.setattr(config, "PAGE_READINGS", tmp_path / "pages")
    store("https://example.test/plain", nuxt_page(PROSE, "short"))
    store("https://example.test/built", nuxt_page("Loading", PROSE))
    store("https://example.test/empty", "<html><body><p>Hi</p></body></html>")
    store("https://example.test/report.pdf", "%PDF", kind="document")
    dated = '[{"data": 1}, {"post_date": 2}, "2026-10-09"]'
    post = (
        f'<html><body><main><p>{PROSE}</p></main><script id="__NUXT_DATA__">'
        f"{dated}</script></body></html>"
    )
    store("https://example.test/news/dated", post, kind="post")
    store("https://other.test/page", nuxt_page(PROSE, PROSE), company_id="other")


def test_pages_are_read_from_html_or_from_data_holding_more(stored: None) -> None:
    report = pages.read_all("acme")

    assert report["totals"] == {
        "pages": 3,
        "posts": 1,
        "from html": 3,
        "from nuxt data": 1,
        "words": 60 + 60 + 1 + 60,
    }
    assert report["thin"] == [{"url": "https://example.test/empty", "words": 1}]
    readings = {
        json.loads(path.read_text())["url"]: json.loads(path.read_text())
        for path in config.PAGE_READINGS.glob("*.json")
        if not path.name.startswith("report")
    }
    built = readings["https://example.test/built"]
    assert built["origin"] == "nuxt data"
    assert built["elements"][0] == {
        "kind": "heading", "text": "From data", "section": [], "level": 1
    }  # fmt: skip
    assert readings["https://example.test/plain"]["origin"] == "html"
    assert readings["https://example.test/plain"]["published"] == ""
    assert readings["https://example.test/news/dated"]["kind"] == "post"
    assert readings["https://example.test/news/dated"]["published"] == "2026-10-09"
    assert (config.PAGE_READINGS / "report-acme.json").exists()


def test_the_command_line_reads_a_companys_pages(
    stored: None, capsys: pytest.CaptureFixture[str]
) -> None:
    write_sources(company())

    assert cli.main(["read-pages", "acme"]) == 0
    out = capsys.readouterr().out
    assert "pages: 3" in out and "from nuxt data: 1" in out
    assert "thin: https://example.test/empty (1 words)" in out
    assert cli.main(["read-pages", "nobody"]) == 1
