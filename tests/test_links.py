"""The link tier: posts and news items kept as metadata, tagged by company."""

import pytest

from petroknowledge import config, fetch, links
from tests.conftest import FEED, NEWS, POST_ONE, POST_TWO, SITE, company


def test_a_pages_own_title_description_and_date_are_read() -> None:
    assert links.page_metadata(POST_ONE) == {
        "title": "NNPC signs deal",
        "description": "A gas deal.",
        "published": "2026-10-01T09:00:00Z",
    }
    assert links.page_metadata(POST_TWO) == {
        "title": "Acme results",
        "description": "Half-year figures.",
        "published": "2026-09-30",
    }
    assert links.page_metadata("<p>nothing</p>") == {
        "title": "",
        "description": "",
        "published": "",
    }
    data = '[{"data": 1}, {"post_date": 2}, "2026-10-09"]'
    built = f'<title>Post</title><script id="__NUXT_DATA__">{data}</script>'
    assert links.page_metadata(built)["published"] == "2026-10-09"


def test_feed_items_become_plain_text_with_iso_dates() -> None:
    assert links.feed_items(FEED) == [
        {
            "url": "https://news.test/a",
            "title": "Acme Oil & NNPC sign deal",
            "description": "The gas deal.",
            "published": "2026-10-09T10:00:00+01:00",
        },
        {
            "url": "https://news.test/b",
            "title": "Football results",
            "description": "",
            "published": "",
        },
        {
            "url": "https://news.test/c",
            "title": "Beta Energy expands",
            "description": "",
            "published": "",
        },
    ]


def test_an_alias_must_appear_as_a_whole_word() -> None:
    assert links.mentions("nnpc signs a deal", ["NNPC"])
    assert not links.mentions("NNPCL signs a deal", ["NNPC"])
    assert links.mentions("Acme Oil, the producer", ["Beta", "Acme Oil"])


def test_merging_keeps_when_a_link_was_first_seen() -> None:
    old = [{"url": "a", "title": "old", "first_seen": "monday"}]
    new = [
        {"url": "a", "title": "new", "first_seen": "tuesday"},
        {"url": "b", "title": "b", "first_seen": "tuesday"},
    ]

    assert links.merged(old, new) == [
        {"url": "a", "title": "new", "first_seen": "monday"},
        {"url": "b", "title": "b", "first_seen": "tuesday"},
    ]


def test_posts_are_read_once_and_again_only_when_their_lastmod_changes(
    web: list[str],
) -> None:
    added, failures = links.collect_posts(company())

    assert added == 3
    assert failures == [
        {"url": SITE + "news/gone", "error": "HTTP 404"},
        {"url": SITE + "private/posts.xml", "error": "disallowed by robots.txt"},
    ]
    registry = links.read_registry("acme")
    assert [entry["url"] for entry in registry] == [
        SITE + "news/one",
        SITE + "news/two",
        SITE + "news/three",
    ]
    assert registry[0]["title"] == "NNPC signs deal"
    assert registry[0]["listed_lastmod"] == "2026-10-01"
    assert registry[0]["source"] == "company site"

    web.clear()
    assert links.collect_posts(company())[0] == 0
    assert "GET example.test/news/one" not in web  # unchanged: not read again

    registry[0]["listed_lastmod"] = "2026-09-01"
    registry[0]["first_seen"] = "earlier"
    registry[1]["reader"] = links.READER - 1  # read before the reader improved
    links.write_registry("acme", registry)
    assert links.collect_posts(company())[0] == 2
    updated = links.read_registry("acme")
    assert updated[0]["listed_lastmod"] == "2026-10-01"
    assert updated[0]["first_seen"] == "earlier"
    assert updated[1]["reader"] == links.READER


def test_links_written_during_a_posts_run_are_kept(
    web: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_get_text = fetch.get_text

    def news_arrives_meanwhile(*args: object) -> object:
        if str(args[2]).endswith("news/three"):
            links.write_registry("acme", [{"url": NEWS + "a", "first_seen": "now"}])
        return real_get_text(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(fetch, "get_text", news_arrives_meanwhile)

    links.collect_posts(company())

    urls = [entry["url"] for entry in links.read_registry("acme")]
    assert urls[0] == NEWS + "a" and len(urls) == 4


def test_news_items_are_kept_for_each_company_they_name(web: list[str]) -> None:
    beta = company("beta", name="Beta Energy", aliases=["Beta"])
    feeds = [
        {"name": "News", "url": NEWS + "feed/"},
        {"name": "Gone", "url": NEWS + "gone/"},
    ]

    counts, failures = links.collect_news([company(), beta], feeds)

    assert counts == {"acme": 1, "beta": 1}
    assert failures == [{"url": NEWS + "gone/", "error": "HTTP 404"}]
    acme = links.read_registry("acme")
    assert acme[0]["url"] == "https://news.test/a" and acme[0]["source"] == "News"
    assert links.read_registry("beta")[0]["url"] == "https://news.test/c"
    assert sorted(path.name for path in config.LINK_REGISTRY.iterdir()) == [
        "acme.json",
        "beta.json",
    ]
