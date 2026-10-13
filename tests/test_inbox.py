"""Files saved by hand are added only with a known public origin."""

import pytest

from petroknowledge import cli, config, fetch, inbox
from tests.conftest import PDF, SITE, company, write_sources


@pytest.fixture
def saved(web: list[str]) -> None:
    """The company's about page is stored; three files and a folder wait in its
    inbox, one of them renamed by the browser."""
    fetch.acquire(
        [{"url": SITE + "about", "company": "acme", "kind": "page"}],
        ["example.test"],
        10_000,
    )
    folder = config.INBOX / "acme"
    (folder / "sub").mkdir(parents=True)
    (folder / "report.pdf").write_bytes(PDF)
    (folder / "b.PDF").write_bytes(b"%PDF other")
    (folder / "renamed (1).pdf").write_bytes(b"%PDF renamed")
    (folder / "a.png").write_bytes(b"PNG a flyer")


def test_known_files_come_from_the_sources_and_the_stored_pages(
    saved: None,
) -> None:
    assert inbox.known_files(company()) == {
        "report.pdf": SITE + "docs/report.pdf",
        "b.PDF": "https://cdn.example.test/docs/b.PDF",
        "a.png": SITE + "img/a.png",
        "lazy.webp": SITE + "img/lazy.webp",
        "banner.jpg": SITE + "img/banner.jpg?v=1",
    }


def test_matching_files_are_stored_with_their_link_and_others_refused(
    saved: None,
) -> None:
    results = inbox.add_folder(company(), config.INBOX / "acme")

    assert [record["url"] for record in results] == [
        SITE + "img/a.png",
        "https://cdn.example.test/docs/b.PDF",
        "renamed (1).pdf",
        SITE + "docs/report.pdf",
    ]
    assert results[0]["kind"] == "picture"
    assert results[2]["error"] == "no known link ends in this name"
    added = fetch.read_manifest()[-1]
    assert added["added_by"] == "hand" and added["kind"] == "document"
    assert added["content_type"] == "application/pdf"
    assert fetch.store_path(added["sha256"]).read_bytes() == PDF
    assert inbox.add_folder(company(), config.INBOX / "acme") == [results[2]]


def test_a_missing_folder_is_reported(web: list[str]) -> None:
    assert inbox.add_folder(company(), config.INBOX / "none") == [
        {"url": str(config.INBOX / "none"), "error": "no such folder"}
    ]


def test_the_command_line_adds_the_companys_inbox(
    saved: None, capsys: pytest.CaptureFixture[str]
) -> None:
    write_sources(company())

    assert cli.main(["add", "acme"]) == 0
    out = capsys.readouterr().out
    assert "3 of 4 fine" in out
    assert "renamed (1).pdf: no known link ends in this name" in out


def test_listed_files_are_downloaded_once_into_the_inbox_and_logged(
    web: list[str],
) -> None:
    folder = config.INBOX / "acme"
    folder.mkdir(parents=True)
    (folder / "b.PDF").write_bytes(b"already saved")
    listed = config.INBOX / "acme-to-download.txt"
    listed.write_text(
        "## notes\n"
        f"{SITE}private/a.pdf\n"  # robots.txt disallows it: downloaded on instruction
        "https://cdn.example.test/docs/b.PDF\n"  # already in the inbox: skipped
        f"{SITE}docs/gone.pdf\n"
        f"{SITE}docs/huge.pdf\n"
        f"{SITE}docs/boom.pdf\n"
        "https://elsewhere.test/x.pdf\n",  # not an allowed host
        encoding="utf-8",
    )

    results = inbox.download_listed(
        inbox.listed_urls(listed), folder, ["example.test"], max_bytes=1000
    )

    assert [record.get("error") for record in results] == [
        "HTTP 404",  # private/a.pdf: the stand-in site has no such file
        "HTTP 404",
        "larger than 0 MB",
        "refused",
        "off the allowed hosts: https://elsewhere.test/x.pdf",
    ]
    assert "GET example.test/robots.txt" not in web  # named files only, no crawl


def test_downloaded_files_are_added_with_how_they_came(
    saved: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = config.INBOX / "acme"
    results = inbox.download_listed(
        [SITE + "docs/report.pdf"], folder, ["example.test"], 10_000
    )
    assert results == []  # report.pdf was already saved by hand
    (folder / "report.pdf").unlink()
    results = inbox.download_listed(
        [SITE + "docs/report.pdf"], folder, ["example.test"], 10_000
    )
    assert results == [{"url": SITE + "docs/report.pdf", "bytes": len(PDF)}]

    added = {
        r["url"]: r for r in inbox.add_folder(company(), folder) if "error" not in r
    }

    assert added[SITE + "docs/report.pdf"]["added_by"] == inbox.BY_DOWNLOAD
    assert added["https://cdn.example.test/docs/b.PDF"]["added_by"] == inbox.BY_HAND
    log = inbox.read_log(folder)
    assert (
        added[SITE + "docs/report.pdf"]["fetched_at"]
        == log["report.pdf"]["downloaded_at"]
    )


def test_the_command_line_downloads_the_companys_list(
    web: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    write_sources(company())
    config.INBOX.mkdir(parents=True)
    (config.INBOX / "acme-to-download.txt").write_text(f"{SITE}docs/report.pdf\n")

    assert cli.main(["download-listed", "acme"]) == 0
    assert capsys.readouterr().out.startswith("1 of 1 fine")
    assert (config.INBOX / "acme" / "report.pdf").read_bytes() == PDF
