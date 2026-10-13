"""The command line, end to end on the invented web."""

import json
import runpy
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from petroknowledge import cli, config, fetch, weights
from tests.conftest import PDF, SITE, company, write_sources


def test_core_sources_are_measured_then_collected(
    web: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    write_sources(company())

    assert cli.main(["acquire", "acme", "--measure"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("3 of 4 fine")
    assert "private/sitemap.xml: disallowed by robots.txt" in out
    saved = json.loads((config.MEASUREMENTS / "acme-core.json").read_text())
    assert [record["bytes"] for record in saved][-1] == len(PDF)
    assert fetch.read_manifest() == []  # measuring stores nothing

    assert cli.main(["acquire", "acme"]) == 0
    assert capsys.readouterr().out.startswith("3 of 4 fine")
    manifest = fetch.read_manifest()
    assert [record["kind"] for record in manifest] == ["page", "page", "document"]
    assert manifest[0]["listed_lastmod"] == "2026-10-01"
    assert manifest[2]["licence"] == "CC BY 4.0"


def test_a_companys_posts_are_collected_as_core_posts(
    web: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    write_sources(company(post_sitemaps=[SITE + "sitemap-posts.xml"]))

    assert cli.main(["acquire", "acme"]) == 0
    assert capsys.readouterr().out.startswith("6 of 8 fine")  # news/gone is 404
    posts = [r for r in fetch.read_manifest() if r["kind"] == "post"]
    assert [r["url"] for r in posts] == [
        SITE + "news/one",
        SITE + "news/two",
        SITE + "news/three",
    ]
    assert posts[0]["listed_lastmod"] == "2026-10-01"


def test_linked_files_are_measured_then_collected(
    web: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    write_sources(company())
    cli.main(["acquire", "acme"])
    capsys.readouterr()

    assert cli.main(["acquire", "acme", "--linked", "--measure", "--pictures"]) == 0
    assert capsys.readouterr().out.startswith("2 of 4 fine")
    assert (config.MEASUREMENTS / "acme-linked.json").exists()
    assert cli.main(["acquire", "acme", "--linked"]) == 0  # documents only
    assert capsys.readouterr().out.startswith("1 of 1 fine")
    assert cli.main(["acquire", "acme", "--linked", "--pictures"]) == 0
    assert capsys.readouterr().out.startswith("1 of 3 fine")
    kinds = [record["kind"] for record in fetch.read_manifest()]
    assert kinds == ["page", "page", "document", "document", "picture"]


def test_links_news_and_browse(
    web: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    write_sources(company())

    assert cli.main(["links", "acme"]) == 0
    assert capsys.readouterr().out.startswith("3 links added or updated")
    assert cli.main(["news"]) == 0
    assert capsys.readouterr().out.startswith("acme: 1 news links added or updated")
    cli.main(["acquire", "acme"])
    capsys.readouterr()
    assert cli.main(["browse"]) == 0
    assert capsys.readouterr().out.startswith("3 new readable names")


def test_an_unknown_company_is_an_error(
    web: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["acquire", "nobody"]) == 1
    assert capsys.readouterr().out.startswith("error: no source file")


@pytest.fixture
def model_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "m.gguf").write_bytes(b"weights")
    entry = {"file": "m.gguf", "bytes": 7, "sha256": weights.sha256(folder / "m.gguf")}
    manifest = tmp_path / "models.json"
    manifest.write_text(json.dumps([entry]))
    monkeypatch.setattr(weights, "MANIFEST", manifest)
    monkeypatch.setattr(weights, "FOLDER", folder)
    return folder


def test_models_are_checked_against_the_manifest(
    model_files: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["models"]) == 0
    assert capsys.readouterr().out == "1 model files checked, 0 problems\n"

    (model_files / "m.gguf").unlink()
    assert cli.main(["models", "--quick"]) == 1
    assert "missing: m.gguf" in capsys.readouterr().out


def test_python_dash_m_runs_the_command_line(
    model_files: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["petroknowledge", "models", "--quick"])

    with pytest.raises(SystemExit) as stopped:
        runpy.run_module("petroknowledge", run_name="__main__")

    assert stopped.value.code == 0


def test_ask_prints_the_answer_with_dated_sources_pointers_and_independence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from petroknowledge import assistant
    from petroknowledge.view import INDEPENDENT
    from tests.test_view import ANSWERED, REFUSED

    partial = replace(ANSWERED, notice="Only verified statements are shown.")
    answers = iter([(partial, 1), (REFUSED, 2)])
    monkeypatch.setattr(assistant, "ask_and_record", lambda question: next(answers))

    assert cli.main(["ask", "How long is the line?"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Answer:\n1. The line runs 614 km. [1]\n")
    assert "\nNote:\nOnly verified statements are shown.\n" in out
    assert "[2] Flare Cut › Flaring (published 2026-09-29, captured 2026-10-10)" in out
    assert "[1] Gas Line › Route (captured 2026-10-10)" in out
    assert "Also on these pages (not quoted):\n- Annual Report (scanned page): " in out
    assert out.rstrip().endswith(INDEPENDENT)
    assert cli.main(["ask", "What does gas cost?"]) == 0
    refused = capsys.readouterr().out
    assert "Closest pages:\n- Gas Line: https://acme.test/gas/line" in refused
    assert "Note:" not in refused  # a refusal is its notice alone


def test_ask_explains_an_unavailable_index(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from petroknowledge import assistant

    def unavailable(question: str) -> None:
        raise assistant.Unavailable("no live index")

    monkeypatch.setattr(assistant, "ask_and_record", unavailable)

    assert cli.main(["ask", "q"]) == 1
    assert "error: no live index" in capsys.readouterr().err


def test_serve_runs_the_api_on_this_machine_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    from petroknowledge import telemetry
    from petroknowledge.app import app

    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        uvicorn, "run", lambda served, **options: calls.append((served, options))
    )

    assert cli.main(["serve"]) == 0
    assert cli.main(["serve", "--port", "8123"]) == 0
    logging = {"log_config": telemetry.LOG_CONFIG, "access_log": False}
    assert calls == [
        (app, {"host": "127.0.0.1", "port": 8091, **logging}),
        (app, {"host": "127.0.0.1", "port": 8123, **logging}),
    ]


def test_history_lists_shows_and_exports_conversations_with_provenance(
    postgres_url: str,
    pg: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    from petroknowledge import store

    monkeypatch.setattr(config, "DATABASE_URL", postgres_url)
    opened = store.open_turn(pg, None, 30, 8, 20)
    assert opened is not None
    call = {"purpose": "answering", "system": "S", "user": "U",
            "reply": {"claims": []}, "seconds": 1.0}  # fmt: skip
    store.record_answer(
        pg, "Who leads NETCO?", "answered", {"claims": []}, [], [], 1, "e", "p", 2.0,
        conversation_id=opened[0], turn=1, calls=[call],
    )  # fmt: skip
    store.record_failure(pg, "And NGIC?", "ended", "conversation ended", opened[0])

    assert cli.main(["history"]) == 0
    listed = capsys.readouterr().out
    assert opened[0] in listed and "Who leads NETCO?" in listed
    assert "Questions that got no answer:" in listed and "And NGIC?" in listed

    assert cli.main(["history", opened[0]]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["answers"][0]["model_calls"] == [call]
    assert shown["failures"][0]["reason"] == "ended"

    exported = tmp_path / "history.jsonl"
    assert cli.main(["history", "--export", str(exported)]) == 0
    lines = exported.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["record"] for line in lines] == ["answer", "failure"]
    assert (
        "1 answers with their model calls and 1 unanswered" in capsys.readouterr().out
    )


def test_history_refuses_a_malformed_id_and_explains_a_missing_database(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["history", "not-a-conversation"]) == 1
    assert "is not a conversation id" in capsys.readouterr().err

    monkeypatch.setattr(config, "DATABASE_URL", "")
    assert cli.main(["history"]) == 1
    assert "PETROKNOWLEDGE_DATABASE_URL is not set" in capsys.readouterr().err
