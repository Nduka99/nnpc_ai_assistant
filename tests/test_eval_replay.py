"""The answer path replayed up to the answer request and scored by evidence reach.
Every key, message and passage here is invented."""

import contextlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from evaluation import __main__ as cli
from evaluation import replay, runs
from petroknowledge.records import Answer, Claim, Evidence, Part, Passage

A = "a" * 64
FILES = {"https://acme.test/gas": A}
LINE = Passage(1, "https://acme.test/gas", "Gas", "Route", "The line runs 614 km.", "d")
OTHER = Passage(2, "https://acme.test/oil", "Oil", "Wells", "Nine wells.", "d")
QUOTE = Evidence(1, LINE.url, LINE.title, LINE.heading, "runs 614 km", "link", "d")


def keyed(item_id: str, status: str = "answered", **more: Any) -> dict[str, Any]:
    evidence = {"url": "u", "sha256": A, "page": None, "form": "text",
                "quote": "runs 614 km"}  # fmt: skip
    parts = [{"evidence": [evidence]}] if status == "answered" else []
    return {"id": item_id, "type": "fact", "expected_status": status,
            "parts": parts, **more}  # fmt: skip


KEY = {
    "cases": [keyed("c1", wordings=["How long is the line?"]),
              keyed("c2", "insufficient_evidence", wordings=["Its price?"])],
    "conversations": [{"id": "v1", "turns": [
        keyed("v1-t1", message="Who runs the line?"),
        keyed("v1-t2", message="and how long is it?"),
        keyed("v1-t3", message="and its length again?"),
    ]}],
}  # fmt: skip
ASKED = runs.items(KEY)


def shown(text: str) -> Answer:
    return Answer("q", "answered", "", (Claim(text, (QUOTE,)),), (LINE,), ())


RUN = [runs.record(item, shown(f"Reply {n}."), n) for n, item in enumerate(ASKED)]


def test_each_turn_reads_the_replies_an_earlier_run_showed_within_the_window() -> None:
    found = replay.histories(ASKED, RUN, 1)

    assert found == {
        "v1-t1": [],
        "v1-t2": [("Who runs the line?", "Reply 2.")],
        "v1-t3": [("and how long is it?", "Reply 3.")],  # the last 1 turn only
    }


def test_only_answerable_questions_are_replayed_and_advice_is_given_nothing() -> None:
    heard: list[tuple[str, Sequence[tuple[str, str]]]] = []

    def understand(message: str, history: Any) -> tuple[bool, list[Part]]:
        heard.append((message, history))
        return message.startswith("Who"), [Part(f"search: {message}", ("x", "y"))]

    found_for: list[tuple[str, Any]] = []

    def find(message: str, history: Any, understood: Sequence[Part]) -> Any:
        found_for.append((message, history))
        return (LINE,), (OTHER,)

    history = {"v1-t2": [("Who runs the line?", "Reply 2.")]}
    found = replay.replay(ASKED, understand, find, history)

    assert [entry["id"] for entry in found] == ["c1", "v1-t1", "v1-t2", "v1-t3"]
    assert heard[2] == ("and how long is it?", [("Who runs the line?", "Reply 2.")])
    assert found_for[1] == heard[2]  # the searches see the same words
    first = found[0]
    assert first["searched"] == [{"question": "search: How long is the line?",
                                  "items": ["x", "y"]}]  # fmt: skip
    assert [p["id"] for p in first["given"]] == [1]
    assert [p["id"] for p in first["pointers"]] == [2]
    advice = found[1]  # "Who …" is read as asking for advice here
    assert (advice["searched"], advice["given"]) == ([], [])

    def no_first_request(message: str, history: Any) -> Any:
        raise AssertionError("the kept search questions are used")

    kept = {entry["id"]: entry["searched"] for entry in found}
    again = replay.replay(ASKED, no_first_request, find, history, kept)
    assert [entry["searched"] for entry in again] == [e["searched"] for e in found]


def test_reach_counts_parts_whose_every_quote_was_given() -> None:
    entries = [
        {"id": "c1", "given": [runs.place(LINE)], "pointers": []},
        {"id": "v1-t1", "given": [runs.place(OTHER)], "pointers": []},
    ]

    found = replay.reach(ASKED, entries, FILES)

    assert (found["parts"], found["parts_reached"]) == (2, 1)
    assert found["items"][1] == {"id": "v1-t1", "parts": 1, "parts_reached": 0}


def test_the_command_replays_a_registered_set_and_can_keep_its_searches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from petroknowledge import answer, assistant, fetch

    root, registry = tmp_path / "eval", tmp_path / "sets.json"
    (root / "keys").mkdir(parents=True)
    (root / "keys" / "cases.json").write_text(json.dumps(KEY))
    common = ["--root", str(root), "--registry", str(registry)]
    assert cli.main([*common, "register", "keys"]) == 0
    run = tmp_path / "run.json"
    run.write_text(json.dumps(RUN))
    monkeypatch.setattr(assistant, "connect", lambda: contextlib.nullcontext())
    monkeypatch.setattr(assistant, "served_index", lambda conn: 1)
    monkeypatch.setattr(assistant, "answer_models", lambda: None)
    monkeypatch.setattr(assistant, "retriever", lambda c, v, m: lambda q: [LINE])
    monkeypatch.setattr(assistant, "extras", lambda c, v, m: None)
    told: list[str] = []

    def understand(message: str, chat: Any, history: Any, named: str = "") -> Any:
        told.append(named)
        return False, [Part(message)]

    monkeypatch.setattr(answer, "understand", understand)
    manifest = [{"url": LINE.url, "sha256": A}, {"url": "x"}]
    monkeypatch.setattr(fetch, "read_manifest", lambda: manifest)
    out = tmp_path / "replay.json"
    replayed = [*common, "replay", "keys", "--file", "cases.json", "--history",
                str(run), "--out", str(out)]  # fmt: skip

    assert cli.main(replayed) == 0
    assert "parts reached: 4 of 4 (4 questions)" in capsys.readouterr().out
    monkeypatch.setattr(answer, "understand", None)  # kept searches need no request
    assert cli.main([*replayed[:-1], str(tmp_path / "again.json"), "--keep",
                     str(out)]) == 0  # fmt: skip
    saved = json.loads((tmp_path / "again.json").read_text())
    assert saved["reach"]["parts_reached"] == 4
    named: list[str] = []

    def asked_units(conn: Any, version: int, message: str, history: Any) -> Any:
        named.append(message)
        return [{"id": 7, "name": "Acme Gas", "acronym": "AG", "url": "u"}]

    monkeypatch.setattr(assistant, "asked_units", asked_units)
    monkeypatch.setattr(assistant, "scoped_retriever",
                        lambda c, v, m, units: lambda q: [OTHER])  # fmt: skip
    scoped = tmp_path / "scoped.json"
    monkeypatch.setattr(answer, "understand", understand)
    assert cli.main([*replayed[:-1], str(scoped), "--keep", str(out),
                     "--scoped"]) == 0  # fmt: skip
    assert cli.main([*replayed[:-1], str(tmp_path / "fresh.json"),
                     "--scoped"]) == 0  # fmt: skip
    assert told[:4] == ["", "", "", ""] and told[-1] == "Acme Gas (AG)"
    monkeypatch.setattr(assistant, "role_lookup",
                        lambda c, v: lambda text: ("line", [OTHER]))  # fmt: skip
    dated = tmp_path / "dated.json"
    assert cli.main([*replayed[:-1], str(dated), "--keep", str(out), "--roles"]) == 0
    assert [p["id"] for p in json.loads(dated.read_text())["replayed"][0]["given"]] == [
        1, 2,
    ]  # fmt: skip
    first = json.loads(scoped.read_text())["replayed"][0]
    assert [p["id"] for p in first["given"]] == [1, 2]  # the unit's passage added
    assert named[0] == "How long is the line?"  # the user's own words
    no_history = [*common, "replay", "keys", "--file", "cases.json", "--out", str(out)]
    assert cli.main(no_history) == 1
    assert "needs --history RUN" in capsys.readouterr().err
    sealed = [*common, "replay", "nnpc-heldout-v1", "--file", "cases.json", "--out",
              str(out)]  # fmt: skip
    assert cli.main(sealed) == 1
