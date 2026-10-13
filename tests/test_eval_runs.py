"""A keyed set asked through the answer path, and its status and evidence reach.
Every key, answer and passage here is invented."""

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation import __main__ as cli
from evaluation import runs
from petroknowledge.records import Answer, Passage

A, B = "a" * 64, "b" * 64
FILES = {"https://acme.test/gas": A, "https://cms.acme.test/r.pdf": B,
         "https://acme.test/r.pdf": B}  # fmt: skip
LINE = Passage(1, "https://acme.test/gas", "Gas", "Route", "The line runs 614 km.", "d")
SCAN = Passage(2, "https://cms.acme.test/r.pdf", "R", "Notes", "x", "d", 7,
               quotable=False)  # fmt: skip


def evidence(quote: str, form: str = "text", sha: str = A, page: int | None = None
             ) -> dict[str, Any]:  # fmt: skip
    return {"url": "u", "sha256": sha, "page": page, "form": form, "quote": quote}


KEY = {
    "cases": [
        {"id": "c1", "type": "fact", "wordings": ["How long is the line?", "x"],
         "expected_status": "answered",
         "parts": [{"evidence": [evidence("runs 614 km")]}]},
        {"id": "c2", "type": "absent", "wordings": ["What does gas cost?"],
         "expected_status": "insufficient_evidence", "parts": []},
    ],
    "conversations": [
        {"id": "v1", "turns": [
            {"id": "v1-t1", "message": "Who runs the line?", "type": "fact",
             "expected_status": "answered",
             "parts": [{"evidence": [evidence("Notes", "scan", B, 7)]}]},
            {"id": "v1-t2", "message": "and how long?", "type": "fact",
             "expected_status": "answered",
             "parts": [{"evidence": [evidence("runs 614 km"),
                                     evidence("not there")]}]},
        ]},
    ],
}  # fmt: skip


def answered(status: str = "answered") -> Answer:
    return Answer("q", status, "", (), (LINE,), (), pointers=(SCAN,))  # type: ignore[arg-type]


def test_items_are_each_case_s_first_wording_and_each_turn_in_order() -> None:
    found = runs.items(KEY)

    assert [(i["id"], i["message"], i["conversation"]) for i in found] == [
        ("c1", "How long is the line?", None),
        ("c2", "What does gas cost?", None),
        ("v1-t1", "Who runs the line?", "v1"),
        ("v1-t2", "and how long?", "v1"),
    ]


def test_a_run_resumes_and_asks_a_half_asked_conversation_again_whole() -> None:
    asked: list[tuple[str, str | None, bool]] = []
    saves: list[int] = []

    def ask(message: str, conversation: str | None, in_one: bool) -> Any:
        asked.append((message, conversation, in_one))
        return answered(), len(asked), "conv-9" if in_one else None

    saved = [{"id": "c1"}, {"id": "v1-t1"}]  # c1 done; the conversation half done
    found = runs.run(runs.items(KEY), ask, saved, lambda s: saves.append(len(s)))

    assert asked == [("What does gas cost?", None, False),
                     ("Who runs the line?", None, True),
                     ("and how long?", "conv-9", True)]  # fmt: skip
    assert [e["id"] for e in found] == ["c1", "c2", "v1-t1", "v1-t2"]
    assert saves == [3, 3, 4]  # saved after each answer
    entry = found[-1]
    assert entry["answer_id"] == 3 and entry["shown"]["status"] == "answered"
    assert entry["given"][0]["text"] == "The line runs 614 km."
    assert entry["pointers"][0] == {"id": 2, "url": SCAN.url, "page": 7, "image": "",
                                    "text": "x"}  # fmt: skip
    assert runs.run(runs.items(KEY), ask, found, lambda s: None) == found  # all done


def test_evidence_is_reached_by_its_quote_in_its_file_or_its_page_pointed_to() -> None:
    entry = runs.record({"id": "x"}, answered(), 1)
    other_file = {**FILES, "https://acme.test/gas": B}

    assert runs.reached(evidence("runs 614 km"), entry, FILES)
    assert not runs.reached(evidence("runs 614 km"), entry, other_file)
    assert not runs.reached(evidence("runs 615 km"), entry, FILES)
    assert runs.reached(evidence("anything", "scan", B, 7), entry, FILES)
    assert not runs.reached(evidence("anything", "scan", B, 8), entry, FILES)
    assert not runs.reached(evidence("anything", "table", B, 7), entry, FILES)


def test_status_and_reach_are_counted_over_what_was_answered() -> None:
    entries = [runs.record(item, answered(), n) for n, item in
               enumerate(runs.items(KEY)) if item["id"] != "c2"]  # fmt: skip

    found = runs.measure(runs.items(KEY), entries, FILES)

    assert found["status"] == {"answered": {"answered": 3}}
    assert found["types"] == {"fact": {"agrees": 3}}
    text = {"reached": 2, "missed": 1}
    assert found["forms"] == {"scan": {"reached": 1}, "text": text}
    assert [(i["id"], i["parts"], i["parts_reached"]) for i in found["items"]] == [
        ("c1", 1, 1), ("v1-t1", 1, 1), ("v1-t2", 1, 0),  # every quote of a part
    ]  # fmt: skip


def test_the_commands_ask_a_registered_set_and_measure_its_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from petroknowledge import assistant, fetch

    root, registry = tmp_path / "eval", tmp_path / "sets.json"
    (root / "keys").mkdir(parents=True)
    (root / "keys" / "cases.json").write_text(json.dumps(KEY))
    common = ["--root", str(root), "--registry", str(registry)]
    assert cli.main([*common, "register", "keys"]) == 0
    turns = iter(range(1, 10))

    class Turn:
        conversation_id = "conv-1"

    def ask_and_record(message: str, turn: Any = None) -> tuple[Answer, int]:
        return answered(), next(turns)

    manifest = [{"url": url, "sha256": sha} for url, sha in FILES.items()]
    monkeypatch.setattr(assistant, "open_turn", lambda conversation: Turn())
    monkeypatch.setattr(assistant, "ask_and_record", ask_and_record)
    monkeypatch.setattr(fetch, "read_manifest", lambda: [*manifest, {"url": "x"}])
    run = tmp_path / "run.json"

    assert cli.main([*common, "ask-keys", "keys", "--file", "cases.json",
                     "--out", str(run)]) == 0  # fmt: skip
    assert len(json.loads(run.read_text())) == 4
    assert cli.main([*common, "reach", "keys", "--file", "cases.json", "--run",
                     str(run), "--out", str(tmp_path / "reach.json")]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "4 answers in" in out and "evidence scan: {'reached': 1}" in out
    sealed = [*common, "ask-keys", "nnpc-heldout-v1", "--file", "cases.json", "--out",
              str(run)]  # fmt: skip
    assert cli.main(sealed) == 1
    assert "sealed until the release run" in capsys.readouterr().err
