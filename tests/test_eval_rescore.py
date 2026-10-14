"""Offline rescoring: a run's recorded replies checked again (K31). Every passage,
reply and verdict here is invented."""

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation import __main__ as cli
from evaluation import rescore
from petroknowledge import assistant, store
from petroknowledge.records import Passage
from tests.test_assistant import Chat, served  # noqa: F401 (the fixture)

LINE = Passage(1, "https://acme.test/gas", "Gas", "Route", "The line runs 614 km.",
               "2026-10-10T10:00:00+00:00")  # fmt: skip
EVIDENCE = [{"source_id": "S1", "quote": "runs 614 km"}]
REPLY = {"claims": [{"part": 1, "text": "The line runs 614 km.",
                     "evidence": EVIDENCE}]}  # fmt: skip
ROWS = [
    ("understanding", "Question: How long is the line?", {}),
    ("answering", "Reference passages: ...", REPLY),
    ("grounding", "Evidence:\nGas\n\nStatement:\nThe line runs 614 km.",
     {"verdict": "unsupported"}),
]  # fmt: skip
ENTRY = {"id": "c1", "answer_id": 7, "shown": {"status": "answered"}, "pointers": []}


def unsupporting(system: str, user: str, schema: dict[str, Any]) -> object:
    return {"verdict": "unsupported"}


def test_a_recorded_judge_reuses_each_verdict_and_asks_only_for_new_statements() -> (
    None
):
    asked: list[str] = []

    def fallback(system: str, user: str, schema: dict[str, Any]) -> object:
        asked.append(user)
        return {"verdict": "supported"}

    replies, verdicts = rescore.recorded_calls(ROWS)
    judge = rescore.recorded(verdicts, fallback)

    assert replies == [REPLY]
    assert judge("s", "Evidence:\nx\n\nStatement:\nThe line runs 614 km.", {}) == {
        "verdict": "unsupported"
    }
    assert judge("s", "Evidence:\nx\n\nStatement:\nNew.", {}) == {
        "verdict": "supported"
    }
    assert asked == ["Evidence:\nx\n\nStatement:\nNew."]
    assert rescore.supporting("s", "u", {}) == {"verdict": "supported"}
    assert rescore.recorded_calls([]) == ([], {})


def test_a_recorded_answer_is_checked_again_with_the_judge_chosen() -> None:
    kept = rescore.rescored(ENTRY, "How long is the line?", ["How long is the line?"],
                            [LINE], [], ROWS,
                            lambda verdicts: rescore.supporting)  # fmt: skip
    removed = rescore.rescored(ENTRY, "How long is the line?", ["How long?"], [LINE],
                               [], ROWS, lambda verdicts: unsupporting)  # fmt: skip
    stale = rescore.rescored(ENTRY, "How long?", ["How long?"], [LINE], [], ROWS,
                             lambda verdicts: rescore.supporting,
                             lambda text, whole: "names an earlier holder")  # fmt: skip
    referral = {**ENTRY, "shown": {"status": "safety_referral"}}

    assert kept["shown"]["status"] == "answered"
    assert [c["text"] for c in kept["shown"]["claims"]] == ["The line runs 614 km."]
    assert (kept["id"], kept["answer_id"]) == ("c1", 7)
    assert removed["shown"]["status"] == "insufficient_evidence"
    assert stale["shown"]["status"] == "insufficient_evidence"
    never = rescore.rescored(referral, "q", [], [], [], [], lambda v: unsupporting)
    assert never == referral
    nothing = rescore.rescored(ENTRY, "q", ["q"], [], [], ROWS[:1],
                               lambda verdicts: unsupporting)  # fmt: skip
    assert nothing["shown"]["status"] == "insufficient_evidence"
    miscopied = {
        "claims": [
            {
                **REPLY["claims"][0],
                "evidence": [{"source_id": "S1", "quote": "runs 615 km"}],
            }
        ]
    }
    drafted = [("understanding", "q", {}), ("answering", "a", miscopied)]
    repaired = rescore.rescored(ENTRY, "q", ["q"], [LINE], [],
                                [*drafted, ("answering", "a", REPLY)],
                                lambda verdicts: rescore.supporting)  # fmt: skip
    unrepaired = rescore.rescored(ENTRY, "q", ["q"], [LINE], [], drafted,
                                  lambda verdicts: rescore.supporting)  # fmt: skip
    assert repaired["shown"]["status"] == "answered"  # the recorded repair applied
    assert unrepaired["shown"]["status"] == "insufficient_evidence"


def test_the_rescore_command_reads_an_answer_s_record_and_writes_a_run(
    served: Chat,  # noqa: F811
    pg: store.Connection,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answer, answer_id = assistant.ask_and_record(
        "How long does the plant take to commission?"
    )
    run = tmp_path / "run.json"
    entry = {"id": "c1", "answer_id": answer_id,
             "shown": {"status": answer.status}, "pointers": []}  # fmt: skip
    run.write_text(json.dumps([entry]), encoding="utf-8")
    out = tmp_path / "rescored.json"
    common = ["rescore", "--run", str(run), "--out", str(out)]

    assert cli.main([*common, "--judge", "none"]) == 0
    (kept,) = json.loads(out.read_text(encoding="utf-8"))
    assert kept["shown"]["status"] == "answered"
    assert "1 answers rescored (1 answered)" in capsys.readouterr().out
    live = store.live_version(pg)
    assert live is not None
    assert cli.main([*common, "--stale-version", str(live[0])]) == 0
    assert cli.main([*common, "--judge", "live"]) == 0
    with pg.cursor() as cursor:
        cursor.execute("DELETE FROM model_calls")
        cursor.execute("DELETE FROM answers")
    pg.commit()
    assert cli.main(common) == 1
    assert "is not recorded" in capsys.readouterr().err
    sealed = tmp_path / "nnpc-wording-check-v1-run.json"
    sealed.write_text("[]", encoding="utf-8")
    assert cli.main(["rescore", "--run", str(sealed), "--out", str(out)]) == 1
    assert "sealed" in capsys.readouterr().err
