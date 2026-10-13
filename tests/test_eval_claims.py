"""The claim benchmark (K28): claims of graded answers, labelled for scoring checks.
Every key, run and grade here is invented."""

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation import __main__ as cli
from evaluation import claims

KEY = {
    "cases": [{"id": "c1", "wordings": ["Who runs AGL?"], "type": "fact",
               "expected_status": "answered", "parts": []}],
    "conversations": [{"id": "v1", "turns": [
        {"id": "v1-t1", "message": "What does AGL do?", "type": "fact",
         "expected_status": "answered", "parts": []},
        {"id": "v1-t2", "message": "And its staff?", "type": "fact",
         "expected_status": "answered", "parts": []},
    ]}],
}  # fmt: skip


def entry(item_id: str, claim: str, quote: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "shown": {"claims": [{"text": claim, "sources": [1]}],
                  "sources": [{"number": 1, "quote": quote, "title": "AGL",
                               "url": "https://acme.test/agl"}]},
        "given": [{"id": 7, "url": "https://acme.test/other", "text": quote},
                  {"id": 8, "url": "https://acme.test/agl", "text": f"Intro. {quote}"}],
    }  # fmt: skip


def test_each_question_comes_with_the_user_messages_before_it() -> None:
    assert claims.asked(KEY) == {
        "c1": ("Who runs AGL?", []),
        "v1-t1": ("What does AGL do?", []),
        "v1-t2": ("And its staff?", ["What does AGL do?"]),
    }


def test_claims_are_labelled_by_their_answer_s_verdict_or_by_reading() -> None:
    sound = entry("c1", "Ada runs AGL.", "Ada runs AGL")
    wrong = entry("v1-t1", "AGL sells oil.", "BGL sells oil")
    again = entry("v1-t2", "Ada runs AGL.", "Ada runs AGL")
    unmatched = entry("v1-t2", "AGL has 9 staff.", "AGL has 9 staff")
    unmatched["given"] = []
    graded = [
        {"entries": [sound, wrong], "version": 3, "asked": claims.asked(KEY),
         "verdicts": {"c1": "sound", "v1-t1": "wrong"}},
        {"entries": [again, unmatched, entry("c1", "x", "y")], "version": 4,
         "asked": claims.asked(KEY), "verdicts": {"v1-t2": "partial", "c1": "missing"}},
    ]  # fmt: skip
    wrong_key = claims.claim_key("AGL sells oil.", ["BGL sells oil"])

    found = claims.collect(graded, {wrong_key: "wrong"})

    first, second, third = found
    assert (first["label"], first["answers"], first["passages"]) == (
        "correct", ["c1", "v1-t2"], [8])  # fmt: skip
    assert (first["question"], first["version"], first["titles"]) == (
        "Who runs AGL?", 3, ["AGL"])  # fmt: skip
    assert (second["key"], second["label"]) == (wrong_key, "wrong")
    assert (third["passages"], third["history"]) == ([], ["What does AGL do?"])
    with pytest.raises(ValueError, match="no label"):
        claims.collect(graded, {})


def test_a_claim_wrong_in_one_answer_is_wrong_wherever_it_appears() -> None:
    first = entry("c1", "AGL sells oil.", "BGL sells oil")
    later = entry("v1-t1", "AGL sells oil.", "BGL sells oil")
    graded = [{"entries": [first, later], "version": 3, "asked": claims.asked(KEY),
               "verdicts": {"c1": "sound", "v1-t1": "wrong"}}]  # fmt: skip
    key = claims.claim_key("AGL sells oil.", ["BGL sells oil"])

    (only,) = claims.collect(graded, {key: "wrong"})

    assert (only["label"], only["answers"]) == ("wrong", ["c1", "v1-t1"])


def test_the_command_writes_the_benchmark_from_a_spec_and_refuses_sealed_sets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, registry = tmp_path / "eval", tmp_path / "sets.json"
    (root / "keys").mkdir(parents=True)
    (root / "keys" / "cases.json").write_text(json.dumps(KEY))
    common = ["--root", str(root), "--registry", str(registry)]
    assert cli.main([*common, "register", "keys"]) == 0
    run, grades = tmp_path / "run.json", tmp_path / "grades.json"
    run.write_text(json.dumps([entry("c1", "AGL sells oil.", "BGL sells oil")]))
    grades.write_text(json.dumps({"c1": {"reason": "r", "verdict": "wrong"}}))
    spec, labels, out = (tmp_path / n for n in ("spec.json", "labels.json", "out.json"))
    source = {"set": "keys", "file": "cases.json", "run": str(run),
              "grades": [str(grades)], "version": 3}  # fmt: skip
    spec.write_text(json.dumps([source]))
    labels.write_text(
        json.dumps({claims.claim_key("AGL sells oil.", ["BGL sells oil"]): "wrong"})
    )
    asked = [*common, "claims-items", "--spec", str(spec), "--labels", str(labels),
             "--out", str(out)]  # fmt: skip

    assert cli.main(asked) == 0
    assert "1 claims (1 wrong)" in capsys.readouterr().out
    assert json.loads(out.read_text())["claims"][0]["label"] == "wrong"
    spec.write_text(json.dumps([source | {"set": "nnpc-heldout-v1"}]))
    assert cli.main(asked) == 1
