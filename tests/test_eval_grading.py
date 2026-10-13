"""Grading items and verdict counts against release bar R1. Every key and answer
here is invented."""

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation import __main__ as cli
from evaluation import grading, runs
from petroknowledge.records import Answer, Claim, Evidence, Passage

LINE = Passage(1, "https://acme.test/gas", "Gas Line", "Route", "It runs 614 km.", "d",
               published="2026-09-29")  # fmt: skip
SCAN = Passage(2, "https://acme.test/r.pdf", "Report", "Notes", "x", "d", 7,
               quotable=False)  # fmt: skip
QUOTE = Evidence(1, LINE.url, LINE.title, LINE.heading, "runs 614 km", "link",
                 "2026-10-10T10:00:00+00:00", "2026-09-29")  # fmt: skip
ANSWER = Answer("q", "answered", "", (Claim("The line runs 614 km.", (QUOTE,)),),
                (LINE,), (), pointers=(SCAN,))  # fmt: skip
REFUSAL = Answer("q", "insufficient_evidence", "Not found.", (), (), ())


def keyed(item_id: str, status: str = "answered", **more: Any) -> dict[str, Any]:
    return {"id": item_id, "type": "fact", "expected_status": status,
            "expected_answer": "614 km.", "must_not": ["615 km"],
            "parts": [{"asks": "How long?", "expected_answer": "614 km.",
                       "evidence": []}], **more}  # fmt: skip


KEY = {
    "cases": [keyed("c1", wordings=["How long is the line?"]),
              keyed("c2", wordings=["Never asked"])],
    "conversations": [{"id": "v1", "turns": [
        keyed("v1-t1", message="Who runs the line?", standalone_question="Who?"),
        keyed("v1-t2", "insufficient_evidence", message="and its cost?",
              standalone_question="What does the line cost?"),
        keyed("v1-t3", message="never asked", standalone_question="?"),
    ]}],
}  # fmt: skip
ENTRIES = [
    runs.record({"id": "c1"}, ANSWER, 1),
    runs.record({"id": "v1-t1"}, ANSWER, 2),
    runs.record({"id": "v1-t2"}, REFUSAL, 3),
]


def test_items_show_the_key_and_what_the_reader_saw_with_the_turns_before() -> None:
    found = grading.items(KEY, ENTRIES)

    assert [item["id"] for item in found] == ["c1", "v1-t1", "v1-t2"]
    first = found[0]
    assert first["question"] == "How long is the line?"
    assert first["key"] == {"expected_status": "answered", "expected_answer": "614 km.",
                            "parts": [{"asks": "How long?",
                                       "expected_answer": "614 km."}],
                            "must_not": ["615 km"]}  # fmt: skip
    assert first["answer"] == {
        "status": "answered", "notice": "",
        "claims": [{"text": "The line runs 614 km.",
                    "quotes": [{"quote": "runs 614 km", "source": "Gas Line",
                                "published": "2026-09-29"}]}],
        "points_to": ["Report (scanned page)"],
    }  # fmt: skip
    turn = found[2]
    assert turn["standalone_question"] == "What does the line cost?"
    assert turn["conversation_so_far"] == [
        {"user": "Who runs the line?", "assistant": "The line runs 614 km."}
    ]
    assert "answer_id" not in json.dumps(found)  # nothing about how it was made


def test_verdicts_are_counted_against_release_bar_r1() -> None:
    types = {f"q{n}": ("fact" if n % 2 else "list") for n in range(100)}
    statuses = {f"q{n}": "answered" for n in range(100)} | {"q2": "insufficient"}
    sound = {f"q{n}": {"verdict": "sound"} for n in range(100)}
    one_wrong = sound | {"q1": {"verdict": "wrong"}, "q2": {"verdict": "missing"}}

    passed = grading.counts(types, statuses, sound)
    failed = grading.counts(types, statuses, one_wrong)

    assert passed["r1_pass"] and passed["wrong_rate"] == 0.0
    assert passed["by_type"] == {"fact": {"sound": 50}, "list": {"sound": 50}}
    assert failed["verdicts"] == {"sound": 98, "wrong": 1, "missing": 1}
    # The refusal (q2) counts in neither the wrong answers nor the answers given.
    assert (failed["answered"], failed["wrong_rate"]) == (99, 1 / 99)
    assert failed["wrong_interval"][1] > grading.R1 and not failed["r1_pass"]
    assert not grading.counts(types, statuses, {})["r1_pass"]  # nothing answered
    with pytest.raises(ValueError, match="unknown verdict 'fine'"):
        grading.counts(types, statuses, {"q1": {"verdict": "fine"}})


def test_the_commands_write_items_and_count_grades(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, registry = tmp_path / "eval", tmp_path / "sets.json"
    (root / "keys").mkdir(parents=True)
    (root / "keys" / "cases.json").write_text(json.dumps(KEY))
    common = ["--root", str(root), "--registry", str(registry)]
    assert cli.main([*common, "register", "keys"]) == 0
    run = tmp_path / "run.json"
    run.write_text(json.dumps(ENTRIES))
    items = tmp_path / "items.json"

    assert cli.main([*common, "grading-items", "keys", "--file", "cases.json",
                     "--run", str(run), "--out", str(items)]) == 0  # fmt: skip
    assert len(json.loads(items.read_text())) == 3
    grades = tmp_path / "grades.json"
    grades.write_text(json.dumps({"c1": {"reason": "r", "verdict": "sound"}}))
    counted = [*common, "grade-counts", "keys", "--file", "cases.json", "--grades",
               str(grades), "--run", str(run), "--out",
               str(tmp_path / "counts.json")]  # fmt: skip
    assert cli.main(counted) == 1  # one graded answer cannot meet R1
    assert "R1 (upper bound <= 0.05): FAIL" in capsys.readouterr().out
    many = {f"c{n}": {"verdict": "sound"} for n in range(80)}
    big = {"cases": [keyed(f"c{n}", wordings=["q"]) for n in range(80)]}
    big_run = tmp_path / "big-run.json"
    big_run.write_text(json.dumps([runs.record({"id": f"c{n}"}, ANSWER, n)
                                   for n in range(80)]))  # fmt: skip
    (root / "big").mkdir()
    (root / "big" / "cases.json").write_text(json.dumps(big))
    assert cli.main([*common, "register", "big"]) == 0
    grades.write_text(json.dumps(many))
    assert cli.main([*common, "grade-counts", "big", "--file", "cases.json",
                     "--grades", str(grades), "--run", str(big_run), "--out",
                     str(tmp_path / "counts.json")]) == 0  # fmt: skip
    assert "wrong among 80 answered: 0.000" in capsys.readouterr().out
    sealed = [*common, "grading-items", "nnpc-heldout-v1", "--file", "cases.json",
              "--run", str(run), "--out", str(items)]  # fmt: skip
    assert cli.main(sealed) == 1


def test_a_second_grader_gets_every_dispute_and_a_seeded_share_of_each_type(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    graded = [{"id": f"q{n}"} for n in range(20)]
    types = {f"q{n}": ("fact" if n < 10 else "list") for n in range(20)}
    grades = {f"q{n}": {"verdict": "sound"} for n in range(20)}
    grades |= {"q3": {"verdict": "wrong"}, "q15": {"verdict": "missing"}}

    found = grading.second_sample(graded, types, grades, 0.3, 7)

    ids = [item["id"] for item in found]
    assert {"q3", "q15"} <= set(ids)  # every dispute
    assert sum(types[i] == "fact" for i in ids) == 1 + 3  # q3 and 30% of the other 9
    assert ids == sorted(ids, key=lambda i: int(i[1:]))  # key order
    assert found == grading.second_sample(graded, types, grades, 0.3, 7)
    assert len(grading.second_sample(graded[:2], types, grades, 0.3, 7)) == 1  # >= 1

    root, registry = tmp_path / "eval", tmp_path / "sets.json"
    (root / "keys").mkdir(parents=True)
    (root / "keys" / "cases.json").write_text(json.dumps(KEY))
    common = ["--root", str(root), "--registry", str(registry)]
    assert cli.main([*common, "register", "keys"]) == 0
    items, primary = tmp_path / "items.json", tmp_path / "grades.json"
    items.write_text(json.dumps(grading.items(KEY, ENTRIES)))
    primary.write_text(json.dumps({"c1": {"verdict": "wrong"},
                                   "v1-t1": {"verdict": "sound"},
                                   "v1-t2": {"verdict": "sound"}}))  # fmt: skip
    out = tmp_path / "second.json"

    assert cli.main([*common, "second-sample", "keys", "--file", "cases.json",
                     "--items", str(items), "--grades", str(primary), "--seed", "1",
                     "--out", str(out)]) == 0  # fmt: skip
    assert "c1" in [item["id"] for item in json.loads(out.read_text())]
    assert "verdict" not in out.read_text()  # the primary's verdicts stay hidden


def test_agreement_is_cohen_s_kappa_on_the_items_both_graded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    primary = {"a": {"verdict": "sound"}, "b": {"verdict": "wrong"},
               "c": {"verdict": "missing"}, "d": {"verdict": "sound"}}  # fmt: skip
    second = {"a": {"verdict": "sound"}, "b": {"verdict": "wrong"},
              "c": {"verdict": "sound"}}  # fmt: skip

    found = grading.agreement(primary, second)

    assert (found["items"], found["agree"], found["differ"]) == (3, 2, ["c"])
    assert 0 < found["kappa"] < 1
    first, other = tmp_path / "p.json", tmp_path / "s.json"
    first.write_text(json.dumps(primary))
    other.write_text(json.dumps(second))
    assert cli.main(["grade-agreement", "--primary", str(first), "--second",
                     str(other), "--out", str(tmp_path / "a.json")]) == 0  # fmt: skip
    assert "2 of 3 agree" in capsys.readouterr().out
