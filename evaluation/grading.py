"""Graded answers (Lime Green's grades and blind items, carried over): the items a
grader reads, and the verdicts counted against release bar R1.

An item is one question, its key and the one answer it got, with nothing about how
the answer was made: the grader judges what the reader was shown, by
`briefs/grading-guide.md`. A conversation's turn comes with the turns before it, each
with the reply its user saw, and its question rewritten to stand alone. Verdicts are
counted per case type, and the wrong-answer rate's Wilson upper bound is compared with
release bar R1 (at most 5% wrong among answered, at 95% confidence).
"""

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast

from evaluation import metrics
from petroknowledge.view import AnswerView, reply_text

VERDICTS = ("sound", "partial", "missing", "wrong")
R1 = 0.05  # the wrong-answer rate's 95% upper bound must not exceed this


def key_view(keyed: Mapping[str, Any]) -> dict[str, Any]:
    """What a grader needs from a case or a turn of the key."""
    return {
        "expected_status": keyed["expected_status"],
        "expected_answer": keyed["expected_answer"],
        "parts": [{"asks": p["asks"], "expected_answer": p["expected_answer"]}
                  for p in keyed["parts"]],
        "must_not": keyed["must_not"],
    }  # fmt: skip


def answer_view(shown: Mapping[str, Any]) -> dict[str, Any]:
    """What the reader was shown: status, notice, claims with their quotes, and the
    pages pointed to."""
    sources = {source["number"]: source for source in shown["sources"]}
    claims = []
    for claim in shown["claims"]:
        quotes = [{"quote": sources[n]["quote"], "source": sources[n]["title"],
                   "published": sources[n]["published"]}
                  for n in claim["sources"]]  # fmt: skip
        claims.append({"text": claim["text"], "quotes": quotes})
    return {
        "status": shown["status"],
        "notice": shown["notice"],
        "claims": claims,
        "points_to": [f"{p['title']} ({p['kind']})" for p in shown["pointers"]],
    }


def items(
    key: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Every answered question of a run as a grading item, in key order."""
    by_id = {entry["id"]: entry for entry in entries}
    found: list[dict[str, Any]] = []
    for case in key.get("cases", []):
        if case["id"] in by_id:
            shown = by_id[case["id"]]["shown"]
            found.append({
                "id": case["id"], "question": case["wordings"][0],
                "key": key_view(case), "answer": answer_view(shown),
            })  # fmt: skip
    for conversation in key.get("conversations", []):
        earlier: list[dict[str, str]] = []
        for turn in conversation["turns"]:
            if turn["id"] not in by_id:
                break
            shown = by_id[turn["id"]]["shown"]
            found.append({
                "id": turn["id"], "conversation_so_far": list(earlier),
                "question": turn["message"],
                "standalone_question": turn["standalone_question"],
                "key": key_view(turn), "answer": answer_view(shown),
            })  # fmt: skip
            reply = reply_text(cast(AnswerView, shown))
            earlier.append({"user": turn["message"], "assistant": reply})
    return found


def second_sample(
    graded_items: Sequence[Mapping[str, Any]],
    types: Mapping[str, str],
    grades: Mapping[str, Mapping[str, Any]],
    share: float,
    seed: int,
) -> list[Mapping[str, Any]]:
    """The items a second grader grades (Lime Green R0): every item the primary
    graded wrong or missing, since R1 rests on them, and a seeded `share` of each
    case type's others, at least one per type; in key order."""
    chosen = {i["id"] for i in graded_items if grades[i["id"]]["verdict"] in
              ("wrong", "missing")}  # fmt: skip
    by_type: dict[str, list[str]] = {}
    for item in graded_items:
        if item["id"] not in chosen:
            by_type.setdefault(types[item["id"]], []).append(item["id"])
    rng = random.Random(seed)
    for name in sorted(by_type):
        ids = by_type[name]
        chosen.update(rng.sample(ids, max(1, round(len(ids) * share))))
    return [item for item in graded_items if item["id"] in chosen]


def agreement(
    primary: Mapping[str, Mapping[str, Any]], second: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Cohen's κ between two graders on the items both graded, and the items where
    they differ (Lime Green R0: below κ 0.8, read them and sharpen the guide)."""
    shared = [item_id for item_id in second if item_id in primary]
    first = [primary[item_id]["verdict"] for item_id in shared]
    other = [second[item_id]["verdict"] for item_id in shared]
    differ = [i for i, a, b in zip(shared, first, other, strict=True) if a != b]
    return {"items": len(shared), "agree": len(shared) - len(differ),
            "kappa": metrics.cohen_kappa(first, other), "differ": differ}  # fmt: skip


def counts(
    types: Mapping[str, str],
    statuses: Mapping[str, str],
    grades: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verdicts overall and per type (`types`: item id to case type), and the
    wrong-answer rate among the answers given (`statuses`: item id to the answer's
    status), as Lime Green's R0 defines risk: answered and graded wrong, over
    answered. A refusal is never counted as safe; its 95% Wilson interval's upper
    bound is compared with R1."""
    overall: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = {}
    answered = 0
    wrong = 0
    for item_id, grade in grades.items():
        verdict = grade["verdict"]
        if verdict not in VERDICTS:
            raise ValueError(f"{item_id}: unknown verdict {verdict!r}")
        overall[verdict] += 1
        by_type.setdefault(types[item_id], Counter())[verdict] += 1
        if statuses[item_id] == "answered":
            answered += 1
            wrong += verdict == "wrong"
    low, high = metrics.wilson(wrong, answered)
    return {
        "verdicts": dict(overall),
        "by_type": {name: dict(found) for name, found in sorted(by_type.items())},
        "graded": sum(overall.values()),
        "answered": answered,
        "wrong_rate": wrong / answered if answered else 0.0,
        "wrong_interval": [low, high],
        "r1_pass": high <= R1,  # with nothing answered the interval is the whole range
    }
