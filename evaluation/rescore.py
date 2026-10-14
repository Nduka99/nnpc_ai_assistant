"""Offline rescoring: a run's recorded model replies checked again by today's
post-generation checks (`answer.checked`: verification, the grounding judge, the
earlier-holder check), so a change to a check is measured in minutes instead of a
two-hour run.

Nothing is generated again: the understanding, the passages found and the answering
replies are the ones each answer recorded (`model_calls`): the draft and, when one was
asked, its repair, applied only if the checks still leave a drafted part without
a claim. Only the grounding judge may be asked again. `recorded` answers a statement
with the verdict recorded for it, and asks its fallback only for a statement never
judged then (a claim verification now keeps); `supporting` is the arm without a judge.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from evaluation import runs
from petroknowledge import answer as answering
from petroknowledge import store
from petroknowledge.records import DraftClaim, Passage

Chat = answering.Chat
Verdicts = Mapping[str, object]


def recorded(verdicts: Verdicts, fallback: Chat) -> Chat:
    """A judge giving each statement the verdict recorded for it, else asking
    `fallback`."""

    def judge(system: str, user: str, schema: dict[str, Any]) -> object:
        statement = user.rsplit("Statement:\n", 1)[1]
        if statement in verdicts:
            return verdicts[statement]
        return fallback(system, user, schema)

    return judge


def supporting(system: str, user: str, schema: dict[str, Any]) -> object:
    """No judge: every statement is supported (the reference arm)."""
    return {"verdict": "supported"}


def recorded_calls(
    rows: Sequence[tuple[str, str, Any]],
) -> tuple[list[Any], dict[str, Any]]:
    """An answer's recorded answering replies in order (none if it had no passages:
    the draft, then any repair) and its grounding verdicts by statement, from its
    model calls (purpose, user, reply)."""
    replies = []
    verdicts: dict[str, Any] = {}
    for purpose, user, output in rows:
        if purpose == "answering":
            replies.append(output)
        elif purpose == "grounding":
            verdicts[user.rsplit("Statement:\n", 1)[1]] = output
    return replies, verdicts


def rescored(
    entry: Mapping[str, Any],
    question: str,
    parts: Sequence[str],
    given: Sequence[Passage],
    pointers: Sequence[Passage],
    rows: Sequence[tuple[str, str, Any]],
    judge_for: Callable[[Verdicts], Chat],
    stale: answering.StaleCheck | None = None,
) -> dict[str, Any]:
    """One recorded answer checked again, as a run entry; a referral stays as it was,
    since it had no answer request."""
    if entry["shown"]["status"] == "safety_referral":
        return dict(entry)
    replies, verdicts = recorded_calls(rows)
    sources = [f"S{number}" for number in range(1, len(given) + 1)]
    judge = judge_for(verdicts)
    drafts: tuple[DraftClaim, ...] = ()
    if replies:
        drafts = answering.read_output(replies[0], sources, len(parts))
    result = answering.checked(question, parts, given, pointers, drafts, judge, stale)
    again = answering.repair_parts(drafts, result.claims)
    if len(replies) > 1 and again:
        redrafts = [d for d in answering.read_output(replies[1], sources, len(parts))
                    if d.part in again]  # fmt: skip
        second = answering.checked(question, parts, given, pointers, redrafts, judge,
                                   stale)  # fmt: skip
        result = answering.joined(result, second)
    return runs.record({"id": entry["id"]}, result, entry["answer_id"])


def recorded_answer(
    conn: store.Connection, entry: Mapping[str, Any]
) -> tuple[str, list[str], list[Passage], list[Passage], list[tuple[str, str, Any]]]:
    """What an entry's answer recorded: its question, the parts it was split into,
    the passages it was given and pointed to, and its model calls."""
    row = conn.execute(
        "SELECT question, passage_ids, search_questions FROM answers WHERE id = %s",
        (entry["answer_id"],),
    ).fetchone()
    if row is None:
        raise ValueError(f"answer {entry['answer_id']} is not recorded")
    question, passage_ids, parts = row
    rows = conn.execute(
        "SELECT purpose, user_message, reply FROM model_calls WHERE answer_id = %s "
        "ORDER BY position",
        (entry["answer_id"],),
    ).fetchall()
    given = store.load_passages(conn, passage_ids)
    pointers = store.load_passages(conn, [p["id"] for p in entry["pointers"]])
    return question, list(parts), given, pointers, rows
