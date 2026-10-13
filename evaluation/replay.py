"""Replay the answer path up to the answer request (Lime Green's E5 replay, carried
over): the first request and the searches, with no answer written.

A retrieval change is judged by whether each keyed part's evidence reaches the
answer request (`runs.part_reach`), in minutes where a run and its grading take
hours. The searches are the code answers run (`answer.understand`,
`answer.find_passages`). A conversation's turn is replayed with the history an
earlier run showed its reader, so every arm reads the same conversation. The search
questions can be kept from an earlier replay, so two retrieval arms differ in
nothing else.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from evaluation import runs
from petroknowledge.answer import History
from petroknowledge.records import Part, Passage
from petroknowledge.view import AnswerView, reply_text

# The first request: whether a message asks for advice, and its parts.
Understand = Callable[[str, History], tuple[bool, list[Part]]]
# The passages given and the pages pointed to, for a message, the conversation
# before it and the first request's parts.
Find = Callable[
    [str, History, Sequence[Part]], tuple[tuple[Passage, ...], tuple[Passage, ...]]
]


def histories(
    asked: Sequence[runs.Item], entries: Sequence[Mapping[str, Any]], window: int
) -> dict[str, list[tuple[str, str]]]:
    """Each conversation turn's history as an earlier run showed it: its last
    `window` earlier turns, each the user's message and the reply its reader saw."""
    shown = {entry["id"]: entry["shown"] for entry in entries}
    earlier: dict[str, list[tuple[str, str]]] = {}
    found: dict[str, list[tuple[str, str]]] = {}
    for item in asked:
        if item["conversation"] is None:
            continue
        before = earlier.setdefault(item["conversation"], [])
        found[item["id"]] = before[-window:]
        reply = reply_text(cast(AnswerView, shown[item["id"]]))
        before.append((item["message"], reply))
    return found


def replay(
    asked: Sequence[runs.Item],
    understand: Understand,
    find: Find,
    history: Mapping[str, Sequence[tuple[str, str]]],
    kept: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Every question the key expects answered, up to the answer request: its search
    questions (kept from an earlier replay when `kept` holds them, else from the
    first request), the passages given, the pages pointed to and the seconds the
    searches took. A message read as asking for advice is given nothing."""
    found = []
    for item in asked:
        if item["expected_status"] != "answered":
            continue
        if kept is not None and item["id"] in kept:
            understood = [Part(p["question"], tuple(p["items"]))
                          for p in kept[item["id"]]]  # fmt: skip
        else:
            advice, understood = understand(
                item["message"], history.get(item["id"], ())
            )
            if advice:
                understood = []
        started = time.perf_counter()
        given: tuple[Passage, ...] = ()
        pointers: tuple[Passage, ...] = ()
        if understood:
            earlier = history.get(item["id"], ())
            given, pointers = find(item["message"], earlier, understood)
        found.append({
            "id": item["id"],
            "searched": [{"question": p.question, "items": list(p.items)}
                         for p in understood],
            "given": [runs.place(p) for p in given],
            "pointers": [runs.place(p) for p in pointers],
            "seconds": round(time.perf_counter() - started, 2),
        })  # fmt: skip
    return found


def reach(
    asked: Sequence[runs.Item],
    entries: Sequence[Mapping[str, Any]],
    files: Mapping[str, str],
) -> dict[str, Any]:
    """Evidence parts reached per replayed question, and in all."""
    by_id = {item["id"]: item for item in asked}
    items = []
    for entry in entries:
        item = by_id[entry["id"]]
        parts_reached, _ = runs.part_reach(item, entry, files)
        items.append({"id": entry["id"], "parts": len(item["parts"]),
                      "parts_reached": parts_reached})  # fmt: skip
    return {
        "parts": sum(item["parts"] for item in items),
        "parts_reached": sum(item["parts_reached"] for item in items),
        "items": items,
    }
