"""A keyed set asked through the answer path, and what can be measured without a
grader (Lime Green's ask and reach, carried over).

Every question of a set, or every turn of each conversation, is asked as a user
would ask it: a conversation's turns through one stored conversation, so each turn's
history is what its reader was shown. A run is saved after each question and resumes;
a conversation left half-asked is asked again from its first turn.

Two measures need no grader:
- **status**: whether each answer's outcome (answered, refused, referred) is the
  key's expected status;
- **reach**: whether the passages the model was given hold each part's evidence,
  found by the matcher claims are verified with (`verify.find_quote`). Evidence on a
  scanned page or in a picture is reached when the answer points to its page.
  Evidence is matched to sources by file (SHA-256), as the key names it, so a file
  collected from two addresses counts once.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from petroknowledge.records import Answer, Passage
from petroknowledge.units import fold
from petroknowledge.verify import find_quote
from petroknowledge.view import view

Item = dict[str, Any]
# Ask one message, in a conversation (its id) or alone (None): the answer, its record
# id and the conversation id it was asked in.
Ask = Callable[[str, str | None, bool], tuple[Answer, int, str | None]]
SEARCH_ONLY = {"scan", "picture"}  # forms reached by a pointer to their page


def items(key: Mapping[str, Any]) -> list[Item]:
    """Every question of a key as asked: a case's first wording, or each turn of a
    conversation in order, with its expected status and parts."""
    found = []
    for case in key.get("cases", []):
        found.append({"id": case["id"], "message": case["wordings"][0],
                      "conversation": None, "type": case["type"],
                      "expected_status": case["expected_status"],
                      "parts": case["parts"]})  # fmt: skip
    for conversation in key.get("conversations", []):
        for turn in conversation["turns"]:
            found.append({"id": turn["id"], "message": turn["message"],
                          "conversation": conversation["id"], "type": turn["type"],
                          "expected_status": turn["expected_status"],
                          "parts": turn["parts"]})  # fmt: skip
    return found


def place(passage: Passage) -> dict[str, Any]:
    return {"id": passage.id, "url": passage.url, "page": passage.page,
            "image": passage.image, "text": passage.text}  # fmt: skip


def record(item: Item, answer: Answer, answer_id: int) -> dict[str, Any]:
    """What a run keeps of one answer: the reader's view, the passages its model was
    given and the pages it pointed to."""
    return {
        "id": item["id"],
        "answer_id": answer_id,
        "shown": dict(view(answer)),
        "given": [place(p) for p in answer.passages],
        "pointers": [place(p) for p in answer.pointers],
    }


def run(
    asked: Sequence[Item],
    ask: Ask,
    saved: list[dict[str, Any]],
    save: Callable[[list[dict[str, Any]]], None],
) -> list[dict[str, Any]]:
    """Ask every item not yet in `saved`, saving after each; a conversation is asked
    whole, so one left half-asked is asked again from its first turn."""
    done = {entry["id"] for entry in saved}
    groups: dict[str, list[Item]] = {}
    for item in asked:
        groups.setdefault(item["conversation"] or item["id"], []).append(item)
    for group in groups.values():
        if all(item["id"] in done for item in group):
            continue
        ids = {item["id"] for item in group}
        saved = [entry for entry in saved if entry["id"] not in ids]
        conversation = None
        for item in group:
            answer, answer_id, conversation = ask(
                item["message"], conversation, item["conversation"] is not None
            )
            saved.append(record(item, answer, answer_id))
            save(saved)
    return saved


def reached(evidence: Mapping[str, Any], entry: Mapping[str, Any],
            files: Mapping[str, str]) -> bool:  # fmt: skip
    """Whether one evidence item reached the answer: its quote in a given passage of
    its file, also with punctuation and spacing folded (an amendment to the measure: a
    compiled list writes "(NGIC); NNPC Gas" where the page's text, as the key quotes
    it, has "(NGIC) NNPC Gas"), or, for a scanned page or a picture, its page pointed
    to. `files` gives each address's file (SHA-256)."""

    def same_file(found: Mapping[str, Any]) -> bool:
        return bool(files.get(found["url"]) == evidence["sha256"])

    folded = fold(evidence["quote"])
    for passage in entry["given"]:
        if not same_file(passage):
            continue
        if find_quote(evidence["quote"], passage["text"]):
            return True
        if folded and folded in fold(passage["text"]):
            return True
    if evidence["form"] not in SEARCH_ONLY:
        return False
    for pointer in entry["pointers"]:
        if same_file(pointer) and pointer["page"] == evidence["page"]:
            return True
    return False


def part_reach(
    item: Item, entry: Mapping[str, Any], files: Mapping[str, str]
) -> tuple[int, list[tuple[str, bool]]]:
    """How many of an item's parts reached the answer (a part counts when every one
    of its evidence quotes did), and each evidence item's form and outcome."""
    parts_reached = 0
    outcomes = []
    for part in item["parts"]:
        hits = [reached(e, entry, files) for e in part["evidence"]]
        for evidence, hit in zip(part["evidence"], hits, strict=True):
            outcomes.append((evidence["form"], hit))
        parts_reached += all(hits)
    return parts_reached, outcomes


def measure(
    asked: Sequence[Item], entries: Sequence[Mapping[str, Any]],
    files: Mapping[str, str],
) -> dict[str, Any]:  # fmt: skip
    """Status agreement per expected status and per type, and evidence reach per
    form, over every item answered in the run."""
    by_id = {entry["id"]: entry for entry in entries}
    status: dict[str, Counter[str]] = {}
    types: dict[str, Counter[str]] = {}
    forms: dict[str, Counter[str]] = {}
    items_found = []
    for item in asked:
        entry = by_id.get(item["id"])
        if entry is None:
            continue
        got = entry["shown"]["status"]
        status.setdefault(item["expected_status"], Counter())[got] += 1
        agreed = "agrees" if got == item["expected_status"] else "differs"
        types.setdefault(item["type"], Counter())[agreed] += 1
        parts_reached, outcomes = part_reach(item, entry, files)
        for form, hit in outcomes:
            forms.setdefault(form, Counter())["reached" if hit else "missed"] += 1
        items_found.append({"id": item["id"], "expected": item["expected_status"],
                            "got": got, "parts": len(item["parts"]),
                            "parts_reached": parts_reached})  # fmt: skip
    return {
        "status": {k: dict(v) for k, v in sorted(status.items())},
        "types": {k: dict(v) for k, v in sorted(types.items())},
        "forms": {k: dict(v) for k, v in sorted(forms.items())},
        "items": items_found,
    }
