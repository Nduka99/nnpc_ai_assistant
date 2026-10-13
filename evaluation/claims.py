"""A benchmark of verified claims (K28): every claim the graded answers show, labelled
correct or wrong, so a claim check can be scored offline, without asking the model
again.

A claim in an answer graded sound or partial is correct: the grading guide gives
those verdicts only when nothing is false and no rule is broken. The claims of an
answer graded wrong are labelled one by one, by reading, and supplied as labels: a
claim is wrong when it states what stands in the asked thing's place, or something
false. A claim shown in several runs is kept once, with every answer it appeared in.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from evaluation import runs
from petroknowledge.verify import find_quote

Claim = dict[str, Any]
LABELLED = {"sound": "correct", "partial": "correct"}


def claim_key(text: str, quotes: Sequence[str]) -> str:
    """One claim with its quotes, as a short stable name."""
    payload = json.dumps([text, list(quotes)], ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def asked(key: Mapping[str, Any]) -> dict[str, tuple[str, list[str]]]:
    """Each keyed question's message and the user messages before it in its
    conversation, by item id."""
    found = {}
    earlier: dict[str, list[str]] = {}
    for item in runs.items(key):
        before = earlier.setdefault(item["conversation"] or item["id"], [])
        found[item["id"]] = (item["message"], list(before))
        before.append(item["message"])
    return found


def cited_passages(entry: Mapping[str, Any], numbers: Sequence[int]) -> list[int]:
    """The ids of the passages a claim's sources quote, among those the model was
    given: the passage at the source's address holding the quote."""
    sources = {source["number"]: source for source in entry["shown"]["sources"]}
    found = []
    for number in numbers:
        source = sources[number]
        for passage in entry["given"]:
            same_page = passage["url"] == source["url"]
            if same_page and find_quote(source["quote"], passage["text"]):
                found.append(passage["id"])
                break
    return found


def collect(
    runs_graded: Sequence[Mapping[str, Any]], labels: Mapping[str, str]
) -> list[Claim]:
    """Every claim of the graded answers, once. Each run is {"entries": the run,
    "verdicts": item id to verdict, "asked": `asked` of its key, "version": its index
    version}; `labels` gives each claim of a wrong answer, by `claim_key`, "correct" or
    "wrong". A wrong answer's claim without a label is an error."""
    found: dict[str, Claim] = {}
    for graded in runs_graded:
        for entry in graded["entries"]:
            verdict = graded["verdicts"].get(entry["id"])
            if verdict not in ("sound", "partial", "wrong"):
                continue
            quotes = {s["number"]: s for s in entry["shown"]["sources"]}
            for claim in entry["shown"]["claims"]:
                cited = [quotes[number] for number in claim["sources"]]
                key = claim_key(claim["text"], [s["quote"] for s in cited])
                label = LABELLED.get(verdict) or labels.get(key)
                if label is None:
                    raise ValueError(f"no label for a claim of {entry['id']}: {key}")
                message, history = graded["asked"][entry["id"]]
                item = found.setdefault(key, {
                    "key": key, "claim": claim["text"],
                    "quotes": [s["quote"] for s in cited],
                    "titles": [s["title"] for s in cited],
                    "label": label, "question": message, "history": history,
                    "version": graded["version"],
                    "passages": cited_passages(entry, claim["sources"]),
                    "answers": [],
                })  # fmt: skip
                # A claim labelled wrong anywhere is wrong: the same words stood in an
                # asked thing's place there.
                if label == "wrong":
                    item["label"] = "wrong"
                item["answers"].append(entry["id"])
    return list(found.values())
