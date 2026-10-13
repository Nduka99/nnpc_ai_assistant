"""K16's picture-text gate (Lime Green X43 B2, carried over for NNPC).

Pictures are drawn by seed from the stored pictures (data/images/places.json): the
posts' own uploads (K13) and PDF figures. The printed words of each chosen picture are
written from its image before any reading is looked at, and the saved readings are
scored against them as the index would keep them. The rules are in each set's rules.md.
"""

import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from evaluation import webpages
from petroknowledge import images

GATE = {"recall": 0.95, "precision": 0.98}  # X43 B2
NO_WORDS = "no words"  # a skipped candidate's reason: nothing printed on it


def interleaved(
    groups: Mapping[str, Sequence[dict[str, Any]]], seed: int
) -> list[dict[str, Any]]:
    """Each group shuffled by the seed, then one from each group in turn, so no
    group dominates the first draws (Lime Green's `imagesets.interleaved`). The
    groups' order is shuffled too: with one picture per group (a post's upload), name
    order would make the draw alphabetical (pictures-v1 amendment 1)."""
    rng = random.Random(seed)
    queues = []
    for name in sorted(groups):
        members = list(groups[name])
        rng.shuffle(members)
        queues.append(members)
    rng.shuffle(queues)
    found = []
    while any(queues):
        for queue in queues:
            if queue:
                found.append(queue.pop(0))
    return found


def candidates(
    places: Sequence[Mapping[str, Any]], seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Each stored picture once, by source in drawing order: "post" pictures grouped
    by their post, "figure" pictures (PDF figures) by their document."""
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {"post": {}, "figure": {}}
    seen = set()
    for place in places:
        if place["id"] in seen:
            continue
        seen.add(place["id"])
        source = "post" if place["page"] is None else "figure"
        picture = {"id": place["id"], "source": source, "url": place["source"],
                   "page": place["page"]}  # fmt: skip
        groups[source].setdefault(place["source"], []).append(picture)
    return {source: interleaved(found, seed) for source, found in groups.items()}


def kept_words(reading: str) -> list[str]:
    """A reading's words as the index keeps them (`images.ocr_lines`), folded as W2
    folds them."""
    return webpages.tokens(" ".join(images.ocr_lines(reading)))


def counts(truth: str, reading: str) -> dict[str, int]:
    """One picture's words in the truth, words read, and words read right (each word
    counted as often as it is printed)."""
    wanted = Counter(webpages.tokens(truth))
    read = Counter(kept_words(reading))
    return {
        "words": wanted.total(),
        "read": read.total(),
        "right": (wanted & read).total(),
    }


def score(truth: Mapping[str, Any], readings: Mapping[str, str]) -> dict[str, Any]:
    """Each chosen picture's counts, recall and precision over the set and per
    source, the words read from skipped pictures with no words, and the gate.
    `readings` holds each picture's saved reading by id."""
    pictures = {}
    totals: dict[str, Counter[str]] = {"all": Counter()}
    for picture in truth["pictures"]:
        found = counts(picture["text"], readings[picture["id"]])
        pictures[str(picture["number"])] = found
        for name in ("all", picture["source"]):
            totals.setdefault(name, Counter()).update(found)
    rates = {}
    for name, total in totals.items():
        rates[name] = {
            "recall": webpages.rate(total["right"], total["words"]),
            "precision": webpages.rate(total["right"], total["read"]),
        }
    invented = 0
    for skipped in truth["skipped"]:
        if skipped["reason"] == NO_WORDS:
            invented += len(kept_words(readings[skipped["id"]]))
    gate = {measure: rates["all"][measure] >= bar for measure, bar in GATE.items()}
    return {"pictures": pictures, "rates": rates, "invented": invented, "gate": gate}
