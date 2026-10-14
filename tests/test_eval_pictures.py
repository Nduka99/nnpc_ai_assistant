"""The picture-text gate (X43 B2): pictures drawn by seed per source, their saved
readings scored by word counts. Every picture, post and text here is invented."""

import json
from pathlib import Path
from typing import Any

import pytest

from evaluation import __main__ as cli
from evaluation import pictures

PLACES: list[dict[str, Any]] = [
    {"id": "p1", "source": "https://x/post-a", "page": None},
    {"id": "p2", "source": "https://x/post-b", "page": None},
    {"id": "f1", "source": "https://x/r.pdf?v=1", "page": 3},
    {"id": "f2", "source": "https://x/r.pdf?v=1", "page": 4},
    {"id": "f3", "source": "https://x/s.pdf", "page": 1},
    {"id": "f1", "source": "https://x/s.pdf", "page": 9},  # the same picture again
]


def test_candidates_are_each_picture_once_by_source_in_seeded_turns() -> None:
    found = pictures.candidates(PLACES, 5)

    assert found == pictures.candidates(PLACES, 5)
    assert sorted(p["id"] for p in found["post"]) == ["p1", "p2"]
    figures = [p["id"] for p in found["figure"]]
    assert sorted(figures) == ["f1", "f2", "f3"]
    assert "f3" in figures[:2]  # one from each document in turn
    assert {"id": "f3", "source": "figure", "url": "https://x/s.pdf",
            "page": 1} in found["figure"]  # fmt: skip


def test_groups_of_one_are_not_drawn_in_name_order() -> None:
    places = [{"id": name, "source": name, "page": None} for name in "abcdefgh"]

    drawn = [p["id"] for p in pictures.candidates(places, 5)["post"]]

    assert sorted(drawn) == list("abcdefgh") and drawn != list("abcdefgh")


def test_counts_fold_case_and_punctuation_and_count_repeats() -> None:
    reading = "<table><tr><td>NNPC Ltd.</td></tr></table>\n```\n---\nnnpc $^{\\circ}$C"

    assert pictures.counts("NNPC LTD, NNPC °C 2024", reading) == {
        "words": 5, "read": 4, "right": 4,
    }  # fmt: skip


TRUTH: dict[str, Any] = {
    "pictures": [
        {"number": 1, "id": "p1", "source": "post", "text": "Gas for growth"},
        {"number": 2, "id": "f1", "source": "figure", "text": "Crude 1.5 mbpd"},
    ],
    "skipped": [
        {"id": "f2", "reason": "no words"},
        {"id": "f3", "reason": "unreadable"},
    ],
}
READINGS = {"p1": "Gas for growth", "f1": "Crude 1.5", "f2": "Lorem", "f3": "x y"}


def test_the_score_pools_words_per_source_and_counts_invented_words() -> None:
    found = pictures.score(TRUTH, READINGS)

    assert found["pictures"]["2"] == {"words": 4, "read": 3, "right": 3}
    assert found["rates"]["all"] == {"recall": 6 / 7, "precision": 1.0}
    assert found["rates"]["post"] == {"recall": 1.0, "precision": 1.0}
    assert found["invented"] == 1
    assert found["gate"] == {"recall": False, "precision": True}


def test_the_command_line_draws_once_and_scores_a_registered_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, registry = tmp_path / "eval", tmp_path / "sets.json"
    stored = tmp_path / "images"
    stored.mkdir()
    (stored / "places.json").write_text(json.dumps(PLACES))
    for identity, text in READINGS.items():
        (stored / f"{identity}.png").write_bytes(identity.encode())
        (stored / f"{identity}.json").write_text(json.dumps({"ocr": text}))
    common = ["--root", str(root), "--registry", str(registry)]
    draw = [*common, "picture-candidates", "pics", "--seed", "5", "--count", "1",
            "--images", str(stored)]  # fmt: skip

    assert cli.main(draw) == 0
    folder = root / "pics"
    drawn = json.loads((folder / "candidates.json").read_text(encoding="utf-8"))
    first = drawn["figure"][0]
    assert first["png"] == "figure-01.png" and "png" not in drawn["figure"][1]
    assert (folder / "images" / "figure-01.png").read_bytes() == first["id"].encode()
    assert "candidates: 2 post, 3 figure" in capsys.readouterr().out
    assert cli.main(draw) == 1  # drawn once

    (folder / "truth.json").write_text(json.dumps(TRUTH))
    assert cli.main([*common, "register", "pics"]) == 0
    out = tmp_path / "score.json"
    score = [*common, "picture-score", "pics", "--out", str(out),
             "--images", str(stored)]  # fmt: skip

    assert cli.main(score) == 1
    printed = capsys.readouterr().out.splitlines()
    assert printed[-3:] == ["figure: recall 0.750, precision 1.000",
                            "words read from skipped pictures with no words: 1",
                            "FAIL"]  # fmt: skip
    assert json.loads(out.read_text(encoding="utf-8"))["invented"] == 1

    passing = {"pictures": TRUTH["pictures"][:1], "skipped": []}
    (root / "good").mkdir()
    (root / "good" / "truth.json").write_text(json.dumps(passing))
    assert cli.main([*common, "register", "good"]) == 0
    good = [*common, "picture-score", "good", "--out", str(out),
            "--images", str(stored)]  # fmt: skip
    assert cli.main(good) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "pass"
